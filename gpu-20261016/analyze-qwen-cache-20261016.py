"""Recheck the cache-race reproduction and any completed cache-isolation controls."""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'gpu-acceptance-20261015'))
from verify_results import hash_file


def checked_run(directory):
    subprocess.run(['sha256sum', '--check', '--quiet', 'SHA256SUMS'], cwd=directory, check=True)
    manifest = json.loads((directory / 'commands.json').read_text())
    assert manifest['cleanup']['exit_code'] == 0
    assert all(job['exit_code'] == 0 for job in manifest['jobs'])
    return manifest


def states(directory, arm, rank):
    result = json.loads((directory / f'repeat-{arm}/result.rank{rank}.json').read_text())
    assert result['gpu_training_run'] and result['pool_verified_before_reset']
    assert result['text_layers'] == 24
    assert result['arguments']['fraction'] == (1.0 if arm == 3 else None)
    assert (result['offload_status']['selected_transfer_bytes'] > 0) == (arm == 3)
    return result['states']


def config_string(config):
    fields = list(config['kwargs'].items()) + [
        (name, config[name]) for name in ('num_warps', 'num_ctas', 'num_stages', 'maxnreg')
    ]
    return ', '.join(f'{name}: {value}' for name, value in fields)


shared = ROOT / 'gpu-results-20261016-qwen-cold-shared-0136'
shared_manifest = checked_run(shared)
assert len(shared_manifest['jobs']) == 3
cache = json.loads((shared / 'repeat-1/autotune-cache.json').read_text())
winners = {}
for filename, entry in cache.items():
    config, _timing = min(entry['configs_timings'], key=lambda row: row[1])
    key = (Path(filename).name.removesuffix('.autotune.json'), repr(tuple(entry['key'])))
    winners[key] = config_string(config)

lost_choices = []
pattern = (
    r'Triton autotuning for function (.*?),\nwith key as (.*?),\n'
    r'finished after .*?\nbest config selected: (.*?);'
)
for path in sorted((shared / 'repeat-1/ranks').rglob('stdout.log')):
    for kernel, key, config in re.findall(pattern, path.read_text()):
        winner = winners.get((kernel, repr(ast.literal_eval(key))))
        if winner is not None and config != winner:
            lost_choices.append(
                {
                    'rank': int(path.parent.name),
                    'kernel': kernel,
                    'key': ast.literal_eval(key),
                    'runtime_config': config,
                    'persisted_config': winner,
                }
            )
assert lost_choices
report = {
    'sources': shared_manifest['sources'],
    'shared_run': str(shared),
    'shared_run_status': shared_manifest['status'],
    'shared_run_error': shared_manifest.get('error'),
    'shared_cache_choices_not_preserved': lost_choices,
    'offload_disabled_repeats': [],
    'isolated_controls': [],
    'interrupted_controls': [],
    'limits': (
        'The shared-cache run is diagnostic: another user occupied GPU 0 during the last '
        'arm, so it is not an acceptance pass. The first two offload-disabled arms '
        'reproduce the cache mismatch. Separate empty caches may legitimately select '
        'different floating-point kernel configurations; the acceptance comparison '
        'must preserve each rank\'s configurations between baseline and offload.'
    ),
}
for rank in range(4):
    arms = [states(shared, arm, rank) for arm in (1, 2, 3)]
    assert arms[0] != arms[1] and arms[1] == arms[2]
    report['offload_disabled_repeats'].append(
        {
            'rank': rank,
            'states_equal': False,
            'first_grad_norms': [arms[0][0]['grad_norm'], arms[1][0]['grad_norm']],
            'repeat_matches_offload': True,
        }
    )
for suffix in ('13', '16', '0136', '1236'):
    directory = ROOT / f'gpu-results-20261016-qwen-cold-isolated-{suffix}'
    if not (directory / 'SHA256SUMS').is_file():
        continue
    manifest = checked_run(directory)
    assert manifest['sources'] == report['sources']
    assert manifest['input_sha256'] == shared_manifest['input_sha256']
    if manifest['status'] != 'isolated_cache_passed':
        report['interrupted_controls'].append(
            {
                'directory': str(directory),
                'status': manifest['status'],
                'error': manifest.get('error'),
                'completed_arms': len(manifest['jobs']),
                'completed_baselines_equal': all(
                    states(directory, 1, rank) == states(directory, 2, rank)
                    for rank in range(len(manifest['gpus']))
                ),
            }
        )
        continue
    assert len(manifest['jobs']) == 3
    assert manifest['status'] == 'isolated_cache_passed'
    configs = [
        json.loads((directory / f'repeat-{arm}/autotune-cache.json').read_text())
        for arm in (1, 2, 3)
    ]
    assert configs[0] and configs[0] == configs[1] == configs[2]
    rank_winners = {
        (
            int(Path(name).parts[0].removeprefix('rank-')),
            Path(name).name.removesuffix('.autotune.json'),
            repr(tuple(entry['key'])),
        ): config_string(entry['config'])
        for name, entry in configs[0].items()
    }
    checked_choices = 0
    for path in sorted((directory / 'repeat-1/ranks').rglob('stdout.log')):
        for kernel, key, config in re.findall(pattern, path.read_text()):
            winner = rank_winners.get((int(path.parent.name), kernel, repr(ast.literal_eval(key))))
            if winner is not None:
                assert config == winner, f'{path}: runtime config lost for {kernel}'
                checked_choices += 1
    assert checked_choices >= len(manifest['gpus'])
    for rank in range(len(manifest['gpus'])):
        assert any(name.startswith(f'rank-{rank}/') for name in configs[0])
        arms = [states(directory, arm, rank) for arm in (1, 2, 3)]
        assert arms[0] == arms[1] == arms[2]
    report['isolated_controls'].append(
        {
            'directory': str(directory),
            'ranks': len(manifest['gpus']),
            'gpus': manifest['gpus'],
            'started_with_empty_cache': not manifest['initial_autotune_cache_sha256'],
            'three_arms_equal': True,
            'cache_unchanged_after_first_arm': True,
            'runtime_configs_match_persisted_choices': checked_choices,
            'manifest_sha256': hash_file(directory / 'commands.json'),
        }
    )
(ROOT / 'qwen-cache-diagnosis-20261016.json').write_text(json.dumps(report, indent=2) + '\n')
print(
    json.dumps(
        {
            'lost_runtime_choices': len(lost_choices),
            'isolated_controls': report['isolated_controls'],
        },
        indent=2,
    )
)
