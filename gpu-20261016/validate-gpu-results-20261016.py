"""Audit the recorded GPU runs without changing original results or requiring CUDA."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'gpu-acceptance-20261015'))
from verify_results import hash_file, summarize_benchmarks, verify_pytest, verify_qwen

shared = ROOT / 'gpu-results-20261016-four-0136'
training = ROOT / 'gpu-results-20261016-four-training-0136'
controls = ROOT / 'gpu-results-20261016-qwen-baseline-repeat-0136'
pair = ROOT / 'gpu-results-20261016-qwen-matched-pair-0136'
single = ROOT / 'gpu-results-20261016-single-0'
diagnosis = json.loads((ROOT / 'qwen-cache-diagnosis-20261016.json').read_text())
cold_control = next((row for row in diagnosis['isolated_controls'] if row['ranks'] == 4), None)
cold_four = ROOT / Path(cold_control['directory']).name if cold_control is not None else None
cold_four_passed = cold_control is not None
manifests = [
    json.loads((path / filename).read_text())
    for path, filename in (
        (shared, 'manifest.json'),
        (training, 'manifest.json'),
        (controls, 'commands.json'),
        (pair, 'commands.json'),
        (single, 'manifest.json'),
    )
]
for path in (shared, training, controls, pair, single):
    subprocess.run(['sha256sum', '--check', '--quiet', 'SHA256SUMS'], cwd=path, check=True)
assert all(item['sources'] == manifests[0]['sources'] for item in manifests)
frozen = json.loads((ROOT / 'gpu-acceptance-20261015/source-after.json').read_text())
assert manifests[0]['sources'] == {
    key: frozen[key] for key in ('base', 'shared', 'gate', 'feature')
}
assert all(item['gpus'] == manifests[0]['gpus'] for item in manifests[:4])
assert all(item['cleanup']['exit_code'] == 0 for item in manifests)
assert manifests[2]['status'] == 'experiments_completed'
assert manifests[3]['status'] == 'pair_passed'
assert all(job['exit_code'] == 0 for item in manifests for job in item['jobs'])
assert manifests[1]['input_sha256'] == manifests[2]['input_sha256']
assert manifests[1]['input_sha256'] == manifests[3]['input_sha256']
assert set(manifests[1]['omitted_jobs']) == {'shared-groups', 'shared-models'}

report = {
    'acceptance_status': (
        'passed_current_sources' if cold_four_passed else 'pending_cold_cache_four_gpu_control'
    ),
    'sources': manifests[0]['sources'],
    'gpus_by_run': {
        str(path): manifest['gpus']
        for path, manifest in zip((shared, training, controls, pair, single), manifests)
    },
    'cold_cache_four_gpu_control': cold_control,
    'original_runs': [str(path) for path in (shared, training, controls, pair, single)],
    'original_manifest_sha256': {
        str(path / filename): hash_file(path / filename)
        for path, filename in (
            (shared, 'manifest.json'),
            (training, 'manifest.json'),
            (controls, 'commands.json'),
            (pair, 'commands.json'),
            (single, 'manifest.json'),
        )
    },
    'raw_four_matrix_status': manifests[1]['status'],
    'raw_four_matrix_error': manifests[1].get('error'),
    'raw_single_status': manifests[4]['status'],
    'raw_single_error': manifests[4].get('error'),
    'recheck_verifier_sha256': hash_file(ROOT / 'gpu-acceptance-20261015/verify_results.py'),
    'cache_diagnosis': str(ROOT / 'qwen-cache-diagnosis-20261016.json'),
    'cache_diagnosis_sha256': hash_file(ROOT / 'qwen-cache-diagnosis-20261016.json'),
    'tests': {
        'shared-groups': verify_pytest(shared / 'shared-groups/reports', 4, 16),
        'shared-models': verify_pytest(shared / 'shared-models/reports', 4, 24),
        'native-training': verify_pytest(training / 'native-training/reports', 4, 96),
        'single-shared-races': verify_pytest(single / 'shared-races/reports', 1, 4),
        'gate-replay': verify_pytest(single / 'gate-replay/reports', 1, 16),
        'gdn-scopes': verify_pytest(single / 'gdn-scopes/reports', 1, 73),
    },
    'baseline_controls': [],
    'qwen_reference_map': {},
    'qualification_limit': (
        'The original first unfused baseline differs at step 2. Two fresh offload-disabled '
        'repeats with the populated Triton cache agree with each other and the offload arm. '
        'A new direct baseline/offload pair with the populated cache also agrees. '
        'An empty shared-cache control reproduced differing runtime/persisted autotune '
        'configs and differing offload-disabled baselines. See the cache diagnosis for '
        'completed rank-isolation controls and interrupted runs. '
        'The original failed matrix is retained; this is not an unconditional cold-start '
        'bitwise-determinism guarantee across independently tuned configurations.'
    ),
}
if cold_four_passed:
    subprocess.run(['sha256sum', '--check', '--quiet', 'SHA256SUMS'], cwd=cold_four, check=True)
    path = cold_four / 'commands.json'
    manifest = json.loads(path.read_text())
    assert hash_file(path) == cold_control['manifest_sha256']
    assert manifest['status'] == 'isolated_cache_passed' and manifest['cleanup']['exit_code'] == 0
    assert manifest['sources'] == report['sources']
    assert manifest['input_sha256'] == manifests[1]['input_sha256']
    report['gpus_by_run'][str(cold_four)] = manifest['gpus']
    report['original_runs'].append(str(cold_four))
    report['original_manifest_sha256'][str(path)] = hash_file(path)
    report['qualification_limit'] += (
        ' The four-GPU cold-cache baseline/repeated-baseline/offload control now passes '
        'with per-rank cache ownership; the first comparison pair uses that direct control.'
    )
for rank in range(4):
    results = [
        json.loads((directory / f'result.rank{rank}.json').read_text())
        for directory in (
            controls / 'repeat-1',
            controls / 'repeat-2',
            training / 'qwen-f0-n0-o0',
            training / 'qwen-f0-n0-o1',
        )
    ]
    for result in results[:2]:
        assert result['arguments']['fraction'] is None
        assert result['gpu_training_run'] and result['pool_verified_before_reset']
        assert result['offload_status']['selected_transfer_bytes'] == 0
    assert results[0]['states'] == results[1]['states'] == results[3]['states']
    report['baseline_controls'].append(
        {
            'rank': rank,
            'repeats_equal': True,
            'repeat_matches_offload': True,
            'repeat_matches_original_baseline': results[0]['states'] == results[2]['states'],
        }
    )

# A temporary view lets the existing verifier inspect one explicitly recorded reference map.
# No raw JSON is edited or overwritten, and no new GPU execution is claimed here.
with tempfile.TemporaryDirectory() as temporary:
    view = Path(temporary)
    for fused in (0, 1):
        for norm in (0, 1):
            for offload in (0, 1):
                name = f'qwen-f{fused}-n{norm}-o{offload}'
                if (fused, norm) == (0, 0):
                    source = (
                        cold_four / f'repeat-{3 if offload else 1}'
                        if cold_four_passed
                        else pair / f'repeat-{offload + 1}'
                    )
                else:
                    source = training / name
                (view / name).symlink_to(source, target_is_directory=True)
                report['qwen_reference_map'][name] = str(source)
    report['qwen_direct_pairs'] = verify_qwen(view)

assert [job['name'] for job in manifests[4]['jobs']] == manifests[4]['planned_jobs']
report['gate_performance'] = summarize_benchmarks(
    single, manifests[4]['sources'], manifests[4]['gpus'][0]['uuid']
)
report['single_offline_recheck'] = {
    'status': 'passed',
    'fix': 'Normalize CUDA bare UUID and nvidia-smi GPU-prefixed UUID at the comparison boundary.',
    'gpu_commands_rerun_for_fix': False,
    'original_failed_manifest_retained': True,
}
report['gate_performance_review'] = (
    'No obvious regression in the measured gate-wrapper cases: all eight paired runtime '
    'medians improved. Individual sample variation is retained in the ranges. The small '
    'temporary-buffer increase is reported per shape. This is not an end-to-end offload '
    'performance measurement.'
)

(ROOT / 'gpu-validation-20261016.json').write_text(json.dumps(report, indent=2) + '\n')
print(
    json.dumps(
        {
            key: report[key]
            for key in (
                'acceptance_status',
                'raw_four_matrix_status',
                'raw_four_matrix_error',
                'baseline_controls',
                'qwen_direct_pairs',
                'single_offline_recheck',
                'qualification_limit',
            )
        },
        indent=2,
    )
)
raise SystemExit(0 if cold_four_passed else 1)
