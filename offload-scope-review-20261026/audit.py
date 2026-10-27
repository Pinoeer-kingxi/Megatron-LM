"""Verify retained results and integrity; failed precision qualifications stay failed."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent

def read(path):
    return json.loads((root / path).read_text())

def manifest(directory, revision):
    item = read(f'{directory}/manifest.json')
    assert item['revision'] == item['revision_after'] == revision
    assert item['exit_code'] == 0 and not item['dirty_after']

def states(directory, revision, offloaded=False):
    manifest(directory, revision)
    item = read(f'{directory}/result.rank0.json')
    assert item['gpu_training_run'] and item['pool_verified_before_reset']
    assert item['text_layers'] == 24 and len(item['states']) == 3
    assert bool(item['offload_status']['selected_transfer_bytes']) == offloaded
    for step in item['states']:
        assert len(step['grads']) == len(step['weights']) == 230
        assert step['optimizer'] and step['grad_norm'] > 0
    return item['states']

for directory, key, passed, skipped in (
    ('shared-scope-gpu0', 'shared', 6, 0),
    ('feature-scope-gpu0', 'feature', 104, 84),
):
    manifest(directory, read('sources.json')[key])
    report = read(f'{directory}/reports/rank-0.json')
    assert report['exit_code'] == 0
    outcomes = [row['outcome'] for row in report['outcomes']]
    assert outcomes.count('passed') == passed and outcomes.count('skipped') == skipped
    assert set(outcomes) <= {'passed', 'skipped'}

revision = read('sources.json')['feature']
outputs = [states(f'qwen-scope-gpu0-bf16-{arm}', revision, arm == 'offloaded')
           for arm in ('disabled', 'repeated-disabled', 'offloaded')]
assert outputs[0] == outputs[1] == outputs[2]
assert read('qwen-scope-gpu0-bf16-comparison.json')['exact_states']

precision_revision = read('sources.json')['precision']
qualifications = {}
for recipe in ('fp8-delayed', 'fp8-tensorwise', 'fp4'):
    arms = ['disabled', 'repeated-disabled', 'offloaded']
    if recipe == 'fp8-delayed':
        arms.append('confirmed-disabled')
    outputs = [states(f'precision/qwen-sm120-gpu0-{recipe}-{arm}', precision_revision,
                      arm == 'offloaded') for arm in arms]
    pairs = {f'{a} vs {b}': outputs[i] == outputs[j]
             for i, a in enumerate(arms) for j, b in enumerate(arms) if i < j}
    comparison = read(f'precision/qwen-sm120-gpu0-{recipe}-comparison.json')
    assert comparison['pairs'] == pairs and comparison['exact_states'] == all(pairs.values())
    qualifications[recipe] = all(pairs.values())
assert qualifications == {'fp8-delayed': False, 'fp8-tensorwise': True, 'fp4': False}
outputs = [states(f'precision/qwen-sm120-gpu0-cold-control-{arm}', precision_revision)
           for arm in ('cold-disabled', 'warm-disabled', 'warm-repeated-disabled')]
comparison = read('precision/qwen-sm120-gpu0-cold-control-comparison.json')
assert comparison['cold_vs_warm_exact'] == (outputs[0] == outputs[1])
assert comparison['warm_repeated_exact'] == (outputs[1] == outputs[2])
for line in (root / 'SHA256SUMS').read_text().splitlines():
    expected, name = line.split('  ', 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print(json.dumps(dict(integrity='passed', shared_gpu_tests='6 passed',
                      feature_gpu_suite='104 passed, 84 skipped', bf16_qwen_exact=True,
                      precision_qualifications=qualifications), indent=2))
