"""Check recorded GPU outcomes, full training fingerprints, and file integrity."""
import hashlib
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent
jobs = {
    'mrope-graph-865d79d-native-gpu1': ('865d79dd37f304a4dffa23158e8df52735d43247', 1, 1),
    'nvfp4-ddp-66b6bb6-unit-gpu12': ('66b6bb62a00bdccf8365096e6828215cffed4ce7', 8, 2),
    'graph-with-mrope-c5264b6-graph-native-gpu12': ('c5264b66197040f4309edfdb22dc9bbaf62189b1', 12, 2),
    'graph-with-mrope-c5264b6-graph-replay-gpu12': ('c5264b66197040f4309edfdb22dc9bbaf62189b1', 14, 2),
    'precision-with-mrope-dfaa633-precision-replay-gpu12': ('dfaa6338c65edab118ba415fcc14c73241f503a6', 18, 2),
}
for name, (revision, count, ranks) in jobs.items():
    folder = root / name
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['revision'] == manifest['revision_after'] == revision
    assert manifest['exit_code'] == 0 and not manifest['dirty_after']
    for rank in range(ranks):
        report = json.loads((folder / 'reports' / f'rank-{rank}.json').read_text())
        assert report['exit_code'] == 0 and report['collected'] == count
        assert Counter(x['outcome'] for x in report['outcomes']) == {'passed': count}

for name, count, cause in (
    ('mrope-native-baseline-v3-20261027-gpu1', 1, 'Mismatched elements'),
    ('alignment-real-te-baseline-final-gpu1', 2, 'CUBLAS_STATUS_NOT_SUPPORTED'),
):
    folder = root / name
    manifest = json.loads((folder / 'manifest.json').read_text())
    report = json.loads((folder / 'reports/rank-0.json').read_text())
    assert manifest['exit_code'] != 0 and report['collected'] == count
    assert len(report['outcomes']) == count
    assert all(x['outcome'] == 'failed' and cause in x['detail'] for x in report['outcomes'])
    test = Path(manifest['command'][-1].split('::')[0]).name
    assert hashlib.sha256((root / 'probes' / test).read_bytes()).hexdigest() == manifest['test_sha256']

revision = 'dfaa6338c65edab118ba415fcc14c73241f503a6'
states = []
keys = ('input_sha256', 'loss', 'grad_norm', 'gradients', 'weights', 'optimizer')
for arm in ('eager', 'graph-disabled', 'graph-repeat', 'graph-offloaded'):
    folder = root / f'qwen-fp4-mrope-dfaa633-{arm}-gpu0'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['revision'] == manifest['revision_after'] == revision
    assert manifest['exit_code'] == 0 and not manifest['dirty_after']
    assert not manifest['disable_rht'] and not manifest['disable_stochastic_rounding']
    data = json.loads((folder / 'result.json').read_text())
    assert len(data['records']) == 6 and len(data['graph_counts']) == (0 if arm == 'eager' else 24)
    assert all(x['selected_bytes'] > 0 for x in data['records'][3:]) == (arm == 'graph-offloaded')
    states.append([{k: step[k] for k in keys} for step in data['records']])
assert states[0] == states[1] == states[2] == states[3]

normal = root / 'qwen-fp4-normal-tp2-default-dfaa633-gpu12'
assert json.loads((normal / 'manifest.json').read_text())['exit_code'] != 0
assert any('CUBLAS_STATUS_NOT_SUPPORTED' in p.read_text() for p in (normal / 'ranks').glob('**/stderr.log'))
states = []
for arm in ('high-precision', 'high-precision-offload'):
    folder = root / f'qwen-fp4-normal-tp2-{arm}-dfaa633-gpu12'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['revision'] == manifest['revision_after'] == revision
    assert manifest['exit_code'] == 0 and not manifest['dirty_after']
    assert manifest['backward_override'] == 'high_precision'
    outputs = [json.loads((folder / f'result.rank{rank}.json').read_text()) for rank in (0, 1)]
    for output in outputs:
        assert output['text_layers'] == 24 and len(output['states']) == 3
        assert output['pool_verified_before_reset']
        assert bool(output['offload_status']['selected_transfer_bytes']) == (arm == 'high-precision-offload')
    states.append([output['states'] for output in outputs])
assert states[0] == states[1]

for line in (root / 'SHA256SUMS').read_text().splitlines():
    digest, relative = line.split('  ', 1)
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
print('Checksums, real GPU regression controls, and Qwen FP4 training fingerprints verified.')
