"""Check recorded GPU outcomes, full training fingerprints, and file integrity."""
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent
jobs = {
    'mrope-graph-865d79d-native-gpu1': ('865d79dd37f304a4dffa23158e8df52735d43247', 1, 1),
    'nvfp4-ddp-66b6bb6-unit-gpu12': ('66b6bb62a00bdccf8365096e6828215cffed4ce7', 8, 2),
    'graph-with-mrope-c5264b6-graph-native-gpu12': ('c5264b66197040f4309edfdb22dc9bbaf62189b1', 12, 2),
    'graph-with-mrope-c5264b6-graph-replay-gpu12': ('c5264b66197040f4309edfdb22dc9bbaf62189b1', 14, 2),
    'precision-with-mrope-dfaa633-precision-replay-gpu12': ('dfaa6338c65edab118ba415fcc14c73241f503a6', 18, 2),
    'precision-with-mrope-dfaa633-scope-gpu3': ('dfaa6338c65edab118ba415fcc14c73241f503a6', 111, 1),
    'precision-with-mrope-dfaa633-native-gpu3': ('dfaa6338c65edab118ba415fcc14c73241f503a6', 54, 1),
    'precision-with-mrope-dfaa633-graph-gpu3': ('dfaa6338c65edab118ba415fcc14c73241f503a6', 23, 1),
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

for arm in ('baseline', 'fixed'):
    folder = root / f'alignment-tp2-{arm}-verbose-20261027-gpu03'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert (manifest['exit_code'] == 0) == (arm == 'fixed')
    assert hashlib.sha256((root / 'probes/test_fp4_ordinary_optimizer.py').read_bytes()).hexdigest() == manifest['test_sha256']
    if arm == 'fixed':
        assert hashlib.sha256((root / 'nvfp4-ddp-alignment-prototype.patch').read_bytes()).hexdigest() == manifest['patch_sha256']
    for rank in (0, 1):
        report = json.loads((folder / 'reports' / f'rank-{rank}.json').read_text())
        assert report['collected'] == 1 and len(report['outcomes']) == 1
        outcome = report['outcomes'][0]
        assert outcome['outcome'] == ('passed' if arm == 'fixed' else 'failed')
        if arm == 'baseline':
            assert 'CUBLAS_STATUS_NOT_SUPPORTED' in outcome['detail']

revision = 'dfaa6338c65edab118ba415fcc14c73241f503a6'
states = []
keys = ('input_sha256', 'loss', 'grad_norm', 'gradients', 'weights', 'optimizer')
old = []
for arm in ('eager', 'graph'):
    folder = root / f'qwen-bf16-{arm}-20261027-gpu1'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['revision'] == manifest['revision_after'] == 'aecc0e8116912554f27a62a51ad2aeca479a7f9e'
    assert manifest['exit_code'] == 0 and not manifest['dirty_after']
    assert hashlib.sha256((root / 'benchmark_nvfp4_controls.py').read_bytes()).hexdigest() == manifest['script_sha256']
    data = json.loads((folder / 'result.json').read_text())
    assert len(data['records']) == 6 and not data['arguments']['offload']
    old.append(data['records'])
for i, (eager, graph) in enumerate(zip(*old)):
    assert eager['input_sha256'] == graph['input_sha256']
    if i < 3:
        assert all(eager[key] == graph[key] for key in keys)
    else:
        assert all(eager[key] != graph[key] for key in ('loss', 'gradients', 'weights', 'optimizer'))

for arm in ('eager', 'graph-disabled', 'graph-repeat', 'graph-offloaded'):
    folder = root / f'qwen-fp4-mrope-dfaa633-{arm}-gpu0'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['revision'] == manifest['revision_after'] == revision
    assert manifest['exit_code'] == 0 and not manifest['dirty_after']
    assert not manifest['disable_rht'] and not manifest['disable_stochastic_rounding']
    data = json.loads((folder / 'result.json').read_text())
    assert len(data['records']) == 6 and len(data['graph_counts']) == (0 if arm == 'eager' else 24)
    assert all((x['selected_bytes'] > 0) == (arm == 'graph-offloaded') for x in data['records'][3:])
    assert data['pool_checks'] == 6 and all(x['pool']['current_in_use'] == 0 for x in data['records'])
    states.append([{k: step[k] for k in keys} for step in data['records']])
assert states[0] == states[1] == states[2] == states[3]

for precision in ('fp8-delayed', 'fp8-tensorwise'):
    states = []
    for arm in ('eager', 'graph-off', 'graph-on'):
        folder = root / f'qwen-{precision}-{arm}-dfaa633-gpu0'
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['revision'] == manifest['revision_after'] == revision
        assert manifest['exit_code'] == 0 and not manifest['dirty_after']
        assert hashlib.sha256((root / 'benchmark_fp8_qwen.py').read_bytes()).hexdigest() == manifest['script_sha256']
        data = json.loads((folder / 'result.json').read_text())
        assert data['arguments']['precision'] == precision
        assert len(data['records']) == 6 and len(data['graph_counts']) == (0 if arm == 'eager' else 24)
        assert all((x['selected_bytes'] > 0) == (arm == 'graph-on') for x in data['records'][3:])
        assert data['pool_checks'] == 6 and all(x['pool']['current_in_use'] == 0 for x in data['records'])
        states.append([{k: step[k] for k in keys} for step in data['records']])
    assert states[0] == states[1] == states[2]

for model, layers in (('qwen', 24), ('gdn2', 8)):
    states = []
    for arm in ('eager', 'off', 'on'):
        folder = root / f'graph-{model}-f0-correct-{arm}'
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['revision'] == manifest['revision_after'] == 'c5264b66197040f4309edfdb22dc9bbaf62189b1'
        assert manifest['exit_code'] == 0 and not manifest['dirty_after'] and not manifest['foreign_samples']
        data = json.loads((folder / 'result.json').read_text())
        assert len(data['records']) == 6 and len(data['graph_counts']) == (0 if arm == 'eager' else layers)
        assert all((x['selected_bytes'] > 0) == (arm == 'on') for x in data['records'][3:])
        assert data['pool_checks'] == 6 and all(x['pool']['current_in_use'] == 0 for x in data['records'])
        states.append([{k: step[k] for k in keys} for step in data['records']])
    assert states[0] == states[1] == states[2]

states = []
for arm in ('eager', 'graph-off', 'graph-on'):
    folder = root / f'qwen-bf16-graph-tp2-{arm}-c5264b6-gpu12'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['revision'] == manifest['revision_after'] == 'c5264b66197040f4309edfdb22dc9bbaf62189b1'
    assert manifest['exit_code'] == 0 and not manifest['dirty_after']
    assert hashlib.sha256((root / 'benchmark_qwen_graph_tp2.py').read_bytes()).hexdigest() == manifest['script_sha256']
    ranks = []
    for rank in (0, 1):
        data = json.loads((folder / f'result.rank{rank}.json').read_text())
        assert len(data['records']) == 6 and len(data['graph_counts']) == (0 if arm == 'eager' else 24)
        assert all((x['selected_bytes'] > 0) == (arm == 'graph-on') for x in data['records'][3:])
        assert data['pool_checks'] == 6 and all(x['pool']['current_in_use'] == 0 for x in data['records'])
        ranks.append([{k: step[k] for k in keys} for step in data['records']])
    states.append(ranks)
assert states[0] == states[1] == states[2]

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

folder = root / 'qwen-bf16-stability-203-b6fa63-gpu3'
manifest = json.loads((folder / 'manifest.json').read_text())
assert manifest['revision'] == manifest['revision_after'] == 'b6fa63ab8c2cb855f0368cd8455245c295cde234'
assert manifest['exit_code'] == 0 and not manifest['dirty_after']
assert hashlib.sha256((root / 'qwen_stability_observer.py').read_bytes()).hexdigest() == manifest['script_sha256']
data = json.loads((folder / 'result.rank0.json').read_text())
pool = json.loads((folder / 'pool.rank0.json').read_text())
assert data['text_layers'] == 24 and len(data['samples']) == 200 and len(pool) == 203
assert all(x['selected_transfer_bytes'] > 0 for x in data['samples'])
assert {x['peak_allocated_bytes'] for x in data['samples']} == {14614417920}
assert {x['peak_reserved_bytes'] for x in data['samples']} == {15195963392}
assert all(x['stats']['current_in_use'] == 0 for x in pool)
assert {x['stats']['total_allocated'] for x in pool} == {126}
assert {x['retained_pinned_bytes'] for x in pool} == {33472512}
logs = list((folder / 'ranks').glob('**/0/stderr.log'))
assert len(logs) == 1
rows = re.findall(r'Rank 0 step (\d+): losses=(\[.*?\]) grad_norm=(.*?) selected_transfer_bytes=(\d+)', logs[0].read_text())
assert [int(row[0]) for row in rows] == list(range(203))
assert all(all(math.isfinite(value) for value in ast.literal_eval(row[1])) for row in rows)
norms = [re.fullmatch(r"tensor\(\[([-+0-9.eE]+)\], device='cuda:[0-9]+'\)", row[2]) for row in rows]
assert all(match and math.isfinite(float(match.group(1))) and float(match.group(1)) > 0 for match in norms)

states = []
for arm, first, last in (('continuous', 0, 6), ('save', 0, 3), ('resume', 3, 6)):
    folder = root / f'qwen-bf16-checkpoint-{arm}-b6fa63-gpu3'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['revision'] == manifest['revision_after'] == 'b6fa63ab8c2cb855f0368cd8455245c295cde234'
    assert manifest['exit_code'] == 0 and not manifest['dirty_after']
    assert hashlib.sha256((root / 'qwen_gdn_offload_training_b6fa63.py').read_bytes()).hexdigest() == manifest['script_sha256']
    data = json.loads((folder / 'result.rank0.json').read_text())
    assert data['text_layers'] == 24 and data['first_step'] == first and data['next_step'] == last
    assert len(data['states']) == last - first and data['pool_verified_before_reset']
    assert data['offload_status']['selected_transfer_bytes'] > 0
    states.append(data['states'])
assert states[0] == states[1] + states[2]

summary = json.loads(subprocess.check_output([sys.executable, '-S', str(root / 'summarize_performance.py')], text=True))
assert summary == json.loads((root / 'performance-summary.json').read_text())

for line in (root / 'SHA256SUMS').read_text().splitlines():
    digest, relative = line.split('  ', 1)
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
print('Checksums, GPU regressions, training fingerprints, and performance summaries verified.')
