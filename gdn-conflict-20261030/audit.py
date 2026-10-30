"""Verify recorded Qwen states and the independent mRoPE GPU result."""
import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).parent
for line in (root / 'SHA256SUMS').read_text().splitlines():
    expected, name = line.split('  ', 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
report = json.loads((root / 'mrope/reports/rank-0.json').read_text())
assert report['exit_code'] == 0 and len(report['outcomes']) == 1
assert report['outcomes'][0]['outcome'] == 'passed'
gd = json.loads((root / 'gdn2-isolated/reports/rank-0.json').read_text())
assert gd['exit_code'] == 0 and len(gd['outcomes']) == 1
assert gd['outcomes'][0]['outcome'] == 'passed'
fp = json.loads((root / 'precision-fp8/reports/rank-0.json').read_text())
assert fp['exit_code'] == 0 and len(fp['outcomes']) == 2
assert all(x['outcome'] == 'passed' for x in fp['outcomes'])
for experiment in ('qwen', 'fp8-qwen'):
    results = []
    for arm in ('eager', 'graph-off', 'graph-on'):
        with gzip.open(root / f'{experiment}/{arm}/result.json.gz', 'rt') as f:
            results.append(json.load(f))
    assert [len(x['graph_counts']) for x in results] == [0, 24, 24], experiment
    assert all(len(x['records']) == 6 for x in results), experiment
    for records in zip(*(x['records'] for x in results)):
        assert all(records[0][key] == records[1][key] == records[2][key] for key in
                   ('input_sha256', 'loss', 'grad_norm', 'gradients', 'weights', 'optimizer')), experiment
        assert records[2]['pool']['current_in_use'] == 0, experiment
    assert all(r['selected_bytes'] > 0 for r in results[2]['records'][3:]), experiment
print('Checksums, mRoPE, GDN2, FP8 GPU regressions and BF16/FP8 Qwen comparisons passed.')
