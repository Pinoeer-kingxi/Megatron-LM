"""Recompute paired end-to-end performance results from the retained run records."""
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
performance = root / 'performance'
records = json.loads((performance / 'progress.json').read_text())
by_name = {row['name']: row for row in records}
assert len(by_name) == len(records)


def norm_value(value):
    match = re.fullmatch(r"tensor\(\[([-+0-9.eE]+)\], device='cuda:[0-9]+'\)", value)
    return float(match.group(1) if match else value)


def paired(kind, model, fused, pairs):
    samples = []
    for pair in pairs:
        prefix = f'{kind}-{model}-f{int(fused)}-pair{pair}'
        off, on = [by_name[f'{prefix}-{arm}'] for arm in ('off', 'on')]
        for row in (off, on):
            manifest = json.loads((performance / row['name'] / 'manifest.json').read_text())
            expected = 'b6fa63ab8c2cb855f0368cd8455245c295cde234' if kind == 'eager' else 'c5264b66197040f4309edfdb22dc9bbaf62189b1'
            assert manifest['revision'] == manifest['revision_after'] == expected
            assert manifest['exit_code'] == 0 and not manifest['dirty_after'] and not manifest['foreign_samples']
            script = 'qwen_resources_20261027.py' if kind == 'eager' else 'benchmark_graph_with_pool.py'
            assert hashlib.sha256((root / script).read_bytes()).hexdigest() == manifest['benchmark_sha256']
            logs = list((performance / row['name'] / 'ranks').glob('**/0/stdout.log'))
            assert len(logs) == 1
            assert f"Imported Megatron Core from {manifest['source']}/megatron/core/__init__.py" in logs[0].read_text()
            telemetry = [json.loads(line) for line in (performance / row['name'] / 'telemetry.jsonl').read_text().splitlines()]
            assert telemetry and not any(sample['foreign'] for sample in telemetry)
            if kind == 'eager' and not fused:
                tp2 = json.loads((root / 'qwen-bf16-graph-tp2-graph-on-c5264b6-gpu12' / 'manifest.json').read_text())
                assert manifest['started_unix'] >= tp2['finished_unix']
            assert len(row['logs']) in (13, 16)
        if fused:
            for a, b in zip(off['logs'], on['logs']):
                assert a['step'] == b['step']
                assert all(math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-8) for x, y in zip(a['loss'], b['loss']))
                assert math.isclose(norm_value(a['norm']), norm_value(b['norm']), rel_tol=1e-5, abs_tol=1e-4)
        else:
            assert off['logs'] == on['logs']
        resources = [r['resource'] if kind == 'eager' else r['pinned_pool'] for r in (off, on)]
        assert all(r['stats']['current_in_use'] == 0 for r in resources)
        samples.append({
            'pair': pair,
            'step_ms_off': off['median_ms'], 'step_ms_on': on['median_ms'],
            'step_change_percent': (on['median_ms'] / off['median_ms'] - 1) * 100,
            'allocated_mib_off': off['allocated'] / 2**20, 'allocated_mib_on': on['allocated'] / 2**20,
            'reserved_mib_off': off['reserved'] / 2**20, 'reserved_mib_on': on['reserved'] / 2**20,
            'pinned_mib_off': resources[0]['retained_pinned_bytes'] / 2**20,
            'pinned_mib_on': resources[1]['retained_pinned_bytes'] / 2**20,
            'pinned_buffers_off': resources[0]['stats']['total_allocated'],
            'pinned_buffers_on': resources[1]['stats']['total_allocated'],
        })
        if kind == 'graph':
            for arm, row in (('off', off), ('on', on)):
                logs = list((performance / row['name'] / 'ranks').glob('**/0/stdout.log'))
                assert len(logs) == 1
                captures = [json.loads(line)['capture_seconds'] for line in logs[0].read_text().splitlines() if line.startswith('{"capture_seconds":')]
                assert len(captures) == 1
                samples[-1][f'capture_seconds_{arm}'] = captures[0]
    changes = [sample['step_change_percent'] for sample in samples]
    return {'pairs': samples, 'step_change_percent_median': statistics.median(changes), 'step_change_percent_range': [min(changes), max(changes)]}


result = {
    'eager_qwen_unfused': paired('eager', 'qwen', False, (2, 3, 4)),
    'eager_qwen_fused': paired('eager', 'qwen', True, (0, 1, 2)),
    'graph_qwen': paired('graph', 'qwen', False, (0, 1, 2, 3, 4)),
    'graph_gdn2': paired('graph', 'gdn2', False, (0, 1, 2)),
}
print(json.dumps(result, indent=2))
