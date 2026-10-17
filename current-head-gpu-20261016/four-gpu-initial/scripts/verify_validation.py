"""Recheck four-rank reports and full-Qwen states on CPU without launching CUDA."""

import hashlib
import json
import math
from pathlib import Path

from verify_results import verify_pytest

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads((ROOT / path).read_text())

manifest = read('validation.json')
assert manifest['source_commit'] == 'fbe4bb03d1c6a60ac4da965ca47bccd47d7ef452'
assert manifest['status'] == 'passed' and manifest['cleanup']['exit_code'] == 0
expected_jobs = ['shared-groups', 'native-training']
expected_jobs += [f'qwen-f{f}-n{n}-{arm}' for f in (0, 1) for n in (0, 1)
                  for arm in ('disabled', 'repeat', 'offload')]
expected_jobs += ['compare-fused-numeric']
assert [job['name'] for job in manifest['jobs']] == expected_jobs
assert all(job['exit_code'] == 0 for job in manifest['jobs'])
shared = verify_pytest(ROOT / 'shared-groups/reports', 4, 16)
native = verify_pytest(ROOT / 'native-training/reports', 4, 96)
comparisons = []
for norm in (0, 1):
    arms = []
    for arm in ('disabled', 'repeat', 'offload'):
        ranks = []
        for rank in range(4):
            result = read(f'qwen-f0-n{norm}-{arm}/result.rank{rank}.json')
            assert result['gpu_training_run'] and result['pool_verified_before_reset']
            assert result['rank'] == rank and result['text_layers'] == 24 and not result['samples']
            args = result['arguments']
            expected = dict(tp=2, sp=True, cp=1, pp=2, vp=None, seq_length=128, microbatches=4,
                            warmup=1, iterations=2, check_state=True, fused_pre_gdr=False,
                            recompute_norm=bool(norm), min_offloaded_tensor_size=1024,
                            fraction=1.0 if arm == 'offload' else None, dry_run=False)
            assert all(args[key] == value for key, value in expected.items())
            states = result['states']
            assert [s['step'] for s in states] == [0, 1, 2]
            for state in states:
                assert len(state['losses']) == (4 if rank >= 2 else 0)
                assert all(math.isfinite(x) for x in state['losses'])
                assert math.isfinite(state['grad_norm']) and state['grad_norm'] > 0
                assert all(state[field] for field in ('grads', 'weights', 'optimizer'))
            selected = result['offload_status']['selected_transfer_bytes']
            assert (selected > 0) if arm == 'offload' else (selected == 0)
            ranks.append(states)
        arms.append(ranks)
    for arm, values in zip(('repeat', 'offload'), arms[1:]):
        for rank in range(4):
            assert arms[0][rank] == values[rank], (norm, arm, rank)
            comparisons.append(dict(norm=bool(norm), rank=rank, arm=arm, compared_steps=3, exact=True))
assert sum(row['compared_steps'] for row in comparisons) == 48
numeric = read('fused-numeric-summary.json')
assert (numeric['rtol'], numeric['atol']) == (1e-5, 1e-8)
assert numeric['all_numerically_close'] and numeric['compared_rank_steps'] == 48
assert numeric['recorded_rank_steps'] == 72 and len(numeric['comparisons']) == 16
for comparison in numeric['comparisons']:
    assert comparison['close'] and comparison['compared_steps'] == 3
    rows = read(comparison['report'])
    changed = [row for row in rows if not row['exact']]
    assert comparison['tensor_leaves'] == sum(not row.get('scalar', False) for row in rows)
    assert len(changed) == comparison['differing_leaves']
    for row in changed:
        assert math.isfinite(row['max_tolerance_ratio']) and row['max_tolerance_ratio'] <= 1
        if not row.get('scalar', False):
            for arm in ('expected', 'actual'):
                assert hashlib.sha256(bytes.fromhex(row[arm + '_bytes_hex'])).hexdigest() == row[arm]['sha256']
            left, right = row['expected_values'], row['actual_values']
            if not isinstance(left, list):
                left, right = [left], [right]
            assert all(math.isfinite(a) and math.isfinite(b) and abs(b-a) <= 1e-8 + 1e-5 * abs(a)
                       for a,b in zip(left,right,strict=True))
summary = dict(source_commit=manifest['source_commit'], shared=shared, native=native,
               qwen_topology='TP2+SP+PP2, CP1, four A6000s',
               qwen_recorded_rank_steps=144, qwen_compared_rank_steps=96,
               unfused_exact_comparisons=comparisons, fused_numeric_comparisons=numeric['comparisons'],
               limits=['Qwen state checks use sequence 128 and three optimizer steps per arm.',
                       'Native fused atomic reductions have no global bitwise guarantee.',
                       'GPU CI, long-run convergence, checkpoint restart and pretrained Qwen CP are not qualified.'])
(ROOT / 'validation-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k:summary[k] for k in ('source_commit', 'qwen_topology', 'qwen_recorded_rank_steps', 'qwen_compared_rank_steps', 'limits')}, indent=2))
