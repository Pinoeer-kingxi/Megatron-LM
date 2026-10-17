"""Recheck this bundle on CPU; this does not execute GPU tests."""

import json
import math
from pathlib import Path

from verify_results import verify_pytest

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads((ROOT / path).read_text())

manifest = read('validation.json')
source = read('source.json')
assert manifest['status'] == 'failed' and manifest['cleanup']['exit_code'] == 0
assert manifest['error'] == "AssertionError: (True, False, 'offload', 1)"
numeric = read('fused-numeric.json')
assert numeric['status'] == 'passed' and numeric['cleanup']['exit_code'] == 0
assert numeric['source_commit'] == manifest['source_commit']
assert len(numeric['jobs']) == 8 and all(j['exit_code'] == 0 for j in numeric['jobs'])
assert manifest['source_commit'] == source['source_commit'] == 'fbe4bb03d1c6a60ac4da965ca47bccd47d7ef452'
assert manifest['gate_source'] == source['gate_source'] == '9d54b8729c9277f1e032110c5e1b149e9654bcc5'
assert manifest['shared_source'] == source['shared_source'] == 'af9569270f476dc201484296315a9c30dcaa3feb'
expected_jobs = ['shared-groups', 'native-training']
expected_jobs += [f'qwen-f{f}-n{n}-{arm}' for f in (0, 1) for n in (0, 1)
                  for arm in ('disabled', 'repeat', 'offload')]
expected_jobs = expected_jobs[:11]
assert [job['name'] for job in manifest['jobs']] == expected_jobs
assert all(job['exit_code'] == 0 for job in manifest['jobs'])
shared = verify_pytest(ROOT / 'shared-groups/reports', 2, 16)
native = verify_pytest(ROOT / 'native-training/reports', 2, 96, allow_skips=True)
for rank in native['ranks']:
    assert rank['passed'] == 72 and len(rank['skipped']) == 24
    assert all('tp_sp_cp-' in row['nodeid'] or 'tp_sp_pp-' in row['nodeid'] for row in rank['skipped'])
gate = verify_pytest(ROOT / 'gate-replay/reports', 1, 16)
comparisons = []
for fused in (0,):
    for norm in (0, 1):
        arms = []
        for arm in ('disabled', 'repeat', 'offload'):
            ranks = []
            for rank in range(2):
                path = f'qwen-f{fused}-n{norm}-{arm}/result.rank{rank}.json'
                result = read(path)
                assert result['gpu_training_run'] and result['pool_verified_before_reset']
                assert result['rank'] == rank and result['text_layers'] == 24
                args = result['arguments']
                expected = dict(tp=2, sp=True, cp=1, pp=1, vp=None, seq_length=128,
                                microbatches=4, warmup=1, iterations=2, check_state=True,
                                fused_pre_gdr=bool(fused), recompute_norm=bool(norm),
                                min_offloaded_tensor_size=1024,
                                fraction=1.0 if arm == 'offload' else None, dry_run=False)
                assert all(args[key] == value for key, value in expected.items()), path
                assert not result['samples']
                states = result['states']
                assert [state['step'] for state in states] == [0, 1, 2]
                for state in states:
                    assert len(state['losses']) == 4 and all(math.isfinite(x) for x in state['losses'])
                    assert math.isfinite(state['grad_norm']) and state['grad_norm'] > 0
                    assert all(state[field] for field in ('grads', 'weights', 'optimizer'))
                selected = result['offload_status']['selected_transfer_bytes']
                assert (selected > 0) if arm == 'offload' else (selected == 0)
                ranks.append(states)
            arms.append(ranks)
        for arm, values in zip(('repeat', 'offload'), arms[1:]):
            for rank in range(2):
                assert values[rank] == arms[0][rank], (fused, norm, arm, rank)
                comparisons.append(dict(fused=bool(fused), norm=bool(norm), arm=arm,
                                        rank=rank, compared_steps=3, exact_recorded_state=True))
assert sum(row['compared_steps'] for row in comparisons) == 24
numeric_summary = read('fused-numeric-summary.json')
assert numeric_summary['all_numerically_close'] and numeric_summary['compared_rank_steps'] == 24
assert (numeric_summary['rtol'], numeric_summary['atol']) == (1e-5, 1e-8)
assert len(numeric_summary['comparisons']) == 8
import hashlib
for comparison in numeric_summary['comparisons']:
    assert comparison['close'] and comparison['compared_steps'] == 3
    rows = read(comparison['report'])
    assert comparison['tensor_leaves'] == sum(not row.get('scalar', False) for row in rows)
    changed = [row for row in rows if not row['exact']]
    assert len(changed) == comparison['differing_leaves']
    for row in changed:
        assert math.isfinite(row['max_tolerance_ratio']) and row['max_tolerance_ratio'] <= 1
        if not row.get('scalar', False):
            for arm in ('expected', 'actual'):
                assert hashlib.sha256(bytes.fromhex(row[arm + '_bytes_hex'])).hexdigest() == row[arm]['sha256']
            left, right = row['expected_values'], row['actual_values']
            if not isinstance(left, list):
                left, right = [left], [right]
            assert len(left) == len(right)
            assert all(math.isfinite(a) and math.isfinite(b) and abs(b-a) <= 1e-8 + 1e-5 * abs(a)
                       for a,b in zip(left,right,strict=True))

summary = dict(source_commit=source['source_commit'], gate_source=source['gate_source'],
               shared_component_source=source['shared_source'], shared=shared, native=native,
               independent_gate_replay=gate, qwen_comparisons=comparisons,
               qwen_recorded_rank_steps=72, qwen_compared_rank_steps=48,
               unfused_exact_compared_rank_steps=24, fused_numeric_compared_rank_steps=24,
               fused_numeric_comparisons=numeric_summary['comparisons'],
               initial_exact_probe='Failed on fused norm-off offload rank 1; raw failure retained in validation.json.',
               limits=['Only two GPUs were available.',
                       'Four-rank combined TP+SP+CP and TP+SP+PP cases are skipped.',
                       'Qwen checks use TP2+SP, PP1, sequence 128 and short training.',
                       'Unfused states match exactly. Native fused states use actual-vector checks with rtol=1e-5, atol=1e-8; tensors not captured as vectors must match full-byte fingerprints. No global bitwise guarantee is made.',
                       'Gate replay tests use independent PR #8244 source, not the restored upstream gate in #7908.'])
(ROOT / 'validation-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: summary[key] for key in ('source_commit', 'qwen_recorded_rank_steps', 'qwen_compared_rank_steps', 'limits')}, indent=2))
