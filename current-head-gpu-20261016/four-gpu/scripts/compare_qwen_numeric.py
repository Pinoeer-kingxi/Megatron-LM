"""Compare all fingerprints, inspecting actual tensors whenever bytes differ."""

import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RTOL, ATOL = 1e-5, 1e-8


def read(path):
    return json.loads(path.read_text())


def raw_bytes(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def compare(expected, actual, left_vectors, right_vectors, path, rows):
    if isinstance(expected, dict) and set(expected) == {'shape', 'dtype', 'sha256'}:
        assert expected['shape'] == actual['shape'] and expected['dtype'] == actual['dtype'], path
        row = dict(path=path, exact=expected == actual)
        if not row['exact']:
            left = left_vectors[(expected['dtype'], tuple(expected['shape']), expected['sha256'])]
            right = right_vectors[(actual['dtype'], tuple(actual['shape']), actual['sha256'])]
            assert hashlib.sha256(raw_bytes(left)).hexdigest() == expected['sha256'], path
            assert hashlib.sha256(raw_bytes(right)).hexdigest() == actual['sha256'], path
            assert list(left.shape) == expected['shape'] and str(left.dtype) == expected['dtype']
            assert list(right.shape) == actual['shape'] and str(right.dtype) == actual['dtype']
            assert torch.isfinite(left).all() and torch.isfinite(right).all(), path
            delta = (right.double() - left.double()).abs()
            ratio = delta / (ATOL + RTOL * left.double().abs())
            row.update(expected=expected, actual=actual,
                       expected_values=left.tolist(), actual_values=right.tolist(),
                       expected_bytes_hex=raw_bytes(left).hex(), actual_bytes_hex=raw_bytes(right).hex(),
                       max_absolute_difference=delta.max().item() if delta.numel() else 0,
                       max_tolerance_ratio=ratio.max().item() if ratio.numel() else 0)
            rows.append(row)
            torch.testing.assert_close(right, left, rtol=RTOL, atol=ATOL, msg=path)
        else:
            rows.append(row)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        for key in expected:
            compare(expected[key], actual[key], left_vectors, right_vectors, path + '/' + str(key), rows)
    elif isinstance(expected, list):
        assert len(expected) == len(actual), path
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare(left, right, left_vectors, right_vectors, path + '/' + str(index), rows)
    elif isinstance(expected, float):
        assert math.isfinite(expected) and math.isfinite(actual), path
        assert abs(actual - expected) <= ATOL + RTOL * abs(expected), path
        if expected != actual:
            rows.append(dict(path=path, scalar=True, exact=False, expected=expected, actual=actual,
                             max_absolute_difference=abs(actual - expected),
                             max_tolerance_ratio=abs(actual - expected)/(ATOL + RTOL * abs(expected))))
    else:
        assert expected == actual, path


summaries = []
for norm in (0, 1):
    for rank in range(4):
        arms = []
        for arm in ('disabled', 'repeat', 'offload'):
            directory = ROOT / f'qwen-f1-n{norm}-{arm}'
            result = read(directory / f'result.rank{rank}.json')
            assert result['gpu_training_run'] and result['pool_verified_before_reset']
            assert result['rank'] == rank and result['text_layers'] == 24 and not result['samples']
            args = result['arguments']
            expected = dict(tp=2, sp=True, cp=1, pp=2, vp=None, seq_length=128, microbatches=4,
                            warmup=1, iterations=2, check_state=True, fused_pre_gdr=True,
                            recompute_norm=bool(norm), min_offloaded_tensor_size=1024,
                            fraction=1.0 if arm == 'offload' else None, dry_run=False)
            assert all(args[key] == value for key, value in expected.items())
            states = result['states']
            assert [s['step'] for s in states] == [0, 1, 2]
            for state in states:
                assert len(state['losses']) == (4 if rank >= 2 else 0) and all(math.isfinite(x) for x in state['losses'])
                assert math.isfinite(state['grad_norm']) and state['grad_norm'] > 0
                assert all(state[field] for field in ('grads', 'weights', 'optimizer'))
            selected = result['offload_status']['selected_transfer_bytes']
            assert (selected > 0) if arm == 'offload' else (selected == 0)
            vectors = torch.load(directory / f'vectors.rank{rank}.pt', map_location='cpu', weights_only=True)
            arms.append((states, vectors))
        for arm, (states, vectors) in zip(('repeat', 'offload'), arms[1:]):
            rows = []
            compare(arms[0][0], states, arms[0][1], vectors, '', rows)
            report = f'numeric-f1-n{norm}-{arm}-rank{rank}-differences.json'
            (ROOT / report).write_text(json.dumps(rows, indent=2) + '\n')
            changed = [row for row in rows if not row['exact']]
            summaries.append(dict(norm=bool(norm), rank=rank, arm=arm, compared_steps=3,
                                  close=True, exact=arms[0][0] == states,
                                  tensor_leaves=sum(not r.get('scalar', False) for r in rows),
                                  differing_leaves=len(changed),
                                  max_absolute_difference=max((row['max_absolute_difference'] for row in changed), default=0),
                                  max_tolerance_ratio=max((row['max_tolerance_ratio'] for row in changed), default=0),
                                  report=report))
result = dict(rtol=RTOL, atol=ATOL, comparisons=summaries, compared_rank_steps=48,
              recorded_rank_steps=72, all_numerically_close=True,
              capture='All vector/scalar tensors captured at the committed fingerprint boundary; every other tensor must have identical full-byte fingerprints. All differing vectors and raw bytes are retained in difference reports.')
(ROOT / 'fused-numeric-summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
