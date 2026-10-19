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


ROOT = Path(__file__).resolve().parents[1]
left=read(ROOT/'state-baseline/result.rank0.json')
left_vectors=torch.load(ROOT/'state-baseline/vectors.rank0.pt',map_location='cpu',weights_only=True)
summary=[]
for arm in ('final',):
 right=read(ROOT/'final-state-candidate/result.rank0.json')
 right_vectors=torch.load(ROOT/'final-state-candidate/vectors.rank0.pt',map_location='cpu',weights_only=True)
 assert left['gpu_training_run'] and right['gpu_training_run']
 assert left['states'] and len(left['states'])==len(right['states'])==3
 rows=[];compare(left['states'],right['states'],left_vectors,right_vectors,'',rows)
 (ROOT/f'state-{arm}-differences.json').write_text(json.dumps(rows,indent=2)+'\n')
 changes=[row for row in rows if not row['exact']]
 summary.append({'arm':arm,'exact':left['states']==right['states'],'compared_steps':3,'differing_leaves':len(changes),'max_tolerance_ratio':max((x['max_tolerance_ratio'] for x in changes),default=0),'close':True})
(ROOT/'final-state-comparison.json').write_text(json.dumps({'rtol':RTOL,'atol':ATOL,'comparisons':summary},indent=2)+'\n')
print(json.dumps(summary,indent=2))
