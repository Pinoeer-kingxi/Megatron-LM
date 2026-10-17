"""Retain actual differing snapshot values for the four-rank regression case."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch

original = torch.testing.assert_close
calls = []


def differences(actual, expected, rtol, atol, path=''):
    if isinstance(expected, torch.Tensor):
        if not torch.equal(actual, expected):
            left, right = expected.detach().cpu(), actual.detach().cpu()
            delta = (right.double() - left.double()).abs()
            row = dict(path=path, dtype=str(left.dtype), shape=list(left.shape),
                       max_absolute_difference=delta.max().item(),
                       max_tolerance_ratio=(delta/(atol + rtol*left.double().abs())).max().item())
            if left.ndim <= 1:
                for name, value in [('expected', left), ('actual', right)]:
                    data = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                    row[name + '_values'] = value.tolist()
                    row[name + '_bytes_hex'] = data.hex()
                    row[name + '_sha256'] = hashlib.sha256(data).hexdigest()
            yield row
    elif isinstance(expected, dict):
        for key in expected:
            yield from differences(actual[key], expected[key], rtol, atol, path+'/'+str(key))
    elif isinstance(expected, (list, tuple)):
        for index, (right, left) in enumerate(zip(actual, expected, strict=True)):
            yield from differences(right, left, rtol, atol, path+'/'+str(index))
    elif actual != expected:
        yield dict(path=path, expected=expected, actual=actual)


def observed(actual, expected, **kwargs):
    if not isinstance(expected, dict) or not {'grads', 'weights', 'optimizer'} <= expected.keys():
        return original(actual, expected, **kwargs)
    passed = False
    try:
        result = original(actual, expected, **kwargs)
        passed = True
        return result
    finally:
        calls.append(dict(passed=passed, rtol=kwargs['rtol'], atol=kwargs['atol'],
                          differences=list(differences(actual, expected, kwargs['rtol'], kwargs['atol']))))


torch.testing.assert_close = observed
try:
    runpy.run_path(str(Path(__file__).with_name('rank_pytest.py')), run_name='__main__')
finally:
    directory = Path(sys.argv[1]).parent
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'numerical.rank{os.environ["RANK"]}.json').write_text(json.dumps(calls, indent=2)+'\n')
