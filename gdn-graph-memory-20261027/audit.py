"""Check the recorded fraction and delayed-offload comparisons."""
import gzip
import json
from pathlib import Path

root = Path(__file__).parent / 'results'

def read(name):
    path = root / name
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)['records']

baseline = read('fraction-off.json')
assert len(baseline) == 16
for arm in ('0.25', '0.5', 'on'):
    other = read(f'fraction-{arm}.json')
    assert len(other) == len(baseline)
    for a, b in zip(baseline, other):
        assert all(a[key] == b[key] for key in ('input_sha256', 'loss', 'grad_norm'))
        if a['replay']:
            assert b['selected_bytes'] > 0

a, b = read('delay-check-off.json.gz'), read('delay-check-on.json.gz')
assert len(a) == len(b) == 6
for left, right in zip(a, b):
    assert all(left[key] == right[key] for key in
               ('input_sha256', 'loss', 'grad_norm', 'gradients', 'weights', 'optimizer'))
    if left['replay']:
        assert right['selected_bytes'] > 0
a, b = read('gdn2-off.json'), read('gdn2-on.json')
assert len(a) == len(b) == 7
for left, right in zip(a, b):
    assert all(left[key] == right[key] for key in ('input_sha256', 'loss', 'grad_norm'))
print('Fraction and GDN2 loss/norm/input and delayed full-state comparisons passed.')
