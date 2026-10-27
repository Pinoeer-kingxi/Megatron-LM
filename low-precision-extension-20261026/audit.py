"""Audit retained test outcomes and failures using only the Python standard library."""
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
head = '1aa1151a213c13de1faab0930c5e6bc91662c042'
prototype = '18fe09f4743ee12ccb972679da7b25fdc4b19068'


def read(path):
    return json.loads((root / path).read_text())


def manifest(directory, revision=head, exit_code=0):
    item = read(f'{directory}/manifest.json')
    assert item['revision'] == item['revision_after'] == revision, directory
    assert item['exit_code'] == exit_code and not item['dirty_after'], directory


def tests(directory, count, ranks=1):
    manifest(directory)
    for rank in range(ranks):
        report = read(f'{directory}/reports/rank-{rank}.json')
        assert report['exit_code'] == 0 and report['collected'] == count, directory
        assert len(report['outcomes']) == count, directory
        assert all(row['when'] == 'call' and row['outcome'] == 'passed'
                   for row in report['outcomes']), directory


tests('precision-graphs-1aa1151-scope-final-gpu3', 108)
tests('precision-graphs-1aa1151-native-final-gpu3', 54)
tests('precision-graphs-1aa1151-graph-final-gpu3', 14)
tests('precision-graphs-1aa1151-graph-two-rank-gpu01', 8, ranks=2)

source = read('source.json')
assert source['head'] == head and len(source['changed_files']) == 7 and not source['dirty']
assert 'megatron/core/pipeline_parallel/fine_grained_activation_offload.py' not in source['changed_files']
assert all(row['exit_code'] == 0 for row in read('static-checks.json')['checks'])
assert read('static-checks.json')['source'] == head
assert read('mypy-comparison.json')['new'] == {}
assert read('mypy-comparison.json')['base_count'] == read('mypy-comparison.json')['head_count'] == 4

stalled = 'precision-graphs-057b70f-graph-two-rank-gpu03'
assert read(f'{stalled}/manifest.json')['exit_code'] != 0
assert read(f'{stalled}/termination.json')['signal'] == 'SIGTERM'

for variant in ('gdn', 'gdn2'):
    directory = f'precision-graphs-18fe09f-{variant}-fp4-graph-diagnostic-gpu3'
    manifest(directory, prototype, exit_code=1)
    logs = '\n'.join(path.read_text() for path in (root / directory).rglob('stderr.log'))
    assert 'Original failure before graph cleanup; offload=False' in logs, variant
    assert ('an illegal memory access' in logs if variant == 'gdn' else 'nan' in logs), variant

for width in (2592, 4096):
    directory = f'precision-te-fp4-graph-{width}-gpu0'
    assert read(f'{directory}/manifest.json')['exit_code'] == 0
    rows = [json.loads(line) for line in (root / directory / 'launcher.log').read_text().splitlines()
            if line.startswith('{')]
    assert len(rows) == 3 and all(math.isfinite(row['loss']) for row in rows), directory

for line in (root / 'SHA256SUMS').read_text().splitlines():
    expected, name = line.split('  ', 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name

print(json.dumps(dict(integrity='passed', source=head, scope_and_tool=108,
    native_low_precision=54, single_rank_graphs=14, two_rank_graphs_per_rank=8,
    nvfp4_graphs='excluded; failed offload-disabled prototypes retained',
    numerical_comparisons='performed on real tensors inside the executed tests; not recomputed by this audit'), indent=2))
