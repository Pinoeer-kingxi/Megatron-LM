"""Audit public run provenance and outcomes without GPU dependencies."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
head = '7d52248294fd759dd69b83c68ec6d1c0b36f2836'
prototype = '18fe09f4743ee12ccb972679da7b25fdc4b19068'


def read(path):
    return json.loads((root / path).read_text())


def manifest(directory, revision, exit_code):
    item = read(f'{directory}/manifest.json')
    assert item['revision'] == item['revision_after'] == revision, directory
    assert item['exit_code'] == exit_code and not item['dirty_after'], directory


def tests(directory, count, ranks=1, fp4=None):
    manifest(directory, head, 0)
    for rank in range(ranks):
        report = read(f'{directory}/reports/rank-{rank}.json')
        assert report['exit_code'] == 0 and report['collected'] == count, directory
        assert len(report['outcomes']) == count, directory
        assert all(row['when'] == 'call' and row['outcome'] == 'passed'
                   for row in report['outcomes']), directory
        if fp4 is not None:
            assert sum('-fp4-' in row['nodeid'] for row in report['outcomes']) == fp4, directory


tests('precision-graphs-7d52248-scope-final-gpu1', 109)
tests('precision-graphs-7d52248-native-final-gpu1', 54)
tests('precision-graphs-7d52248-graph-final-gpu0', 21, fp4=7)
tests('precision-graphs-7d52248-graph-two-rank-gpu01', 16, ranks=2, fp4=8)
source = read('source.json')
assert source['head'] == head and len(source['changed_files']) == 8 and not source['dirty']
assert 'megatron/core/pipeline_parallel/fine_grained_activation_offload.py' not in source['changed_files']
static = read('static-checks.json')
assert static['source'] == head and all(row['exit_code'] == 0 for row in static['checks'])
mypy = read('mypy-comparison.json')
assert mypy['head'] == head and mypy['new'] == {}

for mode, exit_code in [('plain', 1), ('no-weight-cache', 0), ('no-io-reuse', 1),
                         ('clone-param-grads', 1), ('disable-existing-cache', 1),
                         ('disable-both-caches', 0), ('trace', 1)]:
    manifest(f'nvfp4-graph-options-gdn2-{mode}-gpu0', prototype, exit_code)
manifest('nvfp4-graph-options-gdn-disable-both-caches-gpu0', prototype, 0)

for line in (root / 'SHA256SUMS').read_text().splitlines():
    expected, name = line.split('  ', 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name

print(json.dumps(dict(integrity='passed', source=head, scope_and_tool=109,
    native_low_precision=54, single_rank_graphs=21, nvfp4_single_rank_graphs=7,
    two_rank_graphs_per_rank=16, nvfp4_two_rank_graphs_per_rank=8,
    nvfp4_graph_boundary='existing parameter cache disabled; RHT/SR disabled on SM120',
    numerical_comparisons='eager/replay/offload equality executed inside GPU tests; not recomputed offline'), indent=2))
