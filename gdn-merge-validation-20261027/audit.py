"""Verify retained result counts, source identities, Qwen states, and file checksums."""
import hashlib
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent
jobs = {
    'shared-merge-5bc9e72-shared-gpu0': ('5bc9e72bda70129f088dd78328d4064f1d838c62', {'passed': 7}, 1),
    'feature-merge-b6fa63-feature-gpu0': ('b6fa63ab8c2cb855f0368cd8455245c295cde234', {'passed': 104, 'skipped': 84}, 1),
    'feature-merge-b6fa63-pipeline-gpu0': ('b6fa63ab8c2cb855f0368cd8455245c295cde234', {'passed': 4}, 1),
    'feature-merge-b6fa63-feature-native-gpu12': ('b6fa63ab8c2cb855f0368cd8455245c295cde234', {'passed': 24}, 2),
    'graph-merge-bb346eb-gdn2-scope-gpu0': ('bb346eb1c3461f471c420bcb7691b425863df3f5', {'passed': 20}, 1),
    'graph-merge-bb346eb-gdn2-native-gpu0': ('bb346eb1c3461f471c420bcb7691b425863df3f5', {'passed': 6}, 1),
    'graph-merge-bb346eb-gdn2-graph-gpu0': ('bb346eb1c3461f471c420bcb7691b425863df3f5', {'passed': 7}, 1),
    'graph-merge-bb346eb-gdn2-native-gpu12': ('bb346eb1c3461f471c420bcb7691b425863df3f5', {'passed': 12}, 2),
    'graph-merge-bb346eb-gdn2-graph-gpu12': ('bb346eb1c3461f471c420bcb7691b425863df3f5', {'passed': 14}, 2),
    'precision-final-aecc0e-scope-gpu0': ('aecc0e8116912554f27a62a51ad2aeca479a7f9e', {'passed': 111}, 1),
    'precision-final-aecc0e-native-gpu0': ('aecc0e8116912554f27a62a51ad2aeca479a7f9e', {'passed': 54}, 1),
    'precision-final-aecc0e-graph-gpu0': ('aecc0e8116912554f27a62a51ad2aeca479a7f9e', {'passed': 23}, 1),
    'precision-final-aecc0e-two-rank-gpu12': ('aecc0e8116912554f27a62a51ad2aeca479a7f9e', {'passed': 18}, 2),
}
for name, (revision, expected, ranks) in jobs.items():
    folder = root / name
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['revision'] == manifest['revision_after'] == revision, name
    assert manifest['exit_code'] == 0 and not manifest['dirty_after'], name
    for rank in range(ranks):
        data = json.loads((folder / 'reports' / f'rank-{rank}.json').read_text())
        observed = Counter(outcome['outcome'] for outcome in data['outcomes'])
        assert data['rank'] == rank and data['exit_code'] == 0 and observed == expected, (name, rank, observed)
        if name == 'precision-final-aecc0e-two-rank-gpu12':
            assert sum('fp4-True-' in x['nodeid'] for x in data['outcomes']) == 2

for topology, ranks in (('', 1), ('-tp2', 2)):
    states = []
    for arm in ('disabled', 'repeated-disabled', 'offloaded'):
        suffix = 'gpu0' if ranks == 1 else 'gpu12'
        folder = root / f'qwen-merge-b6fa63{topology}-{arm}-{suffix}'
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['revision'] == manifest['revision_after'] == 'b6fa63ab8c2cb855f0368cd8455245c295cde234'
        assert manifest['exit_code'] == 0 and not manifest['dirty_after']
        outputs = [json.loads((folder / f'result.rank{rank}.json').read_text()) for rank in range(ranks)]
        for output in outputs:
            assert output['text_layers'] == 24 and len(output['states']) == 3
            assert bool(output['offload_status']['selected_transfer_bytes']) == (arm == 'offloaded')
            assert output['pool_verified_before_reset']
        for rank in range(ranks):
            logs = list((folder / 'ranks').glob(f'**/{rank}/stdout.log'))
            assert len(logs) == 1 and 'Imported Megatron Core from /home/lwx/.cache/megatron-gdn-20261026/feature-merge-b6fa63/megatron/core/__init__.py' in logs[0].read_text()
        states.append([output['states'] for output in outputs])
    assert states[0] == states[1] == states[2], topology

assert 'DID NOT RAISE ValueError' in (root/'fp4-legacy-old.log').read_text()
assert '17 passed' in (root/'fp4-legacy-final.log').read_text()
for offset in (0, 1, 2, 4, 8, 16):
    result = (root/f'nvfp4-te-wgrad-offset-{offset}.log').read_text()
    assert ('PASS 448 256 1' in result) == (offset not in (1, 2)), offset
    if offset in (1, 2):
        assert 'CUBLAS_STATUS_NOT_SUPPORTED' in result
for line in (root/'SHA256SUMS').read_text().splitlines():
    digest, relative = line.split('  ', 1)
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == digest, relative
print(json.dumps({'integrity': 'passed', 'gpu_jobs': len(jobs), 'qwen_arms': 6,
                  'legacy_graph_guard': 'old control fails, fixed passes',
                  'offset_control': '4/8-byte misalignment fails; 16-byte-or-better passes'}, indent=2))
