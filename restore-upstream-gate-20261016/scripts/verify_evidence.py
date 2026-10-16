"""Check the retained restoration evidence without importing Torch or using CUDA."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = '67fb40d842035a8873979759775b4165381ae60e'

def read(name):
    return json.loads((ROOT / name).read_text())

restoration = read('restoration-check.json')
assert restoration['source_commit'] == REVISION and restoration['gate_source_identical']
assert restoration['exclusive_gate_replay_removed']
assert restoration['gate_sha256'] == 'c8a0247766bbc78ee3f98841a3eb9c033109bf3575bb9d2df99b4cbf2a0df920'
source = read('source-validation.json')
assert source['source_commit'] == REVISION
assert source['archive_files_compared'] == 3532 and source['archive_files_unchanged']
assert source['source_worktree_clean']
final = read('final-head.json')
assert final['final_head'] == 'b113b63614f201572f1e3742b4765449e55ceb3c'
assert final['executed_source'] == REVISION
assert final['complete_tree_identical'] and final['removed_gate_fix_revert_history']
assert final['executed_tree'] == final['final_tree'] == '1265e652350d601b43f66dc8527d6e26f0ac5fe8'
assert final['feature_commits_above_shared'] == 1
commits = read('commit-trees.json')
assert {row['sha'] for row in commits} == {REVISION, final['final_head']}
assert all(row['tree'] == final['final_tree'] for row in commits)
assert next(row for row in commits if row['sha'] == final['final_head'])['parents'] == [final['shared_parent']]
for name in ('checks.json', 'long-checks.json'):
    record = read(name)
    assert record['source_commit'] == REVISION and record['status'] == 'passed'
    assert record['cleanup']['exit_code'] == 0
    assert all(row['exit_code'] == 0 for row in record['jobs'])
for name, count in [('gdn-scopes', 73), ('native-dp', 12)]:
    record = read(name + '/reports/rank-0.json')
    assert record['exit_code'] == 0 and record['rank'] == 0
    assert len(record['outcomes']) == count
    assert all(row['outcome'] == 'passed' for row in record['outcomes'])
for name, count in [('cpu-feature', 30), ('manifest', 269)]:
    assert f'{count} passed' in (ROOT / name / 'launcher.log').read_text()
for norm in (0, 1):
    name = f'long-norm-{norm}'
    result = read(name + '/result.json')
    assert result['source_commit'] == REVISION
    assert result['sequence'] == 2048 and result['steps_per_arm'] == 3
    assert result['microbatches'] == 4 and result['fused_pre_gdr']
    assert result['recompute_norm'] == bool(norm)
    assert (result['rtol'], result['atol']) == (1e-5, 1e-8)
    assert result['pool_ownership_before_reset'] == [0, 0, 0]
    assert result['offload_summary_bytes'] > 0
    assert {row['name'] for row in result['comparisons']} == {'disabled-repeat', 'offload'}
    for comparison in result['comparisons']:
        rows = read(name + '/' + comparison['name'] + '-tensor-differences.json')
        assert len(rows) == comparison['tensor_leaves'] == 1185
        assert all(math.isfinite(row['max_tolerance_ratio']) and row['max_tolerance_ratio'] <= 1
                   and row['elements'] > 0 for row in rows)
        assert comparison['numerically_close'] and comparison['exact']
        assert comparison['differing_tensor_leaves'] == sum(not row['exact'] for row in rows) == 0
        assert all(row['max_absolute_difference'] == 0 for row in rows)
        for category in ('/losses/', '/grads/', '/weights/', '/optimizer/'):
            assert any(category in row['path'] for row in rows), category
checksums = ROOT / 'SHA256SUMS'
if checksums.exists():
    for row in checksums.read_text().splitlines():
        expected, name = row.split('  ', 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
print('Restoration, source provenance, test outcomes and long-sequence reports verified.')
