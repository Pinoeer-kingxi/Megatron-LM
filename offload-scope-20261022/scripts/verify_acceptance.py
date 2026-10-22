"""Audit pinned-source executions, expected old-code failures, and complete Qwen states."""

import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sources = json.loads((root/'sources.json').read_text())


def read(path):
    return json.loads(path.read_text())


def verify_job(name, source, ranks, count, expected_failure=False):
    directory = root/name
    manifest = read(directory/'manifest.json')
    assert manifest['revision'] == sources[source]['revision']
    assert (manifest['exit_code'] == 0) != expected_failure
    assert manifest['revision_after'] == manifest['revision'] and not manifest['dirty_after']
    assert manifest['harness_sha256'] == hashlib.sha256((root/'scripts/run_job.py').read_bytes()).hexdigest()
    for rank in range(ranks):
        report = read(directory/'reports'/f'rank-{rank}.json')
        assert report['exit_code'] == (1 if expected_failure else 0)
        assert report['rank'] == rank and report['collected'] == count
        assert len(report['outcomes']) == count
        assert len({row['nodeid'] for row in report['outcomes']}) == count
        for row in report['outcomes']:
            assert row['when'] == 'call'
            assert row['outcome'] == ('failed' if expected_failure else 'passed')
            if expected_failure:
                assert 'Chunk mismatch' in row['detail']
        for stream in ('stdout','stderr'):
            assert len(list((directory/'ranks').rglob(f'{rank}/{stream}.log'))) == 1
    return dict(name=name,source=source,revision=manifest['revision'],ranks=ranks,
                cases_per_rank=count,status='expected old-code failure' if expected_failure else 'passed',skipped=0)


positive = [verify_job(*arguments) for arguments in [
    ('control-positive-fractions','control',2,2),
    ('shared-core','shared',2,23),
    ('shared-four-fixed','shared',4,4),
    ('feature-four-native','feature',4,96),
    ('feature-scopes','feature',2,92),
    ('extensions-scopes','extensions',2,20),
    ('extensions-native','extensions',2,18),
    ('extensions-graphs','extensions',2,21),
    ('shared-graphs','shared',2,3),
]]
negative = [verify_job(*arguments,expected_failure=True) for arguments in [
    ('control-final-zero','control',2,1),
    ('gdn-old-chunk-control','feature-control',2,1),
]]

arms = []
reference = None
for name,fraction in [('disabled',None),('repeat-disabled',None),('fraction-zero',0.0),('offloaded',1.0)]:
    directory = root/f'qwen-{name}'
    manifest = read(directory/'manifest.json')
    assert manifest['revision'] == sources['feature']['revision'] and manifest['exit_code'] == 0
    assert manifest['revision_after'] == manifest['revision'] and not manifest['dirty_after']
    assert manifest['harness_sha256'] == hashlib.sha256((root/'scripts/run_qwen.py').read_bytes()).hexdigest()
    outputs = []
    for rank in range(4):
        output = read(directory/f'result.rank{rank}.json')
        args = output['arguments']
        assert output['gpu_training_run'] and output['pool_verified_before_reset']
        assert output['rank'] == rank and output['text_layers'] == 24
        expected = dict(tp=2,sp=True,pp=2,cp=1,vp=None,seq_length=128,microbatches=4,
                        warmup=1,iterations=2,min_offloaded_tensor_size=1024,check_state=True,
                        recompute_norm=True,fused_pre_gdr=False,fraction=fraction)
        assert all(args[key]==value for key,value in expected.items())
        common = {key:value for key,value in args.items() if key not in ('fraction','output')}
        if reference is None:
            reference = common
        assert common == reference
        assert (output['offload_status']['selected_transfer_bytes'] > 0) == (name=='offloaded')
        assert [row['step'] for row in output['states']] == [0,1,2]
        for row in output['states']:
            assert len(row['losses']) == (4 if rank>=2 else 0)
            assert all(math.isfinite(loss) for loss in row['losses'])
            assert math.isfinite(row['grad_norm']) and row['grad_norm']>0
            assert all(row[field] for field in ('grads','weights','optimizer'))
        outputs.append(output['states'])
    arms.append(outputs)
assert all(arm == arms[0] for arm in arms[1:])

for name in ('shared','feature','extensions'):
    row = read(root/f'lint-{name}.json')
    assert row['revision'] == sources[name]['revision'] and row['exit_code'] == 0
assert read(root/'mypy-comparison.json')['unchanged']
for row in read(root/'latest-main-merge-check.json'):
    assert row['revision'] == sources[row['source']]['revision']
    assert row['clean_merge'] and row['core_unchanged']
for row in read(root/'kernel-coverage.json'):
    assert row['revision'] == sources[row['source']]['revision'] and row['exit_code'] == 0
assert {row['revision'] for row in read(root/'signature-verification.json')} == {
    sources[name]['revision'] for name in ('shared','feature','extensions')}
assert all(row['exit_code'] == 0 for row in read(root/'signature-verification.json'))

result = dict(status='passed',positive_jobs=positive,negative_controls=negative,
              qwen=dict(revision=sources['feature']['revision'],ranks=4,steps=3,
                        arms=['disabled','repeat-disabled','fraction-zero','offloaded'],
                        complete_recorded_states_exact=True),
              limits=['Fixed-length training-first validation; evaluation-first and tensor-size growth changes withdrawn.',
                      'Current full-Qwen check is unfused with norm recomputation; other full-model settings remain historical.',
                      'No current-head throughput benchmark or pretrained CUDA-graph training run.',
                      'Local A6000 evidence; upstream GPU CI pending.'])
(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
