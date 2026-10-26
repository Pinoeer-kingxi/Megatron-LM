"""Verify current-source outcomes, matched comparisons, and immutable file checksums."""
import hashlib
import json
import statistics
from pathlib import Path

root=Path(__file__).resolve().parents[1]
def read(name): return json.loads((root/name).read_text())
for job, passed, skipped in [('shared-native-lifecycle',9,0),('shared-graph-compatibility',1,0),
    ('shared-graph-fractions',2,0),('feature-native-tests',104,84),('feature-final-lifecycle',9,0)]:
    result=read(job+'/reports/rank-0.json')
    assert result['exit_code']==0 and result['collected']==passed+skipped
    assert sum(x['outcome']=='passed' for x in result['outcomes'])==passed
    assert sum(x['outcome']=='skipped' for x in result['outcomes'])==skipped
for job in ('shared-control-evaluation','shared-control-growth','lifecycle-control-eval-matched','lifecycle-controls-calibrated'):
    result=read(job+'/reports/rank-0.json')
    assert result['exit_code']==1 and sum(x['outcome']=='failed' for x in result['outcomes'])==1
for path in (root/'feature-final-lifecycle/observations').glob('*.rank0.json'):
    value=json.loads(path.read_text())
    if '.baseline-' in path.name:
        if '.baseline-2.' in path.name: assert value==[]
    else:
        assert value['states_exact'] and value['state_differences']==[]
        assert value['pool_before_reset']==[0]*len(value['lengths'])
        steady=sum(x['bytes'] for x in value['transfers'] if not x['warmup'])
        assert bool(steady)==(value['fraction']>0)
        assert value['pool']['current_in_use']==0
outputs=[read('qwen-correctness-'+arm+'/result.rank0.json') for arm in ('disabled','repeated-disabled','offloaded')]
assert len(outputs[0]['states'])==3 and outputs[0]['states']==outputs[1]['states']==outputs[2]['states']
assert outputs[2]['offload_status']['selected_transfer_bytes']>0
records=read('performance-progress.json')['records']
assert len(records)==12
for row in records:
    name=row['name']; data=read(name+'/result.rank0.json'); manifest=read(name+'/manifest.json')
    assert manifest['exit_code']==0 and manifest['revision']==manifest['revision_after'] and not manifest['dirty_after']
    assert len(data['samples'])==10 and not data['arguments']['check_state']
    assert row['median_ms']==1000*statistics.median(x['seconds'] for x in data['samples'])
    assert row['resources']['stats']['current_in_use']==0
precision=read('remote-native-final/native-replay-reports/rank-0.json')
assert precision['exit_code']==0 and precision['collected']==36
assert all(x['outcome']=='passed' for x in precision['outcomes'])
for line in (root/'SHA256SUMS').read_text().splitlines():
    expected,name=line.split('  ',1)
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
print('PASS: recorded outcomes, expected failures, state equality, samples and SHA256SUMS')
