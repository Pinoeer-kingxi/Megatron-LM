"""Check complete fixed-length Qwen training with the minimal chunk prerequisite."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
old = Path('/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/offload-review-20261021')
source = Path('/home/songbinbin/gdn-offload-artifacts/scope-20261022/feature')
previous = '/home/songbinbin/gdn-offload-artifacts/review-20261021/feature'
revision = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
reference = json.loads((old/'four-qwen-f0-n1-disabled/manifest.json').read_text())['command']
records = []
for arm, fraction in [('disabled', None), ('repeat-disabled', None), ('fraction-zero', '0'), ('offloaded', '1')]:
    directory = root/f'qwen-{arm}'
    directory.mkdir()
    command = [item.replace(previous,str(source))
               .replace(str(old/'four-qwen-f0-n1-disabled'),str(directory))
               .replace(str(old/'scripts'),str(root/'scripts'))
               .replace(str(old/'cache/four-qwen-f0-n1'),str(root/'cache/qwen'))
               .replace('megatron-offload-review-four-20261021','megatron-offload-scope-20261022')
               for item in reference]
    if fraction is not None:
        command += ['--fraction',fraction]
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
    manifest = dict(revision=revision,command=command,started_unix=time.time(),
                    dirty_before='',harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = directory/'manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Starting '+directory.name,flush=True)
    with (directory/'launcher.log').open('w') as log:
        result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    manifest.update(exit_code=result.returncode,finished_unix=time.time(),
                    revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                    dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
    path.write_text(json.dumps(manifest,indent=2)+'\n');assert result.returncode==0,directory
    assert manifest['revision_after']==revision and not manifest['dirty_after']
    outputs = [json.loads((directory/f'result.rank{rank}.json').read_text()) for rank in range(4)]
    for rank, output in enumerate(outputs):
        args = output['arguments']
        assert output['gpu_training_run'] and output['pool_verified_before_reset']
        assert output['rank']==rank and output['text_layers']==24
        assert args['tp']==2 and args['sp'] and args['pp']==2 and args['recompute_norm']
        assert not args['fused_pre_gdr'] and args['seq_length']==128
        assert [step['step'] for step in output['states']]==[0,1,2]
        assert (output['offload_status']['selected_transfer_bytes']>0)==(arm=='offloaded')
        assert all(step[field] for step in output['states'] for field in ('grads','weights','optimizer'))
    records.append([output['states'] for output in outputs])
comparison = dict(revision=revision,ranks=4,steps=3,arms=['disabled','repeat-disabled','fraction-zero','offloaded'],
                  complete_recorded_states_exact=all(record==records[0] for record in records[1:]))
(root/'qwen-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
assert comparison['complete_recorded_states_exact'],comparison
print(json.dumps(comparison),flush=True)
