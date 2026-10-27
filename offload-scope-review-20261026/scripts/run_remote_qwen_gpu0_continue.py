"""Run fresh-process complete Qwen checks on the single reserved SM120 GPU."""
import json
import subprocess
import time
from pathlib import Path

base=Path('/home/lwx/.cache/megatron-gdn-20261026')
source=base/'precision'
python='/home/songbinbin/.venvs/megatron-gdn-offload/bin/python'
revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
all_exact = True
for precision in ('fp8-delayed','fp8-tensorwise','fp4'):
    states=[]
    arms = ('disabled','repeated-disabled','offloaded','confirmed-disabled') if precision == 'fp8-delayed' else ('disabled','repeated-disabled','offloaded')
    for arm in arms:
        directory=base/f'qwen-sm120-gpu0-{precision}-{arm}'
        if (directory/'manifest.json').exists():
            manifest=json.loads((directory/'manifest.json').read_text())
            assert manifest['exit_code']==0 and manifest['revision']==revision and manifest['revision_after']==revision and not manifest['dirty_after']
            output=json.loads((directory/'result.rank0.json').read_text())
            assert output['text_layers']==24 and len(output['states'])==3
            states.append(output['states'])
            continue
        directory.mkdir()
        command=['/usr/bin/python3',str(base/'remote_container_gpu0.py'),python,'-m','torch.distributed.run',
            '--standalone','--nproc-per-node=1','--log-dir',str(directory/'ranks'),'--redirects','3',
            str(base/'rank_env.py'),str(source/'tools/ssm/qwen_gdn_offload_training.py'),
            '--weights',str(base/'models/Qwen3.5-0.8B'),'--data',str(base/'data/ks-item-sft.jsonl'),
            '--precision',precision,'--seq-length','128','--microbatches','4',
            '--warmup','1','--iterations','2','--min-offloaded-tensor-size','1024',
            '--recompute-norm','--check-state','--output',str(directory/'result.json')]
        if arm=='offloaded': command+=['--fraction','1']
        manifest=dict(revision=revision,command=command,started_unix=time.time(), gpu_uuid="GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a", nvfp4_disable_rht=True, nvfp4_disable_stochastic_rounding=True)
        assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
        with (directory/'launcher.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        manifest.update(exit_code=result.returncode,finished_unix=time.time(),
            revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
            dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
        (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        assert result.returncode==0 and manifest['revision_after']==revision and not manifest['dirty_after'],directory
        output=json.loads((directory/'result.rank0.json').read_text())
        assert output['text_layers']==24 and len(output['states'])==3
        states.append(output['states'])
        assert bool(output['offload_status']['selected_transfer_bytes'])==(arm=='offloaded')
        print('Finished '+str(directory),flush=True)
    pairs={f'{arms[a]} vs {arms[b]}': states[a]==states[b] for a in range(len(states)) for b in range(a+1,len(states))}
    exact = all(pairs.values())
    all_exact = all_exact and exact
    comparison=dict(revision=revision,exact_states=exact,steps=3,pairs=pairs)
    (base/f'qwen-sm120-gpu0-{precision}-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(comparison),flush=True)
raise SystemExit(0 if all_exact else 1)
