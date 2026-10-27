"""Run fresh-process complete Qwen checks on the single reserved SM120 GPU."""
import json
import subprocess
import time
from pathlib import Path

base=Path('/home/lwx/.cache/megatron-gdn-20261026')
source=base/'precision'
python='/home/songbinbin/.venvs/megatron-gdn-offload/bin/python'
revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
for precision in ('fp8-delayed','fp8-tensorwise','fp4'):
    states=[]
    for arm in ('disabled','repeated-disabled','offloaded'):
        directory=base/f'qwen-sm120-gpu0-{precision}-{arm}'
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
    assert states[0]==states[1]==states[2],precision
    (base/f'qwen-sm120-gpu0-{precision}-comparison.json').write_text(json.dumps(dict(revision=revision,exact_states=True,steps=3),indent=2)+'\n')
