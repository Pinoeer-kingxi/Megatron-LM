"""Compare complete native Qwen training with persistent rank-owned compiler caches."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

base=Path('/home/lwx/.cache/megatron-gdn-20261026')
source=base/'precision-stable'
revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
assert revision=='688159928298844dc00d9d3a9ce33774735d997c'
for precision in ('fp8-delayed','fp4','fp8-tensorwise'):
    outputs=[]
    cache=base/f'cache/final-inductor-{precision}'
    def snapshot():
        return {str(p.relative_to(cache)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(cache.rglob('*.best_config'))}
    for arm in ('disabled','repeated-disabled','offloaded'):
        directory=base/f'qwen-sm120-stable-{precision}-{arm}'
        directory.mkdir()
        command=['/usr/bin/python3',str(base/'remote_container_gpu1_inductor.py'),
           '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python','-m','torch.distributed.run',
           '--standalone','--nproc-per-node=1','--log-dir',str(directory/'ranks'),'--redirects','3',
           str(base/'rank_env.py'),str(base/'observe_qwen_runtime.py'),str(directory/'runtime.json'),
           'NVFP4' if precision=='fp4' else 'Float8',str(source/'tools/ssm/qwen_gdn_offload_training.py'),
           '--weights',str(base/'models/Qwen3.5-0.8B'),'--data',str(base/'data/ks-item-sft.jsonl'),
           '--precision',precision,'--seq-length','128','--microbatches','4','--warmup','1','--iterations','2',
           '--min-offloaded-tensor-size','1024','--recompute-norm','--check-state','--output',str(directory/'result.json')]
        if arm=='offloaded': command+=['--fraction','1']
        assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
        manifest=dict(revision=revision,command=command,started_unix=time.time(),pythonhashseed='0',
                      inductor_cache=str(cache),cache_before=snapshot(),
                      gpu_uuid='GPU-5ceca9be-9cf0-99be-0868-915db7531fae',
                      nvfp4_disable_rht=True,nvfp4_disable_stochastic_rounding=True)
        with (directory/'launcher.log').open('w') as log:
            result=subprocess.run(command,env=dict(os.environ,GDN_SOURCE=str(source),PYTHONHASHSEED='0',GDN_INDUCTOR_CACHE_DIR=str(cache)),stdout=log,stderr=subprocess.STDOUT)
        manifest.update(exit_code=result.returncode,finished_unix=time.time(),cache_after=snapshot(),
                        revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                        dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
        (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        assert result.returncode==0 and revision==manifest['revision_after'] and not manifest['dirty_after'],manifest
        output=json.loads((directory/'result.rank0.json').read_text())
        assert output['text_layers']==24 and len(output['states'])==3
        assert bool(output['offload_status']['selected_transfer_bytes'])==(arm=='offloaded')
        outputs.append(output['states'])
        print('Finished '+precision+' '+arm,flush=True)
    pairs={f'{a} vs {b}':outputs[i]==outputs[j] for i,a in enumerate(('disabled','repeated-disabled','offloaded')) for j,b in enumerate(('disabled','repeated-disabled','offloaded')) if i<j}
    comparison=dict(revision=revision,exact_states=all(pairs.values()),pairs=pairs)
    (base/f'qwen-sm120-stable-{precision}-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(comparison),flush=True)
    assert comparison['exact_states']
