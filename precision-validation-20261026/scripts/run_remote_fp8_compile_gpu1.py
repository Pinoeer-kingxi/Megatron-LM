"""Find the first real forward/backward divergence across independently hashed interpreters."""
import json
import os
import subprocess
import time
from pathlib import Path

base=Path('/home/lwx/.cache/megatron-gdn-20261026')
source=base/'precision-context-test'
revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
for label,seed in (('seed0',0),('seed1',1),('seed0-repeat',0)):
    directory=base/f'qwen-sm120-fp8-compile-gpu1-{label}'
    directory.mkdir()
    command=['/usr/bin/python3',str(base/'remote_container_gpu1_inductor.py'),'/home/songbinbin/.venvs/megatron-gdn-offload/bin/python',
       '-m','torch.distributed.run','--standalone','--nproc-per-node=1','--log-dir',str(directory/'ranks'),'--redirects','3',
       str(base/'rank_env.py'),str(source/'tools/ssm/qwen_gdn_offload_training.py'),
       '--weights',str(base/'models/Qwen3.5-0.8B'),'--data',str(base/'data/ks-item-sft.jsonl'),
       '--precision','fp8-delayed','--seq-length','128','--microbatches','4','--warmup','1','--iterations','2',
       '--min-offloaded-tensor-size','1024','--recompute-norm','--check-state','--output',str(directory/'result.json')]
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
    manifest=dict(revision=revision,command=command,started_unix=time.time(),pythonhashseed=str(seed),inductor_cache=str(base/f'cache/inductor-gpu1-{label}'),
                  gpu_uuid='GPU-5ceca9be-9cf0-99be-0868-915db7531fae')
    with (directory/'launcher.log').open('w') as log:
        result=subprocess.run(command,env=dict(os.environ,GDN_SOURCE=str(source),PYTHONHASHSEED=str(seed),GDN_INDUCTOR_CACHE_DIR=str(base/f'cache/inductor-gpu1-{label}')),stdout=log,stderr=subprocess.STDOUT)
    manifest.update(exit_code=result.returncode,finished_unix=time.time(),revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert manifest['revision_after']==revision and not manifest['dirty_after'] and result.returncode==0,manifest
    print('Finished trace seed '+str(seed),flush=True)
