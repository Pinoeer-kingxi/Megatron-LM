"""Test the existing size filter and inspect saved-storage lifetimes."""
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, '/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/gdn-followup-20261016/scripts')
from run_gpu_acceptance import ARTIFACTS, VENV, container_command, gpu_inventory, python_path, require_idle

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source'
NAME='megatron-gdn2-equal-heads-profile-20261017'
args=SimpleNamespace(source_root=ARTIFACTS/'split-pr-20261015',output=ROOT,stage='four',weights=ARTIFACTS/'models/Qwen3.5-0.8B',data=ARTIFACTS/'training-data/ks-item-sft.jsonl')
gpus=gpu_inventory(['1'])
require_idle(gpus)
manifest=dict(status='running',source=json.loads((ROOT/'source-identity.json').read_text()),gpus=gpus,jobs=[],started_unix=time.time())

def save():
    (ROOT/'equal-heads-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

container=None
try:
    cmd=container_command(args,gpus,NAME)
    manifest['container_command']=cmd
    container=subprocess.check_output(cmd,text=True).strip()
    manifest['container_id']=container
    for model in ('lifetimes',):
        label=model+'-threshold4m' if model!='lifetimes' else 'gdn2-saved-tensors-equal-heads'
        out=ROOT/label
        out.mkdir()
        cache=ROOT/'cache'/('gdn2' if model=='lifetimes' else model)
        cmd=['docker','exec','--workdir',str(SOURCE),'--env','PYTHONPATH='+python_path(SOURCE,args.source_root),'--env','PYTHONDONTWRITEBYTECODE=1','--env','TRITON_CACHE_DIR='+str(cache/'triton'),'--env','TORCHINDUCTOR_CACHE_DIR='+str(cache/'inductor'),NAME,str(VENV/'bin/python'),'-m','torch.distributed.run','--standalone','--nproc-per-node=1','--log-dir',str(out/'ranks'),'--redirects','3','--tee','3']
        if model=='lifetimes':
            cmd += ['tools/ssm/gdn_activation_offload.py','--mode','profile','--variant','gdn2','--layers','1','--hidden-size','1024','--key-heads','16','--value-heads','16','--head-dim','128','--seq-length','2048','--output',str(out/'result.json')]
        else:
            cmd += [str(ROOT/'scripts/benchmark_graph_training.py'),'--model',model,'--weights',str(args.weights),'--data',str(args.data),'--output',str(out/'result.json'),'--offload','--min-offloaded-tensor-size','4194304']
        job=dict(label=label,command=cmd,started_unix=time.time())
        manifest['jobs'].append(job)
        save()
        print('Starting '+label,flush=True)
        with (out/'launcher.log').open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        job.update(exit_code=result.returncode,finished_unix=time.time())
        save()
        if result.returncode:
            raise RuntimeError(f'Inspect {out}/launcher.log')
        print('Finished '+label,flush=True)
    manifest['status']='completed_pending_analysis'
finally:
    if container is not None:
        cleanup=subprocess.run(['docker','stop','--time','10',container],capture_output=True,text=True)
        manifest['cleanup']=dict(exit_code=cleanup.returncode,stdout=cleanup.stdout,stderr=cleanup.stderr)
    manifest['finished_unix']=time.time()
    save()
