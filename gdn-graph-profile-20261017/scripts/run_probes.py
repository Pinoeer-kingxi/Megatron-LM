"""Compare existing configurations after transfer bottleneck identification."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, '/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/gdn-followup-20261016/scripts')
from run_gpu_acceptance import ARTIFACTS, VENV, bridge_state, container_command, gpu_inventory, python_path, require_idle

ROOT=Path(__file__).resolve().parents[1]
NAME='megatron-graph-probes-20261017'
SOURCE=ROOT/'source'
args=SimpleNamespace(source_root=ARTIFACTS/'split-pr-20261015',output=ROOT,stage='four',weights=ARTIFACTS/'models/Qwen3.5-0.8B',data=ARTIFACTS/'training-data/ks-item-sft.jsonl')
gpus=gpu_inventory(['1'])
require_idle(gpus)
manifest=dict(status='running',source=json.loads((ROOT/'source-identity.json').read_text()),bridge=bridge_state(args.source_root),gpus=gpus,jobs=[],started_unix=time.time(),profiler=False)

def save():
    (ROOT/'probe-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

stop=threading.Event()
def telemetry():
    with (ROOT/'probe-telemetry.csv').open('w') as out:
        while not stop.is_set():
            subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu','--format=csv,noheader'],stdout=out,check=True)
            out.flush()
            stop.wait(1)

container=None
monitor=threading.Thread(target=telemetry)
try:
    start=container_command(args,gpus,NAME)
    manifest['container_command']=start
    container=subprocess.check_output(start,text=True).strip()
    manifest['container_id']=container
    monitor.start()
    for model in ('gdn2','qwen'):
        for arm,selection,copy_connections in (
            ('disabled',[],None),
            ('full',['--offload'],None),
            ('full-copy8',['--offload'],'8'),
            ('half',['--fraction','0.5'],None),
        ):
            label=model+'-probe-'+arm
            out=ROOT/label
            out.mkdir()
            cache=ROOT/'cache'/model
            cmd=['docker','exec','--workdir',str(SOURCE),'--env','PYTHONPATH='+python_path(SOURCE,args.source_root),'--env','PYTHONDONTWRITEBYTECODE=1','--env','TRITON_CACHE_DIR='+str(cache/'triton'),'--env','TORCHINDUCTOR_CACHE_DIR='+str(cache/'inductor')]
            if copy_connections:
                cmd+=['--env','CUDA_DEVICE_MAX_COPY_CONNECTIONS='+copy_connections]
            cmd += [NAME,str(VENV/'bin/python'),'-m','torch.distributed.run','--standalone','--nproc-per-node=1','--log-dir',str(out/'ranks'),'--redirects','3','--tee','3',str(ROOT/'scripts/benchmark_graph_training.py'),'--model',model,'--weights',str(args.weights),'--data',str(args.data),'--output',str(out/'result.json')]+selection
            job=dict(label=label,command=cmd,started_unix=time.time(),script_sha256=hashlib.sha256((ROOT/'scripts/benchmark_graph_training.py').read_bytes()).hexdigest())
            manifest['jobs'].append(job)
            save()
            print('Starting '+label,flush=True)
            with (out/'launcher.log').open('w') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            job.update(exit_code=result.returncode,finished_unix=time.time())
            save()
            if result.returncode:
                raise RuntimeError(f'Inspect {out}/launcher.log')
            record=json.loads((out/'result.json').read_text())
            import statistics
            measured=[x for x in record['records'] if x['measured']]
            print('Finished '+label+' '+json.dumps(dict(median_ms=statistics.median(x['seconds'] for x in measured)*1000,selected_mib=measured[0]['selected_bytes']/2**20,reserved_mib=max(x['peak_reserved_bytes'] for x in measured)/2**20)),flush=True)
    manifest['status']='completed_pending_analysis'
finally:
    stop.set()
    if monitor.is_alive():
        monitor.join()
    if container is not None:
        cleanup=subprocess.run(['docker','stop','--time','10',container],capture_output=True,text=True)
        manifest['cleanup']=dict(exit_code=cleanup.returncode,stdout=cleanup.stdout,stderr=cleanup.stderr)
    manifest['finished_unix']=time.time()
    save()
