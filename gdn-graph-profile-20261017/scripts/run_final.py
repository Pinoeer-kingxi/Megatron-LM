"""Qualify the signed GDN2 live-gate change with existing training and graph tests."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0,'/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/gdn-followup-20261016/scripts')
from run_gpu_acceptance import ARTIFACTS,IMAGE,VENV,bridge_state,container_command,gpu_inventory,python_path,require_idle

ROOT=Path(__file__).resolve().parents[1]
NAME='megatron-gdn2-final-20261017'
NSYS=Path('/opt/nvidia/nsight-systems/2025.3.2')
args=SimpleNamespace(source_root=ARTIFACTS/'split-pr-20261015',output=ROOT,stage='four',weights=ARTIFACTS/'models/Qwen3.5-0.8B',data=ARTIFACTS/'training-data/ks-item-sft.jsonl')
gpus=gpu_inventory(['1','2'])
require_idle(gpus)
manifest=dict(status='running',source=json.loads((ROOT/'final-source-identity.json').read_text()),bridge=bridge_state(args.source_root),gpus=gpus,jobs=[],started_unix=time.time())

def save():
    (ROOT/'final-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

stop=threading.Event()
def telemetry():
    with (ROOT/'final-telemetry.csv').open('w') as out:
        while not stop.is_set():
            subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu','--format=csv,noheader'],stdout=out,check=True)
            out.flush()
            stop.wait(1)


def launch(label,source,job,ranks=1,profile=False,expected_exit=0):
    require_idle(gpus[:ranks])
    out=ROOT/label
    out.mkdir()
    cache=ROOT/('cache/distributed' if ranks>1 else 'cache/gdn2')
    cmd=['docker','exec','--workdir',str(source),'--env','PYTHONPATH='+python_path(source,args.source_root),'--env','PYTHONDONTWRITEBYTECODE=1','--env','NVTE_ALLOW_NONDETERMINISTIC_ALGO=0','--env','CUDA_VISIBLE_DEVICES='+','.join(gpu['uuid'] for gpu in gpus[:ranks]),'--env','TRITON_CACHE_DIR='+str(cache/'triton'),'--env','TORCHINDUCTOR_CACHE_DIR='+str(cache/'inductor'),NAME]
    if profile:
        cmd += [str(NSYS/'bin/nsys'),'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--export=sqlite','--output='+str(out/'trace')]
    if ranks>1:
        job=[str(ROOT/'scripts/rank_env.py')]+job
    cmd += [str(VENV/'bin/python'),'-m','torch.distributed.run','--standalone',f'--nproc-per-node={ranks}','--log-dir',str(out/'ranks'),'--redirects','3','--tee','3']+job
    item=dict(label=label,source=str(source),command=cmd,started_unix=time.time(),expected_exit=expected_exit,benchmark_script_sha256=hashlib.sha256((ROOT/'scripts/benchmark_graph_training.py').read_bytes()).hexdigest())
    manifest['jobs'].append(item)
    save()
    print('Starting '+label,flush=True)
    with (out/'launcher.log').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    item.update(exit_code=result.returncode,finished_unix=time.time())
    save()
    if result.returncode!=expected_exit:
        raise RuntimeError(f'Inspect {out}/launcher.log')
    print('Finished '+label,flush=True)


def pytest_job(label,file,selection):
    return [str(ROOT/'scripts/rank_pytest.py'),str(ROOT/label/'reports'),'-q','-x','-p','no:cacheprovider','--confcutdir=tests/unit_tests/ssm',file,'-k',selection]


def training_job(label,extra):
    return [str(ROOT/'scripts/benchmark_graph_training.py'),'--model','gdn2','--weights',str(args.weights),'--data',str(args.data),'--output',str(ROOT/label/'result.json')]+extra


container=None
monitor=threading.Thread(target=telemetry)
try:
    cmd=container_command(args,gpus,NAME)
    cmd[cmd.index(IMAGE):cmd.index(IMAGE)]=['--mount',f'type=bind,source={NSYS},target={NSYS},readonly']
    manifest['container_command']=cmd
    container=subprocess.check_output(cmd,text=True).strip()
    manifest['container_id']=container
    monitor.start()
    final=ROOT/'final-source'
    label='regression-red-control'
    launch(label,ROOT/'red-source',pytest_job(label,'tests/unit_tests/ssm/test_gated_delta_net_offloading.py','test_gdn2_offload_retains_live_sigmoid_gates and False and 4'),expected_exit=1)
    for label,file,selection in (
        ('final-gdn2-scopes','tests/unit_tests/ssm/test_gated_delta_net_offloading.py','gdn2'),
        ('final-gdn2-native','tests/unit_tests/ssm/test_gdn_offload_training.py','gdn2 and (dp or tp_sp or pp) and not (vpp or cp or tp_sp_pp)'),
        ('final-gdn2-graphs','tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py','gdn2'),
    ):
        launch(label,final,pytest_job(label,file,selection),ranks=2)
    for heads in (8,16):
        for enabled in (False,True):
            label=f'final-state-k{heads}-'+('enabled' if enabled else 'disabled')
            extra=['--check-state','--warmup','1','--iterations','2','--gdn2-key-heads',str(heads)]
            if enabled:extra+=['--offload']
            launch(label,final,training_job(label,extra))
    for repeat in range(3):
        for arm in (('previous','final') if repeat!=1 else ('final','previous')):
            label=f'gdn2-paired-{repeat}-{arm}'
            source=ROOT/('source' if arm=='previous' else 'final-source')
            launch(label,source,training_job(label,['--offload']))
    label='gdn2-final-profile'
    launch(label,final,training_job(label,['--offload','--profile','--iterations','3']),profile=True)
    manifest['status']='completed_pending_verification'
finally:
    stop.set()
    if monitor.is_alive():monitor.join()
    if container is not None:
        cleanup=subprocess.run(['docker','stop','--time','10',container],capture_output=True,text=True)
        manifest['cleanup']=dict(exit_code=cleanup.returncode,stdout=cleanup.stdout,stderr=cleanup.stderr)
    manifest['finished_unix']=time.time()
    save()
