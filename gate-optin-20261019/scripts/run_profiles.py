import importlib.util,json,subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parents[1];HELPERS=Path('/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/gdn-followup-20261016/scripts');sys.path.insert(0,str(HELPERS))
spec=importlib.util.spec_from_file_location('gpu_helper',HELPERS/'run_gpu_acceptance.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
from types import SimpleNamespace
args=SimpleNamespace(source_root=Path('/home/songbinbin/gdn-offload-artifacts/split-pr-20261015'),output=OUT,stage='single')
NSYS=Path('/opt/nvidia/nsight-systems/2025.3.2');name='megatron-gate-profile-20261019';gpus=h.gpu_inventory(['1']);h.require_idle(gpus)
cmd=h.container_command(args,gpus,name);cmd[cmd.index(h.IMAGE):cmd.index(h.IMAGE)]=['--mount',f'type=bind,source={NSYS},target={NSYS},readonly','--env','PYTHONDONTWRITEBYTECODE=1']
manifest={'source':json.loads((OUT/'source-identity.json').read_text())['source'],'container_command':cmd,'jobs':[],'status':'running'}
p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==0,p.stderr
try:
 for arm in ('baseline','candidate'):
  h.require_idle(gpus);source=OUT/('upstream' if arm=='baseline' else 'source');directory=OUT/('profile-'+arm);directory.mkdir()
  command=['docker','exec','--workdir',str(source),'--env','PYTHONPATH='+h.python_path(source,args.source_root),name,str(NSYS/'bin/nsys'),'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--export=sqlite','--output='+str(directory/'trace'),str(h.VENV/'bin/python'),'-B','-m','torch.distributed.run','--standalone','--nproc-per-node=1','--log-dir',str(directory/'ranks'),'--redirects','3','--tee','3',str(OUT/'scripts/qwen_profile_driver.py'),'--table-path',str(OUT/'tables'),'--weights','/home/songbinbin/gdn-offload-artifacts/models/Qwen3.5-0.8B','--data','/home/songbinbin/gdn-offload-artifacts/training-data/ks-item-sft.jsonl','--seq-length','2048','--microbatches','4','--fused-pre-gdr','--warmup','3','--iterations','4','--output',str(directory/'result.json')]
  row={'arm':arm,'command':command,'started':time.time()};manifest['jobs'].append(row);(OUT/'profile-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('START profile',arm,flush=True)
  with (directory/'launcher.log').open('w') as log:p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
  row.update(exit_code=p.returncode,finished=time.time());(OUT/'profile-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');assert p.returncode==0
  print('DONE profile',arm,flush=True)
 manifest['status']='completed_pending_analysis'
finally:
 p=subprocess.run(['docker','stop',name],capture_output=True,text=True);manifest['cleanup']={'exit_code':p.returncode};(OUT/'profile-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
