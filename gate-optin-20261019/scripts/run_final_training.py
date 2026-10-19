import csv, hashlib, importlib.util, json, subprocess, sys, time
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parents[1]
HELPERS=Path('/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/gdn-followup-20261016/scripts');sys.path.insert(0,str(HELPERS))
spec=importlib.util.spec_from_file_location('gpu_helper',HELPERS/'run_gpu_acceptance.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
CONTAINER='megatron-gate-optin-dev-20261019';SOURCE_ROOT=Path('/home/songbinbin/gdn-offload-artifacts/split-pr-20261015')
manifest={'source':json.loads((OUT/'final-source-identity.json').read_text())['source'],'status':'running','jobs':[],'note':'Final source after local-only type annotation; same recorded launch table.'}
weights=Path('/home/songbinbin/gdn-offload-artifacts/models/Qwen3.5-0.8B');data=Path('/home/songbinbin/gdn-offload-artifacts/training-data/ks-item-sft.jsonl')

def run(name,source,options,record=False,check=False):
 h.require_idle(h.gpu_inventory(['1']))
 directory=OUT/name;directory.mkdir(exist_ok=False)
 cmd=['docker','exec','--workdir',str(source),'--env','PYTHONPATH='+h.python_path(source,SOURCE_ROOT),CONTAINER,str(h.VENV/'bin/python'),'-B','-m','torch.distributed.run','--standalone','--nproc-per-node=1','--log-dir',str(directory/'ranks'),'--redirects','3','--tee','3',str(OUT/'scripts/qwen_driver.py')]
 cmd+=['--record-configs',str(OUT/'calibration/configs')] if record else ['--table-path',str(OUT/'tables')]
 cmd+=['--weights',str(weights),'--data',str(data),'--seq-length','2048','--microbatches','4','--fused-pre-gdr','--output',str(directory/'result.json'),*options]
 if check:cmd+=['--check-state']
 row={'name':name,'source_path':str(source),'command':cmd,'started':datetime.now(timezone.utc).isoformat()};manifest['jobs'].append(row)
 (OUT/'final-training-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('START',name,flush=True)
 samples=[];owned_seen=set()
 with (directory/'launcher.log').open('w') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  while p.poll() is None:
   owned_seen.update(x.strip() for x in subprocess.check_output(['docker','top',CONTAINER,'-eo','pid'],text=True).splitlines()[1:])
   devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader'],text=True)
   procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
   owned=subprocess.check_output(['docker','top',CONTAINER,'-eo','pid'],text=True).splitlines()[1:];owned_seen.update(s.strip() for s in owned)
   external=[r for r in csv.reader(procs.splitlines(),skipinitialspace=True) if r[0]=='GPU-5e2e1f94-e88e-48e3-247d-328b52fc68db' and r[1] not in owned_seen]
   samples.append({'time':datetime.now(timezone.utc).isoformat(),'gpu':devices,'external':external,'gpu_processes':list(csv.reader(procs.splitlines(),skipinitialspace=True)),'owned_pids':sorted(owned_seen)});time.sleep(2)
 row['exit_code']=p.returncode;row['finished']=datetime.now(timezone.utc).isoformat();row['external_overlap']=any(x['external'] for x in samples)
 (directory/'telemetry.json').write_text(json.dumps(samples,indent=2)+'\n');(OUT/'final-training-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('DONE',name,p.returncode,flush=True)
 assert p.returncode==0 and not row['external_overlap'],row

run('final-state-candidate',OUT/'source-final',['--warmup','1','--iterations','2'],check=True)
for pair in range(1):
 for arm in (('baseline','candidate') if pair%2==0 else ('candidate','baseline')):
  run(f'final-pair-{pair}-{arm}',OUT/('upstream' if arm=='baseline' else 'source-final'),['--warmup','5','--iterations','50'])
manifest['status']='completed_pending_verification';(OUT/'final-training-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
