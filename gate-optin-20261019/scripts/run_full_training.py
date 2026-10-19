import csv, hashlib, importlib.util, json, subprocess, sys, time
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parents[1]
HELPERS=Path('/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/gdn-followup-20261016/scripts');sys.path.insert(0,str(HELPERS))
spec=importlib.util.spec_from_file_location('gpu_helper',HELPERS/'run_gpu_acceptance.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
CONTAINER='megatron-gate-optin-dev-20261019';SOURCE_ROOT=Path('/home/songbinbin/gdn-offload-artifacts/split-pr-20261015')
manifest={'source':json.loads((OUT/'source-identity.json').read_text())['source'],'status':'running','jobs':[]}
weights=Path('/home/songbinbin/gdn-offload-artifacts/models/Qwen3.5-0.8B');data=Path('/home/songbinbin/gdn-offload-artifacts/training-data/ks-item-sft.jsonl')

def run(name,source,options,record=False,check=False):
 h.require_idle(h.gpu_inventory(['1']))
 directory=OUT/name;directory.mkdir(exist_ok=False)
 cmd=['docker','exec','--workdir',str(source),'--env','PYTHONPATH='+h.python_path(source,SOURCE_ROOT),CONTAINER,str(h.VENV/'bin/python'),'-B','-m','torch.distributed.run','--standalone','--nproc-per-node=1','--log-dir',str(directory/'ranks'),'--redirects','3','--tee','3',str(OUT/'scripts/qwen_driver.py')]
 cmd+=['--record-configs',str(OUT/'calibration/configs')] if record else ['--table-path',str(OUT/'tables')]
 cmd+=['--weights',str(weights),'--data',str(data),'--seq-length','2048','--microbatches','4','--fused-pre-gdr','--output',str(directory/'result.json'),*options]
 if check:cmd+=['--check-state']
 row={'name':name,'source_path':str(source),'command':cmd,'started':datetime.now(timezone.utc).isoformat()};manifest['jobs'].append(row)
 (OUT/'full-training-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('START',name,flush=True)
 samples=[]
 with (directory/'launcher.log').open('w') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  while p.poll() is None:
   devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader'],text=True)
   procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
   owned=subprocess.check_output(['docker','top',CONTAINER,'-eo','pid'],text=True).splitlines()[1:];owned={s.strip() for s in owned}
   external=[r for r in csv.reader(procs.splitlines(),skipinitialspace=True) if r[0]=='GPU-5e2e1f94-e88e-48e3-247d-328b52fc68db' and r[1] not in owned]
   samples.append({'time':datetime.now(timezone.utc).isoformat(),'gpu':devices,'external':external});time.sleep(2)
 row['exit_code']=p.returncode;row['finished']=datetime.now(timezone.utc).isoformat();row['external_overlap']=any(x['external'] for x in samples)
 (directory/'telemetry.json').write_text(json.dumps(samples,indent=2)+'\n');(OUT/'full-training-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('DONE',name,p.returncode,flush=True)
 assert p.returncode==0 and not row['external_overlap'],row

assert json.loads((OUT/'test-manifest.json').read_text())['status']=='completed_pending_verification'
h.bridge_state(SOURCE_ROOT)
run('calibration',OUT/'upstream',['--warmup','1','--iterations','2'],record=True)
(OUT/'tables').mkdir()
cmd=['docker','exec','--workdir',str(OUT/'upstream'),'--env','PYTHONPATH='+h.python_path(OUT/'upstream',SOURCE_ROOT),CONTAINER,str(h.VENV/'bin/python'),'-B','-m','megatron.core.tuning','merge',str(OUT/'calibration/configs.rank0.json'),'-o',str(OUT/'tables/sm86.json'),'--source','PR8244 upstream Qwen default-path calibration']
p=subprocess.run(cmd,capture_output=True,text=True);(OUT/'table-merge.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
for pair in range(3):
 for arm in (('baseline','candidate') if pair%2==0 else ('candidate','baseline')):
  run(f'pair-{pair}-{arm}',OUT/('upstream' if arm=='baseline' else 'source'),['--warmup','3','--iterations','10'])
for arm in ('baseline','repeat','candidate'):
 run('state-'+arm,OUT/('source' if arm=='candidate' else 'upstream'),['--warmup','1','--iterations','2'],check=True)
manifest['status']='completed_pending_verification';(OUT/'full-training-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
