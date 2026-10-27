"""Serial paired training benchmarks on one GPU; preserve every arm and process sample."""
import ast, hashlib, json, math, os, re, statistics, subprocess, threading, time
from pathlib import Path

B=Path('/home/lwx/.cache/megatron-gdn-20261026')
R=B/'current-performance-20261027'
R.mkdir(exist_ok=True)
GPU='GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a'
PY='/home/songbinbin/.venvs/megatron-gdn-offload/bin/python'
configs={
 'eager': (B/'feature-merge-b6fa63','b6fa63ab8c2cb855f0368cd8455245c295cde234'),
 'graph': (B/'graph-with-mrope-c5264b6','c5264b66197040f4309edfdb22dc9bbaf62189b1')}
records=json.loads((R/'progress.json').read_text()) if (R/'progress.json').exists() else []
def gpu_processes():
 result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 return [x.strip() for x in result.stdout.splitlines() if x.startswith(GPU)]
def descendants(parent):
 parents={}
 for d in Path('/proc').iterdir():
  if d.name.isdigit():
   try: parents[int(d.name)]=int((d/'stat').read_text().split(') ')[1].split()[1])
   except (OSError,ValueError,IndexError): pass
 result={parent}
 while True:
  extra={pid for pid,ppid in parents.items() if ppid in result}-result
  if not extra:return result
  result.update(extra)
def monitor(proc,path,stop):
 with path.open('w') as f:
  while not stop.wait(1):
   metric=subprocess.run(['nvidia-smi','--id='+GPU,'--query-gpu=timestamp,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm','--format=csv,noheader,nounits'],capture_output=True,text=True)
   rows=gpu_processes(); owned=descendants(proc.pid)
   foreign=[row for row in rows if int(row.split(',')[1].strip()) not in owned]
   f.write(json.dumps({'time':time.time(),'metrics':metric.stdout.strip(),'processes':rows,'foreign':foreign})+'\n');f.flush()
def run(kind,model,enabled,repeat,check=False,fused=False,eager_ref=False):
 name=f'{kind}-{model}-f{int(fused)}-{("correct" if check else "pair"+str(repeat))}-{("eager" if eager_ref else ("on" if enabled else "off"))}'
 previous=[row for row in records if row['name']==name]
 assert len(previous)<=1
 if previous: return previous[0]
 source,rev=configs[kind]
 assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==rev
 assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
 assert not gpu_processes(),f'{GPU} already in use'
 out=R/name;out.mkdir()
 cmd=['/usr/bin/python3',str(B/'precision_graph_container_gpu0.py'),PY,'-m','torch.distributed.run','--standalone','--nproc-per-node=1',
      '--log-dir',str(out/'ranks'),'--redirects','3',str(B/'rank_env_precision_graph.py'),str(B/'verify_qwen_source.py')]
 if kind=='eager':
  cmd += [str(B/'qwen_resources_20261027.py'),'--weights',str(B/'models/Qwen3.5-0.8B'),'--data',str(B/'data/ks-item-sft.jsonl'),
          '--seq-length','4096','--microbatches','4','--warmup','3','--iterations','10','--output',str(out/'result.json')]
  if fused:cmd.append('--fused-pre-gdr')
  if enabled:cmd += ['--fraction','1']
 else:
  cmd += [str(B/'benchmark_graph_with_pool.py'),'--model',model,'--weights',str(B/'models/Qwen3.5-0.8B'),
          '--data',str(B/'data/ks-item-sft.jsonl'),'--output',str(out/'result.json')]
  if enabled:cmd.append('--offload')
  if eager_ref:cmd.append('--no-graph')
  if check:cmd += ['--check-state','--warmup','1','--iterations','2']
 env=dict(os.environ,GDN_SOURCE=str(source),PYTHONHASHSEED='0',GDN_INDUCTOR_CACHE_DIR=str(R/'cache'/kind/model/'inductor'),
          GDN_TRITON_CACHE_DIR=str(R/'cache'/kind/model/'triton'))
 m={'revision':rev,'gpu_uuid':GPU,'command':cmd,'started_unix':time.time(),'source':str(source),'check_state':check,
    'benchmark_sha256':hashlib.sha256((B/('qwen_resources_20261027.py' if kind=='eager' else 'benchmark_graph_with_pool.py')).read_bytes()).hexdigest()}
 with (out/'launcher.log').open('w') as log:
  p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  stop=threading.Event(); t=threading.Thread(target=monitor,args=(p,out/'telemetry.jsonl',stop));t.start()
  try:rc=p.wait()
  finally:stop.set();t.join()
 m.update(exit_code=rc,finished_unix=time.time(),revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
          dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
 telemetry=[json.loads(x) for x in (out/'telemetry.jsonl').read_text().splitlines()]
 m['foreign_samples']=[x for x in telemetry if x['foreign']]
 (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print(name,'exit',rc,'foreign',len(m['foreign_samples']),flush=True)
 assert rc==0 and m['revision_after']==rev and not m['dirty_after'] and not m['foreign_samples'], name
 if kind=='eager':
  data=json.loads((out/'result.rank0.json').read_text());samples=data['samples']; assert data['text_layers']==24 and len(samples)==10
  resource=json.loads((out/'resources.rank0.json').read_text())[-1];assert resource['stats']['current_in_use']==0
  assert bool(data['offload_status']['selected_transfer_bytes'])==enabled
  stderr,=out.rglob('0/stderr.log'); pattern=r'INFO:root:Rank 0 step (\d+): losses=(\[.*?\]) grad_norm=(.*?) selected_transfer_bytes=(\d+)'
  logged=[{'step':int(k),'loss':ast.literal_eval(loss),'norm':norm} for k,loss,norm,_ in re.findall(pattern,stderr.read_text())]
  assert len(logged)==13
  row={'name':name,'kind':kind,'model':model,'fused':fused,'enabled':enabled,'repeat':repeat,'logs':logged,'resource':resource,
       'median_ms':1000*statistics.median(x['seconds'] for x in samples),'allocated':max(x['peak_allocated_bytes'] for x in samples),
       'reserved':max(x['peak_reserved_bytes'] for x in samples)}
 else:
  data=json.loads((out/'result.json').read_text());steps=data['records'];assert len(data['graph_counts'])==(0 if eager_ref else (24 if model=='qwen' else 8))
  assert len(steps)==(6 if check else 16) and all(x['selected_bytes']>0 for x in steps[3:])==enabled
  row={'name':name,'kind':kind,'model':model,'fused':False,'enabled':enabled,'repeat':repeat,'check':check,
       'logs':[{'loss':x['loss'],'norm':x['grad_norm'],'input':x['input_sha256']} for x in steps]}
  if check:row['states']=[{k:x[k] for k in ('loss','grad_norm','gradients','weights','optimizer')} for x in steps]
  else:
   measured=[x for x in steps if x['measured']];assert len(measured)==10
   row.update(median_ms=1000*statistics.median(x['seconds'] for x in measured),allocated=max(x['peak_allocated_bytes'] for x in measured),reserved=max(x['peak_reserved_bytes'] for x in measured),pinned_pool=data['pinned_pool'])
 records.append(row);(R/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
 return row

def norm_value(value):
 match=re.fullmatch(r"tensor\(\[([-+0-9.eE]+)\], device='cuda:[0-9]+'\)", value)
 return float(match.group(1) if match else value)

for model in ('qwen','gdn2'):
 eager=run('graph',model,False,0,check=True,eager_ref=True)
 off=run('graph',model,False,0,check=True);on=run('graph',model,True,0,check=True)
 assert eager['states']==off['states']==on['states'],model
for kind,model,fused in [('eager','qwen',False),('eager','qwen',True),('graph','qwen',False),('graph','gdn2',False)]:
 for pair in range(5 if model=='qwen' and not fused else 3):
  rows=[run(kind,model,on,pair,fused=fused) for on in ((False,True) if pair%2==0 else (True,False))]
  if fused:
   for a, b in zip(rows[0]['logs'], rows[1]['logs']):
    assert a['step']==b['step'] and all(math.isclose(x,y,rel_tol=1e-5,abs_tol=1e-8) for x,y in zip(a['loss'],b['loss']))
    assert math.isclose(norm_value(a['norm']),norm_value(b['norm']),rel_tol=1e-5,abs_tol=1e-4)
  else:
   assert rows[0]['logs']==rows[1]['logs'],(kind,model,fused,pair)
print('all matched benchmark arms complete',flush=True)
