import hashlib,json,os,subprocess,time
from pathlib import Path
b=Path('/home/lwx/.cache/megatron-gdn-20261026');s=b/'precision-with-mrope-dfaa633';rev='dfaa6338c65edab118ba415fcc14c73241f503a6';all_data=[]
for arm in ('eager','graph-disabled','graph-repeat','graph-offloaded'):
 out=b/f'qwen-fp4-mrope-dfaa633-{arm}-gpu0';out.mkdir(exist_ok=False)
 assert subprocess.check_output(['git','-C',str(s),'rev-parse','HEAD'],text=True).strip()==rev and not subprocess.check_output(['git','-C',str(s),'status','--porcelain'],text=True)
 cmd=['/usr/bin/python3',str(b/'precision_graph_container_gpu0_default.py'),'/home/songbinbin/.venvs/megatron-gdn-offload/bin/python','-m','torch.distributed.run','--standalone','--nproc-per-node=1','--log-dir',str(out/'ranks'),'--redirects','3',str(b/'rank_env_precision_graph.py'),str(b/'verify_qwen_source.py'),str(b/'benchmark_current_20261027.py'),'--model','qwen','--precision','fp4','--weights',str(b/'models/Qwen3.5-0.8B'),'--data',str(b/'data/ks-item-sft.jsonl'),'--seq-length','128','--warmup','1','--iterations','2','--threshold','1024','--check-state','--output',str(out/'result.json')]
 if arm=='eager':cmd.append('--no-graph')
 if arm=='graph-offloaded':cmd.append('--offload')
 env=dict(os.environ,GDN_SOURCE=str(s),PYTHONHASHSEED='0',GDN_INDUCTOR_CACHE_DIR=str(b/'cache/qwen-final-fp4-inductor'),GDN_TRITON_CACHE_DIR=str(b/'cache/qwen-final-fp4-triton'))
 m={'revision':rev,'arm':arm,'command':cmd,'gpu_uuid':'GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a','disable_rht':False,'disable_stochastic_rounding':False,'script_sha256':hashlib.sha256((b/'benchmark_current_20261027.py').read_bytes()).hexdigest(),'started_unix':time.time()}
 with (out/'launcher.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 m.update(exit_code=p.returncode,finished_unix=time.time(),revision_after=subprocess.check_output(['git','-C',str(s),'rev-parse','HEAD'],text=True).strip(),dirty_after=subprocess.check_output(['git','-C',str(s),'status','--porcelain'],text=True));(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(arm,p.returncode,flush=True)
 if p.returncode or m['revision_after']!=rev or m['dirty_after']:raise SystemExit(p.returncode or 1)
 d=json.loads((out/'result.json').read_text());assert len(d['records'])==6 and len(d['graph_counts'])==(0 if arm=='eager' else 24);assert all(x['selected_bytes']>0 for x in d['records'][3:])==(arm=='graph-offloaded');all_data.append(d)
keys=('input_sha256','loss','grad_norm','gradients','weights','optimizer');summary={}
for arm,d in zip(('graph-disabled','graph-repeat','graph-offloaded'),all_data[1:]):
 summary[arm]={k:[x[k]==y[k] for x,y in zip(all_data[0]['records'],d['records'])] for k in keys}
(b/'qwen-fp4-mrope-dfaa633-comparison.json').write_text(json.dumps({'revision':rev,'comparison_to_eager':summary},indent=2)+'\n');print(json.dumps(summary),flush=True)
