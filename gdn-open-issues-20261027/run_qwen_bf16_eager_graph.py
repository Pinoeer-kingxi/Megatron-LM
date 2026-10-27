import hashlib,json,os,subprocess,time
from pathlib import Path
b=Path('/home/lwx/.cache/megatron-gdn-20261026');s=b/'precision-final-aecc0e';rev='aecc0e8116912554f27a62a51ad2aeca479a7f9e'
for arm,default,cached,nograph in [('eager',True,False,True),('graph',True,False,False)]:
 out=b/f'qwen-bf16-{arm}-20261027-gpu1';out.mkdir(exist_ok=False)
 assert subprocess.check_output(['git','-C',str(s),'rev-parse','HEAD'],text=True).strip()==rev
 wrapper='precision_graph_container_gpu1_default.py'
 cmd=['/usr/bin/python3',str(b/wrapper),'/home/songbinbin/.venvs/megatron-gdn-offload/bin/python','-m','torch.distributed.run','--standalone','--nproc-per-node=1','--log-dir',str(out/'ranks'),'--redirects','3',str(b/'rank_env_precision_graph.py'),str(b/'verify_qwen_source.py'),str(b/'benchmark_nvfp4_controls.py'),'--model','qwen','--precision','bf16','--weights',str(b/'models/Qwen3.5-0.8B'),'--data',str(b/'data/ks-item-sft.jsonl'),'--seq-length','128','--warmup','1','--iterations','2','--threshold','1024','--check-state','--output',str(out/'result.json')]
 if cached:cmd.append('--cached')
 if nograph:cmd.append('--no-graph')
 env=dict(os.environ,GDN_SOURCE=str(s),PYTHONHASHSEED='0',GDN_INDUCTOR_CACHE_DIR=str(b/'cache/qwen-default-fp4-inductor'),GDN_TRITON_CACHE_DIR=str(b/'cache/qwen-default-fp4-triton'))
 m=dict(revision=rev,arm=arm,command=cmd,started_unix=time.time(),gpu_uuid='GPU-5ceca9be-9cf0-99be-0868-915db7531fae',rht=default,stochastic_rounding=default,cached=cached,graph=not nograph,script_sha256=hashlib.sha256((b/'benchmark_nvfp4_controls.py').read_bytes()).hexdigest())
 with (out/'launcher.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 m.update(exit_code=p.returncode,finished_unix=time.time(),revision_after=subprocess.check_output(['git','-C',str(s),'rev-parse','HEAD'],text=True).strip(),dirty_after=subprocess.check_output(['git','-C',str(s),'status','--porcelain'],text=True))
 (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(arm,'exit',p.returncode,flush=True)
 if arm.startswith('eager') and p.returncode: raise SystemExit(p.returncode)
