import hashlib,json,os,subprocess,time
from pathlib import Path
b=Path('/home/lwx/.cache/megatron-gdn-20261026');rev='aecc0e8116912554f27a62a51ad2aeca479a7f9e'
for arm in ('baseline','fixed'):
 s=b/('precision-final-aecc0e' if arm=='baseline' else 'precision-alignment-prototype')
 out=b/f'alignment-tp2-{arm}-verbose-20261027-gpu03';out.mkdir(exist_ok=False)
 assert subprocess.check_output(['git','-C',str(s),'rev-parse','HEAD'],text=True).strip()==rev
 cmd=['/usr/bin/python3',str(b/'precision_graph_container_gpu03_apex.py'),'/home/songbinbin/.venvs/megatron-gdn-offload/bin/python','-m','torch.distributed.run','--standalone','--nproc-per-node=2','--log-dir',str(out/'ranks'),'--redirects','3',str(b/'rank_env_precision_graph.py'),str(b/'rank_distributed_pytest.py'),str(out/'reports'),'-q','-s','-x','-p','no:cacheprovider','--confcutdir='+str(b/'probes'),str(b/'probes/test_fp4_ordinary_optimizer.py'),'-k','gdn2-False-True-True-True-1.0-fp4-True-tp2-sp']
 env=dict(os.environ,GDN_SOURCE=str(s),PYTHONHASHSEED='0',GDN_INDUCTOR_CACHE_DIR=str(b/'cache/align-tp2-inductor'),GDN_TRITON_CACHE_DIR=str(b/'cache/align-tp2-triton'))
 m={'base':rev,'arm':arm,'command':cmd,'patch_sha256':hashlib.sha256((b/'nvfp4-ddp-alignment-prototype.patch').read_bytes()).hexdigest() if arm=='fixed' else None,'test_sha256':hashlib.sha256((b/'probes/test_fp4_ordinary_optimizer.py').read_bytes()).hexdigest(),'started_unix':time.time(),'gpu_uuids':['GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a','GPU-dfee170b-6263-1438-4231-9e55c7f1c24d'],'disable_rht':True,'disable_stochastic_rounding':True}
 with (out/'launcher.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 m.update(exit_code=p.returncode,finished_unix=time.time(),revision_after=subprocess.check_output(['git','-C',str(s),'rev-parse','HEAD'],text=True).strip());(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print(arm,'exit',p.returncode,flush=True)
 if arm=='fixed' and p.returncode:raise SystemExit(p.returncode)
