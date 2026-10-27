import hashlib,json,os,subprocess,time
from pathlib import Path
b=Path('/home/lwx/.cache/megatron-gdn-20261026');rev='aecc0e8116912554f27a62a51ad2aeca479a7f9e'
for arm in ('baseline',):
 s=b/('precision-final-aecc0e' if arm=='baseline' else 'precision-alignment-prototype');out=b/f'alignment-real-te-{arm}-final-gpu1';out.mkdir(exist_ok=False)
 cmd=['/usr/bin/python3',str(b/'precision_graph_container_gpu1_default.py'),'/home/songbinbin/.venvs/megatron-gdn-offload/bin/python','-m','torch.distributed.run','--standalone','--nproc-per-node=1','--log-dir',str(out/'ranks'),'--redirects','3',str(b/'rank_env_precision_graph.py'),str(b/'rank_distributed_pytest.py'),str(out/'reports'),'-q','-s','-p','no:cacheprovider','--confcutdir='+str(b/'probes'),str(b/'probes/test_ddp_fp4_alignment_final.py')+'::TestDistributedDataParallel::test_nvfp4_fused_weight_gradient_alignment']
 env=dict(os.environ,GDN_SOURCE=str(s),PYTHONHASHSEED='0',GDN_INDUCTOR_CACHE_DIR=str(b/'cache/align-te-inductor'),GDN_TRITON_CACHE_DIR=str(b/'cache/align-te-triton'))
 m={'base':rev,'arm':arm,'command':cmd,'patch_sha256':hashlib.sha256((b/'nvfp4-ddp-alignment-prototype-v3.patch').read_bytes()).hexdigest() if arm=='fixed' else None,'test_sha256':hashlib.sha256((b/'probes/test_ddp_fp4_alignment_final.py').read_bytes()).hexdigest(),'started_unix':time.time()}
 with (out/'launcher.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 m.update(exit_code=p.returncode,finished_unix=time.time());(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(arm,'exit',p.returncode,flush=True)
 if arm=='fixed' and p.returncode:raise SystemExit(p.returncode)
