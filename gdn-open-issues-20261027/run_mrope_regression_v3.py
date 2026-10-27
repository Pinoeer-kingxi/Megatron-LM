import hashlib,json,os,subprocess,time
from pathlib import Path
b=Path('/home/lwx/.cache/megatron-gdn-20261026')
for arm in ('baseline','fixed'):
 s=b/('precision-final-aecc0e' if arm=='baseline' else 'precision-mrope-prototype');out=b/f'mrope-native-{arm}-v3-20261027-gpu1';out.mkdir(exist_ok=False)
 cmd=['/usr/bin/python3',str(b/'precision_graph_container_gpu1_default.py'),'/home/songbinbin/.venvs/megatron-gdn-offload/bin/python','-m','torch.distributed.run','--standalone','--nproc-per-node=1','--log-dir',str(out/'ranks'),'--redirects','3',str(b/'rank_env_precision_graph.py'),str(b/'rank_distributed_pytest.py'),str(out/'reports'),'-q','-s','-p','no:cacheprovider','--confcutdir='+str(b/'probes'),str(b/'probes/test_mrope_graph_regression_v3.py')+'::TestTECudaGraphHelper::test_mrope_replay_uses_current_positions']
 env=dict(os.environ,GDN_SOURCE=str(s),PYTHONHASHSEED='0',GDN_INDUCTOR_CACHE_DIR=str(b/'cache/mrope-native-inductor'),GDN_TRITON_CACHE_DIR=str(b/'cache/mrope-native-triton'))
 m={'base':'aecc0e8116912554f27a62a51ad2aeca479a7f9e','arm':arm,'command':cmd,'test_sha256':hashlib.sha256((b/'probes/test_mrope_graph_regression_v3.py').read_bytes()).hexdigest(),'patch_sha256':hashlib.sha256((b/'mrope-graph-prototype.patch').read_bytes()).hexdigest() if arm=='fixed' else None,'started_unix':time.time(),'shared_gpu':True}
 with (out/'launcher.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 m.update(exit_code=p.returncode,finished_unix=time.time());(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(arm,'exit',p.returncode,flush=True)
 if arm=='fixed' and p.returncode:raise SystemExit(p.returncode)
