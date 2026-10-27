import json,os,subprocess,time
from pathlib import Path
b=Path('/home/lwx/.cache/megatron-gdn-20261026')
jobs=[('graph-native','graph-with-mrope-c5264b6','c5264b66197040f4309edfdb22dc9bbaf62189b1','tests/unit_tests/ssm/test_gdn_offload_training.py','gdn2 and (tp_sp or pp) and not tp_sp_cp and not tp_sp_pp and not pp_vpp'),('graph-replay','graph-with-mrope-c5264b6','c5264b66197040f4309edfdb22dc9bbaf62189b1','tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py','tp2-sp or pp2'),('precision-replay','precision-with-mrope-dfaa633','dfaa6338c65edab118ba415fcc14c73241f503a6','tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py','gdn2 and (tp2-sp or pp2) and (fp8-delayed or fp8-tensorwise or fp4)')]
for name,folder,rev,test,expr in jobs:
 s=b/folder;out=b/f'{folder}-{name}-gpu12';out.mkdir(exist_ok=False)
 assert subprocess.check_output(['git','-C',str(s),'rev-parse','HEAD'],text=True).strip()==rev
 assert not subprocess.check_output(['git','-C',str(s),'status','--porcelain'],text=True)
 cmd=['/usr/bin/python3',str(b/'precision_graph_container_gpu12_apex.py'),'/home/songbinbin/.venvs/megatron-gdn-offload/bin/python','-m','torch.distributed.run','--standalone','--nproc-per-node=2','--log-dir',str(out/'ranks'),'--redirects','3',str(b/'rank_env_precision_graph.py'),str(b/'rank_distributed_pytest.py'),str(out/'reports'),'-q','-s','-x','-p','no:cacheprovider','--confcutdir=tests/unit_tests/pipeline_parallel',test,'-k',expr]
 env=dict(os.environ,GDN_SOURCE=str(s),PYTHONHASHSEED='0',GDN_INDUCTOR_CACHE_DIR=str(b/'cache/mrope-final-two-rank-inductor'),GDN_TRITON_CACHE_DIR=str(b/'cache/mrope-final-two-rank-triton'))
 m={'revision':rev,'command':cmd,'gpu_uuids':['GPU-5ceca9be-9cf0-99be-0868-915db7531fae','GPU-fa4310c6-b684-22a7-6878-e8f0f86cf214'],'shared_gpus':True,'rht_disabled':True,'stochastic_rounding_disabled':True,'started_unix':time.time()}
 with (out/'launcher.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 m.update(exit_code=p.returncode,finished_unix=time.time(),revision_after=subprocess.check_output(['git','-C',str(s),'rev-parse','HEAD'],text=True).strip(),dirty_after=subprocess.check_output(['git','-C',str(s),'status','--porcelain'],text=True));(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(name,p.returncode,flush=True)
 if p.returncode or m['revision_after']!=rev or m['dirty_after']:raise SystemExit(p.returncode or 1)
