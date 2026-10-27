import json,os,subprocess,time
from pathlib import Path
b=Path('/home/lwx/.cache/megatron-gdn-20261026');s=b/'mrope-graph-865d79d';rev='865d79dd37f304a4dffa23158e8df52735d43247';out=b/'mrope-graph-865d79d-native-gpu1';out.mkdir(exist_ok=False)
assert subprocess.check_output(['git','-C',str(s),'rev-parse','HEAD'],text=True).strip()==rev
assert not subprocess.check_output(['git','-C',str(s),'status','--porcelain'],text=True)
cmd=['/usr/bin/python3',str(b/'precision_graph_container_gpu1_default.py'),'/home/songbinbin/.venvs/megatron-gdn-offload/bin/python','-m','torch.distributed.run','--standalone','--nproc-per-node=1','--log-dir',str(out/'ranks'),'--redirects','3',str(b/'rank_env_precision_graph.py'),str(b/'rank_distributed_pytest.py'),str(out/'reports'),'-q','-s','-x','-p','no:cacheprovider','--confcutdir=tests/unit_tests/pipeline_parallel','tests/unit_tests/transformer/test_cuda_graphs.py::TestTECudaGraphHelper::test_mrope_replay_uses_current_positions']
env=dict(os.environ,GDN_SOURCE=str(s),PYTHONHASHSEED='0',GDN_INDUCTOR_CACHE_DIR=str(b/'cache/mrope-final-inductor'),GDN_TRITON_CACHE_DIR=str(b/'cache/mrope-final-triton'))
m={'revision':rev,'command':cmd,'gpu_uuid':'GPU-5ceca9be-9cf0-99be-0868-915db7531fae','shared_gpu':True,'started_unix':time.time()}
with (out/'launcher.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
m.update(exit_code=p.returncode,finished_unix=time.time(),revision_after=subprocess.check_output(['git','-C',str(s),'rev-parse','HEAD'],text=True).strip(),dirty_after=subprocess.check_output(['git','-C',str(s),'status','--porcelain'],text=True));(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('mrope final GPU',p.returncode,flush=True);raise SystemExit(p.returncode)
