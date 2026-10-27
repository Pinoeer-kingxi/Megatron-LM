import json,os,subprocess,time
from pathlib import Path
b=Path('/home/lwx/.cache/megatron-gdn-20261026');s=b/'nvfp4-ddp-66b6bb6';rev='66b6bb62a00bdccf8365096e6828215cffed4ce7';out=b/'nvfp4-ddp-66b6bb6-unit-gpu12';out.mkdir(exist_ok=False)
assert subprocess.check_output(['git','-C',str(s),'rev-parse','HEAD'],text=True).strip()==rev
assert not subprocess.check_output(['git','-C',str(s),'status','--porcelain'],text=True)
cmd=['/usr/bin/python3',str(b/'precision_graph_container_gpu12_apex.py'),'/home/songbinbin/.venvs/megatron-gdn-offload/bin/python','-m','torch.distributed.run','--standalone','--nproc-per-node=2','--log-dir',str(out/'ranks'),'--redirects','3',str(b/'rank_env_precision_graph.py'),str(b/'rank_distributed_pytest.py'),str(out/'reports'),'-q','-s','-p','no:cacheprovider','--confcutdir=tests/unit_tests/pipeline_parallel','tests/unit_tests/distributed/test_param_layout.py::TestDefaultParamLayout','tests/unit_tests/distributed/test_distributed_data_parallel.py::TestDistributedDataParallel::test_nvfp4_fused_weight_gradient_alignment']
env=dict(os.environ,GDN_SOURCE=str(s),PYTHONHASHSEED='0',GDN_INDUCTOR_CACHE_DIR=str(b/'cache/align-66b6-inductor'),GDN_TRITON_CACHE_DIR=str(b/'cache/align-66b6-triton'))
m={'revision':rev,'command':cmd,'gpu_uuids':['GPU-5ceca9be-9cf0-99be-0868-915db7531fae','GPU-fa4310c6-b684-22a7-6878-e8f0f86cf214'],'shared_gpu':True,'started_unix':time.time()}
with (out/'launcher.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
m.update(exit_code=p.returncode,finished_unix=time.time(),revision_after=subprocess.check_output(['git','-C',str(s),'rev-parse','HEAD'],text=True).strip(),dirty_after=subprocess.check_output(['git','-C',str(s),'status','--porcelain'],text=True));(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('alignment final GPU',p.returncode,flush=True)
raise SystemExit(p.returncode)
