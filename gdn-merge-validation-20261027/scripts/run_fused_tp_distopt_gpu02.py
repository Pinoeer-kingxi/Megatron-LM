import json
import os
import subprocess
import time
from pathlib import Path

base = Path('/home/lwx/.cache/megatron-gdn-20261026')
source = base / 'fused-distopt-control'
revision = '5267f8517db5095b476cc3dba49086a4562ae134'
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() == revision
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
directory = base/'fused-distopt-control-tp2-gpu02'
directory.mkdir()
cmd = ['/usr/bin/python3', str(base/'precision_graph_container_gpu02_apex.py'),
       '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python', '-m', 'torch.distributed.run',
       '--standalone', '--nproc-per-node=2', '--log-dir', str(directory/'ranks'), '--redirects', '3',
       str(base/'rank_env_precision_graph.py'), str(base/'rank_distributed_pytest.py'), str(directory/'reports'),
       '-q', '-x', '-p', 'no:cacheprovider', '--confcutdir=tests/unit_tests/pipeline_parallel',
       'tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py', '-k',
       'gdn2-False-True-True-True-1.0-fp4-True-tp2-sp', '--showlocals']
env = dict(os.environ, GDN_SOURCE=str(source), PYTHONHASHSEED='0',
    GDN_INDUCTOR_CACHE_DIR=str(base/'cache/precision-graphs-057b70f-two-rank-inductor'),
    GDN_TRITON_CACHE_DIR=str(base/'cache/precision-graphs-057b70f-two-rank-triton'))
manifest = dict(revision=revision, command=cmd, started_unix=time.time(),
    gpu_uuids=['GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a', 'GPU-fa4310c6-b684-22a7-6878-e8f0f86cf214'],
    shared_gpus=True, nvfp4_disable_rht=True, nvfp4_disable_stochastic_rounding=True)
with (directory/'launcher.log').open('w') as log:
    result = subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
manifest.update(exit_code=result.returncode,finished_unix=time.time(),
    revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
    dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
(directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('two-rank',result.returncode,flush=True)
raise SystemExit(result.returncode)
