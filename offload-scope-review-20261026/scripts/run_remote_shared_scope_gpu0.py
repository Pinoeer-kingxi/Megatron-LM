"""Validate the restored, reviewed chunk-tracking change at its exact new head."""
import json
import os
import subprocess
import time
from pathlib import Path

base=Path('/home/lwx/.cache/megatron-gdn-20261026')
source=base/'shared-scope'
directory=base/'shared-scope-gpu0'
directory.mkdir()
revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
assert revision=='2b4211c0681a3c9d08b9b685af687ed13e6f4053'
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
command=['/usr/bin/python3',str(base/'remote_container_gpu0.py'),
 '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python','-m','torch.distributed.run',
 '--standalone','--nproc-per-node=1','--log-dir',str(directory/'ranks'),'--redirects','3',
 str(base/'rank_env.py'),str(base/'rank_distributed_pytest.py'),str(directory/'reports'),
 '-q','-x','-p','no:cacheprovider','--confcutdir=tests/unit_tests/pipeline_parallel',
 'tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py',
 '-k','resident_groups_preserve_forward_chunks or (with_cuda_graph and (offload_modules6 or offload_modules7 or offload_modules8))']
manifest=dict(revision=revision,command=command,started_unix=time.time(),gpu_uuid='GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a')
with (directory/'launcher.log').open('w') as log:
 result=subprocess.run(command,env=dict(os.environ,GDN_SOURCE=str(source)),stdout=log,stderr=subprocess.STDOUT)
manifest.update(exit_code=result.returncode,finished_unix=time.time(),revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
(directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert manifest['revision_after']==revision and not manifest['dirty_after']
raise SystemExit(result.returncode)
