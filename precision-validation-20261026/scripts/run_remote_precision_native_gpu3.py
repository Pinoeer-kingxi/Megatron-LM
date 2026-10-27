"""Run affected configuration and native precision training tests at the final source."""
import json
import os
import subprocess
import time
from pathlib import Path

base=Path('/home/lwx/.cache/megatron-gdn-20261026')
source=base/'precision-stable'
directory=base/'precision-native-final-gpu3'
directory.mkdir()
revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
assert revision=='688159928298844dc00d9d3a9ce33774735d997c'
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
command=['/usr/bin/python3',str(base/'remote_container_gpu3_inductor.py'),
 '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python','-m','torch.distributed.run',
 '--standalone','--nproc-per-node=1','--log-dir',str(directory/'ranks'),'--redirects','3',
 str(base/'rank_env.py'),str(base/'rank_distributed_pytest.py'),str(directory/'reports'),
 '-q','-x','-p','no:cacheprovider','--confcutdir=tests/unit_tests/pipeline_parallel',
 'tests/unit_tests/ssm/test_gated_delta_net_offloading.py',
 'tests/unit_tests/ssm/test_gdn_offload_training.py',
 '-k','reject_unsupported_gdn_offload or gdn_offload_config or fp8_delayed or fp8_tensorwise or fp4']
manifest=dict(revision=revision,command=command,started_unix=time.time(),gpu_uuid='GPU-dfee170b-6263-1438-4231-9e55c7f1c24d')
with (directory/'launcher.log').open('w') as log:
 result=subprocess.run(command,env=dict(os.environ,GDN_SOURCE=str(source),GDN_INDUCTOR_CACHE_DIR=str(base/'cache/native-final-gpu3')),stdout=log,stderr=subprocess.STDOUT)
manifest.update(exit_code=result.returncode,finished_unix=time.time(),revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
(directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert manifest['revision_after']==revision and not manifest['dirty_after']
raise SystemExit(result.returncode)
