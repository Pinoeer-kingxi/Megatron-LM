"""Validate the restored, reviewed chunk-tracking change at its exact new head."""
import json
import os
import subprocess
import time
from pathlib import Path

base=Path('/home/lwx/.cache/megatron-gdn-20261026')
source=base/'feature-scope'
directory=base/'feature-scope-gpu0'
directory.mkdir()
revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
assert revision=='d8db9192d24e1e5f8e4884f1d5acfb463a75452d'
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
command=['/usr/bin/python3',str(base/'remote_container_gpu0.py'),
 '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python','-m','torch.distributed.run',
 '--standalone','--nproc-per-node=1','--log-dir',str(directory/'ranks'),'--redirects','3',
 str(base/'rank_env.py'),str(base/'rank_distributed_pytest.py'),str(directory/'reports'),
 '-q','-x','-p','no:cacheprovider','--confcutdir=tests/unit_tests/pipeline_parallel',
 'tests/unit_tests/ssm/test_gated_delta_net_offloading.py',
 'tests/unit_tests/ssm/test_gdn_offload_training.py',
 'tests/unit_tests/ssm/test_gdn_activation_offload.py',
 'tests/unit_tests/ssm/test_qwen_gdn_offload_training.py']
manifest=dict(revision=revision,command=command,started_unix=time.time(),gpu_uuid='GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a')
with (directory/'launcher.log').open('w') as log:
 result=subprocess.run(command,env=dict(os.environ,GDN_SOURCE=str(source)),stdout=log,stderr=subprocess.STDOUT)
manifest.update(exit_code=result.returncode,finished_unix=time.time(),revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
(directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert manifest['revision_after']==revision and not manifest['dirty_after']
raise SystemExit(result.returncode)
