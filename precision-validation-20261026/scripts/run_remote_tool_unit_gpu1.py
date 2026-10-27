"""Validate the restored, reviewed chunk-tracking change at its exact new head."""
import json
import os
import subprocess
import time
from pathlib import Path

base=Path('/home/lwx/.cache/megatron-gdn-20261026')
source=base/'precision-stable'
directory=base/'precision-tool-final-gpu1'
directory.mkdir()
revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
assert revision=='688159928298844dc00d9d3a9ce33774735d997c'
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
command=['/usr/bin/python3',str(base/'remote_container_gpu1_inductor.py'),
 '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python','-m','torch.distributed.run',
 '--standalone','--nproc-per-node=1','--log-dir',str(directory/'ranks'),'--redirects','3',
 str(base/'rank_env.py'),str(base/'rank_distributed_pytest.py'),str(directory/'reports'),
 '-q','-x','-p','no:cacheprovider','--confcutdir=tests/unit_tests/pipeline_parallel',
 'tests/unit_tests/ssm/test_qwen_gdn_offload_training.py']
manifest=dict(revision=revision,command=command,started_unix=time.time(),gpu_uuid='GPU-5ceca9be-9cf0-99be-0868-915db7531fae')
with (directory/'launcher.log').open('w') as log:
 result=subprocess.run(command,env=dict(os.environ,GDN_SOURCE=str(source),GDN_INDUCTOR_CACHE_DIR=str(base/'cache/tool-unit-final')),stdout=log,stderr=subprocess.STDOUT)
manifest.update(exit_code=result.returncode,finished_unix=time.time(),revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
(directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert manifest['revision_after']==revision and not manifest['dirty_after']
raise SystemExit(result.returncode)
