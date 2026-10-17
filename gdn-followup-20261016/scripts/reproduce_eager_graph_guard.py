"""Reproduce the eager warmup-zero regression using a real decoder constructor."""

import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
from types import SimpleNamespace

from run_gpu_acceptance import ARTIFACTS, VENV, container_command, gpu_inventory, python_path, require_idle

parent=Path(__file__).resolve().parents[1]
root=parent/'eager-guard-before';root.mkdir()
source=root/'source';source.mkdir()
with tarfile.open(parent/'extensions-ready/source.tar') as archive: archive.extractall(source)
path=source/'tests/unit_tests/ssm/test_gdn_offload_training.py'
before=path.read_text()
after=before.replace('        min_offloaded_tensor_size=1024,\n','        min_offloaded_tensor_size=1024,\n        cuda_graph_warmup_steps=0,\n')
assert before!=after
path.write_text(after)
(root/'fixture.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/tests/unit_tests/ssm/test_gdn_offload_training.py',tofile='b/tests/unit_tests/ssm/test_gdn_offload_training.py')))
gpus=gpu_inventory(['2']);require_idle(gpus)
source_root=ARTIFACTS/'split-pr-20261015'
args=SimpleNamespace(source_root=source_root,output=parent,stage='four',weights=ARTIFACTS/'models/Qwen3.5-0.8B',data=ARTIFACTS/'training-data/ks-item-sft.jsonl')
name='megatron-gdn-eager-guard-before-20261017'
manifest=dict(status='running',base='043e3173451bea74e63c02b197867bf356ad72e9',fixture_patch_sha256=hashlib.sha256((root/'fixture.patch').read_bytes()).hexdigest(),gpus=gpus,started_unix=time.time(),expected_failure='Fine-grained activation offloading needs cuda_graph_warmup_steps > 0.')
container_id=None
try:
 command=container_command(args,gpus,name);manifest['container_command']=command
 container_id=subprocess.check_output(command,text=True).strip()
 command=['docker','exec','--workdir',str(source),'--env','PYTHONPATH='+python_path(source,source_root),'--env','PYTHONDONTWRITEBYTECODE=1',name,str(VENV/'bin/python'),'-m','torch.distributed.run','--standalone','--nproc-per-node=1','--log-dir',str(root/'ranks'),'--redirects','3','--tee','3',str(parent/'scripts/rank_pytest.py'),str(root/'reports'),'-q','-x','-p','no:cacheprovider','--confcutdir=tests/unit_tests/ssm','tests/unit_tests/ssm/test_gdn_offload_training.py','-k','dp and gdn2 and False and 1.0']
 manifest['command']=command
 with (root/'launcher.log').open('w') as log: result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 manifest['exit_code']=result.returncode
 assert result.returncode!=0
 assert manifest['expected_failure'] in (root/'launcher.log').read_text()
 manifest['status']='expected_regression_reproduced'
finally:
 if container_id:
  cleanup=subprocess.run(['docker','stop','--time','10',container_id],capture_output=True,text=True)
  manifest['cleanup']=dict(exit_code=cleanup.returncode)
 manifest['finished_unix']=time.time()
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest['status'])
