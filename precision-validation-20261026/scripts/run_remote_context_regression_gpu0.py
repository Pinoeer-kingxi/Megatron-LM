"""Run one real-kernel regression unchanged against old and corrected tool revisions."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

base=Path('/home/lwx/.cache/megatron-gdn-20261026')
for label,source,expected in [('old',base/'precision',1),('fixed',base/'precision-context-test',0)]:
 d=base/f'fp4-context-regression-{label}';d.mkdir()
 revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
 test=base/'test_qwen_context_repro.py' if label=='old' else source/'tests/unit_tests/ssm/test_qwen_gdn_offload_training.py'
 command=['/usr/bin/python3',str(base/'remote_container_gpu0.py'),'/home/songbinbin/.venvs/megatron-gdn-offload/bin/python','-m','torch.distributed.run','--standalone','--nproc-per-node=1','--log-dir',str(d/'ranks'),'--redirects','3',str(base/'rank_env.py'),str(base/'rank_distributed_pytest.py'),str(d/'reports'),'-q','-x','-p','no:cacheprovider','--confcutdir=tests/unit_tests/pipeline_parallel',str(test),'-k','forward_step_activates_nvfp4']
 manifest=dict(revision=revision,command=command,started_unix=time.time(),expected_exit=expected,test_sha256=hashlib.sha256(test.read_bytes()).hexdigest())
 with (d/'launcher.log').open('w') as log:
  result=subprocess.run(command,env=dict(os.environ,GDN_SOURCE=str(source)),stdout=log,stderr=subprocess.STDOUT)
 manifest.update(exit_code=result.returncode,finished_unix=time.time(),revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
 (d/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 assert result.returncode==expected and revision==manifest['revision_after'] and not manifest['dirty_after']
 print(label+' exit='+str(result.returncode),flush=True)
