import json
import os
import subprocess
import time
from pathlib import Path

base=Path('/home/lwx/.cache/megatron-gdn-20261026')
jobs=[
 ('feature-native', 'feature-merge-b6fa63', 'b6fa63ab8c2cb855f0368cd8455245c295cde234',
  'tests/unit_tests/ssm/test_gdn_offload_training.py',
  '(tp_sp and not tp_sp_cp and not tp_sp_pp) or (pp and not pp_vpp and not tp_sp_pp)'),
 ('gdn2-native', 'graph-merge-bb346eb', 'bb346eb1c3461f471c420bcb7691b425863df3f5',
  'tests/unit_tests/ssm/test_gdn_offload_training.py',
  'gdn2 and (tp_sp or pp) and not tp_sp_cp and not tp_sp_pp and not pp_vpp'),
 ('gdn2-graph', 'graph-merge-bb346eb', 'bb346eb1c3461f471c420bcb7691b425863df3f5',
  'tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py', 'tp2-sp or pp2'),
]
for name, folder, revision, test, expression in jobs:
 source=base/folder
 assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==revision
 assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
 directory=base/(folder+'-'+name+'-gpu12'); directory.mkdir()
 cmd=['/usr/bin/python3',str(base/'precision_graph_container_gpu12_apex.py'),
      '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python','-m','torch.distributed.run',
      '--standalone','--nproc-per-node=2','--log-dir',str(directory/'ranks'),'--redirects','3',
      str(base/'rank_env_precision_graph.py'),str(base/'rank_distributed_pytest.py'),str(directory/'reports'),
      '-q','-x','-p','no:cacheprovider','--confcutdir=tests/unit_tests/pipeline_parallel',test,'-k',expression]
 env=dict(os.environ,GDN_SOURCE=str(source),PYTHONHASHSEED='0',
          GDN_INDUCTOR_CACHE_DIR=str(base/'cache/precision-graphs-057b70f-two-rank-inductor'),
          GDN_TRITON_CACHE_DIR=str(base/'cache/precision-graphs-057b70f-two-rank-triton'))
 manifest=dict(revision=revision,command=cmd,started_unix=time.time(),shared_gpus=True,
     gpu_uuids=['GPU-5ceca9be-9cf0-99be-0868-915db7531fae','GPU-fa4310c6-b684-22a7-6878-e8f0f86cf214'])
 with (directory/'launcher.log').open('w') as log:
  result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 manifest.update(exit_code=result.returncode,finished_unix=time.time(),
  revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
  dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
 (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(name,result.returncode,flush=True)
 if result.returncode or manifest['revision_after'] != revision or manifest['dirty_after']:
  raise SystemExit(result.returncode or 1)
