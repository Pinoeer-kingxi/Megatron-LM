import json
import os
import subprocess
import time
from pathlib import Path

base = Path('/home/lwx/.cache/megatron-gdn-20261026')
jobs = [
    ('gdn2-scope', 'graph-merge-bb346eb', 'bb346eb1c3461f471c420bcb7691b425863df3f5',
     ['tests/unit_tests/ssm/test_gated_delta_net_offloading.py', '-k', 'gdn2']),
    ('gdn2-native', 'graph-merge-bb346eb', 'bb346eb1c3461f471c420bcb7691b425863df3f5',
     ['tests/unit_tests/ssm/test_gdn_offload_training.py', '-k', 'dp and gdn2']),
    ('gdn2-graph', 'graph-merge-bb346eb', 'bb346eb1c3461f471c420bcb7691b425863df3f5',
     ['tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py', '-k', 'tp1']),
]
for name, folder, revision, arguments in jobs:
    source = base / folder
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == revision
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    directory = base / (folder + '-' + name + '-gpu0')
    directory.mkdir()
    cmd = ['/usr/bin/python3', str(base / 'precision_graph_container_gpu0.py'),
           '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python', '-m', 'torch.distributed.run',
           '--standalone', '--nproc-per-node=1', '--log-dir', str(directory / 'ranks'), '--redirects', '3',
           str(base / 'rank_env_precision_graph.py'), str(base / 'rank_distributed_pytest.py'), str(directory / 'reports'),
           '-q', '-x', '-p', 'no:cacheprovider', '--confcutdir=tests/unit_tests/pipeline_parallel', *arguments]
    env = dict(os.environ, GDN_SOURCE=str(source), PYTHONHASHSEED='0',
        GDN_INDUCTOR_CACHE_DIR=str(base / 'cache/precision-graphs-18fe09f-controls-inductor'),
        GDN_TRITON_CACHE_DIR=str(base / 'cache/precision-graphs-18fe09f-controls-triton'))
    manifest = dict(revision=revision, command=cmd, started_unix=time.time(),
                    gpu_uuid='GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a')
    with (directory / 'launcher.log').open('w') as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(exit_code=result.returncode, finished_unix=time.time(),
        revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(name, result.returncode, flush=True)
    if result.returncode or manifest['revision_after'] != revision or manifest['dirty_after']:
        raise SystemExit(result.returncode or 1)
