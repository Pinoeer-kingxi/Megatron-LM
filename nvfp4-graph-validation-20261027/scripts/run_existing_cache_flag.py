"""Record fresh-process NVFP4 graph controls on one otherwise idle GPU."""
import json
import os
import subprocess
import time
from pathlib import Path

base = Path('/home/lwx/.cache/megatron-gdn-20261026')
source = base / 'precision-graphs-18fe09f'
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == '18fe09f4743ee12ccb972679da7b25fdc4b19068'
for mode in ('disable-existing-cache',):
    directory = base / ('nvfp4-graph-options-gdn2-' + mode + '-gpu0')
    directory.mkdir()
    command = ['/usr/bin/python3', str(base / 'precision_graph_container_gpu0.py'),
        '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python', '-m', 'torch.distributed.run',
        '--standalone', '--nproc-per-node=1', '--log-dir', str(directory / 'ranks'), '--redirects', '3',
        str(base / 'rank_env_precision_graph.py'), str(base / 'graph_option_probe.py'),
        str(directory / 'reports'), '-q', '-x', '-p', 'no:cacheprovider',
        '--confcutdir=tests/unit_tests/pipeline_parallel',
        'tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py', '-k',
        'tp1 and gdn2-False-True-True-True-1.0-fp4']
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    manifest = dict(revision=revision, command=command, mode=mode, started_unix=time.time(),
                    gpu_uuid='GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a')
    env = dict(os.environ, GDN_SOURCE=str(source), GDN_GRAPH_PROBE_MODE=mode,
        PYTHONHASHSEED='0',
        GDN_INDUCTOR_CACHE_DIR=str(base / 'cache/precision-graphs-18fe09f-controls-inductor'),
        GDN_TRITON_CACHE_DIR=str(base / 'cache/precision-graphs-18fe09f-controls-triton'))
    with (directory / 'launcher.log').open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(exit_code=result.returncode, finished_unix=time.time(),
        revision_after=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        dirty_after=subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True))
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(mode, result.returncode, flush=True)
