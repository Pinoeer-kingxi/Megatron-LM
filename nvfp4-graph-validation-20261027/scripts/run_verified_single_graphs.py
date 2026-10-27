"""Run affected native training first, followed by actual TE capture/replay."""
import json
import os
import subprocess
import time
from pathlib import Path

base = Path('/home/lwx/.cache/megatron-gdn-20261026')
source = base / 'precision-graphs-7d52248'
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == '7d52248294fd759dd69b83c68ec6d1c0b36f2836'
for name, tests, expression in (
    ('graph-final', ['test_gdn_offload_cuda_graphs.py'], 'tp1'),
):
    directory = base / f'precision-graphs-7d52248-{name}-gpu0'
    directory.mkdir()
    command = ['/usr/bin/python3', str(base / 'precision_graph_container_gpu0.py'),
        '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python', '-m', 'torch.distributed.run',
        '--standalone', '--nproc-per-node=1', '--log-dir', str(directory / 'ranks'), '--redirects', '3',
        str(base / 'rank_env_precision_graph.py'), str(base / 'rank_distributed_pytest.py'),
        str(directory / 'reports'), '-q', '-x', '-p', 'no:cacheprovider',
        '--confcutdir=tests/unit_tests/pipeline_parallel',
        *['tests/unit_tests/ssm/' + test for test in tests]]
    if expression:
        command += ['-k', expression]
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    manifest = dict(revision=revision, command=command, started_unix=time.time(),
                    gpu_uuid='GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a',
                    nvfp4_disable_rht=True, nvfp4_disable_stochastic_rounding=True)
    with (directory / 'launcher.log').open('w') as log:
        result = subprocess.run(command, env=dict(os.environ, GDN_SOURCE=str(source),
            PYTHONHASHSEED='0', GDN_INDUCTOR_CACHE_DIR=str(base / ('cache/precision-graphs-a11c9e2' if name == 'native-final' else 'cache/precision-graphs-18fe09f-controls-inductor')), GDN_TRITON_CACHE_DIR=str(base / ('cache/triton' if name == 'native-final' else 'cache/precision-graphs-18fe09f-controls-triton'))),
            stdout=log, stderr=subprocess.STDOUT)
    manifest.update(exit_code=result.returncode, finished_unix=time.time(),
        revision_after=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        dirty_after=subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True))
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    assert not result.returncode, directory
    print('Finished ' + name, flush=True)
