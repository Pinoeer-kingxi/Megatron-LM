"""Run affected native training first, followed by actual TE capture/replay."""
import json
import os
import subprocess
import time
from pathlib import Path

base = Path('/home/lwx/.cache/megatron-gdn-20261026')
source = base / 'precision-graphs-18fe09f'
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == '18fe09f4743ee12ccb972679da7b25fdc4b19068'
for name, tests, expression in (
    ('gdn2-delayed-graph-plain', ['test_gdn_offload_cuda_graphs.py'],
     'tp1 and gdn2-False-True-True-True-1.0-fp8-delayed'),
    ('gdn2-tensorwise-graph-plain', ['test_gdn_offload_cuda_graphs.py'],
     'tp1 and gdn2-False-False-False-False-1.0-fp8-tensorwise'),
    ('gdn2-fp4-graph-diagnostic', ['test_gdn_offload_cuda_graphs.py'],
     'tp1 and gdn2-False-True-True-True-1.0-fp4'),
    ('gdn-fp4-graph-diagnostic', ['test_gdn_offload_cuda_graphs.py'],
     'tp1 and gdn-False-False-True-False-1.0-fp4'),
):
    directory = base / f'precision-graphs-18fe09f-{name}-gpu3'
    directory.mkdir()
    command = ['/usr/bin/python3', str(base / 'precision_graph_container_gpu3.py'),
        '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python', '-m', 'torch.distributed.run',
        '--standalone', '--nproc-per-node=1', '--log-dir', str(directory / 'ranks'), '--redirects', '3',
        str(base / 'rank_env_precision_graph.py'), str(base / 'rank_distributed_pytest.py'),
        str(directory / 'reports'), '-q', '-x', '-p', 'no:cacheprovider',
        '--confcutdir=tests/unit_tests/pipeline_parallel',
        *['tests/unit_tests/ssm/' + test for test in tests], '-k', expression]
    if name.endswith('diagnostic'):
        command[command.index(str(base / 'rank_distributed_pytest.py'))] = str(base / 'rank_graph_diagnostic.py')
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    manifest = dict(revision=revision, command=command, started_unix=time.time(),
                    gpu_uuid='GPU-dfee170b-6263-1438-4231-9e55c7f1c24d',
                    nvfp4_disable_rht=True, nvfp4_disable_stochastic_rounding=True)
    with (directory / 'launcher.log').open('w') as log:
        result = subprocess.run(command, env=dict(os.environ, GDN_SOURCE=str(source),
            PYTHONHASHSEED='0', GDN_INDUCTOR_CACHE_DIR=str(base / 'cache/precision-graphs-18fe09f-controls-inductor'), GDN_TRITON_CACHE_DIR=str(base / 'cache/precision-graphs-18fe09f-controls-triton')),
            stdout=log, stderr=subprocess.STDOUT)
    manifest.update(exit_code=result.returncode, finished_unix=time.time(),
        revision_after=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        dirty_after=subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True))
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Finished ' + name, flush=True)
