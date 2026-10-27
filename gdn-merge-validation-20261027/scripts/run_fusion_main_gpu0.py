import json
import os
import subprocess
import time
from pathlib import Path

base = Path('/home/lwx/.cache/megatron-gdn-20261026')
source = base / 'precision-fusion-a1110df'
revision = 'a1110dfc9563301dbe10e15f1540a4f719fa3805'
assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == revision
assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
for name, tests, expression in (
    ('scope', ['test_gated_delta_net_offloading.py', 'test_qwen_gdn_offload_training.py'], ''),
    ('native', ['test_gdn_offload_training.py'], 'fp8_delayed or fp8_tensorwise or fp4'),
    ('graph', ['test_gdn_offload_cuda_graphs.py'], 'tp1'),
):
    directory = base / f'precision-fusion-a1110df-{name}-gpu0'
    directory.mkdir()
    if name == 'graph':
        assert list((base / 'apex-wgrad-build/out').glob('fused_weight_gradient_mlp_cuda*.so'))
    command = ['/usr/bin/python3', str(base / 'precision_graph_container_gpu0_apex.py'),
               '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python', '-m', 'torch.distributed.run',
               '--standalone', '--nproc-per-node=1', '--log-dir', str(directory / 'ranks'), '--redirects', '3',
               str(base / 'rank_env_precision_graph.py'), str(base / 'rank_distributed_pytest.py'),
               str(directory / 'reports'), '-q', '-x', '-p', 'no:cacheprovider',
               '--confcutdir=tests/unit_tests/pipeline_parallel',
               *['tests/unit_tests/ssm/' + test for test in tests]]
    if expression:
        command += ['-k', expression]
    environment = dict(os.environ, GDN_SOURCE=str(source), PYTHONHASHSEED='0',
        GDN_INDUCTOR_CACHE_DIR=str(base / 'cache/precision-graphs-18fe09f-controls-inductor'),
        GDN_TRITON_CACHE_DIR=str(base / 'cache/precision-graphs-18fe09f-controls-triton'))
    manifest = dict(revision=revision, command=command, started_unix=time.time(),
                    gpu_uuid='GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a',
                    nvfp4_disable_rht=True, nvfp4_disable_stochastic_rounding=True)
    with (directory / 'launcher.log').open('w') as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(exit_code=result.returncode, finished_unix=time.time(),
                    revision_after=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                    dirty_after=subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True))
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(name, result.returncode, flush=True)
    if result.returncode or manifest['revision_after'] != revision or manifest['dirty_after']:
        raise SystemExit(result.returncode or 1)
