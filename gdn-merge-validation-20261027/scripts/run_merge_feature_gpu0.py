import json
import os
import subprocess
import time
from pathlib import Path

base = Path('/home/lwx/.cache/megatron-gdn-20261026')
source = base / 'feature-merge-b6fa63'
revision = 'b6fa63ab8c2cb855f0368cd8455245c295cde234'
assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == revision
assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
for name, arguments in [
    ('feature', ['tests/unit_tests/ssm/test_gated_delta_net_offloading.py',
                 'tests/unit_tests/ssm/test_gdn_offload_training.py',
                 'tests/unit_tests/ssm/test_gdn_activation_offload.py',
                 'tests/unit_tests/ssm/test_qwen_gdn_offload_training.py']),
    ('pipeline', ['tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py',
                  '-k', 'chunk_offload_handler_skips_checkpoint_without_output_storage or offload_resident_groups_preserve_forward_chunks']),
]:
    directory = base / ('feature-merge-b6fa63-' + name + '-gpu0')
    directory.mkdir()
    cmd = ['/usr/bin/python3', str(base / 'remote_container_gpu0.py'),
           '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python', '-m', 'torch.distributed.run',
           '--standalone', '--nproc-per-node=1', '--log-dir', str(directory / 'ranks'), '--redirects', '3',
           str(base / 'rank_env.py'), str(base / 'rank_distributed_pytest.py'), str(directory / 'reports'),
           '-q', '-x', '-p', 'no:cacheprovider', '--confcutdir=tests/unit_tests/pipeline_parallel', *arguments]
    start = time.time()
    with (directory / 'launcher.log').open('w') as log:
        result = subprocess.run(cmd, env=dict(os.environ, GDN_SOURCE=str(source)), stdout=log, stderr=subprocess.STDOUT)
    manifest = dict(revision=revision, command=cmd, started_unix=start, finished_unix=time.time(),
                    gpu_uuid='GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a', exit_code=result.returncode,
                    revision_after=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                    dirty_after=subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True))
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
