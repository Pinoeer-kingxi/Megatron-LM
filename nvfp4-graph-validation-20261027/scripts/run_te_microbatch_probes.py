"""Run small fresh-process TE-only controls with recorded commands."""
import json
import os
import subprocess
import time
from pathlib import Path

base = Path('/home/lwx/.cache/megatron-gdn-20261026')
for kind, width, caching in [('linear', 4096, 1), ('norm', 4096, 1), ('norm', 2592, 1),
                              ('norm', 4096, 0), ('norm', 2592, 0)]:
    directory = base / f'nvfp4-te-microbatches-{kind}-{width}-cache{caching}-gpu0'
    directory.mkdir()
    command = ['/usr/bin/python3', str(base / 'precision_graph_container_gpu0.py'),
        '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python',
        str(base / 'probe_te_fp4_microbatches.py'), kind, str(width), str(caching)]
    env = dict(os.environ, GDN_SOURCE=str(base / 'precision-graphs-1aa1151'),
        GDN_TRITON_CACHE_DIR=str(base / 'cache/nvfp4-te-probe-triton'),
        GDN_INDUCTOR_CACHE_DIR=str(base / 'cache/nvfp4-te-probe-inductor'))
    manifest = dict(command=command, started_unix=time.time(),
                    gpu_uuid='GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a')
    with (directory / 'launcher.log').open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(exit_code=result.returncode, finished_unix=time.time())
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(kind, width, caching, result.returncode, flush=True)
