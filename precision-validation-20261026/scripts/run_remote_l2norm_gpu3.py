"""Isolate the observed reduction-config differences on real, private Qwen inputs."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

base = Path('/home/lwx/.cache/megatron-gdn-20261026')
source = base / 'precision-stable'
directory = base / 'l2norm-real-input-gpu3'
directory.mkdir()
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == '688159928298844dc00d9d3a9ce33774735d997c'
assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
prefix = ['/usr/bin/python3', str(base / 'remote_container_gpu3_inductor.py'),
          '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python']
capture = prefix + ['-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=1',
    '--log-dir', str(directory / 'ranks'), '--redirects', '3', str(base / 'rank_env.py'),
    str(base / 'capture_l2norm.py'), str(base / 'l2norm-private/captured.pt'),
    str(source / 'tools/ssm/qwen_gdn_offload_training.py'),
    '--weights', str(base / 'models/Qwen3.5-0.8B'), '--data', str(base / 'data/ks-item-sft.jsonl'),
    '--precision', 'fp8-delayed', '--seq-length', '128', '--microbatches', '4',
    '--warmup', '1', '--iterations', '1', '--recompute-norm',
    '--output', str(directory / 'result.json')]
replay = prefix + [str(base / 'replay_l2norm.py')]
manifest = dict(revision=revision, started_unix=time.time(), commands=[capture, replay],
                gpu_uuid='GPU-dfee170b-6263-1438-4231-9e55c7f1c24d',
                scripts={name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                         for name in ('capture_l2norm.py', 'replay_l2norm.py')})
environment = dict(os.environ, GDN_SOURCE=str(source), PYTHONHASHSEED='0',
                   GDN_INDUCTOR_CACHE_DIR=str(base / 'cache/final-inductor-fp8-delayed'))
codes = []
for name, command in (('capture', capture), ('replay', replay)):
    with (directory / f'{name}.log').open('w') as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    codes.append(result.returncode)
    if result.returncode:
        break
manifest.update(exit_codes=codes, finished_unix=time.time(),
                revision_after=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                dirty_after=subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True))
(directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
assert codes == [0, 0], codes
assert manifest['revision_after'] == revision and not manifest['dirty_after']
print((base / 'l2norm-configuration-comparison.json').read_text(), flush=True)
