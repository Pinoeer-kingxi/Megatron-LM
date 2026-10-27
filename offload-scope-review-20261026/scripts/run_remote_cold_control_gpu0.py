"""Isolate a cold autotuning cache from offloading; leave original results intact."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

base = Path('/home/lwx/.cache/megatron-gdn-20261026')
source = base / 'precision'
cache = base / 'cache/cold-control-gpu0'
shutil.copytree(base / 'cache/triton', cache, ignore=shutil.ignore_patterns('*.autotune.json', '*.lock'))
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
python = '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python'
outputs = []

def tuning_snapshot():
    return {str(p.relative_to(cache)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(cache.rglob('*.autotune.json'))}

for arm in ('cold-disabled', 'warm-disabled', 'warm-repeated-disabled'):
    directory = base / f'qwen-sm120-gpu0-cold-control-{arm}'
    directory.mkdir()
    before = tuning_snapshot()
    command = ['/usr/bin/python3', str(base / 'remote_container_gpu0_cold.py'), python,
               '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=1',
               '--log-dir', str(directory / 'ranks'), '--redirects', '3',
               str(base / 'rank_env.py'), str(source / 'tools/ssm/qwen_gdn_offload_training.py'),
               '--weights', str(base / 'models/Qwen3.5-0.8B'), '--data', str(base / 'data/ks-item-sft.jsonl'),
               '--precision', 'fp8-delayed', '--seq-length', '128', '--microbatches', '4',
               '--warmup', '1', '--iterations', '2', '--min-offloaded-tensor-size', '1024',
               '--recompute-norm', '--check-state', '--output', str(directory / 'result.json')]
    environment = dict(os.environ, GDN_SOURCE=str(source), GDN_TRITON_CACHE_DIR=str(cache), TRITON_PRINT_AUTOTUNING='1')
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    manifest = dict(revision=revision, command=command, cache=str(cache), tuning_before=before,
                    started_unix=time.time(), gpu_uuid='GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a')
    with (directory / 'launcher.log').open('w') as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(exit_code=result.returncode, finished_unix=time.time(), tuning_after=tuning_snapshot(),
                    revision_after=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                    dirty_after=subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True))
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    assert result.returncode == 0 and manifest['revision_after'] == revision and not manifest['dirty_after']
    output = json.loads((directory / 'result.rank0.json').read_text())
    assert output['offload_status']['selected_transfer_bytes'] == 0
    outputs.append(output['states'])
    print('Finished ' + str(directory), flush=True)
comparison = dict(revision=revision, cold_vs_warm_exact=outputs[0] == outputs[1],
                  warm_repeated_exact=outputs[1] == outputs[2],
                  different_keys_by_step=[[key for key in a if a[key] != b[key]]
                                          for a, b in zip(outputs[0], outputs[1])])
(base / 'qwen-sm120-gpu0-cold-control-comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
print(json.dumps(comparison), flush=True)
assert comparison['warm_repeated_exact']
