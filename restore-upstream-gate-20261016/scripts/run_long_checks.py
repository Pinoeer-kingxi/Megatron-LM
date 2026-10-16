"""Launch the sequence-2048 controls on a second idle GPU."""

import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, '/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/gpu-e2e-performance-20261016/scripts')
from run_gpu_acceptance import ARTIFACTS, VENV, container_command, gpu_inventory, python_path, require_idle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source'
SPLIT = ARTIFACTS / 'split-pr-20261015'
NAME = 'megatron-gdn-restore-long-20261016'
gpus = gpu_inventory(['6'])
args = SimpleNamespace(source_root=SPLIT, output=ROOT, stage='four',
                       weights=ARTIFACTS / 'models/Qwen3.5-0.8B',
                       data=ARTIFACTS / 'training-data/ks-item-sft.jsonl')
record = {**json.loads((ROOT / 'source.json').read_text()), 'gpus': gpus, 'jobs': []}
start = container_command(args, gpus, NAME)
record['container_command'] = start

def save():
    (ROOT / 'long-checks.json').write_text(json.dumps(record, indent=2) + '\n')

try:
    require_idle(gpus)
    record['container_id'] = subprocess.check_output(start, text=True).strip()
    save()
    prefix = ['docker', 'exec', '--workdir', str(SOURCE), '--env',
              'PYTHONPATH=' + python_path(SOURCE, SPLIT), '--env', 'PYTHONDONTWRITEBYTECODE=1',
              '--env', 'TMPDIR=/tmp', '--env', 'CAUSAL_CONV1D_DETERMINISTIC=1',
              '--env', 'NVTE_ALLOW_NONDETERMINISTIC_ALGO=0', '--env',
              'TRITON_CACHE_DIR=' + str(ROOT / 'long-cache/triton'), '--env',
              'TORCHINDUCTOR_CACHE_DIR=' + str(ROOT / 'long-cache/inductor'), NAME]
    for norm in (False, True):
        name = 'long-norm-' + str(int(norm))
        directory = ROOT / name
        directory.mkdir()
        argv = [*prefix, str(VENV / 'bin/python'), '-m', 'torch.distributed.run', '--standalone',
                '--nproc-per-node=1', '--log-dir', str(directory / 'ranks'), '--redirects', '3',
                '--tee', '3', str(ROOT / 'scripts/rank_env.py'),
                str(ROOT / 'scripts/long_fused_training.py'), '--output', str(directory / 'result.json')]
        if norm:
            argv.append('--recompute-norm')
        entry = {'name': name, 'command': argv, 'started_unix': time.time()}
        record['jobs'].append(entry)
        save()
        print('Starting ' + name, flush=True)
        with (directory / 'launcher.log').open('w') as output:
            completed = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT)
        entry.update(exit_code=completed.returncode, finished_unix=time.time())
        save()
        print(name + ': ' + str(completed.returncode), flush=True)
        if completed.returncode:
            raise RuntimeError(name)
    record['status'] = 'passed'
except BaseException as error:
    record.update(status='failed', error=f'{type(error).__name__}: {error}')
    raise
finally:
    cleanup = subprocess.run(['docker', 'stop', '--time', '10', NAME], capture_output=True, text=True)
    record['cleanup'] = {'exit_code': cleanup.returncode, 'stdout': cleanup.stdout, 'stderr': cleanup.stderr}
    save()
