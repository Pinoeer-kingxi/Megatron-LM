"""Run the current-source GDN scope, alias and frozen-gradient regressions."""

import json
import math
from pathlib import Path
import subprocess
import sys
import signal
import time
from types import SimpleNamespace

from run_gpu_acceptance import (ARTIFACTS, VENV, bridge_state, container_command,
                                gpu_inventory, python_path, require_idle)
from verify_results import verify_pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source'
SPLIT = ARTIFACTS / 'split-pr-20261015'
NAME = 'megatron-gdn-scope-current-20261016'
GPUS = gpu_inventory(['6'])
ARGS = SimpleNamespace(source_root=SPLIT, output=ROOT, stage='single',
                       weights=ARTIFACTS / 'models/Qwen3.5-0.8B',
                       data=ARTIFACTS / 'training-data/ks-item-sft.jsonl')
RECORD = {**json.loads((ROOT / 'source.json').read_text()), 'gpus': GPUS,
          'bridge': bridge_state(SPLIT), 'jobs': [], 'phase': 'single-rank-scope-validation'}
START = container_command(ARGS, GPUS, NAME)
RECORD['container_command'] = START

def save():
    (ROOT / 'validation.json').write_text(json.dumps(RECORD, indent=2) + '\n')

def run(name, command):
    folder = ROOT / name
    folder.mkdir()
    row = {'name': name, 'command': command, 'started_unix': time.time()}
    RECORD['jobs'].append(row)
    save()
    print('Starting ' + name, flush=True)
    with (folder / 'launcher.log').open('w') as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
    row.update(exit_code=result.returncode, finished_unix=time.time())
    save()
    print(name + ': ' + str(result.returncode), flush=True)
    if result.returncode:
        raise RuntimeError(name + ': inspect retained launcher/rank logs')

def prefix(source=SOURCE, single=False, cache='validation'):
    return ['docker', 'exec', '--workdir', str(source), '--env',
            'PYTHONPATH=' + python_path(source, SPLIT), '--env', 'PYTHONDONTWRITEBYTECODE=1',
            '--env', 'TMPDIR=/tmp', '--env',
            'CUDA_VISIBLE_DEVICES=' + ','.join(g['uuid'] for g in GPUS[:1] if single)
                if single else 'CUDA_VISIBLE_DEVICES=' + ','.join(g['uuid'] for g in GPUS),
            '--env', 'TRITON_CACHE_DIR=' + str(ROOT / 'cache' / cache / 'triton'),
            '--env', 'TORCHINDUCTOR_CACHE_DIR=' + str(ROOT / 'cache' / cache / 'inductor')]

def torchrun(name, ranks):
    return [str(VENV / 'bin/python'), '-m', 'torch.distributed.run', '--standalone',
            '--nproc-per-node=' + str(ranks), '--log-dir', str(ROOT / name / 'ranks'),
            '--redirects', '3', '--tee', '3', str(ROOT / 'scripts/rank_env.py')]

def pytest(name, path, expected, selection=None, source=SOURCE, ranks=1, allow_skips=False):
    command = [*prefix(source, ranks == 1), NAME, *torchrun(name, ranks),
               str(ROOT / ('scripts/rank_native_report.py' if name == 'native-regression' else 'scripts/rank_pytest.py')), str(ROOT / name / 'reports'),
               '-q', '-x', '-p', 'no:cacheprovider', '--confcutdir=' + str(Path(path).parent), path]
    if selection:
        command += ['-k', selection]
    run(name, command)
    result = verify_pytest(ROOT / name / 'reports', ranks, expected, allow_skips)
    RECORD.setdefault('pytest_results', {})[name] = result
    save()
    return result

def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Received {signum}')

for signum in (signal.SIGTERM, signal.SIGHUP):
    signal.signal(signum, interrupted)

try:
    require_idle(GPUS)
    RECORD['container_id'] = subprocess.check_output(START, text=True).strip()
    save()
    pytest('gdn-scope', 'tests/unit_tests/ssm/test_gated_delta_net_offloading.py', 73)
    RECORD['status'] = 'passed'
except BaseException as error:
    RECORD.update(status='failed', error=f'{type(error).__name__}: {error}')
    raise
finally:
    cleanup = subprocess.run(['docker', 'stop', '--timeout', '10', NAME], capture_output=True, text=True)
    RECORD['cleanup'] = dict(exit_code=cleanup.returncode, stdout=cleanup.stdout, stderr=cleanup.stderr)
    save()
