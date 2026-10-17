"""Retain the exact-comparison failure and qualify native fusion numerically."""

import json
from pathlib import Path
import subprocess
import time

from run_gpu_acceptance import require_idle
from verify_results import verify_pytest

ROOT = Path(__file__).resolve().parents[1]
initial = json.loads((ROOT / 'validation.json').read_text())
NAME = 'megatron-gdn-current-validation-20261016'
GPUS = initial['gpus']
PYTHON = '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python'
record = dict(source_commit=initial['source_commit'], gpus=GPUS, status='running',
              rtol=1e-5, atol=1e-8, initial_exact_failure='validation.json', jobs=[])


def save():
    (ROOT / 'fused-numeric.json').write_text(json.dumps(record, indent=2) + '\n')


def run(name, command):
    directory = ROOT / name
    directory.mkdir()
    job = dict(name=name, command=command, started_unix=time.time())
    record['jobs'].append(job)
    save()
    print('Starting ' + name, flush=True)
    with (directory / 'launcher.log').open('w') as output:
        process = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
    job.update(exit_code=process.returncode, finished_unix=time.time())
    save()
    print(name + ': ' + str(process.returncode), flush=True)
    assert process.returncode == 0, name


try:
    require_idle(GPUS)
    record['container_command'] = initial['container_command']
    record['container_id'] = subprocess.check_output(record['container_command'], text=True).strip()
    save()
    original = next(j['command'] for j in initial['jobs'] if j['name'] == 'qwen-f1-n0-disabled')
    for norm in (0, 1):
        for arm in ('disabled', 'repeat', 'offload'):
            name = f'numeric-f1-n{norm}-{arm}'
            command = list(original)
            command[command.index('--log-dir') + 1] = str(ROOT / name / 'ranks')
            command[command.index('--output') + 1] = str(ROOT / name / 'result.json')
            command[command.index('tools/ssm/qwen_gdn_offload_training.py')] = str(ROOT / 'scripts/capture_qwen_vectors.py')
            command = [part.replace('cache/qwen-f1-n0', f'cache/qwen-f1-n{norm}') for part in command]
            if norm:
                command += ['--recompute-norm']
            if arm == 'offload':
                command += ['--fraction', '1']
            run(name, command)
    command = original[:original.index(NAME) + 1] + [PYTHON, str(ROOT / 'scripts/compare_qwen_numeric.py')]
    run('compare-fused-numeric', command)
    # Reuse the rank-report launcher with the independent gate source and one GPU.
    original_shared = list(initial['jobs'][0]['command'])
    for index, part in enumerate(original_shared):
        if part == str(ROOT / 'source'):
            original_shared[index] = str(ROOT / 'gate')
        elif part.startswith('PYTHONPATH='):
            original_shared[index] = part.replace(str(ROOT / 'source'), str(ROOT / 'gate'), 1)
        elif part.startswith('CUDA_VISIBLE_DEVICES='):
            original_shared[index] = 'CUDA_VISIBLE_DEVICES=' + GPUS[0]['uuid']
        elif part == '--nproc-per-node=2':
            original_shared[index] = '--nproc-per-node=1'
        elif part == str(ROOT / 'shared-groups/ranks'):
            original_shared[index] = str(ROOT / 'gate-replay/ranks')
        elif part == str(ROOT / 'shared-groups/reports'):
            original_shared[index] = str(ROOT / 'gate-replay/reports')
        elif part.startswith('--confcutdir='):
            original_shared[index] = '--confcutdir=tests/unit_tests/determinism/kernels'
        elif part == 'tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py':
            original_shared[index] = 'tests/unit_tests/determinism/kernels/test_gdn_gate_backward.py'
    original_shared = original_shared[:original_shared.index('-k')]
    run('gate-replay', original_shared)
    record['gate_results'] = verify_pytest(ROOT / 'gate-replay/reports', 1, 16)
    record['status'] = 'passed'
except BaseException as error:
    record.update(status='failed', error=f'{type(error).__name__}: {error}')
    raise
finally:
    cleanup = subprocess.run(['docker', 'stop', '--timeout', '10', NAME], capture_output=True, text=True)
    record['cleanup'] = dict(exit_code=cleanup.returncode, stdout=cleanup.stdout, stderr=cleanup.stderr)
    save()
