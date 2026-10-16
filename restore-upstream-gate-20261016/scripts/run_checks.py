"""Validate the restoration in a frozen source snapshot using existing tests."""

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
NAME = 'megatron-gdn-restore-gate-20261016'
gpu = gpu_inventory(['3'])
args = SimpleNamespace(source_root=SPLIT, output=ROOT, stage='four',
                       weights=ARTIFACTS / 'models/Qwen3.5-0.8B',
                       data=ARTIFACTS / 'training-data/ks-item-sft.jsonl')
record = {**json.loads((ROOT / 'source.json').read_text()), 'gpus': gpu, 'jobs': []}
command = container_command(args, gpu, NAME)
record['container_command'] = command

def save():
    (ROOT / 'checks.json').write_text(json.dumps(record, indent=2) + '\n')

def run(name, argv):
    directory = ROOT / name
    directory.mkdir()
    entry = {'name': name, 'command': argv, 'started_unix': time.time()}
    record['jobs'].append(entry)
    save()
    print('Starting ' + name, flush=True)
    with (directory / 'launcher.log').open('w') as output:
        result = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT)
    entry.update(exit_code=result.returncode, finished_unix=time.time())
    save()
    print(name + ': ' + str(result.returncode), flush=True)
    return result.returncode

def pytest(name, files, selection=None, cpu=False):
    directory = ROOT / name
    runner = [str(VENV / 'bin/python'), '-m', 'torch.distributed.run', '--standalone',
              '--nproc-per-node=1', '--log-dir', str(directory / 'ranks'),
              '--redirects', '3', '--tee', '3']
    if cpu:
        runner += ['-m', 'pytest']
    else:
        runner += [str(ROOT / 'scripts/rank_pytest.py'), str(directory / 'reports')]
    runner += ['-q', '-x', '-p', 'no:cacheprovider', '--confcutdir=' + str(Path(files[0]).parent), *files]
    if selection:
        runner += ['-k', selection]
    return run(name, [*prefix, '--env', 'CUDA_VISIBLE_DEVICES=' + ('' if cpu else gpu[0]['uuid']), NAME, *runner])

try:
    require_idle(gpu)
    record['container_id'] = subprocess.check_output(command, text=True).strip()
    save()
    prefix = ['docker', 'exec', '--workdir', str(SOURCE), '--env',
              'PYTHONPATH=' + python_path(SOURCE, SPLIT), '--env', 'PYTHONDONTWRITEBYTECODE=1',
              '--env', 'TMPDIR=/tmp']
    statuses = []
    statuses.append(pytest('gdn-scopes', ['tests/unit_tests/ssm/test_gated_delta_net_offloading.py']))
    statuses.append(pytest('native-dp', ['tests/unit_tests/ssm/test_gdn_offload_training.py'], 'dp'))
    statuses.append(pytest('cpu-feature', [
        'tests/unit_tests/ssm/test_gdn_activation_offload.py',
        'tests/unit_tests/ssm/test_gated_delta_net_offloading.py',
        'tests/unit_tests/ssm/test_qwen_gdn_offload_training.py'],
        'test_snapshot_tensor_equality or test_reject_unsupported_gdn_offload or test_gdn_offload_config or test_qwen_parallel_input_boundaries', cpu=True))
    statuses.append(pytest('manifest', ['tests/unit_tests/determinism/kernels/test_manifest.py'], cpu=True))
    files = [p for p in subprocess.check_output([
        'git', '-C', '/data/songbinbin/Proj/Proj_LWX/Megatron-LM', 'diff', '--name-only',
        '--diff-filter=ACM', '9131593673d037d360d252904f9198d36bc0c246', record['source_commit']], text=True).splitlines()
        if p.endswith('.py')]
    for name, argv in [
        ('black', [str(VENV / 'bin/black'), '--check', '--diff', '--workers', '1', *files]),
        ('isort', [str(VENV / 'bin/isort'), '--check-only', '--diff', *files]),
        ('ruff', [str(VENV / 'bin/ruff'), 'check', '--no-fix', *files]),
        ('pylint', [str(VENV / 'bin/pylint'), *files]),
    ]:
        statuses.append(run(name, [*prefix, '--env', 'CUDA_VISIBLE_DEVICES=', NAME, *argv]))
    record['status'] = 'passed' if all(status == 0 for status in statuses) else 'failed'
except BaseException as error:
    record.update(status='failed', error=f'{type(error).__name__}: {error}')
    raise
finally:
    cleanup = subprocess.run(['docker', 'stop', '--time', '10', NAME], capture_output=True, text=True)
    record['cleanup'] = {'exit_code': cleanup.returncode, 'stdout': cleanup.stdout, 'stderr': cleanup.stderr}
    save()
