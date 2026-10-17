"""Run four-rank native training and full-Qwen TP2+SP+PP2 controls."""

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
NAME = 'megatron-gdn-four-current-20261016'
GPUS = gpu_inventory(['1', '2', '3', '6'])
ARGS = SimpleNamespace(source_root=SPLIT, output=ROOT, stage='four',
                       weights=ARTIFACTS / 'models/Qwen3.5-0.8B',
                       data=ARTIFACTS / 'training-data/ks-item-sft.jsonl')
RECORD = {**json.loads((ROOT / 'source.json').read_text()), 'gpus': GPUS,
          'bridge': bridge_state(SPLIT), 'jobs': [], 'phase': 'four-rank-validation'}
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

def pytest(name, path, expected, selection=None, source=SOURCE, ranks=4, allow_skips=False):
    command = [*prefix(source, ranks == 1), NAME, *torchrun(name, ranks),
               str(ROOT / 'scripts/rank_pytest.py'), str(ROOT / name / 'reports'),
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
    model_selection = ('test_gpt_fine_grained_activation_offloading_correctness_and_memory or '
        'test_fine_grained_activation_offload_with_ep_a2a_overlap_compatibility or '
        'test_fine_grained_activation_offloading_with_cuda_graph')
    pytest('shared-groups', 'tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py',
           16, 'not (' + model_selection + ')')
    native = pytest('native-training', 'tests/unit_tests/ssm/test_gdn_offload_training.py',
                    96)
    for rank in native['ranks']:
        assert rank['passed'] == 96 and not rank['skipped']
    for fused in (False, True):
        for norm in (False, True):
            runs = []
            for arm in ('disabled', 'repeat', 'offload'):
                name = f'qwen-f{int(fused)}-n{int(norm)}-{arm}'
                command = [*prefix(cache=f'qwen-f{int(fused)}-n{int(norm)}'), NAME,
                    *torchrun(name, 4),
                    str(ROOT / 'scripts/capture_qwen_vectors.py') if fused else 'tools/ssm/qwen_gdn_offload_training.py',
                    '--weights', str(ARGS.weights), '--data', str(ARGS.data), '--tp', '2', '--sp', '--pp', '2',
                    '--seq-length', '128', '--min-offloaded-tensor-size', '1024', '--warmup', '1',
                    '--iterations', '2', '--check-state', '--output', str(ROOT / name / 'result.json')]
                if fused:
                    command += ['--fused-pre-gdr']
                if norm:
                    command += ['--recompute-norm']
                if arm == 'offload':
                    command += ['--fraction', '1']
                run(name, command)
                values = [json.loads((ROOT / name / f'result.rank{rank}.json').read_text())
                          for rank in range(4)]
                for rank, value in enumerate(values):
                    assert value['gpu_training_run'] and value['text_layers'] == 24
                    assert value['pool_verified_before_reset'] and value['rank'] == rank
                    assert [s['step'] for s in value['states']] == [0, 1, 2]
                    assert not value['samples']
                    for state in value['states']:
                        assert len(state['losses']) == (4 if rank >= 2 else 0) and math.isfinite(state['grad_norm'])
                        assert state['grad_norm'] > 0 and all(math.isfinite(x) for x in state['losses'])
                        assert all(state[field] for field in ('grads', 'weights', 'optimizer'))
                    selected = value['offload_status']['selected_transfer_bytes']
                    assert (selected > 0) if arm == 'offload' else (selected == 0)
                runs.append(values)
            comparisons = []
            for arm, values in zip(('repeat', 'offload'), runs[1:]):
                for rank in range(4):
                    if not fused:
                        assert runs[0][rank]['states'] == values[rank]['states'], (fused, norm, arm, rank)
                        comparisons.append({'arm': arm, 'rank': rank, 'steps': 3, 'exact': True})
            RECORD.setdefault('qwen_comparisons', {})[f'f{int(fused)}-n{int(norm)}'] = comparisons
            save()
    run('compare-fused-numeric', [*prefix(), NAME, str(VENV / 'bin/python'),
        str(ROOT / 'scripts/compare_qwen_numeric.py')])
    RECORD['status'] = 'passed'
except BaseException as error:
    RECORD.update(status='failed', error=f'{type(error).__name__}: {error}')
    raise
finally:
    cleanup = subprocess.run(['docker', 'stop', '--timeout', '10', NAME], capture_output=True, text=True)
    RECORD['cleanup'] = {'exit_code': cleanup.returncode, 'stdout': cleanup.stdout, 'stderr': cleanup.stderr}
    save()
