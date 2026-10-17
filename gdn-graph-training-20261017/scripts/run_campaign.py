"""Run pinned-source graph training controls and paired performance measurements."""

import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

OLD = Path('/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/gdn-followup-20261016/scripts')
sys.path.insert(0, str(OLD))
from run_gpu_acceptance import (
    ARTIFACTS, IMAGE, VENV, bridge_state, container_command, gpu_inventory, python_path, require_idle,
)

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/data/songbinbin/Proj/Proj_LWX/Megatron-LM-gdn-extensions')
REVISION = 'f19ebc5b1da1c9b0075eecfd2f93fc6ca4b3ba8d'
SOURCE = ROOT / 'source'
NAME = 'megatron-graph-training-20261017'


def write_manifest():
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def launch(model, offload, label, check=False):
    gpu = gpus[0 if model == 'qwen' else 1]
    directory = ROOT / label
    directory.mkdir()
    cache = ROOT / 'cache' / model
    command = [
        'docker', 'exec', '--workdir', str(SOURCE),
        '--env', 'PYTHONPATH=' + python_path(SOURCE, args.source_root),
        '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--env', 'CUDA_VISIBLE_DEVICES=' + gpu['uuid'],
        '--env', 'TRITON_CACHE_DIR=' + str(cache / 'triton'),
        '--env', 'TORCHINDUCTOR_CACHE_DIR=' + str(cache / 'inductor'),
        NAME, str(VENV / 'bin/python'), '-m', 'torch.distributed.run', '--standalone',
        '--nproc-per-node=1', '--log-dir', str(directory / 'ranks'), '--redirects', '3', '--tee', '3',
        str(ROOT / 'scripts/benchmark_graph_training.py'), '--model', model,
        '--weights', str(args.weights), '--data', str(args.data),
        '--output', str(directory / 'result.json'),
    ]
    if offload:
        command.append('--offload')
    if check:
        command += ['--check-state', '--warmup', '1', '--iterations', '2']
    job = dict(label=label, model=model, offload=offload, correctness=check,
               gpu=gpu, command=command, started_unix=time.time(),
               benchmark_script_sha256=hashlib.sha256((ROOT / 'scripts/benchmark_graph_training.py').read_bytes()).hexdigest())
    manifest['jobs'].append(job)
    write_manifest()
    print('Starting ' + label, flush=True)
    with (directory / 'launcher.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    job.update(exit_code=result.returncode, finished_unix=time.time())
    write_manifest()
    assert result.returncode == 0, f'Inspect {directory / "launcher.log"}'
    print('Finished ' + label, flush=True)


def telemetry():
    with (ROOT / 'telemetry.csv').open('w') as output:
        while not telemetry_stop.is_set():
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu',
                '--format=csv,noheader',
            ], stdout=output, stderr=subprocess.DEVNULL)
            output.flush()
            telemetry_stop.wait(1)


def stop(signum, frame):
    raise KeyboardInterrupt(f'Received signal {signum}')


if __name__ == '__main__':
    import threading

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop)
    archive = ROOT / 'source.tar'
    with archive.open('wb') as output:
        subprocess.run(['git', '-C', str(REPO), 'archive', REVISION], stdout=output, check=True)
    SOURCE.mkdir(exist_ok=True)
    with tarfile.open(archive) as payload:
        payload.extractall(SOURCE, filter='fully_trusted')
    gpus = gpu_inventory(['1', '2'])
    require_idle(gpus)
    args = SimpleNamespace(source_root=ARTIFACTS / 'split-pr-20261015', output=ROOT, stage='four',
                           weights=ARTIFACTS / 'models/Qwen3.5-0.8B',
                           data=ARTIFACTS / 'training-data/ks-item-sft.jsonl')
    manifest = dict(status='running', source=REVISION,
                    core_tree=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', REVISION + ':megatron/core'], text=True).strip(),
                    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    script_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'scripts').glob('*.py')},
                    image=IMAGE, bridge=bridge_state(args.source_root), gpus=gpus,
                    started_unix=time.time(), jobs=[],
                    timing_boundary='zero_grad, native fwd/bwd/grad finalization, BF16 Adam step and CUDA completion; excludes inputs, graph capture, state snapshots, reporting',
                    replay_warmup=3, timed_steps=10, repeats=3, fraction=1,
                    tensor_threshold_elements=1048576, pair_order=['disabled-enabled', 'enabled-disabled', 'disabled-enabled'])
    write_manifest()
    container = None
    telemetry_stop = threading.Event()
    monitor = threading.Thread(target=telemetry)
    try:
        start = container_command(args, gpus, NAME)
        manifest['container_command'] = start
        container = subprocess.check_output(start, text=True).strip()
        manifest['container_id'] = container
        monitor.start()
        for model in ('qwen', 'gdn2'):
            for offload in (False, True):
                launch(model, offload, model + '-correct-' + ('enabled' if offload else 'disabled'), True)
        for model in ('qwen', 'gdn2'):
            for repeat in range(3):
                for offload in ((False, True) if repeat != 1 else (True, False)):
                    launch(model, offload, f'{model}-perf-{repeat}-' + ('enabled' if offload else 'disabled'))
        manifest['status'] = 'completed_pending_verification'
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        telemetry_stop.set()
        if monitor.is_alive():
            monitor.join()
        if container is not None:
            cleanup = subprocess.run(['docker', 'stop', '--time', '10', container], capture_output=True, text=True)
            manifest['cleanup'] = dict(exit_code=cleanup.returncode, stdout=cleanup.stdout, stderr=cleanup.stderr)
        manifest['finished_unix'] = time.time()
        write_manifest()
