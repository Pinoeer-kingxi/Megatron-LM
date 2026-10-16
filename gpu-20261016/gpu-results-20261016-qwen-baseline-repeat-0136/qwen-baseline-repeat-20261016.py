"""Repeat the exact unfused Qwen baseline twice; retain all diagnostic results."""

import json
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

PACKAGE = Path('/home/songbinbin/gdn-offload-artifacts/gpu-acceptance-20261015')
sys.path.insert(0, str(PACKAGE))
import run_gpu_acceptance as runner

reference = PACKAGE.parent / 'gpu-results-20261016-four-training-0136'
output = PACKAGE.parent / 'gpu-results-20261016-qwen-baseline-repeat-0136'
args = SimpleNamespace(
    stage='four',
    source_root=PACKAGE.parent / 'split-pr-20261015',
    output=output,
    weights=PACKAGE.parent / 'models/Qwen3.5-0.8B',
    data=PACKAGE.parent / 'training-data/ks-item-sft.jsonl',
)
sources = runner.source_state(args.source_root)
bridge = runner.bridge_state(args.source_root)
gpus = runner.gpu_inventory(['0', '1', '3', '6'])
runner.require_idle(gpus)
original = json.loads((reference / 'manifest.json').read_text())
assert original['sources'] == sources and original['gpus'] == gpus
assert original['input_sha256'] == {
    path: runner.hash_file(Path(path)) for path in original['input_sha256']
}
output.mkdir()
shutil.copy2(__file__, output / Path(__file__).name)
# Reuse the completed run's persistent Triton cache. The container itself is fresh.
shutil.copytree(reference / 'cache/triton', output / 'cache/triton')
name = 'megatron-gdn-control-' + uuid.uuid4().hex[:12]
start = runner.container_command(args, gpus, name)
manifest = {
    'diagnostic_only': True,
    'sources': sources,
    'bridge': bridge,
    'gpus': gpus,
    'reference': str(reference),
    'input_sha256': original['input_sha256'],
    'container_command': start,
    'jobs': [],
    'status': 'running',
}
job = next(job for job in runner.make_jobs(args) if job['name'] == 'qwen-f0-n0-o0')
for signum in (signal.SIGTERM, signal.SIGHUP):
    signal.signal(signum, runner.stop_run)
try:
    manifest['container_id'] = subprocess.check_output(start, text=True).strip()
    for repeat in (1, 2):
        runner.require_idle(gpus)
        directory = output / f'repeat-{repeat}'
        directory.mkdir()
        command = job['command'].copy()
        command[command.index('--log-dir') + 1] = str(directory / 'ranks')
        command[command.index('--output') + 1] = str(directory / 'result.json')
        invocation = runner.job_command(args, {'source': 'feature', 'command': command}, name)
        manifest['jobs'].append({'repeat': repeat, 'command': invocation})
        (output / 'commands.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'Running exact baseline repeat {repeat}', flush=True)
        with (directory / 'launcher.log').open('w') as log:
            result = subprocess.run(invocation, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        manifest['jobs'][-1]['exit_code'] = result.returncode
        result.check_returncode()
        runner.source_state(args.source_root)
        runner.require_idle(gpus)
    comparisons = []
    for rank in range(4):
        states = [
            json.loads((directory / f'result.rank{rank}.json').read_text())['states']
            for directory in (
                reference / 'qwen-f0-n0-o0',
                reference / 'qwen-f0-n0-o1',
                output / 'repeat-1',
                output / 'repeat-2',
            )
        ]
        comparisons.append(
            {
                'rank': rank,
                'repeats_equal': states[2] == states[3],
                'repeat_matches_original_baseline': states[2] == states[0],
                'repeat_matches_original_offload': states[2] == states[1],
            }
        )
    manifest.update(status='experiments_completed', comparisons=comparisons)
except BaseException as error:
    manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
    raise
finally:
    cleanup = subprocess.run(['docker', 'rm', '--force', name], capture_output=True, text=True)
    manifest['cleanup'] = {
        'exit_code': cleanup.returncode,
        'output': cleanup.stdout + cleanup.stderr,
    }
    (output / 'commands.json').write_text(json.dumps(manifest, indent=2) + '\n')
    files = sorted(
        path
        for path in output.rglob('*')
        if path.is_file() and 'cache' not in path.relative_to(output).parts
    )
    (output / 'SHA256SUMS').write_text(
        ''.join(
            runner.hash_file(path) + '  ' + path.relative_to(output).as_posix() + '\n'
            for path in files
        )
    )
print(json.dumps(manifest['comparisons'], indent=2))
