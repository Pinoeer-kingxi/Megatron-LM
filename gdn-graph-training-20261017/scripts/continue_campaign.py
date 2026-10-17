"""Keep valid Qwen controls, rerun the native GDN2 adapter, then measure both models."""

import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace

import run_campaign as campaign

root = Path(__file__).resolve().parents[1]
campaign.gpus = campaign.gpu_inventory(['1', '2'])
campaign.require_idle(campaign.gpus)
campaign.args = SimpleNamespace(
    source_root=campaign.ARTIFACTS / 'split-pr-20261015', output=root, stage='four',
    weights=campaign.ARTIFACTS / 'models/Qwen3.5-0.8B',
    data=campaign.ARTIFACTS / 'training-data/ks-item-sft.jsonl',
)
campaign.manifest = json.loads((root / 'manifest.json').read_text())
campaign.manifest.pop('error', None)
campaign.manifest.update(status='running', continued_unix=time.time(),
    script_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'scripts').glob('*.py')})
campaign.telemetry_stop = threading.Event()
monitor = threading.Thread(target=campaign.telemetry)
container = None
try:
    start = campaign.container_command(campaign.args, campaign.gpus, campaign.NAME)
    container = subprocess.check_output(start, text=True).strip()
    campaign.manifest.update(container_command=start, container_id=container)
    monitor.start()
    for offload in (False, True):
        campaign.launch('gdn2', offload, 'gdn2-correct-' + ('enabled' if offload else 'disabled'), True)
    for model in ('qwen', 'gdn2'):
        for repeat in range(3):
            for offload in ((False, True) if repeat != 1 else (True, False)):
                campaign.launch(model, offload, f'{model}-perf-{repeat}-' + ('enabled' if offload else 'disabled'))
    campaign.manifest['status'] = 'completed_pending_verification'
except BaseException as error:
    campaign.manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
    raise
finally:
    campaign.telemetry_stop.set()
    if monitor.is_alive():
        monitor.join()
    if container is not None:
        cleanup = subprocess.run(['docker', 'stop', '--time', '10', container], capture_output=True, text=True)
        campaign.manifest['cleanup'] = dict(exit_code=cleanup.returncode, stdout=cleanup.stdout, stderr=cleanup.stderr)
    campaign.manifest['finished_unix'] = time.time()
    campaign.write_manifest()
