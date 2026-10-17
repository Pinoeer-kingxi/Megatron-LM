"""Run the requested 250-step full-Qwen offload stability check on physical GPU 1."""

import json
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

from run_gpu_acceptance import (
    ARTIFACTS, VENV, bridge_state, container_command, gpu_inventory, hash_file,
    python_path, require_idle,
)

root = Path(__file__).resolve().parents[1]
source = root / "source"
source_root = ARTIFACTS / "split-pr-20261015"
gpu = gpu_inventory(["1"])[0]
name = "megatron-gdn-stability-20261016"
weights = ARTIFACTS / "models/Qwen3.5-0.8B"
data = ARTIFACTS / "training-data/ks-item-sft.jsonl"
args = SimpleNamespace(source_root=source_root, output=root, stage="four", weights=weights, data=data)
manifest = {
    "status": "running", "source": json.loads((root / "source.json").read_text()),
    "source_archive_sha256": hash_file(root / "source.tar"),
    "bridge": bridge_state(source_root), "gpu": gpu,
    "input_sha256": {str(p): hash_file(p) for p in [data, *sorted(weights.glob("*.*"))] if p.is_file()},
    "scripts_sha256": {p.name: hash_file(p) for p in (root / "scripts").iterdir() if p.is_file()},
    "scope": "Full 24-layer Qwen3.5-0.8B BF16/FLA text model; unfused, norm recomputation off",
    "gradient_check": "Native optimizer global gradient norm must be finite and positive on every step",
    "started_unix": time.time(),
}


def save():
    (root / "stability-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def interrupted(signum, _frame):
    raise KeyboardInterrupt(f"Received {signum}")


for signum in (signal.SIGTERM, signal.SIGHUP):
    signal.signal(signum, interrupted)
start = container_command(args, [gpu], name)
manifest["container_command"] = start
require_idle([gpu])
try:
    manifest["container_id"] = subprocess.check_output(start, text=True).strip()
    command = [
        "docker", "exec", "--workdir", str(source),
        "--env", "PYTHONPATH=" + python_path(source, source_root),
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", f"TRITON_CACHE_DIR={root / 'cache/stability/triton'}",
        "--env", f"TORCHINDUCTOR_CACHE_DIR={root / 'cache/stability/inductor'}",
        name, str(VENV / "bin/python"), "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node=1", "--log-dir", str(root / "stability-ranks"),
        "--redirects", "3", "--tee", "3", str(root / "scripts/rank_env.py"),
        str(root / "scripts/stability_run.py"), "--weights", str(weights),
        "--data", str(data), "--seq-length", "2048", "--microbatches", "4",
        "--warmup", "3", "--iterations", "247", "--fraction", "1",
        "--output", str(root / "stability-result.json"),
    ]
    manifest["command"] = command
    save()
    with (root / "stability-launcher.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    manifest["exit_code"] = result.returncode
    assert result.returncode == 0, "Inspect stability-launcher.log"
    rows = [json.loads(line) for line in (root / "stability-steps.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == list(range(250))
    assert all(row["pool_before_schedule_reset"]["current_in_use"] == 0 for row in rows)
    assert all(row["selected_transfer_bytes"] > 0 for row in rows)
    output = json.loads((root / "stability-result.rank0.json").read_text())
    assert output["text_layers"] == 24 and len(output["samples"]) == 247
    manifest["status"] = "passed"
except BaseException as error:
    manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
    raise
finally:
    cleanup = subprocess.run(["docker", "stop", "--time", "10", name], capture_output=True, text=True)
    manifest["cleanup"] = {"exit_code": cleanup.returncode, "stdout": cleanup.stdout, "stderr": cleanup.stderr}
    manifest["finished_unix"] = time.time()
    save()
