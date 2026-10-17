"""Three fresh-process GDN2 stack pairs; devices 1 and 2 must be idle at boundaries."""

import json
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace

from run_gpu_acceptance import ARTIFACTS, VENV, container_command, gpu_inventory, hash_file, python_path, require_idle

parent = Path(__file__).resolve().parents[1]
root = parent / "extensions-ready-benchmark"
root.mkdir()
source = parent / "extensions-ready/source"
source_root = ARTIFACTS / "split-pr-20261015"
gpus = gpu_inventory(["1", "2"])
require_idle(gpus)
name = "megatron-gdn-ready-benchmark-20261016"
args = SimpleNamespace(source_root=source_root, output=parent, stage="four", weights=ARTIFACTS / "models/Qwen3.5-0.8B", data=ARTIFACTS / "training-data/ks-item-sft.jsonl")
manifest = dict(status="running", started_unix=time.time(), source=json.loads((source.parent / "source.json").read_text()),
                source_archive_sha256=hash_file(source.parent / "source.tar"), gpu=gpus[1], jobs=[],
                scope="Four GDN2 blocks only: no vocabulary loss, optimizer updates or CUDA graphs; fixed baseline-then-offload order.",
                scripts_sha256={p.name: hash_file(p) for p in (parent / "scripts").glob("*.py")})


def telemetry():
    return {key: subprocess.check_output(command, text=True) for key, command in {
        "devices": ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem", "--format=csv"],
        "processes": ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv"],
    }.items()}


def save():
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


container_id = None
try:
    command = container_command(args, [gpus[1]], name)
    manifest["container_command"] = command
    container_id = subprocess.check_output(command, text=True).strip()
    manifest["container_id"] = container_id
    pairs = []
    for index in range(3):
        require_idle(gpus)
        directory = root / f"pair-{index}"
        directory.mkdir()
        command = ["docker", "exec", "--workdir", str(source),
                   "--env", "PYTHONPATH=" + python_path(source, source_root),
                   "--env", "PYTHONDONTWRITEBYTECODE=1",
                   "--env", f"TRITON_CACHE_DIR={parent / 'extensions/cache/triton'}",
                   "--env", f"TORCHINDUCTOR_CACHE_DIR={parent / 'extensions/cache/inductor'}",
                   name, str(VENV / "bin/python"), "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
                   "--log-dir", str(directory / "ranks"), "--redirects", "3", "--tee", "3",
                   "tools/ssm/gdn_activation_offload.py", "--mode", "benchmark", "--variant", "gdn2",
                   "--layers", "4", "--hidden-size", "1024", "--seq-length", "2048",
                   "--key-heads", "16", "--value-heads", "16", "--head-dim", "128",
                   "--warmup", "3", "--iterations", "5", "--fractions", "1",
                   "--output", str(directory / "result.json")]
        job = dict(name=f"pair-{index}", command=command, started_unix=time.time(), telemetry_before=telemetry())
        manifest["jobs"].append(job)
        save()
        print("Starting " + job["name"], flush=True)
        with (directory / "launcher.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        job.update(exit_code=result.returncode, finished_unix=time.time(), telemetry_after=telemetry())
        save()
        assert result.returncode == 0, f"Inspect {directory / 'launcher.log'}"
        require_idle(gpus)
        baseline, offload = json.loads((directory / "result.json").read_text())["arms"]
        assert offload["bitwise_equal"] and offload["checked_steps"] == 8
        assert offload["max_pinned_buffers_in_use_after_backward"] == 0
        pairs.append(dict(pair=index, baseline_median_ms=baseline["median_step_ms"], offload_median_ms=offload["median_step_ms"],
                          step_time_change_percent=100*(offload["median_step_ms"]/baseline["median_step_ms"]-1),
                          baseline_peak_allocated_bytes=max(baseline["peak_allocated_bytes"]),
                          offload_peak_allocated_bytes=max(offload["peak_allocated_bytes"]),
                          saved_bytes=max(baseline["peak_allocated_bytes"])-max(offload["peak_allocated_bytes"])))
    changes = [p["step_time_change_percent"] for p in pairs]
    (root / "comparison.json").write_text(json.dumps(dict(status="passed", pairs=pairs, median_step_time_change_percent=statistics.median(changes),
                                                        min_step_time_change_percent=min(changes), max_step_time_change_percent=max(changes)), indent=2) + "\n")
    manifest["status"] = "passed"
except BaseException as error:
    manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
    raise
finally:
    if container_id is not None:
        cleanup = subprocess.run(["docker", "stop", "--time", "10", container_id], capture_output=True, text=True)
        manifest["cleanup"] = dict(exit_code=cleanup.returncode, stdout=cleanup.stdout, stderr=cleanup.stderr)
    manifest["finished_unix"] = time.time()
    save()
