"""Measure matched, fresh-process Qwen optimizer steps at one frozen source revision."""

import csv
import json
import math
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

from run_gpu_acceptance import (
    ARTIFACTS, VENV, autotune_cache_state, bridge_state, container_command,
    gpu_inventory, hash_file, python_path, require_idle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
REVISION = "87c22d1abae13abf375e8ed4ddfe62301cde87f4"
SOURCE_ROOT = ARTIFACTS / "split-pr-20261015"
WEIGHTS = ARTIFACTS / "models/Qwen3.5-0.8B"
DATA = ARTIFACTS / "training-data/ks-item-sft.jsonl"
NAME = "megatron-gdn-native-perf-20261016"
PYTHON = str(VENV / "bin/python")


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def process_rows():
    output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    return [row for row in csv.reader(output.splitlines(), skipinitialspace=True)
            if row and row[0] == GPU["uuid"]]


def run_job(name, command):
    require_idle([GPU])
    directory = ROOT / name
    directory.mkdir()
    job = {"name": name, "command": command, "started_unix": time.time()}
    MANIFEST["jobs"].append(job)
    save(ROOT / "manifest.json", MANIFEST)
    print("Starting " + name, flush=True)
    samples, external = [], []
    with (directory / "launcher.log").open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        while process.poll() is None:
            rows = process_rows()
            for row in rows:
                cgroup = Path(f"/proc/{row[1]}/cgroup")
                if cgroup.exists() and MANIFEST["container_id"] not in cgroup.read_text():
                    external.append(row)
            telemetry = subprocess.check_output([
                "nvidia-smi", "-i", GPU["uuid"],
                "--query-gpu=temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ], text=True).strip()
            samples.append({"unix": time.time(), "processes": rows, "telemetry": telemetry})
            if time.time() - job["started_unix"] > 1200:
                raise TimeoutError(name)
            time.sleep(1)
    job.update(exit_code=process.returncode, finished_unix=time.time())
    save(directory / "occupancy.json", {"gpu": GPU, "samples": samples, "external": external})
    save(ROOT / "manifest.json", MANIFEST)
    assert process.returncode == 0, f"{name}: inspect launcher.log"
    assert samples and any(s["processes"] for s in samples)
    assert not external, f"{name}: GPU was shared; timings are not qualified"
    result = json.loads((directory / "result.rank0.json").read_text())
    assert result["gpu_training_run"] and len(result["samples"]) == 10 and not result["states"]
    assert result["text_layers"] == 24
    args = result["arguments"]
    assert (args["seq_length"], args["microbatches"], args["warmup"], args["iterations"]) == (4096, 4, 3, 10)
    assert not args["check_state"] and not args["recompute_norm"]
    assert args["min_offloaded_tensor_size"] == 1048576
    selected = result["offload_status"]["selected_transfer_bytes"]
    assert (selected == 0) if args["fraction"] is None else (selected > 0)
    assert all(math.isfinite(s["seconds"]) and s["seconds"] > 0 for s in result["samples"])
    print("Passed " + name, flush=True)
    return result


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Received {signum}")


GPU = gpu_inventory(["1"])[0]
MANIFEST = {
    "status": "running", "source_commit": REVISION, "source_archive_sha256": hash_file(ROOT / "source.tar"),
    "bridge": bridge_state(SOURCE_ROOT), "gpu": GPU,
    "measurement": "Complete native Qwen text optimizer steps; no state snapshots in timed runs",
    "seed": 31, "sequence": 4096, "microbatches": 4, "warmup": 3,
    "measured_steps": 10, "fresh_process_repeats": 3, "norm_recomputation": False,
    "arms": ["offload disabled", "fraction 1"],
    "cache_policy": "one persistent rank-0 cache per fusion setting across matched arms and repeats",
    "input_sha256": {str(p): hash_file(p) for p in [DATA, *sorted(WEIGHTS.glob("*.*"))] if p.is_file()},
    "script_sha256": {p.name: hash_file(p) for p in sorted((ROOT / "scripts").glob("*")) if p.is_file()},
    "jobs": [],
}
for signum in (signal.SIGTERM, signal.SIGHUP):
    signal.signal(signum, interrupted)
args = SimpleNamespace(source_root=SOURCE_ROOT, output=ROOT, stage="four", weights=WEIGHTS, data=DATA)
start = container_command(args, [GPU], NAME)
MANIFEST["container_command"] = start
require_idle([GPU])
try:
    MANIFEST["container_id"] = subprocess.check_output(start, text=True).strip()
    prefix = ["docker", "exec", "--workdir", str(SOURCE), "--env", "PYTHONPATH=" + python_path(SOURCE, SOURCE_ROOT),
              "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", f"TMPDIR={ROOT / 'tmp'}"]
    (ROOT / "tmp").mkdir()
    probe = """import json, os, importlib.metadata as m, torch, megatron.core
assert torch.cuda.device_count() == 1
print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
 'megatron_core': megatron.core.__file__, 'gpu': torch.cuda.get_device_name(),
 'packages': {n:m.version(n) for n in ('transformer-engine','flash-linear-attention','causal-conv1d','flash-attn','triton')},
 'environment': {k:os.environ.get(k) for k in ('CUDA_DEVICE_MAX_CONNECTIONS','NVTE_CPU_OFFLOAD_V1','TORCH_COMPILE_DISABLE','OMP_NUM_THREADS')}}))
"""
    probe_command = [*prefix, NAME, PYTHON, "-c", probe]
    MANIFEST["environment_command"] = probe_command
    with (ROOT / "environment.json").open("w") as stdout, (ROOT / "environment.stderr.log").open("w") as stderr:
        subprocess.run(probe_command, stdout=stdout, stderr=stderr, check=True)
    results = {}
    for fused in (False, True):
        cache = ROOT / f"cache/fused-{int(fused)}"
        for repeat in range(3):
            order = (False, True) if repeat != 1 else (True, False)
            for offload in order:
                name = f"f{int(fused)}-r{repeat}-o{int(offload)}"
                directory = ROOT / name
                command = [*prefix, "--env", f"TRITON_CACHE_DIR={cache / 'triton'}",
                           "--env", f"TORCHINDUCTOR_CACHE_DIR={cache / 'inductor'}", NAME, PYTHON,
                           "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
                           "--log-dir", str(directory / "ranks"), "--redirects", "3", "--tee", "3",
                           str(ROOT / "scripts/rank_env.py"), "tools/ssm/qwen_gdn_offload_training.py",
                           "--weights", str(WEIGHTS), "--data", str(DATA), "--seq-length", "4096",
                           "--microbatches", "4", "--warmup", "3", "--iterations", "10",
                           "--output", str(directory / "result.json")]
                if offload:
                    command += ["--fraction", "1"]
                if fused:
                    command += ["--fused-pre-gdr"]
                results[name] = run_job(name, command)
                save(ROOT / f"{name}/autotune.json", autotune_cache_state(cache / "triton"))
    MANIFEST["status"] = "measurements_completed"
except BaseException as error:
    MANIFEST.update(status="failed", error=f"{type(error).__name__}: {error}")
    raise
finally:
    cleanup = subprocess.run(["docker", "stop", "--time", "10", NAME], capture_output=True, text=True)
    MANIFEST["cleanup"] = {"exit_code": cleanup.returncode, "stdout": cleanup.stdout, "stderr": cleanup.stderr}
    MANIFEST["finished_unix"] = time.time()
    save(ROOT / "manifest.json", MANIFEST)
