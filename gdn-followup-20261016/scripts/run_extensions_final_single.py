"""Run scope, native training and genuine TE graph tests for the frozen extension."""

import json
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

from run_gpu_acceptance import (
    ARTIFACTS, VENV, bridge_state, container_command, gpu_inventory,
    hash_file, python_path, require_idle,
)

parent = Path(__file__).resolve().parents[1]
root = parent / "extensions-final-validation"
source = root / "source"
source_root = ARTIFACTS / "split-pr-20261015"
name = "megatron-gdn-final-single-20261016"
gpu = gpu_inventory(["1"])[0]
args = SimpleNamespace(source_root=source_root, output=parent, stage="four", weights=ARTIFACTS / "models/Qwen3.5-0.8B", data=ARTIFACTS / "training-data/ks-item-sft.jsonl")
manifest = {
    "status": "running", "source": json.loads((root / "source.json").read_text()),
    "source_archive_sha256": hash_file(root / "source.tar"), "gpu": gpu,
    "bridge": bridge_state(source_root), "started_unix": time.time(), "jobs": [],
    "scripts_sha256": {p.name: hash_file(p) for p in (parent / "scripts").glob("*.py")},
}


def save():
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def interrupted(signum, _frame):
    raise KeyboardInterrupt(f"Received {signum}")


for signum in (signal.SIGTERM, signal.SIGHUP):
    signal.signal(signum, interrupted)
require_idle([gpu])
start = container_command(args, [gpu], name)
manifest["container_command"] = start
try:
    manifest["container_id"] = subprocess.check_output(start, text=True).strip()
    prefix = [
        "docker", "exec", "--workdir", str(source),
        "--env", "PYTHONPATH=" + python_path(source, source_root),
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", f"TRITON_CACHE_DIR={parent / 'extensions/cache/triton'}",
        "--env", f"TORCHINDUCTOR_CACHE_DIR={parent / 'extensions/cache/inductor'}",
        name, str(VENV / "bin/python"),
    ]
    jobs = [("graphs", "tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py", "tp1")]
    for job_name, path, selection in jobs:
        directory = root / job_name
        directory.mkdir()
        command = [*prefix, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
                   "--log-dir", str(directory / "ranks"), "--redirects", "3", "--tee", "3",
                   str(parent / "scripts/rank_pytest.py"), str(directory / "reports"),
                   "-q", "-x", "-p", "no:cacheprovider", "--confcutdir=tests/unit_tests/ssm", path]
        if selection:
            command += ["-k", selection]
        job = {"name": job_name, "command": command, "started_unix": time.time()}
        manifest["jobs"].append(job)
        save()
        print("Starting " + job_name, flush=True)
        with (directory / "launcher.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        job.update(exit_code=result.returncode, finished_unix=time.time())
        save()
        assert result.returncode == 0, f"Inspect {directory / 'launcher.log'}"
        print("Completed " + job_name, flush=True)
    manifest["status"] = "passed"
except BaseException as error:
    manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
    raise
finally:
    cleanup = subprocess.run(["docker", "stop", "--time", "10", name], capture_output=True, text=True)
    manifest["cleanup"] = {"exit_code": cleanup.returncode, "stdout": cleanup.stdout, "stderr": cleanup.stderr}
    manifest["finished_unix"] = time.time()
    save()
