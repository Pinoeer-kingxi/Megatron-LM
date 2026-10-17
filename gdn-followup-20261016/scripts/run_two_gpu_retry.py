"""Run the targeted two-GPU extension tests and one full-Qwen TP2/SP restart pair."""

import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from run_gpu_acceptance import (
    ARTIFACTS, VENV, container_command, gpu_inventory, hash_file, python_path, require_idle,
)

parent = Path(__file__).resolve().parents[1]
root = parent / "two-gpu-retry"
root.mkdir()
extension = parent / "extensions-gdn2-probe/source"
checkpoint_source = parent / "checkpoint/source"
source_root = ARTIFACTS / "split-pr-20261015"
gpus = gpu_inventory(["1", "2"])
require_idle(gpus)
name = "megatron-gdn-followup-two-retry-20261016"
args = SimpleNamespace(source_root=source_root, output=parent, stage="four", weights=ARTIFACTS / "models/Qwen3.5-0.8B", data=ARTIFACTS / "training-data/ks-item-sft.jsonl")
manifest = {"status": "running", "gpus": gpus, "started_unix": time.time(), "jobs": [],
            "extension": json.loads((extension.parent / "source.json").read_text()),
            "checkpoint": json.loads((checkpoint_source.parent / "source.json").read_text()),
            "script_sha256": {p.name: hash_file(p) for p in (parent / "scripts").glob("*.py")}}


def save():
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def execute(job_name, source, arguments, *, pytest=False):
    directory = root / job_name
    directory.mkdir()
    command = ["docker", "exec", "--workdir", str(source),
               "--env", "PYTHONPATH=" + python_path(source, source_root),
               "--env", "PYTHONDONTWRITEBYTECODE=1",
               "--env", f"TRITON_CACHE_DIR={parent / 'two-gpu/cache/triton'}",
               "--env", f"TORCHINDUCTOR_CACHE_DIR={parent / 'two-gpu/cache/inductor'}",
               name, str(VENV / "bin/python"), "-m", "torch.distributed.run", "--standalone",
               "--nproc-per-node=2", "--log-dir", str(directory / "ranks"), "--redirects", "3", "--tee", "3"]
    if pytest:
        command += [str(parent / "scripts/rank_distributed_pytest.py"), str(directory / "reports"),
                    "-q", "-x", "-p", "no:cacheprovider", "--confcutdir=tests/unit_tests/ssm"]
    else:
        command += [str(parent / "scripts/rank_env.py"), str(parent / "scripts/checkpoint_capture.py")]
    command += arguments
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


command = container_command(args, gpus, name)
manifest["container_command"] = command
try:
    manifest["container_id"] = subprocess.check_output(command, text=True).strip()
    execute("gdn2-training", extension,
            ["tests/unit_tests/ssm/test_gdn_offload_training.py", "-k",
             "gdn2 and (tp_sp or pp) and not tp_sp_cp and not tp_sp_pp"], pytest=True)
    for arm, count in [("continuous", 6), ("prefix", 2), ("resumed", 4)]:
        job_name = f"f0-n1-o1-{arm}"
        arguments = ["--weights", str(args.weights), "--data", str(args.data),
                     "--tp", "2", "--sp", "--recompute-norm", "--fraction", "1",
                     "--seq-length", "128", "--microbatches", "4", "--warmup", "1",
                     "--iterations", str(count-1), "--min-offloaded-tensor-size", "1024",
                     "--check-state", "--output", str(root / job_name / "result.json")]
        if arm == "prefix":
            arguments += ["--save-checkpoint", str(Path("/dev/shm/codex-gdn-followup-20261016/two-gpu-native-checkpoint"))]
        elif arm == "resumed":
            arguments += ["--load-checkpoint", str(Path("/dev/shm/codex-gdn-followup-20261016/two-gpu-native-checkpoint"))]
        execute(job_name, checkpoint_source, arguments)
    comparison = ["docker", "exec", name, str(VENV / "bin/python"),
                  str(parent / "scripts/compare_checkpoint.py"), str(root), "2", "--config", "f0-n1-o1"]
    manifest["comparison_command"] = comparison
    with (root / "comparison.log").open("w") as log:
        result = subprocess.run(comparison, stdout=log, stderr=subprocess.STDOUT)
    manifest["comparison_exit_code"] = result.returncode
    assert result.returncode == 0
    manifest["status"] = "passed"
    manifest["cuda_graphs"] = "Not run in this stage; separate graph integration remains under diagnosis."
except BaseException as error:
    manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
    raise
finally:
    cleanup = subprocess.run(["docker", "stop", "--time", "10", name], capture_output=True, text=True)
    manifest["cleanup"] = {"exit_code": cleanup.returncode, "stdout": cleanup.stdout, "stderr": cleanup.stderr}
    manifest["finished_unix"] = time.time()
    save()
