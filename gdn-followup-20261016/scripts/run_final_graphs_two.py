"""Validate the final graph implementation with DP2, TP2/SP and PP2 on GPUs 1 and 2."""

import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from run_gpu_acceptance import ARTIFACTS, VENV, container_command, gpu_inventory, hash_file, python_path, require_idle

parent = Path(__file__).resolve().parents[1]
root = parent / "extensions-final-two"
root.mkdir()
source = parent / "extensions-final-validation/source"
source_root = ARTIFACTS / "split-pr-20261015"
gpus = gpu_inventory(["1", "2"])
require_idle(gpus)
name = "megatron-gdn-final-graphs-two-20261016"
args = SimpleNamespace(source_root=source_root, output=parent, stage="four", weights=ARTIFACTS / "models/Qwen3.5-0.8B", data=ARTIFACTS / "training-data/ks-item-sft.jsonl")
manifest = dict(status="running", started_unix=time.time(), source=json.loads((source.parent / "source.json").read_text()),
                source_archive_sha256=hash_file(source.parent / "source.tar"), gpus=gpus, jobs=[],
                script_sha256={p.name: hash_file(p) for p in (parent / "scripts").glob("*.py")})


def save():
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


container_id = None
try:
    command = container_command(args, gpus, name)
    manifest["container_command"] = command
    container_id = subprocess.check_output(command, text=True).strip()
    manifest["container_id"] = container_id
    command = ["docker", "exec", "--workdir", str(source),
               "--env", "PYTHONPATH=" + python_path(source, source_root),
               "--env", "PYTHONDONTWRITEBYTECODE=1",
               "--env", f"TRITON_CACHE_DIR={root / 'cache/triton'}",
               "--env", f"TORCHINDUCTOR_CACHE_DIR={root / 'cache/inductor'}",
               name, str(VENV / "bin/python"), "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
               "--log-dir", str(root / "ranks"), "--redirects", "3", "--tee", "3",
               str(parent / "scripts/rank_env.py"), str(parent / "scripts/rank_distributed_pytest.py"),
               str(root / "reports"), "-q", "-x", "-p", "no:cacheprovider", "--confcutdir=tests/unit_tests/ssm",
               "tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py"]
    job = dict(name="te-graphs", command=command, started_unix=time.time())
    manifest["jobs"].append(job)
    save()
    print("Starting 21 graph cases on both ranks", flush=True)
    with (root / "launcher.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    job.update(exit_code=result.returncode, finished_unix=time.time())
    save()
    assert result.returncode == 0, f"Inspect {root / 'launcher.log'}"
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
