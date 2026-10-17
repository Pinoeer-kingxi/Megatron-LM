"""Run the final source's native and real TE graph matrices on reserved GPUs 1 and 2."""

import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from run_gpu_acceptance import ARTIFACTS, VENV, container_command, gpu_inventory, hash_file, python_path, require_idle

parent = Path(__file__).resolve().parents[1]
root = parent / "extensions-ready"
source = root / "source"
source_root = ARTIFACTS / "split-pr-20261015"
gpus = gpu_inventory(["1", "2"])
require_idle(gpus)
name = "megatron-gdn-ready-two-20261016"
args = SimpleNamespace(source_root=source_root, output=parent, stage="four", weights=ARTIFACTS / "models/Qwen3.5-0.8B", data=ARTIFACTS / "training-data/ks-item-sft.jsonl")
manifest = dict(status="running", started_unix=time.time(), source=json.loads((root / "source.json").read_text()),
                source_archive_sha256=hash_file(root / "source.tar"), gpus=gpus, jobs=[],
                scripts_sha256={p.name: hash_file(p) for p in (parent / "scripts").glob("*.py")})


def save():
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


container_id = None
try:
    command = container_command(args, gpus, name)
    manifest["container_command"] = command
    container_id = subprocess.check_output(command, text=True).strip()
    manifest["container_id"] = container_id
    jobs = [
        ("native-training", "two-gpu", "tests/unit_tests/ssm/test_gdn_offload_training.py",
         "gdn2 and (tp_sp or pp) and not tp_sp_cp and not tp_sp_pp", False),
        ("graphs", "extensions-final-two", "tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py", "", True),
    ]
    for job_name, cache_name, path, selection, isolated in jobs:
        directory = root / job_name
        directory.mkdir()
        command = ["docker", "exec", "--workdir", str(source),
                   "--env", "PYTHONPATH=" + python_path(source, source_root),
                   "--env", "PYTHONDONTWRITEBYTECODE=1",
                   "--env", f"TRITON_CACHE_DIR={parent / cache_name / 'cache/triton'}",
                   "--env", f"TORCHINDUCTOR_CACHE_DIR={parent / cache_name / 'cache/inductor'}",
                   name, str(VENV / "bin/python"), "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
                   "--log-dir", str(directory / "ranks"), "--redirects", "3", "--tee", "3"]
        if isolated:
            command.append(str(parent / "scripts/rank_env.py"))
        command += [str(parent / "scripts/rank_distributed_pytest.py"), str(directory / "reports"),
                    "-q", "-x", "-p", "no:cacheprovider", "--confcutdir=tests/unit_tests/ssm", path]
        if selection:
            command += ["-k", selection]
        job = dict(name=job_name, command=command, started_unix=time.time())
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
    if container_id is not None:
        cleanup = subprocess.run(["docker", "stop", "--time", "10", container_id], capture_output=True, text=True)
        manifest["cleanup"] = dict(exit_code=cleanup.returncode, stdout=cleanup.stdout, stderr=cleanup.stderr)
    manifest["finished_unix"] = time.time()
    save()
