"""Check the frozen PR files in the existing environment, with no visible GPUs."""

import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from run_gpu_acceptance import ARTIFACTS, VENV, container_command, python_path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
SPLIT = ARTIFACTS / "split-pr-20261015"
NAME = "megatron-gdn-e2e-review-cpu-20261016"
files = [p for p in subprocess.check_output([
    "git", "-C", "/data/songbinbin/Proj/Proj_LWX/Megatron-LM", "diff", "--name-only",
    "9131593673d037d360d252904f9198d36bc0c246", "e9975a744efad181f9ab586ce33f58e7a9649ac8",
], text=True).splitlines() if p.endswith(".py")]
args = SimpleNamespace(source_root=SPLIT, output=ROOT, stage="four",
                       weights=ARTIFACTS / "models/Qwen3.5-0.8B",
                       data=ARTIFACTS / "training-data/ks-item-sft.jsonl")
start = container_command(args, [], NAME)
record = {"source_commit": "e9975a744efad181f9ab586ce33f58e7a9649ac8",
          "gpu_used": False, "container_command": start, "files": files, "checks": []}
folder = ROOT / "cpu-checks"
folder.mkdir()
try:
    record["container_id"] = subprocess.check_output(start, text=True).strip()
    prefix = ["docker", "exec", "--workdir", str(SOURCE), "--env",
              "PYTHONPATH=" + python_path(SOURCE, SPLIT), "--env", "PYTHONDONTWRITEBYTECODE=1",
              "--env", "TMPDIR=/tmp", NAME]
    commands = [
        ("environment", [str(VENV / "bin/python"), "-c",
                         "import torch; assert torch.cuda.device_count()==0; print('CUDA devices: 0')"]),
        ("black", [str(VENV / "bin/black"), "--check", "--diff", "--workers", "1", *files]),
        ("isort", [str(VENV / "bin/isort"), "--check-only", "--diff", *files]),
        ("ruff", [str(VENV / "bin/ruff"), "check", "--no-fix", *files]),
        ("pylint", [str(VENV / "bin/pylint"), *files]),
    ]
    torchrun = [str(VENV / "bin/python"), "-m", "torch.distributed.run",
                "--master-addr=127.0.0.1", "--master-port=29593", "--nnodes=1",
                "--nproc-per-node=1", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"]
    commands.extend([
        ("feature-cpu", [*torchrun, "--confcutdir=tests/unit_tests/ssm",
            "tests/unit_tests/ssm/test_gdn_activation_offload.py",
            "tests/unit_tests/ssm/test_gated_delta_net_offloading.py",
            "tests/unit_tests/ssm/test_qwen_gdn_offload_training.py", "-k",
            "test_snapshot_tensor_equality or test_reject_unsupported_gdn_offload or test_gdn_offload_config or test_qwen_parallel_input_boundaries"]),
        ("manifest", [*torchrun, "--confcutdir=tests/unit_tests/determinism/kernels",
                       "tests/unit_tests/determinism/kernels/test_manifest.py"]),
    ])
    for name, argv in commands:
        row = {"name": name, "command": [*prefix, *argv], "started_unix": time.time()}
        with (folder / f"{name}.log").open("w") as output:
            completed = subprocess.run(row["command"], stdout=output, stderr=subprocess.STDOUT)
        row.update(exit_code=completed.returncode, finished_unix=time.time())
        record["checks"].append(row)
        print(name, completed.returncode, flush=True)
        assert completed.returncode == 0, name
    record["status"] = "passed"
except BaseException as error:
    record.update(status="failed", error=f"{type(error).__name__}: {error}")
    raise
finally:
    cleanup = subprocess.run(["docker", "stop", "--time", "10", NAME], capture_output=True, text=True)
    record["cleanup"] = {"exit_code": cleanup.returncode, "stdout": cleanup.stdout, "stderr": cleanup.stderr}
    (folder / "checks.json").write_text(json.dumps(record, indent=2) + "\n")
