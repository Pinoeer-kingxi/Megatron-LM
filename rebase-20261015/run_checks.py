"""Record checks for the rebased PR heads without changing their source trees."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/songbinbin/gdn-offload-artifacts/split-pr-20261015")
OUTPUT = ROOT / "rebase-20261015"
STATE = json.loads((OUTPUT / "source-after.json").read_text())
MODE = sys.argv[1]
RECORDS = []
TORCHRUN = ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1", "-m"]


def run(branch, name, arguments, *, environment=()):
    source = ROOT / branch
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() == STATE[branch]
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=source)
    python_path = ":".join(["/artifacts/causal-conv1d-python", "/artifacts/bridge-python", str(ROOT / "bridge/src"), str(source)])
    command = ["docker", "exec", "-w", str(source), "-e", "CUDA_VISIBLE_DEVICES=", "-e", "PYTHONPATH=" + python_path]
    for item in environment:
        command += ["-e", item]
    command += ["megatron-gdn-split-cpu", "/artifacts/uv-container-bin", "run", "--no-project", "--active", *arguments]
    record = {"branch": branch, "source_commit": STATE[branch], "name": name, "command": command, "started_utc": datetime.now(timezone.utc).isoformat(), "log": name + ".log", "source_clean_before_run": True}
    with (OUTPUT / record["log"]).open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, finished_utc=datetime.now(timezone.utc).isoformat(), source_clean_after_run=not bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=source)))
    RECORDS.append(record)
    (OUTPUT / (MODE + "-validation.json")).write_text(json.dumps(RECORDS, indent=2) + "\n")
    assert result.returncode == 0 and record["source_clean_after_run"], name
    print(name + " passed on " + STATE[branch], flush=True)


if MODE == "cpu":
    run("feature", "environment", ["python", "-c", "import json,torch; print(json.dumps({'cuda_devices':torch.cuda.device_count(),'torch':torch.__version__})); assert torch.cuda.device_count()==0"])
    run("shared", "shared-cpu", ["python", *TORCHRUN, "pytest", "-q", "-x", "--confcutdir=tests/unit_tests/pipeline_parallel", "tests/unit_tests/pipeline_parallel/test_schedules.py", "tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py", "-k", "test_reset_activation_offload_uses_language_model_group or test_offload_summary_uses_explicit_process_group"])
    for branch in ("gate", "feature"):
        run(branch, branch + "-manifest", ["python", *TORCHRUN, "pytest", "-q", "-x", "--confcutdir=tests/unit_tests/determinism/kernels", "tests/unit_tests/determinism/kernels/test_manifest.py"])
    run("feature", "feature-cpu", ["python", *TORCHRUN, "pytest", "-q", "-x", "--confcutdir=tests/unit_tests/ssm", "tests/unit_tests/ssm/test_gdn_activation_offload.py", "tests/unit_tests/ssm/test_gated_delta_net_offloading.py", "tests/unit_tests/ssm/test_qwen_gdn_offload_training.py", "-k", "test_snapshot_tensor_equality or test_reject_unsupported_gdn_offload or test_gdn_offload_config or test_qwen_parallel_input_boundaries"])
    run("feature", "native-training-collection", ["python", *TORCHRUN, "pytest", "--collect-only", "-q", "--confcutdir=tests/unit_tests/ssm", "tests/unit_tests/ssm/test_gdn_offload_training.py"])
    run("feature", "qwen-preflight", ["python", "tools/ssm/qwen_gdn_offload_training.py", "--dry-run", "--weights", "/artifacts/models/Qwen3.5-0.8B", "--data", "/artifacts/training-data/ks-item-sft.jsonl", "--tp", "2", "--sp", "--pp", "2", "--recompute-norm", "--check-state", "--seq-length", "128", "--min-offloaded-tensor-size", "1024", "--warmup", "1", "--iterations", "2", "--output", str(OUTPUT / "qwen-preflight.json")])
elif MODE == "format":
    for branch in ("shared", "gate", "feature"):
        run(branch, branch + "-autoformat", ["bash", "tools/autoformat.sh"], environment=("BASE_REF=main", "CHECK_ONLY=true", "SKIP_DOCS=false"))
    files = ["tools/ssm/gdn_activation_offload.py", "tools/ssm/qwen_gdn_offload_training.py", "tests/unit_tests/ssm/test_gdn_activation_offload.py", "tests/unit_tests/ssm/test_qwen_gdn_offload_training.py"]
    for tool in (["black", "--check"], ["isort", "--check"], ["pylint"], ["ruff", "check"], ["mypy", "--explicit-package-bases", "--follow-imports=skip"]):
        run("feature", "standalone-" + tool[0], [*tool, *files])
elif MODE == "docs":
    run("feature", "strict-docs", ["python", "-c", "import os; os.execv('/artifacts/split-pr-20261015/docs-venv/bin/sphinx-build', ['sphinx-build','-j','4','-b','html','-W','--keep-going','docs','/artifacts/split-pr-20261015/docs-build'])"])
else:
    raise ValueError(MODE)
