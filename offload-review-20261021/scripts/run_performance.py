"""Check the current head's full-training memory/runtime trade-off in one matched pair."""

import ast
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = Path("/home/songbinbin/gdn-offload-artifacts/review-20261021/feature")
artifacts = Path("/home/songbinbin/gdn-offload-artifacts")
revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
gpu = "GPU-5e2e1f94-e88e-48e3-247d-328b52fc68db"
pythonpath = ":".join(map(str, [source, artifacts / "flash-attn-python", artifacts / "causal-conv1d-python", artifacts / "bridge-python", artifacts / "split-pr-20261015/bridge/src"]))
rows = []
metrics = []
for enabled in (False, True):
    directory = root / ("performance-offloaded" if enabled else "performance-disabled")
    directory.mkdir()
    command = ["docker", "exec", "--workdir", str(source), "--env", "PYTHONPATH=" + pythonpath,
               "--env", "CUDA_VISIBLE_DEVICES=" + gpu,
               "--env", "TRITON_CACHE_DIR=" + str(root / "cache/performance"),
               "megatron-offload-review-20261021", "/home/songbinbin/.venvs/megatron-gdn-offload/bin/python",
               "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
               "--log-dir", str(directory / "ranks"), "--redirects", "3",
               str(root / "scripts/rank_env.py"), str(source / "tools/ssm/qwen_gdn_offload_training.py"),
               "--weights", str(artifacts / "models/Qwen3.5-0.8B"),
               "--data", str(artifacts / "training-data/ks-item-sft.jsonl"),
               "--seq-length", "4096", "--microbatches", "4", "--warmup", "3", "--iterations", "10",
               "--output", str(directory / "result.json")]
    if enabled:
        command += ["--fraction", "1"]
    manifest = dict(revision=revision, command=command, started_unix=time.time())
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Starting " + directory.name, flush=True)
    busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
    assert gpu not in busy, "Selected benchmark GPU has an existing process."
    telemetry_command = ["nvidia-smi", "--id=" + gpu,
                         "--query-gpu=timestamp,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm",
                         "--format=csv,noheader,nounits", "--loop-ms=1000"]
    manifest["telemetry_command"] = telemetry_command
    with (directory / "launcher.log").open("w") as log, (directory / "telemetry.csv").open("w") as telemetry:
        monitor = subprocess.Popen(telemetry_command, stdout=telemetry, stderr=subprocess.STDOUT)
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        finally:
            monitor.terminate()
            monitor.wait()
    manifest.update(exit_code=result.returncode, finished_unix=time.time())
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    assert result.returncode == 0, directory
    output = json.loads((directory / "result.rank0.json").read_text())
    assert output["text_layers"] == 24 and len(output["samples"]) == 10
    assert not output["arguments"]["check_state"]
    logs = list((directory / "ranks").rglob("0/stderr.log"))
    assert len(logs) == 1
    pattern = r"INFO:root:Rank 0 step (\d+): losses=(\[.*?\]) grad_norm=(.*?) selected_transfer_bytes=(\d+)"
    values = [(int(step), ast.literal_eval(losses), norm)
              for step, losses, norm, _ in re.findall(pattern, logs[0].read_text())]
    assert [row[0] for row in values] == list(range(13))
    metrics.append(values)
    rows.append(dict(enabled=enabled,
                     median_ms=1000 * statistics.median(s["seconds"] for s in output["samples"]),
                     peak_allocated_bytes=max(s["peak_allocated_bytes"] for s in output["samples"]),
                     peak_reserved_bytes=max(s["peak_reserved_bytes"] for s in output["samples"])))
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == revision
comparison = dict(revision=revision, pairs=1, rows=rows,
                  allocated_saved_bytes=rows[0]["peak_allocated_bytes"]-rows[1]["peak_allocated_bytes"],
                  median_step_time_change_percent=100 * (rows[1]["median_ms"] / rows[0]["median_ms"] - 1),
                  exact_logged_losses_and_reported_norms_equal=metrics[0] == metrics[1],
                  limitation="One fresh-process pair; no statistical speedup or cross-version claim. Logged gradient norms have print precision; complete state checks are separate.")
(root / "performance-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
assert comparison["allocated_saved_bytes"] > 0, comparison
assert comparison["exact_logged_losses_and_reported_norms_equal"], comparison
print(json.dumps(comparison), flush=True)
