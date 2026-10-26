"""Matched full-Qwen measurements on a pinned source; retain all raw observations."""

import ast
import hashlib
import json
import os
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path


root = Path(__file__).resolve().parents[1]
source = Path(os.environ.get("PERF_SOURCE", "/home/songbinbin/gdn-offload-artifacts/scope-20261022/feature"))
container = "megatron-gdn-final-20261026"
artifacts = Path("/home/songbinbin/gdn-offload-artifacts")
gpu = "GPU-df9e07cc-8ebc-3204-fabd-601044f800f8"
revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
pythonpath = ":".join(map(str, [source, root / "scripts", artifacts / "flash-attn-python",
                              artifacts / "causal-conv1d-python", artifacts / "bridge-python",
                              artifacts / "split-pr-20261015/bridge/src"]))
records = []


def monitor(stop, path):
    with path.open("w") as handle:
        while not stop.is_set():
            metrics = subprocess.run(
                ["nvidia-smi", "--id=" + gpu,
                 "--query-gpu=timestamp,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
            )
            processes = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
            )
            rows = [line for line in processes.stdout.splitlines() if line.startswith(gpu)]
            handle.write(json.dumps(dict(time=time.time(), gpu=metrics.stdout.strip(), processes=rows)) + "\n")
            handle.flush()
            stop.wait(1)


for fused in (False, True):
    for repetition in range(3):
        order = (False, True) if repetition % 2 == 0 else (True, False)
        for enabled in order:
            name = f"performance-f{int(fused)}-pair{repetition}-{'on' if enabled else 'off'}"
            directory = root / name
            directory.mkdir()
            assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
            command = [
                "docker", "exec", "--workdir", str(source), "--env", "PYTHONPATH=" + pythonpath,
                "--env", "CUDA_VISIBLE_DEVICES=" + gpu,
                "--env", "TRITON_CACHE_DIR=" + str(root / f"cache/performance-f{int(fused)}"),
                container, "/home/songbinbin/.venvs/megatron-gdn-offload/bin/python",
                "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
                "--log-dir", str(directory / "ranks"), "--redirects", "3",
                str(root / "scripts/rank_env.py"), str(root / "scripts/qwen_resources.py"),
                "--weights", str(artifacts / "models/Qwen3.5-0.8B"),
                "--data", str(artifacts / "training-data/ks-item-sft.jsonl"),
                "--seq-length", "4096", "--microbatches", "4", "--warmup", "3", "--iterations", "10",
                "--output", str(directory / "result.json"),
            ]
            if fused:
                command.append("--fused-pre-gdr")
            if enabled:
                command += ["--fraction", "1"]
            manifest = dict(revision=revision, command=command, started_unix=time.time(),
                            harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            resource_observer_sha256=hashlib.sha256((root / "scripts/qwen_resources.py").read_bytes()).hexdigest())
            path = directory / "manifest.json"
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            print("Starting " + name + " at " + revision, flush=True)
            stop = threading.Event()
            worker = threading.Thread(target=monitor, args=(stop, directory / "telemetry.jsonl"))
            worker.start()
            try:
                with (directory / "launcher.log").open("w") as handle:
                    result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            finally:
                stop.set()
                worker.join()
            manifest.update(exit_code=result.returncode, finished_unix=time.time(),
                            revision_after=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
                            dirty_after=subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True))
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            assert result.returncode == 0, name
            assert manifest["revision_after"] == revision and not manifest["dirty_after"]
            output = json.loads((directory / "result.rank0.json").read_text())
            resources = json.loads((directory / "resources.rank0.json").read_text())[-1]
            assert output["text_layers"] == 24 and len(output["samples"]) == 10
            assert not output["arguments"]["check_state"]
            assert resources["stats"]["current_in_use"] == 0
            assert (output["offload_status"]["selected_transfer_bytes"] > 0) == enabled
            stderr, = (directory / "ranks").rglob("0/stderr.log")
            pattern = r"INFO:root:Rank 0 step (\d+): losses=(\[.*?\]) grad_norm=(.*?) selected_transfer_bytes=(\d+)"
            logs = [dict(step=int(step), losses=ast.literal_eval(losses), grad_norm=float(norm))
                    for step, losses, norm, _ in re.findall(pattern, stderr.read_text())]
            assert [row["step"] for row in logs] == list(range(13))
            row = dict(name=name, fused=fused, repetition=repetition, enabled=enabled,
                       median_ms=1000 * statistics.median(sample["seconds"] for sample in output["samples"]),
                       peak_allocated_bytes=max(sample["peak_allocated_bytes"] for sample in output["samples"]),
                       peak_reserved_bytes=max(sample["peak_reserved_bytes"] for sample in output["samples"]),
                       resources=resources, logged_losses_and_norms=logs)
            records.append(row)
            (root / "performance-progress.json").write_text(json.dumps(dict(revision=revision, records=records), indent=2) + "\n")
            print("Finished " + name + ": " + str(round(row["median_ms"], 2)) + " ms", flush=True)

summary = []
for fused in (False, True):
    pairs = []
    for repetition in range(3):
        off = next(row for row in records if (row["fused"], row["repetition"], row["enabled"]) == (fused, repetition, False))
        on = next(row for row in records if (row["fused"], row["repetition"], row["enabled"]) == (fused, repetition, True))
        pairs.append(dict(repetition=repetition, median_ms=[off["median_ms"], on["median_ms"]],
                          time_change_percent=100 * (on["median_ms"] / off["median_ms"] - 1),
                          allocated_saved_bytes=off["peak_allocated_bytes"] - on["peak_allocated_bytes"],
                          reserved_saved_bytes=off["peak_reserved_bytes"] - on["peak_reserved_bytes"],
                          exact_logged_losses_and_norms=off["logged_losses_and_norms"] == on["logged_losses_and_norms"]))
    summary.append(dict(fused=fused, pairs=pairs,
                        median_paired_change_percent=statistics.median(row["time_change_percent"] for row in pairs)))
(root / "performance-comparison.json").write_text(json.dumps(dict(revision=revision, settings=summary), indent=2) + "\n")
print(json.dumps(summary), flush=True)
