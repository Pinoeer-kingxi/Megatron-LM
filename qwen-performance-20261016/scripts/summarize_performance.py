"""Validate raw optimizer-step measurements; keep correctness and throughput claims distinct."""

import csv
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "manifest.json").read_text())
assert manifest["status"] == "measurements_completed"
assert manifest["cleanup"]["exit_code"] == 0
assert len(manifest["jobs"]) == 12
raw, summaries = [], []
for fused in (0, 1):
    runs = []
    reference_metrics = None
    tuning_choices = {}
    for repeat in range(3):
        pair = {}
        for offload in (0, 1):
            name = f"f{fused}-r{repeat}-o{offload}"
            folder = ROOT / name
            job = next(j for j in manifest["jobs"] if j["name"] == name)
            assert job["exit_code"] == 0
            occupancy = json.loads((folder / "occupancy.json").read_text())
            assert not occupancy["external"] and occupancy["gpu"] == manifest["gpu"]
            assert occupancy["samples"] and any(s["processes"] for s in occupancy["samples"])
            results = json.loads((folder / "result.rank0.json").read_text())
            args = results["arguments"]
            assert results["gpu_training_run"] and results["text_layers"] == 24
            assert not args["check_state"] and not results["states"]
            assert (args["tp"], args["cp"], args["pp"], args["sp"], args["vp"]) == (1, 1, 1, False, None)
            assert (args["seq_length"], args["microbatches"], args["warmup"], args["iterations"]) == (4096, 4, 3, 10)
            assert args["min_offloaded_tensor_size"] == 1048576
            assert args["fused_pre_gdr"] == bool(fused) and not args["recompute_norm"]
            assert args["fraction"] == (1.0 if offload else None)
            samples = results["samples"]
            assert len(samples) == 10 and [s["step"] for s in samples] == list(range(3, 13))
            selected = results["offload_status"]["selected_transfer_bytes"]
            assert (selected > 0) if offload else (selected == 0)
            for sample in samples:
                assert math.isfinite(sample["seconds"]) and sample["seconds"] > 0
                assert sample["peak_allocated_bytes"] > 0
                assert sample["selected_transfer_bytes"] == selected
                raw.append({"fused_pre_gdr": fused, "repeat": repeat, "offload": offload, **sample})
            metrics = re.findall(
                r"Rank 0 step (\d+): losses=(\[[^\n]*?\]) grad_norm=(.*?) selected_transfer_bytes=(\d+)",
                (folder / "launcher.log").read_text(),
            )
            assert len(metrics) == 13 and [int(m[0]) for m in metrics] == list(range(13)), name
            scalars = [(int(step), json.loads(losses), norm) for step, losses, norm, _ in metrics]
            assert all(math.isfinite(v) for _, losses, _ in scalars for v in losses)
            if reference_metrics is None:
                reference_metrics = scalars
            assert scalars == reference_metrics, f"Logged loss/gradient norm changed in {name}"
            configs = json.loads((folder / "autotune.json").read_text())
            assert configs, name
            for key, value in configs.items():
                if key in tuning_choices:
                    assert value["config"] == tuning_choices[key], f"Tuning choice overwritten in {name}: {key}"
                tuning_choices[key] = value["config"]
            pair[offload] = {
                "median_seconds": statistics.median(s["seconds"] for s in samples),
                "max_peak_allocated_bytes": max(s["peak_allocated_bytes"] for s in samples),
                "max_peak_reserved_bytes": max(s["peak_reserved_bytes"] for s in samples),
                "selected_transfer_bytes": selected,
            }
        pair["runtime_change_pct"] = 100 * (pair[1]["median_seconds"] / pair[0]["median_seconds"] - 1)
        pair["memory_saved_mib"] = (pair[0]["max_peak_allocated_bytes"] - pair[1]["max_peak_allocated_bytes"]) / 2**20
        runs.append(pair)
    changes = [run["runtime_change_pct"] for run in runs]
    baseline = statistics.median(run[0]["max_peak_allocated_bytes"] for run in runs)
    enabled = statistics.median(run[1]["max_peak_allocated_bytes"] for run in runs)
    summaries.append({
        "fused_pre_gdr": bool(fused),
        "disabled_peak_allocated_mib": baseline / 2**20,
        "enabled_peak_allocated_mib": enabled / 2**20,
        "memory_saved_mib": (baseline - enabled) / 2**20,
        "memory_saved_pct": 100 * (1 - enabled / baseline),
        "disabled_median_step_seconds": statistics.median(run[0]["median_seconds"] for run in runs),
        "enabled_median_step_seconds": statistics.median(run[1]["median_seconds"] for run in runs),
        "paired_runtime_change_pct": statistics.median(changes),
        "paired_runtime_change_range_pct": [min(changes), max(changes)],
        "logged_losses_exact_and_printed_gradient_norms_equal_across_six_runs": True,
        "per_repeat": runs,
    })
assert len(raw) == 120
with (ROOT / "raw-samples.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=list(raw[0]))
    writer.writeheader()
    writer.writerows(raw)
report = {
    "status": "passed",
    "source_commit": manifest["source_commit"], "gpu": manifest["gpu"],
    "environment": json.loads((ROOT / "environment.json").read_text()),
    "measured_samples": 120, "runs": 12, "optimizer_steps_including_warmup": 156,
    "scope": "Complete 24-layer Qwen3.5-0.8B text training, single GPU, BF16/FLA; vision/MTP excluded",
    "correctness_limit": "This performance run compares full-precision logged losses and rounded printed gradient norms, not complete gradient/optimizer fingerprints; full-state correctness evidence is separate.",
    "performance_limit": "Paired offload on/off at the same current revision. This is not an old/new shared-manager comparison or a controlled comparison of fusion variants.",
    "statistics": "Median of three paired per-run median-time changes; memory is median of per-run maximum allocated peak.",
    "results": summaries,
}
(ROOT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
