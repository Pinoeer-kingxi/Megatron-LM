"""Check recorded graph training states and recompute paired performance summaries."""

import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "manifest.json").read_text())
assert manifest["status"] in ("completed_pending_verification", "passed")
assert len(manifest["jobs"]) == 16
assert all(job["exit_code"] == 0 for job in manifest["jobs"])
assert manifest["source"] == "f19ebc5b1da1c9b0075eecfd2f93fc6ca4b3ba8d"
for name, expected in manifest["script_sha256"].items():
    assert hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest() == expected, name
for job in manifest["jobs"]:
    expected = job["benchmark_script_sha256"]
    current = ROOT / "scripts/benchmark_graph_training.py"
    path = current if hashlib.sha256(current.read_bytes()).hexdigest() == expected else ROOT / "scripts/versions" / expected / current.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, job["label"]

summary = {}
for model in ("qwen", "gdn2"):
    arms = []
    for arm in ("disabled", "enabled"):
        data = json.loads((ROOT / f"{model}-correct-{arm}/result.json").read_text())
        records = data["records"]
        assert len(records) == data["pool_checks"] == 6
        assert len(data["graph_counts"]) == (24 if model == "qwen" else 8)
        assert all(count == data["helper_microbatches"] > 0 for count in data["graph_counts"])
        assert data["arguments"]["microbatches"] == 4
        assert len({record["input_sha256"] for record in records}) == 6
        assert records[0]["weights"] != records[-1]["weights"]
        assert sum(record["replay"] for record in records) == 3
        for record in records:
            assert record["pool"]["current_in_use"] == 0
            assert math.isfinite(record["grad_norm"]) and record["grad_norm"] > 0
            if arm == "enabled" and record["replay"]:
                assert record["selected_bytes"] > 0
        arms.append(records)
    for expected, actual in zip(*arms, strict=True):
        for key in ("step", "input_sha256", "loss", "grad_norm", "gradients", "weights", "optimizer"):
            assert expected[key] == actual[key], (model, expected["step"], key)
    pairs = []
    for repeat in range(3):
        result = {}
        for arm in ("disabled", "enabled"):
            data = json.loads((ROOT / f"{model}-perf-{repeat}-{arm}/result.json").read_text())
            assert len(data["graph_counts"]) == (24 if model == "qwen" else 8)
            assert all(count == data["helper_microbatches"] > 0 for count in data["graph_counts"])
            assert data["arguments"]["microbatches"] == 4
            assert data["pool_checks"] == len(data["records"]) == 16
            assert len({record["input_sha256"] for record in data["records"]}) == 16
            records = [record for record in data["records"] if record["measured"]]
            assert len(records) == 10 and all(record["replay"] for record in records)
            assert all(record["pool"]["current_in_use"] == 0 for record in data["records"])
            assert all("gradients" not in record and "optimizer" not in record for record in records)
            assert all(math.isfinite(record["seconds"]) and record["seconds"] > 0 for record in records)
            if arm == "enabled":
                assert all(record["selected_bytes"] > 0 for record in records)
                replay = [record for record in data["records"] if record["replay"]]
                for field in ("total_allocated", "allocation_requests", "free_requests"):
                    assert len({record["pool"][field] for record in replay}) == 1, field
            result[arm] = dict(
                median_step_ms=statistics.median(record["seconds"] for record in records) * 1000,
                peak_allocated_bytes=max(record["peak_allocated_bytes"] for record in records),
                peak_reserved_bytes=max(record["peak_reserved_bytes"] for record in records),
                seconds=[record["seconds"] for record in records],
                input_sha256=[record["input_sha256"] for record in records],
                selected_bytes=[record["selected_bytes"] for record in records],
                loss=[record["loss"] for record in data["records"]],
                grad_norm=[record["grad_norm"] for record in data["records"]],
            )
        assert result["disabled"]["input_sha256"] == result["enabled"]["input_sha256"]
        for key in ("loss", "grad_norm"):
            assert result["disabled"][key] == result["enabled"][key], (model, repeat, key)
        baseline, enabled = result["disabled"], result["enabled"]
        pairs.append(dict(
            repeat=repeat,
            **result,
            step_time_change_pct=(enabled["median_step_ms"] / baseline["median_step_ms"] - 1) * 100,
            allocated_saved_bytes=baseline["peak_allocated_bytes"] - enabled["peak_allocated_bytes"],
            reserved_saved_bytes=baseline["peak_reserved_bytes"] - enabled["peak_reserved_bytes"],
        ))
    changes = [pair["step_time_change_pct"] for pair in pairs]
    summary[model] = dict(
        correctness="exact loss, grad norm, all gradients, weights and complete optimizer fingerprints",
        pairs=pairs,
        paired_median_step_change_pct=statistics.median(changes),
        paired_step_change_range_pct=[min(changes), max(changes)],
        median_allocated_saved_bytes=statistics.median(pair["allocated_saved_bytes"] for pair in pairs),
        median_reserved_saved_bytes=statistics.median(pair["reserved_saved_bytes"] for pair in pairs),
        reserved_saved_range_bytes=[min(pair["reserved_saved_bytes"] for pair in pairs), max(pair["reserved_saved_bytes"] for pair in pairs)],
    )
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
