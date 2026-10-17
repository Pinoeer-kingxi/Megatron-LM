"""Verify exported logs and state fingerprints on CPU; never execute GPU tests."""

import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text())


def rank_report(path, expected):
    report = read(path)
    assert report["exit_code"] == 0 and report["collected"] == expected, path
    assert len(report["outcomes"]) == expected, path
    assert all(row["outcome"] == "passed" for row in report["outcomes"]), path


for line in (ROOT / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split(None, 1)
    path = ROOT / name
    assert path.resolve().is_relative_to(ROOT)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
print("PASS: exported file checksums")

assert read("checkpoint/source.json")["feature"] == "485cb5fa2ffedaaaa0a1a4d095e43ddd99487c1d"
assert read("source.json")["production_tree"] == read("checkpoint/source.json")["production_tree"]
assert read("extensions-ready/source.json")["feature"] == "043e3173451bea74e63c02b197867bf356ad72e9"
assert read("extensions-ready/source.json")["core_tree"] == "0b545fb8c83bd73c6c1f323f7e7a0bd8aef8ec34"
print("PASS: original, checkpoint and extension source identities")

stability = [json.loads(line) for line in (ROOT / "stability-steps.jsonl").read_text().splitlines()]
assert [row["step"] for row in stability] == list(range(250))
for row in stability:
    assert math.isfinite(row["grad_norm"]) and row["grad_norm"] > 0
    assert len(row["losses"]) == 4 and all(math.isfinite(x) for x in row["losses"])
    pool = row["pool_before_schedule_reset"]
    assert pool["current_in_use"] == 0 and pool["allocation_requests"] == pool["free_requests"]
    assert pool["total_allocated"] == 72 and row["selected_transfer_bytes"] > 0
for name in ("allocated_bytes", "reserved_bytes", "selected_transfer_bytes"):
    assert len({row[name] for row in stability[3:]}) == 1, name
assert len(read("stability-result.rank0.json")["samples"]) == 247
assert read("gpu1-availability-change.json")["overlapped_stability"]
print("PASS: 250 finite stability steps and stable pool/memory; shared-device timing caveat retained")

for stage, ranks, configurations in [
    ("checkpoint", 1, ["f0-n0-o0", "f0-n0-o1", "f0-n1-o1", "f1-n0-o1", "f1-n1-o1"]),
    ("two-gpu-retry", 2, ["f0-n1-o1"]),
]:
    summary = read(stage + "/checkpoint-comparison.json")
    assert summary["status"] == "passed" and len(summary["comparisons"]) == 2*ranks*len(configurations)
    assert all(c["exact"] and c["close"] and c["rng_exact"] and c["differing_leaves"] == 0 for c in summary["comparisons"])
    for config in configurations:
        for rank in range(ranks):
            continuous = read(f"{stage}/{config}-continuous/result.rank{rank}.json")
            prefix = read(f"{stage}/{config}-prefix/result.rank{rank}.json")
            resumed = read(f"{stage}/{config}-resumed/result.rank{rank}.json")
            assert continuous["first_step"] == prefix["first_step"] == 0 and resumed["first_step"] == 2
            for arm in (continuous, prefix, resumed):
                assert arm["gpu_training_run"] and arm["pool_verified_before_reset"] and arm["text_layers"] == 24
                for state in arm["states"]:
                    assert state["grads"] and state["weights"] and state["optimizer"]
                    assert math.isfinite(state["grad_norm"]) and state["grad_norm"] > 0
                    assert all(math.isfinite(x) for x in state["losses"])
                selected = arm["offload_status"]["selected_transfer_bytes"]
                assert (selected > 0) == (arm["arguments"]["fraction"] == 1.0)
            assert [s["step"] for s in continuous["states"]] == list(range(6))
            assert [s["step"] for s in prefix["states"]] == [0, 1]
            assert [s["step"] for s in resumed["states"]] == [2, 3, 4, 5]
            assert continuous["states"] == prefix["states"] + resumed["states"]
            assert read(f"{stage}/{config}-resumed/restored-rng.rank{rank}.json")["all_rng_states_exact"]
    print(f"PASS: {stage} exact restarted loss, gradients, weights, complete optimizer/master states and restored RNG")

for job, count in [("scopes", 90), ("native-training", 18)]:
    rank_report(f"extensions-graph-contract/{job}/reports/rank-0.json", count)
assert read("extensions-graph-contract/manifest.json")["status"] == "failed"
failed_graphs = read("extensions-graph-contract/graphs/reports/rank-0.json")
assert failed_graphs["exit_code"] != 0
assert any(row["outcome"] == "failed" and "gdn2" in row["nodeid"] and "nan" in row["detail"]
           for row in failed_graphs["outcomes"])
assert read("extensions-ready/manifest.json")["status"] == "passed"
for rank in range(2):
    rank_report(f"extensions-ready/native-training/reports/rank-{rank}.json", 18)
    rank_report(f"extensions-ready/graphs/reports/rank-{rank}.json", 21)
print("PASS: recorded scope/native coverage and final-source real TE graph matrices, with no skips; pre-fix graph failure remains labeled")

summary = read("extensions-ready-benchmark/comparison.json")
assert summary["status"] == read("extensions-ready-benchmark/manifest.json")["status"] == "passed"
changes = []
for pair in summary["pairs"]:
    result = read(f"extensions-ready-benchmark/pair-{pair['pair']}/result.json")
    baseline, offload = result["arms"]
    assert result["config"]["variant"] == "gdn2" and len(baseline["step_ms"]) == len(offload["step_ms"]) == 5
    assert offload["bitwise_equal"] and offload["checked_steps"] == 8 and not offload["mismatches"]
    assert offload["max_pinned_buffers_in_use_after_backward"] == 0
    assert offload["selected_offload_bytes"]["gdn_core_attn"] > 0
    assert baseline["median_step_ms"] == statistics.median(baseline["step_ms"])
    assert offload["median_step_ms"] == statistics.median(offload["step_ms"])
    assert pair["saved_bytes"] == max(baseline["peak_allocated_bytes"]) - max(offload["peak_allocated_bytes"])
    changes.append(100*(offload["median_step_ms"]/baseline["median_step_ms"] - 1))
    assert pair["step_time_change_percent"] == changes[-1]
assert len(changes) == 3 and statistics.median(changes) == summary["median_step_time_change_percent"]
print("PASS: three GDN2 stack-only benchmark pairs and their actual memory/runtime trade-off")
print("PASS: offline evidence validation only; no GPU execution performed")
