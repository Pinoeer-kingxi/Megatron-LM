"""Audit actual outcomes, pinned source revisions and complete Qwen state records."""

import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sources = json.loads((root / "sources.json").read_text())
checks = []
harness_hashes = {
    hashlib.sha256(path.read_bytes()).hexdigest() for path in (root / "scripts").glob("*.py")
}


def read(path):
    return json.loads(path.read_text())


def verify_job(name, source, ranks, count):
    directory = root / name
    manifest = read(directory / "manifest.json")
    assert manifest["revision"] == sources[source]["revision"]
    assert manifest["exit_code"] == 0
    assert manifest["harness_sha256"] in harness_hashes
    if "revision_after" in manifest:
        assert manifest["revision_after"] == manifest["revision"] and not manifest["dirty_after"]
    rows = []
    for rank in range(ranks):
        report = read(directory / "reports" / f"rank-{rank}.json")
        assert report["rank"] == rank and report["exit_code"] == 0
        assert report["collected"] == count
        assert len(report["outcomes"]) == count
        assert all(
            item["when"] == "call" and item["outcome"] == "passed" for item in report["outcomes"]
        )
        assert len({item["nodeid"] for item in report["outcomes"]}) == count
        assert len(list((directory / "ranks").rglob(f"{rank}/stdout.log"))) == 1
        assert len(list((directory / "ranks").rglob(f"{rank}/stderr.log"))) == 1
        rows.append(dict(rank=rank, passed=count, skipped=0))
    checks.append(dict(name=name, revision=manifest["revision"], ranks=rows))


for arguments in [
    ("shared-final-targeted", "shared", 2, 26),
    ("shared-combined-trace", "shared", 2, 1),
    ("shared-graphs", "shared", 2, 3),
    ("shared-four-phase", "shared", 4, 7),
    ("feature-final-native", "feature", 2, 72),
    ("feature-four-native", "feature", 4, 96),
    ("feature-pp-pool-trace", "feature", 2, 2),
    ("feature-scopes", "feature", 2, 94),
    ("extensions-scopes", "extensions", 2, 20),
    ("extensions-native", "extensions", 2, 18),
    ("extensions-graphs", "extensions", 2, 21),
]:
    verify_job(*arguments)

negative_sources = read(root / "negative-control-sources.json")
negatives = []
for variant, reason in (
    ("eval", "assert manager._is_warmup == forward_only"),
    ("fraction", "assert bool(transfers) == (fraction > 0)"),
    ("progress", "Chunk mismatch"),
):
    name = "negative-final-" + variant
    directory = root / name
    manifest = read(directory / "manifest.json")
    assert manifest["revision"] == negative_sources[variant]["revision"]
    assert manifest["exit_code"] != 0
    assert negative_sources[variant]["same_regression_as_shared"]
    for rank in range(2):
        report = read(directory / "reports" / f"rank-{rank}.json")
        assert report["collected"] == 1 and report["exit_code"] == 1
        assert len(report["outcomes"]) == 1
        item = report["outcomes"][0]
        assert item["outcome"] == "failed" and item["when"] == "call"
        assert reason in item["detail"]
    negatives.append(dict(name=name, revision=manifest["revision"], failure=reason))

qwen = []


def verify_qwen(names, ranks, expected, captured=False):
    arms = []
    reference_arguments = None
    for index, name in enumerate(names):
        directory = root / name
        manifest = read(directory / "manifest.json")
        assert manifest["revision"] == sources["feature"]["revision"] and manifest["exit_code"] == 0
        if "harness_sha256" in manifest:
            assert manifest["harness_sha256"] in harness_hashes
        outputs = [read(directory / f"result.rank{rank}.json") for rank in range(ranks)]
        for rank, output in enumerate(outputs):
            arguments = output["arguments"]
            assert all(arguments[key] == value for key, value in expected.items())
            assert arguments["cp"] == 1 and arguments["vp"] is None
            assert arguments["seq_length"] == 128 and arguments["microbatches"] == 4
            assert arguments["warmup"] == 1 and arguments["iterations"] == 2
            assert arguments["min_offloaded_tensor_size"] == 1024 and arguments["check_state"]
            assert arguments["fraction"] == (1.0 if index == 2 else None)
            common_arguments = {
                key: value for key, value in arguments.items() if key not in ("output", "fraction")
            }
            if reference_arguments is None:
                reference_arguments = common_arguments
            assert common_arguments == reference_arguments
            assert output["gpu_training_run"] and output["pool_verified_before_reset"]
            assert output["text_layers"] == 24 and output["rank"] == rank
            assert [step["step"] for step in output["states"]] == [0, 1, 2]
            assert (output["offload_status"]["selected_transfer_bytes"] > 0) == (index == 2)
            for step in output["states"]:
                last_stage = (rank // arguments["tp"]) % arguments["pp"] == arguments["pp"] - 1
                assert len(step["losses"]) == (4 if last_stage else 0)
                assert math.isfinite(step["grad_norm"]) and step["grad_norm"] > 0
                assert all(math.isfinite(value) for value in step["losses"])
                assert all(step[field] for field in ("grads", "weights", "optimizer"))
        arms.append([output["states"] for output in outputs])
    assert arms[0] == arms[1] == arms[2], names
    qwen.append(
        dict(
            arms=names,
            ranks=ranks,
            steps=3,
            complete_recorded_states_exact=True,
            actual_vector_capture=captured,
        )
    )


for topology in ("tp-sp", "pp"):
    expected = dict(
        tp=2 if topology == "tp-sp" else 1,
        sp=topology == "tp-sp",
        pp=1 if topology == "tp-sp" else 2,
        recompute_norm=topology == "pp",
        fused_pre_gdr=False,
    )
    verify_qwen(
        [f"qwen-{topology}-{arm}" for arm in ("disabled", "repeat-disabled", "offloaded")],
        2,
        expected,
    )
for fused in (0, 1):
    for norm in (0, 1):
        prefix = "four-qwen-captured-f1" if fused else "four-qwen-f0"
        verify_qwen(
            [f"{prefix}-n{norm}-{arm}" for arm in ("disabled", "repeat-disabled", "offloaded")],
            4,
            dict(tp=2, sp=True, pp=2, recompute_norm=bool(norm), fused_pre_gdr=bool(fused)),
            captured=bool(fused),
        )

reuse = []
for name in ("shared-final-targeted", "shared-combined-trace", "feature-pp-pool-trace"):
    for path in sorted((root / name / "reports").glob("pool-*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        samples = [sample for row in rows for sample in row["samples"]]
        assert samples
        minimum = min(sample["h2d_end_to_next_d2h_start_ms"] for sample in samples)
        assert minimum >= 0
        reuse.append(
            dict(job=name, report=path.name, observed_reuses=len(samples), minimum_ms=minimum)
        )

performance = read(root / "performance-comparison.json")
assert performance["revision"] == sources["feature"]["revision"]
assert performance["allocated_saved_bytes"] > 0
assert performance["exact_logged_losses_and_reported_norms_equal"]
for index, name in enumerate(("performance-disabled", "performance-offloaded")):
    assert read(root / name / "manifest.json")["exit_code"] == 0
    output = read(root / name / "result.rank0.json")
    assert output["gpu_training_run"] and output["text_layers"] == 24
    arguments = output["arguments"]
    assert arguments["seq_length"] == 4096 and arguments["microbatches"] == 4
    assert arguments["warmup"] == 3 and arguments["iterations"] == 10
    assert arguments["min_offloaded_tensor_size"] == 1048576
    assert not arguments["check_state"] and not arguments["fused_pre_gdr"]
    assert not arguments["recompute_norm"]
    assert [sample["step"] for sample in output["samples"]] == list(range(3, 13))
    assert all(
        (sample["selected_transfer_bytes"] > 0) == bool(index) for sample in output["samples"]
    )

lint = []
for name in ("shared", "feature", "extensions"):
    row = read(root / f"lint-{name}.json")
    assert row["revision"] == sources[name]["revision"] and row["exit_code"] == 0
    lint.append(dict(source=name, autoformat_exit_code=0))
assert read(root / "mypy-comparison.json")["unchanged"]

result = dict(
    status="passed",
    pytest=checks,
    negative_controls=negatives,
    qwen=qwen,
    observed_pool_reuse=reuse,
    performance=performance,
    lint=lint,
    diagnostics=[
        dict(
            name="original-fused-fingerprint-comparison",
            status="strict equality failed, including unchanged disabled control",
            record="four-qwen-comparison.json",
        )
    ],
    limits=[
        "Local RTX A6000 evidence; upstream GPU CI not executed.",
        "One current-head performance pair; historical repeated measurements retain their original revisions.",
        "Pool timing observations do not prove all possible schedules race-free.",
    ],
)
(root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
