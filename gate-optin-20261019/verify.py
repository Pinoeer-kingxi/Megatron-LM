"""Check published reports without CUDA; this does not rerun the training jobs."""

import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text())


def check_sums():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def check_tests(manifest_name, counts):
    manifest = read(manifest_name)
    for job in manifest["jobs"]:
        name = job["name"]
        report = read(f"{name}/reports/rank-0.json")
        expected = job["expected_exit"]
        assert job["exit_code"] == report["exit_code"] == expected, name
        assert report["rank"] == 0 and report["collected"] == counts[name], name
        rows = report["outcomes"]
        assert len(rows) == counts[name], name
        outcome = "failed" if expected else "passed"
        assert all(
            row["when"] == "call" and row["outcome"] == outcome for row in rows
        ), name
        if expected:
            assert "best_config" in rows[0]["detail"], name
    return {name: count for name, count in counts.items()}


def check_training_manifest(name):
    manifest = read(name)
    assert manifest["status"] == "completed_pending_verification", name
    for job in manifest["jobs"]:
        assert job["exit_code"] == 0, job["name"]
        if job["name"] == "calibration":
            continue  # The one unclassified exit-time sample is documented.
        assert not job["external_overlap"], job["name"]
        telemetry = read(f"{job['name']}/telemetry.json")
        assert telemetry and all(not sample["external"] for sample in telemetry), job[
            "name"
        ]


def check_upstream_envelope(expected, actual, control):
    if expected == actual:
        return 0
    if isinstance(expected, dict) and set(expected) == {"shape", "dtype", "sha256"}:
        assert actual == control
        return 1
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys() == control.keys()
        return sum(
            check_upstream_envelope(expected[k], actual[k], control[k])
            for k in expected
        )
    assert isinstance(expected, list) and len(expected) == len(actual) == len(control)
    return sum(
        check_upstream_envelope(a, b, c)
        for a, b, c in zip(expected, actual, control, strict=True)
    )


def compare_training(left, right, steps, control=None):
    left_result = read(f"{left}/result.rank0.json")
    right_result = read(f"{right}/result.rank0.json")
    for result in (left_result, right_result):
        assert result["gpu_training_run"] and not result["arguments"]["dry_run"]
        assert result["arguments"]["fraction"] is None
        assert result["arguments"]["fused_pre_gdr"]
        assert result["arguments"]["seq_length"] == 2048
    assert read(f"{left}/inputs.json") == read(f"{right}/inputs.json"), (left, right)
    assert read(f"{left}/choices.json") == read(f"{right}/choices.json"), (left, right)
    assert read(f"{left}/choices.json")["deterministic_arithmetic"] is False
    if steps:
        assert len(left_result["states"]) == len(right_result["states"]) == steps
        exact = left_result["states"] == right_result["states"]
        if not exact:
            assert control is not None
            control_result = read(f"{control}/result.rank0.json")
            assert read(f"{left}/inputs.json") == read(f"{control}/inputs.json")
            assert read(f"{left}/choices.json") == read(f"{control}/choices.json")
            changes = check_upstream_envelope(
                left_result["states"], right_result["states"], control_result["states"]
            )
        else:
            changes = 0
        for state in left_result["states"]:
            assert set(state) == {
                "step",
                "losses",
                "grad_norm",
                "grads",
                "weights",
                "optimizer",
            }
            assert state["grads"] and state["weights"] and state["optimizer"]
        return {
            "steps": steps,
            "states_byte_exact": exact,
            "changed_tensor_leaves": changes,
            "all_changed_tensors_match_unchanged_upstream_control": bool(control)
            and not exact,
        }
    samples = [left_result["samples"], right_result["samples"]]
    assert len(samples[0]) == len(samples[1]) > 0
    medians = [statistics.median(x["seconds"] for x in arm) * 1000 for arm in samples]
    peaks = [{x["peak_allocated_bytes"] for x in arm} for arm in samples]
    assert all(x["seconds"] > 0 for arm in samples for x in arm)
    assert all(x["selected_transfer_bytes"] == 0 for arm in samples for x in arm)
    assert peaks[0] == peaks[1], (left, right)
    return {
        "baseline": left,
        "candidate": right,
        "samples_per_arm": len(samples[0]),
        "median_ms": medians,
        "change_percent": (medians[1] / medians[0] - 1) * 100,
        "peak_allocated_bytes": sorted(peaks[0]),
    }


def main():
    check_sums()
    identity = read("final-source-identity.json")
    for name, expected in identity["files"].items():
        assert (
            hashlib.sha256((ROOT / "snapshots/final" / name).read_bytes()).hexdigest()
            == expected
        )
    result = {"source": identity["source"]}
    result["tests"] = check_tests(
        "test-manifest.json",
        {"red-default": 1, "gate-replay": 34, "registry": 272, "fusion": 11},
    )
    result["final_tests"] = check_tests(
        "final-test-manifest.json",
        {"final-gate-replay": 34, "final-registry": 272, "final-fusion": 11},
    )
    for name in (
        "full-training-manifest.json",
        "extended-training-manifest.json",
        "final-training-manifest.json",
        "atomic-control-manifest.json",
    ):
        check_training_manifest(name)
    result["state_repeat"] = compare_training("state-baseline", "state-repeat", 3)
    result["state_previous_candidate"] = compare_training(
        "state-baseline", "state-candidate", 3
    )
    result["state_final_candidate"] = compare_training(
        "state-baseline", "final-state-candidate", 3, "state-atomic-control"
    )
    strict_check = read("final-state-strict-check.json")
    assert strict_check["exit_code"] == 1
    result["strict_default_state_comparison"] = (
        "failed; native atomic differences retained"
    )
    result["timings"] = {}
    for prefix, pairs in (("pair", 3), ("extended", 3), ("final-pair", 1)):
        rows = [
            compare_training(f"{prefix}-{i}-baseline", f"{prefix}-{i}-candidate", 0)
            for i in range(pairs)
        ]
        result["timings"][prefix] = {
            "pairs": rows,
            "median_paired_change_percent": statistics.median(
                row["change_percent"] for row in rows
            ),
        }
    for check in read("final-cpu-checks.json"):
        if "mypy" in check["command"]:
            assert (
                check["exit_code"] == 1
                and "Found 4 errors in 1 file" in check["output"]
            )
        else:
            assert check["exit_code"] == 0
    profile = read("profile-analysis.json")
    for base, candidate in zip(profile["baseline"], profile["candidate"], strict=True):
        assert {name: row["count"] for name, row in base["all_kernels"].items()} == {
            name: row["count"] for name, row in candidate["all_kernels"].items()
        }
    kernel_check = read("default-kernel-comparison.json")
    production = "megatron/core/fusions/fused_pre_gated_delta_rule.py"
    assert kernel_check["final_source"] == identity["source"]
    assert (
        kernel_check["arms"]["candidate"]["source_sha256"]
        == identity["files"][production]
    )
    assert (
        kernel_check["arms"]["base"]["source_sha256"]
        == hashlib.sha256((ROOT / "snapshots/upstream.py").read_bytes()).hexdigest()
    )
    instructions = {}
    for arm in ("base", "candidate"):
        sass = (ROOT / "default-kernel-check" / (arm + ".sass.log")).read_text()
        words = re.findall(r"/\*\s*(0x[0-9a-fA-F]{16})\s*\*/", sass)
        assert words and len(words) == kernel_check["arms"][arm]["instruction_words"]
        digest = hashlib.sha256("".join(words).encode()).hexdigest()
        assert digest == kernel_check["arms"][arm]["instruction_sha256"]
        instructions[arm] = words
    assert instructions["base"] == instructions["candidate"]
    result["default_kernel_instruction_words_equal"] = True
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
