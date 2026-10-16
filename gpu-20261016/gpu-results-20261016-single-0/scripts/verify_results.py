"""Validate actual rank reports, complete Qwen states and gate timing samples."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def load(path: Path):
    """Load one required result; missing or malformed files fail validation."""
    return json.loads(path.read_text())


def hash_file(path: Path) -> str:
    """Hash checkpoints and evidence without loading the whole file into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pytest(directory: Path, ranks: int, expected: int, allow_skips: bool = False) -> dict:
    """Require the expected test set on every rank, retaining explicit skip reasons."""
    results = []
    for rank in range(ranks):
        report = load(directory / f"rank-{rank}.json")
        passed = [
            row["nodeid"]
            for row in report["outcomes"]
            if row["when"] == "call" and row["outcome"] == "passed"
        ]
        skipped = [row for row in report["outcomes"] if row["outcome"] == "skipped"]
        failed = [row for row in report["outcomes"] if row["outcome"] == "failed"]
        if report["rank"] != rank or report["exit_code"] != 0 or failed:
            raise ValueError(f"Rank {rank} failed: {failed}")
        if report["collected"] != expected or len(passed) + len(skipped) != expected:
            raise ValueError(
                f"Rank {rank}: expected {expected} tests, got {report['collected']} collected, "
                f"{len(passed)} passed, {len(skipped)} skipped"
            )
        if skipped and not allow_skips:
            raise ValueError(f"Required tests skipped on rank {rank}: {skipped}")
        if len(set(passed)) != len(passed):
            raise ValueError(f"Rank {rank}: duplicate test outcomes")
        results.append({"rank": rank, "passed": len(passed), "skipped": skipped})
    return {"ranks": results, "skipped": sum(len(row["skipped"]) for row in results)}


def verify_qwen(root: Path, ranks: int = 4) -> dict:
    """Compare every recorded loss, gradient, weight and Adam/master state exactly."""
    comparisons = 0
    for fused in (0, 1):
        for norm in (0, 1):
            for rank in range(ranks):
                arms = []
                for offload in (0, 1):
                    path = root / f"qwen-f{fused}-n{norm}-o{offload}" / f"result.rank{rank}.json"
                    result = load(path)
                    args = result["arguments"]
                    if (
                        not result["gpu_training_run"]
                        or not result["pool_verified_before_reset"]
                        or result["rank"] != rank
                    ):
                        raise ValueError(f"{path}: not a completed GPU run with pool checks")
                    if result["text_layers"] != 24:
                        raise ValueError(f"{path}: expected all 24 text decoder layers")
                    expected = {
                        "tp": 2,
                        "sp": True,
                        "cp": 1,
                        "pp": 2,
                        "vp": None,
                        "warmup": 1,
                        "iterations": 2,
                        "microbatches": 4,
                        "seq_length": 128,
                        "check_state": True,
                        "dry_run": False,
                        "fused_pre_gdr": bool(fused),
                        "recompute_norm": bool(norm),
                        "fraction": 1.0 if offload else None,
                    }
                    if any(args[key] != value for key, value in expected.items()):
                        raise ValueError(f"{path}: unexpected training arguments")
                    states = result["states"]
                    if [state["step"] for state in states] != [0, 1, 2]:
                        raise ValueError(f"{path}: missing or reordered steps")
                    for state in states:
                        expected_losses = 4 if rank >= 2 else 0
                        if len(state["losses"]) != expected_losses:
                            raise ValueError(
                                f"{path}: expected {expected_losses} microbatch losses"
                            )
                        if (
                            not math.isfinite(state["grad_norm"])
                            or state["grad_norm"] <= 0
                            or not all(math.isfinite(loss) for loss in state["losses"])
                        ):
                            raise ValueError(f"{path}: invalid loss or gradient norm")
                        for field in ("grads", "weights", "optimizer"):
                            if not state[field]:
                                raise ValueError(f"{path}: missing {field}")
                    if offload:
                        if result["offload_status"]["selected_transfer_bytes"] <= 0:
                            raise ValueError(f"{path}: no positive offload selection")
                    elif result["offload_status"]["selected_transfer_bytes"] != 0:
                        raise ValueError(f"{path}: disabled arm selected transfers")
                    arms.append(states)
                if arms[0] != arms[1]:
                    for expected, actual in zip(arms[0], arms[1]):
                        for field in expected:
                            if expected[field] != actual.get(field):
                                raise ValueError(
                                    f"Qwen fusion={fused}, norm={norm}, rank={rank}, "
                                    f"step={expected['step']}: {field} differs"
                                )
                    raise ValueError("Qwen state structure differs")
                comparisons += len(arms[0])
    return {
        "compared_rank_steps": comparisons,
        "recorded_rank_steps": comparisons * 2,
        "compared_arms": 4,
        "ranks": ranks,
        "exact_recorded_state_equal": True,
    }


def summarize_benchmarks(root: Path, sources: dict, gpu_uuid: str) -> list[dict]:
    """Retain paired measurements; do not invent a performance acceptance threshold."""
    results = []
    for batch in (1, 4):
        for length in (2048, 4096):
            for heads in (16, 32):
                changes, timings, peaks = (
                    [],
                    {"baseline": [], "candidate": []},
                    {"baseline": [], "candidate": []},
                )
                for repeat in range(3):
                    path = root / f"gate-bench-{batch}-{length}-{heads}-{repeat}" / "result.json"
                    result = load(path)
                    case = result["case"]
                    if result["gpu_uuid"] != gpu_uuid or result["offloading"] is not False:
                        raise ValueError(f"{path}: unexpected device or offload setting")
                    for label, source in (("baseline", "base"), ("candidate", "gate")):
                        if result["sources"][label]["revision"] != sources[source]:
                            raise ValueError(f"{path}: wrong {label} source")
                        samples = case["samples"][label]
                        if len(samples) != 3 or any(
                            not math.isfinite(row["wall_ms_per_call"])
                            or row["wall_ms_per_call"] <= 0
                            or row["extra_peak_allocated_bytes"] < 0
                            for row in samples
                        ):
                            raise ValueError(f"{path}: invalid timing samples")
                        timings[label].extend(row["wall_ms_per_call"] for row in samples)
                        peaks[label].extend(row["extra_peak_allocated_bytes"] for row in samples)
                    if case["numerical_reference"] != "passed" or (
                        case["batch"],
                        case["sequence"],
                        case["value_heads"],
                    ) != (batch, length, heads):
                        raise ValueError(f"{path}: numerical check or shape mismatch")
                    changes.extend(
                        (fixed["wall_ms_per_call"] / old["wall_ms_per_call"] - 1) * 100
                        for old, fixed in zip(
                            case["samples"]["baseline"], case["samples"]["candidate"]
                        )
                    )
                results.append(
                    {
                        "batch": batch,
                        "sequence": length,
                        "value_heads": heads,
                        "fresh_processes": 3,
                        "paired_samples": len(changes),
                        "median_paired_runtime_change_pct": statistics.median(changes),
                        "paired_runtime_change_range_pct": [min(changes), max(changes)],
                        "median_ms": {
                            key: statistics.median(value) for key, value in timings.items()
                        },
                        "max_extra_peak_bytes": {key: max(value) for key, value in peaks.items()},
                    }
                )
    return results


def verify_run(root: Path) -> dict:
    """Recheck an exported run on CPU; this does not launch tests or initialize CUDA."""
    manifest = load(root / "manifest.json")
    sources = load(Path(__file__).with_name("source-after.json"))
    if manifest["sources"] != {key: sources[key] for key in ("base", "shared", "gate", "feature")}:
        raise ValueError("Run sources do not match this acceptance package")
    if (
        manifest["status"] not in ("automated_checks_passed", "incomplete")
        or manifest["cleanup"]["exit_code"]
    ):
        raise ValueError(f"Run is not complete: {manifest['status']}")
    if [job["name"] for job in manifest["jobs"]] != manifest["planned_jobs"]:
        raise ValueError("Some planned jobs did not run")
    verified = set()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, filename = line.split(None, 1)
        if hash_file(root / filename) != expected:
            raise ValueError(f"Checksum mismatch: {filename}")
        verified.add(filename)
    if "manifest.json" not in verified:
        raise ValueError("SHA256SUMS must include manifest.json")
    skipped = 0
    for job in manifest["jobs"]:
        if job["exit_code"] != 0 or job["source_commit"] != sources[job["source"]]:
            raise ValueError(f"Invalid job: {job['name']}")
        if "expected_tests" in job:
            result = verify_pytest(
                root / job["name"] / "reports",
                len(manifest["gpus"]),
                job["expected_tests"],
                job["allow_skips"],
            )
            skipped += result["skipped"]
    result = {
        "status": "incomplete" if skipped else "automated_checks_passed",
        "verified_files": len(verified),
        "skipped": skipped,
        "performance_review": manifest["performance_review"],
        "omitted_jobs": manifest.get("omitted_jobs", []),
    }
    if manifest["stage"] == "single":
        result["gate_performance"] = summarize_benchmarks(
            root, sources, manifest["gpus"][0]["uuid"]
        )
    else:
        result["qwen_comparison"] = verify_qwen(root)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recheck a completed acceptance run without a GPU")
    parser.add_argument("output", type=Path)
    result = verify_run(parser.parse_args().output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "automated_checks_passed" else 1)
