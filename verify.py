"""Verify downloaded artifacts and compare the recorded complete Qwen training states."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def verify_checksums(root: Path) -> int:
    """Check every file listed in SHA256SUMS against its downloaded bytes."""
    entries = (root / "SHA256SUMS").read_text().splitlines()
    for line in entries:
        digest, name = line.split("  ", 1)
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"SHA-256 mismatch: {name}")
    return len(entries)


def verify_rank_logs(root: Path, run_directory: str) -> list[dict]:
    """Check the final pytest summary in each original rank stdout log."""
    results = []
    for directory, passed, skipped, deselected in (
        ("rebased-unit-ranks", 104, 0, 24),
        ("rebased-training-ranks", 72, 24, 0),
    ):
        paths = sorted((root / run_directory / directory).glob("*/attempt_0/*/stdout.log"))
        if len(paths) != 2 or {path.parent.name for path in paths} != {"0", "1"}:
            raise ValueError(f"Expected two rank logs: {directory}")
        for path in paths:
            summaries = re.findall(r"^\d+ passed[^\n]*$", path.read_text(), re.MULTILINE)
            expected = f"{passed} passed"
            if skipped:
                expected += f", {skipped} skipped"
            if deselected:
                expected += f", {deselected} deselected"
            if len(summaries) != 1 or not summaries[0].startswith(expected + ","):
                raise ValueError(f"Unexpected pytest summary: {path.relative_to(root)}")
            summary_skips = re.findall(r"(\d+) skipped", summaries[0])
            if summary_skips != ([str(skipped)] if skipped else []):
                raise ValueError(f"Unexpected skips: {path.relative_to(root)}")
            if re.search(r"\b(?:failed|errors?)\b", summaries[0]):
                raise ValueError(f"Unexpected failure in summary: {path.relative_to(root)}")
            results.append({"file": str(path.relative_to(root)), "summary": summaries[0]})
    return results


def verify_qwen(root: Path, report_path: str) -> dict:
    """Compare all recorded losses, gradients, updated weights and Adam fingerprints."""
    report = json.loads((root / report_path).read_text())
    records = {(item["fused"], item["arm"], item["rank"]): item for item in report["results"]}
    expected = {
        (fused, arm, rank)
        for fused in (False, True)
        for arm in ("disabled", "1")
        for rank in (0, 1)
    }
    if len(report["results"]) != 8 or records.keys() != expected:
        raise ValueError("Expected eight distinct Qwen rank results")
    results = {}
    for key, record in records.items():
        path = root / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Original Qwen report checksum mismatch: {record['file']}")
        result = json.loads(path.read_text())
        fused, arm, rank = key
        arguments = result["arguments"]
        wanted = {
            "tp": 2,
            "sp": True,
            "cp": 1,
            "pp": 1,
            "vp": None,
            "fused_pre_gdr": fused,
            "recompute_norm": True,
            "fraction": None if arm == "disabled" else 1.0,
            "check_state": True,
            "seq_length": 128,
            "min_offloaded_tensor_size": 1024,
            "warmup": 1,
            "iterations": 2,
            "microbatches": 4,
        }
        if (
            any(arguments[key] != value for key, value in wanted.items())
            or result["rank"] != rank
            or not result["gpu_training_run"]
            or result["samples"]
        ):
            raise ValueError(f"Unexpected Qwen run configuration: {record['file']}")
        if (result["text_layers"], result["text_hidden_size"], result["dataset_tokens"]) != (
            24,
            1024,
            30727,
        ):
            raise ValueError(f"Unexpected model/data metadata: {record['file']}")
        if not result["pool_verified_before_reset"] or result["pool_in_use_before_resets"] != [
            0
        ] * (2 if arm == "disabled" else 5):
            raise ValueError(f"Pinned pool still in use before reset: {record['file']}")
        selected = result["offload_status"]["selected_transfer_bytes"]
        if (
            selected != (0 if arm == "disabled" else 66015232)
            or selected != record["selected_transfer_bytes"]
        ):
            raise ValueError(f"Unexpected transfer bytes: {record['file']}")
        states = result["states"]
        if [state["step"] for state in states] != [0, 1, 2] or any(
            not all(state[field] for field in ("losses", "grads", "weights", "optimizer"))
            for state in states
        ):
            raise ValueError(f"Incomplete training states: {record['file']}")
        results[key] = states
    for fused in (False, True):
        for rank in (0, 1):
            if results[fused, "disabled", rank] != results[fused, "1", rank]:
                raise ValueError(f"Qwen state mismatch: fused={fused}, rank={rank}")
    return {
        "fresh_two_rank_runs": 4,
        "rank_files": 8,
        "exact_complete_rank_step_states": 24,
        "all_pre_reset_pool_counts_zero": True,
        "enabled_transfer_bytes_per_rank": 66015232,
        "timed_samples": 0,
    }


def main() -> None:
    """Print the independent CPU verification report; fail on missing or changed evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    root = parser.parse_args().directory
    count = verify_checksums(root)
    provenance = json.loads((root / "provenance.json").read_text())
    latest = json.loads((root / "latest-head-20261014/run-provenance.json").read_text())
    commit = latest["source_commit_at_launch"]
    if (
        latest["status"] != "passed"
        or not latest["source_clean_at_launch"]
        or len(latest["jobs"]) != 6
    ):
        raise ValueError("Latest-source GPU jobs did not all pass")
    for job in latest["jobs"]:
        if (
            job["source_commit"] != commit
            or job["exit_code"] != 0
            or not job["source_clean_before_run"]
            or not job["source_clean_after_run"]
        ):
            raise ValueError(f"Incomplete clean-source run: {job['name']}")
    result = {
        "status": "passed",
        "verification_kind": "CPU verification of recorded GPU artifacts",
        "tested_functional_source_snapshot": provenance["tested_functional_source_snapshot"],
        "original_published_pr_head": provenance["original_published_pr_head"],
        "sha256_files_verified": count,
        "earlier_post_rebase": {
            "rank_test_logs": verify_rank_logs(root, "publication-20261011"),
            "qwen": verify_qwen(root, "publication-20261011/qwen-rebased-validation.json"),
        },
        "latest_clean_source": {
            "source_commit": commit,
            "gpu_launcher_exit_codes": [job["exit_code"] for job in latest["jobs"]],
            "rank_test_logs": verify_rank_logs(root, "latest-head-20261014"),
            "qwen": verify_qwen(root, "latest-head-20261014/qwen-validation.json"),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
