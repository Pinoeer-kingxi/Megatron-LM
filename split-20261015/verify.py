"""Verify the frozen CPU evidence; this does not execute or qualify GPU tests."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(name):
    return json.loads((ROOT / name).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    """Check downloads, source/job correspondence, and the stated CPU-only outcomes."""
    checked = []
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), f"Checksum path escapes bundle: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"Checksum mismatch: {name}")
        checked.append(name)
    files = {
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.name not in ("SHA256SUMS", "verification.json")
    }
    require(set(checked) == files, "Checksum manifest does not cover the bundle")
    status = read_json("status.json")
    records = read_json("final-cpu-validation.json")
    expected_jobs = {
        "shared-final-cpu": ("shared", 5),
        "gate-final-manifest": ("gate", 269),
        "feature-final-cpu": ("feature", 21),
        "feature-final-manifest": ("feature", 269),
        "feature-final-native-collection": ("feature", None),
        "feature-final-qwen-preflight": ("feature", None),
    }
    require(
        {item["name"] for item in records} == set(expected_jobs), "CPU job set differs"
    )
    for item in records:
        branch, count = expected_jobs[item["name"]]
        require(item["branch"] == branch, f"Wrong branch: {item['name']}")
        require(
            item["source_commit"] == status["source"][branch], "CPU source SHA differs"
        )
        require(
            item["source_clean_before_run"] and item["source_clean_after_run"],
            "Dirty source",
        )
        require(item["exit_code"] == 0, f"Failed CPU job: {item['name']}")
        require(bool(item["command"]), "Missing reproduction command")
        log = (ROOT / "cpu" / item["log"]).read_text()
        if count is not None:
            require(
                re.search(rf"\b{count} passed\b", log),
                f"Missing outcome: {item['name']}",
            )
        if item["name"] == "feature-final-native-collection":
            require("96 tests collected" in log, "Native collection count differs")
            require(
                "--collect-only" in item["command"],
                "Collection mislabeled as execution",
            )

    preflight = read_json(status["cpu"]["qwen_preflight"])
    require(
        preflight["gpu_training_run"] is False, "Preflight is not a CPU-only result"
    )
    require(
        preflight["text_layers"] == 24 and preflight["dataset_tokens"] == 30727,
        "Preflight differs",
    )
    docs = status["cpu"]["documentation"]
    require(docs["source_commit"] == status["source"]["feature"], "Docs source differs")
    require(
        docs["exit_code"] == 0 and "-W" in docs["command"], "Docs not strictly checked"
    )
    require(
        "build succeeded." in (ROOT / docs["log"]).read_text(),
        "Docs build did not pass",
    )
    require(
        status["gpu"]["status"] == "pending",
        "This bundle does not qualify GPU execution",
    )
    require(not status["gpu"]["completed_for_split_revisions"], "Unexpected GPU claims")
    for item in read_json("mypy-comparison.json"):
        require(item["normalized_diagnostics_equal"], "New mypy diagnostic reported")
        require(
            item["baseline_diagnostics"] == item["candidate_diagnostics"],
            "Mypy counts differ",
        )
    historical = read_json("historical-provenance.json")
    require(
        historical["source_commit"] == status["source"]["original"],
        "Historical source differs",
    )
    for item in historical["files"]:
        actual = hashlib.sha256((ROOT / item["file"]).read_bytes()).hexdigest()
        require(actual == item["sha256"], f"Historical file changed: {item['file']}")

    result = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "result": "passed",
        "checked_files": len(checked),
        "cpu_jobs": len(records),
        "source": status["source"],
        "gpu_qualification": "pending; no split-revision GPU pass is claimed",
    }
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
