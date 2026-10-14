"""Check checksum rejection and real Qwen mismatch/pool failure behavior."""

import hashlib
import json
import runpy
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
functions = runpy.run_path(str(ROOT / "verify.py"))


def rejects(callback, expected):
    try:
        callback()
    except ValueError as error:
        assert expected in str(error), error
        print(f"PASS rejection: {expected}")
    else:
        raise AssertionError(f"Expected rejection: {expected}")


with tempfile.TemporaryDirectory(prefix="gdn-public-evidence-regression-") as temporary:
    root = Path(temporary) / "evidence"
    shutil.copytree(ROOT, root)
    functions["verify_checksums"](root)
    functions["verify_qwen"](root, "latest-head-20261014/qwen-validation.json")
    print("PASS unmodified checksum and full-state comparisons")
    log = root / "latest-head-20261014/rebased-unit.log"
    original_log = log.read_bytes()
    log.write_bytes(original_log + b"tampered\n")
    rejects(lambda: functions["verify_checksums"](root), "SHA-256 mismatch")
    log.write_bytes(original_log)
    report_path = root / "latest-head-20261014/qwen-validation.json"
    report = json.loads(report_path.read_text())
    record = next(item for item in report["results"] if item["arm"] == "1" and item["rank"] == 0)
    path = root / record["file"]
    result = json.loads(path.read_text())
    original_result = path.read_bytes()

    def save_result():
        path.write_text(json.dumps(result, indent=2) + "\n")
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    result["states"][0]["grad_norm"] += 1.0
    save_result()
    rejects(
        lambda: functions["verify_qwen"](root, "latest-head-20261014/qwen-validation.json"),
        "Qwen state mismatch",
    )
    result = json.loads(original_result)
    result["pool_in_use_before_resets"][0] = 1
    save_result()
    rejects(
        lambda: functions["verify_qwen"](root, "latest-head-20261014/qwen-validation.json"),
        "Pinned pool still in use",
    )
print("Passed unmodified evidence and all three failure regressions; original files unchanged.")
