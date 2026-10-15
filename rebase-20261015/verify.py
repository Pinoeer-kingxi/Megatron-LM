"""Verify this pinned evidence bundle with Python's standard library."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
checked = 0
for line in (root / "SHA256SUMS").read_text().splitlines():
    expected, name = line.split(None, 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    checked += 1

state = json.loads((root / "source-after.json").read_text())
for category in ("cpu", "format", "docs"):
    for record in json.loads((root / (category + "-validation.json")).read_text()):
        assert record["source_commit"] == state[record["branch"]]
        assert record["exit_code"] == 0, record["name"]
        assert record["source_clean_before_run"] and record["source_clean_after_run"]
        assert (root / record["log"]).is_file()

for name, result in (("shared-cpu.log", "5 passed"), ("gate-manifest.log", "272 passed"), ("feature-manifest.log", "272 passed"), ("feature-cpu.log", "30 passed"), ("native-training-collection.log", "96 tests collected"), ("strict-docs.log", "build succeeded")):
    assert result in (root / name).read_text(), name

for record in json.loads((root / "mypy-comparison.json").read_text()):
    assert record["baseline"] == state["base"]
    assert record["normalized_diagnostics_equal"]
    assert not record["introduced"] and not record["removed"]

assert json.loads((root / "source-audit.json").read_text())["prerequisite_gate_patch_identical"]
assert not json.loads((root / "gpu-availability.json").read_text())["gpu_run"]
assert not json.loads((root / "qwen-preflight.rank0.json").read_text())["gpu_training_run"]
print(json.dumps({"verified_files": checked, "source": state, "gpu_training_verified": False}, indent=2))
