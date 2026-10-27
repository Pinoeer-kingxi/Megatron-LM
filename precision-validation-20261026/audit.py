"""Audit recorded outcomes without requiring CUDA, model weights or training data."""
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
revision = "688159928298844dc00d9d3a9ce33774735d997c"


def read(path):
    return json.loads((root / path).read_text())


def manifest(directory, source=revision, exit_code=0):
    item = read(f"{directory}/manifest.json")
    assert item["revision"] == item["revision_after"] == source, directory
    assert item["exit_code"] == exit_code and not item["dirty_after"], directory
    return item


def states(directory, source=revision, offloaded=False):
    manifest(directory, source)
    item = read(f"{directory}/result.rank0.json")
    assert item["gpu_training_run"] and item["pool_verified_before_reset"], directory
    assert item["text_layers"] == 24 and len(item["states"]) == 3, directory
    assert bool(item["offload_status"]["selected_transfer_bytes"]) == offloaded, directory
    for step in item["states"]:
        assert len(step["grads"]) == len(step["weights"]) == 230, directory
        assert step["optimizer"] and math.isfinite(step["grad_norm"]), directory
        assert step["grad_norm"] > 0 and all(map(math.isfinite, step["losses"])), directory
        optimizer = step["optimizer"]
        assert len(optimizer["optimizer"]["state"]) == 230, directory
        assert all({"exp_avg", "exp_avg_sq"} <= set(state)
                   for state in optimizer["optimizer"]["state"].values()), directory
        assert sum(map(len, optimizer["fp32_from_fp16_params"])) == 230, directory
    return item["states"]


def tests(directory, passed, source=revision, failed=0):
    manifest(directory, source, int(failed > 0))
    report = read(f"{directory}/reports/rank-0.json")
    outcomes = [row["outcome"] for row in report["outcomes"]]
    assert outcomes.count("passed") == passed and outcomes.count("failed") == failed
    assert set(outcomes) <= {"passed", "failed"}, directory


qualifications = {}
for recipe, prefix in (
    ("fp8-delayed", "qwen-sm120-stable-fp8-delayed"),
    ("fp4", "qwen-sm120-final-gpu3-fp4"),
    ("fp8-tensorwise", "qwen-sm120-final-gpu3-fp8-tensorwise"),
):
    outputs, kernels, cache_paths = [], [], []
    for arm in ("disabled", "repeated-disabled", "offloaded"):
        directory = f"{prefix}-{arm}"
        outputs.append(states(directory, offloaded=arm == "offloaded"))
        record = read(f"{directory}/runtime.json")
        expected = "NVFP4" if recipe == "fp4" else "Float8"
        assert any(expected in key and value > 0
                   for key, value in record["gemm_operand_types"].items()), directory
        assert record["inductor_kernels"], directory
        kernels.append(record["inductor_kernels"])
        cache_paths.append(read(f"{directory}/manifest.json")["inductor_cache"])
    assert outputs[0] == outputs[1] == outputs[2], recipe
    assert kernels[0] == kernels[1] == kernels[2], recipe
    assert len(set(cache_paths)) == 1, recipe
    assert read(f"{prefix}-comparison.json")["exact_states"], recipe
    qualifications[recipe] = True

tests("precision-tool-final-gpu1", 13)
tests("precision-native-final-gpu3", 47)
old = "ebdb25fc493b528764a4116313effaf8504eec3d"
fixed = "e553078013811377b5991ee80c599dc153d8efa2"
tests("fp4-context-regression-old", 0, old, failed=1)
tests("fp4-context-regression-fixed", 1, fixed)
assert (read("fp4-context-regression-old/manifest.json")["test_sha256"]
        == read("fp4-context-regression-fixed/manifest.json")["test_sha256"])
assert (hashlib.sha256((root / "source-files/test_qwen_gdn_offload_training.py").read_bytes()).hexdigest()
        == read("fp4-context-regression-fixed/manifest.json")["test_sha256"])

# The cold-cache controls remain unequal; they must not be relabelled as passes.
cold = [states(f"qwen-sm120-fp8-compile-gpu1-{label}", fixed)
        for label in ("seed0", "seed1", "seed0-repeat")]
assert cold[0] != cold[2]
selections = read("inductor-cold-selection.json")
assert len({(row["selection"]["BT"], row["selection"]["num_warps"])
            for row in selections}) == 3

# Preserve the failed resource-constrained attempt separately from numerical checks.
manifest("qwen-sm120-stable-fp4-disabled", exit_code=1)
assert any("OutOfMemoryError" in path.read_text()
           for path in (root / "qwen-sm120-stable-fp4-disabled").rglob("*.log"))

capture = read("l2norm-real-input-gpu3/manifest.json")
assert capture["revision"] == capture["revision_after"] == revision
assert capture["exit_codes"] == [0, 1] and not capture["dirty_after"]
for name, relative in (("capture_l2norm.py", "scripts/capture_l2norm.py"),
                       ("replay_l2norm.py", "scripts/replay_l2norm.initial.py")):
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == capture["scripts"][name]
replay = manifest("l2norm-replay-gpu3")
assert replay["script_sha256"] == hashlib.sha256((root / "scripts/replay_l2norm.py").read_bytes()).hexdigest()
numerical = read("l2norm-configuration-comparison.json")
assert numerical["shape"] == [4096, 128]
assert all(row["repeat_exact"] for row in numerical["variants"])
assert any(row["matches_captured_inductor_dx"] for row in numerical["variants"])
assert any(row["different_elements"] > 0 for row in numerical["comparisons"])

for line in (root / "SHA256SUMS").read_text().splitlines():
    expected, name = line.split("  ", 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name

print(json.dumps(dict(integrity="passed", source=revision,
                      complete_qwen_exact=qualifications, tool_tests_passed=13,
                      native_tests_passed=47, old_code_regression="expected failure",
                      kernel_replay="repeat-exact configurations, cross-configuration rounding differs",
                      cold_cache_control="unequal states retained",
                      resource_oom="retained, not counted as passed"), indent=2))
