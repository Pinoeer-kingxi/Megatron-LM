"""Run the native four-rank matrix and complete Qwen TP2/SP/PP2 state comparisons."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = Path("/home/songbinbin/gdn-offload-artifacts/review-20261021/feature")
artifacts = Path("/home/songbinbin/gdn-offload-artifacts")
revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
container = "megatron-offload-review-four-20261021"
environment = dict(os.environ, OFFLOAD_TEST_CONTAINER=container)
subprocess.run([sys.executable, str(root / "scripts/run_job.py"), "shared-four-phase",
                str(source.parent / "shared"), "4",
                "tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py",
                "tests/unit_tests/pipeline_parallel/test_schedules.py", "-k",
                "after_evaluation_with_growing_sequences or test_reset_activation_offload or inline_reload_is_ordered"],
               env=environment, check=True)
subprocess.run([sys.executable, str(root / "scripts/run_job.py"), "feature-four-native",
                str(source), "4", "tests/unit_tests/ssm/test_gdn_offload_training.py"],
               env=environment, check=True)
pythonpath = ":".join(map(str, [source, artifacts / "flash-attn-python", artifacts / "causal-conv1d-python", artifacts / "bridge-python", artifacts / "split-pr-20261015/bridge/src"]))
results = []
for fused in (False, True):
    for norm in (False, True):
        arms = []
        for arm in ("disabled", "repeat-disabled", "offloaded"):
            directory = root / f"four-qwen-f{int(fused)}-n{int(norm)}-{arm}"
            directory.mkdir()
            command = ["docker", "exec", "--workdir", str(source), "--env", "PYTHONPATH=" + pythonpath,
                       "--env", "TRITON_CACHE_DIR=" + str(root / f"cache/four-qwen-f{int(fused)}-n{int(norm)}"),
                       container, "/home/songbinbin/.venvs/megatron-gdn-offload/bin/python",
                       "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4",
                       "--log-dir", str(directory / "ranks"), "--redirects", "3",
                       str(root / "scripts/rank_env.py"), str(source / "tools/ssm/qwen_gdn_offload_training.py"),
                       "--weights", str(artifacts / "models/Qwen3.5-0.8B"),
                       "--data", str(artifacts / "training-data/ks-item-sft.jsonl"),
                       "--tp", "2", "--sp", "--pp", "2", "--seq-length", "128",
                       "--microbatches", "4", "--warmup", "1", "--iterations", "2",
                       "--min-offloaded-tensor-size", "1024", "--check-state",
                       "--output", str(directory / "result.json")]
            if arm == "offloaded":
                command += ["--fraction", "1"]
            if norm:
                command += ["--recompute-norm"]
            if fused:
                command += ["--fused-pre-gdr"]
            manifest = dict(revision=revision, command=command, started_unix=time.time())
            path = directory / "manifest.json"
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            print("Starting " + directory.name, flush=True)
            with (directory / "launcher.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            manifest.update(exit_code=result.returncode, finished_unix=time.time())
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            assert result.returncode == 0, directory
            assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == revision
            outputs = [json.loads((directory / f"result.rank{rank}.json").read_text()) for rank in range(4)]
            for rank, output in enumerate(outputs):
                assert output["gpu_training_run"] and output["pool_verified_before_reset"]
                assert output["rank"] == rank and output["text_layers"] == 24
                assert [step["step"] for step in output["states"]] == [0, 1, 2]
                assert (output["offload_status"]["selected_transfer_bytes"] > 0) == (arm == "offloaded")
                assert all(len(step["losses"]) == (4 if rank >= 2 else 0) for step in output["states"])
                assert all(step[field] for step in output["states"] for field in ("grads", "weights", "optimizer"))
            arms.append([output["states"] for output in outputs])
        comparison = dict(fused=fused, recompute_norm=norm, ranks=4, steps=3,
                          disabled_repeat_exact=arms[0] == arms[1], offloaded_exact=arms[0] == arms[2])
        results.append(comparison)
        (root / "four-qwen-comparison.json").write_text(json.dumps(dict(revision=revision, comparisons=results), indent=2) + "\n")
        assert comparison["disabled_repeat_exact"], comparison
        assert comparison["offloaded_exact"], comparison
