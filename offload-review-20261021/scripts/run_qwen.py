"""Run matched full-Qwen state comparisons on the reserved two GPUs."""

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = Path("/home/songbinbin/gdn-offload-artifacts/review-20261021/feature")
artifacts = Path("/home/songbinbin/gdn-offload-artifacts")
revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
pythonpath = ":".join(map(str, [source, artifacts / "flash-attn-python", artifacts / "causal-conv1d-python", artifacts / "bridge-python", artifacts / "split-pr-20261015/bridge/src"]))
for topology, topology_args in (("tp-sp", ["--tp", "2", "--sp"]), ("pp", ["--pp", "2", "--recompute-norm"])):
    arms = []
    for arm in ("disabled", "repeat-disabled", "offloaded"):
        directory = root / ("qwen-" + topology + "-" + arm)
        directory.mkdir()
        command = ["docker", "exec", "--workdir", str(source), "--env", "PYTHONPATH=" + pythonpath,
                   "--env", "TRITON_CACHE_DIR=" + str(root / ("cache/qwen-" + topology)),
                   "megatron-offload-review-20261021", "/home/songbinbin/.venvs/megatron-gdn-offload/bin/python",
                   "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
                   "--log-dir", str(directory / "ranks"), "--redirects", "3",
                   str(root / "scripts/rank_env.py"), str(source / "tools/ssm/qwen_gdn_offload_training.py"),
                   "--weights", str(artifacts / "models/Qwen3.5-0.8B"),
                   "--data", str(artifacts / "training-data/ks-item-sft.jsonl"),
                   "--seq-length", "128", "--microbatches", "4", "--warmup", "1", "--iterations", "2",
                   "--min-offloaded-tensor-size", "1024", "--check-state",
                   "--output", str(directory / "result.json"), *topology_args]
        if arm == "offloaded":
            command += ["--fraction", "1"]
        manifest = dict(revision=revision, command=command, started_unix=time.time(),
                        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print("Starting " + directory.name, flush=True)
        with (directory / "launcher.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        manifest.update(exit_code=result.returncode, finished_unix=time.time())
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        assert result.returncode == 0, directory
        assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == revision
        outputs = [json.loads((directory / f"result.rank{rank}.json").read_text()) for rank in range(2)]
        for rank, output in enumerate(outputs):
            assert output["gpu_training_run"] and output["pool_verified_before_reset"]
            assert output["rank"] == rank and output["text_layers"] == 24
            assert [item["step"] for item in output["states"]] == [0, 1, 2]
            assert (output["offload_status"]["selected_transfer_bytes"] > 0) == (arm == "offloaded")
            for step in output["states"]:
                assert math.isfinite(step["grad_norm"]) and step["grad_norm"] > 0
                assert len(step["losses"]) == (4 if topology == "tp-sp" or rank == 1 else 0)
                assert all(math.isfinite(loss) for loss in step["losses"])
                assert all(step[field] for field in ("grads", "weights", "optimizer"))
        arms.append([output["states"] for output in outputs])
        print("Completed " + directory.name, flush=True)
    assert arms[0] == arms[1], "Repeated disabled baseline differs: " + topology
    assert arms[0] == arms[2], "Offloaded complete state differs: " + topology
    comparison = dict(revision=revision, topology=topology, ranks=2, steps=3,
                      disabled_repeat_exact=True, offloaded_exact=True,
                      fields=["losses", "grad_norm", "grads", "weights", "optimizer"])
    (root / ("qwen-" + topology + "-comparison.json")).write_text(json.dumps(comparison, indent=2) + "\n")
