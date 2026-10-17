"""Compare uninterrupted and genuinely restarted full-Qwen training on GPU 2."""

import json
from pathlib import Path
import subprocess
import time

from run_gpu_acceptance import ARTIFACTS, VENV, hash_file, python_path, gpu_inventory, require_idle

parent = Path(__file__).resolve().parents[1]
root = parent / "checkpoint"
source = root / "source"
container = "megatron-gdn-resume-dev-20261016"
source_root = ARTIFACTS / "split-pr-20261015"
manifest = {
    "status": "running", "source": json.loads((root / "source.json").read_text()),
    "gpu": gpu_inventory(["2"])[0], "started_unix": time.time(),
    "source_archive_sha256": hash_file(root / "source.tar"),
    "container": json.loads((parent / "resume-container.json").read_text()),
    "script_sha256": {name: hash_file(parent / "scripts" / name) for name in ("run_checkpoint.py", "checkpoint_capture.py", "rank_env.py")},
    "comparison": "Six continuous steps vs two prefix plus four steps in a fresh resumed process",
    "jobs": [],
}


def save():
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


require_idle([manifest["gpu"]])
try:
    for fused, norm, offload in [(False, False, False), (False, False, True), (False, True, True), (True, False, True), (True, True, True)]:
        configuration = f"f{int(fused)}-n{int(norm)}-o{int(offload)}"
        cache = root / "cache" / configuration
        checkpoint = root / configuration / "native-checkpoint"
        for arm, count in [("continuous", 6), ("prefix", 2), ("resumed", 4)]:
            name = f"{configuration}-{arm}"
            directory = root / name
            directory.mkdir()
            command = [
                "docker", "exec", "--workdir", str(source),
                "--env", "PYTHONPATH=" + python_path(source, source_root),
                "--env", "PYTHONDONTWRITEBYTECODE=1",
                "--env", f"TRITON_CACHE_DIR={cache / 'triton'}",
                "--env", f"TORCHINDUCTOR_CACHE_DIR={cache / 'inductor'}",
                container, str(VENV / "bin/python"), "-m", "torch.distributed.run", "--standalone",
                "--nproc-per-node=1", "--log-dir", str(directory / "ranks"), "--redirects", "3", "--tee", "3",
                str(parent / "scripts/rank_env.py"), str(parent / "scripts/checkpoint_capture.py"),
                "--weights", str(ARTIFACTS / "models/Qwen3.5-0.8B"),
                "--data", str(ARTIFACTS / "training-data/ks-item-sft.jsonl"),
                "--seq-length", "128", "--microbatches", "4", "--warmup", "1", "--iterations", str(count - 1),
                "--min-offloaded-tensor-size", "1024", "--check-state", "--output", str(directory / "result.json"),
            ]
            if fused:
                command += ["--fused-pre-gdr"]
            if norm:
                command += ["--recompute-norm"]
            if offload:
                command += ["--fraction", "1"]
            if arm == "prefix":
                command += ["--save-checkpoint", str(checkpoint)]
            elif arm == "resumed":
                command += ["--load-checkpoint", str(checkpoint)]
            job = {"name": name, "command": command, "started_unix": time.time()}
            manifest["jobs"].append(job)
            save()
            print("Starting " + name, flush=True)
            with (directory / "launcher.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            job.update(exit_code=result.returncode, finished_unix=time.time())
            save()
            assert result.returncode == 0, f"Inspect {directory / 'launcher.log'}"
            print("Completed " + name, flush=True)
        continuous = json.loads((root / f"{configuration}-continuous/result.rank0.json").read_text())
        prefix = json.loads((root / f"{configuration}-prefix/result.rank0.json").read_text())
        resumed = json.loads((root / f"{configuration}-resumed/result.rank0.json").read_text())
        assert len(continuous["states"]) == 6 and len(prefix["states"]) == 2 and len(resumed["states"]) == 4
        assert continuous["first_step"] == prefix["first_step"] == 0 and resumed["first_step"] == 2
        assert [state["step"] for state in resumed["states"]] == [2, 3, 4, 5]
        assert resumed["pool_verified_before_reset"] and resumed["text_layers"] == 24
        assert (resumed["offload_status"]["selected_transfer_bytes"] > 0) == offload
        if not fused:
            assert continuous["states"] == prefix["states"] + resumed["states"], configuration
        rng = json.loads((root / f"{configuration}-resumed/restored-rng.rank0.json").read_text())
        assert rng["all_rng_states_exact"] and rng["next_step"] == 2
        manifest.setdefault("completed_configurations", []).append({"name": configuration, "unfused_exact": not fused, "rng_exact": True})
        save()
    manifest["status"] = "training_completed_pending_fused_comparison"
except BaseException as error:
    manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
    raise
finally:
    manifest["finished_unix"] = time.time()
    save()
