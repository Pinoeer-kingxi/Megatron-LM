"""Repeat fused full-Qwen controls with actual vector capture, without kernel changes."""

import json
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = Path("/home/songbinbin/gdn-offload-artifacts/review-20261021/feature")
revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
for norm in (False, True):
    for arm in ("disabled", "repeat-disabled", "offloaded"):
        original = root / f"four-qwen-f1-n0-{arm}"
        directory = root / f"four-qwen-captured-f1-n{int(norm)}-{arm}"
        directory.mkdir()
        command = json.loads((original / "manifest.json").read_text())["command"]
        command = [item.replace(str(original), str(directory)) for item in command]
        index = command.index(str(source / "tools/ssm/qwen_gdn_offload_training.py"))
        command[index] = str(root / "scripts/capture_qwen_vectors.py")
        # Preserve the same per-rank cache across fresh-process arms of each setting.
        command = [item.replace("cache/four-qwen-f1-n0", f"cache/four-qwen-f1-n{int(norm)}") for item in command]
        if norm:
            command.append("--recompute-norm")
        manifest = dict(revision=revision, command=command, started_unix=time.time(),
                        capture="Small vector capture only; native arithmetic and kernel choices unchanged.")
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
        print("Starting " + directory.name, flush=True)
        with (directory / "launcher.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        manifest.update(exit_code=result.returncode, finished_unix=time.time())
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        assert result.returncode == 0, directory
        assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == revision
