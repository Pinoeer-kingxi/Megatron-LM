"""Run one pinned-source test job and retain command, rank logs and outcomes."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
name, source, ranks, *arguments = sys.argv[1:]
source = Path(source)
directory = root / name
directory.mkdir(parents=True)
revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
assert not dirty, dirty
artifacts = Path("/home/songbinbin/gdn-offload-artifacts")
pythonpath = ":".join(map(str, [source, root / "scripts", artifacts / "flash-attn-python", artifacts / "causal-conv1d-python", artifacts / "bridge-python", artifacts / "split-pr-20261015/bridge/src"]))
command = ["docker", "exec", "--workdir", str(source), "--env", "PYTHONPATH=" + pythonpath,
           "megatron-offload-review-20261021", "/home/songbinbin/.venvs/megatron-gdn-offload/bin/python",
           "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=" + ranks,
           "--log-dir", str(directory / "ranks"), "--redirects", "3", "--tee", "3",
           str(root / "scripts/rank_env.py"),
           str(root / "scripts/rank_distributed_pytest.py"), str(directory / "reports"),
           "-q", "-x", "-p", "no:cacheprovider", "--confcutdir=tests/unit_tests/pipeline_parallel"] + arguments
manifest = dict(source=str(source), revision=revision, command=command, started_unix=time.time(),
                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
manifest_path = directory / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print("Starting " + name + " at " + revision, flush=True)
with (directory / "launcher.log").open("w") as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
manifest.update(exit_code=result.returncode, finished_unix=time.time())
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print("Finished " + name + ": " + str(result.returncode), flush=True)
sys.exit(result.returncode)
