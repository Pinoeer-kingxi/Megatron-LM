"""Run the repository formatter against a local mirror of pinned upstream main."""

import json
import subprocess
import sys
from pathlib import Path

name, source = sys.argv[1:]
root = Path(__file__).resolve().parents[1]
clone = root / ("lint-" + name)
subprocess.run(["git", "clone", "--shared", source, str(clone)], check=True)
revision = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
subprocess.run(["git", "-C", str(clone), "checkout", "--detach", revision], check=True)
subprocess.run(["git", "-C", str(clone), "config", f"url.{(root / 'lint-upstream.git').as_uri()}.insteadOf",
                "https://github.com/NVIDIA/Megatron-LM.git"], check=True)
command = ["docker", "exec", "--workdir", str(clone), "--env", "CHECK_ONLY=true",
           "--env", "PYTHONPATH=" + str(clone), "megatron-offload-review-20261021",
           "bash", "tools/autoformat.sh"]
with (root / f"lint-{name}.log").open("w") as stream:
    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
manifest = dict(revision=revision, command=command, exit_code=result.returncode,
                main="6c956376e3ef22528413ff35412ad8b938272986")
(root / f"lint-{name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(name, result.returncode)
sys.exit(result.returncode)
