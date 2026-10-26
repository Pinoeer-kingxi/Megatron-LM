import subprocess
from pathlib import Path

remote = "A6000PRO"
base = "/home/lwx/.cache/megatron-gdn-20261026"
with subprocess.Popen(["docker", "export", "megatron-gdn-export-20261026"], stdout=subprocess.PIPE) as exporter:
    subprocess.run(["ssh", remote, "tar --no-same-owner -xpf - -C " + base + "/rootfs"], stdin=exporter.stdout, check=True)
    exporter.stdout.close()
    assert exporter.wait() == 0
for local, target, extra in [
    ("/home/songbinbin/.venvs/megatron-gdn-offload/", "/venv/", []),
    ("/data/songbinbin/minicoda/lib/python3.12/", "/python/lib/python3.12/", ["--exclude=site-packages", "--exclude=__pycache__"]),
    ("/data/songbinbin/minicoda/bin/python3.12", "/python/bin/python3.12", []),
]:
    subprocess.run(["rsync", "-a", "--info=progress2", *extra, local, remote + ":" + base + target], check=True)
print("Transfer complete", flush=True)
