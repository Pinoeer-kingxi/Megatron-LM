"""Run fresh two-rank checks on a clean, recorded source commit and save all rank logs."""

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/songbinbin/gdn-offload-artifacts")
OUT = ROOT / "latest-head-20261014"
SOURCE = ROOT / "latest-source-checkout"
COMMIT = "b22c9f7251a480529a4815c8c5fe0ece64734c03"
NAME = "megatron-gdn-public-latest-gpu36"
UUIDS = ["GPU-c8b76e6b-5a4b-7829-4c91-139b71218405", "GPU-40c36f3c-9282-74a6-63d2-077d1e4a1cb7"]
OUT.mkdir(exist_ok=False)


def clean_source():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE, text=True)
    assert head == COMMIT and not status, (head, status)


clean_source()
occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
assert not any(uuid in occupied for uuid in UUIDS), "Selected GPUs have compute processes"
meta = json.loads((ROOT / "publication-20261011/community-container.json").read_text())
container = []
for item in meta["command"]:
    item = item.replace(meta["name"], NAME)
    item = item.replace("/home/songbinbin/gdn-offload-community-update", str(SOURCE))
    item = {"/dev/nvidia0": "/dev/nvidia3", "/dev/nvidia1": "/dev/nvidia6", "CUDA_VISIBLE_DEVICES=0,1": "CUDA_VISIBLE_DEVICES=" + ",".join(UUIDS)}.get(item, item)
    container.append(item)
report = {"source_commit_at_launch": COMMIT, "source_clean_at_launch": True,
          "started_utc": datetime.now(timezone.utc).isoformat(), "physical_gpu_indices": [3, 6],
          "gpu_uuid_order": UUIDS, "container_command": container, "jobs": []}


def save_report():
    (OUT / "run-provenance.json").write_text(json.dumps(report, indent=2) + "\n")


save_report()
subprocess.run(container, check=True, stdout=subprocess.DEVNULL)
prefix = ["docker", "exec", NAME, "/artifacts/uv-container-bin", "run", "--no-project", "--active", "python", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]


def run(name, arguments):
    clean_source()
    command = prefix + ["--log-dir", f"/artifacts/latest-head-20261014/{name}-ranks", "--redirects", "3", "--tee", "3"] + arguments
    job = {"name": name, "source_commit": COMMIT, "source_clean_before_run": True,
           "started_utc": datetime.now(timezone.utc).isoformat(), "command": command,
           "combined_log": f"latest-head-20261014/{name}.log"}
    report["jobs"].append(job)
    save_report()
    print(f"Start {name} at {COMMIT}", flush=True)
    start = time.monotonic()
    with (OUT / f"{name}.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    job["exit_code"] = result.returncode
    job["elapsed_seconds"] = time.monotonic() - start
    job["finished_utc"] = datetime.now(timezone.utc).isoformat()
    clean_source()
    job["source_clean_after_run"] = True
    save_report()
    if result.returncode:
        raise RuntimeError(f"{name} failed, exit {result.returncode}; see {job['combined_log']}")
    print(f"Passed {name}", flush=True)


try:
    preflight = ["docker", "exec", NAME, "/artifacts/uv-container-bin", "run", "--no-project", "--active", "python", "-c",
                 "import json, torch; assert torch.cuda.device_count() == 2; print(json.dumps({'torch':torch.__version__, 'cuda':torch.version.cuda, 'devices':[{'name':torch.cuda.get_device_properties(i).name, 'uuid':str(torch.cuda.get_device_properties(i).uuid)} for i in range(2)]}))"]
    report["cuda_preflight_command"] = preflight
    report["cuda_preflight"] = json.loads(subprocess.check_output(preflight, text=True))
    save_report()
    assert [device["uuid"] for device in report["cuda_preflight"]["devices"]] == [uuid.removeprefix("GPU-") for uuid in UUIDS], report["cuda_preflight"]
    run("rebased-unit", ["/artifacts/four-gpu-acceptance-20261008/run_reduced.py", "-q", "-x", "--confcutdir=tests/unit_tests/ssm",
        "tests/unit_tests/ssm/test_gated_delta_net_offloading.py", "tests/unit_tests/determinism/kernels/test_gdn_gate_backward.py",
        "tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py", "-k",
        "not test_gpt_fine_grained_activation_offloading_correctness_and_memory and not test_fine_grained_activation_offload_with_ep_a2a_overlap_compatibility and not test_fine_grained_activation_offloading_with_cuda_graph"])
    run("rebased-training", ["/artifacts/four-gpu-acceptance-20261008/run_reduced.py", "-q", "-x", "--confcutdir=tests/unit_tests/ssm", "tests/unit_tests/ssm/test_gdn_offload_training.py"])
    records = []
    for fused in (False, True):
        directory = OUT / "qwen" / ("fused" if fused else "unfused")
        directory.mkdir(parents=True)
        baseline = None
        for arm in ("disabled", "1"):
            name = f"qwen-{'fused' if fused else 'unfused'}-{arm}"
            arguments = ["/artifacts/qwen_pool_lifetime_check.py", "--weights", "/artifacts/models/Qwen3.5-0.8B",
                "--data", "/artifacts/training-data/ks-item-sft.jsonl", "--tp", "2", "--sp", "--recompute-norm", "--check-state",
                "--seq-length", "128", "--min-offloaded-tensor-size", "1024", "--warmup", "1", "--iterations", "2", "--output",
                str(directory / f"{arm}.json").replace(str(ROOT), "/artifacts")]
            if fused:
                arguments.append("--fused-pre-gdr")
            if arm != "disabled":
                arguments.extend(["--fraction", arm])
            run(name, arguments)
            paths = [directory / f"{arm}.rank{rank}.json" for rank in (0, 1)]
            results = [json.loads(path.read_text()) for path in paths]
            states = [result["states"] for result in results]
            if baseline is None:
                baseline = states
            else:
                assert states == baseline, name
            for rank, (path, result) in enumerate(zip(paths, results)):
                assert result["gpu_training_run"] and not result["samples"]
                assert len(result["states"]) == 3
                assert result["pool_verified_before_reset"]
                assert result["pool_in_use_before_resets"] == [0] * (2 if arm == "disabled" else 5)
                assert result["offload_status"]["selected_transfer_bytes"] == (0 if arm == "disabled" else 66015232)
                records.append({"fused": fused, "arm": arm, "rank": rank, "exact_steps": 3,
                                "file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "selected_transfer_bytes": result["offload_status"]["selected_transfer_bytes"]})
    validation = {"status": "passed", "source_commit": COMMIT, "fresh_two_rank_runs": 4, "rank_files": 8,
                  "exact_complete_rank_step_states": 24, "all_pre_reset_pool_counts_zero": True, "timed_samples": 0, "results": records}
    (OUT / "qwen-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    report["status"] = "passed"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_report()
    print("Passed all six latest-source GPU jobs and 24 complete Qwen rank-step states.", flush=True)
finally:
    subprocess.run(["docker", "stop", "-t", "10", NAME], check=True, stdout=subprocess.DEVNULL)
