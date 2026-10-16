"""Run the frozen PR acceptance jobs in this host's validated container environment."""

import argparse
import csv
import hashlib
import json
import shlex
import shutil
import signal
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from verify_results import hash_file, summarize_benchmarks, verify_pytest, verify_qwen

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent
SOURCES = json.loads((HERE / "source-after.json").read_text())
IMAGE = "sha256:061e8fda67a4dd1c1c635129d4ba0442b7a093cc6fd737423e9589eb6be22034"
VENV = Path("/home/songbinbin/.venvs/megatron-gdn-offload")
RUNTIME_FILES = (
    "run.sh",
    "run_gpu_acceptance.py",
    "rank_env.py",
    "rank_pytest.py",
    "verify_results.py",
    "benchmark_gate.py",
    "source-after.json",
    "bridge-compatibility.patch",
)
MANAGER_TEST = "tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py"
MODEL_SELECTION = (
    "test_gpt_fine_grained_activation_offloading_correctness_and_memory or "
    "test_fine_grained_activation_offload_with_ep_a2a_overlap_compatibility or "
    "test_fine_grained_activation_offloading_with_cuda_graph"
)


def source_state(root: Path) -> dict:
    """Reject changed checkouts instead of silently testing another revision."""
    result = {}
    for name in ("base", "shared", "gate", "feature"):
        directory = root / name
        head = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain"], text=True
        )
        if head != SOURCES[name] or dirty:
            raise ValueError(f"{directory}: expected clean {SOURCES[name]}, got {head}\n{dirty}")
        result[name] = head
    return result


def bridge_state(root: Path) -> dict:
    """Check the existing import-only Bridge overlay before acquiring devices."""
    bridge = root / "bridge"
    head = subprocess.check_output(
        ["git", "-C", str(bridge), "rev-parse", "HEAD"], text=True
    ).strip()
    patch = subprocess.check_output(["git", "-C", str(bridge), "diff", "HEAD"])
    if (
        head != "c860f8a5bc5fddd78690f32baa0b8696774308b4"
        or patch != (HERE / "bridge-compatibility.patch").read_bytes()
    ):
        raise ValueError("Bridge must match the pinned revision and canonical-import patch")
    return {"revision": head, "patch_sha256": hashlib.sha256(patch).hexdigest()}


def gpu_inventory(selectors: list[str]) -> list[dict]:
    """Resolve nvidia-smi indices/UUIDs to UUIDs and actual device minor numbers."""
    rows = csv.reader(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
        ).splitlines(),
        skipinitialspace=True,
    )
    aliases = {index: gpu_uuid for index, gpu_uuid in rows}
    xml = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True))
    available = {
        gpu.findtext("uuid"): {
            "uuid": gpu.findtext("uuid"),
            "minor": int(gpu.findtext("minor_number")),
            "name": gpu.findtext("product_name"),
            "driver": xml.findtext("driver_version"),
        }
        for gpu in xml.findall("gpu")
    }
    unknown = [value for value in selectors if aliases.get(value, value) not in available]
    if unknown:
        raise ValueError(f"Unknown GPU indices/UUIDs: {unknown}")
    selected = [available[aliases.get(value, value)] for value in selectors]
    if len({gpu["uuid"] for gpu in selected}) != len(selected):
        raise ValueError("GPU selectors contain duplicate devices")
    return selected


def require_idle(gpus: list[dict]) -> None:
    """Check at job boundaries; device reservation remains the caller's responsibility."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    selected = {gpu["uuid"] for gpu in gpus}
    busy = [
        row for row in csv.reader(output.splitlines(), skipinitialspace=True) if row[0] in selected
    ]
    if busy:
        raise RuntimeError(f"Selected GPUs have existing compute processes: {busy}")


def make_jobs(args) -> list[dict]:
    """Extract the existing acceptance matrix; each Qwen arm is a fresh process."""
    jobs = []
    ranks = 1 if args.stage == "single" else 4

    def pytest_job(name, source, path, expected, selection=None, allow_skips=False):
        directory = args.output / name
        command = [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={ranks}",
            "--log-dir",
            str(directory / "ranks"),
            "--redirects",
            "3",
            "--tee",
            "3",
            str(args.output / "scripts/rank_pytest.py"),
            str(directory / "reports"),
            "-q",
            "-x",
            "-p",
            "no:cacheprovider",
            "--confcutdir=" + str(Path(path).parent),
            path,
        ]
        if selection:
            command += ["-k", selection]
        jobs.append(
            {
                "name": name,
                "source": source,
                "command": command,
                "expected_tests": expected,
                "allow_skips": allow_skips,
            }
        )

    if args.stage == "single":
        pytest_job(
            "shared-races",
            "shared",
            MANAGER_TEST,
            4,
            "test_tensor_pop_waits_for_delayed_offload or "
            "test_offload_with_growing_microbatches or test_bulk_offload_waits_for_pooled_h2d",
        )
        pytest_job(
            "gate-replay",
            "gate",
            "tests/unit_tests/determinism/kernels/test_gdn_gate_backward.py",
            16,
        )
        pytest_job(
            "gdn-scopes", "feature", "tests/unit_tests/ssm/test_gated_delta_net_offloading.py", 73
        )
        for batch in (1, 4):
            for length in (2048, 4096):
                for heads in (16, 32):
                    for repeat in range(3):
                        name = f"gate-bench-{batch}-{length}-{heads}-{repeat}"
                        jobs.append(
                            {
                                "name": name,
                                "source": "gate",
                                "command": [
                                    str(args.output / "scripts/benchmark_gate.py"),
                                    "--baseline",
                                    str(
                                        args.source_root
                                        / "base/megatron/core/fusions/fused_pre_gated_delta_rule.py"
                                    ),
                                    "--candidate",
                                    str(
                                        args.source_root
                                        / "gate/megatron/core/fusions/fused_pre_gated_delta_rule.py"
                                    ),
                                    "--batch-size",
                                    str(batch),
                                    "--seq-length",
                                    str(length),
                                    "--value-heads",
                                    str(heads),
                                    "--output",
                                    str(args.output / name / "result.json"),
                                ],
                            }
                        )
    else:
        pytest_job("shared-groups", "shared", MANAGER_TEST, 16, "not (" + MODEL_SELECTION + ")")
        pytest_job("shared-models", "shared", MANAGER_TEST, 24, MODEL_SELECTION, allow_skips=True)
        pytest_job(
            "native-training", "feature", "tests/unit_tests/ssm/test_gdn_offload_training.py", 96
        )
        for fused in (False, True):
            for norm in (False, True):
                for offload in (False, True):
                    name = f"qwen-f{int(fused)}-n{int(norm)}-o{int(offload)}"
                    command = [
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc-per-node=4",
                        "--log-dir",
                        str(args.output / name / "ranks"),
                        "--redirects",
                        "3",
                        "--tee",
                        "3",
                        str(args.output / "scripts/rank_env.py"),
                        "tools/ssm/qwen_gdn_offload_training.py",
                        "--weights",
                        str(args.weights),
                        "--data",
                        str(args.data),
                        "--tp",
                        "2",
                        "--sp",
                        "--pp",
                        "2",
                        "--seq-length",
                        "128",
                        "--min-offloaded-tensor-size",
                        "1024",
                        "--warmup",
                        "1",
                        "--iterations",
                        "2",
                        "--check-state",
                        "--output",
                        str(args.output / name / "result.json"),
                    ]
                    if fused:
                        command += ["--fused-pre-gdr"]
                    if norm:
                        command += ["--recompute-norm"]
                    if offload:
                        command += ["--fraction", "1"]
                    jobs.append({"name": name, "source": "feature", "command": command})
    return jobs


def python_path(source: Path, source_root: Path) -> str:
    """Use the current checkout before optional dependency overlays."""
    return ":".join(
        map(
            str,
            [
                source,
                ARTIFACTS / "flash-attn-python",
                ARTIFACTS / "causal-conv1d-python",
                ARTIFACTS / "bridge-python",
                source_root / "bridge/src",
            ],
        )
    )


def container_command(args, gpus, name) -> list[str]:
    """Reuse the manually mapped devices/libraries of the validated local environment."""
    common = subprocess.check_output(
        [
            "git",
            "-C",
            str(args.source_root / "feature"),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        text=True,
    ).strip()
    command = [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--name",
        name,
        "--ipc=host",
        "--security-opt",
        "label=disable",
    ]
    for gpu in gpus:
        command += ["--device", f"/dev/nvidia{gpu['minor']}"]
    for device in ("nvidiactl", "nvidia-uvm", "nvidia-uvm-tools"):
        command += ["--device", "/dev/" + device]
    for source, target, mode in [
        (ARTIFACTS, ARTIFACTS, "ro"),
        (ARTIFACTS, Path("/artifacts"), "ro"),
        (Path(common).parent, Path(common).parent, "ro"),
        (args.source_root, args.source_root, "ro"),
        (VENV, VENV, "ro"),
        (Path("/data/songbinbin/minicoda"), Path("/data/songbinbin/minicoda"), "ro"),
        (Path("/usr/local/cuda-13.0"), Path("/usr/local/cuda"), "ro"),
        (args.output, args.output, "rw"),
    ]:
        command += [
            "--mount",
            f"type=bind,source={source},target={target}" + (",readonly" if mode == "ro" else ""),
        ]
    if args.stage == "four":
        for path in (args.weights, args.data):
            command += ["--mount", f"type=bind,source={path},target={path},readonly"]
    for library in (
        "libcuda.so.1",
        "libnvidia-ml.so.1",
        "libnvidia-nvvm.so.4",
        "libnvidia-ptxjitcompiler.so.1",
    ):
        command += ["--volume", f"/lib/x86_64-linux-gnu/{library}:/opt/nvidia/{library}:ro"]
    environment = {
        "PATH": f"{VENV}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/opt/nvidia",
        "CUDA_VISIBLE_DEVICES": ",".join(gpu["uuid"] for gpu in gpus),
        "CUDA_DEVICE_MAX_CONNECTIONS": "8" if args.stage == "single" else "1",
        "NCCL_MAX_NCHANNELS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_CPU_OFFLOAD_V1": "1",
        "OMP_NUM_THREADS": "1",
        "TRITON_CACHE_DIR": str(args.output / "cache/triton"),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": "*",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    for key, value in environment.items():
        command += ["--env", f"{key}={value}"]
    return command + [IMAGE, "sleep", "infinity"]


def job_command(args, job, container) -> list[str]:
    """Build argv without shell interpolation."""
    source = args.source_root / job["source"]
    return [
        "docker",
        "exec",
        "--workdir",
        str(source),
        "--env",
        "PYTHONPATH=" + python_path(source, args.source_root),
        container,
        str(VENV / "bin/python"),
        *job["command"],
    ]


def stop_run(signum, _frame) -> None:
    """Let termination requests reach the owned container's cleanup boundary."""
    raise KeyboardInterrupt(f"Received {signal.Signals(signum).name}")


def autotune_cache_state(directory: Path) -> dict:
    """Retain selected configs without exporting compiled kernel binaries."""
    result = {}
    for path in sorted(directory.rglob("*.autotune.json")):
        entry = json.loads(path.read_text())
        config, _timing = min(entry["configs_timings"], key=lambda row: row[1])
        result[path.relative_to(directory).as_posix()] = {
            "key": entry["key"],
            "config": config,
            "sha256": hash_file(path),
        }
    return result


def main() -> int:
    """Validate inputs and run one acceptance stage, retaining failure evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("single", "four"))
    parser.add_argument(
        "--gpus",
        required=True,
        help="Reserved idle nvidia-smi indices or full UUIDs, comma-separated",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New output directory; never reuses old results"
    )
    parser.add_argument("--source-root", type=Path, default=ARTIFACTS / "split-pr-20261015")
    parser.add_argument("--weights", type=Path, default=ARTIFACTS / "models/Qwen3.5-0.8B")
    parser.add_argument("--data", type=Path, default=ARTIFACTS / "training-data/ks-item-sft.jsonl")
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="four only: run native-training and all Qwen arms; record shared jobs as omitted",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Seconds per job; timeout stops this run's container",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without creating a container or initializing CUDA",
    )
    args = parser.parse_args()
    args.output, args.source_root = args.output.resolve(), args.source_root.resolve()
    args.weights, args.data = args.weights.resolve(), args.data.resolve()
    selectors = [value.strip() for value in args.gpus.split(",")]
    if (
        len(selectors) != (1 if args.stage == "single" else 4)
        or not all(selectors)
        or args.timeout <= 0
    ):
        parser.error("single requires one GPU; four requires four GPUs; timeout must be positive")
    if args.output.exists():
        parser.error("Output directory already exists; choose a fresh path")
    sources = source_state(args.source_root)
    if any(
        args.output.is_relative_to(args.source_root / name)
        for name in ("base", "shared", "gate", "feature", "bridge")
    ):
        parser.error("Output must be outside source checkouts")
    if args.stage == "four" and (
        not (args.weights / "config.json").is_file() or not args.data.is_file()
    ):
        parser.error("four requires the local Qwen checkpoint and JSONL data")
    bridge = bridge_state(args.source_root) if args.stage == "four" else None
    gpus = gpu_inventory(selectors)
    jobs = make_jobs(args)
    omitted_jobs = []
    if args.training_only:
        if args.stage != "four":
            parser.error("--training-only requires stage four")
        omitted_jobs = [job["name"] for job in jobs if job["source"] == "shared"]
        jobs = [job for job in jobs if job["source"] != "shared"]
    container = "megatron-gdn-accept-" + uuid.uuid4().hex[:12]
    start = container_command(args, gpus, container)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "executed": False,
                    "sources": sources,
                    "bridge": bridge,
                    "gpus": gpus,
                    "omitted_jobs": omitted_jobs,
                    "container": shlex.join(start),
                    "jobs": [
                        {
                            "name": job["name"],
                            "command": shlex.join(job_command(args, job, container)),
                        }
                        for job in jobs
                    ],
                },
                indent=2,
            )
        )
        return 0
    require_idle(gpus)
    subprocess.run(["docker", "image", "inspect", IMAGE], check=True, stdout=subprocess.DEVNULL)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "stage": args.stage,
        "status": "running",
        "sources": sources,
        "gpus": gpus,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "container_command": start,
        "planned_jobs": [job["name"] for job in jobs],
        "omitted_jobs": omitted_jobs,
        "jobs": [],
        "performance_review": "pending" if args.stage == "single" else "not measured",
    }
    if bridge is not None:
        manifest["bridge"] = bridge
    payload = args.output / "scripts"
    payload.mkdir()
    for filename in RUNTIME_FILES:
        shutil.copy2(HERE / filename, payload / filename)
    manifest["script_sha256"] = {
        filename: hash_file(payload / filename) for filename in RUNTIME_FILES
    }
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop_run)
    try:
        manifest["container_id"] = subprocess.check_output(start, text=True).strip()
        probe = f"""
import importlib.metadata as metadata
import json
import torch

assert torch.cuda.device_count() == {len(gpus)}
print(json.dumps({{
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    'packages': {{name: metadata.version(name) for name in (
        'transformer-engine', 'flash-linear-attention', 'causal-conv1d', 'flash-attn', 'triton'
    )}},
}}))
"""
        manifest["environment_command"] = job_command(
            args, {"source": "feature", "command": ["-c", probe]}, container
        )
        with (
            (args.output / "environment.json").open("w") as stdout,
            (args.output / "environment.stderr.log").open("w") as stderr,
        ):
            subprocess.run(
                manifest["environment_command"],
                stdout=stdout,
                stderr=stderr,
                check=True,
                timeout=args.timeout,
            )
        if args.stage == "four":
            manifest["input_sha256"] = {
                str(path): hash_file(path)
                for path in [
                    args.data,
                    *sorted(args.weights.glob("*.json")),
                    *sorted(args.weights.glob("*.safetensors")),
                ]
            }
        for job in jobs:
            source_state(args.source_root)
            require_idle(gpus)
            directory = args.output / job["name"]
            directory.mkdir()
            record = {
                "name": job["name"],
                "source": job["source"],
                "source_commit": sources[job["source"]],
                "command": job_command(args, job, container),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            }
            if "expected_tests" in job:
                record.update(expected_tests=job["expected_tests"], allow_skips=job["allow_skips"])
            manifest["jobs"].append(record)
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Running {job['name']} (log: {directory / 'launcher.log'})", flush=True)
            with (directory / "launcher.log").open("w") as stream:
                result = subprocess.run(
                    record["command"], stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout
                )
            record.update(
                exit_code=result.returncode, finished_utc=datetime.now(timezone.utc).isoformat()
            )
            if job["name"].startswith("qwen-"):
                (directory / "autotune-cache.json").write_text(
                    json.dumps(autotune_cache_state(args.output / "cache/triton"), indent=2) + "\n"
                )
            source_state(args.source_root)
            require_idle(gpus)
            if result.returncode:
                raise RuntimeError(
                    f"{job['name']} exited {result.returncode}; inspect {directory / 'launcher.log'}"
                )
            if "expected_tests" in job:
                record["tests"] = verify_pytest(
                    directory / "reports", len(gpus), job["expected_tests"], job["allow_skips"]
                )
        if args.stage == "single":
            manifest["gate_performance"] = summarize_benchmarks(
                args.output, SOURCES, gpus[0]["uuid"]
            )
        else:
            manifest["qwen_comparison"] = verify_qwen(args.output, ranks=4)
        skipped = any(record.get("tests", {}).get("skipped", 0) for record in manifest["jobs"])
        manifest["status"] = "incomplete" if skipped else "automated_checks_passed"
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        if isinstance(error, subprocess.TimeoutExpired) and manifest["jobs"]:
            manifest["jobs"][-1]["exit_code"] = 124
        print(manifest["error"], file=sys.stderr)
    finally:
        cleanup = subprocess.run(
            ["docker", "rm", "--force", container], capture_output=True, text=True
        )
        manifest["cleanup"] = {
            "exit_code": cleanup.returncode,
            "output": cleanup.stdout + cleanup.stderr,
        }
        if cleanup.returncode:
            manifest["status"] = "failed"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Cache binaries are reusable intermediates, not acceptance evidence.
        evidence = sorted(
            path
            for path in args.output.rglob("*")
            if path.is_file() and "cache" not in path.relative_to(args.output).parts
        )
        (args.output / "SHA256SUMS").write_text(
            "".join(
                hash_file(path) + "  " + path.relative_to(args.output).as_posix() + "\n"
                for path in evidence
            )
        )
    print(f"{manifest['status']}: {args.output / 'manifest.json'}", flush=True)
    return 0 if manifest["status"] == "automated_checks_passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
