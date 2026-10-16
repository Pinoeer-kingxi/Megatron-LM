"""Compare the old and fixed BF16 gate backward on one otherwise idle A6000.

Run in the validated environment with CUDA_VISIBLE_DEVICES set to one GPU UUID.
Compilation and numerical checks precede timing. This measures the gate wrapper
with the preallocated projection-gradient buffer used by its production caller.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as functional

WARMUP = 20
ITERATIONS = 100
REPEATS = 3


def load_source(path, name):
    """Import a complete kernel module and record its committed source identity."""
    path = path.resolve()
    status = subprocess.check_output(
        ["git", "-C", str(path.parent), "status", "--porcelain", "--", path.name], text=True
    )
    if status:
        raise ValueError(f"Kernel source has uncommitted changes: {path}")
    revision = subprocess.check_output(
        ["git", "-C", str(path.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, {"revision": revision, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def measure_case(modules, batch, length, heads):
    """Measure one shape per fresh process so baseline autotuning cannot reuse another batch."""
    torch.manual_seed(1234)
    key_heads, dim = 4, 128
    offset = 2 * key_heads * dim + 2 * heads * dim
    projection = torch.randn(length, batch, offset + 2 * heads, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(heads, device="cuda", dtype=torch.bfloat16) * 0.2
    bias = torch.randn_like(a) * 0.2
    dg = torch.randn(batch, length, heads, device="cuda")
    dbeta = torch.randn_like(dg)
    output = torch.empty_like(projection)

    def backward(label):
        return modules[label]._triton_g_beta_backward(
            projection,
            a,
            bias,
            dg,
            dbeta,
            num_value_heads=heads,
            key_head_dim=dim,
            value_head_dim=dim,
            num_key_heads=key_heads,
            d_qkvzba_out=output,
        )

    p = projection.float().detach().requires_grad_()
    pa = a.float().detach().requires_grad_()
    pbias = bias.float().detach().requires_grad_()
    beta = p[..., offset : offset + heads].permute(1, 0, 2).sigmoid()
    alpha = p[..., offset + heads :].permute(1, 0, 2)
    gate = -pa.exp() * functional.softplus(alpha + pbias)
    reference = torch.autograd.grad((gate, beta), (p, pa, pbias), (dg, dbeta))
    for label in modules:
        actual = backward(label)
        torch.testing.assert_close(actual[0][..., offset:], reference[0][..., offset:].bfloat16())
        for index in (1, 2):
            torch.testing.assert_close(actual[index], reference[index], atol=1e-4, rtol=2e-5)
        del actual
        for _ in range(WARMUP):
            backward(label)
    del reference, p, pa, pbias, alpha, beta, gate
    torch.cuda.synchronize()
    samples = {label: [] for label in modules}
    for repeat in range(REPEATS):
        for label in (("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")):
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(ITERATIONS):
                backward(label)
            torch.cuda.synchronize()
            samples[label].append(
                {
                    "repeat": repeat,
                    "wall_ms_per_call": (time.perf_counter() - start) * 1000 / ITERATIONS,
                    "extra_peak_allocated_bytes": torch.cuda.max_memory_allocated() - allocated,
                }
            )
    medians = {
        label: statistics.median(item["wall_ms_per_call"] for item in records)
        for label, records in samples.items()
    }
    changes = [
        (fixed["wall_ms_per_call"] / original["wall_ms_per_call"] - 1) * 100
        for original, fixed in zip(samples["baseline"], samples["candidate"])
    ]
    return {
        "batch": batch,
        "sequence": length,
        "value_heads": heads,
        "head_dim": dim,
        "dtype": "bfloat16",
        "numerical_reference": "passed",
        "samples": samples,
        "median_wall_ms": medians,
        "paired_runtime_changes_percent": changes,
        "median_paired_runtime_change_percent": statistics.median(changes),
    }


def main():
    """Check both implementations numerically and report paired warmed measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 4), required=True)
    parser.add_argument("--seq-length", type=int, choices=(2048, 4096), required=True)
    parser.add_argument("--value-heads", type=int, choices=(16, 32), required=True)
    args = parser.parse_args()
    if any(not path.is_file() or path.suffix != ".py" for path in (args.baseline, args.candidate)):
        parser.error("Provide the original and fixed fused_pre_gated_delta_rule.py files")

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("Expose exactly one otherwise idle GPU")
    modules, sources = {}, {}
    for label, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        modules[label], sources[label] = load_source(path, f"gate_benchmark_{label}")
    case = measure_case(modules, args.batch_size, args.seq_length, args.value_heads)
    report = {
        "sources": sources,
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch": torch.__version__,
        "offloading": False,
        "warmup_calls": WARMUP,
        "calls_per_sample": ITERATIONS,
        "repeats": REPEATS,
        "timing": "synchronized wall time for the full gate wrapper with preallocated projection output",
        "clock_policy": "shared host; unlocked clocks; pair order alternates",
        "case": case,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
