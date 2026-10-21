"""Publish only validated source identities, commands and raw acceptance records."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
destination = Path(
    "/home/songbinbin/gdn-offload-artifacts/review-20261021/evidence/offload-review-20261021"
)
verified = json.loads((root / "verification.json").read_text())
assert verified["status"] == "passed"
sources = json.loads((root / "sources.json").read_text())
destination.mkdir()
jobs = {row["name"] for row in verified["pytest"]}
jobs.update(row["name"] for row in verified["negative_controls"])
jobs.update(name for row in verified["qwen"] for name in row["arms"])
jobs.update(("performance-disabled", "performance-offloaded"))
jobs.update(f"four-qwen-f1-n0-{arm}" for arm in ("disabled", "repeat-disabled", "offloaded"))
for name in sorted(jobs):
    shutil.copytree(
        root / name,
        destination / name,
        ignore=shutil.ignore_patterns("vectors.rank*.pt", "__pycache__", "*.pyc"),
    )
files = [
    "sources.json",
    "verification.json",
    "negative-control-sources.json",
    "negative-controls.bundle",
    "performance-comparison.json",
    "four-qwen-comparison.json",
    "captured-qwen-comparison.json",
    "qwen-tp-sp-comparison.json",
    "qwen-pp-comparison.json",
    "environment-launch.json",
    "environment-launch-four.json",
    "environment-versions.json",
    "bridge-environment.json",
    "bridge-compatibility.patch",
    "signature-verification.json",
    "latest-main-merge-check.json",
    "gate-latest-main-merge.json",
    "main-final-remote.txt",
    "shared-own.patch",
    "feature-own.patch",
    "extensions-own.patch",
    "shared-own-files.txt",
    "mypy-comparison.json",
    "mypy-prerequisite.log",
    "gpu-inventory-four.csv",
    "feature-kernel-coverage.log",
    "extensions-kernel-coverage.log",
]
for name in ("shared", "feature", "extensions"):
    files.extend((f"lint-{name}.json", f"lint-{name}.log"))
for name in files:
    shutil.copy2(root / name, destination / name)
shutil.copytree(
    root / "scripts", destination / "scripts", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
)
table = "\n".join(
    f"| {row['name']} | `{row['revision'][:10]}` | {len(row['ranks'])} | {row['ranks'][0]['passed']} | 0 |"
    for row in verified["pytest"]
)
source_table = "\n".join(f"| {name} | `{row['revision']}` |" for name, row in sources.items())
readme = f"""# Activation-offload review validation, 2026-10-21

This bundle records the final tested source commits for PRs #8243, #7908 and #8308.
Commands, launcher exits, original stdout/stderr for every rank, and call-phase pytest
outcomes are retained. Counts below are per rank. Collected-only or skipped cases
are not counted as passed. Source branches are published on the author's fork.

| Source | Commit |
| --- | --- |
{source_table}

`base` integrates the original #7399 head `73ac08c254a0ff8997a0ff559799a4d7f073248b`
with upstream main `6c956376e3ef22528413ff35412ad8b938272986`. Original prerequisite
commits and authorship are preserved. Merge order is #7399, #8243, #7908, #8308.
The `*-own.patch` files isolate each PR from its prerequisites. #8244 remains
unchanged at `d8115c4c3bf52a868b242d032b5519d7c3331a52`.

During testing, main advanced to `e4eb89b3bb3bb55c31606ee9d672038e17caa2a0`
(SchedulerConfig migration, #8211). Read-only merge checks pass for all three
branches; their merged `megatron/core` trees equal the tested trees. This is a
merge analysis, not another GPU execution of a merged commit. The gate kernel
in the three offload branches is byte-identical to upstream main.

## Executed checks

| Job | Source | Ranks | Passed per rank | Skipped |
| --- | --- | ---: | ---: | ---: |
{table}

The shared 26-case job includes six CPU routing/type-guard checks and 20 real GPU
checks, including seven existing GPT/MoE output/gradient/memory cases. The four-rank
shared job includes four CPU routing checks and three GPU cases. CPU checks do not
qualify transfers. The 94-case feature job combines 73 scope/configuration cases
with 21 CPU tool/input-boundary cases. Native training uses complete reduced
decoders, loss, DDP and BF16 Adam. TE graph tests execute capture and replay.

The EP2 probe reuses the repository's existing layer-overlap compatibility test,
changing only EP size from four to two; original model dimensions and comparisons
are retained. It executes the native model-chunk schedule, not fake tensors/events.

## Why the remaining lifecycle fixes are needed

The new regression executes native GPT evaluation, then training with sequence
lengths 8, 128 and 64 at fractions zero and one. Outputs and all parameter gradients
match the disabled baseline. The copy observer delegates to the real offload method.
The prerequisite baseline fails after evaluation. Two additional controls restore
old fraction-zero or chunk-progress behavior while keeping the other fixes. Each
fails its call-phase regression on both ranks, for the recorded
reason: prematurely completed warmup, unexpected real copies, or `Chunk mismatch`.
These are expected failures of control code, not passing tests of the PR source.

`negative-control-sources.json` records the control commits and proves their test
file matches the final regression exactly. To fetch those signed control commits,
first fetch the published feature branch (which contains `base`), then run
`git fetch /path/to/negative-controls.bundle 'refs/heads/review/*:refs/remotes/controls/*'`.
The bundle requires the published `base` commit and contains no model/data files.

The extra stream-wide H2D-to-D2H wait and its artificial reuse test were removed.
Pool instrumentation records CUDA timing events before actual D2H reuse and after
actual H2D reads; it adds no stream waits, sleeps, pool-policy changes or warmup
overrides. Native GPT/MoE, EP2 overlap and PP/VPP observations are in the per-rank
`pool-*.jsonl` reports and summarized in `verification.json`. All observed reuses
follow H2D completion. This bounded observation does not prove every possible
schedule race-free. Inline-reload synchronization and LIFO ownership come from #7399.

## Complete Qwen training

All Qwen checks load the local Qwen3.5-0.8B checkpoint and train its complete 24-block
text model, including embeddings, MLPs, vocabulary loss and native optimizer state.
Vision and MTP are excluded. Two-card checks cover TP2+SP without norm recomputation
and PP2 with it enabled. Four-card checks use TP2+SP+PP2 and all four fusion/norm
settings. Each has three fresh torchrun runs: disabled, repeated-disabled and
offloaded; every rank runs three optimizer steps, sequence 128 and four microbatches.
The 1024-element threshold exercises real transfers. On four ranks, PP2 ranks 0/1 have no local
loss records; ranks 2/3 have four microbatch losses per step.

The audit recomputes exact equality from every rank's recorded loss, gradient
norm, complete gradients, weights and Adam/master-state fingerprints. The
captured fused comparison procedure also inspects actual vectors if fingerprints
differ, using rtol=1e-5, atol=1e-8, the same tolerance as the reduced-decoder tests.
All fingerprints in the captured runs match exactly, so no tolerance allowance is
needed for them. Bulk captured vectors, model files and data are not published.

The initial `four-qwen-f1-n0-*` run fails strict fingerprint equality on rank 2:
one step-zero `A_log` gradient and the associated Adam moments differ. The repeated
disabled baseline also differs; its loss, weights and other fingerprints match.
This failed strict comparison is retained in `four-qwen-comparison.json` and the
original job directories, and is not counted as an exact pass. The later
`four-qwen-captured-*` runs add actual small-vector capture at the tool's existing
fingerprint boundary, with unchanged native arithmetic. Their per-rank comparison
records are in `comparison.rank*.json`, summarized by
`captured-qwen-comparison.json`; the audit independently checks exact state equality
from the original result files. The initial uncaptured difference has no retained
raw tensor values and is not assigned a numerical magnitude. Native fused gate
backward uses atomic FP32 reductions; these checks do not establish global
bitwise determinism.

## Memory and runtime

`performance-comparison.json` and both performance job directories contain one
current-head, fresh-process disabled/offloaded pair on physical GPU 1. The complete
Qwen text model uses sequence 4096, four microbatches, unfused pre-GDR, norm
recomputation off, fraction 1 and the default 1,048,576-element tensor threshold.
Three warmup steps precede ten timed steps. Timing includes zeroing, forward,
backward, finalization, eager optimizer and CUDA completion; input preparation and
state snapshots are excluded. Correctness fingerprints are collected separately.
Logged losses and printed gradient norms match across the performance pair; printed
norms have limited precision. GPU telemetry and all raw step samples are retained.

This single pair checks the current trade-off; it is not a repeated statistical
benchmark or a cross-version speedup claim. Prior repeated performance, checkpoint
and stability results remain associated with their original commits in the earlier
bundles. They do not count as executions of these rebased heads.

## Reproduction and audit

Environment: RTX A6000, PyTorch 2.11.0+cu130, TE 2.20.2, FLA 0.5.1, Triton 3.6.0,
causal-conv1d 1.6.1 and flash-attn 2.8.3. The image digest, device UUIDs, mounts and
exact commands are recorded. Bridge is pinned separately with its import-only
compatibility overlay. Model weights and private JSONL data are not included.

Check out the exact commit from the fork, use the listed environment, and run the
commands in each job's `manifest.json`, adapting local model/data/output paths.
`rank_env.py` gives each rank its own Triton cache; preserve that cache across matched
fresh-process arms. Frozen launcher versions are retained for the hashes in older
job manifests. All source checkouts were clean at launch.
The original `shared-final-targeted/launcher.log` contains NUL bytes from concurrent
`--tee` output tailing; its separate per-rank logs and outcome records remain intact.
Later launchers use redirects without tee. The audit checks per-rank records,
launcher exit codes and recorded harness hashes.

The public records can be audited without a GPU:

```bash
sha256sum -c SHA256SUMS
python3 scripts/verify_acceptance.py
```

Black/isort/Pylint/Ruff and the project formatter complete successfully. The formatter
treats mypy as nonblocking: its diagnostics are retained, including 34 shared-file
diagnostics identical to the prerequisite after line-number normalization. No claim
of a clean global mypy run is made. Kernel coverage and commit-signature checks are
included. Upstream GPU CI has not been run; #7399 is still an upstream prerequisite.
"""
(destination / "README.md").write_text(readme)
lines = []
for path in sorted(destination.rglob("*")):
    if path.is_file():
        lines.append(
            hashlib.sha256(path.read_bytes()).hexdigest()
            + "  "
            + str(path.relative_to(destination))
        )
(destination / "SHA256SUMS").write_text("\n".join(lines) + "\n")
print(destination)
