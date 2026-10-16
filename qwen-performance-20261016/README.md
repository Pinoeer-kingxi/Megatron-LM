# Current-head Qwen activation-offload measurements

Measured source: `e9975a744efad181f9ab586ce33f58e7a9649ac8` for PR #7908,
including the shared-manager and gate-reduction prerequisites. Every training
process imports an independent `git archive` snapshot of that commit. The source
branch was not modified. `manifest.json` records the archive checksum, input
checksums, exact container/job arguments, GPU identity, exit codes and cleanup.

One RTX A6000 (48 GiB), driver 580.178.04; PyTorch 2.11.0+cu130, CUDA 13.0,
Transformer Engine 2.20.2, FLA 0.5.1, causal-conv1d 1.6.1, flash-attn 2.8.3,
Triton 3.6.0. `environment.json` identifies the imported Core package and settings.
`TORCH_COMPILE_DISABLE` is unset; the existing `jit_fuser` configuration applies.

The workload trains all 24 Qwen3.5-0.8B text decoder blocks from the same local
Hugging Face checkpoint, with BF16/FLA, embeddings, mixed attention, MLPs,
vocabulary loss and native BF16 Adam. Vision and MTP are excluded. TP/CP/PP are
one, sequence length is 4096, microbatch size is one, and each optimizer step
accumulates four microbatches. Norm recomputation is off. The enabled arm selects
`gdn_core_attn`, fraction 1, and the default 1,048,576-element threshold.

Each fusion setting has three paired repeats. Every arm starts a fresh process,
warms up three optimizer steps and records ten further steps. Pair order is
disabled/enabled, enabled/disabled, disabled/enabled. Each fusion setting owns a
separate persistent rank-0 Triton/Inductor cache reused across its matched arms.
The cache is initially empty. No external compute process was observed on the
measured GPU; each job retains its occupancy, temperature, power, clocks and
utilization samples. GPU runs are sequential. CPU lint/test checks run afterward.

| Pre-GDR path | Disabled / enabled peak allocated MiB | Saved allocated memory | Paired step-time change, median [range] |
| --- | --- | --- | --- |
| Unfused | 31599.17 / 30647.17 | 952.00 MiB (3.01%) | +8.52% [+7.60%, +8.57%] |
| Fixed fused | 31302.17 / 30350.17 | 952.00 MiB (3.04%) | +11.48% [+9.38%, +11.84%] |

Runtime statistics are the median of the three paired changes between each run's
median step times. Memory statistics use the median of the three per-run maximum
allocated peaks. `raw-samples.csv` contains all 120 measured steps;
`summary.json` retains each repeat, timings, allocated/reserved peaks and the
statistics definition. Timed steps include gradient-buffer clearing, native
forward/backward/finalization and optimizer updates, with CUDA synchronization at
the timing boundary. Input preparation and checkpoint import are outside timing.
No correctness snapshots are taken in timed runs.

All 13 logged step-loss records match exactly across the six processes within
each fusion setting. Printed gradient norms also match, but their tensor rendering
is rounded; this is not a full-precision norm comparison. Enabled arms select
4,169,138,176 transfer bytes per optimizer step; disabled arms select zero. Policy
transfer bytes do not measure net memory savings or the CPU pinned-memory peak.
No OOM or training-process failure occurred. The GPU container exited and the card
was released before the CPU checks began.

These measurements demonstrate an approximately 3% allocated-memory reduction
with an 8.5%–11.5% median runtime cost in this workload. They do not establish a
speedup or a performance budget. Each row compares offload on/off at the same
current revision; the rows are not a controlled comparison of fusion variants.
They also do not isolate the shared-manager synchronization change from other
offload work. Different devices, sequence lengths, fractions or norm settings
require their own measurements.

The earlier +1.58% unfused and +5.80% fixed-fused runtime results retain their
original sources, `4e18562c6bcf894795d23378a8f1699c79e4b797` and
`2f1ee4f2e3dd6d909d08981338f5879632131139`, in the
[historical bundle](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/cf8df32ec2424bcc6c42a5486d2fe68213c74294/split-20261015/historical).
They are not current-head evidence or a controlled old/new comparison.

Full gradient/weight/Adam fingerprint checks are in the separate
[correctness bundle](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/8c5cc72379de4e9ef8d80856973dd64f23cc0dac/gpu-20261016),
executed at `8a2d19894288a8eb890dfdf20c9d54683035de80`. Core, tools and tests at
that revision are byte-identical to this measured commit; only reproduction
documentation differs. The performance runs here do not repeat those complete
state checks. They do not qualify long-run convergence, restart, pretrained Qwen
CP or composition with #7852. Local results do not replace upstream GPU CI.

To reproduce the individual training arms in an equivalent environment, check out
the measured commit, prepare the pinned Bridge revision/compatibility overlay
in `manifest.json`, and use the supplied `scripts/rank_env.py` before importing
the training target:

```bash
export NVTE_CPU_OFFLOAD_V1=1
# Set IDLE_GPU_UUID to a reserved A6000; keep the same device within a pair.
export CUDA_VISIBLE_DEVICES="${IDLE_GPU_UUID:?Select an idle A6000 UUID first}"
# Create once per fusion setting and preserve between matched processes.
export TRITON_CACHE_DIR=/tmp/qwen-offload-f0-cache
python -m torch.distributed.run --standalone --nproc-per-node=1 \
  /path/to/evidence/scripts/rank_env.py tools/ssm/qwen_gdn_offload_training.py \
  --weights /path/to/Qwen3.5-0.8B --data /path/to/messages.jsonl \
  --seq-length 4096 --microbatches 4 --warmup 3 --iterations 10 \
  --output /tmp/qwen-disabled.json
```

Run the matched enabled process with `--fraction 1` and another output path.
Repeat three pairs, alternating order as above. Repeat with `--fused-pre-gdr`
on both arms and a separate persistent cache. Preserve checkpoint, data,
dependencies and rank mapping. `manifest.json` contains every exact command used
here, including the local paths; the orchestration helpers use this host's
validated container layout and are not general deployment tooling.

Verify the retained evidence without CUDA:

```bash
sha256sum --check SHA256SUMS
python -S scripts/summarize_performance.py
```

The summarizer validates all run exits, sample counts, settings, observed GPU
ownership, logged losses and persistent tuning choices before writing the CSV and
summary. Original logs are not edited. The first setup-only hash scan attempted
to hash the checkpoint's `.cache` directory; its failure is retained in
`preflight-input-directory.log`. No training ran in that attempt.

`cpu-checks/checks.json` and its logs record current-source Black, isort, Ruff,
Pylint and CPU-only configuration/input/snapshot/manifest checks. The first CPU
formatting attempt failed because the long artifact TMPDIR exceeded AF_UNIX's
socket-path limit; `cpu-checks-initial` retains that failure. The retry uses the
container's short `/tmp` path without changing source files. Documentation's
strict Sphinx and four-worker cache-launcher checks at the same source commit
remain in the separate
[documentation bundle](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/bfddfd950bb226648202e5ea40c2642b91a7f805/qwen-doc-cache-20261016).
