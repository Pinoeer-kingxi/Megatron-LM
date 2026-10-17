# Current-source A6000 validation

Feature source: **87c22d1abae13abf375e8ed4ddfe62301cde87f4** (#7908).
Shared-manager source: **af9569270f476dc201484296315a9c30dcaa3feb** (#8243).
Independent gate source: **9d54b8729c9277f1e032110c5e1b149e9654bcc5** (#8244).
Upstream base: **9131593673d037d360d252904f9198d36bc0c246**.

## Current qualification

| Job | Executed source | Result |
| --- | --- | --- |
| Shared manager, four ranks | 87c22d1; component files identical to af956927 | 16 passed per rank, zero skips |
| Targeted native fused regression, four ranks | 87c22d1 | 1 passed per rank, three full-state comparisons each |
| Complete reduced decoder, four ranks | 87c22d1 | 96 passed per rank, zero skips |
| Full 24-layer Qwen3.5-0.8B text model, TP2+SP+PP2 | 87c22d1 | All four fusion/norm settings pass disabled/repeated-disabled/offload comparisons |
| GDN scope/configuration regressions, one rank | 87c22d1 | 73 passed, zero skips |
| Independent gate replay/numerical regressions, one rank | 9d54b872 | 16 passed, zero skips |
| Complete-Qwen optimizer-step measurements, one rank | 87c22d1 | 12 completed runs, 120 timed samples |

The native matrix covers DP, TP, TP+SP, CP, PP, PP+VPP, TP+SP+CP and TP+SP+PP,
both fusion and norm-recomputation settings, and fractions 0/0.5/1. The full-Qwen
checks use all text blocks, local pretrained weights/data, sequence 128, four
microbatches and three optimizer steps per arm. They record 144 per-rank optimizer
states and compare every loss, accumulated gradient, updated weight and complete
Adam/master state on every rank. Enabled arms select transfers and check zero
pinned-buffer ownership before the native schedule resets.

Unfused states match exactly. Fused comparisons use actual vectors with the
predeclared rtol=1e-5, atol=1e-8 bound, while every other tensor must have identical
full-byte fingerprints. All fused four-GPU comparisons actually have zero
differences here. Native atomic reductions still have no global bitwise guarantee.
Each configuration preserves isolated per-rank tuning caches across matched arms.
Counts are per rank; they are not multiplied into independent test cases.

## Current-source performance

One RTX A6000, physical GPU 1, UUID GPU-5e2e1f94-e88e-48e3-247d-328b52fc68db.
Sequence 4096, microbatch size 1, four microbatches, fraction 1, default 1,048,576-
element threshold, norm recomputation off. Three fresh-process pairs per fusion
setting; three warmup and ten measured complete optimizer steps per process.
State snapshots are disabled in timed runs; input preparation is excluded.

| Pre-GDR | Allocated peak, disabled → offload (MiB) | Allocated saved | Reserved peak, disabled → offload (MiB) | Paired step-time change, median [range] |
| --- | --- | --- | --- | --- |
| Unfused | 31599.17 → 30647.17 | 952.00 MiB (3.01%) | 34356 → 33322 | +1.19% [+0.87%, +1.54%] |
| Native fused | 31302.17 → 30350.17 | 952.00 MiB (3.04%) | 34826 → 33880 | +1.89% [+1.35%, +2.57%] |

Both allocated and reserved peaks decrease, with a measured runtime cost. Timing
statistics use the median of three paired changes between per-run median step
times; ranges show those three observations. Memory uses median per-run maximum
peaks. All logged microbatch losses match across six runs of each fusion setting;
these timed runs do not record full gradient/Adam state at sequence 4096.
The measurement GPU was exclusive; other GPUs on the host had unrelated workloads.
Raw occupancy, temperature, clocks and power are retained. No controlled cross-
version, cross-device or fusion-to-fusion attribution is made.

[All raw measurements and commands](performance/README.md) retain exact source,
input and script hashes. Historical measurements remain at their original sources:

| Pre-GDR | Runtime source | Allocated saved | Step-time change, median [range] |
| --- | --- | --- | --- |
| Unfused | e9975a744efad181f9ab586ce33f58e7a9649ac8 | 952 MiB (3.01%) | +8.52% [+7.60%, +8.57%] |
| Fixed fused | e9975a744efad181f9ab586ce33f58e7a9649ac8 | 952 MiB (3.04%) | +11.48% [+9.38%, +11.84%] |
| Unfused | 4e18562c6bcf894795d23378a8f1699c79e4b797 | 952 MiB (3.01%) | +1.58% [+1.56%, +2.01%] |
| Fixed fused | 2f1ee4f2e3dd6d909d08981338f5879632131139 | 952 MiB (3.04%) | +5.80% [+4.95%, +8.10%] |

[The e9975a744 evidence](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/63f37036733080f769b2105c021062229b324b70/qwen-performance-20261016)
and [earlier raw measurements](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/cf8df32ec2424bcc6c42a5486d2fe68213c74294/split-20261015/historical)
retain their original code, devices and conditions. They are historical offload
comparisons, not measurements of this source or controlled kernel comparisons.

## Files and verification

- [four-gpu](four-gpu/README.md): current-source complete four-rank matrix and full
  Qwen TP2+SP+PP2 correctness, all commands/rank logs/states, numerical reports,
  environment/input hashes and frozen-source checks.
- [scope](scope/README.md): current-source 73-case GDN scope/configuration job.
- [performance](performance/README.md): current-source single-GPU measurements.
- [two-gpu](two-gpu/README.md): earlier fbe4bb03 two-rank controls, original fused
  fingerprint failure, actual-vector retests and independent 9d54b872 gate replay.
  Its interrupted performance attempt contributes no qualified measurements.
- [four-gpu-initial](four-gpu-initial/README.md): retained fbe4bb03 four-rank exact-
  assertion failure. Six symlinks were dereferenced in this initial copy; all 3532
  regular files and dereferenced contents match the archive. The strict layout
  failure and complete materialization report are retained. The successful
  87c22d1 runs use archive extraction and pass strict file/symlink checks.
- [Chinese review and validation report](审查报告.txt).

CPU-only rechecking of the published evidence, without GPU, model weights or
training data:

```bash
python3 -S verify_bundle.py
```

This verifies checksums, source/version relationships, actual rank outcomes,
training arguments and recorded comparisons, and recomputes the timing summary.
It does not rerun GPU tests. SHA256SUMS covers every exported file except itself.
The stage manifests retain the exact original GPU launcher commands and exits.
The scripts use the recorded machine's paths, locally available container digest
and dependency overlays; reconstruct those for a fresh GPU execution.

Full checkpoints, private training data, source tarballs, compiler caches and
local .pt vector captures are excluded. Input hashes identify the tested assets.
Full-state fingerprints are retained. Where vectors differ, numerical reports
retain their actual values/raw bytes; current four-GPU comparisons have none.
Source per-file checks are included and can be compared with the public source at
the pinned SHA. The shared helper's historical CLI is not used by these runs.

## Code change and boundaries

This follow-up removes two redundant comments and corrects the native training
test's exact fused comparison to a fixed numerical bound; unfused comparison
stays exact. Production code and training tools are unchanged by the test fix.
The pre-GDR kernel still matches the upstream base byte-for-byte. #8244 is
independent; #7908 depends on #8243 and does not add a deterministic gate kernel.
Original failed exact probes are not relabeled as passing commands. Their
original actual tensors were not captured, so no numerical magnitude is claimed
for those first failures.

Pretrained Qwen CP, checkpoint restart, long-run convergence and composition with
#7852 remain unqualified. Local validation does not replace upstream GPU CI.
After #8243 merges, #7908 still needs rebasing and affected-path validation on the
merged base, and GPU CI approval from a maintainer. No comments or review requests
were posted as part of this validation.
