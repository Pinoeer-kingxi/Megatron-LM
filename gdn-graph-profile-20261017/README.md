# GDN graph transfer diagnosis and live-gate optimization

Executed fix: [`367d014c7bf74ba3553c9db58da47fa5bcba906e`](https://github.com/Pinoeer-kingxi/Megatron-LM/commit/367d014c7bf74ba3553c9db58da47fa5bcba906e).
Previous source: `d18040ba5a41e69ada77406d74f68295ccb8090b`.
The previous runtime matches the `f19ebc5b1d` graph-training evidence;
its published performance figures retain that original revision.

This bundle contains raw Nsight reports/SQLite exports, exact launch commands,
rank logs, complete-state fingerprints and an offline verifier. The code change
uses the existing tensor opt-out to keep live GDN2 sigmoid outputs resident:
`w`, and `b` when no head expansion is needed. Frozen gates remain eligible.
Evaluation bypasses the new marking. No numerical kernel or transfer ordering changes.

## Diagnosis

Four fresh-process controls profile three synchronized replay training steps,
after three native optimizer steps, real whole-layer TE graph capture and three
replay warmup steps. Optimizer updates remain eager outside graph capture. The same selected
RTX A6000 is used serially. No concurrent benchmark arms run on the selected GPU.
Profiler timings below are diagnostic, not the performance qualification.

| Profile | Step wall ms | Kernel busy union ms | Copy without any kernel ms | Kernel/copy overlap ms |
| --- | ---: | ---: | ---: | ---: |
| gdn2-profile-disabled | 231.96 | 229.27 | 1.12 | 0.42 |
| gdn2-profile-enabled | 453.01 | 223.05 | 226.70 | 40.97 |
| qwen-profile-disabled | 854.99 | 842.84 | 2.99 | 1.38 |
| qwen-profile-enabled | 1032.80 | 829.71 | 192.54 | 129.97 |
| gdn2-final-profile | 425.99 | 223.50 | 200.09 | 41.07 |

Before the fix, GDN2 transfers 1600 MiB in each direction per optimizer step;
Qwen transfers 1904 MiB in each direction, plus its existing 32-byte H2D scalar
copies. Kernel name/count multisets agree exactly between each original on/off
control. The principal added intervals contain DMA without overlapping kernels.
The interval unions describe occupancy; they are not an additive sum of kernel,
memcpy and CPU synchronization API durations or a counterfactual critical-path model.

The saved-tensor reports use the existing profiling tool on one real GDN2 layer
at the complete-model dimensions. They identify the write gate's 8 MiB storage
saved both by gate backward outside recurrence and by recurrence. With equal
key/value head counts, the erase gate also shares an 8 MiB storage. Copying these
references does not release the original GPU storage. The old-source red control
fails precisely on the live-write-gate copy assertion; its expected exit code is 1.

The fixed GDN2 trace transfers 1440 MiB in each direction, 10% fewer bytes, with
the same kernel name/count multiset. Both original and fixed graph profiles
retain their per-step input, loss and gradient-norm sequences.

## Qualified before/after performance

Both arms enable offloading: fraction 1, 1,048,576-element threshold, norm
recomputation off and unfused pre-GDR. The model has eight complete blocks
`GGG*GGG*`, hidden size 1024, FFN 3584, vocabulary 4096, linear key/value heads
8/16 with head dimension 128, embeddings, MLPs, vocabulary loss and native BF16
Adam. Weights use seed 31; the synthetic token stream uses a private generator
with seed 19. Sequence length is 2048, microbatch size 1, accumulation 4.

Three fresh-process pairs alternate previous/fixed, fixed/previous,
previous/fixed. Each arm trains three native steps, captures graphs, trains three
replay warmup steps and times ten complete optimizer steps. Timing includes
zeroing, native schedule forward/backward/finalization, Adam and CUDA completion.
Inputs, capture, snapshots and reporting are excluded. Each pair's full 16-step
input hashes, losses and gradient norms agree exactly.

| Metric | Previous | Fixed |
| --- | ---: | ---: |
| Median of process-median step times | 451.71 ms | 424.91 ms |
| Median peak reserved | 4944 MiB | 4922 MiB |
| Selected bytes in PP1 policy summary | 400 MiB | 360 MiB |
| Retained pinned buffers | 54 | 48 |

Paired median step-time change: -5.93%
(range -5.95% to -5.89%).
Median additional reserved reduction: 22 MiB.
Raw samples and each pair's individual allocation peaks are in `verified-results.json`
and `gdn2-paired-*/result.json`. This comparison measures the gate-selection fix;
it does not establish a speedup from enabling offloading or enabling CUDA graphs.
The remaining transfer cost is substantial.

## Current-source correctness

- Each of two ranks passes 20 scope cases, 18 native training cases and 9 real
  TE graph cases, with zero skips. Native coverage includes DP2, TP2+SP and PP2,
  norm recomputation off/on and fractions 0/0.5/1. Graph coverage includes DP2,
  TP2+SP, PP2, whole-layer/attention capture, norm recomputation, delayed offload
  and fraction zero. Test reports contain the complete case IDs.
- New real-GPU regressions cover equal/expanded heads, trainable/frozen gates,
  copy selection, all output/input/parameter gradients, and evaluation with
  gradients and an initially absent offload manager.
- Separate complete-model on/off checks cover key/value heads 8/16 and 16/16.
  All six steps, including three graph replays, have identical losses, gradient
  norms, full gradients, weights and complete optimizer/master-state fingerprints.
- Pinned-pool ownership is zero before native schedule resets. Allocation counters
  remain unchanged throughout actual graph replay.

The four-card composition, pretrained GDN2, upstream GPU CI and a continuous
200-step graph run are not qualified here. Earlier Qwen and GDN2 performance
results retain their source revisions; they are not relabeled as measurements
of this change. CUDA graph private pools reserve addresses across replays, so
allocated and reserved statistics must be interpreted separately.

## Configuration screening

`configuration-screening.json` contains single fresh-process probes, not
three-pair performance qualifications. Increasing copy connections to 8 did not
improve runtime. Fraction 0.5 reduced transfer time but lost most of Qwen's
reserved savings and exceeded the disabled GDN2 reserved peak. Raising the size
threshold traded memory savings for less traffic. No default or synchronization
change was made based on these probes.

## Reproduction

Use the source commits and package versions in `final-source-identity.json` and
`environment.json` (PyTorch 2.11.0+cu130, TE 2.20.2, FLA 0.5.1, Triton 3.6.0).
`final-manifest.json` records actual argv, GPU UUIDs, timestamps, source snapshots,
producer hashes and exit codes. The benchmark also imports Megatron Bridge at
`c860f8a5bc5fddd78690f32baa0b8696774308b4`; the included
`environment/bridge-compatibility.patch` applies the recorded import-only overlay.
The host launcher imports the already validated local container helper;
it is retained for provenance, not as a portable launcher.

For the synthetic GDN2 workload, run the included benchmark from the selected
Megatron checkout, with its directory first on `PYTHONPATH`:

```bash
export NVTE_CPU_OFFLOAD_V1=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
python -m torch.distributed.run --standalone --nproc-per-node=1 \
  /path/to/bundle/scripts/benchmark_graph_training.py --model gdn2 \
  --weights /unused --data /unused --offload --output result.json
```

The required weights/data arguments are Qwen-only inputs and are not read in
synthetic GDN2 mode. Omit `--offload` for the disabled control. Add `--check-state
--warmup 1 --iterations 2` for the six-step full-state check, or
`--gdn2-key-heads 16` for equal linear heads. Reuse the same tuning choices
between comparisons and give distributed ranks separate Triton caches using
`scripts/rank_env.py`. Do not run benchmark arms concurrently.

For a focused trace, prepend the Nsight options recorded in the profile manifest
and add `--profile --iterations 3`. `scripts/benchmark_graph_training.py` version
`10b09f533d4bbaa8090b300f3c7117f9759ef27268d96d3bb4c183ac014e5a39`
produced final jobs. The original diagnostic/screening version is preserved under
`scripts/versions/13b114006faad7f347f0bab3bca9280b0ec04f2edcc0a666b874c93fcd328c40/`.
Use that archived producer path when replaying the earlier commands.

Verify the downloaded records without CUDA dependencies:

```bash
sha256sum -c SHA256SUMS
python scripts/verify_final.py
```

The verifier checks recorded artifacts only; it does not execute GPU tests.
Local source archives, model/data contents, caches and superseded attempts are
excluded from this public bundle.
