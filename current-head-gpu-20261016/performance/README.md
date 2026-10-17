# Complete-Qwen single-A6000 performance

Executed source: `87c22d1abae13abf375e8ed4ddfe62301cde87f4`.
Measurement GPU: physical A6000 1, UUID `GPU-5e2e1f94-e88e-48e3-247d-328b52fc68db`, driver 580.178.04.

The complete 24-layer pretrained Qwen3.5-0.8B text model trains in BF16 with FLA,
including embeddings, mixed GDN/GQA blocks, MLPs, vocabulary loss, gradient
accumulation, finalization and native Adam/master weights. Vision and MTP are
excluded. Model/data hashes match the four-rank correctness job. Attention backend
policy is the provider's auto selection, as recorded in the committed tool.

Each fusion setting has three fresh-process offload on/off pairs, with order
alternating between repeats. Sequence 4096, microbatch size 1, four microbatches,
fraction 1, default minimum tensor size 1,048,576 elements, norm recomputation off.
Each process warms three optimizer steps and measures ten. There are 12 completed
runs and 120 timed samples (156 total optimizer steps). Timed runs disable state
snapshots; input preparation is excluded. Timing synchronizes CUDA around the
complete optimizer step, including gradient-buffer clearing, forward/backward,
gradient finalization and optimizer update.

| Pre-GDR | Allocated peak, disabled → offload (MiB) | Allocated saved | Reserved peak, disabled → offload (MiB) | Paired step-time change, median [range] |
| --- | --- | --- | --- | --- |
| Unfused | 31599.17 → 30647.17 | 952.00 MiB (3.01%) | 34356 → 33322 | +1.19% [+0.87%, +1.54%] |
| Native fused | 31302.17 → 30350.17 | 952.00 MiB (3.04%) | 34826 → 33880 | +1.89% [+1.35%, +2.57%] |

Allocated and reserved peaks both decrease, with a measured runtime cost.
Peaks are medians of each run's maximum; time changes are the median of three
paired changes between per-run median step times. Ranges are the three paired
observations, not confidence intervals. Each row is an offload on/off comparison
within this revision and fusion setting; no controlled fusion-to-fusion or
version-to-version comparison was performed. Historical results on GPU 6 retain
their original source/device identities and are not substituted for these runs.

All full-precision logged loss values match across all six runs of each fusion
setting. This does not establish full gradient/Adam equality at sequence 4096 or
global native-fused bitwise determinism. Separate short four-rank full-state checks
are provided in the adjacent four-gpu bundle. There is no convergence qualification
or agreed numerical performance budget.

## Raw data and validation

- manifest.json: exact source, commands, exits, source/script/input hashes, GPU UUID,
  container settings, cache policy and cleanup.
- environment.json: imported Core path and package versions.
- Each job: original rank stdout/stderr, launcher.log, result.rank0.json,
  per-second occupancy/temperature/power/clocks samples, and tuning-cache metadata.
- performance-samples.csv: all 120 timed samples.
- performance-summary.json and performance-resource-summary.json: derived results.
- offline-performance.log and frozen-source-check.json/log: actual verification.

The measured GPU remained exclusive in all 12 jobs: the sampler rejects external
compute processes. Other GPUs on the host had unrelated workloads; the host was
not exclusive. Defaults for clocks and power were retained and recorded.
Each fusion setting starts with a fresh cache on GPU 1, then retains its isolated
rank-0 cache across matched arms and repeats. No previous timing samples or seeded
compiler entries are reused.

A prior preflight on GPU 6 was rejected before any container or test process was
started because another user's process occupied it. Its log and reason are kept
as launch-gpu6-unavailable.*; it contributes no measurements.

The stage-specific launcher uses recorded host paths and a locally available
container digest. The shared helper's historical standalone CLI/source-after.json
is not used to select this run's source. Caches, weights, data and source archives
are excluded from the public export. CPU result rechecking:

```bash
python3 -S scripts/summarize_performance.py
```
