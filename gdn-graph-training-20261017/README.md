# Complete training steps with TE CUDA graphs

Executed MCore source: `f19ebc5b1da1c9b0075eecfd2f93fc6ca4b3ba8d`.
Core tree: `966683849709faa700a7347d313c9d29b75fd53f`.
These measurements concern PR #8308. Source and dependency identities are in
`source-identity.json`, `environment.json`, `input-identity.json` and `manifest.json`.

## Workloads and measurement boundary

- Qwen3.5-0.8B: all 24 text decoder blocks, loaded from local Hugging Face weights,
  with embeddings, attention/GDN, MLPs, vocabulary loss and native BF16 Adam.
  Vision and MTP are excluded. The existing Qwen tool supplies data and its
  Bridge-specific forward setup.
- GDN2: a seeded, randomly initialized eight-block HybridModel, pattern `GGG*GGG*`,
  hidden size 1024, FFN 3584, attention head dimension 128, GDN key/value heads 8/16,
  GDN head dimension 128 and vocabulary 4096. Its native training forward is reused
  from the existing integration test; no Bridge-specific preprocessing runs here.
- Both: sequence 2048, microbatch size 1, four accumulated microbatches per optimizer
  step, BF16/FLA, unfused pre-GDR, norm recomputation off, fraction 1, default
  1,048,576-element threshold, and whole-layer TE graph capture.
- One A6000 per arm: physical GPU 1 for Qwen, GPU 2 for GDN2. All jobs run serially.
  Each comparison arm starts in a fresh process. Per-model persistent Triton caches
  are shared across matched arms. Pair order is disabled/enabled, enabled/disabled,
  disabled/enabled. Other GPUs on this shared host run unrelated jobs; these are
  single-GPU training-step comparisons, without a multi-GPU scaling claim.
- Each performance arm trains three native steps, creates real graphs, trains three
  replay warmup steps and measures ten replay steps. The helper creates one graph
  per layer without PP and reuses it across the four microbatches.
- Timing includes zeroing gradients, native forward/backward/finalization, the Adam
  update and CUDA completion. Input preparation/hashing, capture, state fingerprints
  and reporting are outside the performance timing boundary. The optimizer itself
  stays eager; this is whole-layer TE capture, not full-iteration capture.

## Correctness

Both models pass six-step offload-disabled/enabled comparisons: three native and
three replay steps with changing inputs. Loss, gradient norm, all gradient and
weight byte fingerprints, and complete optimizer/master-state fingerprints match
exactly. Real graph creation is asserted for all 24/8 layers. Every optimizer step
has finite losses and a finite positive gradient norm. All 16-step performance
pairs also have identical loss and gradient-norm sequences, including later replay.

Pinned-buffer ownership is zero in native gradient finalization before schedule
resets. Enabled replay keeps pool allocation/free counters constant and selects
positive offload bytes. The pool retains 72 buffers for Qwen and 54 for GDN2.
The two correctness arms per model are separate from all timed runs.

## Performance

Each cell below reports one fresh-process pair. Time is the median of ten measured
steps; memory is the maximum of the ten recorded peaks. MiB means 1,048,576 bytes.

| Model | Pair | Step ms, disabled → enabled | Change | Allocated MiB, disabled → enabled | Reserved MiB, disabled → enabled |
| --- | --- | --- | --- | --- | --- |
| qwen | 1 | 849.215 → 1028.444 | +21.11% | 18070.68 → 18070.68 | 24838 → 24394 |
| qwen | 2 | 850.139 → 1025.935 | +20.68% | 18070.68 → 18070.68 | 24838 → 24396 |
| qwen | 3 | 849.409 → 1027.798 | +21.00% | 18070.68 → 18070.68 | 24840 → 24394 |
| gdn2 | 1 | 229.305 → 449.546 | +96.05% | 3211.12 → 3212.12 | 5202 → 4954 |
| gdn2 | 2 | 228.826 → 449.535 | +96.45% | 3211.12 → 3212.12 | 5202 → 4944 |
| gdn2 | 3 | 228.603 → 449.568 | +96.66% | 3211.12 → 3212.12 | 5230 → 4944 |

Paired runtime changes have median **+21.00%** for Qwen (20.68–21.11%) and
**+96.45%** for this GDN2 model (96.05–96.66%). Median paired reserved reductions
are **444 MiB** (442–446) and **258 MiB** (248–286), respectively. Allocated peaks
increase by 512 bytes and approximately 1 MiB. These runs show a reduction in
reserved memory, with no allocated-peak reduction and substantial runtime cost.
The on/off pairs hold TE graph mode fixed. They do not measure graph-versus-eager
speedup, model convergence, or performance on Hopper/Blackwell.

The earlier GDN2 four-block stack measurement at `043e3173451bea74e63c02b197867bf356ad72e9`
saved 192 MiB allocated (16.88%), with +58.42% median runtime. That measurement
excluded loss, optimizer updates and CUDA graphs and remains a separate historical
result. Qwen eager-mode measurements at `87c22d1ab` likewise retain their original
revision and workload in the previous evidence bundle.

## Reproduction and verification

Use the MCore revision above in an environment matching `environment.json`.
For the pretrained Qwen run, use Bridge revision
`c860f8a5bc5fddd78690f32baa0b8696774308b4` with the included import-compatibility
patch. Weights and the local messages JSONL are not published. Their file hashes
are recorded; a different dataset can exercise the same checks but will produce
different losses and input hashes. The GDN2 token stream is generated from seed 19.

From that source checkout, run each arm in a fresh process on the same idle GPU:

```bash
export NVTE_CPU_OFFLOAD_V1=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export TRITON_CACHE_DIR=/path/to/persistent-cache-for-this-model
export TORCHINDUCTOR_CACHE_DIR=/path/to/inductor-cache-for-this-model
export PYTHONPATH="$PWD:/path/to/patched-bridge/src:$PYTHONPATH"
python -m torch.distributed.run --standalone --nproc-per-node=1 \
  /path/to/evidence/scripts/benchmark_graph_training.py \
  --model qwen --weights /path/to/Qwen3.5-0.8B --data /path/to/messages.jsonl \
  --output /tmp/qwen-disabled.json
```

Repeat with `--offload` and a different output path. For the correctness pair add
`--check-state --warmup 1 --iterations 2` to both arms. Repeat for `--model gdn2`,
using its own cache root. The same command-line weight/data arguments are accepted
for this synthetic model but are not read. Raw host-specific container/torchrun
argument lists and exit codes for every executed arm are in `manifest.json`.
`run_campaign.py` and `continue_campaign.py` record this host's orchestration;
`benchmark_graph_training.py` is the direct training entry point.

`manifest.json` lists only the 16 qualified jobs. Earlier harness setup attempts
and the corrected GDN2 forward-adapter attempt are excluded. The valid Qwen
controls used an earlier benchmark-script version; its SHA and exact source are
retained per job under `scripts/versions`. Later GPU jobs use the corrected native
GDN2 adapter. All jobs execute the same pinned MCore source. The initial GPU
script inventory is retained separately from the final offline verifier inventory.

To check the published payload and recompute the recorded comparisons:

```bash
sha256sum -c SHA256SUMS
python -B -S scripts/verify_graph_training.py
```

The verifier compares recorded state hashes and recomputes timing/memory summaries.
It is CPU-only and does not execute or replace the GPU jobs. Each job directory
contains `result.json`, launcher output and the original rank-0 stdout/stderr.
`telemetry-controls.csv` covers the earlier Qwen controls; `telemetry.csv` covers
the resumed GDN2 controls and all performance pairs. Job timestamps identify the
relevant intervals. CPU formatting checks and the 12 Qwen CLI boundary tests
are recorded separately and are not GPU-training evidence.

Upstream GPU CI, four-card GDN2 composition and pretrained GDN2 training remain
unqualified by this bundle. Local tests do not grant upstream review/CI approval.
