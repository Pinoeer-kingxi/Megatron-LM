# GDN activation-offload GPU validation

The local acceptance checks pass for these source revisions. This directory
contains the actual commands, per-rank logs, result JSON, script snapshots and
checksums. It supplements the CPU-only `rebase-20261015` bundle without changing
its historical results.

| PR | Tested source commit |
| --- | --- |
| #8243: offload manager | `af9569270f476dc201484296315a9c30dcaa3feb` |
| #8244: gate backward | `9d54b8729c9277f1e032110c5e1b149e9654bcc5` |
| #7908: GDN integration | `8a2d19894288a8eb890dfdf20c9d54683035de80` |
| Atomic gate baseline | `9131593673d037d360d252904f9198d36bc0c246` |

These were still the three GitHub PR heads when checked after starting the final
control. Source worktrees were clean. No PR implementation was changed for this
validation.

## Results

All GPU runs used RTX A6000s. Counts below are per rank, not multiplied by the
number of ranks. The completed tests had no skips.

| Check | Executed result |
| --- | --- |
| Delayed transfers and pool reuse | 4 tests on one GPU |
| Gate replay and Torch numerical reference | 16 tests on one GPU |
| GDN scope, alias, anchor, configuration and recomputation regressions | 73 tests on one GPU |
| Shared manager | 16 tests on each of four ranks |
| Existing GPT/MoE, EP overlap and CUDA-graph regressions | 24 tests on each of four ranks |
| Native training matrix | 96 tests on each of four ranks |
| Complete Qwen text-model training | Four fusion/recomputation pairs; 48 exact rank/step comparisons over 96 recorded states |
| Gate-wrapper benchmark | Eight shapes, nine paired samples per shape |
| Acceptance-script CPU regressions | 14 tests |

Qwen uses all 24 text decoder layers of the local Qwen3.5-0.8B checkpoint,
BF16/FLA, TP2+SP+PP2, sequence length 128, four microbatches and three optimizer
steps. It compares losses, all recorded parameter gradients, updated weights and
Adam/master-parameter state byte for byte. The offload arms select real transfers
and check pool ownership before manager reset. This is a short correctness run;
it does not train vision or establish long-sequence throughput.

The final cold-cache control ran on physical GPUs **1,2,3,6**. Its baseline,
repeated baseline and offload arm agree on every rank and step. All 48 recorded
autotune selections match their rank's persisted choices; the later arms do not
change those cache files. Each rank selects 32,542,720 offload bytes. This is a
transfer-selection count, not a measurement of net peak-memory savings.

The earlier native matrix and other three Qwen pairs ran on **0,1,3,6** at the
same source revisions. They were not rerun or relabeled as new executions.
`gpu-validation-20261016.json` lists the exact run references, GPU UUIDs and
manifest hashes. Its status is `passed_current_sources`.

## First-baseline drift

The original Qwen matrix failed because its first unfused baseline differed at
step 2. Two subsequent offload-disabled baselines agreed with the offload arm.
A fresh shared-cache reproduction then showed six runtime autotune choices that
were not preserved in the shared disk cache, together with different results in
two offload-disabled runs.

Triton 3.6.0 keys its autotuning cache by architecture, kernel and tuning inputs,
without a GPU UUID or rank. Concurrent cold-start ranks can select different
configurations and overwrite the same disk entry. A later process may therefore
use a different arithmetic configuration from the first process.

The Qwen launcher now gives each rank its own persistent Triton cache before
starting the target interpreter. The final four-rank control starts with empty
caches and validates both baseline repetition and the direct offload comparison.
It changes neither the PR kernels nor the equality criterion. Separate empty
autotuning runs can still select different valid floating-point configurations;
this is not a global bitwise-determinism guarantee across those configurations.

The original failed matrix and runs interrupted by other users' GPU processes
remain unchanged. The single-GPU manifest also retains its UUID-format reporting
failure: all GPU commands exited zero, and the corrected verifier rechecks their
raw output. CUDA reports a bare UUID while nvidia-smi adds `GPU-`; a CPU regression
covers this normalization and still rejects a different device.

## Performance scope

For the eight measured gate-wrapper shapes, paired runtime medians improve by
5.9%–8.7% relative to the original atomic implementation. Extra allocated peak
memory increases by 2–32 KiB. The raw samples and their ranges are retained;
individual samples vary. These are wrapper measurements with offload disabled,
not end-to-end offload speedups.

Earlier end-to-end offload measurements retain their original source attribution:
unfused `4e18562c6bcf894795d23378a8f1699c79e4b797` and fixed-fused
`2f1ee4f2e3dd6d909d08981338f5879632131139`. They are not new performance results
for the rebased heads.

## Recheck without GPUs

From this directory, using only Python's standard library and `sha256sum`:

```bash
sha256sum --check --quiet gpu-evidence-20261016.SHA256SUMS
python3 -S analyze-qwen-cache-20261016.py
python3 -S validate-gpu-results-20261016.py
```

Both Python commands exit zero. They regenerate derived reports for the local
directory; they do not change the raw run files. Verify checksums before
regenerating the reports. The exported bundle was also extracted into a temporary
directory and rechecked without access to the original workspace, weights or GPUs.

The [Chinese report](gpu-acceptance-20261015/README.md) describes the environment,
commands and individual runs. GPU replay requires the recorded local container
environment and data; model weights, training data, compiled caches and Python
bytecode are excluded from this bundle. Local acceptance does not constitute
upstream GPU CI execution or maintainer approval.
