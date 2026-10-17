# Qwen restart/stability and GDN2/TE graph validation

## Executed versions

| Workload | Exact executed commit | Result |
| --- | --- | --- |
| Full local Qwen stability | `87c22d1abae13abf375e8ed4ddfe62301cde87f4` | 250 optimizer steps, finite loss/global gradient norm, stable pool and steady GPU memory |
| Native full-Qwen checkpoint/restart, one and two GPUs | `485cb5fa2ffedaaaa0a1a4d095e43ddd99487c1d` | Continuous 6 steps versus 2 + native save + fresh-process 4; all compared states actually byte-identical |
| GDN/GDN2 scopes/configuration and native DP training | `5b79fba4c3ff38cd00abafa6c699787e3f9a7054` | 90 scope/configuration cases and 18 complete-decoder cases; zero skips |
| Pre-fix real TE graph run | `5b79fba4c3ff38cd00abafa6c699787e3f9a7054` | Failed: GDN2 whole-layer replay produces NaN gradient norm; four earlier cases passed, one later case was not run |
| First effective replay-stream fix | `88fc8a84a9ee41d507a4293fe4e9b36b7981b238` | 6 real graph cases passed on one GPU |
| Single-GPU graphs including fraction zero | `2e84e21d03904a5c51182e7140c683110e574142` | 7 passed; zero skips |
| Two-GPU graph matrix before dependency-test guards | `2e84e21d03904a5c51182e7140c683110e574142` | 21 passed per rank; zero skips |
| Two-GPU matrices before the eager boundary fix | `043e3173451bea74e63c02b197867bf356ad72e9` | 18 native and 21 graph cases passed on each rank; zero skips |
| GDN2 stack benchmark (unchanged standalone recurrence path) | `043e3173451bea74e63c02b197867bf356ad72e9` | Three fresh-process baseline/offload pairs; all 8 changing-input outputs and input/parameter gradients byte-identical in each pair |
| Final native/real-graph matrices after the eager-boundary fix | `f19ebc5b1da1c9b0075eecfd2f93fc6ca4b3ba8d` | 36 native and 21 graph cases passed per rank; zero skips |

Final native coverage adds GDN/GDN2 DP2 to the GDN2 TP2+SP/PP2/VPP2 selection and explicitly sets graph warmup to zero. The graph matrix still uses three graph warmup steps.

A final constructor review found that an empty graph-module list had also marked eager GDN layers as graph-offloaded. This incorrectly imposed graph-version and positive-warmup requirements on eager training. The [before-fix regression](eager-guard-before/manifest.json) uses source `043e31734` plus the exported one-line native-fixture patch and fails in a real decoder constructor with graph warmup zero. The final fix adds the missing TE-implementation condition; [final commands and reports](extensions-eager-guard-final/manifest.json) verify the corrected eager boundary and retained real graph behavior.

The two-GPU checkpoint stage also ran 18 GDN2 native cases per rank at
`9b3145584f9776c2239c1c275569c325feada50b`; the final-source native matrix supersedes those results.

`485cb5fa2` adds only the checkpoint validation tool, its boundary tests and usage documentation to #7908. Its entire `megatron` production tree is identical to `87c22d1ab`:
`b80f676df5fdcb0b6ba1a0337b5034065cb7619d`. The corresponding `megatron/core` tree is
`7840a43cf86ff1649d39899bf3b5b3271870203b`.
The extension's final Core tree is `966683849709faa700a7347d313c9d29b75fd53f`; the earlier `043e31734` Core tree is `0b545fb8c83bd73c6c1f323f7e7a0bd8aef8ec34`.
The stream fix is already present in `88fc8a84a`; the final `f19ebc5b1` change limits graph flags/requirements to enabled TE graphs.
[Source identities and tree hashes](source-provenance.json) and per-stage `source-materialization.json` files retain archive hashes and SHA256 file/symlink inventories. Source archives, caches and tensor/checkpoint payloads are not exported.

These results do not replace the [earlier four-card and complete-Qwen performance evidence](../current-head-gpu-20261016/README.md). That bundle retains the executed revisions and historical fixed-reduction kernel measurements. No new four-card result is claimed here.

## Environment

Two NVIDIA RTX A6000s, physical devices 1 and 2, driver 580.178.04.
Docker image: `sha256:061e8fda67a4dd1c1c635129d4ba0442b7a093cc6fd737423e9589eb6be22034`.
Python 3.12; Torch 2.11.0+cu130/CUDA 13.0; Transformer Engine 2.20.2;
FLA 0.5.1; Triton 3.6.0; causal-conv1d 1.6.1; flash-attn 2.8.3.
[Package versions](package-versions.json) and the manifests retain actual device UUIDs, commands, environment variables and timestamps.

The local Qwen3.5-0.8B text model uses all 24 decoder blocks, embedding, vocabulary loss and BF16 Adam. Vision and MTP are excluded. Megatron Bridge revision
`c860f8a5bc5fddd78690f32baa0b8696774308b4` is used with the import patch recorded by the manifests. Local weight/config/data hashes are retained; model weights, token data and training data are not published.

## Checkpoint restart

[Single-card commands](checkpoint/manifest.json) cover disabled offload and all four offloaded fusion/norm-recomputation settings. Each setting runs three fresh torchrun processes: 6 continuous steps, a 2-step prefix saved with Megatron's native distributed checkpointing, then 4 resumed steps. Sequence length is 128, with four microbatches, fraction 1 when enabled and a 1024-element test threshold.

[Two-card commands](two-gpu-retry/manifest.json) run TP2+SP, unfused offload and norm recomputation with the same 6 versus 2+4 comparison.
[Single-card comparison](checkpoint/checkpoint-comparison.json) and
[two-card comparison](two-gpu-retry/checkpoint-comparison.json) check losses, accumulated gradients, updated model parameters, all Adam tensors/scalars and FP32 master weights. All 14 prefix/resume comparisons have zero differing leaves. Resumed processes restore Python, NumPy, Torch CPU/CUDA and Megatron tracker RNG states exactly to their saved values. Initial unseeded Python/NumPy states across independent baseline processes are not asserted equal.

Raw `result.rank*.json` files contain complete state fingerprints; numeric comparison reports and `restored-rng.rank*.json` files accompany the per-rank stdout/stderr. Native checkpoint payloads were hashed and removed only after successful fresh-process loading/comparison; retention manifests record the hashes and sizes. Resume assumes unchanged model configuration, input data and parallel topology. It rebuilds transient offload/pool state rather than checkpointing it.

Twelve CPU tests for the Qwen validation tool pass at `485cb5fa2`; see [output](checkpoint-cpu-tests-final.log).

## 250-step stability

[Command and input hashes](stability-manifest.json), [all 250 steps](stability-steps.jsonl) and
[measured-step results](stability-result.rank0.json) cover unfused BF16/FLA offload, sequence 2048, four microbatches, fraction 1, the default 1,048,576-element threshold, and norm recomputation off. Three warmup plus 247 measured optimizer steps are recorded.

The pool has 72 buffers throughout; outstanding ownership is zero before every native schedule reset, and allocation/free request counts agree. After warmup, allocated bytes, reserved bytes and selected transfer bytes remain constant. Losses and global gradient norms are finite and norms are positive.

Another user's job started on GPU 1 after this run began and overlapped the remainder; [timestamps](gpu1-availability-change.json) retain this fact. This run qualifies stability, not exclusive-device throughput. Repeated use of a small local dataset does not establish convergence or model quality.

## GDN2 and real TE graphs

[Final commands](extensions-eager-guard-final/manifest.json) and per-rank reports/logs cover GDN/GDN2 native DP2 and GDN2 native TP2+SP, PP2 and interleaved PP2 with norm recomputation off/on and fractions 0/0.5/1. The graph matrix covers DP2, TP2+SP and PP2, both GDN fusion settings, GDN2, attention/whole-layer capture, norm recomputation, delayed offload and fraction zero. The single-GPU run covers seven graph cases separately.

Each graph arm performs three native optimizer steps, creates graphs through `TECudaGraphHelper`, asserts that real graphs exist, then performs three replay steps with changing inputs. Complete losses/gradients/weights/optimizer states are compared. Unfused states compare exactly; native fused states use `rtol=1e-5`, `atol=1e-8` because gate-gradient atomic reductions are not globally bitwise deterministic. These graph tests do not claim bitwise determinism for the fused kernel. Enabled nonzero-fraction arms verify actual transfers and every step verifies zero outstanding pinned-buffer ownership before reset.

The [pre-fix failure](extensions-graph-contract/graphs/launcher.log) remains labeled failed. The fix edits the existing replay-argument method: use TE's caller/autograd stream instead of passing the manager stream/event overrides, while retaining the existing D2H/H2D callbacks. This preserves the ordering required for [graphs sharing a private memory pool](https://docs.pytorch.org/docs/2.11/notes/cuda.html#sharing-memory-across-captures). No numerical kernel, fixed launch geometry, extra clone, transfer fence or second manager is introduced.

A test-only wrapper raises the distributed store timeout to 10 minutes before lazy PP initialization; the first cold-compile attempt exceeded the default 60 seconds asymmetrically across ranks. Repository process-group utilities are unchanged. GDN2 requires `fla.ops.gdn2`; TE graph offload requires Torch >= 2.9 and TE >= 2.14. Local/full-iteration CUDA graphs remain unsupported.

## GDN2 memory/runtime trade-off

[Commands and device telemetry](extensions-ready-benchmark/manifest.json) and
[derived paired results](extensions-ready-benchmark/comparison.json) use four GDN2 blocks, hidden size 1024, 16 key/value heads, head dimension 128, sequence 2048, batch 1, fraction 1 and the default minimum tensor size. There are three fresh-process pairs, each with three warmup and five timed forward/backward steps per arm. Baseline precedes offload in every pair; snapshots are outside timing. Devices 1 and 2 are idle at pair boundaries and no other validation jobs run alongside these measurements.

| Pair | Baseline median (ms) | Offload median (ms) | Paired change | Peak allocated saved (MiB) |
| --- | --- | --- | --- | --- |
| 0 | 32.603 | 51.650 | +58.42% | 192.000 |
| 1 | 33.441 | 53.925 | +61.25% | 192.000 |
| 2 | 36.402 | 51.891 | +42.55% | 192.000 |

The benchmark executed at `043e31734`. Its tool, GDN2 recurrence, common scope, configuration and manager files are identical at final `f19ebc5b1`; their Git blob identities are recorded in `source-provenance.json`. This standalone stack does not construct `TransformerLayer`, the only production file changed by the eager-boundary fix. It has not been rerun at the new commit.

Each pair saves approximately 192 MiB (16.88%) of peak allocated GPU memory and selects 240 MiB of transfers over three steady offloaded groups. The median paired runtime change is +58.42%, with range +42.55% to +61.25%. Outputs and all input/parameter gradients match byte-for-byte across eight distinct warmup/measured inputs; no pinned buffer remains in use after backward.

This is a small GDN2 stack benchmark without vocabulary loss, optimizer updates or CUDA graphs. The transfer cost is material. It is not an end-to-end training benefit or a comparison against the historical complete-Qwen measurements.

## Reproduction and offline checks

Each stage's `manifest.json` contains the exact argument-vector commands and executed source. Each torchrun job retains every rank's stdout/stderr and a per-rank pytest report or full-state result. Native CPU/GPU jobs run inside the recorded container. The exported launchers preserve this workstation's paths; use the recorded native tool/pytest arguments with equivalent local model/data/environment paths on another machine. The shared launcher helper is imported for container/runtime functions only; stage-specific manifests and frozen-source PYTHONPATH identify the actual executed source.

With the dependencies and model/data available, the main entry points are:

```bash
# Final GDN2 native training on two GPUs
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m pytest -q tests/unit_tests/ssm/test_gdn_offload_training.py \
  -k 'dp or (gdn2 and (tp_sp or pp) and not tp_sp_cp and not tp_sp_pp)'
# Final genuine graph matrix on two GPUs
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m pytest -q tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py
# Single GPU: same graph test with --nproc-per-node=1 and -k tp1
```

From this bundle, no GPU or PyTorch is needed to recheck the exported hashes, restarted state equality, stability counters, rank outcomes and benchmark arithmetic:

```bash
python3 -B -S verify_followup.py
```

`SHA256SUMS` covers the exported files except itself and `verification.log`; the latter is generated by that command. The checker validates recorded evidence only and does not execute or independently reproduce GPU training. [Formatting/lint commands](quality-checks.json) record the targeted Black/isort/Ruff/project-Pylint and determinism-association checks; `quality-eager-final.json` and the corresponding logs cover the final two changed files. Upstream GPU CI, full-iteration/local graphs, four-card GDN2 combinations, pretrained GDN2 training and composition with #7852 are not qualified here.
