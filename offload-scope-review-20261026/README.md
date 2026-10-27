# Offload scope review and SM120 validation

The source revisions are recorded in [sources.json](sources.json). This bundle adds evidence; it does not replace the older reports or their executed revisions.

## Scoped changes

- **#8243 — `2b4211c0681a3c9d08b9b685af687ed13e6f4053`:** retain the reviewed forward-chunk completion fix and its fixed-length native GPT regression. The entire source tree is byte-identical to the previously reviewed `f9d0b7afd150caf5f323346f8cddd3c5e6200b1d` tree.
- **#7908 — `d8db9192d24e1e5f8e4884f1d5acfb463a75452d`:** merge the narrowed prerequisite. The diff from #8243 contains no GPT/Hybrid preprocessing or post-warmup policy changes. The feature remains opt-in BF16/FLA recurrence offloading.
- The separate evaluation-first and tensor-size policy follow-ups, including their tests, are retained on `fix/offload-warmup-boundaries` at `a49dd72bfbc71325963e3ced35463c1ad76a91f2`. That tree matches `041a5014201f5175048536b906d22f50d7e0f3c8`. They are excluded from the two scoped PRs.

## Current-head execution

One NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120, 96 GB), physical GPU 0. Other devices on the shared host were in use. These are correctness checks, not performance measurements.

| Source | Actual result |
| --- | --- |
| #8243, `2b4211c06` | Six selected tests passed: fixed-length native GPT at fractions 0/.5/1 and three existing TE graph compatibility cases. No skips. |
| #7908, `d8db9192d` | 104 passed, 84 topology cases skipped in the scope/configuration/native decoder-DDP-Adam/tool suite. |
| #7908, `d8db9192d`, complete Qwen | Three fresh-process arms match exactly through three optimizer steps: disabled, repeated-disabled and offloaded, including losses, gradient norms, all gradients/weights and complete Adam/master state. |

Each job directory contains its exact command, before/after revision, clean-tree checks, exit code and rank-0 stdout/stderr. Pytest reports distinguish executed and skipped cases. No four-rank rerun or upstream GPU CI execution is claimed here.

The complete-Qwen comparison uses local pretrained Qwen3.5-0.8B weights and local text data, all 24 text decoder layers, embeddings, mixed GDN/attention, MLPs, vocabulary loss, native DDP and BF16 Adam/master parameters. Sequence length 128, four microbatches, three optimizer steps, selective output-norm recomputation, unfused pre-GDR, and a 1024-element offload threshold. Each arm starts in a fresh interpreter. Check-state snapshots are outside any performance claim. Weights and data are not distributed in this bundle.

The original fixed-length `Chunk mismatch` failure and two-/four-rank controls at `d8be77f56` remain in the [earlier evidence](../offload-scope-20261022). The A6000 S4096 memory/runtime measurements remain under [their executed revision `1f2c9ae4f`](../gdn-acceptance-20261026). That revision includes the separately retained lifecycle follow-ups; its performance is not presented as a rerun of the narrowed head.

## Experimental low precision

These separate jobs execute `ebdb25fc493b528764a4116313effaf8504eec3d`, not the BF16 PR head. [Environment and dependency versions](precision/environment.json) include the SM120 FlashAttention library checksum. A prior SM80-only FlashAttention binary could not execute the full model on this device. The replacement was built from the unmodified official flash-attn 2.8.3 source, using CUDA 13 and the SM120 target; the source URL/checksum and build-time result are retained. Actual complete-model forward/backward/optimizer execution is recorded in the job logs.

| Requested recipe | Complete-Qwen result |
| --- | --- |
| FP8 tensorwise | Three fresh-process arms match exactly: disabled, repeated-disabled and offloaded. |
| FP8 delayed | The first disabled run differs; the next disabled, offloaded and confirmation-disabled runs match. The full repeated-run qualification remains incomplete. |
| NVFP4 | The Bridge decoder never enters an FP4 context in this tool revision, so these are BF16 executions and do not qualify NVFP4. Disabled runs also differ; the offloaded run matches the first disabled run. |

All 10 original training arms exit successfully. Source inspection subsequently established that the requested FP4 context was absent from Bridge's custom decoder; a tool-only correction and real GEMM-operand checks are being validated separately. Comparisons retain the failed checks instead of counting process completion as correctness. Comparisons include every recorded loss, gradient norm, parameter-gradient, updated-weight and complete Adam/master-state fingerprint; enabled arms select actual transfers.

A further three-arm **offload-disabled** control uses an independent cache, copying compiled kernels but removing tuning-result files in that private copy. The two warm runs also differ, with unchanged autotuning-file checksums between them. This does not establish a cold-cache explanation. The cause remains unresolved, so FP8 delayed complete-Qwen reproducibility is not qualified by this bundle. NVFP4 additionally requires the missing context to be corrected and actual quantized GEMMs to be verified.

NVFP4 uses `NVTE_NVFP4_DISABLE_RHT=1` and `NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1`, set before interpreter startup. This is a constrained recipe. The short runs do not qualify full NVFP4, convergence, throughput or multi-GPU behavior. The BF16 PR retains its FP8/FP4 configuration rejection.

## Reproduce and audit

Launch scripts and runtime wrappers are retained under [scripts](scripts). Paths in the command manifests refer to the validation machine. Supply equivalent local weights/data and the recorded dependencies to rerun them; only GPU 0 is bound into the private runtime. Per-rank persistent Triton cache directories and fresh processes are used. The original failed launchers and comparison JSON files are retained under [precision](precision).

To verify this downloaded bundle using only the Python standard library:

```bash
python audit.py
```

The audit checks checksums, actual test outcomes, source revisions, clean-tree/exit records and all recorded state comparisons. Its integrity result does not convert the unsuccessful precision qualifications into passes.
