# GDN low-precision validation on SM120

This bundle follows the [scoped BF16 review](../offload-scope-review-20261026). It does not replace that report or relabel its failed checks. The low-precision source is the separate experimental branch `experiment/gdn-low-precision-20261026`, tested at **`688159928298844dc00d9d3a9ce33774735d997c`**. #7908 remains at `d8db9192d24e1e5f8e4884f1d5acfb463a75452d`, and #8243 remains at `2b4211c0681a3c9d08b9b685af687ed13e6f4053`. The two evaluation-first/size-policy follow-ups are excluded from all three trees.

## Actual complete-model results

NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120, 96 GB. FP8 delayed ran on physical GPU 1; NVFP4 and FP8 tensorwise ran on GPU 3. All three arms of each recipe ran on the same device. The devices were shared with existing workloads, so these runs support correctness comparisons, not throughput or peak-memory comparisons.

| Recipe | Disabled vs repeated-disabled vs offloaded | Actual GEMM operands |
| --- | --- | --- |
| FP8 delayed | Exact full-state match through three optimizer steps | Float8Tensor / Float8TensorStorage |
| FP8 tensorwise | Exact full-state match through three optimizer steps | Float8Tensor / Float8TensorStorage |
| NVFP4, RHT and stochastic rounding disabled | Exact full-state match through three optimizer steps | NVFP4Tensor / NVFP4TensorStorage |

Each of the nine training jobs uses local pretrained Qwen3.5-0.8B weights and text data, all 24 text decoder layers, embeddings, mixed GDN/attention, MLPs, vocabulary loss, native DDP and BF16 Adam/master weights. Sequence length 128, four microbatches, one warmup plus two further optimizer steps, selective output-norm recomputation enabled, unfused pre-GDR, 1024-element transfer threshold. Enabled arms use fraction 1 and select real transfers. The compared state includes every recorded loss, gradient norm, all 230 parameter-gradient and updated-weight fingerprints, and the complete Adam/master-state fingerprint tree. Pool ownership is checked before the schedule resets its counters.

The observer delegates each GEMM and Inductor kernel to its original implementation and records operand types and selected L2-normalization configurations. It does not substitute kernels or transform tensors. Check-state hashing and observation overhead exclude these runs from performance claims.

## What was corrected

**Missing NVFP4 context in the validation tool.** Bridge's custom Qwen decoder supplies FP8 contexts but does not enter an FP4 context. Consequently, the earlier `--precision fp4` Qwen jobs at `ebdb25fc493b528764a4116313effaf8504eec3d` executed BF16 GEMMs. The tool now enters Megatron's existing `get_fp4_context(config)` around the forward call; BF16/FP8 retain the helper's existing null context. No custom quantization mechanism was added. The same real-kernel regression fails on the old tool and passes at `e553078013811377b5991ee80c599dc153d8efa2`; the unchanged test also passes at the final source. Whole-model GEMM observations independently show the old tool lacks NVFP4 operands and the corrected tool uses them.

**Fresh-process compiler tuning was not held constant.** The private runtime recreates `/tmp` for every process. Its Triton cache was persistent, but TorchInductor's separate cache was discarded. Inductor captures FLA's user Triton L2-normalization kernels and saves its own tuning choices. Three cold-cache FP8 delayed controls at `e553078013811377b5991ee80c599dc153d8efa2` selected backward `(BT, warps)` values `(8, 8)`, `(16, 4)` and `(64, 2)`. The first and third use the same `PYTHONHASHSEED=0` and still have different gradients/optimizer states. Those unsuccessful controls remain unchanged in this bundle.

Matched comparisons now preserve **both** `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR`, with a separate directory per rank and comparison configuration. The three final arms of each recipe reuse the same cache roots and have identical observed forward/backward L2-normalization choices. All recorded states match exactly. No kernel was rewritten, no tile size was fixed, and no comparison tolerance was relaxed.

A separate replay isolates the numerical effect on one actual Qwen backward input, flattened to the native kernel's `(4096, 128)` view. Each observed configuration repeats exactly three times; changing `(BT, warps)` from `(8, 8)` to `(16, 4)` changes three BF16 output elements, with maximum absolute difference `4.656612873077393e-10`. The latter also matches the captured Inductor result. This identifies a real source of rounding differences; it does not indicate an offload data corruption bug. The raw activation/gradient capture remains private; the replay code, configurations and aggregate results are public.

The first FlashAttention binary was built for SM80 and could not execute on SM120. The runtime replacement was built from unmodified official flash-attn 2.8.3 source for SM120 using CUDA 13. The source/archive checksum and resulting library checksum are in this bundle. The build-time report preserves its original `gpu_execution_verified=false`; the later complete-model logs provide actual device execution evidence.

## Other tests and failed attempts

- Current source: 13 Qwen tool tests passed, including real NVFP4 forward/backward; no skips.
- Current source: 47 selected configuration/native training tests passed; no selected test skipped, and 156 other cases were deselected. The native training matrix covers FP8 delayed, FP8 tensorwise and constrained NVFP4; fractions 0/.5/1; output-norm recomputation off/on; native pre-GDR fusion off/on. It compares full decoder/DDP/Adam states for three steps. Unfused cases require exact equality; fused cases retain the existing `rtol=1e-5, atol=1e-8` check for native atomic parameter reductions.
- The old-tool regression is an **expected failure**, not a passed test. Its raw assertion and matching test-source checksum are retained.
- The corrected NVFP4 runs with fresh Inductor caches at `ae126ed2a1e4c6e32cc85283b1fd45d887e52be8` executed real quantized GEMMs but did not have equal repeated baselines. They remain diagnostic failed qualifications.
- The first final-source NVFP4 attempt on GPU 1 ran out of device memory while other workloads occupied the card, before Adam state allocation completed. Its OOM log is retained and is not counted as a successful training run. Existing peer processes were left running; the later GPU 3 run completed with sufficient spare memory.
- The diagnostic input capture completed, but the first standalone replay did not flatten Inductor's multidimensional `rstd` view. That harness error and its original script are retained. The corrected standalone replay uses the native kernel's flat view and passes; it required no training-code change.
- Black, isort, Ruff and code whitespace checks passed; exact commands and output are in `code-checks.json`.

## Limits

This is single-rank, short-run correctness evidence. It does not qualify multi-GPU low-precision training, checkpoint replay of precision-scaling state, long-run convergence, CUDA graphs, GDN2, throughput or memory savings. The FLA recurrence and host transfers remain BF16/FP32; only TE projections/MLPs use the selected low-precision recipe. NVFP4 uses `NVTE_NVFP4_DISABLE_RHT=1` and `NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1` before Python startup. It is not qualification of the full NVFP4 recipe. No new upstream GPU CI run is claimed.

Historical A6000 memory/runtime results remain attached to their original source revisions. They are not presented as performance measurements of this low-precision branch or the narrowed BF16 PR head.

## Reproduce and audit

Every job records its exact command, GPU UUID, before/after revision, clean-tree checks, exit code and rank logs. The environment, runtime wrappers, observer and launch scripts are included. Supply compatible local Qwen3.5 weights and JSONL text data; neither is redistributed here. The final GPU 1/3 wrappers bind only the requested physical device and use persistent compiler caches. In a multi-rank reproduction, append `rank-${RANK}` to **both** cache roots before starting the training interpreter. Keep each recipe/topology/fusion/recomputation comparison's cache directories unchanged between its fresh-process arms.

To audit the downloaded records without CUDA or third-party packages:

```bash
python audit.py
```

The audit recomputes the full-state comparisons and verifies actual quantized GEMM use, compiler-choice consistency, test outcomes, exact revisions, failed controls and checksums. It does not rerun training.
