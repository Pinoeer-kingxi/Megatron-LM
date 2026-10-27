# GDN offloading integration and NVFP4 gradient-accumulation validation

These are validation records, not changes to the production PRs. `audit.py` verifies the recorded source IDs, outcomes, state fingerprints and file checksums; it does not rerun GPU work or recompute GPU tensor equality.

## Revisions and results

| PR | Executed source | Selected tests |
| --- | --- | --- |
| #8243 | `5bc9e72bda70129f088dd78328d4064f1d838c62` | One GPU: 7 passed, including fixed-length fraction 0/.5/1, TE graphs and upstream checkpoint storage. |
| #7908 | `b6fa63ab8c2cb855f0368cd8455245c295cde234` | One GPU: 104 passed, 84 topology skips; four targeted pipeline tests passed. Two GPUs: 24 native TP2+SP/PP2 cases passed per rank. Complete Qwen 0.8B comparisons passed on one GPU and TP2+SP. |
| #8308 | `bb346eb1c3461f471c420bcb7691b425863df3f5` | One GPU: 20 GDN2 scope, 6 native and 7 graph cases passed. Two GPUs: 12 native and 14 graph cases passed per rank. |
| #8543 | `aecc0e8116912554f27a62a51ad2aeca479a7f9e` | One GPU: 111 scope, 54 native low-precision and 23 graph cases passed. Two GPUs: 18 low-precision GDN2 graph cases passed per rank, including fused accumulation. |

The branches incorporate `main` at `3ddfe0fe07efc73dbdc40d4a02acbc1c1f8a9f08`; their dependency commits are ancestors in the listed order. At the final check, `main` had advanced to `9dd63d992a5925dba7f9bfd1aec9dbfb1f54b837` with only a change to `.github/copy-pr-bot.yaml`; all four tested heads merge cleanly with it. No selected cases outside the 84 single-rank topology skips were skipped. Two-GPU jobs ran on devices shared with other processes and are correctness tests, not timing measurements.

The Qwen tool uses local 0.8B checkpoint weights and private local JSONL data, neither included here. Each topology has three fresh processes: offloading disabled, disabled again, and fraction-one offloading enabled. Each process trains three optimizer steps with four microbatches and norm recomputation. The results contain loss, gradient, weight and complete Adam/master-state fingerprints; this audit compares the three sets per rank. Each rank logs the actual imported Megatron Core path. The old `d8db9192` Qwen runner did not record `GDN_SOURCE` in its manifest, so its records alone cannot independently establish the imported Core directory. The new runner explicitly passes it and checks the import path.

## NVFP4 fused accumulation boundary

An unmodified, offload-disabled eager baseline failed at step 0 with TP2+SP, FP4, gradient accumulation fusion and the ordinary optimizer, before any graph capture. On the tested SM120/TE 2.20.2 stack, the failing FP32 `main_grad` address was 8 bytes modulo 256. The real TE Linear control uses NVFP4, the same 448-by-256 weight dimensions and FP32 fused accumulation: addresses offset by 4 or 8 bytes fail with `CUBLAS_STATUS_NOT_SUPPORTED`, while offsets 0, 16, 32 and 64 bytes pass. Changing the FFN partition from 448 to 512 did not fix the baseline. Logs for these failures and controls are included.

This is an existing non-distributed DDP buffer alignment limitation in this environment; it occurs without GDN offloading. It is not fixed by these PRs. The TP2 fusion comparison uses Megatron's distributed optimizer, which aligns parameter buffers; PP2 and one-rank fusion comparisons use the ordinary optimizer. `fused-distopt-control-tp2-gpu02` is an intermediate test commit, not the final-head result.

The CUDA extension `fused_weight_gradient_mlp_cuda` was built in an isolated directory from NVIDIA Apex `575968bc1f9127ccd61003a681472a83af4ff1a1`. Its three source files and `type_shim.h` were copied without source changes; `scripts/setup.py` and `apex-build.log` record the build. The `.so` is not included. The tests use real Apex and Transformer Engine kernels and real process groups.

## Legacy graph option regression

`fp4-legacy-old.log` records the negative test applied to production code at `43e77e7e5fbb53cd86449fc36c841c13ffbc69a3`: `external_cuda_graph=True` incorrectly passes the NVFP4 weight-cache guard. `fp4-legacy-test.diff` is the test-only patch used in that control. The fixed config tests pass in `fp4-legacy-final.log`; final-head GPU scope tests also execute both accepted and rejected settings. The legacy flag is normalized to the TE implementation later in config initialization.

## Environment and reproduction

RTX PRO 6000 Blackwell (SM120), driver 580.126.09, Python 3.12.4, PyTorch 2.11.0+cu130, Transformer Engine 2.20.2, FLA 0.5.1, Triton 3.6.0, CUDA 13.0 compiler, FlashAttention 2.8.3 rebuilt for SM120 and causal-conv1d 1.6.1. NVFP4 runs set `NVTE_NVFP4_DISABLE_RHT=1` and `NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1` before Python starts; they qualify that constrained recipe. No full-recipe, convergence or speedup claim is made.

Each manifest records the torchrun command, GPU UUID, source revision before/after and exit status. The launchers and rank wrappers used for the jobs are under `scripts/`; adapt their local paths, checkpoint and dataset to reproduce them. Triton and TorchInductor caches are kept per rank. `apex-build.log` provides compiler and link commands; reproduce the extension by checking out the specified Apex revision, copying the weight-gradient sources and `type_shim.h` as used in `scripts/setup.py`, and providing the CUDA headers and libraries from the same stack. Upstream GPU CI has not run.

Run `python3 -S audit.py` after downloading this folder to check recorded outcomes, Qwen fingerprints and SHA256 integrity.
