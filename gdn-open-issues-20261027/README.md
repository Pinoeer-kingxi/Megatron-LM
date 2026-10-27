# GDN follow-up GPU validation

The retained manifests contain the actual commands, source revisions and device IDs; `ranks/` contains original stdout and stderr and `reports/` contains per-rank pytest outcomes. Run `python3 -S audit.py` from this directory to check checksums and the recorded comparisons. Paths in the commands refer to the test machine and must be adjusted on another machine. The Qwen runs use local Qwen3.5-0.8B weights and JSONL data, which are not included here.

Environment: RTX PRO 6000 Blackwell, driver 580.126.09, PyTorch 2.11 with CUDA 13.0, Transformer Engine 2.20.2, FLA 0.5.1, Triton 3.6, FlashAttention 2.8.3 and causal-conv1d 1.6.1. Some runs shared GPUs with other processes. Timings from shared runs are not performance claims.

This GPU is SM120. [Transformer Engine 2.20.2 documents NVFP4 training support for SM100 and SM103](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html#supported-devices); the SM120 FP4 training results here are exploratory and require confirmation on a supported training GPU. BF16 results are not subject to that particular NVFP4 limitation.

## Independent regression fixes

* `mrope-graph-865d79d-native-gpu1`: commit `865d79dd37f304a4dffa23158e8df52735d43247`, one real GPU, native GPT/DDP and TE graph, one test passed. The test compares the loss and gradients for two different three-axis position inputs. `mrope-native-baseline-v3-20261027-gpu1` uses the same regression test against `aecc0e8116912554f27a62a51ad2aeca479a7f9e` and fails on the actual replay/eager loss mismatch. The baseline is not a `main` checkout; both commits contain the old graph-input behavior.
* `nvfp4-ddp-66b6bb6-unit-gpu12`: commit `66b6bb62a00bdccf8365096e6828215cffed4ce7`, two real GPUs, eight tests passed per rank. The FP4 tests perform real TE linear forward/backward and DDP reduction, checking FP32 and BF16 gradient buffers, repeated steps and ranks. RHT and stochastic rounding are disabled for this exact-repeat check. `alignment-real-te-baseline-final-gpu1` runs the same regression against `aecc0e8116912554f27a62a51ad2aeca479a7f9e`: both gradient types fail with `CUBLAS_STATUS_NOT_SUPPORTED`.

The `probes/` files are the regression tests used for the old-code controls. `alignment-tp2-{baseline,fixed}-verbose-20261027-gpu03` runs the same eight-layer native GDN2 TP2+SP optimizer training on base `aecc0e8116912554f27a62a51ad2aeca479a7f9e`: both ranks fail before the fix and pass with the retained uncommitted prototype patch. The additional GDN2 TP2+SP integration run at temporary merge `37f34ccaf362ca7e42c0b2e5405455a1fe12993d` includes the earlier alignment commit `37e1ee1`; its production `megatron/` tree matches `66b6bb62a00bdccf8365096e6828215cffed4ce7`, but its tests and manifest are older. Neither prototype nor temporary merge is a test of the final commit.

## Complete Qwen NVFP4 training

`qwen-fp4-mrope-dfaa633-*-gpu0` runs the complete 24-layer Qwen3.5-0.8B language model with local checkpoint/data, sequence length 128 and four microbatches. At `dfaa6338c65edab118ba415fcc14c73241f503a6`, eager, two graph baselines and graph plus offload execute six steps each. RHT and stochastic rounding use the default enabled settings. All recorded input hashes, losses, gradient norms, gradient fingerprints, weights and full optimizer state match across the four arms; the offload arm records actual transfers. These are correctness runs, not benchmarks.

At `c5264b66197040f4309edfdb22dc9bbaf62189b1`, two-GPU GDN2 native and GDN/GDN2 graph suites pass 12 and 14 cases per rank, respectively. At `dfaa6338c65edab118ba415fcc14c73241f503a6`, the two-GPU low-precision graph suite passes 18 cases per rank. These suites cover TP2+SP and PP2 and have no skips.

At `dfaa6338c65edab118ba415fcc14c73241f503a6`, the single-GPU scope/configuration/tool suite passes 111 cases, native low-precision training passes 54 cases and graph capture/replay passes 23 cases, without skips.

The `qwen-mrope-paired-*` runs use an earlier uncommitted mRoPE prototype on base `aecc0e8116912554f27a62a51ad2aeca479a7f9e`. They compare complete BF16 and constrained-FP4 eager and graph training with all recorded states equal. They do not establish coverage of the final graph PR head.

## Known independent limitations

`qwen-mrope-cache-prototype-fp4-20261027-gpu3` enables FP4 parameter caching with the mRoPE prototype and offload disabled; it fails with an illegal CUDA memory access. This run does not establish a root cause. The graph feature retains its cache-disabled restriction.

`qwen-fp4-fused-tp2-trace-fixed-trace-20261027-gpu12` records a separate full-Qwen TP2 FP4 backward failure with a 4112-wide dimension, even with the gradient-buffer alignment fix; the cuBLAS operation is not the misaligned output-buffer operation. The `qwen-fp4-fused-tp2-*-highprecision-20261027-gpu12` runs use the temporary `37f34cc` merge and explicitly set `NVTE_BACKWARD_OVERRIDE=high_precision`: this means FP4 forward and high-precision backward, not default FP4 backward. Repeated and offloaded runs match the retained high-precision baseline at both ranks across loss, gradients, weights and full optimizer state.

The unmodified Qwen runner with ordinary unfused weight-gradient accumulation also fails in the FP4 backward GEMM on both TP2 ranks at `dfaa6338c65edab118ba415fcc14c73241f503a6`, with offload disabled (`qwen-fp4-normal-tp2-default-dfaa633-gpu12`). The accompanying high-precision-backward runs pass with offload disabled and enabled and match all recorded states at each rank. All three use RHT and stochastic rounding disabled; these TP2 runs do not establish behavior of the default FP4 recipe with RHT and stochastic rounding enabled.

These local runs do not replace upstream GPU CI or cover every model, recipe and topology.
