# GDN follow-up GPU validation

The retained manifests contain the actual commands, source revisions and device IDs; `ranks/` contains original stdout and stderr and `reports/` contains per-rank pytest outcomes. Run `python3 -S audit.py` from this directory to check checksums and the recorded comparisons. Paths in the commands refer to the test machine and must be adjusted on another machine. The Qwen runs use the complete language model without vision or MTP, initialized from local Qwen3.5-0.8B weights and JSONL data, which are not included here.

Environment: RTX PRO 6000 Blackwell, driver 580.126.09, PyTorch 2.11 with CUDA 13.0, Transformer Engine 2.20.2, FLA 0.5.1, Triton 3.6, FlashAttention 2.8.3 and causal-conv1d 1.6.1. Some runs shared GPUs with other processes. Timings from shared runs are not performance claims.

This GPU is SM120. [Transformer Engine 2.20.2 documents NVFP4 training support for SM100 and SM103](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html#supported-devices); the SM120 FP4 training results here are exploratory and require confirmation on a supported training GPU. BF16 results are not subject to that particular NVFP4 limitation.

## Independent regression fixes

* `mrope-graph-865d79d-native-gpu1`: commit `865d79dd37f304a4dffa23158e8df52735d43247`, one real GPU, native GPT/DDP and TE graph, one test passed. The test compares the loss and gradients for two different three-axis position inputs. `mrope-native-baseline-v3-20261027-gpu1` uses the same regression test against `aecc0e8116912554f27a62a51ad2aeca479a7f9e` and fails on the actual replay/eager loss mismatch. The baseline is not a `main` checkout; both commits contain the old graph-input behavior.
* `qwen-bf16-{eager,graph}-20261027-gpu1`: complete BF16 Qwen language-model training at `aecc0e8116912554f27a62a51ad2aeca479a7f9e`, with offloading disabled. Eager and graph inputs and states agree for the first three pre-capture steps; loss, gradients, weights and optimizer states diverge at the first replay and subsequent steps. This is a graph-training issue independently of offloading.
* `nvfp4-ddp-66b6bb6-unit-gpu12`: commit `66b6bb62a00bdccf8365096e6828215cffed4ce7`, two real GPUs, eight tests passed per rank. The FP4 tests perform real TE linear forward/backward and DDP reduction, checking FP32 and BF16 gradient buffers, repeated steps and ranks. RHT and stochastic rounding are disabled for this exact-repeat check. `alignment-real-te-baseline-final-gpu1` runs the same regression against `aecc0e8116912554f27a62a51ad2aeca479a7f9e`: both gradient types fail with `CUBLAS_STATUS_NOT_SUPPORTED`.

The `probes/` files are the regression tests used for the old-code controls. `alignment-tp2-{baseline,fixed}-verbose-20261027-gpu03` runs the same eight-layer native GDN2 TP2+SP optimizer training test with synthetic tokens on base `aecc0e8116912554f27a62a51ad2aeca479a7f9e`: both ranks fail before the fix and pass with the retained uncommitted prototype patch. The additional GDN2 TP2+SP integration run at temporary merge `37f34ccaf362ca7e42c0b2e5405455a1fe12993d` includes the earlier alignment commit `37e1ee1`; its production `megatron/` tree matches `66b6bb62a00bdccf8365096e6828215cffed4ce7`, but its tests and manifest are older. Neither prototype nor temporary merge is a test of the final commit.

## Complete Qwen NVFP4 training

`qwen-fp4-mrope-dfaa633-*-gpu0` runs the complete 24-layer Qwen3.5-0.8B language model with local checkpoint/data, sequence length 128 and four microbatches. At `dfaa6338c65edab118ba415fcc14c73241f503a6`, eager, two graph baselines and graph plus offload execute six steps each. RHT and stochastic rounding use the default enabled settings. All recorded input hashes, losses, gradient norms, gradient fingerprints, weights and full optimizer state match across the four arms; the offload arm records actual transfers. These are correctness runs, not benchmarks.

`qwen-fp8-{delayed,tensorwise}-*-dfaa633-gpu0` runs the same full language model and step configuration with the two FP8 recipes. For each recipe, eager, graph without offload and graph with offload match all six steps of inputs, loss, gradients, weights and full optimizer state. The enabled arms record actual transfers.

At `c5264b66197040f4309edfdb22dc9bbaf62189b1`, two-GPU GDN2 native and GDN/GDN2 graph suites pass 12 and 14 cases per rank, respectively. At `dfaa6338c65edab118ba415fcc14c73241f503a6`, the two-GPU low-precision graph suite passes 18 cases per rank. These suites cover TP2+SP and PP2 and have no skips.

At `c5264b66197040f4309edfdb22dc9bbaf62189b1`, the `graph-qwen-f0-correct-{eager,off,on}` runs compare full BF16 Qwen training at sequence length 2048 with four microbatches. Eager, graph without offload, and graph with offload match input, loss, gradients, weights and full optimizer state for six steps. Offload records actual transfers; these are correctness runs and not timing measurements.

The `graph-gdn2-f0-correct-{eager,off,on}` runs make the same comparison for an eight-block native GDN2 model at the same commit; all recorded states match for six steps and the offload arm records actual transfers.

The `qwen-bf16-graph-tp2-*-c5264b6-gpu12` runs use full Qwen BF16 training with TP2+SP, sequence 128, four microbatches and two shared GPUs. Eager, graph without offload and graph with offload match the input and complete training-state fingerprints at both ranks for all six steps. The offload run records transfers and checks pool ownership.

`qwen-bf16-stability-203-b6fa63-gpu3` runs the complete BF16 Qwen language model at `b6fa63ab8c2cb855f0368cd8455245c295cde234` for 203 optimizer steps with sequence 128, four microbatches, fraction 1 and norm recomputation. All logged losses and gradient norms are finite and the measured 200 post-warmup steps perform transfers. GPU allocated and reserved peaks stay at 13937.40 MiB and 14492 MiB respectively; after every step the pinned pool has zero buffers in use and retains 126 buffers totaling 31.92 MiB. The observer calls the real gradient finalization before recording pool state. This fixed-length run is a stability check, not a large-model performance benchmark.

`qwen-bf16-checkpoint-{continuous,save,resume}-b6fa63-gpu3` uses the same model and training settings at the same commit. Six continuous optimizer steps exactly match three steps followed by native Megatron checkpoint save, process restart and three resumed steps. Each step's loss, gradient, weight and full optimizer-state fingerprints match; all arms record transfers and check pinned-buffer ownership. The 9.9 GiB temporary checkpoint was removed after verification and is not included in this evidence; the command and manifests show how it was produced.

At `dfaa6338c65edab118ba415fcc14c73241f503a6`, the single-GPU scope/configuration/tool suite passes 111 cases, native low-precision training passes 54 cases and graph capture/replay passes 23 cases, without skips.

The `qwen-mrope-paired-*` runs use an earlier uncommitted mRoPE prototype on base `aecc0e8116912554f27a62a51ad2aeca479a7f9e`. They compare complete BF16 and constrained-FP4 eager and graph training with all recorded states equal. They do not establish coverage of the final graph PR head.

## Current-head BF16 training performance

`performance/` retains every arm and GPU telemetry. `python3 -S summarize_performance.py` recomputes `performance-summary.json` and verifies source identities, the tested comparisons, pinned-pool ownership and the target-GPU process samples. These runs use one RTX PRO 6000, BF16, four microbatches, fraction one and the default 1,048,576-element offload threshold. The eager Qwen runs use `b6fa63ab8c2cb855f0368cd8455245c295cde234` and sequence 4096; the graph runs use `c5264b66197040f4309edfdb22dc9bbaf62189b1` and sequence 2048. The different sequence lengths mean the eager and graph rows must not be compared to each other.

Each result compares fresh-process off/on arms with three warmup and ten measured optimizer steps. Graph capture and input preparation are outside the timed steps; graph capture times are retained separately. The model is the full Qwen language model without vision and MTP or the specified eight-block native GDN2 model. Peak values below are medians of the per-arm peaks; step changes are medians and ranges of paired percentages.

| Configuration | Allocated peak off → on (MiB) | Reserved peak off → on (MiB) | Retained pinned memory on (MiB) | Step-time change, median [range] |
| --- | --- | --- | --- | --- |
| Qwen eager, unfused | 26067.95 → 25115.95 | 28394 → 27382 | 1008 | +0.67% [−0.08%, +1.39%] |
| Qwen eager, fused pre-GDR | 25770.95 → 24819.58 | 28914 → 27902 | 1008 | −1.40% [−2.65%, +1.53%] |
| Qwen TE graph | 16137.82 → 16137.82 | 22576 → 22122 | 504 | +24.75% [+12.71%, +26.55%] |
| GDN2 TE graph, eight blocks | 3236.12 → 3237.12 | 5212 → 4964 | 432 | +97.83% [+96.40%, +98.97%] |

The eager Qwen unfused summary uses pairs 2–4: pairs 0–1 overlapped the two-GPU TP2 correctness run, so they remain in the raw records but are excluded from the summary. Qwen graph uses five pairs because the initial three showed substantial variation; other rows use three. No foreign compute process was observed on the measured GPU, but the other GPUs were shared and CPU/PCIe isolation is not established. The timing range therefore matters; the fused eager result does not establish a speedup. The graph runs reduced reserved memory but did not reduce the measured allocated peak.

All timed off/on runs check losses and gradient norms. Graph runs also compare input hashes exactly; eager unfused output logs match exactly. For fused eager, the displayed gradient norms have four decimal places, so the check allows `1e-4` absolute rounding in addition to `1e-5` relative tolerance. Full six-step graph state comparisons are recorded separately above; these timed runs are not full gradient/optimizer fingerprint checks.

The initial driver stopped after completing fused pair 0 because it attempted to parse a printed `tensor(...)` norm as a plain float. `run_current_performance_resume.py` retains the completed arms, parses the printed value and continues; `run_current_performance_followup.py` adds the two Qwen graph pairs. The original driver console output was not retained as a file. Each arm's original rank logs, telemetry and manifest are retained.

## Known independent limitations

`qwen-mrope-cache-prototype-fp4-20261027-gpu3` enables FP4 parameter caching with the mRoPE prototype and offload disabled; it fails with an illegal CUDA memory access. This run does not establish a root cause. The graph feature retains its cache-disabled restriction.

`qwen-fp4-fused-tp2-trace-fixed-trace-20261027-gpu12` records a separate full-Qwen TP2 FP4 backward failure with a 4112-wide dimension, even with the gradient-buffer alignment fix; the cuBLAS operation is not the misaligned output-buffer operation. The `qwen-fp4-fused-tp2-*-highprecision-20261027-gpu12` runs use the temporary `37f34cc` merge and explicitly set `NVTE_BACKWARD_OVERRIDE=high_precision`: this means FP4 forward and high-precision backward, not default FP4 backward. Repeated and offloaded runs match the retained high-precision baseline at both ranks across loss, gradients, weights and full optimizer state.

The unmodified Qwen runner with ordinary unfused weight-gradient accumulation also fails in the FP4 backward GEMM on both TP2 ranks at `dfaa6338c65edab118ba415fcc14c73241f503a6`, with offload disabled (`qwen-fp4-normal-tp2-default-dfaa633-gpu12`). The accompanying high-precision-backward runs pass with offload disabled and enabled and match all recorded states at each rank. All three use RHT and stochastic rounding disabled; these TP2 runs do not establish behavior of the default FP4 recipe with RHT and stochastic rounding enabled.

These local runs do not replace upstream GPU CI or cover every model, recipe and topology.
