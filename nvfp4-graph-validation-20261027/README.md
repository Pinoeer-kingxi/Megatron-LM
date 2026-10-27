# NVFP4 GDN/GDN2 graph qualification

Source: **`7d52248294fd759dd69b83c68ec6d1c0b36f2836`**. This extends #8543 at `1aa1151a213c13de1faab0930c5e6bc91662c042`, based on #8308 at `3b6f5e66f64bd381893e0ea814511031c9b5e4af`.

## Change and boundary

TE graph capture previously forced `fp8_weight_caching=True`, including when `TransformerConfig.disable_parameter_transpose_cache=True`. The native TE linear wrappers already honor this configuration field, but TE's graph capture overrides their microbatch flags when graph weight caching is enabled. The helper now passes the existing configuration through. This is one production-line change; no new cache mechanism, numerical kernel or transfer scheduling is introduced.

NVFP4 graph offloading requires `TransformerConfig.disable_parameter_transpose_cache=True` through programmatic configuration. Cached NVFP4 graph offloading remains rejected. The constrained SM120 recipe also disables RHT and stochastic rounding before Python starts. Projections/MLPs use NVFP4; recurrence and offloaded tensors remain BF16/FP32.

Disabling caching re-quantizes weights on every microbatch. This report establishes correctness, not a speedup or memory saving. The underlying failure of cached NVFP4 graphs is not repaired here or attributed to a particular TE kernel. Default BF16/FP8 settings remain unchanged.

## Localization and unsuccessful controls

The unrestricted prototype `18fe09f4743ee12ccb972679da7b25fdc4b19068` reproduces the complete GDN2 baseline NaN on an otherwise idle physical GPU 0, with offloading disabled. The earlier GDN illegal-memory-access and GDN2 NaN failures remain in the [previous report](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/d2e4ab3a5ea6e08166449e6bac86a9fd9bf223d3/low-precision-extension-20261026).

Fresh-process controls use the same model, seeds, stack and GPU. The diagnostic wrapper changes only the named graph option or existing configuration field; it delegates all training and numerical work to the actual repository tests.

| Control at the prototype | Outcome |
| --- | --- |
| Original cached graph | GDN2 baseline fails, NaN gradient norm on first replay step |
| Disable graph input/output reuse | Still fails |
| Clone returned parameter gradients | Still fails |
| Set only the existing config cache-disable field | Still fails: helper forces graph weight caching |
| Disable only the graph API cache flag | Existing offload-on/off comparison passes; not accepted as proof of fresh weights |
| Disable both cache paths | Existing GDN and GDN2 comparisons pass; followed by the stronger final-source tests below |

The optional trace observes finite outputs in all eight layers on the first replay microbatch. On the next microbatch, layer 1 receives finite inputs and returns nonfinite outputs when cached weight updates are skipped. This is a diagnostic observation, not a qualification run.

Standalone TE microbatch probes check finite values only. Such checks are insufficient to establish numerical equivalence or correct weight updates. They are not counted as acceptance tests. Likewise, a passing offload-on/off comparison alone could miss a shared stale-weight error. The final NVFP4 graph tests therefore add a separate eager-training reference.

## Final-source GPU results

All selected cases run at **`7d52248294fd759dd69b83c68ec6d1c0b36f2836`**, with clean trees before and after execution. No selected case is skipped.

| Suite | Actual result |
| --- | --- |
| Scope/configuration/Qwen-tool regressions, physical GPU 1 | 109 passed |
| Native low-precision GDN/GDN2 training, physical GPU 1 | 54 passed; 144 unrelated cases deselected |
| Single-rank BF16/FP8/NVFP4 graphs, physical GPU 0 | 21 passed; 42 other-topology cases deselected |
| Two-rank GDN2 FP8/NVFP4 graphs, physical GPUs 0/1 | 16 passed per rank; 47 other cases deselected per rank |

The graph suite has seven single-rank NVFP4 cases; the two-rank matrix has eight NVFP4 cases per rank across TP2+SP and PP2. Cases cover attention/whole-layer capture, fractions 0/.5/1, norm recomputation, delayed offload and the existing native GDN fusion. GDN2 does not use that GDN-only fusion.

Every NVFP4 graph case builds three fresh models: six continuous eager optimizer steps; three native plus three replay steps without offloading; and three native plus three replay steps with offloading. Inputs change each step. Tests compare every recorded loss, gradient, gradient norm, weight and complete Adam/master state. Unfused checks are exact; fused GDN retains the existing `rtol=1e-5, atol=1e-8` tolerance for atomic gate-gradient variation. Buffer ownership is checked before native schedule resets, and actual offload selection is checked against the requested fraction. FP8/BF16 graph cases retain their existing offload-on/off comparisons.

These are real reduced-size eight-block mixed GDN/attention decoder training paths with embeddings, MLPs, vocabulary loss, native DDP and BF16 Adam/master parameters; sequence 128 and four microbatches. Tokens are generated by the fixed-seed tests. No private weights or dataset are required. Pretrained GDN2, convergence, larger topologies, CP combinations, performance and upstream GPU CI are not qualified. The PR remains a draft pending its prerequisites and integration with current `main`.

## Runtime and reproduction

RTX PRO 6000 Blackwell Server Edition (SM120, 96 GB), driver 580.126.09; Python 3.12.4, PyTorch 2.11.0+cu130, Transformer Engine 2.20.2, FLA 0.5.1, Triton 3.6.0, FlashAttention 2.8.3 rebuilt for SM120 and causal-conv1d 1.6.1. TE graph and Linear source hashes match the official 2.20.2 files. Neither TE nor numerical kernels are patched for this qualification.

The final tests use physical GPUs 0/1 without peer workloads on those devices. Peer processes on GPUs 2/3 are left running. Each job retains the exact command, GPU UUID, source revision, exit status, original rank logs and structured outcomes. Runtime wrappers set `NVTE_CPU_OFFLOAD_V1=1`, `NVTE_NVFP4_DISABLE_RHT=1` and `NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1` before startup. Triton and TorchInductor caches are persistent and separate per rank. Host paths in the reproduction scripts must be adapted to the reader's environment.

Run `python audit.py` to check recorded outcomes, executed revisions and SHA256 integrity. The offline audit does not rerun training or recompute tensor equality; those comparisons execute on real GPU tensors inside the referenced tests. The earlier report retains its own executed revision and results and is not relabelled as this head.

Black, isort, Ruff, project Pylint, determinism coverage and whitespace checks pass on this head. Optional mypy checks the same seven Python files on both revisions: 20 prerequisite diagnostics and 19 current diagnostics, with no new diagnostics. This wider file set includes `cuda_graphs.py`; its counts are not comparable to the earlier six-file report.
