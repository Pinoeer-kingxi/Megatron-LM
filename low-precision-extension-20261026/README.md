# TE low precision with BF16 GDN recurrence offloading

Final source: **`1aa1151a213c13de1faab0930c5e6bc91662c042`**, based on #8308 at `3b6f5e66f64bd381893e0ea814511031c9b5e4af`. The incremental change covers configuration validation, existing training tests, the Qwen validation tool and documentation. It does not change GDN/GDN2 kernels, the offload manager, transfer scheduling or graph implementation.

TE projections/MLPs use FP8 delayed, FP8 tensorwise or NVFP4; the FLA recurrence and offloaded tensors remain BF16/FP32. FP8 supports the existing TE attention/whole-layer graphs. FP4 with graph offloading remains rejected. All NVFP4 tests use the constrained recipe with `NVTE_NVFP4_DISABLE_RHT=1` and `NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1` set before Python startup.

## Final-source execution

NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120, 96 GB; PyTorch 2.11.0+cu130, TE 2.20.2, FLA 0.5.1, Triton 3.6.0. Each run records the exact command, GPU UUID, before/after revision, clean-tree checks, exit code and original rank logs. Single-rank tests use physical GPU 3 with another workload present. The two-rank final run uses physical GPUs 0 and 1 after checking that both are free. These are correctness checks; they do not establish performance or memory savings.

| Tests | Actual outcome |
| --- | --- |
| Scope, configuration and Qwen-tool file | 108 passed |
| Native low-precision decoder/DDP/Adam training | 54 passed; 144 unrelated cases deselected |
| Single-rank TE capture/replay | 14 passed; 28 other-topology cases deselected |
| Two-rank GDN2 FP8 TP2+SP / PP2 capture/replay | 8 passed on each rank |

No selected final-source test is skipped. Native cases span GDN unfused, GDN fused and GDN2; three precision recipes; fractions 0/.5/1; and output-norm recomputation off/on. Graph cases cover attention/whole-layer capture, delayed offload, output-norm recomputation, native GDN fusion and fraction boundaries. The two-rank matrix covers both delayed and tensorwise FP8, including fraction 0/.5/1.

The complete reduced-size decoder has eight mixed GDN/attention blocks, embeddings, MLPs, vocabulary loss, native DDP and BF16 Adam/master parameters, sequence 128 and four microbatches. Eager comparisons run three optimizer steps. Graph comparisons run three native steps, capture, then run three optimizer steps using actual replay with changing token inputs. The executed tests compare losses, accumulated gradients, updated weights, gradient norms and complete Adam/master states; they also check pinned-buffer ownership before counters reset. Unfused comparisons require exact equality. Fused GDN retains the existing `rtol=1e-5, atol=1e-8` check for native atomic gate-gradient variability. Repeated-microbatch equality remains checked except for delayed FP8, whose scaling history changes between microbatches.

These are real reduced-model GPU training paths with synthetic tokens. They do not qualify pretrained GDN2, convergence, CP combinations, larger topologies, full NVFP4, or performance. No new pretrained Qwen graph training result or upstream GPU CI result is claimed.

## Failed qualifications retained

The unrestricted prototype at **`18fe09f4743ee12ccb972679da7b25fdc4b19068`** attempted NVFP4 graphs. Fresh-process complete-model baselines fail **with offloading disabled**:

- GDN encounters a CUDA illegal-memory-access error after capture.
- GDN2 produces a NaN gradient norm at step 3, its first replay step.

The diagnostic runner prints the original exception before graph cleanup can mask it. It still delegates cleanup to the original implementation. These failures are not counted as passes and do not establish an offload regression. Stock TE Linear graph probes at output widths 2592 and 4096 each complete three replay/SGD steps with finite results; the whole-model failure is therefore not attributed to generic device or TE graph support. The complete-model NVFP4 graph cause remains unresolved, and the final feature keeps the combination disabled with configuration tests.

A separate two-rank diagnostic at `057b70fd64735060a73d2a51bcb0a24677688538` stopped making progress when a parameterized process switched from TP2+SP to PP2. It was terminated by the author, and the termination record is retained. TE delayed-scaling global buffers retain amax tensors and process groups across module lifetimes. The final tests follow the project's existing reset practice: clear TE recipe state before fresh modules are built and before their process groups are destroyed. This is test isolation, not a production training or offload-manager fix. The final combined matrix exercises that topology transition.

That diagnostic used GPUs 0/3 on a shared host; the final combined matrix uses free GPUs 0/1. They are not a hardware-matched A/B experiment, so the earlier stall is not claimed as a confirmed production deadlock or attributed solely to the state cleanup.

Earlier prototype successes and failures retain their executed revisions. They are not relabelled as final-head results. The separate historical Qwen/A6000 evidence remains unchanged.

## Reproduction and audit

The final tests require no private model weights or datasets. The repository tests generate their fixed-seed token inputs. Commands use `torch.distributed.run`; both Triton and TorchInductor caches have separate persistent directories per rank. Runtime wrappers and the selected commands are under `scripts` and each job's `manifest.json`. Paths refer to the validation host and must be adapted to the reader's environment. Supply compatible dependencies and the recorded recipe flags. No kernels or tiles are overridden.

The standard-library audit checks file integrity, exact source records and actual per-rank outcomes, while retaining unsuccessful qualifications:

```bash
python audit.py
```

Numerical state comparisons execute on real tensors inside the referenced tests. The offline audit checks their recorded outcomes; it does not rerun training or recompute tensor equality from saved values. Static-check commands/output and the comparison against the prerequisite's optional mypy diagnostics are retained separately.
