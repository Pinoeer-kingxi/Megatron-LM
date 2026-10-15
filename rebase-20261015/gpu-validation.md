# Pending GPU validation for the split PRs

These commands are a plan, not passing GPU evidence. No GPU was available during
the CPU preparation. The old source checkouts have been rebased; use the exact source revisions in `source-after.json`; record clean
Git status before and after each job, the complete command, GPU UUIDs and package
versions, per-rank stdout/stderr, result JSONs and launcher exit codes.

Run in the validated public-package environment, exposing only reserved idle GPU
UUIDs through `CUDA_VISIBLE_DEVICES`. Set `NCCL_MAX_NCHANNELS=1`,
`NCCL_NVLS_ENABLE=0` and `NVTE_CPU_OFFLOAD_V1=1`, matching the earlier isolated
runs. Use the optional causal-conv1d/FlashAttention overlays. For Qwen use Bridge
`c860f8a5bc5fddd78690f32baa0b8696774308b4` with `bridge-compatibility.patch`
applied and the original Transformers/tokenizer dependency overlay. The patch
imports the same `te_checkpoint` function from its canonical extension module;
it does not modify Megatron or the checkpoint algorithm.

## Shared manager and schedules

Checkout `af9569270f476dc201484296315a9c30dcaa3feb`. Run the generic GPU regressions
on four ranks, then run the existing GPT/MoE/CUDA-graph regression cases separately:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --log-dir /outputs/shared-ranks --redirects 3 --tee 3 \
  /evidence/helpers/run_reduced.py -q -x --confcutdir=tests/unit_tests/pipeline_parallel \
  tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py \
  -k 'not test_gpt_fine_grained_activation_offloading_correctness_and_memory and not test_fine_grained_activation_offload_with_ep_a2a_overlap_compatibility and not test_fine_grained_activation_offloading_with_cuda_graph'
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --log-dir /outputs/existing-model-ranks --redirects 3 --tee 3 \
  /evidence/helpers/run_reduced.py -q -x --confcutdir=tests/unit_tests/pipeline_parallel \
  tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py \
  -k 'test_gpt_fine_grained_activation_offloading_correctness_and_memory or test_fine_grained_activation_offload_with_ep_a2a_overlap_compatibility or test_fine_grained_activation_offloading_with_cuda_graph'
```

The first selection includes fraction zero/one with growing saves, both delayed
copy races and existing storage/group tests. Report every skip in the model
selection explicitly; unsupported hardware/topologies are not passing cases.
The five CPU routing tests are already recorded separately.

## Gate backward: replay, numerical accuracy and performance

Checkout `9d54b8729c9277f1e032110c5e1b149e9654bcc5`. Expose one idle GPU for this
stage; the complete integration uses four ranks later.

```bash
python -m torch.distributed.run --standalone --nproc-per-node=1 \
  --log-dir /outputs/gate-ranks --redirects 3 --tee 3 \
  -m pytest -q -x --confcutdir=tests/unit_tests/determinism/kernels \
  tests/unit_tests/determinism/kernels/test_gdn_gate_backward.py
```

The gate source in upstream baseline `9131593673d037d360d252904f9198d36bc0c246`
uses atomic accumulation. Make a separate clean checkout at `/baseline` and use
the clean candidate checkout at `/candidate`. Both paths below point to complete
production source files. Run each shape in a fresh process so the old autotuner
cannot reuse a tile selected for a different batch. Repeat each process three
times; each process collects three paired samples after warmup:

```bash
for batch in 1 4; do
  for sequence in 2048 4096; do
    for heads in 16 32; do
      for repeat in 0 1 2; do
        python /evidence/benchmark_gate.py \
          --baseline /baseline/megatron/core/fusions/fused_pre_gated_delta_rule.py \
          --candidate /candidate/megatron/core/fusions/fused_pre_gated_delta_rule.py \
          --batch-size "$batch" --seq-length "$sequence" --value-heads "$heads" \
          --output "/outputs/gate-${batch}-${sequence}-${heads}-${repeat}.json"
      done
    done
  done
done
```

This measures BF16 gate backward without offloading, with a preallocated
projection-gradient output as in the production caller. Compilation and Torch
numerical-reference checks precede timing. Pair order alternates; timings include
allocation, launch and the final reduction in the full wrapper. Report all raw
samples and median/range of paired runtime changes and extra peak allocations.
It is not a whole-model throughput comparison. The 16-case replay separately
covers FP32 parameters, tails, strided gradients and stream contention.

## Final combined GDN integration

Checkout the feature revision in `source-after.json`, including both prerequisites.
The native reduced-model acceptance condition is 96 passed and zero skipped on
each of four ranks, with launcher exit 0:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --log-dir /outputs/native-ranks --redirects 3 --tee 3 \
  /evidence/helpers/run_reduced.py -q -x --confcutdir=tests/unit_tests/ssm \
  tests/unit_tests/ssm/test_gdn_offload_training.py
```

For complete pretrained Qwen, mount the feature checkout at `/workspace`, keep
its optional Bridge environment on `PYTHONPATH`, and run eight fresh processes:
fusion off/on × norm recomputation off/on × disabled/full offload. For each arm:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --log-dir /outputs/qwen-arm-ranks --redirects 3 --tee 3 \
  /evidence/helpers/qwen_pool_lifetime_check.py \
  --weights /models/Qwen3.5-0.8B --data /data/messages.jsonl \
  --tp 2 --sp --pp 2 --seq-length 128 --min-offloaded-tensor-size 1024 \
  --warmup 1 --iterations 2 --check-state --output /outputs/qwen-arm.json
```

Use distinct result/log paths; add `--fused-pre-gdr`, `--recompute-norm` and/or
`--fraction 1` for the corresponding arm. The original checkpoint and private
dataset hashes are retained in the historical evidence; they are not uploaded.
Compare every rank's three complete states against the disabled process at the
same fusion and norm setting: losses, gradient norms, all gradient/updated-weight
fingerprints and complete Adam/master-weight state. The eight processes produce
32 rank files and 96 recorded rank-step states. Require zero pinned-buffer use
before every reset and positive transfers in enabled arms. These runs are not
timed performance samples. Retain the known, unqualified pretrained-CP result;
this plan does not rerun the unsupported A6000 attention configuration.
