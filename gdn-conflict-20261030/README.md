# Conflict-resolution validation

The source revisions below merge NVIDIA `main` at `581248531984920e92f076946ad4629d359442a4` without rewriting the original PR commits:

- #8555: `5df3661919c5119830d089fb6f8aa0cc5e0d14d8`
- #8308: `03b5ef353a1e657a89247f1ae0f67d05d619166f`
- #8543: `a04f1e65627ec7a5323bdd56d165a746b59f1c6e`

`mrope/` is the actual per-rank output of the #8555 GPU test, run in a private runtime on one RTX PRO 6000 (SM120), using TE CUDA graphs and distributed GPT/DDP forward and backward. The single test `tests/unit_tests/transformer/test_cuda_graphs.py::TestTECudaGraphHelper::test_mrope_replay_uses_current_positions` passed. `mrope-source.sha256` records all regular tracked source files; the copied test tree was checked against it before testing. The launcher was:

```bash
B=/home/lwx/.cache/megatron-gdn-20261026
R=$B/conflict-20261030
export GDN_SOURCE=$B/conflict-mrope-5df3661
export GDN_INDUCTOR_CACHE_DIR=$R/cache/mrope/inductor
export GDN_TRITON_CACHE_DIR=$R/cache/mrope/triton
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
/usr/bin/python3 "$B/precision_graph_container_gpu3_default.py" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=1 \
  --log-dir "$R/mrope/ranks" --redirects 3 \
  "$B/rank_env_precision_graph.py" "$B/rank_distributed_pytest.py" "$R/mrope/reports" \
  -q -s -p no:cacheprovider \
  "$GDN_SOURCE/tests/unit_tests/transformer/test_cuda_graphs.py::TestTECudaGraphHelper::test_mrope_replay_uses_current_positions"
```

For #8543, `precision-cpu.log.gz` records 19 configuration tests passing on the revision above; it is not a GPU training result. For transparency, the two local A6000 attempts at the #8555 test in `mrope-gpu*.log.gz` failed during Transformer Engine NVRTC compilation because the local runtime and CUDA headers do not match. The successful GPU test was run separately in the isolated environment. The test framework attempted to download optional `/opt/data` and reported permission denied; this test does not require that data.

These results do not replace upstream GPU CI, the earlier Qwen/low-precision training results on other revisions, or NVFP4 validation on a supported training GPU.

## Qwen BF16 at the merged #8308 source

At `03b5ef353a1e657a89247f1ae0f67d05d619166f`, one RTX PRO 6000 ran three fresh processes using the complete Qwen3.5-0.8B language model, local pretrained weights and JSONL data, BF16, sequence 128, four microbatches and offload threshold 1024 elements. Each process ran six optimizer steps: three warmup steps and three after graph capture for the graph arms. Eager, graph without offload and graph with offload match input hashes, loss, gradients, weights and full optimizer states for every step. The graph arms capture 24 layers; the offload arm records selected transfer bytes and zero in-use pool buffers after each step. Vision and MTP are excluded. Run `python3 -S audit.py` after verifying `SHA256SUMS` to check these results.

Each arm used the following command (add `--no-graph` for eager, no flag for graph without offload, and `--offload` for graph with offload):

```bash
B=/home/lwx/.cache/megatron-gdn-20261026
R=$B/conflict-20261030
export GDN_SOURCE=$B/conflict-graph-03b5ef3
export GDN_INDUCTOR_CACHE_DIR=$R/cache/qwen/inductor
export GDN_TRITON_CACHE_DIR=$R/cache/qwen/triton
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
/usr/bin/python3 "$B/precision_graph_container_gpu3_default.py" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=1 \
  --log-dir "$R/qwen-ARM/ranks" --redirects 3 \
  "$B/rank_env_precision_graph.py" "$B/benchmark_current_20261027.py" \
  --model qwen --weights "$B/models/Qwen3.5-0.8B" --data "$B/data/ks-item-sft.jsonl" \
  --seq-length 128 --threshold 1024 --check-state --warmup 1 --iterations 2 \
  --output "$R/qwen-ARM/result.json"
```

A separate GPU `-k tp1` graph test matrix on #8308 was manually interrupted after four GDN cases completed because the first GDN2 case was compiling for several minutes. The interrupted run is preserved under `interrupted-graph/`; it has no final pytest report and is not treated as a passing matrix. Its isolated GDN2 follow-up is reported separately if completed. These are correctness tests, not performance measurements.

The isolated GDN2 case on #8308 passed on RTX PRO 6000 GPU 2, shared with an existing process. It runs offload-disabled and offload-enabled training, TE graph capture and replay, and compares the training states. The test took 260.91 seconds, including cold Triton compilation. Diagnostic traceback dumps during the run show progress through FLA Triton compilation and autotuning; the test completed successfully. The exact launch was:

```bash
B=/home/lwx/.cache/megatron-gdn-20261026
R=$B/conflict-20261030
export GDN_SOURCE=$B/conflict-graph-03b5ef3
export GDN_INDUCTOR_CACHE_DIR=$R/cache/gdn2-isolated/inductor
export GDN_TRITON_CACHE_DIR=$R/cache/gdn2-isolated/triton
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
timeout 360 /usr/bin/python3 "$B/precision_graph_container_gpu2.py" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=1 \
  --log-dir "$R/gdn2-isolated/ranks" --redirects 3 \
  "$B/rank_env_precision_graph.py" "$R/watch_run.py" "$B/rank_distributed_pytest.py" \
  "$R/gdn2-isolated/reports" -q -s -x -p no:cacheprovider \
  --confcutdir=tests/unit_tests/pipeline_parallel \
  'tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py::test_gdn_offload_with_te_graphs[gdn2-False-False-True-False-1.0-tp1]'
```

## FP8 GPU cases at the merged #8543 source

At `a04f1e65627ec7a5323bdd56d165a746b59f1c6e`, one RTX PRO 6000 passed two real decoder-training and TE graph cases: GDN with FP8 delayed scaling, and GDN2 with FP8 tensorwise scaling. Each test compares offload disabled and enabled, including losses and complete optimizer states. No cases were skipped. The launcher used:

```bash
B=/home/lwx/.cache/megatron-gdn-20261026
R=$B/conflict-20261030
export GDN_SOURCE=$B/conflict-precision-a04f1e6
export GDN_INDUCTOR_CACHE_DIR=$R/cache/precision/inductor
export GDN_TRITON_CACHE_DIR=$R/cache/precision/triton
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
timeout 480 /usr/bin/python3 "$B/precision_graph_container_gpu3_constrained.py" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=1 \
  --log-dir "$R/precision-fp8/ranks" --redirects 3 \
  "$B/rank_env_precision_graph.py" "$R/watch_run.py" "$B/rank_distributed_pytest.py" \
  "$R/precision-fp8/reports" -q -s -x -p no:cacheprovider \
  --confcutdir=tests/unit_tests/pipeline_parallel \
  'tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py::test_gdn_offload_with_te_graphs[gdn-False-False-False-False-1.0-fp8-delayed-False-tp1]' \
  'tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py::test_gdn_offload_with_te_graphs[gdn2-False-False-False-False-1.0-fp8-tensorwise-False-tp1]'
```

This FP8 test does not validate NVFP4 on a supported training GPU.

The #8543 CPU-only configuration check used the merged checkout and ran:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 LD_LIBRARY_PATH=/path/to/compatible/libstdc \
  python -m pytest -q tests/unit_tests/ssm/test_gated_delta_net_offloading.py \
  -k 'test_reject_unsupported_gdn_offload or test_gdn_offload_config or test_gdn_offload_te_graph_config'
```

## Qwen FP8 at the merged #8543 source

At `a04f1e65627ec7a5323bdd56d165a746b59f1c6e`, the complete Qwen3.5-0.8B language model (excluding vision and MTP) ran on a shared RTX PRO 6000 with FP8 delayed scaling, sequence length 128, four microbatches and an offload threshold of 1024 elements. Each of the eager, graph without offload and graph with offload arms used a fresh process and ran six optimizer steps. The audit compares input hashes, loss, gradient norm, all gradient and parameter fingerprints, and full optimizer-state fingerprints, and checks transfers and pinned pool release. This is a correctness check, not a performance comparison.

Each arm used the following command (add `--no-graph` for eager, no flag for graph without offload, and `--offload` for graph with offload):

```bash
B=/home/lwx/.cache/megatron-gdn-20261026
R=$B/conflict-20261030
export GDN_SOURCE=$B/conflict-precision-a04f1e6
export GDN_INDUCTOR_CACHE_DIR=$R/cache/fp8-qwen/inductor
export GDN_TRITON_CACHE_DIR=$R/cache/fp8-qwen/triton
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
/usr/bin/python3 "$B/precision_graph_container_gpu2.py" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=1 \
  --log-dir "$R/fp8-qwen-ARM/ranks" --redirects 3 \
  "$B/rank_env_precision_graph.py" "$B/benchmark_fp8_qwen.py" \
  --model qwen --precision fp8-delayed --weights "$B/models/Qwen3.5-0.8B" \
  --data "$B/data/ks-item-sft.jsonl" --seq-length 128 --threshold 1024 \
  --check-state --warmup 1 --iterations 2 --output "$R/fp8-qwen-ARM/result.json"
```
