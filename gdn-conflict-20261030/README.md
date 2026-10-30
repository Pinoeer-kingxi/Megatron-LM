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
