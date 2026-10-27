# Qwen BF16 TE graph memory diagnostic

Source revision: `c5264b66197040f4309edfdb22dc9bbaf62189b1` (#8308 with #8555). The source checkout was clean before and after the runs. The model was the 24-layer Qwen3.5-0.8B language model initialized from local pretrained weights and a local JSONL dataset; vision and MTP were not included. One RTX PRO 6000 (SM120, GPU 0), driver 580.126.09, sequence length 2048, four microbatches, BF16, whole-layer TE graphs, no norm recomputation and the default offload threshold were used. Other GPUs on the host were in use; host and PCIe isolation were not established.

`benchmark_memory.py` records the allocator history for measured replay step 6 and prints allocation at start, after the forward/backward schedule, and after the optimizer. The corresponding `results/off.*`, `results/on.*` and compressed memory snapshots are from fresh processes with offload disabled and enabled (fraction 1). Instrumentation changes timings; do not use those timings as performance results. Run `python3 -S analyze_snapshots.py` to reproduce the allocation summary. The snapshots contain Python and C++ stack traces and can be loaded with PyTorch's memory visualizer; pickle files should only be opened if their source is trusted.

Both arms have the same tracked replay-step incremental peak, 2918.56 MiB. At that peak, the output projection and FP32 cross-entropy conversion account for 970 MiB and 1940 MiB respectively. Step-start allocated memory is 13219.25 MiB in both arms and total step peak is 16137.82 MiB in both arms (difference 512 bytes). At the snapshot after the step, allocator pool `(0, 1)`, the graph pool, reserves 4132 MiB without offload and 3690 MiB with offload; its allocated count is 46.11 MiB in both arms. This is one configuration: a smaller graph-pool reserve does not imply an equal decrease in PyTorch's allocated peak. Total reserved peaks are 22576 and 22122 MiB.

`benchmark_fraction.py` changes only the experiment's chosen offload fraction through `GDN_EXPERIMENT_FRACTION`; it does not modify Megatron. `results/fraction-{off,0.25,0.5,on}.*` contain one fresh-process run per arm, three warmup and ten measured replay steps. Median times in that order were 325.32, 353.20, 377.21 and 366.22 ms; reserved peaks were 22576, 22584, 22594 and 22122 MiB. All had the same allocated peak to within 512 bytes. Input hashes, losses and gradient norms match across all 16 steps. These are single, non-interleaved runs without host isolation, so the timing ordering and differences are exploratory, not a statistical performance conclusion; full parameter and optimizer state was not compared in this fraction experiment. Pinned buffers are populated during warmup, even for partial fractions.

The scripts used the same isolated runtime, source and resources as the [current-head performance evidence](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/02890e97b1273d9a8695f76aaa208730054fe415/gdn-open-issues-20261027). With those prerequisites available, each arm was launched as follows (add `--offload` for an enabled arm; use `benchmark_memory.py --warmup 3 --iterations 1` for the snapshots or `benchmark_fraction.py` for the fraction experiment):

```bash
export B=/home/lwx/.cache/megatron-gdn-20261026
export GDN_SOURCE="$B/graph-with-mrope-c5264b6"
export GDN_INDUCTOR_CACHE_DIR="$B/current-performance-20261027/cache/graph/qwen/inductor"
export GDN_TRITON_CACHE_DIR="$B/current-performance-20261027/cache/graph/qwen/triton"
/usr/bin/python3 "$B/precision_graph_container_gpu0.py" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=1 \
  "$B/rank_env_precision_graph.py" "$B/verify_qwen_source.py" \
  "$B/benchmark_memory_20261027.py" --model qwen \
  --weights "$B/models/Qwen3.5-0.8B" --data "$B/data/ks-item-sft.jsonl" \
  --warmup 3 --iterations 1 --output "$B/graph-memory-diagnostic-20261027/off.json"
```

For the partial-fraction runs, set `GDN_EXPERIMENT_FRACTION=0.25` or `0.5` and add `--offload`. `benchmark_delay.py` uses the existing delayed-offload configuration; its results, if present, are separately labeled. Dataset and model weights are not included.

The delayed-offload check compares graph offload disabled (`delay-check-off`) against fraction-1 offload with the existing `delay_offload_until_cuda_graph=True` (`delay-check-on`). For all six steps, input hashes, loss, gradient norm, gradients, weights and complete optimizer-state fingerprints match. A separate ten-step timed run (`delay-on`) had a median of 416.93 ms and the same 16137.82 MiB allocated and 22122 MiB reserved peaks. It was not paired or isolated and is insufficient to establish a timing effect. Delayed and default-offload timings both exclude state fingerprinting; correctness runs are separate. No default or production code was changed.

The same instrumented step was run on the eight-block GDN2 test model, initialized from random weights and synthetic tokens (`gdn2-off` and `gdn2-on`). Its 4096-token vocabulary makes the output projection and FP32 loss buffers 16 and 32 MiB, respectively. The tracked incremental peak is 56.06 MiB in both arms. The graph pool reserves 1888 MiB off and 1604 MiB on, while its allocated count changes from 49.04 to 50.04 MiB; the total allocated peak increases by 1 MiB. These snapshots provide a second configuration, not a general prediction for every model or schedule.
