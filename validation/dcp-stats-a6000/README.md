# DCP schedule-statistics acceptance on A6000

This is an evidence-only integration branch. The metrics PR targets `dev` and contains no attention-backend compatibility fix or benchmark framework. `ORIGIN.json` identifies the upstream baseline, metrics code, validation integration and checkpoint inputs. `correctness.json` records the production source hashes checked by every accepted training pair.

The local eager padding prerequisite follows the rule in NVIDIA/Megatron-LM#7417 and additionally preserves conservative metadata in graph mode. `compatibility.patch` contains the production-only difference from the metrics candidate. No duplicate compatibility PR was opened. The native TE 2.20.2 + FA2 2.8.3 combination runs these aligned BF16 THD workloads on SM86 after that prerequisite; stock dev failures are retained in `backend-debug/` and `fused-debug-2/`. This does not qualify internally padded THD attention on Ampere.

## Results

- **Correctness:** all three off/on pairs restore one fixed topology-specific checkpoint, RNG state and seed 1234. Five real Adam updates per arm match bitwise in schedules, microbatch counts, losses, gradients and updated parameters. See `final-2-*`, `final-tp-*` and `final-pp-*`.
- **TP/PP ownership:** every process receives the same global summary. Only one stdout and TensorBoard scalar per metric is emitted per trained step. Interleaved evaluation does not contribute.
- **VPP:** PP2×CP2/VPP2 really splits 3 microbatches into 4. Logical CP1/CP2 sample counts change from 7/1 to 3/5, with eight logical samples total. DP×CP has two ranks in all three topologies, including world size four. Estimated attention balance is 0.92.
- **Tests:** 84 focused tests per rank on the standalone metrics branch and 49 scheduling tests per rank on the integration, each with four CUDA/NCCL processes. Existing eight-rank topology cases and four unrelated model-builder optional-import failures are excluded by the recorded filter. Black, isort, Ruff and core Pylint passed. The optional mypy stage reports 14 messages also present in the baseline check (22 baseline messages), with no new messages.
- **Steady-state timing:** `bound-performance.json` retains three rotated repetitions per arm, 100 updates each, excluding ten warmups, using the existing training interval timer. No tensor hashing or CPU profiling runs in those measurements. Stdout and TensorBoard are enabled, log interval is one, W&B is disabled. Every step trains 2560 effective tokens. Only the measurement processes have fixed CPU affinity.

| Arm | Mean step ± between-run sample std, ms | Median of run medians, ms | Mean effective tokens/s |
|---|---:|---:|---:|
| Fixed CP2 | 145.377 ± 3.263 | 144.650 | 17,615.3 |
| DCP off | 113.547 ± 4.175 | 109.950 | 22,565.7 |
| DCP on | 116.215 ± 8.273 | 109.200 | 22,099.9 |

The on/off mean difference is **+2.35%**, with paired differences **+1.84%, −1.35%, +6.34%**. This shared-host sample does not resolve small overhead and does not establish zero overhead. It is a small dense model on synthetic tokens, not a production convergence or scaling result. Fixed CP/DCP use the same samples and checkpoint; the ratio of their mean step times is 1.28 in this particular workload.

The initial unbound measurements are retained in `performance.json` and `perf-*`; they have larger paired variation (−15.68% to +11.14%). They are not evidence that enabling metrics makes training faster.

The additional diagnostic `cpu-profile-summary.json` measures CPU wall time around the statistics, window and existing logging functions, and records Python GC events; it does not insert GPU operator timers. In-context statistics average 45–48 µs/step, window update about 12 µs/step, and the emitting rank's full `training_log` averages 1.565 ms off versus 1.604 ms on. Observed Python GC events are too small to account for the multi-millisecond tails. These diagnostic runs are excluded from primary performance aggregates. `cpu-stats-cost.json` separately records an isolated CPU helper microcheck.

## Environment and records

`environment.json` records Python, Torch, CUDA, TE, FA2, cuDNN, NCCL, NumPy and TensorBoard versions. The host driver is 580.178.04. The model is dense GPT: four layers, hidden 128, FFN 512, four heads, vocabulary 256, sequence limit 1024, zero dropout and BF16. The CSV generates valid lengths `[1024, 256, 128, 128, 512, 256, 128, 128]` after the current mock dataset's next-token shift.

- `train.log` and `command.sh` in each run preserve the exact original command and output.
- Correctness `audit/rank*.jsonl` contains source hashes, final schedules, pre-alignment microbatch counts, summaries and gradient/parameter hashes. TensorBoard event files are retained for the three enabled correctness arms.
- Performance `audit/measure-rank*.json` retains already-computed CPU return values from `train_step`; tokens/s is derived from them and the existing step timer. No GPU tensors or collectives are added by that observer.
- Checkpoint bytes stay in the original workspace; their names, sizes and SHA256 hashes are recorded in `ORIGIN.json`. Bootstrap commands regenerate equivalent fixed inputs for a new paired run. Original benchmark/correctness scripts are retained under `executed-scripts/`; the top-level launcher also permits relocated checkout paths and GPU indices for replay.
- The Docker command files describe the actual custom validation image, host venv, CUDA/driver-library mounts and device mapping. They are not a claim that this was the canonical CI container.

## Replay

Use this integration checkout, or apply `compatibility.patch` to the metrics source. Use an environment matching `environment.json`. Mount the repository and a **new** output directory at the same absolute paths in the validation container. The default container name is `dcp-stats-training`; each launcher also accepts `--container NAME`.

```bash
export DCP_REPO_ROOT="$PWD"
export DCP_RUN_DIR="/tmp/dcp-stats-replay-$(date +%s)"
mkdir -p "$DCP_RUN_DIR"
cp validation/dcp-stats-a6000/*.py validation/dcp-stats-a6000/lengths.csv "$DCP_RUN_DIR/"
# Bind DCP_REPO_ROOT and DCP_RUN_DIR into the prepared container at the same paths.
# Original device indices are shown; change them for another host/container.
export DCP_GPU_INDICES=4,5
python "$DCP_RUN_DIR/launch_train.py" initial-aligned --iters 1 --save
python "$DCP_RUN_DIR/launch_train.py" final-2-off --audit --load initial-aligned --save --eval
python "$DCP_RUN_DIR/launch_train.py" final-2-on --audit --load initial-aligned --save --eval --on

export DCP_GPU_INDICES=1,2,4,5
python "$DCP_RUN_DIR/launch_train.py" initial-tp --gpus 4 --tp 2 --iters 1 --save
python "$DCP_RUN_DIR/launch_train.py" final-tp-off --gpus 4 --tp 2 --audit --load initial-tp --save --eval
python "$DCP_RUN_DIR/launch_train.py" final-tp-on --gpus 4 --tp 2 --audit --load initial-tp --save --eval --on
python "$DCP_RUN_DIR/launch_train.py" initial-pp-v2 --gpus 4 --pp 2 --vpp --iters 1 --save
python "$DCP_RUN_DIR/launch_train.py" final-pp-off --gpus 4 --pp 2 --vpp --audit --load initial-pp-v2 --save --eval
python "$DCP_RUN_DIR/launch_train.py" final-pp-on --gpus 4 --pp 2 --vpp --audit --load initial-pp-v2 --save --eval --on
# Run inside the container, with the repository as working directory:
# python "$DCP_RUN_DIR/verify_acceptance.py"

export DCP_GPU_INDICES=4,5
# These CPU indices are host-specific; retain/report the binding used for a replay.
DCP_PERF_PREFIX=bound DCP_PERF_ITERS=101 \
  DCP_MEASURE_CPU_BINDINGS='[[0,2,32,34],[4,15,36,47]]' \
  python "$DCP_RUN_DIR/run_performance.py"
python "$DCP_RUN_DIR/analyze_performance.py" bound
```

The launcher's deterministic correctness flag is paired with `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Locally generated legacy Torch checkpoints use `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` because the saved config contains the QKVLayout enum. Resume uses `--use-checkpoint-opt-param-scheduler`. Measurement settings are validation-only; the product exposes only `--log-dcp-stats`.

Tests recorded on the standalone metrics checkout:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 -m pytest -q \
  tests/unit_tests/test_sequence_packing.py \
  tests/unit_tests/training/test_train_step_schedule_plumbing.py \
  tests/unit_tests/training/test_logger_ownership.py \
  -k 'not wrap_dataloader and not get_batch_on_this_rank_for_sequence_packing and not config_container_forwards_layer_wise_optimizer_to_model_builder'
```

On this integration checkout, the scheduling-only command yields the recorded 49 tests:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 -m pytest -q \
  tests/unit_tests/test_sequence_packing.py \
  -k 'not wrap_dataloader and not get_batch_on_this_rank_for_sequence_packing'
```
