# PR #8244: preserve default autotuning and opt in to ordered gate gradients

Final source: `d8115c4c3bf52a868b242d032b5519d7c3331a52`.
Upstream comparison: `9131593673d037d360d252904f9198d36bc0c246`.
Previous published implementation: `9d54b8729c9277f1e032110c5e1b149e9654bcc5`.

The default gate backward again uses upstream's autotuned tiles and FP32 atomic
accumulators. Ordered FP32 partials are selected only when the existing
`use_deterministic_mode()` setting requests deterministic arithmetic. Its tile
comes from the existing `autotune_configs` selector; the same selection determines
both scratch capacity and launch parameters. There is no fixed 128×16 launch in
the new wrapper. Explicit `block_sizes` are supported. Pinning autotune alone does
not enable ordered arithmetic. This selector does not perform tuned-table lookup.

Full pre-GDR fusion still rejects model `deterministic_mode=True`. The local
kernel replay result does not establish determinism of the full fused model.
No activation-offload code or public configuration flag is added by this PR.

## Executed checks

- Final source: 34 gate GPU cases passed, zero skips. These cover BF16/FP32,
  batch and sequence/head tails, strided upstream gradients, explicit side
  streams, shared default and overridden 256×32 tiles, five byte-exact replays
  under contention, a Torch autograd numerical reference, and runtime mode changes.
- Final source: all 11 existing `TestFusedPreGatedDeltaRule` cases passed, zero
  skips, including dense/packed forward/backward and configuration boundaries.
- Final source: 272 registry cases passed, zero skips. These are CPU registry
  checks, not 272 GPU replay cases.
- Old-source control: the new default-dispatch regression test runs the actual
  backward using `9d54b872`, then fails because the original autotuner was bypassed.
  Its nonzero exit is expected; see `red-default/reports/rank-0.json`.
- `tools/autoformat.sh` passed in check mode on an isolated checkout of the final
  source, as did the explicit format/lint and determinism-coverage checks.
  The formatter treats mypy as nonblocking. Four mypy diagnostics remain, matching
  the upstream baseline: missing Triton/causal-conv stubs and a pre-existing optional
  tensor access. A newly introduced scratch-shape inference diagnostic was fixed.

`final-test-manifest.json` records final-source commands, rank logs and exit codes.
`test-manifest.json` records the earlier checks and old-source control. Both were
executed, not just collected. `autoformat-final.log`, `final-cpu-checks.json` and
`baseline-mypy.json` retain the static-check output.

## Full Qwen training

Single RTX A6000, physical GPU 1; BF16 Qwen3.5-0.8B text model, 24 decoder layers,
852,985,920 parameters, sequence 2048, microbatch size 1 and four accumulated
microbatches per optimizer step. The actual path includes mixed GDN/attention,
MLPs, embeddings, vocabulary loss and the native BF16 Adam optimizer. Pre-GDR
fusion is enabled; activation offload and norm recomputation are disabled.
Local weights/data identities are in `training-input-provenance.json`. Raw data,
weights and tensor payload files are not redistributed.

The external harness is copied under `scripts/`; it is not part of the PR.
Upstream records kernel choices once, then both arms use the same recorded table
with `mode="pinned"`, `on_miss="error"`, and ordered arithmetic disabled. This
controls launch choices while comparing default reduction behavior. Ordinary
timing-based dispatch is tested separately by the gate runtime-mode cases.
All paired jobs have identical input fingerprints and choice logs.

State runs compare three complete optimizer steps, including loss, gradients,
weights and full optimizer/master state. Upstream versus its repeat, upstream
versus the earlier candidate, and upstream versus the final candidate are checked
from `result.rank0.json`. The first upstream repeat and earlier candidate were
byte-exact. The final candidate differs at five tensor leaves: one `dt_bias`
gradient (maximum absolute difference `1.1920928955078125e-7`) and its Adam moments.
Loss and weights are byte-exact. An additional unchanged-upstream rerun reproduces
all five differing tensors byte-for-byte; that control also has `A_log` differences
up to `7.62939453125e-6`.

The original `rtol=1e-5, atol=1e-8` state comparison fails for both the candidate
and that unchanged-upstream control. This failure is retained in
`final-state-strict.log`. Numerical diagnostics use Torch FP32 defaults for
gradients (`rtol=1.3e-6, atol=1e-5`), retaining the original stricter tolerance for
optimizer states, and pass. The offline acceptance check instead requires every
changed candidate tensor to match an actual upstream-control tensor exactly; it
does not relax numerical tolerances to achieve a pass. This is an observed native
atomic envelope, not a full-model bit-exactness guarantee. Ordered gate replay is
tested separately and remains byte-exact.

Timing excludes model loading, compilation, input hashing and state snapshots.
Each measured step includes zeroing gradients, forward, backward, gradient
finalization, Adam and CUDA synchronization. Raw per-step samples and peak
allocated/reserved memory are retained. Sample summaries are recomputed by
`verify.py`; see `verification.json` for the complete tables.

- Initial campaign at `cc09983fd846b7b7f3172fde3580f3ec6bcd161e`: three fresh-process
  pairs, three warmup and ten timed steps per arm; paired changes +0.56%, +8.86%,
  +3.63%, median +3.63%.
- Extended campaign at that same source: three pairs, five warmup and fifty
  timed steps per arm; paired changes +5.51%, −3.45%, +0.64%, median +0.64%.
- Final-source timing and state jobs are in `final-training-manifest.json` and
  `final-pair-*` / `final-state-candidate`: five warmup and fifty timed steps per
  arm, 961.09 → 919.98 ms median, −4.28%, with equal peak allocated memory.
  The only change from `cc09983f` is a
  local type annotation; its normalized executable wrapper AST is unchanged
  (`typing-only-comparison.json`). Final GPU checks nevertheless reran.

Peak allocated memory is 22,793,579,520 bytes in the original paired campaigns.
The shared machine uses unlocked clocks; telemetry is retained. These varying
timings do not support a throughput improvement claim or a stable direction of
regression.

The untimed calibration has one unclassified exit-time PID sample. Its timing
is not qualified. Qualified timing/state jobs retain process ownership history
and have no detected foreign GPU-process overlap. The initial manifest and its
caveat are preserved; the ownership-sampling issue did not change production code.

## Diagnostic timeline

`profile-*` and `profile-manifest.json` contain Nsight Systems traces at
`cc09983f`, using the same launch table. Each captures three complete post-warmup
optimizer steps. Profiling timings are diagnostic, not throughput qualification.
`scripts/analyze_profiles.py` computes clipped activity unions, which must not be
added to overlapping CUDA API durations as if they were separate critical paths.

Both traces have identical kernel-name/count multisets: 8,392 kernels per step,
including 72 gate backward launches. GPU kernel-busy unions are about 787–789 ms
upstream and 785–786 ms in the candidate. Gate backward contributes about
0.73–0.81 ms versus 1.02–1.16 ms. Most profiled wall-time difference lies outside
GPU kernel/memcpy activity. This localizes the difference; it does not identify
the CPU-side cause or prove that all timing variation is external noise.

`default-kernel-check/*.sass.log` separately compares the actual final-source and
upstream SM86 gate kernels using the 32×16, two-warp configuration recorded for
Qwen. All 1,920 instruction-encoding words match, with 63 registers and zero
spills in both arms (`default-kernel-comparison.json`). This check applies to the
tested configuration and hardware; it does not cover other architectures.

## Recheck and reproduce

```bash
python3 -B -S verify.py
```

This verifies SHA256 checksums, source snapshots, actual per-rank outcomes,
state/input/choice comparisons, default-kernel instruction equality, qualified
job exits, telemetry and timing summaries. The default state comparison reports
its nonexact result and checks the observed upstream-control envelope.
It does not rerun CUDA or recover full tensors from fingerprints.

Exact invocations are in the manifests; package/image/GPU versions are in
`runtime-environment.json`. To rerun, check out the recorded source revisions,
map the recorded weights/data and overlay paths, and expose one idle GPU. The
runner scripts retain the original deployment paths; `run_gpu_acceptance.py`
is included for its imported environment helpers, not its older main campaign.
The table is included under `tables/`. Run the commands from each recorded source
root with its corresponding `PYTHONPATH`. The local dataset is required for exact
training reproduction; the gate and original fusion tests do not need it.

The historical [fixed-128×16 wrapper benchmark](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/8c5cc72379de4e9ef8d80856973dd64f23cc0dac/gpu-20261016)
belongs to `9d54b872`. Its 5.9%–8.7% wrapper gains and 2–32 KiB memory increase do
not qualify the current implementation and are not end-to-end speedups.
Upstream GPU CI has not been approved or run for this final head by this campaign.
