# GDN offload acceptance — 2026-10-26

This bundle separates executed revisions from later documentation-only updates.
All counts below are executed outcomes, not collection counts. Local CUDA runs
use RTX A6000; remote precision runs use RTX PRO 6000 Blackwell (SM120).

## Source and results

| Executed source | Check | Actual outcome |
| --- | --- | --- |
| `041a5014201f5175048536b906d22f50d7e0f3c8` | Native GPT: fixed, evaluation-first, sequence growth; fractions 0/.5/1 | 9 passed |
| `041a5014201f5175048536b906d22f50d7e0f3c8` | Existing TE graph/attention/MoE cases, fractions 0/.5/1 | 3 passed |
| `1f2c9ae4f7e617b12caa115e387d95e668517c62` | GDN scope/configuration, complete reduced decoder/DDP/BF16 Adam, profiler and Qwen input tests | 104 passed, 84 skipped because a single rank cannot run the requested parallel topology |
| `1f2c9ae4f7e617b12caa115e387d95e668517c62` | Complete pretrained Qwen3.5-0.8B text model: disabled/repeated-disabled/offloaded, S128, four microbatches, norm recomputation, unfused, three optimizer steps | Exact loss/gradient/weight/complete Adam fingerprints across all three processes; real transfers |
| `1f2c9ae4f7e617b12caa115e387d95e668517c62` | Complete Qwen performance: S4096, four microbatches, default threshold, fraction 1, norm recomputation off | 12 successful processes: three alternating on/off pairs per fusion setting, three warmup plus ten measured steps |
| `ef431833044fc1a180186ee14beb0aa4e8c98e3e` | Complete reduced hybrid decoder, native schedule/DDP/BF16 Adam: fixed, no-grad evaluation-first, S8/256/64/256, fractions 0/.5/1 | 9 passed; exact losses/gradients/weights/complete Adam states; zero pinned ownership before reset; actual transfer observations |
| `c169da398` (independent precision extension) | Complete reduced hybrid decoder: FP8 delayed/tensorwise and constrained NVFP4, fractions 0/.5/1, fusion and norm recomputation off/on | 36 passed on SM120 |

`ef4318330` changes only documentation after `1f2c9ae4f`; their production,
test and tool files are byte-identical. The precision branch later adds only
documentation at `ebdb25fc4`; its executed production/test/tool source remains
that of `c169da398`. Full identities and tree hashes are in
[source-identities.json](source-identities.json). The low-precision extension
is not part of the BF16 feature's acceptance claim.

The previously published four-rank matrix and TP2+SP+PP2 Qwen runs belong to
`e96793feecb0fdd54ce00ef09a67e28fce940d06` in the
[earlier bundle](../offload-scope-20261022/README.md). They were not rerun on four
ranks here. Checkpoint and 250-step results also retain their original revisions;
this bundle does not promote those historical executions to the new head.
Upstream GPU CI and composition with #7852 are not qualified here.

## Reproduced failures and boundaries

At `f9d0b7afd150caf5f323346f8cddd3c5e6200b1d`, the identical native GPT
regression fails after no-grad evaluation initializes warmup without saved
tensors (`shared-control-evaluation`) and when fraction zero starts transfers
after an S8 warmup followed by S256 (`shared-control-growth`). Both reach the
exact loss/gradient comparison before failing the actual-transfer assertion.

The native GDN control at `e96793fee` independently confirms that evaluation-first
leaves subsequent offloading silently disabled, with exact training states.
The calibrated size-growth control records 40 MiB of D2H copies on each S256
step despite fraction zero, again with exact training states. These are offload
lifecycle/policy failures, not evidence of wrong arithmetic. No manager state is
fabricated and no stream delay is injected by these native regressions. The
observer delegates actual copy methods and records ownership before native reset.

The resize case exercises programmatic native training with changing sequence
lengths; it is distinct from the ordinary fixed-length training case. A positive
fraction's group budget is learned during warmup and is not recomputed after
size changes. If no groups met the warmup threshold, .5 and 1 can transfer the
same later groups. This change guarantees zero remains zero; it does not add a
dynamic transfer-budget mechanism.

For the S8 cold-compilation diagnostic, the first disabled model differed from
later disabled models as well as offloading. The retained `baseline-1` files
record that calibration effect. The decisive comparisons require the subsequent
repeated disabled control (`baseline-2`) and offloaded states to match exactly.
No numerical kernel or tolerance was changed to hide that effect.

## Current-source memory/runtime trade-off

All values compare on/off within the same source, hardware, fusion setting and
configuration. Input preparation and snapshots are outside the timed steps;
performance processes do not request state snapshots. Full state checks are
separate correctness processes.

| Pre-GDR | Allocated peak off → on (MiB) | Allocated reduction | Reserved peak off → on (MiB) | Paired median step-time change [range] |
| --- | --- | --- | --- | --- |
| Unfused | 27719.20 → 26767.20 | 952.00 MiB (3.43%) | 28790–28792 → 29442 | +0.50% [−0.26%, +0.55%] |
| Native fused | 27422.20 → 26470.20 | 952.00 MiB (3.47%) | 29244–29262 → 30000 | +4.11% [+3.58%, +6.11%] |

On arms retain 1008 MiB in 72 pinned buffers; all 3552 allocation requests are
matched by returns and ownership is zero at final cleanup. Allocator reserved
memory increases despite reduced active allocations. Device telemetry records
one compute PID on the measured GPU; other GPUs on the shared host were active.
These are trade-off measurements, not convergence or cross-version speedup
claims. All 13 logged loss sequences and printed gradient norms match within
each pair; rounded printed norms do not establish full state equality.

The first measurement completed successfully before its host-side log parser
failed on TE's tensor-valued printed norm. It was reused without rerunning the
measurement after fixing only collection/parsing. Both exact runner versions are
retained; `run_performance.initial.py` matches that manifest's SHA256.

## Precision extension and incomplete remote checks

SM120 uses Torch 2.11.0+cu130, TE 2.20.2, FLA 0.5.1 and Triton 3.6. The 36 native
training cases compare actual losses, gradients, weights and complete Adam state;
fused cases retain the existing `rtol=1e-5`, `atol=1e-8` atomic-reduction tolerance.
An earlier 12-case sequential diagnostic also records exact state equality.
NVFP4 sets `NVTE_NVFP4_DISABLE_RHT=1` and
`NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1` before Python starts. This does not
qualify the full stochastic recipe, MXFP8, other hardware or convergence.
A small default-NVFP4 GEMM smoke passed; it is not a full-model qualification.

Full pretrained Qwen low-precision validation is incomplete. Its disabled baseline
cannot execute the reused A6000-only FlashAttention binary on SM120. Setting
`NVTE_FLASH_ATTN=0` also conflicts with the Bridge vision model's process-wide
`attention_backend=auto` contract. The failed commands/logs are retained in
`remote-environment-attempts`. Neither attempt qualifies offloading or proves an
offload bug. The GPU became occupied by another user's job while checks were underway; no
further training was started after this was identified. A separate SM120 dependency build does not
change PR source or turn these failures into passes.

## Reproduce and audit

Each job's `manifest.json` retains source, command, exit status and source/dirty
checks where the runner supports them. Rank stdout/stderr and pytest outcomes
are kept alongside results. Remote native logs and outcomes are in
`remote-native-final`; the remote command and exported recipe flags are in the
retained launcher scripts/logs. The container environment and import-only
compatibility overlays are described by the environment manifest. Local model
weights and private training data are not redistributed.

```bash
sha256sum -c SHA256SUMS
python -B -S scripts/audit_evidence.py
```

The audit checks recorded pass/skip/failure counts, expected old-code failures,
state comparisons, actual transfers, before-reset ownership, raw performance
samples and every checksum. Black, isort, Ruff and project Pylint pass on the
changed files. `tools/autoformat.sh` was executed using a local mirror of main
`ed9c2c8c1cf4a5438922ae8d215181e7c8b9e0fe` because the container cannot reach
GitHub. Its nonblocking mypy output is retained: the shared change has the same
21 diagnostics as its unchanged prerequisite after line-number normalization.
