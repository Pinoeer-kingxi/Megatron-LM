# Four-A6000 validation of native GDN activation offloading

Executed source: `87c22d1abae13abf375e8ed4ddfe62301cde87f4`, based on upstream
`9131593673d037d360d252904f9198d36bc0c246` and shared-manager fix
`af9569270f476dc201484296315a9c30dcaa3feb`.

## Results

- Shared-manager regression suite: 16 passed on every rank, zero skips.
- Targeted `dp-fused-False-1.0` regression: one case passed on every rank.
  Its three complete snapshot comparisons per rank all had zero actual differences.
- Native eight-block complete-decoder matrix: 96 passed on every rank, zero skips.
  This covers DP, TP, TP+SP, CP, PP, PP+VPP, TP+SP+CP and TP+SP+PP, both
  fusion and norm-recomputation settings, and fractions 0/0.5/1.
- Local pretrained Qwen3.5-0.8B: all 24 text decoder blocks, TP2+SP+PP2, CP1.
  Four fusion/norm settings each run disabled, repeated-disabled and offloaded
  training in fresh processes: 12 completed GPU runs, three optimizer steps
  and four microbatches per run, sequence 128. There are 144 recorded rank-step
  states and 96 rank-step comparisons. Only the last pipeline stage returns losses.
- Unfused Qwen states match exactly. Fused comparisons inspect actual differing
  vectors with predeclared `rtol=1e-5`, `atol=1e-8`; all other tensors must have
  identical full-byte fingerprints. All 16 fused rank/arm comparisons in this run
  actually have zero differences. Losses, gradients, weights and complete
  Adam/master-weight state are checked, including warmup.
- Enabled arms select positive transfer bytes and assert zero pinned-buffer
  ownership before native schedule resets. Disabled arms select no transfers.
- CPU-only evidence verification and frozen-source integrity checks pass.

Counts are per rank, not multiplied into independent test cases. These short
correctness runs include state snapshots and provide no performance or convergence result.

## Reproduction and raw evidence

`validation.json` records every exact launcher command, source identity, GPU UUID,
start/end time, exit code and container cleanup result. Per-rank stdout/stderr,
pytest reports and Qwen result JSONs are retained below each job directory.
`validation-summary.json` is recomputed by:

```bash
python3 -S scripts/verify_validation.py
```

The verifier runs on CPU and does not execute GPU tests. `environment.json` records
the imported Core path and installed package versions. `input-provenance.json`
records all nine local model/data file hashes; they match the pre-four-GPU snapshot.
Weights and training data are not redistributed. The tested Bridge revision is
`c860f8a5bc5fddd78690f32baa0b8696774308b4`; its import-only compatibility patch is
included in `scripts/bridge-compatibility.patch` and checked before the run.

The stage-specific launcher is `scripts/run_validation.py`. Its host paths,
container digest, local dependency overlays and GPU UUIDs are specific to the
recorded machine; adapt those when reconstructing the environment. The reused
`run_gpu_acceptance.py` supplies container/environment helpers; its historical
standalone CLI and `source-after.json` are not the source selector for this run.
The executed source is identified by `source.json`, `validation.json`, the imported
Core path and the per-file archive checks in `frozen-source-check.json`.

The full source archives, compiler caches and local `.pt` vector captures are not
included in the public export. The original committed source can be retrieved at
the exact SHA above. Difference reports retain the actual values and raw bytes
whenever vectors differ; unchanged matrix/vector tensors are checked by their
full-byte fingerprints. This run contains no differing tensors.

## The reproduced failure and minimal fix

The initial four-rank run at `fbe4bb03d1c6a60ac4da965ca47bccd47d7ef452`
failed the exact snapshot assertion in `dp-fused-False-1.0`. The reported differences
were an `A_log` gradient element and corresponding Adam/master-weight state;
losses and BF16 model weights matched. The original failure is retained in the
adjacent `four-gpu-initial` bundle, and is not counted as a passing command.
Those original actual tensors were not retained, so no numerical bound is claimed
for that first failure.

The fix only changes the native training test: fused snapshots use
`torch.testing.assert_close(rtol=1e-5, atol=1e-8)`, and unfused snapshots remain
exact. The targeted regression and full matrix were then executed at the new
commit. `component-provenance.json` verifies that Core and training tools are
byte-identical to the earlier revision, that the shared-manager files match
#8243, and that the pre-GDR kernel remains byte-identical to the upstream base.
No fixed tile sizes, tuning policy or reduction changes were added.

The native fused atomic reductions still have no global bitwise-determinism
guarantee. Zero observed differences here describe these inputs and matching
per-rank tuning choices. Every configuration uses isolated per-rank Triton caches
preserved across its comparison arms.

## Boundaries

This is complete text-model short training, excluding vision and MTP. Pretrained
Qwen CP, checkpoint restart, long-run convergence and composition with #7852 are
not qualified. Local results do not replace upstream GPU CI. Performance belongs
to separately timed runs without state snapshots.
