# PR #7908 split: source revisions and CPU evidence

This bundle prepares the three review scopes requested in
[the review comment](https://github.com/NVIDIA/Megatron-LM/pull/7908#issuecomment-6182348264).
**GPU acceptance is pending.** No GPUs were reserved for this preparation. No replay,
new gate-performance result or end-to-end training pass is claimed for the split
revisions. The older GPU evidence at the parent directory remains tied to its
original source; it does not qualify these branches.

## Source boundaries

| Scope | Commit | Production files | CPU checks |
| --- | --- | --- | --- |
| Shared activation-offload fixes | `fc9e737f49123e690e964d52555ff67bc80485d8` | Shared manager and pipeline schedules | Five phase/group-routing unit cases |
| Gate backward reproducibility | `9ff49d7e857350fbf998371d24d2d7ec7bb8a720` | Existing fused pre-GDR gate kernel | 269 kernel registry checks |
| Combined GDN feature | `8d138829b18fe14f6ea6376d26bf0542ff98b66e` | GDN integration and configuration, on top of both prerequisites | 21 input/config cases, 269 registry checks, 96 native cases collected, Qwen CPU preflight, strict documentation build |

Both independent fixes start at upstream
`1e0bb06695eb5c5d34e9d40f5fb2253beb3d06e8`. The combined feature branch contains
the shared fix, an identical-patch cherry-pick of the independent gate fix, and
one feature commit. `status.json` records the full parent revision, file hashes,
validation state and remaining work. Every source commit has DCO sign-off and a
locally verified SSH signature.

The original source is preserved by the signed tag
[`gdn-offload-before-split-20261015`](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/gdn-offload-before-split-20261015)
at `b22c9f7251a480529a4815c8c5fe0ece64734c03`. The new GDN integration, gate
kernel, feature tests and tools retain their previously reviewed source bytes;
upstream main changes, generic manager comments and a growing-saves regression
at fraction one are tracked explicitly in the split.

The final feature PR depends on both fixes. Until they merge or reviewed upstream
bot mirror refs exist, its base remains `main` and that comparison includes the
prerequisite commits. The PR description provides a separate feature-only diff.
No unsupported fork base or direct upstream push is used.

## CPU evidence and its limits

[Final CPU job records](final-cpu-validation.json) contain exact commands,
source SHAs, timestamps, exit codes, and clean Git checks before and after each
job. The logs in [cpu](cpu) preserve stdout/stderr, including warnings.

- Shared phase/process-group routing: **5 passed**; these are unit tests with
  mocked routing and do not exercise device transfers.
- Gate manifest: **269 passed**; registry validation is not GPU numerical replay.
- Feature configuration/input boundaries: **21 passed**.
- Feature manifest: **269 passed**.
- Native training: **96 cases collected**, not executed.
- Complete Qwen preflight: **24 text layers**, hidden size **1024**, **30,727
  rendered tokens**, TP=2/SP/PP=2 configuration, and `gpu_training_run=false`.
  It does not load a training model or perform optimizer steps.
- Sphinx HTML build runs with warnings treated as errors and autodoc enabled.
  The concise developer guide is registered in `docs/index.md`; the old two
  unregistered pages and raw CSVs are no longer part of the source submission.
- Black/isort/Pylint/Ruff pass. The shared/gate/combined nonblocking mypy runs
  report 34/4/48 diagnostics; after line-number normalization each matches the
  corresponding upstream baseline. Four standalone feature files pass mypy.
  Whole-repository mypy is not clean.

The first Qwen preflight failed because the pinned Bridge Gemma registry imported
`te_checkpoint` through a re-export removed by upstream main. Both the failure
and successful preflight logs are retained. The isolated Bridge worktree uses
`c860f8a5bc5fddd78690f32baa0b8696774308b4` with
[bridge-compatibility.patch](bridge-compatibility.patch); this directly imports
the existing function from `megatron.core.extensions.transformer_engine`.
The original Bridge checkout and all Megatron production files are unchanged by
this environment patch. The docs-only virtual environment uses the locked `docs`
dependency group; the GPU environment is not modified.

## Pending GPU work

[gpu-validation.md](gpu-validation.md) gives the commands and required results:

1. Shared-manager transfer/lifecycle regressions and existing GPT/MoE/graph
   regressions, with actual skips and their reasons recorded.
2. Sixteen gate replay/numerical cases, plus a controlled old-atomic versus fixed
   gate-wrapper benchmark on the same otherwise idle GPU. Each shape runs in a
   fresh process; three process repeats avoid carrying a tile selected for a
   different batch through the old autotuner's cache. This is offload-disabled
   BF16 kernel evidence, not a fusion-versus-unfused or whole-model comparison.
3. Combined native training: 96 passed and zero skipped per four ranks, launcher
   exit 0. Complete Qwen TP=2/SP/PP=2: eight fresh fusion/norm/offload arms,
   32 rank files and 96 exact complete recorded rank-step states, with zero
   pre-reset pool use and positive transfers in enabled arms.
4. Publish the new per-rank outputs and comparisons tied to those source SHAs.
   Upstream GPU CI requires approval for each new head; the earlier `/ok to test`
   applied to `b22c9f7`, not these new revisions.

No new performance conclusion is made while the benchmark is unrun. Pretrained
CP remains unqualified on the recorded A6000/TE attention configuration.

## Historical measurements

[historical-provenance.json](historical-provenance.json) lists the SHA-256 of all
five files copied byte-for-byte from `b22c9f7`: three CSVs and two detailed reports.
They retain the original four-rank and measurement source revisions. Unfused
complete-Qwen measurements use runtime source
`4e18562c6bcf894795d23378a8f1699c79e4b797`; fixed-fused measurements use
`2f1ee4f2e3dd6d909d08981338f5879632131139`.

These are historical results. They are not rerun or relabeled as split-revision
performance, and their separate fusion conditions are not a controlled comparison
of the old and fixed gate algorithms. Checkpoint/model files and private text are
not included.

## Verify downloaded evidence

From this directory, run:

```bash
sha256sum --check SHA256SUMS
python3 verify.py
```

The standard-library verifier checks original file hashes, recorded CPU outcomes
and source identities, and confirms that GPU qualification is still marked
pending. `verification.json` records this CPU verification of the evidence.
Checksums detect changed downloads; these author-recorded logs are not an
independent upstream CI attestation.
