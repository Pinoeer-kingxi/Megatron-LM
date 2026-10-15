# Rebased activation-offload PR evidence

This bundle records the rebase of PRs #8243, #8244 and #7908 onto upstream
`9131593673d037d360d252904f9198d36bc0c246`. It contains CPU checks and commands;
it contains no GPU execution or new performance result.

| PR | Current source |
| --- | --- |
| #8243, shared manager | `af9569270f476dc201484296315a9c30dcaa3feb` |
| #8244, gate backward | `9d54b8729c9277f1e032110c5e1b149e9654bcc5` |
| #7908, combined integration | `8a2d19894288a8eb890dfdf20c9d54683035de80` |
| #7908 prerequisite prefix | `7a00bd79bb105fce5a63fa312e50413d0055bff1` |

`range-diff.log` shows the sole patch change: the feature no longer rewrites the
GDN manifest note. It retains upstream's decode-kernel explanation and adds the
offload-test association. The manager, gate algorithm, GDN scope and profiling
comparison have no additional algorithm changes. The upstream GDN decode update
is inherited from main. `source-audit.json` records changed files and hashes,
clean worktrees, ancestry, local signature verification, and identical standalone
and combined gate patches. `feature-only.patch` excludes both prerequisites.

## Actual checks

- Shared CPU phase/process-group routing and summary: 5 passed.
- Gate manifest: 272 passed; combined manifest: 272 passed. These are CPU registry
  checks, not CUDA replay. New upstream registry entries account for the increase
  from the older 269-case record.
- Combined CPU configuration, input and snapshot comparison: 30 passed.
- Native training: 96 tests collected, not executed.
- Qwen CPU preflight: configuration for 24 text layers, TP=2/SP/PP=2, and 30,727
  tokenized input tokens. No training model or optimizer step was constructed.
- Standard Black/isort/Pylint/Ruff checks on all three heads, separate checks of
  both tools and their CPU tests, and the kernel CI coverage entry point pass.
  Standalone tool/test mypy passes. The shared/gate/combined nonblocking mypy
  counts are 34/4/46 and match the new upstream baseline after line normalization;
  this is not a claim that the repository is mypy-clean.
- The strict Sphinx build runs with `-W --keep-going` and the existing autodoc
  configuration. Its full command, exit code and log are in `docs-validation.json`
  and `strict-docs.log`.

Exact commands, source SHAs, timestamps, exit codes and clean-state checks are in
the validation JSON files. `run_checks.py` reproduces the CPU, format and docs
jobs in the recorded local environment. CPU pytest uses `--confcutdir` to exclude
the suite's GPU/data fixtures; it does not qualify the GPU training path.
The environment is the existing validated container and dependency overlays
recorded by the [preceding split bundle](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/cf8df32ec2424bcc6c42a5486d2fe68213c74294/split-20261015).
Bridge uses its recorded canonical-import compatibility patch; no core dependency
or runtime shim was added.

## Still pending

All eight local A6000s have existing compute processes; no devices were reserved
or used for this bundle. Shared asynchronous transfers and model/schedule GPU
regressions, 16-case gate replay and numerical checks, controlled old-atomic versus
fixed gate performance, and the combined native/Qwen GPU acceptance remain pending.
`gpu-validation.md`, `benchmark_gate.py` and `helpers/` retain the prepared commands
with the new source identities. None of these planned jobs is counted as passed.

Earlier two-/four-rank GPU logs and the offload memory/runtime measurements retain
their original revisions in the preceding immutable evidence bundles. The in-tree
performance summary remains unchanged: its unfused runtime source is
`4e18562c6bcf894795d23378a8f1699c79e4b797`, and its fixed-fused runtime source is
`2f1ee4f2e3dd6d909d08981338f5879632131139`. Those offload comparisons do not
establish old/new gate-kernel performance at the rebased heads.

## Verify

Download this directory from its pinned evidence commit and run:

```bash
python3 verify.py
```

The verifier checks file hashes, recorded exit codes/source versions, CPU totals,
the collected-only training count and explicit absence of GPU qualification.
