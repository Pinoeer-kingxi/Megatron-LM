# Two-A6000 validation before the four-rank test fix

Executed feature source: `fbe4bb03d1c6a60ac4da965ca47bccd47d7ef452`.
Independent gate source: `9d54b8729c9277f1e032110c5e1b149e9654bcc5` (#8244).
Devices: physical A6000s 2 and 6; gate replay used physical GPU 2.

- Shared-manager suite: 16 passed per rank, zero skips. Its files match #8243
  `af9569270f476dc201484296315a9c30dcaa3feb` byte-for-byte.
- Native complete-decoder matrix: 72 passed and 24 skipped per rank. All skips
  require four ranks (TP+SP+CP or TP+SP+PP); they are not counted as passed.
- Complete pretrained Qwen3.5-0.8B text model, TP2+SP, PP1, CP1, sequence 128,
  four microbatches and three optimizer steps: unfused norm-off/on comparisons
  match exactly across disabled, repeated-disabled and offload arms.
- The first native-fused norm-off probe completed its training processes but
  failed exact comparison on rank 1: an A_log gradient and corresponding Adam
  state differed. Original actual tensors were not saved, so no difference
  magnitude is claimed. The failed command and all raw rank outputs remain in
  `validation.json`, `run-validation.log` and the original Qwen directories.
- Fused norm-off/on were then rerun with actual vector capture and the existing
  `rtol=1e-5`, `atol=1e-8` contract. Every other tensor must have an identical
  full-byte fingerprint. All eight rank/arm comparisons actually have zero
  differences in those retests; `fused-numeric.json` records the fresh commands.
- Independent #8244 gate tests: all 16 replay/numerical cases passed, zero skips.
  This uses its own frozen source, not the native kernel in the feature run.
- The two redundant-comment edits preserve ASTs. Black/isort/Ruff/Pylint pass.

`validation.json` deliberately retains the first exact-probe failure; it is not
converted into a passing command. `fused-numeric.json` records the separate
numerical retest and gate replay. CPU-only evidence consistency can be checked:

```bash
python3 -S scripts/verify_validation.py
```

`manifest.json` is a separate, incomplete single-GPU performance attempt. It was
stopped at the user's request to prioritize four-GPU testing and provides no
qualified performance result. It is retained with the interruption reason.

Raw per-rank stdout/stderr, states and pytest reports are included. Source
archives, caches and local vector `.pt` files are excluded from public export.
`frozen-source-check.json` records the source and independent gate file checks.
`component-provenance.json` establishes that the feature preserves the upstream
pre-GDR kernel. Native atomic reductions have no global bitwise guarantee.

The stage-specific launchers record host-specific paths and container bindings.
The first launcher imported helper functions from the preceding performance
bundle; those helper files are byte-identical to the copies exported in scripts.
The common helper's historical standalone CLI/source-after.json does not select
these runs' code. Actual source selectors are source.json, the stage manifests,
imported Core paths and the frozen-source checks. Use the stage-specific launchers
when reconstructing the environment; the offline verifier is path-independent.
