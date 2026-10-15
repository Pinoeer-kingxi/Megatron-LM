# Snapshot comparison correction

Source: `ee2e6c1dc8ebab924fdf884a2532b28d7a16adb9`, following `a0c0709a05c39fb3400d53cba3eb3d409063a35d`.
Only the standalone GDN profiling tool and its new CPU test changed. The five
production runtime files, kernel registry and existing tests are unchanged.
`source-delta.patch` and `status.json` record that boundary.

## Reproduced issue and minimal fix

The profiler labeled successful comparisons `bitwise_equal=True`, but compared
with `torch.equal`. That predicate treats `+0.0` and `-0.0` as equal even though
their bytes differ. `signed-zero-reproduction.log` records the old predicate
and the corrected private comparison on real CPU tensors, with zero CUDA devices.
It is not a GPU offload replay.

The comparison now checks dtype, exact values and logical tensor bytes. It uses
the existing contiguous/flatten/uint8-view idiom also used by Qwen fingerprints
and the kernel replay harness. Exact-value validation remains first so NaNs are
still rejected. The comparison stays outside the measured GPU step.

The tool also stopped maintaining a redundant maximum pool-use counter: each
step already fails on any nonzero count, so every successful checked run derives
zero for the existing output field. The per-step check and output schema remain.
No new manager, transfer mechanism, recovery path or public option was added.

## Validation and limits

`cpu-validation.json` records exact commands, clean-source checks, timestamps
and exit codes at the source revision above. `cpu-regression.log` contains
**30 passed, 62 deselected**: nine snapshot cases plus the existing eleven GDN
configuration and ten Qwen CLI cases. Snapshot coverage includes FP32/BF16
signed zero, scalar tensors, one ULP, non-contiguous layout, shape/dtype changes,
NaN rejection and equal finite values. No model training ran.

Standard autoformat and the explicit tool/test Black, isort, Pylint and Ruff
checks passed on the exact file bytes recorded in `status.json`, before commit.
Mypy passes on both changed files. The 48 nonblocking diagnostics from the
standard core/test run match upstream `1e0bb06695eb5c5d34e9d40f5fb2253beb3d06e8`
after line-number normalization. Whole-repository mypy is not claimed clean.

The last strict documentation build belongs to `a0c0709a05c39fb3400d53cba3eb3d409063a35d`; docs and core
source bytes have not changed since it. That build is not relabeled as a new run.
The prior manifest checks and native-case collection remain tied to their
original revisions in the earlier evidence bundles.

GPU transfer regressions, gate replay/performance and four-rank native/Qwen
training remain pending. Historical standalone-stack `bitwise_equal` fields
used the old exact-value comparison and do not prove signed-zero byte equality.
Qwen state SHA-256 fingerprints and the gate replay harness already compare raw
bytes; this correction does not change those implementations. Historical memory
and runtime data retain their original source revisions, with no new
performance conclusion for this commit.

From this directory, verify the downloaded files with:

```bash
sha256sum --check SHA256SUMS
```
