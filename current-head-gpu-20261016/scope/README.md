# Current-source GDN scope regression

Executed source: `87c22d1abae13abf375e8ed4ddfe62301cde87f4`.
Physical GPU: A6000 6. The existing
`tests/unit_tests/ssm/test_gated_delta_net_offloading.py` suite passed all 73
scope/configuration cases with zero skips. It includes both fusion variants,
norm recomputation, fraction/threshold policies, packed sequences, storage aliases,
frozen projections, trainable-gate anchors and frozen-recurrence bypass.

The exact GPU launcher, source identity, device UUID, original rank stdout/stderr,
pytest report and exit/cleanup codes are retained. `validation-summary.json` and
`offline-validation.log` check the actual report. `frozen-source-check.json`
records that the executed source archive is unchanged. Compiler caches and source
archives are excluded from the public export. There are no timing conclusions
from this correctness job. The common helper's historical CLI is not used; the
stage-specific launcher and source.json identify this run.
