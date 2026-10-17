# Retained initial four-rank failure

Executed source: `fbe4bb03d1c6a60ac4da965ca47bccd47d7ef452`.
Devices: A6000s 1, 2, 3, 6. The shared-manager suite passed 16 cases on every rank.
The native matrix stopped at `test_gdn_offload_training[dp-fused-False-1.0]`:
all four ranks failed the bitwise snapshot assertion for an A_log gradient element
and corresponding optimizer state. Original launchers, stdout/stderr and pytest
failure reports are retained. No complete Qwen run was launched in this attempt.
Actual original tensor values were not saved, so their difference magnitude is
not measured. This attempt is a failed check, not acceptance evidence.

The adjacent four-gpu bundle records the minimal test-contract fix and complete
rerun at `87c22d1abae13abf375e8ed4ddfe62301cde87f4`. Core and training tools are
unchanged; only fused test snapshot comparison changes to a fixed numerical bound.

## Initial source materialization

The initial copy used copytree with dereferencing: 6 symlinks became regular files/directories with exactly the target contents. The strict symlink-layout check therefore failed; its original output is retained in frozen-source-check-initial.log. materialization-check.json verifies all 3532 archived regular files and records every dereferenced link. The successful rerun extracts the archive directly and passes the strict file and symlink checks. No initial failure data were rewritten.
