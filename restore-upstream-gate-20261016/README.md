# Upstream pre-GDR restoration checks

Measured source: `67fb40d842035a8873979759775b4165381ae60e`, based on
`e9975a744efad181f9ab586ce33f58e7a9649ac8`. The pre-GDR source is byte-identical to
the upstream base `9131593673d037d360d252904f9198d36bc0c246`:
SHA-256 `c8a0247766bbc78ee3f98841a3eb9c033109bf3575bb9d2df99b4cbf2a0df920`.
This restores autotuning, atomic parameter-gradient accumulation and the original
buffers. PR #7908 no longer depends on the independent #8244 repeatability change.

Final PR head: `b113b63614f201572f1e3742b4765449e55ceb3c`. It removes the gate
fix/revert from the feature's history, leaving one feature commit above #8243.
Its complete Git tree is identical to the actually executed commit:
`1265e652350d601b43f66dc8527d6e26f0ac5fe8`. `final-head.json` and
`commit-trees.json` retain the mapping and public GitHub commit metadata. No test
result has been relabeled as execution at the new commit.

All tests import a frozen `git archive` snapshot of the measured commit.
`source-validation.json` records the archive checksum and comparison of all 3532
archived files after execution. No production or test source changed during the
runs. `restoration-check.json` records the final diff against the upstream base.

## Results

- One A6000: 73 GDN scope/configuration cases passed; 12 native complete-decoder
  single-rank cases passed with fusion/norm/fraction coverage. The native test's
  other 84 topology cases were explicitly deselected, not passed or skipped.
- CPU: 30 configuration/input/snapshot cases and 269 manifest cases passed.
- Black, isort, Ruff and Pylint passed for the Python files in the final PR diff.
- Sequence 2048: native eight-layer mixed GDN/attention decoder, embedding, MLP,
  loss and BF16 Adam; four microbatches and three optimizer steps per arm. Both
  norm-recomputation settings compare disabled, repeated disabled and offload
  models initialized with identical weights. Each comparison checks 1185 tensor
  leaves across losses, gradients, updated weights and complete optimizer state.
  Predeclared bounds are `rtol=1e-5`, `atol=1e-8`; all four comparisons actually
  have zero numerical differences. Pool ownership before every offload reset is
  zero. The manager reports 85,852,160 offload-summary bytes in each enabled run.

`checks.json` and `long-checks.json` contain exact commands, exit codes and GPU
UUIDs. Both containers exited and released their devices. `environment.json`
records PyTorch 2.11.0+cu130, CUDA 13.0, TE 2.20.2, FLA 0.5.1, Triton 3.6.0,
causal-conv1d 1.6.1 and flash-attn 2.8.3. GPUs 3 and 6 ran independent correctness
jobs; there are no throughput claims from these overlapping runs.

The first long-control setup reused a configuration after the optimizer attached
callbacks to it. Building the next model then failed while copying a ProcessGroup.
`long-initial` retains the unedited failure. The final validation script creates
separate configurations before building any model; production source and the
declared numerical bounds did not change to address that setup error.

## Reproduction

Use the measured checkout in the recorded dependency environment. For example:

```bash
export NVTE_CPU_OFFLOAD_V1=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
python -m torch.distributed.run --standalone --nproc-per-node=1 -m pytest -q \
  --confcutdir=tests/unit_tests/ssm \
  tests/unit_tests/ssm/test_gated_delta_net_offloading.py
python -m torch.distributed.run --standalone --nproc-per-node=1 -m pytest -q \
  --confcutdir=tests/unit_tests/ssm \
  tests/unit_tests/ssm/test_gdn_offload_training.py -k dp

export CAUSAL_CONV1D_DETERMINISTIC=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export TRITON_CACHE_DIR="$(mktemp -d /tmp/gdn-restore-triton.XXXXXX)"
export EVIDENCE_DIR=/path/to/evidence
mkdir -p "${EVIDENCE_DIR}/reproduce-norm-0"
python -m torch.distributed.run --standalone --nproc-per-node=1 \
  "${EVIDENCE_DIR}/scripts/rank_env.py" \
  "${EVIDENCE_DIR}/scripts/long_fused_training.py" \
  --output "${EVIDENCE_DIR}/reproduce-norm-0/result.json"
```

Repeat the last command in a new output directory with `--recompute-norm`.
The training probe uses the repository's native integration-test model builder;
it is a reduced model, not qualification of a pretrained Qwen checkpoint.
The host orchestration scripts retain this host's paths and container layout.
Their imported helpers are included in `dependency-helpers`; these are evidence
tools, not additional PR source or general deployment tooling.

Verify the retained reports without CUDA:

```bash
sha256sum --check SHA256SUMS
python -S scripts/verify_evidence.py
```

Full local tensor dumps are large and excluded from the public bundle;
`tensor-dump-sha256.json` identifies them. Per-tensor numerical difference reports,
all rank stdout/stderr, commands and the training probe are included. The offline
verifier checks report consistency and checksums; it does not rerun tensor math.

These checks do not requalify four-rank training, full pretrained Qwen training,
checkpoint restart, convergence, #7852 composition or performance at this head.
The old four-rank and fixed-reduction measurements retain their original source
commits. No global bitwise determinism is claimed for native atomic/autotuned
fusion. Upstream GPU CI and maintainer review remain outstanding.
