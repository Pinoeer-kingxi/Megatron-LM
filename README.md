# GDN offload: public post-rebase GPU evidence

Evidence for [NVIDIA/Megatron-LM PR #7908](https://github.com/NVIDIA/Megatron-LM/pull/7908).
This independent evidence branch contains logs and results, not additional PR code.
The dated tag `gdn-offload-post-rebase-20261014` is retained unchanged. The PR links
this bundle by its full evidence commit SHA as well.

## Source revisions

| Revision | Full commit | Role |
| --- | --- | --- |
| Rebase base | `e30d863a6d82a3c45f2d71355dc73fbf5610967b` | Upstream base for this submission |
| Earlier functional snapshot | `82ac2c88f7ba2f3b9aae3ad779d8575596ec2fec` | FLA backend guard and Qwen backend selection |
| Earlier published head | `8e8c43c84a4901b8a0d1d68409e6d81e3d33c5ff` | Documentation and registry notes after those GPU runs |
| Latest GPU-tested PR head | `b22c9f7251a480529a4815c8c5fe0ece64734c03` | Clean checkout for every 2026-10-14 GPU job; adds the public evidence link |

The October 14 runs start from the exact committed checkout, with no tracked or
untracked source changes before or after any of the six jobs. The launch commit,
GPU UUIDs, container command, exact job argument lists, timestamps and launcher
exit codes are in [run-provenance.json](latest-head-20261014/run-provenance.json).

The October 12 unit/native jobs began with backend-fix working-tree edits before
commit `82ac2c8`; Qwen jobs spanned that commit. They were not recorded as clean
launches at `82ac2c8`. [provenance.json](provenance.json) describes that distinction
and records file hashes for the corresponding committed functional snapshot.
[The complete 82ac2c8 → 8e8c43c diff](82ac2c8-to-8e8c43c.patch) separates subsequent
docstrings and registry notes from executable changes. The newer `b22c9f7` changes
only the validation document; the October 14 rerun removes the launch ambiguity.

## Latest clean-checkout GPU results

The two devices are RTX A6000s, physical indices 3 and 6. PyTorch exposes their
UUIDs without the `GPU-` prefix; the recorded preflight checks both devices.

| Workload | Result | Original evidence |
| --- | --- | --- |
| GDN/config + gate replay + shared manager | 104 passed, 24 deselected, no skips on each rank | [Rank logs](latest-head-20261014/rebased-unit-ranks), [combined log](latest-head-20261014/rebased-unit.log) |
| Native complete reduced-model training | 72 passed, 24 skipped on each rank; skips require four ranks | [Rank logs](latest-head-20261014/rebased-training-ranks), [combined log](latest-head-20261014/rebased-training.log) |
| Complete Qwen3.5-0.8B TP=2/SP | Four fresh runs; eight rank JSONs; 24 matching complete recorded rank-step states | [Qwen results](latest-head-20261014/qwen), [comparison output](latest-head-20261014/qwen-validation.json) |

[rank-logs.md](rank-logs.md) links each rank's stdout/stderr and Qwen result file
individually. All six launchers exit 0. For Qwen, fusion off/on each has a fresh
disabled baseline and fraction-1 offload run, with output-norm recomputation on.
Both ranks compare one warmup and two steady optimizer steps. States include
losses, gradient norm, every gradient and updated-weight fingerprint, and the
complete recorded Adam/master-weight state. Fingerprints contain shape, dtype
and SHA-256; tensor payloads, pretrained weights and private training text are
not uploaded. Every pre-reset pool count is zero; enabled arms select
66,015,232 transfer bytes per rank. Sequence length is 128 and the threshold is
1024 elements. These are correctness checks with no timed samples.

Two setup attempts are retained in [no-visible-GPU attempt](failed-attempt-20261014-no-visible-gpus)
and [UUID-format preflight attempt](failed-attempt-20261014-uuid-preflight).
The first fails NCCL initialization because numeric CUDA visibility did not
match the exposed device enumeration. The second detects the intended cards
but its preflight comparison includes a `GPU-` prefix absent from PyTorch's UUIDs.
Neither is a passing training run. The successful run uses UUID-based visibility
and compares UUIDs in PyTorch's format. Model/test source is identical throughout.

## Commands

The exact Docker/container and six launcher commands are recorded as argument
arrays in [run-provenance.json](latest-head-20261014/run-provenance.json), rather
than reconstructed from interleaved output. The original host coordinator is
[run_latest_public_gpu_evidence.py](run_latest_public_gpu_evidence.py); it records
clean-source checks and exit codes, and stops its container after the jobs.
Its absolute paths describe the original machine and need adapting for a rerun.

Inside an equivalent environment, the unit and native training invocations are:

```bash
git checkout b22c9f7251a480529a4815c8c5fe0ece64734c03
export PYTHONPATH="$PWD:$PYTHONPATH"
export NCCL_MAX_NCHANNELS=1 NCCL_NVLS_ENABLE=0
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  --log-dir /tmp/gdn-unit-ranks --redirects 3 --tee 3 \
  /path/to/evidence/four-gpu-acceptance-20261008/run_reduced.py -q -x \
  --confcutdir=tests/unit_tests/ssm \
  tests/unit_tests/ssm/test_gated_delta_net_offloading.py \
  tests/unit_tests/determinism/kernels/test_gdn_gate_backward.py \
  tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py \
  -k 'not test_gpt_fine_grained_activation_offloading_correctness_and_memory and not test_fine_grained_activation_offload_with_ep_a2a_overlap_compatibility and not test_fine_grained_activation_offloading_with_cuda_graph'
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  --log-dir /tmp/gdn-training-ranks --redirects 3 --tee 3 \
  /path/to/evidence/four-gpu-acceptance-20261008/run_reduced.py -q -x \
  --confcutdir=tests/unit_tests/ssm tests/unit_tests/ssm/test_gdn_offload_training.py
```

The cleanup wrapper restores distributed session teardown omitted by isolated
`--confcutdir` runs. Qwen uses [qwen_pool_lifetime_check.py](qwen_pool_lifetime_check.py),
which calls the repository tool under `/workspace` and records pool usage before
native reset. Each recorded Qwen command has `--tp 2 --sp --recompute-norm
--check-state --seq-length 128 --min-offloaded-tensor-size 1024 --warmup 1
--iterations 2`; fused arms add `--fused-pre-gdr`, enabled arms add `--fraction 1`.
New Qwen runs also record separate rank stdout/stderr through torchrun.

The package environment is PyTorch 2.11.0+cu130, TE 2.20.2, FLA 0.5.1,
causal-conv1d 1.6.1, FlashAttention 2.8.3 and cuDNN 9.19.0. The optional Bridge
checkout is `c860f8a5bc5fddd78690f32baa0b8696774308b4`, with the documented Qwen VL
importer compatibility overlay. The local container image ID is recorded; this
bundle does not host the image or optional overlays. Full dependency/data
provenance and environment setup remain in the
[source validation document](https://github.com/Pinoeer-kingxi/Megatron-LM/blob/b22c9f7251a480529a4815c8c5fe0ece64734c03/docs/developer/gdn_activation_offload_training.md).
Independent replay of this exact pretrained workload requires the original
weights and dataset. Reviewing the recorded comparisons does not.

## Earlier results and performance versions

The original October 12 [unit rank logs](publication-20261011/rebased-unit-ranks),
[native training rank logs](publication-20261011/rebased-training-ranks),
[Qwen rank JSONs and combined logs](qwen-training/publication-20261011/tp_sp), and
[Qwen comparison report](publication-20261011/qwen-rebased-validation.json) are
copied byte-for-byte. Separate Qwen rank stdout/stderr were not recorded then.
The bundle also includes the configuration guard's before-fix failure and
after-fix pass, formatter output, standalone mypy success, 263 manifest checks
and kernel-coverage output. Broader nonblocking mypy still reports 48 diagnostics;
[the comparison report](publication-20261011/mypy-diagnostic-comparison.json)
documents 45 identical messages and three dependency/upstream message changes.

No historical four-rank or performance matrix is rerun here. Their original
source provenance is retained in the PR and validation document:

| Performance data | Original runtime source |
| --- | --- |
| [Unfused complete-Qwen CSV](https://github.com/Pinoeer-kingxi/Megatron-LM/blob/b22c9f7251a480529a4815c8c5fe0ece64734c03/docs/developer/gdn_activation_offload_training_a6000.csv) | `4e18562c6bcf894795d23378a8f1699c79e4b797` |
| [Fixed-fused complete-Qwen CSV](https://github.com/Pinoeer-kingxi/Megatron-LM/blob/b22c9f7251a480529a4815c8c5fe0ece64734c03/docs/developer/gdn_activation_offload_training_fused_a6000.csv) | `2f1ee4f2e3dd6d909d08981338f5879632131139` |

Pretrained CP remains unqualified on A6000 at head dimension 256. Local package
runs do not qualify upstream GPU CI, checkpoint restart or long-run convergence.

## Independent verification without GPUs

[Download the dated evidence archive](https://github.com/Pinoeer-kingxi/Megatron-LM/archive/refs/tags/gdn-offload-post-rebase-20261014.tar.gz)
or clone the evidence branch. The PR also supplies the full evidence commit SHA
for downloading that exact revision's archive. From the extracted directory:

```bash
sha256sum --check SHA256SUMS
python3 verify.py
```

[verify.py](verify.py) uses only Python's standard library (Python 3.10+). It
checks SHA-256, reads each pytest rank summary, validates Qwen configuration,
pool ownership and transfer counts, and compares complete recorded states
between matching disabled/offloaded rank pairs for each fusion variant.
[verification.json](verification.json) and [verification.log](verification.log)
are newly generated CPU checks of these recorded artifacts, not additional GPU
runs. [SHA256SUMS](SHA256SUMS) covers original files, provenance, this README,
the index and verifier; it excludes itself and the two derived verification
outputs to avoid a checksum/report cycle. The pinned Git commit includes all files.
Checksums establish file integrity, not an independent CI attestation.

[check_verifier.py](check_verifier.py) exercises unchanged evidence and three
failure cases using temporary copies: changed file bytes, a changed Qwen gradient
norm with an updated result checksum, and a nonzero pre-reset pool count.
[Its output](checker-regressions.log) records the expected rejections. Run it with
`python3 check_verifier.py`; it leaves the original evidence unchanged.
