# Activation-offload scope validation, 2026-10-22

Current source validation for PRs #8243, #7908 and #8308 after reducing #8243 to
the independently reproduced forward-chunk bug. Every execution retains its
exact command, launcher exit and original stdout/stderr for every rank. Test
counts below are per rank; no skipped or collected-only case counts as passed.

| Source | Commit |
| --- | --- |
| control | `0b30779a83a939757beb84b8226874262dd396f7` |
| shared | `d8be77f56f4d38713cb0306c5c4893aa3cb2331a` |
| feature | `e96793feecb0fdd54ce00ef09a67e28fce940d06` |
| extensions | `3b6f5e66f64bd381893e0ea814511031c9b5e4af` |
| feature-control | `f9c607f5e3d16a7609bea8f77c76b87eeba87653` |

The prerequisite base is `0a9f3e55459175d7e5621a99efb2d6f5b8378a8e`, integrating
main `6c956376e3ef22528413ff35412ad8b938272986` and the original #7399 head
`73ac08c254a0ff8997a0ff559799a4d7f073248b`. Its original commits are preserved.
Merge order remains #7399, #8243, #7908, #8308. The `*-own.patch` files isolate
each PR from its prerequisites. #8244 is unchanged at
`d8115c4c3bf52a868b242d032b5519d7c3331a52` and is not a prerequisite.

## Retained fix and withdrawn changes

#8243 removes only the queue-empty shortcut in `finish_all_groups()` and clarifies
its docstring. Remaining forward groups and the existing cursor determine chunk
completion. There is no new manager state or scheduler API. Manager code outside
that function and the native schedules are identical to the prerequisite; native
pre-GDR arithmetic, launch grids and autotuning are identical to main.

The added GPT regression starts directly with native training at fixed sequence
length 128: two microbatches, four layers, three SGD steps, and fractions 0/0.5/1.
It compares loss and every produced parameter gradient with offloading disabled.
The observer delegates to real CUDA transfers; it does not simulate copies,
change stream dependencies, alter the pool, or override warmup.

`control-final-zero` uses the unchanged prerequisite manager and the identical
regression file. Both ranks fail inside backward with `Chunk mismatch` at fraction
zero. The same control passes fractions 0.5 and 1. The fix passes all three.
`gdn-old-chunk-control` restores only that old function on the current GDN feature:
the unchanged complete reduced-decoder/DDP/BF16 Adam fraction-zero test also fails
inside backward on both ranks. These are expected control failures, not passing
executions of the fixed source. Neither control requires prior evaluation or
changing tensor sizes. Default fraction-one failure is not claimed.

Evaluation-first warmup reset handling and the special handling of initially empty
groups whose tensors later grow are withdrawn from this PR. Their necessity for
the fixed-length training feature was not established. This does not establish
that those combinations are universally correct. Feature tests now cover normal
training → evaluation/no-grad → training; evaluation-first initialization and
tensor-size growth after warmup are not qualified. The extra stream-wide wait and
duplicate fallback reload wait remain removed; #7399 owns inline reload ordering
and reload-LIFO bookkeeping.

The two signed negative-control commits are in `negative-controls.bundle`.
First fetch the published feature branch, then run:

```bash
git bundle verify /path/to/negative-controls.bundle
git fetch /path/to/negative-controls.bundle 'refs/heads/scope/*:refs/remotes/controls/*'
```

`scope-verification.json` records identical test blobs and unchanged surrounding
manager/schedule/kernel code.

## Executed checks

| Job | Source | Ranks | Passed per rank | Skipped |
| --- | --- | ---: | ---: | ---: |
| control-positive-fractions | `0b30779a83` | 2 | 2 | 0 |
| shared-core | `d8be77f56f` | 2 | 23 | 0 |
| shared-four-fixed | `d8be77f56f` | 4 | 4 | 0 |
| feature-four-native | `e96793feec` | 4 | 96 | 0 |
| feature-scopes | `e96793feec` | 2 | 92 | 0 |
| extensions-scopes | `3b6f5e66f6` | 2 | 20 | 0 |
| extensions-native | `3b6f5e66f6` | 2 | 18 | 0 |
| extensions-graphs | `3b6f5e66f6` | 2 | 21 | 0 |
| shared-graphs | `d8be77f56f` | 2 | 3 | 0 |

The 23-case shared job includes two CPU checks and 21 GPU-dependent checks,
including the existing seven GPT/MoE correctness/memory cases. CPU checks do not
qualify asynchronous transfers. The four-rank shared job runs the three new
fixed-length cases and #7399's inline-reload regression.

The feature matrix executes complete reduced decoders with vocabulary loss, DDP,
BF16 Adam, all parameter gradients, updated weights and complete optimizer/master
state. It covers DP, TP, SP, CP, PP, VPP and the selected combinations, fractions
0/0.5/1 and norm/fusion settings. The 92-case scope/tool job includes 71 scope and
configuration cases plus 21 CPU tool/input-boundary cases. The graph job actually
captures and replays TE attention/whole-layer graphs; all 21 settings run three
native and three replay training steps, with pool ownership checked before native
schedule reset. GDN2 native and scope checks use two ranks, not four-card composition.

Unfused native state comparisons are exact. Native fused decoder comparisons use
`rtol=1e-5`, `atol=1e-8` for gate atomic-reduction variability; no global bitwise
determinism claim is made. Boundary/configuration checks are not described as
end-to-end training tests.

## Complete Qwen check

Four fresh four-rank runs load the local Qwen3.5-0.8B checkpoint and train all 24
text blocks, embeddings, MLPs, vocabulary loss and native BF16 Adam. Vision and
MTP are excluded. The common configuration is TP2+SP+PP2, sequence 128, four
microbatches, norm recomputation enabled, unfused pre-GDR, and a 1024-element
offload threshold. Each run executes three optimizer steps. Arms are disabled,
repeated-disabled, fraction zero and fraction one.

Every rank's recorded losses, gradient norms, all parameter-gradient fingerprints,
weights and complete Adam/master-state fingerprints match exactly across all
four arms. Fraction one observes actual selected transfers; fraction zero does
not select transfers after warmup. Pool ownership is checked before native resets.
PP ranks 0/1 have no local loss records; ranks 2/3 each have four microbatch losses
per step. The offline auditor recomputes equality from the original result files.
This is a short correctness check, not a current-head throughput measurement.

## Environment, source identity and limits

Environment: four RTX A6000s, Python 3.12.4, PyTorch 2.11.0+cu130, TE 2.20.2,
FLA 0.5.1, Triton 3.6.0, causal-conv1d 1.6.1, and flash-attn 2.8.3. The image
digest, GPU UUIDs, mounts and commands are recorded. Bridge's separately pinned
revision/import-only compatibility overlay is retained. Compiled native caches
were copied from the earlier campaign and kept per rank across comparison arms;
native arithmetic and autotuning were not changed. Weights, private JSONL data,
bulk tensor captures and caches are not published.

Check out the pinned fork commit, use the recorded dependencies, then adapt local
source/model/data/output/cache paths in each `manifest.json` command. Rank launch
helpers are included; `run_qwen.py` records the local orchestration used and
references a previous local manifest. The explicit commands in the current job
manifests do not require that previous manifest. Public records can be audited
without a GPU:

```bash
sha256sum -c SHA256SUMS
python3 scripts/verify_acceptance.py
```

Black, isort, Ruff, project Pylint and kernel coverage checks pass on the recorded
heads. Mypy is nonblocking in the project formatter: its logs remain included;
the 13 shared-file diagnostics equal the prerequisite after line normalization.
No clean global mypy result is claimed. SSH commit-signature checks are included.
Read-only merges with main `ede2c586cf70dd452f93b9726b31cbedc8df0dd0` are clean and
their `megatron/core` trees equal the tested trees. These are merge previews,
not GPU executions of new merge commits or of main's updated dependency lock.

Previous performance at `1adb1f6e31`, checkpoint checks at `485cb5fa2`, stability
at `87c22d1ab`, and graph performance at `d18040ba5a`/`367d014c7b` remain historical
with their original public logs. They are not qualifications of these heads.
No new performance, full pretrained graph-training, checkpoint-restart or
200-step stability result is claimed. Composition with #7852 is not qualified.
Upstream GPU CI is pending, and #7399 must land first. #8308 remains draft.
