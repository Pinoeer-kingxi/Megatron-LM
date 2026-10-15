# PR body and source review follow-up

Current feature source: `a0c0709a05c39fb3400d53cba3eb3d409063a35d`. Its parent is `8d138829b18fe14f6ea6376d26bf0542ff98b66e`.
The only source change is a clarification in
`docs/developer/gdn_activation_offloading.md`: fraction zero prevents transfers
after warmup; warmup may still copy eligible tensors to collect size statistics.
No Python production, test or tool file changed. `source-delta.patch` and the
Python hashes in `status.json` allow that relationship to be checked.

The earlier CPU tests and Qwen preflight remain tied to `8d138829b18fe14f6ea6376d26bf0542ff98b66e` in
[the frozen CPU bundle](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/cf8df32ec2424bcc6c42a5486d2fe68213c74294/split-20261015).
They were not rerun or relabeled as tests at the documentation-only revision.
The strict documentation job in `docs-validation.json` records the current
commit, exact command, clean-source checks, timestamps and exit code; its complete
stdout/stderr is `review-docs.log`. Autodoc is enabled, with `-W --keep-going`; no extra
warning suppression is added to the upstream `ref.python` setting.

The prerequisite PR bodies now name their exact upstream mypy comparison base
`1e0bb06695eb5c5d34e9d40f5fb2253beb3d06e8`. The shared-manager body explains the warmup
exception. The feature body explicitly states that CPU preflight reads model
configuration and tokenizes text without constructing a training model or taking
optimizer steps. The pre-checks distinguish unit tests added, functional recipes not added, and
GPU execution still pending. The original human author-review checkbox stays unchecked.

GPU regression, numerical replay, controlled gate performance, four-rank native
training and complete Qwen training remain pending on the split revisions.
Local signature verification and DCO success do not imply GitHub marks the SSH
key as verified; new-head GPU CI still needs maintainer approval. No review
comment, approval, mention or CI dispatch was sent during this follow-up.

To check the downloaded files, run from this directory:

```bash
sha256sum --check SHA256SUMS
```

With source commits fetched, verify the Python code is unchanged:

```bash
git diff --exit-code 8d138829b18fe14f6ea6376d26bf0542ff98b66e a0c0709a05c39fb3400d53cba3eb3d409063a35d -- '*.py'
```
