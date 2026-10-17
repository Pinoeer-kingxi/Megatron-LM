# GDN change-necessity review

Reviewed PR heads: #8243 `af9569270f476dc201484296315a9c30dcaa3feb`, #8244
`9d54b8729c9277f1e032110c5e1b149e9654bcc5` and #7908
`b113b63614f201572f1e3742b4765449e55ceb3c`.

The follow-up commit is `a8d3a787b7ee24bf8bc1b927f8c3208d63b378fb`. Its only
changed file is `docs/developer/gdn_activation_offloading.md`: remove the old
rebase-validation matrix and PR merge-process prose, retain historical performance
versions and reproduction commands, and link the pinned restoration evidence.
The Chinese report explains the remaining production changes and their contracts.
No additional production change was found to be necessary.

`manifest.json` identifies the unchanged `megatron`, `tools` and `tests` trees at
this head and the actual GPU-tested source
`67fb40d842035a8873979759775b4165381ae60e`. The corresponding
[GPU evidence](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/6ca000df1d21df1fe042be13f008bd900de1f9b4/restore-upstream-gate-20261016)
keeps its original execution identity.

`document-check-command.json` retains the exact container command and exit code.
No GPU devices were exposed. MyST parsing reports zero warnings; all four Bash
blocks are identical to the previous document and pass `bash -n`. This is not a
full Sphinx build, a new GPU run, or a new performance measurement. The original
and revised documents are retained for comparison.

Verify the bundle without CUDA:

```bash
sha256sum --check SHA256SUMS
python -S verify_evidence.py
```
