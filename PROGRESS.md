# Issue 1312 progress

## State

The checkout-root routing and ProgramSpec citation fix is implemented, passes
its focused current-main regression and compatibility controls, and is
verified end to end against the held-out SC change through a disposable
backport to the pinned signing bridge. Full gates and the independent
review-fix cycle remain.

## Done

- Created the requested local branch and isolated worktree from `origin/main`.
- Read the repository instructions and applicable debugging/GitHub workflows.
- Established the fail-first, implementation, real-checkout, compatibility,
  validation, and independent-review plan.
- Read issue 1312, rulespec-us PR 1139, issue 1078, and the committed Arizona
  ProgramSpec manifest that establishes checkout-root placement and a bare
  `programs/...` citation.
- Confirmed the pinned bridge mechanism: the ProgramSpec has no leading
  jurisdiction prefix, falls back to `<checkout>/us`, and fails while rebasing
  `<checkout>/programs/...`; its anchor is also incorrectly `us:programs/...`.
- Located the exact held-out SC change in rulespec-us reflog commit `f93f556c`:
  add page 159/remove page 369 in the ProgramSpec and drain three worklist rows.
- Added current-main regressions for the signed ProgramSpec manifest, the four
  requested checkout-root source roots, and checkout-root placement while
  retaining the UK and jurisdiction-content-root controls.
- Captured the fail-first focused run: six failures, including the exact
  `programs/us-sc/snap/fy-2026.yaml` writer case.
- Recorded that `uv run` cannot access `~/.cache/uv` in the sandbox; focused
  current-main tests run with an existing compatible project environment.
- Routed checkout-owned `policies/`, `programs/`, `regulations/`, and
  `statutes/` outputs through the exact checkout root, taught manifest
  placement to accept that root, and made ProgramSpec citations bare
  `programs/...` paths.
- Added the issue 1312 changelog entry.
- Passed the focused current-main regression plus the jurisdiction-prefixed and
  UK issue-1078 controls: nine tests passed; focused Ruff also passed.
- Reconstructed the exact two-file SC change on rulespec-us base
  `187d8d8e`: page 159 entered/page 369 left the ProgramSpec scope and all
  three `SNAP-SC-UTIL` worklist rows changed to `merged`.
- Captured the literal pinned-bridge fail-first: exit 1 at the reported
  `path.relative_to(manifest_root)` line because the root ProgramSpec was
  rebased against `<scratch>/us`.
- Applied the minimal routing/citation backport to a disposable pinned checkout,
  followed by a scratch-only version bump required by its clean-provenance
  gate; the campaign interpreter then signed the reconstructed diff.
- Confirmed the v1 output shape: checkout-root
  `.axiom/encoding-manifests/programs/us-sc/snap/fy-2026.json`, bare
  `programs/us-sc/snap/fy-2026` citation, and checkout-relative applied path.
- Committed the manifest only in the disposable rulespec clone and passed the
  separate external `guard-generated --roots programs` check.
- Confirmed the source `wt-snap-sc` worktree remains unchanged apart from its
  two pre-existing untracked report files.

## Next

- Run the full repository test, Ruff, compile, and changelog/version gates.
- Complete the independent review-fix cycle and rerun affected checks.
- Finalize this progress ledger, verify the origin/main diff, and write the
  untracked worker report.
