# Issue 1312 progress

## State

The checkout-root routing and ProgramSpec citation fix is implemented and
passes its focused current-main regression and compatibility controls. Current
`main` removed the public `sign-applied-files` command in the signing hard cut,
so literal command-level failure/pass and the held-out SC change are next being
verified against a scratch copy of the pinned bridge.

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

## Next

- Port the minimal fix into a scratch pinned bridge and capture literal
  `sign-applied-files` fail/pass evidence.
- Reconstruct the held-out SC ProgramSpec/worklist change in a scratch
  rulespec-us checkout, sign it, and run `guard-generated`.
