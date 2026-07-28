# Issue 1312 progress

## State

The root/placement/citation failure is reproduced and regression coverage is
red before implementation. Current `main` removed the public
`sign-applied-files` command in the signing hard cut, so the committed tests
cover its surviving shared helpers/writer and the literal command-level
failure/pass is being verified against a scratch copy of the pinned bridge.

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

## Next

- Implement checkout-owned source-root placement and bare ProgramSpec citation
  without changing jurisdiction-prefixed or UK country-monorepo behavior.
- Add the changelog entry, rerun focused tests, and port the minimal fix into a
  scratch pinned bridge for literal command and real-checkout verification.
