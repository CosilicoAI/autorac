# Issue 1312 progress

## State

Work has started on an isolated `fix/1312-program-spec-manifest-root`
worktree created from `origin/main`. The reported root-level RuleSpec source
path failure still needs to be reproduced and traced before implementation.

## Done

- Created the requested local branch and isolated worktree from `origin/main`.
- Read the repository instructions and applicable debugging/GitHub workflows.
- Established the fail-first, implementation, real-checkout, compatibility,
  validation, and independent-review plan.

## Next

- Inspect issue 1312, rulespec-us PR 1139, a committed root-level program
  manifest, and the current root/citation helpers.
- Add a regression that reproduces the `programs/us-sc/snap/fy-2026.yaml`
  signing crash and capture its failure.
