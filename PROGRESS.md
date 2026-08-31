# Optional retired-manifest inventory progress

## State

The optional retired-manifest inventory exception is implemented and committed
on local `origin/main` at `f1bfe0a4`. Focused regression, adversarial, workflow,
lint, and compile checks pass. Live upstream comparison remains blocked by
sandbox DNS, so the terminal version bump and GitHub publication are pending.

## Done

- Inspected branch status, all untracked paths, local refs, merge base, recent
  history, and the complete surviving diff before editing.
- Preserved the recovered edits in
  `src/axiom_encode/prepare_signed_backfill.py` and
  `tests/test_prepare_signed_backfill.py`.
- Started independent logic, workflow, and adversarial read-only audits.
- Added a rulespec-be regression proving a valid signed replacement no-ops only
  when `tests/test_encoding_manifests.py` is absent from both HEAD and the
  worktree.
- Added ignored regular-file and dangling-symlink worktree cases, plus an
  ambiguous `lstat` probe that must fail closed.
- Proved invalid normal model-apply shape and wrong, missing, or duplicate
  target bindings fail before the optional-file exception.
- Strengthened workflow assertions to bind one canonical-refresh invocation
  and two replacement invocations to their exact checkout/target arguments.
- Replaced the racy `exists()`/`is_symlink()` pair with the repository's
  symlink-free worktree path inspection helper.
- Added the fixed changelog fragment and committed the implementation.
- Passed 21 reconciliation tests, the eight new focused checks, focused Ruff,
  compileall, and `git diff --check`.
- Passed the canonical-refresh workflow execution test with `PYTHONPATH=src`;
  without that override the sandbox's stale installed `corpus_resolver` lacks
  the current `required_mode` argument.
- Completed three read-only audits; their actionable test and two-probe race
  findings are addressed.

## Next

- Fetch and compare live `origin/main`, rebase if needed without losing WIP,
  and apply the live-main-next synchronized encoder version bump.
- Run relevant broader tests and full Ruff/compile/diff hygiene.
- Complete the required post-implementation independent review-fix cycle.
- Write the final output report, verify commit and PR text, push, and open a
  draft PR without dispatching signing jobs or merging.
