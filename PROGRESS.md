# Optional retired-manifest inventory progress

## State

Resumed from clean head `5db9f866` with the prior implementation and tests
preserved. The implementation is not approved: independent review found that
the absent-from-HEAD worktree check can fail open because it uses path-based
component probes. A descriptor-relative, fail-closed replacement and focused
race/error coverage are now required before broader validation or publication.

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
- Re-read the repository instructions, exact four-commit branch history and
  diff, current implementation/tests, and the independent review report at
  `20260830-194816-delegate-output-gntsoo-4-md/out.md`.
- Confirmed the branch is clean at `5db9f866`, four commits ahead of the local
  `origin/main` reference, with no prior work to discard or recreate.
- Confirmed the outstanding P2: `_checkout_path_exists_without_indirection`
  still relies on `Path.lstat()` component probes and cannot prove absence
  safely across appearance races, symlinked ancestors, or inspection errors.
- Attempted the repository-graph debugging workflow; its query tools are not
  available in this session, so direct source, call-site, and Git inspection is
  the active fallback.

## Next

- Replace the absent-from-HEAD proof with descriptor-relative traversal rooted
  at a verified repository directory descriptor and one fail-closed leaf
  metadata probe.
- Add deterministic adversarial coverage for appearance, ancestor replacement,
  symlink, dangling-link, non-regular leaf, and unexpected syscall errors.
- Run focused and broad validation, then complete the required independent
  review-fix cycle with no actionable findings.
- Verify the exact commits and draft PR body; only then push/open a draft PR,
  without dispatching signing or merging, and write the final output report.
