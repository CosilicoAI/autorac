# Optional retired-manifest inventory progress

## State

Resumed on `fix/optional-retired-manifest-inventory` at review checkpoint
`4602eaf2`. The working-tree implementation and tests are byte-for-byte the
tree saved by salvage ref
`refs/codex-salvage/fix-optional-retired-manifest-inventory-20260830-212800-64246`
(`7a6782b9`) and remain deliberately uncommitted while they are audited. The
local `origin/main` is `f1bfe0a4`; a live fetch and GitHub query were attempted
but are currently blocked by DNS/network access. Independent review rejected
the prior path-based absence proof, so the recovered descriptor-relative,
fail-closed replacement and adversarial tests must be validated and reviewed
before publication.

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
- Read the complete continuation brief and exact independent `REQUEST CHANGES`
  report. The finding is limited to the fail-open worktree absence probe and
  missing race/symlink tests; the report confirms the clean-absence and normal
  tracked/dirty/malformed proof surfaces otherwise remain sound.
- Inspected every branch commit and the salvage commit metadata, confirmed the
  dirty source/test tree exactly matches salvage `7a6782b9`, and preserved it
  without duplication or loss.
- Attempted both `git fetch origin --prune` and a GitHub PR query; both failed
  before changing state because this environment cannot currently resolve
  GitHub hosts.

## Next

- Audit and, where needed, correct the recovered descriptor-relative traversal
  and deterministic appearance, ancestor-replacement, symlink, dangling-link,
  non-regular-leaf, unsupported-platform, and syscall-error tests.
- Commit the coherent implementation/test step after its focused checks pass,
  then update and commit this ledger after each subsequent coherent step.
- Run focused and broad validation, then complete the required independent
  review-fix cycle with no actionable findings.
- Retry live upstream comparison, verify the exact commits and draft PR body,
  and only then push/open a draft PR if all checks are green. Do not dispatch
  signing or merge; write the final report to the task output file.
