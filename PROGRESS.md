# Optional retired-manifest inventory progress

## State

The recovered descriptor-relative implementation and deterministic adversarial
suite are committed at `026a5b5a`, with formatter follow-up `4737e6e3` and
version-contract follow-up `58e85a4d`, on
`fix/optional-retired-manifest-inventory`; the worktree is clean. The helper now
selects the repository root with one atomic no-follow directory open and pins
all later ancestor/leaf inspection to descriptors, so a root replacement after
that open cannot redirect traversal. The local `origin/main` remains
`f1bfe0a4`; live fetch, GitHub CLI, and GitHub page queries are currently
blocked by DNS/network access. Focused and workflow validation are green, and
the independent re-review approves with no actionable findings. The required
encoder version is now `0.2.1751`; full-suite validation and live publication
checks remain required before publication.

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
- Replaced every checkout ancestor probe with descriptor-relative opens using
  `O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC`, followed by one no-follow leaf
  metadata probe where only `ENOENT` proves absence; unsupported capability
  and every other syscall result fail closed.
- Removed an avoidable repository-root stat/open pair in favor of one atomic
  no-follow root open plus `fstat`, which supplies the traversal linearization
  point and pins the selected root across rename/replacement.
- Added deterministic coverage for clean Belgium absence, ignored regular and
  dangling-symlink leaves, appearance before the sole leaf probe, symlinked and
  replaced ancestors, symlinked and replaced repository roots, root/ancestor/
  leaf errors, directory/FIFO leaves, and unsupported descriptor capability.
- Added instrumentation proving malformed/non-normal apply manifests and
  wrong, missing, or duplicate target bindings fail before optional inventory
  probing.
- Passed 36 focused retired-inventory/descriptor tests, focused Ruff,
  `compileall`, and `git diff --check`; committed the source/test step as
  `026a5b5a` (`Harden optional retired inventory traversal`).
- Passed the three exact targeted-reencode workflow checks: shell syntax, exact
  one-canonical/two-replacement invocation binding, and executable canonical
  refresh ordering (`3 passed`).
- Passed repository-wide Ruff lint and compileall. Applied only Ruff's
  mechanical wrapping in the two touched files, re-passed the 36 focused tests,
  focused lint/format/compile/diff checks, and committed it as `4737e6e3`
  (`Format optional inventory hardening`).
- Completed the independent review-fix cycle: the reviewer approved the atomic
  root-open linearization, descriptor-relative ancestors, single leaf probe,
  fail-closed errors/capabilities, proof ordering, and adversarial coverage with
  no actionable findings; their independent revised root/error run passed 5
  tests.
- Started the full configured 14,003-test suite and stopped after 765 passes,
  one skip, and one expected failure identified the repository's mandatory
  version-provenance gate; no other failure had appeared. This superseded-tree
  run is diagnostic only and is not a final green claim.
- Synchronized the encoder version from `0.2.1750` to `0.2.1751` across
  `pyproject.toml`, the package, `uv.lock`, and all exact version fixtures, then
  committed it as `58e85a4d` (`Bump encoder version to 0.2.1751`). The version
  provenance, package metadata, exact registry, and synchronized fixture checks
  pass (`29 passed`).
- Recorded that ordinary `uv run` now attempts an editable rebuild and cannot
  download the declared Hatchling build requirement while DNS is unavailable;
  final checks therefore use `uv run --no-sync` with the existing project
  environment and configured `pythonpath = ["src"]`.

## Next

- Run broader prepare/signing-supervisor tests and full pytest on the committed
  `0.2.1751` tree, then repeat the
  repository-wide Ruff/format, compileall, status, and diff checks on the final
  tree.
- Retry live upstream comparison, verify the exact commits and draft PR body,
  and only then push/open a draft PR if all checks are green. Do not dispatch
  signing or merge; write the final report to the task output file.
