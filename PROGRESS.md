# Retry-diagnostics reconciliation progress

## State

The semantic rebase onto `origin/main` (`2438133d`) is complete. Reconciled
functional commits are `a5383cb9` and `debbb5b8`. Focused conflict-resolution
checks pass; the requested regression/suite matrix and independent review are
next.

## Done

- Rebased stale commits `1be58581` and `bb07d753` onto current main.
- Resolved real overlap in `cli.py`, `evals.py`, `test_cli.py`,
  `test_complete_source_mode_plumbing.py`, and `test_source_completeness.py`.
- Kept main's #1418 `_FailedEncodeAttempt.validation_issues`, bounded issue
  snapshots, latest-candidate-only feedback, candidate capture, and fail-closed
  retry cleanup; dropped the stale parallel `issues` field and historical
  retry aggregation.
- Kept main's #1419/#1420 deferral-path and branch binding unchanged.
- Added numbered `Fix ALL of the following` retry feedback while preserving
  main's 12-item/16K-item/64K-total bounds and candidate-specific wording.
- Updated apply-overlay validation to collect every ordered validator issue,
  fall back to scalar errors, and deduplicate identical file/detail pairs
  across validators.
- Updated RuleSpec CI to co-report missing tests and static companion-test,
  exception, and zero-branch diagnostics despite compile or earlier failures;
  runtime test execution still requires a compiled artifact.
- Confirmed main has one production retry-record producer and it already
  populates candidate-bound `validation_issues`; rejected the stale union that
  would reattach pre-repair metrics to a post-repair candidate.
- Recast that union regression to prove final overlay issues replace stale
  standalone diagnostics.
- Passed seven first-commit focused regressions after adapting main's retry
  fixture, plus the two second-commit focused regressions and collector check.
- Confirmed marker-free syntax with an existing compatible Python environment.

## Next

- Run all nine reconciled regression tests together.
- Run the 439-test affected slice and suites covering #1418–#1420; establish a
  current-main baseline for any failures.
- Run Ruff, compileall, diff/marker/version-pin checks, and an independent
  review-fix cycle.
- Remove `PROGRESS.md`, write the final output report, and verify a clean tree.
