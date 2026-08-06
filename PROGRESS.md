# Retry-diagnostics reconciliation progress

## State

The semantic rebase onto `origin/main` (`2438133d`) is complete. Reconciled
functional commits are `a5383cb9` and `debbb5b8`. The requested regression and
affected-suite matrix, static checks, current-main comparison, and independent
review cycle are complete with no actionable implementation findings.

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
- Passed all nine reconciled regression tests together.
- Passed the exact historical 439-test selection, expanded by current main's
  #1419/#1420 deferral cases to 2,858 tests.
- Passed the touched oracle-registry suite (24 tests). The full RuleSpec suite
  had 1,464 passes plus the same sandbox-blocked `/var/tmp` alias test as main
  (main: 1,463 passes plus that failure).
- The full CLI suite had 1,174 passes and one skip. Its only exceptions were the
  required-version provenance guard (expected because this task prohibits a
  version bump) and one unrelated migration test that passed alone; pristine
  main passed 1,172 tests with one skip. The changed retry/overlay nodes all
  passed.
- Pristine main's full complete-source, deferral, and touched registry matrix
  passed all 2,868 tests. The rebased tree's corresponding files and new tests
  all passed through the affected selection and registry run.
- Ruff and compileall passed. Diff whitespace, conflict-marker, and
  version/dependency-pin audits passed.
- Three independent final reviewers found no actionable semantic, correctness,
  history, or test-coverage issue. They confirmed items (a)-(d), candidate
  binding, compile-gated runtime execution, and unchanged #1419/#1420 binding.

## Next

- Remove `PROGRESS.md`, write the final output report, and verify a clean tree.
