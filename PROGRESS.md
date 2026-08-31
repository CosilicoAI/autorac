# encode#1559 sol-review fixes — progress

Branch `wt/encode-1559` → `origin/fix/editorial-numeric-recall` (PR #1559).
Base head: b2fd624d ("Align Armenian prompt with recall grammar").

## Scope
Three sol-review defects in Armenian ARLIS editorial-history stripping
(`src/axiom_encode/harness/source_completeness.py`).

## Done
- [x] HIGH — nested-parenthetical stripping. Depth gate: only candidates whose
      opening `(` sits at depth zero are stripped. Depth map built lazily.
- [x] MEDIUM — malformed token sequences. Years `\d{2}(?:\d{2})?`; the history
      must fullmatch a separator-delimited action/citation token sequence, which
      bounds glued citations and glued uppercase actions.
- [x] MEDIUM — `;` accepted as a separator, matching prompt line 1172.
- [x] Extra (same defect class as #2): scoped `re.IGNORECASE` to the action
      alternation via `(?i:...)`. The blanket flag also folded the citation
      token, admitting lowercase `հօ-538-ն` identifiers.
- [x] Separator made possessive — behavior-preserving, removes a quadratic
      backtracking path.
- [x] Version ratchet 0.2.1753 across pyproject/`__init__`/uv.lock + test pins
      (origin/main is 0.2.1750, so 1753 stands). Prompt text unchanged, so no
      prompt-generation bump.

## Evidence
- `tests/test_source_completeness.py`: 6186 passed (46 Armenian, was 29).
- `tests/test_rulespec_validation.py`: 2519 passed.
- `tests/test_cli.py` + `tests/test_complete_source_mode_plumbing.py`: 1332 passed.
- `uv run ruff check .` clean; `git diff --check` clean.
- Differential vs an independently written parser of the prompt contract:
  2,000,376 generated bodies, 0 disagreements.
- Differential possessive-vs-greedy separator: 262,264 histories, 0 mismatches.

## Next
Push to `origin/fix/editorial-numeric-recall`. Do not merge.
