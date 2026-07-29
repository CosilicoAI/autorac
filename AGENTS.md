# Axiom Encode Agent Instructions

## Workflow Scope

- Use this repository's docs, CLI commands, RuleSpec specs, oracle registry, eval suites, and validation pipeline as the operating procedure for Axiom work.
- Name generated concepts per the applicable section of the naming contract, `axiom-rules-engine/docs/concept-naming.md` (executable rules, relations, tables/selectors, acronyms). Nothing in `encode --apply` updates the display acronym registries: when a change introduces a new acronym token, the PR author or reviewer flags it per the contract's Acronyms section and closes the flag by updating both registries or linking an owned follow-up.
- Do not use PolicyEngine workflow skills or PolicyEngine implementation skills for Axiom encoding, RuleSpec, corpus, or oracle-parity tasks. PolicyEngine is an oracle/comparison dependency here, not the workflow owner.
- If a reusable workflow is missing, add or propose an Axiom-specific Codex skill or project instruction instead of borrowing a PolicyEngine skill.

## PR Discipline

- Every PR to `TheAxiomFoundation/axiom-encode` must go through `/cycle` or an equivalent review-fix loop before it is marked ready for review, and again before merge if substantive changes were made after review. Draft PRs may be opened before the cycle when needed to get CI or review context.
- A valid cycle means: request an independent review, fix actionable findings, rerun the relevant focused tests/checks, and repeat until there are no actionable findings or a documented blocker.
- A blocker permits stopping the cycle, but it does not permit merging with unresolved actionable findings unless the PR records an explicit user or maintainer override.
- Apply this even to prompt, oracle, harness, and validation changes. For docs-only changes, run a lightweight cycle and record the cycle outcome and checks in the PR description or final handoff comment.
- Do not merge encoder PRs while CI is red, stale, or still pending unless the user explicitly overrides that for an emergency.

## Local Checks

```bash
uv run ruff check pyproject.toml src/axiom_encode scripts tests
python -m compileall -q src/axiom_encode scripts
uv run pytest
```

Use focused tests for small changes, then broaden checks when the change touches shared encoder behavior.
