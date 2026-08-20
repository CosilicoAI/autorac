# EncodeBench

How to measure encoder models against each other on a fixed encoding task
set, and how to read the result. This formalizes the 2026-07-10 encoder
bake-off (8 US citations x 4 models, run from a session scratchpad) as a
durable instrument: EncodeBench, the sister benchmark to PolicyBench.
Issue #1189 tracks it; encodebench.org displays the board.

## What it is

- **Suite**: `benchmarks/encodebench_uk_v1.yaml` — 16 stratified UK
  cases: parameters and thresholds, bracket structure, phaseouts and tapers,
  cross-person and structural mechanics, a grounding trap (a section whose
  rates live in regulations — any rate literal is fabricated), a 2026-act
  recency probe, and three repo-augmented oracle candidates.
- **Board**: `axiom-encode eval-board` — folds one or more `eval-suite`
  outputs into an N-runner leaderboard plus a per-case P/F grid.
- **Headline metric**: deterministic gate-pass rate. A case passes when the
  encode succeeded and the artifact compiled, passed CI, and contains zero
  ungrounded numeric literals. The generalist-reviewer and oracle columns are
  advisory context, never the headline.

## Why the shape

- **UK, not US**: capability runs need a strict 3-key release repo so every
  run binds the same immutable corpus release (`uk-rulespec-2026-07-14`).
  rulespec-us is still pre-migration; the US continuity suite (the bake-off's
  eight citations, including the IL flat-rate fabrication case and the
  1402/71.06 structural canaries) follows the US release cut.
- **Cold mode**: repo-augmented workspaces deliberately include the merged
  target module as precedent context. Right for refresh gates; for a
  capability score it would measure copying and drift as the repo grows. Cold
  workspaces still carry definition stubs and canonical-concept context,
  identically for every runner.
- **Rate gates pinned to 0.0**: the manifest loader refuses omitted or null
  rate gates (no omission-based readiness bypasses), and a capability suite
  must never fail a model run — the board reports rates; `min_cases` is the
  only live gate.
- **Judge caveat**: the generalist reviewer is a pinned Claude CLI model,
  identical for every contestant. It is one judge, it is noisy, and when
  Claude-family models are on the board it is not a neutral party — which is
  why it never folds into the headline.

## Running it

Prerequisites (all local):

- `axiom-corpus` checkout with the pinned release materialized under
  `releases/uk-rulespec-2026-07-14` (`scripts/materialize_corpus_release.py`
  against the rulespec-uk toolchain pin).
- `rulespec-uk` checkout on the commit whose toolchain pins that release.
- `axiom-rules-engine` checkout with a release build (`cargo build --release
  --locked`) or an `axiom-encode engine-bind` receipt. When no engine pin is
  declared, resolution prefers release over debug; a debug-only build still
  binds, with an unverified-binding warning.
- A trusted supervisor attaching the signing broker (release verification
  reads the corpus-release public key from the broker; nothing here signs).
- Backend CLIs logged in: `codex` (ChatGPT subscription) and `claude`
  (subscription or a lane token via `CLAUDE_CODE_OAUTH_TOKEN`). The metered
  `openai:` backend is not part of the roster and needs explicit per-run
  approval.

Serial run (whole roster from the manifest, resumable):

```bash
uv run axiom-encode eval-suite benchmarks/encodebench_uk_v1.yaml \
  --output results/capability-v1/all \
  --corpus-path ../axiom-corpus \
  --policy-repo-path ../rulespec-uk \
  --axiom-rules-engine-path ../axiom-rules-engine \
  --auto-resume-attempts 2
```

Parallel runs (one process per runner, then fold): copy the manifest once
per runner, keep `cases` and `gates` byte-identical, and reduce `runners` to
a single entry. Case identities exclude the runner list, so single-runner
variants fold cleanly. Pin each codex process to its own `CODEX_HOME` lane
and each claude process to its own lane token; run each with its own
`--output` directory.

```bash
axiom-encode eval-board results/capability-v1/terra results/capability-v1/sol \
  results/capability-v1/gpt-5.5 results/capability-v1/luna \
  results/capability-v1/fable \
  --markdown-out results/capability-v1/board.md \
  --csv-out results/capability-v1/grid.csv
```

The board refuses inputs whose suite name, ordered case identities, corpus
release identity, or score-affecting execution identity differ, and refuses
the same runner name from two inputs (two runs of one runner are two
boards, not one). Execution identities are compared after dropping
checkout-location fields, so the same toolchain at different paths folds;
any real difference — encoder version or working tree, rules-engine
checkout, RuleSpec content/toolchain/waiver state, PolicyEngine runtime —
refuses the fold unless `--allow-mixed-toolchains` records the mismatch in
the board output. `--allow-partial` folds incomplete runs, rendering
missing cells as not run; ranking orders by gate-pass rate first, so a
partial run's raw pass count never outranks a complete run's rate. The
board also cross-checks every payload internally: only the v5 results
schema is accepted, execution-identity and result digests are recomputed,
result rows must belong to runners the same payload declared (name,
backend, and model), case identities in each row must match the manifest,
the `coverage.complete` claim is verified against the actual result matrix
in both directions, and malformed metric types are refused rather than
reinterpreted.

## Adding a model

1. Copy the manifest to a single-runner variant naming the new model
   (`newmodel=codex:some-model` or `newmodel=claude:some-model`).
2. Run `eval-suite` for that variant against the same checkouts. If the
   corpus release pin, encoder version, or any other score-affecting
   toolchain input has moved since the incumbents ran, the board will
   refuse the fold — re-run the full roster on the new toolchain instead
   of mixing runs (or fold deliberately with `--allow-mixed-toolchains`,
   which prints the mismatch on the board).
3. Fold the new output directory into the board alongside the existing ones.

## Verifying model resolution

A run is only evidence about the model that actually served it. The suite
records per-result backend/model and full traces under each case output
directory; before quoting a board, spot-check traces for the requested model
id (codex JSONL records the serving model; claude JSON output records the
resolved model). The 2026-07-10 bake-off verified all 32 runs resolved to
the requested model — keep that bar.

## Changing the suite

Any case edit (adding oracle wiring, changing mode or context) changes that
case's identity and therefore starts a new board — old results stop folding,
by design. Batch such changes: the planned oracle flip (verifying the
policyengine-uk runtime path, then setting `oracle: policyengine` on the
three oracle-candidate cases) should land as one revision, as
`encodebench_uk_v2` if v1 boards are still accumulating runners.

## Relation to existing instruments

- The refresh/seed/repair suites in `benchmarks/` are bulk-readiness gates:
  single runner, 1.0 thresholds, repo-augmented by design. This suite is a
  measurement instrument; it gates nothing.
- `eval-suite-report` stays the deep pairwise comparison (it revalidates
  verdicts); `eval-board` is the roster fold.
- Issue #29's harness-optimization loop would use a fixed suite like this
  one as its evaluation target; this suite does not implement that loop.
