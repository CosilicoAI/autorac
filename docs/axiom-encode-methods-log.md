# Axiom Encode Methods Log

> Historical log. Paths name the repo state at each entry's date; artifacts since removed appear as plain code spans and resolve through git history, not links.

This file is the paper-oriented change log for Axiom Encode. It records benchmark-relevant harness changes, the hypothesis behind each change, and the evidence path that should justify later claims in a paper, talk, or grant update.

## Update Protocol

- Add an entry when a change can affect benchmark outcomes, promotion decisions, reviewer behavior, oracle behavior, cost, or artifact observability.
- Record the first enabling commit plus any especially important follow-up commits when a cluster of small changes belongs to one methodological step.
- Prefer benchmark manifests, run-ledger artifact names, and repo paths over chat summaries.
- Do not use this file for pure refactors that cannot change measured behavior.

## Current Baseline Facts

As of 2026-04-10:

- Checked-in jurisdictional RuleSpec corpus: 483 files.
- Repo counts:
  - `rulespec-uk`: 146
  - `rulespec-us`: 266
  - `rulespec-us-ca`: 19
  - `rulespec-us-co`: 18
  - `rulespec-us-ny`: 15
  - `rulespec-us-tx`: 11
  - `rulespec-ca`: 8
- Recent UK benchmark coverage across the main bulk waves: 165 cases.
  - uk_wave18_soak.yaml (`../benchmarks/uk_wave18_soak.yaml`): 25
  - uk_wave19_scale_seed.yaml (`../benchmarks/uk_wave19_scale_seed.yaml`): 32
  - uk_wave20_bulk_seed.yaml (`../benchmarks/uk_wave20_bulk_seed.yaml`): 55
  - uk_wave21_bulk_seed.yaml (`../benchmarks/uk_wave21_bulk_seed.yaml`): 53
- Core UK oracle readiness lane: 11 PolicyEngine-backed cases in uk_readiness.yaml (`../benchmarks/uk_readiness.yaml`).
- Broader UK oracle readiness lane: 33 PolicyEngine-backed cases in uk_policyengine_readiness.yaml (`../benchmarks/uk_policyengine_readiness.yaml`).
- Durable suite artifacts now expected for eval-suite runs:
  - `suite-run.json`
  - `suite-results.jsonl`
  - `results.json`
  - `summary.json`

## 2026-05-04: Eval-backed encode runs gained linked session telemetry and repair manifests

- Primary commit: `c6f21b2` `Log encode sessions for eval runs`
- Hypothesis:
  - Run health, token usage, and failure repair context should be visible from the same operational path. A failed encode should not require manually correlating local DB rows, traces, and generated YAML before a repair pass can start.
- Effect:
  - `axiom-encode encode` now writes `EncodingRun.session_id` as `encode-<run_id>`.
  - Each eval-backed encode creates a linked SDK-style session with request, result, and issue events, model/version metadata, and token counts.
  - Supabase sync uploads the encoding run and the linked session together when write credentials are configured.
  - Failed eval-backed runs emit a sibling `*.repair.json` manifest with the run ID, session ID, citation, trace path, output path, and concrete inspect/repair/rerun actions.
- Evidence:
  - Full local suite: `uv run pytest -q` reported `465 passed`.
  - Live smoke run `600fa816` for `us/guidance/usda/fns/snap-fy2026-cola/page-2` was backfilled to `session_id=encode-600fa816` and synced to Supabase with 3 session events and token columns.
  - Production `/ops` showed the synced encoder run and SDK-session data after the backfill.
- Primary evidence paths:
  - [cli.py](../src/axiom_encode/cli.py)
  - [encoding_db.py](../src/axiom_encode/harness/encoding_db.py)
  - [supabase_sync.py](../src/axiom_encode/supabase_sync.py)
  - [test_cli.py](../tests/test_cli.py)
  - [test_encoding_db.py](../tests/test_encoding_db.py)

## 2026-05-09: Encode logs distinguish raw validation from final apply outcome
- Hypothesis:
  - Runs that fail standalone validation but pass policy-overlay validation and `--apply` should not look failed in run telemetry. Operators still need the raw standalone result, but the public run status should reflect the final workflow outcome.
- Effect:
  - Local `encoding_runs` now stores `outcome_json` with standalone validation, overlay validation, apply status, final status, and applied files.
  - Eval-backed sessions now log an `encode_outcome` event after apply/validation completes.
  - `encode_issue` events and `*.repair.json` manifests are only emitted for final failures; a successful overlay apply suppresses the misleading repair path.
  - Supabase `encodings.encoding_runs` gained an `outcome` JSONB column and the sync path sends it, with a fallback for deployments that have not applied the migration yet.
- Evidence:
  - Full local suite: `uv run pytest -q` reported `509 passed`.
  - Lint: `uv run ruff check .` passed.
- Primary evidence paths:
  - [cli.py](../src/axiom_encode/cli.py)
  - [encoding_db.py](../src/axiom_encode/harness/encoding_db.py)
  - [supabase_sync.py](../src/axiom_encode/supabase_sync.py)
  - [005_encoding_run_outcomes.sql](../migrations/005_encoding_run_outcomes.sql)

## Backfill: 2026-03-29 to 2026-04-10

### 2026-03-29: UK oracle bridge became a first-class harness path

- Primary commit: `1d1a3cf` `Wire UK oracles into eval harness`
- Important follow-ons:
  - `727ed2a` `Harden UK PolicyEngine bridge for benchmark artifacts`
  - `fb718e0` `Tighten UK PolicyEngine benchmark bridge`
  - `e8454c4` `Add UK child benefit regulation 2(1)(b) oracle support`
  - `f304fff` `Add UK pension credit minimum guarantee oracle support`
  - `6f116a3` `Add Scottish Child Payment UK oracle support`
  - `5a77922` `Add UK benefit cap oracle support`
- Hypothesis:
  - Existing UK microsimulation outputs can serve as a fast external comparator for a subset of RuleSpec leaves, especially fixed amounts, rates, caps, and standard elements.
- Effect:
  - PolicyEngine UK became a real oracle lane rather than an aspirational integration.
  - UK source slices gained variable hints and comparability logic for mapped programs.
- Primary evidence paths:
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - uk_expanded.yaml (`../benchmarks/uk_expanded.yaml`)

### 2026-03-30: Readiness-gated eval suites replaced ad hoc spot-checking

- Primary commit: `a097e7c` `Add readiness-gated eval suites`
- Important follow-ons:
  - `3e8af9a` `Harden UK readiness harness and atomic benchmarks`
  - `950ef23` `Add oracle target hints for UK source benchmarks`
  - `ccf9586` `Focus UK row-slice benchmarks on atomic amounts`
- Hypothesis:
  - Bulk progress should be measured by explicit suite gates rather than hand-picked examples.
- Effect:
  - Benchmark manifests became the unit of readiness.
  - Success, compile, CI, zero-ungrounded, oracle, and cost metrics became suite-level gates.
- Primary evidence paths:
  - uk_starter.yaml (`../benchmarks/uk_starter.yaml`)
  - uk_readiness.yaml (`../benchmarks/uk_readiness.yaml`)
  - [evals.py](../src/axiom_encode/harness/evals.py)

### 2026-04-01: Deterministic CI and generalist review became materially stricter

- Primary commits:
  - `9b4b719` `Add generalist reviewer gate to eval suites`
  - `0a632b3` `Reject decomposed date scalars in evals`
  - `14e8306` `Resolve defined-term imports in UK evals`
  - `6b58c08` `Reject placeholder fact variables in UK evals`
  - `c63fc6d` `Tighten UK numeric occurrence coverage checks`
  - `f9468df` `Track repeated source numbers as named scalars`
  - `2996e08` `Require repeated-scalar UK sibling coverage`
- Hypothesis:
  - Many high-cost semantic misses were not compilation failures; they needed deterministic grounding checks plus a broader statutory-fidelity reviewer.
- Effect:
  - The harness began rejecting placeholder facts, decomposed dates, under-grounded repeated numbers, and weak import behavior.
  - Generalist review became a formal promotion gate rather than a sidecar note.
- Primary evidence paths:
  - [test_evals.py](../tests/test_evals.py)
  - [eval_prompt_surface.py](../src/axiom_encode/harness/eval_prompt_surface.py)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)

### 2026-04-04: Eval-suite runs became durable research artifacts

- Primary commit: `fdaf72e` `Persist eval-suite runs and normalize UK pence thresholds`
- Important follow-ons:
  - `d5fe8a1` `Retry transient UK eval suite case failures`
  - `de206b7` `Harden UK eval artifact salvage and rate grounding`
- Hypothesis:
  - Failed or interrupted runs still contain evidence and should not disappear into terminal history.
- Effect:
  - `eval-suite` began writing top-level state and per-case ledgers during execution.
  - This made it possible to justify later repairs from concrete artifacts rather than memory.
- Primary evidence paths:
  - [cli.py](../src/axiom_encode/cli.py)
  - [evals.py](../src/axiom_encode/harness/evals.py)

### 2026-04-07 to 2026-04-08: UK bulk evaluation moved from repair slices to scale waves

- Primary commits:
  - `d547287` `Add UK wave 20 bulk seed manifest`
  - `5a9de67` `Add UK wave 21 bulk benchmark seed`
  - `bead4c6` `Make UK wave 21 seed atomic`
- Hypothesis:
  - A broad benchmark wave should be decomposed into atomic slices so failures are diagnosable and repair manifests are narrow.
- Effect:
  - Wave 20 established a larger bulk benchmark.
  - Wave 21 was then broken into more atomic regulation 17 / 17A slices, which improved failure localization.
- Primary evidence paths:
  - uk_wave20_bulk_seed.yaml (`../benchmarks/uk_wave20_bulk_seed.yaml`)
  - uk_wave21_bulk_seed.yaml (`../benchmarks/uk_wave21_bulk_seed.yaml`)

### 2026-04-08 to 2026-04-10: Wave 21 semantics were repaired through targeted prompt rules

- Representative commits:
  - `70a0081` `Preserve qualifier scope in UK payment disjunctions`
  - `4079b3f` `Preserve residual sibling conditions in UK limbs`
  - `789e11b` `Add payment entity support to eval prompts`
  - `a554503` `Tighten UK omission and subject-to slice guidance`
  - `81e56d1` `Preserve UK complete-cycle averaging qualifiers`
  - `0c622f8` `Tighten UK exhaustive route selection guidance`
  - `f7daada` `Preserve royalty consideration scope in UK evals`
- Hypothesis:
  - The remaining misses were mostly not parser failures; they were recurring semantic compression patterns that could be attacked through narrower slice-aware rules.
- Effect:
  - Prompt-surface guidance became more explicit around subject-to clauses, residual limbs, claimant-or-partner disjunctions, payment-scoped outputs, route selection, deeming tails, and qualifier scope.
- Primary evidence paths:
  - [eval_prompt_surface.py](../src/axiom_encode/harness/eval_prompt_surface.py)
  - wave 21 repair manifests under [benchmarks](../benchmarks)

### 2026-04-10: Local GPT evaluation default moved to Codex, and interrupted suites became resumable

- Primary commits:
  - `e0e628a` `Default GPT evals to Codex`
  - `7ac9763` `Add eval-suite resume support`
- Hypothesis:
  - Local bulk measurement should be cheaper and more robust before another round of scale runs.
- Effect:
  - GPT runners default locally to Codex unless explicitly overridden.
  - Interrupted `eval-suite` runs can resume in place from `suite-results.jsonl`.
- Primary evidence paths:
  - [cli.py](../src/axiom_encode/cli.py)
  - [evals.py](../src/axiom_encode/harness/evals.py)

### 2026-04-10: UK PolicyEngine lane was promoted from implicit support to named readiness manifests

- Primary commit: `71da5e8` `Promote UK PolicyEngine readiness manifests`
- Hypothesis:
  - If a manifest carries a PolicyEngine gate, the cases in that manifest should actually be oracle-backed and reproducible as a named lane.
- Effect:
  - The core UK starter and readiness manifests were upgraded to real UK PolicyEngine oracle suites.
  - A broader 33-case UK oracle regression lane was added.
- Primary evidence paths:
  - uk_starter.yaml (`../benchmarks/uk_starter.yaml`)
  - uk_readiness.yaml (`../benchmarks/uk_readiness.yaml`)
  - uk_policyengine_readiness.yaml (`../benchmarks/uk_policyengine_readiness.yaml`)

### 2026-04-10: Eval-suite runs gained a durable local archive registry

- Hypothesis:
  - If benchmark artifacts remain only in `/tmp`, they are too easy to lose and
    too hard to cite consistently in later methods or appendix work.
- Effect:
  - `axiom_encode eval-suite-archive` now snapshots a suite output tree into a stable
    local archive under `artifacts/eval-suites` by default.
  - Archived `results.json`, `summary.json`, and `suite-results.jsonl` payloads
    are rewritten so artifact paths point at the archived copy rather than the
    original `/tmp` run directory.
  - Each archive writes `archive-metadata.json` and appends a compact record to
    `artifacts/eval-suites/index.jsonl`.
- Primary evidence paths:
  - [cli.py](../src/axiom_encode/cli.py)
  - [test_cli.py](../tests/test_cli.py)
  - [README.md](../README.md)

### 2026-04-11: Colorado repo-augmented validation exposed and repaired import-path mismatches

- Primary commits:
  - `33d3fef` `Fix Colorado repair harness prompts`
  - `8d4e569` `Fix Colorado import closure and prompt rails`
  - `0863433` `Terminate Codex evals after persistent output`
  - `8cc8bf4` `Preserve canonical context import paths`
  - `b9a0223` `Require unquoted RuleSpec import targets`
- Hypothesis:
  - The remaining Colorado Works misses were not mostly parser failures; they were a mix of repo-augmented harness mismatches and prompt-level import/test conventions that the model was inferring incorrectly.
- Effect:
  - The harness now preserves canonical import paths for sibling `rulespec-*` corpora instead of collapsing those files under `external/...`.
  - Validation import-closure handling was tightened for benchmark workspaces and mixed import-block syntax.
  - Prompt guidance became more explicit about direct factual inputs, chart-consistent tests, canonical import targets, and unquoted `imports:` entries.
  - Codex eval waiting logic now terminates once output has persisted long enough, reducing hangs after the model has already emitted a final bundle.
- Primary evidence paths:
  - [evals.py](../src/axiom_encode/harness/evals.py)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [us_co_colorado_works_leaf_repair.yaml](../benchmarks/us_co_colorado_works_leaf_repair.yaml)
  - us-co-colorado-works-leaf-repair1-20260411 (`../artifacts/eval-suites/us-co-colorado-works-leaf-repair1-20260411`)
  - us-co-colorado-works-leaf-repair2-interrupted-20260411 (`../artifacts/eval-suites/us-co-colorado-works-leaf-repair2-interrupted-20260411`)
  - us-co-colorado-works-leaf-repair3-interrupted-20260411 (`../artifacts/eval-suites/us-co-colorado-works-leaf-repair3-interrupted-20260411`)
  - us-co-colorado-works-leaf-repair4-usage-limit-20260411 (`../artifacts/eval-suites/us-co-colorado-works-leaf-repair4-usage-limit-20260411`)

### 2026-04-11: Colorado leaf repair lane closed cleanly after CI import-closure fix

- Primary commit:
  - `342a68a` `Run CI tests with import closure`
- Hypothesis:
  - A meaningful share of the remaining Colorado repair failures were harness artifacts from running the RuleSpec test runner against the benchmark output file without the copied import closure that structural validation already saw.
- Effect:
  - CI test execution now copies the root benchmark artifact, its companion `.test.yaml`, and its resolved import closure into a flattened temporary workspace before invoking the RuleSpec test runner.
  - The Colorado leaf repair rerun moved from mixed CI failures to a clean readiness pass on the full five-case repair manifest.
- Primary evidence paths:
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - [us_co_colorado_works_leaf_repair.yaml](../benchmarks/us_co_colorado_works_leaf_repair.yaml)
  - us-co-colorado-works-leaf-repair8-interrupted-20260411 (`../artifacts/eval-suites/us-co-colorado-works-leaf-repair8-interrupted-20260411`)
  - us-co-colorado-works-leaf-repair9-20260411 (`../artifacts/eval-suites/us-co-colorado-works-leaf-repair9-20260411`)

### 2026-04-11: Colorado K repair required effective-date test normalization and a reviewer fallback

- Primary commit:
  - `5440731` `Normalize eval test dates for Colorado K repair`
- Hypothesis:
  - The remaining `3.606.1(K)` failure was no longer a semantic truncation bug in the generated RuleSpec logic; it was a harness/materialization gap where companion `.test.yaml` periods could remain before the file's first explicit effective date, producing an empty compiled IR during CI.
- Effect:
  - Non-annual test-period normalization now also repairs explicit ISO day strings and YAML-parsed `date` values that fall before the earliest effective date.
  - Bundle materialization now matches generated files by candidate filename, so paths like `./9-CCR-2503-6-3.606.1-K.yaml` still pass the main-file content into `.test.yaml` normalization.
  - The narrow Colorado `K` repair manifest now closes cleanly with compile, CI, and generalist review all passing.
- Primary evidence paths:
  - [evals.py](../src/axiom_encode/harness/evals.py)
  - [test_evals.py](../tests/test_evals.py)
  - [us_co_colorado_works_leaf_k_repair.yaml](../benchmarks/us_co_colorado_works_leaf_k_repair.yaml)
  - us-co-colorado-works-leaf-k-repair3-openai-interrupted-20260411 (`../artifacts/eval-suites/us-co-colorado-works-leaf-k-repair3-openai-interrupted-20260411`)
  - us-co-colorado-works-leaf-k-repair4-openai-claude-20260411 (`../artifacts/eval-suites/us-co-colorado-works-leaf-k-repair4-openai-claude-20260411`)

### 2026-04-11: Federal SNAP reconstruction closed after prompt/validator alignment on annual amendment tables

- Primary commits:
  - `a7ec17f` `Teach SNAP table repairs to amend canonical outputs`
  - `3816c47` `Allow SNAP household-size table indices in CI`
  - `10491a2` `Treat SNAP additional-member math as table indexing`
  - `b49c5a1` `Avoid month-name false positives in date scalar checks`
- Hypothesis:
  - The remaining federal SNAP reconstruction misses were not substantive encoding failures in the statute slices themselves; they were harness mismatches around annual publication amendment structure, table-index numerics, and over-eager date-scalar heuristics.
- Effect:
  - The encoder prompt now explicitly instructs annual USDA parameter tables to amend canonical statute outputs instead of importing and redeclaring them locally.
  - Validator CI now treats household-size row labels and `household_size - 8` style additional-member arithmetic as structural schedule indexing rather than ungrounded free numerics.
  - Date-decomposition checks no longer misclassify ordinary benefit-period names like `initial_month` as calendar-month scalars.
  - The four-case federal SNAP reconstruction seed (`2017(a)`, `2017(c)(1)`, `2017(c)(3)`, and the FY2026 USDA allotment table) reached a clean ready state.
- Primary evidence paths:
  - [evals.py](../src/axiom_encode/harness/evals.py)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - [us_snap_federal_reconstruction_seed.yaml](../benchmarks/us_snap_federal_reconstruction_seed.yaml)
  - [us_snap_fy2026_cola_table_repair.yaml](../benchmarks/us_snap_fy2026_cola_table_repair.yaml)
  - us-snap-fy2026-cola-table-repair1-20260411 (`../artifacts/eval-suites/us-snap-fy2026-cola-table-repair1-20260411`)
  - us-snap-fy2026-cola-table-repair2-20260411 (`../artifacts/eval-suites/us-snap-fy2026-cola-table-repair2-20260411`)
  - us-snap-fy2026-cola-table-repair3-20260411 (`../artifacts/eval-suites/us-snap-fy2026-cola-table-repair3-20260411`)
  - us-snap-fy2026-cola-table-repair4-20260411 (`../artifacts/eval-suites/us-snap-fy2026-cola-table-repair4-20260411`)
  - us-snap-federal-reconstruction-seed-run7-20260411 (`../artifacts/eval-suites/us-snap-federal-reconstruction-seed-run7-20260411`)
  - us-snap-federal-reconstruction-seed-run8-20260411 (`../artifacts/eval-suites/us-snap-federal-reconstruction-seed-run8-20260411`)

### 2026-04-12: SNAP current-effective asset benchmark closed after publication-table and PE replay fixes

- Primary commits:
  - `b3edc32` `Add current-effective SNAP asset benchmark`
  - `38d50d6` `Derive SNAP asset replay inputs from exclusions`
  - `f3901c5` `Normalize monthly test periods for table updates`
  - `083f8f8` `Fix SNAP asset oracle replay periods`
- Hypothesis:
  - The remaining SNAP asset-test failures were not substantive disagreements on the FY2026 asset limits; they were harness mismatches across three layers: (1) a benchmark that needed a current-effective USDA publication slice instead of the bare statute text, (2) monthly test periods being normalized to day-level dates for a monthly oracle lane, and (3) PolicyEngine asset replay using month-keyed overrides for variables that PolicyEngine US expects at year granularity.
- Effect:
  - Axiom Encode now has a dedicated current-effective SNAP asset benchmark rooted in the FY2026 USDA COLA publication slice, with policyengine gating on the comparable downstream boolean `meets_snap_asset_test`.
  - Validator replay now derives `snap_assets` from total resources minus exclusions when needed, applies SNAP asset overrides at annual PolicyEngine periods, and ignores auxiliary `.test.yaml` outputs that do not resolve to the hinted oracle target.
  - Numeric grounding now ignores structural table headings and descriptive row-label ages while preserving the actual value rows.
  - Test-period normalization now keeps monthly rules at `YYYY-MM` instead of converting them to effective-date day strings.
  - The current-effective asset benchmark reached a clean ready state with success, compile, CI, zero-ungrounded numerics, generalist review, and PolicyEngine all passing.
- Primary evidence paths:
  - [us_snap_asset_test_current_effective_refresh.yaml](../benchmarks/us_snap_asset_test_current_effective_refresh.yaml)
  - [evals.py](../src/axiom_encode/harness/evals.py)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - asset-limits-current-effective.txt (`../../rulespec-us/sources/slices/usda/snap/fy-2026-cola/asset-limits-current-effective.txt`)
  - us-snap-asset-test-current-effective-refresh1-interrupted-20260412 (`../artifacts/eval-suites/us-snap-asset-test-current-effective-refresh1-interrupted-20260412`)
  - us-snap-asset-test-current-effective-refresh5-interrupted-20260412 (`../artifacts/eval-suites/us-snap-asset-test-current-effective-refresh5-interrupted-20260412`)
  - us-snap-asset-test-current-effective-refresh6-ready-20260412 (`../artifacts/eval-suites/us-snap-asset-test-current-effective-refresh6-ready-20260412`)

### 2026-04-12: SNAP current-effective eligibility benchmark closed after context and member-proxy repairs

- Primary commits:
  - `650c143` `Add SNAP eligibility refresh benchmark`
  - `d74096c` `Tighten SNAP eligibility eval guidance`
  - `a4c9641` `Refine SNAP eligibility oracle guidance`
  - `6f3a344` `Tighten SNAP eligibility prompt surface`
  - `eccd4ec` `Wire SNAP eligibility inputs into PE replay`
  - `2d259ab` `Align SNAP eligibility benchmark context`
  - `24da0b6` `Tighten SNAP eligibility member proxy handling`
  - `77b563d` `Handle SNAP eligibility proxy synonyms`
- Hypothesis:
  - The remaining `is_snap_eligible` failures were not disagreements on the current-effective federal SNAP eligibility composite itself. They came from benchmark-context drift, over-generic household member proxies in the generated artifacts, and a PE replay bridge that only recognized one earlier proxy spelling.
- Effect:
  - The benchmark now allows the direct copied statute component files for the income and asset tests rather than indirect annual wrapper files that did not export the imported eligibility booleans.
  - The prompt now rejects household count proxies and renamed household participation proxies in this oracle-backed eligibility lane, and it explicitly grounds `130 percent` as `1.3`.
  - PE replay now supports the SNAP eligibility component inputs directly and can synthesize the minimal person-level fallback facts from the older and renamed household proxy inputs when the model still emits them.
  - The current-effective SNAP eligibility benchmark reached a clean ready state with success, compile, CI, zero-ungrounded numerics, generalist review, and PolicyEngine all passing.
- Primary evidence paths:
  - [us_snap_eligibility_refresh.yaml](../benchmarks/us_snap_eligibility_refresh.yaml)
  - [evals.py](../src/axiom_encode/harness/evals.py)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - is_snap_eligible.txt (`../../rulespec-us/sources/slices/7-USC/snap/current-effective/is_snap_eligible.txt`)
  - us-snap-eligibility-refresh1-failed-20260412 (`../artifacts/eval-suites/us-snap-eligibility-refresh1-failed-20260412`)
  - us-snap-eligibility-refresh2-failed-20260412 (`../artifacts/eval-suites/us-snap-eligibility-refresh2-failed-20260412`)
  - us-snap-eligibility-refresh3-interrupted-20260412 (`../artifacts/eval-suites/us-snap-eligibility-refresh3-interrupted-20260412`)
  - us-snap-eligibility-refresh4-interrupted-20260412 (`../artifacts/eval-suites/us-snap-eligibility-refresh4-interrupted-20260412`)
  - us-snap-eligibility-refresh5-interrupted-20260412 (`../artifacts/eval-suites/us-snap-eligibility-refresh5-interrupted-20260412`)
  - us-snap-eligibility-refresh6-ready-20260412 (`../artifacts/eval-suites/us-snap-eligibility-refresh6-ready-20260412`)

### 2026-04-12: Checked-in SNAP deduction benchmarks stopped depending on duplicate-context imports

- Primary commits:
  - `12877e5` `Clarify SNAP pre-shelter source slice`
  - `c7891e5` `Stabilize checked-in SNAP deduction benchmarks`
- Hypothesis:
  - The remaining failures on the checked-in `snap_earned_income_deduction` and `snap_net_income_pre_shelter` refresh manifests were not substantive disagreements on the target slices. They came from benchmark setup defects: copied context files that redefined the same variables as the generated slice, an under-specified pre-shelter source slice that invited the wrong target, and a PE replay bridge that treated harmless naming drift in the generated intermediate income input as an oracle mismatch.
- Effect:
  - The checked-in earned-income and pre-shelter manifests no longer allow the broader `2014(e)` context file, so the generated atomic slices compile in isolation instead of colliding with duplicated canonical declarations.
  - The pre-shelter source slice now names the target variable directly and fixes the period framing to a monthly current-effective benchmark prompt.
  - PE replay now accepts the observed family of generated synonyms for “monthly household income after all other applicable deductions,” so the oracle validates the intended intermediate quantity rather than defaulting it to zero.
  - Both checked-in deduction benchmarks now reach a clean ready state with success, compile, CI, zero-ungrounded numerics, generalist review, and PolicyEngine all passing.
- Primary evidence paths:
  - [us_snap_earned_income_deduction_refresh.yaml](../benchmarks/us_snap_earned_income_deduction_refresh.yaml)
  - [us_snap_net_income_pre_shelter_refresh.yaml](../benchmarks/us_snap_net_income_pre_shelter_refresh.yaml)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - snap_net_income_pre_shelter.txt (`../../rulespec-us/sources/slices/7-USC/snap/current-effective/snap_net_income_pre_shelter.txt`)
  - us-snap-earned-income-deduction-refresh5-failed-20260412 (`../artifacts/eval-suites/us-snap-earned-income-deduction-refresh5-failed-20260412`)
  - us-snap-earned-income-deduction-refresh6-ready-20260412 (`../artifacts/eval-suites/us-snap-earned-income-deduction-refresh6-ready-20260412`)
  - us-snap-net-income-pre-shelter-refresh8-failed-20260412 (`../artifacts/eval-suites/us-snap-net-income-pre-shelter-refresh8-failed-20260412`)
  - us-snap-net-income-pre-shelter-refresh10-failed-20260412 (`../artifacts/eval-suites/us-snap-net-income-pre-shelter-refresh10-failed-20260412`)
  - us-snap-net-income-pre-shelter-refresh11-ready-20260412 (`../artifacts/eval-suites/us-snap-net-income-pre-shelter-refresh11-ready-20260412`)

### 2026-04-12: First SNAP state-overlay utility benchmarks split into one clean closeout and one external-oracle gap

- Primary commits:
  - `f8e93d2` `Add North Carolina SNAP SUA source slice`
  - `00254f2` `Add NC SNAP SUA benchmark and prompt guidance`
  - `c59094b` `Stabilize NC SNAP utility allowance validation`
- Hypothesis:
  - The state-overlay pattern for SNAP utility allowances should generalize cleanly once the benchmark keeps the cited jurisdiction fixed in negative tests, the PE replay bridge understands state utility-allowance variables and aliases, and reviewer supervision stops timing out on non-streaming Claude responses. Any remaining miss after that is more likely to be an external-oracle data gap than a harness failure.
- Effect:
  - North Carolina standard and limited utility allowance slices now have checked-in source excerpts and repo-augmented Axiom Encode benchmarks grounded in the official NC FNS 360 manual, and the validator can replay nationwide SNAP utility-allowance variables with NC-specific inputs and alias normalization.
  - The first NC limited utility allowance benchmark reached a clean ready state with success, compile, CI, zero-ungrounded numerics, generalist review, and PolicyEngine all passing.
  - The first NC telephone utility allowance benchmark did not close. Axiom Encode generated a plausible artifact, but PolicyEngine returned `$42.15` while the NC manual slice states `$41`, and direct inspection of the installed PolicyEngine parameter tree showed that PE currently uprates NC phone allowance data from the older `2023-10-01` entry instead of carrying the official `2024-10-01` NC manual value. This is recorded as an external oracle gap, not as a kept harness regression.
- Primary evidence paths:
  - us_snap_nc_standard_utility_allowance_refresh.yaml (`../benchmarks/us_snap_nc_standard_utility_allowance_refresh.yaml`)
  - us_snap_nc_limited_utility_allowance_refresh.yaml (`../benchmarks/us_snap_nc_limited_utility_allowance_refresh.yaml`)
  - us_snap_nc_individual_utility_allowance_refresh.yaml (`../benchmarks/us_snap_nc_individual_utility_allowance_refresh.yaml`)
  - [evals.py](../src/axiom_encode/harness/evals.py)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - snap_standard_utility_allowance_nc.txt (`../../rulespec-us/sources/slices/ncdhhs/fns/360/current-effective/snap_standard_utility_allowance_nc.txt`)
  - snap_limited_utility_allowance_nc.txt (`../../rulespec-us/sources/slices/ncdhhs/fns/360/current-effective/snap_limited_utility_allowance_nc.txt`)
  - snap_individual_utility_allowance_nc.txt (`../../rulespec-us/sources/slices/ncdhhs/fns/360/current-effective/snap_individual_utility_allowance_nc.txt`)
  - us-snap-nc-standard-utility-allowance-refresh8-reviewer-timeout-20260412 (`../artifacts/eval-suites/us-snap-nc-standard-utility-allowance-refresh8-reviewer-timeout-20260412`)
  - us-snap-nc-standard-utility-allowance-refresh9-ready-20260412 (`../artifacts/eval-suites/us-snap-nc-standard-utility-allowance-refresh9-ready-20260412`)
  - us-snap-nc-limited-utility-allowance-refresh1-ready-20260412 (`../artifacts/eval-suites/us-snap-nc-limited-utility-allowance-refresh1-ready-20260412`)
  - us-snap-nc-individual-utility-allowance-refresh1-policyengine-gap-20260412 (`../artifacts/eval-suites/us-snap-nc-individual-utility-allowance-refresh1-policyengine-gap-20260412`)

### 2026-04-12: North Carolina SNAP telephone utility allowance closed after a PolicyEngine data correction

- Primary commits:
  - `d60e19d` `Add NC FY2025 SNAP phone allowance` (local `policyengine-us` worktree)
  - `1ccf3f9` `Ignore row-labeled SNAP schedule helpers`
- Hypothesis:
  - The remaining NC telephone utility allowance failure was not a substantive disagreement about the cited current-effective state rule. It was a combination of (1) stale PolicyEngine NC phone data that skipped the official `2024-10-01` `$41` value and uprated from the older `2023-10-01` entry instead, and (2) one residual Axiom Encode grounding false negative that treated helper names like `unit_size_row_4` as substantive ungrounded numerics.
- Effect:
  - The local `policyengine-us` parameter tree now carries the explicit NC `2024-10-01: 41` phone allowance entry, with a targeted `2025-01` baseline regression.
  - Direct PolicyEngine evaluation for `2025-01` now returns `41.0` for NC phone allowance, aligning with the official NC FNS 360 manual slice.
  - Axiom Encode grounding now ignores row-labeled schedule helper names the same way it already ignored worded helper labels, so a flat TUA table no longer fails the zero-ungrounded gate on helper labels alone.
  - The refreshed NC telephone utility allowance benchmark now reaches a clean ready state with success, compile, CI, zero-ungrounded numerics, generalist review, and PolicyEngine all passing.
- Primary evidence paths:
  - snap_individual_utility_allowance_nc.txt (`../../rulespec-us/sources/slices/ncdhhs/fns/360/current-effective/snap_individual_utility_allowance_nc.txt`)
  - phone.yaml (`../../../worktrees/policyengine-us-main-view/policyengine_us/parameters/gov/usda/snap/income/deductions/utility/single/phone.yaml`)
  - snap_individual_utility_allowance.yaml (`../../../worktrees/policyengine-us-main-view/policyengine_us/tests/policy/baseline/gov/usda/snap/income/deductions/snap_individual_utility_allowance.yaml`)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - us-snap-nc-individual-utility-allowance-refresh1-policyengine-gap-20260412 (`../artifacts/eval-suites/us-snap-nc-individual-utility-allowance-refresh1-policyengine-gap-20260412`)
  - us-snap-nc-individual-utility-allowance-refresh3-ready-20260412 (`../artifacts/eval-suites/us-snap-nc-individual-utility-allowance-refresh3-ready-20260412`)

### 2026-04-12: Tennessee SNAP standard and limited utility allowance overlays closed on the repaired harness

- Primary commits:
  - `1c6af99` `Add Tennessee SNAP utility allowance source slices`
  - `2f68c83` `Close Tennessee SNAP utility allowance harness gaps`
- Hypothesis:
  - The Tennessee state-overlay misses were not new semantic failures in the utility-allowance pattern. They came from a cluster of harness issues: monthly test normalization rewrote valid pre-effective month cases into the first effective month, PolicyEngine interpreter discovery could miss the active worktree venv, structural manual references like `TennCare ABD Manual 125.020` still leaked into source numeric grounding, and generated current-effective update slices kept inventing speculative `pre_effective_* = 0` tests even when prior values were outside the cited slice.
- Effect:
  - Axiom Encode now preserves explicit earlier `YYYY-MM` monthly test periods instead of clamping them forward to the effective month.
  - PolicyEngine interpreter discovery now supports an explicit env override and known `~/worktrees/policyengine-us-main-view/.venv`-style checkout locations before falling back to ambient installs, reducing stale-oracle risk.
  - Structural manual references in both heading form (`... Manual 125.020`) and `Policy Manual Number 125.020` form are now excluded from source numeric grounding.
  - For monthly update slices, Axiom Encode now drops speculative `pre_effective_*` zero-output tests when the same output already has a positive in-scope case, which prevents false oracle failures on prior-period values that are outside the cited source slice.
  - Tennessee limited utility allowance now reaches a clean ready state against compile, CI, generalist review, and PolicyEngine.
  - Tennessee standard utility allowance now also reaches a clean ready state against compile, CI, generalist review, and PolicyEngine after removing the speculative pre-effective zero test and replaying against the pinned PolicyEngine worktree interpreter.
- Primary evidence paths:
  - us_snap_tn_limited_utility_allowance_refresh.yaml (`../benchmarks/us_snap_tn_limited_utility_allowance_refresh.yaml`)
  - us_snap_tn_standard_utility_allowance_refresh.yaml (`../benchmarks/us_snap_tn_standard_utility_allowance_refresh.yaml`)
  - [evals.py](../src/axiom_encode/harness/evals.py)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - snap_limited_utility_allowance_tn.txt (`../../rulespec-us-tn/sources/slices/tenncare/post-eligibility/current-effective/snap_limited_utility_allowance_tn.txt`)
  - snap_standard_utility_allowance_tn.txt (`../../rulespec-us-tn/sources/slices/tenncare/post-eligibility/current-effective/snap_standard_utility_allowance_tn.txt`)
  - us-snap-tn-limited-utility-allowance-refresh4-ready-20260412 (`../artifacts/eval-suites/us-snap-tn-limited-utility-allowance-refresh4-ready-20260412`)
  - us-snap-tn-standard-utility-allowance-refresh5-failed-20260412 (`../artifacts/eval-suites/us-snap-tn-standard-utility-allowance-refresh5-failed-20260412`)
  - us-snap-tn-standard-utility-allowance-refresh7-ready-20260412 (`../artifacts/eval-suites/us-snap-tn-standard-utility-allowance-refresh7-ready-20260412`)

### 2026-04-12: Tennessee SNAP telephone utility allowance overlay closed on the same state-overlay pattern

- Primary commits:
  - `1c6af99` `Add Tennessee SNAP utility allowance source slices`
  - `261b92d` `Add Tennessee SNAP telephone allowance benchmark`
- Hypothesis:
  - With the repaired monthly-update harness already closing Tennessee standard and limited utility allowances, the telephone utility allowance should close cleanly as the next Tennessee state overlay if the source slice is explicit about the SNAP category gate and the PolicyEngine Tennessee utility table already carries the October 1, 2025 amount.
- Effect:
  - Added a Tennessee telephone utility allowance source slice anchored to TennCare ABD Manual 125.020 section 3.d.ii.1.c.iii.
  - Added a checked-in Axiom Encode benchmark for `snap_individual_utility_allowance` with a PolicyEngine oracle hint, so the Tennessee telephone lane is now benchmarked the same way as North Carolina telephone and Tennessee standard/limited allowances.
  - The fresh Tennessee telephone replay reached a clean ready state against compile, CI, generalist review, and PolicyEngine without additional harness tuning, which is good evidence that the repaired state-overlay utility-allowance pattern now generalizes across all three Tennessee allowance categories.
- Primary evidence paths:
  - us_snap_tn_individual_utility_allowance_refresh.yaml (`../benchmarks/us_snap_tn_individual_utility_allowance_refresh.yaml`)
  - snap_individual_utility_allowance_tn.txt (`../../rulespec-us-tn/sources/slices/tenncare/post-eligibility/current-effective/snap_individual_utility_allowance_tn.txt`)
  - us-snap-tn-individual-utility-allowance-refresh1-ready-20260412 (`../artifacts/eval-suites/us-snap-tn-individual-utility-allowance-refresh1-ready-20260412`)

### 2026-04-12: Tennessee SNAP source corpus moved out of rulespec-us into rulespec-us-tn

- Primary commits:
  - `6302f5a` `Initialize Tennessee jurisdiction source repo`
  - `08360c4` `Remove Tennessee overlays from federal rulespec-us`
- Hypothesis:
  - The Tennessee SNAP utility slices were benchmark-clean, but they still lived in `rulespec-us`, which violates the intended boundary that repos correspond to jurisdictions rather than program labels. Moving those sources into a dedicated `rulespec-us-tn` repo should preserve benchmark behavior while making the repository structure align with the legal/policy authority.
- Effect:
  - Created a new Tennessee jurisdiction repo, `rulespec-us-tn`, with Tennessee source slices and minimal validation/docs scaffolding.
  - Removed the Tennessee utility source slices from `rulespec-us` so the federal repo now stays federal.
  - Retargeted the Tennessee Axiom Encode manifests and manifest-load tests to source from `rulespec-us-tn`.
  - Re-ran all three Tennessee utility overlays against the new repo path and archived fresh ready artifacts, confirming that the repo-boundary correction did not break the benchmarked Tennessee lanes.
- Primary evidence paths:
  - us_snap_tn_standard_utility_allowance_refresh.yaml (`../benchmarks/us_snap_tn_standard_utility_allowance_refresh.yaml`)
  - us_snap_tn_limited_utility_allowance_refresh.yaml (`../benchmarks/us_snap_tn_limited_utility_allowance_refresh.yaml`)
  - us_snap_tn_individual_utility_allowance_refresh.yaml (`../benchmarks/us_snap_tn_individual_utility_allowance_refresh.yaml`)
  - [test_evals.py](../tests/test_evals.py)
  - rulespec-us-tn README (`../../rulespec-us-tn/README.md`)
  - snap_standard_utility_allowance_tn.txt (`../../rulespec-us-tn/sources/slices/tenncare/post-eligibility/current-effective/snap_standard_utility_allowance_tn.txt`)
  - snap_limited_utility_allowance_tn.txt (`../../rulespec-us-tn/sources/slices/tenncare/post-eligibility/current-effective/snap_limited_utility_allowance_tn.txt`)
  - snap_individual_utility_allowance_tn.txt (`../../rulespec-us-tn/sources/slices/tenncare/post-eligibility/current-effective/snap_individual_utility_allowance_tn.txt`)
  - us-snap-tn-standard-utility-allowance-reorg1-ready-20260412 (`../artifacts/eval-suites/us-snap-tn-standard-utility-allowance-reorg1-ready-20260412`)
  - us-snap-tn-limited-utility-allowance-reorg1-ready-20260412 (`../artifacts/eval-suites/us-snap-tn-limited-utility-allowance-reorg1-ready-20260412`)
  - us-snap-tn-individual-utility-allowance-reorg1-ready-20260412 (`../artifacts/eval-suites/us-snap-tn-individual-utility-allowance-reorg1-ready-20260412`)

### 2026-04-13: North Carolina SNAP utility source corpus moved into rulespec-us-nc and revalidated

- Primary commits:
  - `195924d` `Initialize North Carolina SNAP utility source repo`
  - `c5bc3d4` `Move North Carolina SNAP utility slices to rulespec-us-nc`
  - `4062a97` `Route NC SNAP utility benchmarks to rulespec-us-nc`
- Hypothesis:
  - North Carolina SNAP utility overlays were already benchmark-clean locally, but the source slices still lived in `rulespec-us`, which violated the intended `repo == jurisdiction` boundary. Moving those sources into `rulespec-us-nc` should preserve benchmark behavior if Axiom Encode routing and source-manifest handling are actually repo-aware rather than hard-coded to `rulespec-us`.
- Effect:
  - Created a new North Carolina jurisdiction repo, `rulespec-us-nc`, with minimal validation/docs scaffolding and `relation: sets` sidecars that anchor the state-set SNAP utility parameters to the delegated federal CFR slots.
  - Removed the North Carolina utility source slices from `rulespec-us`, leaving the federal repo federal.
  - Retargeted the checked-in NC Axiom Encode manifests and manifest-load tests to source from `rulespec-us-nc`.
  - The first NC standard rerun on the new repo path exposed one remaining harness issue: terminal table buckets like `5 or more` needed explicit threshold guidance, and delegated `sets` metadata still allowed the model to guess broken canonical imports when the target file was not copied into the workspace.
  - After tightening those prompt rules and widening the schedule-index grounding filter for labels like `four_person_unit_size`, North Carolina standard, limited, and individual utility allowance reruns all reached clean ready states on the new jurisdiction-repo path.
- Primary evidence paths:
  - us_snap_nc_standard_utility_allowance_refresh.yaml (`../benchmarks/us_snap_nc_standard_utility_allowance_refresh.yaml`)
  - us_snap_nc_limited_utility_allowance_refresh.yaml (`../benchmarks/us_snap_nc_limited_utility_allowance_refresh.yaml`)
  - us_snap_nc_individual_utility_allowance_refresh.yaml (`../benchmarks/us_snap_nc_individual_utility_allowance_refresh.yaml`)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - rulespec-us-nc README (`../../rulespec-us-nc/README.md`)
  - snap_standard_utility_allowance_nc.txt (`../../rulespec-us-nc/sources/slices/ncdhhs/fns/360/current-effective/snap_standard_utility_allowance_nc.txt`)
  - snap_limited_utility_allowance_nc.txt (`../../rulespec-us-nc/sources/slices/ncdhhs/fns/360/current-effective/snap_limited_utility_allowance_nc.txt`)
  - snap_individual_utility_allowance_nc.txt (`../../rulespec-us-nc/sources/slices/ncdhhs/fns/360/current-effective/snap_individual_utility_allowance_nc.txt`)
  - us-snap-nc-standard-utility-allowance-reorg3-ready-20260413 (`../artifacts/eval-suites/us-snap-nc-standard-utility-allowance-reorg3-ready-20260413`)
  - us-snap-nc-limited-utility-allowance-reorg1-ready-20260413 (`../artifacts/eval-suites/us-snap-nc-limited-utility-allowance-reorg1-ready-20260413`)
  - us-snap-nc-individual-utility-allowance-reorg1-failed-20260413 (`../artifacts/eval-suites/us-snap-nc-individual-utility-allowance-reorg1-failed-20260413`)
  - us-snap-nc-individual-utility-allowance-reorg2-ready-20260413 (`../artifacts/eval-suites/us-snap-nc-individual-utility-allowance-reorg2-ready-20260413`)

### 2026-04-13: New York SNAP telephone utility allowance established as the next jurisdictional overlay

- Hypothesis:
  - After Tennessee and North Carolina closed cleanly, the next useful queue item should be a low-complexity third state overlay on an existing jurisdiction repo, not another federal slice. New York phone utility allowance is the narrowest clean candidate because the OTDA October 1, 2025 SUA notice sets one statewide phone amount and PolicyEngine already exposes `snap_individual_utility_allowance` for New York.
- Effect:
  - Added a New York source slice and `relation: sets` sidecar under `rulespec-us-ny`, anchored to the delegated federal SNAP utility-allowance slot.
  - Added a checked-in Axiom Encode benchmark manifest and manifest-load test for the New York telephone lane.
  - The first two smoke runs were not kept as harness regressions: they exposed that the drafted source slice repeated the same `$32` fact too many times for CI's numeric-occurrence rule. Narrowing the slice to the single substantive statewide fact closed the lane without new validator work.
  - The third smoke run reached a clean ready state against compile, CI, generalist review, and PolicyEngine, establishing New York telephone utility allowance as the next proven jurisdictional SNAP overlay.
- Primary evidence paths:
  - us_snap_ny_individual_utility_allowance_refresh.yaml (`../benchmarks/us_snap_ny_individual_utility_allowance_refresh.yaml`)
  - [test_evals.py](../tests/test_evals.py)
  - snap_individual_utility_allowance_ny.txt (`../../rulespec-us-ny/sources/slices/otda/snap/current-effective/snap_individual_utility_allowance_ny.txt`)
  - snap_individual_utility_allowance_ny.meta.yaml (`../../rulespec-us-ny/sources/slices/otda/snap/current-effective/snap_individual_utility_allowance_ny.meta.yaml`)
  - us-snap-ny-individual-utility-allowance-smoke1-failed-20260413 (`../artifacts/eval-suites/us-snap-ny-individual-utility-allowance-smoke1-failed-20260413`)
  - us-snap-ny-individual-utility-allowance-smoke2-failed-20260413 (`../artifacts/eval-suites/us-snap-ny-individual-utility-allowance-smoke2-failed-20260413`)
  - us-snap-ny-individual-utility-allowance-smoke3-ready-20260413 (`../artifacts/eval-suites/us-snap-ny-individual-utility-allowance-smoke3-ready-20260413`)

### 2026-04-13: New York SNAP standard and limited utility overlays close on the jurisdiction repo path

- Hypothesis:
  - After New York telephone utility allowance closed, the next pressure point was whether the same `rulespec-us-ny` source-slice pattern and PolicyEngine-backed replay would also hold for the multi-region standard and limited utility allowances without adding New York-specific policy hacks.
- Effect:
  - Added New York standard and limited utility source slices plus `relation: sets` sidecars under `rulespec-us-ny`, both anchored to the delegated federal SNAP utility-allowance slots.
  - Added checked-in Axiom Encode manifests and manifest-load tests for the New York standard and limited lanes.
  - The first replay failures were not legal-model misses. They exposed two harness issues in the US PolicyEngine bridge: explicit `snap_utility_region` test inputs were not being passed through to the household-level `snap_utility_region_str` variable that PolicyEngine actually consumes, and the first attempt at that bridge seeded the region at a monthly period even though the PolicyEngine household variable is yearly.
  - After fixing the replay bridge to normalize `snap_utility_region` / `snap_utility_region_str` onto the yearly household variable and adding a prompt rail that `match` fallbacks must use `_ => ...` rather than `else: ...`, both New York standard and limited utility allowance reruns reached clean ready states on compile, CI, generalist review, and PolicyEngine.
- Primary evidence paths:
  - us_snap_ny_standard_utility_allowance_refresh.yaml (`../benchmarks/us_snap_ny_standard_utility_allowance_refresh.yaml`)
  - us_snap_ny_limited_utility_allowance_refresh.yaml (`../benchmarks/us_snap_ny_limited_utility_allowance_refresh.yaml`)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - snap_standard_utility_allowance_ny.txt (`../../rulespec-us-ny/sources/slices/otda/snap/current-effective/snap_standard_utility_allowance_ny.txt`)
  - snap_standard_utility_allowance_ny.meta.yaml (`../../rulespec-us-ny/sources/slices/otda/snap/current-effective/snap_standard_utility_allowance_ny.meta.yaml`)
  - snap_limited_utility_allowance_ny.txt (`../../rulespec-us-ny/sources/slices/otda/snap/current-effective/snap_limited_utility_allowance_ny.txt`)
  - snap_limited_utility_allowance_ny.meta.yaml (`../../rulespec-us-ny/sources/slices/otda/snap/current-effective/snap_limited_utility_allowance_ny.meta.yaml`)
  - us-snap-ny-standard-utility-allowance-smoke5-failed-20260413 (`../artifacts/eval-suites/us-snap-ny-standard-utility-allowance-smoke5-failed-20260413`)
  - us-snap-ny-standard-utility-allowance-refresh1-ready-20260413 (`../artifacts/eval-suites/us-snap-ny-standard-utility-allowance-refresh1-ready-20260413`)
  - us-snap-ny-limited-utility-allowance-smoke4-failed-20260413 (`../artifacts/eval-suites/us-snap-ny-limited-utility-allowance-smoke4-failed-20260413`)
  - us-snap-ny-limited-utility-allowance-refresh1-ready-20260413 (`../artifacts/eval-suites/us-snap-ny-limited-utility-allowance-refresh1-ready-20260413`)

### 2026-04-13: Tennessee SNAP child-support deduction election closes as a delegated state-option boolean

- Hypothesis:
  - The next real state-overlay test after utility tables was whether Axiom Encode could cleanly encode a delegated jurisdictional SNAP option as the option itself, rather than overfitting everything to numeric table parameters or restating a downstream federal formula.
- Effect:
  - Added a Tennessee SNAP Policy Manual source slice plus `relation: sets` sidecar in `rulespec-us-tn`, anchored to `us:statutes/7/2014/e/4#snap_state_uses_child_support_deduction`.
  - Added a checked-in Axiom Encode benchmark for the Tennessee child-support deduction option.
  - Extended the PolicyEngine US bridge so delegated state-option booleans can be compared directly through parameter access, rather than only via downstream benefit variables.
  - Tightened CI so constant booleans are not misclassified as placeholder facts when eval source metadata explicitly marks them as delegated `sets` targets.
  - The first run exposed exactly two non-semantic misses: CI treated the delegated boolean as a forbidden constant fact, and the Tennessee source slice still carried a non-operative page-number citation that grounding counted as a missing scalar. After fixing those two system/source issues, the rerun closed fully ready on compile, CI, generalist review, and PolicyEngine.
- Primary evidence paths:
  - us_snap_tn_child_support_deduction_option_refresh.yaml (`../benchmarks/us_snap_tn_child_support_deduction_option_refresh.yaml`)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - snap_state_uses_child_support_deduction_tn.txt (`../../rulespec-us-tn/sources/slices/tdhs/snap/current-effective/snap_state_uses_child_support_deduction_tn.txt`)
  - snap_state_uses_child_support_deduction_tn.meta.yaml (`../../rulespec-us-tn/sources/slices/tdhs/snap/current-effective/snap_state_uses_child_support_deduction_tn.meta.yaml`)
  - us-snap-tn-child-support-deduction-option-refresh1-failed-20260413 (`../artifacts/eval-suites/us-snap-tn-child-support-deduction-option-refresh1-failed-20260413`)
  - us-snap-tn-child-support-deduction-option-refresh2-ready-20260413 (`../artifacts/eval-suites/us-snap-tn-child-support-deduction-option-refresh2-ready-20260413`)

### 2026-04-13: New York SNAP child-support treatment closes as the delegated false-state counterpart

- Hypothesis:
  - After Tennessee closed as a positive delegated child-support deduction option, the next check was whether the same `sets` pathway also works for the opposite state election: a jurisdiction that does not use the deduction because it excludes the payments from gross income instead.
- Effect:
  - Added a New York OTDA SNAP Source Book source slice plus `relation: sets` sidecar in `rulespec-us-ny`, anchored to `us:statutes/7/2014/e/4#snap_state_uses_child_support_deduction`.
  - Added a checked-in Axiom Encode benchmark for the New York child-support deduction option.
  - Generalized the PolicyEngine replay path so delegated state-option booleans no longer default to Tennessee when no explicit state input is present; replay now derives the state from eval source metadata jurisdiction when available.
  - The first New York run already passed generation, generalist review, and PolicyEngine. Its only miss was CI counting the editorial header `Rev. 7/2025` as a substantive numeric. After teaching numeric extraction to ignore revision markers and revalidating the existing output, the lane closed fully ready.
- Primary evidence paths:
  - us_snap_ny_child_support_deduction_option_refresh.yaml (`../benchmarks/us_snap_ny_child_support_deduction_option_refresh.yaml`)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - snap_state_uses_child_support_deduction_ny.txt (`../../rulespec-us-ny/sources/slices/otda/snap/current-effective/snap_state_uses_child_support_deduction_ny.txt`)
  - snap_state_uses_child_support_deduction_ny.meta.yaml (`../../rulespec-us-ny/sources/slices/otda/snap/current-effective/snap_state_uses_child_support_deduction_ny.meta.yaml`)
  - axiom_encode-us-snap-ny-child-support-deduction-option-refresh1-20260413 (`../artifacts/eval-suites/axiom_encode-us-snap-ny-child-support-deduction-option-refresh1-20260413`)

### 2026-04-13: North Carolina SNAP child-support deduction election closes on the first replay

- Hypothesis:
  - With the delegated-boolean replay path repaired for jurisdiction inference, a second positive state-option lane should close without further harness work if the source slice is clean and current.
- Effect:
  - Added a North Carolina FNS 340 child-support deduction source slice plus `relation: sets` sidecar in `rulespec-us-nc`, anchored to `us:statutes/7/2014/e/4#snap_state_uses_child_support_deduction`.
  - Added a checked-in Axiom Encode benchmark for the North Carolina child-support deduction option.
  - The first run closed fully ready on compile, CI, generalist review, and PolicyEngine with no additional harness changes, confirming that the repaired delegated-boolean path generalizes to another jurisdiction that affirmatively elects the deduction.
- Primary evidence paths:
  - us_snap_nc_child_support_deduction_option_refresh.yaml (`../benchmarks/us_snap_nc_child_support_deduction_option_refresh.yaml`)
  - [test_evals.py](../tests/test_evals.py)
  - snap_state_uses_child_support_deduction_nc.txt (`../../rulespec-us-nc/sources/slices/ncdhhs/fns/340/current-effective/snap_state_uses_child_support_deduction_nc.txt`)
  - snap_state_uses_child_support_deduction_nc.meta.yaml (`../../rulespec-us-nc/sources/slices/ncdhhs/fns/340/current-effective/snap_state_uses_child_support_deduction_nc.meta.yaml`)
  - axiom_encode-us-snap-nc-child-support-deduction-option-refresh1-20260413 (`../artifacts/eval-suites/axiom_encode-us-snap-nc-child-support-deduction-option-refresh1-20260413`)

### 2026-04-13: California SNAP self-employment expense option closes as a parameter-backed delegated state option

- Hypothesis:
  - The delegated `sets` pathway should generalize beyond the child-support election if the PolicyEngine bridge can treat state-option booleans as parameter-backed slots rather than one-off hard-coded cases.
- Effect:
  - Added a California CalFresh source slice plus `relation: sets` sidecar in `rulespec-us-ca`, anchored to `us:regulation/7-cfr/273/11/b/3#snap_self_employment_expense_based_deduction_applies`.
  - Updated `rulespec-us-ca` repo-boundary docs so California-administered SNAP overlays are treated as first-class jurisdiction content, not as out-of-repo federal leftovers.
  - Generalized the PolicyEngine US adapter model with a reusable parameter-path field for delegated state-option booleans, then used it for both the existing child-support election and the new California self-employment expense option.
  - The first California run already passed generation, compile, generalist review, and PolicyEngine; its only miss was CI counting the structural section citation `63-503.413` as a missing scalar. After teaching source numeric extraction to ignore manual section references of that form, the rerun closed fully ready.
- Primary evidence paths:
  - us_snap_ca_self_employment_expense_option_refresh.yaml (`../benchmarks/us_snap_ca_self_employment_expense_option_refresh.yaml`)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - snap_self_employment_expense_based_deduction_applies_ca.txt (`../../rulespec-us-ca/sources/slices/cdss/calfresh/current-effective/snap_self_employment_expense_based_deduction_applies_ca.txt`)
  - snap_self_employment_expense_based_deduction_applies_ca.meta.yaml (`../../rulespec-us-ca/sources/slices/cdss/calfresh/current-effective/snap_self_employment_expense_based_deduction_applies_ca.meta.yaml`)
  - us-snap-ca-self-employment-expense-option-refresh1-ci-false-positive-20260413 (`../artifacts/eval-suites/us-snap-ca-self-employment-expense-option-refresh1-ci-false-positive-20260413`)
  - us-snap-ca-self-employment-expense-option-refresh2-ready-20260413 (`../artifacts/eval-suites/us-snap-ca-self-employment-expense-option-refresh2-ready-20260413`)

### 2026-04-13: Colorado SNAP self-employment expense option closes after generalized code-citation cleanup

- Hypothesis:
  - The parameter-backed delegated `sets` path for SNAP self-employment expense options should transfer from California to another jurisdiction without new oracle-adapter logic. If the first Colorado replay fails, the most likely remaining gap is structural numeric cleanup for state code citations rather than the legal model itself.
- Effect:
  - Added a Colorado SNAP regulation source slice plus `relation: sets` sidecar in `rulespec-us-co`, anchored to `us:regulation/7-cfr/273/11/b/3#snap_self_employment_expense_based_deduction_applies`.
  - Updated `rulespec-us-co` repo-boundary docs so Colorado-administered SNAP overlays are treated as first-class jurisdiction content alongside Colorado Works materials.
  - Added a checked-in Axiom Encode benchmark for the Colorado self-employment expense option.
  - The first Colorado run already passed generation, compile, generalist review, and PolicyEngine; its only miss was CI counting structural citation numerics from `10 CCR 2506-1` and `section 4.403.11(B)-(C)(3)` as missing scalars.
  - Generalized source numeric cleanup to ignore code citations and decimal section references of that form, then reran the same benchmark. The second run closed fully ready without changing the adapter path.
- Primary evidence paths:
  - [us_snap_co_self_employment_expense_option_refresh.yaml](../benchmarks/us_snap_co_self_employment_expense_option_refresh.yaml)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - snap_self_employment_expense_based_deduction_applies_co.txt (`../../rulespec-us-co/sources/slices/cdhs/snap/current-effective/snap_self_employment_expense_based_deduction_applies_co.txt`)
  - snap_self_employment_expense_based_deduction_applies_co.meta.yaml (`../../rulespec-us-co/sources/slices/cdhs/snap/current-effective/snap_self_employment_expense_based_deduction_applies_co.meta.yaml`)
  - us-snap-co-self-employment-expense-option-refresh1-ci-false-positive-20260413 (`../artifacts/eval-suites/us-snap-co-self-employment-expense-option-refresh1-ci-false-positive-20260413`)
  - us-snap-co-self-employment-expense-option-refresh2-ready-20260413 (`../artifacts/eval-suites/us-snap-co-self-employment-expense-option-refresh2-ready-20260413`)

### 2026-04-13: New York SNAP self-employment expense option closes on the event-driven Codex queue

- Hypothesis:
  - After California, Colorado, Tennessee, and North Carolina closed on the delegated parameter-backed `sets` path for SNAP self-employment expense treatment, the same lane should transfer to New York without new harness work. The remaining question was whether the event-driven Codex queue would close the slice cleanly rather than needing a one-off manual replay.
- Effect:
  - Added a New York OTDA SNAP Source Book source slice plus `relation: sets` sidecar in `rulespec-us-ny`, anchored to `us:regulation/7-cfr/273/11/b/3#snap_self_employment_expense_based_deduction_applies`.
  - Added a checked-in Axiom Encode benchmark for the New York self-employment expense option.
  - Seeded the local event-driven SNAP queue with the new New York benchmark and let the Codex-backed runner process it end to end.
  - The first queue run closed fully ready on success, compile, CI, generalist review, and PolicyEngine. The only reviewer notes were non-blocking suggestions about using a dated start boundary instead of the `0001-01-01` sentinel and adding a second temporal-stability test case.
- Primary evidence paths:
  - us_snap_ny_self_employment_expense_option_refresh.yaml (`../benchmarks/us_snap_ny_self_employment_expense_option_refresh.yaml`)
  - [test_evals.py](../tests/test_evals.py)
  - snap_self_employment_expense_based_deduction_applies_ny.txt (`../../rulespec-us-ny/sources/slices/otda/snap/current-effective/snap_self_employment_expense_based_deduction_applies_ny.txt`)
  - snap_self_employment_expense_based_deduction_applies_ny.meta.yaml (`../../rulespec-us-ny/sources/slices/otda/snap/current-effective/snap_self_employment_expense_based_deduction_applies_ny.meta.yaml`)
  - axiom_encode-snap-self-employment-expense-based-deduction-applies-ny-20260413t153209 (`../artifacts/eval-suites/axiom_encode-snap-self-employment-expense-based-deduction-applies-ny-20260413t153209`)

### 2026-04-14: Texas delegated SNAP state-option lanes close after generalized handbook numeric cleanup

- Hypothesis:
  - The next Texas SNAP overlays after utility allowances should be the already-proven delegated state-option classes: child-support deduction election and self-employment actual-expense treatment. If those first Texas runs fail, the likely remaining gap is source-numeric cleanup for handbook section labels, revision codes, and IRS form references rather than a new legal-model or oracle-adapter problem.
- Effect:
  - Added Texas Works Handbook source slices plus `relation: sets` sidecars in `rulespec-us-tx` for `snap_state_uses_child_support_deduction` and `snap_self_employment_expense_based_deduction_applies`.
  - Added checked-in Axiom Encode benchmarks for both Texas lanes.
  - Let the event-driven Codex queue process both lanes end to end.
  - The first Texas child-support and self-employment runs both generated legally plausible delegated booleans, passed compile, passed PolicyEngine, and passed review, but failed CI because the validator still counted handbook section identifiers like `A-1421.1`, revision markers like `Revision 24-2`, and `Form 1040` references as missing substantive numerics.
  - Generalized numeric cleanup to ignore those structural handbook/form references, locked that behavior with focused validator tests, requeued the blocked Texas items on the same event-driven runner, and both reruns closed fully ready.
- Primary evidence paths:
  - us_snap_tx_child_support_deduction_option_refresh.yaml (`../benchmarks/us_snap_tx_child_support_deduction_option_refresh.yaml`)
  - us_snap_tx_self_employment_expense_option_refresh.yaml (`../benchmarks/us_snap_tx_self_employment_expense_option_refresh.yaml`)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - snap_state_uses_child_support_deduction_tx.meta.yaml (`../../rulespec-us-tx/sources/targets/txhhs/twh/current-effective/snap_state_uses_child_support_deduction_tx.meta.yaml`)
  - snap_self_employment_expense_based_deduction_applies_tx.meta.yaml (`../../rulespec-us-tx/sources/targets/txhhs/twh/current-effective/snap_self_employment_expense_based_deduction_applies_tx.meta.yaml`)
  - axiom_encode-snap-state-uses-child-support-deduction-tx-20260413t205409 (`../artifacts/eval-suites/axiom_encode-snap-state-uses-child-support-deduction-tx-20260413t205409`)
  - axiom_encode-snap-self-employment-expense-based-deduction-applies-tx-20260413t205807 (`../artifacts/eval-suites/axiom_encode-snap-self-employment-expense-based-deduction-applies-tx-20260413t205807`)
  - axiom_encode-snap-state-uses-child-support-deduction-tx-20260413t210425 (`../artifacts/eval-suites/axiom_encode-snap-state-uses-child-support-deduction-tx-20260413t210425`)
  - axiom_encode-snap-self-employment-expense-based-deduction-applies-tx-20260413t210753 (`../artifacts/eval-suites/axiom_encode-snap-self-employment-expense-based-deduction-applies-tx-20260413t210753`)

### 2026-04-14: California delegated SNAP child-support option closes on the auto-synced Codex queue

- Hypothesis:
  - California should transfer on the already-proven delegated SNAP child-support state-option lane without any new harness work. The useful system check is whether the manifest-driven queue seeding path discovers and runs the new California benchmark automatically.
- Effect:
  - Added a California CalFresh current-effective source slice plus `relation: sets` sidecar for `snap_state_uses_child_support_deduction`, anchored to `us:statutes/7/2014/e/4#snap_state_uses_child_support_deduction`.
  - Added a checked-in Axiom Encode benchmark for the California child-support option and matching manifest-load coverage.
  - Let the local event-driven Codex queue auto-discover the new benchmark from the checked-in `axiom_encode/benchmarks` directory and process it end to end without a manual queue edit.
  - The first California run closed fully ready on success, compile, CI, zero ungrounded numerics, generalist review, and PolicyEngine, with no harness change required after the benchmark landed.
- Primary evidence paths:
  - us_snap_ca_child_support_deduction_option_refresh.yaml (`../benchmarks/us_snap_ca_child_support_deduction_option_refresh.yaml`)
  - [test_evals.py](../tests/test_evals.py)
  - snap_state_uses_child_support_deduction_ca.txt (`../../rulespec-us-ca/sources/slices/cdss/calfresh/current-effective/snap_state_uses_child_support_deduction_ca.txt`)
  - snap_state_uses_child_support_deduction_ca.meta.yaml (`../../rulespec-us-ca/sources/slices/cdss/calfresh/current-effective/snap_state_uses_child_support_deduction_ca.meta.yaml`)
  - axiom_encode-snap-state-uses-child-support-deduction-ca-20260413t212724 (`../artifacts/eval-suites/axiom_encode-snap-state-uses-child-support-deduction-ca-20260413t212724`)

### 2026-04-14: Colorado delegated SNAP child-support option closes after a real oracle fix plus slash-date cleanup

- Hypothesis:
  - Colorado should transfer on the same delegated SNAP child-support state-option lane as California and New York. If it fails, the most likely causes are either a real PolicyEngine state-option data bug or one more structural-source numeric false positive in the validator.
- Effect:
  - Added a Colorado SNAP current-effective source slice plus `relation: sets` sidecar for `snap_state_uses_child_support_deduction`, anchored to `us:statutes/7/2014/e/4#snap_state_uses_child_support_deduction`.
  - Added a checked-in Axiom Encode benchmark for the Colorado child-support option and matching manifest-load coverage.
  - The first event-driven Codex run generated the correct delegated boolean shape and passed compile, but failed review and PolicyEngine because local PolicyEngine incorrectly flipped Colorado to `true` in the SNAP child-support state-option parameter file despite the cited current Colorado regulation still using an exclusion.
  - Patched local PolicyEngine to keep Colorado `false` under the cited 2023 rule and added a direct Colorado SNAP gross-income-deduction regression.
  - Generalized Axiom Encode numeric cleanup to ignore slash-form effective dates like `10/01/2023`, which had still been leaking a structural `10` into CI.
  - Requeued the same Colorado manifest on the event-driven runner; the rerun closed fully ready on success, compile, CI, zero ungrounded numerics, generalist review, and PolicyEngine.
- Primary evidence paths:
  - [us_snap_co_child_support_deduction_option_refresh.yaml](../benchmarks/us_snap_co_child_support_deduction_option_refresh.yaml)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - snap_state_uses_child_support_deduction_co.txt (`../../rulespec-us-co/sources/slices/cdhs/snap/current-effective/snap_state_uses_child_support_deduction_co.txt`)
  - snap_state_uses_child_support_deduction_co.meta.yaml (`../../rulespec-us-co/sources/slices/cdhs/snap/current-effective/snap_state_uses_child_support_deduction_co.meta.yaml`)
  - axiom_encode-snap-state-uses-child-support-deduction-co-20260413t213415 (`../artifacts/eval-suites/axiom_encode-snap-state-uses-child-support-deduction-co-20260413t213415`)
  - axiom_encode-snap-state-uses-child-support-deduction-co-20260413t214130 (`../artifacts/eval-suites/axiom_encode-snap-state-uses-child-support-deduction-co-20260413t214130`)

### 2026-04-14: Florida delegated SNAP state-option lanes close cleanly on the first Codex queue pass

- Hypothesis:
  - Florida should transfer cleanly on the already-proven delegated SNAP boolean lanes for child-support deduction election and self-employment actual-expense treatment. The useful system check is whether a newly scaffolded jurisdiction repo plus checked-in benchmarks are enough for the event-driven Codex queue to discover and close both lanes without any new harness work.
- Effect:
  - Created `rulespec-us-fl` as a Florida jurisdiction repo for Florida-administered SNAP overlays, with exact Florida ESS manual source slices plus `relation: sets` sidecars for `snap_state_uses_child_support_deduction` and `snap_self_employment_expense_based_deduction_applies`.
  - Added checked-in Axiom Encode benchmarks for both Florida lanes and matching manifest-load coverage.
  - Let the local event-driven Codex queue auto-discover both new benchmarks from `axiom_encode/benchmarks` and process them end to end.
  - Both Florida runs closed fully ready on the first pass, with success, compile, CI, zero ungrounded numerics, generalist review, and PolicyEngine all at 1.0. No harness or oracle repair was needed after the benchmarks landed.
- Primary evidence paths:
  - us_snap_fl_child_support_deduction_option_refresh.yaml (`../benchmarks/us_snap_fl_child_support_deduction_option_refresh.yaml`)
  - us_snap_fl_self_employment_expense_option_refresh.yaml (`../benchmarks/us_snap_fl_self_employment_expense_option_refresh.yaml`)
  - [test_evals.py](../tests/test_evals.py)
  - snap_state_uses_child_support_deduction_fl.txt (`../../rulespec-us-fl/sources/slices/myflfamilies/ess/current-effective/snap_state_uses_child_support_deduction_fl.txt`)
  - snap_state_uses_child_support_deduction_fl.meta.yaml (`../../rulespec-us-fl/sources/slices/myflfamilies/ess/current-effective/snap_state_uses_child_support_deduction_fl.meta.yaml`)
  - snap_self_employment_expense_based_deduction_applies_fl.txt (`../../rulespec-us-fl/sources/slices/myflfamilies/ess/current-effective/snap_self_employment_expense_based_deduction_applies_fl.txt`)
  - snap_self_employment_expense_based_deduction_applies_fl.meta.yaml (`../../rulespec-us-fl/sources/slices/myflfamilies/ess/current-effective/snap_self_employment_expense_based_deduction_applies_fl.meta.yaml`)
  - axiom_encode-snap-state-uses-child-support-deduction-fl-20260413t220610 (`../artifacts/eval-suites/axiom_encode-snap-state-uses-child-support-deduction-fl-20260413t220610`)
  - axiom_encode-snap-self-employment-expense-based-deduction-applies-fl-20260413t220911 (`../artifacts/eval-suites/axiom_encode-snap-self-employment-expense-based-deduction-applies-fl-20260413t220911`)

### 2026-04-14: Maryland self-employment lane is corrected to the simplified deduction-rate slot and closes ready

- Hypothesis:
  - The Maryland DHS source slice does not support the previously benchmarked boolean `snap_self_employment_expense_based_deduction_applies`. It states a concrete 50 percent simplified deduction rate. The correct fix is to change the benchmark ontology, not to keep repairing the wrong boolean target.
- Effect:
  - Replaced the Maryland source slice and sidecar so the jurisdiction sets `us:regulation/7-cfr/273/11/b/3#snap_self_employment_simplified_deduction_rate` instead of the expense-based boolean slot.
  - Replaced the checked-in Maryland benchmark with a new Axiom Encode refresh manifest for `snap_self_employment_simplified_deduction_rate`.
  - Extended the PE-US adapter so parameter-backed SNAP state options can read numeric parameter values, not just booleans.
  - Re-ran the event-driven Codex queue. It retired the deleted Maryland expense-based manifest automatically, queued the new simplified-rate manifest, and the new run closed fully ready on success, compile, CI, zero ungrounded numerics, generalist review, and PolicyEngine.
- Primary evidence paths:
  - us_snap_md_self_employment_simplified_deduction_rate_refresh.yaml (`../benchmarks/us_snap_md_self_employment_simplified_deduction_rate_refresh.yaml`)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - snap_self_employment_simplified_deduction_rate_md.txt (`../../rulespec-us-md/sources/slices/dhs/snap/current-effective/snap_self_employment_simplified_deduction_rate_md.txt`)
  - snap_self_employment_simplified_deduction_rate_md.meta.yaml (`../../rulespec-us-md/sources/slices/dhs/snap/current-effective/snap_self_employment_simplified_deduction_rate_md.meta.yaml`)
  - axiom_encode-snap-self-employment-simplified-deduction-rate-md-20260414t064852 (`../artifacts/eval-suites/axiom_encode-snap-self-employment-simplified-deduction-rate-md-20260414t064852`)

### 2026-04-14: Georgia delegated SNAP option lanes close after integer manual-number cleanup

- Hypothesis:
  - Georgia should transfer on the already-proven delegated SNAP state-option lanes for child-support deduction election, self-employment actual-expense treatment, and the self-employment simplified deduction rate. The likely failure mode was structural grounding noise from the Georgia DFCS manual section numbers `3035` and `3425`, not a new semantic gap.
- Effect:
  - Added `rulespec-us-ga` as a Georgia jurisdiction repo with exact DFCS source slices plus `relation: sets` sidecars for `snap_state_uses_child_support_deduction`, `snap_self_employment_expense_based_deduction_applies`, and `snap_self_employment_simplified_deduction_rate`.
  - Added checked-in Axiom Encode benchmarks for all three Georgia lanes and matching manifest-load coverage.
  - The first child-support and expense runs generated promotion-ready RuleSpec and passed compile, review, and PolicyEngine, but failed CI because numeric grounding treated the DFCS manual section labels `3035` and `3425` as missing scalar values.
  - Generalized validator numeric cleanup to ignore integer SNAP manual section numbers, locked that with a focused regression, and requeued the Georgia blockers on the event-driven Codex runner.
  - The reruns closed all three Georgia lanes fully ready on success, compile, CI, zero ungrounded numerics, generalist review, and PolicyEngine.
- Primary evidence paths:
  - us_snap_ga_child_support_deduction_option_refresh.yaml (`../benchmarks/us_snap_ga_child_support_deduction_option_refresh.yaml`)
  - us_snap_ga_self_employment_expense_option_refresh.yaml (`../benchmarks/us_snap_ga_self_employment_expense_option_refresh.yaml`)
  - us_snap_ga_self_employment_simplified_deduction_rate_refresh.yaml (`../benchmarks/us_snap_ga_self_employment_simplified_deduction_rate_refresh.yaml`)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - [test_evals.py](../tests/test_evals.py)
  - snap_state_uses_child_support_deduction_ga.txt (`../../rulespec-us-ga/sources/slices/dfcs/snap/current-effective/snap_state_uses_child_support_deduction_ga.txt`)
  - snap_state_uses_child_support_deduction_ga.meta.yaml (`../../rulespec-us-ga/sources/slices/dfcs/snap/current-effective/snap_state_uses_child_support_deduction_ga.meta.yaml`)
  - snap_self_employment_expense_based_deduction_applies_ga.txt (`../../rulespec-us-ga/sources/slices/dfcs/snap/current-effective/snap_self_employment_expense_based_deduction_applies_ga.txt`)
  - snap_self_employment_expense_based_deduction_applies_ga.meta.yaml (`../../rulespec-us-ga/sources/slices/dfcs/snap/current-effective/snap_self_employment_expense_based_deduction_applies_ga.meta.yaml`)
  - snap_self_employment_simplified_deduction_rate_ga.txt (`../../rulespec-us-ga/sources/slices/dfcs/snap/current-effective/snap_self_employment_simplified_deduction_rate_ga.txt`)
  - snap_self_employment_simplified_deduction_rate_ga.meta.yaml (`../../rulespec-us-ga/sources/slices/dfcs/snap/current-effective/snap_self_employment_simplified_deduction_rate_ga.meta.yaml`)
  - axiom_encode-snap-state-uses-child-support-deduction-ga-20260414t072502 (`../artifacts/eval-suites/axiom_encode-snap-state-uses-child-support-deduction-ga-20260414t072502`)
  - axiom_encode-snap-self-employment-expense-based-deduction-applies-ga-20260414t072737 (`../artifacts/eval-suites/axiom_encode-snap-self-employment-expense-based-deduction-applies-ga-20260414t072737`)
  - axiom_encode-snap-self-employment-simplified-deduction-rate-ga-20260414t073125 (`../artifacts/eval-suites/axiom_encode-snap-self-employment-simplified-deduction-rate-ga-20260414t073125`)

### 2026-04-14: South Carolina delegated SNAP option lanes close after manual-volume cleanup

- Hypothesis:
  - South Carolina should transfer on the same three delegated SNAP lane shapes as Georgia, using the August 2025 DSS SNAP Manual Vol 65: child-support deduction election, actual self-employment expenses disallowed, and a 40 percent simplified self-employment deduction rate. The likely new failure mode was structural grounding noise from `Vol 65`, not a new ontology gap.
- Effect:
  - Created `rulespec-us-sc` as a South Carolina jurisdiction repo with exact DSS source slices plus `relation: sets` sidecars for `snap_state_uses_child_support_deduction`, `snap_self_employment_expense_based_deduction_applies`, and `snap_self_employment_simplified_deduction_rate`.
  - Added checked-in Axiom Encode benchmarks for all three South Carolina lanes and matching manifest-load coverage.
  - The first South Carolina child-support and expense runs generated correct policy shapes and passed compile, review, and PolicyEngine, but failed CI because numeric grounding treated `Vol 65` from the manual title as a missing scalar; the first expense run also encoded the related `40%` as a non-rate helper and therefore missed grounding on `0.4`.
  - Generalized validator numeric cleanup to ignore manual volume numbers like `Vol 65`, locked that with a focused regression, and requeued the blocked South Carolina lanes on the event-driven Codex runner.
  - On the patched runner, the simplified-rate lane closed ready immediately, the child-support rerun closed ready, and the expense rerun closed ready without any further harness or prompt changes, confirming that the remaining expense miss was generation variance once the volume-number false positive was removed.
- Primary evidence paths:
  - us_snap_sc_child_support_deduction_option_refresh.yaml (`../benchmarks/us_snap_sc_child_support_deduction_option_refresh.yaml`)
  - us_snap_sc_self_employment_expense_option_refresh.yaml (`../benchmarks/us_snap_sc_self_employment_expense_option_refresh.yaml`)
  - us_snap_sc_self_employment_simplified_deduction_rate_refresh.yaml (`../benchmarks/us_snap_sc_self_employment_simplified_deduction_rate_refresh.yaml`)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - [test_evals.py](../tests/test_evals.py)
  - snap_state_uses_child_support_deduction_sc.txt (`../../rulespec-us-sc/sources/slices/scdss/snap/current-effective/snap_state_uses_child_support_deduction_sc.txt`)
  - snap_state_uses_child_support_deduction_sc.meta.yaml (`../../rulespec-us-sc/sources/slices/scdss/snap/current-effective/snap_state_uses_child_support_deduction_sc.meta.yaml`)
  - snap_self_employment_expense_based_deduction_applies_sc.txt (`../../rulespec-us-sc/sources/slices/scdss/snap/current-effective/snap_self_employment_expense_based_deduction_applies_sc.txt`)
  - snap_self_employment_expense_based_deduction_applies_sc.meta.yaml (`../../rulespec-us-sc/sources/slices/scdss/snap/current-effective/snap_self_employment_expense_based_deduction_applies_sc.meta.yaml`)
  - snap_self_employment_simplified_deduction_rate_sc.txt (`../../rulespec-us-sc/sources/slices/scdss/snap/current-effective/snap_self_employment_simplified_deduction_rate_sc.txt`)
  - snap_self_employment_simplified_deduction_rate_sc.meta.yaml (`../../rulespec-us-sc/sources/slices/scdss/snap/current-effective/snap_self_employment_simplified_deduction_rate_sc.meta.yaml`)
  - axiom_encode-snap-state-uses-child-support-deduction-sc-20260414t075329 (`../artifacts/eval-suites/axiom_encode-snap-state-uses-child-support-deduction-sc-20260414t075329`)
  - axiom_encode-snap-self-employment-expense-based-deduction-applies-sc-20260414t075715 (`../artifacts/eval-suites/axiom_encode-snap-self-employment-expense-based-deduction-applies-sc-20260414t075715`)
  - axiom_encode-snap-self-employment-simplified-deduction-rate-sc-20260414t074915 (`../artifacts/eval-suites/axiom_encode-snap-self-employment-simplified-deduction-rate-sc-20260414t074915`)

### 2026-04-14: Alabama delegated SNAP option lanes close after source-slice narrowing and placeholder-period normalization

- Hypothesis:
  - Alabama should transfer on the same delegated SNAP lane shapes as Georgia and South Carolina: child-support deduction election, self-employment actual-expense treatment, and a 40 percent simplified self-employment deduction rate. The likely risk was not ontology mismatch, but source/eval hygiene because Alabama's expense-option source proof relies on the universal standard-deduction rule rather than an explicit "no actual expenses" sentence.
- Effect:
  - Created `rulespec-us-al` as an Alabama jurisdiction repo with exact DHR source slices plus `relation: sets` sidecars for `snap_state_uses_child_support_deduction`, `snap_self_employment_expense_based_deduction_applies`, and `snap_self_employment_simplified_deduction_rate`.
  - Added checked-in Axiom Encode benchmarks for all three Alabama lanes and matching manifest-load coverage.
  - The first Alabama child-support and simplified-rate runs closed ready immediately.
  - The first Alabama expense-option run generated the right boolean policy shape and passed compile, review, and PolicyEngine, but failed CI because the source slice bundled the supporting `40%` rate that belonged to the separate rate lane; narrowing the source slice to the operative "standard will be used for all" sentence fixed that benchmark-target mismatch.
  - The next two Alabama expense-option reruns exposed a reusable harness problem: monthly state-option `.test.yaml` files were allowed to keep placeholder periods like `0001-01` and `0001-01-01`, which broke parameter-backed PolicyEngine replay even when the encoding itself was correct.
  - Normalized placeholder monthly and placeholder day periods in Axiom Encode test materialization, locked both repairs with focused eval tests, and requeued the single blocked Alabama expense lane.
  - The final Alabama expense replay closed fully ready on success, compile, CI, zero ungrounded numerics, generalist review, and PolicyEngine.
- Primary evidence paths:
  - us_snap_al_child_support_deduction_option_refresh.yaml (`../benchmarks/us_snap_al_child_support_deduction_option_refresh.yaml`)
  - us_snap_al_self_employment_expense_option_refresh.yaml (`../benchmarks/us_snap_al_self_employment_expense_option_refresh.yaml`)
  - us_snap_al_self_employment_simplified_deduction_rate_refresh.yaml (`../benchmarks/us_snap_al_self_employment_simplified_deduction_rate_refresh.yaml`)
  - [evals.py](../src/axiom_encode/harness/evals.py)
  - [test_evals.py](../tests/test_evals.py)
  - snap_state_uses_child_support_deduction_al.txt (`../../rulespec-us-al/sources/slices/aldhr/poe/current-effective/snap_state_uses_child_support_deduction_al.txt`)
  - snap_state_uses_child_support_deduction_al.meta.yaml (`../../rulespec-us-al/sources/slices/aldhr/poe/current-effective/snap_state_uses_child_support_deduction_al.meta.yaml`)
  - snap_self_employment_expense_based_deduction_applies_al.txt (`../../rulespec-us-al/sources/slices/aldhr/poe/current-effective/snap_self_employment_expense_based_deduction_applies_al.txt`)
  - snap_self_employment_expense_based_deduction_applies_al.meta.yaml (`../../rulespec-us-al/sources/slices/aldhr/poe/current-effective/snap_self_employment_expense_based_deduction_applies_al.meta.yaml`)
  - snap_self_employment_simplified_deduction_rate_al.txt (`../../rulespec-us-al/sources/slices/aldhr/poe/current-effective/snap_self_employment_simplified_deduction_rate_al.txt`)
  - snap_self_employment_simplified_deduction_rate_al.meta.yaml (`../../rulespec-us-al/sources/slices/aldhr/poe/current-effective/snap_self_employment_simplified_deduction_rate_al.meta.yaml`)
  - axiom_encode-snap-state-uses-child-support-deduction-al-20260414t081202 (`../artifacts/eval-suites/axiom_encode-snap-state-uses-child-support-deduction-al-20260414t081202`)
  - axiom_encode-snap-self-employment-expense-based-deduction-applies-al-20260414t084945 (`../artifacts/eval-suites/axiom_encode-snap-self-employment-expense-based-deduction-applies-al-20260414t084945`)
  - axiom_encode-snap-self-employment-simplified-deduction-rate-al-20260414t081937 (`../artifacts/eval-suites/axiom_encode-snap-self-employment-simplified-deduction-rate-al-20260414t081937`)

### 2026-04-14: Arkansas delegated SNAP option lanes close on the first pass

- Hypothesis:
  - Arkansas should transfer on the already-proven delegated SNAP state-option shapes for child-support deduction election and actual self-employment expense treatment. The official DHS SNAP Certification Manual states both rules directly, and because neither slice carries schedule numerics or multi-branch tables, the expected outcome was a first-pass closeout without new harness work.
- Effect:
  - Created `rulespec-us-ar` as an Arkansas jurisdiction repo with exact DHS source slices plus `relation: sets` sidecars for `snap_state_uses_child_support_deduction` and `snap_self_employment_expense_based_deduction_applies`.
  - Added checked-in Axiom Encode benchmarks for both Arkansas lanes and matching manifest-load coverage.
  - Let the local event-driven Codex queue auto-discover both new benchmarks from `axiom_encode/benchmarks` and process them end to end.
  - Both Arkansas runs closed fully ready on the first pass, with success, compile, CI, zero ungrounded numerics, generalist review, and PolicyEngine all at 1.0. No harness or oracle repair was needed after the benchmarks landed.
- Primary evidence paths:
  - us_snap_ar_child_support_deduction_option_refresh.yaml (`../benchmarks/us_snap_ar_child_support_deduction_option_refresh.yaml`)
  - us_snap_ar_self_employment_expense_option_refresh.yaml (`../benchmarks/us_snap_ar_self_employment_expense_option_refresh.yaml`)
  - [test_evals.py](../tests/test_evals.py)
  - snap_state_uses_child_support_deduction_ar.txt (`../../rulespec-us-ar/sources/slices/ardhs/snap/current-effective/snap_state_uses_child_support_deduction_ar.txt`)
  - snap_state_uses_child_support_deduction_ar.meta.yaml (`../../rulespec-us-ar/sources/slices/ardhs/snap/current-effective/snap_state_uses_child_support_deduction_ar.meta.yaml`)
  - snap_self_employment_expense_based_deduction_applies_ar.txt (`../../rulespec-us-ar/sources/slices/ardhs/snap/current-effective/snap_self_employment_expense_based_deduction_applies_ar.txt`)
  - snap_self_employment_expense_based_deduction_applies_ar.meta.yaml (`../../rulespec-us-ar/sources/slices/ardhs/snap/current-effective/snap_self_employment_expense_based_deduction_applies_ar.meta.yaml`)
  - axiom_encode-snap-state-uses-child-support-deduction-ar-20260414t091035 (`../artifacts/eval-suites/axiom_encode-snap-state-uses-child-support-deduction-ar-20260414t091035`)
  - axiom_encode-snap-self-employment-expense-based-deduction-applies-ar-20260414t091446 (`../artifacts/eval-suites/axiom_encode-snap-self-employment-expense-based-deduction-applies-ar-20260414t091446`)

### 2026-04-14: Texas parity push extends beyond utilities and option booleans

- Hypothesis:
  - Texas was the strongest candidate for a first state-level SNAP parity push because it already had green utility allowances plus the child-support deduction election and self-employment actual-expense option. The next likely PolicyEngine-comparable Texas gaps were the standard medical deduction amount and the homeless shelter deduction availability boolean published in the HHSC MEPD/TW bulletin.
- Effect:
  - Added exact `rulespec-us-tx` source slices and delegated `sets` sidecars for `snap_standard_medical_expense_deduction` and `snap_homeless_shelter_deduction_available`, both anchored to the upstream federal slots those Texas bulletin values fill.
  - Added checked-in Axiom Encode benchmarks for both Texas lanes and extended the PolicyEngine adapter so the standard medical deduction compares as a numeric parameter-backed state setting and the homeless shelter lane compares as a delegated boolean availability flag.
  - The first standard-medical replay already matched PolicyEngine and reviewer expectations, but CI failed only because the source slice repeated the `$35` documentary threshold in both the quoted bulletin fact and the explanatory text. Narrowing the source text to one substantive occurrence fixed that without changing the generated RuleSpec.
  - The first homeless-availability replay fixed the earlier missing-oracle-target problem, then failed only because the source slice repeated `198.99` in both the bulletin fact and an instruction sentence. Removing the second documentary repetition let the queue auto-requeue the manifest and close the rerun cleanly.
  - The local event-driven queue's manifest/source hash tracking did real work here: after the queue-runner bug fix, the corrected Texas source changes automatically requeued the blocked homeless lane and drove it to a clean ready state without manual queue surgery for the second pass.
- Primary evidence paths:
  - us_snap_tx_standard_medical_expense_deduction_refresh.yaml (`../benchmarks/us_snap_tx_standard_medical_expense_deduction_refresh.yaml`)
  - us_snap_tx_homeless_shelter_deduction_available_refresh.yaml (`../benchmarks/us_snap_tx_homeless_shelter_deduction_available_refresh.yaml`)
  - [evals.py](../src/axiom_encode/harness/evals.py)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - snap_standard_medical_expense_deduction_tx.meta.yaml (`../../rulespec-us-tx/sources/targets/txhhs/twh/current-effective/snap_standard_medical_expense_deduction_tx.meta.yaml`)
  - snap_homeless_shelter_deduction_available_tx.meta.yaml (`../../rulespec-us-tx/sources/targets/txhhs/twh/current-effective/snap_homeless_shelter_deduction_available_tx.meta.yaml`)
  - axiom_encode-snap-standard-medical-expense-deduction-tx-20260414t151911 (`../artifacts/eval-suites/axiom_encode-snap-standard-medical-expense-deduction-tx-20260414t151911`)
  - axiom_encode-snap-homeless-shelter-deduction-available-tx-20260414t152243 (`../artifacts/eval-suites/axiom_encode-snap-homeless-shelter-deduction-available-tx-20260414t152243`)

### 2026-04-14: Texas SNAP BBCE parameter lanes close on TANF-NC gross and asset limits

- Hypothesis:
  - Texas SNAP BBCE should be modeled through the same TANF non-cash parameters that PolicyEngine already uses for categorical eligibility: the TANF-NC gross-income limit ratio and TANF-NC asset limit. The likely risk was ontology drift between the state handbook wording and the PE surface, not a generation problem, because the Texas Works Handbook states the 165 percent FPIL gross test and the $5,000 liquid-plus-excess-vehicle resource test directly.
- Effect:
  - Added exact `rulespec-us-tx` source slices and delegated `sets` sidecars for `snap_tanf_non_cash_gross_income_limit_fpg_ratio` and `snap_tanf_non_cash_asset_limit`, both anchored to the federal SNAP categorical-eligibility slot under `7 USC 2014(a)`.
  - Added checked-in Axiom Encode benchmarks for both Texas lanes and extended the PolicyEngine adapter so these benchmark targets compare directly against `gov.hhs.tanf.non_cash.income_limit.gross` and `gov.hhs.tanf.non_cash.asset_limit`.
  - Locked the manifest and adapter shape with targeted eval and validator tests before letting the event-driven Codex queue pick the new manifests up automatically.
  - The local queue discovered both manifests without manual queue edits, ran the asset-limit lane first and the gross-limit lane second, and both closed fully ready on success, compile, CI, zero ungrounded numerics, generalist review, and PolicyEngine.
- Primary evidence paths:
  - us_snap_tx_tanf_non_cash_gross_income_limit_fpg_ratio_refresh.yaml (`../benchmarks/us_snap_tx_tanf_non_cash_gross_income_limit_fpg_ratio_refresh.yaml`)
  - us_snap_tx_tanf_non_cash_asset_limit_refresh.yaml (`../benchmarks/us_snap_tx_tanf_non_cash_asset_limit_refresh.yaml`)
  - [validator_pipeline.py](../src/axiom_encode/harness/validator_pipeline.py)
  - [test_evals.py](../tests/test_evals.py)
  - [test_rulespec_validation.py](../tests/test_rulespec_validation.py)
  - snap_tanf_non_cash_gross_income_limit_fpg_ratio_tx.meta.yaml (`../../rulespec-us-tx/sources/targets/txhhs/twh/current-effective/snap_tanf_non_cash_gross_income_limit_fpg_ratio_tx.meta.yaml`)
  - snap_tanf_non_cash_asset_limit_tx.meta.yaml (`../../rulespec-us-tx/sources/targets/txhhs/twh/current-effective/snap_tanf_non_cash_asset_limit_tx.meta.yaml`)
  - axiom_encode-snap-tanf-non-cash-asset-limit-tx-20260414t154707 (`../artifacts/eval-suites/axiom_encode-snap-tanf-non-cash-asset-limit-tx-20260414t154707`)
  - axiom_encode-snap-tanf-non-cash-gross-income-limit-fpg-ratio-tx-20260414t155013 (`../artifacts/eval-suites/axiom_encode-snap-tanf-non-cash-gross-income-limit-fpg-ratio-tx-20260414t155013`)

## Open Documentation Debt

- Add before/after metric snapshots for every kept harness change rather than relying on commit messages.
- Promote the local archive registry into a shareable or publishable artifact index for externally cited runs.
- Add a formal result appendix generator for one-run and multi-run suite outputs.
- Add a public-comparison note clarifying that standard UK FRS-based microdata is licensing-constrained, so public record-level dashboards need a separate open dataset.
