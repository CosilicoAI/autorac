# PolicyEngine Oracle Registry

RuleSpec is the executable legal computation layer. PolicyEngine mappings are
oracle-adapter data, so they live in `axiom-encode`, outside RuleSpec and outside
`rulespec-*` repositories.

## Boundary

RuleSpec may contain legal identity and execution semantics: canonical legal IDs,
imports, entities, periods, units, dtypes, formulas, parameters, tables,
effective dates, and minimal authority pointers.

RuleSpec must not contain PolicyEngine/TAXSIM mappings, reviewer metadata,
ingestion traces, AKN/source-document artifacts, oracle tolerances, scenario
defaults, or harness heuristics.

## Runtime admission

PolicyEngine evidence requires one RuleSpec-pinned, sealed runtime. Every
canonical `rulespec-<country>` checkout must commit exactly one
`.axiom/policyengine-runtime.toml`:

```toml
[policyengine_runtime]
schema = "axiom-policyengine-runtime-pin/v1"
git_commit = "<lowercase 40-character commit SHA>"
```

The pin deliberately has no repository field. Axiom Encode maps the canonical
country to the literal official PolicyEngine HTTPS URL in code, fetches the pin
into a fresh isolated Git object database, and compares every checkout file and
executable bit with that fetched tree. Checkout remotes, branches, tracking
refs, Git configuration, credentials, and caller strings are not authority.

Pass the absolute sealed checkout path as `--policyengine-runtime-root` to
`validate`, `eval-suite`, `eval-suite-revalidate`, `eval-suite-report`, and
`eval-suite-archive` whenever the validation or suite selects PolicyEngine:

```bash
axiom-encode validate /path/to/rulespec-us/us/statutes/example.yaml \
  --oracle policyengine \
  --policyengine-runtime-root /absolute/path/to/policyengine-us \
  --corpus-path /absolute/path/to/axiom-corpus \
  --axiom-rules-engine-path /absolute/path/to/axiom-rules-engine
```

The runtime must use a protected detached `HEAD` at the exact pin. Its checkout
and complete import closure must be root-owned and caller/ACL-nonwritable under
protected root-owned ancestors, while the launcher runs unprivileged with equal
real and effective UIDs. Symlinks, writable or special entries, and host-owned
Python components are rejected.

`.venv` is a checkout-owned standalone CPython prefix, not a conventional venv
linked to a host interpreter or standard library. It contains a regular
`.venv/bin/python`, the complete `lib/pythonX.Y` standard library, exactly one
corresponding `site-packages`, dependencies, metadata, and native extensions.
`pyvenv.cfg`, `.pth`, `sitecustomize.py`, and `usercustomize.py` are forbidden.

Admission and every oracle launch use `python -I -S -B`. A trusted bootstrap
prepends only the validated checkout and `site-packages`; all initial Python
paths and prefixes must remain inside `.venv`. The parent process—not the
probed child—hashes the full checkout, stdlib, dependency, metadata, native-file,
and PolicyEngine closure before and after imports and around each oracle run.
The resulting `axiom-policyengine-runtime/v2` identity and canonical SHA-256
flow through suite admission, results, verdicts, reports, and archives. Pin or
closure mutation, resume mismatches, and zero comparable evidence fail closed.
Each oracle subprocess has a fixed 300-second wall timeout and a 45-second idle
timeout; idle expiry is reported with the shorter limit that actually fired.

TAXSIM and the former `all` selector are removed. Validation, suite evidence,
prediction scores, run records, telemetry, and exports accept only PolicyEngine
or no oracle; old payloads containing TAXSIM fields are rejected.

## Mapping Keys

Mappings are keyed by canonical Axiom legal output IDs:

```text
us:statutes/7/2014/e/2#snap_earned_income_deduction
```

Friendly suffixes such as `snap_earned_income_deduction` are not registry keys.
The harness may display suffixes for humans, but oracle lookup uses legal IDs.

## Legal ID Lifecycle

Canonical legal IDs should remain stable while the legal authority and computed
concept remain stable. When a provision is renumbered, repealed, replaced, or
split, do not silently retarget the old ID. Add an explicit alias/deprecation
record in oracle or registry metadata and keep the mapping rationale clear.

If an amendment changes the computation without changing the provision path,
RuleSpec should model the change with effective dates under the same legal ID.
If the source authority moves or the computed concept changes, create a new
legal ID and mark the old mapping deprecated outside RuleSpec.

## Coverage Accounting

Oracle runs report coverage separately from match score:

- `comparable`: outputs actually compared to PolicyEngine
- `passed`: comparable outputs that matched
- `failed`: comparable outputs that mismatched
- `unmapped`: legal IDs with no PolicyEngine mapping
- `unsupported`: known not-comparable or unsupported scenario cases
- `adapter_errors`: scenario construction, runtime, or parse failures
- `setup_errors`: missing PolicyEngine runtime/setup failures

Unmapped and unsupported outputs are not scored as mismatches, but they must stay
visible so a green score cannot hide weak oracle coverage.

Known non-comparable legal IDs should be registry entries with
`mapping_type: not_comparable` and a short rationale. Truly new legal IDs should
remain `unmapped` until someone classifies them.

When a non-comparable output looks close to PolicyEngine, keep the adjacent
`policyengine_variable` or `policyengine_parameter` in the registry entry. If the
candidate has already been reviewed and should not stay high in the queue, add
`candidate_priority: P4` and record the concrete reason in `rationale`.

For large legal namespaces where most outputs are source-specific intermediates,
the registry may use `prefixes` entries. Prefix entries only classify
`not_comparable` outputs. Exact legal-ID mappings always override prefixes, so a
specific output can still become PolicyEngine-comparable later without changing
the RuleSpec file.

## Mapping Types

Use `mapping_type: direct_variable` when a RuleSpec output can be compared to a
PolicyEngine variable:

```yaml
- legal_id: us:statutes/26/3101/a#oasdi_wage_tax
  mapping_type: direct_variable
  policyengine_variable: employee_social_security_tax
```

Use `mapping_type: parameter_value` when the RuleSpec output is a statutory or
policy parameter that PolicyEngine stores in its parameter tree rather than as a
variable:

```yaml
- legal_id: us:statutes/26/3101/a#oasdi_wage_tax_rate
  mapping_type: parameter_value
  policyengine_parameter: gov.irs.payroll.social_security.rate.employee
```

Parameterized PE tables can be keyed by a fixed `parameter_key`, by several
fixed `parameter_keys` that must agree, or by a RuleSpec test input through
`parameter_key_input` and `parameter_key_map`:

```yaml
- legal_id: us:statutes/26/3101/b/2#additional_medicare_wage_tax_threshold
  mapping_type: parameter_value
  policyengine_parameter: gov.irs.payroll.medicare.additional.exclusion
  parameter_key_input: filing_status
  parameter_key_map:
    0: SINGLE
    1: JOINT
    2: SEPARATE
    3: HEAD_OF_HOUSEHOLD
```

For nested PE parameter objects, use `parameter_key_path`. Path entries can
select from RuleSpec inputs and can optionally bound numeric keys before
lookup. This is for source-stated capped rows such as "10 or more", where the
RuleSpec output and PE parameter table share the same capped row:

```yaml
- legal_id: us-ca:policies/cdss/calworks/maximum-aid-payment-region-1#calworks_region_1_maximum_aid_payment
  mapping_type: parameter_value
  policyengine_parameter: gov.states.ca.cdss.tanf.cash.monthly_payment.region1
  parameter_key_path:
    - input: assistance_unit_is_exempt
      key_map:
        "True": exempt
        "False": non_exempt
    - input: persons_on_aid
      max_value: 10
```

Use `parameter_keys` when one legal output intentionally groups multiple PE
table cells that should have the same value:

```yaml
- legal_id: us:policies/irs/rev-proc-2025-32/standard-deduction#additional_standard_deduction_married_or_surviving_spouse
  mapping_type: parameter_value
  policyengine_parameter: gov.irs.deductions.standard.aged_or_blind.amount
  parameter_keys:
    - JOINT
    - SEPARATE
    - SURVIVING_SPOUSE
```

Keep these mappings in `axiom-encode`; do not put oracle parameter paths in
RuleSpec repositories.

## Repository Coverage Report

Use the coverage command before assigning autonomous encoding work and after a
batch lands:

```bash
axiom-encode oracle-coverage --root /path/to/rulespec-us --program snap
```

The report scans only the explicit canonical country checkout, constructs
canonical legal IDs for every executable `kind: parameter` and `kind: derived`
output, and classifies each output as:

- `comparable`: an exact registry mapping can be run against PolicyEngine
- `known_not_comparable`: the registry deliberately classifies the output as out
  of oracle scope, usually because it is a legal intermediate or source-specific
  assembly point
- `unmapped`: no registry entry or prefix covers the output yet
- `pending_classification`: a temporary declaration records classification debt;
  release gates reject this status

Autonomous agents should treat `unmapped` as work queue material. They should
either add an exact oracle mapping, add a narrow `not_comparable` classification
with a rationale, or improve the RuleSpec if the unmapped output is accidental.
Do not hide unmapped outputs in RuleSpec metadata.

CI jobs that need complete classification can use:

```bash
axiom-encode oracle-coverage \
  --root /path/to/rulespec-us \
  --fail-on-unmapped \
  --fail-on-pending
```

CI jobs that need oracle-tested RuleSpec should also fail on comparable outputs
missing from companion tests:

```bash
axiom-encode oracle-coverage \
  --root /path/to/workspace \
  --fail-on-unmapped \
  --fail-on-pending \
  --fail-on-untested-comparable
```

For PolicyEngine program-surface parity, a legal-ID mapping is not enough. Active
surfaces that the report classifies as `wired` still require a Populace-backed
comparison before they can be treated as PE-parity-complete. The program-surface
manifest can record this under `populace_validation`:

```yaml
populace_validation:
  status: validated
  command: axiom-encode tax-populace-compare --variable income_tax --year 2024
  dataset: populace-us-2024
  last_run: '2026-06-30'
```

`validated` entries must cite a Populace comparison command, dataset, and run
date. Use `pending_validator`, `blocked`, or `not_applicable` only with a
concrete rationale. CI jobs that want to forbid unvalidated wired program
surfaces can use:

Populace comparison commands load a local engine-native H5 artifact first when
one is available, including cached `policyengine/populace-*` snapshots. Set
`AXIOM_POPULACE_US_2024_H5`, `AXIOM_POPULACE_US_H5`, `AXIOM_POPULACE_H5`,
`AXIOM_POPULACE_DATASET`, or `AXIOM_POPULACE_DATA_PATH` to pin validation to a
specific local artifact.

### CI ownership of the federal income tax (FIIT) comparison

`axiom-encode tax-populace-compare` is the FIIT comparison harness. As of the
A9 runner unification it is **deprecated for
CI / weekly reporting**: the reported FIIT number and its dashboard artifact are
produced by `axiom-oracles`, which wraps this command in its standard
comparison runner (`comparisons/fiit-ecps.yaml` → `scripts/run_comparison.py`),
normalizes the `--json` output — including the `dataset_identity` provenance
block — into the `axiom.comparison_report.v2` schema, and refreshes it through
the weekly matrix. The ~17k LOC oracle bridge stays here in `axiom-encode`; only
the run/report entry point moved.

Invoking `tax-populace-compare` directly is still fully supported and is the
right tool for **local development and residual triage** (e.g. a quick `--json`
dump or a `--tax-unit-id` re-check). Just don't wire it into CI or treat its
output as the reported comparison — run `run_comparison.py fiit-ecps` in
`axiom-oracles` for that.

```bash
axiom-encode oracle-coverage \
  --root /path/to/workspace \
  --include-program-surfaces \
  --fail-on-unvalidated-populace-surfaces
```

## Candidate Triage

Use the candidate command to turn coverage into a priority queue:

```bash
axiom-encode oracle-candidates --root /path/to/rulespec-us --program snap
```

The command prioritizes untested comparable mappings, unmapped outputs that look
like exact PolicyEngine variables, and known adjacent targets that may deserve a
small adapter. `candidate_priority` can lower already-reviewed non-comparables
without hiding them from the report.

## Cloud Queue Export

Use the cloud queue command to export deterministic work items before assigning
parallel encoding work:

```bash
axiom-encode cloud-queue --root /path/to/rulespec-us --json
```

The queue is model-free orchestration input. It converts PolicyEngine program
surfaces into explicit work items without running encoders, opening branches, or
claiming that any surface is legally comparable. The current schema is:

```text
axiom-encode/policyengine-cloud-queue/v1
```

Each item includes:

- `action`: one of `ingest_source`, `encode_rulespec`, or
  `wire_oracle_mapping`
- `priority`: the program-surface priority such as `P1` or `P2`
- `target_repo` and `target_prefix`: where generated RuleSpec or source work
  should eventually land
- `policyengine_variable`: the oracle surface that motivated the item
- `lock_scopes`: deterministic repo, prefix, legal-ID, and source locks for
  future cloud workers
- `oracle_expectation`: what a successful worker should prove or classify

By default the queue includes `pending_source_ingestion`,
`pending_rulespec_encoding`, `pending_oracle_mapping`, and
`deferred_jurisdiction` surfaces. Deferred jurisdiction surfaces are still
PolicyEngine-modeled parity gaps and should not be hidden unless a caller is
explicitly planning only ready-to-encode work. Because Axiom now uses
country-level RuleSpec repositories, deferred jurisdiction surfaces queue as
source-ingestion work rather than repo bootstrap work. Use
`--exclude-deferred-jurisdictions` to omit those source-ingestion tasks.

Cloud workers should emit artifacts using the companion run-artifact schema:

```text
axiom-encode/policyengine-cloud-run-artifact/v1
```

At minimum, run artifacts should preserve the work item ID, encoder version and
git SHA, corpus reference, RuleSpec base SHA, model and prompt hash, generated
diff, validation logs, oracle results, retry history, and final classification.
