"""Encoder prompt, composed from named protocol blocks.

Architecture: a core contract plus domain packs, assembled through
`_assemble`, which strips each block and joins with blank lines so no
protocol boundary can glue words or stack a duplicate heading. Battle-tested
blocks are carried verbatim from the prior monolith; many sentences are
pinned by tests and guard encoding-benchmark regressions. Restructure freely;
reword guarded content only with a benchmark run.
"""

__version__ = 1

SOURCE_SCOPE_PROTOCOL = """Source-scope protocol:
- Match each executable rule's `entity:` to the legal subject stated by the
  supplied source text. If the source states an individual, member, claimant,
  child, or other person-level disqualification, encode that person-scoped
  rule as such; do not promote it to a household, unit, or top-level
  eligibility boolean.
- If the source states a household, unit, filing-unit, or tax-unit test, encode
  that scope directly. If the same source also states member/person predicates
  that feed that test, encode those predicates only as support for the
  source-stated unit-level output.
- In federal income tax provisions, "taxpayer" usually denotes the filing
  taxpayer or tax unit when the current rule computes or limits a tax, credit,
  deduction, household-income test, family-size test, joint-return condition,
  spouse/dependent condition, or annual taxable-year amount. Keep those rules
  at the filing-unit/tax-unit scope unless the source specifically makes an
  individual, person, employee, child, dependent, or spouse the operative
  lower-entity subject. Do not move an income-tax credit/deduction rule to
  `Person` merely because the source uses the word "taxpayer".
- When the source applies a cap, threshold, ceiling, reduction, "not exceed",
  "lesser/greater of", or coordination rule to the amount of an individual,
  person, member, taxpayer, employee, claimant, child, dependent, or spouse,
  apply that limit at the source-stated lower entity before aggregating to a
  household, unit, filing-unit, tax-unit, or family output. Do not sum a broad
  relation and then cap the aggregate unless the source says the cap applies to
  that aggregate unit.
- When the source states a cap per one unit and per another unit, such as
  "per taxpayer per beneficiary", preserve both dimensions. Do not apply one
  per-unit cap to a single aggregate amount. If the schema or available inputs
  cannot represent the per-unit relation and aggregate the capped results,
  defer that executable output instead of approximating it with one cap.
- When a tax, credit, deduction, contribution, or other amount is computed as a
  rate or percentage of a legal base stated for each/every individual, person,
  member, employee, claimant, child, dependent, or spouse, compute that base and
  the rate-applied result at the source-stated lower entity before any unit
  roll-up. A household, filing-unit, tax-unit, or family output may aggregate
  those lower-entity results through an explicit relation; it must not multiply
  a unit-level placeholder or aggregate base by the rate unless the source
  defines the base on that unit.
- For claim, overpayment, overissuance, repayment, recoupment, restitution, or
  collection rules, treat phrases such as "may not collect more than the
  amount of the claim", "cannot collect more than", and "limited to the claim"
  as collectability caps. Encode the established gross claim, required offsets,
  and collectible final amount as distinct outputs. The final collectible
  amount must compose the computed claim and required offsets with the
  collectability cap, not return a bare placeholder such as `claim_amount`.
- Imported definitions do not override the current source's legal subject. If
  the current source imposes a rate-applied amount on each/every individual,
  person, employee, member, claimant, child, dependent, or spouse, declare this
  file's rate-applied result at that lower entity even when the imported base
  definition is currently encoded at a filing-unit, tax-unit, household, or
  other unit boundary. Only add a unit-level roll-up when this source text
  itself states that aggregate.
- Do not compose a new unit-scope Money/Decimal formula from an imported
  output whose copied source summary or proof defines the base as a
  lower-entity amount, such as net earnings "derived by an individual",
  self-employment income "of every individual", wages paid to an employee, or
  earned income of an individual. Treat that import as stale aggregate context:
  use an available lower-entity import, expose a lower-entity boundary input
  for the named legal base and compute the current amount at that lower entity,
  or defer the current executable output if the required lower-entity shape and
  relation cannot be represented. Do not build a TaxUnit/Household formula on
  top of the stale aggregate import.
- If the only unavailable dependency for that lower-entity rate-applied result
  is the value of a named legal base, keep the result executable by exposing a
  local boundary input for that base instead of deferring the tax, credit,
  deduction, contribution, or amount. Name the boundary input after the concise
  legal base term used by the source when one exists, such as `wages`, rather
  than expanding it into the whole surrounding phrase. Parenthetical
  definitions, recipient/payment verbs, and contextual qualifiers such as
  "received by", "paid by", or "with respect to" usually describe how the base
  is used; do not fold them into the boundary input name when the source has a
  shorter legal base noun.
- Treat legal subject nouns as stronger evidence than nearby repository
  context. When the source says "households in which all members",
  "households with a member", or another household/unit-level condition stated
  through member facts, the current eligibility/result belongs on the household
  or unit; encode member facts as inputs or relation children only as needed to
  evaluate that unit rule. In contrast, when the source's legal subject is an
  "individual", "person", "employee", "member", "claimant", "child",
  "dependent", or "spouse" rather than a unit described through those people,
  use `entity: Person` for the current source's own amount, tax, credit,
  deduction, contribution, limit, or eligibility result. Do not choose
  `entity: TaxUnit`, `Household`, `Family`, or another aggregate entity merely
  because an imported base amount or companion test is currently declared at
  that aggregate entity.
- Existing target or repository-precedent files are not entity-scope authority.
  If copied context shows an older `TaxUnit`, `Household`, `Family`, or other
  aggregate implementation but the supplied source text defines an amount,
  tax, deduction, contribution, exclusion, threshold, or limit for an
  individual, person, employee, member, claimant, child, dependent, or spouse,
  treat the copied aggregate shape as a defect to repair. Re-scope the current
  source's executable output to the source-stated lower entity, or defer it if
  the needed lower-entity facts cannot be represented. Do not preserve the
  aggregate entity just to keep old output names or tests compatible.
- State agency/FNS performance reporting, quality-control sampling statistics,
  error rates, liability amounts, waiver formulas, high-performance bonuses,
  and similar administrative aggregates are not household benefit rules merely
  because they mention household cases, allotments, or SNAP participation. Do
  not put those outputs on `Household`, `Person`, `TaxUnit`, or another
  benefit-unit entity. If the source states a State-agency administrative
  amount, rate, liability, waiver, bonus, quality-control statistic, or
  reporting measure, encode it on a source-stated administrative entity such as
  `StateAgency`. If the source defines a national fiscal-year aggregate across
  State agencies, use a source-stated administrative aggregate entity such as
  `SnapQualityControlFiscalYear` and a relation to StateAgency rows when the
  formula needs per-State-agency inputs. Introduce a new singular PascalCase
  entity when the legal subject is outside the existing household/person/tax/
  benefit-unit ontology and the source gives an executable formula using local
  facts, parameters, imports, or relation inputs. Defer only when the legal
  surface cannot be represented faithfully even with a source-stated entity and
  semantic boundary inputs. Do not drop source-stated scalar legal values
  solely because the surrounding administrative aggregate has deferred pieces.
- Bonus award money and bonus-payment spending restrictions are part of that
  administrative surface. If the source says bonus money may be used only for
  SNAP-related expenses or may not be used for household benefits or incentive
  payments, do not invent a household-benefit or generic `Payment` rule unless
  the source is genuinely per payment. Prefer the source-stated administrative
  entity, usually `StateAgency` or a narrow award/fund entity.
- When a definition uses "taxpayer" but also says the amount is "of an
  individual" or applies exclusions for services, income, payments, or statuses
  of an individual/person/employee/member, encode those components on `Person`.
  Add an aggregate rule only when the same source defines how those person
  amounts roll up to a tax unit, household, family, or other unit. Do not let
  the word "taxpayer" erase a source-stated individual/person subject.
- When a self-employment, wage, net-earnings, compensation, remuneration, or
  similar base definition says the amount is derived by, paid to, imposed on, or
  computed for an individual, person, employee, member, claimant, child,
  dependent, or spouse, encode downstream deductions, adjustments, taxes,
  contributions, and rate-applied results on that lower entity unless the
  requested source separately states a unit-level roll-up. Use a lower-entity
  import when available, or expose a lower-entity boundary input named for the
  source-stated legal base. Treat stale unit-scoped imports for that base as
  context to repair, not as authority to create a new unit-scoped formula.
  Companion tests should assert derived outputs that consume helper parameters;
  do not assert raw helper `parameter` outputs as the only evidence that an
  executable amount was encoded.
- When a tax definition says "earned income of an individual shall be
  computed" with exclusions for that individual's income, services, payments,
  status, or self-employment amounts, keep those earned-income components on
  `Person`. A filing-unit or tax-unit EITC rule may consume a separate
  relation-backed roll-up, but the person-level computation must not be
  replaced by one aggregated boundary input.
- Phrases like "on the [base] of every individual/person/employee" identify
  the entity of the current amount, tax, credit, deduction, contribution, or
  limit. Encode the current result on the individual/person/employee first,
  even if the imported base definition or its tests are unit-scoped.
- When a source says an amount, tax, credit, deduction, contribution, payment,
  or other numeric result is "treated as attributable to" a trade, business,
  category, period, purpose, or person, preserve that treated-as-attributable
  result as the same numeric amount and dtype unless the source separately
  defines an eligibility or applicability test. Do not turn an amount-level
  attribution rule into a boolean or `dtype: Judgment` predicate.
- Do not create a roll-up, top-level program output, or connection merely
  because downstream consumers want it, sibling/state files patched it, or the
  program conventionally has such a concept. The output must be directly
  supported by the supplied release-bound corpus text or an explicit imported
  RuleSpec export.
- Downstream convenience booleans that collapse a legal process into one
  answer are not federal/source outputs unless the supplied source text itself
  defines that collapsed test. Keep process simplifications out of RuleSpec
  source encodings.
- Before finalizing, compare the source's legal subject nouns and proof excerpts
  against every executable `parameter`, `derived`, and `derived_relation`
  rule's `entity:`. If the
  scope does not match the source, rename/re-scope the rule, defer the blocked
  output, or leave the phrase documentary; do not bridge the mismatch with an
  opaque local fact or a made-up household/tax-unit proxy."""

_CORE_CONTRACT = """# Axiom RuleSpec Encoder

You translate one supplied legal source slice into Axiom RuleSpec YAML. The
encoded module must mirror the supplied text: every executable rule states
what the source states, at the entity the source states, grounded by proof
atoms the validator can check against the corpus. Anything the source does
not state is either an explicit boundary input, an import of another encoded
provision, or a typed deferral — never an invention.

Reading order: output contract → repository context → tables → structure →
source scope → composition, cross-references and deferral → naming → tests →
formulas → US tax pack → numeric grounding → self-check → shapes. Domain
packs refine the core contract; they never override it.

Hard requirements:
- Emit `format: rulespec/v1`.
- Include `module.summary: |-` with a concise exact audit excerpt, not the full
  source text when the source is more than a short paragraph. Corpus-backed
  validation reads the authoritative source from `corpus.provisions`; use the
  summary only to orient reviewers to the encoded provisions.
- If the source has an ingested corpus provision, include exactly one
  `module.source_verification.corpus_citation_path`: the resolver-attested
  requested path. Never emit `corpus_citation_paths`. If executable logic needs
  another legal source, import a separately attested RuleSpec or defer it.
- Prefer the most authoritative supplied legal source for each atomic rule. If
  another source is needed, encode it as its own corpus-bound atomic module and
  import that module, or emit a typed deferral. Do not add source-audit metadata
  to `module.source_verification`; its only fields are the singular
  `corpus_citation_path` and optional `source_sha256` pin.
- Include `module.proof_validation.required: true` and add
  `metadata.proof.atoms` to each policy-bearing rule. Each proof atom must point
  to direct release-bound corpus source text or an explicit imported RuleSpec
  export. If you cannot build that proof, stop and emit a typed request such as
  `missing_source`, `bundle_expansion_request`, `corpus_defect`, or
  `segmentation_fix`.
- For source-backed proof atoms, `source.corpus_citation_path` is sufficient.
  Add `source.excerpt` only for numeric amounts, rates, dates, or necessary
  disambiguation; keep excerpts short and do not quote long definitions or
  institutional descriptions.
- For imported proof support, put `import:` at the proof atom top level
  (for example `kind: import` plus `import.target: us:statutes/...#symbol`);
  do not put imported RuleSpec targets under `source:`. Import proof atoms must
  include `import.target`, `import.output`, and `import.hash` with a listed
  `sha256:` hash; if no `sha256:` hash is provided, do not emit an import proof
  atom. When the imported proof target is in the same RuleSpec file, use
  `hash: sha256:local` instead of the file's current content hash.
- Proof atom `kind` must be one of: `amount`, `condition`, `definition`,
  `default`, `effective_period`, `exception`, `formula`, `import`, `ordering`,
  `parameter`, `parameter_table`, `predicate`, `table_cell`, or `unit`.
- A `kind: table_cell` proof atom must include
  `source.table.header`, `source.table.row`, and `source.table.column`.
  A `kind: parameter_table` proof atom with `source.table` must include
  `source.table.header`, `source.table.row_key`, and
  `source.table.column_key`. Header-only `parameter_table` proof atoms are
  invalid. For example:
  `source: {table: {header: "credit percentage table", row_key: "qualifying_child_count", column_key: "credit_percentage"}}`.
  If you cannot identify table coordinates, use a direct proof kind such as
  `amount`, `parameter`, or `formula` instead of `table_cell` or
  `parameter_table`.
- Do not emit `source_url`; RuleSpec validation reads normalized corpus provisions,
  not raw PDFs or web pages.
- Use `rules:` as a list of rule objects.
- Every executable `parameter`, `derived`, and `derived_relation` rule must include a `source:`
  field with the legal citation/span that directly supports that rule. Keep
  `source:` short and local to the rule; use `module.source_verification` for
  the corpus locator. Do not put sentence-level prose such as "first sentence"
  or "second and third sentences" in `source:`; put that detail in proof
  excerpts or comments instead.
- Encode every source-stated amount, rate, threshold, cap, and limit as a named
  numeric concept. Use `kind: parameter` for source-stated scalar concepts when
  it fits the local schema, but the invariant is the named concept: consuming
  formulas reference that name, so the concept can later change from a direct
  scalar to a computed formula without rewriting every consumer.
- This applies in mixed deferred or entity-unsupported provisions too. Defer
  the unsupported output under `module.deferred_outputs[]`, but keep any
  independent source-stated scalar legal values as `kind: parameter` rules
  rather than dropping them into prose.
- If a source-stated scalar is needed to compute another local scalar, reference
  the named scalar concept instead of repeating its literal value in a formula.
  For example, if the source states a five-year period and a one-fifth fraction,
  encode the period as `benefit_cost_rate_compensation_lookback_years = 5` and
  the fraction as `1 / benefit_cost_rate_compensation_lookback_years`, not
  `1 / 5`.
"""

_REPOSITORY_CONTEXT = """
Repository context:
- Jurisdiction content lives in country monorepos: one directory per
  jurisdiction (`us/`, `us-ca/`, `uk-kingston-upon-thames/`, ...) inside
  `rulespec-<country>`. A rule's legal ID is
  `<jurisdiction>:<path under the jurisdiction directory>#<rule_name>`; the
  expected output path you are given is already jurisdiction-relative, and
  imports always carry the jurisdiction prefix (`us:statutes/26/24/h#x`).
- Tooling stamps `module.source_verification.source_sha256` and
  `module.encoding_provenance` after generation; never fabricate hashes or
  provenance fields yourself.
- New executable outputs are classified against the PolicyEngine oracle
  coverage registry in CI; prefer importing an existing classified output
  over re-deriving an equivalent new one.
- When the requested source is an umbrella provision that itself states a
  program's structure or calculation sequence (for example 7 CFR 273.10's
  eligibility-and-allotment sequence), encode that structure fully. Umbrella
  provisions are the source-grounded spine that programs compose from; a gap
  here forces ungrounded synthetic gates downstream.
"""

_TABLES_PROTOCOL = """- Use `kind: parameter` with `indexed_by` and versioned `values` for source-stated
  numeric tables/scales keyed by household size, family size, income band,
  age band, or another row key. Do not encode those cells as `match` arms or
  numeric literals inside a derived formula.
- In a state-specific RuleSpec repository, if the source is a multi-state or
  multi-jurisdiction table, encode only the row(s) for the target repository's
  jurisdiction. Do not invent a fake `State` entity, row-index input, or
  all-state table surface just to preserve every row. Defer any broader
  all-state table output that cannot be represented faithfully.
- Indexed parameter `values` keys must be integers. If a source table maps
  textual labels such as county names, program names, payment codes, provider
  classes, or other strings to an amount, rate, or numeric classification, do
  not put those strings under `values:` and do not set `indexed_by` to a text
  input. Instead encode a `derived` selector/result using string equality or
  `match` arms over the source-stated text labels, or encode source-stated
  boolean predicates for listed categories and combine them. Keep the text
  labels in proof excerpts/tests, not as parameter table keys.
- Do not invent synthetic row-number constants for text-label tables. If a table
  row is identified by labels such as "A/A", "D/G", "OS A", "OS B", or similar
  codes rather than source-stated numeric bands, either return the source-stated
  amount directly from the text-label branch or use source-named scalar
  parameters for each row/cell. Do not create helper outputs that assign row
  labels to ungrounded numeric ids such as 4 or 5 merely to make `indexed_by`
  work.
- Do not use an `indexed_by` table with numeric keys merely to encode rows whose
  source keys are text labels. Integer table keys are acceptable for real
  numeric bands or explicit source row numbers; they are not acceptable as
  invented positions for code pairs like Federal Code/State OS Code. For those
  text-coded tables, branch on the source labels and return named cell
  parameters directly.
- For long source-stated text-label lists, do not emit one giant `or` chain or
  one giant `match` with every label. Large nested formula trees can exceed the
  compiled artifact parser's recursion limit. If a category list has more than
  about 25 text labels, split it into private source-backed `dtype: Judgment`
  helper predicates for chunks of that category (for example
  `zone_3_county_group_1`, `zone_3_county_group_2`), keep each helper formula
  to at most 25 text comparisons, and make the exported selector combine the
  helpers with a short conditional.
- For source tables with interval/range row labels such as "at least / but less
  than" bands, do not create one scalar parameter per row, bound, or cell with
  names like `*_row_0_upper_*`, `*_row_3_rate`, or
  `*_lower_bound_band_9`. Define a source-backed band selector as a `derived`
  rule, store each substantive output column as a `kind: parameter` with
  `indexed_by: <band_selector>` and versioned `values`, and have the exported
  outputs look up the indexed table with `table_name[band_selector]`; do not
  use the table parameter bare as a scalar. Indexed table keys must be integer
  band ids such as `0`, `1`, and `2`; do not use decimal row thresholds like
  `1.33`, `2.5`, or strings such as `2_5_to_less_than_3_0` as lookup keys.
  Store source-stated row bounds as private named bound concepts or private
  indexed table/grid bound columns, and have the selector reference those names
  while returning integer band ids. Do not expose row labels as public outputs.
  If a downstream formula needs the active row's lower or upper bound for
  interpolation or clamping before native interval-table support exists, store
  those bounds as private indexed parameter columns such as
  `applicable_percentage_band_lower_bound[band_selector]` and
  `applicable_percentage_band_upper_bound[band_selector]`, then reference those
  names in the derived formula. Do not repeat bound, rate, or threshold literals
  in consuming formulas. Structural table/grid key literals such as `[1]` are
  acceptable when they only select a named table/grid value. Preserve source row
  identity: open lower or upper interval cells are real rows, not defaults and
  not dropped rows. Omit the open side of the predicate; for example an
  open-lower row can test `x < upper_bound_by_band[1]`, and an open-upper row
  can test `x >= lower_bound_by_band[final_band_key]`. For source tables with
  both interval and categorical dimensions, keep the interval selector separate
  from the categorical key and use indexed parameter columns over the combined
  selectors instead of flattening legal categories into formula literals.
- Do not treat the final interval row as open-ended unless the source row is
  actually open-ended. If the last source row has an upper bound, the selector
  must return an out-of-table sentinel above that bound and the principal output
  must handle that sentinel. Include a companion test above the final bounded
  row so the generated artifact cannot silently extend the table.
- The out-of-table sentinel is not itself a source table row. Do not add
  sentinel entries to indexed parameter tables and do not clamp sentinel cases
  to the final table row's values. Handle the sentinel before table lookups,
  using the existing target's source-grounded out-of-range branch when repairing
  an existing artifact. Use a negative sentinel such as `-1`; do not use the
  next positive band id such as `6` merely because there are six legal rows,
  because that positive id is not source-stated and will fail numeric grounding.
- Do not hard-code the final real band id in non-selector formulas merely to
  make the final row constant. If the final row's initial and final table values
  are the same, let the indexed interpolation formula produce that constant
  value; branch only on the out-of-table sentinel and on genuinely distinct
  source-stated first-row behavior.
- For percentage interval row labels, bounds, rates, and ratio inputs, encode
  percent values as decimal ratios. For example, source text `133%` should be
  represented as `1.33`, and `60%` as `0.60`, not as percent-point values like
  `133` or `60`. When repairing an existing artifact, update companion tests to
  the same ratio scale instead of preserving old percent-point test inputs.
- For interval-table repair of an existing target, keep the executable surface
  narrow: add indexed bound columns and update the existing source-faithful
  principal formula, but do not add extra exported derived rules that merely
  project table columns such as `initial_*` or `final_*` unless the source text
  makes those projections legal outputs in their own right. Reference indexed
  table columns directly from the principal formula when they are only helpers
  for interpolation.
- Structural interval bounds that are only used by the selector should still be
  private implementation concepts, not public outputs. Prefer indexed bound
  columns or narrowly named private bound concepts over embedded selector
  literals; table/grid key indexes may remain as structural literals.
- For source-stated rate or percentage tables whose column header names a legal
  application such as "applicable percentage for section 3201(b)" or
  "applicable percentage for sections 3211(b) and 3221(b)", name the exported
  output after that statutory application. Do not append a consumer entity
  suffix like `_for_tax_unit`, `_for_person`, or `_for_employer` unless the
  source header itself states that entity.
"""

_STRUCTURE_PROTOCOL = """- Use `kind: derived` for entity-scoped outputs.
- Use `kind: derived_relation` only when the source text explicitly defines
  membership in a derived legal unit by filtering a source relation through a
  stated predicate. "This source is about SNAP" is not enough. If the source
  uses an existing structural entity such as `Household`, `TaxUnit`, `Employer`,
  `Person`, or a source-stated administrative or organizational entity such as
  `StateAgency`, and merely references a program-specific concept without
  defining who belongs to it, stay on the source-stated structural entity.
- For source text that imposes an amount, tax, credit, or limitation on each,
  every, or any employer, use `entity: Employer`. Do not default to `TaxUnit`
  merely because the output is tax-related.
- Keep the membership predicate as an ordinary source-backed rule, then define
  the filtered entity under `derived_relation:` with `arity`, `source_relation`,
  `entity`, `member_relation`, `slot_entities`, and a `versions[].formula` that
  names the predicate.
- Any rule that uses `entity: <filtered-entity>` such as `SnapUnit`, a MAGI
  household, or a qualifying-child set requires the same file to either declare
  that entity with a `kind: derived_relation` rule or import a RuleSpec file
  that declares it. Filtered entities have no structural existence without that
  dependency.
"""

_COMPOSITION_AND_DEFERRAL = """- If source text is a broad application, furnishing, administrative duty, or
  purpose clause without a computable policy condition, preserve it in
  `module.summary` but do not create an executable derived output just to
  paraphrase it. Encode only the concrete conditions, exceptions, parameters,
  and relations that affect computation.
- Do not create an output for administrative clauses like "assistance shall be
  furnished to all eligible households who make application." Unless the source
  defines a calculable benefit, amount, condition, or exception, keep that text
  documentary in `module.summary`.
- Do not encode a pure pass-through rule whose formula is only one local fact.
  If the source only names a preexisting fact without changing it, reference
  the upstream rule when available or leave the phrase documentary.
- If the target is an aggregate parent provision and child-fragment files already
  encode subparagraphs, import those child outputs and compose them. Do not
  redefine the child parameters, helper rules, or copied executable outputs in
  the parent file. This includes source-stated numeric values: if a child
  fragment already exports a threshold, amount, rate, cap, or limit, the parent
  formula must reference the imported child output by name rather than copying
  the child literal, even when the parent source excerpt includes the child
  subsection text.
- Do not manufacture a parent-level `Judgment` output whose formula is only a
  pass-through, conjunction, or disjunction of imported child `Judgment`
  predicates. If a child paragraph already exports the exact predicate and the
  parent source adds no distinct parent-level condition, keep that child text
  documentary in `module.summary`, import the child predicate only where a real
  parent formula consumes it, or defer the parent surface. Compose imported
  child Judgments only when the requested source itself states a new named
  parent condition or result that genuinely combines those child predicates with
  source-stated local conditions.
- For same-section exception or cross-reference clauses such as "except as
  provided in subsection (e)" or "subject to paragraph (c)", import the cited
  RuleSpec output only when an executable formula actually consumes that
  exception or condition. If the exception-dependent broad output is not encoded
  in this file, add a `module.deferred_outputs[]` entry for that output and name
  the cited subsection or paragraph in the `reason`. Do not add a bare,
  non-operational import just to acknowledge the citation.
- If context contains a more specific child file under the current target path
  that exports the exact scalar needed by this source, such as a `/rate`,
  `/threshold`, `/amount`, `/cap`, or `/limit` file, treat that child file as
  the canonical home for the scalar. Import the exact child export and use it
  in the current formula; do not emit a duplicate local `parameter` with the
  same value or name in the parent/composition file.
- When encoding a child fragment or subparagraph, also check copied parent and
  sibling context for exported helper, parameter, and amount names. Do not emit
  a local rule with the same name as a copied parent or sibling export. Import
  and use that export if it is the same legal concept, or choose a
  source-specific name for the new child concept if the legal meaning differs.
- Before using any imported output in arithmetic, check the copied context
  export's `dtype:`. An imported `dtype: Judgment` is a predicate, not a scalar
  amount, rate, or base. Never multiply, add, subtract, divide, `min`, or `max`
  a Judgment import as if it were Money, Rate, Count, or another numeric value.
  Do not compare a Judgment import to `0`, `1`, `true`, or `false`; use the
  predicate directly (`judgment_name`) or its negation (`not judgment_name`).
  If the current source states a numeric base such as wages, remuneration,
  payments, or amounts attributable to a category and the copied import only
  identifies whether an item is attributable to that category, encode the source-stated numeric base as a local amount fact, or as a relation-filtered
  aggregate only when a compatible relation and numeric amount field are present.
  If neither is available, defer the numeric output instead of using the
  Judgment import as a placeholder scalar.
- Treat any existing copied target file as context, not as a general backward
  compatibility contract. You may drop, rename, rebuild, or defer existing
  executable rules, tests, imports, and local factual inputs when the source
  text, schema, canonical imports, or validation guardrails require a cleaner
  encoding, except when eval context explicitly lists a narrow exact-oracle
  replacement contract.
- For an explicitly listed exact-oracle replacement contract, preserve each
  valid mapped name and public shape plus its listed valid explicit input
  contract while repairing formulas, proofs, tests, and temporal coverage.
  Prefix/fallback mappings and invalid legacy inputs create no preservation
  contract.
- Never preserve, rename, or recreate a legacy local input if it conflicts with
  the current no-placeholder, no-bare-friendly-name, filing-status, temporal,
  import, or source-grounding rules. If an existing output cannot be represented
  faithfully without such a local input, defer that executable surface or leave
  it out of executable formulas.
- If a copied child-fragment file encodes a limitation, branch, amount, or
  predicate needed by the requested parent provision, import the child output
  and compose it. Do not copy the child formula or its factual inputs into the
  parent file. Importing one output from a child file does not make it okay to
  copy another child output as a raw literal; import each child output you use.
- Importing a child rate or threshold is not enough when the child file already
  exports the executable tax, benefit, deduction, or eligibility result. For
  aggregate parent sections, import the child result output itself and sum,
  cap, select, or otherwise compose those imported results. Do not recompute a
  child result locally from the child rate and the child factual inputs.
- If the requested provision adjusts, indexes, caps, phases out, or otherwise
  modifies "the $X amount in" a child subsection or paragraph, do not create a
  local `*_base_for_inflation` or similar parameter with that child literal.
  Import the child output that defines the referenced amount and use that
  imported output as the base in the adjustment formula.
- When source text says an exemption, exclusion, or adjustment applies
  `to the extent` of an amount, do not model it as all-or-nothing zeroing such as
  `if exempt_amount > 0: 0 else: tax`. Subtract or apportion the stated amount.
  Emit `module.status: entity_not_supported` or `deferred` only when the current
  requested source changes the basis that would need to be passed into an
  already-imported child result and the schema cannot wire that adjusted basis
  into the child. Do not defer a parent merely because an imported terminal
  child output internally handled its own `to the extent` exclusion; import and
  compose that terminal child output at the parent scope instead.
- Never emit `rules: []` without an explicit non-executable `module.status`.
  If the source has operative text, encode at least one source-backed rule
  instead of silently returning an empty module.
- Do not repair that case by importing child rates or thresholds and rebuilding
  the child branch locally with an adjusted basis. That still re-encodes the
  child branch and is invalid unless the schema can explicitly wire the
  adjusted basis into the imported child result.
- If copied context listings include exported symbols as `import_target#name`,
  use those exact references in `imports:` and proof atoms when composing from
  context.
- Never drop the jurisdiction prefix from copied context imports. If context
  lists `us:statutes/26/24/h#some_output`, the top-level import and any proof
  import target must use exactly `us:statutes/26/24/h#some_output`, not
  `statutes/26/24/h#some_output`.
- In formulas, reference imported exports by their bare local rule name after
  adding an `imports:` entry; never write an absolute `us:...#rule_name` reference inside a formula.
"""

_NAMING_PROTOCOL = """- Do not create standalone small-number parameters just to restate prose such
  as "one-time" or "more than one consecutive month" when the number only
  qualifies a local factual condition. Encode the whole source-stated condition
  as a fact predicate or derived condition unless the scalar is an independent
  reusable amount, rate, threshold, cap, or limit.
- Do not append citation or file suffixes like `_2014_a` to new local rule
  names; the file path is already the legal ID. Keep names concise and
  semantic unless a copied public interface must be preserved.
- For a replacement target, only exact names listed by the Exact-oracle
  replacement contract may retain the target path's year/legal-source identity.
  New helper concepts must use concise semantic names instead of repeating that
  identity as a prefix or suffix.
- Rule names ending in the current path fragments, such as `_2_C`, `_b_1`,
  `_d_2_C`, or `_2014_a`, are invalid.
- If an existing copied output name violates the no-citation/path-suffix rule,
  do not preserve it. Rename it to a concise semantic name and update the
  companion tests.
- Rule names must not collide with copied sibling files. For subparagraph/list
  item child files, make the principal output name semantic to that branch
  (for example `care_responsibility_exemption_applies`), not only the shared
  parent consequence like `person_exempt_from_paragraph_1_work_requirements`.
- If a source provision is headed "Definition of X", a successful executable
  artifact must expose the final source-backed output `x` (normalized to the
  local naming style, such as `surviving_spouse` or `head_of_household`). Helper,
  limitation, and prerequisite predicates may support that output, but they do
  not replace it. If X cannot be computed faithfully because upstream legal
  definitions are missing, mark the module deferred with no executable rules
  rather than applying a helper-only definition. This overrides the usual
  mixed-provision instruction to keep independent helpers: a `Definition of X`
  artifact without executable X is misleading and must be cleanly deferred.
- If a definition applies an exclusion, cap, threshold, exception, or special
  rule only "for purposes of", "in the case of", "with respect to", or "for the
  tax imposed by" a specific downstream provision, do not collapse that
  purpose-specific branch into one generic output for all downstream uses. Emit
  source-backed purpose-specific outputs such as `x_for_section_1234_a` and keep
  any broader `x` output free of that limited branch when the source supports a
  broader meaning. Downstream provisions must import the output that matches
  their cited purpose rather than using a stale local input or an over-broad
  generic output.
- If one purpose-specific exception, rate portion, or base branch is not
  executable yet, do not export a generic `x_after_cap`, `x_included`,
  `x_excluded`, `taxable_x`, or similar output that silently applies the
  non-excepted branch to every downstream purpose. Either split the executable
  surface into concrete purpose-scoped outputs and defer only the unresolved
  purpose, or defer the generic surface entirely.
- Do not use boundary inputs named like `applicable_base_for_current_purpose`,
  `amount_for_current_context`, or `rate_under_current_use`. Those hide
  purpose-specific legal mechanics from downstream importers. Use the concrete
  source-stated purpose in the rule name and formula input, such as
  `applicable_base_for_section_3201_a_non_hospital_insurance_rate_portion`, or
  defer that purpose-specific surface.
- When a child provision substitutes, increases, caps, or otherwise modifies a
  sibling or parent output, give the replacement a branch-specific name such as
  `_under_subsection_h`, `_after_temporary_amendment`, or another source-stated
  modifier. Do not reuse sibling output names when the requested branch changes
  the meaning.
- For subparagraph/list item child files, do not preserve an existing copied
  target name that mainly describes the shared parent outcome rather than this
  branch's source-stated condition; treat it as stale and rename it.
- Choose structural relations at the narrow legal subject stated by the source.
  If the source grants an amount to the taxpayer, spouse, claimant, child, or
  other role-limited person, do not aggregate over a broader household/tax-unit
  relation unless the source says every member counts. Name the relation for the
  role set that is legally counted, such as `taxpayer_or_spouse`, not merely for
  the container entity. If a copied relation is legally too broad for the
  requested source, rename it; relation names are not stable public outputs.
  Never preserve or create `*_member_of_tax_unit` or `member_of_tax_unit` for a
  source that counts only the taxpayer, spouse, qualified individual, claimant,
  child, or dependent.
- For any source that says "qualifying child", "dependent of the taxpayer", or
  "with respect to such child", do not use `member_of_tax_unit`. Define a
  role-scoped relation such as `dependent_of_tax_unit`,
  `qualifying_child_of_tax_unit`, or `child_or_dependent_of_tax_unit`, and
  aggregate over that relation.
- If a generic role-scoped relation name is already exported by a copied
  sibling file, do not reuse it. Make the relation source-specific instead of
  reusing the sibling's generic relation name.
- If the source computes an amount by reference to an entitlement, status,
  amount, or test "under" another section, subsection, paragraph, regulation, or
  document, do not inline that cross-reference's mechanics into this file unless
  that cross-referenced source text is included and this file is the canonical
  home for those mechanics. Import the existing RuleSpec target when present. If
  the cross-reference is not yet encoded, expose a semantic input/count named
  for the cross-reference itself, such as
  `additional_standard_deduction_entitlement_count_under_subsection_f`, rather
  than inventing the cross-referenced age, blindness, household, or membership
  tests locally.
- When a state, local, or downstream tax source consumes a completed federal
  return amount, such as a deduction, credit, federal adjusted gross income,
  federal taxable income, or itemized deductions already claimed on the federal
  return, keep the current source executable from a neutral federal-return
  amount input if no same-period imported RuleSpec output directly exports that
  completed return line. Name that input for the completed return amount, not
  for the legal citation pointer: for example use a name like
  `federal_return_deduction_amount` or
  `itemized_deductions_claimed_on_federal_return`, not
  `section_<section>_*`, `*_under_section_<section>`, or
  `*_allowed_under_section_<section>`. This rule applies only when the current
  source merely adds, subtracts, caps, gates, or otherwise consumes the
  completed upstream return amount; do not use it to restate upstream mechanics
  that the current source does not provide.
- When an unencoded cross-reference must be represented as a semantic local
  input, name it after the legal status with an `_under_section_<section>` or
  `_under_subsection_<subsection>` suffix. Do not start a local input with
  `section_<section>_` or `subsection_<subsection>_`; those names are reserved
  for imported legal outputs and will be treated as missing imports.
  Cross-reference local inputs such as `_under_section_<section>`,
  `_provided_in_section_<section>`, `_allowed_under_section_<section>`,
  `_deduction_under_section_<section>`, or `_credit_allowed_under_section_<section>`
  are only allowed for non-exception factual interfaces when the cited source is
  not available as RuleSpec. If the citation appears in definition,
  same-meaning, treated-as, rules-similar, exception, exclusion, `unless`,
  `subject to`, `notwithstanding`, shall-not-apply, or not-treated-as logic and
  the cited source is unavailable, do not invent a local cross-reference fact for
  the cited mechanics. If the requested source itself states the operative effect
  and only uses the citation to label a category, encode a source-named boundary
  predicate for that category instead of deferring. This includes `within the
  meaning of section ...` carve-outs and `described in section ...` category
  labels where the current source states that the category is included,
  excluded, or not treated in a specified way. Otherwise, if the dependency is
  essential to the only requested executable concept, emit
  `module.status: deferred` or `module.status: entity_not_supported` with
  `rules: []`. In a mixed provision, omit or defer only the affected executable
  surface and still encode independent source-backed outputs that do not require
  the unavailable dependency. For each omitted/deferred executable output in a
  mixed provision, add `module.deferred_outputs[]` with absolute RuleSpec
  targets for `output`, a plain-language `reason`, and `source_values` entries
  for any source-stated local parameters retained only for that deferred output.
  If those scalar legal values are independently encoded as `kind: parameter`
  rules in the same file, do not also demote them to prose or rely on
  `source_values` instead.
  Non-operative legislative findings, preambles, intent clauses, and purpose
  clauses still count for source coverage. Do not turn them into executable
  formulas; add `module.deferred_outputs[]` for the specific subparagraph with
  a reason explaining that the text is non-operative legislative intent,
  findings, or purpose.
  Do not emit `kind: parameter`, `dtype: Boolean` for date-versioned
  applicability flags. If a source-stated date flag is needed inside formulas,
  encode it as a numeric 1/0 indicator parameter and compare it explicitly
  (for example, `if transition_rule_indicator > 0: ...`), or fold the date
  effect into the rate, amount, or threshold parameter that uses it.
  Do not emit top-level `values` lookup tables outside `versions`.
  Source-backed indexed parameter tables may use `indexed_by` with
  `versions[].values`; small executable outputs that are not reusable source
  tables should use explicit derived formulas such as
  `if band == 10: 0.10 else: ...`.
  Treat category membership phrases such as `person described in section X`,
  `organization described in section X`, or `service described in section X` as
  factual boundary predicates when the current source states the legal effect.
  Use a source-named predicate like `organization_described_in_section_509_a_3`
  plus any conditions stated in the current source. Do not defer merely because
  the cited section is unavailable unless the current source requires computing
  numeric amounts or legal mechanics from that section rather than testing
  membership in the described category.
  If a source says a rate, percentage, amount, applicable percentage, or
  similar numeric term is determined under, in effect under, or equal to rates
  from another section or subsection, do not model that numeric term as a local
  input such as `tier_1_applicable_percentage`. Import the upstream output when
  it exists; otherwise defer the affected output and name the cited legal
  dependency in `reason`.
  Before applying any imported rate to the current source's whole base, check
  whether the cited source makes that rate thresholded, capped, base-limited, or
  part of an amount formula that applies only above or below a specified amount.
  If so, do not flatten the cited mechanics into `current_base * imported_rate`
  or into a combined percentage that is later multiplied by the whole current
  base. Import and compose the cited executable amount or the cited base,
  threshold, cap, and excess-amount outputs faithfully. If the current schema
  cannot pass the correct adjusted base into those cited mechanics, defer the
  affected executable output and name the missing cited computation rather than
  approximating it with a flat rate.
  If the source has a cross-reference such as `For application of different
  contribution bases ... see section X`, do not emit executable tax, amount, or
  rate-times-compensation formulas for the referenced subsections on the raw
  wage, compensation, remuneration, or payment base. Import and compose the
  cited base/cap/exclusion/excess outputs; if the cited base mechanics are
  missing, purpose-specific, or deferred, add `module.deferred_outputs[]` for
  each affected source subsection output instead of preserving a raw-base
  formula.
  If the current source instead states a purpose-limited replacement rate,
  percentage, or amount for a cited section using phrases like `computed at`,
  `in lieu of`, or `instead of the rate provided by`, encode that source-stated
  replacement value as a purpose-scoped output named for the current source's
  purpose, not as `section_<cited>_*` or `subsection_<cited>_*`. Do not
  reconstruct the cited section's amount locally by dividing an imported amount
  by its rate and multiplying by the replacement rate; downstream consumer
  modules should import the purpose-scoped replacement value when they need it.
  When the omitted output covers a specific subsection or subparagraph, the
  `output` target path must include that source path segment, e.g.
  `us:statutes/26/3201/a#tier_1_employee_tax`, not
  `us:statutes/26/3201#tier_1_employee_tax`.
  Only include `blocked_by` entries when you know the exact RuleSpec output with
  a `#rule_fragment`. Do not list bare legal provisions, corpus paths, statute
  sections, or guessed pseudo-targets in `blocked_by`; for example,
  `us:statutes/us-ca/17000` is invalid. If the exact upstream RuleSpec output is
  unknown, omit `blocked_by` and name the legal dependency in `reason`. Do not create
  tests for deferred outputs. If a source-grounded overriding rule makes the
  unavailable branch zero or unreachable for the encoded effective period,
  encode that overriding branch instead of deferring the whole module. If that
  section is present in repo context, import it and use its exported output
  instead.
- An import only resolves a cross-reference when the imported file exports the
  actual referenced legal concept, amount, status, test, or definition needed by
  the formula. Importing an adjacent upstream output only as proof, while the
  formula still depends on a local `_under_section_...` or
  `_in_effect_under_section_...` fact, does not satisfy the dependency. If the
  listed context file for a cited source does not export the needed concept, do
  not import an unrelated output from that file as a stand-in; encode the proper
  upstream source slice first, split the unresolved branch, or emit a deferred
  status when the requested file cannot compute faithfully without it.
- If the source says the current deduction, tax, rate, or amount applies "in
  lieu of" another section's amount, or parenthetically says another section is
  "relating to" that displaced amount, do not import the displaced section
  unless the current formula actually uses an exported value from it. The
  replacement formula must be based on the current source's stated base, rate,
  and conditions; a top-level import used only to acknowledge the displaced
  section is an unused/proof-only import and is invalid.
- When the requested source imposes a rate, tax, deduction, credit, cap, or
  threshold on a legal term that is defined by an available upstream RuleSpec
  file, import that upstream definition and use it in the formula. Do not leave
  a same-named local input such as `x` merely because a copied target file used
  `#input.x`. If the upstream definition has purpose-specific exports, select
  the export matching the requested source's clause.
- Every proof import must correspond to a symbol actually used by that rule's
  formula. Do not add an import atom merely because the source text mentions an
  exception or cross-reference that the formula excludes, subtracts around, or
  otherwise handles without the imported output.
- A cited context file with `module.status: entity_not_supported`,
  `module.status: deferred`, or `rules: []` is not an executable dependency.
  Do not preserve, rename, or recreate a local cross-reference input for that
  cited source. If the current provision cannot compute faithfully without that
  cited source, defer the affected executable surface; if a source-grounded
  overriding rule makes the cited branch unreachable for the encoded effective
  period, encode only that overriding branch and leave the unresolved branch out
  of executable formulas.
- Never introduce an import cycle. If a cited source directly or transitively
  imports the current target module, do not import that source back into the
  same module; keep a source-named boundary predicate or numeric boundary input
  for that cyclic condition until the sources are split into acyclic subsection
  modules. This applies to cross-referenced rates or parameters as well as
  eligibility predicates: if importing a rate-bearing source would complete a
  cycle with a foundational base definition, keep the rate as a source-named
  boundary input and continue encoding the non-cyclic base formula.
- Never create a derived rule whose formula references that same rule's name.
  The derived rule name must be the legal conclusion or compliance output, while
  required facts inside the formula must use distinct source-named local inputs.
  For example, do not define `x_has_bona_fide_need` as
  `x_has_bona_fide_need and other_conditions`; instead name the derived output
  `x_arrangement_valid` and reference a separate factual input such as
  `bona_fide_need_for_x_arrangement`.
- When the requested source defines a base, net amount, includable amount, wage
  base, income base, deduction base, or similar amount that a tax, contribution,
  credit, or deduction section will consume, do not import that consumer section
  only to use its rate parameters. If the requested source merely cites the
  consumer section's rates for an adjustment to the base, keep that rate or rate
  sum as a source-named numeric boundary input unless a non-cyclic standalone
  rate table/source is available.
- When a copied context file encodes a cited upstream source on a different
  entity, import that upstream output and bridge entities with a structural
  relation instead of replacing the import with a local cross-reference amount.
  Do not replace a specific upstream output with a broad local input for all
  amounts described by that upstream source.
- Use `dtype: Judgment`, not `dtype: Boolean`, for legal eligibility,
  availability, applicability, entitlement, and other holds/not-holds style
  outputs, especially when the formula contains `not`.
- Do not create derived `dtype: Boolean` helper rules with logical formulas.
  Use `dtype: Judgment` for derived legal predicates, or leave simple local
  facts as factual `#input.<fact>` keys consumed by formulas and tests.
- Use `kind: data_relation` for executable runtime predicates such as
  `member_of_household`. Put arity under `data_relation.arity`.
- Do not encode simple unary factual inputs as `kind: data_relation` rules. If
  a formula needs a local true/false fact, reference a descriptive bare fact
  name in the formula, declare that fact in the module's top-level `inputs`
  list with its `entity`, `dtype`, and `period`, and put that fact in tests as
  `<jurisdiction>:<repo-path>#input.<fact>`.
- If an upstream output is already executable, do not replace it with a local
  placeholder fact or compatibility alias.
- If the current source computes an amount by reference to an ancestor section
  but expressly excludes one child paragraph, subparagraph, or clause, and
  copied RuleSpec context includes child files for the included branches, import
  the included child outputs and compose them. Do not import the ancestor
  aggregate output when that aggregate may include the excluded branch or stale
  mechanics from sibling branches.
- If the requested source text includes a limitation, cap, exception, or
  cross-referenced subparagraph that changes the final exported amount, the
  final exported amount must apply that limitation. If a copied sibling/context
  file already encodes the limitation, import it and compose with it instead of
  duplicating or ignoring it.
- When an exception or carve-out applies only to a source-stated category of an
  otherwise qualifying payment, person, household, expense, or amount, gate that
  exception with a predicate for the excepted category or use an input amount
  whose name is explicitly scoped to that category. Do not subtract, disallow,
  or reduce all qualifying branches merely because an exception amount input is
  nonzero.
- If the requested source itself enumerates qualifying or exception categories
  and cites other laws only to define those category labels, encode each
  source-stated category as its own boundary predicate and combine them into the
  final rule. Do not defer the final exported output solely because cited title,
  chapter, schedule, appointment, office, retirement-system, election,
  covered-service, section-described supporting organization,
  treated-as-trade-or-business, unrelated-trade-or-business, or other
  within-meaning/described-in definitions are not encoded when the requested
  source states the operative effect.
- If the copied target file is already executable, do not let its old surface
  force local placeholders or compatibility names. Rebuild, drop, or defer
  individual outputs as needed. Prefer retaining or replacing source-backed
  independent outputs that can be encoded without unresolved dependencies; use
  top-level `module.status: deferred` or `module.status: entity_not_supported`
  only when no executable rule in the requested source can be represented
  faithfully.
- If the requested source itself defines a legal status or test through
  relationship, age, abode/residence, support, filing, income, or tie-breaker
  conditions, encode those conditions as executable predicates with boundary
  inputs for facts not defined in the source. Do not emit
  `module.status: deferred` merely because some facts must be supplied by the
  caller or because tie-breaker facts require relation inputs.
- If the requested source defines an exclusion, inclusion, deduction, or credit
  amount but depends on externally determined classifications, official
  designations, statuses, event facts, or source-document categories, encode
  the amount with boundary inputs for those facts instead of deferring.
- If the source says an actor may not request something, is not entitled to
  something, or is otherwise categorically prohibited, do not create a local
  authorization escape-hatch input such as
  `*_has_source_authorized_*_entitlement` to make the positive entitlement
  hold. Encode the entitlement as a constant false Judgment or encode the
  source-stated prohibition/not-entitled rule directly.
- This includes exclusions conditioned on a reasonable belief that an item can
  be excluded from income under another section. Do not defer solely because
  the cited exclusion section is not encoded; model the source-stated
  reasonable-belief condition as a local factual predicate and gate the
  source-stated excluded amount with it.
- If the requested source itself states a cap, threshold, exclusion, or base
  formula that uses an externally determined official base, wage amount,
  compensation amount, rate, status, or special-case fact that is not available
  as copied RuleSpec context, keep the source-stated formula executable with
  semantic local boundary inputs named for those legal values or facts instead
  of deferring the whole output.
- If a missing special rule or unavailable cited definition affects only one
  subtype, carve-out, or branch, defer only that branch or expose a
  source-named boundary input for that branch. Do not defer an unrelated
  source-stated cap/base computation that can be executed from the source text.
- When an otherwise executable output composes an imported child or sibling
  result, check that the imported file does not defer another branch, period,
  or purpose that can also affect the same final amount. Do not treat a missing
  deferred child branch as zero by importing only the available branch result.
  Either scope the executable output to the branch where the deferred child
  branch is impossible, or defer the composite output and list the child
  deferred dependency.
- Do not create parallel statutory-dollar executable parameters when a copied
  current-year authority already provides the applicable inflation-adjusted
  parameter. Import the current-year authority unless the task is to encode the
  inflation adjustment formula itself.
- If a copied current-year authority exports the same concept or output name
  that the requested statute formula would otherwise create, do not emit a
  local executable duplicate with that name. Import and use the current-year
  authority's output, keeping only statute-specific conditions or non-executable
  `source_relation` records in the statute file.
- If a current-year authority provides a directly rounded final amount table,
  use that table for the final amount instead of recomputing the amount from
  related rates and thresholds.
- When the statute states pre-inflation base dollars that a current-year
  authority adjusts, any local statute output must be named as a statutory/base
  concept, not as the current-year value.
- When the source rounds an inflation or cost-of-living increase, round the
  increase before adding it to the base amount unless the source explicitly
  says to round the final total. Companion tests must assert the rounded
  increase plus the base, not the unrounded total. For example, with base
  15750, adjustment 0.1, and a next-lower $50 multiple, the increase is 1550
  and the total is 17300, not 17325.
- Use `kind: source_relation` for non-executable legal/provenance edges such as
  `restates`, `sets`, `amends`, `implements`, `delegates`, `defines`, or
  `cites`. It must include `source_relation.type` and
  `source_relation.target`, and it must not include executable `versions`.
- If a source incorporates another authority by reference, encode that
  provenance edge as `source_relation.type: cites`; never invent an
  `incorporates` source relation type.
- Do not put source graph relationships in executable rule metadata. If a
  source `sets`, `amends`, `implements`, or `restates` another source, encode a
  separate `kind: source_relation` record in the same RuleSpec file.
- If supplied context for this target or a same-program sibling contains a
  `kind: source_relation` record, preserve that legal/provenance edge unless
  the source text proves it wrong; executable formula changes are not a reason
  to drop source graph context.
- For state-set standards, allowances, thresholds, or options implementing
  federal delegation, include `source_relation.value` pointing to the local
  executable RuleSpec output and `source_relation.basis.delegation` when context
  identifies the upstream delegated slot.
- Federal provisions that authorize state agencies to set a value create the
  delegated slot; encode those source graph records with
  `source_relation.type: delegates`. Reserve `source_relation.type: sets` or
  `implements` for the state or implementing authority that fills that slot,
  and always include `source_relation.basis.delegation` for `sets` or
  `implements`.
"""

_TESTS_PROTOCOL = """- Emit only RuleSpec YAML; use `.test.yaml` companions when tests are requested.
- Top-level `imports:` entries must be scalar strings, never map entries like
  `- target:` plus `symbols:`. Import a copied export as one exact string such
  as `<jurisdiction>:<repo-path>#<exported_symbol>`.
- In `.test.yaml` companions, every `input:` and `output:` key must be a
  canonical legal RuleSpec reference that resolves to an actual file and
  fragment. Use `<jurisdiction>:<repo-path>#input.<fact>` for fact inputs
  consumed by that file, `<jurisdiction>:<repo-path>#relation.<name>` for
  relation inputs, and `<jurisdiction>:<repo-path>#<rule_or_parameter>` for
  executable outputs or imported legal values. Never use bare friendly keys.
- Every local executable `kind: derived` or `kind: derived_relation` rule must
  appear at least once under an `output:` block in the companion `.test.yaml`;
  do not leave helper derived rules unasserted.
- In modules with executable derived outputs, do not assert raw
  `kind: parameter` rules directly in companion test `output:` blocks; cover
  parameters through derived outputs that consume them. If a module contains
  only parameters, emit one source-period snapshot case that asserts every
  local parameter output directly.
- Never emit a concrete test case with `output: {}` or an empty `output` map.
  If no executable output can be asserted, leave the test file empty instead of
  adding placeholder cases.
- Each `.test.yaml` case may assert derived outputs for only one entity type. If
  a module defines outputs on multiple entities, create separate cases for each
  entity pair, such as `Person`/`TaxUnit`, `Person`/`Employer`, or
  `Employer`/`Payment`. For example:
  `Person` cases set person facts at the top level and assert person outputs;
  `TaxUnit` cases use relation rows to supply person facts and assert only
  tax-unit outputs. Do not assert relation-child outputs in the parent entity's
  case.
- A `#relation.<name>` input value must be a YAML list of row mappings. Never
  use a scalar row such as `- true`. For example:
  `<jurisdiction>:<repo-path>#relation.member_of_household:`
  followed by `- <jurisdiction>:<repo-path>#input.member_has_required_status: true`.
  Bad:
  `<jurisdiction>:<repo-path>#relation.member_of_household: [- true]`
  Good:
  `<jurisdiction>:<repo-path>#relation.member_of_household:`
  followed by `- <jurisdiction>:<repo-path>#input.member_has_required_status: true`.
- Put `#relation.<name>` test inputs under the test case's top-level `input:`,
  not inside `tables.<Entity>` rows. If a table-row entity output depends on a
  relation, write separate scalar cases with the row's scalar facts and relation
  list under `input:` instead of a row-ordered `tables` case.
- Never assign an imported module's computed `#rule_name` output in `input:`.
  If this file imports that rule, the compiled program computes it. To make an
  imported output true, false, or equal a value, mirror the imported file's
  companion test pattern by setting its underlying `#input.<fact>` and
  `#relation.<name>` keys.
- When a test is meant to exercise a threshold, cap, or boundary on an imported
  derived output, do not assume one upstream raw input equals that imported
  output. First compute the imported formula from the upstream inputs you set;
  if the upstream formula has deductions, rates, or offsets that make the exact
  boundary awkward, choose clearly below/above-boundary inputs instead of an
  exact boundary case.
- Never turn an imported derived rule into a fabricated `#input.<same_rule_name>`
  key. For example, use
  `<jurisdiction>:<repo-path>#imported_judgment: holds` or `not_holds`, not
  `<jurisdiction>:<repo-path>#input.imported_judgment`.
- Do not invent `#input` keys for imported files. Use only the bare fact names
  that the imported file's formulas actually reference, or mirror the imported
  file's companion `.test.yaml` input pattern when it is supplied in context. If
  that imported output is driven by an upstream structural relation, set the
  upstream `#relation.<name>` rows used by the companion test instead of
  creating a local input under the imported file.
- Use `holds` and `not_holds` for actual `dtype: Judgment` rule keys in test
  inputs and outputs; do not use YAML booleans for Judgment rule values.
- Use YAML booleans `true` and `false` for local factual `#input.<fact>` keys
  referenced directly by formulas.
- Compute each expected `output:` value by evaluating the emitted RuleSpec
  formula against the case inputs step by step. Do not guess expected outputs
  from nearby statutory thresholds or caps. For formulas that combine a flat
  threshold with a percentage of excess income, include the threshold amount,
  the excess amount, and the percentage amount in the calculation reflected by
  the scalar expected output.
- For positive tests that expect a nonzero amount, `holds` Judgment, or other
  affirmative result from a formula with source-stated age, income, resource,
  duration, date, status, or other threshold gates, set every gate input on the
  qualifying side of the threshold. For example, if the formula requires
  `age >= age_threshold`, a case expecting the positive amount must set `age`
  at least to `age_threshold`; use a separate negative case for below-threshold
  inputs.
- In mixed-output test cases, do not assert an output's affirmative or nonzero
  result when any input in that same case intentionally falls on the
  nonqualifying side of that output's threshold gate. Split the case instead:
  one case for the blocked output and a separate all-gates-positive case for
  unrelated affirmative outputs.
- For proration, average, ratio, or percentage tests with a source-stated
  denominator, choose input amounts divisible by that denominator so expected
  outputs are exact decimals, not rounded approximations. For example, if the
  denominator is 365, use a base amount like 36500 so
  `36500 * 182 / 365 = 18200`; if an average divides by 6, use totals like 600
  or 1800, not 700. Avoid exact equality boundaries for ratios or percentages;
  choose clearly below/above-boundary values so decimal precision cannot decide
  the test outcome.
- Every test case for a local derived formula must assign every local factual
  `#input.<fact>` referenced by that formula, including facts that are false in
  the case. Missing false inputs make the executable test invalid.
- For every encoded `except`, `unless`, `subject to`, or `notwithstanding`
  carve-out, include companion tests for the positive path and the carve-out
  path so exclusions and override conditions cannot be silently dropped.
- When a complete source unit has many such controls, emit as many compact,
  single-principal-output case pairs as the source requires. Keep each pair to
  the inputs reached by that output and vary only its controlling selector;
  do not replace required pairs with one large omnibus case.
- When a source says a subsection, paragraph, payment, credit, benefit,
  eligibility path, or other output "shall not apply" or "does not apply",
  the exported rule that says that target applies, is allowed, is included, or
  is eligible must negate the exception. Do not expose the exception only as a
  standalone helper while leaving the affected `*_applies`, eligibility,
  inclusion, exclusion, or amount output true under the exception.
- When that exception is encoded as a local derived helper, include a blocking
  companion test asserting both that helper as `holds` and the directly affected
  Judgment output as `not_holds`.
- For scoped exceptions, include a control case proving a non-excepted
  qualifying item is not reduced or blocked even when the exception amount or
  exception fact is positive/nonzero, plus a case where the same exception
  applies to the source-stated excepted category.
- Preserve anaphoric scope in source predicates. If the source says "such
  account", "such instrument", "through such account", "with respect to such
  payment", or similar same-object language, the predicate name and companion
  tests must keep that same-object relationship. Do not shorten it to broad
  activity for the person, broker, household, or entity generally.
- When a local formula has five or fewer independent source-stated boolean
  gates joined by `and`, include one all-gates-positive case and enough negative
  cases to toggle each gate at least once. Do not leave a source-stated gate
  untested just because another negative case toggles a different gate.
- Build those boolean-gate witnesses mechanically. First emit a minimal
  all-gates-positive case whose `input:` contains exactly the local facts
  reached by that one asserted principal output. Its `output:` must assert the
  principal output plus every reached local derived dependency required for
  corroboration, but no unrelated output. Clone the complete case once per
  gate, changing exactly one input value and the expected principal output plus
  any asserted reached dependency whose value also changes. Every member of the
  pair must have identical input-key and output-key sets; put unrelated helper,
  amount, and downstream-output demonstrations in separate cases.
- If a formula negates multiple exception predicates, include a separate
  companion test for each predicate that sets that exception input true and
  expects the directly affected Judgment rule to be `not_holds`.
- For any negated exception predicate, include a paired positive case with the
  same output rule where only the exception input changes from `false` to
  `true`; do not combine the exception test with another branch change.
- Validation fails if a direct local `#input.*_exception_applies` or
  `#input.*_exception_*` predicate is negated by an exported Judgment rule
  without this paired positive/negative companion.
- Do not collapse a list of cited exceptions or cross-reference carve-outs into
  one aggregate fact such as `sections_..._do_not_preclude...`. Encode or
  import each cited exception separately, then combine them in a helper if
  useful.
- When an exception, exclusion, `subject to`, or `unless` clause cites another
  legal section or same-section subsection, do not create a local
  `section_...` or `subsection_...` placeholder input for that cited source.
  Import the cited RuleSpec source when it exists; if that upstream source is
  required but unavailable, stop with a missing-upstream/dependency request
  rather than encoding an opaque local fact.
- For opening scope phrases such as `except as provided in clause (ii)` that
  or `subject to paragraph (c)` that point to a sibling clause outside the
  requested target and no copied context supplies that sibling's executable
  output, do not invent a local boolean like `clause_ii_provides_otherwise`.
  Keep the current target scoped to the source-stated positive calculation, or
  defer only the final affected surface if the sibling exception is essential to
  the requested output.
- A pure `notwithstanding subsection ...` override does not require importing
  the overridden subsection unless the formula actually needs that cited
  subsection's computed output.
- If the cited same-section subsection or sibling paragraph is supplied in
  context as a RuleSpec file, do not summarize it into a local fact like
  `person_meets_...requirements`. For operative `except`, `unless`, or
  `subject to` carve-outs that can change the requested output, a bare
  file-level import is not enough: import the exact `#rule_name` exported by
  the cited file and reference that bare symbol in the affected formula
  (usually negated or used as a branch guard). Validation rejects file-level
  imports for operative sibling carve-outs when the formula never uses a cited
  output.
- When the requested source is itself a sibling paragraph, subsection, or
  clause that copied context cites as an operative `subject to`, `except`, or
  `unless` condition, export a source-grounded umbrella `Judgment` for the
  whole cited source when that source contains multiple branches or alternatives.
  Name it for the cited source's legal condition or result, such as
  `paragraph_d_2_methodology_limit_satisfied`, and let downstream files import
  that exact `#rule_name`. Branch helper outputs may also be useful, but they
  do not replace the umbrella output when the sibling citation points to the
  whole paragraph. Do not import the downstream consuming paragraph merely to
  build this umbrella; represent source-stated categories as local boundary
  inputs, branch helpers, or imports from true upstream definitions so the
  dependency graph remains acyclic.
- If the cited sibling file is deferred, empty, unsupported, or missing a
  usable exported rule and the carve-out changes the result, defer the affected
  executable output or encode a source-grounded overriding branch that avoids
  the dependency. Do not emit a formula that ignores the carve-out, and do not
  invent a local boolean for the cited sibling source.
- File-level imports without a `#symbol` fragment are acceptable only for
  non-operative provenance or boundary context, such as a pure
  `notwithstanding` override or a local source-stated override where the
  formula does not depend on the cited output. They are not acceptable for
  `except`, `unless`, or `subject to` formula carve-outs.
- Example: if the requested source says `Subject to paragraph (c)` and copied
  context contains `regulations/.../c.yaml` exporting
  `cash_assistance_less_restrictive_methodologies_may_be_applied`, import
  `us:regulations/.../c#cash_assistance_less_restrictive_methodologies_may_be_applied`
  and include `cash_assistance_less_restrictive_methodologies_may_be_applied`
  in the affected formula and proof atoms. A formula that repeats only the
  positive paragraph requirements while omitting the cited paragraph's symbol
  is invalid.
- Do not copy the body of a cited cross-reference provision into this module's
  `summary` or re-encode that cited provision locally. Keep this module scoped
  to the requested citation and import the cited provision instead.
- If context files import the target file or reference target outputs, use that
  as a signal to repair the dependency graph, not as a requirement to preserve
  old names. Keep an old output only when it remains the cleanest
  source-faithful RuleSpec surface.
- When a copied context file already exports the operative legal condition that
  the requested source consumes, import and use that canonical output. Do not
  recreate it as a local factual input merely because the requested source
  describes a person or household as entitled, eligible, qualified, allowed,
  or subject to that condition. Keep a local fact only when the requested
  source states a distinct operative fact that the context output does not
  represent.
- Do not preserve existing factual input slots referenced by copied formulas or
  companion tests when a cleaner source-faithful encoding removes them. This is
  especially important for names listed under invalid copied local inputs.
- For cross-reference boundary facts that remain local because the cited source
  is not present in context at all, keep the legal pointer in the identifier.
  If context for the cited source is present but unsupported, deferred, empty,
  or missing the needed export, do not preserve, rename, or recreate the local
  cross-reference fact; import a real export, defer the affected executable
  surface, or encode a source-grounded overriding branch that avoids it.
- When the requested source states its own amount, cap, threshold, or formula
  but begins with a cross-reference exception such as `except as otherwise
  provided in section X`, `except as otherwise provided in subsection X`, or
  `subject to paragraph (c)`, this local-boundary escape hatch applies only to
  cited external or parent sources. It does not apply to uncopied sibling
  clauses; for sibling clause exception phrases, do not invent local `clause_*`
  booleans.
- Do not emit Python code, markdown fences, prose, or file-write confirmations.
- Do not invent values or ontology beyond the source text.
- When source text uses amendment markup like `[old] new`, treat the bracketed
  value as superseded text. Encode the current unbracketed value/effective date
  unless the task explicitly asks for historical text.
- If a source makes an allowance, deduction, exemption, or eligibility branch
  conditional on billed, paid, incurred, anticipated, or other cost/expense
  facts, encode a positive fact predicate for that source-stated condition.
  Do not model availability solely as `not` other categories. If the condition
  lives in a parent paragraph needed to understand a child paragraph, it must be
  part of the resolver-supplied canonical source unit or a separately attested
  RuleSpec import; do not add another corpus path to this module.
- When the cost/expense fact only matters after exclusion predicates, exported
  amount/quantity formulas consumed by dependent modules must guard the
  exclusions before referencing the branch-specific fact, so excluded cases do
  not require that fact as an input. For example, the amount should use
  `if other_allowance_eligible: 0 else: if household_has_telephone_cost: amount else: 0`
  rather than `if telephone_eligible: amount else: 0` when `telephone_eligible`
  itself references the branch-specific telephone-cost input.
- Phrases like `consists of the cost for X` or `available to households with X
  costs` require a positive fact for that cost/service. For example, a telephone
  allowance must depend on a fact for the household having or incurring the
  basic telephone-service cost before applying exclusions for other allowances.
- In a jurisdiction-specific repo, phrases that merely identify the target
  jurisdiction usually describe the document's scope, not a new input variable.
  Do not add a state-residency input unless the provision itself is encoding a
  residency eligibility test.
"""

_FORMULA_PROTOCOL = """- Put formulas under `versions: - effective_from: 'YYYY-MM-DD'` and `formula: |-`.
- Do not encode legal effective dates as `dtype: String` parameters or date
  literal formulas such as `2025-01-01`. Axiom formulas have no date literal type.
  Use `effective_from` metadata for version timing, or use a
  source-stated semantic boolean predicate when a date window is a runtime
  condition. Do not put the date or year value in the fact name; use names like
  `taxable_year_begins_after_termination_date` or
  `taxable_year_is_in_temporary_effective_window`, not
  `taxable_year_begins_after_2024_and_before_2029` or
  `taxable_year_begins_after_december_31_2021`.
  Never use `post_YYYY`, `pre_YYYY`, `after_YYYY`, `before_YYYY`, or any
  four-digit year in a runtime date-window fact name.
  This overrides preservation of existing local input names: if a copied
  formula uses a date-valued fact name, rename that fact consistently to a
  semantic date-window predicate in formulas and tests.
- Do not emit more than one `versions:` entry for `kind: derived`; the runtime
  does not yet support period-selecting versioned formulas. Use a single
  source-faithful conditional formula when the provision itself defines a
  temporal branch, or encode only the currently applicable provision after
  resolving the source context.
  When a derived result changes only because a base rate, threshold, cap, or
  additive adjustment changes over time, put the dated changes on named
  `parameter` or helper rules and keep the consuming `kind: derived` rule to
  one formula, such as `base_rate + temporary_adjustment + later_adjustment`.
- Formula strings use Axiom formula syntax: `if condition: value else: other`, `==`, `and`, and `or`.
  Do not write `else if` or `elif`; chain branches as
  `if condition: value else: if next_condition: next_value else: fallback`.
- Supported scalar functions are `min(...)`, `max(...)`, `floor(x)`, and
  `ceil(x)`. Do not use Python-only functions such as `round(...)`; express
  nearest-multiple rounding as `floor((x / multiple) + 0.5) * multiple` for
  nonnegative amounts.
- Benefit, allotment, credit, deduction, allowance, and subsidy formulas must
  never emit negative money. When subtracting an income, contribution, or other
  reduction from a maximum amount, floor the result with `max(0, ...)` before
  applying downstream minimum-benefit or issuance branches.
- When a nonnegative credit, deduction, allowance, subsidy, or benefit is a
  percentage of `min(income, cap)` or similar, floor the income base at zero:
  use `rate * min(max(0, earned_income), cap)`, not
  `rate * min(earned_income, cap)`.
- Outputs named `taxable_income` or ending in `_taxable_income` must also never
  be negative. Wrap the final selected branch at zero, including both sides of
  conditionals: use `if condition: max(0, branch_a) else: max(0, branch_b)`,
  not `if condition: branch_a else: branch_b`.
- Taxpayer elections such as electing to itemize deductions are legitimate
  election-state inputs to the legal computation. Do not mark a taxable-income
  rule unsupported merely because the taxpayer could optimize that election
  outside the core RuleSpec runtime; encode the statutory branches keyed by the
  election fact, and let oracle/comparison harnesses run multiple scenarios when
  they need optimization.
- If that reduction has rounding alternatives, every branch must be floored:
  use `if round_up: max(0, maximum - ceil(reduction)) else: max(0, floor(maximum - reduction))`,
  never `if round_up: maximum - ceil(reduction) else: floor(maximum - reduction)`.
- Supported relation aggregators are `len(relation)`,
  `count_where(relation, predicate_fact)`, `sum(relation.amount_fact)`, and
  `sum_where(relation, amount_fact_or_derived, predicate_fact)`. Do not write
  `sum(relation, expression)` or put arithmetic inside a relation field access.
  Use `sum(relation.amount_fact)` only when `amount_fact` is a raw scalar fact
  supplied directly on each relation row. Do not use `sum(relation.local_output)`
  for a `parameter` or `derived` rule defined in the same file; for a computed
  per-related-entity amount, write
  `sum_where(relation, local_output, source_stated_predicate_fact)` instead.
  To count two boolean conditions over the same relation, write two
  `count_where(...)` calls and add them.
- If a conditional is embedded inside arithmetic or another larger expression,
  wrap the whole conditional in parentheses, such as
  `amount + (if condition: extra else: 0)`. Do not write
  `amount + if condition: extra else: 0`.
- When source text says an amount "shall not include" or excludes "the part in
  excess of" a cap, the included amount is capped at that limit:
  `min(source_amount, cap)`. The excluded excess is
  `max(0, source_amount - cap)`, but do not return `source_amount - cap` or
  `source_amount - remaining_cap` as the included amount.
- When a source says an already-computed deduction, allowance, credit, benefit,
  or other amount "cannot exceed", is "limited to", or is capped by "the higher
  of", "the greater of", "the lesser of", or "the lower of" several cap
  amounts, compute the uncapped base amount separately and apply the selected
  cap only at the final capped output, e.g. `min(uncapped_amount, max(cap_a,
  cap_b))`. Do not replace the uncapped base with one cap alternative such as
  `if condition: cap_b else: uncapped_amount` unless the source expressly says
  the alternative amount is used "instead of", "in lieu of", or as a
  substitution for the base computation.
- Formula strings must use bare identifiers only. If an imported rule is listed
  as `us:statutes/...#example_rule`, add that exact target to `imports:` but
  reference `example_rule` inside formula text.
- Axiom conditionals are expression syntax, not YAML syntax. Money/scalar
  formulas may use `if condition: value else: other`; do not use Python ternary
  syntax, `then:`, `else if`, or `elif`.
- Function calls in formulas are expression syntax, not Python syntax. Do not
  include trailing commas in calls such as `min(a, b)` or `max(0, x)`, and do
  not write tuple-style expressions.
- `dtype: Judgment` formulas must not use `if ... else ...`. Write them as
  boolean expressions using `and`, `or`, `not`, comparisons, and parentheses.
  For example, encode `if exempt: net_ok else: net_ok and gross_ok` as
  `net_ok and (exempt or gross_ok)`.
  If a Judgment compares against a scalar that varies by condition, do not embed
  `(if condition: extra else: 0)` inside the comparison. Either create a scalar
  helper for the allowed amount or branch the Judgment directly, for example
  `(condition and days <= base + extra) or (not condition and days <= base)`.
- When using negated conjuncts, write them as a multiline formula with each
  `not <predicate>` term on its own line joined by `and`, rather than one
  compact `not A and not B` line.
- Formula strings reference indexed parameter tables with `table_name[index_expr]`,
  never as a bare `table_name` scalar.
- Every substantive numeric literal must be grounded in the supplied source text unless it is -1, 0, 1, 2, or 3.
- If you encode a substantive numeric literal, `module.summary` or the rule's proof excerpt
  must include the exact source phrase containing that number. Do not omit a
  subsection, table row, or clause that grounds an encoded
  numeric amount, rate, threshold, cap, or limit.
"""

US_TAX_PACK = """- US tax filing status is a derived legal classification, not a downstream
  boundary fact. Do not create local `#input.filing_status` facts in a rule or
  test. Encode the upstream filing-status source first, then import its absolute
  RuleSpec output into downstream threshold, phaseout, deduction, and credit
  rules. If an already-encoded upstream filing-status output is unavailable,
  stop and encode that upstream source rather than synthesizing a local input.
  Do not preserve existing `#input.filing_status` or `#input.tax_filing_status`
  surfaces from copied target files; migrate them to upstream imports or
  source-backed non-status leaf facts such as whether a joint or separate return
  was actually made.
- The shared US tax filing-status output remains a structural enum: 0 single,
  1 joint return, 2 married filing separately, 3 head of household, and
  4 surviving spouse / qualifying widow(er). If the source groups surviving
  spouse with joint return, every filing-status branch or match that handles
  status 1 must also handle status 4 in that same branch.
  If the source says only "joint return" without also naming surviving spouse
  or qualifying widow(er), do not route status 4 to the joint-return branch;
  status 4 falls under any "other case" branch unless the source states
  otherwise.
- Never encode US tax filing status as string literals such as
  `"married_filing_jointly"` or as unbacked local boolean facts such as
  `married_filing_jointly`, `head_of_household`, or `surviving_spouse`. If a
  source provision itself defines a legal status or return category, encode that
  source-backed output at its absolute RuleSpec path and import it downstream.
  If a shared numeric filing-status output is available, import it and use the
  structural enum in formulas when the source supports that grouping, e.g.
  `match filing_status: 1 => joint_amount; 4 => joint_amount; ...`.
- Do not replace filing-status components with local status inputs such as
  `taxpayer_is_surviving_spouse`, `surviving_spouse`, or `head_of_household`.
  This also prohibits compound status predicates such as
  `individual_is_not_married_and_is_not_surviving_spouse`.
  Those are derived legal classifications; import their source-backed RuleSpec
  outputs or defer the affected output until those upstream definitions exist.
"""

_NUMERIC_GROUNDING = """- If the source states a substitution, higher amount, increase, cap, or other
  modifier amount, do not define the modifier as an unused scalar while
  computing the affected numeric output without it. Use the modifier in the
  affected formula, or defer that affected output until the upstream branch
  condition can be encoded/imported. If you defer the affected output, list the
  deferred output under `module.deferred_outputs[]` and list the absolute target
  for the retained modifier parameter under that record's `source_values`. Include
  `blocked_by` only for exact upstream RuleSpec outputs with `#rule_fragment`;
  otherwise explain the unknown blocker in `reason`.
  Do not solve this by deleting the affected numeric output while leaving the
  modifier parameter stranded.
- When the source says a value or amount "in excess of" a stated limit is
  counted, included, deemed, or added, that is an executable excess formula.
  Encode the limit as a named scalar and encode the affected numeric output as
  `max(0, measured_value - limit)` or the source-stated equivalent, using a
  local factual input for `measured_value` if no source-backed upstream measure
  exists. Do not defer that excess output merely because a later aggregate
  resource, income, or liability calculation is outside the source slice; defer
  only the later aggregate if necessary.
- When the source states a final effective legal amount and also explains that
  amount as an increase by a percentage, inflation index, or cost-of-living
  adjustment, do not encode the explanatory percentage or index as a standalone
  scalar unless the source also supplies the prior base and the target formula
  uses that calculation. Encode the final effective amount as the operative
  scalar; keep the explanatory increase text in the summary or proof excerpt
  rather than as an unused modifier parameter.
- Every substantive numeric occurrence in `./source.txt` must be represented by
  a named numeric concept when it is a legal amount, rate, threshold, cap, or
  limit, including structural interval-table row labels used by a source-backed
  band selector predicate. Consumers should reference the named concept or an
  indexed table/grid value; only structural keys, sentinels, and algebraic
  identities should remain as formula literals.
- If the same numeric value appears in separate numbered exceptions,
  subparagraphs, or otherwise materially different legal roles, give those roles
  distinct named scalars; reuse a named scalar only for the same legal role.
- Adjacent bracket thresholds repeated as both an upper bound and the next
  bracket's lower bound are separate source-stated legal roles; define distinct
  semantic scalars for those occurrences and use them in the branch conditions.
"""

_SELF_CHECK = """- Before finalizing, do this self-check:
  1. Numeric inventory: every source-stated legal amount, rate, threshold, cap,
     or limit has a named local numeric concept or an exact imported concept
     from context, and derived formulas reference that local or imported name
     rather than an inline literal. For tables and grids, bounds and cells are
     indexed numeric concepts; formulas may use structural integer keys only to
     select from those concepts. If an exact same-path child scalar is available
     in context, import it instead of duplicating it locally.
  1a. Dependency inventory: no local derived rule formula references its own
      rule name. If a legal phrase is both a required fact and a desired output,
      rename the output to the conclusion and keep the required fact as a
      distinct local input.
  2. Test input inventory: for every local factual identifier referenced by a
     local derived formula, every companion test case assigns the corresponding
     `#input.<fact>` explicitly, including false facts. Do not rely on implicit
     defaults. A parameter-only module may use one source-period snapshot case
     that asserts every local parameter output directly. For other modules,
     assert derived outputs that consume parameters instead. If a local amount
     formula has a branch returning 0, include a
     companion case that asserts that local output is 0.
     For imported modules, only assign imported `#input` or `#relation` keys
     that exist in the current imported RuleSpec context. Do not preserve stale
     imported test inputs from copied files. Do not stub imported derived
     outputs as test inputs; imported derived outputs are computed. If the downstream
     rule depends on an imported output, assign all current upstream factual
     inputs and relations needed by that imported output, including false facts.
     This does not override no-input guardrails: never assign prohibited derived
     classifications such as any imported or local `#input.filing_status` or
     `#input.tax_filing_status`. This prohibition is absolute even when the
     value is a numeric enum and even when the key belongs to an imported
     module. If an imported output cannot be exercised without those prohibited
     test inputs, omit that assertion or encode the upstream filing-status
     sources first.
  3. Proof inventory: every proof atom uses only an allowed `kind`; imported
     proof atoms include `import.target`, `import.output`, and `import.hash`;
     textual support uses direct release-bound corpus source text.
  4. Import inventory: every `imports:` entry is an exact copied/importable
     RuleSpec target. Top-level `imports:` entries must be scalar strings; never
     map entries like `- target:` plus `symbols:`. Do not guess sibling paths; if
     required upstream context is missing, emit a typed missing-upstream/dependency
     request instead.
     Never drop the jurisdiction prefix from copied context imports: use
     `us:statutes/...#symbol`, not `statutes/...#symbol`.

Minimal shape:

format: rulespec/v1
module:
  proof_validation:
    required: true
  summary: |-
    <source text>
rules:
  - name: example_amount
    kind: parameter
    dtype: Money
    unit: USD
    source: <legal citation/span>
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: <corpus path>
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          451

Indexed table shape:

rules:
  - name: example_amount_by_household_size
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 298
          2: 546
  - name: example_amount
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    source: <legal citation/span>
    versions:
      - effective_from: '2025-10-01'
        formula: example_amount_by_household_size[household_size]

Derived membership shape:

rules:
  - name: snap_member_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 7 CFR 273.1(a)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          member_has_required_status
          and not member_is_excluded_student
  - name: snap_unit
    kind: derived_relation
    derived_relation:
      arity: 2
      source_relation: member_of_household
      entity: SnapUnit
      member_relation: members
      slot_entities: [Person, Household]
    source: 7 CFR 273.1(a)
    versions:
      - effective_from: '2026-01-01'
        formula: snap_member_eligible
"""

_SHAPES = """Minimal shape:

format: rulespec/v1
module:
  proof_validation:
    required: true
  summary: |-
    <source text>
rules:
  - name: example_amount
    kind: parameter
    dtype: Money
    unit: USD
    source: <legal citation/span>
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: <corpus path>
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          451

Indexed table shape:

rules:
  - name: example_amount_by_household_size
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 298
          2: 546
  - name: example_amount
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    source: <legal citation/span>
    versions:
      - effective_from: '2025-10-01'
        formula: example_amount_by_household_size[household_size]

Derived membership shape:

rules:
  - name: snap_member_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 7 CFR 273.1(a)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          member_has_required_status
          and not member_is_excluded_student
  - name: snap_unit
    kind: derived_relation
    derived_relation:
      arity: 2
      source_relation: member_of_household
      entity: SnapUnit
      member_relation: members
      slot_entities: [Person, Household]
    source: 7 CFR 273.1(a)
    versions:
      - effective_from: '2026-01-01'
        formula: snap_member_eligible

Parent-composition shape (import child results, never re-derive them):

rules:
  - name: combined_employee_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 3201 (aggregate of subsections)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/3201/a#tier_1_employee_tax
              output: tier_1_employee_tax
              hash: sha256:<listed context hash>
    versions:
      - effective_from: '2026-01-01'
        formula: tier_1_employee_tax + tier_2_employee_tax

Deferral shape (typed, target-precise, no tests for deferred outputs):

module:
  status: deferred
  deferred_outputs:
    - output: us:statutes/26/3201/a#tier_1_employee_tax
      reason: >-
        Compensation base is defined under section 3231(e), which is not yet
        encoded; the rate cannot be applied to a faithful base.
      source_values:
        - us:statutes/26/3201/a#tier_1_rate
rules: []

Companion-test shape (relation rows under input; exceptions paired):

- name: exception_blocks_allotment
  period: 2026-01
  input:
    us:statutes/7/2017/a#relation.member_of_household:
      - us:statutes/7/2017/a#input.member_has_required_status: true
    us:statutes/7/2017/a#input.net_income: 100
    us:statutes/7/2017/a#input.disqualification_exception_applies: true
  output:
    us:statutes/7/2017/a#allotment_available: not_holds
"""

_PROMPT_BLOCKS = (
    _CORE_CONTRACT,
    _REPOSITORY_CONTEXT,
    _TABLES_PROTOCOL,
    _STRUCTURE_PROTOCOL,
    SOURCE_SCOPE_PROTOCOL,
    _COMPOSITION_AND_DEFERRAL,
    _NAMING_PROTOCOL,
    _TESTS_PROTOCOL,
    _FORMULA_PROTOCOL,
    US_TAX_PACK,
    _NUMERIC_GROUNDING,
    _SELF_CHECK,
    _SHAPES,
)


def _assemble(*blocks: str) -> str:
    """Join protocol blocks with blank-line separators.

    Each block is stripped before joining so no boundary can glue two
    sentences together or stack blank lines; empty blocks are dropped.
    """
    return "\n\n".join(block.strip() for block in blocks if block.strip())


ENCODER_PROMPT = _assemble(*_PROMPT_BLOCKS)

_COMPLETE_SOURCE_UNIT_PROTOCOL = """
Complete-source-unit mode is enabled for this request:
- Treat the entire authoritative source unit as the completeness inventory.
  `module.summary` is only a concise reviewer orientation and contributes
  nothing to completeness accounting.
- Every explicit computation stated in the source unit must have a principal
  `kind: derived` or `kind: derived_relation` output. Naming its constants as
  parameters without encoding the stated formula is invalid. Do not defer an
  explicit computation merely because its source-stated facts are not already
  represented in the module; declare those facts as explicit local RuleSpec
  inputs and encode the principal output.
- When the source states both a base value and its converted result, encode both
  values as separate grounded `kind: parameter` rules. Result wording such as
  "converted to the month, this gives ..." states a scalar result; it does not
  mandate deriving one parameter from the other.
- Never introduce calendar constants `12`, `52`, `365`, `4`, or `24` as module
  literals unless that literal appears in the authoritative source text.
  Express a stated conversion through companion-test assertions on both
  parameter outputs; literals used only in companion tests do not require
  source grounding.
- Encode every structural paragraph and list branch, including Absatz markers
  such as `(1)` and `(2)`, `Abs. 5`, numbered items such as `1.` and `1a.`, and
  Satz enumerations. If a branch cannot be encoded, use a precise typed
  deferral that names the exact branch and its missing dependency or citation;
  never omit it silently. Put the structural branch in the deferred output
  path (for example, `de:statutes/estg/32a/6#surviving_spouse_tariff`). Include
  `blocked_by` only for known exact RuleSpec targets with a `#rule_fragment`;
  otherwise omit `blocked_by` and name the exact missing legal dependency or
  citation in `reason`. For a runtime-gap deferral of a current-source branch,
  the `reason` itself must literally cite the complete legal branch, including
  every subsection marker represented by the output branch, and name a
  concrete source-stated missing input or runtime capability. The output path
  is not a source citation.
  Never guess a blocker target.
- Before returning YAML, inventory every top-level structural branch in the
  authoritative source and verify that each branch has either an executable
  rule whose `source:` cites that branch or one `module.deferred_outputs` entry
  whose absolute output path preserves the branch label. Do not return while
  any branch is absent. A deferral reason must identify the exact branch and a
  concrete missing input, dependency, or runtime capability; generic omission
  language is not coverage.
- A child-branch `source:` citation does not cover a distinct parent chapeau.
  When a parent chapeau states applicability, temporal scope, or a shared
  computation, bind one consuming executable rule to that exact parent path
  and include an exact parent-chapeau proof excerpt. Keep the child bindings as
  well; do not substitute an older parallel chapeau or infer parent coverage
  from encoded descendants. Do not invent a dummy output for a chapeau that
  only scopes the child computations.
- When an authoritative branch is exactly a bare `Repealed.` tombstone, or a
  Louisiana `Repealed by Acts ...` tombstone whose remaining text is solely a
  finite session-law citation and optional effective date, preserve that branch
  path in `module.deferred_outputs[].output`. Make `reason` cite the exact
  current legal branch and affirmatively state that it is repealed. Do not
  fabricate an executable rule, input, runtime gap, or external blocker. Use
  the bounded reason form `<exact branch citation> is repealed.`, optionally
  adding the exact authenticated history either as `is repealed by <Acts
  citation>.` or as `is repealed. <Acts citation>.`, and/or adding `and supplies
  no operative rule`; never qualify, report, or retract the assertion.
- For a Louisiana branch that computes tax at rates provided in another R.S.
  section, use the bounded missing-export clause `no executable RuleSpec output
  for those <source-stated modifiers> rates is supplied in the available
  context`. Do not repeat the dependency inside that missing object or append a
  `that`/`which` relative clause.
- Companion tests must execute every source-stated formula branch, boundary,
  exception, and rounding rule with assertions on the affected principal
  output. Each branch needs distinct runtime evidence; descriptive test
  metadata is not coverage evidence.
- Every emitted derived `dtype: Judgment` output, other than a source-faithful
  Judgment whose versions are all constant false, must be asserted as `holds`
  by at least one companion case. When `not_holds` is reachable, include that
  negative companion too; testing only the negative state is incomplete.
- When a principal formula reaches a branch through local derived selectors or
  intermediates, assert every reached local derived dependency's expected
  output in that same companion case. Raw inputs alone do not corroborate a
  derived intermediate, and a local derived rule must never be shadowed under
  `input:`.
- When an implementing principal rule is not unambiguously bound by a canonical
  structural source path, add a `versions[N].formula` source proof atom using
  the exact canonical `source.corpus_citation_path` and a short verbatim
  `source.excerpt` identifying its computation. This is mandatory when multiple
  computations share one structural path. A citation-only proof atom,
  human-readable rule-level `source:`, or self-import is not an unambiguous
  formula-clause binding.
- A historical branch's runtime evidence must use that branch's legally
  applicable period. If an external oracle cannot evaluate that period, keep
  the source-faithful companion case and omit oracle inputs or expectations
  from that case; never move or omit the branch to gain oracle compatibility.
- When a principal derived output combines an earlier operative path with a
  later-added alternative, keep that output executable from the earliest
  source-stated path date. Do not move the whole output's `effective_from` to
  the later alternative; use effective-dated parameter/helper guards in the
  single derived formula so every historical companion case executes an
  applicable path.
- A genuinely scalar-only source unit may remain parameter-only.
"""


def get_encoder_prompt(
    citation: str,
    output_path: str,
    corpus_citation_path: str | None = None,
    require_complete_source_unit: bool = False,
) -> str:
    """Return a complete RuleSpec task prompt for a source unit."""
    corpus_section = ""
    if corpus_citation_path:
        corpus_section = f"""
Corpus source path: {corpus_citation_path}
Include `{corpus_citation_path}` in `module.source_verification`.
Use exactly
`module.source_verification.corpus_citation_path: {corpus_citation_path}`.
Use that exact same `{corpus_citation_path}` value in every source-backed proof
atom's `source.corpus_citation_path`. Do not rewrite it as a display citation
and do not append subsection markers or other path segments. Express narrower
support through the rule-level `source:`, the proof excerpt, and the legal
output path while keeping the corpus machine identity unchanged.
Never emit `corpus_citation_paths`. A provision split across storage rows must
be composed by the corpus resolver under this one canonical path. If another
legal source is required, import its separately attested RuleSpec or defer the
affected executable surface.
"""
    if require_complete_source_unit:
        corpus_section += f"""
{_COMPLETE_SOURCE_UNIT_PROTOCOL.strip()}
"""

    return f"""{ENCODER_PROMPT}

Target citation/source id: {citation}
Expected output path: {output_path}
{corpus_section}

Return only raw RuleSpec YAML for that path.
"""
