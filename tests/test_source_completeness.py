from __future__ import annotations

import functools
from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as completeness_module
from axiom_encode.harness import validator_pipeline as validator_pipeline_module
from axiom_encode.harness.source_completeness import (
    analyze_complete_source_unit,
    authoritative_numeric_recall_text,
    collect_artifact_numeric_bindings,
    collect_artifact_numeric_values,
    recognize_source_structure,
    source_states_explicit_computation,
)
from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline,
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

CORPUS_CITATION_PATH = "de/statute/estg/32a"
DE_NUMERIC_OCCURRENCE_EXTRACTOR = functools.partial(
    extract_typed_numeric_inventory_occurrences_from_text,
    profile="de-DE",
)


def _analyze(
    content: str,
    authoritative_source_text: str,
    *,
    test_cases: list[object] | None = None,
    artifact_numeric_values: tuple[float, ...] | None = None,
    artifact_numeric_bindings: tuple[tuple[str, float], ...] | None = None,
):
    return analyze_complete_source_unit(
        content,
        authoritative_source_text,
        corpus_citation_path=CORPUS_CITATION_PATH,
        test_cases=test_cases,
        extract_numeric_occurrences=DE_NUMERIC_OCCURRENCE_EXTRACTOR,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
        artifact_numeric_values=artifact_numeric_values,
        artifact_numeric_bindings=artifact_numeric_bindings,
    )


def _has_issue(result, *needles: str) -> bool:
    lowered_needles = tuple(needle.lower() for needle in needles)
    return any(
        all(needle in issue.lower() for needle in lowered_needles)
        for issue in result.issues
    )


def _pipeline_issues(
    content: str,
    authoritative_source_text: str,
    *,
    test_cases: list[object] | None = None,
) -> list[str]:
    pipeline = ValidatorPipeline(
        policy_repo_path=Path("/tmp/rulespec-de"),
        axiom_rules_path=Path("/tmp/axiom-rules-engine"),
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    return pipeline._complete_source_unit_issues(
        content,
        validation_source_texts={
            CORPUS_CITATION_PATH: authoritative_source_text,
        },
        test_cases=test_cases,
    )


def _formula_test(
    name: str,
    taxable_income: int,
    expected_tax: int,
) -> dict[str, object]:
    return {
        "name": name,
        "period": "2026",
        "input": {"taxable_income": taxable_income},
        "output": {
            "de:statutes/estg/32a#tariff_income_tax_amount": expected_tax,
        },
    }


def test_constants_without_principal_tariff_output_fail():
    source = """\
(1) Für Einkommen von 12349 bis 17799 Euro ist die Steuer
(914,51 * y + 1400) * y; y ist (Einkommen - 12348) / 10000.
"""
    constants_only = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Constants for the first tariff zone.
rules:
  - name: first_zone_start
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 12349
  - name: first_zone_end
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 17799
  - name: first_zone_quadratic_coefficient
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 914.51
  - name: first_zone_linear_coefficient
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 1400
  - name: basic_allowance
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 12348
  - name: tariff_zone_scale
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 10000
"""

    result = _analyze(constants_only, source, test_cases=[])

    assert result.source_numeric_occurrence_count == 6
    assert result.covered_source_numeric_occurrence_count == 6
    assert result.missing_source_numeric_occurrence_count == 0
    assert _has_issue(result, "principal", "derived/relation")
    assert any(
        "principal derived/relation" in issue
        for issue in _pipeline_issues(constants_only, source, test_cases=[])
    )


ABSATZ_1 = "(1) Die Steuer ist das Einkommen multipliziert mit 10 Prozent."
ABSATZ_5 = (
    "(5) Bei zusammen veranlagten Ehegatten ist die Steuer das Doppelte "
    "der Steuer auf die Hälfte ihres Einkommens."
)
ABSATZ_6 = (
    "(6) Das Splittingverfahren gilt für verwitwete Personen, wenn die "
    "Voraussetzungen nach § 26 und § 32 vorliegen, es sei denn, sie haben "
    "wieder geheiratet."
)

ABSATZ_1_RULES = """\
  - name: tariff_rate
    kind: parameter
    dtype: Rate
    source: de/statute/estg/32a(1)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: rate
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: "Die Steuer ist das Einkommen multipliziert mit 10 Prozent."
    versions:
      - effective_from: '2026-01-01'
        formula: 0.1
  - name: tariff_income_tax_amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: "Die Steuer ist das Einkommen multipliziert mit 10 Prozent."
    versions:
      - effective_from: '2026-01-01'
        formula: taxable_income * tariff_rate
"""

ABSATZ_5_RULE = """\
  - name: joint_assessment_tariff_income_tax_amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(5)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: "Bei zusammen veranlagten Ehegatten ist die Steuer das Doppelte der Steuer auf die Hälfte ihres Einkommens."
    versions:
      - effective_from: '2026-01-01'
        formula: 2 * tariff_income_tax_amount(taxable_income / 2)
"""


def _encoded_absatz_content(*, include_absatz_5: bool) -> str:
    rules = ABSATZ_1_RULES
    if include_absatz_5:
        rules += ABSATZ_5_RULE
    return f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Income-tax tariff.
rules:
{rules}"""


ENCODED_FORMULA_TESTS = [
    _formula_test("individual tariff formula", 100, 10),
    {
        "name": "joint assessment formula",
        "period": "2026",
        "input": {"taxable_income": 100},
        "output": {
            "de:statutes/estg/32a#tariff_income_tax_amount": 10,
            "de:statutes/estg/32a#joint_assessment_tariff_income_tax_amount": 10,
        },
    },
]


def test_omitting_absatz_5_fails():
    result = _analyze(
        _encoded_absatz_content(include_absatz_5=False),
        f"{ABSATZ_1}\n{ABSATZ_5}",
        test_cases=ENCODED_FORMULA_TESTS,
    )

    assert _has_issue(result, "(5)", "source branch")
    assert any(
        "(5)" in issue and "source branch" in issue.lower()
        for issue in _pipeline_issues(
            _encoded_absatz_content(include_absatz_5=False),
            f"{ABSATZ_1}\n{ABSATZ_5}",
            test_cases=ENCODED_FORMULA_TESTS,
        )
    )


def test_silently_omitting_absatz_6_fails():
    result = _analyze(
        _encoded_absatz_content(include_absatz_5=True),
        f"{ABSATZ_1}\n{ABSATZ_5}\n{ABSATZ_6}",
        test_cases=ENCODED_FORMULA_TESTS,
    )

    assert _has_issue(result, "(6)", "source branch")
    assert any(
        "(6)" in issue and "source branch" in issue.lower()
        for issue in _pipeline_issues(
            _encoded_absatz_content(include_absatz_5=True),
            f"{ABSATZ_1}\n{ABSATZ_5}\n{ABSATZ_6}",
            test_cases=ENCODED_FORMULA_TESTS,
        )
    )


def test_precise_typed_absatz_6_deferral_passes():
    content = _encoded_absatz_content(include_absatz_5=True).replace(
        "  summary: Income-tax tariff.\n",
        """\
  summary: Income-tax tariff.
  deferred_outputs:
    - output: de:statutes/estg/32a/6#surviving_spouse_splitting_tax
      reason: >-
        Absatz 6 cannot be computed until taxable income under EStG section 26
        and surviving-spouse or remarriage status under EStG section 32 are
        available.
      blocked_by:
        - de:statutes/estg/26#taxable_income
        - de:statutes/estg/32#surviving_spouse_or_remarried
""",
    )

    result = _analyze(
        content,
        f"{ABSATZ_1}\n{ABSATZ_5}\n{ABSATZ_6}",
        test_cases=ENCODED_FORMULA_TESTS,
    )

    assert not result.issues
    assert not _pipeline_issues(
        content,
        f"{ABSATZ_1}\n{ABSATZ_5}\n{ABSATZ_6}",
        test_cases=ENCODED_FORMULA_TESTS,
    )


def test_scalar_only_source_passes_with_parameter_snapshot():
    source = "(1) Der Freibetrag beträgt 259 Euro."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Der Freibetrag beträgt 259 Euro.
rules:
  - name: allowance_amount
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: "Der Freibetrag beträgt 259 Euro."
    versions:
      - effective_from: '2026-01-01'
        formula: 259
"""
    test_cases = [
        {
            "name": "allowance parameter snapshot",
            "period": "2026",
            "input": {},
            "output": {"de:statutes/estg/32a#allowance_amount": 259},
        }
    ]

    result = _analyze(content, source, test_cases=test_cases)

    assert not result.issues
    assert result.source_numeric_occurrence_count == 1
    assert result.covered_source_numeric_occurrence_count == 1
    assert result.missing_source_numeric_occurrence_count == 0
    assert not _pipeline_issues(content, source, test_cases=test_cases)


def test_summary_omission_cannot_hide_authoritative_source_number():
    source = "(1) Der Freibetrag beträgt 259 Euro; der Zuschlag beträgt 73 Euro."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Der Freibetrag beträgt 259 Euro.
rules:
  - name: allowance_amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 259
"""

    result = _analyze(content, source, test_cases=[])

    assert result.source_numeric_occurrence_count == 2
    assert result.covered_source_numeric_occurrence_count == 1
    assert result.missing_source_numeric_occurrence_count == 1
    assert _has_issue(result, "73", "authoritative")
    assert any(
        "authoritative corpus numeric value 73" in issue.lower()
        for issue in _pipeline_issues(content, source, test_cases=[])
    )


def test_de_grouped_decimal_is_one_typed_recall_occurrence():
    source = "(1) Der Betrag beträgt 19 470,38 Euro."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Der Betrag ist festgelegt.
rules:
  - name: grouped_decimal_amount
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 19470.38
"""

    result = _analyze(content, source, test_cases=[])

    assert not result.issues
    assert result.source_numeric_occurrence_count == 1
    assert result.covered_source_numeric_occurrence_count == 1


@pytest.mark.parametrize(
    ("source", "expected_missing"),
    [
        ("(1) Die Frist beträgt 42 Tage.", 1),
        ("(1) Der Satz beträgt 42 Prozent.", 0),
    ],
)
def test_rate_scaling_requires_occurrence_local_context(source, expected_missing):
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Ein Wert ist festgelegt.
rules:
  - name: candidate_value
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 0.42
"""

    result = _analyze(content, source, test_cases=[])

    assert result.source_numeric_occurrence_count == 1
    assert result.missing_source_numeric_occurrence_count == expected_missing


def test_same_valued_boundaries_keep_their_own_rate_context():
    branch = recognize_source_structure("(1) Die Grenze gilt.")[0]
    day_occurrence = DE_NUMERIC_OCCURRENCE_EXTRACTOR("42 Tage")[0]
    rate_occurrence = DE_NUMERIC_OCCURRENCE_EXTRACTOR("42 Prozent")[0]
    principal_rules = {
        "candidate": {
            "versions": [
                {
                    "effective_from": "2026-01-01",
                    "formula": "selector <= 0.42",
                }
            ]
        }
    }
    principal_rule_paths = {"candidate": {branch.path}}
    asserted_by_rule = {
        "candidate": [
            {
                "input": {"selector": 0.42},
                "output": {"candidate": True},
            }
        ]
    }

    assert not completeness_module._branch_boundary_has_test_evidence(
        branch,
        day_occurrence,
        principal_rules=principal_rules,
        principal_rule_paths=principal_rule_paths,
        asserted_by_rule=asserted_by_rule,
        numeric_value_is_grounded=numeric_value_is_grounded,
        extract_numeric_occurrences=DE_NUMERIC_OCCURRENCE_EXTRACTOR,
    )
    assert completeness_module._branch_boundary_has_test_evidence(
        branch,
        rate_occurrence,
        principal_rules=principal_rules,
        principal_rule_paths=principal_rule_paths,
        asserted_by_rule=asserted_by_rule,
        numeric_value_is_grounded=numeric_value_is_grounded,
        extract_numeric_occurrences=DE_NUMERIC_OCCURRENCE_EXTRACTOR,
    )


def test_trusted_requested_path_controls_authoritative_recall_body():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/other
  summary: Der Freibetrag beträgt 259 Euro.
rules:
  - name: allowance_amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 259
"""
    pipeline = ValidatorPipeline(
        policy_repo_path=Path("/tmp/rulespec-de"),
        axiom_rules_path=Path("/tmp/axiom-rules-engine"),
        local_corpus_release=None,
        enable_oracles=False,
        source_citation_path=CORPUS_CITATION_PATH,
        require_complete_source_unit=True,
    )

    issues = pipeline._complete_source_unit_issues(
        content,
        validation_source_texts={
            CORPUS_CITATION_PATH: "(1) Der Zuschlag beträgt 73 Euro.",
            "de/statute/estg/other": "(1) Der Freibetrag beträgt 259 Euro.",
        },
        test_cases=[],
    )

    assert any("numeric value 73" in issue.lower() for issue in issues)


def test_deferral_prose_and_targets_cannot_hide_authoritative_source_number():
    source = "(1) Der Freibetrag beträgt 259 Euro; der Zuschlag beträgt 73 Euro."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Der Freibetrag beträgt 259 Euro.
  deferred_outputs:
    - output: de:statutes/estg/32a/1#supplement_73_amount
      reason: The 73 Euro supplement requires an unavailable dependency.
      blocked_by:
        - de:statutes/estg/99#dependency_73_value
rules:
  - name: allowance_amount
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 259
"""

    issues = _pipeline_issues(content, source, test_cases=[])

    assert any(
        "authoritative corpus numeric value 73" in issue.lower()
        for issue in issues
    )


@pytest.mark.parametrize(
    ("source", "marker"),
    [
        ("(1) Die Hauptregel gilt.", "(1)"),
        ("(2) Die zweite Absatzregel gilt.", "(2)"),
        ("1. Die erste Nummer gilt.", "1."),
        ("1a. Die erweiterte Nummer gilt.", "1a."),
        ("a) Die Buchstabenregel gilt.", "a)"),
        ("Die Hauptregel gilt.2Die Sonderregel gilt.", "Satz 2"),
    ],
)
def test_german_structural_markers_require_coverage(source: str, marker: str):
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Leeres Beispiel.
rules: []
"""

    result = _analyze(content, source, test_cases=[])

    assert _has_issue(result, marker, "source branch")


def test_imprecise_absatz_deferral_does_not_cover_branch():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Splittingverfahren.
  deferred_outputs:
    - output: de:statutes/estg/32a/6#splitting_rule
      reason: Nicht umgesetzt.
rules: []
"""

    result = _analyze(content, ABSATZ_6, test_cases=[])

    assert _has_issue(result, "(6)", "deferral")


def test_deferral_cannot_name_its_own_branch_as_missing_dependency():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Splittingverfahren.
  deferred_outputs:
    - output: de:statutes/estg/32a/6#splitting_rule
      reason: Absatz 6 fehlt.
rules: []
"""

    result = _analyze(content, ABSATZ_6, test_cases=[])

    assert _has_issue(result, "(6)", "deferral", "dependency")


@pytest.mark.parametrize(
    "reason",
    [
        "Absatz 6 hängt von Absatz 5 ab.",
        "Satz 2 fehlt.",
        "Nummer 1 wird benötigt.",
        "Buchstabe a ist nicht codiert.",
    ],
)
def test_deferral_cannot_use_bare_internal_structure_as_dependency(reason: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  deferred_outputs:
    - output: de:statutes/estg/32a/6#splitting_rule
      reason: {reason}
rules: []
"""

    result = _analyze(content, ABSATZ_6, test_cases=[])

    assert _has_issue(result, "(6)", "deferral", "dependency")


def test_deferral_cannot_use_another_output_in_same_unit_as_blocker():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  deferred_outputs:
    - output: de:statutes/estg/32a/6#splitting_rule
      reason: Absatz 6 benötigt die noch fehlende Tarifberechnung.
      blocked_by:
        - de:statutes/estg/32a#tariff_income_tax_amount
rules: []
"""

    result = _analyze(content, ABSATZ_6, test_cases=[])

    assert _has_issue(result, "(6)", "deferral", "dependency")


@pytest.mark.parametrize(
    "reason",
    [
        "Arbitrary placeholder.",
        "Requires an external assessment base.",
    ],
)
def test_deferral_blocker_must_be_identified_by_dependency_reason(reason: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  deferred_outputs:
    - output: de:statutes/estg/32a/6#splitting_rule
      reason: {reason}
      blocked_by:
        - de:statutes/estg/9999#fictional_amount
rules: []
"""

    result = _analyze(content, ABSATZ_6, test_cases=[])

    assert _has_issue(result, "(6)", "deferral", "dependency")


def test_deferral_cannot_invent_source_unstated_section():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  deferred_outputs:
    - output: de:statutes/estg/32a/6#splitting_rule
      reason: Cannot be computed because section 9999 is missing.
      blocked_by:
        - de:statutes/estg/9999#fictional_amount
rules: []
"""
    source = "(6) Das Einkommen wird mit 2 multipliziert."

    result = _analyze(content, source, test_cases=[])

    assert _has_issue(result, "(6)", "deferral", "dependency")


@pytest.mark.parametrize(
    "blocker",
    [
        "de:statutes/estg/9999#bemessungsgrundlage",
        "de:statutes/fake/26#bemessungsgrundlage",
        "xx:statutes/anything/26#bemessungsgrundlage",
        "xx:statutes/estg/26#bemessungsgrundlage",
        "de:regulations/estg/26#bemessungsgrundlage",
    ],
)
def test_deferral_symbol_cannot_mask_a_conflicting_citation(blocker: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  deferred_outputs:
    - output: de:statutes/estg/32a/1#amount
      reason: >-
        Absatz 1 cannot be computed because the Bemessungsgrundlage is missing.
      blocked_by:
        - {blocker}
rules: []
"""
    source = "(1) Maßgeblich ist die Bemessungsgrundlage nach § 26."

    result = _analyze(content, source, test_cases=[])

    assert _has_issue(result, "(1)", "deferral", "dependency")


def test_prose_deferral_rejects_wrong_absolute_jurisdiction():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  deferred_outputs:
    - output: de:statutes/estg/32a/1#amount
      reason: >-
        Cannot be computed until
        xx:statutes/estg/26#bemessungsgrundlage is available.
rules: []
"""
    source = "(1) Maßgeblich ist die Bemessungsgrundlage nach § 26."

    result = _analyze(content, source, test_cases=[])

    assert _has_issue(result, "(1)", "deferral", "dependency")


def test_invalid_present_blocker_cannot_fall_back_to_prose_dependency():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  deferred_outputs:
    - output: de:statutes/estg/32a/6#splitting_rule
      reason: Cannot be computed until the conditions under section 26 exist.
      blocked_by:
        - de:statutes/estg/32a#same_unit
rules: []
"""

    result = _analyze(content, ABSATZ_6, test_cases=[])

    assert _has_issue(result, "(6)", "deferral", "dependency")


def test_root_deferral_does_not_blanket_structured_source_unit():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  deferred_outputs:
    - output: de:statutes/estg/32a#whole_unit
      reason: Requires an external assessment base.
      blocked_by:
        - de:statutes/estg/26#assessment_base
rules: []
"""

    result = _analyze(content, "(1) Regel eins.\n(2) Regel zwei.", test_cases=[])

    assert _has_issue(result, "(1)", "source branch")
    assert _has_issue(result, "(2)", "source branch")


def test_precise_non_root_deferral_covers_nested_list_branches():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  deferred_outputs:
    - output: de:statutes/estg/32a/1#paragraph_rule
      reason: Requires an external assessment base.
      blocked_by:
        - de:statutes/estg/26#assessment_base
rules: []
"""
    source = (
        "(1) Für die Bemessungsgrundlage nach § 26 gelten:\n"
        "1. Zweig eins;\n"
        "2. Zweig zwei."
    )

    result = _analyze(content, source, test_cases=[])

    assert not result.issues


@pytest.mark.parametrize(
    "source",
    [
        "Der Betrag ist ein Drittel des Einkommens.",
        "Der Betrag ist die Summe aus Einkommen und Zuschlag.",
        "Der Betrag wird um 20 Prozent des Einkommens vermindert.",
        "Der Betrag ist durch drei geteilt zu berechnen.",
        "Der Betrag erhöht sich um 25 Euro.",
        "Der Betrag mindert sich um 25 Euro.",
        "Der Betrag entspricht dem 1,5fachen des Einkommens.",
        "Der Betrag beträgt 45 vom Hundert des Einkommens.",
        "Der Betrag ist das Doppelte des Einkommens.",
        "Der Betrag ist das Produkt aus Einkommen und Faktor.",
        "Der Betrag entspricht Einkommen mal zwei.",
        "Der Betrag wird verdoppelt.",
    ],
)
def test_common_german_formula_language_is_computation(source: str):
    assert source_states_explicit_computation(source)


@pytest.mark.parametrize(
    ("source", "parameter_value"),
    [
        ("(1) Der Betrag erhöht sich um 25 Euro.", 25),
        ("(1) Der Betrag mindert sich um 25 Euro.", 25),
        ("(1) Der Betrag entspricht dem 1,5fachen des Einkommens.", 1.5),
        ("(1) Der Betrag beträgt 45 vom Hundert des Einkommens.", 0.45),
        ("(1) Der Betrag ist das Doppelte des Einkommens.", 2),
        ("(1) Der Betrag ist das Produkt aus Einkommen und Faktor.", 2),
        ("(1) Der Betrag entspricht Einkommen mal zwei.", 2),
        ("(1) Der Betrag wird verdoppelt.", 2),
    ],
)
def test_common_german_formula_language_rejects_parameter_only(
    source: str,
    parameter_value: float,
):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: formula_constant
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: {parameter_value}
"""

    result = _analyze(content, source, test_cases=[])

    assert _has_issue(result, "formula-output", "parameter-only")


@pytest.mark.parametrize(
    "source",
    [
        "Der Betrag ist das Dreifache des Einkommens.",
        "The amount equals income plus supplement.",
    ],
)
def test_additional_formula_language_is_computation(source: str):
    assert source_states_explicit_computation(source)


def test_scalar_amount_language_is_not_computation():
    assert not source_states_explicit_computation(
        "Der Freibetrag beträgt 259 Euro."
    )
    assert not source_states_explicit_computation(
        "Der Satz beträgt 45 vom Hundert."
    )


def test_scalar_year_span_is_not_computation():
    assert not source_states_explicit_computation(
        "Für den Zeitraum 2025/2026 beträgt der Freibetrag 259 Euro."
    )


def test_indented_absatz_markers_are_recognized():
    branches = recognize_source_structure("  (1) Regel eins.\n\t(2) Regel zwei.")

    assert {branch.path for branch in branches} >= {("1",), ("2",)}


def test_nested_satz_source_references_bind_to_paragraph_paths():
    source = """\
(1) 1Die erste Berechnung ergibt sich aus Einkommen * 2.2Die zweite Berechnung ergibt sich aus Einkommen * 3.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: first_amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1) Satz 1
    versions:
      - effective_from: '2026-01-01'
        formula: income * 2
  - name: second_amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1) Satz 2
    versions:
      - effective_from: '2026-01-01'
        formula: income * 3
"""
    test_cases = [
        {
            "name": "first sentence",
            "input": {"income": 10},
            "output": {"first_amount": 20},
        },
        {
            "name": "second sentence",
            "input": {"income": 10},
            "output": {"second_amount": 30},
        },
    ]

    paths = completeness_module._paths_from_source_reference(
        "de/statute/estg/32a(1) Satz 1; "
        "de/statute/estg/32a(1) Satz 2",
        corpus_citation_path=CORPUS_CITATION_PATH,
    )
    result = _analyze(content, source, test_cases=test_cases)

    assert {("1", "satz-1"), ("1", "satz-2")} <= paths
    assert not _has_issue(result, "source branch", "neither encoded")


def test_numeric_recall_does_not_credit_rule_or_formula_identifier_digits():
    source = "(1) Der Zuschlag beträgt 73 Euro."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: placeholder_73
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: missing_value_73
    verification:
      values: [73]
"""

    result = _analyze(content, source, test_cases=[])

    assert _has_issue(result, "73", "numeric-recall")


def test_imported_numeric_recall_only_credits_explicit_imported_scalar():
    content = """\
format: rulespec/v1
imports:
  - de:statutes/estg/99#needed_amount
rules: []
"""
    imported = """\
format: rulespec/v1
rules:
  - name: needed_amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 5
  - name: unrelated_73_amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 73
"""

    values = collect_artifact_numeric_values(
        content,
        extract_named_scalars=extract_named_scalar_occurrences,
        imported_symbol_contents=(("needed_amount", imported),),
    )

    assert values == (5.0,)


def test_imported_scalar_bindings_witness_exact_formula_branches():
    source = """\
(1) Bei Ehegatten ist der Betrag Einkommen * 2.
(2) Bei Alleinstehenden ist der Betrag Einkommen * 3.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
imports:
  - de:statutes/constants#married_multiplier
  - de:statutes/constants#single_multiplier
rules:
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1); de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if married:
            income * married_multiplier
          else:
            income * single_multiplier
"""
    bindings = (
        ("married_multiplier", 2.0),
        ("single_multiplier", 3.0),
    )
    cases = [
        {
            "name": "married",
            "input": {"married": True, "income": 10},
            "output": {"amount": 20},
        },
        {
            "name": "single",
            "input": {"married": False, "income": 10},
            "output": {"amount": 30},
        },
    ]

    result = _analyze(
        content,
        source,
        test_cases=cases,
        artifact_numeric_values=(2.0, 3.0),
        artifact_numeric_bindings=bindings,
    )

    assert not result.issues


def test_artifact_numeric_bindings_preserve_imported_symbol_names():
    content = """\
format: rulespec/v1
imports:
  - de:statutes/constants#needed_amount
rules: []
"""
    imported = """\
format: rulespec/v1
rules:
  - name: needed_amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 5
"""

    bindings = collect_artifact_numeric_bindings(
        content,
        extract_named_scalars=extract_named_scalar_occurrences,
        imported_symbol_contents=(("needed_amount", imported),),
    )

    assert bindings == (("needed_amount", 5.0),)


def test_complete_import_inventory_does_not_walk_transitive_same_name(
    tmp_path,
    monkeypatch,
):
    main_file = tmp_path / "main.yaml"
    direct_file = tmp_path / "direct.yaml"
    transitive_file = tmp_path / "transitive.yaml"
    main_content = """\
format: rulespec/v1
imports:
  - de:statutes/direct#threshold
rules: []
"""
    direct_content = """\
format: rulespec/v1
imports:
  - de:statutes/transitive#threshold
rules:
  - name: threshold
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 5
"""
    transitive_content = """\
format: rulespec/v1
rules:
  - name: threshold
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 73
"""
    main_file.write_text(main_content)
    direct_file.write_text(direct_content)
    transitive_file.write_text(transitive_content)
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=Path("/tmp/axiom-rules-engine"),
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    monkeypatch.setattr(
        pipeline,
        "_validation_source_root",
        lambda _rules_file: tmp_path,
    )
    monkeypatch.setattr(
        validator_pipeline_module,
        "_resolve_rulespec_import_file_static",
        lambda import_path, **_kwargs: (
            direct_file
            if import_path == "de:statutes/direct#threshold"
            else transitive_file
        ),
    )

    imported_symbol_contents = (
        pipeline._complete_source_unit_import_symbol_contents(main_file)
    )
    values = collect_artifact_numeric_values(
        main_content,
        extract_named_scalars=extract_named_scalar_occurrences,
        imported_symbol_contents=imported_symbol_contents,
    )

    assert imported_symbol_contents == (("threshold", direct_content),)
    assert values == (5.0,)


def test_complete_import_inventory_binds_same_name_to_exact_direct_file(
    tmp_path,
    monkeypatch,
):
    main_file = tmp_path / "main.yaml"
    first_file = tmp_path / "first.yaml"
    second_file = tmp_path / "second.yaml"
    main_content = """\
format: rulespec/v1
imports:
  - de:statutes/first#threshold
  - de:statutes/second#other
rules: []
"""
    first_content = """\
format: rulespec/v1
rules:
  - name: threshold
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 5
"""
    second_content = """\
format: rulespec/v1
rules:
  - name: other
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 11
  - name: threshold
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 73
"""
    main_file.write_text(main_content)
    first_file.write_text(first_content)
    second_file.write_text(second_content)
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=Path("/tmp/axiom-rules-engine"),
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    monkeypatch.setattr(
        pipeline,
        "_validation_source_root",
        lambda _rules_file: tmp_path,
    )
    monkeypatch.setattr(
        validator_pipeline_module,
        "_resolve_rulespec_import_file_static",
        lambda import_path, **_kwargs: {
            "de:statutes/first#threshold": first_file,
            "de:statutes/second#other": second_file,
        }[import_path],
    )

    imported_symbol_contents = (
        pipeline._complete_source_unit_import_symbol_contents(main_file)
    )
    values = collect_artifact_numeric_values(
        main_content,
        extract_named_scalars=extract_named_scalar_occurrences,
        imported_symbol_contents=imported_symbol_contents,
    )

    assert imported_symbol_contents == (
        ("threshold", first_content),
        ("other", second_content),
    )
    assert values == (5.0, 11.0)


def test_glued_satz_markers_are_paragraph_children_not_list_children():
    source = """\
(1) 1Die tarifliche Einkommensteuer wird nach Nummern berechnet.
1. Grundtarif;
5. Spitzentarif.3Die Steuer ist abzurunden.4Die Grenze gilt entsprechend.
(6) 1Das Verfahren gilt für Ehegatten.
1. Voraussetzung eins,
2. Voraussetzung zwei,
c) Voraussetzung drei.2Voraussetzung ist außerdem der Status.
"""

    paths = {branch.path for branch in recognize_source_structure(source)}

    assert ("1", "satz-3") in paths
    assert ("1", "satz-4") in paths
    assert ("6", "satz-2") in paths
    assert ("1", "5", "satz-3") not in paths
    assert ("6", "2", "c", "satz-2") not in paths


RELEASED_ESTG_32A_BODY = """\
(1) 1Die tarifliche Einkommensteuer bemisst sich nach dem auf volle Euro abgerundeten zu versteuernden Einkommen. 2Sie beträgt ab dem Veranlagungszeitraum 2026 vorbehaltlich der §§ 32b, 32d, 34, 34a, 34b und 34c jeweils in Euro für zu versteuernde Einkommen
1. bis 12 348 Euro (Grundfreibetrag):0;
2. von 12 349 Euro bis 17 799 Euro:(914,51 • y + 1 400) • y;
3. von 17 800 Euro bis 69 878 Euro:(173,10 • z + 2 397) • z + 1 034,87;
4. von 69 879 Euro bis 277 825 Euro:0,42 • x – 11 135,63;
5. von 277 826 Euro an:0,45 • x – 19 470,38.3Die Größe „y“ ist ein Zehntausendstel des den Grundfreibetrag übersteigenden Teils des auf einen vollen Euro-Betrag abgerundeten zu versteuernden Einkommens. 4Die Größe „z“ ist ein Zehntausendstel des 17 799 Euro übersteigenden Teils des auf einen vollen Euro-Betrag abgerundeten zu versteuernden Einkommens. 5Die Größe „x“ ist das auf einen vollen Euro-Betrag abgerundete zu versteuernde Einkommen. 6Der sich ergebende Steuerbetrag ist auf den nächsten vollen Euro-Betrag abzurunden.
(2) bis (4) (weggefallen)
(5) Bei Ehegatten, die nach den §§ 26, 26b zusammen zur Einkommensteuer veranlagt werden, beträgt die tarifliche Einkommensteuer vorbehaltlich der §§ 32b, 32d, 34, 34a, 34b und 34c das Zweifache des Steuerbetrags, der sich für die Hälfte ihres gemeinsam zu versteuernden Einkommens nach Absatz 1 ergibt (Splitting-Verfahren).
(6) 1Das Verfahren nach Absatz 5 ist auch anzuwenden zur Berechnung der tariflichen Einkommensteuer für das zu versteuernde Einkommen
1. bei einem verwitweten Steuerpflichtigen für den Veranlagungszeitraum, der dem Kalenderjahr folgt, in dem der Ehegatte verstorben ist, wenn der Steuerpflichtige und sein verstorbener Ehegatte im Zeitpunkt seines Todes die Voraussetzungen des § 26 Absatz 1 Satz 1 erfüllt haben,
2. bei einem Steuerpflichtigen, dessen Ehe in dem Kalenderjahr, in dem er sein Einkommen bezogen hat, aufgelöst worden ist, wenn in diesem Kalenderjahr
a) der Steuerpflichtige und sein bisheriger Ehegatte die Voraussetzungen des § 26 Absatz 1 Satz 1 erfüllt haben,
b) der bisherige Ehegatte wieder geheiratet hat und
c) der bisherige Ehegatte und dessen neuer Ehegatte ebenfalls die Voraussetzungen des § 26 Absatz 1 Satz 1 erfüllen.2Voraussetzung für die Anwendung des Satzes 1 ist, dass der Steuerpflichtige nicht nach den §§ 26, 26a einzeln zur Einkommensteuer veranlagt wird."""


def test_exact_released_estg_32a_structure_paths():
    paths = {
        branch.path
        for branch in recognize_source_structure(RELEASED_ESTG_32A_BODY)
    }

    assert {("1",), ("5",), ("6",)} <= paths
    assert ("2",) not in paths
    assert {("1", str(number)) for number in range(1, 6)} <= paths
    assert {("6", "1"), ("6", "2")} <= paths
    assert {("6", "2", letter) for letter in ("a", "b", "c")} <= paths
    assert {("1", f"satz-{number}") for number in range(1, 7)} <= paths
    assert {("6", "satz-1"), ("6", "satz-2")} <= paths
    assert not any(
        path[:2] == ("1", "5") and path[-1].startswith("satz-")
        for path in paths
    )
    assert not any(
        path[:3] == ("6", "2", "c") and path[-1].startswith("satz-")
        for path in paths
    )


def test_released_estg_32a_typed_absatz_6_deferral_passes():
    source = "(6) " + RELEASED_ESTG_32A_BODY.split("\n(6) ", 1)[1]
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  deferred_outputs:
    - output: de:statutes/estg/32a/6#surviving_spouse_splitting_tax
      reason: >-
        Absatz 6 cannot be computed until the exact joint-assessment
        eligibility conditions under EStG section 26 are available.
      blocked_by:
        - de:statutes/estg/26#joint_assessment_eligibility
rules: []
"""

    result = _analyze(content, source, test_cases=[])

    assert not result.issues
    assert not _pipeline_issues(content, source, test_cases=[])


def test_released_estg_32a_constants_without_tariff_output_fail():
    values = (
        occurrence.value
        for occurrence in DE_NUMERIC_OCCURRENCE_EXTRACTOR(
            authoritative_numeric_recall_text(RELEASED_ESTG_32A_BODY)
        )
    )
    rules = "\n".join(
        f"""\
  - name: released_constant_{index}
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: {value!r}"""
        for index, value in enumerate(dict.fromkeys(values), start=1)
    )
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Released constants only.
rules:
{rules}
"""

    result = _analyze(content, RELEASED_ESTG_32A_BODY, test_cases=[])

    assert result.missing_source_numeric_occurrence_count == 0
    assert _has_issue(result, "principal", "derived/relation")


def test_released_estg_32a_omitted_absatz_5_and_6_are_visible():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: paragraph_one_snapshot
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 0
"""

    result = _analyze(content, RELEASED_ESTG_32A_BODY, test_cases=[])

    assert _has_issue(result, "(5)", "source branch")
    assert _has_issue(result, "(6)", "source branch")


def test_released_estg_32a_does_not_invent_boundary_from_omission_line():
    branches = recognize_source_structure(RELEASED_ESTG_32A_BODY)
    formula_branches = completeness_module._source_formula_branches(
        RELEASED_ESTG_32A_BODY,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    obligations = completeness_module._source_boundary_obligations(
        branches,
        narrative_formula_branches=formula_branches,
        extract_numeric_occurrences=DE_NUMERIC_OCCURRENCE_EXTRACTOR,
    )

    assert not any(branch.path == () for branch, _value in obligations)
    assert not any(occurrence.value == 34 for _branch, occurrence in obligations)
    assert any(
        occurrence.value == 12348 for _branch, occurrence in obligations
    )


def test_released_estg_32a_feminine_rounding_is_computation():
    assert source_states_explicit_computation(
        "Die Größe x ist das auf volle Euro abgerundete Einkommen."
    )


def test_letter_formula_boundaries_are_inventoried():
    source = """\
(1) Die Berechnung umfasst:
1. folgende Zweige:
a) Bis 73 Euro: Einkommen * 2;
b) Von 74 Euro an: Einkommen * 3.
"""
    branches = recognize_source_structure(source)
    formula_branches = completeness_module._source_formula_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )

    obligations = completeness_module._source_boundary_obligations(
        branches,
        narrative_formula_branches=formula_branches,
        extract_numeric_occurrences=DE_NUMERIC_OCCURRENCE_EXTRACTOR,
    )

    assert any(
        branch.path[-1] == "a" and occurrence.value == 73
        for branch, occurrence in obligations
    )
    assert any(
        branch.path[-1] == "b" and occurrence.value == 74
        for branch, occurrence in obligations
    )


def test_common_german_range_boundaries_are_inventoried():
    source = """\
(1) Die Berechnung umfasst:
1. Zwischen 100 und 200 Euro: Einkommen * 2;
2. Nicht mehr als 300 Euro: Einkommen * 3.
"""
    branches = recognize_source_structure(source)
    formula_branches = completeness_module._source_formula_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )

    obligations = completeness_module._source_boundary_obligations(
        branches,
        narrative_formula_branches=formula_branches,
        extract_numeric_occurrences=DE_NUMERIC_OCCURRENCE_EXTRACTOR,
    )
    boundaries = {
        (branch.path, occurrence.value)
        for branch, occurrence in obligations
    }

    assert (("1", "1"), 100) in boundaries
    assert (("1", "1"), 200) in boundaries
    assert (("1", "2"), 300) in boundaries


@pytest.mark.parametrize(
    "phrase",
    [
        "außer bei einer Befreiung",
        "ausgenommen Härtefälle",
        "abweichend von der Hauptregel",
        "jedoch nicht bei Einzelveranlagung",
    ],
)
def test_common_german_exception_language_is_inventoried(phrase: str):
    source = f"(1) Die Steuer ist Einkommen * 2, {phrase}."
    branches = recognize_source_structure(source)

    obligations = completeness_module._source_exception_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )

    assert obligations
    assert obligations[0].path == ("1",)


def test_nested_editorial_omission_does_not_drop_operative_paragraph():
    source = """\
(1) Die Hauptregel gilt:
1. weggefallen;
2. Die operative Ausnahme gilt.
"""

    branches = recognize_source_structure(source)
    paths = {branch.path for branch in branches}

    assert ("1",) in paths
    assert ("1", "1") not in paths
    assert ("1", "2") in paths


def test_unrelated_proof_citation_cannot_cover_requested_branch():
    source = "(1) Die Steuer ist Einkommen * 10."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Tarif.
rules:
  - name: tariff_income_tax_amount
    kind: derived
    dtype: Money
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: de/statute/bgbl/example
              excerpt: "Die Steuer ist Einkommen * 10."
    versions:
      - effective_from: '2026-01-01'
        formula: taxable_income * 10
"""

    result = _analyze(content, source, test_cases=[])

    assert _has_issue(result, "(1)", "source branch")


def test_unrelated_rule_source_cannot_cover_requested_branch():
    source = "(1) Die Hauptregel gilt."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Hauptregel.
rules:
  - name: unrelated_output
    kind: derived
    dtype: Money
    source: de/statute/other/99 Absatz 1
    versions:
      - effective_from: '2026-01-01'
        formula: unrelated_value
"""

    result = _analyze(content, source, test_cases=[])

    assert _has_issue(result, "(1)", "source branch")


def test_unstructured_formula_requires_principal_output_bound_to_source_unit():
    source = (
        "Die Steuer ergibt sich aus dem Einkommen geteilt durch den Grundwert."
    )
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Steuerberechnung.
rules:
  - name: unrelated_output
    kind: derived
    dtype: Money
    source: de/statute/other/99
    versions:
      - effective_from: '2026-01-01'
        formula: unrelated_value
"""
    test_cases = [
        {
            "name": "unrelated output",
            "period": "2026",
            "input": {"unrelated_value": 5},
            "output": {"de:statutes/estg/32a#unrelated_output": 5},
        }
    ]

    unrelated = _analyze(content, source, test_cases=test_cases)
    bound = _analyze(
        content.replace("de/statute/other/99", CORPUS_CITATION_PATH),
        source,
        test_cases=test_cases,
    )

    assert _has_issue(unrelated, "explicit source computation", "principal")
    assert not _has_issue(bound, "formula-output")


def test_unstructured_formula_requires_unit_level_deferral():
    source = (
        "Die Steuer ergibt sich aus dem Einkommen geteilt durch den Grundwert "
        "nach § 26."
    )
    child_deferral = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Steuerberechnung.
  deferred_outputs:
    - output: de:statutes/estg/32a/99#income_tax_amount
      reason: Requires the missing assessment base under EStG section 26.
      blocked_by:
        - de:statutes/estg/26#assessment_base
rules: []
"""

    unrelated = _analyze(child_deferral, source, test_cases=[])
    whole_unit = _analyze(
        child_deferral.replace("/32a/99#", "/32a#"),
        source,
        test_cases=[],
    )

    assert _has_issue(unrelated, "explicit source computation", "principal")
    assert not whole_unit.issues


def test_each_formula_clause_requires_principal_output_evidence():
    source = (
        "(1) Der erste Betrag ist Einkommen * 2; "
        "der zweite Betrag ist Einkommen * 3 und auf volle Euro abzurunden."
    )
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: first_multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: unused_second_multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 3
  - name: computed_amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: "Der erste Betrag ist Einkommen * 2;"
          - path: dtype
            kind: formula
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: >-
                der zweite Betrag ist Einkommen * 3 und auf volle Euro
                abzurunden.
    versions:
      - effective_from: '2026-01-01'
        formula: floor(income * first_multiplier)
"""
    test_cases = [
        {
            "name": "first ordinary case",
            "input": {"income": 10},
            "output": {"computed_amount": 20},
        },
        {
            "name": "second ordinary case",
            "input": {"income": 20},
            "output": {"computed_amount": 40},
        },
    ]

    result = _analyze(content, source, test_cases=test_cases)

    assert _has_issue(
        result,
        "formula-output",
        "formula clause 2",
        "principal",
    )


COMPANION_COVERAGE_SOURCE = """\
(1) Die Steuer wird nach folgenden Tarifzweigen berechnet:
1. Bis 100 Euro: Einkommen * 5 Prozent;
2. Von 101 Euro an: Einkommen * 7 Prozent.
An der Tarifgrenze von 101 Euro gilt der obere Tarifzweig.
Ausnahme: Bei einer Befreiung beträgt die Steuer null.
Das Ergebnis ist auf volle Euro abzurunden.
"""

COMPANION_COVERAGE_CONTENT = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Tariff branches, boundary, exception, and rounding.
rules:
  - name: lower_tariff_rate_percent
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 5
  - name: upper_tariff_rate_percent
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)(2)
    versions:
      - effective_from: '2026-01-01'
        formula: 7
  - name: lower_tariff_maximum
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 100
  - name: upper_tariff_minimum
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)(2)
    versions:
      - effective_from: '2026-01-01'
        formula: 101
  - name: tariff_income_tax_amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: "1. Bis 100 Euro: Einkommen * 5 Prozent;"
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: "2. Von 101 Euro an: Einkommen * 7 Prozent."
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if exception_applies:
            0
          else:
            floor(
              if taxable_income <= lower_tariff_maximum:
                taxable_income * lower_tariff_rate_percent / 100
              else:
                taxable_income * upper_tariff_rate_percent / 100
            )
"""


def _companion_test(
    name: str,
    taxable_income: float,
    expected_tax: int,
    *,
    exception_applies: bool = False,
) -> dict[str, object]:
    return {
        "name": name,
        "period": "2026",
        "input": {
            "taxable_income": taxable_income,
            "exception_applies": exception_applies,
        },
        "output": {
            "de:statutes/estg/32a#tariff_income_tax_amount": expected_tax,
        },
    }


COMPLETE_COMPANION_TESTS = [
    _companion_test("lower tariff formula branch", 90, 4),
    _companion_test("upper tariff formula branch", 110, 7),
    _companion_test("lower exact tariff boundary", 100, 5),
    _companion_test("upper exact tariff boundary", 101, 7),
    _companion_test("exception does not apply", 90, 4),
    _companion_test("exception applies", 90, 0, exception_applies=True),
    _companion_test("rounding down to full euro", 90.5, 4),
]


def test_complete_companion_suite_covers_source_controls():
    result = _analyze(
        COMPANION_COVERAGE_CONTENT,
        COMPANION_COVERAGE_SOURCE,
        test_cases=COMPLETE_COMPANION_TESTS,
    )

    assert not result.issues


def test_predicate_only_boundary_requires_an_exact_boundary_case():
    source = "(1) Der Anspruch gilt bis 100 Euro Einkommen."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: income_limit
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 100
  - name: eligible
    kind: derived
    dtype: bool
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: income <= income_limit
"""

    def case(income: int) -> dict[str, object]:
        return {
            "name": f"income {income}",
            "input": {"income": income},
            "output": {"eligible": income <= 100},
        }

    below_only = _analyze(content, source, test_cases=[case(50)])
    exact_boundary = _analyze(content, source, test_cases=[case(100)])

    assert _has_issue(below_only, "boundary", "test")
    assert not exact_boundary.issues


def test_predicate_only_exception_requires_paired_cases():
    source = """\
(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: eligible
    kind: derived
    dtype: bool
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if exemption_applies:
            false
          else:
            true
"""

    def case(exemption_applies: bool) -> dict[str, object]:
        return {
            "name": f"exemption={exemption_applies}",
            "input": {"exemption_applies": exemption_applies},
            "output": {"eligible": not exemption_applies},
        }

    ordinary_only = _analyze(content, source, test_cases=[case(False)])
    paired = _analyze(
        content,
        source,
        test_cases=[case(False), case(True)],
    )

    assert _has_issue(ordinary_only, "exception", "test")
    assert not paired.issues


NARRATIVE_PIECEWISE_SOURCE = """\
(1) Bis 100 Euro wird die Steuer als Einkommen * 5 Prozent berechnet; über 100 Euro wird die Steuer als Einkommen * 7 Prozent berechnet.
"""

NARRATIVE_PIECEWISE_CONTENT = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Narrative piecewise tariff.
rules:
  - name: tariff_boundary
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 100
  - name: lower_tariff_rate_percent
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 5
  - name: upper_tariff_rate_percent
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 7
  - name: tariff_income_tax_amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: "Bis 100 Euro wird die Steuer als Einkommen * 5 Prozent berechnet; über 100 Euro wird die Steuer als Einkommen * 7 Prozent berechnet."
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if taxable_income <= tariff_boundary:
            taxable_income * lower_tariff_rate_percent / 100
          else:
            taxable_income * upper_tariff_rate_percent / 100
"""


def _narrative_piecewise_test(
    name: str,
    taxable_income: int,
    expected_tax: int,
) -> dict[str, object]:
    return {
        "name": name,
        "period": "2026",
        "input": {"taxable_income": taxable_income},
        "output": {
            "de:statutes/estg/32a#tariff_income_tax_amount": expected_tax,
        },
    }


def test_narrative_piecewise_formula_requires_every_clause_and_boundary():
    lower_only = _narrative_piecewise_test("lower zone", 90, 4)
    lower_only["covers"] = ["de/statute/estg/32a(1)"]

    result = _analyze(
        NARRATIVE_PIECEWISE_CONTENT,
        NARRATIVE_PIECEWISE_SOURCE,
        test_cases=[lower_only],
    )

    assert _has_issue(result, "do not demonstrate", "formula branch")
    assert _has_issue(result, "boundary", "test")


def test_narrative_piecewise_zone_cases_still_require_exact_boundary():
    result = _analyze(
        NARRATIVE_PIECEWISE_CONTENT,
        NARRATIVE_PIECEWISE_SOURCE,
        test_cases=[
            _narrative_piecewise_test("lower zone", 90, 4),
            _narrative_piecewise_test("upper zone", 110, 7),
        ],
    )

    assert not _has_issue(result, "do not demonstrate", "formula branch")
    assert _has_issue(result, "boundary", "test")


def test_narrative_piecewise_complete_zone_and_boundary_cases_pass():
    result = _analyze(
        NARRATIVE_PIECEWISE_CONTENT,
        NARRATIVE_PIECEWISE_SOURCE,
        test_cases=[
            _narrative_piecewise_test("lower zone", 90, 4),
            _narrative_piecewise_test("upper zone", 110, 7),
            _narrative_piecewise_test("exact boundary", 100, 5),
        ],
    )

    assert not result.issues


@pytest.mark.parametrize(
    ("test_cases", "expected_issue_term"),
    [
        (
            [
                case
                for case in COMPLETE_COMPANION_TESTS
                if case["input"]["taxable_income"] <= 100
            ],
            "formula branch",
        ),
        (
            [
                case
                for case in COMPLETE_COMPANION_TESTS
                if "exact tariff boundary" not in case["name"]
            ],
            "boundary",
        ),
        (
            [
                case
                for case in COMPLETE_COMPANION_TESTS
                if case["name"] != "exception applies"
            ],
            "exception",
        ),
        (
            [
                case
                for case in COMPLETE_COMPANION_TESTS
                if case["name"] != "rounding down to full euro"
            ],
            "rounding",
        ),
    ],
)
def test_missing_companion_coverage_fails(
    test_cases: list[dict[str, object]],
    expected_issue_term: str,
):
    result = _analyze(
        COMPANION_COVERAGE_CONTENT,
        COMPANION_COVERAGE_SOURCE,
        test_cases=test_cases,
    )

    assert _has_issue(result, expected_issue_term, "test")


def test_unrelated_numeric_input_cannot_cover_tariff_boundary():
    test_cases = [
        case
        for case in COMPLETE_COMPANION_TESTS
        if "exact tariff boundary" not in case["name"]
    ]
    test_cases.append(
        {
            **_companion_test("unrelated numeric input", 90, 4),
            "input": {
                "taxable_income": 90,
                "exception_applies": False,
                "unrelated_amount": 101,
            },
        }
    )

    result = _analyze(
        COMPANION_COVERAGE_CONTENT,
        COMPANION_COVERAGE_SOURCE,
        test_cases=test_cases,
    )

    assert _has_issue(result, "boundary", "test")


def test_unrelated_boolean_toggle_cannot_cover_source_exception():
    content_with_unrelated_formula_gate = COMPANION_COVERAGE_CONTENT.replace(
        "          else:\n            floor(\n",
        "          else:\n            if unrelated_toggle:\n              floor(\n",
    )
    test_cases = [
        case
        for case in COMPLETE_COMPANION_TESTS
        if case["name"] != "exception applies"
    ]
    for value in (False, True):
        test_cases.append(
            {
                **_companion_test(f"unrelated toggle {value}", 90, 4),
                "input": {
                    "taxable_income": 90,
                    "exception_applies": False,
                    "unrelated_toggle": value,
                },
            }
        )

    result = _analyze(
        content_with_unrelated_formula_gate,
        COMPANION_COVERAGE_SOURCE,
        test_cases=test_cases,
    )

    assert _has_issue(result, "exception", "test")


def test_exception_toggle_from_another_source_branch_cannot_cover_exception():
    source = (
        COMPANION_COVERAGE_SOURCE
        + "(2) Der Sonderbetrag wird als Einkommen + Einkommen berechnet.\n"
    )
    content = COMPANION_COVERAGE_CONTENT + """\
  - name: special_status_amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if special_status_applies:
            0
          else:
            taxable_income
"""
    test_cases = [
        case
        for case in COMPLETE_COMPANION_TESTS
        if case["name"] != "exception applies"
    ]
    for value, expected in ((False, 90), (True, 0)):
        test_cases.append(
            {
                "name": f"other branch status {value}",
                "period": "2026",
                "input": {
                    "taxable_income": 90,
                    "exception_applies": False,
                    "special_status_applies": value,
                },
                "output": {
                    "de:statutes/estg/32a#special_status_amount": expected,
                },
            }
        )

    result = _analyze(content, source, test_cases=test_cases)

    assert _has_issue(result, "exception", "test")


def test_unrelated_fractional_input_cannot_cover_rounding_rule():
    content_with_unrelated_formula_amount = COMPANION_COVERAGE_CONTENT.replace(
        "            )\n",
        "            ) + unrelated_amount * 0\n",
    )
    test_cases = [
        case
        for case in COMPLETE_COMPANION_TESTS
        if case["name"] != "rounding down to full euro"
    ]
    test_cases.append(
        {
            **_companion_test("unrelated fractional input", 90, 4),
            "input": {
                "taxable_income": 90,
                "exception_applies": False,
                "unrelated_amount": 90.5,
            },
        }
    )

    result = _analyze(
        content_with_unrelated_formula_amount,
        COMPANION_COVERAGE_SOURCE,
        test_cases=test_cases,
    )

    assert _has_issue(result, "rounding", "test")


def test_nested_whitespace_rounding_operand_accepts_fractional_input():
    content = COMPANION_COVERAGE_CONTENT.replace(
        "            floor(\n",
        "            floor (\n              max(0,\n",
    ).replace(
        "            )\n",
        "              )\n            )\n",
    )

    result = _analyze(
        content,
        COMPANION_COVERAGE_SOURCE,
        test_cases=COMPLETE_COMPANION_TESTS,
    )

    assert not _has_issue(result, "rounding", "test")


MULTI_PARAGRAPH_FORMULA_SOURCE = """\
(1) Der erste Betrag ist Einkommen * 2.
(2) Der zweite Betrag ist Einkommen * 3.
"""

MULTI_PARAGRAPH_FORMULA_CONTENT = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: first_multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: second_multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: 3
  - name: combined_amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1); de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: income * first_multiplier + income * second_multiplier
"""


def _combined_formula_test(name: str, income: int) -> dict[str, object]:
    return {
        "name": name,
        "period": "2026",
        "input": {"income": income},
        "output": {"de:statutes/estg/32a#combined_amount": income * 5},
    }


def test_each_paragraph_formula_requires_distinct_executed_case():
    one_case = _combined_formula_test("one execution", 10)
    one_case["covers"] = [
        "de/statute/estg/32a(1)",
        "de/statute/estg/32a(2)",
    ]

    result = _analyze(
        MULTI_PARAGRAPH_FORMULA_CONTENT,
        MULTI_PARAGRAPH_FORMULA_SOURCE,
        test_cases=[one_case],
    )

    assert _has_issue(result, "formula branch", "distinct")


def test_distinct_cases_cover_distinct_paragraph_formulas():
    result = _analyze(
        MULTI_PARAGRAPH_FORMULA_CONTENT,
        MULTI_PARAGRAPH_FORMULA_SOURCE,
        test_cases=[
            _combined_formula_test("first execution", 10),
            _combined_formula_test("second execution", 20),
        ],
    )

    assert not _has_issue(result, "formula branch")


def test_non_range_formula_branches_require_distinct_selector_executions():
    source = """\
(1) Bei Ehegatten ist der Betrag Einkommen * 2.
(2) Bei Alleinstehenden ist der Betrag Einkommen * 3.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: married_multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: single_multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: 3
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1); de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if married or high_income:
            income * married_multiplier
          else:
            income * single_multiplier
"""

    def case(
        name: str,
        *,
        married: bool,
        high_income: bool,
        income: int,
    ) -> dict[str, object]:
        return {
            "name": name,
            "input": {
                "married": married,
                "high_income": high_income,
                "income": income,
            },
            "output": {
                "amount": income * (2 if married or high_income else 3)
            },
        }

    repeated_married = _analyze(
        content,
        source,
        test_cases=[
            case(
                "married ordinary",
                married=True,
                high_income=False,
                income=10,
            ),
            case(
                "married high income",
                married=True,
                high_income=True,
                income=20,
            ),
        ],
    )
    both_branches = _analyze(
        content,
        source,
        test_cases=[
            case("married", married=True, high_income=False, income=10),
            case("single", married=False, high_income=False, income=10),
        ],
    )

    assert _has_issue(repeated_married, "formula branch", "distinct")
    assert not _has_issue(both_branches, "formula branch")


@pytest.mark.parametrize("inline", [False, True])
def test_match_formula_branches_require_distinct_arm_executions(inline: bool):
    source = """\
(1) Bei Verheirateten ist der Betrag Einkommen * 2.
(2) Bei Alleinstehenden ist der Betrag Einkommen * 3.
"""
    match_formula = (
        'match filing_status: "married" => income * married_multiplier; '
        '"single" => income * single_multiplier'
        if inline
        else """\
match filing_status:
  "married" => income * married_multiplier
  "single" => income * single_multiplier"""
    )
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: married_multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: single_multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: 3
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1); de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
{chr(10).join(f"          {line}" for line in match_formula.splitlines())}
"""

    def case(name: str, status: str, income: int) -> dict[str, object]:
        return {
            "name": name,
            "input": {"filing_status": status, "income": income},
            "output": {"amount": income * (2 if status == "married" else 3)},
        }

    repeated_arm = _analyze(
        content,
        source,
        test_cases=[
            case("married one", "married", 10),
            case("married two", "married", 20),
        ],
    )
    both_arms = _analyze(
        content,
        source,
        test_cases=[
            case("married", "married", 10),
            case("single", "single", 10),
        ],
    )

    assert _has_issue(repeated_arm, "formula branch", "distinct")
    assert not _has_issue(both_arms, "formula branch")


def test_inline_if_formula_requires_distinct_runtime_branch_executions():
    source = """\
(1) Bei Ehegatten ist der Betrag Einkommen * 2.
(2) Bei Alleinstehenden ist der Betrag Einkommen * 3.
"""
    content = MULTI_PARAGRAPH_FORMULA_CONTENT.replace(
        "formula: income * first_multiplier + income * second_multiplier",
        "formula: >-\n"
        "          if married or high_income: income * first_multiplier "
        "else: income * second_multiplier",
    ).replace(
        "first_multiplier",
        "married_multiplier",
    ).replace(
        "second_multiplier",
        "single_multiplier",
    )

    def case(name: str, married: bool, high_income: bool) -> dict[str, object]:
        return {
            "name": name,
            "input": {
                "income": 10,
                "married": married,
                "high_income": high_income,
            },
            "output": {"combined_amount": 20 if married or high_income else 30},
        }

    repeated_branch = _analyze(
        content,
        source,
        test_cases=[
            case("married ordinary", True, False),
            case("married high income", True, True),
        ],
    )
    both_branches = _analyze(
        content,
        source,
        test_cases=[
            case("married", True, False),
            case("single", False, False),
        ],
    )

    assert _has_issue(repeated_branch, "formula branch", "distinct")
    assert not _has_issue(both_branches, "formula branch")


def test_judgment_selector_normalizes_holds_and_not_holds():
    rule = {
        "versions": [
            {
                "formula": "if eligible: eligible_amount else: ineligible_amount",
            }
        ]
    }

    assert completeness_module._case_formula_branch_outcome(
        rule,
        {"input": {"eligible": "holds"}},
    ) == "if:0"
    assert completeness_module._case_formula_branch_outcome(
        rule,
        {"input": {"eligible": "not_holds"}},
    ) == "if:1"


def test_inline_elif_formula_reports_each_reachable_branch():
    rule = {
        "versions": [
            {
                "formula": "if first: A elif second: B else: C",
            }
        ]
    }

    assert completeness_module._case_formula_branch_outcome(
        rule,
        {"input": {"first": True, "second": False}},
    ) == "if:0"
    assert completeness_module._case_formula_branch_outcome(
        rule,
        {"input": {"first": False, "second": True}},
    ) == "if:1"
    assert completeness_module._case_formula_branch_outcome(
        rule,
        {"input": {"first": False, "second": False}},
    ) == "if:2"


def test_formula_execution_fails_closed_on_non_boolean_guard_and_alias_conflict():
    rule = {
        "versions": [
            {
                "formula": "if enabled: enabled_amount else: disabled_amount",
            }
        ]
    }

    assert (
        completeness_module._case_formula_branch_outcome(
            rule,
            {"input": {"enabled": 1}},
        )
        is None
    )
    assert (
        completeness_module._case_formula_branch_outcome(
            rule,
            {"input": {"enabled": True, "other#enabled": False}},
        )
        is None
    )


def test_match_uses_last_arm_as_runtime_fallback():
    rule = {
        "versions": [
            {
                "formula": (
                    'match filing_status: "married" => joint_amount; '
                    '"single" => single_amount'
                ),
            }
        ]
    }

    assert completeness_module._case_formula_branch_outcome(
        rule,
        {"input": {"filing_status": "unknown"}},
    ) == "match:1"


def test_quoted_control_text_does_not_confuse_formula_execution():
    rule = {
        "versions": [
            {
                "formula": (
                    'if code == "if x: y else: z": selected_amount '
                    "else: other_amount"
                ),
            }
        ]
    }

    assert completeness_module._case_formula_branch_outcome(
        rule,
        {"input": {"code": "if x: y else: z"}},
    ) == "if:0"


def test_formula_execution_rejects_ambiguous_versions():
    rule = {
        "versions": [
            {"formula": "if enabled: first_amount else: zero_amount"},
            {"formula": "if enabled: second_amount else: zero_amount"},
        ]
    }

    assert (
        completeness_module._case_formula_branch_outcome(
            rule,
            {"input": {"enabled": True}},
        )
        is None
    )


def test_nested_match_arms_do_not_count_when_outer_guard_bypasses_them():
    source = """\
(1) Bei Verheirateten ist der Betrag Einkommen * 2.
(2) Bei Alleinstehenden ist der Betrag Einkommen * 3.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: married_multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: single_multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: 3
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1); de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if disabled:
            0
          else:
            match filing_status:
              "married" => income * married_multiplier
              "single" => income * single_multiplier
"""

    def case(
        name: str,
        *,
        disabled: bool,
        filing_status: str,
    ) -> dict[str, object]:
        multiplier = 2 if filing_status == "married" else 3
        return {
            "name": name,
            "input": {
                "disabled": disabled,
                "filing_status": filing_status,
                "income": 10,
            },
            "output": {"amount": 0 if disabled else 10 * multiplier},
        }

    bypassed_match = _analyze(
        content,
        source,
        test_cases=[
            case("disabled married", disabled=True, filing_status="married"),
            case("disabled single", disabled=True, filing_status="single"),
        ],
    )
    executed_match = _analyze(
        content,
        source,
        test_cases=[
            case("married", disabled=False, filing_status="married"),
            case("single", disabled=False, filing_status="single"),
        ],
    )
    nested_guard_content = content.replace(
        """\
          if disabled:
            0
          else:
""",
        """\
          if disabled:
            if audited:
              0
            else:
              0
          else:
""",
    )
    executed_match_below_nested_guard = _analyze(
        nested_guard_content,
        source,
        test_cases=[
            {
                **case("nested married", disabled=False, filing_status="married"),
                "input": {
                    "disabled": False,
                    "audited": False,
                    "filing_status": "married",
                    "income": 10,
                },
            },
            {
                **case("nested single", disabled=False, filing_status="single"),
                "input": {
                    "disabled": False,
                    "audited": False,
                    "filing_status": "single",
                    "income": 10,
                },
            },
        ],
    )

    assert _has_issue(bypassed_match, "formula branch", "distinct")
    assert not _has_issue(executed_match, "formula branch")
    assert not _has_issue(executed_match_below_nested_guard, "formula branch")


def test_nested_if_bypass_cannot_witness_arithmetic_branches():
    content = MULTI_PARAGRAPH_FORMULA_CONTENT.replace(
        "formula: income * first_multiplier + income * second_multiplier",
        """\
formula: |-
          if disabled:
            0
          else:
            if married:
              income * first_multiplier
            else:
              income * second_multiplier""",
    )

    def case(
        name: str,
        *,
        disabled: bool,
        married: bool,
    ) -> dict[str, object]:
        expected = 0 if disabled else 10 * (2 if married else 3)
        return {
            "name": name,
            "input": {
                "disabled": disabled,
                "married": married,
                "income": 10,
            },
            "output": {"combined_amount": expected},
        }

    bypass_plus_one_branch = _analyze(
        content,
        MULTI_PARAGRAPH_FORMULA_SOURCE,
        test_cases=[
            case("disabled single", disabled=True, married=False),
            case("enabled married", disabled=False, married=True),
        ],
    )
    both_arithmetic_branches = _analyze(
        content,
        MULTI_PARAGRAPH_FORMULA_SOURCE,
        test_cases=[
            case("enabled married", disabled=False, married=True),
            case("enabled single", disabled=False, married=False),
        ],
    )

    assert _has_issue(bypass_plus_one_branch, "formula branch", "distinct")
    assert not _has_issue(both_arithmetic_branches, "formula branch")


def test_match_arm_bypasses_cannot_witness_arithmetic_branches():
    content = MULTI_PARAGRAPH_FORMULA_CONTENT.replace(
        "formula: income * first_multiplier + income * second_multiplier",
        """\
formula: |-
          match filing_status:
            "married" => if disabled: 0 else: income * first_multiplier
            "single" => if disabled: 0 else: income * second_multiplier""",
    )

    def case(
        status: str,
        *,
        disabled: bool,
    ) -> dict[str, object]:
        expected = 0 if disabled else 10 * (2 if status == "married" else 3)
        return {
            "name": f"{status} disabled={disabled}",
            "input": {
                "disabled": disabled,
                "filing_status": status,
                "income": 10,
            },
            "output": {"combined_amount": expected},
        }

    bypassed_arms = _analyze(
        content,
        MULTI_PARAGRAPH_FORMULA_SOURCE,
        test_cases=[
            case("married", disabled=True),
            case("single", disabled=True),
        ],
    )
    executed_arms = _analyze(
        content,
        MULTI_PARAGRAPH_FORMULA_SOURCE,
        test_cases=[
            case("married", disabled=False),
            case("single", disabled=False),
        ],
    )

    assert _has_issue(bypassed_arms, "formula branch", "distinct")
    assert not _has_issue(executed_arms, "formula branch")


def test_interval_cases_must_reach_the_piecewise_formula():
    content = NARRATIVE_PIECEWISE_CONTENT.replace(
        """\
          if taxable_income <= tariff_boundary:
            taxable_income * lower_tariff_rate_percent / 100
          else:
            taxable_income * upper_tariff_rate_percent / 100""",
        """\
          if disabled:
            0
          else:
            if taxable_income <= tariff_boundary:
              taxable_income * lower_tariff_rate_percent / 100
            else:
              taxable_income * upper_tariff_rate_percent / 100""",
    )

    def case(name: str, income: int, *, disabled: bool) -> dict[str, object]:
        expected = (
            0
            if disabled
            else int(income * (5 if income <= 100 else 7) / 100)
        )
        return {
            "name": name,
            "input": {"disabled": disabled, "taxable_income": income},
            "output": {"tariff_income_tax_amount": expected},
        }

    bypassed_intervals = _analyze(
        content,
        NARRATIVE_PIECEWISE_SOURCE,
        test_cases=[
            case("disabled lower", 90, disabled=True),
            case("disabled upper", 110, disabled=True),
            case("disabled boundary", 100, disabled=True),
        ],
    )
    executed_intervals = _analyze(
        content,
        NARRATIVE_PIECEWISE_SOURCE,
        test_cases=[
            case("lower", 90, disabled=False),
            case("upper", 110, disabled=False),
            case("boundary", 100, disabled=False),
        ],
    )

    assert _has_issue(bypassed_intervals, "formula branch", "distinct")
    assert _has_issue(bypassed_intervals, "boundary", "test")
    assert not executed_intervals.issues


def test_exception_toggle_must_be_reached_by_both_cases():
    source = """\
(1) Der Betrag wird als Einkommen * 2 berechnet. Ausnahme: Bei einer Befreiung beträgt der Betrag null.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if disabled:
            0
          else:
            if exemption_applies:
              0
            else:
              income * multiplier
"""

    def case(
        name: str,
        *,
        disabled: bool,
        exemption_applies: bool,
    ) -> dict[str, object]:
        expected = 0 if disabled or exemption_applies else 20
        return {
            "name": name,
            "input": {
                "disabled": disabled,
                "exemption_applies": exemption_applies,
                "income": 10,
            },
            "output": {"amount": expected},
        }

    unreachable_toggle = _analyze(
        content,
        source,
        test_cases=[
            case("disabled ordinary", disabled=True, exemption_applies=False),
            case("disabled exempt", disabled=True, exemption_applies=True),
        ],
    )
    reached_toggle = _analyze(
        content,
        source,
        test_cases=[
            case("ordinary", disabled=False, exemption_applies=False),
            case("exempt", disabled=False, exemption_applies=True),
        ],
    )

    assert _has_issue(unreachable_toggle, "exception", "test")
    assert not reached_toggle.issues


def test_rounding_witness_must_reach_the_rounding_operator():
    source = """\
(1) Der Betrag wird als Einkommen * 2 berechnet und auf volle Euro abzurunden.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if disabled:
            0
          else:
            floor(income * multiplier)
"""

    def case(name: str, *, disabled: bool) -> dict[str, object]:
        return {
            "name": name,
            "input": {"disabled": disabled, "income": 10.25},
            "output": {"amount": 0 if disabled else 20},
        }

    bypassed_rounding = _analyze(
        content,
        source,
        test_cases=[case("disabled fractional", disabled=True)],
    )
    executed_rounding = _analyze(
        content,
        source,
        test_cases=[case("enabled fractional", disabled=False)],
    )

    assert _has_issue(bypassed_rounding, "rounding", "fractional")
    assert not executed_rounding.issues


def test_numbered_prose_formulas_require_distinct_executed_cases():
    source = """\
(1) Es gelten folgende Berechnungen:
1. Der erste Betrag wird als Einkommen * 2 berechnet;
2. Der zweite Betrag wird als Einkommen * 3 berechnet.
"""
    content = MULTI_PARAGRAPH_FORMULA_CONTENT.replace(
        "de/statute/estg/32a(2)",
        "de/statute/estg/32a(1)(2)",
    ).replace(
        "de/statute/estg/32a(1); de/statute/estg/32a(1)(2)",
        "de/statute/estg/32a(1)(1); de/statute/estg/32a(1)(2)",
    )

    result = _analyze(
        content,
        source,
        test_cases=[_combined_formula_test("one execution", 10)],
    )

    assert _has_issue(result, "formula branch", "distinct")


@pytest.mark.parametrize(
    "exception_text",
    [
        "Die Regel gilt nicht, wenn eine Befreiung vorliegt.",
        "Die Regel findet keine Anwendung, wenn eine Befreiung vorliegt.",
        "Die Regel gilt, soweit nicht eine Befreiung vorliegt.",
    ],
)
def test_common_german_exceptions_require_paired_cases(exception_text: str):
    source = (
        "(1) Der Betrag wird als Einkommen * 2 berechnet. " + exception_text
    )
    one_sided_tests = [
        case
        for case in COMPLETE_COMPANION_TESTS
        if case["name"] != "exception applies"
    ]

    result = _analyze(
        COMPANION_COVERAGE_CONTENT,
        source,
        test_cases=one_sided_tests,
    )

    assert _has_issue(result, "exception", "test")


def test_glued_german_satz_exception_requires_paired_cases():
    source = """\
(6) 1Der Betrag wird aus dem Einkommen berechnet.2Voraussetzung für die Anwendung ist, dass die Person nicht befreit ist.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(6) Satz 1; de/statute/estg/32a(6) Satz 2
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if exception_applies:
            0
          else:
            income
"""
    test_cases = [
        {
            "name": "ordinary case",
            "period": "2026",
            "input": {"income": 10, "exception_applies": False},
            "output": {"de:statutes/estg/32a#amount": 10},
        }
    ]

    result = _analyze(content, source, test_cases=test_cases)

    assert _has_issue(result, "exception", "test")


def test_rounding_fractional_evidence_is_bound_to_affected_branch():
    source = """\
(1) Der erste Betrag ist Einkommen * 2 und auf volle Euro abzurunden.
(2) Der zweite Betrag ist Einkommen * 3 und auf volle Euro abzurunden.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: first_multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: second_multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: 3
  - name: first_amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: floor(income * first_multiplier)
  - name: second_amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: floor(income * second_multiplier)
"""
    tests = [
        {
            "name": "first fractional execution",
            "period": "2026",
            "input": {"income": 10.25},
            "output": {"de:statutes/estg/32a#first_amount": 20},
        },
        {
            "name": "another first fractional execution",
            "period": "2026",
            "input": {"income": 11.25},
            "output": {"de:statutes/estg/32a#first_amount": 22},
        },
        {
            "name": "second integer execution",
            "period": "2026",
            "input": {"income": 12},
            "output": {"de:statutes/estg/32a#second_amount": 36},
        },
    ]

    result = _analyze(content, source, test_cases=tests)

    assert _has_issue(result, "rounding", "fractional")


def test_same_computation_leaf_cannot_witness_two_source_formulas():
    content = MULTI_PARAGRAPH_FORMULA_CONTENT.replace(
        "formula: income * first_multiplier + income * second_multiplier",
        """\
formula: |-
          if disabled:
            income * first_multiplier
          else:
            if married:
              income * first_multiplier
            else:
              income * second_multiplier""",
    )
    cases = [
        {
            "name": "outer first computation",
            "input": {"disabled": True, "married": True, "income": 10},
            "output": {"combined_amount": 20},
        },
        {
            "name": "nested first computation",
            "input": {"disabled": False, "married": True, "income": 10},
            "output": {"combined_amount": 20},
        },
    ]

    result = _analyze(
        content,
        MULTI_PARAGRAPH_FORMULA_SOURCE,
        test_cases=cases,
    )

    assert _has_issue(result, "formula branch", "distinct")


def test_commutative_duplicate_cannot_witness_different_source_operations():
    source = """\
(1) Der erste Betrag ist die Summe aus Einkommen und Zuschlag.
(2) Der zweite Betrag ist der Unterschied zwischen Einkommen und Zuschlag.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1); de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if first_order:
            income + supplement
          else:
            supplement + income
"""
    cases = [
        {
            "name": f"sum order {first_order}",
            "input": {
                "first_order": first_order,
                "income": 10,
                "supplement": 3,
            },
            "output": {"amount": 13},
        }
        for first_order in (False, True)
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "formula branch", "distinct")


@pytest.mark.parametrize(
    ("bypass_formula", "extra_input"),
    [
        ("0 + 0", {}),
        ("min(0, 0)", {}),
        ("disabled_amount", {"disabled_amount": 0}),
    ],
)
def test_zero_valued_bypass_expression_is_not_computation_evidence(
    bypass_formula: str,
    extra_input: dict[str, object],
):
    source = "(1) Der Betrag wird als Einkommen * 2 berechnet."
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if disabled:
            {bypass_formula}
          else:
            income * multiplier
"""
    case = {
        "name": "bypass",
        "input": {"disabled": True, "income": 10, **extra_input},
        "output": {"amount": 0},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "formula branch", "distinct")


def test_boundary_evidence_is_bound_to_reached_comparator_and_input():
    source = """\
(1) Der Anspruch besteht für berechtigte Personen.
Er gilt bis 100 Euro Einkommen.
"""
    base_content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: income_limit
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 100
  - name: eligible
    kind: derived
    dtype: bool
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: income <= income_limit
"""
    continuation_missing = _analyze(
        base_content,
        source,
        test_cases=[
            {
                "name": "below only",
                "input": {"income": 50},
                "output": {"eligible": True},
            }
        ],
    )
    bypass_content = base_content.replace(
        "formula: income <= income_limit",
        (
            "formula: >-\n"
            "          if disabled: income == income "
            "else: income <= income_limit"
        ),
    )
    bypassed_comparator = _analyze(
        bypass_content,
        source,
        test_cases=[
            {
                "name": "exact but bypassed",
                "input": {"disabled": True, "income": 100},
                "output": {"eligible": True},
            }
        ],
    )
    other_input_content = base_content.replace(
        "formula: income <= income_limit",
        (
            "formula: >-\n"
            "          if use_age: age <= external_age_limit "
            "else: income <= income_limit"
        ),
    )
    unused_boundary_input = _analyze(
        other_input_content,
        source,
        test_cases=[
            {
                "name": "unused exact income",
                "input": {
                    "use_age": True,
                    "age": 18,
                    "external_age_limit": 21,
                    "income": 100,
                },
                "output": {"eligible": True},
            }
        ],
    )

    assert _has_issue(continuation_missing, "boundary", "test")
    assert _has_issue(bypassed_comparator, "boundary", "test")
    assert _has_issue(unused_boundary_input, "boundary", "test")


def test_exception_toggle_must_change_the_reached_formula_effect():
    source = "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt."
    short_circuited_content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: eligible
    kind: derived
    dtype: bool
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if disabled or exemption_applies:
            false
          else:
            true
"""
    short_circuited_cases = [
        {
            "name": f"disabled exemption={value}",
            "input": {"disabled": True, "exemption_applies": value},
            "output": {"eligible": False},
        }
        for value in (False, True)
    ]
    short_circuited = _analyze(
        short_circuited_content,
        source,
        test_cases=short_circuited_cases,
    )

    direct_relation_content = short_circuited_content.replace(
        """\
formula: |-
          if disabled or exemption_applies:
            false
          else:
            true""",
        "formula: not exemption_applies",
    )
    direct_relation_cases = [
        {
            "name": f"direct exemption={value}",
            "input": {"exemption_applies": value},
            "output": {"eligible": not value},
        }
        for value in (False, True)
    ]
    direct_relation = _analyze(
        direct_relation_content,
        source,
        test_cases=direct_relation_cases,
    )

    assert _has_issue(short_circuited, "exception", "test")
    assert not direct_relation.issues


def test_rounding_fraction_must_belong_to_reached_operand():
    source = """\
(1) Der Betrag wird als Einkommen * 2 berechnet und auf volle Euro abzurunden.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if use_other:
            floor(other_amount)
          else:
            floor(income * multiplier)
"""
    case = {
        "name": "unused fractional operand",
        "input": {
            "use_other": False,
            "other_amount": 10.5,
            "income": 10,
        },
        "output": {"amount": 20},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "rounding", "fractional")


@pytest.mark.parametrize(
    "source",
    [
        "(1) Das Einkommen ist mit 2 zu multiplizieren.",
        "(1) Das Einkommen ist mit dem Faktor 2 zu vervielfachen.",
        "(1) Das Einkommen ist mit dem Faktor von 2 zu multiplizieren.",
        "(1) Das Einkommen ist mit einem Faktor von 2 zu vervielfachen.",
        "(1) Das Einkommen ist durch Multiplikation mit Faktor 2 zu ermitteln.",
        "(1) Das Einkommen ist durch Multiplikation mit dem Faktor 2 zu ermitteln.",
        "(1) Der Betrag ist unter Anwendung des Faktors 2 zu ermitteln.",
        "(1) Das Einkommen ist zu verdoppeln.",
        "(1) Der Betrag ist durch zwei zu teilen.",
        "(1) Der Betrag ist um 2 zu erhöhen.",
        "(1) Der Betrag ist um 2 zu vermindern.",
        "(1) Der Betrag ist aus Einkommen und Zuschlag zu summieren.",
    ],
)
def test_german_infinitive_formula_wording_rejects_parameter_only(source: str):
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
"""

    result = _analyze(content, source, test_cases=[])

    assert source_states_explicit_computation(source)
    assert _has_issue(result, "formula-output", "parameter-only")


def test_nonbranching_formula_must_execute_the_source_computation():
    source = "(1) Der Betrag wird als Einkommen * 2 berechnet."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: unrelated_amount
"""
    case = {
        "name": "unrelated output",
        "period": "2026",
        "input": {"income": 10, "unrelated_amount": 2},
        "output": {"amount": 2},
    }

    unrelated = _analyze(content, source, test_cases=[case])
    correct = _analyze(
        content.replace(
            "formula: unrelated_amount",
            "formula: income * multiplier",
        ),
        source,
        test_cases=[
            {
                **case,
                "name": "operative output",
                "output": {"amount": 20},
            }
        ],
    )

    assert _has_issue(unrelated, "formula branch")
    assert not correct.issues


def test_indexed_expression_multiplied_by_zero_cannot_witness_formula():
    source = "(1) Der Betrag wird als Einkommen * 2 berechnet."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: neutralizer
    kind: parameter
    dtype: Decimal
    indexed_by: Integer
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 1
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if disabled:
            0 * neutralizer[index] * multiplier
          else:
            income * multiplier
"""
    case = {
        "name": "disabled zero",
        "period": "2026",
        "input": {"disabled": True, "index": 1, "income": 10},
        "output": {"amount": 0},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "formula branch")


def test_named_match_pattern_uses_constant_value_not_identifier_text():
    source = """\
(1) Bei Ehegatten ist der Betrag Einkommen * 2.
(2) Bei Alleinstehenden ist der Betrag Einkommen * 3.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: special_status
    kind: parameter
    dtype: String
    versions:
      - effective_from: '2026-01-01'
        formula: '"married"'
  - name: first_multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: second_multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: 3
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1); de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: >-
          match status: special_status => income * first_multiplier;
          "single" => income * second_multiplier
"""
    cases = [
        {
            "name": status,
            "input": {"status": status, "income": 10},
            "output": {"amount": 30},
        }
        for status in ("special_status", "other")
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "formula branch", "distinct")


def test_bare_enum_match_patterns_remain_literal_values():
    rule = {
        "versions": [
            {
                "effective_from": "2026-01-01",
                "formula": (
                    "match status: married => income * 2; "
                    "single => income * 3"
                ),
            }
        ]
    }

    married = completeness_module._case_formula_branch_outcome(
        rule,
        {"input": {"status": "married", "income": 10}},
    )
    single = completeness_module._case_formula_branch_outcome(
        rule,
        {"input": {"status": "single", "income": 10}},
    )

    assert married == "match:0"
    assert single == "match:1"


def test_neutral_terms_cannot_bind_or_distinguish_formula_branches():
    source = """\
(1) Der erste Betrag ist Einkommen + 2.
(2) Der zweite Betrag ist Einkommen + 3.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: two
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: three
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: 3
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1); de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if first:
            income + two
          else:
            income + two + 0 * three
"""
    cases = [
        {
            "name": str(first),
            "input": {"first": first, "income": 10},
            "output": {"amount": 12},
        }
        for first in (True, False)
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "formula branch", "distinct")


@pytest.mark.parametrize(
    ("source", "formula", "expected"),
    [
        (
            "(1) Der Betrag ist die Hälfte des Einkommens.",
            "income * half_factor",
            5,
        ),
        (
            "(1) Der Betrag ist das Doppelte des Einkommens.",
            "income + income",
            20,
        ),
    ],
)
def test_common_algebraic_formula_equivalents_are_accepted(
    source: str,
    formula: str,
    expected: int,
):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: half_factor
    kind: parameter
    dtype: Decimal
    versions:
      - effective_from: '2026-01-01'
        formula: 0.5
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if applies:
            {formula}
          else:
            0
"""
    case = {
        "name": "applies",
        "input": {"applies": True, "income": 10},
        "output": {"amount": expected},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues


def test_case_period_selects_nonbranching_temporal_formula():
    source = "(1) Der Betrag wird als Einkommen * 2 berechnet."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: multiplier
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2025-01-01'
        formula: 1.5
      - effective_from: '2026-01-01'
        formula: 2
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2025-01-01'
        formula: unrelated_amount
      - effective_from: '2026-01-01'
        formula: income * multiplier
"""
    case = {
        "name": "current formula",
        "period": "2026",
        "input": {"income": 10, "unrelated_amount": 99},
        "output": {"amount": 20},
    }

    result = _analyze(content, source, test_cases=[case])
    pipeline_shaped = _analyze(
        content,
        source,
        test_cases=[case],
        artifact_numeric_values=(1.5, 2.0),
        artifact_numeric_bindings=(
            ("multiplier", 1.5),
            ("multiplier", 2.0),
        ),
    )

    assert not result.issues
    assert not pipeline_shaped.issues


def _boundary_control_content(
    *,
    formula: str,
    limit_versions: str,
    extra_rules: str = "",
) -> str:
    return f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
{extra_rules}  - name: income_limit
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
{limit_versions}
  - name: eligible
    kind: derived
    dtype: bool
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: >-
          {formula}
"""


def test_boundary_requires_one_operative_input_threshold_comparison():
    source = "(1) Die Regel gilt für Einkommen bis 100 Euro."
    limit_versions = """\
      - effective_from: '2026-01-01'
        formula: 100"""
    inert = _boundary_control_content(
        formula="if income > 0: income_limit > 0 else: false",
        limit_versions=limit_versions,
    )
    operative = _boundary_control_content(
        formula="income <= income_limit",
        limit_versions=limit_versions,
    )
    case = {
        "name": "at threshold",
        "period": "2026-01",
        "input": {"income": 100},
        "output": {"eligible": True},
    }

    inert_result = _analyze(inert, source, test_cases=[case])
    operative_result = _analyze(operative, source, test_cases=[case])

    assert _has_issue(inert_result, "boundary", "100")
    assert not operative_result.issues


def test_adjacent_integral_boundary_requires_the_equivalent_comparator():
    source = "(1) Die Regel gilt für Einkommen bis 100 Euro."
    source_limit_rule = """\
  - name: source_limit
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 100
"""
    limit_versions = """\
      - effective_from: '2026-01-01'
        formula: 101"""
    correct = _boundary_control_content(
        formula="income < income_limit",
        limit_versions=limit_versions,
        extra_rules=source_limit_rule,
    )
    wrong = _boundary_control_content(
        formula="income <= income_limit",
        limit_versions=limit_versions,
        extra_rules=source_limit_rule,
    )
    case = {
        "name": "inclusive source endpoint",
        "period": "2026",
        "input": {"income": 100},
        "output": {"eligible": True},
    }

    correct_result = _analyze(correct, source, test_cases=[case])
    wrong_result = _analyze(wrong, source, test_cases=[case])

    assert not correct_result.issues
    assert _has_issue(wrong_result, "boundary", "100")


@pytest.mark.parametrize(
    ("wording", "formula"),
    [
        ("höchstens 100 Euro", "income <= income_limit"),
        ("mindestens 100 Euro", "income >= income_limit"),
    ],
)
def test_german_minimum_and_maximum_boundaries_are_controlled(
    wording: str,
    formula: str,
):
    source = f"(1) Die Regel gilt für Einkommen von {wording}."
    content = _boundary_control_content(
        formula=formula,
        limit_versions="""\
      - effective_from: '2026-01-01'
        formula: 100""",
    )
    case = {
        "name": "at threshold",
        "period": "2026",
        "input": {"income": 100},
        "output": {"eligible": True},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues


def test_monthly_case_period_resolves_versioned_boundary_constant():
    source = "(1) Die Regel gilt für Einkommen bis 100 Euro."
    content = _boundary_control_content(
        formula="income <= income_limit",
        limit_versions="""\
      - effective_from: '2025-01-01'
        formula: 90
      - effective_from: '2026-01-01'
        formula: 100""",
    )
    case = {
        "name": "current threshold",
        "period": "2026-01",
        "input": {"income": 100},
        "output": {"eligible": True},
    }

    result = _analyze(content, source, test_cases=[case])
    pipeline_shaped = _analyze(
        content,
        source,
        test_cases=[case],
        artifact_numeric_values=(90.0, 100.0),
        artifact_numeric_bindings=(
            ("income_limit", 90.0),
            ("income_limit", 100.0),
        ),
    )

    assert not result.issues
    assert not pipeline_shaped.issues


def test_equal_asserted_exception_effects_do_not_count_as_toggle():
    source = (
        "(1) Ein Anspruch besteht, es sei denn, eine Befreiung liegt vor."
    )
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: eligible
    kind: derived
    dtype: bool
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if exemption_applies:
            false
          else:
            0 == 1
"""
    same_effect_cases = [
        {
            "name": str(exemption_applies),
            "input": {"exemption_applies": exemption_applies},
            "output": {"eligible": False},
        }
        for exemption_applies in (False, True)
    ]
    proper_content = content.replace(
        """\
formula: |-
          if exemption_applies:
            false
          else:
            0 == 1""",
        "formula: not exemption_applies",
    )
    proper_cases = [
        {
            "name": str(exemption_applies),
            "input": {"exemption_applies": exemption_applies},
            "output": {"eligible": not exemption_applies},
        }
        for exemption_applies in (False, True)
    ]

    same_effect = _analyze(content, source, test_cases=same_effect_cases)
    proper_effect = _analyze(proper_content, source, test_cases=proper_cases)

    assert _has_issue(same_effect, "exception", "test")
    assert not proper_effect.issues


def test_division_cannot_be_witnessed_by_multiplying_by_the_divisor():
    source = "(1) Der Betrag ist Einkommen / 2."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: divisor
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: income * divisor
"""
    case = {
        "name": "wrong inverse",
        "period": "2026",
        "input": {"income": 10},
        "output": {"amount": 20},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "formula branch")


def test_zero_over_provably_nonzero_indexed_term_cannot_witness_division():
    source = "(1) Der Betrag ist Einkommen / 2."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: divisor
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: dummy_table
    kind: parameter
    dtype: Decimal
    indexed_by: Integer
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 10
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 0 / max(1, dummy_table[index]) / divisor
"""
    case = {
        "name": "zero bypass",
        "period": "2026",
        "input": {"income": 10, "index": 1},
        "output": {"amount": 0},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "formula branch")
