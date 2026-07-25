from __future__ import annotations

import pytest

from axiom_encode.harness.source_completeness import (
    analyze_complete_source_unit,
)
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CORPUS_CITATION_PATH = "de/statute/estg/32a"


def _analyze(
    content: str,
    authoritative_source_text: str,
    *,
    test_cases: list[object] | None = None,
):
    return analyze_complete_source_unit(
        content,
        authoritative_source_text,
        corpus_citation_path=CORPUS_CITATION_PATH,
        test_cases=test_cases,
        extract_numeric_occurrences=extract_numeric_occurrences_from_text,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _has_issue(result, *needles: str) -> bool:
    lowered_needles = tuple(needle.lower() for needle in needles)
    return any(
        all(needle in issue.lower() for needle in lowered_needles)
        for issue in result.issues
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
(914.51 * y + 1400) * y; y ist (Einkommen - 12348) / 10000.
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
    versions:
      - effective_from: '2026-01-01'
        formula: 12349
  - name: first_zone_end
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 17799
  - name: first_zone_quadratic_coefficient
    kind: parameter
    dtype: Decimal
    versions:
      - effective_from: '2026-01-01'
        formula: 914.51
  - name: first_zone_linear_coefficient
    kind: parameter
    dtype: Decimal
    versions:
      - effective_from: '2026-01-01'
        formula: 1400
  - name: basic_allowance
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 12348
  - name: tariff_zone_scale
    kind: parameter
    dtype: Decimal
    versions:
      - effective_from: '2026-01-01'
        formula: 10000
"""

    result = _analyze(constants_only, source, test_cases=[])

    assert result.source_numeric_occurrence_count == 6
    assert result.covered_source_numeric_occurrence_count == 6
    assert result.missing_source_numeric_occurrence_count == 0
    assert _has_issue(result, "principal", "derived/relation")


ABSATZ_1 = "(1) Die Steuer ist das Einkommen multipliziert mit 10 Prozent."
ABSATZ_5 = (
    "(5) Bei zusammen veranlagten Ehegatten ist die Steuer das Doppelte "
    "der Steuer auf die Hälfte ihres Einkommens."
)
ABSATZ_6 = (
    "(6) Das Splittingverfahren gilt für verwitwete Personen, wenn die "
    "Voraussetzungen vorliegen, es sei denn, sie haben wieder geheiratet."
)

ABSATZ_1_RULES = """\
  - name: tariff_rate
    kind: parameter
    dtype: Rate
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


def test_silently_omitting_absatz_6_fails():
    result = _analyze(
        _encoded_absatz_content(include_absatz_5=True),
        f"{ABSATZ_1}\n{ABSATZ_5}\n{ABSATZ_6}",
        test_cases=ENCODED_FORMULA_TESTS,
    )

    assert _has_issue(result, "(6)", "source branch")


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
    versions:
      - effective_from: '2026-01-01'
        formula: 5
  - name: upper_tariff_rate_percent
    kind: parameter
    dtype: Decimal
    versions:
      - effective_from: '2026-01-01'
        formula: 7
  - name: lower_tariff_maximum
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 100
  - name: upper_tariff_minimum
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 101
  - name: tariff_income_tax_amount
    kind: derived
    dtype: Money
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
    _companion_test("exception does not apply", 110, 7),
    _companion_test("exception applies", 110, 0, exception_applies=True),
    _companion_test("rounding down to full euro", 90.5, 4),
]


def test_complete_companion_suite_covers_source_controls():
    result = _analyze(
        COMPANION_COVERAGE_CONTENT,
        COMPANION_COVERAGE_SOURCE,
        test_cases=COMPLETE_COMPANION_TESTS,
    )

    assert not result.issues


@pytest.mark.parametrize(
    ("test_cases", "expected_issue_term"),
    [
        ([COMPLETE_COMPANION_TESTS[0]], "formula branch"),
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
