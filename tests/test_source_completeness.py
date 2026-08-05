from __future__ import annotations

import functools
import hashlib
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as completeness_module
from axiom_encode.harness import validator_pipeline as validator_pipeline_module
from axiom_encode.harness.source_completeness import (
    analyze_complete_source_unit,
    authoritative_numeric_recall_text,
    collect_artifact_numeric_bindings,
    collect_artifact_numeric_values,
    recognize_source_structure,
    source_states_explicit_computation,
    source_states_stated_conversion_result,
)
from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline,
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    find_ungrounded_numeric_issues,
    find_ungrounded_numeric_issues_scoped,
    numeric_value_is_grounded,
)

CORPUS_CITATION_PATH = "de/statute/estg/32a"
DE_NUMERIC_OCCURRENCE_EXTRACTOR = functools.partial(
    extract_typed_numeric_inventory_occurrences_from_text,
    profile="de-DE",
)
EN_NUMERIC_OCCURRENCE_EXTRACTOR = functools.partial(
    extract_typed_numeric_inventory_occurrences_from_text,
    profile="en-US",
)
EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR = functools.partial(
    extract_typed_numeric_occurrences_from_text,
    profile="en-US",
)


def _analyze(
    content: str,
    authoritative_source_text: str,
    *,
    corpus_citation_path: str = CORPUS_CITATION_PATH,
    test_cases: list[object] | None = None,
    artifact_numeric_values: tuple[float, ...] | None = None,
    artifact_numeric_bindings: tuple[tuple[str, float], ...] | None = None,
):
    return analyze_complete_source_unit(
        content,
        authoritative_source_text,
        corpus_citation_path=corpus_citation_path,
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
    corpus_citation_path: str = CORPUS_CITATION_PATH,
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
            corpus_citation_path: authoritative_source_text,
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


@pytest.mark.parametrize("corpus_dash", ("‐", "‑", "‒", "–", "—", "−"))
def test_ascii_rulespec_deferral_covers_unicode_dash_corpus_section(corpus_dash):
    citation_path = f"us/statute/42/1437c{corpus_dash}1"
    source = (
        "(a) The public housing agency shall submit a plan for assistance "
        "under section 1437f."
    )
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation_path}
  deferred_outputs:
    - output: us:statutes/42/1437c-1/a#public_housing_agency_plan
      reason: >-
        Subsection (a) cannot be encoded until assistance eligibility under
        section 1437f is available.
rules: []
"""

    result = _analyze(
        content,
        source,
        corpus_citation_path=citation_path,
        test_cases=[],
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


SVBEZGRV_2025_SECTION_1_BODY = (
    "Die Bezugsgröße nach § 18 Absatz 1 des Vierten Buches Sozialgesetzbuch "
    "für das Jahr 2025 beträgt 44 940 Euro. Umgerechnet auf den Monat ergeben "
    "sich 3 745 Euro."
)
SVBEZGRV_2025_SECTION_2_BODY = (
    "(1) Die Jahresarbeitsentgeltgrenze nach § 6 Absatz 6 des Fünften Buches "
    "Sozialgesetzbuch wird für das Jahr 2025 auf 73 800 Euro festgesetzt. "
    "Umgerechnet auf den Monat ergeben sich 6 150 Euro.\n"
    "(2) Die Jahresarbeitsentgeltgrenze nach § 6 Absatz 7 des Fünften Buches "
    "Sozialgesetzbuch wird für das Jahr 2025 auf 66 150 Euro festgesetzt. "
    "Umgerechnet auf den Monat ergeben sich 5 512,50 Euro."
)
MINUHV_SECTION_1_BODY = (
    "Der monatliche Mindestunterhalt minderjähriger Kinder gemäß § 1612a "
    "Absatz 1 des Bürgerlichen Gesetzbuchs beträgt\n"
    "1. in der ersten Altersstufe (§ 1612a Absatz 1 Satz 3 Nummer 1 des "
    "Bürgerlichen Gesetzbuchs) 482 Euro ab dem 1. Januar 2025 und 486 Euro "
    "ab dem 1. Januar 2026,\n"
    "2. in der zweiten Altersstufe (§ 1612a Absatz 1 Satz 3 Nummer 2 des "
    "Bürgerlichen Gesetzbuchs) 554 Euro ab dem 1. Januar 2025 und 558 Euro "
    "ab dem 1. Januar 2026,\n"
    "3. in der dritten Altersstufe (§ 1612a Absatz 1 Satz 3 Nummer 3 des "
    "Bürgerlichen Gesetzbuchs) 649 Euro ab dem 1. Januar 2025 und 653 Euro "
    "ab dem 1. Januar 2026."
)


def _stated_conversion_rulespec_with_divisor(
    citation_path: str,
    parameters: tuple[tuple[str, str], ...],
    *,
    annual_parameter: str,
    divisor: str,
) -> str:
    parameter_rules = "\n".join(
        f"""\
  - name: {name}
    kind: parameter
    dtype: Money
    source: {citation_path}
    versions:
      - effective_from: '2025-01-01'
        formula: {value}"""
        for name, value in parameters
    )
    return f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation_path}
rules:
{parameter_rules}
  - name: derived_monthly_amount
    kind: derived
    dtype: Money
    source: {citation_path}
    versions:
      - effective_from: '2025-01-01'
        formula: {annual_parameter} / {divisor}
"""


def test_exact_svbezgrv_2025_section_1_excludes_temporal_year_from_recall():
    citation_path = "de/regulation/svbezgrv-2025/1"
    assert (
        hashlib.sha256(SVBEZGRV_2025_SECTION_1_BODY.encode()).hexdigest()
        == "c2ab73a6ea8b7573c7073a340e2ec0714b44a50e70390d62cb46f741da614694"
    )
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation_path}
rules:
  - name: annual_reference_amount
    kind: parameter
    dtype: Money
    source: {citation_path}
    versions:
      - effective_from: '2025-01-01'
        formula: 44940
  - name: monthly_reference_amount
    kind: parameter
    dtype: Money
    source: {citation_path}
    versions:
      - effective_from: '2025-01-01'
        formula: 3745
"""

    result = _analyze(
        content,
        SVBEZGRV_2025_SECTION_1_BODY,
        corpus_citation_path=citation_path,
        test_cases=[],
    )

    assert not result.issues
    assert (
        result.source_numeric_occurrence_count,
        result.covered_source_numeric_occurrence_count,
        result.missing_source_numeric_occurrence_count,
    ) == (2, 2, 0)
    assert not _pipeline_issues(
        content,
        SVBEZGRV_2025_SECTION_1_BODY,
        corpus_citation_path=citation_path,
        test_cases=[],
    )


def test_exact_svbezgrv_2025_section_2_excludes_temporal_years_from_recall():
    citation_path = "de/regulation/svbezgrv-2025/2"
    assert (
        hashlib.sha256(SVBEZGRV_2025_SECTION_2_BODY.encode()).hexdigest()
        == "becd570c345b18381e5edd54a33754085b0ba1f80e879de6af961c07a9192bf4"
    )
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation_path}
rules:
  - name: general_annual_income_limit
    kind: parameter
    dtype: Money
    source: {citation_path}(1)
    versions:
      - effective_from: '2025-01-01'
        formula: 73800
  - name: general_monthly_income_limit
    kind: parameter
    dtype: Money
    source: {citation_path}(1)
    versions:
      - effective_from: '2025-01-01'
        formula: 6150
  - name: special_annual_income_limit
    kind: parameter
    dtype: Money
    source: {citation_path}(2)
    versions:
      - effective_from: '2025-01-01'
        formula: 66150
  - name: special_monthly_income_limit
    kind: parameter
    dtype: Money
    source: {citation_path}(2)
    versions:
      - effective_from: '2025-01-01'
        formula: 5512.5
"""

    result = _analyze(
        content,
        SVBEZGRV_2025_SECTION_2_BODY,
        corpus_citation_path=citation_path,
        test_cases=[],
    )

    assert not result.issues
    assert (
        result.source_numeric_occurrence_count,
        result.covered_source_numeric_occurrence_count,
        result.missing_source_numeric_occurrence_count,
    ) == (4, 4, 0)
    assert not _pipeline_issues(
        content,
        SVBEZGRV_2025_SECTION_2_BODY,
        corpus_citation_path=citation_path,
        test_cases=[],
    )


@pytest.mark.parametrize(
    ("citation_path", "source_body", "parameters", "annual_parameter"),
    (
        (
            "de/regulation/svbezgrv-2025/1",
            SVBEZGRV_2025_SECTION_1_BODY,
            (
                ("annual_reference_amount", "44940"),
                ("monthly_reference_amount", "3745"),
            ),
            "annual_reference_amount",
        ),
        (
            "de/regulation/svbezgrv-2025/2",
            SVBEZGRV_2025_SECTION_2_BODY,
            (
                ("general_annual_income_limit", "73800"),
                ("general_monthly_income_limit", "6150"),
                ("special_annual_income_limit", "66150"),
                ("special_monthly_income_limit", "5512.5"),
            ),
            "general_annual_income_limit",
        ),
    ),
)
def test_exact_svbezgrv_bare_calendar_divisor_gets_complete_mode_hint(
    citation_path,
    source_body,
    parameters,
    annual_parameter,
):
    content = _stated_conversion_rulespec_with_divisor(
        citation_path,
        parameters,
        annual_parameter=annual_parameter,
        divisor="12",
    )

    issues = find_ungrounded_numeric_issues_scoped(
        content,
        module_source_text=source_body,
        module_citation_path=citation_path,
        require_complete_source_unit=True,
    )

    assert len(issues) == 1
    assert issues[0].startswith(
        "Ungrounded generated numeric literal: 12 does not appear as a "
        "substantive numeric value in the source text."
    )
    assert "separate grounded `kind: parameter` rules" in issues[0]
    assert "annual and monthly amounts" in issues[0]
    assert "companion tests" in issues[0]


def test_stated_conversion_other_ungrounded_literal_keeps_original_error():
    content = _stated_conversion_rulespec_with_divisor(
        "de/regulation/svbezgrv-2025/1",
        (
            ("annual_reference_amount", "44940"),
            ("monthly_reference_amount", "3745"),
        ),
        annual_parameter="annual_reference_amount",
        divisor="13",
    )

    assert find_ungrounded_numeric_issues(
        content,
        source_text=SVBEZGRV_2025_SECTION_1_BODY,
        require_complete_source_unit=True,
    ) == [
        "Ungrounded generated numeric literal: 13 does not appear as a "
        "substantive numeric value in the source text."
    ]


def test_stated_conversion_calendar_hint_is_complete_mode_only():
    content = _stated_conversion_rulespec_with_divisor(
        "de/regulation/svbezgrv-2025/1",
        (
            ("annual_reference_amount", "44940"),
            ("monthly_reference_amount", "3745"),
        ),
        annual_parameter="annual_reference_amount",
        divisor="12",
    )

    assert find_ungrounded_numeric_issues(
        content,
        source_text=SVBEZGRV_2025_SECTION_1_BODY,
    ) == [
        "Ungrounded generated numeric literal: 12 does not appear as a "
        "substantive numeric value in the source text."
    ]


@pytest.mark.parametrize("calendar_constant", ("4", "12", "24", "52", "365"))
def test_complete_mode_stated_conversion_hint_covers_calendar_constants(
    calendar_constant,
):
    content = _stated_conversion_rulespec_with_divisor(
        "de/regulation/svbezgrv-2025/1",
        (
            ("annual_reference_amount", "44940"),
            ("monthly_reference_amount", "3745"),
        ),
        annual_parameter="annual_reference_amount",
        divisor=calendar_constant,
    )

    issues = find_ungrounded_numeric_issues(
        content,
        source_text=SVBEZGRV_2025_SECTION_1_BODY,
        require_complete_source_unit=True,
    )

    assert len(issues) == 1
    assert "Complete-source stated-conversion hint" in issues[0]


def test_stated_conversion_hint_requires_module_source_citation():
    content = """\
format: rulespec/v1
module:
  summary: Annual and monthly amounts.
rules:
  - name: derived_monthly_amount
    kind: derived
    dtype: Money
    versions:
      - effective_from: '2025-01-01'
        formula: annual_reference_amount / 12
"""

    assert find_ungrounded_numeric_issues(
        content,
        source_text=SVBEZGRV_2025_SECTION_1_BODY,
        require_complete_source_unit=True,
    ) == [
        "Ungrounded generated numeric literal: 12 does not appear as a "
        "substantive numeric value in the source text."
    ]


def test_stated_conversion_hint_requires_exact_source_citation_binding():
    content = _stated_conversion_rulespec_with_divisor(
        "de/regulation/wrong/9",
        (
            ("annual_reference_amount", "44940"),
            ("monthly_reference_amount", "3745"),
        ),
        annual_parameter="annual_reference_amount",
        divisor="12",
    )
    expected = [
        "Ungrounded generated numeric literal: 12 does not appear as a "
        "substantive numeric value in the source text."
    ]

    assert (
        find_ungrounded_numeric_issues(
            content,
            source_text=SVBEZGRV_2025_SECTION_1_BODY,
            source_citation_path="de/regulation/right/1",
            require_complete_source_unit=True,
        )
        == expected
    )
    assert (
        find_ungrounded_numeric_issues_scoped(
            content,
            module_source_text=SVBEZGRV_2025_SECTION_1_BODY,
            module_citation_path="de/regulation/right/1",
            require_complete_source_unit=True,
        )
        == expected
    )


def test_explicit_calendar_literal_in_source_grounds_normally():
    citation_path = "de/regulation/example/1"
    source = "Der Jahresbetrag ist das 12-Fache des Monatsbetrags."
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation_path}
rules:
  - name: annual_amount
    kind: derived
    dtype: Money
    source: {citation_path}
    versions:
      - effective_from: '2025-01-01'
        formula: monthly_amount * 12
"""

    assert (
        find_ungrounded_numeric_issues(
            content,
            source_text=source,
            require_complete_source_unit=True,
        )
        == []
    )


def test_exact_minuhv_section_1_excludes_date_and_nummer_values_from_recall():
    citation_path = "de/regulation/minuhv/1"
    assert (
        hashlib.sha256(MINUHV_SECTION_1_BODY.encode()).hexdigest()
        == "dc5fe67760f3cd298fd3ce3c7fa2625e4ea1bb60216be9cf05353316c44c3b7a"
    )
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation_path}
rules:
  - name: first_age_band_amount
    kind: derived
    dtype: Money
    source: {citation_path}(1)
    versions:
      - effective_from: '2025-01-01'
        formula: 482
      - effective_from: '2026-01-01'
        formula: 486
  - name: second_age_band_amount
    kind: derived
    dtype: Money
    source: {citation_path}(2)
    versions:
      - effective_from: '2025-01-01'
        formula: 554
      - effective_from: '2026-01-01'
        formula: 558
  - name: third_age_band_amount
    kind: derived
    dtype: Money
    source: {citation_path}(3)
    versions:
      - effective_from: '2025-01-01'
        formula: 649
      - effective_from: '2026-01-01'
        formula: 653
"""
    test_cases = [
        {
            "name": "2025",
            "period": "2025",
            "input": {},
            "output": {
                "first_age_band_amount": 482,
                "second_age_band_amount": 554,
                "third_age_band_amount": 649,
            },
        },
        {
            "name": "2026",
            "period": "2026",
            "input": {},
            "output": {
                "first_age_band_amount": 486,
                "second_age_band_amount": 558,
                "third_age_band_amount": 653,
            },
        },
    ]

    result = _analyze(
        content,
        MINUHV_SECTION_1_BODY,
        corpus_citation_path=citation_path,
        test_cases=test_cases,
    )

    assert not result.issues
    assert (
        result.source_numeric_occurrence_count,
        result.covered_source_numeric_occurrence_count,
        result.missing_source_numeric_occurrence_count,
    ) == (6, 6, 0)
    assert not _pipeline_issues(
        content,
        MINUHV_SECTION_1_BODY,
        corpus_citation_path=citation_path,
        test_cases=test_cases,
    )


def test_year_shaped_money_still_requires_complete_mode_representation():
    citation_path = "de/regulation/example/1"
    source = "Die Pauschale beträgt 2025 Euro."
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation_path}
rules: []
"""

    result = _analyze(
        content,
        source,
        corpus_citation_path=citation_path,
        test_cases=[],
    )

    assert result.source_numeric_occurrence_count == 1
    assert result.missing_source_numeric_occurrence_count == 1
    assert _has_issue(result, "2025", "numeric-recall")


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
        "authoritative corpus numeric value 73" in issue.lower() for issue in issues
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


def test_explicit_satz_markers_after_absatz_are_recognized():
    branches = recognize_source_structure(
        "(1) Satz 1 Die Hauptregel gilt; Satz 2 Die Sonderregel gilt."
    )

    assert {branch.path for branch in branches if branch.kind == "sentence"} == {
        ("1", "satz-1"),
        ("1", "satz-2"),
    }


def test_nj_title_54a_citations_are_not_glued_german_sentence_markers():
    branches = recognize_source_structure(
        "54A:4-7 New Jersey credit. N.J.S.54A:1-1 applies. "
        "C.54A:4-6 controls. 1Das ist Satz eins.2Voraussetzung zwei gilt."
    )

    assert [branch.label for branch in branches if branch.kind == "sentence"] == [
        "Satz 1",
        "Satz 2",
    ]


@pytest.mark.parametrize("ordinal", ["1ST", "2ND", "3RD", "4TH", "21st"])
def test_english_ordinals_are_not_glued_german_sentence_markers(ordinal: str):
    branches = recognize_source_structure(
        f"(2) The amount applies. Acts 1983, {ordinal} EX. SESS., No. 1."
    )

    assert [branch.label for branch in branches if branch.kind == "sentence"] == []


def test_louisiana_session_law_ordinal_does_not_invent_source_branch():
    source = """\
(1) Single Individual and Married-Separate $12,500.00

(2) Married-Joint Return, a Qualified Surviving 200% of the dollar amount

Spouse, and Head of Household provided for Single Individuals

Acts 1983, 2ND EX. SESS., NO. 1, §1.
"""

    branches = recognize_source_structure(source)

    assert [(branch.path, branch.kind) for branch in branches] == [
        (("1",), "paragraph"),
        (("2",), "paragraph"),
    ]


def test_louisiana_dotted_subsection_ends_line_wrapped_numeric_paragraph():
    source = """\
A. A standard deduction shall be allowed. For tax year 2025:

(1) Single Individual and Married-Separate $12,500.00

(2) Married-Joint Return, a Qualified Surviving 200% of the dollar amount

Spouse, and Head of Household provided for Single Individuals

B. Beginning January 1, 2026, and thereafter, the standard deduction shall be
adjusted annually by an amount calculated by multiplying the prior year's deduction
by the CPI-U increase.
"""

    branches = recognize_source_structure(source)

    assert [(branch.path, branch.label) for branch in branches] == [
        (("a",), "A."),
        (("1",), "(1)"),
        (("2",), "(2)"),
        (("b",), "B."),
    ]
    paragraph_two = next(branch for branch in branches if branch.path == ("2",))
    assert "200% of the dollar amount" in paragraph_two.text
    assert "Beginning January 1, 2026" not in paragraph_two.text

    formula_branches = completeness_module._source_formula_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    assert not any(
        "200%" in branch.text and "CPI-U" in branch.text for branch in formula_branches
    )
    assert any(branch.path == ("b",) for branch in formula_branches)


def test_lowercase_dotted_prose_is_not_a_top_level_subsection():
    branches = recognize_source_structure("a. ordinary prose\n(1) The rule applies.")

    assert [(branch.path, branch.label) for branch in branches] == [
        (("1",), "(1)"),
    ]


def test_nj_historical_rate_remains_a_source_unit_formula_obligation():
    source = (
        "54A:4-7 New Jersey credit. (2) For the purposes of the calculation of "
        "the New Jersey earned income tax credit, the percentage of the federal "
        "earned income tax credit referred to in paragraph (1) shall be: "
        "(a) 10% for the taxable year beginning on or after January 1, 2000, "
        "but before January 1, 2001;"
    )
    branches = recognize_source_structure(source)

    formula_branches = completeness_module._source_formula_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )

    assert [(branch.label, branch.path) for branch in formula_branches] == [
        ("source unit formula clause 2", ()),
    ]


def test_nj_historical_rate_has_source_faithful_runtime_witness():
    source = (
        "For taxable year 2000, an eligible resident's New Jersey earned income "
        "tax credit is 10% of the federal earned income tax credit. "
        "L.2000, c.80, s.2; amended 2007, c.109; 2020, c.98; 2021, c.130."
    )
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-nj/statute/54a:4-7
rules:
  - name: nj_earned_income_tax_credit_percentage
    kind: parameter
    dtype: Rate
    source: us-nj/statute/54a:4-7
    versions:
      - effective_from: '2000-01-01'
        effective_to: '2000-12-31'
        formula: 0.10
      - effective_from: '2020-01-01'
        formula: 0.40
  - name: nj_earned_income_tax_credit_before_proration
    kind: derived
    dtype: Money
    source: us-nj/statute/54a:4-7
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us-nj/statute/54a:4-7
              excerpt: >-
                For taxable year 2000, an eligible resident's New Jersey earned
                income tax credit is 10% of the federal earned income tax credit.
    versions:
      - effective_from: '2000-01-01'
        formula: |-
          if regular_nj_earned_income_tax_credit_eligible: federal_earned_income_tax_credit * nj_earned_income_tax_credit_percentage else: 0
  - name: regular_nj_earned_income_tax_credit_eligible
    kind: derived
    dtype: Judgment
    source: us-nj/statute/54a:4-7
    versions:
      - effective_from: '2000-01-01'
        formula: |-
          claimant_is_resident
          and eligible_for_federal_earned_income_tax_credit
"""
    test_cases = [
        {
            "name": "historical 10 percent branch",
            "period": {
                "period_kind": "tax_year",
                "start": "2000-01-01",
                "end": "2000-12-31",
            },
            "input": {
                "claimant_is_resident": True,
                "eligible_for_federal_earned_income_tax_credit": True,
                "federal_earned_income_tax_credit": 1000,
            },
            "output": {
                "regular_nj_earned_income_tax_credit_eligible": "holds",
                "nj_earned_income_tax_credit_before_proration": 100,
            },
        },
        {
            "name": "current 40 percent branch",
            "period": {
                "period_kind": "tax_year",
                "start": "2024-01-01",
                "end": "2024-12-31",
            },
            "input": {
                "claimant_is_resident": True,
                "eligible_for_federal_earned_income_tax_credit": True,
                "federal_earned_income_tax_credit": 1000,
            },
            "output": {
                "regular_nj_earned_income_tax_credit_eligible": "holds",
                "nj_earned_income_tax_credit_before_proration": 400,
            },
        },
    ]

    result = _analyze(
        content,
        source,
        corpus_citation_path="us-nj/statute/54a:4-7",
        test_cases=test_cases,
    )

    assert not _has_issue(result, "formula branch", "test")
    uncorroborated_cases = yaml.safe_load(yaml.safe_dump(test_cases))
    del uncorroborated_cases[0]["output"][
        "regular_nj_earned_income_tax_credit_eligible"
    ]
    uncorroborated = _analyze(
        content,
        source,
        corpus_citation_path="us-nj/statute/54a:4-7",
        test_cases=uncorroborated_cases,
    )
    assert _has_issue(
        uncorroborated,
        "formula branch",
        "historical 10 percent branch",
        "regular_nj_earned_income_tax_credit_eligible",
        "local derived dependency selector",
        "same case asserts its expected output",
        "never shadow",
    )
    environment = completeness_module._constant_rule_environment(
        yaml.safe_load(content)
    )
    assert completeness_module._formula_environment_for_case(
        environment,
        test_cases[0],
    )["nj_earned_income_tax_credit_percentage"] == pytest.approx(0.10)
    assert completeness_module._formula_environment_for_case(
        environment,
        test_cases[1],
    )["nj_earned_income_tax_credit_percentage"] == pytest.approx(0.40)
    current_only = _analyze(
        content,
        source,
        corpus_citation_path="us-nj/statute/54a:4-7",
        test_cases=test_cases[1:],
    )
    assert _has_issue(current_only, "formula branch", "test")
    assert _has_issue(
        current_only,
        "exact source computation",
        "taxable year 2000",
        "10%",
        "legally applicable companion-case period",
        "observed asserted candidate-case periods: 2024-01-01",
    )
    unbound_payload = yaml.safe_load(content)
    unbound_principal = next(
        rule
        for rule in unbound_payload["rules"]
        if rule["name"] == "nj_earned_income_tax_credit_before_proration"
    )
    unbound_principal["source"] = "N.J.S.54A:4-7(a)(1), (a)(2)"
    unbound_principal["metadata"]["proof"]["atoms"] = [
        {
            "path": "versions[0].formula",
            "kind": "formula",
            "source": {
                "corpus_citation_path": "us-nj/statute/54a:4-7",
            },
        },
        {
            "path": "versions[0].formula",
            "kind": "parameter",
            "import": {
                "target": (
                    "us-nj:statutes/54a/4-7#nj_earned_income_tax_credit_percentage"
                ),
                "output": "nj_earned_income_tax_credit_percentage",
                "hash": "sha256:local",
            },
        },
    ]
    unbound = _analyze(
        yaml.safe_dump(unbound_payload, sort_keys=False),
        source,
        corpus_citation_path="us-nj/statute/54a:4-7",
        test_cases=test_cases,
    )
    assert _has_issue(
        unbound,
        "already execute this computation",
        "nj_earned_income_tax_credit_before_proration",
        "excluded from source-bound evidence",
        "versions[n].formula",
        "short `source.excerpt`",
        "self-import",
    )
    long_excerpt = completeness_module._bounded_source_feedback_excerpt(
        "head " + "x" * 500 + " operative tail"
    )
    assert len(long_excerpt) == 360
    assert long_excerpt.startswith("head ")
    assert long_excerpt.endswith(" operative tail")
    assert completeness_module._bounded_period_feedback(
        [f"{year}-01-01" for year in range(2000, 2020)]
    ) == (
        "2000-01-01, 2001-01-01, 2002-01-01, 2003-01-01, "
        "... (12 omitted) ..., 2016-01-01, 2017-01-01, 2018-01-01, 2019-01-01"
    )
    assert completeness_module._bounded_identifier_feedback(
        [f"rule_{index}" for index in range(8)] + ["rule_0", "rule`escaped"]
    ) == (
        "`rule\\`escaped`, `rule_0`, `rule_1`, `rule_2`, `rule_3`, `rule_4`, "
        "... (3 omitted)"
    )

    premature_payload = yaml.safe_load(content)
    premature_principal = next(
        rule
        for rule in premature_payload["rules"]
        if rule["name"] == "nj_earned_income_tax_credit_before_proration"
    )
    premature_principal["versions"][0]["effective_from"] = "2020-01-01"
    premature_principal_formula = premature_principal["versions"][0]["formula"]
    assert (
        completeness_module._rule_formula_text_for_case(
            premature_principal,
            test_cases[0],
        )
        is None
    )
    assert (
        completeness_module._rule_formula_text_for_case(
            premature_principal,
            test_cases[1],
        )
        == premature_principal_formula
    )
    expired_principal = yaml.safe_load(yaml.safe_dump(premature_principal))
    expired_principal["versions"][0]["effective_from"] = "2000-01-01"
    expired_principal["versions"][0]["effective_to"] = "2000-12-31"
    assert (
        completeness_module._rule_formula_text_for_case(
            expired_principal,
            test_cases[1],
        )
        is None
    )
    undated_principal = {"versions": [{"formula": premature_principal_formula}]}
    assert (
        completeness_module._rule_formula_text_for_case(
            undated_principal,
            test_cases[0],
        )
        == premature_principal_formula
    )
    malformed_principal = {
        "versions": [
            {
                "effective_from": "2020",
                "formula": premature_principal_formula,
            }
        ]
    }
    assert (
        completeness_module._rule_formula_text_for_case(
            malformed_principal,
            test_cases[1],
        )
        is None
    )
    assert (
        completeness_module._rule_formula_text_for_case(
            premature_principal,
            {"period": "not-a-year"},
        )
        is None
    )
    premature = _analyze(
        yaml.safe_dump(premature_payload, sort_keys=False),
        source,
        corpus_citation_path="us-nj/statute/54a:4-7",
        test_cases=test_cases,
    )
    assert _has_issue(premature, "formula branch", "test")


def test_formula_witness_requires_reached_arithmetic_dependency_assertion():
    source = (
        "For taxable year 2000, an eligible resident's New Jersey earned income "
        "tax credit is 10% of the federal earned income tax credit."
    )
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-zz/statute/1
rules:
  - name: pct
    kind: parameter
    dtype: Rate
    source: us-zz/statute/1
    versions:
      - effective_from: '2000-01-01'
        formula: 0.10
  - name: adjusted_federal_credit
    kind: derived
    dtype: Money
    source: us-zz/statute/1
    versions:
      - effective_from: '2000-01-01'
        formula: federal_credit
  - name: eligible
    kind: derived
    dtype: Judgment
    source: us-zz/statute/1
    versions:
      - effective_from: '2000-01-01'
        formula: claimant_is_resident
  - name: credit
    kind: derived
    dtype: Money
    source: us-zz/statute/1
    metadata:
      proof:
        atoms:
          - path: formula
            kind: formula
            source:
              corpus_citation_path: us-zz/statute/1
              excerpt: >-
                eligible resident's New Jersey earned income tax credit is 10%
                of the federal earned income tax credit
    versions:
      - effective_from: '2000-01-01'
        formula: |-
          if eligible: adjusted_federal_credit * pct else: 0
"""
    cases = [
        {
            "name": "branch",
            "period": "2000-01-01",
            "input": {
                "claimant_is_resident": True,
                "federal_credit": 1000,
            },
            "output": {
                "eligible": "holds",
                "credit": 100,
            },
        },
        {
            "name": "other assertion",
            "period": "2000-01-01",
            "input": {
                "claimant_is_resident": False,
                "federal_credit": 500,
            },
            "output": {"adjusted_federal_credit": 500},
        },
    ]

    uncorroborated = _analyze(
        content,
        source,
        corpus_citation_path="us-zz/statute/1",
        test_cases=cases,
    )

    assert _has_issue(
        uncorroborated,
        "formula branch",
        "branch",
        "adjusted_federal_credit",
        "same case asserts its expected output",
    )
    cases[0]["output"]["adjusted_federal_credit"] = 1000
    corroborated = _analyze(
        content,
        source,
        corpus_citation_path="us-zz/statute/1",
        test_cases=cases,
    )
    assert not _has_issue(corroborated, "formula branch", "test")


def test_formula_interval_witness_traces_derived_numeric_selector():
    source = (
        "For income up to 10000 dollars, the credit is 10% of the base amount "
        "minus the deduction."
    )
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-zz/statute/2
rules:
  - name: rate
    kind: parameter
    dtype: Rate
    source: us-zz/statute/2
    versions:
      - effective_from: '2000-01-01'
        formula: 0.10
  - name: threshold
    kind: parameter
    dtype: Money
    source: us-zz/statute/2
    versions:
      - effective_from: '2000-01-01'
        formula: 10000
  - name: low_income
    kind: derived
    dtype: Judgment
    source: us-zz/statute/2
    versions:
      - effective_from: '2000-01-01'
        formula: income <= threshold
  - name: deduction_positive
    kind: derived
    dtype: Judgment
    source: us-zz/statute/2
    versions:
      - effective_from: '2000-01-01'
        formula: deduction > 0
  - name: credit
    kind: derived
    dtype: Money
    source: us-zz/statute/2
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us-zz/statute/2
              excerpt: >-
                For income up to 10000 dollars, the credit is 10% of the base
                amount minus the deduction.
    versions:
      - effective_from: '2000-01-01'
        formula: |-
          if low_income and deduction_positive: (base_amount - deduction) * rate else: 0
"""
    qualifying_case = {
        "name": "qualifying income",
        "period": "2000-01-01",
        "input": {"income": 5000, "base_amount": 20000, "deduction": 50},
        "output": {
            "low_income": "holds",
            "deduction_positive": "holds",
            "credit": 1995,
        },
    }

    qualifying = _analyze(
        content,
        source,
        corpus_citation_path="us-zz/statute/2",
        test_cases=[qualifying_case],
    )

    assert not _has_issue(qualifying, "formula branch", "test")
    wrong_threshold = content.replace("formula: 10000", "formula: 30000")
    nonqualifying_income = {
        "name": "deduction happens to fit source interval",
        "period": "2000-01-01",
        "input": {"income": 20000, "base_amount": 20000, "deduction": 50},
        "output": {
            "low_income": "holds",
            "deduction_positive": "holds",
            "credit": 1995,
        },
    }
    wrong_selector = _analyze(
        wrong_threshold,
        source,
        corpus_citation_path="us-zz/statute/2",
        test_cases=[nonqualifying_income],
    )
    assert _has_issue(wrong_selector, "formula branch", "test")
    dynamic_bound = content.replace("income <= threshold", "income <= dynamic_limit")
    dynamic_bound_case = yaml.safe_load(yaml.safe_dump(qualifying_case))
    dynamic_bound_case["input"]["dynamic_limit"] = 10000
    ungrounded_bound = _analyze(
        dynamic_bound,
        source,
        corpus_citation_path="us-zz/statute/2",
        test_cases=[dynamic_bound_case],
    )
    assert _has_issue(ungrounded_bound, "formula branch", "test")


@pytest.mark.parametrize(
    ("selector", "eligible_value"),
    [
        ("eligible or special_eligibility", True),
        ("eligible and special_eligibility", False),
    ],
)
def test_reached_formula_dependencies_honor_boolean_short_circuiting(
    selector,
    eligible_value,
):
    execution = completeness_module._FormulaExecution(
        trace=(
            completeness_module._FormulaTraceStep(
                kind="if",
                selectors=(selector,),
                choice=0,
            ),
        ),
        leaf="input_amount * rate",
        evaluated_value=None,
        evaluates_to_zero=False,
        constant_environment={},
    )
    principal_rules = {
        "eligible": {
            "name": "eligible",
            "kind": "derived",
            "versions": [{"formula": "claimant_is_resident"}],
        },
        "special_eligibility": {
            "name": "special_eligibility",
            "kind": "derived",
            "versions": [{"formula": "has_special_status"}],
        },
    }
    case = {
        "input": {
            "claimant_is_resident": eligible_value,
            "has_special_status": not eligible_value,
            "input_amount": 100,
        },
        "output": {
            "eligible": "holds" if eligible_value else "not_holds",
        },
    }

    reached = completeness_module._reached_local_formula_dependency_names(
        execution,
        case,
        principal_rules=principal_rules,
        formula_environment={"rate": 0.1},
        dependency_environment={"eligible": eligible_value},
    )

    assert reached == {"eligible"}


def test_reached_formula_dependencies_short_circuit_boolean_leaf():
    execution = completeness_module._FormulaExecution(
        trace=(),
        leaf="eligible or special_eligibility",
        evaluated_value=("bool", "True"),
        evaluates_to_zero=False,
        constant_environment={},
    )
    principal_rules = {
        "eligible": {
            "name": "eligible",
            "kind": "derived",
            "versions": [{"formula": "claimant_is_resident"}],
        },
        "special_eligibility": {
            "name": "special_eligibility",
            "kind": "derived",
            "versions": [{"formula": "has_special_status"}],
        },
    }
    case = {
        "input": {
            "claimant_is_resident": True,
            "has_special_status": False,
        },
        "output": {"eligible": "holds"},
    }

    reached = completeness_module._reached_local_formula_dependency_names(
        execution,
        case,
        principal_rules=principal_rules,
        formula_environment={},
        dependency_environment={"eligible": True},
    )

    assert reached == {"eligible"}


def test_unbound_formula_diagnostics_cache_case_execution(monkeypatch):
    rule_count = 40
    case_count = 12
    branch_count = 12
    principal_rules = {
        f"amount_{index}": {
            "name": f"amount_{index}",
            "kind": "derived",
            "dtype": "Money",
            "versions": [{"formula": "input_amount * 2"}],
        }
        for index in range(rule_count)
    }
    cases = [
        {
            "name": f"shared asserted case {case_index}",
            "input": {"input_amount": 10},
            "output": {f"amount_{index}": 20 for index in range(rule_count)},
        }
        for case_index in range(case_count)
    ]
    asserted_by_rule = {name: cases for name in principal_rules}
    dependency_cache = {}
    execution_cache = {}
    dependency_calls = 0
    execution_calls = 0
    original_dependency = completeness_module._case_asserted_dependency_environment
    original_execution = completeness_module._case_formula_execution

    def counted_dependency(*args, **kwargs):
        nonlocal dependency_calls
        dependency_calls += 1
        return original_dependency(*args, **kwargs)

    def counted_execution(*args, **kwargs):
        nonlocal execution_calls
        execution_calls += 1
        return original_execution(*args, **kwargs)

    monkeypatch.setattr(
        completeness_module,
        "_case_asserted_dependency_environment",
        counted_dependency,
    )
    monkeypatch.setattr(
        completeness_module,
        "_case_formula_execution",
        counted_execution,
    )

    for index in range(branch_count):
        text = "The amount is input_amount * 2."
        branch = completeness_module.SourceStructureBranch(
            path=(),
            kind="root",
            label=f"formula {index}",
            text=text,
            start=index * len(text),
            end=(index + 1) * len(text),
        )
        diagnostic = completeness_module._unbound_matching_formula_rules(
            branch,
            principal_rules=principal_rules,
            bound_rule_names=set(),
            asserted_by_rule=asserted_by_rule,
            extract_numeric_occurrences=EN_NUMERIC_OCCURRENCE_EXTRACTOR,
            numeric_value_is_grounded=numeric_value_is_grounded,
            formula_environment={},
            dependency_cache=dependency_cache,
            execution_cache=execution_cache,
        )
        assert len(diagnostic.rule_names) == (
            completeness_module._UNBOUND_FORMULA_DIAGNOSTIC_RULE_LIMIT
        )
        assert diagnostic.scan_capped is True

    feedback = completeness_module._unbound_formula_binding_feedback(
        diagnostic,
        corpus_citation_path="us/example/statute/1",
    )
    assert "diagnostic scan was capped at 32 rules and 8 cases per rule" in feedback
    assert "additional matching principal formulas may exist" in feedback
    capped_without_scanned_match = (
        completeness_module._unbound_formula_binding_feedback(
            completeness_module._UnboundFormulaDiagnostic((), True),
            corpus_citation_path="us/example/statute/1",
        )
    )
    assert "diagnostic scan was capped" in capped_without_scanned_match
    assert "additional matching principal formulas may exist" in (
        capped_without_scanned_match
    )

    assert dependency_calls == (
        completeness_module._UNBOUND_FORMULA_DIAGNOSTIC_CASE_LIMIT
    )
    assert execution_calls == (
        completeness_module._UNBOUND_FORMULA_DIAGNOSTIC_RULE_LIMIT
        * completeness_module._UNBOUND_FORMULA_DIAGNOSTIC_CASE_LIMIT
    )


def test_colon_satz_markers_are_recognized_and_independently_required():
    source = "(1) Satz 1: Die Hauptregel gilt. Satz 2: Die Sonderregel gilt."
    branches = recognize_source_structure(source)

    assert {branch.path for branch in branches if branch.kind == "sentence"} == {
        ("1", "satz-1"),
        ("1", "satz-2"),
    }

    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules: []
"""
    result = _analyze(content, source, test_cases=[])

    assert _has_issue(result, "Satz 1", "source branch")
    assert _has_issue(result, "Satz 2", "source branch")


@pytest.mark.parametrize(
    ("source", "source_reference", "parent_label"),
    [
        (
            "(1) Anspruch besteht nur bei Wohnsitz.\n"
            "1. Der Freibetrag beträgt 259 Euro.",
            "de/statute/estg/32a(1)(1)",
            "(1)",
        ),
        (
            "1. Anspruch besteht nur bei Wohnsitz.\n"
            "a) Der Freibetrag beträgt 259 Euro.",
            "de/statute/estg/32a Nummer 1 Buchstabe a",
            "1.",
        ),
    ],
)
def test_child_encoding_cannot_hide_substantive_parent_chapeau(
    source: str,
    source_reference: str,
    parent_label: str,
):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Scalar child only.
rules:
  - name: allowance_amount
    kind: parameter
    dtype: Money
    source: {source_reference}
    versions:
      - effective_from: '2026-01-01'
        formula: 259
"""

    result = _analyze(content, source, test_cases=[])

    assert any(f"Source branch {parent_label} at " in issue for issue in result.issues)


def test_child_encoding_covers_marker_only_parent_container():
    source = "(1)\n1. Der Freibetrag beträgt 259 Euro."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Scalar child.
rules:
  - name: allowance_amount
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 259
"""

    result = _analyze(content, source, test_cases=[])

    assert not result.issues


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

    released_absatz_6 = "(6) " + RELEASED_ESTG_32A_BODY.split("\n(6) ", 1)[1]
    result = _analyze(content, released_absatz_6, test_cases=[])

    assert _has_issue(result, "(6)", "deferral")
    issue = next(
        issue
        for issue in result.issues
        if issue.startswith("[complete-source-unit:deferral]")
    )
    assert "Required shape" in issue
    assert "module:\n  deferred_outputs:" in issue
    assert "the output path is not a source citation" in issue.lower()
    assert "output: de:statutes/estg/32a/6#surviving_spouse_splitting_tax" in issue
    assert "reason: Cannot be computed until" in issue
    assert "EStG § 26" in issue
    assert "`blocked_by` is optional" in issue


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


@pytest.mark.parametrize("dash", ["-", "‐", "‑", "‒", "–", "—", "−"])
def test_exact_usc_branch_can_name_source_bound_runtime_gap(dash: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c–1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/a#five_year_agency_plan
      reason: >-
        Cannot be computed until the agency fiscal-year calendar and plan-submission
        event required by 42 USC 1437c{dash}1(a)(3) are encoded.
rules: []
"""
    source = "(a) Each agency shall submit a plan for its fiscal year."

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c–1",
        test_cases=[],
    )

    assert not result.issues


@pytest.mark.parametrize(
    "reason",
    [
        "Cannot be encoded because 42 USC 1437c-1(a) is unavailable.",
        (
            "Cannot be computed until the agency fiscal-year calendar and "
            "plan-submission event required by 42 USC 1437c-1(b) are encoded."
        ),
        (
            "Cannot be computed until the agency fiscal-year calendar and "
            "plan-submission event required by 42 USC 1437c-2(a) are encoded."
        ),
        (
            "Cannot be computed until the fictional assessment-record workflow "
            "required by 42 USC 1437c-1(a) is encoded."
        ),
    ],
)
def test_current_usc_branch_does_not_launder_imprecise_runtime_gap(reason: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c–1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/a#five_year_agency_plan
      reason: >-
        {reason}
rules: []
"""
    source = "(a) Each agency shall submit a plan for its fiscal year."

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c–1",
        test_cases=[],
    )

    assert _has_issue(result, "(a)", "deferral", "runtime capability")
    issue = next(
        issue
        for issue in result.issues
        if issue.startswith("[complete-source-unit:deferral]")
    )
    assert "`42 U.S.C. 1437c-1(a)`" in issue


@pytest.mark.parametrize(
    ("terminal_fragment", "accepted"),
    [("A", True), ("B", False)],
)
def test_subsection_scoped_runtime_gap_requires_exact_terminal_branch(
    terminal_fragment: str,
    accepted: bool,
):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1396a/a/10
  deferred_outputs:
    - output: us:statutes/42/1396a/a/10/A#hearing_process
      reason: >-
        Cannot be computed until the hearing event and notice record required by
        42 U.S.C. 1396a(a)(10)({terminal_fragment}) are available at runtime.
rules: []
"""
    source = (
        "(A) The State agency shall conduct a hearing and provide notice of "
        "the hearing."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1396a/a/10",
        test_cases=[],
    )

    if accepted:
        assert not result.issues
    else:
        assert _has_issue(result, "(a)", "deferral", "runtime capability")
        issue = next(
            issue
            for issue in result.issues
            if issue.startswith("[complete-source-unit:deferral]")
        )
        assert "`42 U.S.C. 1396a(a)(10)(A)`" in issue


def test_current_usc_branch_cannot_defer_directly_computable_source():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/9999
  deferred_outputs:
    - output: us:statutes/42/9999/a#household_benefit
      reason: >-
        Cannot be computed until the household benefit input and household income
        formula required by 42 USC 9999(a) are encoded.
rules: []
"""
    source = "(a) The household benefit equals household income plus an adjustment."

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/9999",
        test_cases=[],
    )

    assert _has_issue(result, "(a)", "deferral", "runtime capability")


def test_administrative_runtime_gap_cannot_hide_worded_computation():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/9999
  deferred_outputs:
    - output: us:statutes/42/9999/a#household_benefit
      reason: >-
        Cannot be computed until the agency report document and household benefit
        input required by 42 USC 9999(a) are encoded.
rules: []
"""
    source = (
        "(a) The agency shall submit a report showing the household benefit, "
        "which is household income less the adjustment."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/9999",
        test_cases=[],
    )

    assert _has_issue(result, "(a)", "deferral", "runtime capability")


@pytest.mark.parametrize(
    "source_dependency",
    [
        "section 1437f(o) of this title",
        "42 U.S.C. 1437f(o)",
        "42 U.S.C. § 1437f(o)",
        "42 USC section 1437f(o)",
    ],
)
def test_source_bound_usc_dependency_can_defer_computable_source(
    source_dependency: str,
):
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until assistance eligibility under 42 USC 1437f(o)
        is encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        f"{source_dependency} and has 550 or fewer units."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


def test_unrelated_usc_dependency_cannot_defer_computable_source():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until assistance eligibility under 42 USC 9999(a)
        is encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "section 1437f(o) of this title and has 550 or fewer units."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_usc_dependency_requires_exact_subsection_tail():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until assistance eligibility under 42 USC 1437f(o)
        is encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(q) and has 550 or fewer units."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize(
    "dependency",
    [
        "7 USC 2014",
        "7 USC section 2014",
        "7 U.S.C. § 2014",
        "7 U.S.C., section 2014",
        "7 U.S.C., § 2014",
        "section 2014 of title 7",
        "§ 2014 of title 7",
        "section 2014 in title 7",
        "title 7, section 2014",
    ],
)
def test_relative_usc_dependency_cannot_bind_wrong_title_from_unrelated_number(
    dependency: str,
):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: Cannot be computed until {dependency} eligibility is encoded.
rules: []
"""
    source = (
        "(b) The agency shall submit a report within 7 days under section 2014 "
        "of this title and has 550 or fewer units."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize(
    "source_dependency",
    [
        "section 1437f(o) of title 7",
        "section 1437f(o) in title 7",
        "section 1437f(o), as codified in title 7",
    ],
)
def test_relative_usc_dependency_cannot_fall_through_qualified_wrong_title(
    source_dependency: str,
):
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until eligibility under 42 USC 1437f(o) is encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        f"{source_dependency}."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_same_external_citation_must_be_missing_and_source_bound():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until eligibility under 7 USC 9999 is encoded.
        For context, the source mentions 42 USC 1437f(o).
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o) and has 550 or fewer units."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_same_clause_cannot_launder_missingness_between_usc_citations():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until 7 USC 9999 is encoded, but 42 USC 1437f(o)
        is available.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_postpositive_missingness_cannot_transfer_to_later_usc_citation():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed because 7 USC 9999 is missing and 42 USC 1437f(o)
        applies.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize(
    "reason",
    [
        "Cannot be computed because 7 USC 9999 is clearly missing and 42 USC 1437f(o) applies.",
        "Cannot be computed because 7 USC 9999 is very clearly missing and 42 USC 1437f(o) applies.",
        "Cannot be computed because 7 USC 9999, which is missing, and 42 USC 1437f(o) applies.",
        "Cannot be computed because 7 USC 9999, which the agency reports is missing, and 42 USC 1437f(o) applies.",
        "Cannot be computed because § 9999 is missing and 42 USC 1437f(o) applies.",
    ],
)
def test_modified_postpositive_missingness_cannot_transfer(reason: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: {reason}
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_until_state_cannot_transfer_from_later_contextual_citation():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until 42 USC 1437f(o) is mentioned only for context,
        but 7 USC 9999 is encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize(
    "predicate",
    [
        "is discussed only for legislative history",
        "is quoted only for legislative history",
        "is summarized only for legislative history",
    ],
)
def test_contextual_passive_is_not_a_missing_dependency_state(predicate: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until 42 USC 1437f(o) {predicate}.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_contextual_required_citation_cannot_borrow_prior_sentence_missingness():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until 7 USC 9999 is encoded. For context,
        42 USC 1437f(o) is required by a separate example.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize("state", ["missing", "unavailable", "not yet encoded"])
def test_explicit_missing_state_requires_local_computation_framing(state: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until 7 USC 9999 is encoded. For context,
        42 USC 1437f(o) is {state}.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_explicit_missing_state_rejects_contextual_same_sentence_framing():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed because 7 USC 9999 is missing and, for comparison,
        42 USC 1437f(o) is unavailable.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize(
    "predicate",
    [
        "is provided only for legislative history",
        "is known only as historical background",
        "is supplied only for comparison",
    ],
)
def test_dependency_state_rejects_contextual_tail(predicate: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: Cannot be computed until 42 USC 1437f(o) {predicate}.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_coordinated_state_tail_must_begin_with_next_dependency():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until 42 USC 1437f(o) is provided and, according
        to 7 USC 9999, the former provision is cited only for legislative history.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_coordinated_state_tail_requires_next_dependency_state():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until 42 USC 1437f(o) is provided and 7 USC 9999
        states that the former provision is cited only for legislative history.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize(
    "reason",
    [
        "Cannot be computed until 42 USC 1437f(o), included solely as historical authority, and 7 USC 9999 are encoded.",
        "Cannot be computed until 42 USC 1437f(o), supplied solely for comparison, and 7 USC 9999 are encoded.",
        "Cannot be computed until 42 USC 1437f(o) applies, yet 7 USC 9999 is encoded.",
        "Cannot be computed until 42 USC 1437f(o) applies, while 7 USC 9999 is encoded.",
        "Cannot be computed until 42 USC 1437f(o) applies, even though 7 USC 9999 is encoded.",
        "Cannot be computed until 42 USC 1437f(o) applies; nevertheless 7 USC 9999 is encoded.",
        "Cannot be computed until eligibility under 42 USC 1437f(o) and, albeit the former citation is included only for context, 7 USC 9999 are encoded.",
        "Cannot be computed until eligibility under 42 USC 1437f(o) and, even if the former citation is included only for context, 7 USC 9999 are encoded.",
        "Cannot be computed until eligibility under 42 USC 1437f(o) and the former citation governs only as context under 7 USC 9999 are encoded.",
    ],
)
def test_contextual_list_cannot_borrow_terminal_dependency_state(reason: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: {reason}
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize(
    "reason",
    [
        "Cannot be computed until 7 USC 9999 and 42 USC 1437f(o) are encoded.",
        "Cannot be computed until 7 USC 9999 is encoded and 42 USC 1437f(o) is encoded.",
    ],
)
def test_coordinated_usc_dependencies_can_bind_later_citation(reason: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: {reason}
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


def test_coordinated_state_tail_accepts_introduced_next_dependency():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until 42 USC 1437f(o) is encoded and eligibility
        under 7 USC 9999 is encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


def test_descriptive_dependency_list_can_share_terminal_state_predicate():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until assistance receipt under 42 USC 1437f(o) and
        42 USC 1437g, troubled-agency designation under 42 USC 1437d(j)(2),
        and the failing-score determination under the Section 8 Management
        Assessment Program referenced by 42 USC 1437c-1(b)(3)(C) are encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "section 1437f(o) of this title."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


@pytest.mark.parametrize(
    "dependency",
    [
        "7 U.S.C., section 2014",
        "7 U.S.C.; section 2014",
        "title 7, section 2014",
    ],
)
def test_descriptive_dependency_list_preserves_qualified_citation_commas(
    dependency: str,
):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until assistance under 42 USC 1437f(o) and
        program eligibility under {dependency} are encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


def test_descriptive_dependency_list_accepts_benefit_amount_subject():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until assistance under 42 USC 1437f(o) and benefit
        amount under 7 USC 9999 are encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


@pytest.mark.parametrize(
    "subject",
    [
        "income threshold",
        "agency policy",
        "tax rate",
        "eligibility criteria",
        "agency rule",
    ],
)
def test_descriptive_dependency_list_accepts_common_legal_subjects(subject: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until assistance under 42 USC 1437f(o) and {subject}
        under 7 USC 9999 are encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


@pytest.mark.parametrize(
    "reason",
    [
        "Cannot be computed until eligibility under 42 USC 1437f(o) is determined.",
        "Cannot be computed until information required by 42 USC 1437f(o) is provided.",
        "Cannot be computed until eligibility under 42 USC 1437f(o) is set.",
        "Cannot be computed until information under 42 USC 1437f(o) is made available.",
        "Cannot be computed until eligibility under 42 USC 1437f(o) is verified.",
        "Cannot be computed until information under 42 USC 1437f(o) is issued.",
        "Cannot be computed until eligibility under 42 USC 1437f(o) is approved.",
    ],
)
def test_until_dependency_accepts_non_contextual_result_predicates(reason: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: {reason}
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


@pytest.mark.parametrize(
    "predicate",
    [
        "is verified by the agency",
        "is issued by the agency",
        "is approved by HUD",
        "is verified by a public housing agency",
        "is issued by the state agency",
        "is approved by the Secretary of Housing and Urban Development",
        "is verified by the administering agency",
        "is issued by the Department of Housing and Urban Development",
        "is verified by the Social Security Administration",
        "is issued by the Commissioner",
        "is verified by the Internal Revenue Service",
        "is approved by the Department of Education",
        "is verified by the Commissioner of Social Security",
        "is verified by the Commissioner of Internal Revenue",
        "is approved by the Department of Veterans Affairs",
        "is approved by the Department of the Treasury",
        "is issued by the Secretary of Veterans Affairs",
        "is issued by the Secretary of the Treasury",
        "is verified by the IRS",
    ],
)
def test_until_dependency_accepts_legal_actor_tail(predicate: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until eligibility under 42 USC 1437f(o) {predicate}.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


def test_legal_actor_tail_rejects_context_after_actor_name():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until eligibility under 42 USC 1437f(o) is approved
        by the Secretary of Housing and Urban Development only for historical
        comparison.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_contextual_operative_introduction_is_not_source_bound_dependency():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until the citation included only for legislative
        history under 42 USC 1437f(o) is encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize(
    "context",
    ["included only as nonbinding authority", "included for orientation"],
)
def test_non_dependency_subject_cannot_become_operative_via_under(context: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until the citation {context} under
        42 USC 1437f(o) is encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_direct_missing_state_requires_bounded_dependency_introduction():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until the citation included only as nonbinding
        authority under 42 USC 1437f(o) is unavailable.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_incidental_dependency_noun_cannot_launder_weak_linker():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until comparison amount included only as nonbinding
        authority under 42 USC 1437f(o) is encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize(
    "subject",
    [
        "nonbinding comparison amount",
        "the merely illustrative eligibility status",
        "comparison amount under nonbinding authority referenced by",
    ],
)
@pytest.mark.parametrize("state", ["encoded", "unavailable"])
def test_descriptive_subject_cannot_launder_weak_dependency_linker(
    subject: str,
    state: str,
):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until {subject} 42 USC 1437f(o) is {state}.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize(
    "subject",
    [
        "the nonbinding claim that this example depends on",
        "the assertion that this historical example requires",
        "benefit amount under the nonbinding example program referenced by",
        "eligibility status under the merely illustrative statute cited in",
        "benefit amount under the Nonbinding Example Program referenced by",
        "eligibility status under the Merely Illustrative Statute cited in",
    ],
)
def test_contextual_subject_cannot_launder_dependency_linker(subject: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until {subject} 42 USC 1437f(o) is verified.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize(
    "instrument",
    [
        "Section 8 Management Assessment Program",
        "Food and Nutrition Act",
        "Internal Revenue Code",
        "Social Security Act",
        "Administrative Procedure Act",
        "Veterans Benefits Act",
        "Patient Protection and Affordable Care Act",
        "Fair Labor Standards Act",
        "Code of Federal Regulations",
        "Low-Income Home Energy Assistance Program",
        "Housing Choice Voucher Program",
    ],
)
def test_named_legal_instrument_dependency_is_bounded(instrument: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount under the {instrument}
        referenced by 42 USC 1437f(o) is encoded.
rules: []
"""
    source = f"(b) The {instrument} governs assistance received under 42 USC 1437f(o)."

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


@pytest.mark.parametrize(
    "instrument",
    [
        "Social Security Act",
        "Internal Revenue Code",
        "Food and Nutrition Act",
    ],
)
def test_named_legal_instrument_must_be_bound_to_source(instrument: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount under the {instrument}
        referenced by 42 USC 1437f(o) is encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize(
    "source",
    [
        "(b) The Social Security Act appears only in historical context. "
        "Assistance is received under 42 USC 1437f(o).",
        "(b) An unfair labor standards action is discussed under 42 USC 1437f(o).",
    ],
)
def test_named_legal_instrument_requires_exact_title_in_citation_clause(source: str):
    instrument = (
        "Social Security Act"
        if "Social Security Act" in source
        else "Fair Labor Standards Act"
    )
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount under the {instrument}
        referenced by 42 USC 1437f(o) is encoded.
rules: []
"""

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_named_instrument_cannot_mix_contextual_and_operative_citation_occurrences():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount under the Social Security Act
        referenced by 42 USC 1437f(o) is encoded.
rules: []
"""
    sources = [
        "(b) The Social Security Act and 42 USC 1437f(o) appear only as "
        "historical context. Assistance is received under 42 USC 1437f(o).",
        "(b) The Social Security Act is included only for historical context, "
        "but assistance is received under 42 USC 1437f(o).",
    ]

    for source in sources:
        result = _analyze(
            content,
            source,
            corpus_citation_path="us/statute/42/1437c-1",
            test_cases=[],
        )
        assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_named_instrument_cannot_cross_coordinated_finite_clause():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount under the Social Security Act
        referenced by 42 USC 1437f(o) is encoded.
rules: []
"""
    source = (
        "(b) The Social Security Act governs historical records, and assistance "
        "is received under 42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_named_instrument_cannot_cross_punctuation_free_finite_coordination():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount under the Social Security Act
        referenced by 42 USC 1437f(o) is encoded.
rules: []
"""
    source = (
        "(b) The Social Security Act governs historical records and assistance "
        "is received under 42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_named_instrument_cannot_cross_unlisted_finite_coordination():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount under the Social Security Act
        referenced by 42 USC 1437f(o) is encoded.
rules: []
"""
    source = (
        "(b) The Social Security Act governs historical records and assistance "
        "qualifies under 42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize(
    "source",
    [
        "(b) The Social Security Act governs historical records and assistance "
        "may qualify under 42 USC 1437f(o).",
        "(b) The Social Security Act governs historical records and benefits "
        "qualify under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and an agency provides "
        "assistance under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and assistance remains "
        "available under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and benefits qualify "
        "independently under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and assistance existed "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and an agency processes "
        "applications under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and criteria qualify under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and data qualify under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and agencies process "
        "applications under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the Department of "
        "Agriculture processes applications under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the Secretary of "
        "Agriculture approves benefits under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the authority rules "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and agencies apply rules "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and agencies supply benefits "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the Internal Revenue "
        "Service processes applications under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the Social Security "
        "Administration processes applications under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the authority benefits "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and agencies misapply rules "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and agencies tally "
        "applications under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the agency made rules "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the United States "
        "Department of Agriculture processes applications under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the Department of "
        "Commerce processes applications under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and eligibility rules limit "
        "benefits under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and benefit programs limit "
        "assistance under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the agency records data "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the Office of Management "
        "and Budget approves benefits under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the agency input data "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the department input "
        "records under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and data limits benefits "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the office of management "
        "and budget approves benefits under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the OFFICE OF MANAGEMENT "
        "AND BUDGET approves benefits under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the program input data "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and OMB approves benefits "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and omb approves benefits "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and SSA determines benefits "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and USDA administers benefits "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and CMS determines eligibility "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the agency input the "
        "requirements under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the program input the data "
        "requirements under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the agency input applicable "
        "requirements under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and the program input federal "
        "requirements under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and FEMA administers benefits "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and FDA determines eligibility "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and FTC issues rules under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and DOJ enforces requirements "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and DOT administers programs "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and OPM administers benefits "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and SBA administers programs "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and CFPB issues rules under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and EEOC enforces requirements "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and CFTC issues rules under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and TSA administers programs "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and FHA determines eligibility "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and EBSA issues rules under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and SAMHSA administers programs "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and HRSA determines eligibility "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and OCC issues rules under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and FEC issues rules under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and TSA ran programs under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and FHA made eligibility "
        "determinations under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and EBSA wrote rules under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and SAMHSA gave assistance "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and HRSA chose eligibility "
        "criteria under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and OCC kept records under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and TSA and FHA administer "
        "programs under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and TSA, FHA, and EBSA "
        "administer programs under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and Tsa administers programs "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and tsa administers programs "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA ran programs under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and BIA made eligibility "
        "determinations under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NPS wrote rules under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and USGS gave information "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and EIA kept records under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and nhtsa administers programs "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA limits benefits under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA inputs data under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and BIA processes data under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and USGS records information "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and EIA rates programs under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA input the requirements "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and nhtsa limits benefits under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA oversaw programs under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and BIA sought information under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NPS saw benefits under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA records the policy "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA records this policy "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA benefits the program "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA rules the program "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA records its policy "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA records each policy "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA benefits every program "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA rules some program "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA limits any benefit "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA processes each record "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA rates all programs "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA oversaw enforcement "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and BIA sought approval under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NPS saw violations under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA processed applications "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA administers approval "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA records another policy "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA limits one benefit "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA processes most records "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA rates more programs "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA conditions eligibility "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA conditions program "
        "eligibility under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA and BIA condition "
        "eligibility under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and AmeriCorps administers "
        "programs under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and Federal Reserve determines "
        "rates under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and Federal Reserve issued rules "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA DETERMINES eligibility "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA APPROVES eligibility "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA ISSUES rules under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and Federal Bureau of "
        "Investigation administers programs under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and Federal Deposit Insurance "
        "Corporation issued rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and Federal Judiciary determines "
        "eligibility under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA based eligibility under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA based program "
        "eligibility under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and federal deposit insurance "
        "corporation issued rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and Federal deposit insurance "
        "Corporation issued rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and federal judiciary determines "
        "eligibility under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and federal communications "
        "commission issued rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and federal aviation "
        "administration issued rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and federal emergency management "
        "agency administers programs under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and federal housing finance "
        "agency determines eligibility under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and federal labor relations "
        "authority determines eligibility under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and TSA and FHA pass rules under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and SSA and VA pass eligibility "
        "under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and federal highway administrator "
        "determines eligibility under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and federal communications "
        "commissioner issued rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and federal treasury secretary "
        "issued rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and federal energy secretary "
        "determines eligibility under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and federal highway administrators "
        "determine eligibility under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and federal communications "
        "commissioners issue rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and federal treasury secretaries "
        "issue rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and federal housing agencies "
        "administer programs under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and SSI benefits pass rules under "
        "42 USC 1437f(o).",
        "(b) The Social Security Act governs records and CHIP programs pass eligibility "
        "rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and Medicare programs pace "
        "implementation rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and SNAP benefits based eligibility "
        "rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and Medicaid programs chip "
        "eligibility rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and SNAP benefits part eligibility "
        "rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA programs record "
        "eligibility rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA program records "
        "eligibility rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA programs work "
        "eligibility requirements under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and Medicaid programs record "
        "eligibility rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA programs manage "
        "eligibility rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA programs influence "
        "eligibility rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA programs condition "
        "eligibility rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA programs record "
        "retention policies under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA programs work "
        "participation requirements under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA programs manage "
        "retention policies under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA programs influence "
        "management procedures under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA programs wage payment "
        "rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA program conditions "
        "retention policies under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA program limits payment "
        "rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA program processes "
        "payment rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA program rates payment "
        "rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA program works "
        "participation requirements under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA program cashes payment "
        "rules under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA program schools "
        "participation requirements under 42 USC 1437f(o).",
        "(b) The Social Security Act governs records and NHTSA program shelters "
        "management procedures under 42 USC 1437f(o).",
    ],
)
def test_named_instrument_cannot_cross_other_finite_coordination(source: str):
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount under the Social Security Act
        referenced by 42 USC 1437f(o) is encoded.
rules: []
"""

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize(
    "object_phrase",
    [
        "assistance received",
        "benefits eligible",
        "assistance received directly",
        "benefits otherwise eligible",
        "benefits generally available",
        "assistance received recently",
        "benefits now available",
        "benefits not available",
        "agency procedures",
        "department policies",
        "agency requirements",
        "agency guidance",
        "department regulations",
        "benefit payment amounts",
        "rules applied",
        "assistance furnished",
        "benefits payable",
        "records management policies",
        "program administration records",
        "benefit calculation amounts",
        "rules authorized",
        "programs established",
        "benefits calculated",
        "assistance administered",
        "data input requirements",
        "payment processing rules",
        "records retention policies",
        "program participation requirements",
        "income verification procedures",
        "eligibility determined",
        "requirements implemented",
        "benefits verified",
        "information obtained",
        "program input requirements",
        "SNAP benefits",
        "SSI benefits",
        "TANF program",
        "USC rules",
        "SSA benefits",
        "VA benefits",
        "IRS rules",
        "SNAP benefits program",
        "TSA records retention policies",
        "EBSA benefits eligibility criteria",
        "FHA programs administration records",
        "SSA and VA benefit programs",
        "SSA and VA records retention policies",
        "SNAP work requirements",
        "TANF work participation requirements",
        "SSI resource limits",
        "USC amendment rules",
        "Medicaid waiver eligibility rules",
        "Medicare prescription benefit rules",
        "TSA benefits program",
        "Medicare Savings Program rules",
        "SNAP earnings requirements",
        "VA benefits eligibility",
        "IRS records policy",
        "Medicaid income eligibility",
        "SSI disability eligibility",
        "Medicare coverage rules",
        "SNAP training requirements",
        "TANF participation requirements",
        "VA disability eligibility",
        "SNAP student eligibility",
        "SNAP categorical eligibility",
        "SSI child eligibility",
        "Medicare premium rules",
        "Medicaid spenddown eligibility",
        "SNAP noncitizen eligibility",
        "SNAP immigrant eligibility",
        "Medicare drug rules",
        "Medicaid parent eligibility",
        "SSI blind eligibility",
        "Medicaid MAGI eligibility",
        "SNAP ABAWD eligibility",
        "Medicare Part D eligibility",
        "Medicaid HCBS eligibility",
        "SNAP EBT rules",
        "SSI SGA rules",
        "Medicare IRMAA rules",
        "Medicaid MAGI-based eligibility",
        "Medicaid non-MAGI eligibility",
        "Medicaid magi eligibility",
        "Medicare part d eligibility",
        "SNAP ebt rules",
        "Medicaid HCBS-waiver eligibility",
        "Medicaid CHIP eligibility",
        "Medicare QMB eligibility",
        "Medicare SLMB eligibility",
        "SNAP BBCE eligibility",
        "SNAP SUA rules",
        "SSI POMS rules",
        "TANF MOE requirements",
        "Medicare QI eligibility",
        "Medicare QDWI eligibility",
        "Medicare LIS rules",
        "Medicare MSP eligibility",
        "Medicaid EPSDT rules",
        "Medicaid LTSS eligibility",
        "SNAP HEA eligibility",
        "TANF SSP requirements",
        "SSI PASS rules",
        "Medicare SNP eligibility",
        "Medicare PACE eligibility",
        "Medicaid MCO rules",
        "Medicaid DSH rules",
        "Medicaid SPA rules",
        "SNAP LIEAP rules",
        "Medicare D-SNP eligibility",
        "CHIP MAGI eligibility",
        "CHIP FMAP rules",
        "SSDI SGA rules",
        "RSDI SGA rules",
        "Medicare HMO rules",
        "Medicare PPO rules",
        "Medicare PDP rules",
        "Medicare C-SNP eligibility",
        "SSI ISM rules",
        "Medicaid HIPP eligibility",
        "SNAP LIHEAP rules",
        "Medicare MA-PD rules",
        "Medicare I-SNP eligibility",
        "NHTSA program records retention policies",
        "NHTSA benefit rules administration procedures",
        "NHTSA policy records management procedures",
    ],
)
def test_named_instrument_can_cross_coordinated_object_modifier(object_phrase: str):
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount under the Social Security Act
        referenced by 42 USC 1437f(o) is encoded.
rules: []
"""
    source = (
        "(b) The Social Security Act governs historical records and "
        f"{object_phrase} under 42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


def test_named_instrument_accepts_relative_source_citation():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount under the Social Security Act
        referenced by 42 USC 1437f(o) is encoded.
rules: []
"""
    source = (
        "(b) The Social Security Act governs assistance under section 1437f(o) "
        "of this title."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


@pytest.mark.parametrize(
    "subject",
    [
        "historical eligibility data",
        "background-check status",
        "adjusted gross income",
        "household size",
        "residency requirement",
        "eligibility statuses",
        "administrative processes",
        "taxable income",
        "asset limit",
        "income and asset limit",
        "income or asset limit",
    ],
)
def test_dependency_subject_allows_operative_legal_terms(subject: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until {subject} under 42 USC 1437f(o) is provided.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


@pytest.mark.parametrize("linker", ["depends on", "requires"])
@pytest.mark.parametrize("framing", ["until", "because", "since"])
def test_dependency_subject_allows_strong_linker(linker: str, framing: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed {framing} benefit amount {linker} 42 USC 1437f(o) is
        verified.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


@pytest.mark.parametrize(
    "reason",
    [
        "Cannot be computed until benefit amount depends on 42 USC 1437f(o), "
        "but the citation is only nonbinding authority.",
        "Cannot be computed until benefit amount requires 42 USC 1437f(o) "
        "only for historical comparison.",
        "Cannot be computed until the nonbinding assertion requires "
        "benefit amount depends on 42 USC 1437f(o) to be verified.",
        "Cannot be computed until the nonbinding assertion requires "
        "benefit amount depends on 42 USC 1437f(o) is verified.",
    ],
)
def test_strong_linker_requires_bounded_introduction_and_tail(reason: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        {reason}
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


@pytest.mark.parametrize("linker", ["depends on", "requires"])
@pytest.mark.parametrize(
    "state",
    ["not yet encoded", "not yet available", "not  yet encoded"],
)
def test_strong_linker_accepts_not_yet_state(linker: str, state: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount {linker} 42 USC 1437f(o) is
        {state}.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


@pytest.mark.parametrize(
    "tail",
    [
        "is encoded, but the citation is only nonbinding authority",
        "is unavailable even though it is included only for historical comparison",
        "is encoded; but the citation is only nonbinding authority",
        "is encoded. However, the citation is only nonbinding authority",
    ],
)
@pytest.mark.parametrize("linker", ["under", "depends on", "requires"])
def test_dependency_state_rejects_adversative_tail(linker: str, tail: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount {linker} 42 USC 1437f(o)
        {tail}.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_contextual_prior_occurrence_cannot_launder_same_citation():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        For context, 42 USC 1437f(o) is included only as nonbinding authority.
        Cannot be computed until benefit amount depends on 42 USC 1437f(o) is
        encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_contextual_prior_occurrence_recognizes_merely_illustrative():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        42 USC 1437f(o) is merely illustrative. Cannot be computed until benefit
        amount depends on 42 USC 1437f(o) is encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_other_prior_citation_context_is_not_attributed_to_selected_citation():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        42 USC 1437f(o) is binding, while 7 USC 9999 is included only as
        nonbinding authority. Cannot be computed until benefit amount depends
        on 42 USC 1437f(o) is encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


def test_prior_citation_context_before_selected_citation_is_not_attributed():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        7 USC 9999 is included only as nonbinding authority, while 42 USC
        1437f(o) is binding. Cannot be computed until benefit amount depends on
        42 USC 1437f(o) is encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


def test_prior_citation_context_across_although_is_not_attributed():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        7 USC 9999 is included only as nonbinding authority, although 42 USC
        1437f(o) is binding. Cannot be computed until benefit amount depends on
        42 USC 1437f(o) is encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


@pytest.mark.parametrize("coordination", ["even though", "though"])
def test_prior_citation_context_across_though_is_not_attributed(coordination: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        7 USC 9999 is included only as nonbinding authority, {coordination} 42 USC
        1437f(o) is binding. Cannot be computed until benefit amount depends on
        42 USC 1437f(o) is encoded.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


def test_contextual_later_occurrence_is_found_after_other_citation():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount depends on 42 USC 1437f(o) is
        encoded. 7 USC 9999 is also required. However, 42 USC 1437f(o) is
        included only as nonbinding authority.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_contextual_later_relative_occurrence_disqualifies_dependency():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount depends on 42 USC 1437f(o) is
        encoded. However, section 1437f(o) of this title is included only as
        nonbinding authority.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert _has_issue(result, "(b)", "deferral", "runtime capability")


def test_relative_occurrence_from_current_title_does_not_match_external_title():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount depends on 7 USC 2014(a) is
        encoded. However, section 2014(a) of this title is included only as
        nonbinding authority.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "7 USC 2014(a)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


def test_later_independent_dependency_does_not_qualify_selected_citation():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount depends on 42 USC 1437f(o) is
        encoded. However, 7 USC 9999 is also required.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


def test_later_independent_rulespec_dependency_does_not_qualify_selected_citation():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed until benefit amount depends on 42 USC 1437f(o) is
        encoded. However, us:manuals/example#table is also required.
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o), and the applicable table appears in the example "
        "manual."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


def test_source_bound_usc_dependency_accepts_without_wording():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c-1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/b#annual_plan_requirement
      reason: >-
        Cannot be computed without the eligibility standard in 42 USC 1437f(o).
rules: []
"""
    source = (
        "(b) The annual plan applies when an agency receives assistance under "
        "42 USC 1437f(o)."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c-1",
        test_cases=[],
    )

    assert not _has_issue(result, "(b)", "deferral")


@pytest.mark.parametrize(
    "cited_path",
    ["(a)", "(a)(1)", "(a)(3)", "(a)(2)(i)"],
)
def test_source_bound_runtime_gap_requires_full_nested_branch(cited_path: str):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c–1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/a/2#agency_plan_contents
      reason: >-
        Cannot be computed until the agency policies document and submission
        event required by 42 USC 1437c-1{cited_path} are encoded.
rules: []
"""
    source = (
        "(a) Agency plans shall contain:\n"
        "1. goals;\n"
        "2. Agency shall submit plan policies."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c–1",
        test_cases=[],
    )

    assert _has_issue(result, "(a)", "deferral", "runtime capability")


def test_source_bound_runtime_gap_accepts_exact_nested_branch():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/42/1437c–1
  deferred_outputs:
    - output: us:statutes/42/1437c-1/a/2#agency_plan_contents
      reason: >-
        Cannot be computed until the agency policies document and submission
        event required by 42 USC 1437c-1(a)(2) are encoded.
rules: []
"""
    source = (
        "(a) Agency plans shall contain:\n"
        "1. goals;\n"
        "2. Agency shall submit plan policies."
    )

    result = _analyze(
        content,
        source,
        corpus_citation_path="us/statute/42/1437c–1",
        test_cases=[],
    )

    assert not _has_issue(result, "(a)", "deferral")


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
        "Der Betrag beträgt das Einkommen plus 10 Euro.",
        "Der Betrag beträgt das Einkommen minus 10 Euro.",
        "Der Betrag beträgt 10 % des Einkommens.",
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
        ("(1) Der Betrag beträgt das Einkommen plus 10 Euro.", 10),
        ("(1) Der Betrag beträgt das Einkommen minus 10 Euro.", 10),
        ("(1) Der Betrag beträgt 10 % des Einkommens.", 0.1),
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


@pytest.mark.parametrize("result_phrase", ["ergibt sich", "ergeben sich"])
def test_stated_conversion_result_is_not_a_formula_mandate(result_phrase: str):
    source = (
        "Der Jahresbetrag beträgt 73 800 Euro. "
        f"Umgerechnet auf den Monat {result_phrase} 6 150 Euro."
    )

    assert source_states_stated_conversion_result(source)
    assert not source_states_explicit_computation(source)


@pytest.mark.parametrize(
    "source",
    [
        "Der Monatsbetrag ergibt sich aus dem Jahresbetrag geteilt durch 12.",
        "Der Jahresbetrag beträgt das 12-Fache des Monatsbetrags.",
        (
            "Der Jahresbetrag beträgt 73 800 Euro. Umgerechnet auf den Monat "
            "ergibt sich aus dem Jahresbetrag geteilt durch 12 ein Monatsbetrag "
            "von 6 150 Euro."
        ),
        (
            "The annual amount is 73,800 dollars. Converted to a monthly amount, "
            "it is the annual amount divided by 12, or 6,150 dollars."
        ),
        (
            "Der Jahresbetrag beträgt 73 800 Euro. "
            "Umgerechnet auf den Monat ergibt sich 6 150 Euro. "
            "Der Zuschlag ergibt sich aus dem Monatsbetrag plus 10 Euro."
        ),
    ],
)
def test_stated_conversion_exemption_preserves_actual_formulas(source: str):
    assert source_states_explicit_computation(source)


def test_stated_conversion_pair_scans_past_trailing_year():
    source = (
        "Der Jahresbetrag beträgt 73 800 Euro im Jahr 2025. "
        "Umgerechnet auf den Monat ergibt sich 6 150 Euro."
    )

    assert source_states_stated_conversion_result(source)
    assert not source_states_explicit_computation(source)


@pytest.mark.parametrize(
    "source",
    (
        "Nach § 12 umgerechnet auf den Monat ergibt sich 6 150 Euro.",
        "Nach den §§ 10 bis 12 umgerechnet auf den Monat ergibt sich 6 150 Euro.",
        "Nach den §§ 10–12 umgerechnet auf den Monat ergibt sich 6 150 Euro.",
        "Nach Artikel 10 bis 12 umgerechnet ergibt sich 6 150 Euro.",
        "Nach Art. 10 bis 12 umgerechnet ergibt sich 6 150 Euro.",
        "Under Articles 10 to 12, converted monthly, it is 6150 dollars.",
        "Am 1. Januar 2025 gilt: Umgerechnet auf den Monat ergibt sich 6 150 Euro.",
        "Ab 2025-01-01 gilt: Umgerechnet auf den Monat ergibt sich 6 150 Euro.",
        "(1) Umgerechnet auf den Monat ergibt sich 6 150 Euro.",
    ),
)
def test_stated_conversion_pair_rejects_structural_number_as_base(source: str):
    assert not source_states_stated_conversion_result(source)


@pytest.mark.parametrize(
    "source",
    (
        (
            "Der Jahresbetrag beträgt 73 800 Euro. Umgerechnet auf den Monat "
            "ergibt sich aus dem Jahresbetrag geteilt durch 12 ein Monatsbetrag "
            "von 6 150 Euro."
        ),
        (
            "Der Jahresbetrag beträgt 73 800 Euro. Umgerechnet auf den Monat "
            "ergibt sich 1/12 des Jahresbetrags, nämlich 6 150 Euro."
        ),
        (
            "Der Monatsbetrag beträgt 6 150 Euro. Umgerechnet auf das Jahr "
            "ergibt sich das 12-Fache des Monatsbetrags, also 73 800 Euro."
        ),
        (
            "The monthly amount is 6,150 dollars. Converted to an annual amount, "
            "it is 12 times the monthly amount, or 73,800 dollars."
        ),
    ),
)
def test_same_clause_stated_conversion_formula_requires_derived_output(source: str):
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: annual_amount
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a
    versions:
      - effective_from: '2025-01-01'
        formula: 73800
  - name: monthly_amount
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a
    versions:
      - effective_from: '2025-01-01'
        formula: 6150
  - name: months_per_year
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a
    versions:
      - effective_from: '2025-01-01'
        formula: 12
"""

    result = _analyze(content, source, test_cases=[])

    assert _has_issue(result, "formula-output", "parameter-only")
    assert any(
        "formula-output" in issue and "parameter-only" in issue
        for issue in _pipeline_issues(content, source, test_cases=[])
    )


def test_stated_conversion_pair_stops_formula_guard_at_semicolon():
    source = (
        "Der Jahresbetrag beträgt 73 800 Euro. Umgerechnet auf den Monat ergibt "
        "sich 6 150 Euro; der Zuschlag ergibt sich aus dem Monatsbetrag plus "
        "10 Euro."
    )

    assert source_states_stated_conversion_result(source)
    assert source_states_explicit_computation(source)


def test_stated_conversion_metadata_filter_retains_real_amounts():
    source = (
        "Nach § 12 beträgt der Jahresbetrag 73 800 Euro im Jahr 2025. "
        "Umgerechnet auf den Monat ergibt sich 6 150 Euro."
    )
    year_sized_amount = (
        "Der Jahresbetrag beträgt 2 025 Euro. Umgerechnet auf den Monat ergibt "
        "sich 168,75 Euro."
    )

    assert source_states_stated_conversion_result(source)
    assert not source_states_explicit_computation(source)
    assert source_states_stated_conversion_result(year_sized_amount)
    assert not source_states_explicit_computation(year_sized_amount)


@pytest.mark.parametrize(
    "source",
    (
        "Nach den §§ 10 bis 12 umgerechnet auf den Monat ergeben sich 6 150 Euro.",
        "Nach den §§ 10–12 umgerechnet auf den Monat ergeben sich 6 150 Euro.",
        "Nach Artikel 10 bis 12 umgerechnet ergeben sich 6 150 Euro.",
        "Nach Art. 10 bis 12 umgerechnet ergeben sich 6 150 Euro.",
        "Under Articles 10 to 12, converted monthly, it is 6150 dollars.",
        "Am 1. Januar 2025 gilt: Umgerechnet auf den Monat ergeben sich 6 150 Euro.",
        "Ab 2025-01-01 gilt: Umgerechnet auf den Monat ergeben sich 6 150 Euro.",
    ),
)
def test_metadata_only_numbers_do_not_activate_stated_conversion_hint(source: str):
    citation_path = "de/regulation/example/1"
    content = _stated_conversion_rulespec_with_divisor(
        citation_path,
        (("monthly_amount", "6150"),),
        annual_parameter="annual_amount",
        divisor="52",
    )

    assert find_ungrounded_numeric_issues(
        content,
        source_text=source,
        source_citation_path=citation_path,
        require_complete_source_unit=True,
    ) == [
        "Ungrounded generated numeric literal: 52 does not appear as a "
        "substantive numeric value in the source text."
    ]


def test_scalar_amount_language_is_not_computation():
    assert not source_states_explicit_computation("Der Freibetrag beträgt 259 Euro.")
    assert not source_states_explicit_computation("Der Satz beträgt 45 vom Hundert.")


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
        "de/statute/estg/32a(1) Satz 1; de/statute/estg/32a(1) Satz 2",
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


RELEASED_UHVORSCHG_2_ABSATZ_4 = """\
(4) Für Berechtigte, die keine allgemeinbildende Schule mehr besuchen, mindert sich die nach den Absätzen 1 bis 3 ergebende Unterhaltsleistung, soweit ihre in demselben Monat erzielten Einkünfte des Vermögens und der Ertrag ihrer zumutbaren Arbeit zum Unterhalt ausreichen. Als Ertrag der zumutbaren Arbeit des Berechtigten aus nichtselbstständiger Arbeit gelten die Einnahmen in Geld entsprechend der für die maßgeblichen Monate erstellten Lohn- und Gehaltsbescheinigungen des Arbeitgebers abzüglich eines Zwölftels des Arbeitnehmer-Pauschbetrags; bei Auszubildenden sind zusätzlich pauschal 100 Euro als ausbildungsbedingter Aufwand abzuziehen. Einkünfte und Erträge nach den Sätzen 1 und 2 sind nur zur Hälfte zu berücksichtigen."""


def test_released_uhvorschg_word_denominator_is_grounding_not_scalar_recall():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/uhvorschg/2
rules:
  - name: apprenticeship_expense_deduction
    kind: parameter
    dtype: Money
    source: de/statute/uhvorschg/2(4) Satz 2
    versions:
      - effective_from: '2026-01-01'
        formula: 100
  - name: employment_income_after_deductions
    kind: derived
    dtype: Money
    source: de/statute/uhvorschg/2(4) Satz 2
    versions:
      - effective_from: '2026-01-01'
        formula: employment_income - employee_lump_sum / 12 - apprenticeship_expense_deduction
"""

    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        RELEASED_UHVORSCHG_2_ABSATZ_4,
        profile="de-DE",
    )
    result = _analyze(
        content,
        RELEASED_UHVORSCHG_2_ABSATZ_4,
        corpus_citation_path="de/statute/uhvorschg/2",
        test_cases=[],
    )
    grounding_issues = find_ungrounded_numeric_issues(
        content,
        source_text=RELEASED_UHVORSCHG_2_ABSATZ_4,
        source_citation_path="de/statute/uhvorschg/2",
        require_complete_source_unit=True,
    )

    assert [(occurrence.value, occurrence.raw) for occurrence in inventory] == [
        (100.0, "100")
    ]
    assert not _has_issue(result, "numeric-recall", "value 12")
    assert not any(
        "Ungrounded generated numeric literal: 12" in issue
        for issue in grounding_issues
    )


def test_unused_word_denominator_is_silent_for_scalar_recall():
    result = _analyze(
        "format: rulespec/v1\nrules: []\n",
        "Der Pauschbetrag wird um ein Zwölftel gekürzt.",
        corpus_citation_path="de/statute/uhvorschg/2",
        test_cases=[],
    )

    assert not _has_issue(result, "numeric-recall", "value 12")


def test_numeric_money_literal_still_requires_named_scalar_when_used_in_formula():
    source = "Der Abzug beträgt 12 Euro."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/uhvorschg/2
rules:
  - name: amount_after_deduction
    kind: derived
    dtype: Money
    source: de/statute/uhvorschg/2
    versions:
      - effective_from: '2026-01-01'
        formula: amount / 12
"""

    result = _analyze(
        content,
        source,
        corpus_citation_path="de/statute/uhvorschg/2",
        test_cases=[],
    )

    assert (
        find_ungrounded_numeric_issues(
            content,
            source_text=source,
            source_citation_path="de/statute/uhvorschg/2",
            require_complete_source_unit=True,
        )
        == []
    )
    assert _has_issue(result, "numeric-recall", "value 12")


RELEASED_RBEG_2021_8_BODY = """\
Die Regelbedarfsstufen nach der Anlage zu § 28 des Zwölften Buches Sozialgesetzbuch belaufen sich zum 1. Januar 2021
1. in der Regelbedarfsstufe 1 auf 446 Euro für jede erwachsene Person, die in einer Wohnung nach § 42a Absatz 2 Satz 2 des Zwölften Buches Sozialgesetzbuch lebt und für die nicht Nummer 2 gilt,
2. in der Regelbedarfsstufe 2 auf 401 Euro für jede erwachsene Person, die
a) in einer Wohnung nach § 42a Absatz 2 Satz 2 des Zwölften Buches Sozialgesetzbuch mit einem Ehegatten oder Lebenspartner oder in eheähnlicher oder lebenspartnerschaftsähnlicher Gemeinschaft mit einem Partner zusammenlebt oder
b) nicht in einer Wohnung lebt, weil ihr allein oder mit einer weiteren Person ein persönlicher Wohnraum und mit weiteren Personen zusätzliche Räumlichkeiten nach § 42a Absatz 2 Satz 3 des Zwölften Buches Sozialgesetzbuch zur gemeinschaftlichen Nutzung überlassen sind,
3. in der Regelbedarfsstufe 3 auf 357 Euro für eine erwachsene Person, deren notwendiger Lebensunterhalt sich nach § 27b des Zwölften Buches Sozialgesetzbuch bestimmt (Unterbringung in einer stationären Einrichtung),
4. in der Regelbedarfsstufe 4 auf 373 Euro für eine Jugendliche oder einen Jugendlichen vom Beginn des 15. bis zur Vollendung des 18. Lebensjahres,
5. in der Regelbedarfsstufe 5 auf 309 Euro für ein Kind vom Beginn des siebten bis zur Vollendung des 14. Lebensjahres und
6. in der Regelbedarfsstufe 6 auf 283 Euro für ein Kind bis zur Vollendung des sechsten Lebensjahres."""


def test_released_rbeg_stage_and_reference_labels_stay_out_of_numeric_recall():
    assert (
        hashlib.sha256(RELEASED_RBEG_2021_8_BODY.encode()).hexdigest()
        == "0783343f3ce3c3b691ea8bee3047e3a818af42b0da4902c6ff3b28c372e4a964"
    )
    grounding = extract_typed_numeric_occurrences_from_text(
        RELEASED_RBEG_2021_8_BODY,
        profile="de-DE",
    )
    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        RELEASED_RBEG_2021_8_BODY,
        profile="de-DE",
    )

    assert any(
        occurrence.raw == "1" and occurrence.has_structural_context
        for occurrence in grounding
    )
    assert [occurrence.value for occurrence in inventory] == [
        446.0,
        401.0,
        357.0,
        373.0,
        15.0,
        18.0,
        309.0,
        14.0,
        283.0,
    ]

    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/rbeg-2021/8
rules: []
"""
    result = _analyze(
        content,
        RELEASED_RBEG_2021_8_BODY,
        corpus_citation_path="de/statute/rbeg-2021/8",
        artifact_numeric_values=tuple(occurrence.value for occurrence in inventory),
    )

    assert not _has_issue(result, "numeric-recall")


@pytest.mark.parametrize(
    "source_text",
    (
        "Die Leistung richtet sich nach Stufe 1.",
        "Die Einzelheiten stehen in Anlage 1.",
        "Die Verweisung lautet §1 Absatz 2 Satz 3 Nummer 4.",
    ),
)
def test_structural_labels_are_typed_grounding_only(source_text):
    grounding = extract_typed_numeric_occurrences_from_text(
        source_text,
        profile="de-DE",
    )
    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        source_text,
        profile="de-DE",
    )

    assert grounding
    assert all(occurrence.has_structural_context for occurrence in grounding)
    assert not inventory


@pytest.mark.parametrize(
    "source_text",
    (
        "The details appear in regs. 123.45.",
        "Use the 2nd digit and inspect the 3rd digit.",
    ),
)
def test_english_structural_labels_are_typed_grounding_only(source_text):
    grounding = extract_typed_numeric_occurrences_from_text(
        source_text,
        profile="legacy",
    )
    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        source_text,
        profile="legacy",
    )

    assert grounding
    assert all(occurrence.has_structural_context for occurrence in grounding)
    assert not inventory


def test_structural_range_markers_stay_out_of_numeric_recall():
    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        "(1) bis (3) gelten entsprechend.",
        profile="de-DE",
    )

    assert not inventory


def test_structural_stage_label_does_not_exempt_one_euro_from_numeric_recall():
    source = "Regelbedarfsstufe 1: Die Leistung beträgt 1 Euro."
    grounding = extract_typed_numeric_occurrences_from_text(source, profile="de-DE")
    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        source,
        profile="de-DE",
    )

    assert [
        (occurrence.value, occurrence.has_structural_context)
        for occurrence in grounding
    ] == [
        (1.0, True),
        (1.0, False),
    ]
    assert [(occurrence.value, occurrence.raw) for occurrence in inventory] == [
        (1.0, "1")
    ]

    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/rbeg-2021/8
rules: []
"""
    result = _analyze(
        content,
        source,
        corpus_citation_path="de/statute/rbeg-2021/8",
        artifact_numeric_values=(),
    )

    assert _has_issue(result, "numeric-recall", "value 1")


def test_en_us_state_code_citations_are_structural_for_numeric_recall():
    source = (
        "The credit applies against tax due under (N.J.S.54A:1-1) and "
        "N.J.S.A. 54A:9-7, subject to C.54A:4-6 and section 54A:4-7. "
        "C.54:4-8.57 and “R.S.43:21-1” also apply. "
        "Sections 54:4-8.57 and 43:21-1 provide the same references. "
        "'N.J.A.C. 10:90-3.8(b)1' and N.J.A.C. 10:90-3.9(d)5i control. "
        "N.J.S.54A:4-7—this controls; R.S.43:21-1–as amended."
    )

    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        source,
        profile="en-US",
    )

    assert inventory == []


def test_louisiana_revised_statutes_citations_are_structural_for_numeric_recall():
    source = (
        "The amount is determined under R.S. 47:32. Notwithstanding R.S.\n"
        "47:1508, a 25,000 dollar threshold applies, with a separate 2:1 ratio."
    )

    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        source,
        profile="en-US",
    )

    assert [(item.value, item.raw) for item in inventory] == [
        (25000.0, "25,000"),
        (2.0, "2"),
        (1.0, "1"),
    ]


def test_line_wrapped_louisiana_cross_reference_stays_in_one_clause():
    source = """\
Notwithstanding the provisions of R.S.
47:1508, beginning January 1, 2016, waivers of all penalties exceeding twenty-five
thousand dollars shall be subject to oversight. This provision shall not apply to a
penalty waived under the voluntary disclosure program.
"""

    clauses = tuple(completeness_module._source_clause_spans(source, branches=()))
    exception_branches = completeness_module._source_exception_branches(
        source,
        branches=(),
        active_branches=(),
        deferred_paths=set(),
        formula_branches=(),
    )

    assert any(
        "R.S.\n47:1508" in clause and "subject to oversight" in clause
        for _start, _end, clause in clauses
    )
    assert [
        completeness_module._source_exception_requires_paired_witness(branch.text)
        for branch in exception_branches
    ] == [True, True]
    assert [
        (occurrence.raw, occurrence.value)
        for occurrence in EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR(source)
        if occurrence.is_word_number
    ] == [("twenty-five\nthousand", 25000.0)]


@pytest.mark.parametrize(
    "reference",
    (
        "R.S. 47:1508",
        "provision of R.S. 47:1508",
        "the provisions of R.S. 47:1508",
    ),
)
def test_pure_louisiana_notwithstanding_reference_is_not_toggleable(
    reference: str,
):
    source = f"Notwithstanding {reference}, the reporting rule applies."

    assert not completeness_module._source_exception_requires_paired_witness(source)


@pytest.mark.parametrize(
    "reference",
    ("R.S. 47:1508", "the provisions of R.S. 47:1508"),
)
@pytest.mark.parametrize(
    "tail",
    (
        "if income exceeds 25000 dollars, the credit applies",
        "when voluntary disclosure is false, oversight applies",
        "subject to an income limit, the credit applies",
    ),
)
def test_louisiana_notwithstanding_reference_preserves_runtime_condition(
    reference: str,
    tail: str,
):
    source = f"Notwithstanding {reference}, {tail}."

    assert completeness_module._source_exception_requires_paired_witness(source)


@pytest.mark.parametrize(
    "source",
    (
        "twenty thirty dollars",
        "one one dollars",
        "five thousand hundred dollars",
        "one million thousand dollars",
    ),
)
def test_en_us_compound_word_grounding_rejects_malformed_order(source: str):
    assert not any(
        occurrence.is_word_number
        for occurrence in EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR(source)
    )


def test_en_us_compound_word_grounding_preserves_coordinated_values():
    assert [
        (occurrence.raw, occurrence.value)
        for occurrence in EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR(
            "between five and ten years"
        )
        if occurrence.is_word_number
    ] == [("five", 5.0), ("ten", 10.0)]


def test_louisiana_cross_reference_does_not_require_synthetic_case_pair():
    source = """\
Notwithstanding the provisions of R.S.
47:1508, beginning January 1, 2016, waivers of all penalties exceeding twenty-five
thousand dollars shall be subject to oversight. This provision shall not apply to a
penalty waived under the voluntary disclosure program.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-la/statute/47:295
rules:
  - name: penalty_waiver_oversight_threshold
    kind: parameter
    dtype: Money
    unit: USD
    period: Day
    source: us-la/statute/47:295
    versions:
      - formula: 25000
  - name: penalty_waiver_subject_to_oversight
    kind: derived
    dtype: Judgment
    period: Day
    source: us-la/statute/47:295
    versions:
      - formula: penalty_amount > penalty_waiver_oversight_threshold and not voluntary_disclosure_program
"""
    cases = [
        {
            "name": "above threshold",
            "input": {
                "penalty_amount": 26000,
                "voluntary_disclosure_program": False,
            },
            "output": {"penalty_waiver_subject_to_oversight": True},
        },
        {
            "name": "at threshold",
            "input": {
                "penalty_amount": 25000,
                "voluntary_disclosure_program": False,
            },
            "output": {"penalty_waiver_subject_to_oversight": False},
        },
        {
            "name": "voluntary disclosure exception",
            "input": {
                "penalty_amount": 26000,
                "voluntary_disclosure_program": True,
            },
            "output": {"penalty_waiver_subject_to_oversight": False},
        },
    ]

    result = analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path="us-la/statute/47:295",
        test_cases=cases,
        extract_numeric_occurrences=EN_NUMERIC_OCCURRENCE_EXTRACTOR,
        extract_numeric_grounding_occurrences=(
            EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )

    assert not _has_issue(result, "exception", "test")
    assert not _has_issue(result, "numeric-recall")

    missing_threshold_pair = analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path="us-la/statute/47:295",
        test_cases=(cases[0], cases[2]),
        extract_numeric_occurrences=EN_NUMERIC_OCCURRENCE_EXTRACTOR,
        extract_numeric_grounding_occurrences=(
            EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )

    assert _has_issue(missing_threshold_pair, "exception", "test")
    assert any(
        "exceeding twenty-five" in issue for issue in missing_threshold_pair.issues
    )


def test_en_us_state_code_citation_filter_preserves_real_unit_and_ratio_values():
    source = (
        "Under N.J.S.54A:1-1, the amount is 1 dollar, the required ratio is 2:1, "
        "outline A. uses the ratio 10:2, the adjusted ratio is 10:2-3, "
        "outline C. uses 10:2-3, the score range is 10:20-30, "
        "and dose schedule 10mg:2-3 applies."
    )

    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        source,
        profile="en-US",
    )

    assert [(item.value, item.raw) for item in inventory] == [
        (1.0, "1"),
        (2.0, "2"),
        (1.0, "1"),
        (10.0, "10"),
        (2.0, "2"),
        (10.0, "10"),
        (2.0, "2"),
        (3.0, "3"),
        (10.0, "10"),
        (2.0, "2"),
        (3.0, "3"),
        (10.0, "10"),
        (20.0, "20"),
        (30.0, "30"),
        (2.0, "2"),
        (3.0, "3"),
    ]


def test_en_us_post_code_section_marker_citations_are_structural():
    source = (
        "Eligible under section 32 of the federal code (26 U.S.C. s.32) and "
        "section 152 of that code (26 U.S.C. sec. 152), with "
        "26 U.S.C. s.32—applying and 26 U.S.C. s.32–as amended, while "
        "26 dollars is due."
    )

    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        source,
        profile="en-US",
    )

    assert [(item.value, item.raw) for item in inventory] == [(26.0, "26")]


def test_en_us_internal_revenue_code_edition_year_is_structural():
    source = (
        "Eligible under section 32 of the federal Internal Revenue Code of 1986 "
        "(26 U.S.C. s.32), with a 1986 dollar threshold in taxable year 1986."
    )

    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        source,
        profile="en-US",
    )

    assert [(item.value, item.raw) for item in inventory] == [(1986.0, "1986")]


@pytest.mark.parametrize("profile", ("legacy", "en-US"))
def test_named_act_year_is_structural_but_equal_amount_is_preserved(profile):
    source = (
        "The certification must conform to title VI of the Civil Rights Act of "
        "1964, while a separate program has a 1964 dollar threshold."
    )

    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        source,
        profile=profile,
    )

    assert [(item.value, item.raw) for item in inventory] == [(1964.0, "1964")]


@pytest.mark.parametrize("profile", ("legacy", "en-US"))
@pytest.mark.parametrize(
    "source",
    (
        "A household qualifies when its category has a code of 2000.",
        "A household qualifies based on an act of 2000.",
    ),
)
def test_common_noun_act_or_code_year_remains_substantive(source, profile):
    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        source,
        profile=profile,
    )

    assert [(item.value, item.raw) for item in inventory] == [(2000.0, "2000")]


def test_en_us_inline_legal_ordinal_is_structural_after_collapsed_heading():
    source = (
        "54A:4-7 New Jersey Earned Income Tax Credit program. "
        "54A:4-7 New Jersey Earned Income Tax Credit program . "
        "2. There is established the New Jersey Earned Income Tax Credit program. "
        "The filing fee is 2 dollars."
    )

    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        source,
        profile="en-US",
    )

    assert [(item.value, item.raw) for item in inventory] == [(2.0, "2")]


@pytest.mark.parametrize(
    "history_tail",
    (
        "L.2000, c.80, s.2; amended 2007, c.109; 2020, c.98; 2021, c.130.",
        "\nL.2000, c.80, s.2; amended by L.2007, c.109; 2021, c.130.",
        "History: 2020, c.98.",
        "Source.— L.2020, c.98.",
        "L.2000, c.80; as amended L.2021, c.130.",
        "\n\nL.2000,c.80,s.1.",
        "\n\nL.1976, c.47, s.54A:1-1, eff. July 8, 1976, operative Aug. 30, 1976.",
        "\n\namended 1982, c.229, s.1; 2020, c.94, s.1.",
        "\n\nL.1976, c.47, s.54A:9-14, eff. July 8, 1976. "
        "Amended by L.1983, c.36, s.49, eff. Jan. 26, 1983.",
        "\n\nL. 1976, c.47, s.54A:9-14, eff. July 8, 1976. "
        "Amended by L. 1983, c.36, s.49, eff. Jan. 26, 1983.",
        "\n\nP. L. 1976, c.73, s.3. Amended by L. 1978, c.66, s.1, eff. July 3, 1978.",
        "\n\namended 1977, c.40, s.1; 1998, c.57, s.1. "
        "2017, c.313, s.5; 2018, c.131, s.8.",
    ),
)
def test_terminal_session_law_history_is_not_numeric_recall(history_tail):
    operative = "The credit is 40% for 12 months."

    cleaned = authoritative_numeric_recall_text(f"{operative} {history_tail}")

    assert cleaned == operative


def test_louisiana_session_law_citations_are_not_numeric_recall_values():
    source = """\
A. A standard deduction shall be allowed in determining a taxpayer's tax liability
pursuant to this Part. Taxpayers are required to use the same filing status on their return
required to be filed under this Part as they used on their federal income tax return. For tax
year 2025, the amount of the standard deduction shall be as follows:

(1) Single Individual and Married-Separate $12,500.00

(2) Married-Joint Return, a Qualified Surviving 200% of the dollar amount

Spouse, and Head of Household provided for Single Individuals

B. Beginning January 1, 2026, and thereafter, the amount of the standard deduction
provided in Subsection A of this Section shall be adjusted annually by an amount calculated
by multiplying the amount of the prior year's standard deduction by the percentage increase
in the Consumer Price Index United States city average for all urban consumers (CPI-U), as
reported by the United States Department of Labor, Bureau of Labor Statistics, or its
successor, for the previous calendar year.

Acts 1980, No. 316, §1. Acts 1983, 2nd Ex. Sess., No. 1, §1, eff. Dec. 19,
1983; Acts 2024, 3rd Ex. Sess., No. 11, §2, eff. Dec. 4, 2024.

{{NOTE: SECTION 4 OF ACTS 1983, 2ND EX. SESS., NO. 1,
PROVIDES AS FOLLOWS: "THE PROVISIONS OF THIS ACT SHALL
BE APPLICABLE TO TAXABLE YEARS BEGINNING AFTER
DECEMBER 31, 1982. FOR TAXABLE YEARS BEGINNING PRIOR TO
JANUARY 1, 1983, THE TAX SHALL BE AS REQUIRED BY LAW
PRIOR TO THE EFFECTIVE DATE OF THIS ACT."}}
"""

    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        authoritative_numeric_recall_text(source),
        profile="en-US",
    )

    assert [(item.value, item.raw) for item in inventory] == [
        (12500.0, "12,500.00"),
        (200.0, "200"),
    ]


@pytest.mark.parametrize(
    "citation",
    (
        "Acts 2024, 3rd Ex. Sess., No. 11, §2",
        "Acts 1983, 2d Ex. Sess., No. 1, §1",
        "Acts 2000, 2d Ex.Sess., No. 21, §1",
        "Acts 1950, 2nd Ex.Sess., No. 11, §2",
        "Acts 1973, Ex.Sess., No. 8, §1",
        "Acts 2016, 1 st Ex. Sess., No.\n29, §2",
        "Acts 1977, 1st Ex. Sess. No. 2, §1",
        "Acts 2024, Third Ex. Sess., No. 11, §§2, 4",
        "Acts 2002, No. 51, §§1 and 2",
        "Acts 1997, No.\n129, §1",
        "Acts 1995, No. 95-255, §1",
    ),
)
def test_louisiana_session_law_citation_filter_preserves_operative_value(
    citation: str,
):
    source = f"{citation} establishes an 11 dollar fee."

    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        authoritative_numeric_recall_text(source),
        profile="en-US",
    )

    assert [(item.value, item.raw) for item in inventory] == [(11.0, "11")]


@pytest.mark.parametrize("connector", ("and", "through", "to", "-"))
def test_louisiana_singular_section_citation_preserves_adjacent_amount(
    connector: str,
):
    source = f"Under Acts 2024, No. 11, §2 {connector} 50 dollars shall be paid."

    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        authoritative_numeric_recall_text(source),
        profile="en-US",
    )

    assert [(item.value, item.raw) for item in inventory] == [(50.0, "50")]


@pytest.mark.parametrize(
    "references",
    (
        "under z-4 of this title and pursuant to z–3 of this title",
        "under sections 1437z-3 and 1437z-4 of this title",
        "under section 1437z-3 or 1437z-4 of this title",
        "under sections 1437z-3, 1437z-4, and 1437z-5 of this title",
        "under sections 1437z-3 through 1437z-5 of this title",
        "under sections 1437z-3 to 1437z-5 of this title",
    ),
)
def test_us_title_suffix_cross_references_are_not_numeric_recall_values(
    references: str,
):
    source = f"A program operates {references}. The agency must provide 45 days notice."

    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        authoritative_numeric_recall_text(source),
        profile="en-US",
    )

    assert [(item.value, item.raw) for item in inventory] == [(45.0, "45")]


@pytest.mark.parametrize(
    "source_text",
    (
        "The credit is 40% for taxable year 2020 and the amount is $2,020.",
        "The governing act is L.2020, c.98.",
        "The governing acts are L.2020, c.98; 2021, c.130.",
        "L.2020, c.98; amended prose does not identify another session law.",
        "L.2020, c.98; amended 2021, c.130. Operative text follows.",
    ),
)
def test_numeric_recall_preserves_non_history_session_law_text(source_text):
    assert authoritative_numeric_recall_text(source_text) == source_text


@pytest.mark.parametrize(
    "terminal_block",
    (
        "L.2020, c.98. The same block creates a $500 refundable credit.",
        "L.2020, c.98 governs only citations, but taxable year 2024 is operative.",
        "L.2020, c.98; malformed tail; the benefit is 12 months.",
        "History: 2020, c.98. Operative amount: $500.",
    ),
)
def test_blank_line_history_leader_does_not_hide_operative_text(terminal_block):
    source = f"The credit is 40%.\n\n{terminal_block}"

    assert authoritative_numeric_recall_text(source) == source


def test_formula_branch_interval_reads_range_after_chapeau_colon():
    text = "The amount shall be: for income up to 100 dollars, income * 3"
    branch = completeness_module.SourceStructureBranch(
        path=("1",),
        kind="paragraph",
        label="1.",
        text=text,
        start=0,
        end=len(text),
    )

    interval = completeness_module._formula_branch_interval(
        branch,
        extract_numeric_occurrences=EN_NUMERIC_OCCURRENCE_EXTRACTOR,
    )

    assert interval is not None
    assert interval.lower is None
    assert interval.upper is not None
    assert interval.upper.value == 100
    assert interval.upper_inclusive is True


def test_formula_branch_interval_ignores_formula_constant_after_range():
    text = "Between 10 and 20 dollars: amount * 5"
    branch = completeness_module.SourceStructureBranch(
        path=("1",),
        kind="paragraph",
        label="1.",
        text=text,
        start=0,
        end=len(text),
    )

    interval = completeness_module._formula_branch_interval(
        branch,
        extract_numeric_occurrences=EN_NUMERIC_OCCURRENCE_EXTRACTOR,
    )

    assert interval is not None
    assert interval.lower is not None
    assert interval.lower.value == 10
    assert interval.upper is not None
    assert interval.upper.value == 20


def test_formula_selector_supports_parenthesized_multiline_continuation():
    selector = "eligible\nand resident\nand not disqualified"

    assert (
        completeness_module._evaluate_formula_selector(
            selector,
            {"eligible": True, "resident": True, "disqualified": False},
        )
        is True
    )
    assert (
        completeness_module._evaluate_formula_selector(
            selector,
            {"eligible": True, "resident": False, "disqualified": False},
        )
        is False
    )


def test_formula_selector_keeps_adjacent_multiline_statements_unresolved():
    result = completeness_module._evaluate_formula_selector(
        "eligible\nresident",
        {"eligible": True, "resident": True},
    )

    assert result is completeness_module._UNRESOLVED_CONDITION_VALUE


@pytest.mark.parametrize(
    ("heading", "scalar", "unit"),
    [
        ("Filing fee notice", 2, "Dollars are due"),
        ("Applicable rate notice", 25, "Percent applies"),
        ("Deadline notice", 30, "Days are allowed"),
    ],
)
def test_en_us_inline_scalar_after_single_nj_heading_is_not_a_legal_ordinal(
    heading: str,
    scalar: int,
    unit: str,
):
    source = f"54A:4-7 {heading}. {scalar}. {unit}."

    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        source,
        profile="en-US",
    )

    assert [(item.value, item.raw) for item in inventory] == [
        (float(scalar), str(scalar))
    ]


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

    imported_symbol_contents = pipeline._complete_source_unit_import_symbol_contents(
        main_file
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

    imported_symbol_contents = pipeline._complete_source_unit_import_symbol_contents(
        main_file
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
        branch.path for branch in recognize_source_structure(RELEASED_ESTG_32A_BODY)
    }

    assert {("1",), ("5",), ("6",)} <= paths
    assert ("2",) not in paths
    assert {("1", str(number)) for number in range(1, 6)} <= paths
    assert {("6", "1"), ("6", "2")} <= paths
    assert {("6", "2", letter) for letter in ("a", "b", "c")} <= paths
    assert {("1", f"satz-{number}") for number in range(1, 7)} <= paths
    assert {("6", "satz-1"), ("6", "satz-2")} <= paths
    assert not any(
        path[:2] == ("1", "5") and path[-1].startswith("satz-") for path in paths
    )
    assert not any(
        path[:3] == ("6", "2", "c") and path[-1].startswith("satz-") for path in paths
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
    assert any(occurrence.value == 12348 for _branch, occurrence in obligations)


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
    boundaries = {(branch.path, occurrence.value) for branch, occurrence in obligations}

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
    source = "Die Steuer ergibt sich aus dem Einkommen geteilt durch den Grundwert."
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
    issue = next(
        issue
        for issue in result.issues
        if "formula-output" in issue and "formula clause 2" in issue
    )
    assert "internal punctuation-span ordinal" in issue
    assert "not a statutory paragraph number" in issue
    assert "existing path-covering principal output's formula" in issue
    assert "Otherwise create a principal" in issue
    assert "or precisely defer the computation" in issue
    assert "`versions[N].formula` proof atom" in issue
    assert "`source.corpus_citation_path` is exactly `de/statute/estg/32a`" in issue
    assert "characters" in issue
    assert "der zweite Betrag ist Einkommen * 3" in issue
    assert "shorter excerpt that does not itself state the computation" in issue


def test_formula_output_diagnostic_repairs_nj_shaped_root_schedule_binding():
    schedule_clause = (
        "(2) For the purposes of the calculation of the New Jersey earned "
        "income tax credit, the percentage of the federal earned income tax "
        "credit referred to in paragraph (1) of this subsection shall be: "
        "(a) 10% for the taxable year beginning on or after January 1, 2000, "
        "but before January 1, 2001;"
    )
    overpayment_clause = (
        " If the credit exceeds the amount of tax otherwise due, that amount "
        "of excess shall be an overpayment for the purposes of N.J.S.54A:9-7;"
    )
    source_prefix = "New Jersey earned income tax credit program. "
    source = source_prefix + schedule_clause + overpayment_clause
    short_schedule_excerpt = (
        "10% for the taxable year beginning on or after January 1, 2000"
    )

    def content(derived_excerpt: str) -> str:
        return f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-nj/statute/54a:4-7
rules:
  - name: credit_percentage
    kind: parameter
    dtype: Rate
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: parameter
            source:
              corpus_citation_path: us-nj/statute/54a:4-7
              excerpt: "{schedule_clause}"
    versions:
      - effective_from: '2000-01-01'
        formula: 0.10
  - name: credit_amount
    kind: derived
    dtype: Money
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us-nj/statute/54a:4-7
              excerpt: "{derived_excerpt}"
    versions:
      - effective_from: '2000-01-01'
        formula: federal_credit * credit_percentage
  - name: credit_overpayment
    kind: derived
    dtype: Money
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us-nj/statute/54a:4-7
              excerpt: "{overpayment_clause.strip()}"
    versions:
      - effective_from: '2000-01-01'
        formula: max(0, credit_amount - tax_due)
"""

    unbound = _analyze(
        content(short_schedule_excerpt),
        source,
        corpus_citation_path="us-nj/statute/54a:4-7",
        test_cases=[],
    )
    bound = _analyze(
        content(schedule_clause),
        source,
        corpus_citation_path="us-nj/statute/54a:4-7",
        test_cases=[],
    )

    issue = next(
        issue
        for issue in unbound.issues
        if "formula-output" in issue and schedule_clause in issue
    )
    assert f"characters {len(source_prefix)}:" in issue
    assert schedule_clause in issue
    assert "A parameter rule" in issue
    assert not _has_issue(bound, "formula-output")


@pytest.mark.parametrize(
    "source",
    [
        "The amount equals " + "income plus supplement, " * 30 + "income.",
        "The amount equals " + "`" * 200 + " income.",
    ],
)
def test_long_formula_output_locator_does_not_present_ellipsis_as_source_text(
    source: str,
):
    branch = completeness_module.SourceStructureBranch(
        (),
        "formula-clause",
        "source unit formula clause 1",
        source,
        10,
        10 + len(source),
    )

    feedback = completeness_module._formula_output_binding_feedback(
        branch,
        corpus_citation_path=CORPUS_CITATION_PATH,
        has_path_covering_principal=True,
    )
    preview, was_truncated = completeness_module._bounded_source_feedback_preview(
        source
    )

    if "`" in source:
        assert len(source) <= 360
    assert was_truncated
    assert " ... " in preview
    assert " ... " in feedback
    assert "only a bounded locator and is not source text" in feedback
    assert "copy one contiguous verbatim" in feedback


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


@pytest.mark.parametrize(
    ("source", "extra_parameter"),
    [
        (
            "(1) Der Zuschlag beträgt 259 Euro bis zu einem Einkommen von 100 Euro.",
            """\
  - name: income_limit
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions: [{formula: 100}]
""",
        ),
        (
            (
                "(1) Der Zuschlag beträgt 259 Euro für Einkommen bis "
                "einschließlich 100 Euro."
            ),
            """\
  - name: income_limit
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions: [{formula: 100}]
""",
        ),
        (
            ("(1) Der Zuschlag beträgt 259 Euro für Einkommen bis maximal 100 Euro."),
            """\
  - name: income_limit
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions: [{formula: 100}]
""",
        ),
        (
            "(1) Der Zuschlag beträgt 259 Euro, außer bei einer Befreiung.",
            "",
        ),
        (
            "(1) Der Zuschlag beträgt 259 Euro, wenn die Person "
            "anspruchsberechtigt ist.",
            "",
        ),
    ],
)
def test_controlled_scalar_source_rejects_parameter_only(
    source: str,
    extra_parameter: str,
):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: supplement_amount
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions: [{{formula: 259}}]
{extra_parameter}"""

    result = _analyze(content, source, test_cases=[])

    assert _has_issue(result, "formula-output", "control", "parameter-only")


def test_piecewise_condition_rejects_constants_only():
    source = (
        "(1) Wenn das Einkommen 100 Euro nicht übersteigt, beträgt die "
        "Steuer 10 Euro; anderenfalls 20 Euro."
    )
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: income_limit
    kind: parameter
    source: de/statute/estg/32a(1)
    versions: [{formula: 100}]
  - name: lower_amount
    kind: parameter
    source: de/statute/estg/32a(1)
    versions: [{formula: 10}]
  - name: upper_amount
    kind: parameter
    source: de/statute/estg/32a(1)
    versions: [{formula: 20}]
"""

    result = _analyze(content, source, test_cases=[])

    assert _has_issue(result, "formula-output", "control", "parameter-only")


def test_positive_applicability_condition_requires_paired_cases():
    source = (
        "(1) Der Zuschlag beträgt 259 Euro, wenn die Person anspruchsberechtigt ist."
    )
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: supplement_amount
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions: [{formula: 259}]
  - name: payable_supplement
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - formula: 'if is_eligible: supplement_amount else: 0'
"""

    def case(is_eligible: bool) -> dict[str, object]:
        return {
            "name": f"eligible={is_eligible}",
            "input": {"is_eligible": is_eligible},
            "output": {"payable_supplement": 259 if is_eligible else 0},
        }

    positive_only = _analyze(content, source, test_cases=[case(True)])
    paired = _analyze(
        content,
        source,
        test_cases=[case(False), case(True)],
    )

    assert _has_issue(positive_only, "applicability", "paired")
    assert not paired.issues


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
        case for case in COMPLETE_COMPANION_TESTS if case["name"] != "exception applies"
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
    content = (
        COMPANION_COVERAGE_CONTENT
        + """\
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
    )
    test_cases = [
        case for case in COMPLETE_COMPANION_TESTS if case["name"] != "exception applies"
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
    source = (
        "(1) Der Betrag wird als Einkommen * 2 berechnet und auf volle Euro abzurunden."
    )
    content = _single_rounding_content(
        "floor(income * multiplier + unrelated_amount * 0)",
    )
    test_cases = [
        {
            "name": "unrelated fractional input",
            "period": "2026",
            "input": {"income": 10, "unrelated_amount": 90.5},
            "output": {"amount": 20},
        },
    ]

    result = _analyze(
        content,
        source,
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
            "output": {"amount": income * (2 if married or high_income else 3)},
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
    content = (
        MULTI_PARAGRAPH_FORMULA_CONTENT.replace(
            "formula: income * first_multiplier + income * second_multiplier",
            "formula: >-\n"
            "          if married or high_income: income * first_multiplier "
            "else: income * second_multiplier",
        )
        .replace(
            "first_multiplier",
            "married_multiplier",
        )
        .replace(
            "second_multiplier",
            "single_multiplier",
        )
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

    assert (
        completeness_module._case_formula_branch_outcome(
            rule,
            {"input": {"eligible": "holds"}},
        )
        == "if:0"
    )
    assert (
        completeness_module._case_formula_branch_outcome(
            rule,
            {"input": {"eligible": "not_holds"}},
        )
        == "if:1"
    )


def test_inline_elif_formula_reports_each_reachable_branch():
    rule = {
        "versions": [
            {
                "formula": "if first: A elif second: B else: C",
            }
        ]
    }

    assert (
        completeness_module._case_formula_branch_outcome(
            rule,
            {"input": {"first": True, "second": False}},
        )
        == "if:0"
    )
    assert (
        completeness_module._case_formula_branch_outcome(
            rule,
            {"input": {"first": False, "second": True}},
        )
        == "if:1"
    )
    assert (
        completeness_module._case_formula_branch_outcome(
            rule,
            {"input": {"first": False, "second": False}},
        )
        == "if:2"
    )


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

    assert (
        completeness_module._case_formula_branch_outcome(
            rule,
            {"input": {"filing_status": "unknown"}},
        )
        == "match:1"
    )


@pytest.mark.parametrize(
    ("pattern", "pattern_input"),
    [
        ("holds", {"holds": "married"}),
        ("_", {"_": "married"}),
    ],
)
def test_bare_match_patterns_resolve_as_runtime_names(
    pattern: str,
    pattern_input: dict[str, str],
):
    rule = {
        "versions": [
            {
                "formula": (
                    f"match status: {pattern} => married_amount; "
                    '"fallback" => fallback_amount'
                ),
            }
        ]
    }

    assert (
        completeness_module._case_formula_branch_outcome(
            rule,
            {
                "input": {
                    "status": True if pattern == "holds" else "_",
                    **pattern_input,
                }
            },
        )
        == "match:1"
    )


@pytest.mark.parametrize(
    ("guard_name", "guard_value"),
    [
        ("holds", False),
        ("not_holds", False),
        ("TRUE", False),
    ],
)
def test_bare_condition_names_are_not_boolean_aliases(
    guard_name: str,
    guard_value: bool,
):
    rule = {
        "versions": [
            {
                "formula": (f"if {guard_name}: enabled_amount else: disabled_amount"),
            }
        ]
    }

    assert (
        completeness_module._case_formula_branch_outcome(
            rule,
            {"input": {guard_name: guard_value}},
        )
        == "if:1"
    )


def test_quoted_control_text_does_not_confuse_formula_execution():
    rule = {
        "versions": [
            {
                "formula": (
                    'if code == "if x: y else: z": selected_amount else: other_amount'
                ),
            }
        ]
    }

    assert (
        completeness_module._case_formula_branch_outcome(
            rule,
            {"input": {"code": "if x: y else: z"}},
        )
        == "if:0"
    )


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
        expected = 0 if disabled else int(income * (5 if income <= 100 else 7) / 100)
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
    source = "(1) Der Betrag wird als Einkommen * 2 berechnet. " + exception_text
    one_sided_tests = [
        case for case in COMPLETE_COMPANION_TESTS if case["name"] != "exception applies"
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
        "(1) Das Einkommen ist zu verfünffachen.",
        "(1) Das Einkommen ist zu versechsfachen.",
        "(1) Das Einkommen ist zu versiebenfachen.",
        "(1) Das Einkommen ist zu verachtfachen.",
        "(1) Das Einkommen ist zu verneunfachen.",
        "(1) Das Einkommen ist zu verzehnfachen.",
        "(1) Das Einkommen ist zu halbieren.",
        "(1) Das Einkommen ist durch Halbierung zu ermitteln.",
        "(1) Das Einkommen ist in zwei gleiche Teile zu teilen.",
        "(1) Der Betrag ist durch zwei zu teilen.",
        "(1) Der Betrag ist um 2 zu erhöhen.",
        "(1) Der Betrag ist um zwei zu erhöhen.",
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


@pytest.mark.parametrize(
    ("source", "parameter_name", "parameter_value", "correct_formula", "wrong_formula"),
    [
        (
            "(1) Das Einkommen ist zu verdreifachen.",
            "factor",
            3,
            "income * factor",
            "income * 2",
        ),
        (
            "(1) Das Einkommen ist mit drei zu multiplizieren.",
            "factor",
            3,
            "income * factor",
            "income * 2",
        ),
        (
            "(1) Das Einkommen ist mal drei zu nehmen.",
            "factor",
            3,
            "income * factor",
            "income * 2",
        ),
        (
            "(1) Der Betrag ist durch zwei zu teilen.",
            "divisor",
            2,
            "income / divisor",
            "income / 3",
        ),
        (
            "(1) Der Betrag ist um 2 zu erhöhen.",
            "increment",
            2,
            "income + increment",
            "income * increment",
        ),
    ],
)
def test_german_worded_computation_binds_operation_and_factor(
    source: str,
    parameter_name: str,
    parameter_value: int,
    correct_formula: str,
    wrong_formula: str,
):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: {parameter_name}
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - formula: {parameter_value}
  - name: amount
    kind: derived
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - formula: FORMULA
"""

    correct = _analyze(
        content.replace("FORMULA", correct_formula),
        source,
        test_cases=[
            {
                "name": "correct computation",
                "input": {"income": 12},
                "output": {
                    "amount": (
                        6
                        if parameter_name == "divisor"
                        else 14
                        if parameter_name == "increment"
                        else 36
                    )
                },
            }
        ],
    )
    wrong = _analyze(
        content.replace("FORMULA", wrong_formula),
        source,
        test_cases=[
            {
                "name": "wrong computation",
                "input": {"income": 12},
                "output": {"amount": (4 if parameter_name == "divisor" else 24)},
            }
        ],
    )

    assert not correct.issues
    assert _has_issue(wrong, "formula branch")


@pytest.mark.parametrize(
    ("word", "factor"),
    [
        ("verfünffachen", 5),
        ("versechsfachen", 6),
        ("versiebenfachen", 7),
        ("verachtfachen", 8),
        ("verneunfachen", 9),
        ("verzehnfachen", 10),
    ],
)
def test_german_infinitive_multiplier_binds_exact_factor(
    word: str,
    factor: int,
):
    source = f"(1) Das Einkommen ist zu {word}."
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: factor
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - formula: {factor}
  - name: amount
    kind: derived
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - formula: FORMULA
"""

    correct = _analyze(
        content.replace("FORMULA", "income * factor"),
        source,
        test_cases=[
            {
                "name": "exact multiplier",
                "input": {"income": 12},
                "output": {"amount": 12 * factor},
            }
        ],
    )
    wrong = _analyze(
        content.replace("FORMULA", "income * 2"),
        source,
        test_cases=[
            {
                "name": "wrong multiplier",
                "input": {"income": 12},
                "output": {"amount": 24},
            }
        ],
    )

    assert not correct.issues
    assert _has_issue(wrong, "formula branch")


@pytest.mark.parametrize(
    "source",
    [
        "(1) Der Betrag ist zu halbieren.",
        "(1) Der Betrag ist durch Halbierung zu ermitteln.",
        "(1) Der Betrag ist in zwei gleiche Teile zu teilen.",
    ],
)
def test_german_half_wording_binds_exact_division(source: str):
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
      - formula: 2
  - name: amount
    kind: derived
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - formula: FORMULA
"""

    divided = _analyze(
        content.replace("FORMULA", "income / divisor"),
        source,
        test_cases=[
            {
                "name": "divide by two",
                "input": {"income": 12},
                "output": {"amount": 6},
            }
        ],
    )
    multiplied = _analyze(
        content.replace("FORMULA", "income * 0.5"),
        source,
        test_cases=[
            {
                "name": "multiply by half",
                "input": {"income": 12},
                "output": {"amount": 6},
            }
        ],
    )
    wrong = _analyze(
        content.replace("FORMULA", "income / 3"),
        source,
        test_cases=[
            {
                "name": "wrong divisor",
                "input": {"income": 12},
                "output": {"amount": 4},
            }
        ],
    )

    assert not divided.issues
    assert not multiplied.issues
    assert _has_issue(wrong, "formula branch")


@pytest.mark.parametrize(
    ("source", "delta", "correct_formula", "wrong_formula", "correct_output"),
    [
        (
            "(1) Der Betrag ist um zwei zu erhöhen.",
            2,
            "income + delta",
            "income + 3",
            14,
        ),
        (
            "(1) Der Betrag ist um vier zu vermehren.",
            4,
            "income + delta",
            "income + 2",
            16,
        ),
        (
            "(1) Der Betrag ist um drei zu vermindern.",
            3,
            "income - delta",
            "income - 2",
            9,
        ),
        (
            "(1) Der Betrag ist um fünf zu kürzen.",
            5,
            "income - delta",
            "income - 2",
            7,
        ),
    ],
)
def test_german_word_delta_binds_exact_signed_value(
    source: str,
    delta: int,
    correct_formula: str,
    wrong_formula: str,
    correct_output: int,
):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: delta
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - formula: {delta}
  - name: amount
    kind: derived
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - formula: FORMULA
"""

    correct = _analyze(
        content.replace("FORMULA", correct_formula),
        source,
        test_cases=[
            {
                "name": "exact delta",
                "input": {"income": 12},
                "output": {"amount": correct_output},
            }
        ],
    )
    wrong = _analyze(
        content.replace("FORMULA", wrong_formula),
        source,
        test_cases=[
            {
                "name": "wrong delta",
                "input": {"income": 12},
                "output": {"amount": 15},
            }
        ],
    )

    assert not correct.issues
    assert _has_issue(wrong, "formula branch")


def test_unknown_german_word_operand_fails_closed():
    source = "(1) Der Betrag ist um elf zu erhöhen."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: delta
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - formula: 11
  - name: amount
    kind: derived
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - formula: income + delta
"""
    case = {
        "name": "unsupported word numeral",
        "input": {"income": 12},
        "output": {"amount": 23},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "formula branch")


def test_symbolic_german_factor_is_not_treated_as_unknown_number_word():
    source = "(1) Das Einkommen ist mit dem Faktor F zu multiplizieren."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: factor
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - formula: 0.5
  - name: amount
    kind: derived
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - formula: income * factor
"""
    case = {
        "name": "symbolic factor",
        "input": {"income": 10},
        "output": {"amount": 5},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues


def test_word_delta_binding_survives_result_rounding_wrapper():
    source = """\
(1) Der Betrag ist um zwei zu erhöhen und auf volle Euro abzurunden.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: delta
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - formula: 2
  - name: amount
    kind: derived
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions:
      - formula: floor(income + delta)
"""
    case = {
        "name": "fractional result",
        "input": {"income": 10.5},
        "output": {"amount": 12},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues


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


def test_louisiana_cpi_formula_separates_effective_year_from_computation():
    source = """\
Beginning January 1, 2026, and thereafter, the amount of the standard deduction
provided in Subsection A of this Section shall be adjusted annually by an amount calculated
by multiplying the amount of the prior year's standard deduction by the percentage increase
in the Consumer Price Index United States city average for all urban consumers (CPI-U), as
reported by the United States Department of Labor, Bureau of Labor Statistics, or its
successor, for the previous calendar year.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-la/statute/47:294
rules:
  - name: annual_standard_deduction_adjustment
    kind: derived
    dtype: Money
    source: La. R.S. 47:294(B)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us-la/statute/47:294
              excerpt: |-
                Beginning January 1, 2026, and thereafter, the amount of the standard deduction
                provided in Subsection A of this Section shall be adjusted annually by an amount calculated
                by multiplying the amount of the prior year's standard deduction by the percentage increase
                in the Consumer Price Index United States city average for all urban consumers (CPI-U), as
                reported by the United States Department of Labor, Bureau of Labor Statistics, or its
                successor, for the previous calendar year.
    versions:
      - effective_from: '2026-01-01'
        formula: prior_year_standard_deduction * consumer_price_index_u_percentage_increase_for_previous_calendar_year
"""
    case = {
        "name": "2026 CPI-U adjustment",
        "period": {
            "period_kind": "tax_year",
            "start": "2026-01-01",
            "end": "2026-12-31",
        },
        "input": {
            "prior_year_standard_deduction": 20000,
            "consumer_price_index_u_percentage_increase_for_previous_calendar_year": 0.05,
        },
        "output": {"annual_standard_deduction_adjustment": 1000},
    }

    def analyze(candidate_content, candidate_case):
        return analyze_complete_source_unit(
            candidate_content,
            source,
            corpus_citation_path="us-la/statute/47:294",
            test_cases=[candidate_case],
            extract_numeric_occurrences=EN_NUMERIC_OCCURRENCE_EXTRACTOR,
            extract_numeric_grounding_occurrences=(
                EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR
            ),
            extract_named_scalars=extract_named_scalar_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
        )

    correct = analyze(content, case)
    wrong_operation_content = content.replace(
        "prior_year_standard_deduction * consumer_price_index_u_percentage_increase_for_previous_calendar_year",
        "prior_year_standard_deduction + consumer_price_index_u_percentage_increase_for_previous_calendar_year",
    )
    wrong_operation_case = yaml.safe_load(yaml.safe_dump(case))
    wrong_operation_case["output"]["annual_standard_deduction_adjustment"] = 20000.05
    wrong_operation = analyze(wrong_operation_content, wrong_operation_case)
    pre_effective_case = yaml.safe_load(yaml.safe_dump(case))
    pre_effective_case["name"] = "pre-effective CPI-U adjustment"
    pre_effective_case["period"] = {
        "period_kind": "tax_year",
        "start": "2025-01-01",
        "end": "2025-12-31",
    }
    pre_effective = analyze(content, pre_effective_case)

    assert not correct.issues
    assert _has_issue(wrong_operation, "formula branch")
    assert _has_issue(pre_effective, "formula branch")


@pytest.mark.parametrize(
    "source",
    (
        "Effective January 1, 2026, the amount is calculated by multiplying income by the rate.",
        "Beginning January 1, 2026 and thereafter, the amount is calculated by multiplying income by the rate.",
    ),
)
def test_common_formula_applicability_prefaces_are_not_coefficients(source):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-la/statute/47:294
rules:
  - name: adjusted_amount
    kind: derived
    dtype: Money
    source: us-la/statute/47:294
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us-la/statute/47:294
              excerpt: >-
                {source}
    versions:
      - effective_from: '2026-01-01'
        formula: income * rate
"""
    case = {
        "name": "effective formula",
        "period": "2026",
        "input": {"income": 100, "rate": 0.05},
        "output": {"adjusted_amount": 5},
    }

    result = analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path="us-la/statute/47:294",
        test_cases=[case],
        extract_numeric_occurrences=EN_NUMERIC_OCCURRENCE_EXTRACTOR,
        extract_numeric_grounding_occurrences=(
            EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )

    assert not result.issues


def test_temporal_formula_filter_keeps_substantive_multiplier_grounding():
    source = (
        "For tax year 2026, the amount is calculated by multiplying "
        "the prior amount by 3."
    )
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-la/statute/47:294
rules:
  - name: multiplier
    kind: parameter
    dtype: Decimal
    source: us-la/statute/47:294(B)
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: adjusted_amount
    kind: derived
    dtype: Money
    source: us-la/statute/47:294(B)
    versions:
      - effective_from: '2026-01-01'
        formula: prior_amount * multiplier
"""
    case = {
        "name": "wrong substantive multiplier",
        "period": "2026",
        "input": {"prior_amount": 10},
        "output": {"adjusted_amount": 20},
    }

    result = analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path="us-la/statute/47:294",
        test_cases=[case],
        extract_numeric_occurrences=EN_NUMERIC_OCCURRENCE_EXTRACTOR,
        extract_numeric_grounding_occurrences=(
            EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )

    assert _has_issue(result, "formula branch")


@pytest.mark.parametrize(
    "source",
    (
        "The amount is calculated by subtracting tax year 2020 from the current tax year.",
        "Tax year 2020 is subtracted from the current tax year, and the amount is calculated.",
        "Effective rate is calculated by subtracting tax year 2020, from the current tax year.",
        "For purposes of this section, the base is tax year 2020, and the amount is calculated as the current tax year minus the base.",
        "For tax year calculations, the amount is calculated by subtracting tax year 2020 from the current tax year.",
    ),
)
def test_temporal_formula_operand_remains_required_outside_applicability_preface(
    source,
):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-la/statute/47:294
rules:
  - name: base_tax_year
    kind: parameter
    dtype: Decimal
    source: us-la/statute/47:294
    versions:
      - formula: 2020
  - name: elapsed_tax_years
    kind: derived
    dtype: Decimal
    source: us-la/statute/47:294
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us-la/statute/47:294
              excerpt: >-
                {source}
    versions:
      - formula: tax_year - base_tax_year
"""
    case = {
        "name": "elapsed years from the statutory base year",
        "period": "2026",
        "input": {"tax_year": 2026},
        "output": {"elapsed_tax_years": 6},
    }

    def analyze(candidate_content, candidate_case):
        return analyze_complete_source_unit(
            candidate_content,
            source,
            corpus_citation_path="us-la/statute/47:294",
            test_cases=[candidate_case],
            extract_numeric_occurrences=EN_NUMERIC_OCCURRENCE_EXTRACTOR,
            extract_numeric_grounding_occurrences=(
                EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR
            ),
            extract_named_scalars=extract_named_scalar_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
        )

    correct = analyze(content, case)
    wrong_content = content.replace(
        "tax_year - base_tax_year",
        "tax_year - unrelated_year",
    )
    wrong_case = yaml.safe_load(yaml.safe_dump(case))
    wrong_case["input"]["unrelated_year"] = 2000
    wrong_case["output"]["elapsed_tax_years"] = 26
    wrong = analyze(wrong_content, wrong_case)

    assert not correct.issues
    assert _has_issue(wrong, "formula branch")


def test_louisiana_line_wrapped_two_hundred_percent_formula_grounds_factor_two():
    source = """\
(2) Married-Joint Return, a Qualified Surviving 200% of the dollar amount

Spouse, and Head of Household provided for Single Individuals
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-la/statute/47:294
rules:
  - name: joint_standard_deduction_multiplier
    kind: parameter
    dtype: Rate
    source: La. R.S. 47:294(A)(2)
    versions:
      - effective_from: '2025-01-01'
        formula: 2.0
  - name: joint_standard_deduction
    kind: derived
    dtype: Money
    source: La. R.S. 47:294(A)(2)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us-la/statute/47:294
              excerpt: |-
                Married-Joint Return, a Qualified Surviving 200% of the dollar amount

                Spouse, and Head of Household provided for Single Individuals
    versions:
      - effective_from: '2025-01-01'
        formula: single_standard_deduction * joint_standard_deduction_multiplier
"""
    case = {
        "name": "joint deduction is twice the single amount",
        "period": "2025",
        "input": {"single_standard_deduction": 12500},
        "output": {"joint_standard_deduction": 25000},
    }

    result = analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path="us-la/statute/47:294",
        test_cases=[case],
        extract_numeric_occurrences=EN_NUMERIC_OCCURRENCE_EXTRACTOR,
        extract_numeric_grounding_occurrences=(
            EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    wrong_multiplier_case = yaml.safe_load(yaml.safe_dump(case))
    wrong_multiplier_case["output"]["joint_standard_deduction"] = 37500
    wrong_multiplier = analyze_complete_source_unit(
        content.replace("formula: 2.0", "formula: 3.0"),
        source,
        corpus_citation_path="us-la/statute/47:294",
        test_cases=[wrong_multiplier_case],
        extract_numeric_occurrences=EN_NUMERIC_OCCURRENCE_EXTRACTOR,
        extract_numeric_grounding_occurrences=(
            EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )

    assert not result.issues
    assert _has_issue(wrong_multiplier, "formula branch")


def test_formula_witness_preserves_source_operation_topology():
    source = "(1) Der Betrag wird als Einkommen * 2 + 3 berechnet."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: factor
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions: [{formula: 2}]
  - name: supplement
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions: [{formula: 3}]
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - formula: FORMULA
"""

    correct_results = [
        _analyze(
            content.replace("FORMULA", formula),
            source,
            test_cases=[
                {
                    "name": "source computation",
                    "input": {"income": 10},
                    "output": {"amount": 23},
                }
            ],
        )
        for formula in (
            "income * factor + supplement",
            "factor * income + supplement",
            "supplement + (income * factor)",
        )
    ]
    wrong = _analyze(
        content.replace("FORMULA", "income + factor * supplement"),
        source,
        test_cases=[
            {
                "name": "different computation",
                "input": {"income": 10},
                "output": {"amount": 16},
            }
        ],
    )

    assert all(not result.issues for result in correct_results)
    assert _has_issue(wrong, "formula branch")


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


def test_quoted_enum_match_patterns_are_literal_values():
    rule = {
        "versions": [
            {
                "effective_from": "2026-01-01",
                "formula": """\
match status:
  "married" => income * 2
  "single" => income * 3
""",
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


def test_partial_literal_versions_do_not_become_global_formula_constants():
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
        formula: external_factor
      - effective_from: '2026-01-01'
        formula: 2
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2025-01-01'
        formula: income * multiplier
"""
    case = {
        "name": "unresolved historical coefficient",
        "period": "2025",
        "input": {"income": 10},
        "output": {"amount": 20},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "formula branch")


def test_singleton_effective_dated_constant_is_not_timeless():
    payload = {
        "rules": [
            {
                "name": "minimum_age",
                "versions": [
                    {
                        "effective_from": "2020-01-01",
                        "effective_to": "2024-12-31",
                        "formula": 18,
                    }
                ],
            }
        ]
    }
    environment = completeness_module._constant_rule_environment(payload)

    assert (
        completeness_module._formula_environment_for_case(environment, {})[
            "minimum_age"
        ]
        == 18
    )
    assert "minimum_age" not in completeness_module._formula_environment_for_case(
        environment,
        {"period": "2019"},
    )
    assert (
        completeness_module._formula_environment_for_case(
            environment,
            {"period": "2022"},
        )["minimum_age"]
        == 18
    )
    assert "minimum_age" not in completeness_module._formula_environment_for_case(
        environment,
        {"period": "2026"},
    )


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


def test_multiline_boundary_conjunction_binds_named_threshold():
    source = "(1) Die Regel gilt ab einem Alter von 18 Jahren."
    content = _boundary_control_content(
        formula="claimant_age >= income_limit\n          and claimant_is_resident",
        limit_versions="""\
      - effective_from: '2026-01-01'
        formula: 18""",
    )
    cases = [
        {
            "name": "at age threshold",
            "period": "2026",
            "input": {"claimant_age": 18, "claimant_is_resident": True},
            "output": {"eligible": True},
        },
        {
            "name": "below age threshold",
            "period": "2026",
            "input": {"claimant_age": 17, "claimant_is_resident": True},
            "output": {"eligible": False},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


def test_multiline_arithmetic_conjunction_witnesses_source_sum_formula():
    source = "(1) The sum of dwelling units and vouchers is 550 or fewer."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: combined_unit_and_voucher_limit
    kind: parameter
    dtype: Count
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 550
  - name: qualified_agency
    kind: derived
    dtype: Judgment
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          dwelling_units + vouchers <= combined_unit_and_voucher_limit
          and agency_is_not_troubled
"""
    cases = [
        {
            "name": "at combined limit",
            "period": "2026",
            "input": {
                "dwelling_units": 250,
                "vouchers": 300,
                "agency_is_not_troubled": True,
            },
            "output": {"qualified_agency": "holds"},
        },
        {
            "name": "above combined limit",
            "period": "2026",
            "input": {
                "dwelling_units": 250,
                "vouchers": 301,
                "agency_is_not_troubled": True,
            },
            "output": {"qualified_agency": "not_holds"},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not _has_issue(result, "formula branch", "test")

    bypassed_content = content.replace(
        "dwelling_units + vouchers <= combined_unit_and_voucher_limit\n"
        "          and agency_is_not_troubled",
        "agency_override\n"
        "          or dwelling_units + vouchers <= combined_unit_and_voucher_limit",
    )
    bypassed_cases = [
        {
            **case,
            "input": {**case["input"], "agency_override": True},
            "output": {"qualified_agency": "holds"},
        }
        for case in cases
    ]

    bypassed = _analyze(bypassed_content, source, test_cases=bypassed_cases)

    assert _has_issue(bypassed, "formula branch", "test")


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
    ("wording", "correct_formula", "correct_output", "wrong_formula"),
    [
        ("bis 100 Euro", "income <= income_limit", True, "income >= income_limit"),
        ("unter 100 Euro", "income < income_limit", False, "income <= income_limit"),
        ("ab 100 Euro", "income >= income_limit", True, "income <= income_limit"),
        ("über 100 Euro", "income > income_limit", False, "income >= income_limit"),
        (
            "von mehr als 100 Euro",
            "income > income_limit",
            False,
            "income >= income_limit",
        ),
        (
            "mehr als 100 Euro",
            "income > income_limit",
            False,
            "income >= income_limit",
        ),
        (
            "weniger als 100 Euro",
            "income < income_limit",
            False,
            "income <= income_limit",
        ),
        (
            "von weniger als 100 Euro",
            "income < income_limit",
            False,
            "income <= income_limit",
        ),
        (
            "bis einschließlich 100 Euro",
            "income <= income_limit",
            True,
            "income >= income_limit",
        ),
        (
            "bis maximal 100 Euro",
            "income <= income_limit",
            True,
            "income >= income_limit",
        ),
    ],
)
def test_boundary_comparator_preserves_direction_and_inclusivity(
    wording: str,
    correct_formula: str,
    correct_output: bool,
    wrong_formula: str,
):
    source = f"(1) Die Regel gilt für Einkommen {wording}."
    limit_versions = """\
      - effective_from: '2026-01-01'
        formula: 100"""
    correct = _boundary_control_content(
        formula=correct_formula,
        limit_versions=limit_versions,
    )
    wrong = _boundary_control_content(
        formula=wrong_formula,
        limit_versions=limit_versions,
    )
    base_case = {
        "name": "source endpoint",
        "period": "2026",
        "input": {"income": 100},
    }

    correct_result = _analyze(
        correct,
        source,
        test_cases=[
            {**base_case, "output": {"eligible": correct_output}},
        ],
    )
    wrong_result = _analyze(
        wrong,
        source,
        test_cases=[
            {**base_case, "output": {"eligible": True}},
        ],
    )

    assert not correct_result.issues
    assert _has_issue(wrong_result, "boundary", "100")


@pytest.mark.parametrize(
    ("formula", "endpoint_output", "expected_issue"),
    [
        ("income <= income_limit", True, False),
        ("not (income > income_limit)", True, False),
        ("if income > income_limit: false else: true", True, False),
        ("not (income <= income_limit)", False, True),
        ("if income > income_limit: true else: false", False, True),
        ("if income <= income_limit: false else: true", False, True),
    ],
)
def test_boundary_predicate_preserves_applicability_polarity(
    formula: str,
    endpoint_output: bool,
    expected_issue: bool,
):
    source = "(1) Die Regel gilt für Einkommen bis 100 Euro."
    content = _boundary_control_content(
        formula=formula,
        limit_versions="""\
      - effective_from: '2026-01-01'
        formula: 100""",
    )
    case = {
        "name": "endpoint polarity",
        "period": "2026",
        "input": {"income": 100},
        "output": {"eligible": endpoint_output},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "boundary", "100") is expected_issue


def test_negative_boundary_prose_inverts_boolean_applicability():
    source = "(1) Einkommen bis 100 Euro begründen keine Berechtigung."
    content = _boundary_control_content(
        formula="income > income_limit",
        limit_versions="""\
      - effective_from: '2026-01-01'
        formula: 100""",
    )
    case = {
        "name": "excluded endpoint",
        "period": "2026",
        "input": {"income": 100},
        "output": {"eligible": False},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues


def test_repeated_boundary_value_keeps_fragment_span_and_polarity():
    source = (
        "(1) Die Regel gilt für Einkommen bis 100 Euro. "
        "Einkommen bis 100 Euro begründen keine Berechtigung."
    )
    branches = recognize_source_structure(source)
    obligations = completeness_module._source_boundary_obligations(
        branches,
        extract_numeric_occurrences=DE_NUMERIC_OCCURRENCE_EXTRACTOR,
    )
    repeated = [
        (branch, occurrence)
        for branch, occurrence in obligations
        if occurrence.value == 100
    ]

    assert len(repeated) == 2
    assert len({occurrence.start for _branch, occurrence in repeated}) == 2
    assert [
        completeness_module._source_interval_and_polarity_for_boundary(
            branch,
            occurrence,
            extract_numeric_occurrences=DE_NUMERIC_OCCURRENCE_EXTRACTOR,
        )[1]
        for branch, occurrence in repeated
    ] == [1, -1]


def test_repeated_opposite_boundary_requires_both_polarity_witnesses():
    source = (
        "(1) Die Regel gilt für Einkommen bis 100 Euro. "
        "Einkommen bis 100 Euro begründen keine Berechtigung."
    )
    content = _boundary_control_content(
        formula="income <= income_limit",
        limit_versions="""\
      - effective_from: '2026-01-01'
        formula: 100""",
        extra_rules="""\
  - name: excluded_income_limit
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 100
""",
    )
    case = {
        "name": "positive endpoint only",
        "period": "2026",
        "input": {"income": 100},
        "output": {"eligible": True},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "boundary", "100")


@pytest.mark.parametrize(
    ("formula", "endpoint_output", "expected_issue"),
    [
        (
            "if income <= income_limit: holds else: not_holds",
            "holds",
            False,
        ),
        (
            "if income <= income_limit: not_holds else: holds",
            "not_holds",
            True,
        ),
    ],
)
def test_judgment_boundary_polarity_is_checked(
    formula: str,
    endpoint_output: str,
    expected_issue: bool,
):
    source = "(1) Die Regel gilt für Einkommen bis 100 Euro."
    content = _boundary_control_content(
        formula=formula,
        limit_versions="""\
      - effective_from: '2026-01-01'
        formula: 100""",
    ).replace("dtype: bool", "dtype: Judgment")
    case = {
        "name": "Judgment endpoint",
        "period": "2026",
        "input": {"income": 100},
        "output": {"eligible": endpoint_output},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "boundary", "100") is expected_issue


@pytest.mark.parametrize(
    ("formula", "endpoint_output", "expected_issue"),
    [
        (
            "match income <= income_limit: true => true; false => false",
            True,
            False,
        ),
        (
            "match income <= income_limit: true => false; false => true",
            False,
            True,
        ),
    ],
)
def test_match_boundary_selector_preserves_polarity(
    formula: str,
    endpoint_output: bool,
    expected_issue: bool,
):
    source = "(1) Die Regel gilt für Einkommen bis 100 Euro."
    content = _boundary_control_content(
        formula=formula,
        limit_versions="""\
      - effective_from: '2026-01-01'
        formula: 100""",
    )
    case = {
        "name": "match endpoint",
        "period": "2026",
        "input": {"income": 100},
        "output": {"eligible": endpoint_output},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "boundary", "100") is expected_issue


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
    source = "(1) Ein Anspruch besteht, es sei denn, eine Befreiung liegt vor."
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


def _single_rounding_content(formula: str) -> str:
    return f"""\
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
        formula: {formula}
"""


def test_neutralized_fractional_input_cannot_witness_rounding():
    source = (
        "(1) Der Betrag wird als Einkommen * 2 berechnet und auf volle Euro abzurunden."
    )
    content = _single_rounding_content(
        "floor(income * multiplier + fractional_probe * 0)",
    )
    case = {
        "name": "neutral fractional probe",
        "period": "2026",
        "input": {"income": 10, "fractional_probe": 0.5},
        "output": {"amount": 20},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "rounding", "fractional")


def test_fractional_input_must_leave_the_rounded_operand_fractional():
    source = (
        "(1) Der Betrag wird als Einkommen * 2 berechnet und auf volle Euro abzurunden."
    )
    content = _single_rounding_content("floor(income * multiplier)")
    case = {
        "name": "integral computed operand",
        "period": "2026",
        "input": {"income": 10.5},
        "output": {"amount": 21},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "rounding", "fractional")


def test_dead_rounding_call_cannot_witness_source_rounding():
    source = (
        "(1) Der Betrag wird als Einkommen * 2 berechnet. "
        "Das Ergebnis ist auf volle Euro abzurunden."
    )
    content = _single_rounding_content(
        "floor(income * multiplier) + 0 * floor(dummy_amount)",
    )
    case = {
        "name": "dead fractional call",
        "period": "2026",
        "input": {"income": 10, "dummy_amount": 10.5},
        "output": {"amount": 20},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "rounding", "fractional")


MULTI_ROUNDING_SOURCE = """\
(1) Der erste Betrag ist auf volle Euro abzurunden.
(2) Der zweite Betrag ist auf volle Euro abzurunden.
"""

MULTI_ROUNDING_CONTENT = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: total_amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1); de/statute/estg/32a(2)
    versions:
      - effective_from: '2026-01-01'
        formula: floor(first_amount) + floor(second_amount)
"""


def _multi_rounding_case(
    name: str,
    *,
    first_amount: float,
    second_amount: float,
) -> dict[str, object]:
    return {
        "name": name,
        "period": "2026",
        "input": {
            "first_amount": first_amount,
            "second_amount": second_amount,
        },
        "output": {
            "total_amount": int(first_amount // 1 + second_amount // 1),
        },
    }


def test_each_source_rounding_clause_needs_its_own_affected_call():
    first_only = [
        _multi_rounding_case(
            "first fractional one",
            first_amount=10.5,
            second_amount=20,
        ),
        _multi_rounding_case(
            "first fractional two",
            first_amount=11.5,
            second_amount=20,
        ),
    ]
    complete = [
        _multi_rounding_case(
            "first fractional",
            first_amount=10.5,
            second_amount=20,
        ),
        _multi_rounding_case(
            "second fractional",
            first_amount=10,
            second_amount=20.5,
        ),
    ]

    missing_second = _analyze(
        MULTI_ROUNDING_CONTENT,
        MULTI_ROUNDING_SOURCE,
        test_cases=first_only,
    )
    both_calls = _analyze(
        MULTI_ROUNDING_CONTENT,
        MULTI_ROUNDING_SOURCE,
        test_cases=complete,
    )

    assert _has_issue(missing_second, "rounding", "fractional")
    assert not both_calls.issues


def test_standalone_result_rounding_modifies_preceding_computation():
    source = (
        "(1) Der Betrag wird als Einkommen * 2 berechnet. "
        "Das Ergebnis ist auf volle Euro abzurunden."
    )
    content = _single_rounding_content("floor(income * multiplier)")
    case = {
        "name": "split rounding sentence",
        "period": "2026",
        "input": {"income": 10.25},
        "output": {"amount": 20},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues


def test_standalone_result_rounding_requires_the_final_computation():
    source = (
        "(1) Der Betrag wird als Einkommen * 2 + Zuschlag berechnet. "
        "Das Ergebnis ist auf volle Euro abzurunden."
    )
    content = _single_rounding_content(
        "floor(income * multiplier) + supplement",
    )
    case = {
        "name": "rounded component only",
        "period": "2026",
        "input": {"income": 10.25, "supplement": 0.25},
        "output": {"amount": 20.25},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "rounding", "fractional")


@pytest.mark.parametrize(
    "result_clause",
    [
        (
            "6Der sich ergebende Steuerbetrag ist auf den nächsten vollen "
            "Euro-Betrag abzurunden."
        ),
        "Satz 2 Das Ergebnis ist auf volle Euro abzurunden.",
        (
            "Satz 6: Der sich ergebende Steuerbetrag ist auf den nächsten "
            "vollen Euro-Betrag abzurunden."
        ),
    ],
)
def test_statutory_result_rounding_attaches_to_preceding_computation(
    result_clause: str,
):
    source = f"(1) Der Steuerbetrag ist Einkommen * 2. {result_clause}"
    content = _single_rounding_content("floor(income * multiplier)")
    if result_clause.startswith(("6", "Satz 2 ", "Satz 6:")):
        content = content.replace(
            """\
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:""",
            f"""\
  - name: amount
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
              excerpt: {result_clause!r}
    versions:""",
        )
    case = {
        "name": "statutory result rounding",
        "period": "2026",
        "input": {"income": 10.25},
        "output": {"amount": 20},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues


def test_result_rounding_cannot_bind_an_earlier_component():
    source = (
        "(1) Der Grundbetrag ist Einkommen * 2. "
        "Der Endbetrag ist Grundbetrag + Zuschlag. "
        "Das Ergebnis ist auf volle Euro abzurunden."
    )
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: multiplier
    kind: parameter
    source: de/statute/estg/32a(1)
    versions: [{formula: 2}]
  - name: base_amount
    kind: derived
    source: de/statute/estg/32a(1)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: Der Grundbetrag ist Einkommen * 2.
    versions: [{formula: 'income * multiplier'}]
  - name: unrounded_amount
    kind: derived
    source: de/statute/estg/32a(1)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: Der Endbetrag ist Grundbetrag + Zuschlag.
    versions: [{formula: 'base_amount + supplement'}]
  - name: amount
    kind: derived
    source: de/statute/estg/32a(1)
    versions: [{formula: 'floor(base_amount)'}]
"""
    case = {
        "name": "earlier component rounded",
        "input": {"income": 10.25, "supplement": 0.25},
        "output": {
            "base_amount": 20.5,
            "unrounded_amount": 20.75,
            "amount": 20,
        },
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "rounding", "fractional")


def test_integral_input_can_produce_fractional_rounding_operand():
    source = (
        "(1) Der Betrag wird als Einkommen * 1,5 berechnet und "
        "auf volle Euro abgerundet."
    )
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
    versions: [{formula: 1.5}]
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions: [{formula: 'floor(income * multiplier)'}]
"""
    case = {
        "name": "computed fractional operand",
        "input": {"income": 1},
        "output": {"amount": 1},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues


def test_rounding_accepts_asserted_reached_intermediate_output():
    source = (
        "(1) Der Betrag wird als Einkommen * 2 berechnet und auf volle Euro abgerundet."
    )
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: multiplier
    kind: parameter
    source: de/statute/estg/32a(1)
    versions: [{formula: 2}]
  - name: unrounded_amount
    kind: derived
    source: de/statute/estg/32a(1)
    versions: [{formula: 'income * multiplier'}]
  - name: amount
    kind: derived
    source: de/statute/estg/32a(1)
    versions: [{formula: 'floor(unrounded_amount)'}]
"""
    case = {
        "name": "derived rounding operand",
        "input": {"income": 10.25},
        "output": {"unrounded_amount": 20.5, "amount": 20},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues


def test_mismatched_asserted_intermediate_cannot_witness_rounding():
    source = (
        "(1) Der Betrag wird als Einkommen * 2 berechnet und auf volle Euro abgerundet."
    )
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: multiplier
    kind: parameter
    source: de/statute/estg/32a(1)
    versions: [{formula: 2}]
  - name: unrounded_amount
    kind: derived
    source: de/statute/estg/32a(1)
    versions: [{formula: 'income * multiplier'}]
  - name: amount
    kind: derived
    source: de/statute/estg/32a(1)
    versions: [{formula: 'floor(unrounded_amount)'}]
"""
    case = {
        "name": "false intermediate assertion",
        "input": {"income": 10},
        "output": {"unrounded_amount": 20.5, "amount": 20},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "rounding", "fractional")


@pytest.mark.parametrize(
    ("rule_name", "formula", "expected_issue"),
    [
        (
            "basic_allowance",
            "floor(unrounded_basic_allowance)",
            False,
        ),
        (
            "basic_allowance_amount",
            "floor(unrounded_basic_allowance)",
            False,
        ),
        ("basic_allowance", "floor(child_allowance)", True),
        ("basic_allowance", "floor(basic_allowance_child)", True),
        ("basic_allowance_amount", "floor(basic)", True),
    ],
)
def test_single_rounding_binds_to_affected_output_stage(
    rule_name: str,
    formula: str,
    expected_issue: bool,
):
    source = "(1) Der Grundfreibetrag ist auf volle Euro abzurunden."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: RULE_NAME
    kind: derived
    source: de/statute/estg/32a(1)
    versions:
      - formula: FORMULA
""".replace("FORMULA", formula).replace("RULE_NAME", rule_name)
    operand_name = formula.removeprefix("floor(").removesuffix(")")
    case = {
        "name": "single pure rounding",
        "input": {operand_name: 10.5},
        "output": {rule_name: 10},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "rounding", "fractional") is expected_issue


def test_translated_multi_rounding_uses_distinct_operative_calls():
    source = """\
(1) Der Grundfreibetrag ist auf volle Euro abzurunden.
(2) Der Kinderfreibetrag ist auf volle Euro abzurunden.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: total_amount
    kind: derived
    source: de/statute/estg/32a(1); de/statute/estg/32a(2)
    versions:
      - formula: floor(basic_allowance) + floor(child_allowance)
"""
    cases = [
        {
            "name": "basic fractional",
            "input": {"basic_allowance": 10.5, "child_allowance": 20},
            "output": {"total_amount": 30},
        },
        {
            "name": "child fractional",
            "input": {"basic_allowance": 10, "child_allowance": 20.5},
            "output": {"total_amount": 30},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


def test_one_ambiguous_rounding_call_cannot_cover_two_source_clauses():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: total_amount
    kind: derived
    source: de/statute/estg/32a(1); de/statute/estg/32a(2)
    versions:
      - formula: floor(first_second_amount)
"""
    cases = [
        {
            "name": "fractional one",
            "input": {"first_second_amount": 10.5},
            "output": {"total_amount": 10},
        },
        {
            "name": "fractional two",
            "input": {"first_second_amount": 11.5},
            "output": {"total_amount": 11},
        },
    ]

    result = _analyze(content, MULTI_ROUNDING_SOURCE, test_cases=cases)

    assert _has_issue(result, "rounding", "fractional")


def test_boundary_accepts_asserted_reached_derived_selector():
    source = "(1) Der Anspruch gilt bis 100 Euro zu versteuerndes Einkommen."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: income_limit
    kind: parameter
    source: de/statute/estg/32a(1)
    versions: [{formula: 100}]
  - name: taxable_income
    kind: derived
    source: de/statute/estg/32a(1)
    versions: [{formula: 'gross_income - deduction'}]
  - name: eligible
    kind: derived
    source: de/statute/estg/32a(1)
    versions: [{formula: 'taxable_income <= income_limit'}]
"""
    case = {
        "name": "derived boundary selector",
        "input": {"gross_income": 120, "deduction": 20},
        "output": {"taxable_income": 100, "eligible": True},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues


def test_range_formula_branches_accept_asserted_derived_selector_values():
    source = """\
(1) Für Einkommen bis 100 Euro beträgt der Betrag Einkommen * 2.
(2) Für Einkommen über 100 Euro beträgt der Betrag Einkommen * 3.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: first_limit
    kind: parameter
    source: de/statute/estg/32a(1)
    versions: [{formula: 100}]
  - name: first_multiplier
    kind: parameter
    source: de/statute/estg/32a(1)
    versions: [{formula: 2}]
  - name: second_multiplier
    kind: parameter
    source: de/statute/estg/32a(2)
    versions: [{formula: 3}]
  - name: taxable_income
    kind: derived
    source: de/statute/estg/32a(1); de/statute/estg/32a(2)
    versions: [{formula: 'gross_income - deduction'}]
  - name: amount
    kind: derived
    source: de/statute/estg/32a(1); de/statute/estg/32a(2)
    versions:
      - formula: >-
          if taxable_income <= first_limit:
            taxable_income * first_multiplier
          else:
            taxable_income * second_multiplier
"""
    cases = [
        {
            "name": "derived lower selector",
            "input": {"gross_income": 120, "deduction": 20},
            "output": {"taxable_income": 100, "amount": 200},
        },
        {
            "name": "derived upper selector",
            "input": {"gross_income": 121, "deduction": 20},
            "output": {"taxable_income": 101, "amount": 303},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


def test_companion_input_cannot_shadow_local_derived_selector():
    source = "(1) Der Anspruch gilt bis 100 Euro Einkommen."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: income_limit
    kind: parameter
    source: de/statute/estg/32a(1)
    versions: [{formula: 100}]
  - name: taxable_income
    kind: derived
    source: de/statute/estg/32a(1)
    versions: [{formula: 'gross_income - deduction'}]
  - name: eligible
    kind: derived
    source: de/statute/estg/32a(1)
    versions: [{formula: 'taxable_income <= income_limit'}]
"""
    case = {
        "name": "shadowed derived selector",
        "input": {
            "gross_income": 999,
            "deduction": 0,
            "taxable_income": 100,
        },
        "output": {"taxable_income": 100, "eligible": True},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "shadowed derived selector", "shadowed")


def test_exception_accepts_asserted_reached_derived_selector_toggle():
    source = "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: exemption_applies
    kind: derived
    dtype: bool
    source: de/statute/estg/32a(1)
    versions: [{formula: 'has_certificate'}]
  - name: eligible
    kind: derived
    dtype: bool
    source: de/statute/estg/32a(1)
    versions: [{formula: 'if exemption_applies: false else: true'}]
"""
    cases = [
        {
            "name": "ordinary",
            "input": {"has_certificate": False},
            "output": {"exemption_applies": False, "eligible": True},
        },
        {
            "name": "exempt",
            "input": {"has_certificate": True},
            "output": {"exemption_applies": True, "eligible": False},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


def _exception_control_content(formula: str, *, extra_rules: str = "") -> str:
    return f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
{extra_rules}  - name: result
    kind: derived
    source: de/statute/estg/32a(1)
    versions:
      - formula: '{formula}'
"""


def test_exception_toggle_cannot_borrow_effect_from_changed_numeric_input():
    source = "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt."
    content = _exception_control_content(
        "if exemption_applies: income else: income",
    )
    cases = [
        {
            "name": "ordinary low",
            "input": {"exemption_applies": False, "income": 10},
            "output": {"result": 10},
        },
        {
            "name": "exempt high",
            "input": {"exemption_applies": True, "income": 20},
            "output": {"result": 20},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test")


def test_exception_toggle_cannot_borrow_effect_from_another_boolean():
    source = "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt."
    content = _exception_control_content(
        "if exemption_applies: "
        "(if bonus_applies: income + 1 else: income) "
        "else: (if bonus_applies: income + 1 else: income)",
    )
    cases = [
        {
            "name": "ordinary without bonus",
            "input": {
                "exemption_applies": False,
                "bonus_applies": False,
                "income": 10,
            },
            "output": {"result": 10},
        },
        {
            "name": "exempt with bonus",
            "input": {
                "exemption_applies": True,
                "bonus_applies": True,
                "income": 10,
            },
            "output": {"result": 11},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test")


def test_exception_toggle_cases_must_share_period():
    source = "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt."
    content = _exception_control_content(
        "if exemption_applies: base_amount else: base_amount",
        extra_rules="""\
  - name: base_amount
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        effective_to: '2025-12-31'
        formula: 10
      - effective_from: '2026-01-01'
        formula: 20
""",
    )
    cases = [
        {
            "name": "ordinary old",
            "period": "2025",
            "input": {"exemption_applies": False},
            "output": {"result": 10},
        },
        {
            "name": "exempt new",
            "period": "2026",
            "input": {"exemption_applies": True},
            "output": {"result": 20},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test")


def test_boolean_exception_selector_must_block_not_enable():
    source = "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt."
    content = _exception_control_content("if exemption_applies: true else: false")
    cases = [
        {
            "name": "ordinary",
            "input": {"exemption_applies": False},
            "output": {"result": False},
        },
        {
            "name": "enabling exception",
            "input": {"exemption_applies": True},
            "output": {"result": True},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test")


def test_numeric_exclusion_effect_is_direction_neutral_without_amount_wording():
    source = "(1) Der Abzug gilt nicht, wenn eine Befreiung vorliegt."
    content = _exception_control_content(
        "if exemption_applies: tax_without_deduction else: tax_with_deduction"
    )
    cases = [
        {
            "name": "deduction applies",
            "input": {
                "exemption_applies": False,
                "tax_with_deduction": 10,
                "tax_without_deduction": 20,
            },
            "output": {"result": 10},
        },
        {
            "name": "deduction excluded",
            "input": {
                "exemption_applies": True,
                "tax_with_deduction": 10,
                "tax_without_deduction": 20,
            },
            "output": {"result": 20},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


@pytest.mark.parametrize(
    ("formula", "ordinary_output", "exception_output", "expected_issue"),
    [
        ("if has_certificate: true else: false", False, True, True),
        ("if has_certificate: false else: true", True, False, False),
    ],
)
def test_german_ausser_binds_neutral_selector_orientation(
    formula: str,
    ordinary_output: bool,
    exception_output: bool,
    expected_issue: bool,
):
    source = "(1) Der Anspruch besteht, außer bei einer Bescheinigung."
    content = _exception_control_content(formula)
    cases = [
        {
            "name": "without certificate",
            "input": {"has_certificate": False},
            "output": {"result": ordinary_output},
        },
        {
            "name": "with certificate",
            "input": {"has_certificate": True},
            "output": {"result": exception_output},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test") is expected_issue


@pytest.mark.parametrize(
    ("formula", "ordinary_input", "exception_input"),
    [
        ("if no_exemption: true else: false", True, False),
        ("no_exemption", True, False),
        ("if has_certificate: false else: true", False, True),
        ("not has_certificate", False, True),
        ("exemption_applies == false", False, True),
    ],
)
def test_exception_selector_forms_preserve_source_orientation(
    formula: str,
    ordinary_input: bool,
    exception_input: bool,
):
    selector_name = (
        "no_exemption"
        if "no_exemption" in formula
        else (
            "has_certificate" if "has_certificate" in formula else "exemption_applies"
        )
    )
    source_condition = (
        "eine Bescheinigung vorliegt"
        if selector_name == "has_certificate"
        else "eine Befreiung vorliegt"
    )
    source = f"(1) Der Anspruch gilt nicht, wenn {source_condition}."
    content = _exception_control_content(formula)
    cases = [
        {
            "name": "ordinary",
            "input": {selector_name: ordinary_input},
            "output": {"result": True},
        },
        {
            "name": "exception",
            "input": {selector_name: exception_input},
            "output": {"result": False},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


def test_compound_direct_exception_expression_is_discovered():
    source = "(1) Der Anspruch gilt nicht, wenn eine Befreiung oder Sperre vorliegt."
    content = _exception_control_content("not (exemption_applies or barred)")
    cases = [
        {
            "name": "ordinary",
            "input": {"exemption_applies": False, "barred": False},
            "output": {"result": True},
        },
        {
            "name": "exempt",
            "input": {"exemption_applies": True, "barred": False},
            "output": {"result": False},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


def test_multiline_boolean_applicability_selector_is_discovered():
    source = "(1) The claimant is eligible if a certificate is present."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: result
    kind: derived
    source: de/statute/estg/32a(1)
    versions:
      - formula: |-
          claimant_is_resident
          and has_certificate
"""
    cases = [
        {
            "name": "without certificate",
            "input": {"claimant_is_resident": True, "has_certificate": False},
            "output": {"result": False},
        },
        {
            "name": "with certificate",
            "input": {"claimant_is_resident": True, "has_certificate": True},
            "output": {"result": True},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


def test_multiline_direct_formula_returns_every_boolean_selector_name():
    rule = {"versions": [{"formula": ("eligible\nand resident\nand not disqualified")}]}

    assert completeness_module._rule_exception_selector_names(rule) == {
        "eligible",
        "resident",
        "disqualified",
    }


def test_arithmetic_if_clause_is_formula_evidence_not_exception_toggle():
    source = """\
(1) If the credit exceeds the tax due, the amount of the excess is an overpayment.
"""
    branches = recognize_source_structure(source)
    formula_branches = completeness_module._source_formula_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    exception_branches = completeness_module._source_exception_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
        formula_branches=formula_branches,
    )

    assert len(formula_branches) == 1
    assert not exception_branches


def test_formula_clause_with_explicit_carveout_keeps_exception_obligation():
    source = """\
(1) The amount is computed from income, except when exempt, when it is zero.
"""
    branches = recognize_source_structure(source)
    formula_branches = completeness_module._source_formula_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    exception_branches = completeness_module._source_exception_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
        formula_branches=formula_branches,
    )

    assert len(formula_branches) == 1
    assert len(exception_branches) == 1


def test_formula_clause_keeps_nonarithmetic_applicability_obligation():
    source = """\
(1) If the claimant is eligible, the credit is computed as income * 10 percent.
"""
    branches = recognize_source_structure(source)
    formula_branches = completeness_module._source_formula_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    exception_branches = completeness_module._source_exception_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
        formula_branches=formula_branches,
    )

    assert len(formula_branches) == 1
    assert len(exception_branches) == 1


def test_positive_eligibility_condition_requires_enabling_effect():
    source = """\
(1) The claimant shall be eligible if the claimant is ineligible due to age.
"""
    correct = _exception_control_content("if ineligible_due_to_age: true else: false")
    wrong = _exception_control_content("if ineligible_due_to_age: false else: true")
    correct_cases = [
        {
            "name": "ordinary",
            "input": {"ineligible_due_to_age": False},
            "output": {"result": False},
        },
        {
            "name": "age modification",
            "input": {"ineligible_due_to_age": True},
            "output": {"result": True},
        },
    ]
    wrong_cases = [
        {
            "name": "ordinary",
            "input": {"ineligible_due_to_age": False},
            "output": {"result": True},
        },
        {
            "name": "age modification",
            "input": {"ineligible_due_to_age": True},
            "output": {"result": False},
        },
    ]

    correct_result = _analyze(correct, source, test_cases=correct_cases)
    wrong_result = _analyze(wrong, source, test_cases=wrong_cases)

    assert not correct_result.issues
    assert _has_issue(wrong_result, "exception", "test")


@pytest.mark.parametrize(
    "source",
    [
        "(1) The claimant is ineligible unless an exception applies.",
        "(1) The claim is excluded except when a waiver applies.",
        (
            "(1) If the claimant is ineligible solely due to age, "
            "the claimant shall be eligible."
        ),
    ],
)
def test_enabling_effect_is_independent_of_proposition_order(source: str):
    assert completeness_module._source_exception_effect_requirement(source) == (
        "enable"
    )


def test_notwithstanding_exemption_preserves_affirmative_duty():
    source = """\
(b) Qualified public housing agencies
(A) The requirement under paragraph (1) shall not apply to a qualified agency.
(B) Notwithstanding that qualified public housing agencies are exempt under
subparagraph (A) from the requirement under this section to prepare and submit an
annual public housing plan, each qualified public housing agency shall, on an annual
basis, make the certification described in paragraph (16) of subsection (d), except
that the paragraph shall use substitute language.
"""

    assert completeness_module._source_exception_effect_requirement(source) == (
        "enable"
    )


@pytest.mark.parametrize(
    "source",
    [
        (
            "Notwithstanding that qualified public housing agencies are not exempt "
            "under subparagraph (A) from the requirement under this section to "
            "prepare and submit an annual public housing plan, each qualified public "
            "housing agency shall, on an annual basis, make the certification "
            "described in paragraph (16) of subsection (d), except that it is revised."
        ),
        (
            "Notwithstanding that no exemption applies, each qualified public housing "
            "agency shall, on an annual basis, make the certification described in "
            "paragraph (16) of subsection (d), except that it is revised."
        ),
        (
            "Notwithstanding that qualified agencies must certify an exemption, each "
            "qualified public housing agency shall, on an annual basis, make the "
            "certification described in paragraph (16) of subsection (d), except that "
            "it is revised."
        ),
        (
            "Notwithstanding that qualified public housing agencies are exempt under "
            "subparagraph (A) from the requirement under this section to prepare and "
            "submit an annual public housing plan, each qualified public housing "
            "agency must certify."
        ),
        (
            "Notwithstanding that qualified public housing agencies are exempt under "
            "subparagraph (A) from the requirement under this section to prepare and "
            "submit an annual public housing plan, each qualified public housing "
            "agency shall make the certification if requested."
        ),
    ],
)
def test_notwithstanding_exemption_rejects_near_misses(source: str):
    assert (
        completeness_module._notwithstanding_exemption_effect_requirement(source)
        is None
    )


def test_notwithstanding_exemption_accepts_enabling_companion_pair():
    source = """\
(1) Notwithstanding that qualified public housing agencies are exempt under
subparagraph (A) from the requirement under this section to prepare and submit an
annual public housing plan, each qualified public housing agency shall, on an annual
basis, make the certification described in paragraph (16) of subsection (d), except
that the paragraph shall use substitute language.
"""
    content = _exception_control_content(
        "if qualified_agency: true else: false",
    )
    cases = [
        {
            "name": "ordinary agency",
            "input": {"qualified_agency": False},
            "output": {"result": False},
        },
        {
            "name": "qualified agency certification",
            "input": {"qualified_agency": True},
            "output": {"result": True},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


def test_preposed_german_negative_condition_enables_positive_result():
    source = """\
(1) Wenn der Antragsteller nicht berechtigt ist, ist er ausnahmsweise berechtigt.
"""

    assert completeness_module._source_exception_effect_requirement(source) == (
        "enable"
    )


@pytest.mark.parametrize(
    "source",
    [
        "(1) The claimant is not allowed if a disqualification applies.",
        "(1) The claimant is not qualified when a disqualification applies.",
        "(1) The claimant is not entitled if a disqualification applies.",
    ],
)
def test_explicit_negative_eligibility_condition_is_excluding(source: str):
    assert completeness_module._source_exception_effect_requirement(source) == (
        "exclude"
    )


def test_qualification_exception_requires_enabling_effect():
    source = """\
(1) The claimant shall meet all qualifications, except the age requirement, in
order to be eligible for the credit.
"""
    content = _exception_control_content(
        "if age_requirement_exception_applies: true else: false"
    )
    cases = [
        {
            "name": "ordinary qualification path",
            "input": {"age_requirement_exception_applies": False},
            "output": {"result": False},
        },
        {
            "name": "age qualification exception",
            "input": {"age_requirement_exception_applies": True},
            "output": {"result": True},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


@pytest.mark.parametrize(
    "source",
    [
        "(1) The credit is subject to the provisions of this chapter.",
        (
            "(1) The credit is subject to all provisions of this chapter, "
            "except as may otherwise be provided."
        ),
    ],
)
def test_non_toggleable_cross_reference_does_not_require_paired_cases(
    source: str,
):
    content = _exception_control_content("false")
    case = {"name": "nonapplicable", "input": {}, "output": {"result": False}}

    result = _analyze(content, source, test_cases=[case])

    assert not _has_issue(result, "exception", "test")


@pytest.mark.parametrize(
    "source",
    [
        "(1) The credit applies except as provided in section 5.",
        "(1) The credit is subject to section 32.",
        "(1) The credit is subject to 26 U.S.C. 32.",
    ],
)
def test_formal_cross_reference_does_not_require_synthetic_toggle(source: str):
    assert not completeness_module._source_exception_requires_paired_witness(source)


@pytest.mark.parametrize(
    ("formula", "output", "expected_issue"),
    [("false", False, False), ("true", True, True)],
)
def test_unconditional_nonapplicability_requires_asserted_false_output(
    formula: str,
    output: bool,
    expected_issue: bool,
):
    source = "(1) Subsection (f) shall not apply."
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: subsection_f_applies
    kind: derived
    dtype: Judgment
    source: de/statute/estg/32a(1)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: Subsection (f) shall not apply
    versions:
      - formula: '{formula}'
"""
    case = {
        "name": "subsection f treatment",
        "input": {},
        "output": {"subsection_f_applies": output},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "non-applicability", "tests") is expected_issue


@pytest.mark.parametrize(
    "source",
    [
        "(1) The provisions of section 5 shall not apply.",
        "(1) The requirements of section 5 shall not apply.",
        "(1) The limitations under subsection (b) shall not apply.",
        "(1) The preceding sentence shall not apply.",
        "(1) Subsections (a) and (b) shall not apply.",
        "(1) Subsections (a), (b), and (c) shall not apply.",
        "(1) Section 5 of this title shall not apply.",
        "(1) Paragraphs (1) through (3) shall not apply.",
    ],
)
def test_unconditional_legal_reference_subject_is_recognized(source: str):
    assert completeness_module._source_unconditional_nonapplicability(source)
    assert not completeness_module._source_exception_requires_paired_witness(source)


def test_german_unconditional_nonapplicability_requires_false_output():
    source = "(1) Absatz 2 findet keine Anwendung."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: absatz_2_applies
    kind: derived
    dtype: Judgment
    source: de/statute/estg/32a(1)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: Absatz 2 findet keine Anwendung
    versions:
      - formula: 'false'
"""
    case = {
        "name": "absatz 2 treatment",
        "input": {},
        "output": {"absatz_2_applies": False},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues


@pytest.mark.parametrize(
    "source",
    [
        "(1) The credit shall not apply to nonresidents.",
        "(1) The credit shall not apply after December 31.",
        "(1) The credit shall not apply for taxpayers with income over 100.",
        "(1) The credit shall not apply until the claimant reaches age 65.",
        "(1) The credit shall not apply while the claimant is incarcerated.",
        "(1) The credit shall not apply on a joint return.",
        "(1) The credit shall not apply in respect of foreign income.",
        "(1) The credit shall not apply under subsection (b).",
        "(1) The credit shall not apply in taxable years ending in 2025.",
        "(1) The credit shall not apply whenever the claimant is married.",
        "(1) After December 31, the credit shall not apply.",
        "(1) After December 31 the credit shall not apply.",
        "(1) Taxpayers earning more than 100 dollars are not eligible.",
        "(1) Credits exceeding 100 shall not apply.",
        "(1) Credits claimed by nonresidents shall not apply.",
        "(1) Individuals aged 65 or older are not eligible.",
        "(1) The married taxpayers are not eligible.",
        "(1) The nonresident taxpayer is not eligible.",
        "(1) The taxpayer, a nonresident, is not eligible.",
        "(1) Der Anspruch gilt für Nichtansässige nicht.",
    ],
)
def test_scoped_nonapplicability_is_not_treated_as_unconditional(source: str):
    assert not completeness_module._source_unconditional_nonapplicability(source)
    assert completeness_module._source_exception_requires_paired_witness(source)


@pytest.mark.parametrize(
    "source",
    [
        "(1) The credit is subject to income restrictions.",
        "(1) The credit is subject to residency restrictions.",
        "(1) The credit is subject to restrictions on married taxpayers.",
    ],
)
def test_local_restrictions_are_not_treated_as_formal_cross_references(source: str):
    assert completeness_module._source_exception_requires_paired_witness(source)


@pytest.mark.parametrize(
    "source",
    [
        "(1) The credit is subject to this chapter.",
        "(1) The credit is subject to chapter 5.",
        "(1) The credit is subject to title 54A.",
        "(1) The credit is subject to the provisions of chapter 5.",
        "(1) The credit is subject to the provisions of title 54A.",
        "(1) The credit is subject to this article.",
        "(1) The credit is subject to Article IV.",
        "(1) The credit is subject to this part.",
        "(1) The credit is subject to Part II.",
        "(1) The credit is subject to subchapter B.",
        "(1) The credit is subject to subtitle A.",
        "(1) The credit is subject to division 2.",
        "(1) The credit is subject to the provisions of Article IV.",
        (
            "(1) The credit is subject to the restrictions of this subsection "
            "and subsections b., c., d. and e. of this section."
        ),
        (
            "(1) The credit is subject to all provisions of N.J.S.54A:1-1 et "
            "seq., except as may be otherwise specifically provided in "
            "P.L.2000, c.80 (C.54A:4-6 et al.)."
        ),
    ],
)
def test_direct_chapter_and_title_cross_references_are_formal(source: str):
    assert not completeness_module._source_exception_requires_paired_witness(source)


@pytest.mark.parametrize(
    "source",
    [
        ("(1) The credit is subject to section 5 and the claimant must be a resident."),
        "(1) The credit is subject to section 5 and residency restrictions.",
        ("(1) The credit is subject to chapter 5, but only for married taxpayers."),
        (
            "(1) The credit is subject to the provisions of chapter 5 and an "
            "income limit."
        ),
        ("(1) The credit is subject to section 5, the claimant must be a resident."),
        ("(1) The credit is subject to section 5; the claimant must be a resident."),
        "(1) The credit is subject to section 5 with an income limit.",
        "(1) The credit is subject to section 5 for married taxpayers.",
        (
            "(1) The credit is subject to section 5 as modified by residency "
            "requirements."
        ),
        "(1) The credit is subject to section 5 or residency restrictions.",
        "(1) The credit is subject to section 5 except for nonresidents.",
    ],
)
def test_mixed_reference_and_local_condition_requires_paired_evidence(source: str):
    assert completeness_module._source_exception_requires_paired_witness(source)


@pytest.mark.parametrize(
    "text",
    [
        "part time employment",
        "division of income",
        "title requirements",
        "article requirements",
        "section eligibility rules",
        "part ownership",
    ],
)
def test_ordinary_language_is_not_a_formal_structural_reference(text: str):
    assert not completeness_module._source_has_formal_cross_reference(text)


@pytest.mark.parametrize(
    "source",
    [
        "(1) The credit applies except as otherwise provided.",
        "(1) The credit applies except as may otherwise be provided.",
        "(1) The credit applies except as may be provided.",
        "(1) The credit applies except as provided by law.",
    ],
)
def test_terminal_boilerplate_reservation_does_not_require_toggle(source: str):
    assert not completeness_module._source_exception_requires_paired_witness(source)


@pytest.mark.parametrize(
    "source",
    [
        "(1) Except as otherwise provided, the credit applies.",
        "(1) Except as may otherwise be provided by law, the credit applies.",
        "(1) Except as provided in section 5, the credit applies.",
        "(1) Subject to section 5, the credit applies.",
        "(1) Subject to the provisions of this chapter, the credit applies.",
        "(1) Vorbehaltlich § 5 gilt der Anspruch.",
        "(1) Subject to section 5, a credit applies.",
        "(1) Subject to section 5, such credit applies.",
        "(1) Subject to section 5, the tax credit applies.",
        ("(1) Subject to section 5, the state earned income tax credit applies."),
        ("(1) Subject to section 5, the federal earned income tax credit applies."),
        "(1) Subject to section 5, the claimant is eligible.",
        "(1) Subject to section 5, the deduction is allowed.",
        "(1) Except as provided in section 5, a benefit is available.",
        "(1) Vorbehaltlich § 5 besteht der Anspruch.",
    ],
)
def test_preposed_reference_reservation_does_not_require_toggle(source: str):
    assert not completeness_module._source_exception_requires_paired_witness(source)


@pytest.mark.parametrize(
    "source",
    [
        (
            "(1) Subject to section 5, but only for married taxpayers, "
            "the credit applies."
        ),
        "(1) Subject to section 5, the claimant must be a resident.",
        ("(1) Subject to section 5 and residency restrictions, the credit applies."),
        ("(1) Except as provided in section 5, the credit applies only to residents."),
        "(1) Subject to section 5, the credit applies if income is low.",
        "(1) Subject to section 5, the credit applies to residents.",
        "(1) Subject to section 5, the credit applies only when married.",
        "(1) Subject to section 5, if eligible the credit applies.",
        "(1) Subject to section 5, when eligible the credit applies.",
        "(1) Subject to section 5, if resident the benefit is available.",
        "(1) Subject to section 5, while eligible the credit applies.",
        "(1) Subject to section 5, once eligible the credit applies.",
        "(1) Subject to section 5, assuming eligibility the credit applies.",
        "(1) Subject to section 5, the resident-only credit applies.",
    ],
)
def test_preposed_reference_with_local_condition_requires_toggle(source: str):
    assert completeness_module._source_exception_requires_paired_witness(source)


def test_one_false_case_cannot_cover_two_unconditional_obligations():
    source = """\
(1) Subsection (a) shall not apply; subsection (b) shall not apply.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: both_subsections_apply
    kind: derived
    dtype: Judgment
    source: de/statute/estg/32a(1)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: Subsection (a) shall not apply
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: subsection (b) shall not apply
    versions:
      - formula: 'false'
"""
    case = {
        "name": "aggregate false assertion",
        "input": {},
        "output": {"both_subsections_apply": False},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "non-applicability", "tests")


def test_conditional_nonapplicability_still_requires_paired_cases():
    source = "(1) The credit shall not apply if an exemption applies."
    content = _exception_control_content("false")
    case = {"name": "blocked", "input": {}, "output": {"result": False}}

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "exception", "tests")


@pytest.mark.parametrize(
    ("formula", "ordinary_output", "exception_output", "expected_issue"),
    [
        (
            "match exemption_applies: true => false; false => true",
            True,
            False,
            False,
        ),
        (
            "match exemption_applies: true => true; false => false",
            False,
            True,
            True,
        ),
    ],
)
def test_match_exception_selector_preserves_polarity(
    formula: str,
    ordinary_output: bool,
    exception_output: bool,
    expected_issue: bool,
):
    source = "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt."
    content = _exception_control_content(formula)
    cases = [
        {
            "name": "ordinary",
            "input": {"exemption_applies": False},
            "output": {"result": ordinary_output},
        },
        {
            "name": "exception",
            "input": {"exemption_applies": True},
            "output": {"result": exception_output},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test") is expected_issue


def test_exception_semantics_override_ordinary_word_in_selector_name():
    source = "(1) Der Anspruch gilt nicht, wenn eine Ausnahme vorliegt."
    content = _exception_control_content("if eligible_for_exception: false else: true")
    cases = [
        {
            "name": "ordinary",
            "input": {"eligible_for_exception": False},
            "output": {"result": True},
        },
        {
            "name": "exception",
            "input": {"eligible_for_exception": True},
            "output": {"result": False},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


@pytest.mark.parametrize(
    ("formula", "eligible_output", "ineligible_output", "expected_issue"),
    [
        ("if eligible: true else: false", True, False, False),
        ("if eligible: false else: true", False, True, True),
    ],
)
def test_positive_ordinary_selector_is_false_active_for_ineligibility(
    formula: str,
    eligible_output: bool,
    ineligible_output: bool,
    expected_issue: bool,
):
    source = "(1) Der Anspruch gilt nicht, wenn keine Berechtigung besteht."
    content = _exception_control_content(formula)
    cases = [
        {
            "name": "eligible",
            "input": {"eligible": True},
            "output": {"result": eligible_output},
        },
        {
            "name": "ineligible",
            "input": {"eligible": False},
            "output": {"result": ineligible_output},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test") is expected_issue


@pytest.mark.parametrize(
    (
        "source",
        "selector_name",
        "formula",
        "false_output",
        "true_output",
    ),
    [
        (
            "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt.",
            "eligible",
            "if eligible: false else: true",
            True,
            False,
        ),
        (
            "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt.",
            "eligible",
            "not eligible",
            True,
            False,
        ),
        (
            "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt.",
            "eligible",
            "match eligible: true => false; false => true",
            True,
            False,
        ),
        (
            "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt.",
            "qualified",
            "if qualified: false else: true",
            True,
            False,
        ),
        (
            "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt.",
            "ordinary_case",
            "if ordinary_case: true else: false",
            False,
            True,
        ),
        (
            "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt.",
            "regular_case",
            "if regular_case: true else: false",
            False,
            True,
        ),
        (
            "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt.",
            "default_case",
            "if default_case: true else: false",
            False,
            True,
        ),
        (
            "(1) The claim does not apply when no certificate is present.",
            "has_claim",
            "if has_claim: false else: true",
            True,
            False,
        ),
        (
            "(1) Der Anspruch gilt nicht, wenn keine Bescheinigung vorliegt.",
            "has_anspruch",
            "if has_anspruch: false else: true",
            True,
            False,
        ),
    ],
)
def test_exception_selector_must_match_the_condition_not_the_claim(
    source: str,
    selector_name: str,
    formula: str,
    false_output: bool,
    true_output: bool,
):
    content = _exception_control_content(formula)
    cases = [
        {
            "name": "selector false",
            "input": {selector_name: False},
            "output": {"result": false_output},
        },
        {
            "name": "selector true",
            "input": {selector_name: True},
            "output": {"result": true_output},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test")


def test_exception_condition_does_not_borrow_a_preceding_main_clause_concept():
    source = """\
(1) Bei einer Bescheinigung besteht der Anspruch, außer wenn kein Kind vorhanden ist.
"""
    content = _exception_control_content("if has_certificate: false else: true")
    cases = [
        {
            "name": "without certificate",
            "input": {"has_certificate": False},
            "output": {"result": True},
        },
        {
            "name": "with certificate",
            "input": {"has_certificate": True},
            "output": {"result": False},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test")


@pytest.mark.parametrize(
    "source",
    [
        "(1) Wenn keine Berechtigung besteht, gilt der Anspruch nicht.",
        "(1) Bei fehlender Berechtigung gilt der Anspruch nicht.",
        "(1) Ohne Berechtigung gilt der Anspruch nicht.",
    ],
)
def test_preposed_exception_condition_preserves_ordinary_selector(source: str):
    branches = recognize_source_structure(source)
    obligations = completeness_module._source_exception_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    content = _exception_control_content("eligible")
    cases = [
        {
            "name": "eligible",
            "input": {"eligible": True},
            "output": {"result": True},
        },
        {
            "name": "ineligible",
            "input": {"eligible": False},
            "output": {"result": False},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert len(obligations) == 1
    assert not result.issues


@pytest.mark.parametrize(
    "source",
    [
        "(1) Der Anspruch besteht nicht, außer bei einer Bescheinigung.",
        "(1) Der Anspruch gilt nicht, es sei denn, eine Bescheinigung liegt vor.",
        (
            "(1) Für diese Person gilt der Anspruch weiterhin nicht, "
            "außer bei einer Bescheinigung."
        ),
        "(1) The claim does not apply unless there is a certificate.",
    ],
)
def test_exception_to_negative_rule_must_enable_the_claim(source: str):
    correct = _exception_control_content("if has_certificate: true else: false")
    wrong = _exception_control_content("if has_certificate: false else: true")
    correct_cases = [
        {
            "name": "without certificate",
            "input": {"has_certificate": False},
            "output": {"result": False},
        },
        {
            "name": "with certificate",
            "input": {"has_certificate": True},
            "output": {"result": True},
        },
    ]
    wrong_cases = [
        {
            "name": "without certificate",
            "input": {"has_certificate": False},
            "output": {"result": True},
        },
        {
            "name": "with certificate",
            "input": {"has_certificate": True},
            "output": {"result": False},
        },
    ]

    correct_result = _analyze(correct, source, test_cases=correct_cases)
    wrong_result = _analyze(wrong, source, test_cases=wrong_cases)

    assert not correct_result.issues
    assert _has_issue(wrong_result, "exception", "test")


def test_exception_to_negative_rule_accepts_numeric_deduction_effect():
    source = """\
(1) Der Abzug gilt nicht, außer bei einer Bescheinigung.
"""
    content = _exception_control_content(
        "if has_certificate: tax_with_deduction else: tax_without_deduction"
    )
    cases = [
        {
            "name": "without certificate",
            "input": {
                "has_certificate": False,
                "tax_with_deduction": 10,
                "tax_without_deduction": 20,
            },
            "output": {"result": 20},
        },
        {
            "name": "with certificate",
            "input": {
                "has_certificate": True,
                "tax_with_deduction": 10,
                "tax_without_deduction": 20,
            },
            "output": {"result": 10},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


def test_coordinated_exception_cues_require_distinct_witnesses():
    source = """\
(1) Außer bei einer Befreiung und außer bei einer Sperre besteht der Anspruch.
"""
    complete = _exception_control_content(
        "if exemption_applies: false else: if barred: false else: true"
    )
    incomplete = _exception_control_content("if exemption_applies: false else: true")
    complete_cases = [
        {
            "name": "ordinary",
            "input": {"exemption_applies": False, "barred": False},
            "output": {"result": True},
        },
        {
            "name": "exemption",
            "input": {"exemption_applies": True, "barred": False},
            "output": {"result": False},
        },
        {
            "name": "barred",
            "input": {"exemption_applies": False, "barred": True},
            "output": {"result": False},
        },
    ]
    incomplete_cases = [
        {
            "name": "ordinary",
            "input": {"exemption_applies": False},
            "output": {"result": True},
        },
        {
            "name": "exemption",
            "input": {"exemption_applies": True},
            "output": {"result": False},
        },
    ]

    complete_result = _analyze(complete, source, test_cases=complete_cases)
    incomplete_result = _analyze(
        incomplete,
        source,
        test_cases=incomplete_cases,
    )

    assert not complete_result.issues
    assert _has_issue(incomplete_result, "exception", "test")


def test_exception_witness_allocation_uses_maximum_matching():
    source = """\
(1) Der Anspruch besteht, außer bei einer Bescheinigung oder einem Kind.
Der Anspruch besteht, außer bei einer Bescheinigung oder einem Status.
Der Anspruch besteht, außer bei einer Bescheinigung oder einem Status.
"""
    content = _exception_control_content(
        "if has_certificate: false else: "
        "if has_child: false else: "
        "if has_status: false else: true"
    )
    ordinary = {
        "has_certificate": False,
        "has_child": False,
        "has_status": False,
    }
    cases = [
        {
            "name": "ordinary",
            "input": ordinary,
            "output": {"result": True},
        }
    ]
    for selector in ordinary:
        inputs = dict(ordinary)
        inputs[selector] = True
        cases.append(
            {
                "name": selector,
                "input": inputs,
                "output": {"result": False},
            }
        )

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


def test_synonymous_exception_cues_in_one_condition_are_one_obligation():
    source = "(1) Außer bei einer Befreiung gilt der Anspruch nicht."
    branches = recognize_source_structure(source)
    language_matches = tuple(completeness_module._EXCEPTION_LANGUAGE.finditer(source))

    obligations = completeness_module._source_exception_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )

    assert len(language_matches) == 2
    assert len(obligations) == 1


@pytest.mark.parametrize(
    ("source", "selector_name", "formula"),
    [
        (
            "(1) Der Anspruch gilt nicht, wenn keine Bescheinigung vorliegt.",
            "has_certificate",
            "has_certificate",
        ),
        (
            "(1) Der Anspruch gilt nicht, wenn kein Kind vorhanden ist.",
            "kind_vorhanden",
            "kind_vorhanden",
        ),
        (
            "(1) Der Anspruch gilt nicht, wenn das Kind nicht vorhanden ist.",
            "kind_vorhanden",
            "kind_vorhanden",
        ),
        (
            "(1) The claim does not apply when the child is not present.",
            "child_present",
            "child_present",
        ),
    ],
)
def test_source_negation_controls_neutral_selector_orientation(
    source: str,
    selector_name: str,
    formula: str,
):
    content = _exception_control_content(formula)
    cases = [
        {
            "name": "positive fact",
            "input": {selector_name: True},
            "output": {"result": True},
        },
        {
            "name": "missing fact",
            "input": {selector_name: False},
            "output": {"result": False},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


def test_adjectival_source_negation_rejects_opposite_selector_polarity():
    source = "(1) Der Anspruch gilt nicht, wenn das Kind nicht vorhanden ist."
    content = _exception_control_content("not kind_vorhanden")
    cases = [
        {
            "name": "child absent",
            "input": {"kind_vorhanden": False},
            "output": {"result": True},
        },
        {
            "name": "child present",
            "input": {"kind_vorhanden": True},
            "output": {"result": False},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test")


def test_unrelated_excluding_selector_cannot_witness_source_exception():
    source = "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt."
    content = _exception_control_content("if bonus_applies: false else: true")
    cases = [
        {
            "name": "without bonus",
            "input": {"bonus_applies": False},
            "output": {"result": True},
        },
        {
            "name": "with bonus",
            "input": {"bonus_applies": True},
            "output": {"result": False},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test")


def test_generic_exception_name_cannot_witness_unrelated_source_condition():
    source = "(1) The claim does not apply when a certificate is absent."
    content = _exception_control_content(
        "if age_requirement_exception_applies: false else: true"
    )
    cases = [
        {
            "name": "ordinary age rule",
            "input": {"age_requirement_exception_applies": False},
            "output": {"result": True},
        },
        {
            "name": "age exception",
            "input": {"age_requirement_exception_applies": True},
            "output": {"result": False},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test")


def test_federal_except_tokens_cannot_bind_age_witness_to_joint_return_clause():
    age_selector = (
        "resident_individual_meets_all_federal_eitc_qualifications_except_age"
    )
    joint_return_condition = (
        "if the claimant is married, except for a claimant who files as a head "
        "of household or surviving spouse for federal income tax purposes, the "
        "claimant shall file a joint return"
    )
    age_qualification_condition = (
        "The resident individual shall meet all qualifications, except for the "
        "minimum or maximum age, for the federal earned income tax credit in "
        "order to be eligible for the credit"
    )

    assert not completeness_module._source_exception_selector_is_relevant(
        joint_return_condition,
        age_selector,
    )
    assert completeness_module._source_exception_selector_is_relevant(
        age_qualification_condition,
        age_selector,
    )
    assert (
        completeness_module._source_exception_condition_text(
            age_qualification_condition
        )
        == age_qualification_condition
    )


@pytest.mark.parametrize("reverse_clause_order", [False, True])
def test_age_qualification_witness_is_not_allocated_to_joint_return_clause(
    reverse_clause_order: bool,
):
    joint_selector = "married_claimant_joint_return_requirement_satisfied"
    age_selector = (
        "resident_individual_meets_all_federal_eitc_qualifications_except_age"
    )
    joint_clause = (
        "To qualify for the credit, if the claimant is married, except for a "
        "claimant who files as a head of household or surviving spouse for "
        "federal income tax purposes, the claimant shall file a joint return."
    )
    age_clause = (
        "The resident individual shall meet all qualifications, except for the "
        "minimum or maximum age, for the federal earned income tax credit in "
        "order to be eligible for the credit."
    )
    ordered_clauses = (
        (age_clause, joint_clause)
        if reverse_clause_order
        else (
            joint_clause,
            age_clause,
        )
    )
    source = "(1) " + " ".join(ordered_clauses)
    content = _exception_control_content(f"{joint_selector} and {age_selector}")
    positive = {
        joint_selector: True,
        age_selector: True,
    }
    cases = [
        {
            "name": "all qualifications met",
            "input": positive,
            "output": {"result": True},
        },
        {
            "name": "non-age qualification not met",
            "input": {**positive, age_selector: False},
            "output": {"result": False},
        },
    ]

    missing_joint = _analyze(content, source, test_cases=cases)
    exception_issue = next(
        issue
        for issue in missing_joint.issues
        if "[complete-source-unit:tests] Source-stated exceptions" in issue
    )

    assert "joint return" in exception_issue
    assert "meet all qualifications" not in exception_issue

    cases.append(
        {
            "name": "joint return requirement not met",
            "input": {**positive, joint_selector: False},
            "output": {"result": False},
        }
    )
    complete = _analyze(content, source, test_cases=cases)

    assert not _has_issue(complete, "exception", "test")


def test_selector_relevance_uses_tokens_not_substrings():
    source = "(1) The claim does not apply when a separate status exists."
    content = _exception_control_content("if rate: false else: true")
    cases = [
        {
            "name": "rate false",
            "input": {"rate": False},
            "output": {"result": True},
        },
        {
            "name": "rate true",
            "input": {"rate": True},
            "output": {"result": False},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not completeness_module._source_exception_selector_is_relevant(
        "at a separate status",
        "rate",
    )
    assert _has_issue(result, "exception", "test")


@pytest.mark.parametrize(
    ("exception_value", "expected_issue"),
    [(5, True), (0, False)],
)
def test_explicit_numeric_zero_exception_reaches_zero(
    exception_value: int,
    expected_issue: bool,
):
    source = "(1) Im Ausnahmefall beträgt der Betrag 0 Euro."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: zero_amount
    kind: parameter
    source: de/statute/estg/32a(1)
    versions: [{formula: 0}]
  - name: result
    kind: derived
    source: de/statute/estg/32a(1)
    versions:
      - formula: 'if exception_applies: EXCEPTION_VALUE else: 10'
""".replace("EXCEPTION_VALUE", str(exception_value))
    cases = [
        {
            "name": "ordinary",
            "input": {"exception_applies": False},
            "output": {"result": 10},
        },
        {
            "name": "exception",
            "input": {"exception_applies": True},
            "output": {"result": exception_value},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test") is expected_issue


@pytest.mark.parametrize(
    "source",
    [
        "(1) Im Ausnahmefall beträgt der Betrag 0,5 Euro.",
        "(1) In the exception case, the amount equals 0.5 euros.",
    ],
)
def test_nonzero_decimal_exception_is_not_classified_as_zero(source: str):
    content = _exception_control_content(
        "if exception_applies: exception_amount else: 10",
        extra_rules="""\
  - name: exception_amount
    kind: parameter
    source: de/statute/estg/32a(1)
    versions: [{formula: 0.5}]
""",
    )
    cases = [
        {
            "name": "ordinary",
            "input": {"exception_applies": False},
            "output": {"result": 10},
        },
        {
            "name": "exception",
            "input": {"exception_applies": True},
            "output": {"result": 0.5},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


def test_generic_deviation_may_increase_the_principal_output():
    source = "(1) Abweichend von der Hauptregel erhöht eine Ausnahme den Betrag."
    content = _exception_control_content(
        "if exception_applies: income + bonus else: income"
    )
    cases = [
        {
            "name": "ordinary",
            "input": {"exception_applies": False, "income": 10, "bonus": 5},
            "output": {"result": 10},
        },
        {
            "name": "exception",
            "input": {"exception_applies": True, "income": 10, "bonus": 5},
            "output": {"result": 15},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


def test_mixed_exception_clauses_keep_local_effect_requirements():
    source = """\
(1) Abweichend von der Hauptregel erhöht eine Ausnahme den Betrag.
Der Anspruch besteht, außer bei einer Befreiung.
"""
    content = _exception_control_content(
        "if exemption_applies: 0 else: if exception_applies: 20 else: 10"
    )
    cases = [
        {
            "name": "ordinary",
            "input": {
                "exception_applies": False,
                "exemption_applies": False,
            },
            "output": {"result": 10},
        },
        {
            "name": "increasing exception",
            "input": {
                "exception_applies": True,
                "exemption_applies": False,
            },
            "output": {"result": 20},
        },
        {
            "name": "blocking exemption",
            "input": {
                "exception_applies": False,
                "exemption_applies": True,
            },
            "output": {"result": 0},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


def test_derived_exception_selector_may_be_driven_by_one_numeric_input():
    source = "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: exemption_applies
    kind: derived
    dtype: bool
    source: de/statute/estg/32a(1)
    versions: [{formula: 'exemption_code == 1'}]
  - name: eligible
    kind: derived
    dtype: bool
    source: de/statute/estg/32a(1)
    versions: [{formula: 'not exemption_applies'}]
"""
    cases = [
        {
            "name": "ordinary code",
            "input": {"exemption_code": 0},
            "output": {"exemption_applies": False, "eligible": True},
        },
        {
            "name": "exemption code",
            "input": {"exemption_code": 1},
            "output": {"exemption_applies": True, "eligible": False},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


@pytest.mark.parametrize(
    ("formula", "ordinary_output", "exception_output", "expected_issue"),
    [
        (
            "if income > income_limit: false else: true",
            True,
            False,
            False,
        ),
        (
            "if income > income_limit: true else: false",
            False,
            True,
            True,
        ),
    ],
)
def test_numeric_predicate_can_directly_witness_exception(
    formula: str,
    ordinary_output: bool,
    exception_output: bool,
    expected_issue: bool,
):
    source = "(1) Der Anspruch gilt nicht, wenn Einkommen über 100 Euro liegt."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: income_limit
    kind: parameter
    source: de/statute/estg/32a(1)
    versions: [{formula: 100}]
  - name: eligible
    kind: derived
    dtype: bool
    source: de/statute/estg/32a(1)
    versions:
      - formula: 'FORMULA'
""".replace("FORMULA", formula)
    cases = [
        {
            "name": "ordinary endpoint",
            "input": {"income": 100},
            "output": {"eligible": ordinary_output},
        },
        {
            "name": "over-limit exception",
            "input": {"income": 101},
            "output": {"eligible": exception_output},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test") is expected_issue


def test_numeric_input_change_cannot_hide_identical_exception_branches():
    source = "(1) Der Anspruch gilt nicht, wenn Einkommen über 100 Euro liegt."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: income_limit
    kind: parameter
    source: de/statute/estg/32a(1)
    versions: [{formula: 100}]
  - name: result
    kind: derived
    source: de/statute/estg/32a(1)
    versions:
      - formula: 'if income > income_limit: income else: income'
"""
    cases = [
        {
            "name": "ordinary endpoint",
            "input": {"income": 100},
            "output": {"result": 100},
        },
        {
            "name": "over-limit",
            "input": {"income": 101},
            "output": {"result": 101},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test")


def test_judgment_exception_requires_holds_to_not_holds_effect():
    source = "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt."
    content = _exception_control_content(
        "if exemption_applies: not_holds else: holds"
    ).replace("kind: derived", "kind: derived\n    dtype: Judgment")
    cases = [
        {
            "name": "ordinary",
            "input": {"exemption_applies": False},
            "output": {"result": "holds"},
        },
        {
            "name": "exempt",
            "input": {"exemption_applies": True},
            "output": {"result": "not_holds"},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert not result.issues


def test_derived_exception_selector_wrong_polarity_is_rejected():
    source = "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: exemption_applies
    kind: derived
    dtype: bool
    source: de/statute/estg/32a(1)
    versions: [{formula: 'has_certificate'}]
  - name: eligible
    kind: derived
    dtype: bool
    source: de/statute/estg/32a(1)
    versions: [{formula: 'if exemption_applies: true else: false'}]
"""
    cases = [
        {
            "name": "ordinary",
            "input": {"has_certificate": False},
            "output": {"exemption_applies": False, "eligible": False},
        },
        {
            "name": "exempt",
            "input": {"has_certificate": True},
            "output": {"exemption_applies": True, "eligible": True},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test")


def test_derived_match_pattern_cannot_fall_back_to_identifier_text():
    source = """\
(1) Bei Status A ist der Betrag Einkommen * 2.
(2) Bei Status B ist der Betrag Einkommen * 3.
"""
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: special_status
    kind: derived
    dtype: String
    versions:
      - formula: 'if flag: "married" else: "joint"'
  - name: first_multiplier
    kind: parameter
    source: de/statute/estg/32a(1)
    versions: [{formula: 2}]
  - name: second_multiplier
    kind: parameter
    source: de/statute/estg/32a(2)
    versions: [{formula: 3}]
  - name: amount
    kind: derived
    source: de/statute/estg/32a(1); de/statute/estg/32a(2)
    versions:
      - formula: |-
          match status:
            special_status => income * first_multiplier
            "single" => income * second_multiplier
"""
    cases = [
        {
            "name": "identifier text falls through",
            "input": {"flag": True, "status": "special_status", "income": 10},
            "output": {"special_status": "married", "amount": 30},
        },
        {
            "name": "ordinary fallback",
            "input": {"flag": True, "status": "other", "income": 10},
            "output": {"special_status": "married", "amount": 30},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "formula branch", "distinct")


def test_exception_numeric_representations_do_not_fake_changed_effect():
    source = "(1) Der Anspruch gilt nicht, wenn eine Befreiung vorliegt."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: amount
    kind: derived
    source: de/statute/estg/32a(1)
    versions:
      - formula: 'if exemption_applies: 0 else: 0.0'
"""
    cases = [
        {
            "name": "ordinary",
            "input": {"exemption_applies": False},
            "output": {"amount": 0},
        },
        {
            "name": "exempt",
            "input": {"exemption_applies": True},
            "output": {"amount": 0.0},
        },
    ]

    result = _analyze(content, source, test_cases=cases)

    assert _has_issue(result, "exception", "test")


def test_opposite_bare_boolean_predicate_cannot_witness_boundary():
    source = "(1) Die Regel gilt für Einkommen bis 100 Euro."
    content = _boundary_control_content(
        formula="income > income_limit",
        limit_versions="""\
      - effective_from: '2026-01-01'
        formula: 100""",
    )
    case = {
        "name": "opposite endpoint",
        "period": "2026",
        "input": {"income": 100},
        "output": {"eligible": False},
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "boundary", "100")


def _scalar_snapshot_content(
    *,
    rule_name: str = "allowance_amount",
    value: int = 259,
) -> str:
    return f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: {rule_name}
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions:
      - effective_from: '2026-01-01'
        formula: {value}
"""


@pytest.mark.parametrize(
    "source",
    [
        "(1) Der Freibetrag Plus beträgt 259 Euro.",
        "(1) Unter diesem Gesetz beträgt der Freibetrag 259 Euro.",
        (
            "(1) Der Freibetrag beträgt 259 Euro, auch wenn die "
            "Veröffentlichung später erfolgt."
        ),
        (
            "(1) Der Freibetrag beträgt 259 Euro, selbst wenn die "
            "Veröffentlichung später erfolgt."
        ),
    ],
)
def test_unconditional_scalar_wording_remains_scalar_only(source: str):
    content = _scalar_snapshot_content()
    case = {
        "name": "scalar snapshot",
        "period": "2026",
        "input": {},
        "output": {"allowance_amount": 259},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues


def test_leading_effective_month_does_not_steal_later_scalar_as_boundary():
    source = "(1) Ab Januar beträgt der Freibetrag 259 Euro."
    content = _scalar_snapshot_content()
    case = {
        "name": "effective scalar snapshot",
        "period": "2026",
        "input": {},
        "output": {"allowance_amount": 259},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues


@pytest.mark.parametrize(
    "source",
    [
        (
            "(1) Für das Kalenderjahr 2026 beträgt der Zuschlag 259 Euro "
            "bis zu einem Einkommen von 2000 Euro."
        ),
        "(1) Der Anspruch gilt bis 2000 Euro Einkommen pro Jahr.",
    ],
)
def test_monetary_threshold_near_year_language_is_not_temporal(source: str):
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: tax_year
    kind: parameter
    dtype: Integer
    source: de/statute/estg/32a(1)
    versions: [{formula: 2026}]
  - name: supplement_amount
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions: [{formula: 259}]
  - name: annual_limit
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions: [{formula: 2000}]
"""
    case = {
        "name": "parameter snapshot",
        "input": {},
        "output": {
            "tax_year": 2026,
            "supplement_amount": 259,
            "annual_limit": 2000,
        },
    }

    result = _analyze(content, source, test_cases=[case])

    assert _has_issue(result, "formula-output", "control", "parameter-only")


@pytest.mark.parametrize(
    "source",
    [
        (
            "(1) Vorausgesetzt, dass Anspruchsberechtigung besteht, "
            "beträgt der Zuschlag 259 Euro."
        ),
        (
            "(1) Unter der Voraussetzung, dass Anspruchsberechtigung "
            "besteht, beträgt der Zuschlag 259 Euro."
        ),
        "(1) Bei Anspruchsberechtigung beträgt der Zuschlag 259 Euro.",
        ("(1) Der Zuschlag beträgt 259 Euro, soweit Anspruchsberechtigung besteht."),
    ],
)
def test_common_positive_conditions_control_scalar_outputs(source: str):
    content = _scalar_snapshot_content(
        rule_name="supplement_amount",
    )

    result = _analyze(content, source, test_cases=[])

    assert _has_issue(result, "formula-output", "control", "parameter-only")


def test_unter_anwendung_computation_is_not_a_numeric_upper_boundary():
    source = (
        "(1) Der Betrag ist unter Anwendung eines Faktors 2 auf das "
        "Einkommen zu ermitteln."
    )
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: factor
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions: [{formula: 2}]
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions: [{formula: 'income * factor'}]
"""
    case = {
        "name": "factor computation",
        "input": {"income": 10},
        "output": {"amount": 20},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues


@pytest.mark.parametrize(
    "rounding_text",
    [
        "auf volle Euro gerundet",
        "auf volle Euro zu runden",
    ],
)
def test_generic_german_rounding_requires_nearest_rounding_and_fractional_proof(
    rounding_text: str,
):
    source = f"(1) Der Betrag wird als Einkommen * 2 berechnet und ist {rounding_text}."
    unrounded = _single_rounding_content("income * multiplier")
    rounded = _single_rounding_content(
        "floor(income * multiplier + 0.5)",
    )
    fractional_case = {
        "name": "nearest fractional result",
        "period": "2026",
        "input": {"income": 10.25},
        "output": {"amount": 21},
    }
    integral_case = {
        "name": "integral result only",
        "period": "2026",
        "input": {"income": 10},
        "output": {"amount": 20},
    }

    missing_operator = _analyze(
        unrounded,
        source,
        test_cases=[
            {
                **fractional_case,
                "output": {"amount": 20.5},
            }
        ],
    )
    missing_fractional_proof = _analyze(
        rounded,
        source,
        test_cases=[integral_case],
    )
    complete = _analyze(
        rounded,
        source,
        test_cases=[fractional_case],
    )

    assert _has_issue(missing_operator, "rounding", "principal formula")
    assert _has_issue(missing_fractional_proof, "rounding", "fractional")
    assert not complete.issues


@pytest.mark.parametrize(
    ("source", "output"),
    [
        (
            (
                "(1) Der Zuschlag beträgt 259 Euro bis zu einem Einkommen "
                "von 100 Euro nach § 26."
            ),
            "de:statutes/estg/32a/1#income_limited_supplement",
        ),
        (
            (
                "Der Zuschlag beträgt 259 Euro bis zu einem Einkommen "
                "von 100 Euro nach § 26."
            ),
            "de:statutes/estg/32a#income_limited_supplement",
        ),
    ],
)
def test_precisely_deferred_control_is_not_rescanned(
    source: str,
    output: str,
):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  deferred_outputs:
    - output: {output}
      reason: >-
        The income-limited supplement cannot be computed until EStG section 26
        eligibility is available.
      blocked_by:
        - de:statutes/estg/26#eligibility
rules:
  - name: supplement_amount
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions: [{{formula: 259}}]
  - name: income_limit
    kind: parameter
    dtype: Money
    source: de/statute/estg/32a(1)
    versions: [{{formula: 100}}]
"""

    result = _analyze(content, source, test_cases=[])

    assert not result.issues


@pytest.mark.parametrize(
    "source",
    [
        "(1) Der Betrag ist Einkommen * 2. Unberührt bleibt § 26.",
        (
            "(1) Der Betrag ist Einkommen * 2. "
            "§ 26 regelt ausschließlich das Inkrafttreten."
        ),
    ],
)
def test_unrelated_source_citation_cannot_authenticate_typed_deferral(
    source: str,
):
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  deferred_outputs:
    - output: de:statutes/estg/32a/1#amount
      reason: >-
        The amount cannot be computed until the EStG section 26 base is
        available.
      blocked_by:
        - de:statutes/estg/26#base
rules:
  - name: factor
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions: [{formula: 2}]
"""

    result = _analyze(content, source, test_cases=[])

    assert _has_issue(result, "deferral", "dependency")


def _three_term_topology_content(formula: str) -> str:
    return f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: two
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions: [{{formula: 2}}]
  - name: three
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions: [{{formula: 3}}]
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions: [{{formula: {formula!r}}}]
"""


@pytest.mark.parametrize(
    ("source_expression", "correct_formula", "correct_output", "wrong_formula"),
    [
        (
            "(Einkommen * 2 + 3)",
            "income * two + three",
            23,
            "income + two * three",
        ),
        (
            "Einkommen * (2 + 3)",
            "income * (two + three)",
            50,
            "income * two + three",
        ),
    ],
)
def test_parenthesized_source_formula_preserves_topology(
    source_expression: str,
    correct_formula: str,
    correct_output: int,
    wrong_formula: str,
):
    source = f"(1) Der Betrag wird als {source_expression} berechnet."
    correct = _analyze(
        _three_term_topology_content(correct_formula),
        source,
        test_cases=[
            {
                "name": "parenthesized source computation",
                "input": {"income": 10},
                "output": {"amount": correct_output},
            }
        ],
    )
    wrong = _analyze(
        _three_term_topology_content(wrong_formula),
        source,
        test_cases=[
            {
                "name": "different parenthesized computation",
                "input": {"income": 10},
                "output": {"amount": 16 if source_expression.startswith("(") else 23},
            }
        ],
    )

    assert not correct.issues
    assert _has_issue(wrong, "formula branch")


def test_formula_topology_accepts_associative_regrouping():
    source = "(1) Der Betrag wird als Einkommen + 2 + 3 berechnet."
    content = _three_term_topology_content("income + (two + three)")
    case = {
        "name": "associatively regrouped sum",
        "input": {"income": 10},
        "output": {"amount": 15},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues


def test_formula_topology_preserves_distinct_source_variables():
    source = "(1) Der Betrag wird als Einkommen * Partnereinkommen + 3 berechnet."
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: supplement
    kind: parameter
    dtype: Decimal
    source: de/statute/estg/32a(1)
    versions: [{formula: 3}]
  - name: amount
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions: [{formula: FORMULA}]
"""
    case = {
        "name": "two distinct inputs",
        "input": {"income": 10, "spouse_income": 4},
    }
    correct = _analyze(
        content.replace("FORMULA", "income * spouse_income + supplement"),
        source,
        test_cases=[{**case, "output": {"amount": 43}}],
    )
    duplicated = _analyze(
        content.replace("FORMULA", "income * income + supplement"),
        source,
        test_cases=[{**case, "output": {"amount": 103}}],
    )

    assert not correct.issues
    assert _has_issue(duplicated, "formula branch")


@pytest.mark.parametrize(
    "source",
    [
        (
            "(1) Der nach Maßgabe des Absatzes 2 festgelegte Freibetrag "
            "beträgt 259 Euro."
        ),
        ("(1) Der nach den Absätzen 2 und 3 festgelegte Freibetrag beträgt 259 Euro."),
    ],
)
def test_inflected_absatz_references_are_not_numeric_inventory(source: str):
    content = _scalar_snapshot_content()
    case = {
        "name": "scalar with structural cross-reference",
        "period": "2026",
        "input": {},
        "output": {"allowance_amount": 259},
    }

    result = _analyze(content, source, test_cases=[case])

    assert not result.issues
    assert result.source_numeric_occurrence_count == 1


def test_worded_arithmetic_chain_preserves_formula_topology():
    source = "(1) Der Betrag ist Einkommen mal 2 plus 3."
    correct = _analyze(
        _three_term_topology_content("income * two + three"),
        source,
        test_cases=[
            {
                "name": "worded source formula",
                "input": {"income": 10},
                "output": {"amount": 23},
            }
        ],
    )
    wrong = _analyze(
        _three_term_topology_content("income + two * three"),
        source,
        test_cases=[
            {
                "name": "different worded formula",
                "input": {"income": 10},
                "output": {"amount": 16},
            }
        ],
    )

    assert not correct.issues
    assert _has_issue(wrong, "formula branch")


@pytest.mark.parametrize(
    "source",
    [
        "(1) Das Einkommen ist auf volle Euro gerundet.",
        (
            "(1) Der Betrag entspricht dem Einkommen. "
            "Das Ergebnis ist kaufmännisch zu runden."
        ),
    ],
)
def test_nearest_rounding_offset_cannot_cancel_itself(source: str):
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: rounded_income
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    versions: [{formula: FORMULA}]
"""
    correct = _analyze(
        content.replace("FORMULA", "floor(income + 0.5)"),
        source,
        test_cases=[
            {
                "name": "nearest result",
                "input": {"income": 10.75},
                "output": {"rounded_income": 11},
            }
        ],
    )
    cancelled = _analyze(
        content.replace("FORMULA", "floor(income - 0.5 + 0.5)"),
        source,
        test_cases=[
            {
                "name": "cancelled offset",
                "input": {"income": 10.75},
                "output": {"rounded_income": 10},
            }
        ],
    )

    assert not correct.issues
    assert _has_issue(cancelled, "rounding", "fractional")
