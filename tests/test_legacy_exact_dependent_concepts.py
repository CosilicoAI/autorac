from pathlib import Path

import pytest

from axiom_encode.legacy_exact_dependent_concepts import (
    LegacyExactDependentConceptError,
    canonicalized_concept_replacements,
    derive_exact_dependent_parameter_replacements,
    validate_exact_dependent_concept_rewrite,
)
from axiom_encode.rulespec_path_migration import rewrite_exact_references

LEGACY_PATH = Path("us-la/statutes/47:297/4.yaml")
SUCCESSOR_PATH = Path("us-la/statutes/47/297/4.yaml")


def _legacy_parameter() -> bytes:
    return b"""format: rulespec/v1
rules:
  - name: later_low_income_unreduced_federal_credit_percentage
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '0001-01-01'
        formula: '0.50'
"""


def _successor_parameter(*, duplicate: bool = False) -> bytes:
    duplicate_rule = (
        """
  - name: other_low_income_percentage
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2007-01-01'
        formula: '0.50'
"""
        if duplicate
        else ""
    )
    return f"""format: rulespec/v1
rules:
  - name: low_income_unreduced_federal_credit_percentage
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2006-01-01'
        formula: '0.25'
      - effective_from: '2007-01-01'
        formula: '0.50'
{duplicate_rule}""".encode()


def _dependent(*, year: int = 2026, source_note: str = "") -> bytes:
    return f"""format: rulespec/v1
module:
  summary: {source_note or "Exact dependent"}
imports:
  - us-la:statutes/47:297/4#later_low_income_unreduced_federal_credit_percentage
rules:
  - name: nominal_child_care_credit
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '{year}-01-01'
        effective_to: '{year}-12-31'
        formula: amount * later_low_income_unreduced_federal_credit_percentage
""".encode()


def _retained_modules(*, duplicate: bool = False):
    return (
        (
            LEGACY_PATH,
            SUCCESSOR_PATH,
            _legacy_parameter(),
            _successor_parameter(duplicate=duplicate),
        ),
    )


def test_derives_unique_active_period_equivalent_parameter_rewrite():
    replacements = derive_exact_dependent_parameter_replacements(
        dependent_primary_raw=_dependent(),
        retained_modules=_retained_modules(),
    )

    assert replacements == {
        "us-la:statutes/47:297/4#later_low_income_unreduced_federal_credit_percentage": (
            "us-la:statutes/47/297/4#low_income_unreduced_federal_credit_percentage"
        ),
        "later_low_income_unreduced_federal_credit_percentage": (
            "low_income_unreduced_federal_credit_percentage"
        ),
    }


def test_rejects_parameter_that_differs_during_dependent_use_window():
    with pytest.raises(
        LegacyExactDependentConceptError,
        match="no unique active-period-equivalent canonical successor",
    ):
        derive_exact_dependent_parameter_replacements(
            dependent_primary_raw=_dependent(year=2006),
            retained_modules=_retained_modules(),
        )


def test_rejects_ambiguous_equivalent_successors():
    with pytest.raises(
        LegacyExactDependentConceptError,
        match="candidates: low_income_unreduced.*other_low_income_percentage",
    ):
        derive_exact_dependent_parameter_replacements(
            dependent_primary_raw=_dependent(),
            retained_modules=_retained_modules(duplicate=True),
        )


def test_rejects_parameter_effective_to_not_enforced_by_scalar_runtime():
    successor = _successor_parameter().replace(
        b"        formula: '0.50'\n",
        b"        effective_to: '2030-12-31'\n        formula: '0.50'\n",
    )
    with pytest.raises(
        LegacyExactDependentConceptError,
        match="effective_to.*cannot be proved",
    ):
        derive_exact_dependent_parameter_replacements(
            dependent_primary_raw=_dependent(),
            retained_modules=(
                (LEGACY_PATH, SUCCESSOR_PATH, _legacy_parameter(), successor),
            ),
        )


@pytest.mark.parametrize(
    "dependent",
    [
        _dependent().replace(
            b"imports:\n",
            b"imports:\n"
            b"  - other:module#later_low_income_unreduced_federal_credit_percentage\n",
        ),
        _dependent().replace(
            b"rules:\n",
            b"rules:\n"
            b"  - name: later_low_income_unreduced_federal_credit_percentage\n"
            b"    kind: parameter\n"
            b"    dtype: Rate\n"
            b"    versions:\n"
            b"      - effective_from: '2026-01-01'\n"
            b"        formula: '0.50'\n",
        ),
    ],
)
def test_rejects_ambiguous_imported_symbol(dependent):
    with pytest.raises(
        LegacyExactDependentConceptError,
        match="not an unambiguous imported symbol",
    ):
        derive_exact_dependent_parameter_replacements(
            dependent_primary_raw=dependent,
            retained_modules=_retained_modules(),
        )


def test_validates_rewrite_is_limited_to_import_and_formula_surfaces():
    path_replacements = {"us-la:statutes/47:297/4": "us-la:statutes/47/297/4"}
    concepts = derive_exact_dependent_parameter_replacements(
        dependent_primary_raw=_dependent(),
        retained_modules=_retained_modules(),
    )
    path_rewritten, _counts = rewrite_exact_references(_dependent(), path_replacements)
    canonical = canonicalized_concept_replacements(
        concepts, path_replacements=path_replacements
    )
    concept_rewritten, _counts = rewrite_exact_references(path_rewritten, canonical)

    validate_exact_dependent_concept_rewrite(
        path_rewritten_raw=path_rewritten,
        concept_rewritten_raw=concept_rewritten,
        replacements=canonical,
        primary=True,
    )


def test_validates_proof_import_target_and_output_rewrite():
    raw = _dependent().replace(
        b"        formula: amount * later_low_income_unreduced_federal_credit_percentage\n",
        b"        formula: amount * later_low_income_unreduced_federal_credit_percentage\n"
        b"    metadata:\n"
        b"      proof:\n"
        b"        atoms:\n"
        b"          - kind: import\n"
        b"            import:\n"
        b"              target: us-la:statutes/47:297/4#later_low_income_unreduced_federal_credit_percentage\n"
        b"              output: later_low_income_unreduced_federal_credit_percentage\n",
    )
    path_replacements = {"us-la:statutes/47:297/4": "us-la:statutes/47/297/4"}
    concepts = derive_exact_dependent_parameter_replacements(
        dependent_primary_raw=raw,
        retained_modules=_retained_modules(),
    )
    path_rewritten, _counts = rewrite_exact_references(raw, path_replacements)
    canonical = canonicalized_concept_replacements(
        concepts, path_replacements=path_replacements
    )
    concept_rewritten, _counts = rewrite_exact_references(path_rewritten, canonical)

    validate_exact_dependent_concept_rewrite(
        path_rewritten_raw=path_rewritten,
        concept_rewritten_raw=concept_rewritten,
        replacements=canonical,
        primary=True,
    )


def test_rejects_rewrite_of_unrelated_summary_text():
    raw = _dependent(source_note="later_low_income_unreduced_federal_credit_percentage")
    path_replacements = {"us-la:statutes/47:297/4": "us-la:statutes/47/297/4"}
    concepts = derive_exact_dependent_parameter_replacements(
        dependent_primary_raw=raw,
        retained_modules=_retained_modules(),
    )
    path_rewritten, _counts = rewrite_exact_references(raw, path_replacements)
    canonical = canonicalized_concept_replacements(
        concepts, path_replacements=path_replacements
    )
    concept_rewritten, _counts = rewrite_exact_references(path_rewritten, canonical)

    with pytest.raises(
        LegacyExactDependentConceptError,
        match="unauthorized YAML surface",
    ):
        validate_exact_dependent_concept_rewrite(
            path_rewritten_raw=path_rewritten,
            concept_rewritten_raw=concept_rewritten,
            replacements=canonical,
            primary=True,
        )


def test_rejects_global_rewrite_of_quoted_formula_literal():
    raw = _dependent().replace(
        b"amount * later_low_income_unreduced_federal_credit_percentage",
        b'if label == "later_low_income_unreduced_federal_credit_percentage" then '
        b"amount * later_low_income_unreduced_federal_credit_percentage else 0",
    )
    path_replacements = {"us-la:statutes/47:297/4": "us-la:statutes/47/297/4"}
    concepts = derive_exact_dependent_parameter_replacements(
        dependent_primary_raw=raw,
        retained_modules=_retained_modules(),
    )
    path_rewritten, _counts = rewrite_exact_references(raw, path_replacements)
    canonical = canonicalized_concept_replacements(
        concepts, path_replacements=path_replacements
    )
    concept_rewritten, _counts = rewrite_exact_references(path_rewritten, canonical)

    with pytest.raises(
        LegacyExactDependentConceptError,
        match="unauthorized YAML surface",
    ):
        validate_exact_dependent_concept_rewrite(
            path_rewritten_raw=path_rewritten,
            concept_rewritten_raw=concept_rewritten,
            replacements=canonical,
            primary=True,
        )
