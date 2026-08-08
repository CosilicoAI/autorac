"""Focused tests for structured companion-test apply contracts."""

import argparse
import json
from pathlib import Path

import pytest

from axiom_encode.cli import (
    _parse_deferred_output_review_contract_json,
    _required_deferred_output_contract_issues,
)
from axiom_encode.harness.evals import _format_required_test_case_contracts

CITATION = "us-la/statute/47:294"
RULESPEC_PATH = "us-la/statutes/47/294.yaml"
SINGLE = (
    "us-la:statutes/47/294#input."
    "federal_return_filing_status_is_single_or_married_separate"
)
JOINT = (
    "us-la:statutes/47/294#input."
    "federal_return_filing_status_is_joint_surviving_spouse_or_head_of_household"
)
PRINCIPAL = "us-la:statutes/47/294#standard_deduction"
CPI = (
    "us-la:statutes/47/294#input."
    "previous_calendar_year_cpi_u_percentage_increase"
)
PRIOR_JOINT = (
    "us-la:statutes/47/294#input."
    "prior_year_standard_deduction_joint_surviving_spouse_or_head_of_household"
)
PRIOR_SINGLE = (
    "us-la:statutes/47/294#input."
    "prior_year_standard_deduction_single_or_married_separate"
)
CASE = {
    "name": "2025 single individual or married separate standard deduction",
    "period": {
        "period_kind": "tax_year",
        "start": "2025-01-01",
        "end": "2025-12-31",
    },
    "input": {SINGLE: True, JOINT: False},
    "required_output": {PRINCIPAL: 12500},
}


def _raw_contract(*, test_cases: list[dict[str, object]] | None = None) -> str:
    return json.dumps(
        {
            "schema": "axiom-encode/review-contract/v2",
            "citation": CITATION,
            "rulespec_path": RULESPEC_PATH,
            "required_deferred_outputs": [],
            "required_test_cases": test_cases if test_cases is not None else [CASE],
        },
        separators=(",", ":"),
    )


def _write_candidate(tmp_path: Path, cases: list[dict[str, object]]) -> Path:
    rulespec = tmp_path / "294.yaml"
    rulespec.write_text("format: rulespec/v1\nmodule: {}\nrules: []\n")
    rulespec.with_name("294.test.yaml").write_text(
        json.dumps(cases),
        encoding="utf-8",
    )
    return rulespec


def _issues(rulespec: Path, raw: str | None = None) -> list[str]:
    contract = _parse_deferred_output_review_contract_json(raw or _raw_contract())
    return _required_deferred_output_contract_issues(
        rulespec,
        contract,
        citation=CITATION,
        rulespec_path=RULESPEC_PATH,
    )


def test_required_test_case_accepts_exact_input_and_required_output_subset(
    tmp_path: Path,
) -> None:
    candidate = {
        "name": CASE["name"],
        "period": CASE["period"],
        "input": CASE["input"],
        "output": {PRINCIPAL: 12500, "us-la:statutes/47/294#helper": 12500},
    }

    assert _issues(_write_candidate(tmp_path, [candidate])) == []


def test_required_test_case_rejects_pr_1253_extra_2025_inputs(tmp_path: Path) -> None:
    candidate = {
        "name": CASE["name"],
        "period": CASE["period"],
        "input": {
            **CASE["input"],
            CPI: 0,
            PRIOR_JOINT: 0,
            PRIOR_SINGLE: 0,
        },
        "output": {PRINCIPAL: 12500},
    }

    issues = _issues(_write_candidate(tmp_path, [candidate]))

    assert len(issues) == 1
    assert "input map does not exactly match" in issues[0]
    assert "unexpected:" in issues[0]


def test_required_test_case_contract_covers_all_four_294_cases(tmp_path: Path) -> None:
    contracts = []
    candidates = []
    for name, single, joint, expected in (
        (
            "2025 single individual or married separate standard deduction",
            True,
            False,
            12500,
        ),
        (
            "2025 joint surviving spouse or head of household standard deduction",
            False,
            True,
            25000,
        ),
        ("2025 no listed filing status group fails closed", False, False, 0),
        ("2025 conflicting filing status groups fail closed", True, True, 0),
    ):
        contract = {
            "name": name,
            "period": CASE["period"],
            "input": {SINGLE: single, JOINT: joint},
            "required_output": {PRINCIPAL: expected},
        }
        contracts.append(contract)
        candidates.append(
            {
                "name": name,
                "period": CASE["period"],
                "input": {SINGLE: single, JOINT: joint},
                "output": {PRINCIPAL: expected},
            }
        )

    assert _issues(
        _write_candidate(tmp_path, candidates),
        _raw_contract(test_cases=contracts),
    ) == []

    for candidate in candidates:
        candidate["input"] = {
            **candidate["input"],
            CPI: 0,
            PRIOR_JOINT: 0,
            PRIOR_SINGLE: 0,
        }
    rejected = _issues(
        _write_candidate(tmp_path, candidates),
        _raw_contract(test_cases=contracts),
    )
    assert len(rejected) == 4
    assert all(
        all(key in issue for key in (CPI, PRIOR_JOINT, PRIOR_SINGLE))
        and "unexpected:" in issue
        for issue in rejected
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"period": {**CASE["period"], "end": "2026-12-31"}}, "period"),
        ({"output": {PRINCIPAL: 0}}, "missing or changes required output"),
    ],
)
def test_required_test_case_rejects_period_or_output_drift(
    tmp_path: Path,
    mutation: dict[str, object],
    expected: str,
) -> None:
    candidate = {
        "name": CASE["name"],
        "period": CASE["period"],
        "input": CASE["input"],
        "output": {PRINCIPAL: 12500},
        **mutation,
    }

    assert expected in "\n".join(_issues(_write_candidate(tmp_path, [candidate])))


@pytest.mark.parametrize("count", [0, 2])
def test_required_test_case_requires_exactly_one_named_case(
    tmp_path: Path,
    count: int,
) -> None:
    candidate = {
        "name": CASE["name"],
        "period": CASE["period"],
        "input": CASE["input"],
        "output": {PRINCIPAL: 12500},
    }

    issues = _issues(_write_candidate(tmp_path, [candidate] * count))

    assert len(issues) == 1
    assert f"found {count}" in issues[0]


def test_v2_review_contract_rejects_empty_contract() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="must require"):
        _parse_deferred_output_review_contract_json(_raw_contract(test_cases=[]))


@pytest.mark.parametrize("period_kind", ["month", "benefit_week"])
def test_v2_review_contract_accepts_engine_period_kinds(period_kind: str) -> None:
    case = {
        **CASE,
        "period": {
            "period_kind": period_kind,
            "start": "2025-01-01",
            "end": "2025-01-31",
        },
    }

    parsed = _parse_deferred_output_review_contract_json(
        _raw_contract(test_cases=[case])
    )
    assert parsed.required_test_cases[0].period["period_kind"] == period_kind


def test_v2_review_contract_rejects_duplicate_nested_json_key() -> None:
    raw = _raw_contract().replace(
        f'"{SINGLE}":true',
        f'"{SINGLE}":true,"{SINGLE}":false',
    )

    with pytest.raises(argparse.ArgumentTypeError, match="duplicate JSON key"):
        _parse_deferred_output_review_contract_json(raw)


def test_v2_review_contract_rejects_oversized_integer_token() -> None:
    raw = _raw_contract().replace("12500", "9" * 5000)

    with pytest.raises(argparse.ArgumentTypeError, match="valid JSON"):
        _parse_deferred_output_review_contract_json(raw)


def test_required_test_case_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    rulespec = tmp_path / "294.yaml"
    rulespec.write_text("format: rulespec/v1\nmodule: {}\nrules: []\n")
    rulespec.with_name("294.test.yaml").write_text(
        "- name: " + str(CASE["name"]) + "\n"
        "  period:\n"
        "    period_kind: tax_year\n"
        "    start: '2025-01-01'\n"
        "    end: '2025-12-31'\n"
        "  input:\n"
        f"    {SINGLE}: true\n"
        f"    {SINGLE}: false\n"
        f"    {JOINT}: false\n"
        "  output:\n"
        f"    {PRINCIPAL}: 12500\n",
        encoding="utf-8",
    )

    issues = _issues(rulespec)
    assert len(issues) == 1
    assert "duplicate key" in issues[0]


def test_required_test_case_rejects_unsigned_table_inputs(tmp_path: Path) -> None:
    candidate = {
        "name": CASE["name"],
        "period": CASE["period"],
        "input": CASE["input"],
        "tables": {
            "TaxUnit": [
                {
                    "id": "tax-unit-1",
                    "us-la:statutes/47/294#input.prior_year_deduction": 999,
                }
            ]
        },
        "output": {PRINCIPAL: 12500},
    }

    issues = _issues(_write_candidate(tmp_path, [candidate]))
    assert len(issues) == 1
    assert "unsigned runtime input field(s): tables" in issues[0]


def test_required_test_case_prompt_preserves_exact_contract() -> None:
    rendered = _format_required_test_case_contracts([CASE])

    assert json.dumps(CASE, separators=(",", ":"), sort_keys=True) in rendered
    assert "complete `input` map" in rendered
    assert "final repaired overlay" in rendered
