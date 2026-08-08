"""Focused tests for structured deferred-output apply contracts."""

import argparse
from pathlib import Path

import pytest

from axiom_encode.cli import (
    _DeferredOutputReviewContract,
    _parse_deferred_output_review_contract_json,
    _required_deferred_output_contract_issues,
)

OUTPUT = "us-la:statutes/47/295/a#individual_louisiana_income_tax_amount"
REASON = (
    "us-la/statute/47:295(a) requires the individual Louisiana income tax amount "
    "to be determined in accordance with R.S. 47:32, but the available R.S. "
    "47:32 RuleSpec exports only the individual income tax rate and lacks an "
    "executable tax-amount computation accepting the applicable Louisiana income "
    "or net-income tax base."
)
CONTRACT = _DeferredOutputReviewContract(
    citation="us-la/statute/47:295",
    rulespec_path="us-la/statutes/47/295.yaml",
    required_deferred_outputs=((OUTPUT, REASON),),
)


def _issues(path: Path) -> list[str]:
    return _required_deferred_output_contract_issues(
        path,
        CONTRACT,
        citation="us-la/statute/47:295",
        rulespec_path="us-la/statutes/47/295.yaml",
    )


def test_required_deferred_output_contract_accepts_exact_yaml_decoded_pair(
    tmp_path: Path,
) -> None:
    rulespec = tmp_path / "295.yaml"
    rulespec.write_text(
        "module:\n"
        "  deferred_outputs:\n"
        f"    - output: {OUTPUT}\n"
        "      reason: >-\n"
        f"        {REASON}\n",
        encoding="utf-8",
    )

    assert _issues(rulespec) == []


@pytest.mark.parametrize(
    "candidate_reason",
    [
        "R.S. 47:32 is not yet executable.",
        REASON.replace("lacks", "does not provide"),
        REASON + " ",
        REASON + "\n",
    ],
)
def test_required_deferred_output_contract_rejects_reason_drift(
    tmp_path: Path,
    candidate_reason: str,
) -> None:
    rulespec = tmp_path / "295.yaml"
    rulespec.write_text(
        "module:\n  deferred_outputs:\n"
        f"    - output: {OUTPUT}\n"
        f"      reason: {candidate_reason!r}\n",
        encoding="utf-8",
    )

    issues = _issues(rulespec)

    assert len(issues) == 1
    assert "byte-for-byte" in issues[0]


def test_required_deferred_output_contract_rejects_duplicate_output(
    tmp_path: Path,
) -> None:
    rulespec = tmp_path / "295.yaml"
    rulespec.write_text(
        "module:\n"
        "  deferred_outputs:\n"
        f"    - output: {OUTPUT}\n"
        f"      reason: {REASON!r}\n"
        f"    - output: {OUTPUT}\n"
        f"      reason: {REASON!r}\n",
        encoding="utf-8",
    )

    issues = _issues(rulespec)

    assert len(issues) == 1
    assert "found 2" in issues[0]


def test_parse_review_contract_rejects_duplicate_outputs() -> None:
    raw = (
        '{"schema":"axiom-encode/review-contract/v1",'
        '"citation":"us-la/statute/47:295",'
        '"rulespec_path":"us-la/statutes/47/295.yaml",'
        '"required_deferred_outputs":['
        '{"output":"x","reason":"one"},'
        '{"output":"x","reason":"two"}]}'
    )

    with pytest.raises(argparse.ArgumentTypeError, match="outputs must be unique"):
        _parse_deferred_output_review_contract_json(raw)


def test_parse_review_contract_rejects_empty_contract() -> None:
    raw = (
        '{"schema":"axiom-encode/review-contract/v1",'
        '"citation":"us-la/statute/47:295",'
        '"rulespec_path":"us-la/statutes/47/295.yaml",'
        '"required_deferred_outputs":[]}'
    )

    with pytest.raises(argparse.ArgumentTypeError, match="nonempty array"):
        _parse_deferred_output_review_contract_json(raw)


@pytest.mark.parametrize(
    ("citation", "rulespec_path"),
    [
        ("us-la/statute/47:295(a)", "us-la/statutes/47/295.yaml"),
        ("us-la/statute/47:295", "us-la/statutes/47/295/a.yaml"),
    ],
)
def test_required_deferred_output_contract_rejects_binding_drift(
    tmp_path: Path,
    citation: str,
    rulespec_path: str,
) -> None:
    candidate = tmp_path / "295.yaml"
    candidate.write_text("module: {}\n", encoding="utf-8")

    issues = _required_deferred_output_contract_issues(
        candidate,
        CONTRACT,
        citation=citation,
        rulespec_path=rulespec_path,
    )

    assert len(issues) == 1
    assert "signed dispatch review contract" in issues[0]
