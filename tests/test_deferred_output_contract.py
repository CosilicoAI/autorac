"""Focused tests for structured deferred-output apply contracts."""

import argparse
from pathlib import Path

import pytest

from axiom_encode.cli import (
    _DeferredOutputReviewContract,
    _parse_deferred_output_review_contract_json,
    _required_deferred_output_contract_issues,
)
from axiom_encode.harness.evals import _format_required_deferred_output_contracts

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


def test_required_contract_prompt_preserves_exact_pair() -> None:
    rendered = _format_required_deferred_output_contracts(
        CONTRACT.required_deferred_outputs
    )

    assert '"output":"' + OUTPUT + '"' in rendered
    assert '"reason":"' + REASON + '"' in rendered
    assert "every character" in rendered


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


@pytest.mark.parametrize(
    "document",
    [
        "rules: []\n",
        "module:\n  deferred_outputs: invalid\n",
        "module:\n  deferred_outputs:\n    - invalid\n",
        "module:\n  deferred_outputs:\n    - output: 47\n      reason: exact\n",
        (
            "module:\n  deferred_outputs:\n"
            f"    - output: {OUTPUT.upper()}\n      reason: {REASON!r}\n"
        ),
        (
            "module:\n  deferred_outputs:\n"
            f"    - output: '{OUTPUT} '\n      reason: {REASON!r}\n"
        ),
        (
            "module:\n  deferred_outputs:\n"
            f"    - output: {OUTPUT}\n      reason: {REASON.lower()!r}\n"
        ),
        (
            "module:\n  deferred_outputs:\n"
            f"    - output: {OUTPUT}\n"
            f"      reason: {REASON.replace('47:32,', '47:32;')!r}\n"
        ),
        (
            "module:\n  deferred_outputs:\n"
            f"    - output: {OUTPUT}\n"
            f"      reason: {REASON.replace(' but ', '  but ')!r}\n"
        ),
        (
            "module:\n  deferred_outputs:\n"
            f"    - output: {OUTPUT}\n"
            "      reason: |\n"
            f"        {REASON}\n"
        ),
        (
            "module:\n  deferred_outputs:\n"
            f"    - output: {OUTPUT}\n      reason: 47\n"
        ),
    ],
)
def test_required_deferred_output_contract_rejects_adversarial_shape_and_drift(
    tmp_path: Path,
    document: str,
) -> None:
    candidate = tmp_path / "295.yaml"
    candidate.write_text(document, encoding="utf-8")

    assert _issues(candidate)


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
    "raw",
    [
        '{"schema":"axiom-encode/review-contract/v1",'
        '"citation":"us-la/statute/47:295",'
        '"citation":"us-la/statute/47:295",'
        '"rulespec_path":"us-la/statutes/47/295.yaml",'
        '"required_deferred_outputs":[{"output":"x","reason":"one"}]}',
        '{"schema":"axiom-encode/review-contract/v1",'
        '"citation":"us-la/statute/47:295",'
        '"rulespec_path":"us-la/statutes/47/295.yaml",'
        '"required_deferred_outputs":['
        '{"output":"x","reason":"one","reason":"two"}]}',
    ],
)
def test_parse_review_contract_rejects_duplicate_json_keys(raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="duplicate JSON key"):
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
