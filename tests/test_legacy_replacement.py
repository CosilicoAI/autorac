from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode.cli import (
    _require_locked_legacy_replacement_base,
    _resolve_legacy_replacement_contract,
)
from axiom_encode.legacy_replacement import (
    LEGACY_MANIFEST_SCHEMA,
    legacy_manual_manifest_issues,
    legacy_source_verification_citation_paths,
    receipt_identity_payload,
    receipt_identity_sha256,
)


@pytest.mark.parametrize(
    "verification",
    [
        {},
        {"corpus_citation_path": ""},
        {"corpus_citation_path": 1},
        {"corpus_citation_paths": []},
        {"corpus_citation_paths": ["us-la/statute/47/32", "us-la/statute/47/32"]},
        {"corpus_citation_paths": [1]},
        {
            "corpus_citation_path": "us-la/statute/47/32",
            "corpus_citation_paths": ["us-la/statute/47/32"],
        },
    ],
)
def test_legacy_citation_admission_rejects_ambiguous_shapes(verification) -> None:
    assert legacy_source_verification_citation_paths(verification) == ()


@pytest.mark.parametrize(
    "verification",
    [
        {"corpus_citation_path": "us-la/statute/47/32"},
        {"corpus_citation_paths": ["us-la/statute/47/32"]},
    ],
)
def test_legacy_citation_admission_accepts_one_exclusive_source(verification) -> None:
    assert legacy_source_verification_citation_paths(verification) == (
        "us-la/statute/47/32",
    )


def test_legacy_citation_admission_preserves_historical_source_order() -> None:
    assert legacy_source_verification_citation_paths(
        {
            "corpus_citation_paths": [
                "us-me/guidance/revenue/rate-schedule",
                "us-me/statute/36/5111",
            ]
        }
    ) == (
        "us-me/guidance/revenue/rate-schedule",
        "us-me/statute/36/5111",
    )


def _manual_manifest() -> dict[str, object]:
    return {
        "schema_version": LEGACY_MANIFEST_SCHEMA,
        "tool": "axiom-encode sign-applied-files",
        "backend": "manual",
        "runner": "manual-attestation",
        "manual_exception": "composition",
        "applied_files": [
            {"path": "us-la/statutes/47:32.yaml", "sha256": "a" * 64},
            {"path": "us-la/statutes/47:32.test.yaml", "sha256": "b" * 64},
        ],
        "signature": {
            "algorithm": "hmac-sha256",
            "key_id": "historical-v1",
            "value": "opaque-untrusted-evidence",
        },
    }


def test_manual_manifest_is_admitted_only_as_exact_deletion_evidence() -> None:
    expected = {
        "us-la/statutes/47:32.yaml": "a" * 64,
        "us-la/statutes/47:32.test.yaml": "b" * 64,
    }
    assert (
        legacy_manual_manifest_issues(
            _manual_manifest(),
            expected_files=expected,
        )
        == []
    )


def test_unmarked_manual_manifest_is_only_admitted_for_explicit_in_place_use() -> None:
    payload = _manual_manifest()
    payload["manual_exception"] = None
    expected = {
        "us-la/statutes/47:32.yaml": "a" * 64,
        "us-la/statutes/47:32.test.yaml": "b" * 64,
    }
    assert any(
        "no manual exception" in issue
        for issue in legacy_manual_manifest_issues(
            payload,
            expected_files=expected,
        )
    )
    assert (
        legacy_manual_manifest_issues(
            payload,
            expected_files=expected,
            allow_unmarked_manual_exception=True,
        )
        == []
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"backend": "codex"}, "backend is not manual"),
        ({"backend": "deterministic"}, "backend is not manual"),
        ({"runner": "unknown"}, "runner is not manual-attestation"),
        ({"validation_execution": {}}, "unsupported provenance"),
        (
            {"signature": {"algorithm": "ed25519-domain-v1"}},
            "unknown signature provenance",
        ),
    ],
)
def test_manual_manifest_cannot_be_elevated_to_generated_provenance(
    mutation,
    match,
) -> None:
    payload = _manual_manifest()
    payload.update(mutation)
    issues = legacy_manual_manifest_issues(
        payload,
        expected_files={
            "us-la/statutes/47:32.yaml": "a" * 64,
            "us-la/statutes/47:32.test.yaml": "b" * 64,
        },
    )
    assert any(match in issue for issue in issues)


def test_receipt_identity_binds_every_transaction_dimension() -> None:
    payload = receipt_identity_payload(
        base_commit="a" * 40,
        base_tree="b" * 40,
        legacy_manifest_sha256="c" * 64,
        model_manifest_sha256="d" * 64,
        live_files=[{"path": "us-la/statutes/47/32.yaml", "sha256": "e" * 64}],
        deleted_files=[{"path": "us-la/statutes/47:32.yaml", "deleted": True}],
        rewrites=[],
        scheduled_dependents=[],
    )
    baseline = receipt_identity_sha256(payload)
    for field in (
        "base_commit",
        "base_tree",
        "legacy_manifest_sha256",
        "model_manifest_sha256",
        "live_files",
        "deleted_files",
        "rewrites",
        "scheduled_dependents",
    ):
        changed = copy.deepcopy(payload)
        changed[field] = f"changed-{field}"
        assert receipt_identity_sha256(changed) != baseline


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _legacy_checkout(tmp_path: Path) -> tuple[Path, Path, SimpleNamespace]:
    checkout = tmp_path / "rulespec-us"
    content_root = checkout / "us-la"
    primary = content_root / "statutes/47:32.yaml"
    companion = content_root / "statutes/47:32.test.yaml"
    primary.parent.mkdir(parents=True)
    primary.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_paths:\n"
        "      - us-la/statute/47:32\n"
        "rules: []\n"
    )
    companion.write_text("[]\n")
    expected_files = {
        "us-la/statutes/47:32.yaml": hashlib.sha256(primary.read_bytes()).hexdigest(),
        "us-la/statutes/47:32.test.yaml": hashlib.sha256(
            companion.read_bytes()
        ).hexdigest(),
    }
    manifest = checkout / ".axiom/encoding-manifests/us-la/statutes/47:32.json"
    manifest.parent.mkdir(parents=True)
    payload = _manual_manifest()
    payload["applied_files"] = [
        {"path": path, "sha256": digest} for path, digest in expected_files.items()
    ]
    manifest.write_text(json.dumps(payload) + "\n")
    index = checkout / ".axiom/index/provisions_to_rules.json"
    index.parent.mkdir(parents=True)
    index.write_text('{"module":"us-la:statutes/47:32"}\n')
    _git(checkout, "init", "-q")
    _git(checkout, "config", "user.email", "test@example.com")
    _git(checkout, "config", "user.name", "Test")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "base")
    source = SimpleNamespace(
        requested="us-la/statute/47:32",
        citation_path="us-la/statute/47:32",
        body="official body",
        resolved_source=object(),
    )
    return checkout, content_root, source


def _in_place_legacy_checkout(
    tmp_path: Path,
) -> tuple[Path, Path, SimpleNamespace]:
    checkout = tmp_path / "rulespec-us"
    content_root = checkout / "us-me"
    primary = content_root / "policies/income_tax/pilot_liability_pipeline.yaml"
    companion = primary.with_name("pilot_liability_pipeline.test.yaml")
    primary.parent.mkdir(parents=True)
    primary.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_paths:\n"
        "      - us-me/guidance/revenue/rate-schedule\n"
        "      - us-me/statute/36/5111\n"
        "rules: []\n"
    )
    companion.write_text("[]\n")
    expected_files = {
        primary.relative_to(checkout).as_posix(): hashlib.sha256(
            primary.read_bytes()
        ).hexdigest(),
        companion.relative_to(checkout).as_posix(): hashlib.sha256(
            companion.read_bytes()
        ).hexdigest(),
    }
    manifest = (
        checkout
        / ".axiom/encoding-manifests/us-me/policies/income_tax/"
        "pilot_liability_pipeline.json"
    )
    manifest.parent.mkdir(parents=True)
    payload = _manual_manifest()
    payload["manual_exception"] = None
    payload["applied_files"] = [
        {"path": path, "sha256": digest} for path, digest in expected_files.items()
    ]
    manifest.write_text(json.dumps(payload) + "\n")
    _git(checkout, "init", "-q")
    _git(checkout, "config", "user.email", "test@example.com")
    _git(checkout, "config", "user.name", "Test")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "base")
    source = SimpleNamespace(
        requested="us-me/guidance/revenue/rate-schedule",
        citation_path="us-me/guidance/revenue/rate-schedule",
        body="official schedule",
        resolved_source=object(),
    )
    return checkout, content_root, source


def test_contract_binds_in_place_plural_source_and_unmarked_v1(
    tmp_path: Path,
) -> None:
    checkout, content_root, source = _in_place_legacy_checkout(tmp_path)
    relative = Path("us-me/policies/income_tax/pilot_liability_pipeline.yaml")
    with patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source):
        contract = _resolve_legacy_replacement_contract(
            source_raw=relative,
            destination_raw=relative,
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
        )

    assert contract.source == relative
    assert contract.destination == relative
    assert contract.rewrites == ()
    assert contract.scheduled_dependents == ()
    assert [item.path for item in contract.deleted_files] == [
        relative,
        relative.with_name("pilot_liability_pipeline.test.yaml"),
    ]


def test_contract_rejects_in_place_source_choice_outside_first_legacy_slot(
    tmp_path: Path,
) -> None:
    checkout, content_root, source = _in_place_legacy_checkout(tmp_path)
    source.requested = "us-me/statute/36/5111"
    source.citation_path = source.requested
    relative = Path("us-me/policies/income_tax/pilot_liability_pipeline.yaml")
    with (
        patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source),
        pytest.raises(ValueError, match="first historical source"),
    ):
        _resolve_legacy_replacement_contract(
            source_raw=relative,
            destination_raw=relative,
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
        )


def test_contract_rejects_in_place_path_dependents(tmp_path: Path) -> None:
    checkout, content_root, source = _in_place_legacy_checkout(tmp_path)
    relative = Path("us-me/policies/income_tax/pilot_liability_pipeline.yaml")
    with (
        patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source),
        pytest.raises(ValueError, match="cannot schedule path dependents"),
    ):
        _resolve_legacy_replacement_contract(
            source_raw=relative,
            destination_raw=relative,
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
            scheduled_dependent_paths=(relative.with_name("dependent.yaml"),),
        )


def test_contract_binds_clean_head_and_exact_metadata_rewrite(tmp_path: Path) -> None:
    checkout, content_root, source = _legacy_checkout(tmp_path)
    with patch(
        "axiom_encode.cli.resolve_corpus_source_unit",
        return_value=source,
    ):
        contract = _resolve_legacy_replacement_contract(
            source_raw=Path("us-la/statutes/47:32.yaml"),
            destination_raw=Path("us-la/statutes/47/32.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
        )

    assert contract.source == Path("us-la/statutes/47:32.yaml")
    assert contract.destination == Path("us-la/statutes/47/32.yaml")
    assert [item.path for item in contract.deleted_files] == [
        Path("us-la/statutes/47:32.yaml"),
        Path("us-la/statutes/47:32.test.yaml"),
    ]
    assert [item.path for item in contract.rewrites] == [
        Path(".axiom/index/provisions_to_rules.json")
    ]
    assert b"us-la:statutes/47/32" in contract.rewrites[0].raw


def test_contract_rejects_protected_dependent_rewrite(tmp_path: Path) -> None:
    checkout, content_root, source = _legacy_checkout(tmp_path)
    dependent = content_root / "policies/income_tax/dependent.yaml"
    dependent.parent.mkdir(parents=True)
    dependent.write_text(
        "format: rulespec/v1\nimports:\n  - us-la:statutes/47:32#amount\nrules: []\n"
    )
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "dependent")

    with (
        patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source),
        pytest.raises(ValueError, match="protected-dependent reencode"),
    ):
        _resolve_legacy_replacement_contract(
            source_raw=Path("us-la/statutes/47:32.yaml"),
            destination_raw=Path("us-la/statutes/47/32.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
        )


def test_contract_binds_only_explicit_scheduled_dependent(tmp_path: Path) -> None:
    checkout, content_root, source = _legacy_checkout(tmp_path)
    dependent = content_root / "policies/income_tax/dependent.yaml"
    dependent.parent.mkdir(parents=True)
    dependent.write_text(
        "format: rulespec/v1\nimports:\n  - us-la:statutes/47:32#amount\nrules: []\n"
    )
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "dependent")

    with patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source):
        contract = _resolve_legacy_replacement_contract(
            source_raw=Path("us-la/statutes/47:32.yaml"),
            destination_raw=Path("us-la/statutes/47/32.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
            scheduled_dependent_paths=(
                Path("us-la/policies/income_tax/dependent.yaml"),
            ),
        )

    assert [item.primary for item in contract.scheduled_dependents] == [
        Path("us-la/policies/income_tax/dependent.yaml")
    ]
    assert [item.path for item in contract.scheduled_dependents[0].files] == [
        Path("us-la/policies/income_tax/dependent.yaml")
    ]
    assert contract.scheduled_dependents[0].files[0].replacements == (
        {
            "from": "us-la:statutes/47:32",
            "to": "us-la:statutes/47/32",
            "count": 1,
        },
    )


def test_contract_rejects_dirty_or_existing_destination(tmp_path: Path) -> None:
    checkout, content_root, source = _legacy_checkout(tmp_path)
    destination = content_root / "statutes/47/32.yaml"
    destination.parent.mkdir(parents=True)
    destination.write_text("collision\n")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "destination collision")

    with (
        patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source),
        pytest.raises(ValueError, match="absent at clean HEAD and live"),
    ):
        _resolve_legacy_replacement_contract(
            source_raw=Path("us-la/statutes/47:32.yaml"),
            destination_raw=Path("us-la/statutes/47/32.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
        )


@pytest.mark.parametrize(
    "relative",
    [
        "us-la/policies/income_tax/untracked.yaml",
        ".axiom/index/untracked.json",
    ],
)
def test_contract_rejects_untracked_checkout_influence(
    tmp_path: Path,
    relative: str,
) -> None:
    checkout, content_root, source = _legacy_checkout(tmp_path)
    untracked = checkout / relative
    untracked.parent.mkdir(parents=True, exist_ok=True)
    untracked.write_text("untrusted\n")

    with (
        patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source),
        pytest.raises(ValueError, match="exact clean HEAD"),
    ):
        _resolve_legacy_replacement_contract(
            source_raw=Path("us-la/statutes/47:32.yaml"),
            destination_raw=Path("us-la/statutes/47/32.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
        )


def test_locked_recheck_rejects_untracked_file_introduced_after_planning(
    tmp_path: Path,
) -> None:
    checkout, content_root, source = _legacy_checkout(tmp_path)
    with patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source):
        contract = _resolve_legacy_replacement_contract(
            source_raw=Path("us-la/statutes/47:32.yaml"),
            destination_raw=Path("us-la/statutes/47/32.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
        )

    injected = checkout / ".axiom/index/lock-race.json"
    injected.parent.mkdir(parents=True, exist_ok=True)
    injected.write_text("{}\n")

    with pytest.raises(ValueError, match="exact clean HEAD"):
        _require_locked_legacy_replacement_base(checkout, contract)
