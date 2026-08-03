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
    _legacy_replacement_authoritative_map,
    _require_locked_legacy_replacement_base,
    _resolve_legacy_replacement_contract,
)
from axiom_encode.legacy_replacement import (
    LEGACY_MANIFEST_SCHEMA,
    legacy_generated_manifest_issues,
    legacy_manual_manifest_issues,
    legacy_receipt_v1_manifest_issues,
    legacy_source_verification_citation_paths,
    legacy_v1_manifest_issues,
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
    payload.pop("manual_exception")
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
    payload["manual_exception"] = None
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


def _generated_manifest() -> dict[str, object]:
    return {
        "schema_version": LEGACY_MANIFEST_SCHEMA,
        "tool": "axiom-encode encode --apply",
        "backend": "codex",
        "runner": "codex-gpt-5.5",
        "model": "gpt-5.5",
        "citation": "us/statute/42/1437c–1",
        "generated_at": "2026-07-16T19:07:09.279448+00:00",
        "run_id": "183b7163",
        "axiom_encode_version": "0.2.1200",
        "axiom_encode_git": {
            "commit": "c" * 40,
            "dirty_tracked": False,
            "root": "/private/tmp/axiom-encode-codex",
            "version": "0.2.1200",
            "version_commit": "c" * 40,
        },
        "generation_prompt_sha256": "d" * 64,
        "generated_output_root": "/tmp/axiom-generated",
        "generated_output_file": "/tmp/axiom-generated/statutes/42/1437c–1.yaml",
        "generated_output_sha256": "a" * 64,
        "trace_file": "/tmp/axiom-generated/trace.json",
        "trace_sha256": "e" * 64,
        "context_manifest_file": "/tmp/axiom-generated/context.json",
        "context_manifest_sha256": "f" * 64,
        "applied_files": [
            {"path": "statutes/42/1437c–1.yaml", "sha256": "a" * 64},
            {"path": "statutes/42/1437c–1.test.yaml", "sha256": "b" * 64},
        ],
        "signature": {
            "algorithm": "hmac-sha256",
            "key_id": "axiom-encode-apply-v1",
            "value": "1" * 64,
        },
    }


def _generated_manifest_issues(payload: object) -> list[str]:
    return legacy_generated_manifest_issues(
        payload,
        expected_files={
            "us/statutes/42/1437c–1.yaml": "a" * 64,
            "us/statutes/42/1437c–1.test.yaml": "b" * 64,
        },
        expected_primary_path="us/statutes/42/1437c–1.yaml",
        expected_citation="us/statute/42/1437c–1",
        jurisdiction_prefix="us",
    )


def test_generated_manifest_admits_uniform_jurisdiction_relative_scope() -> None:
    assert _generated_manifest_issues(_generated_manifest()) == []


def test_generated_manifest_admits_exact_checkout_relative_scope() -> None:
    payload = _generated_manifest()
    payload["applied_files"] = [
        {"path": f"us/{item['path']}", "sha256": item["sha256"]}
        for item in payload["applied_files"]
    ]
    assert _generated_manifest_issues(payload) == []


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"citation": "us/statute/42/1437c-1"}, "citation is stale"),
        ({"generated_output_sha256": "9" * 64}, "output digest is stale"),
        ({"runner": "openai-gpt-5.5"}, "runner/backend mismatch"),
        ({"runner": "codex-attacker"}, "runner/backend mismatch"),
        (
            {"tool": "axiom-encode encode --apply (recovered from sandbox)"},
            "tool is unsupported",
        ),
        ({"manual_exception": "relabelled"}, "claims a manual exception"),
        (
            {
                "signature": {
                    "algorithm": "hmac-sha256",
                    "key_id": "historical-v1",
                    "value": "1" * 64,
                }
            },
            "unknown signature shape",
        ),
    ],
)
def test_generated_manifest_rejects_stale_or_relabelled_evidence(
    mutation: dict[str, object], match: str
) -> None:
    payload = _generated_manifest()
    payload.update(mutation)
    assert any(match in issue for issue in _generated_manifest_issues(payload))


def test_generated_manifest_rejects_mixed_applied_file_scopes() -> None:
    payload = _generated_manifest()
    payload["applied_files"][0]["path"] = "us/statutes/42/1437c–1.yaml"
    assert any(
        "one admitted path scope" in issue
        for issue in _generated_manifest_issues(payload)
    )


def test_generated_manifest_rejects_unknown_provenance_fields() -> None:
    payload = _generated_manifest()
    payload["unknown_provenance"] = {"trusted": True}
    assert any(
        "fields are noncanonical" in issue
        for issue in _generated_manifest_issues(payload)
    )

    payload = _generated_manifest()
    payload["axiom_encode_git"]["unknown_identity"] = "trusted"
    assert any(
        "encoder identity is malformed" in issue
        for issue in _generated_manifest_issues(payload)
    )


@pytest.mark.parametrize("backend", [[], {}])
def test_generated_manifest_rejects_nonscalar_backend_without_crashing(
    backend: object,
) -> None:
    payload = _generated_manifest()
    payload["backend"] = backend
    issues = legacy_v1_manifest_issues(
        payload,
        expected_files={
            "us/statutes/42/1437c–1.yaml": "a" * 64,
            "us/statutes/42/1437c–1.test.yaml": "b" * 64,
        },
        expected_primary_path="us/statutes/42/1437c–1.yaml",
        expected_citation="us/statute/42/1437c–1",
        jurisdiction_prefix="us",
    )
    assert any("backend is unsupported" in issue for issue in issues)


def test_v1_dispatcher_preserves_manual_owner_admission() -> None:
    assert (
        legacy_v1_manifest_issues(
            _manual_manifest(),
            expected_files={
                "us-la/statutes/47:32.yaml": "a" * 64,
                "us-la/statutes/47:32.test.yaml": "b" * 64,
            },
            expected_primary_path="us-la/statutes/47:32.yaml",
            expected_citation="us-la/statute/47:32",
            jurisdiction_prefix="us-la",
        )
        == []
    )


def test_old_manual_receipt_class_cannot_relabel_generated_evidence() -> None:
    issues = legacy_receipt_v1_manifest_issues(
        _generated_manifest(),
        owner_class="v1-manual-hmac-untrusted",
        expected_files={
            "us/statutes/42/1437c–1.yaml": "a" * 64,
            "us/statutes/42/1437c–1.test.yaml": "b" * 64,
        },
        expected_primary_path="us/statutes/42/1437c–1.yaml",
        expected_citation="us/statute/42/1437c–1",
        jurisdiction_prefix="us",
    )
    assert any("tool is not sign-applied-files" in issue for issue in issues)


@pytest.mark.parametrize("owner_class", [[], {}])
def test_receipt_dispatcher_rejects_nonscalar_owner_class_without_crashing(
    owner_class: object,
) -> None:
    issues = legacy_receipt_v1_manifest_issues(
        _manual_manifest(),
        owner_class=owner_class,
        expected_files={
            "us-la/statutes/47:32.yaml": "a" * 64,
            "us-la/statutes/47:32.test.yaml": "b" * 64,
        },
        expected_primary_path="us-la/statutes/47:32.yaml",
        expected_citation="us-la/statute/47:32",
        jurisdiction_prefix="us-la",
    )
    assert issues == ["legacy receipt ownership class is unsupported"]


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
        exact_dependents=[],
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
        "exact_dependents",
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


def _generated_unicode_checkout(
    tmp_path: Path,
) -> tuple[Path, Path, SimpleNamespace]:
    checkout = tmp_path / "rulespec-us"
    content_root = checkout / "us"
    primary = content_root / "statutes/42/1437c–1.yaml"
    companion = primary.with_name("1437c–1.test.yaml")
    primary.parent.mkdir(parents=True)
    primary.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_path: us/statute/42/1437c–1\n"
        "rules: []\n"
    )
    companion.write_text("[]\n")
    primary_sha256 = hashlib.sha256(primary.read_bytes()).hexdigest()
    companion_sha256 = hashlib.sha256(companion.read_bytes()).hexdigest()
    manifest = checkout / ".axiom/encoding-manifests/us/statutes/42/1437c–1.json"
    manifest.parent.mkdir(parents=True)
    payload = _generated_manifest()
    payload["generated_output_sha256"] = primary_sha256
    payload["applied_files"] = [
        {"path": "statutes/42/1437c–1.yaml", "sha256": primary_sha256},
        {
            "path": "statutes/42/1437c–1.test.yaml",
            "sha256": companion_sha256,
        },
    ]
    manifest.write_text(json.dumps(payload) + "\n")
    _git(checkout, "init", "-q")
    _git(checkout, "config", "user.email", "test@example.com")
    _git(checkout, "config", "user.name", "Test")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "generated Unicode base")
    source = SimpleNamespace(
        requested="us/statute/42/1437c–1",
        citation_path="us/statute/42/1437c–1",
        body="official body",
        resolved_source=object(),
    )
    return checkout, content_root, source


def test_contract_admits_generated_v1_unicode_path_with_relative_file_scope(
    tmp_path: Path,
) -> None:
    checkout, content_root, source = _generated_unicode_checkout(tmp_path)
    with patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source):
        contract = _resolve_legacy_replacement_contract(
            source_raw=Path("us/statutes/42/1437c–1.yaml"),
            destination_raw=Path("us/statutes/42/1437c-1.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
        )

    assert contract.source == Path("us/statutes/42/1437c–1.yaml")
    assert contract.destination == Path("us/statutes/42/1437c-1.yaml")
    assert {item.path for item in contract.deleted_files} == {
        Path("us/statutes/42/1437c–1.yaml"),
        Path("us/statutes/42/1437c–1.test.yaml"),
    }


def _generated_unicode_receipt_blocks(
    checkout: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    source = Path("us/statutes/42/1437c–1.yaml")
    companion = source.with_name("1437c–1.test.yaml")
    manifest = Path(".axiom/encoding-manifests/us/statutes/42/1437c–1.json")
    return (
        {
            "owner_class": "v1-hmac-untrusted",
            "trusted_generated_provenance": False,
            "manifest": {
                "path": manifest.as_posix(),
                "sha256": hashlib.sha256(
                    (checkout / manifest).read_bytes()
                ).hexdigest(),
            },
            "files": [
                {
                    "path": path.as_posix(),
                    "sha256": hashlib.sha256(
                        (checkout / path).read_bytes()
                    ).hexdigest(),
                }
                for path in (source, companion)
            ],
        },
        {
            "source": source.as_posix(),
            "destination": "us/statutes/42/1437c-1.yaml",
            "model_manifest_path": (
                ".axiom/encoding-manifests/us/statutes/42/1437c-1.json"
            ),
        },
    )


def test_receipt_authority_reconstructs_generated_v1_unicode_path(
    tmp_path: Path,
) -> None:
    checkout, _content_root, _source = _generated_unicode_checkout(tmp_path)
    legacy, replacement = _generated_unicode_receipt_blocks(checkout)
    base_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    replacements, issues = _legacy_replacement_authoritative_map(
        checkout,
        base_commit=base_commit,
        manifest_label=(".axiom/encoding-manifests/us/statutes/42/1437c-1.json"),
        legacy=legacy,
        replacement=replacement,
    )

    assert issues == []
    assert replacements is not None


def test_receipt_authority_rejects_relabelled_generated_v1_bytes(
    tmp_path: Path,
) -> None:
    checkout, _content_root, _source = _generated_unicode_checkout(tmp_path)
    manifest = checkout / ".axiom/encoding-manifests/us/statutes/42/1437c–1.json"
    payload = json.loads(manifest.read_text())
    payload["runner"] = "codex-attacker"
    manifest.write_text(json.dumps(payload) + "\n")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "relabel generated owner")
    legacy, replacement = _generated_unicode_receipt_blocks(checkout)
    base_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    replacements, issues = _legacy_replacement_authoritative_map(
        checkout,
        base_commit=base_commit,
        manifest_label=(".axiom/encoding-manifests/us/statutes/42/1437c-1.json"),
        legacy=legacy,
        replacement=replacement,
    )

    assert replacements is None
    assert any("runner/backend mismatch" in issue for issue in issues)


def test_receipt_authority_rejects_generated_bytes_under_old_manual_class(
    tmp_path: Path,
) -> None:
    checkout, _content_root, _source = _generated_unicode_checkout(tmp_path)
    legacy, replacement = _generated_unicode_receipt_blocks(checkout)
    legacy["owner_class"] = "v1-manual-hmac-untrusted"
    base_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    replacements, issues = _legacy_replacement_authoritative_map(
        checkout,
        base_commit=base_commit,
        manifest_label=(".axiom/encoding-manifests/us/statutes/42/1437c-1.json"),
        legacy=legacy,
        replacement=replacement,
    )

    assert replacements is None
    assert any("tool is not sign-applied-files" in issue for issue in issues)


@pytest.mark.parametrize("owner_class", [[], {}])
def test_receipt_authority_rejects_nonscalar_owner_class_without_crashing(
    tmp_path: Path,
    owner_class: object,
) -> None:
    checkout, _content_root, _source = _generated_unicode_checkout(tmp_path)
    legacy, replacement = _generated_unicode_receipt_blocks(checkout)
    legacy["owner_class"] = owner_class
    base_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    replacements, issues = _legacy_replacement_authoritative_map(
        checkout,
        base_commit=base_commit,
        manifest_label=(".axiom/encoding-manifests/us/statutes/42/1437c-1.json"),
        legacy=legacy,
        replacement=replacement,
    )

    assert replacements is None
    assert any("ownership classification is invalid" in issue for issue in issues)


def test_contract_rejects_generated_v1_with_wrong_citation(tmp_path: Path) -> None:
    checkout, content_root, source = _generated_unicode_checkout(tmp_path)
    manifest = checkout / ".axiom/encoding-manifests/us/statutes/42/1437c–1.json"
    payload = json.loads(manifest.read_text())
    payload["citation"] = "us/statute/42/1437c-1"
    manifest.write_text(json.dumps(payload) + "\n")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "stale generated citation")

    with (
        patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source),
        pytest.raises(ValueError, match="citation is stale"),
    ):
        _resolve_legacy_replacement_contract(
            source_raw=Path("us/statutes/42/1437c–1.yaml"),
            destination_raw=Path("us/statutes/42/1437c-1.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
        )


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
        checkout / ".axiom/encoding-manifests/us-me/policies/income_tax/"
        "pilot_liability_pipeline.json"
    )
    manifest.parent.mkdir(parents=True)
    payload = _manual_manifest()
    payload.pop("manual_exception")
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


def _add_exact_dependent(checkout: Path, content_root: Path) -> Path:
    dependent = content_root / "policies/income_tax/2026_resident_core.yaml"
    companion = dependent.with_name("2026_resident_core.test.yaml")
    dependent.parent.mkdir(parents=True)
    dependent.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_paths:\n"
        "      - us-la/statute/47:32\n"
        "      - us-la/statute/47:294\n"
        "imports:\n"
        "  - us-la:statutes/47:32#amount\n"
        "rules: []\n"
    )
    companion.write_text("[]\n")
    expected_files = {
        path.relative_to(checkout).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in (dependent, companion)
    }
    manifest = (
        checkout / ".axiom/encoding-manifests/us-la/policies/income_tax/"
        "2026_resident_core.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = _manual_manifest()
    payload["applied_files"] = [
        {"path": path, "sha256": digest} for path, digest in expected_files.items()
    ]
    manifest.write_text(json.dumps(payload) + "\n")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "exact dependent")
    return dependent.relative_to(checkout)


def test_contract_binds_exact_composite_dependent_to_clean_base(
    tmp_path: Path,
) -> None:
    checkout, content_root, source = _legacy_checkout(tmp_path)
    dependent = _add_exact_dependent(checkout, content_root)

    with patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source):
        contract = _resolve_legacy_replacement_contract(
            source_raw=Path("us-la/statutes/47:32.yaml"),
            destination_raw=Path("us-la/statutes/47/32.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
            exact_dependent_paths=(dependent,),
        )

    assert contract.scheduled_dependents == ()
    assert len(contract.exact_dependents) == 1
    exact = contract.exact_dependents[0]
    assert exact.primary == dependent
    assert {item.path for item in exact.legacy_files} == {
        dependent,
        dependent.with_name("2026_resident_core.test.yaml"),
    }
    assert {item.path for item in exact.live_files} == {
        dependent,
        dependent.with_name("2026_resident_core.test.yaml"),
    }
    assert [item.path for item in exact.rewrites] == [dependent]
    assert b"us-la:statutes/47/32#amount" in next(
        item.raw for item in exact.live_files if item.path == dependent
    )
    assert next(
        item.sha256
        for item in exact.legacy_files
        if item.path.name.endswith(".test.yaml")
    ) == next(
        item.sha256
        for item in exact.live_files
        if item.path.name.endswith(".test.yaml")
    )


def test_contract_rejects_exact_dependent_without_v1_ownership(
    tmp_path: Path,
) -> None:
    checkout, content_root, source = _legacy_checkout(tmp_path)
    dependent = _add_exact_dependent(checkout, content_root)
    manifest = (
        checkout / ".axiom/encoding-manifests/us-la/policies/income_tax/"
        "2026_resident_core.json"
    )
    payload = json.loads(manifest.read_text())
    payload["backend"] = "codex"
    manifest.write_text(json.dumps(payload) + "\n")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "forged ownership")

    with (
        patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source),
        pytest.raises(ValueError, match="not admissible"),
    ):
        _resolve_legacy_replacement_contract(
            source_raw=Path("us-la/statutes/47:32.yaml"),
            destination_raw=Path("us-la/statutes/47/32.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
            exact_dependent_paths=(dependent,),
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
