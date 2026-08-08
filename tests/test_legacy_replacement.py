from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode.cli import (
    _legacy_destination_manifest_claimants_at_base,
    _legacy_metadata_reconciliation_bytes,
    _legacy_replacement_authoritative_map,
    _require_locked_legacy_replacement_base,
    _resolve_legacy_replacement_contract,
)
from axiom_encode.legacy_replacement import (
    LEGACY_MANIFEST_SCHEMA,
    LegacyReplacementContract,
    LegacyReplacementFile,
    LegacyReplacementRetainedSuccessor,
    LegacyReplacementRewrite,
    legacy_generated_manifest_issues,
    legacy_manual_manifest_issues,
    legacy_receipt_v1_manifest_issues,
    legacy_source_verification_citation_paths,
    legacy_v1_manifest_issues,
    receipt_identity_payload,
    receipt_identity_sha256,
)
from axiom_encode.legacy_replacement_overlay import (
    LegacyReplacementOverlayError,
    _prune_empty_overlay_parent_directories,
    scope_canonical_replacement_overlay,
    stage_legacy_replacement_overlay,
)
from axiom_encode.rulespec_path_migration import PlannedMove


def test_canonical_replacement_overlay_omits_only_active_colon_paths(tmp_path) -> None:
    checkout = tmp_path / "rulespec-us"
    legacy_file = checkout / "us-la/statutes/47:32.yaml"
    legacy_nested = checkout / "us-la/statutes/47:297/4.yaml"
    legacy_legislation = checkout / "us-la/legislation/act:1.yaml"
    canonical_file = checkout / "us-la/statutes/47/294.yaml"
    canonical_debt = checkout / "us-la/statutes/47/BROKEN.yaml"
    non_atomic_form = checkout / "us-la/forms/r-540:2026.yaml"
    sibling = checkout / "us-nj/statutes/54a:4-7.yaml"
    for path in (
        legacy_file,
        legacy_nested,
        legacy_legislation,
        canonical_file,
        canonical_debt,
        non_atomic_form,
        sibling,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("format: rulespec/v1\nrules: []\n")

    omitted = scope_canonical_replacement_overlay(
        checkout,
        active_jurisdiction="us-la",
    )

    assert omitted == (
        Path("us-la/legislation/act:1.yaml"),
        Path("us-la/statutes/47:297"),
        Path("us-la/statutes/47:32.yaml"),
    )
    assert not legacy_file.exists()
    assert not legacy_nested.exists()
    assert not legacy_legislation.exists()
    assert canonical_file.is_file()
    assert canonical_debt.is_file()
    assert non_atomic_form.is_file()
    assert not (checkout / "us-nj").exists()


def test_canonical_replacement_overlay_rejects_symlink(tmp_path) -> None:
    checkout = tmp_path / "rulespec-us"
    statutes = checkout / "us-la/statutes"
    statutes.mkdir(parents=True)
    outside = tmp_path / "outside.yaml"
    outside.write_text("format: rulespec/v1\nrules: []\n")
    (statutes / "47:32.yaml").symlink_to(outside)

    with pytest.raises(
        LegacyReplacementOverlayError,
        match="contains a symlink",
    ):
        scope_canonical_replacement_overlay(
            checkout,
            active_jurisdiction="us-la",
        )


def test_canonical_replacement_overlay_rejects_special_file(tmp_path) -> None:
    checkout = tmp_path / "rulespec-us"
    statutes = checkout / "us-la/statutes"
    statutes.mkdir(parents=True)
    fifo = statutes / "47:32.yaml"
    fifo.parent.mkdir(parents=True, exist_ok=True)
    fifo.touch()
    fifo.unlink()
    os.mkfifo(fifo)

    with pytest.raises(
        LegacyReplacementOverlayError,
        match="contains a special file",
    ):
        scope_canonical_replacement_overlay(
            checkout,
            active_jurisdiction="us-la",
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
        destination_predecessor_class="canonicalized-unowned-duplicate",
        destination_predecessor_files=[
            {"path": "us-la/statutes/47/32.yaml", "sha256": "f" * 64}
        ],
        retained_successors=[{"source": "us-la/statutes/47:294.yaml"}],
        metadata_reconciliations=[{"path": "known-validation-gaps.yaml"}],
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
        "destination_predecessor_class",
        "destination_predecessor_files",
        "retained_successors",
        "metadata_reconciliations",
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


def test_contract_absorbs_exact_unowned_canonical_destination_predecessor(
    tmp_path: Path,
) -> None:
    checkout, content_root, source = _generated_unicode_checkout(tmp_path)
    legacy = checkout / "us/statutes/42/1437c–1.yaml"
    legacy_test = legacy.with_name("1437c–1.test.yaml")
    destination = checkout / "us/statutes/42/1437c-1.yaml"
    destination_test = destination.with_name("1437c-1.test.yaml")
    destination.write_bytes(legacy.read_bytes())
    destination_test.write_bytes(legacy_test.read_bytes())
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "unowned canonical predecessor")

    with patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source):
        contract = _resolve_legacy_replacement_contract(
            source_raw=Path("us/statutes/42/1437c–1.yaml"),
            destination_raw=Path("us/statutes/42/1437c-1.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
        )

    assert {item.path for item in contract.destination_predecessor_files} == {
        Path("us/statutes/42/1437c-1.yaml"),
        Path("us/statutes/42/1437c-1.test.yaml"),
    }
    assert contract.destination_predecessor_class == "canonicalized-unowned-duplicate"
    with pytest.raises(
        LegacyReplacementOverlayError,
        match="classification differs",
    ):
        stage_legacy_replacement_overlay(
            contract._replace(destination_predecessor_class="absent"),
            checkout,
        )
    stage_legacy_replacement_overlay(contract, checkout)
    assert not legacy.exists()
    assert not legacy_test.exists()
    assert not destination.exists()
    assert not destination_test.exists()
    assert not (
        checkout / ".axiom/encoding-manifests/us/statutes/42/1437c–1.json"
    ).exists()


def test_primary_move_with_destination_predecessor_reconciles_metadata(
    tmp_path: Path,
) -> None:
    checkout, content_root, source = _legacy_checkout(tmp_path)
    legacy = checkout / "us-la/statutes/47:32.yaml"
    legacy_test = legacy.with_name("47:32.test.yaml")
    destination = checkout / "us-la/statutes/47/32.yaml"
    destination_test = destination.with_name("32.test.yaml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(legacy.read_bytes())
    destination_test.write_bytes(legacy_test.read_bytes())

    old_module = "us-la/statutes/47:32.yaml"
    new_module = "us-la/statutes/47/32.yaml"
    old_manifest = ".axiom/encoding-manifests/us-la/statutes/47:32.json"
    index = checkout / ".axiom/index/provisions_to_rules.json"
    index.write_text(
        json.dumps(
            {
                "records": [
                    {"module": old_module},
                    {"module": new_module},
                ]
            }
        )
        + "\n"
    )
    pending = checkout / ".axiom/pending-validation-fingerprints.json"
    pending.write_text(
        json.dumps(
            {
                old_module: {"fingerprint": "legacy"},
                new_module: {"fingerprint": "canonical"},
            }
        )
        + "\n"
    )
    known_gaps = checkout / "known-validation-gaps.yaml"
    known_gaps.write_text(
        "validate_failures:\n"
        f"  '{old_module}':\n"
        "    pending:\n"
        "      fingerprint: sha256:legacy\n"
        "  'us-hi/statutes/example.yaml':\n"
        "    pending:\n"
        "      fingerprint: sha256:keep\n"
    )
    base_waiver_sha256 = hashlib.sha256(known_gaps.read_bytes()).hexdigest()
    toolchain = checkout / ".axiom/toolchain.toml"
    toolchain.write_text(
        'rulespec_root = "rulespec-us"\n'
        f'validation_waiver_set_sha256 = "{base_waiver_sha256}"\n'
    )
    oracle_pending = checkout / "oracle-coverage-pending.yaml"
    oracle_pending.write_text(
        "version: 1\n"
        "ceiling: 1\n"
        "entries:\n"
        "- legal_id: us-hi:statutes/example#amount\n"
        "  source: manual\n"
        "  since: '2026-01-01'\n"
    )
    oracle_pending_before = oracle_pending.read_bytes()
    manifest_inventory = checkout / "tests/test_encoding_manifests.py"
    manifest_inventory.parent.mkdir(parents=True)
    manifest_inventory.write_text(
        f"ALLOW = {{\n    '{old_manifest}',\n    'keep.json',\n}}\n"
    )
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "canonical predecessor metadata")

    with patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source):
        contract = _resolve_legacy_replacement_contract(
            source_raw=Path(old_module),
            destination_raw=Path(new_module),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
        )

    reconciled = {item.path: item for item in contract.metadata_reconciliations}
    assert set(reconciled) == {
        Path(".axiom/index/provisions_to_rules.json"),
        Path(".axiom/pending-validation-fingerprints.json"),
        Path(".axiom/toolchain.toml"),
        Path("known-validation-gaps.yaml"),
        Path("tests/test_encoding_manifests.py"),
    }
    assert Path("oracle-coverage-pending.yaml") not in reconciled
    assert oracle_pending.read_bytes() == oracle_pending_before
    assert all(item.path not in reconciled for item in contract.rewrites)

    reconciled_waivers = reconciled[Path("known-validation-gaps.yaml")].raw
    assert old_module.encode() not in reconciled_waivers
    assert new_module.encode() not in reconciled_waivers
    reconciled_waiver_sha256 = hashlib.sha256(reconciled_waivers).hexdigest()
    reconciled_toolchain = reconciled[Path(".axiom/toolchain.toml")].raw.decode()
    assert (
        f'validation_waiver_set_sha256 = "{reconciled_waiver_sha256}"'
        in reconciled_toolchain
    )


def test_contract_admits_cryptographically_verified_retained_successor(
    tmp_path: Path,
) -> None:
    checkout, content_root, source_unit = _legacy_checkout(tmp_path)
    old = checkout / "us-la/statutes/47:294.yaml"
    old_test = old.with_name("47:294.test.yaml")
    successor = checkout / "us-la/statutes/47/294.yaml"
    successor_test = successor.with_name("294.test.yaml")
    old.parent.mkdir(parents=True, exist_ok=True)
    successor.parent.mkdir(parents=True, exist_ok=True)
    old.write_text(
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        "    corpus_citation_path: us-la/statute/47:294\nrules: []\n"
    )
    old_test.write_text("[]\n")
    successor.write_bytes(old.read_bytes())
    successor_test.write_bytes(old_test.read_bytes())
    old_manifest = checkout / ".axiom/encoding-manifests/us-la/statutes/47:294.json"
    old_manifest.parent.mkdir(parents=True, exist_ok=True)
    old_payload = _manual_manifest()
    old_payload["applied_files"] = [
        {
            "path": path.relative_to(checkout).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (old, old_test)
    ]
    old_manifest.write_text(json.dumps(old_payload) + "\n")
    successor_manifest = (
        checkout / ".axiom/encoding-manifests/us-la/statutes/47/294.json"
    )
    successor_manifest.parent.mkdir(parents=True, exist_ok=True)
    successor_payload = {
        "schema_version": "axiom-encode/applied-rulespec/v5",
        "tool": "axiom-encode encode --apply",
        "backend": "openai",
        "citation": "us-la/statute/47:294",
        "validation_execution": {"axiom_encode": {"commit": "a" * 40}},
        "applied_files": [
            {
                "path": path.relative_to(checkout).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (successor, successor_test)
        ],
    }
    successor_manifest.write_text(
        json.dumps(successor_payload, indent=2, sort_keys=True) + "\n"
    )
    index = checkout / ".axiom/index/provisions_to_rules.json"
    index.write_text(
        json.dumps(
            {
                "records": [
                    {"module": "us-la/statutes/47:32.yaml"},
                    {"module": "us-la/statutes/47/32.yaml"},
                    {"module": "us-la/statutes/47:294.yaml"},
                    {"module": "us-la/statutes/47/294.yaml"},
                ]
            }
        )
        + "\n"
    )
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "retained successor base")

    resolved_294 = SimpleNamespace(
        requested="us-la/statute/47:294",
        citation_path="us-la/statute/47:294",
        body="official body",
        resolved_source=object(),
    )
    with (
        patch(
            "axiom_encode.cli.resolve_corpus_source_unit",
            side_effect=lambda citation, _release: (
                resolved_294 if citation.endswith("47:294") else source_unit
            ),
        ),
        patch(
            "axiom_encode.cli._applied_encoding_manifest_verifier",
            return_value=object(),
        ),
        patch(
            "axiom_encode.cli.verify_rulespec_validation_waiver_set",
            return_value="f" * 64,
        ),
        patch(
            "axiom_encode.cli._load_verified_applied_encoding_manifest_payload",
            return_value=(
                successor_payload,
                "",
                hashlib.sha256(successor_manifest.read_bytes()).hexdigest(),
                [],
            ),
        ) as verified,
    ):
        contract = _resolve_legacy_replacement_contract(
            source_raw=Path("us-la/statutes/47:32.yaml"),
            destination_raw=Path("us-la/statutes/47/32.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source_unit,
            corpus_release=SimpleNamespace(),
            retained_successor_paths=(Path("us-la/statutes/47:294.yaml"),),
        )

    assert verified.call_count == 1
    assert len(contract.retained_successors) == 1
    assert contract.retained_successors[0].destination == Path(
        "us-la/statutes/47/294.yaml"
    )
    assert [item.path for item in contract.metadata_reconciliations] == [
        Path(".axiom/index/provisions_to_rules.json")
    ]


def test_destination_manifest_claimant_scan_is_one_conservative_base_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_commit = "a" * 40
    claimant = Path(".axiom/encoding-manifests/us/statutes/claimant.json")
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{base_commit}:{claimant.as_posix()}\0".encode(),
            stderr=b"",
        )

    monkeypatch.setattr("axiom_encode.cli.subprocess.run", run)

    assert _legacy_destination_manifest_claimants_at_base(
        tmp_path,
        base_commit=base_commit,
        destination_paths={
            Path("us/statutes/42/1437c-1.yaml"),
            Path("us/statutes/42/1437c-1.test.yaml"),
        },
    ) == [claimant]
    assert len(commands) == 1
    assert commands[0][3:9] == ["grep", "-z", "-l", "-a", "-F", "-e"]


@pytest.mark.parametrize(
    "claimant_schema",
    ["axiom-encode/applied-rulespec/v1", "axiom-encode/applied-rulespec/v5"],
)
def test_canonical_destination_predecessor_rejects_owner_and_overlay_race(
    tmp_path: Path,
    claimant_schema: str,
) -> None:
    checkout, content_root, source = _generated_unicode_checkout(tmp_path)
    legacy = checkout / "us/statutes/42/1437c–1.yaml"
    legacy_test = legacy.with_name("1437c–1.test.yaml")
    destination = checkout / "us/statutes/42/1437c-1.yaml"
    destination_test = destination.with_name("1437c-1.test.yaml")
    destination.write_bytes(legacy.read_bytes())
    destination_test.write_bytes(legacy_test.read_bytes())
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "unowned canonical predecessor")

    with patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source):
        contract = _resolve_legacy_replacement_contract(
            source_raw=Path("us/statutes/42/1437c–1.yaml"),
            destination_raw=Path("us/statutes/42/1437c-1.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
        )
    destination.write_text("changed after planning\n")
    with pytest.raises(LegacyReplacementOverlayError, match="predecessor changed"):
        stage_legacy_replacement_overlay(contract, checkout)

    destination.write_bytes(
        next(
            item.raw
            for item in contract.destination_predecessor_files
            if item.path.name == "1437c-1.yaml"
        )
    )
    claimant = checkout / ".axiom/encoding-manifests/us/policies/claimant.json"
    claimant.parent.mkdir(parents=True, exist_ok=True)
    claimant.write_text(
        json.dumps(
            {
                "schema_version": claimant_schema,
                "applied_files": [
                    {
                        "path": "statutes/42/1437c-1.yaml",
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
        + "\n"
    )
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "claim canonical predecessor")
    with (
        patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source),
        pytest.raises(ValueError, match="already manifest-owned"),
    ):
        _resolve_legacy_replacement_contract(
            source_raw=Path("us/statutes/42/1437c–1.yaml"),
            destination_raw=Path("us/statutes/42/1437c-1.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
        )


def test_canonical_destination_predecessor_rejects_malformed_manifest_candidate(
    tmp_path: Path,
) -> None:
    checkout, content_root, source = _generated_unicode_checkout(tmp_path)
    legacy = checkout / "us/statutes/42/1437c–1.yaml"
    destination = checkout / "us/statutes/42/1437c-1.yaml"
    destination.write_bytes(legacy.read_bytes())
    destination.with_name("1437c-1.test.yaml").write_bytes(
        legacy.with_name("1437c–1.test.yaml").read_bytes()
    )
    claimant = checkout / ".axiom/encoding-manifests/us/statutes/malformed.json"
    claimant.parent.mkdir(parents=True, exist_ok=True)
    claimant.write_text('{"applied_files":["us/statutes/42/1437c-1.yaml"\n')
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "malformed destination claimant")

    with (
        patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source),
        pytest.raises(ValueError, match="already manifest-owned"),
    ):
        _resolve_legacy_replacement_contract(
            source_raw=Path("us/statutes/42/1437c–1.yaml"),
            destination_raw=Path("us/statutes/42/1437c-1.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
        )


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
        "    upstream_source_check:\n"
        "      status: checked_higher_authority\n"
        "      checked_paths:\n"
        "        - us-la/statute/47:32\n"
        "        - us-la/statute/47:294\n"
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


def test_exact_dependent_source_verification_migration_preserves_full_history() -> None:
    from axiom_encode.cli import (
        _migrate_legacy_exact_dependent_source_verification,
    )

    raw = (
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_paths:\n"
        "      - us-la/statute/47:32\n"
        "      - us-la/statute/47:294\n"
        "    upstream_source_check:\n"
        "      status: checked_higher_authority\n"
        "      checked_paths:\n"
        "        - us-la/statute/47:32\n"
        "        - us-la/statute/47:294\n"
        "rules:\n"
        "  - metadata:\n"
        "      proof:\n"
        "        atoms:\n"
        "          - source:\n"
        "              corpus_citation_path: us-la/statute/47:294\n"
    ).encode()

    migrated, migration = _migrate_legacy_exact_dependent_source_verification(raw)

    assert migration is not None
    assert migration.legacy_corpus_citation_paths == (
        "us-la/statute/47:32",
        "us-la/statute/47:294",
    )
    assert migration.corpus_citation_path == "us-la/statute/47:32"
    assert migrated == raw.replace(
        (
            "    corpus_citation_paths:\n"
            "      - us-la/statute/47:32\n"
            "      - us-la/statute/47:294\n"
        ).encode(),
        b"    corpus_citation_path: us-la/statute/47:32\n",
        1,
    )
    assert b"      checked_paths:\n" in migrated
    assert b"              corpus_citation_path: us-la/statute/47:294\n" in migrated


@pytest.mark.parametrize(
    "checked_paths",
    [
        "",
        "      checked_paths:\n        - us-la/statute/47:32\n",
        (
            "      checked_paths:\n"
            "        - us-la/statute/47:294\n"
            "        - us-la/statute/47:32\n"
        ),
    ],
)
def test_exact_dependent_source_verification_migration_rejects_unbound_history(
    checked_paths: str,
) -> None:
    from axiom_encode.cli import (
        _migrate_legacy_exact_dependent_source_verification,
    )

    raw = (
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_paths:\n"
        "      - us-la/statute/47:32\n"
        "      - us-la/statute/47:294\n"
        "    upstream_source_check:\n"
        "      status: checked_higher_authority\n"
        f"{checked_paths}"
        "rules: []\n"
    ).encode()

    with pytest.raises(ValueError, match="not preserved exactly"):
        _migrate_legacy_exact_dependent_source_verification(raw)


def test_exact_dependent_source_verification_migration_leaves_singular_unchanged() -> (
    None
):
    from axiom_encode.cli import (
        _migrate_legacy_exact_dependent_source_verification,
    )

    raw = (
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_path: us-la/statute/47:32\n"
        "rules: []\n"
    ).encode()

    assert _migrate_legacy_exact_dependent_source_verification(raw) == (raw, None)


@pytest.mark.parametrize(
    ("extra_verification", "match"),
    [
        (
            "    corpus_citation_path: us-la/statute/47:32\n",
            "mixes singular and plural",
        ),
        ("    source_sha256: " + "a" * 64 + "\n", "ambiguous aggregate"),
    ],
)
def test_exact_dependent_source_verification_migration_rejects_ambiguous_primary(
    extra_verification: str,
    match: str,
) -> None:
    from axiom_encode.cli import (
        _migrate_legacy_exact_dependent_source_verification,
    )

    raw = (
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_paths:\n"
        "      - us-la/statute/47:32\n"
        f"{extra_verification}"
        "    upstream_source_check:\n"
        "      checked_paths:\n"
        "        - us-la/statute/47:32\n"
        "rules: []\n"
    ).encode()

    with pytest.raises(ValueError, match=match):
        _migrate_legacy_exact_dependent_source_verification(raw)


def test_exact_dependent_source_verification_migration_rejects_duplicate_history() -> (
    None
):
    from axiom_encode.cli import (
        _migrate_legacy_exact_dependent_source_verification,
    )

    raw = (
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_paths:\n"
        "      - us-la/statute/47:32\n"
        "      - us-la/statute/47:32\n"
        "    upstream_source_check:\n"
        "      checked_paths:\n"
        "        - us-la/statute/47:32\n"
        "        - us-la/statute/47:32\n"
        "rules: []\n"
    ).encode()

    with pytest.raises(ValueError, match="not a unique canonical citation list"):
        _migrate_legacy_exact_dependent_source_verification(raw)


def test_exact_dependent_source_verification_migration_keeps_singular_aliases() -> None:
    from axiom_encode.cli import (
        _migrate_legacy_exact_dependent_source_verification,
    )

    raw = (
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_path: &citation us-la/statute/47:32\n"
        "rules:\n"
        "  - metadata:\n"
        "      proof:\n"
        "        atoms:\n"
        "          - source: *citation\n"
    ).encode()

    migrated, migration = _migrate_legacy_exact_dependent_source_verification(raw)

    assert migrated == raw
    assert migration is None


@pytest.mark.parametrize(
    "raw",
    [
        (
            "format: rulespec/v1\n"
            "module:\n"
            "  source_verification:\n"
            "    corpus_citation_path: us-la/statute/47:32\n"
            "    corpus_citation_path: us-la/statute/47:294\n"
            "rules: []\n"
        ).encode(),
        (
            "format: rulespec/v1\n"
            "module:\n"
            "  source_verification:\n"
            "    corpus_citation_paths:\n"
            "      - us-la/statute/47:32\n"
            "  source_verification:\n"
            "    corpus_citation_path: us-la/statute/47:294\n"
            "rules: []\n"
        ).encode(),
    ],
)
def test_exact_dependent_source_verification_migration_rejects_noop_duplicate_keys(
    raw: bytes,
) -> None:
    from axiom_encode.cli import (
        _migrate_legacy_exact_dependent_source_verification,
    )

    with pytest.raises(ValueError, match="duplicate key"):
        _migrate_legacy_exact_dependent_source_verification(raw)


def test_exact_dependent_source_verification_migration_rejects_yaml_aliases() -> None:
    from axiom_encode.cli import (
        _migrate_legacy_exact_dependent_source_verification,
    )

    raw = (
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification: &verification\n"
        "    corpus_citation_paths:\n"
        "      - us-la/statute/47:32\n"
        "    upstream_source_check:\n"
        "      checked_paths:\n"
        "        - us-la/statute/47:32\n"
        "unrelated_audit_copy: *verification\n"
        "rules: []\n"
    ).encode()

    with pytest.raises(ValueError, match="cannot contain YAML anchors or aliases"):
        _migrate_legacy_exact_dependent_source_verification(raw)


def test_exact_dependent_source_verification_migration_rejects_duplicate_keys() -> None:
    from axiom_encode.cli import (
        _migrate_legacy_exact_dependent_source_verification,
    )

    raw = (
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_paths:\n"
        "      - us-la/statute/47:32\n"
        "    upstream_source_check:\n"
        "      checked_paths:\n"
        "        - us-la/statute/47:32\n"
        "    upstream_source_check:\n"
        "      checked_paths:\n"
        "        - us-la/statute/47:32\n"
        "rules: []\n"
    ).encode()

    with pytest.raises(ValueError, match="duplicate key"):
        _migrate_legacy_exact_dependent_source_verification(raw)


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
    live_primary = next(item.raw for item in exact.live_files if item.path == dependent)
    assert b"    corpus_citation_path: us-la/statute/47:32\n" in live_primary
    assert b"    corpus_citation_paths:\n" not in live_primary
    assert b"      checked_paths:\n" in live_primary
    assert exact.source_verification_migration is not None
    assert exact.source_verification_migration.corpus_citation_path == (
        "us-la/statute/47:32"
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


def test_contract_rejects_exact_dependent_primary_outside_signed_source(
    tmp_path: Path,
) -> None:
    checkout, content_root, source = _legacy_checkout(tmp_path)
    dependent = _add_exact_dependent(checkout, content_root)
    dependent_path = checkout / dependent
    dependent_path.write_text(
        dependent_path.read_text()
        .replace(
            ("      - us-la/statute/47:32\n      - us-la/statute/47:294\n"),
            ("      - us-la/statute/47:294\n      - us-la/statute/47:32\n"),
        )
        .replace(
            ("        - us-la/statute/47:32\n        - us-la/statute/47:294\n"),
            ("        - us-la/statute/47:294\n        - us-la/statute/47:32\n"),
        )
    )
    manifest = (
        checkout
        / ".axiom/encoding-manifests/us-la/policies/income_tax/2026_resident_core.json"
    )
    payload = json.loads(manifest.read_text())
    for applied in payload["applied_files"]:
        applied_path = checkout / applied["path"]
        applied["sha256"] = hashlib.sha256(applied_path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload) + "\n")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "reordered exact dependent source history")

    with (
        patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source),
        pytest.raises(ValueError, match="must match the signed replacement source"),
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


def test_contract_rejects_dirty_or_inexact_existing_destination(tmp_path: Path) -> None:
    checkout, content_root, source = _legacy_checkout(tmp_path)
    destination = content_root / "statutes/47/32.yaml"
    destination.parent.mkdir(parents=True)
    destination.write_text("collision\n")
    destination.with_name("32.test.yaml").write_text("collision\n")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "destination collision")

    with (
        patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source),
        pytest.raises(ValueError, match="not an exact canonicalized source duplicate"),
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


def test_five_inventory_reconciliation_is_exact_and_deterministic() -> None:
    moves = (
        PlannedMove(
            source=Path("us-la/statutes/47:294.yaml"),
            destination=Path("us-la/statutes/47/294.yaml"),
        ),
    )
    old_module = "us-la/statutes/47:294.yaml"
    new_module = "us-la/statutes/47/294.yaml"
    old_identity = "us-la:statutes/47:294"
    new_identity = "us-la:statutes/47/294"
    old_manifest = ".axiom/encoding-manifests/us-la/statutes/47:294.json"
    cases = {
        Path(".axiom/index/provisions_to_rules.json"): (
            json.dumps(
                {
                    "records": [
                        {"module": old_module},
                        {"module": new_module},
                    ]
                }
            ).encode(),
            "remove_legacy_module_records",
        ),
        Path(".axiom/pending-validation-fingerprints.json"): (
            json.dumps({old_module: "old", new_module: "new"}).encode(),
            "remove_legacy_fingerprints",
        ),
        Path("known-validation-gaps.yaml"): (
            f"'{old_module}':\n  reason: old\n'{new_module}':\n  reason: new\n".encode(),
            "remove_legacy_validation_gaps",
        ),
        Path("oracle-coverage-pending.yaml"): (
            (
                f"- legal_id: {old_identity}#output.tax\n  reason: old\n"
                f"- legal_id: {new_identity}#output.tax\n  reason: new\n"
            ).encode(),
            "remove_legacy_oracle_pending",
        ),
        Path("tests/test_encoding_manifests.py"): (
            f"ALLOW = {{\n    '{old_manifest}',\n    'keep.json',\n}}\n".encode(),
            "remove_retired_manifest_allowances",
        ),
    }
    for path, (raw, operation) in cases.items():
        rewritten, operations = _legacy_metadata_reconciliation_bytes(
            path, raw, moves=moves
        )
        assert operations == ({"operation": operation, "count": 1},)
        assert old_module.encode() not in rewritten
        assert old_identity.encode() not in rewritten
        assert old_manifest.encode() not in rewritten


def test_toolchain_reconciliation_binds_exact_post_migration_waiver_digest() -> None:
    moves = (
        PlannedMove(
            source=Path("us-la/statutes/47:294.yaml"),
            destination=Path("us-la/statutes/47/294.yaml"),
        ),
    )
    digest = "e" * 64
    raw = (
        f'rulespec_root = "rulespec-us"\nvalidation_waiver_set_sha256 = "{"f" * 64}"\n'
    ).encode()

    rewritten, operations = _legacy_metadata_reconciliation_bytes(
        Path(".axiom/toolchain.toml"),
        raw,
        moves=moves,
        validation_waiver_set_sha256=digest,
    )

    assert (
        rewritten
        == (
            'rulespec_root = "rulespec-us"\n'
            f'validation_waiver_set_sha256 = "{digest}"\n'
        ).encode()
    )
    assert operations == (
        {"operation": "update_validation_waiver_set_sha256", "count": 1},
    )
    with pytest.raises(ValueError, match="lacks the post-migration waiver digest"):
        _legacy_metadata_reconciliation_bytes(
            Path(".axiom/toolchain.toml"),
            raw,
            moves=moves,
        )
    with pytest.raises(ValueError, match="not one canonical entry"):
        _legacy_metadata_reconciliation_bytes(
            Path(".axiom/toolchain.toml"),
            raw + raw,
            moves=moves,
            validation_waiver_set_sha256=digest,
        )


def test_oracle_reconciliation_supports_canonical_mapping_schema() -> None:
    moves = (
        PlannedMove(
            source=Path("us-la/statutes/47:294.yaml"),
            destination=Path("us-la/statutes/47/294.yaml"),
        ),
    )
    raw = (
        "version: 1\n"
        "ceiling: 2\n"
        "entries:\n"
        "- legal_id: us-la:statutes/47:294#legacy_amount\n"
        "  source: manual\n"
        "  since: '2026-01-01'\n"
        "- legal_id: us-la:statutes/47/294#canonical_amount\n"
        "  source: manual\n"
        "  since: '2026-01-01'\n"
    ).encode()

    rewritten, operations = _legacy_metadata_reconciliation_bytes(
        Path("oracle-coverage-pending.yaml"),
        raw,
        moves=moves,
    )

    assert b"ceiling: 1\n" in rewritten
    assert b"us-la:statutes/47:294#legacy_amount" not in rewritten
    assert b"us-la:statutes/47/294#canonical_amount" in rewritten
    assert operations == ({"operation": "remove_legacy_oracle_pending", "count": 1},)


def test_oracle_reconciliation_no_match_preserves_mapping_bytes() -> None:
    moves = (
        PlannedMove(
            source=Path("us-la/statutes/47:294.yaml"),
            destination=Path("us-la/statutes/47/294.yaml"),
        ),
    )
    raw = (
        "version: 1\n"
        "ceiling: 1\n"
        "entries:\n"
        "- legal_id: us-hi:statutes/example#amount\n"
        "  source: manual\n"
        "  since: '2026-01-01'\n"
    ).encode()

    rewritten, operations = _legacy_metadata_reconciliation_bytes(
        Path("oracle-coverage-pending.yaml"),
        raw,
        moves=moves,
    )

    assert rewritten == raw
    assert operations == ({"operation": "remove_legacy_oracle_pending", "count": 0},)


def test_oracle_reconciliation_requires_successor_only_for_removed_source() -> None:
    moves = (
        PlannedMove(
            source=Path("us-la/statutes/47:294.yaml"),
            destination=Path("us-la/statutes/47/294.yaml"),
        ),
        PlannedMove(
            source=Path("us-la/statutes/47:295.yaml"),
            destination=Path("us-la/statutes/47/295.yaml"),
        ),
    )
    raw = (
        "version: 1\n"
        "entries:\n"
        "- legal_id: us-la:statutes/47:294#legacy_amount\n"
        "  source: manual\n"
        "  since: '2026-01-01'\n"
        "- legal_id: us-la:statutes/47/294#canonical_amount\n"
        "  source: manual\n"
        "  since: '2026-01-01'\n"
    ).encode()

    rewritten, operations = _legacy_metadata_reconciliation_bytes(
        Path("oracle-coverage-pending.yaml"),
        raw,
        moves=moves,
    )

    assert b"us-la:statutes/47:294#legacy_amount" not in rewritten
    assert b"us-la:statutes/47/294#canonical_amount" in rewritten
    assert b"ceiling:" not in rewritten
    assert operations == ({"operation": "remove_legacy_oracle_pending", "count": 1},)


def test_oracle_reconciliation_preserves_explicit_null_ceiling() -> None:
    moves = (
        PlannedMove(
            source=Path("us-la/statutes/47:294.yaml"),
            destination=Path("us-la/statutes/47/294.yaml"),
        ),
    )
    raw = (
        "version: 1\n"
        "ceiling: null\n"
        "entries:\n"
        "- legal_id: us-la:statutes/47:294#legacy_amount\n"
        "  source: manual\n"
        "  since: '2026-01-01'\n"
        "- legal_id: us-la:statutes/47/294#canonical_amount\n"
        "  source: manual\n"
        "  since: '2026-01-01'\n"
    ).encode()

    rewritten, operations = _legacy_metadata_reconciliation_bytes(
        Path("oracle-coverage-pending.yaml"),
        raw,
        moves=moves,
    )

    assert b"ceiling: null\n" in rewritten
    assert b"us-la:statutes/47:294#legacy_amount" not in rewritten
    assert b"us-la:statutes/47/294#canonical_amount" in rewritten
    assert operations == ({"operation": "remove_legacy_oracle_pending", "count": 1},)


@pytest.mark.parametrize(
    "records",
    [
        [
            {"module": "us-la/statutes/47:294.yaml"},
            {"note": "successor us-la/statutes/47/294.yaml is available"},
        ],
        [{"module": "us-la/statutes/47:294.yaml"}],
        [
            {"module": "us-la/statutes/47:294.yaml"},
            {"module": "us-la/statutes/47/295.yaml"},
        ],
    ],
)
def test_provision_index_requires_exact_canonical_module_record(
    records: list[dict[str, str]],
) -> None:
    moves = (
        PlannedMove(
            source=Path("us-la/statutes/47:294.yaml"),
            destination=Path("us-la/statutes/47/294.yaml"),
        ),
    )

    with pytest.raises(
        ValueError,
        match="lacks exact canonical module records: us-la/statutes/47/294.yaml",
    ):
        _legacy_metadata_reconciliation_bytes(
            Path(".axiom/index/provisions_to_rules.json"),
            json.dumps({"records": records}).encode(),
            moves=moves,
        )


def test_provision_index_rejects_out_of_band_structured_successor_field() -> None:
    moves = (
        PlannedMove(
            source=Path("us-la/statutes/47:294.yaml"),
            destination=Path("us-la/statutes/47/294.yaml"),
        ),
    )
    payload = {
        "records": [{"module": "us-la/statutes/47:294.yaml"}],
        "metadata": {"module": "us-la/statutes/47/294.yaml"},
    }

    with pytest.raises(ValueError, match="lacks exact canonical module records"):
        _legacy_metadata_reconciliation_bytes(
            Path(".axiom/index/provisions_to_rules.json"),
            json.dumps(payload).encode(),
            moves=moves,
        )


def test_provision_index_reconciliation_preserves_literal_unicode() -> None:
    moves = (
        PlannedMove(
            source=Path("us-la/statutes/47:294.yaml"),
            destination=Path("us-la/statutes/47/294.yaml"),
        ),
    )
    unrelated_citation = "us/statute/42/1437c–1"
    payload = {
        "records": [
            {"module": "us-la/statutes/47:294.yaml"},
            {"module": "us-la/statutes/47/294.yaml"},
        ],
        unrelated_citation: [{"module": "us/statutes/42/1437c-1.yaml"}],
    }
    raw = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode()

    rewritten, operations = _legacy_metadata_reconciliation_bytes(
        Path(".axiom/index/provisions_to_rules.json"),
        raw,
        moves=moves,
    )

    assert operations == ({"operation": "remove_legacy_module_records", "count": 1},)
    assert unrelated_citation.encode() in rewritten
    assert b"\\u2013" not in rewritten
    assert json.loads(rewritten)[unrelated_citation] == payload[unrelated_citation]


def test_retained_successor_overlay_preflights_then_removes_one_composite(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "rulespec-us"
    main = Path("us-la/statutes/47:32.yaml")
    main_manifest = Path(".axiom/encoding-manifests/us-la/statutes/47:32.json")
    old = Path("us-la/statutes/47:297/4.yaml")
    old_manifest = Path(".axiom/encoding-manifests/us-la/statutes/47:297/4.json")
    successor = Path("us-la/statutes/47/297/4.yaml")
    successor_manifest = Path(".axiom/encoding-manifests/us-la/statutes/47/297/4.json")
    metadata = Path("known-validation-gaps.yaml")
    for path, raw in {
        main: b"main\n",
        main_manifest: b"main manifest\n",
        old: b"old\n",
        old_manifest: b"old manifest\n",
        successor: b"successor\n",
        successor_manifest: b"signed successor manifest\n",
        metadata: b"old metadata\n",
    }.items():
        target = checkout / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)

    def evidence(path: Path) -> LegacyReplacementFile:
        raw = (checkout / path).read_bytes()
        return LegacyReplacementFile(path, hashlib.sha256(raw).hexdigest(), raw)

    successor_manifest_evidence = evidence(successor_manifest)
    contract = LegacyReplacementContract(
        base_commit="a" * 40,
        base_tree="b" * 40,
        source=main,
        destination=main,
        legacy_manifest=evidence(main_manifest),
        deleted_files=(evidence(main),),
        rewrites=(),
        scheduled_dependents=(),
        exact_dependents=(),
        retained_successors=(
            LegacyReplacementRetainedSuccessor(
                source=old,
                destination=successor,
                legacy_manifest=evidence(old_manifest),
                legacy_files=(evidence(old),),
                successor_manifest=successor_manifest_evidence,
                successor_files=(evidence(successor),),
            ),
        ),
        metadata_reconciliations=(
            LegacyReplacementRewrite(
                path=metadata,
                before_sha256=hashlib.sha256(b"old metadata\n").hexdigest(),
                after_sha256=hashlib.sha256(b"new metadata\n").hexdigest(),
                replacements=({"operation": "test", "count": 1},),
                raw=b"new metadata\n",
            ),
        ),
    )

    (checkout / successor_manifest).write_text("tampered\n")
    with pytest.raises(LegacyReplacementOverlayError, match="signed manifest changed"):
        stage_legacy_replacement_overlay(contract, checkout)
    assert (checkout / main).exists()
    assert (checkout / old).exists()
    assert (checkout / metadata).read_bytes() == b"old metadata\n"

    (checkout / successor_manifest).write_bytes(successor_manifest_evidence.raw)
    stage_legacy_replacement_overlay(contract, checkout)
    assert not (checkout / main).exists()
    assert not (checkout / main_manifest).exists()
    assert not (checkout / old).exists()
    assert not (checkout / old_manifest).exists()
    assert not (checkout / old.parent).exists()
    assert not (checkout / old_manifest.parent).exists()
    assert (checkout / successor).read_bytes() == b"successor\n"
    assert (
        checkout / successor_manifest
    ).read_bytes() == successor_manifest_evidence.raw
    assert (checkout / metadata).read_bytes() == b"new metadata\n"


def test_empty_overlay_pruning_preserves_unowned_legacy_content(tmp_path: Path) -> None:
    checkout = tmp_path / "rulespec-us"
    removed = checkout / "us-la/statutes/47:297/4.yaml"
    sentinel = checkout / "us-la/statutes/47:297/unowned.yaml"
    sentinel.parent.mkdir(parents=True)
    removed.write_text("legacy\n")
    sentinel.write_text("unowned\n")

    removed.unlink()
    _prune_empty_overlay_parent_directories(checkout, [removed])

    assert sentinel.read_text() == "unowned\n"
    assert sentinel.parent.is_dir()


def test_empty_overlay_pruning_rejects_target_equal_to_protected_floor(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "rulespec-us"
    protected_floor = checkout / "us-la/statutes"
    protected_floor.mkdir(parents=True)

    with pytest.raises(
        LegacyReplacementOverlayError,
        match="strict descendant of its protected prune floor",
    ):
        _prune_empty_overlay_parent_directories(checkout, [protected_floor])
