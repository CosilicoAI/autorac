from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from axiom_encode.cli import _legacy_replacement_manifest_issues
from axiom_encode.legacy_replacement import (
    receipt_identity_payload,
    receipt_identity_sha256,
)
from scripts.prepare_signed_backfill import (
    MAX_SOURCE_BUNDLE_JSON_BYTES,
    REVIEWED_RULESPEC_PR_BASE_BRANCHES,
    REVIEWED_RULESPEC_REFS,
    authorized_changed_paths,
    branch_name,
    parse_existing_signed_imports,
    parse_source_bundle,
    stage_authorized_changes,
    validate_country,
    validate_dependent_cascade,
    validate_queue_tracking,
    validate_rulespec_base,
)
from scripts.prepare_signed_backfill import (
    main as prepare_signed_backfill_main,
)


def test_parse_source_bundle_accepts_ordered_same_jurisdiction_citations() -> None:
    raw = json.dumps(
        [
            "us-ri/statute/44-30-1",
            "us-ri/guidance/revenue/2026/rate-schedule",
        ]
    )

    assert parse_source_bundle(
        raw,
        primary_citation="us-ri/statute/44-30-2.6",
        excluded_citations=("us-ri/statute/44-30-5",),
    ) == (
        "us-ri/statute/44-30-1",
        "us-ri/guidance/revenue/2026/rate-schedule",
    )


def test_parse_source_bundle_accepts_empty_array() -> None:
    assert (
        parse_source_bundle(
            "[]",
            primary_citation="us-ri/statute/44-30-2.6",
        )
        == ()
    )


@pytest.mark.parametrize("raw", ['{"citation": "x"}', '"x"', "null"])
def test_parse_source_bundle_requires_json_array(raw: str) -> None:
    with pytest.raises(ValueError, match="must be an array"):
        parse_source_bundle(
            raw,
            primary_citation="us-ri/statute/44-30-2.6",
        )


def test_parse_source_bundle_rejects_more_than_sixteen_items() -> None:
    raw = json.dumps([f"us-ri/statute/44-30-{index}" for index in range(17)])

    with pytest.raises(ValueError, match="more than 16"):
        parse_source_bundle(
            raw,
            primary_citation="us-ri/statute/44-30-99",
        )


def test_parse_source_bundle_rejects_oversized_json_before_parsing() -> None:
    raw = " " * (MAX_SOURCE_BUNDLE_JSON_BYTES + 1)

    with pytest.raises(ValueError, match="maximum input size"):
        parse_source_bundle(
            raw,
            primary_citation="us-ri/statute/44-30-2.6",
        )


@pytest.mark.parametrize("item", ["", None, 4])
def test_parse_source_bundle_requires_nonempty_string_items(item: object) -> None:
    with pytest.raises(ValueError, match="nonempty citation string"):
        parse_source_bundle(
            json.dumps([item]),
            primary_citation="us-ri/statute/44-30-2.6",
        )


def test_parse_source_bundle_requires_exact_canonical_items() -> None:
    with pytest.raises(ValueError, match="exact canonical corpus citation path"):
        parse_source_bundle(
            json.dumps([" us-ri/statute/44-30-1"]),
            primary_citation="us-ri/statute/44-30-2.6",
        )


def test_parse_source_bundle_rejects_duplicate_citations() -> None:
    citation = "us-ri/statute/44-30-1"

    with pytest.raises(ValueError, match="must be unique"):
        parse_source_bundle(
            json.dumps([citation, citation]),
            primary_citation="us-ri/statute/44-30-2.6",
        )


@pytest.mark.parametrize(
    "forbidden",
    ["us-ri/statute/44-30-2.6", "us-ri/statute/44-30-5"],
)
def test_parse_source_bundle_rejects_primary_and_excluded_citations(
    forbidden: str,
) -> None:
    with pytest.raises(ValueError, match="primary and excluded citations"):
        parse_source_bundle(
            json.dumps([forbidden]),
            primary_citation="us-ri/statute/44-30-2.6",
            excluded_citations=("us-ri/statute/44-30-5",),
        )


def test_parse_source_bundle_rejects_other_jurisdiction() -> None:
    with pytest.raises(ValueError, match="jurisdiction and country"):
        parse_source_bundle(
            json.dumps(["us-ma/statute/62/4"]),
            primary_citation="us-ri/statute/44-30-2.6",
        )


def test_parse_source_bundle_rejects_exclusion_from_other_jurisdiction() -> None:
    with pytest.raises(ValueError, match="jurisdiction and country"):
        parse_source_bundle(
            "[]",
            primary_citation="us-ri/statute/44-30-2.6",
            excluded_citations=("us-ma/statute/62/4",),
        )


def test_parse_source_bundle_rejects_citation_rulespec_path_collision() -> None:
    with pytest.raises(ValueError, match="unique, unreserved canonical RuleSpec"):
        parse_source_bundle(
            json.dumps(
                [
                    "us-ri/guidance/revenue/section/1.2",
                    "us-ri/guidance/revenue/section/1/2",
                ]
            ),
            primary_citation="us-ri/statute/44-30-2.6",
        )


def test_parse_source_bundle_rejects_primary_rulespec_path_collision() -> None:
    with pytest.raises(ValueError, match="unique, unreserved canonical RuleSpec"):
        parse_source_bundle(
            json.dumps(["us-ri/guidance/revenue/section/1/2"]),
            primary_citation="us-ri/guidance/revenue/section/1.2",
        )


def test_parse_source_bundle_cli_emits_normalized_json_array(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_signed_backfill.py",
            "parse-source-bundle",
            '["us-ri/statute/44-30-1"]',
            "--primary-citation",
            "us-ri/statute/44-30-2.6",
        ],
    )

    prepare_signed_backfill_main()

    assert capsys.readouterr().out == '["us-ri/statute/44-30-1"]\n'


def _add_existing_signed_import(repo: Path, path: str) -> None:
    module = repo / path
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("rules: []\n", encoding="utf-8")
    manifest = repo / ".axiom/encoding-manifests" / Path(path).with_suffix(".json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"schema_version": "axiom-encode/applied-rulespec/v5"}) + "\n",
        encoding="utf-8",
    )
    _git(repo, "add", path, manifest.relative_to(repo).as_posix())
    _git(repo, "commit", "-m", f"add {path}")


def test_parse_existing_signed_imports_accepts_ordered_tracked_v5_modules(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    paths = (
        "us-ri/statutes/44-30-1.yaml",
        "us-ri/policies/revenue/rate-schedule.yaml",
    )
    for path in paths:
        _add_existing_signed_import(repo, path)

    assert parse_existing_signed_imports(
        repo,
        json.dumps(paths),
        primary_citation="us-ri/statute/44-30-2.6",
    ) == tuple(PurePosixPath(path) for path in paths)


def test_parse_existing_signed_imports_enforces_combined_sixteen_limit(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)

    with pytest.raises(ValueError, match="more than 16 modules"):
        parse_existing_signed_imports(
            repo,
            '["us-ri/statutes/44-30-1.yaml"]',
            primary_citation="us-ri/statute/44-30-99",
            source_bundle_citations=tuple(
                f"us-ri/statute/44-30-{index}" for index in range(16)
            ),
        )


@pytest.mark.parametrize(
    ("path", "kwargs"),
    [
        (
            "us-ri/statutes/44-30-2/6.yaml",
            {},
        ),
        (
            "us-ri/statutes/44-30-1.yaml",
            {"source_bundle_citations": ("us-ri/statute/44-30-1",)},
        ),
        (
            "us-ri/statutes/44-30-5.yaml",
            {"excluded_citations": ("us-ri/statute/44-30-5",)},
        ),
        (
            "us-ri/policies/income_tax/target.yaml",
            {"excluded_rulespec_paths": ("us-ri/policies/income_tax/target.yaml",)},
        ),
    ],
)
def test_parse_existing_signed_imports_rejects_reserved_paths(
    tmp_path: Path,
    path: str,
    kwargs: dict[str, tuple[str, ...]],
) -> None:
    repo = _repo(tmp_path)

    with pytest.raises(ValueError, match="must exclude"):
        parse_existing_signed_imports(
            repo,
            json.dumps([path]),
            primary_citation="us-ri/statute/44-30-2.6",
            **kwargs,
        )


@pytest.mark.parametrize(
    "payload",
    [
        '["us-ma/statutes/62/4.yaml"]',
        '["us-ri/statutes/44-30-1.test.yaml"]',
        '["us-ri/statutes/44-30-1.yaml", "us-ri/statutes/44-30-1.yaml"]',
    ],
)
def test_parse_existing_signed_imports_rejects_noncanonical_or_duplicate_paths(
    tmp_path: Path,
    payload: str,
) -> None:
    repo = _repo(tmp_path)

    with pytest.raises(ValueError):
        parse_existing_signed_imports(
            repo,
            payload,
            primary_citation="us-ri/statute/44-30-2.6",
        )


def test_parse_existing_signed_imports_requires_tracked_v5_manifest(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    path = "us-ri/statutes/44-30-1.yaml"
    module = repo / path
    module.parent.mkdir(parents=True)
    module.write_text("rules: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly tracked"):
        parse_existing_signed_imports(
            repo,
            json.dumps([path]),
            primary_citation="us-ri/statute/44-30-2.6",
        )


def test_parse_existing_signed_imports_cli_emits_normalized_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _repo(tmp_path)
    path = "us-ri/statutes/44-30-1.yaml"
    _add_existing_signed_import(repo, path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_signed_backfill.py",
            "parse-existing-signed-imports",
            str(repo),
            json.dumps([path]),
            "--primary-citation",
            "us-ri/statute/44-30-2.6",
        ],
    )

    prepare_signed_backfill_main()

    assert capsys.readouterr().out == f'["{path}"]\n'


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    return repo


def _add_origin_main(repo: Path) -> str:
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    return base


def _write_signed_change(repo: Path) -> tuple[Path, Path]:
    rule = repo / "us/regulations/example.yaml"
    rule.parent.mkdir(parents=True)
    rule.write_text("rules: []\n", encoding="utf-8")
    manifest = repo / ".axiom/encoding-manifests/us/regulations/example.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "axiom-encode/applied-rulespec/v5",
                "applied_files": [{"path": "us/regulations/example.yaml"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return rule, manifest


def _legacy_receipt_identity(receipt: dict[str, object]) -> dict[str, object]:
    repository = receipt["repository"]
    legacy = receipt["legacy"]
    replacement = receipt["replacement"]
    assert isinstance(repository, dict)
    assert isinstance(legacy, dict)
    assert isinstance(replacement, dict)
    legacy_manifest = legacy["manifest"]
    legacy_files = legacy["files"]
    live_files = replacement["live_files"]
    assert isinstance(legacy_manifest, dict)
    assert isinstance(legacy_files, list)
    assert isinstance(live_files, list)
    live_paths = {item["path"] for item in live_files if isinstance(item, dict)}
    deleted_files = [
        {"path": item["path"], "deleted": True}
        for item in legacy_files
        if isinstance(item, dict) and item.get("path") not in live_paths
    ]
    schema = receipt["schema_version"]
    retained_successors = replacement.get("retained_successors")
    if schema == "axiom-encode/legacy-fresh-reencode-receipt/v4":
        assert isinstance(retained_successors, list)
        deleted_files.extend(
            {"path": item["path"], "deleted": True}
            for successor in retained_successors
            if isinstance(successor, dict)
            for item in successor.get("legacy_files", [])
            if isinstance(item, dict)
        )
    return receipt_identity_payload(
        base_commit=str(repository["base_commit"]),
        base_tree=str(repository["base_tree"]),
        legacy_manifest_sha256=str(legacy_manifest["sha256"]),
        model_manifest_sha256=str(replacement["model_manifest_sha256"]),
        live_files=live_files,
        deleted_files=deleted_files,
        rewrites=replacement["rewrites"],
        scheduled_dependents=replacement["scheduled_dependents"],
        exact_dependents=(
            replacement["exact_dependents"]
            if schema
            in {
                "axiom-encode/legacy-fresh-reencode-receipt/v2",
                "axiom-encode/legacy-fresh-reencode-receipt/v3",
                "axiom-encode/legacy-fresh-reencode-receipt/v4",
            }
            else None
        ),
        destination_predecessor_class=(
            str(replacement["destination_predecessor_class"])
            if schema
            in {
                "axiom-encode/legacy-fresh-reencode-receipt/v3",
                "axiom-encode/legacy-fresh-reencode-receipt/v4",
            }
            else None
        ),
        destination_predecessor_files=(
            replacement["destination_predecessor_files"]
            if schema
            in {
                "axiom-encode/legacy-fresh-reencode-receipt/v3",
                "axiom-encode/legacy-fresh-reencode-receipt/v4",
            }
            else None
        ),
        retained_successors=(
            retained_successors
            if schema == "axiom-encode/legacy-fresh-reencode-receipt/v4"
            else None
        ),
        metadata_reconciliations=(
            replacement["metadata_reconciliations"]
            if schema == "axiom-encode/legacy-fresh-reencode-receipt/v4"
            else None
        ),
    )


def _write_legacy_replacement_change(
    repo: Path,
    *,
    scheduled_pending: bool = False,
    omitted_dependent: bool = False,
    omitted_scheduled_companion: bool = False,
    exact_dependent: bool = False,
    generated_exact_dependent: bool = False,
    destination_predecessor: bool = False,
    retained_successor: bool = False,
    legacy_owner_class: str = "v1-hmac-untrusted",
) -> tuple[Path, Path, Path, Path]:
    old_rule = repo / "us/statutes/47:32.yaml"
    old_test = repo / "us/statutes/47:32.test.yaml"
    old_rule.parent.mkdir(parents=True)
    old_rule.write_text("rules: []\n", encoding="utf-8")
    old_test.write_text("[]\n", encoding="utf-8")
    old_rule_sha256 = hashlib.sha256(old_rule.read_bytes()).hexdigest()
    old_test_sha256 = hashlib.sha256(old_test.read_bytes()).hexdigest()
    old_manifest = repo / ".axiom/encoding-manifests/us/statutes/47:32.json"
    old_manifest.parent.mkdir(parents=True, exist_ok=True)
    old_manifest.write_text(
        json.dumps(
            {
                "schema_version": "axiom-encode/applied-rulespec/v1",
                "tool": "axiom-encode sign-applied-files",
                "backend": "manual",
                "runner": "manual-attestation",
                "manual_exception": "test legacy evidence",
                "applied_files": [
                    {
                        "path": old_rule.relative_to(repo).as_posix(),
                        "sha256": old_rule_sha256,
                    },
                    {
                        "path": old_test.relative_to(repo).as_posix(),
                        "sha256": old_test_sha256,
                    },
                ],
                "signature": {
                    "algorithm": "hmac-sha256",
                    "key_id": "historical-v1",
                    "value": "opaque-untrusted-evidence",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = repo / ".axiom/index/provisions_to_rules.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"module":"us:statutes/47:32"}\n', encoding="utf-8")
    dependent = repo / "us/policies/income_tax/dependent.yaml"
    if scheduled_pending:
        dependent.parent.mkdir(parents=True)
        dependent.write_text(
            "format: rulespec/v1\n"
            "imports:\n"
            "  - us:statutes/47:32\n"
            "  - us:statutes/99:1\n"
            "rules: []\n",
            encoding="utf-8",
        )
        if omitted_scheduled_companion:
            dependent.with_name("dependent.test.yaml").write_text(
                "imports:\n  - us:statutes/47:32\n",
                encoding="utf-8",
            )
    if omitted_dependent:
        omitted = repo / "us/policies/income_tax/omitted.yaml"
        omitted.parent.mkdir(parents=True, exist_ok=True)
        omitted.write_text(
            "format: rulespec/v1\nimports:\n  - us:statutes/47:32\nrules: []\n",
            encoding="utf-8",
        )
    exact_primary = repo / "us/policies/income_tax/composite.yaml"
    exact_companion = repo / "us/policies/income_tax/composite.test.yaml"
    exact_manifest = (
        repo / ".axiom/encoding-manifests/us/policies/income_tax/composite.json"
    )
    new_rule = repo / "us/statutes/47/32.yaml"
    new_test = repo / "us/statutes/47/32.test.yaml"
    retained_old_rule = repo / "us/statutes/47:294.yaml"
    retained_old_test = repo / "us/statutes/47:294.test.yaml"
    retained_old_manifest = repo / ".axiom/encoding-manifests/us/statutes/47:294.json"
    retained_rule = repo / "us/statutes/47/294.yaml"
    retained_test = repo / "us/statutes/47/294.test.yaml"
    retained_manifest = repo / ".axiom/encoding-manifests/us/statutes/47/294.json"
    retained_metadata = repo / "known-validation-gaps.yaml"
    retained_old_files: list[dict[str, str]] = []
    retained_files: list[dict[str, str]] = []
    retained_old_manifest_raw = b""
    retained_manifest_raw = b""
    retained_manifest_payload: dict[str, object] = {}
    retained_metadata_before = b""
    if retained_successor:
        retained_old_rule.parent.mkdir(parents=True, exist_ok=True)
        retained_rule.parent.mkdir(parents=True, exist_ok=True)
        retained_old_rule.write_text("format: rulespec/v1\nrules: []\n")
        retained_old_test.write_text("[]\n")
        retained_rule.write_bytes(retained_old_rule.read_bytes())
        retained_test.write_bytes(retained_old_test.read_bytes())
        retained_old_files = [
            {
                "path": path.relative_to(repo).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (retained_old_rule, retained_old_test)
        ]
        retained_files = [
            {
                "path": path.relative_to(repo).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (retained_rule, retained_test)
        ]
        retained_old_manifest.parent.mkdir(parents=True, exist_ok=True)
        retained_old_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "axiom-encode/applied-rulespec/v1",
                    "tool": "axiom-encode sign-applied-files",
                    "backend": "manual",
                    "runner": "manual-attestation",
                    "manual_exception": "test retained legacy evidence",
                    "applied_files": retained_old_files,
                    "signature": {
                        "algorithm": "hmac-sha256",
                        "key_id": "historical-v1",
                        "value": "opaque-untrusted-evidence",
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )
        retained_old_manifest_raw = retained_old_manifest.read_bytes()
        retained_manifest_payload = {
            "schema_version": "axiom-encode/applied-rulespec/v5",
            "tool": "axiom-encode encode --apply",
            "backend": "codex",
            "citation": "us/statute/47:294",
            "applied_files": retained_files,
            "signature": {
                "algorithm": "ed25519-domain-v1",
                "key_id": f"sha256:{'d' * 64}",
                "value": "signed-test-value",
            },
        }
        retained_manifest.parent.mkdir(parents=True, exist_ok=True)
        retained_manifest.write_text(
            json.dumps(retained_manifest_payload, indent=2, sort_keys=True) + "\n"
        )
        retained_manifest_raw = retained_manifest.read_bytes()
        retained_metadata.write_text(
            "'us/statutes/47:294.yaml':\n  reason: legacy\n"
            "'us/statutes/47/294.yaml':\n  reason: canonical\n"
        )
        retained_metadata_before = retained_metadata.read_bytes()
    destination_predecessor_files: list[dict[str, str]] = []
    if destination_predecessor:
        new_rule.parent.mkdir(parents=True, exist_ok=True)
        new_rule.write_bytes(old_rule.read_bytes())
        new_test.write_bytes(old_test.read_bytes())
        destination_predecessor_files = [
            {
                "path": new_rule.relative_to(repo).as_posix(),
                "sha256": hashlib.sha256(new_rule.read_bytes()).hexdigest(),
            },
            {
                "path": new_test.relative_to(repo).as_posix(),
                "sha256": hashlib.sha256(new_test.read_bytes()).hexdigest(),
            },
        ]
    if exact_dependent:
        exact_primary.parent.mkdir(parents=True, exist_ok=True)
        exact_primary.write_text(
            "format: rulespec/v1\n"
            + (
                "module:\n"
                "  source_verification:\n"
                "    corpus_citation_path: us/statute/26/1\n"
                if generated_exact_dependent
                else ""
            )
            + "imports:\n"
            "  - us:statutes/47:32#amount\n"
            "rules:\n"
            "  - name: liability\n"
            "    kind: derived\n"
            "    dtype: Money\n"
            "    period: Year\n"
            "    metadata:\n"
            "      proof:\n"
            "        atoms:\n"
            "          - path: versions[0].formula\n"
            "            kind: import\n"
            "            import:\n"
            "              target: us:statutes/47:32#amount\n"
            "              output: amount\n"
            f"              hash: sha256:{old_rule_sha256}\n"
            "    versions:\n"
            "      - effective_from: '2026-01-01'\n"
            "        formula: amount\n",
            encoding="utf-8",
        )
        exact_companion.write_text("[]\n", encoding="utf-8")
        exact_manifest.parent.mkdir(parents=True, exist_ok=True)
        exact_applied_files = [
            {
                "path": exact_primary.relative_to(repo).as_posix(),
                "sha256": hashlib.sha256(exact_primary.read_bytes()).hexdigest(),
            },
            {
                "path": exact_companion.relative_to(repo).as_posix(),
                "sha256": hashlib.sha256(exact_companion.read_bytes()).hexdigest(),
            },
        ]
        exact_legacy_payload = {
            "schema_version": "axiom-encode/applied-rulespec/v1",
            "tool": "axiom-encode sign-applied-files",
            "backend": "manual",
            "runner": "manual-attestation",
            "manual_exception": "test exact dependent evidence",
            "applied_files": exact_applied_files,
            "signature": {
                "algorithm": "hmac-sha256",
                "key_id": "historical-v1",
                "value": "opaque-untrusted-evidence",
            },
        }
        if generated_exact_dependent:
            exact_legacy_payload = {
                "schema_version": "axiom-encode/applied-rulespec/v1",
                "tool": "axiom-encode encode --apply",
                "backend": "codex",
                "runner": "codex-gpt-5.5",
                "model": "gpt-5.5",
                "citation": "us/statute/26/1",
                "generated_at": "2026-07-16T19:07:09.279448+00:00",
                "run_id": "generated-exact-dependent",
                "axiom_encode_version": "0.2.1200",
                "axiom_encode_git": {
                    "commit": "c" * 40,
                    "dirty_tracked": False,
                    "root": "/tmp/axiom-encode",
                    "version": "0.2.1200",
                    "version_commit": "c" * 40,
                },
                "generation_prompt_sha256": "d" * 64,
                "generated_output_root": "/tmp/axiom-generated",
                "generated_output_file": (
                    "/tmp/axiom-generated/us/policies/income_tax/composite.yaml"
                ),
                "generated_output_sha256": exact_applied_files[0]["sha256"],
                "trace_file": "/tmp/axiom-generated/trace.json",
                "trace_sha256": "e" * 64,
                "context_manifest_file": "/tmp/axiom-generated/context.json",
                "context_manifest_sha256": "f" * 64,
                "applied_files": exact_applied_files,
                "signature": {
                    "algorithm": "hmac-sha256",
                    "key_id": "axiom-encode-apply-v1",
                    "value": "1" * 64,
                },
            }
        exact_manifest.write_text(
            json.dumps(exact_legacy_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "legacy baseline")
    base_commit = _git(repo, "rev-parse", "HEAD")
    base_tree = _git(repo, "rev-parse", "HEAD^{tree}")

    old_manifest_sha256 = hashlib.sha256(old_manifest.read_bytes()).hexdigest()
    metadata_before_sha256 = hashlib.sha256(metadata.read_bytes()).hexdigest()
    dependent_before_sha256 = (
        hashlib.sha256(dependent.read_bytes()).hexdigest()
        if scheduled_pending
        else None
    )
    exact_primary_before = exact_primary.read_bytes() if exact_dependent else None
    exact_companion_before = exact_companion.read_bytes() if exact_dependent else None
    exact_manifest_before = exact_manifest.read_bytes() if exact_dependent else None
    old_rule.unlink()
    old_test.unlink()
    old_manifest.unlink()
    if retained_successor:
        retained_old_rule.unlink()
        retained_old_test.unlink()
        retained_old_manifest.unlink()
        retained_metadata.write_text(
            "'us/statutes/47/294.yaml':\n  reason: canonical\n"
        )
    metadata.write_text('{"module":"us:statutes/47/32"}\n', encoding="utf-8")
    if exact_dependent:
        exact_primary.write_text(
            exact_primary.read_text(encoding="utf-8").replace(
                "us:statutes/47:32",
                "us:statutes/47/32",
            ),
            encoding="utf-8",
        )

    new_rule.parent.mkdir(parents=True, exist_ok=True)
    new_rule.write_text("format: rulespec/v1\nrules: []\n", encoding="utf-8")
    new_test.write_text("[]\n", encoding="utf-8")
    if exact_dependent:
        exact_primary.write_text(
            exact_primary.read_text(encoding="utf-8").replace(
                old_rule_sha256,
                hashlib.sha256(new_rule.read_bytes()).hexdigest(),
            ),
            encoding="utf-8",
        )
    live_files = [
        {
            "path": new_rule.relative_to(repo).as_posix(),
            "sha256": hashlib.sha256(new_rule.read_bytes()).hexdigest(),
        },
        {
            "path": new_test.relative_to(repo).as_posix(),
            "sha256": hashlib.sha256(new_test.read_bytes()).hexdigest(),
        },
    ]
    legacy_files = [
        {
            "path": old_rule.relative_to(repo).as_posix(),
            "sha256": old_rule_sha256,
        },
        {
            "path": old_test.relative_to(repo).as_posix(),
            "sha256": old_test_sha256,
        },
    ]
    metadata_after_sha256 = hashlib.sha256(metadata.read_bytes()).hexdigest()
    nested_manifest = {
        "schema_version": "axiom-encode/applied-rulespec/v5",
        "tool": "axiom-encode encode --apply",
        "backend": "codex",
        "applied_files": live_files,
    }
    nested_raw = (json.dumps(nested_manifest, indent=2, sort_keys=True) + "\n").encode()
    receipt_relative = Path(".axiom/legacy-replacements") / "pending.json"
    receipt = repo / receipt_relative
    receipt.parent.mkdir(parents=True)
    receipt_payload = {
        "schema_version": (
            "axiom-encode/legacy-fresh-reencode-receipt/v3"
            if destination_predecessor
            else (
                "axiom-encode/legacy-fresh-reencode-receipt/v2"
                if exact_dependent
                else "axiom-encode/legacy-fresh-reencode-receipt/v1"
            )
        ),
        "tool": "axiom-encode encode --apply --replace-legacy-rulespec-path",
        "repository": {
            "base_commit": base_commit,
            "head_commit": base_commit,
            "base_tree": base_tree,
        },
        "legacy": {
            "owner_class": legacy_owner_class,
            "trusted_generated_provenance": False,
            "manifest": {
                "path": old_manifest.relative_to(repo).as_posix(),
                "sha256": old_manifest_sha256,
            },
            "files": legacy_files,
        },
        "replacement": {
            "source": old_rule.relative_to(repo).as_posix(),
            "destination": new_rule.relative_to(repo).as_posix(),
            "model_manifest_path": (".axiom/encoding-manifests/us/statutes/47/32.json"),
            "model_manifest_sha256": hashlib.sha256(nested_raw).hexdigest(),
            "rewrites": [
                {
                    "path": metadata.relative_to(repo).as_posix(),
                    "before_sha256": metadata_before_sha256,
                    "after_sha256": metadata_after_sha256,
                    "replacements": [
                        {
                            "from": "us:statutes/47:32",
                            "to": "us:statutes/47/32",
                            "count": 1,
                        }
                    ],
                }
            ],
            "live_files": live_files,
            "scheduled_dependents": (
                [
                    {
                        "primary": dependent.relative_to(repo).as_posix(),
                        "files": [
                            {
                                "path": dependent.relative_to(repo).as_posix(),
                                "before_sha256": dependent_before_sha256,
                                "replacements": [
                                    {
                                        "from": "us:statutes/47:32",
                                        "to": "us:statutes/47/32",
                                        "count": 1,
                                    }
                                ],
                            }
                        ],
                    }
                ]
                if scheduled_pending
                else []
            ),
        },
        "replacement_manifest": nested_manifest,
    }
    if destination_predecessor:
        receipt_payload["replacement"]["destination_predecessor_class"] = (
            "canonicalized-unowned-duplicate"
        )
        receipt_payload["replacement"]["destination_predecessor_files"] = (
            destination_predecessor_files
        )
        receipt_payload["replacement"]["exact_dependents"] = []
    if exact_dependent:
        assert exact_primary_before is not None
        assert exact_companion_before is not None
        assert exact_manifest_before is not None
        exact_live_files = [
            {
                "path": exact_companion.relative_to(repo).as_posix(),
                "sha256": hashlib.sha256(exact_companion.read_bytes()).hexdigest(),
            },
            {
                "path": exact_primary.relative_to(repo).as_posix(),
                "sha256": hashlib.sha256(exact_primary.read_bytes()).hexdigest(),
            },
        ]
        receipt_payload.update(
            {
                "generated_at": "2026-07-30T00:00:00+00:00",
                "axiom_encode_version": "0.2.1414",
                "axiom_encode_git": {
                    "commit": "a" * 40,
                    "dirty_tracked": False,
                },
                "validation_waiver_set_sha256": "b" * 64,
            }
        )
        receipt_payload["replacement"]["exact_dependents"] = [
            {
                "primary": exact_primary.relative_to(repo).as_posix(),
                "legacy_manifest": {
                    "path": exact_manifest.relative_to(repo).as_posix(),
                    "sha256": hashlib.sha256(exact_manifest_before).hexdigest(),
                },
                "legacy_files": [
                    {
                        "path": exact_companion.relative_to(repo).as_posix(),
                        "sha256": hashlib.sha256(exact_companion_before).hexdigest(),
                    },
                    {
                        "path": exact_primary.relative_to(repo).as_posix(),
                        "sha256": hashlib.sha256(exact_primary_before).hexdigest(),
                    },
                ],
                "live_files": exact_live_files,
                "rewrites": [
                    {
                        "path": exact_primary.relative_to(repo).as_posix(),
                        "before_sha256": hashlib.sha256(
                            exact_primary_before
                        ).hexdigest(),
                        "after_sha256": hashlib.sha256(
                            exact_primary.read_bytes()
                        ).hexdigest(),
                        "replacements": [
                            {
                                "from": "us:statutes/47:32",
                                "to": "us:statutes/47/32",
                                "count": 2,
                            }
                        ],
                        "proof_import_repairs": 1,
                    }
                ],
            }
        ]
    if retained_successor:
        retained_metadata_after = retained_metadata.read_bytes()
        receipt_payload["schema_version"] = (
            "axiom-encode/legacy-fresh-reencode-receipt/v4"
        )
        receipt_payload["replacement"].setdefault("exact_dependents", [])
        receipt_payload["replacement"].setdefault(
            "destination_predecessor_class", "none"
        )
        receipt_payload["replacement"].setdefault("destination_predecessor_files", [])
        receipt_payload["replacement"]["retained_successors"] = [
            {
                "source": retained_old_rule.relative_to(repo).as_posix(),
                "destination": retained_rule.relative_to(repo).as_posix(),
                "legacy_owner_class": "v1-manual-hmac-untrusted",
                "legacy_manifest": {
                    "path": retained_old_manifest.relative_to(repo).as_posix(),
                    "sha256": hashlib.sha256(retained_old_manifest_raw).hexdigest(),
                },
                "legacy_files": retained_old_files,
                "successor_manifest": {
                    "path": retained_manifest.relative_to(repo).as_posix(),
                    "sha256": hashlib.sha256(retained_manifest_raw).hexdigest(),
                    "payload": retained_manifest_payload,
                },
                "successor_files": retained_files,
            }
        ]
        receipt_payload["replacement"]["metadata_reconciliations"] = [
            {
                "path": retained_metadata.relative_to(repo).as_posix(),
                "before_sha256": hashlib.sha256(retained_metadata_before).hexdigest(),
                "after_sha256": hashlib.sha256(retained_metadata_after).hexdigest(),
                "operations": [
                    {
                        "operation": "remove_legacy_validation_gaps",
                        "count": 1,
                    }
                ],
            }
        ]
    receipt_relative = Path(".axiom/legacy-replacements") / (
        receipt_identity_sha256(_legacy_receipt_identity(receipt_payload)) + ".json"
    )
    receipt = repo / receipt_relative
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(receipt_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if retained_successor:
        retained_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "axiom-encode/applied-rulespec/v5",
                    "generated_at": "2026-07-30T00:00:00+00:00",
                    "tool": (
                        "axiom-encode encode --apply "
                        "--legacy-retained-successor-rulespec-path"
                    ),
                    "axiom_encode_version": "0.2.1414",
                    "axiom_encode_git": {
                        "commit": "a" * 40,
                        "dirty_tracked": False,
                    },
                    "validation_waiver_set_sha256": "b" * 64,
                    "applied_files": retained_files,
                    "retained_successor_manifest": retained_manifest_payload,
                    "legacy_migration": {
                        "receipt_path": receipt_relative.as_posix(),
                        "receipt_sha256": hashlib.sha256(
                            receipt.read_bytes()
                        ).hexdigest(),
                        "source": retained_old_rule.relative_to(repo).as_posix(),
                        "destination": retained_rule.relative_to(repo).as_posix(),
                        "legacy_manifest_path": retained_old_manifest.relative_to(
                            repo
                        ).as_posix(),
                        "legacy_manifest_sha256": hashlib.sha256(
                            retained_old_manifest_raw
                        ).hexdigest(),
                        "successor_manifest_sha256": hashlib.sha256(
                            retained_manifest_raw
                        ).hexdigest(),
                    },
                    "signature": {
                        "algorithm": "ed25519-domain-v1",
                        "key_id": f"sha256:{'d' * 64}",
                        "value": "signed-test-value",
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if exact_dependent:
        exact_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "axiom-encode/applied-rulespec/v5",
                    "generated_at": "2026-07-30T00:00:00+00:00",
                    "tool": (
                        "axiom-encode encode --apply "
                        "--legacy-exact-dependent-rulespec-path"
                    ),
                    "axiom_encode_version": receipt_payload["axiom_encode_version"],
                    "axiom_encode_git": receipt_payload["axiom_encode_git"],
                    "validation_waiver_set_sha256": receipt_payload[
                        "validation_waiver_set_sha256"
                    ],
                    "applied_files": receipt_payload["replacement"]["exact_dependents"][
                        0
                    ]["live_files"],
                    "legacy_migration": {
                        "receipt_path": receipt_relative.as_posix(),
                        "receipt_sha256": hashlib.sha256(
                            receipt.read_bytes()
                        ).hexdigest(),
                        "primary": exact_primary.relative_to(repo).as_posix(),
                        "legacy_manifest_path": exact_manifest.relative_to(
                            repo
                        ).as_posix(),
                        "legacy_manifest_sha256": hashlib.sha256(
                            exact_manifest_before
                        ).hexdigest(),
                    },
                    "signature": {
                        "algorithm": "ed25519-domain-v1",
                        "key_id": f"sha256:{'d' * 64}",
                        "value": "signed-test-value",
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    manifest = repo / ".axiom/encoding-manifests/us/statutes/47/32.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "axiom-encode/applied-rulespec/v5",
                "tool": ("axiom-encode encode --apply --replace-legacy-rulespec-path"),
                "replacement_manifest": nested_manifest,
                "replacement": {
                    "receipt_path": receipt_relative.as_posix(),
                    "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                    "legacy_manifest_path": old_manifest.relative_to(repo).as_posix(),
                    "legacy_manifest_sha256": old_manifest_sha256,
                },
                "applied_files": [
                    *live_files,
                    {
                        "path": metadata.relative_to(repo).as_posix(),
                        "sha256": metadata_after_sha256,
                    },
                    *(
                        [
                            {
                                "path": retained_metadata.relative_to(repo).as_posix(),
                                "sha256": hashlib.sha256(
                                    retained_metadata.read_bytes()
                                ).hexdigest(),
                            }
                        ]
                        if retained_successor
                        else []
                    ),
                    *[{"path": item["path"], "deleted": True} for item in legacy_files],
                    *(
                        [
                            {"path": item["path"], "deleted": True}
                            for item in retained_old_files
                        ]
                        if retained_successor
                        else []
                    ),
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, receipt, old_manifest, metadata


def _complete_scheduled_legacy_dependent(
    repo: Path,
    *,
    omitted_scheduled_companion: bool = False,
) -> tuple[Path, Path, Path]:
    manifest, receipt, _old_manifest, _metadata = _write_legacy_replacement_change(
        repo,
        scheduled_pending=True,
        omitted_scheduled_companion=omitted_scheduled_companion,
    )
    dependent = repo / "us/policies/income_tax/dependent.yaml"
    dependent.write_text(
        dependent.read_text(encoding="utf-8").replace(
            "us:statutes/47:32",
            "us:statutes/47/32",
        ),
        encoding="utf-8",
    )
    dependent_manifest = (
        repo / ".axiom/encoding-manifests/us/policies/income_tax/dependent.json"
    )
    dependent_manifest.parent.mkdir(parents=True, exist_ok=True)
    dependent_manifest.write_text(
        json.dumps(
            {
                "schema_version": "axiom-encode/applied-rulespec/v5",
                "tool": "axiom-encode encode --apply",
                "backend": "codex",
                "applied_files": [
                    {
                        "path": dependent.relative_to(repo).as_posix(),
                        "sha256": hashlib.sha256(dependent.read_bytes()).hexdigest(),
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, receipt, dependent


def _write_module(
    repo: Path,
    relative: str,
    *,
    imports: tuple[str, ...] = (),
) -> Path:
    path = repo / "us" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "format: rulespec/v1\n"
        + (
            "imports:\n" + "".join(f"  - {value}\n" for value in imports)
            if imports
            else "imports: []\n"
        )
        + "rules: []\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("country", ["../x", "us/x", "${{ inputs.x }}", "us\n"])
def test_validate_country_rejects_adversarial_values(country: str) -> None:
    with pytest.raises(ValueError, match="two-letter lowercase"):
        validate_country(country)


def test_validate_queue_tracking_accepts_complete_or_empty_metadata() -> None:
    assert validate_queue_tracking("", "", "", "") == "ad-hoc"
    assert (
        validate_queue_tracking(
            "us-snap-or-ut-2026-07",
            "ut-0001",
            "a" * 64,
            "b" * 64,
        )
        == "tracked"
    )


@pytest.mark.parametrize(
    ("queue_id", "item_id", "digest", "generation", "message"),
    [
        ("queue", "", "a" * 64, "b" * 64, "supplied together"),
        ("queue/unsafe", "ut-0001", "a" * 64, "b" * 64, "queue_id"),
        ("queue", "ut/0001", "a" * 64, "b" * 64, "queue_item_id"),
        ("queue", "ut-0001", "abc", "b" * 64, "queue_manifest_sha256"),
        ("queue", "ut-0001", "a" * 64, "abc", "queue_item_generation"),
    ],
)
def test_validate_queue_tracking_rejects_incomplete_or_unsafe_metadata(
    queue_id: str,
    item_id: str,
    digest: str,
    generation: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_queue_tracking(queue_id, item_id, digest, generation)


def test_stage_authorized_changes_stages_only_manifest_and_applied_files(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    rule, manifest = _write_signed_change(repo)

    stage_authorized_changes(repo)

    assert _git(repo, "diff", "--cached", "--name-only").splitlines() == sorted(
        [str(manifest.relative_to(repo)), str(rule.relative_to(repo))]
    )


def test_stage_authorized_changes_rejects_unexpected_file(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write_signed_change(repo)
    (repo / "encoder-surprise.txt").write_text("must not publish\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside signed manifest authorization"):
        stage_authorized_changes(repo)

    assert _git(repo, "diff", "--cached", "--name-only") == ""


def test_stage_rejects_manifest_authorizing_non_rulespec_path(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _rule, manifest = _write_signed_change(repo)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["applied_files"] = [{"path": ".github/workflows/pwn.yml"}]
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    unexpected = repo / ".github/workflows/pwn.yml"
    unexpected.parent.mkdir(parents=True)
    unexpected.write_text("name: unexpected\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a canonical RuleSpec YAML path"):
        stage_authorized_changes(repo)

    assert _git(repo, "diff", "--cached", "--name-only") == ""


def test_stage_accepts_receipt_linked_legacy_deletion_with_dependent_manifest(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_legacy_replacement_change(repo)
    _write_signed_change(repo)

    expected = authorized_changed_paths(repo)
    stage_authorized_changes(repo)

    assert set(
        _git(repo, "diff", "--cached", "--name-only", "--no-renames").splitlines()
    ) == {path.as_posix() for path in expected}


def _refresh_legacy_receipt_bindings(
    repo: Path,
    manifest: Path,
    receipt: Path,
) -> None:
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_relative = Path(".axiom/legacy-replacements") / (
        receipt_identity_sha256(_legacy_receipt_identity(receipt_payload)) + ".json"
    )
    refreshed_receipt = repo / receipt_relative
    if refreshed_receipt != receipt:
        receipt.replace(refreshed_receipt)
    receipt_sha256 = hashlib.sha256(refreshed_receipt.read_bytes()).hexdigest()
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["replacement"]["receipt_path"] = receipt_relative.as_posix()
    manifest_payload["replacement"]["receipt_sha256"] = receipt_sha256
    manifest.write_text(
        json.dumps(manifest_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    exact_manifest = (
        repo / ".axiom/encoding-manifests/us/policies/income_tax/composite.json"
    )
    if exact_manifest.is_file():
        exact_payload = json.loads(exact_manifest.read_text(encoding="utf-8"))
        exact_payload["legacy_migration"]["receipt_path"] = receipt_relative.as_posix()
        exact_payload["legacy_migration"]["receipt_sha256"] = receipt_sha256
        exact_manifest.write_text(
            json.dumps(exact_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    replacement = receipt_payload.get("replacement")
    retained = (
        replacement.get("retained_successors", [])
        if isinstance(replacement, dict)
        else []
    )
    for successor in retained if isinstance(retained, list) else []:
        if not isinstance(successor, dict):
            continue
        evidence = successor.get("successor_manifest")
        if not isinstance(evidence, dict) or not isinstance(evidence.get("path"), str):
            continue
        retained_manifest = repo / evidence["path"]
        if not retained_manifest.is_file():
            continue
        retained_payload = json.loads(retained_manifest.read_text(encoding="utf-8"))
        retained_payload["legacy_migration"]["receipt_path"] = (
            receipt_relative.as_posix()
        )
        retained_payload["legacy_migration"]["receipt_sha256"] = receipt_sha256
        retained_manifest.write_text(
            json.dumps(retained_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@pytest.mark.parametrize(
    ("legacy_owner_class", "generated_exact_dependent"),
    [
        ("v1-hmac-untrusted", False),
        ("v1-manual-hmac-untrusted", False),
        ("v1-hmac-untrusted", True),
    ],
)
def test_stage_accepts_v2_exact_dependent_with_unchanged_companion(
    tmp_path: Path,
    legacy_owner_class: str,
    generated_exact_dependent: bool,
) -> None:
    repo = _repo(tmp_path)
    _write_legacy_replacement_change(
        repo,
        exact_dependent=True,
        generated_exact_dependent=generated_exact_dependent,
        legacy_owner_class=legacy_owner_class,
    )

    expected = authorized_changed_paths(repo)
    stage_authorized_changes(repo)

    assert PurePosixPath("us/policies/income_tax/composite.test.yaml") not in expected
    assert (
        "us/policies/income_tax/composite.test.yaml"
        not in _git(repo, "diff", "--cached", "--name-only").splitlines()
    )


def test_stage_rejects_generated_exact_dependent_under_old_manual_owner_class(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_legacy_replacement_change(
        repo,
        exact_dependent=True,
        generated_exact_dependent=True,
        legacy_owner_class="v1-manual-hmac-untrusted",
    )

    with pytest.raises(ValueError, match="exact dependent legacy ownership is invalid"):
        authorized_changed_paths(repo)


def test_stage_keeps_v1_legacy_replacement_compatible(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _manifest, receipt, _old_manifest, _metadata = _write_legacy_replacement_change(
        repo
    )

    assert (
        json.loads(receipt.read_text(encoding="utf-8"))["schema_version"]
        == "axiom-encode/legacy-fresh-reencode-receipt/v1"
    )
    authorized_changed_paths(repo)


def test_stage_accepts_v3_unowned_canonical_destination_predecessor(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _manifest, receipt, _old_manifest, _metadata = _write_legacy_replacement_change(
        repo,
        destination_predecessor=True,
    )

    assert (
        json.loads(receipt.read_text(encoding="utf-8"))["schema_version"]
        == "axiom-encode/legacy-fresh-reencode-receipt/v3"
    )
    expected = authorized_changed_paths(repo)
    stage_authorized_changes(repo)

    assert PurePosixPath("us/statutes/47/32.yaml") in expected


def test_stage_accepts_v4_additive_reconciliation_contract(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    manifest, receipt, _old_manifest, _metadata = _write_legacy_replacement_change(
        repo,
        destination_predecessor=True,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["schema_version"] = "axiom-encode/legacy-fresh-reencode-receipt/v4"
    payload["replacement"]["retained_successors"] = []
    payload["replacement"]["metadata_reconciliations"] = []
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_legacy_receipt_bindings(repo, manifest, receipt)

    expected = authorized_changed_paths(repo)
    stage_authorized_changes(repo)

    assert PurePosixPath("us/statutes/47/32.yaml") in expected


def test_stage_accepts_v4_nonempty_retained_successor_identity(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _manifest, receipt, _old_manifest, _metadata = _write_legacy_replacement_change(
        repo, destination_predecessor=True, retained_successor=True
    )

    expected = authorized_changed_paths(repo)
    stage_authorized_changes(repo)

    assert (
        PurePosixPath(".axiom/encoding-manifests/us/statutes/47:294.json") in expected
    )
    assert PurePosixPath("us/statutes/47:294.yaml") in expected
    assert (
        ".axiom/encoding-manifests/us/statutes/47:294.json"
        in _git(repo, "diff", "--cached", "--name-only", "--no-renames").splitlines()
    )
    assert json.loads(receipt.read_text())["replacement"]["retained_successors"]


def test_persisted_verifier_reconstructs_nonempty_v4_retained_identity(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    manifest, receipt, _old_manifest, _metadata = _write_legacy_replacement_change(
        repo, destination_predecessor=True, retained_successor=True
    )
    manifest_label = manifest.relative_to(repo).as_posix()
    signing_key = Ed25519PrivateKey.generate().public_key()

    issues = _legacy_replacement_manifest_issues(
        json.loads(manifest.read_text()),
        repo_path=repo,
        manifest_label=manifest_label,
        signing_broker=signing_key,
        expected_waiver_set_sha256="b" * 64,
        local_corpus_release=None,
    )
    assert not any("replacement receipt identity is stale" in issue for issue in issues)

    receipt_payload = json.loads(receipt.read_text())
    receipt_payload["replacement"]["retained_successors"][0]["legacy_files"][0][
        "sha256"
    ] = "0" * 64
    receipt.write_text(json.dumps(receipt_payload, sort_keys=True) + "\n")
    manifest_payload = json.loads(manifest.read_text())
    manifest_payload["replacement"]["receipt_sha256"] = hashlib.sha256(
        receipt.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(manifest_payload, sort_keys=True) + "\n")

    tampered_issues = _legacy_replacement_manifest_issues(
        manifest_payload,
        repo_path=repo,
        manifest_label=manifest_label,
        signing_broker=signing_key,
        expected_waiver_set_sha256="b" * 64,
        local_corpus_release=None,
    )
    assert any(
        "replacement receipt identity is stale" in issue for issue in tampered_issues
    )


def test_stage_rejects_v4_retained_deleted_file_evidence_tamper(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    manifest, receipt, _old_manifest, _metadata = _write_legacy_replacement_change(
        repo, destination_predecessor=True, retained_successor=True
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["replacement"]["retained_successors"][0]["legacy_files"][0]["sha256"] = (
        "0" * 64
    )
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_legacy_receipt_bindings(repo, manifest, receipt)

    with pytest.raises(ValueError, match="retained successor.*base group"):
        authorized_changed_paths(repo)


def test_stage_rejects_v4_missing_retained_deleted_file_from_outer_manifest(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    manifest, _receipt, _old_manifest, _metadata = _write_legacy_replacement_change(
        repo, destination_predecessor=True, retained_successor=True
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["applied_files"] = [
        item
        for item in payload["applied_files"]
        if item.get("path") != "us/statutes/47:294.yaml"
    ]
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="outer applied_files differ"):
        authorized_changed_paths(repo)


def test_stage_rejects_v3_destination_predecessor_hash_tamper(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    manifest, receipt, _old_manifest, _metadata = _write_legacy_replacement_change(
        repo,
        destination_predecessor=True,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["replacement"]["destination_predecessor_files"][0]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_legacy_receipt_bindings(repo, manifest, receipt)

    with pytest.raises(ValueError, match="destination predecessor differs"):
        authorized_changed_paths(repo)


def test_stage_rejects_v2_exact_dependent_live_file_tamper(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_legacy_replacement_change(repo, exact_dependent=True)
    primary = repo / "us/policies/income_tax/composite.yaml"
    primary.write_text(primary.read_text(encoding="utf-8") + "# tampered\n")

    with pytest.raises(ValueError, match="live file hash differs"):
        authorized_changed_paths(repo)


def test_stage_rejects_v2_exact_dependent_manifest_binding_tamper(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_legacy_replacement_change(repo, exact_dependent=True)
    exact_manifest = (
        repo / ".axiom/encoding-manifests/us/policies/income_tax/composite.json"
    )
    payload = json.loads(exact_manifest.read_text(encoding="utf-8"))
    payload["legacy_migration"]["primary"] = "us/policies/income_tax/unrelated.yaml"
    exact_manifest.write_text(json.dumps(payload, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="manifest binding differs"):
        authorized_changed_paths(repo)


def test_stage_rejects_v2_exact_dependent_signature_tamper(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_legacy_replacement_change(repo, exact_dependent=True)
    exact_manifest = (
        repo / ".axiom/encoding-manifests/us/policies/income_tax/composite.json"
    )
    payload = json.loads(exact_manifest.read_text(encoding="utf-8"))
    payload["signature"] = {}
    exact_manifest.write_text(json.dumps(payload, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="manifest is malformed"):
        authorized_changed_paths(repo)


def test_stage_rejects_v2_exact_dependent_rewrite_tamper(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    manifest, receipt, _old_manifest, _metadata = _write_legacy_replacement_change(
        repo, exact_dependent=True
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["replacement"]["exact_dependents"][0]["rewrites"][0]["replacements"][0][
        "count"
    ] = 3
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n")
    _refresh_legacy_receipt_bindings(repo, manifest, receipt)

    with pytest.raises(ValueError, match="transformation differs"):
        authorized_changed_paths(repo)


def test_stage_rejects_v2_exact_dependent_proof_hash_repair_tamper(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    manifest, receipt, _old_manifest, _metadata = _write_legacy_replacement_change(
        repo, exact_dependent=True
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["replacement"]["exact_dependents"][0]["rewrites"][0][
        "proof_import_repairs"
    ] = 0
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n")
    _refresh_legacy_receipt_bindings(repo, manifest, receipt)

    with pytest.raises(ValueError, match="transformation differs"):
        authorized_changed_paths(repo)


def test_stage_rejects_v2_exact_dependent_incomplete_legacy_group(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    manifest, receipt, _old_manifest, _metadata = _write_legacy_replacement_change(
        repo, exact_dependent=True
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["replacement"]["exact_dependents"][0]["legacy_files"] = payload[
        "replacement"
    ]["exact_dependents"][0]["legacy_files"][1:]
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n")
    _refresh_legacy_receipt_bindings(repo, manifest, receipt)

    with pytest.raises(ValueError, match="exact full file group"):
        authorized_changed_paths(repo)


def test_stage_rejects_tampered_legacy_replacement_receipt(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _manifest, receipt, _old_manifest, _metadata = _write_legacy_replacement_change(
        repo
    )
    receipt.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="receipt digest differs"):
        stage_authorized_changes(repo)


def test_stage_rejects_signed_transition_while_dependent_is_pending(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_legacy_replacement_change(repo, scheduled_pending=True)

    with pytest.raises(
        ValueError,
        match="scheduled dependent lacks changed manifest",
    ):
        stage_authorized_changes(repo)


def test_stage_accepts_exact_completed_legacy_dependent(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _complete_scheduled_legacy_dependent(repo)

    authorized_changed_paths(repo)


def test_stage_rejects_inexact_legacy_dependent_replacement_count(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    manifest, receipt, _dependent = _complete_scheduled_legacy_dependent(repo)
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_payload["replacement"]["scheduled_dependents"][0]["files"][0][
        "replacements"
    ][0]["count"] = 2
    receipt.write_text(
        json.dumps(receipt_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_legacy_receipt_bindings(repo, manifest, receipt)

    with pytest.raises(ValueError, match="exact base proof differs"):
        stage_authorized_changes(repo)


def test_stage_rejects_unrelated_legacy_dependent_replacement_pair(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    manifest, receipt, dependent = _complete_scheduled_legacy_dependent(repo)
    dependent.write_text(
        "format: rulespec/v1\n"
        "imports:\n"
        "  - us:statutes/47:32\n"
        "  - us:statutes/99/1\n"
        "rules: []\n",
        encoding="utf-8",
    )
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_payload["replacement"]["scheduled_dependents"][0]["files"][0][
        "replacements"
    ] = [
        {
            "from": "us:statutes/99:1",
            "to": "us:statutes/99/1",
            "count": 1,
        }
    ]
    receipt.write_text(
        json.dumps(receipt_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_legacy_receipt_bindings(repo, manifest, receipt)
    dependent_manifest = (
        repo / ".axiom/encoding-manifests/us/policies/income_tax/dependent.json"
    )
    dependent_payload = json.loads(dependent_manifest.read_text(encoding="utf-8"))
    dependent_payload["applied_files"][0]["sha256"] = hashlib.sha256(
        dependent.read_bytes()
    ).hexdigest()
    dependent_manifest.write_text(
        json.dumps(dependent_payload) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact base proof differs"):
        stage_authorized_changes(repo)


def test_stage_rejects_omitted_legacy_dependent(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write_legacy_replacement_change(repo, omitted_dependent=True)

    with pytest.raises(
        ValueError,
        match=r"base reference inventory omits protected dependent .*omitted\.yaml",
    ):
        stage_authorized_changes(repo)


def test_stage_rejects_omitted_scheduled_dependent_companion(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _complete_scheduled_legacy_dependent(
        repo,
        omitted_scheduled_companion=True,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"base reference inventory omits protected dependent "
            r".*dependent\.test\.yaml"
        ),
    ):
        stage_authorized_changes(repo)


def test_stage_accepts_fresh_model_edits_beyond_legacy_reference_rewrite(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _manifest, _receipt, dependent = _complete_scheduled_legacy_dependent(repo)
    dependent.write_text(
        dependent.read_text(encoding="utf-8") + "# unrelated manual edit\n",
        encoding="utf-8",
    )
    dependent_manifest = (
        repo / ".axiom/encoding-manifests/us/policies/income_tax/dependent.json"
    )
    dependent_payload = json.loads(dependent_manifest.read_text(encoding="utf-8"))
    dependent_payload["applied_files"][0]["sha256"] = hashlib.sha256(
        dependent.read_bytes()
    ).hexdigest()
    dependent_manifest.write_text(
        json.dumps(dependent_payload) + "\n",
        encoding="utf-8",
    )

    authorized_changed_paths(repo)


def test_stage_rejects_unauthorized_legacy_metadata_rewrite(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    manifest, receipt, _old_manifest, metadata = _write_legacy_replacement_change(repo)
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_payload["replacement"]["rewrites"][0]["path"] = "README.md"
    receipt.write_text(
        json.dumps(receipt_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_legacy_receipt_bindings(repo, manifest, receipt)
    metadata.write_text('{"module":"us:statutes/47/32"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="rewrite\\[0\\] path is unauthorized"):
        stage_authorized_changes(repo)


def test_stage_rejects_unlinked_extra_deleted_manifest(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    extra = repo / ".axiom/encoding-manifests/us/statutes/extra.json"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("{}\n", encoding="utf-8")
    _write_legacy_replacement_change(repo)
    extra.unlink()

    with pytest.raises(
        ValueError,
        match="deleted manifests are not authenticated",
    ):
        stage_authorized_changes(repo)


def test_rerun_attempt_uses_recoverable_distinct_branch() -> None:
    first = branch_name("us", "12345", "1")
    rerun = branch_name("us", "12345", "2")

    assert first == "axiom/signed-backfill-us-12345-1"
    assert rerun == "axiom/signed-backfill-us-12345-2"
    assert rerun != first


def test_validate_dependent_cascade_accepts_only_direct_dependent(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_module(repo, "regulations/42-cfr/435/555.yaml")
    dependent = _write_module(
        repo,
        "regulations/42-cfr/435/559.yaml",
        imports=("us:regulations/42-cfr/435/555#target_rule",),
    )

    assert validate_dependent_cascade(
        repo,
        "us/regulation/42/435/555",
        "us/regulation/42/435/559",
    ) == (dependent.relative_to(repo / "us"),)


def test_validate_dependent_cascade_uses_nondefault_replacement_target(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    replacement = _write_module(
        repo,
        "policies/income_tax/pilot_liability_pipeline.yaml",
    )
    dependent = _write_module(
        repo,
        "regulations/42-cfr/435/559.yaml",
        imports=("us:policies/income_tax/pilot_liability_pipeline#target_rule",),
    )
    _write_module(repo, "regulations/42-cfr/435/555.yaml")

    assert validate_dependent_cascade(
        repo,
        "us/regulation/42/435/555",
        "us/regulation/42/435/559",
        target_rulespec_path=replacement.relative_to(repo).as_posix(),
    ) == (dependent.relative_to(repo / "us"),)


@pytest.mark.parametrize(
    "replacement_path",
    [
        "../outside.yaml",
        "us-nc/policies/income_tax/pipeline.yaml",
        "us/policies/income_tax/pipeline.test.yaml",
    ],
)
def test_validate_dependent_cascade_rejects_invalid_replacement_target(
    tmp_path: Path,
    replacement_path: str,
) -> None:
    repo = _repo(tmp_path)
    _write_module(repo, "regulations/42-cfr/435/555.yaml")
    _write_module(
        repo,
        "regulations/42-cfr/435/559.yaml",
        imports=("us:regulations/42-cfr/435/555",),
    )

    with pytest.raises(ValueError, match="target RuleSpec path must be"):
        validate_dependent_cascade(
            repo,
            "us/regulation/42/435/555",
            "us/regulation/42/435/559",
            target_rulespec_path=replacement_path,
        )


def test_validate_dependent_cascade_rejects_symlinked_replacement_parent(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "target.yaml").write_text(
        "rules:\n  target_rule:\n    kind: constant\n    value: 1\n",
        encoding="utf-8",
    )
    linked_parent = repo / "us" / "policies" / "linked"
    linked_parent.parent.mkdir(parents=True)
    linked_parent.symlink_to(outside, target_is_directory=True)
    _write_module(
        repo,
        "regulations/42-cfr/435/559.yaml",
        imports=("us:policies/linked/target#target_rule",),
    )

    with pytest.raises(
        ValueError,
        match="target citation has no regular baseline RuleSpec module",
    ):
        validate_dependent_cascade(
            repo,
            "us/regulation/42/435/555",
            "us/regulation/42/435/559",
            target_rulespec_path="us/policies/linked/target.yaml",
        )


def test_validate_dependent_cascade_rejects_symlinked_jurisdiction_root(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    outside = tmp_path / "outside-us"
    _write_module(
        outside.parent / "rulespec-outside",
        "policies/income_tax/target.yaml",
    )
    outside_source = outside.parent / "rulespec-outside" / "us"
    (repo / "us").symlink_to(outside_source, target_is_directory=True)
    _write_module(
        outside.parent / "rulespec-outside",
        "regulations/42-cfr/435/559.yaml",
        imports=("us:policies/income_tax/target#target_rule",),
    )

    with pytest.raises(
        ValueError,
        match="target citation has no regular baseline RuleSpec module",
    ):
        validate_dependent_cascade(
            repo,
            "us/regulation/42/435/555",
            "us/regulation/42/435/559",
            target_rulespec_path="us/policies/income_tax/target.yaml",
        )


def test_validate_dependent_cascade_rejects_unrelated_dependent(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_module(repo, "regulations/42-cfr/435/555.yaml")
    _write_module(
        repo,
        "regulations/42-cfr/435/559.yaml",
        imports=("us:regulations/42-cfr/435/555",),
    )
    _write_module(repo, "regulations/42-cfr/435/561.yaml")

    with pytest.raises(ValueError, match="does not exactly match"):
        validate_dependent_cascade(
            repo,
            "us/regulation/42/435/555",
            "us/regulation/42/435/561",
        )


def test_validate_dependent_cascade_accepts_all_direct_dependents(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_module(repo, "regulations/42-cfr/435/555.yaml")
    dependents = []
    for section in ("559", "561"):
        dependents.append(
            _write_module(
                repo,
                f"regulations/42-cfr/435/{section}.yaml",
                imports=("regulations/42-cfr/435/555",),
            ).relative_to(repo / "us")
        )

    assert validate_dependent_cascade(
        repo,
        "us/regulation/42/435/555",
        "us/regulation/42/435/559",
        "us/regulation/42/435/561",
    ) == tuple(dependents)


def test_validate_dependent_cascade_rejects_incomplete_direct_dependents(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_module(repo, "regulations/42-cfr/435/555.yaml")
    for section in ("559", "561"):
        _write_module(
            repo,
            f"regulations/42-cfr/435/{section}.yaml",
            imports=("regulations/42-cfr/435/555",),
        )

    with pytest.raises(ValueError, match="does not exactly match"):
        validate_dependent_cascade(
            repo,
            "us/regulation/42/435/555",
            "us/regulation/42/435/559",
        )


def test_validate_rulespec_base_accepts_main_ancestor(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _add_origin_main(repo)

    assert validate_rulespec_base(repo, "us", base, open_pr=True) == "main"


def test_validate_rulespec_base_rejects_non_main_pr_for_main_ancestor(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    base = _add_origin_main(repo)

    with pytest.raises(ValueError, match="must target main"):
        validate_rulespec_base(
            repo,
            "us",
            base,
            open_pr=True,
            pr_base_branch="hard-cut/canonical-layout-us",
        )


def test_validate_rulespec_base_rejects_stale_main_pr_base(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    stale_base = _add_origin_main(repo)
    (repo / "README.md").write_text("advanced\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "advance main")
    advanced_base = _git(repo, "rev-parse", "HEAD")
    _git(
        repo,
        "update-ref",
        "refs/remotes/origin/main",
        advanced_base,
    )
    _git(repo, "checkout", "--detach", stale_base)

    with pytest.raises(ValueError, match="exact pull request base branch tip"):
        validate_rulespec_base(repo, "us", stale_base, open_pr=True)


@pytest.mark.parametrize(
    ("country", "reviewed_ref"),
    [
        ("us", "b61918da93fe8a1a29b35b9330aef2085291a5d0"),
        ("us", "251d8d66dabdebcb763d9e7c9b8322a281440c36"),
        ("us", "68cca4a6fa806b63f95277c129575d88d2ac07f1"),
        ("us", "1e04e456ab404860050586c34eef51321eea95e9"),
        ("ca", "f60f7a84c30e38c7d4961d70647eb0457e7d76c2"),
    ],
)
def test_validate_rulespec_base_accepts_exact_reviewed_head_artifact_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    country: str,
    reviewed_ref: str,
) -> None:
    repo = tmp_path / f"rulespec-{country}"
    assert REVIEWED_RULESPEC_REFS == frozenset(
        {
            ("us", "b61918da93fe8a1a29b35b9330aef2085291a5d0"),
            ("us", "251d8d66dabdebcb763d9e7c9b8322a281440c36"),
            ("us", "68cca4a6fa806b63f95277c129575d88d2ac07f1"),
            ("us", "1e04e456ab404860050586c34eef51321eea95e9"),
            ("ca", "f60f7a84c30e38c7d4961d70647eb0457e7d76c2"),
        }
    )
    assert REVIEWED_RULESPEC_PR_BASE_BRANCHES == frozenset(
        {("us", "hard-cut/canonical-layout-us")}
    )
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill._git",
        lambda _repo, *_args: f"{reviewed_ref}\n".encode(),
    )
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1),
    )

    assert (
        validate_rulespec_base(repo, country, reviewed_ref, open_pr=False)
        == "reviewed-head-artifact"
    )

    with pytest.raises(ValueError, match="artifact-only"):
        validate_rulespec_base(repo, country, reviewed_ref, open_pr=True)


def test_validate_rulespec_base_accepts_exact_reviewed_protected_branch_tip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "rulespec-us"
    reviewed_ref = "b61918da93fe8a1a29b35b9330aef2085291a5d0"
    git_calls: list[tuple[str, ...]] = []

    def fake_git(_repo: Path, *args: str) -> bytes:
        git_calls.append(args)
        return f"{reviewed_ref}\n".encode()

    monkeypatch.setattr("scripts.prepare_signed_backfill._git", fake_git)
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1),
    )

    assert (
        validate_rulespec_base(
            repo,
            "us",
            reviewed_ref,
            open_pr=True,
            pr_base_branch="hard-cut/canonical-layout-us",
        )
        == "reviewed-head-pr"
    )
    assert (
        "rev-parse",
        "refs/remotes/origin/hard-cut/canonical-layout-us",
    ) in git_calls


def test_validate_rulespec_base_rejects_stale_reviewed_protected_branch_tip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "rulespec-us"
    reviewed_ref = "b61918da93fe8a1a29b35b9330aef2085291a5d0"

    def fake_git(_repo: Path, *args: str) -> bytes:
        if args[-1] == "refs/remotes/origin/hard-cut/canonical-layout-us":
            return f"{'f' * 40}\n".encode()
        return f"{reviewed_ref}\n".encode()

    monkeypatch.setattr("scripts.prepare_signed_backfill._git", fake_git)
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1),
    )

    with pytest.raises(ValueError, match="exact pull request base branch tip"):
        validate_rulespec_base(
            repo,
            "us",
            reviewed_ref,
            open_pr=True,
            pr_base_branch="hard-cut/canonical-layout-us",
        )


def test_validate_rulespec_base_rejects_unapproved_reviewed_pr_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "rulespec-us"
    reviewed_ref = "b61918da93fe8a1a29b35b9330aef2085291a5d0"
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill._git",
        lambda _repo, *_args: f"{reviewed_ref}\n".encode(),
    )
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1),
    )

    with pytest.raises(ValueError, match="artifact-only"):
        validate_rulespec_base(
            repo,
            "us",
            reviewed_ref,
            open_pr=True,
            pr_base_branch="migration/other",
        )


@pytest.mark.parametrize(
    "retired_ref",
    [
        "991f5375b92dffca57b08069093c24a463365cbc",
        "10f7a16ef4a40cf1e26d6273e1aff9ebb79d002f",
        "670e6d6642c70168a4ecfcd7ccfc47c3e7cf51c3",
        "08f7d595d7d4b9ed8565eb90b1e19308fd7aecdd",
        "eb43611e4fdc2d3bdd93f123a2ee5e0b97b2fed0",
    ],
)
def test_validate_rulespec_base_rejects_retired_reviewed_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retired_ref: str,
) -> None:
    repo = tmp_path / "rulespec-us"
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill._git",
        lambda _repo, *_args: f"{retired_ref}\n".encode(),
    )
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1),
    )

    with pytest.raises(ValueError, match="neither on main nor an approved"):
        validate_rulespec_base(repo, "us", retired_ref, open_pr=False)


def test_validate_rulespec_base_rejects_unreviewed_non_main_head(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _add_origin_main(repo)
    _git(repo, "commit", "--allow-empty", "-m", "unreviewed")
    head = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="neither on main nor an approved"):
        validate_rulespec_base(repo, "us", head, open_pr=False)
