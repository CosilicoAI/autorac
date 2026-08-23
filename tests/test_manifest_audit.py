"""Tests for the committed-manifest audit (axiom-encode#1282)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from axiom_encode import manifest_audit
from axiom_encode.manifest_audit import (
    AMBIGUOUS,
    DUPLICATE,
    MISMATCH,
    MISSING,
    RATCHET_INVALID,
    RATCHET_UNUSED,
    RESURRECTED,
    UNATTESTED,
    UNREADABLE,
    UNSAFE,
    RatchetEntry,
    RatchetSchemaError,
    audit_repository,
    load_ratchet,
    plan_prune,
)

MANIFEST_DIR = ".axiom/encoding-manifests"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_rule(repo: Path, relpath: str, content: str) -> str:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return sha256(content)


def write_manifest(
    repo: Path,
    relpath: str,
    applied: list[dict[str, object]],
    *,
    generated_at: str = "2026-06-01T00:00:00+00:00",
) -> str:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "axiom-encode/applied-rulespec/v5",
                "generated_at": generated_at,
                "applied_files": applied,
                "backend": "codex",
            },
            indent=2,
        )
        + "\n"
    )
    return relpath


def write_ratchet(repo: Path, entries: list[dict[str, object]]) -> None:
    path = repo / manifest_audit.RATCHET_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": manifest_audit.RATCHET_SCHEMA,
                manifest_audit.RATCHET_TOP_KEY: entries,
            },
            sort_keys=False,
        )
    )


def entry_for(
    manifest: str, attested_path: str, content: str, **extra: object
) -> dict[str, object]:
    return {
        "manifest": manifest,
        "attested_path": attested_path,
        "attested_sha256": sha256(content),
        "note": "test",
        **extra,
    }


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A country monorepo with one jurisdiction root and both manifest trees."""
    (tmp_path / "us" / "statutes").mkdir(parents=True)
    return tmp_path


def kinds(result) -> list[str]:
    return sorted(finding.kind for finding in result.findings)


# --------------------------------------------------------------------------
# Core invariant: every attestation matches
# --------------------------------------------------------------------------


def test_clean_repo_passes(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    result = audit_repository(repo)
    assert result.passed, result.findings
    assert result.matched == 1
    assert result.manifests == 1


def test_country_relative_key_under_root_tree_resolves(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    assert audit_repository(repo).passed


def test_repo_relative_key_under_root_tree_resolves(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [{"path": "us/statutes/26/32.yaml", "sha256": digest}],
    )
    assert audit_repository(repo).passed


def test_jurisdiction_subtree_root_record_with_country_relative_key(repo: Path) -> None:
    (repo / "us-ak" / "policies").mkdir(parents=True)
    digest = write_rule(repo, "us-ak/policies/atap/standard.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/us-ak/policies/atap/standard.json",
        [{"path": "policies/atap/standard.yaml", "sha256": digest}],
    )
    assert audit_repository(repo).passed


def test_nested_manifest_tree_is_audited(repo: Path) -> None:
    """A record parked at depth >= 2 must be seen, not silently ignored."""
    write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"us/statutes/{MANIFEST_DIR}/26/32.json",
        [{"path": "26/32.yaml", "sha256": sha256("rules: [old]\n")}],
    )
    assert kinds(audit_repository(repo)) == [MISMATCH]


def test_mismatch_is_reported(repo: Path) -> None:
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [changed]\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("rules: []\n")}],
    )
    assert kinds(audit_repository(repo)) == [MISMATCH]


def test_stale_duplicate_is_reported_even_when_a_good_record_exists(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")}],
    )
    result = audit_repository(repo)
    assert MISMATCH in kinds(result)
    assert DUPLICATE in kinds(result)


def test_agreeing_duplicate_still_violates_uniqueness(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    assert kinds(audit_repository(repo)) == [DUPLICATE]


def test_one_manifest_listing_a_path_twice_identically_is_one_claim(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/1401.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/1401.json",
        [
            {"path": "statutes/26/1401.yaml", "sha256": digest},
            {"path": "statutes/26/1401.yaml", "sha256": digest},
        ],
    )
    result = audit_repository(repo)
    assert result.passed
    assert result.attestations == 1


def test_self_contradictory_manifest_fails(repo: Path) -> None:
    """Two different claims about one path in one record is a malformed record."""
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [
            {"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")},
            {"path": "statutes/26/32.yaml", "sha256": digest},
        ],
    )
    assert UNREADABLE in kinds(audit_repository(repo))


def test_missing_file_is_reported(repo: Path) -> None:
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("gone")}],
    )
    assert kinds(audit_repository(repo)) == [MISSING]


def test_deletion_marker_matches_an_absent_file(repo: Path) -> None:
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "deleted": True}],
    )
    result = audit_repository(repo)
    assert result.passed
    assert result.matched == 1


def test_resurrected_file_is_reported(repo: Path) -> None:
    write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "deleted": True}],
    )
    assert kinds(audit_repository(repo)) == [RESURRECTED]


def test_deletion_marker_is_not_satisfied_by_a_symlink(repo: Path) -> None:
    (repo / "us/statutes/26").mkdir(parents=True, exist_ok=True)
    (repo / "us/statutes/26/32.yaml").symlink_to("nowhere.yaml")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "deleted": True}],
    )
    result = audit_repository(repo)
    assert RESURRECTED in kinds(result)
    assert UNSAFE in kinds(result)  # the symlink itself is reported


def test_legacy_root_record_matching_two_jurisdictions_is_ambiguous(repo: Path) -> None:
    (repo / "us-co" / "statutes").mkdir(parents=True)
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [a]\n")
    write_rule(repo, "us-co/statutes/26/32.yaml", "rules: [b]\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("rules: [a]\n")}],
    )
    assert kinds(audit_repository(repo)) == [AMBIGUOUS]


# --------------------------------------------------------------------------
# Hardening
# --------------------------------------------------------------------------


def test_blanked_manifest_fails_rather_than_vanishing(repo: Path) -> None:
    path = repo / MANIFEST_DIR / "statutes" / "26" / "32.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    assert kinds(audit_repository(repo)) == [UNREADABLE]


def test_empty_applied_files_fails(repo: Path) -> None:
    path = repo / MANIFEST_DIR / "statutes" / "26" / "32.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"applied_files": []}))
    assert kinds(audit_repository(repo)) == [UNREADABLE]


def test_duplicate_json_members_fail(repo: Path) -> None:
    """``json.loads`` keeps the last duplicate; that must not erase claims."""
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    path = repo / MANIFEST_DIR / "statutes" / "26" / "32.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"applied_files": [{"path": "statutes/26/32.yaml", "sha256": "%s"}],'
        ' "applied_files": [{"path": "statutes/26/gone.yaml", "deleted": true}]}'
        % sha256("rules: [old]\n")
    )
    assert kinds(audit_repository(repo)) == [UNREADABLE]


def test_unreadable_manifest_is_reported(repo: Path) -> None:
    path = repo / MANIFEST_DIR / "broken.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert kinds(audit_repository(repo)) == [UNREADABLE]


def test_manifest_that_is_a_symlink_fails(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    real = repo / "decoy.json"
    real.write_text(
        json.dumps(
            {"applied_files": [{"path": "statutes/26/32.yaml", "sha256": digest}]}
        )
    )
    link = repo / MANIFEST_DIR / "statutes" / "26" / "32.json"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real)
    assert set(kinds(audit_repository(repo))) == {UNREADABLE, UNSAFE}


def test_dangling_symlink_manifest_fails_rather_than_vanishing(repo: Path) -> None:
    link = repo / MANIFEST_DIR / "statutes" / "26" / "32.json"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to("nowhere.json")
    result = audit_repository(repo)
    assert set(kinds(result)) == {UNREADABLE, UNSAFE}
    assert result.manifests == 1


def test_empty_path_deletion_marker_is_unsafe(repo: Path) -> None:
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "", "deleted": True}],
    )
    assert kinds(audit_repository(repo)) == [UNSAFE]


def test_entry_without_sha256_fails(repo: Path) -> None:
    write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml"}],
    )
    assert kinds(audit_repository(repo)) == [UNREADABLE]


@pytest.mark.parametrize(
    "entry_path",
    [
        "../secrets.yaml",
        "/etc/passwd",
        "statutes/../../x.yaml",
        "a\\b.yaml",
        "./x.yaml",
    ],
)
def test_escaping_or_relative_paths_are_unsafe(repo: Path, entry_path: str) -> None:
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": entry_path, "sha256": sha256("x")}],
    )
    assert kinds(audit_repository(repo)) == [UNSAFE]


def test_symlink_target_is_not_a_match(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/real.yaml", "rules: []\n")
    (repo / "us/statutes/26/32.yaml").symlink_to(repo / "us/statutes/26/real.yaml")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    assert MISSING in kinds(audit_repository(repo))


def test_symlinked_parent_directory_is_not_a_match(repo: Path) -> None:
    """``us/statutes -> ../us-co/statutes`` must not let a US claim hash CO bytes."""
    (repo / "us-co" / "statutes" / "26").mkdir(parents=True)
    digest = write_rule(repo, "us-co/statutes/26/32.yaml", "rules: []\n")
    (repo / "us" / "statutes").rmdir()
    (repo / "us" / "statutes").symlink_to(repo / "us-co" / "statutes")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    assert MISSING in kinds(audit_repository(repo))


def test_jurisdiction_tree_record_never_rebinds_across_jurisdictions(
    repo: Path,
) -> None:
    (repo / "us-co" / "statutes").mkdir(parents=True)
    digest = write_rule(repo, "us-co/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    assert kinds(audit_repository(repo)) == [MISSING]


def test_root_jurisdiction_subtree_record_never_rebinds(repo: Path) -> None:
    (repo / "us-co" / "statutes").mkdir(parents=True)
    (repo / "us-ak" / "statutes").mkdir(parents=True)
    digest = write_rule(repo, "us-co/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/us-ak/statutes/26/32.json",
        [{"path": "us-ak/statutes/26/32.yaml", "sha256": digest}],
    )
    assert kinds(audit_repository(repo)) == [MISSING]


def test_binding_survives_removal_of_the_jurisdiction_directory(tmp_path: Path) -> None:
    """A record's jurisdiction is read from its own path, not today's tree."""
    repo = tmp_path
    (repo / "us-co" / "statutes" / "26").mkdir(parents=True)
    digest = write_rule(repo, "us-co/statutes/26/32.yaml", "rules: []\n")
    # No us/ directory exists at all.
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    assert kinds(audit_repository(repo)) == [MISSING]


def test_case_aliased_records_do_not_split_one_file(repo: Path) -> None:
    """Paths differing only by case are one uniqueness bucket, fail-closed.

    On a case-sensitive filesystem two genuinely distinct files exist and the
    bucket collapses them (DUPLICATE); on a case-insensitive one the second
    write lands on the first file, and the two records are a plain agreeing
    duplicate.  Either way the audit must not pass.
    """
    write_rule(repo, "us/statutes/26/32a.yaml", "rules: [lower]\n")
    write_rule(repo, "us/statutes/26/32A.yaml", "rules: [upper]\n")
    lower = sha256_file_text(repo / "us/statutes/26/32a.yaml")
    upper = sha256_file_text(repo / "us/statutes/26/32A.yaml")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32a.json",
        [{"path": "statutes/26/32a.yaml", "sha256": lower}],
    )
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32A.json",
        [{"path": "statutes/26/32A.yaml", "sha256": upper}],
    )
    result = audit_repository(repo)
    assert kinds(result) == [DUPLICATE]


def sha256_file_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_non_manifest_json_outside_tree_is_ignored(repo: Path) -> None:
    (repo / ".axiom" / "index").mkdir(parents=True, exist_ok=True)
    (repo / ".axiom" / "index" / "reverse.json").write_text(json.dumps({"a": 1}))
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    assert audit_repository(repo).passed


# --------------------------------------------------------------------------
# Retired-claims ratchet
# --------------------------------------------------------------------------


def _stale_with_successor(repo: Path) -> tuple[str, str]:
    """A stale claim on a record kept alive by its companion-test coverage."""
    current = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    write_rule(repo, "us/statutes/26/32.test.yaml", "cases: []\n")
    stale = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [
            {"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")},
            {"path": "statutes/26/32.test.yaml", "sha256": sha256("cases: []\n")},
        ],
    )
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": current}],
    )
    return stale, "statutes/26/32.yaml"


def test_retired_stale_claim_with_successor_passes(repo: Path) -> None:
    stale, attested_path = _stale_with_successor(repo)
    write_ratchet(repo, [entry_for(stale, attested_path, "rules: [old]\n")])
    result = audit_repository(repo)
    assert result.passed, [str(f) for f in result.findings]
    assert result.retired == 1


def test_retired_true_claim_with_successor_passes_and_is_not_a_duplicate(
    repo: Path,
) -> None:
    """A byte-identical re-encode retires the old claim in place (sol S2/S8)."""
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [same]\n")
    write_rule(repo, "us/statutes/26/32.test.yaml", "cases: []\n")
    old = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [
            {"path": "statutes/26/32.yaml", "sha256": digest},
            {"path": "statutes/26/32.test.yaml", "sha256": sha256("cases: []\n")},
        ],
    )
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    assert kinds(audit_repository(repo)) == [DUPLICATE]
    write_ratchet(repo, [entry_for(old, "statutes/26/32.yaml", "rules: [same]\n")])
    result = audit_repository(repo)
    assert result.passed, [str(f) for f in result.findings]


def test_retiring_a_true_sole_claim_is_invalid(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    only = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    write_ratchet(repo, [entry_for(only, "statutes/26/32.yaml", "rules: []\n")])
    assert kinds(audit_repository(repo)) == [RATCHET_INVALID]


def test_entry_with_wrong_sha_does_not_apply(repo: Path) -> None:
    stale, attested_path = _stale_with_successor(repo)
    write_ratchet(repo, [entry_for(stale, attested_path, "something else")])
    result = audit_repository(repo)
    assert MISMATCH in kinds(result)
    assert RATCHET_UNUSED in kinds(result)


def test_unattested_gap_must_be_declared(repo: Path) -> None:
    (repo / "gh" / "policies").mkdir(parents=True, exist_ok=True)
    write_rule(repo, "gh/policies/free-shs/fee.yaml", "rules: [migrated]\n")
    only = write_manifest(
        repo,
        f"{MANIFEST_DIR}/gh/policies/free-shs/fee.json",
        [{"path": "gh/policies/free-shs/fee.yaml", "sha256": sha256("rules: [old]\n")}],
    )
    entry = entry_for(only, "gh/policies/free-shs/fee.yaml", "rules: [old]\n")
    write_ratchet(repo, [entry])
    assert UNATTESTED in kinds(audit_repository(repo))
    write_ratchet(repo, [{**entry, "unattested": True}])
    result = audit_repository(repo)
    assert result.passed, [str(f) for f in result.findings]
    assert result.retired == 1


def test_stale_unattested_flag_is_rejected(repo: Path) -> None:
    stale, attested_path = _stale_with_successor(repo)
    write_ratchet(
        repo, [entry_for(stale, attested_path, "rules: [old]\n", unattested=True)]
    )
    assert RATCHET_INVALID in kinds(audit_repository(repo))


def test_ratchet_retires_a_missing_file_claim(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    moved = write_manifest(
        repo,
        f"{MANIFEST_DIR}/programs/us/income-tax/fy-2026.json",
        [{"path": "programs/us/income-tax/fy-2026.yaml", "sha256": sha256("spec\n")}],
    )
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    write_ratchet(
        repo,
        [
            entry_for(
                moved, "programs/us/income-tax/fy-2026.yaml", "spec\n", unattested=True
            )
        ],
    )
    result = audit_repository(repo)
    assert result.passed, [str(f) for f in result.findings]


def test_unused_entry_fails(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    write_ratchet(repo, [entry_for("gone.json", "statutes/26/32.yaml", "x")])
    assert kinds(audit_repository(repo)) == [RATCHET_UNUSED]


def test_absent_ratchet_is_not_an_error(repo: Path) -> None:
    assert load_ratchet(repo) == {}


def test_symlinked_ratchet_is_rejected(repo: Path) -> None:
    (repo / "elsewhere.yaml").write_text("schema_version: x\n")
    target = repo / manifest_audit.RATCHET_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(repo / "elsewhere.yaml")
    result = audit_repository(repo)
    assert RATCHET_INVALID in kinds(result)
    assert UNSAFE in kinds(result)


@pytest.mark.parametrize(
    "entry, expected",
    [
        ({"manifest": "m.json"}, "missing keys"),
        (
            {
                "manifest": "m.json",
                "attested_path": "a.yaml",
                "attested_sha256": "nothex",
                "note": "n",
            },
            "sha256 hex digest",
        ),
        (
            {
                "manifest": "m.json",
                "attested_path": "a.yaml",
                "attested_sha256": "0" * 64,
                "note": "n",
                "surprise": 1,
            },
            "unknown keys",
        ),
        (
            {
                "manifest": "m.json",
                "attested_path": "a.yaml",
                "attested_sha256": "0" * 64,
                "note": "n",
                "unattested": "yes",
            },
            "must be a boolean",
        ),
        (
            {
                "manifest": "m.json",
                "attested_path": "a.yaml",
                "attested_sha256": "0" * 64,
                "note": "   ",
            },
            "non-empty string",
        ),
    ],
)
def test_ratchet_schema_is_strict(repo: Path, entry: dict, expected: str) -> None:
    write_ratchet(repo, [entry])
    with pytest.raises(RatchetSchemaError, match=expected):
        load_ratchet(repo)


def test_ratchet_rejects_duplicate_entries(repo: Path) -> None:
    entry = {
        "manifest": "m.json",
        "attested_path": "a.yaml",
        "attested_sha256": "0" * 64,
        "note": "n",
    }
    write_ratchet(repo, [entry, dict(entry)])
    with pytest.raises(RatchetSchemaError, match="duplicate entry"):
        load_ratchet(repo)


def test_render_round_trips_numeric_looking_digests(repo: Path) -> None:
    entry = RatchetEntry(
        manifest="m.json", attested_path="a.yaml", attested_sha256="0" * 64, note="n"
    )
    manifest_audit.write_ratchet(repo, [entry])
    assert load_ratchet(repo)[entry.key].attested_sha256 == "0" * 64


# --------------------------------------------------------------------------
# Prune-on-supersede
# --------------------------------------------------------------------------


def test_plan_prune_retires_a_fully_superseded_record_pinned_to_read_bytes(
    repo: Path,
) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    old = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")}],
    )
    new = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    plan = plan_prune(repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"])
    assert dict(plan.retire) == {
        old: hashlib.sha256((repo / old).read_bytes()).hexdigest()
    }
    assert plan.disclose == ()


def test_plan_prune_retires_partly_superseded_claims_in_place(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    write_rule(repo, "us/statutes/26/32.test.yaml", "cases: []\n")
    old = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [
            {"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")},
            {"path": "statutes/26/32.test.yaml", "sha256": sha256("cases: []\n")},
        ],
    )
    new = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    plan = plan_prune(repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"])
    assert plan.retire == {}
    assert [(e.manifest, e.attested_path) for e in plan.disclose] == [
        (old, "statutes/26/32.yaml")
    ]


def test_plan_prune_retires_a_still_true_overlap_claim_too(repo: Path) -> None:
    """Supersession, not staleness, decides retirement (sol S2)."""
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [same]\n")
    write_rule(repo, "us/statutes/26/32.test.yaml", "cases: []\n")
    old = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [
            {"path": "statutes/26/32.yaml", "sha256": digest},
            {"path": "statutes/26/32.test.yaml", "sha256": sha256("cases: []\n")},
        ],
    )
    new = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    plan = plan_prune(repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"])
    assert [e.manifest for e in plan.disclose] == [old]


@pytest.mark.parametrize(
    "extra_claim",
    [
        {"path": "statutes/26/32-repealed.yaml", "deleted": True},
        {"path": "statutes/26/vanished.yaml", "sha256": "ab" * 32},
        {"path": "../escape.yaml", "sha256": "ab" * 32},
    ],
)
def test_plan_prune_never_retires_a_record_with_an_uncovered_claim(
    repo: Path, extra_claim: dict
) -> None:
    """Deletion markers, unresolved and unsafe claims all block retirement."""
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    old = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [
            {"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")},
            extra_claim,
        ],
    )
    new = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    plan = plan_prune(repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"])
    assert plan.retire == {}
    assert [e.attested_path for e in plan.disclose] == ["statutes/26/32.yaml"]
    assert (repo / old).is_file()


def test_plan_prune_never_retires_a_structurally_broken_record(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    broken = repo / MANIFEST_DIR / "statutes" / "26" / "32.json"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text(
        json.dumps(
            {
                "applied_files": [
                    {"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")},
                    "garbage",
                ]
            }
        )
    )
    new = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    plan = plan_prune(repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"])
    assert plan.retire == {}


def test_plan_prune_drops_ratchet_rows_of_retired_records(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    old = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")}],
    )
    write_ratchet(repo, [entry_for(old, "statutes/26/32.yaml", "rules: [old]\n")])
    new = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    plan = plan_prune(repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"])
    assert list(plan.retire) == [old]
    assert plan.drop_keys == ((old, "statutes/26/32.yaml"),)
    manifest_audit.apply_prune(
        repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"]
    )
    assert not (repo / manifest_audit.RATCHET_RELATIVE_PATH).exists()
    assert audit_repository(repo).passed


def test_plan_prune_ignores_unrelated_records(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    other = write_rule(repo, "us/statutes/26/24.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/24.json",
        [{"path": "statutes/26/24.yaml", "sha256": other}],
    )
    new = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    assert plan_prune(
        repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"]
    ).empty


def test_apply_prune_deletes_and_retires(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    write_rule(repo, "us/statutes/26/32.test.yaml", "cases: []\n")
    fully = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")}],
    )
    partly = write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [
            {"path": "us/statutes/26/32.yaml", "sha256": sha256("rules: [older]\n")},
            {"path": "us/statutes/26/32.test.yaml", "sha256": sha256("cases: []\n")},
        ],
    )
    new = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    plan = manifest_audit.apply_prune(
        repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"]
    )
    assert list(plan.retire) == [fully]
    assert not (repo / fully).exists()
    assert (repo / partly).exists()
    assert list(load_ratchet(repo)) == [(partly, "us/statutes/26/32.yaml")]
    final = audit_repository(repo)
    assert final.passed, [str(f) for f in final.findings]


def test_ratchet_update_keeps_existing_entries_on_collision(repo: Path) -> None:
    existing = RatchetEntry(
        manifest="m.json",
        attested_path="a.yaml",
        attested_sha256="0" * 64,
        note="reviewed note",
    )
    manifest_audit.write_ratchet(repo, [existing])
    incoming = RatchetEntry(
        manifest="m.json",
        attested_path="a.yaml",
        attested_sha256="f" * 64,
        note="machine note",
    )
    update = manifest_audit.ratchet_update(repo, additions=[incoming])
    assert update is None  # nothing changes: key already present
    update = manifest_audit.ratchet_update(repo, additions=[], drop_keys=[existing.key])
    assert update is not None and update.delete
    assert (
        update.expected_sha256
        == hashlib.sha256(
            (repo / manifest_audit.RATCHET_RELATIVE_PATH).read_bytes()
        ).hexdigest()
    )


# --------------------------------------------------------------------------
# Production apply-path glue and transaction
# --------------------------------------------------------------------------


def _monorepo(tmp_path: Path) -> Path:
    """A checkout named like a country monorepo, as the allowlist requires."""
    repo = tmp_path / "rulespec-us"
    (repo / "us" / "statutes" / "26").mkdir(parents=True)
    return repo


def test_apply_prune_transaction_plans_against_the_live_checkout(
    tmp_path: Path,
) -> None:
    from axiom_encode import cli

    repo = _monorepo(tmp_path)
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    write_rule(repo, "us/statutes/26/32.test.yaml", "cases: []\n")
    fully = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")}],
    )
    partly = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [
            {"path": "statutes/26/32.yaml", "sha256": sha256("rules: [older]\n")},
            {"path": "statutes/26/32.test.yaml", "sha256": sha256("cases: []\n")},
        ],
    )
    prune = cli._plan_apply_prune_transaction(
        checkout_root=repo,
        content_root=repo / "us",
        manifest_relative=Path(f"{MANIFEST_DIR}/us/statutes/26/32.json"),
        planned={"statutes/26/32.yaml": b"rules: [new]\n"},
    )
    targets = dict(prune.files)
    assert targets[repo / fully] is None
    ratchet_target = repo / manifest_audit.RATCHET_RELATIVE_PATH
    assert b"statutes/26/32.yaml" in targets[ratchet_target]
    assert (
        prune.expected[repo / fully]
        == hashlib.sha256((repo / fully).read_bytes()).hexdigest()
    )
    assert prune.expected[ratchet_target] is None
    assert (repo / partly) not in targets


def test_apply_prune_transaction_is_empty_without_predecessors(tmp_path: Path) -> None:
    from axiom_encode import cli

    repo = _monorepo(tmp_path)
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    prune = cli._plan_apply_prune_transaction(
        checkout_root=repo,
        content_root=repo / "us",
        manifest_relative=Path(f"{MANIFEST_DIR}/us/statutes/26/32.json"),
        planned={"statutes/26/32.yaml": b"rules: [new]\n"},
    )
    assert prune.files == [] and prune.expected == {}


def test_install_transaction_accepts_prune_targets_end_to_end(tmp_path: Path) -> None:
    """The allowlist admits dead-layout deletions and the ratchet write (sol S1)."""
    from axiom_encode import cli

    repo = _monorepo(tmp_path)
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [old]\n")
    write_rule(repo, "us/statutes/26/32.test.yaml", "cases: []\n")
    juris_tree = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")}],
    )
    legacy_root = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [
            {"path": "statutes/26/32.yaml", "sha256": sha256("rules: [older]\n")},
            {"path": "statutes/26/32.test.yaml", "sha256": sha256("cases: []\n")},
        ],
    )
    new_manifest_relative = Path(f"{MANIFEST_DIR}/us/statutes/26/32.json")
    new_bytes = (
        json.dumps(
            {
                "applied_files": [
                    {
                        "path": "us/statutes/26/32.yaml",
                        "sha256": sha256("rules: [new]\n"),
                    }
                ]
            }
        ).encode()
        + b"\n"
    )
    prune = cli._plan_apply_prune_transaction(
        checkout_root=repo,
        content_root=repo / "us",
        manifest_relative=new_manifest_relative,
        planned={"statutes/26/32.yaml": b"rules: [new]\n"},
    )
    files = [
        (repo / "us/statutes/26/32.yaml", b"rules: [new]\n"),
        (repo / new_manifest_relative, new_bytes),
        *prune.files,
    ]
    expected = {
        repo / "us/statutes/26/32.yaml": hashlib.sha256(b"rules: [old]\n").hexdigest(),
        repo / new_manifest_relative: None,
        **prune.expected,
    }
    cli._install_apply_transaction(
        files, checkout_root=repo, expected_originals=expected
    )
    assert not (repo / juris_tree).exists()
    assert (repo / legacy_root).exists()
    assert (repo / manifest_audit.RATCHET_RELATIVE_PATH).is_file()
    final = audit_repository(repo)
    assert final.passed, [str(f) for f in final.findings]


def test_install_transaction_still_rejects_writes_to_dead_layouts(
    tmp_path: Path,
) -> None:
    from axiom_encode import cli

    repo = _monorepo(tmp_path)
    target = repo / f"us/{MANIFEST_DIR}/statutes/26/32.json"
    with pytest.raises(RuntimeError, match="outside the canonical"):
        cli._install_apply_transaction([(target, b"{}")], checkout_root=repo)


def test_install_transaction_cas_rejects_a_record_changed_after_planning(
    tmp_path: Path,
) -> None:
    """A record that grew after the plan read it must not be deleted (sol S3)."""
    from axiom_encode import cli

    repo = _monorepo(tmp_path)
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    old = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")}],
    )
    prune = cli._plan_apply_prune_transaction(
        checkout_root=repo,
        content_root=repo / "us",
        manifest_relative=Path(f"{MANIFEST_DIR}/us/statutes/26/32.json"),
        planned={"statutes/26/32.yaml": b"rules: [new]\n"},
    )
    # Another sanctioned writer expands the record before the locked install.
    write_manifest(
        repo,
        old,
        [
            {"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")},
            {"path": "statutes/26/33.yaml", "sha256": sha256("other\n")},
        ],
    )
    with pytest.raises(RuntimeError, match="changed after validation"):
        cli._install_apply_transaction(
            prune.files, checkout_root=repo, expected_originals=prune.expected
        )
    assert (repo / old).exists()


def test_post_install_closure_models_prune_effects(tmp_path: Path) -> None:
    from axiom_encode import cli

    content_root = tmp_path / "rulespec-us" / "us"
    content_root.mkdir(parents=True)
    expected = {
        "statutes/26/32.yaml": "a" * 64,
        ".axiom/encoding-manifests/statutes/26/32.json": "b" * 64,
    }
    cli._model_prune_in_expected_post_files(
        expected,
        content_root=content_root,
        pruned_paths=[content_root / ".axiom/encoding-manifests/statutes/26/32.json"],
        ratchet_write=(
            content_root.parent / ".axiom/retired-manifest-claims.yaml",
            b"x",
        ),
    )
    assert expected == {"statutes/26/32.yaml": "a" * 64}
    # Flat checkout: the ratchet lives under the content root and is modeled.
    flat = {}
    cli._model_prune_in_expected_post_files(
        flat,
        content_root=content_root,
        pruned_paths=[],
        ratchet_write=(content_root / ".axiom/retired-manifest-claims.yaml", b"x"),
    )
    assert flat == {
        ".axiom/retired-manifest-claims.yaml": hashlib.sha256(b"x").hexdigest()
    }


# --------------------------------------------------------------------------
# Round-3 regressions
# --------------------------------------------------------------------------


def test_symlinked_manifest_tree_ancestor_is_a_finding(repo: Path) -> None:
    """A linked ``.axiom`` hides a tree from os.walk; the link itself must fail."""
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    store = repo / "us" / "manifest-store" / "encoding-manifests" / "statutes" / "26"
    store.mkdir(parents=True)
    (store / "32.json").write_text(
        json.dumps(
            {
                "applied_files": [
                    {"path": "statutes/26/32.yaml", "sha256": sha256("old")}
                ]
            }
        )
    )
    (repo / "us" / ".axiom").symlink_to(repo / "us" / "manifest-store")
    result = audit_repository(repo)
    assert UNSAFE in kinds(result)
    assert any("us/.axiom" in (f.path or "") for f in result.findings)


def test_any_symlink_in_checkout_is_a_finding(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    (repo / "README.md").symlink_to(repo / "us/statutes/26/32.yaml")
    assert kinds(audit_repository(repo)) == [UNSAFE]


def test_record_parked_under_a_junk_directory_is_still_audited(repo: Path) -> None:
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/node_modules/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("old")}],
    )
    assert kinds(audit_repository(repo)) == [MISMATCH]


def test_aliased_keys_cannot_contradict_within_one_record(repo: Path) -> None:
    """``statutes/f.yaml`` and ``us/statutes/f.yaml`` resolve to one file."""
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [
            {"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")},
            {"path": "us/statutes/26/32.yaml", "sha256": digest},
        ],
    )
    assert UNREADABLE in kinds(audit_repository(repo))


def test_a_record_cannot_be_its_own_other_live_attestation(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [same]\n")
    only = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [
            {"path": "statutes/26/32.yaml", "sha256": digest},
            {"path": "us/statutes/26/32.yaml", "sha256": digest},
        ],
    )
    write_ratchet(repo, [entry_for(only, "statutes/26/32.yaml", "rules: [same]\n")])
    result = audit_repository(repo)
    # Identical aliased claims collapse to one; retiring it leaves no live owner.
    assert RATCHET_INVALID in kinds(result) or RATCHET_UNUSED in kinds(result)


def test_plan_prune_respects_the_retirable_predicate(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    deep = write_manifest(
        repo,
        f"us/statutes/{MANIFEST_DIR}/26/32.json",
        [{"path": "26/32.yaml", "sha256": sha256("old")}],
    )
    new = write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [{"path": "us/statutes/26/32.yaml", "sha256": digest}],
    )
    plan = plan_prune(
        repo,
        new_manifest=new,
        attested_paths=["us/statutes/26/32.yaml"],
        retirable=lambda relative: False,
    )
    assert plan.retire == {}
    assert [e.manifest for e in plan.disclose] == [deep]


def test_plan_prune_drops_rows_of_an_overwritten_destination(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    dest = f"{MANIFEST_DIR}/us/statutes/26/32.json"
    write_ratchet(
        repo, [entry_for(dest, "us/statutes/26/32.yaml", "rules: [ancient]\n")]
    )
    write_manifest(repo, dest, [{"path": "us/statutes/26/32.yaml", "sha256": digest}])
    plan = plan_prune(
        repo, new_manifest=dest, attested_paths=["us/statutes/26/32.yaml"]
    )
    assert plan.drop_keys == ((dest, "us/statutes/26/32.yaml"),)


def test_successor_install_closes_a_declared_gap(repo: Path) -> None:
    """An ``unattested: true`` row flips off when a successor arrives (sol #5)."""
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    write_rule(repo, "us/statutes/26/32.test.yaml", "cases: []\n")
    old = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [
            {"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")},
            {"path": "statutes/26/32.test.yaml", "sha256": sha256("cases: []\n")},
        ],
    )
    write_ratchet(
        repo, [entry_for(old, "statutes/26/32.yaml", "rules: [old]\n", unattested=True)]
    )
    assert audit_repository(repo).passed
    new = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    assert RATCHET_INVALID in kinds(audit_repository(repo))  # flag now stale
    manifest_audit.apply_prune(
        repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"]
    )
    entry = load_ratchet(repo)[(old, "statutes/26/32.yaml")]
    assert entry.unattested is False
    assert entry.note == "test"  # reviewed note preserved
    assert audit_repository(repo).passed


def test_locked_replan_detects_inventory_changes(tmp_path: Path) -> None:
    """The plan computed outside the lock must match the locked tree (sol #2)."""
    from axiom_encode import cli

    repo = _monorepo(tmp_path)
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    write_rule(repo, "us/statutes/26/33.yaml", "rules: [b]\n")
    old = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [
            {"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")},
            {"path": "statutes/26/33.yaml", "sha256": sha256("rules: [b]\n")},
        ],
    )
    kwargs = dict(
        checkout_root=repo,
        content_root=repo / "us",
        manifest_relative=Path(f"{MANIFEST_DIR}/us/statutes/26/32.json"),
        planned={
            "statutes/26/32.yaml": b"rules: [new]\n",
            "statutes/26/33.yaml": b"rules: [b]\n",
        },
    )
    provisional = cli._plan_apply_prune_transaction(**kwargs)
    assert list(provisional.files) and (repo / old) in dict(provisional.files)
    # A concurrent apply lands a distinct successor for 33 and retires old/33.
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/33.json",
        [{"path": "us/statutes/26/33.yaml", "sha256": sha256("rules: [b]\n")}],
    )
    write_ratchet(repo, [entry_for(old, "statutes/26/33.yaml", "rules: [b]\n")])
    locked = cli._plan_apply_prune_transaction(**kwargs)
    assert not locked.same_as(provisional)


# --------------------------------------------------------------------------
# Round-3 regressions (fable)
# --------------------------------------------------------------------------


def test_deletion_marker_on_a_recreated_path_is_retirable(repo: Path) -> None:
    """Re-encoding a retired rule must not leave a permanent RESURRECTED."""
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [back]\n")
    write_rule(repo, "us/statutes/26/33.yaml", "rules: [b]\n")
    retire_record = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [
            {"path": "statutes/26/32.yaml", "deleted": True},
            {"path": "statutes/26/33.yaml", "sha256": sha256("rules: [b]\n")},
        ],
    )
    new = write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [{"path": "us/statutes/26/32.yaml", "sha256": digest}],
    )
    assert kinds(audit_repository(repo)) == [RESURRECTED]
    plan = plan_prune(repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"])
    assert plan.retire == {}  # the record still protects 33
    assert [(e.attested_path, e.deletion) for e in plan.disclose] == [
        ("statutes/26/32.yaml", True)
    ]
    manifest_audit.apply_prune(
        repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"]
    )
    result = audit_repository(repo)
    assert result.passed, [str(f) for f in result.findings]
    assert result.retired == 1
    # Without a live record for the re-created file the retirement is invalid.
    (repo / new).unlink()
    assert RATCHET_INVALID in kinds(audit_repository(repo))
    assert (repo / retire_record).exists()


def test_fully_superseded_retire_record_is_deleted(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [back]\n")
    old = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "deleted": True}],
    )
    new = write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [{"path": "us/statutes/26/32.yaml", "sha256": digest}],
    )
    plan = plan_prune(repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"])
    assert list(plan.retire) == [old]


def test_deletion_entry_schema(repo: Path) -> None:
    write_ratchet(
        repo,
        [
            {
                "manifest": "m.json",
                "attested_path": "a.yaml",
                "note": "n",
                "deletion": True,
            }
        ],
    )
    entry = load_ratchet(repo)[("m.json", "a.yaml")]
    assert entry.deletion and entry.attested_sha256 is None
    write_ratchet(
        repo,
        [
            {
                "manifest": "m.json",
                "attested_path": "a.yaml",
                "note": "n",
                "deletion": True,
                "attested_sha256": "0" * 64,
            }
        ],
    )
    with pytest.raises(RatchetSchemaError, match="exactly one of"):
        load_ratchet(repo)
    write_ratchet(
        repo,
        [
            {
                "manifest": "m.json",
                "attested_path": "a.yaml",
                "note": "n",
                "deletion": True,
                "unattested": True,
            }
        ],
    )
    with pytest.raises(RatchetSchemaError, match="cannot be unattested"):
        load_ratchet(repo)


def test_malformed_digest_is_a_structural_finding(repo: Path) -> None:
    """A bad digest must never reach the ratchet and poison it (fable #4)."""
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": "ABCDEF"}],
    )
    assert kinds(audit_repository(repo)) == [UNREADABLE]


def test_non_ascii_paths_round_trip_through_the_ratchet(repo: Path) -> None:
    name = "us/statutes/42/1437c\u2013\U0001f600.yaml"
    digest = write_rule(repo, name, "rules: [new]\n")
    write_rule(repo, "us/statutes/42/sib.yaml", "rules: [s]\n")
    old = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/42/x.json",
        [
            {"path": "statutes/42/1437c\u2013\U0001f600.yaml", "sha256": sha256("old")},
            {"path": "statutes/42/sib.yaml", "sha256": sha256("rules: [s]\n")},
        ],
    )
    new = write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/42/x.json",
        [{"path": name, "sha256": digest}],
    )
    manifest_audit.apply_prune(repo, new_manifest=new, attested_paths=[name])
    assert (old, "statutes/42/1437c\u2013\U0001f600.yaml") in load_ratchet(repo)
    assert audit_repository(repo).passed


# --------------------------------------------------------------------------
# Round-4 regressions (sol)
# --------------------------------------------------------------------------


def test_legacy_deletion_marker_contradicted_in_two_jurisdictions_is_ambiguous(
    repo: Path,
) -> None:
    (repo / "us-co" / "statutes").mkdir(parents=True)
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [a]\n")
    write_rule(repo, "us-co/statutes/26/32.yaml", "rules: [b]\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "deleted": True}],
    )
    assert kinds(audit_repository(repo)) == [AMBIGUOUS]


def test_contradicted_marker_is_superseded_only_at_its_own_path(repo: Path) -> None:
    """A marker resurrected in us-co is not covered by an apply that writes us/."""
    (repo / "us-co" / "statutes").mkdir(parents=True)
    write_rule(repo, "us-co/statutes/26/32.yaml", "rules: [b]\n")
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [a]\n")
    (repo / "us/statutes/26/32.yaml").unlink()
    marker = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "deleted": True}],
    )
    new = write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [{"path": "us/statutes/26/32.yaml", "sha256": digest}],
    )
    plan = plan_prune(repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"])
    assert plan.retire == {} and plan.disclose == ()
    assert (repo / marker).exists()


def test_case_variant_manifest_tree_is_audited(tmp_path: Path) -> None:
    repo = tmp_path
    (repo / "us" / "statutes").mkdir(parents=True)
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    tree = repo / ".AXIOM" / "ENCODING-MANIFESTS" / "us" / "statutes" / "26"
    tree.mkdir(parents=True)
    (tree / "32.JSON").write_text(
        json.dumps(
            {
                "applied_files": [
                    {"path": "us/statutes/26/32.yaml", "sha256": sha256("old")}
                ]
            }
        )
    )
    assert MISMATCH in kinds(audit_repository(repo))


def test_case_aliased_keys_cannot_contradict_within_one_record(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32a.yaml", "rules: [new]\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32a.json",
        [
            {"path": "statutes/26/32a.yaml", "sha256": sha256("old")},
            {"path": "statutes/26/32A.yaml", "sha256": digest},
        ],
    )
    result = audit_repository(repo)
    assert not result.passed
    assert set(kinds(result)) <= {UNREADABLE, MISSING, MISMATCH}


def test_apply_prune_transaction_skips_a_deletion_that_is_the_written_file(
    tmp_path: Path,
) -> None:
    """On a case-insensitive filesystem ``C.json`` IS ``c.json`` (sol #1)."""
    import os

    from axiom_encode import cli

    repo = tmp_path / "rulespec-us"
    (repo / "us" / "statutes" / "26").mkdir(parents=True)
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    old = write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/C.json",
        [{"path": "us/statutes/26/32.yaml", "sha256": sha256("old")}],
    )
    new_target = repo / f"{MANIFEST_DIR}/us/statutes/26/c.json"
    prune = cli._plan_apply_prune_transaction(
        checkout_root=repo,
        content_root=repo / "us",
        manifest_relative=Path(f"{MANIFEST_DIR}/us/statutes/26/c.json"),
        planned={"statutes/26/32.yaml": b"rules: [new]\n"},
        written_targets=[new_target],
    )
    case_insensitive = new_target.exists()
    deletions = [t for t, raw in prune.files if raw is None]
    if case_insensitive:
        assert deletions == [], "the write replaces the same file; no separate deletion"
    else:
        assert deletions == [repo / old]
    assert os.path.samefile is not None  # keep the import honest


def test_apply_prune_transaction_opens_gaps_for_paths_it_deletes(
    tmp_path: Path,
) -> None:
    """A legacy move that deletes a file flips that file's rows to unattested (sol #2)."""
    from axiom_encode import cli

    repo = tmp_path / "rulespec-us"
    (repo / "us" / "statutes" / "26").mkdir(parents=True)
    current = write_rule(repo, "us/statutes/26/old.yaml", "rules: [v2]\n")
    write_rule(repo, "us/statutes/26/sib.yaml", "rules: [s]\n")
    historical = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/old.json",
        [
            {"path": "statutes/26/old.yaml", "sha256": sha256("v1")},
            {"path": "statutes/26/sib.yaml", "sha256": sha256("rules: [s]\n")},
        ],
    )
    successor = write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/old.json",
        [{"path": "us/statutes/26/old.yaml", "sha256": current}],
    )
    write_ratchet(repo, [entry_for(historical, "statutes/26/old.yaml", "v1")])
    assert audit_repository(repo).passed
    prune = cli._plan_apply_prune_transaction(
        checkout_root=repo,
        content_root=repo / "us",
        manifest_relative=Path(f"{MANIFEST_DIR}/us/statutes/26/new.json"),
        planned={"statutes/26/new.yaml": b"rules: [v2]\n"},
        deleted_paths={"us/statutes/26/old.yaml"},
        deleted_manifests={successor},
    )
    raw = dict(prune.files)[repo / manifest_audit.RATCHET_RELATIVE_PATH]
    parsed = manifest_audit.parse_ratchet(raw)
    assert parsed[(historical, "statutes/26/old.yaml")].unattested is True


def test_canonical_record_may_attest_checkout_root_metadata(repo: Path) -> None:
    """Legacy-replacement records attest the metadata they reconciled (real layout)."""
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    (repo / ".axiom").mkdir(exist_ok=True)
    (repo / ".axiom" / "toolchain.toml").write_text("[toolchain]\n")
    (repo / "known-validation-gaps.yaml").write_text("validate_failures: {}\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [
            {"path": "us/statutes/26/32.yaml", "sha256": digest},
            {"path": ".axiom/toolchain.toml", "sha256": sha256("[toolchain]\n")},
            {
                "path": "known-validation-gaps.yaml",
                "sha256": sha256("validate_failures: {}\n"),
            },
        ],
    )
    result = audit_repository(repo)
    assert result.passed, [str(f) for f in result.findings]


def test_metadata_allowance_does_not_reopen_cross_jurisdiction_binding(
    repo: Path,
) -> None:
    (repo / "us-co" / "statutes").mkdir(parents=True)
    digest = write_rule(repo, "us-co/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [{"path": "us-co/statutes/26/32.yaml", "sha256": digest}],
    )
    assert kinds(audit_repository(repo)) == [MISSING]


# --------------------------------------------------------------------------
# Round-5 regressions (sol)
# --------------------------------------------------------------------------


def test_tree_prefix_is_component_wise_and_length_safe() -> None:
    """Casefolding can change string length (ß → ss); never slice by it."""
    assert (
        manifest_audit._tree_prefix("us/statutes/ß/.AXIOM/ENCODING-MANIFESTS/x.json")
        == "us/statutes/ß"
    )
    assert manifest_audit._tree_prefix(".axiom/encoding-manifests/us/x.json") == ""
    assert (
        manifest_audit._manifest_subpath_head(
            ".AXIOM/encoding-manifests/us-ak/p/x.json"
        )
        == "us-ak"
    )


def test_case_variant_files_in_one_record_are_not_collapsed_blindly(repo: Path) -> None:
    """On a case-sensitive filesystem F.yaml and f.yaml are two claims."""
    upper = write_rule(repo, "us/statutes/26/F.yaml", "rules: [upper]\n")
    write_rule(repo, "us/statutes/26/f.yaml", "rules: [lower]\n")
    same_file = (
        os.stat(repo / "us/statutes/26/F.yaml").st_ino
        == os.stat(repo / "us/statutes/26/f.yaml").st_ino
    )
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/F.json",
        [
            {
                "path": "statutes/26/F.yaml",
                "sha256": sha256_file_text(repo / "us/statutes/26/F.yaml"),
            },
            {
                "path": "statutes/26/f.yaml",
                "sha256": upper,
            },  # stale on a case-sensitive FS
        ],
    )
    result = audit_repository(repo)
    if same_file:
        # One inode, two different digests: a self-contradictory record.
        assert UNREADABLE in kinds(result)
    else:
        assert MISMATCH in kinds(result)


def test_metadata_claim_cannot_be_captured_by_a_jurisdiction_decoy(repo: Path) -> None:
    """A head-bound record's `.axiom/...` key denotes the checkout root only."""
    (repo / ".axiom" / "index").mkdir(parents=True, exist_ok=True)
    (repo / ".axiom" / "index" / "provisions.json").write_text("real-v2\n")
    decoy = repo / "us" / ".axiom" / "index" / "provisions.json"
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_text("real-v1\n")
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [
            {"path": "us/statutes/26/32.yaml", "sha256": digest},
            {"path": ".axiom/index/provisions.json", "sha256": sha256("real-v1\n")},
        ],
    )
    assert MISMATCH in kinds(audit_repository(repo))


def test_metadata_deletion_marker_uses_the_same_base(repo: Path) -> None:
    (repo / ".axiom" / "index").mkdir(parents=True, exist_ok=True)
    (repo / ".axiom" / "index" / "provisions.json").write_text("x\n")
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [
            {"path": "us/statutes/26/32.yaml", "sha256": digest},
            {"path": ".axiom/index/provisions.json", "deleted": True},
        ],
    )
    assert RESURRECTED in kinds(audit_repository(repo))


def test_programs_key_is_ambiguous_when_both_layouts_exist(repo: Path) -> None:
    write_rule(repo, "programs/us-nh/income-tax/fy.yaml", "spec: root\n")
    write_rule(repo, "us/programs/us-nh/income-tax/fy.yaml", "spec: juris\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/programs/us-nh/income-tax/fy.json",
        [
            {
                "path": "programs/us-nh/income-tax/fy.yaml",
                "sha256": sha256("spec: root\n"),
            }
        ],
    )
    assert kinds(audit_repository(repo)) == [AMBIGUOUS]
    (repo / "us/programs/us-nh/income-tax/fy.yaml").unlink()
    assert audit_repository(repo).passed


def test_backstop_treats_grown_findings_as_new() -> None:
    from axiom_encode import cli

    key = ("duplicate", None, "us/statutes/26/32.yaml")
    before = {key: frozenset({"a.json", "b.json"})}
    assert cli._audit_findings_introduced(before, {key: frozenset({"a.json"})}) == []
    assert (
        cli._audit_findings_introduced(before, {key: frozenset({"a.json", "b.json"})})
        == []
    )
    assert cli._audit_findings_introduced(
        before, {key: frozenset({"a.json", "b.json", "c.json"})}
    ) == [key]
    assert cli._audit_findings_introduced({}, {key: frozenset()}) == [key]


def test_staged_manifest_attested_paths_include_metadata(tmp_path: Path) -> None:
    from axiom_encode import cli

    repo = tmp_path / "rulespec-us"
    (repo / "us" / "statutes").mkdir(parents=True)
    staged = json.dumps(
        {
            "applied_files": [
                {"path": "us/statutes/26/32.yaml", "sha256": "0" * 64},
                {"path": ".axiom/toolchain.toml", "sha256": "1" * 64},
                {"path": "us/statutes/26/gone.yaml", "deleted": True},
            ]
        }
    ).encode()
    paths = cli._staged_manifest_attested_paths(
        repo, Path(f"{MANIFEST_DIR}/us/statutes/26/32.json"), staged
    )
    assert paths == ["us/statutes/26/32.yaml", ".axiom/toolchain.toml"]
