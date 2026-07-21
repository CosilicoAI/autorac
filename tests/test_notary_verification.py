"""Strict notary-verifier profile and provisional receipt tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from axiom_encode import cli
from axiom_encode.cli import (
    _append_generated_zero_branch_tests_if_missing,
    _zero_branch_coverage_issues_for_files,
)
from axiom_encode.harness.evals import _deterministic_tree_identity
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from axiom_encode.notary_verification import (
    NOTARY_PROFILE_ID,
    NOTARY_RECEIPT_SCHEMA_ID,
    NotaryVerificationError,
    attach_receipt_sha256,
    canonical_receipt_body_bytes,
    canonical_receipt_bytes,
    receipt_body_sha256,
    run_notary_verification,
)
from tests.release_object_fixtures import bind_test_corpus_release

_RELEASE_NAME = "notary-verifier-test-release"
_RELEASE_VERSION = "2026-notary-verifier-test"
_MODULE_RELATIVE = Path("us/statutes/26/151.yaml")
_TEST_RELATIVE = Path("us/statutes/26/151.test.yaml")
_FIXED_NOW = datetime(2026, 7, 21, 15, 4, 5, tzinfo=timezone.utc)

_RULESPEC = """format: rulespec/v1
rules:
  - name: section_151_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if amount_allowed:
              amount
          else:
              0
"""

_COMPANION = """- name: positive_amount
  period: 2026
  input:
    us:statutes/26/151#input.amount_allowed: true
    us:statutes/26/151#input.amount: 5
  output:
    us:statutes/26/151#section_151_amount: 5
"""


@dataclass(frozen=True)
class _NotaryFixture:
    policy_root: Path
    corpus_root: Path
    engine_root: Path
    module: Path
    companion: Path


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "notary-test@example.com")
    _git(repo, "config", "user.name", "Notary Test")


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A", "--force")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _write_corpus(corpus_root: Path):
    provision = (
        corpus_root / "data/corpus/provisions/us/statute" / f"{_RELEASE_VERSION}.jsonl"
    )
    provision.parent.mkdir(parents=True)
    provision.write_text(
        json.dumps(
            {
                "id": "notary-verifier-source",
                "citation_path": "us/statute/26/151",
                "body": "The amount is allowed when the condition holds; otherwise zero.",
                "jurisdiction": "us",
                "document_class": "statute",
                "version": _RELEASE_VERSION,
                "source_path": "sources/us/statute/notary-verifier-source",
                "source_as_of": "2026-07-21",
                "expression_date": "2026-07-21",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return bind_test_corpus_release(
        corpus_root,
        _RELEASE_NAME,
        [("us", "statute", _RELEASE_VERSION)],
    )


def _write_fake_engine(engine_root: Path) -> None:
    binary = engine_root / "target/debug/axiom-rules-engine"
    binary.parent.mkdir(parents=True)
    binary.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

if len(sys.argv) < 2 or sys.argv[1] != "compile":
    raise SystemExit(2)
output = pathlib.Path(sys.argv[sys.argv.index("--output") + 1])
output.write_text(json.dumps({
    "program": {
        "parameters": [],
        "relations": [],
        "derived": [{
            "name": "section_151_amount",
            "id": "us:statutes/26/151#section_151_amount",
            "entity": "TaxUnit"
        }]
    }
}))
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)


def _write_notary_fixture(tmp_path: Path) -> _NotaryFixture:
    policy_root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    engine_root = tmp_path / "axiom-rules-engine"
    _init_git_repo(policy_root)
    _init_git_repo(engine_root)

    module = policy_root / _MODULE_RELATIVE
    companion = policy_root / _TEST_RELATIVE
    module.parent.mkdir(parents=True)
    module.write_text(_RULESPEC, encoding="utf-8")
    companion.write_text(_COMPANION, encoding="utf-8")

    waiver = policy_root / "known-validation-gaps.yaml"
    waiver.write_text("validate_failures: {}\n", encoding="utf-8")
    release = _write_corpus(corpus_root)
    toolchain = policy_root / ".axiom/toolchain.toml"
    toolchain.parent.mkdir()
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{_RELEASE_NAME}"\n'
        "axiom_corpus_release_content_sha256 = "
        f'"{release.content_sha256}"\n'
        "validation_waiver_set_sha256 = "
        f'"{hashlib.sha256(waiver.read_bytes()).hexdigest()}"\n',
        encoding="utf-8",
    )
    _write_fake_engine(engine_root)
    _commit_all(policy_root, "notary policy fixture")
    _commit_all(engine_root, "notary engine fixture")
    return _NotaryFixture(
        policy_root=policy_root,
        corpus_root=corpus_root,
        engine_root=engine_root,
        module=module,
        companion=companion,
    )


def _policy_snapshot(root: Path) -> dict[str, object]:
    return _deterministic_tree_identity(
        root,
        excluded_directory_names=frozenset({".git"}),
    )


def _gate(receipt: dict[str, object], name: str) -> dict[str, str]:
    gates = receipt["gates"]
    assert isinstance(gates, list)
    return next(item for item in gates if item["gate"] == name)


def test_receipt_canonicalization_and_self_hash_match_byte_golden():
    body = {
        "subject_tree": "0" * 40,
        "schema_status": "PROVISIONAL",
        "schema_id": NOTARY_RECEIPT_SCHEMA_ID,
        "gates": [
            {
                "status": "passed",
                "reproducibility": "public",
                "gate": "compile",
            }
        ],
        "run": {
            "timestamp": "2026-07-21T15:04:05Z",
            "profile_id": NOTARY_PROFILE_ID,
            "encoder_version": "9.9.9",
        },
    }
    receipt = attach_receipt_sha256(body)
    golden = Path("tests/fixtures/notary_receipt_v0.hex")

    assert canonical_receipt_bytes(receipt) == bytes.fromhex(golden.read_text())
    assert receipt["receipt_sha256"] == receipt_body_sha256(receipt)
    assert (
        receipt["receipt_sha256"]
        == hashlib.sha256(canonical_receipt_body_bytes(receipt)).hexdigest()
    )
    assert attach_receipt_sha256(receipt) == receipt

    reordered = dict(reversed(list(body.items())))
    assert canonical_receipt_bytes(attach_receipt_sha256(reordered)) == (
        canonical_receipt_bytes(receipt)
    )


def test_strict_profile_does_not_repair_apply_repairable_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    receipt_out = tmp_path / "evidence/receipt.json"
    before = _policy_snapshot(fixture.policy_root)
    reviewer_calls: list[str] = []

    def passing_reviewer(prompt: str, **_kwargs):
        reviewer_calls.append(prompt)
        return '{"score":10,"passed":true,"issues":[]}', 0

    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline.run_claude_code",
        passing_reviewer,
    )
    result = run_notary_verification(
        policy_repo_path=fixture.policy_root,
        corpus_path=fixture.corpus_root,
        axiom_rules_engine_path=fixture.engine_root,
        receipt_out=receipt_out,
        changed_files=[_MODULE_RELATIVE],
        allow_reduced=True,
        now=_FIXED_NOW,
    )

    assert result.passed is False
    assert any("Zero branch test coverage missing" in issue for issue in result.issues)
    assert len(reviewer_calls) == 4
    for reviewer_gate in (
        "rulespec-reviewer",
        "formula-reviewer",
        "parameter-reviewer",
        "integration-reviewer",
    ):
        assert _gate(result.receipt, reviewer_gate)["status"] == "passed"
    assert _gate(result.receipt, "companion-tests")["status"] == "failed"
    assert _policy_snapshot(fixture.policy_root) == before
    assert _git(fixture.policy_root, "status", "--porcelain").stdout == ""
    assert "auto_zero_section_151_amount" not in fixture.companion.read_text()

    repair_root = tmp_path / "apply-repair-copy" / "rulespec-us" / "us"
    repair_rules = repair_root / "statutes/26/151.yaml"
    repair_test = repair_root / "statutes/26/151.test.yaml"
    repair_rules.parent.mkdir(parents=True)
    shutil.copy2(fixture.module, repair_rules)
    shutil.copy2(fixture.companion, repair_test)
    repair_issues = _zero_branch_coverage_issues_for_files(
        rules_file=repair_rules,
        test_file=repair_test,
    )
    repaired = _append_generated_zero_branch_tests_if_missing(
        rules_file=repair_rules,
        test_file=repair_test,
        repo_path=repair_root,
        relative_output=Path("statutes/26/151.yaml"),
        issues=repair_issues,
    )

    assert repaired == ["auto_zero_section_151_amount"]
    repaired_text = repair_test.read_text()
    assert "auto_zero_section_151_amount" in repaired_text
    assert "us:statutes/26/151#input.amount_allowed: false" in repaired_text
    assert "us:statutes/26/151#section_151_amount: 0" in repaired_text


def test_command_never_writes_inside_policy_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    receipt_out = tmp_path / "notary-output/receipt.json"
    before = _policy_snapshot(fixture.policy_root)
    reviewer_calls: list[str] = []
    validations: list[tuple[Path, bool]] = []

    def passing_reviewer(prompt: str, **_kwargs):
        reviewer_calls.append(prompt)
        return '{"score":10,"passed":true,"issues":[]}', 0

    original_validate = ValidatorPipeline.validate

    def tracked_validate(self, path: Path, *, skip_reviewers: bool = False):
        validations.append((Path(path), skip_reviewers))
        return original_validate(self, path, skip_reviewers=skip_reviewers)

    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline.run_claude_code",
        passing_reviewer,
    )
    monkeypatch.setattr(ValidatorPipeline, "validate", tracked_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "axiom-encode",
            "notary-verify",
            "--policy-repo-path",
            str(fixture.policy_root),
            "--corpus-path",
            str(fixture.corpus_root),
            "--axiom-rules-engine-path",
            str(fixture.engine_root),
            "--changed-files",
            _MODULE_RELATIVE.as_posix(),
            "--receipt-out",
            str(receipt_out),
            "--allow-reduced",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 1
    assert receipt_out.is_file()
    assert _policy_snapshot(fixture.policy_root) == before
    assert _git(fixture.policy_root, "status", "--porcelain").stdout == ""
    assert len(reviewer_calls) == 4
    assert validations == [(fixture.module, False)]


def test_dirty_policy_worktree_is_refused_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    dirty = fixture.policy_root / "untracked.txt"
    dirty.write_text("uncommitted\n", encoding="utf-8")
    receipt_out = tmp_path / "receipt.json"

    class ForbiddenPipeline:
        def __init__(self, **_kwargs):
            raise AssertionError("dirty checkout must fail before validation")

    monkeypatch.setattr(
        "axiom_encode.notary_verification.ValidatorPipeline",
        ForbiddenPipeline,
    )
    with pytest.raises(NotaryVerificationError, match="worktree is dirty"):
        run_notary_verification(
            policy_repo_path=fixture.policy_root,
            corpus_path=fixture.corpus_root,
            axiom_rules_engine_path=fixture.engine_root,
            receipt_out=receipt_out,
            changed_files=[_MODULE_RELATIVE],
            allow_reduced=True,
            now=_FIXED_NOW,
        )

    assert not receipt_out.exists()


def test_absent_restricted_oracle_fails_closed_or_is_explicitly_reduced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    refused_receipt = tmp_path / "refused.json"

    with pytest.raises(NotaryVerificationError, match="fails closed"):
        run_notary_verification(
            policy_repo_path=fixture.policy_root,
            corpus_path=fixture.corpus_root,
            axiom_rules_engine_path=fixture.engine_root,
            receipt_out=refused_receipt,
            changed_files=[_MODULE_RELATIVE],
            now=_FIXED_NOW,
        )
    assert not refused_receipt.exists()

    reviewer_calls: list[str] = []

    def passing_reviewer(prompt: str, **_kwargs):
        reviewer_calls.append(prompt)
        return '{"score":10,"passed":true,"issues":[]}', 0

    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline.run_claude_code",
        passing_reviewer,
    )
    reduced_receipt = tmp_path / "reduced.json"
    result = run_notary_verification(
        policy_repo_path=fixture.policy_root,
        corpus_path=fixture.corpus_root,
        axiom_rules_engine_path=fixture.engine_root,
        receipt_out=reduced_receipt,
        changed_files=[_MODULE_RELATIVE],
        allow_reduced=True,
        now=_FIXED_NOW,
    )

    assert result.passed is False
    assert result.receipt["status"] == "failed"
    assert result.receipt["dependencies"]["policyengine_oracle"] is None
    assert len(reviewer_calls) == 4
    assert _gate(result.receipt, "policyengine-oracle") == {
        "gate": "policyengine-oracle",
        "status": "reduced",
        "reproducibility": "restricted-pinned",
    }
    assert json.loads(reduced_receipt.read_bytes()) == result.receipt
    assert reduced_receipt.read_bytes() == canonical_receipt_bytes(result.receipt)


def test_receipt_output_inside_policy_repo_is_rejected(tmp_path: Path):
    fixture = _write_notary_fixture(tmp_path)

    with pytest.raises(NotaryVerificationError, match="outside"):
        run_notary_verification(
            policy_repo_path=fixture.policy_root,
            corpus_path=fixture.corpus_root,
            axiom_rules_engine_path=fixture.engine_root,
            receipt_out=fixture.policy_root / "receipt.json",
            changed_files=[_MODULE_RELATIVE],
            allow_reduced=True,
            now=_FIXED_NOW,
        )


def test_whole_repo_target_set_is_explicit_and_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)

    def passing_reviewer(_prompt: str, **_kwargs):
        return '{"score":10,"passed":true,"issues":[]}', 0

    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline.run_claude_code",
        passing_reviewer,
    )
    result = run_notary_verification(
        policy_repo_path=fixture.policy_root,
        corpus_path=fixture.corpus_root,
        axiom_rules_engine_path=fixture.engine_root,
        receipt_out=tmp_path / "whole-repo.json",
        whole_repo=True,
        allow_reduced=True,
        now=_FIXED_NOW,
    )

    assert result.receipt["targets"] == {
        "mode": "whole-repo",
        "files": [_MODULE_RELATIVE.as_posix()],
    }


def test_receipt_has_no_signing_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)

    def passing_reviewer(_prompt: str, **_kwargs):
        return '{"score":10,"passed":true,"issues":[]}', 0

    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline.run_claude_code",
        passing_reviewer,
    )
    result = run_notary_verification(
        policy_repo_path=fixture.policy_root,
        corpus_path=fixture.corpus_root,
        axiom_rules_engine_path=fixture.engine_root,
        receipt_out=tmp_path / "receipt.json",
        changed_files=[_MODULE_RELATIVE],
        allow_reduced=True,
        now=_FIXED_NOW,
    )

    serialized = canonical_receipt_bytes(result.receipt)
    assert b"signature" not in serialized
    assert b"hostname" not in serialized
    assert result.receipt["schema_id"] == NOTARY_RECEIPT_SCHEMA_ID
    assert (
        result.receipt["subject_commit"]
        == _git(fixture.policy_root, "rev-parse", "HEAD").stdout.strip()
    )
    assert (
        result.receipt["subject_tree"]
        == _git(fixture.policy_root, "rev-parse", "HEAD^{tree}").stdout.strip()
    )
    assert result.receipt["dependencies"]["corpus_release"]["name"] == _RELEASE_NAME
    assert len(result.receipt["dependencies"]["corpus_release"]["content_sha256"]) == 64
    assert result.receipt["waiver_set"]["count"] == 0
    engine_identity = result.receipt["dependencies"]["axiom_rules_engine"]
    assert (
        engine_identity["commit"]
        == _git(fixture.engine_root, "rev-parse", "HEAD").stdout.strip()
    )
    assert engine_identity["executable"]["path"] == ("target/debug/axiom-rules-engine")
