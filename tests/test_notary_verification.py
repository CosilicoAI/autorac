"""Strict notary-verifier profile and provisional receipt tests."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode import cli
from axiom_encode import notary_verification as notary_module
from axiom_encode.cli import _try_repair_generated_zero_branch_tests_for_apply
from axiom_encode.harness.evals import _deterministic_tree_identity
from axiom_encode.harness.policyengine_runtime import (
    POLICYENGINE_RUNTIME_PIN_SCHEMA,
    POLICYENGINE_RUNTIME_SCHEMA,
)
from axiom_encode.harness.validator_pipeline import (
    PipelineResult,
    ValidationResult,
    ValidatorPipeline,
)
from axiom_encode.notary_verification import (
    NOTARY_PROFILE_ID,
    NOTARY_RECEIPT_SCHEMA_ID,
    NotaryVerificationError,
    _ci_compile_passed,
    _oracle_passed,
    attach_receipt_sha256,
    canonical_receipt_body_bytes,
    canonical_receipt_bytes,
    receipt_body_sha256,
    run_notary_verification,
)
from tests.release_object_fixtures import bind_test_corpus_release
from tests.test_policyengine_runtime import (
    _install_fixture_trust,
    _runtime_repo,
)

_RELEASE_NAME = "notary-verifier-test-release"
_RELEASE_VERSION = "2026-notary-verifier-test"
_MODULE_RELATIVE = Path("us/statutes/26/151.yaml")
_FIXED_NOW = datetime(2026, 7, 21, 15, 4, 5, tzinfo=timezone.utc)
_ORACLE_OUTPUT_ID = "us:statutes/26/151#senior_deduction_age_threshold"
_ORACLE_SOURCE_TEXT = "The senior deduction age threshold is 65 years."

_RULESPEC = """format: rulespec/v1
module:
  title: Section 151 test amount
  jurisdiction: us
  citation_path: us/statutes/26/151
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/151
rules:
  - name: section_151_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    source: 26 USC 151
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us/statute/26/151
              excerpt: The amount is allowed when the condition holds; otherwise zero.
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if amount_allowed:
              amount
          else:
              0
"""

_COMPANION = """- name: positive_amount
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/151#input.amount_allowed: true
    us:statutes/26/151#input.amount: 5
  output:
    us:statutes/26/151#section_151_amount: 5
- name: zero_amount
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/151#input.amount_allowed: false
    us:statutes/26/151#input.amount: 5
  output:
    us:statutes/26/151#section_151_amount: 0
"""

_ORACLE_RULESPEC = """format: rulespec/v1
module:
  title: Section 151 senior deduction age threshold
  jurisdiction: us
  citation_path: us/statutes/26/151
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/151
rules:
  - name: senior_deduction_age_threshold
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    source: 26 USC 151
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us/statute/26/151
              excerpt: The senior deduction age threshold is 65 years.
    versions:
      - effective_from: '2026-01-01'
        formula: '65'
"""

_ORACLE_COMPANION = """- name: senior_age_threshold
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input: {}
  output:
    us:statutes/26/151#senior_deduction_age_threshold: 65
"""

# Exact generated candidate from
# TestEncode.test_encode_apply_auto_repairs_generic_zero_branch_test.  Keeping
# this fixture in sync with that apply-path regression proves the notary profile
# observes the repairable bytes without invoking the repair path.
_APPLY_REPAIRABLE_RELATIVE = Path("us/statutes/26/213.yaml")
_APPLY_REPAIRABLE_RULESPEC = """format: rulespec/v1
rules:
  - name: lodging_treated_as_medical_care
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          lodging_not_lavish_or_extravagant
          and lodging_away_from_home_primarily_for_and_essential_to_medical_care
  - name: lodging_medical_care_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if lodging_treated_as_medical_care:
              min(lodging_amount_paid, lodging_medical_care_nightly_cap * lodging_nights * lodging_individuals)
          else:
              0
  - name: lodging_medical_care_secondary_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if secondary_lodging_condition:
              secondary_lodging_amount
          else:
              0
"""
_APPLY_REPAIRABLE_COMPANION = """- name: qualifying_lodging_positive
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/213#input.lodging_not_lavish_or_extravagant: true
    us:statutes/26/213#input.lodging_away_from_home_primarily_for_and_essential_to_medical_care: true
    us:statutes/26/213#input.lodging_amount_paid: 300
    us:statutes/26/213#input.lodging_medical_care_nightly_cap: 50
    us:statutes/26/213#input.lodging_nights: 2
    us:statutes/26/213#input.lodging_individuals: 2
  output:
    us:statutes/26/213#lodging_medical_care_amount: 200
"""


@dataclass(frozen=True)
class _NotaryFixture:
    policy_root: Path
    corpus_root: Path
    engine_root: Path
    module: Path
    companion: Path
    corpus_content_sha256: str
    waiver_sha256: str
    engine_executable_sha256: str


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    ).stdout


def _overwrite_loose_git_object(
    repo: Path,
    object_id: str,
    object_type: str,
    payload: bytes,
) -> None:
    object_path = repo / ".git/objects" / object_id[:2] / object_id[2:]
    assert object_path.is_file()
    object_path.chmod(0o644)
    object_path.write_bytes(
        zlib.compress(
            f"{object_type} {len(payload)}\0".encode("ascii") + payload
        )
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

if len(sys.argv) < 2:
    raise SystemExit(2)
if sys.argv[1] == "compile":
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
elif sys.argv[1] == "run-compiled":
    request = json.load(sys.stdin)
    inputs = {
        item["name"]: item["value"].get("value")
        for item in request["dataset"]["inputs"]
    }
    allowed = bool(inputs.get("us:statutes/26/151#input.amount_allowed"))
    amount = int(inputs.get("us:statutes/26/151#input.amount", 0)) if allowed else 0
    print(json.dumps({
        "results": [{
            "outputs": {
                "us:statutes/26/151#section_151_amount": {
                    "kind": "scalar",
                    "value": {"kind": "integer", "value": amount}
                }
            }
        }]
    }))
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)


def _write_fake_oracle_engine(engine_root: Path) -> None:
    binary = engine_root / "target/debug/axiom-rules-engine"
    binary.write_text(
        f"""#!/usr/bin/env python3
import json
import pathlib
import sys

output_id = {_ORACLE_OUTPUT_ID!r}
if len(sys.argv) < 2:
    raise SystemExit(2)
if sys.argv[1] == "compile":
    output = pathlib.Path(sys.argv[sys.argv.index("--output") + 1])
    output.write_text(json.dumps({{
        "program": {{
            "parameters": [],
            "relations": [],
            "derived": [{{
                "name": "senior_deduction_age_threshold",
                "id": output_id,
                "entity": "TaxUnit"
            }}]
        }}
    }}))
elif sys.argv[1] == "run-compiled":
    json.load(sys.stdin)
    print(json.dumps({{
        "results": [{{
            "outputs": {{
                output_id: {{
                    "kind": "scalar",
                    "value": {{"kind": "integer", "value": 65}}
                }}
            }}
        }}]
    }}))
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)


def _bind_oracle_corpus(fixture: _NotaryFixture):
    provision = (
        fixture.corpus_root
        / "data/corpus/provisions/us/statute"
        / f"{_RELEASE_VERSION}.jsonl"
    )
    payload = json.loads(provision.read_text(encoding="utf-8"))
    payload["body"] = _ORACLE_SOURCE_TEXT
    provision.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    release = bind_test_corpus_release(
        fixture.corpus_root,
        _RELEASE_NAME,
        [("us", "statute", _RELEASE_VERSION)],
    )
    (fixture.policy_root / ".axiom/toolchain.toml").write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{_RELEASE_NAME}"\n'
        "axiom_corpus_release_content_sha256 = "
        f'"{release.content_sha256}"\n'
        "validation_waiver_set_sha256 = "
        f'"{fixture.waiver_sha256}"\n',
        encoding="utf-8",
    )
    return release


def _write_executable_oracle_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    parameter_value: float = 65.0,
) -> tuple[Path, str]:
    runtime_root = _runtime_repo(tmp_path)
    (runtime_root / "policyengine_us/__init__.py").write_text(
        f"""class _Parameter:
    def __getattr__(self, _name):
        return self

    def __float__(self):
        return {parameter_value!r}


class CountryTaxBenefitSystem:
    def parameters(self, _period):
        return _Parameter()


class Simulation:
    pass
""",
        encoding="utf-8",
    )
    _git(runtime_root, "add", "policyengine_us/__init__.py")
    _git(runtime_root, "commit", "-qm", "executable oracle fixture")
    runtime_commit = _git(runtime_root, "rev-parse", "HEAD").stdout.strip()

    # The shared admission fixture deliberately contains only a skeletal
    # stdlib. This regular, hash-bound wrapper makes its oracle command execute
    # using pytest's known interpreter without weakening production admission.
    python_path = runtime_root / ".venv/bin/python"
    python_path.write_text(
        "#!/bin/sh\nexec " + shlex.quote(sys._base_executable) + ' "$@"\n',
        encoding="utf-8",
    )
    python_path.chmod(0o755)
    _install_fixture_trust(
        monkeypatch,
        runtime_root,
        allowed_commit=runtime_commit,
    )
    return runtime_root, runtime_commit


def _write_notary_fixture(
    tmp_path: Path,
    *,
    module_relative: Path = _MODULE_RELATIVE,
    rulespec: str = _RULESPEC,
    companion: str = _COMPANION,
) -> _NotaryFixture:
    policy_root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    engine_root = tmp_path / "axiom-rules-engine"
    _init_git_repo(policy_root)
    _init_git_repo(engine_root)

    module = policy_root / module_relative
    companion_path = module.with_name(f"{module.stem}.test.yaml")
    module.parent.mkdir(parents=True)
    module.write_text(rulespec, encoding="utf-8")
    companion_path.write_text(companion, encoding="utf-8")

    waiver = policy_root / "known-validation-gaps.yaml"
    waiver.write_text("validate_failures: {}\n", encoding="utf-8")
    release = _write_corpus(corpus_root)
    toolchain = policy_root / ".axiom/toolchain.toml"
    toolchain.parent.mkdir()
    waiver_sha256 = hashlib.sha256(waiver.read_bytes()).hexdigest()
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{_RELEASE_NAME}"\n'
        "axiom_corpus_release_content_sha256 = "
        f'"{release.content_sha256}"\n'
        "validation_waiver_set_sha256 = "
        f'"{waiver_sha256}"\n',
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
        companion=companion_path,
        corpus_content_sha256=release.content_sha256,
        waiver_sha256=waiver_sha256,
        engine_executable_sha256=hashlib.sha256(
            (engine_root / "target/debug/axiom-rules-engine").read_bytes()
        ).hexdigest(),
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
    fixture = _write_notary_fixture(
        tmp_path,
        module_relative=_APPLY_REPAIRABLE_RELATIVE,
        rulespec=_APPLY_REPAIRABLE_RULESPEC,
        companion=_APPLY_REPAIRABLE_COMPANION,
    )
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
        changed_files=[_APPLY_REPAIRABLE_RELATIVE],
        allow_reduced=True,
        now=_FIXED_NOW,
    )

    assert result.passed is False
    assert any(
        "Zero branch test coverage missing" in issue
        and "lodging_medical_care_amount" in issue
        for issue in result.issues
    )
    assert len(reviewer_calls) == 4
    for reviewer_gate in (
        "rulespec-reviewer",
        "formula-reviewer",
        "parameter-reviewer",
        "integration-reviewer",
    ):
        assert _gate(result.receipt, reviewer_gate) == {
            "gate": reviewer_gate,
            "status": "passed",
            "reproducibility": "ci-attested",
        }
    deterministic_gates = {
        gate["gate"]
        for gate in result.receipt["gates"]
        if gate["gate"]
        in {
            "proof-revalidation",
            "companion-tests",
            "grounding-contract",
            "layout-inspection",
        }
    }
    assert deterministic_gates == {
        "proof-revalidation",
        "companion-tests",
        "grounding-contract",
        "layout-inspection",
    }
    assert _gate(result.receipt, "grounding-contract")["status"] == "passed"
    assert _gate(result.receipt, "companion-tests")["status"] == "failed"
    assert result.receipt["status"] == "failed"
    assert receipt_out.read_bytes() == canonical_receipt_bytes(result.receipt)
    assert _policy_snapshot(fixture.policy_root) == before
    assert _git(fixture.policy_root, "status", "--porcelain").stdout == ""
    assert "auto_zero_lodging_medical_care_amount" not in fixture.companion.read_text()
    assert (
        "auto_zero_lodging_medical_care_secondary_amount"
        not in fixture.companion.read_text()
    )

    output_root = tmp_path / "apply-repair-copy" / "out"
    runner = "codex-test-model"
    repair_rules = output_root / runner / "statutes/26/213.yaml"
    repair_test = repair_rules.with_name("213.test.yaml")
    repair_rules.parent.mkdir(parents=True)
    shutil.copy2(fixture.module, repair_rules)
    shutil.copy2(fixture.companion, repair_test)
    apply_policy_root = tmp_path / "apply-repair-copy/rulespec-us/us"
    apply_policy_root.mkdir(parents=True)
    repaired = _try_repair_generated_zero_branch_tests_for_apply(
        SimpleNamespace(output_file=str(repair_rules), runner=runner),
        output_root=output_root,
        policy_repo_path=apply_policy_root,
        issues=[
            "statutes/26/213.yaml: ci: Zero branch test coverage missing: "
            "`lodging_medical_care_amount` has a formula branch that returns 0.",
            "statutes/26/213.yaml: ci: Zero branch test coverage missing: "
            "`lodging_medical_care_secondary_amount` has a formula branch that "
            "returns 0.",
        ],
    )

    assert repaired == [
        "auto_zero_lodging_medical_care_amount",
        "auto_zero_lodging_medical_care_secondary_amount",
    ]
    repaired_text = repair_test.read_text()
    assert "auto_zero_lodging_medical_care_amount" in repaired_text
    assert "auto_zero_lodging_medical_care_secondary_amount" in repaired_text
    assert (
        "us:statutes/26/213#input.lodging_not_lavish_or_extravagant: false"
        in repaired_text
    )
    assert "us:statutes/26/213#lodging_medical_care_amount: 0" in repaired_text


def test_command_never_writes_inside_policy_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    receipt_out = tmp_path / "notary-output/receipt.json"
    before = _policy_snapshot(fixture.policy_root)
    reviewer_calls: list[str] = []
    validations: list[tuple[Path, Path, Path, str, bool]] = []

    def passing_reviewer(prompt: str, **_kwargs):
        reviewer_calls.append(prompt)
        return '{"score":10,"passed":true,"issues":[]}', 0

    original_validate = ValidatorPipeline.validate

    def tracked_validate(self, path: Path, *, skip_reviewers: bool = False):
        validations.append(
            (
                Path(path),
                Path(self.policy_repo_path),
                Path(self.axiom_rules_path),
                hashlib.sha256(
                    (
                        Path(self.axiom_rules_path)
                        / "target/debug/axiom-rules-engine"
                    ).read_bytes()
                ).hexdigest(),
                skip_reviewers,
            )
        )
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

    assert exit_info.value.code == 0
    assert receipt_out.is_file()
    assert _policy_snapshot(fixture.policy_root) == before
    assert _git(fixture.policy_root, "status", "--porcelain").stdout == ""
    assert len(reviewer_calls) == 4
    assert len(validations) == 1
    target, policy_snapshot, engine_snapshot, engine_sha256, skip_reviewers = (
        validations[0]
    )
    assert target.relative_to(policy_snapshot) == Path("statutes/26/151.yaml")
    assert policy_snapshot != fixture.policy_root / "us"
    assert engine_snapshot != fixture.engine_root
    assert skip_reviewers is False
    assert engine_sha256 == fixture.engine_executable_sha256


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


def test_git_executable_mode_change_is_refused_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    fixture.module.chmod(0o755)
    _commit_all(fixture.policy_root, "make RuleSpec executable")
    fixture.module.chmod(0o654)
    receipt_out = tmp_path / "mode-change-receipt.json"

    assert _git(fixture.policy_root, "status", "--porcelain").stdout != ""

    class ForbiddenPipeline:
        def __init__(self, **_kwargs):
            raise AssertionError("mode changes must fail before validation")

    monkeypatch.setattr(
        "axiom_encode.notary_verification.ValidatorPipeline",
        ForbiddenPipeline,
    )
    with pytest.raises(NotaryVerificationError, match="raw HEAD tree"):
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


def test_intent_to_add_index_entry_is_refused_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    intent_to_add = fixture.policy_root / "intent-to-add.txt"
    intent_to_add.write_text("uncommitted\n", encoding="utf-8")
    _git(fixture.policy_root, "add", "--intent-to-add", intent_to_add.name)
    receipt_out = tmp_path / "intent-to-add-receipt.json"

    assert _git(fixture.policy_root, "status", "--porcelain").stdout != ""

    class ForbiddenPipeline:
        def __init__(self, **_kwargs):
            raise AssertionError("dirty index must fail before validation")

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


def test_staged_gitlink_is_not_hidden_by_local_diff_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    nested = fixture.policy_root / "nested-repository"
    _init_git_repo(nested)
    (nested / "tracked.txt").write_text("nested\n", encoding="utf-8")
    _commit_all(nested, "nested fixture")
    _git(fixture.policy_root, "add", nested.name)
    _git(fixture.policy_root, "config", "diff.ignoreSubmodules", "all")
    receipt_out = tmp_path / "staged-gitlink-receipt.json"

    assert _git(fixture.policy_root, "status", "--porcelain").stdout != ""

    class ForbiddenPipeline:
        def __init__(self, **_kwargs):
            raise AssertionError("staged gitlinks must fail before validation")

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


def test_noncommit_head_is_refused_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    tree = _git(fixture.policy_root, "rev-parse", "HEAD^{tree}").stdout.strip()
    (fixture.policy_root / ".git/HEAD").write_text(tree + "\n", encoding="ascii")

    class ForbiddenPipeline:
        def __init__(self, **_kwargs):
            raise AssertionError("non-commit HEAD must fail before validation")

    monkeypatch.setattr(
        "axiom_encode.notary_verification.ValidatorPipeline",
        ForbiddenPipeline,
    )
    receipt_out = tmp_path / "noncommit-head.json"
    with pytest.raises(NotaryVerificationError, match="Cannot inspect.*Git identity"):
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


def test_ignored_rulespec_input_is_not_accepted_as_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    gitignore = fixture.policy_root / ".gitignore"
    gitignore.write_text("/us/statutes/26/ignored.yaml\n", encoding="utf-8")
    _commit_all(fixture.policy_root, "ignore one RuleSpec path")
    ignored = fixture.policy_root / "us/statutes/26/ignored.yaml"
    ignored.write_text(_RULESPEC, encoding="utf-8")
    receipt_out = tmp_path / "ignored-input.json"

    assert _git(fixture.policy_root, "status", "--porcelain").stdout == ""

    class ForbiddenPipeline:
        def __init__(self, **_kwargs):
            raise AssertionError("unbound input must fail before validation")

    monkeypatch.setattr(
        "axiom_encode.notary_verification.ValidatorPipeline",
        ForbiddenPipeline,
    )
    with pytest.raises(NotaryVerificationError, match="not tracked by HEAD"):
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


def test_assume_unchanged_rulespec_input_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    relative = fixture.module.relative_to(fixture.policy_root).as_posix()
    _git(fixture.policy_root, "update-index", "--assume-unchanged", relative)
    fixture.module.write_text(_RULESPEC + "# hidden edit\n", encoding="utf-8")
    receipt_out = tmp_path / "assume-unchanged.json"

    assert _git(fixture.policy_root, "status", "--porcelain").stdout == ""

    class ForbiddenPipeline:
        def __init__(self, **_kwargs):
            raise AssertionError("hidden index state must fail before validation")

    monkeypatch.setattr(
        "axiom_encode.notary_verification.ValidatorPipeline",
        ForbiddenPipeline,
    )
    with pytest.raises(NotaryVerificationError, match="unsupported Git index flags"):
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


def test_git_clean_filter_cannot_hide_modified_rulespec_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    attributes = fixture.policy_root / ".gitattributes"
    attributes.write_text(
        "us/statutes/26/151.yaml filter=notary-clean\n",
        encoding="utf-8",
    )
    _git(
        fixture.policy_root,
        "config",
        "filter.notary-clean.clean",
        "/usr/bin/sed '/^# hidden filtered edit$/d'",
    )
    _git(fixture.policy_root, "config", "filter.notary-clean.smudge", "/bin/cat")
    _commit_all(fixture.policy_root, "configure clean-filter fixture")
    fixture.module.write_text(
        _RULESPEC + "# hidden filtered edit\n",
        encoding="utf-8",
    )
    receipt_out = tmp_path / "filtered-dirty.json"

    assert (
        _git(
            fixture.policy_root,
            "diff",
            "--raw",
            "HEAD",
            "--",
            _MODULE_RELATIVE.as_posix(),
        ).stdout
        == ""
    )

    class ForbiddenPipeline:
        def __init__(self, **_kwargs):
            raise AssertionError("raw HEAD mismatch must fail before validation")

    monkeypatch.setattr(
        "axiom_encode.notary_verification.ValidatorPipeline",
        ForbiddenPipeline,
    )
    with pytest.raises(NotaryVerificationError, match="raw HEAD tree"):
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


def test_repo_local_fsmonitor_is_never_executed(
    tmp_path: Path,
):
    fixture = _write_notary_fixture(tmp_path)
    sentinel = tmp_path / "fsmonitor-was-executed"
    hook = tmp_path / "fsmonitor-hook"
    hook.write_text(
        "#!/bin/sh\n: > " + shlex.quote(str(sentinel)) + "\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    _git(fixture.policy_root, "config", "core.fsmonitor", str(hook))
    receipt_out = tmp_path / "fsmonitor-receipt.json"

    with pytest.raises(NotaryVerificationError, match="fails closed"):
        run_notary_verification(
            policy_repo_path=fixture.policy_root,
            corpus_path=fixture.corpus_root,
            axiom_rules_engine_path=fixture.engine_root,
            receipt_out=receipt_out,
            changed_files=[_MODULE_RELATIVE],
            now=_FIXED_NOW,
        )

    assert not sentinel.exists()
    assert not receipt_out.exists()


def test_missing_partial_clone_blob_fails_without_git_metadata_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    git_executable = shutil.which("git")
    assert git_executable is not None
    _git(fixture.policy_root, "config", "uploadpack.allowFilter", "true")
    partial_root = tmp_path / "partial" / "rulespec-us"
    partial_root.parent.mkdir()
    subprocess.run(
        [
            git_executable,
            "-c",
            "protocol.file.allow=always",
            "clone",
            "--quiet",
            "--filter=blob:none",
            "--no-checkout",
            fixture.policy_root.as_uri(),
            str(partial_root),
        ],
        check=True,
    )
    _git(partial_root, "read-tree", "HEAD")
    for relative_text in _git(
        fixture.policy_root,
        "ls-files",
        "-z",
        "--",
    ).stdout.split("\0"):
        if not relative_text:
            continue
        source = fixture.policy_root / relative_text
        destination = partial_root / relative_text
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    blob_id = _git(
        partial_root,
        "rev-parse",
        f"HEAD:{_MODULE_RELATIVE.as_posix()}",
    ).stdout.strip()
    missing = subprocess.run(
        [git_executable, "-C", str(partial_root), "cat-file", "-e", blob_id],
        check=False,
        capture_output=True,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
    )
    if missing.returncode == 0:
        pytest.skip("Git did not create a blobless partial clone")

    metadata_before = _deterministic_tree_identity(partial_root / ".git")

    class ForbiddenPipeline:
        def __init__(self, **_kwargs):
            raise AssertionError("missing Git objects must fail before validation")

    monkeypatch.setattr(
        "axiom_encode.notary_verification.ValidatorPipeline",
        ForbiddenPipeline,
    )
    receipt_out = tmp_path / "partial-clone-receipt.json"
    with pytest.raises(NotaryVerificationError):
        run_notary_verification(
            policy_repo_path=partial_root,
            corpus_path=fixture.corpus_root,
            axiom_rules_engine_path=fixture.engine_root,
            receipt_out=receipt_out,
            changed_files=[_MODULE_RELATIVE],
            allow_reduced=True,
            now=_FIXED_NOW,
        )

    assert _deterministic_tree_identity(partial_root / ".git") == metadata_before
    assert not receipt_out.exists()


def test_git_replacement_ref_cannot_substitute_subject_tree_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    original_tree = _git(
        fixture.policy_root,
        "rev-parse",
        "HEAD^{tree}",
    ).stdout.strip()
    replacement_rulespec = _RULESPEC + "# replacement-ref content\n"
    fixture.module.write_text(replacement_rulespec, encoding="utf-8")
    _git(fixture.policy_root, "add", _MODULE_RELATIVE.as_posix())
    replacement_tree = _git(fixture.policy_root, "write-tree").stdout.strip()
    _git(fixture.policy_root, "reset", "--hard", "-q", "HEAD")
    _git(fixture.policy_root, "replace", original_tree, replacement_tree)
    _git(fixture.policy_root, "reset", "--hard", "-q", "HEAD")

    assert fixture.module.read_text(encoding="utf-8") == replacement_rulespec
    assert _git(fixture.policy_root, "status", "--porcelain").stdout == ""

    class ForbiddenPipeline:
        def __init__(self, **_kwargs):
            raise AssertionError("replacement-ref checkout must fail before validation")

    monkeypatch.setattr(
        "axiom_encode.notary_verification.ValidatorPipeline",
        ForbiddenPipeline,
    )
    receipt_out = tmp_path / "replacement-ref.json"
    with pytest.raises(NotaryVerificationError, match="worktree is dirty|raw HEAD tree"):
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


def test_corrupt_loose_git_blob_cannot_substitute_snapshot_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    object_id = _git(
        fixture.policy_root,
        "rev-parse",
        f"HEAD:{_MODULE_RELATIVE.as_posix()}",
    ).stdout.strip()
    replacement = (_RULESPEC + "# corrupt object payload\n").encode()
    _overwrite_loose_git_object(
        fixture.policy_root,
        object_id,
        "blob",
        replacement,
    )

    class ForbiddenPipeline:
        def __init__(self, **_kwargs):
            raise AssertionError("corrupt Git object must fail before validation")

    monkeypatch.setattr(
        "axiom_encode.notary_verification.ValidatorPipeline",
        ForbiddenPipeline,
    )
    receipt_out = tmp_path / "corrupt-object.json"
    with pytest.raises(NotaryVerificationError, match="does not match its object"):
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


def test_corrupt_loose_git_commit_cannot_substitute_subject_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    commit_id = _git(fixture.policy_root, "rev-parse", "HEAD").stdout.strip()
    replacement = _git_bytes(
        fixture.policy_root,
        "cat-file",
        "commit",
        commit_id,
    ) + b"\ncorrupt commit payload\n"
    _overwrite_loose_git_object(
        fixture.policy_root,
        commit_id,
        "commit",
        replacement,
    )

    assert _git(fixture.policy_root, "status", "--porcelain").stdout == ""

    class ForbiddenPipeline:
        def __init__(self, **_kwargs):
            raise AssertionError("corrupt Git commit must fail before validation")

    monkeypatch.setattr(
        "axiom_encode.notary_verification.ValidatorPipeline",
        ForbiddenPipeline,
    )
    receipt_out = tmp_path / "corrupt-commit.json"
    with pytest.raises(
        NotaryVerificationError,
        match="hash mismatch|Git commit payload does not match its object identity",
    ):
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


def test_corrupt_loose_git_root_tree_cannot_substitute_snapshot_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    root_tree_id = _git(
        fixture.policy_root,
        "rev-parse",
        "HEAD^{tree}",
    ).stdout.strip()
    replacement_rulespec = _RULESPEC + "# corrupt root tree payload\n"
    fixture.module.write_text(replacement_rulespec, encoding="utf-8")
    _git(fixture.policy_root, "add", _MODULE_RELATIVE.as_posix())
    replacement_tree_id = _git(
        fixture.policy_root,
        "write-tree",
    ).stdout.strip()
    replacement = _git_bytes(
        fixture.policy_root,
        "cat-file",
        "tree",
        replacement_tree_id,
    )
    _git(fixture.policy_root, "reset", "--hard", "-q", "HEAD")
    _overwrite_loose_git_object(
        fixture.policy_root,
        root_tree_id,
        "tree",
        replacement,
    )

    class ForbiddenPipeline:
        def __init__(self, **_kwargs):
            raise AssertionError("corrupt root tree must fail before validation")

    monkeypatch.setattr(
        "axiom_encode.notary_verification.ValidatorPipeline",
        ForbiddenPipeline,
    )
    receipt_out = tmp_path / "corrupt-root-tree.json"
    with pytest.raises(
        NotaryVerificationError,
        match="hash mismatch|Git tree payload does not match its object identity",
    ):
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


def test_corrupt_loose_git_nested_tree_cannot_substitute_snapshot_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    nested_path = _MODULE_RELATIVE.parent.as_posix()
    nested_tree_id = _git(
        fixture.policy_root,
        "rev-parse",
        f"HEAD:{nested_path}",
    ).stdout.strip()
    replacement_rulespec = _RULESPEC + "# corrupt nested tree payload\n"
    fixture.module.write_text(replacement_rulespec, encoding="utf-8")
    _git(fixture.policy_root, "add", _MODULE_RELATIVE.as_posix())
    replacement_root_id = _git(
        fixture.policy_root,
        "write-tree",
    ).stdout.strip()
    replacement_nested_id = _git(
        fixture.policy_root,
        "rev-parse",
        f"{replacement_root_id}:{nested_path}",
    ).stdout.strip()
    replacement = _git_bytes(
        fixture.policy_root,
        "cat-file",
        "tree",
        replacement_nested_id,
    )
    _git(fixture.policy_root, "reset", "--hard", "-q", "HEAD")
    _overwrite_loose_git_object(
        fixture.policy_root,
        nested_tree_id,
        "tree",
        replacement,
    )

    class ForbiddenPipeline:
        def __init__(self, **_kwargs):
            raise AssertionError("corrupt nested tree must fail before validation")

    monkeypatch.setattr(
        "axiom_encode.notary_verification.ValidatorPipeline",
        ForbiddenPipeline,
    )
    receipt_out = tmp_path / "corrupt-nested-tree.json"
    with pytest.raises(
        NotaryVerificationError,
        match="hash mismatch|Git tree payload does not match its object identity",
    ):
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

    assert result.passed is True
    assert result.receipt["status"] == "passed-reduced"
    assert result.receipt["dependencies"]["policyengine_oracle"] is None
    assert len(reviewer_calls) == 4
    assert _gate(result.receipt, "policyengine-oracle") == {
        "gate": "policyengine-oracle",
        "status": "reduced",
        "reproducibility": "restricted-pinned",
    }
    for gate in (
        "compile",
        "proof-revalidation",
        "companion-tests",
        "grounding-contract",
        "layout-inspection",
        "waiver-set-verification",
        "policy-repo-nonmutation",
    ):
        assert _gate(result.receipt, gate)["status"] == "passed"
    assert json.loads(reduced_receipt.read_bytes()) == result.receipt
    assert reduced_receipt.read_bytes() == canonical_receipt_bytes(result.receipt)


def test_present_pinned_oracle_runs_and_passes_strict_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(
        tmp_path,
        rulespec=_ORACLE_RULESPEC,
        companion=_ORACLE_COMPANION,
    )
    release = _bind_oracle_corpus(fixture)
    _write_fake_oracle_engine(fixture.engine_root)
    _commit_all(fixture.engine_root, "oracle engine fixture")
    runtime_root, runtime_commit = _write_executable_oracle_runtime(
        tmp_path,
        monkeypatch,
    )
    runtime_pin = fixture.policy_root / ".axiom/policyengine-runtime.toml"
    runtime_pin.write_text(
        "[policyengine_runtime]\n"
        f'schema = "{POLICYENGINE_RUNTIME_PIN_SCHEMA}"\n'
        f'git_commit = "{runtime_commit}"\n',
        encoding="utf-8",
    )
    _commit_all(fixture.policy_root, "pin executable oracle fixture")

    reviewer_calls: list[str] = []

    def passing_reviewer(prompt: str, **_kwargs):
        reviewer_calls.append(prompt)
        return '{"score":10,"passed":true,"issues":[]}', 0

    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline.run_claude_code",
        passing_reviewer,
    )
    receipt_out = tmp_path / "oracle-receipt.json"

    result = run_notary_verification(
        policy_repo_path=fixture.policy_root,
        corpus_path=fixture.corpus_root,
        axiom_rules_engine_path=fixture.engine_root,
        policyengine_runtime_root=runtime_root,
        receipt_out=receipt_out,
        changed_files=[_MODULE_RELATIVE],
        now=_FIXED_NOW,
    )

    assert result.passed is True
    assert result.issues == ()
    assert result.receipt["status"] == "passed"
    assert len(reviewer_calls) == 4
    assert _gate(result.receipt, "policyengine-oracle") == {
        "gate": "policyengine-oracle",
        "status": "passed",
        "reproducibility": "restricted-pinned",
    }
    runtime_identity = result.receipt["dependencies"]["policyengine_oracle"]
    assert runtime_identity["schema"] == POLICYENGINE_RUNTIME_SCHEMA
    assert runtime_identity["country"] == "us"
    assert runtime_identity["trusted_git_commit"] == runtime_commit
    assert runtime_identity["locked_versions"] == {
        "policyengine-core": "3.4.5",
        "policyengine-us": "1.2.3",
    }
    assert result.receipt["dependencies"]["corpus_release"] == {
        "name": _RELEASE_NAME,
        "content_sha256": release.content_sha256,
    }
    assert receipt_out.read_bytes() == canonical_receipt_bytes(result.receipt)
    assert _git(fixture.policy_root, "status", "--porcelain").stdout == ""
    assert _git(runtime_root, "status", "--porcelain").stdout == ""


def test_present_failing_oracle_never_degrades_with_allow_reduced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(
        tmp_path,
        rulespec=_ORACLE_RULESPEC,
        companion=_ORACLE_COMPANION,
    )
    _bind_oracle_corpus(fixture)
    _write_fake_oracle_engine(fixture.engine_root)
    _commit_all(fixture.engine_root, "mismatching oracle engine fixture")
    runtime_root, runtime_commit = _write_executable_oracle_runtime(
        tmp_path,
        monkeypatch,
        parameter_value=60.0,
    )
    (fixture.policy_root / ".axiom/policyengine-runtime.toml").write_text(
        "[policyengine_runtime]\n"
        f'schema = "{POLICYENGINE_RUNTIME_PIN_SCHEMA}"\n'
        f'git_commit = "{runtime_commit}"\n',
        encoding="utf-8",
    )
    _commit_all(fixture.policy_root, "pin mismatching oracle fixture")

    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline.run_claude_code",
        lambda _prompt, **_kwargs: (
            '{"score":10,"passed":true,"issues":[]}',
            0,
        ),
    )
    receipt_out = tmp_path / "failed-oracle-receipt.json"
    result = run_notary_verification(
        policy_repo_path=fixture.policy_root,
        corpus_path=fixture.corpus_root,
        axiom_rules_engine_path=fixture.engine_root,
        policyengine_runtime_root=runtime_root,
        receipt_out=receipt_out,
        changed_files=[_MODULE_RELATIVE],
        allow_reduced=True,
        now=_FIXED_NOW,
    )

    assert result.passed is False
    assert result.receipt["status"] == "failed"
    assert result.receipt["dependencies"]["policyengine_oracle"] is not None
    assert _gate(result.receipt, "policyengine-oracle") == {
        "gate": "policyengine-oracle",
        "status": "failed",
        "reproducibility": "restricted-pinned",
    }
    assert all(gate["status"] != "reduced" for gate in result.receipt["gates"])
    assert receipt_out.read_bytes() == canonical_receipt_bytes(result.receipt)


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


def test_receipt_cannot_overwrite_engine_dependency(tmp_path: Path):
    fixture = _write_notary_fixture(tmp_path)
    engine_binary = fixture.engine_root / "target/debug/axiom-rules-engine"
    before = engine_binary.read_bytes()

    with pytest.raises(NotaryVerificationError, match="outside every verification"):
        run_notary_verification(
            policy_repo_path=fixture.policy_root,
            corpus_path=fixture.corpus_root,
            axiom_rules_engine_path=fixture.engine_root,
            receipt_out=engine_binary,
            changed_files=[_MODULE_RELATIVE],
            allow_reduced=True,
            now=_FIXED_NOW,
        )

    assert engine_binary.read_bytes() == before


def test_receipt_policy_case_alias_is_rejected_when_filesystem_has_aliases(
    tmp_path: Path,
):
    fixture = _write_notary_fixture(tmp_path)
    alias_root = fixture.policy_root.with_name(fixture.policy_root.name.swapcase())
    if not alias_root.exists() or not alias_root.samefile(fixture.policy_root):
        pytest.skip("filesystem is case-sensitive")

    receipt_out = alias_root / "notary-case-alias.json"
    with pytest.raises(NotaryVerificationError, match="outside every verification"):
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


def test_receipt_cannot_write_linked_worktree_git_metadata(tmp_path: Path):
    fixture = _write_notary_fixture(tmp_path)
    linked = tmp_path / "linked" / "rulespec-us"
    linked.parent.mkdir()
    _git(
        fixture.policy_root,
        "worktree",
        "add",
        "--detach",
        str(linked),
        "HEAD",
    )
    git_dir = Path(
        _git(linked, "rev-parse", "--absolute-git-dir").stdout.strip()
    )
    receipt_out = git_dir / "notary-receipt.json"

    with pytest.raises(NotaryVerificationError, match="Git metadata"):
        run_notary_verification(
            policy_repo_path=linked,
            corpus_path=fixture.corpus_root,
            axiom_rules_engine_path=fixture.engine_root,
            receipt_out=receipt_out,
            changed_files=[_MODULE_RELATIVE],
            allow_reduced=True,
            now=_FIXED_NOW,
        )

    assert not receipt_out.exists()


def test_receipt_parent_symlink_swap_cannot_write_into_policy_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    receipt_parent = tmp_path / "receipt-parent"
    displaced_parent = tmp_path / "displaced-receipt-parent"
    receipt_parent.mkdir()
    receipt_out = receipt_parent / "notary-receipt.json"
    original_validate = ValidatorPipeline.validate
    swapped = False

    def swapping_validate(self, path: Path, *, skip_reviewers: bool = False):
        nonlocal swapped
        if not swapped:
            receipt_parent.rename(displaced_parent)
            receipt_parent.symlink_to(fixture.policy_root, target_is_directory=True)
            swapped = True
        return original_validate(self, path, skip_reviewers=skip_reviewers)

    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline.run_claude_code",
        lambda _prompt, **_kwargs: (
            '{"score":10,"passed":true,"issues":[]}',
            0,
        ),
    )
    monkeypatch.setattr(ValidatorPipeline, "validate", swapping_validate)

    with pytest.raises(NotaryVerificationError, match="contains a symlink"):
        run_notary_verification(
            policy_repo_path=fixture.policy_root,
            corpus_path=fixture.corpus_root,
            axiom_rules_engine_path=fixture.engine_root,
            receipt_out=receipt_out,
            changed_files=[_MODULE_RELATIVE],
            allow_reduced=True,
            now=_FIXED_NOW,
        )

    assert swapped is True
    assert not (fixture.policy_root / receipt_out.name).exists()
    assert not (displaced_parent / receipt_out.name).exists()
    assert _git(fixture.policy_root, "status", "--porcelain").stdout == ""


def test_receipt_parent_directory_swap_cannot_write_into_policy_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_notary_fixture(tmp_path)
    receipt_parent = tmp_path / "receipt-parent"
    displaced_parent = tmp_path / "displaced-receipt-parent"
    receipt_parent.mkdir()
    receipt_out = receipt_parent / "notary-receipt.json"
    original_open_parent = notary_module._open_receipt_parent
    swapped = False

    def swapping_open_parent(path: Path) -> int:
        nonlocal swapped
        if not swapped:
            receipt_parent.rename(displaced_parent)
            fixture.policy_root.rename(receipt_parent)
            swapped = True
        return original_open_parent(path)

    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline.run_claude_code",
        lambda _prompt, **_kwargs: (
            '{"score":10,"passed":true,"issues":[]}',
            0,
        ),
    )
    monkeypatch.setattr(
        notary_module,
        "_open_receipt_parent",
        swapping_open_parent,
    )

    try:
        with pytest.raises(NotaryVerificationError, match="protected verification"):
            run_notary_verification(
                policy_repo_path=fixture.policy_root,
                corpus_path=fixture.corpus_root,
                axiom_rules_engine_path=fixture.engine_root,
                receipt_out=receipt_out,
                changed_files=[_MODULE_RELATIVE],
                allow_reduced=True,
                now=_FIXED_NOW,
            )
    finally:
        if swapped:
            receipt_parent.rename(fixture.policy_root)
            displaced_parent.rename(receipt_parent)

    assert swapped is True
    assert not (fixture.policy_root / receipt_out.name).exists()
    assert not (receipt_parent / receipt_out.name).exists()
    assert _git(fixture.policy_root, "status", "--porcelain").stdout == ""


@pytest.mark.parametrize(
    ("issues", "coverage", "expected"),
    [
        (
            ["PolicyEngine unavailable for one admitted test case"],
            {"comparable": 1, "adapter_errors": 0, "unmapped": 0},
            "PolicyEngine unavailable",
        ),
        (
            [],
            {"comparable": 1, "adapter_errors": 1, "unmapped": 0},
            "adapter execution error",
        ),
        (
            [],
            {"comparable": 1, "unsupported": 1, "unmapped": 0},
            "unsupported output",
        ),
    ],
)
def test_oracle_gate_fails_closed_on_partial_execution_failure(
    issues: list[str],
    coverage: dict[str, int],
    expected: str,
):
    result = PipelineResult(
        results={
            "policyengine": ValidationResult(
                validator_name="policyengine",
                passed=True,
                score=1.0,
                issues=issues,
                details={"coverage": coverage},
            )
        },
        total_duration_ms=1,
        all_passed=True,
    )

    passed, oracle_issues = _oracle_passed([result])

    assert passed is False
    assert any(expected in issue for issue in oracle_issues)


def test_compile_gate_requires_ci_recompile_evidence():
    result = PipelineResult(
        results={
            "ci": ValidationResult(
                validator_name="ci",
                passed=True,
                details={"compile_passed": False},
            )
        },
        total_duration_ms=1,
        all_passed=True,
    )

    assert _ci_compile_passed([result]) is False


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
    waiver = fixture.policy_root / "known-validation-gaps.yaml"
    waiver.write_text(
        "validate_failures:\n"
        f"  {_MODULE_RELATIVE.as_posix()}:\n"
        "    active:\n"
        f'      fingerprint: "sha256:{"a" * 64}"\n'
        '      owner: "@AxiomNotary"\n'
        "      issue: "
        '"https://github.com/TheAxiomFoundation/axiom-encode/issues/1192"\n'
        '      expires: "2026-08-01"\n',
        encoding="utf-8",
    )
    waiver_sha256 = hashlib.sha256(waiver.read_bytes()).hexdigest()
    (fixture.policy_root / ".axiom/toolchain.toml").write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{_RELEASE_NAME}"\n'
        "axiom_corpus_release_content_sha256 = "
        f'"{fixture.corpus_content_sha256}"\n'
        "validation_waiver_set_sha256 = "
        f'"{waiver_sha256}"\n',
        encoding="utf-8",
    )
    _commit_all(fixture.policy_root, "active waiver receipt fixture")

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
    assert result.receipt["schema_status"] == "PROVISIONAL"
    assert (
        result.receipt["subject_commit"]
        == _git(fixture.policy_root, "rev-parse", "HEAD").stdout.strip()
    )
    assert (
        result.receipt["subject_tree"]
        == _git(fixture.policy_root, "rev-parse", "HEAD^{tree}").stdout.strip()
    )
    assert result.receipt["dependencies"]["corpus_release"] == {
        "name": _RELEASE_NAME,
        "content_sha256": fixture.corpus_content_sha256,
    }
    assert result.receipt["waiver_set"] == {
        "sha256": waiver_sha256,
        "count": 1,
    }
    engine_identity = result.receipt["dependencies"]["axiom_rules_engine"]
    assert (
        engine_identity["commit"]
        == _git(fixture.engine_root, "rev-parse", "HEAD").stdout.strip()
    )
    assert engine_identity["executable"] == {
        "path": "target/debug/axiom-rules-engine",
        "sha256": fixture.engine_executable_sha256,
        "size": len(
            (fixture.engine_root / "target/debug/axiom-rules-engine").read_bytes()
        ),
    }
    encoder_identity = result.receipt["dependencies"]["axiom_encode"]
    assert encoder_identity["package"] == "axiom-encode"
    assert encoder_identity["version"] == result.receipt["run"]["encoder_version"]
    expected_package = _deterministic_tree_identity(
        Path(run_notary_verification.__code__.co_filename).resolve().parent,
        excluded_directory_names=frozenset({"__pycache__"}),
    )
    assert encoder_identity["package_identity"] == {
        "tree_sha256": expected_package["tree_sha256"],
        "file_count": expected_package["file_count"],
    }
    assert result.receipt["run"] == {
        "encoder_version": encoder_identity["version"],
        "profile_id": NOTARY_PROFILE_ID,
        "timestamp": "2026-07-21T15:04:05Z",
    }
    expected_classes = {
        "subject-clean": "public",
        "corpus-release-binding": "public",
        "compile": "public",
        "proof-revalidation": "public",
        "companion-tests": "public",
        "grounding-contract": "public",
        "layout-inspection": "public",
        "waiver-set-verification": "public",
        "policy-repo-nonmutation": "public",
        "policyengine-oracle": "restricted-pinned",
        "rulespec-reviewer": "ci-attested",
        "formula-reviewer": "ci-attested",
        "parameter-reviewer": "ci-attested",
        "integration-reviewer": "ci-attested",
    }
    assert {
        gate["gate"]: gate["reproducibility"]
        for gate in result.receipt["gates"]
    } == expected_classes
    assert result.receipt["receipt_sha256"] == receipt_body_sha256(result.receipt)
