"""Focused tests for the strict validation-waiver CLI."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode import cli
from tests.release_object_fixtures import bind_test_corpus_release

TEST_CORPUS_RELEASE = "validation-waiver-test-release"


def _validator(*, passed: bool, issues=(), error=None):
    return SimpleNamespace(
        passed=passed,
        issues=list(issues),
        error=error,
        duration_ms=999,
        raw_output="volatile raw output",
        score=0.123,
    )


def _pipeline_result(*, passed: bool, issues=("broken",), error="broken"):
    return SimpleNamespace(
        all_passed=passed,
        results={
            "compile": _validator(
                passed=passed,
                issues=() if passed else issues,
                error=None if passed else error,
            ),
            "ci": _validator(passed=True),
        },
    )


class _FakePipeline:
    instances = []
    result = _pipeline_result(passed=False)
    validate_error = None

    def __init__(
        self,
        *,
        policy_repo_path,
        axiom_rules_path,
        enable_oracles,
        local_corpus_release,
        rulespec_dependency_roots=(),
    ):
        self.policy_repo_path = Path(policy_repo_path)
        self.axiom_rules_path = Path(axiom_rules_path)
        self.enable_oracles = enable_oracles
        self.local_corpus_release = local_corpus_release
        self.rulespec_dependency_roots = tuple(rulespec_dependency_roots)
        self.validated = []
        type(self).instances.append(self)

    def _axiom_rules_binary(self):
        return self.axiom_rules_path / "axiom-rules-engine"

    def _rulespec_compile_roots(self):
        return (
            self.policy_repo_path.parent.resolve(),
            *(path.resolve() for path in self.rulespec_dependency_roots),
        )

    def _rulespec_engine_env(self):
        return {
            "AXIOM_CORPUS_REPO": str(self.policy_repo_path.parent / "axiom-corpus"),
        }

    def validate(self, module, *, skip_reviewers):
        self.validated.append((Path(module), skip_reviewers))
        if self.validate_error is not None:
            raise self.validate_error
        return self.result


@pytest.fixture(autouse=True)
def _reset_fake_pipeline():
    _FakePipeline.instances = []
    _FakePipeline.result = _pipeline_result(passed=False)
    _FakePipeline.validate_error = None


@pytest.fixture(autouse=True)
def _serial_waiver_audit(monkeypatch):
    # These tests fake the validator pipeline in-process; worker processes
    # cannot see those patches, so pin the audit to its serial code path.
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "1")


def _engine(tmp_path: Path) -> Path:
    engine = tmp_path / "engine"
    engine.mkdir()
    (engine / "axiom-rules-engine").write_text("")
    return engine


def _bound_corpus(tmp_path: Path, rulespec_root: Path) -> Path:
    waiver = rulespec_root / "known-validation-gaps.yaml"
    if not waiver.exists():
        waiver.parent.mkdir(parents=True, exist_ok=True)
        waiver.write_text("validate_failures: {}\n", encoding="utf-8")
    waiver_sha256 = hashlib.sha256(waiver.read_bytes()).hexdigest()
    corpus = tmp_path / "axiom-corpus"
    provision = corpus / "data/corpus/provisions/us/statute/test-version.jsonl"
    provision.parent.mkdir(parents=True, exist_ok=True)
    provision.write_text(
        json.dumps(
            {
                "id": "test:us/statute/1",
                "citation_path": "us/statute/1",
                "body": "authoritative source",
                "jurisdiction": "us",
                "document_class": "statute",
                "version": "test-version",
                "source_path": "sources/us/statute/test",
                "source_as_of": "2026-01-01",
                "expression_date": "2026-01-01",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selector = corpus / "manifests" / "releases" / f"{TEST_CORPUS_RELEASE}.json"
    selector.parent.mkdir(parents=True, exist_ok=True)
    selector.write_text(
        json.dumps(
            {
                "name": TEST_CORPUS_RELEASE,
                "scopes": [
                    {
                        "jurisdiction": "us",
                        "document_class": "statute",
                        "version": "test-version",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    release = bind_test_corpus_release(
        corpus,
        TEST_CORPUS_RELEASE,
        [("us", "statute", "test-version")],
    )
    toolchain = rulespec_root / ".axiom" / "toolchain.toml"
    toolchain.parent.mkdir(parents=True, exist_ok=True)
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{TEST_CORPUS_RELEASE}"\n'
        f'axiom_corpus_release_content_sha256 = "{release.content_sha256}"\n'
        f'validation_waiver_set_sha256 = "{waiver_sha256}"\n',
        encoding="utf-8",
    )
    return corpus


def test_validate_outcome_keeps_all_failed_issues_and_excludes_volatile_fields():
    result = SimpleNamespace(
        all_passed=False,
        results={
            "compile": _validator(
                passed=False,
                issues=("first", "second"),
                error="compile error",
            ),
            "ci": _validator(passed=True),
        },
    )

    outcome = cli._validation_waiver_validate_outcome(result)

    assert outcome == {
        "passed": False,
        "validators": {
            "compile": {
                "passed": False,
                "issues": ["first", "second"],
                "error": "compile error",
            }
        },
    }
    assert "duration_ms" not in json.dumps(outcome)
    assert "raw_output" not in json.dumps(outcome)
    assert "score" not in json.dumps(outcome)


def test_inconsistent_pipeline_aggregate_is_fatal_infrastructure_error():
    result = SimpleNamespace(
        all_passed=False,
        results={"ci": _validator(passed=True)},
    )

    with pytest.raises(RuntimeError, match="inconsistent aggregate"):
        cli._validation_waiver_validate_outcome(result)


def test_fingerprint_reuses_pipeline_and_marks_missing_companion_explicit(
    tmp_path: Path,
):
    root = tmp_path / "rulespec-us"
    content_root = root / "us/statutes"
    content_root.mkdir(parents=True)
    for name in ("b.yaml", "a.yaml"):
        (content_root / name).write_text("format: rulespec/v1\n")
    engine = _engine(tmp_path)

    with patch.object(cli, "ValidatorPipeline", _FakePipeline):
        results = cli._fingerprint_validation_waiver_modules(
            [Path("us/statutes/b.yaml"), Path("us/statutes/a.yaml")],
            root=root,
            corpus_path=_bound_corpus(tmp_path, root),
            axiom_rules_path=engine,
        )

    assert len(_FakePipeline.instances) == 1
    assert [result["path"] for result in results] == [
        "us/statutes/a.yaml",
        "us/statutes/b.yaml",
    ]
    validation_pipelines = [
        pipeline for pipeline in _FakePipeline.instances if pipeline.validated
    ]
    assert len(validation_pipelines) == 1
    assert [path.name for path, skip in validation_pipelines[0].validated] == [
        "a.yaml",
        "b.yaml",
    ]
    assert all(skip is True for _, skip in validation_pipelines[0].validated)
    assert all(result["outcome"]["companion"]["present"] is False for result in results)
    assert all(result["outcome"]["companion"]["passed"] is True for result in results)
    assert all(result["fingerprint"].startswith("sha256:") for result in results)


def test_fingerprint_boundary_reuses_nested_pipeline_admissions(tmp_path: Path):
    from axiom_encode.harness import validator_pipeline

    root = tmp_path / "rulespec-us"
    modules = []
    for index in range(3):
        relative = Path(f"us/statutes/{index}.yaml")
        module = root / relative
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("format: rulespec/v1\nrules: []\n", encoding="utf-8")
        modules.append(relative)
    engine = _engine(tmp_path)
    walked_roots = []
    original_walk = os.walk

    def counting_walk(checkout, *walk_args, **walk_kwargs):
        walked_roots.append(Path(checkout))
        return original_walk(checkout, *walk_args, **walk_kwargs)

    class AdmissionPipeline(validator_pipeline.ValidatorPipeline):
        def _axiom_rules_binary(self):
            return self.axiom_rules_path / "axiom-rules-engine"

        def _validate_with_authoritative_roots(
            self,
            rulespec_file,
            skip_reviewers=False,
        ):
            validator_pipeline._rulespec_checkout_root_for_active_path(
                self.policy_repo_path
            )
            return _pipeline_result(passed=False)

    with (
        patch.object(cli, "ValidatorPipeline", AdmissionPipeline),
        patch.object(validator_pipeline.os, "walk", side_effect=counting_walk),
    ):
        results = cli._fingerprint_validation_waiver_modules(
            modules,
            root=root,
            corpus_path=_bound_corpus(tmp_path, root),
            axiom_rules_path=engine,
        )

    assert len(results) == 3
    # One symlink-audit walk up front plus one per module: each module's
    # validate opens a FRESH resolution scope (the shared batch scope leaked
    # per-module resolution state between neighbors and made waiver
    # fingerprints partition-dependent; observed on rulespec-us as 5/3,099
    # fingerprints changing between chunk layouts). This walk count pins the
    # per-module resolution audits; the wrapper's routing-only scope is
    # pinned directly by test_fingerprint_batch_holds_no_shared_resolution_scope.
    assert set(walked_roots) == {root}
    assert len(walked_roots) == 1 + len(results)


def test_fingerprint_passes_explicit_rulespec_dependency_roots(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    module = root / "us/statutes/a.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")
    dependency_root = tmp_path / "rulespec-uk"
    dependency_root.mkdir()

    with patch.object(cli, "ValidatorPipeline", _FakePipeline):
        cli._fingerprint_validation_waiver_modules(
            [Path("us/statutes/a.yaml")],
            root=root,
            corpus_path=_bound_corpus(tmp_path, root),
            axiom_rules_path=_engine(tmp_path),
            rulespec_dependency_roots=(dependency_root,),
        )

    assert _FakePipeline.instances[0].rulespec_dependency_roots == (dependency_root,)


def test_validation_and_fingerprint_use_exact_content_roots(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    modules = []
    for relative in ("us/statutes/1.yaml", "us-ca/regulations/1.yaml"):
        module = root / relative
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("format: rulespec/v1\n")
        modules.append(Path(relative))
    engine = _engine(tmp_path)

    with patch.object(cli, "ValidatorPipeline", _FakePipeline):
        cli._fingerprint_validation_waiver_modules(
            modules,
            root=root,
            corpus_path=_bound_corpus(tmp_path, root),
            axiom_rules_path=engine,
        )

    assert (
        cli._resolve_validation_repo_roots(root / modules[0], engine)[0] == root / "us"
    )
    assert (
        cli._resolve_validation_repo_roots(root / modules[1], engine)[0]
        == root / "us-ca"
    )
    assert {pipeline.policy_repo_path for pipeline in _FakePipeline.instances} == {
        root / "us",
        root / "us-ca",
    }


def test_validation_rejects_legacy_flat_checkout_layout(tmp_path: Path):
    module = tmp_path / "rulespec-us-ca/regulations/1.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")

    with pytest.raises(ValueError, match="canonical country checkout"):
        cli._canonical_validation_checkout_root(module)


def test_validation_accepts_jurisdiction_in_composition_checkout(tmp_path: Path):
    checkout = tmp_path / "rulespec-us"
    module = checkout / "us/statutes/1.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")
    (checkout / "programs/us/snap").mkdir(parents=True)

    assert cli._canonical_validation_checkout_root(module) == checkout


def test_validation_rejects_missing_explicit_engine_checkout(tmp_path: Path):
    module = tmp_path / "rulespec-us/us/statutes/1.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")

    with pytest.raises(ValueError, match="does not exist"):
        cli._resolve_validation_repo_roots(
            module,
            tmp_path / "missing-axiom-rules-engine",
        )


def test_validation_rejects_symlinked_explicit_engine_checkout(tmp_path: Path):
    module = tmp_path / "rulespec-us/us/statutes/1.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")
    engine = _engine(tmp_path)
    symlink = tmp_path / "engine-link"
    symlink.symlink_to(engine, target_is_directory=True)

    with pytest.raises(ValueError, match="contains a symlink"):
        cli._resolve_validation_repo_roots(module, symlink)


def test_independent_companion_matches_shared_test_command_root(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    module = root / "us-ca/regulations/1.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")
    module.with_name("1.test.yaml").write_text("cases: []\n")
    engine = _engine(tmp_path)
    companion_policy_roots = []

    def execute(*args, **kwargs):
        companion_policy_roots.append(kwargs["policy_repo_path"])
        return {"cases": 0, "compiled": 1, "failures": []}

    with (
        patch.object(cli, "ValidatorPipeline", _FakePipeline),
        patch.object(cli, "_execute_rulespec_test_file", side_effect=execute),
    ):
        cli._fingerprint_validation_waiver_modules(
            [Path("us-ca/regulations/1.yaml")],
            root=root,
            corpus_path=_bound_corpus(tmp_path, root),
            axiom_rules_path=engine,
        )

    # Validation, the ordinary companion command, and the independent audit
    # share the exact jurisdiction content root as their policy root.
    assert companion_policy_roots == [root / "us-ca"]


def test_fingerprint_rejects_symlinked_companion(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    module = root / "us/statutes/module.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")
    target = module.with_name("target.test.yaml")
    target.write_text("cases: []\n")
    module.with_name("module.test.yaml").symlink_to(target.name)

    with (
        patch.object(cli, "ValidatorPipeline", _FakePipeline),
        pytest.raises(ValueError, match="must not be a symlink"),
    ):
        cli._fingerprint_validation_waiver_modules(
            [Path("us/statutes/module.yaml")],
            root=root,
            corpus_path=_bound_corpus(tmp_path, root),
            axiom_rules_path=_engine(tmp_path),
        )


def test_companion_parse_failure_is_deterministic_and_paths_are_normalized(
    tmp_path: Path,
):
    root = tmp_path / "rulespec-us"
    module = root / "us/statutes/module.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")
    module.with_name("module.test.yaml").write_text("not: valid cases\n")
    engine = _engine(tmp_path)
    _FakePipeline.result = _pipeline_result(passed=True)

    def fail_parse(*args, **kwargs):
        raise ValueError(
            f"bad test under {root}; engine={engine}; tmp={kwargs['tmp_path']}"
        )

    with (
        patch.object(cli, "ValidatorPipeline", _FakePipeline),
        patch.object(cli, "_execute_rulespec_test_file", side_effect=fail_parse),
    ):
        result = cli._fingerprint_validation_waiver_modules(
            [Path("us/statutes/module.yaml")],
            root=root,
            corpus_path=_bound_corpus(tmp_path, root),
            axiom_rules_path=engine,
        )[0]

    serialized = json.dumps(result["outcome"])
    assert result["passed"] is False
    assert "ValueError: bad test" in serialized
    assert str(root) not in serialized
    assert str(engine) not in serialized
    assert "<repo>" in serialized
    assert "<engine>" in serialized
    assert "<tmp>" in serialized


def test_unexpected_pipeline_exception_is_not_fingerprintable(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    module = root / "us/statutes/module.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")
    engine = _engine(tmp_path)
    _FakePipeline.validate_error = RuntimeError("validator infrastructure broke")

    with (
        patch.object(cli, "ValidatorPipeline", _FakePipeline),
        pytest.raises(RuntimeError, match="infrastructure broke"),
    ):
        cli._fingerprint_validation_waiver_modules(
            [Path("us/statutes/module.yaml")],
            root=root,
            corpus_path=_bound_corpus(tmp_path, root),
            axiom_rules_path=engine,
        )


@pytest.mark.parametrize(
    "module_path",
    [
        "module.yaml",
        "us/statutes/module.yml",
        "us/statutes/module.test.yaml",
    ],
)
def test_fingerprint_rejects_paths_the_waiver_schema_cannot_store(
    tmp_path: Path,
    module_path: str,
):
    root = tmp_path / "rulespec-us"
    module = root / module_path
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("format: rulespec/v1\n")

    with pytest.raises(
        cli._validation_waivers.WaiverSchemaError,
        match="unsafe validate_failures path",
    ):
        cli._fingerprint_validation_waiver_modules(
            [Path(module_path)],
            root=root,
            corpus_path=_bound_corpus(tmp_path, root),
            axiom_rules_path=_engine(tmp_path),
        )


def _metadata(fingerprint: str):
    return SimpleNamespace(fingerprint=fingerprint)


def _entry(path: str, *, active=None, pending=None):
    return SimpleNamespace(path=path, active=active, pending=pending)


def _waiver_set(entries):
    return SimpleNamespace(
        entries=entries,
        active_paths={path for path, entry in entries.items() if entry.active},
        pending_paths={path for path, entry in entries.items() if entry.pending},
    )


def _audit_args(
    root: Path,
    protected_base: Path,
    changed_paths: Path,
    corpus_path: Path,
):
    return SimpleNamespace(
        root=root,
        corpus_path=corpus_path,
        waivers=None,
        protected_base=protected_base,
        changed_paths=changed_paths,
        axiom_rules_path=None,
        json=True,
    )


@pytest.mark.parametrize(
    "value",
    [
        "us/statutes/a.yaml\nsecond",
        "us/statutes/\x7fa.yaml",
        "us/statutes/\u202ea.yaml",
        "us/statutes/\u2028a.yaml",
    ],
)
def test_transition_paths_reject_control_characters(value: str):
    with pytest.raises(ValueError, match="control characters"):
        cli._validation_waiver_repo_relative_path(value, label="changed path")


def test_changed_paths_rejects_unicode_separator_before_deduplication(
    tmp_path: Path,
):
    changed_paths = tmp_path / "changed.txt"
    changed_paths.write_text(
        "known-validation-gaps.yaml\nknown-validation-gaps.yaml\u2028\n"
    )

    with pytest.raises(ValueError, match="control characters"):
        cli._validation_waiver_changed_paths(changed_paths)


def test_audit_checks_active_pending_and_removed_waivers(tmp_path: Path, capsys):
    root = tmp_path / "rulespec-us"
    root.mkdir()
    (root / "known-validation-gaps.yaml").write_text("validate_failures: {}\n")
    for path in ("active.yaml", "both.yaml", "pending.yaml", "removed.yaml"):
        (root / path).write_text("format: rulespec/v1\n")
    base_file = tmp_path / "base.yaml"
    base_file.write_text("validate_failures: {}\n")
    changed_file = tmp_path / "changed.txt"
    changed_file.write_text("known-validation-gaps.yaml\n")

    head = _waiver_set(
        {
            "active.yaml": _entry("active.yaml", active=_metadata("sha256:active")),
            "both.yaml": _entry(
                "both.yaml",
                active=_metadata("sha256:both"),
                pending=_metadata("sha256:future"),
            ),
            "pending.yaml": _entry("pending.yaml", pending=_metadata("sha256:pending")),
        }
    )
    base = _waiver_set(
        {
            "active.yaml": _entry("active.yaml", active=_metadata("sha256:active")),
            "both.yaml": _entry("both.yaml", active=_metadata("sha256:both")),
            "removed.yaml": _entry("removed.yaml", active=_metadata("sha256:removed")),
        }
    )
    executed = [
        {
            "path": "active.yaml",
            "passed": False,
            "fingerprint": "sha256:active",
            "outcome": {},
        },
        {
            "path": "both.yaml",
            "passed": False,
            "fingerprint": "sha256:both",
            "outcome": {},
        },
        {
            "path": "pending.yaml",
            "passed": True,
            "fingerprint": "sha256:passing",
            "outcome": {},
        },
        {
            "path": "removed.yaml",
            "passed": True,
            "fingerprint": "sha256:passing",
            "outcome": {},
        },
    ]

    with (
        patch.object(
            cli._validation_waivers,
            "load_validation_waivers",
            side_effect=[head, base],
        ),
        patch.object(
            cli._validation_waivers,
            "protected_base_transition_issues",
            return_value=(),
        ),
        patch.object(
            cli,
            "_fingerprint_validation_waiver_modules",
            return_value=executed,
        ),
    ):
        exit_code = cli._cmd_validation_waivers_audit(
            _audit_args(
                root,
                base_file,
                changed_file,
                _bound_corpus(tmp_path, root),
            )
        )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["success"] is True
    assert {item["kind"] for item in report["results"]} == {
        "active",
        "pending",
        "removed",
    }
    both = next(item for item in report["results"] if item["path"] == "both.yaml")
    assert both["expected_fingerprint"] == "sha256:both"


@pytest.mark.parametrize(
    ("kind", "passed", "actual_fingerprint", "expected_error"),
    [
        ("active", True, "sha256:passing", "active validation waiver is stale"),
        ("active", False, "sha256:changed", "failure fingerprint drifted"),
        ("pending", False, "sha256:failing", "pending waiver requires"),
    ],
)
def test_audit_rejects_invalid_runtime_waiver_state(
    tmp_path: Path,
    capsys,
    kind: str,
    passed: bool,
    actual_fingerprint: str,
    expected_error: str,
):
    root = tmp_path / "rulespec-us"
    module_path = "us/statutes/module.yaml"
    module = root / module_path
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")
    (root / "known-validation-gaps.yaml").write_text("validate_failures: {}\n")
    base_file = tmp_path / "base.yaml"
    base_file.write_text("validate_failures: {}\n")
    changed_file = tmp_path / "changed.txt"
    changed_file.write_text("")

    expected_fingerprint = "sha256:expected"
    entry = _entry(
        module_path,
        active=_metadata(expected_fingerprint) if kind == "active" else None,
        pending=_metadata(expected_fingerprint) if kind == "pending" else None,
    )
    waivers = _waiver_set({module_path: entry})
    executed = [
        {
            "path": module_path,
            "passed": passed,
            "fingerprint": actual_fingerprint,
            "outcome": {},
        }
    ]

    with (
        patch.object(
            cli._validation_waivers,
            "load_validation_waivers",
            side_effect=[waivers, waivers],
        ),
        patch.object(
            cli._validation_waivers,
            "protected_base_transition_issues",
            return_value=(),
        ),
        patch.object(
            cli,
            "_fingerprint_validation_waiver_modules",
            return_value=executed,
        ),
    ):
        exit_code = cli._cmd_validation_waivers_audit(
            _audit_args(
                root,
                base_file,
                changed_file,
                _bound_corpus(tmp_path, root),
            )
        )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["success"] is False
    assert len(report["errors"]) == 1
    assert expected_error in report["errors"][0]


def test_removed_waiver_must_pass_unless_module_was_deleted(tmp_path: Path, capsys):
    root = tmp_path / "rulespec-us"
    root.mkdir()
    (root / "known-validation-gaps.yaml").write_text("validate_failures: {}\n")
    (root / "still-broken.yaml").write_text("format: rulespec/v1\n")
    base_file = tmp_path / "base.yaml"
    base_file.write_text("validate_failures: {}\n")
    changed_file = tmp_path / "changed.txt"
    changed_file.write_text("known-validation-gaps.yaml\n")
    head = _waiver_set({})
    base = _waiver_set(
        {
            "still-broken.yaml": _entry(
                "still-broken.yaml", active=_metadata("sha256:old")
            ),
            "deleted.yaml": _entry("deleted.yaml", active=_metadata("sha256:deleted")),
        }
    )
    executed = [
        {
            "path": "still-broken.yaml",
            "passed": False,
            "fingerprint": "sha256:old",
            "outcome": {},
        }
    ]
    with (
        patch.object(
            cli._validation_waivers,
            "load_validation_waivers",
            side_effect=[head, base],
        ),
        patch.object(
            cli._validation_waivers,
            "protected_base_transition_issues",
            return_value=(),
        ),
        patch.object(
            cli,
            "_fingerprint_validation_waiver_modules",
            return_value=executed,
        ),
    ):
        exit_code = cli._cmd_validation_waivers_audit(
            _audit_args(
                root,
                base_file,
                changed_file,
                _bound_corpus(tmp_path, root),
            )
        )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert any("still-broken.yaml" in error for error in report["errors"])
    deleted = next(item for item in report["results"] if item["path"] == "deleted.yaml")
    assert deleted == {
        "path": "deleted.yaml",
        "kind": "removed",
        "success": True,
        "module_deleted": True,
    }


def test_active_paths_emits_active_entries_only(tmp_path: Path, capsys):
    waivers = SimpleNamespace(
        active_paths=frozenset({"us/statutes/26/1.yaml", "us/statutes/26/2.yaml"}),
        pending_paths=frozenset({"us/statutes/26/3.yaml"}),
    )
    args = SimpleNamespace(root=tmp_path, waivers=None, json=False)

    with patch.object(
        cli._validation_waivers,
        "load_validation_waivers",
        return_value=waivers,
    ) as loader:
        exit_code = cli._cmd_validation_waivers_active_paths(args)

    assert exit_code == 0
    assert loader.call_args.args[0] == tmp_path / "known-validation-gaps.yaml"
    assert capsys.readouterr().out.splitlines() == [
        "us/statutes/26/1.yaml",
        "us/statutes/26/2.yaml",
    ]


def test_audit_requires_protected_base_and_changed_paths(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    root.mkdir()
    (root / "known-validation-gaps.yaml").write_text("validate_failures: {}\n")
    args = SimpleNamespace(
        root=root,
        corpus_path=_bound_corpus(tmp_path, root),
        protected_base=None,
        changed_paths=None,
        axiom_rules_path=None,
        json=True,
    )

    with pytest.raises(ValueError, match="are required"):
        cli._cmd_validation_waivers_audit(args)


def test_audit_cli_registers_partition_pair(tmp_path: Path):
    with (
        patch(
            "sys.argv",
            [
                "axiom-encode",
                "validation-waivers",
                "audit",
                "--root",
                str(tmp_path),
                "--corpus-path",
                str(tmp_path),
                "--protected-base",
                str(tmp_path / "base.yaml"),
                "--changed-paths",
                str(tmp_path / "changed.txt"),
                "--axiom-rules-engine-path",
                str(tmp_path),
                "--partition-key",
                "us-ak",
                "--partition-keys-json",
                '["us", "us-ak"]',
            ],
        ),
        patch.object(cli, "cmd_validation_waivers") as command,
    ):
        cli.main()

    args = command.call_args.args[0]
    assert args.partition_key == "us-ak"
    assert args.partition_keys_json == '["us", "us-ak"]'


@pytest.mark.parametrize(
    "argv",
    [
        ["validation-waivers", "audit"],
        ["validation-waivers", "active-paths", "--waivers", "alternate.yaml"],
    ],
)
def test_cli_rejects_missing_transition_inputs_and_alternate_waiver_files(
    argv: list[str],
):
    with (
        patch("sys.argv", ["axiom-encode", *argv]),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.main()

    assert exc_info.value.code == 2


def test_main_registers_validation_waiver_subcommands(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    root.mkdir()
    corpus = _bound_corpus(tmp_path, root)
    with (
        patch(
            "sys.argv",
            [
                "axiom-encode",
                "validation-waivers",
                "fingerprint",
                "module.yaml",
                "--root",
                str(root),
                "--corpus-path",
                str(corpus),
                "--axiom-rules-engine-path",
                str(tmp_path / "axiom-rules-engine"),
                "--json",
            ],
        ),
        patch.object(cli, "cmd_validation_waivers") as command,
    ):
        cli.main()

    args = command.call_args.args[0]
    assert args.validation_waivers_command == "fingerprint"
    assert args.modules == [Path("module.yaml")]
    assert args.root == root
    assert args.corpus_path == corpus
    assert args.axiom_rules_path == tmp_path / "axiom-rules-engine"
    assert args.json is True
