"""Tests for model comparison eval helpers."""

import hashlib
import inspect
import json
import os
import subprocess
import tempfile
import threading
import uuid
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import requests
import yaml

from axiom_encode import corpus_resolver
from axiom_encode.corpus_resolver import (
    InvalidCorpusCitationError,
    LocalCorpusRelease,
    resolve_scoped_local_corpus_source,
)
from axiom_encode.harness import evals as evals_module
from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.dependency_stubs import UnsafeRulespecContextPath
from axiom_encode.harness.encoding_db import TokenUsage
from axiom_encode.harness.eval_evidence import (
    APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV,
    EVAL_EVIDENCE_PRIVATE_KEY_ENV,
    sign_eval_evidence,
)
from axiom_encode.harness.evals import (
    CorpusAmendmentDocument,
    CorpusSourceUnit,
    EvalArtifactMetrics,
    EvalContextFile,
    EvalPromptResponse,
    EvalReadinessGates,
    EvalResult,
    EvalSuiteCase,
    EvalSuiteManifest,
    EvalWorkspace,
    GroundingMetric,
    ValidationRetryCandidate,
    _bind_eval_result_payload,
    _build_empty_artifact_retry_prompt,
    _build_eval_prompt,
    _build_eval_suite_execution_identity,
    _candidate_import_rule_files,
    _canonical_rulespec_target_for_path,
    _canonical_target_ref_prefix,
    _clean_generated_file_content,
    _clear_eval_target_artifacts,
    _codex_prompt_timeouts,
    _command_looks_out_of_bounds,
    _command_uses_policyengine_skill,
    _contained_eval_output_file,
    _context_file_executable_surfaces,
    _context_import_target,
    _eval_result_from_payload,
    _eval_review_contract_manifest_payload,
    _eval_suite_execution_identity_sha256,
    _eval_suite_rulespec_roots,
    _evaluate_generated_artifact_with_repairs,
    _expected_eval_source_attestation,
    _format_existing_target_contract_guidance,
    _format_subparagraph_coverage_checklist,
    _hydrate_eval_root,
    _imported_named_scalar_occurrences,
    _is_single_amount_table_slice,
    _materialize_eval_artifact,
    _normalize_nonannual_test_period_value,
    _normalize_test_case_value,
    _normalize_test_periods_to_effective_dates,
    _numeric_occurrence_source_text,
    _policyengine_hint_upstream_composition_issues,
    _post_openai_eval_request,
    _prepare_codex_eval_home,
    _prompt_corpus_citation_path,
    _repo_augmented_context_root,
    _resolve_context_imports,
    _resolve_eval_output_path,
    _reviewer_independent_metrics,
    _rulespec_validation_target,
    _run_claude_prompt_eval,
    _run_codex_prompt_eval,
    _secure_eval_read,
    _select_cross_section_context_files,
    _shallowest_active_source_path_row,
    _slugify,
    _source_identifier_to_relative_rulespec_path,
    _source_metadata_citation_path,
    _target_rel_for_eval_identifier,
    _target_source_scope_for_heuristics,
    _validate_eval_result_artifacts,
    _validate_eval_suite_execution_identity,
    _validation_policy_repo_root,
    _validation_rulespec_dependency_roots,
    _wait_for_codex_process,
    evaluate_artifact,
    find_admin_agency_aggregate_entity_issues,
    load_eval_suite_manifest,
    parse_runner_spec,
    prepare_eval_workspace,
    resolve_corpus_source_unit,
    run_eval_suite,
    run_model_eval,
    run_source_eval,
    select_context_files,
    summarize_readiness,
)
from axiom_encode.harness.policyengine_runtime import (
    PolicyEngineRuntime,
    PolicyEngineRuntimeError,
)
from axiom_encode.harness.pricing import estimate_usage_cost_usd
from axiom_encode.harness.validator_pipeline import (
    ValidationResult,
    ValidatorPipeline,
    find_test_input_assignment_issues,
)
from axiom_encode.legacy_replacement import (
    LegacyReplacementContract,
    LegacyReplacementFile,
    LegacyReplacementRetainedSuccessor,
    LegacyReplacementRewrite,
)
from axiom_encode.repo_routing import find_policy_repo_root, monorepo_checkout_name
from axiom_encode.signing_broker import get_signing_broker
from axiom_encode.statute import CitationParts, citation_to_relative_rulespec_path
from tests.eval_evidence_fixtures import (
    install_test_eval_evidence_keys,
)
from tests.release_object_fixtures import bind_test_corpus_release

_TEST_POLICYENGINE_RUNTIME_IDENTITY = {
    "schema": "axiom-policyengine-runtime/v2",
    "country": "us",
    "repository_root": "/tmp/policyengine-us",
}
_TEST_POLICYENGINE_RUNTIME_IDENTITY_SHA256 = hashlib.sha256(
    json.dumps(
        _TEST_POLICYENGINE_RUNTIME_IDENTITY,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def _test_policyengine_runtime(country: str = "us") -> PolicyEngineRuntime:
    root = Path(f"/tmp/policyengine-{country}")
    identity = {
        "schema": "axiom-policyengine-runtime/v2",
        "country": country,
        "repository_root": str(root),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    runtime = object.__new__(PolicyEngineRuntime)
    object.__setattr__(runtime, "root", root)
    object.__setattr__(runtime, "country", country)
    object.__setattr__(runtime, "python_path", root / ".venv" / "bin" / "python")
    object.__setattr__(
        runtime,
        "site_packages_path",
        root / ".venv" / "lib" / "python3.13" / "site-packages",
    )
    object.__setattr__(
        runtime, "rulespec_checkout_root", Path(f"/tmp/rulespec-{country}")
    )
    object.__setattr__(runtime, "identity", identity)
    object.__setattr__(
        runtime,
        "identity_sha256",
        hashlib.sha256(canonical.encode()).hexdigest(),
    )
    return runtime


def _test_eval_suite_execution_identity() -> dict[str, object]:
    """Build an execution identity without depending on ambient git state."""

    with patch(
        "axiom_encode.harness.evals._git_checkout_execution_identity",
        side_effect=lambda *_args, **_kwargs: {
            "kind": "tree",
            "tree_sha256": "1" * 64,
        },
    ):
        return _build_eval_suite_execution_identity(Path("/tmp/axiom-rules"), ())


@pytest.fixture(autouse=True)
def _mock_generalist_reviewer(monkeypatch):
    """Keep eval tests deterministic unless they explicitly inspect reviewer behavior."""
    install_test_eval_evidence_keys(monkeypatch)
    monkeypatch.delenv(APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV, raising=False)
    with patch.object(
        ValidatorPipeline,
        "_run_reviewer",
        return_value=ValidationResult(
            "generalist-reviewer", True, score=8.0, issues=[]
        ),
    ):
        yield


_TEST_CORPUS_RELEASE_NAME = "eval-test-release"
_TEST_CORPUS_VERSION = "2026-eval-test"
_DK_FULL_PARITY_AMENDMENT_FIXTURE = (
    Path(__file__).parent / "fixtures" / "dk_full_parity_amendment_discovery.json"
)
_DK_FULL_PARITY_CORPUS_SHA256 = (
    "5c348870772b2e79cc10a978f63380ec4e814c9ed13e570930fcf4c5643eaca0"
)


def _canonical_rulespec_content_root(base: Path, jurisdiction: str) -> Path:
    """Create and return ``rulespec-<country>/<jurisdiction>``."""

    checkout = base / monorepo_checkout_name(jurisdiction)
    content_root = checkout / jurisdiction
    content_root.mkdir(parents=True, exist_ok=True)
    return content_root


def _generated_rulespec_file_path(base: Path, relative: str) -> Path:
    """Create a generated-artifact path with a canonical RuleSpec surface."""

    path = base / "generated" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_test_corpus_release(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    selected_scopes: list[tuple[str, str, str]] | None = None,
) -> LocalCorpusRelease:
    """Write canonical corpus rows and bind them to one named release."""

    corpus_root = tmp_path / "axiom-corpus"
    grouped_rows: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for raw_row in rows:
        citation_path = raw_row.get("citation_path")
        assert isinstance(citation_path, str)
        citation_parts = citation_path.split("/")
        assert len(citation_parts) >= 3
        jurisdiction = str(raw_row.get("jurisdiction") or citation_parts[0])
        document_class = str(raw_row.get("document_class") or citation_parts[1])
        version = str(raw_row.get("version") or _TEST_CORPUS_VERSION)
        row = {
            "id": f"test:{version}:{citation_path}",
            "citation_path": citation_path,
            "jurisdiction": jurisdiction,
            "document_class": document_class,
            "version": version,
            "source_path": f"sources/{jurisdiction}/{document_class}/{version}",
            "source_as_of": "2026-01-01",
            "expression_date": "2026-01-01",
            **raw_row,
        }
        grouped_rows.setdefault((jurisdiction, document_class, version), []).append(row)

    for (jurisdiction, document_class, version), scope_rows in grouped_rows.items():
        provision_file = (
            corpus_root
            / "data"
            / "corpus"
            / "provisions"
            / jurisdiction
            / document_class
            / f"{version}.jsonl"
        )
        provision_file.parent.mkdir(parents=True, exist_ok=True)
        provision_file.write_text(
            "".join(json.dumps(row) + "\n" for row in scope_rows),
            encoding="utf-8",
        )

    scopes = selected_scopes or sorted(grouped_rows)
    selector = (
        corpus_root / "manifests" / "releases" / f"{_TEST_CORPUS_RELEASE_NAME}.json"
    )
    selector.parent.mkdir(parents=True, exist_ok=True)
    selector.write_text(
        json.dumps(
            {
                "name": _TEST_CORPUS_RELEASE_NAME,
                "scopes": [
                    {
                        "jurisdiction": jurisdiction,
                        "document_class": document_class,
                        "version": version,
                    }
                    for jurisdiction, document_class, version in scopes
                ],
            }
        ),
        encoding="utf-8",
    )
    release = bind_test_corpus_release(
        corpus_root,
        _TEST_CORPUS_RELEASE_NAME,
        list(scopes),
    )
    waiver_bytes = b"validate_failures: {}\n"
    waiver_digest = hashlib.sha256(waiver_bytes).hexdigest()
    for jurisdiction, _document_class, _version in grouped_rows:
        rulespec_checkout = tmp_path / monorepo_checkout_name(jurisdiction)
        (rulespec_checkout / jurisdiction).mkdir(parents=True, exist_ok=True)
        (rulespec_checkout / "known-validation-gaps.yaml").write_bytes(waiver_bytes)
        toolchain_path = rulespec_checkout / ".axiom" / "toolchain.toml"
        toolchain_path.parent.mkdir(parents=True, exist_ok=True)
        toolchain_path.write_text(
            "[toolchain]\n"
            f'axiom_corpus_release = "{_TEST_CORPUS_RELEASE_NAME}"\n'
            f'axiom_corpus_release_content_sha256 = "{release.content_sha256}"\n'
            f'validation_waiver_set_sha256 = "{waiver_digest}"\n',
            encoding="utf-8",
        )
    return release


def _write_test_corpus_provision(
    tmp_path: Path,
    citation_path: str = "us/statute/7/2017",
    body: str = "authoritative source text",
) -> LocalCorpusRelease:
    return _write_test_corpus_release(
        tmp_path,
        [{"citation_path": citation_path, "body": body}],
    )


def _dk_full_parity_amendment_fixture_rows() -> list[dict[str, object]]:
    """Load the trimmed 48+3 amendment-act discovery corpus."""

    payload = json.loads(_DK_FULL_PARITY_AMENDMENT_FIXTURE.read_text())
    assert payload["provenance"]["source_sha256"] == _DK_FULL_PARITY_CORPUS_SHA256
    amendment_rows = payload["amendment_rows"]
    assert len(amendment_rows) == 51
    rows: list[dict[str, object]] = []
    for fixture_row in [*payload["target_rows"], *amendment_rows]:
        row = dict(fixture_row)
        row.pop("source_line")
        rows.append(row)
    return rows


def _write_test_source_unit(
    tmp_path: Path,
    body: str,
    *,
    citation_path: str = "us/statute/7/2017",
) -> tuple[LocalCorpusRelease, CorpusSourceUnit]:
    release = _write_test_corpus_provision(
        tmp_path / "source-fixture",
        citation_path=citation_path,
        body=body,
    )
    return release, resolve_corpus_source_unit(citation_path, release)


def test_source_metadata_citation_path_requires_exact_canonical_identity():
    assert (
        _source_metadata_citation_path(
            {
                "source_attestation": {
                    "requested_corpus_citation_path": "us/statute/7/2017"
                }
            }
        )
        == "us/statute/7/2017"
    )

    for invalid in (
        " us/statute/7/2017",
        "us/statute/7/2017/",
        "us:statutes/7/2017",
    ):
        with pytest.raises(InvalidCorpusCitationError):
            _source_metadata_citation_path(
                {"source_attestation": {"requested_corpus_citation_path": invalid}}
            )


def _test_eval_suite_release_identity(
    corpus_release: LocalCorpusRelease,
) -> dict[str, str]:
    return {
        "corpus_release": corpus_release.name,
        "corpus_release_content_sha256": corpus_release.content_sha256,
        "corpus_release_selector_sha256": corpus_release.selector_sha256,
    }


def _bind_fake_source_results(
    results: list[EvalResult],
    kwargs: dict,
) -> list[EvalResult]:
    """Make mocked source results obey the resolver-owned result contract."""

    source_unit = kwargs["source_unit"]
    policy_path = kwargs["policy_path"]
    attestation = _expected_eval_source_attestation(
        source_unit,
        rulespec_root=policy_path,
    )
    for result in results:
        case_output_root = Path(kwargs["output_root"])
        output_file = (
            case_output_root
            / result.runner
            / _resolve_eval_output_path(source_unit.requested)
        )
        trace_file = (
            case_output_root
            / "traces"
            / result.runner
            / f"{_slugify(source_unit.requested)}.json"
        )
        context_manifest_file = (
            case_output_root
            / "_eval_workspaces"
            / result.runner
            / _slugify(source_unit.requested)
            / "workspace"
            / "context-manifest.json"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        context_manifest_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            f"format: rulespec/v1\nmodule:\n  summary: {result.citation}\nrules: []\n"
        )
        trace_file.write_text(json.dumps({"runner": result.runner}, sort_keys=True))
        context_manifest_file.write_text(
            json.dumps({"citation": source_unit.requested}, sort_keys=True)
        )
        result.citation = source_unit.requested
        result.mode = kwargs["mode"]
        result.source_attestation = dict(attestation)
        result.output_file = str(output_file)
        result.trace_file = str(trace_file)
        result.context_manifest_file = str(context_manifest_file)
        result.generated_output_sha256 = hashlib.sha256(
            output_file.read_bytes()
        ).hexdigest()
        result.trace_sha256 = hashlib.sha256(trace_file.read_bytes()).hexdigest()
        result.context_manifest_sha256 = hashlib.sha256(
            context_manifest_file.read_bytes()
        ).hexdigest()
        result.estimated_cost_usd = estimate_usage_cost_usd(
            result.model,
            TokenUsage(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cache_read_tokens=result.cache_read_tokens,
                cache_creation_tokens=result.cache_creation_tokens,
                reasoning_output_tokens=result.reasoning_output_tokens,
            ),
        )
    return results


def _fake_source_runner(*outcomes):
    pending = iter(outcomes)

    def run(**kwargs):
        outcome = next(pending)
        if isinstance(outcome, BaseException):
            raise outcome
        return _bind_fake_source_results(outcome, kwargs)

    return run


def _complete_test_eval_suite(
    tmp_path: Path,
    *,
    runners: list[str] | None = None,
    case_count: int = 1,
    gates: dict[str, object] | None = None,
) -> tuple[EvalSuiteManifest, LocalCorpusRelease, Path, Path]:
    """Create one fully persisted suite using deterministic mocked results."""

    runner_specs = runners or ["openai:gpt-5.4"]
    manifest_gates = {
        "min_cases": 1,
        "min_success_rate": 1.0,
        "min_compile_pass_rate": 1.0,
        "min_ci_pass_rate": 1.0,
        "min_zero_ungrounded_rate": 1.0,
        "min_generalist_review_pass_rate": 1.0,
        **(gates or {}),
    }
    manifest_file = tmp_path / "suite.yaml"
    manifest_file.write_text(
        yaml.safe_dump(
            {
                "name": "Resume hardening suite",
                "runners": runner_specs,
                "gates": manifest_gates,
                "cases": [
                    {
                        "kind": "source",
                        "name": f"case-{index}",
                        "corpus_citation_path": "us/statute/7/2017",
                    }
                    for index in range(1, case_count + 1)
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest = load_eval_suite_manifest(manifest_file)
    corpus_release = _write_test_corpus_provision(tmp_path)
    output_root = tmp_path / "out"
    axiom_rules_path = tmp_path / "axiom-rules-engine"
    axiom_rules_path.mkdir(exist_ok=True)
    call_index = 0

    def fake_source_results(**kwargs):
        nonlocal call_index
        assert EVAL_EVIDENCE_PRIVATE_KEY_ENV not in os.environ
        assert APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV not in os.environ
        call_index += 1
        results = []
        for raw_spec in kwargs["runner_specs"]:
            runner = parse_runner_spec(raw_spec)
            result = _fake_eval_result(runner.name, f"case-{call_index}")
            result.backend = runner.backend
            result.model = runner.model
            results.append(result)
        return _bind_fake_source_results(results, kwargs)

    with patch(
        "axiom_encode.harness.evals.run_source_eval",
        side_effect=fake_source_results,
    ):
        run_eval_suite(
            manifest=manifest,
            output_root=output_root,
            axiom_rules_path=axiom_rules_path,
            policy_repo_path=tmp_path / "rulespec-us",
            corpus_release=corpus_release,
        )
    return manifest, corpus_release, output_root, axiom_rules_path


def _strict_eval_suite_manifest_payload() -> dict[str, object]:
    return {
        "name": "Strict suite",
        "runners": ["openai:gpt-5.4"],
        "gates": {
            "min_cases": 1,
            "min_success_rate": 1.0,
            "min_compile_pass_rate": 1.0,
            "min_ci_pass_rate": 1.0,
            "min_zero_ungrounded_rate": 1.0,
            "min_generalist_review_pass_rate": 1.0,
        },
        "cases": [
            {
                "kind": "source",
                "name": "sample",
                "corpus_citation_path": "us/statute/7/2017",
            }
        ],
    }


class TestParseRunnerSpec:
    def test_parses_named_runner(self):
        runner = parse_runner_spec("gpt=codex:gpt-5.4")
        assert runner.name == "gpt"
        assert runner.backend == "codex"
        assert runner.model == "gpt-5.4"

    def test_parses_default_name(self):
        runner = parse_runner_spec("claude:opus")
        assert runner.name == "claude-opus"
        assert runner.backend == "claude"
        assert runner.model == "opus"

    def test_parses_openai_runner(self):
        runner = parse_runner_spec("openai:gpt-5.4")
        assert runner.name == "openai-gpt-5.4"
        assert runner.backend == "openai"
        assert runner.model == "gpt-5.4"

    @pytest.mark.parametrize("alias", ["../../other", ".", "-runner"])
    def test_rejects_unsafe_runner_alias(self, alias):
        with pytest.raises(ValueError, match="Unsafe runner name"):
            parse_runner_spec(f"{alias}=openai:gpt-5.4")

    @pytest.mark.parametrize(
        "spec",
        [
            "",
            " openai:gpt-5.4",
            "openai:gpt-5.4 ",
            "openai:",
            "=openai:gpt-5.4",
            "runner =openai:gpt-5.4",
            "runner= openai:gpt-5.4",
            "runner=openai: gpt-5.4",
            "runner=openai:gpt 5.4",
        ],
    )
    def test_rejects_noncanonical_or_empty_runner_identity(self, spec):
        with pytest.raises(ValueError, match="Invalid runner spec"):
            parse_runner_spec(spec)


def test_source_identifier_maps_corpus_regulation_to_repo_path():
    assert _source_identifier_to_relative_rulespec_path(
        "us-ny/regulation/18-nycrr/387/12/f/3/v/c"
    ) == Path("regulations/18-nycrr/387/12/f/3/v/c.yaml")


def test_source_identifier_maps_colon_prefixed_regulation_to_repo_path():
    assert _source_identifier_to_relative_rulespec_path(
        "us-co:regulations/10-ccr-2506-1/4.804.1"
    ) == Path("regulations/10-ccr-2506-1/4.804.1.yaml")


def test_resolve_corpus_source_unit_normalizes_colon_prefixed_local_citation(tmp_path):
    corpus_release = _write_test_corpus_provision(
        tmp_path,
        citation_path="us-ca/regulation/cdss/eas/49/49-040",
        body="CAPI resource limits.",
    )

    source_unit = resolve_corpus_source_unit(
        "us-ca:regulation/cdss/eas/49/49-040",
        corpus_release,
    )

    assert source_unit.source == "local"
    assert source_unit.requested == "us-ca/regulation/cdss/eas/49/49-040"
    assert source_unit.citation_path == "us-ca/regulation/cdss/eas/49/49-040"
    assert source_unit.body == "CAPI resource limits."


def test_resolve_corpus_source_unit_reads_text_field_rows(tmp_path):
    corpus_release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": "us-me/regulation/dhhs/ofi/chapter-331/block-4",
                "body": "Maine TANF grant table source text.",
            }
        ],
    )

    source_unit = resolve_corpus_source_unit(
        "us-me/regulation/dhhs/ofi/chapter-331/block-4",
        corpus_release,
    )

    assert source_unit.source == "local"
    assert source_unit.citation_path == "us-me/regulation/dhhs/ofi/chapter-331/block-4"
    assert source_unit.body == "Maine TANF grant table source text."


def test_production_generation_resolver_requires_bound_release(tmp_path):
    corpus_release = _write_test_corpus_provision(tmp_path)

    with pytest.raises(TypeError, match="validated LocalCorpusRelease"):
        resolve_corpus_source_unit(
            "us/statute/7/2017",
            corpus_release.root,  # type: ignore[arg-type]
        )


def test_generation_resolver_uses_active_release_and_attests_full_body(tmp_path):
    citation_path = "us/statute/26/1"
    active_body = "active current-release body"
    corpus_release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": citation_path,
                "version": "2025",
                "body": "inactive body",
            },
            {
                "citation_path": citation_path,
                "version": "2026",
                "body": active_body,
            },
        ],
        selected_scopes=[("us", "statute", "2026")],
    )

    source_unit = resolve_corpus_source_unit(citation_path, corpus_release)

    assert source_unit.body == active_body
    assert source_unit.source_attestation is not None
    assert source_unit.source_attestation["corpus_release"] == _TEST_CORPUS_RELEASE_NAME
    assert (
        source_unit.source_attestation["source_sha256"]
        == hashlib.sha256(active_body.encode()).hexdigest()
    )
    assert source_unit.source_attestation["row"]["version"] == "2026"


def _workspace_prompt_for_source_unit(
    tmp_path: Path,
    source_unit: CorpusSourceUnit,
    *,
    required_import_targets: tuple[str, ...] = (),
    validation_retry_candidate: ValidationRetryCandidate | None = None,
    validation_retry_feedback: tuple[str, ...] = (),
    repair_candidate_tests_only: bool = False,
):
    workspace = prepare_eval_workspace(
        citation=source_unit.requested,
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text=source_unit.body,
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "dk"),
        mode="cold",
        source_metadata_payload={"source_attestation": source_unit.source_attestation},
        provision_metadata=source_unit.provision_metadata,
        amendment_documents=source_unit.amendment_documents,
        extra_context_paths=[],
    )
    prompt = _build_eval_prompt(
        source_unit.requested,
        "cold",
        workspace,
        workspace.context_files,
        target_file_name="target.yaml",
        include_tests=True,
        runner_backend="openai",
        required_import_targets=required_import_targets,
        validation_retry_candidate=validation_retry_candidate,
        validation_retry_feedback=validation_retry_feedback,
        repair_candidate_tests_only=repair_candidate_tests_only,
    )
    return workspace, prompt


def test_workspace_prompt_embeds_latest_retry_candidate_as_untrusted_edit_context(
    tmp_path,
):
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": "dk/statute/benefit/section-1",
                "body": "The benefit is ten percent of the qualifying amount.",
            }
        ],
    )
    source_unit = resolve_corpus_source_unit("dk/statute/benefit/section-1", release)
    candidate = ValidationRetryCandidate(
        rulespec=(
            "format: rulespec/v1\n"
            "# preserve historical TY2000 rate\n"
            "# preserve current TY2026 rate\n"
            "rules: []\n"
        ),
        tests=("# TY2000 historical branch case\n# TY2026 current branch case\n[]\n"),
    )

    _workspace, prompt = _workspace_prompt_for_source_unit(
        tmp_path,
        source_unit,
        validation_retry_candidate=candidate,
    )

    assert "UNTRUSTED REJECTED CANDIDATE RULESPEC" in prompt
    assert "preserve historical TY2000 rate" in prompt
    assert "preserve current TY2026 rate" in prompt
    assert "TY2000 historical branch case" in prompt
    assert "TY2026 current branch case" in prompt
    assert candidate.rulespec_sha256 in prompt or candidate.rulespec in prompt
    assert prompt.index("RuleSpec requirements:") < prompt.index(
        "UNTRUSTED REJECTED CANDIDATE RULESPEC"
    )
    assert prompt.index("UNTRUSTED REJECTED CANDIDATE RULESPEC") < prompt.index(
        "Return exactly this two-file bundle"
    )
    assert str(tmp_path) not in prompt


def test_tests_only_prompt_preserves_cases_wrapper(tmp_path):
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": "dk/statute/benefit/section-1",
                "body": "The benefit is ten percent of the qualifying amount.",
            }
        ],
    )
    source_unit = resolve_corpus_source_unit("dk/statute/benefit/section-1", release)
    candidate = ValidationRetryCandidate(
        rulespec="format: rulespec/v1\nrules: []\n",
        tests="cases:\n  - name: existing\n    period: 2026-01\n"
        "    input: {}\n    output:\n      result: holds\n",
    )

    _workspace, prompt = _workspace_prompt_for_source_unit(
        tmp_path,
        source_unit,
        validation_retry_candidate=candidate,
        repair_candidate_tests_only=True,
    )

    assert "mapping with exactly one `cases` test-case list" in prompt
    assert "must preserve its top-level `cases:` mapping" in prompt
    assert "must be a YAML list beginning" not in prompt


def test_retry_candidate_formatter_requires_fresh_content_digest():
    candidate = ValidationRetryCandidate(
        rulespec="format: rulespec/v1\nrules: []\n",
        tests=None,
    )
    object.__setattr__(candidate, "rulespec_sha256", "0" * 64)

    with pytest.raises(ValueError, match="digest does not match"):
        evals_module._format_validation_retry_candidate(
            candidate,
            target_file_name="section.yaml",
        )


def test_retry_candidate_formatter_explicitly_handles_missing_tests():
    section = evals_module._format_validation_retry_candidate(
        ValidationRetryCandidate(
            rulespec="format: rulespec/v1\nrules: []\n",
            tests=None,
        ),
        target_file_name="section.yaml",
    )

    assert "No companion test file was present" in section
    assert "you may omit unchanged named rules" in section
    assert "`repair_remove: true`" in section


def test_repair_candidate_overlay_restores_omitted_named_items(tmp_path):
    artifact_root = tmp_path / "generated"
    rulespec_file = artifact_root / "regulations" / "section.yaml"
    rulespec_file.parent.mkdir(parents=True)
    rulespec_file.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  deferred_outputs:\n"
        "    - output: us:section#new_deferred\n"
        "      reason: new\n"
        "imports:\n"
        "  - us:source#new\n"
        "rules:\n"
        "  - name: existing_changed\n"
        "    kind: parameter\n"
        "    dtype: Count\n"
        "    versions: [{effective_from: '2026-01-01', formula: 2}]\n"
        "  - name: added\n"
        "    kind: parameter\n"
        "    dtype: Count\n"
        "    versions: [{effective_from: '2026-01-01', formula: 3}]\n",
        encoding="utf-8",
    )
    rulespec_file.with_suffix(".test.yaml").write_text(
        "- name: added_case\n  period: 2026-01\n  input: {}\n  output: {}\n",
        encoding="utf-8",
    )
    candidate = ValidationRetryCandidate(
        rulespec=(
            "format: rulespec/v1\n"
            "module:\n"
            "  deferred_outputs:\n"
            "    - output: us:section#preserved_deferred\n"
            "      reason: preserved\n"
            "    - output: us:section#added\n"
            "      reason: replaced by the generated rule\n"
            "imports:\n"
            "  - us:source#preserved\n"
            "rules:\n"
            "  - name: existing_changed\n"
            "    kind: parameter\n"
            "    dtype: Count\n"
            "    versions: [{effective_from: '2026-01-01', formula: 1}]\n"
            "  - name: omitted\n"
            "    kind: parameter\n"
            "    dtype: Count\n"
            "    versions: [{effective_from: '2026-01-01', formula: 4}]\n"
        ),
        tests=(
            "- name: preserved_case\n  period: 2026-01\n  input: {}\n  output: {}\n"
        ),
    )

    repairs = evals_module._overlay_validation_retry_candidate(
        rulespec_file,
        artifact_root=artifact_root,
        candidate=candidate,
    )

    payload = yaml.safe_load(rulespec_file.read_text(encoding="utf-8"))
    rules = {rule["name"]: rule for rule in payload["rules"]}
    assert rules["existing_changed"]["versions"][0]["formula"] == 2
    assert set(rules) == {"existing_changed", "added", "omitted"}
    assert payload["imports"] == ["us:source#new", "us:source#preserved"]
    assert {item["output"] for item in payload["module"]["deferred_outputs"]} == {
        "us:section#new_deferred",
        "us:section#preserved_deferred",
    }
    tests = yaml.safe_load(
        rulespec_file.with_suffix(".test.yaml").read_text(encoding="utf-8")
    )
    assert {case["name"] for case in tests} == {"added_case", "preserved_case"}
    assert "rule:omitted" in repairs
    assert "test:preserved_case" in repairs
    assert "resolved_deferred_output:us:section#added" in repairs


def test_repair_candidate_overlay_removes_explicit_named_tombstones(tmp_path):
    artifact_root = tmp_path / "generated"
    rulespec_file = artifact_root / "regulations" / "section.yaml"
    rulespec_file.parent.mkdir(parents=True)
    rulespec_file.write_text(
        "format: rulespec/v1\n"
        "rules:\n"
        "  - name: obsolete\n"
        "    repair_remove: true\n"
        "  - name: retained\n"
        "    kind: parameter\n"
        "    dtype: Count\n"
        "    versions: [{effective_from: '2026-01-01', formula: 2}]\n",
        encoding="utf-8",
    )
    rulespec_file.with_suffix(".test.yaml").write_text(
        "- name: obsolete_case\n  repair_remove: true\n"
        "- name: retained_case\n  period: 2026-01\n  input: {}\n  output: {}\n",
        encoding="utf-8",
    )
    candidate = ValidationRetryCandidate(
        rulespec=(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: obsolete\n"
            "    kind: parameter\n"
            "    dtype: Count\n"
            "    versions: [{effective_from: '2026-01-01', formula: 1}]\n"
            "  - name: retained\n"
            "    kind: parameter\n"
            "    dtype: Count\n"
            "    versions: [{effective_from: '2026-01-01', formula: 1}]\n"
        ),
        tests=(
            "- name: obsolete_case\n  period: 2026-01\n  input: {}\n  output: {}\n"
            "- name: retained_case\n  period: 2026-01\n  input: {}\n  output: {}\n"
        ),
    )

    repairs = evals_module._overlay_validation_retry_candidate(
        rulespec_file,
        artifact_root=artifact_root,
        candidate=candidate,
    )

    payload = yaml.safe_load(rulespec_file.read_text(encoding="utf-8"))
    assert [rule["name"] for rule in payload["rules"]] == ["retained"]
    assert "repair_remove" not in rulespec_file.read_text(encoding="utf-8")
    tests = yaml.safe_load(
        rulespec_file.with_suffix(".test.yaml").read_text(encoding="utf-8")
    )
    assert [case["name"] for case in tests] == ["retained_case"]
    assert "removed_rule:obsolete" in repairs
    assert "removed_test:obsolete_case" in repairs


def test_repair_tombstone_survives_materialization_before_overlay(tmp_path):
    artifact_root = tmp_path / "generated"
    rulespec_file = artifact_root / "regulations" / "section.yaml"
    candidate = ValidationRetryCandidate(
        rulespec=(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: obsolete\n"
            "    kind: parameter\n"
            "    dtype: Count\n"
            "    versions: [{effective_from: '2026-01-01', formula: 1}]\n"
        ),
        tests=("- name: obsolete_case\n  period: 2026-01\n  input: {}\n  output: {}\n"),
    )
    response = (
        "=== FILE: section.yaml ===\n"
        "format: rulespec/v1\n"
        "rules:\n"
        "  - name: obsolete\n"
        "    repair_remove: true\n"
        "=== FILE: section.test.yaml ===\n"
        "- name: obsolete_case\n"
        "  repair_remove: true\n"
    )

    assert evals_module._materialize_eval_artifact(
        response,
        rulespec_file,
        artifact_root=artifact_root,
    )
    assert "repair_remove: true" in rulespec_file.read_text(encoding="utf-8")
    test_file = rulespec_file.with_suffix(".test.yaml")
    assert yaml.safe_load(test_file.read_text(encoding="utf-8")) == [
        {"name": "obsolete_case", "repair_remove": True}
    ]

    repairs = evals_module._overlay_validation_retry_candidate(
        rulespec_file,
        artifact_root=artifact_root,
        candidate=candidate,
    )

    assert yaml.safe_load(rulespec_file.read_text(encoding="utf-8"))["rules"] == []
    assert yaml.safe_load(test_file.read_text(encoding="utf-8")) == []
    assert repairs == ("removed_rule:obsolete", "removed_test:obsolete_case")


def test_single_amount_test_normalizer_preserves_exact_repair_tombstone():
    normalized = evals_module._normalize_single_amount_row_test_content(
        "- name: alternate_branch_case\n  repair_remove: true\n",
        rulespec_content=(
            "format: rulespec/v1\n"
            "period: Year\n"
            "rules:\n"
            "  - name: amount\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    versions: [{effective_from: '2026-01-01', formula: 1}]\n"
        ),
    )

    assert yaml.safe_load(normalized) == [
        {"name": "alternate_branch_case", "repair_remove": True}
    ]


@pytest.mark.parametrize(
    "marker,match",
    [
        ({"name": "obsolete", "repair_remove": False}, "repair_remove: true"),
        (
            {"name": "obsolete", "repair_remove": True, "kind": "derived"},
            "only name and repair_remove",
        ),
        ({"name": "unknown", "repair_remove": True}, "no preserved item"),
    ],
)
def test_repair_candidate_overlay_rejects_invalid_tombstones(tmp_path, marker, match):
    artifact_root = tmp_path / "generated"
    rulespec_file = artifact_root / "section.yaml"
    artifact_root.mkdir()
    rulespec_file.write_text(
        yaml.safe_dump({"format": "rulespec/v1", "rules": [marker]}),
        encoding="utf-8",
    )
    candidate = ValidationRetryCandidate(
        rulespec=(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: obsolete\n"
            "    kind: parameter\n"
            "    dtype: Count\n"
            "    versions: [{effective_from: '2026-01-01', formula: 1}]\n"
        )
    )

    with pytest.raises(ValueError, match=match):
        evals_module._overlay_validation_retry_candidate(
            rulespec_file,
            artifact_root=artifact_root,
            candidate=candidate,
        )


def test_repair_candidate_overlay_normalizes_destination_root_deferrals(tmp_path):
    artifact_root = tmp_path / "generated"
    rulespec_file = artifact_root / "regulations" / "7-cfr" / "273" / "4.yaml"
    rulespec_file.parent.mkdir(parents=True)
    rulespec_file.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_path: us/regulation/7/273/4\n"
        "  deferred_outputs:\n"
        "    - output: us:regulations/7-cfr/273/4/a/6/ii/h#generated\n"
        "      reason: generated\n"
        "rules: []\n",
        encoding="utf-8",
    )
    candidate = ValidationRetryCandidate(
        rulespec=(
            "format: rulespec/v1\n"
            "module:\n"
            "  source_verification:\n"
            "    corpus_citation_path: us/regulation/7/273/4\n"
            "  deferred_outputs:\n"
            "    - output: us:regulations/7-cfr/273/4/a/6/ii/h#generated\n"
            "      reason: preserved duplicate\n"
            "    - output: us:regulations/7-cfr/273/4/c/1#preserved\n"
            "      reason: preserved\n"
            "    - output: us:statutes/8/1641/b#external_dependency\n"
            "      reason: external\n"
            "rules: []\n"
        ),
    )

    repairs = evals_module._overlay_validation_retry_candidate(
        rulespec_file,
        artifact_root=artifact_root,
        candidate=candidate,
    )

    payload = yaml.safe_load(rulespec_file.read_text(encoding="utf-8"))
    deferred = payload["module"]["deferred_outputs"]
    assert deferred == [
        {
            "output": "us:regulations/7/273/4/a/6/ii/h#generated",
            "reason": "generated",
        },
        {
            "output": "us:regulations/7/273/4/c/1#preserved",
            "reason": "preserved",
        },
        {
            "output": "us:statutes/8/1641/b#external_dependency",
            "reason": "external",
        },
    ]
    assert not any("7-cfr" in item["output"] for item in deferred)
    assert any(repair.startswith("deferred_output_source_root:") for repair in repairs)


def test_run_model_eval_appends_repair_parameters_after_existing_public_parameters():
    parameters = list(inspect.signature(run_model_eval).parameters)

    assert parameters[-5:] == [
        "required_import_targets",
        "legacy_replacement",
        "replacement_overlay_scope",
        "validation_retry_candidate",
        "repair_candidate_tests_only",
    ]


def test_workspace_prompt_requires_each_atomic_composition_import(tmp_path):
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": "dk/statute/benefit/section-1",
                "body": "The benefit uses the separately determined amount.",
            }
        ],
    )
    source_unit = resolve_corpus_source_unit("dk/statute/benefit/section-1", release)

    _workspace, prompt = _workspace_prompt_for_source_unit(
        tmp_path,
        source_unit,
        required_import_targets=(
            "dk:guidance/benefit/current-amount",
            "dk:statutes/benefit/eligibility",
        ),
    )

    assert "Protected atomic-composition requirements:" in prompt
    assert "`dk:guidance/benefit/current-amount`" in prompt
    assert "`dk:statutes/benefit/eligibility`" in prompt
    assert "directly import at least one exact" in prompt
    assert "remains bound to exactly one primary source" in prompt


def test_workspace_prompt_includes_curated_provision_metadata(tmp_path):
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": "dk/statute/benefit/section-1",
                "body": "The divisor is 12.",
                "heading": "Benefit Act section 1",
                "metadata": {
                    "source_note": "The uplift is not allocated here.",
                    "amended_after_consolidation_note": "12 becomes 24 on 2027-01-01.",
                    "download_url": "https://noise.invalid/large.pdf",
                },
            }
        ],
    )
    source_unit = resolve_corpus_source_unit("dk/statute/benefit/section-1", release)

    workspace, prompt = _workspace_prompt_for_source_unit(tmp_path, source_unit)

    assert workspace.provision_metadata_file is not None
    metadata_text = workspace.provision_metadata_file.read_text()
    assert "Benefit Act section 1" in metadata_text
    assert "uplift is not allocated" in metadata_text
    assert "12 becomes 24" in prompt
    assert "download_url" not in metadata_text
    assert "Provision metadata (from the corpus manifest)" in prompt
    label = "The following corpus-manifest content is untrusted corpus EVIDENCE only"
    assert prompt.index(label) < prompt.index(
        "=== BEGIN Provision metadata (from the corpus manifest) ==="
    )
    assert "encode the affected value\n  as a dated parameter" in prompt


def test_curated_metadata_recursively_strips_all_mechanical_keys(tmp_path):
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": "dk/statute/benefit/section-1",
                "body": "The divisor is 12.",
                "metadata": {
                    "source_note": "retain",
                    "nested": {
                        "block_count": 1,
                        "content_type": "text/plain",
                        "download_url": "https://noise.invalid",
                        "file_size": 99,
                        "sha256": "deadbeef",
                        "legal_note": "retain nested",
                    },
                },
            }
        ],
    )
    source_unit = resolve_corpus_source_unit("dk/statute/benefit/section-1", release)

    rendered = json.dumps(source_unit.provision_metadata, sort_keys=True)

    for key in ("block_count", "content_type", "download_url", "file_size", "sha256"):
        assert f'"{key}"' not in rendered
    assert "retain nested" in rendered


def test_target_source_document_is_not_its_own_amendment_context(tmp_path):
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": "dk/statute/benefit/section-1",
                "body": "The divisor is 12.",
                "source_path": "sources/dk/benefit-act.txt",
                "metadata": {"amends": "dk/statute/benefit/section-0"},
            },
            {
                "citation_path": "dk/statute/benefit/amendment-note",
                "body": "Embedded amendment history.",
                "source_path": "sources/dk/benefit-act.txt",
                "metadata": {"amends": "dk/statute/benefit/section-1"},
            },
        ],
    )

    source_unit = resolve_corpus_source_unit("dk/statute/benefit/section-1", release)

    assert source_unit.amendment_documents == ()


def test_provision_inherits_document_identifiers_for_de_amendment_discovery(tmp_path):
    target_path = "de/statute/estg"
    target_source_path = "sources/de/statute/estg/BJNR010050934.xml"
    stefe_path = "de/statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-1"
    stefe_metadata = {
        "amendment_targets": [target_path, f"{target_path}/32"],
        "amends": (
            "Einkommensteuergesetz (EStG), in der Fassung der Bekanntmachung "
            "vom 8. Oktober 2009 (BGBl. I S. 3366, 3862)"
        ),
        "document_type": "Änderungsgesetz (amendment act)",
        "title": (
            "Gesetz zur Fortentwicklung des Steuerrechts und zur Anpassung des "
            "Einkommensteuertarifs (Steuerfortentwicklungsgesetz – SteFeG) "
            "(BGBl. 2024 I Nr. 449)"
        ),
        "source_note": "Official electronic Bundesgesetzblatt PDF.",
    }
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": target_path,
                "body": "Einkommensteuergesetz",
                "source_path": target_source_path,
                "heading": "Einkommensteuergesetz",
                "metadata": {
                    "ausfertigung_datum": "1934-10-16",
                    "jurabk": "EStG",
                    "law_metadata": {
                        "jurabk": "EStG",
                        "langtitel": "Einkommensteuergesetz",
                    },
                    "law_slug": "estg",
                    "law_title": "Einkommensteuergesetz (EStG)",
                    "legal_authority_url": ("https://www.gesetze-im-internet.de/estg/"),
                },
            },
            {
                "citation_path": f"{target_path}/3",
                "body": "§ 3 Steuerfreie Einnahmen",
                "source_path": target_source_path,
                "heading": "§ 3",
                "metadata": {},
            },
            {
                "citation_path": stefe_path,
                "body": "Artikel 1 ändert das Einkommensteuergesetz.",
                "source_path": "sources/de/statute/stefeg/regelungstext.pdf",
                "expression_date": "2024-12-23",
                "metadata": stefe_metadata,
            },
        ],
    )

    provision = resolve_corpus_source_unit(f"{target_path}/3", release)
    document = resolve_corpus_source_unit(target_path, release)

    assert [item.citation_path for item in provision.amendment_documents] == [
        stefe_path
    ]
    assert (
        provision.amendment_documents[0].metadata["source_note"].startswith("Official")
    )
    assert provision.amendment_documents == document.amendment_documents


def test_dk_full_parity_structured_amendment_timelines_are_exhaustive(tmp_path):
    """Pin every relevant structured match among the live 48+3 amendment acts."""

    release = _write_test_corpus_release(
        tmp_path,
        _dk_full_parity_amendment_fixture_rows(),
    )
    target_724 = "dk/statute/lbk-724-2022/boerne-og-ungeydelsesloven"
    target_603 = "dk/statute/lbk-603-2025/boerne-og-ungeydelsesloven"
    expected_724 = [
        "dk/statute/lov-665-2024/aendringslov/aendringcentreretparagraf-1",
        "dk/statute/lov-632-2024/aendringslov/aendringcentreretparagraf-1",
        ("dk/statute/lov-482-2024/reform-af-personskat/aendringcentreretparagraf-1"),
        "dk/statute/lov-1389-2022/aendringslov/aendringcentreretparagraf-1",
    ]
    expected_603 = [
        (
            "dk/statute/lov-303-2026/aendring-af-social-pension-mv/"
            "aendringcentreretparagraf-1"
        ),
        (
            "dk/statute/lov-1642-2025/aendring-af-boerne-og-ungeydelsesloven-mv/"
            "aendringcentreretparagraf-1"
        ),
        *expected_724,
    ]

    root_724 = resolve_corpus_source_unit(target_724, release)
    provision_724 = resolve_corpus_source_unit(
        f"{target_724}/paragraf-1-a",
        release,
    )
    root_603 = resolve_corpus_source_unit(target_603, release)

    for source_unit, expected in (
        (root_724, expected_724),
        (provision_724, expected_724),
        (root_603, expected_603),
    ):
        assert [
            (document.citation_path, document.match_tier)
            for document in source_unit.amendment_documents
        ] == [(citation_path, "structured") for citation_path in expected]
        workspace, _prompt = _workspace_prompt_for_source_unit(tmp_path, source_unit)
        assert [
            item.citation_path
            for item in workspace.context_files
            if item.kind == "corpus_amendment_act"
        ] == expected

    # Issue #1275 inheritance must expose the document timeline to a provision
    # at the same depth as the shallowest active sibling row.
    assert provision_724.amendment_documents == root_724.amendment_documents


def test_provision_identifier_inheritance_does_not_cross_source_documents(tmp_path):
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": "de/statute/act-a",
                "source_path": "sources/de/act-a.xml",
                "metadata": {"title": "Alpha Family Benefits Act"},
            },
            {
                "citation_path": "de/statute/act-a/3",
                "body": "Act A provision.",
                "source_path": "sources/de/act-a.xml",
                "metadata": {},
            },
            {
                "citation_path": "de/statute/act-b",
                "source_path": "sources/de/act-b.xml",
                "metadata": {"title": "Beta Housing Support Act"},
            },
            {
                "citation_path": "de/statute/act-b/amendment",
                "body": "Act B amendment.",
                "source_path": "sources/de/act-b-amendment.xml",
                "metadata": {
                    "document_type": "amendment act",
                    "amends": "Beta Housing Support Act",
                },
            },
        ],
    )

    assert (
        resolve_corpus_source_unit("de/statute/act-a/3", release).amendment_documents
        == ()
    )
    # Active release rows require source_path, so exercise the resolver's
    # fail-closed boundary against populated candidate rows through its selector.
    candidate_rows = tuple(
        corpus_resolver.iter_active_local_corpus_rows(
            release, jurisdiction="de", document_class="statute"
        )
    )
    assert candidate_rows
    assert (
        _shallowest_active_source_path_row(
            candidate_rows, source_path=None, version=_TEST_CORPUS_VERSION
        )
        is None
    )


def test_provision_inherits_parent_metadata_identifiers(tmp_path):
    source_path = "sources/de/family-benefit.xml"
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": "de/statute/family-benefit",
                "source_path": source_path,
                "metadata": {"title": "German Family Benefit Act"},
            },
            {
                "citation_path": "de/statute/family-benefit/3",
                "body": "Provision text.",
                "source_path": source_path,
                "metadata": {},
            },
            {
                "citation_path": "de/statute/amendment-2026",
                "body": "Amendment text.",
                "source_path": "sources/de/amendment-2026.xml",
                "metadata": {
                    "document_type": "amendment act",
                    "amends": "Amendment of the German Family Benefit Act",
                },
            },
        ],
    )

    source_unit = resolve_corpus_source_unit("de/statute/family-benefit/3", release)

    assert [item.citation_path for item in source_unit.amendment_documents] == [
        "de/statute/amendment-2026"
    ]


def test_provision_inheritance_preserves_same_document_amendment_exclusion(tmp_path):
    source_path = "sources/dk/benefit-act.txt"
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": "dk/statute/benefit",
                "source_path": source_path,
                "metadata": {"title": "Danish Family Benefit Act"},
            },
            {
                "citation_path": "dk/statute/benefit/section-1",
                "body": "The divisor is 12.",
                "source_path": source_path,
                "metadata": {},
            },
            {
                "citation_path": "dk/statute/benefit/amendment-note",
                "body": "Embedded amendment history.",
                "source_path": source_path,
                "metadata": {
                    "document_type": "amendment act",
                    "amends": "Danish Family Benefit Act",
                },
            },
        ],
    )

    source_unit = resolve_corpus_source_unit("dk/statute/benefit/section-1", release)

    assert source_unit.amendment_documents == ()


def test_sibling_amendments_are_context_with_bounded_bodies(tmp_path):
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": "dk/statute/benefit/section-1",
                "body": "The divisor is 12.",
                "metadata": {"title": "Børne- og ungeydelsesloven"},
            },
            {
                "citation_path": "dk/statute/amendment-2027",
                "body": "Change 12 to 24.",
                "source_path": "sources/dk/amendment-2027.txt",
                "expression_date": "2027-01-01",
                "metadata": {
                    "title": "2027 Amendment Act",
                    "amends": "dk/statute/benefit/section-1",
                    "commencement_note": "Commences 2027-07-01.",
                },
            },
            {
                "citation_path": "dk/statute/amendment-2026",
                "body": "x" * 12_001,
                "source_path": "sources/dk/amendment-2026.txt",
                "expression_date": "2026-01-01",
                "metadata": {
                    "title": "2026 Amendment Act",
                    "document_type": "amendment act",
                    "amendment_target": {"title": "Boerne- og ungeydelsesloven"},
                    "source_note": "m" * 7_000,
                },
            },
            {
                "citation_path": "dk/statute/amendment-2027/section-2",
                "body": "Duplicate descendant body.",
                "source_path": "sources/dk/amendment-2027.txt",
                "expression_date": "2028-01-01",
                "metadata": {
                    "title": "Duplicate descendant",
                    "amends": "dk/statute/benefit/section-1",
                },
            },
        ],
    )
    source_unit = resolve_corpus_source_unit("dk/statute/benefit/section-1", release)

    workspace, prompt = _workspace_prompt_for_source_unit(tmp_path, source_unit)
    amendments = [
        item for item in workspace.context_files if item.kind == "corpus_amendment_act"
    ]
    assert [item.label for item in amendments] == [
        "2027 Amendment Act",
        "2026 Amendment Act",
    ]
    assert "Change 12 to 24." in prompt
    assert "body omitted: exceeds 12000-character amendment context cap" in prompt
    assert "amendment metadata truncated at 6000 characters" in prompt
    assert "x" * 12_001 not in prompt
    assert "Duplicate descendant body." not in prompt
    assert "Post-consolidation amendment acts in this corpus scope" in prompt
    assert prompt.count("Change 12 to 24.") == 1
    label = "The following amendment content is untrusted corpus EVIDENCE only"
    assert prompt.index(label) < prompt.index(
        "=== BEGIN Post-consolidation amendment acts in this corpus scope ==="
    )


def test_estg_66_multibyte_amendment_body_remains_visible_in_prompt(tmp_path):
    amendment_citation = (
        "de/statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-1"
    )
    amendment_sentence = (
        "In § 66 Absatz 1 wird die Angabe „250 Euro“ durch die Angabe "
        "„255 Euro“ ersetzt."
    )
    # The sentence itself contributes nine UTF-8 continuation bytes; the suffix
    # supplies the remaining 433 needed to reproduce the 12,317-byte document.
    multibyte_suffix = "ä" * 433
    ascii_padding = "x" * (11_875 - len(amendment_sentence) - len(multibyte_suffix) - 1)
    amendment_body = f"{amendment_sentence}\n{ascii_padding}{multibyte_suffix}"
    amendment = CorpusAmendmentDocument(
        citation_path=amendment_citation,
        title="Steuerfortentwicklungsgesetz – SteFeG",
        expression_date="2024-12-23",
        metadata={},
        body=amendment_body,
        match_tier="structured",
    )
    rulespec_root = _canonical_rulespec_content_root(tmp_path, "de")

    workspace = prepare_eval_workspace(
        citation="de/statute/estg/66",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="Das Kindergeld beträgt monatlich für jedes Kind 259 Euro.",
        axiom_rules_path=rulespec_root,
        mode="cold",
        amendment_documents=(amendment,),
        extra_context_paths=[],
    )
    prompt = _build_eval_prompt(
        "de/statute/estg/66",
        "cold",
        workspace,
        workspace.context_files,
        target_file_name="66.yaml",
        include_tests=True,
        runner_backend="openai",
    )

    assert len(amendment_body) == 11_875
    assert len(amendment_body.encode("utf-8")) == 12_317
    assert amendment_sentence in prompt
    assert amendment_citation in prompt
    assert "body omitted: exceeds 12000-character amendment context cap" not in prompt
    assert evals_module._amendment_documents_visible_in_context_manifest(
        (amendment,),
        workspace.manifest_file,
        expected_manifest_sha256=hashlib.sha256(
            workspace.manifest_file.read_bytes()
        ).hexdigest(),
    ) == (amendment,)


def test_bkgg_amendment_context_is_proof_evidence_never_an_import_target(tmp_path):
    amendment_citation = (
        "de/statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-1"
    )
    amendment = CorpusAmendmentDocument(
        citation_path=amendment_citation,
        title="Steuerfortentwicklungsgesetz – SteFeG",
        expression_date="2024-12-23",
        metadata={},
        body="Artikel 3 ändert § 6a des Bundeskindergeldgesetzes.",
    )
    rulespec_root = _canonical_rulespec_content_root(tmp_path, "de")
    workspace = prepare_eval_workspace(
        citation="de/statute/bkgg/6a",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="§ 6a Kinderzuschlag.",
        axiom_rules_path=rulespec_root,
        mode="cold",
        amendment_documents=(amendment,),
        extra_context_paths=[],
    )

    prompt = _build_eval_prompt(
        "de/statute/bkgg/6a",
        "cold",
        workspace,
        workspace.context_files,
        target_file_name="statutes/bkgg/6a.yaml",
        include_tests=True,
        runner_backend="openai",
    )

    assert amendment_citation in prompt
    assert f"import target `{amendment_citation}`" not in prompt
    assert "proof-citation targets only" in prompt
    assert "NEVER top-level\n`imports:` targets" in prompt
    assert "effective-dated\n`versions` of rules in this target module" in prompt
    amendment_item = next(
        item for item in workspace.context_files if item.kind == "corpus_amendment_act"
    )
    assert amendment_item.import_path is None
    assert amendment_item.citation_path == amendment_citation
    manifest = json.loads(workspace.manifest_file.read_text())
    amendment_manifest_item = next(
        item
        for item in manifest["context_files"]
        if item["kind"] == "corpus_amendment_act"
    )
    assert "import_path" not in amendment_manifest_item
    assert amendment_manifest_item["citation_path"] == amendment_citation

    hydrated_root = tmp_path / "hydrated-eval-root"
    hydrated_root.mkdir()
    _hydrate_eval_root(hydrated_root, workspace)

    assert not list(hydrated_root.rglob("document-1.yaml"))


def test_unrelated_newer_amendment_is_excluded_from_target_context(tmp_path):
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": "dk/statute/boerne-og-ungeydelsesloven/section-1",
                "body": "The divisor is 12.",
                "metadata": {"title": "Børne- og ungeydelsesloven"},
            },
            {
                "citation_path": "dk/statute/amendment-2028",
                "body": "This must not become evidence for the benefit act.",
                "source_path": "sources/dk/amendment-2028.txt",
                "expression_date": "2028-01-01",
                "metadata": {
                    "title": "Unrelated 2028 Amendment Act",
                    "document_type": "amendment act",
                    "amends": "Dagtilbudsloven",
                },
            },
            {
                "citation_path": "dk/statute/amendment-2027",
                "body": "Change the benefit divisor.",
                "source_path": "sources/dk/amendment-2027.txt",
                "expression_date": "2027-01-01",
                "metadata": {
                    "title": "Related 2027 Amendment Act",
                    "amends": "Lov om ændring af Boerne- og ungeydelsesloven",
                },
            },
        ],
    )

    source_unit = resolve_corpus_source_unit(
        "dk/statute/boerne-og-ungeydelsesloven/section-1", release
    )

    assert [item.title for item in source_unit.amendment_documents] == [
        "Related 2027 Amendment Act"
    ]
    assert source_unit.amendment_documents[0].body == "Change the benefit divisor."


@pytest.mark.parametrize(
    ("citation_path", "target_metadata", "relation", "expected"),
    [
        (
            "us/regulation/regulation/section-1",
            {},
            "This amendment updates regulation fees",
            False,
        ),
        (
            "us/statute/benefit/section-1",
            {"title": "Benefit"},
            "Changes benefit administration in another act",
            False,
        ),
        (
            "dk/statute/lbk-603-2025/section-1",
            {},
            "Lov om ændring af LBK nr 603",
            True,
        ),
        (
            "dk/statute/lbk-603-2025/section-1",
            {},
            "Amends lbk-603-2025",
            True,
        ),
        (
            "dk/statute/lbk-603-2025/section-1",
            {},
            "lov om en boerne og ungeydelse jf lovbekendtgoerelse nr 603 af "
            "12 maj 2025 its 1 plus ligningsloven and other acts",
            True,
        ),
        (
            "dk/statute/lbk-603-2025/section-1",
            {
                "title": (
                    "Bekendtgørelse af lov om en børne- og ungeydelse "
                    "(LBK nr 603 af 12/05/2025)"
                )
            },
            "lov om social pension lov om hoejeste mellemste forhoejet almindelig "
            "og almindelig foertidspension m v and various other acts including "
            "lov om en boerne og ungeydelse its 4 e",
            True,
        ),
        (
            "dk/statute/lbk-603-2025/section-1",
            {
                "title": (
                    "Bekendtgoerelse af lov om en boerne- og ungeydelse "
                    "(LBK nr 603 af 12/05/2025)"
                )
            },
            "various acts including lov om en boerne og ungeydelse its 4 e",
            True,
        ),
        ("dk/statute/lbk-603-2025/section-1", {}, "LBK nr 604", False),
        (
            "dk/statute/lbk-603-2025/section-1",
            {},
            "lovbekendtgørelse nr. 604 af 12. maj 2025",
            False,
        ),
        ("dk/statute/lbk-603-2025/section-1", {}, "LBK nr 6030", False),
        ("uk/statute/act-766/section-1", {}, "Act 1766", False),
        (
            "eu/regulation/benefit/section-1",
            {"act_number": "979/2016"},
            "1979/2016",
            False,
        ),
        (
            "dk/statute/lbk-603-2025/section-1",
            {
                "title": (
                    "Bekendtgørelse af lov om en børne- og ungeydelse "
                    "(LBK nr 603 af 12/05/2025)"
                )
            },
            "lov om en boerne under andre regler og ungeydelse",
            False,
        ),
        (
            "dk/statute/lbk-999-2025/section-1",
            {"title": "Bekendtgørelse af lov om skat (LBK nr 999 af 01/01/2025)"},
            "lov om skat",
            False,
        ),
        (
            "dk/statute/boerne-og-ungeydelsesloven/section-1",
            {},
            "Lov om ændring af børne- og ungeydelsesloven",
            True,
        ),
        (
            "uk/statute/act-766/section-1",
            {},
            "This amendment modifies Act 766",
            True,
        ),
        (
            "eu/regulation/benefit/section-1",
            {"act_number": "979/2016"},
            "Amends Regulation 979/2016",
            True,
        ),
        (
            "uk/statute/benefit/section-1",
            {"eli": "urn:lex:opaque-code"},
            "Amends urn:lex:opaque-code",
            True,
        ),
        (
            "uk/statute/benefit/section-1",
            {"act_number": "Series X"},
            "Series X",
            True,
        ),
        (
            "uk/statute/benefit/section-1",
            {"title": "Law of the Act"},
            "Consequential changes of the administration",
            False,
        ),
        (
            "uk/statute/child-benefit/section-1",
            {},
            "Changes child payment rules and benefit administration",
            False,
        ),
        (
            "uk/statute/child-benefit/section-1",
            {},
            "Changes child act benefit administration",
            False,
        ),
    ],
)
def test_amendment_context_requires_distinctive_target_identifier(
    tmp_path, citation_path, target_metadata, relation, expected
):
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": citation_path,
                "body": "Target provision.",
                "source_path": "sources/target-act.txt",
                "metadata": target_metadata,
            },
            {
                "citation_path": "/".join(
                    (*citation_path.split("/")[:2], "amendment-2028")
                ),
                "body": "Candidate amendment.",
                "source_path": "sources/amendment-2028.txt",
                "metadata": {
                    "document_type": "amendment act",
                    "amends": relation,
                },
            },
        ],
    )

    source_unit = resolve_corpus_source_unit(citation_path, release)

    assert bool(source_unit.amendment_documents) is expected


def test_workspace_without_manifest_metadata_stays_minimal(tmp_path):
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": "dk/statute/benefit/section-1",
                "body": "Bare provision.",
            }
        ],
    )
    source_unit = resolve_corpus_source_unit("dk/statute/benefit/section-1", release)
    workspace, prompt = _workspace_prompt_for_source_unit(tmp_path, source_unit)
    prompt_without_injection_feature = _build_eval_prompt(
        source_unit.requested,
        "cold",
        workspace,
        workspace.context_files,
        target_file_name="target.yaml",
        include_tests=True,
        runner_backend="openai",
        include_corpus_context_injection=False,
    )

    assert workspace.provision_metadata_file is None
    assert workspace.context_files == []
    assert "Provision metadata (from the corpus manifest)" not in prompt
    assert "Post-consolidation amendment acts in this corpus scope" not in prompt
    assert (
        "- Treat that source text as the only source of legal truth for this artifact."
        in prompt
    )
    assert "together with supplied corpus-manifest metadata" not in prompt
    assert "encode the affected value\n  as a dated parameter" not in prompt
    assert prompt == prompt_without_injection_feature


def test_name_tier_workspace_remains_byte_identical_at_legacy_two_document_cap(
    tmp_path,
):
    amendments = tuple(
        CorpusAmendmentDocument(
            citation_path=f"dk/statute/amendment-{year}",
            title=f"Name Amendment {year}",
            expression_date=f"{year}-01-01",
            metadata={"source_note": f"note-{year}"},
            body=f"Body {year}.",
            match_tier="name",
        )
        for year in (2027, 2026, 2025, 2024)
    )
    workspace = prepare_eval_workspace(
        citation="dk/statute/benefit/section-1",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="Provision.",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "dk"),
        mode="cold",
        provision_metadata={"title": "Target"},
        amendment_documents=amendments,
        extra_context_paths=[],
    )
    amendment_items = [
        item for item in workspace.context_files if item.kind == "corpus_amendment_act"
    ]
    manifest_bytes = workspace.manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)

    assert [item.citation_path for item in amendment_items] == [
        "dk/statute/amendment-2027",
        "dk/statute/amendment-2026",
    ]
    assert workspace.amendment_documents == amendments[:2]
    expected_drops = tuple(
        {
            "citation_path": f"dk/statute/amendment-{year}",
            "expression_date": f"{year}-01-01",
            "match_tier": "name",
            "reason": "document_count_limit",
        }
        for year in (2025, 2024)
    )
    assert workspace.dropped_amendment_documents == expected_drops
    assert manifest["dropped_amendment_documents"] == list(expected_drops)
    assert hashlib.sha256(manifest_bytes).hexdigest() == (
        "17e2f84c82ccc1a3944e83348d6d125408e919126b629c6c9104b4f639bb532c"
    )
    assert [
        hashlib.sha256((workspace.root / item.workspace_path).read_bytes()).hexdigest()
        for item in amendment_items
    ] == [
        "f50ca5de1f2a93844982fa10b160cca5239d911a31452a4ae05b2aed71599ea6",
        "613163fdb740b3f32deeafb1019f502ac9ac5e359b9d53c2d91c1489a876e591",
    ]


def test_name_tier_legacy_budget_records_sliced_document_without_changing_bytes():
    first_two = tuple(
        CorpusAmendmentDocument(
            citation_path=f"n{year}",
            title=f"Name Amendment {year}",
            expression_date=f"{year}-01-01",
            metadata={"detail": letter * 6_500},
            body=letter * 11_500,
            match_tier="name",
        )
        for year, letter in ((2027, "N"), (2026, "O"))
    )
    omitted = CorpusAmendmentDocument(
        citation_path="n2025",
        title="Name Amendment 2025",
        expression_date="2025-01-01",
        metadata={"detail": "P" * 6_500},
        body="P" * 11_500,
        match_tier="name",
    )

    rendered = evals_module._render_injected_context({}, (*first_two, omitted))
    legacy_provision, legacy_texts, _ = evals_module._render_legacy_name_tier_context(
        {}, first_two
    )

    assert rendered.provision_text == legacy_provision
    assert rendered.amendment_texts == tuple(legacy_texts)
    assert [document.citation_path for document in rendered.amendment_documents] == [
        document.citation_path for document in first_two
    ]
    assert rendered.amendment_documents[0].body == first_two[0].body
    assert first_two[1].body.startswith(rendered.amendment_documents[1].body)
    assert len(rendered.amendment_documents[1].body) < len(first_two[1].body)
    assert rendered.dropped_amendment_documents == (
        {
            "citation_path": "n2025",
            "expression_date": "2025-01-01",
            "match_tier": "name",
            "reason": "aggregate_context_limit",
        },
    )


def test_name_tier_exactly_two_documents_records_no_omissions():
    amendments = tuple(
        CorpusAmendmentDocument(
            citation_path=f"n{year}",
            title=f"Name Amendment {year}",
            expression_date=f"{year}-01-01",
            metadata={},
            body="Body.",
            match_tier="name",
        )
        for year in (2027, 2026)
    )

    rendered = evals_module._render_injected_context({}, amendments)

    assert rendered.amendment_documents == amendments
    assert rendered.dropped_amendment_documents == ()


def test_injected_context_enforces_aggregate_character_cap_newest_last(tmp_path):
    amendments = tuple(
        CorpusAmendmentDocument(
            citation_path=f"dk/statute/amendment-{year}",
            title=f"Amendment {year}",
            expression_date=f"{year}-01-01",
            metadata={"source_note": "m" * 5_500},
            body=letter * 11_500,
        )
        for year, letter in ((2027, "N"), (2026, "O"))
    )
    workspace = prepare_eval_workspace(
        citation="dk/statute/benefit/section-1",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="Provision.",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "dk"),
        mode="cold",
        provision_metadata={"source_note": "p" * 5_500},
        amendment_documents=amendments,
        extra_context_paths=[],
    )

    amendment_texts = [
        (workspace.root / item.workspace_path).read_text().rstrip("\n")
        for item in workspace.context_files
        if item.kind == "corpus_amendment_act"
    ]
    injected_length = len(workspace.provision_metadata_text or "") + sum(
        map(len, amendment_texts)
    )

    assert injected_length <= 32_000
    assert "N" * 11_500 in amendment_texts[0]
    assert (
        "amendment body truncated to satisfy aggregate context cap"
        in amendment_texts[1]
    )


def test_legacy_truncated_amendment_exposes_only_authenticated_body_prefix(tmp_path):
    amendments = (
        CorpusAmendmentDocument(
            citation_path="dk/statute/amendment-2027",
            title="Amendment 2027",
            expression_date="2027-01-01",
            metadata={"source_note": "m" * 5_500},
            body="A" * 11_500,
            match_tier="name",
        ),
        CorpusAmendmentDocument(
            citation_path="dk/statute/amendment-2026",
            title="Amendment 2026",
            expression_date="2026-01-01",
            metadata={"source_note": "m" * 5_500},
            body="250 " + ("B" * 11_490) + " 255",
            match_tier="name",
        ),
    )
    workspace = prepare_eval_workspace(
        citation="dk/statute/benefit/section-1",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="Provision.",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "dk"),
        mode="cold",
        provision_metadata={"source_note": "p" * 5_500},
        amendment_documents=amendments,
        extra_context_paths=[],
    )

    visible = evals_module._amendment_documents_visible_in_context_manifest(
        amendments,
        workspace.manifest_file,
        expected_manifest_sha256=hashlib.sha256(
            workspace.manifest_file.read_bytes()
        ).hexdigest(),
    )

    assert [document.citation_path for document in visible] == [
        document.citation_path for document in amendments
    ]
    assert "250" in visible[1].body
    assert "255" not in visible[1].body
    assert workspace.amendment_documents[1].body == visible[1].body


def test_structured_omitted_amendment_body_is_not_grounding_evidence(tmp_path):
    amendment = CorpusAmendmentDocument(
        citation_path="dk/statute/amendment-2027",
        title="Amendment 2027",
        expression_date="2027-01-01",
        metadata={},
        body=("A" * 12_001) + " hidden 255",
        match_tier="structured",
    )
    workspace = prepare_eval_workspace(
        citation="dk/statute/benefit/section-1",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="Provision.",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "dk"),
        mode="cold",
        amendment_documents=(amendment,),
        extra_context_paths=[],
    )

    visible = evals_module._amendment_documents_visible_in_context_manifest(
        (amendment,),
        workspace.manifest_file,
        expected_manifest_sha256=hashlib.sha256(
            workspace.manifest_file.read_bytes()
        ).hexdigest(),
    )

    assert workspace.amendment_documents[0].body == ""
    assert visible == ()


def test_amendment_visibility_rejects_manifest_digest_drift(tmp_path):
    amendment = CorpusAmendmentDocument(
        citation_path="dk/statute/amendment-2027",
        title="Amendment 2027",
        expression_date="2027-01-01",
        metadata={},
        body="Visible body.",
        match_tier="structured",
    )
    workspace = prepare_eval_workspace(
        citation="dk/statute/benefit/section-1",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="Provision.",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "dk"),
        mode="cold",
        amendment_documents=(amendment,),
        extra_context_paths=[],
    )
    expected_sha256 = hashlib.sha256(workspace.manifest_file.read_bytes()).hexdigest()
    workspace.manifest_file.write_text(workspace.manifest_file.read_text() + "\n")

    with pytest.raises(ValueError, match="digest does not match"):
        evals_module._amendment_documents_visible_in_context_manifest(
            (amendment,),
            workspace.manifest_file,
            expected_manifest_sha256=expected_sha256,
        )


def test_structured_aggregate_cap_drops_oldest_document_and_records_manifest(
    tmp_path,
):
    amendments = tuple(
        CorpusAmendmentDocument(
            citation_path=f"dk/statute/amendment-{year}",
            title=f"Structured Amendment {year}",
            expression_date=f"{year}-01-01",
            metadata={},
            body=letter * 11_500,
            match_tier="structured",
        )
        for year, letter in ((2027, "N"), (2026, "O"), (2025, "P"))
    )
    workspace = prepare_eval_workspace(
        citation="dk/statute/benefit/section-1",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="Provision.",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "dk"),
        mode="cold",
        amendment_documents=amendments,
        extra_context_paths=[],
    )
    amendment_items = [
        item for item in workspace.context_files if item.kind == "corpus_amendment_act"
    ]
    amendment_texts = [
        (workspace.root / item.workspace_path).read_text().rstrip("\n")
        for item in amendment_items
    ]
    manifest = json.loads(workspace.manifest_file.read_text())
    expected_drop = {
        "citation_path": "dk/statute/amendment-2025",
        "expression_date": "2025-01-01",
        "match_tier": "structured",
        "reason": "aggregate_context_limit",
    }

    assert [item.citation_path for item in amendment_items] == [
        "dk/statute/amendment-2027",
        "dk/statute/amendment-2026",
    ]
    assert (
        len((workspace.provision_metadata_text or "").encode("utf-8"))
        + sum(len(text.encode("utf-8")) for text in amendment_texts)
        <= 32_000
    )
    assert "N" * 11_500 in amendment_texts[0]
    assert "O" * 11_500 in amendment_texts[1]
    assert all("aggregate context cap" not in text for text in amendment_texts)
    assert workspace.dropped_amendment_documents == (expected_drop,)
    assert manifest["dropped_amendment_documents"] == [expected_drop]
    assert (
        evals_module._amendment_documents_visible_in_context_manifest(
            amendments,
            workspace.manifest_file,
            expected_manifest_sha256=hashlib.sha256(
                workspace.manifest_file.read_bytes()
            ).hexdigest(),
        )
        == amendments[:2]
    )


def test_structured_aggregate_cap_drops_name_tier_before_structured(tmp_path):
    amendments = (
        CorpusAmendmentDocument(
            citation_path="dk/statute/structured-2025",
            title="Structured Amendment 2025",
            expression_date="2025-01-01",
            metadata={"detail": "S" * 6_500},
            body="S" * 11_500,
            match_tier="structured",
        ),
        CorpusAmendmentDocument(
            citation_path="dk/statute/name-2027",
            title="Name Amendment 2027",
            expression_date="2027-01-01",
            metadata={"detail": "N" * 6_500},
            body="N" * 11_500,
            match_tier="name",
        ),
    )
    workspace = prepare_eval_workspace(
        citation="dk/statute/benefit/section-1",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="Provision.",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "dk"),
        mode="cold",
        amendment_documents=amendments,
        extra_context_paths=[],
    )

    assert [
        item.citation_path
        for item in workspace.context_files
        if item.kind == "corpus_amendment_act"
    ] == ["dk/statute/structured-2025"]
    assert workspace.dropped_amendment_documents == (
        {
            "citation_path": "dk/statute/name-2027",
            "expression_date": "2027-01-01",
            "match_tier": "name",
            "reason": "aggregate_context_limit",
        },
    )


def test_structured_aggregate_cap_never_retains_name_after_structured_drop():
    amendments = (
        CorpusAmendmentDocument(
            citation_path="dk/statute/structured-2025",
            title="Structured Amendment 2025",
            expression_date="2025-01-01",
            metadata={"detail": "S" * 6_500},
            body="S" * 11_500,
            match_tier="structured",
        ),
        CorpusAmendmentDocument(
            citation_path="dk/statute/name-2027",
            title="Name Amendment 2027",
            expression_date="2027-01-01",
            metadata={"detail": "N" * 6_500},
            body="N" * 11_500,
            match_tier="name",
        ),
        CorpusAmendmentDocument(
            citation_path="dk/statute/structured-2024",
            title="Structured Amendment 2024",
            expression_date="2024-01-01",
            metadata={"detail": "O" * 6_500},
            body="O" * 11_500,
            match_tier="structured",
        ),
    )

    rendered = evals_module._render_injected_context({}, amendments)
    dropped_structured = any(
        item["match_tier"] == "structured"
        for item in rendered.dropped_amendment_documents
    )
    retained_name = any(
        document.match_tier == "name" for document in rendered.amendment_documents
    )

    assert dropped_structured
    assert not retained_name
    assert rendered.dropped_amendment_documents[0]["match_tier"] == "name"


def test_structured_aggregate_cap_accounts_for_prompt_join_separator():
    first_two = tuple(
        CorpusAmendmentDocument(
            citation_path=f"dk/statute/amendment-{year}",
            title=f"Amendment {year}",
            expression_date=f"{year}-01-01",
            metadata={},
            body=letter * 10_000,
            match_tier="structured",
        )
        for year, letter in ((2027, "A"), (2026, "B"))
    )
    rendered_first_two = tuple(
        evals_module._render_amendment_document(document, body=document.body)
        for document in first_two
    )
    empty_third = CorpusAmendmentDocument(
        citation_path="dk/statute/amendment-2025",
        title="Amendment 2025",
        expression_date="2025-01-01",
        metadata={},
        body="",
        match_tier="structured",
    )
    empty_third_text = evals_module._render_amendment_document(empty_third, body="")
    third_body_bytes = (
        31_997 - sum(map(len, rendered_first_two)) - len(empty_third_text)
    )
    assert 0 < third_body_bytes <= 12_000
    third = replace(empty_third, body="C" * third_body_bytes)
    amendments = (*first_two, third)
    pre_fix_texts = tuple(
        evals_module._render_amendment_document(document, body=document.body)
        for document in amendments
    )
    assert sum(len(text.encode("utf-8")) for text in pre_fix_texts) == 31_997
    assert (
        len(evals_module._AMENDMENT_CONTEXT_SEPARATOR.join(pre_fix_texts).encode())
        == 32_001
    )

    rendered = evals_module._render_injected_context({}, amendments)
    post_join = evals_module._AMENDMENT_CONTEXT_SEPARATOR.join(rendered.amendment_texts)

    assert len(post_join.encode("utf-8")) <= 32_000
    assert [item["citation_path"] for item in rendered.dropped_amendment_documents] == [
        "dk/statute/amendment-2025"
    ]


def test_provision_metadata_rendering_enforces_character_cap(tmp_path):
    workspace = prepare_eval_workspace(
        citation="dk/statute/benefit/section-1",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="Provision.",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "dk"),
        mode="cold",
        provision_metadata={"source_note": "z" * 10_000},
        extra_context_paths=[],
    )

    assert workspace.provision_metadata_text is not None
    assert len(workspace.provision_metadata_text) == 6_000
    assert workspace.provision_metadata_text.endswith(
        "[provision metadata truncated at 6000 characters]"
    )


def test_run_source_eval_rejects_forged_resolver_source(tmp_path):
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path,
        "authoritative source",
    )
    forged_resolved = replace(
        source_unit.resolved_source,
        body="caller-supplied replacement",
        resolved_text_sha256=hashlib.sha256(b"caller-supplied replacement").hexdigest(),
    )
    forged = replace(
        source_unit,
        body=forged_resolved.body,
        source_attestation=forged_resolved.to_attestation(),
        resolved_source=forged_resolved,
    )

    with pytest.raises(ValueError, match="fresh resolution"):
        run_source_eval(
            source_unit=forged,
            runner_specs=["openai:gpt-5.4"],
            output_root=tmp_path / "out",
            policy_path=_canonical_rulespec_content_root(tmp_path, "us"),
            local_corpus_release=corpus_release,
            runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
        )

    assert not (tmp_path / "out").exists()


def test_run_source_eval_accepts_fresh_resolver_owned_child_slice(tmp_path):
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path,
        "(a) First child source.\n\n(b) Second child source.",
    )
    scoped_resolved = resolve_scoped_local_corpus_source(
        source_unit.resolved_source,
        "us/statute/7/2017/a",
        corpus_release,
    )
    assert scoped_resolved.requested == "us/statute/7/2017/a"
    assert scoped_resolved.slice_required is True
    scoped_source_unit = replace(
        source_unit,
        requested=scoped_resolved.requested,
        citation_path=scoped_resolved.citation_path,
        body=scoped_resolved.body,
        source_attestation=scoped_resolved.to_attestation(),
        resolved_source=scoped_resolved,
    )
    response = EvalPromptResponse(
        text=(
            "=== FILE: a.yaml ===\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: First child source.\n"
            "rules: []\n"
            "=== FILE: a.test.yaml ===\n"
            "[]\n"
        ),
        duration_ms=10,
        trace={},
    )

    with (
        patch("axiom_encode.harness.evals._run_prompt_eval", return_value=response),
        patch("axiom_encode.harness.evals.evaluate_artifact", return_value=None),
    ):
        [result] = run_source_eval(
            source_unit=scoped_source_unit,
            runner_specs=["openai:gpt-5.4"],
            output_root=tmp_path / "out",
            policy_path=_canonical_rulespec_content_root(tmp_path, "us"),
            local_corpus_release=corpus_release,
            runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
        )

    manifest = json.loads(Path(result.context_manifest_file).read_text())
    assert (
        manifest["source_metadata"]["source_attestation"][
            "requested_corpus_citation_path"
        ]
        == "us/statute/7/2017/a"
    )
    source_text = (
        Path(result.context_manifest_file).parent / manifest["source_text_file"]
    ).read_text()
    assert "First child source" in source_text
    assert "Second child source" not in source_text


def test_run_source_eval_rejects_removed_source_id_override(tmp_path):
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path,
        "(a) First child source.\n\n(b) Second child source.",
    )

    with pytest.raises(TypeError, match="source_id"):
        run_source_eval(
            source_id="us/statute/7/2017/a",
            source_unit=source_unit,
            runner_specs=["openai:gpt-5.4"],
            output_root=tmp_path / "out",
            policy_path=_canonical_rulespec_content_root(tmp_path, "us"),
            local_corpus_release=corpus_release,
            runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
        )

    assert not (tmp_path / "out").exists()


def test_run_source_eval_requires_explicit_runtime_engine_path(tmp_path):
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path,
        "authoritative source",
    )

    with pytest.raises(TypeError, match="runtime_axiom_rules_path"):
        run_source_eval(
            source_unit=source_unit,
            runner_specs=["openai:gpt-5.4"],
            output_root=tmp_path / "out",
            policy_path=_canonical_rulespec_content_root(tmp_path, "us"),
            local_corpus_release=corpus_release,
            mode="cold",
        )

    assert not (tmp_path / "out").exists()


def test_model_eval_reuses_identical_resolved_source_for_all_runners(tmp_path):
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path,
        "authoritative source",
        citation_path="us/statute/26/1",
    )
    results = [Mock(name="first-result"), Mock(name="second-result")]

    with (
        patch(
            "axiom_encode.harness.evals.resolve_corpus_source_unit",
            return_value=source_unit,
        ) as mock_resolve,
        patch(
            "axiom_encode.harness.evals._run_single_eval",
            side_effect=results,
        ) as mock_run,
    ):
        actual = run_model_eval(
            citations=["us/statute/26/1"],
            runner_specs=["codex:model-a", "openai:model-b"],
            output_root=tmp_path / "out",
            policy_path=tmp_path / "rulespec",
            runtime_axiom_rules_path=tmp_path / "engine",
            corpus_release=corpus_release,
        )

    assert actual == results
    mock_resolve.assert_called_once_with("us/statute/26/1", corpus_release)
    assert mock_run.call_count == 2
    assert all(
        call.kwargs["source_unit"] is source_unit for call in mock_run.call_args_list
    )


def test_model_eval_passes_single_target_output_override(tmp_path):
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path,
        "authoritative source",
        citation_path="us-nc/statute/105/105-153.7",
    )
    target = Path("policies/income_tax/pilot_liability_pipeline.yaml")
    legacy_replacement = Mock(spec=LegacyReplacementContract)
    result = Mock(name="result")

    with (
        patch(
            "axiom_encode.harness.evals.resolve_corpus_source_unit",
            return_value=source_unit,
        ),
        patch(
            "axiom_encode.harness.evals._run_single_eval",
            return_value=result,
        ) as mock_run,
    ):
        actual = run_model_eval(
            citations=["us-nc/statute/105/105-153.7"],
            runner_specs=["openai:model-a"],
            output_root=tmp_path / "out",
            policy_path=tmp_path / "rulespec-us" / "us-nc",
            runtime_axiom_rules_path=tmp_path / "engine",
            corpus_release=corpus_release,
            target_relative_output=target,
            legacy_replacement=legacy_replacement,
            replacement_overlay_scope=True,
        )

    assert actual == [result]
    assert mock_run.call_args.kwargs["target_relative_output"] == target
    assert mock_run.call_args.kwargs["legacy_replacement"] is legacy_replacement
    assert mock_run.call_args.kwargs["replacement_overlay_scope"] is True


def test_model_eval_tests_only_repair_forces_companion_test_prompting(tmp_path):
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path,
        "authoritative source",
        citation_path="us/regulation/7/273/4",
    )
    candidate = ValidationRetryCandidate(
        "format: rulespec/v1\nrules: []\n",
        "- name: existing\n  period: 2025-06\n  input: {}\n  output: {}\n",
    )
    with (
        patch(
            "axiom_encode.harness.evals.resolve_corpus_source_unit",
            return_value=source_unit,
        ),
        patch(
            "axiom_encode.harness.evals._run_single_eval",
            return_value=Mock(name="result"),
        ) as mock_run,
    ):
        run_model_eval(
            citations=["us/regulation/7/273/4"],
            runner_specs=["openai:model-a"],
            output_root=tmp_path / "out",
            policy_path=tmp_path / "rulespec-us" / "us",
            runtime_axiom_rules_path=tmp_path / "engine",
            corpus_release=corpus_release,
            validation_retry_candidate=candidate,
            repair_candidate_tests_only=True,
        )

    assert mock_run.call_args.kwargs["include_tests"] is True


def test_model_eval_rejects_replacement_scope_without_target_override(tmp_path):
    corpus_release, _source_unit = _write_test_source_unit(
        tmp_path,
        "authoritative source",
        citation_path="us/statute/26/1",
    )

    with pytest.raises(ValueError, match="requires an explicit target"):
        run_model_eval(
            citations=["us/statute/26/1"],
            runner_specs=["openai:model-a"],
            output_root=tmp_path / "out",
            policy_path=tmp_path / "rulespec-us" / "us",
            runtime_axiom_rules_path=tmp_path / "engine",
            corpus_release=corpus_release,
            replacement_overlay_scope=True,
        )


def test_model_eval_rejects_target_override_for_multiple_citations(tmp_path):
    corpus_release, _source_unit = _write_test_source_unit(
        tmp_path,
        "authoritative source",
        citation_path="us/statute/26/1",
    )

    with pytest.raises(ValueError, match="requires exactly one citation"):
        run_model_eval(
            citations=["us/statute/26/1", "us/statute/26/2"],
            runner_specs=["openai:model-a"],
            output_root=tmp_path / "out",
            policy_path=tmp_path / "rulespec-us" / "us",
            runtime_axiom_rules_path=tmp_path / "engine",
            corpus_release=corpus_release,
            target_relative_output=Path("policies/income_tax/pipeline.yaml"),
        )


def test_workspace_classifies_output_override_as_existing_target(tmp_path):
    policy_root = _canonical_rulespec_content_root(tmp_path, "us-nc")
    target_relative = Path("policies/income_tax/pilot_liability_pipeline.yaml")
    target = policy_root / target_relative
    target.parent.mkdir(parents=True)
    target.write_text("format: rulespec/v1\nrules: []\n")
    companion = target.with_name("pilot_liability_pipeline.test.yaml")
    companion.write_text("[]\n")

    workspace = prepare_eval_workspace(
        citation="us-nc/statute/105/105-153.7",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="Authoritative source.",
        axiom_rules_path=policy_root,
        mode="repo-augmented",
        extra_context_paths=[target, companion],
        target_relative_output=target_relative,
    )

    kinds = {Path(item.source_path): item.kind for item in workspace.context_files}
    assert kinds[target.resolve()] == "existing_target"
    assert kinds[companion.resolve()] == "existing_target_test_context"


def test_model_eval_uses_output_override_for_prompt_and_artifact_path(tmp_path):
    corpus_release, _source_unit = _write_test_source_unit(
        tmp_path,
        "Authoritative source.",
        citation_path="us-nc/statute/105/105-153.7",
    )
    policy_root = _canonical_rulespec_content_root(tmp_path, "us-nc")
    target_relative = Path("policies/income_tax/pilot_liability_pipeline.yaml")
    target = policy_root / target_relative
    target.parent.mkdir(parents=True)
    target.write_text("format: rulespec/v1\nrules: []\n")
    companion = target.with_name("pilot_liability_pipeline.test.yaml")
    companion.write_text("[]\n")
    output_root = tmp_path / "out"

    def generate_override_artifact(**kwargs):
        output_file = kwargs["output_file"]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("format: rulespec/v1\nrules: []\n")
        return (
            EvalPromptResponse(text="generated", duration_ms=1),
            True,
            0,
            frozenset({output_file}),
        )

    with (
        patch(
            "axiom_encode.harness.evals._run_prompt_eval_with_empty_artifact_retry",
            side_effect=generate_override_artifact,
        ),
        patch(
            "axiom_encode.harness.evals._evaluate_generated_artifact_with_repairs",
            return_value=EvalArtifactMetrics(
                compile_pass=True,
                compile_issues=[],
                ci_pass=True,
                ci_issues=[],
                embedded_source_present=False,
                grounded_numeric_count=0,
                ungrounded_numeric_count=0,
                grounding=[],
            ),
        ),
        patch(
            "axiom_encode.harness.evals._build_eval_prompt",
            wraps=evals_module._build_eval_prompt,
        ) as mock_prompt,
    ):
        result = run_model_eval(
            citations=["us-nc/statute/105/105-153.7"],
            runner_specs=["openai:model-a"],
            output_root=output_root,
            policy_path=policy_root,
            runtime_axiom_rules_path=tmp_path / "engine",
            corpus_release=corpus_release,
            mode="repo-augmented",
            extra_context_paths=[target, companion],
            target_relative_output=target_relative,
        )[0]

    assert Path(result.output_file) == (
        output_root / "openai-model-a" / target_relative
    )
    assert mock_prompt.call_args.kwargs["target_file_name"] == target.name
    assert mock_prompt.call_args.kwargs["target_ref_prefix"] == (
        "us-nc:policies/income_tax/pilot_liability_pipeline"
    )


def test_resolve_corpus_source_unit_concatenates_descendant_text_rows(tmp_path):
    rows = [
        {
            "citation_path": "us-me/regulation/dhhs/ofi/chapter-331",
        },
        {
            "citation_path": "us-me/regulation/dhhs/ofi/chapter-331/block-1",
            "ordinal": 2,
            "body": "Second extracted block.",
        },
        {
            "citation_path": "us-me/regulation/dhhs/ofi/chapter-331/block-2",
            "ordinal": 1,
            "heading": "Need standards",
            "body": "First extracted block.",
        },
    ]
    corpus_release = _write_test_corpus_release(tmp_path, rows)

    source_unit = resolve_corpus_source_unit(
        "us-me/regulation/dhhs/ofi/chapter-331",
        corpus_release,
    )

    assert source_unit.body == (
        "Need standards\n\nFirst extracted block.\n\nSecond extracted block."
    )


def test_resolve_corpus_source_unit_matches_root_to_active_source_path(tmp_path):
    target_path = "dk/statute/lbk-603-2025/boerne-og-ungeydelsesloven"
    target_source_path = "sources/dk/target.pdf"
    release = _write_test_corpus_release(
        tmp_path,
        [
            {"citation_path": target_path, "source_path": target_source_path},
            {
                "citation_path": f"{target_path}/document-1",
                "body": "Consolidated target text.",
                "source_path": target_source_path,
                "metadata": {
                    "title": (
                        "Bekendtgørelse af lov om en børne- og ungeydelse "
                        "(LBK nr 603 af 12/05/2025)"
                    ),
                    "source_note": "Curated target note.",
                },
            },
            {
                "citation_path": "dk/statute/lov-1642-2025/amendment/document-1",
                "body": "Amendment text.",
                "source_path": "sources/dk/amendment.pdf",
                "metadata": {
                    "document_type": "amendment act",
                    "amends": target_path,
                    "title": "LOV nr 1642 af 16/12/2025",
                },
            },
        ],
    )

    source_unit = resolve_corpus_source_unit(target_path, release)

    assert source_unit.provision_metadata["source_note"] == "Curated target note."
    assert [item.citation_path for item in source_unit.amendment_documents] == [
        "dk/statute/lov-1642-2025/amendment/document-1"
    ]


def test_source_identifier_maps_state_manual_to_policies_repo_path():
    assert _source_identifier_to_relative_rulespec_path(
        "us-az/manual/des/faa5/na-child-support-expense/block-2"
    ) == Path("policies/des/faa5/na-child-support-expense/block-2.yaml")


def test_admin_agency_aggregate_rejects_household_executable_rule():
    source_text = (
        "FNS shall estimate each State agency's active case, payment, and negative "
        "case error rate. y2′ = y2 + b2(X2−x2), where X2 is the average value of "
        "allotments underissued to participating households in the State agency "
        "full quality control sample."
    )
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/regulation/7/275/23
rules:
  - name: average_allotments_underissued_active_error_rate
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    source: 7 CFR 275.23(b)(2)(i)(B)
    versions:
      - effective_from: '0001-01-01'
        formula: y2 + b2 * (x2_full_sample - x2_rereview)
"""

    issues = find_admin_agency_aggregate_entity_issues(content, source_text)

    assert issues == [
        "Unsupported administrative aggregate entity: "
        "`average_allotments_underissued_active_error_rate` is declared on "
        "`Household`, but the authoritative source defines a State agency/FNS "
        "aggregate performance, sampling, liability, waiver, or bonus measure. "
        "Use a source-stated administrative entity such as `StateAgency` "
        "instead of a household/person/tax/payment entity, or defer only if the "
        "administrative surface still cannot be represented faithfully."
    ]


def test_admin_agency_aggregate_rejects_bonus_payment_spending_restriction():
    source_text = (
        "Bonus payments shall not be used for household benefits, including "
        "incentive payments."
    )
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/regulation/7/275/24
rules:
  - name: bonus_payment_may_be_used_for_household_benefits
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    source: 7 CFR 275.24(a)(8)(i)
    versions:
      - effective_from: '0001-01-01'
        formula: not payment_is_bonus_payment or not payment_use_is_household_benefit
"""

    issues = find_admin_agency_aggregate_entity_issues(content, source_text)

    assert issues == [
        "Unsupported administrative aggregate entity: "
        "`bonus_payment_may_be_used_for_household_benefits` is declared on "
        "`Payment`, but the authoritative source defines a State agency/FNS "
        "aggregate performance, sampling, liability, waiver, or bonus measure. "
        "Use a source-stated administrative entity such as `StateAgency` "
        "instead of a household/person/tax/payment entity, or defer only if the "
        "administrative surface still cannot be represented faithfully."
    ]


def test_admin_agency_aggregate_allows_state_agency_entity():
    source_text = (
        "The amount of the liability shall be equal to the product of the value "
        "of all allotments issued by the State agency, the difference between "
        "the State agency's payment error rate and 6 percent, and 10 percent."
    )
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/regulation/7/275/23
rules:
  - name: state_agency_payment_error_rate_liability
    kind: derived
    entity: StateAgency
    dtype: Money
    period: Year
    source: 7 CFR 275.23(d)(2)
    versions:
      - effective_from: '2003-10-01'
        formula: all_allotments_issued_by_state_agency * (state_agency_payment_error_rate - 0.06) * 0.10
"""

    assert find_admin_agency_aggregate_entity_issues(content, source_text) == []


def test_admin_agency_aggregate_allows_household_level_source():
    source_text = (
        "A household is eligible for SNAP if it meets the household income "
        "standard and resource test."
    )
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/regulation/7/273/9
rules:
  - name: household_snap_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    source: 7 CFR 273.9
    versions:
      - effective_from: '0001-01-01'
        formula: household_income_eligible and household_resource_eligible
"""

    assert find_admin_agency_aggregate_entity_issues(content, source_text) == []


def test_admin_agency_aggregate_allows_long_income_exclusion_list():
    source_text = (
        "P.L. No. 100-175, Section 166, Older Americans Act. Funds received by "
        "persons fifty-five (55) years of age and older under the Senior "
        "Community Service Employment Program under Title V of the Older "
        "Americans Act are excluded from income. State agencies and eight "
        "organizations receive funding under Title V. "
        + ("Separate income exclusion text. " * 40)
        + "P.L. No. 101-508 amended Section 402(i) of the Social Security Act. "
        "At-risk block grant child care payments are excluded from being "
        "counted as income for SNAP purposes."
    )
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-co/regulation/10-ccr-2506-1/4.405.2
rules:
  - name: payment_excluded_as_income
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Month
    source: 10 CCR 2506-1, 4.405.2
    versions:
      - effective_from: '0001-01-01'
        formula: senior_community_service_payment or at_risk_child_care_payment
"""

    assert find_admin_agency_aggregate_entity_issues(content, source_text) == []


def test_source_identifier_maps_federal_regulation_to_cfr_repo_path():
    assert _source_identifier_to_relative_rulespec_path(
        "us/regulation/7/273/10"
    ) == Path("regulations/7-cfr/273/10.yaml")


def test_source_identifier_maps_federal_form_to_allowed_policy_repo_path():
    assert _source_identifier_to_relative_rulespec_path(
        "us/form/cms/medicaid-chip-bhp-eligibility-levels"
    ) == Path("policies/cms/medicaid-chip-bhp-eligibility-levels.yaml")


def test_resolve_corpus_source_unit_accepts_form_citation_path(tmp_path):
    citation = "us/form/cms/medicaid-chip-bhp-eligibility-levels"
    corpus_release = _write_test_corpus_provision(
        tmp_path,
        citation_path=citation,
        body="CMS Medicaid, CHIP, and BHP eligibility levels table",
    )

    source_unit = resolve_corpus_source_unit(citation, corpus_release)

    assert source_unit.citation_path == citation
    assert source_unit.source == "local"
    assert source_unit.body == "CMS Medicaid, CHIP, and BHP eligibility levels table"


def test_resolve_corpus_source_unit_slices_before_bracketed_sibling(tmp_path):
    corpus_release = _write_test_corpus_provision(
        tmp_path,
        citation_path="us/statute/26/3306",
        body=(
            "(k) Agricultural labor For purposes of this chapter, the term "
            "agricultural labor has the meaning assigned by section 3121(g).\n\n"
            "[(l) Repealed. Sept. 1, 1954.]\n\n"
            "(m) American vessel and aircraft For purposes of this chapter, "
            "the term American vessel means a documented vessel."
        ),
    )

    source_unit = resolve_corpus_source_unit("26 USC 3306(k)", corpus_release)

    assert source_unit.citation_path == "us/statute/26/3306"
    assert source_unit.body.startswith("(k) Agricultural labor")
    assert "[(l) Repealed" not in source_unit.body
    assert "(m) American vessel" not in source_unit.body


def test_resolve_corpus_source_unit_slices_bracketed_repealed_subsection(tmp_path):
    corpus_release = _write_test_corpus_provision(
        tmp_path,
        citation_path="us/statute/26/3306",
        body=(
            "(k) Agricultural labor For purposes of this chapter, the term "
            "agricultural labor has the meaning assigned by section 3121(g).\n\n"
            "[(l) Repealed. Sept. 1, 1954.]\n\n"
            "(m) American vessel and aircraft For purposes of this chapter, "
            "the term American vessel means a documented vessel."
        ),
    )

    source_unit = resolve_corpus_source_unit("26 USC 3306(l)", corpus_release)

    assert source_unit.citation_path == "us/statute/26/3306"
    assert source_unit.body == "[(l) Repealed. Sept. 1, 1954.]"


def test_resolve_corpus_source_unit_uses_form_child_blocks(tmp_path):
    citation = "us/form/cms/medicaid-chip-bhp-eligibility-levels"
    corpus_release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": citation,
                "body": None,
                "heading": "Medicaid, CHIP, and BHP Eligibility Levels",
                "level": 1,
                "ordinal": 1,
            },
            {
                "citation_path": f"{citation}/block-1",
                "body": "Colorado 142% 142% 142% 260% 195% 260% 68% 133%",
                "heading": "State Medicaid, CHIP and BHP Income Eligibility Standards",
                "level": 2,
                "ordinal": 1,
            },
        ],
    )

    source_unit = resolve_corpus_source_unit(citation, corpus_release)

    assert source_unit.citation_path == citation
    assert source_unit.source == "local"
    assert (
        "State Medicaid, CHIP and BHP Income Eligibility Standards" in source_unit.body
    )
    assert "Colorado 142% 142% 142% 260% 195% 260% 68% 133%" in source_unit.body


def test_resolve_corpus_source_unit_uses_headingless_child_blocks(tmp_path):
    citation = "us/regulation/42/435/119"
    corpus_release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": citation,
                "body": None,
                "heading": "Coverage for adults",
                "level": 1,
                "ordinal": 1,
            },
            {
                "citation_path": f"{citation}/b",
                "body": "The agency must provide Medicaid to adults.",
                "heading": None,
                "level": 2,
                "ordinal": 2,
            },
            {
                "citation_path": f"{citation}/a",
                "body": "This section applies beginning January 1, 2014.",
                "heading": None,
                "level": 2,
                "ordinal": 1,
            },
        ],
    )

    source_unit = resolve_corpus_source_unit(citation, corpus_release)

    assert source_unit.citation_path == citation
    assert source_unit.source == "local"
    assert source_unit.body == (
        "This section applies beginning January 1, 2014.\n\n"
        "The agency must provide Medicaid to adults."
    )


def test_resolve_corpus_source_unit_parses_bare_cfr_citation(tmp_path):
    citation = "us/regulation/42/435/119"
    corpus_release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": citation,
                "body": (
                    "(a) Basis. This section implements coverage for adults.\n\n"
                    "(b) Eligibility. The agency must provide Medicaid to adults."
                ),
                "heading": "Coverage for adults",
                "level": 2,
                "ordinal": 119,
            }
        ],
    )

    source_unit = resolve_corpus_source_unit("42 CFR 435.119(b)", corpus_release)

    assert source_unit.citation_path == citation
    assert source_unit.source == "local"
    assert source_unit.body == (
        "(b) Eligibility. The agency must provide Medicaid to adults."
    )


def test_resolve_corpus_source_unit_ignores_cfr_through_references(tmp_path):
    citation = "us/regulation/42/435/601"
    corpus_release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": citation,
                "body": (
                    "(d) Use of less restrictive methodologies.\n\n"
                    "(1) At State option, and subject to the conditions of "
                    "paragraphs (d)(2) through (5) of this section, the agency "
                    "may apply less restrictive methodologies.\n\n"
                    "(2) The methodologies may be less restrictive but no more "
                    "restrictive than SSI methodologies.\n\n"
                    "(3) A methodology is no more restrictive if additional "
                    "individuals may be eligible and none are made ineligible.\n\n"
                    "(4) The methodology must be comparable within each category.\n\n"
                    "(5) The methodology must be consistent with subpart K FFP "
                    "limitations."
                ),
                "heading": "Financial methodologies",
                "level": 2,
                "ordinal": 601,
            }
        ],
    )

    paragraph_one = resolve_corpus_source_unit(
        "us/regulation/42/435/601/d/1",
        corpus_release,
    )
    paragraph_five = resolve_corpus_source_unit(
        "us/regulation/42/435/601/d/5",
        corpus_release,
    )

    assert paragraph_one.body.startswith("(1) At State option")
    assert "paragraphs (d)(2) through (5)" in paragraph_one.body
    assert "(2) The methodologies may be" not in paragraph_one.body
    assert paragraph_five.body == (
        "(5) The methodology must be consistent with subpart K FFP limitations."
    )


def test_resolve_corpus_source_unit_slices_cfr_top_level_with_inline_child(
    tmp_path,
):
    citation = "us/regulation/42/435/602"
    corpus_release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": citation,
                "body": (
                    "(a)(1) This section only applies to MAGI-excepted "
                    "individuals.\n\n"
                    "(2) Basic requirements. The agency must apply these "
                    "requirements:\n\n"
                    "(i) Except for spouses and parents, the agency must not "
                    "consider relative income.\n\n"
                    "(ii) For individuals under age 21, title IV-A rules apply.\n\n"
                    "(b) Requirements for States using more restrictive "
                    "requirements. The agency must apply SSI relative "
                    "responsibility rules.\n\n"
                    "(1) SSI relative responsibility rules apply; or\n\n"
                    "(2) More extensive requirements may apply.\n\n"
                    "(c) Use of less restrictive methodologies. The agency may "
                    "apply less restrictive methodologies."
                ),
                "heading": "Financial responsibility of relatives",
                "level": 2,
                "ordinal": 602,
            }
        ],
    )

    subsection_a = resolve_corpus_source_unit(
        "us/regulation/42/435/602/a",
        corpus_release,
    )
    paragraph_one = resolve_corpus_source_unit(
        "us/regulation/42/435/602/a/1",
        corpus_release,
    )
    clause_i = resolve_corpus_source_unit(
        "us/regulation/42/435/602/a/2/i",
        corpus_release,
    )
    subsection_b = resolve_corpus_source_unit(
        "us/regulation/42/435/602/b",
        corpus_release,
    )

    assert subsection_a.body.startswith("(a)(1) This section only applies")
    assert "(b) Requirements for States" not in subsection_a.body
    assert paragraph_one.body == (
        "(1) This section only applies to MAGI-excepted individuals."
    )
    assert clause_i.body == (
        "(i) Except for spouses and parents, the agency must not consider "
        "relative income."
    )
    assert subsection_b.body.startswith("(b) Requirements for States")
    assert "(1) SSI relative responsibility rules apply" in subsection_b.body
    assert "(c) Use of less restrictive methodologies" not in subsection_b.body


def test_canonical_target_ref_prefix_handles_canonical_source_id():
    assert (
        _canonical_target_ref_prefix(
            "us:regulations/7-cfr/273/9/d/6/iii",
            Path("regulations/7-cfr/273/9/d/6/iii.yaml"),
        )
        == "us:regulations/7-cfr/273/9/d/6/iii"
    )


def test_canonical_target_ref_prefix_uses_policy_repo_for_repo_relative_source_id(
    tmp_path,
):
    repo = _canonical_rulespec_content_root(tmp_path, "us-ny")

    assert (
        _canonical_target_ref_prefix(
            "regulations/18-nycrr/387/14/a/1",
            Path("regulations/18-nycrr/387/14/a/1.yaml"),
            policy_repo_path=repo,
        )
        == "us-ny:regulations/18-nycrr/387/14/a/1"
    )


def test_canonical_target_ref_prefix_rejects_flat_policy_repo(tmp_path):
    flat_repo = tmp_path / "rulespec-us-ny"
    flat_repo.mkdir()

    assert (
        _canonical_target_ref_prefix(
            "regulations/18-nycrr/387/14/a/1",
            Path("regulations/18-nycrr/387/14/a/1.yaml"),
            policy_repo_path=flat_repo,
        )
        is None
    )


def test_canonical_rulespec_target_for_path_uses_jurisdiction_root(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us-co")
    rules_file = policy_repo / "regulations/10-ccr-2506-1/4.130.1.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")

    assert _canonical_rulespec_target_for_path(rules_file) == (
        "us-co:regulations/10-ccr-2506-1/4.130.1"
    )


def test_canonical_rulespec_target_for_path_rejects_flat_layout(tmp_path):
    rules_file = (
        tmp_path / "rulespec-us-co" / "regulations" / "10-ccr-2506-1" / "4.130.1.yaml"
    )
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")

    assert _canonical_rulespec_target_for_path(rules_file) is None


def test_canonical_target_ref_prefix_omits_repo_relative_source_without_repo():
    assert (
        _canonical_target_ref_prefix(
            "regulations/18-nycrr/387/14/a/1",
            Path("regulations/18-nycrr/387/14/a/1.yaml"),
        )
        is None
    )


def test_subparagraph_coverage_checklist_requires_exact_corpus_source_keys():
    checklist = _format_subparagraph_coverage_checklist(
        "(a) First category is eligible.\n(b) Second category is ineligible.",
        "us-ny/regulation/18-nycrr/387/14/a/5",
    )

    assert "copy the relevant string exactly" in checklist
    assert "human-readable source like `18 NYCRR 387.14(a)(5)(i)(a)`" in checklist
    assert "us-ny/regulation/18-nycrr/387/14/a/5(a)" in checklist


@pytest.mark.parametrize(
    "citation,expected",
    [
        # Issue #71: dot-separated CDSS-style citations must keep subsection
        # identity in the output path. Before the fix, every sibling collapsed
        # onto the section-level path because pathlib's with_suffix() treated
        # the dotted leaf as a file extension.
        ("us-ca/regulation/mpp/63-503", "regulations/mpp/63-503.yaml"),
        ("us-ca/regulation/mpp/63-503.1", "regulations/mpp/63-503/1.yaml"),
        ("us-ca/regulation/mpp/63-503.131", "regulations/mpp/63-503/131.yaml"),
        ("us-ca/regulation/mpp/63-503.132", "regulations/mpp/63-503/132.yaml"),
        ("us-ca/regulation/mpp/63-300.234", "regulations/mpp/63-300/234.yaml"),
        # Deeper dotted nesting must also split correctly.
        (
            "us-ca/regulation/mpp/63-503.131.a",
            "regulations/mpp/63-503/131/a.yaml",
        ),
        # Colorado CCR section numbers use dotted file stems in rulespec-us/us-co;
        # splitting the leaf would create a parallel wrong tree like 4/207/2.yaml.
        (
            "us-co/regulation/10-ccr-2506-1/4.207.2",
            "regulations/10-ccr-2506-1/4.207.2.yaml",
        ),
        (
            "us-co/regulation/10-ccr-2506-1/4.403.11/b/c/3",
            "regulations/10-ccr-2506-1/4.403.11/b/c/3.yaml",
        ),
        # Colorado statutes also use dotted legal labels for sections and
        # subsections; those labels are file stems, not nested MPP-style parts.
        (
            "us-co/statute/39/39-22-104.5",
            "statutes/39/39-22-104.5.yaml",
        ),
        (
            "us-co/statute/39/39-22-104/1.5",
            "statutes/39/39-22-104/1.5.yaml",
        ),
        # Louisiana's corpus preserves the official R.S. title:section label,
        # while RuleSpec represents the separator as a directory boundary.
        ("us-la/statute/47:294", "statutes/47/294.yaml"),
        ("us-la/statute/47:297.4", "statutes/47/297/4.yaml"),
        # Colon expansion is a Louisiana source convention, not a generic
        # rewrite for every jurisdiction.
        ("us-tx/statute/1:2", "statutes/1:2.yaml"),
        # Slash-separated citations (USC, NYCRR, CFR) are unaffected — these
        # are regression cases for the dot-stripping fix.
        (
            "us-ny/regulation/18-nycrr/387/14/a/1",
            "regulations/18-nycrr/387/14/a/1.yaml",
        ),
        ("us/statute/26/1/a/1", "statutes/26/1/a/1.yaml"),
        ("us/regulation/7/273/8", "regulations/7-cfr/273/8.yaml"),
        (
            "uk/statute/legislation.gov.uk/ukpga/1992/4/section/8",
            "statutes/ukpga/1992/4/8.yaml",
        ),
        (
            "uk/regulation/legislation.gov.uk/uksi/2013/376/regulation/36",
            "regulations/uksi/2013/376/36.yaml",
        ),
        (
            "uk/regulation/legislation.gov.uk/uksi/2013/376/schedule/5/paragraph/2",
            "regulations/uksi/2013/376/schedule/5/paragraph/2.yaml",
        ),
    ],
)
def test_source_identifier_handles_dotted_leaf_segments(citation, expected):
    assert str(_source_identifier_to_relative_rulespec_path(citation)) == expected


@pytest.mark.parametrize(
    "dash",
    [
        "\u2010",
        "\u2011",
        "\u2012",
        "\u2013",
        "\u2014",
        "\u2015",
        "\u2212",
        "\ufe58",
        "\ufe63",
        "\uff0d",
    ],
)
def test_source_identifier_normalizes_unicode_dashes_only_in_output_path(dash):
    citation = f"us/statute/42/1437c{dash}1"

    assert _source_identifier_to_relative_rulespec_path(citation) == Path(
        "statutes/42/1437c-1.yaml"
    )


def test_slash_usc_alias_normalizes_unicode_dash_in_output_path():
    assert _resolve_eval_output_path(
        "42/1437c\u20131",
        fallback=citation_to_relative_rulespec_path,
    ) == Path("statutes/42/1437c-1.yaml")


def test_structured_usc_citation_normalizes_only_output_path():
    citation = CitationParts(title="42", section="1437c\u20131", fragments=("d",))

    assert citation_to_relative_rulespec_path(citation) == Path(
        "statutes/42/1437c-1/d.yaml"
    )
    assert citation.section == "1437c\u20131"


@pytest.mark.parametrize(
    "section",
    [
        "39-28.5-107",
        "39-30.5-104",
        "39-22-123.5",
        "39-1-104.5",
        "39-26-702",
    ],
)
def test_colorado_statute_paths_are_hyphen_structured_and_dot_atomic(tmp_path, section):
    citation = f"us-co/statute/39/{section}"
    module = _source_identifier_to_relative_rulespec_path(citation)
    companion = module.with_name(f"{module.stem}.test.yaml")

    assert module == Path(f"statutes/39/{section}.yaml")
    assert companion == Path(f"statutes/39/{section}.test.yaml")

    # Applied manifests mirror the complete module path. Exercise the same
    # transformation here so a dotted article cannot diverge at apply time.
    from axiom_encode.cli import _applied_encoding_manifest_path

    assert _applied_encoding_manifest_path(Path("us-co") / module) == Path(
        f".axiom/encoding-manifests/us-co/statutes/39/{section}.json"
    )

    # The reverse index reconstructs the canonical target from the filename;
    # resolving that citation again must recover the byte-identical path.
    content_root = _canonical_rulespec_content_root(tmp_path, "us-co")
    rules_file = content_root / module
    rules_file.parent.mkdir(parents=True, exist_ok=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")
    reversed_citation = _canonical_rulespec_target_for_path(rules_file)

    assert reversed_citation == f"us-co:statutes/39/{section}"
    assert _source_identifier_to_relative_rulespec_path(reversed_citation) == module


def test_resolve_eval_output_path_uses_path_like_citation_directly():
    """Sanity: a citation that already looks like a corpus path is used as-is."""
    from axiom_encode.harness.evals import _resolve_eval_output_path

    assert _resolve_eval_output_path("us/statute/7/2014/e/2/B") == Path(
        "statutes/7/2014/e/2/B.yaml"
    )


def test_resolve_eval_output_path_parses_bare_cfr_citation():
    from axiom_encode.harness.evals import _resolve_eval_output_path

    assert _resolve_eval_output_path("42 CFR 435.119") == Path(
        "regulations/42-cfr/435/119.yaml"
    )
    assert _resolve_eval_output_path("42 C.F.R. § 435.119(b)(5)") == Path(
        "regulations/42-cfr/435/119/b/5.yaml"
    )


def test_resolve_eval_output_path_uses_repo_relative_source_root_directly():
    """The internal path mapper recognizes RuleSpec source-root paths."""
    from axiom_encode.harness.evals import _resolve_eval_output_path

    assert _resolve_eval_output_path(
        "policies/otda/snap/fy-2026-benefit-calculation",
    ) == Path("policies/otda/snap/fy-2026-benefit-calculation.yaml")


def test_resolve_eval_output_path_uses_colon_prefixed_rulespec_identifier(tmp_path):
    repo = _canonical_rulespec_content_root(tmp_path, "us-co")

    identifier = "us-co:regulations/10-ccr-2506-1/4.804.1"
    relative_output = _resolve_eval_output_path(identifier)

    assert relative_output == Path("regulations/10-ccr-2506-1/4.804.1.yaml")
    assert (
        _canonical_target_ref_prefix(
            identifier,
            relative_output,
            policy_repo_path=repo,
        )
        == "us-co:regulations/10-ccr-2506-1/4.804.1"
    )


def test_resolve_eval_output_path_rejects_free_text_source_identity():
    with pytest.raises(ValueError, match="not a canonical citation path"):
        _resolve_eval_output_path(
            "SNAP earned income deduction under 7 USC 2014(e)(2)(B)"
        )


def test_resolve_eval_output_path_uses_exact_canonical_path():
    assert _resolve_eval_output_path("us/statute/26/63") == Path("statutes/26/63.yaml")


@pytest.mark.parametrize(
    "citation,expected",
    [
        ("us-la/statute/47:294", "statutes/47/294.yaml"),
        ("us-la/statute/47:297.4", "statutes/47/297/4.yaml"),
    ],
)
def test_resolve_eval_output_path_expands_louisiana_title_section_separator(
    citation, expected
):
    assert _resolve_eval_output_path(citation) == Path(expected)


def test_resolve_eval_output_path_rejects_empty_colon_statute_component():
    with pytest.raises(ValueError, match="Invalid Louisiana title:section"):
        _resolve_eval_output_path("us-la/statute/47::294")


def test_target_rel_preserves_colon_prefixed_louisiana_jurisdiction():
    assert _target_rel_for_eval_identifier("us-la:statute/47:294") == Path(
        "statutes/47/294.yaml"
    )


@pytest.mark.parametrize(
    "citation",
    [
        "us-la/statute/47:..",
        "us-la/statute/47:294/../outside",
        "us-la/statute/47:294:outside",
        "us-la/statute/47:294/sub:section",
        "us-la/statute/47:.294",
        "us-la/statute/47:294.",
        "us-la/statute/47:294..4",
        "us-la/statute/47:294/.subsection",
        "us-la/statute/47:294/subsection.",
        "us-la/statute/47:294/sub..section",
    ],
)
def test_resolve_eval_output_path_rejects_unsafe_louisiana_components(citation):
    with pytest.raises(ValueError):
        _resolve_eval_output_path(citation)


def test_contained_eval_output_file_rejects_runner_root_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes runner root"):
        _contained_eval_output_file(tmp_path, "runner", Path("../../outside.yaml"))


def test_contained_eval_output_file_rejects_output_root_escape(tmp_path):
    with pytest.raises(ValueError, match="runner path escapes output root"):
        _contained_eval_output_file(tmp_path, "../outside", Path("artifact.yaml"))


def test_secure_eval_read_rejects_fifo_without_blocking(tmp_path):
    os.mkfifo(tmp_path / "artifact.yaml")

    with pytest.raises(ValueError, match="not a regular file"):
        _secure_eval_read(tmp_path, Path("artifact.yaml"))


def test_contained_eval_output_file_preserves_lexical_target_symlink(tmp_path):
    runner_root = tmp_path / "runner"
    canonical = runner_root / "statutes" / "26" / "1.yaml"
    requested = runner_root / "statutes" / "47" / "294.yaml"
    canonical.parent.mkdir(parents=True)
    requested.parent.mkdir(parents=True)
    canonical.write_text("canonical sentinel\n", encoding="utf-8")
    requested.symlink_to(canonical)

    output_file = _contained_eval_output_file(
        tmp_path,
        "runner",
        Path("statutes/47/294.yaml"),
    )
    wrote = _materialize_eval_artifact(
        "format: rulespec/v1\nrules: []\n",
        output_file,
        artifact_root=tmp_path,
    )

    assert wrote is True
    assert output_file == requested
    assert not requested.is_symlink()
    assert canonical.read_text(encoding="utf-8") == "canonical sentinel\n"


class TestCorpusSourceResolution:
    def test_resolves_state_manual_corpus_path_without_statute_rewrite(self, tmp_path):
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-az/manual/des/faa5/na-child-support-expense/block-2",
            body="manual child support text",
        )

        source = resolve_corpus_source_unit(
            "us-az/manual/des/faa5/na-child-support-expense/block-2",
            corpus_release,
        )

        assert source.citation_path == (
            "us-az/manual/des/faa5/na-child-support-expense/block-2"
        )
        assert source.body == "manual child support text"
        assert source.source == "local"

    def test_resolves_state_statute_child_path_to_sliced_section_provision(
        self, tmp_path
    ):
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-co/statute/39/39-22-104",
            body=(
                "(1.7) (a) A prior rate applies.\n"
                "(b) A second prior rate applies.\n"
                "(c) Except as otherwise provided, a tax of four and "
                "forty one-hundredths percent is imposed.\n"
                "(2) Federal taxable income shall be modified before the rate."
            ),
        )

        source = resolve_corpus_source_unit(
            "us-co/statute/39/39-22-104/1.7/c",
            corpus_release,
        )

        assert source.citation_path == "us-co/statute/39/39-22-104"
        assert source.body == (
            "(c) Except as otherwise provided, a tax of four and "
            "forty one-hundredths percent is imposed."
        )
        assert "(2) Federal taxable income" not in source.body

    def test_resolves_state_statute_child_path_stops_at_dotted_alpha_sibling(
        self, tmp_path
    ):
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-co/statute/39/39-22-104",
            body=(
                "(4) There shall be subtracted from federal taxable income:\n"
                "(a) United States obligation interest included in federal "
                "taxable income.\n"
                "(a.5) Repealed.\n"
                "(b) Basis adjustment subtraction.\n"
                "(3) There shall be added to federal taxable income:\n"
                "(p) Itemized deduction addback.\n"
                "(p.5) Healthy school meals deduction addback.\n"
                "(p.5) Alternate healthy school meals deduction addback.\n"
                "(p.7) Additional healthy school meals deduction addback.\n"
                "(q) Food and beverage expense addback."
            ),
        )

        us_interest = resolve_corpus_source_unit(
            "us-co/statute/39/39-22-104/4/a",
            corpus_release,
        )
        itemized_addback = resolve_corpus_source_unit(
            "us-co/statute/39/39-22-104/3/p",
            corpus_release,
        )
        healthy_school_meals_addback = resolve_corpus_source_unit(
            "us-co/statute/39/39-22-104/3/p/5",
            corpus_release,
        )

        assert us_interest.body == (
            "(a) United States obligation interest included in federal taxable income."
        )
        assert "(a.5) Repealed" not in us_interest.body
        assert itemized_addback.body == "(p) Itemized deduction addback."
        assert "(p.5) Healthy school meals" not in itemized_addback.body
        assert healthy_school_meals_addback.body.startswith(
            "(p.5) Healthy school meals deduction addback."
        )
        assert (
            "(p.5) Alternate healthy school meals" in healthy_school_meals_addback.body
        )
        assert (
            "(p.7) Additional healthy school meals"
            not in healthy_school_meals_addback.body
        )
        assert "(q) Food and beverage" not in healthy_school_meals_addback.body

    def test_resolves_nested_child_before_dotted_sibling_fallback(self, tmp_path):
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-co/statute/39/39-22-104",
            body=(
                "(4) There shall be subtracted from federal taxable income:\n"
                "(a) United States obligation interest.\n"
                "(1) Nested qualifying amount.\n"
                "(a.1) Dotted sibling subtraction.\n"
                "(b) Basis adjustment subtraction."
            ),
        )

        source = resolve_corpus_source_unit(
            "us-co/statute/39/39-22-104/4/a/1",
            corpus_release,
        )

        assert source.body == "(1) Nested qualifying amount."
        assert "(a.1) Dotted sibling" not in source.body

    def test_resolves_alpha_child_path_stops_at_later_omitted_sibling(self, tmp_path):
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-co/statute/39/39-22-104",
            body=(
                "(1) Tax is imposed as follows:\n"
                "(a) First rate period.\n"
                "(c) Third rate period after omitted subsection."
            ),
        )

        source = resolve_corpus_source_unit(
            "us-co/statute/39/39-22-104/1/a",
            corpus_release,
        )

        assert source.body == "(a) First rate period."
        assert "(c) Third rate period" not in source.body

    def test_resolves_numeric_child_path_stops_at_later_omitted_sibling(self, tmp_path):
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-co/statute/39/39-22-104",
            body=(
                "(1) First addition rule.\n"
                "(3) Third addition rule after omitted subsection."
            ),
        )

        source = resolve_corpus_source_unit(
            "us-co/statute/39/39-22-104/1",
            corpus_release,
        )

        assert source.body == "(1) First addition rule."
        assert "(3) Third addition rule" not in source.body

    def test_resolves_usc_child_citation_to_sliced_section_provision(self, tmp_path):
        corpus_release = _write_test_corpus_release(
            tmp_path,
            [
                {
                    "citation_path": "us/statute/26/3101",
                    "body": (
                        "(a) Old-age, survivors, and disability insurance "
                        "states 6.2 percent.\n\n"
                        "(b) Hospital insurance states 1.45 percent."
                    ),
                }
            ],
        )

        source = resolve_corpus_source_unit("26 USC 3101(a)", corpus_release)

        assert source.citation_path == "us/statute/26/3101"
        assert source.body == (
            "(a) Old-age, survivors, and disability insurance states 6.2 percent."
        )
        assert source.source == "local"

    def test_resolves_nested_usc_child_citation_to_sliced_section_provision(
        self, tmp_path
    ):
        corpus_release = _write_test_corpus_release(
            tmp_path,
            [
                {
                    "citation_path": "us/statute/7/2015",
                    "body": (
                        "(a) General eligibility.\n\n"
                        "(d) Work requirements\n\n(1) Paragraph one.\n\n"
                        "(2) Exemptions\n\n(A) First exemption.\n\n"
                        "(B) Second exemption.\n\n"
                        "(C) Student exemption states 20 hours.\n\n"
                        "(D) Next exemption.\n\n"
                        "(3) Other work rule.\n\n"
                        "(e) Students."
                    ),
                }
            ],
        )

        source = resolve_corpus_source_unit("7 USC 2015(d)(2)(C)", corpus_release)

        assert source.citation_path == "us/statute/7/2015"
        assert source.body == "(C) Student exemption states 20 hours."

    def test_resolves_nested_cfr_child_path_to_sliced_section_provision(self, tmp_path):
        corpus_release = _write_test_corpus_release(
            tmp_path,
            [
                {
                    "citation_path": "us/regulation/7/273/9",
                    "body": (
                        "(a) Income standards.\n\n"
                        "(d) Deductions.\n\n"
                        "(5) Child support deduction. "
                        "(i) Not a top-level sibling.\n\n"
                        "(6) Shelter costs--"
                        "(i) Homeless shelter deduction. "
                        "(ii) Excess shelter deduction. "
                        "(iii) Standard utility allowances. "
                        "(A) Utility standard. "
                        "(1) Heating. (2) Cooling. (3) Other utilities.\n\n"
                        "(e) Benefit calculation."
                    ),
                }
            ],
        )

        source = resolve_corpus_source_unit(
            "us/regulation/7/273/9/d/6",
            corpus_release,
        )

        assert source.citation_path == "us/regulation/7/273/9"
        assert source.body.startswith("(6) Shelter costs")
        assert "(3) Other utilities" in source.body
        assert "(5) Child support deduction" not in source.body
        assert "(e) Benefit calculation" not in source.body

    def test_nested_slicing_ignores_parenthetical_cross_reference_list(self, tmp_path):
        corpus_release = _write_test_corpus_release(
            tmp_path,
            [
                {
                    "citation_path": "us/statute/26/63",
                    "body": (
                        "(c) Standard deduction "
                        "(4) Adjustments for inflation Each dollar amount "
                        "contained in paragraph (2)(B), (2)(C), or (5) or "
                        "subsection (f) shall be increased. "
                        "(5) Limitation on basic standard deduction in the case "
                        "of certain dependents In the case of an individual with "
                        "respect to whom a deduction under section 151 is "
                        "allowable to another taxpayer, the basic standard "
                        "deduction shall not exceed the greater of— (A) $500, "
                        "or (B) the sum of $250 and earned income. "
                        "(6) Certain individuals, etc., not eligible for "
                        "standard deduction."
                    ),
                }
            ],
        )

        source = resolve_corpus_source_unit("26 USC 63(c)(5)", corpus_release)

        assert source.citation_path == "us/statute/26/63"
        assert source.body.startswith("(5) Limitation on basic standard deduction")
        assert "paragraph (2)(B)" not in source.body

    def test_nested_slicing_ignores_plural_parenthetical_cross_reference_list(
        self, tmp_path
    ):
        corpus_release = _write_test_corpus_release(
            tmp_path,
            [
                {
                    "citation_path": "us/statute/26/63",
                    "body": (
                        "(c) Standard deduction "
                        "(4) Adjustments for inflation Each dollar amount "
                        "contained in paragraphs (2)(B), (2)(C), or (5) or "
                        "subsection (f) shall be increased. "
                        "(5) Limitation on basic standard deduction in the case "
                        "of certain dependents In the case of an individual with "
                        "respect to whom a deduction under section 151 is "
                        "allowable to another taxpayer, the basic standard "
                        "deduction shall not exceed the greater of— (A) $500, "
                        "or (B) the sum of $250 and earned income. "
                        "(6) Certain individuals, etc., not eligible for "
                        "standard deduction."
                    ),
                }
            ],
        )

        source = resolve_corpus_source_unit("26 USC 63(c)(5)", corpus_release)

        assert source.citation_path == "us/statute/26/63"
        assert source.body.startswith("(5) Limitation on basic standard deduction")
        assert "paragraphs (2)(B)" not in source.body

    def test_build_prompt_requires_resolved_corpus_locator(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="26 USC 3101(a)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Section text states 6.2 percent.",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            source_metadata_payload={
                "source_attestation": {
                    "requested_corpus_citation_path": "us/statute/26/3101"
                },
            },
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 3101(a)",
            "cold",
            workspace,
            [],
            target_file_name="a.yaml",
            include_tests=True,
            runner_backend="codex",
        )

        assert "read from `corpus.provisions` at `us/statute/26/3101`" in prompt
        assert (
            "module.source_verification.corpus_citation_path: us/statute/26/3101"
            in prompt
        )
        assert "Do not emit `source_url`" in prompt

    def test_child_slice_prompt_uses_requested_corpus_locator(self, tmp_path):
        corpus_release = _write_test_corpus_release(
            tmp_path,
            [
                {
                    "citation_path": "us-ca/statute/wic/11450",
                    "body": (
                        "(a) (1) (A) Aid shall be paid after deducting income. "
                        "(B) Federal contribution adjustment. "
                        "(2) Cost-of-living adjustment pause. "
                        "(b) Pregnancy aid."
                    ),
                }
            ],
        )

        source = resolve_corpus_source_unit(
            "us-ca/statute/wic/11450/a/1/A", corpus_release
        )

        assert source.citation_path == "us-ca/statute/wic/11450"
        assert source.body.startswith("(A) Aid shall be paid")
        assert _prompt_corpus_citation_path(source) == ("us-ca/statute/wic/11450/a/1/A")

        workspace = prepare_eval_workspace(
            citation="policies/cdss/calworks/monthly-aid-payment",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=source.body,
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us-ca"),
            mode="cold",
            source_metadata_payload={
                "source_attestation": dict(source.source_attestation),
            },
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "policies/cdss/calworks/monthly-aid-payment",
            "cold",
            workspace,
            [],
            target_file_name="monthly-aid-payment.yaml",
            include_tests=True,
            runner_backend="codex",
        )

        assert (
            "read from `corpus.provisions` at `us-ca/statute/wic/11450/a/1/A`"
        ) in prompt
        assert (
            "module.source_verification.corpus_citation_path: "
            "us-ca/statute/wic/11450/a/1/A"
        ) in prompt

    def test_workspace_writes_corpus_source_metadata_payload(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="26 USC 3101(a)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Section text states 6.2 percent.",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            source_metadata_payload={
                "source_name": "Federal Insurance Contributions Act",
                "source_attestation": {
                    "requested_corpus_citation_path": "us/statute/26/3101"
                },
            },
            extra_context_paths=[],
        )

        assert workspace.source_metadata is not None
        assert workspace.source_metadata["source_name"] == (
            "Federal Insurance Contributions Act"
        )
        assert (
            workspace.source_metadata["source_attestation"][
                "requested_corpus_citation_path"
            ]
            == "us/statute/26/3101"
        )

    def test_generation_result_fails_when_post_encode_ci_fails(self, tmp_path):
        corpus_release, source_unit = _write_test_source_unit(tmp_path, "source")
        response = Mock()
        response.text = (
            "=== FILE: sample.yaml ===\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: source\n"
            "rules: []\n"
            "=== FILE: sample.test.yaml ===\n"
            "[]\n"
        )
        response.duration_ms = 1
        response.tokens = None
        response.estimated_cost_usd = None
        response.actual_cost_usd = None
        response.trace = {}
        response.unexpected_accesses = []
        response.error = None

        with (
            patch("axiom_encode.harness.evals._run_prompt_eval", return_value=response),
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
                return_value=EvalArtifactMetrics(
                    compile_pass=True,
                    compile_issues=[],
                    ci_pass=False,
                    ci_issues=["missing corpus source verification"],
                    embedded_source_present=True,
                    grounded_numeric_count=0,
                    ungrounded_numeric_count=0,
                    grounding=[],
                ),
            ),
        ):
            [result] = run_source_eval(
                source_unit=source_unit,
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                policy_path=_canonical_rulespec_content_root(tmp_path, "us"),
                local_corpus_release=corpus_release,
                runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
                mode="cold",
            )

        assert result.success is False
        assert result.error == "Generated RuleSpec failed CI validation"


class TestClaudePromptEval:
    def test_prompt_eval_uses_configurable_encoder_timeout_and_records_it(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setenv("AXIOM_ENCODE_ENCODER_TIMEOUT_SECONDS", "1234")
        runner = parse_runner_spec("claude:opus")
        workspace = EvalWorkspace(
            root=tmp_path,
            source_text_file=tmp_path / "source.txt",
            manifest_file=tmp_path / "context-manifest.json",
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"result": "review complete", "usage": {}}),
            stderr="",
        )

        with patch(
            "axiom_encode.harness.evals.subprocess.run",
            return_value=completed,
        ) as mock_run:
            response = _run_claude_prompt_eval(runner, workspace, "review this")

        assert mock_run.call_args.kwargs["timeout"] == 1234
        assert response.trace["timed_out"] is False
        assert response.trace["timeout_reason"] is None
        assert response.trace["timeout_seconds"] == 1234

    def test_prompt_eval_returns_structured_timeout_failure(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setenv("AXIOM_ENCODE_ENCODER_TIMEOUT_SECONDS", "1234")
        runner = parse_runner_spec("claude:opus")
        workspace = EvalWorkspace(
            root=tmp_path,
            source_text_file=tmp_path / "source.txt",
            manifest_file=tmp_path / "context-manifest.json",
        )

        with patch(
            "axiom_encode.harness.evals.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["claude"], timeout=1234),
        ):
            response = _run_claude_prompt_eval(runner, workspace, "review this")

        assert response.text == ""
        assert response.error == "Claude eval timed out"
        assert response.trace == {
            "provider": "anthropic",
            "backend": "claude-print",
            "model": "opus",
            "timed_out": True,
            "timeout_reason": "wall",
            "timeout_seconds": 1234,
        }

    def test_claude_prompt_eval_rejects_success_returned_after_case_deadline(
        self,
        tmp_path,
        monkeypatch,
    ):
        runner = parse_runner_spec("claude:opus")
        workspace = EvalWorkspace(
            root=tmp_path,
            source_text_file=tmp_path / "source.txt",
            manifest_file=tmp_path / "context-manifest.json",
        )
        clock = [100.0]
        monkeypatch.setattr(evals_module.time, "monotonic", lambda: clock[0])
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"result": "late success", "usage": {}}),
            stderr="",
        )

        def return_after_deadline(*_args, **_kwargs):
            clock[0] = 106.0
            return completed

        with (
            evals_module._active_eval_case_budget(5),
            patch(
                "axiom_encode.harness.evals.subprocess.run",
                side_effect=return_after_deadline,
            ),
        ):
            response = _run_claude_prompt_eval(runner, workspace, "review this")

        assert response.text == ""
        assert response.error == "Eval case budget timed out"
        assert response.timed_out is True
        assert response.timeout_stage == "case_budget"
        assert response.timeout_reason == "wall"
        assert response.timeout_seconds == 5
        assert response.timeout_attempts == 1
        assert response.trace["timed_out"] is True
        assert response.trace["timeout_stage"] == "case_budget"
        assert response.trace["timeout_reason"] == "wall"
        assert response.trace["timeout_seconds"] == 5

    @pytest.mark.parametrize(
        "configured_value",
        [None, "not-a-number", "0", "-1"],
    )
    def test_prompt_eval_uses_fair_default_for_invalid_encoder_timeout(
        self,
        tmp_path,
        monkeypatch,
        configured_value,
    ):
        if configured_value is None:
            monkeypatch.delenv(
                "AXIOM_ENCODE_ENCODER_TIMEOUT_SECONDS",
                raising=False,
            )
        else:
            monkeypatch.setenv(
                "AXIOM_ENCODE_ENCODER_TIMEOUT_SECONDS",
                configured_value,
            )
        runner = parse_runner_spec("claude:opus")
        workspace = EvalWorkspace(
            root=tmp_path,
            source_text_file=tmp_path / "source.txt",
            manifest_file=tmp_path / "context-manifest.json",
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"result": "review complete", "usage": {}}),
            stderr="",
        )

        with patch(
            "axiom_encode.harness.evals.subprocess.run",
            return_value=completed,
        ) as mock_run:
            response = _run_claude_prompt_eval(runner, workspace, "review this")

        assert mock_run.call_args.kwargs["timeout"] == 1800
        assert response.trace["timeout_seconds"] == 1800

    def test_prompt_eval_disables_tools_and_scrubs_signing_capabilities(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setenv(EVAL_EVIDENCE_PRIVATE_KEY_ENV, "eval-private")
        monkeypatch.setenv(APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV, "apply-private")
        monkeypatch.setenv("AXIOM_ENCODE_SIGNING_BROKER_FD", "91")
        monkeypatch.setenv("AXIOM_ENCODE_SIGNING_BROKER_PID", "92")
        monkeypatch.setenv("AXIOM_ENCODE_SIGNING_BROKER_ACTIVE", "1")
        runner = parse_runner_spec("claude:opus")
        workspace = EvalWorkspace(
            root=tmp_path,
            source_text_file=tmp_path / "source.txt",
            manifest_file=tmp_path / "context-manifest.json",
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "result": "review complete",
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                }
            ),
            stderr="",
        )

        with patch(
            "axiom_encode.harness.evals.subprocess.run",
            return_value=completed,
        ) as mock_run:
            response = _run_claude_prompt_eval(runner, workspace, "review this")

        command = mock_run.call_args.args[0]
        assert command[command.index("--permission-mode") + 1] == "dontAsk"
        assert command[command.index("--tools") + 1] == ""
        assert command[command.index("--allowed-tools") + 1] == ""
        assert command[command.index("--mcp-config") + 1] == '{"mcpServers": {}}'
        for flag in (
            "--safe-mode",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--no-chrome",
            "--strict-mcp-config",
        ):
            assert flag in command
        assert "bypassPermissions" not in command
        child_environment = mock_run.call_args.kwargs["env"]
        for name in (
            EVAL_EVIDENCE_PRIVATE_KEY_ENV,
            APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV,
            "AXIOM_ENCODE_SIGNING_BROKER_FD",
            "AXIOM_ENCODE_SIGNING_BROKER_PID",
            "AXIOM_ENCODE_SIGNING_BROKER_ACTIVE",
        ):
            assert name not in child_environment
        assert response.text == "review complete"


class TestCodexPromptEval:
    def test_codex_prompt_eval_rejects_success_returned_after_case_deadline(
        self,
        tmp_path,
        monkeypatch,
    ):
        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = EvalWorkspace(
            root=tmp_path,
            source_text_file=tmp_path / "source.txt",
            manifest_file=tmp_path / "context-manifest.json",
        )
        bundle = "=== FILE: example.yaml ===\nformat: rulespec/v1\nrules: []\n"
        clock = [100.0]
        monkeypatch.setattr(evals_module.time, "monotonic", lambda: clock[0])

        class FakePopen:
            def __init__(self, cmd, stdout, stderr, text, cwd, stdin=None, env=None):
                self.args = cmd
                self.returncode = 0
                Path(cwd, ".codex-last-message.txt").write_text(bundle)

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        def return_after_deadline(*_args, **_kwargs):
            clock[0] = 106.0
            return False

        with (
            evals_module._active_eval_case_budget(5),
            patch("axiom_encode.harness.evals.subprocess.Popen", FakePopen),
            patch(
                "axiom_encode.harness.evals._wait_for_codex_process",
                side_effect=return_after_deadline,
            ),
        ):
            response = _run_codex_prompt_eval(runner, workspace, "prompt")

        assert response.text == ""
        assert response.error == "Eval case budget timed out"
        assert response.timed_out is True
        assert response.timeout_stage == "case_budget"
        assert response.timeout_reason == "wall"
        assert response.timeout_seconds == 5
        assert response.timeout_attempts == 1
        assert response.trace["timed_out"] is True
        assert response.trace["timeout_stage"] == "case_budget"
        assert response.trace["timeout_reason"] == "wall"
        assert response.trace["timeout_seconds"] == 5

    def test_wait_for_codex_process_terminates_after_stable_last_message(
        self, tmp_path
    ):
        last_message = tmp_path / ".codex-last-message.txt"
        last_message.write_text("ready\n")

        class FakeProcess:
            def __init__(self):
                self.args = ["codex", "exec"]
                self.returncode = None
                self.terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        process = FakeProcess()
        terminated = _wait_for_codex_process(
            process,
            last_message,
            timeout=1,
            settle_seconds=0,
            poll_interval=0,
        )

        assert terminated is True
        assert process.terminated is True


def test_build_eval_prompt_targets_rulespec_yaml(tmp_path):
    runner = parse_runner_spec("codex:gpt-5.4")
    workspace = prepare_eval_workspace(
        citation="tn_snap_standard_utility_allowance",
        runner=runner,
        output_root=tmp_path / "out",
        source_text="The standard utility allowance is $451.",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us-tn"),
        mode="cold",
    )

    prompt = _build_eval_prompt(
        "tn_snap_standard_utility_allowance",
        "cold",
        workspace,
        [],
        target_file_name="tn-snap-standard-utility-allowance.yaml",
        include_tests=True,
        policyengine_rule_hint="snap_standard_utility_allowance",
    )

    assert "format: rulespec/v1" in prompt
    assert "RuleSpec YAML" in prompt
    assert "=== FILE: tn-snap-standard-utility-allowance.yaml ===" in prompt
    assert "=== FILE: tn-snap-standard-utility-allowance.test.yaml ===" in prompt
    assert "apply that limit at the source-stated lower entity" in prompt
    assert "then cap the aggregate" in prompt
    assert "rate-applied result at the source-stated lower entity" in prompt
    assert "unit-level placeholder or aggregate base by the rate" in prompt
    assert '"per taxpayer per beneficiary"' in prompt
    assert "Do not apply one\n  per-unit cap to a single aggregate amount" in prompt
    assert "This is Axiom encoding work" in prompt
    assert "Do not read, load, or apply PolicyEngine skills" in prompt
    assert "For claim, overpayment, overissuance, repayment" in prompt
    assert "as collectability caps" in prompt
    assert "not return a bare placeholder such as `claim_amount`" in prompt
    assert "evaluating the emitted RuleSpec\n  formula" in prompt
    assert "flat\n  threshold with a percentage of excess income" in prompt
    assert "positive tests that expect a nonzero amount" in prompt
    assert "set every gate input on the\n  qualifying side of the threshold" in prompt
    assert "`age >= age_threshold`" in prompt
    assert "In mixed-output test cases" in prompt
    assert "nonqualifying side of that output's threshold gate" in prompt
    assert "separate all-gates-positive case" in prompt
    assert (
        "Imported definitions do not override the current source's legal subject"
        in prompt
    )
    assert "rate-applied result at that lower entity" in prompt
    assert "keep the result executable by exposing a\n  local boundary input" in prompt
    assert "such as `wages`" in prompt
    assert 'contextual qualifiers such as\n  "received by"' in prompt
    assert "do not fold them into the boundary input name" in prompt
    assert "Treat legal subject nouns as stronger evidence" in prompt
    assert "use `entity: Person` for the current source's own amount" in prompt
    assert (
        "Existing target or repository-precedent files are not entity-scope authority"
        in prompt
    )
    assert "treat the copied aggregate shape as a defect to repair" in prompt
    assert (
        "Do not preserve the\n  aggregate entity just to keep old output names or tests compatible"
        in prompt
    )
    assert "earned income of an individual shall be\n  computed" in prompt
    assert "replaced by one aggregated boundary input" in prompt
    assert "thresholded, capped, base-limited" in prompt
    assert (
        "do not flatten the cited mechanics into `current_base * imported_rate`"
        in prompt
    )
    assert "defer the\n  affected executable output" in prompt
    assert 'definition uses "taxpayer" but also says the amount is "of an' in prompt
    assert 'Do not let\n  the word "taxpayer"' in prompt
    assert "on the [base] of every individual/person/employee" in prompt
    assert "even if the imported base definition or its tests are unit-scoped" in prompt
    assert "Do not narrate your plan" in prompt
    assert "snap_standard_utility_allowance" in prompt
    assert "Do not use bare year periods like `2024`" in prompt
    assert "never use `period_kind: calendar_year`" in prompt
    assert "period_kind: tax_year" in prompt
    assert "period_kind: custom" in prompt
    assert "period: Day" in prompt
    assert "never use bare `YYYY-MM-DD` shorthand" in prompt
    assert "Do not preserve existing `#input.filing_status`" in prompt
    assert 'If the source says only "joint return"' in prompt
    assert 'status 4 falls under any "other case" branch' in prompt
    assert "Existing executable output names are public API contracts" not in prompt
    assert "only exact names listed by the Exact-oracle" in prompt
    assert "target path's year/legal-source identity" in prompt
    assert "applicable_amount_in_effect_under_section_<section>" not in prompt
    assert "Do not put the date or year value in the fact name" in prompt
    assert "Never use `post_YYYY`, `pre_YYYY`, `after_YYYY`, `before_YYYY`" in prompt
    assert "overrides preservation of existing local input names" in prompt
    assert "Never introduce an import cycle" in prompt
    assert (
        "Never create a derived rule whose formula references that same rule's name"
        in prompt
    )
    assert "no local derived rule formula references its own\n      rule name" in prompt
    assert "directly or transitively" in prompt
    assert "numeric boundary input" in prompt
    assert "do not import that consumer section" in prompt
    assert "purpose-specific outputs such as `x_for_section_1234_a`" in prompt
    assert "purpose-specific branch into one generic output" in prompt
    assert "same-named local input such as `x`" in prompt
    assert "export matching the requested source's clause" in prompt
    assert "trailing commas in calls" in prompt
    assert "do not assume one upstream raw input equals that imported output" in prompt
    assert "For IRC section 151 repairs" not in prompt
    assert "named numeric concept" in prompt
    assert "1 / benefit_cost_rate_compensation_lookback_years" in prompt
    assert "`1 / 5`" in prompt
    assert "if the source is a multi-state or\n  multi-jurisdiction table" in prompt
    assert "Do not invent a fake `State` entity" in prompt
    assert "do not create one scalar parameter per row, bound, or cell" in prompt
    assert "`*_lower_bound_band_9`" in prompt
    assert "`indexed_by: <band_selector>`" in prompt
    assert "integer band ids such as `0`, `1`, and `2`" in prompt
    assert "do not use decimal row thresholds like `1.33`, `2.5`" in prompt
    assert "or strings such as `2_5_to_less_than_3_0`" in prompt
    assert "Indexed parameter `values` keys must be integers" in prompt
    assert "such as county names" in prompt
    assert "do not emit one giant `or` chain" in prompt
    assert "at most 25 text comparisons" in prompt
    assert "For interval-table repair of an existing target" in prompt
    assert "do not add extra exported derived rules" in prompt
    assert "`clause_ii_provides_otherwise`" in prompt
    assert "Do not treat the final interval row as open-ended" in prompt
    assert "Include a companion test above the final bounded row" in prompt
    assert "The out-of-table sentinel is not itself a source table row" in prompt
    assert "do not clamp sentinel cases" in prompt
    assert "Use a negative sentinel such as `-1`" in prompt
    assert "do not use the next positive band id such as `6`" in prompt
    assert "Do not hard-code the final real band id" in prompt
    assert "let the indexed interpolation formula produce that constant" in prompt
    assert "source text `133%` should be represented as `1.33`" in prompt
    assert "old percent-point test inputs" in prompt
    assert "Structural interval bounds that are only used by the selector" in prompt
    assert "private bound concepts" in prompt
    assert "do not preserve, rename, or recreate the local" in prompt
    assert "for sibling clause\n  exception phrases" in prompt
    assert "Before finalizing, do this self-check:" in prompt
    assert "Numeric inventory: every source-stated legal amount" in prompt
    assert "exact imported concept\n     from context" in prompt
    assert "indexed numeric concepts" in prompt
    assert "import it instead of duplicating it locally" in prompt
    assert "An imported `dtype: Judgment` is a predicate, not a scalar" in prompt
    assert "Never multiply, add, subtract, divide, `min`, or `max`" in prompt
    assert "encode the source-stated numeric base as a local amount fact" in prompt
    assert "Test input inventory: for every local factual identifier" in prompt
    assert "For proration, average, ratio, or percentage tests" in prompt
    assert "use totals like 600" in prompt
    assert "Avoid exact equality boundaries for ratios or percentages" in prompt
    normalized_prompt = " ".join(prompt.split())
    assert (
        "If a module contains only parameters, emit one source-period snapshot "
        "case that asserts every local parameter output directly." in normalized_prompt
    )
    assert "cover parameters through derived outputs" in normalized_prompt
    assert "modifier parameter stranded" in prompt
    assert "module.deferred_outputs[]" in prompt
    assert "source_values" in prompt
    assert "in excess of" in prompt
    assert "max(0, measured_value - limit)" in prompt
    assert "Do not defer that excess output merely" in prompt
    assert "final effective legal amount" in prompt
    assert "explanatory percentage or index" in prompt
    assert "unused modifier parameter" in prompt
    assert "do not model that numeric term as a local" in prompt
    assert "tier_1_applicable_percentage" in prompt
    assert "output` target path must include that source path segment" in prompt
    assert "us:statutes/26/3201/a#tier_1_employee_tax" in prompt
    assert (
        "Only include `blocked_by` entries when you know the exact RuleSpec output"
        in prompt
    )
    assert "Do not list bare legal provisions" in prompt
    assert "us:statutes/us-ca/17000" in prompt
    assert "exclusions conditioned on a reasonable belief" in prompt
    assert "Do not defer solely because" in prompt
    assert "model the source-stated\n  reasonable-belief condition" in prompt
    assert "enumerates qualifying or exception categories" in prompt
    assert "cites other laws only to define those category labels" in prompt
    assert "only uses the citation to label a category" in prompt
    assert "appointment, office, retirement-system, election" in prompt
    assert "`described in section ...` category\n  labels" in prompt
    assert "section-described supporting organization" in prompt
    assert "covered-service, section-described supporting organization" in prompt
    assert "`within the\n  meaning of section ...` carve-outs" in prompt
    assert "category membership phrases" in prompt
    assert "`organization described in section X`" in prompt
    assert "organization_described_in_section_509_a_3" in prompt
    assert "testing\n  membership in the described category" in prompt
    assert (
        "unrelated-trade-or-business, or other\n  within-meaning/described-in definitions"
        in prompt
    )
    assert "Validation fails if a direct local `#input.*_exception_applies`" in prompt
    assert "imported test inputs from copied files" in prompt
    assert "Do not stub imported derived" in prompt
    assert "never assign prohibited derived" in prompt
    assert (
        "classifications such as any imported or local `#input.filing_status`" in prompt
    )
    assert "omit that assertion or encode the" in prompt
    assert "upstream filing-status" in prompt
    assert "sources first" in prompt
    assert "#relation.<name>` input value must be a YAML list of row mappings" in prompt
    assert "member_of_household: [- true]" in prompt
    assert "tables.<Entity>` rows" in prompt
    assert "Proof inventory: every proof atom uses only an allowed `kind`" in prompt
    assert (
        "Import inventory: every `imports:` entry is an exact copied/importable"
        in prompt
    )
    assert "Top-level `imports:` entries must be scalar strings" in prompt
    assert "map entries like `- target:`" in prompt
    assert (
        "Supported scalar functions are `min(...)`, `max(...)`, `floor(x)`, and `ceil(x)`"
        in prompt
    )
    assert "Do not use Python-only functions such as `round(...)`" in prompt
    assert (
        "Use `sum(relation.amount_fact)` only when `amount_fact` is a raw scalar fact"
        in prompt
    )
    assert "Do not use `sum(relation.local_output)`" in prompt
    assert "Do not write `amount + if condition: extra else: 0`" in prompt
    assert "Do not emit more than one `versions:` entry for `kind: derived`" in prompt
    assert (
        "A `kind: table_cell` proof atom must include `source.table.header`" in prompt
    )
    assert "header-only `parameter_table` proof atoms are invalid" in prompt
    assert "row_key" in prompt
    assert "column_key" in prompt
    assert "kind: derived_relation" in prompt
    assert "derived_relation:" in prompt
    assert "arity: 2" in prompt
    assert "source_relation: member_of_household" in prompt
    assert "formula: snap_member_eligible" in prompt
    assert "explicitly defines" in prompt
    assert "membership in a derived legal unit" in prompt
    assert '"This source is about SNAP" is not enough' in prompt
    assert "stay on the source-stated structural entity" in prompt
    assert "Any rule that uses `entity: <filtered-entity>`" in prompt
    assert "declare" in prompt
    assert "that entity with a `kind: derived_relation` rule" in prompt
    assert "import a RuleSpec file" in prompt
    assert "that declares it" in prompt
    assert "example_output\n    kind: derived\n    entity: Household" in prompt
    assert (
        "Adjacent bracket thresholds repeated as both an upper bound and the next"
        in prompt
    )


def test_build_eval_prompt_for_rate_only_source_id_limits_scope(tmp_path):
    runner = parse_runner_spec("codex:gpt-5.4")
    workspace = prepare_eval_workspace(
        citation="us/statute/26/1401/a/rate",
        runner=runner,
        output_root=tmp_path / "out",
        source_text=(
            "(a) Old-age, survivors, and disability insurance There shall be "
            "imposed for each taxable year, on the self-employment income of "
            "every individual, a tax equal to 12.4 percent of the amount of "
            "the self-employment income for such taxable year."
        ),
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
        mode="cold",
    )

    prompt = _build_eval_prompt(
        "us/statute/26/1401/a/rate",
        "cold",
        workspace,
        [],
        target_file_name="rate.yaml",
        target_ref_prefix="us:statutes/26/1401/a/rate",
        include_tests=True,
    )

    assert "Rate-only source boundary:" in prompt
    assert "source-stated rate or percentage" in prompt
    assert "parameters anchored in `./source.txt`" in prompt
    assert "Do not encode the downstream tax" in prompt
    assert "Prefer `kind: parameter`, `dtype: Rate`" in prompt
    assert "boundary must stay acyclic" in prompt
    assert "companion tests may assert" in prompt
    assert "canonical parameter output directly" in prompt
    assert "Source-boundary artifacts that contain only scalar parameters" in prompt
    assert "one source-period snapshot case" in prompt


def test_target_source_scope_ignores_cross_references_before_structural_marker():
    source = "\n\n".join(
        [
            "(a) Sampling plan. The plan references paragraph (b)(4), paragraph "
            "(b)(1)(iii), and paragraph (b)(2)(ii) before the actual sample-size "
            "paragraph.",
            "(b) Sample size. The State agency shall review active and negative cases.",
            "(1) Active cases. (i) All active cases shall be selected.",
            "(ii) Unless the alternate active case formula applies, the sample size is:",
            "Average monthly reviewable caseload (N) | Minimum annual sample size (n)\n"
            "60,000 and over | n = 2400\n"
            "10,000 to 59,999 | n = 300 + [0.042(N-10,000)]\n"
            "Under 10,000 | n = 300",
            "(iii) A State agency with the certification may instead use 0.0153.",
            "(2) Negative cases. (i) Unless the State agency uses paragraph "
            "(b)(2)(ii), the negative sample size is:",
            "Average monthly reviewable negative caseload (N) | Minimum annual sample size (n)\n"
            "5,000 and over | n = 800\n"
            "500 to 4,999 | n = 150 + [0.144(N-500)]\n"
            "Under 500 | n = 150",
            "(ii) A State agency with the certification may determine the negative "
            "sample size as follows:",
            "Average monthly reviewable negative caseload (N) | Minimum annual sample size (n)\n"
            "5,000 and over | n = 680\n"
            "684 to 4,999 | n = 150 + [0.1224(N-683)]\n"
            "Under 684 | n = 150",
            "(iii) In the formulas, n is the required negative sample size.",
            "(c) Review process.",
        ]
    )

    regular_negative = _target_source_scope_for_heuristics(
        source,
        "us:regulations/7-cfr/275/11/b/2/i",
    )
    assert regular_negative.lstrip().startswith("(i) Unless")
    assert "0.144" in regular_negative
    assert "0.1224" not in regular_negative

    alternate_negative = _target_source_scope_for_heuristics(
        source,
        "us:regulations/7-cfr/275/11/b/2/ii",
    )
    assert alternate_negative.lstrip().startswith("(ii) A State agency")
    assert "0.1224" in alternate_negative
    assert "0.144" not in alternate_negative

    regular_active = _target_source_scope_for_heuristics(
        source,
        "us:regulations/7-cfr/275/11/b/1/ii",
    )
    assert regular_active.lstrip().startswith("(ii) Unless")
    assert "0.042" in regular_active
    assert "0.0153" not in regular_active


def test_target_source_scope_distinguishes_alpha_marker_case_by_level():
    source = "\n\n".join(
        [
            "(d) Validation of State Agency error rates.",
            "(1) Payment error rate. (i) FNS will select a subsample.",
            "(A) First active subsample formula.",
            "(B) Second active subsample formula.",
            "(E) N is the State agency's minimum active case sample size.",
            "(2) Other payment-error review steps.",
            "(3) Negative case error rate. (i) FNS will select a subsample of "
            "completed negative cases as follows:",
            "Average monthly reviewable negative caseload (N) | Federal subsample target (n')\n"
            "12,000 and over | n' = 400\n"
            "1,001 to 11,999 | n' = .011634 N + 40\n"
            "1,000 and under | n' = 150",
            "(ii) The negative case record review follows.",
            "(e) State corrective action.",
        ]
    )

    negative_review = _target_source_scope_for_heuristics(
        source,
        "us:regulations/7-cfr/275/3/d/3/i",
    )

    assert negative_review.lstrip().startswith("(i) FNS will select")
    assert "Federal subsample target" in negative_review
    assert "Second active subsample formula" not in negative_review
    assert "(e) State corrective action" not in negative_review


def test_target_source_scope_treats_uppercase_alpha_as_non_roman():
    source = "\n\n".join(
        [
            "(b) State agency error rates.",
            "(2) Determination of payment error rates.",
            "(i) FNS shall calculate regressed error rates.",
            "(A) y1' = y1 + b1 (X1 - x1).",
            "(B) y2' = y2 + b2 (X2 - x2).",
            "(C) The regressed error rates are r1' = y1'/u and r2' = y2'/u.",
            "(D) The adjusted regressed payment error rate is r1'' + r2''.",
            "(ii) Other review steps.",
        ]
    )

    regressed_rates = _target_source_scope_for_heuristics(
        source,
        "us:regulations/7-cfr/275/23/b/2/i/C",
    )

    assert regressed_rates.lstrip().startswith("(C) The regressed error rates")
    assert "r1' = y1'/u" in regressed_rates
    assert "adjusted regressed payment error rate" not in regressed_rates


def test_target_source_scope_prefers_line_start_nested_markers():
    source = "\n\n".join(
        [
            "(f) Good cause.",
            "(1) Natural disasters. (i) The State agency shall document impacts.",
            "(ii) (A) The following criteria apply:",
            "(1) Geographic impact;",
            "(2) Duration;",
            "(3) The proportion of caseload affected; and/or",
            "(4) Operational impact.",
            "(2) Strikes.",
            "(3) Caseload growth.",
            "(i) A State agency may request relief for unusual caseload growth.",
            "(ii) Criteria apply.",
            "(iii) If information is insufficient, use this five-step calculation:",
            "(A) Step 1--determine the base-period average.",
            "(B) Step 2--determine the percentage increase.",
            "(C) Step 3--determine the percentage the error rate exceeds the national performance measure.",
            "(D) Step 4--divide the percentage increase by the percentage excess.",
            "(E) Step 5--multiply the quotient by the liability amount.",
            "(iv) Caseload growth of less than 15 percent is not considered.",
            "(4) Program changes.",
            "(g) Results of appeals.",
        ]
    )

    step_three = _target_source_scope_for_heuristics(
        source,
        "us:regulations/7-cfr/275/23/f/3/iii/C",
    )

    assert step_three.lstrip().startswith("(C) Step 3")
    assert "percentage the error rate exceeds" in step_three
    assert "Step 4" not in step_three
    assert "The proportion of caseload affected" not in step_three


def test_build_eval_prompt_does_not_treat_rates_path_as_rate_only(tmp_path):
    runner = parse_runner_spec("codex:gpt-5.4")
    workspace = prepare_eval_workspace(
        citation="us/statute/26/1401/rates",
        runner=runner,
        output_root=tmp_path / "out",
        source_text="The table states several percentage rates.",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
        mode="cold",
    )

    prompt = _build_eval_prompt(
        "us/statute/26/1401/rates",
        "cold",
        workspace,
        [],
        target_file_name="rates.yaml",
        target_ref_prefix="us:statutes/26/1401/rates",
        include_tests=True,
    )

    assert "Rate-only source boundary:" not in prompt


def test_build_eval_prompt_does_not_treat_monetary_rate_as_rate_only(tmp_path):
    runner = parse_runner_spec("codex:gpt-5.4")
    workspace = prepare_eval_workspace(
        citation="us/statute/26/9999/rate",
        runner=runner,
        output_root=tmp_path / "out",
        source_text="The reimbursement rate is 67 cents per mile.",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
        mode="cold",
    )

    prompt = _build_eval_prompt(
        "us/statute/26/9999/rate",
        "cold",
        workspace,
        [],
        target_file_name="rate.yaml",
        target_ref_prefix="us:statutes/26/9999/rate",
        include_tests=True,
    )

    assert "Rate-only source boundary:" not in prompt


def test_context_file_surfaces_include_derived_relation(tmp_path):
    context_file = tmp_path / "snap_unit.yaml"
    context_file.write_text(
        """format: rulespec/v1
rules:
  - name: snap_unit
    kind: derived_relation
    derived_relation:
      arity: 2
      source_relation: member_of_household
      entity: SnapUnit
      member_relation: members
      slot_entities: [Person, Household]
    versions:
      - effective_from: '2026-01-01'
        formula: snap_member_eligible
"""
    )

    surfaces = _context_file_executable_surfaces(str(context_file))

    assert surfaces["snap_unit"]["kind"] == "derived_relation"
    assert surfaces["snap_unit"]["entity"] == "SnapUnit"


def test_existing_target_prompt_marks_exact_oracle_surface_as_required(tmp_path):
    target = tmp_path / "2026_section_40_18_5_schedule_before_credits.yaml"
    target.write_text(
        """format: rulespec/v1
rules:
  - name: al_pit_2026_section_40_18_5_schedule_before_credits
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        effective_to: '2026-12-31'
        formula: max(0, al_pit_completed_taxable_income)
inputs:
  - name: al_pit_completed_taxable_income
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
"""
    )
    context = EvalContextFile(
        source_path=str(target),
        workspace_path="context/existing_target.yaml",
        import_path=(
            "us-al:policies/income_tax/2026_section_40_18_5_schedule_before_credits"
        ),
        kind="existing_target",
    )

    guidance = _format_existing_target_contract_guidance([context])

    assert "Exact-oracle replacement contract:" in guidance
    assert "#al_pit_2026_section_40_18_5_schedule_before_credits`" in guidance
    assert "#input.al_pit_completed_taxable_income`" in guidance
    assert "visibility=public" in guidance
    assert "`2026_section_40_18_5`" in guidance
    assert "exact mapped legacy surface names listed above are the only" in guidance
    assert "invalid" in guidance
    assert "legacy input" in guidance


def test_build_eval_prompt_lists_existing_target_surfaces(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    target = policy_repo / "statutes/26/999.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(
        """format: rulespec/v1
rules:
  - name: existing_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          existing_fact
          and filing_status == 1
          and taxable_year_begins_after_2024
          and applicable_amount_in_effect_under_section_68_b > 0
  - name: existing_table
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 100
"""
    )
    workspace = prepare_eval_workspace(
        citation="26 USC 999",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="The amount is allowed.",
        axiom_rules_path=policy_repo,
        mode="repo-augmented",
        extra_context_paths=[],
    )
    cyclic_context = policy_repo / "statutes/26/7703.yaml"
    cyclic_context.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/999#existing_amount
rules:
  - name: upstream_married_rule
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: existing_amount > 0
"""
    )
    section_68_context = policy_repo / "statutes/26/68/b.yaml"
    section_68_context.parent.mkdir(parents=True)
    section_68_context.write_text(
        """format: rulespec/v1
rules:
  - name: section_68_applied_after_other_itemized_deduction_limitations
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: true
"""
    )
    workspace.context_files.append(
        EvalContextFile(
            source_path=str(cyclic_context),
            workspace_path="context/statutes/26/7703.yaml",
            import_path="us:statutes/26/7703",
            kind="citation_context",
        )
    )
    workspace.context_files.append(
        EvalContextFile(
            source_path=str(section_68_context),
            workspace_path="context/statutes/26/68/b.yaml",
            import_path="us:statutes/26/68/b",
            kind="citation_context",
        )
    )

    prompt = _build_eval_prompt(
        "26 USC 999",
        "repo-augmented",
        workspace,
        workspace.context_files,
        target_file_name="999.yaml",
        target_ref_prefix="us:statutes/26/999",
        include_tests=True,
        runner_backend="openai",
    )

    assert "Existing target executable surfaces:" in prompt
    assert "not compatibility contracts" in prompt
    assert "`us:statutes/26/999#existing_amount`" in prompt
    assert "entity=TaxUnit" in prompt
    assert "effective_from=2026-01-01" in prompt
    assert "`us:statutes/26/999#existing_table`" in prompt
    assert "indexed_by=household_size" in prompt
    assert "local input slots" in prompt
    assert "`us:statutes/26/999#input.existing_fact`" in prompt
    assert "Never copy a `#input` key from" in prompt
    assert "sibling context test" in prompt
    assert "Invalid copied local input names:" in prompt
    assert "`us:statutes/26/999#input.filing_status`" in prompt
    assert "filing status is a derived legal classification" in prompt
    assert "`us:statutes/26/999#input.taxable_year_begins_after_2024`" in prompt
    assert "date/year-valued temporal fact" in prompt
    assert "`post_YYYY`, `pre_YYYY`, or any four-digit year" in prompt
    assert (
        "`us:statutes/26/999#input.applicable_amount_in_effect_under_section_68_b`"
        in prompt
    )
    assert "encoded cross-reference placeholder" in prompt
    assert "us:statutes/26/68/b" in prompt
    assert "Existing target local factual inputs:" in prompt
    assert "`us:statutes/26/999#input.existing_fact`" in prompt
    assert "Cycle-prone context imports:" in prompt
    assert "`us:statutes/26/7703` already imports `us:statutes/26/999`" in prompt
    assert "more specific child file under the current target path" in prompt
    assert "such as a `/rate`" in prompt
    assert "do not emit a duplicate local `parameter`" in prompt


def test_build_eval_prompt_lists_existing_target_validation_failures(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    target = policy_repo / "statutes/26/63/f.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(
        """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: unmarried_not_surviving_spouse_additional_amount
    kind: parameter
    dtype: Money
    source: IRC section 63(f)(3)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              excerpt: "applied by substituting $750 for $600"
    versions:
      - effective_from: '2026-01-01'
        formula: 750
"""
    )
    workspace = prepare_eval_workspace(
        citation="26 USC 63(f)",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="The additional amount is $600. Substitute $750 for $600.",
        axiom_rules_path=policy_repo,
        mode="repo-augmented",
        extra_context_paths=[],
    )

    prompt = _build_eval_prompt(
        "26 USC 63(f)",
        "repo-augmented",
        workspace,
        workspace.context_files,
        target_file_name="f.yaml",
        target_ref_prefix="us:statutes/26/63/f",
        include_tests=True,
        runner_backend="openai",
    )

    assert "Copied existing target fails current RuleSpec validation:" in prompt
    assert "`us:statutes/26/63/f`" in prompt
    assert "unmarried_not_surviving_spouse_additional_amount" in prompt
    assert "`module.deferred_outputs[].source_values`" in prompt
    assert "preserve the failing shape" in prompt


def test_materialize_eval_artifact_writes_rulespec_bundle(tmp_path):
    output_file = tmp_path / "runner" / "source" / "tn-snap.yaml"
    materialized_paths: set[Path] = set()
    llm_response = """=== FILE: tn-snap.yaml ===
format: rulespec/v1
module:
  summary: |-
    The standard utility allowance is $451.
rules:
  - name: snap_standard_utility_allowance_value
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          451
=== FILE: tn-snap.test.yaml ===
- name: base
  period: 2024-01
  input: {}
  output:
    snap_standard_utility_allowance: 451
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="The standard utility allowance is $451.",
        materialized_paths=materialized_paths,
    )

    assert wrote is True
    assert output_file.exists()
    assert output_file.with_name("tn-snap.test.yaml").exists()
    assert output_file.read_text().startswith("format: rulespec/v1")
    assert materialized_paths == {
        output_file,
        output_file.with_name("tn-snap.test.yaml"),
    }


def test_materialize_eval_artifact_rejects_test_only_bundle_with_stale_main(
    tmp_path,
):
    output_file = tmp_path / "runner" / "statutes" / "47" / "294.yaml"
    output_file.parent.mkdir(parents=True)
    output_file.write_text("stale prior-run main\n", encoding="utf-8")
    materialized_paths: set[Path] = set()

    wrote = _materialize_eval_artifact(
        "=== FILE: 294.test.yaml ===\n- name: current test only\n",
        output_file,
        materialized_paths=materialized_paths,
    )

    assert wrote is False
    assert output_file.read_text(encoding="utf-8") == "stale prior-run main\n"
    assert materialized_paths == {output_file.with_name("294.test.yaml")}


def test_clear_eval_target_artifacts_prevents_reused_companion_test(
    tmp_path,
):
    output_root = tmp_path / "output"
    output_file = output_root / "runner" / "statutes" / "47" / "294.yaml"
    companion_test = output_file.with_name("294.test.yaml")
    output_file.parent.mkdir(parents=True)
    output_file.write_text("stale prior-run main\n", encoding="utf-8")
    companion_test.write_text("- name: stale prior-run test\n", encoding="utf-8")

    _clear_eval_target_artifacts(output_file, output_root)
    wrote = _materialize_eval_artifact(
        "format: rulespec/v1\nrules: []\n",
        output_file,
        artifact_root=output_root,
    )

    assert wrote is True
    assert output_file.read_text(encoding="utf-8") == (
        "format: rulespec/v1\nrules: []\n"
    )
    assert not companion_test.exists()


def test_clear_eval_target_artifacts_rejects_symlinked_ancestor(tmp_path):
    output_root = tmp_path / "output"
    outside = tmp_path / "outside"
    output_root.mkdir()
    outside.mkdir()
    (output_root / "runner").symlink_to(outside, target_is_directory=True)
    outside_target = outside / "statutes" / "47" / "294.yaml"
    outside_target.parent.mkdir(parents=True)
    outside_target.write_text("outside sentinel\n", encoding="utf-8")

    with pytest.raises(OSError):
        _clear_eval_target_artifacts(
            output_root / "runner" / "statutes" / "47" / "294.yaml",
            output_root,
        )

    assert outside_target.read_text(encoding="utf-8") == "outside sentinel\n"


def test_materialize_eval_artifact_replaces_target_symlink_without_following_it(
    tmp_path,
):
    output_file = tmp_path / "runner" / "statutes" / "47" / "294.yaml"
    outside = tmp_path / "outside.yaml"
    output_file.parent.mkdir(parents=True)
    outside.write_text("outside sentinel\n", encoding="utf-8")
    output_file.symlink_to(outside)

    wrote = _materialize_eval_artifact(
        "format: rulespec/v1\nrules: []\n",
        output_file,
    )

    assert wrote is True
    assert outside.read_text(encoding="utf-8") == "outside sentinel\n"
    assert not output_file.is_symlink()
    assert output_file.read_text(encoding="utf-8") == (
        "format: rulespec/v1\nrules: []\n"
    )


def test_materialize_eval_artifact_rejects_symlinked_output_ancestor(tmp_path):
    output_root = tmp_path / "output"
    outside = tmp_path / "outside"
    output_root.mkdir()
    outside.mkdir()
    (output_root / "runner").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        _materialize_eval_artifact(
            "format: rulespec/v1\nrules: []\n",
            output_root / "runner" / "statutes" / "47" / "294.yaml",
            artifact_root=output_root,
        )

    assert list(outside.iterdir()) == []


def test_materialize_eval_artifact_repairs_copied_cross_reference_summary(tmp_path):
    output_file = tmp_path / "runner" / "statutes" / "39" / "39-22-104" / "1.5.yaml"
    llm_response = """=== FILE: 1.5.yaml ===
format: rulespec/v1
module:
  summary: |-
    (1.5) Subject to subsection (2) of this section, a tax of four and three-quarters percent is imposed.

    (2) Prior to the application of the rate of tax prescribed in subsection (1), (1.5), or (1.7) of this section, federal taxable income shall be modified.
rules:
  - name: individual_estate_trust_income_tax_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '1999-01-01'
        formula: '0.0475'
"""

    wrote = _materialize_eval_artifact(llm_response, output_file)

    assert wrote is True
    payload = yaml.safe_load(output_file.read_text())
    summary = payload["module"]["summary"]
    assert "four and three-quarters percent" in summary
    assert "Prior to the application" not in summary


def test_rulespec_validation_overlay_does_not_copy_ambient_source_metadata(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us-co")
    output_file = (
        tmp_path
        / "out"
        / "openai-gpt-5.5"
        / "statutes"
        / "39"
        / "39-22-104"
        / "1.5.yaml"
    )
    output_file.parent.mkdir(parents=True)
    output_file.write_text("format: rulespec/v1\nrules: []\n")
    workspace = (
        tmp_path
        / "out"
        / "_eval_workspaces"
        / "openai-gpt-5.5"
        / "us-co-statute-39-39-22-104-1.5"
        / "workspace"
    )
    workspace.mkdir(parents=True)
    (workspace / "context-manifest.json").write_text(
        json.dumps(
            {
                "citation": "us-co/statute/39/39-22-104/1.5",
                "source_metadata": {
                    "source_attestation": {
                        "requested_corpus_citation_path": (
                            "us-co/statute/39/39-22-104/1.5"
                        ),
                        "resolved_corpus_citation_path": ("us-co/statute/39/39-22-104"),
                    },
                },
            }
        )
    )

    with _rulespec_validation_target(output_file, policy_repo) as validation_file:
        validation_root = find_policy_repo_root(validation_file)
        assert validation_root is not None
        assert not (validation_root.parent.parent / "_eval_workspaces").exists()


def _unicode_path_replacement_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, LegacyReplacementContract]:
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    checkout = policy_repo.parent
    source = Path("us/statutes/42/1437c–1.yaml")
    destination = Path("us/statutes/42/1437c-1.yaml")
    source_file = checkout / source
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_raw = b"format: rulespec/v1\nrules: []\n"
    source_file.write_bytes(source_raw)
    companion = source.with_name(f"{source.stem}.test.yaml")
    companion_file = checkout / companion
    companion_raw = b"cases: []\n"
    companion_file.write_bytes(companion_raw)
    manifest = Path(".axiom/encoding-manifests") / source.with_suffix(".json")
    manifest_file = checkout / manifest
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_raw = b'{"schema_version":"axiom-encode/applied-rulespec/v1"}\n'
    manifest_file.write_bytes(manifest_raw)
    orphan_manifest = checkout / (
        ".axiom/encoding-manifests/us-tx/statutes/11/orphan.json"
    )
    orphan_manifest.parent.mkdir(parents=True, exist_ok=True)
    orphan_manifest.write_text("{}\n")
    unrelated_legacy = checkout / "us-nj/statutes/54a:4-7.yaml"
    unrelated_legacy.parent.mkdir(parents=True, exist_ok=True)
    unrelated_legacy.write_text("format: rulespec/v1\nrules: []\n")

    def bound_file(path: Path, raw: bytes) -> LegacyReplacementFile:
        return LegacyReplacementFile(path, hashlib.sha256(raw).hexdigest(), raw)

    contract = LegacyReplacementContract(
        base_commit="1" * 40,
        base_tree="2" * 40,
        source=source,
        destination=destination,
        legacy_manifest=bound_file(manifest, manifest_raw),
        deleted_files=(
            bound_file(source, source_raw),
            bound_file(companion, companion_raw),
        ),
        rewrites=(),
        scheduled_dependents=(),
        exact_dependents=(),
    )
    generated = tmp_path / "out" / "openai" / "statutes" / "42" / destination.name
    generated.parent.mkdir(parents=True)
    generated.write_text("format: rulespec/v1\nrules: []\n")
    return policy_repo, generated, contract


def test_rulespec_prevalidation_stages_authenticated_unicode_path_replacement(
    tmp_path,
):
    policy_repo, generated, contract = _unicode_path_replacement_fixture(tmp_path)

    with _rulespec_validation_target(
        generated,
        policy_repo,
        legacy_replacement=contract,
    ) as validation_file:
        checkout = validation_file.parents[3]
        assert validation_file.read_text() == "format: rulespec/v1\nrules: []\n"
        assert not (checkout / contract.source).exists()
        assert not (checkout / contract.legacy_manifest.path).exists()
        assert not (checkout / "us-nj").exists()
        assert not (checkout / ".axiom/encoding-manifests/us-tx").exists()
        assert all(
            part.isascii() for path in checkout.rglob("*") for part in path.parts
        )


def test_legacy_replacement_overlay_keeps_active_ancestor_chain(tmp_path):
    state_root = _canonical_rulespec_content_root(tmp_path, "us-or")
    checkout = state_root.parent
    (checkout / "us").mkdir()
    (checkout / "us-nj").mkdir()
    manifest_root = checkout / ".axiom/encoding-manifests"
    for jurisdiction in ("us", "us-or", "us-nj", "us-tx"):
        manifest_dir = manifest_root / jurisdiction
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "marker.json").write_text("{}\n")

    copied = tmp_path / "copy" / "rulespec-us"
    evals_module._copy_validation_overlay_tree(
        checkout,
        copied,
        ignore=evals_module._legacy_replacement_overlay_ignore(
            checkout,
            active_jurisdiction="us-or",
        ),
    )

    assert (copied / "us").is_dir()
    assert (copied / state_root.name).is_dir()
    assert not (copied / "us-nj").exists()
    assert (copied / ".axiom/encoding-manifests/us/marker.json").is_file()
    assert (copied / ".axiom/encoding-manifests/us-or/marker.json").is_file()
    assert not (copied / ".axiom/encoding-manifests/us-nj").exists()
    assert not (copied / ".axiom/encoding-manifests/us-tx").exists()


def test_canonical_replacement_validation_excludes_unrelated_jurisdictions(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    checkout = policy_repo.parent
    sibling = checkout / "us-nj/statutes/54a:4-7.yaml"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("format: rulespec/v1\nrules: []\n")
    sibling_manifest = (
        checkout / ".axiom/encoding-manifests/us-nj/statutes/54a:4-7.json"
    )
    sibling_manifest.parent.mkdir(parents=True)
    sibling_manifest.write_text("{}\n")
    generated = tmp_path / "out/openai/statutes/42/1437c-1.yaml"
    generated.parent.mkdir(parents=True)
    generated.write_text("format: rulespec/v1\nrules: []\n")

    with _rulespec_validation_target(
        generated,
        policy_repo,
        replacement_overlay_scope=True,
    ) as validation_file:
        overlay_checkout = validation_file.parents[3]
        assert validation_file.is_file()
        assert not (overlay_checkout / "us-nj").exists()
        assert not (overlay_checkout / ".axiom/encoding-manifests/us-nj").exists()


def test_canonical_replacement_validation_omits_active_colon_paths_only(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us-la")
    colon_path = policy_repo / "statutes/47:32.yaml"
    canonical_debt = policy_repo / "statutes/47/BROKEN.yaml"
    target = policy_repo / "statutes/47/294.yaml"
    for path in (colon_path, canonical_debt, target):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("format: rulespec/v1\nrules: []\n")
    generated = tmp_path / "out/openai/statutes/47/294.yaml"
    generated.parent.mkdir(parents=True)
    generated.write_text("format: rulespec/v1\nrules: []\n")

    with _rulespec_validation_target(
        generated,
        policy_repo,
        replacement_overlay_scope=True,
    ) as validation_file:
        overlay_policy = validation_file.parents[3] / "us-la"
        assert not (overlay_policy / "statutes/47:32.yaml").exists()
        assert (overlay_policy / "statutes/47/BROKEN.yaml").is_file()
        assert validation_file.is_file()


def test_fresh_encode_validation_preserves_unrelated_jurisdictions(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    checkout = policy_repo.parent
    sibling = checkout / "us-nj/statutes/54a:4-7.yaml"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("format: rulespec/v1\nrules: []\n")
    generated = tmp_path / "out/openai/statutes/42/new.yaml"
    generated.parent.mkdir(parents=True)
    generated.write_text("format: rulespec/v1\nrules: []\n")

    with _rulespec_validation_target(generated, policy_repo) as validation_file:
        overlay_checkout = validation_file.parents[3]
        assert (overlay_checkout / "us-nj/statutes/54a:4-7.yaml").is_file()


def test_legacy_replacement_overlay_rejects_manifest_jurisdiction_symlink(tmp_path):
    state_root = _canonical_rulespec_content_root(tmp_path, "us-or")
    manifest_root = state_root.parent / ".axiom/encoding-manifests"
    manifest_root.mkdir(parents=True)
    outside = tmp_path / "outside-manifests"
    outside.mkdir()
    (manifest_root / "us-tx").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        UnsafeRulespecContextPath,
        match="manifest jurisdiction is unsafe",
    ):
        evals_module._legacy_replacement_overlay_ignore(
            state_root.parent,
            active_jurisdiction="us-or",
        )


def test_rulespec_prevalidation_rejects_excluded_sibling_rewrite(tmp_path):
    policy_repo, generated, contract = _unicode_path_replacement_fixture(tmp_path)
    sibling = Path("us-nj/statutes/54a:4-7.yaml")
    sibling_raw = (policy_repo.parent / sibling).read_bytes()
    replacement_raw = b"format: rulespec/v1\nrules:\n  - name: changed\n"
    contract = contract._replace(
        rewrites=(
            LegacyReplacementRewrite(
                path=sibling,
                before_sha256=hashlib.sha256(sibling_raw).hexdigest(),
                after_sha256=hashlib.sha256(replacement_raw).hexdigest(),
                replacements=(),
                raw=replacement_raw,
            ),
        ),
    )

    with pytest.raises(ValueError, match="rewrite input is not a regular file"):
        with _rulespec_validation_target(
            generated,
            policy_repo,
            legacy_replacement=contract,
        ):
            pass

    assert (policy_repo.parent / sibling).read_bytes() == sibling_raw


def test_rulespec_prevalidation_rejects_changed_legacy_replacement_input(tmp_path):
    policy_repo, generated, contract = _unicode_path_replacement_fixture(tmp_path)
    (policy_repo.parent / contract.source).write_text("changed\n")

    with pytest.raises(ValueError, match="overlay input changed"):
        with _rulespec_validation_target(
            generated,
            policy_repo,
            legacy_replacement=contract,
        ):
            pass


@pytest.mark.parametrize("collision", ["primary", "companion", "manifest"])
def test_rulespec_prevalidation_rejects_replacement_destination_collision(
    tmp_path,
    collision,
):
    policy_repo, generated, contract = _unicode_path_replacement_fixture(tmp_path)
    checkout = policy_repo.parent
    targets = {
        "primary": checkout / contract.destination,
        "companion": checkout
        / contract.destination.with_name(f"{contract.destination.stem}.test.yaml"),
        "manifest": checkout
        / (Path(".axiom/encoding-manifests") / contract.destination).with_suffix(
            ".json"
        ),
    }
    targets[collision].parent.mkdir(parents=True, exist_ok=True)
    targets[collision].write_text("collision\n")

    with pytest.raises(ValueError, match="destination.*collision"):
        with _rulespec_validation_target(
            generated,
            policy_repo,
            legacy_replacement=contract,
        ):
            pass


def test_rulespec_prevalidation_rejects_live_checkout_artifact(tmp_path):
    policy_repo, _generated, contract = _unicode_path_replacement_fixture(tmp_path)

    with pytest.raises(ValueError, match="separately generated artifact"):
        with _rulespec_validation_target(
            policy_repo / Path(*contract.source.parts[1:]),
            policy_repo,
            legacy_replacement=contract,
        ):
            pass


def test_evaluate_artifact_uses_post_replacement_tree_for_first_compile(tmp_path):
    policy_repo, generated, contract = _unicode_path_replacement_fixture(tmp_path)
    checkout = policy_repo.parent
    retained_source = Path("us/statutes/47:297/4.yaml")
    retained_source_manifest = Path(
        ".axiom/encoding-manifests/us/statutes/47:297/4.json"
    )
    retained_destination = Path("us/statutes/47/297/4.yaml")
    retained_destination_manifest = Path(
        ".axiom/encoding-manifests/us/statutes/47/297/4.json"
    )
    retained_raw = {
        retained_source: b"format: rulespec/v1\nrules: []\n",
        retained_source_manifest: b"legacy manifest\n",
        retained_destination: b"format: rulespec/v1\nrules: []\n",
        retained_destination_manifest: b"signed successor manifest\n",
    }
    for path, raw in retained_raw.items():
        target = checkout / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)

    def retained_file(path: Path) -> LegacyReplacementFile:
        raw = retained_raw[path]
        return LegacyReplacementFile(path, hashlib.sha256(raw).hexdigest(), raw)

    contract = contract._replace(
        retained_successors=(
            LegacyReplacementRetainedSuccessor(
                source=retained_source,
                destination=retained_destination,
                legacy_manifest=retained_file(retained_source_manifest),
                legacy_files=(retained_file(retained_source),),
                successor_manifest=retained_file(retained_destination_manifest),
                successor_files=(retained_file(retained_destination),),
            ),
        )
    )
    observed = []

    def reject_noncanonical_checkout(_pipeline, validation_file):
        checkout = validation_file.parents[3]
        unsafe = [
            path
            for path in checkout.rglob("*")
            if any(
                not part.isascii() or ":" in part
                for part in path.relative_to(checkout).parts
            )
        ]
        observed.append((validation_file, unsafe))
        return ValidationResult("hard-cut", passed=not unsafe)

    with (
        patch.object(
            ValidatorPipeline, "_run_compile_check", reject_noncanonical_checkout
        ),
        patch.object(ValidatorPipeline, "_run_ci", reject_noncanonical_checkout),
    ):
        metrics = evaluate_artifact(
            local_corpus_release=_write_test_corpus_provision(tmp_path / "release"),
            rulespec_file=generated,
            policy_repo_root=policy_repo,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            source_text="No numeric values.",
            skip_reviewers=True,
            legacy_replacement=contract,
        )

    assert metrics.compile_pass is True
    assert metrics.ci_pass is True
    assert len(observed) == 2
    assert all(not unsafe for _path, unsafe in observed)


@pytest.mark.parametrize("indirection", ["file", "directory"])
def test_rulespec_validation_overlay_rejects_repo_symlinks(tmp_path, indirection):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel.yaml").write_text("secret: do-not-copy\n")
    if indirection == "file":
        (policy_repo.parent / "unrelated.yaml").symlink_to(outside / "sentinel.yaml")
    else:
        (policy_repo.parent / "unrelated").symlink_to(outside, target_is_directory=True)

    generated = tmp_path / "out" / "openai" / "statutes" / "1" / "new.yaml"
    generated.parent.mkdir(parents=True)
    generated.write_text("format: rulespec/v1\nrules: []\n")

    with pytest.raises(UnsafeRulespecContextPath, match="overlay source.*symlink"):
        with _rulespec_validation_target(generated, policy_repo):
            pass


def test_rulespec_validation_overlay_copies_safe_cross_repo_context(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    dependency_content_root = _canonical_rulespec_content_root(tmp_path, "uk")
    dependency_root = dependency_content_root.parent
    sibling = dependency_content_root / "statutes" / "1" / "child.yaml"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("format: rulespec/v1\nrules: []\n")
    generated = tmp_path / "out" / "openai" / "statutes" / "1" / "new.yaml"
    generated.parent.mkdir(parents=True)
    generated.write_text(
        "format: rulespec/v1\nimports:\n  - uk:statutes/1/child\nrules: []\n"
    )

    with _rulespec_validation_target(
        generated,
        policy_repo,
        rulespec_dependency_roots=(dependency_root,),
    ) as validation_file:
        validation_policy_root = _validation_policy_repo_root(
            validation_file,
            policy_repo,
        )
        validation_dependency_roots = _validation_rulespec_dependency_roots(
            validation_file=validation_file,
            policy_repo_root=policy_repo,
            rulespec_dependency_roots=(dependency_root,),
        )
        target_ref = validator_pipeline._parse_rulespec_target("uk:statutes/1/child")
        assert target_ref is not None
        resolved = validator_pipeline._resolve_rulespec_target_file(
            target_ref,
            validation_policy_root,
            rulespec_dependency_roots=validation_dependency_roots,
        )

        assert resolved is not None
        assert resolved.read_text() == "format: rulespec/v1\nrules: []\n"


def test_rulespec_validation_overlay_ignores_ambient_sibling_checkout(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    ambient_content_root = _canonical_rulespec_content_root(tmp_path, "uk")
    sibling = ambient_content_root / "statutes" / "1" / "child.yaml"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("format: rulespec/v1\nrules: []\n")
    generated = tmp_path / "out" / "openai" / "statutes" / "1" / "new.yaml"
    generated.parent.mkdir(parents=True)
    generated.write_text("format: rulespec/v1\nrules: []\n")

    with _rulespec_validation_target(generated, policy_repo) as validation_file:
        validation_policy_root = _validation_policy_repo_root(
            validation_file,
            policy_repo,
        )
        target_ref = validator_pipeline._parse_rulespec_target("uk:statutes/1/child")
        assert target_ref is not None
        assert (
            validator_pipeline._resolve_rulespec_target_file(
                target_ref,
                validation_policy_root,
            )
            is None
        )


def test_rulespec_validation_overlay_accepts_system_temp_directory_alias(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")

    with tempfile.TemporaryDirectory() as tmpdir:
        generated = Path(tmpdir) / "openai-gpt-5.5" / "statutes" / "1" / "new.yaml"
        generated.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")

        with _rulespec_validation_target(generated, policy_repo) as validation_file:
            assert validation_file.read_text() == "format: rulespec/v1\nrules: []\n"


def test_rulespec_validation_overlay_rejects_aliased_checkout(tmp_path):
    policy_repo = tmp_path / "rulespec-us-medicaid-program-composite-20260626"
    policy_repo.mkdir()
    subprocess.run(["git", "init"], cwd=policy_repo, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "remote",
            "add",
            "origin",
            "https://github.com/TheAxiomFoundation/rulespec-us.git",
        ],
        cwd=policy_repo,
        check=True,
        capture_output=True,
    )
    (policy_repo / "us").mkdir()

    output_file = (
        tmp_path
        / "out"
        / "codex-gpt-5.5"
        / "regulations"
        / "42-cfr"
        / "435"
        / "120"
        / "ssi-mandatory-group.yaml"
    )
    output_file.parent.mkdir(parents=True)
    output_file.write_text(
        """format: rulespec/v1
rules:
  - name: ssi_mandatory_group
    kind: derived
    entity: Person
    dtype: Boolean
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: person_is_aged_blind_or_disabled
"""
    )
    output_file.with_name("ssi-mandatory-group.test.yaml").write_text(
        """- name: aged_blind_or_disabled_person_is_in_group
  period:
    period_kind: month
    start: '2026-01-01'
    end: '2026-01-31'
  input:
    us:regulations/42-cfr/435/120/ssi-mandatory-group#input.person_is_aged_blind_or_disabled: true
  output:
    us:regulations/42-cfr/435/120/ssi-mandatory-group#ssi_mandatory_group: true
"""
    )

    with pytest.raises(UnsafeRulespecContextPath, match="canonical rulespec-<country>"):
        with _rulespec_validation_target(output_file, policy_repo / "us"):
            pass


def test_materialize_eval_artifact_repairs_source_table_band_scalars(tmp_path):
    output_file = tmp_path / "runner" / "statutes" / "26" / "3241" / "b.yaml"
    llm_response = """=== FILE: b.yaml ===
format: rulespec/v1
module:
  summary: Section defines applicable percentages by benefits ratio.
rules:
  - name: average_account_benefits_ratio_lower_bound_band_0
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2026-01-01'
        formula: 2.5
  - name: average_account_benefits_ratio_upper_bound_band_0
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2026-01-01'
        formula: 3.0
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if average_account_benefits_ratio < average_account_benefits_ratio_lower_bound_band_0:
            -1
          elif average_account_benefits_ratio < average_account_benefits_ratio_upper_bound_band_0:
            0
          else:
            1
  - name: applicable_percentage_3201_by_average_account_benefits_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_band
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.049
          1: 0
=== FILE: b.test.yaml ===
- name: selector_band
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/3241/b#input.average_account_benefits_ratio: 2.75
  output:
    us:statutes/26/3241/b#average_account_benefits_ratio_lower_bound_band_0: 2.5
    us:statutes/26/3241/b#applicable_percentage_3201_by_average_account_benefits_ratio_band:
      0: 0.049
      1: 0
    us:statutes/26/3241/b#average_account_benefits_ratio_band: 0
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="Tax rate schedule | Average account benefits ratio | 2.5 | 3.0",
    )

    assert wrote is True
    content = output_file.read_text()
    assert "average_account_benefits_ratio_lower_bound_band_0" in content
    assert (
        "average_account_benefits_ratio < "
        "average_account_benefits_ratio_lower_bound_band_0"
    ) in content
    assert "elif" not in content
    assert "else if" not in content
    test_content = output_file.with_name("b.test.yaml").read_text()
    assert "average_account_benefits_ratio_lower_bound_band_0" in test_content
    assert (
        "applicable_percentage_3201_by_average_account_benefits_ratio_band"
        not in test_content
    )
    assert "average_account_benefits_ratio_band" in test_content


def test_materialize_eval_artifact_repairs_named_band_threshold_scalars(tmp_path):
    output_file = tmp_path / "runner" / "statutes" / "26" / "3241" / "b.yaml"
    llm_response = """=== FILE: b.yaml ===
format: rulespec/v1
module:
  summary: Section defines applicable percentages by benefits ratio.
rules:
  - name: average_account_benefits_ratio_band_threshold_2_5
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2026-01-01'
        formula: 2.5
  - name: average_account_benefits_ratio_band_threshold_3_0
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2026-01-01'
        formula: 3.0
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if average_account_benefits_ratio < average_account_benefits_ratio_band_threshold_2_5: 1
          else: if average_account_benefits_ratio < average_account_benefits_ratio_band_threshold_3_0: 2
          else: 3
  - name: applicable_percentage_3201_by_average_account_benefits_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_band
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 0.049
          2: 0.049
          3: 0
=== FILE: b.test.yaml ===
- name: selector_band
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/3241/b#input.average_account_benefits_ratio: 2.75
  output:
    us:statutes/26/3241/b#average_account_benefits_ratio_band_threshold_2_5: 2.5
    us:statutes/26/3241/b#applicable_percentage_3201_by_average_account_benefits_ratio_band:
      1: 0.049
      2: 0.049
      3: 0
    us:statutes/26/3241/b#average_account_benefits_ratio_band: 2
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="Tax rate schedule | Average account benefits ratio | 2.5 | 3.0",
    )

    assert wrote is True
    content = output_file.read_text()
    assert "average_account_benefits_ratio_band_threshold_2_5" in content
    assert "average_account_benefits_ratio_band_threshold_3_0" in content
    assert (
        "average_account_benefits_ratio < "
        "average_account_benefits_ratio_band_threshold_2_5"
    ) in content
    assert (
        "average_account_benefits_ratio < "
        "average_account_benefits_ratio_band_threshold_3_0"
    ) in content
    test_content = output_file.with_name("b.test.yaml").read_text()
    assert "average_account_benefits_ratio_band_threshold_2_5" in test_content
    assert (
        "applicable_percentage_3201_by_average_account_benefits_ratio_band"
        not in test_content
    )
    assert "average_account_benefits_ratio_band" in test_content


def test_materialize_eval_artifact_repairs_chained_conditionals_without_table_scalars(
    tmp_path,
):
    output_file = tmp_path / "runner" / "statutes" / "26" / "3241" / "b.yaml"
    llm_response = """=== FILE: b.yaml ===
format: rulespec/v1
module:
  summary: Section defines applicable percentages by benefits ratio.
rules:
  - name: average_account_benefits_ratio_at_least_threshold_by_band
    kind: parameter
    dtype: Decimal
    indexed_by: average_account_benefits_ratio_band
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 2.5
          2: 3.0
  - name: average_account_benefits_ratio_but_less_than_threshold_by_band
    kind: parameter
    dtype: Decimal
    indexed_by: average_account_benefits_ratio_band
    metadata:
      proof:
        atoms:
          - path: versions[0].values
            kind: parameter_table
            source:
              corpus_citation_path: us/statute/26/3241
              table:
                header: Tax rate schedule
                row_key: Average account benefits ratio band
                column_key: But less than
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 3.0
          2: 1000000.0
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if average_account_benefits_ratio < average_account_benefits_ratio_at_least_threshold_by_band[1]:
            0
          else if average_account_benefits_ratio >= average_account_benefits_ratio_at_least_threshold_by_band[2]:
            2
          else:
            1
=== FILE: b.test.yaml ===
- name: selector_band
  period: 2026
  input:
    average_account_benefits_ratio: 2.75
  output:
    average_account_benefits_ratio_band: 1
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="Tax rate schedule | Average account benefits ratio | 2.5 | 3.0",
    )

    assert wrote is True
    content = output_file.read_text()
    assert "elif" not in content
    assert "else if" not in content
    assert "1000000.0" not in content
    payload = yaml.safe_load(content)
    assert 2 not in payload["rules"][1]["versions"][0]["values"]
    formula = payload["rules"][2]["versions"][0]["formula"]
    assert formula.startswith("if average_account_benefits_ratio < ")
    assert " else: if average_account_benefits_ratio >= " in formula


def test_materialize_eval_artifact_repairs_multiline_conditional_branch(
    tmp_path,
):
    output_file = tmp_path / "runner" / "statutes" / "26" / "1402" / "b.yaml"
    llm_response = """=== FILE: b.yaml ===
format: rulespec/v1
module:
  summary: Section defines self-employment income.
rules:
  - name: self_employment_income_for_section_1401_a
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if self_employment_income_excluded_as_nonresident_alien:
            0
          else if net_earnings_from_self_employment < minimum_self_employment_income_threshold:
            0
          else:
            min(
              max(0, net_earnings_from_self_employment),
              max(
                0,
                contribution_and_benefit_base_under_section_230_of_social_security_act
                  - wages_paid_to_individual_during_taxable_year
              )
            )
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="The OASDI base is reduced by wages paid during the year.",
    )

    assert wrote is True
    payload = yaml.safe_load(output_file.read_text())
    formula = payload["rules"][0]["versions"][0]["formula"]
    assert "else if" not in formula
    assert "elif" not in formula
    assert (
        "else: if net_earnings_from_self_employment "
        "< minimum_self_employment_income_threshold: 0 else: min("
    ) in formula
    assert (
        "contribution_and_benefit_base_under_section_230_of_social_security_act"
        in formula
    )


def test_materialize_eval_artifact_repairs_multiline_else_if_conditions(
    tmp_path,
):
    output_file = tmp_path / "runner" / "statutes" / "26" / "1402" / "b.yaml"
    llm_response = """=== FILE: b.yaml ===
format: rulespec/v1
module:
  summary: Section defines self-employment income.
rules:
  - name: self_employment_income_for_section_1401_a
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if is_nonresident_alien_individual_for_chapter_1402:
            0
          else if church_employee_income
              and apply_section_1402_j_2_special_rules_for_church_income:
            self_employment_income_for_church_employee_under_section_1402_j_2
          else if net_earnings_from_self_employment
              < self_employment_income_small_amount_exclusion_threshold:
            0
          else:
            net_earnings_from_self_employment
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="The special rules apply in the case of church employee income.",
    )

    assert wrote is True
    payload = yaml.safe_load(output_file.read_text())
    formula = payload["rules"][0]["versions"][0]["formula"]
    assert "else if" not in formula
    assert "elif" not in formula
    assert (
        "if church_employee_income and "
        "apply_section_1402_j_2_special_rules_for_church_income: "
        "self_employment_income_for_church_employee_under_section_1402_j_2"
    ) in formula
    assert (
        "if net_earnings_from_self_employment "
        "< self_employment_income_small_amount_exclusion_threshold: 0"
    ) in formula


def test_materialize_eval_artifact_repairs_then_conditionals(
    tmp_path,
):
    output_file = tmp_path / "runner" / "statutes" / "26" / "1402" / "b.yaml"
    llm_response = """=== FILE: b.yaml ===
format: rulespec/v1
module:
  summary: Section defines self-employment income.
rules:
  - name: self_employment_income
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if net_earnings_from_self_employment >= self_employment_income_minimum_amount
              and not individual_is_nonresident_alien_individual
              then: net_earnings_from_self_employment else: 0
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="Self-employment income excludes net earnings below $400.",
    )

    assert wrote is True
    payload = yaml.safe_load(output_file.read_text())
    formula = payload["rules"][0]["versions"][0]["formula"]
    assert "then:" not in formula
    assert (
        "if net_earnings_from_self_employment >= self_employment_income_minimum_amount "
        "and not individual_is_nonresident_alien_individual: "
        "net_earnings_from_self_employment else: 0"
    ) == formula


def test_materialize_eval_artifact_repairs_python_ternary_formulas(
    tmp_path,
):
    output_file = tmp_path / "runner" / "statutes" / "26" / "164" / "f.yaml"
    llm_response = """=== FILE: f.yaml ===
format: rulespec/v1
module:
  summary: Section allows a self-employment tax deduction.
rules:
  - name: self_employment_tax_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          self_employment_tax_deduction_fraction * (
            old_age_survivors_and_disability_insurance_tax
            + self_employment_income_tax
          ) if taxpayer_is_individual else 0
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="An individual may deduct one-half of self-employment taxes.",
    )

    assert wrote is True
    payload = yaml.safe_load(output_file.read_text())
    formula = payload["rules"][0]["versions"][0]["formula"]
    assert (
        "if taxpayer_is_individual: self_employment_tax_deduction_fraction * "
        "( old_age_survivors_and_disability_insurance_tax + "
        "self_employment_income_tax ) else: 0"
    ) == formula


def test_materialize_eval_artifact_preserves_open_interval_source_table_rows(
    tmp_path,
):
    output_file = tmp_path / "runner" / "statutes" / "26" / "3241" / "b.yaml"
    source_text = """Tax rate schedule | Average account benefits ratio | Applicable percentage for sections 3211(b) and 3221(b) | Applicable percentage for section 3201(b)
| At least | But less than | | | .............. | 2.5 | 22.1 | 4.9 |
| 2.5 | 6.1 | 18.1 | 4.9 |
| 6.1 | 9.0 | 12.6 | 4.4 |
| 9.0 | .............. | 8.2 | 0 |"""
    llm_response = """=== FILE: b.yaml ===
format: rulespec/v1
module:
  summary: |-
    Tax rate schedule | Average account benefits ratio | Applicable percentage for sections 3211(b) and 3221(b) | Applicable percentage for section 3201(b)
    | At least | But less than | | | .............. | 2.5 | 22.1 | 4.9 |
    | 2.5 | 6.1 | 18.1 | 4.9 |
    | 6.1 | 9.0 | 12.6 | 4.4 |
    | 9.0 | .............. | 8.2 | 0 |
rules:
  - name: section_3241b_average_account_benefits_ratio_bracket_lower_bound
    kind: parameter
    dtype: Decimal
    indexed_by: average_account_benefits_ratio_bracket
    metadata:
      proof:
        atoms:
          - path: versions[0].values
            kind: parameter_table
            source:
              table:
                header: Tax rate schedule
                row_key: average account benefits ratio band
                column_key: At least
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 2.5
          2: 6.1
          3: 9.0
  - name: section_3241b_average_account_benefits_ratio_bracket_upper_bound
    kind: parameter
    dtype: Decimal
    indexed_by: average_account_benefits_ratio_bracket
    metadata:
      proof:
        atoms:
          - path: versions[0].values
            kind: parameter_table
            source:
              table:
                header: Tax rate schedule
                row_key: average account benefits ratio band
                column_key: But less than
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 6.1
          2: 9.0
  - name: average_account_benefits_ratio_bracket
    kind: derived
    entity: TaxUnit
    dtype: Integer
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if average_account_benefits_ratio >= section_3241b_average_account_benefits_ratio_bracket_lower_bound[1] and average_account_benefits_ratio < section_3241b_average_account_benefits_ratio_bracket_upper_bound[1]:
            1
          else if average_account_benefits_ratio >= section_3241b_average_account_benefits_ratio_bracket_lower_bound[2] and average_account_benefits_ratio < section_3241b_average_account_benefits_ratio_bracket_upper_bound[2]:
            2
          else:
            0
  - name: section_3241b_sections_3211b_and_3221b_applicable_percentage_points
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_bracket
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.0
          1: 0.221
          2: 0.181
          3: 0.126
  - name: section_3241b_section_3201b_applicable_percentage_points
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_bracket
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.0
          1: 0.049
          2: 0.049
          3: 0.044
  - name: section_3211b_applicable_percentage
    kind: derived
    entity: TaxUnit
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        formula: section_3241b_sections_3211b_and_3221b_applicable_percentage_points[average_account_benefits_ratio_bracket]
  - name: section_3201b_applicable_percentage
    kind: derived
    entity: TaxUnit
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        formula: section_3241b_section_3201b_applicable_percentage_points[average_account_benefits_ratio_bracket]
=== FILE: b.test.yaml ===
- name: below_open_lower_row
  period: 2026
  input:
    average_account_benefits_ratio: 2.49
  output:
    average_account_benefits_ratio_bracket: 0
    section_3211b_applicable_percentage: 0.0
    section_3201b_applicable_percentage: 0.0
- name: boundary_2_5_uses_second_source_row
  period: 2026
  input:
    average_account_benefits_ratio: 2.5
  output:
    section_3211b_applicable_percentage: 0.221
    section_3201b_applicable_percentage: 0.049
- name: boundary_6_1_uses_next_source_row
  period: 2026
  input:
    average_account_benefits_ratio: 6.1
  output:
    section_3211b_applicable_percentage: 0.181
    section_3201b_applicable_percentage: 0.049
- name: open_upper_row
  period: 2026
  input:
    average_account_benefits_ratio: 9.0
  output:
    section_3211b_applicable_percentage: 0.126
    section_3201b_applicable_percentage: 0.044
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text=source_text,
    )

    assert wrote is True
    payload = yaml.safe_load(output_file.read_text())
    rule_names = {rule["name"] for rule in payload["rules"]}
    assert (
        "section_3241b_average_account_benefits_ratio_bracket_lower_bound" in rule_names
    )
    assert (
        "section_3241b_average_account_benefits_ratio_bracket_upper_bound" in rule_names
    )
    lower_bound = next(
        rule
        for rule in payload["rules"]
        if rule["name"]
        == "section_3241b_average_account_benefits_ratio_bracket_lower_bound"
    )
    upper_bound = next(
        rule
        for rule in payload["rules"]
        if rule["name"]
        == "section_3241b_average_account_benefits_ratio_bracket_upper_bound"
    )
    assert lower_bound["versions"][0]["values"] == {2: 2.5, 3: 6.1, 4: 9.0}
    assert upper_bound["versions"][0]["values"] == {1: 2.5, 2: 6.1, 3: 9.0}
    selector = next(
        rule
        for rule in payload["rules"]
        if rule["name"] == "average_account_benefits_ratio_bracket"
    )
    assert selector["versions"][0]["formula"] == (
        "if average_account_benefits_ratio < "
        "section_3241b_average_account_benefits_ratio_bracket_upper_bound[1]: 1 else: "
        "if average_account_benefits_ratio >= "
        "section_3241b_average_account_benefits_ratio_bracket_lower_bound[2] and "
        "average_account_benefits_ratio < "
        "section_3241b_average_account_benefits_ratio_bracket_upper_bound[2]: 2 else: "
        "if average_account_benefits_ratio >= "
        "section_3241b_average_account_benefits_ratio_bracket_lower_bound[3] and "
        "average_account_benefits_ratio < "
        "section_3241b_average_account_benefits_ratio_bracket_upper_bound[3]: 3 else: 4"
    )
    rates = next(
        rule
        for rule in payload["rules"]
        if rule["name"]
        == "section_3241b_sections_3211b_and_3221b_applicable_percentage_points"
    )
    assert rates["versions"][0]["values"] == {
        1: 0.221,
        2: 0.181,
        3: 0.126,
        4: 0.082,
    }
    cases = {
        case["name"]: case
        for case in yaml.safe_load(output_file.with_name("b.test.yaml").read_text())
    }
    assert (
        cases["below_open_lower_row"]["output"][
            "average_account_benefits_ratio_bracket"
        ]
        == 1
    )
    assert (
        cases["below_open_lower_row"]["output"]["section_3211b_applicable_percentage"]
        == 0.221
    )
    assert (
        cases["boundary_2_5_uses_second_source_row"]["output"][
            "section_3211b_applicable_percentage"
        ]
        == 0.181
    )
    assert (
        cases["boundary_6_1_uses_next_source_row"]["output"][
            "section_3211b_applicable_percentage"
        ]
        == 0.126
    )
    assert (
        cases["open_upper_row"]["output"]["section_3211b_applicable_percentage"]
        == 0.082
    )


def test_run_source_eval_retries_once_when_first_response_has_no_rulespec(tmp_path):
    policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path, "source states 451."
    )
    first_response = EvalPromptResponse(
        text="I'm going to encode a compact source-faithful slice.",
        duration_ms=10,
        trace={"attempt": "initial"},
    )
    second_response = EvalPromptResponse(
        text=(
            "=== FILE: sample.yaml ===\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: source states 451.\n"
            "rules: []\n"
            "=== FILE: sample.test.yaml ===\n"
            "[]\n"
        ),
        duration_ms=20,
        trace={"attempt": "retry"},
    )

    with (
        patch(
            "axiom_encode.harness.evals._run_prompt_eval",
            side_effect=[first_response, second_response],
        ) as mock_prompt_eval,
        patch("axiom_encode.harness.evals.evaluate_artifact", return_value=None),
    ):
        [result] = run_source_eval(
            source_unit=source_unit,
            runner_specs=["codex:gpt-5.4"],
            output_root=tmp_path / "out",
            policy_path=policy_repo_root,
            local_corpus_release=corpus_release,
            runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
        )

    assert result.success is True
    assert result.retry_count == 1
    assert result.duration_ms == 30
    assert Path(result.output_file).exists()
    assert mock_prompt_eval.call_count == 2
    retry_prompt = mock_prompt_eval.call_args_list[1].args[2]
    assert "previous response did not contain a RuleSpec artifact" in retry_prompt
    assert "Do not narrate your plan" in retry_prompt


@pytest.mark.parametrize("mode", ["cold", "repo-augmented"])
def test_run_source_eval_does_not_promote_context_to_source_authority(tmp_path, mode):
    policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path,
        "Primary page text ends mid-sentence.",
        citation_path="us/regulation/example/page-1",
    )
    continuation = tmp_path / "continuation.txt"
    continuation.write_text(
        "Primary source continuation for sample.\n"
        "Corpus citation path: us/regulation/example/page-2\n\n"
        "The amount shall be rounded down to the next lower whole dollar.\n"
    )
    response = EvalPromptResponse(
        text=(
            "=== FILE: sample.yaml ===\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  source_verification:\n"
            "    corpus_citation_path: us/regulation/example/page-1\n"
            "  summary: source states 451.\n"
            "rules: []\n"
            "=== FILE: sample.test.yaml ===\n"
            "[]\n"
        ),
        duration_ms=10,
        trace={},
    )

    with (
        patch("axiom_encode.harness.evals._run_prompt_eval", return_value=response),
        patch("axiom_encode.harness.evals.evaluate_artifact", return_value=None),
    ):
        [result] = run_source_eval(
            source_unit=source_unit,
            runner_specs=["codex:gpt-5.4"],
            output_root=tmp_path / "out",
            policy_path=policy_repo_root,
            local_corpus_release=corpus_release,
            runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
            extra_context_paths=[continuation],
            mode=mode,
        )

    manifest = json.loads(Path(result.context_manifest_file).read_text())
    source_text = (
        Path(result.context_manifest_file).parent / manifest["source_text_file"]
    ).read_text()
    assert "Primary page text ends mid-sentence." in source_text
    assert "Primary source continuation" not in source_text
    assert "rounded down to the next lower whole dollar" not in source_text
    assert "corpus_citation_paths" not in manifest["source_metadata"]
    assert "primary_source_continuations" not in manifest["source_metadata"]
    assert manifest["source_metadata"]["source_attestation"]["rulespec_root"] == str(
        policy_repo_root.resolve()
    )
    assert result.source_attestation is not None
    assert result.source_attestation["rulespec_root"] == str(policy_repo_root.resolve())
    generation_input = source_text.strip().encode()
    assert (
        manifest["source_metadata"]["source_attestation"]["generation_input_sha256"]
        == hashlib.sha256(generation_input).hexdigest()
    )


def test_run_source_eval_rejects_symlinked_explicit_context(tmp_path):
    policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
    outside_file = tmp_path / "outside-continuation.txt"
    outside_file.write_text(
        "Primary source continuation for sample.\n\n"
        "OPENAI_API_KEY=sentinel-secret-value\n"
    )
    continuation = tmp_path / "continuation.txt"
    continuation.symlink_to(outside_file)
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path, "Primary source text."
    )

    with pytest.raises(UnsafeRulespecContextPath, match="symlink"):
        run_source_eval(
            source_unit=source_unit,
            runner_specs=["openai:gpt-5.4"],
            output_root=tmp_path / "out",
            policy_path=policy_repo_root,
            local_corpus_release=corpus_release,
            runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
            extra_context_paths=[continuation],
            mode="repo-augmented",
        )


def test_review_findings_are_persisted_and_mandatory_in_prompt(tmp_path):
    policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
    findings = tmp_path / "review-findings.md"
    findings.write_bytes(
        b"- Start the FY 2025 amount on October 1.\r\n"
        b"- Preserve the FY 2026 imported amount and boundary test.\r\n"
    )
    workspace = prepare_eval_workspace(
        citation="us/manual/example/block-1",
        runner=parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="The amount is determined for each federal fiscal year.",
        axiom_rules_path=policy_repo_root,
        mode="cold",
        review_findings_paths=[findings],
    )

    prompt = _build_eval_prompt(
        "us/manual/example/block-1",
        "cold",
        workspace,
        workspace.context_files,
        target_file_name="block-1.yaml",
    )
    manifest = json.loads(workspace.manifest_file.read_text())

    assert "Mandatory independent-review corrections:" in prompt
    assert "address every source-faithful finding" in prompt
    assert "Do not narrow the module" in prompt
    assert "Start the FY 2025 amount on October 1" in prompt
    assert "Preserve the FY 2026 imported amount" in prompt
    assert len(manifest["review_findings_files"]) == 1
    evidence = manifest["review_findings_files"][0]
    persisted = workspace.root / evidence["workspace_path"]
    expected = (
        "- Start the FY 2025 amount on October 1.\n"
        "- Preserve the FY 2026 imported amount and boundary test.\n"
    )
    assert persisted.read_text() == expected
    assert evidence["content"] == expected
    assert evidence["sha256"] == hashlib.sha256(expected.encode()).hexdigest()


def test_structured_review_contract_is_bound_into_context_manifest(tmp_path):
    policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
    required_case = {
        "name": "exact case",
        "period": {
            "period_kind": "tax_year",
            "start": "2025-01-01",
            "end": "2025-12-31",
        },
        "input": {"us:statutes/26/1#input.single": True},
        "required_output": {"us:statutes/26/1#deduction": 12500},
    }
    contract = _eval_review_contract_manifest_payload(
        citation="us/statute/26/1",
        rulespec_path="us/statutes/26/1.yaml",
        required_deferred_output_contracts=(("missing#output", "exact reason"),),
        required_test_case_contracts=(required_case,),
    )
    workspace = prepare_eval_workspace(
        citation="us/statute/26/1",
        runner=parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="Source text.",
        axiom_rules_path=policy_repo_root,
        mode="cold",
        review_contract=contract,
    )

    manifest = json.loads(workspace.manifest_file.read_text())
    assert manifest["review_contract"] == {
        "schema": "axiom-encode/review-contract/v2",
        "citation": "us/statute/26/1",
        "rulespec_path": "us/statutes/26/1.yaml",
        "required_deferred_outputs": [
            {"output": "missing#output", "reason": "exact reason"}
        ],
        "required_test_cases": [required_case],
    }


def test_review_findings_reject_empty_file(tmp_path):
    policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
    findings = tmp_path / "review-findings.md"
    findings.write_text("\n")

    with pytest.raises(ValueError, match="Review findings file is empty"):
        prepare_eval_workspace(
            citation="us/manual/example/block-1",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Source text.",
            axiom_rules_path=policy_repo_root,
            mode="cold",
            review_findings_paths=[findings],
        )


def test_review_findings_reject_symlink(tmp_path):
    policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
    target = tmp_path / "actual-findings.md"
    target.write_text("- Correct the date.\n")
    findings = tmp_path / "review-findings.md"
    findings.symlink_to(target)

    with pytest.raises(UnsafeRulespecContextPath, match="symlink"):
        prepare_eval_workspace(
            citation="us/manual/example/block-1",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Source text.",
            axiom_rules_path=policy_repo_root,
            mode="cold",
            review_findings_paths=[findings],
        )


def test_empty_artifact_retry_prompt_uses_minimal_source_scope_protocol():
    from axiom_encode.prompts.encoder import SOURCE_SCOPE_PROTOCOL

    original_prompt = (
        "Task preface.\n"
        f"{SOURCE_SCOPE_PROTOCOL}\n"
        "Additional encoding guidance:\n"
        "- Keep this instruction.\n"
    )

    retry_prompt = _build_empty_artifact_retry_prompt(
        original_prompt,
        target_file_name="sample.yaml",
        include_tests=True,
    )

    assert "Source-scope protocol (minimal):" in retry_prompt
    assert "Additional encoding guidance:" in retry_prompt
    assert "- Keep this instruction." in retry_prompt
    assert "do not promote it to a household, unit" not in retry_prompt
    assert "Return exactly this two-file bundle" in retry_prompt


def test_empty_artifact_retry_prompt_preserves_cache_boundary():
    stable_prefix = "Stable source, context, and schema sections.\n"
    dynamic_suffix = "Output rules.\n"
    original_prompt = evals_module._PromptWithCacheBoundary(
        stable_prefix + dynamic_suffix,
        cache_prefix_length=len(stable_prefix),
    )

    retry_prompt = _build_empty_artifact_retry_prompt(
        original_prompt,
        target_file_name="sample.yaml",
        include_tests=True,
    )

    assert retry_prompt.startswith(str(original_prompt))
    retry_prefix, retry_suffix = evals_module._openai_prompt_cache_parts(retry_prompt)
    assert retry_prefix == stable_prefix
    assert "previous response did not contain a RuleSpec artifact" in retry_suffix
    assert "Return exactly this two-file bundle" in retry_suffix
    assert evals_module._openai_prompt_cache_key(
        "gpt-5.6-terra", retry_prefix
    ) == evals_module._openai_prompt_cache_key("gpt-5.6-terra", stable_prefix)


def test_empty_artifact_runtime_uses_bound_max_attempts(monkeypatch, tmp_path):
    monkeypatch.setattr(evals_module, "_EMPTY_ARTIFACT_MAX_ATTEMPTS", 1)
    response = EvalPromptResponse(text="", duration_ms=1)

    with (
        patch(
            "axiom_encode.harness.evals._run_prompt_eval",
            return_value=response,
        ) as mock_prompt,
        patch(
            "axiom_encode.harness.evals._materialize_eval_artifact",
            return_value=False,
        ),
    ):
        _response, wrote_artifact, retry_count, materialized_paths = (
            evals_module._run_prompt_eval_with_empty_artifact_retry(
                parse_runner_spec("codex:gpt-5.4"),
                SimpleNamespace(root=tmp_path),
                "prompt",
                tmp_path / "output.yaml",
                "source",
                "output.yaml",
                False,
            )
        )

    assert mock_prompt.call_count == 1
    assert wrote_artifact is False
    assert retry_count == 0
    assert materialized_paths == frozenset()


def test_materialized_artifact_crossing_case_budget_is_discarded(
    monkeypatch,
    tmp_path,
):
    clock = [0.0]
    artifact_root = tmp_path / "out"
    artifact_root.mkdir()
    output_file = artifact_root / "sample.yaml"
    response = EvalPromptResponse(
        text="format: rulespec/v1\nrules: []\n",
        duration_ms=4000,
    )
    monkeypatch.setattr(evals_module.time, "monotonic", lambda: clock[0])

    def finish_generation(*_args, **_kwargs):
        clock[0] = 4.0
        return response

    def materialize_after_deadline(*args, **kwargs):
        wrote_artifact = _materialize_eval_artifact(*args, **kwargs)
        clock[0] = 6.0
        return wrote_artifact

    with (
        evals_module._active_eval_case_budget(5),
        patch(
            "axiom_encode.harness.evals._run_prompt_eval",
            side_effect=finish_generation,
        ),
        patch(
            "axiom_encode.harness.evals._materialize_eval_artifact",
            side_effect=materialize_after_deadline,
        ),
    ):
        result, wrote_artifact, retry_count, materialized_paths = (
            evals_module._run_prompt_eval_with_empty_artifact_retry(
                parse_runner_spec("codex:gpt-5.4"),
                SimpleNamespace(root=tmp_path),
                "prompt",
                output_file,
                "source",
                "sample.yaml",
                False,
                artifact_root=artifact_root,
            )
        )

    assert wrote_artifact is False
    assert retry_count == 0
    assert materialized_paths == frozenset()
    assert not output_file.exists()
    assert not output_file.with_suffix(".test.yaml").exists()
    assert result.timed_out is True
    assert result.timeout_stage == "case_budget"
    assert result.timeout_reason == "wall"
    assert result.timeout_seconds == 5
    assert result.timeout_attempts == 1
    assert "case budget" in (result.error or "").lower()


def test_workspace_artifact_crossing_case_budget_is_discarded(
    monkeypatch,
    tmp_path,
):
    clock = [0.0]
    artifact_root = tmp_path / "out"
    artifact_root.mkdir()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    output_file = artifact_root / "sample.yaml"
    workspace_file = workspace_root / "sample.yaml"
    workspace_test_file = workspace_root / "sample.test.yaml"
    response = EvalPromptResponse(
        text="Artifact written directly to the workspace.",
        duration_ms=4000,
    )
    monkeypatch.setattr(evals_module.time, "monotonic", lambda: clock[0])

    def finish_generation(*_args, **_kwargs):
        workspace_file.write_text("format: rulespec/v1\nrules: []\n")
        workspace_test_file.write_text("[]\n")
        clock[0] = 4.0
        return response

    def materialize_after_deadline(*args, **kwargs):
        wrote_artifact = _materialize_eval_artifact(*args, **kwargs)
        clock[0] = 6.0
        return wrote_artifact

    with (
        evals_module._active_eval_case_budget(5),
        patch(
            "axiom_encode.harness.evals._run_prompt_eval",
            side_effect=finish_generation,
        ),
        patch(
            "axiom_encode.harness.evals._materialize_eval_artifact",
            side_effect=materialize_after_deadline,
        ),
    ):
        result, wrote_artifact, retry_count, materialized_paths = (
            evals_module._run_prompt_eval_with_empty_artifact_retry(
                parse_runner_spec("codex:gpt-5.4"),
                SimpleNamespace(root=workspace_root),
                "prompt",
                output_file,
                "source",
                "sample.yaml",
                True,
                artifact_root=artifact_root,
            )
        )

    assert wrote_artifact is False
    assert retry_count == 0
    assert materialized_paths == frozenset()
    assert not output_file.exists()
    assert not output_file.with_suffix(".test.yaml").exists()
    assert not workspace_file.exists()
    assert not workspace_test_file.exists()
    assert result.timed_out is True
    assert result.timeout_stage == "case_budget"


def test_encoder_timeout_discards_response_and_direct_workspace_artifacts(tmp_path):
    artifact_root = tmp_path / "out"
    artifact_root.mkdir()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    output_file = artifact_root / "sample.yaml"
    workspace_file = workspace_root / "sample.yaml"
    workspace_test_file = workspace_root / "sample.test.yaml"
    bundle = (
        "=== FILE: sample.yaml ===\n"
        "format: rulespec/v1\n"
        "rules: []\n"
        "=== FILE: sample.test.yaml ===\n"
        "[]\n"
    )

    def timed_out_generation(*_args, **_kwargs):
        workspace_file.write_text("format: rulespec/v1\nrules: []\n")
        workspace_test_file.write_text("[]\n")
        return EvalPromptResponse(
            text=bundle,
            duration_ms=600_000,
            error="Codex eval timed out",
            timed_out=True,
            timeout_stage="encoder",
            timeout_reason="wall",
            timeout_seconds=600,
            timeout_attempts=1,
        )

    with patch(
        "axiom_encode.harness.evals._run_prompt_eval",
        side_effect=timed_out_generation,
    ) as mock_prompt_eval:
        response, wrote_artifact, retry_count, materialized_paths = (
            evals_module._run_prompt_eval_with_empty_artifact_retry(
                parse_runner_spec("codex:gpt-5.4"),
                SimpleNamespace(root=workspace_root),
                "prompt",
                output_file,
                "source",
                "sample.yaml",
                True,
                artifact_root=artifact_root,
            )
        )

    assert response.timed_out is True
    assert response.text == ""
    assert wrote_artifact is False
    assert retry_count == 1
    assert materialized_paths == frozenset()
    assert not output_file.exists()
    assert not output_file.with_suffix(".test.yaml").exists()
    assert not workspace_file.exists()
    assert not workspace_test_file.exists()
    assert mock_prompt_eval.call_count == 2


def test_run_source_eval_retries_once_when_first_response_times_out(tmp_path):
    policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path, "source states 451."
    )
    first_response = EvalPromptResponse(
        text="",
        duration_ms=300000,
        trace={"timed_out": True, "timeout_reason": "idle"},
        error="Codex eval timed out",
    )
    second_response = EvalPromptResponse(
        text=(
            "=== FILE: sample.yaml ===\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: source states 451.\n"
            "rules: []\n"
            "=== FILE: sample.test.yaml ===\n"
            "[]\n"
        ),
        duration_ms=20,
        trace={"attempt": "retry"},
    )

    with (
        patch(
            "axiom_encode.harness.evals._run_prompt_eval",
            side_effect=[first_response, second_response],
        ) as mock_prompt_eval,
        patch("axiom_encode.harness.evals.evaluate_artifact", return_value=None),
    ):
        [result] = run_source_eval(
            source_unit=source_unit,
            runner_specs=["codex:gpt-5.4"],
            output_root=tmp_path / "out",
            policy_path=policy_repo_root,
            local_corpus_release=corpus_release,
            runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
        )

    assert result.success is True
    assert result.retry_count == 1
    assert result.error is None
    assert result.duration_ms == 300020
    assert Path(result.output_file).exists()
    assert mock_prompt_eval.call_count == 2


def test_exhausted_encoder_timeout_classification_survives_result_round_trip(
    tmp_path,
):
    policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path, "source states 451."
    )
    timed_out_responses = [
        EvalPromptResponse(
            text="",
            duration_ms=1234000,
            trace={
                "timed_out": True,
                "timeout_reason": "wall",
                "timeout_seconds": 1234,
            },
            error="Claude eval timed out",
            timed_out=True,
            timeout_stage="encoder",
            timeout_reason="wall",
            timeout_seconds=1234,
            timeout_attempts=1,
        )
        for _ in range(2)
    ]

    with patch(
        "axiom_encode.harness.evals._run_prompt_eval",
        side_effect=timed_out_responses,
    ):
        [result] = run_source_eval(
            source_unit=source_unit,
            runner_specs=["claude:opus"],
            output_root=tmp_path / "out",
            policy_path=policy_repo_root,
            local_corpus_release=corpus_release,
            runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
        )

    restored = _eval_result_from_payload(result.to_dict())
    assert restored.success is False
    assert restored.failure_kind == "timeout"
    assert restored.timed_out is True
    assert restored.timeout_stage == "encoder"
    assert restored.timeout_reason == "wall"
    assert restored.timeout_seconds == 1234
    assert restored.timeout_attempts == 2
    assert restored.metrics is None
    assert restored.output_file == ""


def test_timed_out_truncated_artifact_is_never_scored_as_validation_failure(
    tmp_path,
):
    policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path, "source states 451."
    )
    truncated_bundle = (
        "=== FILE: sample.yaml ===\n"
        "format: rulespec/v1\n"
        "module:\n"
        "  summary: truncated timeout output\n"
        "rules: []\n"
        "=== FILE: sample.test.yaml ===\n"
        "[]\n"
    )
    timed_out_responses = [
        EvalPromptResponse(
            text=truncated_bundle,
            duration_ms=600_000,
            trace={
                "timed_out": True,
                "timeout_stage": "encoder",
                "timeout_reason": "wall",
                "timeout_seconds": 600,
            },
            timed_out=True,
            timeout_stage="encoder",
            timeout_reason="wall",
            timeout_seconds=600,
            timeout_attempts=1,
        )
        for _ in range(2)
    ]
    failed_metrics = EvalArtifactMetrics(
        compile_pass=False,
        compile_issues=["truncated artifact"],
        ci_pass=False,
        ci_issues=[],
        embedded_source_present=False,
        grounded_numeric_count=0,
        ungrounded_numeric_count=0,
        grounding=[],
    )

    with (
        patch(
            "axiom_encode.harness.evals._run_prompt_eval",
            side_effect=timed_out_responses,
        ) as mock_prompt_eval,
        patch(
            "axiom_encode.harness.evals.evaluate_artifact",
            return_value=failed_metrics,
        ) as mock_evaluate,
    ):
        [result] = run_source_eval(
            source_unit=source_unit,
            runner_specs=["codex:gpt-5.4"],
            output_root=tmp_path / "out",
            policy_path=policy_repo_root,
            local_corpus_release=corpus_release,
            runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
        )

    restored = _eval_result_from_payload(result.to_dict())
    assert restored.success is False
    assert restored.failure_kind == "timeout"
    assert restored.timed_out is True
    assert restored.timeout_stage == "encoder"
    assert restored.timeout_reason == "wall"
    assert restored.timeout_seconds == 600
    assert restored.timeout_attempts == 2
    assert restored.metrics is None
    assert restored.output_file == ""
    assert restored.generated_output_sha256 is None
    assert mock_prompt_eval.call_count == 2
    mock_evaluate.assert_not_called()


def test_timed_out_response_outcome_prioritizes_timeout_over_artifact_validation():
    response = EvalPromptResponse(
        text="truncated",
        duration_ms=600_000,
        timed_out=True,
        timeout_stage="encoder",
        timeout_reason="wall",
        timeout_seconds=600,
        timeout_attempts=1,
    )

    outcome = evals_module._eval_result_outcome(
        response,
        wrote_artifact=True,
        validation_error="Generated RuleSpec failed compile validation",
    )

    assert outcome == {
        "failure_kind": "timeout",
        "timed_out": True,
        "timeout_stage": "encoder",
        "timeout_reason": "wall",
        "timeout_seconds": 600,
        "timeout_attempts": 1,
    }


def test_timeout_then_plain_error_keeps_terminal_error_classification():
    initial = EvalPromptResponse(
        text="",
        duration_ms=600_000,
        trace={
            "timed_out": True,
            "timeout_stage": "encoder",
            "timeout_reason": "wall",
            "timeout_seconds": 600,
        },
        error="Codex eval timed out",
        timed_out=True,
        timeout_stage="encoder",
        timeout_reason="wall",
        timeout_seconds=600,
        timeout_attempts=1,
    )
    retry = EvalPromptResponse(
        text="",
        duration_ms=20,
        trace={"error": "authentication failed"},
        error="authentication failed",
    )

    combined = evals_module._combine_retry_response(initial, retry, "retry")
    outcome = evals_module._eval_result_outcome(
        combined,
        wrote_artifact=False,
        validation_error=None,
    )

    assert combined.timed_out is False
    assert outcome == {
        "failure_kind": "error",
        "timed_out": False,
        "timeout_stage": "encoder",
        "timeout_reason": "wall",
        "timeout_seconds": 600,
        "timeout_attempts": 1,
    }


def test_retry_timeout_history_before_http_error_is_not_terminal_timeout():
    initial = EvalPromptResponse(
        text="",
        duration_ms=10,
        trace={"response": "empty"},
    )
    retry = EvalPromptResponse(
        text="",
        duration_ms=20,
        trace={
            "error": "OpenAI eval HTTP 503",
            "timeout_attempts": 5,
            "timeout_stage": "encoder",
            "timeout_reason": "read",
            "timeout_seconds": 180,
        },
        error="OpenAI eval HTTP 503",
        timed_out=False,
        timeout_stage="encoder",
        timeout_reason="read",
        timeout_seconds=180,
        timeout_attempts=5,
    )

    combined = evals_module._combine_retry_response(initial, retry, "retry")
    outcome = evals_module._eval_result_outcome(
        combined,
        wrote_artifact=False,
        validation_error=None,
    )

    assert combined.timed_out is False
    assert outcome == {
        "failure_kind": "error",
        "timed_out": False,
        "timeout_stage": "encoder",
        "timeout_reason": "read",
        "timeout_seconds": 180,
        "timeout_attempts": 5,
    }


def test_retry_cost_aggregation_propagates_unknown_priced_usage():
    initial = EvalPromptResponse(
        text="",
        duration_ms=10,
        tokens=TokenUsage(input_tokens=272000, output_tokens=100),
        estimated_cost_usd=0.5452,
    )
    retry = EvalPromptResponse(
        text="",
        duration_ms=20,
        tokens=TokenUsage(input_tokens=272001, output_tokens=100),
        estimated_cost_usd=None,
    )

    combined = evals_module._combine_retry_response(initial, retry, "retry")

    assert combined.tokens is not None
    assert combined.tokens.input_tokens == 544001
    assert combined.estimated_cost_usd is None


def test_result_binding_rejects_failed_row_without_failure_kind():
    payload = _fake_eval_result("openai-gpt-5.4", "sample").to_dict()
    payload["success"] = False
    payload["error"] = "generation failed"
    payload["failure_kind"] = None

    with pytest.raises(ValueError, match="failed result without a failure_kind"):
        evals_module._validate_eval_result_artifact_binding(payload)


def test_policyengine_binding_rejects_artifact_without_oracle_evidence():
    case = EvalSuiteCase(
        kind="source",
        name="policyengine-case",
        mode="cold",
        corpus_citation_path="us/statute/7/2017",
        oracle="policyengine",
    )
    result = _fake_eval_result("openai-gpt-5.4", "us/statute/7/2017")
    result.success = False
    result.error = "oracle evidence was dropped"
    result.failure_kind = "error"
    result.metrics = None
    execution_identity = {
        "policyengine_runtime": {
            "identity": _TEST_POLICYENGINE_RUNTIME_IDENTITY,
            "sha256": _TEST_POLICYENGINE_RUNTIME_IDENTITY_SHA256,
        }
    }

    with pytest.raises(ValueError, match="PolicyEngine artifact.*oracle evidence"):
        evals_module._validate_eval_result_policyengine_binding(
            case,
            result,
            execution_identity,
        )


def test_policyengine_binding_rejects_success_with_failed_oracle():
    case = EvalSuiteCase(
        kind="source",
        name="policyengine-case",
        mode="cold",
        corpus_citation_path="us/statute/7/2017",
        oracle="policyengine",
    )
    result = _fake_eval_result(
        "openai-gpt-5.4",
        "us/statute/7/2017",
        policyengine_pass=False,
        policyengine_score=None,
    )
    execution_identity = {
        "policyengine_runtime": {
            "identity": _TEST_POLICYENGINE_RUNTIME_IDENTITY,
            "sha256": _TEST_POLICYENGINE_RUNTIME_IDENTITY_SHA256,
        }
    }

    with pytest.raises(ValueError, match="succeeded.*PolicyEngine.*pass"):
        evals_module._validate_eval_result_policyengine_binding(
            case,
            result,
            execution_identity,
        )


def test_result_rehydration_rejects_non_boolean_success():
    payload = _fake_eval_result("openai-gpt-5.4", "sample").to_dict()
    payload["success"] = "false"
    payload["failure_kind"] = None
    payload = _bind_eval_result_payload(payload)

    with pytest.raises(ValueError, match="success must be a boolean"):
        _eval_result_from_payload(payload)


def test_codex_prompt_timeouts_use_default_for_short_source(tmp_path):
    workspace = prepare_eval_workspace(
        citation="us/statute/7/2012",
        runner=parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="short source",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
        mode="cold",
        extra_context_paths=[],
    )

    assert _codex_prompt_timeouts(workspace) == (600, 300)


def test_claude_encoder_timeout_does_not_change_codex_timeouts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AXIOM_ENCODE_ENCODER_TIMEOUT_SECONDS", "1234")
    workspace = prepare_eval_workspace(
        citation="us/statute/7/2012",
        runner=parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="short source",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
        mode="cold",
        extra_context_paths=[],
    )

    assert _codex_prompt_timeouts(workspace) == (600, 300)


def test_codex_prompt_timeouts_use_env_for_short_source(tmp_path, monkeypatch):
    monkeypatch.setenv("AXIOM_ENCODE_CODEX_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("AXIOM_ENCODE_CODEX_IDLE_TIMEOUT_SECONDS", "30")
    workspace = prepare_eval_workspace(
        citation="us/statute/7/2012",
        runner=parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="short source",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
        mode="cold",
        extra_context_paths=[],
    )

    assert _codex_prompt_timeouts(workspace) == (90, 30)


def test_codex_prompt_timeouts_ignore_invalid_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AXIOM_ENCODE_CODEX_TIMEOUT_SECONDS", "not-a-number")
    monkeypatch.setenv("AXIOM_ENCODE_CODEX_IDLE_TIMEOUT_SECONDS", "0")
    workspace = prepare_eval_workspace(
        citation="us/statute/7/2012",
        runner=parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="short source",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
        mode="cold",
        extra_context_paths=[],
    )

    assert _codex_prompt_timeouts(workspace) == (600, 300)


def test_codex_prompt_timeouts_use_long_limits_for_large_source(tmp_path):
    workspace = prepare_eval_workspace(
        citation="us/statute/7/2014",
        runner=parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="x" * 40000,
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
        mode="cold",
        extra_context_paths=[],
    )

    assert _codex_prompt_timeouts(workspace) == (1800, 900)


def test_codex_prompt_timeouts_use_long_env_for_large_source(tmp_path, monkeypatch):
    monkeypatch.setenv("AXIOM_ENCODE_CODEX_LONG_TIMEOUT_SECONDS", "240")
    monkeypatch.setenv("AXIOM_ENCODE_CODEX_LONG_IDLE_TIMEOUT_SECONDS", "60")
    workspace = prepare_eval_workspace(
        citation="us/statute/7/2014",
        runner=parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="x" * 40000,
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
        mode="cold",
        extra_context_paths=[],
    )

    assert _codex_prompt_timeouts(workspace) == (240, 60)


def test_run_codex_prompt_eval_records_idle_timeout_threshold(tmp_path):
    runner = parse_runner_spec("codex:gpt-5.4")
    workspace = prepare_eval_workspace(
        citation="uksi/2002/1792/schedule/VI/paragraph/4A/1",
        runner=runner,
        output_root=tmp_path / "out",
        source_text="maximum disregard",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
        mode="cold",
        extra_context_paths=[],
    )
    bundle = "=== FILE: example.yaml ===\nformat: rulespec/v1\nrules: []\n"

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, cwd, stdin=None, env=None):
            self.args = cmd
            self.returncode = None
            Path(cwd, ".codex-last-message.txt").write_text(bundle)

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    with (
        patch("axiom_encode.harness.evals.subprocess.Popen", FakePopen),
        patch(
            "axiom_encode.harness.evals._wait_for_codex_process",
            side_effect=subprocess.TimeoutExpired(
                cmd=["codex", "exec"],
                timeout=300,
            ),
        ),
    ):
        response = _run_codex_prompt_eval(runner, workspace, "prompt")

    assert response.error is None
    assert response.text == bundle.strip()
    assert response.timeout_reason == "idle"
    assert response.timeout_seconds == 300
    assert response.trace["timeout_reason"] == "idle"
    assert response.trace["timeout_seconds"] == 300


def test_codex_idle_timeout_precedes_longer_case_budget(tmp_path, monkeypatch):
    runner = parse_runner_spec("codex:gpt-5.4")
    workspace = prepare_eval_workspace(
        citation="uksi/2002/1792/schedule/VI/paragraph/4A/1",
        runner=runner,
        output_root=tmp_path / "out",
        source_text="maximum disregard",
        axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
        mode="cold",
        extra_context_paths=[],
    )
    bundle = "=== FILE: example.yaml ===\nformat: rulespec/v1\nrules: []\n"
    monkeypatch.setattr(evals_module.time, "monotonic", lambda: 100.0)

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, cwd, stdin=None, env=None):
            self.args = cmd
            self.returncode = None
            Path(cwd, ".codex-last-message.txt").write_text(bundle)

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    deadline_token = evals_module._EVAL_CASE_DEADLINE_MONOTONIC.set(600.0)
    timeout_token = evals_module._EVAL_CASE_TIMEOUT_SECONDS.set(500)
    try:
        with (
            patch("axiom_encode.harness.evals.subprocess.Popen", FakePopen),
            patch(
                "axiom_encode.harness.evals._wait_for_codex_process",
                side_effect=subprocess.TimeoutExpired(
                    cmd=["codex", "exec"],
                    timeout=300,
                ),
            ),
        ):
            response = _run_codex_prompt_eval(runner, workspace, "prompt")
    finally:
        evals_module._EVAL_CASE_TIMEOUT_SECONDS.reset(timeout_token)
        evals_module._EVAL_CASE_DEADLINE_MONOTONIC.reset(deadline_token)

    assert response.timed_out is True
    assert response.timeout_stage == "encoder"
    assert response.timeout_reason == "idle"
    assert response.timeout_seconds == 300
    assert response.trace["timeout_stage"] == "encoder"
    assert response.trace["wall_timeout_seconds"] == 500
    assert response.trace["idle_timeout_seconds"] == 300


def test_equal_codex_wall_and_idle_limits_preserve_triggering_reason(tmp_path):
    class NeverCompletes:
        args = ["codex", "exec"]

        def poll(self):
            return None

    with (
        patch(
            "axiom_encode.harness.evals.time.monotonic",
            side_effect=[0.0, 601.0],
        ),
        pytest.raises(subprocess.TimeoutExpired) as exc_info,
    ):
        _wait_for_codex_process(
            NeverCompletes(),
            tmp_path / "last-message.txt",
            timeout=600,
            max_idle_seconds=600,
            poll_interval=0,
        )

    assert exc_info.value.timeout == 600
    assert exc_info.value.timeout_reason == "wall"


def test_wait_for_codex_process_rejects_completion_after_wall_timeout(
    tmp_path,
    monkeypatch,
):
    clock = [100.0]
    monkeypatch.setattr(evals_module.time, "monotonic", lambda: clock[0])

    class CompletesLate:
        args = ["codex", "exec"]

        def poll(self):
            clock[0] = 106.0
            return 0

    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        _wait_for_codex_process(
            CompletesLate(),
            tmp_path / "last-message.txt",
            timeout=5,
            poll_interval=0,
        )

    assert exc_info.value.timeout == 5
    assert exc_info.value.timeout_reason == "wall"


def test_run_source_eval_does_not_retry_when_first_response_writes_rulespec(tmp_path):
    policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path, "source states 451."
    )
    response = EvalPromptResponse(
        text=(
            "=== FILE: sample.yaml ===\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: source states 451.\n"
            "rules: []\n"
            "=== FILE: sample.test.yaml ===\n"
            "[]\n"
        ),
        duration_ms=10,
        trace={"attempt": "initial"},
    )

    with (
        patch(
            "axiom_encode.harness.evals._run_prompt_eval",
            return_value=response,
        ) as mock_prompt_eval,
        patch("axiom_encode.harness.evals.evaluate_artifact", return_value=None),
    ):
        [result] = run_source_eval(
            source_unit=source_unit,
            runner_specs=["codex:gpt-5.4"],
            output_root=tmp_path / "out",
            policy_path=policy_repo_root,
            local_corpus_release=corpus_release,
            runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
        )

    assert result.success is True
    assert result.retry_count == 0
    assert mock_prompt_eval.call_count == 1


def test_eval_result_payload_round_trips_prompt_digests():
    result = EvalResult(
        citation="snap_test",
        runner="codex-gpt-5.4",
        backend="codex",
        model="gpt-5.4",
        mode="repo-augmented",
        output_file="/tmp/snap_test.yaml",
        trace_file="/tmp/snap_test.trace.json",
        context_manifest_file="/tmp/snap_test.context.json",
        generated_output_sha256="a" * 64,
        trace_sha256="b" * 64,
        context_manifest_sha256="c" * 64,
        duration_ms=1234,
        success=True,
        error=None,
        input_tokens=11,
        output_tokens=22,
        cache_read_tokens=33,
        cache_creation_tokens=44,
        reasoning_output_tokens=55,
        estimated_cost_usd=0.12,
        actual_cost_usd=None,
        retrieved_files=["/tmp/context.yaml"],
        unexpected_accesses=[],
        retry_count=1,
        metrics=EvalArtifactMetrics(
            compile_pass=True,
            compile_issues=[],
            ci_pass=True,
            ci_issues=[],
            embedded_source_present=True,
            grounded_numeric_count=1,
            ungrounded_numeric_count=0,
            grounding=[],
            generalist_review_pass=True,
            generalist_review_score=9.0,
            generalist_review_issues=[],
            generalist_review_prompt_sha256="review-digest",
            policyengine_pass=True,
            policyengine_score=1.0,
            policyengine_issues=[],
            policyengine_runtime_identity=_TEST_POLICYENGINE_RUNTIME_IDENTITY,
            policyengine_runtime_identity_sha256=(
                _TEST_POLICYENGINE_RUNTIME_IDENTITY_SHA256
            ),
        ),
        generation_prompt_sha256="generation-digest",
        source_attestation={
            "requested_corpus_citation_path": "us/statute/7/2014/e/6/A",
            "source_sha256": "a" * 64,
        },
        require_complete_source_unit=True,
    )

    strict_payload = result.to_dict()
    restored = _eval_result_from_payload(strict_payload)

    assert strict_payload["require_complete_source_unit"] is True
    assert restored.require_complete_source_unit is True
    assert restored.generation_prompt_sha256 == "generation-digest"
    assert restored.retry_count == 1
    assert restored.metrics is not None
    assert restored.metrics.generalist_review_prompt_sha256 == "review-digest"
    assert restored.source_attestation == result.source_attestation
    result.require_complete_source_unit = False
    assert "require_complete_source_unit" not in result.to_dict()

    malformed_payload = dict(strict_payload)
    malformed_payload["require_complete_source_unit"] = "true"
    malformed_payload = _bind_eval_result_payload(malformed_payload)
    with pytest.raises(ValueError, match="invalid boolean field"):
        _eval_result_from_payload(malformed_payload)

    def test_wait_for_codex_process_terminates_after_persistent_output(self, tmp_path):
        last_message = tmp_path / ".codex-last-message.txt"
        last_message.write_text("ready\n")

        class FakeProcess:
            def __init__(self):
                self.args = ["codex", "exec"]
                self.returncode = None
                self.terminated = False

            def poll(self):
                if self.returncode is None:
                    last_message.touch()
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        process = FakeProcess()
        terminated = _wait_for_codex_process(
            process,
            last_message,
            timeout=1,
            settle_seconds=1,
            max_output_wait_seconds=0,
            poll_interval=0,
        )

        assert terminated is True
        assert process.terminated is True

    def test_wait_for_codex_process_times_out_when_heartbeat_stalls(self, tmp_path):
        last_message = tmp_path / ".codex-last-message.txt"
        stdout_path = tmp_path / "stdout.log"
        stdout_path.write_text("")

        class FakeProcess:
            def __init__(self):
                self.args = ["codex", "exec"]
                self.returncode = None
                self.terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        process = FakeProcess()
        with pytest.raises(subprocess.TimeoutExpired):
            _wait_for_codex_process(
                process,
                last_message,
                timeout=1,
                heartbeat_paths=[stdout_path],
                max_idle_seconds=0,
                poll_interval=0,
            )

        assert process.terminated is True

    def test_run_codex_prompt_eval_accepts_stable_last_message_on_termination(
        self, tmp_path
    ):
        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/6/3/a",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="nil amount",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        bundle = "=== FILE: example.yaml ===\nformat: rulespec/v1\nrules: []\n"
        event_lines = "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "fallback"},
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "cached_input_tokens": 0,
                        },
                    }
                ),
            ]
        )

        class FakePopen:
            def __init__(self, cmd, stdout, stderr, text, cwd, stdin=None, env=None):
                self.args = cmd
                self.returncode = None
                Path(cwd, ".codex-last-message.txt").write_text(bundle)
                stdout.write(event_lines + "\n")
                stdout.flush()

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        def fake_wait(
            process,
            last_message_file,
            timeout,
            heartbeat_paths=None,
            settle_seconds=5.0,
            max_output_wait_seconds=30.0,
            max_idle_seconds=120.0,
            poll_interval=0.5,
        ):
            process.terminate()
            process.wait()
            return True

        with (
            patch("axiom_encode.harness.evals.subprocess.Popen", FakePopen),
            patch(
                "axiom_encode.harness.evals._wait_for_codex_process",
                side_effect=fake_wait,
            ),
        ):
            response = _run_codex_prompt_eval(runner, workspace, "prompt")

        assert response.error is None
        assert response.text == bundle.strip()
        assert response.tokens is not None
        assert response.tokens.input_tokens == 10
        assert response.tokens.output_tokens == 4

    def test_run_codex_prompt_eval_salvages_last_message_on_timeout(self, tmp_path):
        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/schedule/VI/paragraph/4A/1",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="maximum disregard",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        bundle = "=== FILE: example.yaml ===\nformat: rulespec/v1\nrules: []\n"

        class FakePopen:
            def __init__(self, cmd, stdout, stderr, text, cwd, stdin=None, env=None):
                self.args = cmd
                self.returncode = None
                Path(cwd, ".codex-last-message.txt").write_text(bundle)

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        with (
            patch("axiom_encode.harness.evals.subprocess.Popen", FakePopen),
            patch(
                "axiom_encode.harness.evals._wait_for_codex_process",
                side_effect=subprocess.TimeoutExpired(
                    cmd=["codex", "exec"], timeout=600
                ),
            ),
        ):
            response = _run_codex_prompt_eval(runner, workspace, "prompt")

        assert response.error is None
        assert response.text == bundle.strip()

    def test_run_codex_prompt_eval_uses_longer_idle_timeout(self, tmp_path):
        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="10-ccr-2506-1/4.403.11/b/c/3",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="self-employment expenses",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        class FakePopen:
            def __init__(self, cmd, stdout, stderr, text, cwd, stdin=None, env=None):
                self.args = cmd
                self.returncode = 0

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        observed: dict[str, float] = {}

        def fake_wait(
            process,
            last_message_file,
            timeout,
            heartbeat_paths=None,
            settle_seconds=5.0,
            max_output_wait_seconds=30.0,
            max_idle_seconds=120.0,
            poll_interval=0.5,
        ):
            observed["max_idle_seconds"] = max_idle_seconds
            process.returncode = 0
            return False

        with (
            patch("axiom_encode.harness.evals.subprocess.Popen", FakePopen),
            patch(
                "axiom_encode.harness.evals._wait_for_codex_process",
                side_effect=fake_wait,
            ),
        ):
            response = _run_codex_prompt_eval(runner, workspace, "prompt")

        assert observed["max_idle_seconds"] == 300
        assert response.text == ""


class TestEvaluateArtifact:
    def test_reviewer_independent_metrics_are_byte_identical_across_staging_roots(
        self,
        tmp_path,
    ):
        policy_repo = _canonical_rulespec_content_root(tmp_path / "repos", "us")
        generated = _generated_rulespec_file_path(
            tmp_path / "out",
            "statutes/1/a.yaml",
        )
        generated.write_text("format: rulespec/v1\nrules: []\n")
        corpus_release = _write_test_corpus_provision(tmp_path / "bound-release")
        observed_staging_roots: list[Path] = []

        def fake_binary(pipeline):
            assert pipeline.validation_staging_root is not None
            observed_staging_roots.append(pipeline.validation_staging_root)
            return Path("/opt/axiom-rules-engine")

        def compile_timeout(**kwargs):
            raise subprocess.TimeoutExpired(
                [
                    str(kwargs["binary"]),
                    "compile",
                    "--program",
                    str(kwargs["program"]),
                    "--output",
                    str(kwargs["output"]),
                ],
                timeout=60,
            )

        with (
            patch.object(ValidatorPipeline, "_axiom_rules_binary", fake_binary),
            patch(
                "axiom_encode.harness.validator_pipeline.run_rulespec_compile",
                side_effect=compile_timeout,
            ),
        ):
            metrics_by_run = [
                evaluate_artifact(
                    local_corpus_release=corpus_release,
                    rulespec_file=generated,
                    policy_repo_root=policy_repo,
                    axiom_rules_path=tmp_path / "axiom-rules-engine",
                    source_text="No numeric values.",
                    skip_reviewers=True,
                )
                for _ in range(2)
            ]

        projections = [
            json.dumps(
                _reviewer_independent_metrics(metrics),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            for metrics in metrics_by_run
        ]
        unique_staging_roots = list(dict.fromkeys(observed_staging_roots))
        assert len(unique_staging_roots) == 2
        assert all(
            b"<rulespec-validation-temp>/compiled.json" in projection
            for projection in projections
        )
        assert projections[0] == projections[1]

    def test_validates_generated_artifact_inside_policy_repo_overlay(self, tmp_path):
        policy_repo = _canonical_rulespec_content_root(tmp_path / "repos", "us-ny")
        generated = (
            tmp_path
            / "out"
            / "openai-gpt-5.5"
            / "regulations"
            / "18-nycrr"
            / "387"
            / "12"
            / "f"
            / "3"
            / "v"
            / "c.yaml"
        )
        generated.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        generated.with_name("c.test.yaml").write_text("[]\n")
        seen_targets: list[tuple[tuple[str, ...], str, bool]] = []
        seen_policy_repo_roots: list[tuple[str, ...]] = []

        def fake_compile(_pipeline, path):
            seen_targets.append(
                (path.parts, path.name, path.with_name("c.test.yaml").exists())
            )
            seen_policy_repo_roots.append(_pipeline.policy_repo_path.parts)
            return ValidationResult("compile", passed=True)

        def fake_ci(_pipeline, path):
            seen_targets.append(
                (path.parts, path.name, path.with_name("c.test.yaml").exists())
            )
            seen_policy_repo_roots.append(_pipeline.policy_repo_path.parts)
            return ValidationResult("ci", passed=True)

        with (
            patch.object(ValidatorPipeline, "_run_compile_check", fake_compile),
            patch.object(ValidatorPipeline, "_run_ci", fake_ci),
        ):
            evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=generated,
                policy_repo_root=policy_repo,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                source_text="No numeric values.",
            )

        assert len(seen_targets) == 2
        for parts, name, companion_test_exists in seen_targets:
            assert "rulespec-us" in parts
            assert "us-ny" in parts
            assert name == "c.yaml"
            assert companion_test_exists
        for parts in seen_policy_repo_roots:
            assert "rulespec-us" in parts
            assert "us-ny" in parts
            assert parts != policy_repo.parts

    def test_normalizes_ephemeral_validation_root_in_compile_and_ci_issues(
        self, tmp_path
    ):
        policy_repo = _canonical_rulespec_content_root(tmp_path / "repos", "us")
        generated = _generated_rulespec_file_path(
            tmp_path / "out",
            "statutes/1/a.yaml",
        )
        generated.write_text("format: rulespec/v1\nrules: []\n")
        observed_paths: list[Path] = []

        def fake_compile(_pipeline, path):
            observed_paths.append(path)
            issue = f"Axiom rules engine compile failed: failed to load `{path}`"
            return ValidationResult(
                "compile",
                passed=False,
                issues=[issue],
                error=issue,
                raw_output=issue,
            )

        def fake_ci(_pipeline, path):
            observed_paths.append(path)
            issue = f"Axiom rules engine compile failed: failed to load `{path}`"
            return ValidationResult(
                "ci",
                passed=False,
                issues=[issue],
                error=issue,
                raw_output=issue,
            )

        with (
            patch.object(
                ValidatorPipeline,
                "_run_rulespec_compile_check",
                fake_compile,
            ),
            patch.object(ValidatorPipeline, "_run_rulespec_ci", fake_ci),
        ):
            corpus_release = _write_test_corpus_provision(tmp_path / "bound-release")
            metrics_by_run = [
                evaluate_artifact(
                    local_corpus_release=corpus_release,
                    rulespec_file=generated,
                    policy_repo_root=policy_repo,
                    axiom_rules_path=tmp_path / "axiom-rules-engine",
                    source_text="No numeric values.",
                    skip_reviewers=True,
                )
                for _ in range(2)
            ]

        stable_issue = (
            "Axiom rules engine compile failed: failed to load "
            "`<rulespec-validation-root>/rulespec-us/us/statutes/1/a.yaml`"
        )
        assert observed_paths[0] == observed_paths[1]
        assert observed_paths[2] == observed_paths[3]
        assert observed_paths[0] != observed_paths[2]
        for metrics in metrics_by_run:
            assert metrics.compile_issues == [stable_issue]
            assert metrics.ci_issues == [stable_issue]

    def test_validation_overlay_preserves_country_monorepo_state_shape(self, tmp_path):
        monorepo = tmp_path / "repos" / "rulespec-us"
        policy_repo = monorepo / "us-mn"
        policy_repo.mkdir(parents=True)
        generated = (
            tmp_path
            / "out"
            / "codex-gpt-5.5"
            / "policies"
            / "dhs"
            / "combined-manual"
            / "0020-21"
            / "msa-assistance-standards-2026.yaml"
        )
        generated.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        generated.with_name("msa-assistance-standards-2026.test.yaml").write_text(
            "[]\n"
        )

        with _rulespec_validation_target(generated, policy_repo) as validation_file:
            validation_root = _validation_policy_repo_root(validation_file, policy_repo)

            assert validation_file.parts[-6:] == (
                "us-mn",
                "policies",
                "dhs",
                "combined-manual",
                "0020-21",
                "msa-assistance-standards-2026.yaml",
            )
            assert validation_root.name == "us-mn"
            assert validation_root.parent.name == "rulespec-us"
            assert validation_file.with_name(
                "msa-assistance-standards-2026.test.yaml"
            ).exists()

    def test_validation_root_prefers_nearest_nested_canonical_root(self, tmp_path):
        outer_root = _canonical_rulespec_content_root(tmp_path / "outer", "us")
        nested_root = _canonical_rulespec_content_root(
            outer_root / "nested",
            "us",
        )
        validation_file = nested_root / "statutes" / "1" / "a.yaml"
        validation_file.parent.mkdir(parents=True)
        validation_file.write_text("format: rulespec/v1\nrules: []\n")

        assert _validation_policy_repo_root(validation_file, outer_root) == nested_root

    def test_validation_overlay_rejects_aliased_country_worktree(self, tmp_path):
        monorepo = tmp_path / "repos" / "rulespec-us-mn-msa-20260627"
        policy_repo = monorepo / "us-mn"
        policy_repo.mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=monorepo, check=True, capture_output=True)
        subprocess.run(
            [
                "git",
                "remote",
                "add",
                "origin",
                "git@github.com:TheAxiomFoundation/rulespec-us.git",
            ],
            cwd=monorepo,
            check=True,
            capture_output=True,
        )
        generated = (
            tmp_path
            / "out"
            / "codex-gpt-5.5"
            / "policies"
            / "dhs"
            / "combined-manual"
            / "0020-21"
            / "msa-assistance-standards-2026.yaml"
        )
        generated.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")

        with pytest.raises(
            UnsafeRulespecContextPath,
            match="canonical rulespec-<country>",
        ):
            with _rulespec_validation_target(generated, policy_repo):
                pass

    def test_passes_resolved_source_text_to_validation_pipeline(self, tmp_path):
        policy_repo = _canonical_rulespec_content_root(tmp_path / "repos", "us")
        generated = tmp_path / "out" / "codex-gpt-5.5" / "statutes" / "1" / "a.yaml"
        generated.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        source_text = "The official source states the standard is $1,055.00."
        source_metadata = {
            "source_attestation": {"requested_corpus_citation_path": "us/statute/1/a"}
        }
        seen_source_texts: list[str | None] = []
        seen_source_metadata: list[dict[str, object] | None] = []

        def fake_compile(_pipeline, _path):
            return ValidationResult("compile", passed=True)

        def fake_ci(_pipeline, _path):
            seen_source_texts.append(_pipeline.source_text)
            seen_source_metadata.append(_pipeline.source_metadata)
            return ValidationResult("ci", passed=True)

        with (
            patch.object(ValidatorPipeline, "_run_compile_check", fake_compile),
            patch.object(ValidatorPipeline, "_run_ci", fake_ci),
        ):
            evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=generated,
                policy_repo_root=policy_repo,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                source_text=source_text,
                source_metadata=source_metadata,
            )

        assert seen_source_texts == [source_text]
        assert seen_source_metadata == [source_metadata]

    def test_uses_fallback_source_text_for_grounding(self, tmp_path):
        rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/24/a.yaml")
        rulespec_file.write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: (a) Allowance of credit There shall be allowed a credit of $1,000.\n"
            "rules:\n"
            "  - name: ctc_amount\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: USD\n"
            "    versions:\n"
            "      - effective_from: '2018-01-01'\n"
            "        formula: 1000\n"
            "      - effective_from: '2025-01-01'\n"
            "        formula: 2200\n"
        )

        compile_result = ValidationResult("compile", passed=True)
        ci_result = ValidationResult("ci", passed=True)

        with (
            patch.object(
                PolicyEngineRuntime,
                "assert_matches_rulespec_root",
                return_value=None,
            ),
            patch(
                "axiom_encode.harness.validator_pipeline.ValidatorPipeline._run_compile_check",
                return_value=compile_result,
            ),
            patch(
                "axiom_encode.harness.validator_pipeline.ValidatorPipeline._run_ci",
                return_value=ci_result,
            ),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "(a) Allowance of credit There shall be allowed a credit of $1,000."
                ),
            )

        assert metrics.compile_pass
        assert not metrics.ci_pass
        assert metrics.grounded_numeric_count == 1
        assert metrics.ungrounded_numeric_count == 1
        assert [item.raw for item in metrics.grounding if not item.grounded] == ["2200"]
        assert any(
            "Ungrounded generated numeric literal" in issue and "2200" in issue
            for issue in metrics.ci_issues
        )

    def test_attached_amendment_value_stays_ungrounded_with_citation_hint(
        self,
        tmp_path,
    ):
        source_citation = "de/statute/solzg-1995/3"
        source_text = "Die konsolidierte Freigrenze beträgt 40 700 Euro."
        amendment_citation = (
            "de/statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-1"
        )
        amendment_source = (
            "aa) In Nummer 1 wird die Angabe „36 260 Euro“ durch die Angabe "
            "„39 900 Euro“ ersetzt."
        )
        rulespec_file = _generated_rulespec_file_path(
            tmp_path,
            "statutes/solzg-1995/3.yaml",
        )
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/solzg-1995/3
rules:
  - name: exemption_threshold
    kind: parameter
    dtype: Money
    unit: EUR
    versions:
      - effective_from: '2025-01-01'
        formula: 39900
""",
            encoding="utf-8",
        )
        amendment = CorpusAmendmentDocument(
            citation_path=amendment_citation,
            title="Steuerfortentwicklungsgesetz – SteFeG",
            expression_date="2024-12-23",
            metadata={},
            body=amendment_source,
        )

        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                return_value=ValidationResult("ci", passed=True),
            ),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release",
                    citation_path=source_citation,
                    body=source_text,
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "de"),
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                source_text=source_text,
                source_citation_path=source_citation,
                amendment_documents=(amendment,),
                skip_reviewers=True,
            )

        assert not metrics.ci_pass
        assert metrics.ungrounded_numeric_count == 1
        amendment_issues = [
            issue
            for issue in metrics.ci_issues
            if "Attached-amendment grounding hint" in issue
        ]
        assert len(amendment_issues) == 1
        assert amendment_citation in amendment_issues[0]

    def test_unresolved_bkgg_amendment_import_gets_exact_retry_steering(
        self,
        tmp_path,
    ):
        source_citation = "de/statute/bkgg/6a"
        amendment_citation = (
            "de/statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-1"
        )
        invalid_import = (
            "de:statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-1"
        )
        engine_error = (
            f"RuleSpec import `{invalid_import}` in `/work/de/statutes/bkgg/6a.yaml` "
            "could not be resolved"
        )
        compile_result = ValidationResult(
            "compile",
            passed=False,
            issues=[engine_error],
            error=engine_error,
        )
        rulespec_file = _generated_rulespec_file_path(
            tmp_path,
            "statutes/bkgg/6a.yaml",
        )
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/bkgg/6a
rules: []
""",
            encoding="utf-8",
        )
        amendment = CorpusAmendmentDocument(
            citation_path=amendment_citation,
            title="Steuerfortentwicklungsgesetz – SteFeG",
            expression_date="2024-12-23",
            metadata={},
            body="Artikel 3 ändert § 6a des Bundeskindergeldgesetzes.",
        )

        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=compile_result,
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                return_value=ValidationResult("ci", passed=True),
            ),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release",
                    citation_path=source_citation,
                    body="§ 6a Kinderzuschlag.",
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "de"),
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                source_text="§ 6a Kinderzuschlag.",
                source_citation_path=source_citation,
                amendment_documents=(amendment,),
                skip_reviewers=True,
            )

        assert metrics.compile_issues[0] == engine_error
        retry_issue = metrics.compile_issues[1]
        assert invalid_import in retry_issue
        assert amendment_citation in retry_issue
        assert "Amendment documents are proof-citation targets" in retry_issue
        assert "effective-dated `versions` in this target module" in retry_issue
        assert (
            f"metadata.proof.atoms[].source.corpus_citation_path: {amendment_citation}"
        ) in retry_issue
        assert compile_result.error is not None
        assert retry_issue in compile_result.error

    @pytest.mark.parametrize(
        "unresolved_import",
        [
            "de:statutes/bkgg/6#upstream_rule",
            ("de:statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-10"),
        ],
    )
    def test_unrelated_unresolved_import_does_not_get_amendment_steering(
        self,
        unresolved_import,
    ):
        amendment = CorpusAmendmentDocument(
            citation_path=(
                "de/statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-1"
            ),
            title="Steuerfortentwicklungsgesetz – SteFeG",
            expression_date="2024-12-23",
            metadata={},
            body="Amendment body.",
        )
        engine_error = (
            f"RuleSpec import `{unresolved_import}` in `/work/target.yaml` "
            "could not be resolved"
        )
        compile_result = ValidationResult(
            "compile",
            passed=False,
            issues=[engine_error],
            error=engine_error,
        )

        evals_module._add_attached_amendment_import_retry_guidance(
            compile_result,
            (amendment,),
        )

        assert compile_result.issues == [engine_error]
        assert compile_result.error == engine_error

    def test_evaluate_artifact_skips_reviewers_when_requested(self, tmp_path):
        rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/24/a.yaml")
        rulespec_file.write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: Source text says the amount is $1,000.\n"
            "rules:\n"
            "  - name: ctc_amount\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: USD\n"
            "    versions:\n"
            "      - effective_from: '2018-01-01'\n"
            "        formula: 1000\n"
        )

        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                return_value=ValidationResult("ci", passed=True),
            ),
            patch.object(ValidatorPipeline, "_run_reviewer") as mock_reviewer,
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="Source text says the amount is $1,000.",
                skip_reviewers=True,
            )

        mock_reviewer.assert_not_called()
        assert metrics.generalist_review_pass
        assert metrics.generalist_review_score is None
        assert metrics.generalist_review_issues == []

    def test_generated_eval_revalidation_keeps_attached_amendments(self, tmp_path):
        amendment = CorpusAmendmentDocument(
            citation_path=(
                "de/statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-1"
            ),
            title="Steuerfortentwicklungsgesetz – SteFeG",
            expression_date="2024-12-23",
            metadata={},
            body=(
                "In § 3 Absatz 3 Satz 1 wird die Angabe „36 260 Euro“ durch "
                "die Angabe „39 900 Euro“ ersetzt."
            ),
        )
        metrics = SimpleNamespace(ci_issues=["repairable"])
        protected_review_excerpts = frozenset({"Preserve this exact phrase"})
        with (
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
                return_value=metrics,
            ) as mock_evaluate,
            patch(
                "axiom_encode.harness.evals._apply_generated_eval_repairs",
                side_effect=[["companion-test-repair"], []],
            ) as mock_repair,
        ):
            result = _evaluate_generated_artifact_with_repairs(
                rulespec_file=tmp_path / "artifact.yaml",
                policy_repo_root=tmp_path / "rulespec-de",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                source_text="Source body",
                local_corpus_release=object(),
                require_complete_source_unit=True,
                amendment_documents=(amendment,),
                protected_review_excerpts=protected_review_excerpts,
            )

        assert result is metrics
        assert mock_evaluate.call_count == 2
        assert all(
            call.kwargs["amendment_documents"] == (amendment,)
            for call in mock_evaluate.call_args_list
        )
        assert all(
            call.kwargs["protected_review_excerpts"] == protected_review_excerpts
            for call in mock_repair.call_args_list
        )

    def test_generated_eval_can_disable_post_materialization_repairs(self, tmp_path):
        metrics = SimpleNamespace(ci_issues=["repairable"])
        with (
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
                return_value=metrics,
            ) as mock_evaluate,
            patch(
                "axiom_encode.harness.evals._apply_generated_eval_repairs"
            ) as mock_repair,
        ):
            result = _evaluate_generated_artifact_with_repairs(
                rulespec_file=tmp_path / "artifact.yaml",
                policy_repo_root=tmp_path / "rulespec-us",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                source_text="Source body",
                local_corpus_release=object(),
                allow_artifact_repairs=False,
            )

        assert result is metrics
        mock_evaluate.assert_called_once()
        mock_repair.assert_not_called()

    def test_generated_eval_reanchors_same_and_amendment_document_proofs(
        self, capsys, tmp_path
    ):
        current_citation = "de/statute/estg/32a"
        amendment_citation = (
            "de/statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-1"
        )
        current_source = (
            "5. von 277 826 Euro an:0,45 • x – 19 470,38.3Die Größe „y“ "
            "ist ein Zehntausendstel"
        )
        amendment_source = (
            "Sie beträgt ab dem Veranlagungszeitraum 2025 für zu versteuernde "
            "Einkommen 1. bis 12 096 Euro (Grundfreibetrag): 0;"
        )
        rulespec_file = tmp_path / "generated" / "statutes" / "estg" / "32a.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            yaml.safe_dump(
                {
                    "format": "rulespec/v1",
                    "rules": [
                        {
                            "name": "fifth_band_rate",
                            "kind": "parameter",
                            "metadata": {
                                "proof": {
                                    "atoms": [
                                        {
                                            "path": "versions[0].formula",
                                            "kind": "parameter",
                                            "source": {
                                                "corpus_citation_path": (
                                                    current_citation
                                                ),
                                                "excerpt": (
                                                    "5. von 277 826 Euro an: 0,45 "
                                                    "* x - 19 470,38. 3 Die Grosse "
                                                    '"y" ist ein Zehntausendstel'
                                                ),
                                            },
                                        }
                                    ]
                                }
                            },
                            "versions": [
                                {
                                    "effective_from": "2026-01-01",
                                    "formula": 0.45,
                                }
                            ],
                        },
                        {
                            "name": "basic_allowance_upper_income",
                            "kind": "parameter",
                            "metadata": {
                                "proof": {
                                    "atoms": [
                                        {
                                            "path": "versions[0].formula",
                                            "kind": "parameter",
                                            "source": {
                                                "corpus_citation_path": (
                                                    amendment_citation
                                                ),
                                                "excerpt": (
                                                    "Sie beträgt ab dem "
                                                    "Veranlagungszeitraum 2025 für "
                                                    "zu versteuernde Einkommen 1. "
                                                    "bis 12 096 Euro "
                                                    "(Grundfreibetrag):0;"
                                                ),
                                            },
                                        }
                                    ]
                                }
                            },
                            "versions": [
                                {
                                    "effective_from": "2025-01-01",
                                    "formula": 12096,
                                }
                            ],
                        },
                    ],
                },
                sort_keys=False,
                allow_unicode=True,
            )
        )
        fifth_band_issue = (
            "Proof source evidence not found: rule `fifth_band_rate` proof "
            "atom 0 `source.excerpt` does not appear in "
            f"`{current_citation}`."
        )
        basic_allowance_issue = (
            "Proof source evidence not found: rule "
            "`basic_allowance_upper_income` proof atom 0 `source.excerpt` "
            f"does not appear in `{amendment_citation}`."
        )
        before = SimpleNamespace(ci_issues=[fifth_band_issue])
        after_first_repair = SimpleNamespace(ci_issues=[basic_allowance_issue])
        after = SimpleNamespace(ci_issues=[], ci_pass=True)
        release = object()
        requested_paths = []

        def source_for_citation(citation_path, *, corpus_release):
            requested_paths.append((citation_path, corpus_release))
            return {
                current_citation: current_source,
                amendment_citation: amendment_source,
            }[citation_path]

        with (
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
                side_effect=[before, after_first_repair, after],
            ) as mock_evaluate,
            patch(
                "axiom_encode.cli._local_source_text_for_corpus_path",
                side_effect=source_for_citation,
            ),
        ):
            result = _evaluate_generated_artifact_with_repairs(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path / "rulespec-de",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                source_text=current_source,
                local_corpus_release=release,
                skip_reviewers=True,
            )

        repaired = yaml.safe_load(rulespec_file.read_text())
        repaired_sources = [
            rule["metadata"]["proof"]["atoms"][0]["source"]
            for rule in repaired["rules"]
        ]
        assert result is after
        assert mock_evaluate.call_count == 3
        assert requested_paths == [
            (current_citation, release),
            (amendment_citation, release),
        ]
        assert repaired_sources[0]["excerpt"] == current_source
        assert repaired_sources[1]["excerpt"] == amendment_source
        output = capsys.readouterr().out
        assert "auto_reanchored_proof_excerpts:fifth_band_rate[0]" in output
        assert (
            "auto_reanchored_proof_excerpts:basic_allowance_upper_income[0]" in output
        )

    def test_generated_eval_logs_failed_closed_reanchor_attempt(self, capsys, tmp_path):
        citation = "de/statute/estg/32a"
        rulespec_file = tmp_path / "generated" / "statutes" / "estg" / "32a.yaml"
        rulespec_file.parent.mkdir(parents=True)
        original = """format: rulespec/v1
rules:
- name: fifth_band_rate
  kind: parameter
  metadata:
    proof:
      atoms:
      - path: versions[0].formula
        kind: parameter
        source:
          corpus_citation_path: de/statute/estg/32a
          excerpt: The reduction rate is 5 percent.
  versions:
  - effective_from: '2026-01-01'
    formula: 0.05
"""
        rulespec_file.write_text(original)
        issue = (
            "Proof source evidence not found: rule `fifth_band_rate` proof "
            "atom 0 `source.excerpt` does not appear in "
            f"`{citation}`."
        )
        metrics = SimpleNamespace(ci_issues=[issue])

        with (
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
                return_value=metrics,
            ) as mock_evaluate,
            patch(
                "axiom_encode.cli._local_source_text_for_corpus_path",
                return_value="The annual amount is 277 826 Euro.",
            ),
        ):
            result = _evaluate_generated_artifact_with_repairs(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path / "rulespec-de",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                source_text="The annual amount is 277 826 Euro.",
                local_corpus_release=object(),
                skip_reviewers=True,
            )

        assert result is metrics
        mock_evaluate.assert_called_once()
        assert rulespec_file.read_text() == original
        assert (
            "auto_reanchored_proof_excerpts:"
            "none;attempted=fifth_band_rate[0]" in capsys.readouterr().out
        )

    def test_generated_eval_repairs_unreferenced_proof_imports(self, tmp_path):
        rulespec_file = tmp_path / "regulations" / "example.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
imports:
  - us-co:regulations/example#deadline
rules:
  - name: result_rule
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              excerpt: missed deadline
          - path: versions[0].formula
            kind: import
            import:
              target: us-co:regulations/example#deadline
              output: deadline
              hash: sha256:local
    versions:
      - effective_from: '2025-01-01'
        formula: missed_deadline
"""
        )

        ci_issue = (
            "Proof import not referenced: `result_rule` proof imports `deadline`, "
            "but the rule formula does not reference that imported symbol."
        )
        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                side_effect=[
                    ValidationResult("ci", passed=False, issues=[ci_issue]),
                    ValidationResult("ci", passed=True, issues=[]),
                ],
            ) as mock_ci,
        ):
            metrics = _evaluate_generated_artifact_with_repairs(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="The office missed the deadline.",
                skip_reviewers=True,
            )

        repaired_text = rulespec_file.read_text()
        assert mock_ci.call_count == 2
        assert metrics.ci_pass
        assert "kind: import" not in repaired_text
        assert "output: deadline" not in repaired_text

    def test_generated_eval_repairs_unused_imports(self, tmp_path):
        repo = _canonical_rulespec_content_root(tmp_path, "us")
        rulespec_file = repo / "statutes" / "26" / "example.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/source#unused_rate
  - us:statutes/26/source#used_rate
rules:
  - name: copied_rate
    kind: derived
    dtype: Rate
    period: Year
    versions:
      - effective_from: '2025-01-01'
        formula: used_rate
"""
        )

        ci_issue = (
            "Unused import `us:statutes/26/source#unused_rate`: imported symbol "
            "`unused_rate` is not referenced by any formula or proof import."
        )
        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                side_effect=[
                    ValidationResult("ci", passed=False, issues=[ci_issue]),
                    ValidationResult("ci", passed=True, issues=[]),
                ],
            ) as mock_ci,
        ):
            metrics = _evaluate_generated_artifact_with_repairs(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=repo,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="The copied rate uses the used rate.",
                skip_reviewers=True,
            )

        repaired_text = rulespec_file.read_text()
        assert mock_ci.call_count == 2
        assert metrics.ci_pass
        assert "unused_rate" not in repaired_text
        assert "used_rate" in repaired_text

    def test_generated_eval_repairs_positive_judgment_companions(self, tmp_path):
        repo = _canonical_rulespec_content_root(tmp_path, "us-co")
        dependency_content_root = _canonical_rulespec_content_root(tmp_path, "uk")
        dependency_marker = dependency_content_root / "statutes/1/dependency.yaml"
        dependency_marker.parent.mkdir(parents=True)
        dependency_marker.write_text("format: rulespec/v1\nrules: []\n")
        dependency_root = dependency_content_root.parent
        relative_output = Path("regulations/example.yaml")
        rulespec_file = tmp_path / "generated" / "openai" / relative_output
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
rules:
  - name: work_study_exemption
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-01-01'
        formula: enrolled_in_work_study
"""
        )
        rulespec_file.with_name("example.test.yaml").write_text(
            """- name: existing_negative
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us-co:regulations/example#input.enrolled_in_work_study: false
  output:
    us-co:regulations/example#work_study_exemption: not_holds
"""
        )
        ci_issue = (
            "Judgment rule missing positive companion output coverage: "
            "`us-co:regulations/example#work_study_exemption` is not asserted "
            "as `holds` by the companion `.test.yaml` file."
        )
        checked: dict[str, Path] = {}

        def check_companion(
            staged_test_file,
            *,
            root,
            axiom_rules_path,
            rulespec_dependency_roots=(),
        ):
            staged_rules_file = staged_test_file.with_name("example.yaml")
            checked["rules"] = validator_pipeline._canonical_rulespec_compile_path(
                staged_rules_file,
                root,
            )
            checked["test"] = staged_test_file.resolve()
            checked["root"] = root.resolve()
            [staged_dependency_root] = rulespec_dependency_roots
            checked["dependency"] = staged_dependency_root.resolve()
            assert (staged_dependency_root / "uk/statutes/1/dependency.yaml").is_file()
            return []

        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                side_effect=[
                    ValidationResult("ci", passed=False, issues=[ci_issue]),
                    ValidationResult("ci", passed=True, issues=[]),
                ],
            ) as mock_ci,
            patch(
                "axiom_encode.cli._rulespec_companion_test_failures",
                side_effect=check_companion,
            ),
        ):
            metrics = _evaluate_generated_artifact_with_repairs(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=repo,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="Students in work study are exempt.",
                skip_reviewers=True,
                rulespec_dependency_roots=(dependency_root,),
            )

        repaired_tests = yaml.safe_load(
            rulespec_file.with_name("example.test.yaml").read_text()
        )
        assert mock_ci.call_count == 2
        assert metrics.ci_pass
        assert checked["rules"].is_relative_to(checked["root"])
        assert checked["test"].is_relative_to(checked["root"])
        assert checked["root"] != repo.resolve()
        assert checked["dependency"] != dependency_root.resolve()
        assert checked["dependency"].parent == checked["root"].parent.parent
        assert not (repo / relative_output).exists()
        assert any(
            case.get("output", {}).get("us-co:regulations/example#work_study_exemption")
            == "holds"
            for case in repaired_tests
        )

    def test_generated_eval_repairs_companions_with_unrelated_issues(self, tmp_path):
        repo = _canonical_rulespec_content_root(tmp_path, "us-co")
        rulespec_file = repo / "regulations" / "example.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
rules:
  - name: work_study_exemption
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-01-01'
        formula: enrolled_in_work_study
"""
        )
        rulespec_file.with_name("example.test.yaml").write_text(
            """- name: existing_negative
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us-co:regulations/example#input.enrolled_in_work_study: false
  output:
    us-co:regulations/example#work_study_exemption: not_holds
"""
        )
        companion_issue = (
            "Judgment rule missing positive companion output coverage: "
            "`us-co:regulations/example#work_study_exemption` is not asserted "
            "as `holds` by the companion `.test.yaml` file."
        )
        unrelated_issue = (
            "Source scope mismatch: `work_study_exemption` is declared on "
            "`Person`, but the embedded source states a household/unit-scoped test."
        )
        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                side_effect=[
                    ValidationResult(
                        "ci",
                        passed=False,
                        issues=[companion_issue, unrelated_issue],
                    ),
                    ValidationResult("ci", passed=False, issues=[unrelated_issue]),
                ],
            ) as mock_ci,
            patch(
                "axiom_encode.cli._rulespec_companion_test_failures",
                return_value=[],
            ),
        ):
            metrics = _evaluate_generated_artifact_with_repairs(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=repo,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="Students in work study are exempt.",
                skip_reviewers=True,
            )

        repaired_tests = yaml.safe_load(
            rulespec_file.with_name("example.test.yaml").read_text()
        )
        assert mock_ci.call_count == 2
        assert not metrics.ci_pass
        assert metrics.ci_issues == [unrelated_issue]
        assert any(
            case.get("output", {}).get("us-co:regulations/example#work_study_exemption")
            == "holds"
            for case in repaired_tests
        )

    def test_generated_eval_repairs_scalar_relation_rows(self, tmp_path):
        repo = _canonical_rulespec_content_root(tmp_path, "us-co")
        rulespec_file = repo / "regulations" / "example.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/7/2012/j#relation.member_of_household
rules:
  - name: household_has_elderly_or_disabled_member
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: count_where(member_of_household, member_of_household.snap_member_is_elderly_or_disabled) > 0
"""
        )
        rulespec_file.with_name("example.test.yaml").write_text(
            """- name: elderly_case
  period: 2026-01
  input:
    us:statutes/7/2012/j#relation.member_of_household:
      - true
  output:
    us-co:regulations/example#household_has_elderly_or_disabled_member: holds
"""
        )
        relation_issue = (
            "Test case `elderly_case` input invalid: relation "
            "`us:statutes/7/2012/j#relation.member_of_household` item #1 "
            "must be a mapping"
        )
        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                side_effect=[
                    ValidationResult("ci", passed=False, issues=[relation_issue]),
                    ValidationResult("ci", passed=True, issues=[]),
                ],
            ) as mock_ci,
        ):
            metrics = _evaluate_generated_artifact_with_repairs(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=repo,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="An elderly or disabled member qualifies the household.",
                skip_reviewers=True,
            )

        repaired_tests = yaml.safe_load(
            rulespec_file.with_name("example.test.yaml").read_text()
        )
        rows = repaired_tests[0]["input"][
            "us:statutes/7/2012/j#relation.member_of_household"
        ]
        assert mock_ci.call_count == 2
        assert metrics.ci_pass
        assert rows == [
            {"us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled": True}
        ]

    def test_generated_eval_repairs_zero_branch_companions(self, tmp_path):
        repo = _canonical_rulespec_content_root(tmp_path, "us-co")
        rulespec_file = repo / "regulations" / "example.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
rules:
  - name: benefit_limit
    kind: parameter
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-01-01'
        formula: 100
  - name: benefit_amount
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-01-01'
        formula: 'if household_eligible: benefit_limit else: 0'
"""
        )
        rulespec_file.with_name("example.test.yaml").write_text(
            """- name: positive
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us-co:regulations/example#input.household_eligible: true
  output:
    us-co:regulations/example#benefit_amount: 100
"""
        )
        ci_issue = (
            "Zero branch test coverage missing: `benefit_amount` has a formula "
            "branch that returns 0, but no companion test asserts that output "
            "is zero."
        )
        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                side_effect=[
                    ValidationResult("ci", passed=False, issues=[ci_issue]),
                    ValidationResult("ci", passed=True, issues=[]),
                ],
            ) as mock_ci,
        ):
            metrics = _evaluate_generated_artifact_with_repairs(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=repo,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="Eligible households receive a $100 benefit limit; otherwise zero.",
                skip_reviewers=True,
            )

        repaired_tests = yaml.safe_load(
            rulespec_file.with_name("example.test.yaml").read_text()
        )
        assert mock_ci.call_count == 2
        assert metrics.ci_pass
        assert any(
            case.get("output", {}).get("us-co:regulations/example#benefit_amount") == 0
            for case in repaired_tests
        )

    def test_test_input_assignment_ignores_formula_builtins(self):
        content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: deadline_days
    kind: parameter
    dtype: Integer
    versions:
      - effective_from: '2025-01-01'
        formula: 7
  - name: deadline
    kind: derived
    entity: Household
    dtype: Date
    period: Month
    versions:
      - effective_from: '2025-01-01'
        formula: date_add_days(application_date, deadline_days)
  - name: period_span_days
    kind: derived
    entity: Household
    dtype: Integer
    period: Month
    versions:
      - effective_from: '2025-01-01'
        formula: days_between(period_start, period_end)
"""
        test_cases = [
            {
                "name": "deadline case",
                "input": {"#input.application_date": "2026-01-01"},
                "output": {
                    "#deadline": "2026-01-08",
                    "#period_span_days": 30,
                },
            }
        ]

        assert find_test_input_assignment_issues(content, test_cases) == []

    def test_numeric_occurrence_check_uses_embedded_operating_excerpt(self, tmp_path):
        source_text = (
            "(a) Households in which each member receives qualifying public "
            "assistance shall be eligible.\n\n"
            "(e) The unrelated standard deduction is 8.31 percent, $144, and $246."
        )
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="us/statute/7/2014",
            body=source_text,
        )
        rulespec_file = tmp_path / "statutes" / "7" / "2014" / "a.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2014
  summary: |-
    (a) Households in which each member receives qualifying public assistance shall be eligible.
rules:
  - name: snap_public_assistance_categorical_eligibility
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2008-10-01'
        formula: each_member_receives_qualifying_public_assistance
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=source_text,
                local_corpus_release=corpus_release,
            )

        assert metrics.ci_pass
        assert metrics.source_numeric_occurrence_count == 0
        assert metrics.numeric_occurrence_issues == []

    @pytest.mark.parametrize("pass_source_citation_path", [False, True])
    def test_complete_mode_numeric_recall_uses_authoritative_body(
        self,
        tmp_path,
        pass_source_citation_path,
    ):
        authoritative_source_text = (
            "(1) Der Freibetrag beträgt 259 Euro; der Zuschlag beträgt 73 Euro."
        )
        caller_source_summary = "(1) Der Freibetrag beträgt 259 Euro."
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="de/statute/estg/32a",
            body=authoritative_source_text,
        )
        rulespec_file = tmp_path / "statutes" / "estg" / "32a.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
  summary: Der Freibetrag beträgt 259 Euro.
rules:
  - name: allowance_amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 259
"""
        )
        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "de"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=caller_source_summary,
                local_corpus_release=corpus_release,
                source_citation_path=(
                    "de/statute/estg/32a" if pass_source_citation_path else None
                ),
                require_complete_source_unit=True,
            )

        assert not metrics.ci_pass
        assert metrics.source_numeric_occurrence_count == 2
        assert metrics.covered_source_numeric_occurrence_count == 1
        assert metrics.missing_source_numeric_occurrence_count == 1
        assert any("73" in issue for issue in metrics.numeric_occurrence_issues)

    def test_complete_mode_typed_recall_excludes_stage_labels_but_demands_one_euro(
        self,
        tmp_path,
    ):
        citation_path = "de/statute/rbeg-2021/8"
        source_text = (
            "1. in der Regelbedarfsstufe 1 auf 446 Euro für jede erwachsene "
            "Person. Ein Eigenanteil von 1 Euro wird verlangt."
        )
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path=citation_path,
            body=source_text,
        )
        rulespec_file = tmp_path / "statutes/rbeg-2021/8.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/rbeg-2021/8
rules:
  - name: regelbedarfsstufe_one_amount
    kind: parameter
    dtype: Money
    unit: EUR
    versions:
      - effective_from: '2021-01-01'
        formula: 446
""",
            encoding="utf-8",
        )

        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                return_value=ValidationResult("ci", passed=True),
            ),
            patch(
                "axiom_encode.harness.evals._numeric_occurrence_source_text",
                side_effect=AssertionError(
                    "complete-mode recall must use the typed raw-source path"
                ),
            ),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "de"),
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                source_text=source_text,
                local_corpus_release=corpus_release,
                source_citation_path=citation_path,
                require_complete_source_unit=True,
                skip_reviewers=True,
            )

        assert metrics.source_numeric_occurrence_count == 2
        assert metrics.covered_source_numeric_occurrence_count == 1
        assert metrics.missing_source_numeric_occurrence_count == 1
        assert metrics.numeric_occurrence_issues == [
            "Source numeric value 1 appears 1 time(s), but only 0 named scalar "
            "definition(s) with that value were found."
        ]

    def test_complete_mode_numeric_recall_is_summary_invariant(self, tmp_path):
        source_text = "If the 3rd digit is 5 or more, increase the 2nd digit by 1."
        citation_path = "ca/policy/cra/example/rounding"
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path=citation_path,
            body=source_text,
        )
        rulespec_file = tmp_path / "policies" / "cra" / "example" / "rounding.yaml"
        rulespec_file.parent.mkdir(parents=True)
        signatures = []

        for summary in (
            source_text,
            "A deliberately terse summary with no numeric inventory.",
        ):
            rulespec_file.write_text(
                f"""format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation_path}
  summary: {summary}
rules:
  - name: rounding_half_unit
    kind: parameter
    dtype: Decimal
    versions:
      - effective_from: '2026-01-01'
        formula: 0.5
"""
            )
            with (
                patch.object(
                    ValidatorPipeline,
                    "_run_compile_check",
                    return_value=ValidationResult("compile", True, issues=[]),
                ),
                patch.object(
                    ValidatorPipeline,
                    "_run_ci",
                    return_value=ValidationResult("ci", True, issues=[]),
                ),
            ):
                metrics = evaluate_artifact(
                    rulespec_file=rulespec_file,
                    policy_repo_root=_canonical_rulespec_content_root(
                        tmp_path,
                        "ca",
                    ),
                    axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                    source_text=source_text,
                    local_corpus_release=corpus_release,
                    source_citation_path=citation_path,
                    require_complete_source_unit=True,
                )
            signatures.append(
                (
                    metrics.source_numeric_occurrence_count,
                    metrics.covered_source_numeric_occurrence_count,
                    metrics.missing_source_numeric_occurrence_count,
                    metrics.numeric_occurrence_issues,
                )
            )

        assert signatures[0] == signatures[1]

    def test_numeric_occurrence_check_counts_inline_source_table_bounds(self, tmp_path):
        rulespec_file = tmp_path / "statutes" / "26" / "3241" / "b.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    Tax rate schedule | Average account benefits ratio | Applicable percentage
    | At least | But less than | Section 3201(b) |
    | .............. | 2.5 | 4.9 |
    | 2.5 | 3.0 | 4.9 |
rules:
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if average_account_benefits_ratio < 2.5: 1
          else: if average_account_benefits_ratio < 3.0: 2
          else: 3
  - name: section_3201_applicable_percentage_by_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_band
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 0.049
          2: 0.049
          3: 0
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "Tax rate schedule | Average account benefits ratio | Applicable percentage\n"
                    "| At least | But less than | Section 3201(b) |\n"
                    "| .............. | 2.5 | 4.9 |\n"
                    "| 2.5 | 3.0 | 4.9 |"
                ),
            )

        assert metrics.ci_pass
        assert metrics.missing_source_numeric_occurrence_count == 0
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_does_not_require_digit_scale_components(
        self, tmp_path
    ):
        rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/example.yaml")
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: The maximum amount is 10 million Euros.
rules:
  - name: maximum_fixed_penalty_amount
    kind: parameter
    dtype: Money
    unit: EUR
    versions:
      - effective_from: '2026-01-01'
        formula: 10000000
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="The maximum amount is 10 million Euros.",
            )

        assert metrics.ci_pass
        assert metrics.source_numeric_occurrence_count == 1
        assert metrics.missing_source_numeric_occurrence_count == 0
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_does_not_require_mixed_fraction_components(
        self, tmp_path
    ):
        rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/example.yaml")
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: Amount B is 2 6/7 per cent of the difference.
rules:
  - name: daily_excess_income_taper_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        formula: 0.02857142857142857
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="Amount B is 2 6/7 per cent of the difference.",
            )

        assert metrics.ci_pass
        assert metrics.source_numeric_occurrence_count == 1
        assert metrics.missing_source_numeric_occurrence_count == 0
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_skips_empty_deferred_artifact(self, tmp_path):
        source_text = (
            "The department shall establish a program under 7 U.S.C. Sec. "
            "2014(a). Categorical eligibility applies to households "
            "receiving or eligible to receive cash assistance under Part "
            "5 (commencing with Section 17000), or food assistance under "
            "Chapter 10.1 (commencing with Section 18930)."
        )
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-ca/statute/wic/18901.5",
            body=source_text,
        )
        rulespec_file = tmp_path / "statutes" / "wic" / "18901" / "5.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  status: deferred
  source_verification:
    corpus_citation_path: us-ca/statute/wic/18901.5
  summary: |-
    The department shall establish a program under 7 U.S.C. Sec. 2014(a). Categorical eligibility applies to households receiving or eligible to receive cash assistance under Part 5 (commencing with Section 17000), or food assistance under Chapter 10.1 (commencing with Section 18930).
  deferred_outputs:
    - output: us-ca:statutes/wic/18901/5#individual_categorically_eligible_for_calfresh
      reason: Requires upstream rules under Part 5 commencing with Section 17000 and Chapter 10.1 commencing with Section 18930, but no exact RuleSpec outputs were available in context.
rules: []
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us-ca"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=source_text,
                local_corpus_release=corpus_release,
            )

        assert metrics.ci_pass
        assert metrics.source_numeric_occurrence_count == 0
        assert metrics.numeric_occurrence_issues == []

    @pytest.mark.parametrize("require_complete_source_unit", [False, True])
    def test_generated_numeric_grounding_uses_authoritative_module_source(
        self,
        tmp_path,
        require_complete_source_unit,
    ):
        source_text = (
            "(a) Households in which each member receives qualifying public "
            "assistance shall be eligible.\n\n"
            "(e) The unrelated standard deduction is $144."
        )
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="us/statute/7/2014",
            body=source_text,
        )
        rulespec_file = tmp_path / "statutes" / "7" / "2014" / "a.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2014
  summary: |-
    (a) Households in which each member receives qualifying public assistance shall be eligible.
rules:
  - name: unrelated_standard_deduction_amount
    kind: parameter
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: 144
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=source_text,
                local_corpus_release=corpus_release,
                require_complete_source_unit=require_complete_source_unit,
            )

        assert metrics.ci_pass
        assert metrics.grounded_numeric_count == 1
        assert metrics.ungrounded_numeric_count == 0
        if require_complete_source_unit:
            assert metrics.source_numeric_occurrence_count == 1
            assert metrics.missing_source_numeric_occurrence_count == 0

    def test_generated_numeric_grounding_never_uses_module_summary(self, tmp_path):
        source_text = (
            "(a) Households in which each member receives qualifying public "
            "assistance shall be eligible."
        )
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="us/statute/7/2014",
            body=source_text,
        )
        rulespec_file = tmp_path / "statutes" / "7" / "2014" / "a.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2014
  summary: The unrelated standard deduction is $144.
rules:
  - name: unrelated_standard_deduction_amount
    kind: parameter
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: 144
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=compile_result,
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=source_text,
                local_corpus_release=corpus_release,
            )

        assert not metrics.ci_pass
        assert metrics.ungrounded_numeric_count == 1
        assert any("144" in issue for issue in metrics.ci_issues)

    def test_numeric_grounding_uses_de_profile_from_source_citation(self, tmp_path):
        source_text = "Der Betrag beläuft sich auf 1 034,87 Punkte."
        citation_path = "de/statute/estg/32a"
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path=citation_path,
            body=source_text,
        )
        rulespec_file = tmp_path / "statutes" / "estg" / "32a.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            f"""format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation_path}
rules:
  - name: german_amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 1034.87
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=compile_result,
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "de"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=source_text,
                source_citation_path=citation_path,
                local_corpus_release=corpus_release,
            )

        assert metrics.ci_pass
        assert metrics.grounded_numeric_count == 1
        assert metrics.ungrounded_numeric_count == 0
        assert metrics.source_numeric_occurrence_count == 1
        assert metrics.missing_source_numeric_occurrence_count == 0

    def test_numeric_grounding_uses_citation_only_de_proof_source(self, tmp_path):
        module_citation = "de/statute/example/1"
        proof_citation = "de/statute/example/2"
        module_source = "Der Haupttext enthält keinen maßgeblichen Betrag."
        proof_source = "Der maßgebliche Betrag ist 1 034,87 Punkte."
        corpus_release = _write_test_corpus_release(
            tmp_path,
            [
                {"citation_path": module_citation, "body": module_source},
                {"citation_path": proof_citation, "body": proof_source},
            ],
        )
        rulespec_file = tmp_path / "statutes" / "example" / "1.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            f"""format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {module_citation}
rules:
  - name: german_amount
    kind: parameter
    dtype: Money
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: {proof_citation}
    versions:
      - effective_from: '2026-01-01'
        formula: 1034.87
"""
        )

        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", True, issues=[]),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                return_value=ValidationResult("ci", True, issues=[]),
            ),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "de"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=module_source,
                source_citation_path=module_citation,
                local_corpus_release=corpus_release,
                skip_reviewers=True,
            )

        assert metrics.ci_pass
        assert metrics.grounded_numeric_count == 1
        assert metrics.ungrounded_numeric_count == 0

    def test_generated_numeric_grounding_uses_proof_excerpts_with_compact_summary(
        self, tmp_path
    ):
        source_text = "A different paragraph contains $144."
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="us/statute/26/3121",
            body=source_text,
        )
        rulespec_file = tmp_path / "statutes" / "26" / "3121" / "w.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/3121
  summary: |-
    Church election timing rule.
rules:
  - name: election_timing_days_after_enactment_threshold
    kind: parameter
    dtype: Count
    period: Day
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: parameter
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: more than 90 days after July 18, 1984
    versions:
      - effective_from: '1990-01-01'
        formula: 90
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=source_text,
                local_corpus_release=corpus_release,
            )

        assert metrics.ci_pass
        assert metrics.grounded_numeric_count == 1
        assert metrics.ungrounded_numeric_count == 0
        assert metrics.source_numeric_occurrence_count == 0

    def test_rounding_occurrence_cleanup_does_not_cross_source_clause(self):
        source_text = (
            "Increase the allowance by 1 dollar. Round up the second digit if "
            "the third digit is 5 or more."
        )

        cleaned = _numeric_occurrence_source_text(source_text)

        assert validator_pipeline.extract_numeric_occurrences_from_text(cleaned) == [
            1.0,
            5.0,
        ]
        same_clause = _numeric_occurrence_source_text(
            "Increase the second digit and decrease the allowance by 1 if needed."
        )
        assert validator_pipeline.extract_numeric_occurrences_from_text(
            same_clause
        ) == [1.0]
        preceding_amount = _numeric_occurrence_source_text(
            "Reduce the allowance by 1, increase the second digit by 1 if the "
            "third digit is 5 or more."
        )
        assert validator_pipeline.extract_numeric_occurrences_from_text(
            preceding_amount
        ) == [1.0, 1.0, 5.0]
        normalized_preceding_amount = _numeric_occurrence_source_text(
            "Reduce the allowance by 1, increase the second digit by 1 if the "
            "third digit is 5 or more.",
            suppress_source_backed_half_up_increment=True,
        )
        assert validator_pipeline.extract_numeric_occurrences_from_text(
            normalized_preceding_amount
        ) == [1.0, 1.0, 5.0]

        unrecognized_threshold = _numeric_occurrence_source_text(
            "Increase the second digit by 1 if the third digit is 4 or more."
        )
        assert validator_pipeline.extract_numeric_occurrences_from_text(
            unrecognized_threshold
        ) == [1.0, 4.0]

        negated_instruction = _numeric_occurrence_source_text(
            "Do not, under any circumstances, increase the second digit by 1 "
            "if the third digit is 5 or more."
        )
        assert validator_pipeline.extract_numeric_occurrences_from_text(
            negated_instruction
        ) == [1.0, 5.0]

        recognized_instruction = (
            "Increase the second digit by 1 if the third digit is 5 or more."
        )
        assert validator_pipeline.extract_numeric_occurrences_from_text(
            _numeric_occurrence_source_text(recognized_instruction)
        ) == [1.0, 5.0]
        assert validator_pipeline.extract_numeric_occurrences_from_text(
            _numeric_occurrence_source_text(
                recognized_instruction,
                suppress_source_backed_half_up_increment=True,
            )
        ) == [5.0]

    def test_half_up_helper_eval_metric_uses_authoritative_rounding_instruction(
        self, tmp_path
    ):
        active_instruction = (
            "If the 3rd digit is 5 or more, increase the 2nd digit by 1"
        )
        passive_instruction = (
            "If the 3rd digit is 5 or more, the 2nd digit shall be increased by 1."
        )
        source_text = f"{active_instruction}\n{passive_instruction}"
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="ca/policy/cra/example/rounding",
            body=source_text,
        )
        rulespec_file = tmp_path / "policies" / "cra" / "example" / "rounding.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: ca/policy/cra/example/rounding
rules:
  - name: rounding_half_unit
    kind: parameter
    dtype: Decimal
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: parameter
            source:
              corpus_citation_path: ca/policy/cra/example/rounding
              excerpt: if the 3rd digit is 5 or more, increase the 2nd digit by 1
    versions:
      - effective_from: '2025-01-01'
        formula: 0.5
      - effective_from: '2026-01-01'
        formula: 0.5
"""
        )

        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", True, issues=[]),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                return_value=ValidationResult("ci", True, issues=[]),
            ),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "ca"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=source_text,
                local_corpus_release=corpus_release,
                skip_reviewers=True,
            )

        assert metrics.ci_pass
        assert metrics.grounded_numeric_count == 2
        assert metrics.ungrounded_numeric_count == 0
        assert metrics.grounding[0].grounded
        assert metrics.source_numeric_occurrence_count == 2
        assert metrics.covered_source_numeric_occurrence_count == 2
        assert metrics.missing_source_numeric_occurrence_count == 0
        assert not any(
            "Ungrounded generated numeric literal" in issue
            for issue in metrics.ci_issues
        )

    def test_half_up_helper_does_not_cover_independent_source_value(self, tmp_path):
        source_text = (
            "Increase the second digit after the decimal point by one if the "
            "third digit is five or more. Charge a fee of $5."
        )
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="ca/policy/cra/example/rounding-and-fee",
            body=source_text,
        )
        rulespec_file = (
            tmp_path / "policies" / "cra" / "example" / "rounding-and-fee.yaml"
        )
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: ca/policy/cra/example/rounding-and-fee
  summary: Charge a fee of $5.
rules:
  - name: rounding_half_unit
    kind: parameter
    dtype: Decimal
    versions:
      - effective_from: '2025-01-01'
        formula: 0.5
      - effective_from: '2026-01-01'
        formula: 0.5
"""
        )

        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", True, issues=[]),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                return_value=ValidationResult("ci", True, issues=[]),
            ),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "ca"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=source_text,
                local_corpus_release=corpus_release,
                skip_reviewers=True,
            )

        assert not metrics.ci_pass
        assert metrics.grounded_numeric_count == 2
        assert metrics.ungrounded_numeric_count == 0
        assert metrics.source_numeric_occurrence_count == 1
        assert metrics.covered_source_numeric_occurrence_count == 0
        assert metrics.missing_source_numeric_occurrence_count == 1
        assert any("Source numeric value 5" in issue for issue in metrics.ci_issues)

    def test_parameter_table_grounding_uses_corpus_source_with_compact_summary(
        self, tmp_path
    ):
        source_text = "Family Size Fee Level 1 Income Maximum 1 0-1,110 2 0-1,499"
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-az/manual/des/ccap/income-chart-ffy2026/page-1",
            body=source_text,
        )
        rulespec_file = tmp_path / "policies" / "des" / "ccap" / "chart.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-az/manual/des/ccap/income-chart-ffy2026/page-1
  summary: |-
    Child Care Assistance Gross Monthly Income Eligibility Chart and Fee Schedule.
rules:
  - name: fee_level_1_monthly_income_maximum
    kind: parameter
    dtype: Money
    period: Month
    unit: USD
    indexed_by: family_size
    metadata:
      proof:
        atoms:
          - path: versions[0].values
            kind: parameter_table
            source:
              corpus_citation_path: us-az/manual/des/ccap/income-chart-ffy2026/page-1
              excerpt: Fee Level 1 monthly income maximum
              table:
                header: Gross Monthly Income Eligibility Chart
                row_key: Family Size
                column_key: Fee Level 1
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 1110
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us-az"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=source_text,
                local_corpus_release=corpus_release,
            )

        assert metrics.ci_pass
        assert metrics.grounded_numeric_count == 1
        assert metrics.ungrounded_numeric_count == 0
        assert metrics.source_numeric_occurrence_count == 0

    def test_collapsed_household_size_schedule_treats_row_keys_as_structural(
        self, tmp_path
    ):
        source_text = (
            "Household Size Allowable TCA Monthly Payment 1 $348 2 $612 3 $773"
        )
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-md/guidance/dhs/fia/im-26-13/fip-schedule",
            body=source_text,
        )
        rulespec_file = (
            tmp_path
            / "policies"
            / "dhs"
            / "fia"
            / "im-26-13"
            / "allowable-tca-monthly-payment.yaml"
        )
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-md/guidance/dhs/fia/im-26-13/fip-schedule
rules:
  - name: allowable_tca_monthly_payment
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    metadata:
      proof:
        atoms:
          - path: versions[0].values
            kind: parameter_table
            source:
              corpus_citation_path: us-md/guidance/dhs/fia/im-26-13/fip-schedule
              excerpt: Household Size Allowable TCA Monthly Payment
              table:
                header: Household Size Allowable TCA Monthly Payment
                row_key: Household Size
                column_key: Allowable TCA Monthly Payment
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 348
          2: 612
          3: 773
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us-md"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=source_text,
                local_corpus_release=corpus_release,
            )

        assert metrics.ci_pass
        assert metrics.grounded_numeric_count == 3
        assert metrics.ungrounded_numeric_count == 0
        assert metrics.source_numeric_occurrence_count == 6
        assert metrics.covered_source_numeric_occurrence_count == 6
        assert metrics.missing_source_numeric_occurrence_count == 0

    def test_numeric_occurrence_check_counts_imported_named_scalars(self, tmp_path):
        policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
        child = policy_repo / "statutes" / "7" / "2015" / "d" / "2" / "B.yaml"
        child.parent.mkdir(parents=True)
        child.write_text(
            """format: rulespec/v1
rules:
  - name: dependent_child_age_exemption_threshold_years
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '2008-10-01'
        formula: |-
          6
"""
        )
        parent = policy_repo / "statutes" / "7" / "2015" / "d" / "2.yaml"
        parent.write_text(
            """format: rulespec/v1
module:
  summary: |-
    A household member with responsibility for care of a dependent child under age 6 is exempt.
imports:
  - us:statutes/7/2015/d/2/B
rules:
  - name: person_exempt_from_paragraph_1_work_requirements
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2008-10-01'
        formula: care_responsibility_exemption_applies
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=parent,
                policy_repo_root=policy_repo,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="A household member with responsibility for care of a dependent child under age 6 is exempt.",
            )

        assert metrics.ci_pass
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_counts_imported_numeric_concept_names(
        self, tmp_path
    ):
        policy_repo = _canonical_rulespec_content_root(tmp_path, "us-co")
        federal_repo = _canonical_rulespec_content_root(tmp_path, "us")
        child = (
            federal_repo
            / "policies"
            / "usda"
            / "snap"
            / "fy-2026-cola"
            / "income-eligibility-standards.yaml"
        )
        child.parent.mkdir(parents=True)
        child.write_text(
            """format: rulespec/v1
rules:
  - name: snap_gross_income_limit_130_percent_fpl_48_states_dc
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: snap_gross_income_limit_130_percent_fpl_48_states_dc_table[household_size]
"""
        )
        parent = policy_repo / "regulations" / "example.yaml"
        parent.parent.mkdir(parents=True)
        parent.write_text(
            """format: rulespec/v1
module:
  summary: |-
    Other households must meet the 130% gross income standard.
imports:
  - us:policies/usda/snap/fy-2026-cola/income-eligibility-standards#snap_gross_income_limit_130_percent_fpl_48_states_dc
rules:
  - name: colorado_snap_gross_income_limit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: snap_gross_income_limit_130_percent_fpl_48_states_dc
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=parent,
                policy_repo_root=policy_repo,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="Other households must meet the 130% gross income standard.",
            )

        assert metrics.ci_pass
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_counts_formula_identifier_numbers(self, tmp_path):
        rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/example.yaml")
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    Children under age 18 qualify.
rules:
  - name: qualifying_child
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: child_under_18
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="Children under age 18 qualify.",
            )

        assert metrics.ci_pass
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_counts_verification_values(self, tmp_path):
        rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/example.yaml")
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    The standard deduction amounts are $209 for household sizes 1 through 3 and $223 for household size 4.
rules:
  - name: restates_standard_deduction
    kind: source_relation
    source_relation:
      type: restates
      target: us:policies/usda/snap/fy-2026-cola/deductions#snap_standard_deduction
      authority: federal
    verification:
      values:
        snap_standard_deduction_table:
          1: 209
          2: 209
          3: 209
          4: 223
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "The standard deduction amounts are $209 for household sizes "
                    "1 through 3 and $223 for household size 4."
                ),
            )

        assert metrics.ci_pass
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_counts_deferred_output_reasons(self, tmp_path):
        rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/example.yaml")
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  deferred_outputs:
    - output: example:provision#full_deduction_for_aged_households
      reason: This branch depends on whether the household contains a person aged 60 or older.
  summary: |-
    A deduction is 10 dollars unless the household has a person aged 60 or older.
rules:
  - name: base_deduction_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2025-01-01'
        formula: '10'
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "A deduction is 10 dollars unless the household has a "
                    "person aged 60 or older."
                ),
            )

        assert metrics.ci_pass
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_ignores_deferred_reason_section_numbers(
        self, tmp_path
    ):
        rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/example.yaml")
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  deferred_outputs:
    - output: example:provision#separate_branch
      reason: This branch is deferred until Section 4.000.1 is encoded.
rules: []
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="The operative non-citation amount is 4 dollars.",
            )

        assert not metrics.ci_pass
        assert metrics.numeric_occurrence_issues == [
            "Source numeric value 4 appears 1 time(s), but only 0 named scalar "
            "definition(s) with that value were found."
        ]

    def test_repeated_source_scalar_is_covered_by_one_named_definition(self, tmp_path):
        rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/example.yaml")
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    2A. Where earnings are less than £20 in any week and would not exceed £20.
rules:
  - name: pc_special_employment_maximum_weekly_amount
    kind: parameter
    entity: Person
    dtype: Money
    period: Week
    versions:
      - effective_from: '2025-03-31'
        formula: 20
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "2A. Where earnings are less than £20 in any week and "
                    "would not exceed £20."
                ),
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.source_numeric_occurrence_count == 2
        assert metrics.covered_source_numeric_occurrence_count == 2
        assert metrics.missing_source_numeric_occurrence_count == 0
        assert metrics.numeric_occurrence_issues == []

    def test_prefixed_context_import_uses_declared_authority(self, tmp_path):
        policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
        source = policy_repo / "statutes" / "1" / "source.yaml"
        shadow = policy_repo / "statutes" / "1" / "child.yaml"
        target = (
            _canonical_rulespec_content_root(tmp_path, "uk")
            / "statutes"
            / "1"
            / "child.yaml"
        )
        source.parent.mkdir(parents=True)
        content = "format: rulespec/v1\nimports:\n  - uk:statutes/1/child\nrules: []\n"
        source.write_text(content)

        def parameter_payload(value):
            return f"""format: rulespec/v1
rules:
  - name: authority_marker
    kind: parameter
    dtype: Integer
    versions:
      - effective_from: '2026-01-01'
        formula: {value}
"""

        for path, value in ((shadow, 11), (target, 22)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(parameter_payload(value))

        with validator_pipeline._authoritative_rulespec_dependency_scope(
            (tmp_path / "rulespec-uk",)
        ):
            assert _candidate_import_rule_files(
                "uk:statutes/1/child",
                policy_repo,
            ) == [target.resolve()]
            assert _resolve_context_imports(source, policy_repo) == [target.resolve()]
            occurrences = _imported_named_scalar_occurrences(content, policy_repo)
            assert occurrences[22.0] == 1
            assert occurrences[11.0] == 0

    @pytest.mark.parametrize(
        "source_root",
        ["legislation", "policies", "regulations", "statutes"],
    )
    def test_unprefixed_context_import_accepts_each_atomic_root(
        self, tmp_path, source_root
    ):
        policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
        source = policy_repo / "statutes" / "source.yaml"
        target = policy_repo / source_root / "example" / "child.yaml"
        source.parent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        import_target = f"{source_root}/example/child"
        source.write_text(
            f"format: rulespec/v1\nimports:\n  - {import_target}\nrules: []\n"
        )
        target.write_text("format: rulespec/v1\nrules: []\n")

        assert _candidate_import_rule_files(import_target, policy_repo) == [
            target.resolve()
        ]
        assert _resolve_context_imports(source, policy_repo) == [target.resolve()]

    @pytest.mark.parametrize(
        "import_target",
        [
            "programs/example/fy-2026",
            "us:programs/example/fy-2026",
        ],
    )
    def test_context_import_rejects_composition_specs(
        self,
        tmp_path,
        import_target,
    ):
        policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
        source = policy_repo / "statutes" / "source.yaml"
        program_spec = policy_repo / "programs" / "example" / "fy-2026.yaml"
        source.parent.mkdir(parents=True, exist_ok=True)
        program_spec.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            f"format: rulespec/v1\nimports:\n  - {import_target}\nrules: []\n"
        )
        program_spec.write_text("format: axiom-compose/program/v1\nsteps: []\n")

        assert _candidate_import_rule_files(import_target, policy_repo) == []
        assert _resolve_context_imports(source, policy_repo) == []

    def test_context_manifest_rejects_composition_specs(self, tmp_path):
        policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
        program_spec = policy_repo / "programs" / "example" / "fy-2026.yaml"
        program_spec.parent.mkdir(parents=True)
        program_spec.write_text("format: axiom-compose/program/v1\nsteps: []\n")

        with pytest.raises(UnsafeRulespecContextPath, match="ProgramSpecs"):
            _context_import_target(
                program_spec,
                Path("programs/example/fy-2026.yaml"),
            )

    def test_context_import_does_not_probe_parent_or_jurisdiction_alias(self, tmp_path):
        policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
        source = policy_repo / "statutes" / "source.yaml"
        parent_shadow = policy_repo.parent / "statutes" / "example" / "child.yaml"
        source.parent.mkdir(parents=True, exist_ok=True)
        parent_shadow.parent.mkdir(parents=True, exist_ok=True)
        parent_shadow.write_text("format: rulespec/v1\nrules: []\n")

        for import_target in ("statutes/example/child", "us/statutes/example/child"):
            source.write_text(
                f"format: rulespec/v1\nimports:\n  - {import_target}\nrules: []\n"
            )
            assert _candidate_import_rule_files(import_target, policy_repo) == []
            assert _resolve_context_imports(source, policy_repo) == []

    def test_numeric_occurrence_check_ignores_section_cross_references(self, tmp_path):
        rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/example.yaml")
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: Households must receive an opportunity to participate within thirty days.
rules:
  - name: standard_opportunity_to_participate_deadline_days
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '2025-01-01'
        formula: 30
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "Households shall receive an opportunity to participate within "
                    "thirty (30) calendar days. The office shall determine delay "
                    "cause as outlined in Sections 4.205.3 through 4.205.4."
                ),
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.source_numeric_occurrence_count == 1
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_ignores_leading_zero_manual_sections(
        self, tmp_path
    ):
        rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/example.yaml")
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: Combined Manual 0020.21 sets the person living alone standard at $1,055.00.
rules:
  - name: mn_msa_person_living_alone_standard
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: 1055
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "Combined Manual 0020.21 provides: Person living alone $1,055.00."
                ),
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.source_numeric_occurrence_count == 1
        assert metrics.numeric_occurrence_issues == []

    def test_ignores_bracketed_superseded_numeric_source_text(self, tmp_path):
        rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/example.yaml")
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: As of October 1, [2024] 2025, the allowance is [$31] $32.
rules:
  - name: telephone_standard_allowance_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: 32
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "As of October 1, [2024] 2025, the allowance is [$31] $32."
                ),
            )

        assert metrics.ci_pass
        assert not any("31" in issue for issue in metrics.numeric_occurrence_issues)

    def test_preserves_bracketed_formula_numeric_source_text(self):
        numbers = validator_pipeline.extract_numbers_from_text(
            "684 to 4,999 | n = 150 + [ 0.1224(N-683)]"
        )

        assert 0.1224 in numbers
        assert 683 in numbers

    def test_extracts_split_california_schedule_table_money_cells(self):
        numbers = validator_pipeline.extract_numbers_from_text(
            """The schedule of minimum basic standards of adequate care is as follows:
Number of eligible needy persons in the same family
Minimum basic standards of adequate care
1 ........................
$ 341
2 ........................
560
3 ........................
694
4 ........................
824
5 ........................
940
6 ........................
1,057
10 ........................
1,489
plus fourteen dollars ($14) for each additional needy person."""
        )

        for value in (10, 341, 560, 694, 824, 940, 1057, 1489, 14):
            assert float(value) in numbers

    def test_accepts_pence_threshold_grounded_as_decimal_gbp(self, tmp_path):
        rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/example.yaml")
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    13. Small amounts of state pension credit

    Where the amount of state pension credit payable is less than 10 pence per week,
    the credit shall not be payable unless the claimant is in receipt of another benefit
    payable with the credit.
rules:
  - name: small_amount_threshold
    kind: parameter
    entity: Person
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '2025-03-21'
        formula: 0.10
  - name: amount_payable
    kind: input
    entity: Person
    dtype: Money
    period: Week
    unit: GBP
  - name: is_payable
    kind: derived
    entity: Person
    dtype: Boolean
    period: Week
    versions:
      - effective_from: '2025-03-21'
        formula: amount_payable >= small_amount_threshold
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "uk"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "Where the amount of state pension credit payable is less than "
                    "10 pence per week, the credit shall not be payable."
                ),
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.ungrounded_numeric_count == 0
        assert metrics.missing_source_numeric_occurrence_count == 0

    def test_runs_generalist_reviewer_and_records_result(self, tmp_path):
        rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/example.yaml")
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: Provision text with £10.
rules:
  - name: example_amount
    kind: parameter
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2025-01-01'
        formula: 10
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])
        reviewer_result = ValidationResult(
            "generalist-reviewer",
            False,
            score=4.5,
            issues=["Merged distinct statutory branches."],
        )

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
            patch.object(
                ValidatorPipeline, "_run_reviewer", return_value=reviewer_result
            ) as mock_reviewer,
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "uk"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="Provision text with £10.",
            )

        assert metrics.compile_pass is True
        assert metrics.ci_pass is True
        assert metrics.generalist_review_pass is False
        assert metrics.generalist_review_score == 4.5
        assert metrics.generalist_review_issues == [
            "Merged distinct statutory branches."
        ]
        mock_reviewer.assert_called_once()
        assert mock_reviewer.call_args.args[0] == "generalist-reviewer"
        assert "atomic source slice" in mock_reviewer.call_args.kwargs["review_context"]
        assert (
            "stale, generic, or misleading"
            in mock_reviewer.call_args.kwargs["review_context"]
        )

    def test_timing_clause_review_context_mentions_boolean_day_predicate(
        self, tmp_path
    ):
        rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/example.yaml")
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: On the first day of the next benefit week.
rules:
  - name: example_timing_rule
    kind: parameter
    entity: Person
    dtype: Boolean
    period: Day
    versions:
      - effective_from: '2025-01-01'
        formula: true
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])
        reviewer_result = ValidationResult(
            "generalist-reviewer",
            True,
            score=8.0,
            issues=[],
        )

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
            patch.object(
                ValidatorPipeline, "_run_reviewer", return_value=reviewer_result
            ) as mock_reviewer,
        ):
            evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rulespec_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="On the first day of the next benefit week.",
            )

        assert (
            "boolean day-predicate helper"
            in mock_reviewer.call_args.kwargs["review_context"]
        )

    def test_build_eval_prompt_for_uk_timing_leaf_discourages_invented_day_offsets(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/10",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Where the assessed amount comprises income from capital, it shall be "
                "deemed to increase or decrease on the first day of the next benefit "
                "week to commence on or after the day on which the income increases "
                "or decreases."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/10",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-10.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Do not convert relative temporal phrases" in prompt
        assert "`*_offset = 1`" in prompt

    def test_build_eval_prompt_for_atomic_conjunctive_branch_discourages_normative_names(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/10",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Where the Secretary of State is informed that the arrangements under "
                "which the assessed amount is paid contains provision—\n\n"
                "(b)\n\n"
                "for the date on which the increase is to be paid; and"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/10",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-10.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "atomic conjunctive branch slices" in prompt
        assert "do not pretend to encode the whole parent consequence" in prompt
        assert "avoid standalone normative names like `..._must_...`" in prompt
        assert "do not make the principal output a bare input stub" in prompt
        assert "feed the asserted output back into `input:`" in prompt
        assert "treat the carve-out as displacing this slice" in prompt

    def test_build_eval_prompt_for_comparative_month_apart_phrase_discourages_numeric_thresholds(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "the last four payments if the last two payments are less than one month apart; or"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "less than one month apart" in prompt
        assert "one_month_threshold = 1" in prompt
        assert "the `one month` comparator is not a standalone numeric scalar" in prompt
        assert "do not invent `1`-valued threshold/count helpers" in prompt
        assert (
            "branch-specific output is a `Count` or other non-Boolean basis selector"
            in prompt
        )
        assert "do not write an inline conditional without `else`" in prompt
        assert (
            "negative tests should usually assert only the `_applies` boolean" in prompt
        )
        assert (
            "expect the principal basis-count output to remain the active legal basis"
            in prompt
        )
        assert "trigger decomposed-date CI failures" in prompt

    def test_build_eval_prompt_for_single_payment_period_discourages_parallel_units(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "where the period in respect of which a payment is made exceeds a week, "
                "and in a case where that period is three months, the amount is calculated ..."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "keep one canonical fact or classification for that single period" in prompt
        )
        assert "parallel free inputs like `*_in_weeks` and `*_in_months`" in prompt
        assert "do not require a second independent duration input" in prompt
        assert (
            "do not feed the same legal period through contradictory units or categories"
            in prompt
        )

    def test_build_eval_prompt_for_amount_included_determination_requires_applicability_bound_money_output(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "the amount to be included in the claimant's weekly income shall be determined—\n\n"
                "(ii)\n\n"
                "in a case where that period is three months, by multiplying the amount of the payment by 4 and dividing the product by 52;"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "do not leave that money output unconditional" in prompt
        assert "typically with an explicit `else: 0`" in prompt
        assert (
            "paragraph-level exceptions or a different payment period displace the limb"
            in prompt
        )

    def test_build_eval_prompt_for_subject_to_includes_leaf_discourages_blanket_negation(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                'Subject to paragraphs (3), (4) and (4A), "earnings" in the case '
                "of employment as an employed earner, means any remuneration or "
                "profit derived from that employment and includes—\n\n"
                "(a)\n\n"
                "any bonus or commission;"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Subject to paragraphs (3), (4) and (4A), ... includes—" in prompt
        assert "blanket negating gate" in prompt
        assert "Do not make a composite `subject_to_*_satisfied`" in prompt
        assert "branch-specific fact gate" in prompt
        assert "permits this branch to count" in prompt
        assert (
            "do not collapse all cited qualifications into one opaque helper" in prompt
        )
        assert (
            "one paragraph-specific qualification input or import per cited paragraph"
            in prompt
        )

    def test_build_eval_prompt_for_payment_level_slice_discourages_blind_unsupported_fallback(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Except where paragraph (2) and (4) apply, where the period in respect "
                "of which a payment is made does not exceed a week, the whole of that "
                "payment shall be included in the claimant's weekly income."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "individual payment or `that payment`" in prompt
        assert "preserve that payment-scoped subject" in prompt
        assert "prefer `entity: Payment`" in prompt
        assert "prefer `entity: Asset`" in prompt
        assert "provide per-payment rows under `tables:`" in prompt
        assert "provide per-item rows under `tables:`" in prompt
        assert "exact entity name `Payment:`" in prompt
        assert "Use `status: entity_not_supported`" in prompt
        assert "only as a last resort" in prompt
        assert "Do not prefer that fallback" in prompt

    def test_build_eval_prompt_for_except_where_and_citations_discourages_joint_exception(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Except where paragraph (2) and (4) apply, the amount to be included "
                "shall be determined—"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Except where paragraph (2) and (4) apply" in prompt
        assert (
            "do not assume the exception is displaced only when both cited paragraphs apply simultaneously"
            in prompt
        )
        assert (
            "treat the slice as inoperative when any cited paragraph applies" in prompt
        )

    def test_build_eval_prompt_for_payable_phrase_preserves_payability_fact(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "statutory sick pay and statutory maternity pay payable by the "
                "employer under the 1992 Act;"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "model payability as the legal fact" in prompt
        assert "Do not replace `payable` with `receives` or `received`" in prompt

    def test_build_eval_prompt_for_regular_pattern_clause_preserves_full_qualifier(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "the claimant's regular pattern of work is such that he does not "
                "work the same hours every week;"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "regular pattern of work is such that" in prompt
        assert (
            "Do not shorten the branch to only `does not work the same hours every week`"
            in prompt
        )

    def test_build_eval_prompt_for_enumerated_payments_discourages_or_collapse(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "statutory sick pay and statutory maternity pay payable by the "
                "employer under the 1992 Act;"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "do not collapse them into a single `x_or_y` principal output" in prompt
        assert "statutory_sick_pay_or_statutory_maternity_pay_*" in prompt

    def test_build_eval_prompt_for_branch_slice_preserves_binding_lead_in_conjuncts(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/10",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "This paragraph applies where the period which—\n\n"
                "is a period of the same length as the period in respect of which "
                "the last payment of the pre-increase assessed amount was made.\n\n"
                "(b)\n\n"
                "ends on the first increased payment date,"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/10",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-10.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "distinguish mere placement context from binding lead-in conjuncts"
            in prompt
        )
        assert "preserve both conjuncts" in prompt
        assert "do not drop the same-length requirement" in prompt

    def test_build_eval_prompt_for_where_on_branch_discourages_material_implication(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/13B",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "All benefits except those mentioned in paragraph (1) shall be treated as paid—\n\n"
                "(b)\n\n"
                "where the benefit is paid in arrears, on the last day of the benefit week "
                "in which the benefit is payable."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/13B",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-13B.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "treat `X` and `Y` as a positive conjunction for this branch" in prompt
        assert "Do not rewrite that as material implication like `not X or Y`" in prompt
        assert (
            "if the branch-triggering condition itself is false, the branch-specific output should usually be `false`"
            in prompt
        )

    def test_build_eval_prompt_for_disjunctive_payment_description_preserves_qualifier_scope(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/15",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "For the purposes of section 15(1)(j) (income to include income of prescribed descriptions), "
                "income of the following descriptions is prescribed—\n\n"
                "(ac)\n\n"
                "any retired pay, pension or allowance granted in respect of disablement or any pension or "
                "allowance granted to a widow, widower or surviving civil partner in respect of a death due to "
                "service or war injury under an instrument specified in section 639(2) of the Income Tax "
                "(Earnings and Pensions) Act 2003, where such payment does not fall within paragraph (a) of the "
                "definition of “war disablement pension” in section 17(1) of the State Pension Credit Act 2002 or, "
                "in respect of any retired pay or pension granted in respect of disablement, where such payment "
                "does not fall within paragraph (b) of that definition;"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/15",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-15.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "preserve the scope of the first qualifier across every antecedent payment type it grammatically modifies"
            in prompt
        )
        assert (
            "do not narrow the first `where ...` clause to only the later-mentioned category"
            in prompt
        )
        assert "preserve the paragraph-(a) path for retired pay and pension" in prompt

    def test_build_eval_prompt_for_royalties_slice_preserves_consideration_scope(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17/5/a",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "PART III Income\n\n"
                "17. Calculation of weekly income\n\n"
                "(5)\n\n"
                "This paragraph applies to—\n\n"
                "(a)\n\n"
                "royalties or other sums received as a consideration for the use of, or the "
                "right to use, any copyright, design, patent or trade mark;"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17/5/a",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17-5-a.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "preserve the consideration-for-use/right-to-use qualifier across both `royalties` and `other sums`"
            in prompt
        )
        assert "do not model `royalty` as a free-standing qualifying limb" in prompt
        assert "a bare `payment_is_royalty` fact is too broad" in prompt

    def test_build_eval_prompt_for_employed_earner_definition_preserves_shared_qualifier(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A/5",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "17A. Earnings of an employed earner\n\n"
                "(5)\n\n"
                "In this regulation “employed earner” means a person who is gainfully "
                "employed in Great Britain either under a contract of service, or in an "
                "office (including elective office) with emoluments chargeable to income "
                "tax under Schedule E."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A/5",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A-5.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "preserve the shared qualifying employment/office across each alternative limb"
            in prompt
        )
        assert (
            "do not decompose the rule into one free-standing `person_is_X` fact plus separate `under_A` and `in_B` facts"
            in prompt
        )
        assert (
            "distribute the shared qualifier across the alternatives with branch-specific combined facts"
            in prompt
        )

    def test_build_eval_prompt_for_complete_capital_bands_discourages_fractional_division(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/15",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "For the purposes of section 15(2) (deemed income from capital) and subject to "
                "regulation 17(8) (capital to be disregarded), a claimant’s capital shall be "
                "deemed to yield a weekly income of—\n\n"
                "(a)\n\n"
                "£1 for each £500 in excess of £10,000; and"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/15",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-15.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "treat `for each £500` as counting complete bands, not proportional fractions"
            in prompt
        )
        assert "derive the band count with `floor(excess / band_size)`" in prompt
        assert (
            "include a non-exact-multiple excess case like `£750` above threshold"
            in prompt
        )


class TestGeneratedBundleCleaning:
    def test_clean_generated_file_content_strips_fence_and_trailing_prose(self):
        content = (
            "```yaml\n"
            "- name: base\n"
            "  output:\n"
            "    child_benefit_enhanced_rate: 26.05\n"
            "```\n\n"
            "The encoding captures the enhanced rate."
        )

        cleaned = _clean_generated_file_content(content)

        assert cleaned == (
            "- name: base\n  output:\n    child_benefit_enhanced_rate: 26.05\n"
        )

    def test_clean_generated_file_content_strips_inline_currency_suffixes(self):
        content = (
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: The enhanced rate is 26.05 GBP.\n"
            "rules:\n"
            "  - name: child_benefit_enhanced_rate_amount\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    versions:\n"
            "      - effective_from: '2025-04-07'\n"
            "        formula: 26.05 GBP\n"
        )

        cleaned = _clean_generated_file_content(content)

        assert "26.05 GBP" not in cleaned.split("formula:", 1)[1]
        assert "formula: 26.05" in cleaned

    def test_clean_generated_file_content_repairs_yaml_apostrophe_escapes(self):
        content = (
            "format: rulespec/v1\n"
            "module:\n"
            '  summary: "Double quoted taxpayer\\\'s amount"\n'
            "rules:\n"
            "  - name: military_retirement_benefits_definition\n"
            "    kind: derived\n"
            "    metadata:\n"
            "      proof:\n"
            "        atoms:\n"
            "          - source:\n"
            "              excerpt: 'benefits received as a result of the individual\\'s service'\n"
            "    versions:\n"
            "      - effective_from: '2019-01-01'\n"
            "        formula: benefit_amount\n"
        )

        cleaned = _clean_generated_file_content(content)
        payload = yaml.safe_load(cleaned)

        assert payload["module"]["summary"] == "Double quoted taxpayer's amount"
        excerpt = payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"][
            "excerpt"
        ]
        assert excerpt == "benefits received as a result of the individual's service"

    def test_clean_generated_file_content_repairs_semicolon_excerpts(self):
        content = (
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: Payment exclusions.\n"
            "rules:\n"
            "  - name: excluded_payment_limit\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: USD\n"
            "    metadata:\n"
            "      proof:\n"
            "        atoms:\n"
            "          - source:\n"
            '              excerpt: "per capita Payments ... of two thousand dollars ($2,000) or less"; "up to two thousand dollars ($2,000) per year"; "The first two thousand dollars ($2,000) of each payment is excluded"\n'
            "    versions:\n"
            "      - effective_from: '0001-01-01'\n"
            "        formula: 2000\n"
        )

        cleaned = _clean_generated_file_content(content)
        payload = yaml.safe_load(cleaned)

        excerpt = payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"][
            "excerpt"
        ]
        assert excerpt == (
            "per capita Payments ... of two thousand dollars ($2,000) or less; "
            "up to two thousand dollars ($2,000) per year; "
            "The first two thousand dollars ($2,000) of each payment is excluded"
        )

    def test_clean_generated_file_content_repairs_conjoined_quoted_excerpts(self):
        content = (
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: Fuel cell cap.\n"
            "rules:\n"
            "  - name: residential_clean_energy_fuel_cell_credit_component\n"
            "    kind: derived\n"
            "    metadata:\n"
            "      proof:\n"
            "        atoms:\n"
            "          - source:\n"
            '              excerpt: "applicable percentages of qualified fuel cell property expenditures" and "shall not exceed $500 with respect to each half kilowatt of capacity"\n'
            "    versions:\n"
            "      - effective_from: '2006-01-01'\n"
            "        formula: 500\n"
        )

        cleaned = _clean_generated_file_content(content)
        payload = yaml.safe_load(cleaned)

        excerpt = payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"][
            "excerpt"
        ]
        assert excerpt == (
            "applicable percentages of qualified fuel cell property expenditures "
            "and shall not exceed $500 with respect to each half kilowatt of capacity"
        )

    def test_materialize_eval_artifact_rejects_non_rulespec_bundle(self, tmp_path):
        output_file = tmp_path / "source" / "example.yaml"
        response = (
            "=== FILE: example.yaml ===\nrules:\n  - name: missing_format_header\n"
        )

        wrote = _materialize_eval_artifact(response, output_file)

        assert wrote is False
        assert not output_file.exists()

    def test_materialize_tests_only_repair_preserves_rulespec_and_expands_tests(
        self, tmp_path
    ):
        output_file = tmp_path / "regulation/7/273/4.yaml"
        rulespec = "format: rulespec/v1\nrules: []\n"
        original_tests = (
            "- name: existing\n"
            "  period: 2025-06\n"
            "  input:\n    fact: true\n"
            "  output:\n    result: holds\n"
        )
        candidate = ValidationRetryCandidate(rulespec, original_tests)
        required_contract = {
            "name": "required-witness",
            "period": "2025-06",
            "input": {"other_fact": True},
            "required_output": {"other_result": "holds"},
        }
        response = (
            "=== FILE: 4.test.yaml ===\n"
            + original_tests
            + "- name: required-witness\n"
            "  period: 2025-06\n"
            "  input:\n    other_fact: true\n"
            "  output:\n    other_result: holds\n"
        )
        materialized: set[Path] = set()

        wrote = _materialize_eval_artifact(
            response,
            output_file,
            artifact_root=tmp_path,
            materialized_paths=materialized,
            repair_candidate=candidate,
            required_test_case_contracts=(required_contract,),
        )

        assert wrote is True
        assert output_file.read_text() == rulespec
        assert "required-witness" in output_file.with_suffix(".test.yaml").read_text()
        assert materialized == {output_file, output_file.with_suffix(".test.yaml")}

    def test_materialize_tests_only_repair_preserves_cases_wrapper(self, tmp_path):
        output_file = tmp_path / "regulation/7/273/4.yaml"
        rulespec = "format: rulespec/v1\nrules: []\n"
        original_tests = (
            "cases:\n"
            "  - name: existing\n"
            "    period: 2025-06\n"
            "    input: {}\n"
            "    output:\n      result: holds\n"
        )
        candidate = ValidationRetryCandidate(rulespec, original_tests)
        contract = {
            "name": "required",
            "period": "2025-06",
            "input": {},
            "required_output": {"other_result": "holds"},
        }
        response = (
            "=== FILE: 4.test.yaml ===\n" + original_tests + "  - name: required\n"
            "    period: 2025-06\n"
            "    input: {}\n"
            "    output:\n      other_result: holds\n"
        )

        wrote = _materialize_eval_artifact(
            response,
            output_file,
            artifact_root=tmp_path,
            repair_candidate=candidate,
            required_test_case_contracts=(contract,),
        )

        assert wrote is True
        assert yaml.safe_load(output_file.with_suffix(".test.yaml").read_text()) == {
            "cases": [
                {
                    "name": "existing",
                    "period": "2025-06",
                    "input": {},
                    "output": {"result": "holds"},
                },
                {
                    "name": "required",
                    "period": "2025-06",
                    "input": {},
                    "output": {"other_result": "holds"},
                },
            ]
        }

    def test_materialize_tests_only_repair_allows_helper_output_assertions(
        self, tmp_path
    ):
        output_file = tmp_path / "regulation/7/273/4.yaml"
        candidate = ValidationRetryCandidate(
            "format: rulespec/v1\nrules: []\n",
            "- name: existing\n  period: 2025-06\n  input: {}\n"
            "  output:\n    result: holds\n",
        )
        contract = {
            "name": "required",
            "period": "2025-06",
            "input": {"fact": True},
            "required_output": {"required_result": "holds"},
        }
        response = (
            "=== FILE: 4.test.yaml ===\n"
            + candidate.tests
            + "- name: required\n  period: 2025-06\n"
            "  input:\n    fact: true\n"
            "  output:\n    required_result: holds\n    reached_helper: true\n"
        )

        assert _materialize_eval_artifact(
            response,
            output_file,
            artifact_root=tmp_path,
            repair_candidate=candidate,
            required_test_case_contracts=(contract,),
        )

    @pytest.mark.parametrize(
        "proposed_tests",
        [
            "- name: replacement\n  period: 2025-06\n  input: {}\n  output: {}\n",
            "- name: existing\n  period: 2025-07\n  input:\n    fact: true\n  output:\n    result: holds\n",
            "- name: existing\n  period: 2025-06\n  input:\n    fact: false\n  output:\n    result: holds\n",
            "- name: existing\n  period: 2025-06\n  input:\n    fact: true\n  output:\n    result: not_holds\n",
        ],
    )
    def test_materialize_tests_only_repair_rejects_test_weakening(
        self, tmp_path, proposed_tests
    ):
        output_file = tmp_path / "regulation/7/273/4.yaml"
        candidate = ValidationRetryCandidate(
            "format: rulespec/v1\nrules: []\n",
            "- name: existing\n"
            "  period: 2025-06\n"
            "  input:\n    fact: true\n"
            "  output:\n    result: holds\n",
        )

        wrote = _materialize_eval_artifact(
            f"=== FILE: 4.test.yaml ===\n{proposed_tests}",
            output_file,
            artifact_root=tmp_path,
            repair_candidate=candidate,
        )

        assert wrote is False
        assert not output_file.exists()

    def test_materialize_tests_only_repair_rejects_emitted_rulespec(self, tmp_path):
        output_file = tmp_path / "regulation/7/273/4.yaml"
        candidate = ValidationRetryCandidate(
            "format: rulespec/v1\nrules: []\n",
            "- name: existing\n  period: 2025-06\n  input: {}\n  output: {}\n",
        )
        response = (
            "=== FILE: 4.yaml ===\nformat: rulespec/v1\nrules: []\n"
            "=== FILE: 4.test.yaml ===\n" + candidate.tests
        )

        wrote = _materialize_eval_artifact(
            response,
            output_file,
            artifact_root=tmp_path,
            repair_candidate=candidate,
        )

        assert wrote is False
        assert not output_file.exists()

    def test_materialize_tests_only_repair_rejects_type_changes_and_unsigned_cases(
        self, tmp_path
    ):
        output_file = tmp_path / "regulation/7/273/4.yaml"
        candidate = ValidationRetryCandidate(
            "format: rulespec/v1\nrules: []\n",
            "- name: existing\n"
            "  period: 2025-06\n"
            "  input:\n    fact: true\n"
            "  output:\n    result: holds\n",
        )
        contract = {
            "name": "required",
            "period": "2025-06",
            "input": {"required_fact": True},
            "required_output": {"required_result": "holds"},
        }
        type_changed = (
            "=== FILE: 4.test.yaml ===\n"
            "- name: existing\n  period: 2025-06\n"
            "  input:\n    fact: 1\n  output:\n    result: holds\n"
        )
        unsigned_case = (
            "=== FILE: 4.test.yaml ===\n"
            + candidate.tests
            + "- name: unsigned\n  period: 2025-06\n"
            "  input: {}\n  output:\n    result: holds\n"
        )

        for response in (type_changed, unsigned_case):
            assert (
                _materialize_eval_artifact(
                    response,
                    output_file,
                    artifact_root=tmp_path,
                    repair_candidate=candidate,
                    required_test_case_contracts=(contract,),
                )
                is False
            )
            assert not output_file.exists()

    def test_materialize_eval_artifact_repairs_single_file_conjoined_excerpts(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "section-25d.yaml"
        response = (
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: Fuel cell cap.\n"
            "rules:\n"
            "  - name: residential_clean_energy_fuel_cell_credit_component\n"
            "    kind: derived\n"
            "    metadata:\n"
            "      proof:\n"
            "        atoms:\n"
            "          - source:\n"
            '              excerpt: "applicable percentages of qualified fuel cell property expenditures" and "shall not exceed $500 with respect to each half kilowatt of capacity"\n'
            "    versions:\n"
            "      - effective_from: '2006-01-01'\n"
            "        formula: |-\n"
            "          if expenditures_after_termination_date:\n"
            "              0\n"
            "          else:\n"
            "              base_expenditures =\n"
            "                  max(0, qualified_solar_electric_property_expenditures)\n"
            "                  + max(0, qualified_battery_storage_technology_expenditures)\n"
            "              credit = residential_clean_energy_applicable_percentage * base_expenditures\n"
            "              max(0, credit + residential_clean_energy_fuel_cell_credit_component)\n"
        )

        wrote = _materialize_eval_artifact(response, output_file)

        assert wrote is True
        payload = yaml.safe_load(output_file.read_text())
        excerpt = payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"][
            "excerpt"
        ]
        assert excerpt == (
            "applicable percentages of qualified fuel cell property expenditures "
            "and shall not exceed $500 with respect to each half kilowatt of capacity"
        )
        formula = payload["rules"][0]["versions"][0]["formula"]
        assert "base_expenditures =" not in formula
        assert "credit =" not in formula
        assert (
            "residential_clean_energy_applicable_percentage * "
            "(max(0, qualified_solar_electric_property_expenditures)"
        ) in formula

    def test_materialize_eval_artifact_cleans_bundled_rulespec_fences(self, tmp_path):
        output_file = tmp_path / "source" / "uksi-2006-965-regulation-2.yaml"
        llm_response = (
            "=== FILE: uksi-2006-965-regulation-2.yaml ===\n"
            "```yaml\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: The enhanced rate is £26.05.\n"
            "rules:\n"
            "  - name: child_benefit_enhanced_rate\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    versions:\n"
            "      - effective_from: '2025-04-07'\n"
            "        formula: 26.05\n"
            "```\n"
            "=== FILE: uksi-2006-965-regulation-2.test.yaml ===\n"
            "```yaml\n"
            "- name: base\n"
            "  output:\n"
            "    child_benefit_enhanced_rate: 26.05\n"
            "```\n\n"
            "Trailing prose.\n"
        )

        wrote = _materialize_eval_artifact(llm_response, output_file)

        assert wrote is True
        assert output_file.read_text().startswith("format: rulespec/v1\n")
        test_text = output_file.with_suffix(".test.yaml").read_text()
        assert "child_benefit_enhanced_rate: 26.05" in test_text
        assert "period: '2025-04-07'" in test_text

    def test_materialize_eval_artifact_salvages_rulespec_workspace_files_when_response_is_summary(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "uksi-2002-1792-2025-03-31.yaml"
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir(parents=True)
        (workspace_root / output_file.name).write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: The weekly amount is £19.30.\n"
            "rules:\n"
            "  - name: pc_housing_non_dependant_deduction_other_weekly_amount\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    versions:\n"
            "      - effective_from: '2025-03-21'\n"
            "        formula: 19.30\n"
        )
        (workspace_root / output_file.with_suffix(".test.yaml").name).write_text(
            "- name: base_case\n"
            "  period: 2025-04-01\n"
            "  input: {}\n"
            "  output:\n"
            "    pc_housing_non_dependant_deduction_other_weekly_amount: 19.30\n"
        )

        wrote = _materialize_eval_artifact(
            "Both files written.",
            output_file,
            source_text="£19.30",
            workspace_root=workspace_root,
        )

        assert wrote is True
        assert output_file.read_text().startswith("format: rulespec/v1\n")
        assert output_file.with_suffix(".test.yaml").exists()

    def test_materialize_eval_artifact_rejects_non_rulespec_workspace_main(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "example.yaml"
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir(parents=True)
        (workspace_root / output_file.name).write_text(
            "rules:\n  - name: missing_format_header\n"
        )

        wrote = _materialize_eval_artifact(
            "Both files written.",
            output_file,
            workspace_root=workspace_root,
        )

        assert wrote is False
        assert not output_file.exists()

    def test_materialize_eval_artifact_normalizes_rulespec_test_periods(self, tmp_path):
        output_file = tmp_path / "source" / "example.yaml"
        response = """=== FILE: example.yaml ===
format: rulespec/v1
module:
  summary: |-
    The standard utility allowance is $451, effective October 1, 2025.
rules:
  - name: snap_standard_utility_allowance
    kind: parameter
    entity: SnapUnit
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: 451
=== FILE: example.test.yaml ===
- name: pre_effective_zero
  period: 2025-09
  input: {}
  output:
    snap_standard_utility_allowance: 0
- name: applies
  period: 2026-01
  input: {}
  output:
    snap_standard_utility_allowance: 451
"""

        wrote = _materialize_eval_artifact(response, output_file)

        assert wrote is True
        test_text = output_file.with_suffix(".test.yaml").read_text()
        assert "pre_effective_zero" not in test_text
        assert "period: 2026-01" in test_text

    def test_materialize_eval_artifact_normalizes_quoted_date_outputs(self, tmp_path):
        output_file = tmp_path / "source" / "example.yaml"
        response = """=== FILE: example.yaml ===
format: rulespec/v1
module:
  summary: The deadline is seven days after application.
rules:
  - name: deadline_days
    kind: parameter
    dtype: Integer
    versions:
      - effective_from: '2025-01-01'
        formula: 7
  - name: deadline
    kind: derived
    entity: Household
    dtype: Date
    period: Month
    versions:
      - effective_from: '2025-01-01'
        formula: date_add_days(application_date, deadline_days)
=== FILE: example.test.yaml ===
- name: applies
  period: 2026-01
  input:
    '#input.application_date': '2026-01-01'
  output:
    '#deadline': '2026-01-08'
"""

        wrote = _materialize_eval_artifact(response, output_file)

        assert wrote is True
        test_text = output_file.with_suffix(".test.yaml").read_text()
        payload = yaml.safe_load(test_text)
        assert payload[0]["input"]["#input.application_date"] == "2026-01-01"
        assert payload[0]["output"]["#deadline"] == date(2026, 1, 8)

    def test_normalize_test_periods_repairs_misindented_period_end(self):
        rulespec_text = """format: rulespec/v1
module:
  summary: Section defines an annual table.
rules:
  - name: annual_rate
    kind: parameter
    dtype: Rate
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: 0.05
"""
        test_text = _normalize_test_periods_to_effective_dates(
            """- name: ratio_at_9_0
  period:
    period_kind: tax_year
    start: '2026-01-01'
  end: '2026-12-31'
  input: {}
  output:
    annual_rate: 0.05
""",
            rulespec_content=rulespec_text,
        )

        cases = yaml.safe_load(test_text)

        assert cases[0]["period"] == {
            "period_kind": "tax_year",
            "start": "2026-01-01",
            "end": "2026-12-31",
        }
        assert "end" not in cases[0]

    def test_normalize_test_periods_rewrites_iso_week_shorthand(self):
        rulespec_text = """format: rulespec/v1
module:
  summary: Weekly child benefit rate.
rules:
  - name: child_benefit_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '0001-01-01'
        formula: 27.05
"""
        test_text = _normalize_test_periods_to_effective_dates(
            """- name: weekly_case
  period: 2025-W01
  input: {}
  output:
    child_benefit_weekly_rate: 27.05
""",
            rulespec_content=rulespec_text,
        )

        cases = yaml.safe_load(test_text)

        assert cases[0]["period"] == {
            "period_kind": "benefit_week",
            "start": "2024-12-30",
            "end": "2025-01-05",
        }

    def test_materialize_eval_artifact_normalizes_mapping_style_tests_to_list(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "uksi-2002-1792-regulation-10-5-b-ii.yaml"
        response = """=== FILE: uksi-2002-1792-regulation-10-5-b-ii.yaml ===
format: rulespec/v1
module:
  summary: The day referred to in branch ii is true when the condition applies.
rules:
  - name: day_referred_to_10_5_b_ii
    kind: derived
    entity: Person
    dtype: Boolean
    period: Day
    versions:
      - effective_from: '2025-03-21'
        formula: some_fact
=== FILE: uksi-2002-1792-regulation-10-5-b-ii.test.yaml ===
case_branch_ii_applies:
  period: 2025-03-21
  input:
    some_fact: true
  output:
    day_referred_to_10_5_b_ii: true
"""

        wrote = _materialize_eval_artifact(response, output_file)

        assert wrote is True
        test_text = output_file.with_suffix(".test.yaml").read_text()
        assert test_text.lstrip().startswith("- ")
        assert "name: case_branch_ii_applies" in test_text
        assert "case_branch_ii_applies:" not in test_text

    def test_normalize_test_periods_drops_speculative_pre_effective_zero_case_for_monthly_update(
        self,
    ):
        rulespec_text = """format: rulespec/v1
module:
  summary: The SUA is $451 effective October 1, 2025.
rules:
  - name: snap_standard_utility_allowance
    kind: parameter
    entity: SnapUnit
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: 451
"""
        source_text = (
            "Current-effective Tennessee utility allowance slice.\n\n"
            "The Standard Utility Allowance (SUA) is used when the household is\n"
            "responsible for heating or cooling costs.\n"
            "The SUA is $451, effective October 1, 2025.\n"
        )
        test_text = _normalize_test_periods_to_effective_dates(
            "- name: applies\n"
            "  period: 2026-01\n"
            "  output:\n"
            "    snap_standard_utility_allowance: 451\n"
            "- name: pre_effective_month_zero\n"
            "  period: 2025-09\n"
            "  output:\n"
            "    snap_standard_utility_allowance: 0\n",
            rulespec_content=rulespec_text,
            source_text=source_text,
        )

        assert "pre_effective_month_zero" not in test_text
        assert "period: 2026-01" in test_text

    def test_normalize_test_case_value_preserves_invalid_numeric_expression(self):
        assert _normalize_test_case_value("30 / 0") == "30 / 0"

    @pytest.mark.parametrize(
        "literal",
        (
            "432.09845",
            "117.60035",
            "12345.67",
            "+0.00035",
            "-0.00035",
        ),
    )
    def test_normalize_test_case_value_preserves_decimal_literal_precision(
        self,
        literal,
    ):
        assert _normalize_test_case_value(literal) == literal

    def test_normalize_test_case_value_converts_plain_integer_literal(self):
        assert _normalize_test_case_value("350") == 350

    def test_materialize_eval_artifact_preserves_decimal_literal_precision(
        self,
        tmp_path,
    ):
        output_file = tmp_path / "source" / "kentucky_rate.yaml"
        response = """=== FILE: kentucky_rate.yaml ===
format: rulespec/v1
module:
  summary: Kentucky tax before credits is 3.5 percent of net income.
rules:
  - name: normal_tax_before_credits
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        effective_to: '2026-12-31'
        formula: net_income * 0.035
inputs:
  - name: net_income
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
=== FILE: kentucky_rate.test.yaml ===
- name: fractional_net_income_before_credits
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    '#input.net_income': '12345.67'
  output:
    '#normal_tax_before_credits': '432.09845'
"""

        wrote = _materialize_eval_artifact(response, output_file)

        assert wrote is True
        cases = yaml.safe_load(output_file.with_suffix(".test.yaml").read_text())
        assert cases[0]["input"]["#input.net_income"] == "12345.67"
        assert cases[0]["output"]["#normal_tax_before_credits"] == "432.09845"

    def test_materialize_eval_artifact_adds_missing_oracle_hint_output_from_rulespec(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "example.yaml"
        response = """=== FILE: example.yaml ===
format: rulespec/v1
module:
  summary: Homeless Shelter Deduction - $198.99.
rules:
  - name: snap_homeless_shelter_deduction_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: 198.99
  - name: snap_homeless_shelter_deduction_available
    kind: parameter
    dtype: Boolean
    versions:
      - effective_from: '2025-10-01'
        formula: true
=== FILE: example.test.yaml ===
- name: base
  period: 2025-10
  output:
    snap_homeless_shelter_deduction_amount: 198.99
"""

        wrote = _materialize_eval_artifact(
            response,
            output_file,
            policyengine_rule_hint="snap_homeless_shelter_deduction_available",
        )

        assert wrote is True
        test_payload = yaml.safe_load(output_file.with_suffix(".test.yaml").read_text())
        assert (
            test_payload[0]["output"]["snap_homeless_shelter_deduction_available"]
            is True
        )

    def test_materialize_eval_artifact_uses_canonical_oracle_hint_output_key(
        self, tmp_path
    ):
        output_file = (
            tmp_path
            / "rulespec-us"
            / "us"
            / "policies"
            / "usda"
            / "snap"
            / "homeless.yaml"
        )
        response = """=== FILE: homeless.yaml ===
format: rulespec/v1
module:
  summary: Homeless Shelter Deduction availability.
rules:
  - name: snap_homeless_shelter_deduction_available
    kind: parameter
    dtype: Boolean
    versions:
      - effective_from: '2025-10-01'
        formula: true
=== FILE: homeless.test.yaml ===
- name: base
  period: 2025-10
  output:
    snap_homeless_shelter_deduction_available: true
"""

        wrote = _materialize_eval_artifact(
            response,
            output_file,
            policyengine_rule_hint="snap_homeless_shelter_deduction_available",
        )

        assert wrote is True
        output = yaml.safe_load(output_file.with_suffix(".test.yaml").read_text())[0][
            "output"
        ]
        assert "snap_homeless_shelter_deduction_available" not in output
        assert (
            output[
                "us:policies/usda/snap/homeless#snap_homeless_shelter_deduction_available"
            ]
            is True
        )

    def test_can_include_policyengine_metrics_for_uk_artifact(self, tmp_path):
        rules_file = _generated_rulespec_file_path(
            tmp_path, "statutes/uksi-2006-965-regulation-2.yaml"
        )
        rules_file.write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: https://www.legislation.gov.uk/uksi/2006/965/regulation/2 states the enhanced rate is £26.05.\n"
            "rules:\n"
            "  - name: child_benefit_enhanced_rate\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    versions:\n"
            "      - effective_from: '2025-04-07'\n"
            "        formula: 26.05\n"
        )

        compile_result = ValidationResult("compile", passed=True)
        ci_result = ValidationResult("ci", passed=True)
        pe_result = ValidationResult(
            "policyengine",
            passed=True,
            score=1.0,
            issues=[],
        )

        with (
            patch.object(
                PolicyEngineRuntime,
                "assert_matches_rulespec_root",
                return_value=None,
            ),
            patch(
                "axiom_encode.harness.validator_pipeline.ValidatorPipeline._run_compile_check",
                return_value=compile_result,
            ),
            patch(
                "axiom_encode.harness.validator_pipeline.ValidatorPipeline._run_ci",
                return_value=ci_result,
            ),
            patch(
                "axiom_encode.harness.validator_pipeline.ValidatorPipeline._run_policyengine",
                return_value=pe_result,
            ) as mock_policyengine,
        ):
            metrics = evaluate_artifact(
                local_corpus_release=_write_test_corpus_provision(
                    tmp_path / "bound-release"
                ),
                rulespec_file=rules_file,
                policy_repo_root=_canonical_rulespec_content_root(tmp_path, "uk"),
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="The enhanced rate is £26.05 from 2025-04-07.",
                oracle="policyengine",
                policyengine_runtime=_test_policyengine_runtime("uk"),
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.policyengine_pass is True
        assert metrics.policyengine_score == 1.0
        assert metrics.policyengine_issues == []
        mock_policyengine.assert_called_once()


class TestEvalPrompt:
    def test_build_eval_prompt_includes_rulespec_schema_guardrails(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Grant standard is 165 for one child.",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "9 CCR 2503-6 3.606.1",
            "cold",
            workspace,
            [],
            target_file_name="9-CCR-2503-6-3.606.1.yaml",
            include_tests=True,
        )

        assert (
            "Do not invent schema keys like `namespace:`, `parameter`, `variable`, or `rule:`."
            in prompt
        )
        assert "entity:" in prompt
        assert "period:" in prompt
        assert "direct release-bound corpus source text" in prompt
        assert "dtype:" in prompt
        assert "RuleSpec requirements:" in prompt
        assert "The RuleSpec file must begin with `format: rulespec/v1`" in prompt
        assert (
            "Use chained `if condition: value else: other_value` expressions" in prompt
        )
        assert "Do not write `else if` or `elif`" in prompt
        assert "do not inline that cross-reference's mechanics into this file" in prompt
        assert (
            "additional_standard_deduction_entitlement_count_under_subsection_f"
            in prompt
        )
        assert "Do not start a local input with" in prompt
        assert "_under_section_<section>" in prompt
        assert "completed federal\n  return amount" in prompt
        assert "itemized_deductions_claimed_on_federal_return" in prompt
        assert "For IRC section 22" not in prompt
        assert "dependent_of_tax_unit" in prompt
        assert "only the exception input changes" in prompt
        assert (
            "Do not replace a specific upstream output with a broad local input"
            in prompt
        )
        assert "only one entity type" in prompt
        assert "Do not assert relation-child outputs" in prompt
        assert "Do not use bare year periods like `2024`" in prompt
        assert "Never encode US tax filing status" in prompt
        assert "Do not create local `#input.filing_status` facts" in prompt
        assert 'If the source says only "joint return"' in prompt
        assert 'status 4 falls under any "other case" branch' in prompt
        assert "Hard requirement for IRC section 151(d)" not in prompt
        assert "must use the numeric `filing_status` enum input directly" not in prompt
        assert "Importing an adjacent upstream output only as proof" in prompt
        assert "does not satisfy the dependency" in prompt
        assert "is not an executable dependency" in prompt
        assert "Never drop the jurisdiction prefix" in prompt
        assert "listed under invalid copied local inputs" in prompt
        assert "do not preserve, rename, or recreate" in prompt
        assert "bare file-level import is not enough" in prompt
        assert "import the exact `#rule_name`" in prompt
        assert (
            "They are not acceptable for `except`, `unless`, or `subject to` formula carve-outs"
            in prompt
        )
        assert "Subject to paragraph (c)" in prompt
        assert "cash_assistance_less_restrictive_methodologies_may_be_applied" in prompt
        assert "omitting the cited paragraph's symbol is invalid" in prompt
        assert "treated as attributable to" in prompt
        assert "amount-level" in prompt
        assert "boolean or `dtype: Judgment` predicate" in prompt

    def test_build_eval_prompt_for_broad_application_clause_discourages_passthrough_outputs(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="7 USC 2014(a)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Participation in the supplemental nutrition assistance program "
                "shall be limited to those households whose incomes and other "
                "financial resources are determined to be a substantial limiting "
                "factor in permitting them to obtain a more nutritious diet. "
                "Assistance under this program shall be furnished to all eligible "
                "households who make application for such participation."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "7 USC 2014(a)",
            "cold",
            workspace,
            [],
            target_file_name="statutes/7/2014/a.yaml",
            target_ref_prefix="us:statutes/7/2014/a",
            include_tests=True,
        )

        assert (
            "broad application, furnishing, administrative duty, or purpose clause"
            in prompt
        )
        assert (
            "do not create an executable derived output just to paraphrase it" in prompt
        )
        assert "assistance shall be furnished to all eligible households" in prompt
        assert (
            "Do not encode a pure pass-through rule whose formula is only one local fact"
            in prompt
        )
        assert "one-time" in prompt
        assert "more than one consecutive month" in prompt
        assert "Do not append citation or file suffixes like `_2014_a`" in prompt
        assert (
            "For every encoded `except`, `unless`, or `notwithstanding` carve-out"
            in prompt
        )
        assert 'shall not apply" or "does not apply"' in prompt
        assert "that helper as `holds`" in prompt
        assert '"through such account"' in prompt
        assert "sets that exception input true" in prompt
        assert "Do not collapse a list of cited exceptions" in prompt
        assert "Do not create derived `dtype: Boolean` helper rules" in prompt
        assert (
            "copied context file already exports the operative legal condition"
            in prompt
        )
        assert "Do not recreate it as a local factual input" in prompt

    def test_build_eval_prompt_includes_supported_schema_enums(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="2. Rate of child benefit ... 25.60 ... 16.95",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
        )

        assert "Do not invent arbitrary entities." in prompt
        assert (
            "Standard `entity:` examples are `Payment`, `Person`, `TaxUnit`, `Household`, "
            "`Family`, `TanfUnit`, `SnapUnit`, `SPMUnit`, `Corporation`, `Business`, "
            "`Employer`, `Asset`, `StateAgency`."
        ) in prompt
        assert "introduce a narrow singular" in prompt
        assert "Allowed `period:` values are `Year`, `Month`, `Week`, `Day`." in prompt
        assert "do not use ISO week shorthands like `2025-W01`" in prompt
        assert (
            "Allowed `dtype:` values are `Money`, `Rate`, `Boolean`, `Integer`, "
            "`Count`, `String`, `Decimal`, `Float`, or `Enum[Name]`."
        ) in prompt

    def test_build_eval_prompt_includes_unsupported_entity_fallback(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="ukpga/2010/1/section/1",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) cease to be in force",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "ukpga/2010/1/section/1",
            "cold",
            workspace,
            [],
            target_file_name="ukpga-2010-1-section-1.yaml",
            include_tests=True,
        )

        assert (
            "If the source cannot be represented faithfully with the supported schema"
            in prompt
        )
        assert "`module.status: entity_not_supported`" in prompt
        assert "`module.status: deferred`" in prompt
        assert (
            "In a mixed provision, omit or defer only the affected executable" in prompt
        )
        assert "module.deferred_outputs[]" in prompt
        assert "do not model that numeric term as a local" in prompt
        assert "output` target path must include that source path segment" in prompt
        assert "Do not create tests for deferred" in prompt
        assert (
            "only when no executable rule in the requested source can be represented"
            in prompt
        )
        assert "leave the companion `.test.yaml` empty" in prompt
        assert "assertions against deferred symbols" in prompt

    def test_build_eval_prompt_for_filing_status_upstream_sources_requires_executable(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="26 USC 7703",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "(a) General rule The determination of whether an individual is "
                "married shall be made as of the close of his taxable year."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 7703",
            "cold",
            workspace,
            [],
            target_file_name="7703.yaml",
            include_tests=True,
        )

        assert "Hard requirement for IRC sections 2, 6013, and 7703" not in prompt
        assert "section 151 deduction is `allowed` or `allowable`" not in prompt
        assert "Never introduce an import cycle" in prompt
        assert "same rule's name" in prompt
        assert "rate-bearing source" in prompt
        assert "cycle with a foundational base definition" in prompt
        assert "rate or rate" in prompt
        assert "source-named numeric boundary input" in prompt

    def test_build_eval_prompt_for_editorially_omitted_slice_allows_deferred_docstring(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17/10/c",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-03-21.\n\n"
                "(c)\n\n"
                ". . . . . . . . . . . . . . . . . . . . . . . ."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17/10/c",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17-10-c.yaml",
            include_tests=True,
        )

        assert (
            "editorially omitted or repealed text shown by ellipses or dotted placeholders"
            in prompt
        )
        assert "leave `.test.yaml` empty" in prompt

    def test_build_eval_prompt_forbids_python_inline_ternaries(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="2. Rate of child benefit ... 26.05",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
        )

        assert "Do not use Python inline ternaries" in prompt
        assert "`x if cond else y`" in prompt

    def test_build_eval_prompt_requires_rulespec_conditional_expression_syntax(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="2. Rate of child benefit ... 26.05",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
        )

        assert "`if condition: value else: other_value`" in prompt
        assert "do not use YAML-style `if:` / `then:` / `else:` blocks" in prompt
        assert (
            "Do not append a multiline conditional directly onto another expression"
            in prompt
        )

    def test_build_eval_prompt_requires_decimal_ratios_for_rate_dtype(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/7",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="The percentage prescribed is 60 per cent.",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/7",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-7.yaml",
            include_tests=True,
        )

        assert (
            "For `dtype: Rate`, encode percentages as decimal ratios like `0.60` or `0.40`, never as `%` literals"
            in prompt
        )
        assert "never as arithmetic like `25 / 100`" in prompt
        assert "source.corpus_citation_path` is sufficient" in prompt
        assert (
            "Do not respond with summaries, markdown prose, or file-write confirmations"
            in prompt
        )
        assert "concise exact audit excerpt" in prompt
        assert "not the full source text" in prompt
        assert (
            "do not use inline assignment syntax like `:=` inside formula blocks"
            in prompt
        )

    def test_build_eval_prompt_for_uk_leaf_prefers_person_over_family_constant(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("claude:opus"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-04-07.\n\n"
                "The weekly rate of child benefit payable in respect of a child "
                "or qualifying young person shall be 26.05."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
        )

        assert (
            'Prefer `Person` when the source states an amount or condition "in respect of"'
            in prompt
        )
        assert (
            "do not collapse it into an unconditional family-level constant" in prompt
        )

    def test_build_eval_prompt_for_uk_pence_threshold_requires_gbp_decimal_and_weekly_cadence(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/13",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-03-21.\n\n"
                "Where the amount of state pension credit payable is less than 10 pence per week, "
                "the credit shall not be payable unless the claimant is in receipt of another "
                "benefit payable with the credit."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/13",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-13.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "include `unit: GBP`" in prompt
        assert "`10 pence` should become `0.10`, not `10`" in prompt
        assert "do not disguise it as arithmetic like `1 / 10`" in prompt
        assert "prefer a money variable with matching `period:` cadence" in prompt

    def test_build_eval_prompt_for_positive_conditional_uk_leaf_requires_zero_or_false_else_case(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/schedule/VI/paragraph/4/1/a/iva",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-03-21.\n\n"
                "£20 is disregarded if the claimant or, if he has a partner, his partner "
                "is in receipt of Scottish adult disability living allowance."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/schedule/VI/paragraph/4/1/a/iva",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-schedule-vi-paragraph-4-1-a-iva.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "positive conditional leaves" in prompt
        assert (
            "the inapplicable case should usually be `0` for `dtype: Money` or `false` for `dtype: Boolean`"
            in prompt
        )
        assert "do not use an unconditional amount or `else: true`" in prompt
        assert (
            "fixed supplement, allowance, or addition is payable only while an eligibility condition holds"
            in prompt
        )
        assert "do not leave that money output unconditional" in prompt
        assert (
            "do not collapse source-stated component facts into an opaque local input like `*_eligible_for_*`"
            in prompt
        )
        assert "prefer direct facts like `client_is_pregnant_parent`" in prompt

    def test_build_eval_prompt_for_determination_limb_discourages_invented_fallback(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "the weekly amount of that claimant's income shall be determined—\n\n"
                "(i)\n\n"
                "if there is a recognised cycle of work, by reference to his average "
                "weekly income over the period of the complete cycle; or"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "do not invent sibling outcomes for non-applicable cases with `else: 0`"
            in prompt
        )
        assert "leave other cases to sibling limbs" in prompt
        assert "keep a branch-specific money or rate output for that basis" in prompt
        assert (
            "do not invent sibling outcomes for inapplicable cases with `else: 0`"
            in prompt
        )
        assert (
            "pair the branch-specific money or rate output with a separate applicability boolean"
            in prompt
        )
        assert (
            "omit assertions about the branch-specific money or rate output" in prompt
        )
        assert (
            "qualifies its averaging basis with operative parenthetical text" in prompt
        )
        assert (
            "includes periods in which the claimant does no work but disregards other absences"
            in prompt
        )
        assert "generic `average_weekly_income_*` input" in prompt
        assert (
            "`such other payments as may ... enable the claimant's average weekly income to be determined more accurately`"
            in prompt
        )
        assert (
            "do not leave the branch money output unconditionally equal to the input average"
            in prompt
        )
        assert (
            "do not reuse the parent provision's generic final-amount phrase" in prompt
        )
        assert (
            "name the principal money or rate output after this limb's own basis or method"
            in prompt
        )

    def test_build_eval_prompt_for_purpose_limited_deeming_discourages_unsupported_fallback(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17/9A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "For the purposes of paragraph (9)(b), and for that purpose only, "
                "the amounts specified in paragraph (5) shall be treated as though "
                "they were earnings."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17/9A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17-9A.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "purpose-limited deeming clauses" in prompt
        assert "do not use `status: entity_not_supported`" in prompt
        assert (
            "paragraph-(5) amounts treated as earnings for paragraph-(9)(b) only"
            in prompt
        )

    def test_build_eval_prompt_for_uk_residual_determination_limb_requires_other_case_condition(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "the amount to be included in the claimant's weekly income shall be determined—\n\n"
                "(iv)\n\n"
                "in any other case, by multiplying the amount of the payment by 7 and dividing "
                "the product by the number of days in the period in respect of which it is made."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "For residual sibling limbs phrased like `in any other case`" in prompt
        assert "Do not treat the shared parent triggers alone as sufficient" in prompt
        assert "model a local residual-case fact or applicability helper" in prompt
        assert "no more specific sibling case applies" in prompt
        assert (
            "include a case where the parent conditions hold but the residual `other case` condition is false"
            in prompt
        )

    def test_build_eval_prompt_for_shall_be_treated_discourages_fact_input_and_vacuous_true(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "If a claimant is entitled to receive a payment to which paragraph (5) "
                "applies, the amount of that payment shall be treated as if made in "
                "respect of a period of a year."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "do not introduce a `*_fact` input" in prompt
        assert "do not use vacuous `else: true`" in prompt
        assert (
            "do not replace the amount-level legal effect with a `Person`/`Day` boolean stand-in"
            in prompt
        )
        assert (
            "prefer `status: entity_not_supported` over a pseudo-boolean approximation"
            in prompt
        )
        assert (
            "If the current ontology cannot faithfully tie the deeming effect to the same payment amount"
            in prompt
        )

    def test_build_eval_prompt_for_claimant_incurred_expenses_preserves_claimant_predicate(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A/2/f/i",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "travelling expenses incurred by the claimant between his home and place "
                "of employment;"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A/2/f/i",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A-2-f-i.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "If an expenses limb says the expenses are `incurred by the claimant`"
            in prompt
        )
        assert "preserve that claimant-incurred predicate explicitly" in prompt
        assert "Do not collapse it into only an employer-made-payment fact" in prompt

    def test_build_eval_prompt_for_claim_date_reference_day_uses_single_operative_date(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "For the purposes of paragraph (2)(b) the last payments are the last payments "
                "before the date the claim was made or treated as made or, if there is a "
                "subsequent supersession under section 10 of the Social Security Act 1998, "
                "the last payments before the date of the supersession."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "preserve a single legally operative reference day" in prompt
        assert "model one canonical operative claim-date fact" in prompt
        assert (
            "do not encode separate `day_is_date_claim_was_made` and `day_is_date_claim_was_treated_as_made` facts and then combine them with `or`"
            in prompt
        )
        assert (
            "include one no-supersession case for the operative claim date and one supersession case for the supersession date"
            in prompt
        )

    def test_build_eval_prompt_for_subject_to_override_discourages_permission_gate(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Subject to regulation 17B(6), in the case of any income taken into "
                "account for the purpose of calculating a person's income, there shall "
                "be disregarded any amount payable by way of tax."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "treat the cited provision as a possible override or displacement" in prompt
        )
        assert "model a local override/displacement boolean" in prompt
        assert (
            "Do not encode those `Subject to ...` qualifiers as helper names like `*_permits_*`"
            in prompt
        )

    def test_build_eval_prompt_for_subject_to_unavailable_imports_allows_paragraph_specific_inputs(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A/2/e",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Subject to paragraphs (3), (4) and (4A), “earnings” in the case of "
                "employment as an employed earner, means any remuneration or profit "
                "derived from that employment and includes any payment by way of a retainer."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A/2/e",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A-2-e.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "When canonical imports for those cited paragraphs are available in the workspace, import them."
            in prompt
        )
        assert (
            "paragraph-specific local inputs are acceptable for an isolated slice artifact"
            in prompt
        )
        assert (
            "preserve the cited paragraph numbers and the branch-specific legal effect"
            in prompt
        )

    def test_build_eval_prompt_for_missing_cross_reference_exception_requires_defer(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="26 USC 45A(d)",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Paragraph (1) shall not apply to a transaction to which section "
                "381(a) applies if the employee continues to be employed by the "
                "acquiring corporation."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="repo-augmented",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 45A(d)",
            "repo-augmented",
            workspace,
            [],
            target_file_name="d.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Missing cited RuleSpec sources detected" in prompt
        assert "`us:statutes/26/381/a`" in prompt
        assert "Do not create local facts such as" in prompt
        assert "`section_381_a...`" in prompt
        assert (
            "emit `module.status: deferred` or `module.status: entity_not_supported`"
            in prompt
        )
        assert "leave any tests" in prompt
        assert "deferred surface empty" in prompt
        assert "module.deferred_outputs[]" in prompt
        assert "absolute `output` and `blocked_by` targets" in prompt
        assert "copied child output" in prompt
        assert "parent composition" in prompt

    def test_build_eval_prompt_for_missing_definition_dependency_requires_defer(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="26 USC 45A(e)",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "The term wages has the same meaning given to such term in "
                "section 51. All employers treated as a single employer under "
                "section 52 shall be treated as a single employer for purposes "
                "of this section."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="repo-augmented",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 45A(e)",
            "repo-augmented",
            workspace,
            [],
            target_file_name="e.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Missing cited RuleSpec sources detected" in prompt
        assert "`us:statutes/26/51`" in prompt
        assert "`us:statutes/26/52`" in prompt
        assert "same-meaning" in prompt
        assert "treated-as" in prompt
        assert (
            "emit `module.status: deferred` or `module.status: entity_not_supported`"
            in prompt
        )
        assert "omit or defer only the" in prompt
        assert "blocked surface" in prompt

    def test_build_eval_prompt_for_proration_tests_prefers_exact_division(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="26 USC 45A(e)(5)",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "For any taxable year having less than 12 months, the amount "
                "shall be multiplied by a fraction, the numerator of which is "
                "the number of days in the taxable year and the denominator of "
                "which is 365."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 45A(e)(5)",
            "cold",
            workspace,
            [],
            target_file_name="5.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "For proration tests with a source-stated denominator" in prompt
        assert "choose input amounts divisible by that denominator" in prompt
        assert "36500 * 182 / 365 = 18200" in prompt

    def test_build_eval_prompt_for_pure_cross_reference_computation_preserves_distinct_cited_alternatives(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "In the case of the earnings of self-employed earners, the amounts "
                "specified in paragraph (10) shall be taken into account in accordance "
                "with paragraph (4) or, as the case may be, paragraph (10) of regulation "
                "13 of the Computation of Earnings Regulations, as having effect in the "
                "case of state pension credit."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "do not replace the cited computation with local boolean `*_route_is_satisfied` or `*_fact` placeholders"
            in prompt
        )
        assert "do not emit a top-level `status: deferred` stub" in prompt
        assert (
            "do not collapse those cited alternatives into one generic treatment gate"
            in prompt
        )
        assert (
            "preserve the distinct cited alternatives with paragraph-specific imports or local facts/amounts"
            in prompt
        )
        assert "do not invent an extra `no treatment applies` branch" in prompt
        assert (
            "do not make the cited route-selection flags part of whether the paragraph itself applies"
            in prompt
        )
        assert (
            "do not encode the consequence as an unqualified `if paragraph_4_route: paragraph_4_amount else: paragraph_10_amount`"
            in prompt
        )
        assert (
            "Paragraph (10) must be selected by a paragraph-(10) route fact/import or by a derived paragraph-(10) route helper"
            in prompt
        )
        assert "prefer a single mutually exclusive route selector" in prompt
        assert (
            "Do not expose two independent route booleans that allow both routes or neither route to be selected"
            in prompt
        )
        assert "a safe local-placeholder shape is" in prompt
        assert (
            "paragraph-(10) route is derived as the applicable paragraph with not paragraph-(4) route"
            in prompt
        )
        assert "Do not create an invalid-route output branch that returns `0`" in prompt
        assert "self-employed earnings trigger the paragraph" in prompt
        assert (
            "regulation 13 paragraph (4) or paragraph (10) chooses the accounting route"
            in prompt
        )
        assert (
            "avoid a false case that makes a self-employed-earner branch fail merely because neither local route flag was selected"
            in prompt
        )
        assert "include separate cases for the distinct cited alternatives" in prompt

    def test_build_eval_prompt_requires_calendar_date_test_periods(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/13A/3/b",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-03-21.\n\n"
                "The amount of the guarantee credit payable in respect of the part-week "
                "shall be determined by multiplying the resulting figure by the number "
                "of days in the part-week."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/13A/3/b",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-13A-3-b.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Use concrete ISO calendar dates like `2025-03-21`" in prompt
        assert "do not use ISO week strings like `2025-W13`" in prompt

    def test_build_eval_prompt_for_uk_leaf_forbids_speculative_future_period_tests(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("claude:opus"),
            output_root=tmp_path / "out",
            source_text="Editorial note: current text valid from 2025-04-07.\n26.05",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
        )

        assert "The test file must contain YAML only" in prompt
        assert "must be a YAML list beginning with `- name:` entries" in prompt
        assert "Do not add speculative future-period tests" in prompt
        assert (
            "Use factual predicates or quantities in `input:`, not the output variable being asserted"
            in prompt
        )
        assert "Use concrete scalar values, not formula strings" in prompt
        assert "Use `period`, `input`, and `output` keys" in prompt

    def test_build_eval_prompt_for_uk_branch_leaves_requires_branch_specific_names(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/6",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) 332.95 per week in the case of a claimant who has a partner.",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/6",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-6.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "encode the branch identity in the output variable name" in prompt
        assert "principal output variable must encode that deepest token" in prompt
        assert "`standard_minimum_guarantee`" in prompt
        assert "`child_benefit_weekly_rate`" in prompt

    def test_build_eval_prompt_for_where_must_clauses_requires_inapplicable_case(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/4A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Where the young person is aged 19, he or she must have started the education "
                "or training before reaching that age."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/4A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-4A.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Where X, Y must ..." in prompt
        assert "Include a `.test.yaml` case where `X` is false" in prompt

    def test_build_eval_prompt_for_uk_leaf_discourages_opaque_condition_helpers(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) ... only person or elder or eldest person ... £26.05.",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "avoid opaque placeholders like `*_condition`" in prompt
        assert "`child_benefit_is_only_person`" in prompt
        assert "`claimant_has_partner`" in prompt
        assert "`is_joint_claimant`" in prompt

    def test_build_eval_prompt_for_single_row_fixed_amount_discourages_placeholder_names_and_applies_helpers(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2013/376/regulation/36",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-04-07.\n\n"
                "Structured table:\n"
                "Element | Amount for each assessment period\n"
                "single claimant aged under 25 | £316.98"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2013/376/regulation/36",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2013-376-regulation-36.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Use a descriptive legal variable name" in prompt
        assert "not a path- or source-id-derived placeholder" in prompt
        assert "do not invent a fresh `*_applies` helper" in prompt
        assert "do not invent alternate zero-amount tests" in prompt
        assert "Do not emit `otherwise:`" in prompt
        assert "Do not emit `before YYYY-MM-DD: 0`" in prompt
        assert "Do not emit malformed date blocks like `from 0:`" in prompt
        assert "use boolean or fact-shaped helper inputs" in prompt
        assert "Do not invent sample ages like `2`, `3`, `24`, or `25`" in prompt
        assert "keep `.test.yaml` outputs scalar" in prompt
        assert "keep the row-defining conditions satisfied" in prompt
        assert "principal amount rule should usually be a grounded constant" in prompt
        assert "Do not include `alternate_branch_*` tests" in prompt
        assert "write `2500`, not `2,500`" in prompt

    def test_build_eval_prompt_requires_named_scalars_for_repeated_and_threshold_numbers(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/schedule/VI/2A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Where a person is engaged in employments specified in paragraph 2 but "
                "his earnings are less than £20 and he is also engaged in other employment, "
                "so much of his other earnings as would not exceed £20. "
                "A non-dependant aged 18 or over is treated differently. "
                "See section 3(4)."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/schedule/VI/2A",
            "cold",
            workspace,
            [],
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "Every substantive numeric occurrence in `./source.txt` must be represented by a named scalar definition in RuleSpec"
            in prompt
        )
        assert (
            "If the same numeric value appears twice in materially different legal roles"
            in prompt
        )
        assert (
            "reuse that named scalar everywhere the rule compares against or computes with that number"
            in prompt
        )
        assert "Do not simplify source-stated ratios or fractions" in prompt
        assert "ungrounded decimal such as `0.10`" in prompt
        assert (
            'If `./source.txt` says someone is "aged 18 or over", "under 25"' in prompt
        )
        assert "Do not create scalar variables for citation numbers" in prompt
        assert (
            "Do not invent `dtype: String` variables just to restate the effective date"
            in prompt
        )
        assert "Axiom formulas have no date literal type" in prompt
        assert "Do not put the date or year value in the fact name" in prompt
        assert "taxable_year_begins_after_termination_date" in prompt
        assert "`taxable_year_begins_after_2024_and_before_2029` or" in prompt
        assert (
            "Never use `post_YYYY`, `pre_YYYY`, `after_YYYY`, `before_YYYY`" in prompt
        )
        assert "overrides preservation of existing local input names" in prompt
        assert (
            "Do not decompose legal dates into numeric `year`, `month`, or `day` scalar variables"
            in prompt
        )
        assert "module.summary` or the rule's proof excerpt" in prompt
        assert "exact source phrase containing that number" in prompt
        assert "`==` for equality" in prompt

    def test_prepare_eval_workspace_injects_resolved_defined_term_stub(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/7A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="A person who is a member of a mixed-age couple is not entitled.",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        definition_files = [
            item for item in workspace.context_files if item.kind == "definition_stub"
        ]
        assert len(definition_files) == 1
        assert (
            definition_files[0].workspace_path
            == "context/legislation/ukpga/2002/16/section/3ZA/3.yaml"
        )
        assert (
            definition_files[0].import_path == "legislation/ukpga/2002/16/section/3ZA/3"
        )
        stub_path = workspace.root / definition_files[0].workspace_path
        assert stub_path.exists()
        assert "is_member_of_mixed_age_couple" in stub_path.read_text()

    def test_prepare_eval_workspace_copies_resolved_canonical_concept_file(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us-co")
        concept_file = policy_repo_root / "statutes" / "crs" / "26-2-703" / "12.yaml"
        concept_file.parent.mkdir(parents=True, exist_ok=True)
        concept_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    C.R.S. § 26-2-703(12)
    Definitions

    "Individual responsibility contract" or "IRC" means the contract entered into by the participant and the county department pursuant to section 26-2-708.
rules:
  - name: is_individual_responsibility_contract
    kind: input
    entity: Person
    dtype: Boolean
    period: Month
"""
        )

        workspace = prepare_eval_workspace(
            citation="co/regulation/3.609.1/A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="The participant must comply with the individual responsibility contract.",
            axiom_rules_path=policy_repo_root,
            mode="cold",
            extra_context_paths=[],
        )

        concept_files = [
            item for item in workspace.context_files if item.kind == "canonical_concept"
        ]
        assert len(concept_files) == 1
        assert (
            concept_files[0].workspace_path == "context/statutes/crs/26-2-703/12.yaml"
        )
        assert concept_files[0].import_path == "us-co:statutes/crs/26-2-703/12"
        copied_path = workspace.root / concept_files[0].workspace_path
        assert copied_path.exists()
        assert "is_individual_responsibility_contract" in copied_path.read_text()

    def test_hydrate_eval_root_places_resolved_definition_stub_under_runner_root(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/7A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="A person who is a member of a mixed-age couple is not entitled.",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        runner_root = tmp_path / "case" / "openai-gpt-5.4"
        source_dir = runner_root / "source"
        source_dir.mkdir(parents=True)
        (source_dir / "example.yaml").write_text("format: rulespec/v1\nrules: []\n")

        _hydrate_eval_root(runner_root, workspace)

        hydrated = (
            runner_root
            / "legislation"
            / "ukpga"
            / "2002"
            / "16"
            / "section"
            / "3ZA"
            / "3.yaml"
        )
        assert hydrated.exists()
        assert "is_member_of_mixed_age_couple" in hydrated.read_text()

    def test_build_eval_prompt_includes_resolved_defined_term_guidance(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/7A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="A person who is a member of a mixed-age couple is not entitled.",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/7A",
            "cold",
            workspace,
            workspace.context_files,
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Resolved definition files are available below." in prompt
        assert "mixed-age couple" in prompt
        assert (
            "legislation/ukpga/2002/16/section/3ZA/3#is_member_of_mixed_age_couple"
            in prompt
        )
        assert (
            "import that canonical definition instead of inventing a leaf-local helper"
            in prompt
        )
        assert "Do not replace that import with a local deferred stub" in prompt
        assert (
            "Do not encode such local factual predicates as placeholder constants like `true` or `false`."
            in prompt
        )
        assert (
            "Do not encode such local factual predicates as `status: deferred`"
            in prompt
        )

    def test_build_eval_prompt_includes_resolved_canonical_concept_guidance(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us-co")
        concept_file = policy_repo_root / "statutes" / "crs" / "26-2-703" / "12.yaml"
        concept_file.parent.mkdir(parents=True, exist_ok=True)
        concept_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    C.R.S. § 26-2-703(12)
    Definitions

    "Individual responsibility contract" or "IRC" means the contract entered into by the participant and the county department pursuant to section 26-2-708.
rules:
  - name: is_individual_responsibility_contract
    kind: input
    entity: Person
    dtype: Boolean
    period: Month
"""
        )

        workspace = prepare_eval_workspace(
            citation="co/regulation/3.609.1/A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="The participant must comply with the individual responsibility contract.",
            axiom_rules_path=policy_repo_root,
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "co/regulation/3.609.1/A",
            "cold",
            workspace,
            workspace.context_files,
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "Resolved canonical concept files from this corpus are available below."
            in prompt
        )
        assert "individual responsibility contract" in prompt
        assert (
            "statutes/crs/26-2-703/12#is_individual_responsibility_contract" in prompt
        )
        assert (
            "import or re-export that exact canonical concept instead of duplicating it locally"
            in prompt
        )

    def test_build_eval_prompt_includes_import_vs_local_helper_protocol(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="26 USC 24(c)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text='The term "qualifying child" means a qualifying child as defined in section 152(c).',
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 24(c)",
            "cold",
            workspace,
            [],
            target_file_name="24-c.yaml",
            include_tests=True,
        )

        assert (
            "emit the upstream import instead of restating the concept locally"
            in prompt
        )
        assert "already executable" in prompt
        assert "do not replace it with" in prompt
        assert "requested source itself defines a legal status or test" in prompt
        assert "IRC section 112" not in prompt
        assert "Hard requirement for IRC section 112" not in prompt
        assert "same concept or output name" in prompt
        assert "directly rounded final amount table" in prompt
        assert "round the" in prompt
        assert "increase before adding it to the base amount" in prompt
        assert "17300, not 17325" in prompt
        assert "Outputs named `taxable_income`" in prompt
        assert "if condition: max(0, branch_a) else: max(0, branch_b)" in prompt
        assert "rate * min(max(0, earned_income), cap)" in prompt
        assert "says a value is determined `in accordance with section X`" in prompt
        assert "do not invent `import` statements or `imports:` blocks" in prompt
        assert "Importing a child rate or threshold is not enough" in prompt
        assert "`to the extent`" in prompt
        assert "all-or-nothing zeroing" in prompt
        assert "current\n  requested source changes the basis" in prompt
        assert "internally handled its own `to the extent` exclusion" in prompt

    def test_build_eval_prompt_highlights_cited_context_import_exports(self, tmp_path):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        cited_file = policy_repo_root / "statutes" / "26" / "1211.yaml"
        cited_file.parent.mkdir(parents=True, exist_ok=True)
        cited_file.write_text(
            """format: rulespec/v1
rules:
  - name: other_taxpayer_capital_losses_allowed
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: allowed_capital_losses
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 1222",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "The term net capital loss means the excess of the losses from "
                "sales or exchanges of capital assets over the sum allowed under "
                "section 1211."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[cited_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 1222",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="1222.yaml",
            target_ref_prefix="us:statutes/26/1222",
            include_tests=True,
        )

        assert "Mandatory cited RuleSpec imports detected from source text" in prompt
        assert "Source cites `1211`" in prompt
        assert "`us:statutes/26/1211#other_taxpayer_capital_losses_allowed`" in prompt
        assert "Do not keep a local `_under_section_...`" in prompt

    def test_build_eval_prompt_treats_in_lieu_citation_as_displaced_context(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        cited_file = policy_repo_root / "statutes" / "26" / "164" / "f.yaml"
        cited_file.parent.mkdir(parents=True)
        cited_file.write_text(
            """format: rulespec/v1
rules:
  - name: self_employment_tax_deduction
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: self_employment_tax * 0.5
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 1402(a)(12)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "In lieu of the deduction provided by section 164(f), there "
                "shall be allowed a deduction equal to the product of the "
                "taxpayer's net earnings from self-employment and one-half of "
                "the rates imposed by section 1401."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[cited_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 1402(a)(12)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="12.yaml",
            target_ref_prefix="us:statutes/26/1402/a/12",
            include_tests=True,
        )

        assert "Source cites `164(f)`" in prompt
        assert "displacement or replacement phrase" in prompt
        assert "Do not import the cited final amount" in prompt
        assert "prefer the final imported output" not in prompt

    def test_build_eval_prompt_guides_excluded_child_branch_imports(self, tmp_path):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        parent_file = policy_repo_root / "statutes" / "26" / "1401.yaml"
        child_a_file = policy_repo_root / "statutes" / "26" / "1401" / "a.yaml"
        child_b1_file = policy_repo_root / "statutes" / "26" / "1401" / "b" / "1.yaml"
        parent_file.parent.mkdir(parents=True, exist_ok=True)
        child_a_file.parent.mkdir(parents=True, exist_ok=True)
        child_b1_file.parent.mkdir(parents=True, exist_ok=True)
        parent_file.write_text(
            """format: rulespec/v1
rules:
  - name: self_employment_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: old_age_survivors_and_disability_insurance_tax + self_employment_income_tax + additional_tax
"""
        )
        child_a_file.write_text(
            """format: rulespec/v1
rules:
  - name: old_age_survivors_and_disability_insurance_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: limited_income * rate
"""
        )
        child_b1_file.write_text(
            """format: rulespec/v1
rules:
  - name: self_employment_income_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: income * rate
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 164(f)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "There shall be allowed a deduction equal to one-half of the "
                "taxes imposed by section 1401 (other than the taxes imposed by "
                "section 1401(b)(2))."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[parent_file, child_a_file, child_b1_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 164(f)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="f.yaml",
            target_ref_prefix="us:statutes/26/164/f",
            include_tests=True,
        )

        assert "Excluded cited child branch guidance" in prompt
        assert "excludes `1401(b)(2)`" in prompt
        assert (
            "`us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax`"
            in prompt
        )
        assert "`us:statutes/26/1401/b/1#self_employment_income_tax`" in prompt
        assert "do not import an ancestor aggregate" in prompt

    def test_build_eval_prompt_highlights_terminal_child_exports(self, tmp_path):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        child_file = policy_repo_root / "statutes" / "26" / "3101" / "b" / "2.yaml"
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
rules:
  - name: additional_medicare_tax_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2013-01-01'
        formula: 0.009
  - name: additional_medicare_excess_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2013-01-01'
        formula: max(0, wages - additional_medicare_wage_tax_threshold)
  - name: additional_medicare_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2013-01-01'
        formula: additional_medicare_excess_wages * additional_medicare_tax_rate
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 3101",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Section 3101 imposes the taxes described in subsection (b)(2).",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 3101",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="3101.yaml",
            target_ref_prefix="us:statutes/26/3101",
            include_tests=True,
        )

        assert (
            "terminal exports `us:statutes/26/3101/b/2#additional_medicare_tax`"
            in prompt
        )
        assert "Aggregate parent child outputs detected" in prompt
        assert (
            "`us:statutes/26/3101/b/2#additional_medicare_tax`"
            " (derived, Money, TaxUnit)" in prompt
        )
        assert "Do not rebuild a child branch in the parent" in prompt
        assert "Do not manufacture a parent-level `Judgment` output" in prompt
        assert (
            "pass-through, conjunction, or disjunction of imported child `Judgment`"
            in prompt
        )

    def test_build_eval_prompt_requires_child_exception_imports_for_parent_list(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        child_file = (
            policy_repo_root
            / "statutes"
            / "26"
            / "163"
            / "h"
            / "4"
            / "B"
            / "ii"
            / "I.yaml"
        )
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
module:
  summary: Such term shall not include a loan to finance fleet sales.
rules:
  - name: fleet_sales_loan_exception_applies
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
    versions:
      - effective_from: '2025-01-01'
        formula: loan_finances_fleet_sales
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 163(h)(4)(B)",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "The term qualified passenger vehicle loan interest means "
                "interest paid on qualifying indebtedness. Such term shall not "
                "include any amount paid or incurred on any of the following:"
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 163(h)(4)(B)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="B.yaml",
            target_ref_prefix="us:statutes/26/163/h/4/B",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Parent exception-list child fragments detected" in prompt
        assert (
            "`us:statutes/26/163/h/4/B/ii/I#fleet_sales_loan_exception_applies`"
            in prompt
        )
        assert "Import each listed child exception output" in prompt
        assert "This overrides the usual small-test-count preference" in prompt
        assert "one blocking companion test" in prompt
        assert "for each listed child exception output" in prompt

    def test_build_eval_prompt_forces_partial_extent_child_parent_defer(self, tmp_path):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        child_file = policy_repo_root / "statutes" / "26" / "3101" / "a.yaml"
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
rules:
  - name: oasdi_wage_tax_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '1990-01-01'
        formula: 0.062
  - name: oasdi_wage_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: wages * oasdi_wage_tax_rate
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 3101",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Wages shall be exempt from the taxes imposed by this section "
                "to the extent that such wages are subject exclusively to "
                "another country's social security laws."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 3101",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="3101.yaml",
            target_ref_prefix="us:statutes/26/3101",
            include_tests=True,
        )

        assert "Target-specific schema limit" in prompt
        assert "`us:statutes/26/3101/a#oasdi_wage_tax`" in prompt
        assert "entity_not_supported" in prompt
        assert "`rules: []`" in prompt
        assert "`*_before_exemption`" in prompt

    def test_build_eval_prompt_does_not_defer_parent_for_child_internal_extent(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        child_file = policy_repo_root / "statutes" / "26" / "32" / "c" / "2.yaml"
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    Earned income excludes subsidized work-activity amounts only to the extent
    subsidized under the State program.
rules:
  - name: earned_income
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 32(c)(2)
    versions:
      - effective_from: '1990-01-01'
        formula: max(0, wages - subsidized_state_work_activity_service_compensation)
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 32",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "(a) In the case of an eligible individual, there shall be "
                "allowed as a credit an amount equal to the credit percentage "
                "of so much of the taxpayer's earned income as does not exceed "
                "the earned income amount. (c)(2) The term earned income means "
                "employee compensation plus self-employment income, and no "
                "amount received for work activities shall be taken into "
                "account, but only to the extent such amount is subsidized "
                "under such State program."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 32",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="32.yaml",
            target_ref_prefix="us:statutes/26/32",
            include_tests=True,
        )

        assert "Target-specific schema limit" not in prompt
        assert "Aggregate parent child outputs detected" in prompt
        assert "`us:statutes/26/32/c/2#earned_income`" in prompt
        assert "internally handled its own `to the extent` exclusion" in prompt

    def test_build_eval_prompt_does_not_defer_amount_adjustment_parent_list(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us-co")
        child_file = policy_repo_root / "statutes" / "39" / "39-22-104" / "4" / "i.yaml"
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
rules:
  - name: qualifying_income_subtraction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: income_included_in_federal_taxable_income
"""
        )
        workspace = prepare_eval_workspace(
            citation="us-co/statute/39/39-22-104/4",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "(4) There shall be subtracted from federal taxable income:\n"
                "(i) Qualifying income to the extent included in federal "
                "taxable income and exempt from taxes imposed by this article."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "us-co/statute/39/39-22-104/4",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="4.yaml",
            target_ref_prefix="us-co:statutes/39/39-22-104/4",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Target-specific schema limit" not in prompt
        assert "Aggregate parent child outputs detected" in prompt
        assert (
            "`us-co:statutes/39/39-22-104/4/i#qualifying_income_subtraction`" in prompt
        )

    def test_build_eval_prompt_does_not_defer_taxable_income_for_incidental_extent(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        child_file = policy_repo_root / "statutes" / "26" / "63" / "c.yaml"
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
rules:
  - name: standard_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: basic_standard_deduction
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 63",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "Taxable income means gross income minus deductions. Unless an "
                "individual elects to itemize deductions, taxable income means "
                "adjusted gross income minus the standard deduction. Marital "
                "status is determined in accordance with section 7703. The "
                "taxpayer and spouse consent to assessment of any deficiency to "
                "the extent attributable to such change of election."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 63",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="63.yaml",
            target_ref_prefix="us:statutes/26/63",
            include_tests=True,
        )

        assert "Target-specific schema limit" not in prompt
        assert "Taxpayer elections such as electing to itemize deductions" in prompt
        assert "Outputs named `taxable_income`" in prompt

    def test_build_eval_prompt_scopes_partial_extent_to_target_paragraph(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us-co")
        child_file = (
            policy_repo_root / "statutes" / "39" / "39-22-104" / "3" / "p" / "5.yaml"
        )
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
rules:
  - name: initial_window_addition_to_federal_taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2023-01-01'
        formula: federal_deduction_addition
"""
        )
        workspace = prepare_eval_workspace(
            citation="us-co/statute/39/39-22-104/3/p",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "(3) There shall be added to federal taxable income:\n"
                "(p) Except as otherwise provided in subsection (3)(p.5), "
                "for income tax years commencing on or after January 1, 2022, "
                "for taxpayers who claim itemized deductions and have federal "
                "adjusted gross income equal to or exceeding four hundred "
                "thousand dollars: (I) For a taxpayer who files a single "
                "return, the amount by which itemized deductions exceed thirty "
                "thousand dollars; and (II) For taxpayers who file a joint "
                "return, the amount by which itemized deductions exceed sixty "
                "thousand dollars.\n"
                "(p.5) For income tax years commencing on or after January 1, "
                "2023, a different addition applies.\n"
                "(4)(i) A subtraction is allowed to the extent included in "
                "federal taxable income."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "us-co/statute/39/39-22-104/3/p",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="p.yaml",
            target_ref_prefix="us-co:statutes/39/39-22-104/3/p",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Target-specific schema limit" not in prompt
        assert "except as otherwise provided in section" in prompt
        assert "If copied context\n  for the cited source is present" in prompt
        assert "do not preserve, rename, or recreate the local" in prompt
        assert "follow the copied-context rule above instead" in prompt

    def test_build_eval_prompt_no_tests_includes_copied_context_boundary_rule(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us-co")
        child_file = (
            policy_repo_root / "statutes" / "39" / "39-22-104" / "3" / "p" / "5.yaml"
        )
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
rules:
  - name: initial_window_addition_to_federal_taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2023-01-01'
        formula: federal_deduction_addition
"""
        )
        workspace = prepare_eval_workspace(
            citation="us-co/statute/39/39-22-104/3/p",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "(p) Except as otherwise provided in subsection (3)(p.5), "
                "for taxpayers who claim itemized deductions, the amount by "
                "which itemized deductions exceed thirty thousand dollars."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "us-co/statute/39/39-22-104/3/p",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="p.yaml",
            target_ref_prefix="us-co:statutes/39/39-22-104/3/p",
            runner_backend="openai",
        )

        assert "Test file rules:" not in prompt
        assert "do not preserve, rename, or recreate the local" in prompt
        assert "follow the copied-context rule above instead" in prompt

    def test_build_eval_prompt_scopes_partial_extent_to_numeric_target_paragraph(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        child_file = policy_repo_root / "statutes" / "26" / "999" / "1" / "a.yaml"
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
rules:
  - name: child_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: taxable_amount * rate
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 999(1)",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "(1) First rule imposes a tax.\n"
                "(3) Wages are exempt from taxes imposed by this section "
                "to the extent such wages are subject exclusively to another "
                "country's social security laws."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 999(1)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="1.yaml",
            target_ref_prefix="us:statutes/26/999/1",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Target-specific schema limit" not in prompt

    def test_build_eval_prompt_recommends_final_deduction_imports(self, tmp_path):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        cited_file = policy_repo_root / "statutes" / "26" / "170" / "p.yaml"
        cited_file.parent.mkdir(parents=True, exist_ok=True)
        cited_file.write_text(
            """format: rulespec/v1
rules:
  - name: nonitemizer_charitable_deduction_cap
    kind: parameter
    dtype: Money
    period: Year
    values:
      2026-01-01: 1000
  - name: nonitemizer_charitable_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: min(charitable_contributions, nonitemizer_charitable_deduction_cap)
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 63",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Taxable income is adjusted gross income minus any deduction "
                "provided in section 170(p)."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[cited_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 63",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="63.yaml",
            target_ref_prefix="us:statutes/26/63",
            include_tests=True,
        )

        assert "For the cited deduction/exemption/credit reference" in prompt
        assert "`us:statutes/26/170/p#nonitemizer_charitable_deduction`" in prompt
        assert "`*_provided_in_section_*`" in prompt

    def test_build_eval_prompt_discourages_fabricated_same_instrument_imports(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/6/5/a",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) except where paragraph (b) applies, £81.50 per week if paragraph 1(1)(a), (b) or (c) of Part I of Schedule I is satisfied.",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/6/5/a",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-6-5-a.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Do not fabricate sibling-file imports" in prompt
        assert "do not guess" in prompt
        assert "do not invent `import` statements or `imports:` blocks" in prompt

    def test_build_eval_prompt_for_openai_inlines_source_text(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Editorial note: current text valid from 2025-04-07.\n26.05",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "You do not have filesystem tool access in this eval" in prompt
        assert "=== BEGIN SOURCE.TXT ===" in prompt
        assert "Editorial note: current text valid from 2025-04-07." in prompt
        assert "26.05" in prompt

    def test_build_eval_prompt_for_date_silent_source_includes_neutral_scaffold_fallback(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1(E)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Applications received will be certified for six (6) consecutive months "
                "beginning the first month the assistance unit is found eligible for basic cash assistance."
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "9 CCR 2503-6 3.606.1(E)",
            "cold",
            workspace,
            [],
            target_file_name="9-CCR-2503-6-3.606.1-E.yaml",
            include_tests=True,
            runner_backend="codex",
        )

        assert "effective_from: '0001-01-01'" in prompt
        assert "harness-only fallback" in prompt


class TestOpenAIEvalRequest:
    @pytest.mark.parametrize(
        ("model", "expected_max_output_tokens", "explicit_cache"),
        [("gpt-5.6-sol", 32768, True), ("gpt-4o", 16384, False)],
        ids=["extended-gpt-5", "compatible-fallback"],
    )
    def test_openai_prompt_eval_requests_full_rulespec_output_budget(
        self,
        monkeypatch,
        model,
        expected_max_output_tokens,
        explicit_cache,
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        ok_response = Mock(status_code=200, headers={}, text="")
        ok_response.json.return_value = {
            "output_text": "format: rulespec/v1\nrules: []\n",
            "usage": {},
        }

        with patch(
            "axiom_encode.harness.evals._post_openai_eval_request",
            return_value=ok_response,
        ) as mock_post:
            response = evals_module._run_openai_prompt_eval(
                parse_runner_spec(f"openai:{model}"),
                SimpleNamespace(),
                "encode the complete provision",
            )

        prompt = "encode the complete provision"
        prompt_input, stable_prefix, _ = evals_module._openai_prompt_input(
            model,
            prompt,
        )
        expected_body = {
            "model": model,
            "input": prompt_input,
            "max_output_tokens": expected_max_output_tokens,
            "reasoning": {"effort": "low", "summary": "auto"},
            "prompt_cache_key": evals_module._openai_prompt_cache_key(
                model,
                stable_prefix,
            ),
        }
        if explicit_cache:
            expected_body["prompt_cache_options"] = {"mode": "explicit"}
            expected_body["service_tier"] = "default"
        assert mock_post.call_args.kwargs["body"] == expected_body
        assert response.trace["request_body"] == expected_body
        assert response.error is None

    def test_openai_prompt_eval_records_cache_reads_and_writes(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        ok_response = Mock(status_code=200, headers={}, text="")
        ok_response.json.return_value = {
            "output_text": "format: rulespec/v1\nrules: []\n",
            "usage": {
                "input_tokens": 4096,
                "output_tokens": 100,
                "input_tokens_details": {
                    "cached_tokens": 3072,
                    "cache_write_tokens": 1024,
                },
            },
        }

        with patch(
            "axiom_encode.harness.evals._post_openai_eval_request",
            return_value=ok_response,
        ):
            response = evals_module._run_openai_prompt_eval(
                parse_runner_spec("openai:gpt-5.6-terra"),
                SimpleNamespace(),
                "stable prompt",
            )

        assert response.tokens is not None
        assert response.tokens.cache_read_tokens == 3072
        assert response.tokens.cache_creation_tokens == 1024

    def test_post_openai_eval_request_retries_transient_status(self):
        error_response = Mock()
        error_response.status_code = 502
        ok_response = Mock()
        ok_response.status_code = 200

        with (
            patch("axiom_encode.harness.evals.requests.post") as mock_post,
            patch("axiom_encode.harness.evals.time.sleep"),
        ):
            mock_post.side_effect = [error_response, ok_response]

            response = _post_openai_eval_request(
                headers={"Authorization": "Bearer test"},
                body={"model": "gpt-5.4", "input": "hi"},
            )

        assert response is ok_response
        assert mock_post.call_count == 2

    def test_post_openai_eval_request_retries_request_exception(self):
        ok_response = Mock()
        ok_response.status_code = 200

        with (
            patch("axiom_encode.harness.evals.requests.post") as mock_post,
            patch("axiom_encode.harness.evals.time.sleep"),
        ):
            mock_post.side_effect = [
                requests.exceptions.ReadTimeout("timed out"),
                ok_response,
            ]

            response = _post_openai_eval_request(
                headers={"Authorization": "Bearer test"},
                body={"model": "gpt-5.4", "input": "hi"},
            )

        assert response is ok_response
        assert mock_post.call_count == 2

    def test_post_openai_eval_request_uses_bound_backoff_policy(self, monkeypatch):
        error_response = Mock(status_code=503)
        ok_response = Mock(status_code=200)
        monkeypatch.setattr(
            evals_module,
            "_OPENAI_REQUEST_BACKOFF_SECONDS",
            (7, 11),
        )

        with (
            patch(
                "axiom_encode.harness.evals.requests.post",
                side_effect=[error_response, error_response, ok_response],
            ),
            patch("axiom_encode.harness.evals.time.sleep") as mock_sleep,
        ):
            response = _post_openai_eval_request(
                headers={"Authorization": "Bearer test"},
                body={"model": "gpt-5.4", "input": "hi"},
                attempts=3,
            )

        assert response is ok_response
        assert [item.args[0] for item in mock_sleep.call_args_list] == [7, 11]

    def test_post_openai_eval_request_is_bounded_by_case_deadline(
        self,
        monkeypatch,
    ):
        clock = [100.0]
        monkeypatch.setattr(evals_module.time, "monotonic", lambda: clock[0])

        def exhaust_deadline(*_args, **kwargs):
            assert kwargs["timeout"] == (5.0, 5.0)
            clock[0] = 106.0
            raise requests.exceptions.ReadTimeout("timed out")

        deadline_token = evals_module._EVAL_CASE_DEADLINE_MONOTONIC.set(105.0)
        timeout_token = evals_module._EVAL_CASE_TIMEOUT_SECONDS.set(5)
        try:
            with (
                patch(
                    "axiom_encode.harness.evals.requests.post",
                    side_effect=exhaust_deadline,
                ) as mock_post,
                patch("axiom_encode.harness.evals.time.sleep") as mock_sleep,
                pytest.raises(requests.Timeout) as exc_info,
            ):
                _post_openai_eval_request(
                    headers={"Authorization": "Bearer test"},
                    body={"model": "gpt-5.4", "input": "hi"},
                    attempts=3,
                )
        finally:
            evals_module._EVAL_CASE_TIMEOUT_SECONDS.reset(timeout_token)
            evals_module._EVAL_CASE_DEADLINE_MONOTONIC.reset(deadline_token)

        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()
        assert exc_info.value.timeout_stage == "case_budget"
        assert exc_info.value.timeout_reason == "wall"
        assert exc_info.value.timeout_seconds == 5
        assert exc_info.value.timeout_attempts == 1

    def test_post_openai_eval_request_returns_at_case_wall_deadline(self):
        release_request = threading.Event()

        def block_beyond_deadline(*_args, **_kwargs):
            release_request.wait(timeout=1)
            return Mock(status_code=200)

        started = evals_module.time.monotonic()
        try:
            with (
                evals_module._active_eval_case_budget(0.05),
                patch(
                    "axiom_encode.harness.evals.requests.post",
                    side_effect=block_beyond_deadline,
                ),
                pytest.raises(requests.Timeout) as exc_info,
            ):
                _post_openai_eval_request(
                    headers={"Authorization": "Bearer test"},
                    body={"model": "gpt-5.4", "input": "hi"},
                    attempts=1,
                )
        finally:
            release_request.set()

        assert evals_module.time.monotonic() - started < 0.5
        assert exc_info.value.timeout_stage == "case_budget"

    def test_post_openai_eval_request_rejects_response_after_case_deadline(
        self,
        monkeypatch,
    ):
        clock = [100.0]
        monkeypatch.setattr(evals_module.time, "monotonic", lambda: clock[0])
        ok_response = Mock(status_code=200)

        def return_after_deadline(*_args, **kwargs):
            assert kwargs["timeout"] == (5.0, 5.0)
            clock[0] = 106.0
            return ok_response

        deadline_token = evals_module._EVAL_CASE_DEADLINE_MONOTONIC.set(105.0)
        timeout_token = evals_module._EVAL_CASE_TIMEOUT_SECONDS.set(5)
        try:
            with (
                patch(
                    "axiom_encode.harness.evals.requests.post",
                    side_effect=return_after_deadline,
                ),
                pytest.raises(requests.Timeout) as exc_info,
            ):
                _post_openai_eval_request(
                    headers={"Authorization": "Bearer test"},
                    body={"model": "gpt-5.4", "input": "hi"},
                    attempts=1,
                )
        finally:
            evals_module._EVAL_CASE_TIMEOUT_SECONDS.reset(timeout_token)
            evals_module._EVAL_CASE_DEADLINE_MONOTONIC.reset(deadline_token)

        assert exc_info.value.timeout_stage == "case_budget"
        assert exc_info.value.timeout_reason == "wall"
        assert exc_info.value.timeout_seconds == 5
        assert exc_info.value.timeout_attempts == 1

    def test_wrapped_openai_wall_timeout_counts_the_terminal_request(
        self,
        monkeypatch,
    ):
        clock = [100.0]
        request_count = 0
        monkeypatch.setattr(evals_module.time, "monotonic", lambda: clock[0])

        def request_then_expire(**_kwargs):
            nonlocal request_count
            request_count += 1
            if request_count < 3:
                raise requests.exceptions.ReadTimeout("read timed out")
            clock[0] = 106.0
            raise requests.exceptions.ConnectionError("wrapped alarm timeout")

        deadline_token = evals_module._EVAL_CASE_DEADLINE_MONOTONIC.set(105.0)
        timeout_token = evals_module._EVAL_CASE_TIMEOUT_SECONDS.set(5)
        try:
            with (
                patch(
                    "axiom_encode.harness.evals._post_openai_request_with_wall_deadline",
                    side_effect=request_then_expire,
                ),
                patch("axiom_encode.harness.evals.time.sleep"),
                pytest.raises(requests.Timeout) as exc_info,
            ):
                _post_openai_eval_request(
                    headers={"Authorization": "Bearer test"},
                    body={"model": "gpt-5.4", "input": "hi"},
                    attempts=3,
                )
        finally:
            evals_module._EVAL_CASE_TIMEOUT_SECONDS.reset(timeout_token)
            evals_module._EVAL_CASE_DEADLINE_MONOTONIC.reset(deadline_token)

        assert request_count == 3
        assert exc_info.value.timeout_stage == "case_budget"
        assert exc_info.value.timeout_attempts == 3

    @pytest.mark.parametrize(
        ("error", "expected_reason", "expected_seconds"),
        [
            (requests.exceptions.ConnectTimeout("connect"), "connect", 30),
            (requests.exceptions.ReadTimeout("read"), "read", 180),
            (requests.exceptions.Timeout("ambiguous"), "request", None),
        ],
        ids=["connect", "read", "ambiguous"],
    )
    def test_openai_prompt_eval_records_triggering_timeout_policy(
        self,
        monkeypatch,
        error,
        expected_reason,
        expected_seconds,
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch(
            "axiom_encode.harness.evals._post_openai_eval_request",
            side_effect=error,
        ):
            response = evals_module._run_openai_prompt_eval(
                parse_runner_spec("openai:gpt-5.4"),
                SimpleNamespace(),
                "prompt",
            )

        assert response.timed_out is True
        assert response.timeout_stage == "encoder"
        assert response.timeout_reason == expected_reason
        assert response.timeout_seconds == expected_seconds
        assert response.trace["timeout_reason"] == expected_reason
        assert response.trace["timeout_seconds"] == expected_seconds

    def test_openai_prompt_eval_classifies_case_budget_timeout(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        error = requests.exceptions.Timeout("Eval case budget timed out")
        error.timeout_stage = "case_budget"
        error.timeout_reason = "wall"
        error.timeout_seconds = 17
        error.timeout_attempts = 2

        with patch(
            "axiom_encode.harness.evals._post_openai_eval_request",
            side_effect=error,
        ):
            response = evals_module._run_openai_prompt_eval(
                parse_runner_spec("openai:gpt-5.4"),
                SimpleNamespace(),
                "prompt",
            )

        assert response.timed_out is True
        assert response.timeout_stage == "case_budget"
        assert response.timeout_reason == "wall"
        assert response.timeout_seconds == 17
        assert response.timeout_attempts == 2
        assert response.trace["timeout_stage"] == "case_budget"

    def test_openai_prompt_eval_preserves_timeout_history_before_success(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        ok_response = Mock(
            status_code=200,
            headers={},
            text="",
        )
        ok_response.json.return_value = {
            "output_text": "format: rulespec/v1\nrules: []\n",
            "usage": {},
        }

        with (
            patch(
                "axiom_encode.harness.evals.requests.post",
                side_effect=[
                    requests.exceptions.ReadTimeout("read timed out"),
                    ok_response,
                ],
            ),
            patch("axiom_encode.harness.evals.time.sleep"),
        ):
            response = evals_module._run_openai_prompt_eval(
                parse_runner_spec("openai:gpt-5.4"),
                SimpleNamespace(),
                "prompt",
            )

        assert response.error is None
        assert response.timed_out is False
        assert response.timeout_attempts == 1
        assert response.timeout_stage == "encoder"
        assert response.timeout_reason == "read"
        assert response.timeout_seconds == 180
        assert response.trace["timeout_attempts"] == 1


def test_repeated_openai_timeouts_reach_durable_result_attempt_count(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path,
        "source states 451.",
    )
    request_timeouts = [
        requests.exceptions.ConnectTimeout("connection timed out")
        for _attempt in range(12)
    ]

    with (
        patch(
            "axiom_encode.harness.evals.requests.post",
            side_effect=request_timeouts,
        ) as mock_post,
        patch("axiom_encode.harness.evals.time.sleep"),
    ):
        [result] = run_source_eval(
            source_unit=source_unit,
            runner_specs=["openai:gpt-5.4"],
            output_root=tmp_path / "out",
            policy_path=policy_repo_root,
            local_corpus_release=corpus_release,
            runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
        )

    restored = _eval_result_from_payload(result.to_dict())
    assert mock_post.call_count == 12
    assert restored.failure_kind == "timeout"
    assert restored.timed_out is True
    assert restored.timeout_stage == "encoder"
    assert restored.timeout_reason == "connect"
    assert restored.timeout_seconds == 30
    assert restored.timeout_attempts == 12


def test_openai_timeouts_before_http_error_reach_durable_result(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path,
        "source states 451.",
    )
    unavailable_response = Mock(
        status_code=503,
        headers={},
        text="unavailable",
    )
    unavailable_response.json.return_value = {
        "error": {"message": "unavailable"},
    }

    with (
        patch(
            "axiom_encode.harness.evals.requests.post",
            side_effect=[
                *[
                    requests.exceptions.ConnectTimeout("connection timed out")
                    for _attempt in range(5)
                ],
                unavailable_response,
            ],
        ) as mock_post,
        patch("axiom_encode.harness.evals.time.sleep"),
    ):
        [result] = run_source_eval(
            source_unit=source_unit,
            runner_specs=["openai:gpt-5.4"],
            output_root=tmp_path / "out",
            policy_path=policy_repo_root,
            local_corpus_release=corpus_release,
            runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
        )

    restored = _eval_result_from_payload(result.to_dict())
    assert mock_post.call_count == 6
    assert restored.failure_kind == "error"
    assert restored.timed_out is False
    assert restored.timeout_stage == "encoder"
    assert restored.timeout_reason == "connect"
    assert restored.timeout_seconds == 30
    assert restored.timeout_attempts == 5


class TestEvalSuiteManifest:
    @pytest.fixture(autouse=True)
    def _stable_persisted_result_revalidation(self):
        """Make resume-verdict recomputation deterministic in suite-state tests."""

        metrics = _fake_eval_result("runner", "citation").metrics

        def evaluate_without_private_key(**_kwargs):
            assert EVAL_EVIDENCE_PRIVATE_KEY_ENV not in os.environ
            assert APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV not in os.environ
            return metrics

        with patch(
            "axiom_encode.harness.evals.evaluate_artifact",
            side_effect=evaluate_without_private_key,
        ) as mock_evaluate:
            self.persisted_result_revalidation = mock_evaluate
            yield

    def test_manifest_case_identity_exposes_oracle_mode(self, tmp_path):
        manifest = EvalSuiteManifest(
            name="Oracle identity",
            path=tmp_path / "not-written.yaml",
            runners=["openai:gpt-5.4"],
            mode="cold",
            allow_context=[],
            gates=EvalReadinessGates(),
            cases=[
                EvalSuiteCase(
                    kind="source",
                    name="policyengine-case",
                    mode="cold",
                    corpus_citation_path="us/statute/7/2017",
                    oracle="policyengine",
                ),
                EvalSuiteCase(
                    kind="source",
                    name="non-oracle-case",
                    mode="cold",
                    corpus_citation_path="us/statute/7/2017",
                ),
            ],
        )

        identity = evals_module._build_eval_suite_manifest_identity(manifest)

        assert [
            case_identity["oracle"] for case_identity in identity["case_identities"]
        ] == ["policyengine", "none"]

    def test_rejects_removed_source_id_field(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            "name: legacy identity\n"
            "runners:\n"
            "  - openai:gpt-5.4\n"
            "gates:\n"
            "  min_cases: 1\n"
            "  min_success_rate: 1.0\n"
            "  min_compile_pass_rate: 1.0\n"
            "  min_ci_pass_rate: 1.0\n"
            "  min_zero_ungrounded_rate: 1.0\n"
            "  min_generalist_review_pass_rate: 1.0\n"
            "cases:\n"
            "  - kind: source\n"
            "    name: display only\n"
            "    source_id: legacy-alias\n"
            "    corpus_citation_path: us/statute/7/2017\n"
        )

        with pytest.raises(ValueError, match="sole source identity"):
            load_eval_suite_manifest(manifest_file)

    @pytest.mark.parametrize(
        "alias",
        [
            " us/statute/7/2017 ",
            "us//statute/7/2017",
            "us:statutes/7/2017",
            "7 USC 2017",
            "us/statutes/7/2017",
        ],
    )
    def test_source_case_rejects_corpus_identity_aliases(self, tmp_path, alias):
        payload = _strict_eval_suite_manifest_payload()
        payload["cases"][0]["corpus_citation_path"] = alias
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        with pytest.raises(ValueError, match="canonical"):
            load_eval_suite_manifest(manifest_file)

    def test_human_citation_case_retains_human_parsing(self, tmp_path):
        payload = _strict_eval_suite_manifest_payload()
        case = payload["cases"][0]
        case["kind"] = "citation"
        case["citation"] = "7 USC 2017"
        del case["corpus_citation_path"]
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        manifest = load_eval_suite_manifest(manifest_file)

        assert manifest.cases[0].citation == "7 USC 2017"

    def test_complete_source_unit_case_mode_is_default_off_and_identity_preserving(
        self,
        tmp_path,
    ):
        payload = _strict_eval_suite_manifest_payload()
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        default_case = load_eval_suite_manifest(manifest_file).cases[0]
        explicit_off_case = replace(default_case, require_complete_source_unit=False)
        complete_case = replace(default_case, require_complete_source_unit=True)

        default_identity = evals_module._canonical_eval_suite_case_payload(default_case)
        assert default_case.require_complete_source_unit is False
        assert (
            evals_module._canonical_eval_suite_case_payload(explicit_off_case)
            == default_identity
        )
        assert "require_complete_source_unit" not in default_identity
        assert (
            evals_module._canonical_eval_suite_case_payload(complete_case)[
                "require_complete_source_unit"
            ]
            is True
        )

    @pytest.mark.parametrize("value", [None, 0, 1, "true", []])
    def test_complete_source_unit_case_mode_requires_a_boolean(
        self,
        tmp_path,
        value,
    ):
        payload = _strict_eval_suite_manifest_payload()
        payload["cases"][0]["require_complete_source_unit"] = value
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        with pytest.raises(
            ValueError,
            match="require_complete_source_unit.*must be a boolean",
        ):
            load_eval_suite_manifest(manifest_file)

    def test_complete_source_unit_case_mode_loads_when_enabled(self, tmp_path):
        payload = _strict_eval_suite_manifest_payload()
        payload["cases"][0]["require_complete_source_unit"] = True
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        manifest = load_eval_suite_manifest(manifest_file)

        assert manifest.cases[0].require_complete_source_unit is True

    def test_complete_source_unit_case_rejects_mismatched_runner_result(self):
        case = EvalSuiteCase(
            kind="source",
            name="sample",
            mode="cold",
            corpus_citation_path="us/statute/7/2017",
            require_complete_source_unit=True,
        )
        result = _fake_eval_result(
            "openai-gpt-5.4",
            "us/statute/7/2017",
        )

        with pytest.raises(ValueError, match="different complete-source-unit mode"):
            evals_module._validate_new_eval_suite_case_results(
                case,
                [result],
                [parse_runner_spec("openai:gpt-5.4")],
            )

    def test_manifest_context_path_does_not_rewrite_legacy_checkout_layout(
        self, tmp_path
    ):
        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        monorepo_context = tmp_path / "rulespec-us/us-co/context.yaml"
        monorepo_context.parent.mkdir(parents=True)
        monorepo_context.write_text("format: rulespec/v1\nrules: []\n")
        manifest_file = suite_dir / "suite.yaml"
        manifest_file.write_text(
            """
name: Exact path suite
runners:
  - openai:gpt-5.4
gates:
  min_cases: 1
  min_success_rate: 1.0
  min_compile_pass_rate: 1.0
  min_ci_pass_rate: 1.0
  min_zero_ungrounded_rate: 1.0
  min_generalist_review_pass_rate: 1.0
cases:
  - kind: source
    name: sample
    corpus_citation_path: us/statute/7/2017
    allow_context:
      - ../rulespec-us-co/context.yaml
            """.strip()
        )

        manifest = load_eval_suite_manifest(manifest_file)

        assert manifest.cases[0].allow_context == [
            tmp_path / "rulespec-us-co/context.yaml"
        ]
        assert not manifest.cases[0].allow_context[0].exists()

    @pytest.mark.parametrize("symlink_kind", ["file", "parent"])
    def test_eval_suite_context_preserves_and_rejects_symlink_paths(
        self, tmp_path, symlink_kind
    ):
        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text(
            "Primary source continuation for sample.\n\n"
            "OPENAI_API_KEY=sentinel-secret-value\n"
        )
        if symlink_kind == "file":
            context_entry = Path("context.txt")
            (suite_dir / context_entry).symlink_to(outside_file)
        else:
            context_entry = Path("redirect") / "secret.txt"
            (suite_dir / "redirect").symlink_to(
                outside_dir,
                target_is_directory=True,
            )

        manifest_file = suite_dir / "suite.yaml"
        manifest_file.write_text(
            f"""
name: Unsafe context suite
mode: repo-augmented
runners:
  - openai:gpt-5.4
gates:
  min_cases: 1
  min_success_rate: 1.0
  min_compile_pass_rate: 1.0
  min_ci_pass_rate: 1.0
  min_zero_ungrounded_rate: 1.0
  min_generalist_review_pass_rate: 1.0
  min_policyengine_pass_rate: 1.0
cases:
  - kind: source
    name: sample
    corpus_citation_path: us/statute/7/2017
    allow_context:
      - {context_entry.as_posix()}
            """.strip()
        )

        manifest = load_eval_suite_manifest(manifest_file)
        case = manifest.cases[0]
        assert case.allow_context == [suite_dir / context_entry]
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        corpus_release, source_unit = _write_test_source_unit(
            tmp_path, "Primary source text."
        )

        with pytest.raises(UnsafeRulespecContextPath, match="symlink"):
            run_source_eval(
                source_unit=source_unit,
                runner_specs=manifest.runners,
                output_root=tmp_path / "out",
                policy_path=policy_repo_root,
                local_corpus_release=corpus_release,
                runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
                mode=case.mode,
                extra_context_paths=case.allow_context,
            )

    def test_load_eval_suite_manifest_supports_policyengine_rule_hint(self, tmp_path):
        manifest_file = tmp_path / "uk-expanded.yaml"
        manifest_file.write_text(
            """
name: UK expanded
runners:
  - openai:gpt-5.4
gates:
  min_cases: 1
  min_success_rate: 1.0
  min_compile_pass_rate: 1.0
  min_ci_pass_rate: 1.0
  min_zero_ungrounded_rate: 1.0
  min_generalist_review_pass_rate: 1.0
  min_policyengine_pass_rate: 1.0
cases:
  - kind: source
    name: uc-standard-allowance-single-young
    corpus_citation_path: us/statute/7/2017
    oracle: policyengine
    policyengine_rule_hint: uc_standard_allowance_single_claimant_aged_under_25
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative row text")

        manifest = load_eval_suite_manifest(manifest_file)

        assert (
            manifest.cases[0].policyengine_rule_hint
            == "uc_standard_allowance_single_claimant_aged_under_25"
        )

    def test_load_eval_suite_manifest_supports_generalist_review_gate(self, tmp_path):
        manifest_file = tmp_path / "uk-expanded.yaml"
        manifest_file.write_text(
            """
name: UK expanded
runners:
  - openai:gpt-5.4
gates:
  min_cases: 1
  min_success_rate: 1.0
  min_compile_pass_rate: 1.0
  min_ci_pass_rate: 1.0
  min_zero_ungrounded_rate: 1.0
  min_generalist_review_pass_rate: 0.95
cases:
  - kind: source
    name: sample
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative row text")

        manifest = load_eval_suite_manifest(manifest_file)

        assert manifest.gates.min_generalist_review_pass_rate == 0.95

    @pytest.mark.parametrize(
        "value",
        [True, 0, -1, 1.5, "1"],
        ids=["bool", "zero", "negative", "float", "string"],
    )
    def test_load_eval_suite_manifest_rejects_invalid_min_cases(self, tmp_path, value):
        payload = {
            "name": "Strict gates",
            "runners": ["openai:gpt-5.4"],
            "gates": {
                "min_cases": value,
                "min_success_rate": 1.0,
                "min_compile_pass_rate": 1.0,
                "min_ci_pass_rate": 1.0,
                "min_zero_ungrounded_rate": 1.0,
                "min_generalist_review_pass_rate": 1.0,
            },
            "cases": [
                {
                    "kind": "source",
                    "name": "sample",
                    "corpus_citation_path": "us/statute/7/2017",
                }
            ],
        }
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        with pytest.raises(ValueError, match="min_cases"):
            load_eval_suite_manifest(manifest_file)

    @pytest.mark.parametrize(
        "value",
        [None, True, -0.01, 1.01, float("nan"), float("inf"), "1.0", 10**400],
        ids=[
            "null",
            "bool",
            "negative",
            "above-one",
            "nan",
            "inf",
            "string",
            "huge-int",
        ],
    )
    def test_load_eval_suite_manifest_rejects_invalid_rate_gate(self, tmp_path, value):
        payload = {
            "name": "Strict rates",
            "runners": ["openai:gpt-5.4"],
            "gates": {
                "min_cases": 1,
                "min_success_rate": value,
                "min_compile_pass_rate": 1.0,
                "min_ci_pass_rate": 1.0,
                "min_zero_ungrounded_rate": 1.0,
                "min_generalist_review_pass_rate": 1.0,
            },
            "cases": [
                {
                    "kind": "source",
                    "name": "sample",
                    "corpus_citation_path": "us/statute/7/2017",
                }
            ],
        }
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        with pytest.raises(ValueError, match="min_success_rate"):
            load_eval_suite_manifest(manifest_file)

    @pytest.mark.parametrize(
        "gate_name",
        [
            "min_success_rate",
            "min_compile_pass_rate",
            "min_ci_pass_rate",
            "min_zero_ungrounded_rate",
            "min_generalist_review_pass_rate",
        ],
    )
    def test_load_eval_suite_manifest_requires_every_core_gate(
        self, tmp_path, gate_name
    ):
        gates = {
            "min_cases": 1,
            "min_success_rate": 1.0,
            "min_compile_pass_rate": 1.0,
            "min_ci_pass_rate": 1.0,
            "min_zero_ungrounded_rate": 1.0,
            "min_generalist_review_pass_rate": 1.0,
        }
        gates.pop(gate_name)
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            yaml.safe_dump(
                {
                    "runners": ["openai:gpt-5.4"],
                    "gates": gates,
                    "cases": [
                        {
                            "kind": "source",
                            "corpus_citation_path": "us/statute/7/2017",
                        }
                    ],
                }
            )
        )

        with pytest.raises(ValueError, match=gate_name):
            load_eval_suite_manifest(manifest_file)

    def test_load_eval_suite_manifest_rejects_unknown_fields(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            yaml.safe_dump(
                {
                    "runners": ["openai:gpt-5.4"],
                    "gates": {
                        "min_cases": 1,
                        "min_success_rate": 1.0,
                        "min_compile_pass_rate": 1.0,
                        "min_ci_pass_rate": 1.0,
                        "min_zero_ungrounded_rate": 1.0,
                        "min_generalist_review_pass_rate": 1.0,
                        "min_succes_rate": 0.0,
                    },
                    "cases": [
                        {
                            "kind": "source",
                            "corpus_citation_path": "us/statute/7/2017",
                        }
                    ],
                }
            )
        )

        with pytest.raises(ValueError, match="unsupported keys"):
            load_eval_suite_manifest(manifest_file)

    def test_load_eval_suite_manifest_requires_policyengine_gate_for_oracle(
        self, tmp_path
    ):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            yaml.safe_dump(
                {
                    "runners": ["openai:gpt-5.4"],
                    "gates": {
                        "min_cases": 1,
                        "min_success_rate": 1.0,
                        "min_compile_pass_rate": 1.0,
                        "min_ci_pass_rate": 1.0,
                        "min_zero_ungrounded_rate": 1.0,
                        "min_generalist_review_pass_rate": 1.0,
                    },
                    "cases": [
                        {
                            "kind": "source",
                            "corpus_citation_path": "us/statute/7/2017",
                            "oracle": "policyengine",
                        }
                    ],
                }
            )
        )

        with pytest.raises(ValueError, match="min_policyengine_pass_rate"):
            load_eval_suite_manifest(manifest_file)

    @pytest.mark.parametrize(
        "value",
        [None, True, -0.01, float("nan"), float("inf"), "1.0", 10**400],
        ids=["null", "bool", "negative", "nan", "inf", "string", "huge-int"],
    )
    def test_load_eval_suite_manifest_rejects_invalid_max_cost(self, tmp_path, value):
        payload = _strict_eval_suite_manifest_payload()
        payload["gates"]["max_mean_estimated_cost_usd"] = value
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        with pytest.raises(ValueError, match="max_mean_estimated_cost_usd"):
            load_eval_suite_manifest(manifest_file)

    @pytest.mark.parametrize(
        "runners",
        [None, "openai:gpt-5.4", [], [1], [""], [" openai:gpt-5.4"]],
        ids=["null", "string", "empty", "non-string", "blank", "whitespace"],
    )
    def test_load_eval_suite_manifest_rejects_noncanonical_runners(
        self, tmp_path, runners
    ):
        payload = _strict_eval_suite_manifest_payload()
        payload["runners"] = runners
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        with pytest.raises(ValueError, match="runners must be a nonempty list"):
            load_eval_suite_manifest(manifest_file)

    @pytest.mark.parametrize("gates", [None, [], "min_cases: 1"])
    def test_load_eval_suite_manifest_requires_gate_mapping(self, tmp_path, gates):
        payload = _strict_eval_suite_manifest_payload()
        payload["gates"] = gates
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        with pytest.raises(ValueError, match="gates must be a mapping"):
            load_eval_suite_manifest(manifest_file)

    @pytest.mark.parametrize("scope", ["manifest", "case"])
    def test_load_eval_suite_manifest_rejects_unknown_schema_key(self, tmp_path, scope):
        payload = _strict_eval_suite_manifest_payload()
        if scope == "manifest":
            payload["runner"] = "openai:gpt-5.4"
        else:
            payload["cases"][0]["policyengine_rule_hnit"] = "typo"
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        with pytest.raises(ValueError, match="unsupported keys"):
            load_eval_suite_manifest(manifest_file)

    @pytest.mark.parametrize("scope", ["manifest", "case"])
    @pytest.mark.parametrize("value", ["context.yaml", [""], [1]])
    def test_load_eval_suite_manifest_rejects_malformed_allow_context(
        self, tmp_path, scope, value
    ):
        payload = _strict_eval_suite_manifest_payload()
        target = payload if scope == "manifest" else payload["cases"][0]
        target["allow_context"] = value
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        with pytest.raises(ValueError, match="allow_context"):
            load_eval_suite_manifest(manifest_file)

    @pytest.mark.parametrize("scope", ["manifest", "case"])
    @pytest.mark.parametrize("value", [None, "", " padded ", 123])
    def test_load_eval_suite_manifest_rejects_noncanonical_explicit_name(
        self, tmp_path, scope, value
    ):
        payload = _strict_eval_suite_manifest_payload()
        target = payload if scope == "manifest" else payload["cases"][0]
        target["name"] = value
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        with pytest.raises(ValueError, match="name must be a canonical nonempty"):
            load_eval_suite_manifest(manifest_file)

    @pytest.mark.parametrize("oracle", ["policyengin", "taxsim", "all"])
    def test_load_eval_suite_manifest_rejects_unknown_oracle(self, tmp_path, oracle):
        payload = _strict_eval_suite_manifest_payload()
        payload["cases"][0]["oracle"] = oracle
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        with pytest.raises(ValueError, match="unsupported oracle"):
            load_eval_suite_manifest(manifest_file)

    def test_load_eval_suite_manifest_rejects_removed_country_override(self, tmp_path):
        payload = _strict_eval_suite_manifest_payload()
        payload["cases"][0]["policyengine_country"] = "uk"
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        with pytest.raises(ValueError, match="unsupported keys"):
            load_eval_suite_manifest(manifest_file)

    @pytest.mark.parametrize(
        ("kind", "extra_field"),
        [("source", "citation"), ("citation", "corpus_citation_path")],
    )
    def test_load_eval_suite_manifest_rejects_second_case_identity(
        self, tmp_path, kind, extra_field
    ):
        payload = _strict_eval_suite_manifest_payload()
        case = payload["cases"][0]
        case["kind"] = kind
        case["citation"] = "us/statute/7/2017"
        case["corpus_citation_path"] = "us/statute/7/2017"
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))

        with pytest.raises(ValueError, match=f"cannot declare '{extra_field}'"):
            load_eval_suite_manifest(manifest_file)

    def test_load_eval_suite_manifest_resolves_explicit_dependency_roots(
        self, tmp_path
    ):
        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        dependency_root = _canonical_rulespec_content_root(tmp_path, "uk").parent
        manifest_file = suite_dir / "suite.yaml"
        manifest_file.write_text(
            """
name: Explicit dependency suite
rulespec_dependency_roots:
  - ../rulespec-uk
runners:
  - openai:gpt-5.4
gates:
  min_cases: 1
  min_success_rate: 1.0
  min_compile_pass_rate: 1.0
  min_ci_pass_rate: 1.0
  min_zero_ungrounded_rate: 1.0
  min_generalist_review_pass_rate: 1.0
cases:
  - kind: source
    name: sample
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )

        manifest = load_eval_suite_manifest(manifest_file)

        assert manifest.rulespec_dependency_roots == [dependency_root]

    def test_eval_suite_identity_expands_dependency_checkout_to_jurisdictions(
        self, tmp_path
    ):
        active_root = _canonical_rulespec_content_root(tmp_path, "us")
        dependency_checkout = _canonical_rulespec_content_root(tmp_path, "uk").parent
        (dependency_checkout / "uk-sc").mkdir()
        (dependency_checkout / "README.md").write_text("checkout notes\n")
        manifest = EvalSuiteManifest(
            name="Dependency identity",
            path=tmp_path / "suite.yaml",
            runners=["openai:gpt-5.4"],
            mode="cold",
            allow_context=[],
            gates=EvalReadinessGates(),
            cases=[
                EvalSuiteCase(
                    kind="source",
                    name="sample",
                    corpus_citation_path="us/statute/7/2017",
                    mode="cold",
                )
            ],
            rulespec_dependency_roots=[dependency_checkout],
        )

        roots = _eval_suite_rulespec_roots(manifest, active_root.parent)

        assert roots == tuple(
            sorted(
                {
                    str(active_root.resolve()),
                    str((dependency_checkout / "uk").resolve()),
                    str((dependency_checkout / "uk-sc").resolve()),
                }
            )
        )
        assert str(dependency_checkout.resolve()) not in roots

    @pytest.mark.parametrize(
        "raw_value",
        ["rulespec-uk", "[rulespec-uk, '']"],
    )
    def test_load_eval_suite_manifest_rejects_invalid_dependency_roots(
        self, tmp_path, raw_value
    ):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            f"""
name: Invalid dependency suite
rulespec_dependency_roots: {raw_value}
cases:
  - kind: source
    name: sample
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )

        with pytest.raises(ValueError, match="list of non-empty paths"):
            load_eval_suite_manifest(manifest_file)

    def test_load_eval_suite_manifest_rejects_dependency_workspace_root(self, tmp_path):
        workspace_root = tmp_path / "dependencies"
        workspace_root.mkdir()
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Invalid dependency suite
rulespec_dependency_roots:
  - dependencies
cases:
  - kind: source
    name: sample
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )

        with pytest.raises(
            UnsafeRulespecContextPath,
            match="exact canonical checkout roots",
        ):
            load_eval_suite_manifest(manifest_file)

    def test_run_eval_suite_passes_explicit_dependency_roots_to_runner(self, tmp_path):
        dependency_root = _canonical_rulespec_content_root(tmp_path, "uk").parent
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Explicit dependency suite
rulespec_dependency_roots:
  - rulespec-uk
runners:
  - openai:gpt-5.4
gates:
  min_cases: 1
  min_success_rate: 1.0
  min_compile_pass_rate: 1.0
  min_ci_pass_rate: 1.0
  min_zero_ungrounded_rate: 1.0
  min_generalist_review_pass_rate: 1.0
  min_policyengine_pass_rate: 1.0
cases:
  - kind: source
    name: sample
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        manifest = load_eval_suite_manifest(manifest_file)
        source_result = _fake_eval_result("openai-gpt-5.4", "sample")

        def fake_execution_identity(
            _engine_path,
            roots,
            *,
            suite_retry_attempts,
        ):
            assert suite_retry_attempts == 2
            return {
                "schema": "test",
                "case_timeout_seconds": 3600,
                "rulespec_roots": [
                    {
                        "path": root,
                        "content_sha256": "content",
                        "toolchain_contract_sha256": "toolchain",
                        "validation_waiver_set_sha256": "waivers",
                    }
                    for root in roots
                ],
            }

        with (
            patch(
                "axiom_encode.harness.evals._build_eval_suite_execution_identity",
                side_effect=fake_execution_identity,
            ),
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=lambda **kwargs: _bind_fake_source_results(
                    [source_result], kwargs
                ),
            ) as mock_source,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
            )

        assert results == [source_result]
        assert mock_source.call_args.kwargs["rulespec_dependency_roots"] == [
            dependency_root
        ]

    def test_run_eval_suite_forwards_complete_source_unit_mode_per_case(
        self,
        tmp_path,
    ):
        payload = _strict_eval_suite_manifest_payload()
        payload["cases"] = [
            {
                "kind": "source",
                "name": "source-case",
                "corpus_citation_path": "us/statute/7/2017",
                "require_complete_source_unit": True,
            },
            {
                "kind": "citation",
                "name": "citation-case",
                "citation": "7 USC 2017",
                "require_complete_source_unit": True,
            },
        ]
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(yaml.safe_dump(payload))
        manifest = load_eval_suite_manifest(manifest_file)
        corpus_release = _write_test_corpus_provision(tmp_path)

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=RuntimeError("source runner stopped"),
            ) as mock_source,
            patch(
                "axiom_encode.harness.evals.run_model_eval",
                side_effect=RuntimeError("citation runner stopped"),
            ) as mock_model,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                suite_retry_attempts=0,
            )

        assert mock_source.call_args.kwargs["require_complete_source_unit"] is True
        assert mock_model.call_args.kwargs["require_complete_source_unit"] is True
        assert len(results) == 2
        assert all(result.require_complete_source_unit is True for result in results)

    def test_run_eval_suite_passes_policyengine_rule_hint_to_source_runner(
        self, tmp_path
    ):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: UK source suite
runners:
  - openai:gpt-5.4
gates:
  min_cases: 1
  min_success_rate: 1.0
  min_compile_pass_rate: 1.0
  min_ci_pass_rate: 1.0
  min_zero_ungrounded_rate: 1.0
  min_generalist_review_pass_rate: 1.0
  min_policyengine_pass_rate: 1.0
cases:
  - kind: source
    name: uc-standard-allowance-single-young
    corpus_citation_path: us/statute/7/2017
    oracle: policyengine
    policyengine_rule_hint: uc_standard_allowance_single_claimant_aged_under_25
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative row text")
        manifest = load_eval_suite_manifest(manifest_file)
        source_result = _fake_eval_result(
            "openai-gpt-5.4",
            "uc-std-allowance-single",
            policyengine_pass=True,
            policyengine_score=1.0,
        )
        runtime = _test_policyengine_runtime("us")

        with (
            patch.object(
                PolicyEngineRuntime,
                "assert_matches_rulespec_root",
                return_value=None,
            ),
            patch.object(
                PolicyEngineRuntime,
                "assert_unchanged",
                return_value=None,
            ),
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=lambda **kwargs: _bind_fake_source_results(
                    [source_result], kwargs
                ),
            ) as mock_source,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
                policyengine_runtime=runtime,
            )

        assert results == [source_result]
        assert (
            mock_source.call_args.kwargs["policyengine_rule_hint"]
            == "uc_standard_allowance_single_claimant_aged_under_25"
        )

    def test_run_eval_suite_requires_runtime_before_creating_output(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: PolicyEngine runtime required
runners:
  - openai:gpt-5.4
gates:
  min_cases: 1
  min_success_rate: 1.0
  min_compile_pass_rate: 1.0
  min_ci_pass_rate: 1.0
  min_zero_ungrounded_rate: 1.0
  min_generalist_review_pass_rate: 1.0
  min_policyengine_pass_rate: 1.0
cases:
  - kind: source
    name: snap-slice
    corpus_citation_path: us/statute/7/2017
    oracle: policyengine
            """.strip()
        )
        manifest = load_eval_suite_manifest(manifest_file)
        output_root = tmp_path / "out"

        with pytest.raises(
            PolicyEngineRuntimeError,
            match="no explicit admitted runtime",
        ):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
            )

        assert not output_root.exists()

    def test_run_eval_suite_records_active_case_before_dispatch(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Active case suite
runners:
  - openai:gpt-5.4
gates:
  min_cases: 1
  min_success_rate: 1.0
  min_compile_pass_rate: 1.0
  min_ci_pass_rate: 1.0
  min_zero_ungrounded_rate: 1.0
  min_generalist_review_pass_rate: 1.0
  min_policyengine_pass_rate: 1.0
cases:
  - kind: source
    name: tanf-slice
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        output_root = tmp_path / "out"
        source_result = _fake_eval_result("openai-gpt-5.4", "co-tanf-f")
        corpus_release = _write_test_corpus_provision(tmp_path)
        expected_release_identity = _test_eval_suite_release_identity(corpus_release)
        snapshots: list[dict] = []

        def fake_run_source_eval(**kwargs):
            snapshots.append(json.loads((output_root / "suite-run.json").read_text()))
            return _bind_fake_source_results([source_result], kwargs)

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=fake_run_source_eval,
            ),
        ):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
            )

        assert len(snapshots) == 1
        active_state = snapshots[0]
        assert active_state["status"] == "running"
        assert active_state["completed_cases"] == 0
        assert active_state["result_count"] == 0
        assert {
            key: active_state[key] for key in expected_release_identity
        } == expected_release_identity
        assert active_state["active_case"]["index"] == 1
        assert active_state["active_case"]["name"] == "tanf-slice"
        assert active_state["active_case"]["output_root"] == str(
            output_root / "01-tanf-slice"
        )
        assert active_state["rulespec_roots"] == [
            str((tmp_path / "rulespec-us" / "us").resolve())
        ]
        final_state = json.loads((output_root / "suite-run.json").read_text())
        assert final_state["status"] == "completed"
        assert "active_case" not in final_state
        ledger_row = json.loads(
            (output_root / "suite-results.jsonl").read_text().strip()
        )
        admission = ledger_row["result"]["admission"]
        assert {
            key: admission["corpus"][key] for key in expected_release_identity
        } == expected_release_identity
        assert admission["rulespec"]["policy_repo_root"] == str(
            (tmp_path / "rulespec-us" / "us").resolve()
        )

    def test_run_eval_suite_routes_source_case_to_monorepo_content_root(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us" / "us-tn"
        policy_repo.mkdir(parents=True)
        runtime_axiom_rules = tmp_path / "axiom-rules-engine"
        runtime_axiom_rules.mkdir()
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-tn/policy/snap-standard-utility-allowance",
            body="Tennessee source text",
        )
        output_root = tmp_path / "out"

        manifest = EvalSuiteManifest(
            name="TN suite",
            path=tmp_path / "suite.yaml",
            runners=["openai:gpt-5.4"],
            mode="repo-augmented",
            allow_context=[],
            gates=EvalReadinessGates(),
            cases=[
                EvalSuiteCase(
                    kind="source",
                    name="snap-tn-sua",
                    corpus_citation_path="us-tn/policy/snap-standard-utility-allowance",
                    mode="repo-augmented",
                )
            ],
        )
        source_result = _fake_eval_result("openai-gpt-5.4", "snap-tn-sua")

        resolved_source_unit = resolve_corpus_source_unit(
            "us-tn/policy/snap-standard-utility-allowance",
            corpus_release,
        )
        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=lambda **kwargs: _bind_fake_source_results(
                    [source_result], kwargs
                ),
            ) as mock_run_source_eval,
        ):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=runtime_axiom_rules,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
            )

        assert mock_run_source_eval.call_args.kwargs["policy_path"] == policy_repo
        assert (
            mock_run_source_eval.call_args.kwargs["source_unit"] == resolved_source_unit
        )
        assert (
            mock_run_source_eval.call_args.kwargs["local_corpus_release"]
            is corpus_release
        )
        assert (
            mock_run_source_eval.call_args.kwargs["runtime_axiom_rules_path"]
            == runtime_axiom_rules
        )
        ledger_row = json.loads(
            (output_root / "suite-results.jsonl").read_text().strip()
        )
        assert ledger_row["result"]["admission"]["rulespec"]["policy_repo_root"] == str(
            policy_repo.resolve()
        )

    def test_run_eval_suite_rejects_new_result_with_wrong_mode(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            "name: mode binding\n"
            "mode: repo-augmented\n"
            "runners:\n"
            "  - openai:gpt-5.4\n"
            "gates:\n"
            "  min_cases: 1\n"
            "  min_success_rate: 1.0\n"
            "  min_compile_pass_rate: 1.0\n"
            "  min_ci_pass_rate: 1.0\n"
            "  min_zero_ungrounded_rate: 1.0\n"
            "  min_generalist_review_pass_rate: 1.0\n"
            "cases:\n"
            "  - kind: source\n"
            "    name: case-one\n"
            "    corpus_citation_path: us/statute/7/2017\n"
        )
        manifest = load_eval_suite_manifest(manifest_file)
        corpus_release = _write_test_corpus_provision(tmp_path)

        def wrong_mode_result(**kwargs):
            result = _fake_eval_result("openai-gpt-5.4", "case-one")
            bound = _bind_fake_source_results([result], kwargs)
            bound[0].mode = "cold"
            return bound

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=wrong_mode_result,
            ),
            pytest.raises(ValueError, match="instead of 'repo-augmented'"),
        ):
            run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
            )

    def test_run_eval_suite_validates_unsigned_artifacts_before_signing(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            "name: validate before signing\n"
            "runners:\n"
            "  - openai:gpt-5.4\n"
            "gates:\n"
            "  min_cases: 1\n"
            "  min_success_rate: 1.0\n"
            "  min_compile_pass_rate: 1.0\n"
            "  min_ci_pass_rate: 1.0\n"
            "  min_zero_ungrounded_rate: 1.0\n"
            "  min_generalist_review_pass_rate: 1.0\n"
            "cases:\n"
            "  - kind: source\n"
            "    name: case-one\n"
            "    corpus_citation_path: us/statute/7/2017\n"
        )
        manifest = load_eval_suite_manifest(manifest_file)
        corpus_release = _write_test_corpus_provision(tmp_path)
        output_root = tmp_path / "out"

        def outside_artifact_result(**kwargs):
            result = _fake_eval_result("openai-gpt-5.4", "case-one")
            bound = _bind_fake_source_results([result], kwargs)
            generated = Path(bound[0].output_file)
            outside = tmp_path / "outside-generated.yaml"
            outside.write_bytes(generated.read_bytes())
            bound[0].output_file = str(outside)
            bound[0].generated_output_sha256 = hashlib.sha256(
                outside.read_bytes()
            ).hexdigest()
            return bound

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=outside_artifact_result,
            ),
            patch(
                "axiom_encode.harness.evals._write_eval_result_verdict_evidence"
            ) as mock_write_verdict,
            pytest.raises(ValueError, match="outside its runner-owned artifact"),
        ):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
            )

        mock_write_verdict.assert_not_called()
        assert not (output_root / "verdicts").exists()

    def test_run_eval_suite_retries_transient_exception(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Retry suite
runners:
  - openai:gpt-5.4
gates:
  min_cases: 1
  min_success_rate: 1.0
  min_compile_pass_rate: 1.0
  min_ci_pass_rate: 1.0
  min_zero_ungrounded_rate: 1.0
  min_generalist_review_pass_rate: 1.0
  min_policyengine_pass_rate: 1.0
cases:
  - kind: source
    name: tanf-slice
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        source_result = _fake_eval_result("openai-gpt-5.4", "co-tanf-f")

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            side_effect=_fake_source_runner(
                RuntimeError("stream disconnected"), [source_result]
            ),
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
            )

        assert results == [source_result]
        assert mock_source.call_count == 2

    def test_terminal_encoder_timeout_is_not_retried_by_suite(self, tmp_path):
        manifest = EvalSuiteManifest(
            name="Terminal timeout suite",
            path=tmp_path / "suite.yaml",
            runners=["claude:opus"],
            mode="cold",
            allow_context=[],
            gates=EvalReadinessGates(),
            cases=[
                EvalSuiteCase(
                    kind="source",
                    name="case-one",
                    corpus_citation_path="us/statute/7/2017",
                    mode="cold",
                )
            ],
        )

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            side_effect=subprocess.TimeoutExpired(["claude"], timeout=600),
        ) as mock_source:
            [result] = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
            )

        mock_source.assert_called_once()
        assert result.failure_kind == "timeout"
        assert result.timed_out is True
        assert result.timeout_stage == "case"
        assert result.timeout_reason == "wall"
        assert result.timeout_seconds == 600
        assert result.timeout_attempts == 1

    def test_case_budget_stops_suite_retry_and_marks_terminal_timeout(
        self,
        tmp_path,
        monkeypatch,
    ):
        manifest = EvalSuiteManifest(
            name="Case budget suite",
            path=tmp_path / "suite.yaml",
            runners=["openai:gpt-5.4"],
            mode="cold",
            allow_context=[],
            gates=EvalReadinessGates(),
            cases=[
                EvalSuiteCase(
                    kind="source",
                    name="case-one",
                    corpus_citation_path="us/statute/7/2017",
                    mode="cold",
                )
            ],
        )
        clock = [0.0]
        monkeypatch.setenv("AXIOM_ENCODE_EVAL_CASE_TIMEOUT_SECONDS", "10")
        monkeypatch.setattr(evals_module.time, "monotonic", lambda: clock[0])

        def exhaust_budget(**_kwargs):
            clock[0] = 11.0
            raise RuntimeError("stream disconnected")

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            side_effect=exhaust_budget,
        ) as mock_source:
            [result] = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
            )

        mock_source.assert_called_once()
        assert result.failure_kind == "timeout"
        assert result.timed_out is True
        assert result.timeout_stage == "case_budget"
        assert result.timeout_reason == "wall"
        assert result.timeout_seconds == 10
        assert result.timeout_attempts == 1
        assert "case budget" in (result.error or "").lower()

    def test_validation_time_does_not_consume_generation_retry_budget(
        self,
        tmp_path,
        monkeypatch,
    ):
        policy_repo_checkout = _canonical_rulespec_content_root(tmp_path, "us").parent
        manifest = EvalSuiteManifest(
            name="Generation-only case budget",
            path=tmp_path / "suite.yaml",
            runners=["codex:gpt-5.4"],
            mode="cold",
            allow_context=[],
            gates=EvalReadinessGates(),
            cases=[
                EvalSuiteCase(
                    kind="source",
                    name="case-one",
                    corpus_citation_path="us/statute/7/2017",
                    mode="cold",
                )
            ],
        )
        clock = [0.0]
        artifact = (
            "=== FILE: sample.yaml ===\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: source states 451.\n"
            "rules: []\n"
            "=== FILE: sample.test.yaml ===\n"
            "[]\n"
        )
        validation_results = [
            _revalidation_metrics(
                compile_pass=False,
                compile_issues=["deterministic compile failure"],
            ),
            _revalidation_metrics(),
        ]
        monkeypatch.setenv("AXIOM_ENCODE_EVAL_CASE_TIMEOUT_SECONDS", "10")
        monkeypatch.setattr(evals_module.time, "monotonic", lambda: clock[0])

        def generate_artifact(*_args, **_kwargs):
            clock[0] += 1.0
            return EvalPromptResponse(text=artifact, duration_ms=1000)

        def validate_artifact(**_kwargs):
            clock[0] += 20.0
            return validation_results.pop(0)

        with (
            patch(
                "axiom_encode.harness.evals._run_prompt_eval",
                side_effect=generate_artifact,
            ) as mock_prompt,
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
                side_effect=validate_artifact,
            ) as mock_validate,
        ):
            [result] = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=policy_repo_checkout,
                corpus_release=_write_test_corpus_provision(tmp_path),
                suite_retry_attempts=1,
            )

        assert mock_prompt.call_count == 2
        assert mock_validate.call_count == 2
        assert result.success is True
        assert result.timed_out is False

    def test_suite_retry_setup_consumes_generation_budget(
        self,
        tmp_path,
        monkeypatch,
    ):
        policy_repo_checkout = _canonical_rulespec_content_root(tmp_path, "us").parent
        manifest = EvalSuiteManifest(
            name="Retry setup case budget",
            path=tmp_path / "suite.yaml",
            runners=["codex:gpt-5.4"],
            mode="cold",
            allow_context=[],
            gates=EvalReadinessGates(),
            cases=[
                EvalSuiteCase(
                    kind="source",
                    name="case-one",
                    corpus_citation_path="us/statute/7/2017",
                    mode="cold",
                )
            ],
        )
        clock = [0.0]
        observed_generation_budgets: list[float | None] = []
        artifact = (
            "=== FILE: sample.yaml ===\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: source states 451.\n"
            "rules: []\n"
            "=== FILE: sample.test.yaml ===\n"
            "[]\n"
        )
        real_prepare_workspace = prepare_eval_workspace
        prepare_calls = 0
        monkeypatch.setenv("AXIOM_ENCODE_EVAL_CASE_TIMEOUT_SECONDS", "10")
        monkeypatch.setattr(evals_module.time, "monotonic", lambda: clock[0])

        def prepare_with_retry_setup_cost(*args, **kwargs):
            nonlocal prepare_calls
            prepare_calls += 1
            workspace = real_prepare_workspace(*args, **kwargs)
            if prepare_calls == 2:
                clock[0] += 9.0
            return workspace

        def generate_artifact(*_args, **_kwargs):
            observed_generation_budgets.append(
                evals_module._remaining_eval_case_budget_seconds()
            )
            clock[0] += 1.0
            return EvalPromptResponse(text=artifact, duration_ms=1000)

        def reject_first_artifact(**_kwargs):
            clock[0] += 20.0
            return _revalidation_metrics(
                compile_pass=False,
                compile_issues=["deterministic compile failure"],
            )

        with (
            patch(
                "axiom_encode.harness.evals.prepare_eval_workspace",
                side_effect=prepare_with_retry_setup_cost,
            ) as mock_prepare,
            patch(
                "axiom_encode.harness.evals._run_prompt_eval",
                side_effect=generate_artifact,
            ) as mock_prompt,
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
                side_effect=reject_first_artifact,
            ) as mock_validate,
        ):
            [result] = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=policy_repo_checkout,
                corpus_release=_write_test_corpus_provision(tmp_path),
                suite_retry_attempts=1,
            )

        assert mock_prepare.call_count == 2
        assert mock_prompt.call_count == 2
        assert mock_validate.call_count == 1
        assert observed_generation_budgets == [10.0, 0.0]
        assert result.failure_kind == "timeout"
        assert result.timed_out is True
        assert result.timeout_stage == "case_budget"
        assert not result.output_file
        assert result.metrics is None

    def test_case_budget_scope_does_not_relabel_completed_artifact_outcome(self):
        result = _fake_eval_result("openai-gpt", "case-one")

        evals_module._mark_suite_case_budget_timeout(
            [result],
            timeout_seconds=10,
        )

        assert result.success is True
        assert result.output_file
        assert result.metrics is not None
        assert result.failure_kind is None
        assert result.timed_out is False

    def test_each_runner_case_gets_an_independent_generation_budget(
        self,
        tmp_path,
        monkeypatch,
    ):
        manifest = EvalSuiteManifest(
            name="Independent runner budgets",
            path=tmp_path / "suite.yaml",
            runners=[
                "alpha=openai:gpt-5.4",
                "beta=codex:gpt-5.4",
            ],
            mode="cold",
            allow_context=[],
            gates=EvalReadinessGates(),
            cases=[
                EvalSuiteCase(
                    kind="source",
                    name="case-one",
                    corpus_citation_path="us/statute/7/2017",
                    mode="cold",
                )
            ],
        )
        clock = [0.0]
        observed_remaining: list[tuple[str, float | None]] = []
        monkeypatch.setenv("AXIOM_ENCODE_EVAL_CASE_TIMEOUT_SECONDS", "10")
        monkeypatch.setattr(evals_module.time, "monotonic", lambda: clock[0])

        def run_one_runner(**kwargs):
            [runner_spec] = kwargs["runner_specs"]
            runner = parse_runner_spec(runner_spec)
            observed_remaining.append(
                (
                    runner.name,
                    evals_module._remaining_eval_case_budget_seconds(),
                )
            )
            result = _fake_eval_result(runner.name, "case-one")
            result.backend = runner.backend
            result.model = runner.model
            if runner.name == "alpha":
                clock[0] = 11.0
            return _bind_fake_source_results([result], kwargs)

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            side_effect=run_one_runner,
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
            )

        assert [result.runner for result in results] == ["alpha", "beta"]
        assert mock_source.call_count == 2
        assert observed_remaining == [
            ("alpha", pytest.approx(10.0)),
            ("beta", pytest.approx(10.0)),
        ]

    def test_runner_timeout_does_not_suppress_other_runner_retry(self, tmp_path):
        manifest = EvalSuiteManifest(
            name="Independent runner retries",
            path=tmp_path / "suite.yaml",
            runners=[
                "alpha=claude:opus",
                "beta=openai:gpt-5.4",
            ],
            mode="cold",
            allow_context=[],
            gates=EvalReadinessGates(),
            cases=[
                EvalSuiteCase(
                    kind="source",
                    name="case-one",
                    corpus_citation_path="us/statute/7/2017",
                    mode="cold",
                )
            ],
        )
        attempts = {"alpha": 0, "beta": 0}

        def run_one_runner(**kwargs):
            [runner_spec] = kwargs["runner_specs"]
            runner = parse_runner_spec(runner_spec)
            attempts[runner.name] += 1
            source_attestation = _expected_eval_source_attestation(
                kwargs["source_unit"],
                rulespec_root=kwargs["policy_path"],
            )
            if runner.name == "alpha":
                return evals_module._suite_case_failure_results(
                    manifest.cases[0],
                    [runner],
                    subprocess.TimeoutExpired(["claude"], timeout=600),
                    source_attestation=source_attestation,
                )
            if attempts[runner.name] == 1:
                return evals_module._suite_case_failure_results(
                    manifest.cases[0],
                    [runner],
                    RuntimeError("stream disconnected"),
                    source_attestation=source_attestation,
                )
            result = _fake_eval_result(runner.name, "case-one")
            result.backend = runner.backend
            result.model = runner.model
            return _bind_fake_source_results([result], kwargs)

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            side_effect=run_one_runner,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
                suite_retry_attempts=1,
            )

        assert attempts == {"alpha": 1, "beta": 2}
        assert [result.runner for result in results] == ["alpha", "beta"]
        assert results[0].timed_out is True
        assert results[1].success is True

    def test_suite_timeout_history_does_not_promote_later_error_to_timeout(
        self,
    ):
        timeout_result = replace(
            _fake_eval_result("claude-opus", "case-one"),
            output_file="",
            trace_file="",
            context_manifest_file="",
            generated_output_sha256=None,
            trace_sha256=None,
            context_manifest_sha256=None,
            success=False,
            error="Claude eval timed out",
            metrics=None,
            failure_kind="timeout",
            timed_out=True,
            timeout_stage="encoder",
            timeout_reason="wall",
            timeout_seconds=600,
            timeout_attempts=1,
        )
        error_result = replace(
            timeout_result,
            error="authentication failed",
            failure_kind="error",
            timed_out=False,
            timeout_stage=None,
            timeout_reason=None,
            timeout_seconds=None,
            timeout_attempts=0,
        )
        timeout_history = {}

        evals_module._accumulate_suite_case_timeout_attempts(
            [timeout_result],
            timeout_history,
        )
        evals_module._accumulate_suite_case_timeout_attempts(
            [error_result],
            timeout_history,
        )

        assert error_result.failure_kind == "error"
        assert error_result.timed_out is False
        assert error_result.timeout_attempts == 1
        assert error_result.timeout_stage == "encoder"
        assert error_result.timeout_reason == "wall"
        assert error_result.timeout_seconds == 600

    def test_suite_does_not_retry_final_error_with_prior_timeout_evidence(
        self,
        tmp_path,
    ):
        manifest = EvalSuiteManifest(
            name="Timeout history suite",
            path=tmp_path / "suite.yaml",
            runners=["claude:opus"],
            mode="cold",
            allow_context=[],
            gates=EvalReadinessGates(),
            cases=[
                EvalSuiteCase(
                    kind="source",
                    name="case-one",
                    corpus_citation_path="us/statute/7/2017",
                    mode="cold",
                )
            ],
        )
        final_error = replace(
            _fake_eval_result("claude-opus", "us/statute/7/2017"),
            backend="claude",
            model="opus",
            output_file="",
            trace_file="",
            context_manifest_file="",
            generated_output_sha256=None,
            trace_sha256=None,
            context_manifest_sha256=None,
            success=False,
            error="authentication failed",
            metrics=None,
            failure_kind="error",
            timed_out=False,
            timeout_stage="encoder",
            timeout_reason="wall",
            timeout_seconds=600,
            timeout_attempts=1,
        )

        def return_final_error(**kwargs):
            final_error.source_attestation = _expected_eval_source_attestation(
                kwargs["source_unit"],
                rulespec_root=kwargs["policy_path"],
            )
            return [final_error]

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            side_effect=return_final_error,
        ) as mock_source:
            [result] = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
            )

        mock_source.assert_called_once()
        assert result.failure_kind == "error"
        assert result.timed_out is False
        assert result.error == "authentication failed"
        assert result.timeout_attempts == 1
        assert result.timeout_seconds == 600

    def test_run_eval_suite_retries_error_results(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Retry suite
runners:
  - openai:gpt-5.4
gates:
  min_cases: 1
  min_success_rate: 1.0
  min_compile_pass_rate: 1.0
  min_ci_pass_rate: 1.0
  min_zero_ungrounded_rate: 1.0
  min_generalist_review_pass_rate: 1.0
  min_policyengine_pass_rate: 1.0
cases:
  - kind: source
    name: tanf-slice
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        failed = _fake_eval_result("openai-gpt-5.4", "co-tanf-f")
        failed.success = False
        failed.error = "Reconnecting..."
        source_result = _fake_eval_result("openai-gpt-5.4", "co-tanf-f")

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            side_effect=_fake_source_runner([failed], [source_result]),
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
            )

        assert results == [source_result]
        assert mock_source.call_count == 2

    def test_run_eval_suite_does_not_retry_compile_failures(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Retry suite
runners:
  - openai:gpt-5.4
gates:
  min_cases: 1
  min_success_rate: 1.0
  min_compile_pass_rate: 1.0
  min_ci_pass_rate: 1.0
  min_zero_ungrounded_rate: 1.0
  min_generalist_review_pass_rate: 1.0
  min_policyengine_pass_rate: 1.0
cases:
  - kind: source
    name: tanf-slice
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        failed = _fake_eval_result(
            "openai-gpt-5.4",
            "co-tanf-f",
            compile_pass=False,
        )

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            side_effect=_fake_source_runner([failed]),
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
            )

        assert results == [failed]
        assert mock_source.call_count == 1

    def test_run_eval_suite_stops_after_usage_limit_error(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Usage limit suite
runners:
  - openai:gpt-5.4
gates:
  min_cases: 1
  min_success_rate: 1.0
  min_compile_pass_rate: 1.0
  min_ci_pass_rate: 1.0
  min_zero_ungrounded_rate: 1.0
  min_generalist_review_pass_rate: 1.0
  min_policyengine_pass_rate: 1.0
cases:
  - kind: source
    name: case-one
    corpus_citation_path: us/statute/7/2017
  - kind: source
    name: case-two
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        output_root = tmp_path / "out"

        usage_limited = _fake_eval_result("openai-gpt-5.4", "case-one")
        usage_limited.metrics.generalist_review_pass = False
        usage_limited.metrics.generalist_review_score = None
        usage_limited.metrics.generalist_review_issues = [
            "Reviewer CLI exited 1: You've hit your usage limit."
        ]
        second = _fake_eval_result("openai-gpt-5.4", "case-two")

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=_fake_source_runner([usage_limited], [second]),
            ) as mock_source,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
            )

        assert results == [usage_limited]
        assert mock_source.call_count == 1
        state = json.loads((output_root / "suite-run.json").read_text())
        assert state["status"] == "failed"
        assert "usage limit" in state["error"].lower()
        assert state["completed_cases"] == 1
        lines = (output_root / "suite-results.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1

    @pytest.mark.parametrize(
        ("error", "expected_status"),
        [
            ("generation failed before producing output", "completed"),
            ("You've hit your usage limit.", "failed"),
        ],
        ids=["generation-failure", "usage-limit"],
    )
    def test_run_eval_suite_persists_verdict_only_failure_result(
        self,
        tmp_path,
        error,
        expected_status,
    ):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            "name: Verdict-only failure suite\n"
            "runners:\n"
            "  - openai:gpt-5.4\n"
            "gates:\n"
            "  min_cases: 1\n"
            "  min_success_rate: 1.0\n"
            "  min_compile_pass_rate: 1.0\n"
            "  min_ci_pass_rate: 1.0\n"
            "  min_zero_ungrounded_rate: 1.0\n"
            "  min_generalist_review_pass_rate: 1.0\n"
            "cases:\n"
            "  - kind: source\n"
            "    name: case-one\n"
            "    corpus_citation_path: us/statute/7/2017\n"
        )
        manifest = load_eval_suite_manifest(manifest_file)
        output_root = tmp_path / "out"

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            side_effect=RuntimeError(error),
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
                suite_retry_attempts=0,
            )

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error == error
        mock_source.assert_called_once()
        row = json.loads((output_root / "suite-results.jsonl").read_text())
        result_payload = row["result"]
        assert result_payload["output_file"] == ""
        assert result_payload["trace_file"] == ""
        assert result_payload["context_manifest_file"] == ""
        assert result_payload["metrics"] is None
        assert result_payload["verdict_file"]
        assert result_payload["verdict_sha256"]
        persisted_result = _eval_result_from_payload(
            result_payload,
            artifact_name="Verdict-only failure result",
            require_verdict_evidence=True,
        )
        verified = _validate_eval_result_artifacts(
            persisted_result,
            output_root,
            artifact_name="Verdict-only failure result",
        )
        assert set(verified) == {"verdict_file"}
        state = json.loads((output_root / "suite-run.json").read_text())
        assert state["status"] == expected_status

    def test_run_eval_suite_retries_reviewer_timeout(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Timeout retry suite
runners:
  - openai:gpt-5.4
gates:
  min_cases: 1
  min_success_rate: 1.0
  min_compile_pass_rate: 1.0
  min_ci_pass_rate: 1.0
  min_zero_ungrounded_rate: 1.0
  min_generalist_review_pass_rate: 1.0
  min_policyengine_pass_rate: 1.0
cases:
  - kind: source
    name: case-one
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)

        timed_out = _fake_eval_result("openai-gpt-5.4", "case-one")
        timed_out.metrics.generalist_review_pass = False
        timed_out.metrics.generalist_review_score = None
        timed_out.metrics.generalist_review_issues = [
            "Reviewer error: Reviewer CLI exited 1: Timeout after 300s"
        ]
        recovered = _fake_eval_result("openai-gpt-5.4", "case-one")

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=_fake_source_runner([timed_out], [recovered]),
            ) as mock_source,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
            )

        assert results == [recovered]
        assert mock_source.call_count == 2

    def test_run_eval_suite_resume_skips_completed_cases(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Resume suite
runners:
  - openai:gpt-5.4
gates:
  min_cases: 1
  min_success_rate: 1.0
  min_compile_pass_rate: 1.0
  min_ci_pass_rate: 1.0
  min_zero_ungrounded_rate: 1.0
  min_generalist_review_pass_rate: 1.0
  min_policyengine_pass_rate: 1.0
cases:
  - kind: source
    name: case-one
    corpus_citation_path: us/statute/7/2017
  - kind: source
    name: case-two
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        output_root = tmp_path / "out"
        corpus_release = _write_test_corpus_provision(tmp_path)
        release_identity = _test_eval_suite_release_identity(corpus_release)
        rulespec_roots = [str((tmp_path / "rulespec-us" / "us").resolve())]

        first = _fake_eval_result("openai-gpt-5.4", "case-one")
        second = _fake_eval_result("openai-gpt-5.4", "case-two")
        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            side_effect=_fake_source_runner([first], KeyboardInterrupt()),
        ):
            with pytest.raises(KeyboardInterrupt):
                run_eval_suite(
                    manifest=manifest,
                    output_root=output_root,
                    axiom_rules_path=tmp_path / "axiom-rules-engine",
                    policy_repo_path=tmp_path / "rulespec-us",
                    corpus_release=corpus_release,
                )
        initial_state = json.loads((output_root / "suite-run.json").read_text())
        initial_run_id = initial_state["run_id"]
        initial_started_at = initial_state["started_at"]

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=_fake_source_runner([second]),
            ) as mock_source,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

        assert [result.citation for result in results] == [
            "us/statute/7/2017",
            "us/statute/7/2017",
        ]
        mock_source.assert_called_once()
        assert (
            mock_source.call_args.kwargs["source_unit"].requested == "us/statute/7/2017"
        )
        state = json.loads((output_root / "suite-run.json").read_text())
        assert state["status"] == "completed"
        assert state["run_id"] == initial_run_id
        assert state["started_at"] == initial_started_at
        assert state["completed_cases"] == 2
        assert {key: state[key] for key in release_identity} == release_identity
        assert state["rulespec_roots"] == rulespec_roots
        assert (
            state["manifest"]["content_sha256"]
            == hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        )
        assert len(state["manifest"]["case_identities"]) == 2
        assert (
            state["validation_waiver_sets"][0]["validation_waiver_set_sha256"]
            == hashlib.sha256(b"validate_failures: {}\n").hexdigest()
        )
        lines = (output_root / "suite-results.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        assert all(
            {
                key: json.loads(line)["result"]["admission"]["corpus"][key]
                for key in release_identity
            }
            == release_identity
            for line in lines
        )
        assert all(
            json.loads(line)["result"]["admission"]["rulespec"]["policy_repo_root"]
            == str((tmp_path / "rulespec-us" / "us").resolve())
            for line in lines
        )

    def test_run_eval_suite_requires_validated_local_corpus_release(self, tmp_path):
        manifest = EvalSuiteManifest(
            name="Bound release suite",
            path=tmp_path / "suite.yaml",
            runners=["openai:gpt-5.4"],
            mode="cold",
            allow_context=[],
            gates=EvalReadinessGates(),
            cases=[],
        )

        with pytest.raises(TypeError, match="validated LocalCorpusRelease"):
            run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=None,  # type: ignore[arg-type]
            )

        assert not (tmp_path / "out").exists()

    def test_run_eval_suite_resume_refuses_silent_fresh_start(self, tmp_path):
        manifest = EvalSuiteManifest(
            name="Resume requires state",
            path=tmp_path / "suite.yaml",
            runners=["openai:gpt-5.4"],
            mode="cold",
            allow_context=[],
            gates=EvalReadinessGates(),
            cases=[],
        )

        with pytest.raises(ValueError, match="silently start a fresh run"):
            run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
                resume_existing=True,
            )

    @pytest.mark.parametrize(
        ("managed_name", "is_directory"),
        [
            ("suite-results.jsonl", False),
            ("results.json", False),
            ("verdicts", True),
            ("01-prior-case", True),
        ],
    )
    def test_run_eval_suite_fresh_refuses_prior_managed_artifacts(
        self,
        tmp_path,
        managed_name,
        is_directory,
    ):
        manifest = EvalSuiteManifest(
            name="Fresh output required",
            path=tmp_path / "suite.yaml",
            runners=["openai:gpt-5.4"],
            mode="cold",
            allow_context=[],
            gates=EvalReadinessGates(),
            cases=[],
        )
        output = tmp_path / "out"
        output.mkdir()
        managed = output / managed_name
        if is_directory:
            managed.mkdir()
        else:
            managed.write_text("old suite bytes\n")

        with pytest.raises(ValueError, match="already contains managed artifacts"):
            run_eval_suite(
                manifest=manifest,
                output_root=output,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=_write_test_corpus_provision(tmp_path),
            )

        assert managed.exists()

    @pytest.mark.parametrize(
        "persisted_roots",
        [None, ["/tmp/different-rulespec-root"]],
        ids=["missing", "changed"],
    )
    def test_run_eval_suite_resume_rejects_rulespec_root_change(
        self,
        tmp_path,
        persisted_roots,
    ):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        state = json.loads((output_root / "suite-run.json").read_text())
        if persisted_roots is None:
            state.pop("rulespec_roots")
        else:
            state["rulespec_roots"] = persisted_roots
        (output_root / "suite-run.json").write_text(json.dumps(state) + "\n")

        with pytest.raises(ValueError, match="RuleSpec root identity"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    @pytest.mark.parametrize(
        ("persisted_identity", "expected_error"),
        [
            ({}, "missing corpus release identity"),
            (
                {
                    "corpus_release": "different-release",
                    "corpus_release_content_sha256": "1" * 64,
                    "corpus_release_selector_sha256": "0" * 64,
                },
                "different corpus release identity",
            ),
        ],
        ids=["missing", "mismatch"],
    )
    def test_run_eval_suite_resume_rejects_unbound_run_state(
        self,
        tmp_path,
        persisted_identity,
        expected_error,
    ):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        state = json.loads((output_root / "suite-run.json").read_text())
        state.pop("corpus_release")
        state.pop("corpus_release_content_sha256")
        state.pop("corpus_release_selector_sha256")
        state.update(persisted_identity)
        (output_root / "suite-run.json").write_text(json.dumps(state) + "\n")

        with pytest.raises(ValueError, match=expected_error):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    @pytest.mark.parametrize("mutation", ["missing", "mismatch"])
    def test_run_eval_suite_resume_rejects_unbound_results_ledger(
        self, tmp_path, mutation
    ):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        ledger_path = output_root / "suite-results.jsonl"
        row = json.loads(ledger_path.read_text())
        if mutation == "missing":
            row["result"].pop("admission")
        else:
            row["result"]["admission"]["corpus"]["corpus_release"] = "different-release"
        row["result"] = _bind_eval_result_payload(row["result"])
        ledger_path.write_text(json.dumps(row) + "\n")

        with pytest.raises(ValueError, match="signed admission|uses different"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    @pytest.mark.parametrize("mutation", ["missing", "mismatch"])
    def test_run_eval_suite_resume_rejects_unbound_policy_repo_root(
        self, tmp_path, mutation
    ):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        ledger_path = output_root / "suite-results.jsonl"
        row = json.loads(ledger_path.read_text())
        rulespec = row["result"]["admission"]["rulespec"]
        if mutation == "missing":
            rulespec.pop("policy_repo_root")
        else:
            rulespec["policy_repo_root"] = "/tmp/rulespec-us/us-ca"
        row["result"] = _bind_eval_result_payload(row["result"])
        ledger_path.write_text(json.dumps(row) + "\n")

        with pytest.raises(ValueError, match="uses different"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_run_eval_suite_resume_rejects_old_signed_verdict_schema(self, tmp_path):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        ledger_path = output_root / "suite-results.jsonl"
        row = json.loads(ledger_path.read_text())
        result_payload = row["result"]
        verdict_path = Path(result_payload["verdict_file"])
        verdict_payload = json.loads(verdict_path.read_text())
        verdict_payload["schema"] = "axiom-encode/eval-result-verdict/v5"
        verdict_payload["signature"] = sign_eval_evidence(
            verdict_payload,
            get_signing_broker(capability="eval_ed25519"),
        )
        verdict_raw = (
            json.dumps(verdict_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode()
        verdict_path.write_bytes(verdict_raw)
        result_payload["verdict_sha256"] = hashlib.sha256(verdict_raw).hexdigest()
        row["result"] = _bind_eval_result_payload(result_payload)
        ledger_path.write_text(json.dumps(row, sort_keys=True) + "\n")

        with pytest.raises(ValueError, match="unsupported authenticated verdict"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    @pytest.mark.parametrize("run_field", ["run_id", "started_at"])
    def test_run_eval_suite_resume_rejects_cross_run_verdict_replay(
        self, tmp_path, run_field
    ):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        state_path = output_root / "suite-run.json"
        state = json.loads(state_path.read_text())
        state[run_field] = (
            str(uuid.uuid4()) if run_field == "run_id" else "2030-01-01T00:00:00+00:00"
        )
        state_path.write_text(json.dumps(state, sort_keys=True) + "\n")

        with pytest.raises(ValueError, match="uses different run"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )
        self.persisted_result_revalidation.assert_not_called()

    @pytest.mark.parametrize(
        ("field_name", "value", "expected_error"),
        [
            ("run_id", None, "run_id"),
            ("run_id", "not-a-uuid", "run_id"),
            ("started_at", None, "started_at"),
            ("started_at", "2026-01-01T00:00:00", "started_at"),
        ],
    )
    def test_run_eval_suite_resume_requires_canonical_run_identity(
        self, tmp_path, field_name, value, expected_error
    ):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        state_path = output_root / "suite-run.json"
        state = json.loads(state_path.read_text())
        if value is None:
            state.pop(field_name)
        else:
            state[field_name] = value
        state_path.write_text(json.dumps(state, sort_keys=True) + "\n")

        with pytest.raises(ValueError, match=expected_error):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_run_eval_suite_resume_rejects_cross_case_verdict_replay(self, tmp_path):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path, case_count=2)
        )
        ledger_path = output_root / "suite-results.jsonl"
        rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
        rows[1]["result"] = rows[0]["result"]
        ledger_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )

        with pytest.raises(ValueError, match="uses different run, manifest, case"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_run_eval_suite_resume_rejects_cross_release_verdict_replay(self, tmp_path):
        manifest, _old_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        new_release = _write_test_corpus_release(
            tmp_path / "new-release",
            [
                {
                    "citation_path": "us/statute/7/2017",
                    "body": "different authoritative source text",
                }
            ],
        )
        state_path = output_root / "suite-run.json"
        state = json.loads(state_path.read_text())
        state.update(_test_eval_suite_release_identity(new_release))
        state_path.write_text(json.dumps(state, sort_keys=True) + "\n")

        with pytest.raises(ValueError, match="uses different run, manifest, case"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=new_release,
                resume_existing=True,
            )

    def test_run_eval_suite_resume_rejects_cross_rulespec_verdict_replay(
        self, tmp_path
    ):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        policy_repo_path = tmp_path / "rulespec-us"
        marker = policy_repo_path / "us" / "statutes" / "replay-marker.yaml"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("format: rulespec/v1\nrules: []\n")
        rulespec_roots = _eval_suite_rulespec_roots(manifest, policy_repo_path)
        new_execution_identity = _build_eval_suite_execution_identity(
            axiom_rules_path,
            rulespec_roots,
        )
        state_path = output_root / "suite-run.json"
        state = json.loads(state_path.read_text())
        state["execution_identity"] = new_execution_identity
        state["execution_identity_sha256"] = _eval_suite_execution_identity_sha256(
            new_execution_identity
        )
        state_path.write_text(json.dumps(state, sort_keys=True) + "\n")

        with pytest.raises(ValueError, match="uses different run, manifest, case"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=policy_repo_path,
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_resume_identity_rejects_different_policyengine_runtime(self):
        original_runtime = _test_policyengine_runtime("us")
        replacement_runtime = _test_policyengine_runtime("uk")

        def identity(runtime: PolicyEngineRuntime) -> dict[str, object]:
            return {
                "schema": "axiom-encode/eval-execution-identity/v3",
                "case_timeout_seconds": 3600,
                "runner_timeouts": {
                    "claude": {"wall_seconds": 1800},
                    "codex": {
                        "short_source": {
                            "wall_seconds": 600,
                            "idle_seconds": 300,
                        },
                        "long_source": {
                            "wall_seconds": 1800,
                            "idle_seconds": 900,
                        },
                        "long_source_char_threshold": 40_000,
                    },
                    "openai": {
                        "request_connect_seconds": 30,
                        "request_read_seconds": 180,
                    },
                },
                "timeout_retry_policy": {
                    "empty_artifact_max_attempts": 2,
                    "suite_max_attempts": 3,
                    "suite_retries_after_timeout": False,
                    "openai_request_max_attempts": 6,
                    "openai_request_backoff_seconds": [1, 2, 4, 8, 10],
                },
                "axiom_encode": {"tree_sha256": "1" * 64},
                "axiom_rules_engine": {"tree_sha256": "2" * 64},
                "policyengine_runtime": {
                    "identity": runtime.canonical_identity(),
                    "sha256": runtime.identity_sha256,
                },
                "rulespec_roots": [{"path": "/tmp/rulespec-us/us"}],
            }

        persisted_identity = identity(original_runtime)
        payload = {
            "execution_identity": persisted_identity,
            "execution_identity_sha256": _eval_suite_execution_identity_sha256(
                persisted_identity
            ),
        }

        with pytest.raises(
            ValueError,
            match="different PolicyEngine runtime identity",
        ):
            _validate_eval_suite_execution_identity(
                payload,
                identity(replacement_runtime),
            )

    def test_execution_identity_records_effective_timeout_and_retry_policy(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("AXIOM_ENCODE_ENCODER_TIMEOUT_SECONDS", "1234")
        monkeypatch.setenv("AXIOM_ENCODE_EVAL_CASE_TIMEOUT_SECONDS", "2400")
        monkeypatch.setenv("AXIOM_ENCODE_CODEX_TIMEOUT_SECONDS", "456")
        monkeypatch.setenv("AXIOM_ENCODE_CODEX_IDLE_TIMEOUT_SECONDS", "123")
        monkeypatch.setenv("AXIOM_ENCODE_CODEX_LONG_TIMEOUT_SECONDS", "2345")
        monkeypatch.setenv("AXIOM_ENCODE_CODEX_LONG_IDLE_TIMEOUT_SECONDS", "678")

        identity = _test_eval_suite_execution_identity()

        assert identity["schema"] == "axiom-encode/eval-execution-identity/v3"
        assert identity["case_timeout_seconds"] == 2400
        assert identity["runner_timeouts"] == {
            "claude": {"wall_seconds": 1234},
            "codex": {
                "short_source": {
                    "wall_seconds": 456,
                    "idle_seconds": 123,
                },
                "long_source": {
                    "wall_seconds": 2345,
                    "idle_seconds": 678,
                },
                "long_source_char_threshold": 40_000,
            },
            "openai": {
                "request_connect_seconds": 30,
                "request_read_seconds": 180,
            },
        }
        assert identity["timeout_retry_policy"] == {
            "empty_artifact_max_attempts": 2,
            "suite_max_attempts": 3,
            "suite_retries_after_timeout": False,
            "openai_request_max_attempts": 6,
            "openai_request_backoff_seconds": [1, 2, 4, 8, 10],
        }

    def test_resume_identity_rejects_different_claude_timeout(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("AXIOM_ENCODE_ENCODER_TIMEOUT_SECONDS", "1200")
        persisted_identity = _test_eval_suite_execution_identity()
        payload = {
            "execution_identity": persisted_identity,
            "execution_identity_sha256": _eval_suite_execution_identity_sha256(
                persisted_identity
            ),
        }
        monkeypatch.setenv("AXIOM_ENCODE_ENCODER_TIMEOUT_SECONDS", "1800")

        with pytest.raises(ValueError, match="execution identity"):
            _validate_eval_suite_execution_identity(
                payload,
                _test_eval_suite_execution_identity(),
            )

    def test_resume_identity_rejects_different_case_budget(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("AXIOM_ENCODE_EVAL_CASE_TIMEOUT_SECONDS", "2400")
        persisted_identity = _test_eval_suite_execution_identity()
        payload = {
            "execution_identity": persisted_identity,
            "execution_identity_sha256": _eval_suite_execution_identity_sha256(
                persisted_identity
            ),
        }
        monkeypatch.setenv("AXIOM_ENCODE_EVAL_CASE_TIMEOUT_SECONDS", "3600")

        with pytest.raises(ValueError, match="generation/retry case timeout"):
            _validate_eval_suite_execution_identity(
                payload,
                _test_eval_suite_execution_identity(),
            )

    def test_resume_identity_rejects_different_codex_timeout(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("AXIOM_ENCODE_CODEX_TIMEOUT_SECONDS", "600")
        persisted_identity = _test_eval_suite_execution_identity()
        payload = {
            "execution_identity": persisted_identity,
            "execution_identity_sha256": _eval_suite_execution_identity_sha256(
                persisted_identity
            ),
        }
        monkeypatch.setenv("AXIOM_ENCODE_CODEX_TIMEOUT_SECONDS", "601")

        with pytest.raises(ValueError, match="execution identity"):
            _validate_eval_suite_execution_identity(
                payload,
                _test_eval_suite_execution_identity(),
            )

    def test_execution_identity_records_effective_suite_attempt_limit(self):
        with patch(
            "axiom_encode.harness.evals._git_checkout_execution_identity",
            side_effect=lambda *_args, **_kwargs: {
                "kind": "tree",
                "tree_sha256": "1" * 64,
            },
        ):
            identity = _build_eval_suite_execution_identity(
                Path("/tmp/axiom-rules"),
                (),
                suite_retry_attempts=0,
            )

        assert identity["timeout_retry_policy"]["suite_max_attempts"] == 1

    def test_execution_identity_retry_policy_round_trips_nondefault_suite_attempts(
        self,
    ):
        with patch(
            "axiom_encode.harness.evals._git_checkout_execution_identity",
            side_effect=lambda *_args, **_kwargs: {
                "kind": "tree",
                "tree_sha256": "1" * 64,
            },
        ):
            identity = _build_eval_suite_execution_identity(
                Path("/tmp/axiom-rules"),
                (),
                suite_retry_attempts=0,
            )

        assert (
            evals_module._suite_retry_attempts_from_execution_identity(
                identity,
                artifact_name="test identity",
            )
            == 0
        )

    @pytest.mark.parametrize(
        "suite_max_attempts",
        [True, 0, "1"],
        ids=["boolean", "zero", "string"],
    )
    def test_execution_identity_retry_policy_rejects_malformed_suite_attempts(
        self,
        suite_max_attempts,
    ):
        identity = _test_eval_suite_execution_identity()
        identity["timeout_retry_policy"]["suite_max_attempts"] = suite_max_attempts

        with pytest.raises(ValueError, match="timeout retry policy"):
            evals_module._suite_retry_attempts_from_execution_identity(
                identity,
                artifact_name="test identity",
            )

    def test_run_eval_suite_resume_rejects_tampered_source_attestation(self, tmp_path):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        ledger_path = output_root / "suite-results.jsonl"
        row = json.loads(ledger_path.read_text())
        row["result"]["source_attestation"]["requested_corpus_citation_path"] = (
            "us/statute/7/2018"
        )
        row["result"] = _bind_eval_result_payload(row["result"])
        ledger_path.write_text(json.dumps(row, sort_keys=True) + "\n")

        with pytest.raises(ValueError, match="authenticated generation"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    @pytest.mark.parametrize("mutation", ["success", "metrics"])
    def test_run_eval_suite_resume_rejects_rehashed_mutable_verdict(
        self,
        tmp_path,
        mutation,
    ):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        ledger_path = output_root / "suite-results.jsonl"
        row = json.loads(ledger_path.read_text())
        if mutation == "success":
            row["result"]["success"] = False
            row["result"]["error"] = "tampered failure"
            row["result"]["failure_kind"] = "error"
        else:
            row["result"]["metrics"]["compile_pass"] = False
        row["result"] = _bind_eval_result_payload(row["result"])
        ledger_path.write_text(json.dumps(row, sort_keys=True) + "\n")

        with pytest.raises(ValueError, match="authenticated generation"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_run_eval_suite_resume_revalidates_without_rerunning_reviewer(
        self,
        tmp_path,
    ):
        """Resume recomputes deterministic verdicts; advisory review is not rerun.

        Generalist review is model-generated and nondeterministic, so a resume
        that rerun it could never reproduce the persisted output byte-for-byte
        and every legitimate resume of a reviewed suite would be refused.
        """
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        self.persisted_result_revalidation.reset_mock()

        results = run_eval_suite(
            manifest=manifest,
            output_root=output_root,
            axiom_rules_path=axiom_rules_path,
            policy_repo_path=tmp_path / "rulespec-us",
            corpus_release=corpus_release,
            resume_existing=True,
        )

        assert len(results) == 1
        self.persisted_result_revalidation.assert_called_once()
        assert (
            self.persisted_result_revalidation.call_args.kwargs["skip_reviewers"]
            is True
        )

    def test_revalidation_admission_authenticates_without_old_verdict_equality(
        self,
        tmp_path,
    ):
        from axiom_encode.harness.evals import (
            _build_eval_suite_manifest_identity,
            _load_eval_suite_resume_state,
        )

        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        policy_repo_path = tmp_path / "rulespec-us"
        rulespec_roots = _eval_suite_rulespec_roots(manifest, policy_repo_path)
        manifest_identity = _build_eval_suite_manifest_identity(manifest)
        execution_identity = _build_eval_suite_execution_identity(
            axiom_rules_path,
            rulespec_roots,
        )
        self.persisted_result_revalidation.reset_mock()

        _run_id, _started_at, results, completed = _load_eval_suite_resume_state(
            output_root=output_root,
            manifest=manifest,
            resolved_runners=list(manifest.runners),
            parsed_runners=[parse_runner_spec(spec) for spec in manifest.runners],
            corpus_release=corpus_release,
            axiom_rules_path=axiom_rules_path,
            policy_repo_path=policy_repo_path,
            rulespec_roots=rulespec_roots,
            manifest_identity=manifest_identity,
            execution_identity=execution_identity,
            revalidate_persisted_results=False,
        )

        assert len(results) == 1
        assert completed == {1}
        self.persisted_result_revalidation.assert_not_called()

    def _mutate_and_rehash_first_row(self, output_root, metrics_mutation):
        """Mutate persisted metrics with full verdict re-sign and rehash."""
        ledger_path = output_root / "suite-results.jsonl"
        row = json.loads(ledger_path.read_text())
        result_payload = row["result"]
        verdict_path = Path(result_payload["verdict_file"])
        verdict_payload = json.loads(verdict_path.read_text())
        for field_name, value in metrics_mutation.items():
            result_payload["metrics"][field_name] = value
            verdict_payload["validation"]["metrics"][field_name] = value
        verdict_payload["signature"] = sign_eval_evidence(
            verdict_payload,
            get_signing_broker(capability="eval_ed25519"),
        )
        verdict_raw = (
            json.dumps(verdict_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        verdict_path.write_bytes(verdict_raw)
        result_payload["verdict_sha256"] = hashlib.sha256(verdict_raw).hexdigest()
        row["result"] = _bind_eval_result_payload(result_payload)
        ledger_path.write_text(json.dumps(row, sort_keys=True) + "\n")

    def test_run_eval_suite_resume_rejects_fully_rehashed_verdict_mutation(
        self,
        tmp_path,
    ):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        self._mutate_and_rehash_first_row(output_root, {"policyengine_pass": True})

        with pytest.raises(
            ValueError, match="PolicyEngine evidence without its runtime identity"
        ):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )
        self.persisted_result_revalidation.assert_not_called()

    def test_run_eval_suite_resume_rejects_fully_rehashed_deterministic_mutation(
        self,
        tmp_path,
    ):
        """Deterministic metric drift is refused by recompute, not hydration.

        The mutation survives every hydration-layer guard (verdict re-signed,
        hashes rebound, no runtime-identity coupling), so the refusal must
        come from the deterministic revalidation comparison itself.
        """
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        row = json.loads((output_root / "suite-results.jsonl").read_text())
        original_count = row["result"]["metrics"]["grounded_numeric_count"]
        self._mutate_and_rehash_first_row(
            output_root, {"grounded_numeric_count": original_count + 4}
        )
        self.persisted_result_revalidation.reset_mock()

        with pytest.raises(ValueError, match="fresh validation of the bound artifact"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )
        self.persisted_result_revalidation.assert_called_once()
        assert (
            self.persisted_result_revalidation.call_args.kwargs["skip_reviewers"]
            is True
        )

    def test_run_eval_suite_resume_admits_resigned_advisory_review_mutation(
        self,
        tmp_path,
    ):
        """A key-holding mutation of advisory review fields is admitted.

        Recompute-equality cannot police the generalist reviewer: it is
        nondeterministic, so rerunning it rejects every legitimate resume
        (the failure this contract replaces). Advisory review integrity is
        signature-bound instead; this attacker re-signs with the live broker
        key, which no deterministic check can distinguish from a real run.
        Deterministic gate fields stay recompute-verified regardless.
        """
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        self._mutate_and_rehash_first_row(
            output_root,
            {"generalist_review_pass": False, "generalist_review_score": 1.0},
        )
        self.persisted_result_revalidation.reset_mock()

        results = run_eval_suite(
            manifest=manifest,
            output_root=output_root,
            axiom_rules_path=axiom_rules_path,
            policy_repo_path=tmp_path / "rulespec-us",
            corpus_release=corpus_release,
            resume_existing=True,
        )

        assert len(results) == 1
        assert results[0].metrics.generalist_review_pass is False
        assert results[0].metrics.generalist_review_score == 1.0
        assert (
            self.persisted_result_revalidation.call_args.kwargs["skip_reviewers"]
            is True
        )

    def test_run_eval_suite_resume_rejects_fully_rehashed_cost_laundering(
        self,
        tmp_path,
    ):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(
                tmp_path,
                gates={"max_mean_estimated_cost_usd": 0.0},
            )
        )
        ledger_path = output_root / "suite-results.jsonl"
        row = json.loads(ledger_path.read_text())
        result_payload = row["result"]
        verdict_path = Path(result_payload["verdict_file"])
        verdict_payload = json.loads(verdict_path.read_text())
        result_payload["estimated_cost_usd"] = 0.0
        verdict_payload["generation"]["estimated_cost_usd"] = 0.0
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
            "reasoning_output_tokens",
        ):
            result_payload[field_name] = 0
            verdict_payload["generation"][field_name] = 0
        verdict_raw = (
            json.dumps(verdict_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        verdict_path.write_bytes(verdict_raw)
        result_payload["verdict_sha256"] = hashlib.sha256(verdict_raw).hexdigest()
        row["result"] = _bind_eval_result_payload(result_payload)
        ledger_path.write_text(json.dumps(row, sort_keys=True) + "\n")

        with pytest.raises(ValueError, match="signature is invalid"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )
        self.persisted_result_revalidation.assert_not_called()

    def test_run_eval_suite_resume_rejects_resigned_error_mismatch(self, tmp_path):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        ledger_path = output_root / "suite-results.jsonl"
        row = json.loads(ledger_path.read_text())
        result_payload = row["result"]
        verdict_path = Path(result_payload["verdict_file"])
        verdict_payload = json.loads(verdict_path.read_text())
        result_payload["error"] = "forged validation error"
        verdict_payload["validation"]["error"] = "forged validation error"
        verdict_payload["signature"] = sign_eval_evidence(
            verdict_payload,
            get_signing_broker(capability="eval_ed25519"),
        )
        verdict_raw = (
            json.dumps(verdict_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        verdict_path.write_bytes(verdict_raw)
        result_payload["verdict_sha256"] = hashlib.sha256(verdict_raw).hexdigest()
        row["result"] = _bind_eval_result_payload(result_payload)
        ledger_path.write_text(json.dumps(row, sort_keys=True) + "\n")

        with pytest.raises(ValueError, match="success, error, or metrics"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_run_eval_suite_resume_rejects_resigned_mode_mismatch(self, tmp_path):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        ledger_path = output_root / "suite-results.jsonl"
        row = json.loads(ledger_path.read_text())
        result_payload = row["result"]
        verdict_path = Path(result_payload["verdict_file"])
        verdict_payload = json.loads(verdict_path.read_text())
        result_payload["mode"] = "cold"
        verdict_payload["identity"]["mode"] = "cold"
        verdict_payload["signature"] = sign_eval_evidence(
            verdict_payload,
            get_signing_broker(capability="eval_ed25519"),
        )
        verdict_raw = (
            json.dumps(verdict_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        verdict_path.write_bytes(verdict_raw)
        result_payload["verdict_sha256"] = hashlib.sha256(verdict_raw).hexdigest()
        row["result"] = _bind_eval_result_payload(result_payload)
        ledger_path.write_text(json.dumps(row, sort_keys=True) + "\n")

        with pytest.raises(ValueError, match="different mode"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_run_eval_suite_resume_rejects_cross_runner_artifact_substitution(
        self,
        tmp_path,
    ):
        runners = ["first=openai:gpt-5.4", "second=codex:gpt-5.4"]
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path, runners=runners)
        )
        ledger_path = output_root / "suite-results.jsonl"
        rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
        source_result = rows[0]["result"]
        target_result = rows[1]["result"]
        verdict_path = Path(target_result["verdict_file"])
        verdict_payload = json.loads(verdict_path.read_text())
        for path_field, digest_field in (
            ("output_file", "generated_output_sha256"),
            ("trace_file", "trace_sha256"),
            ("context_manifest_file", "context_manifest_sha256"),
        ):
            target_result[path_field] = source_result[path_field]
            target_result[digest_field] = source_result[digest_field]
            verdict_payload["artifacts"][digest_field] = source_result[digest_field]
        verdict_payload["signature"] = sign_eval_evidence(
            verdict_payload,
            get_signing_broker(capability="eval_ed25519"),
        )
        verdict_raw = (
            json.dumps(verdict_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        verdict_path.write_bytes(verdict_raw)
        target_result["verdict_sha256"] = hashlib.sha256(verdict_raw).hexdigest()
        rows[1]["result"] = _bind_eval_result_payload(target_result)
        ledger_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )

        with pytest.raises(ValueError, match="runner-owned artifact directory"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    @pytest.mark.parametrize(
        "path_field",
        ["output_file", "trace_file", "context_manifest_file"],
    )
    @pytest.mark.parametrize("mutation", ["change", "delete"])
    def test_run_eval_suite_resume_rejects_mutated_or_missing_result_artifacts(
        self,
        tmp_path,
        path_field,
        mutation,
    ):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        ledger_row = json.loads((output_root / "suite-results.jsonl").read_text())
        artifact_path = Path(ledger_row["result"][path_field])
        if mutation == "change":
            artifact_path.write_bytes(artifact_path.read_bytes() + b"\nmutated\n")
            expected_error = "bytes do not match"
        else:
            artifact_path.unlink()
            expected_error = "could not safely load"

        with pytest.raises(ValueError, match=expected_error):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_run_eval_suite_resume_rejects_missing_result_artifact_digest(
        self,
        tmp_path,
    ):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        ledger_path = output_root / "suite-results.jsonl"
        ledger_row = json.loads(ledger_path.read_text())
        ledger_row["result"].pop("trace_sha256")
        ledger_row["result"] = _bind_eval_result_payload(ledger_row["result"])
        ledger_path.write_text(json.dumps(ledger_row) + "\n")

        with pytest.raises(ValueError, match="missing immutable model trace digest"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_run_eval_suite_resume_rejects_same_path_manifest_edit(self, tmp_path):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        manifest.path.write_text(manifest.path.read_text() + "\n# changed bytes\n")
        changed_manifest = load_eval_suite_manifest(manifest.path)

        with pytest.raises(ValueError, match="different manifest content"):
            run_eval_suite(
                manifest=changed_manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_run_eval_suite_resume_rejects_rulespec_content_change(self, tmp_path):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        rulespec_file = tmp_path / "rulespec-us" / "us" / "statutes" / "changed.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text("format: rulespec/v1\nrules: []\n")

        with pytest.raises(ValueError, match="different RuleSpec content"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_run_eval_suite_resume_rejects_rulespec_checkout_sibling_change(
        self, tmp_path
    ):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        sibling_file = tmp_path / "rulespec-us" / "us-ca" / "statutes" / "new.yaml"
        sibling_file.parent.mkdir(parents=True)
        sibling_file.write_text("format: rulespec/v1\nrules: []\n")

        with pytest.raises(
            ValueError,
            match="different canonical RuleSpec root identity|different RuleSpec content",
        ):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_rulespec_execution_identity_scopes_working_tree_to_rulespec_inputs(
        self, tmp_path
    ):
        from axiom_encode.harness.evals import _rulespec_root_execution_identity

        _manifest, _release, _output, _engine = _complete_test_eval_suite(tmp_path)
        checkout = tmp_path / "rulespec-us"
        runtime_pin = checkout / ".axiom" / "policyengine-runtime.toml"
        runtime_pin_bytes = (
            b'[policyengine_runtime]\nschema = "axiom-policyengine-runtime-pin/v1"\n'
            b'git_commit = "1111111111111111111111111111111111111111"\n'
        )
        runtime_pin.write_bytes(runtime_pin_bytes)
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        subprocess.run(
            ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "config", "user.name", "Test"],
            check=True,
        )
        subprocess.run(["git", "-C", str(checkout), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(checkout), "commit", "-qm", "fixture"],
            check=True,
        )

        clean_identity = _rulespec_root_execution_identity(checkout / "us")
        checkout_identity = clean_identity["checkout_identity"]
        assert checkout_identity["kind"] == "git"
        assert checkout_identity["pathspecs"] == [
            "us",
            ".axiom/toolchain.toml",
            ".axiom/policyengine-runtime.toml",
            "known-validation-gaps.yaml",
        ]
        assert (
            clean_identity["policyengine_runtime_pin_sha256"]
            == hashlib.sha256(runtime_pin_bytes).hexdigest()
        )
        assert checkout_identity["dirty"] is False
        assert (
            checkout_identity["commit"]
            == subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )

        (checkout / "README.md").write_text("non-executable checkout notes\n")
        (checkout / ".git" / "FETCH_HEAD").write_text("mutable git metadata\n")
        irrelevant_identity = _rulespec_root_execution_identity(checkout / "us")
        assert irrelevant_identity == clean_identity

        changed_pin_bytes = runtime_pin_bytes.replace(b"1111", b"2222", 1)
        runtime_pin.write_bytes(changed_pin_bytes)
        dirty_pin_identity = _rulespec_root_execution_identity(checkout / "us")
        assert (
            dirty_pin_identity["policyengine_runtime_pin_sha256"]
            == hashlib.sha256(changed_pin_bytes).hexdigest()
        )
        assert dirty_pin_identity["checkout_identity"]["dirty"] is True
        assert (
            dirty_pin_identity["checkout_identity"]["working_tree_sha256"]
            != checkout_identity["working_tree_sha256"]
        )
        runtime_pin.write_bytes(runtime_pin_bytes)

        rulespec_file = checkout / "us" / "statutes" / "untracked.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text("format: rulespec/v1\nrules: []\n")
        dirty_identity = _rulespec_root_execution_identity(checkout / "us")
        assert dirty_identity["checkout_identity"]["dirty"] is True
        assert (
            dirty_identity["checkout_identity"]["working_tree_sha256"]
            != checkout_identity["working_tree_sha256"]
        )
        assert dirty_identity["content_sha256"] != clean_identity["content_sha256"]

    def test_run_state_atomic_replace_failure_preserves_previous_state(self, tmp_path):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        state_path = output_root / "suite-run.json"
        previous_state = state_path.read_bytes()

        with (
            patch(
                "axiom_encode.harness.evals.os.replace",
                side_effect=OSError("replace failed"),
            ),
            pytest.raises(OSError, match="replace failed"),
        ):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

        assert state_path.read_bytes() == previous_state
        assert not list(output_root.glob(".suite-run.json.*.tmp"))

    def test_run_eval_suite_resume_rejects_missing_ledger_with_progress(self, tmp_path):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        (output_root / "suite-results.jsonl").unlink()

        with pytest.raises(ValueError, match="indicates progress.*is missing"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    @pytest.mark.parametrize("runner_mutation", ["duplicate", "unknown"])
    def test_run_eval_suite_resume_rejects_invalid_runner_rows(
        self,
        tmp_path,
        runner_mutation,
    ):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        ledger_path = output_root / "suite-results.jsonl"
        row = json.loads(ledger_path.read_text())
        if runner_mutation == "duplicate":
            ledger_path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            expected_error = "duplicate runner"
        else:
            row["result"]["runner"] = "unknown-runner"
            ledger_path.write_text(json.dumps(row) + "\n")
            expected_error = "unknown runner"

        with pytest.raises(ValueError, match=expected_error):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [("case_name", "different-case"), ("case_kind", "citation")],
    )
    def test_run_eval_suite_resume_rejects_wrong_case_identity(
        self,
        tmp_path,
        field,
        value,
    ):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        ledger_path = output_root / "suite-results.jsonl"
        row = json.loads(ledger_path.read_text())
        row[field] = value
        ledger_path.write_text(json.dumps(row) + "\n")

        with pytest.raises(ValueError, match="wrong case identity"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_run_eval_suite_resume_rejects_partial_runner_group(self, tmp_path):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(
                tmp_path,
                runners=["openai:gpt-5.4", "codex:gpt-5.4"],
            )
        )
        ledger_path = output_root / "suite-results.jsonl"
        first_row = ledger_path.read_text().splitlines()[0]
        ledger_path.write_text(first_row + "\n")

        with pytest.raises(ValueError, match="incomplete runner group"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_run_eval_suite_resume_rejects_waiver_and_contract_change(self, tmp_path):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        rulespec_checkout = tmp_path / "rulespec-us"
        waiver_bytes = b"validate_failures:\n  changed-gap: changed reason\n"
        waiver_digest = hashlib.sha256(waiver_bytes).hexdigest()
        (rulespec_checkout / "known-validation-gaps.yaml").write_bytes(waiver_bytes)
        (rulespec_checkout / ".axiom" / "toolchain.toml").write_text(
            "[toolchain]\n"
            f'axiom_corpus_release = "{_TEST_CORPUS_RELEASE_NAME}"\n'
            f'axiom_corpus_release_content_sha256 = "{corpus_release.content_sha256}"\n'
            f'validation_waiver_set_sha256 = "{waiver_digest}"\n'
        )

        with pytest.raises(ValueError, match="validation waiver-set identity"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_run_eval_suite_resume_rejects_engine_content_change(self, tmp_path):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )
        (axiom_rules_path / "Cargo.toml").write_text("[package]\nname = 'changed'\n")

        with pytest.raises(ValueError, match="axiom-rules-engine execution identity"):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_run_eval_suite_resume_rejects_encoder_version_change(self, tmp_path):
        manifest, corpus_release, output_root, axiom_rules_path = (
            _complete_test_eval_suite(tmp_path)
        )

        with (
            patch("axiom_encode.harness.evals.__version__", "999.0.0"),
            pytest.raises(ValueError, match="axiom-encode execution identity"),
        ):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=axiom_rules_path,
                policy_repo_path=tmp_path / "rulespec-us",
                corpus_release=corpus_release,
                resume_existing=True,
            )

    def test_run_eval_suite_rejects_duplicate_effective_runner_names(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            "name: duplicate runners\n"
            "runners:\n"
            "  - same=openai:gpt-5.4\n"
            "  - same=codex:gpt-5.4\n"
            "cases:\n"
            "  - kind: source\n"
            "    name: case-one\n"
            "    corpus_citation_path: us/statute/7/2017\n"
        )
        with pytest.raises(ValueError, match="runner names must be unique"):
            load_eval_suite_manifest(manifest_file)

    @pytest.mark.parametrize(
        ("manifest_filename", "expected_corpus_paths"),
        [
            (
                "us_co_colorado_works_seed.yaml",
                [
                    "us-co/regulation/9-ccr-2503-6/3.606.1/F",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/G",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/H",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/I",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/K",
                ],
            ),
            (
                "us_co_colorado_works_leaf_seed.yaml",
                [
                    "us-co/regulation/9-ccr-2503-6/3.606.1/E",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/G",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/H",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/I",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/J",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/K",
                ],
            ),
            (
                "us_co_colorado_works_leaf_repair.yaml",
                [
                    "us-co/regulation/9-ccr-2503-6/3.606.1/G",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/H",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/I",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/J",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/K",
                ],
            ),
            (
                "us_co_colorado_works_leaf_k_repair.yaml",
                ["us-co/regulation/9-ccr-2503-6/3.606.1/K"],
            ),
            (
                "us_co_colorado_works_leaf_h_repair.yaml",
                ["us-co/regulation/9-ccr-2503-6/3.606.1/H"],
            ),
            (
                "us_co_colorado_works_leaf_closeout.yaml",
                [
                    "us-co/regulation/9-ccr-2503-6/3.606.1/H",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/K",
                ],
            ),
            (
                "us_snap_federal_reconstruction_seed.yaml",
                [
                    "us/statute/7/2017/a",
                    "us/statute/7/2017/c/1",
                    "us/statute/7/2017/c/3",
                    "us/guidance/usda/fns/snap-fy2026-cola/page-1",
                ],
            ),
            ("us_snap_federal_c3_repair.yaml", ["us/statute/7/2017/c/3"]),
            (
                "us_snap_fy2026_cola_table_repair.yaml",
                ["us/guidance/usda/fns/snap-fy2026-cola/page-1"],
            ),
            ("us_snap_asset_test_refresh.yaml", ["us/statute/7/2014/g/1"]),
            (
                "us_snap_asset_test_current_effective_refresh.yaml",
                ["us/guidance/usda/fns/snap-fy2026-cola/page-2"],
            ),
            ("us_snap_eligibility_refresh.yaml", ["us/statute/7/2014"]),
            (
                "us_snap_earned_income_deduction_refresh.yaml",
                ["us/statute/7/2014/e/2/B"],
            ),
            (
                "us_snap_net_income_pre_shelter_refresh.yaml",
                ["us/statute/7/2014/e/6/A"],
            ),
            (
                "us_snap_co_self_employment_expense_option_refresh.yaml",
                ["us-co/regulation/10-ccr-2506-1/4.403.11"],
            ),
            (
                "us_snap_co_child_support_deduction_option_refresh.yaml",
                ["us-co/regulation/10-ccr-2506-1/4.407.5"],
            ),
        ],
    )
    def test_repo_benchmark_manifests_are_corpus_backed(
        self,
        manifest_filename,
        expected_corpus_paths,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / manifest_filename
        )

        assert manifest.mode == "repo-augmented"
        assert [
            case.corpus_citation_path for case in manifest.cases
        ] == expected_corpus_paths
        assert all(case.kind == "source" for case in manifest.cases)
        for case in manifest.cases:
            for context_path in case.allow_context:
                assert "sources" not in context_path.parts


class TestReadinessSummary:
    def test_summarize_readiness_applies_suite_gates(self):
        gates = EvalReadinessGates(
            min_cases=3,
            min_success_rate=1.0,
            min_compile_pass_rate=1.0,
            min_ci_pass_rate=1.0,
            min_zero_ungrounded_rate=1.0,
            min_generalist_review_pass_rate=1.0,
            min_policyengine_pass_rate=0.8,
            max_mean_estimated_cost_usd=0.5,
        )
        results = [
            _fake_eval_result(
                "codex-gpt-5.4",
                "case-a",
                compile_pass=True,
                ci_pass=True,
                policyengine_pass=True,
                policyengine_score=1.0,
                estimated_cost_usd=0.20,
            ),
            _fake_eval_result(
                "codex-gpt-5.4",
                "case-b",
                compile_pass=True,
                ci_pass=True,
                generalist_review_pass=False,
                generalist_review_score=4.0,
                policyengine_pass=False,
                policyengine_score=0.5,
                estimated_cost_usd=0.40,
            ),
            _fake_eval_result(
                "codex-gpt-5.4",
                "case-c",
                compile_pass=True,
                ci_pass=True,
                generalist_review_pass=True,
                generalist_review_score=7.5,
                policyengine_pass=None,
                policyengine_score=None,
                estimated_cost_usd=0.30,
            ),
        ]

        summary = summarize_readiness(results, gates)

        assert summary.total_cases == 3
        assert summary.compile_pass_rate == 1.0
        assert summary.ci_pass_rate == 1.0
        assert summary.zero_ungrounded_rate == 1.0
        assert summary.generalist_review_pass_rate == pytest.approx(
            2 / 3, rel=0, abs=1e-6
        )
        assert summary.mean_generalist_review_score == pytest.approx(6.5)
        assert summary.policyengine_case_count == 2
        assert summary.policyengine_pass_rate == 0.5
        assert summary.mean_estimated_cost_usd == 0.3
        assert summary.ready is False
        gate_results = {gate.name: gate for gate in summary.gate_results}
        assert gate_results["min_cases"].passed is True
        assert gate_results["min_generalist_review_pass_rate"].passed is False
        assert gate_results["min_policyengine_pass_rate"].passed is False
        assert gate_results["max_mean_estimated_cost_usd"].passed is True

    def test_timeout_rows_are_excluded_from_artifact_readiness_denominators(self):
        artifact = _fake_eval_result(
            "runner",
            "case-a",
            compile_pass=True,
            ci_pass=True,
            generalist_review_pass=True,
        )
        timeout = replace(
            _fake_eval_result("runner", "case-b"),
            output_file="",
            trace_file="",
            context_manifest_file="",
            generated_output_sha256=None,
            trace_sha256=None,
            context_manifest_sha256=None,
            success=False,
            error="encoder timed out",
            metrics=None,
            failure_kind="timeout",
            timed_out=True,
            timeout_stage="encoder",
            timeout_reason="wall",
            timeout_seconds=600,
            timeout_attempts=1,
        )

        summary = summarize_readiness(
            [artifact, timeout],
            EvalReadinessGates(
                min_compile_pass_rate=1.0,
                min_ci_pass_rate=1.0,
                min_zero_ungrounded_rate=1.0,
                min_generalist_review_pass_rate=1.0,
            ),
        )

        assert summary.total_cases == 2
        assert summary.artifact_case_count == 1
        assert summary.timeout_count == 1
        assert summary.success_rate == 0.5
        assert summary.compile_pass_rate == 1.0
        assert summary.ci_pass_rate == 1.0
        assert summary.zero_ungrounded_rate == 1.0
        assert summary.generalist_review_pass_rate == 1.0
        assert summary.ready is True

    def test_all_timeout_readiness_has_no_artifact_rates(self):
        timeout = replace(
            _fake_eval_result("runner", "case-a"),
            output_file="",
            trace_file="",
            context_manifest_file="",
            generated_output_sha256=None,
            trace_sha256=None,
            context_manifest_sha256=None,
            success=False,
            error="encoder timed out",
            metrics=None,
            failure_kind="timeout",
            timed_out=True,
            timeout_stage="encoder",
            timeout_reason="wall",
            timeout_seconds=600,
            timeout_attempts=1,
        )

        summary = summarize_readiness(
            [timeout],
            EvalReadinessGates(
                min_compile_pass_rate=1.0,
                min_ci_pass_rate=1.0,
                min_zero_ungrounded_rate=1.0,
                min_generalist_review_pass_rate=1.0,
            ),
        )

        assert summary.artifact_case_count == 0
        assert summary.timeout_count == 1
        assert summary.compile_pass_rate is None
        assert summary.ci_pass_rate is None
        assert summary.zero_ungrounded_rate is None
        assert summary.generalist_review_pass_rate is None
        assert all(gate.actual is None for gate in summary.gate_results[1:])
        assert summary.ready is False

    def test_policyengine_exception_without_score_counts_as_oracle_failure(self):
        passed = _fake_eval_result(
            "runner",
            "case-a",
            policyengine_pass=True,
            policyengine_score=1.0,
        )
        errored = _fake_eval_result(
            "runner",
            "case-b",
            policyengine_pass=False,
            policyengine_score=None,
        )
        errored.metrics.policyengine_issues = ["oracle raised"]

        summary = summarize_readiness(
            [passed, errored],
            EvalReadinessGates(
                min_generalist_review_pass_rate=None,
                min_policyengine_pass_rate=1.0,
            ),
        )

        assert summary.policyengine_case_count == 2
        assert summary.policyengine_pass_rate == 0.5
        assert summary.mean_policyengine_score == 1.0
        assert summary.ready is False

    def test_policyengine_pass_does_not_require_optional_score(self):
        result = _fake_eval_result(
            "runner",
            "case-a",
            policyengine_pass=True,
            policyengine_score=None,
        )

        assert (
            evals_module._eval_artifact_validation_error(
                result.metrics,
                require_policyengine=True,
            )
            is None
        )

    @pytest.mark.parametrize(
        "cost",
        [None, -0.01, float("nan"), float("inf")],
        ids=["missing", "negative", "nan", "infinite"],
    )
    def test_cost_gate_fails_without_complete_finite_nonnegative_evidence(
        self,
        cost,
    ):
        results = [
            _fake_eval_result("runner", "case-a", estimated_cost_usd=0.05),
            _fake_eval_result("runner", "case-b", estimated_cost_usd=cost),
        ]

        summary = summarize_readiness(
            results,
            EvalReadinessGates(
                min_generalist_review_pass_rate=None,
                max_mean_estimated_cost_usd=0.1,
            ),
        )

        cost_gate = next(
            gate
            for gate in summary.gate_results
            if gate.name == "max_mean_estimated_cost_usd"
        )
        assert summary.mean_estimated_cost_usd is None
        assert cost_gate.actual is None
        assert cost_gate.passed is False
        assert summary.ready is False

    @pytest.mark.parametrize(
        ("field_name", "value", "expected_error"),
        [
            ("input_tokens", -1, "nonnegative accounting"),
            ("estimated_cost_usd", -0.01, "nonnegative finite cost"),
            ("estimated_cost_usd", float("nan"), "nonnegative finite cost"),
        ],
    )
    def test_persisted_accounting_rejects_negative_or_nonfinite_values(
        self,
        field_name,
        value,
        expected_error,
    ):
        payload = _fake_eval_result(
            "runner",
            "case-a",
            estimated_cost_usd=None,
        ).to_dict()
        payload[field_name] = value
        payload = _bind_eval_result_payload(payload)

        with pytest.raises(ValueError, match=expected_error):
            _eval_result_from_payload(payload)


class TestRepoAugmentedContext:
    def test_repo_augmented_context_rejects_engine_root(self, tmp_path):
        engine = tmp_path / "axiom-rules-engine"
        engine.mkdir()
        sibling = tmp_path / "rulespec-us"
        sibling.mkdir()

        with pytest.raises(
            UnsafeRulespecContextPath, match="exact direct jurisdiction"
        ):
            _repo_augmented_context_root(engine)

    @pytest.mark.parametrize("alias_kind", ["checkout", "content"])
    def test_repo_augmented_context_rejects_symlinked_root(
        self,
        tmp_path,
        alias_kind,
    ):
        real_checkout = tmp_path / "real" / "rulespec-us"
        real_content = real_checkout / "us-co"
        real_content.mkdir(parents=True)
        if alias_kind == "checkout":
            alias_checkout = tmp_path / "alias" / "rulespec-us"
            alias_checkout.parent.mkdir()
            alias_checkout.symlink_to(real_checkout, target_is_directory=True)
            aliased_root = alias_checkout / "us-co"
        else:
            alias_checkout = tmp_path / "alias" / "rulespec-us"
            alias_checkout.mkdir(parents=True)
            aliased_root = alias_checkout / "us-co"
            aliased_root.symlink_to(real_content, target_is_directory=True)

        with pytest.raises(
            UnsafeRulespecContextPath,
            match="exact direct jurisdiction",
        ):
            _repo_augmented_context_root(aliased_root)

    def test_hydration_omits_cross_authority_hidden_dependency_copies(self, tmp_path):
        rulespec_us = _canonical_rulespec_content_root(tmp_path, "us")
        rulespec_uk = _canonical_rulespec_content_root(tmp_path, "uk")
        relative = Path("statutes/1/shared.yaml")
        us_file = rulespec_us / relative
        uk_file = rulespec_uk / relative
        for path, marker in ((us_file, "US authority"), (uk_file, "UK authority")):
            path.parent.mkdir(parents=True)
            path.write_text(
                f"format: rulespec/v1\nmodule:\n  summary: {marker}\nrules: []\n"
            )

        workspace = prepare_eval_workspace(
            citation="custom-source",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Primary source text.",
            axiom_rules_path=rulespec_us,
            mode="repo-augmented",
            extra_context_paths=[us_file, uk_file],
        )

        items = {item.import_path: item for item in workspace.context_files}
        assert items["us:statutes/1/shared"].workspace_path == (
            "context/statutes/1/shared.yaml"
        )
        assert items["uk:statutes/1/shared"].workspace_path == (
            "context/rulespec-uk/statutes/1/shared.yaml"
        )

        eval_root = tmp_path / "eval-root"
        _hydrate_eval_root(eval_root, workspace)
        assert "US authority" in (eval_root / "statutes/1/shared.yaml").read_text()
        assert not (eval_root / "_axiom").exists()
        assert not (eval_root / "rulespec-uk").exists()

    def test_select_context_files_rejects_symlinked_section_scan_root(self, tmp_path):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        section_root = policy_repo_root / "statutes" / "26" / "24"
        section_root.parent.mkdir(parents=True)
        outside_root = tmp_path / "outside"
        outside_root.mkdir()
        (outside_root / "secret.yaml").write_text(
            "OPENAI_API_KEY: sentinel-secret-value\n"
        )
        section_root.symlink_to(outside_root, target_is_directory=True)

        with pytest.raises(UnsafeRulespecContextPath, match="directory.*symlink"):
            select_context_files("26 USC 24(a)", policy_repo_root)

    def test_prepare_eval_workspace_rejects_symlinked_child_scan_root(self, tmp_path):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        child_root = policy_repo_root / "statutes" / "26" / "152"
        child_root.parent.mkdir(parents=True)
        outside_root = tmp_path / "outside"
        outside_root.mkdir()
        (outside_root / "secret.yaml").write_text(
            "OPENAI_API_KEY: sentinel-secret-value\n"
        )
        child_root.symlink_to(outside_root, target_is_directory=True)

        with pytest.raises(UnsafeRulespecContextPath, match="directory.*symlink"):
            prepare_eval_workspace(
                citation="26 USC 152",
                runner=parse_runner_spec("openai:gpt-5.4"),
                output_root=tmp_path / "out",
                source_text="Section source text.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

    def test_cited_context_selection_rejects_symlink_before_export_probe(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        cited_file = policy_repo_root / "statutes" / "26" / "152.yaml"
        cited_file.parent.mkdir(parents=True)
        outside_file = tmp_path / "outside-secret.yaml"
        outside_file.write_text("OPENAI_API_KEY: sentinel-secret-value\n")
        cited_file.symlink_to(outside_file)

        with pytest.raises(UnsafeRulespecContextPath, match="symlink"):
            _select_cross_section_context_files(
                "26 USC 151",
                "A dependent is defined in section 152.",
                policy_repo_root,
            )

    def test_prepare_eval_workspace_rejects_symlinked_context_before_reading(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        context_file = policy_repo_root / "statutes" / "26" / "24" / "b.yaml"
        context_file.parent.mkdir(parents=True)
        outside_file = tmp_path / "outside-secret.yaml"
        outside_file.write_text("OPENAI_API_KEY: sentinel-secret-value\n")
        context_file.symlink_to(outside_file)

        with pytest.raises(UnsafeRulespecContextPath, match="symlink"):
            prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=parse_runner_spec("openai:gpt-5.4"),
                output_root=tmp_path / "out",
                source_text="A child tax credit is allowed.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        workspace_files = [
            path
            for path in (tmp_path / "out").rglob("*")
            if path.is_file() and not path.is_symlink()
        ]
        assert all(
            "sentinel-secret-value" not in path.read_text(errors="replace")
            for path in workspace_files
        )

    def test_prepare_eval_workspace_rejects_context_outside_rulespec_roots(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        outside_file = tmp_path / "private" / "context.yaml"
        outside_file.parent.mkdir()
        outside_file.write_text("OPENAI_API_KEY: sentinel-secret-value\n")
        context_file = policy_repo_root / "statutes" / "26" / "24" / "b.yaml"
        context_file.parent.mkdir(parents=True)
        context_file.write_text(
            f"format: rulespec/v1\nimports:\n  - {outside_file.as_posix()}\nrules: []\n"
        )

        with pytest.raises(
            UnsafeRulespecContextPath,
            match="outside the active policy root",
        ):
            prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=parse_runner_spec("openai:gpt-5.4"),
                output_root=tmp_path / "out",
                source_text="Primary source text.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

    def test_prepare_eval_workspace_rejects_symlinked_canonical_companion_test(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        concept_file = policy_repo_root / "statutes" / "26" / "1402" / "b.yaml"
        concept_file.parent.mkdir(parents=True)
        concept_file.write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: '\"Self-employment income\" means net earnings.'\n"
            "rules:\n"
            "  - name: self_employment_income\n"
            "    kind: derived\n"
            "    entity: TaxUnit\n"
            "    dtype: Money\n"
            "    period: Year\n"
            "    versions:\n"
            "      - effective_from: '2026-01-01'\n"
            "        formula: 0\n"
        )
        outside_file = tmp_path / "outside-secret.yaml"
        outside_file.write_text("OPENAI_API_KEY: sentinel-secret-value\n")
        concept_file.with_name("b.test.yaml").symlink_to(outside_file)

        with pytest.raises(UnsafeRulespecContextPath, match="symlink"):
            prepare_eval_workspace(
                citation="26 USC 1401(a)",
                runner=parse_runner_spec("openai:gpt-5.4"),
                output_root=tmp_path / "out",
                source_text="The self-employment income is subject to tax.",
                axiom_rules_path=policy_repo_root,
                mode="cold",
                extra_context_paths=[],
            )

    def test_prepare_eval_workspace_allows_arbitrary_identifier_with_explicit_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        context_file = (
            _canonical_rulespec_content_root(repo_root, "us")
            / "statutes"
            / "26"
            / "32"
            / "b"
            / "2"
            / "A.yaml"
        )
        context_file.parent.mkdir(parents=True)
        context_file.write_text("format: rulespec/v1\nrules: []\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1(F)",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="F. Determining Eligibility ... 165 345 518",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[context_file],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        assert manifest["mode"] == "repo-augmented"
        assert manifest["context_files"][0]["source_path"] == str(context_file)
        assert manifest["context_files"][0]["import_path"] == "us:statutes/26/32/b/2/A"
        copied = workspace.root / manifest["context_files"][0]["workspace_path"]
        assert copied.exists()

    def test_prepare_eval_workspace_copies_existing_corpus_target(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us-ny")
        target_file = (
            policy_repo_root / "regulations" / "18-nycrr" / "387" / "12" / "f.yaml"
        )
        target_file.parent.mkdir(parents=True)
        target_file.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: existing_provenance\n"
            "    kind: source_relation\n"
        )
        target_file.with_name("f.test.yaml").write_text("- name: existing_case\n")

        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="us-ny/regulation/18-nycrr/387/12/f",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="Existing NY regulation text.",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(target_file)]["kind"] == "existing_target"
        assert (
            copied_sources[str(target_file.with_name("f.test.yaml"))]["kind"]
            == "existing_target_test_context"
        )

    def test_select_context_files_excludes_target(self, tmp_path):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        section_dir = policy_repo_root / "statutes" / "26" / "24"
        section_dir.mkdir(parents=True)
        (section_dir / "a.yaml").write_text("target")
        (section_dir / "b.yaml").write_text("sibling b")
        (section_dir / "c.yaml").write_text("sibling c")

        selected = select_context_files("26 USC 24(a)", policy_repo_root)

        assert section_dir / "a.yaml" not in selected
        assert section_dir / "b.yaml" in selected
        assert section_dir / "c.yaml" in selected

    @pytest.mark.parametrize(
        "citation",
        [
            "42/1437c\u20131/d",
            CitationParts(title="42", section="1437c\u20131", fragments=("d",)),
        ],
        ids=["slash-alias", "structured-citation"],
    )
    def test_select_context_files_uses_normalized_section_path(
        self,
        tmp_path,
        citation,
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        title_dir = policy_repo_root / "statutes" / "42"
        section_dir = title_dir / "1437c-1"
        section_dir.mkdir(parents=True)
        sibling = section_dir / "e.yaml"
        sibling.write_text("same-section sibling")
        for index in range(6):
            (title_dir / f"{index}.yaml").write_text("unrelated title context")

        selected = select_context_files(citation, policy_repo_root)

        assert sibling in selected

    def test_prepare_eval_workspace_writes_manifest_and_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        statute_root = (
            _canonical_rulespec_content_root(repo_root, "us") / "statutes" / "26" / "24"
        )
        statute_root.mkdir(parents=True)
        context_file = statute_root / "b.yaml"
        context_file.write_text("format: rulespec/v1\nrules: []\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        assert manifest["mode"] == "repo-augmented"
        assert manifest["source_text_file"] == "source.txt"
        assert manifest["context_files"][0]["source_path"] == str(context_file)
        assert manifest["context_files"][0]["import_path"] == "us:statutes/26/24/b"
        copied = workspace.root / manifest["context_files"][0]["workspace_path"]
        assert copied.exists()

    def test_prepare_eval_workspace_copies_context_companion_tests(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        statute_root = (
            _canonical_rulespec_content_root(repo_root, "us") / "statutes" / "26" / "24"
        )
        statute_root.mkdir(parents=True)
        context_file = statute_root / "b.yaml"
        context_test = statute_root / "b.test.yaml"
        context_file.write_text("format: rulespec/v1\nrules: []\n")
        context_test.write_text("- name: context_case\n  period: 2026-01\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(context_file)]["kind"] == "implementation_precedent"
        assert (
            copied_sources[str(context_test)]["kind"] == "implementation_test_context"
        )
        copied_test = (
            workspace.root / copied_sources[str(context_test)]["workspace_path"]
        )
        assert copied_test.read_text() == "- name: context_case\n  period: 2026-01\n"

    def test_prepare_eval_workspace_canonical_concepts_use_absolute_imports_and_tests(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        statute_root = policy_repo_root / "statutes" / "26" / "1402"
        statute_root.mkdir(parents=True)
        context_file = statute_root / "b.yaml"
        context_test = statute_root / "b.test.yaml"
        context_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    (b) Self-employment income The term "self-employment income" means the net earnings from self-employment; except that the section 1401(a) cap applies.
rules:
  - name: self_employment_income_for_section_1401_a
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: net_earnings_from_self_employment
"""
        )
        context_test.write_text(
            "- name: context_case\n"
            "  input:\n"
            "    us:statutes/26/1402/a#input.self_employment_trade_or_business_gross_income: 1000\n"
            "  output:\n"
            "    us:statutes/26/1402/b#self_employment_income_for_section_1401_a: 923.5\n"
        )

        workspace = prepare_eval_workspace(
            citation="26 USC 1401(a)",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "There shall be imposed on the self-employment income of every "
                "individual a tax equal to 12.4 percent."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            (item["source_path"], item["kind"]): item
            for item in manifest["context_files"]
        }
        canonical_item = copied_sources[(str(context_file), "canonical_concept")]
        assert canonical_item["import_path"] == "us:statutes/26/1402/b"
        assert (
            "`self-employment income` -> import "
            "`us:statutes/26/1402/b#self_employment_income_for_section_1401_a`"
            in canonical_item["label"]
        )
        companion_item = copied_sources[
            (str(context_test), "implementation_test_context")
        ]
        assert companion_item["import_path"] == "us:statutes/26/1402/b.test"
        copied_test = workspace.root / companion_item["workspace_path"]
        assert copied_test.read_text() == context_test.read_text()
        assert not any(
            item["import_path"] == "statutes/26/1402/b"
            for item in manifest["context_files"]
        )

    def test_prepare_eval_workspace_copies_existing_target_file_as_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        target = policy_repo_root / "statutes" / "26" / "3111" / "a.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: employer_oasdi_excise_tax\n"
            "    kind: derived\n"
        )
        target_test = target.with_name("a.test.yaml")
        target_test.write_text("- name: existing_case\n  period: 2026-01\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="us/statute/26/3111/a",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="3111(a) imposes 6.2 percent employer OASDI tax.",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(target)]["kind"] == "existing_target"
        assert copied_sources[str(target)]["import_path"] == "us:statutes/26/3111/a"
        assert (
            copied_sources[str(target_test)]["kind"] == "existing_target_test_context"
        )

    def test_build_eval_prompt_preserves_existing_executable_surface(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        target = policy_repo_root / "statutes" / "26" / "45A" / "a.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: qualified_wages\n"
            "    kind: derived\n"
            "    entity: Employer\n"
            "    dtype: Money\n"
            "    period: Year\n"
        )

        workspace = prepare_eval_workspace(
            citation="us/statute/26/45A/a",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text="The amount of the credit shall be 20 percent of qualified wages.",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 45A(a)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="statutes/26/45A/a.yaml",
            target_ref_prefix="us:statutes/26/45A/a",
            include_tests=True,
            runner_backend="openai",
        )

        assert "copied current target files as context" in prompt
        assert "not as general backward compatibility contracts" in prompt
        assert (
            "Keep an old output only when it remains the cleanest source-faithful "
            "RuleSpec surface" in prompt
        )
        assert "Never preserve, rename, or recreate a legacy local input" in prompt
        assert "source-stated formula executable" in prompt
        assert "defer only that branch" in prompt
        assert "Do not treat a missing deferred child branch\n  as zero" in prompt
        assert 'excess of" a cap' in prompt
        assert "min(source_amount, cap)" in prompt

    def test_prepare_eval_workspace_adds_same_section_subsection_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        statute_root = (
            _canonical_rulespec_content_root(repo_root, "us")
            / "statutes"
            / "7"
            / "2015"
        )
        statute_root.mkdir(parents=True)
        context_file = statute_root / "e.yaml"
        context_test = statute_root / "e.test.yaml"
        context_file.write_text("format: rulespec/v1\nrules: []\n")
        context_test.write_text("- name: student_exception_case\n  period: 2026-01\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="7 USC 2015(d)(2)(C)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=(
                    "A higher education student is ineligible unless the student "
                    "meets the requirements of subsection (e) of this section."
                ),
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(context_file)]["kind"] == "implementation_precedent"
        assert (
            copied_sources[str(context_file)]["import_path"] == "us:statutes/7/2015/e"
        )
        assert (
            copied_sources[str(context_test)]["kind"] == "implementation_test_context"
        )

    def test_prepare_eval_workspace_adds_country_monorepo_child_context(self, tmp_path):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")
        child_root = policy_repo_root / "statutes" / "26" / "36B" / "b" / "3"
        child_root.mkdir(parents=True)
        child_file = child_root / "A.yaml"
        child_file.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: applicable_percentage_income_tier\n"
            "    kind: derived\n"
            "    entity: TaxUnit\n"
            "    dtype: Integer\n"
            "    period: Year\n"
            "    versions:\n"
            "      - effective_from: '2024-01-01'\n"
            "        formula: 0\n"
        )

        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="26 USC 36B",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="Section 36B defines the premium assistance credit amount.",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(child_file)]["kind"] == "implementation_precedent"
        assert (
            copied_sources[str(child_file)]["workspace_path"]
            == "context/statutes/26/36B/b/3/A.yaml"
        )
        assert (
            copied_sources[str(child_file)]["import_path"] == "us:statutes/26/36B/b/3/A"
        )

    def test_prepare_eval_workspace_adds_plural_same_section_subsection_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us-co")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        additions_root = policy_repo_root / "statutes" / "39" / "39-22-104" / "3"
        subtractions_root = policy_repo_root / "statutes" / "39" / "39-22-104" / "4"
        additions_root.mkdir(parents=True)
        subtractions_root.mkdir(parents=True)
        addition_parent = additions_root.with_suffix(".yaml")
        subtraction_parent = subtractions_root.with_suffix(".yaml")
        addition_parent.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: subsection_3_additions\n"
            "    kind: derived\n"
            "    entity: TaxUnit\n"
            "    dtype: Money\n"
            "    period: Year\n"
            "    formula: 0\n"
        )
        subtraction_parent.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: subsection_4_subtractions\n"
            "    kind: derived\n"
            "    entity: TaxUnit\n"
            "    dtype: Money\n"
            "    period: Year\n"
            "    formula: 0\n"
        )
        for label in "abcdefghij":
            addition_file = additions_root / f"{label}.yaml"
            addition_test = additions_root / f"{label}.test.yaml"
            subtraction_file = subtractions_root / f"{label}.yaml"
            subtraction_test = subtractions_root / f"{label}.test.yaml"
            addition_file.write_text("format: rulespec/v1\nrules: []\n")
            addition_test.write_text(f"- name: addition_{label}_case\n  period: 2026\n")
            subtraction_file.write_text("format: rulespec/v1\nrules: []\n")
            subtraction_test.write_text(
                f"- name: subtraction_{label}_case\n  period: 2026\n"
            )
        addition_file = additions_root / "d.yaml"
        addition_test = additions_root / "d.test.yaml"
        late_addition_file = additions_root / "j.yaml"
        late_addition_test = additions_root / "j.test.yaml"
        subtraction_file = subtractions_root / "a.yaml"
        subtraction_test = subtractions_root / "a.test.yaml"
        late_subtraction_file = subtractions_root / "j.yaml"
        late_subtraction_test = subtractions_root / "j.test.yaml"

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="us-co/statute/39/39-22-104/2",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=(
                    "Federal taxable income shall be modified as provided in "
                    "subsections (3) and (4) of this section."
                ),
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(addition_file)]["kind"] == "implementation_precedent"
        assert (
            copied_sources[str(addition_parent)]["kind"] == "implementation_precedent"
        )
        assert (
            copied_sources[str(addition_file)]["import_path"]
            == "us-co:statutes/39/39-22-104/3/d"
        )
        assert (
            copied_sources[str(addition_test)]["kind"] == "implementation_test_context"
        )
        assert (
            copied_sources[str(late_addition_file)]["kind"]
            == "implementation_precedent"
        )
        assert (
            copied_sources[str(late_addition_test)]["kind"]
            == "implementation_test_context"
        )
        assert (
            copied_sources[str(subtraction_file)]["import_path"]
            == "us-co:statutes/39/39-22-104/4/a"
        )
        assert (
            copied_sources[str(subtraction_parent)]["kind"]
            == "implementation_precedent"
        )
        assert (
            copied_sources[str(subtraction_test)]["kind"]
            == "implementation_test_context"
        )
        assert (
            copied_sources[str(late_subtraction_file)]["kind"]
            == "implementation_precedent"
        )
        assert (
            copied_sources[str(late_subtraction_test)]["kind"]
            == "implementation_test_context"
        )

    def test_prepare_eval_workspace_adds_same_section_under_subsection_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        statute_root = (
            _canonical_rulespec_content_root(repo_root, "us")
            / "statutes"
            / "26"
            / "3121"
        )
        statute_root.mkdir(parents=True)
        context_file = statute_root / "y.yaml"
        context_test = statute_root / "y.test.yaml"
        context_file.write_text("format: rulespec/v1\nrules: []\n")
        context_test.write_text("- name: transferred_employee_case\n  period: 2026\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 3121(b)(15)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=(
                    "Service performed in the employ of an international "
                    "organization, except service which constitutes employment "
                    "under subsection (y)."
                ),
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(context_file)]["kind"] == "implementation_precedent"
        assert (
            copied_sources[str(context_file)]["import_path"] == "us:statutes/26/3121/y"
        )
        assert (
            copied_sources[str(context_test)]["kind"] == "implementation_test_context"
        )

    def test_prepare_eval_workspace_adds_nested_same_section_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        subsection_root = (
            _canonical_rulespec_content_root(repo_root, "us")
            / "statutes"
            / "26"
            / "3121"
            / "a"
        )
        subsection_root.mkdir(parents=True)
        cap_file = subsection_root / "1.yaml"
        domestic_file = subsection_root / "7.yaml"
        domestic_test = subsection_root / "7.test.yaml"
        cap_file.write_text("format: rulespec/v1\nrules: []\n")
        domestic_file.write_text("format: rulespec/v1\nrules: []\n")
        domestic_test.write_text("- name: domestic_service_case\n  period: 2026\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 3121(i)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=(
                    "Wages shall be subject to the provisions of subsection "
                    "(a)(1) of this section. Domestic service described in "
                    "subsection (a)(7)(B) shall be computed to the nearest dollar."
                ),
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(cap_file)]["kind"] == "implementation_precedent"
        assert copied_sources[str(cap_file)]["import_path"] == "us:statutes/26/3121/a/1"
        assert (
            copied_sources[str(domestic_file)]["import_path"]
            == "us:statutes/26/3121/a/7"
        )
        assert (
            copied_sources[str(domestic_test)]["kind"] == "implementation_test_context"
        )

    def test_prepare_eval_workspace_adds_cross_section_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        context_root = (
            _canonical_rulespec_content_root(repo_root, "us")
            / "statutes"
            / "26"
            / "104"
            / "a"
        )
        context_root.mkdir(parents=True)
        context_file = context_root / "4.yaml"
        context_test = context_root / "4.test.yaml"
        context_file.write_text("format: rulespec/v1\nrules: []\n")
        context_test.write_text("- name: service_injury_case\n  period: 2026\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 22",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=(
                    "No reduction shall be made for any amount described in "
                    "section 104(a)(4)."
                ),
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(context_file)]["kind"] == "implementation_precedent"
        assert (
            copied_sources[str(context_file)]["import_path"] == "us:statutes/26/104/a/4"
        )
        assert (
            copied_sources[str(context_test)]["kind"] == "implementation_test_context"
        )

    def test_prepare_eval_workspace_adds_state_regulation_cross_section_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us-co")
        regulations_root = policy_repo_root / "regulations" / "10-ccr-2506-1"
        regulations_root.mkdir(parents=True)
        disqualification_period = regulations_root / "4.803.2.yaml"
        disqualification_period.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: consent_agreement_disqualification_start_deadline_calendar_days\n"
            "    kind: parameter\n"
            "    dtype: Integer\n"
            "    period: Year\n"
            "    versions:\n"
            "      - effective_from: '2026-01-01'\n"
            "        formula: '30'\n"
        )
        fair_hearing_parent = regulations_root / "4.411.yaml"
        fair_hearing_parent.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: fair_hearing_request_deadline_days\n"
            "    kind: parameter\n"
            "    dtype: Integer\n"
            "    period: Year\n"
            "    versions:\n"
            "      - effective_from: '2026-01-01'\n"
            "        formula: '90'\n"
        )
        source_text = (
            "Disqualification shall continue uninterrupted until completed, "
            "regardless of household eligibility, and shall be imposed in "
            "accordance with Section 4.803.2, F unless contrary to the court "
            "order. The household may also request a hearing under Section "
            "4.411.1."
        )

        selected = _select_cross_section_context_files(
            "us-co/regulation/10-ccr-2506-1/4.804.1",
            source_text,
            policy_repo_root,
        )

        assert selected == [disqualification_period, fair_hearing_parent]

        workspace = prepare_eval_workspace(
            citation="us-co/regulation/10-ccr-2506-1/4.804.1",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=source_text,
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            source_metadata_payload={
                "source_attestation": {
                    "requested_corpus_citation_path": (
                        "us-co/regulation/10-ccr-2506-1/4.804.1"
                    )
                },
            },
            extra_context_paths=[],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert (
            copied_sources[str(disqualification_period)]["import_path"]
            == "us-co:regulations/10-ccr-2506-1/4.803.2"
        )
        assert (
            copied_sources[str(fair_hearing_parent)]["import_path"]
            == "us-co:regulations/10-ccr-2506-1/4.411"
        )

        prompt = _build_eval_prompt(
            "us-co/regulation/10-ccr-2506-1/4.804.1",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="4.804.1.yaml",
            target_ref_prefix="us-co:regulations/10-ccr-2506-1/4.804.1",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Mandatory cited RuleSpec imports detected" in prompt
        assert "`us-co:regulations/10-ccr-2506-1/4.803.2`" in prompt
        assert "`us-co:regulations/10-ccr-2506-1/4.411`" in prompt
        assert "Missing cited RuleSpec sources detected" not in prompt
        assert "us:statutes/us-co:regulations" not in prompt

    def test_prepare_eval_workspace_adds_child_context_for_unavailable_cited_parent(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        section_root = (
            _canonical_rulespec_content_root(repo_root, "us")
            / "statutes"
            / "26"
            / "3101"
        )
        section_root.mkdir(parents=True)
        parent = section_root.with_suffix(".yaml")
        parent.write_text(
            "format: rulespec/v1\nmodule:\n  status: entity_not_supported\nrules: []\n"
        )
        oasdi = section_root / "a.yaml"
        oasdi.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: oasdi_wage_tax_rate\n"
            "    kind: parameter\n"
            "    versions:\n"
            "      - effective_from: '1990-01-01'\n"
            "        formula: '0.062'\n"
        )
        hi = section_root / "b" / "1.yaml"
        hi.parent.mkdir()
        hi.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: hospital_insurance_wage_tax_rate\n"
            "    kind: parameter\n"
            "    versions:\n"
            "      - effective_from: '1986-01-01'\n"
            "        formula: '0.0145'\n"
        )
        source_text = (
            "For purposes of the preceding sentence, the term applicable "
            "percentage means the percentage equal to the sum of the rates "
            "of tax in effect under subsections (a) and (b) of section 3101 "
            "for the calendar year."
        )

        selected = _select_cross_section_context_files(
            "26 USC 3201",
            source_text,
            _canonical_rulespec_content_root(repo_root, "us"),
        )

        assert selected == [parent, oasdi, hi]

        runner = parse_runner_spec("openai:gpt-5.5")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 3201",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=source_text,
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(parent)]["import_path"] == "us:statutes/26/3101"
        assert copied_sources[str(oasdi)]["import_path"] == "us:statutes/26/3101/a"
        assert copied_sources[str(hi)]["import_path"] == "us:statutes/26/3101/b/1"

    def test_prepare_eval_workspace_adds_child_rate_context_for_exporting_cited_parent(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        section_root = (
            _canonical_rulespec_content_root(repo_root, "us")
            / "statutes"
            / "26"
            / "1401"
        )
        section_root.mkdir(parents=True)
        parent = section_root.with_suffix(".yaml")
        parent.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: self_employment_oasdi_tax_rate\n"
            "    kind: parameter\n"
            "    versions:\n"
            "      - effective_from: '1990-01-01'\n"
            "        formula: '0.124'\n"
        )
        oasdi_rate = section_root / "a" / "rate.yaml"
        oasdi_rate.parent.mkdir(parents=True)
        oasdi_rate.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: old_age_survivors_and_disability_insurance_tax_rate\n"
            "    kind: parameter\n"
            "    versions:\n"
            "      - effective_from: '1990-01-01'\n"
            "        formula: '0.124'\n"
        )
        hi_rate = section_root / "b" / "1" / "rate.yaml"
        hi_rate.parent.mkdir(parents=True)
        hi_rate.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: self_employment_income_tax_rate\n"
            "    kind: parameter\n"
            "    versions:\n"
            "      - effective_from: '1990-01-01'\n"
            "        formula: '0.029'\n"
        )
        source_text = (
            "There shall be allowed a deduction equal to the product of net "
            "earnings and one-half of the sum of the rates imposed by "
            "subsections (a) and (b) of section 1401."
        )

        selected = _select_cross_section_context_files(
            "26 USC 1402(a)(12)",
            source_text,
            _canonical_rulespec_content_root(repo_root, "us"),
        )

        assert parent in selected
        assert oasdi_rate in selected
        assert hi_rate in selected

    def test_prepare_eval_workspace_adds_cross_section_list_and_parent_fallback_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        rules_root = (
            _canonical_rulespec_content_root(repo_root, "us") / "statutes" / "26"
        )

        section_911_child = rules_root / "911" / "a.yaml"
        section_911_child.parent.mkdir(parents=True)
        section_911_child.write_text("format: rulespec/v1\nrules: []\n")
        section_931 = rules_root / "931.yaml"
        section_931.write_text("format: rulespec/v1\nrules: []\n")
        section_933 = rules_root / "933.yaml"
        section_933.write_text("format: rulespec/v1\nrules: []\n")
        source_text = (
            "Modified adjusted gross income means adjusted gross income "
            "increased by any amount excluded from gross income under "
            "sections 911, 931, or 933."
        )

        selected = _select_cross_section_context_files(
            "26 USC 151",
            source_text,
            _canonical_rulespec_content_root(repo_root, "us"),
        )

        assert section_911_child in selected
        assert section_931 in selected
        assert section_933 in selected

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 151",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=source_text,
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert (
            copied_sources[str(section_911_child)]["kind"] == "implementation_precedent"
        )
        assert (
            copied_sources[str(section_911_child)]["import_path"]
            == "us:statutes/26/911/a"
        )
        assert copied_sources[str(section_931)]["import_path"] == "us:statutes/26/931"
        assert copied_sources[str(section_933)]["import_path"] == "us:statutes/26/933"

    def test_prepare_eval_workspace_adds_cross_section_ancestor_context_for_deep_citation(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        section_3511 = (
            _canonical_rulespec_content_root(repo_root, "us")
            / "statutes"
            / "26"
            / "3511.yaml"
        )
        section_3511.parent.mkdir(parents=True)
        section_3511.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: specified_credit_applies_to_customer_not_cpeo\n"
            "    kind: derived\n"
            "    entity: Employer\n"
            "    dtype: Judgment\n"
            "    period: Year\n"
        )
        source_text = (
            "Any credit allowed under this section shall be treated as a "
            "credit described in section 3511(d)(2)."
        )

        selected = _select_cross_section_context_files(
            "26 USC 3134(i)",
            source_text,
            _canonical_rulespec_content_root(repo_root, "us"),
        )

        assert selected == [section_3511]

        runner = parse_runner_spec("codex:gpt-5.5")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 3134(i)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=source_text,
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(section_3511)]["kind"] == "implementation_precedent"
        assert copied_sources[str(section_3511)]["import_path"] == "us:statutes/26/3511"

    def test_build_eval_prompt_warns_on_unavailable_cited_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        section_152 = policy_repo_root / "statutes" / "26" / "152.yaml"
        section_152.parent.mkdir(parents=True)
        section_152.write_text(
            "format: rulespec/v1\nmodule:\n  status: entity_not_supported\nrules: []\n"
        )

        workspace = prepare_eval_workspace(
            citation="26 USC 151",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text="A dependent is defined in section 152.",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 151",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="151.yaml",
            target_ref_prefix="us:statutes/26/151",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Unavailable cited RuleSpec context detected" in prompt
        assert (
            "`us:statutes/26/152` has `module.status: entity_not_supported`" in prompt
        )
        assert "`_under_section_152`" in prompt
        assert "omit or defer only the affected executable surface" in prompt

    def test_build_eval_prompt_warns_on_unavailable_cited_context_ancestors(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        section_408_p_2_a = (
            policy_repo_root / "statutes" / "26" / "408" / "p" / "2" / "A.yaml"
        )
        section_408_p_2_a.parent.mkdir(parents=True)
        section_408_p_2_a.write_text(
            "format: rulespec/v1\nmodule:\n  status: entity_not_supported\nrules: []\n"
        )

        workspace = prepare_eval_workspace(
            citation="26 USC 3121(a)(5)(H)",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "under an arrangement to which section 408(p) applies, "
                "other than elective contributions under paragraph (2)(A)(i) thereof"
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 3121(a)(5)(H)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="H.yaml",
            target_ref_prefix="us:statutes/26/3121/a/5/H",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Unavailable cited RuleSpec context detected" in prompt
        assert "`408_p_2_A`, `408_p_2`, `408_p`" in prompt
        assert "`*_to_which_section_408_p_applies`" in prompt

    def test_prepare_eval_workspace_adds_child_fragment_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        child_root = (
            _canonical_rulespec_content_root(repo_root, "us")
            / "statutes"
            / "7"
            / "2015"
            / "d"
            / "2"
        )
        child_root.mkdir(parents=True)
        child_files = []
        for fragment in ("A", "B", "C", "D", "E", "F"):
            child_file = child_root / f"{fragment}.yaml"
            child_file.write_text("format: rulespec/v1\nrules: []\n")
            child_files.append(child_file)
        nested_child_file = child_root / "G" / "1.yaml"
        nested_child_file.parent.mkdir()
        nested_child_file.write_text("format: rulespec/v1\nrules: []\n")
        child_files.append(nested_child_file)

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="7 USC 2015(d)(2)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="A person shall be exempt if subparagraphs (A) through (F) apply.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        for child_file in child_files:
            assert copied_sources[str(child_file)]["kind"] == "implementation_precedent"
            assert copied_sources[str(child_file)]["import_path"] == "us:" + (
                child_file.relative_to(
                    _canonical_rulespec_content_root(repo_root, "us")
                )
                .with_suffix("")
                .as_posix()
            )

    def test_prepare_eval_workspace_materializes_corpus_source_metadata(self, tmp_path):
        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="snap_sua_tn",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="Tennessee source text",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            source_metadata_payload={
                "relations": [
                    {
                        "relation": "sets",
                        "target": "us:regulation/7-cfr/273/9/d/6/iii#snap_standard_utility_allowance",
                        "jurisdiction": "TN",
                    }
                ]
            },
            extra_context_paths=[],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        assert manifest["source_metadata_file"] == "source-metadata.json"
        assert (
            manifest["source_metadata"]["relations"][0]["target"]
            == "us:regulation/7-cfr/273/9/d/6/iii#snap_standard_utility_allowance"
        )
        assert workspace.source_metadata_file is not None
        assert workspace.source_metadata_file.exists()

    def test_prepare_eval_workspace_canonicalizes_crlf_before_hash_and_write(
        self, tmp_path
    ):
        metadata = {"source_attestation": {}}
        workspace = prepare_eval_workspace(
            citation="us/statute/1",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="First\r\nSecond\rThird\r\n",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us"),
            mode="cold",
            source_metadata_payload=metadata,
            extra_context_paths=[],
        )

        assert workspace.source_text_file.read_bytes() == b"First\nSecond\nThird\n"
        assert metadata["source_attestation"]["generation_input_sha256"] == (
            hashlib.sha256(workspace.source_text_file.read_bytes()).hexdigest()
        )

    def test_build_eval_prompt_lists_canonical_context_import_target(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        external_file = (
            _canonical_rulespec_content_root(repo_root, "us-co")
            / "regulations"
            / "9-CCR-2503-6"
            / "3.606.1"
            / "F.yaml"
        )
        external_file.parent.mkdir(parents=True, exist_ok=True)
        external_file.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: grant_standard_for_assistance_unit\n"
            "    kind: input\n"
            "    entity: TanfUnit\n"
            "    dtype: Money\n"
            "    period: Month\n"
        )

        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1(I)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Deduct the total from step 2, above, from the grant amount.",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[external_file],
        )

        prompt = _build_eval_prompt(
            "9 CCR 2503-6 3.606.1(I)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="9-CCR-2503-6-3.606.1-I.yaml",
        )

        assert (
            "inspect `context/rulespec-us-co/regulations/"
            "9-CCR-2503-6/3.606.1/F.yaml`; "
            "import target `us-co:regulations/9-CCR-2503-6/3.606.1/F`"
        ) in prompt
        expected_hash = (
            "sha256:" + hashlib.sha256(external_file.read_bytes()).hexdigest()
        )
        assert f"context hash `{expected_hash}`" in prompt
        assert (
            "exports `us-co:regulations/9-CCR-2503-6/3.606.1/F#grant_standard_for_assistance_unit`"
            in prompt
        )
        assert "import.output" in prompt
        assert "import.hash" in prompt
        assert "use `hash: sha256:local`" in prompt
        assert "never use `sha256:self`" in prompt
        assert "do not wrap import targets in quotes" in prompt
        assert (
            "Use the listed import target rather than the `./context/...` inspection path"
            in prompt
        )
        assert (
            "do not guess contradictory expectations for those imported values"
            in prompt
        )
        assert (
            "keep `.test.yaml` inputs and expected outputs consistent with the rows visible in that imported file"
            in prompt
        )
        assert (
            "Do not invent degenerate placeholder rows like `number_of_children_in_assistance_unit: 0` plus `number_of_caretakers_in_assistance_unit: 0`"
            in prompt
        )
        assert (
            "Do not assert an exact zero imported standard, grant, or threshold unless that exact imported row is visible in the copied chart file"
            in prompt
        )
        assert (
            "In formulas, reference imported exports by their bare local rule name"
            in prompt
        )
        assert "import and use the listed exported symbol from that" in prompt
        assert "context instead of creating a local `section_...`" in prompt
        assert (
            "never write an absolute `us:...#rule_name` reference inside a formula"
            in prompt
        )

    def test_build_eval_prompt_flags_child_branch_sibling_name_collisions(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        child_root = (
            _canonical_rulespec_content_root(repo_root, "us")
            / "statutes"
            / "7"
            / "2015"
            / "d"
            / "2"
        )
        child_root.mkdir(parents=True)
        for fragment in ("A", "B"):
            child_file = child_root / f"{fragment}.yaml"
            child_file.write_text(
                "format: rulespec/v1\n"
                "rules:\n"
                "  - name: person_exempt_from_paragraph_1_work_requirements\n"
                "    kind: derived\n"
                "    entity: Person\n"
                "    dtype: Judgment\n"
                "    period: Month\n"
            )

        workspace = prepare_eval_workspace(
            citation="7 USC 2015(d)(2)(A)",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "A person otherwise required to comply with paragraph (1) shall be "
                "exempt if the person is subject to and complying with any work "
                "registration requirement under title IV or the Federal-State "
                "unemployment compensation system."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "7 USC 2015(d)(2)(A)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="A.yaml",
            target_ref_prefix="us:statutes/7/2015/d/2/A",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Sibling export naming for this target" in prompt
        assert "`person_exempt_from_paragraph_1_work_requirements`" in prompt
        assert "copied target currently exports invalid colliding names" in prompt
        assert "Do not export any local rule with a copied sibling's name" in prompt
        assert "not the shared parent consequence" in prompt
        assert "treat that name as stale and rename it" in prompt

    def test_build_eval_prompt_qualifies_generic_relation_when_sibling_reserved(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        statute_root = (
            _canonical_rulespec_content_root(repo_root, "us") / "statutes" / "26"
        )
        statute_root.mkdir(parents=True)
        sibling_file = statute_root / "32.yaml"
        sibling_file.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: qualifying_child_of_tax_unit\n"
            "    kind: data_relation\n"
            "    data_relation:\n"
            "      predicate: qualifying_child_of_tax_unit\n"
            "      arity: 2\n"
            "      arguments:\n"
            "        - TaxUnit\n"
            "        - Person\n"
        )

        workspace = prepare_eval_workspace(
            citation="26 USC 24",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "There shall be allowed a credit with respect to each qualifying "
                "child of the taxpayer."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )
        context_file = EvalContextFile(
            source_path=sibling_file,
            workspace_path=Path("context/statutes/26/32.yaml"),
            import_path="us:statutes/26/32",
            kind="implementation_precedent",
        )

        prompt = _build_eval_prompt(
            "26 USC 24",
            "repo-augmented",
            workspace,
            [context_file],
            target_file_name="24.yaml",
            target_ref_prefix="us:statutes/26/24",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Sibling export naming for this target" in prompt
        assert "`qualifying_child_of_tax_unit`" in prompt
        assert "Make the relation source-specific" in prompt
        assert "copied target currently exports invalid colliding names" not in prompt

    def test_hydrate_eval_root_copies_context_into_import_tree(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        statute_root = (
            _canonical_rulespec_content_root(repo_root, "us") / "statutes" / "26" / "24"
        )
        statute_root.mkdir(parents=True)
        context_file = statute_root / "c.yaml"
        context_file.write_text(
            "format: rulespec/v1\nmodule:\n  status: stub\nrules: []\n"
        )

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        eval_root = tmp_path / "eval-root"
        _hydrate_eval_root(eval_root, workspace)

        assert (eval_root / "statutes" / "26" / "24" / "c.yaml").read_text() == (
            "format: rulespec/v1\nmodule:\n  status: stub\nrules: []\n"
        )

    def test_hydrate_eval_root_preserves_generated_target_and_copies_sibling(
        self, tmp_path
    ):
        workspace_root = tmp_path / "workspace"
        context_root = workspace_root / "context" / "statutes" / "47"
        context_root.mkdir(parents=True)
        old_target = context_root / "294.yaml"
        old_target_test = context_root / "294.test.yaml"
        sibling = context_root / "295.yaml"
        old_target.write_text("old target\n", encoding="utf-8")
        old_target_test.write_text("old target test\n", encoding="utf-8")
        sibling.write_text("sibling context\n", encoding="utf-8")
        workspace = EvalWorkspace(
            root=workspace_root,
            source_text_file=workspace_root / "source.txt",
            manifest_file=workspace_root / "context-manifest.json",
            context_files=[
                EvalContextFile(
                    source_path=old_target,
                    workspace_path=Path("context/statutes/47/294.yaml"),
                    import_path="us-la:statutes/47/294",
                    kind="implementation_precedent",
                ),
                EvalContextFile(
                    source_path=old_target_test,
                    workspace_path=Path("context/statutes/47/294.test.yaml"),
                    import_path="us-la:statutes/47/294.test",
                    kind="existing_target_test_context",
                ),
                EvalContextFile(
                    source_path=sibling,
                    workspace_path=Path("context/statutes/47/295.yaml"),
                    import_path="us-la:statutes/47/295",
                    kind="implementation_precedent",
                ),
            ],
            policy_prefix="us-la",
        )
        eval_root = tmp_path / "eval-root"
        generated_target = eval_root / "statutes" / "47" / "294.yaml"
        generated_target.parent.mkdir(parents=True)
        generated_target.write_text("generated target\n", encoding="utf-8")
        generated_target_test = eval_root / "statutes" / "47" / "294.test.yaml"
        generated_target_test.write_text("generated target test\n", encoding="utf-8")

        _hydrate_eval_root(
            eval_root,
            workspace,
            protected_paths=(
                Path("statutes/47/294.yaml"),
                Path("statutes/47/294.test.yaml"),
            ),
        )

        assert generated_target.read_text(encoding="utf-8") == "generated target\n"
        assert (
            generated_target_test.read_text(encoding="utf-8")
            == "generated target test\n"
        )
        assert (eval_root / "statutes/47/295.yaml").read_text(
            encoding="utf-8"
        ) == "sibling context\n"

    def test_prepare_eval_workspace_expands_transitive_context_imports(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)

        section_root = (
            _canonical_rulespec_content_root(repo_root, "us") / "statutes" / "26" / "24"
        )
        section_root.mkdir(parents=True)
        aggregator = section_root / "24.yaml"
        aggregator.write_text(
            "format: rulespec/v1\n"
            "imports:\n"
            "  - statutes/26/24/a#ctc_allowance\n"
            "  - statutes/26/24/c#qualifying_child_count\n"
            "rules:\n"
            "  - name: section_24_credit\n"
            "    kind: derived\n"
            "    entity: TaxUnit\n"
            "    dtype: Money\n"
            "    period: Year\n"
        )
        selected = section_root / "c.yaml"
        selected.write_text(
            "format: rulespec/v1\n"
            "imports:\n"
            "  - us:statutes/26/24/c/2#ctc_meets_citizenship_requirement\n"
            "  - us:statutes/26/152/c#qualifying_child_of_taxpayer\n"
            "rules:\n"
            "  - name: qualifying_child_count\n"
            "    kind: derived\n"
            "    entity: TaxUnit\n"
            "    dtype: Integer\n"
            "    period: Year\n"
        )

        dep_local = section_root / "c" / "2.yaml"
        dep_local.parent.mkdir(parents=True)
        dep_local.write_text("format: rulespec/v1\nrules: []\n")

        dep_cross_section = (
            _canonical_rulespec_content_root(repo_root, "us")
            / "statutes"
            / "26"
            / "152"
            / "c.yaml"
        )
        dep_cross_section.parent.mkdir(parents=True)
        dep_cross_section.write_text("format: rulespec/v1\nrules: []\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[aggregator, selected],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item["kind"] for item in manifest["context_files"]
        }

        assert copied_sources[str(selected)] == "implementation_precedent"
        assert copied_sources[str(dep_local)] == "implementation_dependency"
        assert copied_sources[str(dep_cross_section)] == "implementation_dependency"
        assert str(section_root / "a.yaml") not in copied_sources

        eval_root = tmp_path / "eval-root"
        _hydrate_eval_root(eval_root, workspace)
        assert (
            eval_root / "statutes" / "26" / "24" / "c" / "2.yaml"
        ).read_text() == "format: rulespec/v1\nrules: []\n"
        assert (
            eval_root / "statutes" / "26" / "152" / "c.yaml"
        ).read_text() == "format: rulespec/v1\nrules: []\n"

    def test_build_eval_prompt_flags_existing_target_unresolved_import(self, tmp_path):
        repo_root = tmp_path / "repos"
        rulespec_us = _canonical_rulespec_content_root(repo_root, "us")
        target = rulespec_us / "statutes" / "26" / "63" / "f.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "format: rulespec/v1\n"
            "imports:\n"
            "  - us:statutes/26/151#exemption_individual_eligible\n"
            "rules:\n"
            "  - name: spouse_aged_additional_amount_person_entitlement\n"
            "    kind: derived\n"
            "    entity: Person\n"
            "    dtype: Judgment\n"
            "    period: Year\n"
            "    versions:\n"
            "      - effective_from: '2018-01-01'\n"
            "        formula: |-\n"
            "          spouse_age_before_close_of_taxable_year >= 65\n"
            "          and exemption_individual_eligible\n"
        )

        section_151 = rulespec_us / "statutes" / "26" / "151.yaml"
        section_151.parent.mkdir(parents=True, exist_ok=True)
        section_151.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: taxpayer_exemption_allowed\n"
            "    kind: derived\n"
            "    entity: TaxUnit\n"
            "    dtype: Judgment\n"
            "    period: Year\n"
        )

        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        source_text = workspace_root / "source.txt"
        source_text.write_text(
            "The taxpayer shall be entitled to an additional amount for the "
            "spouse if an additional exemption is allowable under section 151(b)."
        )
        workspace = EvalWorkspace(
            root=workspace_root,
            source_text_file=source_text,
            manifest_file=workspace_root / "manifest.json",
        )
        context_files = [
            EvalContextFile(
                source_path=str(target),
                workspace_path="context/statutes/26/63/f.yaml",
                import_path="us:statutes/26/63/f",
                kind="existing_target",
            )
        ]

        prompt = _build_eval_prompt(
            "26 USC 63(f)",
            "repo-augmented",
            workspace,
            context_files,
            target_file_name="f.yaml",
            target_ref_prefix="us:statutes/26/63/f",
        )

        assert "Copied existing target fails current RuleSpec validation" in prompt
        assert "us:statutes/26/151#exemption_individual_eligible" in prompt
        assert "does not export `exemption_individual_eligible`" in prompt
        assert "defer the affected executable surface" in prompt

    def test_repo_augmented_context_resolves_statute_prefixed_dependencies(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        statute_root = policy_repo_root / "statutes" / "7" / "2014"
        statute_root.mkdir(parents=True)

        selected = statute_root / "e.yaml"
        selected.write_text(
            "format: rulespec/v1\n"
            "imports:\n"
            "  - us:statutes/7/2014/2014#snap_household_has_elderly_or_disabled_member\n"
            "  - us:statutes/7/2014/d#snap_gross_income\n"
            "rules:\n"
            "  - name: snap_net_income\n"
            "    kind: derived\n"
            "    entity: Household\n"
            "    dtype: Money\n"
            "    period: Month\n"
        )

        section_file = statute_root / "2014.yaml"
        section_file.write_text("format: rulespec/v1\nrules: []\n")
        cross_file = statute_root / "d.yaml"
        cross_file.write_text("format: rulespec/v1\nrules: []\n")

        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="7 USC 2017(a)",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="2017(a) ...",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[selected],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item["kind"] for item in manifest["context_files"]
        }

        assert copied_sources[str(selected)] == "implementation_precedent"
        assert copied_sources[str(section_file)] in {
            "implementation_precedent",
            "implementation_dependency",
        }
        assert copied_sources[str(cross_file)] in {
            "implementation_precedent",
            "implementation_dependency",
        }

    def test_prompt_includes_scaffold_dates_from_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = _canonical_rulespec_content_root(repo_root, "us")
        policy_repo_root.mkdir(parents=True, exist_ok=True)
        statute_root = (
            _canonical_rulespec_content_root(repo_root, "us") / "statutes" / "26" / "24"
        )
        statute_root.mkdir(parents=True)
        context_file = statute_root / "b.yaml"
        context_file.write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: The threshold is $1,000 and later $2,000.\n"
            "rules:\n"
            "  - name: threshold\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: USD\n"
            "    versions:\n"
            "      - effective_from: '1998-01-01'\n"
            "        formula: 1000\n"
            "      - effective_from: '2018-01-01'\n"
            "        formula: 2000\n"
        )

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        prompt = _build_eval_prompt(
            "26 USC 24(a)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            "a.yaml",
        )

        assert "`1998-01-01`" in prompt
        assert "`2018-01-01`" in prompt
        assert "Prefer the earliest scaffold date" in prompt


class TestCodexPromptEvalPolicyEngineSkillIsolation:
    def test_prepare_codex_eval_home_omits_user_skills(self, tmp_path, monkeypatch):
        source_home = tmp_path / "real-codex-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}\n")
        (source_home / "skills").mkdir()
        (source_home / "skills" / "encode-policy-v2-skill").mkdir()
        monkeypatch.setenv("CODEX_HOME", str(source_home))

        eval_home = _prepare_codex_eval_home(tmp_path / "eval-home")

        assert (eval_home / "auth.json").exists()
        assert (eval_home / "skills").is_dir()
        assert not (eval_home / "skills" / "encode-policy-v2-skill").exists()

    def test_run_codex_prompt_eval_ignores_user_config(self, tmp_path):
        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/6/3/a",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="nil amount",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        observed_cmds: list[list[str]] = []
        observed_envs: list[dict[str, str]] = []
        observed_skills_dirs: list[bool] = []

        class FakePopen:
            def __init__(self, cmd, stdout, stderr, text, cwd, stdin=None, env=None):
                self.args = cmd
                self.returncode = 0
                observed_cmds.append(cmd)
                observed_envs.append(env or {})
                observed_skills_dirs.append(
                    bool(env) and (Path(env["CODEX_HOME"]) / "skills").is_dir()
                )

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        with (
            patch("axiom_encode.harness.evals.subprocess.Popen", FakePopen),
            patch(
                "axiom_encode.harness.evals._wait_for_codex_process",
                return_value=False,
            ),
        ):
            _run_codex_prompt_eval(runner, workspace, "prompt")

        assert "--ignore-user-config" in observed_cmds[0]
        codex_home = Path(observed_envs[0]["CODEX_HOME"])
        assert codex_home.name.startswith("axiom-codex-home-")
        with pytest.raises(ValueError):
            codex_home.resolve().relative_to(workspace.root.resolve())
        assert observed_skills_dirs == [True]

    def test_run_codex_prompt_eval_errors_on_policyengine_skill_use(self, tmp_path):
        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="us-wa/regulation/388/388-478/388-478-0035",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="income limit",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us-wa"),
            mode="cold",
            extra_context_paths=[],
        )

        event_lines = "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "command": (
                                "sed -n '1,200p' "
                                "/Users/maxghenis/.codex/policyengine-skills/"
                                "skills/workflows/encode-policy-v2-skill/"
                                "SKILL.md"
                            ),
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": "format: rulespec/v1\nrules: []\n",
                        },
                    }
                ),
            ]
        )

        class FakePopen:
            def __init__(self, cmd, stdout, stderr, text, cwd, stdin=None, env=None):
                self.args = cmd
                self.returncode = 0
                stdout.write(event_lines + "\n")
                stdout.flush()

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        with (
            patch("axiom_encode.harness.evals.subprocess.Popen", FakePopen),
            patch(
                "axiom_encode.harness.evals._wait_for_codex_process",
                return_value=False,
            ),
        ):
            response = _run_codex_prompt_eval(runner, workspace, "prompt")

        assert response.error is not None
        assert "PolicyEngine skills" in response.error
        assert response.unexpected_accesses


class TestUnexpectedAccessDetection:
    def test_flags_parent_directory_traversal(self, tmp_path):
        assert _command_looks_out_of_bounds("bash -lc 'find .. -name *.yaml'", tmp_path)

    def test_flags_policyengine_skill_reads(self):
        command = (
            "sed -n '1,200p' "
            "/Users/maxghenis/.codex/policyengine-skills/skills/workflows/"
            "encode-policy-v2-skill/SKILL.md"
        )
        assert _command_uses_policyengine_skill(command)

    def test_allows_workspace_paths(self, tmp_path):
        local = tmp_path / "context" / "b.yaml"
        local.parent.mkdir(parents=True)
        local.write_text("format: rulespec/v1\nrules: []\n")


def test_evaluate_artifact_binds_named_corpus_release_through_metrics(
    tmp_path, monkeypatch
):
    corpus_release = _write_test_corpus_provision(
        tmp_path,
        citation_path="us/statute/1",
        body="trusted source",
    )
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    rules_file = policy_repo / "policies/guidance/target.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: zz/guidance/not-in-checkout
rules: []
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ValidatorPipeline,
        "_run_compile_check",
        lambda _self, _path: ValidationResult("compile", passed=True),
    )
    monkeypatch.setattr(
        ValidatorPipeline,
        "_run_ci",
        lambda _self, _path: ValidationResult("ci", passed=True),
    )

    with patch(
        "axiom_encode.harness.evals._authoritative_corpus_scope",
        wraps=validator_pipeline._authoritative_corpus_scope,
    ) as mock_scope:
        evaluate_artifact(
            rulespec_file=rules_file,
            policy_repo_root=policy_repo,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            source_text="trusted source",
            skip_reviewers=True,
            local_corpus_release=corpus_release,
            source_citation_path="us/statute/1",
        )

    mock_scope.assert_called_once_with(corpus_release)


def test_evaluate_artifact_passes_exact_attached_amendment_sources(
    tmp_path, monkeypatch
):
    source_citation_path = "de/statute/estg/66"
    source_text = "Das Kindergeld beträgt monatlich für jedes Kind 259 Euro."
    corpus_release = _write_test_corpus_provision(
        tmp_path,
        citation_path=source_citation_path,
        body=source_text,
    )
    policy_repo = _canonical_rulespec_content_root(tmp_path, "de")
    rules_file = policy_repo / "statutes/estg/66.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/66
rules: []
""",
        encoding="utf-8",
    )
    amendment_citation_path = (
        "de/statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-1"
    )
    amendment_body = (
        "4. In § 66 Absatz 1 wird die Angabe „250 Euro“ durch die Angabe "
        "„255 Euro“ ersetzt."
    )
    amendment = CorpusAmendmentDocument(
        citation_path=amendment_citation_path,
        title="Steuerfortentwicklungsgesetz – SteFeG",
        expression_date="2024-12-23",
        metadata={},
        body=amendment_body,
    )
    observed_amendment_sources: list[dict[str, str] | None] = []
    original_init = ValidatorPipeline.__init__

    def recording_init(
        pipeline,
        *args,
        amendment_source_texts=None,
        **kwargs,
    ):
        observed_amendment_sources.append(amendment_source_texts)
        original_init(
            pipeline,
            *args,
            amendment_source_texts=amendment_source_texts,
            **kwargs,
        )

    monkeypatch.setattr(ValidatorPipeline, "__init__", recording_init)
    monkeypatch.setattr(
        ValidatorPipeline,
        "_run_compile_check",
        lambda _self, _path: ValidationResult("compile", passed=True),
    )
    monkeypatch.setattr(
        ValidatorPipeline,
        "_run_ci",
        lambda _self, _path: ValidationResult("ci", passed=True),
    )

    evaluate_artifact(
        rulespec_file=rules_file,
        policy_repo_root=policy_repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        source_text=source_text,
        skip_reviewers=True,
        local_corpus_release=corpus_release,
        source_citation_path=source_citation_path,
        require_complete_source_unit=True,
        amendment_documents=(amendment,),
    )

    assert observed_amendment_sources == [{amendment_citation_path: amendment_body}]


class TestSourceEval:
    def test_run_model_eval_passes_validation_options_to_evaluate_artifact(
        self,
        tmp_path,
    ):
        corpus_release = _write_test_corpus_provision(
            tmp_path,
            citation_path="us/statute/7/2017/a",
            body="The source amount is 100.",
        )
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us")

        with (
            patch(
                "axiom_encode.harness.evals._run_prompt_eval",
            ) as mock_prompt_eval,
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
            ) as mock_evaluate_artifact,
        ):
            mock_prompt_eval.return_value.text = (
                "=== FILE: a.yaml ===\n"
                "format: rulespec/v1\n"
                "module:\n"
                "  summary: The source amount is 100.\n"
                "rules:\n"
                "  - name: source_amount\n"
                "    kind: parameter\n"
                "    dtype: Number\n"
                "    versions:\n"
                "      - effective_from: '2025-01-01'\n"
                "        formula: 100\n"
                "=== FILE: a.test.yaml ===\n"
                "- name: base\n"
                "  input: {}\n"
                "  output:\n"
                "    source_amount: 100\n"
            )
            mock_prompt_eval.return_value.duration_ms = 123
            mock_prompt_eval.return_value.tokens = None
            mock_prompt_eval.return_value.estimated_cost_usd = None
            mock_prompt_eval.return_value.actual_cost_usd = None
            mock_prompt_eval.return_value.trace = {}
            mock_prompt_eval.return_value.unexpected_accesses = []
            mock_prompt_eval.return_value.error = None
            mock_evaluate_artifact.return_value = None

            run_model_eval(
                citations=["us/statute/7/2017/a"],
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                policy_path=policy_repo_root,
                runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
                corpus_release=corpus_release,
                mode="cold",
                include_tests=True,
                policyengine_rule_hint="source_amount",
                skip_reviewers=True,
            )

        assert mock_evaluate_artifact.call_args.kwargs["skip_reviewers"] is True
        assert (
            mock_evaluate_artifact.call_args.kwargs["policyengine_rule_hint"]
            == "source_amount"
        )

    def test_run_source_eval_uses_explicit_context_without_statute_lookup(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "us-co")
        context_file = tmp_path / "examples" / "piecewise.yaml"
        context_file.parent.mkdir(parents=True)
        context_file.write_text("format: rulespec/v1\nrules: []\n")
        corpus_release, source_unit = _write_test_source_unit(
            tmp_path,
            "F. Determining Eligibility ... 165",
            citation_path="us-co/regulation/9/3.606.1/F",
        )

        with (
            patch(
                "axiom_encode.harness.evals._run_prompt_eval",
            ) as mock_prompt_eval,
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
            ) as mock_evaluate_artifact,
        ):
            mock_prompt_eval.return_value.text = (
                "=== FILE: 9-CCR-2503-6-3.606.1-F.yaml ===\n"
                "format: rulespec/v1\n"
                "module:\n"
                "  summary: F. Determining Eligibility ...\n"
                "rules:\n"
                "  - name: grant_standard\n"
                "    kind: parameter\n"
                "    entity: TaxUnit\n"
                "    dtype: Money\n"
                "    period: Month\n"
                "    versions:\n"
                "      - effective_from: '2024-07-01'\n"
                "        formula: 165\n"
                "=== FILE: 9-CCR-2503-6-3.606.1-F.test.yaml ===\n"
                "- name: base case\n"
                "  period: 2024-07\n"
                "  input: {}\n"
                "  output:\n"
                "    grant_standard: 165\n"
            )
            mock_prompt_eval.return_value.duration_ms = 123
            mock_prompt_eval.return_value.tokens = None
            mock_prompt_eval.return_value.estimated_cost_usd = None
            mock_prompt_eval.return_value.actual_cost_usd = None
            mock_prompt_eval.return_value.trace = {}
            mock_prompt_eval.return_value.unexpected_accesses = []
            mock_prompt_eval.return_value.error = None

            mock_evaluate_artifact.return_value = None

            results = run_source_eval(
                source_unit=source_unit,
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                policy_path=policy_repo_root,
                local_corpus_release=corpus_release,
                runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
                mode="repo-augmented",
                extra_context_paths=[context_file],
            )

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert Path(result.output_file).exists()
        assert Path(result.output_file).with_suffix(".test.yaml").exists()
        assert result.retrieved_files == [str(context_file)]

        prompt = mock_prompt_eval.call_args.args[2]
        assert ".test.yaml" in prompt
        assert "=== FILE:" in prompt
        assert mock_evaluate_artifact.call_args.kwargs["policy_repo_root"] == (
            policy_repo_root
        )
        assert (
            mock_evaluate_artifact.call_args.kwargs["source_citation_path"]
            == "us-co/regulation/9/3.606.1/F"
        )

    def test_run_source_eval_passes_oracle_settings_to_evaluate_artifact(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "uk")
        corpus_release, source_unit = _write_test_source_unit(
            tmp_path,
            "26.05",
            citation_path="uk/regulation/uksi/2006/965/regulation/2",
        )

        with (
            patch.object(
                PolicyEngineRuntime,
                "assert_matches_rulespec_root",
                return_value=None,
            ),
            patch.object(
                PolicyEngineRuntime,
                "assert_unchanged",
                return_value=None,
            ),
            patch(
                "axiom_encode.harness.evals._run_prompt_eval",
            ) as mock_prompt_eval,
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
            ) as mock_evaluate_artifact,
        ):
            mock_prompt_eval.return_value.text = (
                "=== FILE: uksi-2006-965-regulation-2.yaml ===\n"
                "format: rulespec/v1\n"
                "module:\n"
                "  summary: https://www.legislation.gov.uk/uksi/2006/965/regulation/2 states 26.05.\n"
                "rules:\n"
                "  - name: child_benefit_enhanced_rate\n"
                "    kind: parameter\n"
                "    dtype: Money\n"
                "    unit: GBP\n"
                "    versions:\n"
                "      - effective_from: '2025-04-07'\n"
                "        formula: 26.05\n"
                "=== FILE: uksi-2006-965-regulation-2.test.yaml ===\n"
                "- name: base\n"
                "  input: {}\n"
                "  output:\n"
                "    child_benefit_enhanced_rate: 26.05\n"
            )
            mock_prompt_eval.return_value.duration_ms = 123
            mock_prompt_eval.return_value.tokens = None
            mock_prompt_eval.return_value.estimated_cost_usd = None
            mock_prompt_eval.return_value.actual_cost_usd = None
            mock_prompt_eval.return_value.trace = {}
            mock_prompt_eval.return_value.unexpected_accesses = []
            mock_prompt_eval.return_value.error = None
            mock_evaluate_artifact.return_value = None

            run_source_eval(
                source_unit=source_unit,
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                policy_path=policy_repo_root,
                local_corpus_release=corpus_release,
                runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
                mode="cold",
                oracle="policyengine",
                policyengine_runtime=_test_policyengine_runtime("uk"),
                skip_reviewers=True,
            )

        assert mock_evaluate_artifact.call_args.kwargs["oracle"] == "policyengine"
        assert (
            mock_evaluate_artifact.call_args.kwargs["policyengine_runtime"].country
            == "uk"
        )
        assert mock_evaluate_artifact.call_args.kwargs["skip_reviewers"] is True

    def test_run_source_eval_passes_policyengine_rule_hint_to_evaluate_artifact(
        self, tmp_path
    ):
        policy_repo_root = _canonical_rulespec_content_root(tmp_path, "uk")
        corpus_release, source_unit = _write_test_source_unit(
            tmp_path,
            "317.82",
            citation_path="uk/regulation/uksi/2013/376/regulation/36/3",
        )

        with (
            patch.object(
                PolicyEngineRuntime,
                "assert_matches_rulespec_root",
                return_value=None,
            ),
            patch.object(
                PolicyEngineRuntime,
                "assert_unchanged",
                return_value=None,
            ),
            patch(
                "axiom_encode.harness.evals._run_prompt_eval",
            ) as mock_prompt_eval,
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
            ) as mock_evaluate_artifact,
        ):
            mock_prompt_eval.return_value.text = (
                "=== FILE: uksi-2013-376-regulation-36-3-single-under-25.yaml ===\n"
                "format: rulespec/v1\n"
                "module:\n"
                "  summary: The amount is 317.82.\n"
                "rules:\n"
                "  - name: source_row_amount\n"
                "    kind: parameter\n"
                "    dtype: Money\n"
                "    unit: GBP\n"
                "    versions:\n"
                "      - effective_from: '2025-04-07'\n"
                "        formula: 317.82\n"
                "=== FILE: uksi-2013-376-regulation-36-3-single-under-25.test.yaml ===\n"
                "- name: base\n"
                "  input: {}\n"
                "  output:\n"
                "    source_row_amount: 317.82\n"
            )
            mock_prompt_eval.return_value.duration_ms = 123
            mock_prompt_eval.return_value.tokens = None
            mock_prompt_eval.return_value.estimated_cost_usd = None
            mock_prompt_eval.return_value.actual_cost_usd = None
            mock_prompt_eval.return_value.trace = {}
            mock_prompt_eval.return_value.unexpected_accesses = []
            mock_prompt_eval.return_value.error = None
            mock_evaluate_artifact.return_value = None

            run_source_eval(
                source_unit=source_unit,
                runner_specs=["openai:gpt-5.4"],
                output_root=tmp_path / "out",
                policy_path=policy_repo_root,
                local_corpus_release=corpus_release,
                runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
                mode="cold",
                oracle="policyengine",
                policyengine_runtime=_test_policyengine_runtime("uk"),
                policyengine_rule_hint="uc_standard_allowance_single_claimant_aged_under_25",
            )

        assert (
            mock_evaluate_artifact.call_args.kwargs["policyengine_rule_hint"]
            == "uc_standard_allowance_single_claimant_aged_under_25"
        )

    def test_build_eval_prompt_includes_policyengine_rule_hint(self, tmp_path):
        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="uksi/2013/376/regulation/36/3",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="317.82",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )
        context_source = tmp_path / "adult-group.yaml"
        context_source.write_text(
            """format: rulespec/v1
rules:
  - name: adult_group_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2024-01-01'
        formula: age >= 19
""",
            encoding="utf-8",
        )
        workspace_context = workspace.root / "context" / "adult-group.yaml"
        workspace_context.parent.mkdir(parents=True)
        workspace_context.write_text(context_source.read_text(), encoding="utf-8")
        context_file = EvalContextFile(
            source_path=str(context_source),
            workspace_path="context/adult-group.yaml",
            import_path="us:regulations/42-cfr/435/119",
            kind="allowed_context",
        )

        prompt = _build_eval_prompt(
            "uksi/2013/376/regulation/36/3",
            "cold",
            workspace,
            [context_file],
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
            policyengine_rule_hint="uc_standard_allowance_single_claimant_aged_under_25",
            require_complete_source_unit=True,
        )

        assert "uc_standard_allowance_single_claimant_aged_under_25" in prompt
        assert (
            "Treat the hinted policy surface as a required oracle-facing surface"
            in prompt
        )
        assert "Do not" in prompt
        assert "module.deferred_outputs[]" in prompt
        assert "import that" in prompt
        assert "concrete output instead of leaving the broad phrase" in prompt
        assert "person_is_in_*_category" in prompt
        assert "Keep `.test.yaml` inputs oracle-comparable" in prompt
        assert "prefer a contemporary monthly `.test.yaml`" in prompt
        assert "period like `2022-01` or `2024-01`" in prompt
        assert (
            "canonical RuleSpec output whose local name is `uc_standard_allowance_single_claimant_aged_under_25`"
            in prompt
        )
        assert (
            "prefer the oracle's direct component facts over inverted household proxy inputs"
            in prompt
        )
        assert "oracle_inputs.policyengine" in prompt
        assert "equivalent" in prompt
        assert "PolicyEngine-native scenario inputs" in prompt
        assert "assert that canonical copied output" in prompt
        assert "key the test by that id rather than the friendly local name" in prompt
        assert "Key inputs by their resolving legal RuleSpec target too" in prompt
        assert "For an aggregate/composite hinted output" in prompt
        assert "first enumerate the executable" in prompt
        assert "Executable Judgment exports visible in copied context" in prompt
        assert "us:regulations/42-cfr/435/119#adult_group_eligible" in prompt
        assert "person_covered_by_*category" in prompt
        assert "Do not let the oracle-facing hinted" in prompt
        assert "Avoid pre-2015 periods only for cases" in prompt
        assert "intended to supply PolicyEngine oracle evidence" in prompt
        assert "Never replace or omit a" in prompt
        assert "source-required historical branch" in prompt
        assert (
            "historical branch's runtime evidence must use that branch's legally"
            in prompt
        )
        assert "not unambiguously bound by a canonical\n  structural source path" in (
            prompt
        )
        assert "assert every reached local derived dependency's expected" in prompt
        assert "child-branch `source:` citation does not cover a distinct parent" in (
            prompt
        )
        assert "Do not invent a dummy output for a chapeau" in prompt
        assert "Every emitted derived `dtype: Judgment` output" in prompt
        assert "source-faithful\n  Judgment whose versions are all constant false" in (
            prompt
        )
        assert "testing only the negative state is incomplete" in prompt
        assert "mandatory when multiple\n  computations share" in prompt
        assert "citation-only proof atom" in prompt
        assert "keep that output executable from the earliest" in prompt
        assert "parameter/helper guards in the\n  single derived formula" in prompt
        assert "omit oracle inputs or expectations" in prompt

    def test_policyengine_hint_upstream_composition_flags_broad_placeholders(self):
        content = """
format: rulespec/v1
module:
  deferred_outputs:
    - output: us:statutes/42/1396a/a/10#other_surface
      reason: not relevant
rules:
  - name: is_medicaid_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 42 USC 1396a(a)(10)
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          adult_group_eligible
          or person_covered_by_other_mandatory_subparagraph_A_i_category
          or person_is_described_in_previous_mandatory_subclause
          or income_as_determined_under_subsection_e_14 <= income_limit
"""

        issues = _policyengine_hint_upstream_composition_issues(
            content,
            "is_medicaid_eligible",
        )

        assert len(issues) == 1
        assert "broad upstream placeholder" in issues[0]
        assert (
            "person_covered_by_other_mandatory_subparagraph_A_i_category" in issues[0]
        )
        assert "person_is_described_in_previous_mandatory_subclause" in issues[0]
        assert "income_as_determined_under_subsection_e_14" in issues[0]

    def test_policyengine_hint_upstream_composition_flags_transitive_placeholders(self):
        content = """
format: rulespec/v1
rules:
  - name: adult_expansion_mandatory_group_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 42 USC 1396a(a)(10)(A)(i)(VIII)
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          not person_is_described_in_previous_mandatory_subclause
          and income_determined_for_adult_expansion <= income_limit
          and adult_expansion_subject_to_subsections_k_and_xx_satisfied
  - name: is_medicaid_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 42 USC 1396a(a)(10)
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          person_receives_aid_or_assistance_under_listed_state_plan
          or person_meets_ssi_related_mandatory_group
          or adult_expansion_mandatory_group_eligible
"""

        issues = _policyengine_hint_upstream_composition_issues(
            content,
            "is_medicaid_eligible",
        )

        assert len(issues) == 1
        assert "person_receives_aid_or_assistance_under_listed_state_plan" in issues[0]
        assert "person_meets_ssi_related_mandatory_group" in issues[0]
        assert "person_is_described_in_previous_mandatory_subclause" in issues[0]
        assert "income_determined_for_adult_expansion" in issues[0]
        assert "adult_expansion_subject_to_subsections_k_and_xx_satisfied" in issues[0]

    def test_policyengine_hint_upstream_composition_flags_deferred_hint(self):
        content = """
format: rulespec/v1
module:
  deferred_outputs:
    - output: us:statutes/42/1396a/a/10#is_medicaid_eligible
      reason: broad source
rules: []
"""

        issues = _policyengine_hint_upstream_composition_issues(
            content,
            "is_medicaid_eligible",
        )

        assert len(issues) == 1
        assert "is deferred" in issues[0]

    def test_build_eval_prompt_includes_sets_source_metadata_guidance(self, tmp_path):
        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="snap_standard_utility_allowance_tn",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="The SUA is $451, effective October 1, 2025.",
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "us-tn"),
            mode="cold",
            source_metadata_payload={
                "relations": [
                    {
                        "relation": "sets",
                        "target": "us:regulation/7-cfr/273/9/d/6/iii#snap_standard_utility_allowance",
                        "jurisdiction": "TN",
                    }
                ]
            },
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "snap_standard_utility_allowance_tn",
            "cold",
            workspace,
            [],
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
            policyengine_rule_hint="snap_standard_utility_allowance",
        )

        assert "./source-metadata.json" in prompt
        assert "\n  - relation: sets" not in prompt
        assert '"relation": "sets"' in prompt
        assert "kind: source_relation" in prompt
        assert "source_relation.type" in prompt
        assert "record that legal/provenance edge as a separate" in prompt
        assert "Preserve existing or copied `kind: source_relation` records" in prompt
        assert "source_relation.basis.delegation" in prompt
        assert "mirror the imported file's companion test pattern" in prompt
        assert "Never turn an imported derived rule into a fabricated" in prompt
        assert (
            "Every local executable `kind: derived` or `kind: derived_relation` rule"
            in prompt
        )
        assert "source-period snapshot case" in prompt
        assert "local parameter output directly" in prompt
        assert "Use `holds` and `not_holds` for actual `dtype: Judgment`" in prompt
        assert "Use YAML booleans `true` and `false` for local factual" in prompt
        assert (
            "us:regulation/7-cfr/273/9/d/6/iii#snap_standard_utility_allowance"
            in prompt
        )
        assert "...#*_applies` or `...#*_uses_*" in prompt
        assert (
            "do not add a top-level `imports:` entry to the absolute canonical target path"
            in prompt
        )
        assert "`*_is_in_state` or `*_is_in_jurisdiction`" in prompt
        assert (
            "use only positive/continuity cases rather than a fabricated out-of-jurisdiction false case"
            in prompt
        )
        assert (
            "encode the canonical boolean slot as a direct dated constant `true` or `false`"
            in prompt
        )
        assert (
            "omit an inapplicable false test unless `./source.txt` itself states a narrower in-jurisdiction condition"
            in prompt
        )

    def test_build_eval_prompt_single_amount_slice_disallows_speculative_future_tests(
        self, tmp_path
    ):
        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="uksi/2002/2005/schedule/2",
            runner=runner,
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-04-06.\n\n"
                "Structured table:\n"
                "Relevant element | Maximum annual rate\n"
                "Severe disability element | £1734\n"
            ),
            axiom_rules_path=_canonical_rulespec_content_root(tmp_path, "uk"),
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/2005/schedule/2",
            "cold",
            workspace,
            [],
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "For a single fixed-amount source slice, a base case is sufficient."
            in prompt
        )
        assert (
            "For a one-row fixed-amount slice with `period: Year`, a base case is sufficient; do not synthesize an `effective_date_boundary` test."
            in prompt
        )
        assert (
            "Add a later same-amount case only when `./source.txt` explicitly says the amount remains unchanged through that later date."
            in prompt
        )
        assert "Do not add speculative future-period tests" in prompt

    def test_single_amount_slice_detection_excludes_conditional_money_leaf(self):
        assert (
            _is_single_amount_table_slice(
                "£20 is disregarded if the claimant is in receipt of Scottish adult disability living allowance."
            )
            is False
        )

    def test_normalize_nonannual_test_period_value_converts_iso_week_to_effective_date(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value("2025-W13", date(2025, 3, 21))
            == "2025-03-21"
        )

    def test_normalize_nonannual_test_period_value_bumps_explicit_day_before_effective_date(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value("2026-04-01", date(2026, 4, 2))
            == "2026-04-02"
        )

    def test_normalize_nonannual_test_period_value_bumps_yaml_date_before_effective_date(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value(date(2026, 4, 1), date(2026, 4, 2))
            == "2026-04-02"
        )

    def test_normalize_nonannual_test_period_value_uses_month_period_for_monthly_rules(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value(
                "2025-10-01",
                date(2025, 10, 1),
                granularity="Month",
            )
            == "2025-10"
        )

    def test_normalize_nonannual_test_period_value_preserves_prior_month_for_monthly_rules(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value(
                "2025-09",
                date(2025, 10, 1),
                granularity="Month",
            )
            == "2025-09"
        )

    def test_normalize_nonannual_test_period_value_preserves_prior_day_month_for_monthly_rules(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value(
                "2025-09-30",
                date(2025, 10, 1),
                granularity="Month",
            )
            == "2025-09"
        )

    def test_allows_relative_workspace_reads(self, tmp_path):
        (tmp_path / "source.txt").write_text("text\n")
        command = "bash -lc 'cat ./source.txt && sed -n \"1,40p\" context/statutes/26/24/b.yaml'"
        assert not _command_looks_out_of_bounds(command, tmp_path)


def _fake_eval_result(
    runner: str,
    citation: str,
    *,
    compile_pass: bool = True,
    ci_pass: bool = True,
    generalist_review_pass: bool | None = True,
    generalist_review_score: float | None = 8.0,
    policyengine_pass: bool | None = None,
    policyengine_score: float | None = None,
    estimated_cost_usd: float | None = 0.25,
    ungrounded_numeric_count: int = 0,
) -> EvalResult:
    return EvalResult(
        citation=citation,
        runner=runner,
        backend="openai",
        model="gpt-5.4",
        mode="cold",
        output_file=f"/tmp/{citation}.yaml",
        trace_file=f"/tmp/{citation}.json",
        context_manifest_file=f"/tmp/{citation}.manifest.json",
        generated_output_sha256="a" * 64,
        trace_sha256="b" * 64,
        context_manifest_sha256="c" * 64,
        duration_ms=1000,
        success=True,
        error=None,
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        reasoning_output_tokens=0,
        estimated_cost_usd=estimated_cost_usd,
        actual_cost_usd=None,
        retrieved_files=[],
        unexpected_accesses=[],
        metrics=EvalArtifactMetrics(
            compile_pass=compile_pass,
            compile_issues=[],
            ci_pass=ci_pass,
            ci_issues=[],
            embedded_source_present=True,
            grounded_numeric_count=1 if ungrounded_numeric_count == 0 else 0,
            ungrounded_numeric_count=ungrounded_numeric_count,
            grounding=[
                GroundingMetric(
                    line=1,
                    raw="26.05",
                    value=26.05,
                    grounded=ungrounded_numeric_count == 0,
                )
            ],
            generalist_review_pass=generalist_review_pass,
            generalist_review_score=generalist_review_score,
            generalist_review_issues=[],
            policyengine_pass=policyengine_pass,
            policyengine_score=policyengine_score,
            policyengine_issues=[],
            policyengine_runtime_identity=(
                _TEST_POLICYENGINE_RUNTIME_IDENTITY
                if policyengine_pass is not None or policyengine_score is not None
                else None
            ),
            policyengine_runtime_identity_sha256=(
                _TEST_POLICYENGINE_RUNTIME_IDENTITY_SHA256
                if policyengine_pass is not None or policyengine_score is not None
                else None
            ),
        ),
    )


@pytest.mark.parametrize(
    ("issue", "expected"),
    [
        (
            "Derived rule missing companion output coverage: `us:x#amount` is not asserted by the companion `.test.yaml` file.",
            "companion_coverage",
        ),
        (
            "Proof atom missing path: rule `amount` proof atom 0 must declare `path`.",
            "proof_atoms",
        ),
        (
            "Proof import hash mismatch: rule `amount` proof atom 0 declares sha256 `abc` but resolved import has sha256 `def`.",
            "proof_atoms",
        ),
        (
            "Ungrounded generated numeric literal: 0.15 does not appear as a substantive numeric value in the source text.",
            "ungrounded_literal",
        ),
        (
            "Test case `basic` output `amount` expected decimal 10, got decimal 9.",
            "fixture_execution",
        ),
        (
            "PolicyEngine produced zero comparable oracle evidence",
            "oracle_coverage",
        ),
        ("PE=10.00, RuleSpec expects=9.00", "oracle_coverage"),
        ("No PolicyEngine-comparable tests found", "oracle_coverage"),
        (
            "Import `us:statutes/26/1` does not resolve to a RuleSpec file in the clean policy repository.",
            "import_resolution",
        ),
        ("rules.test.yaml YAML parse failed: mapping values are not allowed", "schema"),
        (
            "Canonical concept import missing: `household_income` uniquely resolves nearby.",
            "concept_registry",
        ),
        (
            "Numeric source required: RuleSpec defines policy numeric literals but does not provide source text.",
            "embedded_source",
        ),
        ("Axiom rules engine compile failed: unknown rule kind", "compile"),
        (
            "Axiom rules engine compile did not return an artifact payload.",
            "compile",
        ),
    ],
)
def test_classify_validation_issue_known_families(issue, expected):
    from axiom_encode.harness.evals import classify_validation_issue

    assert classify_validation_issue(issue) == expected


def test_classify_validation_issue_fallback_is_deterministic_and_bounded():
    from axiom_encode.harness.evals import classify_validation_issue

    issue = 'Unexpected 123 item at /tmp/build/thing.yaml named "private value" remains wrong'
    assert classify_validation_issue(issue) == "unexpected_item_at_named_remains_wrong"
    assert classify_validation_issue(issue) == classify_validation_issue(issue)
    assert len(classify_validation_issue("word " * 100)) <= 60
    assert classify_validation_issue("123 '/tmp/x'") == "unclassified"


def test_classify_validation_issue_fallback_collapses_filenames_and_lines():
    from axiom_encode.harness.evals import classify_validation_issue

    first = classify_validation_issue(
        "Unexpected validator failure at alpha.yaml line 12"
    )
    second = classify_validation_issue(
        "Unexpected validator failure at beta.yaml line 98"
    )

    assert first == second == "unexpected_validator_failure_at_line"


def test_classify_validation_issue_handles_pathological_text():
    from axiom_encode.harness.evals import classify_validation_issue

    result = classify_validation_issue(("\udcff" * 100_000) + " 123")
    assert result == "unclassified"


def test_summarize_validation_failures_empty():
    from axiom_encode.harness.evals import summarize_validation_failures

    assert summarize_validation_failures([]) == {}


def test_summarize_validation_failures_deduplicates_caps_and_counts_before_cap():
    from axiom_encode.harness.evals import summarize_validation_failures

    issues = [f"Novel validator issue {index}" for index in range(41)]
    summary = summarize_validation_failures(
        [("ci", [issues[0], issues[0], *issues[1:]]), ("compile", ["x" * 300])]
    )

    assert len(summary["validation_failures"]) == 40
    assert summary["validation_failures_truncated"] == 2
    assert summary["validation_failures"][0]["detail"] == issues[0]
    assert summary["validation_failures"][1]["detail"] == issues[1]
    assert sum(summary["validation_failure_counts"].values()) == 42
    assert summary["validation_failure_counts"][f"compile:{'x' * 60}"] == 1
    assert all(len(item["detail"]) <= 240 for item in summary["validation_failures"])


def _revalidation_metrics(**overrides) -> EvalArtifactMetrics:
    base = dict(
        compile_pass=True,
        compile_issues=[],
        ci_pass=True,
        ci_issues=[],
        embedded_source_present=True,
        grounded_numeric_count=3,
        ungrounded_numeric_count=0,
        grounding=[GroundingMetric(line=4, raw="20", value=20.0, grounded=True)],
        source_numeric_occurrence_count=3,
        covered_source_numeric_occurrence_count=3,
        missing_source_numeric_occurrence_count=0,
        numeric_occurrence_issues=[],
        generalist_review_pass=True,
        generalist_review_score=8.5,
        generalist_review_issues=["style nit"],
        generalist_review_prompt_sha256="a" * 64,
    )
    base.update(overrides)
    return EvalArtifactMetrics(**base)


def _revalidation_result(
    metrics: EvalArtifactMetrics | None,
    *,
    output_file: str = "artifact.yaml",
    success: bool = True,
    error: str | None = None,
) -> evals_module.EvalResult:
    return evals_module.EvalResult(
        citation="uk/statute/ukpga/1994/23/2",
        runner="fable",
        backend="claude",
        model="claude-fable-5",
        mode="cold",
        output_file=output_file,
        trace_file="trace.json",
        context_manifest_file="context.json",
        generated_output_sha256="b" * 64,
        trace_sha256="c" * 64,
        context_manifest_sha256="d" * 64,
        duration_ms=1000,
        success=success,
        error=error,
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        reasoning_output_tokens=0,
        estimated_cost_usd=None,
        actual_cost_usd=1.0,
        retrieved_files=[],
        unexpected_accesses=[],
        metrics=metrics,
    )


def _run_case_revalidation(
    persisted: EvalArtifactMetrics | None,
    fresh: EvalArtifactMetrics | None,
    *,
    require_complete_source_unit: bool = False,
    amendment_documents: tuple[CorpusAmendmentDocument, ...] = (),
    visible_amendment_citations: tuple[str, ...] = (),
    historical_amendment_manifest: bool = False,
    **result_overrides,
):
    case = evals_module.EvalSuiteCase(
        kind="source",
        name="vat_standard_rate",
        mode="cold",
        corpus_citation_path="uk/statute/ukpga/1994/23/2",
        require_complete_source_unit=require_complete_source_unit,
    )
    source_unit = SimpleNamespace(
        body="The rate of VAT is 20 percent.",
        amendment_documents=amendment_documents,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        context_manifest = Path(tmpdir) / "context.json"
        context_items = []
        documents_by_citation = {
            document.citation_path: document for document in amendment_documents
        }
        for index, citation in enumerate(visible_amendment_citations, start=1):
            workspace_path = Path("context") / f"amendment-act-{index}.txt"
            materialized = context_manifest.parent / workspace_path
            materialized.parent.mkdir(parents=True, exist_ok=True)
            document = documents_by_citation[citation]
            materialized.write_text(
                evals_module._render_amendment_document(
                    document,
                    body=document.body,
                )
                + "\n"
            )
            context_items.append(
                (
                    {
                        "kind": "corpus_amendment_act",
                        "source_path": citation,
                        "workspace_path": str(workspace_path),
                        "import_path": citation,
                    }
                    if historical_amendment_manifest
                    else {
                        "kind": "corpus_amendment_act",
                        "citation_path": citation,
                        "workspace_path": str(workspace_path),
                    }
                )
            )
        context_manifest.write_text(
            json.dumps(
                {
                    "context_files": context_items,
                    "dropped_amendment_documents": [
                        {
                            "citation_path": document.citation_path,
                            "reason": "aggregate_context_limit",
                        }
                        for document in amendment_documents
                        if document.citation_path not in visible_amendment_citations
                    ],
                }
            )
        )
        result = _revalidation_result(persisted, **result_overrides)
        result.context_manifest_file = str(context_manifest)
        result.context_manifest_sha256 = hashlib.sha256(
            context_manifest.read_bytes()
        ).hexdigest()
        with (
            patch.object(
                evals_module, "resolve_corpus_source_unit", return_value=source_unit
            ),
            patch.object(
                evals_module, "_source_metadata_with_attestation", return_value={}
            ),
            patch.object(
                evals_module,
                "_source_metadata_citation_path",
                return_value="uk/statute/ukpga/1994/23/2",
            ),
            patch.object(
                evals_module, "evaluate_artifact", return_value=fresh
            ) as evaluate_mock,
        ):
            evals_module._revalidate_persisted_eval_suite_case_results(
                case,
                [result],
                policy_repo_root=Path("/nonexistent/policy"),
                axiom_rules_path=Path("/nonexistent/engine"),
                corpus_release=SimpleNamespace(),
                policyengine_runtime=None,
                rulespec_dependency_roots=(),
            )
    return evaluate_mock


def test_persisted_revalidation_ignores_reviewer_outcomes():
    """Advisory reviewer output is nondeterministic; it must not gate resume."""
    persisted = _revalidation_metrics()
    fresh = _revalidation_metrics(
        generalist_review_pass=None,
        generalist_review_score=None,
        generalist_review_issues=[],
        generalist_review_prompt_sha256=None,
    )
    evaluate_mock = _run_case_revalidation(persisted, fresh)
    assert evaluate_mock.call_args.kwargs["skip_reviewers"] is True


def test_persisted_revalidation_keeps_complete_source_unit_mode():
    metrics = _revalidation_metrics()

    evaluate_mock = _run_case_revalidation(
        metrics,
        metrics,
        require_complete_source_unit=True,
    )

    assert evaluate_mock.call_args.kwargs["require_complete_source_unit"] is True


def test_persisted_revalidation_uses_only_manifest_visible_amendments():
    metrics = _revalidation_metrics()
    amendments = tuple(
        CorpusAmendmentDocument(
            citation_path=f"dk/statute/amendment-{year}",
            title=f"Amendment {year}",
            expression_date=f"{year}-01-01",
            metadata={},
            body="Body.",
            match_tier="structured",
        )
        for year in (2027, 2026, 2025)
    )

    evaluate_mock = _run_case_revalidation(
        metrics,
        metrics,
        amendment_documents=amendments,
        visible_amendment_citations=tuple(
            document.citation_path for document in amendments[:2]
        ),
    )

    assert evaluate_mock.call_args.kwargs["amendment_documents"] == amendments[:2]


def test_persisted_revalidation_accepts_historical_amendment_manifest_schema():
    metrics = _revalidation_metrics()
    amendment = CorpusAmendmentDocument(
        citation_path="dk/statute/amendment-1",
        title="Amendment 1",
        expression_date="2026-01-01",
        metadata={},
        body="Body.",
        match_tier="structured",
    )

    evaluate_mock = _run_case_revalidation(
        metrics,
        metrics,
        amendment_documents=(amendment,),
        visible_amendment_citations=(amendment.citation_path,),
        historical_amendment_manifest=True,
    )

    assert evaluate_mock.call_args.kwargs["amendment_documents"] == (amendment,)


@pytest.mark.parametrize(
    "persisted_overrides,fresh_overrides",
    [
        ({}, {"ci_pass": False, "ci_issues": ["fixture failed"]}),
        (
            {"policyengine_pass": True, "policyengine_score": 0.97},
            {"policyengine_pass": False, "policyengine_score": 0.41},
        ),
    ],
    ids=["ci", "policyengine-oracle"],
)
def test_persisted_revalidation_still_rejects_deterministic_drift(
    persisted_overrides, fresh_overrides
):
    """Deterministic validator fields — oracle included — stay recompute-bound."""
    reviewer_blank = dict(
        generalist_review_pass=None,
        generalist_review_score=None,
        generalist_review_issues=[],
        generalist_review_prompt_sha256=None,
    )
    persisted = _revalidation_metrics(**persisted_overrides)
    fresh = _revalidation_metrics(**reviewer_blank, **fresh_overrides)
    with pytest.raises(ValueError, match="do not match fresh validation"):
        _run_case_revalidation(persisted, fresh)


@pytest.mark.parametrize("field_name", ["compile_issues", "ci_issues"])
def test_persisted_revalidation_still_rejects_issue_text_drift(field_name):
    """Location normalization must not remove issue lists from tamper checks."""
    persisted = _revalidation_metrics(
        **{
            field_name: [
                "failed to load "
                "<rulespec-validation-root>/rulespec-uk/uk/statutes/1/a.yaml"
            ]
        }
    )
    fresh = _revalidation_metrics(
        generalist_review_pass=None,
        generalist_review_score=None,
        generalist_review_issues=[],
        generalist_review_prompt_sha256=None,
        **{
            field_name: [
                "failed to load "
                "<rulespec-validation-root>/rulespec-uk/uk/statutes/1/b.yaml"
            ]
        },
    )

    with pytest.raises(ValueError, match="do not match fresh validation"):
        _run_case_revalidation(persisted, fresh)


def test_persisted_revalidation_admits_unvalidated_rows_without_reviewer_calls():
    """A row with no bound artifact (no output, metrics=None) resumes cleanly."""
    evaluate_mock = _run_case_revalidation(
        None, None, output_file="", success=False, error="encode failed"
    )
    evaluate_mock.assert_not_called()


def test_retry_feedback_appends_after_static_prompt_prefix(tmp_path):
    """#1491: per-attempt material must extend, not split, the cacheable prefix.

    The explicit stable-prefix segment must be byte-identical between the first
    and retry attempts. Feedback lands after the static instruction tail, next
    to the rejected candidate it describes, while each complete model-visible
    prompt remains unchanged by request segmentation.
    """
    _release, source_unit = _write_test_source_unit(
        tmp_path,
        "The benefit is ten percent of the qualifying amount.",
        citation_path="dk/statute/benefit/section-1",
    )

    _, first_attempt = _workspace_prompt_for_source_unit(tmp_path, source_unit)
    candidate = ValidationRetryCandidate(rulespec="format: rulespec/v1\nrules: []\n")
    _, retry_attempt = _workspace_prompt_for_source_unit(
        tmp_path,
        source_unit,
        validation_retry_candidate=candidate,
        validation_retry_feedback=(
            "ci: [complete-source-unit:structure] branch missing",
        ),
    )

    first_prefix, first_suffix = evals_module._openai_prompt_cache_parts(first_attempt)
    retry_prefix, retry_suffix = evals_module._openai_prompt_cache_parts(retry_attempt)

    # Cache segmentation never changes the model-visible prompt text.
    assert first_prefix + first_suffix == first_attempt
    assert retry_prefix + retry_suffix == retry_attempt
    # Initial and retry calls write/read the exact same reusable prefix.
    assert retry_prefix == first_prefix
    assert len(first_prefix) > 100_000
    assert "RuleSpec requirements:" in first_prefix
    assert "Deterministic validation feedback" not in retry_prefix
    assert retry_suffix.startswith("\nDeterministic validation feedback")
    assert retry_suffix.index("Deterministic validation feedback") < retry_suffix.index(
        "BEGIN UNTRUSTED REJECTED CANDIDATE"
    )
    assert "Return exactly this two-file bundle" in first_suffix
    assert "Return exactly this two-file bundle" in retry_suffix

    first_input, _, first_explicit = evals_module._openai_prompt_input(
        "gpt-5.6-terra",
        first_attempt,
    )
    retry_input, _, retry_explicit = evals_module._openai_prompt_input(
        "gpt-5.6-terra",
        retry_attempt,
    )
    assert first_explicit is retry_explicit is True

    def rendered_text(request_input):
        return "".join(item["text"] for item in request_input[0]["content"])

    assert rendered_text(first_input) == first_attempt
    assert rendered_text(retry_input) == retry_attempt
    assert first_input[0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    assert retry_input[0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }


def test_openai_prompt_cache_key_is_stable_per_prompt_family():
    prefix = "prompt head " * 500
    key = evals_module._openai_prompt_cache_key("gpt-5.6-terra", prefix)
    assert key == evals_module._openai_prompt_cache_key("gpt-5.6-terra", prefix)
    assert key.startswith("axiom-encode-")
    assert key != evals_module._openai_prompt_cache_key(
        "gpt-5.6-terra", "other citation " * 500
    )
    assert key != evals_module._openai_prompt_cache_key("gpt-5.6-sol", prefix)


def test_older_openai_models_keep_the_original_single_string_prompt():
    prompt = "unchanged model-visible prompt"
    request_input, stable_prefix, explicit = evals_module._openai_prompt_input(
        "gpt-5.4",
        prompt,
    )
    assert request_input == prompt
    assert stable_prefix == prompt
    assert explicit is False


def test_older_openai_models_reuse_stable_prefix_key_across_retries(tmp_path):
    _release, source_unit = _write_test_source_unit(
        tmp_path,
        "The benefit is ten percent of the qualifying amount.",
        citation_path="dk/statute/benefit/section-1",
    )
    _, first_attempt = _workspace_prompt_for_source_unit(tmp_path, source_unit)
    _, retry_attempt = _workspace_prompt_for_source_unit(
        tmp_path,
        source_unit,
        validation_retry_candidate=ValidationRetryCandidate(
            rulespec="format: rulespec/v1\nrules: []\n"
        ),
        validation_retry_feedback=("ci: branch missing",),
    )

    first_input, first_prefix, first_explicit = evals_module._openai_prompt_input(
        "gpt-5.4",
        first_attempt,
    )
    retry_input, retry_prefix, retry_explicit = evals_module._openai_prompt_input(
        "gpt-5.4",
        retry_attempt,
    )

    assert first_input is first_attempt
    assert retry_input is retry_attempt
    assert first_explicit is retry_explicit is False
    assert first_prefix == retry_prefix
    assert evals_module._openai_prompt_cache_key(
        "gpt-5.4", first_prefix
    ) == evals_module._openai_prompt_cache_key("gpt-5.4", retry_prefix)
