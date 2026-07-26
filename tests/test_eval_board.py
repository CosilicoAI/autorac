"""Tests for the N-runner eval board fold and its capability manifest."""

import json
from pathlib import Path

import pytest

from axiom_encode import cli
from axiom_encode.harness import eval_board as eval_board_module
from axiom_encode.harness.eval_board import (
    SUPPORTED_EXECUTION_IDENTITY_SCHEMA,
    SUPPORTED_RESULTS_SCHEMA,
    EvalBoardError,
    eval_board_case_rows,
    eval_board_to_json,
    fold_eval_board,
    load_eval_suite_results,
    normalized_execution_identity,
    render_eval_board_markdown,
    render_eval_board_text,
    result_gate_pass,
)
from axiom_encode.harness.evals import (
    _build_eval_suite_execution_identity,
    _eval_suite_execution_identity_sha256,
    load_eval_suite_manifest,
)
from axiom_encode.harness.evals import (
    _canonical_json_sha256 as evals_canonical_json_sha256,
)
from axiom_encode.harness.policyengine_runtime import (
    POLICYENGINE_RUNTIME_PIN_SCHEMA,
    POLICYENGINE_RUNTIME_SCHEMA,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_MANIFEST = REPO_ROOT / "benchmarks" / "encodebench_uk_v1.yaml"

CASE_IDENTITIES = [
    {
        "index": 1,
        "name": "alpha",
        "kind": "source",
        "corpus_citation_path": "uk/statute/ukpga/2007/3/35",
        "sha256": "aa" * 32,
    },
    {
        "index": 2,
        "name": "beta",
        "kind": "source",
        "corpus_citation_path": "uk/statute/ukpga/1994/23/2",
        "sha256": "bb" * 32,
    },
    {
        "index": 3,
        "name": "gamma",
        "kind": "source",
        "corpus_citation_path": "uk/statute/ukpga/2012/5/8",
        "sha256": "cc" * 32,
    },
]

CORPUS_IDENTITY = {
    "corpus_release": "uk-rulespec-2026-07-14",
    "corpus_release_content_sha256": "dd" * 32,
    "corpus_release_selector_sha256": "dc" * 32,
}


def _policyengine_runtime_identity(*, root="/ci/pe-uk", pe_version="1.9.0"):
    """Mirror the sealed-runtime identity shape, paths included."""
    identity = {
        "schema": POLICYENGINE_RUNTIME_SCHEMA,
        "country": "uk",
        "official_repository_url": "https://github.com/PolicyEngine/policyengine-uk",
        "trusted_git_commit": "9" * 40,
        "official_tree_sha256": "13" * 32,
        "official_tree_file_count": 5000,
        "official_tree_byte_count": 12345678,
        "rulespec_runtime_pin_path": f"{root}/rulespec/.axiom/policyengine.toml",
        "rulespec_runtime_pin_schema": POLICYENGINE_RUNTIME_PIN_SCHEMA,
        "rulespec_runtime_pin_sha256": "14" * 32,
        "repository_root": root,
        "checkout_execution_tree_sha256": "15" * 32,
        "checkout_execution_file_count": 5100,
        "checkout_execution_byte_count": 22345678,
        "venv_root": f"{root}/.venv",
        "venv_execution_tree_sha256": "16" * 32,
        "venv_execution_file_count": 21000,
        "venv_execution_byte_count": 923456789,
        "stdlib_root": f"{root}/.venv/lib/python3.13",
        "site_packages_root": f"{root}/.venv/lib/python3.13/site-packages",
        "pyproject_sha256": "17" * 32,
        "uv_lock_sha256": "18" * 32,
        "locked_versions": {
            "policyengine-core": "3.20.0",
            "policyengine-uk": pe_version,
        },
        "python_version": "3.13.5",
        "python_implementation": "cpython",
        "python_executable": f"{root}/.venv/bin/python",
        "python_prefix": f"{root}/.venv",
        "python_base_prefix": f"{root}/toolcache/python-3.13.5",
        "python_exec_prefix": f"{root}/.venv",
        "python_base_exec_prefix": f"{root}/toolcache/python-3.13.5",
        "initial_sys_path": [
            f"{root}/toolcache/python-3.13.5/lib/python313.zip",
            f"{root}/toolcache/python-3.13.5/lib/python3.13",
            f"{root}/toolcache/python-3.13.5/lib/python3.13/lib-dynload",
        ],
        "effective_sys_path": [
            f"{root}/toolcache/python-3.13.5/lib/python313.zip",
            f"{root}/toolcache/python-3.13.5/lib/python3.13",
            f"{root}/toolcache/python-3.13.5/lib/python3.13/lib-dynload",
            f"{root}/.venv/lib/python3.13/site-packages",
        ],
        "isolated": 1,
        "no_site": 1,
        "packages": {
            "policyengine-uk": {
                "distribution": "policyengine-uk",
                "version": pe_version,
                "module_origin": (
                    f"{root}/.venv/lib/python3.13/site-packages/"
                    "policyengine_uk/__init__.py"
                ),
                "metadata_root": f"{root}/.venv/lib/python3.13/site-packages",
            },
            "policyengine-core": {
                "distribution": "policyengine-core",
                "version": "3.20.0",
                "module_origin": (
                    f"{root}/.venv/lib/python3.13/site-packages/"
                    "policyengine_core/__init__.py"
                ),
                "metadata_root": f"{root}/.venv/lib/python3.13/site-packages",
            },
        },
    }
    return {
        "identity": identity,
        "sha256": evals_canonical_json_sha256(identity),
    }


def _execution_identity(
    *,
    encoder_commit="1" * 40,
    checkout="/ci/axiom-encode",
    policyengine_runtime=None,
    case_timeout_seconds=3600,
    claude_timeout_seconds=1800,
    codex_timeout_seconds=600,
    suite_max_attempts=3,
):
    """A payload execution identity mirroring the current producer shape."""
    return {
        "schema": SUPPORTED_EXECUTION_IDENTITY_SCHEMA,
        "case_timeout_seconds": case_timeout_seconds,
        "runner_timeouts": {
            "claude": {"wall_seconds": claude_timeout_seconds},
            "codex": {
                "short_source": {
                    "wall_seconds": codex_timeout_seconds,
                    "idle_seconds": min(300, codex_timeout_seconds),
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
            "suite_max_attempts": suite_max_attempts,
            "suite_retries_after_timeout": False,
            "openai_request_max_attempts": 6,
            "openai_request_backoff_seconds": [1, 2, 4, 8, 10],
        },
        "axiom_encode": {
            "kind": "git",
            "path": checkout,
            "commit": encoder_commit,
            "origin_repository": "github.com/TheAxiomFoundation/axiom-encode",
            "dirty": False,
            "working_tree_sha256": "ee" * 32,
            "pathspecs": ["src/axiom_encode", "pyproject.toml", "uv.lock"],
            "version": "0.2.1303",
        },
        "axiom_rules_engine": {
            "kind": "git",
            "path": f"{checkout}-engine",
            "commit": "2" * 40,
            "origin_repository": "github.com/TheAxiomFoundation/axiom-rules-engine",
            "dirty": False,
            "working_tree_sha256": "ff" * 32,
        },
        "policyengine_runtime": policyengine_runtime,
        "rulespec_roots": [
            {
                "path": f"{checkout}-rulespec/uk",
                "content_state": "directory",
                "content_sha256": "ab" * 32,
                "file_count": 400,
                "toolchain_root": f"{checkout}-rulespec",
                "checkout_identity": {
                    "kind": "git",
                    "path": f"{checkout}-rulespec",
                    "commit": "3" * 40,
                    "origin_repository": "github.com/TheAxiomFoundation/rulespec-uk",
                    "dirty": False,
                    "working_tree_sha256": "cd" * 32,
                    "pathspecs": [
                        "uk",
                        ".axiom/toolchain.toml",
                        "known-validation-gaps.yaml",
                    ],
                },
                "toolchain_contract_sha256": "ef" * 32,
                "validation_waiver_set_sha256": "12" * 32,
            }
        ],
    }


def _metrics(
    *,
    compile_pass=True,
    ci_pass=True,
    ungrounded=0,
    occurrences=10,
    covered=10,
    review_pass=True,
    review_score=8.5,
    policyengine_pass=None,
    policyengine_score=None,
):
    return {
        "compile_pass": compile_pass,
        "compile_issues": [],
        "ci_pass": ci_pass,
        "ci_issues": [],
        "embedded_source_present": True,
        "grounded_numeric_count": 4,
        "ungrounded_numeric_count": ungrounded,
        "grounding": [],
        "source_numeric_occurrence_count": occurrences,
        "covered_source_numeric_occurrence_count": covered,
        "missing_source_numeric_occurrence_count": occurrences - covered,
        "numeric_occurrence_issues": [],
        "generalist_review_pass": review_pass,
        "generalist_review_score": review_score,
        "generalist_review_issues": [],
        "policyengine_pass": policyengine_pass,
        "policyengine_score": policyengine_score,
    }


def _result(
    runner,
    case,
    *,
    backend="codex",
    model="gpt-5.6-terra",
    success=True,
    error=None,
    duration_ms=60_000,
    cost=None,
    metrics="default",
    eval_case_overrides=None,
    failure_kind=None,
    timed_out=False,
    timeout_stage=None,
    timeout_reason=None,
    timeout_seconds=None,
    timeout_attempts=0,
):
    if metrics == "default":
        metrics = _metrics()
    eval_case = {
        "index": case["index"],
        "name": case["name"],
        "kind": case["kind"],
        "corpus_citation_path": case["corpus_citation_path"],
        "sha256": case["sha256"],
    }
    if eval_case_overrides:
        eval_case.update(eval_case_overrides)
    return {
        "citation": case["name"],
        "runner": runner,
        "backend": backend,
        "model": model,
        "mode": "cold",
        "success": success,
        "error": error,
        "failure_kind": failure_kind,
        "timed_out": timed_out,
        "timeout_stage": timeout_stage,
        "timeout_reason": timeout_reason,
        "timeout_seconds": timeout_seconds,
        "timeout_attempts": timeout_attempts,
        "duration_ms": duration_ms,
        "estimated_cost_usd": cost,
        "metrics": metrics,
        "eval_case": eval_case,
    }


def _payload(
    runners,
    results,
    *,
    suite_name="EncodeBench UK v1",
    case_identities=None,
    corpus=None,
    complete=True,
    execution_identity=None,
    execution_identity_sha256=None,
    results_sha256=None,
    coverage_overrides=None,
    evidence_overrides=None,
    schema=SUPPORTED_RESULTS_SCHEMA,
):
    case_identities = CASE_IDENTITIES if case_identities is None else case_identities
    if execution_identity is None:
        execution_identity = _execution_identity()
    if execution_identity_sha256 is None:
        execution_identity_sha256 = evals_canonical_json_sha256(execution_identity)
    # Bind rows exactly like the producer: every persisted row carries a
    # digest over itself minus the digest field.
    results = [
        (
            {**row, "result_sha256": cli._eval_suite_json_sha256(row)}
            if isinstance(row, dict) and "result_sha256" not in row
            else row
        )
        for row in results
    ]
    if results_sha256 is None:
        results_sha256 = cli._eval_suite_json_sha256(results)
    completed_cases = len(
        {
            result["eval_case"]["index"]
            for result in results
            if isinstance(result, dict) and isinstance(result.get("eval_case"), dict)
        }
    )
    coverage = {
        "expected_case_count": len(case_identities),
        "completed_case_count": completed_cases,
        "expected_runner_count": len(runners),
        "expected_result_count": len(case_identities) * len(runners),
        "actual_result_count": len(results),
        "complete": complete,
        "results_sha256": results_sha256,
    }
    if coverage_overrides:
        coverage.update(coverage_overrides)
    evidence = {
        "schema": cli._EVAL_SUITE_EVIDENCE_SCHEMA,
        "manifest": {
            "name": suite_name,
            "path": "benchmarks/encodebench_uk_v1.yaml",
            "content_sha256": "77" * 32,
            "case_identities": case_identities,
        },
        "corpus": dict(CORPUS_IDENTITY if corpus is None else corpus),
        "effective_runner_identities": [
            {
                "name": name,
                "backend": backend,
                "model": model,
            }
            for name, backend, model in runners
        ],
        "execution_identity": execution_identity,
        "execution_identity_sha256": execution_identity_sha256,
    }
    evidence["sha256"] = cli._eval_suite_json_sha256(evidence)
    if evidence_overrides:
        evidence.update(evidence_overrides)
    return {
        "schema": schema,
        "manifest": {
            "name": suite_name,
            "path": "benchmarks/encodebench_uk_v1.yaml",
            "runners": [
                f"{name}={backend}:{model}" for name, backend, model in runners
            ],
            "effective_runners": [
                f"{name}={backend}:{model}" for name, backend, model in runners
            ],
        },
        "evidence": evidence,
        "coverage": coverage,
        "results": results,
    }


def _write_payload(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_supported_schema_matches_producer():
    assert SUPPORTED_RESULTS_SCHEMA == cli._EVAL_SUITE_RESULTS_SCHEMA
    assert SUPPORTED_RESULTS_SCHEMA == "axiom-encode/eval-suite-results/v6"
    assert (
        SUPPORTED_EXECUTION_IDENTITY_SCHEMA == "axiom-encode/eval-execution-identity/v3"
    )
    assert (
        eval_board_module.SUPPORTED_EVIDENCE_SCHEMA == cli._EVAL_SUITE_EVIDENCE_SCHEMA
    )


def test_canonical_digest_matches_both_producer_functions():
    sample = {
        "zeta": [1, {"nested": "väl"}],
        "alpha": None,
        "count": 3,
    }
    board_digest = eval_board_module._canonical_json_sha256(sample)
    assert board_digest == evals_canonical_json_sha256(sample)
    assert board_digest == cli._eval_suite_json_sha256(sample)


def test_real_producer_identity_matches_consumer_contract():
    """Non-circular lock: build a REAL producer execution identity.

    The repo checkout itself serves as the git identity target, so this
    exercises the producer's actual schema string, digest function, and
    field shapes against the consumer's constants and normalizer.
    """
    identity = _build_eval_suite_execution_identity(REPO_ROOT, ())
    assert identity["schema"] == SUPPORTED_EXECUTION_IDENTITY_SCHEMA
    digest = _eval_suite_execution_identity_sha256(identity)
    assert digest == eval_board_module._canonical_json_sha256(identity)
    normalized = normalized_execution_identity(identity)
    rendered = json.dumps(normalized)
    assert str(REPO_ROOT) not in rendered
    assert '"path"' not in rendered
    # Score-affecting fields survive normalization.
    assert identity["axiom_encode"]["version"] in rendered
    assert identity["case_timeout_seconds"] == normalized["case_timeout_seconds"]
    assert identity["runner_timeouts"] == normalized["runner_timeouts"]
    assert identity["timeout_retry_policy"] == normalized["timeout_retry_policy"]


def test_fold_two_single_runner_payloads(tmp_path):
    terra_results = [
        _result("terra", CASE_IDENTITIES[0], duration_ms=40_000),
        _result(
            "terra",
            CASE_IDENTITIES[1],
            duration_ms=50_000,
            metrics=_metrics(ungrounded=2, covered=8),
        ),
        _result(
            "terra",
            CASE_IDENTITIES[2],
            duration_ms=60_000,
            success=False,
            error="encode timed out",
            metrics=None,
            failure_kind="timeout",
            timed_out=True,
            timeout_stage="encoder",
            timeout_reason="wall",
            timeout_seconds=600,
            timeout_attempts=1,
        ),
    ]
    sol_results = [
        _result(
            "sol",
            case,
            model="gpt-5.6-sol",
            duration_ms=90_000,
            cost=0.02,
        )
        for case in CASE_IDENTITIES
    ]
    terra_path = _write_payload(
        tmp_path,
        "terra.json",
        _payload([("terra", "codex", "gpt-5.6-terra")], terra_results),
    )
    sol_path = _write_payload(
        tmp_path,
        "sol.json",
        _payload([("sol", "codex", "gpt-5.6-sol")], sol_results),
    )

    board = fold_eval_board([terra_path, sol_path])

    assert board.suite_name == "EncodeBench UK v1"
    assert [case.name for case in board.cases] == ["alpha", "beta", "gamma"]
    by_name = {stats.runner: stats for stats in board.runners}
    terra = by_name["terra"]
    assert terra.cases_run == 3
    assert terra.gate_pass_count == 1
    assert terra.success_count == 2
    assert terra.zero_ungrounded_count == 1
    assert terra.compile_pass_count == 2
    assert terra.median_duration_seconds == 50.0
    assert terra.mean_cost_usd is None
    assert terra.source_numeric_coverage_rate == pytest.approx(18 / 20)
    sol = by_name["sol"]
    assert sol.gate_pass_count == 3
    assert sol.mean_cost_usd == pytest.approx(0.02)
    # Sol leads the ordering on gate-pass rate.
    assert board.ordered_runners()[0].runner == "sol"
    assert board.cells[(3, "terra")].state == "timeout"
    assert board.cells[(2, "terra")].state == "fail"
    assert "ungrounded=2" in board.cells[(2, "terra")].detail
    assert board.cells[(1, "terra")].state == "pass"
    assert board.mixed_toolchain_sources == []


def test_board_distinguishes_timeout_validation_failure_and_plain_error(tmp_path):
    results = [
        _result(
            "fable",
            CASE_IDENTITIES[0],
            backend="claude",
            model="claude-fable-5",
            success=False,
            error="Claude eval timed out",
            metrics=None,
            failure_kind="timeout",
            timed_out=True,
            timeout_stage="encoder",
            timeout_reason="wall",
            timeout_seconds=600,
            timeout_attempts=3,
        ),
        _result(
            "fable",
            CASE_IDENTITIES[1],
            backend="claude",
            model="claude-fable-5",
            success=False,
            error="Generated RuleSpec failed CI validation",
            metrics=_metrics(ci_pass=False),
            failure_kind="validation",
        ),
        _result(
            "fable",
            CASE_IDENTITIES[2],
            backend="claude",
            model="claude-fable-5",
            success=False,
            error="Claude CLI failed",
            metrics=None,
            failure_kind="error",
        ),
    ]
    path = _write_payload(
        tmp_path,
        "fable.json",
        _payload(
            [("fable", "claude", "claude-fable-5")],
            results,
        ),
    )

    board = fold_eval_board([path])
    stats = board.runners[0]
    assert board.cells[(1, "fable")].state == "timeout"
    assert board.cells[(2, "fable")].state == "fail"
    assert board.cells[(3, "fable")].state == "error"
    assert stats.timeout_count == 1
    assert stats.artifact_case_count == 1
    assert stats.compile_pass_rate == 1.0
    assert stats.ci_pass_rate == 0.0
    assert stats.zero_ungrounded_rate == 1.0
    assert "T = encoder/case timeout" in render_eval_board_markdown(board)
    assert "T timeout" in render_eval_board_text(board)


def test_fold_refuses_timeout_row_that_claims_a_generated_artifact(tmp_path):
    timeout_result = _result(
        "fable",
        CASE_IDENTITIES[0],
        backend="claude",
        model="claude-fable-5",
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
    timeout_result.update(
        {
            "output_file": "/tmp/generated.yaml",
            "generated_output_sha256": "a" * 64,
        }
    )
    path = _write_payload(
        tmp_path,
        "artifact-mislabeled-as-timeout.json",
        _payload(
            [("fable", "claude", "claude-fable-5")],
            [
                timeout_result,
                _result(
                    "fable",
                    CASE_IDENTITIES[1],
                    backend="claude",
                    model="claude-fable-5",
                ),
                _result(
                    "fable",
                    CASE_IDENTITIES[2],
                    backend="claude",
                    model="claude-fable-5",
                ),
            ],
        ),
    )

    with pytest.raises(EvalBoardError, match="no generated artifact"):
        fold_eval_board([path])


def test_gate_pass_requires_all_deterministic_checks():
    passing = _result("terra", CASE_IDENTITIES[0])
    assert result_gate_pass(passing)
    assert not result_gate_pass(
        _result("terra", CASE_IDENTITIES[0], metrics=_metrics(ci_pass=False))
    )
    assert not result_gate_pass(
        _result("terra", CASE_IDENTITIES[0], metrics=_metrics(compile_pass=False))
    )
    assert not result_gate_pass(
        _result("terra", CASE_IDENTITIES[0], metrics=_metrics(ungrounded=1))
    )
    assert not result_gate_pass(
        _result("terra", CASE_IDENTITIES[0], success=False, metrics=None)
    )
    assert not result_gate_pass(
        _result("terra", CASE_IDENTITIES[0], error="late failure")
    )


def test_fold_refuses_unknown_schema(tmp_path):
    path = _write_payload(
        tmp_path,
        "old.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            schema="axiom-encode/eval-suite-results/v1",
        ),
    )
    with pytest.raises(EvalBoardError, match="folds only"):
        load_eval_suite_results(path)


def test_fold_refuses_unknown_execution_identity_schema(tmp_path):
    identity = _execution_identity()
    identity["schema"] = "axiom-encode/eval-execution-identity/v2"
    path = _write_payload(
        tmp_path,
        "old-identity.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=identity,
        ),
    )
    with pytest.raises(EvalBoardError, match="execution identity carries schema"):
        fold_eval_board([path])


def test_fold_refuses_execution_identity_without_timeout_policy(tmp_path):
    identity = _execution_identity()
    identity.pop("runner_timeouts")
    path = _write_payload(
        tmp_path,
        "missing-timeout-policy.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=identity,
        ),
    )

    with pytest.raises(EvalBoardError, match="runner timeout"):
        fold_eval_board([path])


def test_fold_refuses_execution_identity_without_case_budget(tmp_path):
    identity = _execution_identity()
    identity.pop("case_timeout_seconds")
    path = _write_payload(
        tmp_path,
        "missing-case-budget.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=identity,
        ),
    )

    with pytest.raises(EvalBoardError, match="generation/retry case timeout"):
        fold_eval_board([path])


@pytest.mark.parametrize(
    "field_name",
    [
        "axiom_encode",
        "axiom_rules_engine",
        "rulespec_roots",
        "policyengine_runtime",
    ],
)
def test_fold_refuses_execution_identity_without_core_field(tmp_path, field_name):
    identity = _execution_identity()
    identity.pop(field_name)
    path = _write_payload(
        tmp_path,
        f"missing-{field_name}.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=identity,
        ),
    )

    with pytest.raises(EvalBoardError, match="core toolchain fields"):
        fold_eval_board([path])


@pytest.mark.parametrize(
    "pathspecs",
    [
        None,
        ["pyproject.toml", "src/axiom_encode", "uv.lock"],
        ["src/axiom_encode"],
        ["src/axiom_encode", "pyproject.toml", "uv.lock", "tests"],
    ],
)
def test_fold_refuses_encoder_identity_with_nonproducer_pathspecs(
    tmp_path,
    pathspecs,
):
    identity = _execution_identity()
    if pathspecs is None:
        identity["axiom_encode"].pop("pathspecs")
    else:
        identity["axiom_encode"]["pathspecs"] = pathspecs
    path = _write_payload(
        tmp_path,
        "invalid-encoder-pathspecs.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=identity,
        ),
    )

    with pytest.raises(EvalBoardError, match="core toolchain fields"):
        fold_eval_board([path])


def test_fold_refuses_rules_engine_identity_with_pathspecs(tmp_path):
    identity = _execution_identity()
    identity["axiom_rules_engine"]["pathspecs"] = ["src"]
    path = _write_payload(
        tmp_path,
        "invalid-engine-pathspecs.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=identity,
        ),
    )

    with pytest.raises(EvalBoardError, match="core toolchain fields"):
        fold_eval_board([path])


@pytest.mark.parametrize(
    "pathspecs",
    [
        None,
        ["uk", ".axiom/toolchain.toml"],
        [".axiom/toolchain.toml", "uk", "known-validation-gaps.yaml"],
        ["us", ".axiom/toolchain.toml", "known-validation-gaps.yaml"],
        [
            "uk",
            ".axiom/toolchain.toml",
            "known-validation-gaps.yaml",
            "README.md",
        ],
    ],
)
def test_fold_refuses_rulespec_identity_with_nonproducer_pathspecs(
    tmp_path,
    pathspecs,
):
    identity = _execution_identity()
    checkout_identity = identity["rulespec_roots"][0]["checkout_identity"]
    if pathspecs is None:
        checkout_identity.pop("pathspecs")
    else:
        checkout_identity["pathspecs"] = pathspecs
    path = _write_payload(
        tmp_path,
        "invalid-rulespec-pathspecs.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=identity,
        ),
    )

    with pytest.raises(EvalBoardError, match="core toolchain fields"):
        fold_eval_board([path])


def test_fold_refuses_path_only_engine_identities_that_normalize_equal(tmp_path):
    left_identity = _execution_identity()
    left_identity["axiom_rules_engine"] = {"path": "/engine/left"}
    right_identity = _execution_identity()
    right_identity["axiom_rules_engine"] = {"path": "/engine/right"}
    left = _write_payload(
        tmp_path,
        "left-path-only-engine.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=left_identity,
        ),
    )
    right = _write_payload(
        tmp_path,
        "right-path-only-engine.json",
        _payload(
            [("sol", "codex", "gpt-5.6-sol")],
            [_result("sol", case, model="gpt-5.6-sol") for case in CASE_IDENTITIES],
            execution_identity=right_identity,
        ),
    )

    with pytest.raises(EvalBoardError, match="core toolchain fields"):
        fold_eval_board([left, right])


def test_fold_refuses_execution_identity_without_any_rulespec_root(tmp_path):
    identity = _execution_identity()
    identity["rulespec_roots"] = []
    path = _write_payload(
        tmp_path,
        "empty-rulespec-roots.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=identity,
        ),
    )

    with pytest.raises(EvalBoardError, match="core toolchain fields"):
        fold_eval_board([path])


@pytest.mark.parametrize(
    ("content_state", "file_count"),
    [
        ("missing", 0),
        ("file", 1),
    ],
)
def test_fold_refuses_nondirectory_rulespec_root(
    tmp_path,
    content_state,
    file_count,
):
    identity = _execution_identity()
    identity["rulespec_roots"][0]["content_state"] = content_state
    identity["rulespec_roots"][0]["file_count"] = file_count
    path = _write_payload(
        tmp_path,
        f"{content_state}-rulespec-root.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=identity,
        ),
    )

    with pytest.raises(EvalBoardError, match="core toolchain fields"):
        fold_eval_board([path])


@pytest.mark.parametrize(
    "field_name",
    [
        "path",
        "content_state",
        "content_sha256",
        "file_count",
        "toolchain_root",
        "checkout_identity",
        "toolchain_contract_sha256",
        "validation_waiver_set_sha256",
    ],
)
def test_fold_refuses_rulespec_root_without_required_identity_field(
    tmp_path,
    field_name,
):
    identity = _execution_identity()
    identity["rulespec_roots"][0].pop(field_name)
    path = _write_payload(
        tmp_path,
        f"missing-rulespec-root-{field_name}.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=identity,
        ),
    )

    with pytest.raises(EvalBoardError, match="core toolchain fields"):
        fold_eval_board([path])


def test_fold_refuses_policyengine_runtime_wrapper_digest_mismatch(tmp_path):
    runtime = _policyengine_runtime_identity()
    runtime["sha256"] = "99" * 32
    identity = _execution_identity(policyengine_runtime=runtime)
    path = _write_payload(
        tmp_path,
        "stale-policyengine-wrapper-digest.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=identity,
        ),
    )

    with pytest.raises(EvalBoardError, match="core toolchain fields"):
        fold_eval_board([path])


def test_fold_refuses_location_only_policyengine_runtime_identity(tmp_path):
    runtime_identity = {"repository_root": "/ci/pe-uk"}
    identity = _execution_identity(
        policyengine_runtime={
            "identity": runtime_identity,
            "sha256": evals_canonical_json_sha256(runtime_identity),
        }
    )
    path = _write_payload(
        tmp_path,
        "location-only-policyengine-runtime.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=identity,
        ),
    )

    with pytest.raises(EvalBoardError, match="core toolchain fields"):
        fold_eval_board([path])


def test_fold_refuses_execution_identity_without_timeout_retry_policy(tmp_path):
    identity = _execution_identity()
    identity.pop("timeout_retry_policy")
    path = _write_payload(
        tmp_path,
        "missing-timeout-retry-policy.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=identity,
        ),
    )

    with pytest.raises(EvalBoardError, match="timeout retry"):
        fold_eval_board([path])


def test_fold_refuses_stale_execution_identity_digest(tmp_path):
    path = _write_payload(
        tmp_path,
        "stale-digest.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity_sha256="99" * 32,
        ),
    )
    with pytest.raises(EvalBoardError, match="digest does not match"):
        fold_eval_board([path])


def test_fold_refuses_tampered_evidence_digest(tmp_path):
    path = _write_payload(
        tmp_path,
        "tampered-evidence.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            evidence_overrides={"sha256": "42" * 32},
        ),
    )
    with pytest.raises(EvalBoardError, match="evidence digest"):
        fold_eval_board([path])


def test_fold_refuses_unknown_evidence_schema(tmp_path):
    path = _write_payload(
        tmp_path,
        "old-evidence.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            evidence_overrides={"schema": "axiom-encode/eval-suite-evidence/v3"},
        ),
    )
    with pytest.raises(EvalBoardError, match="evidence schema"):
        fold_eval_board([path])


def test_fold_refuses_incomplete_corpus_identity(tmp_path):
    partial_corpus = {
        "corpus_release": "uk-rulespec-2026-07-14",
        "corpus_release_content_sha256": "dd" * 32,
    }
    path = _write_payload(
        tmp_path,
        "no-selector.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            corpus=partial_corpus,
        ),
    )
    with pytest.raises(EvalBoardError, match="corpus release identity"):
        fold_eval_board([path])


def test_fold_refuses_tampered_result_row_binding(tmp_path):
    rows = [_result("terra", case) for case in CASE_IDENTITIES]
    rows[1]["result_sha256"] = "24" * 32
    path = _write_payload(
        tmp_path,
        "tampered-row.json",
        _payload([("terra", "codex", "gpt-5.6-terra")], rows),
    )
    with pytest.raises(EvalBoardError, match="result_sha256"):
        fold_eval_board([path])


def test_fold_refuses_tampered_results_digest(tmp_path):
    path = _write_payload(
        tmp_path,
        "tampered-results.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            results_sha256="00" * 32,
        ),
    )
    with pytest.raises(EvalBoardError, match="results_sha256"):
        fold_eval_board([path])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"expected_result_count": 99}, "expected_result_count"),
        ({"completed_case_count": 99}, "completed_case_count"),
    ],
)
def test_fold_refuses_inconsistent_coverage_counts(tmp_path, override, message):
    path = _write_payload(
        tmp_path,
        "bad-counts.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            coverage_overrides=override,
        ),
    )
    with pytest.raises(EvalBoardError, match=message):
        fold_eval_board([path])


def test_fold_refuses_mismatched_case_identities(tmp_path):
    left = _write_payload(
        tmp_path,
        "left.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
        ),
    )
    changed = [dict(identity) for identity in CASE_IDENTITIES]
    changed[1] = {**changed[1], "sha256": "ee" * 32}
    right = _write_payload(
        tmp_path,
        "right.json",
        _payload(
            [("sol", "codex", "gpt-5.6-sol")],
            [_result("sol", case, model="gpt-5.6-sol") for case in changed],
            case_identities=changed,
        ),
    )
    with pytest.raises(EvalBoardError, match="case identities"):
        fold_eval_board([left, right])


def test_fold_refuses_mismatched_corpus_release(tmp_path):
    left = _write_payload(
        tmp_path,
        "left.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
        ),
    )
    right = _write_payload(
        tmp_path,
        "right.json",
        _payload(
            [("sol", "codex", "gpt-5.6-sol")],
            [_result("sol", case, model="gpt-5.6-sol") for case in CASE_IDENTITIES],
            corpus={
                "corpus_release": "uk-rulespec-2026-08-01",
                "corpus_release_content_sha256": "ff" * 32,
                "corpus_release_selector_sha256": "fe" * 32,
            },
        ),
    )
    with pytest.raises(EvalBoardError, match="corpus release"):
        fold_eval_board([left, right])


def test_fold_refuses_mismatched_execution_identity(tmp_path):
    left = _write_payload(
        tmp_path,
        "left.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
        ),
    )
    right = _write_payload(
        tmp_path,
        "right.json",
        _payload(
            [("sol", "codex", "gpt-5.6-sol")],
            [_result("sol", case, model="gpt-5.6-sol") for case in CASE_IDENTITIES],
            execution_identity=_execution_identity(encoder_commit="4" * 40),
        ),
    )
    with pytest.raises(EvalBoardError, match="execution identity"):
        fold_eval_board([left, right])

    board = fold_eval_board([left, right], allow_mixed_toolchains=True)
    assert board.mixed_toolchain_sources == [str(right)]
    markdown = render_eval_board_markdown(board)
    assert "Mixed toolchains" in markdown


def test_fold_refuses_mismatched_encoder_timeout(tmp_path):
    left = _write_payload(
        tmp_path,
        "left.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=_execution_identity(claude_timeout_seconds=1200),
        ),
    )
    right = _write_payload(
        tmp_path,
        "right.json",
        _payload(
            [("sol", "codex", "gpt-5.6-sol")],
            [_result("sol", case, model="gpt-5.6-sol") for case in CASE_IDENTITIES],
            execution_identity=_execution_identity(claude_timeout_seconds=1800),
        ),
    )

    with pytest.raises(EvalBoardError, match="execution identity"):
        fold_eval_board([left, right])


def test_fold_refuses_mismatched_codex_timeout(tmp_path):
    left = _write_payload(
        tmp_path,
        "left.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=_execution_identity(codex_timeout_seconds=599),
        ),
    )
    right = _write_payload(
        tmp_path,
        "right.json",
        _payload(
            [("sol", "codex", "gpt-5.6-sol")],
            [_result("sol", case, model="gpt-5.6-sol") for case in CASE_IDENTITIES],
            execution_identity=_execution_identity(codex_timeout_seconds=600),
        ),
    )

    with pytest.raises(EvalBoardError, match="execution identity"):
        fold_eval_board([left, right])


def test_fold_refuses_mismatched_generation_retry_case_timeout(tmp_path):
    left = _write_payload(
        tmp_path,
        "left.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=_execution_identity(case_timeout_seconds=2400),
        ),
    )
    right = _write_payload(
        tmp_path,
        "right.json",
        _payload(
            [("sol", "codex", "gpt-5.6-sol")],
            [_result("sol", case, model="gpt-5.6-sol") for case in CASE_IDENTITIES],
            execution_identity=_execution_identity(case_timeout_seconds=3600),
        ),
    )

    with pytest.raises(EvalBoardError, match="execution identity"):
        fold_eval_board([left, right])


def test_fold_refuses_mismatched_timeout_retry_policy(tmp_path):
    left = _write_payload(
        tmp_path,
        "left.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=_execution_identity(suite_max_attempts=1),
        ),
    )
    right = _write_payload(
        tmp_path,
        "right.json",
        _payload(
            [("sol", "codex", "gpt-5.6-sol")],
            [_result("sol", case, model="gpt-5.6-sol") for case in CASE_IDENTITIES],
            execution_identity=_execution_identity(suite_max_attempts=3),
        ),
    )

    with pytest.raises(EvalBoardError, match="execution identity"):
        fold_eval_board([left, right])


def test_fold_ignores_checkout_locations_in_execution_identity(tmp_path):
    left = _write_payload(
        tmp_path,
        "left.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=_execution_identity(checkout="/home/ci/encode"),
        ),
    )
    right = _write_payload(
        tmp_path,
        "right.json",
        _payload(
            [("sol", "codex", "gpt-5.6-sol")],
            [_result("sol", case, model="gpt-5.6-sol") for case in CASE_IDENTITIES],
            execution_identity=_execution_identity(checkout="/Users/max/encode"),
        ),
    )
    board = fold_eval_board([left, right])
    assert board.mixed_toolchain_sources == []
    assert len(board.runners) == 2


def test_fold_ignores_policyengine_runtime_locations(tmp_path):
    left = _write_payload(
        tmp_path,
        "left.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            execution_identity=_execution_identity(
                policyengine_runtime=_policyengine_runtime_identity(
                    root="/home/ci/pe-uk"
                ),
            ),
        ),
    )
    right = _write_payload(
        tmp_path,
        "right.json",
        _payload(
            [("sol", "codex", "gpt-5.6-sol")],
            [_result("sol", case, model="gpt-5.6-sol") for case in CASE_IDENTITIES],
            execution_identity=_execution_identity(
                policyengine_runtime=_policyengine_runtime_identity(
                    root="/Users/max/pe-uk"
                ),
            ),
        ),
    )
    board = fold_eval_board([left, right])
    assert board.mixed_toolchain_sources == []

    # A genuinely different PolicyEngine version still refuses.
    upgraded = _write_payload(
        tmp_path,
        "upgraded.json",
        _payload(
            [("luna", "codex", "gpt-5.6-luna")],
            [_result("luna", case, model="gpt-5.6-luna") for case in CASE_IDENTITIES],
            execution_identity=_execution_identity(
                policyengine_runtime=_policyengine_runtime_identity(
                    root="/home/ci/pe-uk", pe_version="1.10.0"
                ),
            ),
        ),
    )
    with pytest.raises(EvalBoardError, match="execution identity"):
        fold_eval_board([left, upgraded])


def test_normalized_execution_identity_drops_location_fields():
    identity = _execution_identity(
        checkout="/somewhere/deep",
        policyengine_runtime=_policyengine_runtime_identity(root="/pe/elsewhere"),
    )
    normalized = normalized_execution_identity(identity)
    rendered = json.dumps(normalized)
    assert "/somewhere/deep" not in rendered
    assert "/pe/elsewhere" not in rendered
    assert "/usr/local/python-3.13.5" not in rendered
    for key in (
        '"path"',
        '"toolchain_root"',
        '"repository_root"',
        '"venv_root"',
        '"python_executable"',
        '"effective_sys_path"',
        '"module_origin"',
        '"metadata_root"',
    ):
        assert key not in rendered
    # Score-affecting fields survive.
    assert identity["axiom_encode"]["commit"] in rendered
    assert identity["rulespec_roots"][0]["validation_waiver_set_sha256"] in rendered
    assert '"locked_versions"' in rendered
    assert '"1.9.0"' in rendered


def test_fold_refuses_duplicate_runner(tmp_path):
    first = _write_payload(
        tmp_path,
        "first.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
        ),
    )
    second = _write_payload(
        tmp_path,
        "second.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
        ),
    )
    with pytest.raises(EvalBoardError, match="two boards"):
        fold_eval_board([first, second])


def test_fold_refuses_rows_for_undeclared_runner(tmp_path):
    first = _write_payload(
        tmp_path,
        "first.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", CASE_IDENTITIES[0])],
            complete=False,
        ),
    )
    # The second payload declares sol but smuggles a row for terra.
    second = _write_payload(
        tmp_path,
        "second.json",
        _payload(
            [("sol", "codex", "gpt-5.6-sol")],
            [
                _result("sol", CASE_IDENTITIES[0], model="gpt-5.6-sol"),
                _result("terra", CASE_IDENTITIES[1]),
            ],
            complete=False,
        ),
    )
    with pytest.raises(EvalBoardError, match="never declared"):
        fold_eval_board([first, second], allow_partial=True)


@pytest.mark.parametrize(
    "row_overrides",
    [{"model": "gpt-5.6-luna"}, {"backend": "claude"}],
)
def test_fold_refuses_rows_with_wrong_backend_or_model(tmp_path, row_overrides):
    rows = [_result("terra", case) for case in CASE_IDENTITIES]
    rows[1].update(row_overrides)
    path = _write_payload(
        tmp_path,
        "wrong-identity.json",
        _payload([("terra", "codex", "gpt-5.6-terra")], rows),
    )
    with pytest.raises(EvalBoardError, match="declared as"):
        fold_eval_board([path])


def test_fold_refuses_malformed_runner_declarations(tmp_path):
    payload = _payload(
        [("terra", "codex", "gpt-5.6-terra")],
        [_result("terra", case) for case in CASE_IDENTITIES],
    )
    # Tamper below the digest layer: keep the evidence digest consistent so
    # the declaration validator (not the digest check) is what fires.
    del payload["evidence"]["effective_runner_identities"][0]["backend"]
    unsigned = dict(payload["evidence"])
    unsigned.pop("sha256", None)
    payload["evidence"]["sha256"] = cli._eval_suite_json_sha256(unsigned)
    path = _write_payload(tmp_path, "no-backend.json", payload)
    with pytest.raises(EvalBoardError, match="without a\nvalid backend|valid backend"):
        fold_eval_board([path])


def test_fold_refuses_complete_claim_with_missing_rows(tmp_path):
    partial_matrix = _write_payload(
        tmp_path,
        "claimed-complete.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", CASE_IDENTITIES[0])],
            complete=True,
            coverage_overrides={"actual_result_count": 1},
        ),
    )
    with pytest.raises(EvalBoardError, match="coverage.complete"):
        fold_eval_board([partial_matrix])


def test_fold_refuses_incomplete_claim_with_full_matrix(tmp_path):
    contradictory = _write_payload(
        tmp_path,
        "contradictory.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            complete=False,
        ),
    )
    with pytest.raises(EvalBoardError, match="full\nresult matrix|full result matrix"):
        fold_eval_board([contradictory], allow_partial=True)


def test_fold_refuses_non_boolean_complete_flag(tmp_path):
    payload = _payload(
        [("terra", "codex", "gpt-5.6-terra")],
        [_result("terra", case) for case in CASE_IDENTITIES],
        coverage_overrides={"complete": "false"},
    )
    path = _write_payload(tmp_path, "stringy.json", payload)
    with pytest.raises(EvalBoardError, match="boolean"):
        fold_eval_board([path])


def test_fold_refuses_out_of_range_case_index(tmp_path):
    path = _write_payload(
        tmp_path,
        "out-of-range.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [
                _result("terra", CASE_IDENTITIES[0]),
                _result("terra", CASE_IDENTITIES[1]),
                _result(
                    "terra",
                    CASE_IDENTITIES[2],
                    eval_case_overrides={"index": 9},
                ),
            ],
        ),
    )
    with pytest.raises(EvalBoardError, match="outside the manifest"):
        fold_eval_board([path])


@pytest.mark.parametrize("bad_index", [True, 1.0])
def test_fold_refuses_non_integer_reference_indexes(tmp_path, bad_index):
    loosened = [dict(identity) for identity in CASE_IDENTITIES]
    loosened[0] = {**loosened[0], "index": bad_index}
    path = _write_payload(
        tmp_path,
        "loose-index.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            case_identities=loosened,
        ),
    )
    with pytest.raises(EvalBoardError, match="malformed at position 1"):
        fold_eval_board([path])


def test_fold_refuses_malformed_reference_case_indexes(tmp_path):
    duplicated = [dict(identity) for identity in CASE_IDENTITIES]
    duplicated[1] = dict(duplicated[0])
    path = _write_payload(
        tmp_path,
        "dup-index.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
            case_identities=duplicated,
        ),
    )
    with pytest.raises(EvalBoardError, match="malformed at position 2"):
        fold_eval_board([path])


@pytest.mark.parametrize(
    "mutation",
    [
        {"name": "tampered"},
        {"kind": "citation"},
        {"corpus_citation_path": "uk/statute/ukpga/9999/1/1"},
        {"sha256": "f0" * 32},
    ],
)
def test_fold_refuses_case_identity_mismatch_in_result(tmp_path, mutation):
    path = _write_payload(
        tmp_path,
        "mutated.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [
                _result("terra", CASE_IDENTITIES[0], eval_case_overrides=mutation),
                _result("terra", CASE_IDENTITIES[1]),
                _result("terra", CASE_IDENTITIES[2]),
            ],
        ),
    )
    with pytest.raises(EvalBoardError, match="does not match the manifest"):
        fold_eval_board([path])


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda result: result.update(metrics="bad"),
            "metrics must be null or an object",
        ),
        (
            lambda result: result.update(duration_ms=True),
            "duration_ms must be an integer",
        ),
        (
            lambda result: result.update(duration_ms=-5),
            "duration_ms must be nonnegative",
        ),
        (
            lambda result: result.update(estimated_cost_usd=-0.5),
            "estimated_cost_usd must be nonnegative",
        ),
        (
            lambda result: result["metrics"].update(compile_pass="false"),
            "compile_pass must be a boolean",
        ),
        (
            lambda result: result["metrics"].update(
                source_numeric_occurrence_count=2,
                covered_source_numeric_occurrence_count=5,
            ),
            "covers 5 source numeric occurrences out of",
        ),
        (
            lambda result: (
                result["metrics"].pop("source_numeric_occurrence_count"),
                result["metrics"].pop("covered_source_numeric_occurrence_count"),
            ),
            "source_numeric_occurrence_count must be an integer",
        ),
        (
            lambda result: result["metrics"].update(ungrounded_numeric_count=-1),
            "ungrounded_numeric_count must be nonnegative",
        ),
        (
            lambda result: result["metrics"].update(
                source_numeric_occurrence_count=-3,
            ),
            "source_numeric_occurrence_count must be nonnegative",
        ),
    ],
)
def test_fold_refuses_malformed_result_rows(tmp_path, mutator, message):
    results = [_result("terra", case) for case in CASE_IDENTITIES]
    mutator(results[1])
    path = _write_payload(
        tmp_path,
        "malformed.json",
        _payload([("terra", "codex", "gpt-5.6-terra")], results),
    )
    with pytest.raises(EvalBoardError, match=message):
        fold_eval_board([path])


def test_out_of_range_reviewer_scores_fold_like_the_producer_emits_them(tmp_path):
    """The producer does not range-check reviewer scores, so the board must
    not refuse a payload over one; sign and range stay the producer's
    contract, not the consumer's."""
    path = _write_payload(
        tmp_path,
        "rogue-score.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [
                _result(
                    "terra",
                    CASE_IDENTITIES[0],
                    metrics=_metrics(review_score=-2.0),
                ),
                _result("terra", CASE_IDENTITIES[1]),
                _result("terra", CASE_IDENTITIES[2]),
            ],
        ),
    )
    board = fold_eval_board([path])
    assert board.runners[0].mean_generalist_review_score == pytest.approx(
        (-2.0 + 8.5 + 8.5) / 3
    )


def test_oracle_failures_without_scores_stay_in_denominator(tmp_path):
    path = _write_payload(
        tmp_path,
        "oracle.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [
                _result(
                    "terra",
                    CASE_IDENTITIES[0],
                    metrics=_metrics(policyengine_pass=True, policyengine_score=1.0),
                ),
                # A legitimate oracle exception: pass=False with no score.
                _result(
                    "terra",
                    CASE_IDENTITIES[1],
                    metrics=_metrics(policyengine_pass=False, policyengine_score=None),
                ),
                _result("terra", CASE_IDENTITIES[2]),
            ],
            execution_identity=_execution_identity(
                policyengine_runtime=_policyengine_runtime_identity(),
            ),
        ),
    )
    board = fold_eval_board([path])
    stats = board.runners[0]
    assert stats.policyengine_case_count == 2
    assert stats.policyengine_pass_count == 1
    assert stats.policyengine_pass_rate == pytest.approx(0.5)


def test_fold_refuses_oracle_evidence_without_bound_policyengine_runtime(tmp_path):
    path = _write_payload(
        tmp_path,
        "unbound-oracle.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [
                _result(
                    "terra",
                    CASE_IDENTITIES[0],
                    metrics=_metrics(policyengine_pass=True, policyengine_score=1.0),
                ),
                _result("terra", CASE_IDENTITIES[1]),
                _result("terra", CASE_IDENTITIES[2]),
            ],
        ),
    )

    with pytest.raises(EvalBoardError, match="PolicyEngine runtime"):
        fold_eval_board([path])


def test_fold_allows_honest_partial_run(tmp_path):
    partial = _write_payload(
        tmp_path,
        "partial.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", CASE_IDENTITIES[0])],
            complete=False,
        ),
    )
    with pytest.raises(EvalBoardError, match="incomplete"):
        fold_eval_board([partial])

    board = fold_eval_board([partial], allow_partial=True)
    assert board.incomplete_sources == [str(partial)]
    stats = board.runners[0]
    assert stats.cases_run == 1
    assert board.cells[(2, "terra")].state == "missing"
    assert board.cells[(3, "terra")].state == "missing"


def test_partial_runner_ranks_by_rate_not_raw_count(tmp_path):
    # Partial runner: 1 pass of 1 run (rate 1.0). Complete runner: 2 passes
    # of 3 (rate 0.667) with the higher raw count. Rate-first ordering puts
    # the partial runner first; count-first ordering would invert it.
    partial = _write_payload(
        tmp_path,
        "partial.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", CASE_IDENTITIES[0])],
            complete=False,
        ),
    )
    complete = _write_payload(
        tmp_path,
        "complete.json",
        _payload(
            [("sol", "codex", "gpt-5.6-sol")],
            [
                _result("sol", CASE_IDENTITIES[0], model="gpt-5.6-sol"),
                _result("sol", CASE_IDENTITIES[1], model="gpt-5.6-sol"),
                _result(
                    "sol",
                    CASE_IDENTITIES[2],
                    model="gpt-5.6-sol",
                    metrics=_metrics(ci_pass=False),
                ),
            ],
        ),
    )
    board = fold_eval_board([partial, complete], allow_partial=True)
    ordered = [stats.runner for stats in board.ordered_runners()]
    assert ordered == ["terra", "sol"]


def test_ordering_breaks_ties_by_speed_before_name(tmp_path):
    # The faster runner sorts LAST alphabetically, so a name-ordered fold
    # would invert this board.
    fast = _write_payload(
        tmp_path,
        "fast.json",
        _payload(
            [("zulu", "codex", "gpt-5.6-terra")],
            [_result("zulu", case, duration_ms=10_000) for case in CASE_IDENTITIES],
        ),
    )
    slow = _write_payload(
        tmp_path,
        "slow.json",
        _payload(
            [("alpha", "codex", "gpt-5.6-sol")],
            [
                _result("alpha", case, model="gpt-5.6-sol", duration_ms=90_000)
                for case in CASE_IDENTITIES
            ],
        ),
    )
    board = fold_eval_board([fast, slow])
    assert [stats.runner for stats in board.ordered_runners()] == ["zulu", "alpha"]


def test_even_case_count_median(tmp_path):
    two_cases = CASE_IDENTITIES[:2]
    path = _write_payload(
        tmp_path,
        "two.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [
                _result("terra", two_cases[0], duration_ms=40_000),
                _result("terra", two_cases[1], duration_ms=50_000),
            ],
            case_identities=two_cases,
        ),
    )
    board = fold_eval_board([path])
    assert board.runners[0].median_duration_seconds == pytest.approx(45.0)


def test_fold_accepts_directory_input(tmp_path):
    output_dir = tmp_path / "terra-run"
    output_dir.mkdir()
    _write_payload(
        output_dir,
        "results.json",
        _payload(
            [("terra", "codex", "gpt-5.6-terra")],
            [_result("terra", case) for case in CASE_IDENTITIES],
        ),
    )
    board = fold_eval_board([output_dir])
    assert board.runners[0].gate_pass_count == 3


def test_load_rejects_payload_without_evidence(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": SUPPORTED_RESULTS_SCHEMA, "results": []}))
    with pytest.raises(EvalBoardError, match="evidence"):
        load_eval_suite_results(path)


def test_renderers_and_exports(tmp_path):
    terra_payload = _payload(
        [("terra", "codex", "gpt-5.6-terra")],
        [
            _result("terra", CASE_IDENTITIES[0]),
            _result(
                "terra",
                CASE_IDENTITIES[1],
                metrics=_metrics(ungrounded=1),
            ),
            _result(
                "terra",
                CASE_IDENTITIES[2],
                success=False,
                error="boom",
                metrics=None,
                failure_kind="error",
            ),
        ],
    )
    terra_path = _write_payload(tmp_path, "terra.json", terra_payload)
    board = fold_eval_board([terra_path])

    markdown = render_eval_board_markdown(board)
    assert "# Eval board — EncodeBench UK v1" in markdown
    assert "uk-rulespec-2026-07-14" in markdown
    assert "| terra |" in markdown
    assert "01 alpha" in markdown

    text = render_eval_board_text(board)
    assert "gate 1/3" in text
    grid_lines = [line for line in text.splitlines() if line.startswith("  0")]
    assert [line.split()[-1] for line in grid_lines] == ["P", "F", "E"]

    payload = eval_board_to_json(board)
    assert payload["schema"] == "axiom-encode/eval-board/v2"
    assert payload["runners"][0]["gate_pass_count"] == 1
    expected_digest = terra_payload["evidence"]["execution_identity_sha256"]
    assert payload["execution_identity_sha256s"] == {str(terra_path): expected_digest}
    assert len(payload["cells"]) == 3

    rows = eval_board_case_rows(board)
    assert [row["case_name"] for row in rows] == ["alpha", "beta", "gamma"]
    assert rows[0]["terra"] == "pass"
    assert rows[1]["terra"] == "fail"
    assert rows[2]["terra"] == "error"


def test_capability_manifest_locks_shape():
    manifest = load_eval_suite_manifest(CAPABILITY_MANIFEST)

    assert manifest.name == "EncodeBench UK v1"
    assert len(manifest.cases) == 16
    names = [case.name for case in manifest.cases]
    assert len(set(names)) == 16

    # Rate gates are pinned to 0.0: the suite reports rates, it does not
    # gate, and the loader refuses omitted or null rate gates.
    assert manifest.gates.min_cases == 16
    assert manifest.gates.min_success_rate == 0.0
    assert manifest.gates.min_compile_pass_rate == 0.0
    assert manifest.gates.min_ci_pass_rate == 0.0
    assert manifest.gates.min_zero_ungrounded_rate == 0.0
    assert manifest.gates.min_generalist_review_pass_rate == 0.0
    assert manifest.gates.min_policyengine_pass_rate is None

    # Capability cases run cold; only the three oracle candidates are
    # repo-augmented, and none carries a live oracle yet.
    repo_augmented = [
        case.name for case in manifest.cases if case.mode == "repo-augmented"
    ]
    assert repo_augmented == [
        "income_tax_main_rates",
        "class_1_primary_nic",
        "child_benefit_weekly_rates",
    ]
    assert all(case.oracle == "none" for case in manifest.cases)
    assert all(case.kind == "source" for case in manifest.cases)

    # The roster is subscription-billed backends only.
    backends = {spec.split("=", 1)[-1].split(":", 1)[0] for spec in manifest.runners}
    assert backends == {"codex", "claude"}
    assert manifest.runners == [
        "terra=codex:gpt-5.6-terra",
        "sol=codex:gpt-5.6-sol",
        "gpt-5.5=codex:gpt-5.5",
        "luna=codex:gpt-5.6-luna",
        "fable=claude:claude-fable-5",
        "opus-5=claude:claude-opus-5",
    ]
