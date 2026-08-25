"""An unreadable generated RuleSpec must not kill the bounded retry loop.

A protected re-encode of ``ca/policy/cra/benefits-2026/federal-family-and-
climate-benefits`` died on its second attempt because the model emitted one
proof excerpt as an unquoted YAML scalar containing ``": "``. The overlay
step raised ``ValueError`` before any validator ran, the exception escaped
``_run_single_eval``, and the run lost its two remaining attempts, its
final rejected candidate, and the near-passing artifact from attempt one.

An artifact the overlay cannot read is an ordinary validator rejection: the
overlay failure is recorded as a CI issue and returned with the standard CI
failure classification so the bounded retry loop regenerates it with feedback.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from axiom_encode.harness.evals import (
    EvalPromptResponse,
    EvalWorkspace,
    ValidationRetryCandidate,
    _repair_overlay_candidate_base_issue,
    _run_single_eval,
    parse_runner_spec,
    resolve_corpus_source_unit,
)
from tests.release_object_fixtures import bind_test_corpus_release

_OVERLAY_ERROR = "repair overlay RuleSpec must be valid UTF-8 YAML"


def _make_workspace(root: Path) -> EvalWorkspace:
    root.mkdir(parents=True, exist_ok=True)
    source_text_file = root / "source.txt"
    source_text_file.write_text("operative source text\n")
    manifest_file = root / "context-manifest.json"
    manifest_file.write_text("{}")
    return EvalWorkspace(
        root=root,
        source_text_file=source_text_file,
        manifest_file=manifest_file,
    )


def _bind_corpus(tmp_path: Path):
    corpus_path = tmp_path / "corpus"
    (corpus_path / "data/corpus/provisions").mkdir(parents=True)
    selector = corpus_path / "manifests/releases/overlay-retry-test-release.json"
    selector.parent.mkdir(parents=True)
    selector.write_text(
        json.dumps(
            {
                "name": "overlay-retry-test-release",
                "scopes": [
                    {
                        "jurisdiction": "us-ca",
                        "document_class": "regulation",
                        "version": "test-version",
                    }
                ],
            }
        )
    )
    provision_file = (
        corpus_path / "data/corpus/provisions/us-ca/regulation/test-version.jsonl"
    )
    provision_file.parent.mkdir(parents=True, exist_ok=True)
    provision_file.write_text(
        json.dumps(
            {
                "id": "overlay-retry",
                "citation_path": "us-ca/regulation/mpp/63-503",
                "body": "63-503 operative source text",
                "jurisdiction": "us-ca",
                "document_class": "regulation",
                "version": "test-version",
                "source_path": "sources/us-ca/regulation/test-version",
                "source_as_of": "2026-01-01",
                "expression_date": "2026-01-01",
            }
        )
        + "\n"
    )
    return bind_test_corpus_release(
        corpus_path,
        "overlay-retry-test-release",
        [("us-ca", "regulation", "test-version")],
    )


def test_unreadable_overlay_candidate_does_not_abort_the_eval(tmp_path, capsys):
    output_root = tmp_path / "out"
    workspace = _make_workspace(output_root / "_eval_workspaces" / "workspace")
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    rules_path = tmp_path / "rules"
    rules_path.mkdir()

    corpus_release = _bind_corpus(tmp_path)
    source_unit = resolve_corpus_source_unit(
        "us-ca/regulation/mpp/63-503",
        corpus_release,
    )
    response = EvalPromptResponse(
        text="format: rulespec/v1\nmodule:\n  summary: stub\nrules: []\n",
        duration_ms=1,
    )
    candidate = ValidationRetryCandidate(
        rulespec=(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: preserved\n"
            "    kind: parameter\n"
            "    dtype: Count\n"
            "    versions: [{effective_from: '2026-01-01', formula: 1}]\n"
        )
    )

    with (
        patch(
            "axiom_encode.harness.evals.resolve_corpus_source_unit",
            return_value=source_unit,
        ),
        patch(
            "axiom_encode.harness.evals.prepare_eval_workspace",
            return_value=workspace,
        ),
        patch("axiom_encode.harness.evals._run_prompt_eval", return_value=response),
        patch("axiom_encode.harness.evals.evaluate_artifact", return_value=None),
        patch("axiom_encode.harness.evals._hydrate_eval_root"),
        patch(
            "axiom_encode.harness.evals._overlay_validation_retry_candidate",
            side_effect=ValueError(_OVERLAY_ERROR),
        ),
    ):
        result = _run_single_eval(
            citation="us-ca/regulation/mpp/63-503",
            runner=parse_runner_spec("codex:gpt-5.5"),
            output_root=output_root,
            policy_path=policy_path,
            runtime_axiom_rules_path=rules_path,
            corpus_release=corpus_release,
            mode="cold",
            extra_context_paths=[],
            source_unit=source_unit,
            validation_retry_candidate=candidate,
        )

    assert result is not None, (
        "An unreadable overlay candidate must be reported through the normal "
        "eval result so the bounded retry loop keeps its remaining attempts"
    )
    assert result.success is False
    assert result.error == "Generated RuleSpec failed CI validation"
    assert result.failure_kind == "validation"
    assert Path(result.output_file).read_text() == candidate.rulespec
    assert (
        f"repair_candidate_overlay_skipped:{_OVERLAY_ERROR}" in capsys.readouterr().out
    )


def test_unreadable_retained_candidate_does_not_replace_valid_generation(
    tmp_path, capsys
):
    output_root = tmp_path / "out"
    workspace = _make_workspace(output_root / "_eval_workspaces" / "workspace")
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    rules_path = tmp_path / "rules"
    rules_path.mkdir()

    corpus_release = _bind_corpus(tmp_path)
    source_unit = resolve_corpus_source_unit(
        "us-ca/regulation/mpp/63-503",
        corpus_release,
    )
    generated = "format: rulespec/v1\nmodule:\n  summary: recovered\nrules: []\n"
    response = EvalPromptResponse(text=generated, duration_ms=1)
    candidate = ValidationRetryCandidate(rulespec="format: [unterminated\n")

    with (
        patch(
            "axiom_encode.harness.evals.resolve_corpus_source_unit",
            return_value=source_unit,
        ),
        patch(
            "axiom_encode.harness.evals.prepare_eval_workspace",
            return_value=workspace,
        ),
        patch("axiom_encode.harness.evals._run_prompt_eval", return_value=response),
        patch("axiom_encode.harness.evals.evaluate_artifact", return_value=None),
        patch("axiom_encode.harness.evals._hydrate_eval_root"),
    ):
        result = _run_single_eval(
            citation="us-ca/regulation/mpp/63-503",
            runner=parse_runner_spec("codex:gpt-5.5"),
            output_root=output_root,
            policy_path=policy_path,
            runtime_axiom_rules_path=rules_path,
            corpus_release=corpus_release,
            mode="cold",
            extra_context_paths=[],
            source_unit=source_unit,
            validation_retry_candidate=candidate,
        )

    assert result is not None
    assert result.success is False
    assert result.error == "Generated RuleSpec failed CI validation"
    assert Path(result.output_file).read_text() == generated
    assert "repair_candidate_overlay_base_skipped:" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("candidate", "expected_issue"),
    [
        (
            ValidationRetryCandidate(
                rulespec="format: rulespec/v1\nimports: [valid, 7]\nrules: []\n"
            ),
            "imports must be a list of strings",
        ),
        (
            ValidationRetryCandidate(
                rulespec=(
                    "format: rulespec/v1\nrules:\n"
                    "  - {name: duplicate, kind: parameter}\n"
                    "  - {name: duplicate, kind: parameter}\n"
                )
            ),
            "preserved rules contains duplicate name `duplicate`",
        ),
        (
            ValidationRetryCandidate(
                rulespec="format: rulespec/v1\nrules: {name: broken}\n"
            ),
            "preserved repair overlay rules must be a list",
        ),
        (
            ValidationRetryCandidate(
                rulespec="format: rulespec/v1\nrules: []\n",
                tests=(
                    "- {name: duplicate, period: '2026-01-01'}\n"
                    "- {name: duplicate, period: '2026-01-01'}\n"
                ),
            ),
            "preserved companion tests contains duplicate name `duplicate`",
        ),
        (
            ValidationRetryCandidate(
                rulespec="format: rulespec/v1\nrules: []\n",
                tests="cases: broken\n",
            ),
            "preserved companion test cases must be a list",
        ),
    ],
)
def test_retained_overlay_base_rejects_ambiguous_structures(candidate, expected_issue):
    issue = _repair_overlay_candidate_base_issue(candidate)
    assert issue is not None
    assert expected_issue in issue
