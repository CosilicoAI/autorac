"""Focused opt-in complete-source-unit mode plumbing tests."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode import cli
from axiom_encode.harness import evals
from axiom_encode.harness.evals import EvalWorkspace
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from axiom_encode.prompts.encoder import get_encoder_prompt

_REQUIRED_PATH_ARGS = [
    "--corpus-path",
    "/tmp/axiom-corpus",
    "--axiom-rules-engine-path",
    "/tmp/axiom-rules-engine",
]


@pytest.mark.parametrize(
    ("command_args", "handler"),
    [
        (["validate", "fixture.yaml", *_REQUIRED_PATH_ARGS], "cmd_validate"),
        (
            [
                "encode",
                "us/statute/26/32a",
                *_REQUIRED_PATH_ARGS,
                "--policy-repo-path",
                "/tmp/rulespec-us",
            ],
            "cmd_encode",
        ),
        (
            [
                "eval",
                "us/statute/26/32a",
                *_REQUIRED_PATH_ARGS,
                "--policy-repo-path",
                "/tmp/rulespec-us",
            ],
            "cmd_eval",
        ),
        (
            [
                "eval-source",
                "de/statute/estg/32a",
                *_REQUIRED_PATH_ARGS,
                "--policy-repo-path",
                "/tmp/rulespec-de",
            ],
            "cmd_eval_source",
        ),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_complete_source_unit_flag_is_default_off_and_opt_in(
    command_args,
    handler,
    enabled,
):
    argv = ["axiom-encode", *command_args]
    if enabled:
        argv.append("--require-complete-source-unit")

    with patch("sys.argv", argv), patch.object(cli, handler) as command:
        cli.main()

    parsed = command.call_args.args[0]
    assert parsed.require_complete_source_unit is enabled


def test_generic_encoder_prompt_adds_completeness_only_when_enabled():
    kwargs = {
        "citation": "de/statute/estg/32a",
        "output_path": "de/statutes/estg/32a.yaml",
        "corpus_citation_path": "de/statute/estg/32a",
    }

    default_prompt = get_encoder_prompt(**kwargs)
    explicit_off_prompt = get_encoder_prompt(
        **kwargs,
        require_complete_source_unit=False,
    )
    complete_prompt = get_encoder_prompt(
        **kwargs,
        require_complete_source_unit=True,
    )

    assert explicit_off_prompt == default_prompt
    assert "Complete-source-unit mode is enabled" not in default_prompt
    assert "Complete-source-unit mode is enabled" in complete_prompt
    assert (
        "parameters without encoding the stated formula is invalid" in complete_prompt
    )
    assert "declare those facts as explicit local RuleSpec\n  inputs" in complete_prompt
    assert (
        "encode both\n  values as separate grounded `kind: parameter` rules"
        in complete_prompt
    )
    assert "calendar constants `12`, `52`, `365`, `4`, or `24`" in complete_prompt
    assert "companion-test assertions on both\n  parameter outputs" in complete_prompt
    assert "separate grounded `kind: parameter` rules" not in default_prompt
    assert "missing dependency or citation" in complete_prompt
    assert "the `reason` itself must literally cite the complete legal branch" in (
        complete_prompt
    )
    assert "The output path\n  is not a source citation" in complete_prompt
    assert (
        "authoritative branch is exactly a bare `Repealed.` tombstone"
        in complete_prompt
    )
    assert "Do not\n  fabricate an executable rule" in complete_prompt
    assert (
        "the bounded reason form `<exact branch citation> is repealed.`"
        in complete_prompt
    )
    assert (
        "For a Louisiana branch that computes tax at rates provided" in complete_prompt
    )
    assert "exact missing RuleSpec targets under `blocked_by`" not in complete_prompt
    assert (
        "Only include `blocked_by` entries when you know the exact" in complete_prompt
    )
    assert (
        "formula branch, boundary,\n  exception, and rounding rule" in complete_prompt
    )
    assert "child-branch `source:` citation does not cover a distinct parent" in (
        complete_prompt
    )
    assert "exact parent-chapeau proof excerpt" in complete_prompt
    assert "Do not invent a dummy output for a chapeau" in complete_prompt
    assert "Every emitted derived `dtype: Judgment` output" in complete_prompt
    assert "source-faithful\n  Judgment whose versions are all constant false" in (
        complete_prompt
    )
    assert "testing only the negative state is incomplete" in complete_prompt
    assert "historical branch's runtime evidence must use" in complete_prompt
    assert "not unambiguously bound by a canonical\n  structural source path" in (
        complete_prompt
    )
    assert "assert every reached local derived dependency's expected" in (
        complete_prompt
    )
    assert "mandatory when multiple\n  computations share" in complete_prompt
    assert "citation-only proof atom" in complete_prompt
    assert "keep that output executable from the earliest" in complete_prompt
    assert "parameter/helper guards in the\n  single derived formula" in complete_prompt
    assert "omit oracle inputs or expectations" in complete_prompt
    assert "scalar-only source unit may remain parameter-only" in complete_prompt
    assert "In Armenian ARLIS text" in default_prompt
    assert "only a parenthetical on\n  its own line" in default_prompt
    assert "single ordinal-numbered article label" in default_prompt
    assert "date immediately followed by" in default_prompt
    assert "Keep every inline, plural or unnumbered" in default_prompt
    assert "unpaired-date, nested, or otherwise unrecognized" in default_prompt
    assert "otherwise lists only amendment dates" not in default_prompt
    assert "parenthetical that states an operative amount" in complete_prompt
    assert "single-principal-output case pairs" in complete_prompt
    assert "one large omnibus case" in complete_prompt
    assert "Build those boolean-gate witnesses mechanically" in complete_prompt
    assert "plus every reached local derived dependency required for" in complete_prompt
    assert "identical input-key and output-key sets" in complete_prompt


def test_validator_pipeline_complete_source_mode_is_default_off(tmp_path):
    kwargs = {
        "policy_repo_path": tmp_path / "rulespec-de",
        "axiom_rules_path": tmp_path / "axiom-rules-engine",
        "local_corpus_release": None,
        "enable_oracles": False,
    }

    default_pipeline = ValidatorPipeline(**kwargs)
    complete_pipeline = ValidatorPipeline(
        **kwargs,
        require_complete_source_unit=True,
    )

    assert default_pipeline.require_complete_source_unit is False
    assert complete_pipeline.require_complete_source_unit is True


def test_pipeline_complete_mode_uses_resolver_body_not_summary(tmp_path):
    content = """\
format: rulespec/v1
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
    authoritative_body = (
        "(1) Der Freibetrag beträgt 259 Euro; der Zuschlag beträgt 73 Euro."
    )
    kwargs = {
        "policy_repo_path": tmp_path / "rulespec-de",
        "axiom_rules_path": tmp_path / "axiom-rules-engine",
        "local_corpus_release": None,
        "enable_oracles": False,
    }
    default_pipeline = ValidatorPipeline(**kwargs)
    complete_pipeline = ValidatorPipeline(
        **kwargs,
        require_complete_source_unit=True,
    )
    source_texts = {"de/statute/estg/32a": authoritative_body}

    assert (
        default_pipeline._complete_source_unit_issues(
            content,
            validation_source_texts=source_texts,
            test_cases=[],
        )
        == []
    )
    strict_issues = complete_pipeline._complete_source_unit_issues(
        content,
        validation_source_texts=source_texts,
        test_cases=[],
    )

    assert any(
        "authoritative corpus numeric value 73" in issue.lower()
        for issue in strict_issues
    )
    assert not any(
        "authoritative corpus numeric value 259" in issue.lower()
        for issue in strict_issues
    )


def test_eval_prompt_adds_completeness_only_when_enabled(tmp_path):
    source_file = tmp_path / "source.txt"
    source_file.write_text("The amount is 12.")
    workspace = EvalWorkspace(
        root=tmp_path,
        source_text_file=source_file,
        manifest_file=tmp_path / "context-manifest.json",
    )
    kwargs = {
        "citation": "de/statute/example/1",
        "mode": "cold",
        "workspace": workspace,
        "context_files": [],
        "target_file_name": "1.yaml",
        "include_tests": True,
        "runner_backend": "openai",
    }

    default_prompt = evals._build_eval_prompt(**kwargs)
    explicit_off_prompt = evals._build_eval_prompt(
        **kwargs,
        require_complete_source_unit=False,
    )
    complete_prompt = evals._build_eval_prompt(
        **kwargs,
        require_complete_source_unit=True,
    )

    assert explicit_off_prompt == default_prompt
    assert "Complete-source-unit mode is enabled" not in default_prompt
    assert "Complete-source-unit mode is enabled" in complete_prompt
    assert "authoritative `./source.txt` body" in complete_prompt
    assert (
        "parameters without encoding the stated formula is invalid" in complete_prompt
    )
    assert "declare those facts as explicit local RuleSpec\n  inputs" in complete_prompt
    assert (
        "encode both\n  values as separate grounded `kind: parameter` rules"
        in complete_prompt
    )
    assert "calendar constants `12`, `52`, `365`, `4`, or `24`" in complete_prompt
    assert "companion-test assertions on both\n  parameter outputs" in complete_prompt
    assert "separate grounded `kind: parameter` rules" not in default_prompt
    assert "exact missing RuleSpec targets under `blocked_by`" not in complete_prompt
    assert "the `reason` itself must literally cite the complete legal branch" in (
        complete_prompt
    )
    assert "The output path\n  is not a source citation" in complete_prompt
    assert (
        "authoritative branch is exactly a bare `Repealed.` tombstone"
        in complete_prompt
    )
    assert (
        "the bounded reason form `<exact branch citation> is repealed.`"
        in complete_prompt
    )
    assert (
        "For a Louisiana branch that computes tax at rates provided" in complete_prompt
    )
    assert (
        "Only include `blocked_by` entries when you know the exact" in complete_prompt
    )
    assert "source-driven\n  coverage rule overrides the 1-4 case default" in (
        complete_prompt
    )
    assert "one large omnibus case" in complete_prompt


def test_eval_prompt_adds_prior_validation_feedback_only_when_supplied(tmp_path):
    source_file = tmp_path / "source.txt"
    source_file.write_text("The annual amount is 73 800.")
    workspace = EvalWorkspace(
        root=tmp_path,
        source_text_file=source_file,
        manifest_file=tmp_path / "context-manifest.json",
    )
    kwargs = {
        "citation": "de/regulation/example/1",
        "mode": "cold",
        "workspace": workspace,
        "context_files": [],
        "target_file_name": "1.yaml",
        "include_tests": True,
        "runner_backend": "openai",
    }
    feedback = (
        "Ungrounded generated numeric literal: 12 does not appear as a "
        "substantive numeric value in the source text. Complete-source "
        "stated-conversion hint: encode separate grounded parameters.",
    )
    candidate = evals.ValidationRetryCandidate(
        rulespec="format: rulespec/v1\nrules: []\n",
        tests="[]\n",
    )

    default_prompt = evals._build_eval_prompt(**kwargs)
    retry_prompt = evals._build_eval_prompt(
        **kwargs,
        validation_retry_feedback=feedback,
        validation_retry_candidate=candidate,
    )

    assert "PRIOR VALIDATION FEEDBACK" not in default_prompt
    assert "PRIOR VALIDATION FEEDBACK" in retry_prompt
    assert "repair guidance from the validator, not legal authority" in retry_prompt
    assert "feedback for the rejected candidate below" in retry_prompt
    assert feedback[0] in retry_prompt


def test_eval_prompt_preserves_compound_validation_feedback_tail(tmp_path):
    source_file = tmp_path / "source.txt"
    source_file.write_text("The annual amount is 73 800.")
    workspace = EvalWorkspace(
        root=tmp_path,
        source_text_file=source_file,
        manifest_file=tmp_path / "context-manifest.json",
    )
    final_control = "FINAL_REQUIRED_CONTROL_PAIR_MUST_REACH_MODEL"
    feedback = ("missing controls: " + "x" * 5_000 + final_control,)
    candidate = evals.ValidationRetryCandidate(
        rulespec="format: rulespec/v1\nrules: []\n",
        tests="[]\n",
    )

    prompt = evals._build_eval_prompt(
        citation="de/regulation/example/1",
        mode="cold",
        workspace=workspace,
        context_files=[],
        target_file_name="1.yaml",
        include_tests=True,
        runner_backend="openai",
        validation_retry_feedback=feedback,
        validation_retry_candidate=candidate,
    )

    assert feedback[0] in prompt
    assert final_control in prompt


def test_eval_prompt_rejects_retry_feedback_without_matching_candidate(tmp_path):
    source_file = tmp_path / "source.txt"
    source_file.write_text("The annual amount is 73 800.")
    workspace = EvalWorkspace(
        root=tmp_path,
        source_text_file=source_file,
        manifest_file=tmp_path / "context-manifest.json",
    )

    with pytest.raises(
        ValueError,
        match="Validation retry feedback requires its matching rejected candidate",
    ):
        evals._build_eval_prompt(
            citation="de/regulation/example/1",
            mode="cold",
            workspace=workspace,
            context_files=[],
            target_file_name="1.yaml",
            include_tests=True,
            runner_backend="openai",
            validation_retry_feedback=("candidate-specific issue",),
        )


def test_run_model_eval_forces_tests_and_forwards_complete_mode(tmp_path):
    source_unit = object()
    result = object()
    with (
        patch.object(evals, "_validate_eval_oracle_runtime"),
        patch.object(evals, "resolve_corpus_source_unit", return_value=source_unit),
        patch.object(
            evals,
            "_authoritative_rulespec_dependency_scope",
            return_value=nullcontext(),
        ),
        patch.object(evals, "_run_single_eval", return_value=result) as run_single,
    ):
        actual = evals.run_model_eval(
            citations=["de/statute/estg/32a"],
            runner_specs=["codex:test-model"],
            output_root=tmp_path / "output",
            policy_path=tmp_path / "rulespec-de",
            runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
            corpus_release=object(),
            include_tests=False,
            require_complete_source_unit=True,
            validation_retry_feedback=("prior validator issue",),
        )

    assert actual == [result]
    assert run_single.call_args.kwargs["include_tests"] is True
    assert run_single.call_args.kwargs["require_complete_source_unit"] is True
    assert run_single.call_args.kwargs["validation_retry_feedback"] == (
        "prior validator issue",
    )


def test_repair_revalidation_keeps_complete_mode(tmp_path):
    metrics = SimpleNamespace(ci_issues=["repairable"])
    with (
        patch.object(evals, "evaluate_artifact", return_value=metrics) as evaluate,
        patch.object(
            evals,
            "_apply_generated_eval_repairs",
            return_value=["companion-test-repair"],
        ),
    ):
        actual = evals._evaluate_generated_artifact_with_repairs(
            rulespec_file=tmp_path / "artifact.yaml",
            policy_repo_root=tmp_path / "rulespec-de",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            source_text="Source body",
            local_corpus_release=object(),
            require_complete_source_unit=True,
        )

    assert actual is metrics
    assert evaluate.call_count == 2
    assert all(
        call.kwargs["require_complete_source_unit"] is True
        for call in evaluate.call_args_list
    )
