"""Focused tests for the signed-apply reusable workflow contract."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/signed-apply-reusable.yml"
LEGACY_RULES_ENGINE_REF = "05eac9d2f89dabe5c6673176260762cef3a58f47"
COMPLETE_SOURCE_FLAG = "--require-complete-source-unit"
MODEL_FLAG = "--model"
ESCALATE_FLAG = "--escalate-after"
EMIT_REJECTED_FLAG = "--emit-final-rejected-candidate"
REPAIR_FLAG_QUAD = (
    "--repair-candidate-root",
    "--repair-candidate-path",
    "--repair-candidate-rulespec-sha256",
    "--repair-candidate-tests-sha256",
)


def _workflow_payload() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _signed_apply_step() -> dict:
    return next(
        step
        for step in _workflow_payload()["jobs"]["encode"]["steps"]
        if step.get("name") == "Signed re-encode under supervisor + leaf signer"
    )


def _workflow_step(name: str) -> dict:
    return next(
        step
        for step in _workflow_payload()["jobs"]["encode"]["steps"]
        if step.get("name") == name
    )


def _dispatch_step(name: str) -> dict:
    return next(
        step
        for step in _workflow_payload()["jobs"]["dispatch"]["steps"]
        if step.get("name") == name
    )


def _signed_apply_script(*, require_complete_source_unit: bool = False) -> str:
    script = _signed_apply_step()["run"]
    replacements = {
        "${{ inputs['require-complete-source-unit'] }}": str(
            require_complete_source_unit
        ).lower(),
        "${{ github.event.repository.name }}": "rulespec-de",
        "${{ github.repository }}": "TheAxiomFoundation/rulespec-de",
        "${{ github.event.repository.default_branch }}": "main",
    }
    for expression, value in replacements.items():
        script = script.replace(expression, value)
    assert "${{" not in script
    return script


def test_rules_engine_pin_is_caller_selectable_and_preserves_existing_default():
    workflow = _workflow_payload()
    checkout = _workflow_step("Checkout axiom-rules-engine (pinned)")
    engine_input = workflow["on"]["workflow_call"]["inputs"]["axiom-rules-engine-ref"]

    assert engine_input == {
        "description": (
            "Immutable axiom-rules-engine commit SHA; keep aligned with the "
            "caller's validation workflow."
        ),
        "required": "false",
        "type": "string",
        "default": LEGACY_RULES_ENGINE_REF,
    }
    assert checkout["with"]["ref"] == ("${{ needs.dispatch.outputs.rules_engine_ref }}")


@pytest.mark.parametrize(
    "rules_engine_ref",
    [LEGACY_RULES_ENGINE_REF, "f" * 40, "0123456789abcdef" * 2 + "01234567"],
)
def test_rules_engine_ref_accepts_only_immutable_lowercase_shas(
    tmp_path, rules_engine_ref
):
    step = _dispatch_step("Validate axiom-rules-engine ref")
    output = tmp_path / "github-output"

    result = subprocess.run(
        ["/bin/bash", "-c", step["run"]],
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(output),
            "RULES_ENGINE_REF": rules_engine_ref,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"ref={rules_engine_ref}\n"


@pytest.mark.parametrize(
    "rules_engine_ref",
    ["", "main", "f" * 39, "f" * 41, "F" * 40, "g" * 40, "$(touch injected)"],
)
def test_rules_engine_ref_rejects_non_commit_refs(tmp_path, rules_engine_ref):
    step = _dispatch_step("Validate axiom-rules-engine ref")

    result = subprocess.run(
        ["/bin/bash", "-c", step["run"]],
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(tmp_path / "github-output"),
            "RULES_ENGINE_REF": rules_engine_ref,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "expected a lowercase 40-character commit SHA" in result.stderr
    assert not (tmp_path / "injected").exists()


def _base_encoder_argv(tmp_path: Path) -> list[str]:
    workspace = tmp_path / "rulespec-de"
    return [
        "/opt/axiom-verification/axiom-encode",
        "encode",
        "de/statute/estg/32a",
        "--apply",
        "--backend",
        "openai",
        "--corpus-path",
        f"{workspace}/_axiom/axiom-corpus",
        "--axiom-rules-engine-path",
        f"{workspace}/_axiom/axiom-rules-engine",
        "--policy-repo-path",
        f"{tmp_path}/genroot/rulespec-de",
        "--mode",
        "cold",
        "--skip-reviewers",
        "--db",
        f"{tmp_path}/enc.db",
        EMIT_REJECTED_FLAG,
        f"{tmp_path}/failed-encode",
    ]


def _base_launcher_argv(tmp_path: Path) -> list[str]:
    return [
        "run",
        "--scope",
        "apply_ed25519",
        "--key-env",
        "AXIOM_ENCODE_APPLY_SIGNING_KEY",
        "--supervisor",
        "/opt/axiom-verification/axiom-encode-signing-supervisor",
        "--trusted-signing-roots",
        "/opt/axiom-verification/signing-trust-roots.json",
        "--trusted-python-runtime-root",
        "/opt/axiom-verification/python",
        "--trusted-python-import-root",
        "/opt/axiom-verification/python/lib/python3.14/site-packages",
        "--trusted-python-package-root",
        "/opt/axiom-verification/python/lib/python3.14/site-packages/axiom_encode",
        "--expected-github-repository",
        "TheAxiomFoundation/rulespec-de",
        "--allowed-workflow-ref",
        "TheAxiomFoundation/rulespec-de/.github/workflows/"
        "signed-apply.yml@refs/heads/main",
        "--allowed-event-name",
        "workflow_dispatch",
        "--",
        *_base_encoder_argv(tmp_path),
    ]


def _run_signed_apply_script(
    tmp_path: Path,
    *,
    require_complete_source_unit: bool = False,
    initial_model: str = "",
    escalate_after: str = "",
    repair_candidate_root: str = "",
    repair_candidate_path: str = "",
    repair_candidate_rulespec_sha256: str = "",
    repair_candidate_tests_sha256: str = "",
) -> tuple[subprocess.CompletedProcess[str], bytes | None]:
    launcher = tmp_path / "axiom-encode-apply-signer"
    launcher.write_text(
        '#!/bin/bash\nprintf \'%s\\0\' "$@" > "$RUNNER_TEMP/launcher-argv.bin"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    (tmp_path / "pyseg").write_text("python3.14\n", encoding="utf-8")

    script = _signed_apply_script(
        require_complete_source_unit=require_complete_source_unit
    )
    subprocess.run(
        ["/bin/bash", "-n"],
        input=script,
        text=True,
        check=True,
    )
    workspace = tmp_path / "rulespec-de"
    result = subprocess.run(
        ["/bin/bash", "-c", script],
        env={
            **os.environ,
            "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-only-key",
            "CITATION": "de/statute/estg/32a",
            "ESCALATE_AFTER": escalate_after,
            "GITHUB_REPOSITORY": "TheAxiomFoundation/rulespec-de",
            "GITHUB_WORKSPACE": str(workspace),
            "INITIAL_MODEL": initial_model,
            "REPAIR_CANDIDATE_PATH": repair_candidate_path,
            "REPAIR_CANDIDATE_ROOT": repair_candidate_root,
            "REPAIR_CANDIDATE_RULESPEC_SHA256": repair_candidate_rulespec_sha256,
            "REPAIR_CANDIDATE_TESTS_SHA256": repair_candidate_tests_sha256,
            "RUNNER_TEMP": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    argv_file = tmp_path / "launcher-argv.bin"
    return result, argv_file.read_bytes() if argv_file.exists() else None


def _decode_argv(raw_argv: bytes) -> list[str]:
    return [field.decode() for field in raw_argv.removesuffix(b"\0").split(b"\0")]


def _encoder_argv(raw_argv: bytes) -> list[str]:
    launcher_argv = _decode_argv(raw_argv)
    return launcher_argv[launcher_argv.index("--") + 1 :]


def test_generation_budget_reusable_inputs_are_typed_and_default_off():
    inputs = _workflow_payload()["on"]["workflow_call"]["inputs"]

    assert inputs["require-complete-source-unit"] == {
        "description": "Require complete source-unit coverage during encoding.",
        "required": "false",
        "type": "boolean",
        "default": "false",
    }
    assert inputs["initial-model"] == {
        "description": "Optional allowlisted initial generation model.",
        "required": "false",
        "type": "string",
        "default": "",
    }
    assert inputs["escalate-after"] == {
        "description": "Optional generation attempt count before escalation (1-99).",
        "required": "false",
        "type": "string",
        "default": "",
    }
    assert inputs["repair-candidate-run-id"] == {
        "description": "Optional failed signed-apply run containing this leg's repair candidate.",
        "required": "false",
        "type": "string",
        "default": "",
    }


def test_generation_budget_inputs_enter_only_through_step_environment():
    step = _signed_apply_step()

    assert step["env"]["INITIAL_MODEL"] == "${{ inputs['initial-model'] }}"
    assert step["env"]["ESCALATE_AFTER"] == "${{ inputs['escalate-after'] }}"
    assert "${{ inputs['initial-model'] }}" not in step["run"]
    assert "${{ inputs['escalate-after'] }}" not in step["run"]


def test_repair_candidate_download_has_minimal_read_permission_and_exact_binding():
    workflow = _workflow_payload()
    encode_job = workflow["jobs"]["encode"]
    assert encode_job["permissions"] == {"actions": "read", "contents": "read"}

    validate = _workflow_step("Validate repair candidate run ID")
    assert validate["if"] == "${{ inputs['repair-candidate-run-id'] != '' }}"
    assert validate["env"]["REPAIR_CANDIDATE_RUN_ID"] == (
        "${{ inputs['repair-candidate-run-id'] }}"
    )
    assert "${{ inputs['repair-candidate-run-id'] }}" not in validate["run"]

    download = _workflow_step("Download failed encode candidate")
    assert download["if"] == "${{ inputs['repair-candidate-run-id'] != '' }}"
    assert download["with"] == {
        "name": "failed-encode-${{ matrix.item.slug }}",
        "path": "${{ runner.temp }}/repair-candidate",
        "run-id": "${{ steps.repair_run.outputs.run_id }}",
        "github-token": "${{ github.token }}",
    }
    verify = _workflow_step("Verify downloaded repair candidate")
    assert verify["env"] == {"CITATION": "${{ matrix.item.citation }}"}
    assert '--citation "$CITATION"' in verify["run"]


@pytest.mark.parametrize("run_id", ["0", "7", "001", "32062103227"])
def test_repair_candidate_run_id_accepts_only_numeric_strings(tmp_path, run_id):
    step = _workflow_step("Validate repair candidate run ID")
    output = tmp_path / "github-output"

    result = subprocess.run(
        ["/bin/bash", "-c", step["run"]],
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(output),
            "REPAIR_CANDIDATE_RUN_ID": run_id,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"run_id={run_id}\n"


@pytest.mark.parametrize(
    "run_id", ["", "-1", "+1", "1.0", "1 2", "12x", "$(touch injected)"]
)
def test_repair_candidate_run_id_rejects_non_numeric_strings(tmp_path, run_id):
    step = _workflow_step("Validate repair candidate run ID")

    result = subprocess.run(
        ["/bin/bash", "-c", step["run"]],
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(tmp_path / "github-output"),
            "REPAIR_CANDIDATE_RUN_ID": run_id,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "expected a numeric workflow run ID" in result.stderr
    assert not (tmp_path / "injected").exists()


def test_default_generation_budget_inputs_preserve_exact_launcher_argv(tmp_path):
    result, raw_argv = _run_signed_apply_script(tmp_path)

    assert result.returncode == 0, result.stderr
    expected_raw = b"".join(
        argument.encode() + b"\0" for argument in _base_launcher_argv(tmp_path)
    )
    assert raw_argv == expected_raw


def test_repair_candidate_absence_preserves_exact_default_launcher_argv(tmp_path):
    baseline, baseline_raw = _run_signed_apply_script(tmp_path)
    explicit_absent, absent_raw = _run_signed_apply_script(
        tmp_path,
        repair_candidate_root="",
        repair_candidate_path="",
        repair_candidate_rulespec_sha256="",
        repair_candidate_tests_sha256="",
    )

    assert baseline.returncode == explicit_absent.returncode == 0
    assert absent_raw == baseline_raw


def test_repair_candidate_composes_exact_flag_quad(tmp_path):
    digest_a = "a" * 64
    digest_b = "b" * 64
    root = str(tmp_path / "repair-candidate")
    path = "de/statutes/estg/66.yaml"

    result, raw_argv = _run_signed_apply_script(
        tmp_path,
        repair_candidate_root=root,
        repair_candidate_path=path,
        repair_candidate_rulespec_sha256=digest_a,
        repair_candidate_tests_sha256=digest_b,
    )

    assert result.returncode == 0, result.stderr
    assert raw_argv is not None
    encoder_argv = _encoder_argv(raw_argv)
    assert encoder_argv == [
        *_base_encoder_argv(tmp_path),
        "--repair-candidate-root",
        root,
        "--repair-candidate-path",
        path,
        "--repair-candidate-rulespec-sha256",
        digest_a,
        "--repair-candidate-tests-sha256",
        digest_b,
    ]
    for flag in REPAIR_FLAG_QUAD:
        assert encoder_argv.count(flag) == 1


def test_failed_candidate_upload_is_failure_only_and_exactly_preverified():
    verify = _workflow_step("Verify final rejected candidate artifact")
    upload = _workflow_step("Upload failed encode candidate")

    assert verify["if"] == "${{ failure() && !cancelled() }}"
    assert "verify_failed_encode_candidate.py" in verify["run"]
    assert upload["if"] == (
        "${{ failure() && !cancelled() "
        "&& steps.failed_candidate.outputs.present == 'true' }}"
    )
    assert upload["with"] == {
        "name": "failed-encode-${{ matrix.item.slug }}",
        "path": "${{ runner.temp }}/failed-encode",
        "if-no-files-found": "error",
        "retention-days": "30",
    }


@pytest.mark.parametrize(
    ("kwargs", "expected_tail"),
    [
        ({"require_complete_source_unit": True}, [COMPLETE_SOURCE_FLAG]),
        ({"initial_model": "gpt-5.6-terra"}, [MODEL_FLAG, "gpt-5.6-terra"]),
        ({"initial_model": "gpt-5.6-sol"}, [MODEL_FLAG, "gpt-5.6-sol"]),
        ({"escalate_after": "1"}, [ESCALATE_FLAG, "1"]),
        ({"escalate_after": "99"}, [ESCALATE_FLAG, "99"]),
        (
            {
                "require_complete_source_unit": True,
                "initial_model": "gpt-5.6-sol",
                "escalate_after": "7",
            },
            [
                COMPLETE_SOURCE_FLAG,
                MODEL_FLAG,
                "gpt-5.6-sol",
                ESCALATE_FLAG,
                "7",
            ],
        ),
    ],
)
def test_generation_budget_inputs_compose_exact_validated_literal_pairs(
    tmp_path,
    kwargs,
    expected_tail,
):
    result, raw_argv = _run_signed_apply_script(tmp_path, **kwargs)

    assert result.returncode == 0, result.stderr
    assert raw_argv is not None
    encoder_argv = _encoder_argv(raw_argv)
    assert encoder_argv == [*_base_encoder_argv(tmp_path), *expected_tail]
    assert encoder_argv.count(MODEL_FLAG) == expected_tail.count(MODEL_FLAG)
    assert encoder_argv.count(ESCALATE_FLAG) == expected_tail.count(ESCALATE_FLAG)


@pytest.mark.parametrize(
    ("kwargs", "expected_error"),
    [
        ({"initial_model": "gpt-5.6"}, "invalid initial-model"),
        (
            {"initial_model": "gpt-5.6-terra --escalate-after 1"},
            "invalid initial-model",
        ),
        ({"initial_model": "gpt-5.6-terra; touch injected"}, "invalid initial-model"),
        ({"escalate_after": "0"}, "invalid escalate-after"),
        ({"escalate_after": "00"}, "invalid escalate-after"),
        ({"escalate_after": "100"}, "invalid escalate-after"),
        ({"escalate_after": "-1"}, "invalid escalate-after"),
        ({"escalate_after": "1 --model gpt-5.6-sol"}, "invalid escalate-after"),
    ],
)
def test_generation_budget_inputs_fail_closed_before_launcher_exec(
    tmp_path,
    kwargs,
    expected_error,
):
    result, raw_argv = _run_signed_apply_script(tmp_path, **kwargs)

    assert result.returncode == 1
    assert expected_error in result.stderr
    assert raw_argv is None
    assert not (tmp_path / "injected").exists()
