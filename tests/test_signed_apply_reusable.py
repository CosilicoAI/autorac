"""Focused tests for the signed-apply reusable workflow contract."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/signed-apply-reusable.yml"
COMPLETE_SOURCE_FLAG = "--require-complete-source-unit"
MODEL_FLAG = "--model"
ESCALATE_FLAG = "--escalate-after"


def _workflow_payload() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _signed_apply_step() -> dict:
    return next(
        step
        for step in _workflow_payload()["jobs"]["encode"]["steps"]
        if step.get("name") == "Signed re-encode under supervisor + leaf signer"
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
        "--no-sync",
        "--skip-reviewers",
        "--db",
        f"{tmp_path}/enc.db",
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


def test_generation_budget_inputs_enter_only_through_step_environment():
    step = _signed_apply_step()

    assert step["env"]["INITIAL_MODEL"] == "${{ inputs['initial-model'] }}"
    assert step["env"]["ESCALATE_AFTER"] == "${{ inputs['escalate-after'] }}"
    assert "${{ inputs['initial-model'] }}" not in step["run"]
    assert "${{ inputs['escalate-after'] }}" not in step["run"]


def test_default_generation_budget_inputs_preserve_exact_launcher_argv(tmp_path):
    result, raw_argv = _run_signed_apply_script(tmp_path)

    assert result.returncode == 0, result.stderr
    expected_raw = b"".join(
        argument.encode() + b"\0" for argument in _base_launcher_argv(tmp_path)
    )
    assert raw_argv == expected_raw


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
