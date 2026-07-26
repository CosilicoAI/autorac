"""Focused tests for the signed-apply reusable workflow contract."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/signed-apply-reusable.yml"
COMPLETE_SOURCE_FLAG = "--require-complete-source-unit"


def _workflow_payload() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _signed_apply_script(enabled: bool) -> str:
    payload = _workflow_payload()
    script = next(
        step["run"]
        for step in payload["jobs"]["encode"]["steps"]
        if step.get("name") == "Signed re-encode under supervisor + leaf signer"
    )
    replacements = {
        "${{ inputs['require-complete-source-unit'] }}": str(enabled).lower(),
        "${{ github.event.repository.name }}": "rulespec-de",
        "${{ github.repository }}": "TheAxiomFoundation/rulespec-de",
        "${{ github.event.repository.default_branch }}": "main",
    }
    for expression, value in replacements.items():
        script = script.replace(expression, value)
    assert "${{" not in script
    return script


def test_complete_source_unit_reusable_input_is_optional_boolean_default_off():
    inputs = _workflow_payload()["on"]["workflow_call"]["inputs"]
    complete_source_input = inputs["require-complete-source-unit"]

    assert complete_source_input["required"] == "false"
    assert complete_source_input["type"] == "boolean"
    assert complete_source_input["default"] == "false"


@pytest.mark.parametrize(
    ("enabled", "expected_mode_args"),
    [
        (False, []),
        (True, [COMPLETE_SOURCE_FLAG]),
    ],
)
def test_complete_source_unit_input_composes_fixed_encoder_argv(
    tmp_path,
    enabled,
    expected_mode_args,
):
    launcher = tmp_path / "axiom-encode-apply-signer"
    launcher.write_text(
        '#!/bin/bash\nprintf \'%s\\0\' "$@" > "$RUNNER_TEMP/launcher-argv.bin"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    (tmp_path / "pyseg").write_text("python3.14\n", encoding="utf-8")

    script = _signed_apply_script(enabled)
    subprocess.run(
        ["/bin/bash", "-n"],
        input=script,
        text=True,
        check=True,
    )
    workspace = tmp_path / "rulespec-de"
    environment = {
        **os.environ,
        "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-only-key",
        "CITATION": "de/statute/estg/32a",
        "GITHUB_REPOSITORY": "TheAxiomFoundation/rulespec-de",
        "GITHUB_WORKSPACE": str(workspace),
        "RUNNER_TEMP": str(tmp_path),
    }
    subprocess.run(
        ["/bin/bash", "-c", script],
        env=environment,
        check=True,
    )

    raw_argv = (tmp_path / "launcher-argv.bin").read_bytes()
    launcher_argv = [
        field.decode() for field in raw_argv.removesuffix(b"\0").split(b"\0")
    ]
    separator = launcher_argv.index("--")
    encoder_argv = launcher_argv[separator + 1 :]
    assert encoder_argv == [
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
        *expected_mode_args,
    ]
    assert launcher_argv.count(COMPLETE_SOURCE_FLAG) == int(enabled)
