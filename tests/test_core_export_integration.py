"""Opt-in interoperability checks against an explicitly supplied real core CLI.

Set AXIOM_CORE_BIN to an executable axiom-core file to run these tests. They do
not discover, download, build, or replace core, so public encoder CI can skip
them without access to the core repository. All RuleSpec and inputs below are
synthetic software fixtures; Python only transports JSON and checks results.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from axiom_encode.core_export import build_spec_from_eval_result
from tests.core_eval_fixture import candidate_result

_REPO = Path(__file__).resolve().parents[1]
_ROOT = "zz:policies/demo"
_PARAMETERS = "zz:policies/parameters"
_OUTPUT = f"{_ROOT}#benefit"
_CANDIDATE = """# SYNTHETIC SOFTWARE FIXTURE — café. This does not encode law.
format: rulespec/v1
imports:
  - zz:policies/parameters#base_amount
rules:
  - name: benefit
    kind: derived
    entity: Household
    dtype: Money
    unit: USD
    period: Month
    rounding: half_up
    versions:
      - effective_from: 2026-01-01
        effective_to: 2026-12-31
        formula: max(0, base_amount - income * 0.3)
"""
_DEPENDENCY = """# SYNTHETIC SOFTWARE FIXTURE — 金額. These amounts do not encode law.
format: rulespec/v1
rules:
  - name: base_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: 2026-01-01
        effective_to: 2026-12-31
        formula: "200"
"""


def _command(args, *, stdin=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(_REPO / "src"), env.get("PYTHONPATH")))
    )
    return subprocess.run(
        [str(arg) for arg in args],
        cwd=_REPO,
        env=env,
        input=stdin,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


def _success(process):
    assert process.returncode == 0, (process.stdout, process.stderr)
    assert process.stderr == "", process.stderr
    return json.loads(process.stdout)


@pytest.fixture(scope="module")
def core_binary():
    configured = os.environ.get("AXIOM_CORE_BIN")
    if not configured:
        pytest.skip("AXIOM_CORE_BIN is unset; provide a real axiom-core executable")
    binary = Path(configured).expanduser().resolve()
    assert binary.is_file(), f"AXIOM_CORE_BIN is not a file: {binary}"
    assert os.access(binary, os.X_OK), f"AXIOM_CORE_BIN is not executable: {binary}"
    capabilities = _success(_command([binary, "capabilities"]))
    assert capabilities["format"] == "axiom/capabilities/v0"
    assert capabilities["engine"]["repository"] == (
        "https://github.com/TheAxiomFoundation/axiom-rules-engine"
    )
    assert (
        capabilities["engine"]["execution_host_sha256"]
        == hashlib.sha256(binary.read_bytes()).hexdigest()
    )
    return binary


@pytest.fixture
def sources(tmp_path):
    candidate = tmp_path / "synthetic-candidate.yaml"
    dependency = tmp_path / "synthetic-parameters.yaml"
    # write_bytes deliberately avoids universal-newline translation.
    candidate.write_bytes(_CANDIDATE.replace("\n", "\r\n").encode("utf-8"))
    dependency.write_bytes(_DEPENDENCY.replace("\n", "\r\n").encode("utf-8"))
    return candidate, dependency


def _export(candidate, modules, out):
    args = [
        sys.executable,
        "-m",
        "axiom_encode.entrypoint",
        "export-core-build-spec",
        "--root",
        _ROOT,
        "--candidate",
        candidate,
        "--expect-candidate-sha256",
        hashlib.sha256(candidate.read_bytes()).hexdigest(),
    ]
    for target, path in modules:
        args.extend(["--module", target, path])
    args.extend(["--out", out])
    exported = _success(_command(args))
    assert exported["ok"] is True
    assert exported["format"] == "axiom/build-spec/v0"
    assert exported["assurance"] == "unvalidated_candidate"
    assert (
        exported["candidate_sha256"]
        == hashlib.sha256(candidate.read_bytes()).hexdigest()
    )
    assert exported["build_spec_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert exported["module_count"] == 1 + len(modules)
    return json.loads(out.read_bytes())


def _build_and_verify(binary, spec, bundle):
    built = _success(_command([binary, "build", "--spec", spec, "--out", bundle]))
    assert built["ok"] is True
    assert built["assurance"] == "development_unsigned"
    verified = _success(
        _command(
            [binary, "verify", "--bundle", bundle, "--expect", built["bundle_sha256"]]
        )
    )
    assert verified == built
    return built


def _request(*, pin=False):
    request = {
        "mode": "explain",
        "dataset": {
            "inputs": [
                {
                    "name": f"{_ROOT}#input.income",
                    "entity": "Household",
                    "entity_id": "synthetic-household:1",
                    "interval": {"start": "2026-01-01", "end": "2026-01-31"},
                    "value": {"kind": "decimal", "value": "123.456"},
                }
            ]
        },
        "queries": [
            {
                "entity_id": "synthetic-household:1",
                "period": {
                    "period_kind": "month",
                    "start": "2026-01-01",
                    "end": "2026-01-31",
                },
                "outputs": [_OUTPUT],
            }
        ],
    }
    if pin:
        request["pins"] = [
            {"rule": "benefit", "value": {"kind": "decimal", "value": "0"}}
        ]
    return request


def _execute(binary, bundle, built, *, pin=False):
    receipt = _success(
        _command(
            [binary, "run", "--bundle", bundle, "--expect", built["bundle_sha256"]],
            stdin=json.dumps(_request(pin=pin)),
        )
    )
    assert receipt["format"] == "axiom/execution-receipt/v0"
    assert receipt["assurance"] == "development_unsigned"
    assert receipt["context"]["bundle_sha256"] == built["bundle_sha256"]
    assert receipt["context"]["artifact_sha256"] == built["artifact_sha256"]
    assert receipt["context"]["engine"] == built["engine"]
    assert receipt["result"]["metadata"] == {
        "requested_mode": "explain",
        "actual_mode": "explain",
        "fallback_reason": None,
    }
    return receipt


def _value(receipt):
    return receipt["result"]["results"][0]["outputs"][_OUTPUT]["value"]


def test_fragment_import_export_build_verify_and_native_trace(
    core_binary, sources, tmp_path
):
    candidate, dependency = sources
    spec, bundle = tmp_path / "build-spec.json", tmp_path / "bundle.json"
    _export(candidate, [(_PARAMETERS, dependency)], spec)
    built = _build_and_verify(core_binary, spec, bundle)

    receipt = _execute(core_binary, bundle, built)

    assert _value(receipt) == {"kind": "decimal", "value": "162.96"}
    result = receipt["result"]["results"][0]
    assert result["entity_id"] == "synthetic-household:1"
    trace = result["trace"][_OUTPUT]
    assert trace["rounding"] == "half_up"
    assert trace["pre_rounding_value"] == {"kind": "decimal", "value": "162.9632"}
    assert f"{_PARAMETERS}#base_amount" in trace["executed_expression"]
    assert trace["parameter_reads"][0]["id"] == f"{_PARAMETERS}#base_amount"
    assert trace["parameter_reads"][0]["value"] == {"kind": "integer", "value": 200}


def test_completed_eval_result_exports_to_real_core_with_native_trace(
    core_binary, sources, tmp_path
):
    candidate, dependency = sources
    result = candidate_result(candidate)
    original_result = result.to_dict()
    exported, metadata = build_spec_from_eval_result(
        result, root=_ROOT, modules=[(_PARAMETERS, dependency)]
    )
    spec, bundle = tmp_path / "eval-spec.json", tmp_path / "eval-bundle.json"
    spec.write_text(json.dumps(exported), encoding="utf-8")
    built = _build_and_verify(core_binary, spec, bundle)
    receipt = _execute(core_binary, bundle, built)

    assert metadata["candidate_sha256"] == result.generated_output_sha256
    assert metadata["assurance"] == "unvalidated_candidate"
    stored = json.loads(bundle.read_bytes())
    assert stored["modules"][_ROOT].encode() == candidate.read_bytes()
    assert _value(receipt) == {"kind": "decimal", "value": "162.96"}
    trace = receipt["result"]["results"][0]["trace"][_OUTPUT]
    assert trace["pre_rounding_value"] == {"kind": "decimal", "value": "162.9632"}
    assert trace["parameter_reads"][0]["id"] == f"{_PARAMETERS}#base_amount"
    assert result.to_dict() == original_result


def test_crlf_unicode_source_bytes_survive_export_and_core_bundle(
    core_binary, sources, tmp_path
):
    candidate, dependency = sources
    spec, bundle = tmp_path / "build-spec.json", tmp_path / "bundle.json"
    exported = _export(candidate, [(_PARAMETERS, dependency)], spec)
    _build_and_verify(core_binary, spec, bundle)
    stored = json.loads(bundle.read_bytes())

    assert set(exported) == {"format", "root", "modules"}
    assert exported["root"] == _ROOT
    assert set(exported["modules"]) == {_ROOT, _PARAMETERS}
    for target, source in ((_ROOT, candidate), (_PARAMETERS, dependency)):
        original = source.read_bytes()
        assert b"\r\n" in original
        assert any(byte > 127 for byte in original)
        assert exported["modules"][target].encode("utf-8") == original
        assert stored["modules"][target].encode("utf-8") == original
        assert (
            stored["manifest"]["source_hashes"][target]
            == hashlib.sha256(original).hexdigest()
        )


def test_missing_explicit_import_is_rejected_by_real_core(
    core_binary, sources, tmp_path
):
    candidate, _ = sources
    spec, bundle = tmp_path / "build-spec.json", tmp_path / "bundle.json"
    exported = _export(candidate, [], spec)
    assert set(exported["modules"]) == {_ROOT}

    failed = _command([core_binary, "build", "--spec", spec, "--out", bundle])

    assert failed.returncode != 0
    assert failed.stdout == ""
    error = json.loads(failed.stderr)
    assert error["ok"] is False
    assert error["error"]["code"] == "compile_error"
    assert _PARAMETERS in error["error"]["message"]
    assert not bundle.exists()


def test_unused_explicit_module_is_rejected_by_real_core(
    core_binary, sources, tmp_path
):
    candidate, dependency = sources
    unused = tmp_path / "synthetic-unused.yaml"
    unused.write_bytes(
        _DEPENDENCY.replace("base_amount", "unused_amount").encode("utf-8")
    )
    unused_target = "zz:policies/unused"
    spec, bundle = tmp_path / "build-spec.json", tmp_path / "bundle.json"
    _export(candidate, [(_PARAMETERS, dependency), (unused_target, unused)], spec)

    failed = _command([core_binary, "build", "--spec", spec, "--out", bundle])

    assert failed.returncode != 0
    assert failed.stdout == ""
    error = json.loads(failed.stderr)
    assert error["ok"] is False
    assert error["error"]["code"] == "unused_modules"
    assert unused_target in error["error"]["message"]
    assert not bundle.exists()


def test_dependency_change_changes_bundle_identity_and_native_pin_stays_zero(
    core_binary, sources, tmp_path
):
    candidate, dependency = sources
    spec, bundle = tmp_path / "build-spec.json", tmp_path / "bundle.json"
    _export(candidate, [(_PARAMETERS, dependency)], spec)
    built = _build_and_verify(core_binary, spec, bundle)
    stored_before = bundle.read_bytes()
    baseline = _execute(core_binary, bundle, built)
    pinned = _execute(core_binary, bundle, built, pin=True)
    assert _value(baseline) == {"kind": "decimal", "value": "162.96"}
    assert _value(pinned) == {"kind": "decimal", "value": "0"}
    assert baseline["context"]["scenario"] == {"pins": []}
    assert pinned["context"]["scenario"] == {"pins": _request(pin=True)["pins"]}
    assert (
        baseline["context"]["scenario_sha256"] != pinned["context"]["scenario_sha256"]
    )
    assert baseline["context_sha256"] != pinned["context_sha256"]
    assert bundle.read_bytes() == stored_before

    dependency.write_bytes(
        dependency.read_bytes().replace(b'formula: "200"', b'formula: "250"')
    )
    changed_spec, changed_bundle = (
        tmp_path / "changed-spec.json",
        tmp_path / "changed-bundle.json",
    )
    _export(candidate, [(_PARAMETERS, dependency)], changed_spec)
    changed = _build_and_verify(core_binary, changed_spec, changed_bundle)
    assert changed["bundle_sha256"] != built["bundle_sha256"]
    assert changed["artifact_sha256"] != built["artifact_sha256"]
    old_manifest = json.loads(stored_before)["manifest"]
    new_manifest = json.loads(changed_bundle.read_bytes())["manifest"]
    assert old_manifest["source_hashes"][_ROOT] == new_manifest["source_hashes"][_ROOT]
    assert (
        old_manifest["source_hashes"][_PARAMETERS]
        != new_manifest["source_hashes"][_PARAMETERS]
    )
    assert (
        old_manifest["source_closure_sha256"] != new_manifest["source_closure_sha256"]
    )
    assert _value(_execute(core_binary, changed_bundle, changed)) == {
        "kind": "decimal",
        "value": "212.96",
    }
    assert _value(_execute(core_binary, changed_bundle, changed, pin=True)) == {
        "kind": "decimal",
        "value": "0",
    }
    # Exporting a changed candidate closure cannot alter the stored first bundle.
    assert _value(_execute(core_binary, bundle, built)) == {
        "kind": "decimal",
        "value": "162.96",
    }
    assert bundle.read_bytes() == stored_before
