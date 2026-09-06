"""Unvalidated candidate export: bytes, bounds, and console dispatch only.

These tests do not compile RuleSpec or substitute a calculation implementation.
The separate core integration suite exercises the real Rust compiler.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from axiom_encode import core_export, entrypoint, signing_broker

ROOT = "zz:policies/synthetic-candidate"
IMPORTED = "zz:policies/synthetic-import"
LIMIT = 16 * 1024 * 1024


def _file(
    tmp_path: Path, name: str = "candidate.yaml", data: bytes = b"# SYNTHETIC\n"
) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _args(candidate: Path, output: Path, *modules: tuple[str, Path]) -> list[str]:
    argv = ["--root", ROOT, "--candidate", str(candidate), "--out", str(output)]
    for target, path in modules:
        argv.extend(["--module", target, str(path)])
    return argv


def _rejected(capsys, candidate: Path, output: Path, *modules, extra=()) -> dict:
    result = core_export.run_export_core_build_spec(
        [*_args(candidate, output, *modules), *extra]
    )
    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["ok"] is False
    assert isinstance(error["error"]["code"], str) and error["error"]["code"]
    assert isinstance(error["error"]["message"], str) and error["error"]["message"]
    assert not output.exists(), "validation must finish before output creation"
    return error


def _isolated_python(tmp_path: Path, script: str, *args: str, env=None):
    source = Path(__file__).resolve().parents[1] / "src"
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys; sys.path.insert(0, {str(source)!r});\n{script}",
            *args,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_library_preserves_exact_utf8_crlf_and_unparsed_candidate_bytes(tmp_path):
    # Deliberately malformed YAML: export must not parse or normalize it.
    raw = "# SYNTHETIC café 雪\r\nnot_yaml: [\r\n".encode()
    imported = b"# SYNTHETIC imported\r\nno newline"
    candidate = _file(tmp_path, data=raw)
    module = _file(tmp_path, "import.yaml", imported)
    candidate_sha = hashlib.sha256(raw).hexdigest()

    spec, metadata = core_export.build_spec(
        ROOT,
        candidate,
        [(IMPORTED, module)],
        expected_candidate_sha256=candidate_sha,
    )

    assert spec == {
        "format": "axiom/build-spec/v0",
        "root": ROOT,
        "modules": {ROOT: raw.decode(), IMPORTED: imported.decode()},
    }
    assert spec["modules"][ROOT].encode() == raw
    assert spec["modules"][IMPORTED].encode() == imported
    assert metadata["candidate_sha256"] == candidate_sha
    assert candidate.read_bytes() == raw
    assert module.read_bytes() == imported


def test_cli_emits_unvalidated_identity_for_exact_output_bytes(tmp_path, capsys):
    raw = "# SYNTHETIC Unicode λ\r\n".encode()
    candidate = _file(tmp_path, data=raw)
    module = _file(tmp_path, "import.yaml")
    output = tmp_path / "build-spec.json"

    assert (
        core_export.run_export_core_build_spec(
            _args(candidate, output, (IMPORTED, module))
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    result = json.loads(captured.out)
    assert result["ok"] is True
    assert result["format"] == "axiom/build-spec/v0"
    assert result["assurance"] == "unvalidated_candidate"
    assert result["candidate_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (
        result["build_spec_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    assert result["module_count"] == 2
    assert result["output"] == str(output)
    spec = json.loads(output.read_bytes())
    assert set(spec) == {"format", "root", "modules"}
    assert spec["modules"][ROOT].encode() == raw


@pytest.mark.parametrize(
    "assignment", ["root", "repeat_same_file", "conflicting_files"]
)
def test_duplicate_target_assignments_are_rejected(tmp_path, capsys, assignment):
    candidate = _file(tmp_path)
    module = _file(tmp_path, "import.yaml", b"# SYNTHETIC other\n")
    modules = {
        "root": [(ROOT, module)],
        "repeat_same_file": [(IMPORTED, module), (IMPORTED, module)],
        "conflicting_files": [(IMPORTED, candidate), (IMPORTED, module)],
    }[assignment]
    error = _rejected(capsys, candidate, tmp_path / "out.json", *modules)
    assert error["error"]["code"] == "duplicate_module"


@pytest.mark.parametrize("target", ["", "  ", "zz:policies/\udcff"])
def test_invalid_target_returns_structured_error_before_output(
    tmp_path, capsys, target
):
    candidate = _file(tmp_path)
    output = tmp_path / "out.json"
    error = _rejected(capsys, candidate, output, (target, candidate))
    assert error["error"]["code"] == "invalid_target"


@pytest.mark.parametrize("digest_kind", ["short", "uppercase", "nonhex", "mismatch"])
def test_expected_candidate_digest_is_strict_and_checked(tmp_path, capsys, digest_kind):
    candidate = _file(tmp_path)
    expected = hashlib.sha256(candidate.read_bytes()).hexdigest()
    supplied = {
        "short": expected[:-1],
        "uppercase": expected.upper(),
        "nonhex": "g" * 64,
        "mismatch": "0" * 64,
    }[digest_kind]
    error = _rejected(
        capsys,
        candidate,
        tmp_path / "out.json",
        extra=("--expect-candidate-sha256", supplied),
    )
    assert error["error"]["code"] == (
        "candidate_digest_mismatch" if digest_kind == "mismatch" else "invalid_digest"
    )


@pytest.mark.parametrize("invalid_target", ["candidate", "module"])
def test_invalid_utf8_is_rejected_before_output(tmp_path, capsys, invalid_target):
    candidate = _file(tmp_path)
    module = _file(tmp_path, "import.yaml")
    (candidate if invalid_target == "candidate" else module).write_bytes(
        b"# SYNTHETIC\n\xff"
    )
    error = _rejected(capsys, candidate, tmp_path / "out.json", (IMPORTED, module))
    assert error["error"]["code"] == "invalid_utf8"


@pytest.mark.parametrize("kind", ["directory", "symlink"])
@pytest.mark.parametrize("invalid_target", ["candidate", "module"])
def test_inputs_require_regular_files_without_final_symlinks(
    tmp_path, capsys, kind, invalid_target
):
    candidate = _file(tmp_path)
    module = _file(tmp_path, "import.yaml")
    invalid = tmp_path / "invalid.yaml"
    if kind == "directory":
        invalid.mkdir()
    else:
        invalid.symlink_to(candidate)
    if invalid_target == "candidate":
        candidate = invalid
    else:
        module = invalid
    _rejected(capsys, candidate, tmp_path / "out.json", (IMPORTED, module))


def test_fifo_is_rejected_without_blocking(tmp_path):
    candidate = tmp_path / "pipe.yaml"
    os.mkfifo(candidate)
    output = tmp_path / "out.json"
    completed = _isolated_python(
        tmp_path,
        "from axiom_encode.entrypoint import main; raise SystemExit(main())",
        "export-core-build-spec",
        *_args(candidate, output),
        env=signing_broker.scrub_private_signing_environment(),
    )
    assert completed.returncode == 1, completed.stderr
    assert json.loads(completed.stderr)["ok"] is False
    assert json.loads(completed.stderr)["error"]["code"] == "unsafe_input"
    assert not output.exists()


def test_per_file_byte_limit_rejects_before_output_creation(tmp_path, capsys):
    candidate = _file(tmp_path, data=b"x" * (LIMIT + 1))
    error = _rejected(capsys, candidate, tmp_path / "out.json")
    assert error["error"]["code"] == "input_too_large"


def test_serialized_build_spec_limit_includes_json_overhead(tmp_path, capsys):
    candidate = _file(tmp_path, data=b"x" * (LIMIT // 2))
    module = _file(tmp_path, "import.yaml", b"x" * (LIMIT // 2))
    # Each source is individually valid and raw bytes sum to the limit;
    # serializing format, root, targets, and JSON delimiters exceeds it.
    error = _rejected(capsys, candidate, tmp_path / "out.json", (IMPORTED, module))
    assert error["error"]["code"] == "build_spec_too_large"


@pytest.mark.parametrize("kind", ["regular_file", "symlink"])
def test_output_is_exclusively_created_and_existing_bytes_survive(
    tmp_path, capsys, kind
):
    candidate = _file(tmp_path)
    protected = _file(
        tmp_path, "protected.json", b"existing output must survive\x00\r\n"
    )
    output = protected
    if kind == "symlink":
        output = tmp_path / "out.json"
        output.symlink_to(protected)
    original = output.read_bytes()
    assert core_export.run_export_core_build_spec(_args(candidate, output)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["ok"] is False
    assert output.read_bytes() == original
    assert protected.read_bytes() == original
    assert output.is_symlink() == (kind == "symlink")


def test_exporter_import_does_not_load_main_cli(tmp_path):
    completed = _isolated_python(
        tmp_path,
        "import axiom_encode.core_export\nassert 'axiom_encode.cli' not in sys.modules",
    )
    assert completed.returncode == 0, completed.stderr


def test_export_module_itself_does_not_import_evaluation_harness(tmp_path):
    # The package initializer already eagerly exports EvalResult. Load the
    # actual leaf module alone to check that its annotation adds no runtime
    # harness/CLI import; this does not change package initialization behavior.
    completed = _isolated_python(
        tmp_path,
        f"import runpy; runpy.run_path({core_export.__file__!r})\n"
        "assert 'axiom_encode.harness.evals' not in sys.modules\n"
        "assert 'axiom_encode.cli' not in sys.modules",
    )
    assert completed.returncode == 0, completed.stderr


def test_entrypoint_runs_broker_checks_before_export_dispatch(monkeypatch):
    events = []
    argv = ["--root", ROOT, "--candidate", "candidate.yaml", "--out", "out.json"]
    monkeypatch.setattr(
        signing_broker,
        "reject_direct_private_signing_environment",
        lambda: events.append("reject"),
    )
    monkeypatch.setattr(
        signing_broker,
        "attach_signing_broker_from_environment",
        lambda: events.append("attach"),
    )

    def dispatch(args):
        events.append(("export", args))
        return 19

    monkeypatch.setattr(core_export, "run_export_core_build_spec", dispatch)
    monkeypatch.setattr(sys, "argv", ["axiom-encode", "export-core-build-spec", *argv])
    assert entrypoint.main() == 19
    assert events == ["reject", "attach", ("export", argv)]


def test_entrypoint_private_environment_rejects_before_export(tmp_path):
    candidate = _file(tmp_path)
    output = tmp_path / "out.json"
    environment = signing_broker.scrub_private_signing_environment()
    environment[signing_broker.APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV] = (
        "test-only-must-not-reach-export"
    )
    completed = _isolated_python(
        tmp_path,
        "from axiom_encode.entrypoint import main; raise SystemExit(main())",
        "export-core-build-spec",
        *_args(candidate, output),
        env=environment,
    )
    assert completed.returncode != 0
    assert "externally provisioned signing broker" in completed.stderr
    assert completed.stdout == ""
    assert not output.exists()
