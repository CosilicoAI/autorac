import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from axiom_encode import cli
from axiom_encode.engine_binding import (
    ENGINE_PIN_FIELD,
    EngineBindingError,
    EnginePin,
    engine_binding_receipt_path,
    load_declared_engine_pin,
    read_engine_binding_receipt,
    resolve_pinned_engine_binary,
    write_engine_binding_receipt,
)

_ZERO_SHA256 = "0" * 64


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _engine_checkout(tmp_path: Path) -> tuple[Path, str]:
    """Create a committed engine checkout; return (checkout, pinned HEAD)."""

    checkout = tmp_path / "axiom-rules-engine"
    checkout.mkdir()
    _git(checkout, "init", "-q")
    _git(checkout, "config", "user.email", "t@t")
    _git(checkout, "config", "user.name", "t")
    (checkout / ".gitignore").write_text("target/\n")
    (checkout / "src.rs").write_text("fn main() {}\n")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "pinned")
    return checkout, _git(checkout, "rev-parse", "HEAD")


def _drift_commit(checkout: Path) -> str:
    """Advance HEAD one commit past the pin; return the new HEAD."""

    (checkout / "src.rs").write_text("fn main() { /* drift */ }\n")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "drift")
    return _git(checkout, "rev-parse", "HEAD")


def _stale_debug_binary(checkout: Path) -> Path:
    debug = checkout / "target" / "debug" / "axiom-rules-engine"
    debug.parent.mkdir(parents=True, exist_ok=True)
    debug.write_bytes(b"stale debug build")
    debug.chmod(0o755)
    return debug


def _cargo_shim(tmp_path: Path, monkeypatch) -> Path:
    """Put a fake cargo on PATH that writes a stub release binary and exits 0."""

    shim_dir = tmp_path / "cargo-shim"
    shim_dir.mkdir(exist_ok=True)
    shim = shim_dir / "cargo"
    shim.write_text(
        "#!/bin/sh\n"
        "mkdir -p target/release\n"
        'printf "release-built-from-pin" > target/release/axiom-rules-engine\n'
        "chmod +x target/release/axiom-rules-engine\n"
        "exit 0\n"
    )
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ['PATH']}")
    return shim


def _write_toolchain(checkout_root: Path, *, engine_ref: str | None) -> Path:
    config_dir = checkout_root / ".axiom"
    config_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "[toolchain]",
        'axiom_corpus_release = "test-release"',
        f'axiom_corpus_release_content_sha256 = "{_ZERO_SHA256}"',
        f'validation_waiver_set_sha256 = "{_ZERO_SHA256}"',
    ]
    if engine_ref is not None:
        lines.append(f'{ENGINE_PIN_FIELD} = "{engine_ref}"')
    config_path = config_dir / "toolchain.toml"
    config_path.write_text("\n".join(lines) + "\n")
    return config_path


def test_stale_debug_is_not_bound_and_release_is_rebuilt_at_pin(tmp_path, monkeypatch):
    checkout, pin_sha = _engine_checkout(tmp_path)
    stale = _stale_debug_binary(checkout)
    _cargo_shim(tmp_path, monkeypatch)
    pin = EnginePin(sha=pin_sha, source=tmp_path / "toolchain.toml")

    binary = resolve_pinned_engine_binary(checkout, pin)

    assert binary == checkout / "target" / "release" / "axiom-rules-engine"
    assert binary.read_bytes() == b"release-built-from-pin"
    receipt = read_engine_binding_receipt(binary)
    assert receipt is not None
    assert receipt["engine_commit"] == pin_sha
    assert receipt["cargo_profile"] == "release"
    assert (
        receipt["binary_sha256"]
        == hashlib.sha256(b"release-built-from-pin").hexdigest()
    )
    assert stale.read_bytes() == b"stale debug build"


def test_no_build_raises_naming_the_stale_debug_candidate(tmp_path):
    checkout, pin_sha = _engine_checkout(tmp_path)
    stale = _stale_debug_binary(checkout)
    pin = EnginePin(sha=pin_sha, source=tmp_path / "toolchain.toml")

    with pytest.raises(EngineBindingError) as error:
        resolve_pinned_engine_binary(checkout, pin, allow_build=False)

    message = str(error.value)
    assert str(stale) in message
    assert "no receipt" in message
    assert pin_sha in message
    assert "cargo build --release --locked" in message
    assert "axiom-encode engine-bind" in message


def test_wrong_commit_checkout_raises_naming_both_shas_and_pin_source(
    tmp_path, monkeypatch
):
    checkout, pin_sha = _engine_checkout(tmp_path)
    head = _drift_commit(checkout)
    source = tmp_path / "rulespec-us" / ".axiom" / "toolchain.toml"
    pin = EnginePin(sha=pin_sha, source=source)
    _cargo_shim(tmp_path, monkeypatch)

    with pytest.raises(EngineBindingError) as error:
        resolve_pinned_engine_binary(checkout, pin)

    message = str(error.value)
    assert pin_sha in message
    assert head in message
    assert str(source) in message
    # Skew fails before any build runs.
    assert not (checkout / "target" / "release" / "axiom-rules-engine").exists()


def test_forged_receipt_for_a_different_commit_does_not_bind(tmp_path):
    checkout, pin_sha = _engine_checkout(tmp_path)
    head = _drift_commit(checkout)
    debug = _stale_debug_binary(checkout)
    write_engine_binding_receipt(debug, head, "debug")
    pin = EnginePin(sha=pin_sha, source=tmp_path / "toolchain.toml")

    with pytest.raises(EngineBindingError) as error:
        resolve_pinned_engine_binary(checkout, pin, allow_build=False)

    message = str(error.value)
    assert f"receipt commit {head} does not match the pin" in message
    assert str(debug) in message


def test_tampered_binary_bytes_invalidate_the_receipt(tmp_path, monkeypatch):
    checkout, pin_sha = _engine_checkout(tmp_path)
    release = checkout / "target" / "release" / "axiom-rules-engine"
    release.parent.mkdir(parents=True)
    release.write_bytes(b"original")
    write_engine_binding_receipt(release, pin_sha, "release")
    release.write_bytes(b"tampered")
    pin = EnginePin(sha=pin_sha, source=tmp_path / "toolchain.toml")

    with pytest.raises(EngineBindingError) as error:
        resolve_pinned_engine_binary(checkout, pin, allow_build=False)
    assert "binary bytes do not match the receipt binary_sha256" in str(error.value)

    _cargo_shim(tmp_path, monkeypatch)
    binary = resolve_pinned_engine_binary(checkout, pin)
    assert binary == release
    assert binary.read_bytes() == b"release-built-from-pin"
    receipt = read_engine_binding_receipt(binary)
    assert receipt is not None
    assert (
        receipt["binary_sha256"]
        == hashlib.sha256(b"release-built-from-pin").hexdigest()
    )


def test_dirty_checkout_raises(tmp_path, monkeypatch):
    checkout, pin_sha = _engine_checkout(tmp_path)
    (checkout / "src.rs").write_text("fn main() { /* uncommitted */ }\n")
    _cargo_shim(tmp_path, monkeypatch)
    pin = EnginePin(sha=pin_sha, source=tmp_path / "toolchain.toml")

    with pytest.raises(EngineBindingError, match="dirty"):
        resolve_pinned_engine_binary(checkout, pin)


def test_legacy_toolchain_with_extra_keys_yields_the_pin(tmp_path):
    checkout_root = tmp_path / "rulespec-us"
    content_root = checkout_root / "us"
    content_root.mkdir(parents=True)
    config_path = _write_toolchain(checkout_root, engine_ref="a" * 40)

    pin = load_declared_engine_pin(content_root)

    assert pin is not None
    assert pin.sha == "a" * 40
    assert pin.source == config_path


def test_strict_three_key_toolchain_yields_no_pin(tmp_path):
    checkout_root = tmp_path / "rulespec-us"
    content_root = checkout_root / "us"
    content_root.mkdir(parents=True)
    _write_toolchain(checkout_root, engine_ref=None)

    assert load_declared_engine_pin(content_root) is None


def test_malformed_engine_ref_raises(tmp_path):
    checkout_root = tmp_path / "rulespec-us"
    content_root = checkout_root / "us"
    content_root.mkdir(parents=True)
    for bad_ref in ("HEAD", "abc123", "A" * 40):
        _write_toolchain(checkout_root, engine_ref=bad_ref)
        with pytest.raises(EngineBindingError, match=ENGINE_PIN_FIELD):
            load_declared_engine_pin(content_root)


def test_no_toolchain_anywhere_yields_no_pin(tmp_path):
    content_root = tmp_path / "rulespec-us" / "us"
    content_root.mkdir(parents=True)

    assert load_declared_engine_pin(content_root) is None


def test_nonexistent_policy_repo_path_raises(tmp_path):
    # A path typo must not silently downgrade to unpinned binding.
    with pytest.raises(EngineBindingError, match="does not exist"):
        load_declared_engine_pin(tmp_path / "missing")


def test_multiple_ancestor_toolchains_fail_closed(tmp_path):
    checkout_root = tmp_path / "rulespec-us"
    content_root = checkout_root / "us"
    content_root.mkdir(parents=True)
    _write_toolchain(checkout_root, engine_ref="a" * 40)
    _write_toolchain(content_root, engine_ref="b" * 40)

    with pytest.raises(EngineBindingError, match="multiple ancestor"):
        load_declared_engine_pin(content_root)


def test_toolchain_walk_is_bounded_at_the_git_root(tmp_path):
    outer = tmp_path / "outer"
    checkout_root = outer / "rulespec-us"
    content_root = checkout_root / "us"
    content_root.mkdir(parents=True)
    _write_toolchain(outer, engine_ref="a" * 40)
    _git(checkout_root, "init", "-q")

    assert load_declared_engine_pin(content_root) is None


def test_bare_binary_shim_binds_only_with_a_valid_receipt(tmp_path):
    shim_checkout = tmp_path / "engine-shim"
    shim_checkout.mkdir()
    bare = shim_checkout / "axiom-rules-engine"
    bare.write_bytes(b"prebuilt engine")
    pin = EnginePin(sha="a" * 40, source=tmp_path / "toolchain.toml")

    with pytest.raises(EngineBindingError, match="not a git repository") as error:
        resolve_pinned_engine_binary(shim_checkout, pin)
    assert str(bare) in str(error.value)

    write_engine_binding_receipt(bare, "a" * 40, "prebuilt")
    assert resolve_pinned_engine_binary(shim_checkout, pin) == bare


def test_receipt_round_trip_and_malformed_reads(tmp_path):
    binary = tmp_path / "axiom-rules-engine"
    binary.write_bytes(b"bits")

    receipt_path = write_engine_binding_receipt(binary, "b" * 40, "release")

    assert receipt_path == engine_binding_receipt_path(binary)
    receipt = read_engine_binding_receipt(binary)
    assert receipt is not None
    assert receipt["engine_commit"] == "b" * 40
    assert receipt["binary_sha256"] == hashlib.sha256(b"bits").hexdigest()
    assert receipt["cargo_profile"] == "release"
    assert receipt["written_by"].startswith("axiom-encode ")
    assert "built_at" in receipt

    receipt_path.write_text("{not json")
    assert read_engine_binding_receipt(binary) is None
    receipt_path.write_text("[1, 2]")
    assert read_engine_binding_receipt(binary) is None
    receipt_path.unlink()
    assert read_engine_binding_receipt(binary) is None
    # The atomic write leaves no staged temp files behind.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["axiom-rules-engine"]


def test_write_receipt_wraps_os_errors_as_binding_errors(tmp_path):
    with pytest.raises(EngineBindingError, match="binding receipt"):
        write_engine_binding_receipt(tmp_path / "missing-binary", "b" * 40, "release")


def test_cli_engine_bind_builds_and_reports_json(tmp_path, monkeypatch, capsys):
    checkout, pin_sha = _engine_checkout(tmp_path)
    _cargo_shim(tmp_path, monkeypatch)
    policy_checkout = tmp_path / "rulespec-us"
    (policy_checkout / "us").mkdir(parents=True)
    _write_toolchain(policy_checkout, engine_ref=pin_sha)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "axiom-encode",
            "engine-bind",
            "--axiom-rules-engine-path",
            str(checkout),
            "--policy-repo",
            str(policy_checkout / "us"),
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["binary"] == str(
        checkout / "target" / "release" / "axiom-rules-engine"
    )
    assert payload["engine_commit"] == pin_sha
    assert Path(payload["receipt"]).is_file()


def test_cli_engine_bind_no_build_failure_exits_one(tmp_path, monkeypatch, capsys):
    checkout, pin_sha = _engine_checkout(tmp_path)
    _stale_debug_binary(checkout)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "axiom-encode",
            "engine-bind",
            "--axiom-rules-engine-path",
            str(checkout),
            "--pin",
            pin_sha,
            "--no-build",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "Engine binding failed" in err
    assert "no receipt" in err


def test_cli_engine_bind_requires_a_declared_pin(tmp_path, monkeypatch, capsys):
    checkout, _pin_sha = _engine_checkout(tmp_path)
    policy_checkout = tmp_path / "rulespec-us"
    (policy_checkout / "us").mkdir(parents=True)
    _write_toolchain(policy_checkout, engine_ref=None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "axiom-encode",
            "engine-bind",
            "--axiom-rules-engine-path",
            str(checkout),
            "--policy-repo",
            str(policy_checkout / "us"),
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 1
    assert ENGINE_PIN_FIELD in capsys.readouterr().err


def test_cli_compile_threads_engine_ref_override(tmp_path, monkeypatch, capsys):
    rules_file = tmp_path / "rulespec-us" / "us" / "statutes" / "1" / "1.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")
    engine_root = tmp_path / "axiom-rules-engine"
    engine_root.mkdir()
    captured: dict[str, object] = {}

    class _RecordingPipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def _compile_rulespec_to_artifact(self, rulespec_file, output_path):
            output_path.write_text('{"program": {"derived": []}}')
            return (
                subprocess.CompletedProcess(["engine"], 0, stdout="", stderr=""),
                {"program": {"derived": []}},
            )

    monkeypatch.setattr(cli, "ValidatorPipeline", _RecordingPipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "axiom-encode",
            "compile",
            str(rules_file),
            "--axiom-rules-engine-path",
            str(engine_root),
            "--axiom-rules-engine-ref",
            "e" * 40,
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    assert captured["axiom_rules_engine_ref"] == "e" * 40
