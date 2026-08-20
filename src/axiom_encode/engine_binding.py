"""Deterministic binding of the Axiom rules engine binary to a declared pin.

Binding receipts are unsigned local attestations: they protect against
accidental drift (stale builds, tampered bytes, wrong-commit checkouts), not
against an adversary — forging a receipt requires the same filesystem access
as replacing the binary itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .corpus_resolver import UnsafeCorpusPathError, read_bounded_regular_file

ENGINE_PIN_FIELD = "axiom_rules_engine_ref"
RECEIPT_SUFFIX = ".provenance.json"
ENGINE_BINARY_NAME = "axiom-rules-engine"
# Release-profile build CI runs for the engine (targeted-signed-reencode.yml).
ENGINE_RELEASE_BUILD_COMMAND = ("cargo", "build", "--release", "--locked")
# Mirrors toolchain.MAX_RULESPEC_TOOLCHAIN_BYTES without importing its loader.
MAX_ENGINE_PIN_TOOLCHAIN_BYTES = 64 * 1024
MAX_ENGINE_BINDING_RECEIPT_BYTES = 64 * 1024
_ENGINE_BUILD_TIMEOUT_SECONDS = 3600
_GIT_QUERY_TIMEOUT_SECONDS = 60
_ENGINE_REF_RE = re.compile(r"^[0-9a-f]{40}$")
_RECEIPT_VERIFIED = "verified receipt"
# Mirrors validator_pipeline._SENSITIVE_ENV_NAME_MARKERS; PATH and HOME
# survive so git and cargo resolve normally.
_SENSITIVE_ENV_NAME_MARKERS = (
    "AUTH",
    "CREDENTIAL",
    "KEY",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)


class EngineBindingError(ValueError):
    """The engine binary cannot be provably bound to the declared engine pin."""


@dataclass(frozen=True, slots=True)
class EnginePin:
    """One declared engine commit and the source that declared it."""

    sha: str
    source: Path


def require_engine_ref_sha(value: object, *, description: str) -> str:
    """Validate one caller-supplied engine ref as a full lowercase commit SHA."""

    if not isinstance(value, str) or _ENGINE_REF_RE.fullmatch(value) is None:
        raise EngineBindingError(
            f"{description} must be a full lowercase 40-hex commit SHA: {value!r}"
        )
    return value


def _declared_toolchain_file(policy_repo_path: Path) -> Path | None:
    """Locate one ancestor .axiom/toolchain.toml, bounded at the first git root."""

    raw = Path(os.path.abspath(Path(policy_repo_path).expanduser()))
    if raw.is_symlink():
        raise EngineBindingError(f"Policy repo path must not be a symlink: {raw}")
    try:
        root = raw.resolve(strict=True)
    except OSError as exc:
        raise EngineBindingError(
            f"Policy repo path does not exist: {raw}; a missing path cannot "
            f"prove that no {ENGINE_PIN_FIELD} pin is declared"
        ) from exc
    search_root = root.parent if root.is_file() else root
    if not search_root.is_dir():
        return None
    search_chain = (search_root, *search_root.parents)
    repository_root: Path | None = None
    repository_root_index: int | None = None
    for index, candidate in enumerate(search_chain):
        git_marker = candidate / ".git"
        if git_marker.is_symlink():
            raise EngineBindingError(
                f"Policy repo .git marker must not be a symlink: {git_marker}"
            )
        if git_marker.exists():
            repository_root = candidate
            repository_root_index = index
            break
    scoped_chain = (
        search_chain
        if repository_root_index is None
        else search_chain[: repository_root_index + 1]
    )
    configured_roots = [
        candidate
        for candidate in scoped_chain
        if (candidate / ".axiom" / "toolchain.toml").exists()
        or (candidate / ".axiom" / "toolchain.toml").is_symlink()
    ]
    if not configured_roots:
        return None
    if repository_root is not None and configured_roots != [repository_root]:
        raise EngineBindingError(
            "Policy repo must declare exactly one .axiom/toolchain.toml at its "
            f"repository root {repository_root}; found configuration under: "
            + ", ".join(str(path) for path in configured_roots)
        )
    if len(configured_roots) > 1:
        raise EngineBindingError(
            "Policy repo path has multiple ancestor .axiom/toolchain.toml files: "
            + ", ".join(str(path) for path in configured_roots)
        )
    config_path = configured_roots[0] / ".axiom" / "toolchain.toml"
    if config_path.is_symlink():
        raise EngineBindingError(
            f"RuleSpec toolchain file must not be a symlink: {config_path}"
        )
    return config_path


def load_declared_engine_pin(policy_repo_path: Path) -> EnginePin | None:
    """Return the policy repo's declared engine pin, or None when undeclared.

    Tolerates the legacy multi-key [toolchain] format: only the
    axiom_rules_engine_ref field is read here, so strict 3-key repos parse
    and yield None instead of erroring like the strict toolchain loader.
    The reverse does not hold: the strict loader still rejects a table that
    declares this field, so strict repos cannot declare the pin until that
    schema is widened; the --axiom-rules-engine-ref override pins them
    meanwhile.
    """

    config_path = _declared_toolchain_file(policy_repo_path)
    if config_path is None:
        return None
    configured_root = config_path.parent.parent
    try:
        raw = read_bounded_regular_file(
            configured_root,
            config_path,
            label="RuleSpec toolchain file",
            max_bytes=MAX_ENGINE_PIN_TOOLCHAIN_BYTES,
        )
        payload = tomllib.loads(raw.decode("utf-8"))
    except UnsafeCorpusPathError as exc:
        raise EngineBindingError(str(exc)) from exc
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise EngineBindingError(
            f"RuleSpec toolchain is not valid UTF-8 TOML: {config_path}"
        ) from exc
    toolchain = payload.get("toolchain")
    if toolchain is None:
        return None
    if not isinstance(toolchain, dict):
        raise EngineBindingError(f"[toolchain] must be a TOML table: {config_path}")
    if ENGINE_PIN_FIELD not in toolchain:
        return None
    declared = toolchain[ENGINE_PIN_FIELD]
    if not isinstance(declared, str) or _ENGINE_REF_RE.fullmatch(declared) is None:
        raise EngineBindingError(
            f"[toolchain].{ENGINE_PIN_FIELD} must be a full lowercase 40-hex "
            f"commit SHA in {config_path}: {declared!r}"
        )
    return EnginePin(sha=declared, source=config_path)


def engine_binding_receipt_path(binary: Path) -> Path:
    """Return the binding receipt path recorded beside one engine binary."""

    binary = Path(binary)
    return binary.with_name(binary.name + RECEIPT_SUFFIX)


def read_engine_binding_receipt(binary: Path) -> dict[str, Any] | None:
    """Return the parsed binding receipt beside a binary, or None if unusable."""

    receipt_path = engine_binding_receipt_path(binary)
    try:
        if receipt_path.is_symlink() or not receipt_path.is_file():
            return None
        if receipt_path.stat().st_size > MAX_ENGINE_BINDING_RECEIPT_BYTES:
            return None
        raw = receipt_path.read_bytes()
    except OSError:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def write_engine_binding_receipt(
    binary: Path,
    engine_commit: str,
    cargo_profile: str,
) -> Path:
    """Atomically record which engine commit produced one binary's exact bytes."""

    from axiom_encode import __version__

    binary = Path(binary)
    receipt_path = engine_binding_receipt_path(binary)
    sha = require_engine_ref_sha(engine_commit, description="engine_commit")
    try:
        payload = {
            "engine_commit": sha,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "cargo_profile": str(cargo_profile),
            "built_at": datetime.now(timezone.utc).isoformat(),
            "written_by": f"axiom-encode {__version__}",
        }
        descriptor, staged_name = tempfile.mkstemp(
            dir=str(receipt_path.parent),
            prefix=receipt_path.name + ".",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as staged:
                json.dump(payload, staged, indent=2, sort_keys=True)
                staged.write("\n")
            os.replace(staged_name, receipt_path)
        finally:
            if os.path.exists(staged_name):
                os.unlink(staged_name)
    except OSError as exc:
        raise EngineBindingError(
            f"Failed to write the engine binding receipt {receipt_path}: {exc}"
        ) from exc
    return receipt_path


def _engine_binary_candidates(engine_checkout: Path) -> tuple[Path, ...]:
    return (
        engine_checkout / "target" / "release" / ENGINE_BINARY_NAME,
        engine_checkout / "target" / "debug" / ENGINE_BINARY_NAME,
        engine_checkout / ENGINE_BINARY_NAME,
    )


def _receipt_status(binary: Path, pin: EnginePin) -> str:
    receipt = read_engine_binding_receipt(binary)
    if receipt is None:
        return "no receipt"
    commit = receipt.get("engine_commit")
    digest = receipt.get("binary_sha256")
    if not isinstance(commit, str) or not isinstance(digest, str):
        return "malformed receipt"
    if commit != pin.sha:
        return f"receipt commit {commit} does not match the pin"
    try:
        actual = hashlib.sha256(binary.read_bytes()).hexdigest()
    except OSError:
        return "unreadable binary"
    if actual != digest:
        return "binary bytes do not match the receipt binary_sha256"
    return _RECEIPT_VERIFIED


def _describe_candidates(statuses: Sequence[tuple[Path, str]]) -> str:
    if not statuses:
        return "none"
    return ", ".join(f"{path} ({status})" for path, status in statuses)


def _binding_context(
    pin: EnginePin,
    engine_checkout: Path,
    *,
    head: str | None,
    statuses: Sequence[tuple[Path, str]],
) -> str:
    remediation = (
        f"run `{' '.join(ENGINE_RELEASE_BUILD_COMMAND)}` in {engine_checkout}, "
        f"or `axiom-encode engine-bind --axiom-rules-engine-path "
        f"{engine_checkout} --pin {pin.sha}`"
    )
    return (
        f"; pin {pin.sha} declared by {pin.source}; "
        f"engine checkout {engine_checkout} (HEAD {head or 'unknown'}); "
        f"existing candidate binaries: {_describe_candidates(statuses)}; "
        f"remediation: {remediation}"
    )


def _binding_subprocess_env() -> dict[str, str]:
    """Copy the ambient environment without credential-named variables."""

    return {
        name: value
        for name, value in os.environ.items()
        if not any(marker in name.upper() for marker in _SENSITIVE_ENV_NAME_MARKERS)
    }


def _git_output(
    engine_checkout: Path,
    *args: str,
    pin: EnginePin,
    statuses: Sequence[tuple[Path, str]],
) -> str:
    rendered = "git " + " ".join(args)
    try:
        result = subprocess.run(
            ["git", "-C", str(engine_checkout), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_QUERY_TIMEOUT_SECONDS,
            env=_binding_subprocess_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EngineBindingError(
            f"{rendered} failed in the engine checkout: {exc}"
            + _binding_context(pin, engine_checkout, head=None, statuses=statuses)
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise EngineBindingError(
            f"{rendered} failed in the engine checkout: {detail}"
            + _binding_context(pin, engine_checkout, head=None, statuses=statuses)
        )
    return result.stdout.strip()


def _build_pinned_release_binary(
    engine_checkout: Path,
    pin: EnginePin,
    *,
    head: str,
    statuses: Sequence[tuple[Path, str]],
) -> Path:
    rendered = " ".join(ENGINE_RELEASE_BUILD_COMMAND)
    try:
        result = subprocess.run(
            list(ENGINE_RELEASE_BUILD_COMMAND),
            capture_output=True,
            text=True,
            timeout=_ENGINE_BUILD_TIMEOUT_SECONDS,
            cwd=str(engine_checkout),
            env=_binding_subprocess_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EngineBindingError(
            f"`{rendered}` failed for the pinned engine: {exc}"
            + _binding_context(pin, engine_checkout, head=head, statuses=statuses)
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr.strip() or result.stdout.strip())[-2000:]
        raise EngineBindingError(
            f"`{rendered}` failed for the pinned engine: {detail}"
            + _binding_context(pin, engine_checkout, head=head, statuses=statuses)
        )
    release_binary = engine_checkout / "target" / "release" / ENGINE_BINARY_NAME
    if not release_binary.exists():
        raise EngineBindingError(
            f"`{rendered}` succeeded but produced no {release_binary}"
            + _binding_context(pin, engine_checkout, head=head, statuses=statuses)
        )
    write_engine_binding_receipt(release_binary, pin.sha, "release")
    return release_binary


def resolve_pinned_engine_binary(
    engine_checkout: Path,
    pin: EnginePin,
    *,
    allow_build: bool = True,
) -> Path:
    """Return a binary provably built from the pinned engine commit.

    Receipted candidates bind first (release, debug, bare order); otherwise a
    clean git checkout at the pin may build the release profile on demand and
    receipt the result. Everything else fails closed.
    """

    engine_checkout = Path(engine_checkout)
    statuses: list[tuple[Path, str]] = []
    for candidate in _engine_binary_candidates(engine_checkout):
        if not candidate.exists():
            continue
        status = _receipt_status(candidate, pin)
        if status == _RECEIPT_VERIFIED:
            return candidate
        statuses.append((candidate, status))
    if not (engine_checkout / ".git").exists():
        raise EngineBindingError(
            "Engine checkout is not a git repository and no candidate binary "
            "carries a verified receipt for the pinned commit; use a git "
            "checkout at the pin, or write a receipt with "
            "'axiom-encode engine-bind'"
            + _binding_context(pin, engine_checkout, head=None, statuses=statuses)
        )
    head = _git_output(engine_checkout, "rev-parse", "HEAD", pin=pin, statuses=statuses)
    if head != pin.sha:
        raise EngineBindingError(
            f"Engine checkout HEAD {head} does not match the pinned commit "
            f"{pin.sha}"
            + _binding_context(pin, engine_checkout, head=head, statuses=statuses)
        )
    porcelain = _git_output(
        engine_checkout, "status", "--porcelain", pin=pin, statuses=statuses
    )
    if porcelain:
        raise EngineBindingError(
            "Engine checkout is dirty at the pinned commit; binary provenance "
            "is unprovable"
            + _binding_context(pin, engine_checkout, head=head, statuses=statuses)
        )
    if not allow_build:
        raise EngineBindingError(
            "No candidate binary carries a verified receipt for the pinned "
            "commit and building is disabled"
            + _binding_context(pin, engine_checkout, head=head, statuses=statuses)
        )
    return _build_pinned_release_binary(
        engine_checkout, pin, head=head, statuses=statuses
    )
