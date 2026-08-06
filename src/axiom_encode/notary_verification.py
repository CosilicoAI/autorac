"""Strict, non-mutating verification receipts for RuleSpec checkouts.

This module deliberately does not share the generated-apply overlay.  The
overlay is an installation surface and is allowed to repair generated output;
the notary profile verifies the bytes already present in a clean checkout.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from axiom_encode import __version__

from .constants import (
    RULESPEC_FILE_SUFFIX,
    RULESPEC_TEST_FILE_SUFFIX,
)
from .harness.eval_evidence import scrub_attestation_signing_keys
from .harness.evals import (
    _deterministic_tree_identity,
)
from .harness.policyengine_runtime import (
    POLICYENGINE_RUNTIME_PIN_PATH,
    PolicyEngineRuntime,
)
from .harness.validator_pipeline import (
    PipelineResult,
    ValidatorPipeline,
    resolve_axiom_rules_engine_binary,
)
from .repo_routing import (
    RULESPEC_PROTECTED_ATOMIC_MODULE_ROOTS,
    atomic_rulespec_module_paths,
    canonical_rulespec_repo_name,
    find_policy_repo_root,
    inspect_canonical_rulespec_checkout,
    jurisdiction_subdir_names,
)
from .signing_broker import SigningBrokerError, get_signing_broker
from .toolchain import (
    VALIDATION_WAIVER_SET_PATH,
    load_rulespec_local_corpus_release,
    verify_rulespec_validation_waiver_set,
)
from .validation_waivers import load_validation_waivers

NOTARY_RECEIPT_SCHEMA_ID = "axiom/notary-verification-receipt/v0"
NOTARY_RECEIPT_SCHEMA_STATUS = "PROVISIONAL"
NOTARY_PROFILE_ID = "axiom/notary-verifier/strict-v0"
MIN_POLICYENGINE_MATCH = 0.95

GateReproducibility = Literal["public", "restricted-pinned"]
GateStatus = Literal["passed", "failed", "reduced"]
_ProtectedRootIdentity = tuple[int, int, str]
_FilesystemIdentity = tuple[int, int]
_TargetDisposition = Literal["verified", "deleted", "out-of-scope"]

_PUBLIC_GATES = (
    "subject-clean",
    "corpus-release-binding",
    "compile",
    "proof-revalidation",
    "companion-tests",
    "grounding-contract",
    "layout-inspection",
    "waiver-set-verification",
    "policy-repo-nonmutation",
)
_RESTRICTED_PINNED_GATES = ("policyengine-oracle",)
_AUTHORITY_SURFACE_PATHS = frozenset(
    {
        ".gitmodules",
        ".github/CODEOWNERS",
        "CODEOWNERS",
        "docs/CODEOWNERS",
        str(VALIDATION_WAIVER_SET_PATH),
    }
)
_GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRIVILEGED_FILE_MODE_MASK = (
    stat.S_ISUID
    | stat.S_ISGID
    | stat.S_ISVTX
    | stat.S_IXUSR
    | stat.S_IXGRP
    | stat.S_IXOTH
)


class NotaryVerificationError(RuntimeError):
    """The strict verifier could not produce trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class NotaryVerificationResult:
    """Receipt and diagnostic outcome from one strict verification run."""

    receipt: dict[str, Any]
    passed: bool
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CleanGitIdentity:
    root: Path
    commit: str
    tree: str
    git_dir: Path
    git_common_dir: Path
    root_identity: _FilesystemIdentity
    git_dir_identity: _FilesystemIdentity
    git_common_dir_identity: _FilesystemIdentity


@dataclass(frozen=True, slots=True)
class _GitTreeEntry:
    mode: str
    object_id: str
    relative: PurePosixPath


@dataclass(frozen=True, slots=True)
class _NotaryTargetFile:
    relative: PurePosixPath
    disposition: _TargetDisposition


@dataclass(frozen=True, slots=True)
class _NotaryTargetSet:
    mode: Literal["diff", "whole-repo"]
    base_commit: str | None
    files: tuple[_NotaryTargetFile, ...]
    verification_targets: tuple[PurePosixPath, ...]
    authority_surfaces: tuple[PurePosixPath, ...]


@dataclass(frozen=True, slots=True)
class _MaterializedVerificationEvidence:
    corpus_release_name: str
    corpus_release_content_sha256: str
    waiver_sha256: str
    waiver_count: int
    policyengine_runtime: PolicyEngineRuntime | None
    compile_passed: bool
    deterministic_statuses: Mapping[str, bool]
    oracle_status: GateStatus
    validators_passed: bool
    issues: tuple[str, ...]


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return the canonical JSON encoding used by the provisional receipt."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_receipt_body_bytes(receipt: Mapping[str, Any]) -> bytes:
    """Canonicalize a receipt body without its detached-style self-hash."""

    body = copy.deepcopy(dict(receipt))
    body.pop("receipt_sha256", None)
    return canonical_json_bytes(body)


def receipt_body_sha256(receipt: Mapping[str, Any]) -> str:
    """Hash only the canonical receipt body, excluding ``receipt_sha256``."""

    return hashlib.sha256(canonical_receipt_body_bytes(receipt)).hexdigest()


def attach_receipt_sha256(body: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached-style, self-hashed receipt without mutating ``body``."""

    receipt = copy.deepcopy(dict(body))
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = receipt_body_sha256(receipt)
    return receipt


def canonical_receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    """Serialize a complete receipt in its sole accepted JSON representation."""

    expected = receipt_body_sha256(receipt)
    if receipt.get("receipt_sha256") != expected:
        raise NotaryVerificationError(
            "receipt_sha256 does not match the canonical receipt body"
        )
    return canonical_json_bytes(receipt)


def gate_reproducibility(gate: str) -> GateReproducibility:
    """Return the charter classification for one strict-profile gate."""

    if gate in _PUBLIC_GATES:
        return "public"
    if gate in _RESTRICTED_PINNED_GATES:
        return "restricted-pinned"
    raise ValueError(f"Unknown notary verification gate: {gate}")


def _gate(gate: str, status: GateStatus) -> dict[str, str]:
    return {
        "gate": gate,
        "status": status,
        "reproducibility": gate_reproducibility(gate),
    }


def _resolve_existing_directory(raw_path: Path, *, label: str) -> Path:
    raw = Path(os.path.abspath(Path(raw_path).expanduser()))
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise NotaryVerificationError(f"{label} path contains a symlink: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise NotaryVerificationError(f"{label} path does not exist: {raw}") from exc
    if not resolved.is_dir():
        raise NotaryVerificationError(f"{label} path is not a directory: {raw}")
    return resolved


def _resolve_policy_checkout(raw_path: Path) -> Path:
    root = _resolve_existing_directory(raw_path, label="RuleSpec checkout")
    inspection = inspect_canonical_rulespec_checkout(
        root,
        allow_composition_specs=True,
    )
    if inspection.name is None:
        raise NotaryVerificationError(
            "RuleSpec checkout must be the exact canonical rulespec-<country> "
            f"checkout ({inspection.rejection}): {root}"
        )
    return root


def _git_bytes(
    root: Path,
    *args: str,
    input_bytes: bytes | None = None,
) -> bytes:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise NotaryVerificationError("Cannot inspect Git identity: git is unavailable")
    environment = scrub_attestation_signing_keys()
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_NO_LAZY_FETCH"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        completed = subprocess.run(
            [
                git_executable,
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(root),
                *args,
            ],
            input=input_bytes,
            capture_output=True,
            check=False,
            env=environment,
        )
    except OSError as exc:
        raise NotaryVerificationError(
            f"Cannot inspect {root.name} Git identity: {exc}"
        ) from exc
    if completed.returncode != 0:
        raw_detail = completed.stderr.strip() or completed.stdout.strip()
        detail = raw_detail.decode("utf-8", errors="replace")
        raise NotaryVerificationError(
            f"Cannot inspect {root.name} Git identity: {detail or 'git failed'}"
        )
    return completed.stdout


def _git_stdout(root: Path, *args: str) -> str:
    try:
        return _git_bytes(root, *args).decode("utf-8")
    except UnicodeError as exc:
        raise NotaryVerificationError(
            f"Cannot inspect {root.name} Git identity: non-UTF-8 path"
        ) from exc


def _git_text(root: Path, *args: str) -> str:
    return _git_stdout(root, *args).strip()


def _git_object_hasher(object_type: str, size: int, object_id: str):
    if object_type not in {"blob", "commit", "tree"}:
        raise NotaryVerificationError(f"Unsupported Git object type: {object_type}")
    if len(object_id) == 40:
        digest = hashlib.sha1(usedforsecurity=False)
    elif len(object_id) == 64:
        digest = hashlib.sha256()
    else:  # pragma: no cover - guarded by _GIT_OID_RE
        raise NotaryVerificationError("Unsupported Git object format")
    digest.update(f"{object_type} {size}\0".encode("ascii"))
    return digest


def _git_object_payloads(
    root: Path,
    expected_objects: Sequence[tuple[str, str]],
) -> tuple[bytes, ...]:
    """Read and independently hash-check exact Git object preimages."""

    if not expected_objects:
        return ()
    queries = b"".join(
        object_id.encode("ascii") + b"\n"
        for object_id, _object_type in expected_objects
    )
    output = _git_bytes(
        root,
        "cat-file",
        "--batch",
        input_bytes=queries,
    )
    payloads: list[bytes] = []
    cursor = 0
    for expected_id, expected_type in expected_objects:
        header_end = output.find(b"\n", cursor)
        if header_end < 0:
            raise NotaryVerificationError(
                f"Cannot read {root.name} Git {expected_type} object {expected_id}"
            )
        try:
            raw_object_id, raw_object_type, raw_size = output[cursor:header_end].split(
                b" "
            )
            object_id = raw_object_id.decode("ascii")
            object_type = raw_object_type.decode("ascii")
            size = int(raw_size.decode("ascii"))
        except (UnicodeError, ValueError) as exc:
            raise NotaryVerificationError(
                f"Cannot read {root.name} Git {expected_type} object {expected_id}"
            ) from exc
        payload_start = header_end + 1
        payload_end = payload_start + size
        if (
            object_id != expected_id
            or object_type != expected_type
            or size < 0
            or payload_end >= len(output)
            or output[payload_end : payload_end + 1] != b"\n"
        ):
            raise NotaryVerificationError(
                f"Cannot read {root.name} Git {expected_type} object {expected_id}"
            )
        payload = output[payload_start:payload_end]
        digest = _git_object_hasher(expected_type, size, expected_id)
        digest.update(payload)
        if digest.hexdigest() != expected_id:
            raise NotaryVerificationError(
                f"{root.name} Git {expected_type} payload does not match its "
                f"object identity: {expected_id}"
            )
        payloads.append(payload)
        cursor = payload_end + 1
    if cursor != len(output):
        raise NotaryVerificationError(
            f"Cannot read {root.name} Git objects: unexpected batch output"
        )
    return tuple(payloads)


def _verified_commit_tree(root: Path, commit: str) -> str:
    payload = _git_object_payloads(root, ((commit, "commit"),))[0]
    first_line = payload.partition(b"\n")[0]
    try:
        marker, raw_tree = first_line.split(b" ", 1)
        tree = raw_tree.decode("ascii")
    except (UnicodeError, ValueError) as exc:
        raise NotaryVerificationError(
            f"{root.name} HEAD commit has no valid tree identity"
        ) from exc
    if (
        marker != b"tree"
        or _GIT_OID_RE.fullmatch(tree) is None
        or len(tree) != len(commit)
    ):
        raise NotaryVerificationError(
            f"{root.name} HEAD commit has no valid tree identity"
        )
    return tree


def _commit_parent_ids(payload: bytes, *, commit: str) -> tuple[str, ...]:
    """Parse parent identities from one independently hash-checked commit."""

    header, separator, _message = payload.partition(b"\n\n")
    if not separator:
        raise NotaryVerificationError(f"Git commit has no complete header: {commit}")
    parents: list[str] = []
    for line in header.split(b"\n"):
        if not line.startswith(b"parent "):
            continue
        try:
            parent = line.removeprefix(b"parent ").decode("ascii")
        except UnicodeError as exc:
            raise NotaryVerificationError(
                f"Git commit has a malformed parent identity: {commit}"
            ) from exc
        if len(parent) != len(commit) or _GIT_OID_RE.fullmatch(parent) is None:
            raise NotaryVerificationError(
                f"Git commit has a malformed parent identity: {commit}"
            )
        parents.append(parent)
    if len(parents) != len(set(parents)):
        raise NotaryVerificationError(f"Git commit repeats a parent identity: {commit}")
    return tuple(parents)


def _assert_verified_commit_ancestor(
    root: Path,
    *,
    base_commit: str,
    head_commit: str,
) -> None:
    """Prove base reachability using hash-checked commit parent preimages."""

    if base_commit == head_commit:
        return
    try:
        candidate_commits = tuple(
            line
            for line in _git_stdout(
                root,
                "rev-list",
                "--ancestry-path",
                f"{base_commit}..{head_commit}",
            ).splitlines()
            if line
        )
    except NotaryVerificationError as exc:
        raise NotaryVerificationError(
            "--base-commit must be an ancestor of HEAD"
        ) from exc
    if (
        not candidate_commits
        or head_commit not in candidate_commits
        or len(candidate_commits) != len(set(candidate_commits))
        or any(
            len(commit) != len(head_commit) or _GIT_OID_RE.fullmatch(commit) is None
            for commit in candidate_commits
        )
    ):
        raise NotaryVerificationError("--base-commit must be an ancestor of HEAD")

    payloads = _git_object_payloads(
        root,
        tuple((commit, "commit") for commit in candidate_commits),
    )
    parents_by_commit = {
        commit: _commit_parent_ids(payload, commit=commit)
        for commit, payload in zip(candidate_commits, payloads, strict=True)
    }
    pending = [head_commit]
    visited: set[str] = set()
    while pending:
        commit = pending.pop()
        if commit in visited:
            continue
        visited.add(commit)
        for parent in parents_by_commit.get(commit, ()):
            if parent == base_commit:
                return
            if parent in parents_by_commit and parent not in visited:
                pending.append(parent)
    raise NotaryVerificationError("--base-commit must be an ancestor of HEAD")


def _git_tree_entries(identity: _CleanGitIdentity) -> tuple[_GitTreeEntry, ...]:
    """Walk only independently hash-checked tree objects from the commit root."""

    entries: list[_GitTreeEntry] = []
    casefold_paths: dict[str, str] = {}
    observed_paths: set[str] = set()
    pending: list[tuple[str, PurePosixPath]] = [(identity.tree, PurePosixPath("."))]
    object_id_bytes = len(identity.tree) // 2
    while pending:
        tree_id, parent = pending.pop()
        payload = _git_object_payloads(identity.root, ((tree_id, "tree"),))[0]
        cursor = 0
        while cursor < len(payload):
            mode_end = payload.find(b" ", cursor)
            name_end = payload.find(b"\0", mode_end + 1)
            object_end = name_end + 1 + object_id_bytes
            if (
                mode_end <= cursor
                or name_end <= mode_end + 1
                or object_end > len(payload)
            ):
                raise NotaryVerificationError(
                    f"{identity.root.name} HEAD tree object is malformed: {tree_id}"
                )
            raw_mode = payload[cursor:mode_end]
            raw_name = payload[mode_end + 1 : name_end]
            raw_object_id = payload[name_end + 1 : object_end]
            cursor = object_end
            try:
                mode = raw_mode.decode("ascii")
                name = raw_name.decode("utf-8")
            except UnicodeError as exc:
                raise NotaryVerificationError(
                    f"{identity.root.name} HEAD tree contains an unsafe Git entry"
                ) from exc
            if (
                not name
                or "/" in name
                or name in {".", ".."}
                or name.casefold() == ".git"
            ):
                raise NotaryVerificationError(
                    f"{identity.root.name} HEAD tree contains an unsafe path: {name!r}"
                )
            relative = parent / name
            relative_text = relative.as_posix()
            folded = relative_text.casefold()
            previous = casefold_paths.setdefault(folded, relative_text)
            if relative_text in observed_paths or previous != relative_text:
                conflicting = previous if previous != relative_text else relative_text
                raise NotaryVerificationError(
                    f"{identity.root.name} HEAD tree has a colliding path: "
                    f"{conflicting!r} and {relative_text!r}"
                )
            observed_paths.add(relative_text)
            object_id = raw_object_id.hex()
            if _GIT_OID_RE.fullmatch(object_id) is None:
                raise NotaryVerificationError(
                    f"{identity.root.name} HEAD tree contains an invalid object identity"
                )
            if mode in {"40000", "040000"}:
                pending.append((object_id, relative))
                continue
            if mode not in {"100644", "100755", "120000"}:
                raise NotaryVerificationError(
                    f"{identity.root.name} HEAD tree contains unsupported entry "
                    f"{mode}: {relative_text}"
                )
            entries.append(
                _GitTreeEntry(
                    mode=mode,
                    object_id=object_id,
                    relative=relative,
                )
            )
    return tuple(sorted(entries, key=lambda entry: entry.relative.as_posix()))


def _git_blob_hasher(size: int, object_id: str):
    return _git_object_hasher("blob", size, object_id)


def _worktree_blob_identity(path: Path, entry: _GitTreeEntry) -> tuple[str, str]:
    if entry.mode == "120000":
        if not path.is_symlink():
            raise NotaryVerificationError(f"tracked symlink is not a symlink: {path}")
        raw = os.fsencode(os.readlink(path))
        digest = _git_blob_hasher(len(raw), entry.object_id)
        digest.update(raw)
        return entry.mode, digest.hexdigest()

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise NotaryVerificationError(
            "Strict verification requires no-follow file opens on this platform"
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise NotaryVerificationError(f"tracked path is not a regular file: {path}")
        digest = _git_blob_hasher(before.st_size, entry.object_id)
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise NotaryVerificationError(
            f"Cannot read tracked file safely: {path}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise NotaryVerificationError(f"tracked file changed while hashing: {path}")
    mode = "100755" if before.st_mode & stat.S_IXUSR else "100644"
    return mode, digest.hexdigest()


def _require_head_tree_matches_worktree(
    identity: _CleanGitIdentity,
    *,
    label: str,
    entries: Sequence[_GitTreeEntry] | None = None,
) -> None:
    """Raw-bind every tracked worktree byte to HEAD without Git filters."""

    mismatches: list[str] = []
    admitted_entries = entries if entries is not None else _git_tree_entries(identity)
    for entry in admitted_entries:
        path = identity.root.joinpath(*entry.relative.parts)
        try:
            mode, object_id = _worktree_blob_identity(path, entry)
        except NotaryVerificationError as exc:
            mismatches.append(f"{entry.relative.as_posix()} ({exc})")
            continue
        if mode != entry.mode or object_id != entry.object_id:
            mismatches.append(entry.relative.as_posix())
    if mismatches:
        preview = ", ".join(mismatches[:5])
        if len(mismatches) > 5:
            preview += ", ..."
        raise NotaryVerificationError(
            f"{label} tracked bytes do not match the raw HEAD tree: {preview}"
        )


def _git_metadata_directory(root: Path, *args: str) -> Path:
    raw = Path(_git_text(root, "rev-parse", *args))
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise NotaryVerificationError(
            f"Cannot resolve {root.name} Git metadata directory: {candidate}"
        ) from exc
    if not resolved.is_dir():
        raise NotaryVerificationError(
            f"Git metadata path is not a directory: {resolved}"
        )
    return resolved


def _directory_identity(path: Path, *, label: str) -> _FilesystemIdentity:
    """Return a stable device/inode identity for one admitted directory."""

    try:
        metadata = path.stat()
    except OSError as exc:
        raise NotaryVerificationError(
            f"Cannot bind {label} filesystem identity: {path}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotaryVerificationError(f"{label} is not a directory: {path}")
    return metadata.st_dev, metadata.st_ino


def _git_blob_payloads(
    identity: _CleanGitIdentity,
    entries: Sequence[_GitTreeEntry],
) -> tuple[bytes, ...]:
    return _git_object_payloads(
        identity.root,
        tuple((entry.object_id, "blob") for entry in entries),
    )


def _materialize_git_tree(identity: _CleanGitIdentity, destination: Path) -> None:
    """Materialize exact raw HEAD blobs without checkout filters or Git metadata."""

    if destination.exists() or destination.is_symlink():
        raise NotaryVerificationError(
            f"Verification snapshot destination already exists: {destination}"
        )
    destination.mkdir(parents=True)
    entries = _git_tree_entries(identity)
    payloads = _git_blob_payloads(identity, entries)
    for entry, payload in zip(entries, payloads, strict=True):
        target = destination.joinpath(*entry.relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise NotaryVerificationError(
                f"Verification snapshot has a colliding path: {entry.relative}"
            )
        try:
            if entry.mode == "120000":
                os.symlink(os.fsdecode(payload), target)
            else:
                with target.open("xb") as handle:
                    handle.write(payload)
                target.chmod(0o755 if entry.mode == "100755" else 0o644)
        except OSError as exc:
            raise NotaryVerificationError(
                f"Cannot materialize verification snapshot path: {entry.relative}"
            ) from exc


def _copy_verified_engine_binary(
    source_root: Path,
    destination_root: Path,
    expected: Mapping[str, object],
) -> None:
    raw_relative = expected.get("path")
    expected_sha256 = expected.get("sha256")
    expected_size = expected.get("size")
    if (
        not isinstance(raw_relative, str)
        or not isinstance(expected_sha256, str)
        or _SHA256_RE.fullmatch(expected_sha256) is None
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise NotaryVerificationError("Axiom rules engine identity is malformed")
    relative = PurePosixPath(raw_relative)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise NotaryVerificationError("Axiom rules engine path is unsafe")
    source = source_root.joinpath(*relative.parts)
    destination = destination_root.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise NotaryVerificationError(
            "Strict verification requires no-follow file opens on this platform"
        )
    descriptor: int | None = None
    digest = hashlib.sha256()
    copied_size = 0
    try:
        descriptor = os.open(
            source,
            os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise NotaryVerificationError(
                f"Axiom rules engine executable is not a regular file: {source}"
            )
        with destination.open("xb") as output:
            while chunk := os.read(descriptor, 1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                copied_size += len(chunk)
        after = os.fstat(descriptor)
        destination.chmod(0o755 if before.st_mode & 0o111 else 0o644)
    except OSError as exc:
        raise NotaryVerificationError(
            "Cannot create the hash-bound Axiom rules engine snapshot"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise NotaryVerificationError(
            "Axiom rules engine executable changed while being snapshotted"
        )
    if copied_size != expected_size or digest.hexdigest() != expected_sha256:
        raise NotaryVerificationError(
            "Axiom rules engine executable changed before snapshot execution"
        )


def _require_plain_git_index(root: Path, *, label: str) -> None:
    """Reject index flags that can hide live bytes from clean-tree checks."""

    entries = _git_stdout(root, "ls-files", "-v", "-z", "--").split("\0")
    hidden = [entry for entry in entries if entry and not entry.startswith("H ")]
    if not hidden:
        return
    preview = ", ".join(f"{entry[:1]}:{entry[2:]}" for entry in hidden[:5])
    if len(hidden) > 5:
        preview += ", ..."
    raise NotaryVerificationError(
        f"{label} uses unsupported Git index flags ({preview}); clear "
        "assume-unchanged/skip-worktree state and materialize the full checkout"
    )


def _require_head_tree_matches_index(
    identity: _CleanGitIdentity,
    entries: Sequence[_GitTreeEntry],
    *,
    label: str,
) -> None:
    """Require an exact stage-zero index projection of the verified HEAD tree."""

    raw_index = _git_bytes(
        identity.root,
        "ls-files",
        "--stage",
        f"--abbrev={len(identity.tree)}",
        "-z",
        "--",
    )
    index_entries: list[tuple[str, str, str, str]] = []
    for record in raw_index.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            raw_mode, raw_object_id, raw_stage = metadata.split(b" ")
            index_entries.append(
                (
                    raw_mode.decode("ascii"),
                    raw_object_id.decode("ascii"),
                    raw_stage.decode("ascii"),
                    raw_path.decode("utf-8"),
                )
            )
        except (UnicodeError, ValueError) as exc:
            raise NotaryVerificationError(
                f"{label} Git index contains an unsafe entry"
            ) from exc
    expected = {
        (
            entry.mode,
            entry.object_id,
            "0",
            entry.relative.as_posix(),
        )
        for entry in entries
    }
    actual = set(index_entries)
    if (
        len(actual) != len(index_entries)
        or any(stage != "0" for _mode, _object_id, stage, _path in index_entries)
        or actual != expected
    ):
        raise NotaryVerificationError(
            f"{label} worktree is dirty; Git index does not exactly match HEAD"
        )


def _require_clean_git_checkout(root: Path, *, label: str) -> _CleanGitIdentity:
    root_identity = _directory_identity(root, label=label)
    try:
        identity_root = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve(
            strict=True
        )
    except OSError as exc:
        raise NotaryVerificationError(
            f"{label} must be the top level of a Git worktree: {root}"
        ) from exc
    if identity_root != root:
        raise NotaryVerificationError(
            f"{label} must be the top level of a Git worktree: {root}"
        )
    commit = _git_text(root, "rev-parse", "--verify", "HEAD^{commit}")
    if _GIT_OID_RE.fullmatch(commit) is None:
        raise NotaryVerificationError(f"{label} HEAD is not a full Git commit SHA")
    tree = _verified_commit_tree(root, commit)
    untracked = tuple(
        path
        for path in _git_stdout(
            root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
        ).split("\0")
        if path
    )
    if untracked:
        raise NotaryVerificationError(
            f"{label} worktree is dirty; commit or remove every tracked and "
            f"untracked change before verification: {root}"
        )
    _require_plain_git_index(root, label=label)
    git_dir = _git_metadata_directory(root, "--absolute-git-dir")
    git_common_dir = _git_metadata_directory(
        root,
        "--path-format=absolute",
        "--git-common-dir",
    )
    git_dir_identity = _directory_identity(
        git_dir,
        label=f"{label} Git directory",
    )
    git_common_dir_identity = _directory_identity(
        git_common_dir,
        label=f"{label} common Git directory",
    )
    clean = _CleanGitIdentity(
        root=root,
        commit=commit,
        tree=tree,
        git_dir=git_dir,
        git_common_dir=git_common_dir,
        root_identity=root_identity,
        git_dir_identity=git_dir_identity,
        git_common_dir_identity=git_common_dir_identity,
    )
    entries = _git_tree_entries(clean)
    _require_head_tree_matches_index(clean, entries, label=label)
    _require_head_tree_matches_worktree(clean, label=label, entries=entries)
    final_identities = (
        _directory_identity(root, label=label),
        _directory_identity(git_dir, label=f"{label} Git directory"),
        _directory_identity(
            git_common_dir,
            label=f"{label} common Git directory",
        ),
    )
    if final_identities != (
        root_identity,
        git_dir_identity,
        git_common_dir_identity,
    ):
        raise NotaryVerificationError(
            f"{label} filesystem identity changed during strict admission"
        )
    return clean


def _tracked_git_paths(root: Path) -> frozenset[str]:
    return frozenset(
        path for path in _git_stdout(root, "ls-files", "-z", "--").split("\0") if path
    )


def _rulespec_yaml_input_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for jurisdiction in jurisdiction_subdir_names(
        root,
        allow_composition_specs=True,
    ):
        content_root = root / jurisdiction
        for candidate in content_root.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in {".yaml", ".yml"}:
                paths.add(candidate.relative_to(root).as_posix())
    return paths


def _require_tracked_rulespec_inputs(
    root: Path,
    *,
    label: str,
    require_toolchain: bool = False,
    require_policyengine_pin: bool = False,
) -> None:
    """Require every live RuleSpec input to be bound by the recorded Git tree."""

    required = _rulespec_yaml_input_paths(root)
    if require_toolchain:
        required.update(
            {
                ".axiom/toolchain.toml",
                str(VALIDATION_WAIVER_SET_PATH),
            }
        )
    if require_policyengine_pin:
        required.add(POLICYENGINE_RUNTIME_PIN_PATH.as_posix())
    missing = sorted(required - _tracked_git_paths(root))
    if missing:
        preview = ", ".join(missing[:5])
        if len(missing) > 5:
            preview += ", ..."
        raise NotaryVerificationError(
            f"{label} verification input is not tracked by HEAD: {preview}"
        )


def _policy_content_identity(root: Path) -> tuple[str, int]:
    identity = _deterministic_tree_identity(
        root,
        excluded_directory_names=frozenset({".git"}),
        hash_symlinks=True,
    )
    tree_sha256 = identity.get("tree_sha256")
    file_count = identity.get("file_count")
    if (
        identity.get("state") != "directory"
        or not isinstance(tree_sha256, str)
        or _SHA256_RE.fullmatch(tree_sha256) is None
        or not isinstance(file_count, int)
    ):
        raise NotaryVerificationError("Cannot hash the RuleSpec checkout contents")
    return tree_sha256, file_count


def _encoder_package_identity() -> dict[str, object]:
    identity = _deterministic_tree_identity(
        Path(__file__).resolve().parent,
        excluded_directory_names=frozenset({"__pycache__"}),
    )
    tree_sha256 = identity.get("tree_sha256")
    file_count = identity.get("file_count")
    if (
        identity.get("state") != "directory"
        or not isinstance(tree_sha256, str)
        or _SHA256_RE.fullmatch(tree_sha256) is None
        or not isinstance(file_count, int)
    ):
        raise NotaryVerificationError("Cannot hash the executing axiom-encode package")
    return {"tree_sha256": tree_sha256, "file_count": file_count}


def _sha256_file(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise NotaryVerificationError(f"{label} is not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise NotaryVerificationError(f"Cannot hash {label}: {path}") from exc
    return {"sha256": digest.hexdigest(), "size": size}


def _axiom_rules_engine_execution_identity(root: Path) -> dict[str, object]:
    """Bind the executable selected by the existing validator search order."""

    try:
        candidate = resolve_axiom_rules_engine_binary(root)
        relative = candidate.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise NotaryVerificationError(str(exc)) from exc
    return {
        "path": relative.as_posix(),
        **_sha256_file(candidate, label="Axiom rules engine executable"),
    }


def _assert_same_clean_checkout(
    expected: _CleanGitIdentity,
    *,
    label: str,
) -> None:
    actual = _require_clean_git_checkout(expected.root, label=label)
    if (
        actual.root_identity != expected.root_identity
        or actual.git_dir_identity != expected.git_dir_identity
        or actual.git_common_dir_identity != expected.git_common_dir_identity
    ):
        raise NotaryVerificationError(
            f"{label} filesystem identity changed during strict verification"
        )
    if actual.commit != expected.commit or actual.tree != expected.tree:
        raise NotaryVerificationError(
            f"{label} commit or tree changed during strict verification"
        )


def _path_is_within_root(path: Path, root: Path) -> bool:
    """Use filesystem identity to handle case aliases and nonexistent leaf paths."""

    existing = path
    while not existing.exists():
        parent = existing.parent
        if parent == existing:
            return False
        existing = parent
    for candidate in (existing, *existing.parents):
        try:
            if candidate.samefile(root):
                return True
        except OSError:
            continue
    return False


def _assert_output_outside_verification_roots(
    receipt_out: Path,
    protected_roots: Sequence[tuple[Path, str]],
) -> Path:
    raw = Path(os.path.abspath(Path(receipt_out).expanduser()))
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise NotaryVerificationError(
                f"Receipt output path contains a symlink: {raw}"
            )
    for root, label in protected_roots:
        if _path_is_within_root(raw, root):
            raise NotaryVerificationError(
                "--receipt-out must be outside every verification input; "
                f"path is inside {label}: {raw}"
            )
    if raw.exists() and not raw.is_file():
        raise NotaryVerificationError(f"Receipt output is not a regular file: {raw}")
    return raw


def _protected_root_identities(
    protected_roots: Sequence[tuple[Path, str]],
) -> tuple[_ProtectedRootIdentity, ...]:
    """Capture stable filesystem identities for every protected input root."""

    identities: list[_ProtectedRootIdentity] = []
    observed: set[tuple[int, int]] = set()
    for root, label in protected_roots:
        try:
            metadata = root.stat()
        except OSError as exc:
            raise NotaryVerificationError(
                f"Cannot bind {label} filesystem identity: {root}"
            ) from exc
        identity = (metadata.st_dev, metadata.st_ino)
        if identity not in observed:
            observed.add(identity)
            identities.append((*identity, label))
    return tuple(identities)


def _is_primary_rulespec_module(path: Path, *, checkout: Path) -> bool:
    content_root = find_policy_repo_root(path)
    if content_root is None or content_root.parent != checkout:
        return False
    try:
        relative = path.relative_to(content_root)
    except ValueError:
        return False
    return (
        path.suffix == RULESPEC_FILE_SUFFIX
        and not path.name.endswith(RULESPEC_TEST_FILE_SUFFIX)
        and len(relative.parts) >= 2
        and relative.parts[0] in RULESPEC_PROTECTED_ATOMIC_MODULE_ROOTS
    )


def _protected_rulespec_root_index(relative: PurePosixPath) -> int | None:
    """Match the guard's protected atomic-YAML path boundary, including legacy."""

    parts = relative.parts
    if len(parts) < 2:
        return None
    if parts[0] in RULESPEC_PROTECTED_ATOMIC_MODULE_ROOTS:
        return 0
    if (
        len(parts) >= 3
        and parts[1] in RULESPEC_PROTECTED_ATOMIC_MODULE_ROOTS
        and re.fullmatch(r"[a-z]{2}(?:-[a-z0-9_]+)*", parts[0])
    ):
        return 1
    return None


def _is_protected_rulespec_relative(relative: PurePosixPath) -> bool:
    return (
        relative.suffix == RULESPEC_FILE_SUFFIX
        and _protected_rulespec_root_index(relative) is not None
    )


def _authority_surface_path_reason(relative: PurePosixPath) -> str | None:
    """Return the authority class for one path, independent of its mode."""

    relative_text = relative.as_posix()
    if ".axiom" in relative.parts:
        return "protected verifier, trust-root, or repository configuration"
    if relative.name == ".gitignore":
        return None
    if relative.parts and relative.parts[0] == ".github":
        return "protected GitHub workflow or repository configuration"
    if relative.name == ".gitattributes":
        return "protected per-directory Git attributes"
    if relative_text in _AUTHORITY_SURFACE_PATHS:
        return "protected repository or waiver-policy configuration"
    return None


def _authority_surface_change_reason(
    relative: PurePosixPath,
    *,
    base_entry: _GitTreeEntry | None,
    head_entry: _GitTreeEntry | None,
) -> str | None:
    """Return why trusted preflight must reject one changed Git leaf."""

    for entry in (base_entry, head_entry):
        if entry is None:
            continue
        if entry.mode == "120000":
            return "symlink changes require a separately privileged flow"
        if entry.mode == "100755":
            return "executable changes require a separately privileged flow"
    return _authority_surface_path_reason(relative)


def _whole_repo_structural_anomaly_reason(
    checkout: Path,
    entry: _GitTreeEntry,
) -> str | None:
    """Return a protected-content structural anomaly, if one is present."""

    if _protected_rulespec_root_index(entry.relative) is None:
        return None
    if entry.mode == "120000":
        return "symlink"
    if entry.mode != "100644":
        return "non-regular or privileged file mode"
    path = checkout.joinpath(*entry.relative.parts)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise NotaryVerificationError(
            "Cannot inspect protected-content file mode: "
            f"{entry.relative.as_posix()}"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & _PRIVILEGED_FILE_MODE_MASK
    ):
        return "non-regular or privileged file mode"
    return None


def _assert_whole_repo_structural_integrity(
    policy_git: _CleanGitIdentity,
    *,
    entries: Sequence[_GitTreeEntry] | None = None,
) -> None:
    """Fail closed on structural anomalies inside protected content trees."""

    admitted_entries = (
        entries if entries is not None else _git_tree_entries(policy_git)
    )
    for entry in admitted_entries:
        reason = _whole_repo_structural_anomaly_reason(policy_git.root, entry)
        if reason is not None:
            raise NotaryVerificationError(
                "Whole-repository verification refuses structural anomaly in "
                "protected content tree "
                f"{entry.relative.as_posix()}: {reason}"
            )


def _primary_relative_for_target(relative: PurePosixPath) -> PurePosixPath:
    if not relative.name.endswith(RULESPEC_TEST_FILE_SUFFIX):
        return relative
    primary_name = (
        relative.name.removesuffix(RULESPEC_TEST_FILE_SUFFIX) + RULESPEC_FILE_SUFFIX
    )
    return relative.with_name(primary_name)


def _companion_relative_for_primary(relative: PurePosixPath) -> PurePosixPath:
    return relative.with_name(
        relative.name.removesuffix(RULESPEC_FILE_SUFFIX) + RULESPEC_TEST_FILE_SUFFIX
    )


def _resolve_verification_target(
    relative: PurePosixPath,
    *,
    checkout: Path,
) -> Path:
    primary_relative = _primary_relative_for_target(relative)
    candidate = checkout.joinpath(*primary_relative.parts)
    if candidate.is_symlink():
        raise NotaryVerificationError(
            f"Protected RuleSpec target is a symlink: {relative.as_posix()}"
        )
    try:
        candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise NotaryVerificationError(
            "Protected RuleSpec target has no surviving primary module: "
            f"{relative.as_posix()}"
        ) from exc
    if not candidate.is_file():
        raise NotaryVerificationError(
            f"Protected RuleSpec target is not a regular file: {relative.as_posix()}"
        )
    if not _is_primary_rulespec_module(candidate, checkout=checkout):
        raise NotaryVerificationError(
            "Protected target must resolve to a primary atomic RuleSpec module "
            f"or its companion test: {relative.as_posix()}"
        )
    return candidate


def _verified_base_identity(
    policy_git: _CleanGitIdentity,
    base_commit: str,
) -> _CleanGitIdentity:
    if (
        not isinstance(base_commit, str)
        or _GIT_OID_RE.fullmatch(base_commit) is None
        or len(base_commit) != len(policy_git.commit)
    ):
        raise NotaryVerificationError(
            "--base-commit must be a full lowercase SHA for this repository"
        )
    if base_commit == policy_git.commit:
        raise NotaryVerificationError(
            "Diff verification refuses --base-commit equal to HEAD; "
            "use --whole-repo for epoch-coverage backfill"
        )
    try:
        tree = _verified_commit_tree(policy_git.root, base_commit)
    except NotaryVerificationError as exc:
        raise NotaryVerificationError(
            "--base-commit must identify a present Git commit object"
        ) from exc
    _assert_verified_commit_ancestor(
        policy_git.root,
        base_commit=base_commit,
        head_commit=policy_git.commit,
    )
    return _CleanGitIdentity(
        root=policy_git.root,
        commit=base_commit,
        tree=tree,
        git_dir=policy_git.git_dir,
        git_common_dir=policy_git.git_common_dir,
        root_identity=policy_git.root_identity,
        git_dir_identity=policy_git.git_dir_identity,
        git_common_dir_identity=policy_git.git_common_dir_identity,
    )


def _whole_repo_target_set(
    policy_git: _CleanGitIdentity,
) -> _NotaryTargetSet:
    """Derive backfill-only targets after a fail-closed whole-tree preflight."""

    checkout = policy_git.root
    entries = _git_tree_entries(policy_git)
    _assert_whole_repo_structural_integrity(policy_git, entries=entries)
    authority_surfaces = tuple(
        entry.relative
        for entry in entries
        if _authority_surface_path_reason(entry.relative) is not None
    )
    try:
        modules = atomic_rulespec_module_paths(
            checkout,
            allow_symlinks_outside_atomic_roots=True,
        )
    except ValueError as exc:
        raise NotaryVerificationError(str(exc)) from exc
    if not modules:
        raise NotaryVerificationError(
            "Whole-repository verification found no atomic RuleSpec modules"
        )
    files: list[_NotaryTargetFile] = []
    verification_targets: set[PurePosixPath] = set()
    for entry in entries:
        if not _is_protected_rulespec_relative(entry.relative):
            continue
        if entry.mode != "100644":
            raise NotaryVerificationError(
                "Protected RuleSpec targets must be non-executable regular files: "
                f"{entry.relative.as_posix()}"
            )
        target = _resolve_verification_target(
            entry.relative,
            checkout=checkout,
        )
        verification_targets.add(PurePosixPath(target.relative_to(checkout).as_posix()))
        files.append(_NotaryTargetFile(entry.relative, "verified"))
    expected_modules = {
        PurePosixPath(module.relative_to(checkout).as_posix()) for module in modules
    }
    if verification_targets != expected_modules:
        raise NotaryVerificationError(
            "Whole-repository target coverage does not match the strict layout scan"
        )
    return _NotaryTargetSet(
        mode="whole-repo",
        base_commit=None,
        files=tuple(files),
        verification_targets=tuple(sorted(verification_targets)),
        authority_surfaces=authority_surfaces,
    )


def _diff_target_set(
    policy_git: _CleanGitIdentity,
    *,
    base_commit: str,
) -> _NotaryTargetSet:
    base_git = _verified_base_identity(policy_git, base_commit)
    base_entries = {entry.relative: entry for entry in _git_tree_entries(base_git)}
    head_entries = {entry.relative: entry for entry in _git_tree_entries(policy_git)}
    changed = tuple(
        sorted(
            relative
            for relative in base_entries.keys() | head_entries.keys()
            if base_entries.get(relative) != head_entries.get(relative)
        )
    )
    if not changed:
        raise NotaryVerificationError(
            "Diff verification found no changed Git leaves between "
            "--base-commit and HEAD; empty-diff receipts are forbidden; "
            "use --whole-repo for epoch-coverage backfill"
        )
    files: list[_NotaryTargetFile] = []
    verification_targets: set[PurePosixPath] = set()
    for relative in changed:
        base_entry = base_entries.get(relative)
        head_entry = head_entries.get(relative)
        authority_reason = _authority_surface_change_reason(
            relative,
            base_entry=base_entry,
            head_entry=head_entry,
        )
        if authority_reason is not None:
            raise NotaryVerificationError(
                "Diff verification refuses authority-surface change "
                f"{relative.as_posix()}: {authority_reason}"
            )
        if not _is_protected_rulespec_relative(relative):
            files.append(_NotaryTargetFile(relative, "out-of-scope"))
            continue
        if head_entry is None:
            files.append(_NotaryTargetFile(relative, "deleted"))
            primary_relative = _primary_relative_for_target(relative)
            if relative.name.endswith(RULESPEC_TEST_FILE_SUFFIX):
                primary_entry = head_entries.get(primary_relative)
                if primary_entry is not None:
                    if primary_entry.mode != "100644":
                        raise NotaryVerificationError(
                            "Protected RuleSpec targets must be non-executable "
                            f"regular files: {primary_relative.as_posix()}"
                        )
                    target = _resolve_verification_target(
                        primary_relative,
                        checkout=policy_git.root,
                    )
                    verification_targets.add(
                        PurePosixPath(target.relative_to(policy_git.root).as_posix())
                    )
            elif _companion_relative_for_primary(relative) in head_entries:
                raise NotaryVerificationError(
                    "A deleted primary RuleSpec module leaves a surviving "
                    f"companion test: {relative.as_posix()}"
                )
            continue
        if head_entry.mode != "100644":
            raise NotaryVerificationError(
                "Protected RuleSpec targets must be non-executable regular files: "
                f"{relative.as_posix()}"
            )
        primary_relative = _primary_relative_for_target(relative)
        primary_entry = head_entries.get(primary_relative)
        if primary_entry is None or primary_entry.mode != "100644":
            raise NotaryVerificationError(
                "Protected RuleSpec targets require a non-executable regular "
                f"primary module: {primary_relative.as_posix()}"
            )
        target = _resolve_verification_target(
            relative,
            checkout=policy_git.root,
        )
        verification_targets.add(
            PurePosixPath(target.relative_to(policy_git.root).as_posix())
        )
        files.append(_NotaryTargetFile(relative, "verified"))
    return _NotaryTargetSet(
        mode="diff",
        base_commit=base_commit,
        files=tuple(files),
        verification_targets=tuple(sorted(verification_targets)),
        authority_surfaces=(),
    )


def _resolve_notary_target_set(
    policy_git: _CleanGitIdentity,
    *,
    base_commit: str | None,
    whole_repo: bool,
) -> _NotaryTargetSet:
    """Derive the complete target set from admitted Git objects."""

    if whole_repo:
        if base_commit is not None:
            raise NotaryVerificationError("--whole-repo does not accept --base-commit")
        return _whole_repo_target_set(policy_git)
    if base_commit is None:
        raise NotaryVerificationError("Diff verification requires --base-commit")
    return _diff_target_set(policy_git, base_commit=base_commit)


def _pipeline_issue_strings(
    relative_target: str,
    result: PipelineResult,
) -> list[str]:
    issues: list[str] = []
    for name, validation in sorted(result.results.items()):
        if validation.passed:
            continue
        details = list(validation.issues or ())
        if not details and validation.error:
            details = [validation.error]
        if not details:
            details = ["gate failed without a diagnostic"]
        issues.extend(f"{relative_target} [{name}]: {detail}" for detail in details)
    return issues


def _result_gate_passed(results: Sequence[PipelineResult], gate: str) -> bool:
    return bool(results) and all(
        gate in result.results and result.results[gate].passed for result in results
    )


def _deterministic_gate_passed(
    results: Sequence[PipelineResult],
    gate: str,
) -> bool:
    if not results:
        return False
    for result in results:
        ci_result = result.results.get("ci")
        if ci_result is None:
            return False
        outcomes = ci_result.details.get("deterministic_gates")
        if not isinstance(outcomes, dict) or outcomes.get(gate) is not True:
            return False
    return True


def _ci_compile_passed(results: Sequence[PipelineResult]) -> bool:
    if not results:
        return False
    for result in results:
        ci_result = result.results.get("ci")
        if ci_result is None or ci_result.details.get("compile_passed") is not True:
            return False
    return True


def _oracle_passed(results: Sequence[PipelineResult]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for result in results:
        oracle = result.results.get("policyengine")
        if oracle is None:
            issues.append("PolicyEngine oracle result is missing")
            continue
        if not oracle.passed:
            issues.append("PolicyEngine oracle reported failure")
        if oracle.score is None or oracle.score < MIN_POLICYENGINE_MATCH:
            rendered = (
                "no comparable score" if oracle.score is None else f"{oracle.score:.1%}"
            )
            issues.append(
                "PolicyEngine oracle score "
                f"{rendered} is below {MIN_POLICYENGINE_MATCH:.0%}"
            )
        coverage = oracle.details.get("coverage")
        if not isinstance(coverage, dict):
            issues.append("PolicyEngine oracle coverage evidence is missing")
        else:
            comparable = int(coverage.get("comparable", 0) or 0)
            for field, description in (
                ("unmapped", "unclassified output"),
                ("unsupported", "unsupported output"),
                ("adapter_errors", "adapter execution error"),
                ("setup_errors", "setup error"),
            ):
                count = int(coverage.get(field, 0) or 0)
                if count:
                    issues.append(f"PolicyEngine oracle had {count} {description}(s)")
            if comparable < 1:
                issues.append(
                    "PolicyEngine oracle produced no comparable output evidence"
                )
        issues.extend(
            str(issue)
            for issue in oracle.issues
            if str(issue).startswith("PolicyEngine unavailable")
        )
    return not issues, issues


def _portable_policyengine_identity(
    runtime: PolicyEngineRuntime | None,
) -> dict[str, object] | None:
    if runtime is None:
        return None
    identity = runtime.canonical_identity()
    fields = (
        "schema",
        "country",
        "trusted_git_commit",
        "official_tree_sha256",
        "checkout_execution_tree_sha256",
        "venv_execution_tree_sha256",
        "pyproject_sha256",
        "uv_lock_sha256",
        "locked_versions",
    )
    portable = {field: identity[field] for field in fields if field in identity}
    portable["identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(portable)
    ).hexdigest()
    return portable


def _require_verification_only_broker() -> None:
    """Refuse Job 1 when its corpus trust broker can also sign."""

    try:
        broker = get_signing_broker()
        capabilities = frozenset(broker.capabilities)
    except (AttributeError, SigningBrokerError, TypeError) as exc:
        raise NotaryVerificationError(
            "Notary verification requires an attached verification-only broker"
        ) from exc
    if capabilities:
        rendered = ", ".join(sorted(str(item) for item in capabilities))
        raise NotaryVerificationError(
            "Notary verification requires a verification-only broker; signing "
            f"capabilities are forbidden: {rendered}"
        )


def _open_receipt_parent(
    path: Path,
    *,
    protected_identities: Sequence[_ProtectedRootIdentity],
) -> int:
    """Open a pre-existing output parent without following mutable symlinks."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise NotaryVerificationError(
            "Strict receipt publication requires no-follow directory opens"
        )
    flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path.anchor, flags)
        _assert_open_directory_outside_protected_roots(
            descriptor,
            protected_identities,
        )
        for part in path.parent.parts[1:]:
            try:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError as exc:
                raise NotaryVerificationError(
                    f"Notary receipt output parent must already exist: {path.parent}"
                ) from exc
            os.close(descriptor)
            descriptor = next_descriptor
            _assert_open_directory_outside_protected_roots(
                descriptor,
                protected_identities,
            )
        return descriptor
    except NotaryVerificationError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise NotaryVerificationError(
            f"Cannot safely open notary receipt output parent: {path.parent}"
        ) from exc


def _assert_open_directory_outside_protected_roots(
    descriptor: int,
    protected_identities: Sequence[_ProtectedRootIdentity],
) -> None:
    """Check the opened directory and its live ancestors by inode identity."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise NotaryVerificationError(
            "Strict receipt publication requires no-follow directory opens"
        )
    protected = {
        (device, inode): label for device, inode, label in protected_identities
    }
    flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    current = os.dup(descriptor)
    try:
        for _depth in range(4096):
            metadata = os.fstat(current)
            label = protected.get((metadata.st_dev, metadata.st_ino))
            if label is not None:
                raise NotaryVerificationError(
                    "Refusing to publish the receipt inside a protected "
                    f"verification root: {label}"
                )
            parent = os.open("..", flags, dir_fd=current)
            try:
                parent_metadata = os.fstat(parent)
            except BaseException:
                os.close(parent)
                raise
            if (parent_metadata.st_dev, parent_metadata.st_ino) == (
                metadata.st_dev,
                metadata.st_ino,
            ):
                os.close(parent)
                return
            os.close(current)
            current = parent
        raise NotaryVerificationError(
            "Cannot prove the receipt output directory ancestry is finite"
        )
    except OSError as exc:
        raise NotaryVerificationError(
            "Cannot verify the opened notary receipt output directory"
        ) from exc
    finally:
        os.close(current)


def _write_receipt(
    path: Path,
    receipt: Mapping[str, Any],
    *,
    protected_identities: Sequence[_ProtectedRootIdentity],
    publication_check: Callable[[], None],
) -> None:
    """Atomically publish a receipt into a trusted external output directory.

    The output host is trusted to honor descriptor-relative replacement and
    unlink operations.  Cleanup after a rejected publication is best effort:
    the original verification error remains authoritative, and an unlink
    failure emits a stderr warning because the rejected receipt may remain.
    """

    raw = canonical_receipt_bytes(receipt)
    parent_descriptor: int | None = None
    temporary_name: str | None = None
    published = False
    complete = False
    has_primary_error = False
    try:
        parent_descriptor = _open_receipt_parent(
            path,
            protected_identities=protected_identities,
        )
        _assert_open_directory_outside_protected_roots(
            parent_descriptor,
            protected_identities,
        )
        descriptor: int | None = None
        for _attempt in range(10):
            candidate = f".{path.name}.{secrets.token_hex(16)}"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if descriptor is None or temporary_name is None:
            raise NotaryVerificationError(
                "Cannot allocate a unique notary receipt temporary file"
            )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_open_directory_outside_protected_roots(
            parent_descriptor,
            protected_identities,
        )
        publication_check()
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = None
        published = True
        _assert_open_directory_outside_protected_roots(
            parent_descriptor,
            protected_identities,
        )
        publication_check()
        complete = True
    except OSError as exc:
        has_primary_error = True
        raise NotaryVerificationError(f"Cannot write notary receipt: {path}") from exc
    except BaseException:
        has_primary_error = True
        raise
    finally:
        if parent_descriptor is not None:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=parent_descriptor)
                except OSError as exc:
                    print(
                        "notary-verify: warning: failed to remove rejected "
                        f"temporary receipt {path.parent / temporary_name}: {exc!r}",
                        file=sys.stderr,
                    )
            if published and not complete:
                try:
                    os.unlink(path.name, dir_fd=parent_descriptor)
                except OSError as exc:
                    print(
                        "notary-verify: warning: failed to remove rejected "
                        f"published receipt {path}: {exc!r}",
                        file=sys.stderr,
                    )
            try:
                os.close(parent_descriptor)
            except OSError as exc:
                if not has_primary_error:
                    raise NotaryVerificationError(
                        "Cannot close notary receipt output directory: "
                        f"{path.parent}"
                    ) from exc
                print(
                    "notary-verify: warning: failed to close the receipt output "
                    f"directory while preserving the primary error: {exc!r}",
                    file=sys.stderr,
                )


def _utc_timestamp(now: datetime | None = None) -> str:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise NotaryVerificationError("Notary receipt timestamp must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _run_materialized_verification(
    *,
    policy_git: _CleanGitIdentity,
    dependency_git: Sequence[_CleanGitIdentity],
    corpus_root: Path,
    engine_root: Path,
    engine_execution: Mapping[str, object],
    target_relatives: Sequence[PurePosixPath],
    runtime_root: Path | None,
    allow_reduced: bool,
    waiver_date: date,
) -> _MaterializedVerificationEvidence:
    """Validate immutable raw Git-tree snapshots and a hash-bound engine copy."""

    oracle_reduced = runtime_root is None
    if oracle_reduced and not allow_reduced:
        raise NotaryVerificationError(
            "PolicyEngine oracle dependency is absent; strict verification "
            "fails closed unless --allow-reduced is passed"
        )

    with tempfile.TemporaryDirectory(prefix="axiom-notary-verification-") as raw_tmp:
        workspace = Path(raw_tmp).resolve(strict=True)
        policy_root = workspace / "subject" / policy_git.root.name
        _materialize_git_tree(policy_git, policy_root)
        policy_root = policy_root.resolve(strict=True)
        dependency_roots: list[Path] = []
        for index, identity in enumerate(dependency_git):
            snapshot = workspace / "dependencies" / str(index) / identity.root.name
            _materialize_git_tree(identity, snapshot)
            dependency_roots.append(snapshot.resolve(strict=True))
        engine_snapshot = workspace / "engine" / engine_root.name
        _copy_verified_engine_binary(
            engine_root,
            engine_snapshot,
            engine_execution,
        )
        engine_snapshot = engine_snapshot.resolve(strict=True)

        try:
            checkout_modules = atomic_rulespec_module_paths(
                policy_root,
                allow_symlinks_outside_atomic_roots=True,
            )
        except ValueError as exc:
            raise NotaryVerificationError(str(exc)) from exc
        if target_relatives and not checkout_modules:
            raise NotaryVerificationError(
                "Strict verification found no atomic RuleSpec modules"
            )
        targets = tuple(
            _resolve_verification_target(relative, checkout=policy_root)
            for relative in target_relatives
        )
        observed_relatives = tuple(
            PurePosixPath(target.relative_to(policy_root).as_posix())
            for target in targets
        )
        if observed_relatives != tuple(target_relatives):
            raise NotaryVerificationError(
                "Materialized target set does not match the admitted checkout"
            )

        waiver_sha256 = verify_rulespec_validation_waiver_set(policy_root)
        waivers = load_validation_waivers(
            policy_root / VALIDATION_WAIVER_SET_PATH,
            repo_root=policy_root,
            today=waiver_date,
        )
        corpus_release = load_rulespec_local_corpus_release(policy_root, corpus_root)

        module_content_roots = {
            module: policy_root / module.relative_to(policy_root).parts[0]
            for module in checkout_modules
        }
        try:
            admitted_content_roots = tuple(
                dict.fromkeys(module_content_roots[target] for target in targets)
            )
        except KeyError as exc:
            raise NotaryVerificationError(
                "A target is absent from the strict RuleSpec layout scan"
            ) from exc
        identity_content_roots = {
            content_root: policy_git.root / content_root.relative_to(policy_root)
            for content_root in admitted_content_roots
        }

        policyengine_runtime: PolicyEngineRuntime | None = None
        if runtime_root is not None:
            if admitted_content_roots:
                identity_roots = tuple(identity_content_roots.values())
            else:
                identity_roots = tuple(
                    dict.fromkeys(
                        policy_git.root / module.relative_to(policy_root).parts[0]
                        for module in checkout_modules
                    )
                )
            if not identity_roots:
                raise NotaryVerificationError(
                    "PolicyEngine runtime cannot be bound because the verified "
                    "checkout has no surviving atomic RuleSpec content root"
                )
            policyengine_runtime = PolicyEngineRuntime.for_rulespec_root(
                runtime_root,
                policy_repo_root=identity_roots[0],
            )
            for identity_root in identity_roots[1:]:
                policyengine_runtime.assert_matches_rulespec_root(identity_root)
            policyengine_runtime.assert_unchanged()

        pipelines: dict[Path, ValidatorPipeline] = {}
        pipeline_results: list[PipelineResult] = []
        issues: list[str] = []
        for target in targets:
            content_root = module_content_roots[target]
            pipeline = pipelines.get(content_root)
            if pipeline is None:
                pipeline = ValidatorPipeline(
                    policy_repo_path=content_root,
                    axiom_rules_path=engine_snapshot,
                    enable_oracles=policyengine_runtime is not None,
                    oracle_validators=("policyengine",)
                    if policyengine_runtime is not None
                    else (),
                    policyengine_runtime=policyengine_runtime,
                    policyengine_rulespec_identity_root=identity_content_roots[
                        content_root
                    ],
                    require_policy_proofs=True,
                    enforce_repository_layout=True,
                    local_corpus_release=corpus_release,
                    rulespec_dependency_roots=tuple(dependency_roots),
                    expose_deterministic_gate_evidence=True,
                )
                pipelines[content_root] = pipeline
            result = pipeline.validate(target, skip_reviewers=True)
            pipeline_results.append(result)
            relative_target = target.relative_to(policy_root).as_posix()
            issues.extend(_pipeline_issue_strings(relative_target, result))

        if pipeline_results:
            compile_passed = _result_gate_passed(
                pipeline_results,
                "compile",
            ) and _ci_compile_passed(pipeline_results)
            ci_passed = _result_gate_passed(pipeline_results, "ci")
            deterministic_statuses = {
                gate: _deterministic_gate_passed(pipeline_results, gate)
                for gate in (
                    "proof-revalidation",
                    "companion-tests",
                    "grounding-contract",
                    "layout-inspection",
                )
            }
        else:
            compile_passed = True
            ci_passed = True
            deterministic_statuses = {
                gate: True
                for gate in (
                    "proof-revalidation",
                    "companion-tests",
                    "grounding-contract",
                    "layout-inspection",
                )
            }
        for gate, passed in deterministic_statuses.items():
            if not passed and ci_passed:
                issues.append(
                    f"{gate}: deterministic gate did not complete successfully"
                )
        if policyengine_runtime is None:
            oracle_passed = True
            oracle_status: GateStatus = "reduced"
        else:
            oracle_passed, oracle_issues = _oracle_passed(pipeline_results)
            issues.extend(f"policyengine-oracle: {issue}" for issue in oracle_issues)
            oracle_status = "passed" if oracle_passed else "failed"

        validators_passed = (
            compile_passed
            and ci_passed
            and all(deterministic_statuses.values())
            and oracle_passed
            and all(result.all_passed for result in pipeline_results)
        )
        return _MaterializedVerificationEvidence(
            corpus_release_name=corpus_release.name,
            corpus_release_content_sha256=corpus_release.content_sha256,
            waiver_sha256=waiver_sha256,
            waiver_count=len(waivers.active_paths),
            policyengine_runtime=policyengine_runtime,
            compile_passed=compile_passed,
            deterministic_statuses=deterministic_statuses,
            oracle_status=oracle_status,
            validators_passed=validators_passed,
            issues=tuple(issues),
        )


def run_notary_verification(
    *,
    policy_repo_path: Path,
    corpus_path: Path,
    axiom_rules_engine_path: Path,
    receipt_out: Path,
    base_commit: str | None,
    whole_repo: bool = False,
    policyengine_runtime_root: Path | None = None,
    rulespec_dependency_roots: Sequence[Path] = (),
    allow_reduced: bool = False,
    now: datetime | None = None,
) -> NotaryVerificationResult:
    """Run the strict profile and emit one unsigned, content-addressed receipt.

    In diff mode, the trusted workflow must resolve the protected branch and
    bind its full commit SHA as ``base_commit``.  Treating caller input as the
    protected base would move a trust decision outside this verifier.

    ``whole_repo`` is only a backfill instrument for epoch coverage of the
    exact subject tree.  Its receipt has an explicit null ``base_commit`` and
    ``targets.mode == "whole-repo"`` and must never authorize a diff.
    """

    timestamp = _utc_timestamp(now)
    waiver_date = datetime.fromisoformat(timestamp.removesuffix("Z") + "+00:00").date()
    _require_verification_only_broker()
    policy_root = _resolve_policy_checkout(policy_repo_path)
    policy_git = _require_clean_git_checkout(
        policy_root,
        label="RuleSpec checkout",
    )
    target_set = _resolve_notary_target_set(
        policy_git,
        base_commit=base_commit,
        whole_repo=whole_repo,
    )
    _require_tracked_rulespec_inputs(
        policy_root,
        label="RuleSpec checkout",
        require_toolchain=True,
        require_policyengine_pin=policyengine_runtime_root is not None,
    )
    policy_content_before = _policy_content_identity(policy_root)
    corpus_root = _resolve_existing_directory(corpus_path, label="Axiom Corpus")
    engine_root = _resolve_existing_directory(
        axiom_rules_engine_path,
        label="Axiom rules engine",
    )
    engine_git = _require_clean_git_checkout(
        engine_root,
        label="Axiom rules engine",
    )
    engine_execution_before = _axiom_rules_engine_execution_identity(engine_root)

    dependency_roots = tuple(
        _resolve_policy_checkout(root) for root in rulespec_dependency_roots
    )
    if len(set(dependency_roots)) != len(dependency_roots):
        raise NotaryVerificationError(
            "--rulespec-dependency-root contains a duplicate checkout"
        )
    if policy_root in dependency_roots:
        raise NotaryVerificationError(
            "The verified policy checkout cannot also be a dependency root"
        )
    dependency_git = tuple(
        _require_clean_git_checkout(root, label="RuleSpec dependency checkout")
        for root in dependency_roots
    )
    for root in dependency_roots:
        _require_tracked_rulespec_inputs(
            root,
            label="RuleSpec dependency checkout",
        )

    runtime_root: Path | None = None
    runtime_git: _CleanGitIdentity | None = None
    if policyengine_runtime_root is not None:
        runtime_root = _resolve_existing_directory(
            policyengine_runtime_root,
            label="PolicyEngine runtime",
        )
        runtime_git = _require_clean_git_checkout(
            runtime_root,
            label="PolicyEngine runtime",
        )
    git_identities: tuple[tuple[_CleanGitIdentity, str], ...] = (
        (policy_git, "the verified RuleSpec checkout"),
        (engine_git, "the Axiom rules engine checkout"),
        *((identity, "a RuleSpec dependency checkout") for identity in dependency_git),
        *(((runtime_git, "the PolicyEngine runtime"),) if runtime_git else ()),
    )
    protected_roots = (
        (policy_root, "the verified RuleSpec checkout"),
        (corpus_root, "the Axiom Corpus root"),
        (engine_root, "the Axiom rules engine checkout"),
        *((root, "a RuleSpec dependency checkout") for root in dependency_roots),
        *(((runtime_root, "the PolicyEngine runtime"),) if runtime_root else ()),
        (Path(__file__).resolve().parent, "the executing axiom-encode package"),
        *(
            (metadata_root, f"{label} Git metadata")
            for identity, label in git_identities
            for metadata_root in (identity.git_dir, identity.git_common_dir)
        ),
    )
    protected_identities = _protected_root_identities(protected_roots)
    output_path = _assert_output_outside_verification_roots(
        receipt_out,
        protected_roots,
    )

    encoder_package_before = _encoder_package_identity()
    evidence = _run_materialized_verification(
        policy_git=policy_git,
        dependency_git=dependency_git,
        corpus_root=corpus_root,
        engine_root=engine_root,
        engine_execution=engine_execution_before,
        target_relatives=target_set.verification_targets,
        runtime_root=runtime_root,
        allow_reduced=allow_reduced,
        waiver_date=waiver_date,
    )
    policyengine_runtime = evidence.policyengine_runtime

    def assert_verification_inputs_unchanged() -> None:
        _assert_same_clean_checkout(policy_git, label="RuleSpec checkout")
        if target_set.mode == "whole-repo":
            _assert_whole_repo_structural_integrity(policy_git)
        _require_tracked_rulespec_inputs(
            policy_root,
            label="RuleSpec checkout",
            require_toolchain=True,
            require_policyengine_pin=policyengine_runtime is not None,
        )
        if _policy_content_identity(policy_root) != policy_content_before:
            raise NotaryVerificationError(
                "Strict verification mutated files inside the policy repository"
            )
        _assert_same_clean_checkout(engine_git, label="Axiom rules engine")
        if (
            _axiom_rules_engine_execution_identity(engine_root)
            != engine_execution_before
        ):
            raise NotaryVerificationError(
                "The Axiom rules engine executable changed during verification"
            )
        for identity in dependency_git:
            _assert_same_clean_checkout(
                identity,
                label="RuleSpec dependency checkout",
            )
            _require_tracked_rulespec_inputs(
                identity.root,
                label="RuleSpec dependency checkout",
            )
        if policyengine_runtime is not None:
            policyengine_runtime.assert_unchanged()
        if runtime_git is not None:
            _assert_same_clean_checkout(runtime_git, label="PolicyEngine runtime")
        if _encoder_package_identity() != encoder_package_before:
            raise NotaryVerificationError(
                "The executing axiom-encode package changed during verification"
            )

    assert_verification_inputs_unchanged()

    gate_outcomes = [
        _gate("subject-clean", "passed"),
        _gate("corpus-release-binding", "passed"),
        _gate("compile", "passed" if evidence.compile_passed else "failed"),
        *[
            _gate(
                gate,
                "passed" if evidence.deterministic_statuses[gate] else "failed",
            )
            for gate in (
                "proof-revalidation",
                "companion-tests",
                "grounding-contract",
                "layout-inspection",
            )
        ],
        _gate("waiver-set-verification", "passed"),
        _gate("policyengine-oracle", evidence.oracle_status),
        _gate("policy-repo-nonmutation", "passed"),
    ]

    status = "passed"
    if not evidence.validators_passed:
        status = "failed"
    elif runtime_root is None:
        status = "passed-reduced"
    body: dict[str, Any] = {
        "schema_id": NOTARY_RECEIPT_SCHEMA_ID,
        "schema_status": NOTARY_RECEIPT_SCHEMA_STATUS,
        "status": status,
        "subject_tree": policy_git.tree,
        "subject_commit": policy_git.commit,
        "base_commit": target_set.base_commit,
        "targets": {
            "mode": target_set.mode,
            "files": [
                {
                    "path": target.relative.as_posix(),
                    "disposition": target.disposition,
                }
                for target in target_set.files
            ],
        },
        "dependencies": {
            "corpus_release": {
                "name": evidence.corpus_release_name,
                "content_sha256": evidence.corpus_release_content_sha256,
            },
            "axiom_rules_engine": {
                "commit": engine_git.commit,
                "executable": engine_execution_before,
            },
            "axiom_encode": {
                "package": "axiom-encode",
                "version": __version__,
                "package_identity": encoder_package_before,
            },
            "policyengine_oracle": _portable_policyengine_identity(
                policyengine_runtime
            ),
            "rulespec_dependencies": [
                {
                    "repository": canonical_rulespec_repo_name(identity.root),
                    "commit": identity.commit,
                    "tree": identity.tree,
                }
                for identity in dependency_git
            ],
        },
        "waiver_set_sha256": evidence.waiver_sha256,
        "waiver_count": evidence.waiver_count,
        "gates": gate_outcomes,
        "run": {
            "encoder_version": __version__,
            "profile_id": NOTARY_PROFILE_ID,
            "timestamp": timestamp,
        },
    }
    if target_set.mode == "whole-repo":
        body["authority_surfaces"] = [
            {"path": relative.as_posix()}
            for relative in target_set.authority_surfaces
        ]
    receipt = attach_receipt_sha256(body)
    output_path = _assert_output_outside_verification_roots(
        output_path,
        protected_roots,
    )
    _write_receipt(
        output_path,
        receipt,
        protected_identities=protected_identities,
        publication_check=assert_verification_inputs_unchanged,
    )
    return NotaryVerificationResult(
        receipt=receipt,
        passed=evidence.validators_passed,
        issues=evidence.issues,
    )


__all__ = [
    "MIN_POLICYENGINE_MATCH",
    "NOTARY_PROFILE_ID",
    "NOTARY_RECEIPT_SCHEMA_ID",
    "NOTARY_RECEIPT_SCHEMA_STATUS",
    "NotaryVerificationError",
    "NotaryVerificationResult",
    "attach_receipt_sha256",
    "canonical_json_bytes",
    "canonical_receipt_body_bytes",
    "canonical_receipt_bytes",
    "gate_reproducibility",
    "receipt_body_sha256",
    "run_notary_verification",
]
