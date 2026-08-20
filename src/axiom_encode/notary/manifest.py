"""Raw-Git tree manifests for schema-independent notary verification."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

from .canonical import body_digest
from .refusal import Refusal

type ManifestEntry = tuple[str, str, str]
type Manifest = list[ManifestEntry]

_DIRECTORY_MODE = b"40000"
_GITLINK_MODE = b"160000"
_SYMLINK_MODE = b"120000"
_TERMINAL_MODES = {b"100644", b"100755"}


class ManifestDiffEntry(NamedTuple):
    """One per-path manifest addition, deletion, or modification."""

    path: str
    before_mode: str | None
    before_entry_sha256: str | None
    after_mode: str | None
    after_entry_sha256: str | None


@dataclass(frozen=True, slots=True)
class _RawTreeEntry:
    mode: bytes
    path: bytes
    oid: bytes


@dataclass(frozen=True, slots=True)
class _PendingTerminal:
    path: bytes
    mode: bytes
    oid: bytes


def _git(
    repo: Path,
    *args: str,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", "-C", str(repo), *args],
            input=input_bytes,
            check=False,
            capture_output=True,
        )
    except OSError:
        return None


def _read_raw_object(repo: Path, oid: bytes) -> tuple[bytes, bytes] | None:
    result = _git(
        repo,
        "cat-file",
        "--batch",
        input_bytes=oid.hex().encode("ascii") + b"\n",
    )
    if result is None or result.returncode != 0:
        return None

    header_end = result.stdout.find(b"\n")
    if header_end < 0:
        return None
    header = result.stdout[:header_end].split()
    if len(header) != 3 or header[0] != oid.hex().encode("ascii"):
        return None
    try:
        size = int(header[2])
    except ValueError:
        return None
    if size < 0:
        return None

    content_start = header_end + 1
    content_end = content_start + size
    if result.stdout[content_end:] != b"\n":
        return None
    return header[1], result.stdout[content_start:content_end]


def _parse_tree(raw: bytes, oid_size: int) -> list[_RawTreeEntry] | None:
    entries: list[_RawTreeEntry] = []
    position = 0
    while position < len(raw):
        mode_end = raw.find(b" ", position)
        if mode_end < 0:
            return None
        mode = raw[position:mode_end]
        if not mode or not mode.isdigit():
            return None

        path_end = raw.find(b"\0", mode_end + 1)
        if path_end < 0:
            return None
        path = raw[mode_end + 1 : path_end]
        oid_start = path_end + 1
        oid_end = oid_start + oid_size
        if oid_end > len(raw):
            return None
        entries.append(_RawTreeEntry(mode, path, raw[oid_start:oid_end]))
        position = oid_end
    return entries


def _decoded_path(path: bytes) -> str | None:
    if not path:
        return None
    try:
        return path.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None


def _least_path(paths: Sequence[bytes]) -> bytes:
    return min(paths)


def _structural(path: bytes, detail: str) -> Refusal:
    return Refusal("structural", _decoded_path(path), detail)


def build_tree_manifest(repo: Path, tree_ish: str) -> Manifest | Refusal:
    """Build a terminal-entry manifest by parsing reachable raw Git trees."""
    resolved = _git(
        repo,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{tree_ish}^{{tree}}",
    )
    if resolved is None:
        return Refusal("structural", None, "git_unavailable")
    if resolved.returncode != 0:
        return Refusal("structural", None, "tree_unresolvable")
    try:
        root_oid = bytes.fromhex(resolved.stdout.strip().decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return Refusal("structural", None, "tree_unresolvable")
    if len(root_oid) not in {20, 32}:
        return Refusal("structural", None, "unsupported_object_format")

    oid_size = len(root_oid)
    tree_cache: dict[bytes, list[_RawTreeEntry] | None] = {}
    pending_trees: list[tuple[bytes, bytes]] = [(root_oid, b"")]
    terminals: list[_PendingTerminal] = []
    terminal_counts: dict[bytes, int] = {}
    gitlinks: list[bytes] = []
    non_utf8_paths: list[bytes] = []
    symlinks: list[bytes] = []
    bad_modes: list[tuple[bytes, bytes]] = []
    empty_subtrees: list[bytes] = []
    unreadable_trees: list[bytes] = []

    while pending_trees:
        tree_oid, prefix = pending_trees.pop()
        if tree_oid not in tree_cache:
            tree_object = _read_raw_object(repo, tree_oid)
            if tree_object is None or tree_object[0] != b"tree":
                tree_cache[tree_oid] = None
            else:
                tree_cache[tree_oid] = _parse_tree(tree_object[1], oid_size)
        entries = tree_cache[tree_oid]
        if entries is None:
            unreadable_trees.append(prefix)
            continue
        if not entries:
            empty_subtrees.append(prefix)

        for entry in entries:
            path = entry.path if not prefix else prefix + b"/" + entry.path
            try:
                path.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                non_utf8_paths.append(path)

            if entry.mode == _GITLINK_MODE:
                gitlinks.append(path)
                continue
            if entry.mode == _DIRECTORY_MODE:
                pending_trees.append((entry.oid, path))
                continue

            terminal_counts[path] = terminal_counts.get(path, 0) + 1
            terminals.append(_PendingTerminal(path, entry.mode, entry.oid))
            if entry.mode == _SYMLINK_MODE:
                symlinks.append(path)
            elif entry.mode not in _TERMINAL_MODES:
                bad_modes.append((path, entry.mode))

    duplicate_paths = [path for path, count in terminal_counts.items() if count > 1]
    if gitlinks:
        return _structural(_least_path(gitlinks), "gitlink")
    if non_utf8_paths:
        return _structural(_least_path(non_utf8_paths), "non_utf8_path")
    if symlinks:
        return _structural(_least_path(symlinks), "symlink")
    if bad_modes:
        path, mode = min(bad_modes, key=lambda item: item[0])
        return _structural(path, f"inadmissible_mode={mode.decode('ascii')}")
    if empty_subtrees:
        return _structural(_least_path(empty_subtrees), "empty_subtree")
    if duplicate_paths:
        return _structural(
            _least_path(duplicate_paths),
            "duplicate_flattened_terminal_path",
        )
    if unreadable_trees:
        return _structural(_least_path(unreadable_trees), "tree_object_unreadable")

    blob_digests: dict[bytes, str] = {}
    manifest: Manifest = []
    for terminal in terminals:
        if terminal.oid not in blob_digests:
            blob = _read_raw_object(repo, terminal.oid)
            if blob is None or blob[0] != b"blob":
                return _structural(terminal.path, "blob_object_unreadable")
            blob_digests[terminal.oid] = hashlib.sha256(blob[1]).hexdigest()
        manifest.append(
            (
                terminal.path.decode("utf-8"),
                terminal.mode.decode("ascii"),
                blob_digests[terminal.oid],
            )
        )
    manifest.sort(key=lambda entry: entry[0].encode("utf-8"))
    return manifest


def validate_manifest(manifest: Sequence[ManifestEntry]) -> None | Refusal:
    """Validate bytewise path order and uniqueness for a manifest."""
    previous: bytes | None = None
    for index, entry in enumerate(manifest):
        try:
            path = entry[0]
            encoded = path.encode("utf-8", errors="strict")
        except (IndexError, AttributeError, UnicodeEncodeError):
            return Refusal(
                "structural",
                f"manifest[{index}]",
                "invalid_manifest_path",
            )
        if previous is not None:
            if encoded == previous:
                return Refusal(
                    "structural",
                    path,
                    "duplicate_flattened_terminal_path",
                )
            if encoded < previous:
                return Refusal("structural", path, "manifest_out_of_order")
        previous = encoded
    return None


def manifest_sha256(manifest: Sequence[ManifestEntry]) -> str:
    """Return the JCS SHA-256 of the manifest's normatively sorted array."""
    entries = sorted(manifest, key=lambda entry: entry[0].encode("utf-8"))
    return body_digest([[path, mode, digest] for path, mode, digest in entries])


def manifest_diff(
    base: Sequence[ManifestEntry],
    subject: Sequence[ManifestEntry],
) -> list[ManifestDiffEntry]:
    """Return sorted per-path additions, deletions, and modifications."""
    base_entries = {path: (mode, digest) for path, mode, digest in base}
    subject_entries = {path: (mode, digest) for path, mode, digest in subject}
    paths = sorted(
        base_entries.keys() | subject_entries.keys(),
        key=lambda path: path.encode("utf-8"),
    )

    changes: list[ManifestDiffEntry] = []
    for path in paths:
        before = base_entries.get(path)
        after = subject_entries.get(path)
        if before == after:
            continue
        changes.append(
            ManifestDiffEntry(
                path=path,
                before_mode=None if before is None else before[0],
                before_entry_sha256=None if before is None else before[1],
                after_mode=None if after is None else after[0],
                after_entry_sha256=None if after is None else after[1],
            )
        )
    return changes


def fsck_clean(repo: Path) -> bool | Refusal:
    """Return whether ``git fsck --strict`` reports a clean object database."""
    result = _git(repo, "fsck", "--strict")
    if result is None:
        return Refusal("structural", None, "git_unavailable")
    return result.returncode == 0
