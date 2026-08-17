#!/usr/bin/env python3
"""Verify one exact signed-apply failed-candidate artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import stat
from pathlib import Path, PurePosixPath

SCHEMA = "axiom-encode/failed-encode-candidate/v1"
ATOMIC_ROOTS = frozenset({"legislation", "policies", "regulations", "statutes"})
PROTECTED_SEGMENTS = frozenset({".git", ".github", "_axiom", "scripts"})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
JURISDICTION_PATTERN = re.compile(r"[a-z]{2}(?:-[a-z0-9_]+)*")
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,127}")
MAX_ISSUES_BYTES = 512 * 1024
MAX_ISSUES = 4096
CONTRACT = runpy.run_path(
    Path(__file__).parents[1] / "src/axiom_encode/repair_candidate_contract.py"
)
MAX_CANDIDATE_BYTES = CONTRACT["VALIDATION_RETRY_CANDIDATE_MAX_FILE_BYTES"]
MAX_CANDIDATE_TOTAL_BYTES = CONTRACT["VALIDATION_RETRY_CANDIDATE_MAX_TOTAL_BYTES"]
METADATA_FIELDS = {
    "schema",
    "citation",
    "path",
    "issues",
    "rulespec_sha256",
    "tests_sha256",
    "encoder_version",
    "attempt_count",
}


def _canonical_candidate_path(raw_path: object) -> PurePosixPath:
    if not isinstance(raw_path, str):
        raise ValueError("failed candidate path must be text")
    path = PurePosixPath(raw_path)
    if (
        not raw_path
        or path.is_absolute()
        or path.as_posix() != raw_path
        or "\\" in raw_path
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(part in PROTECTED_SEGMENTS for part in path.parts)
        or any(
            any(ord(character) < 32 or ord(character) == 127 for character in part)
            for part in path.parts
        )
    ):
        raise ValueError("failed candidate path is not canonical")
    parts = path.parts
    if parts and parts[0] in ATOMIC_ROOTS:
        module_parts = parts
    elif (
        len(parts) >= 2
        and JURISDICTION_PATTERN.fullmatch(parts[0]) is not None
        and parts[1] in ATOMIC_ROOTS
    ):
        module_parts = parts[1:]
    else:
        module_parts = ()
    if (
        len(module_parts) < 2
        or path.suffix != ".yaml"
        or path.name.endswith(".test.yaml")
    ):
        raise ValueError("failed candidate path is not a canonical RuleSpec module")
    return path


def _read_bounded_regular_file(
    root: Path,
    relative: PurePosixPath,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise ValueError(f"{label} cannot be opened safely on this platform")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | directory
    descriptors: list[int] = []
    file_descriptor: int | None = None
    try:
        parent_descriptor = os.open(root, directory_flags)
        descriptors.append(parent_descriptor)
        for part in relative.parts[:-1]:
            parent_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            descriptors.append(parent_descriptor)
        file_descriptor = os.open(
            relative.parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | nofollow | os.O_NONBLOCK,
            dir_fd=parent_descriptor,
        )
        file_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"{label} is not a regular file")
        if file_stat.st_size > max_bytes:
            raise ValueError(f"{label} exceeds its size limit")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(file_descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_bytes:
            raise ValueError(f"{label} exceeds its size limit")
        return payload
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"could not safely open {label}") from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _directory_inventory(root: Path) -> tuple[set[str], set[str]]:
    try:
        root_stat = os.lstat(root)
    except OSError as exc:
        raise ValueError("failed candidate root is unavailable") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("failed candidate root must be a real directory")

    directories: set[str] = set()
    files: set[str] = set()
    for current_raw, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(current_raw)
        for name in dirnames:
            candidate = current / name
            try:
                candidate_stat = os.lstat(candidate)
            except OSError as exc:
                raise ValueError(
                    "failed candidate directory changed during scan"
                ) from exc
            if not stat.S_ISDIR(candidate_stat.st_mode):
                raise ValueError("failed candidate artifact contains a symlink")
            directories.add(candidate.relative_to(root).as_posix())
        for name in filenames:
            candidate = current / name
            try:
                candidate_stat = os.lstat(candidate)
            except OSError as exc:
                raise ValueError("failed candidate file changed during scan") from exc
            if not stat.S_ISREG(candidate_stat.st_mode):
                raise ValueError(
                    "failed candidate artifact contains a symlink or non-regular file"
                )
            files.add(candidate.relative_to(root).as_posix())
    return directories, files


def _expected_directories(*paths: PurePosixPath) -> set[str]:
    expected: set[str] = set()
    for path in paths:
        for parent in path.parents:
            if parent == PurePosixPath("."):
                break
            expected.add(parent.as_posix())
    return expected


def verify_candidate_directory(
    raw_root: Path,
    *,
    citation: str | None = None,
) -> dict[str, str]:
    """Verify exact content, metadata shape, paths, and content digests."""

    root = Path(os.path.abspath(raw_root))
    directories, files = _directory_inventory(root)
    issues_raw = _read_bounded_regular_file(
        root,
        PurePosixPath("issues.json"),
        label="failed candidate issues.json",
        max_bytes=MAX_ISSUES_BYTES,
    )
    try:
        metadata = json.loads(issues_raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(
            "failed candidate issues.json is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(metadata, dict) or set(metadata) != METADATA_FIELDS:
        raise ValueError("failed candidate issues.json has an invalid shape")
    if metadata.get("schema") != SCHEMA:
        raise ValueError("failed candidate issues.json schema is invalid")
    metadata_citation = metadata.get("citation")
    if not isinstance(metadata_citation, str) or not metadata_citation:
        raise ValueError("failed candidate citation is invalid")
    if citation is not None and metadata_citation != citation:
        raise ValueError("failed candidate citation mismatch")

    relative_rulespec = _canonical_candidate_path(metadata.get("path"))
    relative_tests = relative_rulespec.with_name(f"{relative_rulespec.stem}.test.yaml")
    expected_files = {
        "issues.json",
        relative_rulespec.as_posix(),
        relative_tests.as_posix(),
    }
    expected_directories = _expected_directories(relative_rulespec, relative_tests)
    if files != expected_files or directories != expected_directories:
        raise ValueError(
            "failed candidate artifact must contain exactly the rejected RuleSpec, "
            "companion tests, and issues.json"
        )

    issues = metadata.get("issues")
    if (
        not isinstance(issues, list)
        or not issues
        or len(issues) > MAX_ISSUES
        or any(not isinstance(issue, str) or not issue for issue in issues)
    ):
        raise ValueError("failed candidate issues must be a nonempty string array")
    attempt_count = metadata.get("attempt_count")
    if (
        not isinstance(attempt_count, int)
        or isinstance(attempt_count, bool)
        or attempt_count < 1
    ):
        raise ValueError("failed candidate attempt count must be positive")
    encoder_version = metadata.get("encoder_version")
    if (
        not isinstance(encoder_version, str)
        or VERSION_PATTERN.fullmatch(encoder_version) is None
    ):
        raise ValueError("failed candidate encoder version is invalid")
    for field in ("rulespec_sha256", "tests_sha256"):
        digest = metadata.get(field)
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"failed candidate {field} is invalid")

    rulespec = _read_bounded_regular_file(
        root,
        relative_rulespec,
        label="failed candidate RuleSpec",
        max_bytes=MAX_CANDIDATE_BYTES,
    )
    tests = _read_bounded_regular_file(
        root,
        relative_tests,
        label="failed candidate companion tests",
        max_bytes=MAX_CANDIDATE_BYTES,
    )
    if len(rulespec) + len(tests) > MAX_CANDIDATE_TOTAL_BYTES:
        raise ValueError("failed candidate pair exceeds its aggregate size limit")
    try:
        rulespec.decode("utf-8", errors="strict")
        tests.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError("failed candidate pair is not valid UTF-8") from exc
    rulespec_sha256 = hashlib.sha256(rulespec).hexdigest()
    tests_sha256 = hashlib.sha256(tests).hexdigest()
    if rulespec_sha256 != metadata["rulespec_sha256"]:
        raise ValueError("failed candidate RuleSpec SHA-256 mismatch")
    if tests_sha256 != metadata["tests_sha256"]:
        raise ValueError("failed candidate tests SHA-256 mismatch")

    return {
        "root": str(root),
        "path": relative_rulespec.as_posix(),
        "rulespec_sha256": rulespec_sha256,
        "tests_sha256": tests_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("--citation", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = verify_candidate_directory(
        args.candidate_root,
        citation=args.citation,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
