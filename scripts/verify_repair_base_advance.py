#!/usr/bin/env python3
"""Verify that a repair candidate's RuleSpec target survived a base advance."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path, PurePosixPath

ATOMIC_ROOTS = frozenset(
    {"guidance", "manuals", "policies", "programs", "regulations", "statutes"}
)
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
JURISDICTION_PATTERN = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]+)*")


def _git(
    repository: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "--no-replace-objects", "-C", str(repository), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def _blob_identity(repository: Path, commit: str, path: PurePosixPath) -> str | None:
    """Return one tracked blob identity without reading checkout-controlled bytes."""

    result = _git(
        repository,
        "rev-parse",
        "--verify",
        f"{commit}:{path.as_posix()}",
        check=False,
    )
    if result.returncode != 0:
        return None
    identity = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40,64}", identity) is None:
        raise RuntimeError("repair replay target returned an invalid blob identity")
    object_type = _git(repository, "cat-file", "-t", identity).stdout.strip()
    if object_type != "blob":
        raise ValueError("repair replay target identity is not a tracked blob")
    return identity


def _unique_manifest_identity(
    repository: Path,
    commit: str,
    manifest_paths: tuple[PurePosixPath, ...],
    *,
    label: str,
) -> str:
    """Resolve one legacy or canonical manifest path, rejecting ambiguity."""

    identities = tuple(
        identity
        for path in manifest_paths
        if (identity := _blob_identity(repository, commit, path)) is not None
    )
    if not identities:
        raise ValueError(
            f"repair replay target identity is missing at its {label} RuleSpec base: "
            "ownership manifest"
        )
    if len(identities) != 1:
        raise ValueError(
            f"repair replay target identity is ambiguous at its {label} RuleSpec "
            "base: ownership manifest"
        )
    return identities[0]


def verify_base_advance(
    repository: Path,
    *,
    country: str,
    source_ref: str,
    current_ref: str,
    candidate_path: str,
    rulespec_path: str | None = None,
) -> None:
    repository = repository.resolve()
    path = PurePosixPath(candidate_path)
    if (
        re.fullmatch(r"[a-z][a-z0-9-]*", country) is None
        or path.is_absolute()
        or path.as_posix() != candidate_path
        or len(path.parts) < 2
        or path.parts[0] not in ATOMIC_ROOTS
        or path.suffix != ".yaml"
        or path.name.endswith(".test.yaml")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("repair candidate path is not canonical")
    for label, commit in (("source", source_ref), ("current", current_ref)):
        if COMMIT_PATTERN.fullmatch(commit) is None:
            raise ValueError(f"repair {label} RuleSpec ref is not a full commit SHA")

    head = _git(repository, "rev-parse", "HEAD").stdout.strip()
    if head != current_ref:
        raise ValueError("repair current RuleSpec ref is not the checkout HEAD")
    source_commit = _git(
        repository,
        "rev-parse",
        "--verify",
        f"{source_ref}^{{commit}}",
    ).stdout.strip()
    if source_commit != source_ref:
        raise ValueError("repair source RuleSpec ref does not identify its commit")
    repository_path = (
        PurePosixPath(rulespec_path)
        if rulespec_path is not None
        else PurePosixPath(country, path)
    )
    jurisdiction = repository_path.parts[0] if repository_path.parts else ""
    if (
        repository_path.is_absolute()
        or (rulespec_path is not None and repository_path.as_posix() != rulespec_path)
        or len(repository_path.parts) < 3
        or JURISDICTION_PATTERN.fullmatch(jurisdiction) is None
        or (jurisdiction != country and not jurisdiction.startswith(f"{country}-"))
        or repository_path.parts[1] not in ATOMIC_ROOTS
        or repository_path.suffix != ".yaml"
        or repository_path.name.endswith(".test.yaml")
        or any(part in {"", ".", ".."} for part in repository_path.parts)
        or PurePosixPath(*repository_path.parts[1:]) != path
    ):
        raise ValueError("repair repository RuleSpec path is not canonical")
    test_path = path.with_suffix(".test.yaml")
    legacy_manifest_path = (
        PurePosixPath(".axiom", "encoding-manifests") / repository_path
    ).with_suffix(".json")
    canonical_manifest_path = (
        PurePosixPath(".axiom", "encoding-manifests") / path
    ).with_suffix(".json")
    tracked_paths = (
        repository_path.as_posix(),
        PurePosixPath(repository_path.parts[0], test_path).as_posix(),
    )
    for tracked_path in tracked_paths:
        path_identity = PurePosixPath(tracked_path)
        source_identity = _blob_identity(repository, source_ref, path_identity)
        if source_identity is None:
            raise ValueError(
                "repair replay target identity is missing at its source RuleSpec "
                f"base: {tracked_path}"
            )
        current_identity = _blob_identity(repository, current_ref, path_identity)
        if current_identity is None:
            raise ValueError(
                "repair replay target identity is missing at its current RuleSpec "
                f"base: {tracked_path}"
            )
        if source_identity != current_identity:
            raise ValueError(
                "repair replay target identity changed after its source RuleSpec base"
            )
    manifest_paths = tuple(
        dict.fromkeys((legacy_manifest_path, canonical_manifest_path))
    )
    source_manifest_identity = _unique_manifest_identity(
        repository,
        source_ref,
        manifest_paths,
        label="source",
    )
    current_manifest_identity = _unique_manifest_identity(
        repository,
        current_ref,
        manifest_paths,
        label="current",
    )
    if source_manifest_identity != current_manifest_identity:
        raise ValueError(
            "repair replay target identity changed after its source RuleSpec base"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("--country", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--current-ref", required=True)
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--rulespec-path")
    return parser


def main() -> int:
    args = _parser().parse_args()
    verify_base_advance(
        args.repository,
        country=args.country,
        source_ref=args.source_ref,
        current_ref=args.current_ref,
        candidate_path=args.candidate_path,
        rulespec_path=args.rulespec_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
