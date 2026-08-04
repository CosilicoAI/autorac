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


def _git(
    repository: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def verify_base_advance(
    repository: Path,
    *,
    country: str,
    source_ref: str,
    current_ref: str,
    candidate_path: str,
) -> None:
    repository = repository.resolve()
    path = PurePosixPath(candidate_path)
    if (
        re.fullmatch(r"[a-z][a-z0-9-]*", country) is None
        or path.is_absolute()
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
    ancestor = _git(
        repository,
        "merge-base",
        "--is-ancestor",
        source_ref,
        current_ref,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("repair source RuleSpec ref is not an ancestor of current")

    rulespec_path = PurePosixPath(country, path)
    test_path = path.with_suffix(".test.yaml")
    manifest_path = (
        PurePosixPath(".axiom", "encoding-manifests") / rulespec_path
    ).with_suffix(".json")
    tracked_paths = (
        rulespec_path.as_posix(),
        PurePosixPath(country, test_path).as_posix(),
        manifest_path.as_posix(),
    )
    unchanged = _git(
        repository,
        "diff",
        "--quiet",
        source_ref,
        current_ref,
        "--",
        *tracked_paths,
        check=False,
    )
    if unchanged.returncode == 1:
        raise ValueError(
            "repair replay target identity changed after its source RuleSpec base"
        )
    if unchanged.returncode != 0:
        raise RuntimeError("could not compare repair replay target identity")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("--country", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--current-ref", required=True)
    parser.add_argument("--candidate-path", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    verify_base_advance(
        args.repository,
        country=args.country,
        source_ref=args.source_ref,
        current_ref=args.current_ref,
        candidate_path=args.candidate_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
