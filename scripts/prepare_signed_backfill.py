#!/usr/bin/env python3
"""Fail-closed helpers for publishing targeted signed RuleSpec backfills."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

COUNTRY_PATTERN = re.compile(r"[a-z]{2}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
QUEUE_TRACKING_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
MANIFEST_ROOT = PurePosixPath(".axiom/encoding-manifests")
RULESPEC_ATOMIC_ROOTS = frozenset(
    {"legislation", "policies", "regulations", "statutes"}
)
REVIEWED_RULESPEC_REFS = frozenset(
    {
        (
            "us",
            "b61918da93fe8a1a29b35b9330aef2085291a5d0",
        ),
        (
            "ca",
            "f60f7a84c30e38c7d4961d70647eb0457e7d76c2",
        ),
    }
)
REVIEWED_RULESPEC_PR_BASE_BRANCHES = frozenset(
    {
        ("us", "hard-cut/canonical-layout-us"),
    }
)


def validate_country(value: str) -> str:
    if COUNTRY_PATTERN.fullmatch(value) is None:
        raise ValueError("country must be a two-letter lowercase country code")
    return value


def validate_queue_tracking(
    queue_id: str,
    queue_item_id: str,
    queue_manifest_sha256: str,
    queue_item_generation_sha256: str,
) -> str:
    """Require complete, shell-safe queue tracking metadata or no metadata."""

    values = (
        queue_id,
        queue_item_id,
        queue_manifest_sha256,
        queue_item_generation_sha256,
    )
    if not any(values):
        return "ad-hoc"
    if not all(values):
        raise ValueError("queue tracking fields must be supplied together")
    if QUEUE_TRACKING_PATTERN.fullmatch(queue_id) is None:
        raise ValueError("queue_id is malformed")
    if QUEUE_TRACKING_PATTERN.fullmatch(queue_item_id) is None:
        raise ValueError("queue_item_id is malformed")
    if DIGEST_PATTERN.fullmatch(queue_manifest_sha256) is None:
        raise ValueError("queue_manifest_sha256 is malformed")
    if DIGEST_PATTERN.fullmatch(queue_item_generation_sha256) is None:
        raise ValueError("queue_item_generation_sha256 is malformed")
    return "tracked"


def branch_name(country: str, run_id: str, run_attempt: str) -> str:
    validate_country(country)
    if not run_id.isdecimal() or not run_attempt.isdecimal():
        raise ValueError("run id and attempt must be decimal integers")
    return f"axiom/signed-backfill-{country}-{run_id}-{run_attempt}"


def validate_rulespec_base(
    repo: Path,
    country: str,
    requested_ref: str,
    *,
    open_pr: bool,
    pr_base_branch: str = "main",
) -> str:
    """Admit main ancestry or an exact independently reviewed protected head."""

    validate_country(country)
    if COMMIT_PATTERN.fullmatch(requested_ref) is None:
        raise ValueError("rulespec ref must be a full lowercase commit SHA")
    actual_ref = _git(repo, "rev-parse", "HEAD").decode().strip()
    if actual_ref != requested_ref:
        raise ValueError("rulespec checkout does not match the requested ref")
    main_ancestor = (
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                "HEAD",
                "refs/remotes/origin/main",
            ],
            check=False,
        ).returncode
        == 0
    )
    if main_ancestor:
        if open_pr:
            if pr_base_branch != "main":
                raise ValueError("main-ancestor pull requests must target main")
            _require_remote_branch_tip(repo, pr_base_branch, requested_ref)
        return "main"
    if (country, requested_ref) not in REVIEWED_RULESPEC_REFS:
        raise ValueError(
            "rulespec ref is neither on main nor an approved reviewed head"
        )
    if open_pr:
        if (country, pr_base_branch) not in REVIEWED_RULESPEC_PR_BASE_BRANCHES:
            raise ValueError(
                "reviewed-head runs are artifact-only unless the pull request "
                "targets an approved protected base branch"
            )
        _require_remote_branch_tip(repo, pr_base_branch, requested_ref)
        return "reviewed-head-pr"
    return "reviewed-head-artifact"


def _require_remote_branch_tip(
    repo: Path,
    branch: str,
    requested_ref: str,
) -> None:
    try:
        branch_ref = (
            _git(repo, "rev-parse", f"refs/remotes/origin/{branch}").decode().strip()
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"pull request base branch is unavailable: {branch}") from exc
    if branch_ref != requested_ref:
        raise ValueError("rulespec ref is not the exact pull request base branch tip")


def _citation_rulespec_path(citation: str) -> tuple[str, PurePosixPath]:
    from axiom_encode.harness.evals import _resolve_eval_output_path

    jurisdiction, separator, _remainder = citation.partition("/")
    if not separator:
        raise ValueError("citation must be a canonical corpus citation path")
    relative = PurePosixPath(_resolve_eval_output_path(citation).as_posix())
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) < 2
        or relative.parts[0] not in RULESPEC_ATOMIC_ROOTS
        or relative.suffix != ".yaml"
    ):
        raise ValueError("citation does not resolve to a canonical RuleSpec path")
    return jurisdiction, relative


def validate_dependent_cascade(
    repo: Path,
    target_citation: str,
    dependent_citation: str,
) -> PurePosixPath:
    """Require the supplied module to be the target's only direct dependent."""

    import yaml

    target_jurisdiction, target_relative = _citation_rulespec_path(target_citation)
    dependent_jurisdiction, dependent_relative = _citation_rulespec_path(
        dependent_citation
    )
    if target_jurisdiction != dependent_jurisdiction:
        raise ValueError("target and dependent must use the same jurisdiction")
    if target_relative == dependent_relative:
        raise ValueError("dependent citation must differ from the target citation")

    repo_prefix = "rulespec-"
    if not repo.name.startswith(repo_prefix):
        raise ValueError("repository directory must use the rulespec-<country> name")
    country = validate_country(repo.name.removeprefix(repo_prefix))
    if target_jurisdiction != country and not target_jurisdiction.startswith(
        f"{country}-"
    ):
        raise ValueError("citation jurisdiction does not belong to the RuleSpec repo")

    content_root = repo / target_jurisdiction
    target_path = content_root / target_relative
    dependent_path = content_root / dependent_relative
    if not target_path.is_file() or target_path.is_symlink():
        raise ValueError("target citation has no regular baseline RuleSpec module")
    if not dependent_path.is_file() or dependent_path.is_symlink():
        raise ValueError("dependent citation has no regular baseline RuleSpec module")

    target_import = target_relative.with_suffix("").as_posix()
    canonical_target_import = f"{target_jurisdiction}:{target_import}"
    direct_dependents: set[PurePosixPath] = set()
    for atomic_root in sorted(RULESPEC_ATOMIC_ROOTS):
        root = content_root / atomic_root
        if not root.exists():
            continue
        for candidate in sorted(root.rglob("*.yaml")):
            if candidate.name.endswith(".test.yaml") or candidate == target_path:
                continue
            if not candidate.is_file() or candidate.is_symlink():
                raise ValueError(
                    "baseline RuleSpec scan encountered a non-regular module"
                )
            try:
                payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, yaml.YAMLError) as exc:
                raise ValueError(
                    f"cannot inspect baseline RuleSpec module {candidate}"
                ) from exc
            if not isinstance(payload, dict):
                continue
            imports = payload.get("imports")
            if not isinstance(imports, list):
                continue
            if any(
                isinstance(raw_import, str)
                and raw_import.split("#", 1)[0].strip().strip("/")
                in {target_import, canonical_target_import}
                for raw_import in imports
            ):
                direct_dependents.add(
                    PurePosixPath(candidate.relative_to(content_root).as_posix())
                )

    expected = {dependent_relative}
    if direct_dependents != expected:
        rendered = ", ".join(map(str, sorted(direct_dependents))) or "<none>"
        raise ValueError(
            "target direct-dependent set does not exactly match supplied dependent: "
            f"{rendered}"
        )
    return dependent_relative


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def _changed_paths(repo: Path) -> set[PurePosixPath]:
    output = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    paths: set[PurePosixPath] = set()
    fields = output.split(b"\0")
    index = 0
    while index < len(fields) and fields[index]:
        entry = fields[index]
        status = entry[:2]
        if len(entry) < 4 or status[:1] in {b"R", b"C"} or status[1:] in {b"R", b"C"}:
            raise ValueError(
                "renamed/copied or malformed changed paths are not publishable"
            )
        paths.add(PurePosixPath(entry[3:].decode("utf-8")))
        index += 1
    return paths


def _safe_relative_path(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} is not a safe repository-relative path")
    return path


def _validate_rulespec_path(repo: Path, path: PurePosixPath, *, label: str) -> None:
    repo_prefix = "rulespec-"
    if not repo.name.startswith(repo_prefix):
        raise ValueError("repository directory must use the rulespec-<country> name")
    country = validate_country(repo.name.removeprefix(repo_prefix))
    if (
        len(path.parts) < 3
        or (path.parts[0] != country and not path.parts[0].startswith(f"{country}-"))
        or path.parts[1] not in RULESPEC_ATOMIC_ROOTS
        or path.suffix != ".yaml"
    ):
        raise ValueError(f"{label} is not a canonical RuleSpec YAML path")


def authorized_changed_paths(repo: Path) -> set[PurePosixPath]:
    changed = _changed_paths(repo)
    manifests = {
        path
        for path in changed
        if path.is_relative_to(MANIFEST_ROOT) and path.suffix == ".json"
    }
    if not manifests:
        raise ValueError(
            "no changed signed apply manifest is available to authorize publication"
        )

    authorized = set(manifests)
    for relative in manifests:
        manifest_path = repo / relative
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError(f"changed manifest is not a regular file: {relative}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "axiom-encode/applied-rulespec/v5":
            raise ValueError(f"changed manifest has an unsupported schema: {relative}")
        applied_files = payload.get("applied_files")
        if not isinstance(applied_files, list) or not applied_files:
            raise ValueError(
                f"changed manifest has no applied_files authorization: {relative}"
            )
        for index, entry in enumerate(applied_files):
            if not isinstance(entry, dict):
                raise ValueError(f"{relative} applied_files[{index}] is malformed")
            label = f"{relative} applied_files[{index}].path"
            applied_path = _safe_relative_path(entry.get("path"), label=label)
            _validate_rulespec_path(repo, applied_path, label=label)
            authorized.add(applied_path)

    unexpected = changed - authorized
    missing = authorized - changed
    if unexpected:
        raise ValueError(
            "publication found changed paths outside signed manifest authorization: "
            + ", ".join(map(str, sorted(unexpected)))
        )
    if missing:
        raise ValueError(
            "signed manifest authorizes paths that are not changed: "
            + ", ".join(map(str, sorted(missing)))
        )
    return authorized


def stage_authorized_changes(repo: Path) -> None:
    authorized = authorized_changed_paths(repo)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "core.hooksPath=/dev/null",
            "add",
            "--",
            *map(str, sorted(authorized)),
        ],
        check=True,
    )
    staged = {
        PurePosixPath(value.decode("utf-8"))
        for value in _git(repo, "diff", "--cached", "--name-only", "-z").split(b"\0")
        if value
    }
    if staged != authorized:
        raise ValueError("staged paths differ from signed manifest authorization")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    country_parser = subparsers.add_parser("validate-country")
    country_parser.add_argument("country")
    queue_parser = subparsers.add_parser("validate-queue-tracking")
    queue_parser.add_argument("queue_id")
    queue_parser.add_argument("queue_item_id")
    queue_parser.add_argument("queue_manifest_sha256")
    queue_parser.add_argument("queue_item_generation_sha256")
    branch_parser = subparsers.add_parser("branch-name")
    branch_parser.add_argument("country")
    branch_parser.add_argument("run_id")
    branch_parser.add_argument("run_attempt")
    base_parser = subparsers.add_parser("validate-rulespec-base")
    base_parser.add_argument("repo", type=Path)
    base_parser.add_argument("country")
    base_parser.add_argument("requested_ref")
    base_parser.add_argument("open_pr", choices=("true", "false"))
    base_parser.add_argument("pr_base_branch", nargs="?", default="main")
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("repo", type=Path)
    cascade_parser = subparsers.add_parser("validate-dependent-cascade")
    cascade_parser.add_argument("repo", type=Path)
    cascade_parser.add_argument("target_citation")
    cascade_parser.add_argument("dependent_citation")
    args = parser.parse_args()
    try:
        if args.command == "validate-country":
            print(validate_country(args.country))
        elif args.command == "validate-queue-tracking":
            print(
                validate_queue_tracking(
                    args.queue_id,
                    args.queue_item_id,
                    args.queue_manifest_sha256,
                    args.queue_item_generation_sha256,
                )
            )
        elif args.command == "branch-name":
            print(branch_name(args.country, args.run_id, args.run_attempt))
        elif args.command == "validate-rulespec-base":
            print(
                validate_rulespec_base(
                    args.repo,
                    args.country,
                    args.requested_ref,
                    open_pr=args.open_pr == "true",
                    pr_base_branch=args.pr_base_branch,
                )
            )
        elif args.command == "validate-dependent-cascade":
            print(
                validate_dependent_cascade(
                    args.repo,
                    args.target_citation,
                    args.dependent_citation,
                )
            )
        else:
            stage_authorized_changes(args.repo)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
