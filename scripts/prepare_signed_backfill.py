#!/usr/bin/env python3
"""Fail-closed helpers for publishing targeted signed RuleSpec backfills."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

COUNTRY_PATTERN = re.compile(r"[a-z]{2}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
QUEUE_TRACKING_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
MANIFEST_ROOT = PurePosixPath(".axiom/encoding-manifests")
LEGACY_REPLACEMENT_RECEIPT_ROOT = PurePosixPath(".axiom/legacy-replacements")
LEGACY_REPLACEMENT_TOOL = "axiom-encode encode --apply --replace-legacy-rulespec-path"
MODEL_APPLY_TOOL = "axiom-encode encode --apply"
MODEL_APPLY_BACKENDS = frozenset({"claude", "codex", "openai"})
LEGACY_REPLACEMENT_METADATA_PATHS = frozenset(
    {
        PurePosixPath(".axiom/index/provisions_to_rules.json"),
        PurePosixPath("oracle-coverage-pending.yaml"),
    }
)
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


def _read_bounded_regular(
    repo: Path,
    relative: PurePosixPath,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    """Read one canonical in-repo 0644 file without following symlinks."""

    from axiom_encode.corpus_resolver import (
        UnsafeCorpusPathError,
        read_bounded_regular_file,
    )

    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} path is not canonical: {relative}")
    root = repo.resolve(strict=True)
    try:
        return read_bounded_regular_file(
            root,
            root.joinpath(*relative.parts),
            label=label,
            max_bytes=max_bytes,
            required_mode=0o644,
        )
    except UnsafeCorpusPathError as exc:
        raise ValueError(str(exc)) from exc


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


def citation_rulespec_path(citation: str) -> PurePosixPath:
    jurisdiction, relative = _citation_rulespec_path(citation)
    return PurePosixPath(jurisdiction) / relative


def _is_regular_file_beneath(root: Path, relative: PurePosixPath) -> bool:
    """Reject files reached through any symlink beneath the checkout root."""

    if root.is_symlink() or not root.is_dir():
        return False
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            return False
    return cursor.is_file()


def validate_dependent_cascade(
    repo: Path,
    target_citation: str,
    *dependent_citations: str,
    target_rulespec_path: str | None = None,
) -> tuple[PurePosixPath, ...]:
    """Require the supplied modules to be all of the target's direct dependents."""

    import yaml

    target_jurisdiction, target_relative = _citation_rulespec_path(target_citation)
    if target_rulespec_path:
        replacement = PurePosixPath(target_rulespec_path)
        if (
            replacement.is_absolute()
            or replacement.as_posix() != target_rulespec_path
            or any(part in {"", ".", ".."} for part in replacement.parts)
            or len(replacement.parts) < 3
            or replacement.parts[0] != target_jurisdiction
            or replacement.parts[1] not in RULESPEC_ATOMIC_ROOTS
            or replacement.suffix != ".yaml"
            or replacement.name.endswith(".test.yaml")
        ):
            raise ValueError(
                "target RuleSpec path must be a canonical checkout-relative "
                "primary module in the citation jurisdiction"
            )
        target_relative = PurePosixPath(*replacement.parts[1:])
    if not dependent_citations:
        raise ValueError("at least one dependent citation is required")
    dependent_relatives: list[PurePosixPath] = []
    for dependent_citation in dependent_citations:
        dependent_jurisdiction, dependent_relative = _citation_rulespec_path(
            dependent_citation
        )
        if target_jurisdiction != dependent_jurisdiction:
            raise ValueError("target and dependents must use the same jurisdiction")
        if target_relative == dependent_relative:
            raise ValueError("dependent citation must differ from the target citation")
        dependent_relatives.append(dependent_relative)
    if len(set(dependent_relatives)) != len(dependent_relatives):
        raise ValueError("dependent citations must be unique")

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
    if not _is_regular_file_beneath(content_root, target_relative):
        raise ValueError("target citation has no regular baseline RuleSpec module")
    for dependent_relative in dependent_relatives:
        if not _is_regular_file_beneath(content_root, dependent_relative):
            raise ValueError(
                "dependent citation has no regular baseline RuleSpec module"
            )

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

    expected = set(dependent_relatives)
    if direct_dependents != expected:
        rendered = ", ".join(map(str, sorted(direct_dependents))) or "<none>"
        raise ValueError(
            "target direct-dependent set does not exactly match supplied dependents: "
            f"{rendered}"
        )
    return tuple(dependent_relatives)


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
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or not path.parts
    ):
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
    live_manifests = {path for path in manifests if (repo / path).is_file()}
    deleted_manifests = manifests - live_manifests
    if not live_manifests:
        raise ValueError(
            "no changed signed apply manifest is available to authorize publication"
        )

    manifest_payloads: dict[PurePosixPath, dict[str, object]] = {}
    for relative in live_manifests:
        payload = json.loads(
            _read_bounded_regular(
                repo,
                relative,
                label="changed manifest",
                max_bytes=1024 * 1024,
            ).decode("utf-8")
        )
        if not isinstance(payload, dict):
            raise ValueError(f"changed manifest is malformed: {relative}")
        if payload.get("schema_version") != "axiom-encode/applied-rulespec/v5":
            raise ValueError(f"changed manifest has an unsupported schema: {relative}")
        manifest_payloads[relative] = payload
    legacy_manifests = {
        relative
        for relative, payload in manifest_payloads.items()
        if payload.get("tool") == LEGACY_REPLACEMENT_TOOL
    }
    if len(legacy_manifests) > 1 or (deleted_manifests and len(legacy_manifests) != 1):
        raise ValueError(
            "deleted manifests require exactly one receipt-linked legacy replacement"
        )

    authorized = set(live_manifests)
    for relative, payload in manifest_payloads.items():
        replacement = payload.get("replacement")
        receipt_rewrites: set[PurePosixPath] = set()
        if payload.get("tool") == LEGACY_REPLACEMENT_TOOL:
            if not isinstance(replacement, dict):
                raise ValueError(f"legacy replacement binding is malformed: {relative}")
            receipt_relative = _safe_relative_path(
                replacement.get("receipt_path"),
                label=f"{relative} replacement.receipt_path",
            )
            if (
                receipt_relative.parent != LEGACY_REPLACEMENT_RECEIPT_ROOT
                or receipt_relative.suffix != ".json"
                or DIGEST_PATTERN.fullmatch(receipt_relative.stem) is None
            ):
                raise ValueError(
                    f"legacy replacement receipt path is invalid: {relative}"
                )
            receipt_raw = _read_bounded_regular(
                repo,
                receipt_relative,
                label="legacy replacement receipt",
                max_bytes=4 * 1024 * 1024,
            )
            if hashlib.sha256(receipt_raw).hexdigest() != replacement.get(
                "receipt_sha256"
            ):
                raise ValueError(
                    f"legacy replacement receipt digest differs: {relative}"
                )
            receipt = json.loads(receipt_raw.decode("utf-8"))
            if (
                receipt.get("schema_version")
                != "axiom-encode/legacy-fresh-reencode-receipt/v1"
                or receipt.get("tool") != LEGACY_REPLACEMENT_TOOL
            ):
                raise ValueError(
                    f"legacy replacement receipt schema differs: {relative}"
                )
            legacy = receipt.get("legacy")
            receipt_replacement = receipt.get("replacement")
            if not isinstance(legacy, dict) or not isinstance(
                receipt_replacement, dict
            ):
                raise ValueError(f"legacy replacement receipt is malformed: {relative}")
            old_manifest = legacy.get("manifest")
            if not isinstance(old_manifest, dict):
                raise ValueError(
                    f"legacy replacement old manifest evidence is malformed: {relative}"
                )
            old_manifest_path = _safe_relative_path(
                old_manifest.get("path"),
                label=f"{relative} legacy.manifest.path",
            )
            if (
                old_manifest_path not in deleted_manifests
                or replacement.get("legacy_manifest_path")
                != old_manifest_path.as_posix()
                or replacement.get("legacy_manifest_sha256")
                != old_manifest.get("sha256")
            ):
                raise ValueError(
                    f"legacy replacement old manifest deletion differs: {relative}"
                )
            raw_rewrites = receipt_replacement.get("rewrites")
            live_files = receipt_replacement.get("live_files")
            legacy_files = legacy.get("files")
            scheduled_dependents = receipt_replacement.get("scheduled_dependents")
            if not isinstance(raw_rewrites, list):
                raise ValueError(
                    f"legacy replacement receipt rewrites are malformed: {relative}"
                )
            if (
                not isinstance(live_files, list)
                or not isinstance(legacy_files, list)
                or not isinstance(scheduled_dependents, list)
            ):
                raise ValueError(
                    f"legacy replacement receipt file sets are malformed: {relative}"
                )
            nested_manifest = payload.get("replacement_manifest")
            if (
                not isinstance(nested_manifest, dict)
                or live_files != nested_manifest.get("applied_files")
                or receipt.get("replacement_manifest") != nested_manifest
            ):
                raise ValueError(
                    f"legacy replacement live files differ from fresh model "
                    f"manifest: {relative}"
                )
            repository = receipt.get("repository")
            base_commit = (
                repository.get("base_commit") if isinstance(repository, dict) else None
            )
            from axiom_encode.cli import (
                _legacy_replacement_authoritative_map,
                _legacy_replacement_reference_inventory_issues,
                _strict_legacy_replacement_map,
            )

            authoritative_replacements, authority_issues = (
                _legacy_replacement_authoritative_map(
                    repo,
                    base_commit=str(base_commit or ""),
                    manifest_label=relative.as_posix(),
                    legacy=legacy,
                    replacement=receipt_replacement,
                )
            )
            if authoritative_replacements is None or authority_issues:
                raise ValueError(
                    f"legacy replacement authority differs: {relative}: "
                    + "; ".join(authority_issues)
                )
            from axiom_encode.rulespec_path_migration import (
                PathMigrationPlanError,
                rewrite_exact_references,
            )

            for index, rewrite in enumerate(raw_rewrites):
                if not isinstance(rewrite, dict) or set(rewrite) != {
                    "path",
                    "before_sha256",
                    "after_sha256",
                    "replacements",
                }:
                    raise ValueError(
                        f"legacy replacement rewrite[{index}] is malformed"
                    )
                rewrite_path = _safe_relative_path(
                    rewrite.get("path"),
                    label=f"{relative} rewrite[{index}].path",
                )
                if (
                    rewrite_path not in LEGACY_REPLACEMENT_METADATA_PATHS
                    or rewrite_path in receipt_rewrites
                ):
                    raise ValueError(
                        f"legacy replacement rewrite[{index}] path is unauthorized"
                    )
                before = rewrite.get("before_sha256")
                after = rewrite.get("after_sha256")
                base_raw = _git(
                    repo,
                    "show",
                    f"HEAD:{rewrite_path.as_posix()}",
                )
                live_raw = _read_bounded_regular(
                    repo,
                    rewrite_path,
                    label="legacy replacement metadata rewrite",
                    max_bytes=16 * 1024 * 1024,
                )
                replacement_records = rewrite["replacements"]
                if _strict_legacy_replacement_map(replacement_records) is None:
                    raise ValueError(
                        f"legacy replacement rewrite[{index}] records are malformed"
                    )
                try:
                    expected_live, observed_counts = rewrite_exact_references(
                        base_raw,
                        authoritative_replacements,
                    )
                except PathMigrationPlanError as exc:
                    raise ValueError(
                        f"legacy replacement rewrite[{index}] is unreadable"
                    ) from exc
                if (
                    not isinstance(before, str)
                    or DIGEST_PATTERN.fullmatch(before) is None
                    or not isinstance(after, str)
                    or DIGEST_PATTERN.fullmatch(after) is None
                    or hashlib.sha256(base_raw).hexdigest() != before
                    or hashlib.sha256(live_raw).hexdigest() != after
                    or expected_live != live_raw
                    or list(observed_counts) != replacement_records
                ):
                    raise ValueError(
                        f"legacy replacement rewrite[{index}] state differs"
                    )
                receipt_rewrites.add(rewrite_path)
            expected_applied_files = [
                *live_files,
                *[
                    {
                        "path": rewrite.get("path"),
                        "sha256": rewrite.get("after_sha256"),
                    }
                    for rewrite in raw_rewrites
                    if isinstance(rewrite, dict)
                ],
                *[
                    {"path": item.get("path"), "deleted": True}
                    for item in legacy_files
                    if isinstance(item, dict)
                ],
            ]
            if payload.get("applied_files") != expected_applied_files:
                raise ValueError(
                    f"legacy replacement outer applied_files differ from receipt: "
                    f"{relative}"
                )
            for index, item in enumerate(live_files):
                if (
                    not isinstance(item, dict)
                    or set(item) != {"path", "sha256"}
                    or not isinstance(item.get("sha256"), str)
                    or DIGEST_PATTERN.fullmatch(item["sha256"]) is None
                ):
                    raise ValueError(
                        f"legacy replacement live_files[{index}] is malformed"
                    )
                live_path = _safe_relative_path(
                    item.get("path"),
                    label=f"{relative} live_files[{index}].path",
                )
                if (
                    hashlib.sha256(
                        _read_bounded_regular(
                            repo,
                            live_path,
                            label="legacy replacement live file",
                            max_bytes=16 * 1024 * 1024,
                        )
                    ).hexdigest()
                    != item["sha256"]
                ):
                    raise ValueError(
                        f"legacy replacement live file differs: {live_path}"
                    )
            for index, item in enumerate(legacy_files):
                if (
                    not isinstance(item, dict)
                    or set(item) != {"path", "sha256"}
                    or not isinstance(item.get("sha256"), str)
                    or DIGEST_PATTERN.fullmatch(item["sha256"]) is None
                ):
                    raise ValueError(
                        f"legacy replacement legacy.files[{index}] is malformed"
                    )
                deleted_path = _safe_relative_path(
                    item.get("path"),
                    label=f"{relative} legacy.files[{index}].path",
                )
                if (repo / deleted_path).exists() or (repo / deleted_path).is_symlink():
                    raise ValueError(
                        f"legacy replacement deleted file still exists: {deleted_path}"
                    )
                base_raw = _git(repo, "show", f"HEAD:{deleted_path.as_posix()}")
                if hashlib.sha256(base_raw).hexdigest() != item["sha256"]:
                    raise ValueError(
                        f"legacy replacement deleted file base differs: {deleted_path}"
                    )
            for index, dependent in enumerate(scheduled_dependents):
                if (
                    not isinstance(dependent, dict)
                    or set(dependent) != {"primary", "files"}
                    or not isinstance(dependent.get("files"), list)
                    or not dependent["files"]
                ):
                    raise ValueError(
                        f"legacy scheduled dependent[{index}] is malformed"
                    )
                primary = _safe_relative_path(
                    dependent.get("primary"),
                    label=f"{relative} scheduled[{index}].primary",
                )
                dependent_manifest = MANIFEST_ROOT / primary.with_suffix(".json")
                if dependent_manifest not in live_manifests:
                    raise ValueError(
                        f"legacy scheduled dependent lacks changed manifest: {primary}"
                    )
                dependent_payload = manifest_payloads[dependent_manifest]
                dependent_applied_files = dependent_payload.get("applied_files")
                dependent_hashes = (
                    {
                        str(item["path"]): str(item["sha256"])
                        for item in dependent_applied_files
                        if isinstance(item, dict)
                        and set(item) == {"path", "sha256"}
                        and isinstance(item.get("path"), str)
                        and isinstance(item.get("sha256"), str)
                        and DIGEST_PATTERN.fullmatch(item["sha256"]) is not None
                    }
                    if isinstance(dependent_applied_files, list)
                    else {}
                )
                if (
                    dependent_payload.get("tool") != MODEL_APPLY_TOOL
                    or dependent_payload.get("backend") not in MODEL_APPLY_BACKENDS
                    or primary.as_posix() not in dependent_hashes
                    or len(dependent_hashes) != len(dependent_applied_files or [])
                ):
                    raise ValueError(
                        f"legacy scheduled dependent lacks a fresh model manifest: "
                        f"{primary}"
                    )
                for file_index, evidence in enumerate(dependent["files"]):
                    if (
                        not isinstance(evidence, dict)
                        or set(evidence) != {"path", "before_sha256", "replacements"}
                        or not isinstance(evidence.get("replacements"), list)
                    ):
                        raise ValueError(
                            f"legacy scheduled dependent file is malformed: {primary}"
                        )
                    pending_path = _safe_relative_path(
                        evidence.get("path"),
                        label=(
                            f"{relative} scheduled[{index}].files[{file_index}].path"
                        ),
                    )
                    if pending_path not in changed:
                        raise ValueError(
                            f"legacy scheduled dependent was not freshly changed: "
                            f"{pending_path}"
                        )
                    base_raw = _git(repo, "show", f"HEAD:{pending_path.as_posix()}")
                    if hashlib.sha256(base_raw).hexdigest() != evidence.get(
                        "before_sha256"
                    ):
                        raise ValueError(
                            f"legacy scheduled dependent base differs: {pending_path}"
                        )
                    if _strict_legacy_replacement_map(evidence["replacements"]) is None:
                        raise ValueError(
                            "legacy scheduled dependent replacement records "
                            f"are malformed: {primary} file {file_index}"
                        )
                    try:
                        _base_rewritten, observed_counts = rewrite_exact_references(
                            base_raw,
                            authoritative_replacements,
                        )
                    except PathMigrationPlanError as exc:
                        raise ValueError(
                            f"legacy scheduled dependent is unreadable: {pending_path}"
                        ) from exc
                    live_raw = _read_bounded_regular(
                        repo,
                        pending_path,
                        label="legacy scheduled dependent",
                        max_bytes=16 * 1024 * 1024,
                    )
                    if (
                        dependent_hashes.get(pending_path.as_posix())
                        != hashlib.sha256(live_raw).hexdigest()
                    ):
                        raise ValueError(
                            "legacy scheduled dependent model manifest does not "
                            f"bind live file: {pending_path}"
                        )
                    try:
                        _live_rewritten, remaining = rewrite_exact_references(
                            live_raw,
                            authoritative_replacements,
                        )
                    except PathMigrationPlanError as exc:
                        raise ValueError(
                            f"legacy scheduled dependent is unreadable: {pending_path}"
                        ) from exc
                    if list(observed_counts) != evidence["replacements"]:
                        raise ValueError(
                            f"legacy scheduled dependent exact base proof differs: "
                            f"{pending_path}"
                        )
                    if remaining:
                        raise ValueError(
                            f"legacy scheduled dependent retains an old reference: "
                            f"{pending_path}"
                        )
            inventory_issues = _legacy_replacement_reference_inventory_issues(
                repo,
                base_commit=str(base_commit or ""),
                authoritative_replacements=authoritative_replacements,
                legacy=legacy,
                replacement=receipt_replacement,
                allow_pending_scheduled=False,
            )
            if inventory_issues:
                raise ValueError(
                    f"legacy replacement reference inventory differs: {relative}: "
                    + "; ".join(inventory_issues)
                )
            authorized.update({receipt_relative, old_manifest_path, *receipt_rewrites})

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
            if applied_path not in receipt_rewrites:
                _validate_rulespec_path(repo, applied_path, label=label)
            authorized.add(applied_path)

    if deleted_manifests - authorized:
        raise ValueError(
            "deleted manifests are not authenticated by a replacement receipt: "
            + ", ".join(map(str, sorted(deleted_manifests - authorized)))
        )

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
        for value in _git(
            repo,
            "diff",
            "--cached",
            "--name-only",
            "--no-renames",
            "-z",
        ).split(b"\0")
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
    cascade_parser.add_argument("--target-rulespec-path")
    cascade_parser.add_argument("dependent_citations", nargs="+")
    citation_path_parser = subparsers.add_parser("citation-rulespec-path")
    citation_path_parser.add_argument("citation")
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
                    *args.dependent_citations,
                    target_rulespec_path=args.target_rulespec_path,
                )
            )
        elif args.command == "citation-rulespec-path":
            print(citation_rulespec_path(args.citation))
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
