#!/usr/bin/env python3
"""Fail-closed helpers for publishing targeted signed RuleSpec backfills."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

from axiom_encode.constants import MAX_SIGNED_SOURCE_MODULES

COUNTRY_PATTERN = re.compile(r"[a-z]{2}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
QUEUE_TRACKING_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
MANIFEST_ROOT = PurePosixPath(".axiom/encoding-manifests")
LEGACY_REPLACEMENT_RECEIPT_ROOT = PurePosixPath(".axiom/legacy-replacements")
LEGACY_REPLACEMENT_TOOL = "axiom-encode encode --apply --replace-legacy-rulespec-path"
LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V1 = "axiom-encode/legacy-fresh-reencode-receipt/v1"
LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V2 = "axiom-encode/legacy-fresh-reencode-receipt/v2"
LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V3 = "axiom-encode/legacy-fresh-reencode-receipt/v3"
LEGACY_EXACT_DEPENDENT_TOOL = (
    "axiom-encode encode --apply --legacy-exact-dependent-rulespec-path"
)
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
MAX_SOURCE_BUNDLE_CITATIONS = MAX_SIGNED_SOURCE_MODULES
MAX_SOURCE_BUNDLE_JSON_BYTES = 512 * 1024
REVIEWED_RULESPEC_REFS = frozenset(
    {
        (
            "us",
            "b61918da93fe8a1a29b35b9330aef2085291a5d0",
        ),
        (
            "us",
            "251d8d66dabdebcb763d9e7c9b8322a281440c36",
        ),
        (
            "us",
            "68cca4a6fa806b63f95277c129575d88d2ac07f1",
        ),
        (
            "us",
            "b9b46dd845c61a49091146b3a3510fa3b8204ee7",
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
REVIEWED_RULESPEC_CONTINUATION_ROOTS = {
    (
        "us",
        "hard-cut/canonical-layout-us",
    ): "b9b46dd845c61a49091146b3a3510fa3b8204ee7",
}
MAX_REVIEWED_CONTINUATION_MODULES = 4096


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
    exact_reviewed_head = (country, requested_ref) in REVIEWED_RULESPEC_REFS
    if not exact_reviewed_head:
        if (
            not open_pr
            or (country, pr_base_branch) not in REVIEWED_RULESPEC_PR_BASE_BRANCHES
        ):
            raise ValueError(
                "rulespec ref is neither on main nor an approved reviewed head"
            )
        _require_remote_branch_tip(repo, pr_base_branch, requested_ref)
        reviewed_continuation_paths(
            repo,
            country,
            requested_ref,
            pr_base_branch=pr_base_branch,
        )
        return "reviewed-head-continuation-pr"
    if open_pr:
        if (country, pr_base_branch) not in REVIEWED_RULESPEC_PR_BASE_BRANCHES:
            raise ValueError(
                "reviewed-head runs are artifact-only unless the pull request "
                "targets an approved protected base branch"
            )
        _require_remote_branch_tip(repo, pr_base_branch, requested_ref)
        return "reviewed-head-pr"
    return "reviewed-head-artifact"


def _is_commit_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            check=False,
        ).returncode
        == 0
    )


def _reviewed_continuation_root(
    repo: Path,
    country: str,
    requested_ref: str,
    pr_base_branch: str,
) -> str:
    root = REVIEWED_RULESPEC_CONTINUATION_ROOTS.get((country, pr_base_branch))
    if (
        root is None
        or (country, root) not in REVIEWED_RULESPEC_REFS
        or not _is_commit_ancestor(repo, root, requested_ref)
    ):
        raise ValueError(
            "reviewed-head continuation does not descend from its configured "
            "approved reviewed root"
        )
    return root


def _require_commit_regular_file(
    repo: Path,
    commit: str,
    path: PurePosixPath,
    *,
    label: str,
) -> None:
    records = [
        record
        for record in _git(
            repo,
            "ls-tree",
            "-z",
            "--full-tree",
            commit,
            "--",
            path.as_posix(),
        ).split(b"\0")
        if record
    ]
    if len(records) != 1:
        raise ValueError(f"{label} is not exactly present at the continuation head")
    try:
        metadata, encoded_path = records[0].split(b"\t", 1)
        mode, object_type, _object_id = metadata.decode("ascii").split(" ")
        listed_path = encoded_path.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} tree entry is malformed") from exc
    if mode != "100644" or object_type != "blob" or listed_path != path.as_posix():
        raise ValueError(f"{label} is not a regular 0644 continuation blob")


def _reviewed_continuation_groups(
    repo: Path,
    country: str,
    requested_ref: str,
    *,
    pr_base_branch: str,
) -> dict[PurePosixPath, frozenset[PurePosixPath]]:
    """Return exact changed YAML groups in a reviewed-head continuation.

    The cumulative continuation may contain only in-place primary/test/manifest
    groups.  The workflow subsequently verifies every returned v5 manifest with
    the protected signing supervisor before model or signing credentials are used.
    """

    validate_country(country)
    if COMMIT_PATTERN.fullmatch(requested_ref) is None:
        raise ValueError("rulespec ref must be a full lowercase commit SHA")
    if (country, pr_base_branch) not in REVIEWED_RULESPEC_PR_BASE_BRANCHES:
        raise ValueError("reviewed-head continuation branch is not approved")
    if (country, requested_ref) in REVIEWED_RULESPEC_REFS:
        return {}
    if _git(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ):
        raise ValueError("reviewed-head continuation checkout must be exactly clean")
    root = _reviewed_continuation_root(
        repo,
        country,
        requested_ref,
        pr_base_branch,
    )
    raw = _git(
        repo,
        "diff",
        "--name-status",
        "-z",
        "--no-renames",
        root,
        requested_ref,
        "--",
    )
    fields = [field for field in raw.split(b"\0") if field]
    if len(fields) % 2:
        raise ValueError("reviewed-head continuation diff is malformed")

    changed_primary_files: set[PurePosixPath] = set()
    changed_groups: dict[PurePosixPath, set[PurePosixPath]] = {}
    changed_manifests: set[PurePosixPath] = set()
    for index in range(0, len(fields), 2):
        try:
            status = fields[index].decode("ascii")
            value = fields[index + 1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("reviewed-head continuation path is malformed") from exc
        if status not in {"A", "M"}:
            raise ValueError(
                "reviewed-head continuation may not delete, rename, or change "
                f"file types: {value}"
            )
        path = PurePosixPath(value)
        if path.is_relative_to(MANIFEST_ROOT):
            relative = path.relative_to(MANIFEST_ROOT)
            primary = relative.with_suffix(".yaml")
            if (
                len(primary.parts) < 3
                or primary.parts[1] not in RULESPEC_ATOMIC_ROOTS
                or path.suffix != ".json"
            ):
                raise ValueError(
                    f"reviewed-head continuation manifest path is invalid: {path}"
                )
            changed_manifests.add(primary)
            continue
        if (
            len(path.parts) < 3
            or path.parts[1] not in RULESPEC_ATOMIC_ROOTS
            or path.suffix != ".yaml"
        ):
            raise ValueError(
                f"reviewed-head continuation contains a non-generated path: {path}"
            )
        primary = (
            path.with_name(path.name.removesuffix(".test.yaml") + ".yaml")
            if path.name.endswith(".test.yaml")
            else path
        )
        changed_groups.setdefault(primary, set()).add(path)
        if not path.name.endswith(".test.yaml"):
            changed_primary_files.add(primary)

    if (
        not changed_primary_files
        or changed_primary_files != set(changed_groups)
        or changed_primary_files != changed_manifests
    ):
        raise ValueError(
            "reviewed-head continuation must contain exact primary/manifest groups"
        )
    if len(changed_primary_files) > MAX_REVIEWED_CONTINUATION_MODULES:
        raise ValueError("reviewed-head continuation contains too many modules")
    for primary in sorted(changed_primary_files):
        if primary.parts[0] != country and not primary.parts[0].startswith(
            f"{country}-"
        ):
            raise ValueError(
                "reviewed-head continuation module does not belong to the checkout"
            )
        manifest = _existing_import_manifest_path(primary)
        _require_commit_regular_file(
            repo,
            requested_ref,
            primary,
            label="reviewed-head continuation primary",
        )
        _require_commit_regular_file(
            repo,
            requested_ref,
            manifest,
            label="reviewed-head continuation manifest",
        )
        for changed_path in sorted(changed_groups[primary]):
            _require_commit_regular_file(
                repo,
                requested_ref,
                changed_path,
                label="reviewed-head continuation generated file",
            )
    return {
        primary: frozenset(changed_groups[primary])
        for primary in sorted(changed_primary_files)
    }


def reviewed_continuation_paths(
    repo: Path,
    country: str,
    requested_ref: str,
    *,
    pr_base_branch: str,
) -> tuple[PurePosixPath, ...]:
    """Return direct signed primaries in a protected reviewed-head continuation."""

    return tuple(
        _reviewed_continuation_groups(
            repo,
            country,
            requested_ref,
            pr_base_branch=pr_base_branch,
        )
    )


def validate_reviewed_continuation_inventories(
    repo: Path,
    country: str,
    requested_ref: str,
    inventories: object,
    *,
    pr_base_branch: str,
) -> None:
    """Require protected inventories to authenticate every changed YAML blob."""

    groups = _reviewed_continuation_groups(
        repo,
        country,
        requested_ref,
        pr_base_branch=pr_base_branch,
    )
    if not isinstance(inventories, list) or not all(
        isinstance(inventory, dict) for inventory in inventories
    ):
        raise ValueError("reviewed-head continuation inventories are malformed")
    items_by_primary: dict[PurePosixPath, dict[str, object]] = {}
    for inventory in inventories:
        if (
            inventory.get("schema") != "axiom-encode/signed-import-inventory/v1"
            or inventory.get("rulespec_base") != requested_ref
            or not isinstance(inventory.get("items"), list)
        ):
            raise ValueError("reviewed-head continuation inventory is malformed")
        for item in inventory["items"]:
            if not isinstance(item, dict) or not isinstance(
                item.get("rulespec_path"), str
            ):
                raise ValueError(
                    "reviewed-head continuation inventory item is malformed"
                )
            primary = PurePosixPath(item["rulespec_path"])
            if primary in items_by_primary:
                raise ValueError(
                    "reviewed-head continuation inventory contains duplicate primaries"
                )
            items_by_primary[primary] = item
    if set(items_by_primary) != set(groups):
        raise ValueError(
            "reviewed-head continuation inventory primary set differs from the diff"
        )
    for primary, changed_paths in groups.items():
        item = items_by_primary[primary]
        applied_files = item.get("applied_files")
        if not isinstance(applied_files, list):
            raise ValueError(
                f"reviewed-head continuation inventory lacks applied files: {primary}"
            )
        applied_paths: set[PurePosixPath] = set()
        for applied in applied_files:
            if (
                not isinstance(applied, dict)
                or set(applied) != {"path", "sha256"}
                or not isinstance(applied.get("path"), str)
                or not isinstance(applied.get("sha256"), str)
                or DIGEST_PATTERN.fullmatch(applied["sha256"]) is None
            ):
                raise ValueError(
                    "reviewed-head continuation applied-file inventory is malformed"
                )
            path = PurePosixPath(applied["path"])
            if path in applied_paths:
                raise ValueError(
                    "reviewed-head continuation applied-file inventory has duplicates"
                )
            applied_paths.add(path)
        unauthenticated = changed_paths - applied_paths
        if unauthenticated:
            raise ValueError(
                "reviewed-head continuation changed files are absent from the signed "
                f"manifest inventory: {', '.join(map(str, sorted(unauthenticated)))}"
            )


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


def parse_source_bundle(
    source_bundle_json: str,
    *,
    primary_citation: str,
    excluded_citations: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Validate a bounded same-jurisdiction corpus source citation bundle."""

    from axiom_encode.corpus_resolver import (
        require_canonical_corpus_citation_path,
    )

    if not isinstance(source_bundle_json, str):
        raise ValueError("source bundle JSON must be a string")
    if len(source_bundle_json.encode("utf-8")) > MAX_SOURCE_BUNDLE_JSON_BYTES:
        raise ValueError("source bundle JSON exceeds the maximum input size")
    payload = json.loads(source_bundle_json)
    if not isinstance(payload, list):
        raise ValueError("source bundle JSON must be an array")
    if len(payload) > MAX_SOURCE_BUNDLE_CITATIONS:
        raise ValueError(
            f"source bundle contains more than {MAX_SOURCE_BUNDLE_CITATIONS} citations"
        )

    def validate_citation(value: object, *, label: str) -> tuple[str, PurePosixPath]:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a nonempty citation string")
        try:
            citation = require_canonical_corpus_citation_path(value)
            path = citation_rulespec_path(citation)
        except ValueError as exc:
            raise ValueError(
                f"{label} must be an exact canonical corpus citation path"
            ) from exc
        return citation, path

    primary, primary_path = validate_citation(
        primary_citation,
        label="primary citation",
    )
    primary_jurisdiction = primary.partition("/")[0]
    validate_country(primary_jurisdiction.partition("-")[0])

    reserved_citations = {primary}
    reserved_paths = {primary_path}
    for index, value in enumerate(excluded_citations):
        citation, path = validate_citation(
            value,
            label=f"excluded citation #{index + 1}",
        )
        jurisdiction = citation.partition("/")[0]
        if jurisdiction != primary_jurisdiction:
            raise ValueError(
                "excluded citations must use the primary citation jurisdiction "
                "and country"
            )
        reserved_citations.add(citation)
        reserved_paths.add(path)

    citations: list[str] = []
    seen_citations: set[str] = set()
    seen_paths: set[PurePosixPath] = set()
    for index, value in enumerate(payload):
        citation, path = validate_citation(
            value,
            label=f"source bundle item #{index + 1}",
        )
        jurisdiction = citation.partition("/")[0]
        if jurisdiction != primary_jurisdiction:
            raise ValueError(
                "source bundle citations must use the primary citation "
                "jurisdiction and country"
            )
        if citation in reserved_citations:
            raise ValueError(
                "source bundle must exclude the primary and excluded citations"
            )
        if citation in seen_citations:
            raise ValueError("source bundle citations must be unique")
        if path in reserved_paths or path in seen_paths:
            raise ValueError(
                "source bundle citations must resolve to unique, unreserved "
                "canonical RuleSpec paths"
            )
        citations.append(citation)
        seen_citations.add(citation)
        seen_paths.add(path)
    return tuple(citations)


def _existing_import_manifest_path(path: PurePosixPath) -> PurePosixPath:
    return MANIFEST_ROOT / path.with_suffix(".json")


def parse_existing_signed_imports(
    repo: Path,
    existing_signed_imports_json: str,
    *,
    primary_citation: str,
    source_bundle_citations: tuple[str, ...] = (),
    excluded_citations: tuple[str, ...] = (),
    excluded_rulespec_paths: tuple[str, ...] = (),
    replacement_rulespec_path: str | None = None,
) -> tuple[PurePosixPath, ...]:
    """Validate bounded, tracked signed-v5 modules reused as direct imports."""

    if not isinstance(existing_signed_imports_json, str):
        raise ValueError("existing signed imports JSON must be a string")
    if len(existing_signed_imports_json.encode("utf-8")) > MAX_SOURCE_BUNDLE_JSON_BYTES:
        raise ValueError("existing signed imports JSON exceeds the maximum input size")
    payload = json.loads(existing_signed_imports_json)
    if not isinstance(payload, list):
        raise ValueError("existing signed imports JSON must be an array")
    if len(payload) + len(source_bundle_citations) > MAX_SOURCE_BUNDLE_CITATIONS:
        raise ValueError(
            "fresh source bundle and existing signed imports contain more than "
            f"{MAX_SOURCE_BUNDLE_CITATIONS} modules"
        )

    primary_jurisdiction, primary_relative = _citation_rulespec_path(primary_citation)
    primary_path = PurePosixPath(primary_jurisdiction) / primary_relative
    validate_country(primary_jurisdiction.partition("-")[0])
    reserved_paths = {primary_path}
    if replacement_rulespec_path:
        replacement_path = _safe_relative_path(
            replacement_rulespec_path,
            label="replacement RuleSpec path",
        )
        if (
            len(replacement_path.parts) < 3
            or replacement_path.parts[0] != primary_jurisdiction
            or replacement_path.parts[1] not in RULESPEC_ATOMIC_ROOTS
            or replacement_path.suffix != ".yaml"
            or replacement_path.name.endswith(".test.yaml")
        ):
            raise ValueError(
                "replacement RuleSpec path must be a canonical primary module "
                "in the primary citation jurisdiction"
            )
        reserved_paths.add(replacement_path)
        if replacement_path != primary_path:
            # A composition module may share its citation with a separately
            # signed source module.  The explicit, distinct replacement target
            # makes that source module an admissible direct import; it remains
            # subject to the full signed-v5 inventory verification below.
            reserved_paths.discard(primary_path)
    for citation in (*source_bundle_citations, *excluded_citations):
        jurisdiction, relative = _citation_rulespec_path(citation)
        if jurisdiction != primary_jurisdiction:
            raise ValueError(
                "fresh source and excluded citations must use the primary "
                "citation jurisdiction and country"
            )
        reserved_paths.add(PurePosixPath(jurisdiction) / relative)
    for index, value in enumerate(excluded_rulespec_paths):
        path = _safe_relative_path(
            value,
            label=f"excluded RuleSpec path #{index + 1}",
        )
        if (
            len(path.parts) < 3
            or path.parts[0] != primary_jurisdiction
            or path.parts[1] not in RULESPEC_ATOMIC_ROOTS
            or path.suffix != ".yaml"
            or path.name.endswith(".test.yaml")
        ):
            raise ValueError(
                "excluded RuleSpec paths must be canonical primary modules in "
                "the primary citation jurisdiction"
            )
        reserved_paths.add(path)

    if not payload:
        return ()

    repo = repo.resolve(strict=True)
    expected_repo_name = f"rulespec-{primary_jurisdiction.partition('-')[0]}"
    if repo.name != expected_repo_name:
        raise ValueError(
            "repository directory must match the primary citation country: "
            f"{expected_repo_name}"
        )

    paths: list[PurePosixPath] = []
    seen: set[PurePosixPath] = set()
    for index, value in enumerate(payload):
        label = f"existing signed import #{index + 1}"
        path = _safe_relative_path(value, label=label)
        if (
            any(
                ord(character) < 32 or ord(character) == 127 for character in str(value)
            )
            or len(path.parts) < 3
            or path.parts[0] != primary_jurisdiction
            or path.parts[1] not in RULESPEC_ATOMIC_ROOTS
            or path.suffix != ".yaml"
            or path.name.endswith(".test.yaml")
        ):
            raise ValueError(
                f"{label} must be a canonical primary RuleSpec path in the "
                "primary citation jurisdiction"
            )
        if path in reserved_paths:
            raise ValueError(
                "existing signed imports must exclude the primary, fresh source, "
                "and dependent paths"
            )
        if path in seen:
            raise ValueError("existing signed import paths must be unique")
        manifest_path = _existing_import_manifest_path(path)
        for tracked_path, tracked_label, max_bytes in (
            (path, label, 16 * 1024 * 1024),
            (manifest_path, f"{label} manifest", 1024 * 1024),
        ):
            try:
                tracked = (
                    _git(
                        repo,
                        "ls-files",
                        "--error-unmatch",
                        "--",
                        tracked_path.as_posix(),
                    )
                    .decode("utf-8")
                    .splitlines()
                )
            except (subprocess.CalledProcessError, UnicodeDecodeError) as exc:
                raise ValueError(f"{tracked_label} must be exactly tracked") from exc
            if tracked != [tracked_path.as_posix()]:
                raise ValueError(f"{tracked_label} must be exactly tracked")
            _read_bounded_regular(
                repo,
                tracked_path,
                label=tracked_label,
                max_bytes=max_bytes,
            )
        manifest = json.loads(
            _read_bounded_regular(
                repo,
                manifest_path,
                label=f"{label} manifest",
                max_bytes=1024 * 1024,
            ).decode("utf-8")
        )
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != "axiom-encode/applied-rulespec/v5"
        ):
            raise ValueError(f"{label} manifest must use signed-v5 schema")
        paths.append(path)
        seen.add(path)
    return tuple(paths)


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


def _rulespec_companion_path(primary: PurePosixPath) -> PurePosixPath:
    return primary.with_name(f"{primary.stem}.test.yaml")


def _base_regular_blob(
    repo: Path,
    base_commit: str,
    path: PurePosixPath,
    *,
    required: bool,
) -> bytes | None:
    listing = _git(
        repo,
        "ls-tree",
        "-z",
        "--full-tree",
        base_commit,
        "--",
        path.as_posix(),
    )
    records = [record for record in listing.split(b"\0") if record]
    if not records and not required:
        return None
    if len(records) != 1:
        raise ValueError(
            f"legacy exact dependent base does not contain exactly one {path}"
        )
    try:
        metadata, encoded_path = records[0].split(b"\t", 1)
        mode, object_type, _object_id = metadata.decode("ascii").split(" ")
        listed_path = encoded_path.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("legacy exact dependent base entry is malformed") from exc
    if mode != "100644" or object_type != "blob" or listed_path != path.as_posix():
        raise ValueError(
            f"legacy exact dependent base path is not regular 0644: {path}"
        )
    return _git(repo, "show", f"{base_commit}:{path.as_posix()}")


def _exact_file_entries(
    repo: Path,
    value: object,
    *,
    label: str,
) -> tuple[list[dict[str, object]], list[PurePosixPath]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} is malformed")
    entries: list[dict[str, object]] = []
    paths: list[PurePosixPath] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            or not isinstance(item.get("sha256"), str)
            or DIGEST_PATTERN.fullmatch(item["sha256"]) is None
        ):
            raise ValueError(f"{label}[{index}] is malformed")
        path = _safe_relative_path(
            item.get("path"),
            label=f"{label}[{index}].path",
        )
        _validate_rulespec_path(repo, path, label=f"{label}[{index}].path")
        if path in paths:
            raise ValueError(f"{label} contains duplicate paths")
        entries.append(item)
        paths.append(path)
    return entries, paths


def _validate_legacy_exact_dependents(
    repo: Path,
    *,
    root_manifest: PurePosixPath,
    receipt_relative: PurePosixPath,
    receipt_sha256: str,
    receipt: dict[str, object],
    receipt_replacement: dict[str, object],
    base_commit: str,
    authoritative_replacements: dict[str, str],
    live_manifests: set[PurePosixPath],
    manifest_payloads: dict[PurePosixPath, dict[str, object]],
    changed: set[PurePosixPath],
) -> set[PurePosixPath]:
    """Verify every v2 exact dependent and return unchanged authorized files."""

    from axiom_encode.cli import (
        _repair_proof_import_hashes,
        _strict_legacy_replacement_map,
    )
    from axiom_encode.rulespec_path_migration import (
        PathMigrationPlanError,
        rewrite_exact_references,
    )

    raw_dependents = receipt_replacement.get("exact_dependents")
    if not isinstance(raw_dependents, list):
        raise ValueError(
            f"legacy replacement exact_dependents are malformed: {root_manifest}"
        )
    unchanged_authorized: set[PurePosixPath] = set()
    seen_primaries: set[PurePosixPath] = set()
    seen_group_paths: set[PurePosixPath] = set()
    seen_manifests: set[PurePosixPath] = set()
    for index, raw_dependent in enumerate(raw_dependents):
        label = f"{root_manifest} exact_dependents[{index}]"
        if not isinstance(raw_dependent, dict) or set(raw_dependent) != {
            "primary",
            "legacy_manifest",
            "legacy_files",
            "live_files",
            "rewrites",
        }:
            raise ValueError(f"{label} is malformed")
        primary = _safe_relative_path(
            raw_dependent.get("primary"),
            label=f"{label}.primary",
        )
        _validate_rulespec_path(repo, primary, label=f"{label}.primary")
        if primary.name.endswith(".test.yaml") or primary in seen_primaries:
            raise ValueError(f"{label}.primary is invalid or duplicated")
        seen_primaries.add(primary)

        manifest_evidence = raw_dependent.get("legacy_manifest")
        expected_manifest = MANIFEST_ROOT / primary.with_suffix(".json")
        if (
            not isinstance(manifest_evidence, dict)
            or set(manifest_evidence) != {"path", "sha256"}
            or manifest_evidence.get("path") != expected_manifest.as_posix()
            or not isinstance(manifest_evidence.get("sha256"), str)
            or DIGEST_PATTERN.fullmatch(manifest_evidence["sha256"]) is None
            or expected_manifest in seen_manifests
        ):
            raise ValueError(f"{label}.legacy_manifest is malformed")
        seen_manifests.add(expected_manifest)
        base_manifest_raw = _base_regular_blob(
            repo,
            base_commit,
            expected_manifest,
            required=True,
        )
        assert base_manifest_raw is not None
        if hashlib.sha256(base_manifest_raw).hexdigest() != manifest_evidence["sha256"]:
            raise ValueError(f"{label} legacy manifest base hash differs")

        legacy_files, legacy_paths = _exact_file_entries(
            repo,
            raw_dependent.get("legacy_files"),
            label=f"{label}.legacy_files",
        )
        live_files, live_paths = _exact_file_entries(
            repo,
            raw_dependent.get("live_files"),
            label=f"{label}.live_files",
        )
        companion = _rulespec_companion_path(primary)
        companion_raw = _base_regular_blob(
            repo,
            base_commit,
            companion,
            required=False,
        )
        expected_paths = sorted(
            [primary, *([companion] if companion_raw is not None else [])],
            key=PurePosixPath.as_posix,
        )
        if legacy_paths != expected_paths or live_paths != expected_paths:
            raise ValueError(f"{label} does not bind the exact full file group")
        overlap = set(expected_paths) & seen_group_paths
        if overlap:
            raise ValueError(f"{label} overlaps another exact dependent group")
        seen_group_paths.update(expected_paths)

        base_by_path: dict[PurePosixPath, bytes] = {}
        live_by_path: dict[PurePosixPath, bytes] = {}
        for file_index, (path, legacy_item, live_item) in enumerate(
            zip(expected_paths, legacy_files, live_files, strict=True)
        ):
            base_raw = _base_regular_blob(
                repo,
                base_commit,
                path,
                required=True,
            )
            assert base_raw is not None
            live_raw = _read_bounded_regular(
                repo,
                path,
                label=f"{label}.live_files[{file_index}]",
                max_bytes=16 * 1024 * 1024,
            )
            if hashlib.sha256(base_raw).hexdigest() != legacy_item["sha256"]:
                raise ValueError(f"{label} legacy file base hash differs: {path}")
            if hashlib.sha256(live_raw).hexdigest() != live_item["sha256"]:
                raise ValueError(f"{label} live file hash differs: {path}")
            base_by_path[path] = base_raw
            live_by_path[path] = live_raw

        raw_rewrites = raw_dependent.get("rewrites")
        if not isinstance(raw_rewrites, list) or not raw_rewrites:
            raise ValueError(f"{label}.rewrites is malformed")
        rewrite_paths: set[PurePosixPath] = set()
        for rewrite_index, rewrite in enumerate(raw_rewrites):
            rewrite_label = f"{label}.rewrites[{rewrite_index}]"
            if not isinstance(rewrite, dict) or set(rewrite) != {
                "path",
                "before_sha256",
                "after_sha256",
                "replacements",
                "proof_import_repairs",
            }:
                raise ValueError(f"{rewrite_label} is malformed")
            rewrite_path = _safe_relative_path(
                rewrite.get("path"),
                label=f"{rewrite_label}.path",
            )
            if rewrite_path not in base_by_path or rewrite_path in rewrite_paths:
                raise ValueError(f"{rewrite_label}.path is invalid or duplicated")
            before = rewrite.get("before_sha256")
            after = rewrite.get("after_sha256")
            replacement_records = rewrite.get("replacements")
            if (
                not isinstance(before, str)
                or DIGEST_PATTERN.fullmatch(before) is None
                or not isinstance(after, str)
                or DIGEST_PATTERN.fullmatch(after) is None
                or before == after
                or _strict_legacy_replacement_map(replacement_records) is None
                or not isinstance(rewrite.get("proof_import_repairs"), int)
                or isinstance(rewrite.get("proof_import_repairs"), bool)
                or rewrite["proof_import_repairs"] < 0
            ):
                raise ValueError(f"{rewrite_label} proof is malformed")
            try:
                expected_live, observed_counts = rewrite_exact_references(
                    base_by_path[rewrite_path],
                    authoritative_replacements,
                )
                observed_proof_repairs = 0
                if rewrite_path == primary:
                    content_root = repo / primary.parts[0]
                    expected_text, observed_proof_repairs = _repair_proof_import_hashes(
                        expected_live.decode("utf-8"),
                        target_base=(
                            f"{content_root.name}:"
                            f"{PurePosixPath(*primary.parts[1:]).with_suffix('').as_posix()}"
                        ),
                        rules_file=repo.joinpath(*primary.parts),
                        repo_path=content_root,
                    )
                    expected_live = expected_text.encode("utf-8")
            except (PathMigrationPlanError, UnicodeError) as exc:
                raise ValueError(f"{rewrite_label} is unreadable") from exc
            if (
                hashlib.sha256(base_by_path[rewrite_path]).hexdigest() != before
                or hashlib.sha256(live_by_path[rewrite_path]).hexdigest() != after
                or expected_live != live_by_path[rewrite_path]
                or list(observed_counts) != replacement_records
                or rewrite["proof_import_repairs"] != observed_proof_repairs
                or rewrite_path not in changed
            ):
                raise ValueError(f"{rewrite_label} transformation differs")
            rewrite_paths.add(rewrite_path)

        for path in expected_paths:
            if path in rewrite_paths:
                continue
            if base_by_path[path] != live_by_path[path] or path in changed:
                raise ValueError(f"{label} unrewritten file differs: {path}")
            unchanged_authorized.add(path)

        if expected_manifest not in live_manifests or expected_manifest not in changed:
            raise ValueError(f"{label} lacks a changed exact dependent manifest")
        dependent_manifest = manifest_payloads[expected_manifest]
        exact_manifest_fields = {
            "schema_version",
            "generated_at",
            "tool",
            "axiom_encode_version",
            "axiom_encode_git",
            "validation_waiver_set_sha256",
            "applied_files",
            "legacy_migration",
            "signature",
        }
        signature = dependent_manifest.get("signature")
        if (
            set(dependent_manifest) != exact_manifest_fields
            or dependent_manifest.get("schema_version")
            != "axiom-encode/applied-rulespec/v5"
            or dependent_manifest.get("tool") != LEGACY_EXACT_DEPENDENT_TOOL
            or dependent_manifest.get("applied_files") != live_files
            or dependent_manifest.get("axiom_encode_version")
            != receipt.get("axiom_encode_version")
            or dependent_manifest.get("axiom_encode_git")
            != receipt.get("axiom_encode_git")
            or dependent_manifest.get("validation_waiver_set_sha256")
            != receipt.get("validation_waiver_set_sha256")
            or not isinstance(dependent_manifest.get("axiom_encode_version"), str)
            or not dependent_manifest["axiom_encode_version"]
            or not isinstance(dependent_manifest.get("axiom_encode_git"), dict)
            or not isinstance(
                dependent_manifest.get("validation_waiver_set_sha256"), str
            )
            or DIGEST_PATTERN.fullmatch(
                dependent_manifest["validation_waiver_set_sha256"]
            )
            is None
            or not isinstance(dependent_manifest.get("generated_at"), str)
            or not dependent_manifest["generated_at"]
            or not isinstance(signature, dict)
            or set(signature) != {"algorithm", "key_id", "value"}
            or not all(
                isinstance(signature.get(field), str) and signature[field]
                for field in ("algorithm", "key_id", "value")
            )
        ):
            raise ValueError(f"{label} exact dependent manifest is malformed")
        migration = dependent_manifest.get("legacy_migration")
        if not isinstance(migration, dict) or migration != {
            "receipt_path": receipt_relative.as_posix(),
            "receipt_sha256": receipt_sha256,
            "primary": primary.as_posix(),
            "legacy_manifest_path": expected_manifest.as_posix(),
            "legacy_manifest_sha256": manifest_evidence["sha256"],
        }:
            raise ValueError(f"{label} exact dependent manifest binding differs")
    return unchanged_authorized


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
    authorized_unchanged: set[PurePosixPath] = set()
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
            receipt_schema = receipt.get("schema_version")
            if (
                receipt_schema
                not in {
                    LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V1,
                    LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V2,
                    LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V3,
                }
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
            exact_dependents = receipt_replacement.get("exact_dependents")
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
            if receipt_schema == LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V1:
                if {
                    "exact_dependents",
                    "source_closure",
                } & set(receipt_replacement):
                    raise ValueError(
                        f"legacy replacement v1 receipt has newer fields: {relative}"
                    )
            elif not isinstance(exact_dependents, list):
                raise ValueError(
                    f"legacy replacement exact_dependents are malformed: {relative}"
                )
            if receipt_schema == LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V2:
                if "source_closure" in receipt_replacement:
                    raise ValueError(
                        f"legacy replacement v2 receipt has v3 fields: {relative}"
                    )
            elif receipt_schema == LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V3:
                source_closure = receipt_replacement.get("source_closure")
                if source_closure is not None and not isinstance(source_closure, dict):
                    raise ValueError(
                        f"legacy replacement source closure is malformed: {relative}"
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
            if receipt_schema in {
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V2,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V3,
            }:
                authorized_unchanged.update(
                    _validate_legacy_exact_dependents(
                        repo,
                        root_manifest=relative,
                        receipt_relative=receipt_relative,
                        receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
                        receipt=receipt,
                        receipt_replacement=receipt_replacement,
                        base_commit=str(base_commit or ""),
                        authoritative_replacements=authoritative_replacements,
                        live_manifests=live_manifests,
                        manifest_payloads=manifest_payloads,
                        changed=changed,
                    )
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

    authorized.difference_update(authorized_unchanged)
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
    continuation_parser = subparsers.add_parser("reviewed-continuation-paths")
    continuation_parser.add_argument("repo", type=Path)
    continuation_parser.add_argument("country")
    continuation_parser.add_argument("requested_ref")
    continuation_parser.add_argument("pr_base_branch")
    continuation_inventory_parser = subparsers.add_parser(
        "validate-reviewed-continuation-inventories"
    )
    continuation_inventory_parser.add_argument("repo", type=Path)
    continuation_inventory_parser.add_argument("country")
    continuation_inventory_parser.add_argument("requested_ref")
    continuation_inventory_parser.add_argument("pr_base_branch")
    continuation_inventory_parser.add_argument("inventories", type=Path)
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("repo", type=Path)
    cascade_parser = subparsers.add_parser("validate-dependent-cascade")
    cascade_parser.add_argument("repo", type=Path)
    cascade_parser.add_argument("target_citation")
    cascade_parser.add_argument("--target-rulespec-path")
    cascade_parser.add_argument("dependent_citations", nargs="+")
    citation_path_parser = subparsers.add_parser("citation-rulespec-path")
    citation_path_parser.add_argument("citation")
    source_bundle_parser = subparsers.add_parser(
        "parse-source-bundle",
        help="validate a bounded source bundle and emit one normalized JSON array",
    )
    source_bundle_parser.add_argument(
        "source_bundle_json",
        help=(
            "JSON array containing at most "
            f"{MAX_SOURCE_BUNDLE_CITATIONS} canonical corpus citation strings"
        ),
    )
    source_bundle_parser.add_argument(
        "--primary-citation",
        required=True,
        help="canonical primary citation, which is forbidden in the bundle",
    )
    source_bundle_parser.add_argument(
        "--exclude-citation",
        action="append",
        default=[],
        help="additional forbidden canonical citation; may be repeated",
    )
    existing_imports_parser = subparsers.add_parser(
        "parse-existing-signed-imports",
        help=(
            "validate tracked signed-v5 modules reused as direct imports and "
            "emit one normalized JSON array"
        ),
    )
    existing_imports_parser.add_argument("repo", type=Path)
    existing_imports_parser.add_argument(
        "existing_signed_imports_json",
        help="JSON array of canonical checkout-relative primary module paths",
    )
    existing_imports_parser.add_argument("--primary-citation", required=True)
    existing_imports_parser.add_argument(
        "--source-citation",
        action="append",
        default=[],
        help="fresh source citation already counted toward the 16-import limit",
    )
    existing_imports_parser.add_argument(
        "--exclude-citation",
        action="append",
        default=[],
        help="additional forbidden canonical citation; may be repeated",
    )
    existing_imports_parser.add_argument(
        "--exclude-rulespec-path",
        action="append",
        default=[],
        help="additional forbidden checkout-relative RuleSpec path; may be repeated",
    )
    existing_imports_parser.add_argument(
        "--replacement-rulespec-path",
        help=(
            "explicit replacement target; when distinct from the primary citation "
            "path, the latter may be reused as a signed direct import"
        ),
    )
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
        elif args.command == "reviewed-continuation-paths":
            print(
                json.dumps(
                    [
                        path.as_posix()
                        for path in reviewed_continuation_paths(
                            args.repo,
                            args.country,
                            args.requested_ref,
                            pr_base_branch=args.pr_base_branch,
                        )
                    ],
                    separators=(",", ":"),
                )
            )
        elif args.command == "validate-reviewed-continuation-inventories":
            inventory_raw = args.inventories.read_bytes()
            if len(inventory_raw) > 16 * 1024 * 1024:
                raise ValueError("reviewed-head continuation inventories are oversized")
            validate_reviewed_continuation_inventories(
                args.repo,
                args.country,
                args.requested_ref,
                [
                    json.loads(line)
                    for line in inventory_raw.decode("utf-8").splitlines()
                    if line.strip()
                ],
                pr_base_branch=args.pr_base_branch,
            )
            print("reviewed-head continuation inventories verified")
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
        elif args.command == "parse-source-bundle":
            print(
                json.dumps(
                    parse_source_bundle(
                        args.source_bundle_json,
                        primary_citation=args.primary_citation,
                        excluded_citations=tuple(args.exclude_citation),
                    ),
                    separators=(",", ":"),
                )
            )
        elif args.command == "parse-existing-signed-imports":
            print(
                json.dumps(
                    [
                        path.as_posix()
                        for path in parse_existing_signed_imports(
                            args.repo,
                            args.existing_signed_imports_json,
                            primary_citation=args.primary_citation,
                            source_bundle_citations=tuple(args.source_citation),
                            excluded_citations=tuple(args.exclude_citation),
                            excluded_rulespec_paths=tuple(args.exclude_rulespec_path),
                            replacement_rulespec_path=args.replacement_rulespec_path,
                        )
                    ],
                    separators=(",", ":"),
                )
            )
        else:
            stage_authorized_changes(args.repo)
    except (
        OSError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
