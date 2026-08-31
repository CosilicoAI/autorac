#!/usr/bin/env python3
"""Fail-closed helpers for publishing targeted signed RuleSpec backfills."""

from __future__ import annotations

import argparse
import ast
import contextlib
import errno
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import tokenize
from datetime import date
from pathlib import Path, PurePosixPath

COUNTRY_PATTERN = re.compile(r"[a-z]{2}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
QUEUE_TRACKING_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
MANIFEST_ROOT = PurePosixPath(".axiom/encoding-manifests")
LEGACY_REPLACEMENT_RECEIPT_ROOT = PurePosixPath(".axiom/legacy-replacements")
LEGACY_REPLACEMENT_TOOL = "axiom-encode encode --apply --replace-legacy-rulespec-path"
APPLIED_MANIFEST_SCHEMA_V5 = "axiom-encode/applied-rulespec/v5"
APPLIED_MANIFEST_SIGNATURE_ALGORITHM = "ed25519-domain-v1"
PROVISIONS_TO_RULES_INDEX = PurePosixPath(".axiom/index/provisions_to_rules.json")
RETIRED_MANIFEST_INVENTORY = PurePosixPath("tests/test_encoding_manifests.py")
LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V1 = "axiom-encode/legacy-fresh-reencode-receipt/v1"
LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V2 = "axiom-encode/legacy-fresh-reencode-receipt/v2"
LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V3 = "axiom-encode/legacy-fresh-reencode-receipt/v3"
LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V4 = "axiom-encode/legacy-fresh-reencode-receipt/v4"
LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5 = "axiom-encode/legacy-fresh-reencode-receipt/v5"
LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6 = "axiom-encode/legacy-fresh-reencode-receipt/v6"
LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7 = "axiom-encode/legacy-fresh-reencode-receipt/v7"
LEGACY_EXACT_DEPENDENT_TOOL = (
    "axiom-encode encode --apply --legacy-exact-dependent-rulespec-path"
)
LEGACY_RETAINED_SUCCESSOR_TOOL = (
    "axiom-encode encode --apply --legacy-retained-successor-rulespec-path"
)
MODEL_APPLY_TOOL = "axiom-encode encode --apply"
MODEL_APPLY_BACKENDS = frozenset({"claude", "codex", "openai"})
LEGACY_REPLACEMENT_METADATA_PATHS = frozenset(
    {
        PurePosixPath(".axiom/index/provisions_to_rules.json"),
        PurePosixPath(".axiom/pending-validation-fingerprints.json"),
        PurePosixPath(".axiom/retired-schema-freeze.json"),
        PurePosixPath(".axiom/toolchain.toml"),
        PurePosixPath("known-validation-gaps.yaml"),
        PurePosixPath("oracle-coverage-pending.yaml"),
        PurePosixPath("tests/test_encoding_manifests.py"),
        PurePosixPath("tests/test_legacy_rulespec_freeze.py"),
    }
)
RULESPEC_ATOMIC_ROOTS = frozenset(
    {"legislation", "policies", "regulations", "statutes"}
)
MAX_SOURCE_BUNDLE_CITATIONS = 16
MAX_SOURCE_BUNDLE_JSON_BYTES = 512 * 1024
MAX_CANONICAL_REFRESH_BUNDLE_CITATIONS = MAX_SOURCE_BUNDLE_CITATIONS - 1
MAX_DEFERRED_OUTPUT_CONTRACTS = 16
MAX_REQUIRED_TEST_CASES = 32
MAX_REQUIRED_TEST_CASE_FIELDS = 64
MAX_DEFERRED_OUTPUT_REVIEW_CONTRACT_JSON_BYTES = 64 * 1024
DEFERRED_OUTPUT_REVIEW_CONTRACT_SCHEMA = "axiom-encode/review-contract/v1"
STRUCTURED_REVIEW_CONTRACT_SCHEMA = "axiom-encode/review-contract/v2"
REVIEWED_RULESPEC_REFS = frozenset(
    {
        (
            "dk",
            "06489d04e7d4b8d424d1711d99df883c6411248a",
        ),
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
            "1e04e456ab404860050586c34eef51321eea95e9",
        ),
        (
            "us",
            "b1a6e07af093d62f613f83afe26fcb4dd87de491",
        ),
        (
            "us",
            "38ddc92d4160a0d39af13bfe232a446b554a15c5",
        ),
        (
            "us",
            "ef9dd5f72d529ebc70f539c42144361e536d7563",
        ),
        (
            "us",
            "f4fd3203db560c0d4661542388b6ae2f353e0bd3",
        ),
        (
            "us",
            "e942ce50546b1c3a1c0c8f3f0404a217eddbe071",
        ),
        (
            "us",
            "dc87ef6212accbc4ff67b81f97b6ddf0cf3b5a5c",
        ),
        (
            "us",
            "2a503a5c9a2227c363aceaece6c547429c3c0878",
        ),
        (
            "us",
            "6535019ce780d9e78f10509f2fe7a2607fb2bdc4",
        ),
        (
            "us",
            "c482ef6506c50b54236354926bbce1bcd6434132",
        ),
        (
            "ca",
            "f60f7a84c30e38c7d4961d70647eb0457e7d76c2",
        ),
    }
)
REVIEWED_RULESPEC_PR_BASE_BRANCHES = frozenset(
    {
        ("dk", "pin/dk-rulespec-2026-08-07"),
        ("us", "hard-cut/canonical-layout-us"),
    }
)

_DESCRIPTOR_OPEN = os.open
_DESCRIPTOR_STAT = os.stat
_DESCRIPTOR_OS_IS_POSIX = os.name == "posix"
_DESCRIPTOR_O_DIRECTORY = getattr(os, "O_DIRECTORY", None)
_DESCRIPTOR_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", None)
_DESCRIPTOR_O_CLOEXEC = getattr(os, "O_CLOEXEC", None)
_DESCRIPTOR_OPEN_SUPPORTS_DIR_FD = os.open in getattr(os, "supports_dir_fd", ())
_DESCRIPTOR_STAT_SUPPORTS_DIR_FD = os.stat in getattr(os, "supports_dir_fd", ())
_DESCRIPTOR_STAT_SUPPORTS_NOFOLLOW = os.stat in getattr(
    os,
    "supports_follow_symlinks",
    (),
)


def _load_unambiguous_json(raw: str, *, label: str) -> object:
    """Decode untrusted transaction JSON while rejecting duplicate object keys."""

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        decoded: dict[str, object] = {}
        for key, value in pairs:
            if key in decoded:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            decoded[key] = value
        return decoded

    return json.loads(raw, object_pairs_hook=reject_duplicates)


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


def split_atomic_source_input(atomic_source_json: str) -> dict[str, object]:
    """Split the bounded dispatch input into exactly one atomic source mode."""

    if not isinstance(atomic_source_json, str):
        raise ValueError("atomic source JSON must be a string")
    if len(atomic_source_json.encode("utf-8")) > MAX_SOURCE_BUNDLE_JSON_BYTES:
        raise ValueError("atomic source JSON exceeds the maximum input size")
    payload = _load_unambiguous_json(atomic_source_json, label="atomic source JSON")
    if isinstance(payload, list):
        return {
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
            "source_bundle": payload,
        }
    if isinstance(payload, dict) and set(payload) == {"canonical_refresh_bundle"}:
        refresh_bundle = payload["canonical_refresh_bundle"]
        if not isinstance(refresh_bundle, list):
            raise ValueError("canonical_refresh_bundle must be an array")
        return {
            "canonical_refresh_bundle": refresh_bundle,
            "primary_required_test_cases": [],
            "source_bundle": [],
        }
    v2_fields = {
        "schema",
        "source_bundle",
        "canonical_refresh_bundle",
        "primary_required_test_cases",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != v2_fields
        or payload.get("schema") != "axiom-encode/atomic-source-transaction/v2"
    ):
        raise ValueError(
            "atomic source JSON must be a source citation array or an exact "
            "canonical_refresh_bundle or atomic-source-transaction/v2 object"
        )
    refresh_bundle = payload["canonical_refresh_bundle"]
    source_bundle = payload["source_bundle"]
    primary_required_test_cases = payload["primary_required_test_cases"]
    if not all(
        isinstance(value, list)
        for value in (refresh_bundle, source_bundle, primary_required_test_cases)
    ):
        raise ValueError("atomic source transaction bundle fields must be arrays")
    if source_bundle and (refresh_bundle or primary_required_test_cases):
        raise ValueError(
            "atomic source transaction must select exactly one source mode"
        )
    return {
        "canonical_refresh_bundle": refresh_bundle,
        "primary_required_test_cases": primary_required_test_cases,
        "source_bundle": source_bundle,
    }


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
    payload = _load_unambiguous_json(source_bundle_json, label="source bundle JSON")
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


def _checkout_path_exists_without_indirection(
    repo: Path,
    relative: PurePosixPath,
    *,
    label: str,
) -> bool:
    """Probe one checkout path through pinned, no-follow directory handles."""

    if (
        relative.is_absolute()
        or not relative.parts
        or any(component in {"", ".", ".."} for component in relative.parts)
    ):
        raise ValueError(f"cannot inspect {label}: noncanonical path {relative}")
    required_flags = (
        _DESCRIPTOR_O_DIRECTORY,
        _DESCRIPTOR_O_NOFOLLOW,
        _DESCRIPTOR_O_CLOEXEC,
    )
    if (
        not _DESCRIPTOR_OS_IS_POSIX
        or not all(isinstance(flag, int) and flag for flag in required_flags)
        or not _DESCRIPTOR_OPEN_SUPPORTS_DIR_FD
        or not _DESCRIPTOR_STAT_SUPPORTS_DIR_FD
        or not _DESCRIPTOR_STAT_SUPPORTS_NOFOLLOW
    ):
        raise ValueError(
            f"cannot inspect {label} safely: descriptor traversal is unsupported"
        )
    directory_flags = (
        os.O_RDONLY
        | _DESCRIPTOR_O_DIRECTORY
        | _DESCRIPTOR_O_NOFOLLOW
        | _DESCRIPTOR_O_CLOEXEC
    )
    raw_repo = Path(os.path.abspath(repo))
    if not raw_repo.name:
        raise ValueError(f"cannot inspect {label}: repository root is invalid")

    descriptors: list[int] = []
    try:
        try:
            repo_fd = _DESCRIPTOR_OPEN(raw_repo, directory_flags)
            descriptors.append(repo_fd)
            opened_repo = os.fstat(repo_fd)
            if not stat.S_ISDIR(opened_repo.st_mode):
                raise ValueError(f"cannot inspect {label}: repository is not a directory")
        except (TypeError, NotImplementedError) as exc:
            raise ValueError(
                f"cannot inspect {label} safely: descriptor traversal is unsupported"
            ) from exc
        except OSError as exc:
            raise ValueError(f"cannot inspect {label}: {relative}") from exc

        current_fd = repo_fd
        for component in relative.parts[:-1]:
            try:
                current_fd = _DESCRIPTOR_OPEN(
                    component,
                    directory_flags,
                    dir_fd=current_fd,
                )
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    return False
                raise ValueError(f"cannot inspect {label}: {relative}") from exc
            except (TypeError, NotImplementedError) as exc:
                raise ValueError(
                    f"cannot inspect {label} safely: descriptor traversal is unsupported"
                ) from exc
            descriptors.append(current_fd)

        try:
            _DESCRIPTOR_STAT(
                relative.name,
                dir_fd=current_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                return False
            raise ValueError(f"cannot inspect {label}: {relative}") from exc
        except (TypeError, NotImplementedError) as exc:
            raise ValueError(
                f"cannot inspect {label} safely: descriptor traversal is unsupported"
            ) from exc
        return True
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def validate_source_add_targets(
    repo: Path,
    source_bundle_json: str,
    *,
    primary_citation: str,
    primary_rulespec_path: str = "",
) -> tuple[str, ...]:
    """Require source-add destinations to be absent in the pinned checkout.

    Existing canonical modules belong to the canonical-refresh protocol.  A
    primary replacement may still compose genuinely new source modules, but
    those additions must not collide with a primary, companion, or ownership
    manifest already present in the immutable RuleSpec base.
    """

    sources = parse_source_bundle(
        source_bundle_json,
        primary_citation=primary_citation,
    )
    primary_jurisdiction = primary_citation.partition("/")[0]
    expected_repo_name = f"rulespec-{primary_jurisdiction.partition('-')[0]}"
    repo = Path(repo).resolve(strict=True)
    if not repo.is_dir() or repo.name != expected_repo_name:
        raise ValueError(
            "repository directory must match the primary citation country: "
            f"{expected_repo_name}"
        )

    if primary_rulespec_path:
        _safe_relative_path(
            primary_rulespec_path,
            label="source-add primary replacement",
        )
        additions = sources
    else:
        additions = (primary_citation, *sources)

    conflicts: list[str] = []
    for citation in additions:
        rulespec_path = citation_rulespec_path(citation)
        companion_path = rulespec_path.with_name(f"{rulespec_path.stem}.test.yaml")
        manifest_path = MANIFEST_ROOT / rulespec_path.with_suffix(".json")
        for label, path in (
            ("RuleSpec primary", rulespec_path),
            ("RuleSpec companion", companion_path),
            ("RuleSpec ownership manifest", manifest_path),
        ):
            if _checkout_path_exists_without_indirection(
                repo,
                path,
                label=label,
            ):
                conflicts.append(path.as_posix())

    if conflicts:
        raise ValueError(
            "source-add destination already exists in the pinned RuleSpec "
            "checkout; existing modules must use canonical_refresh_bundle: "
            + ", ".join(sorted(conflicts))
        )
    return sources


def _normalize_required_test_cases(
    value: object,
    *,
    label: str,
) -> tuple[dict[str, object], ...]:
    """Validate bounded exact companion-case admission requirements."""

    if not isinstance(value, list) or len(value) > MAX_REQUIRED_TEST_CASES:
        raise ValueError(
            f"{label} must be an array with at most {MAX_REQUIRED_TEST_CASES} entries"
        )
    normalized_cases: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for index, item in enumerate(value):
        case_label = f"{label} case #{index + 1}"
        if not isinstance(item, dict) or set(item) != {
            "name",
            "period",
            "input",
            "required_output",
        }:
            raise ValueError(
                f"{case_label} must contain exactly name, period, input, and "
                "required_output"
            )
        name = item["name"]
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or "\r" in name
            or any(ord(character) < 32 or ord(character) == 127 for character in name)
        ):
            raise ValueError(f"{case_label} name must be a normalized string")
        if name in seen_names:
            raise ValueError(f"{label} case names must be unique")
        seen_names.add(name)
        period = item["period"]
        period_fields = {"period_kind", "start", "end"}
        if isinstance(period, dict) and period.get("period_kind") == "custom":
            period_fields.add("name")
        if (
            not isinstance(period, dict)
            or set(period) != period_fields
            or any(
                not isinstance(field, str)
                or not field
                or field != field.strip()
                or any(
                    ord(character) < 32 or ord(character) == 127 for character in field
                )
                for field in period.values()
            )
        ):
            raise ValueError(
                f"{case_label} period must be an exact normalized RuleSpec period "
                "mapping"
            )
        if period["period_kind"] not in {
            "month",
            "benefit_week",
            "tax_year",
            "custom",
        }:
            raise ValueError(
                f"{case_label} period_kind is not supported by the RuleSpec engine"
            )
        try:
            period_start = date.fromisoformat(period["start"])
            period_end = date.fromisoformat(period["end"])
        except ValueError as exc:
            raise ValueError(f"{case_label} period dates must be ISO dates") from exc
        if period_start > period_end:
            raise ValueError(f"{case_label} period start must not follow end")
        normalized_fields: dict[str, dict[str, object]] = {}
        for field_name in ("input", "required_output"):
            mapping = item[field_name]
            if (
                not isinstance(mapping, dict)
                or (field_name == "required_output" and not mapping)
                or len(mapping) > MAX_REQUIRED_TEST_CASE_FIELDS
            ):
                raise ValueError(
                    f"{case_label} {field_name} must be an object with at "
                    f"most {MAX_REQUIRED_TEST_CASE_FIELDS} fields"
                )
            normalized_mapping: dict[str, object] = {}
            for key, field_value in mapping.items():
                if (
                    not isinstance(key, str)
                    or not key
                    or key != key.strip()
                    or any(
                        ord(character) < 32 or ord(character) == 127
                        for character in key
                    )
                ):
                    raise ValueError(
                        f"{case_label} {field_name} keys must be normalized strings"
                    )
                if not isinstance(field_value, (str, int, float, bool)) or (
                    isinstance(field_value, float) and not math.isfinite(field_value)
                ):
                    raise ValueError(
                        f"{case_label} {field_name} values must be finite JSON scalars"
                    )
                if isinstance(field_value, str) and (
                    field_value != field_value.strip()
                    or "\r" in field_value
                    or any(
                        (ord(character) < 32 and character not in {"\n", "\t"})
                        or ord(character) == 127
                        for character in field_value
                    )
                ):
                    raise ValueError(
                        f"{case_label} {field_name} string values must be normalized"
                    )
                normalized_mapping[key] = field_value
            normalized_fields[field_name] = normalized_mapping
        normalized_period = {
            "period_kind": period["period_kind"],
            **({"name": period["name"]} if period["period_kind"] == "custom" else {}),
            "start": period["start"],
            "end": period["end"],
        }
        normalized_cases.append(
            {
                "name": name,
                "period": normalized_period,
                "input": dict(sorted(normalized_fields["input"].items())),
                "required_output": dict(
                    sorted(normalized_fields["required_output"].items())
                ),
            }
        )
    return tuple(normalized_cases)


def _validate_wrapped_review_contract_size(
    *,
    citation: str,
    path: PurePosixPath,
    deferred_output_contracts: tuple[dict[str, str], ...],
    required_test_cases: tuple[dict[str, object], ...],
    label: str,
) -> None:
    """Keep helper normalization within the installed CLI's exact size bound."""

    if not deferred_output_contracts and not required_test_cases:
        return
    payload: dict[str, object] = {
        "schema": (
            STRUCTURED_REVIEW_CONTRACT_SCHEMA
            if required_test_cases
            else DEFERRED_OUTPUT_REVIEW_CONTRACT_SCHEMA
        ),
        "citation": citation,
        "rulespec_path": path.as_posix(),
        "required_deferred_outputs": list(deferred_output_contracts),
    }
    if required_test_cases:
        payload["required_test_cases"] = list(required_test_cases)
    wrapped_contract = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(wrapped_contract) > MAX_DEFERRED_OUTPUT_REVIEW_CONTRACT_JSON_BYTES:
        raise ValueError(
            f"{label} wrapped review contract exceeds the maximum input size"
        )


def parse_canonical_refresh_bundle(
    repo: Path,
    refresh_bundle_json: str,
    *,
    primary_citation: str,
    primary_rulespec_path: str,
    primary_required_test_cases_json: str = "[]",
) -> tuple[dict[str, str | None], ...]:
    """Validate independent existing canonical modules for atomic fresh encoding."""

    from axiom_encode.corpus_resolver import (
        require_canonical_corpus_citation_path,
    )

    if not isinstance(refresh_bundle_json, str):
        raise ValueError("canonical refresh bundle JSON must be a string")
    if len(refresh_bundle_json.encode("utf-8")) > MAX_SOURCE_BUNDLE_JSON_BYTES:
        raise ValueError("canonical refresh bundle JSON exceeds the maximum input size")
    payload = _load_unambiguous_json(
        refresh_bundle_json,
        label="canonical refresh bundle JSON",
    )
    if not isinstance(payload, list):
        raise ValueError("canonical refresh bundle JSON must be an array")
    if len(payload) > MAX_CANONICAL_REFRESH_BUNDLE_CITATIONS:
        raise ValueError(
            "canonical refresh bundle and its primary contain more than "
            f"{MAX_SOURCE_BUNDLE_CITATIONS} modules"
        )
    if not isinstance(primary_required_test_cases_json, str):
        raise ValueError("primary required test cases JSON must be a string")
    if (
        len(primary_required_test_cases_json.encode("utf-8"))
        > MAX_DEFERRED_OUTPUT_REVIEW_CONTRACT_JSON_BYTES
    ):
        raise ValueError(
            "primary required test cases JSON exceeds the maximum input size"
        )
    primary_required_test_cases = _normalize_required_test_cases(
        _load_unambiguous_json(
            primary_required_test_cases_json,
            label="primary required test cases JSON",
        ),
        label="primary required test cases",
    )
    if not payload and not primary_required_test_cases:
        return ()

    repo = repo.resolve(strict=True)
    try:
        primary_citation = require_canonical_corpus_citation_path(primary_citation)
    except ValueError as exc:
        raise ValueError(
            "canonical refresh primary citation must be an exact canonical "
            "corpus citation path"
        ) from exc
    primary_jurisdiction, _primary_relative = _citation_rulespec_path(primary_citation)
    expected_repo_name = f"rulespec-{primary_jurisdiction.partition('-')[0]}"
    if repo.name != expected_repo_name:
        raise ValueError(
            "repository directory must match the primary citation country: "
            f"{expected_repo_name}"
        )

    primary_path = _safe_relative_path(
        primary_rulespec_path,
        label="canonical refresh primary RuleSpec",
    )
    expected_primary_path = citation_rulespec_path(primary_citation)
    if primary_path != expected_primary_path:
        raise ValueError(
            "canonical refresh primary path must equal the citation's canonical "
            "RuleSpec path"
        )

    _validate_wrapped_review_contract_size(
        citation=primary_citation,
        path=primary_path,
        deferred_output_contracts=(),
        required_test_cases=primary_required_test_cases,
        label="canonical refresh primary",
    )
    requested: list[
        tuple[
            str,
            PurePosixPath,
            str | None,
            tuple[dict[str, str], ...],
            tuple[dict[str, object], ...],
        ]
    ] = [(primary_citation, primary_path, None, (), primary_required_test_cases)]
    seen_citations = {primary_citation}
    seen_paths = {primary_path}
    for index, value in enumerate(payload):
        label = f"canonical refresh addition #{index + 1}"
        required_fields = {"citation", "replace_rulespec_path"}
        allowed_fields = {
            *required_fields,
            "review_finding",
            "deferred_output_contracts",
            "required_test_cases",
        }
        if (
            not isinstance(value, dict)
            or not required_fields.issubset(value)
            or not set(value).issubset(allowed_fields)
        ):
            raise ValueError(
                f"{label} must contain citation and replace_rulespec_path, with "
                "only optional review_finding, deferred_output_contracts, and "
                "required_test_cases fields"
            )
        citation = value["citation"]
        raw_path = value["replace_rulespec_path"]
        review_finding = value.get("review_finding")
        deferred_output_contracts = value.get("deferred_output_contracts", [])
        required_test_cases = _normalize_required_test_cases(
            value.get("required_test_cases", []),
            label=f"{label} required test cases",
        )
        if (
            not isinstance(citation, str)
            or not citation
            or citation != citation.strip()
            or any(
                ord(character) < 32 or ord(character) == 127 for character in citation
            )
            or not isinstance(raw_path, str)
            or not raw_path
            or any(
                ord(character) < 32 or ord(character) == 127 for character in raw_path
            )
        ):
            raise ValueError(f"{label} fields must be nonempty canonical strings")
        if review_finding is not None and (
            not isinstance(review_finding, str)
            or not review_finding
            or review_finding != review_finding.strip()
            or "\r" in review_finding
            or any(
                (ord(character) < 32 and character not in {"\n", "\t"})
                or ord(character) == 127
                for character in review_finding
            )
        ):
            raise ValueError(
                f"{label} review_finding must be a nonempty normalized string"
            )
        if (
            not isinstance(deferred_output_contracts, list)
            or len(deferred_output_contracts) > MAX_DEFERRED_OUTPUT_CONTRACTS
        ):
            raise ValueError(
                f"{label} deferred_output_contracts must be an array with at most "
                f"{MAX_DEFERRED_OUTPUT_CONTRACTS} entries"
            )
        normalized_contracts: list[dict[str, str]] = []
        seen_contract_outputs: set[str] = set()
        for contract_index, contract in enumerate(deferred_output_contracts):
            contract_label = f"{label} deferred output contract #{contract_index + 1}"
            if not isinstance(contract, dict) or set(contract) != {
                "output",
                "reason",
            }:
                raise ValueError(
                    f"{contract_label} must contain exactly output and reason"
                )
            output = contract["output"]
            reason = contract["reason"]
            if any(
                not isinstance(field, str)
                or not field
                or field != field.strip()
                or "\r" in field
                or any(
                    (ord(character) < 32 and character not in {"\n", "\t"})
                    or ord(character) == 127
                    for character in field
                )
                for field in (output, reason)
            ):
                raise ValueError(
                    f"{contract_label} fields must be nonempty normalized strings"
                )
            if output in seen_contract_outputs:
                raise ValueError(
                    f"{label} deferred output contract outputs must be unique"
                )
            seen_contract_outputs.add(output)
            normalized_contracts.append({"output": output, "reason": reason})
        try:
            citation = require_canonical_corpus_citation_path(citation)
        except ValueError as exc:
            raise ValueError(
                f"{label} citation must be an exact canonical corpus citation path"
            ) from exc
        jurisdiction, _relative = _citation_rulespec_path(citation)
        if jurisdiction != primary_jurisdiction:
            raise ValueError(
                "canonical refresh targets must use the primary citation "
                "jurisdiction and country"
            )
        path = _safe_relative_path(raw_path, label=f"{label} RuleSpec path")
        expected_path = citation_rulespec_path(citation)
        if path != expected_path:
            raise ValueError(
                f"{label} path must equal the citation's canonical RuleSpec path"
            )
        _validate_wrapped_review_contract_size(
            citation=citation,
            path=path,
            deferred_output_contracts=tuple(normalized_contracts),
            required_test_cases=required_test_cases,
            label=label,
        )
        if citation in seen_citations or path in seen_paths:
            raise ValueError("canonical refresh citations and paths must be unique")
        requested.append(
            (
                citation,
                path,
                review_finding,
                tuple(normalized_contracts),
                required_test_cases,
            )
        )
        seen_citations.add(citation)
        seen_paths.add(path)

    return tuple(
        _canonical_refresh_target_inventory(
            repo,
            citation=citation,
            path=path,
            review_finding=review_finding,
            deferred_output_contracts=deferred_output_contracts,
            required_test_cases=required_test_cases,
        )
        for (
            citation,
            path,
            review_finding,
            deferred_output_contracts,
            required_test_cases,
        ) in requested
    )


def _canonical_refresh_target_inventory(
    repo: Path,
    *,
    citation: str,
    path: PurePosixPath,
    review_finding: str | None,
    deferred_output_contracts: tuple[dict[str, str], ...],
    required_test_cases: tuple[dict[str, object], ...],
) -> dict[str, object]:
    """Bind one refresh target and its untrusted predecessor manifest to HEAD."""

    manifest_path = _existing_import_manifest_path(path)
    companion_path = path.with_name(f"{path.stem}.test.yaml")
    raw_by_path: dict[PurePosixPath, bytes] = {}
    for tracked_path, label, max_bytes in (
        (path, f"canonical refresh RuleSpec for {citation}", 10 * 1024 * 1024),
        (manifest_path, f"canonical refresh manifest for {citation}", 1024 * 1024),
    ):
        try:
            stage = _git(
                repo,
                "ls-files",
                "--stage",
                "--",
                tracked_path.as_posix(),
            ).decode("utf-8")
        except (subprocess.CalledProcessError, UnicodeDecodeError) as exc:
            raise ValueError(f"{label} must be exactly tracked 100644") from exc
        lines = stage.splitlines()
        if (
            len(lines) != 1
            or not lines[0].startswith("100644 ")
            or lines[0].split("\t", 1)[-1] != tracked_path.as_posix()
        ):
            raise ValueError(f"{label} must be exactly tracked 100644")
        raw = _read_bounded_regular(
            repo,
            tracked_path,
            label=label,
            max_bytes=max_bytes,
        )
        try:
            base_raw = _git(repo, "show", f"HEAD:{tracked_path.as_posix()}")
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"{label} is absent from HEAD") from exc
        if raw != base_raw:
            raise ValueError(f"{label} differs from HEAD")
        raw_by_path[tracked_path] = raw

    companion_label = f"canonical refresh companion for {citation}"
    try:
        companion_stage = _git(
            repo,
            "ls-files",
            "--stage",
            "--",
            companion_path.as_posix(),
        ).decode("utf-8")
    except (subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"{companion_label} tracked state cannot be determined"
        ) from exc
    companion_lines = companion_stage.splitlines()
    if companion_lines:
        if (
            len(companion_lines) != 1
            or not companion_lines[0].startswith("100644 ")
            or companion_lines[0].split("\t", 1)[-1] != companion_path.as_posix()
        ):
            raise ValueError(f"{companion_label} must be exactly tracked 100644")
        companion_raw = _read_bounded_regular(
            repo,
            companion_path,
            label=companion_label,
            max_bytes=10 * 1024 * 1024,
        )
        try:
            companion_base_raw = _git(
                repo,
                "show",
                f"HEAD:{companion_path.as_posix()}",
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"{companion_label} is absent from HEAD") from exc
        if companion_raw != companion_base_raw:
            raise ValueError(f"{companion_label} differs from HEAD")
        companion_sha256: str | None = hashlib.sha256(companion_raw).hexdigest()
    else:
        companion = repo.joinpath(*companion_path.parts)
        if companion.exists() or companion.is_symlink():
            raise ValueError(f"{companion_label} is untracked")
        if _git(
            repo,
            "ls-tree",
            "--full-tree",
            "-z",
            "HEAD",
            "--",
            companion_path.as_posix(),
        ):
            raise ValueError(f"{companion_label} differs from HEAD")
        companion_sha256 = None

    try:
        manifest = json.loads(raw_by_path[manifest_path].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(
            f"canonical refresh manifest for {citation} is invalid JSON"
        ) from exc
    target_sha256 = hashlib.sha256(raw_by_path[path]).hexdigest()
    applied_files = (
        manifest.get("applied_files") if isinstance(manifest, dict) else None
    )
    target_entries = [
        item
        for item in applied_files or []
        if isinstance(item, dict) and item.get("path") == path.as_posix()
    ]
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "axiom-encode/applied-rulespec/v5"
        or manifest.get("tool") != MODEL_APPLY_TOOL
        or manifest.get("citation") != citation
        or not isinstance(applied_files, list)
        or len(target_entries) != 1
        or target_entries[0].get("sha256") != target_sha256
    ):
        raise ValueError(
            f"canonical refresh manifest for {citation} does not structurally "
            "cover the requested target"
        )
    return {
        "citation": citation,
        "review_finding": review_finding,
        "deferred_output_contracts": list(deferred_output_contracts),
        "required_test_cases": list(required_test_cases),
        "rulespec_path": path.as_posix(),
        "rulespec_sha256": target_sha256,
        "companion_path": companion_path.as_posix(),
        "companion_sha256": companion_sha256,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": hashlib.sha256(raw_by_path[manifest_path]).hexdigest(),
    }


def verify_canonical_refresh_target(
    repo: Path,
    target_json: str,
) -> dict[str, object]:
    """Require one normalized target to remain byte-identical before its lane."""

    try:
        target = _load_unambiguous_json(
            target_json,
            label="canonical refresh target",
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("canonical refresh target is invalid JSON") from exc
    expected_fields = {
        "citation",
        "rulespec_path",
        "rulespec_sha256",
        "companion_path",
        "companion_sha256",
        "manifest_path",
        "manifest_sha256",
        "review_finding",
        "deferred_output_contracts",
        "required_test_cases",
    }
    deferred_output_contracts = (
        target.get("deferred_output_contracts") if isinstance(target, dict) else None
    )
    try:
        required_test_cases = _normalize_required_test_cases(
            target.get("required_test_cases") if isinstance(target, dict) else None,
            label="canonical refresh target required test cases",
        )
    except ValueError as exc:
        raise ValueError("canonical refresh target inventory is malformed") from exc
    if (
        not isinstance(target, dict)
        or set(target) != expected_fields
        or not all(
            isinstance(target[field], str) and target[field]
            for field in expected_fields
            - {
                "companion_sha256",
                "review_finding",
                "deferred_output_contracts",
                "required_test_cases",
            }
        )
        or (
            target["companion_sha256"] is not None
            and (
                not isinstance(target["companion_sha256"], str)
                or DIGEST_PATTERN.fullmatch(target["companion_sha256"]) is None
            )
        )
        or DIGEST_PATTERN.fullmatch(target["rulespec_sha256"]) is None
        or DIGEST_PATTERN.fullmatch(target["manifest_sha256"]) is None
        or not isinstance(deferred_output_contracts, list)
        or len(deferred_output_contracts) > MAX_DEFERRED_OUTPUT_CONTRACTS
        or any(
            not isinstance(contract, dict)
            or set(contract) != {"output", "reason"}
            or any(
                not isinstance(contract[field], str)
                or not contract[field]
                or contract[field] != contract[field].strip()
                or "\r" in contract[field]
                or any(
                    (ord(character) < 32 and character not in {"\n", "\t"})
                    or ord(character) == 127
                    for character in contract[field]
                )
                for field in ("output", "reason")
            )
            for contract in deferred_output_contracts or []
        )
        or len(
            {
                contract["output"]
                for contract in deferred_output_contracts or []
                if isinstance(contract, dict)
                and isinstance(contract.get("output"), str)
            }
        )
        != len(deferred_output_contracts or [])
        or list(required_test_cases) != target["required_test_cases"]
        or (
            target["review_finding"] is not None
            and (
                not isinstance(target["review_finding"], str)
                or not target["review_finding"]
                or target["review_finding"] != target["review_finding"].strip()
                or "\r" in target["review_finding"]
                or any(
                    (ord(character) < 32 and character not in {"\n", "\t"})
                    or ord(character) == 127
                    for character in target["review_finding"]
                )
            )
        )
    ):
        raise ValueError("canonical refresh target inventory is malformed")
    repo = repo.resolve(strict=True)
    rulespec_path = _safe_relative_path(
        target["rulespec_path"], label="canonical refresh target RuleSpec"
    )
    manifest_path = _safe_relative_path(
        target["manifest_path"], label="canonical refresh target manifest"
    )
    companion_path = _safe_relative_path(
        target["companion_path"], label="canonical refresh target companion"
    )
    _validate_wrapped_review_contract_size(
        citation=target["citation"],
        path=rulespec_path,
        deferred_output_contracts=tuple(deferred_output_contracts),
        required_test_cases=required_test_cases,
        label="canonical refresh target",
    )
    if (
        citation_rulespec_path(target["citation"]) != rulespec_path
        or _existing_import_manifest_path(rulespec_path) != manifest_path
        or rulespec_path.with_name(f"{rulespec_path.stem}.test.yaml") != companion_path
    ):
        raise ValueError("canonical refresh target inventory paths are inconsistent")
    for path, digest, label, max_bytes in (
        (
            rulespec_path,
            target["rulespec_sha256"],
            "canonical refresh target RuleSpec",
            10 * 1024 * 1024,
        ),
        (
            manifest_path,
            target["manifest_sha256"],
            "canonical refresh target manifest",
            1024 * 1024,
        ),
    ):
        raw = _read_bounded_regular(repo, path, label=label, max_bytes=max_bytes)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"{label} changed before its signed refresh lane")
    companion_sha256 = target["companion_sha256"]
    if companion_sha256 is None:
        companion = repo.joinpath(*companion_path.parts)
        if companion.exists() or companion.is_symlink():
            raise ValueError(
                "canonical refresh target companion changed before its signed "
                "refresh lane"
            )
    else:
        companion_raw = _read_bounded_regular(
            repo,
            companion_path,
            label="canonical refresh target companion",
            max_bytes=10 * 1024 * 1024,
        )
        if hashlib.sha256(companion_raw).hexdigest() != companion_sha256:
            raise ValueError(
                "canonical refresh target companion changed before its signed "
                "refresh lane"
            )
    return target


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


def _git_quiet(repo: Path, *args: str) -> bytes:
    """Run a Git probe whose failure is handled without leaking diagnostics."""

    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        stderr=subprocess.PIPE,
    )


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


def _rulespec_manifest_path(primary: PurePosixPath) -> PurePosixPath:
    return MANIFEST_ROOT / primary.with_suffix(".json")


def _retired_manifest_inventory_without_entry(
    raw: bytes,
    manifest_path: PurePosixPath,
) -> bytes | None:
    """Remove one exact literal from the canonical retired-manifest set."""

    manifest = manifest_path.as_posix()
    try:
        text = raw.decode("utf-8")
        module = ast.parse(text, filename=RETIRED_MANIFEST_INVENTORY.as_posix())
    except (UnicodeError, SyntaxError, ValueError) as exc:
        raise ValueError(
            "retired manifest inventory is not valid UTF-8 Python"
        ) from exc

    inventory_name_present = any(
        token.type == tokenize.NAME and token.string == "KNOWN_RETIRED_SCHEMA_MANIFESTS"
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
    )
    assignments: list[ast.AnnAssign] = []
    for node in ast.walk(module):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "KNOWN_RETIRED_SCHEMA_MANIFESTS"
        ):
            assignments.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "KNOWN_RETIRED_SCHEMA_MANIFESTS"
            for target in node.targets
        ):
            raise ValueError("retired manifest inventory assignment is not canonical")
    if not inventory_name_present:
        if manifest in text:
            raise ValueError(
                "retired manifest inventory path is present without an inventory"
            )
        return None
    if len(assignments) != 1 or assignments[0] not in module.body:
        raise ValueError(
            "retired manifest inventory lacks one canonical top-level assignment"
        )
    assignment = assignments[0]
    value = assignment.value
    if (
        not isinstance(value, ast.Call)
        or not isinstance(value.func, ast.Name)
        or value.func.id != "frozenset"
        or value.keywords
        or len(value.args) > 1
        or (len(value.args) == 1 and not isinstance(value.args[0], ast.Set))
    ):
        raise ValueError("retired manifest inventory set shape is not canonical")
    elements = value.args[0].elts if value.args else []
    if not all(
        isinstance(element, ast.Constant) and isinstance(element.value, str)
        for element in elements
    ):
        raise ValueError("retired manifest inventory contains a non-literal entry")
    values = [element.value for element in elements]
    if len(values) != len(set(values)):
        raise ValueError("retired manifest inventory contains duplicate entries")

    matches = [
        element
        for element in elements
        if isinstance(element, ast.Constant) and element.value == manifest
    ]
    textual_matches = text.count(manifest)
    if not matches:
        if textual_matches:
            raise ValueError(
                "retired manifest inventory path is present outside an exact entry"
            )
        return None
    if len(matches) != 1 or textual_matches != 1:
        raise ValueError("retired manifest inventory path match is ambiguous")

    entry = matches[0]
    if entry.end_lineno != entry.lineno:
        raise ValueError("retired manifest inventory entry is not one canonical line")
    lines = text.splitlines(keepends=True)
    if entry.lineno < 1 or entry.lineno > len(lines):
        raise ValueError("retired manifest inventory entry location is invalid")
    line = lines[entry.lineno - 1]
    if line not in {f"    '{manifest}',\n", f'    "{manifest}",\n'}:
        raise ValueError("retired manifest inventory match is not an exact entry")
    if len(elements) == 1:
        start_line = value.lineno - 1
        end_line = value.end_lineno - 1
        lines[start_line : end_line + 1] = [
            lines[start_line][: value.col_offset]
            + "frozenset()"
            + lines[end_line][value.end_col_offset :]
        ]
    else:
        del lines[entry.lineno - 1]
    rewritten = "".join(lines).encode("utf-8")
    if manifest.encode("utf-8") in rewritten:
        raise ValueError("retired manifest inventory still contains the removed path")
    try:
        compile(rewritten, RETIRED_MANIFEST_INVENTORY.as_posix(), "exec")
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            "retired manifest inventory removal produced invalid Python"
        ) from exc
    return rewritten


def _require_normal_model_apply_target_binding(
    repo: Path,
    target: PurePosixPath,
    payload: dict[str, object],
    *,
    label: str,
) -> None:
    """Require one normal model manifest to hash-bind its canonical primary."""

    signature = payload.get("signature")
    if (
        payload.get("schema_version") != APPLIED_MANIFEST_SCHEMA_V5
        or payload.get("tool") != MODEL_APPLY_TOOL
        or payload.get("backend") not in MODEL_APPLY_BACKENDS
        or not isinstance(signature, dict)
        or set(signature) != {"algorithm", "key_id", "value"}
        or signature.get("algorithm") != APPLIED_MANIFEST_SIGNATURE_ALGORITHM
        or not all(
            isinstance(signature.get(field), str) and signature[field]
            for field in ("key_id", "value")
        )
    ):
        raise ValueError(f"{label} is not an exact signed-v5 model apply")
    applied_files = payload.get("applied_files")
    target_records = (
        [
            item
            for item in applied_files
            if isinstance(item, dict) and item.get("path") == target.as_posix()
        ]
        if isinstance(applied_files, list)
        else []
    )
    target_raw = _read_bounded_regular(
        repo,
        target,
        label=f"{label} RuleSpec",
        max_bytes=16 * 1024 * 1024,
    )
    if (
        len(target_records) != 1
        or set(target_records[0]) != {"path", "sha256"}
        or target_records[0].get("sha256") != hashlib.sha256(target_raw).hexdigest()
    ):
        raise ValueError(f"{label} does not bind exact target bytes")


def _normal_model_apply_manifest_for_target(
    repo: Path,
    target: PurePosixPath,
) -> tuple[PurePosixPath, dict[str, object]]:
    """Bind one changed signed-v5 model manifest to its exact target bytes."""

    manifest_path = _rulespec_manifest_path(target)
    changed = _changed_paths(repo)
    if manifest_path not in changed:
        raise ValueError("targeted replacement lacks its exact changed apply manifest")
    try:
        payload = _load_unambiguous_json(
            _read_bounded_regular(
                repo,
                manifest_path,
                label="targeted replacement apply manifest",
                max_bytes=1024 * 1024,
            ).decode("utf-8"),
            label="targeted replacement apply manifest",
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("targeted replacement apply manifest is malformed") from exc
    if not isinstance(payload, dict):
        raise ValueError("targeted replacement apply manifest is malformed")
    _require_normal_model_apply_target_binding(
        repo,
        target,
        payload,
        label="targeted replacement apply manifest",
    )
    return manifest_path, payload


def reconcile_retired_manifest_inventory(
    repo: Path,
    target_rulespec_path: str,
) -> PurePosixPath | None:
    """Retire one stale schema allowance after a normal signed replacement."""

    target = _safe_relative_path(
        target_rulespec_path,
        label="target RuleSpec path",
    )
    _validate_rulespec_path(repo, target, label="target RuleSpec path")
    if target.name.endswith(".test.yaml"):
        raise ValueError("target RuleSpec path must be a primary module")
    manifest_path, _payload = _normal_model_apply_manifest_for_target(repo, target)
    if RETIRED_MANIFEST_INVENTORY in _changed_paths(repo):
        raise ValueError(
            "retired manifest inventory changed before exact reconciliation"
        )
    inventory_tree_entry = _git(
        repo,
        "ls-tree",
        "--full-tree",
        "-z",
        "HEAD",
        "--",
        RETIRED_MANIFEST_INVENTORY.as_posix(),
    )
    if not inventory_tree_entry:
        try:
            inventory_present = _checkout_path_exists_without_indirection(
                repo,
                RETIRED_MANIFEST_INVENTORY,
                label="retired manifest inventory",
            )
        except ValueError as exc:
            raise ValueError(
                "retired manifest inventory is absent from HEAD but not safely "
                "absent from the worktree"
            ) from exc
        if inventory_present:
            raise ValueError(
                "retired manifest inventory is absent from HEAD but not safely "
                "absent from the worktree"
            )
        return None
    try:
        base_raw = _git(
            repo,
            "show",
            f"HEAD:{RETIRED_MANIFEST_INVENTORY.as_posix()}",
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError("retired manifest inventory is absent from HEAD") from exc
    live_raw = _read_bounded_regular(
        repo,
        RETIRED_MANIFEST_INVENTORY,
        label="retired manifest inventory",
        max_bytes=16 * 1024 * 1024,
    )
    if live_raw != base_raw:
        raise ValueError("retired manifest inventory differs from clean HEAD")
    rewritten = _retired_manifest_inventory_without_entry(base_raw, manifest_path)
    if rewritten is None:
        return None

    _secure_replace_regular_file(
        repo,
        RETIRED_MANIFEST_INVENTORY,
        expected=live_raw,
        replacement=rewritten,
    )
    observed = _read_bounded_regular(
        repo,
        RETIRED_MANIFEST_INVENTORY,
        label="reconciled retired manifest inventory",
        max_bytes=16 * 1024 * 1024,
    )
    if observed != rewritten:
        raise ValueError("retired manifest inventory changed during reconciliation")
    return manifest_path


def _secure_replace_regular_file(
    repo: Path,
    relative: PurePosixPath,
    *,
    expected: bytes,
    replacement: bytes,
) -> None:
    """Replace one existing 0644 file through symlink-free directory handles."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("secure reconciliation requires no-follow directory opens")
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow | directory
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | nofollow
    )
    descriptors: list[int] = []

    def read_current(parent_fd: int, name: str) -> bytes:
        descriptor = os.open(name, file_flags, dir_fd=parent_fd)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o644
                or metadata.st_nlink != 1
                or metadata.st_uid != os.geteuid()
                or metadata.st_size > 16 * 1024 * 1024
            ):
                raise ValueError("reconciliation target is not one owned 0644 file")
            chunks: list[bytes] = []
            remaining = 16 * 1024 * 1024 + 1
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > 16 * 1024 * 1024:
                raise ValueError("reconciliation target exceeds its size limit")
            return raw
        finally:
            os.close(descriptor)

    temporary_name: str | None = None
    temporary_fd: int | None = None
    try:
        current_fd = os.open(repo.resolve(strict=True), directory_flags)
        descriptors.append(current_fd)
        for component in relative.parts[:-1]:
            current_fd = os.open(component, directory_flags, dir_fd=current_fd)
            descriptors.append(current_fd)
        target_name = relative.name
        if read_current(current_fd, target_name) != expected:
            raise ValueError("reconciliation target changed before secure replacement")
        create_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | nofollow
        )
        for _attempt in range(100):
            candidate = f".{target_name}.{os.urandom(16).hex()}.tmp"
            try:
                temporary_fd = os.open(
                    candidate,
                    create_flags,
                    0o600,
                    dir_fd=current_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if temporary_fd is None or temporary_name is None:
            raise OSError("could not reserve a reconciliation temporary file")
        remaining = memoryview(replacement)
        while remaining:
            written = os.write(temporary_fd, remaining)
            if written <= 0:
                raise OSError("short write during retired inventory reconciliation")
            remaining = remaining[written:]
        os.fchmod(temporary_fd, 0o644)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        if read_current(current_fd, target_name) != expected:
            raise ValueError("reconciliation target changed during secure replacement")
        os.replace(
            temporary_name,
            target_name,
            src_dir_fd=current_fd,
            dst_dir_fd=current_fd,
        )
        temporary_name = None
        os.fsync(current_fd)
        if read_current(current_fd, target_name) != replacement:
            raise ValueError("secure reconciliation replacement bytes differ")
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_name is not None and descriptors:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=descriptors[-1])
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def authorize_legacy_index_manifest_shrink(
    repo: Path,
    target_rulespec_path: str,
) -> bool:
    """Authorize replacing one migration manifest that has a stale index claim."""

    target = _safe_relative_path(
        target_rulespec_path,
        label="target RuleSpec path",
    )
    _validate_rulespec_path(repo, target, label="target RuleSpec path")
    companion = _rulespec_companion_path(target)
    manifest_relative = MANIFEST_ROOT / target.with_suffix(".json")
    manifest_path = repo / manifest_relative
    try:
        manifest_path.lstat()
    except FileNotFoundError:
        return False
    payload = json.loads(
        _read_bounded_regular(
            repo,
            manifest_relative,
            label="existing target apply manifest",
            max_bytes=8 * 1024 * 1024,
        ).decode("utf-8")
    )
    if not isinstance(payload, dict) or payload.get("tool") != LEGACY_REPLACEMENT_TOOL:
        return False
    if payload.get("schema_version") != APPLIED_MANIFEST_SCHEMA_V5:
        raise ValueError("legacy replacement manifest is not current v5")

    entries = payload.get("applied_files")
    if not isinstance(entries, list):
        raise ValueError("legacy replacement manifest has malformed applied_files")
    live: dict[PurePosixPath, str] = {}
    deleted: list[PurePosixPath] = []
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"applied_files[{index}] is not an object")
        path = _safe_relative_path(
            item.get("path"),
            label=f"applied_files[{index}].path",
        )
        if item.get("deleted") is True:
            if set(item) != {"path", "deleted"}:
                raise ValueError(f"applied_files[{index}] deletion is malformed")
            deleted.append(path)
            continue
        digest = item.get("sha256")
        if set(item) != {"path", "sha256"} or not isinstance(digest, str):
            raise ValueError(f"applied_files[{index}] live entry is malformed")
        if DIGEST_PATTERN.fullmatch(digest) is None or path in live:
            raise ValueError(f"applied_files[{index}] digest or path is malformed")
        live[path] = digest

    expected_live = {target, companion, PROVISIONS_TO_RULES_INDEX}
    if set(live) != expected_live or len(deleted) != 2:
        return False
    deleted_primary = next(
        (path for path in deleted if not path.name.endswith(".test.yaml")),
        None,
    )
    if (
        deleted_primary is None
        or deleted_primary == target
        or set(deleted) != {deleted_primary, _rulespec_companion_path(deleted_primary)}
    ):
        return False

    replacement = payload.get("replacement")
    if not isinstance(replacement, dict):
        raise ValueError("legacy replacement manifest has no replacement receipt")
    expected_legacy_manifest = MANIFEST_ROOT / deleted_primary.with_suffix(".json")
    if replacement.get("legacy_manifest_path") != expected_legacy_manifest.as_posix():
        return False
    receipt_path = _safe_relative_path(
        replacement.get("receipt_path"),
        label="replacement receipt path",
    )
    receipt_digest = replacement.get("receipt_sha256")
    if (
        not receipt_path.is_relative_to(LEGACY_REPLACEMENT_RECEIPT_ROOT)
        or not isinstance(receipt_digest, str)
        or DIGEST_PATTERN.fullmatch(receipt_digest) is None
    ):
        raise ValueError("legacy replacement receipt binding is malformed")
    receipt_raw = _read_bounded_regular(
        repo,
        receipt_path,
        label="legacy replacement receipt",
        max_bytes=8 * 1024 * 1024,
    )
    if hashlib.sha256(receipt_raw).hexdigest() != receipt_digest:
        raise ValueError("legacy replacement receipt digest does not match")

    embedded = payload.get("replacement_manifest")
    if (
        not isinstance(embedded, dict)
        or embedded.get("schema_version") != APPLIED_MANIFEST_SCHEMA_V5
        or embedded.get("tool") != MODEL_APPLY_TOOL
        or embedded.get("backend") not in MODEL_APPLY_BACKENDS
    ):
        raise ValueError("embedded model apply manifest is malformed")
    embedded_entries = embedded.get("applied_files")
    if not isinstance(embedded_entries, list):
        raise ValueError("embedded model apply manifest has malformed applied_files")
    embedded_live: dict[PurePosixPath, str] = {}
    for index, item in enumerate(embedded_entries):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError(f"embedded applied_files[{index}] is malformed")
        path = _safe_relative_path(
            item.get("path"),
            label=f"embedded applied_files[{index}].path",
        )
        digest = item.get("sha256")
        if (
            not isinstance(digest, str)
            or DIGEST_PATTERN.fullmatch(digest) is None
            or path in embedded_live
        ):
            raise ValueError(
                f"embedded applied_files[{index}] digest or path is malformed"
            )
        embedded_live[path] = digest
    expected_model_live = {target: live[target], companion: live[companion]}
    if embedded_live != expected_model_live:
        return False

    for path in (target, companion):
        raw = _read_bounded_regular(
            repo,
            path,
            label=f"legacy replacement live file {path}",
            max_bytes=32 * 1024 * 1024,
        )
        if hashlib.sha256(raw).hexdigest() != live[path]:
            raise ValueError(f"legacy replacement live file is stale: {path}")
    current_index = _read_bounded_regular(
        repo,
        PROVISIONS_TO_RULES_INDEX,
        label="provisions-to-rules index",
        max_bytes=64 * 1024 * 1024,
    )
    return hashlib.sha256(current_index).hexdigest() != live[PROVISIONS_TO_RULES_INDEX]


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
    corpus_release: object | None,
) -> set[tuple[PurePosixPath, PurePosixPath]]:
    """Verify v2 exact dependents and return manifest-bound unchanged claims."""

    from axiom_encode.cli import (
        _legacy_primary_source_citations,
        _reanchor_legacy_exact_dependent_proof_excerpts,
        _repair_proof_import_hashes,
        _strict_legacy_replacement_map,
    )
    from axiom_encode.legacy_exact_dependent_concepts import (
        canonicalized_concept_replacements,
        derive_exact_dependent_parameter_replacements,
        validate_exact_dependent_concept_rewrite,
    )
    from axiom_encode.legacy_replacement import (
        legacy_receipt_v1_manifest_issues,
        migrate_legacy_exact_dependent_source_verification,
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
    unchanged_authorized: set[tuple[PurePosixPath, PurePosixPath]] = set()
    seen_primaries: set[PurePosixPath] = set()
    seen_group_paths: set[PurePosixPath] = set()
    seen_manifests: set[PurePosixPath] = set()
    receipt_schema = receipt.get("schema_version")
    replacement_source = _safe_relative_path(
        receipt_replacement.get("source"),
        label=f"{root_manifest}.replacement.source",
    )
    replacement_source_raw = _base_regular_blob(
        repo,
        base_commit,
        replacement_source,
        required=True,
    )
    assert replacement_source_raw is not None
    replacement_source_citations = _legacy_primary_source_citations(
        replacement_source_raw
    )
    retained_modules: list[tuple[Path, Path, bytes, bytes]] = []
    if receipt_schema == LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7:
        raw_successors = receipt_replacement.get("retained_successors")
        if not isinstance(raw_successors, list):
            raise ValueError("legacy replacement retained successors are malformed")
        for successor in raw_successors:
            if not isinstance(successor, dict):
                raise ValueError("legacy replacement retained successor is malformed")
            source = _safe_relative_path(
                successor.get("source"), label="retained successor source"
            )
            destination = _safe_relative_path(
                successor.get("destination"), label="retained successor destination"
            )
            source_raw = _base_regular_blob(repo, base_commit, source, required=True)
            destination_raw = _base_regular_blob(
                repo, base_commit, destination, required=True
            )
            assert source_raw is not None and destination_raw is not None
            retained_modules.append(
                (
                    Path(source),
                    Path(destination),
                    source_raw,
                    destination_raw,
                )
            )
    for index, raw_dependent in enumerate(raw_dependents):
        label = f"{root_manifest} exact_dependents[{index}]"
        expected_dependent_fields = {
            "primary",
            "legacy_manifest",
            "legacy_files",
            "live_files",
            "rewrites",
        }
        if receipt_schema in {
            LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
            LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
            LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
        }:
            expected_dependent_fields.add("source_verification_migration")
        if receipt_schema == LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7:
            expected_dependent_fields.add("concept_replacements")
        if (
            not isinstance(raw_dependent, dict)
            or set(raw_dependent) != expected_dependent_fields
        ):
            raise ValueError(f"{label} is malformed")
        receipt_source_verification_migration = (
            raw_dependent.get("source_verification_migration")
            if receipt_schema
            in {
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
            }
            else None
        )
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

        try:
            base_manifest_payload = json.loads(base_manifest_raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError, RecursionError):
            base_manifest_payload = None
        primary_citations = _legacy_primary_source_citations(base_by_path[primary])
        receipt_legacy = receipt.get("legacy")
        legacy_issues = legacy_receipt_v1_manifest_issues(
            base_manifest_payload,
            owner_class=(
                receipt_legacy.get("owner_class")
                if isinstance(receipt_legacy, dict)
                else None
            ),
            expected_files={
                path.as_posix(): str(item["sha256"])
                for path, item in zip(expected_paths, legacy_files, strict=True)
            },
            expected_primary_path=primary.as_posix(),
            expected_citation=primary_citations[0] if primary_citations else "",
            jurisdiction_prefix=primary.parts[0],
            allow_unmarked_manual_exception=True,
        )
        if legacy_issues:
            raise ValueError(
                f"{label} exact dependent legacy ownership is invalid: "
                + "; ".join(legacy_issues)
            )

        expected_source_verification_migration = None
        source_verification_migration = None
        if receipt_schema in {
            LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
            LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
            LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
        }:
            _unused_primary, source_verification_migration = (
                migrate_legacy_exact_dependent_source_verification(
                    base_by_path[primary]
                )
            )
            expected_source_verification_migration = (
                {
                    "legacy_corpus_citation_paths": list(
                        source_verification_migration.legacy_corpus_citation_paths
                    ),
                    "corpus_citation_path": (
                        source_verification_migration.corpus_citation_path
                    ),
                }
                if source_verification_migration is not None
                else None
            )
            if (
                receipt_source_verification_migration
                != expected_source_verification_migration
            ):
                raise ValueError(f"{label}.source_verification_migration differs")
            if (
                source_verification_migration is not None
                and source_verification_migration.corpus_citation_path
                != (
                    replacement_source_citations[0]
                    if replacement_source_citations
                    else None
                )
            ):
                raise ValueError(
                    f"{label}.source_verification_migration source differs"
                )

        raw_rewrites = raw_dependent.get("rewrites")
        if not isinstance(raw_rewrites, list) or not raw_rewrites:
            raise ValueError(f"{label}.rewrites is malformed")
        exact_authoritative_replacements = authoritative_replacements
        exact_concept_replacements: dict[str, str] = {}
        if receipt_schema == LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7:
            derived_concepts = derive_exact_dependent_parameter_replacements(
                dependent_primary_raw=base_by_path[primary],
                retained_modules=retained_modules,
            )
            expected_concept_records = [
                {"from": old, "to": new}
                for old, new in sorted(derived_concepts.items())
            ]
            if raw_dependent.get("concept_replacements") != expected_concept_records:
                raise ValueError(f"{label} concept rewrite proof differs")
            exact_concept_replacements = derived_concepts
            exact_authoritative_replacements = {
                **authoritative_replacements,
                **exact_concept_replacements,
            }
        rewrite_paths: set[PurePosixPath] = set()
        for rewrite_index, rewrite in enumerate(raw_rewrites):
            rewrite_label = f"{label}.rewrites[{rewrite_index}]"
            expected_rewrite_fields = {
                "path",
                "before_sha256",
                "after_sha256",
                "replacements",
                "proof_import_repairs",
            }
            if receipt_schema in {
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
            }:
                expected_rewrite_fields.add("proof_excerpt_reanchors")
            if not isinstance(rewrite, dict) or set(rewrite) != expected_rewrite_fields:
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
                or (
                    receipt_schema
                    in {
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
                    }
                    and not isinstance(rewrite.get("proof_excerpt_reanchors"), list)
                )
            ):
                raise ValueError(f"{rewrite_label} proof is malformed")
            try:
                expected_live, observed_counts = rewrite_exact_references(
                    base_by_path[rewrite_path],
                    exact_authoritative_replacements,
                )
                if exact_concept_replacements:
                    path_rewritten, _path_counts = rewrite_exact_references(
                        base_by_path[rewrite_path],
                        authoritative_replacements,
                    )
                    canonical_concepts = canonicalized_concept_replacements(
                        exact_concept_replacements,
                        path_replacements=authoritative_replacements,
                    )
                    concept_rewritten, _concept_counts = rewrite_exact_references(
                        path_rewritten,
                        canonical_concepts,
                    )
                    if concept_rewritten != expected_live:
                        raise ValueError(
                            f"{rewrite_label} concept rewrite order differs"
                        )
                    validate_exact_dependent_concept_rewrite(
                        path_rewritten_raw=path_rewritten,
                        concept_rewritten_raw=concept_rewritten,
                        replacements=canonical_concepts,
                        primary=rewrite_path == primary,
                    )
                observed_proof_repairs = 0
                if rewrite_path == primary:
                    if receipt_schema in {
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
                    }:
                        expected_live, observed_source_migration = (
                            migrate_legacy_exact_dependent_source_verification(
                                expected_live
                            )
                        )
                        if observed_source_migration != source_verification_migration:
                            raise ValueError(
                                f"{label}.source_verification_migration replay differs"
                            )
                    elif source_verification_migration is not None:
                        raise ValueError(
                            f"{label}.source_verification_migration schema differs"
                        )
                    if receipt_schema in {
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
                    }:
                        if corpus_release is None:
                            raise ValueError(
                                f"{rewrite_label} proof-excerpt verification requires "
                                "the authenticated signed corpus"
                            )
                        expected_live, observed_reanchors = (
                            _reanchor_legacy_exact_dependent_proof_excerpts(
                                expected_live,
                                corpus_release=corpus_release,
                            )
                        )
                        if (
                            list(observed_reanchors)
                            != rewrite["proof_excerpt_reanchors"]
                        ):
                            raise ValueError(
                                f"{rewrite_label} proof-excerpt corpus replay differs"
                            )
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
                elif (
                    receipt_schema
                    in {
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
                    }
                    and rewrite["proof_excerpt_reanchors"]
                ):
                    raise ValueError(
                        f"{rewrite_label} companion has proof-excerpt rewrites"
                    )
            except ValueError as exc:
                raise ValueError(
                    f"{rewrite_label} transformation is invalid: {exc}"
                ) from exc
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

        if (
            receipt_schema
            in {
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
            }
            and primary not in rewrite_paths
            and source_verification_migration is not None
        ):
            raise ValueError(
                f"{label}.source_verification_migration lacks a primary rewrite"
            )

        for path in expected_paths:
            if path in rewrite_paths:
                continue
            if base_by_path[path] != live_by_path[path] or path in changed:
                raise ValueError(f"{label} unrewritten file differs: {path}")
            unchanged_authorized.add((expected_manifest, path))

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


def _authorized_retired_manifest_inventory_reconciliation(
    repo: Path,
    *,
    manifest_payloads: dict[PurePosixPath, dict[str, object]],
) -> PurePosixPath:
    """Authenticate the one unsigned inventory delta derived from signed paths."""

    try:
        base_raw = _git(
            repo,
            "show",
            f"HEAD:{RETIRED_MANIFEST_INVENTORY.as_posix()}",
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError("retired manifest inventory is absent from HEAD") from exc
    live_raw = _read_bounded_regular(
        repo,
        RETIRED_MANIFEST_INVENTORY,
        label="changed retired manifest inventory",
        max_bytes=16 * 1024 * 1024,
    )
    candidates: list[PurePosixPath] = []
    for manifest_path, payload in sorted(
        manifest_payloads.items(),
        key=lambda item: item[0].as_posix(),
    ):
        if (
            payload.get("tool") != MODEL_APPLY_TOOL
            or payload.get("backend") not in MODEL_APPLY_BACKENDS
        ):
            continue
        target = PurePosixPath(
            *manifest_path.relative_to(MANIFEST_ROOT).parts
        ).with_suffix(".yaml")
        try:
            _validate_rulespec_path(
                repo,
                target,
                label="retired inventory model apply target",
            )
            if target.name.endswith(".test.yaml"):
                raise ValueError("model apply target is not a primary module")
            _require_normal_model_apply_target_binding(
                repo,
                target,
                payload,
                label="retired inventory model apply manifest",
            )
        except ValueError:
            continue
        expected = _retired_manifest_inventory_without_entry(
            base_raw,
            manifest_path,
        )
        if expected is not None and expected == live_raw:
            candidates.append(manifest_path)
    if len(candidates) != 1:
        raise ValueError(
            "retired manifest inventory change is not the exact removal for one "
            "changed signed-v5 model manifest"
        )
    return candidates[0]


def authorized_changed_paths(
    repo: Path,
    *,
    corpus_root: Path | None = None,
) -> set[PurePosixPath]:
    corpus_release = None
    if corpus_root is not None:
        from axiom_encode.toolchain import load_rulespec_local_corpus_release

        corpus_release = load_rulespec_local_corpus_release(repo, corpus_root)
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
        payload = _load_unambiguous_json(
            _read_bounded_regular(
                repo,
                relative,
                label="changed manifest",
                max_bytes=1024 * 1024,
            ).decode("utf-8"),
            label=f"changed manifest {relative.as_posix()}",
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
    if RETIRED_MANIFEST_INVENTORY in changed and not legacy_manifests:
        _authorized_retired_manifest_inventory_reconciliation(
            repo,
            manifest_payloads=manifest_payloads,
        )
        authorized.add(RETIRED_MANIFEST_INVENTORY)
    authorized_unchanged: set[PurePosixPath] = set()
    authenticated_unchanged_claims: set[tuple[PurePosixPath, PurePosixPath]] = set()
    unchanged_claims: set[tuple[PurePosixPath, PurePosixPath]] = set()
    for relative, payload in manifest_payloads.items():
        tool = payload.get("tool")
        backend = payload.get("backend")
        if tool == MODEL_APPLY_TOOL and (
            not isinstance(backend, str) or backend not in MODEL_APPLY_BACKENDS
        ):
            raise ValueError(
                f"model apply manifest has an unsupported backend: {relative}"
            )
        is_model_apply = tool == MODEL_APPLY_TOOL
        replacement = payload.get("replacement")
        receipt_rewrites: set[PurePosixPath] = set()
        if tool == LEGACY_REPLACEMENT_TOOL:
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
                    LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V4,
                    LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
                    LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                    LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
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
                if "exact_dependents" in receipt_replacement:
                    raise ValueError(
                        f"legacy replacement v1 receipt has v2 fields: {relative}"
                    )
            elif not isinstance(exact_dependents, list):
                raise ValueError(
                    f"legacy replacement exact_dependents are malformed: {relative}"
                )
            retained_successors = receipt_replacement.get("retained_successors")
            metadata_reconciliations = receipt_replacement.get(
                "metadata_reconciliations"
            )
            if receipt_schema in {
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V4,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
            }:
                if not isinstance(retained_successors, list) or not isinstance(
                    metadata_reconciliations, list
                ):
                    raise ValueError(
                        f"legacy replacement v4 reconciliation fields are malformed: "
                        f"{relative}"
                    )
            elif (
                "retained_successors" in receipt_replacement
                or "metadata_reconciliations" in receipt_replacement
            ):
                raise ValueError(
                    f"legacy replacement pre-v4 receipt has v4 fields: {relative}"
                )
            else:
                retained_successors = []
                metadata_reconciliations = []
            repository = receipt.get("repository")
            live_paths = {
                item.get("path") for item in live_files if isinstance(item, dict)
            }
            identity_deleted_files = [
                {"path": item.get("path"), "deleted": True}
                for item in legacy_files
                if isinstance(item, dict) and item.get("path") not in live_paths
            ]
            if receipt_schema in {
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V4,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
            }:
                identity_deleted_files.extend(
                    {"path": item.get("path"), "deleted": True}
                    for successor in retained_successors
                    if isinstance(successor, dict)
                    for item in (
                        successor.get("legacy_files", [])
                        if isinstance(successor.get("legacy_files"), list)
                        else []
                    )
                    if isinstance(item, dict)
                )
            from axiom_encode.legacy_replacement import (
                receipt_identity_payload,
                receipt_identity_sha256,
            )

            identity_payload = receipt_identity_payload(
                base_commit=(
                    str(repository.get("base_commit"))
                    if isinstance(repository, dict)
                    else ""
                ),
                base_tree=(
                    str(repository.get("base_tree"))
                    if isinstance(repository, dict)
                    else ""
                ),
                legacy_manifest_sha256=str(
                    replacement.get("legacy_manifest_sha256") or ""
                ),
                model_manifest_sha256=str(
                    receipt_replacement.get("model_manifest_sha256") or ""
                ),
                live_files=live_files,
                deleted_files=identity_deleted_files,
                rewrites=raw_rewrites,
                scheduled_dependents=scheduled_dependents,
                exact_dependents=(
                    exact_dependents
                    if receipt_schema
                    in {
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V2,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V3,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V4,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
                    }
                    else None
                ),
                destination_predecessor_class=(
                    receipt_replacement.get("destination_predecessor_class")
                    if receipt_schema
                    in {
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V3,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V4,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
                    }
                    else None
                ),
                destination_predecessor_files=(
                    receipt_replacement.get("destination_predecessor_files")
                    if receipt_schema
                    in {
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V3,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V4,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
                    }
                    else None
                ),
                retained_successors=(
                    retained_successors
                    if receipt_schema
                    in {
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V4,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
                    }
                    else None
                ),
                metadata_reconciliations=(
                    metadata_reconciliations
                    if receipt_schema
                    in {
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V4,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                        LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
                    }
                    else None
                ),
            )
            if receipt_relative.stem != receipt_identity_sha256(identity_payload):
                raise ValueError(
                    f"legacy replacement receipt identity differs: {relative}"
                )
            if receipt_schema in {
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V3,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V4,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
            } and (
                not isinstance(
                    receipt_replacement.get("destination_predecessor_class"), str
                )
                or not isinstance(
                    receipt_replacement.get("destination_predecessor_files"), list
                )
            ):
                raise ValueError(
                    "legacy replacement destination predecessor files are malformed: "
                    f"{relative}"
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
                _legacy_destination_predecessor_issues,
                _legacy_metadata_reconciliation_bytes,
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
            if receipt_schema in {
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V3,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V4,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
            }:
                predecessor_issues = _legacy_destination_predecessor_issues(
                    repo,
                    base_commit=str(base_commit or ""),
                    authoritative_replacements=authoritative_replacements,
                    legacy=legacy,
                    replacement=receipt_replacement,
                )
                if predecessor_issues:
                    raise ValueError(
                        f"legacy replacement destination predecessor differs: {relative}: "
                        + "; ".join(predecessor_issues)
                    )
                authorized_unchanged.update(
                    PurePosixPath(item["path"])
                    for item in receipt_replacement["destination_predecessor_files"]
                    if isinstance(item, dict)
                    and isinstance(item.get("path"), str)
                    and PurePosixPath(item["path"]) not in changed
                )
            from axiom_encode.rulespec_path_migration import (
                PathMigrationPlanError,
                PlannedMove,
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
            primary_moves = [
                PlannedMove(source=Path(old), destination=Path(new))
                for old, new in authoritative_replacements.items()
                if old.endswith(".yaml") and not old.endswith(".test.yaml")
            ]
            exact_metadata_manifest_paths: set[str] = set()
            exact_metadata_retired_schema_modules: set[str] = set()
            exact_metadata_reindexed_modules: dict[str, bytes] = {}
            if receipt_schema == LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7:
                assert isinstance(exact_dependents, list)
                for index, dependent in enumerate(exact_dependents):
                    label = f"{relative} exact_dependents[{index}]"
                    if not isinstance(dependent, dict):
                        raise ValueError(f"{label} is malformed")
                    primary = _safe_relative_path(
                        dependent.get("primary"),
                        label=f"{label}.primary",
                    )
                    manifest = dependent.get("legacy_manifest")
                    expected_manifest = MANIFEST_ROOT / primary.with_suffix(".json")
                    if (
                        not isinstance(manifest, dict)
                        or manifest.get("path") != expected_manifest.as_posix()
                    ):
                        raise ValueError(f"{label}.legacy_manifest is malformed")
                    dependent_live_files = dependent.get("live_files")
                    primary_records = (
                        [
                            item
                            for item in dependent_live_files
                            if isinstance(item, dict)
                            and item.get("path") == primary.as_posix()
                        ]
                        if isinstance(dependent_live_files, list)
                        else []
                    )
                    live_primary = _read_bounded_regular(
                        repo,
                        primary,
                        label="legacy exact dependent primary",
                        max_bytes=16 * 1024 * 1024,
                    )
                    if (
                        len(primary_records) != 1
                        or primary_records[0].get("sha256")
                        != hashlib.sha256(live_primary).hexdigest()
                    ):
                        raise ValueError(f"{label}.live_files do not bind primary")
                    exact_metadata_manifest_paths.add(expected_manifest.as_posix())
                    exact_metadata_reindexed_modules[primary.as_posix()] = live_primary
                    if dependent.get("source_verification_migration") is not None:
                        exact_metadata_retired_schema_modules.add(primary.as_posix())
            post_migration_waiver_sha256: str | None = None
            try:
                base_waiver_raw = _git_quiet(
                    repo,
                    "show",
                    "HEAD:known-validation-gaps.yaml",
                )
                rewritten_waiver_raw, _waiver_operations = (
                    _legacy_metadata_reconciliation_bytes(
                        Path("known-validation-gaps.yaml"),
                        base_waiver_raw,
                        moves=primary_moves,
                    )
                )
                post_migration_waiver_sha256 = hashlib.sha256(
                    rewritten_waiver_raw
                ).hexdigest()
            except (subprocess.CalledProcessError, ValueError):
                pass
            retired_schema_count_transition: tuple[int, int] | None = None
            if receipt_schema == LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7:
                try:
                    base_retired_freeze_raw = _git_quiet(
                        repo,
                        "show",
                        "HEAD:.axiom/retired-schema-freeze.json",
                    )
                    rewritten_retired_freeze_raw, _retired_freeze_operations = (
                        _legacy_metadata_reconciliation_bytes(
                            Path(".axiom/retired-schema-freeze.json"),
                            base_retired_freeze_raw,
                            moves=primary_moves,
                            retired_schema_modules=frozenset(
                                exact_metadata_retired_schema_modules
                            ),
                        )
                    )
                    before_retired_count = len(
                        json.loads(base_retired_freeze_raw)["artifacts"]
                    )
                    after_retired_count = len(
                        json.loads(rewritten_retired_freeze_raw)["artifacts"]
                    )
                    if after_retired_count < before_retired_count:
                        retired_schema_count_transition = (
                            before_retired_count,
                            after_retired_count,
                        )
                except (subprocess.CalledProcessError, ValueError):
                    pass
            metadata_paths: set[PurePosixPath] = set()
            for index, reconciliation in enumerate(metadata_reconciliations):
                if not isinstance(reconciliation, dict) or set(reconciliation) != {
                    "path",
                    "before_sha256",
                    "after_sha256",
                    "operations",
                }:
                    raise ValueError(
                        f"legacy metadata reconciliation[{index}] is malformed"
                    )
                metadata_path = _safe_relative_path(
                    reconciliation.get("path"),
                    label=f"{relative} metadata_reconciliations[{index}].path",
                )
                if (
                    metadata_path not in LEGACY_REPLACEMENT_METADATA_PATHS
                    or metadata_path in metadata_paths
                ):
                    raise ValueError(
                        f"legacy metadata reconciliation[{index}] path is unauthorized"
                    )
                base_raw = _git(repo, "show", f"HEAD:{metadata_path.as_posix()}")
                live_raw = _read_bounded_regular(
                    repo,
                    metadata_path,
                    label="legacy metadata reconciliation",
                    max_bytes=16 * 1024 * 1024,
                )
                try:
                    expected_live, expected_operations = (
                        _legacy_metadata_reconciliation_bytes(
                            Path(metadata_path),
                            base_raw,
                            moves=primary_moves,
                            validation_waiver_set_sha256=(post_migration_waiver_sha256),
                            retired_manifest_paths=frozenset(
                                exact_metadata_manifest_paths
                            ),
                            retired_schema_modules=frozenset(
                                exact_metadata_retired_schema_modules
                            ),
                            retired_schema_count_transition=(
                                retired_schema_count_transition
                            ),
                            reindexed_modules=exact_metadata_reindexed_modules,
                        )
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"legacy metadata reconciliation[{index}] is unverifiable"
                    ) from exc
                if (
                    reconciliation.get("before_sha256")
                    != hashlib.sha256(base_raw).hexdigest()
                    or reconciliation.get("after_sha256")
                    != hashlib.sha256(live_raw).hexdigest()
                    or live_raw != expected_live
                    or reconciliation.get("operations") != list(expected_operations)
                ):
                    raise ValueError(
                        f"legacy metadata reconciliation[{index}] state differs"
                    )
                metadata_paths.add(metadata_path)
            expected_metadata_paths: set[PurePosixPath] = set()
            for metadata_path in LEGACY_REPLACEMENT_METADATA_PATHS:
                try:
                    base_raw = _git_quiet(
                        repo,
                        "show",
                        f"HEAD:{metadata_path.as_posix()}",
                    )
                except subprocess.CalledProcessError:
                    continue
                try:
                    expected_live, _expected_operations = (
                        _legacy_metadata_reconciliation_bytes(
                            Path(metadata_path),
                            base_raw,
                            moves=primary_moves,
                            validation_waiver_set_sha256=(post_migration_waiver_sha256),
                            retired_manifest_paths=frozenset(
                                exact_metadata_manifest_paths
                            ),
                            retired_schema_modules=frozenset(
                                exact_metadata_retired_schema_modules
                            ),
                            retired_schema_count_transition=(
                                retired_schema_count_transition
                            ),
                            reindexed_modules=exact_metadata_reindexed_modules,
                        )
                    )
                except ValueError:
                    continue
                if expected_live != base_raw:
                    expected_metadata_paths.add(metadata_path)
            if metadata_paths != expected_metadata_paths:
                raise ValueError(
                    "legacy metadata reconciliation inventory is not exact"
                )
            receipt_rewrites.update(metadata_paths)
            retained_deleted_files: list[dict[str, object]] = []
            for index, successor in enumerate(retained_successors):
                if not isinstance(successor, dict) or set(successor) != {
                    "source",
                    "destination",
                    "legacy_owner_class",
                    "legacy_manifest",
                    "legacy_files",
                    "successor_manifest",
                    "successor_files",
                }:
                    raise ValueError(f"legacy retained successor[{index}] is malformed")
                old_manifest_evidence = successor.get("legacy_manifest")
                successor_manifest_evidence = successor.get("successor_manifest")
                successor_files = successor.get("successor_files")
                old_files = successor.get("legacy_files")
                if (
                    not isinstance(old_manifest_evidence, dict)
                    or not isinstance(successor_manifest_evidence, dict)
                    or not isinstance(successor_files, list)
                    or not isinstance(old_files, list)
                ):
                    raise ValueError(
                        f"legacy retained successor[{index}] evidence is malformed"
                    )
                retained_old_manifest = _safe_relative_path(
                    old_manifest_evidence.get("path"),
                    label=f"{relative} retained_successors[{index}].legacy_manifest",
                )
                retained_manifest = _safe_relative_path(
                    successor_manifest_evidence.get("path"),
                    label=f"{relative} retained_successors[{index}].successor_manifest",
                )
                retained_payload = manifest_payloads.get(retained_manifest)
                if (
                    retained_old_manifest not in deleted_manifests
                    or retained_manifest not in live_manifests
                    or not isinstance(retained_payload, dict)
                    or retained_payload.get("tool") != LEGACY_RETAINED_SUCCESSOR_TOOL
                    or retained_payload.get("applied_files") != successor_files
                    or retained_payload.get("retained_successor_manifest")
                    != successor_manifest_evidence.get("payload")
                ):
                    raise ValueError(
                        f"legacy retained successor[{index}] manifest transition differs"
                    )
                migration = retained_payload.get("legacy_migration")
                if (
                    not isinstance(migration, dict)
                    or migration.get("receipt_path") != receipt_relative.as_posix()
                    or migration.get("receipt_sha256")
                    != hashlib.sha256(receipt_raw).hexdigest()
                    or migration.get("source") != successor.get("source")
                    or migration.get("destination") != successor.get("destination")
                    or migration.get("legacy_manifest_path")
                    != retained_old_manifest.as_posix()
                    or migration.get("legacy_manifest_sha256")
                    != old_manifest_evidence.get("sha256")
                    or migration.get("successor_manifest_sha256")
                    != successor_manifest_evidence.get("sha256")
                ):
                    raise ValueError(
                        f"legacy retained successor[{index}] receipt binding differs"
                    )
                for file_index, item in enumerate(old_files):
                    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                        raise ValueError(
                            f"legacy retained successor[{index}] old file is malformed"
                        )
                    old_path = _safe_relative_path(
                        item.get("path"),
                        label=(
                            f"{relative} retained_successors[{index}]."
                            f"legacy_files[{file_index}]"
                        ),
                    )
                    if (repo / old_path).exists() or (repo / old_path).is_symlink():
                        raise ValueError(
                            f"legacy retained successor source still exists: {old_path}"
                        )
                    if hashlib.sha256(
                        _git(repo, "show", f"HEAD:{old_path.as_posix()}")
                    ).hexdigest() != item.get("sha256"):
                        raise ValueError(
                            f"legacy retained successor base differs: {old_path}"
                        )
                    retained_deleted_files.append(
                        {"path": old_path.as_posix(), "deleted": True}
                    )
                for file_index, item in enumerate(successor_files):
                    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                        raise ValueError(
                            f"legacy retained successor[{index}] live file is malformed"
                        )
                    live_path = _safe_relative_path(
                        item.get("path"),
                        label=(
                            f"{relative} retained_successors[{index}]."
                            f"successor_files[{file_index}]"
                        ),
                    )
                    if hashlib.sha256(
                        _read_bounded_regular(
                            repo,
                            live_path,
                            label="legacy retained successor file",
                            max_bytes=16 * 1024 * 1024,
                        )
                    ).hexdigest() != item.get("sha256"):
                        raise ValueError(
                            f"legacy retained successor live file differs: {live_path}"
                        )
                    if live_path not in changed:
                        authorized_unchanged.add(live_path)
                        authenticated_unchanged_claims.add(
                            (retained_manifest, live_path)
                        )
                authorized.add(retained_old_manifest)
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
                    {
                        "path": item.get("path"),
                        "sha256": item.get("after_sha256"),
                    }
                    for item in metadata_reconciliations
                    if isinstance(item, dict)
                ],
                *[
                    {"path": item.get("path"), "deleted": True}
                    for item in legacy_files
                    if isinstance(item, dict)
                ],
                *retained_deleted_files,
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
            if receipt_schema in {
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V2,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V3,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V4,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V5,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V6,
                LEGACY_REPLACEMENT_RECEIPT_SCHEMA_V7,
            }:
                exact_unchanged_claims = _validate_legacy_exact_dependents(
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
                    corpus_release=corpus_release,
                )
                authenticated_unchanged_claims.update(exact_unchanged_claims)
                authorized_unchanged.update(
                    path for _manifest, path in exact_unchanged_claims
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
            if is_model_apply:
                digest = entry.get("sha256")
                if (
                    set(entry) != {"path", "sha256"}
                    or not isinstance(digest, str)
                    or DIGEST_PATTERN.fullmatch(digest) is None
                ):
                    raise ValueError(
                        f"{relative} applied_files[{index}] must bind an exact sha256"
                    )
                live_raw = _read_bounded_regular(
                    repo,
                    applied_path,
                    label="model-applied file",
                    max_bytes=16 * 1024 * 1024,
                )
                if hashlib.sha256(live_raw).hexdigest() != digest:
                    raise ValueError(
                        f"{relative} applied_files[{index}] differs from its signed "
                        "sha256"
                    )
                if applied_path not in changed:
                    try:
                        base_raw = _git(
                            repo,
                            "show",
                            f"HEAD:{applied_path.as_posix()}",
                        )
                    except subprocess.CalledProcessError as exc:
                        raise ValueError(
                            f"{relative} unchanged applied_files[{index}] is not an "
                            "existing clean-HEAD file"
                        ) from exc
                    if hashlib.sha256(base_raw).hexdigest() != digest:
                        raise ValueError(
                            f"{relative} unchanged applied_files[{index}] differs "
                            "from its signed sha256"
                        )
                    authorized_unchanged.add(applied_path)
                    authenticated_unchanged_claims.add((relative, applied_path))
            elif (
                tool == LEGACY_REPLACEMENT_TOOL and applied_path in authorized_unchanged
            ):
                authenticated_unchanged_claims.add((relative, applied_path))
            if applied_path not in changed:
                unchanged_claims.add((relative, applied_path))

    if deleted_manifests - authorized:
        raise ValueError(
            "deleted manifests are not authenticated by a replacement receipt: "
            + ", ".join(map(str, sorted(deleted_manifests - authorized)))
        )

    authorized_unchanged.difference_update(
        path
        for manifest, path in unchanged_claims
        if (manifest, path) not in authenticated_unchanged_claims
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


def stage_authorized_changes(
    repo: Path,
    *,
    corpus_root: Path | None = None,
) -> None:
    authorized = authorized_changed_paths(repo, corpus_root=corpus_root)
    authorized_bytes: dict[PurePosixPath, bytes | None] = {}
    for path in authorized:
        if not _checkout_path_exists_without_indirection(
            repo,
            path,
            label="authorized publication path",
        ):
            authorized_bytes[path] = None
            continue
        authorized_bytes[path] = _read_bounded_regular(
            repo,
            path,
            label="authorized publication file",
            max_bytes=16 * 1024 * 1024,
        )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
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
    index_entries: dict[PurePosixPath, tuple[str, str, str]] = {}
    for raw_entry in _git(repo, "ls-files", "--stage", "-z").split(b"\0"):
        if not raw_entry:
            continue
        raw_metadata, separator, raw_path = raw_entry.partition(b"\t")
        try:
            mode, object_id, stage = raw_metadata.decode("ascii").split(" ")
            path = PurePosixPath(raw_path.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("Git index contains a malformed staged entry") from exc
        if path not in authorized:
            continue
        if not separator or path in index_entries:
            raise ValueError("authorized path has ambiguous Git index entries")
        index_entries[path] = (mode, object_id, stage)
    for path, expected_bytes in authorized_bytes.items():
        entry = index_entries.get(path)
        if expected_bytes is None:
            if entry is not None:
                raise ValueError("deleted authorized path remains in the Git index")
            continue
        if (
            entry is None
            or entry[0] != "100644"
            or COMMIT_PATTERN.fullmatch(entry[1]) is None
            or entry[2] != "0"
        ):
            raise ValueError("authorized file has an invalid Git index entry")
        staged_bytes = _git(repo, "cat-file", "blob", entry[1])
        if staged_bytes != expected_bytes:
            raise ValueError("staged file bytes differ from signed authorization")
        live_bytes = _read_bounded_regular(
            repo,
            path,
            label="authorized publication file after staging",
            max_bytes=16 * 1024 * 1024,
        )
        if live_bytes != expected_bytes:
            raise ValueError("authorized file changed while it was being staged")


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
    stage_parser.add_argument("--corpus-path", dest="corpus_root", type=Path)
    cascade_parser = subparsers.add_parser("validate-dependent-cascade")
    cascade_parser.add_argument("repo", type=Path)
    cascade_parser.add_argument("target_citation")
    cascade_parser.add_argument("--target-rulespec-path")
    cascade_parser.add_argument("dependent_citations", nargs="+")
    citation_path_parser = subparsers.add_parser("citation-rulespec-path")
    citation_path_parser.add_argument("citation")
    shrink_parser = subparsers.add_parser("authorize-legacy-index-manifest-shrink")
    shrink_parser.add_argument("repo", type=Path)
    shrink_parser.add_argument("target_rulespec_path")
    retired_inventory_parser = subparsers.add_parser(
        "reconcile-retired-manifest-inventory"
    )
    retired_inventory_parser.add_argument("repo", type=Path)
    retired_inventory_parser.add_argument("target_rulespec_path")
    source_bundle_parser = subparsers.add_parser(
        "parse-source-bundle",
        help="validate a bounded source bundle and emit one normalized JSON array",
    )
    source_bundle_parser.add_argument(
        "source_bundle_json",
        help="JSON array containing at most 16 canonical corpus citation strings",
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
    source_add_parser = subparsers.add_parser(
        "validate-source-add-targets",
        help=(
            "reject source-add destinations already present in the pinned "
            "RuleSpec checkout"
        ),
    )
    source_add_parser.add_argument("repo", type=Path)
    source_add_parser.add_argument("source_bundle_json")
    source_add_parser.add_argument("--primary-citation", required=True)
    source_add_parser.add_argument("--primary-rulespec-path", default="")
    atomic_source_parser = subparsers.add_parser(
        "split-atomic-source-input",
        help=(
            "split the bounded source input into mutually exclusive source "
            "composition and canonical refresh payloads"
        ),
    )
    atomic_source_parser.add_argument(
        "atomic_source_json",
        help=(
            "legacy source citation array or exact "
            '{"canonical_refresh_bundle":[...]} object'
        ),
    )
    canonical_refresh_parser = subparsers.add_parser(
        "parse-canonical-refresh-bundle",
        help=(
            "validate existing independent canonical modules for one atomic fresh "
            "signed refresh and emit normalized JSON"
        ),
    )
    canonical_refresh_parser.add_argument("repo", type=Path)
    canonical_refresh_parser.add_argument(
        "refresh_bundle_json",
        help="JSON array containing at most 15 additional canonical citations",
    )
    canonical_refresh_parser.add_argument("--primary-citation", required=True)
    canonical_refresh_parser.add_argument("--primary-rulespec-path", required=True)
    canonical_refresh_parser.add_argument(
        "--primary-required-test-cases-json",
        default="[]",
        help="bounded JSON array of exact companion cases required for the primary",
    )
    canonical_refresh_target_parser = subparsers.add_parser(
        "verify-canonical-refresh-target",
        help="verify one normalized canonical refresh target remains unchanged",
    )
    canonical_refresh_target_parser.add_argument("repo", type=Path)
    canonical_refresh_target_parser.add_argument("target_json")
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
        elif args.command == "authorize-legacy-index-manifest-shrink":
            print(
                "true"
                if authorize_legacy_index_manifest_shrink(
                    args.repo,
                    args.target_rulespec_path,
                )
                else "false"
            )
        elif args.command == "reconcile-retired-manifest-inventory":
            reconciled = reconcile_retired_manifest_inventory(
                args.repo,
                args.target_rulespec_path,
            )
            if reconciled is None:
                print("retired manifest inventory unchanged")
            else:
                print(f"retired manifest inventory removed {reconciled.as_posix()}")
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
        elif args.command == "split-atomic-source-input":
            print(
                json.dumps(
                    split_atomic_source_input(args.atomic_source_json),
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        elif args.command == "validate-source-add-targets":
            print(
                json.dumps(
                    validate_source_add_targets(
                        args.repo,
                        args.source_bundle_json,
                        primary_citation=args.primary_citation,
                        primary_rulespec_path=args.primary_rulespec_path,
                    ),
                    separators=(",", ":"),
                )
            )
        elif args.command == "parse-canonical-refresh-bundle":
            print(
                json.dumps(
                    parse_canonical_refresh_bundle(
                        args.repo,
                        args.refresh_bundle_json,
                        primary_citation=args.primary_citation,
                        primary_rulespec_path=args.primary_rulespec_path,
                        primary_required_test_cases_json=(
                            args.primary_required_test_cases_json
                        ),
                    ),
                    separators=(",", ":"),
                )
            )
        elif args.command == "verify-canonical-refresh-target":
            print(
                json.dumps(
                    verify_canonical_refresh_target(args.repo, args.target_json),
                    separators=(",", ":"),
                    sort_keys=True,
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
                        )
                    ],
                    separators=(",", ":"),
                )
            )
        else:
            stage_authorized_changes(args.repo, corpus_root=args.corpus_root)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
