"""Pure contracts for authenticated legacy-to-v5 fresh re-encoding.

This module deliberately cannot sign, invoke a model, read Git, or mutate a
checkout.  It defines the exact historical evidence class and deterministic
receipt identity used by the CLI's broker-authenticated transaction.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Final, Mapping, NamedTuple

from .corpus_resolver import normalize_corpus_identifier

TOOL: Final = "axiom-encode encode --apply --replace-legacy-rulespec-path"
RECEIPT_SCHEMA: Final = "axiom-encode/legacy-fresh-reencode-receipt/v1"
RECEIPT_DIR: Final = ".axiom/legacy-replacements"
LEGACY_MANIFEST_SCHEMA: Final = "axiom-encode/applied-rulespec/v1"
LEGACY_OWNER_CLASS: Final = "v1-manual-hmac-untrusted"

_SHA256 = re.compile(r"[0-9a-f]{64}")


class LegacyReplacementFile(NamedTuple):
    path: Path
    sha256: str
    raw: bytes


class LegacyReplacementRewrite(NamedTuple):
    path: Path
    before_sha256: str
    after_sha256: str
    replacements: tuple[dict[str, object], ...]
    raw: bytes


class LegacyReplacementPendingFile(NamedTuple):
    path: Path
    before_sha256: str
    replacements: tuple[dict[str, object], ...]


class LegacyReplacementScheduledDependent(NamedTuple):
    primary: Path
    files: tuple[LegacyReplacementPendingFile, ...]


class LegacyReplacementContract(NamedTuple):
    base_commit: str
    base_tree: str
    source: Path
    destination: Path
    legacy_manifest: LegacyReplacementFile
    deleted_files: tuple[LegacyReplacementFile, ...]
    rewrites: tuple[LegacyReplacementRewrite, ...]
    scheduled_dependents: tuple[LegacyReplacementScheduledDependent, ...]


def legacy_source_verification_citation_paths(
    verification: Mapping[str, object],
) -> tuple[str, ...]:
    """Admit exactly one exclusive singular or historical plural citation."""

    admitted = {"corpus_citation_path", "corpus_citation_paths"}
    present = set(verification) & admitted
    if present == {"corpus_citation_path"}:
        raw_paths: object = [verification["corpus_citation_path"]]
    elif present == {"corpus_citation_paths"}:
        raw_paths = verification["corpus_citation_paths"]
    else:
        return ()
    if (
        not isinstance(raw_paths, list)
        or len(raw_paths) != 1
        or not isinstance(raw_paths[0], str)
        or not raw_paths[0].strip()
    ):
        return ()
    try:
        return (normalize_corpus_identifier(raw_paths[0]),)
    except (TypeError, ValueError):
        return ()


def legacy_manual_manifest_issues(
    payload: object,
    *,
    expected_files: Mapping[str, str],
) -> list[str]:
    """Require the one known manual v1 class without trusting its signature."""

    if not isinstance(payload, dict):
        return ["legacy ownership manifest is not a JSON object"]
    issues: list[str] = []
    if payload.get("schema_version") != LEGACY_MANIFEST_SCHEMA:
        issues.append("legacy ownership manifest is not schema v1")
    if payload.get("tool") != "axiom-encode sign-applied-files":
        issues.append("legacy ownership manifest tool is not sign-applied-files")
    if payload.get("backend") != "manual":
        issues.append("legacy ownership manifest backend is not manual")
    if payload.get("runner") != "manual-attestation":
        issues.append("legacy ownership manifest runner is not manual-attestation")
    if (
        not isinstance(payload.get("manual_exception"), str)
        or not str(payload["manual_exception"]).strip()
    ):
        issues.append("legacy ownership manifest has no manual exception")
    if any(
        field in payload
        for field in (
            "deterministic_execution",
            "validation_execution",
            "migrated_manifest",
            "retired_manifest",
            "replacement_manifest",
        )
    ):
        issues.append("legacy ownership manifest claims unsupported provenance")
    signature = payload.get("signature")
    if (
        not isinstance(signature, dict)
        or signature.get("algorithm") != "hmac-sha256"
        or not isinstance(signature.get("key_id"), str)
        or not isinstance(signature.get("value"), str)
    ):
        issues.append("legacy ownership manifest has unknown signature provenance")
    entries = payload.get("applied_files")
    actual: dict[str, str] = {}
    if not isinstance(entries, list):
        issues.append("legacy ownership manifest applied_files is malformed")
    else:
        for index, entry in enumerate(entries):
            if (
                not isinstance(entry, dict)
                or set(entry) != {"path", "sha256"}
                or not isinstance(entry.get("path"), str)
                or not isinstance(entry.get("sha256"), str)
                or _SHA256.fullmatch(str(entry["sha256"])) is None
                or entry["path"] in actual
            ):
                issues.append(
                    f"legacy ownership manifest applied_files[{index}] is malformed"
                )
                continue
            actual[str(entry["path"])] = str(entry["sha256"])
    if actual != dict(expected_files):
        issues.append(
            "legacy ownership manifest does not bind the exact old primary/test bytes"
        )
    return issues


def receipt_identity_payload(
    *,
    base_commit: str,
    base_tree: str,
    legacy_manifest_sha256: str,
    model_manifest_sha256: str,
    live_files: list[dict[str, object]],
    deleted_files: list[dict[str, object]],
    rewrites: list[dict[str, object]],
    scheduled_dependents: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "base_commit": base_commit,
        "base_tree": base_tree,
        "legacy_manifest_sha256": legacy_manifest_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "live_files": live_files,
        "deleted_files": deleted_files,
        "rewrites": rewrites,
        "scheduled_dependents": scheduled_dependents,
    }


def receipt_identity_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
