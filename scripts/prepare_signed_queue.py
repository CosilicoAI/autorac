#!/usr/bin/env python3
"""Build, validate, and select bounded protected encoding queue tranches."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

SCHEMA = "axiom-encode/signed-encoding-queue/v1"
ACTIVATION_CHANGED_FILES = (
    "data/encoding-queues/us-snap-or-ut-2026-07.json",
    "pyproject.toml",
    "src/axiom_encode/__init__.py",
    "uv.lock",
)
QUEUE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
ITEM_ID_PATTERN = re.compile(r"^[a-z]{2}-[0-9]{4,5}$")
RELEASE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
BLOCK_SUFFIX_PATTERN = re.compile(r"/block-[0-9]+$")
SNAP_PATTERN = re.compile(r"SNAP", re.IGNORECASE)
ALLOWED_STATUSES = frozenset(
    {
        "blocked",
        "completed",
        "no-executable-rule",
        "pending",
        "retryable",
    }
)
SELECTABLE_STATUSES = frozenset({"pending", "retryable"})
TERMINAL_STATUSES = frozenset({"blocked", "completed", "no-executable-rule"})
MAX_BATCH_SIZE = 4
EXPECTED_COUNTS = {"total": 831, "us-or": 530, "us-ut": 301}
ALL_STATE_EXPECTED_COUNTS = {
    "total": 17784,
    "us-ak": 239,
    "us-al": 17,
    "us-ar": 8,
    "us-az": 7,
    "us-ca": 15,
    "us-co": 302,
    "us-ct": 279,
    "us-dc": 349,
    "us-de": 264,
    "us-fl": 48,
    "us-ga": 100,
    "us-hi": 20,
    "us-ia": 12,
    "us-id": 70,
    "us-il": 4750,
    "us-in": 483,
    "us-ks": 255,
    "us-ky": 2,
    "us-la": 346,
    "us-ma": 9,
    "us-md": 51,
    "us-me": 2,
    "us-mi": 196,
    "us-mn": 1353,
    "us-mo": 515,
    "us-ms": 384,
    "us-mt": 83,
    "us-nc": 79,
    "us-nd": 67,
    "us-ne": 5,
    "us-nh": 1,
    "us-nj": 265,
    "us-nm": 28,
    "us-nv": 51,
    "us-ny": 17,
    "us-oh": 82,
    "us-ok": 87,
    "us-or": 1322,
    "us-pa": 297,
    "us-ri": 272,
    "us-sc": 459,
    "us-sd": 339,
    "us-tn": 27,
    "us-tx": 11,
    "us-ut": 655,
    "us-va": 677,
    "us-vt": 32,
    "us-wa": 200,
    "us-wi": 372,
    "us-wv": 2276,
    "us-wy": 4,
}
RULESPEC_PR_PATTERN = re.compile(
    r"^https://github\.com/TheAxiomFoundation/rulespec-us/pull/[0-9]+$"
)
FINALIZER_RUN_PATTERN = re.compile(
    r"^https://github\.com/TheAxiomFoundation/axiom-encode/actions/runs/[0-9]+$"
)
TARGET_RUN_MARKER_PATTERN = re.compile(
    r"(?:^|\n)Axiom Encode run: "
    r"(https://github\.com/TheAxiomFoundation/axiom-encode/actions/runs/[0-9]+)"
    r"(?:\n|$)"
)
PRIORITY_CITATIONS = (
    "us-ut/manual/dws/eligibility-manual/"
    "tables-appendicies-and-charts-tables-appendicies-and-charts-table-2-"
    "snap-monthly-income-limits-and-m/block-2",
    "us-or/manual/odhs/open/page-272",
    "us-or/manual/odhs/open/page-273",
    "us-or/manual/odhs/open/page-274",
)
ALL_STATE_QUEUE_ID = "us-snap-all-states-2026-07"
ALL_STATE_QUEUE_ISSUE = 1287
ALL_STATE_INVENTORY_PATH = "manifests/state-snap-manual-agent-queue.yaml"
ALL_STATE_INVENTORY_SHA256 = (
    "23b32cef7957d56b171abe51ed63de6cf22c6c29475ba7a2190b3da4692422b6"
)
ALL_STATE_SCOPE_ALIASES = {
    (
        "us-ma",
        "regulation",
        "2026-07-17-ma-dta-snap-regulations",
    ): (
        "us-ma",
        "regulation",
        "2026-07-24-ma-dta-regulations-snap-current-union",
    ),
}


def _queue_profile(queue_id: str) -> dict[str, Any]:
    if queue_id == "us-snap-or-ut-2026-07":
        return {
            "activation_path": "data/encoding-queues/us-snap-or-ut-2026-07.json",
            "expected_counts": EXPECTED_COUNTS,
            "issue": 1257,
            "inventory": None,
            "item_width": 4,
            "priority_citations": PRIORITY_CITATIONS,
            "supersedes": None,
        }
    if queue_id == ALL_STATE_QUEUE_ID:
        return {
            "activation_path": f"data/encoding-queues/{ALL_STATE_QUEUE_ID}.json",
            "expected_counts": ALL_STATE_EXPECTED_COUNTS,
            "issue": ALL_STATE_QUEUE_ISSUE,
            "inventory": {
                "jurisdictions": 51,
                "path": ALL_STATE_INVENTORY_PATH,
                "sha256": ALL_STATE_INVENTORY_SHA256,
            },
            "item_width": 5,
            "priority_citations": (),
            "supersedes": ["us-snap-or-ut-2026-07"],
        }
    raise ValueError(f"unsupported protected SNAP queue: {queue_id}")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("queue manifest must be a JSON object")
    return payload


def _load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"malformed corpus JSONL at {path}:{line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"corpus JSONL row is not an object at {path}:{line_number}"
                    )
                records.append(record)
    return records


def _has_snap_marker(record: dict[str, Any]) -> bool:
    text = " ".join(
        str(record.get(field) or "") for field in ("body", "heading", "citation_label")
    )
    return SNAP_PATTERN.search(text) is not None


def _record_label(record: dict[str, Any]) -> str:
    for field in ("heading", "citation_label", "citation_path"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("corpus record has no usable label")


def _logical_source_units(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one stable encoder target per logical source unit."""

    by_kind: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        kind = record.get("kind")
        citation = record.get("citation_path")
        if not isinstance(kind, str) or not isinstance(citation, str):
            raise ValueError("SNAP scope record lacks kind or citation_path")
        by_kind.setdefault(kind, []).append(record)

    documents = by_kind.get("document", [])
    if len(documents) > 1:
        selected = documents
    else:
        selected = []
        for kind in ("page", "section", "rule", "sheet"):
            if by_kind.get(kind):
                selected = by_kind[kind]
                break
        if not selected and documents:
            selected = documents
        if not selected:
            citations = {
                record["citation_path"]
                for record in records
                if isinstance(record.get("citation_path"), str)
            }
            selected = [
                record
                for record in records
                if not any(
                    other.startswith(f"{record['citation_path']}/")
                    for other in citations
                    if other != record["citation_path"]
                )
            ]

    if any(
        any(character.isspace() for character in record["citation_path"])
        for record in selected
    ):
        if not documents:
            raise ValueError(
                "logical source-unit citations contain whitespace and have no "
                "document fallback"
            )
        selected = documents

    return sorted(
        selected,
        key=lambda record: (
            record.get("ordinal")
            if isinstance(record.get("ordinal"), int)
            else sys.maxsize,
            record["citation_path"],
        ),
    )


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase commit SHA")
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _json_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _git(corpus_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(corpus_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_corpus_provenance(
    corpus_root: Path,
    *,
    corpus_ref: str,
    release_name: str,
    source_paths: list[Path],
) -> str:
    """Verify queue inputs come from the exact clean released corpus commit."""

    if RELEASE_NAME_PATTERN.fullmatch(release_name) is None:
        raise ValueError("release_name is malformed")
    try:
        head = _git(corpus_root, "rev-parse", "HEAD")
        checkout_status = _git(corpus_root, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("corpus_root must be a readable Git checkout") from exc
    if head != corpus_ref:
        raise ValueError(f"corpus checkout HEAD {head} does not match {corpus_ref}")
    if checkout_status:
        raise ValueError("corpus checkout is not clean")

    manifest_path = corpus_root / "manifests/releases" / f"{release_name}.json"
    release = _load_json(manifest_path)
    if release.get("name") != release_name:
        raise ValueError("corpus release manifest name does not match release_name")
    scopes = release.get("scopes")
    if not isinstance(scopes, list):
        raise ValueError("corpus release manifest scopes are missing")
    provisions = corpus_root / "data/corpus/provisions"
    required_scopes = set()
    for source_path in source_paths:
        try:
            relative = source_path.relative_to(provisions)
        except ValueError as exc:
            raise ValueError("queue source is outside canonical provisions") from exc
        if len(relative.parts) != 3 or relative.suffix != ".jsonl":
            raise ValueError(f"queue source path is malformed: {relative}")
        required_scopes.add((relative.parts[0], relative.parts[1], relative.stem))
    available_scopes = {
        (
            scope.get("jurisdiction"),
            scope.get("document_class"),
            scope.get("version"),
        )
        for scope in scopes
        if isinstance(scope, dict)
    }
    missing = required_scopes - available_scopes
    if missing:
        raise ValueError(
            "corpus release manifest is missing SNAP manual scopes: "
            + ", ".join("/".join(scope) for scope in sorted(missing))
        )
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _verify_signed_release_binding(
    corpus_root: Path,
    rulespec_root: Path,
    *,
    corpus_ref: str,
    rulespec_ref: str,
    release_name: str,
    release_content_sha256: str,
    release_object_path: Path,
    release_public_key_path: Path,
    source_paths: list[Path],
) -> None:
    """Authenticate the release and bind it to RuleSpec and local source bytes."""

    try:
        rulespec_head = _git(rulespec_root, "rev-parse", "HEAD")
        rulespec_status = _git(rulespec_root, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("rulespec_root must be a readable Git checkout") from exc
    if rulespec_head != rulespec_ref:
        raise ValueError(
            f"RuleSpec checkout HEAD {rulespec_head} does not match {rulespec_ref}"
        )
    if rulespec_status:
        raise ValueError("RuleSpec checkout is not clean")

    try:
        from scripts.materialize_corpus_release import load_release_pin
    except ModuleNotFoundError:
        from materialize_corpus_release import load_release_pin

    pinned_name, pinned_sha = load_release_pin(rulespec_root / ".axiom/toolchain.toml")
    if (pinned_name, pinned_sha) != (release_name, release_content_sha256):
        raise ValueError("RuleSpec toolchain release pin does not match the queue")

    try:
        release_object = _load_json(release_object_path)
        public_keys = tuple(
            line.strip()
            for line in release_public_key_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except OSError as exc:
        raise ValueError("signed release verification inputs are unavailable") from exc
    if not public_keys:
        raise ValueError("release public key file is empty")

    from axiom_encode.corpus_release import (
        CorpusReleaseObjectError,
        verify_release_object,
    )

    try:
        verified = verify_release_object(release_object, public_key=public_keys)
    except CorpusReleaseObjectError as exc:
        raise ValueError(f"signed release object is invalid: {exc}") from exc
    if (
        verified.name != release_name
        or verified.content_sha256 != release_content_sha256
    ):
        raise ValueError("signed release identity does not match the queue")
    content = release_object.get("content")
    git = content.get("git") if isinstance(content, dict) else None
    if not isinstance(git, dict) or git.get("commit") != corpus_ref:
        raise ValueError("signed release Git provenance does not match corpus_ref")

    artifacts = {artifact.path: artifact for artifact in verified.artifacts}
    for source_path in source_paths:
        relative = source_path.relative_to(corpus_root).as_posix()
        artifact = artifacts.get(relative)
        if artifact is None:
            raise ValueError(f"signed release omits queue source artifact: {relative}")
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != artifact.sha256:
            raise ValueError(
                f"queue source digest differs from signed release: {relative}"
            )


def build_snap_queue(
    corpus_root: Path,
    rulespec_root: Path,
    *,
    corpus_ref: str,
    rulespec_ref: str,
    rules_engine_ref: str,
    release_name: str,
    release_content_sha256: str,
    release_object_path: Path,
    release_public_key_path: Path,
    state: str,
    pause_reason: str | None,
) -> dict[str, Any]:
    """Build the reviewed Oregon and Utah SNAP source-unit inventory."""

    _require_sha(corpus_ref, "corpus_ref")
    _require_sha(rulespec_ref, "rulespec_ref")
    _require_sha(rules_engine_ref, "rules_engine_ref")
    _require_digest(release_content_sha256, "release_content_sha256")
    provisions = corpus_root / "data/corpus/provisions"
    or_paths = sorted((provisions / "us-or/manual").glob("*.jsonl"))
    ut_paths = sorted((provisions / "us-ut/manual").glob("*.jsonl"))
    source_paths = or_paths + ut_paths
    release_manifest_sha256 = _verify_corpus_provenance(
        corpus_root,
        corpus_ref=corpus_ref,
        release_name=release_name,
        source_paths=source_paths,
    )
    _verify_signed_release_binding(
        corpus_root,
        rulespec_root,
        corpus_ref=corpus_ref,
        rulespec_ref=rulespec_ref,
        release_name=release_name,
        release_content_sha256=release_content_sha256,
        release_object_path=release_object_path,
        release_public_key_path=release_public_key_path,
        source_paths=source_paths,
    )
    or_records = _load_jsonl(or_paths)
    ut_records = _load_jsonl(ut_paths)
    if not or_records or not ut_records:
        raise ValueError("Oregon and Utah manual corpus records are required")

    candidates: dict[str, tuple[str, str]] = {}
    for record in or_records:
        citation = record.get("citation_path")
        if not isinstance(citation, str):
            raise ValueError("Oregon corpus record has no citation_path")
        if record.get("kind") == "document" or _has_snap_marker(record):
            candidates[citation] = (
                "document" if record.get("kind") == "document" else "page",
                _record_label(record),
            )

    for record in ut_records:
        citation = record.get("citation_path")
        if not isinstance(citation, str):
            raise ValueError("Utah corpus record has no citation_path")
        if not _has_snap_marker(record):
            continue
        topic = BLOCK_SUFFIX_PATTERN.sub("", citation)
        candidates.setdefault(topic, ("topic", _record_label(record)))

    exact_block = PRIORITY_CITATIONS[0]
    exact_record = next(
        (record for record in ut_records if record.get("citation_path") == exact_block),
        None,
    )
    if exact_record is None:
        raise ValueError(f"priority corpus block is unavailable: {exact_block}")
    candidates[exact_block] = ("block", _record_label(exact_record))

    priority = list(PRIORITY_CITATIONS)
    missing_priority = [citation for citation in priority if citation not in candidates]
    if missing_priority:
        raise ValueError(
            "priority citations are unavailable: " + ", ".join(missing_priority)
        )
    remaining = sorted(set(candidates) - set(priority))
    ordered_citations = priority + remaining

    counters = {"us-or": 0, "us-ut": 0}
    items: list[dict[str, Any]] = []
    for sequence, citation in enumerate(ordered_citations, start=1):
        jurisdiction = citation.split("/", 1)[0]
        if jurisdiction not in counters:
            raise ValueError(f"unexpected queue jurisdiction: {jurisdiction}")
        counters[jurisdiction] += 1
        prefix = "or" if jurisdiction == "us-or" else "ut"
        source_kind, label = candidates[citation]
        items.append(
            {
                "attempt": 1,
                "citation": citation,
                "id": f"{prefix}-{counters[jurisdiction]:04d}",
                "jurisdiction": jurisdiction,
                "label": label,
                "sequence": sequence,
                "source_kind": source_kind,
                "status": "pending",
            }
        )

    counts = {"total": len(items), **counters}
    if counts != EXPECTED_COUNTS:
        raise ValueError(
            f"SNAP queue inventory drifted: expected {EXPECTED_COUNTS}, got {counts}"
        )

    payload = {
        "schema": SCHEMA,
        "queue_id": "us-snap-or-ut-2026-07",
        "state": state,
        "description": (
            "Conservative Oregon and Utah SNAP manual source-unit inventory. "
            "Each item requires an explicit encoder disposition."
        ),
        "issue": 1257,
        "release": {
            "content_sha256": release_content_sha256,
            "manifest_sha256": release_manifest_sha256,
            "name": release_name,
        },
        "dispatch": {
            "corpus_ref": corpus_ref,
            "country": "us",
            "max_batch_size": MAX_BATCH_SIZE,
            "open_pr": True,
            "pr_base_branch": "hard-cut/canonical-layout-us",
            "rules_engine_ref": rules_engine_ref,
            "rulespec_ref": rulespec_ref,
        },
        "expected_counts": EXPECTED_COUNTS,
        "items": items,
    }
    if pause_reason is not None:
        payload["pause_reason"] = pause_reason
    validate_queue(payload)
    return payload


def build_all_state_snap_queue(
    corpus_root: Path,
    rulespec_root: Path,
    *,
    corpus_ref: str,
    rulespec_ref: str,
    rules_engine_ref: str,
    release_name: str,
    release_content_sha256: str,
    release_object_path: Path,
    release_public_key_path: Path,
    state: str,
    pause_reason: str | None,
) -> dict[str, Any]:
    """Build the signed all-state SNAP logical source-unit inventory."""

    _require_sha(corpus_ref, "corpus_ref")
    _require_sha(rulespec_ref, "rulespec_ref")
    _require_sha(rules_engine_ref, "rules_engine_ref")
    _require_digest(release_content_sha256, "release_content_sha256")

    inventory_path = corpus_root / ALL_STATE_INVENTORY_PATH
    try:
        inventory = yaml.safe_load(inventory_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("state SNAP source inventory is unavailable") from exc
    if not isinstance(inventory, dict) or inventory.get("program") != "SNAP":
        raise ValueError("state SNAP source inventory has the wrong program")
    states = inventory.get("states")
    if not isinstance(states, list) or len(states) != 51:
        raise ValueError("state SNAP source inventory must contain 51 jurisdictions")

    release_path = (
        corpus_root / "manifests/releases" / f"{release_name}.json"
    )
    release = _load_json(release_path)
    release_scopes = release.get("scopes")
    if not isinstance(release_scopes, list):
        raise ValueError("corpus release manifest scopes are missing")
    available_scopes = {
        (
            scope.get("jurisdiction"),
            scope.get("document_class"),
            scope.get("version"),
        )
        for scope in release_scopes
        if isinstance(scope, dict)
    }

    provisions = corpus_root / "data/corpus/provisions"
    source_paths: list[Path] = []
    state_sources: list[tuple[str, Path]] = []
    seen_jurisdictions: set[str] = set()
    for entry in states:
        if not isinstance(entry, dict):
            raise ValueError("state SNAP source inventory entry is malformed")
        jurisdiction = entry.get("jurisdiction")
        if (
            not isinstance(jurisdiction, str)
            or re.fullmatch(r"us-[a-z]{2}", jurisdiction) is None
            or jurisdiction in seen_jurisdictions
        ):
            raise ValueError("state SNAP source inventory jurisdiction is invalid")
        seen_jurisdictions.add(jurisdiction)
        if entry.get("queue_status") != "published_current":
            raise ValueError(
                f"state SNAP source is not published_current: {jurisdiction}"
            )
        for scope_field in ("target_scope", "supporting_scope"):
            scope = entry.get(scope_field)
            if scope is None:
                continue
            if not isinstance(scope, dict):
                raise ValueError(
                    f"{jurisdiction} {scope_field} is malformed"
                )
            scope_key = (
                scope.get("jurisdiction"),
                scope.get("document_class"),
                scope.get("version"),
            )
            resolved_scope = ALL_STATE_SCOPE_ALIASES.get(scope_key, scope_key)
            if resolved_scope not in available_scopes:
                raise ValueError(
                    "state SNAP scope is absent from signed release: "
                    + "/".join(str(value) for value in scope_key)
                )
            path = (
                provisions
                / str(resolved_scope[0])
                / str(resolved_scope[1])
                / f"{resolved_scope[2]}.jsonl"
            )
            source_paths.append(path)
            state_sources.append((jurisdiction, path))

    if len(seen_jurisdictions) != 51:
        raise ValueError("state SNAP source inventory has duplicate jurisdictions")

    release_manifest_sha256 = _verify_corpus_provenance(
        corpus_root,
        corpus_ref=corpus_ref,
        release_name=release_name,
        source_paths=source_paths,
    )
    _verify_signed_release_binding(
        corpus_root,
        rulespec_root,
        corpus_ref=corpus_ref,
        rulespec_ref=rulespec_ref,
        release_name=release_name,
        release_content_sha256=release_content_sha256,
        release_object_path=release_object_path,
        release_public_key_path=release_public_key_path,
        source_paths=source_paths,
    )

    candidates: list[tuple[str, dict[str, Any]]] = []
    seen_citations: set[str] = set()
    for jurisdiction, source_path in state_sources:
        records = _load_jsonl([source_path])
        for record in _logical_source_units(records):
            citation = record["citation_path"]
            if citation in seen_citations:
                raise ValueError(f"duplicate all-state SNAP citation: {citation}")
            seen_citations.add(citation)
            candidates.append((jurisdiction, record))

    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]["citation_path"]))
    counters = {jurisdiction: 0 for jurisdiction in sorted(seen_jurisdictions)}
    items: list[dict[str, Any]] = []
    for sequence, (jurisdiction, record) in enumerate(candidates, start=1):
        counters[jurisdiction] += 1
        items.append(
            {
                "attempt": 1,
                "citation": record["citation_path"],
                "id": f"{jurisdiction[3:]}-{counters[jurisdiction]:05d}",
                "jurisdiction": jurisdiction,
                "label": _record_label(record),
                "sequence": sequence,
                "source_kind": record["kind"],
                "status": "pending",
            }
        )

    counts = {"total": len(items), **counters}
    if counts != ALL_STATE_EXPECTED_COUNTS:
        raise ValueError(
            "all-state SNAP inventory drifted: "
            f"expected {ALL_STATE_EXPECTED_COUNTS}, got {counts}"
        )

    payload = {
        "schema": SCHEMA,
        "queue_id": ALL_STATE_QUEUE_ID,
        "state": state,
        "description": (
            "All-state SNAP logical source-unit inventory for the 50 states and "
            "District of Columbia. Every item requires an encoder disposition."
        ),
        "issue": ALL_STATE_QUEUE_ISSUE,
        "supersedes": ["us-snap-or-ut-2026-07"],
        "inventory": {
            "jurisdictions": 51,
            "path": ALL_STATE_INVENTORY_PATH,
            "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        },
        "release": {
            "content_sha256": release_content_sha256,
            "manifest_sha256": release_manifest_sha256,
            "name": release_name,
        },
        "dispatch": {
            "corpus_ref": corpus_ref,
            "country": "us",
            "max_batch_size": MAX_BATCH_SIZE,
            "open_pr": True,
            "pr_base_branch": "hard-cut/canonical-layout-us",
            "rules_engine_ref": rules_engine_ref,
            "rulespec_ref": rulespec_ref,
        },
        "expected_counts": ALL_STATE_EXPECTED_COUNTS,
        "items": items,
    }
    if pause_reason is not None:
        payload["pause_reason"] = pause_reason
    validate_queue(payload)
    return payload


def validate_queue(payload: dict[str, Any]) -> None:
    """Validate the complete queue and immutable dispatch trust boundary."""

    if payload.get("schema") != SCHEMA:
        raise ValueError(f"queue schema must be {SCHEMA}")
    queue_id = payload.get("queue_id")
    if not isinstance(queue_id, str) or QUEUE_ID_PATTERN.fullmatch(queue_id) is None:
        raise ValueError("queue_id is malformed")
    profile = _queue_profile(queue_id)
    issue = payload.get("issue")
    if not isinstance(issue, int) or isinstance(issue, bool) or issue <= 0:
        raise ValueError("queue issue must be a positive integer")
    if issue != profile["issue"]:
        raise ValueError(f"queue issue must be {profile['issue']}")
    if payload.get("supersedes") != profile["supersedes"]:
        if profile["supersedes"] is None and "supersedes" not in payload:
            pass
        else:
            raise ValueError("queue supersession binding is invalid")
    if payload.get("inventory") != profile["inventory"]:
        if profile["inventory"] is None and "inventory" not in payload:
            pass
        else:
            raise ValueError("queue inventory binding is invalid")
    state = payload.get("state")
    if state not in {"active", "paused"}:
        raise ValueError("queue state must be active or paused")
    pause_reason = payload.get("pause_reason")
    if state == "paused":
        if not isinstance(pause_reason, str) or not pause_reason.strip():
            raise ValueError("paused queue requires a pause_reason")
    elif pause_reason is not None:
        raise ValueError("active queue must not have a pause_reason")
    activation = payload.get("activation")
    suspension = payload.get("suspension")
    if state == "paused":
        if activation is not None:
            raise ValueError("paused queue must not have activation evidence")
        if suspension is not None:
            if (
                not isinstance(suspension, dict)
                or set(suspension) != {"active_queue_sha256", "schema"}
                or suspension.get("schema")
                != "axiom-encode/signed-encoding-queue-suspension/v1"
            ):
                raise ValueError("paused queue suspension evidence is malformed")
            _require_digest(
                suspension.get("active_queue_sha256"),
                "suspension active_queue_sha256",
            )
    else:
        if suspension is not None:
            raise ValueError("active queue must not have suspension evidence")
        expected_activation_fields = {
            "check_runs_sha256",
            "finalizer_head_sha",
            "finalizer_run_attempt",
            "finalizer_run_url",
            "previous_queue_object_sha256",
            "pull_requests_sha256",
            "rulespec_ref",
            "schema",
            "workflow_runs_sha256",
        }
        if (
            not isinstance(activation, dict)
            or set(activation) != expected_activation_fields
            or activation.get("schema")
            != "axiom-encode/signed-encoding-queue-activation/v1"
            or activation.get("rulespec_ref")
            != (
                payload.get("dispatch", {}).get("rulespec_ref")
                if isinstance(payload.get("dispatch"), dict)
                else None
            )
            or not isinstance(activation.get("finalizer_run_url"), str)
            or FINALIZER_RUN_PATTERN.fullmatch(activation["finalizer_run_url"]) is None
            or activation.get("finalizer_run_attempt") != 1
        ):
            raise ValueError("active queue requires valid finalization evidence")
        _require_sha(activation.get("finalizer_head_sha"), "finalizer_head_sha")
        for field in (
            "check_runs_sha256",
            "previous_queue_object_sha256",
            "pull_requests_sha256",
            "workflow_runs_sha256",
        ):
            _require_digest(activation.get(field), field)

    release = payload.get("release")
    if not isinstance(release, dict):
        raise ValueError("queue release binding is missing")
    if not isinstance(release.get("name"), str) or not release["name"].strip():
        raise ValueError("queue release name is missing")
    if RELEASE_NAME_PATTERN.fullmatch(release["name"]) is None:
        raise ValueError("queue release name is malformed")
    _require_digest(release.get("content_sha256"), "release content_sha256")
    _require_digest(release.get("manifest_sha256"), "release manifest_sha256")

    dispatch = payload.get("dispatch")
    if not isinstance(dispatch, dict):
        raise ValueError("queue dispatch binding is missing")
    if dispatch.get("country") != "us":
        raise ValueError("SNAP queue country must be us")
    if dispatch.get("pr_base_branch") != "hard-cut/canonical-layout-us":
        raise ValueError("SNAP queue PR base branch is not approved")
    if dispatch.get("open_pr") is not True:
        raise ValueError("SNAP queue must publish draft pull requests")
    if dispatch.get("max_batch_size") != MAX_BATCH_SIZE:
        raise ValueError(f"SNAP queue max_batch_size must be {MAX_BATCH_SIZE}")
    for key in ("corpus_ref", "rules_engine_ref", "rulespec_ref"):
        _require_sha(dispatch.get(key), key)

    expected_counts = payload.get("expected_counts")
    profile_counts = profile["expected_counts"]
    if expected_counts != profile_counts:
        raise ValueError(f"SNAP queue expected_counts must be {profile_counts}")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("queue items must be a list")
    if len(items) != profile_counts["total"]:
        raise ValueError("queue item count does not match expected_counts")

    seen_ids: set[str] = set()
    seen_citations: set[str] = set()
    jurisdictions = set(profile_counts) - {"total"}
    counts = {"total": len(items), **dict.fromkeys(sorted(jurisdictions), 0)}
    for expected_sequence, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError("queue item must be an object")
        item_id = item.get("id")
        if not isinstance(item_id, str) or ITEM_ID_PATTERN.fullmatch(item_id) is None:
            raise ValueError("queue item id is malformed")
        jurisdiction = item.get("jurisdiction")
        expected_item_pattern = re.compile(
            rf"^{str(jurisdiction)[3:]}-[0-9]{{{profile['item_width']}}}$"
        )
        if expected_item_pattern.fullmatch(item_id) is None:
            raise ValueError(f"queue item id does not match jurisdiction: {item_id}")
        if item_id in seen_ids:
            raise ValueError(f"duplicate queue item id: {item_id}")
        seen_ids.add(item_id)
        citation = item.get("citation")
        if jurisdiction not in jurisdictions:
            raise ValueError(f"queue item jurisdiction is invalid: {item_id}")
        if (
            not isinstance(citation, str)
            or not citation.startswith(f"{jurisdiction}/")
            or any(character.isspace() for character in citation)
        ):
            raise ValueError(f"queue item citation is invalid: {item_id}")
        if citation in seen_citations:
            raise ValueError(f"duplicate queue citation: {citation}")
        seen_citations.add(citation)
        counts[jurisdiction] += 1
        if item.get("sequence") != expected_sequence:
            raise ValueError(f"queue item sequence is invalid: {item_id}")
        attempt = item.get("attempt")
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt <= 0:
            raise ValueError(f"queue item attempt is invalid: {item_id}")
        if item.get("status") not in ALLOWED_STATUSES:
            raise ValueError(f"queue item status is invalid: {item_id}")
        status = item["status"]
        evidence = item.get("evidence")
        if status not in TERMINAL_STATUSES and evidence is not None:
            raise ValueError(
                f"nonterminal queue item must not have evidence: {item_id}"
            )
        if status in TERMINAL_STATUSES:
            if not isinstance(evidence, dict):
                raise ValueError(
                    f"terminal queue item requires durable evidence: {item_id}"
                )
            url = evidence.get("url")
            if status == "completed":
                if (
                    evidence.get("type") != "merged-rulespec-pr"
                    or not isinstance(url, str)
                    or RULESPEC_PR_PATTERN.fullmatch(url) is None
                ):
                    raise ValueError(
                        f"completed queue item requires a merged RuleSpec PR: {item_id}"
                    )
                _require_sha(
                    evidence.get("merge_commit"),
                    f"{item_id} evidence merge_commit",
                )
                _require_digest(
                    evidence.get("generation_sha256"),
                    f"{item_id} evidence generation_sha256",
                )
                _require_digest(
                    evidence.get("applied_manifest_sha256"),
                    f"{item_id} evidence applied_manifest_sha256",
                )
                _require_digest(
                    evidence.get("artifact_metadata_sha256"),
                    f"{item_id} evidence artifact_metadata_sha256",
                )
                _require_sha(
                    evidence.get("rulespec_pr_head_sha"),
                    f"{item_id} evidence rulespec_pr_head_sha",
                )
                _require_sha(
                    evidence.get("target_run_head_sha"),
                    f"{item_id} evidence target_run_head_sha",
                )
                target_run_url = evidence.get("target_run_url")
                target_run_attempt = evidence.get("target_run_attempt")
                applied_manifest_path = evidence.get("applied_manifest_path")
                if (
                    not isinstance(target_run_url, str)
                    or FINALIZER_RUN_PATTERN.fullmatch(target_run_url) is None
                    or target_run_attempt != 1
                    or not isinstance(applied_manifest_path, str)
                    or not applied_manifest_path.startswith(
                        ".axiom/encoding-manifests/"
                    )
                    or Path(applied_manifest_path).is_absolute()
                    or ".." in Path(applied_manifest_path).parts
                ):
                    raise ValueError(
                        f"completed queue item evidence is malformed: {item_id}"
                    )
            else:
                note = evidence.get("note")
                issue_comment_pattern = re.compile(
                    r"^https://github\.com/TheAxiomFoundation/axiom-encode/issues/"
                    rf"{issue}#issuecomment-[0-9]+$"
                )
                if (
                    evidence.get("type") != "issue-comment"
                    or not isinstance(url, str)
                    or issue_comment_pattern.fullmatch(url) is None
                    or not isinstance(note, str)
                    or not note.strip()
                ):
                    raise ValueError(
                        f"{status} queue item requires an issue comment and note: "
                        f"{item_id}"
                    )
        if (
            not isinstance(item.get("source_kind"), str)
            or re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", item["source_kind"]) is None
        ):
            raise ValueError(f"queue item source_kind is invalid: {item_id}")
        if not isinstance(item.get("label"), str) or not item["label"].strip():
            raise ValueError(f"queue item label is missing: {item_id}")
    if counts != profile_counts:
        raise ValueError(
            f"queue jurisdiction counts drifted: expected {profile_counts}, "
            f"got {counts}"
        )
    priority_citations = profile["priority_citations"]
    if (
        priority_citations
        and tuple(item["citation"] for item in items[: len(priority_citations)])
        != priority_citations
    ):
        raise ValueError("queue priority tranche has drifted")


def queue_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def queue_object_file_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def dispatch_queue_sha256(payload: dict[str, Any]) -> str:
    """Return the exact active queue digest that authorized target runs."""

    if payload.get("state") == "active":
        return queue_object_file_sha256(payload)
    suspension = payload.get("suspension")
    if not isinstance(suspension, dict):
        raise ValueError("paused queue lacks active dispatch digest evidence")
    return _require_digest(
        suspension.get("active_queue_sha256"),
        "suspension active_queue_sha256",
    )


def item_generation_sha256(
    payload: dict[str, Any],
    item: dict[str, Any],
) -> str:
    """Return the stable trust identity for one queue item attempt."""

    generation = {
        "attempt": item["attempt"],
        "citation": item["citation"],
        "dispatch": payload["dispatch"],
        "item_id": item["id"],
        "queue_id": payload["queue_id"],
        "release": payload["release"],
    }
    canonical = json.dumps(
        generation,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _selection_payload(
    payload: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_items = []
    for item in items:
        selected = copy.deepcopy(item)
        selected["generation_sha256"] = item_generation_sha256(payload, item)
        selected_items.append(selected)
    selection = {
        "dispatch": payload["dispatch"],
        "issue": payload["issue"],
        "items": selected_items,
        "queue_id": payload["queue_id"],
        "schema": SCHEMA,
        "state": payload["state"],
    }
    if payload.get("pause_reason") is not None:
        selection["pause_reason"] = payload["pause_reason"]
    return selection


def selectable_items(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the full ordered candidate pool for history-aware selection."""

    validate_queue(payload)
    items = (
        [item for item in payload["items"] if item["status"] in SELECTABLE_STATUSES]
        if payload["state"] == "active"
        else []
    )
    return _selection_payload(payload, items)


def select_items(
    payload: dict[str, Any],
    *,
    item_ids: str,
    limit: int,
) -> dict[str, Any]:
    """Select an explicit or leading bounded tranche from a valid queue."""

    validate_queue(payload)
    if limit < 1 or limit > MAX_BATCH_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_BATCH_SIZE}")
    requested = [
        value.strip() for value in re.split(r"[,\n]", item_ids) if value.strip()
    ]
    if len(requested) != len(set(requested)):
        raise ValueError("item_ids contains duplicates")
    by_id = {item["id"]: item for item in payload["items"]}
    unknown = [item_id for item_id in requested if item_id not in by_id]
    if unknown:
        raise ValueError("unknown queue item ids: " + ", ".join(unknown))
    if requested:
        if payload["state"] != "active":
            raise ValueError("queue is paused")
        if len(requested) > limit:
            raise ValueError("explicit item_ids exceed the requested limit")
        selected = [by_id[item_id] for item_id in requested]
    else:
        selected = (
            [
                item
                for item in payload["items"]
                if item["status"] in SELECTABLE_STATUSES
            ][:limit]
            if payload["state"] == "active"
            else []
        )
    blocked = [
        item["id"] for item in selected if item["status"] not in SELECTABLE_STATUSES
    ]
    if blocked:
        raise ValueError("queue items are not selectable: " + ", ".join(blocked))
    return _selection_payload(payload, selected)


def validate_tracked_dispatch(
    path: Path,
    *,
    queue_id: str,
    item_id: str,
    manifest_sha256: str,
    item_generation_sha256_value: str,
    citation: str,
    country: str,
    rulespec_ref: str,
    pr_base_branch: str,
    corpus_ref: str,
    rules_engine_ref: str,
    open_pr: bool,
) -> None:
    """Bind targeted workflow inputs to one exact committed queue item."""

    payload = _load_json(path)
    validate_queue(payload)
    if payload["queue_id"] != queue_id:
        raise ValueError("tracked dispatch queue_id does not match the manifest")
    if queue_file_sha256(path) != manifest_sha256:
        raise ValueError("tracked dispatch queue manifest digest does not match")
    matches = [item for item in payload["items"] if item["id"] == item_id]
    if len(matches) != 1:
        raise ValueError("tracked dispatch queue item is unavailable")
    item = matches[0]
    if item["citation"] != citation:
        raise ValueError("tracked dispatch citation does not match the queue item")
    if item["status"] not in SELECTABLE_STATUSES:
        raise ValueError("tracked dispatch queue item is not selectable")
    if payload["state"] != "active":
        raise ValueError("tracked dispatch queue is paused")
    if item_generation_sha256(payload, item) != item_generation_sha256_value:
        raise ValueError("tracked dispatch queue item generation does not match")
    dispatch = payload["dispatch"]
    expected = {
        "country": country,
        "rulespec_ref": rulespec_ref,
        "pr_base_branch": pr_base_branch,
        "corpus_ref": corpus_ref,
        "rules_engine_ref": rules_engine_ref,
        "open_pr": open_pr,
    }
    mismatches = [key for key, value in expected.items() if dispatch.get(key) != value]
    if mismatches:
        raise ValueError(
            "tracked dispatch trust inputs do not match the queue: "
            + ", ".join(mismatches)
        )


def validate_release_pin(
    queue_path: Path,
    *,
    toolchain_path: Path,
    manifest_sha256: str,
) -> None:
    """Bind the runtime RuleSpec toolchain pin to the exact tracked queue."""

    payload = _load_json(queue_path)
    validate_queue(payload)
    if queue_file_sha256(queue_path) != manifest_sha256:
        raise ValueError("runtime queue manifest digest does not match")
    try:
        from scripts.materialize_corpus_release import load_release_pin
    except ModuleNotFoundError:
        from materialize_corpus_release import load_release_pin

    release_name, release_sha = load_release_pin(toolchain_path)
    release = payload["release"]
    if (release_name, release_sha) != (
        release["name"],
        release["content_sha256"],
    ):
        raise ValueError("runtime RuleSpec corpus release pin does not match the queue")


def verify_activation_evidence(
    queue_path: Path,
    *,
    previous_queue_path: Path,
    expected_base_sha: str,
    finalized_queue_path: Path,
    check_runs: object,
    pull_requests: object,
    workflow_runs: object,
    finalizer_run: object,
    finalizer_jobs: object,
    require_success: bool,
) -> None:
    """Authenticate an active queue against one finalizer run and its artifact."""

    queue = _load_json(queue_path)
    previous_queue = _load_json(previous_queue_path)
    finalized = _load_json(finalized_queue_path)
    validate_queue(queue)
    validate_queue(previous_queue)
    if (
        queue["state"] != "active"
        or queue != finalized
        or queue_path.read_bytes() != finalized_queue_path.read_bytes()
    ):
        raise ValueError("active queue differs from the finalizer artifact")
    if previous_queue["state"] != "paused":
        raise ValueError("activation base queue must be paused")
    activation = queue["activation"]
    _require_sha(expected_base_sha, "expected_base_sha")
    if activation["finalizer_head_sha"] != expected_base_sha:
        raise ValueError("activation finalizer head does not match the pull request base")
    if activation["previous_queue_object_sha256"] != _json_sha256(previous_queue):
        raise ValueError("activation base queue digest does not match")
    if activation["check_runs_sha256"] != _json_sha256(check_runs):
        raise ValueError("activation check run evidence digest does not match")
    if activation["pull_requests_sha256"] != _json_sha256(pull_requests):
        raise ValueError("activation pull request evidence digest does not match")
    if activation["workflow_runs_sha256"] != _json_sha256(workflow_runs):
        raise ValueError("activation workflow run evidence digest does not match")
    _validate_green_check_runs(check_runs)
    if not isinstance(finalizer_run, dict):
        raise ValueError("finalizer run evidence must be an object")
    expected_run = {
        "event": "workflow_dispatch",
        "head_branch": "main",
        "head_sha": activation["finalizer_head_sha"],
        "html_url": activation["finalizer_run_url"],
        "path": ".github/workflows/finalize-signed-snap-queue.yml",
    }
    mismatches = [
        field
        for field, expected in expected_run.items()
        if finalizer_run.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            "activation finalizer run identity does not match: " + ", ".join(mismatches)
        )
    _verify_attempt_one_job(
        finalizer_run,
        finalizer_jobs,
        expected_name="Finalize protected SNAP queue tranche",
        label="activation finalizer",
        require_success=require_success,
    )


def verify_activation_commit(
    queue_path: Path,
    *,
    provenance: object,
    pull_request: object,
    current_base_sha: str,
    current_head_sha: str,
    current_tree_sha: str,
    current_changed_files: object,
) -> None:
    """Bind an activation PR to the exact finalizer-created commit and tree."""

    _require_sha(current_base_sha, "current_base_sha")
    _require_sha(current_head_sha, "current_head_sha")
    _require_sha(current_tree_sha, "current_tree_sha")
    queue = _load_json(queue_path)
    profile = _queue_profile(queue["queue_id"])
    activation_changed_files = (
        profile["activation_path"],
        "pyproject.toml",
        "src/axiom_encode/__init__.py",
        "uv.lock",
    )
    if not isinstance(provenance, dict):
        raise ValueError("activation commit provenance must be an object")
    expected_fields = {
        "base_sha",
        "changed_files",
        "head_sha",
        "queue_path",
        "queue_sha256",
        "repository",
        "schema",
        "tree_sha",
    }
    if (
        set(provenance) != expected_fields
        or provenance.get("schema")
        != "axiom-encode/snap-queue-activation-commit/v1"
        or provenance.get("repository") != "TheAxiomFoundation/axiom-encode"
        or provenance.get("queue_path") != activation_changed_files[0]
        or provenance.get("queue_sha256") != queue_file_sha256(queue_path)
        or provenance.get("base_sha") != current_base_sha
        or provenance.get("head_sha") != current_head_sha
        or provenance.get("tree_sha") != current_tree_sha
        or provenance.get("changed_files") != list(activation_changed_files)
        or current_changed_files != list(activation_changed_files)
    ):
        raise ValueError("activation commit or changed-file inventory does not match")
    if not isinstance(pull_request, dict):
        raise ValueError("activation pull request must be an object")
    base = pull_request.get("base")
    head = pull_request.get("head")
    head_repo = head.get("repo") if isinstance(head, dict) else None
    if (
        pull_request.get("state") != "open"
        or not isinstance(base, dict)
        or base.get("ref") != "main"
        or base.get("sha") != current_base_sha
        or not isinstance(head, dict)
        or head.get("sha") != current_head_sha
        or not isinstance(head_repo, dict)
        or head_repo.get("full_name") != "TheAxiomFoundation/axiom-encode"
    ):
        raise ValueError("activation pull request identity does not match")


def verify_merge_authorization(
    queue_path: Path,
    *,
    authorization: object,
    merge_run: object,
    merge_jobs: object,
    pull_request: object,
    current_head_sha: str,
    queue_change_sha: str,
) -> None:
    """Authenticate the trusted workflow that merged an active queue."""

    queue = _load_json(queue_path)
    validate_queue(queue)
    profile = _queue_profile(queue["queue_id"])
    if queue["state"] != "active":
        raise ValueError("merge authorization is only valid for an active queue")
    _require_sha(current_head_sha, "current_head_sha")
    _require_sha(queue_change_sha, "queue_change_sha")
    if not isinstance(authorization, dict):
        raise ValueError("merge authorization must be an object")
    expected_fields = {
        "activation_pr_head_sha",
        "activation_pr_number",
        "merge_commit",
        "merge_workflow_run_id",
        "merge_workflow_run_attempt",
        "merge_workflow_run_url",
        "queue_path",
        "queue_sha256",
        "repository",
        "schema",
    }
    if (
        set(authorization) != expected_fields
        or authorization.get("schema")
        != "axiom-encode/snap-queue-merge-authorization/v1"
        or authorization.get("repository") != "TheAxiomFoundation/axiom-encode"
        or authorization.get("queue_path") != profile["activation_path"]
        or authorization.get("queue_sha256") != queue_file_sha256(queue_path)
        or authorization.get("merge_workflow_run_attempt") != 1
    ):
        raise ValueError("merge authorization does not match the active queue")
    _require_sha(authorization.get("activation_pr_head_sha"), "activation_pr_head_sha")
    _require_sha(authorization.get("merge_commit"), "merge_commit")
    run_id = authorization.get("merge_workflow_run_id")
    pr_number = authorization.get("activation_pr_number")
    if (
        not isinstance(run_id, int)
        or isinstance(run_id, bool)
        or run_id <= 0
        or not isinstance(pr_number, int)
        or isinstance(pr_number, bool)
        or pr_number <= 0
    ):
        raise ValueError("merge authorization identifiers are malformed")
    expected_run_url = (
        "https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/"
        f"{run_id}"
    )
    if authorization.get("merge_workflow_run_url") != expected_run_url:
        raise ValueError("merge authorization workflow URL is malformed")
    if not isinstance(merge_run, dict):
        raise ValueError("merge workflow run evidence must be an object")
    expected_run = {
        "event": "workflow_dispatch",
        "head_branch": "main",
        "html_url": expected_run_url,
        "path": ".github/workflows/merge-snap-queue-activation.yml",
    }
    if any(merge_run.get(field) != value for field, value in expected_run.items()):
        raise ValueError("merge workflow run identity or conclusion does not match")
    _verify_attempt_one_job(
        merge_run,
        merge_jobs,
        expected_name="Merge reviewed SNAP queue activation",
        label="activation merge",
        require_success=True,
    )
    if not isinstance(pull_request, dict):
        raise ValueError("activation pull request evidence must be an object")
    base = pull_request.get("base")
    head = pull_request.get("head")
    head_repo = head.get("repo") if isinstance(head, dict) else None
    merged_by = pull_request.get("merged_by")
    if (
        pull_request.get("number") != pr_number
        or pull_request.get("state") != "closed"
        or pull_request.get("merged_at") is None
        or pull_request.get("merge_commit_sha") != authorization["merge_commit"]
        or not isinstance(base, dict)
        or base.get("ref") != "main"
        or not isinstance(head, dict)
        or head.get("sha") != authorization["activation_pr_head_sha"]
        or not isinstance(head_repo, dict)
        or head_repo.get("full_name") != "TheAxiomFoundation/axiom-encode"
        or not isinstance(merged_by, dict)
        or merged_by.get("login") != "github-actions[bot]"
    ):
        raise ValueError("activation pull request was not merged by the trusted workflow")
    if queue_change_sha != authorization["merge_commit"]:
        raise ValueError("active queue was changed outside its authorized merge commit")


def verify_paused_transition(
    queue_path: Path,
    *,
    previous_queue_path: Path,
) -> None:
    """Reject manual completion and terminal-evidence rewrites in paused PRs."""

    queue = _load_json(queue_path)
    previous = _load_json(previous_queue_path)
    validate_queue(queue)
    validate_queue(previous)
    if queue["state"] != "paused":
        raise ValueError("paused transition verification requires a paused queue")
    for field in (
        "description",
        "dispatch",
        "expected_counts",
        "issue",
        "queue_id",
        "release",
        "schema",
    ):
        if queue[field] != previous[field]:
            raise ValueError(f"paused queue transition changed trusted field {field}")
    if previous["state"] == "paused" and queue.get("suspension") != previous.get(
        "suspension"
    ):
        raise ValueError("paused queue transition changed suspension evidence")
    previous_items = {item["id"]: item for item in previous["items"]}
    current_items = {item["id"]: item for item in queue["items"]}
    if set(previous_items) != set(current_items):
        raise ValueError("paused queue transition changed the item inventory")
    immutable_fields = (
        "citation",
        "id",
        "jurisdiction",
        "label",
        "sequence",
        "source_kind",
    )
    for item_id, current in current_items.items():
        prior = previous_items[item_id]
        if any(current[field] != prior[field] for field in immutable_fields):
            raise ValueError(f"paused queue transition rewrote item {item_id}")
        if current["status"] == "completed" and prior["status"] != "completed":
            raise ValueError(
                f"paused queue transition cannot complete item {item_id}"
            )
        if prior["status"] in TERMINAL_STATUSES and current != prior:
            raise ValueError(
                f"paused queue transition rewrote terminal item {item_id}"
            )
        if current["attempt"] < prior["attempt"]:
            raise ValueError(
                f"paused queue transition decreased attempt for item {item_id}"
            )
        if current["attempt"] > prior["attempt"] + 1:
            raise ValueError(
                f"paused queue transition skipped an attempt for item {item_id}"
            )
        if current["attempt"] == prior["attempt"] + 1 and current["status"] != "retryable":
            raise ValueError(
                f"paused queue transition changed attempt without retrying {item_id}"
            )
        if (
            current["status"] == "retryable"
            and current != prior
            and current["attempt"] != prior["attempt"] + 1
        ):
            raise ValueError(
                f"paused queue transition retried {item_id} without a new attempt"
            )
    if previous["state"] == "active":
        expected = pause_queue(
            previous,
            reason=queue["pause_reason"],
            active_queue_sha256=queue_file_sha256(previous_queue_path),
        )
        if queue != expected:
            raise ValueError("active queue pause transition changed queue items or trust")


def _flatten_pull_requests(payload: object) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("pull request history must be a JSON array")
    flattened: list[dict[str, Any]] = []
    for page in payload:
        values = page if isinstance(page, list) else [page]
        for value in values:
            if not isinstance(value, dict):
                raise ValueError("pull request history contains a non-object")
            flattened.append(value)
    return flattened


def _flatten_workflow_runs(payload: object) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("workflow run history must be a JSON array")
    flattened: list[dict[str, Any]] = []
    for page in payload:
        if not isinstance(page, dict):
            raise ValueError("workflow run history page must be an object")
        runs = page.get("workflow_runs")
        if not isinstance(runs, list):
            raise ValueError("workflow run history page has no workflow_runs")
        for run in runs:
            if not isinstance(run, dict):
                raise ValueError("workflow run history contains a non-object")
            flattened.append(run)
    return flattened


def _flatten_workflow_jobs(payload: object) -> list[dict[str, Any]]:
    pages = payload if isinstance(payload, list) else [payload]
    flattened: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError("workflow job history page must be an object")
        jobs = page.get("jobs")
        if not isinstance(jobs, list):
            raise ValueError("workflow job history page has no jobs")
        for job in jobs:
            if not isinstance(job, dict):
                raise ValueError("workflow job history contains a non-object")
            flattened.append(job)
    return flattened


def _verify_attempt_one_job(
    run: dict[str, Any],
    jobs_payload: object,
    *,
    expected_name: str,
    label: str,
    require_success: bool,
) -> None:
    run_attempt = run.get("run_attempt")
    if (
        not isinstance(run_attempt, int)
        or isinstance(run_attempt, bool)
        or run_attempt < 1
    ):
        raise ValueError(f"{label} run attempt is malformed")
    status = run.get("status")
    conclusion = run.get("conclusion")
    if require_success:
        if status != "completed" or conclusion != "success":
            raise ValueError(f"{label} run has not completed successfully")
    elif status not in {"in_progress", "completed"} or (
        status == "completed" and conclusion != "success"
    ):
        raise ValueError(f"{label} run is not successful or in progress")

    jobs = [
        job
        for job in _flatten_workflow_jobs(jobs_payload)
        if job.get("name") == expected_name
    ]
    by_attempt: dict[int, list[dict[str, Any]]] = {}
    for job in jobs:
        attempt = job.get("run_attempt")
        if isinstance(attempt, int) and not isinstance(attempt, bool):
            by_attempt.setdefault(attempt, []).append(job)
    if set(by_attempt) != set(range(1, run_attempt + 1)) or any(
        len(values) != 1 for values in by_attempt.values()
    ):
        raise ValueError(f"{label} job history is incomplete")
    first = by_attempt[1][0]
    first_success = first.get("status") == "completed" and first.get(
        "conclusion"
    ) == "success"
    first_in_progress = (
        not require_success
        and run_attempt == 1
        and first.get("status") == "in_progress"
        and first.get("conclusion") is None
    )
    if not first_success and not first_in_progress:
        raise ValueError(f"{label} attempt 1 was not successful or in progress")
    if any(
        job.get("status") != "completed" or job.get("conclusion") != "skipped"
        for attempt, values in by_attempt.items()
        if attempt > 1
        for job in values
    ):
        raise ValueError(f"{label} later attempt executed the protected job")


def _validate_green_check_runs(payload: object) -> None:
    if not isinstance(payload, list):
        raise ValueError("RuleSpec check run evidence must be a JSON array")
    checks: list[dict[str, Any]] = []
    for page in payload:
        if not isinstance(page, dict) or not isinstance(page.get("check_runs"), list):
            raise ValueError("RuleSpec check run evidence page is malformed")
        for check in page["check_runs"]:
            if not isinstance(check, dict):
                raise ValueError("RuleSpec check run evidence contains a non-object")
            checks.append(check)
    if not checks:
        raise ValueError("RuleSpec tip has no check run evidence")
    unacceptable = [
        str(check.get("name") or "unknown")
        for check in checks
        if check.get("status") != "completed"
        or check.get("conclusion") not in {"neutral", "skipped", "success"}
    ]
    if unacceptable:
        raise ValueError(
            "RuleSpec tip does not have green check runs: " + ", ".join(unacceptable)
        )


def reconcile_candidates(
    selection: dict[str, Any],
    *,
    pull_requests: object,
    workflow_runs: object,
) -> dict[str, Any]:
    """Reconcile selectable items against durable PR state and run recovery state."""

    queue_id = selection.get("queue_id")
    items = selection.get("items")
    if not isinstance(queue_id, str) or QUEUE_ID_PATTERN.fullmatch(queue_id) is None:
        raise ValueError("selection queue_id is malformed")
    if not isinstance(items, list):
        raise ValueError("selection items must be a list")
    prs = _flatten_pull_requests(pull_requests)
    runs = _flatten_workflow_runs(workflow_runs)
    reconciled: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("selection item must be an object")
        item_id = item.get("id")
        if not isinstance(item_id, str) or ITEM_ID_PATTERN.fullmatch(item_id) is None:
            raise ValueError("selection item id is malformed")
        generation_sha256 = _require_digest(
            item.get("generation_sha256"),
            f"{item_id} generation_sha256",
        )
        pr_marker = f"Queue item: `{queue_id}/{item_id}`"
        pr_identity_markers = (
            pr_marker,
            f"Citation: `{item['citation']}`",
            f"Base commit: `{selection['dispatch']['rulespec_ref']}`",
            f"Base branch: `{selection['dispatch']['pr_base_branch']}`",
            f"Queue generation SHA-256: `{generation_sha256}`",
        )
        run_marker = f"[{queue_id}:{item_id}:{generation_sha256}]"
        matching_prs = []
        for pr in prs:
            base = pr.get("base")
            head = pr.get("head")
            head_repo = head.get("repo") if isinstance(head, dict) else None
            body = str(pr.get("body") or "")
            if (
                all(marker in body for marker in pr_identity_markers)
                and isinstance(base, dict)
                and base.get("ref") == selection["dispatch"]["pr_base_branch"]
                and isinstance(head_repo, dict)
                and head_repo.get("full_name") == "TheAxiomFoundation/rulespec-us"
            ):
                matching_prs.append(pr)
        matching_prs.sort(key=lambda pr: str(pr.get("updated_at") or ""), reverse=True)
        merged_pr = None
        untrusted_merged_pr = None
        target_run = None
        for pr in matching_prs:
            if pr.get("merged_at") is None:
                continue
            untrusted_merged_pr = untrusted_merged_pr or pr
            match = TARGET_RUN_MARKER_PATTERN.search(str(pr.get("body") or ""))
            run_url = match.group(1) if match is not None else None
            candidate_run = next(
                (run for run in runs if run.get("html_url") == run_url),
                None,
            )
            actor = (
                candidate_run.get("actor")
                if isinstance(candidate_run, dict)
                else None
            )
            if (
                isinstance(candidate_run, dict)
                and candidate_run.get("status") == "completed"
                and candidate_run.get("conclusion") == "success"
                and candidate_run.get("event") == "workflow_dispatch"
                and candidate_run.get("head_branch") == "main"
                and isinstance(candidate_run.get("run_attempt"), int)
                and not isinstance(candidate_run.get("run_attempt"), bool)
                and candidate_run["run_attempt"] >= 1
                and candidate_run.get("path")
                == ".github/workflows/targeted-signed-reencode.yml"
                and isinstance(actor, dict)
                and actor.get("login") == "github-actions[bot]"
                and run_marker
                in str(candidate_run.get("display_title") or "")
            ):
                merged_pr = pr
                target_run = candidate_run
                break
        open_pr = next(
            (pr for pr in matching_prs if pr.get("state") == "open"),
            None,
        )

        decision = copy.deepcopy(item)
        if merged_pr is not None:
            decision.update(
                {
                    "dispatchable": False,
                    "reason": "merged-rulespec-pr",
                    "history_url": merged_pr.get("html_url"),
                    "target_run_url": target_run.get("html_url"),
                }
            )
            reconciled.append(decision)
            continue
        if untrusted_merged_pr is not None:
            decision.update(
                {
                    "dispatchable": False,
                    "reason": "untrusted-merged-rulespec-pr",
                    "history_url": untrusted_merged_pr.get("html_url"),
                }
            )
            reconciled.append(decision)
            continue
        if open_pr is not None:
            decision.update(
                {
                    "dispatchable": False,
                    "reason": "open-rulespec-pr",
                    "history_url": open_pr.get("html_url"),
                }
            )
            reconciled.append(decision)
            continue
        if matching_prs:
            decision.update(
                {
                    "dispatchable": False,
                    "reason": "closed-unmerged-rulespec-pr",
                    "history_url": matching_prs[0].get("html_url"),
                }
            )
            reconciled.append(decision)
            continue

        matching_runs = [
            run for run in runs if run_marker in str(run.get("display_title") or "")
        ]
        matching_runs.sort(
            key=lambda run: str(run.get("created_at") or ""),
            reverse=True,
        )
        latest_run = matching_runs[0] if matching_runs else None
        if latest_run is not None:
            run_url = latest_run.get("html_url")
            status = latest_run.get("status")
            conclusion = latest_run.get("conclusion")
            if status != "completed" or conclusion is None:
                decision.update(
                    {
                        "dispatchable": False,
                        "reason": "active-workflow-run",
                        "history_url": run_url,
                    }
                )
                reconciled.append(decision)
                continue
            if conclusion == "success":
                decision.update(
                    {
                        "dispatchable": False,
                        "reason": "successful-run-without-durable-pr",
                        "history_url": run_url,
                    }
                )
                reconciled.append(decision)
                continue
            if conclusion != "success":
                decision.update(
                    {
                        "dispatchable": False,
                        "reason": f"failed-workflow-run:{conclusion}",
                        "history_url": run_url,
                    }
                )
                reconciled.append(decision)
                continue

        decision.update(
            {
                "dispatchable": True,
                "reason": "new",
                "history_url": None,
            }
        )
        reconciled.append(decision)

    return {
        **{key: value for key, value in selection.items() if key != "items"},
        "items": reconciled,
    }


def finalization_target_plan(
    payload: dict[str, Any],
    *,
    pull_requests: object,
    workflow_runs: object,
) -> dict[str, Any]:
    """Return newly merged queue items whose target artifacts must be fetched."""

    validate_queue(payload)
    if payload["state"] != "paused":
        raise ValueError("target evidence planning requires a paused queue")
    selectable = [
        item for item in payload["items"] if item["status"] in SELECTABLE_STATUSES
    ]
    reconciled = reconcile_candidates(
        _selection_payload(payload, selectable),
        pull_requests=pull_requests,
        workflow_runs=workflow_runs,
    )
    return {
        "queue_id": payload["queue_id"],
        "items": [
            {
                "id": item["id"],
                "pull_request_url": item["history_url"],
                "target_run_url": item["target_run_url"],
            }
            for item in reconciled["items"]
            if item["reason"] == "merged-rulespec-pr"
        ],
        "schema": "axiom-encode/finalization-target-plan/v1",
    }


def _verify_target_evidence(
    *,
    payload: dict[str, Any],
    item: dict[str, Any],
    decision: dict[str, Any],
    pull_request: dict[str, Any],
    rulespec_root: Path,
    evidence: object,
) -> dict[str, Any]:
    if not isinstance(evidence, dict) or set(evidence) != {
        "apply_manifests",
        "jobs",
        "metadata",
        "pull_request",
        "run",
    }:
        raise ValueError(f"{item['id']} target evidence is malformed")
    run = evidence["run"]
    jobs = evidence["jobs"]
    metadata = evidence["metadata"]
    artifact_pr = evidence["pull_request"]
    inventory = evidence["apply_manifests"]
    if not all(
        isinstance(value, dict)
        for value in (run, metadata, artifact_pr, inventory)
    ):
        raise ValueError(f"{item['id']} target evidence contains a non-object")
    actor = run.get("actor")
    expected_run_url = decision["target_run_url"]
    if (
        run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or run.get("event") != "workflow_dispatch"
        or run.get("head_branch") != "main"
        or run.get("html_url") != expected_run_url
        or run.get("path") != ".github/workflows/targeted-signed-reencode.yml"
        or not isinstance(actor, dict)
        or actor.get("login") != "github-actions[bot]"
    ):
        raise ValueError(f"{item['id']} target run is not trusted and successful")
    _verify_attempt_one_job(
        run,
        jobs,
        expected_name="Queue protected signed RuleSpec re-encode",
        label=f"{item['id']} target",
        require_success=True,
    )
    run_id = run.get("id")
    if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
        raise ValueError(f"{item['id']} target run id is malformed")
    expected_metadata = {
        "citation": item["citation"],
        "encoder_commit": run.get("head_sha"),
        "pr_base_branch": payload["dispatch"]["pr_base_branch"],
        "queue_id": payload["queue_id"],
        "queue_item_generation_sha256": decision["generation_sha256"],
        "queue_item_id": item["id"],
        "queue_manifest_sha256": dispatch_queue_sha256(payload),
        "rulespec_base": payload["dispatch"]["rulespec_ref"],
        "schema": "axiom-encode/targeted-reencode-artifact/v1",
        "workflow_run_attempt": 1,
        "workflow_run_id": str(run_id),
    }
    mismatches = [
        field
        for field, expected in expected_metadata.items()
        if metadata.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            f"{item['id']} target artifact metadata does not match: "
            + ", ".join(mismatches)
        )
    pr_head = pull_request.get("head")
    artifact_pr_head = artifact_pr.get("head")
    if (
        artifact_pr.get("number") != pull_request.get("number")
        or artifact_pr.get("html_url") != pull_request.get("html_url")
        or not isinstance(pr_head, dict)
        or not isinstance(artifact_pr_head, dict)
        or artifact_pr_head.get("sha") != pr_head.get("sha")
    ):
        raise ValueError(f"{item['id']} target artifact PR identity does not match")
    inventory_items = inventory.get("items")
    if (
        inventory.get("schema") != "axiom-encode/applied-manifest-inventory/v1"
        or not isinstance(inventory_items, list)
    ):
        raise ValueError(f"{item['id']} applied manifest inventory is malformed")
    matches = [
        value
        for value in inventory_items
        if isinstance(value, dict) and value.get("citation") == item["citation"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{item['id']} target artifact lacks one applied manifest"
        )
    applied = matches[0]
    relative_path = applied.get("path")
    digest = _require_digest(
        applied.get("sha256"),
        f"{item['id']} applied manifest sha256",
    )
    if (
        not isinstance(relative_path, str)
        or not relative_path.startswith(".axiom/encoding-manifests/")
        or Path(relative_path).is_absolute()
        or ".." in Path(relative_path).parts
    ):
        raise ValueError(f"{item['id']} applied manifest path is unsafe")
    manifest_path = rulespec_root / relative_path
    if (
        not manifest_path.is_file()
        or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != digest
    ):
        raise ValueError(
            f"{item['id']} merged RuleSpec tree lacks the signed applied manifest"
        )
    return {
        "applied_manifest_path": relative_path,
        "applied_manifest_sha256": digest,
        "artifact_metadata_sha256": _json_sha256(metadata),
        "rulespec_pr_head_sha": _require_sha(
            pr_head.get("sha"),
            f"{item['id']} RuleSpec PR head",
        ),
        "target_run_head_sha": _require_sha(
            run.get("head_sha"),
            f"{item['id']} target run head",
        ),
        "target_run_attempt": 1,
        "target_run_url": expected_run_url,
    }


def finalize_and_repin(
    payload: dict[str, Any],
    *,
    rulespec_root: Path,
    check_runs: object,
    pull_requests: object,
    workflow_runs: object,
    target_evidence: object,
    new_rulespec_ref: str,
    finalizer_head_sha: str,
    finalizer_run_url: str,
    finalizer_run_attempt: int = 1,
    reviewed_rulespec_refs: frozenset[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Finalize a quiet tranche and advance the queue to one reviewed base tip."""

    validate_queue(payload)
    if payload["state"] != "paused":
        raise ValueError("finalize-repin requires a paused queue")
    _require_sha(new_rulespec_ref, "new_rulespec_ref")
    _require_sha(finalizer_head_sha, "finalizer_head_sha")
    if FINALIZER_RUN_PATTERN.fullmatch(finalizer_run_url) is None:
        raise ValueError("finalizer_run_url is malformed")
    if finalizer_run_attempt != 1:
        raise ValueError("finalizer_run_attempt must be 1")
    _validate_green_check_runs(check_runs)
    previous_queue_object_sha256 = _json_sha256(payload)
    if new_rulespec_ref == payload["dispatch"]["rulespec_ref"]:
        raise ValueError("new_rulespec_ref must advance the queue base")
    if reviewed_rulespec_refs is None:
        try:
            from scripts.prepare_signed_backfill import REVIEWED_RULESPEC_REFS
        except ModuleNotFoundError:
            from prepare_signed_backfill import REVIEWED_RULESPEC_REFS

        reviewed_rulespec_refs = REVIEWED_RULESPEC_REFS
    country = payload["dispatch"]["country"]
    if (country, new_rulespec_ref) not in reviewed_rulespec_refs:
        raise ValueError(
            "new_rulespec_ref is not independently reviewed and allowlisted"
        )
    try:
        checkout_head = _git(rulespec_root, "rev-parse", "HEAD")
        checkout_status = _git(rulespec_root, "status", "--porcelain")
        remote_tip = _git(
            rulespec_root,
            "rev-parse",
            f"refs/remotes/origin/{payload['dispatch']['pr_base_branch']}",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "finalize-repin requires an exact RuleSpec branch checkout"
        ) from exc
    if checkout_head != new_rulespec_ref or remote_tip != new_rulespec_ref:
        raise ValueError(
            "new_rulespec_ref is not the exact checked-out remote branch tip"
        )
    if checkout_status:
        raise ValueError("finalize-repin RuleSpec checkout is not clean")
    old_rulespec_ref = payload["dispatch"]["rulespec_ref"]
    try:
        _git(
            rulespec_root,
            "merge-base",
            "--is-ancestor",
            old_rulespec_ref,
            new_rulespec_ref,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "new RuleSpec base does not contain the prior queue base"
        ) from exc
    try:
        from scripts.materialize_corpus_release import load_release_pin
    except ModuleNotFoundError:
        from materialize_corpus_release import load_release_pin

    pinned_release = load_release_pin(rulespec_root / ".axiom/toolchain.toml")
    release = payload["release"]
    if pinned_release != (release["name"], release["content_sha256"]):
        raise ValueError("new RuleSpec base changed the queue corpus release pin")
    prs = _flatten_pull_requests(pull_requests)
    runs = _flatten_workflow_runs(workflow_runs)
    queue_prefix = f"Queue item: `{payload['queue_id']}/"
    trusted_queue_prs = []
    for pr in prs:
        base = pr.get("base")
        head = pr.get("head")
        head_repo = head.get("repo") if isinstance(head, dict) else None
        if (
            queue_prefix in str(pr.get("body") or "")
            and isinstance(base, dict)
            and base.get("ref") == payload["dispatch"]["pr_base_branch"]
            and isinstance(head_repo, dict)
            and head_repo.get("full_name") == "TheAxiomFoundation/rulespec-us"
        ):
            trusted_queue_prs.append(pr)
    open_prs = [pr for pr in trusted_queue_prs if pr.get("state") == "open"]
    if open_prs:
        urls = ", ".join(str(pr.get("html_url") or "unknown") for pr in open_prs)
        raise ValueError(f"cannot repin while queue pull requests are open: {urls}")

    run_prefix = f"[{payload['queue_id']}:"
    active_runs = [
        run
        for run in runs
        if run_prefix in str(run.get("display_title") or "")
        and (run.get("status") != "completed" or run.get("conclusion") is None)
    ]
    if active_runs:
        urls = ", ".join(str(run.get("html_url") or "unknown") for run in active_runs)
        raise ValueError(f"cannot repin while queue workflow runs are active: {urls}")

    prs_by_url = {
        pr.get("html_url"): pr
        for pr in trusted_queue_prs
        if isinstance(pr.get("html_url"), str)
    }
    for item in payload["items"]:
        if item["status"] != "completed":
            continue
        evidence = item["evidence"]
        pr_url = evidence["url"]
        pr = prs_by_url.get(pr_url)
        body = str(pr.get("body") or "") if isinstance(pr, dict) else ""
        expected_markers = (
            f"Queue item: `{payload['queue_id']}/{item['id']}`",
            f"Citation: `{item['citation']}`",
            (
                "Queue generation SHA-256: "
                f"`{evidence['generation_sha256']}`"
            ),
            f"Axiom Encode run: {evidence['target_run_url']}",
        )
        pr_head = pr.get("head") if isinstance(pr, dict) else None
        if (
            not isinstance(pr, dict)
            or pr.get("state") != "closed"
            or pr.get("merged_at") is None
            or pr.get("merge_commit_sha") != evidence["merge_commit"]
            or not isinstance(pr_head, dict)
            or pr_head.get("sha") != evidence["rulespec_pr_head_sha"]
            or not all(marker in body for marker in expected_markers)
        ):
            raise ValueError(
                f"completed queue item {item['id']} lacks a verified merged RuleSpec PR"
            )
        try:
            _git(
                rulespec_root,
                "merge-base",
                "--is-ancestor",
                evidence["merge_commit"],
                new_rulespec_ref,
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"new RuleSpec base omits completed queue item {item['id']}"
            ) from exc
        manifest_path = rulespec_root / evidence["applied_manifest_path"]
        if (
            not manifest_path.is_file()
            or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            != evidence["applied_manifest_sha256"]
        ):
            raise ValueError(
                f"completed queue item {item['id']} lacks its applied manifest"
            )

    selectable = [
        item for item in payload["items"] if item["status"] in SELECTABLE_STATUSES
    ]
    selection = _selection_payload(payload, selectable)
    reconciled = reconcile_candidates(
        selection,
        pull_requests=pull_requests,
        workflow_runs=workflow_runs,
    )
    unresolved = [
        item
        for item in reconciled["items"]
        if item["reason"] not in {"merged-rulespec-pr", "new"}
    ]
    if unresolved:
        details = ", ".join(f"{item['id']}={item['reason']}" for item in unresolved)
        raise ValueError("queue dispositions are required before repinning: " + details)

    updated = copy.deepcopy(payload)
    if not isinstance(target_evidence, dict):
        raise ValueError("target evidence must be an object")
    by_id = {item["id"]: item for item in updated["items"]}
    for decision in reconciled["items"]:
        if decision["reason"] != "merged-rulespec-pr":
            continue
        pr_url = decision.get("history_url")
        pr = prs_by_url.get(pr_url)
        merge_commit = pr.get("merge_commit_sha") if isinstance(pr, dict) else None
        _require_sha(merge_commit, f"{decision['id']} merged PR commit")
        try:
            _git(
                rulespec_root,
                "merge-base",
                "--is-ancestor",
                merge_commit,
                new_rulespec_ref,
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"new RuleSpec base omits merged queue item {decision['id']}"
            ) from exc
        authenticated = _verify_target_evidence(
            payload=payload,
            item=by_id[decision["id"]],
            decision=decision,
            pull_request=pr,
            rulespec_root=rulespec_root,
            evidence=target_evidence.get(decision["id"]),
        )
        item = by_id[decision["id"]]
        item["status"] = "completed"
        item["evidence"] = {
            **authenticated,
            "generation_sha256": decision["generation_sha256"],
            "merge_commit": merge_commit,
            "type": "merged-rulespec-pr",
            "url": pr_url,
        }

    updated["dispatch"]["rulespec_ref"] = new_rulespec_ref
    updated["state"] = "active"
    updated.pop("pause_reason", None)
    updated.pop("suspension", None)
    updated["activation"] = {
        "check_runs_sha256": _json_sha256(check_runs),
        "finalizer_head_sha": finalizer_head_sha,
        "finalizer_run_attempt": finalizer_run_attempt,
        "finalizer_run_url": finalizer_run_url,
        "previous_queue_object_sha256": previous_queue_object_sha256,
        "pull_requests_sha256": _json_sha256(pull_requests),
        "rulespec_ref": new_rulespec_ref,
        "schema": "axiom-encode/signed-encoding-queue-activation/v1",
        "workflow_runs_sha256": _json_sha256(workflow_runs),
    }
    validate_queue(updated)
    return updated


def record_disposition(
    payload: dict[str, Any],
    *,
    item_id: str,
    status: str,
    evidence_url: str | None,
    note: str | None,
) -> dict[str, Any]:
    """Return a queue manifest with one reviewable durable item disposition."""

    validate_queue(payload)
    if status not in {"blocked", "no-executable-rule", "retryable"}:
        raise ValueError(
            "disposition status must be blocked, no-executable-rule, or retryable"
        )
    updated = copy.deepcopy(payload)
    matches = [item for item in updated["items"] if item["id"] == item_id]
    if len(matches) != 1:
        raise ValueError("disposition queue item is unavailable")
    item = matches[0]
    item["status"] = status
    item.pop("evidence", None)
    if status in {"blocked", "no-executable-rule"}:
        item["evidence"] = {
            "note": note,
            "type": "issue-comment",
            "url": evidence_url,
        }
    elif status == "retryable":
        item["attempt"] += 1
    validate_queue(updated)
    return updated


def pause_queue(
    payload: dict[str, Any],
    *,
    reason: str,
    active_queue_sha256: str,
) -> dict[str, Any]:
    """Return a reviewable fail-closed transition after dispatching one tranche."""

    validate_queue(payload)
    if payload["state"] != "active":
        raise ValueError("only an active queue can be paused")
    if not reason.strip():
        raise ValueError("pause reason must not be empty")
    _require_digest(active_queue_sha256, "active_queue_sha256")
    updated = copy.deepcopy(payload)
    updated["state"] = "paused"
    updated["pause_reason"] = reason.strip()
    updated.pop("activation", None)
    updated["suspension"] = {
        "active_queue_sha256": active_queue_sha256,
        "schema": "axiom-encode/signed-encoding-queue-suspension/v1",
    }
    validate_queue(updated)
    return updated


def _write_json(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-snap")
    build.add_argument("corpus_root", type=Path)
    build.add_argument("rulespec_root", type=Path)
    build.add_argument("--corpus-ref", required=True)
    build.add_argument("--rulespec-ref", required=True)
    build.add_argument("--rules-engine-ref", required=True)
    build.add_argument("--release-name", required=True)
    build.add_argument("--release-content-sha256", required=True)
    build.add_argument("--release-object", type=Path, required=True)
    build.add_argument("--release-public-key", type=Path, required=True)
    build.add_argument("--state", choices=("active", "paused"), required=True)
    build.add_argument("--pause-reason")

    build_all = subparsers.add_parser("build-snap-all-states")
    build_all.add_argument("corpus_root", type=Path)
    build_all.add_argument("rulespec_root", type=Path)
    build_all.add_argument("--corpus-ref", required=True)
    build_all.add_argument("--rulespec-ref", required=True)
    build_all.add_argument("--rules-engine-ref", required=True)
    build_all.add_argument("--release-name", required=True)
    build_all.add_argument("--release-content-sha256", required=True)
    build_all.add_argument("--release-object", type=Path, required=True)
    build_all.add_argument("--release-public-key", type=Path, required=True)
    build_all.add_argument("--state", choices=("active", "paused"), required=True)
    build_all.add_argument("--pause-reason")

    validate = subparsers.add_parser("validate")
    validate.add_argument("queue", type=Path)

    select = subparsers.add_parser("select")
    select.add_argument("queue", type=Path)
    select.add_argument("--item-ids", default="")
    select.add_argument("--limit", type=int, default=1)

    candidates = subparsers.add_parser("candidates")
    candidates.add_argument("queue", type=Path)

    reconcile = subparsers.add_parser("reconcile")
    reconcile.add_argument("selection", type=Path)
    reconcile.add_argument("--pull-requests", type=Path, required=True)
    reconcile.add_argument("--workflow-runs", type=Path, required=True)

    target_plan = subparsers.add_parser("finalization-target-plan")
    target_plan.add_argument("queue", type=Path)
    target_plan.add_argument("--pull-requests", type=Path, required=True)
    target_plan.add_argument("--workflow-runs", type=Path, required=True)

    disposition = subparsers.add_parser("record-disposition")
    disposition.add_argument("queue", type=Path)
    disposition.add_argument("--item-id", required=True)
    disposition.add_argument(
        "--status",
        choices=("blocked", "no-executable-rule", "retryable"),
        required=True,
    )
    disposition.add_argument("--evidence-url")
    disposition.add_argument("--note")

    pause = subparsers.add_parser("pause")
    pause.add_argument("queue", type=Path)
    pause.add_argument("--reason", required=True)

    repin = subparsers.add_parser("finalize-repin")
    repin.add_argument("queue", type=Path)
    repin.add_argument("rulespec_root", type=Path)
    repin.add_argument("--check-runs", type=Path, required=True)
    repin.add_argument("--pull-requests", type=Path, required=True)
    repin.add_argument("--workflow-runs", type=Path, required=True)
    repin.add_argument("--target-evidence", type=Path, required=True)
    repin.add_argument("--new-rulespec-ref", required=True)
    repin.add_argument("--finalizer-head-sha", required=True)
    repin.add_argument("--finalizer-run-url", required=True)

    tracked = subparsers.add_parser("validate-dispatch")
    tracked.add_argument("queue", type=Path)
    tracked.add_argument("--queue-id", required=True)
    tracked.add_argument("--item-id", required=True)
    tracked.add_argument("--manifest-sha256", required=True)
    tracked.add_argument("--item-generation-sha256", required=True)
    tracked.add_argument("--citation", required=True)
    tracked.add_argument("--country", required=True)
    tracked.add_argument("--rulespec-ref", required=True)
    tracked.add_argument("--pr-base-branch", required=True)
    tracked.add_argument("--corpus-ref", required=True)
    tracked.add_argument("--rules-engine-ref", required=True)
    tracked.add_argument("--open-pr", choices=("true", "false"), required=True)

    digest = subparsers.add_parser("sha256")
    digest.add_argument("queue", type=Path)

    release_pin = subparsers.add_parser("validate-release-pin")
    release_pin.add_argument("queue", type=Path)
    release_pin.add_argument("--toolchain", type=Path, required=True)
    release_pin.add_argument("--manifest-sha256", required=True)

    activation = subparsers.add_parser("verify-activation")
    activation.add_argument("queue", type=Path)
    activation.add_argument("--previous-queue", type=Path, required=True)
    activation.add_argument("--expected-base-sha", required=True)
    activation.add_argument("--finalized-queue", type=Path, required=True)
    activation.add_argument("--check-runs", type=Path, required=True)
    activation.add_argument("--pull-requests", type=Path, required=True)
    activation.add_argument("--workflow-runs", type=Path, required=True)
    activation.add_argument("--finalizer-run", type=Path, required=True)
    activation.add_argument("--finalizer-jobs", type=Path, required=True)
    activation.add_argument(
        "--require-success",
        choices=("true", "false"),
        required=True,
    )

    activation_commit = subparsers.add_parser("verify-activation-commit")
    activation_commit.add_argument("queue", type=Path)
    activation_commit.add_argument("--provenance", type=Path, required=True)
    activation_commit.add_argument("--pull-request", type=Path, required=True)
    activation_commit.add_argument("--current-base-sha", required=True)
    activation_commit.add_argument("--current-head-sha", required=True)
    activation_commit.add_argument("--current-tree-sha", required=True)
    activation_commit.add_argument("--current-changed-files", type=Path, required=True)

    authorization = subparsers.add_parser("verify-merge-authorization")
    authorization.add_argument("queue", type=Path)
    authorization.add_argument("--authorization", type=Path, required=True)
    authorization.add_argument("--merge-run", type=Path, required=True)
    authorization.add_argument("--merge-jobs", type=Path, required=True)
    authorization.add_argument("--pull-request", type=Path, required=True)
    authorization.add_argument("--current-head-sha", required=True)
    authorization.add_argument("--queue-change-sha", required=True)

    transition = subparsers.add_parser("verify-paused-transition")
    transition.add_argument("queue", type=Path)
    transition.add_argument("--previous-queue", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "build-snap":
            _write_json(
                build_snap_queue(
                    args.corpus_root,
                    args.rulespec_root,
                    corpus_ref=args.corpus_ref,
                    rulespec_ref=args.rulespec_ref,
                    rules_engine_ref=args.rules_engine_ref,
                    release_name=args.release_name,
                    release_content_sha256=args.release_content_sha256,
                    release_object_path=args.release_object,
                    release_public_key_path=args.release_public_key,
                    state=args.state,
                    pause_reason=args.pause_reason,
                )
            )
        elif args.command == "build-snap-all-states":
            _write_json(
                build_all_state_snap_queue(
                    args.corpus_root,
                    args.rulespec_root,
                    corpus_ref=args.corpus_ref,
                    rulespec_ref=args.rulespec_ref,
                    rules_engine_ref=args.rules_engine_ref,
                    release_name=args.release_name,
                    release_content_sha256=args.release_content_sha256,
                    release_object_path=args.release_object,
                    release_public_key_path=args.release_public_key,
                    state=args.state,
                    pause_reason=args.pause_reason,
                )
            )
        elif args.command == "validate":
            validate_queue(_load_json(args.queue))
            print("valid")
        elif args.command == "select":
            _write_json(
                select_items(
                    _load_json(args.queue),
                    item_ids=args.item_ids,
                    limit=args.limit,
                )
            )
        elif args.command == "candidates":
            _write_json(selectable_items(_load_json(args.queue)))
        elif args.command == "reconcile":
            _write_json(
                reconcile_candidates(
                    _load_json(args.selection),
                    pull_requests=json.loads(
                        args.pull_requests.read_text(encoding="utf-8")
                    ),
                    workflow_runs=json.loads(
                        args.workflow_runs.read_text(encoding="utf-8")
                    ),
                )
            )
        elif args.command == "finalization-target-plan":
            _write_json(
                finalization_target_plan(
                    _load_json(args.queue),
                    pull_requests=json.loads(
                        args.pull_requests.read_text(encoding="utf-8")
                    ),
                    workflow_runs=json.loads(
                        args.workflow_runs.read_text(encoding="utf-8")
                    ),
                )
            )
        elif args.command == "record-disposition":
            _write_json(
                record_disposition(
                    _load_json(args.queue),
                    item_id=args.item_id,
                    status=args.status,
                    evidence_url=args.evidence_url,
                    note=args.note,
                )
            )
        elif args.command == "pause":
            _write_json(
                pause_queue(
                    _load_json(args.queue),
                    reason=args.reason,
                    active_queue_sha256=queue_file_sha256(args.queue),
                )
            )
        elif args.command == "finalize-repin":
            _write_json(
                finalize_and_repin(
                    _load_json(args.queue),
                    rulespec_root=args.rulespec_root,
                    check_runs=json.loads(args.check_runs.read_text(encoding="utf-8")),
                    pull_requests=json.loads(
                        args.pull_requests.read_text(encoding="utf-8")
                    ),
                    workflow_runs=json.loads(
                        args.workflow_runs.read_text(encoding="utf-8")
                    ),
                    target_evidence=json.loads(
                        args.target_evidence.read_text(encoding="utf-8")
                    ),
                    new_rulespec_ref=args.new_rulespec_ref,
                    finalizer_head_sha=args.finalizer_head_sha,
                    finalizer_run_url=args.finalizer_run_url,
                )
            )
        elif args.command == "validate-dispatch":
            validate_tracked_dispatch(
                args.queue,
                queue_id=args.queue_id,
                item_id=args.item_id,
                manifest_sha256=args.manifest_sha256,
                item_generation_sha256_value=args.item_generation_sha256,
                citation=args.citation,
                country=args.country,
                rulespec_ref=args.rulespec_ref,
                pr_base_branch=args.pr_base_branch,
                corpus_ref=args.corpus_ref,
                rules_engine_ref=args.rules_engine_ref,
                open_pr=args.open_pr == "true",
            )
            print("valid")
        elif args.command == "sha256":
            print(queue_file_sha256(args.queue))
        elif args.command == "validate-release-pin":
            validate_release_pin(
                args.queue,
                toolchain_path=args.toolchain,
                manifest_sha256=args.manifest_sha256,
            )
            print("valid")
        elif args.command == "verify-activation":
            verify_activation_evidence(
                args.queue,
                previous_queue_path=args.previous_queue,
                expected_base_sha=args.expected_base_sha,
                finalized_queue_path=args.finalized_queue,
                check_runs=json.loads(args.check_runs.read_text(encoding="utf-8")),
                pull_requests=json.loads(
                    args.pull_requests.read_text(encoding="utf-8")
                ),
                workflow_runs=json.loads(
                    args.workflow_runs.read_text(encoding="utf-8")
                ),
                finalizer_run=json.loads(
                    args.finalizer_run.read_text(encoding="utf-8")
                ),
                finalizer_jobs=json.loads(
                    args.finalizer_jobs.read_text(encoding="utf-8")
                ),
                require_success=args.require_success == "true",
            )
            print("valid")
        elif args.command == "verify-merge-authorization":
            verify_merge_authorization(
                args.queue,
                authorization=json.loads(
                    args.authorization.read_text(encoding="utf-8")
                ),
                merge_run=json.loads(args.merge_run.read_text(encoding="utf-8")),
                merge_jobs=json.loads(args.merge_jobs.read_text(encoding="utf-8")),
                pull_request=json.loads(
                    args.pull_request.read_text(encoding="utf-8")
                ),
                current_head_sha=args.current_head_sha,
                queue_change_sha=args.queue_change_sha,
            )
            print("valid")
        elif args.command == "verify-activation-commit":
            verify_activation_commit(
                args.queue,
                provenance=json.loads(args.provenance.read_text(encoding="utf-8")),
                pull_request=json.loads(
                    args.pull_request.read_text(encoding="utf-8")
                ),
                current_base_sha=args.current_base_sha,
                current_head_sha=args.current_head_sha,
                current_tree_sha=args.current_tree_sha,
                current_changed_files=json.loads(
                    args.current_changed_files.read_text(encoding="utf-8")
                ),
            )
            print("valid")
        elif args.command == "verify-paused-transition":
            verify_paused_transition(
                args.queue,
                previous_queue_path=args.previous_queue,
            )
            print("valid")
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
