"""Pure contracts for authenticated legacy-to-v5 fresh re-encoding.

This module deliberately cannot sign, invoke a model, read Git, or mutate a
checkout.  It defines the exact historical evidence class and deterministic
receipt identity used by the CLI's broker-authenticated transaction.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Final, Mapping, NamedTuple

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.tokens import AliasToken, AnchorToken

from .corpus_resolver import normalize_corpus_identifier

TOOL: Final = "axiom-encode encode --apply --replace-legacy-rulespec-path"
EXACT_DEPENDENT_TOOL: Final = (
    "axiom-encode encode --apply --legacy-exact-dependent-rulespec-path"
)
RETAINED_SUCCESSOR_TOOL: Final = (
    "axiom-encode encode --apply --legacy-retained-successor-rulespec-path"
)
RECEIPT_SCHEMA_V1: Final = "axiom-encode/legacy-fresh-reencode-receipt/v1"
RECEIPT_SCHEMA_V2: Final = "axiom-encode/legacy-fresh-reencode-receipt/v2"
RECEIPT_SCHEMA_V3: Final = "axiom-encode/legacy-fresh-reencode-receipt/v3"
RECEIPT_SCHEMA_V4: Final = "axiom-encode/legacy-fresh-reencode-receipt/v4"
RECEIPT_SCHEMA_V5: Final = "axiom-encode/legacy-fresh-reencode-receipt/v5"
RECEIPT_SCHEMA: Final = "axiom-encode/legacy-fresh-reencode-receipt/v6"
RECEIPT_DIR: Final = ".axiom/legacy-replacements"
DESTINATION_PREDECESSOR_ABSENT: Final = "absent"
DESTINATION_PREDECESSOR_CANONICALIZED_UNOWNED_DUPLICATE: Final = (
    "canonicalized-unowned-duplicate"
)
ENCODING_MANIFEST_DIR: Final = Path(".axiom") / "encoding-manifests"
LEGACY_MANIFEST_SCHEMA: Final = "axiom-encode/applied-rulespec/v1"
LEGACY_OWNER_CLASS: Final = "v1-hmac-untrusted"
LEGACY_MANUAL_OWNER_CLASS: Final = "v1-manual-hmac-untrusted"
LEGACY_OWNER_CLASSES: Final = frozenset({LEGACY_OWNER_CLASS, LEGACY_MANUAL_OWNER_CLASS})
LEGACY_GENERATED_TOOL: Final = "axiom-encode encode --apply"
LEGACY_GENERATED_BACKENDS: Final = frozenset({"codex", "openai", "claude"})
LEGACY_SIGNATURE_KEY_ID: Final = "axiom-encode-apply-v1"
LEGACY_GENERATED_FIELDS: Final = frozenset(
    {
        "applied_files",
        "axiom_encode_git",
        "axiom_encode_version",
        "backend",
        "citation",
        "context_manifest_file",
        "context_manifest_sha256",
        "generated_at",
        "generated_output_file",
        "generated_output_root",
        "generated_output_sha256",
        "generation_prompt_sha256",
        "model",
        "run_id",
        "runner",
        "schema_version",
        "signature",
        "tool",
        "trace_file",
        "trace_sha256",
    }
)
LEGACY_GENERATED_GIT_FIELDS: Final = frozenset(
    {"commit", "dirty_tracked", "root", "version", "version_commit"}
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicated = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicated:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


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
    proof_import_repairs: int = 0
    proof_excerpt_reanchors: tuple[dict[str, object], ...] = ()


class LegacyReplacementSourceVerificationMigration(NamedTuple):
    legacy_corpus_citation_paths: tuple[str, ...]
    corpus_citation_path: str


class LegacyReplacementPendingFile(NamedTuple):
    path: Path
    before_sha256: str
    replacements: tuple[dict[str, object], ...]


class LegacyReplacementScheduledDependent(NamedTuple):
    primary: Path
    files: tuple[LegacyReplacementPendingFile, ...]


class LegacyReplacementExactDependent(NamedTuple):
    primary: Path
    legacy_manifest: LegacyReplacementFile
    legacy_files: tuple[LegacyReplacementFile, ...]
    live_files: tuple[LegacyReplacementFile, ...]
    rewrites: tuple[LegacyReplacementRewrite, ...]
    source_verification_migration: (
        LegacyReplacementSourceVerificationMigration | None
    ) = None


class LegacyReplacementRetainedSuccessor(NamedTuple):
    source: Path
    destination: Path
    legacy_manifest: LegacyReplacementFile
    legacy_files: tuple[LegacyReplacementFile, ...]
    successor_manifest: LegacyReplacementFile
    successor_files: tuple[LegacyReplacementFile, ...]


class LegacyReplacementContract(NamedTuple):
    base_commit: str
    base_tree: str
    source: Path
    destination: Path
    legacy_manifest: LegacyReplacementFile
    deleted_files: tuple[LegacyReplacementFile, ...]
    rewrites: tuple[LegacyReplacementRewrite, ...]
    scheduled_dependents: tuple[LegacyReplacementScheduledDependent, ...]
    exact_dependents: tuple[LegacyReplacementExactDependent, ...]
    destination_predecessor_class: str = DESTINATION_PREDECESSOR_ABSENT
    destination_predecessor_files: tuple[LegacyReplacementFile, ...] = ()
    retained_successors: tuple[LegacyReplacementRetainedSuccessor, ...] = ()
    metadata_reconciliations: tuple[LegacyReplacementRewrite, ...] = ()


def legacy_source_verification_citation_paths(
    verification: Mapping[str, object],
) -> tuple[str, ...]:
    """Parse one exclusive singular citation or a historical plural list."""

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
        or not raw_paths
        or not all(isinstance(item, str) and item.strip() for item in raw_paths)
    ):
        return ()
    try:
        normalized = tuple(normalize_corpus_identifier(item) for item in raw_paths)
    except (TypeError, ValueError):
        return ()
    if len(set(normalized)) != len(normalized):
        return ()
    return normalized


def migrate_legacy_exact_dependent_source_verification(
    raw: bytes,
) -> tuple[bytes, LegacyReplacementSourceVerificationMigration | None]:
    """Canonicalize one authenticated legacy composite source declaration."""

    try:
        text = raw.decode("utf-8")
        payload = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except ConstructorError as exc:
        raise ValueError(
            "legacy exact dependent source verification has an invalid mapping: "
            f"{exc.problem}"
        ) from exc
    except (UnicodeError, yaml.YAMLError, RecursionError) as exc:
        raise ValueError(
            "legacy exact dependent source verification is not valid UTF-8 YAML"
        ) from exc
    module = payload.get("module") if isinstance(payload, dict) else None
    verification = (
        module.get("source_verification") if isinstance(module, dict) else None
    )
    if not isinstance(verification, dict):
        return raw, None
    if "corpus_citation_paths" not in verification:
        return raw, None
    try:
        tokens = tuple(yaml.scan(text))
        if any(isinstance(token, (AnchorToken, AliasToken)) for token in tokens):
            raise ValueError(
                "legacy exact dependent source verification cannot contain YAML "
                "anchors or aliases"
            )
    except (yaml.YAMLError, RecursionError) as exc:
        raise ValueError(
            "legacy exact dependent source verification is not valid YAML"
        ) from exc
    if "corpus_citation_path" in verification:
        raise ValueError(
            "legacy exact dependent source verification mixes singular and plural "
            "citation fields"
        )
    if "source_sha256" in verification:
        raise ValueError(
            "legacy exact dependent plural source verification has an ambiguous "
            "aggregate source_sha256"
        )
    raw_paths = verification.get("corpus_citation_paths")
    citations = legacy_source_verification_citation_paths(verification)
    if (
        not isinstance(raw_paths, list)
        or not citations
        or tuple(raw_paths) != citations
    ):
        raise ValueError(
            "legacy exact dependent plural source history is not a unique canonical "
            "citation list"
        )
    upstream = verification.get("upstream_source_check")
    checked_paths = (
        upstream.get("checked_paths") if isinstance(upstream, dict) else None
    )
    if checked_paths != raw_paths:
        raise ValueError(
            "legacy exact dependent plural source history is not preserved exactly "
            "by upstream_source_check.checked_paths"
        )
    if any(
        not isinstance(path, str)
        or not path
        or path.strip() != path
        or any(character in path for character in "\r\n#'\"")
        for path in raw_paths
    ):
        raise ValueError(
            "legacy exact dependent plural source history is not in canonical plain "
            "YAML form"
        )

    legacy_block = "    corpus_citation_paths:\n" + "".join(
        f"      - {path}\n" for path in raw_paths
    )
    if text.count(legacy_block) != 1:
        raise ValueError(
            "legacy exact dependent plural source history does not have one exact "
            "canonical YAML representation"
        )
    migrated_block = f"    corpus_citation_path: {raw_paths[0]}\n"
    migrated_text = text.replace(legacy_block, migrated_block, 1)
    try:
        migrated_payload = yaml.load(migrated_text, Loader=_UniqueKeySafeLoader)
    except (yaml.YAMLError, RecursionError) as exc:
        raise ValueError(
            "legacy exact dependent singular source verification is invalid"
        ) from exc
    expected_payload = copy.deepcopy(payload)
    expected_verification = expected_payload["module"]["source_verification"]
    del expected_verification["corpus_citation_paths"]
    expected_verification["corpus_citation_path"] = raw_paths[0]
    if migrated_payload != expected_payload:
        raise ValueError(
            "legacy exact dependent source-verification migration changed unrelated "
            "RuleSpec structure"
        )
    return migrated_text.encode("utf-8"), (
        LegacyReplacementSourceVerificationMigration(
            legacy_corpus_citation_paths=tuple(raw_paths),
            corpus_citation_path=raw_paths[0],
        )
    )


def legacy_manual_manifest_issues(
    payload: object,
    *,
    expected_files: Mapping[str, str],
    allow_unmarked_manual_exception: bool = False,
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
    manual_exception = payload.get("manual_exception")
    unmarked_exception_admitted = allow_unmarked_manual_exception and (
        "manual_exception" not in payload or manual_exception is None
    )
    if not unmarked_exception_admitted and (
        not isinstance(manual_exception, str) or not manual_exception.strip()
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


def legacy_generated_manifest_issues(
    payload: object,
    *,
    expected_files: Mapping[str, str],
    expected_primary_path: str,
    expected_citation: str,
    jurisdiction_prefix: str,
) -> list[str]:
    """Require one strict historical generated-v1 shape as untrusted evidence.

    The historical HMAC is deliberately not verified or treated as authority.
    Authority comes from the caller's separately authorized Git base and exact
    blob hashes.  This parser only establishes that those blobs have the known
    generated-v1 ownership shape and bind the expected source identity.
    """

    if not isinstance(payload, dict):
        return ["legacy ownership manifest is not a JSON object"]
    issues: list[str] = []
    if set(payload) != LEGACY_GENERATED_FIELDS:
        issues.append("legacy generated ownership manifest fields are noncanonical")
    if payload.get("schema_version") != LEGACY_MANIFEST_SCHEMA:
        issues.append("legacy ownership manifest is not schema v1")
    if payload.get("tool") != LEGACY_GENERATED_TOOL:
        issues.append("legacy generated ownership manifest tool is unsupported")
    backend = payload.get("backend")
    runner = payload.get("runner")
    model = payload.get("model")
    if not isinstance(backend, str) or backend not in LEGACY_GENERATED_BACKENDS:
        issues.append("legacy generated ownership manifest backend is unsupported")
    if (
        not isinstance(runner, str)
        or not isinstance(backend, str)
        or not isinstance(model, str)
        or not model.strip()
        or runner != f"{backend}-{model}"
    ):
        issues.append("legacy generated ownership manifest runner/backend mismatch")
    if "manual_exception" in payload:
        issues.append("legacy generated ownership manifest claims a manual exception")
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
        or set(signature) != {"algorithm", "key_id", "value"}
        or signature.get("algorithm") != "hmac-sha256"
        or signature.get("key_id") != LEGACY_SIGNATURE_KEY_ID
        or not isinstance(signature.get("value"), str)
        or _SHA256.fullmatch(str(signature.get("value"))) is None
    ):
        issues.append("legacy generated ownership manifest has unknown signature shape")

    actual: dict[str, str] = {}
    entries = payload.get("applied_files")
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

    canonical_files = dict(expected_files)
    prefix = f"{jurisdiction_prefix}/"
    uniformly_relative = bool(canonical_files) and all(
        path.startswith(prefix) and len(path) > len(prefix) for path in canonical_files
    )
    relative_files = (
        {path[len(prefix) :]: digest for path, digest in canonical_files.items()}
        if uniformly_relative
        else {}
    )
    if actual != canonical_files and actual != relative_files:
        issues.append(
            "legacy generated ownership manifest does not bind the exact old "
            "primary/test bytes in one admitted path scope"
        )

    try:
        canonical_citation = normalize_corpus_identifier(expected_citation)
    except (TypeError, ValueError):
        canonical_citation = ""
    if not canonical_citation or payload.get("citation") != canonical_citation:
        issues.append("legacy generated ownership manifest citation is stale")

    primary_digest = canonical_files.get(expected_primary_path)
    if (
        primary_digest is None
        or payload.get("generated_output_sha256") != primary_digest
    ):
        issues.append("legacy generated ownership manifest output digest is stale")

    if not isinstance(model, str) or not model.strip():
        issues.append("legacy generated ownership manifest model is malformed")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        issues.append("legacy generated ownership manifest run id is malformed")
    generated_at = payload.get("generated_at")
    try:
        parsed_generated_at = (
            datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            if isinstance(generated_at, str)
            else None
        )
    except ValueError:
        parsed_generated_at = None
    if parsed_generated_at is None or parsed_generated_at.tzinfo is None:
        issues.append("legacy generated ownership manifest timestamp is malformed")

    version = payload.get("axiom_encode_version")
    git_identity = payload.get("axiom_encode_git")
    if (
        not isinstance(version, str)
        or not version.strip()
        or not isinstance(git_identity, dict)
        or set(git_identity) != LEGACY_GENERATED_GIT_FIELDS
        or not isinstance(git_identity.get("commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", str(git_identity.get("commit"))) is None
        or git_identity.get("dirty_tracked") is not False
        or not isinstance(git_identity.get("root"), str)
        or not git_identity.get("root")
        or git_identity.get("version") != version
        or not isinstance(git_identity.get("version_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", str(git_identity.get("version_commit")))
        is None
    ):
        issues.append(
            "legacy generated ownership manifest encoder identity is malformed"
        )

    for path_field, digest_field in (
        ("generated_output_file", "generated_output_sha256"),
        ("trace_file", "trace_sha256"),
        ("context_manifest_file", "context_manifest_sha256"),
    ):
        path_value = payload.get(path_field)
        digest_value = payload.get(digest_field)
        if (
            not isinstance(path_value, str)
            or not path_value.strip()
            or not isinstance(digest_value, str)
            or _SHA256.fullmatch(digest_value) is None
        ):
            issues.append(
                f"legacy generated ownership manifest {path_field} binding is malformed"
            )
    if (
        not isinstance(payload.get("generated_output_root"), str)
        or not payload.get("generated_output_root")
        or not isinstance(payload.get("generation_prompt_sha256"), str)
        or _SHA256.fullmatch(str(payload.get("generation_prompt_sha256"))) is None
    ):
        issues.append(
            "legacy generated ownership manifest generation binding is malformed"
        )
    return issues


def legacy_v1_manifest_issues(
    payload: object,
    *,
    expected_files: Mapping[str, str],
    expected_primary_path: str,
    expected_citation: str,
    jurisdiction_prefix: str,
    allow_unmarked_manual_exception: bool = False,
) -> list[str]:
    """Dispatch only the two explicitly supported historical v1 owner classes."""

    if not isinstance(payload, dict):
        return ["legacy ownership manifest is not a JSON object"]
    tool = payload.get("tool")
    backend = payload.get("backend")
    runner = payload.get("runner")
    if (
        tool == "axiom-encode sign-applied-files"
        or backend == "manual"
        or runner == "manual-attestation"
    ):
        return legacy_manual_manifest_issues(
            payload,
            expected_files=expected_files,
            allow_unmarked_manual_exception=allow_unmarked_manual_exception,
        )
    if tool == LEGACY_GENERATED_TOOL or (
        isinstance(backend, str) and backend in LEGACY_GENERATED_BACKENDS
    ):
        return legacy_generated_manifest_issues(
            payload,
            expected_files=expected_files,
            expected_primary_path=expected_primary_path,
            expected_citation=expected_citation,
            jurisdiction_prefix=jurisdiction_prefix,
        )
    return ["legacy ownership manifest class is unsupported"]


def legacy_receipt_v1_manifest_issues(
    payload: object,
    *,
    owner_class: object,
    expected_files: Mapping[str, str],
    expected_primary_path: str,
    expected_citation: str,
    jurisdiction_prefix: str,
    allow_unmarked_manual_exception: bool = False,
) -> list[str]:
    """Preserve old manual receipts without permitting owner-class relabeling."""

    if owner_class == LEGACY_MANUAL_OWNER_CLASS:
        return legacy_manual_manifest_issues(
            payload,
            expected_files=expected_files,
            allow_unmarked_manual_exception=allow_unmarked_manual_exception,
        )
    if owner_class == LEGACY_OWNER_CLASS:
        return legacy_v1_manifest_issues(
            payload,
            expected_files=expected_files,
            expected_primary_path=expected_primary_path,
            expected_citation=expected_citation,
            jurisdiction_prefix=jurisdiction_prefix,
            allow_unmarked_manual_exception=allow_unmarked_manual_exception,
        )
    return ["legacy receipt ownership class is unsupported"]


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
    exact_dependents: list[dict[str, object]] | None = None,
    destination_predecessor_class: str | None = None,
    destination_predecessor_files: list[dict[str, object]] | None = None,
    retained_successors: list[dict[str, object]] | None = None,
    metadata_reconciliations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    payload = {
        "base_commit": base_commit,
        "base_tree": base_tree,
        "legacy_manifest_sha256": legacy_manifest_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "live_files": live_files,
        "deleted_files": deleted_files,
        "rewrites": rewrites,
        "scheduled_dependents": scheduled_dependents,
    }
    if exact_dependents is not None:
        payload["exact_dependents"] = exact_dependents
    if destination_predecessor_class is not None:
        payload["destination_predecessor_class"] = destination_predecessor_class
    if destination_predecessor_files is not None:
        payload["destination_predecessor_files"] = destination_predecessor_files
    if retained_successors is not None:
        payload["retained_successors"] = retained_successors
    if metadata_reconciliations is not None:
        payload["metadata_reconciliations"] = metadata_reconciliations
    return payload


def receipt_identity_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
