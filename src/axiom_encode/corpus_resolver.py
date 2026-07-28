"""Release-aware, fail-closed resolution of normalized corpus provisions.

This module is deliberately dependency-light so generation, validation,
proof-checking, staleness checks, and oracle tooling can share one definition
of the exact corpus text and provenance used for an encoding.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import AbstractSet, Any, Literal

from axiom_encode.corpus_release import (
    CorpusReleaseObjectError,
    VerifiedReleaseArtifact,
    verify_release_object,
)
from axiom_encode.statute import citation_to_citation_path

MAX_CORPUS_PROVISION_BYTES = 64 * 1024 * 1024
MAX_LOCAL_CORPUS_AGGREGATE_BYTES = 512 * 1024 * 1024
MAX_LOCAL_CORPUS_FILES = 4_096
MAX_LOCAL_CORPUS_ROWS = 1_000_000
MAX_COMPOSED_CORPUS_BYTES = 64 * 1024 * 1024
MAX_COMPOSITION_NODES = 100_000
MAX_COMPOSITION_PREFIX_BYTES = 64 * 1024 * 1024
MAX_RELEASE_OBJECT_BYTES = 16 * 1024 * 1024
MAX_CORPUS_DESCENDANT_ROWS = 10_000
PROOF_EVIDENCE_SEGMENT_SEPARATOR = "\n\x1e\n"
MAX_CORPUS_CITATION_SEGMENT_LENGTH = 512
MAX_CORPUS_CITATION_SEGMENTS = 64
MAX_CORPUS_CITATION_LENGTH = 4_096
MAX_LEGAL_MARKER_TOKEN_LENGTH = 32
MAX_GENERIC_PARENT_SLICE_CHARACTER_WORK = 64 * 1024 * 1024
_MAX_INTERNAL_MARK_EVIDENCE_CHARS = 64
_MAX_DELIMITER_REPLAY_TOKENS = 1_000_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CITATION_JURISDICTION_RE = re.compile(r"^[a-z]{2,3}(?:-[a-z]{2,32})*$")
_CITATION_HIERARCHY_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .:\-–]*$")

_DOCUMENT_CLASSES = frozenset(
    {
        "form",
        "guidance",
        "district-plan",
        "legislation",
        "manual",
        "other",
        "policy",
        "regulation",
        "rulemaking",
        "statute",
    }
)


class CorpusResolutionError(ValueError):
    """Base class for corpus resolution failures."""


class InvalidCorpusCitationError(CorpusResolutionError):
    """A requested citation is not a normalized corpus citation path."""


class CorpusSourceNotFoundError(CorpusResolutionError):
    """No active corpus record exists for the requested citation."""


class InactiveCorpusSourceError(CorpusSourceNotFoundError):
    """The requested citation exists, but only outside the active release."""


class CorpusLayoutError(CorpusResolutionError):
    """A corpus checkout exists but its required provisions layout is malformed."""


class InvalidActiveCorpusSourceError(CorpusResolutionError):
    """An active corpus record exists but cannot produce a valid source body."""


class CorpusRowStructureError(CorpusResolutionError):
    """A matching corpus row is missing required scope or release metadata."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        super().__init__(detail)


class CorpusDescendantStructureError(InvalidActiveCorpusSourceError):
    """Active descendants are malformed, cross-scope, or cannot be composed."""


class AmbiguousCorpusSourceError(CorpusResolutionError):
    """More than one active corpus record claims the requested citation."""

    def __init__(self, citation_path: str, rows: tuple[CorpusRowIdentity, ...]):
        self.citation_path = citation_path
        self.rows = rows
        locations = ", ".join(f"{row.provision_file}:{row.line_number}" for row in rows)
        super().__init__(
            f"Ambiguous active corpus citation {citation_path!r}: {locations}"
        )


class InvalidCorpusReleaseError(CorpusResolutionError):
    """The configured corpus release is absent or violates its contract."""


class UnsafeCorpusPathError(CorpusResolutionError):
    """A corpus path is indirect, unbounded, or escapes the explicit root."""


class CorpusSourceSliceError(CorpusResolutionError):
    """A requested child could not be isolated from a parent provision."""


_CFR_IDENTIFIER_RE = re.compile(
    r"^(?P<title>\d+)\s+C\.?\s*F\.?\s*R\.?\s+"
    r"(?:(?:part|pt\.?)\s+)?(?:§+\s*)?"
    r"(?P<section>[0-9A-Za-z.-]+)"
    r"(?P<tail>(?:\([^)]+\))*)$",
    re.IGNORECASE,
)

_RULESPEC_DOCUMENT_CLASS = {
    "forms": "form",
    "guidance": "guidance",
    "manuals": "manual",
    "policies": "policy",
    "regulations": "regulation",
    "statutes": "statute",
}


def normalize_corpus_identifier(identifier: str) -> str:
    """Normalize one user/RuleSpec legal identifier for resolver-owned lookup.

    Parent enumeration is intentionally not exposed here.  The resolver's
    lookup groups are the sole implementation of aliases, parent fallback,
    slicing policy, and ambiguity handling.
    """

    value = identifier.strip().strip("/")
    if not value:
        raise InvalidCorpusCitationError("Corpus citation path must not be empty")
    parts = [part for part in value.split("/") if part]
    if parts and ":" in parts[0]:
        jurisdiction, source_root = parts[0].split(":", 1)
        document_class = _RULESPEC_DOCUMENT_CLASS.get(source_root, source_root)
        if jurisdiction and document_class:
            return _normalize_citation_path(
                "/".join((jurisdiction, document_class, *parts[1:]))
            )
    if len(parts) >= 2 and parts[1] in _DOCUMENT_CLASSES:
        return _normalize_citation_path(value)
    cfr = _CFR_IDENTIFIER_RE.fullmatch(value)
    if cfr is not None:
        tail = re.findall(r"\(([^)]+)\)", cfr.group("tail"))
        section = tuple(part for part in cfr.group("section").split(".") if part)
        return _normalize_citation_path(
            "/".join(("us", "regulation", cfr.group("title"), *section, *tail))
        )
    try:
        return _normalize_citation_path(citation_to_citation_path(value))
    except ValueError as exc:
        raise InvalidCorpusCitationError(
            f"Could not normalize corpus identifier {identifier!r}"
        ) from exc


def require_canonical_corpus_citation_path(identifier: str) -> str:
    """Return one exact canonical corpus citation path or reject aliases.

    Human-facing citation arguments may use :func:`normalize_corpus_identifier`.
    Persisted and explicitly named ``corpus_citation_path`` fields are machine
    identities, so accepting import-style identifiers or normalizing whitespace
    would retain a second spelling for the same source.
    """

    try:
        citation_path = _normalize_citation_path(identifier)
    except InvalidCorpusCitationError as exc:
        raise InvalidCorpusCitationError(
            "Corpus citation path must be an exact canonical "
            f"<jurisdiction>/<document-class>/... path: {identifier!r}"
        ) from exc
    if citation_path != identifier:
        raise InvalidCorpusCitationError(
            "Corpus citation path must be an exact canonical path; expected "
            f"{citation_path!r}, got {identifier!r}"
        )
    return citation_path


def scope_resolved_corpus_source(
    source: ResolvedCorpusSource, target_identifier: str
) -> ResolvedCorpusSource:
    """Fail-closed resolver-owned slicing for the exact generation target."""

    target = normalize_corpus_identifier(target_identifier)
    body = slice_corpus_source_text(
        source.body,
        target_identifier=target,
        resolved_identifier=source.citation_path,
    )
    return replace(
        source,
        requested=target,
        body=body,
        resolved_text_sha256=_sha256_text(body),
        source_history=source.source_history if target == source.citation_path else (),
        slice_required=target != source.citation_path,
    )


def resolve_scoped_local_corpus_source(
    source: ResolvedCorpusSource,
    normalized_target_identifier: str,
    release: LocalCorpusRelease,
) -> ResolvedCorpusSource:
    """Resolve an exact child before considering a slice of ``source``.

    This API deliberately accepts only normalized corpus paths so callers cannot
    accidentally give exact-child lookup and parent slicing different identities.
    """

    if ":" in normalized_target_identifier:
        raise InvalidCorpusCitationError(
            "Scoped corpus target must be a normalized corpus citation path"
        )
    target = _normalize_citation_path(normalized_target_identifier)
    if target != normalized_target_identifier:
        raise InvalidCorpusCitationError(
            "Scoped corpus target must be a normalized corpus citation path"
        )
    if (
        source.release_name != release.name
        or source.release_content_sha256 != release.content_sha256
    ):
        raise InvalidCorpusReleaseError(
            "Scoped source and local corpus release identities do not match"
        )
    try:
        exact = resolve_local_corpus_source(
            target,
            release,
            _exact_only=True,
        )
    except CorpusSourceNotFoundError:
        return scope_resolved_corpus_source(source, target)
    if exact.row != source.row:
        raise AmbiguousCorpusSourceError(target, (exact.row, source.row))
    return exact


def slice_corpus_source_text(
    body: str, *, target_identifier: str, resolved_identifier: str | None = None
) -> str:
    """Slice exact target text from a resolver-selected parent body."""

    target_path = normalize_corpus_identifier(target_identifier)
    if resolved_identifier is None:
        parts = target_path.split("/")
        if parts[:2] == ["us", "regulation"] and len(parts) >= 5:
            resolved_path = "/".join(parts[:5])
        elif parts[:2] == ["us", "statute"] and len(parts) >= 4:
            resolved_path = "/".join(parts[:4])
        else:
            resolved_path = target_path
    else:
        resolved_path = normalize_corpus_identifier(resolved_identifier)
    if target_path == resolved_path:
        scoped = body
    else:
        scoped = _slice_parent_body(
            body,
            requested_path=target_path,
            resolved_path=resolved_path,
        )
    if not scoped.strip():
        raise CorpusSourceSliceError(
            f"Resolver produced an empty source slice for {target_identifier!r}"
        )
    return scoped


@dataclass(frozen=True, order=True)
class ReleaseScope:
    """One active ``jurisdiction x document class x version`` scope."""

    jurisdiction: str
    document_class: str
    version: str


@dataclass(frozen=True)
class LocalCorpusRelease:
    """One local checkout bound to one verified immutable release object."""

    root: Path
    name: str
    content_sha256: str
    public_key: str | Sequence[str] = field(repr=False, compare=False)
    provisions_root: Path = field(init=False)
    selector_sha256: str = field(init=False)
    scopes: tuple[ReleaseScope, ...] = field(init=False)
    artifacts: tuple[VerifiedReleaseArtifact, ...] = field(init=False)
    release_object_path: Path = field(init=False)

    def __post_init__(self) -> None:
        name = validate_corpus_release_name(self.name)
        if _SHA256_RE.fullmatch(self.content_sha256) is None:
            raise InvalidCorpusReleaseError(
                "Corpus release content_sha256 must be a lowercase sha256 digest"
            )
        root, provisions_root, repository_root = _resolve_corpus_layout(self.root)
        if repository_root != root:
            raise CorpusLayoutError(
                "Local corpus releases require the canonical repository root"
            )
        release_object_path = root / "releases" / name / f"{self.content_sha256}.json"
        release_object_file = _safe_file(
            root,
            release_object_path,
            label="corpus release object",
            max_bytes=MAX_RELEASE_OBJECT_BYTES,
        )
        if release_object_file is None:
            raise InvalidCorpusReleaseError(
                f"Corpus release object not found: {release_object_path}"
            )
        try:
            raw = read_bounded_regular_file(
                root,
                release_object_file,
                label="corpus release object",
                max_bytes=MAX_RELEASE_OBJECT_BYTES,
            )
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise InvalidCorpusReleaseError(
                f"Corpus release object is not valid UTF-8 JSON: {release_object_file}"
            ) from exc
        if not isinstance(payload, dict):
            raise InvalidCorpusReleaseError(
                "Corpus release object must be a JSON object"
            )
        try:
            verified = verify_release_object(payload, public_key=self.public_key)
        except CorpusReleaseObjectError as exc:
            raise InvalidCorpusReleaseError(str(exc)) from exc
        if verified.name != name or verified.content_sha256 != self.content_sha256:
            raise InvalidCorpusReleaseError(
                "Corpus release object does not match its configured name and content digest"
            )
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "provisions_root", provisions_root)
        object.__setattr__(self, "selector_sha256", verified.selector_sha256)
        object.__setattr__(
            self,
            "scopes",
            tuple(
                ReleaseScope(
                    scope.jurisdiction,
                    scope.document_class,
                    scope.version,
                )
                for scope in verified.scopes
            ),
        )
        object.__setattr__(self, "artifacts", verified.artifacts)
        object.__setattr__(self, "release_object_path", release_object_file)


@dataclass(frozen=True)
class CorpusRowIdentity:
    """Stable location and source metadata for one stored provision row."""

    provision_file: str
    provision_file_sha256: str
    line_number: int
    record_id: str
    citation_path: str
    jurisdiction: str
    document_class: str
    version: str
    source_path: str | None
    source_as_of: str | None
    expression_date: str | None
    body_sha256: str | None


@dataclass(frozen=True)
class ActiveCorpusBodyRow:
    """One body-bearing row selected by the active corpus release."""

    row: CorpusRowIdentity
    body: str
    heading: str | None
    citation_label: str | None
    metadata: Mapping[str, Any]
    release_name: str
    release_content_sha256: str
    release_selector_sha256: str


@dataclass(frozen=True)
class ResolvedCorpusSource:
    """Exact source text plus the release, artifact, and row that supplied it."""

    requested: str
    citation_path: str
    body: str
    stored_body_sha256: str
    resolved_text_sha256: str
    source: Literal["local"]
    provision_file: str
    provision_file_sha256: str
    row: CorpusRowIdentity
    component_rows: tuple[CorpusRowIdentity, ...]
    release_name: str
    release_content_sha256: str
    release_selector_sha256: str
    source_history: tuple[str, ...] = ()
    slice_required: bool = False

    @property
    def proof_evidence_segments(self) -> tuple[str, ...]:
        """Return independently matchable, release-bound proof evidence."""

        segments = (self.body, *self.source_history)
        if any(PROOF_EVIDENCE_SEGMENT_SEPARATOR in segment for segment in segments):
            raise CorpusResolutionError(
                "Corpus proof evidence contains the reserved segment separator"
            )
        return segments

    @property
    def proof_evidence_text(self) -> str:
        """Return segmented release-bound text available to proof validation."""

        return PROOF_EVIDENCE_SEGMENT_SEPARATOR.join(self.proof_evidence_segments)

    def to_attestation(self) -> dict[str, Any]:
        """Return the JSON-compatible provenance bound into apply manifests."""

        return {
            "requested_corpus_citation_path": self.requested,
            "resolved_corpus_citation_path": self.citation_path,
            "corpus_source": self.source,
            "corpus_release": self.release_name,
            "corpus_release_content_sha256": self.release_content_sha256,
            "corpus_release_selector_sha256": self.release_selector_sha256,
            "provision_file": self.provision_file,
            "provision_file_sha256": self.provision_file_sha256,
            "row": asdict(self.row),
            "component_rows": [asdict(row) for row in self.component_rows],
            "source_sha256": self.stored_body_sha256,
            "resolved_text_sha256": self.resolved_text_sha256,
            "source_as_of": self.row.source_as_of,
            "expression_date": self.row.expression_date,
        }


@dataclass(frozen=True)
class _StoredRecord:
    row: CorpusRowIdentity
    body: str | None
    heading: str | None
    status: str | None
    level: int | None
    ordinal: int | None
    file_path: Path
    file_sha256: str
    source_history: tuple[str, ...]


@dataclass(frozen=True)
class _LookupCandidate:
    citation_path: str
    requested_path: str
    slice_required: bool


@dataclass(frozen=True)
class _ParsedCorpusArtifact:
    """One immutable read of a JSONL artifact and its parsed rows."""

    path: Path
    sha256: str
    byte_size: int
    rows: tuple[tuple[int, dict[str, Any]], ...]


@dataclass
class _LocalCorpusReadBudget:
    file_count: int = 0
    byte_count: int = 0
    row_count: int = 0


@dataclass
class _ParentheticalSliceBudget:
    """Cumulative character work for generic parent-body fallback slicing."""

    character_work: int = 0

    def charge(self, character_count: int) -> None:
        if character_count < 0:
            raise AssertionError("Parenthetical slice work cannot be negative")
        if (
            character_count
            > MAX_GENERIC_PARENT_SLICE_CHARACTER_WORK - self.character_work
        ):
            raise CorpusSourceSliceError(
                "Generic parent slicing exceeds the cumulative character-work "
                "safety limit of "
                f"{MAX_GENERIC_PARENT_SLICE_CHARACTER_WORK}"
            )
        self.character_work += character_count


_DelimiterFrame = tuple[str, str, int, str | None]


class _IndexedDelimiterStack(list[_DelimiterFrame]):
    """A stack that forbids accidental depth-proportional scans.

    Delimiter replay indexes aggregate frame state separately. Keeping this
    container deliberately non-iterable turns a future ``sum``/``any``/linear
    stack walk back into an immediate implementation error instead of silently
    reintroducing a depth multiplier for every source character.
    """

    def __iter__(self) -> Iterator[_DelimiterFrame]:
        raise AssertionError("Delimiter replay stack must not be scanned")

    def __reversed__(self) -> Iterator[_DelimiterFrame]:
        raise AssertionError("Delimiter replay stack must not be scanned")


def resolve_local_corpus_source(
    identifier: str,
    release: LocalCorpusRelease,
    *,
    _exact_only: bool = False,
) -> ResolvedCorpusSource:
    """Resolve exactly one active local provision row, or fail closed."""

    citation_path = _normalize_citation_path(identifier)
    if not isinstance(release, LocalCorpusRelease):
        raise TypeError("release must be a validated LocalCorpusRelease")
    provisions_root = release.provisions_root

    parts = citation_path.split("/")
    jurisdiction, document_class = parts[0], parts[1]
    scopes = tuple(
        scope
        for scope in release.scopes
        if scope.jurisdiction == jurisdiction and scope.document_class == document_class
    )

    files = _candidate_provision_files(
        release=release,
        jurisdiction=jurisdiction,
        document_class=document_class,
        scopes=scopes,
    )
    artifacts = _read_corpus_artifacts(
        files,
        release=release,
        budget=_LocalCorpusReadBudget(),
    )
    active_scopes = frozenset(scopes)
    inactive_match = False
    lookup_groups = _citation_lookup_groups(citation_path)
    if _exact_only:
        lookup_groups = lookup_groups[:1]
    for group in lookup_groups:
        records: list[_StoredRecord] = []
        by_path = {candidate.citation_path: candidate for candidate in group}
        for artifact in artifacts:
            for candidate in group:
                records.extend(
                    _read_matching_records(
                        artifact,
                        citation_path=candidate.citation_path,
                        active_scopes=active_scopes,
                        provisions_root=provisions_root,
                    )
                )
        if not records:
            inactive_match = inactive_match or any(
                record.get("citation_path") == candidate.citation_path
                for artifact in artifacts
                for _line, record in artifact.rows
                for candidate in group
            )
            continue
        if len(records) > 1:
            raise AmbiguousCorpusSourceError(
                citation_path, tuple(record.row for record in records)
            )
        selected = records[0]
        candidate = by_path[selected.row.citation_path]
        if selected.status == "repealed":
            raise InvalidActiveCorpusSourceError(
                f"Active corpus source {selected.row.citation_path!r} is repealed"
            )
        component_rows: tuple[CorpusRowIdentity, ...] = ()
        stored_body = selected.body
        if stored_body is None:
            selected_scope = ReleaseScope(
                selected.row.jurisdiction,
                selected.row.document_class,
                selected.row.version,
            )
            descendants: list[_StoredRecord] = []
            for artifact in artifacts:
                descendants.extend(
                    _read_descendant_records(
                        artifact,
                        citation_path=selected.row.citation_path,
                        active_scopes=active_scopes,
                        provisions_root=provisions_root,
                    )
                )
                if len(descendants) > MAX_CORPUS_DESCENDANT_ROWS:
                    raise CorpusDescendantStructureError(
                        f"Local descendants for {selected.row.citation_path!r} "
                        f"exceed the {MAX_CORPUS_DESCENDANT_ROWS}-row safety limit"
                    )
            stored_body, component_rows = _compose_descendant_text(
                selected.row.citation_path,
                descendants,
                expected_scope=selected_scope,
            )
        body = stored_body
        if candidate.slice_required:
            body = _slice_parent_body(
                stored_body,
                requested_path=candidate.requested_path,
                resolved_path=candidate.citation_path,
            )
        return _resolved_source(
            requested=identifier,
            citation_path=selected.row.citation_path,
            body=body,
            stored_body=stored_body,
            selected=selected,
            release=release,
            component_rows=component_rows,
            slice_required=candidate.slice_required,
        )
    if inactive_match:
        raise InactiveCorpusSourceError(
            f"Corpus source exists but is inactive for {citation_path!r}"
        )
    raise CorpusSourceNotFoundError(
        f"No active corpus source found for {citation_path!r}"
    )


def resolve_local_corpus_dependency_artifacts(
    identifier: str,
    release: LocalCorpusRelease,
) -> tuple[Path, ...]:
    """Resolve one dependency source and return its attested local artifacts.

    This is the dependency-check API: alias handling, statutes/regulations-only
    parent fallback, slicing, and ambiguity are decided once by the resolver.
    """

    source = resolve_local_corpus_source(
        identifier,
        release,
    )
    root = release.root
    repository_root = release.root
    artifacts: list[Path] = []
    seen: set[Path] = set()
    for row in (source.row, *source.component_rows):
        if not row.provision_file_sha256:
            raise CorpusResolutionError(
                f"Resolved local corpus row lacks artifact digest: {row.provision_file}"
            )
        relative = Path(row.provision_file)
        if relative.is_absolute():
            raise UnsafeCorpusPathError(
                f"Resolved corpus artifact path is absolute: {row.provision_file}"
            )
        artifact = _safe_file(
            root,
            repository_root / relative,
            label="resolved corpus provision",
            max_bytes=MAX_CORPUS_PROVISION_BYTES,
        )
        if artifact is None:
            raise CorpusResolutionError(
                f"Resolved corpus artifact changed after resolution: {row.provision_file}"
            )
        artifact_bytes = read_bounded_regular_file(
            root,
            artifact,
            label="resolved corpus provision",
            max_bytes=MAX_CORPUS_PROVISION_BYTES,
        )
        if hashlib.sha256(artifact_bytes).hexdigest() != row.provision_file_sha256:
            raise CorpusResolutionError(
                f"Resolved corpus artifact changed after resolution: {row.provision_file}"
            )
        if artifact not in seen:
            seen.add(artifact)
            artifacts.append(artifact)
    return tuple(artifacts)


def iter_active_local_corpus_rows(
    release: LocalCorpusRelease,
    *,
    jurisdiction: str | None = None,
    document_class: str | None = None,
) -> Iterator[ActiveCorpusBodyRow]:
    """Iterate body-bearing rows in an exact, validated active release.

    The complete selected scope is validated before the iterator is returned,
    so callers never observe partial results before a duplicate citation or
    malformed active row fails closed.
    """

    if jurisdiction is not None:
        jurisdiction = _clean_string_value(
            jurisdiction, label="corpus jurisdiction filter"
        )
        _validate_safe_segment(jurisdiction, label="corpus jurisdiction filter")
    if document_class is not None and document_class not in _DOCUMENT_CLASSES:
        raise CorpusResolutionError(
            f"Unsupported corpus document class filter {document_class!r}"
        )

    if not isinstance(release, LocalCorpusRelease):
        raise TypeError("release must be a validated LocalCorpusRelease")
    provisions_root = release.provisions_root
    selected_scopes = tuple(
        scope
        for scope in release.scopes
        if (jurisdiction is None or scope.jurisdiction == jurisdiction)
        and (document_class is None or scope.document_class == document_class)
    )
    scopes_by_bucket: dict[tuple[str, str], list[ReleaseScope]] = {}
    for scope in selected_scopes:
        scopes_by_bucket.setdefault(
            (scope.jurisdiction, scope.document_class), []
        ).append(scope)

    body_rows: list[ActiveCorpusBodyRow] = []
    identities_by_citation: dict[str, list[CorpusRowIdentity]] = {}
    read_budget = _LocalCorpusReadBudget()
    for (scope_jurisdiction, scope_document_class), bucket_scopes in sorted(
        scopes_by_bucket.items()
    ):
        active_scopes = frozenset(bucket_scopes)
        files = _candidate_provision_files(
            release=release,
            jurisdiction=scope_jurisdiction,
            document_class=scope_document_class,
            scopes=tuple(bucket_scopes),
        )
        for artifact in _read_corpus_artifacts(
            files,
            release=release,
            budget=read_budget,
        ):
            for line_number, record in artifact.rows:
                row_scope = _record_scope(
                    record,
                    path=artifact.path,
                    line_number=line_number,
                )
                if row_scope not in active_scopes:
                    continue
                if (
                    row_scope.jurisdiction != scope_jurisdiction
                    or row_scope.document_class != scope_document_class
                ):
                    raise CorpusResolutionError(
                        f"Active corpus row is stored in the wrong scope bucket: "
                        f"{artifact.path}:{line_number}"
                    )
                _validate_release_row_metadata(
                    record, path=artifact.path, line_number=line_number
                )
                raw_citation_path = _clean_string_value(
                    record.get("citation_path"),
                    label=f"citation_path at {artifact.path}:{line_number}",
                )
                citation_path = _normalize_citation_path(raw_citation_path)
                if citation_path != raw_citation_path:
                    raise CorpusResolutionError(
                        "citation_path must be normalized at "
                        f"{artifact.path}:{line_number}"
                    )
                citation_parts = citation_path.split("/")
                if citation_parts[:2] != [
                    row_scope.jurisdiction,
                    row_scope.document_class,
                ]:
                    raise CorpusResolutionError(
                        f"Active corpus citation does not match its scope at "
                        f"{artifact.path}:{line_number}"
                    )
                body = _record_body(
                    record,
                    path=artifact.path,
                    line_number=line_number,
                )
                artifact_root = _repository_root_for_provisions(provisions_root)
                relative_file = artifact.path.relative_to(artifact_root).as_posix()
                identity = CorpusRowIdentity(
                    provision_file=relative_file,
                    provision_file_sha256=artifact.sha256,
                    line_number=line_number,
                    record_id=str(record["id"]),
                    citation_path=citation_path,
                    jurisdiction=row_scope.jurisdiction,
                    document_class=row_scope.document_class,
                    version=row_scope.version,
                    source_path=str(record["source_path"]),
                    source_as_of=str(record["source_as_of"]),
                    expression_date=str(record["expression_date"]),
                    body_sha256=_sha256_text(body) if body is not None else None,
                )
                identities_by_citation.setdefault(citation_path, []).append(identity)
                if body is None:
                    continue
                raw_metadata = record.get("metadata")
                metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
                body_rows.append(
                    ActiveCorpusBodyRow(
                        row=identity,
                        body=body,
                        heading=_optional_string(record.get("heading")),
                        citation_label=_optional_string(record.get("citation_label")),
                        metadata=MappingProxyType(dict(metadata)),
                        release_name=release.name,
                        release_content_sha256=release.content_sha256,
                        release_selector_sha256=release.selector_sha256,
                    )
                )

    ambiguous = next(
        (
            (citation_path, identities)
            for citation_path, identities in sorted(identities_by_citation.items())
            if len(identities) > 1
        ),
        None,
    )
    if ambiguous is not None:
        citation_path, identities = ambiguous
        raise AmbiguousCorpusSourceError(citation_path, tuple(identities))
    body_rows.sort(
        key=lambda record: (
            record.row.citation_path,
            record.row.provision_file,
            record.row.line_number,
        )
    )
    return iter(tuple(body_rows))


def _resolved_source(
    *,
    requested: str,
    citation_path: str,
    body: str,
    stored_body: str,
    selected: _StoredRecord,
    release: LocalCorpusRelease,
    component_rows: tuple[CorpusRowIdentity, ...] = (),
    slice_required: bool = False,
) -> ResolvedCorpusSource:
    return ResolvedCorpusSource(
        requested=requested,
        citation_path=citation_path,
        body=body,
        stored_body_sha256=_sha256_text(stored_body),
        resolved_text_sha256=_sha256_text(body),
        source="local",
        provision_file=selected.row.provision_file,
        provision_file_sha256=selected.file_sha256,
        row=selected.row,
        component_rows=component_rows,
        release_name=release.name,
        release_content_sha256=release.content_sha256,
        release_selector_sha256=release.selector_sha256,
        source_history=() if slice_required else selected.source_history,
        slice_required=slice_required,
    )


def _resolve_corpus_layout(corpus_root: Path) -> tuple[Path, Path, Path]:
    raw_root = Path(os.path.abspath(Path(corpus_root).expanduser()))
    if raw_root.is_symlink():
        raise UnsafeCorpusPathError(f"corpus root is a symlink: {raw_root}")
    try:
        root = raw_root.resolve(strict=True)
    except OSError as exc:
        raise CorpusLayoutError(f"corpus root does not exist: {raw_root}") from exc
    if not root.is_dir():
        raise UnsafeCorpusPathError(f"corpus root is not a directory: {raw_root}")
    provisions_root = _safe_directory(
        root,
        root / "data" / "corpus" / "provisions",
        label="corpus provisions root",
    )
    if provisions_root is None:
        raise CorpusLayoutError(
            f"Canonical data/corpus/provisions directory not found under {root}"
        )
    provisions_root = provisions_root.resolve(strict=True)
    return root, provisions_root, root


def _candidate_provision_files(
    *,
    release: LocalCorpusRelease,
    jurisdiction: str,
    document_class: str,
    scopes: tuple[ReleaseScope, ...],
) -> tuple[Path, ...]:
    if not scopes:
        return ()
    versions = {scope.version for scope in scopes}
    candidates: list[Path] = []
    for artifact in release.artifacts:
        if artifact.artifact_class != "provisions":
            continue
        parts = artifact.path.split("/")
        if (
            len(parts) != 6
            or parts[:3] != ["data", "corpus", "provisions"]
            or parts[3] != jurisdiction
            or parts[4] != document_class
            or not parts[5].endswith(".jsonl")
            or parts[5].removesuffix(".jsonl") not in versions
        ):
            continue
        candidate = _safe_file(
            release.root,
            release.root / artifact.path,
            label="corpus provision file",
            max_bytes=MAX_CORPUS_PROVISION_BYTES,
        )
        if candidate is None:
            raise CorpusResolutionError(
                f"Verified corpus release artifact is missing: {artifact.path}"
            )
        candidates.append(candidate)
    if len(candidates) != len(scopes):
        raise CorpusResolutionError(
            "Verified corpus release does not provide exactly one provisions "
            f"artifact per requested scope: {jurisdiction}/{document_class}"
        )
    return tuple(candidates)


def _read_corpus_artifact(
    path: Path,
    *,
    release: LocalCorpusRelease,
) -> _ParsedCorpusArtifact:
    """Read, hash, decode, and parse an artifact from one byte snapshot."""

    raw = read_bounded_regular_file(
        release.root,
        path,
        label="corpus provision file",
        max_bytes=MAX_CORPUS_PROVISION_BYTES,
    )
    file_sha256 = hashlib.sha256(raw).hexdigest()
    relative_path = path.relative_to(release.root).as_posix()
    expected = next(
        (artifact for artifact in release.artifacts if artifact.path == relative_path),
        None,
    )
    if expected is None or expected.artifact_class != "provisions":
        raise CorpusResolutionError(
            f"Corpus provision is outside the verified release inventory: {relative_path}"
        )
    if expected.sha256 != file_sha256 or expected.byte_count != len(raw):
        raise CorpusResolutionError(
            f"Corpus provision bytes do not match the verified release: {relative_path}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise CorpusResolutionError(
            f"Corpus artifact is not valid UTF-8: {path}"
        ) from exc
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusResolutionError(
                f"Invalid JSON in corpus artifact {path}:{line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise CorpusResolutionError(
                f"Corpus row must be an object: {path}:{line_number}"
            )
        rows.append((line_number, record))
        if len(rows) > MAX_LOCAL_CORPUS_ROWS:
            raise CorpusResolutionError(
                f"Corpus artifact exceeds the {MAX_LOCAL_CORPUS_ROWS}-row "
                f"safety limit: {path}"
            )
    return _ParsedCorpusArtifact(
        path=path,
        sha256=file_sha256,
        byte_size=len(raw),
        rows=tuple(rows),
    )


def _read_corpus_artifacts(
    paths: tuple[Path, ...],
    *,
    release: LocalCorpusRelease,
    budget: _LocalCorpusReadBudget,
) -> tuple[_ParsedCorpusArtifact, ...]:
    if budget.file_count + len(paths) > MAX_LOCAL_CORPUS_FILES:
        raise CorpusResolutionError(
            "Local corpus resolution exceeds the "
            f"{MAX_LOCAL_CORPUS_FILES}-file safety limit"
        )
    artifacts: list[_ParsedCorpusArtifact] = []
    for path in paths:
        artifact = _read_corpus_artifact(path, release=release)
        budget.file_count += 1
        budget.byte_count += artifact.byte_size
        budget.row_count += len(artifact.rows)
        if budget.byte_count > MAX_LOCAL_CORPUS_AGGREGATE_BYTES:
            raise CorpusResolutionError(
                "Local corpus resolution exceeds the "
                f"{MAX_LOCAL_CORPUS_AGGREGATE_BYTES}-byte aggregate safety limit"
            )
        if budget.row_count > MAX_LOCAL_CORPUS_ROWS:
            raise CorpusResolutionError(
                "Local corpus resolution exceeds the "
                f"{MAX_LOCAL_CORPUS_ROWS}-row aggregate safety limit"
            )
        artifacts.append(artifact)
    return tuple(artifacts)


def _read_matching_records(
    artifact: _ParsedCorpusArtifact,
    *,
    citation_path: str,
    active_scopes: frozenset[ReleaseScope],
    provisions_root: Path,
) -> list[_StoredRecord]:
    path = artifact.path
    records: list[_StoredRecord] = []
    for line_number, record in artifact.rows:
        if record.get("citation_path") != citation_path:
            continue
        citation_parts = citation_path.split("/")
        row_scope = _record_scope(
            record,
            path=path,
            line_number=line_number,
        )
        if row_scope not in active_scopes:
            continue
        _validate_release_row_metadata(record, path=path, line_number=line_number)
        if citation_parts[:2] != [
            row_scope.jurisdiction,
            row_scope.document_class,
        ]:
            raise CorpusResolutionError(
                f"Active corpus citation does not match its scope at "
                f"{path}:{line_number}"
            )
        body = _record_body(record, path=path, line_number=line_number)
        artifact_root = _repository_root_for_provisions(provisions_root)
        relative_file = path.relative_to(artifact_root).as_posix()
        row = CorpusRowIdentity(
            provision_file=relative_file,
            provision_file_sha256=artifact.sha256,
            line_number=line_number,
            record_id=str(record.get("id") or ""),
            citation_path=citation_path,
            jurisdiction=row_scope.jurisdiction,
            document_class=row_scope.document_class,
            version=row_scope.version,
            source_path=_optional_string(record.get("source_path")),
            source_as_of=_optional_string(record.get("source_as_of")),
            expression_date=_optional_string(record.get("expression_date")),
            body_sha256=_sha256_text(body) if body is not None else None,
        )
        records.append(
            _StoredRecord(
                row=row,
                body=body,
                heading=_optional_string(record.get("heading")),
                status=_record_status(record),
                level=_optional_int(record.get("level")),
                ordinal=_optional_int(record.get("ordinal")),
                file_path=path,
                file_sha256=artifact.sha256,
                source_history=_record_source_history(record),
            )
        )
    return records


def _read_descendant_records(
    artifact: _ParsedCorpusArtifact,
    *,
    citation_path: str,
    active_scopes: frozenset[ReleaseScope],
    provisions_root: Path,
) -> list[_StoredRecord]:
    path = artifact.path
    prefix = f"{citation_path}/"
    records: list[_StoredRecord] = []
    for line_number, record in artifact.rows:
        record_path = record.get("citation_path")
        if not isinstance(record_path, str) or not record_path.startswith(prefix):
            continue
        row_scope = _record_scope(
            record,
            path=path,
            line_number=line_number,
        )
        if row_scope not in active_scopes:
            continue
        try:
            normalized_record_path = _normalize_citation_path(record_path)
        except InvalidCorpusCitationError as exc:
            raise CorpusDescendantStructureError(
                f"Invalid descendant citation_path at {path}:{line_number}: "
                f"{record_path!r}"
            ) from exc
        if normalized_record_path != record_path:
            raise CorpusDescendantStructureError(
                f"citation_path must be normalized at {path}:{line_number}"
            )
        citation_parts = record_path.split("/")
        if citation_parts[:2] != [
            row_scope.jurisdiction,
            row_scope.document_class,
        ]:
            raise CorpusDescendantStructureError(
                f"Active corpus descendant does not match its scope at "
                f"{path}:{line_number}"
            )
        _validate_release_row_metadata(record, path=path, line_number=line_number)
        body = _record_body(record, path=path, line_number=line_number)
        artifact_root = _repository_root_for_provisions(provisions_root)
        relative_file = path.relative_to(artifact_root).as_posix()
        row = CorpusRowIdentity(
            provision_file=relative_file,
            provision_file_sha256=artifact.sha256,
            line_number=line_number,
            record_id=str(record.get("id") or ""),
            citation_path=record_path,
            jurisdiction=row_scope.jurisdiction,
            document_class=row_scope.document_class,
            version=row_scope.version,
            source_path=_optional_string(record.get("source_path")),
            source_as_of=_optional_string(record.get("source_as_of")),
            expression_date=_optional_string(record.get("expression_date")),
            body_sha256=_sha256_text(body) if body is not None else None,
        )
        try:
            level = _optional_int(record.get("level"))
            ordinal = _optional_int(record.get("ordinal"))
        except CorpusResolutionError as exc:
            raise CorpusDescendantStructureError(
                f"Malformed descendant ordering metadata at {path}:{line_number}: {exc}"
            ) from exc
        records.append(
            _StoredRecord(
                row=row,
                body=body,
                heading=_optional_string(record.get("heading")),
                status=_record_status(record),
                level=level,
                ordinal=ordinal,
                file_path=path,
                file_sha256=artifact.sha256,
                source_history=_record_source_history(record),
            )
        )
    return records


def _compose_descendant_text(
    citation_path: str,
    records: list[_StoredRecord],
    *,
    expected_scope: ReleaseScope,
) -> tuple[str, tuple[CorpusRowIdentity, ...]]:
    if not records:
        raise CorpusDescendantStructureError(
            f"Active corpus source {citation_path!r} has no body-bearing descendants"
        )
    by_path: dict[str, list[_StoredRecord]] = {}
    for record in records:
        record_scope = ReleaseScope(
            record.row.jurisdiction,
            record.row.document_class,
            record.row.version,
        )
        if record_scope != expected_scope:
            raise CorpusDescendantStructureError(
                f"Corpus descendants for {citation_path!r} cross active release scopes"
            )
        by_path.setdefault(record.row.citation_path, []).append(record)
    ambiguous = next((items for items in by_path.values() if len(items) > 1), None)
    if ambiguous is not None:
        raise AmbiguousCorpusSourceError(
            ambiguous[0].row.citation_path,
            tuple(record.row for record in ambiguous),
        )
    if not any(record.body is not None for record in records):
        raise CorpusDescendantStructureError(
            f"Active corpus source {citation_path!r} has no body-bearing descendants"
        )
    unique_by_path = {path: items[0] for path, items in by_path.items()}
    children: dict[str, set[str]] = {}
    prefix = f"{citation_path}/"
    composition_nodes = 1
    composition_prefix_bytes = 4 * len(citation_path.encode("utf-8"))
    for path in unique_by_path:
        if not path.startswith(prefix):
            raise CorpusDescendantStructureError(
                f"Corpus descendant {path!r} is outside parent {citation_path!r}"
            )
        parent = citation_path
        for segment in path[len(prefix) :].split("/"):
            child = f"{parent}/{segment}"
            child_paths = children.setdefault(parent, set())
            if child not in child_paths:
                composition_nodes += 1
                composition_prefix_bytes += 4 * len(child.encode("utf-8"))
                if composition_nodes > MAX_COMPOSITION_NODES:
                    raise CorpusDescendantStructureError(
                        "Corpus descendant composition exceeds the "
                        f"{MAX_COMPOSITION_NODES}-node safety limit"
                    )
                if composition_prefix_bytes > MAX_COMPOSITION_PREFIX_BYTES:
                    raise CorpusDescendantStructureError(
                        "Corpus descendant composition exceeds the "
                        f"{MAX_COMPOSITION_PREFIX_BYTES}-byte prefix safety limit"
                    )
                child_paths.add(child)
            parent = child

    for record in records:
        if record.status != "repealed":
            continue
        if record.body is not None or children.get(record.row.citation_path):
            raise CorpusDescendantStructureError(
                f"Active corpus source {citation_path!r} has malformed "
                f"repealed descendant {record.row.citation_path!r}"
            )

    ordered: list[_StoredRecord] = []
    contributes_cache: dict[str, bool] = {}

    def contributes_current_text(path: str) -> bool:
        cached = contributes_cache.get(path)
        if cached is not None:
            return cached
        record = unique_by_path.get(path)
        if record is not None and record.status == "repealed":
            contributes_cache[path] = False
            return False
        if record is not None and record.body is not None:
            contributes_cache[path] = True
            return True
        child_paths = children.get(path, set())
        if not child_paths:
            # Preserve the existing fail-closed visit for ordinary bodyless leaves.
            contributes_cache[path] = True
            return True
        contributes = any(contributes_current_text(child) for child in child_paths)
        contributes_cache[path] = contributes
        return contributes

    def visit(path: str) -> None:
        record = unique_by_path.get(path)
        if record is not None and record.status == "repealed":
            # A repealed bodyless leaf is an explicit current-law tombstone,
            # not a missing branch of the parent source. It contributes no text.
            return
        if record is not None and record.body is not None:
            # A body-bearing provision is the shallowest complete source for
            # its subtree. Including represented children as well would
            # duplicate legal text and alter the attested hash.
            ordered.append(record)
            return
        raw_child_paths = children.get(path, set())
        child_paths = {
            child for child in raw_child_paths if contributes_current_text(child)
        }
        if raw_child_paths and not child_paths:
            return
        if not child_paths:
            raise CorpusDescendantStructureError(
                f"Active corpus source {citation_path!r} has uncovered "
                f"bodyless descendant {path!r}"
            )
        child_records = [unique_by_path.get(child) for child in child_paths]
        use_ordinals = bool(child_paths) and all(
            child_record is not None and child_record.ordinal is not None
            for child_record in child_records
        )
        for child in sorted(
            child_paths,
            key=lambda item: _descendant_sibling_order_key(
                item,
                unique_by_path.get(item),
                use_ordinal=use_ordinals,
            ),
        ):
            visit(child)

    visit(citation_path)
    if not ordered:
        raise CorpusDescendantStructureError(
            f"Active corpus source {citation_path!r} has no body-bearing descendants"
        )
    chunks: list[str] = []
    composed_bytes = 0
    for record in ordered:
        chunk = (
            f"{record.heading}\n\n{record.body}"
            if record.heading
            else record.body or ""
        )
        composed_bytes += len(chunk.encode("utf-8"))
        if chunks:
            composed_bytes += 2
        if composed_bytes > MAX_COMPOSED_CORPUS_BYTES:
            raise CorpusDescendantStructureError(
                "Composed corpus source exceeds the "
                f"{MAX_COMPOSED_CORPUS_BYTES}-byte safety limit"
            )
        chunks.append(chunk)
    return "\n\n".join(chunks), tuple(record.row for record in ordered)


def _descendant_sibling_order_key(
    path: str,
    record: _StoredRecord | None,
    *,
    use_ordinal: bool,
) -> tuple[object, ...]:
    segment = path.rsplit("/", 1)[-1]
    natural = tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold(), part)
        for part in re.split(r"(\d+)", segment)
        if part
    )
    legal_ordinal = _canonical_us_descendant_ordinal(path)
    legal_order = (0, legal_ordinal) if legal_ordinal is not None else (1, 0)
    if use_ordinal and record is not None and record.ordinal is not None:
        return (record.ordinal, legal_order, natural, path)
    return (legal_order, natural, path)


def _canonical_us_descendant_ordinal(path: str) -> int | None:
    parts = path.split("/")
    fragments = _us_legal_hierarchy_fragments(parts)
    if not fragments:
        return None
    kind = _legal_marker_kind_for_level(
        len(fragments) - 1,
        document_class=parts[1],
    )
    return dict(_legal_marker_kind_ordinals(fragments[-1])).get(kind)


def _citation_lookup_groups(
    citation_path: str,
) -> tuple[tuple[_LookupCandidate, ...], ...]:
    groups: list[tuple[_LookupCandidate, ...]] = [
        (_LookupCandidate(citation_path, citation_path, False),)
    ]
    seen = {citation_path}
    parent_paths = ((citation_path, _parent_citation_paths(citation_path)),)
    max_parent_depth = max((len(paths) for _, paths in parent_paths), default=0)
    for parent_index in range(max_parent_depth):
        group: list[_LookupCandidate] = []
        for requested_path, paths in parent_paths:
            if parent_index >= len(paths):
                continue
            parent = paths[parent_index]
            if parent in seen:
                continue
            seen.add(parent)
            group.append(_LookupCandidate(parent, requested_path, True))
        if group:
            groups.append(tuple(group))
    return tuple(groups)


def _parent_citation_paths(citation_path: str) -> tuple[str, ...]:
    parts = citation_path.split("/")
    if len(parts) < 4 or parts[1] not in {"statute", "regulation", "legislation"}:
        return ()
    return tuple("/".join(parts[:end]) for end in range(len(parts) - 1, 2, -1))


def _slice_parent_body(text: str, *, requested_path: str, resolved_path: str) -> str:
    requested_parts = requested_path.split("/")
    resolved_parts = resolved_path.split("/")
    if (
        len(requested_parts) <= len(resolved_parts)
        or requested_parts[: len(resolved_parts)] != resolved_parts
        or resolved_parts[1] not in {"statute", "regulation", "legislation"}
    ):
        raise CorpusSourceSliceError(
            f"Cannot slice {requested_path!r} from unrelated parent {resolved_path!r}"
        )
    if (
        len(requested_parts) == 4
        and requested_parts[:2] == ["us", "regulation"]
        and requested_parts[2].isdigit()
    ):
        raise CorpusSourceSliceError(
            f"Cannot synthesize canonical CFR part path {requested_path!r} "
            f"from parent {resolved_path!r}"
        )
    fragments = tuple(requested_parts[len(resolved_parts) :])
    hierarchy_base_length = _us_legal_hierarchy_base_length(requested_parts)
    if hierarchy_base_length is not None:
        hierarchical_fragments = _us_legal_hierarchy_fragments(requested_parts)
        if hierarchical_fragments is None:
            raise CorpusSourceSliceError(
                f"Cannot isolate canonical U.S. legal path {requested_path!r}; "
                "its descendant fragments are not legal hierarchy markers"
            )
        ancestor_fragments = _us_legal_hierarchy_fragments(
            resolved_parts,
            allow_empty=True,
        )
        if (
            ancestor_fragments is None
            or hierarchical_fragments[: len(ancestor_fragments)] != ancestor_fragments
        ):
            raise CorpusSourceSliceError(
                f"Cannot establish legal hierarchy depth for parent {resolved_path!r}"
            )
        parser_ancestor_options = _legal_ancestor_parse_options(
            text,
            ancestor_fragments,
        )
        successful_slice: str | None = None
        retained_interpretation_without_target = False
        errors: list[CorpusSourceSliceError] = []
        for parser_ancestors in parser_ancestor_options:
            retained_interpretation = parser_ancestors != ancestor_fragments
            try:
                hierarchical = _slice_us_legal_hierarchy(
                    text,
                    hierarchical_fragments,
                    document_class=resolved_parts[1],
                    ancestor_fragments=parser_ancestors,
                )
            except CorpusSourceSliceError as exc:
                errors.append(exc)
                if retained_interpretation:
                    retained_interpretation_without_target = True
                continue
            if hierarchical is None:
                if retained_interpretation:
                    retained_interpretation_without_target = True
                continue
            candidate_slice = hierarchical.strip()
            if successful_slice is None:
                successful_slice = candidate_slice
            elif candidate_slice != successful_slice:
                raise CorpusSourceSliceError(
                    f"Ambiguous retained parent-marker representation for "
                    f"{requested_path!r}"
                )
        if successful_slice is not None:
            if retained_interpretation_without_target:
                raise CorpusSourceSliceError(
                    f"Ambiguous retained parent-marker representation for "
                    f"{requested_path!r}"
                )
            return successful_slice
        if errors:
            raise errors[0]
        missing = "/".join(fragments)
        raise CorpusSourceSliceError(
            f"Could not isolate {requested_path!r} from active parent "
            f"{resolved_path!r}; missing structural marker path {missing!r}"
        )
    sliced = _slice_parenthetical_fragments(
        text,
        fragments,
        depth=0,
        budget=_ParentheticalSliceBudget(),
    )
    if sliced is None:
        missing = "/".join(fragments)
        raise CorpusSourceSliceError(
            f"Could not isolate {requested_path!r} from active parent "
            f"{resolved_path!r}; missing structural marker path {missing!r}"
        )
    return sliced.strip()


def _us_legal_hierarchy_fragments(
    parts: list[str],
    *,
    allow_empty: bool = False,
) -> tuple[str, ...] | None:
    """Return structural suffixes for canonical US statute/CFR citations."""

    base_length = _us_legal_hierarchy_base_length(parts)
    if base_length is None:
        return None
    fragments = tuple(parts[base_length:])
    if not fragments:
        return () if allow_empty else None
    if any(not _legal_marker_kind_ordinals(item) for item in fragments):
        return None
    return fragments


def _us_legal_hierarchy_base_length(parts: list[str]) -> int | None:
    """Return the canonical section-path width for U.S. Code or CFR paths."""

    if len(parts) < 4 or parts[0] != "us" or not parts[2].isdigit():
        return None
    if parts[1] == "statute":
        return 4
    if parts[1] == "regulation" and len(parts) >= 5:
        # Normalized CFR paths split a dotted section such as 273.2 into
        # ``.../273/2``; structural paragraph markers begin after that pair.
        return 5
    return None


def _legal_ancestor_parse_options(
    text: str,
    ancestor_fragments: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    """Enumerate omitted/retained parent-marker representations.

    Corpus producers have emitted both ``(1) ...`` for a stored ``.../a`` row
    and ``(a)(1) ...`` for the same row.  The first marker at the beginning of
    the stored body narrows the plausible retained depths. Hierarchy kinds
    cycle, so a token can match multiple ancestor depths; the caller runs every
    plausible parse and rejects divergent successful slices.
    """

    starts_with_outer_marker = bool(
        ancestor_fragments
        and _body_starts_with_legal_marker_after_preamble(text, ancestor_fragments[0])
    )
    starts_with_own_marker = bool(
        len(ancestor_fragments) > 1
        and _body_starts_with_legal_marker_after_preamble(text, ancestor_fragments[-1])
    )
    if not starts_with_outer_marker and not starts_with_own_marker:
        for index in range(1, len(ancestor_fragments) - 1):
            retained_suffix = ancestor_fragments[index:]
            if _body_may_begin_with_retained_ancestor_suffix(text, retained_suffix):
                raise CorpusSourceSliceError(
                    "Ambiguous middle retained-ancestor marker chain"
                )

    options = [ancestor_fragments]
    if starts_with_outer_marker:
        options.append(())
    if starts_with_own_marker:
        options.append(ancestor_fragments[:-1])
    return tuple(dict.fromkeys(options))


def _body_may_begin_with_retained_ancestor_suffix(
    text: str,
    retained_suffix: tuple[str, ...],
) -> bool:
    """Conservatively detect a middle retained-parent marker chain.

    A deep child can have the same first token as a middle ancestor because
    legal marker kinds cycle.  Reject only when the following bounded marker
    stream can traverse the retained suffix.  A different strong line-start
    marker proves the stream exited that interpretation; a weak intervening
    marker remains ambiguous and therefore fails closed.
    """

    if len(retained_suffix) < 2:
        return False
    marker_pattern = re.compile(
        r"(?P<marker>\((?P<token>[A-Za-z0-9]+)\))"
        r"(?=\s+|\([A-Za-z0-9]+\))"
    )
    first: re.Match[str] | None = None
    skipped_strong_marker = False
    for index, match in enumerate(marker_pattern.finditer(text)):
        if index >= _MAX_US_LEGAL_HIERARCHY_MARKERS:
            raise CorpusSourceSliceError(
                "Retained-parent preamble exceeds the bounded marker limit of "
                f"{_MAX_US_LEGAL_HIERARCHY_MARKERS}"
            )
        structurally_anchored = _parenthetical_marker_has_strong_boundary(
            text, match.start("marker")
        ) or not text[: match.start("marker")].lstrip("\ufeff[ \t\r\n")
        if not structurally_anchored:
            continue
        if match.group("token") == retained_suffix[0]:
            first = match
            break
        skipped_strong_marker = True
    if first is None:
        return False
    if skipped_strong_marker:
        return True
    preamble = text[: first.start("marker")].lstrip("\ufeff")
    if preamble.strip("[ \t\r\n") and (
        len(preamble) > _MAX_RETAINED_PARENT_PREAMBLE_CHARS
        or re.search(r"\n\s*\n\s*$", preamble) is None
        or "(" in preamble
        or ")" in preamble
    ):
        # The first strong marker matches a middle ancestor but its preamble
        # cannot be conclusively classified within the bounded heading form.
        return True
    matches = tuple(
        islice(
            marker_pattern.finditer(text, first.start("marker")),
            len(retained_suffix),
        )
    )
    for expected, match in zip(retained_suffix[1:], matches[1:], strict=False):
        if match.group("token") == expected:
            continue
        return not _parenthetical_marker_has_strong_boundary(
            text, match.start("marker")
        )
    return len(matches) >= len(retained_suffix)


def _body_starts_with_legal_marker(text: str, token: str) -> bool:
    return (
        re.match(
            rf"\s*\[?\({re.escape(token)}\)(?=\s|\()",
            text,
        )
        is not None
    )


def _body_starts_with_legal_marker_after_preamble(text: str, token: str) -> bool:
    """Recognize a retained parent marker after a bounded heading preamble.

    Stored provisions sometimes retain their own marker after a plain-text
    heading.  Considering only byte zero lets a later repeated hierarchy token
    masquerade as the requested child.  The first structurally anchored legal
    marker is therefore authoritative; an unbounded or non-heading preamble is
    rejected rather than silently interpreted as an omitted parent marker.
    """

    if _body_starts_with_legal_marker(text, token):
        return True
    marker_pattern = re.compile(
        r"(?P<marker>\((?P<token>[A-Za-z0-9]+)\))"
        r"(?=\s+|\([A-Za-z0-9]+\))"
    )
    marker_matches = tuple(
        islice(
            marker_pattern.finditer(text),
            _MAX_US_LEGAL_HIERARCHY_MARKERS + 1,
        )
    )
    if len(marker_matches) > _MAX_US_LEGAL_HIERARCHY_MARKERS:
        raise CorpusSourceSliceError(
            "Retained-parent preamble exceeds the bounded marker limit of "
            f"{_MAX_US_LEGAL_HIERARCHY_MARKERS}"
        )
    delimiter_enclosed_starts = _delimiter_enclosed_marker_starts(
        text,
        (match.start("marker") for match in marker_matches),
    )
    for match in marker_matches:
        marker_start = match.start("marker")
        if not _legal_marker_kind_ordinals(match.group("token")):
            continue
        if not _parenthetical_marker_has_strong_boundary(text, marker_start):
            continue
        if not _parenthetical_marker_is_structural(
            text,
            marker_start,
            delimiter_enclosed_starts=delimiter_enclosed_starts,
        ):
            continue
        if match.group("token") != token:
            return False
        preamble = text[:marker_start].lstrip("\ufeff")
        if (
            len(preamble) > _MAX_RETAINED_PARENT_PREAMBLE_CHARS
            or re.search(r"\n\s*\n\s*\[?$", preamble) is None
        ):
            raise CorpusSourceSliceError(
                "Could not safely classify retained-parent marker preamble"
            )
        return True
    return False


def _slice_us_legal_hierarchy(
    text: str,
    fragments: tuple[str, ...],
    *,
    document_class: str,
    ancestor_fragments: tuple[str, ...] = (),
) -> str | None:
    """Slice a US Code/CFR parenthetical path while tracking marker depth."""

    if len(ancestor_fragments) >= len(fragments):
        return None
    first_scope = text

    stack: list[tuple[str, str, int]] = []
    for level, token in enumerate(ancestor_fragments):
        kind = _legal_marker_kind_for_level(level, document_class=document_class)
        ordinal = dict(_legal_marker_kind_ordinals(token)).get(kind)
        if ordinal is None:
            return None
        stack.append((kind, token, ordinal))
    last_strong_top_level_ordinal = stack[0][2] if stack else None
    target_start: int | None = None
    target_level: int | None = None
    target_kind: str | None = None
    target_ordinal: int | None = None
    rejected_target_seen = False
    ambiguous_unassigned_before_target = False
    provisional_levels: set[int] = set()
    ambiguous_deep_advancing_level: int | None = None
    last_structural_end: int | None = None
    marker_pattern = re.compile(
        r"(?P<marker>\((?P<token>[A-Za-z0-9]+)\))"
        r"(?=\s+|\([A-Za-z0-9]+\))"
    )
    marker_matches = tuple(
        islice(
            marker_pattern.finditer(first_scope),
            _MAX_US_LEGAL_HIERARCHY_MARKERS + 1,
        )
    )
    if len(marker_matches) > _MAX_US_LEGAL_HIERARCHY_MARKERS:
        raise CorpusSourceSliceError(
            "US legal hierarchy exceeds the bounded marker limit of "
            f"{_MAX_US_LEGAL_HIERARCHY_MARKERS}"
        )
    delimiter_enclosed_starts = _delimiter_enclosed_marker_starts(
        first_scope,
        (match.start("marker") for match in marker_matches),
    )
    evidence_budget = _HierarchyEvidenceBudget()
    for marker_index, match in enumerate(marker_matches):
        token = match.group("token")
        directly_structural = _parenthetical_marker_is_structural(
            first_scope,
            match.start("marker"),
            delimiter_enclosed_starts=delimiter_enclosed_starts,
        )
        has_strong_boundary = _parenthetical_marker_has_strong_boundary(
            first_scope, match.start("marker")
        )
        begins_roman_sequence = _marker_begins_roman_sequence(
            first_scope,
            marker_matches,
            marker_index,
            stack=stack,
            document_class=document_class,
            delimiter_enclosed_starts=delimiter_enclosed_starts,
        )
        assigned = _assign_legal_marker_level(
            stack,
            token,
            document_class=document_class,
            target_fragments=fragments,
            prefer_target_child=not directly_structural,
            prefer_roman_child=begins_roman_sequence,
        )
        marker_kind_ordinals = _legal_marker_kind_ordinals(token)
        top_level_kind = _legal_marker_kind_for_level(0, document_class=document_class)
        top_level_ordinal = dict(marker_kind_ordinals).get(top_level_kind)
        unambiguous_top_level_token = len(marker_kind_ordinals) == 1
        if (
            assigned is None
            and directly_structural
            and has_strong_boundary
            and unambiguous_top_level_token
            and top_level_ordinal is not None
            and (
                last_strong_top_level_ordinal is None
                or top_level_ordinal > last_strong_top_level_ordinal
            )
        ):
            # Weak inline Roman/alpha collisions can temporarily corrupt the
            # inferred stack in flattened text. A later strongly anchored,
            # advancing top-level marker re-establishes the provision boundary.
            assigned = (0, top_level_kind, top_level_ordinal)
        if assigned is None:
            if (
                target_start is None
                and directly_structural
                and has_strong_boundary
                and stack
                and _unassigned_marker_blocks_hierarchy(
                    match,
                    scan_stack=stack,
                    highest_relevant_level=len(stack) - 1,
                    document_class=document_class,
                )
            ):
                ambiguous_unassigned_before_target = True
            if (
                directly_structural
                and target_start is not None
                and target_level is not None
                and target_kind is not None
                and target_ordinal is not None
            ):
                ordinals = dict(_legal_marker_kind_ordinals(token))
                same_level_ordinal = ordinals.get(target_kind)
                if same_level_ordinal is not None:
                    if same_level_ordinal > target_ordinal:
                        sibling_start = _bracket_aware_marker_start(
                            first_scope,
                            match.start("marker"),
                        )
                        return first_scope[target_start:sibling_start]
                    raise CorpusSourceSliceError(
                        "Malformed duplicate or backward legal marker sequence"
                    )
                if _unassigned_marker_blocks_hierarchy(
                    match,
                    scan_stack=stack,
                    highest_relevant_level=target_level,
                    document_class=document_class,
                ):
                    raise CorpusSourceSliceError(
                        "Malformed duplicate or backward legal marker sequence"
                    )
            continue
        level, kind, ordinal = assigned
        potential_ambiguous_advancing = (
            target_start is not None
            and target_level is not None
            and target_kind is not None
            and target_ordinal is not None
            and level > target_level
            and kind == target_kind
            and any(
                target_level < provisional_level <= level
                for provisional_level in provisional_levels
            )
            and dict(_legal_marker_kind_ordinals(token)).get(target_kind, 0)
            > target_ordinal
        )
        candidate_path = tuple(item[1] for item in stack[:level]) + (token,)
        has_preceding_heading = _marker_has_allowlisted_preceding_heading(
            first_scope,
            match,
            last_structural_end=last_structural_end,
        )
        marker_is_structural = _legal_marker_is_structural(
            first_scope,
            match,
            marker_matches=marker_matches,
            marker_index=marker_index,
            document_class=document_class,
            stack=stack,
            assigned_level=level,
            assigned_kind=kind,
            assigned_ordinal=ordinal,
            last_structural_end=last_structural_end,
            evidence_budget=evidence_budget,
            delimiter_enclosed_starts=delimiter_enclosed_starts,
        )
        if not marker_is_structural:
            if candidate_path == fragments and (
                (directly_structural and has_strong_boundary) or has_preceding_heading
            ):
                rejected_target_seen = True
            if (directly_structural or has_preceding_heading) and level == len(stack):
                # Flattened CFR text can put a genuine descendant immediately
                # after its heading (for example ``Disability. (A)``) without
                # enough evidence to resolve that descendant itself. Preserve
                # it only as provisional scope so its strong children are not
                # mistaken for siblings of an already-resolved ancestor.
                if level >= _MAX_US_LEGAL_HIERARCHY_DEPTH:
                    raise CorpusSourceSliceError(
                        "US legal hierarchy exceeds the bounded depth limit of "
                        f"{_MAX_US_LEGAL_HIERARCHY_DEPTH}"
                    )
                stack.append((kind, token, ordinal))
                provisional_levels.add(level)
            continue
        stack = stack[:level]
        provisional_levels = {
            provisional_level
            for provisional_level in provisional_levels
            if provisional_level < level
        }
        stack.append((kind, token, ordinal))
        last_structural_end = match.end("marker")
        path = tuple(item[1] for item in stack)
        proves_top_level_boundary = (
            level == 0 and has_strong_boundary and unambiguous_top_level_token
        )
        if proves_top_level_boundary:
            last_strong_top_level_ordinal = ordinal

        if potential_ambiguous_advancing:
            ambiguous_deep_advancing_level = (
                level
                if ambiguous_deep_advancing_level is None
                else min(ambiguous_deep_advancing_level, level)
            )
        elif (
            ambiguous_deep_advancing_level is not None
            and target_level is not None
            and target_level < level < ambiguous_deep_advancing_level
        ):
            # Moving to an accepted ancestor that is still inside the active
            # target proves the advancing marker belonged to the deeper scope.
            ambiguous_deep_advancing_level = None

        if target_start is None:
            if path == fragments:
                if ambiguous_unassigned_before_target:
                    raise CorpusSourceSliceError(
                        "Ambiguous structural target follows an unassigned legal marker"
                    )
                if provisional_levels:
                    rejected_target_seen = True
                    continue
                if rejected_target_seen:
                    raise CorpusSourceSliceError(
                        "Ambiguous structural target follows a rejected marker "
                        f"for {'/'.join(fragments)!r}"
                    )
                target_start = _bracket_aware_marker_start(
                    first_scope, match.start("marker")
                )
                target_level = level
                target_kind = kind
                target_ordinal = ordinal
            elif proves_top_level_boundary:
                ambiguous_unassigned_before_target = False
            continue
        if target_level is not None and level <= target_level:
            if ambiguous_deep_advancing_level is not None:
                raise CorpusSourceSliceError(
                    "Ambiguous advancing legal marker below provisional scope"
                )
            sibling_start = _bracket_aware_marker_start(
                first_scope, match.start("marker")
            )
            return first_scope[target_start:sibling_start]

    if target_start is not None:
        if ambiguous_deep_advancing_level is not None:
            raise CorpusSourceSliceError(
                "Ambiguous advancing legal marker below provisional scope"
            )
        return first_scope[target_start:]
    return None


def _bracket_aware_marker_start(text: str, marker_start: int) -> int:
    """Include an editorial ``[`` that immediately wraps a legal marker."""

    if marker_start > 0 and text[marker_start - 1] == "[":
        return marker_start - 1
    return marker_start


def _marker_begins_roman_sequence(
    text: str,
    matches: tuple[re.Match[str], ...],
    index: int,
    *,
    stack: list[tuple[str, str, int]],
    document_class: str,
    delimiter_enclosed_starts: AbstractSet[int],
) -> bool:
    """Return whether a line-start Roman marker is followed by its sibling.

    A lone ``(i)`` after subsection ``(h)`` is ambiguous and remains a
    top-level sibling. The canonical ``(i)``, ``(ii)`` sequence provides the
    additional structural evidence needed to treat line-start ``(i)`` as the
    expected nested Roman child.
    """

    if index + 1 >= len(matches):
        return False
    expected_kind = _legal_marker_kind_for_level(
        len(stack),
        document_class=document_class,
    )
    if expected_kind not in {"lower_roman", "upper_roman"}:
        return False
    current_ordinals = dict(_legal_marker_kind_ordinals(matches[index].group("token")))
    next_match = matches[index + 1]
    next_ordinals = dict(_legal_marker_kind_ordinals(next_match.group("token")))
    current_ordinal = current_ordinals.get(expected_kind)
    next_ordinal = next_ordinals.get(expected_kind)
    if current_ordinal is None or next_ordinal != current_ordinal + 1:
        return False
    return _parenthetical_marker_is_structural(
        text,
        next_match.start("marker"),
        delimiter_enclosed_starts=delimiter_enclosed_starts,
    )


def _assign_legal_marker_level(
    stack: list[tuple[str, str, int]],
    token: str,
    *,
    document_class: str,
    target_fragments: tuple[str, ...],
    prefer_target_child: bool,
    prefer_roman_child: bool,
) -> tuple[int, str, int] | None:
    possible = _legal_marker_kind_ordinals(token)
    if not possible:
        return None

    child_level = len(stack)
    expected_child_kind = _legal_marker_kind_for_level(
        child_level,
        document_class=document_class,
    )
    if prefer_roman_child:
        for kind, ordinal in possible:
            if kind == expected_child_kind:
                return (child_level, kind, ordinal)
    stack_path = tuple(item[1] for item in stack)
    if (
        prefer_target_child
        and stack_path == target_fragments[:child_level]
        and child_level < len(target_fragments)
        and token == target_fragments[child_level]
    ):
        for kind, ordinal in possible:
            if kind == expected_child_kind:
                return (child_level, kind, ordinal)
    roman_alpha_kind = {
        "lower_roman": "lower_alpha",
        "upper_roman": "upper_alpha",
    }.get(expected_child_kind)
    if roman_alpha_kind is not None:
        for level in range(len(stack) - 1, -1, -1):
            expected_kind = _legal_marker_kind_for_level(
                level,
                document_class=document_class,
            )
            for kind, ordinal in possible:
                if (
                    kind == expected_kind == roman_alpha_kind
                    and ordinal == stack[level][2] + 1
                ):
                    return (level, kind, ordinal)
        for kind, ordinal in possible:
            if kind == expected_child_kind:
                return (child_level, kind, ordinal)

    for level in range(len(stack) - 1, -1, -1):
        expected_kind = _legal_marker_kind_for_level(
            level,
            document_class=document_class,
        )
        for kind, ordinal in possible:
            if kind == expected_kind and ordinal > stack[level][2]:
                return (level, kind, ordinal)

    for kind, ordinal in possible:
        if kind == expected_child_kind:
            return (child_level, kind, ordinal)
    return None


def _legal_marker_kind_for_level(level: int, *, document_class: str) -> str:
    if level == 0:
        return "lower_alpha"
    if document_class == "regulation":
        return ("numeric", "lower_roman", "upper_alpha")[(level - 1) % 3]
    return ("numeric", "upper_alpha", "lower_roman", "upper_roman")[(level - 1) % 4]


def _legal_marker_kind_ordinals(token: str) -> list[tuple[str, int]]:
    kinds: list[tuple[str, int]] = []
    if len(token) > MAX_LEGAL_MARKER_TOKEN_LENGTH:
        return kinds
    if re.fullmatch(r"[1-9]\d*", token):
        numeric = int(token)
        kinds.append(("numeric", numeric))
    if re.fullmatch(r"[a-z]", token):
        kinds.append(("lower_alpha", ord(token) - ord("a") + 1))
    if re.fullmatch(r"[A-Z]", token):
        kinds.append(("upper_alpha", ord(token) - ord("A") + 1))
    if _is_canonical_roman_marker(token, uppercase=False):
        roman = _roman_marker_to_int(token.lower())
        if roman is not None:
            kinds.append(("lower_roman", roman))
    if _is_canonical_roman_marker(token, uppercase=True):
        roman = _roman_marker_to_int(token.lower())
        if roman is not None:
            kinds.append(("upper_roman", roman))
    return kinds


def _is_canonical_roman_marker(token: str, *, uppercase: bool) -> bool:
    pattern = (
        r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})"
        r"(?:IX|IV|V?I{0,3})"
    )
    if not uppercase:
        pattern = pattern.lower()
    return bool(token and re.fullmatch(pattern, token))


def _roman_marker_to_int(token: str) -> int | None:
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    previous = 0
    for character in reversed(token):
        value = values.get(character)
        if value is None:
            return None
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total or None


_INLINE_STRUCTURAL_HEADING_EXACT = frozenset(
    {
        "additional tax",
        "aged or blind additional amounts",
        "allowable financial resources",
        "availability of allowance to recipients of energy assistance",
        "deduction for child support payments",
        "deductions from income",
        "dependent care deduction",
        "earned income deduction",
        "excess medical expense deduction",
        "excess shelter expense deduction",
        "first month benefits prorated",
        "guam",
        "hospital insurance",
        "in general",
        "mandatory allowance",
        "method of claiming deduction",
        "qualified employee health insurance costs",
        "reduction of public assistance benefits",
        "standard deduction",
        "standard utility allowance",
        "subsequent eligibility",
        "work requirement",
    }
)
_FOLLOWING_STRUCTURAL_HEADING_PREFIX_EXACT = (
    _INLINE_STRUCTURAL_HEADING_EXACT
    | frozenset(
        {
            "adjustments for inflation",
            "certain individuals, etc., not eligible for standard deduction",
            "deduction",
            "earned income deduction",
            "exception",
            "excess medical expense deduction",
            "excess shelter deduction",
            "excess shelter expense deduction",
            "homeless shelter deduction",
            "homeless shelter deduction.",
            "included assets.—",
            "in general .—",
            "limitation on basic standard deduction in the case of certain dependents",
            "maximum amount of deduction",
            "minimum amount",
            "optional combined allotment for expedited households .—",
            "shelter costs--",
            "shelter costs—",
            "standard deduction",
            "standard deduction—",
            "standard utility allowances",
            "standard utility allowances.",
            "total amount.—",
            "utility standard",
            "utility standard.",
        }
    )
)
_FOLLOWING_HEADING_CONTENT_PREFIXES = MappingProxyType(
    {
        "additional tax": ("in addition to the tax imposed by",),
        "adjustments for inflation": (
            "in the case of any taxable year beginning in a calendar year after 1988",
        ),
        "certain individuals, etc., not eligible for standard deduction": (
            "in the case of—",
        ),
        "deduction": (
            "except as provided in subparagraph (c)",
            "the secretary shall allow a standard deduction",
        ),
        "exception": ("the deduction described in subparagraph (b)",),
        "excess shelter deduction.": ("monthly shelter expenses in excess of",),
        "homeless shelter deduction.": ("a state agency may provide",),
        "hospital insurance": ("in addition to the tax imposed by",),
        "limitation on basic standard deduction in the case of certain dependents": (
            "in the case of an individual with respect to whom",
        ),
        "maximum amount of deduction": (
            "in the case of a household that does not contain",
        ),
        "minimum amount": ("notwithstanding clause (i)",),
    }
)
_STRUCTURAL_CONTENT_PREFIX_EXACT = ("as used in this subsection, the term",)
_SELF_DELIMITED_HEADING_SUFFIXES = (".—", "—", "--")
_TERMINAL_SIBLING_HEADING_EXACT = frozenset(
    {"certain individuals, etc., not eligible for standard deduction."}
)
_MAX_TERMINAL_SIBLING_HEADING_RAW_CHARS = 256
_MAX_INLINE_HEADING_EVIDENCE_DEPTH = 32
_MAX_INLINE_HEADING_EVIDENCE_WORK = 10_000
_MAX_PRECEDING_REFERENCE_CONTEXT_CHARS = 32_768
_MAX_RETAINED_PARENT_PREAMBLE_CHARS = 512
_MAX_US_LEGAL_HIERARCHY_MARKERS = 4_096
_MAX_US_LEGAL_HIERARCHY_DEPTH = MAX_CORPUS_CITATION_SEGMENTS
_MAX_DELIMITER_NESTING_DEPTH = 256


@dataclass
class _HierarchyEvidenceBudget:
    remaining: int = _MAX_INLINE_HEADING_EVIDENCE_WORK

    def consume(self) -> None:
        self.remaining -= 1
        if self.remaining < 0:
            raise CorpusSourceSliceError(
                "US legal hierarchy exceeds the bounded inline-heading "
                f"evidence work limit of {_MAX_INLINE_HEADING_EVIDENCE_WORK}"
            )


def _is_allowlisted_inline_structural_heading(heading: str) -> bool:
    """Return whether prose has positive evidence of being a legal heading.

    Flattened U.S. Code bodies can place a child marker directly after a
    heading (``Deductions from income (1)``).  Ordinary prose can do the same
    with an inline reference (``See (1)``), so short, unpunctuated prose alone
    is not structural evidence.  Accept only the exact heading forms exercised
    by the supported hierarchy and live corpus; preserving punctuation and
    digits keeps citation-like prose out of the allowlist.
    """

    if not heading[0].isupper():
        return False
    normalized = " ".join(heading.split()).casefold()
    return normalized in _INLINE_STRUCTURAL_HEADING_EXACT


def _legal_marker_is_structural(
    text: str,
    match: re.Match[str],
    *,
    marker_matches: tuple[re.Match[str], ...],
    marker_index: int,
    document_class: str,
    stack: list[tuple[str, str, int]],
    assigned_level: int,
    assigned_kind: str,
    assigned_ordinal: int,
    last_structural_end: int | None,
    evidence_budget: _HierarchyEvidenceBudget,
    delimiter_enclosed_starts: AbstractSet[int],
    evidence_depth: int = 0,
) -> bool:
    marker_start = match.start("marker")
    directly_structural = _parenthetical_marker_is_structural(
        text,
        marker_start,
        delimiter_enclosed_starts=delimiter_enclosed_starts,
    )
    has_strong_boundary = _parenthetical_marker_has_strong_boundary(text, marker_start)
    if not _parenthetical_marker_has_paragraph_boundary(
        text, marker_start
    ) and _marker_follows_explicit_reference_cue(
        text,
        marker_start,
        context_start=last_structural_end,
    ):
        return False
    if directly_structural and has_strong_boundary:
        return True
    if evidence_depth >= _MAX_INLINE_HEADING_EVIDENCE_DEPTH:
        return False
    if any(
        _legal_marker_kind_for_level(level, document_class=document_class)
        == assigned_kind
        for level in range(assigned_level)
    ):
        # Once a hierarchy kind cycles, a later matching ordinal is equally
        # plausible as an ancestor sibling and cannot independently prove this
        # inline marker's depth. Direct structural markers remain supported.
        return False
    if directly_structural:
        # Punctuation alone also introduces sentence-level references.  A
        # later real sibling cannot retroactively distinguish such a reference
        # from a flattened structural marker, so require explicit heading
        # evidence at this weak boundary.
        if _marker_has_allowlisted_following_heading(
            text,
            match,
            allow_replayable_content=False,
            context_start=last_structural_end,
        ):
            return True
        if not _marker_has_allowlisted_following_heading(
            text,
            match,
            context_start=last_structural_end,
        ):
            return False
        return _preceding_scope_references_candidate_marker(
            text,
            match,
            last_structural_end=last_structural_end,
            assigned_kind=assigned_kind,
        ) or _has_following_structural_sibling(
            text,
            marker_matches,
            marker_index,
            document_class=document_class,
            stack=stack,
            assigned_level=assigned_level,
            assigned_kind=assigned_kind,
            assigned_ordinal=assigned_ordinal,
            evidence_budget=evidence_budget,
            evidence_depth=evidence_depth,
            delimiter_enclosed_starts=delimiter_enclosed_starts,
        )
    if assigned_level != len(stack) or last_structural_end is None:
        return False
    return _marker_has_allowlisted_preceding_heading(
        text,
        match,
        last_structural_end=last_structural_end,
    ) and _has_following_structural_sibling(
        text,
        marker_matches,
        marker_index,
        document_class=document_class,
        stack=stack,
        assigned_level=assigned_level,
        assigned_kind=assigned_kind,
        assigned_ordinal=assigned_ordinal,
        evidence_budget=evidence_budget,
        evidence_depth=evidence_depth,
        delimiter_enclosed_starts=delimiter_enclosed_starts,
    )


def _marker_has_allowlisted_preceding_heading(
    text: str,
    match: re.Match[str],
    *,
    last_structural_end: int | None,
) -> bool:
    if last_structural_end is None:
        return False
    if match.start("marker") - last_structural_end > 160:
        return False
    heading = text[last_structural_end : match.start("marker")]
    if "\n\n" in heading:
        return False
    normalized = " ".join(heading.split())
    return bool(
        normalized
        and not re.search(r"[.;:]", normalized)
        and _is_allowlisted_inline_structural_heading(normalized)
    )


def _preceding_scope_references_candidate_marker(
    text: str,
    match: re.Match[str],
    *,
    last_structural_end: int | None,
    assigned_kind: str,
) -> bool:
    if last_structural_end is None:
        return False
    labels = {
        "lower_alpha": r"(?:subsection|paragraph)",
        "numeric": r"paragraph",
        "upper_alpha": r"subparagraph",
        "lower_roman": r"clause",
        "upper_roman": r"subclause",
    }.get(assigned_kind)
    if labels is None:
        return False
    marker_start = match.start("marker")
    if marker_start - last_structural_end > _MAX_PRECEDING_REFERENCE_CONTEXT_CHARS:
        return False
    context = text[last_structural_end:marker_start]
    token = re.escape(match.group("token"))
    return (
        re.search(
            rf"\b(?:except\s+as\s+provided\s+in|subject\s+to|"
            rf"as\s+specified\s+in)\s+(?:the\s+)?{labels}s?\s+\({token}\)",
            context,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _marker_has_allowlisted_following_heading(
    text: str,
    match: re.Match[str],
    *,
    allow_replayable_content: bool = True,
    context_start: int | None = None,
) -> bool:
    """Recognize only vetted headings immediately after a weak marker boundary."""

    tail_start = match.end("marker")
    tail_end = min(len(text), tail_start + 200)
    tail_window_complete = tail_end == len(text)
    tail = " ".join(text[tail_start:tail_end].split())
    if not tail or not tail[0].isupper():
        return False
    folded = tail.casefold()
    if _marker_follows_explicit_reference_cue(
        text,
        match.start("marker"),
        context_start=context_start,
    ):
        return False
    if allow_replayable_content and any(
        _starts_with_evidence_prefix(
            folded,
            prefix,
            window_complete=tail_window_complete,
        )
        for prefix in _STRUCTURAL_CONTENT_PREFIX_EXACT
    ):
        return True
    for heading in sorted(
        _FOLLOWING_STRUCTURAL_HEADING_PREFIX_EXACT,
        key=len,
        reverse=True,
    ):
        if folded == heading:
            return tail_window_complete and heading.endswith(
                _SELF_DELIMITED_HEADING_SUFFIXES
            )
        if not folded.startswith(heading):
            continue
        remainder = tail[len(heading) :]
        if not remainder:
            return True
        next_text = remainder.lstrip()
        if (
            remainder.startswith("(")
            or (remainder.startswith(" ") and next_text.startswith("("))
        ) and (
            heading in _INLINE_STRUCTURAL_HEADING_EXACT
            or heading.endswith(".")
            or heading.endswith(_SELF_DELIMITED_HEADING_SUFFIXES)
        ):
            return True
        if heading.endswith(_SELF_DELIMITED_HEADING_SUFFIXES) and remainder.startswith(
            " "
        ):
            return True
        if not remainder.startswith(" "):
            continue
        next_folded = next_text.casefold()
        if allow_replayable_content and any(
            _starts_with_evidence_prefix(
                next_folded,
                prefix,
                window_complete=tail_window_complete,
            )
            for prefix in _FOLLOWING_HEADING_CONTENT_PREFIXES.get(heading, ())
        ):
            return True
    return False


def _starts_with_evidence_prefix(
    text: str,
    prefix: str,
    *,
    window_complete: bool,
) -> bool:
    if not text.startswith(prefix):
        return False
    if len(text) == len(prefix):
        return window_complete
    return not text[len(prefix)].isalnum()


def _marker_has_vetted_terminal_sibling_heading(
    text: str,
    match: re.Match[str],
) -> bool:
    tail_start = match.end("marker")
    if len(text) - tail_start > _MAX_TERMINAL_SIBLING_HEADING_RAW_CHARS:
        return False
    tail = " ".join(text[tail_start:].split()).casefold()
    return tail in _TERMINAL_SIBLING_HEADING_EXACT


def _marker_follows_explicit_reference_cue(
    text: str,
    marker_start: int,
    *,
    context_start: int | None = None,
) -> bool:
    if context_start is None:
        context_start = max(0, marker_start - _MAX_PRECEDING_REFERENCE_CONTEXT_CHARS)
        context_is_truncated = context_start > 0
    else:
        context_is_truncated = (
            context_start < 0
            or context_start > marker_start
            or marker_start - context_start > _MAX_PRECEDING_REFERENCE_CONTEXT_CHARS
        )
        context_start = max(0, min(context_start, marker_start))
    if context_is_truncated:
        # Absence of a cue in a truncated structural scope is not evidence
        # that the complete scope is safe.
        return True
    context = text[context_start:marker_start]
    legal_label = (
        r"(?:provision|paragraph|section|rule|clause|subparagraph|"
        r"subsection|item)s?"
    )
    direction = (
        r"(?:prior|previous|previously|preceding|foregoing|earlier|former|"
        r"above|aforementioned|following|next|below|subsequent|later)"
    )
    reference_verb = (
        r"(?:compare|consult|review|see|look\s+at|"
        r"refer(?:ring)?(?:\s+to)?|cross[- ]reference)"
    )
    return (
        re.search(
            rf"\b{reference_verb}\b(?:\W+\w+){{0,8}}\W+\b{legal_label}\b|"
            rf"\b{reference_verb}\b(?:\W+\w+){{0,4}}\W+\b{direction}\b"
            rf"(?:\W+\w+){{0,4}}\W+\b{legal_label}\b|"
            rf"\b{reference_verb}\b(?:\W+\w+){{0,4}}\W+\b{legal_label}\b"
            rf"(?:\W+\w+){{0,4}}\W+\b{direction}\b|"
            rf"\b{direction}\b(?:\W+\w+){{0,4}}\W+\b{legal_label}\b|"
            rf"\b{legal_label}\b(?:\W+\w+){{0,4}}\W+\b{direction}\b|"
            r"\b(?:described|noted|stated|specified|discussed|mentioned|"
            r"set\s+(?:out|forth))\s+(?:above|earlier|previously|before)\b|"
            r"\b(?:example|illustration|quoted\s+phrase)\s*[:—-]",
            context,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _has_following_structural_sibling(
    text: str,
    marker_matches: tuple[re.Match[str], ...],
    marker_index: int,
    *,
    document_class: str,
    stack: list[tuple[str, str, int]],
    assigned_level: int,
    assigned_kind: str,
    assigned_ordinal: int,
    evidence_budget: _HierarchyEvidenceBudget,
    evidence_depth: int,
    delimiter_enclosed_starts: AbstractSet[int],
) -> bool:
    """Prove an inline marker has a sibling inside the same parent scope."""

    current_match = marker_matches[marker_index]
    scan_stack = stack[:assigned_level]
    scan_stack.append((assigned_kind, current_match.group("token"), assigned_ordinal))
    last_structural_end = current_match.end("marker")
    for later_index, later_match in enumerate(
        marker_matches[marker_index + 1 :],
        start=marker_index + 1,
    ):
        evidence_budget.consume()
        directly_structural = _parenthetical_marker_is_structural(
            text,
            later_match.start("marker"),
            delimiter_enclosed_starts=delimiter_enclosed_starts,
        )
        follows_reference_cue = not _parenthetical_marker_has_paragraph_boundary(
            text,
            later_match.start("marker"),
        ) and _marker_follows_explicit_reference_cue(
            text,
            later_match.start("marker"),
            context_start=last_structural_end,
        )
        has_provisional_evidence = not follows_reference_cue and (
            (
                directly_structural
                and (
                    _parenthetical_marker_has_strong_boundary(
                        text,
                        later_match.start("marker"),
                    )
                    or _marker_has_allowlisted_following_heading(
                        text,
                        later_match,
                        context_start=last_structural_end,
                    )
                    or _marker_has_vetted_terminal_sibling_heading(
                        text,
                        later_match,
                    )
                )
            )
            or _marker_has_allowlisted_preceding_heading(
                text,
                later_match,
                last_structural_end=last_structural_end,
            )
        )
        raw_sibling_ordinal = dict(
            _legal_marker_kind_ordinals(later_match.group("token"))
        ).get(assigned_kind)
        later_marker_kinds = {
            kind
            for kind, _ordinal in _legal_marker_kind_ordinals(
                later_match.group("token")
            )
        }
        competing_hierarchy_kinds = {
            _legal_marker_kind_for_level(level, document_class=document_class)
            for level in range(len(scan_stack) + 1)
            if level != assigned_level
        }
        if (
            raw_sibling_ordinal is not None
            and raw_sibling_ordinal > assigned_ordinal
            and has_provisional_evidence
            and later_marker_kinds.isdisjoint(competing_hierarchy_kinds)
            and not any(
                item_kind == assigned_kind
                for item_kind, _item_token, _item_ordinal in scan_stack[
                    assigned_level + 1 :
                ]
            )
        ):
            return True
        begins_roman_sequence = _marker_begins_roman_sequence(
            text,
            marker_matches,
            later_index,
            stack=scan_stack,
            document_class=document_class,
            delimiter_enclosed_starts=delimiter_enclosed_starts,
        )
        later_assigned = _assign_legal_marker_level(
            scan_stack,
            later_match.group("token"),
            document_class=document_class,
            target_fragments=(),
            prefer_target_child=False,
            prefer_roman_child=begins_roman_sequence,
        )
        if later_assigned is None:
            if directly_structural and _unassigned_marker_blocks_hierarchy(
                later_match,
                scan_stack=scan_stack,
                highest_relevant_level=assigned_level,
                document_class=document_class,
            ):
                return False
            continue
        later_level, later_kind, later_ordinal = later_assigned
        if later_level > assigned_level:
            if has_provisional_evidence:
                scan_stack = scan_stack[:later_level]
                scan_stack.append(
                    (later_kind, later_match.group("token"), later_ordinal)
                )
                last_structural_end = later_match.end("marker")
            continue
        if not _legal_marker_is_structural(
            text,
            later_match,
            marker_matches=marker_matches,
            marker_index=later_index,
            document_class=document_class,
            stack=scan_stack,
            assigned_level=later_level,
            assigned_kind=later_kind,
            assigned_ordinal=later_ordinal,
            last_structural_end=last_structural_end,
            evidence_budget=evidence_budget,
            evidence_depth=evidence_depth + 1,
            delimiter_enclosed_starts=delimiter_enclosed_starts,
        ):
            if later_level < assigned_level and (
                directly_structural
                or _marker_has_allowlisted_following_heading(
                    text,
                    later_match,
                    context_start=last_structural_end,
                )
                or _marker_has_allowlisted_preceding_heading(
                    text,
                    later_match,
                    last_structural_end=last_structural_end,
                )
            ):
                # A plausible parent boundary that cannot itself be proved is
                # still enough to make evidence beyond it ambiguous.
                return False
            continue
        if later_level <= assigned_level:
            return (
                later_level == assigned_level
                and later_kind == assigned_kind
                and later_ordinal > assigned_ordinal
            )
        scan_stack = scan_stack[:later_level]
        scan_stack.append((later_kind, later_match.group("token"), later_ordinal))
        last_structural_end = later_match.end("marker")
    return False


def _unassigned_marker_blocks_hierarchy(
    match: re.Match[str],
    *,
    scan_stack: list[tuple[str, str, int]],
    highest_relevant_level: int,
    document_class: str,
) -> bool:
    """Reject malformed markers plausible at established hierarchy levels."""

    token = match.group("token")
    if re.fullmatch(r"0+", token):
        return True
    ordinals = dict(_legal_marker_kind_ordinals(token))
    highest_relevant_level = min(highest_relevant_level, len(scan_stack) - 1)
    return any(
        _legal_marker_kind_for_level(level, document_class=document_class) in ordinals
        for level in range(highest_relevant_level + 1)
    )


def _slice_parenthetical_fragments(
    text: str,
    fragments: tuple[str, ...],
    *,
    depth: int,
    budget: _ParentheticalSliceBudget,
) -> str | None:
    if not fragments:
        return text

    fragment = fragments[0]
    simple_slice = _slice_parenthetical_fragment(
        text,
        fragment,
        depth=depth,
        top_level=depth == 0,
        budget=budget,
    )
    if simple_slice is not None:
        simple_result = _slice_parenthetical_fragments(
            simple_slice,
            fragments[1:],
            depth=depth + 1,
            budget=budget,
        )
        if simple_result is not None:
            return simple_result

    if len(fragments) >= 2:
        combined = _combined_dotted_fragment(fragment, fragments[1])
        if combined is not None:
            combined_slice = _slice_parenthetical_fragment(
                text,
                combined,
                depth=depth,
                top_level=depth == 0,
                budget=budget,
            )
            if combined_slice is not None:
                return _slice_parenthetical_fragments(
                    combined_slice,
                    fragments[2:],
                    depth=depth + 2,
                    budget=budget,
                )
    return None


def _combined_dotted_fragment(fragment: str, next_fragment: str) -> str | None:
    if not next_fragment.isdigit():
        return None
    if re.fullmatch(r"(?:[A-Za-z]|\d+)(?:\.\d+)*", fragment):
        return f"{fragment}.{next_fragment}"
    return None


def _slice_parenthetical_fragment(
    text: str,
    fragment: str,
    *,
    depth: int,
    top_level: bool,
    budget: _ParentheticalSliceBudget,
) -> str | None:
    escaped = re.escape(fragment)
    delimiter = r"(?:\s+|(?=\())"
    marker = rf"\({escaped}\){delimiter}"
    if top_level:
        marker_pattern = re.compile(rf"(?:^|\n\s*)(\[?{marker})")
    else:
        marker_pattern = re.compile(rf"(?<![A-Za-z0-9])({marker})")
    budget.charge(len(text))
    marker_matches = tuple(
        islice(
            marker_pattern.finditer(text),
            _MAX_US_LEGAL_HIERARCHY_MARKERS + 1,
        )
    )
    if len(marker_matches) > _MAX_US_LEGAL_HIERARCHY_MARKERS:
        raise CorpusSourceSliceError(
            "Parenthetical slicing exceeds the bounded marker limit of "
            f"{_MAX_US_LEGAL_HIERARCHY_MARKERS}"
        )
    if not marker_matches:
        return None
    budget.charge(len(text))
    delimiter_enclosed_starts = _delimiter_enclosed_marker_starts(
        text,
        (match.start(1) for match in marker_matches),
    )
    marker_match: re.Match[str] | None = None
    for match in marker_matches:
        budget.charge(min(match.start(1), 60))
        if _parenthetical_marker_is_structural(
            text,
            match.start(1),
            delimiter_enclosed_starts=delimiter_enclosed_starts,
        ):
            marker_match = match
            break
    if marker_match is None:
        return None
    sibling_pattern = _sibling_marker_pattern(
        fragment,
        depth=depth,
        top_level=top_level,
    )
    end = len(text)
    budget.charge(len(text) - marker_match.end(1))
    sibling_matches = tuple(
        islice(
            sibling_pattern.finditer(text, marker_match.end(1)),
            _MAX_US_LEGAL_HIERARCHY_MARKERS + 1,
        )
    )
    if len(sibling_matches) > _MAX_US_LEGAL_HIERARCHY_MARKERS:
        raise CorpusSourceSliceError(
            "Parenthetical sibling slicing exceeds the bounded marker limit of "
            f"{_MAX_US_LEGAL_HIERARCHY_MARKERS}"
        )
    sibling_enclosed_starts: AbstractSet[int] = frozenset()
    if sibling_matches:
        budget.charge(len(text))
        sibling_enclosed_starts = _delimiter_enclosed_marker_starts(
            text,
            (sibling.start(1) for sibling in sibling_matches),
        )
    for sibling in sibling_matches:
        budget.charge(min(sibling.start(1), 60))
        if _parenthetical_marker_is_structural(
            text,
            sibling.start(1),
            delimiter_enclosed_starts=sibling_enclosed_starts,
        ):
            end = sibling.start(1)
            break
    return text[marker_match.start(1) : end]


def _sibling_marker_pattern(
    fragment: str,
    *,
    depth: int,
    top_level: bool,
) -> re.Pattern[str]:
    if depth > 0 and re.fullmatch(r"[ivxlcdm]+", fragment):
        marker = r"\([ivxlcdm]+\)"
    elif re.fullmatch(r"\d+(?:\.\d+)*", fragment):
        marker = _numeric_sibling_marker(fragment)
    elif re.fullmatch(r"[A-Z](?:\.\d+)*", fragment):
        marker = _alpha_sibling_marker(fragment, top_level=top_level)
    elif re.fullmatch(r"[a-z](?:\.\d+)*", fragment):
        marker = _alpha_sibling_marker(fragment, top_level=top_level)
    else:
        marker = r"\([A-Za-z0-9]+(?:\.\d+)*\)"
    prefix = r"\n\s*" if top_level else r"(?<![A-Za-z0-9])"
    delimiter = r"(?:\s+|(?=\())"
    return re.compile(rf"{prefix}(\[?{marker}{delimiter})")


def _numeric_sibling_marker(fragment: str) -> str:
    stem = fragment.split(".", 1)[0]
    same_stem = _same_stem_dotted_sibling_marker(stem, fragment)
    following_stem = _following_numeric_stem_marker(stem)
    return rf"\((?:{same_stem}|{following_stem}(?:\.[0-9]+)*)\)"


def _following_numeric_stem_marker(stem: str) -> str:
    digits = str(int(stem))
    same_width_patterns: list[str] = []
    for index, digit in enumerate(digits):
        lower = int(digit) + 1
        if index == 0:
            lower = max(lower, 1)
        if lower > 9:
            continue
        prefix = re.escape(digits[:index])
        suffix_length = len(digits) - index - 1
        suffix = rf"[0-9]{{{suffix_length}}}" if suffix_length else ""
        same_width_patterns.append(rf"{prefix}[{lower}-9]{suffix}")
    longer_width = rf"[1-9][0-9]{{{len(digits)},}}"
    return "(?:" + "|".join([*same_width_patterns, longer_width]) + ")"


def _alpha_sibling_marker(fragment: str, *, top_level: bool) -> str:
    stem = fragment[0]
    same_stem = _same_stem_dotted_sibling_marker(stem, fragment)
    following_stem = (
        _next_alpha_stem_marker(stem)
        if top_level
        else _following_alpha_stem_marker(stem)
    )
    return rf"\((?:{same_stem}|{following_stem}(?:\.[0-9]+)*)\)"


def _next_alpha_stem_marker(stem: str) -> str:
    next_codepoint = ord(stem) + 1
    if stem.islower() and next_codepoint <= ord("z"):
        return chr(next_codepoint)
    if stem.isupper() and next_codepoint <= ord("Z"):
        return chr(next_codepoint)
    return r"\b\B"


def _following_alpha_stem_marker(stem: str) -> str:
    next_codepoint = ord(stem) + 1
    if stem.islower() and next_codepoint <= ord("z"):
        return f"[{chr(next_codepoint)}-z]"
    if stem.isupper() and next_codepoint <= ord("Z"):
        return f"[{chr(next_codepoint)}-Z]"
    return r"\b\B"


def _same_stem_dotted_sibling_marker(stem: str, fragment: str) -> str:
    escaped_stem = re.escape(stem)
    if "." not in fragment:
        return rf"{escaped_stem}(?:\.[0-9]+)+"
    suffix = re.escape(fragment.split(".", 1)[1])
    return rf"{escaped_stem}\.(?!{suffix}(?:\.|\)))[0-9]+(?:\.[0-9]+)*"


_NONSTRUCTURAL_PARENTHETICAL_REFERENCE_LABEL = (
    r"(?:paragraphs?|subparagraphs?|clauses?|subclauses?|sections?|"
    r"subsections?|chapters?|titles?|parts?|items?|sentences?|regulations?)"
)
_NONSTRUCTURAL_PARENTHETICAL_REFERENCE_PREFIX = re.compile(
    rf"\b{_NONSTRUCTURAL_PARENTHETICAL_REFERENCE_LABEL}\s+$",
    re.IGNORECASE,
)


def _parenthetical_marker_is_structural(
    text: str,
    marker_start: int,
    *,
    delimiter_enclosed_starts: AbstractSet[int] | None = None,
) -> bool:
    prefix = _bounded_structural_prefix(text, marker_start)
    previous = prefix.rstrip()[-1:] if prefix.rstrip() else ""
    if (
        marker_start in delimiter_enclosed_starts
        if delimiter_enclosed_starts is not None
        else _marker_is_within_quote_or_bracket(text, marker_start)
    ):
        return False
    if re.search(r"[\"'“”‘’\]}][ \t]*$", prefix):
        return False
    if re.search(r"[\[{][ \t]*$", prefix) and not re.search(
        r"\n\s*[\[{][ \t]*$", prefix
    ):
        return False
    if previous == ")" and not re.search(
        r"\n\s*\[?(?:\([A-Za-z0-9]+(?:\.[0-9]+)*\)\s*)+$",
        prefix,
    ):
        return False
    # A marker following prose is an inline reference (for example,
    # ``conditions under (1)``), not a new structural child. Sentence-ending
    # punctuation and an immediately preceding structural marker remain
    # candidates here; the hierarchy parser separately requires corroborating
    # sibling evidence for punctuation-separated first children.
    starts_line = re.search(r"\n\s*$", prefix) is not None
    if not starts_line and (previous.isalnum() or previous in {",", ";", ":"}):
        return False
    if not starts_line and _NONSTRUCTURAL_PARENTHETICAL_REFERENCE_PREFIX.search(prefix):
        return False
    return not _parenthetical_marker_is_in_reference_list(prefix)


def _delimiter_enclosed_marker_starts(
    text: str,
    marker_starts: Iterable[int],
) -> frozenset[int]:
    """Replay delimiter state once and identify enclosed marker offsets.

    The legal hierarchy parser may inspect thousands of markers in a large
    parent provision. Replaying from byte zero for each marker would be
    quadratic, while a fixed look-behind window forgets long quotations. This
    single forward pass retains balanced quote/bracket state with bounded
    memory and records only the supplied marker positions.
    """

    starts_list: list[int] = []
    previous_start: int | None = None
    for supplied_count, marker_start in enumerate(marker_starts, start=1):
        if supplied_count > _MAX_US_LEGAL_HIERARCHY_MARKERS:
            raise CorpusSourceSliceError(
                "Delimiter replay exceeds the bounded marker limit of "
                f"{_MAX_US_LEGAL_HIERARCHY_MARKERS}"
            )
        if (
            not isinstance(marker_start, int)
            or isinstance(marker_start, bool)
            or (previous_start is not None and marker_start < previous_start)
        ):
            raise CorpusSourceSliceError(
                "Delimiter replay marker offsets must be ordered integers"
            )
        if previous_start is None or marker_start != previous_start:
            starts_list.append(marker_start)
        previous_start = marker_start
    starts = tuple(starts_list)
    if not starts:
        return frozenset()
    if starts[0] < 0 or starts[-1] >= len(text):
        raise CorpusSourceSliceError("Marker offset is outside corpus source text")

    # Each frame is ``(opening token, closing token, opening offset,
    # continuation kind)``. Quotes and brackets share one stack so crossed
    # constructs remain fail-closed. The final field is set only for a
    # positively identified repeated quotation opener.
    delimiter_stack = _IndexedDelimiterStack()
    same_pair_run_lengths: list[int] = []
    pair_counts: dict[tuple[str, str], int] = {}
    closing_counts: dict[str, int] = {}
    enclosed: set[int] = set()
    marker_index = 0
    quote_pairs = {"“": "”", "‘": "’", "«": "»", "‹": "›"}
    bracket_pairs = {"(": ")", "[": "]", "{": "}"}
    directional_closers = {closing: opening for opening, closing in quote_pairs.items()}
    bracket_closers = {closing: opening for opening, closing in bracket_pairs.items()}
    delimiter_tokens = frozenset(
        (
            *quote_pairs,
            *directional_closers,
            *bracket_pairs,
            *bracket_closers,
            '"',
            "'",
            "`",
        )
    )
    directional_remaining = {token: 0 for token in (*quote_pairs, *directional_closers)}
    symmetric_open_remaining = {'"': 0, "'": 0}
    symmetric_close_remaining = {'"': 0, "'": 0}
    prepass_backslash_run = 0
    delimiter_token_count = 0
    for character_index, character in enumerate(text):
        if character == "\\":
            prepass_backslash_run += 1
            continue
        escaped = prepass_backslash_run % 2 == 1
        lexical_previous_index = character_index - prepass_backslash_run - 1
        previous = text[lexical_previous_index] if lexical_previous_index >= 0 else ""
        following = text[character_index + 1] if character_index + 1 < len(text) else ""
        prepass_backslash_run = 0
        if character in delimiter_tokens and not escaped:
            delimiter_token_count += 1
            if delimiter_token_count > _MAX_DELIMITER_REPLAY_TOKENS:
                raise CorpusSourceSliceError(
                    "Delimiter replay exceeds the bounded token limit of "
                    f"{_MAX_DELIMITER_REPLAY_TOKENS}"
                )
        if character == "’" and previous.isalnum() and following.isalnum():
            continue
        if character in directional_remaining and not escaped:
            directional_remaining[character] += 1
        if character in symmetric_open_remaining and not escaped:
            is_word_apostrophe = (
                character == "'" and previous.isalnum() and following.isalnum()
            )
            if is_word_apostrophe:
                continue
            opening_context = (
                not previous or previous.isspace() or previous in "([{<:;—–-"
            ) and bool(following and not following.isspace())
            closing_context = bool(previous and not previous.isspace()) and (
                not following or following.isspace() or following in ".,;:!?)]}>—–-"
            )
            if opening_context:
                symmetric_open_remaining[character] += 1
            elif closing_context:
                symmetric_close_remaining[character] += 1
    malformed_delimiters = False

    def check_nesting_depth() -> None:
        if len(delimiter_stack) >= _MAX_DELIMITER_NESTING_DEPTH:
            raise CorpusSourceSliceError(
                "Corpus delimiter nesting exceeds the bounded depth limit of "
                f"{_MAX_DELIMITER_NESTING_DEPTH}"
            )

    def push(
        opening: str,
        closing: str,
        offset: int,
        continuation_kind: str | None = None,
    ) -> None:
        check_nesting_depth()
        pair = (opening, closing)
        same_pair_run_lengths.append(
            same_pair_run_lengths[-1] + 1
            if delimiter_stack and delimiter_stack[-1][:2] == pair
            else 1
        )
        delimiter_stack.append((opening, closing, offset, continuation_kind))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        closing_counts[closing] = closing_counts.get(closing, 0) + 1

    def pop_frame() -> tuple[str, str, int, str | None]:
        frame = delimiter_stack.pop()
        same_pair_run_lengths.pop()
        pair = frame[:2]
        remaining_pair_count = pair_counts[pair] - 1
        if remaining_pair_count:
            pair_counts[pair] = remaining_pair_count
        else:
            del pair_counts[pair]
        closing = frame[1]
        remaining_closing_count = closing_counts[closing] - 1
        if remaining_closing_count:
            closing_counts[closing] = remaining_closing_count
        else:
            del closing_counts[closing]
        return frame

    def repeated_directional_continuation_kind(
        opening: str,
        offset: int,
    ) -> str | None:
        if not delimiter_stack or delimiter_stack[-1][:2] != (
            opening,
            quote_pairs[opening],
        ):
            return None
        prefix = text[max(0, offset - 256) : offset]
        if re.search(r"\n[ \t]*$", prefix):
            return "line"
        outer_offset = delimiter_stack[-1][2]
        outer_prefix = text[max(0, outer_offset - 128) : outer_offset]
        previous_nonspace = prefix.rstrip()[-1:] if prefix.rstrip() else ""
        following_character = text[offset + 1 : offset + 2]
        if (
            re.search(r":\s*$", outer_prefix)
            and previous_nonspace in {".", "!", "?"}
            and following_character.isupper()
        ):
            return "colon_sentence"
        return None

    def directional_close_is_terminal(offset: int, continuation_kind: str) -> bool:
        tail = text[offset + 1 : offset + 257]
        if continuation_kind == "colon_sentence":
            return (
                re.match(
                    r"^;[ \t]*(?:and|or)\b[ \t]*\r?\n[ \t]*\r?\n",
                    tail,
                    flags=re.IGNORECASE,
                )
                is not None
            )
        return (
            re.match(
                r"^(?:[.,;:!?][ \t]*)?(?:(?:and|or)\b[ \t]*)?"
                r"(?:\r?\n[ \t]*\r?\n|$)",
                tail,
                flags=re.IGNORECASE,
            )
            is not None
        )

    def mark_has_positive_internal_apostrophe_or_measurement_evidence(
        character: str,
        offset: int,
        previous: str,
        following: str,
    ) -> bool:
        if delimiter_stack:
            opening_offset = delimiter_stack[-1][2]
            opening_prefix = text[max(0, opening_offset - 128) : opening_offset]
            # A quote immediately introduced as a term, word, label, or
            # similar named expression is much stronger evidence that this
            # mark closes an ordinary balanced quotation than that it is a
            # possessive apostrophe or measurement mark inside a longer one.
            if re.search(
                r"\b(?:term|word|label|phrase|name|expression)\b"
                r"[\s,;:=—–()\[\]-]*"
                r"(?:(?:the|a|an)\s+)?$",
                opening_prefix,
                flags=re.IGNORECASE,
            ):
                return False
        if character in {"’", "'"}:
            if following.isdigit():
                return True
            preceding_word_match = re.search(
                r"([A-Za-z]+)$",
                text[max(0, offset - 64) : offset],
            )
            following_word_match = re.match(
                r"^[\s,;:—–-]*([A-Za-z]+)",
                text[offset + 1 : offset + 65],
            )
            if preceding_word_match is None or following_word_match is None:
                return False
            following_word = following_word_match.group(1).casefold()
            quote_following_verbs = {
                "applies",
                "are",
                "denotes",
                "has",
                "includes",
                "is",
                "means",
                "refers",
                "shall",
                "was",
            }
            return (
                preceding_word_match.group(1).casefold().endswith("s")
                and following_word not in quote_following_verbs
            )
        if character == '"' and previous.isdigit() and delimiter_stack:
            opening_offset = delimiter_stack[-1][2]
            quoted_prefix = text[
                max(
                    opening_offset + 1, offset - _MAX_INTERNAL_MARK_EVIDENCE_CHARS
                ) : offset
            ]
            return (
                re.search(
                    r"(?:^|\s)(?:a|an)\s+\d+(?:\.\d+)?$",
                    quoted_prefix,
                    flags=re.IGNORECASE,
                )
                is not None
            )
        return False

    index = 0
    backslash_run = 0
    while index < len(text):
        character = text[index]
        while marker_index < len(starts) and starts[marker_index] == index:
            editorial_bracket_only = (
                not malformed_delimiters
                and len(delimiter_stack) == 1
                and delimiter_stack[0] == ("[", "]", index - 1, None)
                and (index < 2 or text[index - 2] == "\n")
            )
            if (malformed_delimiters or delimiter_stack) and not editorial_bracket_only:
                enclosed.add(index)
            marker_index += 1
        if marker_index >= len(starts) and index >= starts[-1]:
            break

        following = text[index + 1] if index + 1 < len(text) else ""
        if character == "\\":
            backslash_run += 1
            index += 1
            continue
        preceding_backslashes = backslash_run
        escaped = preceding_backslashes % 2 == 1
        backslash_run = 0
        lexical_previous_index = index - preceding_backslashes - 1
        previous = text[lexical_previous_index] if lexical_previous_index >= 0 else ""

        directional_word_apostrophe = (
            character == "’" and previous.isalnum() and following.isalnum()
        )
        if (
            character in directional_remaining
            and not escaped
            and not directional_word_apostrophe
        ):
            directional_remaining[character] -= 1

        if character == "`" and not escaped:
            run_end = index + 1
            while run_end < len(text) and text[run_end] == "`":
                run_end += 1
            token = text[index:run_end]
            if delimiter_stack and delimiter_stack[-1][1] == token:
                pop_frame()
            else:
                push(token, token, index)
            index = run_end
            continue

        paired_close = quote_pairs.get(character)
        if paired_close is not None and not escaped:
            push(
                character,
                paired_close,
                index,
                repeated_directional_continuation_kind(character, index),
            )
            index += 1
            continue

        directional_opening = directional_closers.get(character)
        if directional_opening is not None and not escaped:
            if directional_word_apostrophe:
                index += 1
                continue
            if delimiter_stack and delimiter_stack[-1][1] == character:
                same_frame_count = pair_counts.get(
                    (directional_opening, character),
                    0,
                )
                future_surplus_before_pop = (
                    directional_remaining[character]
                    - directional_remaining[directional_opening]
                )
                if same_frame_count == 1 and (
                    future_surplus_before_pop >= 1
                    or mark_has_positive_internal_apostrophe_or_measurement_evidence(
                        character,
                        index,
                        previous,
                        following,
                    )
                ):
                    index += 1
                    continue
                (
                    popped_opening,
                    _popped_closing,
                    _popped_offset,
                    continuation_kind,
                ) = pop_frame()
                if popped_opening == directional_opening:
                    same_frames = (
                        same_pair_run_lengths[-1]
                        if delimiter_stack
                        and delimiter_stack[-1][:2] == (directional_opening, character)
                        else 0
                    )
                    future_surplus = (
                        directional_remaining[character]
                        - directional_remaining[directional_opening]
                    )
                    unmatched_same_frames = max(
                        0,
                        same_frames - max(0, future_surplus),
                    )
                    if continuation_kind is not None and directional_close_is_terminal(
                        index,
                        continuation_kind,
                    ):
                        for _unused in range(unmatched_same_frames):
                            pop_frame()
            elif closing_counts.get(character, 0):
                malformed_delimiters = True
            index += 1
            continue

        if character in {'"', "'"} and not escaped:
            is_word_apostrophe = (
                character == "'" and previous.isalnum() and following.isalnum()
            )
            if not is_word_apostrophe:
                opening_context = (
                    not previous or previous.isspace() or previous in "([{<:;—–-"
                ) and bool(following and not following.isspace())
                closing_context = bool(previous and not previous.isspace()) and (
                    not following or following.isspace() or following in ".,;:!?)]}>—–-"
                )
                if opening_context:
                    symmetric_open_remaining[character] -= 1
                elif closing_context:
                    symmetric_close_remaining[character] -= 1
                if delimiter_stack and delimiter_stack[-1][1] == character:
                    same_frame_count = pair_counts.get((character, character), 0)
                    future_surplus = (
                        symmetric_close_remaining[character]
                        - symmetric_open_remaining[character]
                    )
                    ambiguous_internal_mark = (
                        closing_context
                        and same_frame_count == 1
                        and (
                            future_surplus >= 1
                            or mark_has_positive_internal_apostrophe_or_measurement_evidence(
                                character,
                                index,
                                previous,
                                following,
                            )
                        )
                    )
                    if opening_context and not closing_context:
                        push(character, character, index)
                    elif ambiguous_internal_mark:
                        pass
                    else:
                        pop_frame()
                elif closing_context and closing_counts.get(character, 0):
                    malformed_delimiters = True
                elif opening_context:
                    push(character, character, index)
            index += 1
            continue

        if character in bracket_pairs and not escaped:
            push(character, bracket_pairs[character], index)
            index += 1
            continue

        bracket_opening = bracket_closers.get(character)
        if bracket_opening is not None and not escaped:
            if delimiter_stack and delimiter_stack[-1][1] == character:
                pop_frame()
            elif closing_counts.get(character, 0):
                malformed_delimiters = True
            index += 1
            continue

        index += 1

    return frozenset(enclosed)


def _marker_is_within_quote_or_bracket(text: str, marker_start: int) -> bool:
    return marker_start in _delimiter_enclosed_marker_starts(text, (marker_start,))


def _parenthetical_marker_has_strong_boundary(text: str, marker_start: int) -> bool:
    """Return whether a marker begins a line or follows its line-leading parent."""

    prefix = _bounded_structural_prefix(text, marker_start)
    return (
        re.search(r"\n\s*\[?$", prefix) is not None
        or re.search(
            r"\n\s*\[?(?:\([A-Za-z0-9]+(?:\.[0-9]+)*\)\s*)+$",
            prefix,
        )
        is not None
    )


def _parenthetical_marker_has_paragraph_boundary(
    text: str,
    marker_start: int,
) -> bool:
    prefix = _bounded_structural_prefix(text, marker_start)
    return re.search(r"\n[ \t]*\n[ \t]*\[?$", prefix) is not None


def _bounded_structural_prefix(text: str, marker_start: int) -> str:
    """Return bounded context without inventing a start-of-line boundary."""

    window_start = max(0, marker_start - 60)
    prefix = text[window_start:marker_start]
    if window_start == 0 or text[window_start - 1] == "\n":
        return f"\n{prefix}"
    return prefix


def _parenthetical_marker_is_in_reference_list(prefix: str) -> bool:
    segment = re.split(r"(?:[.;]\s+|\n+)", prefix)[-1]
    if not re.search(
        rf"\b{_NONSTRUCTURAL_PARENTHETICAL_REFERENCE_LABEL}\b",
        segment,
        flags=re.IGNORECASE,
    ):
        return False
    if not re.search(r"\([A-Za-z0-9]+\)", segment):
        return False
    return (
        re.search(
            r"(?:,\s*|\b(?:or|and|through|to)\s+|[-–—]\s*)$",
            segment,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _record_scope(
    record: dict[str, Any],
    *,
    path: Path,
    line_number: int,
) -> ReleaseScope:
    jurisdiction = _row_string(
        record,
        "jurisdiction",
        path,
        line_number,
    )
    if "doc_type" in record:
        raise CorpusRowStructureError(
            "legacy-document-class",
            f"Legacy doc_type is not allowed at {path}:{line_number}; "
            "use document_class",
        )
    raw_document_class = record.get("document_class")
    try:
        document_class = _clean_string_value(
            raw_document_class,
            label=f"document_class at {path}:{line_number}",
        )
    except CorpusResolutionError as exc:
        raise CorpusRowStructureError("missing-document-class", str(exc)) from exc
    version = _row_string(
        record,
        "version",
        path,
        line_number,
    )
    return ReleaseScope(jurisdiction, document_class, version)


def _validate_release_row_metadata(
    record: dict[str, Any], *, path: Path, line_number: int
) -> None:
    for key in ("id", "source_path", "source_as_of", "expression_date"):
        try:
            _clean_string_value(record.get(key), label=f"{key} at {path}:{line_number}")
        except CorpusResolutionError as exc:
            raise CorpusRowStructureError(
                "missing-release-metadata",
                f"Active release row is missing required metadata: {exc}",
            ) from exc


def _repository_root_for_provisions(provisions_root: Path) -> Path:
    if (
        provisions_root.parent.name == "corpus"
        and provisions_root.parent.parent.name == "data"
    ):
        return provisions_root.parent.parent.parent
    return provisions_root.parent


def _normalize_citation_path(identifier: str) -> str:
    if not isinstance(identifier, str):
        raise InvalidCorpusCitationError("Corpus citation path must be a string")
    if len(identifier) > MAX_CORPUS_CITATION_LENGTH:
        raise InvalidCorpusCitationError(
            "Corpus citation path exceeds the maximum length of "
            f"{MAX_CORPUS_CITATION_LENGTH} characters"
        )
    normalized = identifier.strip().strip("/")
    parts = normalized.split("/") if normalized else []
    if len(parts) > MAX_CORPUS_CITATION_SEGMENTS:
        raise InvalidCorpusCitationError(
            "Corpus citation path exceeds the maximum segment count of "
            f"{MAX_CORPUS_CITATION_SEGMENTS}"
        )
    if len(parts) < 2:
        raise InvalidCorpusCitationError(
            f"Corpus citation path must have at least two segments: {identifier!r}"
        )
    try:
        for part in parts:
            _validate_safe_segment(part, label="citation path segment")
    except CorpusResolutionError as exc:
        raise InvalidCorpusCitationError(str(exc)) from exc
    if parts[1] not in _DOCUMENT_CLASSES:
        raise InvalidCorpusCitationError(
            f"Unsupported corpus document class {parts[1]!r} in {identifier!r}"
        )
    if _CITATION_JURISDICTION_RE.fullmatch(parts[0]) is None:
        raise InvalidCorpusCitationError(
            f"Invalid corpus jurisdiction {parts[0]!r} in {identifier!r}"
        )
    for part in parts[2:]:
        if _CITATION_HIERARCHY_SEGMENT_RE.fullmatch(part) is None:
            raise InvalidCorpusCitationError(
                f"Invalid corpus citation path segment {part!r} in {identifier!r}"
            )
    return "/".join(parts)


def _validate_safe_segment(value: str, *, label: str) -> None:
    if (
        len(value) > MAX_CORPUS_CITATION_SEGMENT_LENGTH
        or value in {"", ".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or "*" in value
        or "%" in value
    ):
        raise CorpusResolutionError(f"Unsafe {label}: {value!r}")


def validate_corpus_release_name(value: Any) -> str:
    """Return one canonical immutable release name or reject it."""

    try:
        release_name = _clean_string_value(value, label="corpus release name")
        _validate_safe_segment(release_name, label="corpus release name")
    except CorpusResolutionError as exc:
        raise InvalidCorpusReleaseError(str(exc)) from exc
    if (
        len(release_name) > 128
        or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", release_name) is None
    ):
        raise InvalidCorpusReleaseError(
            "Corpus release names must be at most 128 characters and contain "
            "lowercase alphanumeric segments separated by single hyphens"
        )
    if release_name == "current":
        raise InvalidCorpusReleaseError(
            "Corpus release name 'current' is reserved; configure an immutable "
            "named release"
        )
    return release_name


def _safe_directory(root: Path, candidate: Path, *, label: str) -> Path | None:
    raw_root = Path(os.path.abspath(root))
    raw_candidate = Path(os.path.abspath(candidate))
    if raw_root.is_symlink():
        raise UnsafeCorpusPathError(f"{label} root is a symlink: {raw_root}")
    try:
        relative = raw_candidate.relative_to(raw_root)
    except ValueError as exc:
        raise UnsafeCorpusPathError(
            f"{label} is outside {raw_root}: {candidate}"
        ) from exc
    cursor = raw_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise UnsafeCorpusPathError(f"{label} contains a symlink: {cursor}")
    if not cursor.exists():
        return None
    resolved_root = raw_root.resolve(strict=True)
    resolved = cursor.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise UnsafeCorpusPathError(
            f"{label} escapes {resolved_root}: {cursor}"
        ) from exc
    if not resolved.is_dir():
        raise UnsafeCorpusPathError(f"{label} is not a directory: {cursor}")
    return resolved


def _safe_file(
    root: Path,
    candidate: Path,
    *,
    label: str,
    max_bytes: int,
) -> Path | None:
    parent = _safe_directory(root, candidate.parent, label=f"{label} parent")
    if parent is None:
        return None
    raw_candidate = parent / candidate.name
    if raw_candidate.is_symlink():
        raise UnsafeCorpusPathError(f"{label} is a symlink: {raw_candidate}")
    if not raw_candidate.exists():
        return None
    resolved_root = root.resolve(strict=True)
    resolved = raw_candidate.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise UnsafeCorpusPathError(
            f"{label} escapes {resolved_root}: {raw_candidate}"
        ) from exc
    if not resolved.is_file():
        raise UnsafeCorpusPathError(f"{label} is not a regular file: {raw_candidate}")
    if resolved.stat().st_size > max_bytes:
        raise UnsafeCorpusPathError(
            f"{label} exceeds the {max_bytes}-byte safety limit: {raw_candidate}"
        )
    return resolved


def read_bounded_regular_file(
    root: Path,
    candidate: Path,
    *,
    label: str,
    max_bytes: int,
    required_mode: int | None = None,
) -> bytes:
    """Atomically open a contained regular file without following symlinks.

    Each path component is opened relative to the already validated parent
    descriptor. This closes the check-then-open race where an attacker swaps a
    provision or selector for an out-of-root symlink after ``_safe_file``.
    """

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise UnsafeCorpusPathError(f"{label} cannot be opened safely on this platform")
    raw_root = Path(os.path.abspath(root))
    raw_candidate = Path(os.path.abspath(candidate))
    try:
        relative = raw_candidate.relative_to(raw_root)
    except ValueError as exc:
        raise UnsafeCorpusPathError(
            f"{label} is outside {raw_root}: {candidate}"
        ) from exc
    if not relative.parts:
        raise UnsafeCorpusPathError(f"{label} is not a file: {candidate}")

    base_flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    nonblocking = getattr(os, "O_NONBLOCK", 0)
    descriptors: list[int] = []
    file_descriptor: int | None = None
    try:
        parent_descriptor = os.open(raw_root, base_flags | directory)
        descriptors.append(parent_descriptor)
        for part in relative.parts[:-1]:
            parent_descriptor = os.open(
                part,
                base_flags | directory,
                dir_fd=parent_descriptor,
            )
            descriptors.append(parent_descriptor)
        file_descriptor = os.open(
            relative.parts[-1],
            base_flags | nonblocking,
            dir_fd=parent_descriptor,
        )
        file_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise UnsafeCorpusPathError(f"{label} is not a regular file: {candidate}")
        actual_mode = stat.S_IMODE(file_stat.st_mode)
        if required_mode is not None and actual_mode != required_mode:
            raise UnsafeCorpusPathError(
                f"{label} must have mode {required_mode:04o}, found "
                f"{actual_mode:04o}: {candidate}"
            )
        if file_stat.st_size > max_bytes:
            raise UnsafeCorpusPathError(
                f"{label} exceeds the {max_bytes}-byte safety limit: {candidate}"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(file_descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise UnsafeCorpusPathError(
                f"{label} exceeds the {max_bytes}-byte safety limit: {candidate}"
            )
        return raw
    except UnsafeCorpusPathError:
        raise
    except OSError as exc:
        raise UnsafeCorpusPathError(
            f"Could not safely open {label}: {candidate}"
        ) from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _record_body(
    record: dict[str, Any],
    *,
    path: Path,
    line_number: int,
) -> str | None:
    if "text" in record:
        raise CorpusRowStructureError(
            "legacy-body-field",
            f"Legacy text body is not allowed at {path}:{line_number}; use body",
        )
    value = record.get("body")
    if isinstance(value, str) and value.strip():
        return value
    return None


def _record_status(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    status = metadata.get("status")
    return status if status == "repealed" else None


def _record_source_history(record: dict[str, Any]) -> tuple[str, ...]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return ()
    history = metadata.get("source_history")
    if not isinstance(history, list):
        return ()
    return tuple(
        entry.strip() for entry in history if isinstance(entry, str) and entry.strip()
    )


def split_proof_evidence_text(source_text: str) -> tuple[str, ...]:
    """Split proof evidence without permitting matches across source records."""

    return tuple(str(source_text).split(PROOF_EVIDENCE_SEGMENT_SEPARATOR))


def _clean_string_value(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CorpusResolutionError(f"{label} must be a non-empty trimmed string")
    return value


def _row_string(
    record: dict[str, Any],
    key: str,
    path: Path,
    line_number: int,
) -> str:
    try:
        return _clean_string_value(
            record.get(key), label=f"{key} at {path}:{line_number}"
        )
    except CorpusResolutionError as exc:
        reason = {
            "jurisdiction": "missing-jurisdiction",
            "version": "missing-version",
        }.get(key, "resolution-error")
        raise CorpusRowStructureError(reason, str(exc)) from exc


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CorpusResolutionError(
            f"Optional corpus ordering metadata must be a non-negative integer, got "
            f"{value!r}"
        )
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
