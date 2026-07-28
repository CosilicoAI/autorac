"""Pure planning helpers for authenticated canonical RuleSpec path migrations.

This module deliberately cannot sign or install files.  It turns a minimal,
exact-schema move plan into deterministic path and durable-reference
replacements.  The CLI owns provenance verification and the journaled install.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

PLAN_SCHEMA: Final = "axiom-encode/rulespec-path-migration-plan/v1"
RECEIPT_SCHEMA: Final = "axiom-encode/rulespec-path-migration-receipt/v1"
MIGRATION_TOOL: Final = "axiom-encode migrate-rulespec-paths"
RECEIPT_DIR: Final = Path(".axiom") / "path-migrations"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_JURISDICTION = re.compile(r"[a-z]{2}(?:-[a-z0-9_]+)*")
_SAFE_COMPONENT = re.compile(r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?")
_DURABLE_REFERENCE_CHARACTER = r"A-Za-z0-9._:/-"
_NON_ASCII_DASHES = str.maketrans(
    {
        "\N{HYPHEN}": "-",
        "\N{NON-BREAKING HYPHEN}": "-",
        "\N{FIGURE DASH}": "-",
        "\N{EN DASH}": "-",
        "\N{EM DASH}": "-",
        "\N{HORIZONTAL BAR}": "-",
        "\N{SMALL EM DASH}": "-",
        "\N{SMALL HYPHEN-MINUS}": "-",
        "\N{FULLWIDTH HYPHEN-MINUS}": "-",
    }
)


class PathMigrationPlanError(ValueError):
    """The caller supplied a noncanonical or over-broad migration plan."""


@dataclass(frozen=True, slots=True)
class PlannedMove:
    """One primary RuleSpec path move admitted by the exact plan."""

    source: Path
    destination: Path


@dataclass(frozen=True, slots=True)
class PathMigrationPlan:
    """Parsed and canonically serialized path-migration authority."""

    base_commit: str
    moves: tuple[PlannedMove, ...]
    payload: Mapping[str, object]
    canonical_bytes: bytes
    sha256: str


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _safe_repo_path(raw: object, *, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PathMigrationPlanError(f"{field} must be a non-empty string")
    path = Path(raw)
    if (
        path.is_absolute()
        or path.as_posix() != raw
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PathMigrationPlanError(f"{field} is not a canonical repository path")
    return path


def _normalize_component(component: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", component).translate(_NON_ASCII_DASHES)
    normalized = normalized.replace(":", "/")
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = normalized.lower()
    pieces = tuple(piece for piece in normalized.split("/") if piece)
    if not pieces or any(_SAFE_COMPONENT.fullmatch(piece) is None for piece in pieces):
        raise PathMigrationPlanError(
            f"path component {component!r} cannot be normalized safely"
        )
    return pieces


def canonical_destination(source: Path) -> Path:
    """Return the sole engine-safe identity allowed for ``source``."""

    parts: list[str] = []
    for component in source.parts:
        parts.extend(_normalize_component(component))
    return Path(*parts)


def _validate_primary_path(path: Path, *, field: str) -> None:
    if len(path.parts) < 3:
        raise PathMigrationPlanError(
            f"{field} must be jurisdiction-prefixed and under a content root"
        )
    if _JURISDICTION.fullmatch(path.parts[0]) is None:
        raise PathMigrationPlanError(f"{field} has an invalid jurisdiction root")
    if path.parts[1] not in {"forms", "policies", "regulations", "statutes"}:
        raise PathMigrationPlanError(f"{field} is outside a protected RuleSpec root")
    if path.suffix != ".yaml" or path.name.endswith(".test.yaml"):
        raise PathMigrationPlanError(f"{field} must name a primary .yaml RuleSpec")


def load_plan_bytes(raw: bytes) -> PathMigrationPlan:
    """Parse one exact, canonical JSON migration plan."""

    if len(raw) > 1024 * 1024:
        raise PathMigrationPlanError("migration plan exceeds 1 MiB")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PathMigrationPlanError("migration plan is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise PathMigrationPlanError("migration plan must be a JSON object")
    if set(payload) != {"schema_version", "base_commit", "moves"}:
        raise PathMigrationPlanError(
            "migration plan must contain exactly schema_version, base_commit, moves"
        )
    if payload.get("schema_version") != PLAN_SCHEMA:
        raise PathMigrationPlanError("migration plan schema_version is unsupported")
    base_commit = payload.get("base_commit")
    # This repository still uses SHA-1 object IDs.  Refuse abbreviations and
    # algorithm-ambiguous strings rather than silently resolving a prefix.
    if (
        not isinstance(base_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", base_commit) is None
    ):
        raise PathMigrationPlanError(
            "migration plan base_commit must be one full Git object ID"
        )
    raw_moves = payload.get("moves")
    if not isinstance(raw_moves, list) or not raw_moves:
        raise PathMigrationPlanError("migration plan moves must be a non-empty list")
    if len(raw_moves) > 256:
        raise PathMigrationPlanError("migration plan contains too many moves")

    moves: list[PlannedMove] = []
    sources: set[str] = set()
    destinations: set[str] = set()
    for index, item in enumerate(raw_moves):
        field = f"moves[{index}]"
        if not isinstance(item, dict) or set(item) != {"from", "to"}:
            raise PathMigrationPlanError(f"{field} must contain exactly from and to")
        source = _safe_repo_path(item.get("from"), field=f"{field}.from")
        destination = _safe_repo_path(item.get("to"), field=f"{field}.to")
        _validate_primary_path(source, field=f"{field}.from")
        _validate_primary_path(destination, field=f"{field}.to")
        if destination != canonical_destination(source):
            raise PathMigrationPlanError(
                f"{field}.to is not the unique canonical normalization of "
                f"{source.as_posix()}"
            )
        if source == destination:
            raise PathMigrationPlanError(f"{field} is a no-op")
        if source.as_posix() in sources:
            raise PathMigrationPlanError(f"{field}.from is duplicated")
        if destination.as_posix() in destinations:
            raise PathMigrationPlanError(f"{field}.to collides with another move")
        sources.add(source.as_posix())
        destinations.add(destination.as_posix())
        moves.append(PlannedMove(source=source, destination=destination))

    if sources & destinations:
        raise PathMigrationPlanError(
            "migration plan cannot chain or swap paths in one transaction"
        )
    canonical_bytes = _canonical_json_bytes(payload)
    return PathMigrationPlan(
        base_commit=base_commit,
        moves=tuple(moves),
        payload=payload,
        canonical_bytes=canonical_bytes,
        sha256=hashlib.sha256(canonical_bytes).hexdigest(),
    )


def companion_path(primary: Path) -> Path:
    """Return the mechanically coupled companion test path."""

    return primary.with_name(f"{primary.stem}.test.yaml")


def rulespec_identity(path: Path) -> str:
    """Convert a protected RuleSpec path to its durable module identity."""

    _validate_primary_path(path, field="path")
    without_suffix = path.with_suffix("")
    jurisdiction, content_root, *remainder = without_suffix.parts
    return f"{jurisdiction}:{content_root}/{'/'.join(remainder)}"


def exact_reference_replacements(
    moves: Sequence[PlannedMove],
    *,
    existing_companions: set[Path],
) -> dict[str, str]:
    """Return every exact durable identity/path rewrite implied by ``moves``."""

    replacements: dict[str, str] = {}

    def add(old: str, new: str) -> None:
        prior = replacements.get(old)
        if prior is not None and prior != new:
            raise PathMigrationPlanError(f"conflicting replacement for {old!r}")
        replacements[old] = new

    for move in moves:
        add(
            move.source.as_posix(),
            move.destination.as_posix(),
        )
        add(
            rulespec_identity(move.source),
            rulespec_identity(move.destination),
        )
        old_test = companion_path(move.source)
        if old_test in existing_companions:
            add(old_test.as_posix(), companion_path(move.destination).as_posix())
    return replacements


def rewrite_exact_references(
    raw: bytes,
    replacements: Mapping[str, str],
) -> tuple[bytes, tuple[dict[str, object], ...]]:
    """Apply exact UTF-8 token rewrites and return an auditable count record."""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        matching = [old for old in replacements if old.encode("utf-8") in raw]
        if matching:
            raise PathMigrationPlanError(
                "a durable reference occurs in a non-UTF-8 tracked file"
            ) from exc
        return raw, ()

    counts: list[dict[str, object]] = []
    for old in sorted(replacements, key=lambda item: (-len(item), item)):
        pattern = re.compile(
            rf"(?<![{_DURABLE_REFERENCE_CHARACTER}])"
            rf"{re.escape(old)}"
            rf"(?![{_DURABLE_REFERENCE_CHARACTER}])"
        )
        count = len(pattern.findall(text))
        if not count:
            continue
        new = replacements[old]
        text = pattern.sub(lambda _match: new, text)
        counts.append({"from": old, "to": new, "count": count})
    return text.encode("utf-8"), tuple(counts)
