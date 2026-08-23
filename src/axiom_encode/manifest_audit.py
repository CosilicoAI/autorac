"""Audit every committed encoding manifest against the file it attests.

The apply guard (``guard-generated``) asks *"is there a matching attestation
for this changed file?"*.  One good record answers that question, and the guard
says nothing about the other records committed alongside it.  So a rule can be
re-generated, gain a fresh manifest in a new location, and leave its previous
manifest behind asserting a hash the file no longer has — with CI green.

This module asks the auditor's question instead: **does every committed
attestation match its file?**  Two invariants implement it:

``every attestation matches``
    Every ``applied_files[]`` entry in every committed manifest must resolve to
    a tracked regular file whose sha256 equals the attested value (or, for a
    deletion marker, to no filesystem entry at all).

``at most one live manifest per rule path``
    Two live manifests attesting the same file is the ambiguity that let the
    first invariant rot unnoticed: the guard found the good record and never
    looked at the bad one.  Superseded records are pruned, or their superseded
    claims are retired in place; they are not accumulated.

The audit is deliberately **schema- and signature-agnostic**.  It hashes files
and compares bytes.  Signature and schema validity are enforced by the apply
guard; re-checking them here would make the audit inherit the guard's blind
spot — the current verifier accepts only the newest manifest schema, while
whole repos of live records predate it, which is precisely where stale claims
accumulate.  Agnostic does not mean permissive: a ``.json`` under any manifest
tree that is not a well-formed, non-empty, internally consistent apply manifest
reachable without symlinks fails the audit rather than being skipped, so a
record cannot be blanked, aliased, or linked out of existence.

A claim that is superseded but cannot be deleted — the record also carries the
only attestation for a *sibling* file, so deleting it would destroy a true
record — is **retired in place** by an entry in
``.axiom/retired-manifest-claims.yaml``.  A retired claim is no longer live: it
does not count toward uniqueness and its staleness is not a finding.  An entry
is a reviewed exception and nothing more.  It names one exact ``(manifest,
path, attested sha256)`` claim, verifies nothing, authorizes nothing, and is
accepted only while it is still real: the claim must exist in the manifest
with that exact digest; a still-true claim may be retired only if another
live record attests the file's current content; and a stale claim whose file
has no matching live attestation at all must say so (``unattested: true``).
An entry that stops meeting those conditions fails the audit, so the list is
removal-forced.  It is not growth-capped: additions arrive only in reviewed
diffs (or as the apply transaction's own artifact, alongside the successor
manifest that justifies them), and the file cannot bind an author with write
access — pull-request review does that.  What the audit guarantees is that
every tolerated divergence is enumerated, exact, and visible.

See TheAxiomFoundation/axiom-encode#1282.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from yaml.constructor import ConstructorError

from axiom_encode.constants import (
    RULESPEC_COMPOSITION_SPEC_ROOT,
    RULESPEC_FILESYSTEM_ROOTS,
)

MANIFEST_DIR = PurePosixPath(".axiom") / "encoding-manifests"
RATCHET_RELATIVE_PATH = PurePosixPath(".axiom") / "retired-manifest-claims.yaml"
RATCHET_SCHEMA = "axiom-encode/retired-manifest-claims/v1"
RATCHET_TOP_KEY = "retired_claims"

_JURISDICTION_DIR_RE = re.compile(r"[a-z]{2}(?:-[a-z0-9_]+)*")
#: Content directories a country-relative key may begin with, beyond the
#: atomic/composition roots (rulespec-us carries ``us-ca/guidance/...``).
_COUNTRY_RELATIVE_ROOTS = frozenset({"guidance"})
MAX_RATCHET_BYTES = 2_000_000
MAX_RATCHET_ENTRIES = 5_000
MAX_MANIFEST_BYTES = 4_000_000
SHA256_LENGTH = 64
_HEX = frozenset("0123456789abcdef")

#: Finding kinds, ordered most to least severe for reporting.
MISMATCH = "mismatch"
MISSING = "missing"
RESURRECTED = "resurrected"
UNSAFE = "unsafe-path"
AMBIGUOUS = "ambiguous"
DUPLICATE = "duplicate"
UNATTESTED = "unattested"
UNREADABLE = "unreadable"
RATCHET_INVALID = "retired-claim-invalid"
RATCHET_UNUSED = "retired-claim-unused"

_FINDING_ORDER = (
    MISMATCH,
    MISSING,
    RESURRECTED,
    UNSAFE,
    AMBIGUOUS,
    DUPLICATE,
    UNATTESTED,
    UNREADABLE,
    RATCHET_INVALID,
    RATCHET_UNUSED,
)


class ManifestAuditError(ValueError):
    """Raised when the audit cannot be performed at all."""


class RatchetSchemaError(ManifestAuditError):
    """Raised when the retired-claims file is malformed."""


@dataclass(frozen=True, slots=True)
class AuditFinding:
    kind: str
    detail: str
    manifest: str | None = None
    path: str | None = None
    #: Set-valued content of the finding (duplicate manifests, ambiguous
    #: candidates) so a caller can tell growth from mere re-rendering.
    members: frozenset[str] = frozenset()

    def __str__(self) -> str:
        where = self.manifest or ""
        if self.path:
            where = f"{where} -> {self.path}" if where else self.path
        return (
            f"[{self.kind}] {where}: {self.detail}"
            if where
            else f"[{self.kind}] {self.detail}"
        )


@dataclass(frozen=True, slots=True)
class Claim:
    """One distinct ``applied_files[]`` claim resolved against the working tree.

    ``manifest_sha256`` is the digest of the manifest bytes this claim was
    parsed from, so a caller that acts on the claim can pin the record it
    actually read rather than re-sampling the file later.
    """

    manifest: str
    manifest_sha256: str
    entry_path: str
    resolved_path: str | None
    attested_sha256: str | None
    deleted: bool
    candidates: tuple[str, ...] = ()
    unsafe_reason: str | None = None


@dataclass
class AuditResult:
    findings: list[AuditFinding] = field(default_factory=list)
    manifests: int = 0
    attestations: int = 0
    matched: int = 0
    retired: int = 0

    @property
    def passed(self) -> bool:
        return not self.findings

    def by_kind(self) -> dict[str, list[AuditFinding]]:
        grouped: dict[str, list[AuditFinding]] = defaultdict(list)
        for finding in self.findings:
            grouped[finding.kind].append(finding)
        return dict(grouped)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "manifests": self.manifests,
            "attestations": self.attestations,
            "matched": self.matched,
            "retired": self.retired,
            "findings": [
                {
                    "kind": finding.kind,
                    "manifest": finding.manifest,
                    "path": finding.path,
                    "detail": finding.detail,
                    "members": sorted(finding.members),
                }
                for finding in self.findings
            ],
        }


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


@dataclass
class CheckoutScan:
    """One walk of the checkout: manifest trees, manifest entries, symlinks."""

    tree_prefixes: list[str] = field(default_factory=list)
    manifest_entries: list[str] = field(default_factory=list)
    symlinks: list[str] = field(default_factory=list)


def scan_checkout(repo_path: Path) -> CheckoutScan:
    """Walk the whole checkout once, never following links, pruning only ``.git``.

    Everything is discovered from one walk so trees, records, and symlinks are
    seen under a single consistent view.  Symlinks are reported wherever they
    occur: a rulespec checkout contains no tracked symlinks, the apply path
    refuses them, and a symlinked ``.axiom`` or ``encoding-manifests`` ancestor
    would otherwise hide an entire tree from ``os.walk(followlinks=False)``
    while the records behind it stayed reachable through the link.  Nothing
    but ``.git`` is pruned — a record parked under ``node_modules/`` or
    ``target/`` inside a tree is still a record.
    """
    repo_path = Path(repo_path)
    scan = CheckoutScan()
    for directory, directory_names, file_names in os.walk(repo_path, followlinks=False):
        directory = Path(directory)
        directory_names[:] = sorted(name for name in directory_names if name != ".git")
        for name in directory_names:
            if (directory / name).is_symlink():
                scan.symlinks.append(
                    (directory / name).relative_to(repo_path).as_posix()
                )
        for name in sorted(file_names):
            path = directory / name
            if path.is_symlink():
                scan.symlinks.append(path.relative_to(repo_path).as_posix())
        # Name matching is case-insensitive: on a case-insensitive filesystem
        # ``.AXIOM`` *is* ``.axiom``, and a record must not disappear from the
        # audit through a metadata-only rename that the guard does not read.
        if directory.name.casefold() == ".axiom" and any(
            name.casefold() == "encoding-manifests" for name in directory_names
        ):
            parent = directory.parent
            scan.tree_prefixes.append(
                "" if parent == repo_path else parent.relative_to(repo_path).as_posix()
            )
        relative_dir = directory.relative_to(repo_path).as_posix()
        if _manifest_tree_index(relative_dir.split("/")) is not None:
            for name in sorted(file_names):
                if name.casefold().endswith(".json"):
                    scan.manifest_entries.append(
                        (directory / name).relative_to(repo_path).as_posix()
                    )
    scan.tree_prefixes = sorted(set(scan.tree_prefixes))
    scan.manifest_entries = sorted(set(scan.manifest_entries))
    scan.symlinks = sorted(set(scan.symlinks))
    return scan


def manifest_trees(repo_path: Path) -> list[str]:
    """Return the prefix of every ``.axiom/encoding-manifests`` tree anywhere.

    A country monorepo carries one tree per jurisdiction subroot
    (``us/.axiom/...``) *and* a checkout-root tree that the #1078 relocation
    writes into; 3,927 of rulespec-us's 5,235 records live in jurisdiction
    trees the existing guard never enumerated.  The walk is unbounded in depth
    so a record parked in an unexpected location is audited rather than
    invisible.
    """
    return scan_checkout(repo_path).tree_prefixes


def jurisdiction_roots(repo_path: Path) -> list[str]:
    """Return top-level directories that hold RuleSpec content.

    Used only to resolve legacy country-relative manifest keys.  Derived from
    the tree rather than from ``repository-structure.yaml`` so the audit works
    on a checkout that predates (or postdates) any given layout gate.
    """
    repo_path = Path(repo_path)
    roots: list[str] = []
    for child in sorted(repo_path.iterdir()):
        if not child.is_dir() or child.is_symlink() or child.name.startswith("."):
            continue
        if any((child / root).is_dir() for root in sorted(RULESPEC_FILESYSTEM_ROOTS)):
            roots.append(child.name)
    return roots


def manifest_paths(repo_path: Path) -> list[str]:
    """Return every ``.json`` entry under any manifest tree, repo-relative.

    Entries are returned whether or not they are regular files; symlinks are
    additionally reported by :func:`scan_checkout`.
    """
    return scan_checkout(repo_path).manifest_entries


_TREE_COMPONENTS = tuple(part.casefold() for part in MANIFEST_DIR.parts)


def _manifest_tree_index(parts: Sequence[str]) -> int | None:
    """Index of the ``.axiom`` component that starts a manifest tree, or None.

    Compared component-wise and case-insensitively; never by slicing a
    case-folded string, whose length can differ from the original's.
    """
    for index in range(len(parts) - len(_TREE_COMPONENTS) + 1):
        window = tuple(part.casefold() for part in parts[index : index + 2])
        if window == _TREE_COMPONENTS:
            return index
    return None


def _tree_prefix(manifest_relpath: str) -> str:
    parts = manifest_relpath.split("/")
    index = _manifest_tree_index(parts)
    if not index:
        return ""
    return "/".join(parts[:index])


def _manifest_subpath_head(manifest_relpath: str) -> str:
    """First path component under the manifest tree, or ``""``."""
    parts = manifest_relpath.split("/")
    index = _manifest_tree_index(parts)
    if index is None:
        return ""
    inner = parts[index + 2 :]
    return inner[0] if len(inner) > 1 else ""


def validate_entry_path(entry_path: str) -> str | None:
    """Return the reason ``entry_path`` is unsafe to resolve, or ``None``.

    A manifest key is data an author committed; without these checks a crafted
    key could point outside the repository (``../``), at an absolute path, or
    at nothing (empty), and the audit would either escape its root or count a
    vacuous claim as matched.
    """
    if not entry_path or not entry_path.strip():
        return "is empty"
    if "\\" in entry_path or "\0" in entry_path:
        return "contains a backslash or NUL"
    if entry_path.startswith("/") or entry_path.startswith("~"):
        return "is not repository-relative"
    # Split on the raw separator: PurePosixPath normalizes "./x" to "x", which
    # would let a dot-prefixed key slip past a parts-based check.
    parts = entry_path.split("/")
    if any(part in ("..", ".") for part in parts):
        return "contains a dot or dot-dot component"
    if any(not part.strip() for part in parts):
        return "contains an empty component"
    return None


class _LinkFreeChecker:
    """Answer 'is this repo-relative path a regular file reached without links?'

    Every component is checked, not just the last — ``us/statutes`` pointing
    at ``../us-co/statutes`` would otherwise let a US claim hash Colorado
    bytes.  Directory verdicts are memoized because a large repo asks about
    thousands of files under a few hundred directories.
    """

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = Path(repo_path)
        self._clean_dirs: dict[str, bool] = {"": True}

    def directory_is_clean(self, relative_dir: str) -> bool:
        cached = self._clean_dirs.get(relative_dir)
        if cached is not None:
            return cached
        parent, _, _name = relative_dir.rpartition("/")
        if not self.directory_is_clean(parent):
            verdict = False
        else:
            absolute = self.repo_path / relative_dir
            try:
                verdict = absolute.is_dir() and not absolute.is_symlink()
            except OSError:
                verdict = False
        self._clean_dirs[relative_dir] = verdict
        return verdict

    def traverses_link(self, relative: str) -> bool:
        parent, _, _name = relative.rpartition("/")
        if not self.directory_is_clean(parent):
            return True
        try:
            return (self.repo_path / relative).is_symlink()
        except OSError:
            return True

    def regular_file(self, relative: str) -> bool:
        if self.traverses_link(relative):
            return False
        try:
            return (self.repo_path / relative).is_file()
        except OSError:
            return False

    def exists_at_all(self, relative: str) -> bool:
        """Any filesystem entry at the path, including a dangling symlink.

        ``lexists`` follows directory symlinks on the way down and stops at
        the final component, so a link *at* the attested path, or a file
        reached through a linked parent, both count as present; a path whose
        parent does not exist does not.
        """
        return os.path.lexists(self.repo_path / relative)


def _head_bound_candidates(
    head: str, entry_path: str, jurisdictions: Sequence[str]
) -> list[str]:
    """The paths a jurisdiction-bound root record's key can denote.

    The base is decided by the key itself, never by probing what exists:

    - a key under the record's own jurisdiction is repo-relative;
    - a key starting with an atomic source root (or ``guidance``) is
      country-relative to that jurisdiction;
    - a key under another jurisdiction (by name or by shape) denotes nothing;
    - a ``programs/`` key may be a checkout-root composition spec
      (rulespec-us) or one under the jurisdiction (gh, ug, et); both are
      returned and the caller reports ambiguity if both exist;
    - anything else is checkout-root metadata the record reconciled
      (``.axiom/toolchain.toml``, ``known-validation-gaps.yaml``, the index).

    Probing order once let a planted ``<juris>/.axiom/index/...`` decoy capture
    a record's claim on the real root index; a fixed base cannot be steered,
    and where two bases are legitimate a decoy produces a finding instead.
    """
    if entry_path == head or entry_path.startswith(head + "/"):
        return [entry_path]
    first = entry_path.split("/", 1)[0]
    if first == RULESPEC_COMPOSITION_SPEC_ROOT:
        return [entry_path, f"{head}/{entry_path}"]
    if first in RULESPEC_FILESYSTEM_ROOTS or first in _COUNTRY_RELATIVE_ROOTS:
        return [f"{head}/{entry_path}"]
    if first in jurisdictions or _JURISDICTION_DIR_RE.fullmatch(first) is not None:
        return []
    return [entry_path]


def resolve_attested_path(
    repo_path: Path,
    manifest_relpath: str,
    entry_path: str,
    *,
    jurisdictions: Sequence[str],
    checker: _LinkFreeChecker | None = None,
) -> tuple[str | None, tuple[str, ...]]:
    """Resolve an ``applied_files[].path`` to a repo-relative file path.

    Manifest keys are not uniform: a jurisdiction-tree manifest keys its files
    country-relative (``statutes/26/32/c/2.yaml`` under ``us/.axiom/...``),
    the relocated checkout-root tree keys them repo-relative
    (``us/statutes/...``), and the pre-relocation checkout-root records key
    them country-relative with no jurisdiction at all.  Getting this wrong
    manufactures phantom findings — resolving rulespec-us with a repo-relative
    reading alone reports 828 files as missing that are simply keyed the older
    way.

    Resolution never crosses a jurisdiction the manifest is bound to, and the
    binding is read from the manifest's own location, not from whatever
    directories happen to exist today (a jurisdiction directory that was
    removed must not silently turn its records into free-floating ones):

    - a manifest under ``<juris>/.axiom/...`` resolves ONLY under that
      jurisdiction;
    - a checkout-root manifest whose subpath head is not a source root is
      bound to that head and resolves only under it (repo-relative or
      country-relative keying);
    - only a legacy checkout-root record whose subpath begins with a source
      root may scan jurisdictions, and then a unique match is accepted while
      multiple matches are reported as ambiguous rather than guessed.
    """
    repo_path = Path(repo_path)
    checker = checker or _LinkFreeChecker(repo_path)
    prefix = _tree_prefix(manifest_relpath)

    if prefix:
        candidate = f"{prefix}/{entry_path}"
        return (candidate if checker.regular_file(candidate) else None), ()

    head = _manifest_subpath_head(manifest_relpath)
    if head and head not in RULESPEC_FILESYSTEM_ROOTS:
        present = tuple(
            candidate
            for candidate in _head_bound_candidates(head, entry_path, jurisdictions)
            if checker.regular_file(candidate)
        )
        if len(present) > 1:
            return None, present
        return (present[0] if present else None), ()

    if checker.regular_file(entry_path):
        return entry_path, ()
    matches = tuple(
        f"{juris}/{entry_path}"
        for juris in jurisdictions
        if checker.regular_file(f"{juris}/{entry_path}")
    )
    if len(matches) == 1:
        return matches[0], ()
    return None, matches


def _file_identity(repo_path: Path, relative: str) -> object:
    """``(st_dev, st_ino)`` of a file, or the path itself when unavailable."""
    try:
        info = os.stat(Path(repo_path) / relative)
    except OSError:
        return relative
    return (info.st_dev, info.st_ino)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Claim collection
# --------------------------------------------------------------------------


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` that refuses duplicate members.

    ``json.loads`` keeps the last duplicate, so ``{"applied_files": [...],
    "applied_files": []}`` would silently replace a record's claims.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def collect_claims(repo_path: Path) -> tuple[list[Claim], list[AuditFinding]]:
    """Parse every committed manifest into distinct resolved claims.

    Fail-closed on shape: a ``.json`` under a manifest tree must be a regular
    file reached without symlinks, parse as a JSON object with no duplicate
    members, and carry a non-empty ``applied_files`` list of well-formed
    entries that do not contradict each other.  Anything else is a finding,
    not a skip.  Identical entries (same path, same digest) are one claim.
    """
    repo_path = Path(repo_path)
    jurisdictions = jurisdiction_roots(repo_path)
    checker = _LinkFreeChecker(repo_path)
    claims: list[Claim] = []
    findings: list[AuditFinding] = []
    scan = scan_checkout(repo_path)
    for link in scan.symlinks:
        findings.append(
            AuditFinding(
                kind=UNSAFE,
                path=link,
                detail=(
                    "is a symlink; a rulespec checkout carries no tracked symlinks, "
                    "and a linked directory can hide a manifest tree from the walk "
                    "while its records stay reachable"
                ),
            )
        )

    def unreadable(manifest: str, detail: str, path: str | None = None) -> None:
        findings.append(
            AuditFinding(kind=UNREADABLE, manifest=manifest, path=path, detail=detail)
        )

    for manifest_relpath in scan.manifest_entries:
        if checker.traverses_link(manifest_relpath):
            unreadable(manifest_relpath, "is a symlink or sits behind one")
            continue
        absolute = repo_path / manifest_relpath
        if not absolute.is_file():
            unreadable(manifest_relpath, "is not a regular file")
            continue
        try:
            if absolute.stat().st_size > MAX_MANIFEST_BYTES:
                unreadable(manifest_relpath, f"exceeds {MAX_MANIFEST_BYTES} bytes")
                continue
            raw = absolute.read_bytes()
            payload = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
            )
        except (OSError, UnicodeError, ValueError) as error:
            unreadable(manifest_relpath, f"cannot be read as JSON ({error})")
            continue
        manifest_digest = hashlib.sha256(raw).hexdigest()
        if not isinstance(payload, dict):
            unreadable(manifest_relpath, "does not contain a JSON object")
            continue
        entries = payload.get("applied_files")
        if not isinstance(entries, list) or not entries:
            unreadable(
                manifest_relpath,
                "has no non-empty applied_files list; a manifest that attests "
                "nothing is a blanked record, not a record",
            )
            continue

        seen: dict[str, tuple[str | None, bool]] = {}
        distinct: list[tuple[str, str | None, bool]] = []
        malformed = False
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                unreadable(manifest_relpath, f"has a malformed applied_files[{index}]")
                malformed = True
                continue
            entry_path = entry["path"]
            deleted = entry.get("deleted") is True
            attested = entry.get("sha256")
            attested = attested if isinstance(attested, str) else None
            if not deleted and attested is None:
                unreadable(manifest_relpath, "has an entry with no sha256", entry_path)
                malformed = True
                continue
            if attested is not None and (
                len(attested) != SHA256_LENGTH or not set(attested) <= _HEX
            ):
                unreadable(
                    manifest_relpath,
                    "has an entry whose sha256 is not a lowercase 64-character hex digest",
                    entry_path,
                )
                malformed = True
                continue
            signature = (attested, deleted)
            previous = seen.get(entry_path)
            if previous is None:
                seen[entry_path] = signature
                distinct.append((entry_path, attested, deleted))
            elif previous != signature:
                unreadable(
                    manifest_relpath,
                    "makes two different claims about one path "
                    "(self-contradictory record)",
                    entry_path,
                )
                malformed = True
        if malformed:
            # Keep the well-formed claims so the audit still checks them, but
            # the structural finding above already fails the repo and blocks
            # retirement of this record.
            pass

        for entry_path, attested, deleted in distinct:
            unsafe = validate_entry_path(entry_path)
            if unsafe is not None:
                claims.append(
                    Claim(
                        manifest=manifest_relpath,
                        manifest_sha256=manifest_digest,
                        entry_path=entry_path,
                        resolved_path=None,
                        attested_sha256=attested,
                        deleted=deleted,
                        unsafe_reason=unsafe,
                    )
                )
                continue
            if deleted:
                # A deletion marker is satisfied only by the total absence of
                # any entry — a symlink, directory, or unreachable path at the
                # attested location is a resurrection, not a match.
                present, ambiguous = _deletion_target_present(
                    repo_path,
                    manifest_relpath,
                    entry_path,
                    jurisdictions=jurisdictions,
                    checker=checker,
                )
                claims.append(
                    Claim(
                        manifest=manifest_relpath,
                        manifest_sha256=manifest_digest,
                        entry_path=entry_path,
                        resolved_path=present,
                        attested_sha256=None,
                        deleted=True,
                        candidates=ambiguous,
                    )
                )
                continue
            resolved, candidates = resolve_attested_path(
                repo_path,
                manifest_relpath,
                entry_path,
                jurisdictions=jurisdictions,
                checker=checker,
            )
            claims.append(
                Claim(
                    manifest=manifest_relpath,
                    manifest_sha256=manifest_digest,
                    entry_path=entry_path,
                    resolved_path=resolved,
                    attested_sha256=attested,
                    deleted=False,
                    candidates=candidates,
                )
            )

    # Two raw keys of one record can resolve to one file (``statutes/f.yaml``
    # and ``us/statutes/f.yaml`` under a legacy root record).  Differing
    # digests for one resolved file inside one record is the same
    # self-contradiction as differing digests for one raw key; identical
    # digests collapse to one claim so a record cannot be its own "other
    # live attestation".
    # Aliases are decided by filesystem identity: on a case-sensitive
    # filesystem ``F.yaml`` and ``f.yaml`` are two files and two claims, on a
    # case-insensitive one they are one file, and only the filesystem knows.
    by_record_file: dict[tuple[str, object], list[Claim]] = defaultdict(list)
    for claim in claims:
        if claim.resolved_path is not None and not claim.deleted:
            by_record_file[(claim.manifest, _file_identity(repo_path, claim.resolved_path))].append(claim)
    collapsed: set[int] = set()
    for (manifest, _identity), group in by_record_file.items():
        resolved = group[0].resolved_path
        if len(group) < 2:
            continue
        digests = {claim.attested_sha256 for claim in group}
        if len(digests) > 1:
            unreadable(
                manifest,
                "makes two different claims about one file through aliased keys "
                "(self-contradictory record)",
                resolved,
            )
            continue
        for claim in group[1:]:
            collapsed.add(id(claim))
    if collapsed:
        claims = [claim for claim in claims if id(claim) not in collapsed]
    return claims, findings


def deletion_candidates(
    manifest_relpath: str, entry_path: str, *, jurisdictions: Sequence[str]
) -> list[str]:
    """Every repo-relative path a deletion marker could denote, by layout."""
    prefix = _tree_prefix(manifest_relpath)
    if prefix:
        return [f"{prefix}/{entry_path}"]
    head = _manifest_subpath_head(manifest_relpath)
    if head and head not in RULESPEC_FILESYSTEM_ROOTS:
        return _head_bound_candidates(head, entry_path, jurisdictions)
    return [entry_path, *(f"{j}/{entry_path}" for j in jurisdictions)]


def _deletion_target_present(
    repo_path: Path,
    manifest_relpath: str,
    entry_path: str,
    *,
    jurisdictions: Sequence[str],
    checker: _LinkFreeChecker,
) -> tuple[str | None, tuple[str, ...]]:
    """Return ``(present_path, ambiguous_candidates)`` for a deletion marker.

    A legacy root marker can denote one path per jurisdiction; when more than
    one of them exists the marker is contradicted in more than one place, and
    reporting only the first would let the others go unexamined.
    """
    present = tuple(
        candidate
        for candidate in deletion_candidates(
            manifest_relpath, entry_path, jurisdictions=jurisdictions
        )
        if checker.exists_at_all(candidate)
    )
    if len(present) > 1:
        return None, present
    return (present[0] if present else None), ()


# --------------------------------------------------------------------------
# Retired-claims ratchet
# --------------------------------------------------------------------------


_ENTRY_REQUIRED = frozenset({"manifest", "attested_path", "note"})
_ENTRY_OPTIONAL = frozenset({"attested_sha256", "deletion", "issue", "unattested"})
_ENTRY_ALLOWED = _ENTRY_REQUIRED | _ENTRY_OPTIONAL


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe loader that refuses duplicate and merged mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == "<<":
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "YAML merge keys are not allowed",
                key_node.start_mark,
            )
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True, slots=True)
class RatchetEntry:
    """One retired claim.

    ``note`` is context for the human removing the entry later; the audit
    assigns it no meaning.  ``unattested`` must be true exactly when the
    claim is stale AND the file's current content carries no matching live
    attestation at all — the provenance gap is then declared in the diff
    that introduced it instead of passing silently.
    """

    manifest: str
    attested_path: str
    attested_sha256: str | None
    note: str
    unattested: bool = False
    issue: str | None = None
    deletion: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return (self.manifest, self.attested_path)


def _require_sha256(value: Any, *, field_name: str, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or not set(value) <= _HEX
    ):
        raise RatchetSchemaError(
            f"{where}: {field_name} must be a lowercase 64-character sha256 hex digest"
        )
    return value


def _read_ratchet_bytes(repo_path: Path) -> bytes | None:
    """Return the ratchet file's bytes, or ``None`` when absent."""
    ratchet_path = Path(repo_path) / RATCHET_RELATIVE_PATH
    if ratchet_path.is_symlink():
        raise RatchetSchemaError(f"{RATCHET_RELATIVE_PATH}: must not be a symlink")
    if not ratchet_path.is_file():
        return None
    if ratchet_path.stat().st_size > MAX_RATCHET_BYTES:
        raise RatchetSchemaError(
            f"{RATCHET_RELATIVE_PATH}: exceeds {MAX_RATCHET_BYTES} bytes"
        )
    return ratchet_path.read_bytes()


def parse_ratchet(raw: bytes | None) -> dict[tuple[str, str], RatchetEntry]:
    """Parse and schema-check ratchet bytes (``None`` means no file)."""
    if raw is None:
        return {}
    try:
        document = yaml.load(  # noqa: S506 - hardened loader, not yaml.Loader
            raw.decode("utf-8"), Loader=_UniqueKeyLoader
        )
    except (yaml.YAMLError, UnicodeError) as error:
        raise RatchetSchemaError(f"{RATCHET_RELATIVE_PATH}: {error}") from error
    if not isinstance(document, Mapping):
        raise RatchetSchemaError(
            f"{RATCHET_RELATIVE_PATH}: top level must be a mapping"
        )
    unknown = set(document) - {"schema_version", RATCHET_TOP_KEY}
    if unknown:
        raise RatchetSchemaError(
            f"{RATCHET_RELATIVE_PATH}: unknown top-level keys {sorted(unknown)}"
        )
    if document.get("schema_version") != RATCHET_SCHEMA:
        raise RatchetSchemaError(
            f"{RATCHET_RELATIVE_PATH}: schema_version must be {RATCHET_SCHEMA!r}"
        )
    rows = document.get(RATCHET_TOP_KEY) or []
    if not isinstance(rows, list):
        raise RatchetSchemaError(
            f"{RATCHET_RELATIVE_PATH}: {RATCHET_TOP_KEY} must be a list"
        )
    if len(rows) > MAX_RATCHET_ENTRIES:
        raise RatchetSchemaError(
            f"{RATCHET_RELATIVE_PATH}: exceeds {MAX_RATCHET_ENTRIES} entries"
        )

    entries: dict[tuple[str, str], RatchetEntry] = {}
    for index, row in enumerate(rows):
        where = f"{RATCHET_RELATIVE_PATH}[{index}]"
        if not isinstance(row, Mapping):
            raise RatchetSchemaError(f"{where}: entry must be a mapping")
        unknown = set(row) - _ENTRY_ALLOWED
        if unknown:
            raise RatchetSchemaError(f"{where}: unknown keys {sorted(unknown)}")
        missing = _ENTRY_REQUIRED - set(row)
        if missing:
            raise RatchetSchemaError(f"{where}: missing keys {sorted(missing)}")
        for name in ("manifest", "attested_path", "note"):
            if not isinstance(row[name], str) or not row[name].strip():
                raise RatchetSchemaError(f"{where}: {name} must be a non-empty string")
        unattested = row.get("unattested", False)
        if not isinstance(unattested, bool):
            raise RatchetSchemaError(f"{where}: unattested must be a boolean")
        deletion = row.get("deletion", False)
        if not isinstance(deletion, bool):
            raise RatchetSchemaError(f"{where}: deletion must be a boolean")
        if deletion == ("attested_sha256" in row):
            raise RatchetSchemaError(
                f"{where}: exactly one of attested_sha256 or deletion: true is required"
            )
        if deletion and unattested:
            raise RatchetSchemaError(
                f"{where}: a retired deletion marker cannot be unattested"
            )
        issue = row.get("issue")
        if issue is not None and not isinstance(issue, str):
            raise RatchetSchemaError(f"{where}: issue must be a string")
        entry = RatchetEntry(
            manifest=row["manifest"],
            attested_path=row["attested_path"],
            attested_sha256=(
                None
                if deletion
                else _require_sha256(
                    row["attested_sha256"], field_name="attested_sha256", where=where
                )
            ),
            note=row["note"],
            unattested=unattested,
            issue=issue,
            deletion=deletion,
        )
        if entry.key in entries:
            raise RatchetSchemaError(
                f"{where}: duplicate entry for {entry.manifest} -> {entry.attested_path}"
            )
        entries[entry.key] = entry
    return entries


def load_ratchet(repo_path: Path) -> dict[tuple[str, str], RatchetEntry]:
    """Load and schema-check the retired-claims file; absent is empty."""
    return parse_ratchet(_read_ratchet_bytes(repo_path))


_RATCHET_HEADER = """\
# Retired encoding-manifest claims
#
# Each entry retires ONE committed manifest claim — (manifest, attested_path,
# attested_sha256), or one deletion marker — in place. The record it lives in
# is kept rather than deleted: usually because it also carries the only true
# attestation for another file it covers, sometimes because it sits where the
# apply transaction does not delete. A retired claim is not live: it does not
# count toward "one live record per file" and its staleness is not a finding.
#
# An entry is a reviewed exception, nothing more. It verifies nothing and
# authorizes nothing; the pull request that adds it (or the apply transaction
# that wrote it alongside a successor manifest) is where the retirement was
# accepted. `axiom-encode manifest-audit` keeps an entry only while it is
# still real: the claim must exist in the manifest with that exact digest; a
# claim that is still true may be retired only if another live record attests
# the file's current content; and a stale claim whose file has no matching
# live attestation at all must say so with `unattested: true`. An entry that
# stops meeting those conditions FAILS the audit and must be removed.
# Re-encoding the rule is what removes `unattested` entries.
#
# See https://github.com/TheAxiomFoundation/axiom-encode/issues/1282
"""


def render_ratchet(entries: Sequence[RatchetEntry]) -> str:
    # String scalars are JSON-quoted: bare YAML scalars re-type on round-trip
    # (an all-digit sha256 loads back as an integer) and a path containing
    # ": " would change the document shape.
    lines = [
        _RATCHET_HEADER,
        f"schema_version: {RATCHET_SCHEMA}",
        f"{RATCHET_TOP_KEY}:",
    ]
    for entry in sorted(entries, key=lambda e: (e.manifest, e.attested_path)):
        dumps = lambda value: json.dumps(value, ensure_ascii=False)  # noqa: E731
        lines.append(f"  - manifest: {dumps(entry.manifest)}")
        lines.append(f"    attested_path: {dumps(entry.attested_path)}")
        if entry.deletion:
            lines.append("    deletion: true")
        else:
            lines.append(f"    attested_sha256: {dumps(entry.attested_sha256)}")
        if entry.unattested:
            lines.append("    unattested: true")
        lines.append("    note: >-")
        wrapped = textwrap.wrap(" ".join(entry.note.split()), width=82) or [""]
        lines.extend(f"      {chunk}" for chunk in wrapped)
        if entry.issue:
            lines.append(f"    issue: {dumps(entry.issue)}")
    return "\n".join(lines) + "\n"


def write_ratchet(repo_path: Path, entries: Sequence[RatchetEntry]) -> Path | None:
    """Write (or remove, when empty) the retired-claims file."""
    path = Path(repo_path) / RATCHET_RELATIVE_PATH
    if not entries:
        if path.is_file() or path.is_symlink():
            path.unlink()
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ratchet(list(entries)))
    return path


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def _uniqueness_key(path: str) -> str:
    """Normalize a resolved path for duplicate detection.

    Distinct byte-strings can alias one file (NFC/NFD, case on insensitive
    filesystems); grouping on a normalized casefolded key keeps two aliased
    records from counting as records of two different files.  The trade is
    deliberate and fail-closed: two genuinely distinct files whose paths
    differ only by case or normalization form are reported as one.
    """
    return unicodedata.normalize("NFC", path).casefold()


def audit_repository(repo_path: Path) -> AuditResult:
    """Audit every committed manifest in ``repo_path``."""
    repo_path = Path(repo_path).resolve()
    if not repo_path.is_dir():
        raise ManifestAuditError(f"{repo_path} is not a directory")

    result = AuditResult()
    claims, findings = collect_claims(repo_path)
    result.findings.extend(findings)
    result.manifests = len(
        {claim.manifest for claim in claims}
        | {finding.manifest for finding in findings if finding.manifest}
    )
    result.attestations = len(claims)

    try:
        ratchet = load_ratchet(repo_path)
    except RatchetSchemaError as error:
        result.findings.append(AuditFinding(kind=RATCHET_INVALID, detail=str(error)))
        ratchet = {}

    claims_by_file: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        if claim.resolved_path is not None and not claim.deleted:
            claims_by_file[claim.resolved_path].append(claim)
    hashes = {path: sha256_file(repo_path / path) for path in claims_by_file}

    def retired_entry(claim: Claim) -> RatchetEntry | None:
        entry = ratchet.get((claim.manifest, claim.entry_path))
        if entry is None:
            return None
        if claim.deleted:
            return entry if entry.deletion else None
        if entry.deletion or claim.attested_sha256 != entry.attested_sha256:
            return None
        return entry

    used_keys: set[tuple[str, str]] = set()
    retired_claims: list[tuple[Claim, RatchetEntry, bool]] = []  # (claim, entry, stale)

    for claim in claims:
        if claim.unsafe_reason is not None:
            result.findings.append(
                AuditFinding(
                    kind=UNSAFE,
                    manifest=claim.manifest,
                    path=claim.entry_path,
                    detail=f"applied_files path {claim.unsafe_reason}",
                )
            )
            continue
        if claim.candidates:
            result.findings.append(
                AuditFinding(
                    kind=AMBIGUOUS,
                    manifest=claim.manifest,
                    path=claim.entry_path,
                    detail=(
                        "resolves to more than one jurisdiction: "
                        + ", ".join(claim.candidates)
                    ),
                    members=frozenset(claim.candidates),
                )
            )
            continue
        if claim.deleted:
            if claim.resolved_path is not None:
                entry = retired_entry(claim)
                if entry is not None:
                    # A re-created file: the marker is superseded by whatever
                    # record attests the new content; validated below.
                    used_keys.add(entry.key)
                    retired_claims.append((claim, entry, True))
                    result.retired += 1
                else:
                    result.findings.append(
                        AuditFinding(
                            kind=RESURRECTED,
                            manifest=claim.manifest,
                            path=claim.resolved_path,
                            detail="is recorded as deleted but something exists there",
                        )
                    )
            else:
                result.matched += 1
            continue
        entry = retired_entry(claim)
        if claim.resolved_path is None:
            if entry is not None:
                used_keys.add(entry.key)
                retired_claims.append((claim, entry, True))
                result.retired += 1
            else:
                result.findings.append(
                    AuditFinding(
                        kind=MISSING,
                        manifest=claim.manifest,
                        path=claim.entry_path,
                        detail="is attested but no such file exists in the working tree",
                    )
                )
            continue
        current = hashes[claim.resolved_path]
        stale = current != claim.attested_sha256
        if entry is not None:
            used_keys.add(entry.key)
            retired_claims.append((claim, entry, stale))
            result.retired += 1
            continue
        if not stale:
            result.matched += 1
            continue
        result.findings.append(
            AuditFinding(
                kind=MISMATCH,
                manifest=claim.manifest,
                path=claim.resolved_path,
                detail=(
                    f"attests {claim.attested_sha256[:12]}… but the file is "
                    f"{current[:12]}…"
                ),
            )
        )

    # Uniqueness: at most one LIVE manifest may attest a given file.  Retired
    # claims are not live.  Grouping is by normalized path so aliasing
    # byte-strings cannot split one file across buckets; the invariant is over
    # manifests, not entries.
    live_by_key: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"manifests": set(), "paths": set()}
    )
    matching_live_by_file: dict[str, set[str]] = defaultdict(set)
    for path, path_claims in claims_by_file.items():
        current = hashes[path]
        for claim in path_claims:
            if claim.candidates or claim.unsafe_reason:
                continue
            if retired_entry(claim) is not None:
                continue
            bucket = live_by_key[_uniqueness_key(path)]
            bucket["manifests"].add(claim.manifest)
            bucket["paths"].add(path)
            if claim.attested_sha256 == current:
                matching_live_by_file[path].add(claim.manifest)
    for _key, bucket in sorted(live_by_key.items()):
        if len(bucket["manifests"]) <= 1:
            continue
        result.findings.append(
            AuditFinding(
                kind=DUPLICATE,
                path=" / ".join(sorted(bucket["paths"])),
                detail=(
                    "is attested by more than one live manifest (paths differing "
                    "only by case or normalization form count as one file): "
                    + ", ".join(sorted(bucket["manifests"]))
                ),
                members=frozenset(bucket["manifests"]),
            )
        )

    # Retired-entry validity.  A still-true claim may be retired only when
    # another live record attests the file; a stale claim whose file has no
    # matching live attestation must declare the gap.
    for claim, entry, stale in retired_claims:
        target = claim.resolved_path
        has_live_match = target is not None and bool(
            matching_live_by_file.get(target, set()) - {claim.manifest}
        )
        if entry.deletion:
            if not has_live_match:
                result.findings.append(
                    AuditFinding(
                        kind=RATCHET_INVALID,
                        manifest=entry.manifest,
                        path=entry.attested_path,
                        detail=(
                            "retires a deletion marker, but nothing live attests "
                            "the file that now exists there; a re-created file "
                            "needs a live record"
                        ),
                    )
                )
            continue
        if not stale:
            if not has_live_match:
                result.findings.append(
                    AuditFinding(
                        kind=RATCHET_INVALID,
                        manifest=entry.manifest,
                        path=entry.attested_path,
                        detail=(
                            "retires a claim that is still true and is the file's "
                            "only live attestation; a sole true claim cannot be "
                            "retired — remove the entry"
                        ),
                    )
                )
            elif entry.unattested:
                result.findings.append(
                    AuditFinding(
                        kind=RATCHET_INVALID,
                        manifest=entry.manifest,
                        path=entry.attested_path,
                        detail="is marked unattested but the claim is still true; drop the flag",
                    )
                )
            continue
        if has_live_match and entry.unattested:
            result.findings.append(
                AuditFinding(
                    kind=RATCHET_INVALID,
                    manifest=entry.manifest,
                    path=entry.attested_path,
                    detail=(
                        "is marked unattested, but the file's current content "
                        "has a matching live attestation; drop the flag"
                    ),
                )
            )
        elif not has_live_match and not entry.unattested:
            result.findings.append(
                AuditFinding(
                    kind=UNATTESTED,
                    manifest=entry.manifest,
                    path=entry.attested_path,
                    detail=(
                        "retiring this claim leaves the file's current content "
                        "with no matching live attestation; the entry must "
                        "declare `unattested: true` (or the rule must be re-encoded)"
                    ),
                )
            )

    for key, entry in sorted(ratchet.items()):
        if key in used_keys:
            continue
        result.findings.append(
            AuditFinding(
                kind=RATCHET_UNUSED,
                manifest=entry.manifest,
                path=entry.attested_path,
                detail="no longer corresponds to a committed claim; remove it",
            )
        )

    result.findings.sort(
        key=lambda finding: (
            _FINDING_ORDER.index(finding.kind)
            if finding.kind in _FINDING_ORDER
            else len(_FINDING_ORDER),
            finding.manifest or "",
            finding.path or "",
        )
    )
    return result


# --------------------------------------------------------------------------
# Prune-on-supersede (write path)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PrunePlan:
    """What writing ``new_manifest`` implies for previously committed records.

    ``retire`` maps each fully superseded manifest to the sha256 of the bytes
    the plan was computed from, so the caller can pin exactly that record in
    its compare-and-swap and never delete a record that changed after
    planning.  ``disclose`` lists the claims to retire in place on records
    that survive.  ``drop_keys`` are existing retired-claim entries that
    belong to manifests being deleted.
    """

    retire: Mapping[str, str]
    disclose: tuple[RatchetEntry, ...]
    drop_keys: tuple[tuple[str, str], ...] = ()

    @property
    def empty(self) -> bool:
        return not self.retire and not self.disclose and not self.drop_keys


def plan_prune(
    repo_path: Path,
    *,
    new_manifest: str,
    attested_paths: Iterable[str],
    claims: Sequence[Claim] | None = None,
    structural_findings: Sequence[AuditFinding] | None = None,
    existing_ratchet: Mapping[tuple[str, str], RatchetEntry] | None = None,
    retirable: Callable[[str], bool] | None = None,
    issue: str
    | None = "https://github.com/TheAxiomFoundation/axiom-encode/issues/1282",
) -> PrunePlan:
    """Plan the retirement of records superseded by ``new_manifest``.

    ``retirable`` says whether a record's file may be deleted at all (the
    apply transaction admits deletions only in recognized layouts); a fully
    superseded record that is not retirable has its claims retired in place
    instead, so the plan never schedules a deletion the installer will refuse.

    A prior manifest is deleted only when EVERYTHING it protects is covered by
    ``new_manifest``: every one of its claims must be a resolved file claim on
    a path the new manifest attests.  A deletion marker, an unresolved,
    ambiguous, or unsafe claim, or any structural finding on the record is a
    claim the new manifest says nothing about, and a record carrying one is
    never auto-retired — deleting it would erase the only evidence that claim
    was ever made.  A record that survives has each of its claims on the new
    manifest's paths retired in place, whether or not the old claim is still
    true: staleness does not decide retirement, supersession does, and the
    audit re-checks the entry's validity against the installed bytes.
    """
    repo_path = Path(repo_path)
    targets = set(attested_paths)
    if claims is None or structural_findings is None:
        claims, structural_findings = collect_claims(repo_path)
    if existing_ratchet is None:
        existing_ratchet = load_ratchet(repo_path)

    unsafe_to_retire: set[str] = {
        finding.manifest for finding in structural_findings if finding.manifest
    }
    jurisdictions = jurisdiction_roots(repo_path)
    covered: dict[str, set[str]] = defaultdict(set)
    superseded_markers: dict[str, list[Claim]] = defaultdict(list)
    by_manifest: dict[str, list[Claim]] = defaultdict(list)
    digests: dict[str, str] = {}
    drop_keys: list[tuple[str, str]] = []
    # Overwriting an existing record: its old claims vanish with its bytes, so
    # any retired-claim rows that belonged to them must go too.
    drop_keys.extend(key for key in existing_ratchet if key[0] == new_manifest)
    for claim in claims:
        if claim.manifest == new_manifest:
            continue
        by_manifest[claim.manifest].append(claim)
        digests[claim.manifest] = claim.manifest_sha256
        if claim.deleted and claim.unsafe_reason is None and not claim.candidates:
            # A deletion marker on a path the new manifest re-creates is
            # superseded by it; anywhere else it is a live claim the new
            # manifest says nothing about.  Where the marker is already
            # contradicted, the contradiction must be at exactly the path the
            # apply writes; an unresolved marker may be claimed by any of its
            # candidate paths, since nothing else exists to prefer.
            candidates = set(
                deletion_candidates(
                    claim.manifest, claim.entry_path, jurisdictions=jurisdictions
                )
            )
            if claim.resolved_path is not None:
                hits = {claim.resolved_path} & targets
            else:
                hits = candidates & targets
            if hits:
                covered[claim.manifest].update(hits)
                superseded_markers[claim.manifest].append(claim)
                continue
        if (
            claim.deleted
            or claim.resolved_path is None
            or claim.candidates
            or claim.unsafe_reason
        ):
            unsafe_to_retire.add(claim.manifest)
            continue
        covered[claim.manifest].add(claim.resolved_path)

    retire: dict[str, str] = {}
    disclose: list[RatchetEntry] = []
    for manifest, paths in sorted(covered.items()):
        overlap = paths & targets
        if not overlap:
            continue
        may_delete = retirable is None or retirable(manifest)
        if paths <= targets and manifest not in unsafe_to_retire and may_delete:
            retire[manifest] = digests[manifest]
            drop_keys.extend(key for key in existing_ratchet if key[0] == manifest)
            continue
        for marker in superseded_markers.get(manifest, ()):
            if (manifest, marker.entry_path) in existing_ratchet:
                continue
            disclose.append(
                RatchetEntry(
                    manifest=manifest,
                    attested_path=marker.entry_path,
                    attested_sha256=None,
                    note=(
                        f"Deletion marker superseded by {new_manifest}, which "
                        "re-creates this path; retired in place because the "
                        "record is kept."
                    ),
                    issue=issue,
                    deletion=True,
                )
            )
        for claim in by_manifest[manifest]:
            if (
                claim.deleted
                or claim.resolved_path not in overlap
                or claim.attested_sha256 is None
            ):
                continue
            existing = existing_ratchet.get((manifest, claim.entry_path))
            if (
                existing is not None
                and existing.attested_sha256 == claim.attested_sha256
            ):
                if existing.unattested:
                    # The successor being installed now attests this file;
                    # the declared provenance gap closes with it.
                    disclose.append(
                        RatchetEntry(
                            manifest=existing.manifest,
                            attested_path=existing.attested_path,
                            attested_sha256=existing.attested_sha256,
                            note=existing.note,
                            unattested=False,
                            issue=existing.issue,
                        )
                    )
                continue
            disclose.append(
                RatchetEntry(
                    manifest=manifest,
                    attested_path=claim.entry_path,
                    attested_sha256=claim.attested_sha256,
                    note=(
                        f"Superseded by {new_manifest}, which attests this "
                        "rule's current content; the claim is retired in place "
                        "because the record is kept (it still covers other "
                        "files, or its location is not one the apply "
                        "transaction deletes)."
                    ),
                    issue=issue,
                )
            )
    return PrunePlan(
        retire=dict(retire), disclose=tuple(disclose), drop_keys=tuple(drop_keys)
    )


@dataclass(frozen=True, slots=True)
class RatchetUpdate:
    """The retired-claims file after applying a plan, plus what it was read from.

    ``expected_sha256`` is the digest of the bytes that were merged into (or
    ``None`` when the file did not exist), for compare-and-swap.  ``raw`` is
    ``None`` with ``delete`` set when the result is empty.
    """

    raw: bytes | None
    delete: bool
    expected_sha256: str | None

    @property
    def changed(self) -> bool:
        return self.raw is not None or self.delete


def ratchet_update(
    repo_path: Path,
    *,
    additions: Sequence[RatchetEntry],
    drop_keys: Iterable[tuple[str, str]] = (),
    replacements: Sequence[RatchetEntry] = (),
) -> RatchetUpdate | None:
    """Compute the retired-claims file content after a prune plan.

    Returns ``None`` when nothing changes.  Existing entries win on key
    collision so a reviewed note is never silently rewritten; ``replacements``
    override by key and are reserved for entries whose facts the caller itself
    is changing (it is deleting the file the entry refers to).
    """
    before = _read_ratchet_bytes(repo_path)
    expected = hashlib.sha256(before).hexdigest() if before is not None else None
    entries = dict(parse_ratchet(before))
    changed = False
    for key in drop_keys:
        if entries.pop(key, None) is not None:
            changed = True
    for entry in replacements:
        if entry.key in entries and entries[entry.key] != entry:
            entries[entry.key] = entry
            changed = True
    for entry in additions:
        current = entries.get(entry.key)
        if current is None:
            entries[entry.key] = entry
            changed = True
        elif (
            current.unattested
            and not entry.unattested
            and current.attested_sha256 == entry.attested_sha256
        ):
            # Only the declared-gap flag may be rewritten by machine, and only
            # in the closing direction; the reviewed note is preserved.
            entries[entry.key] = entry
            changed = True
    if not changed:
        return None
    if not entries:
        return RatchetUpdate(
            raw=None, delete=before is not None, expected_sha256=expected
        )
    return RatchetUpdate(
        raw=render_ratchet(list(entries.values())).encode(),
        delete=False,
        expected_sha256=expected,
    )


def apply_prune(
    repo_path: Path,
    *,
    new_manifest: str,
    attested_paths: Iterable[str],
) -> PrunePlan:
    """Apply a prune plan directly to a checkout (non-transactional helper).

    The production apply path stages the same plan through its install
    transaction (see ``cli._plan_apply_prune_transaction``); this helper backs
    tests and offline cleanups.
    """
    repo_path = Path(repo_path)
    plan = plan_prune(
        repo_path, new_manifest=new_manifest, attested_paths=attested_paths
    )
    for manifest in plan.retire:
        target = repo_path / manifest
        if target.is_file():
            target.unlink()
        parent = target.parent
        while parent != repo_path and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    update = ratchet_update(
        repo_path, additions=plan.disclose, drop_keys=plan.drop_keys
    )
    if update is not None:
        ratchet_path = repo_path / RATCHET_RELATIVE_PATH
        if update.delete:
            ratchet_path.unlink(missing_ok=True)
        else:
            assert update.raw is not None
            ratchet_path.parent.mkdir(parents=True, exist_ok=True)
            ratchet_path.write_bytes(update.raw)
    return plan


def format_report(result: AuditResult, *, limit: int | None = None) -> str:
    """Render a human-readable audit report."""
    lines: list[str] = []
    if result.passed:
        lines.append(
            f"Every committed attestation matches its file "
            f"({result.matched} attestation(s) across {result.manifests} manifest(s)"
            + (
                f", {result.retired} claim(s) retired in place"
                if result.retired
                else ""
            )
            + ")."
        )
        return "\n".join(lines)
    lines.append(
        f"{len(result.findings)} manifest audit finding(s) across "
        f"{result.manifests} manifest(s) / {result.attestations} attestation(s)."
    )
    grouped = result.by_kind()
    for kind in _FINDING_ORDER:
        entries = grouped.get(kind)
        if not entries:
            continue
        lines.append(f"\n{kind} ({len(entries)}):")
        shown = entries if limit is None else entries[:limit]
        for finding in shown:
            where = finding.manifest or finding.path or ""
            if finding.manifest and finding.path:
                where = f"{finding.manifest} -> {finding.path}"
            lines.append(f"  - {where}: {finding.detail}")
        if limit is not None and len(entries) > limit:
            lines.append(f"  … and {len(entries) - limit} more")
    return "\n".join(lines)
