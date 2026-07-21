"""Canonical country-checkout routing for RuleSpec content.

RuleSpec content has one supported on-disk shape::

    rulespec-<country>/<jurisdiction>/...

For example, ``us-ca:regulations/mpp/63-300/1`` resolves beneath
``rulespec-us/us-ca``.  A flat ``rulespec-us-ca`` checkout, a workspace that
merely contains a matching checkout, or a checkout alias inferred from Git
origin is not a RuleSpec content root.  Callers must authorize the exact
country checkout (for dependencies) or its direct jurisdiction child (for the
active content root).
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, NamedTuple

from .constants import (
    RULESPEC_ATOMIC_MODULE_ROOTS,
    RULESPEC_COMPOSITION_SPEC_ROOT,
    RULESPEC_FILE_SUFFIX,
    RULESPEC_FILESYSTEM_ROOTS,
    RULESPEC_TEST_FILE_SUFFIX,
)


class _GitProbeError(RuntimeError):
    """Git identity could not be checked for an observed repository boundary."""

    def __init__(self, message: str, *, category: str = "error") -> None:
        super().__init__(message)
        self.category = category


class CanonicalRuleSpecCheckoutInspection(NamedTuple):
    """Canonical checkout identity and a stable rejection code."""

    name: str | None
    rejection: str | None


class _PathMutationStamp(NamedTuple):
    """Cheap identity and metadata stamp for one filesystem entry."""

    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    content_sha256: str | None


@dataclass(frozen=True)
class _CheckoutMutationFingerprint:
    """Filesystem and environment inputs that determine checkout admission."""

    path_identities: tuple[tuple[Path, tuple[int, int, int]], ...]
    directory_stamps: tuple[tuple[Path, _PathMutationStamp], ...]
    git_path_identities: tuple[
        tuple[Path, tuple[tuple[Path, tuple[int, int, int]], ...] | None], ...
    ]
    git_input_stamps: tuple[tuple[Path, _PathMutationStamp | None], ...]
    git_environment: tuple[tuple[str, str | None], ...]


@dataclass(frozen=True)
class _CachedCheckoutInspection:
    inspection: CanonicalRuleSpecCheckoutInspection
    git_config_inputs: tuple[Path, ...]
    fingerprint: _CheckoutMutationFingerprint


@dataclass(frozen=True)
class _CheckoutAdmissionSnapshot:
    """Pre-inspection inputs used to admit one stable cache entry."""

    git_config_inputs: tuple[Path, ...]
    fingerprint: _CheckoutMutationFingerprint


@dataclass
class _RuleSpecRoutingCache:
    """Successful checkout identity probes admitted for one bounded operation."""

    checkout_inspections: dict[tuple[Path, bool, bool], _CachedCheckoutInspection] = (
        field(default_factory=dict)
    )


_RULESPEC_ROUTING_CACHE: ContextVar[_RuleSpecRoutingCache | None] = ContextVar(
    "axiom_rulespec_routing_cache",
    default=None,
)


@contextmanager
def _rulespec_routing_cache_scope() -> Iterator[_RuleSpecRoutingCache]:
    """Reuse successful identity probes only for one explicit operation."""

    current = _RULESPEC_ROUTING_CACHE.get()
    if current is not None:
        yield current
        return
    cache = _RuleSpecRoutingCache()
    token = _RULESPEC_ROUTING_CACHE.set(cache)
    try:
        yield cache
    finally:
        _RULESPEC_ROUTING_CACHE.reset(token)


def _path_mutation_stamp(
    path: Path,
    *,
    hash_regular_file: bool = False,
) -> _PathMutationStamp | None:
    """Return a stable lstat stamp, optionally binding regular-file contents."""

    path = Path(path)
    try:
        before = path.lstat()
    except OSError:
        return None
    digest = None
    if hash_regular_file and stat.S_ISREG(before.st_mode):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            after = path.lstat()
        except OSError:
            return None
        if _stat_identity(after) != _stat_identity(before):
            return None
    return _PathMutationStamp(
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        digest,
    )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Return fields that must remain stable while a file is fingerprinted."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _path_identity_fingerprint(
    path: Path,
) -> tuple[tuple[Path, tuple[int, int, int]], ...] | None:
    """Bind every lexical component to its inode and file type."""

    absolute = Path(os.path.abspath(Path(path).expanduser()))
    components: list[tuple[Path, tuple[int, int, int]]] = []
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        stamp = _path_mutation_stamp(cursor)
        if stamp is None:
            return None
        components.append((cursor, (stamp.device, stamp.inode, stamp.mode)))
    return tuple(components)


def jurisdiction_country(prefix: str) -> str:
    """Return the country portion of a jurisdiction repo prefix (us-ca -> us)."""
    return prefix.split("-", 1)[0]


def monorepo_checkout_name(prefix: str) -> str:
    """Return the country monorepo repository name for a jurisdiction prefix."""
    return f"rulespec-{jurisdiction_country(prefix)}"


def _lexical_rulespec_path(path: Path) -> Path | None:
    """Return one absolute path only when no component is caller-controlled indirection."""

    raw = Path(os.path.abspath(Path(path).expanduser()))
    if sys.platform == "darwin":
        for alias, expected_target in (
            (Path("/var"), Path("/private/var")),
            (Path("/tmp"), Path("/private/tmp")),
            (Path("/etc"), Path("/private/etc")),
        ):
            try:
                relative = raw.relative_to(alias)
            except ValueError:
                continue
            try:
                if alias.is_symlink() and alias.resolve(strict=True) == expected_target:
                    raw = expected_target / relative
            except OSError:
                pass
            break

    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            return None
    return raw


def _canonical_country_checkout_name(
    path: Path,
    *,
    allow_composition_specs: bool = False,
    allow_checkout_source_roots: bool = False,
) -> str | None:
    """Return the exact canonical country-checkout name for ``path``."""

    return inspect_canonical_rulespec_checkout(
        path,
        allow_composition_specs=allow_composition_specs,
        allow_checkout_source_roots=allow_checkout_source_roots,
    ).name


def inspect_canonical_rulespec_checkout(
    path: Path,
    *,
    allow_composition_specs: bool = False,
    allow_checkout_source_roots: bool = False,
) -> CanonicalRuleSpecCheckoutInspection:
    """Inspect one exact country checkout without weakening route acceptance."""

    cache = _RULESPEC_ROUTING_CACHE.get()
    cache_key = (
        Path(os.path.abspath(Path(path).expanduser())),
        allow_composition_specs,
        allow_checkout_source_roots,
    )
    cached = cache.checkout_inspections.get(cache_key) if cache is not None else None
    if cached is not None:
        current_fingerprint = _checkout_mutation_fingerprint(
            cache_key[0],
            git_config_inputs=cached.git_config_inputs,
        )
        if current_fingerprint == cached.fingerprint:
            return cached.inspection
        cache.checkout_inspections.pop(cache_key, None)

    if cache is None:
        return _inspect_canonical_rulespec_checkout_uncached(
            path,
            allow_composition_specs=allow_composition_specs,
            allow_checkout_source_roots=allow_checkout_source_roots,
        )

    for _attempt in range(2):
        before = _checkout_admission_snapshot(cache_key[0])
        inspection = _inspect_canonical_rulespec_checkout_uncached(
            path,
            allow_composition_specs=allow_composition_specs,
            allow_checkout_source_roots=allow_checkout_source_roots,
        )
        if inspection.name is None or before is None:
            return inspection
        cached_inspection = _cacheable_checkout_inspection(
            cache_key[0],
            inspection,
            before=before,
        )
        if cached_inspection is not None:
            cache.checkout_inspections[cache_key] = cached_inspection
            return inspection

    return CanonicalRuleSpecCheckoutInspection(
        None, "checkout-mutated-during-inspection"
    )


def _checkout_admission_snapshot(
    checkout: Path,
) -> _CheckoutAdmissionSnapshot | None:
    """Capture admission inputs before a successful identity inspection."""

    try:
        git_boundary = _nearest_git_boundary(checkout)
        git_config_inputs = (
            _git_config_input_paths(str(checkout)) if git_boundary is not None else ()
        )
    except _GitProbeError:
        return None
    fingerprint = _checkout_mutation_fingerprint(
        checkout,
        git_config_inputs=git_config_inputs,
    )
    if fingerprint is None:
        return None
    return _CheckoutAdmissionSnapshot(
        git_config_inputs=git_config_inputs,
        fingerprint=fingerprint,
    )


def _cacheable_checkout_inspection(
    checkout: Path,
    inspection: CanonicalRuleSpecCheckoutInspection,
    *,
    before: _CheckoutAdmissionSnapshot,
) -> _CachedCheckoutInspection | None:
    """Cache a successful admission only when its input snapshot stayed stable."""

    try:
        git_boundary = _nearest_git_boundary(checkout)
        git_config_inputs = (
            _git_config_input_paths(str(checkout)) if git_boundary is not None else ()
        )
    except _GitProbeError:
        return None
    if git_config_inputs != before.git_config_inputs:
        return None
    after = _checkout_mutation_fingerprint(
        checkout,
        git_config_inputs=git_config_inputs,
    )
    if after is None or after != before.fingerprint:
        return None
    return _CachedCheckoutInspection(
        inspection=inspection,
        git_config_inputs=git_config_inputs,
        fingerprint=after,
    )


def _checkout_mutation_fingerprint(
    checkout: Path,
    *,
    git_config_inputs: tuple[Path, ...],
) -> _CheckoutMutationFingerprint | None:
    """Fingerprint checkout admission inputs without invoking Git or walking files."""

    lexical = _lexical_rulespec_path(checkout)
    if lexical is None:
        return None
    path_identities = _path_identity_fingerprint(lexical)
    if path_identities is None:
        return None

    try:
        git_boundary = _nearest_git_boundary(lexical)
    except _GitProbeError:
        return None
    directory_stamps: list[tuple[Path, _PathMutationStamp]] = []
    for directory in (lexical,):
        stamp = _path_mutation_stamp(directory)
        if stamp is None or not stat.S_ISDIR(stamp.mode):
            return None
        directory_stamps.append((directory, stamp))
    git_paths = _git_identity_input_paths(
        lexical,
        git_boundary=git_boundary,
        git_config_inputs=git_config_inputs,
    )
    git_path_identities = tuple(
        (path, _path_identity_fingerprint(path)) for path in git_paths
    )
    git_input_stamps = tuple(
        (path, _path_mutation_stamp(path, hash_regular_file=True)) for path in git_paths
    )
    git_environment = tuple(
        (name, os.environ.get(name))
        for name in sorted(
            {
                "HOME",
                "XDG_CONFIG_HOME",
                *(name for name in os.environ if name.startswith("GIT_")),
            }
        )
    )
    return _CheckoutMutationFingerprint(
        path_identities=path_identities,
        directory_stamps=tuple(directory_stamps),
        git_path_identities=git_path_identities,
        git_input_stamps=git_input_stamps,
        git_environment=git_environment,
    )


def _git_identity_input_paths(
    checkout: Path,
    *,
    git_boundary: Path | None,
    git_config_inputs: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Return files whose mutation can change Git worktree/origin identity."""

    candidate_markers = {
        directory / ".git" for directory in (checkout, *checkout.parents)
    }
    if git_boundary is None:
        return tuple(sorted(candidate_markers, key=str))
    marker = git_boundary / ".git"
    paths = {marker, *git_config_inputs}
    paths.update(candidate_markers)
    git_directory = _git_directory_from_marker(marker)
    if git_directory is not None:
        paths.update(
            {
                git_directory,
                git_directory / "commondir",
                git_directory / "config",
                git_directory / "config.worktree",
                git_directory / "HEAD",
            }
        )
        common_directory = _git_common_directory(git_directory)
        if common_directory is not None:
            paths.update(
                {
                    common_directory,
                    common_directory / "config",
                    common_directory / "config.worktree",
                }
            )
    home = Path(os.environ.get("HOME", str(Path.home()))).expanduser()
    xdg_config_home = Path(
        os.environ.get("XDG_CONFIG_HOME", str(home / ".config"))
    ).expanduser()
    paths.update({home / ".gitconfig", xdg_config_home / "git/config"})
    return tuple(sorted((Path(path) for path in paths), key=str))


def _git_directory_from_marker(marker: Path) -> Path | None:
    """Resolve a directory marker or linked-worktree gitdir file lexically."""

    if marker.is_dir():
        return marker
    if not marker.is_file() or marker.is_symlink():
        return None
    try:
        marker_text = marker.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    prefix = "gitdir:"
    if not marker_text.lower().startswith(prefix):
        return None
    raw_git_directory = Path(marker_text[len(prefix) :].strip()).expanduser()
    if not raw_git_directory.is_absolute():
        raw_git_directory = marker.parent / raw_git_directory
    return Path(os.path.abspath(raw_git_directory))


def _git_common_directory(git_directory: Path) -> Path | None:
    """Resolve the common Git directory used by linked worktrees."""

    marker = git_directory / "commondir"
    if not marker.is_file() or marker.is_symlink():
        return git_directory
    try:
        raw_common_directory = Path(marker.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeError):
        return None
    if not raw_common_directory.is_absolute():
        raw_common_directory = git_directory / raw_common_directory
    return Path(os.path.abspath(raw_common_directory))


def _inspect_canonical_rulespec_checkout_uncached(
    path: Path,
    *,
    allow_composition_specs: bool = False,
    allow_checkout_source_roots: bool = False,
) -> CanonicalRuleSpecCheckoutInspection:
    """Inspect one checkout before it is admitted to an operation cache."""

    lexical = _lexical_rulespec_path(path)
    if lexical is None:
        return CanonicalRuleSpecCheckoutInspection(None, "lexical-path-rejected")
    if not lexical.is_dir():
        return CanonicalRuleSpecCheckoutInspection(None, "checkout-not-directory")
    checkout = lexical.resolve()
    name = checkout.name
    if not name.startswith("rulespec-"):
        return CanonicalRuleSpecCheckoutInspection(None, "checkout-name-prefix")
    country = name.removeprefix("rulespec-")
    if re.fullmatch(r"[a-z]{2}", country) is None:
        return CanonicalRuleSpecCheckoutInspection(None, "checkout-country-name")
    blocked_roots = RULESPEC_FILESYSTEM_ROOTS
    if allow_checkout_source_roots:
        for root_name in RULESPEC_FILESYSTEM_ROOTS:
            source_root = checkout / root_name
            if (source_root.exists() or source_root.is_symlink()) and (
                source_root.is_symlink() or not source_root.is_dir()
            ):
                return CanonicalRuleSpecCheckoutInspection(
                    None, "checkout-source-root-not-directory"
                )
        blocked_roots = frozenset()
    elif allow_composition_specs:
        composition_root = checkout / RULESPEC_COMPOSITION_SPEC_ROOT
        if (composition_root.exists() or composition_root.is_symlink()) and (
            composition_root.is_symlink() or not composition_root.is_dir()
        ):
            return CanonicalRuleSpecCheckoutInspection(
                None, "composition-root-not-directory"
            )
        blocked_roots = blocked_roots - {RULESPEC_COMPOSITION_SPEC_ROOT}
    if any(
        (checkout / root_name).exists() or (checkout / root_name).is_symlink()
        for root_name in blocked_roots
    ):
        return CanonicalRuleSpecCheckoutInspection(None, "atomic-root-at-checkout")
    try:
        git_boundary = _nearest_git_boundary(checkout)
    except _GitProbeError as exc:
        return CanonicalRuleSpecCheckoutInspection(
            None, f"git-boundary-probe-{exc.category}"
        )
    if git_boundary is not None and git_boundary != checkout:
        return CanonicalRuleSpecCheckoutInspection(None, "git-boundary-mismatch")
    if git_boundary is None:
        return CanonicalRuleSpecCheckoutInspection(name, None)
    try:
        git_top_level = _git_top_level(str(checkout))
    except _GitProbeError as exc:
        return CanonicalRuleSpecCheckoutInspection(
            None, f"git-top-level-probe-{exc.category}"
        )
    if git_top_level is None:
        return CanonicalRuleSpecCheckoutInspection(None, "git-top-level-unavailable")
    if git_top_level is not None and git_top_level != checkout:
        return CanonicalRuleSpecCheckoutInspection(None, "git-top-level-mismatch")
    try:
        origin_name = _git_origin_repo_name(str(checkout))
    except _GitProbeError as exc:
        return CanonicalRuleSpecCheckoutInspection(
            None, f"git-origin-probe-{exc.category}"
        )
    if origin_name is not None and origin_name != name:
        return CanonicalRuleSpecCheckoutInspection(None, "git-origin-name-mismatch")
    return CanonicalRuleSpecCheckoutInspection(name, None)


def canonical_rulespec_root_identity(path: Path) -> str | None:
    """Return a stable identity for an exact canonical jurisdiction root.

    ``rulespec-us/us-co`` is returned for the direct ``us-co`` child of the
    canonical ``rulespec-us`` checkout.  Checkout roots, flat legacy roots,
    nested files, workspace directories, and aliased checkout names return
    ``None``.
    """

    lexical = _lexical_rulespec_path(path)
    if lexical is None:
        return None
    content_root = lexical.resolve()
    if not content_root.is_dir():
        return None
    jurisdiction = content_root.name
    checkout = content_root.parent
    expected_checkout = monorepo_checkout_name(jurisdiction)
    if checkout.name != expected_checkout:
        return None
    if (
        _canonical_country_checkout_name(checkout, allow_composition_specs=True)
        != expected_checkout
    ):
        return None
    country = jurisdiction_country(jurisdiction)
    if not _is_jurisdiction_dir_name(jurisdiction, country):
        return None
    return f"{expected_checkout}/{jurisdiction}"


def is_policy_repo_root(path: Path) -> bool:
    """Return True for an exact canonical country checkout root."""

    return _canonical_country_checkout_name(Path(path)) is not None


def is_composition_policy_repo_root(path: Path) -> bool:
    """Return True for an exact country checkout that may contain ProgramSpecs."""

    return (
        _canonical_country_checkout_name(Path(path), allow_composition_specs=True)
        is not None
    )


def is_rulespec_source_checkout_root(path: Path) -> bool:
    """Return True for an exact checkout that may own RuleSpec source roots."""

    return (
        _canonical_country_checkout_name(
            Path(path),
            allow_checkout_source_roots=True,
        )
        is not None
    )


def is_jurisdiction_content_root(path: Path) -> bool:
    """Return True for an exact direct jurisdiction child of a country checkout."""

    return canonical_rulespec_root_identity(path) is not None


def find_policy_repo_root(path: Path) -> Path | None:
    """Return the canonical jurisdiction content root containing ``path``."""

    lexical = _lexical_rulespec_path(path)
    if lexical is None:
        return None
    current = lexical.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if canonical_rulespec_root_identity(candidate) is not None:
            return candidate
    return None


def canonical_rulespec_repo_name(path: Path) -> str | None:
    """Return the canonical import-repository name for ``path`` when known."""

    lexical = _lexical_rulespec_path(path)
    if lexical is None:
        return None
    current = lexical.resolve()
    content_root = find_policy_repo_root(current)
    if content_root is not None:
        return f"rulespec-{content_root.name}"
    return _canonical_country_checkout_name(current, allow_composition_specs=True)


def jurisdiction_content_dir(checkout: Path, prefix: str) -> Path:
    """Return ``prefix``'s direct root inside one canonical country checkout."""

    candidates = candidate_jurisdiction_content_dirs(checkout, prefix)
    if len(candidates) != 1:
        raise ValueError(
            f"{checkout} is not the canonical RuleSpec root for {prefix!r}"
        )
    return candidates[0]


def candidate_jurisdiction_content_dirs(base: Path, prefix: str) -> list[Path]:
    """Return the one canonical content-root candidate authorized by ``base``.

    ``base`` must be either the exact country checkout or the exact matching
    jurisdiction content root.  Workspace and sibling-checkout probing is not
    performed.  The returned direct child is not existence-checked.
    """

    lexical = _lexical_rulespec_path(base)
    if lexical is None:
        return []
    base = lexical.resolve()
    expected_checkout = monorepo_checkout_name(prefix)
    if (
        base.name == expected_checkout
        and _canonical_country_checkout_name(base, allow_composition_specs=True)
        == expected_checkout
    ):
        return [base / prefix]
    if canonical_rulespec_root_identity(base) == f"{expected_checkout}/{prefix}":
        return [base]
    return []


def resolve_jurisdiction_content_dir(
    bases: Iterable[Path],
    prefix: str,
) -> Path | None:
    """Return the first existing content root for ``prefix`` across ``bases``."""
    for base in bases:
        for candidate in candidate_jurisdiction_content_dirs(base, prefix):
            if candidate.is_dir():
                return candidate
    return None


def jurisdiction_subdir_names(
    checkout: Path,
    *,
    allow_composition_specs: bool = False,
) -> set[str]:
    """Return direct jurisdiction children of one canonical country checkout."""

    lexical = _lexical_rulespec_path(checkout)
    if lexical is None:
        return set()
    checkout = lexical.resolve()
    checkout_name = _canonical_country_checkout_name(
        checkout,
        allow_composition_specs=allow_composition_specs,
    )
    if checkout_name is None or not checkout.is_dir():
        return set()
    country = checkout_name.removeprefix("rulespec-")
    return {
        child.name
        for child in checkout.iterdir()
        if not child.is_symlink()
        and child.is_dir()
        and _is_jurisdiction_dir_name(child.name, country)
    }


def atomic_rulespec_module_paths(checkout: Path) -> tuple[Path, ...]:
    """Strictly inspect a country checkout and return its atomic modules."""

    checkout = Path(checkout).resolve()
    inspection = inspect_canonical_rulespec_checkout(
        checkout,
        allow_composition_specs=True,
    )
    if inspection.name is None:
        raise ValueError(
            "RuleSpec module scan requires an exact canonical rulespec-<country> "
            f"checkout ({inspection.rejection}): {checkout}"
        )
    flat_roots = [checkout / root for root in sorted(RULESPEC_ATOMIC_MODULE_ROOTS)]
    invalid_flat_roots = [
        path for path in flat_roots if path.exists() or path.is_symlink()
    ]
    if invalid_flat_roots:
        raise ValueError(
            "RuleSpec checkout contains repository-root atomic content; atomic roots "
            "must live under rulespec-<country>/<jurisdiction>/<content-root>: "
            + ", ".join(str(path) for path in invalid_flat_roots)
        )

    country = checkout.name.removeprefix("rulespec-")
    jurisdiction_pattern = re.compile(rf"{re.escape(country)}(?:-[a-z0-9]+)*")
    modules: list[Path] = []
    for content_root in sorted(checkout.iterdir()):
        if content_root.is_symlink():
            raise ValueError(
                f"RuleSpec checkout content must not contain symlinks: {content_root}"
            )
        root_markers = [
            content_root / root_name
            for root_name in sorted(RULESPEC_FILESYSTEM_ROOTS)
            if (content_root / root_name).exists()
            or (content_root / root_name).is_symlink()
        ]
        if jurisdiction_pattern.fullmatch(content_root.name) is None:
            if root_markers:
                raise ValueError(
                    "RuleSpec content roots must be under a matching direct "
                    f"jurisdiction directory: {content_root}"
                )
            continue
        if not content_root.is_dir():
            raise ValueError(
                "RuleSpec jurisdiction roots must be regular direct directories: "
                f"{content_root}"
            )
        if canonical_rulespec_root_identity(content_root) is None:
            raise ValueError(f"Noncanonical RuleSpec jurisdiction root: {content_root}")
        for candidate in sorted(content_root.rglob("*")):
            if candidate.is_symlink():
                raise ValueError(
                    f"RuleSpec content must not contain symlinks: {candidate}"
                )
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(content_root)
            yaml_like = candidate.suffix.lower() in {".yaml", ".yml"}
            if yaml_like and candidate.suffix != RULESPEC_FILE_SUFFIX:
                raise ValueError(
                    f"RuleSpec content must use the exact .yaml extension: {candidate}"
                )
            if candidate.suffix != RULESPEC_FILE_SUFFIX:
                continue
            if not relative.parts or relative.parts[0] not in RULESPEC_FILESYSTEM_ROOTS:
                raise ValueError(
                    "RuleSpec YAML must be beneath exactly one canonical filesystem "
                    f"root {sorted(RULESPEC_FILESYSTEM_ROOTS)}: {candidate}"
                )
            if relative.parts[0] not in RULESPEC_ATOMIC_MODULE_ROOTS:
                continue
            if candidate.name.endswith(RULESPEC_TEST_FILE_SUFFIX):
                continue
            modules.append(candidate)
    return tuple(modules)


def iter_jurisdiction_content_dirs(workspace_root: Path) -> list[tuple[str, Path]]:
    """Enumerate direct jurisdiction roots of one explicit country checkout."""

    lexical = _lexical_rulespec_path(workspace_root)
    if lexical is None:
        return []
    checkout = lexical.resolve()
    return [
        (name, checkout / name)
        for name in sorted(
            jurisdiction_subdir_names(checkout, allow_composition_specs=True)
        )
    ]


def _is_jurisdiction_dir_name(name: str, suffix: str) -> bool:
    """Return whether a directory name is a jurisdiction dir of ``rulespec-<suffix>``."""
    if re.fullmatch(r"[a-z]{2}", suffix) is None:
        return False
    return re.fullmatch(rf"{re.escape(suffix)}(?:-[a-z0-9]+)*", name) is not None


def _nearest_git_boundary(path: Path) -> Path | None:
    """Return the nearest lexical `.git` boundary without invoking Git."""

    for candidate in (path, *path.parents):
        marker = candidate / ".git"
        if marker.is_symlink():
            raise _GitProbeError(f"Git boundary is a symlink: {marker}")
        if marker.exists():
            return candidate
    return None


def _git_top_level(root: str) -> Path | None:
    """Return the exact Git worktree root containing ``root``, when present."""

    try:
        completed = subprocess.run(
            ["git", "-C", root, "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _GitProbeError(
            f"Could not inspect Git top-level for {root}",
            category=_git_probe_exception_category(exc),
        ) from exc
    if completed.returncode != 0:
        return None
    top_level = completed.stdout.strip()
    if not top_level:
        return None
    return Path(top_level).resolve()


def _git_origin_repo_name(root: str) -> str | None:
    """Best-effort repository basename from Git origin."""
    try:
        completed = subprocess.run(
            ["git", "-C", root, "remote", "get-url", "origin"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _GitProbeError(
            f"Could not inspect Git origin for {root}",
            category=_git_probe_exception_category(exc),
        ) from exc
    if completed.returncode != 0:
        return None
    remote = completed.stdout.strip().rstrip("/")
    if not remote:
        return None
    name = remote.rsplit("/", 1)[-1]
    return name.removesuffix(".git") or None


def _git_config_input_paths(root: str) -> tuple[Path, ...]:
    """Return loaded configs and every configured include target.

    Git omits missing, empty, and inactive conditional include files from its
    origin list. Include directives still appear in their containing config,
    so collect their resolved targets from the same bounded Git probe.
    """

    custom_config_environment = {
        name
        for name in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM")
        if name in os.environ
    }
    if custom_config_environment:
        raise _GitProbeError(
            f"Custom Git configuration environment observed for {root}",
            category="custom-config-environment",
        )
    system_config_inputs = _git_system_config_input_paths(root)
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                root,
                "config",
                "--show-origin",
                "--null",
                "--list",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _GitProbeError(
            f"Could not inspect Git configuration inputs for {root}",
            category=_git_probe_exception_category(exc),
        ) from exc
    if completed.returncode != 0:
        raise _GitProbeError(
            f"Could not inspect Git configuration inputs for {root}",
            category="unavailable",
        )
    fields = completed.stdout.removesuffix("\0").split("\0")
    if len(fields) % 2 != 0:
        raise _GitProbeError(
            f"Could not parse Git configuration inputs for {root}",
            category="invalid-output",
        )
    inputs = set(system_config_inputs)
    for origin, setting in zip(fields[::2], fields[1::2], strict=True):
        if not origin.startswith("file:"):
            raise _GitProbeError(
                f"Non-file Git configuration observed for {root}",
                category="non-file-config",
            )
        key, separator, value = setting.partition("\n")
        normalized_key = key.casefold()
        is_include = separator and (
            normalized_key == "include.path"
            or (
                normalized_key.startswith("includeif.")
                and normalized_key.endswith(".path")
            )
        )
        origin_path = _git_config_file_path(root, origin.removeprefix("file:"))
        if origin_path is None:
            raise _GitProbeError(
                f"Could not resolve Git configuration input for {root}",
                category="invalid-config-path",
            )
        inputs.add(origin_path)
        if not is_include:
            continue
        include_path = _git_config_file_path(
            str(origin_path.parent),
            value,
        )
        if include_path is None:
            raise _GitProbeError(
                f"Could not resolve Git configuration include for {root}",
                category="invalid-include-path",
            )
        inputs.add(include_path)
        if len(inputs) > 1024:
            raise _GitProbeError(
                f"Too many Git configuration inputs for {root}",
                category="input-limit",
            )
    return tuple(sorted(inputs, key=str))


def _git_system_config_input_paths(root: str) -> tuple[Path, ...]:
    """Return Git's platform-specific system config candidates, even if absent."""

    try:
        completed = subprocess.run(
            ["git", "-C", root, "var", "GIT_CONFIG_SYSTEM"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _GitProbeError(
            f"Could not inspect Git system configuration inputs for {root}",
            category=_git_probe_exception_category(exc),
        ) from exc
    raw_paths = completed.stdout.splitlines()
    if completed.returncode != 0 or not raw_paths or len(raw_paths) > 16:
        raise _GitProbeError(
            f"Could not inspect Git system configuration inputs for {root}",
            category="unavailable",
        )
    paths = tuple(_git_config_file_path(root, raw_path) for raw_path in raw_paths)
    if any(path is None for path in paths):
        raise _GitProbeError(
            f"Could not resolve Git system configuration input for {root}",
            category="invalid-config-path",
        )
    return tuple(path for path in paths if path is not None)


def _git_config_file_path(base: str, raw_path: str) -> Path | None:
    """Resolve one Git config path without accepting aliases or expansions."""

    if not raw_path or "\0" in raw_path or raw_path.startswith("%("):
        return None
    try:
        path = Path(raw_path).expanduser()
    except (KeyError, RuntimeError):
        return None
    if not path.is_absolute():
        path = Path(base) / path
    return _lexical_rulespec_path(Path(os.path.abspath(path)))


def _git_probe_exception_category(exc: BaseException) -> str:
    """Return a bounded, non-secret category for a failed Git process launch."""

    if isinstance(exc, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(exc, OSError):
        return f"oserror-{exc.errno}" if exc.errno is not None else "oserror"
    return "subprocess-error"
