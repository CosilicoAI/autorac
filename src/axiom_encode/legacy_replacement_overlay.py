"""Authenticated legacy-replacement transformations for isolated checkouts."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
from pathlib import Path

from .constants import RULESPEC_ATOMIC_MODULE_ROOTS
from .legacy_replacement import (
    DESTINATION_PREDECESSOR_ABSENT,
    DESTINATION_PREDECESSOR_CANONICALIZED_UNOWNED_DUPLICATE,
    ENCODING_MANIFEST_DIR,
    LegacyReplacementContract,
    LegacyReplacementFile,
    LegacyReplacementRewrite,
)
from .repo_routing import jurisdiction_subdir_names
from .rulespec_path_migration import PathMigrationPlanError, canonical_destination


class LegacyReplacementOverlayError(ValueError):
    """Raised when an authenticated replacement cannot be staged exactly."""


def _manifest_jurisdiction_subdir_names(
    source_checkout: Path,
    *,
    country: str,
) -> frozenset[str]:
    """Return validated jurisdiction directories in the manifest tree."""

    manifest_root = source_checkout / ENCODING_MANIFEST_DIR
    if not manifest_root.exists() and not manifest_root.is_symlink():
        return frozenset()
    if manifest_root.is_symlink() or not manifest_root.is_dir():
        raise LegacyReplacementOverlayError(
            f"Legacy replacement manifest root is unsafe: {manifest_root}"
        )
    jurisdiction_pattern = re.compile(rf"{re.escape(country)}(?:-[a-z0-9]+)*")
    jurisdictions: set[str] = set()
    for child in manifest_root.iterdir():
        if jurisdiction_pattern.fullmatch(child.name) is None:
            continue
        if child.is_symlink() or not child.is_dir():
            raise LegacyReplacementOverlayError(
                f"Legacy replacement manifest jurisdiction is unsafe: {child}"
            )
        jurisdictions.add(child.name)
    return frozenset(jurisdictions)


def replacement_excluded_jurisdictions(
    source_checkout: Path,
    *,
    active_jurisdiction: str,
) -> frozenset[str]:
    """Return sibling jurisdictions excluded from a replacement capability.

    Replacement validation admits the active jurisdiction and its
    country ancestors. Unrelated siblings remain outside the isolated overlay,
    so their paths cannot broaden or block the authenticated replacement. The
    live checkout is never changed.
    """

    parts = active_jurisdiction.split("-")
    admitted = {"-".join(parts[:length]) for length in range(1, len(parts) + 1)}
    content_jurisdictions = jurisdiction_subdir_names(
        source_checkout,
        allow_composition_specs=True,
    )
    if active_jurisdiction not in content_jurisdictions:
        raise LegacyReplacementOverlayError(
            "Replacement source is missing the active jurisdiction: "
            f"{active_jurisdiction}"
        )
    manifest_jurisdictions = _manifest_jurisdiction_subdir_names(
        source_checkout,
        country=active_jurisdiction.split("-", 1)[0],
    )
    return frozenset((content_jurisdictions | manifest_jurisdictions) - admitted)


def legacy_replacement_excluded_jurisdictions(
    source_checkout: Path,
    *,
    active_jurisdiction: str,
) -> frozenset[str]:
    """Backward-compatible alias for replacement overlay scoping."""

    return replacement_excluded_jurisdictions(
        source_checkout,
        active_jurisdiction=active_jurisdiction,
    )


def scope_replacement_overlay(
    overlay_checkout: Path,
    *,
    active_jurisdiction: str,
) -> None:
    """Remove unrelated jurisdictions from one isolated replacement overlay."""

    excluded = replacement_excluded_jurisdictions(
        overlay_checkout,
        active_jurisdiction=active_jurisdiction,
    )
    manifest_root = overlay_checkout / ENCODING_MANIFEST_DIR
    for jurisdiction in sorted(excluded):
        for candidate in (
            overlay_checkout / jurisdiction,
            manifest_root / jurisdiction,
        ):
            if not candidate.exists() and not candidate.is_symlink():
                continue
            if candidate.is_symlink() or not candidate.is_dir():
                raise LegacyReplacementOverlayError(
                    f"Replacement overlay jurisdiction is unsafe: {candidate}"
                )
            shutil.rmtree(candidate)


def scope_canonical_replacement_overlay(
    overlay_checkout: Path,
    *,
    active_jurisdiction: str,
) -> tuple[Path, ...]:
    """Scope one canonical replacement around frozen colon-path predecessors.

    The hard-cut engine rejects every path with a colon-bearing component. A
    canonical fresh replacement must therefore validate in an isolated view
    that excludes those pre-hard-cut entries while retaining every canonical
    path and all canonical validation debt. The caller binds the complete live
    pre-apply tree identity separately; this function never mutates that tree.
    """

    scope_replacement_overlay(
        overlay_checkout,
        active_jurisdiction=active_jurisdiction,
    )
    active_root = overlay_checkout / active_jurisdiction
    if active_root.is_symlink() or not active_root.is_dir():
        raise LegacyReplacementOverlayError(
            f"Canonical replacement jurisdiction is unsafe: {active_root}"
        )

    omitted: list[Path] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise LegacyReplacementOverlayError(
                f"Cannot inspect canonical replacement overlay: {directory}"
            ) from exc
        for entry in entries:
            candidate = Path(entry.path)
            relative = candidate.relative_to(active_root)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise LegacyReplacementOverlayError(
                    f"Cannot inspect canonical replacement overlay entry: {candidate}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise LegacyReplacementOverlayError(
                    f"Canonical replacement overlay contains a symlink: {candidate}"
                )
            is_directory = stat.S_ISDIR(metadata.st_mode)
            is_file = stat.S_ISREG(metadata.st_mode)
            if not is_directory and not is_file:
                raise LegacyReplacementOverlayError(
                    "Canonical replacement overlay contains a special file: "
                    f"{candidate}"
                )

            if any(":" in part for part in relative.parts):
                checkout_relative = Path(active_jurisdiction) / relative
                try:
                    destination = canonical_destination(checkout_relative)
                except PathMigrationPlanError as exc:
                    raise LegacyReplacementOverlayError(
                        "Colon-path replacement predecessor has no canonical "
                        f"destination: {checkout_relative}"
                    ) from exc
                if destination == checkout_relative:
                    raise LegacyReplacementOverlayError(
                        "Colon-path replacement predecessor did not canonicalize: "
                        f"{checkout_relative}"
                    )
                if is_directory:
                    shutil.rmtree(candidate)
                else:
                    candidate.unlink()
                omitted.append(checkout_relative)
                continue
            if is_directory:
                visit(candidate)

    for name in sorted(RULESPEC_ATOMIC_MODULE_ROOTS):
        root = active_root / name
        if not root.exists() and not root.is_symlink():
            continue
        if root.is_symlink() or not root.is_dir():
            raise LegacyReplacementOverlayError(
                f"Canonical replacement content root is unsafe: {root}"
            )
        visit(root)
    return tuple(omitted)


def scope_legacy_replacement_overlay(
    overlay_checkout: Path,
    *,
    active_jurisdiction: str,
) -> None:
    """Backward-compatible alias for replacement overlay scoping."""

    scope_replacement_overlay(
        overlay_checkout,
        active_jurisdiction=active_jurisdiction,
    )


def _overlay_path(checkout_root: Path, relative: Path) -> Path:
    """Resolve one contract path beneath an already isolated checkout copy."""

    relative = Path(relative)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.as_posix() != str(relative)
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise LegacyReplacementOverlayError(
            f"Legacy replacement overlay path is unsafe: {relative}"
        )
    root = Path(checkout_root).resolve(strict=True)
    target = root / relative
    try:
        target.resolve(strict=False).relative_to(root)
    except (OSError, ValueError) as exc:
        raise LegacyReplacementOverlayError(
            f"Legacy replacement overlay path escapes checkout: {relative}"
        ) from exc
    return target


def _require_overlay_file(
    checkout_root: Path,
    item: LegacyReplacementFile,
    *,
    label: str,
) -> Path:
    target = _overlay_path(checkout_root, item.path)
    if target.is_symlink() or not target.is_file():
        raise LegacyReplacementOverlayError(
            f"{label} is not a regular file: {item.path.as_posix()}"
        )
    try:
        observed = hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError as exc:
        raise LegacyReplacementOverlayError(
            f"Cannot read {label}: {item.path.as_posix()}"
        ) from exc
    if observed != item.sha256:
        raise LegacyReplacementOverlayError(f"{label} changed: {item.path.as_posix()}")
    return target


def stage_legacy_replacement_overlay(
    contract: LegacyReplacementContract,
    checkout_root: Path,
) -> None:
    """Apply an authenticated replacement plan to an isolated checkout copy.

    Every original byte identity and destination collision is checked before the
    copy is mutated. The caller may then overlay the freshly generated primary
    and companion at ``contract.destination`` and validate the post-replacement
    repository shape.
    """

    if type(contract) is not LegacyReplacementContract:
        raise LegacyReplacementOverlayError(
            "Legacy replacement overlay contract is malformed"
        )
    root = Path(checkout_root)
    if root.is_symlink() or not root.is_dir():
        raise LegacyReplacementOverlayError(
            "Legacy replacement overlay checkout is not a regular directory"
        )

    deleted_targets = [
        _require_overlay_file(
            root,
            item,
            label="Legacy replacement overlay input",
        )
        for item in contract.deleted_files
    ]
    manifest_target = _require_overlay_file(
        root,
        contract.legacy_manifest,
        label="Legacy replacement ownership manifest",
    )

    rewrite_targets: list[tuple[Path, LegacyReplacementRewrite]] = []
    for rewrite in contract.rewrites:
        target = _require_overlay_file(
            root,
            LegacyReplacementFile(rewrite.path, rewrite.before_sha256, b""),
            label="Legacy replacement rewrite input",
        )
        if hashlib.sha256(rewrite.raw).hexdigest() != rewrite.after_sha256:
            raise LegacyReplacementOverlayError(
                "Legacy replacement rewrite output does not match its authenticated "
                f"digest: {rewrite.path.as_posix()}"
            )
        rewrite_targets.append((target, rewrite))

    exact_targets: list[tuple[Path, LegacyReplacementFile]] = []
    for dependent in contract.exact_dependents:
        _require_overlay_file(
            root,
            dependent.legacy_manifest,
            label="Legacy exact dependent ownership manifest",
        )
        legacy_by_path = {item.path: item for item in dependent.legacy_files}
        for live_file in dependent.live_files:
            legacy_file = legacy_by_path.get(live_file.path)
            if legacy_file is None:
                raise LegacyReplacementOverlayError(
                    "Legacy exact dependent live file lacks an authenticated input: "
                    f"{live_file.path.as_posix()}"
                )
            target = _require_overlay_file(
                root,
                legacy_file,
                label="Legacy exact dependent overlay input",
            )
            if hashlib.sha256(live_file.raw).hexdigest() != live_file.sha256:
                raise LegacyReplacementOverlayError(
                    "Legacy exact dependent output does not match its authenticated "
                    f"digest: {live_file.path.as_posix()}"
                )
            exact_targets.append((target, live_file))

    retained_deleted_targets: list[Path] = []
    retained_manifest_targets: list[Path] = []
    for successor in contract.retained_successors:
        retained_deleted_targets.extend(
            _require_overlay_file(
                root,
                item,
                label="Legacy retained-successor input",
            )
            for item in successor.legacy_files
        )
        retained_manifest_targets.append(
            _require_overlay_file(
                root,
                successor.legacy_manifest,
                label="Legacy retained-successor ownership manifest",
            )
        )
        _require_overlay_file(
            root,
            successor.successor_manifest,
            label="Legacy retained-successor signed manifest",
        )
        for item in successor.successor_files:
            _require_overlay_file(
                root,
                item,
                label="Legacy retained-successor live file",
            )

    metadata_targets: list[tuple[Path, LegacyReplacementRewrite]] = []
    for reconciliation in contract.metadata_reconciliations:
        target = _require_overlay_file(
            root,
            LegacyReplacementFile(
                reconciliation.path,
                reconciliation.before_sha256,
                b"",
            ),
            label="Legacy metadata reconciliation input",
        )
        if (
            hashlib.sha256(reconciliation.raw).hexdigest()
            != reconciliation.after_sha256
        ):
            raise LegacyReplacementOverlayError(
                "Legacy metadata reconciliation output does not match its digest: "
                f"{reconciliation.path.as_posix()}"
            )
        metadata_targets.append((target, reconciliation))

    predecessor_present = bool(contract.destination_predecessor_files)
    if (
        predecessor_present
        and contract.destination_predecessor_class
        != DESTINATION_PREDECESSOR_CANONICALIZED_UNOWNED_DUPLICATE
    ) or (
        not predecessor_present
        and contract.destination_predecessor_class != DESTINATION_PREDECESSOR_ABSENT
    ):
        raise LegacyReplacementOverlayError(
            "Legacy replacement canonical destination predecessor classification "
            "differs from its evidence"
        )
    predecessor_targets = [
        _require_overlay_file(
            root,
            item,
            label="Legacy replacement canonical destination predecessor",
        )
        for item in contract.destination_predecessor_files
    ]
    if contract.source != contract.destination:
        destination = _overlay_path(root, contract.destination)
        destination_manifest = _overlay_path(
            root,
            (ENCODING_MANIFEST_DIR / contract.destination).with_suffix(".json"),
        )
        destination_group = (
            destination,
            destination.with_name(f"{destination.stem}.test.yaml"),
            destination_manifest,
        )
        admitted_predecessors = set(predecessor_targets)
        if admitted_predecessors and (
            destination not in admitted_predecessors
            or not admitted_predecessors.issubset(set(destination_group[:2]))
        ):
            raise LegacyReplacementOverlayError(
                "Legacy replacement canonical destination predecessor group is malformed"
            )
        for target in destination_group:
            if target not in admitted_predecessors and (
                target.exists() or target.is_symlink()
            ):
                raise LegacyReplacementOverlayError(
                    "Legacy replacement destination primary/companion/manifest "
                    f"collision: {target.relative_to(root).as_posix()}"
                )

    for target in deleted_targets:
        target.unlink()
    for target in retained_deleted_targets:
        target.unlink()
    for target in predecessor_targets:
        target.unlink()
    manifest_target.unlink()
    for target in retained_manifest_targets:
        target.unlink()
    for target, rewrite in rewrite_targets:
        target.write_bytes(rewrite.raw)
    for target, live_file in exact_targets:
        target.write_bytes(live_file.raw)
    for target, reconciliation in metadata_targets:
        target.write_bytes(reconciliation.raw)
