"""Authenticated concept rewrites for legacy exact-dependent migrations."""

from __future__ import annotations

import copy
import re
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from .rulespec_path_migration import rulespec_identity

_CONCEPT_NAME = re.compile(r"[a-z][a-z0-9_]*")
_NUMERIC_LITERAL = re.compile(r"[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
_IDENTIFIER_CHARACTER = "A-Za-z0-9_"


class LegacyExactDependentConceptError(ValueError):
    """Raised when a concept rewrite cannot be proved behavior-preserving."""


def _load_mapping(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError, RecursionError) as exc:
        raise LegacyExactDependentConceptError(
            f"{label} is not valid UTF-8 YAML"
        ) from exc
    if not isinstance(payload, dict):
        raise LegacyExactDependentConceptError(f"{label} is not a YAML mapping")
    return payload


def _rules_by_name(payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return {}
    result: dict[str, dict[str, object]] = {}
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        name = rule.get("name")
        if not isinstance(name, str) or _CONCEPT_NAME.fullmatch(name) is None:
            continue
        if name in result:
            raise LegacyExactDependentConceptError(
                f"RuleSpec module exports duplicate concept {name!r}"
            )
        result[name] = rule
    return result


def _parse_date(value: object, *, label: str) -> date:
    if not isinstance(value, str):
        raise LegacyExactDependentConceptError(f"{label} is not an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise LegacyExactDependentConceptError(f"{label} is not an ISO date") from exc


def _formula_segments(formula: str) -> tuple[tuple[bool, str], ...]:
    """Split formula text into unquoted and quoted segments without parsing it."""

    segments: list[tuple[bool, str]] = []
    start = 0
    index = 0
    quote: str | None = None
    while index < len(formula):
        character = formula[index]
        if quote is None:
            if character in {'"', "'"}:
                if start < index:
                    segments.append((True, formula[start:index]))
                start = index
                quote = character
            index += 1
            continue
        if character == "\\":
            index = min(index + 2, len(formula))
            continue
        if character == quote:
            if index + 1 < len(formula) and formula[index + 1] == quote:
                index += 2
                continue
            index += 1
            segments.append((False, formula[start:index]))
            start = index
            quote = None
            continue
        index += 1
    if start < len(formula):
        segments.append((quote is None, formula[start:]))
    return tuple(segments)


def _formula_mentions(formula: object, concept: str) -> bool:
    if not isinstance(formula, str):
        return False
    pattern = re.compile(
        rf"(?<![{_IDENTIFIER_CHARACTER}]){re.escape(concept)}"
        rf"(?![{_IDENTIFIER_CHARACTER}])"
    )
    return any(
        unquoted and pattern.search(segment) is not None
        for unquoted, segment in _formula_segments(formula)
    )


def _dependent_use_windows(
    payload: Mapping[str, object], concept: str
) -> tuple[tuple[date, date], ...]:
    rules = payload.get("rules")
    windows: list[tuple[date, date]] = []
    if not isinstance(rules, list):
        return ()
    for rule_index, rule in enumerate(rules):
        versions = rule.get("versions") if isinstance(rule, dict) else None
        if not isinstance(versions, list):
            continue
        for version_index, version in enumerate(versions):
            if not isinstance(version, dict) or not _formula_mentions(
                version.get("formula"), concept
            ):
                continue
            start = _parse_date(
                version.get("effective_from"),
                label=(
                    f"dependent rule[{rule_index}].versions[{version_index}]"
                    ".effective_from"
                ),
            )
            end_raw = version.get("effective_to")
            end = (
                _parse_date(
                    end_raw,
                    label=(
                        f"dependent rule[{rule_index}].versions[{version_index}]"
                        ".effective_to"
                    ),
                )
                if end_raw is not None
                else date.max
            )
            if end < start:
                raise LegacyExactDependentConceptError(
                    f"dependent concept {concept!r} has an inverted use window"
                )
            windows.append((start, end))
    return tuple(sorted(set(windows)))


def _parameter_versions(
    rule: Mapping[str, object], *, concept: str
) -> tuple[tuple[date, date | None, Decimal], ...]:
    if rule.get("kind") != "parameter" or rule.get("indexed_by") is not None:
        raise LegacyExactDependentConceptError(
            f"legacy concept {concept!r} is not a scalar parameter"
        )
    versions = rule.get("versions")
    if not isinstance(versions, list) or not versions:
        raise LegacyExactDependentConceptError(f"parameter {concept!r} has no versions")
    parsed: list[tuple[date, date | None, Decimal]] = []
    for index, version in enumerate(versions):
        if not isinstance(version, dict):
            raise LegacyExactDependentConceptError(
                f"parameter {concept!r} version {index} is malformed"
            )
        formula = version.get("formula")
        formula_text = str(formula).strip() if isinstance(formula, (str, int)) else ""
        if _NUMERIC_LITERAL.fullmatch(formula_text) is None:
            raise LegacyExactDependentConceptError(
                f"parameter {concept!r} version {index} is not a numeric literal"
            )
        try:
            value = Decimal(formula_text)
        except InvalidOperation as exc:
            raise LegacyExactDependentConceptError(
                f"parameter {concept!r} version {index} is not numeric"
            ) from exc
        start = _parse_date(
            version.get("effective_from"),
            label=f"parameter {concept!r} version {index} effective_from",
        )
        end_raw = version.get("effective_to")
        if end_raw is not None:
            raise LegacyExactDependentConceptError(
                f"parameter {concept!r} version {index} uses effective_to, which "
                "cannot be proved by the scalar-parameter runtime contract"
            )
        end = (
            _parse_date(
                end_raw,
                label=f"parameter {concept!r} version {index} effective_to",
            )
            if end_raw is not None
            else None
        )
        if end is not None and end < start:
            raise LegacyExactDependentConceptError(
                f"parameter {concept!r} version {index} has an inverted period"
            )
        parsed.append((start, end, value))
    if [item[0] for item in parsed] != sorted(item[0] for item in parsed):
        raise LegacyExactDependentConceptError(
            f"parameter {concept!r} versions are not ordered"
        )
    return tuple(parsed)


def _active_parameter_value(
    versions: Sequence[tuple[date, date | None, Decimal]], at: date
) -> Decimal | None:
    active: Decimal | None = None
    for start, end, value in versions:
        if start <= at and (end is None or at <= end):
            active = value
    return active


def _parameters_equal_over_windows(
    old_versions: Sequence[tuple[date, date | None, Decimal]],
    new_versions: Sequence[tuple[date, date | None, Decimal]],
    windows: Sequence[tuple[date, date]],
) -> bool:
    for window_start, window_end in windows:
        probes = {window_start, window_end}
        for versions in (old_versions, new_versions):
            for start, end, _value in versions:
                if window_start <= start <= window_end:
                    probes.add(start)
                if end is not None and end < date.max:
                    after = end + timedelta(days=1)
                    if window_start <= after <= window_end:
                        probes.add(after)
        for probe in probes:
            old_value = _active_parameter_value(old_versions, probe)
            new_value = _active_parameter_value(new_versions, probe)
            if old_value is None or new_value is None or old_value != new_value:
                return False
    return True


def _compatible_parameter_surface(
    old_rule: Mapping[str, object], new_rule: Mapping[str, object]
) -> bool:
    return all(
        old_rule.get(field) == new_rule.get(field)
        for field in ("kind", "dtype", "entity", "unit", "period", "indexed_by")
    )


def derive_exact_dependent_parameter_replacements(
    *,
    dependent_primary_raw: bytes,
    retained_modules: Sequence[tuple[Path, Path, bytes, bytes]],
) -> dict[str, str]:
    """Derive unique active-period-equivalent parameter renames.

    The returned mapping is intentionally suitable for the existing exact-token
    replacement ledger: it contains the full legacy import reference and the bare
    formula symbol. No mapping is emitted when the canonical successor preserves
    the old fragment.
    """

    dependent = _load_mapping(dependent_primary_raw, label="legacy exact dependent")
    imports = dependent.get("imports")
    if imports is None:
        return {}
    if not isinstance(imports, list) or not all(
        isinstance(item, str) for item in imports
    ):
        raise LegacyExactDependentConceptError(
            "legacy exact dependent imports are malformed"
        )
    dependent_rules = _rules_by_name(dependent)

    replacements: dict[str, str] = {}
    destination_fragments: set[str] = set()
    for source_path, destination_path, source_raw, destination_raw in retained_modules:
        source_base = rulespec_identity(source_path)
        destination_base = rulespec_identity(destination_path)
        source_rules = _rules_by_name(
            _load_mapping(source_raw, label=f"retained legacy module {source_path}")
        )
        destination_rules = _rules_by_name(
            _load_mapping(
                destination_raw,
                label=f"retained canonical module {destination_path}",
            )
        )
        for imported in imports:
            prefix = f"{source_base}#"
            if not imported.startswith(prefix):
                continue
            old_name = imported.removeprefix(prefix)
            if _CONCEPT_NAME.fullmatch(old_name) is None:
                raise LegacyExactDependentConceptError(
                    f"legacy imported concept {imported!r} is malformed"
                )
            same_fragment_imports = [
                item for item in imports if item.rsplit("#", 1)[-1] == old_name
            ]
            if same_fragment_imports != [imported] or old_name in dependent_rules:
                raise LegacyExactDependentConceptError(
                    f"legacy concept {old_name!r} is not an unambiguous imported symbol"
                )
            if old_name in destination_rules:
                continue
            old_rule = source_rules.get(old_name)
            if old_rule is None:
                raise LegacyExactDependentConceptError(
                    f"legacy import {imported!r} is not exported by its source module"
                )
            windows = _dependent_use_windows(dependent, old_name)
            if not windows:
                raise LegacyExactDependentConceptError(
                    f"retired imported concept {imported!r} has no formula use window"
                )
            old_versions = _parameter_versions(old_rule, concept=old_name)
            candidates: list[str] = []
            for new_name, new_rule in destination_rules.items():
                if not _compatible_parameter_surface(old_rule, new_rule):
                    continue
                try:
                    new_versions = _parameter_versions(new_rule, concept=new_name)
                except LegacyExactDependentConceptError as exc:
                    if "uses effective_to" in str(exc):
                        raise
                    continue
                if _parameters_equal_over_windows(old_versions, new_versions, windows):
                    candidates.append(new_name)
            if len(candidates) != 1:
                detail = "none" if not candidates else ", ".join(sorted(candidates))
                raise LegacyExactDependentConceptError(
                    "retired imported parameter has no unique active-period-equivalent "
                    f"canonical successor: {imported!r}; candidates: {detail}"
                )
            new_name = candidates[0]
            destination_import = f"{destination_base}#{new_name}"
            if any(
                item == destination_import or item.rsplit("#", 1)[-1] == new_name
                for item in imports
            ):
                raise LegacyExactDependentConceptError(
                    f"canonical concept {new_name!r} is already imported"
                )
            if old_name in replacements and replacements[old_name] != new_name:
                raise LegacyExactDependentConceptError(
                    f"legacy concept {old_name!r} has conflicting canonical successors"
                )
            if new_name in destination_fragments:
                raise LegacyExactDependentConceptError(
                    "multiple retired concepts would collapse into one imported "
                    f"successor {new_name!r}"
                )
            destination_fragments.add(new_name)
            replacements[imported] = destination_import
            replacements[old_name] = new_name
    return replacements


def canonicalized_concept_replacements(
    replacements: Mapping[str, str],
    *,
    path_replacements: Mapping[str, str],
) -> dict[str, str]:
    """Translate full legacy refs so they apply after path-only rewriting."""

    canonical: dict[str, str] = {}
    for old, new in replacements.items():
        canonical_old = old
        for path_old in sorted(path_replacements, key=lambda item: (-len(item), item)):
            if canonical_old == path_old or canonical_old.startswith(f"{path_old}#"):
                canonical_old = (
                    path_replacements[path_old] + canonical_old[len(path_old) :]
                )
                break
        canonical[canonical_old] = new
    return canonical


def _replace_identifier(text: str, old: str, new: str) -> str:
    pattern = re.compile(
        rf"(?<![{_IDENTIFIER_CHARACTER}]){re.escape(old)}"
        rf"(?![{_IDENTIFIER_CHARACTER}])"
    )
    return "".join(
        pattern.sub(lambda _match: new, segment) if unquoted else segment
        for unquoted, segment in _formula_segments(text)
    )


def _expected_primary_payload(
    payload: dict[str, object], replacements: Mapping[str, str]
) -> dict[str, object]:
    expected = copy.deepcopy(payload)
    full = {old: new for old, new in replacements.items() if "#" in old}
    bare = {old: new for old, new in replacements.items() if "#" not in old}
    imports = expected.get("imports")
    if isinstance(imports, list):
        expected["imports"] = [full.get(item, item) for item in imports]
    rules = expected.get("rules")
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            versions = rule.get("versions")
            if isinstance(versions, list):
                for version in versions:
                    formula = (
                        version.get("formula") if isinstance(version, dict) else None
                    )
                    if not isinstance(formula, str):
                        continue
                    for old, new in bare.items():
                        formula = _replace_identifier(formula, old, new)
                    version["formula"] = formula
            metadata = rule.get("metadata")
            proof = metadata.get("proof") if isinstance(metadata, dict) else None
            atoms = proof.get("atoms") if isinstance(proof, dict) else None
            if not isinstance(atoms, list):
                continue
            for atom in atoms:
                imported = atom.get("import") if isinstance(atom, dict) else None
                target = imported.get("target") if isinstance(imported, dict) else None
                if isinstance(target, str) and target in full:
                    imported["target"] = full[target]
                output = imported.get("output") if isinstance(imported, dict) else None
                if isinstance(output, str) and output in bare:
                    imported["output"] = bare[output]
    return expected


def _replace_test_mapping_keys(value: object, full: Mapping[str, str]) -> object:
    if isinstance(value, dict):
        return {
            full.get(key, key)
            if isinstance(key, str)
            else key: _replace_test_mapping_keys(item, full)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_test_mapping_keys(item, full) for item in value]
    return value


def validate_exact_dependent_concept_rewrite(
    *,
    path_rewritten_raw: bytes,
    concept_rewritten_raw: bytes,
    replacements: Mapping[str, str],
    primary: bool,
) -> None:
    """Reject concept substitutions outside imports, formulas, proof imports, and tests."""

    try:
        before = yaml.safe_load(path_rewritten_raw.decode("utf-8"))
        after = yaml.safe_load(concept_rewritten_raw.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError, RecursionError) as exc:
        raise LegacyExactDependentConceptError(
            "exact-dependent concept rewrite produced invalid YAML"
        ) from exc
    if primary:
        if not isinstance(before, dict):
            raise LegacyExactDependentConceptError(
                "exact-dependent primary is not a YAML mapping"
            )
        expected: object = _expected_primary_payload(before, replacements)
    else:
        full = {old: new for old, new in replacements.items() if "#" in old}
        expected = _replace_test_mapping_keys(before, full)
    if after != expected:
        raise LegacyExactDependentConceptError(
            "exact-dependent concept rewrite changed an unauthorized YAML surface"
        )
