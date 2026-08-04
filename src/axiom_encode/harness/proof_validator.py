"""RuleSpec proof-tree validation.

This module intentionally validates proof structure without knowing how to
encode policy. It checks that executable RuleSpec atoms point to direct,
release-bound corpus source text or explicit RuleSpec import exports. Mutable
claim references are rejected.

It also derives *money proof obligations* (see
``find_missing_money_proof_atoms``). Where explicit proof validation is
per-file opt-in via ``module.proof_validation.required: true``, the money-atom
derivation is unconditional: every policy-bearing monetary value (currency
parameters, currency parameter-table cells, and currency literals inside
derived-rule formulas) is expected to carry a proof atom that cites a
provision. This closes the enforcement gap where a repository can ship
monetary parameters with no proof atoms at all because no file opts in.
"""

from __future__ import annotations

import contextlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping

import yaml

from ..corpus_resolver import (
    InvalidCorpusCitationError,
    require_canonical_corpus_citation_path,
    split_proof_evidence_text,
)


@dataclass(frozen=True)
class ProofValidationResult:
    """Result for standalone RuleSpec proof validation."""

    passed: bool
    issues: list[str]
    atoms_checked: int
    proof_required: bool


PROOF_ATOM_KINDS = frozenset(
    {
        "amount",
        "condition",
        "definition",
        "default",
        "effective_period",
        "exception",
        "formula",
        "import",
        "ordering",
        "parameter",
        "parameter_table",
        "predicate",
        "table_cell",
        "unit",
    }
)
POLICY_RULE_KINDS = frozenset({"derived", "derived_relation", "parameter"})


def find_plural_corpus_citation_path_issues(payload: object) -> list[str]:
    """Reject the retired plural corpus-source field anywhere in a RuleSpec.

    This accepts an already parsed payload so proof, validation, and signed
    manifest gates can enforce the same recursive hard cut without reparsing
    different byte snapshots.
    """

    plural_locations: list[str] = []
    seen_containers: set[int] = set()
    stack: list[tuple[str, object]] = [("$", payload)]
    while stack:
        location, value = stack.pop()
        if isinstance(value, (dict, list)):
            identity = id(value)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{location}.{key}"
                if key == "corpus_citation_paths":
                    plural_locations.append(child)
                stack.append((child, item))
        elif isinstance(value, list):
            stack.extend(
                (f"{location}[{index}]", item) for index, item in enumerate(value)
            )
    if not plural_locations:
        return []
    return [
        "Plural corpus source paths are not supported: replace "
        + ", ".join(sorted(plural_locations)[:5])
        + ("; ..." if len(plural_locations) > 5 else "")
        + " with exactly one resolver-attested `corpus_citation_path` per "
        "RuleSpec module. Encode other sources separately and import them."
    ]


def find_noncanonical_corpus_citation_path_issues(payload: object) -> list[str]:
    """Reject aliases in every singular corpus machine-identity field."""

    issues: list[str] = []
    seen_containers: set[int] = set()
    stack: list[tuple[str, object]] = [("$", payload)]
    while stack:
        location, value = stack.pop()
        if isinstance(value, (dict, list)):
            identity = id(value)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{location}.{key}"
                if key == "corpus_citation_path":
                    if not isinstance(item, str):
                        issues.append(
                            f"Corpus citation path at `{child}` must be a string."
                        )
                    else:
                        try:
                            require_canonical_corpus_citation_path(item)
                        except InvalidCorpusCitationError as exc:
                            issues.append(
                                f"Noncanonical corpus citation path at `{child}`: {exc}"
                            )
                stack.append((child, item))
        elif isinstance(value, list):
            stack.extend(
                (f"{location}[{index}]", item) for index, item in enumerate(value)
            )
    return issues


def find_rulespec_proof_issues(
    content: str,
    *,
    source_texts: Mapping[str, str | None] | None = None,
) -> list[str]:
    """Return proof-validation issues for a RuleSpec YAML document."""
    return validate_rulespec_proofs(content, source_texts=source_texts).issues


def proof_source_citation_paths(content: str) -> tuple[str, ...]:
    """Return direct corpus citations used by proof atoms in one RuleSpec."""

    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return ()
    if not isinstance(payload, dict):
        return ()
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return ()
    paths: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        proof = _rule_proof(rule)
        atoms = proof.get("atoms") if isinstance(proof, dict) else None
        if not isinstance(atoms, list):
            continue
        for atom in atoms:
            source = atom.get("source") if isinstance(atom, dict) else None
            if not isinstance(source, dict):
                continue
            citation_path = str(source.get("corpus_citation_path") or "").strip()
            if citation_path and citation_path not in paths:
                paths.append(citation_path)
    return tuple(paths)


def validate_rulespec_proofs(
    content: str,
    *,
    require_policy_proofs: bool = False,
    source_texts: Mapping[str, str | None] | None = None,
) -> ProofValidationResult:
    """Validate explicit proof trees in a RuleSpec YAML document.

    Strict proof validation is enabled with:

    ```yaml
    module:
      proof_validation:
        required: true
    ```

    When strict mode is off, any proof blocks that are present are still checked.
    """
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError) as exc:
        return ProofValidationResult(
            passed=False,
            issues=[f"RuleSpec proof YAML parse failed: {exc}"],
            atoms_checked=0,
            proof_required=False,
        )

    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return ProofValidationResult(
            passed=True,
            issues=[],
            atoms_checked=0,
            proof_required=False,
        )

    module = payload.get("module")
    proof_required = require_policy_proofs or _module_requires_proofs(module)
    issues = [
        *find_plural_corpus_citation_path_issues(payload),
        *find_noncanonical_corpus_citation_path_issues(payload),
    ]
    if isinstance(module, dict) and "source_claims" in module:
        issues.append(
            "Source claims are not supported: `module.source_claims` must be "
            "migrated to immutable release-bound corpus citation paths and "
            "direct proof `source` atoms."
        )

    rules = payload.get("rules")
    if not isinstance(rules, list):
        return ProofValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            atoms_checked=0,
            proof_required=proof_required,
        )

    atoms_checked = 0
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        rule_name = str(rule.get("name") or f"rules[{index}]").strip() or (
            f"rules[{index}]"
        )
        proof = _rule_proof(rule)
        if proof is None:
            if proof_required and _is_policy_bearing_rule(rule):
                issues.append(
                    "Proof missing: "
                    f"rule `{rule_name}` is policy-bearing and must declare "
                    "`metadata.proof.atoms`."
                )
            continue
        rule_issues, rule_atom_count = _validate_rule_proof(
            rule_name=rule_name,
            rule=rule,
            proof=proof,
            source_texts=source_texts,
        )
        issues.extend(rule_issues)
        atoms_checked += rule_atom_count

    return ProofValidationResult(
        passed=len(issues) == 0,
        issues=issues,
        atoms_checked=atoms_checked,
        proof_required=proof_required,
    )


def _module_requires_proofs(module: Any) -> bool:
    if not isinstance(module, dict):
        return False
    proof_validation = module.get("proof_validation")
    if isinstance(proof_validation, dict):
        return proof_validation.get("required") is True
    return False


def _rule_proof(rule: dict[str, Any]) -> dict[str, Any] | None:
    metadata = rule.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("proof"), dict):
        return metadata["proof"]
    proof = rule.get("proof")
    if isinstance(proof, dict):
        return proof
    return None


def _is_policy_bearing_rule(rule: dict[str, Any]) -> bool:
    kind = str(rule.get("kind") or "").strip()
    if kind in POLICY_RULE_KINDS:
        return True
    if kind in {"data_relation", "source_relation"}:
        return False
    return bool(rule.get("versions"))


def _validate_rule_proof(
    *,
    rule_name: str,
    rule: dict[str, Any],
    proof: dict[str, Any],
    source_texts: Mapping[str, str | None] | None,
) -> tuple[list[str], int]:
    atoms = proof.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        return (
            [
                "Proof malformed: "
                f"rule `{rule_name}` must declare non-empty "
                "`metadata.proof.atoms`."
            ],
            0,
        )

    issues: list[str] = []
    atoms_checked = 0
    for index, atom in enumerate(atoms):
        label = f"rule `{rule_name}` proof atom {index}"
        if not isinstance(atom, dict):
            issues.append(f"Proof atom malformed: {label} must be a mapping.")
            continue
        atoms_checked += 1
        issues.extend(
            _validate_proof_atom(
                atom=atom,
                label=label,
                rule=rule,
                source_texts=source_texts,
            )
        )
    return issues, atoms_checked


def _validate_proof_atom(
    *,
    atom: dict[str, Any],
    label: str,
    rule: dict[str, Any],
    source_texts: Mapping[str, str | None] | None,
) -> list[str]:
    issues: list[str] = []
    path = str(atom.get("path") or "").strip()
    if not path:
        issues.append(f"Proof atom missing path: {label} must declare `path`.")
    else:
        anchor_issue = _atom_anchor_issue(path=path, label=label, rule=rule)
        if anchor_issue is not None:
            issues.append(anchor_issue)

    kind = str(atom.get("kind") or "").strip()
    if kind not in PROOF_ATOM_KINDS:
        allowed = ", ".join(sorted(PROOF_ATOM_KINDS))
        issues.append(
            "Proof atom kind invalid: "
            f"{label} has kind `{kind or '<missing>'}`; allowed kinds are {allowed}."
        )

    proof_source_count = sum(
        1 for key in ("source", "import") if atom.get(key) is not None
    )
    if proof_source_count == 0:
        issues.append(
            "Proof atom missing evidence: "
            f"{label} must declare at least one of `source` or `import`."
        )

    if "source" in atom:
        issues.extend(
            _validate_source_proof_atom(
                source=atom.get("source"),
                label=label,
                kind=kind,
                rule_source=rule.get("source"),
                source_texts=source_texts,
            )
        )
    if "claim" in atom:
        issues.append(
            "Proof claim references are not supported: "
            f"{label} must cite an immutable release-bound corpus provision "
            "through proof atom `source` with `corpus_citation_path`, or an "
            "explicit RuleSpec `import`."
        )
    if "import" in atom:
        issues.extend(_validate_import_proof_atom(atom.get("import"), label))

    return issues


# A ``versions[i].values`` proof atom anchors a parameter *table* (the cells
# live under ``values``); a ``versions[i].formula`` atom anchors a scalar
# formula. Enforcing that an atom's declared anchor matches the version's actual
# shape makes ``versions[i].values`` the single, authoritative contract for
# table atoms across the whole proof system, rather than a shape only the
# money-atom gate happens to accept. Without this, a table atom could claim
# ``versions[i].formula`` (which no ``values``-only version has) and pass base
# validation while failing the money gate — the mutual-exclusivity trap from
# rulespec-be#31/#80 and axiom-encode#1032.
_ANCHOR_FIELD_PATTERN = re.compile(r"^versions\[(\d+)\]\.(values|formula)\b")


def _atom_anchor_issue(
    *,
    path: str,
    label: str,
    rule: dict[str, Any],
) -> str | None:
    """Return an issue when a ``versions[i].{values,formula}`` anchor is wrong.

    The atom's path is only enforced when it targets a ``versions[i].values`` or
    ``versions[i].formula`` location and that version index resolves against the
    rule. Any other path shape (e.g. a bare field, a table-cell suffix, or a
    non-``versions`` anchor) is left to the other checks so this never
    over-rejects. A ``.values`` anchor requires the version to carry a ``values``
    table; a ``.formula`` anchor requires the version to carry a ``formula``.
    """
    normalized = re.sub(r"\s+", "", str(path))
    normalized = _ATOM_PATH_INDEX_PATTERN.sub(r"[\1]", normalized)
    normalized = re.sub(r"^versions\.", "versions[0].", normalized)
    match = _ANCHOR_FIELD_PATTERN.match(normalized)
    if match is None:
        return None

    version_index = int(match.group(1))
    field = match.group(2)
    versions = rule.get("versions")
    if not isinstance(versions, list) or version_index >= len(versions):
        return (
            f"Proof atom anchor invalid: {label} anchors `versions[{version_index}]."
            f"{field}`, but the rule has no such version."
        )
    version = versions[version_index]
    if not isinstance(version, dict):
        return (
            f"Proof atom anchor invalid: {label} anchors `versions[{version_index}]."
            f"{field}`, but that version is malformed."
        )

    if field == "values":
        if not isinstance(version.get("values"), dict):
            return (
                f"Proof atom anchor mismatch: {label} anchors `versions[{version_index}]"
                ".values`, but that version declares no `values` table. Anchor a "
                "scalar rule at `versions[i].formula` instead."
            )
    elif version.get("formula") is None:
        return (
            f"Proof atom anchor mismatch: {label} anchors `versions[{version_index}]"
            ".formula`, but that version declares no `formula`. Anchor a parameter "
            "table at `versions[i].values` instead."
        )
    return None


def _validate_source_proof_atom(
    *,
    source: Any,
    label: str,
    kind: str,
    rule_source: Any,
    source_texts: Mapping[str, str | None] | None,
) -> list[str]:
    if not isinstance(source, dict):
        return [f"Proof source malformed: {label} `source` must be a mapping."]

    issues: list[str] = []
    citation_path = str(source.get("corpus_citation_path") or "").strip()
    if not citation_path:
        issues.append(
            "Proof source missing corpus path: "
            f"{label} `source.corpus_citation_path` is required."
        )

    if citation_path and source_texts is not None:
        resolved_text = source_texts.get(citation_path)
        if resolved_text is None:
            issues.append(
                "Proof source unresolved: "
                f"{label} cites `{citation_path}`, which was not found in "
                "corpus.provisions."
            )
        else:
            for field in ("excerpt", "quote"):
                evidence_text = source.get(field)
                if not isinstance(evidence_text, str) or not evidence_text.strip():
                    continue
                evidence_text = evidence_text.strip()
                if not _source_contains_proof_evidence(
                    source_text=resolved_text,
                    evidence_text=evidence_text,
                ):
                    issues.append(
                        "Proof source evidence not found: "
                        f"{label} `source.{field}` does not appear in "
                        f"`{citation_path}`."
                    )
                else:
                    issues.extend(
                        _proof_excerpt_subsection_scope_issues(
                            evidence_text=evidence_text,
                            rule_source=rule_source,
                            label=label,
                            field=field,
                        )
                    )

    table = source.get("table")
    if kind == "table_cell":
        if not isinstance(table, dict):
            issues.append(
                "Proof table provenance missing: "
                f"{label} is `table_cell` and must declare `source.table`."
            )
        else:
            missing = [
                field
                for field in ("header", "row", "column")
                if not str(table.get(field) or "").strip()
            ]
            if missing:
                issues.append(
                    "Proof table cell provenance incomplete: "
                    f"{label} `source.table` must declare "
                    + ", ".join(f"`{field}`" for field in missing)
                    + "."
                )
    elif kind == "parameter_table" and isinstance(table, dict):
        if not str(table.get("header") or "").strip():
            issues.append(
                "Proof table provenance incomplete: "
                f"{label} `source.table.header` is required for parameter tables."
            )
        has_row_key = bool(str(table.get("row_key") or table.get("row") or "").strip())
        has_column_key = bool(
            str(table.get("column_key") or table.get("column") or "").strip()
        )
        if not has_row_key or not has_column_key:
            issues.append(
                "Proof table provenance incomplete: "
                f"{label} parameter table proof should declare row and column keys."
            )

    return issues


def _source_contains_proof_evidence(
    *,
    source_text: str,
    evidence_text: str,
) -> bool:
    normalized_evidence = re.sub(r"\s+", " ", evidence_text).strip()
    if not normalized_evidence:
        return False
    for segment in split_proof_evidence_text(source_text):
        if _bounded_source_evidence_match(evidence_text, segment):
            return True
        normalized_segment = re.sub(r"\s+", " ", segment).strip()
        if _bounded_source_evidence_match(normalized_evidence, normalized_segment):
            return True
    return False


def _bounded_source_evidence_match(evidence_text: str, source_text: str) -> bool:
    if not evidence_text:
        return False
    return any(
        _source_evidence_span_is_bounded(
            evidence_text=evidence_text,
            source_text=source_text,
            start=match.start(),
            end=match.end(),
        )
        for match in re.finditer(re.escape(evidence_text), source_text)
    )


def _source_evidence_span_is_bounded(
    *,
    evidence_text: str,
    source_text: str,
    start: int,
    end: int,
) -> bool:
    """Reject source spans that omit part of a word or numeric token."""

    before = source_text[:start]
    after = source_text[end:]
    if (
        before
        and (before[-1].isalnum() or before[-1] == "_")
        and not (
            _span_starts_after_fused_footnote_marker(
                evidence_text=evidence_text,
                before=before,
            )
        )
    ):
        return False
    if after and (after[0].isalnum() or after[0] == "_"):
        return False

    evidence_begins_numeric = _evidence_begins_with_numeric_token(evidence_text)
    evidence_ends_numeric = _evidence_ends_with_numeric_token(evidence_text)

    if evidence_begins_numeric:
        if _span_starts_inside_space_grouped_number(
            evidence_text=evidence_text,
            before=before,
        ):
            return False
        if _left_context_omits_numeric_sign(before):
            return False
        if _left_context_continues_numeric_token(before):
            return False
        if _span_omits_accounting_parentheses(
            evidence_text=evidence_text,
            before=before,
            after=after,
        ):
            return False

    if evidence_ends_numeric:
        if _span_ends_inside_space_grouped_number(after=after):
            return False
        right = after.lstrip()
        while right and unicodedata.category(right[0]) == "Cf":
            right = right[1:]
        if right and _is_numeric_suffix_marker(right[0]):
            return False
        if _right_context_starts_textual_numeric_suffix(after):
            return False
        if _right_context_continues_numeric_token(after):
            return False
    if _span_omits_trailing_accounting_sign(
        evidence_text=evidence_text,
        before=before,
        after=after,
    ):
        return False
    return True


def _evidence_begins_with_numeric_token(evidence_text: str) -> bool:
    text = evidence_text.lstrip()
    while text and _is_currency_marker(text[0]):
        text = text[1:].lstrip()
    return bool(text and text[0].isdecimal())


def _evidence_ends_with_numeric_token(evidence_text: str) -> bool:
    text = evidence_text.rstrip()
    return bool(text and text[-1].isdecimal())


_NUMERIC_SIGN_MARKERS = frozenset(
    {
        "+",
        "-",
        "\N{PLUS-MINUS SIGN}",
        "\N{FIGURE DASH}",
        "\N{MINUS SIGN}",
        "\N{MINUS-OR-PLUS SIGN}",
        "\N{SMALL PLUS SIGN}",
        "\N{SMALL HYPHEN-MINUS}",
        "\N{FULLWIDTH PLUS SIGN}",
        "\N{FULLWIDTH HYPHEN-MINUS}",
        "\N{SUPERSCRIPT PLUS SIGN}",
        "\N{SUPERSCRIPT MINUS}",
        "\N{SUBSCRIPT PLUS SIGN}",
        "\N{SUBSCRIPT MINUS}",
    }
)
_SPACED_NUMERIC_CONNECTOR_NAME_WORDS = frozenset(
    {"ASTERISK", "CARET", "DIVISION", "MINUS", "MULTIPLICATION", "RATIO"}
)
_SPACED_NUMERIC_CONNECTOR_NAME_PHRASES = (
    "FRACTION SLASH",
    "SOLIDUS",
)


def _span_starts_inside_space_grouped_number(
    *,
    evidence_text: str,
    before: str,
) -> bool:
    text = evidence_text.lstrip()
    leading_digits = re.match(r"\d+", text)
    return bool(
        leading_digits
        and len(leading_digits.group(0)) == 3
        and len(before) >= 2
        and before[-1].isspace()
        and before[-2].isdecimal()
    )


def _span_ends_inside_space_grouped_number(*, after: str) -> bool:
    return bool(
        len(after) >= 4
        and after[0].isspace()
        and all(character.isdecimal() for character in after[1:4])
        and (len(after) == 4 or not after[4].isdecimal())
    )


def _left_context_omits_numeric_sign(before: str) -> bool:
    left = before.rstrip()
    while left:
        if unicodedata.category(left[-1]) == "Cf" or _is_currency_marker(left[-1]):
            left = left[:-1].rstrip()
            continue
        break
    return bool(left and _is_numeric_sign_marker(left[-1]))


def _left_context_continues_numeric_token(before: str) -> bool:
    separated_from_evidence = bool(before and before[-1].isspace())
    left = before.rstrip()
    connectors = ""
    while left and _is_connector_character(left[-1]):
        connectors = left[-1] + connectors
        left = left[:-1]
    connector_preceded_by_space = bool(left and left[-1].isspace())
    left = left.rstrip()
    if not connectors or not left or not left[-1].isdecimal():
        return False
    return bool(
        (not separated_from_evidence and not connector_preceded_by_space)
        or any(_is_spaced_numeric_connector(character) for character in connectors)
        or (
            connector_preceded_by_space
            and any(
                unicodedata.normalize("NFKC", character) == ":"
                for character in connectors
            )
        )
    )


def _right_context_continues_numeric_token(after: str) -> bool:
    separated_from_evidence = bool(after and after[0].isspace())
    right = after.lstrip()
    connectors = ""
    while right and _is_connector_character(right[0]):
        connectors += right[0]
        right = right[1:]
    connector_followed_by_space = bool(right and right[0].isspace())
    right = right.lstrip()
    if not connectors or not right or not right[0].isdecimal():
        return False
    return bool(
        (not separated_from_evidence and not connector_followed_by_space)
        or any(_is_spaced_numeric_connector(character) for character in connectors)
        or (
            connector_followed_by_space
            and any(
                unicodedata.normalize("NFKC", character) == ":"
                for character in connectors
            )
        )
    )


def _span_omits_accounting_parentheses(
    *,
    evidence_text: str,
    before: str,
    after: str,
) -> bool:
    left = before.rstrip()
    while left and _is_currency_marker(left[-1]):
        left = left[:-1].rstrip()
    carried_closing = evidence_text.rstrip().endswith(")")
    if carried_closing:
        return left.endswith("(")
    if _is_numeric_date_label(evidence_text) or _is_substantive_numeric_prose(
        evidence_text
    ):
        return False
    omitted_closing = after.lstrip().startswith(")")
    return bool(left.endswith("(") and omitted_closing)


def _is_numeric_date_label(text: str) -> bool:
    """Recognize a complete date-only legal effective-period label."""

    date = r"\d{1,4}/\d{1,2}/\d{1,4}"
    return bool(
        re.fullmatch(
            rf"\s*{date}(?:\s*(?:-|[\u2010-\u2015]|to)\s*{date})?\s*",
            text,
            flags=re.IGNORECASE,
        )
    )


def _span_starts_after_fused_footnote_marker(
    *,
    evidence_text: str,
    before: str,
) -> bool:
    """Recognize OCR text that fuses a one-digit footnote to sentence prose."""

    text = evidence_text.lstrip()
    lead = re.match(r"[^\W\d_]+", text)
    words = re.findall(r"[^\W\d_]+", text)
    left_context = before[:-1].rstrip()
    follows_document_boundary = bool(
        re.search(r"\$\d[\d,]*(?:\.\d+)?$", left_context)
        or left_context.endswith((".", "!", "?"))
    )
    return bool(
        lead
        and lead.group(0) == "The"
        and len(words) >= 4
        and len(before) >= 2
        and before[-1] in "123456789"
        and before[-2].isspace()
        and left_context
        and follows_document_boundary
    )


def _is_substantive_numeric_prose(text: str) -> bool:
    """Distinguish a parenthesized legal clause from accounting notation."""

    normalized = re.sub(r"\s+", " ", text).strip()
    return bool(
        re.fullmatch(
            r"\d+(?:\.\d+)?%\s+of\s+\w+\s+meals\s+(?:a|per)\s+"
            r"(?:day|week|month|year)",
            normalized,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            r"\d+\s+(?:days?|weeks?|months?|years?)\s+or\s+the\s+period\s+"
            r"for\s+which\b.+",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _span_omits_trailing_accounting_sign(
    *,
    evidence_text: str,
    before: str,
    after: str,
) -> bool:
    del evidence_text, before
    if not after or after[0].isspace():
        return False
    right = after
    while right and unicodedata.category(right[0]) == "Cf":
        right = right[1:]
    return bool(right and _is_numeric_sign_marker(right[0]))


def _is_currency_marker(character: str) -> bool:
    return unicodedata.category(character) == "Sc"


def _is_connector_character(character: str) -> bool:
    return bool(
        not character.isspace() and not character.isalnum() and character != "_"
    )


def _is_spaced_numeric_connector(character: str) -> bool:
    normalized = unicodedata.normalize("NFKC", character)
    if (
        normalized in "/*^+-"
        or character == "\N{EN DASH}"
        or _is_numeric_sign_marker(character)
    ):
        return True
    name = unicodedata.name(character, "")
    name_words = frozenset(name.split())
    return bool(
        name_words & _SPACED_NUMERIC_CONNECTOR_NAME_WORDS
        or any(phrase in name for phrase in _SPACED_NUMERIC_CONNECTOR_NAME_PHRASES)
    )


def _is_numeric_suffix_marker(character: str) -> bool:
    if _is_currency_marker(character):
        return True
    name = unicodedata.name(character, "")
    return any(
        phrase in name
        for phrase in ("PERCENT SIGN", "PER MILLE SIGN", "PER TEN THOUSAND SIGN")
    )


def _is_numeric_sign_marker(character: str) -> bool:
    if character in _NUMERIC_SIGN_MARKERS:
        return True
    normalized = unicodedata.normalize("NFKC", character)
    if normalized in "+-":
        return True
    name = unicodedata.name(character, "")
    return any(word in name.split() for word in ("HYPHEN", "MINUS", "PLUS"))


def _right_context_starts_textual_numeric_suffix(after: str) -> bool:
    normalized = "".join(
        " "
        if (
            character.isspace()
            or unicodedata.category(character) == "Cf"
            or _is_hyphen_marker(character)
        )
        else character
        for character in after
    )
    return bool(
        re.match(
            r"\s*(?:"
            r"per\s+cent|percent(?:age)?s?|"
            r"per\s+mille|per\s+ten\s+thousand|"
            r"basis\s+points?|bps"
            r")\b",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _is_hyphen_marker(character: str) -> bool:
    normalized = unicodedata.normalize("NFKC", character)
    return bool(
        normalized == "-" or "HYPHEN" in unicodedata.name(character, "").split()
    )


def _proof_excerpt_subsection_scope_issues(
    *,
    evidence_text: str,
    rule_source: Any,
    label: str,
    field: str,
) -> list[str]:
    """Reject an explicit excerpt marker outside the rule citation."""
    if not isinstance(rule_source, str):
        return []
    scope = _rule_source_subsection_scope(rule_source)
    declared = {
        match.group("marker")
        for match in re.finditer(r"\((?P<marker>[a-z])\)", rule_source)
    }
    excerpt_marker = re.match(r"^\s*\((?P<marker>[a-z])\)(?:\s|$)", evidence_text)
    marker = excerpt_marker.group("marker") if excerpt_marker is not None else None
    if declared and marker is not None and marker not in declared:
        return [
            "Proof source evidence not found: "
            f"{label} `source.{field}` appears outside the rule's declared "
            f"subsection scope `{rule_source}` (excerpt begins at `({marker})`)."
        ]

    numeric_marker = re.match(r"^\s*\((?P<marker>\d+)\)(?:\s|$)", evidence_text)
    if numeric_marker is None or not scope:
        return []
    # A broad top-level citation, such as ``(f)``, admits all of its children.
    # Only reject a numeric sibling when every declared top-level subsection is
    # itself narrowed to explicit numeric children.
    if any(children is None for children in scope.values()):
        return []
    numeric = numeric_marker.group("marker")
    declared_numeric = {
        child for children in scope.values() for child in (children or ())
    }
    if numeric in declared_numeric:
        return []
    return [
        "Proof source evidence not found: "
        f"{label} `source.{field}` appears outside the rule's declared "
        f"subsection scope `{rule_source}` (excerpt begins at `({numeric})`)."
    ]


def _rule_source_subsection_scope(
    rule_source: str,
) -> dict[str, frozenset[str] | None]:
    """Parse top-level subsection markers and explicit numeric child ranges."""
    scope: dict[str, set[str] | None] = {}
    current_top: str | None = None
    for segment in str(rule_source).split(","):
        top_match = re.search(r"\((?P<top>[a-z])\)", segment)
        if top_match is not None:
            current_top = top_match.group("top")
            suffix = segment[top_match.end() :]
        elif current_top is not None:
            if (
                re.fullmatch(
                    r"\s*\(\d+\)(?:\s*-\s*\(\d+\))?\s*",
                    segment,
                )
                is None
            ):
                current_top = None
                continue
            suffix = segment
        else:
            continue
        top = current_top
        assert top is not None
        numeric = {
            match.group("value") for match in re.finditer(r"\((?P<value>\d+)\)", suffix)
        }
        for match in re.finditer(r"\((?P<start>\d+)\)\s*-\s*\((?P<end>\d+)\)", suffix):
            start = int(match.group("start"))
            end = int(match.group("end"))
            if start <= end and end - start <= 100:
                numeric.update(str(value) for value in range(start, end + 1))
        if not numeric:
            scope[top] = None
        elif scope.get(top, set()) is not None:
            current = scope.setdefault(top, set())
            assert isinstance(current, set)
            current.update(numeric)
    return {
        top: None if children is None else frozenset(children)
        for top, children in scope.items()
    }


def _validate_import_proof_atom(raw_import: Any, label: str) -> list[str]:
    if not isinstance(raw_import, dict):
        return [f"Proof import malformed: {label} `import` must be a mapping."]

    missing = [
        field
        for field in ("target", "output", "hash")
        if not str(raw_import.get(field) or "").strip()
    ]
    if missing:
        return [
            "Proof import contract incomplete: "
            f"{label} `import` must declare "
            + ", ".join(f"`{field}`" for field in missing)
            + "."
        ]

    hash_value = str(raw_import.get("hash") or "")
    if not hash_value.startswith("sha256:"):
        return [
            "Proof import hash invalid: "
            f"{label} `import.hash` must start with `sha256:`."
        ]
    return []


def source_proof_paths(content: str) -> list[str]:
    """Return explicit proof atom paths, useful for debugging and tests."""
    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if not isinstance(payload, dict):
            return []
        rules = payload.get("rules")
        if not isinstance(rules, list):
            return []
        paths: list[str] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            proof = _rule_proof(rule)
            if not isinstance(proof, dict):
                continue
            atoms = proof.get("atoms")
            if not isinstance(atoms, list):
                continue
            for atom in atoms:
                if isinstance(atom, dict) and str(atom.get("path") or "").strip():
                    paths.append(str(atom["path"]))
        return paths
    return []


# ---------------------------------------------------------------------------
# Money proof obligations
#
# Strict proof validation (``validate_rulespec_proofs``) is per-file opt-in via
# ``module.proof_validation.required: true``. That leaves a gap: a repository
# with zero opted-in files can ship monetary parameters carrying no proof atom
# at all. ``find_missing_money_proof_atoms`` closes the money-scoped part of
# that gap by deriving obligations directly from the compiled RuleSpec: every
# policy-bearing monetary value must have a proof atom at a matching path whose
# source cites a provision (direct corpus source or an explicit import).
# ---------------------------------------------------------------------------

# Currency units that mark a value as monetary. Matched case-insensitively.
MONEY_UNITS = frozenset({"eur", "usd", "gbp", "cad", "aud", "nzd", "chf", "jpy"})
# dtypes that mark a rule as monetary regardless of unit.
MONEY_DTYPES = frozenset({"money", "currency"})


@dataclass(frozen=True)
class MoneyProofObligation:
    """A single derived money proof obligation for one RuleSpec location."""

    rule_name: str
    path: str
    kind: str
    reason: str
    satisfied: bool


@dataclass
class MoneyAtomReport:
    """Per-module result of money proof-obligation derivation."""

    obligations: list[MoneyProofObligation] = field(default_factory=list)

    @property
    def missing(self) -> list[MoneyProofObligation]:
        return [
            obligation for obligation in self.obligations if not obligation.satisfied
        ]

    @property
    def missing_count(self) -> int:
        return len(self.missing)

    @property
    def obligation_count(self) -> int:
        return len(self.obligations)


def find_missing_money_proof_atoms(content: str) -> MoneyAtomReport:
    """Derive money proof obligations for a RuleSpec YAML document.

    An obligation is *satisfied* when the rule declares a proof atom whose
    ``path`` matches the money-bearing location (``versions[i].formula`` or
    ``versions[i].values``) and whose evidence cites an immutable provision (a
    ``source`` with a ``corpus_citation_path``) or an explicit ``import``.

    Non-monetary values, and monetary formulas whose only numeric literals are
    structural sentinels (``{-1, 0, 1, 2, 3}`` and half-up ``0.5``), create no
    obligation. This mirrors the encoder's own numeric-grounding exclusions so
    the money-atom surface never diverges from what grounding treats as a
    policy number.
    """
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return MoneyAtomReport()

    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return MoneyAtomReport()

    rules = payload.get("rules")
    if not isinstance(rules, list):
        return MoneyAtomReport()

    selector_table_keys = _selector_table_keys(rules)

    report = MoneyAtomReport()
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        if not _is_policy_bearing_rule(rule):
            continue
        if not _rule_is_monetary(rule):
            continue

        rule_name = str(rule.get("name") or f"rules[{index}]").strip() or (
            f"rules[{index}]"
        )
        satisfied_paths = _cited_proof_atom_paths(rule)
        selector_keys = (
            selector_table_keys.get(rule_name)
            if _rule_is_structural_selector(rule)
            else None
        )

        versions = rule.get("versions")
        if not isinstance(versions, list):
            continue
        for version_index, version in enumerate(versions):
            if not isinstance(version, dict):
                continue
            for path, kind, reason in _money_locations_for_version(
                version,
                version_index=version_index,
                selector_keys=selector_keys,
            ):
                report.obligations.append(
                    MoneyProofObligation(
                        rule_name=rule_name,
                        path=path,
                        kind=kind,
                        reason=reason,
                        satisfied=_normalize_atom_path(path) in satisfied_paths,
                    )
                )
    return report


def _rule_is_monetary(rule: dict[str, Any]) -> bool:
    dtype = str(rule.get("dtype") or "").strip().lower()
    if dtype in MONEY_DTYPES:
        return True
    unit = str(rule.get("unit") or "").strip().lower()
    return unit in MONEY_UNITS


def _rule_is_structural_selector(rule: dict[str, Any]) -> bool:
    """Reuse the encoder's structural-selector test for formula extraction."""
    from .validator_pipeline import _is_structural_selector_rule

    return bool(_is_structural_selector_rule(rule))


def _selector_table_keys(rules: list[Any]) -> dict[str, set[str]]:
    """Reuse the encoder's index-selector key map for structural exclusions."""
    from .validator_pipeline import _rulespec_index_selector_keys

    return _rulespec_index_selector_keys(rules)


def _money_locations_for_version(
    version: dict[str, Any],
    *,
    version_index: int,
    selector_keys: set[str] | None,
) -> list[tuple[str, str, str]]:
    """Return (path, atom_kind, reason) for each money-bearing spot in a version.

    ``atom_kind`` is the proof-atom ``kind`` the location is expected to use so
    error messages can point the encoder at the right shape.
    """
    from .validator_pipeline import (
        GROUNDING_ALLOWED_VALUES,
        _extract_formula_grounding_values,
        _numeric_rule_value,
    )

    locations: list[tuple[str, str, str]] = []

    formula = version.get("formula")
    formula_has_money_literal = False
    if isinstance(formula, (int, float)) and not isinstance(formula, bool):
        formula_has_money_literal = float(formula) not in GROUNDING_ALLOWED_VALUES
    elif isinstance(formula, str):
        formula_has_money_literal = bool(
            _extract_formula_grounding_values(
                1,
                formula,
                structural_selector_keys=selector_keys,
            )
        )
    if formula_has_money_literal:
        locations.append(
            (
                f"versions[{version_index}].formula",
                "parameter",
                "monetary formula carries a policy numeric literal",
            )
        )

    table_values = version.get("values")
    if isinstance(table_values, dict):
        for cell_value in table_values.values():
            extracted = _numeric_rule_value(cell_value)
            if extracted is None:
                continue
            _, value = extracted
            if value in GROUNDING_ALLOWED_VALUES:
                continue
            locations.append(
                (
                    f"versions[{version_index}].values",
                    "parameter_table",
                    "monetary parameter table carries policy cell values",
                )
            )
            break

    return locations


def _cited_proof_atom_paths(rule: dict[str, Any]) -> set[str]:
    """Return normalized proof-atom paths that cite a provision.

    A proof atom counts only when it names a path AND carries at least one of
    ``source`` (with a ``corpus_citation_path``), ``claim``, or ``import``. An
    atom with a path but no evidence does not satisfy a money obligation.
    """
    proof = _rule_proof(rule)
    if not isinstance(proof, dict):
        return set()
    atoms = proof.get("atoms")
    if not isinstance(atoms, list):
        return set()

    paths: set[str] = set()
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        path = str(atom.get("path") or "").strip()
        if not path:
            continue
        if _atom_cites_provision(atom):
            normalized = _normalize_atom_path(path)
            paths.add(normalized)
            if normalized == "versions" and atom.get("kind") == "parameter_table":
                paths.update(_version_table_atom_paths(rule))
    return paths


def _version_table_atom_paths(rule: dict[str, Any]) -> set[str]:
    """Return table-value paths covered by a broad ``path: versions`` atom."""
    versions = rule.get("versions")
    if not isinstance(versions, list):
        return set()
    return {
        f"versions[{index}].values"
        for index, version in enumerate(versions)
        if isinstance(version, dict) and isinstance(version.get("values"), dict)
    }


def _atom_cites_provision(atom: dict[str, Any]) -> bool:
    source = atom.get("source")
    if (
        isinstance(source, dict)
        and str(source.get("corpus_citation_path") or "").strip()
    ):
        return True
    raw_import = atom.get("import")
    if isinstance(raw_import, dict) and str(raw_import.get("target") or "").strip():
        return True
    return False


_ATOM_PATH_INDEX_PATTERN = re.compile(r"\[\s*(\d+)\s*\]")


def _normalize_atom_path(path: str) -> str:
    """Normalize a proof-atom path for matching.

    A ``versions[0].values`` atom covers the whole table, so cell suffixes such
    as ``versions[0].values.household_size_1`` are folded to the table path. A
    bare ``versions.formula`` (no index) is treated as ``versions[0].formula``
    so authors are not forced to index a single-version rule.
    """
    normalized = re.sub(r"\s+", "", str(path)).strip().rstrip(".")
    normalized = _ATOM_PATH_INDEX_PATTERN.sub(r"[\1]", normalized)
    # Fold a bare `versions.<field>` to `versions[0].<field>`.
    normalized = re.sub(r"^versions\.", "versions[0].", normalized)
    # Fold table-cell suffixes onto the table path.
    match = re.match(r"^(versions\[\d+\]\.values)\b", normalized)
    if match:
        return match.group(1)
    match = re.match(r"^(versions\[\d+\]\.formula)\b", normalized)
    if match:
        return match.group(1)
    return normalized


# ---------------------------------------------------------------------------
# Ratchet file support
#
# A repository burns its money-atom debt down over time. The ratchet file
# records the allowance so CI can fail on any *new* atom-less monetary value
# while tolerating the known backlog. Two shapes are accepted:
#
#   total_allowed: 217          # a single repo-wide budget, OR
#   paths:                      # a per-file budget (path relative to repo root)
#     be/statutes/x.yaml: 12
#     be/statutes/y.yaml: 3
#
# When both are present, per-path budgets take precedence for listed files and
# ``total_allowed`` covers everything else. An absent ratchet file means a
# strict zero allowance.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoneyAtomRatchet:
    """Parsed money-atom ratchet allowances."""

    total_allowed: int | None
    per_path: dict[str, int]

    @classmethod
    def empty(cls) -> MoneyAtomRatchet:
        return cls(total_allowed=None, per_path={})


def load_money_atom_ratchet(text: str) -> MoneyAtomRatchet:
    """Parse a money-atom ratchet YAML document.

    Raises ``ValueError`` for a malformed ratchet so CI fails loudly rather
    than silently treating a typo as a zero allowance.
    """
    payload = yaml.safe_load(text)
    if payload is None:
        return MoneyAtomRatchet.empty()
    if not isinstance(payload, dict):
        raise ValueError("money-atom ratchet must be a mapping")

    total_allowed: int | None = None
    if "total_allowed" in payload:
        raw_total = payload.get("total_allowed")
        if isinstance(raw_total, bool) or not isinstance(raw_total, int):
            raise ValueError("`total_allowed` must be an integer")
        if raw_total < 0:
            raise ValueError("`total_allowed` must not be negative")
        total_allowed = raw_total

    per_path: dict[str, int] = {}
    raw_paths = payload.get("paths")
    if raw_paths is not None:
        if not isinstance(raw_paths, dict):
            raise ValueError("`paths` must be a mapping of file path to integer")
        for key, value in raw_paths.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"`paths.{key}` must be an integer")
            if value < 0:
                raise ValueError(f"`paths.{key}` must not be negative")
            per_path[str(key)] = value

    return MoneyAtomRatchet(total_allowed=total_allowed, per_path=per_path)


@dataclass
class MoneyAtomFileResult:
    """Money-atom result for a single file, keyed by its display path."""

    path: str
    report: MoneyAtomReport

    @property
    def missing_count(self) -> int:
        return self.report.missing_count

    @property
    def obligation_count(self) -> int:
        return self.report.obligation_count


@dataclass
class MoneyAtomRun:
    """Aggregate money-atom result across a set of files, with ratchet verdict."""

    files: list[MoneyAtomFileResult]
    ratchet: MoneyAtomRatchet
    over_budget_paths: dict[str, tuple[int, int]] = field(default_factory=dict)
    total_over_budget: tuple[int, int] | None = None

    @property
    def total_missing(self) -> int:
        return sum(file.missing_count for file in self.files)

    @property
    def total_obligations(self) -> int:
        return sum(file.obligation_count for file in self.files)

    @property
    def passed(self) -> bool:
        return not self.over_budget_paths and self.total_over_budget is None


def evaluate_money_atoms(
    files: list[tuple[str, str]],
    ratchet: MoneyAtomRatchet | None = None,
) -> MoneyAtomRun:
    """Run the money-atom check across ``(display_path, content)`` pairs.

    The ratchet verdict works as follows. Any file with an explicit per-path
    budget is checked against that budget individually. All remaining files'
    missing counts are summed and checked against ``total_allowed`` (0 when the
    ratchet omits it). This keeps a single repo-wide budget simple while still
    letting a repo pin per-file allowances where that is clearer.
    """
    ratchet = ratchet or MoneyAtomRatchet.empty()

    file_results = [
        MoneyAtomFileResult(
            path=display_path, report=find_missing_money_proof_atoms(content)
        )
        for display_path, content in files
    ]

    over_budget_paths: dict[str, tuple[int, int]] = {}
    untracked_missing = 0
    for result in file_results:
        if result.path in ratchet.per_path:
            allowed = ratchet.per_path[result.path]
            if result.missing_count > allowed:
                over_budget_paths[result.path] = (result.missing_count, allowed)
        else:
            untracked_missing += result.missing_count

    total_over_budget: tuple[int, int] | None = None
    total_budget = ratchet.total_allowed if ratchet.total_allowed is not None else 0
    if untracked_missing > total_budget:
        total_over_budget = (untracked_missing, total_budget)

    return MoneyAtomRun(
        files=file_results,
        ratchet=ratchet,
        over_budget_paths=over_budget_paths,
        total_over_budget=total_over_budget,
    )


def emit_money_atom_ratchet(files: list[tuple[str, str]]) -> str:
    """Return a seed ratchet YAML capturing the current missing-atom backlog.

    The seed uses a single ``total_allowed`` equal to the current repo-wide
    missing count, plus a commented per-path breakdown so the backlog is
    visible and can be burned down file by file.
    """
    file_results = [
        (display_path, find_missing_money_proof_atoms(content).missing_count)
        for display_path, content in files
    ]
    per_path = {display_path: count for display_path, count in file_results if count}
    total = sum(per_path.values())

    lines = [
        "# Money-atom proof-obligation ratchet.",
        "#",
        "# Each count is the number of monetary values (currency parameters,",
        "# currency parameter-table cells, and currency literals in derived",
        "# formulas) that still lack a proof atom citing a provision.",
        "# `axiom-encode proof-validate --require-money-atoms` fails when the",
        "# untracked missing count exceeds `total_allowed`. Burn this down; do",
        "# not raise it. Regenerate with `--emit-ratchet`.",
        f"total_allowed: {total}",
    ]
    if per_path:
        lines.append("# Current per-file backlog (informational):")
        for display_path in sorted(per_path):
            lines.append(f"#   {display_path}: {per_path[display_path]}")
    return "\n".join(lines) + "\n"
