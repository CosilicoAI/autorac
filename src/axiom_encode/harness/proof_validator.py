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
    PROOF_EVIDENCE_SEGMENT_SEPARATOR,
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
                            source_text=resolved_text,
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


_ALPHA_SUBSECTION_RANGE_RE = re.compile(
    r"\((?P<start>[a-z])\)\s*(?P<separator>[^\w\s])\s*"
    r"\((?P<end>[a-z])\)"
)
_NUMERIC_PARENT_ALPHA_RANGE_RE = re.compile(
    r"\((?P<parent>\d+)\)\s*\((?P<start>[a-z])\)\s*"
    r"(?P<separator>[^\w\s])\s*\((?P<end>[a-z])\)"
)
_STRUCTURAL_COORDINATE_TOKEN = r"(?:\d+|[a-z]|[IVXLCDM]|[ivxlcdmIVXLCDM]{2,})"
_NUMERIC_PARENT_ALPHA_SINGLETON_RE = re.compile(
    r"\((?P<parent>\d+)\)\s*\((?P<child>[a-z])\)"
    rf"(?!\s*\({_STRUCTURAL_COORDINATE_TOKEN}\))"
)
_DEEPER_ROMAN_COORDINATE_RE = re.compile(
    r"(?P<outer_chain>(?:\(\d+\)\s*)*)"
    r"\((?P<parent>[a-z])\)(?P<numeric_chain>(?:\s*\(\d+\))*)\s*"
    r"\((?P<start>[ivxlcdmIVXLCDM]+)\)"
    r"(?:\s*(?P<separator>[^\w\s])\s*"
    r"\((?P<end>[ivxlcdmIVXLCDM]+)\))?"
)
_SUBSECTION_RANGE_SEPARATORS = frozenset(
    {
        "-",  # HYPHEN-MINUS
        "\u2010",  # HYPHEN
        "\u2011",  # NON-BREAKING HYPHEN
        "\u2012",  # FIGURE DASH
        "\u2013",  # EN DASH
        "\u2014",  # EM DASH
        "\u2015",  # HORIZONTAL BAR
        "\u2212",  # MINUS SIGN
        "\uff0d",  # FULLWIDTH HYPHEN-MINUS
    }
)
_UNRESOLVED_NUMERIC_ALPHA_PARENT = "<unresolved>"


def _is_subsection_range_separator(character: str) -> bool:
    """Recognize legal hyphen/dash glyphs without admitting other punctuation."""
    return character in _SUBSECTION_RANGE_SEPARATORS


def _canonical_roman_value(token: str) -> int | None:
    """Return a bounded canonical Roman value, independent of source case."""
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    upper = token.upper()
    total = 0
    previous = 0
    for character in reversed(upper):
        value = values[character]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    if not 0 < total <= 3999:
        return None

    remainder = total
    rendered = []
    for value, numeral in (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ):
        count, remainder = divmod(remainder, value)
        rendered.append(numeral * count)
    return total if "".join(rendered) == upper else None


def _canonical_roman_marker(value: int) -> str:
    """Render a validated Roman coordinate in its normalized lowercase form."""
    remainder = value
    rendered = []
    for number, numeral in (
        (1000, "m"),
        (900, "cm"),
        (500, "d"),
        (400, "cd"),
        (100, "c"),
        (90, "xc"),
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    ):
        count, remainder = divmod(remainder, number)
        rendered.append(numeral * count)
    return "".join(rendered)


def _deeper_roman_match_markers(match: re.Match[str]) -> frozenset[str]:
    """Expand one canonical singleton/range, failing closed on malformed ranges."""
    start = _canonical_roman_value(match.group("start"))
    if start is None:
        return frozenset()
    end_token = match.group("end")
    if end_token is None:
        return frozenset({_canonical_roman_marker(start)})
    separator = match.group("separator")
    end = _canonical_roman_value(end_token)
    if (
        separator is None
        or not _is_subsection_range_separator(separator)
        or end is None
        or start > end
        or end - start > 100
    ):
        return frozenset()
    return frozenset(_canonical_roman_marker(value) for value in range(start, end + 1))


def _rule_source_deeper_roman_scope(
    rule_source: str,
) -> dict[tuple[str, ...], frozenset[str] | None]:
    """Parse exact Roman children with their complete alpha/numeric ownership."""
    scope: dict[tuple[str, ...], set[str] | None] = {}
    for match in _DEEPER_ROMAN_COORDINATE_RE.finditer(rule_source):
        chain = (
            *re.findall(r"\((\d+)\)", match.group("outer_chain")),
            match.group("parent"),
            *re.findall(r"\((\d+)\)", match.group("numeric_chain")),
        )
        markers = _deeper_roman_match_markers(match)
        if not markers:
            scope[chain] = None
        elif scope.get(chain, set()) is not None:
            current = scope.setdefault(chain, set())
            assert isinstance(current, set)
            current.update(markers)
    return {
        chain: None if markers is None else frozenset(markers)
        for chain, markers in scope.items()
    }


def _roman_chain_alpha_parent(chain: tuple[str, ...]) -> str:
    """Return the single alphabetic owner retained in a Roman coordinate chain."""
    return next(coordinate for coordinate in chain if not coordinate.isdigit())


def _rule_source_broad_structural_chains(
    rule_source: str,
) -> frozenset[tuple[str, ...]]:
    """Retain broad numeric/alpha ancestry at arbitrary structural depth."""
    chains: set[tuple[str, ...]] = set()
    coordinate_pattern = re.compile(
        r"\((?P<coordinate>\d+|[a-z]|[ivxlcdmIVXLCDM]{2,}|[IVXLCDM])\)"
    )
    groups: list[tuple[int, int, tuple[str, ...]]] = []
    current: list[re.Match[str]] = []
    for match in coordinate_pattern.finditer(rule_source):
        if current and rule_source[current[-1].end() : match.start()].strip():
            groups.append(
                (
                    current[0].start(),
                    current[-1].end(),
                    tuple(item.group("coordinate") for item in current),
                )
            )
            current = []
        current.append(match)
    if current:
        groups.append(
            (
                current[0].start(),
                current[-1].end(),
                tuple(item.group("coordinate") for item in current),
            )
        )

    previous_end: int | None = None
    previous_chain: tuple[str, ...] | None = None
    connected_range_active = False
    connected_range_chains: set[tuple[str, ...]] = set()
    for start, end, coordinates in groups:
        gap = "" if previous_end is None else rule_source[previous_end:start].strip()
        if len(gap) == 1 and (_is_subsection_range_separator(gap) or gap == "/"):
            if connected_range_active:
                chains.difference_update(connected_range_chains)
                connected_range_chains.clear()
                previous_chain = None
                previous_end = end
                continue
            suffix = coordinates[0] if len(coordinates) == 1 else None
            range_values: list[str] = []
            if (
                previous_chain is not None
                and _is_subsection_range_separator(gap)
                and suffix is not None
            ):
                if previous_chain[-1].isdigit() and suffix.isdigit():
                    if len(previous_chain[-1]) <= 10 and len(suffix) <= 10:
                        range_start = int(previous_chain[-1])
                        range_end = int(suffix)
                        if range_start <= range_end and range_end - range_start <= 100:
                            range_values = [
                                str(value)
                                for value in range(range_start, range_end + 1)
                            ]
                elif (
                    re.fullmatch(r"[a-z]", previous_chain[-1]) is not None
                    and re.fullmatch(r"[a-z]", suffix) is not None
                ):
                    range_start = ord(previous_chain[-1])
                    range_end = ord(suffix)
                    if range_start <= range_end and range_end - range_start <= 25:
                        range_values = [
                            chr(value) for value in range(range_start, range_end + 1)
                        ]
            if previous_chain is not None:
                chains.discard(previous_chain)
            if range_values:
                prefix = previous_chain[:-1]
                connected_range_chains = {(*prefix, value) for value in range_values}
                chains.update(connected_range_chains)
                previous_chain = (*prefix, range_values[-1])
            else:
                previous_chain = None
            connected_range_active = True
            previous_end = end
            continue

        connected_range_active = False
        connected_range_chains.clear()

        if gap == "," and len(coordinates) == 1:
            if (
                previous_chain is not None
                and previous_chain[-1].isdigit() == coordinates[0].isdigit()
            ):
                shorthand = (*previous_chain[:-1], coordinates[0])
                chains.add(shorthand)
                previous_chain = shorthand
            else:
                previous_chain = None
            previous_end = end
            continue

        alpha_coordinates = [
            coordinate for coordinate in coordinates if not coordinate.isdigit()
        ]
        chain = (
            coordinates
            if len(alpha_coordinates) == 1
            and re.fullmatch(r"[a-z]", alpha_coordinates[0]) is not None
            else None
        )
        if chain is not None:
            chains.add(chain)
        previous_chain = chain
        previous_end = end
    return frozenset(chains)


def _expanded_alpha_subsection_range_markers(value: str) -> frozenset[str]:
    """Expand bounded ascending single-letter subsection ranges."""
    expanded: set[str] = set()
    for match in _ALPHA_SUBSECTION_RANGE_RE.finditer(value):
        if not _is_subsection_range_separator(match.group("separator")):
            continue
        start = ord(match.group("start"))
        end = ord(match.group("end"))
        if start <= end and end - start <= 25:
            expanded.update(chr(codepoint) for codepoint in range(start, end + 1))
    return frozenset(expanded)


def _numeric_parent_alpha_range_match_markers(
    segment: str,
    match: re.Match[str],
) -> frozenset[str]:
    """Return a direct alphabetic child range or fail closed for nested/Roman use."""
    prefix = segment[: match.start()]
    if re.search(rf"\({_STRUCTURAL_COORDINATE_TOKEN}\)\s*$", prefix):
        return frozenset()
    return _expanded_alpha_subsection_range_markers(match.group(0))


def _masked_numeric_parent_alpha_segment(
    segment: str,
) -> tuple[str, dict[str, set[str]], str | None]:
    """Mask handled direct numeric-parent alpha coordinates in one segment."""
    masked = segment
    scope: dict[str, set[str]] = {}
    rightmost_parent: tuple[int, str] | None = None
    for match in reversed(list(_NUMERIC_PARENT_ALPHA_RANGE_RE.finditer(masked))):
        markers = _numeric_parent_alpha_range_match_markers(masked, match)
        if not markers:
            continue
        scope.setdefault(match.group("parent"), set()).update(markers)
        if rightmost_parent is None or match.end() > rightmost_parent[0]:
            rightmost_parent = (match.end(), match.group("parent"))
        masked = (
            masked[: match.start()]
            + " " * (match.end() - match.start())
            + masked[match.end() :]
        )
    for match in reversed(list(_NUMERIC_PARENT_ALPHA_SINGLETON_RE.finditer(masked))):
        prefix = masked[: match.start()]
        if re.search(rf"\({_STRUCTURAL_COORDINATE_TOKEN}\)\s*$", prefix):
            continue
        scope.setdefault(match.group("parent"), set()).add(match.group("child"))
        if rightmost_parent is None or match.end() > rightmost_parent[0]:
            rightmost_parent = (match.end(), match.group("parent"))
        masked = (
            masked[: match.start()]
            + " " * (match.end() - match.start())
            + masked[match.end() :]
        )
    if rightmost_parent is not None and re.search(
        rf"\({_STRUCTURAL_COORDINATE_TOKEN}\)",
        masked[rightmost_parent[0] :],
    ):
        rightmost_parent = None
    return masked, scope, None if rightmost_parent is None else rightmost_parent[1]


def _masked_deeper_roman_segment(segment: str) -> tuple[str, frozenset[str]]:
    """Mask unsupported deeper Roman coordinates while preserving other scope."""
    masked = segment
    parents: set[str] = set()
    for match in reversed(list(_DEEPER_ROMAN_COORDINATE_RE.finditer(segment))):
        parents.add(match.group("parent"))
        masked = (
            masked[: match.start()]
            + " " * (match.end() - match.start())
            + masked[match.end() :]
        )
    return masked, frozenset(parents)


def _masked_numeric_parent_alpha_segments(
    rule_source: str,
) -> list[tuple[str, dict[str, set[str]]]]:
    """Mask direct coordinates and carry exact alpha comma shorthand ownership."""
    results: list[tuple[str, dict[str, set[str]]]] = []
    current_parent: str | None = None
    carry_blocked = False
    for raw_segment in str(rule_source).split(","):
        masked, segment_scope, rightmost_parent = _masked_numeric_parent_alpha_segment(
            raw_segment
        )
        if segment_scope:
            current_parent = rightmost_parent
            carry_blocked = rightmost_parent is None
        elif shorthand := re.fullmatch(
            r"\s*\((?P<child>[a-z])\)\s*",
            raw_segment,
        ):
            if current_parent is not None:
                segment_scope = {current_parent: {shorthand.group("child")}}
                masked = " " * len(raw_segment)
            elif carry_blocked:
                segment_scope = {
                    _UNRESOLVED_NUMERIC_ALPHA_PARENT: {shorthand.group("child")}
                }
                masked = " " * len(raw_segment)
        else:
            current_parent = None
            carry_blocked = False
        results.append((masked, segment_scope))
    return results


def _rule_source_numeric_parent_alpha_scope(
    rule_source: str,
) -> dict[str, frozenset[str]]:
    """Parse direct numeric-parent alphabetic coordinates without flattening."""
    raw_scope: dict[str, set[str]] = {}
    for _, segment_scope in _masked_numeric_parent_alpha_segments(rule_source):
        for parent, markers in segment_scope.items():
            raw_scope.setdefault(parent, set()).update(markers)
    valid_pairs = {
        chain
        for chain in _rule_source_broad_structural_chains(rule_source)
        if len(chain) == 2 and chain[0].isdigit() and not chain[1].isdigit()
    }
    scope: dict[str, set[str]] = {}
    for parent, markers in raw_scope.items():
        if parent == _UNRESOLVED_NUMERIC_ALPHA_PARENT:
            scope.setdefault(parent, set()).update(markers)
            continue
        for marker in markers:
            if (parent, marker) in valid_pairs:
                scope.setdefault(parent, set()).add(marker)
            else:
                scope.setdefault(_UNRESOLVED_NUMERIC_ALPHA_PARENT, set()).add(marker)
    return {parent: frozenset(markers) for parent, markers in scope.items()}


def _proof_excerpt_subsection_scope_issues(
    *,
    source_text: str,
    evidence_text: str,
    rule_source: Any,
    label: str,
    field: str,
) -> list[str]:
    """Reject an explicit excerpt marker outside the rule citation."""
    if not isinstance(rule_source, str):
        return []
    scope = _rule_source_subsection_scope(rule_source)
    numeric_parent_alpha_scope = _rule_source_numeric_parent_alpha_scope(rule_source)
    declared = {
        match.group("marker")
        for match in re.finditer(r"\((?P<marker>[a-z])\)", rule_source)
    }
    excerpt_marker = re.match(r"^\s*\((?P<marker>[a-z])\)(?:\s|$)", evidence_text)
    marker = excerpt_marker.group("marker") if excerpt_marker is not None else None
    parent_scoped_markers = {
        child for children in numeric_parent_alpha_scope.values() for child in children
    }
    unresolved_parent_scoped_markers = numeric_parent_alpha_scope.get(
        _UNRESOLVED_NUMERIC_ALPHA_PARENT,
        frozenset(),
    )
    if marker is not None and marker in unresolved_parent_scoped_markers:
        resolved_numeric_parent_alpha_scope = {
            parent: children
            for parent, children in numeric_parent_alpha_scope.items()
            if parent != _UNRESOLVED_NUMERIC_ALPHA_PARENT
        }
        if marker in {
            child
            for children in resolved_numeric_parent_alpha_scope.values()
            for child in children
        } and _is_alpha_marker_in_declared_numeric_parent_scope(
            source_text=source_text,
            evidence_text=evidence_text,
            marker=marker,
            scope=resolved_numeric_parent_alpha_scope,
        ):
            return []
        authoritative_broad_chains = _rule_source_broad_structural_chains(rule_source)
        if (
            marker,
        ) in authoritative_broad_chains and _source_evidence_is_direct_alpha_header(
            source_text=source_text,
            evidence_text=evidence_text,
        ):
            return []
        return [
            "Proof source evidence not found: "
            f"{label} `source.{field}` appears outside the rule's declared "
            f"subsection scope `{rule_source}` (excerpt begins at `({marker})`)."
        ]
    if marker is not None and marker in parent_scoped_markers and marker not in scope:
        if _is_alpha_marker_in_declared_numeric_parent_scope(
            source_text=source_text,
            evidence_text=evidence_text,
            marker=marker,
            scope=numeric_parent_alpha_scope,
        ):
            return []
        return [
            "Proof source evidence not found: "
            f"{label} `source.{field}` appears outside the rule's declared "
            f"subsection scope `{rule_source}` (excerpt begins at `({marker})`)."
        ]
    deeper_roman_scope = _rule_source_deeper_roman_scope(rule_source)
    deeper_roman_parents = frozenset(
        _roman_chain_alpha_parent(chain) for chain in deeper_roman_scope
    )
    independently_broad = (
        marker is not None and marker in scope and scope[marker] is None
    )
    roman_evidence_marker = re.match(
        r"^\s*\([a-z]\)\s*\((?P<marker>[ivxlcdmIVXLCDM]+)\)(?:\s|$)",
        evidence_text,
    ) or re.match(
        r"^\s*\((?P<marker>[ivxlcdmIVXLCDM]+)\)(?:\s|$)",
        evidence_text,
    )
    source_chain = _source_roman_evidence_chain(
        source_text=source_text,
        evidence_text=evidence_text,
    )
    verified_broad_alpha = independently_broad and (
        _source_text_is_exact_evidence_scalar(source_text, evidence_text)
        or _source_evidence_is_direct_alpha_header(
            source_text=source_text,
            evidence_text=evidence_text,
        )
    )
    roman_marker_is_authorized = False
    if (
        deeper_roman_parents
        and roman_evidence_marker is not None
        and not verified_broad_alpha
    ):
        roman_marker = roman_evidence_marker.group("marker")
        normalized_roman_marker = _canonical_roman_value(roman_marker)
        broad_roman_chains = set(_rule_source_broad_structural_chains(rule_source))
        broad_roman_parents = {
            _roman_chain_alpha_parent(chain) for chain in broad_roman_chains
        }
        matching_broad_chains = (
            []
            if source_chain is None
            else [
                chain
                for chain in broad_roman_chains
                if source_chain[: len(chain)] == chain
            ]
        )
        broad_roman_parent = (
            source_chain is not None
            and len(matching_broad_chains) == 1
            and normalized_roman_marker is not None
            and _source_roman_coordinate_occurrence_count(
                source_text=source_text,
                chain=source_chain,
                marker=_canonical_roman_marker(normalized_roman_marker),
            )
            == 1
        ) or (
            source_chain is None
            and _source_text_is_exact_evidence_scalar(source_text, evidence_text)
            and len(broad_roman_chains) == 1
            and len(deeper_roman_parents) == 1
            and deeper_roman_parents <= broad_roman_parents
        )
        roman_marker_is_owned = (
            normalized_roman_marker is not None
            and _is_roman_marker_in_declared_scope(
                source_text=source_text,
                evidence_text=evidence_text,
                marker=_canonical_roman_marker(normalized_roman_marker),
                scope=deeper_roman_scope,
            )
        )
        if not broad_roman_parent and not roman_marker_is_owned:
            return [
                "Proof source evidence not found: "
                f"{label} `source.{field}` appears outside the rule's declared "
                f"subsection scope `{rule_source}` "
                f"(excerpt begins at `({roman_marker})`)."
            ]
        roman_marker_is_authorized = True
    if (
        marker is not None
        and marker in deeper_roman_parents
        and not independently_broad
        and not roman_marker_is_authorized
    ):
        return [
            "Proof source evidence not found: "
            f"{label} `source.{field}` appears outside the rule's declared "
            f"subsection scope `{rule_source}` (excerpt begins at `({marker})`)."
        ]
    if declared and marker is not None and marker not in declared:
        has_explicit_deeper_alpha_coordinate = (
            re.search(r"\([a-z]\)\s*\(\d+\)\s*\([a-z]\)", rule_source) is not None
        )
        if not has_explicit_deeper_alpha_coordinate and (
            _is_nested_alpha_marker_in_declared_numeric_scope(
                source_text=source_text,
                evidence_text=evidence_text,
                marker=marker,
                scope=scope,
            )
        ):
            return []
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


def _is_roman_marker_in_declared_scope(
    *,
    source_text: str,
    evidence_text: str,
    marker: str,
    scope: Mapping[tuple[str, ...], frozenset[str] | None],
) -> bool:
    """Require a Roman excerpt to belong to its cited alpha/numeric chain."""
    declared_chains = {
        chain
        for chain, children in scope.items()
        if children is not None and marker in children
    }
    if not declared_chains:
        return False

    if _source_text_is_exact_evidence_scalar(source_text, evidence_text):
        # Some corpus units are already the exact cited scalar, with no parent
        # headings available to resolve. Marker membership is the tightest
        # ownership proof possible for that established representation.
        return len(declared_chains) == 1

    source_chain = _source_roman_evidence_chain(
        source_text=source_text,
        evidence_text=evidence_text,
    )
    return source_chain in declared_chains and (
        _source_roman_coordinate_occurrence_count(
            source_text=source_text,
            chain=source_chain,
            marker=marker,
        )
        == 1
    )


def _source_roman_coordinate_occurrence_count(
    *,
    source_text: str,
    chain: tuple[str, ...],
    marker: str,
) -> int:
    """Count a complete Roman coordinate without carrying ancestry across records."""
    count = 0
    header_pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)"
        r"\((?P<marker>\d+|[ivxlcdmIVXLCDM]{2,}|[IVXLCDM]|[a-z])\)"
        r"[ \t]+"
    )
    for record in source_text.split(PROOF_EVIDENCE_SEGMENT_SEPARATOR):
        stack: list[tuple[int, str]] = []
        for header in header_pattern.finditer(record):
            indent = len(header.group("indent").expandtabs(2))
            while stack and indent <= stack[-1][0]:
                stack.pop()
            raw_marker = header.group("marker")
            roman_value = (
                _canonical_roman_value(raw_marker)
                if re.fullmatch(r"[ivxlcdmIVXLCDM]+", raw_marker) is not None
                else None
            )
            if (
                tuple(value for _, value in stack) == chain
                and roman_value is not None
                and _canonical_roman_marker(roman_value) == marker
            ):
                count += 1
            stack.append((indent, raw_marker))
    return count + sum(
        1
        for flattened_chain, flattened_marker, _ in (
            _source_flattened_roman_coordinate_occurrences(source_text)
        )
        if flattened_chain == chain and flattened_marker == marker
    )


def _source_flattened_roman_coordinate_occurrences(
    source_text: str,
) -> list[tuple[tuple[str, ...], str, int]]:
    """Recover unambiguous Roman ownership from flat leading coordinates.

    Some released statute text puts compound parents such as ``(1) (a)`` and
    ``(b) (i)`` at the record's baseline indentation, followed by Roman peers
    such as ``(ii)`` at that same baseline. The ordinary indentation stack must
    remain authoritative for structured text, so this fallback considers only
    baseline, line-leading coordinate bundles. It resets at every evidence
    record and numeric peer, and carries a Roman continuation only in canonical
    ascending order from an established alpha owner.
    """
    header_pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)(?P<bundle>"
        rf"(?:\({_STRUCTURAL_COORDINATE_TOKEN}\)[ \t]*)+"
        r")"
    )
    coordinate_pattern = re.compile(rf"\((?P<value>{_STRUCTURAL_COORDINATE_TOKEN})\)")
    occurrences: list[tuple[tuple[str, ...], str, int]] = []
    record_offset = 0
    for record in source_text.split(PROOF_EVIDENCE_SEGMENT_SEPARATOR):
        headers = list(header_pattern.finditer(record))
        baseline_indent = min(
            (len(header.group("indent").expandtabs(2)) for header in headers),
            default=None,
        )
        numeric_prefix: tuple[str, ...] = ()
        current_chain: tuple[str, ...] | None = None
        next_roman_value = 1
        roman_sequence_established = False
        for header_index, header in enumerate(headers):
            indent = len(header.group("indent").expandtabs(2))
            if baseline_indent is None or indent != baseline_indent:
                continue
            tokens = [
                match.group("value")
                for match in coordinate_pattern.finditer(header.group("bundle"))
            ]
            if not tokens:
                continue
            header_start = record_offset + header.start("bundle")
            if len(tokens) == 1:
                token = tokens[0]
                if token.isdigit():
                    numeric_prefix = (token,)
                    current_chain = None
                    next_roman_value = 1
                    roman_sequence_established = False
                    continue
                roman_value = (
                    _canonical_roman_value(token)
                    if re.fullmatch(r"[ivxlcdmIVXLCDM]+", token) is not None
                    else None
                )
                ambiguous_lower_alpha = len(token) == 1 and token.islower()
                has_roman_successor = False
                has_direct_alpha_progression = False
                if ambiguous_lower_alpha and roman_value is not None:
                    for successor in headers[header_index + 1 :]:
                        successor_indent = len(successor.group("indent").expandtabs(2))
                        if successor_indent != baseline_indent:
                            continue
                        successor_tokens = [
                            match.group("value")
                            for match in coordinate_pattern.finditer(
                                successor.group("bundle")
                            )
                        ]
                        successor_roman_value = (
                            _canonical_roman_value(successor_tokens[0])
                            if len(successor_tokens) == 1
                            and re.fullmatch(r"[ivxlcdmIVXLCDM]+", successor_tokens[0])
                            is not None
                            else None
                        )
                        has_roman_successor = successor_roman_value == roman_value + 1
                        owner_alpha_index = next(
                            (
                                index
                                for index, coordinate in enumerate(current_chain or ())
                                if len(coordinate) == 1 and coordinate.islower()
                            ),
                            None,
                        )
                        owner_alpha = (
                            current_chain[owner_alpha_index]
                            if current_chain is not None
                            and owner_alpha_index is not None
                            else None
                        )
                        successor_alpha_index = next(
                            (
                                index
                                for index, coordinate in enumerate(successor_tokens)
                                if not coordinate.isdigit()
                            ),
                            None,
                        )
                        successor_alpha = (
                            successor_tokens[successor_alpha_index]
                            if successor_alpha_index is not None
                            else None
                        )
                        successor_owner_prefix = (
                            tuple(successor_tokens[:successor_alpha_index])
                            if successor_alpha_index is not None
                            else ()
                        )
                        successor_suffix = (
                            successor_tokens[successor_alpha_index + 1 :]
                            if successor_alpha_index is not None
                            else []
                        )
                        successor_roman_suffix = (
                            successor_suffix[-1]
                            if successor_suffix
                            and re.fullmatch(r"[ivxlcdmIVXLCDM]+", successor_suffix[-1])
                            is not None
                            else None
                        )
                        successor_numeric_suffix = (
                            successor_suffix[:-1]
                            if successor_roman_suffix is not None
                            else successor_suffix
                        )
                        has_direct_alpha_progression = (
                            owner_alpha is not None
                            and successor_alpha is not None
                            and len(successor_alpha) == 1
                            and successor_alpha.islower()
                            and (
                                not successor_owner_prefix
                                or successor_owner_prefix
                                == current_chain[:owner_alpha_index]
                            )
                            and all(
                                coordinate.isdigit()
                                for coordinate in successor_numeric_suffix
                            )
                            and ord(token) == ord(owner_alpha) + 1
                            and ord(successor_alpha) == ord(token) + 1
                        )
                        break
                if (
                    current_chain is not None
                    and roman_value == next_roman_value
                    and not has_direct_alpha_progression
                    and (
                        not ambiguous_lower_alpha
                        or roman_sequence_established
                        or has_roman_successor
                    )
                ):
                    marker = _canonical_roman_marker(roman_value)
                    occurrences.append((current_chain, marker, header_start))
                    next_roman_value = roman_value + 1
                    roman_sequence_established = True
                    continue
                if len(token) == 1 and token.islower() and numeric_prefix:
                    current_chain = (*numeric_prefix, token)
                    next_roman_value = 1
                    roman_sequence_established = False
                else:
                    current_chain = None
                    next_roman_value = 1
                    roman_sequence_established = False
                continue

            alpha_index = next(
                (index for index, token in enumerate(tokens) if not token.isdigit()),
                None,
            )
            if alpha_index is None:
                numeric_prefix = tuple(tokens)
                current_chain = None
                next_roman_value = 1
                roman_sequence_established = False
                continue
            alpha = tokens[alpha_index]
            explicit_numeric_prefix = tuple(tokens[:alpha_index])
            owner_prefix = explicit_numeric_prefix or numeric_prefix
            suffix = tokens[alpha_index + 1 :]
            roman_token = (
                suffix[-1]
                if suffix
                and re.fullmatch(r"[ivxlcdmIVXLCDM]+", suffix[-1]) is not None
                and all(token.isdigit() for token in suffix[:-1])
                else None
            )
            numeric_suffix = suffix[:-1] if roman_token is not None else suffix
            if (
                not owner_prefix
                or len(alpha) != 1
                or not alpha.islower()
                or not all(token.isdigit() for token in numeric_suffix)
            ):
                current_chain = None
                next_roman_value = 1
                roman_sequence_established = False
                continue
            numeric_prefix = owner_prefix
            current_chain = (*owner_prefix, alpha, *numeric_suffix)
            next_roman_value = 1
            roman_sequence_established = False
            if roman_token is None:
                continue
            roman_value = _canonical_roman_value(roman_token)
            if roman_value is None:
                current_chain = None
                roman_sequence_established = False
                continue
            marker = _canonical_roman_marker(roman_value)
            occurrences.append((current_chain, marker, header_start))
            next_roman_value = roman_value + 1
            roman_sequence_established = True
        record_offset += len(record) + len(PROOF_EVIDENCE_SEGMENT_SEPARATOR)
    occurrences.extend(
        _source_flattened_alpha_numeric_roman_coordinate_occurrences(source_text)
    )
    return list(dict.fromkeys(occurrences))


def _source_flattened_alpha_numeric_roman_coordinate_occurrences(
    source_text: str,
) -> list[tuple[tuple[str, ...], str, int]]:
    """Recover flat ``(a) -> (1) -> (i)`` federal-style ownership.

    Flat federal regulation text commonly places every coordinate at the same
    indentation.  This pass recognizes only alphabetic parents established in
    ascending order and Roman children established in canonical ascending
    order.  Those constraints keep an isolated Roman-looking letter or a
    numeric sibling from being assigned to the claimed parent.
    """
    flat_coordinate_token = rf"(?:{_STRUCTURAL_COORDINATE_TOKEN}|[A-Z])"
    header_pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)(?P<bundle>"
        rf"(?:\({flat_coordinate_token}\)[ \t]*)+"
        r")"
    )
    coordinate_pattern = re.compile(rf"\((?P<value>{flat_coordinate_token})\)")
    occurrences: list[tuple[tuple[str, ...], str, int]] = []
    record_offset = 0
    for record in source_text.split(PROOF_EVIDENCE_SEGMENT_SEPARATOR):
        headers = list(header_pattern.finditer(record))
        baseline_indent = min(
            (len(header.group("indent").expandtabs(2)) for header in headers),
            default=None,
        )
        baseline_headers = [
            header
            for header in headers
            if baseline_indent is not None
            and len(header.group("indent").expandtabs(2)) == baseline_indent
        ]
        alpha_parent: str | None = None
        next_alpha = "a"
        current_chain: tuple[str, str] | None = None
        next_roman_value = 1
        roman_sequence_established = False
        alpha_sequence_started = False
        leading_numeric_owner_seen = False
        suspended_chain: tuple[str, str] | None = None
        suspended_next_roman_value = 1
        suspended_chain_tainted = False
        for header_index, header in enumerate(baseline_headers):
            tokens = [
                match.group("value")
                for match in coordinate_pattern.finditer(header.group("bundle"))
            ]
            if len(tokens) != 1:
                if suspended_chain is not None:
                    suspended_chain_tainted = True
                alpha_parent = None
                current_chain = None
                roman_sequence_established = False
                continue
            token = tokens[0]
            successor_tokens = (
                [
                    match.group("value")
                    for match in coordinate_pattern.finditer(
                        baseline_headers[header_index + 1].group("bundle")
                    )
                ]
                if header_index + 1 < len(baseline_headers)
                else []
            )
            roman_value = (
                _canonical_roman_value(token)
                if re.fullmatch(r"[ivxlcdmIVXLCDM]+", token) is not None
                else None
            )
            successor_roman_value = (
                _canonical_roman_value(successor_tokens[0])
                if len(successor_tokens) == 1
                and re.fullmatch(r"[ivxlcdmIVXLCDM]+", successor_tokens[0]) is not None
                else None
            )
            successor_is_numeric = (
                len(successor_tokens) == 1 and successor_tokens[0].isdigit()
            )
            successor_is_expected_alpha = (
                len(successor_tokens) == 1
                and len(successor_tokens[0]) == 1
                and successor_tokens[0] == next_alpha
            )
            ambiguous_single_letter = len(token) == 1 and token.islower()
            successor_is_next_alpha_after_token = (
                len(successor_tokens) == 1
                and len(successor_tokens[0]) == 1
                and ambiguous_single_letter
                and successor_tokens[0] == chr(ord(token) + 1)
            )
            successor_is_upper_alpha = (
                len(successor_tokens) == 1
                and len(successor_tokens[0]) == 1
                and successor_tokens[0].isupper()
            )
            uppercase_roman_transition = (
                token.isupper()
                and current_chain is not None
                and roman_value == next_roman_value
                and (
                    len(token) > 1
                    or successor_roman_value == roman_value + 1
                    or successor_is_numeric
                    or successor_is_expected_alpha
                    or successor_is_upper_alpha
                    or not successor_tokens
                )
            )
            if token.isupper() and not uppercase_roman_transition:
                if current_chain is not None and suspended_chain is None:
                    suspended_chain = current_chain
                    suspended_next_roman_value = next_roman_value
                    suspended_chain_tainted = False
                current_chain = None
                roman_sequence_established = False
                continue
            if suspended_chain is not None and token.isdigit():
                suspended_chain_tainted = True
                current_chain = None
                roman_sequence_established = False
                continue
            if suspended_chain is not None and roman_value is not None:
                if (
                    not suspended_chain_tainted
                    and roman_value == suspended_next_roman_value
                ):
                    current_chain = suspended_chain
                    next_roman_value = suspended_next_roman_value
                suspended_chain = None
                suspended_chain_tainted = False
            alpha_transition = (
                ambiguous_single_letter
                and token == next_alpha
                and not (leading_numeric_owner_seen and not alpha_sequence_started)
                and (
                    successor_is_numeric
                    or current_chain is None
                    or roman_value != next_roman_value
                    or successor_is_next_alpha_after_token
                )
            )
            roman_transition = (
                current_chain is not None
                and roman_value == next_roman_value
                and not alpha_transition
                and (
                    token != next_alpha
                    or successor_roman_value == roman_value + 1
                    or successor_is_numeric
                    or successor_is_upper_alpha
                )
                and (
                    not ambiguous_single_letter
                    or roman_sequence_established
                    or successor_roman_value == roman_value + 1
                    or successor_is_numeric
                    or successor_is_expected_alpha
                    or successor_is_upper_alpha
                    or not successor_tokens
                )
            )
            if alpha_transition:
                alpha_parent = token
                suspended_chain = None
                suspended_chain_tainted = False
                alpha_sequence_started = True
                next_alpha = chr(ord(token) + 1)
                current_chain = None
                next_roman_value = 1
                roman_sequence_established = False
                continue
            if token.isdigit():
                if alpha_parent is None and not alpha_sequence_started:
                    leading_numeric_owner_seen = True
                current_chain = (
                    (alpha_parent, token) if alpha_parent is not None else None
                )
                next_roman_value = 1
                roman_sequence_established = False
                continue
            if roman_transition:
                occurrences.append(
                    (
                        current_chain,
                        _canonical_roman_marker(roman_value),
                        record_offset + header.start("bundle"),
                    )
                )
                next_roman_value = roman_value + 1
                roman_sequence_established = True
                continue
            alpha_parent = None
            current_chain = None
            suspended_chain = None
            suspended_chain_tainted = False
            roman_sequence_established = False
            if ambiguous_single_letter:
                next_alpha = ""
        record_offset += len(record) + len(PROOF_EVIDENCE_SEGMENT_SEPARATOR)
    return occurrences


def _source_text_is_exact_evidence_scalar(
    source_text: str,
    evidence_text: str,
) -> bool:
    """Recognize an established corpus unit containing only the proof scalar."""
    return source_text.strip() == evidence_text.strip()


def _source_roman_evidence_chain(
    *,
    source_text: str,
    evidence_text: str,
) -> tuple[str, ...] | None:
    """Resolve one Roman excerpt's structural owner within one evidence record."""
    words = re.split(r"\s+", evidence_text.strip())
    evidence_pattern = r"\s+".join(re.escape(word) for word in words)
    evidence_matches = [
        match
        for match in re.finditer(evidence_pattern, source_text)
        if _source_evidence_span_is_bounded(
            evidence_text=evidence_text,
            source_text=source_text,
            start=match.start(),
            end=match.end(),
        )
    ]
    if len(evidence_matches) != 1:
        return None
    evidence_match = evidence_matches[0]
    record_start = source_text.rfind(
        PROOF_EVIDENCE_SEGMENT_SEPARATOR,
        0,
        evidence_match.start(),
    )
    record_start = (
        0
        if record_start < 0
        else (record_start + len(PROOF_EVIDENCE_SEGMENT_SEPARATOR))
    )

    structural_headers = list(
        re.compile(
            r"(?m)^(?P<indent>[ \t]*)"
            r"\((?P<marker>\d+|[ivxlcdmIVXLCDM]{2,}|[IVXLCDM]|[a-z])\)"
            r"[ \t]+",
        ).finditer(source_text, record_start, evidence_match.end())
    )
    stack: list[tuple[int, str]] = []
    for header in structural_headers:
        if header.start() > evidence_match.start():
            break
        indent = len(header.group("indent").expandtabs(2))
        while stack and indent <= stack[-1][0]:
            stack.pop()
        marker_start = header.start("marker") - 1
        if marker_start == evidence_match.start():
            ancestry = tuple(value for _, value in stack)
            alpha_indexes = [
                index
                for index, coordinate in enumerate(ancestry)
                if not coordinate.isdigit()
            ]
            if len(alpha_indexes) != 1:
                break
            alpha = ancestry[alpha_indexes[0]]
            if len(alpha) != 1 or not alpha.islower():
                break
            return ancestry
        stack.append((indent, header.group("marker")))
    flattened_chains = {
        chain
        for chain, _, header_start in _source_flattened_roman_coordinate_occurrences(
            source_text
        )
        if header_start == evidence_match.start()
    }
    if len(flattened_chains) == 1:
        return next(iter(flattened_chains))
    return None


def _source_evidence_is_direct_alpha_header(
    *,
    source_text: str,
    evidence_text: str,
) -> bool:
    """Prove a Roman-looking singleton is a direct alpha header, not nested."""
    words = re.split(r"\s+", evidence_text.strip())
    evidence_pattern = r"\s+".join(re.escape(word) for word in words)
    evidence_matches = [
        match
        for match in re.finditer(evidence_pattern, source_text)
        if _source_evidence_span_is_bounded(
            evidence_text=evidence_text,
            source_text=source_text,
            start=match.start(),
            end=match.end(),
        )
    ]
    if len(evidence_matches) != 1:
        return False
    evidence_match = evidence_matches[0]
    record_start = source_text.rfind(
        PROOF_EVIDENCE_SEGMENT_SEPARATOR,
        0,
        evidence_match.start(),
    )
    record_start = (
        0
        if record_start < 0
        else (record_start + len(PROOF_EVIDENCE_SEGMENT_SEPARATOR))
    )
    record_end = source_text.find(
        PROOF_EVIDENCE_SEGMENT_SEPARATOR,
        evidence_match.end(),
    )
    record_end = len(source_text) if record_end < 0 else record_end
    headers = list(
        re.compile(
            r"(?m)^(?P<indent>[ \t]*)"
            r"\((?P<marker>\d+|[ivxlcdmIVXLCDM]{2,}|[IVXLCDM]|[a-z])\)"
            r"[ \t]+"
        ).finditer(source_text, record_start, record_end)
    )
    evidence_header = next(
        (
            header
            for header in headers
            if header.start("marker") - 1 == evidence_match.start()
            and re.fullmatch(r"[a-z]", header.group("marker")) is not None
        ),
        None,
    )
    if evidence_header is None:
        return False
    evidence_indent = len(evidence_header.group("indent").expandtabs(2))
    direct_indent = min(
        (len(header.group("indent").expandtabs(2)) for header in headers),
        default=evidence_indent,
    )
    if evidence_indent != direct_indent:
        return False
    same_depth_same_marker = [
        header
        for header in headers
        if header.group("marker") == evidence_header.group("marker")
        and len(header.group("indent").expandtabs(2)) == evidence_indent
    ]
    if len(same_depth_same_marker) != 1:
        return False
    direct_alpha_headers = [
        header
        for header in headers
        if re.fullmatch(r"[a-z]", header.group("marker")) is not None
        and len(header.group("indent").expandtabs(2)) == evidence_indent
    ]
    if len(direct_alpha_headers) > 1:
        expected = direct_alpha_headers[0].group("marker")
        ordered_headers = []
        previous_header = None
        for header in direct_alpha_headers:
            if header.group("marker") != expected:
                if previous_header is None or not any(
                    previous_header.start() < nested.start() < header.start()
                    and len(nested.group("indent").expandtabs(2)) > evidence_indent
                    for nested in headers
                ):
                    break
                expected = header.group("marker")
            ordered_headers.append(header)
            expected = chr(ord(expected) + 1)
            previous_header = header
        if evidence_header not in ordered_headers:
            return False
    prior_same_depth_numerics = [
        header
        for header in headers
        if header.start() < evidence_header.start()
        and header.group("marker").isdigit()
        and len(header.group("indent").expandtabs(2)) == evidence_indent
    ]
    if not prior_same_depth_numerics:
        return True
    prior_numeric = prior_same_depth_numerics[-1]
    return any(
        prior_numeric.start() < header.start() < evidence_header.start()
        and len(header.group("indent").expandtabs(2)) > evidence_indent
        for header in headers
    )


def _is_alpha_marker_in_declared_numeric_parent_scope(
    *,
    source_text: str,
    evidence_text: str,
    marker: str,
    scope: Mapping[str, frozenset[str]],
) -> bool:
    """Match a ranged alpha child to its declared numeric parent in the source."""
    words = re.split(r"\s+", evidence_text.strip())
    evidence_pattern = r"\s+".join(re.escape(word) for word in words)
    scoped_blocks = [
        (start, end)
        for _, child, start, end in _source_numeric_parent_alpha_range_blocks(
            source_text,
            scope,
        )
        if child == marker
    ]
    for evidence_match in re.finditer(evidence_pattern, source_text):
        if not _source_evidence_span_is_bounded(
            evidence_text=evidence_text,
            source_text=source_text,
            start=evidence_match.start(),
            end=evidence_match.end(),
        ):
            continue
        if any(start <= evidence_match.start() < end for start, end in scoped_blocks):
            return True
    return False


def _source_numeric_parent_alpha_range_blocks(
    source_text: str,
    scope: Mapping[str, frozenset[str]],
) -> list[tuple[str, str, int, int]]:
    """Return ordered alpha-child blocks owned by cited numeric parents."""
    events = [
        *(
            (match.start(), "numeric", match)
            for match in re.finditer(
                r"(?m)^(?P<indent>[ \t]*)\((?P<marker>\d+)\)[ \t]+",
                source_text,
            )
        ),
        *(
            (match.start(), "alpha", match)
            for match in re.finditer(
                r"(?m)^(?P<indent>[ \t]*)\((?P<marker>[a-z])\)[ \t]+",
                source_text,
            )
        ),
        *(
            (match.start(), "boundary", match)
            for match in re.finditer(
                re.escape(PROOF_EVIDENCE_SEGMENT_SEPARATOR),
                source_text,
            )
        ),
    ]
    events.sort(key=lambda event: event[0])
    candidate_blocks: dict[str, list[list[tuple[str, str, int, int]]]] = {}
    parent_occurrence_counts: dict[str, int] = {}
    for event_index, (parent_start, kind, parent_match) in enumerate(events):
        if kind != "numeric":
            continue
        parent = parent_match.group("marker")
        allowed_children = scope.get(parent)
        if not allowed_children:
            continue
        parent_indent = len(parent_match.group("indent").expandtabs(2))
        record_start = source_text.rfind(
            PROOF_EVIDENCE_SEGMENT_SEPARATOR,
            0,
            parent_start,
        )
        record_start = (
            0
            if record_start < 0
            else (record_start + len(PROOF_EVIDENCE_SEGMENT_SEPARATOR))
        )
        record_end = source_text.find(
            PROOF_EVIDENCE_SEGMENT_SEPARATOR,
            parent_start,
        )
        record_end = len(source_text) if record_end < 0 else record_end
        record_structural_indents = [
            len(match.group("indent").expandtabs(2))
            for start, event_kind, match in events
            if event_kind in {"numeric", "alpha"} and record_start <= start < record_end
        ]
        if parent_indent != min(record_structural_indents, default=parent_indent):
            continue
        parent_occurrence_counts[parent] = parent_occurrence_counts.get(parent, 0) + 1
        prior_record_events = [
            (start, event_kind, match)
            for start, event_kind, match in events[:event_index]
            if record_start <= start < parent_start
            and event_kind in {"numeric", "alpha"}
        ]
        previous_peer_numeric_position = max(
            (
                position
                for position, (_, event_kind, match) in enumerate(prior_record_events)
                if event_kind == "numeric"
                and len(match.group("indent").expandtabs(2)) == parent_indent
            ),
            default=-1,
        )
        prior_alpha_after_peer = any(
            event_kind == "alpha"
            for _, event_kind, _ in prior_record_events[
                previous_peer_numeric_position + 1 :
            ]
        )
        if prior_alpha_after_peer:
            continue
        sequence: list[tuple[str, int]] = []
        expected = "a"
        inside_nested_numeric = False
        ambiguous_nested_alpha = False
        direct_alpha_indent: int | None = None
        boundary = len(source_text)
        for start, child_kind, child_match in events[event_index + 1 :]:
            if child_kind == "boundary":
                boundary = start
                break
            if child_kind == "numeric":
                numeric_indent = len(child_match.group("indent").expandtabs(2))
                if not sequence or numeric_indent <= parent_indent:
                    boundary = start
                    break
                inside_nested_numeric = True
                continue
            child = child_match.group("marker")
            child_indent = len(child_match.group("indent").expandtabs(2))
            if (
                not sequence
                and child in allowed_children
                and child_indent > parent_indent
            ):
                direct_alpha_indent = child_indent
                sequence.append((child, start))
                expected = chr(ord(child) + 1)
                inside_nested_numeric = False
                continue
            if (
                child in allowed_children
                and any(existing == child for existing, _ in sequence)
                and child_indent == direct_alpha_indent
            ):
                ambiguous_nested_alpha = True
                break
            if (
                inside_nested_numeric
                and child in "ivxlcdm"
                and direct_alpha_indent is not None
                and direct_alpha_indent > parent_indent
                and child_indent > direct_alpha_indent
            ):
                continue
            if child == expected:
                if direct_alpha_indent is None:
                    if child_indent < parent_indent:
                        ambiguous_nested_alpha = True
                        break
                    direct_alpha_indent = child_indent
                elif child_indent != direct_alpha_indent:
                    ambiguous_nested_alpha = True
                    break
            if child != expected:
                boundary = start
                break
            sequence.append((child, start))
            expected = chr(ord(expected) + 1)
            inside_nested_numeric = False
        if ambiguous_nested_alpha:
            continue
        candidate: list[tuple[str, str, int, int]] = []
        for child_index, (child, start) in enumerate(sequence):
            if child not in allowed_children:
                continue
            end = (
                sequence[child_index + 1][1]
                if child_index + 1 < len(sequence)
                else boundary
            )
            candidate.append((parent, child, start, end))
        if candidate:
            candidate_blocks.setdefault(parent, []).append(candidate)
    blocks: list[tuple[str, str, int, int]] = []
    for parent, candidates in candidate_blocks.items():
        if len(candidates) == 1 and parent_occurrence_counts.get(parent) == 1:
            blocks.extend(candidates[0])
    return blocks


def _is_nested_alpha_marker_in_declared_numeric_scope(
    *,
    source_text: str,
    evidence_text: str,
    marker: str,
    scope: Mapping[str, frozenset[str] | None],
) -> bool:
    """Recognize an alphabetic list nested inside a cited numeric paragraph."""
    if not any(children for children in scope.values()):
        return False

    words = re.split(r"\s+", evidence_text.strip())
    evidence_pattern = r"\s+".join(re.escape(word) for word in words)
    reference_context_cache: dict[int, bool] = {}
    for evidence_match in re.finditer(evidence_pattern, source_text):
        if not _source_evidence_span_is_bounded(
            evidence_text=evidence_text,
            source_text=source_text,
            start=evidence_match.start(),
            end=evidence_match.end(),
        ):
            continue
        prefix = source_text[: evidence_match.start()]
        numeric_matches = [
            match
            for match in re.finditer(r"\((?P<marker>\d+)\)", prefix)
            if _source_marker_has_numeric_paragraph_context(
                source_text,
                match.start(),
                reference_context_cache=reference_context_cache,
            )
        ]
        if not numeric_matches:
            continue
        numeric_parent = numeric_matches[-1]
        top_level_scope = _source_top_level_marker_for_numeric_parent(
            source_text, numeric_parent.start()
        )
        if top_level_scope is None:
            continue
        top_level, top_level_style = top_level_scope
        declared_children = scope.get(top_level)
        if (
            declared_children is None
            or numeric_parent.group("marker") not in declared_children
        ):
            continue

        nested_matches = list(
            re.finditer(
                r"\((?P<marker>[a-z])\)(?:\s|$)",
                source_text[numeric_parent.end() : evidence_match.start()],
            )
        )
        nested_markers = []
        for nested_match in nested_matches:
            nested_start = numeric_parent.end() + nested_match.start()
            if _source_marker_has_nested_list_context(
                source_text,
                start=nested_start,
                lower_bound=numeric_parent.end(),
                allow_period=(bool(nested_markers) and top_level_style == "dotted"),
            ):
                nested_markers.append(nested_match.group("marker"))
        if not _source_marker_has_nested_list_context(
            source_text,
            start=evidence_match.start(),
            lower_bound=numeric_parent.end(),
            allow_period=bool(nested_markers) and top_level_style == "dotted",
        ):
            continue
        nested_markers.append(marker)
        expected = [chr(ord("a") + index) for index in range(len(nested_markers))]
        if nested_markers == expected:
            return True
    return False


def _source_top_level_marker_for_numeric_parent(
    source_text: str, numeric_start: int
) -> tuple[str, str] | None:
    """Resolve alpha parents from one ordered structural-marker stream."""
    numeric_end = source_text.find(")", numeric_start)
    end_position = len(source_text) if numeric_end < 0 else numeric_end + 1
    events = [
        *(
            (match.start(), "dotted", match)
            for match in re.compile(r"(?<![A-Za-z])(?P<marker>[a-z])\.(?=\s)").finditer(
                source_text, 0, end_position
            )
        ),
        *(
            (match.start(), "parenthesized", match)
            for match in re.compile(r"\((?P<marker>[a-z])\)(?=\s)").finditer(
                source_text, 0, end_position
            )
        ),
        *(
            (match.start(), "numeric", match)
            for match in re.compile(r"\(\d+\)").finditer(source_text, 0, end_position)
        ),
    ]
    events.sort(key=lambda event: event[0])

    current_parent: tuple[str, str] | None = None
    numeric_parent_end: int | None = None
    numeric_parent_scope: tuple[str, str] | None = None
    nested_markers: list[str] = []
    reference_context_cache: dict[int, bool] = {}
    for start, kind, match in events:
        if start > numeric_start:
            break
        if kind == "numeric":
            if not _source_marker_has_numeric_paragraph_context(
                source_text,
                start,
                reference_context_cache=reference_context_cache,
            ):
                continue
            if start == numeric_start:
                return current_parent
            numeric_parent_end = match.end()
            numeric_parent_scope = current_parent
            nested_markers = []
            continue

        if kind == "dotted":
            if _source_marker_has_top_level_context(source_text, start):
                current_parent = (match.group("marker"), "dotted")
            continue

        is_subordinate = False
        if numeric_parent_end is not None and numeric_parent_scope is not None:
            _, numeric_parent_style = numeric_parent_scope
            if _source_marker_has_nested_list_context(
                source_text,
                start=start,
                lower_bound=numeric_parent_end,
                allow_period=(
                    bool(nested_markers) and numeric_parent_style == "dotted"
                ),
            ):
                nested_markers.append(match.group("marker"))
                expected = [
                    chr(ord("a") + index) for index in range(len(nested_markers))
                ]
                is_subordinate = nested_markers == expected
        if not is_subordinate and _source_marker_has_top_level_context(
            source_text, start
        ):
            current_parent = (match.group("marker"), "parenthesized")
    return None


def _source_marker_has_top_level_context(source_text: str, start: int) -> bool:
    line_start = source_text.rfind("\n", 0, start)
    index = start - 1
    while index >= 0 and source_text[index].isspace():
        index -= 1
    if index < 0 or index <= line_start:
        return True
    return source_text[index] in ".?!:"


def _source_marker_has_numeric_paragraph_context(
    source_text: str,
    start: int,
    *,
    reference_context_cache: dict[int, bool] | None = None,
) -> bool:
    if _source_position_follows_reference_word(
        source_text, start, cache=reference_context_cache
    ):
        return False
    index = start - 1
    while index >= 0 and source_text[index].isspace():
        index -= 1
    context_index = index
    line_start = source_text.rfind("\n", 0, start)
    if context_index < 0 or context_index <= line_start:
        return True
    follows_parenthesized_alpha = (
        source_text[context_index] == ")"
        and context_index >= 2
        and source_text[context_index - 2] == "("
        and source_text[context_index - 1].islower()
        and source_text[context_index - 1].isalpha()
    )
    follows_dotted_alpha = (
        source_text[context_index] == "."
        and context_index >= 1
        and source_text[context_index - 1].islower()
        and source_text[context_index - 1].isalpha()
        and (context_index < 2 or not source_text[context_index - 2].isalpha())
    )
    return (
        source_text[context_index] in ".?!:;"
        or follows_parenthesized_alpha
        or follows_dotted_alpha
    )


def _source_position_follows_reference_word(
    source_text: str,
    start: int,
    *,
    cache: dict[int, bool] | None = None,
) -> bool:
    if cache is not None and start in cache:
        return cache[start]
    chain_starts = [start]
    current_start = start
    result: bool | None = None
    while True:
        index = current_start - 1
        while index >= 0 and source_text[index].isspace():
            index -= 1
        if index >= 0 and source_text[index] == ")":
            marker_start = source_text.rfind("(", 0, index)
            if marker_start >= 0 and re.fullmatch(
                r"(?:[a-z]|\d+)", source_text[marker_start + 1 : index]
            ):
                if cache is not None and marker_start in cache:
                    result = cache[marker_start]
                    break
                chain_starts.append(marker_start)
                current_start = marker_start
                continue
        if (
            index >= 1
            and source_text[index] == "."
            and source_text[index - 1].islower()
            and source_text[index - 1].isalpha()
            and (index < 2 or not source_text[index - 2].isalpha())
        ):
            marker_start = index - 1
            if cache is not None and marker_start in cache:
                result = cache[marker_start]
                break
            chain_starts.append(marker_start)
            current_start = marker_start
            continue
        break
    if result is None:
        word_end = index + 1
        while index >= 0 and source_text[index].isalpha():
            index -= 1
        result = source_text[index + 1 : word_end].lower() in {
            "paragraph",
            "paragraphs",
            "subparagraph",
            "subparagraphs",
            "subsection",
            "subsections",
            "section",
            "sections",
            "clause",
            "clauses",
            "item",
            "items",
        }
    if cache is not None:
        cache.update(dict.fromkeys(chain_starts, result))
    return result


def _source_marker_has_nested_list_context(
    source_text: str,
    *,
    start: int,
    lower_bound: int,
    allow_period: bool,
) -> bool:
    line_start = source_text.rfind("\n", lower_bound, start)
    if line_start >= lower_bound:
        line_prefix = source_text[line_start + 1 : start]
        if line_prefix and not line_prefix.strip():
            return len(line_prefix.expandtabs(2)) >= 2

    index = start - 1
    while index >= lower_bound and source_text[index].isspace():
        index -= 1
    word_end = index + 1
    while index >= lower_bound and source_text[index].isalpha():
        index -= 1
    if source_text[index + 1 : word_end].lower() in {"and", "or"}:
        while index >= lower_bound and source_text[index].isspace():
            index -= 1
    punctuation = (":", ";", ",", ".") if allow_period else (":", ";", ",")
    return index >= lower_bound and source_text[index] in punctuation


def _rule_source_subsection_scope(
    rule_source: str,
) -> dict[str, frozenset[str] | None]:
    """Parse top-level markers and explicit numeric child ranges."""
    scope: dict[str, set[str] | None] = {}
    current_top: str | None = None
    for segment, parent_alpha_scope in _masked_numeric_parent_alpha_segments(
        rule_source
    ):
        segment, deeper_roman_parents = _masked_deeper_roman_segment(segment)
        if deeper_roman_parents and re.search(r"\([a-z]\)", segment) is None:
            current_top = None
            continue
        if parent_alpha_scope and re.search(r"\([a-z]\)", segment) is None:
            current_top = None
            continue
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
            start_token = match.group("start")
            end_token = match.group("end")
            if len(start_token) <= 10 and len(end_token) <= 10:
                start = int(start_token)
                end = int(end_token)
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
