"""Opt-in authoritative source-unit completeness accounting.

This module deliberately sits beside, rather than inside, the numeric
extractor.  It consumes the extractor and scalar-inventory interfaces supplied
by :mod:`validator_pipeline`, which keeps locale-tokenizer work independent
from the source-unit admission policy implemented here.
"""

from __future__ import annotations

import ast
import contextlib
import math
import re
import textwrap
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import yaml


class NumericOccurrenceLike(Protocol):
    """Typed numeric source evidence consumed from the shared extractor."""

    value: float
    start: int
    end: int
    raw: str
    has_rate_context: bool
    source_value: float | None
    requires_rate_context: bool


NumericOccurrenceExtractor = Callable[[str], Sequence[NumericOccurrenceLike]]
NamedScalarExtractor = Callable[[str], Sequence[Any]]
NumericGroundingPredicate = Callable[
    [float, Iterable[NumericOccurrenceLike]],
    bool,
]


@dataclass(frozen=True)
class SourceStructureBranch:
    """One operative structural branch in an authoritative source unit."""

    path: tuple[str, ...]
    kind: str
    label: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class CompleteSourceUnitAnalysis:
    """Deterministic completeness results for one RuleSpec/source pair."""

    issues: tuple[str, ...]
    branches: tuple[SourceStructureBranch, ...]
    source_numeric_occurrence_count: int
    covered_source_numeric_occurrence_count: int
    missing_source_numeric_occurrence_count: int


@dataclass(frozen=True)
class _NumericInterval:
    """One source-stated range retaining exact endpoint evidence."""

    lower: NumericOccurrenceLike | None
    lower_inclusive: bool
    upper: NumericOccurrenceLike | None
    upper_inclusive: bool


@dataclass(frozen=True)
class _FormulaTraceStep:
    """One control-flow decision evaluated by a companion case."""

    kind: str
    selectors: tuple[str, ...]
    choice: int


@dataclass(frozen=True)
class _FormulaExecution:
    """The selected control-flow path and reconstructed reachable formula."""

    trace: tuple[_FormulaTraceStep, ...]
    leaf: str


@dataclass(frozen=True)
class _FormulaBranchNode:
    """One parsed RuleSpec conditional or match expression."""

    start: int
    end: int
    kind: str
    selectors: tuple[str, ...]
    patterns: tuple[str, ...]
    choices: tuple[str, ...]


_PARAGRAPH_MARKER = re.compile(
    r"(?m)^[ \t]*(?P<marker>\((?P<label>\d+[a-z]?|[a-z])\))(?=\s|bis\b)",
    flags=re.IGNORECASE,
)
_NUMBER_MARKER = re.compile(
    r"(?m)^[ \t]*(?P<marker>(?P<label>\d+[a-z]?)\.)[ \t]+",
    flags=re.IGNORECASE,
)
_LETTER_MARKER = re.compile(
    r"(?m)^[ \t]*(?P<marker>(?P<label>[a-z]{1,2})\))[ \t]+",
    flags=re.IGNORECASE,
)
_GLUED_SENTENCE_MARKER = re.compile(
    r"(?<![\w])(?P<label>[1-9]\d?)(?=[A-ZÄÖÜ])"
)
_EXPLICIT_SENTENCE_MARKER = re.compile(
    r"(?:(?<=^)|(?<=[.;]))[ \t]*Satz[ \t]+(?P<label>[1-9]\d?)(?=[ \t]+[A-ZÄÖÜ])",
    flags=re.IGNORECASE | re.MULTILINE,
)
_EDITORIAL_OMISSION_ONLY = re.compile(
    r"^\s*"
    r"(?:(?:\((?:\d+[a-z]?|[a-z])\)|\d+[a-z]?\.|[a-z]\))\s*)?"
    r"(?:bis\s+\(\d+[a-z]?\)\s*)?"
    r"(?:\(\s*)?"
    r"(?:weggefallen|aufgehoben|repealed|omitted|\.{3,})"
    r"(?:\s*\))?"
    r"\s*[.;]?\s*$",
    flags=re.IGNORECASE,
)
_ARITHMETIC_EXPRESSION = re.compile(
    r"(?:\d+(?:[.,]\d+)?|[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß]*)"
    r"[ \t]*(?:[+*/=×·•∗∙]|(?<!\w)[−–-](?!\w))[ \t]*"
    r"(?:\d+(?:[.,]\d+)?|[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß]*)"
)
_COMPUTATION_LANGUAGE = re.compile(
    r"\b(?:"
    r"bemisst\s+sich\s+nach|berechn(?:et|en|ung)|ergibt\s+sich|"
    r"(?:zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn)fache|"
    r"(?:das\s+)?(?:doppelte|dreifache|vierfache|fünffache|sechsfache|"
    r"siebenfache|achtfache|neunfache|zehnfache)\s+(?:des|der|von)|"
    r"hälfte|zehntausendstel|übersteigenden\s+teils|"
    r"(?:ein(?:e[nsrm]?)?\s+)?(?:drittel|viertel|fünftel|sechstel|"
    r"siebtel|achtel|neuntel|zehntel)\s+(?:des|der|von)|"
    r"summe\s+(?:aus|der|von)|produkt\s+(?:aus|der|von)|"
    r"unterschied|differenz|"
    r"geteilt\s+durch|durch\s+(?:\d+|[a-zäöüß]+)\s+geteilt|"
    r"multipliziert\s+mit|"
    r"\bmal\s+(?:\d+(?:[.,]\d+)?|"
    r"ein(?:s|e[nsrm]?)?|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn)|"
    r"(?:wird|werden)\s+(?:verdoppelt|verdreifacht|vervierfacht)|"
    r"\d+(?:[.,]\d+)?\s*-?fach(?:e[nsrm]?)?|"
    r"(?:vermindert|erhöht|gekürzt|vermehrt)\s+um|"
    r"(?:erhöht|mindert|vermindert|kürzt|vermehrt)\s+sich\s+um|"
    r"(?:abzüglich|zuzüglich)|prozent\s+(?:des|der|von)|"
    r"\d+(?:[.,]\d+)?\s+(?:vom\s+hundert|v\.?\s*h\.?)\s+"
    r"(?:des|der|von)|"
    r"splitting-verfahren|verfahren\s+nach\s+absatz|"
    r"calculated|computed|computation|multiplied|divided|"
    r"sum\s+of|difference\s+between|product\s+of|twice|half\s+of|"
    r"percentage\s+of|in\s+excess\s+of|"
    r"equals?[^.;]{0,100}\b(?:plus|minus|times)\b"
    r")\b",
    flags=re.IGNORECASE,
)
_ROUNDING_LANGUAGE = re.compile(
    r"\b(?:"
    r"abgerundet(?:e|en|er|es)?|abzurunden|aufgerundet(?:e|en|er|es)?|"
    r"aufzurunden|kaufmännisch(?:\s+zu)?\s+runden|"
    r"round(?:ed|ing)?(?:\s+(?:down|up|to\s+the\s+nearest))?"
    r")\b",
    flags=re.IGNORECASE,
)
_DOWN_ROUNDING_LANGUAGE = re.compile(
    r"\b(?:abgerundet(?:e|en|er|es)?|abzurunden|round(?:ed|ing)?\s+down)\b",
    flags=re.IGNORECASE,
)
_UP_ROUNDING_LANGUAGE = re.compile(
    r"\b(?:aufgerundet(?:e|en|er|es)?|aufzurunden|round(?:ed|ing)?\s+up)\b",
    flags=re.IGNORECASE,
)
_NEAREST_ROUNDING_LANGUAGE = re.compile(
    r"\b(?:kaufmännisch(?:\s+zu)?\s+runden|round(?:ed|ing)?\s+to\s+the\s+nearest)\b",
    flags=re.IGNORECASE,
)
_EXCEPTION_LANGUAGE = re.compile(
    r"\b(?:"
    r"except(?:ion)?|unless|subject\s+to|shall\s+not\s+apply|"
    r"does\s+not\s+apply|notwithstanding|vorbehaltlich|ausnahme|"
    r"es\s+sei\s+denn|gilt\s+nicht(?:\s*,?\s*wenn)?|"
    r"findet\s+keine\s+anwendung|soweit\s+nicht|"
    r"außer|ausgenommen|abweichend\s+von|jedoch\s+nicht|"
    r"(?:[1-9]\d?)?voraussetzung[^.;]{0,160}\bnicht\b"
    r")",
    flags=re.IGNORECASE,
)
_PRECISE_DEFERRAL_DEPENDENCY = re.compile(
    r"(?:"
    r"§{1,2}\s*\d+[a-z]?|"
    r"\b(?:section|subsection|paragraph|absatz|abs\.|satz|nummer|nr\.)"
    r"\s*\d+[a-z]?|"
    r"\b[a-z]{2}(?:-[a-z0-9-]+)?/(?:statute|regulation|guidance|manual)/"
    r"[A-Za-z0-9_./-]+|"
    r"\b[a-z]{2}(?:-[a-z0-9-]+)?:"
    r"(?:statutes|regulations|guidance|manuals)/[A-Za-z0-9_./-]+#[A-Za-z0-9_]+"
    r")",
    flags=re.IGNORECASE,
)
_MISSING_DEPENDENCY_LANGUAGE = re.compile(
    r"\b(?:"
    r"requires?|depends?\s+on|missing|not\s+yet\s+encoded|unavailable|"
    r"cannot\s+be\s+(?:computed|encoded|resolved)|until|"
    r"benötigt|abhängig|fehlt|nicht\s+codiert"
    r")\b",
    flags=re.IGNORECASE,
)
_ABSATZ_REFERENCE = re.compile(
    r"\b(?:Absatz|Abs\.)\s*(?P<label>\d+[a-z]?)\b",
    flags=re.IGNORECASE,
)
_SATZ_REFERENCE = re.compile(
    r"\b(?:Satz(?:es)?|Sätze)\s*(?P<label>\d+)\b",
    flags=re.IGNORECASE,
)
_NUMMER_REFERENCE = re.compile(
    r"\b(?:Nummer|Nr\.)\s*(?P<label>\d+[a-z]?)\b",
    flags=re.IGNORECASE,
)
_BUCHSTABE_REFERENCE = re.compile(
    r"\b(?:Buchstabe|Buchst\.)\s*(?P<label>[a-z])\b",
    flags=re.IGNORECASE,
)
_GERMAN_LEGAL_CITATION = re.compile(
    r"§{1,2}\s*\d+[a-z]?"
    r"(?:\s*,\s*\d+[a-z]?)*"
    r"(?:\s*(?:und|bis)\s*\d+[a-z]?)*",
    flags=re.IGNORECASE,
)
_EXPLICIT_LEGAL_SECTION_REFERENCE = re.compile(
    r"(?:§{1,2}\s*|\b(?:sections?|paragra(?:f|phs?))\s+)"
    r"(?P<section>\d+[a-z]?)",
    flags=re.IGNORECASE,
)
_ENGLISH_LEGAL_CITATION = re.compile(
    r"\b(?:sections?|secs?\.?|regulations?|paragraphs?)\s+"
    r"\d+(?:\.\d+)*(?:\s*(?:through|to|-|and|,)\s*\d+(?:\.\d+)*)*",
    flags=re.IGNORECASE,
)
_STRUCTURAL_REFERENCE = re.compile(
    r"\b(?:Absatz|Abs\.|Satz(?:es)?|Sätze|Nummer|Nr\.|Buchstabe|Buchst\.)"
    r"\s*\d*[a-z]?\b",
    flags=re.IGNORECASE,
)
_FORMULA_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def recognize_source_structure(source_text: str) -> tuple[SourceStructureBranch, ...]:
    """Recognize paragraph, list, letter, and glued German sentence markers."""

    paragraph_matches = list(_PARAGRAPH_MARKER.finditer(source_text))
    branches: list[SourceStructureBranch] = []
    paragraph_segments: list[tuple[tuple[str, ...], int, int, str]] = []

    for index, match in enumerate(paragraph_matches):
        start = match.start()
        end = (
            paragraph_matches[index + 1].start()
            if index + 1 < len(paragraph_matches)
            else len(source_text)
        )
        label = match.group("label").lower()
        text = source_text[start:end].strip()
        if _is_editorial_omission(text):
            continue
        path = (label,)
        branches.append(
            SourceStructureBranch(path, "paragraph", match.group("marker"), text, start, end)
        )
        paragraph_segments.append((path, start, end, text))

    if not paragraph_segments:
        paragraph_segments = [((), 0, len(source_text), source_text)]

    for paragraph_path, paragraph_start, paragraph_end, _ in paragraph_segments:
        paragraph_text = source_text[paragraph_start:paragraph_end]
        number_matches = list(_NUMBER_MARKER.finditer(paragraph_text))
        number_segments: list[tuple[tuple[str, ...], int, int]] = []
        for index, match in enumerate(number_matches):
            start = paragraph_start + match.start()
            end = (
                paragraph_start + number_matches[index + 1].start()
                if index + 1 < len(number_matches)
                else paragraph_end
            )
            label = match.group("label").lower()
            path = (*paragraph_path, label)
            text = source_text[start:end].strip()
            if _is_editorial_omission(text):
                continue
            branches.append(
                SourceStructureBranch(
                    path, "number", match.group("marker"), text, start, end
                )
            )
            number_segments.append((path, start, end))

        letter_containers = number_segments or [
            (paragraph_path, paragraph_start, paragraph_end)
        ]
        for container_path, container_start, container_end in letter_containers:
            container_text = source_text[container_start:container_end]
            letter_matches = list(_LETTER_MARKER.finditer(container_text))
            for index, match in enumerate(letter_matches):
                start = container_start + match.start()
                end = (
                    container_start + letter_matches[index + 1].start()
                    if index + 1 < len(letter_matches)
                    else container_end
                )
                label = match.group("label").lower()
                path = (*container_path, label)
                text = source_text[start:end].strip()
                if _is_editorial_omission(text):
                    continue
                branches.append(
                    SourceStructureBranch(
                        path, "letter", match.group("marker"), text, start, end
                    )
                )

        sentence_matches = sorted(
            (
                *_GLUED_SENTENCE_MARKER.finditer(paragraph_text),
                *_EXPLICIT_SENTENCE_MARKER.finditer(paragraph_text),
            ),
            key=lambda item: item.start(),
        )
        for index, match in enumerate(sentence_matches):
            start = paragraph_start + match.start()
            end = (
                paragraph_start + sentence_matches[index + 1].start()
                if index + 1 < len(sentence_matches)
                else paragraph_end
            )
            label = match.group("label")
            path = (*paragraph_path, f"satz-{label}")
            text = source_text[start:end].strip()
            if _is_editorial_omission(text):
                continue
            branches.append(
                SourceStructureBranch(path, "sentence", f"Satz {label}", text, start, end)
            )

    unique = {
        (branch.path, branch.kind, branch.start): branch for branch in branches
    }
    return tuple(
        sorted(
            unique.values(),
            key=lambda branch: (branch.start, len(branch.path), branch.kind),
        )
    )


def _is_editorial_omission(text: str) -> bool:
    return bool(_EDITORIAL_OMISSION_ONLY.fullmatch(text))


def source_states_explicit_computation(source_text: str) -> bool:
    """Return whether text states a computation rather than only a scalar."""

    return bool(
        _has_substantive_arithmetic_expression(source_text)
        or _COMPUTATION_LANGUAGE.search(source_text)
        or _ROUNDING_LANGUAGE.search(source_text)
    )


def _has_substantive_arithmetic_expression(source_text: str) -> bool:
    """Ignore slash-separated year spans while recognizing actual arithmetic."""

    for match in _ARITHMETIC_EXPRESSION.finditer(source_text):
        expression = re.sub(r"\s+", "", match.group(0))
        if re.fullmatch(r"(?:19|20)\d{2}/(?:19|20)\d{2}", expression):
            continue
        return True
    return False


def analyze_complete_source_unit(
    content: str,
    authoritative_source_text: str,
    *,
    corpus_citation_path: str,
    test_cases: Sequence[object] | None,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    extract_named_scalars: NamedScalarExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
    artifact_numeric_values: Sequence[float] | None = None,
) -> CompleteSourceUnitAnalysis:
    """Analyze one artifact against its authoritative, resolver-owned body."""

    source_text = authoritative_source_text.strip()
    if not source_text:
        issue = (
            "[complete-source-unit:authoritative-source] "
            "The authoritative corpus body is required; `module.summary` is "
            "never completeness evidence."
        )
        return CompleteSourceUnitAnalysis((issue,), (), 0, 0, 0)

    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if isinstance(payload, dict) and payload.get("format") == "rulespec/v1":
            return _analyze_rulespec_payload(
                payload,
                content=content,
                source_text=source_text,
                corpus_citation_path=corpus_citation_path,
                test_cases=test_cases,
                extract_numeric_occurrences=extract_numeric_occurrences,
                extract_named_scalars=extract_named_scalars,
                numeric_value_is_grounded=numeric_value_is_grounded,
                artifact_numeric_values=artifact_numeric_values,
            )

    return CompleteSourceUnitAnalysis((), (), 0, 0, 0)


def _analyze_rulespec_payload(
    payload: dict[str, Any],
    *,
    content: str,
    source_text: str,
    corpus_citation_path: str,
    test_cases: Sequence[object] | None,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    extract_named_scalars: NamedScalarExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
    artifact_numeric_values: Sequence[float] | None,
) -> CompleteSourceUnitAnalysis:
    branches = recognize_source_structure(source_text)
    (
        all_covered_paths,
        principal_paths,
        principal_rules,
        principal_rule_paths,
    ) = _rule_coverage(
        payload,
        source_text=source_text,
        branches=branches,
        corpus_citation_path=corpus_citation_path,
    )
    deferred_paths, imprecise_deferrals = _deferred_coverage(
        payload,
        corpus_citation_path=corpus_citation_path,
        source_text=source_text,
        branches=branches,
    )
    issues: list[str] = []
    issues.extend(imprecise_deferrals)

    for branch in branches:
        if _path_covered(branch.path, all_covered_paths, deferred_paths):
            continue
        issues.append(
            "[complete-source-unit:structure] "
            f"Source branch {branch.label} at "
            f"{_branch_citation(corpus_citation_path, branch)} is neither "
            "encoded nor precisely deferred."
        )

    active_branches = tuple(
        branch
        for branch in branches
        if not _path_is_deferred(branch.path, deferred_paths)
    )
    all_formula_branches = _source_formula_branches(
        source_text,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    formula_branches = _source_formula_branches(
        source_text,
        branches=branches,
        active_branches=active_branches,
        deferred_paths=deferred_paths,
    )
    principal_formula_clause_rules = _principal_formula_clause_rules(
        formula_branches,
        principal_rules=principal_rules,
        principal_rule_paths=principal_rule_paths,
        corpus_citation_path=corpus_citation_path,
    )
    source_has_computation = source_states_explicit_computation(source_text)
    if source_has_computation:
        if formula_branches:
            for branch in formula_branches:
                if principal_formula_clause_rules[branch]:
                    continue
                issues.append(
                    "[complete-source-unit:formula-output] "
                    f"Explicit source computation {branch.label} in "
                    f"{_branch_citation(corpus_citation_path, branch)} has no "
                    "principal derived/relation output (`derived` or "
                    "`derived_relation`) and is not "
                    "precisely deferred; parameter-only representation is invalid."
                )
        elif (
            not all_formula_branches
            and not _path_covered((), principal_paths, deferred_paths)
        ):
            issues.append(
                "[complete-source-unit:formula-output] Explicit source computation "
                "has no principal derived/relation output (`derived` or "
                "`derived_relation`); "
                "parameter-only representation is invalid."
            )

    source_occurrences = tuple(
        extract_numeric_occurrences(authoritative_numeric_recall_text(source_text))
    )
    named_values = (
        tuple(float(value) for value in artifact_numeric_values)
        if artifact_numeric_values is not None
        else tuple(
            float(item.value)
            for item in extract_named_scalars(content)
            if hasattr(item, "value")
        )
    )
    covered_source_values = 0
    missing_source_values: list[float] = []
    for occurrence in source_occurrences:
        if any(
            numeric_value_is_grounded(named_value, (occurrence,))
            for named_value in named_values
        ):
            covered_source_values += 1
        else:
            missing_source_values.append(float(occurrence.value))
    for value in sorted(set(missing_source_values)):
        issues.append(
            "[complete-source-unit:numeric-recall] Authoritative corpus numeric "
            f"value {value:g} has no named scalar representation. "
            "`module.summary` is not consulted."
        )

    if principal_rules:
        issues.extend(
            _companion_test_issues(
                principal_rules,
                principal_rule_paths=principal_rule_paths,
                principal_formula_clause_rules=principal_formula_clause_rules,
                formula_branches=formula_branches,
                branches=branches,
                source_text=source_text,
                corpus_citation_path=corpus_citation_path,
                deferred_paths=deferred_paths,
                test_cases=test_cases,
                extract_numeric_occurrences=extract_numeric_occurrences,
                numeric_value_is_grounded=numeric_value_is_grounded,
                formula_environment=_constant_rule_environment(payload),
            )
        )

    return CompleteSourceUnitAnalysis(
        tuple(dict.fromkeys(issues)),
        branches,
        len(source_occurrences),
        covered_source_values,
        len(missing_source_values),
    )


def _rule_coverage(
    payload: dict[str, Any],
    *,
    source_text: str,
    branches: Sequence[SourceStructureBranch],
    corpus_citation_path: str,
) -> tuple[
    set[tuple[str, ...]],
    set[tuple[str, ...]],
    dict[str, dict[str, Any]],
    dict[str, set[tuple[str, ...]]],
]:
    all_paths: set[tuple[str, ...]] = set()
    principal_paths: set[tuple[str, ...]] = set()
    principal_rules: dict[str, dict[str, Any]] = {}
    principal_rule_paths: dict[str, set[tuple[str, ...]]] = {}
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return all_paths, principal_paths, principal_rules, principal_rule_paths

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        kind = str(rule.get("kind") or "").strip().lower()
        name = str(rule.get("name") or "").strip()
        paths = _paths_from_source_reference(
            str(rule.get("source") or ""),
            corpus_citation_path=corpus_citation_path,
        )
        for excerpt_citation_path, excerpt in _rule_source_excerpts(rule):
            if (
                excerpt_citation_path.strip("/").lower()
                != corpus_citation_path.strip("/").lower()
            ):
                continue
            excerpt_branch = _most_specific_excerpt_branch(excerpt, branches)
            if excerpt_branch is not None:
                paths.add(excerpt_branch.path)
            elif not branches:
                paths.add(())
        for path in paths:
            all_paths.update(_path_prefixes(path))
        if kind in {"derived", "derived_relation"} and name:
            principal_rules[name] = rule
            principal_rule_paths.setdefault(name, set()).update(paths)
            for path in paths:
                principal_paths.update(_path_prefixes(path))
    return all_paths, principal_paths, principal_rules, principal_rule_paths


def _rule_source_excerpts(rule: dict[str, Any]) -> Iterable[tuple[str, str]]:
    return tuple(
        (citation_path, excerpt)
        for _path, citation_path, excerpt in _rule_source_excerpt_atoms(rule)
    )


def _rule_formula_source_excerpts(
    rule: dict[str, Any],
) -> Iterable[tuple[str, str]]:
    return tuple(
        (citation_path, excerpt)
        for path, citation_path, excerpt in _rule_source_excerpt_atoms(rule)
        if re.fullmatch(r"versions(?:\[\d+\])?\.formula", path)
    )


def _rule_source_excerpt_atoms(
    rule: dict[str, Any],
) -> Iterable[tuple[str, str, str]]:
    metadata = rule.get("metadata")
    proof = metadata.get("proof") if isinstance(metadata, dict) else None
    if not isinstance(proof, dict):
        proof = rule.get("proof")
    atoms = proof.get("atoms") if isinstance(proof, dict) else None
    if not isinstance(atoms, list):
        return ()
    excerpts: list[tuple[str, str, str]] = []
    for atom in atoms:
        source = atom.get("source") if isinstance(atom, dict) else None
        path = str(atom.get("path") or "").strip() if isinstance(atom, dict) else ""
        excerpt = source.get("excerpt") if isinstance(source, dict) else None
        citation_path = (
            str(source.get("corpus_citation_path") or "").strip()
            if isinstance(source, dict)
            else ""
        )
        if isinstance(excerpt, str) and excerpt.strip() and citation_path and path:
            excerpts.append((path, citation_path, excerpt.strip()))
    return excerpts


def _principal_formula_clause_rules(
    formula_branches: Sequence[SourceStructureBranch],
    *,
    principal_rules: dict[str, dict[str, Any]],
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
    corpus_citation_path: str,
) -> dict[SourceStructureBranch, set[str]]:
    """Bind each computation clause to principal output evidence.

    A direct source path is sufficient when its structural branch contains one
    computation. When several computations share that path, each one needs a
    source-verbatim proof excerpt on the principal rule that claims it.
    """

    clause_count_by_path: dict[tuple[str, ...], int] = {}
    for clause in formula_branches:
        clause_count_by_path[clause.path] = (
            clause_count_by_path.get(clause.path, 0) + 1
        )

    clause_rules: dict[SourceStructureBranch, set[str]] = {}
    normalized_citation_path = corpus_citation_path.strip("/").lower()
    for clause in formula_branches:
        path_rules = set(_rules_covering_branch(clause, principal_rule_paths))
        if clause_count_by_path[clause.path] == 1:
            clause_rules[clause] = path_rules
            continue
        clause_text = _normalized_formula_clause_text(clause.text)
        rounding_direction = _rounding_only_direction(clause.text)
        clause_rules[clause] = {
            rule_name
            for rule_name in path_rules
            if (
                rounding_direction is not None
                and _rule_implements_rounding(
                    principal_rules[rule_name],
                    rounding_direction,
                )
            )
            or any(
                (
                    excerpt_text := _normalized_formula_clause_text(excerpt)
                )
                and excerpt_citation_path.strip("/").lower()
                == normalized_citation_path
                and source_states_explicit_computation(excerpt)
                and (
                    excerpt_text in clause_text
                    or clause_text in excerpt_text
                )
                for excerpt_citation_path, excerpt in _rule_formula_source_excerpts(
                    principal_rules[rule_name]
                )
            )
        }
    return clause_rules


def _normalized_formula_clause_text(text: str) -> str:
    """Normalize a clause while ignoring its structural marker."""

    unmarked = re.sub(
        r"^\s*(?:\((?:\d+[a-z]?|[a-z])\)|\d+[a-z]?\.|[a-z]\))\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return _collapse_text(unmarked)


def _rounding_direction(text: str) -> str | None:
    if not _ROUNDING_LANGUAGE.search(text):
        return None
    if _NEAREST_ROUNDING_LANGUAGE.search(text):
        return "nearest"
    if _UP_ROUNDING_LANGUAGE.search(text):
        return "upward"
    return "downward"


def _rounding_only_direction(text: str) -> str | None:
    """Return a direction only when rounding is the clause's sole computation."""

    direction = _rounding_direction(text)
    if direction is None:
        return None
    without_rounding = _ROUNDING_LANGUAGE.sub("", text)
    if (
        _has_substantive_arithmetic_expression(without_rounding)
        or _COMPUTATION_LANGUAGE.search(without_rounding)
    ):
        return None
    return direction


def _most_specific_excerpt_branch(
    excerpt: str,
    branches: Sequence[SourceStructureBranch],
) -> SourceStructureBranch | None:
    normalized_excerpt = _collapse_text(excerpt)
    if not normalized_excerpt:
        return None
    candidates = [
        branch
        for branch in branches
        if normalized_excerpt in _collapse_text(branch.text)
    ]
    return max(
        candidates,
        key=lambda branch: (
            len(branch.path),
            -(branch.end - branch.start),
        ),
        default=None,
    )


def _paths_from_source_reference(
    source: str,
    *,
    corpus_citation_path: str,
) -> set[tuple[str, ...]]:
    value = source.strip()
    if not value or not _source_reference_targets_authoritative_unit(
        value,
        corpus_citation_path=corpus_citation_path,
    ):
        return set()
    paths: set[tuple[str, ...]] = set()
    escaped_path = re.escape(corpus_citation_path.rstrip("/"))
    for match in re.finditer(
        rf"(?<![A-Za-z0-9_]){escaped_path}"
        r"(?P<suffix>(?:/[A-Za-z0-9-]+)+|(?:\([A-Za-z0-9-]+\))+)",
        value,
        flags=re.IGNORECASE,
    ):
        suffix = match.group("suffix")
        components = (
            [part for part in suffix.split("/") if part]
            if suffix.startswith("/")
            else re.findall(r"\(([A-Za-z0-9-]+)\)", suffix)
        )
        if components:
            base_components = tuple(component.lower() for component in components)
            trailing = _reference_qualifier_tail(
                value,
                start=match.end(),
                next_reference=corpus_citation_path,
            )
            paths.add(
                (*base_components, *_keyword_path_components(trailing, base_components))
            )

    keyword_components = _keyword_path_components(value)
    if keyword_components:
        paths.add(tuple(keyword_components))

    current_section = corpus_citation_path.rstrip("/").rsplit("/", 1)[-1]
    for match in re.finditer(
        rf"§\s*{re.escape(current_section)}"
        r"(?P<suffix>(?:\([A-Za-z0-9-]+\))+)",
        value,
        flags=re.IGNORECASE,
    ):
        components = re.findall(r"\(([A-Za-z0-9-]+)\)", match.group("suffix"))
        if components:
            base_components = tuple(component.lower() for component in components)
            trailing = _reference_qualifier_tail(
                value,
                start=match.end(),
                next_reference=f"§{current_section}",
            )
            paths.add(
                (*base_components, *_keyword_path_components(trailing, base_components))
            )
    if not paths:
        paths.add(())
    return paths


def _reference_qualifier_tail(
    source: str,
    *,
    start: int,
    next_reference: str,
) -> str:
    """Limit nested Absatz/Nummer/Satz qualifiers to one source reference."""

    trailing = source[start:]
    for separator in (";", "\n", next_reference):
        trailing = trailing.split(separator, 1)[0]
    return trailing


def _keyword_path_components(
    source: str,
    existing: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return hierarchical German structure qualifiers not already explicit."""

    components: list[str] = []
    existing_set = {item.lower() for item in existing}
    for pattern, prefix in (
        (_ABSATZ_REFERENCE, ""),
        (_NUMMER_REFERENCE, ""),
        (_BUCHSTABE_REFERENCE, ""),
        (_SATZ_REFERENCE, "satz-"),
    ):
        if match := pattern.search(source):
            component = f"{prefix}{match.group('label').lower()}"
            if component not in existing_set:
                components.append(component)
                existing_set.add(component)
    return tuple(components)


def _source_reference_targets_authoritative_unit(
    source: str,
    *,
    corpus_citation_path: str,
) -> bool:
    """Require branch claims to identify this unit, not merely a branch label."""

    candidates = {
        corpus_citation_path.rstrip("/"),
        _rulespec_target_base(corpus_citation_path),
    }
    if any(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(candidate)}(?![A-Za-z0-9_.-])",
            source,
            flags=re.IGNORECASE,
        )
        for candidate in candidates
        if candidate
    ):
        return True
    current_section = corpus_citation_path.rstrip("/").rsplit("/", 1)[-1]
    return bool(
        current_section
        and re.search(
            rf"§{{1,2}}\s*{re.escape(current_section)}(?![A-Za-z0-9.])",
            source,
            flags=re.IGNORECASE,
        )
    )


def _deferred_coverage(
    payload: dict[str, Any],
    *,
    corpus_citation_path: str,
    source_text: str,
    branches: Sequence[SourceStructureBranch],
) -> tuple[set[tuple[str, ...]], list[str]]:
    module = payload.get("module")
    records = module.get("deferred_outputs") if isinstance(module, dict) else None
    if not isinstance(records, list):
        return set(), []
    base_target = _rulespec_target_base(corpus_citation_path)
    covered: set[tuple[str, ...]] = set()
    issues: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        output = str(record.get("output") or "").strip()
        output_path = output.split("#", 1)[0]
        path: tuple[str, ...] | None = None
        if output_path == base_target:
            path = ()
        elif output_path.startswith(f"{base_target}/"):
            path = tuple(
                part.lower()
                for part in output_path[len(base_target) + 1 :].split("/")
                if part
            )
        if path is None:
            continue
        reason = str(record.get("reason") or "").strip()
        blocked_by = record.get("blocked_by")
        normalized_base_target = base_target.lower()
        blocker_targets = tuple(
            item.strip()
            for item in blocked_by
            if isinstance(item, str)
        ) if isinstance(blocked_by, list) else ()
        exact_blockers = (
            bool(blocker_targets)
            and len(blocker_targets) == len(blocked_by)
            and all(
                re.fullmatch(
                    r"[a-z]{2}(?:-[a-z0-9-]+)?:"
                    r"[A-Za-z0-9_./-]+#[A-Za-z_][A-Za-z0-9_]*",
                    item,
                    flags=re.IGNORECASE,
                )
                and item.lower() != output.lower()
                and not (
                    item.lower().split("#", 1)[0]
                    == normalized_base_target
                    or item.lower().split("#", 1)[0].startswith(
                        f"{normalized_base_target}/"
                    )
                )
                for item in blocker_targets
            )
        )
        reason_identifies_dependency = _reason_names_external_dependency(
            reason,
            corpus_citation_path=corpus_citation_path,
        )
        source_scope_text = _deferred_source_scope_text(
            path,
            source_text=source_text,
            branches=branches,
        )
        if "blocked_by" in record:
            precise = (
                exact_blockers
                and bool(_MISSING_DEPENDENCY_LANGUAGE.search(reason))
                and all(
                    _reason_identifies_blocker(reason, blocker)
                    for blocker in blocker_targets
                )
                and all(
                    _reason_identifies_blocker(source_scope_text, blocker)
                    for blocker in blocker_targets
                )
            )
        else:
            precise = (
                reason_identifies_dependency
                and _reason_dependency_is_source_bound(reason, source_scope_text)
            )
        if precise:
            covered.add(path)
        else:
            branch_label = path[0] if path else "source unit"
            rendered_path = "/".join(path) or "<source-unit>"
            issues.append(
                "[complete-source-unit:deferral] "
                f"`module.deferred_outputs[{index}]` identifies source branch "
                f"({branch_label}) (`{rendered_path}`) but its deferral does not "
                "name an exact missing "
                "dependency/citation."
            )
    return covered, issues


def _deferred_source_scope_text(
    path: tuple[str, ...],
    *,
    source_text: str,
    branches: Sequence[SourceStructureBranch],
) -> str:
    if not path:
        return source_text
    candidates = [branch.text for branch in branches if branch.path == path]
    return max(candidates, key=len, default="")


def _reason_identifies_blocker(reason: str, blocker: str) -> bool:
    """Bind a typed blocker to the legal section or concept named in prose."""

    target_path, _separator, symbol = blocker.partition("#")
    section = target_path.rstrip("/").rsplit("/", 1)[-1]
    explicit_sections = {
        match.group("section").lower()
        for match in _EXPLICIT_LEGAL_SECTION_REFERENCE.finditer(reason)
    }
    if explicit_sections and section.lower() not in explicit_sections:
        return False
    if section and re.search(
        rf"(?:§{{1,2}}\s*|\b(?:section|paragraph|paragraf|abschnitt)\s+)"
        rf"{re.escape(section)}(?![A-Za-z0-9])",
        reason,
        flags=re.IGNORECASE,
    ):
        return True
    normalized_reason = re.sub(r"[^a-z0-9äöüß]+", " ", reason.lower()).strip()
    normalized_symbol = re.sub(r"[^a-z0-9äöüß]+", " ", symbol.lower()).strip()
    return bool(
        normalized_symbol
        and re.search(
            rf"(?<![a-z0-9äöüß]){re.escape(normalized_symbol)}"
            r"(?![a-z0-9äöüß])",
            normalized_reason,
        )
    )


def _reason_dependency_is_source_bound(reason: str, source_scope_text: str) -> bool:
    """Require a prose-only dependency citation to occur in the deferred source."""

    for match in _PRECISE_DEFERRAL_DEPENDENCY.finditer(reason):
        dependency = match.group(0).strip()
        if "#" in dependency and ":" in dependency:
            if _reason_identifies_blocker(source_scope_text, dependency):
                return True
            continue
        section_match = re.search(r"\d+[a-z]?", dependency, flags=re.IGNORECASE)
        if section_match and _reason_identifies_blocker(
            source_scope_text,
            f"xx:statutes/source/{section_match.group(0)}#dependency",
        ):
            return True
        normalized_dependency = re.sub(
            r"[^a-z0-9äöüß]+",
            " ",
            dependency.lower(),
        ).strip()
        normalized_source = re.sub(
            r"[^a-z0-9äöüß]+",
            " ",
            source_scope_text.lower(),
        ).strip()
        if normalized_dependency and normalized_dependency in normalized_source:
            return True
    return False


def _reason_names_external_dependency(
    reason: str,
    *,
    corpus_citation_path: str,
) -> bool:
    if not _MISSING_DEPENDENCY_LANGUAGE.search(reason):
        return False
    current_section = corpus_citation_path.rstrip("/").rsplit("/", 1)[-1].lower()
    for match in _PRECISE_DEFERRAL_DEPENDENCY.finditer(reason):
        dependency = match.group(0).strip()
        normalized = dependency.lower()
        if any(
            pattern.fullmatch(dependency)
            for pattern in (
                _ABSATZ_REFERENCE,
                _SATZ_REFERENCE,
                _NUMMER_REFERENCE,
                _BUCHSTABE_REFERENCE,
            )
        ):
            # A bare structural reference names another part of this same
            # authoritative unit, not a missing external dependency.
            continue
        if section_match := re.fullmatch(
            r"§{1,2}\s*(\d+[a-z]?)",
            dependency,
            flags=re.IGNORECASE,
        ):
            if section_match.group(1).lower() == current_section:
                continue
        if normalized.rstrip("/").endswith(
            f"/{current_section}"
        ) and "#" not in normalized:
            continue
        return True
    return False


def _rulespec_target_base(corpus_citation_path: str) -> str:
    parts = [part for part in corpus_citation_path.strip("/").split("/") if part]
    if len(parts) < 3:
        return corpus_citation_path
    jurisdiction, document_class, *tail = parts
    plural = {
        "statute": "statutes",
        "regulation": "regulations",
        "manual": "manuals",
        "guidance": "guidance",
        "policy": "policies",
        "form": "forms",
    }.get(document_class, f"{document_class}s")
    return f"{jurisdiction}:{plural}/{'/'.join(tail)}"


def _path_prefixes(path: tuple[str, ...]) -> set[tuple[str, ...]]:
    if not path:
        return {()}
    return {path[:index] for index in range(1, len(path) + 1)}


def _path_covered(
    path: tuple[str, ...],
    encoded_paths: set[tuple[str, ...]],
    deferred_paths: set[tuple[str, ...]],
) -> bool:
    return path in encoded_paths or _path_is_deferred(path, deferred_paths)


def _path_is_deferred(
    path: tuple[str, ...],
    deferred_paths: set[tuple[str, ...]],
) -> bool:
    """Let a precise non-root deferral cover its structural descendants."""

    if not path:
        return () in deferred_paths
    return any(
        deferred_path
        and path[: len(deferred_path)] == deferred_path
        for deferred_path in deferred_paths
    )


def _branch_citation(
    corpus_citation_path: str,
    branch: SourceStructureBranch,
) -> str:
    if not branch.path:
        return f"{corpus_citation_path} [source unit]"
    citation = corpus_citation_path
    components: list[str] = []
    for index, component in enumerate(branch.path):
        if component.startswith("satz-"):
            components.append(f"Satz {component.removeprefix('satz-')}")
        elif index == 0:
            components.append(f"Absatz {component}")
        elif branch.kind == "letter" and index == len(branch.path) - 1:
            components.append(f"Buchstabe {component}")
        else:
            components.append(f"Nummer {component}")
    return f"{citation}({branch.path[0]}) [{', '.join(components)}]"


def authoritative_numeric_recall_text(source_text: str) -> str:
    """Remove structural/citation ordinals, never substantive source values."""

    cleaned = _GERMAN_LEGAL_CITATION.sub("", source_text)
    cleaned = _ENGLISH_LEGAL_CITATION.sub("", cleaned)
    cleaned = _STRUCTURAL_REFERENCE.sub("", cleaned)
    cleaned = re.sub(
        r"(?m)^[ \t]*\((?:\d+[a-z]?|[a-z])\)(?:[ \t]+bis[ \t]+\(\d+[a-z]?\))?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?m)^[ \t]*\d+[a-z]?\.[ \t]+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = _GLUED_SENTENCE_MARKER.sub("", cleaned)
    return cleaned


def collect_artifact_numeric_values(
    content: str,
    *,
    extract_named_scalars: NamedScalarExtractor,
    imported_symbol_contents: Sequence[tuple[str, str]] = (),
    additional_values: Iterable[float] = (),
) -> tuple[float, ...]:
    """Collect numeric representations credited by strict recall accounting."""

    values = [float(value) for value in additional_values]
    values.extend(
        float(item.value)
        for item in extract_named_scalars(content)
        if hasattr(item, "value")
    )
    for imported_symbol, artifact_content in imported_symbol_contents:
        values.extend(
            float(item.value)
            for item in extract_named_scalars(artifact_content)
            if hasattr(item, "value")
            and _named_scalar_base_name(str(getattr(item, "name", "")))
            == imported_symbol
        )
    return tuple(values)


def _named_scalar_base_name(name: str) -> str:
    return name.split("[", 1)[0]


def _companion_test_issues(
    principal_rules: dict[str, dict[str, Any]],
    *,
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
    principal_formula_clause_rules: dict[SourceStructureBranch, set[str]],
    formula_branches: Sequence[SourceStructureBranch],
    branches: Sequence[SourceStructureBranch],
    source_text: str,
    corpus_citation_path: str,
    deferred_paths: set[tuple[str, ...]],
    test_cases: Sequence[object] | None,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
    formula_environment: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    cases = [case for case in (test_cases or ()) if isinstance(case, dict)]
    if not cases:
        return [
            "[complete-source-unit:tests] Complete source-unit controls require "
            "a companion test suite covering outputs, branches, boundaries, "
            "exceptions, and rounding rules."
        ]

    asserted_by_rule = {
        name: [
            case
            for case in cases
            if name in _test_case_output_names(case)
        ]
        for name in principal_rules
    }
    for name, asserted_cases in asserted_by_rule.items():
        if not asserted_cases:
            issues.append(
                "[complete-source-unit:tests] Principal output "
                f"`{name}` is never asserted by a companion test."
            )

    active_branches = [
        branch
        for branch in branches
        if not _path_is_deferred(branch.path, deferred_paths)
    ]
    missing_formula_branches = _unwitnessed_formula_branches(
        formula_branches,
        principal_rules=principal_rules,
        principal_formula_clause_rules=principal_formula_clause_rules,
        asserted_by_rule=asserted_by_rule,
        extract_numeric_occurrences=extract_numeric_occurrences,
        formula_environment=formula_environment,
    )
    for branch in missing_formula_branches:
        issues.append(
            "[complete-source-unit:tests] Companion tests do not demonstrate "
            f"formula branch {branch.label} at "
            f"{_branch_citation(corpus_citation_path, branch)} with a principal "
            "output assertion and, when the branch states a range, a selector "
            "input in that branch. Each formula branch needs distinct executed "
            "test evidence."
        )

    boundary_branches = active_branches or [
        SourceStructureBranch(
            (),
            "source-unit",
            "source unit",
            source_text,
            0,
            len(source_text),
        )
    ]
    boundary_obligations = _source_boundary_obligations(
        boundary_branches,
        narrative_formula_branches=formula_branches,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    missing_boundaries: list[
        tuple[SourceStructureBranch, NumericOccurrenceLike]
    ] = []
    for branch, boundary in boundary_obligations:
        if _branch_boundary_has_test_evidence(
            branch,
            boundary,
            principal_rules=principal_rules,
            principal_rule_paths=principal_rule_paths,
            asserted_by_rule=asserted_by_rule,
            numeric_value_is_grounded=numeric_value_is_grounded,
            formula_environment=formula_environment,
        ):
            continue
        missing_boundaries.append((branch, boundary))
    if missing_boundaries:
        rendered = ", ".join(
            f"{branch.label}={occurrence.value:g}"
            for branch, occurrence in missing_boundaries
        )
        issues.append(
            "[complete-source-unit:tests] Companion tests do not exercise every "
            f"source-stated boundary input; missing: {rendered}."
        )

    exception_branches = _source_exception_branches(
        source_text,
        branches=branches,
        active_branches=active_branches,
        deferred_paths=deferred_paths,
    )
    if exception_branches:
        toggled_exception_selectors = _toggled_formula_boolean_selectors(
            principal_rules,
            asserted_by_rule=asserted_by_rule,
            formula_environment=formula_environment,
        )
        missing_exception_branches = _unwitnessed_exception_branches(
            exception_branches,
            principal_rule_paths=principal_rule_paths,
            toggled_exception_selectors=toggled_exception_selectors,
        )
        if missing_exception_branches:
            missing_citations = ", ".join(
                dict.fromkeys(
                    _branch_citation(corpus_citation_path, branch)
                    for branch in missing_exception_branches
                )
            )
            issues.append(
                "[complete-source-unit:tests] Source-stated exceptions require "
                "paired positive/blocking cases that assert the affected principal "
                "output and toggle its excluding formula selector; missing at "
                f"{missing_citations}."
            )

    rounding_obligations = _source_rounding_obligations(
        source_text,
        branches=branches,
        active_branches=active_branches,
        deferred_paths=deferred_paths,
    )
    missing_rounding_formula: list[tuple[SourceStructureBranch, str]] = []
    rounding_witnesses: dict[
        tuple[SourceStructureBranch, str], set[tuple[str, int]]
    ] = {}
    for obligation in rounding_obligations:
        branch, direction = obligation
        rule_names = {
            name
            for name in _rules_covering_branch(branch, principal_rule_paths)
            if _rule_implements_rounding(principal_rules[name], direction)
        }
        if not rule_names:
            missing_rounding_formula.append(obligation)
            continue
        rounding_witnesses[obligation] = set().union(
            *(
                _fractional_rounding_case_witnesses(
                    name,
                    principal_rules[name],
                    asserted_by_rule=asserted_by_rule,
                    direction=direction,
                    formula_environment=formula_environment,
                )
                for name in rule_names
            )
        )
    for branch, direction in missing_rounding_formula:
        issues.append(
            "[complete-source-unit:tests] A source-stated "
            f"{direction} rounding rule at "
            f"{_branch_citation(corpus_citation_path, branch)} is absent from "
            "the principal formula."
        )
    missing_rounding_tests = _unmatched_evidence_obligations(rounding_witnesses)
    if missing_rounding_tests:
        rendered = ", ".join(
            _branch_citation(corpus_citation_path, branch)
            for branch, _direction in missing_rounding_tests
        )
        issues.append(
            "[complete-source-unit:tests] Companion tests do not demonstrate "
            "every source-stated rounding rule with distinct fractional input "
            f"evidence on its affected principal output; missing at {rendered}."
        )
    return issues


def _test_case_output_names(case: dict[str, Any]) -> set[str]:
    outputs = case.get("output")
    if not isinstance(outputs, dict):
        return set()
    names: set[str] = set()
    for key in outputs:
        text = str(key)
        names.add(text.rsplit("#", 1)[-1])
    return names


def _source_formula_branches(
    source_text: str,
    *,
    branches: Sequence[SourceStructureBranch],
    active_branches: Sequence[SourceStructureBranch],
    deferred_paths: set[tuple[str, ...]],
) -> tuple[SourceStructureBranch, ...]:
    """Return every explicit computation clause with its structural owner."""

    obligations: list[SourceStructureBranch] = []
    for clause_index, (start, end, clause) in enumerate(
        _source_clause_spans(source_text, branches=branches),
        start=1,
    ):
        if _span_is_deferred(
            start,
            end,
            branches=branches,
            deferred_paths=deferred_paths,
        ):
            continue
        if not source_states_explicit_computation(clause):
            continue
        owner = _most_specific_containing_branch(
            start,
            end,
            branches=active_branches,
        )
        if owner is None:
            owner = SourceStructureBranch(
                (),
                "source-unit",
                "source unit",
                source_text,
                0,
                len(source_text),
            )
        obligations.append(
            SourceStructureBranch(
                owner.path,
                "formula-clause",
                f"{owner.label} formula clause {clause_index}",
                clause,
                start,
                end,
            )
        )
    return tuple(obligations)


def _source_clause_spans(
    source_text: str,
    *,
    branches: Sequence[SourceStructureBranch],
) -> Iterable[tuple[int, int, str]]:
    """Yield offset-preserving clauses split at punctuation and structure."""

    boundary = re.compile(
        r";|[.!?](?=(?:[ \t]+[A-ZÄÖÜ(]|\s*$))",
        flags=re.MULTILINE,
    )
    split_points = {
        0,
        len(source_text),
        *(match.end() for match in boundary.finditer(source_text)),
        *(branch.start for branch in branches),
    }
    for start, end in zip(sorted(split_points), sorted(split_points)[1:]):
        raw = source_text[start:end]
        left_trimmed = len(raw) - len(raw.lstrip())
        right_trimmed = len(raw.rstrip())
        if right_trimmed > left_trimmed:
            yield (
                start + left_trimmed,
                start + right_trimmed,
                raw[left_trimmed:right_trimmed],
            )


def _span_is_deferred(
    start: int,
    end: int,
    *,
    branches: Sequence[SourceStructureBranch],
    deferred_paths: set[tuple[str, ...]],
) -> bool:
    if not branches:
        return () in deferred_paths
    return any(
        branch.start <= start
        and end <= branch.end
        and _path_is_deferred(branch.path, deferred_paths)
        for branch in branches
    )


def _most_specific_containing_branch(
    start: int,
    end: int,
    *,
    branches: Sequence[SourceStructureBranch],
) -> SourceStructureBranch | None:
    candidates = [
        branch
        for branch in branches
        if branch.start <= start and end <= branch.end
    ]
    return max(
        candidates,
        key=lambda branch: (
            len(branch.path),
            -(branch.end - branch.start),
        ),
        default=None,
    )


def _unwitnessed_formula_branches(
    branches: Sequence[SourceStructureBranch],
    *,
    principal_rules: dict[str, dict[str, Any]],
    principal_formula_clause_rules: dict[SourceStructureBranch, set[str]],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    formula_environment: dict[str, Any],
) -> tuple[SourceStructureBranch, ...]:
    """Consume each executed rule/case witness for at most one source formula."""

    candidate_witnesses = {
        branch: _formula_branch_test_witnesses(
            branch,
            principal_rules=principal_rules,
            rule_names=principal_formula_clause_rules[branch],
            asserted_by_rule=asserted_by_rule,
            extract_numeric_occurrences=extract_numeric_occurrences,
            formula_environment=formula_environment,
        )
        for branch in branches
    }
    return _unmatched_evidence_obligations(candidate_witnesses)


def _unmatched_evidence_obligations(
    candidate_witnesses: dict[Any, set[Any]],
) -> tuple[Any, ...]:
    """Return obligations excluded by a maximum one-witness-per-item match."""

    assigned_obligation_by_witness: dict[Any, Any] = {}

    def assign(obligation: Any, visited: set[Any]) -> bool:
        for witness in sorted(candidate_witnesses[obligation]):
            if witness in visited:
                continue
            visited.add(witness)
            incumbent = assigned_obligation_by_witness.get(witness)
            if incumbent is None or assign(incumbent, visited):
                assigned_obligation_by_witness[witness] = obligation
                return True
        return False

    missing: list[Any] = []
    for obligation in sorted(
        candidate_witnesses,
        key=lambda item: len(candidate_witnesses[item]),
    ):
        if not assign(obligation, set()):
            missing.append(obligation)
    return tuple(missing)


def _formula_branch_test_witnesses(
    branch: SourceStructureBranch,
    *,
    principal_rules: dict[str, dict[str, Any]],
    rule_names: set[str],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    formula_environment: dict[str, Any],
) -> set[tuple[str, str]]:
    interval = _formula_branch_interval(
        branch,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    witnesses: set[tuple[str, str]] = set()
    for rule_name in sorted(rule_names):
        rule = principal_rules[rule_name]
        selector_names = _rule_numeric_selector_names(rule)
        has_branching_formula = _rule_has_branching_formula(rule)
        for case in asserted_by_rule.get(rule_name, ()):
            execution = (
                _case_formula_execution(
                    rule,
                    case,
                    formula_environment=formula_environment,
                )
                if has_branching_formula or interval is not None
                else None
            )
            if interval is None:
                if has_branching_formula:
                    if (
                        execution is not None
                        and _formula_execution_leaf_is_computational(execution)
                    ):
                        outcome = _formula_execution_outcome(execution)
                        witnesses.add((rule_name, f"branch:{outcome}"))
                else:
                    witnesses.add((rule_name, f"case:{id(case)}"))
                continue
            if (
                execution is not None
                and _formula_execution_leaf_is_computational(execution)
                and _formula_execution_references_names(
                    execution,
                    selector_names,
                )
                and selector_names
                and any(
                    _interval_contains(interval, value)
                    for value in _case_numeric_selector_values(
                        case,
                        selector_names,
                    )
                )
            ):
                witnesses.add((rule_name, f"case:{id(case)}"))
    return witnesses


def _rule_has_branching_formula(rule: dict[str, Any]) -> bool:
    return _rule_text_has_branching_formula(_rule_formula_text(rule))


_UNRESOLVED_CONDITION_VALUE = object()


def _case_formula_execution(
    rule: dict[str, Any],
    case: dict[str, Any],
    *,
    formula_environment: dict[str, Any] | None = None,
) -> _FormulaExecution | None:
    """Resolve the reachable RuleSpec formula for one asserted test case."""

    inputs = case.get("input")
    if not isinstance(inputs, dict):
        return None
    environment: dict[str, Any] = dict(formula_environment or {})
    input_environment: dict[str, Any] = {}
    for key, value in inputs.items():
        boolean_value = _boolean_value(value)
        normalized_value = boolean_value if boolean_value is not None else value
        for name in _input_key_names(key):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                continue
            if name in input_environment and not _same_formula_value(
                input_environment[name],
                normalized_value,
            ):
                return None
            input_environment[name] = normalized_value
    environment.update(input_environment)
    formula_text = _unambiguous_rule_formula_text(rule)
    if formula_text is None:
        return None
    return _execute_formula_text(
        formula_text,
        environment=environment,
    )


def _execute_formula_text(
    formula_text: str,
    *,
    environment: dict[str, Any],
) -> _FormulaExecution | None:
    """Interpret formula control flow without evaluating the formula itself."""

    selected_text = formula_text.strip()
    trace: list[_FormulaTraceStep] = []
    for _depth in range(32):
        node = _first_formula_branch_node(selected_text)
        if node is None:
            if _rule_text_has_branching_formula(selected_text):
                return None
            return _FormulaExecution(tuple(trace), selected_text.strip())
        selected = _select_formula_branch(node, environment=environment)
        if selected is None:
            return None
        choice, evaluated_selectors = selected
        selected_body = textwrap.dedent(node.choices[choice]).strip()
        if not selected_body:
            return None
        trace.append(
            _FormulaTraceStep(
                node.kind,
                evaluated_selectors,
                choice,
            )
        )
        selected_text = (
            selected_text[: node.start]
            + selected_body
            + selected_text[node.end :]
        )
    return None


def _first_formula_branch_node(text: str) -> _FormulaBranchNode | None:
    candidates = tuple(
        candidate
        for candidate in (
            _first_multiline_if_node(text),
            _first_multiline_match_node(text),
            _first_inline_if_node(text),
            _first_inline_match_node(text),
        )
        if candidate is not None
    )
    return min(candidates, key=lambda item: item.start, default=None)


def _formula_line_records(
    text: str,
) -> tuple[tuple[int, int, int, str], ...]:
    records: list[tuple[int, int, int, str]] = []
    start = 0
    for raw_line in text.splitlines(keepends=True):
        content = raw_line.rstrip("\r\n")
        content_end = start + len(content)
        full_end = start + len(raw_line)
        records.append((start, content_end, full_end, content))
        start = full_end
    if not records or start < len(text):
        records.append((start, len(text), len(text), text[start:]))
    return tuple(records)


def _first_multiline_if_node(text: str) -> _FormulaBranchNode | None:
    lines = _formula_line_records(text)
    masked_lines = _formula_line_records(_mask_formula_strings(text))
    for index, (start, _content_end, _full_end, line) in enumerate(lines):
        masked_line = masked_lines[index][3]
        header = re.match(
            r"^(?P<indent>[ \t]*)if[ \t]+"
            r"(?P<condition>[^:\n]+):[ \t]*$",
            masked_line,
        )
        if header is None:
            continue
        base_indent = _formula_indent_width(header.group("indent"))
        headers: list[tuple[int, str, str]] = [
            (
                index,
                "if",
                line[header.start("condition") : header.end("condition")].strip(),
            )
        ]
        cursor = index + 1
        while cursor < len(lines):
            candidate = lines[cursor][3]
            masked_candidate = masked_lines[cursor][3]
            if not candidate.strip():
                cursor += 1
                continue
            indent = _formula_indent_width(
                candidate[: len(candidate) - len(candidate.lstrip())]
            )
            if indent > base_indent:
                cursor += 1
                continue
            chain_header = (
                re.match(
                    r"^[ \t]*elif[ \t]+(?P<condition>[^:\n]+):[ \t]*$",
                    masked_candidate,
                )
                if indent == base_indent
                else None
            )
            if chain_header is not None:
                headers.append(
                    (
                        cursor,
                        "elif",
                        candidate[
                            chain_header.start("condition") :
                            chain_header.end("condition")
                        ].strip(),
                    )
                )
                cursor += 1
                continue
            if indent == base_indent and re.match(
                r"^[ \t]*else:[ \t]*$",
                masked_candidate,
            ):
                headers.append((cursor, "else", ""))
                cursor += 1
                continue
            break
        chain_end = lines[cursor][0] if cursor < len(lines) else len(text)
        conditions = tuple(
            condition
            for _line_index, kind, condition in headers
            if kind != "else"
        )
        choices: list[str] = []
        for header_index, (line_index, _kind, _condition) in enumerate(headers):
            body_start = lines[line_index][2]
            body_end = (
                lines[headers[header_index + 1][0]][0]
                if header_index + 1 < len(headers)
                else chain_end
            )
            choices.append(text[body_start:body_end])
        if not conditions or not choices:
            continue
        return _FormulaBranchNode(
            start,
            chain_end,
            "if",
            conditions,
            (),
            tuple(choices),
        )
    return None


def _first_multiline_match_node(text: str) -> _FormulaBranchNode | None:
    lines = _formula_line_records(text)
    masked_lines = _formula_line_records(_mask_formula_strings(text))
    for index, (start, _content_end, _full_end, line) in enumerate(lines):
        masked_line = masked_lines[index][3]
        header = re.match(
            r"^(?P<indent>[ \t]*)match[ \t]+"
            r"(?P<selector>[^:\n]+):[ \t]*$",
            masked_line,
        )
        if header is None:
            continue
        base_indent = _formula_indent_width(header.group("indent"))
        arm_headers: list[tuple[int, str, str]] = []
        arm_indent: int | None = None
        cursor = index + 1
        while cursor < len(lines):
            candidate = lines[cursor][3]
            masked_candidate = masked_lines[cursor][3]
            if not candidate.strip():
                cursor += 1
                continue
            indent_text = candidate[: len(candidate) - len(candidate.lstrip())]
            indent = _formula_indent_width(indent_text)
            if indent <= base_indent:
                break
            arrow = masked_candidate.find("=>", len(indent_text))
            pattern = candidate[len(indent_text) : arrow].strip()
            is_arm = arrow >= 0 and bool(pattern)
            if arm_indent is None and is_arm:
                arm_indent = indent
            if is_arm and indent == arm_indent:
                arm_headers.append(
                    (
                        cursor,
                        pattern,
                        candidate[arrow + 2 :].lstrip(),
                    )
                )
            cursor += 1
        chain_end = lines[cursor][0] if cursor < len(lines) else len(text)
        if not arm_headers:
            continue
        choices: list[str] = []
        for arm_index, (line_index, _pattern, inline_body) in enumerate(
            arm_headers
        ):
            body_end = (
                lines[arm_headers[arm_index + 1][0]][0]
                if arm_index + 1 < len(arm_headers)
                else chain_end
            )
            trailing_body = text[lines[line_index][2] : body_end]
            choices.append(
                "\n".join(
                    part
                    for part in (inline_body, trailing_body)
                    if part.strip()
                )
            )
        return _FormulaBranchNode(
            start,
            chain_end,
            "match",
            (
                line[
                    header.start("selector") : header.end("selector")
                ].strip(),
            ),
            tuple(pattern for _line_index, pattern, _body in arm_headers),
            tuple(choices),
        )
    return None


def _first_inline_if_node(text: str) -> _FormulaBranchNode | None:
    masked_text = _mask_formula_strings(text)
    for candidate in re.finditer(
        r"(?<![A-Za-z0-9_])if[ \t]+",
        masked_text,
    ):
        line_end = text.find("\n", candidate.start())
        if line_end < 0:
            line_end = len(text)
        colon = masked_text.find(":", candidate.end(), line_end)
        if colon < 0 or not text[colon + 1 : line_end].strip():
            continue
        chain_headers = _find_inline_chain_headers(
            masked_text,
            body_start=colon + 1,
            limit=line_end,
        )
        if not chain_headers or chain_headers[-1][0] != "else":
            continue
        else_body_start = chain_headers[-1][2]
        end = _formula_expression_end(
            text,
            body_start=else_body_start,
            branch_start=candidate.start(),
            limit=line_end,
        )
        conditions = [text[candidate.end() : colon].strip()]
        conditions.extend(
            text[condition_start:condition_end].strip()
            for kind, _start, _body_start, condition_start, condition_end
            in chain_headers
            if kind == "elif"
        )
        body_boundaries = [
            header_start
            for _kind, header_start, _body_start, _condition_start, _condition_end
            in chain_headers
        ]
        body_starts = [
            colon + 1,
            *(
                header_body_start
                for kind, _header_start, header_body_start, _start, _end
                in chain_headers
                if kind == "elif"
            ),
        ]
        choices = [
            text[start:stop]
            for start, stop in zip(body_starts, body_boundaries)
        ]
        choices.append(text[else_body_start:end])
        return _FormulaBranchNode(
            candidate.start(),
            end,
            "if",
            tuple(conditions),
            (),
            tuple(choices),
        )
    return None


def _find_inline_chain_headers(
    masked_text: str,
    *,
    body_start: int,
    limit: int,
) -> tuple[tuple[str, int, int, int, int], ...]:
    nested_if_depth = 0
    headers: list[tuple[str, int, int, int, int]] = []
    for token in re.finditer(
        r"\b(?:if|elif|else)\b",
        masked_text[body_start:limit],
    ):
        token_start = body_start + token.start()
        token_end = body_start + token.end()
        kind = token.group(0)
        if kind == "if":
            colon = masked_text.find(":", token_end, limit)
            if colon >= 0:
                nested_if_depth += 1
            continue
        colon = masked_text.find(":", token_end, limit)
        if colon < 0:
            continue
        if kind == "elif":
            if nested_if_depth == 0:
                headers.append(("elif", token_start, colon + 1, token_end, colon))
            continue
        cursor = token_end
        while cursor < limit and masked_text[cursor] in " \t":
            cursor += 1
        if cursor >= limit or masked_text[cursor] != ":":
            continue
        if nested_if_depth:
            nested_if_depth -= 1
            continue
        headers.append(("else", token_start, cursor + 1, cursor, cursor))
        return tuple(headers)
    return ()


def _first_inline_match_node(text: str) -> _FormulaBranchNode | None:
    masked_text = _mask_formula_strings(text)
    for candidate in re.finditer(
        r"(?<![A-Za-z0-9_])match[ \t]+",
        masked_text,
    ):
        line_end = text.find("\n", candidate.start())
        if line_end < 0:
            line_end = len(text)
        colon = masked_text.find(":", candidate.end(), line_end)
        if colon < 0 or not text[colon + 1 : line_end].strip():
            continue
        end = _formula_expression_end(
            text,
            body_start=colon + 1,
            branch_start=candidate.start(),
            limit=line_end,
            stop_at_semicolon=False,
        )
        patterns: list[str] = []
        choices: list[str] = []
        for arm in _split_formula_arms(text[colon + 1 : end]):
            pattern, separator, body = arm.partition("=>")
            if not separator or not pattern.strip() or not body.strip():
                patterns = []
                choices = []
                break
            patterns.append(pattern.strip())
            choices.append(body)
        if not patterns:
            continue
        return _FormulaBranchNode(
            candidate.start(),
            end,
            "match",
            (text[candidate.end() : colon].strip(),),
            tuple(patterns),
            tuple(choices),
        )
    return None


def _split_formula_arms(text: str) -> tuple[str, ...]:
    arms: list[str] = []
    start = 0
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, character in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "([{":
            stack.append(character)
        elif character in pairs and stack and stack[-1] == pairs[character]:
            stack.pop()
        elif character == ";" and not stack:
            arms.append(text[start:index])
            start = index + 1
    arms.append(text[start:])
    return tuple(arms)


def _formula_expression_end(
    text: str,
    *,
    body_start: int,
    branch_start: int,
    limit: int,
    stop_at_semicolon: bool = True,
) -> int:
    initial_stack = _formula_bracket_stack(text[:branch_start])
    stack = list(initial_stack)
    quote: str | None = None
    escaped = False
    pairs = {")": "(", "]": "[", "}": "{"}
    for index in range(body_start, limit):
        character = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "([{":
            stack.append(character)
        elif character in pairs:
            if len(stack) <= len(initial_stack):
                return index
            if stack[-1] == pairs[character]:
                stack.pop()
        elif character == "," and len(stack) == len(initial_stack):
            return index
        elif (
            stop_at_semicolon
            and character == ";"
            and len(stack) == len(initial_stack)
        ):
            return index
    return limit


def _formula_bracket_stack(text: str) -> tuple[str, ...]:
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    pairs = {")": "(", "]": "[", "}": "{"}
    for character in text:
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "([{":
            stack.append(character)
        elif character in pairs and stack and stack[-1] == pairs[character]:
            stack.pop()
    return tuple(stack)


def _select_formula_branch(
    node: _FormulaBranchNode,
    *,
    environment: dict[str, Any],
) -> tuple[int, tuple[str, ...]] | None:
    if node.kind == "if":
        evaluated: list[str] = []
        for index, condition in enumerate(node.selectors):
            evaluated.append(condition)
            value = _evaluate_formula_selector(condition, environment)
            if not isinstance(value, bool):
                return None
            if value:
                return index, tuple(evaluated)
        if len(node.choices) > len(node.selectors):
            return len(node.selectors), tuple(evaluated)
        return None

    selector = _evaluate_formula_selector(node.selectors[0], environment)
    if selector is _UNRESOLVED_CONDITION_VALUE:
        return None
    if not node.patterns:
        return None
    for index, pattern in enumerate(node.patterns[:-1]):
        expected = _match_arm_value(pattern.strip())
        if expected is _UNRESOLVED_CONDITION_VALUE:
            return None
        if _same_formula_value(selector, expected):
            return index, node.selectors
    return len(node.patterns) - 1, node.selectors


def _evaluate_formula_selector(
    selector: str,
    environment: dict[str, Any],
) -> Any:
    try:
        expression = ast.parse(selector, mode="eval").body
    except SyntaxError:
        return _UNRESOLVED_CONDITION_VALUE
    return _evaluate_condition_expression(expression, environment)


def _rule_text_has_branching_formula(formula_text: str) -> bool:
    return bool(
        re.search(
            r"(?<![A-Za-z0-9_])(?:if|elif|else|match)(?![A-Za-z0-9_])",
            _mask_formula_strings(formula_text),
        )
    )


def _formula_execution_outcome(execution: _FormulaExecution) -> str:
    return "/".join(
        f"{step.kind}:{step.choice}"
        for step in execution.trace
    )


def _formula_execution_leaf_is_computational(
    execution: _FormulaExecution,
) -> bool:
    leaf = execution.leaf.strip()
    with contextlib.suppress(SyntaxError, ValueError):
        literal = ast.literal_eval(leaf)
        if literal is False or (
            isinstance(literal, (int, float, complex))
            and not isinstance(literal, bool)
            and literal == 0
        ):
            return False
    return not bool(
        re.fullmatch(
            r"(?:true|false|holds|not_holds|null|none)",
            leaf,
            flags=re.IGNORECASE,
        )
    )


def _formula_execution_references_names(
    execution: _FormulaExecution,
    names: set[str],
    *,
    selectors_only: bool = False,
) -> bool:
    texts = [
        selector
        for step in execution.trace
        for selector in step.selectors
    ]
    if not selectors_only:
        texts.append(execution.leaf)
    return any(
        set(_FORMULA_IDENTIFIER.findall(text)) & names
        for text in texts
    )


def _same_formula_value(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _mask_formula_strings(text: str) -> str:
    """Blank quoted content while preserving offsets and newlines."""

    masked = list(text)
    quote: str | None = None
    escaped = False
    for index, character in enumerate(text):
        if quote is None:
            if character in {'"', "'"}:
                quote = character
                masked[index] = " "
            continue
        if character != "\n":
            masked[index] = " "
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == quote:
            quote = None
    return "".join(masked)


def _case_formula_branch_outcome(
    rule: dict[str, Any],
    case: dict[str, Any],
) -> str | None:
    """Return the complete reachable branch path for one test case."""

    execution = _case_formula_execution(rule, case)
    if (
        execution is None
        or not execution.trace
        or not _formula_execution_leaf_is_computational(execution)
    ):
        return None
    return _formula_execution_outcome(execution)


def _match_arm_value(value: str) -> Any:
    if value == "_":
        return value
    lowered = value.lower()
    if lowered in {"true", "holds"}:
        return True
    if lowered in {"false", "not_holds"}:
        return False
    with contextlib.suppress(SyntaxError, ValueError):
        return ast.literal_eval(value)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        return value
    return _UNRESOLVED_CONDITION_VALUE


def _evaluate_condition_expression(
    expression: ast.expr,
    environment: dict[str, Any],
) -> Any:
    """Evaluate a small side-effect-free subset of RuleSpec conditions."""

    if isinstance(expression, ast.Constant):
        return expression.value
    if isinstance(expression, ast.Name):
        lowered = expression.id.lower()
        if lowered in {"true", "holds"}:
            return True
        if lowered in {"false", "not_holds"}:
            return False
        return environment.get(expression.id, _UNRESOLVED_CONDITION_VALUE)
    if isinstance(expression, ast.BoolOp):
        if isinstance(expression.op, ast.And):
            for item in expression.values:
                value = _evaluate_condition_expression(item, environment)
                if value is _UNRESOLVED_CONDITION_VALUE:
                    return value
                if not bool(value):
                    return False
            return True
        if isinstance(expression.op, ast.Or):
            for item in expression.values:
                value = _evaluate_condition_expression(item, environment)
                if value is _UNRESOLVED_CONDITION_VALUE:
                    return value
                if bool(value):
                    return True
            return False
    if isinstance(expression, ast.UnaryOp):
        value = _evaluate_condition_expression(expression.operand, environment)
        if value is _UNRESOLVED_CONDITION_VALUE:
            return value
        if isinstance(expression.op, ast.Not):
            return not bool(value)
        if isinstance(expression.op, ast.USub) and isinstance(value, (int, float)):
            return -value
    if isinstance(expression, ast.BinOp):
        left = _evaluate_condition_expression(expression.left, environment)
        right = _evaluate_condition_expression(expression.right, environment)
        if (
            left is _UNRESOLVED_CONDITION_VALUE
            or right is _UNRESOLVED_CONDITION_VALUE
        ):
            return _UNRESOLVED_CONDITION_VALUE
        with contextlib.suppress(ArithmeticError, TypeError, ValueError):
            if isinstance(expression.op, ast.Add):
                return left + right
            if isinstance(expression.op, ast.Sub):
                return left - right
            if isinstance(expression.op, ast.Mult):
                return left * right
            if isinstance(expression.op, ast.Div):
                return left / right
            if isinstance(expression.op, ast.Mod):
                return left % right
        return _UNRESOLVED_CONDITION_VALUE
    if isinstance(expression, ast.Compare):
        left = _evaluate_condition_expression(expression.left, environment)
        if left is _UNRESOLVED_CONDITION_VALUE:
            return left
        for operator, comparator in zip(expression.ops, expression.comparators):
            right = _evaluate_condition_expression(comparator, environment)
            if right is _UNRESOLVED_CONDITION_VALUE:
                return right
            try:
                if isinstance(operator, ast.Eq):
                    if type(left) is not type(right):
                        return _UNRESOLVED_CONDITION_VALUE
                    matched = left == right
                elif isinstance(operator, ast.NotEq):
                    if type(left) is not type(right):
                        return _UNRESOLVED_CONDITION_VALUE
                    matched = left != right
                elif isinstance(operator, ast.Lt):
                    matched = left < right
                elif isinstance(operator, ast.LtE):
                    matched = left <= right
                elif isinstance(operator, ast.Gt):
                    matched = left > right
                elif isinstance(operator, ast.GtE):
                    matched = left >= right
                elif isinstance(operator, ast.In):
                    matched = left in right
                elif isinstance(operator, ast.NotIn):
                    matched = left not in right
                else:
                    return _UNRESOLVED_CONDITION_VALUE
                if not matched:
                    return False
            except (TypeError, ValueError):
                return _UNRESOLVED_CONDITION_VALUE
            left = right
        return True
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id in {"holds", "not_holds"}
        and len(expression.args) == 1
        and not expression.keywords
    ):
        value = _evaluate_condition_expression(expression.args[0], environment)
        if value is _UNRESOLVED_CONDITION_VALUE:
            return value
        return bool(value) if expression.func.id == "holds" else not bool(value)
    return _UNRESOLVED_CONDITION_VALUE


def _branch_boundary_has_test_evidence(
    branch: SourceStructureBranch,
    boundary: NumericOccurrenceLike,
    *,
    principal_rules: dict[str, dict[str, Any]],
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    numeric_value_is_grounded: NumericGroundingPredicate,
    formula_environment: dict[str, Any] | None = None,
) -> bool:
    for rule_name in _rules_covering_branch(branch, principal_rule_paths):
        rule = principal_rules[rule_name]
        selector_names = _rule_numeric_selector_names(rule)
        if not selector_names:
            continue
        for case in asserted_by_rule.get(rule_name, ()):
            execution = _case_formula_execution(
                rule,
                case,
                formula_environment=formula_environment,
            )
            if (
                execution is not None
                and _formula_execution_references_names(
                    execution,
                    selector_names,
                )
                and any(
                numeric_value_is_grounded(value, (boundary,))
                for value in _case_numeric_selector_values(case, selector_names)
                )
            ):
                return True
    return False


def _rules_covering_branch(
    branch: SourceStructureBranch,
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
) -> tuple[str, ...]:
    return tuple(
        name
        for name, paths in principal_rule_paths.items()
        if branch.path in paths
    )


def _formula_branch_interval(
    branch: SourceStructureBranch,
    *,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> _NumericInterval | None:
    first_line = branch.text.splitlines()[0] if branch.text.splitlines() else ""
    range_text = first_line.split(":", 1)[0]
    range_text = _NUMBER_MARKER.sub("", range_text).strip()
    return _formula_interval_from_text(
        range_text,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )


def _formula_interval_from_text(
    text: str,
    *,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> _NumericInterval | None:
    lowered = text.lower()
    keyword = re.search(
        r"\b(?:zwischen|between|von|from|unter|less\s+than|below|"
        r"bis|up\s+to|höchstens|nicht\s+mehr\s+als|"
        r"at\s+most|über|more\s+than|above|ab|at\s+least|mindestens)\b",
        lowered,
    )
    if keyword is None:
        return None
    range_text = text[keyword.start() :]
    if re.match(
        r"ab\s+(?:dem\s+)?(?:veranlagungszeitraum|tax\s+year|calendar\s+year)\b",
        range_text,
        flags=re.IGNORECASE,
    ):
        return None
    occurrences = tuple(extract_numeric_occurrences(range_text))
    if not occurrences:
        return None
    lowered_range = range_text.lower()
    if re.match(r"(?:zwischen|between)\b", lowered_range) and re.search(
        r"\b(?:und|and)\b",
        lowered_range,
    ):
        if len(occurrences) < 2:
            return None
        return _NumericInterval(occurrences[0], True, occurrences[1], True)
    if re.match(r"(?:von|from)\b", lowered_range) and re.search(
        r"\b(?:bis|to|through)\b",
        lowered_range,
    ):
        if len(occurrences) < 2:
            return None
        return _NumericInterval(occurrences[0], True, occurrences[1], True)
    if re.match(r"(?:unter|less\s+than|below)\b", lowered_range):
        return _NumericInterval(None, False, occurrences[0], False)
    if re.match(
        r"(?:bis|up\s+to|höchstens|nicht\s+mehr\s+als|at\s+most)\b",
        lowered_range,
    ):
        return _NumericInterval(None, False, occurrences[0], True)
    if re.match(r"(?:über|more\s+than|above)\b", lowered_range):
        return _NumericInterval(occurrences[0], False, None, False)
    if re.match(
        r"(?:von|ab|from|at\s+least|mindestens)\b",
        lowered_range,
    ):
        return _NumericInterval(occurrences[0], True, None, False)
    return None


def _interval_contains(
    interval: _NumericInterval,
    value: float,
) -> bool:
    lower = interval.lower.value if interval.lower is not None else None
    upper = interval.upper.value if interval.upper is not None else None
    if lower is not None and (
        value < lower
        or (not interval.lower_inclusive and math.isclose(value, lower))
    ):
        return False
    if upper is not None and (
        value > upper
        or (not interval.upper_inclusive and math.isclose(value, upper))
    ):
        return False
    return True


def _rule_numeric_selector_names(rule: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    formula_text = _rule_formula_text(rule)
    for match in re.finditer(
        r"\b(?P<left>[A-Za-z_][A-Za-z0-9_]*)\s*"
        r"(?:<=|>=|<|>)\s*"
        r"(?P<right>[A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?)",
        formula_text,
    ):
        names.add(match.group("left"))
        right = match.group("right")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", right):
            names.add(right)
    if not names:
        names.update(
            identifier
            for identifier in _FORMULA_IDENTIFIER.findall(formula_text)
            if identifier.lower()
            not in {
                "if",
                "else",
                "and",
                "or",
                "not",
                "floor",
                "ceil",
                "min",
                "max",
                "true",
                "false",
                "holds",
                "not_holds",
            }
        )
    return names


def _case_numeric_selector_values(
    case: dict[str, Any],
    selector_names: set[str],
) -> tuple[float, ...]:
    inputs = case.get("input")
    if not isinstance(inputs, dict):
        return ()
    values: list[float] = []
    for key, value in inputs.items():
        if not (_input_key_names(key) & selector_names):
            continue
        values.extend(_numeric_test_input_values(value))
    return tuple(values)


def _input_key_names(key: object) -> set[str]:
    text = str(key).strip()
    fragment = text.rsplit("#", 1)[-1]
    return {
        text,
        fragment,
        fragment.rsplit(".", 1)[-1],
        fragment.rsplit("/", 1)[-1],
    }


def _source_exception_branches(
    source_text: str,
    *,
    branches: Sequence[SourceStructureBranch],
    active_branches: Sequence[SourceStructureBranch],
    deferred_paths: set[tuple[str, ...]],
) -> tuple[SourceStructureBranch, ...]:
    obligations: list[SourceStructureBranch] = []
    for match in _EXCEPTION_LANGUAGE.finditer(source_text):
        if _span_is_deferred(
            match.start(),
            match.end(),
            branches=branches,
            deferred_paths=deferred_paths,
        ):
            continue
        owner = _most_specific_containing_branch(
            match.start(),
            match.end(),
            branches=active_branches,
        )
        obligations.append(
            owner
            or SourceStructureBranch(
                (),
                "source-unit",
                "source unit",
                source_text,
                0,
                len(source_text),
            )
        )
    return tuple(obligations)


def _source_rounding_obligations(
    source_text: str,
    *,
    branches: Sequence[SourceStructureBranch],
    active_branches: Sequence[SourceStructureBranch],
    deferred_paths: set[tuple[str, ...]],
) -> tuple[tuple[SourceStructureBranch, str], ...]:
    obligations: list[tuple[SourceStructureBranch, str]] = []
    for match in _ROUNDING_LANGUAGE.finditer(source_text):
        if _span_is_deferred(
            match.start(),
            match.end(),
            branches=branches,
            deferred_paths=deferred_paths,
        ):
            continue
        owner = _most_specific_containing_branch(
            match.start(),
            match.end(),
            branches=active_branches,
        ) or SourceStructureBranch(
            (),
            "source-unit",
            "source unit",
            source_text,
            0,
            len(source_text),
        )
        matched_text = match.group(0)
        if _NEAREST_ROUNDING_LANGUAGE.search(matched_text):
            direction = "nearest"
        elif _UP_ROUNDING_LANGUAGE.search(matched_text):
            direction = "upward"
        else:
            direction = "downward"
        obligation_branch = SourceStructureBranch(
            owner.path,
            "rounding-clause",
            owner.label,
            matched_text,
            match.start(),
            match.end(),
        )
        obligations.append((obligation_branch, direction))
    return tuple(obligations)


def _unwitnessed_exception_branches(
    exception_branches: Sequence[SourceStructureBranch],
    *,
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
    toggled_exception_selectors: set[tuple[str, str]],
) -> tuple[SourceStructureBranch, ...]:
    available_witnesses = set(toggled_exception_selectors)
    missing: list[SourceStructureBranch] = []
    for branch in sorted(
        exception_branches,
        key=lambda item: len(_exception_witnesses_for_branch(
            item,
            principal_rule_paths=principal_rule_paths,
            toggled_exception_selectors=available_witnesses,
        )),
    ):
        candidates = _exception_witnesses_for_branch(
            branch,
            principal_rule_paths=principal_rule_paths,
            toggled_exception_selectors=available_witnesses,
        )
        if not candidates:
            missing.append(branch)
            continue
        available_witnesses.remove(sorted(candidates)[0])
    return tuple(missing)


def _exception_witnesses_for_branch(
    branch: SourceStructureBranch,
    *,
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
    toggled_exception_selectors: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    affecting_rules = {
        rule_name
        for rule_name, paths in principal_rule_paths.items()
        if not branch.path
        or any(
            path == branch.path[: len(path)]
            for path in paths
            if path
        )
    }
    return {
        witness
        for witness in toggled_exception_selectors
        if witness[0] in affecting_rules
    }


def _toggled_formula_boolean_selectors(
    principal_rules: dict[str, dict[str, Any]],
    *,
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    formula_environment: dict[str, Any],
) -> set[tuple[str, str]]:
    toggled: set[tuple[str, str]] = set()
    for rule_name, rule in principal_rules.items():
        selector_names = _rule_exception_selector_names(rule)
        if not selector_names:
            continue
        cases = asserted_by_rule.get(rule_name, ())
        for selector_name in selector_names:
            reachable_cases = [
                case
                for case in cases
                if (
                    (
                        execution := _case_formula_execution(
                            rule,
                            case,
                            formula_environment=formula_environment,
                        )
                    )
                    is not None
                    and _formula_execution_references_names(
                        execution,
                        {selector_name},
                        selectors_only=True,
                    )
                )
            ]
            for key in _paired_boolean_toggle_keys(reachable_cases):
                if selector_name in _input_key_names(key):
                    toggled.add((rule_name, selector_name))
    return toggled


def _rule_exception_selector_names(rule: dict[str, Any]) -> set[str]:
    """Return selectors attached to an excluding conditional branch."""

    formula_text = _rule_formula_text(rule)
    names: set[str] = set()
    for condition, true_body, false_body in _formula_condition_blocks(formula_text):
        condition_names = {
            identifier
            for identifier in re.findall(
                r"\b[A-Za-z_][A-Za-z0-9_]*\b",
                condition,
            )
            if identifier.lower()
            not in {"and", "or", "not", "true", "false", "holds", "not_holds"}
        }
        if (
            _formula_branch_is_excluding(true_body)
            or _formula_branch_is_excluding(false_body)
            or any(_exception_semantic_identifier(name) for name in condition_names)
        ):
            names.update(condition_names)
    return names


def _formula_condition_blocks(
    formula_text: str,
) -> Iterable[tuple[str, str, str]]:
    lines = formula_text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(
            r"^(?P<indent>[ \t]*)if\s+(?P<condition>[^:\n]+):\s*$",
            line,
        )
        if match is None:
            continue
        indent = _formula_indent_width(match.group("indent"))
        true_lines: list[str] = []
        false_lines: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            candidate = lines[cursor]
            stripped = candidate.strip()
            candidate_indent = _formula_indent_width(
                candidate[: len(candidate) - len(candidate.lstrip())]
            )
            if stripped and candidate_indent <= indent:
                break
            true_lines.append(candidate)
            cursor += 1
        if (
            cursor < len(lines)
            and _formula_indent_width(
                lines[cursor][
                    : len(lines[cursor]) - len(lines[cursor].lstrip())
                ]
            )
            == indent
            and re.match(r"^[ \t]*else:\s*$", lines[cursor])
        ):
            cursor += 1
            while cursor < len(lines):
                candidate = lines[cursor]
                stripped = candidate.strip()
                candidate_indent = _formula_indent_width(
                    candidate[: len(candidate) - len(candidate.lstrip())]
                )
                if stripped and candidate_indent <= indent:
                    break
                false_lines.append(candidate)
                cursor += 1
        yield (
            match.group("condition"),
            "\n".join(true_lines),
            "\n".join(false_lines),
        )


def _formula_indent_width(value: str) -> int:
    return len(value.expandtabs(4))


def _formula_branch_is_excluding(body: str) -> bool:
    first_statement = next(
        (
            line.strip()
            for line in body.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ),
        "",
    )
    return bool(
        re.fullmatch(
            r"(?:return\s+)?(?:0(?:\.0+)?|false|not_holds)",
            first_statement,
            flags=re.IGNORECASE,
        )
    )


def _exception_semantic_identifier(identifier: str) -> bool:
    return bool(
        re.search(
            r"(?:exception|exempt|exclud|disqual|inelig|eligib|remarri|"
            r"barred|blocking|waiver)",
            identifier,
            flags=re.IGNORECASE,
        )
    )


def _rule_implements_rounding(rule: dict[str, Any], direction: str) -> bool:
    formula_text = _rule_formula_text(rule)
    if direction == "nearest":
        return bool(
            re.search(r"\bfloor\s*\(", formula_text)
            and re.search(r"\+\s*0?\.5\b", formula_text)
        )
    function_name = "ceil" if direction == "upward" else "floor"
    return re.search(rf"\b{function_name}\s*\(", formula_text) is not None


def _fractional_rounding_case_witnesses(
    rule_name: str,
    rule: dict[str, Any],
    *,
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    direction: str,
    formula_environment: dict[str, Any],
) -> set[tuple[str, int]]:
    evidence: set[tuple[str, int]] = set()
    functions = {
        "nearest" if direction == "nearest" else (
            "ceil" if direction == "upward" else "floor"
        )
    }
    rounded_operand_identifiers = _rounding_operand_identifier_names(
        rule,
        functions=functions,
    )
    for case in asserted_by_rule.get(rule_name, ()):
        inputs = case.get("input")
        if not isinstance(inputs, dict):
            continue
        execution = _case_formula_execution(
            rule,
            case,
            formula_environment=formula_environment,
        )
        if execution is None or not _formula_execution_implements_rounding(
            execution,
            direction,
        ):
            continue
        if any(
            _input_key_names(key) & rounded_operand_identifiers
            and any(
                not float(value).is_integer()
                for value in _numeric_test_input_values(input_value)
            )
            for key, input_value in inputs.items()
        ):
            evidence.add((rule_name, id(case)))
    return evidence


def _formula_execution_implements_rounding(
    execution: _FormulaExecution,
    direction: str,
) -> bool:
    if direction == "nearest":
        return bool(
            re.search(r"\bfloor\s*\(", execution.leaf)
            and re.search(r"\+\s*0?\.5\b", execution.leaf)
        )
    function_name = "ceil" if direction == "upward" else "floor"
    return re.search(
        rf"\b{function_name}\s*\(",
        execution.leaf,
    ) is not None


def _rounding_operand_identifier_names(
    rule: dict[str, Any],
    *,
    functions: set[str],
) -> set[str]:
    names: set[str] = set()
    formula_text = _rule_formula_text(rule)
    function_names = {"floor", "ceil"} & functions
    if "nearest" in functions:
        function_names.add("floor")
    for function_name in function_names:
        for operand in _balanced_call_operands(formula_text, function_name):
            if (
                functions == {"nearest"}
                and not re.search(r"\+\s*0?\.5\b", operand)
            ):
                continue
            names.update(
                identifier
                for identifier in _FORMULA_IDENTIFIER.findall(operand)
                if identifier.lower()
                not in {
                    "if",
                    "else",
                    "and",
                    "or",
                    "not",
                    "floor",
                    "ceil",
                    "min",
                    "max",
                    "true",
                    "false",
                    "holds",
                    "not_holds",
                }
            )
    return names


def _balanced_call_operands(text: str, function_name: str) -> Iterable[str]:
    for match in re.finditer(rf"\b{re.escape(function_name)}\s*\(", text):
        open_index = match.end() - 1
        depth = 1
        cursor = open_index + 1
        while cursor < len(text) and depth:
            character = text[cursor]
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
            cursor += 1
        if depth == 0:
            yield text[open_index + 1 : cursor - 1]


def _source_boundary_obligations(
    branches: Sequence[SourceStructureBranch],
    *,
    narrative_formula_branches: Sequence[SourceStructureBranch] = (),
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> tuple[tuple[SourceStructureBranch, NumericOccurrenceLike], ...]:
    obligations: list[tuple[SourceStructureBranch, NumericOccurrenceLike]] = []
    for branch in branches:
        if branch.kind not in {
            "paragraph",
            "number",
            "letter",
            "sentence",
            "source-unit",
        }:
            continue
        first_line = branch.text.splitlines()[0] if branch.text.splitlines() else ""
        prefix = first_line.split(":", 1)[0]
        if not re.search(
            r"\b(?:zwischen|between|bis|von|ab|unter|über|from|to|through|"
            r"less\s+than|more\s+than|at\s+least|up\s+to|above|below|"
            r"nicht\s+mehr\s+als)\b",
            prefix,
            flags=re.IGNORECASE,
        ):
            continue
        prefix = authoritative_numeric_recall_text(prefix)
        interval = _formula_interval_from_text(
            prefix,
            extract_numeric_occurrences=extract_numeric_occurrences,
        )
        if interval is None:
            continue
        obligations.extend(
            (branch, boundary)
            for boundary in (interval.lower, interval.upper)
            if boundary is not None
        )
    for branch in narrative_formula_branches:
        interval = _formula_branch_interval(
            branch,
            extract_numeric_occurrences=extract_numeric_occurrences,
        )
        if interval is None:
            continue
        obligations.extend(
            (branch, boundary)
            for boundary in (interval.lower, interval.upper)
            if boundary is not None
        )
    return tuple(
        {
            (
                branch.path,
                float(occurrence.value),
                occurrence.has_rate_context,
                occurrence.source_value,
                occurrence.requires_rate_context,
            ): (branch, occurrence)
            for branch, occurrence in obligations
        }.values()
    )


def _numeric_test_input_values(value: Any) -> tuple[float, ...]:
    values: list[float] = []
    if isinstance(value, bool):
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, dict):
        for item in value.values():
            values.extend(_numeric_test_input_values(item))
    elif isinstance(value, list):
        for item in value:
            values.extend(_numeric_test_input_values(item))
    return tuple(values)


def _paired_boolean_toggle_keys(cases: Iterable[dict[str, Any]]) -> set[str]:
    flattened = [
        (
            case,
            {
                str(key): _boolean_value(value)
                for key, value in (case.get("input") or {}).items()
                if _boolean_value(value) is not None
            },
        )
        for case in cases
        if isinstance(case.get("input"), dict)
    ]
    toggled: set[str] = set()
    for left_index, (left_case, left_values) in enumerate(flattened):
        for right_case, right_values in flattened[left_index + 1 :]:
            if set(left_values) != set(right_values):
                continue
            changed = [
                key for key in left_values if left_values[key] != right_values[key]
            ]
            if len(changed) != 1:
                continue
            if _non_boolean_inputs(left_case) != _non_boolean_inputs(right_case):
                continue
            toggled.add(changed[0])
    return toggled


def _boolean_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"holds", "true"}:
        return True
    if normalized in {"not_holds", "false"}:
        return False
    return None


def _non_boolean_inputs(case: dict[str, Any]) -> dict[str, Any]:
    inputs = case.get("input")
    if not isinstance(inputs, dict):
        return {}
    return {
        str(key): value
        for key, value in inputs.items()
        if _boolean_value(value) is None
    }


def _rule_formula_text(rule: dict[str, Any]) -> str:
    versions = rule.get("versions")
    if not isinstance(versions, list):
        return ""
    return "\n".join(
        str(version.get("formula"))
        for version in versions
        if isinstance(version, dict) and version.get("formula") is not None
    )


def _unambiguous_rule_formula_text(rule: dict[str, Any]) -> str | None:
    versions = rule.get("versions")
    if not isinstance(versions, list):
        return None
    formulas = tuple(
        dict.fromkeys(
            str(version["formula"])
            for version in versions
            if isinstance(version, dict) and version.get("formula") is not None
        )
    )
    return formulas[0] if len(formulas) == 1 else None


def _constant_rule_environment(payload: dict[str, Any]) -> dict[str, Any]:
    """Return unambiguous literal rule values usable by condition evaluation."""

    environment: dict[str, Any] = {}
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return environment
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        name = str(rule.get("name") or "").strip()
        versions = rule.get("versions")
        if not name or not isinstance(versions, list):
            continue
        values: list[Any] = []
        for version in versions:
            if not isinstance(version, dict) or "formula" not in version:
                continue
            formula = version["formula"]
            with contextlib.suppress(SyntaxError, ValueError):
                value = ast.literal_eval(str(formula))
                if isinstance(value, (str, int, float, bool)) and not isinstance(
                    value,
                    complex,
                ):
                    values.append(value)
        if values and all(
            type(value) is type(values[0]) and value == values[0]
            for value in values[1:]
        ):
            environment[name] = values[0]
    return environment


def _collapse_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
