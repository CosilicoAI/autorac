"""Opt-in authoritative source-unit completeness accounting.

This module deliberately sits beside, rather than inside, the numeric
extractor.  It consumes the extractor and scalar-inventory interfaces supplied
by :mod:`validator_pipeline`, which keeps locale-tokenizer work independent
from the source-unit admission policy implemented here.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import yaml

NumericOccurrenceExtractor = Callable[[str], Sequence[float]]
NamedScalarExtractor = Callable[[str], Sequence[Any]]
NumericGroundingPredicate = Callable[[float, set[float]], bool]


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


_PARAGRAPH_MARKER = re.compile(
    r"(?m)^(?P<marker>\((?P<label>\d+[a-z]?|[a-z])\))(?=\s|bis\b)",
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
_EDITORIAL_OMISSION = re.compile(
    r"\b(?:weggefallen|aufgehoben|repealed|omitted)\b|\.{3,}",
    flags=re.IGNORECASE,
)
_ARITHMETIC_EXPRESSION = re.compile(
    r"(?:\d|[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß]*)"
    r"[ \t]*(?:[+*/=×·•∗∙]|(?<!\w)[−–-](?!\w))[ \t]*"
    r"(?:\d|[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß]*)"
)
_COMPUTATION_LANGUAGE = re.compile(
    r"\b(?:"
    r"bemisst\s+sich\s+nach|berechn(?:et|en|ung)|ergibt\s+sich|"
    r"zweifache|hälfte|zehntausendstel|übersteigenden\s+teils|"
    r"splitting-verfahren|verfahren\s+nach\s+absatz|"
    r"calculated|computed|computation|multiplied|divided|"
    r"sum\s+of|difference\s+between|product\s+of|twice|half\s+of|"
    r"percentage\s+of|in\s+excess\s+of"
    r")\b",
    flags=re.IGNORECASE,
)
_ROUNDING_LANGUAGE = re.compile(
    r"\b(?:"
    r"abgerundet(?:en|er|es)?|abzurunden|aufgerundet(?:en|er|es)?|"
    r"aufzurunden|kaufmännisch(?:\s+zu)?\s+runden|"
    r"round(?:ed|ing)?(?:\s+(?:down|up|to\s+the\s+nearest))?"
    r")\b",
    flags=re.IGNORECASE,
)
_DOWN_ROUNDING_LANGUAGE = re.compile(
    r"\b(?:abgerundet(?:en|er|es)?|abzurunden|round(?:ed|ing)?\s+down)\b",
    flags=re.IGNORECASE,
)
_UP_ROUNDING_LANGUAGE = re.compile(
    r"\b(?:aufgerundet(?:en|er|es)?|aufzurunden|round(?:ed|ing)?\s+up)\b",
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
    r"es\s+sei\s+denn|voraussetzung[^.;]{0,160}\bnicht\b"
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
    r"benötigt|abhängig|fehlt|nicht\s+codiert"
    r")\b",
    flags=re.IGNORECASE,
)
_PRECISE_NONOPERATIVE_REASON = re.compile(
    r"\b(?:non-operative|legislative\s+(?:intent|finding|purpose)|"
    r"administrative\s+workflow|weggefallen|aufgehoben)\b",
    flags=re.IGNORECASE,
)
_ABSATZ_REFERENCE = re.compile(
    r"\b(?:Absatz|Abs\.)\s*(?P<label>\d+[a-z]?)\b",
    flags=re.IGNORECASE,
)
_SATZ_REFERENCE = re.compile(
    r"\b(?:Satz|Sätze)\s*(?P<label>\d+)\b",
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
_ENGLISH_LEGAL_CITATION = re.compile(
    r"\b(?:sections?|secs?\.?|regulations?|paragraphs?)\s+"
    r"\d+(?:\.\d+)*(?:\s*(?:through|to|-|and|,)\s*\d+(?:\.\d+)*)*",
    flags=re.IGNORECASE,
)
_STRUCTURAL_REFERENCE = re.compile(
    r"\b(?:Absatz|Abs\.|Satz|Sätze|Nummer|Nr\.|Buchstabe|Buchst\.)"
    r"\s*\d*[a-z]?\b",
    flags=re.IGNORECASE,
)


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
        if _EDITORIAL_OMISSION.search(text):
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
            if _EDITORIAL_OMISSION.search(text):
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
                if _EDITORIAL_OMISSION.search(text):
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
            container = max(
                (
                    branch
                    for branch in branches
                    if branch.start <= start < branch.end
                    and branch.kind in {"paragraph", "number", "letter"}
                ),
                key=lambda branch: len(branch.path),
                default=None,
            )
            parent_path = container.path if container is not None else paragraph_path
            label = match.group("label")
            path = (*parent_path, f"satz-{label}")
            text = source_text[start:end].strip()
            if _EDITORIAL_OMISSION.search(text):
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


def source_states_explicit_computation(source_text: str) -> bool:
    """Return whether text states a computation rather than only a scalar."""

    return bool(
        _ARITHMETIC_EXPRESSION.search(source_text)
        or _COMPUTATION_LANGUAGE.search(source_text)
        or _ROUNDING_LANGUAGE.search(source_text)
    )


def analyze_complete_source_unit(
    content: str,
    authoritative_source_text: str,
    *,
    corpus_citation_path: str,
    test_cases: Sequence[object] | None,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    extract_named_scalars: NamedScalarExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
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
) -> CompleteSourceUnitAnalysis:
    branches = recognize_source_structure(source_text)
    all_covered_paths, principal_paths, principal_rules = _rule_coverage(
        payload,
        source_text=source_text,
        branches=branches,
        corpus_citation_path=corpus_citation_path,
    )
    deferred_paths, imprecise_deferrals = _deferred_coverage(
        payload,
        corpus_citation_path=corpus_citation_path,
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

    computation_branches = tuple(
        branch for branch in branches if source_states_explicit_computation(branch.text)
    )
    source_has_computation = source_states_explicit_computation(source_text)
    if source_has_computation:
        if computation_branches:
            for branch in computation_branches:
                if _path_covered(branch.path, principal_paths, deferred_paths):
                    continue
                issues.append(
                    "[complete-source-unit:formula-output] "
                    f"Explicit source computation in "
                    f"{_branch_citation(corpus_citation_path, branch)} has no "
                    "principal derived/relation output (`derived` or "
                    "`derived_relation`) and is not "
                    "precisely deferred; parameter-only representation is invalid."
                )
        elif not principal_rules and not deferred_paths:
            issues.append(
                "[complete-source-unit:formula-output] Explicit source computation "
                "has no principal derived/relation output (`derived` or "
                "`derived_relation`); "
                "parameter-only representation is invalid."
            )

    source_values = tuple(
        extract_numeric_occurrences(_numeric_recall_source_text(source_text))
    )
    named_values = tuple(
        float(item.value)
        for item in extract_named_scalars(content)
        if hasattr(item, "value")
    )
    covered_source_values = 0
    missing_source_values: list[float] = []
    for value in source_values:
        if any(
            numeric_value_is_grounded(named_value, {float(value)})
            for named_value in named_values
        ):
            covered_source_values += 1
        else:
            missing_source_values.append(float(value))
    for value in sorted(set(missing_source_values)):
        issues.append(
            "[complete-source-unit:numeric-recall] Authoritative corpus numeric "
            f"value {value:g} has no named scalar representation. "
            "`module.summary` is not consulted."
        )

    if source_has_computation and principal_rules:
        issues.extend(
            _companion_test_issues(
                principal_rules,
                branches=branches,
                source_text=source_text,
                deferred_paths=deferred_paths,
                test_cases=test_cases,
                extract_numeric_occurrences=extract_numeric_occurrences,
                numeric_value_is_grounded=numeric_value_is_grounded,
            )
        )

    return CompleteSourceUnitAnalysis(
        tuple(dict.fromkeys(issues)),
        branches,
        len(source_values),
        covered_source_values,
        len(missing_source_values),
    )


def _rule_coverage(
    payload: dict[str, Any],
    *,
    source_text: str,
    branches: Sequence[SourceStructureBranch],
    corpus_citation_path: str,
) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]], dict[str, dict[str, Any]]]:
    all_paths: set[tuple[str, ...]] = set()
    principal_paths: set[tuple[str, ...]] = set()
    principal_rules: dict[str, dict[str, Any]] = {}
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return all_paths, principal_paths, principal_rules

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        kind = str(rule.get("kind") or "").strip().lower()
        name = str(rule.get("name") or "").strip()
        paths = _paths_from_source_reference(
            str(rule.get("source") or ""),
            corpus_citation_path=corpus_citation_path,
        )
        for excerpt in _rule_source_excerpts(rule):
            excerpt_branch = _most_specific_excerpt_branch(excerpt, branches)
            if excerpt_branch is not None:
                paths.add(excerpt_branch.path)
        for path in paths:
            all_paths.update(_path_prefixes(path))
        if kind in {"derived", "derived_relation"} and name:
            principal_rules[name] = rule
            for path in paths:
                principal_paths.update(_path_prefixes(path))
    return all_paths, principal_paths, principal_rules


def _rule_source_excerpts(rule: dict[str, Any]) -> Iterable[str]:
    metadata = rule.get("metadata")
    proof = metadata.get("proof") if isinstance(metadata, dict) else None
    if not isinstance(proof, dict):
        proof = rule.get("proof")
    atoms = proof.get("atoms") if isinstance(proof, dict) else None
    if not isinstance(atoms, list):
        return ()
    excerpts: list[str] = []
    for atom in atoms:
        source = atom.get("source") if isinstance(atom, dict) else None
        excerpt = source.get("excerpt") if isinstance(source, dict) else None
        if isinstance(excerpt, str) and excerpt.strip():
            excerpts.append(excerpt.strip())
    return excerpts


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
    return max(candidates, key=lambda branch: len(branch.path), default=None)


def _paths_from_source_reference(
    source: str,
    *,
    corpus_citation_path: str,
) -> set[tuple[str, ...]]:
    value = source.strip()
    if not value:
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
            paths.add(tuple(component.lower() for component in components))

    keyword_components: list[str] = []
    if match := _ABSATZ_REFERENCE.search(value):
        keyword_components.append(match.group("label").lower())
    if match := _NUMMER_REFERENCE.search(value):
        keyword_components.append(match.group("label").lower())
    if match := _BUCHSTABE_REFERENCE.search(value):
        keyword_components.append(match.group("label").lower())
    if match := _SATZ_REFERENCE.search(value):
        keyword_components.append(f"satz-{match.group('label')}")
    if keyword_components:
        paths.add(tuple(keyword_components))

    for match in re.finditer(
        r"§\s*\d+[a-z]?(?P<suffix>(?:\([A-Za-z0-9-]+\))+)",
        value,
        flags=re.IGNORECASE,
    ):
        components = re.findall(r"\(([A-Za-z0-9-]+)\)", match.group("suffix"))
        if components:
            paths.add(tuple(component.lower() for component in components))
    return paths


def _deferred_coverage(
    payload: dict[str, Any],
    *,
    corpus_citation_path: str,
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
        path: tuple[str, ...] = ()
        if output_path.startswith(f"{base_target}/"):
            path = tuple(
                part.lower()
                for part in output_path[len(base_target) + 1 :].split("/")
                if part
            )
        if not path:
            continue
        reason = str(record.get("reason") or "").strip()
        blocked_by = record.get("blocked_by")
        exact_blockers = (
            isinstance(blocked_by, list)
            and bool(blocked_by)
            and all(
                isinstance(item, str)
                and re.fullmatch(
                    r"[a-z]{2}(?:-[a-z0-9-]+)?:"
                    r"[A-Za-z0-9_./-]+#[A-Za-z_][A-Za-z0-9_]*",
                    item.strip(),
                    flags=re.IGNORECASE,
                )
                for item in blocked_by
            )
        )
        reason_identifies_dependency = bool(
            _MISSING_DEPENDENCY_LANGUAGE.search(reason)
            and _PRECISE_DEFERRAL_DEPENDENCY.search(reason)
        )
        precise = (
            exact_blockers
            or reason_identifies_dependency
            or bool(_PRECISE_NONOPERATIVE_REASON.search(reason))
        )
        if precise:
            covered.add(path)
        else:
            issues.append(
                "[complete-source-unit:deferral] "
                f"`module.deferred_outputs[{index}]` identifies source branch "
                f"({path[0]}) (`{'/'.join(path)}`) but its deferral does not "
                "name an exact missing "
                "dependency/citation (or a precise non-operative reason)."
            )
    return covered, issues


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
    return {path[:index] for index in range(1, len(path) + 1)}


def _path_covered(
    path: tuple[str, ...],
    encoded_paths: set[tuple[str, ...]],
    deferred_paths: set[tuple[str, ...]],
) -> bool:
    if path in encoded_paths:
        return True
    return any(path[: len(deferred)] == deferred for deferred in deferred_paths)


def _branch_citation(
    corpus_citation_path: str,
    branch: SourceStructureBranch,
) -> str:
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


def _numeric_recall_source_text(source_text: str) -> str:
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


def _companion_test_issues(
    principal_rules: dict[str, dict[str, Any]],
    *,
    branches: Sequence[SourceStructureBranch],
    source_text: str,
    deferred_paths: set[tuple[str, ...]],
    test_cases: Sequence[object] | None,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> list[str]:
    issues: list[str] = []
    cases = [case for case in (test_cases or ()) if isinstance(case, dict)]
    if not cases:
        return [
            "[complete-source-unit:tests] Explicit source computations require "
            "a companion test suite covering branches, boundaries, exceptions, "
            "and rounding rules."
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
        if not any(
            branch.path[: len(deferred)] == deferred for deferred in deferred_paths
        )
    ]
    formula_leaf_count = sum(
        1
        for branch in active_branches
        if branch.kind == "number" and _number_branch_is_formula_leaf(branch.text)
    )
    if formula_leaf_count:
        most_assertions = max(
            (len(asserted_cases) for asserted_cases in asserted_by_rule.values()),
            default=0,
        )
        if most_assertions < formula_leaf_count:
            issues.append(
                "[complete-source-unit:tests] Source states "
                f"{formula_leaf_count} formula branches/list branches, but no principal "
                f"output is asserted in {formula_leaf_count} distinct cases."
            )

    boundaries = _source_boundary_values(
        active_branches,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    test_input_values = tuple(
        value
        for case in cases
        for value in _numeric_test_input_values(case.get("input"))
    )
    missing_boundaries = [
        boundary
        for boundary in boundaries
        if not any(
            numeric_value_is_grounded(test_value, {boundary})
            for test_value in test_input_values
        )
    ]
    if missing_boundaries:
        rendered = ", ".join(f"{value:g}" for value in sorted(set(missing_boundaries)))
        issues.append(
            "[complete-source-unit:tests] Companion tests do not exercise every "
            f"source-stated boundary input; missing: {rendered}."
        )

    active_text = _source_text_without_deferred_branches(
        source_text,
        branches=branches,
        deferred_paths=deferred_paths,
    )
    exception_count = len(_EXCEPTION_LANGUAGE.findall(active_text))
    if exception_count:
        toggled_boolean_keys = _paired_boolean_toggle_keys(
            case
            for asserted_cases in asserted_by_rule.values()
            for case in asserted_cases
        )
        if len(toggled_boolean_keys) < exception_count:
            issues.append(
                "[complete-source-unit:tests] Source-stated exceptions require "
                "paired positive/blocking cases that toggle each exception fact; "
                f"found {len(toggled_boolean_keys)} pair(s) for "
                f"{exception_count} exception obligation(s)."
            )

    rounding_count = len(_ROUNDING_LANGUAGE.findall(active_text))
    if rounding_count:
        formula_text = "\n".join(
            _rule_formula_text(rule) for rule in principal_rules.values()
        )
        if _DOWN_ROUNDING_LANGUAGE.search(active_text) and "floor(" not in formula_text:
            issues.append(
                "[complete-source-unit:tests] A source-stated downward-rounding "
                "rule is absent from the principal formula (`floor(...)`)."
            )
        if _UP_ROUNDING_LANGUAGE.search(active_text) and "ceil(" not in formula_text:
            issues.append(
                "[complete-source-unit:tests] A source-stated upward-rounding "
                "rule is absent from the principal formula (`ceil(...)`)."
            )
        if _NEAREST_ROUNDING_LANGUAGE.search(active_text) and not (
            "floor(" in formula_text and re.search(r"\+\s*0?\.5\b", formula_text)
        ):
            issues.append(
                "[complete-source-unit:tests] A source-stated nearest/commercial "
                "rounding rule is absent from the principal formula."
            )
        fractional_cases = sum(
            1
            for case in cases
            if any(
                not float(value).is_integer()
                for value in _numeric_test_input_values(case.get("input"))
            )
        )
        if fractional_cases < rounding_count:
            issues.append(
                "[complete-source-unit:tests] Companion tests do not demonstrate "
                f"all {rounding_count} source-stated rounding rule(s) with "
                "fractional input values."
            )
    return issues


def _source_text_without_deferred_branches(
    source_text: str,
    *,
    branches: Sequence[SourceStructureBranch],
    deferred_paths: set[tuple[str, ...]],
) -> str:
    """Remove precisely deferred spans without duplicating nested branch text."""

    deferred_ranges = sorted(
        (
            branch.start,
            branch.end,
        )
        for branch in branches
        if any(
            branch.path[: len(deferred)] == deferred for deferred in deferred_paths
        )
        and not any(
            ancestor.path != branch.path
            and ancestor.start <= branch.start
            and branch.end <= ancestor.end
            and any(
                ancestor.path[: len(deferred)] == deferred
                for deferred in deferred_paths
            )
            for ancestor in branches
        )
    )
    if not deferred_ranges:
        return source_text
    parts: list[str] = []
    cursor = 0
    for start, end in deferred_ranges:
        if start < cursor:
            continue
        parts.append(source_text[cursor:start])
        cursor = end
    parts.append(source_text[cursor:])
    return "".join(parts)


def _test_case_output_names(case: dict[str, Any]) -> set[str]:
    outputs = case.get("output")
    if not isinstance(outputs, dict):
        return set()
    names: set[str] = set()
    for key in outputs:
        text = str(key)
        names.add(text.rsplit("#", 1)[-1])
    return names


def _number_branch_is_formula_leaf(text: str) -> bool:
    first_line = text.splitlines()[0] if text.splitlines() else text
    return ":" in first_line and bool(
        _ARITHMETIC_EXPRESSION.search(first_line)
        or re.search(r":\s*-?\d", first_line)
    )


def _source_boundary_values(
    branches: Sequence[SourceStructureBranch],
    *,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> tuple[float, ...]:
    values: list[float] = []
    for branch in branches:
        if branch.kind != "number":
            continue
        first_line = branch.text.splitlines()[0] if branch.text.splitlines() else ""
        prefix = first_line.split(":", 1)[0]
        if not re.search(
            r"\b(?:bis|von|ab|unter|über|from|to|through|less\s+than|"
            r"more\s+than|at\s+least|up\s+to|above|below)\b",
            prefix,
            flags=re.IGNORECASE,
        ):
            continue
        prefix = _NUMBER_MARKER.sub("", prefix)
        values.extend(float(value) for value in extract_numeric_occurrences(prefix))
    return tuple(dict.fromkeys(values))


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


def _collapse_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
