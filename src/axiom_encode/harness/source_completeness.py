"""Opt-in authoritative source-unit completeness accounting.

This module deliberately sits beside, rather than inside, the numeric
extractor.  It consumes the extractor and scalar-inventory interfaces supplied
by :mod:`validator_pipeline`, which keeps locale-tokenizer work independent
from the source-unit admission policy implemented here.
"""

from __future__ import annotations

import contextlib
import math
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
    r"(?:\d|[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß]*)"
    r"[ \t]*(?:[+*/=×·•∗∙]|(?<!\w)[−–-](?!\w))[ \t]*"
    r"(?:\d|[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß]*)"
)
_COMPUTATION_LANGUAGE = re.compile(
    r"\b(?:"
    r"bemisst\s+sich\s+nach|berechn(?:et|en|ung)|ergibt\s+sich|"
    r"zweifache|hälfte|zehntausendstel|übersteigenden\s+teils|"
    r"(?:ein(?:e[nsrm]?)?\s+)?(?:drittel|viertel|fünftel|sechstel|"
    r"siebtel|achtel|neuntel|zehntel)\s+(?:des|der|von)|"
    r"summe\s+(?:aus|der|von)|unterschied|differenz|"
    r"geteilt\s+durch|durch\s+(?:\d+|[a-zäöüß]+)\s+geteilt|"
    r"multipliziert\s+mit|"
    r"(?:vermindert|erhöht|gekürzt|vermehrt)\s+um|"
    r"(?:abzüglich|zuzüglich)|prozent\s+(?:des|der|von)|"
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
_DECIMAL_PERCENT_IDENTIFIER = re.compile(
    r"(?<!\d)(?P<integer>\d{1,3})_(?P<fraction>\d{1,4})_percent\b",
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
                principal_rule_paths=principal_rule_paths,
                branches=branches,
                source_text=source_text,
                corpus_citation_path=corpus_citation_path,
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
        for path in paths:
            all_paths.update(_path_prefixes(path))
        if kind in {"derived", "derived_relation"} and name:
            principal_rules[name] = rule
            principal_rule_paths.setdefault(name, set()).update(paths)
            for path in paths:
                principal_paths.update(_path_prefixes(path))
    return all_paths, principal_paths, principal_rules, principal_rule_paths


def _rule_source_excerpts(rule: dict[str, Any]) -> Iterable[tuple[str, str]]:
    metadata = rule.get("metadata")
    proof = metadata.get("proof") if isinstance(metadata, dict) else None
    if not isinstance(proof, dict):
        proof = rule.get("proof")
    atoms = proof.get("atoms") if isinstance(proof, dict) else None
    if not isinstance(atoms, list):
        return ()
    excerpts: list[tuple[str, str]] = []
    for atom in atoms:
        source = atom.get("source") if isinstance(atom, dict) else None
        excerpt = source.get("excerpt") if isinstance(source, dict) else None
        citation_path = (
            str(source.get("corpus_citation_path") or "").strip()
            if isinstance(source, dict)
            else ""
        )
        if isinstance(excerpt, str) and excerpt.strip() and citation_path:
            excerpts.append((citation_path, excerpt.strip()))
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
                and item.strip().lower() != output.lower()
                for item in blocked_by
            )
        )
        reason_identifies_dependency = _reason_names_external_dependency(
            reason,
            deferred_path=path,
            corpus_citation_path=corpus_citation_path,
        )
        precise = (
            exact_blockers
            or reason_identifies_dependency
        )
        if precise:
            covered.add(path)
        else:
            issues.append(
                "[complete-source-unit:deferral] "
                f"`module.deferred_outputs[{index}]` identifies source branch "
                f"({path[0]}) (`{'/'.join(path)}`) but its deferral does not "
                "name an exact missing "
                "dependency/citation."
            )
    return covered, issues


def _reason_names_external_dependency(
    reason: str,
    *,
    deferred_path: tuple[str, ...],
    corpus_citation_path: str,
) -> bool:
    if not _MISSING_DEPENDENCY_LANGUAGE.search(reason):
        return False
    current_section = corpus_citation_path.rstrip("/").rsplit("/", 1)[-1].lower()
    for match in _PRECISE_DEFERRAL_DEPENDENCY.finditer(reason):
        dependency = match.group(0).strip()
        normalized = dependency.lower()
        if paragraph_match := _ABSATZ_REFERENCE.fullmatch(dependency):
            if (
                deferred_path
                and paragraph_match.group("label").lower() == deferred_path[0]
            ):
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
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    imported_contents: Sequence[str] = (),
    additional_values: Iterable[float] = (),
) -> tuple[float, ...]:
    """Collect numeric representations credited by strict recall accounting."""

    values = [float(value) for value in additional_values]
    for index, artifact_content in enumerate((content, *imported_contents)):
        values.extend(
            float(item.value)
            for item in extract_named_scalars(artifact_content)
            if hasattr(item, "value")
        )
        with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
            payload = yaml.safe_load(artifact_content)
            if not (
                isinstance(payload, dict)
                and payload.get("format") == "rulespec/v1"
            ):
                continue
            rules = payload.get("rules")
            if isinstance(rules, list):
                for rule in rules:
                    if not isinstance(rule, dict):
                        continue
                    values.extend(
                        _numeric_concept_values(
                            str(rule.get("name") or ""),
                            extract_numeric_occurrences=extract_numeric_occurrences,
                        )
                    )
                    versions = rule.get("versions")
                    if isinstance(versions, list):
                        for version in versions:
                            formula = (
                                version.get("formula")
                                if isinstance(version, dict)
                                else None
                            )
                            if isinstance(formula, str):
                                for identifier in set(
                                    _FORMULA_IDENTIFIER.findall(formula)
                                ):
                                    values.extend(
                                        _numeric_concept_values(
                                            identifier,
                                            extract_numeric_occurrences=(
                                                extract_numeric_occurrences
                                            ),
                                        )
                                    )
                    verification = rule.get("verification")
                    if isinstance(verification, dict):
                        values.extend(
                            _verification_numeric_values(
                                verification.get("values"),
                                extract_numeric_occurrences=(
                                    extract_numeric_occurrences
                                ),
                            )
                        )
            if index == 0:
                values.extend(
                    _deferred_numeric_values(
                        payload,
                        extract_numeric_occurrences=extract_numeric_occurrences,
                    )
                )
    return tuple(values)


def _numeric_concept_values(
    text: str,
    *,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> tuple[float, ...]:
    normalized = _DECIMAL_PERCENT_IDENTIFIER.sub(
        lambda match: (
            f"{match.group('integer')}.{match.group('fraction')} percent"
        ),
        text,
    )
    normalized = normalized.replace("_", " ").replace("-", " ")
    return tuple(float(value) for value in extract_numeric_occurrences(normalized))


def _verification_numeric_values(
    value: object,
    *,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> tuple[float, ...]:
    if isinstance(value, bool) or value is None:
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, str):
        return tuple(
            float(item) for item in extract_numeric_occurrences(value.strip())
        )
    values: list[float] = []
    if isinstance(value, dict):
        for key, item in value.items():
            values.extend(
                _verification_numeric_values(
                    key,
                    extract_numeric_occurrences=extract_numeric_occurrences,
                )
            )
            values.extend(
                _verification_numeric_values(
                    item,
                    extract_numeric_occurrences=extract_numeric_occurrences,
                )
            )
    elif isinstance(value, list):
        for item in value:
            values.extend(
                _verification_numeric_values(
                    item,
                    extract_numeric_occurrences=extract_numeric_occurrences,
                )
            )
    return tuple(values)


def _deferred_numeric_values(
    payload: dict[str, Any],
    *,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> tuple[float, ...]:
    module = payload.get("module")
    records = module.get("deferred_outputs") if isinstance(module, dict) else None
    if not isinstance(records, list):
        return ()
    values: list[float] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        reason = record.get("reason")
        if isinstance(reason, str):
            values.extend(
                float(value)
                for value in extract_numeric_occurrences(
                    authoritative_numeric_recall_text(reason)
                )
            )
        for field in ("output", "blocked_by", "source_values"):
            raw = record.get(field)
            items = raw if isinstance(raw, list) else [raw]
            for item in items:
                if not isinstance(item, str):
                    continue
                values.extend(
                    _numeric_concept_values(
                        item.rsplit("#", 1)[-1],
                        extract_numeric_occurrences=extract_numeric_occurrences,
                    )
                )
    return tuple(values)


def _companion_test_issues(
    principal_rules: dict[str, dict[str, Any]],
    *,
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
    branches: Sequence[SourceStructureBranch],
    source_text: str,
    corpus_citation_path: str,
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
    structured_formula_branches = [
        branch
        for branch in active_branches
        if branch.kind in {"number", "letter"}
        and _structured_branch_is_formula_leaf(
            branch,
            extract_numeric_occurrences=extract_numeric_occurrences,
        )
    ]
    narrative_formula_branches = _narrative_formula_branches(
        source_text,
        branches=branches,
        active_branches=active_branches,
        deferred_paths=deferred_paths,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    formula_branches = (*structured_formula_branches, *narrative_formula_branches)
    for branch in formula_branches:
        if _formula_branch_has_test_evidence(
            branch,
            principal_rules=principal_rules,
            principal_rule_paths=principal_rule_paths,
            asserted_by_rule=asserted_by_rule,
            corpus_citation_path=corpus_citation_path,
            extract_numeric_occurrences=extract_numeric_occurrences,
            allow_explicit_cover=branch.kind != "formula-clause",
        ):
            continue
        issues.append(
            "[complete-source-unit:tests] Companion tests do not demonstrate "
            f"formula branch {branch.label} at "
            f"{_branch_citation(corpus_citation_path, branch)} with a principal "
            "output assertion and selector input in that branch (or an exact "
            "`covers` source-branch declaration)."
        )

    boundary_obligations = _source_boundary_obligations(
        active_branches,
        narrative_formula_branches=narrative_formula_branches,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    missing_boundaries: list[tuple[SourceStructureBranch, float]] = []
    for branch, boundary in boundary_obligations:
        if _branch_boundary_has_test_evidence(
            branch,
            boundary,
            principal_rules=principal_rules,
            principal_rule_paths=principal_rule_paths,
            asserted_by_rule=asserted_by_rule,
            numeric_value_is_grounded=numeric_value_is_grounded,
        ):
            continue
        missing_boundaries.append((branch, boundary))
    if missing_boundaries:
        rendered = ", ".join(
            f"{branch.label}={value:g}"
            for branch, value in missing_boundaries
        )
        issues.append(
            "[complete-source-unit:tests] Companion tests do not exercise every "
            f"source-stated boundary input; missing: {rendered}."
        )

    active_text = _source_text_without_deferred_branches(
        source_text,
        branches=branches,
        deferred_paths=deferred_paths,
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

    rounding_count = len(_ROUNDING_LANGUAGE.findall(active_text))
    if rounding_count:
        down_rules = {
            name
            for name, rule in principal_rules.items()
            if re.search(r"\bfloor\s*\(", _rule_formula_text(rule))
        }
        up_rules = {
            name
            for name, rule in principal_rules.items()
            if re.search(r"\bceil\s*\(", _rule_formula_text(rule))
        }
        nearest_rules = {
            name
            for name, rule in principal_rules.items()
            if re.search(r"\bfloor\s*\(", _rule_formula_text(rule))
            and re.search(r"\+\s*0?\.5\b", _rule_formula_text(rule))
        }
        applicable_rounding_functions: dict[str, set[str]] = {}
        if _DOWN_ROUNDING_LANGUAGE.search(active_text) and not down_rules:
            issues.append(
                "[complete-source-unit:tests] A source-stated downward-rounding "
                "rule is absent from the principal formula (`floor(...)`)."
            )
        if _DOWN_ROUNDING_LANGUAGE.search(active_text):
            for rule_name in down_rules:
                applicable_rounding_functions.setdefault(rule_name, set()).add(
                    "floor"
                )
        if _UP_ROUNDING_LANGUAGE.search(active_text) and not up_rules:
            issues.append(
                "[complete-source-unit:tests] A source-stated upward-rounding "
                "rule is absent from the principal formula (`ceil(...)`)."
            )
        if _UP_ROUNDING_LANGUAGE.search(active_text):
            for rule_name in up_rules:
                applicable_rounding_functions.setdefault(rule_name, set()).add(
                    "ceil"
                )
        if _NEAREST_ROUNDING_LANGUAGE.search(active_text) and not nearest_rules:
            issues.append(
                "[complete-source-unit:tests] A source-stated nearest/commercial "
                "rounding rule is absent from the principal formula."
            )
        if _NEAREST_ROUNDING_LANGUAGE.search(active_text):
            for rule_name in nearest_rules:
                applicable_rounding_functions.setdefault(rule_name, set()).add(
                    "nearest"
                )
        fractional_cases = _fractional_formula_input_case_count(
            principal_rules,
            asserted_by_rule=asserted_by_rule,
            rule_functions=applicable_rounding_functions,
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


def _structured_branch_is_formula_leaf(
    branch: SourceStructureBranch,
    *,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> bool:
    first_line = (
        branch.text.splitlines()[0] if branch.text.splitlines() else branch.text
    )
    return bool(
        (
            source_states_explicit_computation(branch.text)
            or re.search(r":\s*-?\d", branch.text)
        )
        and (
            ":" in first_line
            or _formula_branch_interval(
                branch,
                extract_numeric_occurrences=extract_numeric_occurrences,
            )
            is not None
        )
    )


def _narrative_formula_branches(
    source_text: str,
    *,
    branches: Sequence[SourceStructureBranch],
    active_branches: Sequence[SourceStructureBranch],
    deferred_paths: set[tuple[str, ...]],
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> tuple[SourceStructureBranch, ...]:
    """Return range-formula clauses not already represented by list branches."""

    structured_ranges = [
        (branch.start, branch.end)
        for branch in branches
        if branch.kind in {"number", "letter"}
    ]
    obligations: list[SourceStructureBranch] = []
    for clause_index, (start, end, clause) in enumerate(
        _source_clause_spans(source_text),
        start=1,
    ):
        if any(
            start < structured_end and structured_start < end
            for structured_start, structured_end in structured_ranges
        ):
            continue
        if _span_is_deferred(
            start,
            end,
            branches=branches,
            deferred_paths=deferred_paths,
        ):
            continue
        if not source_states_explicit_computation(clause):
            continue
        interval = _formula_interval_from_text(
            clause,
            extract_numeric_occurrences=extract_numeric_occurrences,
        )
        if interval is None:
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


def _source_clause_spans(source_text: str) -> Iterable[tuple[int, int, str]]:
    """Yield offset-preserving semicolon and true sentence clauses."""

    boundary = re.compile(
        r";|[.!?](?=(?:[ \t]+[A-ZÄÖÜ(]|\s*$))",
        flags=re.MULTILINE,
    )
    start = 0
    for match in boundary.finditer(source_text):
        end = match.end()
        raw = source_text[start:end]
        left_trimmed = len(raw) - len(raw.lstrip())
        right_trimmed = len(raw.rstrip())
        if right_trimmed > left_trimmed:
            yield (
                start + left_trimmed,
                start + right_trimmed,
                raw[left_trimmed:right_trimmed],
            )
        start = end
    raw = source_text[start:]
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
    return any(
        branch.start <= start
        and end <= branch.end
        and any(
            branch.path[: len(deferred)] == deferred
            for deferred in deferred_paths
        )
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


def _formula_branch_has_test_evidence(
    branch: SourceStructureBranch,
    *,
    principal_rules: dict[str, dict[str, Any]],
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    corpus_citation_path: str,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    allow_explicit_cover: bool,
) -> bool:
    interval = _formula_branch_interval(
        branch,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    for rule_name in _rules_covering_branch(branch, principal_rule_paths):
        selector_names = _rule_numeric_selector_names(principal_rules[rule_name])
        for case in asserted_by_rule.get(rule_name, ()):
            if allow_explicit_cover and _case_explicitly_covers_branch(
                case,
                branch,
                corpus_citation_path=corpus_citation_path,
            ):
                return True
            if interval is None or not selector_names:
                continue
            if any(
                _interval_contains(interval, value)
                for value in _case_numeric_selector_values(case, selector_names)
            ):
                return True
    return False


def _branch_boundary_has_test_evidence(
    branch: SourceStructureBranch,
    boundary: float,
    *,
    principal_rules: dict[str, dict[str, Any]],
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> bool:
    for rule_name in _rules_covering_branch(branch, principal_rule_paths):
        selector_names = _rule_numeric_selector_names(principal_rules[rule_name])
        if not selector_names:
            continue
        for case in asserted_by_rule.get(rule_name, ()):
            if any(
                numeric_value_is_grounded(value, {boundary})
                for value in _case_numeric_selector_values(case, selector_names)
            ):
                return True
    return False


def _rules_covering_branch(
    branch: SourceStructureBranch,
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
) -> tuple[str, ...]:
    if not branch.path:
        return tuple(principal_rule_paths)
    return tuple(
        name
        for name, paths in principal_rule_paths.items()
        if branch.path in paths
    )


def _formula_branch_interval(
    branch: SourceStructureBranch,
    *,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> tuple[float | None, bool, float | None, bool] | None:
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
) -> tuple[float | None, bool, float | None, bool] | None:
    lowered = text.lower()
    keyword = re.search(
        r"\b(?:von|from|unter|less\s+than|below|bis|up\s+to|höchstens|"
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
    values = tuple(
        float(value) for value in extract_numeric_occurrences(range_text)
    )
    if not values:
        return None
    lowered_range = range_text.lower()
    if re.match(r"(?:von|from)\b", lowered_range) and re.search(
        r"\b(?:bis|to|through)\b",
        lowered_range,
    ):
        if len(values) < 2:
            return None
        return values[0], True, values[1], True
    if re.match(r"(?:unter|less\s+than|below)\b", lowered_range):
        return None, False, values[0], False
    if re.match(r"(?:bis|up\s+to|höchstens|at\s+most)\b", lowered_range):
        return None, False, values[0], True
    if re.match(r"(?:über|more\s+than|above)\b", lowered_range):
        return values[0], False, None, False
    if re.match(
        r"(?:von|ab|from|at\s+least|mindestens)\b",
        lowered_range,
    ):
        return values[0], True, None, False
    return None


def _interval_contains(
    interval: tuple[float | None, bool, float | None, bool],
    value: float,
) -> bool:
    lower, lower_inclusive, upper, upper_inclusive = interval
    if lower is not None and (
        value < lower or (not lower_inclusive and math.isclose(value, lower))
    ):
        return False
    if upper is not None and (
        value > upper or (not upper_inclusive and math.isclose(value, upper))
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
) -> set[tuple[str, str]]:
    toggled: set[tuple[str, str]] = set()
    for rule_name, rule in principal_rules.items():
        selector_names = _rule_exception_selector_names(rule)
        if not selector_names:
            continue
        for key in _paired_boolean_toggle_keys(asserted_by_rule.get(rule_name, ())):
            matched_names = _input_key_names(key) & selector_names
            toggled.update((rule_name, name) for name in matched_names)
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


def _fractional_formula_input_case_count(
    principal_rules: dict[str, dict[str, Any]],
    *,
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    rule_functions: dict[str, set[str]],
) -> int:
    evidence: set[tuple[str, int]] = set()
    for rule_name, functions in rule_functions.items():
        rounded_operand_identifiers = _rounding_operand_identifier_names(
            principal_rules[rule_name],
            functions=functions,
        )
        for index, case in enumerate(asserted_by_rule.get(rule_name, ())):
            inputs = case.get("input")
            if not isinstance(inputs, dict):
                continue
            if any(
                _input_key_names(key) & rounded_operand_identifiers
                and any(
                    not float(value).is_integer()
                    for value in _numeric_test_input_values(input_value)
                )
                for key, input_value in inputs.items()
            ):
                evidence.add((rule_name, index))
    return len(evidence)


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


def _case_explicitly_covers_branch(
    case: dict[str, Any],
    branch: SourceStructureBranch,
    *,
    corpus_citation_path: str,
) -> bool:
    raw_covers = case.get("covers")
    if isinstance(raw_covers, (str, dict)):
        raw_covers = [raw_covers]
    if not isinstance(raw_covers, list):
        return False
    for item in raw_covers:
        value = (
            item.get("source_branch")
            if isinstance(item, dict)
            else item
        )
        if not isinstance(value, str):
            continue
        if branch.path in _paths_from_source_reference(
            value,
            corpus_citation_path=corpus_citation_path,
        ):
            return True
    return False


def _source_boundary_obligations(
    branches: Sequence[SourceStructureBranch],
    *,
    narrative_formula_branches: Sequence[SourceStructureBranch] = (),
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> tuple[tuple[SourceStructureBranch, float], ...]:
    obligations: list[tuple[SourceStructureBranch, float]] = []
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
        obligations.extend(
            (branch, float(value))
            for value in dict.fromkeys(extract_numeric_occurrences(prefix))
        )
    for branch in narrative_formula_branches:
        interval = _formula_branch_interval(
            branch,
            extract_numeric_occurrences=extract_numeric_occurrences,
        )
        if interval is None:
            continue
        lower, _, upper, _ = interval
        obligations.extend(
            (branch, boundary)
            for boundary in (lower, upper)
            if boundary is not None
        )
    return tuple(
        {
            (branch.path, float(value), branch.kind): (branch, float(value))
            for branch, value in obligations
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


def _collapse_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
