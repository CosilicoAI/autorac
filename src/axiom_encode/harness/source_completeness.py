"""Opt-in authoritative source-unit completeness accounting.

This module deliberately sits beside, rather than inside, the numeric
extractor.  It consumes the extractor and scalar-inventory interfaces supplied
by :mod:`validator_pipeline`, which keeps locale-tokenizer work independent
from the source-unit admission policy implemented here.
"""

from __future__ import annotations

import ast
import bisect
import contextlib
import copy
import functools
import heapq
import itertools
import math
import re
import textwrap
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import (
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    Decimal,
    InvalidOperation,
    localcontext,
)
from typing import Any, Mapping, Protocol

import yaml

from axiom_encode.harness.proof_validator import _normalize_atom_path
from axiom_encode.statute import (
    CitationParts,
    normalize_rulespec_path_segment,
    parse_usc_citation,
)

_UNBOUND_FORMULA_DIAGNOSTIC_RULE_LIMIT = 32
_UNBOUND_FORMULA_DIAGNOSTIC_CASE_LIMIT = 8
_TEMPORAL_WITNESS_NAME_LIMIT = 32
_TEMPORAL_WITNESS_VERSION_LIMIT = 8
_PARAMETER_ALIAS_FORMULA_KEYWORDS = {
    "and",
    "else",
    "false",
    "holds",
    "if",
    "match",
    "none",
    "not",
    "not_holds",
    "null",
    "or",
    "true",
}


class NumericOccurrenceLike(Protocol):
    """Typed numeric source evidence consumed from the shared extractor."""

    value: float
    start: int
    end: int
    raw: str
    has_rate_context: bool
    has_temporal_context: bool
    has_structural_context: bool
    source_value: float | None
    requires_rate_context: bool
    is_word_number: bool
    alternative_values: tuple[float, ...]


@dataclass(frozen=True)
class _NumericOccurrenceView:
    """One typed occurrence rebased into a containing source fragment."""

    value: float
    start: int
    end: int
    raw: str
    has_rate_context: bool
    has_temporal_context: bool
    has_structural_context: bool
    source_value: float | None
    requires_rate_context: bool
    is_word_number: bool
    alternative_values: tuple[float, ...]


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
class _OutlineMarker:
    """One resolved marker in a nested parenthesized legal outline."""

    start: int
    path: tuple[str, ...]
    kind: str
    label: str


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
    upper_directly_conjoined: bool = False


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
    evaluated_value: tuple[str, str] | None
    evaluates_to_zero: bool
    constant_environment: dict[str, Any]


@dataclass(frozen=True)
class _SourceConditionClause:
    """One exact source proposition owned by a formula proof excerpt."""

    branch_path: tuple[str, ...]
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _TerminalGateAlternative:
    """Terminal facts used by one active formula path and temporal partition."""

    names: frozenset[str]
    start: str | None
    end: str | None
    resolved: bool = True


@dataclass(frozen=True)
class _TerminalGateEvidence:
    """Semantic evidence from one terminal with fail-closed polarity metadata."""

    entities: frozenset[str]
    predicates: frozenset[str]
    polarity_conflict: bool
    explicit_name_polarities: frozenset[tuple[str, bool]] = frozenset()


@dataclass(frozen=True)
class _UnboundFormulaDiagnostic:
    """Best-effort unbound formula matches and scan completeness."""

    rule_names: tuple[str, ...]
    scan_capped: bool


@dataclass(frozen=True)
class _UncorroboratedFormulaDependencyDiagnostic:
    """Best-effort missing derived assertions for a formula witness."""

    dependency_names: tuple[str, ...]
    case_names: tuple[str, ...]
    scan_capped: bool


@dataclass(frozen=True)
class _TemporalFormulaValue:
    """Literal parameter values selectable by a companion case period."""

    versions: tuple[tuple[str, str, Any], ...]
    version_formula_excerpts: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class _FormulaBranchNode:
    """One parsed RuleSpec conditional or match expression."""

    start: int
    end: int
    kind: str
    selectors: tuple[str, ...]
    patterns: tuple[str, ...]
    choices: tuple[str, ...]


@dataclass(frozen=True, order=True)
class _ExceptionWitness:
    """One isolated exception selector effect demonstrated by paired cases."""

    rule_name: str
    selector_name: str
    active_value: bool
    blocks: bool
    boolean_effect: bool
    zeroes: bool
    numeric_transition: tuple[float, float] | None
    relational_transitions: tuple[tuple[str, str, str], ...] = ()
    case_pair_identity: tuple[int, ...] = ()


@dataclass(frozen=True)
class _NamedSourceNumber:
    """A matched source number whose word value may be unsupported."""

    value: float | None


_PARAGRAPH_MARKER = re.compile(
    r"(?m)^[ \t]*(?P<marker>\((?P<label>\d+[a-z]?|[a-z])\))(?=\s|bis\b)",
    flags=re.IGNORECASE,
)
_DOTTED_SUBSECTION_MARKER = re.compile(
    r"(?m)^[ \t]*(?P<marker>(?P<label>[A-Z])\.)(?:[ \t]+|(?=\r?$)|(?=\())"
)
_JOINED_DOTTED_BOUNDARY_MARKER = re.compile(
    r"(?m)^[ \t]*(?P<marker>(?P<label>[A-Z])\.)"
    r"(?=(?:\d+[A-Z]?(?::|\.\d)|(?i:e)\.(?:[ \t]|(?=\r?$))))"
)
_NUMBER_MARKER = re.compile(
    r"(?m)^[ \t]*(?P<marker>(?P<label>\d+[a-z]?)\.)[ \t]+",
    flags=re.IGNORECASE,
)
_INLINE_DOTTED_NUMBER_MARKER = re.compile(
    r"(?<![A-Za-z0-9.])(?P<marker>(?P<label>[1-9]\d{0,2})\.)[ \t]+"
)
_INLINE_DOTTED_LIST_MINIMUM_LENGTH = 3
_INLINE_DOTTED_LIST_MATCH_LIMIT = 256
_INLINE_DOTTED_LIST_CHAPEAU_LIMIT = 512
_INLINE_DOTTED_LIST_STRUCTURAL_CHAPEAU = re.compile(
    r"^\s*(?:(?:(?:all|each)\s+of\s+the|the)\s+)?following\s+(?:tax\s+)?"
    r"(?:allowances?|benefits?|conditions?|credits?|deductions?|eligibility|"
    r"exceptions?|exemptions?|qualifications?|requirements?)\b"
    r"(?:\s*,\s*(?:when|where|if)\s+applicable\s*,|"
    r"\s+(?:when|where|if)\s+applicable)?\s+"
    r"(?:appl(?:y|ies)|(?:shall|must|may)\s+(?:apply|hold|be\s+"
    r"(?:allowed|available|claimed|required|satisfied|met))|"
    r"(?:are|is)\s+(?:allowed|available|required|satisfied|met)|"
    r"(?:shall|must)(?:\s*,\s*(?:when|where|if)\s+applicable\s*,)?\s+"
    r"be\s+deducted(?:\s+from\s+the\s+result\s+obtained\s+under\s+"
    r"subsection\s+\(\d+[a-z]?\)\s+of\s+this\s+section\s+to\s+arrive\s+at\s+"
    r"the\s+annual\s+tax)?)"
    r"(?:\s+(?:(?:only\s+)?when|if)\s+(?:"
    r"no\s+(?:[a-z][a-z'-]*\s+){0,5}"
    r"(?:income|earnings?|wages?|benefits?|credits?|deductions?|allowances?|"
    r"exemptions?)\s+(?:(?:is|are|was|were)|(?:has|have|had)\s+been)\s+"
    r"(?:reported|received|earned|available|claimed|allowed|paid|provided)|"
    r"(?:[a-z][a-z'-]*\s+){0,5}"
    r"(?:income|earnings?|wages?|benefits?|credits?|deductions?|allowances?|"
    r"exemptions?)\s+(?:is|are|was|were)\s+not\s+"
    r"(?:reported|received|earned|available|claimed|allowed|paid|provided)|"
    r"(?:the\s+)?(?:applicant|claimant|dependent|household|individual|person|"
    r"recipient|spouse|taxpayer|taxpayer(?:'s|’s)\s+spouse)\s+"
    r"(?:(?:is|are|was|were)\s+not\s+(?:"
    r"(?:(?:an?|the)\s+)?(?:(?:another\s+)?taxpayer(?:'s|’s)\s+)?"
    r"(?:dependent|eligible|qualified|blind|disabled|married|resident)|"
    r"the\s+dependent\s+of\s+another\s+taxpayer)|"
    r"(?:has|have|had)\s+no\s+(?:[a-z][a-z'-]*\s+){0,5}"
    r"(?:income|earnings?|wages?|benefits?|credits?|deductions?|allowances?|"
    r"exemptions?)|"
    r"(?:is|are|was|were)\s+"
    r"(?:dependent|eligible|qualified|blind|disabled|married|resident))"
    r"))?\s*:\s*$",
    flags=re.IGNORECASE,
)
_LETTER_MARKER = re.compile(
    r"(?m)^[ \t]*(?P<marker>(?P<label>[a-z]{1,2})\))[ \t]+",
    flags=re.IGNORECASE,
)
_PARENTHESIZED_OUTLINE_MARKER = re.compile(
    r"(?m)^[ \t]*(?:[A-Z]\.)?"
    r"(?P<marker>\((?:\d+[a-z]?|[a-z]|[ivxlcdm]{2,15})\)"
    r"(?:[ \t]*\((?:\d+[a-z]?|[a-z]|[ivxlcdm]{2,15})\))*)"
    r"(?=\s|bis\b)",
    flags=re.IGNORECASE,
)
_INLINE_PARENTHESIZED_OUTLINE_MARKER = re.compile(
    r"(?:(?<=[.!?;:])[ \t]+|(?<=—)[ \t]*)"
    r"(?P<marker>\((?:\d+[a-z]?|[a-z]|[ivxlcdm]{2,15})\)"
    r"(?:[ \t]*\((?:\d+[a-z]?|[a-z]|[ivxlcdm]{2,15})\))*)"
    r"(?=\s|bis\b)",
    flags=re.IGNORECASE,
)
_INLINE_OUTLINE_REFERENCE_CONTEXT = re.compile(
    r"\b(?:section|subsection|paragraph|subparagraph|clause|subclause|item|"
    r"subitem|division|subdivision|part|subpart|chapter|subchapter|title|article|"
    r"subarticle)s?"
    r"(?:\s+(?:no\.?|number)(?:\s*[A-Za-z0-9][A-Za-z0-9.-]*)?)?"
    r"\s*[:;,.!?—]?\s*$|"
    r"\b(?:see|compare(?:\s+(?:to|with))?|refer(?:\s+back)?(?:\s+to)?|"
    r"subject\s+to|under|pursuant\s+to|according\s+to|"
    r"in\s+accordance\s+with|for\s+purposes\s+of|"
    r"as\s+(?:provided|specified|described|stated)\s+in)\s*:\s*$",
    flags=re.IGNORECASE,
)
_INLINE_OUTLINE_NAMED_REFERENCE_CONTEXT = re.compile(
    r"\b(?:see|compare(?:\s+(?:to|with))?|refer(?:\s+back)?(?:\s+to)?|"
    r"subject\s+to|under|pursuant\s+to|according\s+to|"
    r"in\s+accordance\s+with|for\s+purposes\s+of|"
    r"as\s+(?:provided|specified|described|stated)\s+in)\s+"
    r"(?:(?!\n|[;!?—]|\.(?=\s+[A-Z])).){1,96}[:;.!?—]\s*$",
    flags=re.IGNORECASE,
)
_INLINE_OUTLINE_STRUCTURAL_CHAPEAU_CONTEXT = re.compile(
    r"\b(?:the\s+)?following\s+"
    r"(?:sections?|subsections?|paragraphs?|subparagraphs?|clauses?|"
    r"subclauses?|items?|subitems?|divisions?|subdivisions?|parts?|subparts?)"
    r"\s*:\s*$",
    flags=re.IGNORECASE,
)
_INLINE_OUTLINE_CHAPEAU_REFERENCE_LEAD = re.compile(
    r"\b(?:see|compare(?:\s+(?:to|with))?|refer(?:\s+back)?(?:\s+to)?|"
    r"subject\s+to|under|pursuant\s+to|according\s+to|"
    r"in\s+accordance\s+with|for\s+purposes\s+of|"
    r"as\s+(?:provided|specified|described|stated)\s+in)"
    r"\s+(?:the\s+)?following\b",
    flags=re.IGNORECASE,
)
_INLINE_OUTLINE_EXPLICIT_REFERENCE_COMMAND = re.compile(
    r"(?:^|[.!?;:—]\s+)(?:"
    r"(?:see(?:\s*,\s*e\.g\.,?)?|refer(?:\s+back)?(?:\s+to)?|"
    r"compare(?:\s+(?:to|with))?|subject\s+to|pursuant\s+to|"
    r"according\s+to|in\s+accordance\s+with|for\s+purposes\s+of|under)\b|"
    r"(?:except\s+)?as\s+(?:provided|specified|described|stated)\s+"
    r"(?:in|under)\b)"
    r".{0,160}[:;.!?—]\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_GLUED_SENTENCE_MARKER = re.compile(
    r"(?<![\w])(?P<label>[1-9]\d?)"
    r"(?!(?i:st|nd|rd|th)\b)"
    r"(?=[A-ZÄÖÜ](?!:)(?![ \t]*[.:/\-\u2010-\u2015\u2212\ufe58\ufe63\uff0d]"
    r"[ \t]*\d))"
)
_EXPLICIT_SENTENCE_MARKER = re.compile(
    r"(?:(?<=^)|(?<=[.;])|(?<=\)))[ \t]*Satz[ \t]+"
    r"(?P<label>[1-9]\d?)(?:[ \t]*:[ \t]*|[ \t]+)(?=[A-ZÄÖÜ])",
    flags=re.IGNORECASE | re.MULTILINE,
)
_EDITORIAL_OMISSION_ONLY = re.compile(
    r"^\s*"
    r"(?:(?:\((?:\d+[a-z]?|[a-z])\)|\d+[a-z]?\.|[a-z]\))\s*)?"
    r"(?:bis\s+\(\d+[a-z]?\)\s*)?"
    r"(?:\(\s*)?"
    r"(?:weggefallen|aufgehoben|omitted|\.{3,})"
    r"(?:\s*\))?"
    r"\s*[.;]?\s*$",
    flags=re.IGNORECASE,
)
_GERMAN_CARDINAL_VALUES = {
    "ein": 1.0,
    "eins": 1.0,
    "eine": 1.0,
    "einen": 1.0,
    "einem": 1.0,
    "einer": 1.0,
    "eines": 1.0,
    "zwei": 2.0,
    "drei": 3.0,
    "vier": 4.0,
    "fünf": 5.0,
    "sechs": 6.0,
    "sieben": 7.0,
    "acht": 8.0,
    "neun": 9.0,
    "zehn": 10.0,
}
_ARITHMETIC_EXPRESSION = re.compile(
    r"(?:\d+(?:[.,]\d+)?|[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß]*)"
    r"[ \t]*(?:[+*/=×·•∗∙]|(?<!\w)[−–-](?!\w))[ \t]*"
    r"(?:\d+(?:[.,]\d+)?|[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß]*)"
)
_SYMBOLIC_ARITHMETIC_OPERAND = (
    r"(?:"
    r"[A-Za-zÄÖÜäöüß_][A-Za-z0-9ÄÖÜäöüß_]*|"
    r"\d{1,3}(?:[ .]\d{3})+(?:,\d+)?|"
    r"\d+(?:[.,]\d+)?"
    r")"
)
_WORDED_ARITHMETIC_EXPRESSION = re.compile(
    rf"{_SYMBOLIC_ARITHMETIC_OPERAND}\s+"
    r"(?:plus|minus|mal|less(?!\s+than\b))\s+(?:the\s+)?"
    r"(?!(?:beträgt|gilt|ist|sind|wird|werden|equals?|applies?)\b)"
    rf"{_SYMBOLIC_ARITHMETIC_OPERAND}",
    flags=re.IGNORECASE,
)
_SYMBOLIC_ARITHMETIC_TOKEN = re.compile(
    rf"\b(?:plus|minus|mal)\b|{_SYMBOLIC_ARITHMETIC_OPERAND}|"
    r"[()+*/×·•∗∙−–-]",
    flags=re.IGNORECASE,
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
    r"mit\s+(?:(?:dem|einem)\s+faktor\s+)?"
    r"(?:von\s+)?"
    r"(?:\d+(?:[.,]\d+)?|[a-zäöüß]+)\s+zu\s+"
    r"(?:multiplizieren|vervielfachen)|"
    r"durch\s+multiplikation\s+mit\s+"
    r"(?:(?:(?:dem|einem)\s+)?faktor\s+)?(?:von\s+)?"
    r"(?:\d+(?:[.,]\d+)?|[a-zäöüß]+)\s+zu\s+ermitteln|"
    r"unter\s+anwendung\s+(?:des|eines)\s+faktors?\s+"
    r"(?:\d+(?:[.,]\d+)?|[a-zäöüß]+)\s+zu\s+ermitteln|"
    r"(?:ist|sind)\s+zu\s+(?:"
    r"verdoppeln|verdreifachen|vervierfachen|verfünffachen|"
    r"versechsfachen|versiebenfachen|verachtfachen|verneunfachen|"
    r"verzehnfachen|halbieren)|"
    r"durch\s+halbierung\s+zu\s+ermitteln|"
    r"in\s+(?:2|zwei)\s+gleiche\s+teile\s+zu\s+teilen|"
    r"durch\s+(?:\d+(?:[.,]\d+)?|[a-zäöüß]+)\s+zu\s+teilen|"
    r"um\s+(?:\d+(?:[.,]\d+)?|[a-zäöüß]+)\s+zu\s+"
    r"(?:erhöhen|vermindern|kürzen|vermehren)|"
    r"aus\s+[^.;]{1,100}\s+zu\s+(?:summieren|addieren)|"
    r"\bmal\s+(?:\d+(?:[.,]\d+)?|"
    r"ein(?:s|e[nsrm]?)?|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn)|"
    r"(?:wird|werden)\s+(?:verdoppelt|verdreifacht|vervierfacht)|"
    r"\d+(?:[.,]\d+)?\s*-?fach(?:e[nsrm]?)?|"
    r"(?:vermindert|erhöht|gekürzt|vermehrt)\s+um|"
    r"(?:erhöht|mindert|vermindert|kürzt|vermehrt)\s+sich\s+um|"
    r"(?:abzüglich|zuzüglich)|"
    r"(?:prozent|\d+(?:[.,]\d+)?\s*%)\s+(?:des|der|von|of)|"
    r"\d+(?:[.,]\d+)?\s+(?:vom\s+hundert|v\.?\s*h\.?)\s+"
    r"(?:des|der|von)|"
    r"splitting-verfahren|verfahren\s+nach\s+absatz|"
    r"calculated|computed|computation|multiplied|divided|"
    r"twice|"
    r"amount\s+of\s+(?:the\s+)?excess|"
    r"\d+(?:[.,]\d+)?\s+times\b|"
    r"percentage\s+of|"
    r"equals?[^.;]{0,100}\b(?:plus|minus|times)\b"
    r")\b",
    flags=re.IGNORECASE,
)
_ENGLISH_NUMBER_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
    r"thousand|million|billion|trillion)"
)
_ENGLISH_WORDED_PERCENTAGE_OF = re.compile(
    rf"\b{_ENGLISH_NUMBER_WORD}"
    rf"(?:(?:[-\s]+(?:and[-\s]+)?){_ENGLISH_NUMBER_WORD})*"
    r"\s+(?:percent|per\s+cent)\s+of\b",
    flags=re.IGNORECASE,
)
_EXPLICIT_NUMERIC_PERCENTAGE_OF = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|per\s+cent|percent)\s+of\b",
    flags=re.IGNORECASE,
)
_ENGLISH_CARDINAL_PHRASE = (
    rf"{_ENGLISH_NUMBER_WORD}"
    rf"(?:(?:[-\s]+(?:and[-\s]+)?){_ENGLISH_NUMBER_WORD})*"
)
_EXCESS_BOUND_TAX_RATE_VALUE = (
    rf"(?:\d+(?:[.,]\d+)?\s*(?:%|percent|per\s+cent)|"
    rf"{_ENGLISH_CARDINAL_PHRASE}\s+(?:percent|per\s+cent))"
)
_EXCESS_BOUND_TAX_RATE_PREDICATE = (
    rf"(?:"
    rf"(?:is|are|shall|must|will)\s+(?:be\s+)?taxed\s+at\s+"
    rf"(?:the\s+rate\s+of\s+)?{_EXCESS_BOUND_TAX_RATE_VALUE}|"
    rf"tax\s+(?:is|shall\s+be|must\s+be|will\s+be)\s+imposed\s+at\s+"
    rf"(?:a\s+)?rate\s+of\s+{_EXCESS_BOUND_TAX_RATE_VALUE}"
    rf")"
)
_EXCESS_BOUND_TAX_RATE_LANGUAGE = re.compile(
    rf"\b(?:not\s+)?in\s+excess\s+of\b[^.;]{{0,180}}"
    rf"\b{_EXCESS_BOUND_TAX_RATE_PREDICATE}(?=\W|$)|"
    rf"\b{_EXCESS_BOUND_TAX_RATE_PREDICATE}(?=\W|$)[^.;]{{0,180}}"
    rf"\b(?:not\s+)?in\s+excess\s+of\b",
    flags=re.IGNORECASE,
)
_ENGLISH_ORDINAL_WORD = (
    r"(?:second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|"
    r"eighteenth|nineteenth|twentieth|thirtieth|fortieth|fiftieth|sixtieth|"
    r"seventieth|eightieth|ninetieth|hundredth|thousandth|millionth|billionth|"
    r"trillionth)"
)
_ENGLISH_FRACTION_DENOMINATOR = rf"(?:half|halves|quarters?|{_ENGLISH_ORDINAL_WORD}s?)"
_ENGLISH_FRACTION_PHRASE = (
    rf"(?:half|quarter|(?:a|an)[-\s]+(?:half|quarter|{_ENGLISH_ORDINAL_WORD})|"
    rf"{_ENGLISH_CARDINAL_PHRASE}[-\s]+"
    rf"(?:(?:{_ENGLISH_NUMBER_WORD})[-\s]+)*{_ENGLISH_FRACTION_DENOMINATOR})"
)
_ENGLISH_FRACTION_OF = re.compile(
    rf"\b(?P<fraction>{_ENGLISH_FRACTION_PHRASE})\s+of\s+"
    rf"(?P<target>[^.;:\n]+)",
    flags=re.IGNORECASE,
)
_ENGLISH_CARDINAL_SEQUENCE = rf"{_ENGLISH_NUMBER_WORD}(?:[-\s]+{_ENGLISH_NUMBER_WORD})*"
_ENGLISH_FRACTIONAL_PERCENTAGE_OF = re.compile(
    rf"\b(?:{_ENGLISH_CARDINAL_SEQUENCE}\s+and\s+)?"
    rf"(?:half|quarter|(?:a|an|{_ENGLISH_CARDINAL_SEQUENCE})[-\s]+"
    rf"{_ENGLISH_FRACTION_DENOMINATOR})\s+"
    rf"(?:percent|per\s+cent)\s+of\b",
    flags=re.IGNORECASE,
)
_FORMULA_TABLE_NAME = r"(?:table|schedule)(?:\s+[A-Z0-9]+(?:-[A-Z0-9]+)*)?"
_FORMULA_GENERIC_TABLE_TARGET = (
    r"(?:the\s+)?(?:"
    r"following\s+(?:amounts?|percentages?|rates?|values?|tables?|schedules?)"
    r"(?:\s+for\s+(?:the\s+)?following\s+tax\s+years?)?|"
    rf"following\s+{_FORMULA_TABLE_NAME}|"
    rf"{_FORMULA_TABLE_NAME}\s+(?:set\s+forth\s+)?below|"
    rf"{_FORMULA_TABLE_NAME}|"
    r"(?:amounts?|percentages?|rates?|values?)\s+"
    r"(?:(?:set\s+forth|shown|prescribed)\s+)?in\s+(?:the\s+)?"
    rf"(?:following\s+{_FORMULA_TABLE_NAME}|{_FORMULA_TABLE_NAME}\s+below))"
)
_FORMULA_GENERIC_TABLE_REFERENCE = (
    r"(?:pursuant\s+to|as\s+set\s+forth\s+in|in\s+accordance\s+with|"
    r"from|by\s+reference\s+to)\s+"
    rf"{_FORMULA_GENERIC_TABLE_TARGET}"
)
_FORMULA_NONOPERATIVE_TABLE_HEADING = re.compile(
    rf"(?:\b(?:calculated|computed|determined)\s+(?:"
    r"as\s+follows|"
    rf"{_FORMULA_GENERIC_TABLE_REFERENCE}|"
    rf"(?:using|according\s+to|based\s+on|by|under|in|with|"
    rf"as\s+(?:shown|provided)\s+in|in\s+the\s+manner\s+shown\s+in)\s+"
    rf"{_FORMULA_GENERIC_TABLE_TARGET}"
    rf"|based\s+on\s+[^.;:\n]{{1,240}}?\band\s+(?:\w+\s+)?"
    rf"(?:be\s+)?equal\s+to\s+{_FORMULA_GENERIC_TABLE_TARGET}"
    rf")|\b(?:equals?|(?:be\s+)?equal\s+to)\s+{_FORMULA_GENERIC_TABLE_TARGET})"
    r"(?:\s+for\s+(?:taxable|tax|calendar|fiscal)\s+years?\s+"
    r"(?:\d{4}(?:(?:\s*,\s*(?:(?:and|or)\s+)?|\s+(?:and|or|through|to)\s+|"
    r"\s*[-–]\s*)\d{4})*|beginning\s+[^.;:\n]{0,80}\b\d{4}))?"
    r"\s*:\s*$",
    flags=re.IGNORECASE,
)
_FORMULA_COMPUTED_OPERATION_LANGUAGE = re.compile(
    r"\b(?:calculated|computed|determined)\s+(?:"
    r"(?:by|through)\s+(?:adding|addition|subtracting|subtraction|dividing|"
    r"division|multiplying|multiplication|reducing|reduction|deducting|"
    r"deduction|increasing|increase|decreasing|decrease)\b|"
    r"as\s+(?:the\s+)?(?:addition|average|difference|mean|median|ratio|"
    r"quotient|remainder|product|sum|total|min|max|minimum|maximum|least|"
    r"greatest|lesser|greater|lower|higher|lowest|highest|smallest|largest|"
    r"smaller|larger|percent|percentage)\s+(?:of|between)\b"
    r")",
    flags=re.IGNORECASE,
)
_FORMULA_INDEPENDENT_COORDINATE = (
    r"(?:,\s*(?:(?:although|and|because|but|considering\s+that|given\s+that|"
    r"inasmuch\s+as|nor|notwithstanding\s+that|or|provided(?:\s+that)?|since|"
    r"seeing\s+that|then|though|whereas|while|yet)\s+)?|"
    r"\s+(?:although|and|because|but|considering\s+that|given\s+that|"
    r"inasmuch\s+as|nor|notwithstanding\s+that|or|provided(?:\s+that)?|since|"
    r"seeing\s+that|then|though|whereas|while|yet)\s+)"
    r"(?:the\s+)?"
    r"(?!(?:shall|must|may|is|are|equals?)\b)[^,.;:\n]+?"
    r"\b(?:shall|must|may|is|are|equals?)\b"
    r"|\s+as\s+(?:the\s+)?(?:amounts?|assessments?|benefits?|credits?|"
    r"deductions?|exemptions?|incomes?|liabilit(?:y|ies)|payments?|"
    r"surcharges?|surtaxes?|taxes?)\b[^,.;:\n]*?"
    r"\b(?:shall|must|may|is|are|equals?)\b"
)
_FORMULA_APPLIED_OPERATION_LANGUAGE = re.compile(
    r"\b(?:calculated|computed|determined)\s+(?:"
    r"(?:by|through)\s+(?:applying|application\s+of))\s+"
    r"(?P<operands>[^.;:\n]+?)"
    rf"(?={_FORMULA_INDEPENDENT_COORDINATE}|[.;:\n]|$)",
    flags=re.IGNORECASE,
)
_FORMULA_RESULT_PREDICATE = (
    r"(?:(?:is|are|shall\s+be|must\s+be|may\s+be)\s+equal\s+to|"
    r"(?:shall|must|may)\s+equal|is|are|equals?|shall\s+be|must\s+be|"
    r"may\s+be|means|constitutes?)"
)
_FORMULA_UNCONDITIONAL_OPERATOR_LANGUAGE = re.compile(
    rf"\b(?:calculated|computed)\s+"
    rf"(?!(?:using|according\s+to|under|as\s+follows|based\s+on|"
    rf"by|in|with|without|pursuant\s+to|on\s+the\s+basis\s+of|"
    rf"in\s+the\s+manner|"
    rf"{_FORMULA_GENERIC_TABLE_REFERENCE})\b)|"
    r"\b(?:sum|product)\s+of\b[^.;:\n]{1,120}\band\b[^.;:\n]{1,120}"
    r"\b(?:is|equals?|shall\s+be)\s+\d+(?:[.,]\d+)?\b|"
    r"\bdifference\s+between\b[^.;:\n]{1,120}\band\b[^.;:\n]{1,120}"
    r"\b(?:is|equals?|shall\s+be)\s+\d+(?:[.,]\d+)?\b",
    flags=re.IGNORECASE,
)
_FORMULA_ROUNDED_OPERATION_LANGUAGE = re.compile(
    r"\b(?:(?:sum|product)\s+of|difference\s+between)\b"
    r"[^.;:\n]{1,120}\band\b[^.;:\n]{1,120}"
    r"\b(?:shall|must|may|is|are)\s+(?:be\s+)?rounded\b",
    flags=re.IGNORECASE,
)
_FORMULA_FOLLOWING_OPERAND_PROVISO = re.compile(
    r"\s*(?:(?:and\s+)?provided"
    r"(?:\s*,?\s*(?:always|further|however|in\s+any\s+event|"
    r"nevertheless|nonetheless|only)\s*,?)*\s+that|"
    r"(?:on|upon)\s+(?:the\s+)?condition\s+that|(?:only\s+)?if|when|where|"
    r"(?:so|as)\s+long\s+as|in\s+which\s+case|in\s+the\s+event(?:\s+that)?|"
    r"in\s+(?:(?:a|any|the)\s+)?cases?(?:\s+where)?|save\s+(?:that|where)|"
    r"to\s+the\s+extent\s+that|except\s+as\s+(?:(?:otherwise\s+)?"
    r"(?:provided|specified)|required|prescribed(?:\s+by)?)|"
    r"except\s+(?:that|if|when|where|"
    r"to\s+the\s+extent\s+that|in\s+(?:(?:the|any)\s+)?cases?\s+"
    r"(?:of|where))|"
    r"however|but|unless|subject(?:\s*,?\s*however\s*,?)?\s+to|"
    r"notwithstanding|(?:the|such|a|an)\s+(?:\w+\s+){0,8}"
    r"(?:shall|must|may|is|are|equals?))\b",
    flags=re.IGNORECASE,
)
_FORMULA_NONNEGATIVE_FLOOR_CONTROL = (
    r"(?:(?:can|could|may|might|must|shall|should|will|would)\s+|"
    r"(?:is|are)\s+(?:(?:designed|expected|intended|required|supposed)\s+)?"
    r"to\s+|"
    r"(?:is|are)\s+|(?:has|have|had)\s+(?:been\s+)?(?:"
    r"(?:always|at\s+all\s+times)\s+)?)?(?:"
    r"(?:(?:in\s+(?:no\s+event|no\s+case)\s+|never\s+)?"
    r"(?:(?:not\s+)?(?:be\s+)?|be\s+not\s+)(?:(?:less|lower)\s+than|below|"
    r"no\s+(?:less|lower)\s+than|at\s+least|"
    r"greater\s+than\s+or\s+equal\s+to|equal\s+to\s+or\s+greater\s+than)\s+"
    r"(?:zero|\$?\s*0(?:\.0+)?))|"
    r"(?:(?:always|at\s+all\s+times)\s+)?(?:remain|stay)(?:s|ed)?\s+"
    r"(?:(?:at\s+least|at\s+or\s+above|no\s+(?:less|lower)\s+than|"
    r"greater\s+than\s+or\s+equal\s+to)\s+"
    r"(?:zero|\$?\s*0(?:\.0+)?)|(?:at\s+)?zero\s+or\s+"
    r"(?:above|greater|higher|more))|"
    r"not\s+(?:be\s+(?:an?\s+)?negative(?:\s+amount)?|"
    r"(?:become|turn)\s+negative|"
    r"(?:drop|go|sink)\s+below\s+(?:zero|\$?\s*0(?:\.0+)?)|"
    r"fall\s+below\s+(?:zero|\$?\s*0(?:\.0+)?)|"
    r"result\s+in\s+(?:an?\s+)?negative\s+(?:amount|balance|value))|"
    r"never\s+(?:(?:be|become|remain(?:s|ed)?|turn)\s+)?negative|"
    r"(?:(?:always|never|at\s+all\s+times|in\s+all\s+cases)\s+)?"
    r"(?:(?:be|remain(?:s|ed)?|stay(?:s|ed)?)\s+)?"
    r"(?:not\s+negative|non-?negative)"
    r"(?:\s+(?:at\s+all\s+times|throughout))?|"
    r"(?:does?|do)\s+not\s+(?:become|turn)\s+negative|"
    r"(?:at\s+no\s+time|in\s+no\s+(?:event|case)|never|"
    r"under\s+no\s+circumstances)\s+(?:be|become|turn)\s+negative|"
    r"(?:be\s+)?maintained\s+at\s+no\s+(?:less|lower)\s+than\s+"
    r"(?:zero|\$?\s*0(?:\.0+)?)|"
    r"be\s+(?:(?:treated\s+as|deemed|set\s+(?:at|to))\s+)?zero\s+"
    r"(?:if|when|whenever)\s+(?:(?:(?:it|they)|(?:the\s+)?"
    r"(?:amounts?|benefits?|credits?|incomes?|results?|taxes?|totals?|values?))"
    r"\s+(?:is|are)\s+)?negative|"
    r"(?:at\s+no\s+time|in\s+no\s+(?:event|case)|"
    r"under\s+no\s+circumstances|never)\s+"
    r"(?:(?:can|could|may|might|must|shall|should|will|would|is|are)\s+)?"
    r"(?:(?:it|they)\s+|(?:(?:a|an|any|each|every|the|this|that|such)\s+)?"
    r"(?:[A-Za-z][A-Za-z-]*\s+){0,5}"
    r"(?:amounts?|benefits?|credits?|deductions?|incomes?|"
    r"liabilit(?:y|ies)|payments?|results?|taxes?|totals?|values?)\s+)?"
    r"(?:(?:be|become|turn)\s+negative|"
    r"be\s+(?:less|lower)\s+than\s+(?:zero|\$?\s*0(?:\.0+)?)|"
    r"fall\s+below\s+(?:zero|\$?\s*0(?:\.0+)?)|"
    r"result\s+in\s+(?:an?\s+)?negative\s+(?:amount|balance|value))|"
    r"never\s+(?:carry|have|show)\s+(?:an?\s+)?negative\s+"
    r"(?:amount|balance|value)|"
    r"be\s+(?:zero|\$?\s*0(?:\.0+)?)\s+or\s+(?:above|greater|higher|more)|"
    r"(?:have|has|had)\s+(?:(?:a\s+)?(?:floor|lower\s+bound|"
    r"minimum(?:\s+value)?)\s+(?:at|of)\s+(?:zero|\$?\s*0(?:\.0+)?)|"
    r"(?:a\s+)?(?:zero|\$?\s*0(?:\.0+)?)\s+(?:as\s+a\s+)?lower\s+bound)|"
    r"be\s+bounded\s+(?:from\s+)?below\s+(?:at|by)\s+"
    r"(?:zero|\$?\s*0(?:\.0+)?)|"
    r"(?:cannot|can\s+not)\s+(?:(?:be|become|turn)\s+negative|"
    r"(?:drop|fall|go)\s+below\s+(?:zero|\$?\s*0(?:\.0+)?)|"
    r"(?:carry|have|show)\s+(?:an?\s+)?negative\s+"
    r"(?:amount|balance|value)))"
)
_FORMULA_ADMINISTRATIVE_TITLE_LANGUAGE = re.compile(
    r"\b(?:acts?|agreements?|administration|amendments?|appendices|appendix|"
    r"briefs?|bulletins?|bylaws?|"
    r"archives?|books?|brochures?|catalogs?|chapters?|circulars?|codes?|"
    r"contracts?|criteria|decrees?|directives?|documents?|"
    r"dashboards?|databases?|decisions?|documentation|exhibits?|factsheets?|"
    r"faqs?|files?|flyers?|forms?|guidance|infographics?|"
    r"guides?|guidelines|handbooks?|"
    r"instructions?|letters?|"
    r"laws?|legislation|manuals?|memos?|memorand(?:a|ums?)|methodolog(?:y|ies)|"
    r"news|newsletters?|notes?|notices?|operations?|opinions?|orders?|"
    r"ordinances?|pamphlets?|primers?|"
    r"plans?|polic(?:y|ies)|procedures?|processes?|programs?|protocols?|"
    r"overviews?|portals?|press|publications?|records?|registers?|regulations?|"
    r"references?|releases?|"
    r"reports?|repositories?|requirements?|rules?|rulings?|sites?|summaries?|"
    r"tutorials?|"
    r"websites?|"
    r"advisor(?:y|ies)|articles?|papers?|resolutions?|schedules?|sections?|"
    r"specifications?|standards?|"
    r"statutes?|stud(?:y|ies)|tax|whitepapers?|work|workbooks?|workpapers?|"
    r"worksheets?|workflows?)\b",
    flags=re.IGNORECASE,
)
_FORMULA_ADMINISTRATIVE_TITLE_QUALIFIER_LANGUAGE = re.compile(
    r"\b(?:archived|attachments?|contents?|copies|copy|data|drafts?|excerpts?|"
    r"final|pages?|revs?|revisions?|supplements?|texts?|versions?)\b",
    flags=re.IGNORECASE,
)
_FORMULA_CODE_NAMESPACE_TITLE_LANGUAGE = re.compile(
    r"\b(?:polic(?:y|ies)|rules?|schedules?|sections?|tax|work)\b",
    flags=re.IGNORECASE,
)
_FORMULA_ROUNDING_ACTOR_ROLE_LANGUAGE = re.compile(
    r"\b(?:administrators?|agenc(?:y|ies)|authorit(?:y|ies)|boards?|bureaus?|"
    r"accountants?|advisers?|advisors?|agents?|analysts?|assessors?|attorneys?|"
    r"auditors?|clerks?|collectors?|consultants?|contractors?|inspectors?|"
    r"judges?|liaisons?|preparers?|"
    r"commissions?|commissioners?|committees?|"
    r"councils?|"
    r"chairs?|coordinators?|counsels?|custodians?|departments?|directors?|"
    r"divisions?|employees?|examiners?|leads?|managers?|officers?|offices?|"
    r"representatives?|secretar(?:y|ies)|services?|specialists?|supervisors?|"
    r"trustees?)\b",
    flags=re.IGNORECASE,
)
_FORMULA_RESULT_OPERATION_LANGUAGE = re.compile(
    rf"\b{_FORMULA_RESULT_PREDICATE}\s+(?:the\s+)?(?:"
    r"min|max|minimum|maximum|least|greatest|lesser|greater|lower|higher|"
    r"lowest|highest|smallest|largest|smaller|larger|sum|total|average|mean|"
    r"median|ratio|quotient|difference|product|remainder|percentage|percent)\s+"
    r"(?:amount\s+)?(?:of|between)\b|"
    rf"\b{_FORMULA_RESULT_PREDICATE}\s+(?:the\s+)?(?:"
    r"addition\s+of\b[^.;:\n]{1,120}\band\b|"
    r"(?:subtraction|deduction)\s+of\b[^.;:\n]{1,120}\bfrom\b|"
    r"(?:reduction|increase|decrease|division|multiplication)\s+of\b"
    r"[^.;:\n]{1,120}\bby\b)",
    flags=re.IGNORECASE,
)
_FORMULA_DIRECT_FOLLOWING_OPERANDS = re.compile(
    r"\b(?:addition|average|difference|greater|greatest|higher|highest|larger|"
    r"largest|least|lesser|lower|lowest|max|maximum|mean|median|min|minimum|"
    r"product|quotient|ratio|remainder|smaller|smallest|sum|total)\s+"
    r"(?:of|between)\s+(?:the\s+)?following\s+"
    r"(?:amounts?|percentages?|rates?|values?)\b",
    flags=re.IGNORECASE,
)
_FORMULA_PARTICIPIAL_RESULT_LANGUAGE = re.compile(
    r"\b(?:(?:is|are|shall\s+be|must\s+be|may\s+be)\s+)?"
    r"(?:reduced|deducted|increased|decreased)\s+by\b",
    flags=re.IGNORECASE,
)
_FORMULA_NUMERIC_RESULT_HEADS = frozenset(
    {
        "allowance",
        "amount",
        "addition",
        "assessment",
        "average",
        "balance",
        "base",
        "benefit",
        "bracket",
        "cap",
        "ceiling",
        "charge",
        "count",
        "credit",
        "decrease",
        "deduction",
        "denominator",
        "deficiency",
        "distribution",
        "difference",
        "divisor",
        "excess",
        "exemption",
        "expense",
        "factor",
        "fee",
        "floor",
        "fraction",
        "income",
        "increase",
        "liability",
        "limit",
        "loss",
        "margin",
        "mean",
        "median",
        "networth",
        "number",
        "numerator",
        "offset",
        "overpayment",
        "payment",
        "percent",
        "percentage",
        "product",
        "proceeds",
        "quantity",
        "quotient",
        "rate",
        "ratio",
        "rebate",
        "reduction",
        "refund",
        "remainder",
        "result",
        "shortfall",
        "surcharge",
        "surtax",
        "sum",
        "tax",
        "threshold",
        "total",
        "value",
        "withholding",
        "worth",
    }
)
_FORMULA_NUMERIC_RESULT_HEAD_PATTERN = "|".join(
    sorted(_FORMULA_NUMERIC_RESULT_HEADS, key=len, reverse=True)
)
_FORMULA_SUBJECT_PHRASE_BREAK = re.compile(
    r"\b(?:of|for|to|under|by|from|in|on|with|without|that|which|who|whose|"
    r"before|after|during|throughout|within)\b",
    flags=re.IGNORECASE,
)
_FORMULA_SUBJECT_TRAILING_MODIFIERS = frozenset(
    {
        "adjusted",
        "allocable",
        "allowed",
        "allowable",
        "applicable",
        "attributable",
        "available",
        "authorized",
        "calculated",
        "computed",
        "currently",
        "described",
        "derived",
        "determined",
        "due",
        "earned",
        "hereby",
        "imposed",
        "incurred",
        "herein",
        "hereunder",
        "now",
        "otherwise",
        "paid",
        "payable",
        "permitted",
        "properly",
        "prescribed",
        "provided",
        "pursuant",
        "received",
        "reported",
        "responsible",
        "shown",
        "so",
        "specified",
        "set",
        "subject",
        "taxable",
        "forth",
        "then",
        "thereon",
        "previously",
        "below",
        "duly",
        "legally",
        "levied",
        "realized",
        "recognized",
        "assessed",
        "worked",
    }
)
_FORMULA_SUBJECT_LEADING_NONHEAD_WORDS = frozenset(
    {
        "a",
        "adjusted",
        "allowable",
        "an",
        "applicable",
        "earned",
        "federal",
        "gross",
        "imposed",
        "individual",
        "operating",
        "qualified",
        "resident",
        "state",
        "taxable",
        "the",
    }
)
_FORMULA_SUBJECT_PARENTHETICAL = re.compile(
    r"(?:as|adjusted|calculated|computed|described|determined|if|provided|"
    r"specified|subject|when|where)\b",
    flags=re.IGNORECASE,
)
_FORMULA_LEGAL_PROVISION_REFERENCE = (
    r"(?:"
    r"(?:(?:this|the)\s+)?(?:section|subsection|paragraph|chapter|part|title|"
    r"article|subdivision)"
    r"(?:\s+(?:(?:\([^)]+\))+|[A-Z]+|[IVXLCDM]+|\d+(?:\.\d+)*))?|"
    r"R\.?\s*S\.?\s*\d+[A-Za-z]?\s*:\s*\d+(?:\.\d+)*(?:\([^)]+\))*|"
    r"§+\s*\d+[A-Za-z]?\s*:\s*\d+(?:\.\d+)*(?:\([^)]+\))*)"
)
_FORMULA_SUBJECT_PREAMBLE_PREFIX = re.compile(
    rf"(?:"
    rf"(?:for\s+purposes\s+of|pursuant\s+to|under|according\s+to|"
    rf"in\s+accordance\s+with|except\s+under|to\s+the\s+extent\s+provided\s+in)"
    rf"\s+{_FORMULA_LEGAL_PROVISION_REFERENCE}|"
    rf"subject\s+to\s+(?:(?:the\s+)?(?:limitations?|provisions?|requirements?)"
    rf"\s+of\s+)?{_FORMULA_LEGAL_PROVISION_REFERENCE}|"
    rf"(?:except\s+)?as\s+(?:otherwise\s+)?provided\s+(?:in|by)\s+"
    rf"{_FORMULA_LEGAL_PROVISION_REFERENCE}|"
    r"for\s+(?:taxable|tax|calendar|fiscal)\s+years?[^,.;:\n]*?"
    r"(?:18|19|20)\d{2}|"
    r"effective\s+for\s+(?:taxable|tax|calendar|fiscal)\s+years?"
    r"[^,.;:\n]*?(?:18|19|20)\d{2}|"
    r"(?:after|beginning\s+after)\s+(?:18|19|20)\d{2}|"
    r"(?:effective\s+)?on\s+or\s+after\s+[A-Za-z]+\s+\d{1,2}\s*,?\s*"
    r"(?:18|19|20)\d{2}|"
    r"beginning\s+on\s+[A-Za-z]+\s+\d{1,2}\s*,?\s*(?:18|19|20)\d{2}"
    r")(?:\s*,\s*|\s+)",
    flags=re.IGNORECASE,
)
_FORMULA_NUMERIC_OPERAND_HEADS = frozenset(
    {
        "allowance",
        "amount",
        "asset",
        "average",
        "base",
        "basis",
        "benefit",
        "bonus",
        "cap",
        "coefficient",
        "contribution",
        "credit",
        "deduction",
        "difference",
        "dividend",
        "divisor",
        "dollar",
        "denominator",
        "expense",
        "earning",
        "factor",
        "fee",
        "formula",
        "fraction",
        "gain",
        "hour",
        "income",
        "index",
        "interest",
        "liability",
        "limit",
        "loss",
        "margin",
        "mean",
        "median",
        "month",
        "multiplier",
        "numerator",
        "number",
        "offset",
        "payment",
        "percent",
        "percentage",
        "product",
        "proceeds",
        "rate",
        "ratio",
        "receipt",
        "salary",
        "schedule",
        "supplement",
        "surcharge",
        "table",
        "tax",
        "threshold",
        "total",
        "unit",
        "value",
        "voucher",
        "wage",
        "quotient",
        "remainder",
        "sum",
    }
)
_FORMULA_NUMERIC_OPERAND_LITERAL_CORE = (
    rf"(?:(?:[$€£]\s*)?-?\d+(?:,\d{{3}})*(?:\.\d+)?|"
    rf"{_ENGLISH_NUMBER_WORD}"
    rf"(?:(?:[-\s]+(?:and[-\s]+)?){_ENGLISH_NUMBER_WORD})*)"
    r"(?:\s+(?:dollars?|cents?|percent(?:age)?(?:\s+points?)?|hours?|units?))?"
)
_FORMULA_NUMERIC_OPERAND_LITERAL = re.compile(
    rf"\s*(?:the\s+)?(?:\(\s*{_FORMULA_NUMERIC_OPERAND_LITERAL_CORE}\s*\)|"
    rf"{_FORMULA_NUMERIC_OPERAND_LITERAL_CORE})\s*",
    flags=re.IGNORECASE,
)
_FORMULA_ADMINISTRATIVE_PARTICIPLE = frozenset(
    {"administering", "handling", "providing"}
)
_FORMULA_OPERAND_IRREGULAR_POSTMODIFIERS = frozenset(
    {
        "borne",
        "bought",
        "built",
        "caught",
        "dealt",
        "done",
        "drawn",
        "driven",
        "exempt",
        "felt",
        "found",
        "given",
        "grown",
        "held",
        "kept",
        "known",
        "laid",
        "led",
        "left",
        "lost",
        "made",
        "paid",
        "read",
        "run",
        "said",
        "set",
        "shown",
        "sold",
        "spent",
        "taught",
        "told",
        "understood",
        "withheld",
        "won",
        "written",
    }
)
_VALID_ROMAN_OUTLINE_LABEL = re.compile(
    r"(?=[ivxlcdm]+\Z)m{0,3}(?:cm|cd|d?c{0,3})"
    r"(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})",
    flags=re.IGNORECASE,
)
_FORMULA_APPLICABILITY_YEAR = r"(?:18|19|20)\d{2}"
_FORMULA_APPLICABILITY_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\.?"
)
_FORMULA_APPLICABILITY_DAY = r"(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?"
_FORMULA_APPLICABILITY_DATE = (
    rf"(?:{_FORMULA_APPLICABILITY_MONTH}\s+"
    rf"(?:{_FORMULA_APPLICABILITY_DAY}\s*,?\s*)?"
    rf"{_FORMULA_APPLICABILITY_YEAR}|{_FORMULA_APPLICABILITY_YEAR})"
)
_FORMULA_APPLICABILITY_BOUNDARY = (
    r"(?:after|before|(?:no|not)\s+(?:earlier|later)\s+than|"
    r"prior\s+to|subsequent\s+to|"
    r"on(?:\s+(?:or|and)\s+(?:after|before))?)"
)
_FORMULA_APPLICABILITY_START = (
    r"(?:beginning|commencing|starting|(?:that|which)\s+"
    r"(?:begin|begins|commence|commences|start|starts))"
)
_FORMULA_APPLICABILITY_RANGE_END = (
    rf"(?:,?\s*(?:(?:and|but)\s+(?:ending\s+)?"
    rf"{_FORMULA_APPLICABILITY_BOUNDARY}|through|to|"
    rf"up\s+to\s+and\s+including)\s+"
    rf"{_FORMULA_APPLICABILITY_DATE})"
)
_FORMULA_APPLICABILITY_YEAR_SPAN = (
    rf"(?>{_FORMULA_APPLICABILITY_YEAR}(?:(?:\s*,\s*"
    rf"(?:(?:and|or)\s+)?|\s+(?:and|or|through|to)\s+|\s*[-–]\s*)"
    rf"{_FORMULA_APPLICABILITY_YEAR})*)"
)
_FORMULA_APPLICABILITY_YEAR_SERIES = (
    rf"(?:from\s+)?{_FORMULA_APPLICABILITY_YEAR_SPAN}"
    rf"(?:\s*,?\s*(?:inclusive|and\s+(?:thereafter|later|onward)))?"
)
_FORMULA_APPLICABILITY_PREFACE = re.compile(
    rf"^\s*(?:(?:\([^)]+\)|[A-Z]\.)\s*)*(?:"
    rf"(?:effective\s+)?(?:for|during)\s+(?:the\s+)?"
    rf"(?:tax(?:able)?|calendar|fiscal|assessment)\s+years?\s+"
    rf"{_FORMULA_APPLICABILITY_START}\s+"
    rf"(?:(?:{_FORMULA_APPLICABILITY_BOUNDARY}|in)\s+)?"
    rf"{_FORMULA_APPLICABILITY_DATE}"
    rf"(?:{_FORMULA_APPLICABILITY_RANGE_END})?\s*,?|(?:"
    rf"(?:effective\s+)?(?:for|during)\s+(?:the\s+)?"
    rf"(?:tax(?:able)?|calendar|fiscal|assessment)\s+years?\s+"
    rf"ending\s+(?:(?:{_FORMULA_APPLICABILITY_BOUNDARY}|in)\s+)?"
    rf"{_FORMULA_APPLICABILITY_DATE}"
    rf"(?:{_FORMULA_APPLICABILITY_RANGE_END})?\s*,?|"
    rf"{_FORMULA_APPLICABILITY_START}\s+"
    rf"(?:{_FORMULA_APPLICABILITY_BOUNDARY}\s+)?"
    rf"{_FORMULA_APPLICABILITY_DATE}"
    rf"(?:,?\s+and\s+thereafter)?|"
    rf"(?:for|during)\s+(?:the\s+)?"
    rf"(?:tax(?:able)?|calendar|fiscal|assessment)\s+years?\s+"
    rf"{_FORMULA_APPLICABILITY_YEAR_SERIES}|"
    rf"(?:for|during)\s+(?:the\s+)?"
    rf"(?:tax(?:able)?|calendar|fiscal|assessment)\s+years?\s+"
    rf"{_FORMULA_APPLICABILITY_BOUNDARY}\s+{_FORMULA_APPLICABILITY_DATE}"
    rf"(?:{_FORMULA_APPLICABILITY_RANGE_END})?|"
    rf"(?:effective(?:\s+(?:on|from))?|starting(?:\s+(?:on|from))?|"
    rf"as\s+of|on\s+or\s+after)\s+{_FORMULA_APPLICABILITY_DATE}"
    rf")(?:(?:\s*,)|(?=\s*[.;:\n]|$)))",
    flags=re.IGNORECASE,
)
_STATED_CONVERSION_CUE = re.compile(
    r"\b(?:umgerechnet|converted)\b",
    flags=re.IGNORECASE,
)
_STATED_CONVERSION_RESULT = re.compile(
    r"\b(?P<verb>"
    r"ergibt\s+sich|ergeben\s+sich|entspricht|entsprechen|beträgt|betragen|"
    r"equals?|results?\s+in|is|are"
    r")\b"
    r"[^.!?\n]{0,80}?"
    r"(?P<value>\d{1,3}(?:[ .]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)",
    flags=re.IGNORECASE,
)
_STATED_CONVERSION_BASE_VALUE = re.compile(
    r"\d{1,3}(?:[ .]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?"
)
_STATED_CONVERSION_GERMAN_MONTH = (
    r"(?:jan(?:uar)?|feb(?:ruar)?|märz|maerz|mrz|apr(?:il)?|mai|"
    r"jun(?:i)?|jul(?:i)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"okt(?:ober)?|nov(?:ember)?|dez(?:ember)?)\.?"
)
_STATED_CONVERSION_ENGLISH_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_STATED_CONVERSION_DATE = re.compile(
    rf"(?:"
    rf"(?<!\w)\d{{1,2}}\.\s*{_STATED_CONVERSION_GERMAN_MONTH}\s+"
    rf"(?:des\s+Jahres\s+)?(?:18|19|20)\d{{2}}|"
    rf"\b{_STATED_CONVERSION_ENGLISH_MONTH}\s+\d{{1,2}}"
    rf"(?:st|nd|rd|th)?\s*,?\s*(?:18|19|20)\d{{2}}|"
    rf"(?<!\w)\d{{1,2}}(?:st|nd|rd|th)?\s+"
    rf"{_STATED_CONVERSION_ENGLISH_MONTH}\s+(?:18|19|20)\d{{2}}|"
    rf"(?<!\w)\d{{1,2}}[./]\d{{1,2}}[./]\d{{2,4}}(?!\w)|"
    rf"(?<!\w)(?:18|19|20)\d{{2}}-\d{{2}}-\d{{2}}(?!\w)"
    rf")",
    flags=re.IGNORECASE,
)
_ROUNDING_LANGUAGE = re.compile(
    r"\b(?:"
    r"abgerundet(?:e|en|er|es)?|abzurunden|aufgerundet(?:e|en|er|es)?|"
    r"aufzurunden|gerundet(?:e|en|er|es)?|zu\s+runden|"
    r"kaufmännisch(?:\s+zu)?\s+runden|"
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
    r"\b(?:gerundet(?:e|en|er|es)?|zu\s+runden|"
    r"kaufmännisch(?:\s+zu)?\s+runden|"
    r"round(?:ed|ing)?\s+to\s+the\s+nearest)\b",
    flags=re.IGNORECASE,
)
_EXCEPTION_LANGUAGE = re.compile(
    r"\b(?:"
    r"except(?:ion)?|unless|subject\s+to|shall\s+not\s+apply|"
    r"does\s+not\s+apply|notwithstanding|vorbehaltlich|ausnahme|"
    r"es\s+sei\s+denn|gilt\b[^.;]{0,80}\bnicht(?:\s*,?\s*wenn)?|"
    r"findet\s+keine\s+anwendung|soweit\s+nicht|"
    r"außer|ausgenommen|abweichend\s+von|jedoch\s+nicht|"
    r"(?:[1-9]\d?)?voraussetzung[^.;]{0,160}\bnicht\b"
    r")",
    flags=re.IGNORECASE,
)
_APPLICABILITY_LANGUAGE = re.compile(
    r"\b(?:"
    r"vorausgesetzt\s*,?\s+dass|"
    r"unter\s+der\s+voraussetzung\s*,?\s+dass|"
    r"wenn|falls|sofern|soweit|when|if|"
    r"bei\s+(?:(?:vorliegen|bestehen)\b|"
    r"(?:(?:einer?|bestehender)\s+)?(?:anspruchsberechtigung|berechtigung)\b)"
    r")\b",
    flags=re.IGNORECASE,
)
_SOURCE_SELECTOR_TOKEN_STOPWORDS = frozenset(
    {
        "applies",
        "apply",
        "case",
        "condition",
        "except",
        "federal",
        "flag",
        "for",
        "has",
        "is",
        "no",
        "non",
        "not",
        "status",
        "the",
        "without",
    }
)
_NEGATIVE_NONAPPLICABILITY_LANGUAGE = re.compile(
    r"\b(?:"
    r"(?:shall|does)\s+not\s+apply|"
    r"(?:is|are)\s+not\s+(?:eligible|qualified|allowed|entitled)|"
    r"findet\s+keine\s+anwendung|"
    r"gilt\b[^.;]{0,80}\bnicht|"
    r"nicht\s+berechtigt|"
    r"kein(?:e|en|em|er|es)?\s+(?:anspruch|berechtigung)"
    r")\b",
    flags=re.IGNORECASE,
)
_LOCAL_CONDITION_LANGUAGE = re.compile(
    r"\b(?:"
    r"if|when|where|unless|except|in\s+the\s+case\s+of|"
    r"wenn|falls|sofern|soweit|bei|ohne|mangels|im\s+falle|"
    r"au(?:ß|ss)er|es\s+sei\s+denn"
    r")\b",
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
_SAME_ACT_SECTION_DEPENDENCY = re.compile(
    r"\bsection\s+(?P<section>\d+[a-z]?)\s+of\s+(?P<act>this|the)\s+act\b",
    flags=re.IGNORECASE,
)
_USC_DEFERRAL_DEPENDENCY = re.compile(
    r"\b(?P<title>\d+)\s+U\.?\s*S\.?\s*C\.?(?:\s*[,;:])?\s*"
    r"(?:(?:§{1,2}|sections?)\s*)?"
    r"(?P<section>\d+[a-z0-9]*(?:[-\u2010\u2011\u2012\u2013\u2014\u2015"
    r"\u2212\ufe58\ufe63\uff0d]\d+)?)"
    r"(?P<tail>(?:\s*\(\s*[A-Za-z0-9]+\s*\))*)",
    flags=re.IGNORECASE,
)
_REVERSED_USC_DEFERRAL_DEPENDENCY = re.compile(
    r"(?:\bsections?|§{1,2})\s*"
    r"(?P<section>\d+[a-z0-9]*(?:[-\u2010\u2011\u2012\u2013\u2014\u2015"
    r"\u2212\ufe58\ufe63\uff0d]\d+)?)"
    r"(?P<tail>(?:\s*\(\s*[A-Za-z0-9]+\s*\))*)"
    r"\s*,?\s*(?:of|in|as\s+codified\s+in)\s+title\s+(?P<title>\d+)\b",
    flags=re.IGNORECASE,
)
_TITLE_FIRST_USC_DEFERRAL_DEPENDENCY = re.compile(
    r"\btitle\s+(?P<title>\d+)\s*,?\s*(?:sections?|§{1,2})\s*"
    r"(?P<section>\d+[a-z0-9]*(?:[-\u2010\u2011\u2012\u2013\u2014\u2015"
    r"\u2212\ufe58\ufe63\uff0d]\d+)?)"
    r"(?P<tail>(?:\s*\(\s*[A-Za-z0-9]+\s*\))*)",
    flags=re.IGNORECASE,
)
_RELATIVE_USC_DEFERRAL_DEPENDENCY = re.compile(
    r"\bsections?\s+"
    r"(?P<section>\d+[a-z0-9]*(?:[-\u2010\u2011\u2012\u2013\u2014\u2015"
    r"\u2212\ufe58\ufe63\uff0d]\d+)?)"
    r"(?P<tail>(?:\s*\(\s*[A-Za-z0-9]+\s*\))*)",
    flags=re.IGNORECASE,
)
_LOUISIANA_RS_DEFERRAL_DEPENDENCY = re.compile(
    r"(?<![A-Za-z0-9_])(?:(?:La\.?\s*)|(?:LSA\s*[-\u2010-\u2015]?\s*))?"
    r"R\.?\s*S\.?\s*(?:§{1,2}\s*)?"
    r"(?P<title>\d+[A-Za-z]?)\s*:\s*"
    r"(?P<section>\d+[A-Za-z]?(?:(?:[-\u2010\u2011\u2012\u2013\u2014\u2015"
    r"\u2212\ufe58\ufe63\uff0d][A-Za-z0-9]+)|(?:\.\d+[A-Za-z]?))*)"
    r"(?P<tail>(?:\s*\(\s*[A-Za-z0-9]+\s*\))*"
    r"(?:\s*(?:"
    r",\s*(?:and\s+)?(?:(?:clauses?|divisions?|items?|lines?|paragraphs?|parts?|"
    r"schedules?|subclauses?|subdivisions?|subitems?|subparagraphs?|subparts?|"
    r"subsections?)\s+)?|"
    r"(?:and/or|and|or|plus|&|as\s+well\s+as|together\s+with|along\s+with|"
    r"in\s+(?:conjunction|combination|tandem)\s+with|in\s+relation\s+(?:to|with)|"
    r"as\s+supplemented\s+by|"
    r"combined\s+with|in\s+addition\s+to|as\s+amended\s+by|including|through|to|"
    r"followed\s+by|as\s+provided\s+in|read\s+with|read\s+together\s+with|"
    r"subject\s+to|except\s+as\s+provided\s+in|as\s+(?:modified|qualified)\s+by|"
    r"[-\u2013\u2014])\s*"
    r"(?:(?:clauses?|divisions?|items?|lines?|paragraphs?|parts?|schedules?|"
    r"subclauses?|subdivisions?|subitems?|subparagraphs?|subparts?|subsections?)\s+)?"
    r"|(?:clauses?|divisions?|items?|lines?|paragraphs?|parts?|schedules?|"
    r"subclauses?|subdivisions?|subitems?|subparagraphs?|subparts?|subsections?)\s+"
    r")\(\s*[A-Za-z0-9]+(?:[-\u2013][A-Za-z0-9]+)*\s*\)"
    r"(?!\s+(?:filers?\b|of\s+(?:the\s+)?(?:calculation|explanatory\s+report|"
    r"form|report|return|tax\s+return|worksheet)\b)))*)"
    r"(?!\s*\()(?![A-Za-z0-9_])(?![./:]\s*(?:[A-Za-z0-9]|\())"
    r"(?![-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]"
    r"\s*(?:[A-Za-z0-9]|\())",
    flags=re.IGNORECASE,
)
_MISSING_DEPENDENCY_LANGUAGE = re.compile(
    r"\b(?:"
    r"requires?|depends?\s+on|missing|not\s+yet\s+encoded|unavailable|"
    r"cannot\s+be\s+(?:computed|encoded|resolved)|until|without|"
    r"benötigt|abhängig|fehlt|nicht\s+codiert"
    r")\b",
    flags=re.IGNORECASE,
)
_ADVERSATIVE_LANGUAGE = re.compile(
    r"\b(?:"
    r"although|but|despite|even\s+though|except|however|nevertheless|nonetheless|"
    r"notwithstanding|though|unless|whereas|while|yet|aber|jedoch|obwohl"
    r")\b",
    flags=re.IGNORECASE,
)
_CONTEXTUAL_AUTHORITY_LANGUAGE = re.compile(
    r"\b(?:"
    r"(?:included|mentioned|quoted|summarized|discussed|provided|supplied|known)\s+"
    r"only\s+(?:as|for)\s+(?:an?\s+|the\s+)?"
    r"(?:background|comparison|context|historical|history|illustration|"
    r"legislative\s+history|nonbinding\s+authority|orientation)|"
    r"(?:historical|legislative)\s+(?:background|context|history)|"
    r"nonbinding\s+authority|for\s+(?:context|orientation)|"
    r"(?:merely|only)\s+(?:illustrative|nonbinding)"
    r")\b",
    flags=re.IGNORECASE,
)
_COORDINATED_FINITE_CLAUSE = re.compile(
    r"(?:,\s*)?\b(?:and|but|or|yet)\s+"
    r"(?:the\s+|a\s+|an\s+|this\s+|that\s+|these\s+|those\s+)?"
    r"[a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){0,8}\s+"
    r"(?:can|could|is|are|was|were|has|have|had|does|do|did|may|might|must|"
    r"shall|should|will|would)\b",
    flags=re.IGNORECASE,
)
_DEPENDENCY_CONTEXT_COORDINATION = re.compile(
    r"(?:,\s*)?\b(?:although|and|but|even\s+though|or|though|while|whereas|yet)\s+",
    flags=re.IGNORECASE,
)


def _has_adversative_language(text: str) -> bool:
    without_not_yet = re.sub(
        r"\bnot\s+yet\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return bool(_ADVERSATIVE_LANGUAGE.search(without_not_yet))


_DEPENDENCY_SUBJECT_TERMS = frozenset(
    {
        "amount",
        "assistance",
        "benefit",
        "calendar",
        "calculation",
        "classification",
        "condition",
        "criteria",
        "data",
        "date",
        "designation",
        "determination",
        "document",
        "eligibility",
        "fact",
        "filing",
        "form",
        "guidance",
        "income",
        "information",
        "input",
        "limit",
        "management",
        "notice",
        "plan",
        "policy",
        "payment",
        "participation",
        "process",
        "processing",
        "procedure",
        "program",
        "administration",
        "rate",
        "receipt",
        "record",
        "regulation",
        "retention",
        "requirement",
        "rule",
        "size",
        "standard",
        "status",
        "submission",
        "threshold",
        "timing",
        "verification",
        "workflow",
    }
)
_DEPENDENCY_MODIFIER_TERMS = _DEPENDENCY_SUBJECT_TERMS | {
    "a",
    "an",
    "administrative",
    "adjusted",
    "and",
    "agency",
    "annual",
    "applicable",
    "asset",
    "background-check",
    "failing-score",
    "federal",
    "fiscal",
    "fiscal-year",
    "gross",
    "historical",
    "household",
    "initial",
    "legal",
    "local",
    "of",
    "or",
    "plan",
    "plan-submission",
    "program",
    "public",
    "public-housing",
    "residency",
    "state",
    "tax",
    "taxable",
    "the",
    "troubled",
    "troubled-agency",
    "year",
}
_DEPENDENCY_OBJECT_MODIFIER_TERMS = frozenset(
    {
        "administered",
        "applied",
        "approved",
        "available",
        "authorized",
        "awarded",
        "calculated",
        "cited",
        "defined",
        "described",
        "determined",
        "eligible",
        "senior",
        "established",
        "furnished",
        "issued",
        "implemented",
        "obtained",
        "paid",
        "payable",
        "promulgated",
        "provided",
        "received",
        "referenced",
        "required",
        "set",
        "specified",
        "supplied",
        "verified",
    }
)
_DEPENDENCY_OBJECT_MODIFIER_ADVERBS = frozenset(
    {
        "already",
        "also",
        "not",
        "now",
        "otherwise",
        "still",
        "then",
        "yet",
    }
)
_DEPENDENCY_DETERMINER_TERMS = frozenset(
    {
        "a",
        "all",
        "an",
        "another",
        "any",
        "both",
        "each",
        "either",
        "every",
        "few",
        "her",
        "his",
        "its",
        "many",
        "more",
        "most",
        "my",
        "neither",
        "no",
        "our",
        "one",
        "several",
        "some",
        "that",
        "the",
        "their",
        "these",
        "this",
        "those",
        "your",
    }
)
_DEPENDENCY_ACTOR_AMBIGUOUS_OBJECT_TERMS = frozenset(
    {"benefit", "limit", "process", "program", "rate", "record", "rule"}
)
_DEPENDENCY_ACTOR_ALWAYS_FINITE_PREDICATE_TERMS = frozenset(
    {"based", "condition", "limit", "process", "rate"}
)
_DEPENDENCY_IRREGULAR_PLURAL_TERMS = frozenset({"criteria", "data"})
_DEPENDENCY_NUMBER_AMBIGUOUS_TERMS = frozenset({"data"})
_DEPENDENCY_ZERO_MARKED_FINITE_VERB_TERMS = frozenset(
    {"cost", "cut", "hit", "input", "put", "read", "set", "spread"}
)
_DEPENDENCY_ACTOR_NOMINAL_PREFIX_TERMS = frozenset(
    {
        "amendment",
        "benefit",
        "blind",
        "cash",
        "child",
        "drug",
        "earning",
        "elderly",
        "family",
        "household",
        "immigrant",
        "medical",
        "noncitizen",
        "parent",
        "payment",
        "policy",
        "premium",
        "prescription",
        "program",
        "record",
        "refugee",
        "resource",
        "rule",
        "saving",
        "school",
        "senior",
        "shelter",
        "spenddown",
        "student",
        "utility",
        "wage",
        "waiver",
        "work",
    }
)
_DEPENDENCY_ACTOR_NOMINAL_HEAD_TERMS = frozenset(
    {
        "administration",
        "criteria",
        "eligibility",
        "limit",
        "policy",
        "procedure",
        "program",
        "requirement",
        "rule",
        "standard",
        "threshold",
    }
)
_DEPENDENCY_POLICY_ACRONYM_TERMS = frozenset(
    {
        "abawd",
        "bbce",
        "c",
        "chip",
        "d",
        "dsh",
        "ebt",
        "epsdt",
        "esrd",
        "fmap",
        "fpl",
        "hcbs",
        "hea",
        "hipp",
        "hmo",
        "i",
        "irmaa",
        "ism",
        "lis",
        "lieap",
        "liheap",
        "ltss",
        "ma",
        "magi",
        "mco",
        "moe",
        "msp",
        "pace",
        "pass",
        "pd",
        "pdp",
        "poms",
        "ppo",
        "qdwi",
        "qi",
        "qmb",
        "rsdi",
        "sga",
        "snp",
        "spa",
        "slmb",
        "ssdi",
        "ssp",
        "sua",
        "tfp",
        "tpl",
        "wpr",
    }
)
_DEPENDENCY_POLICY_PROGRAM_SUBJECT_TERMS = frozenset(
    {
        "afdc",
        "chip",
        "liheap",
        "medicaid",
        "medicare",
        "rsdi",
        "snap",
        "ssdi",
        "ssi",
        "tanf",
        "wic",
    }
)
_DEPENDENCY_POLICY_COMPOUND_CONNECTOR_TERMS = frozenset(
    {"based", "non", "part", "waiver"}
)
_DEPENDENCY_POLICY_LEADING_ONLY_TERMS = frozenset(
    {"based", "chip", "pace", "part", "pass"}
)
_DEPENDENCY_COMPOUND_NOMINAL_TERMS = frozenset(
    {
        "administration",
        "calculation",
        "management",
        "participation",
        "payment",
        "processing",
        "retention",
        "verification",
    }
)
_DEPENDENCY_NESTED_NOMINAL_COMPOUNDS = frozenset(
    {
        ("records", "management"),
        ("records", "retention"),
        ("rules", "administration"),
    }
)
_LEGAL_ACTOR_HEAD_TERMS = frozenset(
    {
        "administrator",
        "agency",
        "authority",
        "board",
        "bureau",
        "commission",
        "commissioner",
        "corporation",
        "department",
        "judiciary",
        "office",
        "reserve",
        "secretary",
        "service",
    }
)
_LEGAL_ACTOR_ACRONYMS = frozenset(
    {
        "acf",
        "acl",
        "aphis",
        "atf",
        "bjs",
        "bls",
        "bop",
        "cdc",
        "cfpb",
        "cftc",
        "cia",
        "cms",
        "cpsc",
        "csb",
        "dea",
        "dhs",
        "dod",
        "doe",
        "doj",
        "dol",
        "dot",
        "eeoc",
        "ebsa",
        "epa",
        "faa",
        "fbi",
        "fcc",
        "fda",
        "fdic",
        "fec",
        "fema",
        "fha",
        "fhfa",
        "fincen",
        "fmc",
        "fmcs",
        "fns",
        "fra",
        "ftc",
        "gao",
        "gsa",
        "hhs",
        "hrsa",
        "hud",
        "ice",
        "irs",
        "nara",
        "nasa",
        "ncua",
        "nih",
        "nlrb",
        "nrc",
        "nsa",
        "ntsb",
        "occ",
        "omb",
        "opm",
        "osc",
        "osha",
        "pbgc",
        "sba",
        "samhsa",
        "sec",
        "ssa",
        "tva",
        "tsa",
        "usaid",
        "uscis",
        "usda",
        "usps",
        "usss",
        "ustr",
        "va",
    }
)
_LEGAL_INSTRUMENT_TERMS = frozenset(
    {
        "act",
        "administrative",
        "affordable",
        "assessment",
        "assistance",
        "benefit",
        "benefits",
        "care",
        "choice",
        "code",
        "education",
        "energy",
        "families",
        "fair",
        "federal",
        "food",
        "housing",
        "home",
        "internal",
        "labor",
        "low-income",
        "management",
        "needy",
        "nutrition",
        "patient",
        "procedure",
        "program",
        "protection",
        "public",
        "regulation",
        "regulations",
        "revenue",
        "section",
        "security",
        "social",
        "standards",
        "statute",
        "supplemental",
        "temporary",
        "veterans",
        "voucher",
    }
)
_DEPENDENCY_STATE_VALUE = (
    r"(?:approved|ascertained|available|calculated|computed|determined|encoded|"
    r"established|furnished|implemented|known|made\s+available|missing|"
    r"issued|needed|obtained|produced|provided|received|required|resolved|set|"
    r"supplied|unavailable|verified)"
)
_EXECUTABLE_DEPENDENCY_READY_STATE_VALUE = (
    r"(?:approved|ascertained|available|calculated|computed|determined|encoded|"
    r"established|furnished|implemented|known|made\s+available|obtained|produced|"
    r"provided|received|resolved|set|supplied|verified)"
)
_SOURCE_BOUND_RUNTIME_GAP_LANGUAGE = re.compile(
    r"\b(?:"
    r"administrative|calendar|capabilit(?:y|ies)|classification|complaint|"
    r"content|data|date\s+arithmetic|designation|determination|document|event|"
    r"fact|filing|form|hearing|information|input|investigation|membership|"
    r"notice|process|procedure|record|relation|representation|status|submission|"
    r"timing|workflow"
    r")[a-z-]*\b",
    flags=re.IGNORECASE,
)
_ADMINISTRATIVE_SOURCE_ARTIFACT_LANGUAGE = re.compile(
    r"\b(?:"
    r"amendment|approval|audit|certification|comment|complaint|consultation|"
    r"copy|document|filing|hearing|inspection|notice|plan|procedure|"
    r"recommendation|record|report|return|submission|waiver"
    r")[a-z-]*\b",
    flags=re.IGNORECASE,
)
_ADMINISTRATIVE_SOURCE_ACTION_LANGUAGE = re.compile(
    r"\b(?:"
    r"adopt|approve|certif|consult|contain|conduct|disapprov|enforce|establish|"
    r"file|investigat|notify|prescrib|publish|recommend|review|submit|waiv"
    r")[a-z-]*\b",
    flags=re.IGNORECASE,
)
_SOURCE_BOUND_RUNTIME_GAP_STOPWORDS = frozenset(
    {
        "administrative",
        "available",
        "because",
        "cannot",
        "capability",
        "capabilities",
        "computed",
        "current",
        "encoded",
        "encoding",
        "exact",
        "input",
        "inputs",
        "missing",
        "required",
        "requires",
        "representation",
        "representations",
        "runtime",
        "section",
        "source",
        "statute",
        "under",
        "unavailable",
        "until",
    }
)
_IMPRECISE_DEFERRAL_RETRY_SHAPE = """\
Required shape (adapt the output and cited dependency to the omitted branch):
module:
  deferred_outputs:
    - output: de:statutes/estg/32a/6#surviving_spouse_splitting_tax
      reason: Cannot be computed until the joint-assessment conditions cited in EStG § 26 are encoded.
`output` and `reason` are required; `blocked_by` is optional and, when present, \
must list exact absolute upstream RuleSpec outputs. When no external legal dependency \
exists, cite the exact current source branch and name its concrete source-stated missing \
input or runtime capability in the `reason` itself; the output path is not a source \
citation, and a generic claim that the branch is unavailable is invalid. When the \
authoritative branch is a bare `Repealed.` tombstone or a finite Louisiana \
`Repealed by Acts ...` session-law tombstone, cite that exact current source branch and \
affirmatively state the repeal in `reason`; do not invent a missing dependency or \
executable rule. Use the bounded form `<exact branch citation> is repealed.`, optionally \
adding the exact authenticated history either as `is repealed by <Acts citation>.` or \
as `is repealed. <Acts citation>.`, and/or adding `and supplies no operative rule`. \
For Louisiana delegated rates, use `no executable RuleSpec output \
for those <source-stated modifiers> rates is supplied in the available context`; do \
not repeat the dependency in that missing object or append a relative clause."""


def _imprecise_deferral_retry_shape(
    *,
    corpus_citation_path: str,
    path: tuple[str, ...],
    reason: str = "",
) -> str:
    """Render branch-specific retry guidance without inventing source facts."""

    branch_hint = ""
    if path and corpus_citation_path.startswith("us/statute/"):
        with contextlib.suppress(ValueError):
            citation = parse_usc_citation(corpus_citation_path)
            section = normalize_rulespec_path_segment(citation.section)
            fragments = "".join(
                f"({normalize_rulespec_path_segment(part)})"
                for part in (*citation.fragments, *path)
            )
            branch_hint = (
                "\nFor this rejected current-source branch, the literal citation "
                f"required in `reason` is `{citation.title} U.S.C. "
                f"{section}{fragments}`."
            )
    elif (
        path
        and reason
        and _reason_cites_exact_current_statute_branch(
            reason,
            corpus_citation_path=corpus_citation_path,
            path=path,
            strict_terminal=True,
        )
    ):
        branch_hint = (
            "\nThe current-source branch citation in `reason` is already "
            "recognized; correct the bounded dependency, runtime-gap, or repeal "
            "tail without replacing that authenticated citation."
        )
    elif path:
        citation_parts = [
            part for part in corpus_citation_path.strip("/").split("/") if part
        ]
        if len(citation_parts) >= 3 and citation_parts[1] == "statute":
            fragments = "".join(
                f"({normalize_rulespec_path_segment(part)})" for part in path
            )
            branch_hint = (
                "\nFor this rejected current-source branch, the literal canonical "
                f"citation required in `reason` is "
                f"`{corpus_citation_path.rstrip('/')}{fragments}`."
            )
    return f"{_IMPRECISE_DEFERRAL_RETRY_SHAPE}{branch_hint}"


_ABSATZ_REFERENCE = re.compile(
    r"\b(?:Absatz(?:es)?|Absätze(?:n)?|Abs\.)\s*(?P<label>\d+[a-z]?)\b",
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
    r"§{1,2}\s*\d+(?:\.\d+)*[a-z]?"
    r"(?:\s*,\s*\d+(?:\.\d+)*[a-z]?)*"
    r"(?:\s*(?:und|bis|[-–—])\s*\d+(?:\.\d+)*[a-z]?)*",
    flags=re.IGNORECASE,
)
_EXPLICIT_LEGAL_SECTION_REFERENCE = re.compile(
    r"(?:§{1,2}\s*|\b(?:sections?|paragra(?:f|phs?))\s+)"
    r"(?P<section>\d+[a-z]?)",
    flags=re.IGNORECASE,
)
_ENGLISH_LEGAL_CITATION = re.compile(
    r"\b(?:articles?|sections?|secs?\.?|regulations?|paragraphs?)\s+"
    r"\d+(?:\.\d+)*(?:\s*(?:through|to|[-–—]|and|,)\s*\d+(?:\.\d+)*)*",
    flags=re.IGNORECASE,
)
_TITLE_SUFFIX_LEGAL_CITATION = re.compile(
    r"\b(?:sections?\s+)?(?:\d+)?[a-z]\s*"
    r"[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]"
    r"\s*\d+[a-z]?(?:\s*(?:,\s*(?:(?:and|or)\s+)?|"
    r"(?:and|or|through|to)\s+)(?:\d+)?[a-z]\s*"
    r"[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]"
    r"\s*\d+[a-z]?)*\s+of\s+this\s+title\b",
    flags=re.IGNORECASE,
)
_LOUISIANA_RS_NUMERIC_RECALL_TARGET = (
    r"\d+[A-Za-z]?\s*:\s*"
    r"\d+[A-Za-z]?(?:(?:[-\u2010\u2011\u2012\u2013\u2014\u2015"
    r"\u2212\ufe58\ufe63\uff0d][A-Za-z0-9]+)|(?:\.\d+[A-Za-z]?))*"
    r"(?:\s*\(\s*(?:[A-Za-z]+|\d{1,2})\s*\))*"
)
_LOUISIANA_RS_NUMERIC_RECALL_CITATION = re.compile(
    r"(?<![A-Za-z0-9_])(?:(?:La\.?\s*)|(?:LSA\s*[-\u2010-\u2015]?\s*))?"
    r"R\.?\s*S\.?\s*(?:§{1,2}\s*)?"
    + _LOUISIANA_RS_NUMERIC_RECALL_TARGET
    + r"(?:\s*(?:,\s*(?:(?:and/or|and|or)\s+)?|"
    + r"(?:and/or|and|or|through|to)\s+|&\s*)"
    + _LOUISIANA_RS_NUMERIC_RECALL_TARGET
    + r")*"
    + r"(?![A-Za-z0-9_])(?![./:]\s*(?:[A-Za-z0-9]|\())"
    + r"(?![-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]"
    + r"\s*(?:[A-Za-z0-9]|\())",
    flags=re.IGNORECASE,
)
_LOUISIANA_SESSION_LAW_CITATION = re.compile(
    r"\bActs?(?:\s+of)?\s+\d{4}\s*,?\s*"
    r"(?:"
    r"(?:(?:\d+\s*(?:st|nd|rd|th|d)|"
    r"[A-Za-z]+(?:[-\s]+[A-Za-z]+){0,3})\s+)?"
    r"(?:E\.?\s*S\.?|Ex\.?|Extra\.?|Extraordinary)\s*"
    r"(?:Sess\.?|Session)?\s*,?\s*"
    r")?"
    r"Nos?\.?\s*\d+(?:-\d+)?\b"
    r"(?:\s*(?:,|and)\s*(?:Nos?\.?\s*)?\d+(?:-\d+)?\b)*"
    r"(?:\s*,?\s*(?:"
    r"§§\s*\d+(?:\s*(?:,|and|through|to|[-–—])\s*\d+)*"
    r"|§\s*\d+"
    r"))?",
    flags=re.IGNORECASE,
)
_LOUISIANA_NOTWITHSTANDING_RS_REFERENCE_PREFIX = re.compile(
    r"^\s*notwithstanding\s+"
    r"(?:(?:the\s+)?provisions?\s+of\s+)?"
    r"R\.S\.\s*\d+[A-Za-z]?:\d+[A-Za-z]?"
    r"(?:-\d+[A-Za-z]?(?:\.\d+)*)?\s*[,;]\s*"
    r"(?P<tail>.+)$",
    flags=re.IGNORECASE,
)
_STRUCTURAL_REFERENCE = re.compile(
    r"\b(?:"
    r"Artikel(?:s|n)?|Art\.|"
    r"Absatz(?:es)?|Absätze(?:n)?|Abs\.|"
    r"Satz(?:es)?|Sätze(?:n)?|"
    r"Nummer(?:n)?|Nr\.|Buchstabe(?:n)?|Buchst\."
    r")\s*\d*[a-z]?"
    r"(?:\s*(?:,|und|bis|[-–—])\s*\d+[a-z]?)*\b",
    flags=re.IGNORECASE,
)
_SESSION_LAW_TAIL_START = re.compile(
    r"(?:(?P<history_label>history|source)\s*[:.\-–—]+\s*|"
    r"(?P<action>amended(?:\s+by)?|as\s+amended|added|supplemented|repealed)"
    r"\s+)?"
    r"(?:(?:P\.?\s*L\.?|L\.)\s*)?"
    r"\d{4}\s*,\s*c\.\s*\d+",
    flags=re.IGNORECASE,
)
_SESSION_LAW_ENTRY = re.compile(
    r"\s*"
    r"(?:(?P<history_label>history|source)\s*[:.\-–—]+\s*)?"
    r"(?:(?P<action>amended(?:\s+by)?|as\s+amended|added|supplemented|repealed)"
    r"\s+)?"
    r"(?:(?:P\.?\s*L\.?|L\.)\s*)?"
    r"\d{4}\s*,\s*c\.\s*\d+"
    r"(?:\s*(?:,|\.)\s*(?:s|ss)\.\s*[A-Za-z0-9]+"
    r"(?:[.:\-–—][A-Za-z0-9]+)*"
    r"(?:\s*(?:through|to|[-–—])\s*[A-Za-z0-9]+)?)?"
    r"(?:\s*,\s*eff(?:ective)?\.\s*"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
    r"Aug(?:ust)?|Sept(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+\d{1,2}\s*,\s*\d{4})?"
    r"(?:\s*,\s*operative\s+"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
    r"Aug(?:ust)?|Sept(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+\d{1,2}\s*,\s*\d{4})?"
    r"\s*\.?\s*",
    flags=re.IGNORECASE,
)
_SESSION_LAW_SENTENCE_SEPARATOR = re.compile(
    r"(?<=\d)\.\s+(?=(?:(?:amended(?:\s+by)?|as\s+amended|added|supplemented|repealed)"
    r"\s+)?(?:(?:P\.?\s*L\.?|L\.)\s*)?\d{4}\s*,\s*c\.)",
    flags=re.IGNORECASE,
)
_STANDALONE_HISTORY_LEADER = re.compile(
    r"(?:(?:history|source)\s*[:.\-–—]+\s*"
    r"(?:(?:P\.?\s*L\.?|L\.)\s*)?|"
    r"(?:amended(?:\s+by)?|as\s+amended|added|supplemented|repealed)\s+"
    r"(?:(?:P\.?\s*L\.?|L\.)\s*)?|"
    r"(?:P\.?\s*L\.?|L\.)\s*)"
    r"\d{4}\s*,\s*c\.\s*\d+",
    flags=re.IGNORECASE,
)
_SESSION_LAW_YEAR_CHAPTER = re.compile(
    r"(?:(?:P\.?\s*L\.?|L\.)\s*)?\d{4}\s*,\s*c\.\s*\d+",
    flags=re.IGNORECASE,
)
_ALABAMA_TERMINAL_ACT_HISTORY_ENTRY = re.compile(
    r"\s*Acts?\s+"
    r"(?:"
    r"\d{4}\s*,\s*"
    r"(?:"
    r"(?:(?:\d+\s*(?:st|nd|rd|th|d)|"
    r"[A-Za-z]+(?:[-\s]+[A-Za-z]+){0,3})\s+)?"
    r"(?:E\.?\s*S\.?|Ex\.?|Extra\.?|Extraordinary)\s*"
    r"(?:Sess\.?|Session)\s*,\s*"
    r")?"
    r"Nos?\.?\s*\d+(?:-\d+)?"
    r"|\d{2}-\d+"
    r")"
    r"\s*,\s*p\.?\s*\d+"
    r"(?:\s*,\s*§{1,2}\s*\d+(?:\s*(?:,|and|through|to|[-–—])\s*\d+)*)?"
    r"\s*\.?\s*",
    flags=re.IGNORECASE,
)
_ALABAMA_TERMINAL_CODE_HISTORY_ENTRY = re.compile(
    r"\s*Code\s+\d{4}\s*,\s*T\.?\s*\d+\s*,\s*"
    r"§{1,2}\s*\d+(?:\s*(?:,|and|through|to|[-–—])\s*\d+)*"
    r"\s*\.?\s*",
    flags=re.IGNORECASE,
)
_FORMULA_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def recognize_source_structure(source_text: str) -> tuple[SourceStructureBranch, ...]:
    """Recognize paragraph, list, letter, and glued German sentence markers."""

    branches: list[SourceStructureBranch] = []
    paragraph_segments: list[tuple[tuple[str, ...], int, int, str]] = []
    dotted_matches = _qualified_dotted_subsection_matches(source_text)
    dotted_candidates = _dotted_subsection_boundary_matches(source_text)
    outer_segments = [
        (
            match,
            match.start(),
            next(
                (
                    candidate.start()
                    for candidate in dotted_candidates
                    if candidate.start() > match.start()
                ),
                len(source_text),
            ),
        )
        for match in dotted_matches
    ]

    if outer_segments:
        prefix_end = outer_segments[0][1]
        prefix_matches = list(_PARAGRAPH_MARKER.finditer(source_text, 0, prefix_end))
        for index, match in enumerate(prefix_matches):
            start = match.start()
            end = (
                prefix_matches[index + 1].start()
                if index + 1 < len(prefix_matches)
                else prefix_end
            )
            label = match.group("label").lower()
            text = source_text[start:end].strip()
            if _is_editorial_omission(text):
                continue
            path = (label,)
            branches.append(
                SourceStructureBranch(
                    path, "paragraph", match.group("marker"), text, start, end
                )
            )
            paragraph_segments.append((path, start, end, text))

        for outer_match, outer_start, outer_end in outer_segments:
            outer_label = outer_match.group("label").lower()
            outer_path = (outer_label,)
            outer_text = source_text[outer_start:outer_end].strip()
            if not _is_editorial_omission(outer_text):
                branches.append(
                    SourceStructureBranch(
                        outer_path,
                        "paragraph",
                        outer_match.group("marker"),
                        outer_text,
                        outer_start,
                        outer_end,
                    )
                )
            nested_markers = _nested_parenthesized_outline_markers(
                source_text,
                outer_path=outer_path,
                start=outer_start,
                end=outer_end,
            )
            if not nested_markers:
                paragraph_segments.append(
                    (outer_path, outer_start, outer_end, outer_text)
                )
                continue
            for marker, end in _nested_outline_marker_spans(
                nested_markers,
                outer_end=outer_end,
            ):
                text = source_text[marker.start : end].strip()
                if _is_editorial_omission(text):
                    continue
                branches.append(
                    SourceStructureBranch(
                        marker.path,
                        marker.kind,
                        marker.label,
                        text,
                        marker.start,
                        end,
                    )
                )
                paragraph_segments.append((marker.path, marker.start, end, text))
    else:
        nested_markers = _qualified_parenthesized_legal_outline_markers(source_text)
        if nested_markers:
            for marker, end in _nested_outline_marker_spans(
                nested_markers,
                outer_end=len(source_text),
            ):
                text = source_text[marker.start : end].strip()
                if _is_editorial_omission(text):
                    continue
                branches.append(
                    SourceStructureBranch(
                        marker.path,
                        marker.kind,
                        marker.label,
                        text,
                        marker.start,
                        end,
                    )
                )
                paragraph_segments.append((marker.path, marker.start, end, text))
        else:
            paragraph_matches = list(_PARAGRAPH_MARKER.finditer(source_text))
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
                    SourceStructureBranch(
                        path, "paragraph", match.group("marker"), text, start, end
                    )
                )
                paragraph_segments.append((path, start, end, text))

    if not paragraph_segments:
        paragraph_segments = [((), 0, len(source_text), source_text)]

    legacy_marker_offsets = {
        *(match.start() for match in _NUMBER_MARKER.finditer(source_text)),
        *(match.start() for match in _LETTER_MARKER.finditer(source_text)),
        *(match.start() for match in _GLUED_SENTENCE_MARKER.finditer(source_text)),
        *(match.start() for match in _EXPLICIT_SENTENCE_MARKER.finditer(source_text)),
    }
    owner_paths = _most_specific_segment_paths_at_offsets(
        paragraph_segments,
        legacy_marker_offsets,
    )

    for paragraph_path, paragraph_start, paragraph_end, _ in paragraph_segments:
        paragraph_text = source_text[paragraph_start:paragraph_end]
        number_matches = [
            match
            for match in _NUMBER_MARKER.finditer(paragraph_text)
            if owner_paths.get(paragraph_start + match.start()) == paragraph_path
        ]
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
            letter_matches = [
                match
                for match in _LETTER_MARKER.finditer(container_text)
                if owner_paths.get(container_start + match.start()) == paragraph_path
            ]
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

        sentence_matches = [
            match
            for match in sorted(
                (
                    *_GLUED_SENTENCE_MARKER.finditer(paragraph_text),
                    *_EXPLICIT_SENTENCE_MARKER.finditer(paragraph_text),
                ),
                key=lambda item: item.start(),
            )
            if owner_paths.get(paragraph_start + match.start()) == paragraph_path
        ]
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
                SourceStructureBranch(
                    path, "sentence", f"Satz {label}", text, start, end
                )
            )

    unique = {(branch.path, branch.kind, branch.start): branch for branch in branches}
    return tuple(
        sorted(
            unique.values(),
            key=lambda branch: (branch.start, len(branch.path), branch.kind),
        )
    )


def _most_specific_segment_paths_at_offsets(
    segments: Sequence[tuple[tuple[str, ...], int, int, str]],
    offsets: Iterable[int],
) -> dict[int, tuple[str, ...]]:
    """Assign legacy markers to their deepest active structural segment."""

    ordered_segments = sorted(segments, key=lambda item: item[1])
    active: list[tuple[int, int, int, tuple[str, ...]]] = []
    owners: dict[int, tuple[str, ...]] = {}
    segment_index = 0
    for offset in sorted(set(offsets)):
        while (
            segment_index < len(ordered_segments)
            and ordered_segments[segment_index][1] <= offset
        ):
            path, start, end, _text = ordered_segments[segment_index]
            heapq.heappush(active, (-len(path), -start, end, path))
            segment_index += 1
        while active and active[0][2] <= offset:
            heapq.heappop(active)
        if active:
            owners[offset] = active[0][3]
    return owners


def _qualified_dotted_subsection_matches(source_text: str) -> tuple[re.Match[str], ...]:
    """Return an ordered legal subsection sequence, excluding isolated initials."""

    candidates = tuple(_DOTTED_SUBSECTION_MARKER.finditer(source_text))
    for start, match in enumerate(candidates):
        if match.group("label") != "A":
            continue
        sequence = [match]
        for candidate_index in range(start + 1, len(candidates)):
            candidate = candidates[candidate_index]
            expected = chr(ord(sequence[-1].group("label")) + 1)
            if candidate.group("label") == expected:
                if len(sequence) >= 2 and _dotted_marker_is_ignorable(
                    source_text, candidate
                ):
                    continue
                sequence.append(candidate)
                continue
            if _dotted_marker_is_ignorable(source_text, candidate):
                continue
            break
        if len(sequence) >= 2:
            return tuple(sequence)
    return ()


def _nested_parenthesized_outline_markers(
    source_text: str,
    *,
    outer_path: tuple[str, ...],
    start: int,
    end: int,
) -> tuple[_OutlineMarker, ...]:
    """Resolve attached and line-leading U.S. subsection marker chains."""

    matches = tuple(_PARENTHESIZED_OUTLINE_MARKER.finditer(source_text, start, end))
    markers: list[_OutlineMarker] = []
    active: list[str] = []
    active_roman = False
    for index, match in enumerate(matches):
        labels = tuple(
            label.lower()
            for label in re.findall(r"\(([A-Za-z0-9]+)\)", match.group("marker"))
        )
        if not labels or not all(
            _is_parenthesized_outline_label(label) for label in labels
        ):
            continue
        marker_start = match.start("marker")
        if len(labels) > 1:
            active = []
            for label in labels:
                active.append(label)
                markers.append(
                    _OutlineMarker(
                        marker_start,
                        (*outer_path, *active),
                        "paragraph" if label[0].isdigit() else "letter",
                        f"({label})",
                    )
                )
            active_roman = len(active) >= 3 and _is_roman_outline_label(active[-1])
            continue

        label = labels[0]
        if label[0].isdigit():
            numeric_depth = next(
                (
                    position
                    for position in range(len(active) - 1, -1, -1)
                    if active[position][0].isdigit()
                ),
                None,
            )
            if active and not active[-1][0].isdigit() and label == "1":
                active = [*active, label]
            elif numeric_depth is None:
                active = [*active, label] if active else [label]
            else:
                active = [*active[:numeric_depth], label]
            active_roman = False
        else:
            next_labels = (
                tuple(
                    item.lower()
                    for item in re.findall(
                        r"\(([A-Za-z0-9]+)\)", matches[index + 1].group("marker")
                    )
                )
                if index + 1 < len(matches)
                else ()
            )
            begins_roman_children = (
                label == "i"
                and len(active) >= 2
                and bool(next_labels)
                and next_labels[0] == "ii"
            )
            continues_roman_children = (
                active_roman and len(active) >= 3 and _is_roman_outline_label(label)
            )
            if begins_roman_children:
                active = [*active, label]
                active_roman = True
            elif continues_roman_children:
                active = [*active[:-1], label]
                active_roman = True
            else:
                numeric_depth = next(
                    (
                        position
                        for position in range(len(active) - 1, -1, -1)
                        if active[position][0].isdigit()
                    ),
                    None,
                )
                if active and active[-1][0].isdigit():
                    if label == "a":
                        active = [*active, label]
                    else:
                        parent_letter_depth = next(
                            (
                                position
                                for position in range(len(active) - 2, -1, -1)
                                if not active[position][0].isdigit()
                            ),
                            None,
                        )
                        active = (
                            [*active[:parent_letter_depth], label]
                            if parent_letter_depth is not None
                            else [label]
                        )
                elif numeric_depth is not None:
                    active = [*active[: numeric_depth + 1], label]
                else:
                    active = [label]
                active_roman = False
        markers.append(
            _OutlineMarker(
                marker_start,
                (*outer_path, *active),
                "paragraph" if label[0].isdigit() else "letter",
                f"({label})",
            )
        )
    return tuple(markers)


_PARENTHESIZED_LEGAL_OUTLINE_LEVELS = (
    "lower-alpha",
    "numeric",
    "roman",
    "upper-alpha",
    "numeric",
    "roman",
    "upper-alpha",
    "numeric",
    "roman",
)
_NUMERIC_ROOT_PARENTHESIZED_LEGAL_OUTLINE_LEVELS = (
    "numeric",
    "lower-alpha",
    "roman",
    "upper-alpha",
    "numeric",
    "roman",
    "upper-alpha",
    "numeric",
    "roman",
)


def _qualified_parenthesized_legal_outline_markers(
    source_text: str,
) -> tuple[_OutlineMarker, ...]:
    """Resolve a proven all-parenthesized U.S. legal outline.

    A flat parenthesized scan cannot distinguish repeated labels such as
    ``(a)(3)``, ``(a)(6)(ii)(G)(3)``, and ``(c)(3)``.  Use the conventional
    U.S. legal hierarchy only when sequential root subsections prove that the
    source is an outline; otherwise retain the legacy flat recognizer.
    """

    line_matches = tuple(_PARENTHESIZED_OUTLINE_MARKER.finditer(source_text))
    all_inline_matches = tuple(
        _INLINE_PARENTHESIZED_OUTLINE_MARKER.finditer(source_text)
    )
    flattened_root_starts = _flattened_inline_numeric_root_starts(
        source_text,
        line_matches,
        all_inline_matches,
    )
    inline_matches = tuple(
        match
        for match in all_inline_matches
        if (
            match.start("marker") in flattened_root_starts
            or not _inline_outline_marker_has_reference_context(source_text, match)
        )
    )
    inline_marker_starts = {match.start("marker") for match in inline_matches}
    matches = tuple(
        heapq.merge(
            line_matches,
            inline_matches,
            key=lambda match: match.start("marker"),
        )
    )
    if not matches:
        return ()

    first_labels = tuple(re.findall(r"\(([A-Za-z0-9]+)\)", matches[0].group("marker")))
    outline_levels = (
        _NUMERIC_ROOT_PARENTHESIZED_LEGAL_OUTLINE_LEVELS
        if first_labels and first_labels[0][0].isdigit()
        else _PARENTHESIZED_LEGAL_OUTLINE_LEVELS
    )

    markers: list[_OutlineMarker] = []
    emitted_paths: set[tuple[str, ...]] = set()
    active: list[tuple[int, str, str, str]] = []
    for match in matches:
        raw_labels = tuple(re.findall(r"\(([A-Za-z0-9]+)\)", match.group("marker")))
        if not raw_labels or not all(
            _is_parenthesized_outline_label(label) for label in raw_labels
        ):
            continue
        marker_start = match.start("marker")
        attached_parent_level: int | None = None
        for label_index, raw_label in enumerate(raw_labels):
            level, category = _parenthesized_legal_outline_level(
                raw_label,
                active=active,
                attached_parent_level=attached_parent_level,
                outline_levels=outline_levels,
            )
            if (
                marker_start in inline_marker_starts
                and marker_start not in flattened_root_starts
                and label_index == 0
                and level == 0
            ):
                break
            active = [entry for entry in active if entry[0] < level]
            normalized = raw_label.lower()
            active.append((level, raw_label, normalized, category))
            path = tuple(entry[2] for entry in active)
            is_repeated_attached_ancestor = (
                len(raw_labels) > 1
                and label_index < len(raw_labels) - 1
                and path in emitted_paths
            )
            if not is_repeated_attached_ancestor:
                markers.append(
                    _OutlineMarker(
                        marker_start,
                        path,
                        (
                            "paragraph"
                            if category == "numeric" or level == 0
                            else "letter"
                        ),
                        f"({raw_label})",
                    )
                )
                emitted_paths.add(path)
            attached_parent_level = level

    root_category = outline_levels[0]
    root_labels = [marker.label[1:-1] for marker in markers if len(marker.path) == 1]
    has_sequential_roots = any(
        _parenthesized_outline_label_is_first(first, root_category)
        and _parenthesized_outline_label_follows(first, second, root_category)
        for first, second in itertools.pairwise(root_labels)
    )
    return tuple(markers) if has_sequential_roots else ()


def _flattened_inline_numeric_root_starts(
    source_text: str,
    line_matches: Sequence[re.Match[str]],
    inline_matches: Sequence[re.Match[str]],
) -> frozenset[int]:
    """Recover a proven numeric root sequence from flattened statutory prose.

    Official PDF extraction can place ``(1)`` through ``(N)`` after sentence
    punctuation on one physical line.  Those markers are ambiguous in
    isolation, so restore them only when the first outline candidate is an
    inline ``(1)`` and independently qualified inline candidates prove
    contiguous ``(2)`` and ``(3)`` successors.  Every recovered root must be
    free of direct cross-reference context.  A trailing ``of/under this
    section.`` from the preceding statutory sentence is treated as stale only
    when that sentence has no explicit see/refer/compare command.  Attached
    children and capitalization never override an explicit reference.  The
    initial ``(1)`` remains provisional until those successors are found.
    """

    ordered_matches = tuple(
        heapq.merge(
            line_matches,
            inline_matches,
            key=lambda match: match.start("marker"),
        )
    )
    if not ordered_matches or ordered_matches[0] not in inline_matches:
        return frozenset()
    has_inline_first_root = any(
        tuple(re.findall(r"\(([A-Za-z0-9]+)\)", match.group("marker")))[:1] == ("1",)
        for match in inline_matches
    )
    if not has_inline_first_root:
        return frozenset()

    expected = 1
    selected_starts: list[int] = []
    for match in inline_matches:
        labels = tuple(re.findall(r"\(([A-Za-z0-9]+)\)", match.group("marker")))
        if not labels or re.fullmatch(r"\d+", labels[0]) is None:
            continue
        value = int(labels[0])
        if value == expected:
            candidate_is_proven = (
                _flattened_inline_numeric_first_root_is_proven(source_text, match)
                if expected == 1
                else _flattened_inline_numeric_root_is_proven(source_text, match)
            )
            if candidate_is_proven:
                selected_starts.append(match.start("marker"))
                expected += 1
            continue
        if value > expected and _flattened_inline_numeric_root_is_proven(
            source_text,
            match,
        ):
            break
    return frozenset(selected_starts) if expected >= 4 else frozenset()


def _flattened_inline_numeric_root_has_explicit_reference_command(
    source_text: str,
    match: re.Match[str],
) -> bool:
    prefix = _flattened_inline_numeric_root_immediate_clause(source_text, match)
    return _INLINE_OUTLINE_EXPLICIT_REFERENCE_COMMAND.search(prefix) is not None


def _flattened_inline_numeric_first_root_is_proven(
    source_text: str,
    match: re.Match[str],
) -> bool:
    """Keep the first root provisional without admitting a reference marker."""

    prefix = _flattened_inline_numeric_root_immediate_clause(source_text, match)
    if _INLINE_OUTLINE_EXPLICIT_REFERENCE_COMMAND.search(prefix):
        return False
    has_reference_context = bool(
        _INLINE_OUTLINE_REFERENCE_CONTEXT.search(prefix)
        or _INLINE_OUTLINE_NAMED_REFERENCE_CONTEXT.search(prefix)
    )
    if not has_reference_context:
        return True
    return (
        re.search(
            r"\bKRS\s+\d+[A-Za-z]?(?:\.\d+[A-Za-z]?)*[.!?]\s*$",
            prefix,
            re.IGNORECASE,
        )
        is not None
    )


def _flattened_inline_numeric_root_context_prefix(
    source_text: str,
    match: re.Match[str],
) -> str:
    """Return local context without bleeding through a prior inline marker."""

    marker_start = match.start("marker")
    context_start = max(0, marker_start - 256)
    for previous in _INLINE_PARENTHESIZED_OUTLINE_MARKER.finditer(
        source_text,
        context_start,
        marker_start,
    ):
        context_start = max(context_start, previous.end("marker"))
    return source_text[context_start:marker_start]


def _flattened_inline_numeric_root_immediate_clause(
    source_text: str,
    match: re.Match[str],
) -> str:
    """Return the final clause while preserving periods in legal abbreviations."""

    prefix = _flattened_inline_numeric_root_context_prefix(source_text, match)
    masked = list(prefix)
    abbreviation_patterns = (
        re.compile(r"\b(?:e\.g|i\.e|U\.S)\.", re.IGNORECASE),
        re.compile(r"\b(?:[A-Z]\.){2,}"),
        re.compile(r"\b[A-Z][A-Za-z]{0,7}\.(?=\s+(?:Code|Rev|Stat|Comp|Gen)\b)"),
        re.compile(r"\b(?:Code|Rev|Stat|Comp|Gen)\.(?=\s+\d)"),
    )
    for pattern in abbreviation_patterns:
        for abbreviation in pattern.finditer(prefix):
            for index in range(abbreviation.start(), abbreviation.end()):
                if masked[index] == ".":
                    masked[index] = "\u2024"
    masked_text = "".join(masked)
    content_end = len(masked_text.rstrip())
    boundary_end = content_end
    if content_end and masked_text[content_end - 1] in ".!?":
        boundary_end -= 1
    clause_start = 0
    for boundary in re.finditer(
        r"[.!?]\s+(?=[A-Z\dÄÖÜ\"“])",
        masked_text[:boundary_end],
    ):
        clause_start = boundary.end()
    return prefix[clause_start:content_end]


def _flattened_inline_numeric_root_is_proven(
    source_text: str,
    match: re.Match[str],
) -> bool:
    """Require local structural evidence before restoring one inline root."""

    if _flattened_inline_numeric_root_has_explicit_reference_command(
        source_text,
        match,
    ):
        return False
    prefix = _flattened_inline_numeric_root_immediate_clause(source_text, match)
    if _INLINE_OUTLINE_STRUCTURAL_CHAPEAU_CONTEXT.search(prefix) and not (
        _INLINE_OUTLINE_CHAPEAU_REFERENCE_LEAD.search(prefix)
    ):
        return True
    has_reference_context = bool(
        _INLINE_OUTLINE_REFERENCE_CONTEXT.search(prefix)
        or _INLINE_OUTLINE_NAMED_REFERENCE_CONTEXT.search(prefix)
    )
    if not has_reference_context:
        return True
    stale_section_tail = re.search(
        r"\b(?:of|under)\s+this\s+section[.!?]\s*$",
        prefix,
        re.IGNORECASE,
    )
    if stale_section_tail is not None:
        return True
    return False


def _inline_outline_marker_has_reference_context(
    source_text: str,
    match: re.Match[str],
) -> bool:
    prefix = source_text[max(0, match.start("marker") - 128) : match.start("marker")]
    if _INLINE_OUTLINE_STRUCTURAL_CHAPEAU_CONTEXT.search(prefix) and not (
        _INLINE_OUTLINE_CHAPEAU_REFERENCE_LEAD.search(prefix)
    ):
        return False
    return bool(
        _INLINE_OUTLINE_REFERENCE_CONTEXT.search(prefix)
        or _INLINE_OUTLINE_NAMED_REFERENCE_CONTEXT.search(prefix)
    )


def _parenthesized_legal_outline_level(
    raw_label: str,
    *,
    active: Sequence[tuple[int, str, str, str]],
    attached_parent_level: int | None,
    outline_levels: Sequence[str],
) -> tuple[int, str]:
    categories = _parenthesized_legal_outline_categories(raw_label)
    candidate_levels = tuple(
        level for level, category in enumerate(outline_levels) if category in categories
    )
    if not candidate_levels:
        return 0, categories[0]

    if attached_parent_level is not None:
        level = next(
            (level for level in candidate_levels if level > attached_parent_level),
            candidate_levels[-1],
        )
        return level, outline_levels[level]

    continuing = [
        (level, category)
        for level, previous, _normalized, category in active
        if level in candidate_levels
        and _parenthesized_outline_label_follows(previous, raw_label, category)
    ]
    if continuing:
        return max(continuing, key=lambda item: item[0])

    parent_level = active[-1][0] if active else -1
    first_descending = [
        level
        for level in candidate_levels
        if level > parent_level
        and _parenthesized_outline_label_is_first(
            raw_label,
            outline_levels[level],
        )
    ]
    if first_descending:
        level = first_descending[0]
        return level, outline_levels[level]

    same_category = [
        (level, category)
        for level, _previous, _normalized, category in active
        if category in categories
    ]
    if same_category:
        return max(same_category, key=lambda item: item[0])

    descending = [level for level in candidate_levels if level > parent_level]
    if descending:
        level = descending[0]
        return level, outline_levels[level]

    level = min(candidate_levels, key=lambda item: abs(item - parent_level))
    return level, outline_levels[level]


def _parenthesized_legal_outline_categories(raw_label: str) -> tuple[str, ...]:
    if raw_label[0].isdigit():
        return ("numeric",)
    if raw_label.isupper():
        return ("upper-alpha",)
    categories = ["lower-alpha"]
    if _is_roman_outline_label(raw_label):
        categories.append("roman")
    return tuple(categories)


def _parenthesized_outline_label_follows(
    previous: str,
    current: str,
    category: str,
) -> bool:
    if category == "numeric":
        previous_match = re.fullmatch(r"(\d+)([a-z]?)", previous, re.IGNORECASE)
        current_match = re.fullmatch(r"(\d+)([a-z]?)", current, re.IGNORECASE)
        if previous_match is None or current_match is None:
            return False
        previous_number = int(previous_match.group(1))
        current_number = int(current_match.group(1))
        previous_suffix = previous_match.group(2).lower()
        current_suffix = current_match.group(2).lower()
        return (current_number == previous_number + 1 and not current_suffix) or (
            current_number == previous_number
            and len(current_suffix) == 1
            and (
                (not previous_suffix and current_suffix == "a")
                or (
                    len(previous_suffix) == 1
                    and ord(current_suffix) == ord(previous_suffix) + 1
                )
            )
        )
    if category in {"lower-alpha", "upper-alpha"}:
        return (
            len(previous) == len(current) == 1
            and ord(current.lower()) == ord(previous.lower()) + 1
        )
    if category == "roman":
        previous_value = _roman_outline_value(previous)
        current_value = _roman_outline_value(current)
        return previous_value is not None and current_value == previous_value + 1
    return False


def _parenthesized_outline_label_is_first(label: str, category: str) -> bool:
    if category == "numeric":
        match = re.fullmatch(r"(\d+)([a-z]?)", label, re.IGNORECASE)
        return match is not None and int(match.group(1)) == 1 and not match.group(2)
    if category in {"lower-alpha", "upper-alpha"}:
        return label.lower() == "a"
    if category == "roman":
        return _roman_outline_value(label) == 1
    return False


def _roman_outline_value(label: str) -> int | None:
    if not _is_roman_outline_label(label):
        return None
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    previous = 0
    for character in reversed(label.lower()):
        value = values[character]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total


def _nested_outline_marker_spans(
    markers: Sequence[_OutlineMarker],
    *,
    outer_end: int,
) -> tuple[tuple[_OutlineMarker, int], ...]:
    """Close nested marker spans in one forward stack pass."""

    ends: dict[_OutlineMarker, int] = {}
    active: list[_OutlineMarker] = []
    for start, grouped in itertools.groupby(markers, key=lambda marker: marker.start):
        group = tuple(grouped)
        shallowest_depth = min(len(marker.path) for marker in group)
        while active and len(active[-1].path) >= shallowest_depth:
            ends[active.pop()] = start
        active.extend(group)
    while active:
        ends[active.pop()] = outer_end
    return tuple((marker, ends[marker]) for marker in markers)


def _is_parenthesized_outline_label(label: str) -> bool:
    return bool(
        re.fullmatch(r"\d+[a-z]?", label, flags=re.IGNORECASE)
        or re.fullmatch(r"[a-z]", label, flags=re.IGNORECASE)
        or _is_roman_outline_label(label)
    )


def _is_roman_outline_label(label: str) -> bool:
    return bool(_VALID_ROMAN_OUTLINE_LABEL.fullmatch(label))


def _dotted_subsection_boundary_matches(source_text: str) -> tuple[re.Match[str], ...]:
    """Return structural and joined rejected markers as ownership boundaries."""

    return tuple(
        heapq.merge(
            _DOTTED_SUBSECTION_MARKER.finditer(source_text),
            _JOINED_DOTTED_BOUNDARY_MARKER.finditer(source_text),
            key=lambda match: match.start(),
        )
    )


def _dotted_marker_starts_citation(source_text: str, match: re.Match[str]) -> bool:
    line_end = source_text.find("\n", match.end())
    if line_end < 0:
        line_end = len(source_text)
    remainder = source_text[match.end() : line_end].lstrip()
    return bool(re.match(r"\d+[A-Z]?(?::|\.\d)", remainder, flags=re.IGNORECASE))


def _dotted_marker_is_ignorable(source_text: str, match: re.Match[str]) -> bool:
    if _dotted_marker_starts_citation(source_text, match):
        return True
    line_end = source_text.find("\n", match.end())
    if line_end < 0:
        line_end = len(source_text)
    remainder = source_text[match.end() : line_end].lstrip()
    return match.group("label") == "I" and bool(
        re.match(r"e\.(?:\s|$)", remainder, flags=re.IGNORECASE)
    )


def _is_editorial_omission(text: str) -> bool:
    return bool(_EDITORIAL_OMISSION_ONLY.fullmatch(text))


def _normalize_formula_result_head(head: str) -> str:
    head = head.lower()
    if head in _FORMULA_NUMERIC_RESULT_HEADS:
        return head
    if head.endswith("ies") and head[:-3] + "y" in _FORMULA_NUMERIC_RESULT_HEADS:
        return head[:-3] + "y"
    if head.endswith("es") and head[:-2] in _FORMULA_NUMERIC_RESULT_HEADS:
        return head[:-2]
    if head.endswith("s") and head[:-1] in _FORMULA_NUMERIC_RESULT_HEADS:
        return head[:-1]
    return ""


def _formula_subject_segment_head(subject: str) -> str:
    """Return the grammatical head of one comma-delimited subject segment."""

    subject = _FORMULA_SUBJECT_PHRASE_BREAK.split(subject, maxsplit=1)[0]
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", subject)
    for index, word in enumerate(words):
        lowered = word.lower()
        if not lowered.endswith("ing"):
            continue
        preceding_has_head = any(
            preceding.lower() not in _FORMULA_SUBJECT_LEADING_NONHEAD_WORDS
            for preceding in words[:index]
        )
        following = words[index + 1 :]
        following_has_numeric_head = any(
            _normalize_formula_result_head(candidate) for candidate in following
        )
        if preceding_has_head and (
            lowered in _FORMULA_ADMINISTRATIVE_PARTICIPLE
            or not following_has_numeric_head
        ):
            words = words[:index]
            break
    while len(words) > 1 and (
        words[-1].lower() in _FORMULA_SUBJECT_TRAILING_MODIFIERS
        or words[-1].lower().endswith("ly")
    ):
        words.pop()
    return words[-1].lower() if words else ""


def _formula_result_subject_head(prefix: str) -> str:
    """Return the grammatical head of a formula predicate's subject phrase."""

    subject = _strip_source_clause_marker(prefix).strip()
    preamble = _FORMULA_SUBJECT_PREAMBLE_PREFIX.match(subject)
    if preamble is not None:
        subject = subject[preamble.end() :].strip()

    segments = [segment.strip() for segment in subject.split(",")]
    while segments and not segments[-1]:
        segments.pop()
    if subject.rstrip().endswith(","):
        while len(segments) > 1 and _FORMULA_SUBJECT_PARENTHETICAL.match(segments[-1]):
            segments.pop()
    selected = segments[-1] if segments else ""
    if not _normalize_formula_result_head(_formula_subject_segment_head(selected)):
        for segment in segments[:-1]:
            if not _FORMULA_SUBJECT_PHRASE_BREAK.search(segment):
                continue
            if _normalize_formula_result_head(_formula_subject_segment_head(segment)):
                selected = segment
                break
    return _formula_subject_segment_head(selected)


def _formula_operand_is_numeric(operand: str) -> bool:
    if _FORMULA_NUMERIC_OPERAND_LITERAL.fullmatch(operand):
        return True
    literal_operand = re.fullmatch(
        rf"\s*(?:(?:the\s+)?(?:first|next|each|every|any)\s+)?"
        rf"{_FORMULA_NUMERIC_OPERAND_LITERAL_CORE}"
        rf"(?:\s+of\s+(?P<base>.+))?\s*",
        operand,
        flags=re.IGNORECASE,
    )
    if literal_operand is not None:
        base = literal_operand.group("base")
        return base is None or _formula_operand_is_numeric(base)
    bracket_base = re.fullmatch(
        r"\s*(?:(?:the|that)\s+(?:part|portion)\s+of\s+(?P<portion>.+?)\s+"
        r"(?:over|above|exceeding|not\s+(?:exceeding|to\s+exceed)|"
        r"in\s+excess\s+of|(?:which|that)\s+"
        r"(?:equals\s+or\s+exceeds|exceeds\s+or\s+equals|exceeds|"
        r"(?:is\s+)?greater\s+than|does\s+not\s+exceed|"
        r"(?:is\s+)?(?:not|no)\s+more\s+than|(?:is\s+)?at\s+most|"
        r"(?:is\s+)?less\s+than(?:\s+or\s+equal\s+to)?))\s+"
        r"(?P<portion_limit>.+)|so\s+much\s+of\s+(?P<much>.+?)\s+as\s+"
        r"(?:equals\s+or\s+exceeds|exceeds\s+or\s+equals|exceeds|"
        r"is\s+in\s+excess\s+of|(?:is\s+)?greater\s+than|"
        r"does\s+not\s+exceed|(?:is\s+)?(?:not|no)\s+more\s+than|"
        r"(?:is\s+)?at\s+most|(?:is\s+)?less\s+than"
        r"(?:\s+or\s+equal\s+to)?)\s+"
        r"(?P<much_limit>.+)|(?:the\s+)?excess\s+of\s+(?P<excess>.+?)\s+"
        r"(?:over|above)\s+(?P<excess_limit>.+))\s*",
        operand,
        flags=re.IGNORECASE,
    )
    if bracket_base is not None:
        base = next(
            bracket_base.group(name)
            for name in ("portion", "much", "excess")
            if bracket_base.group(name) is not None
        )
        limit = next(
            bracket_base.group(name)
            for name in ("portion_limit", "much_limit", "excess_limit")
            if bracket_base.group(name) is not None
        )
        return _formula_operand_is_numeric(base) and _formula_operand_is_numeric(limit)
    if re.fullmatch(
        rf"\s*(?:the\s+)?{_FORMULA_TABLE_NAME}\s*",
        operand,
        flags=re.IGNORECASE,
    ):
        return True
    if re.match(
        r"\s*(?:the\s+)?(?:addition|average|difference|greater|greatest|higher|"
        r"highest|larger|largest|least|lesser|lower|lowest|max|maximum|mean|"
        r"median|min|minimum|product|quotient|ratio|remainder|smaller|smallest|"
        r"sum|total)\s+(?:of|between)\b",
        operand,
        flags=re.IGNORECASE,
    ):
        return _formula_operation_has_numeric_operands(operand)
    head = _formula_subject_segment_head(operand)
    if _normalize_formula_operand_head(head):
        return True
    operand_segment = _FORMULA_SUBJECT_PHRASE_BREAK.split(operand, maxsplit=1)[0]
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", operand_segment)
    for index, word in enumerate(words):
        normalized_word = _normalize_formula_operand_head(word)
        if not normalized_word:
            continue
        trailing = words[index + 1 :]
        if trailing:
            if all(
                candidate.lower().endswith(("ed", "en"))
                or candidate.lower().endswith("ly")
                or candidate.lower() in _FORMULA_OPERAND_IRREGULAR_POSTMODIFIERS
                for candidate in trailing
            ):
                return True
            if normalized_word in {"coefficient", "formula", "index"} and all(
                re.fullmatch(
                    r"(?:[A-Z0-9]+(?:-[A-Z0-9]+)*|[IVXLCDM]+)",
                    candidate,
                )
                is not None
                for candidate in trailing
            ):
                return True
    return False


def _normalize_formula_operand_head(head: str) -> str:
    head = head.lower()
    if head in _FORMULA_NUMERIC_OPERAND_HEADS:
        return head
    candidates = []
    if head.endswith("ies"):
        candidates.append(head[:-3] + "y")
    if head.endswith("es"):
        candidates.append(head[:-2])
    if head.endswith("s"):
        candidates.append(head[:-1])
    return next(
        (
            candidate
            for candidate in candidates
            if candidate in _FORMULA_NUMERIC_OPERAND_HEADS
        ),
        "",
    )


def _formula_aggregate_operands_are_numeric(operands: str) -> bool:
    comma_parts = [
        re.sub(r"^(?:and|or)\s+", "", part.strip(), flags=re.IGNORECASE)
        for part in re.split(r"\s*,\s*", operands)
        if part.strip()
    ]
    if len(comma_parts) >= 2:
        return all(_formula_operand_is_numeric(part) for part in comma_parts)
    conjunctions = list(re.finditer(r"\s+(?:and|or)\s+", operands, flags=re.IGNORECASE))
    for conjunction in conjunctions:
        left = operands[: conjunction.start()]
        right = operands[conjunction.end() :]
        if _formula_operand_is_numeric(left) and _formula_operand_is_numeric(right):
            return True
        left_words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", left)
        if (
            _formula_operand_is_numeric(right)
            and left_words
            and all(
                word.lower() in _FORMULA_SUBJECT_LEADING_NONHEAD_WORDS
                for word in left_words
            )
        ):
            return True
    return False


def _formula_inline_operands_are_numeric(operands: str) -> bool:
    operands = operands.strip().strip(".;:, ")
    semicolon_parts = [part.strip() for part in operands.split(";") if part.strip()]
    if len(semicolon_parts) >= 2:
        return all(_formula_operand_is_numeric(part) for part in semicolon_parts)
    return _formula_operand_is_numeric(operands) or (
        _formula_aggregate_operands_are_numeric(operands)
    )


def _truncate_following_operand_tail(tail: str) -> str:
    """Stop a following-values list before a proviso or sibling outline."""

    tail = re.split(r"\n(?=\s*[A-Z]\.)", tail, maxsplit=1)[0]
    proviso = re.search(
        rf";(?={_FORMULA_FOLLOWING_OPERAND_PROVISO.pattern})",
        tail,
        flags=re.IGNORECASE,
    )
    if proviso is not None:
        tail = tail[: proviso.start()]

    markers = tuple(
        re.finditer(
            r"(?m)^\s*(?:\((?P<paren>[^)]+)\)|(?P<number>\d+)\.)\s*",
            tail,
        )
    )
    if not markers:
        return tail
    first_is_numeric = bool(markers[0].group("number")) or bool(
        (markers[0].group("paren") or "")[0:1].isdigit()
    )
    for marker in markers[1:]:
        label = marker.group("number") or marker.group("paren") or ""
        if bool(label[0:1].isdigit()) != first_is_numeric:
            return tail[: marker.start()]
    return tail


def _applied_operation_match_is_numeric(text: str, match: re.Match[str]) -> bool:
    """Return whether one applying/application clause states numeric work."""

    prefix = re.sub(
        r"\b(?:(?:shall|must|may)\s+be|is|are)\s*$",
        "",
        text[: match.start()],
        flags=re.IGNORECASE,
    )
    if not _normalize_formula_result_head(_formula_result_subject_head(prefix)):
        return False
    operands = re.split(
        _FORMULA_INDEPENDENT_COORDINATE,
        match.group("operands"),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    target_match = re.search(r"\s+to\s+", operands, flags=re.IGNORECASE)
    applied = operands[: target_match.start()] if target_match is not None else operands
    target = operands[target_match.end() :] if target_match is not None else ""
    # A named formula or index must end in a bounded code identifier, including
    # hyphenated codes such as CPI-U.  Prose document titles remain
    # administrative even when they name a numeric target.  Coefficients also
    # permit one targeted proper name ("coefficient Alpha to income").
    named_match = re.search(
        r"\b(?P<head>coefficient|formula|index)\b(?P<tail>.*)\Z",
        applied,
        flags=re.IGNORECASE,
    )
    named_reference_is_numeric = False
    if named_match is not None:
        identifier_tail = named_match.group("tail").strip()
        if identifier_tail:
            administrative_title = _FORMULA_ADMINISTRATIVE_TITLE_LANGUAGE.search(
                identifier_tail
            )
            # Code length does not distinguish an identifier from a titled
            # administrative document.  Accept the complete bounded uppercase
            # code grammar, then reject semantic document-title heads above.
            bounded_code = re.fullmatch(r"[A-Z0-9]+(?:-[A-Z0-9]+)*", identifier_tail)
            code_segments = identifier_tail.split("-")
            administrative_segments = [
                segment
                for segment in code_segments
                if _FORMULA_ADMINISTRATIVE_TITLE_LANGUAGE.fullmatch(segment)
            ]
            semantic_document_segment = any(
                not _FORMULA_CODE_NAMESPACE_TITLE_LANGUAGE.fullmatch(segment)
                for segment in administrative_segments
            )
            title_qualifier_segment = any(
                _FORMULA_ADMINISTRATIVE_TITLE_QUALIFIER_LANGUAGE.fullmatch(segment)
                for segment in code_segments
            )
            numeric_code_tail = code_segments[-1].lower() in (
                _FORMULA_NUMERIC_RESULT_HEADS
                | {
                    "adjustment",
                    "coefficient",
                    "index",
                    "multiplier",
                    "scale",
                    "weight",
                }
            )
            # Ambiguous namespace atoms (RULE, SECTION, SCHEDULE, POLICY, TAX,
            # WORK) may participate in a multi-atom code.  An unambiguous
            # document atom such as MANUAL, REPORT, GUIDE, or WEBSITE wins over
            # every acronym, digit, and payload suffix; those signals cannot
            # turn a titled document into executable work.
            code_like = bool(
                bounded_code is not None
                and not semantic_document_segment
                and (not title_qualifier_segment or numeric_code_tail)
                and (not administrative_segments or len(code_segments) > 1)
            )
            targeted_coefficient_name = (
                named_match.group("head").lower() == "coefficient"
                and target_match is not None
                and (administrative_title is None or numeric_code_tail)
                and re.fullmatch(
                    r"[A-Z][A-Za-z0-9]*(?:-[A-Z][A-Za-z0-9]*)*",
                    identifier_tail,
                )
            )
            uppercase_coefficient_code = (
                named_match.group("head").lower() == "coefficient"
                and (administrative_title is None or numeric_code_tail)
                and re.fullmatch(
                    r"[A-Z0-9]+(?:-[A-Z0-9]+)*",
                    identifier_tail,
                )
            )
            if not (
                code_like or targeted_coefficient_name or uppercase_coefficient_code
            ):
                return False
            named_reference_is_numeric = True
    return (named_reference_is_numeric or _formula_operand_is_numeric(applied)) and (
        target_match is None or _formula_operand_is_numeric(target)
    )


def _formula_states_applied_operation(text: str) -> bool:
    """Recognize applying a numeric rate/formula, not administrative criteria."""

    return any(
        _applied_operation_match_is_numeric(text, match)
        for match in _FORMULA_APPLIED_OPERATION_LANGUAGE.finditer(text)
    )


def _without_unproven_applied_operations(text: str) -> str:
    """Mask administrative applying clauses from generic computation cues."""

    characters = list(text)
    for match in _FORMULA_APPLIED_OPERATION_LANGUAGE.finditer(text):
        if _applied_operation_match_is_numeric(text, match):
            continue
        characters[match.start() : match.end()] = " " * (match.end() - match.start())
    return "".join(characters)


def _formula_numeric_noun_phrase_head(phrase: str) -> str:
    """Return the normalized numeric result head of a bounded noun phrase."""

    phrase = re.sub(
        r"^(?:that|this|such)\s+",
        "the ",
        phrase.strip(),
        flags=re.IGNORECASE,
    )
    return _normalize_formula_result_head(_formula_subject_segment_head(phrase))


def _formula_rounding_noun_phrase_is_numeric(phrase: str) -> bool:
    """Recognize a numeric antecedent without admitting a titled document."""

    last_word = phrase.strip().rsplit(maxsplit=1)[-1]
    administrative_titles = tuple(
        _FORMULA_ADMINISTRATIVE_TITLE_LANGUAGE.finditer(phrase)
    )
    return bool(
        not _FORMULA_ADMINISTRATIVE_TITLE_LANGUAGE.fullmatch(last_word)
        and all(match.group().lower() == "tax" for match in administrative_titles)
        and _formula_numeric_noun_phrase_head(phrase)
    )


def _formula_rounding_actor_is_bounded(actor: str) -> bool:
    """Recognize a finite-clause actor while rejecting titled documents."""

    actor = actor.strip().lstrip(",").strip()
    if not actor:
        return True
    if re.fullmatch(r"(?:it|they)", actor, flags=re.IGNORECASE):
        return True
    determined_actor = re.fullmatch(
        r"(?:the|a|an|each|any|every|this|that|such)\s+"
        r"[A-Za-z][A-Za-z'-]*(?:\s+(?:(?:of|for)\s+(?:the\s+)?)?"
        r"[A-Za-z][A-Za-z'-]*){0,5}",
        actor,
        flags=re.IGNORECASE,
    )
    proper_actor = re.fullmatch(
        r"[A-Z][A-Za-z'-]*(?:\s+(?:(?:of|for)\s+(?:the\s+)?|and\s+)?"
        r"[A-Z][A-Za-z'-]*){0,5}",
        actor,
    )
    bare_actor = re.fullmatch(
        r"(?!.*\b(?:a|an|any|each|every|that|the|this)\b)"
        r"[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,5}",
        actor,
    )
    if not (determined_actor or proper_actor or bare_actor):
        return False
    actor_core = re.sub(
        r"^(?:the|a|an|each|any|every|this|that|such)\s+",
        "",
        actor,
        flags=re.IGNORECASE,
    )
    actor_starts_with_role = bool(
        _FORMULA_ROUNDING_ACTOR_ROLE_LANGUAGE.match(actor_core)
    )
    actor_last_word = actor_core.rsplit(maxsplit=1)[-1]
    actor_ends_with_role = bool(
        _FORMULA_ROUNDING_ACTOR_ROLE_LANGUAGE.fullmatch(actor_last_word)
    )
    actor_has_title = bool(_FORMULA_ADMINISTRATIVE_TITLE_LANGUAGE.search(actor))
    role_starts_scoped_actor = bool(
        actor_starts_with_role
        and re.match(
            r"^[A-Za-z][A-Za-z'-]*\s+(?:of|for)\b",
            actor_core,
            flags=re.IGNORECASE,
        )
    )
    if not actor_has_title:
        return True
    return bool(role_starts_scoped_actor or actor_ends_with_role)


def _formula_passive_rounding_modifier_is_bounded(modifier: str) -> bool:
    """Recognize bounded adverbial, actor, and legal passive modifiers."""

    adverbial_by_actor = re.fullmatch(
        r"[A-Za-z]+ly(?:\s+and\s+[A-Za-z]+ly)*\s+by\s+"
        r"(?P<actor>[^,.;:\n]{1,60})",
        modifier,
        flags=re.IGNORECASE,
    )
    if re.fullmatch(
        r"[A-Za-z]+ly(?:\s+and\s+[A-Za-z]+ly)*",
        modifier,
        flags=re.IGNORECASE,
    ):
        return True
    if adverbial_by_actor is not None:
        return _formula_rounding_actor_is_bounded(adverbial_by_actor.group("actor"))
    if modifier.lower().startswith("by "):
        return _formula_rounding_actor_is_bounded(modifier[3:])
    legal_reference = (
        r"(?:(?:the|this|that|such)\s+)?"
        r"[A-Za-z§][A-Za-z0-9 .():§'-]{0,60}"
    )
    legal_connector = (
        r"(?:according\s+to|consistent\s+with|for\s+(?:the\s+)?purposes?\s+of|"
        r"in\s+accordance\s+with|in\s+compliance\s+with|"
        r"in\s+conformity\s+with|per|subject\s+to|under|pursuant\s+to)"
    )
    return bool(
        re.fullmatch(
            rf"as\s+(?:otherwise\s+)?(?:authorized|described|directed|"
            rf"established|mandated|outlined|required|provided|specified|"
            rf"prescribed|stipulated)"
            rf"(?:\s+(?:by|in|under|{legal_connector})\s+{legal_reference})?",
            modifier,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            rf"as\s+set\s+forth\s+in\s+{legal_reference}",
            modifier,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            rf"as\s+provided\s+for(?:\s+in\s+{legal_reference})?",
            modifier,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            rf"as\s+{legal_reference}\s+requires",
            modifier,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            rf"in\s+(?:the\s+)?manner\s+(?:authorized|established|mandated|"
            rf"outlined|provided|required|specified|prescribed|stipulated)\s+"
            rf"(?:by|in|under)\s+"
            rf"{legal_reference}",
            modifier,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            rf"in\s+(?:the\s+)?manner\s+set\s+forth\s+in\s+{legal_reference}",
            modifier,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            rf"{legal_connector}\s+{legal_reference}",
            modifier,
            flags=re.IGNORECASE,
        )
    )


def _formula_passive_rounding_actor(tail: str) -> str | None:
    """Separate a passive determination modifier from the rounding actor."""

    tail = tail.strip()
    if not tail:
        return ""
    if _formula_passive_rounding_modifier_is_bounded(tail):
        return ""
    if "," in tail:
        modifier, actor = tail.rsplit(",", maxsplit=1)
        modifier = modifier.strip()
        if modifier and not _formula_passive_rounding_modifier_is_bounded(modifier):
            return None
        return actor.strip()
    adverbial_actor = re.fullmatch(
        r"[A-Za-z]+ly\s+(?P<actor>[^,.;:\n]{1,60})",
        tail,
        flags=re.IGNORECASE,
    )
    if adverbial_actor is not None:
        return adverbial_actor.group("actor")
    return tail


def _formula_rounding_antecedent_is_numeric(prefix: str) -> bool:
    """Recognize a bounded numeric antecedent before a rounding pronoun."""

    gerund = re.search(
        r"\b(?:after|before|once|upon|when)\s+"
        r"(?:calculating|computing|determining)\s+"
        r"(?P<body>[^.;:\n]{1,120})$",
        prefix,
        flags=re.IGNORECASE,
    )
    passive = re.search(
        r"\b(?:after|before|once|upon|when)\s+"
        r"(?P<noun_phrase>[^,.;:\n]{1,80}?)\s+"
        r"(?:(?:is|are)\s+|(?:has|have|had)\s+been\s+)"
        r"(?:calculated|computed|determined)"
        r"(?P<tail>\s*[^.;:\n]{0,100})$",
        prefix,
        flags=re.IGNORECASE,
    )
    if passive is not None:
        passive_actor = _formula_passive_rounding_actor(passive.group("tail"))
        if (
            passive_actor is not None
            and _formula_rounding_noun_phrase_is_numeric(passive.group("noun_phrase"))
            and _formula_rounding_actor_is_bounded(passive_actor)
        ):
            return True
    if gerund is None:
        return False
    body = gerund.group("body").strip()
    if "," in body:
        body = body.split(",", maxsplit=1)[0].strip()
        return _formula_rounding_noun_phrase_is_numeric(body)
    # Without punctuation, find a numeric result phrase followed by any
    # bounded finite-clause actor.  This avoids hard-coding agency titles such
    # as department, commissioner, secretary, or board.
    words = body.split()
    for split_at in range(len(words) - 1, 0, -1):
        noun_phrase = " ".join(words[:split_at])
        actor = " ".join(words[split_at:])
        if _formula_rounding_actor_is_bounded(
            actor
        ) and _formula_rounding_noun_phrase_is_numeric(noun_phrase):
            return True
    return False


def _formula_nonnegative_floor_tail_start(clause: str) -> int | None:
    """Return the final coordinated zero-floor start for a numeric result."""

    connectors = tuple(
        re.finditer(
            r"(?:,\s*)?\bbut\b|\s+\band\b|,\s*\bwhich\b",
            clause,
            flags=re.IGNORECASE,
        )
    )
    for connector in reversed(connectors):
        tail = clause[connector.end() :].strip()
        if re.fullmatch(
            _FORMULA_NONNEGATIVE_FLOOR_CONTROL,
            tail,
            flags=re.IGNORECASE,
        ) or re.fullmatch(
            rf"it\s+{_FORMULA_NONNEGATIVE_FLOOR_CONTROL}",
            tail,
            flags=re.IGNORECASE,
        ):
            return connector.start()
        subject_floor = re.fullmatch(
            rf"(?P<subject>(?:the\s+)?[A-Za-z][A-Za-z '-]{{0,40}}?)\s+"
            rf"{_FORMULA_NONNEGATIVE_FLOOR_CONTROL}",
            tail,
            flags=re.IGNORECASE,
        )
        if subject_floor is not None and _formula_numeric_noun_phrase_head(
            subject_floor.group("subject")
        ):
            return connector.start()
    return None


def _rounding_language_is_computational(text: str) -> bool:
    """Require a numeric subject, operation, or precision for rounding prose."""

    for match in _ROUNDING_LANGUAGE.finditer(text):
        if not re.match(r"round", match.group(), flags=re.IGNORECASE):
            return True
        rounding_word = match.group().lower()
        active_target = None
        if (
            rounding_word == "round"
            or rounding_word.startswith("rounding")
            or (rounding_word.startswith("round "))
        ):
            clause_start = max(
                text.rfind(".", 0, match.start()),
                text.rfind(";", 0, match.start()),
                text.rfind("\n", 0, match.start()),
            )
            active_prefix = _strip_source_clause_marker(
                text[clause_start + 1 : match.start()]
            ).strip()
            modal_directive = re.search(
                r"\b(?:can|may|must|shall|should|will)\b"
                r"(?:(?:\s+(?:and\s+)?(?:then|\w+ly))|"
                r"(?:\s+(?:as\s+necessary|from\s+time\s+to\s+time|"
                r"without\s+delay|in\s+all\s+cases))|"
                r"(?:\s*,\s*[^,]{1,80},))*\s*$",
                active_prefix,
                flags=re.IGNORECASE,
            )
            to_directive = re.search(
                r"\b(?:is|are)\s+to\s*$",
                active_prefix,
                flags=re.IGNORECASE,
            )
            conditional_directive = re.fullmatch(
                r"(?:after|before|for|if|once|unless|upon|when|where)\b"
                r"[^.;:\n]{0,120},\s*"
                r"(?:(?:and\s+)?(?:then|\w+ly)\s*)*",
                active_prefix,
                flags=re.IGNORECASE,
            )
            coordinated_directive = re.search(
                r"\b(?:can|may|must|shall|should|will)\s+"
                r"(?:(?:then|thereafter|[A-Za-z]+ly|first|next)\s+)*"
                r"(?:calculate|compute|determine)\s+"
                r"(?P<noun_phrase>[^,.;:\n]{1,80}?)\s*,?\s+and"
                r"(?:\s+(?:then|thereafter|[A-Za-z]+ly|first|next))*"
                r"(?:\s+(?:can|may|must|shall|should|will)"
                r"(?:\s+(?:then|thereafter|[A-Za-z]+ly|first|next))*)?\s*$",
                re.sub(r"\s*,\s*", " ", active_prefix),
                flags=re.IGNORECASE,
            )
            shared_object_directive = re.search(
                r"\b(?:can|may|must|shall|should|will)\s+"
                r"(?:(?:then|thereafter|[A-Za-z]+ly|first|next)\s+)*"
                r"(?:calculate|compute|determine)\s*,?\s+and"
                r"(?:\s+(?:then|thereafter|[A-Za-z]+ly|first|next))*"
                r"(?:\s+(?:can|may|must|shall|should|will)"
                r"(?:\s+(?:then|thereafter|[A-Za-z]+ly|first|next))*)?\s*$",
                re.sub(r"\s*,\s*", " ", active_prefix),
                flags=re.IGNORECASE,
            )
            if active_prefix and not (
                modal_directive
                or to_directive
                or conditional_directive
                or coordinated_directive
                or shared_object_directive
            ):
                continue
            target_match = re.match(
                r"\s*(?:,\s*[^,]{1,80},\s*)?"
                r"(?P<target>[^.;\n]{1,120}?)(?=[.;\n]|$)",
                text[match.end() :],
                flags=re.IGNORECASE,
            )
            if target_match is None:
                continue
            active_target = re.sub(
                r"^(?:down(?:ward)?|up(?:ward)?|off)\s+",
                "",
                target_match.group("target").strip(),
                flags=re.IGNORECASE,
            )
            active_target = re.split(
                r"\s+\b(?:is|are|shall|must|may)\b",
                active_target,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            active_target = re.sub(
                r"\s+(?:down(?:ward)?|up(?:ward)?|off)\s*$",
                "",
                active_target,
                flags=re.IGNORECASE,
            )
            active_target = re.sub(
                r",?\s*(?:as|if|unless|when|whenever|where)\b[^,.;:\n]*$",
                "",
                active_target,
                flags=re.IGNORECASE,
            ).strip()
        if re.match(r"\s+out\b", text[match.end() :], flags=re.IGNORECASE):
            continue
        if active_target is not None:
            if _formula_operand_is_numeric(active_target) or (
                _normalize_formula_result_head(
                    _formula_subject_segment_head(active_target)
                )
            ):
                return True
            if re.fullmatch(r"(?:it|them)", active_target, flags=re.IGNORECASE):
                prefix_without_modal = re.sub(
                    r"\b(?:can|may|must|shall|should|will)\b"
                    r"(?:(?:\s+(?:and\s+)?(?:then|\w+ly))|"
                    r"(?:\s+(?:as\s+necessary|from\s+time\s+to\s+time|"
                    r"without\s+delay|in\s+all\s+cases))|"
                    r"(?:\s*,\s*[^,]{1,80},))*\s*$",
                    "",
                    active_prefix,
                    flags=re.IGNORECASE,
                )
                antecedent_head = _normalize_formula_result_head(
                    _formula_result_subject_head(prefix_without_modal)
                )
                coordinated_antecedent_head = (
                    _formula_numeric_noun_phrase_head(
                        re.sub(
                            r"\s+(?:first|next|then|thereafter|[A-Za-z]+ly)\s*$",
                            "",
                            coordinated_directive.group("noun_phrase"),
                            flags=re.IGNORECASE,
                        )
                    )
                    if coordinated_directive is not None
                    else ""
                )
                if (
                    antecedent_head
                    or _formula_rounding_antecedent_is_numeric(prefix_without_modal)
                    or coordinated_antecedent_head
                ):
                    return True
        clause_start = max(
            text.rfind(".", 0, match.start()),
            text.rfind(";", 0, match.start()),
            text.rfind("\n", 0, match.start()),
        )
        clause_end_match = re.search(r"[.;\n]", text[match.end() :])
        clause_end = (
            match.end() + clause_end_match.start()
            if clause_end_match is not None
            else len(text)
        )
        clause = text[clause_start + 1 : clause_end]
        prefix = re.sub(
            r"\b(?:(?:shall|must|may|should|will)\s+be|"
            r"(?:is|are)(?:\s+required)?\s+to\s+be|is|are)\s*$",
            "",
            text[clause_start + 1 : match.start()],
            flags=re.IGNORECASE,
        )
        passive_pronoun = re.fullmatch(
            r"(?P<antecedent>.+?)(?:,\s*|\s+)(?:it|they)\s*",
            prefix,
            flags=re.IGNORECASE,
        )
        if passive_pronoun is not None and _formula_rounding_antecedent_is_numeric(
            passive_pronoun.group("antecedent")
        ):
            return True
        if _normalize_formula_result_head(_formula_result_subject_head(prefix)):
            return True
        if _formula_operation_has_numeric_operands(clause):
            return True
        if re.search(
            r"\b(?:down|up|to\s+the\s+nearest)\b[^.;\n]{0,80}"
            r"\b(?:dollars?|cents?|percent(?:age)?(?:\s+points?)?|hours?|units?)\b",
            text[match.start() : clause_end],
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _formula_operation_has_numeric_operands(text: str) -> bool:
    """Require actual numeric operand heads for a noun-form operation."""

    without_parentheticals = re.sub(
        r",\s*(?:(?:as|adjusted|calculated|computed|described|determined|"
        r"provided|subject|whether)\s+[^,]+|if\s+[^,]+|net\s+of\s+[^,]+|"
        r"including\s+[^,]+|excluding\s+[^,]+)\s*,",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    direct_following = _FORMULA_DIRECT_FOLLOWING_OPERANDS.search(without_parentheticals)
    if direct_following is not None:
        tail = without_parentheticals[direct_following.end() :].lstrip()
        if tail.lstrip().startswith(":"):
            tail = tail.lstrip()[1:]
        elif tail.startswith(","):
            tail = tail[1:]
        tail = _truncate_following_operand_tail(tail)
        marker = re.search(r"(?m)^\s*(?:\([^)]+\)|\d+\.)\s*", tail)
        if marker is not None:
            entries = [
                entry.strip()
                for entry in re.split(
                    r"(?m)^\s*(?:\([^)]+\)|\d+\.)\s*",
                    tail[marker.start() :],
                )
                if entry.strip()
            ]
            return bool(entries) and all(
                source_states_explicit_computation(entry)
                or _formula_inline_operands_are_numeric(entry)
                for entry in entries
            )
        inline = re.split(r"\n(?=\s*[A-Z]\.)", tail, maxsplit=1)[0]
        inline = inline.strip().strip(".;:, ")
        if inline:
            return _formula_inline_operands_are_numeric(inline)
        return True

    clause = re.split(r"[.;:\n]", without_parentheticals, maxsplit=1)[0]
    floor_tail_start = _formula_nonnegative_floor_tail_start(clause)
    if floor_tail_start is not None:
        clause = clause[:floor_tail_start]
    clause = re.sub(
        r",\s*(?:whichever[^,]{0,80}\b(?:is|shall\s+be|may\s+be|would\s+be)\s+[^,]+|as\s+applicable|"
        r"if\s+applicable|"
        r"(?:in\s+)?no\s+case\s+(?:less\s+than|below)\s+"
        r"(?:zero|\$?\s*0(?:\.0+)?)|except\s+that\s+it\s+"
        r"(?:shall|must|may)\s+not\s+be\s+less\s+than\s+"
        r"(?:zero|\$?\s*0(?:\.0+)?)|with\s+a\s+minimum\s+of\s+"
        r"(?:zero|\$?\s*0(?:\.0+)?)|subject\s+to\s+a\s+floor\s+of\s+"
        r"(?:zero|\$?\s*0(?:\.0+)?))\s*$",
        "",
        clause,
        flags=re.IGNORECASE,
    )
    if re.search(
        r"\b(?:addition|average|difference|greater|greatest|higher|highest|"
        r"larger|largest|least|lesser|lower|lowest|max|maximum|mean|median|min|"
        r"minimum|product|quotient|ratio|remainder|smaller|smallest|sum|total)"
        r"\s+(?:of|between)\s*$",
        clause,
        flags=re.IGNORECASE,
    ):
        return True
    remainder = re.search(
        r"\bremainder\s+of\s+(?P<left>.+?)\s+after\s+(?P<right>.+)$",
        clause,
        flags=re.IGNORECASE,
    )
    if remainder is not None:
        return _formula_operand_is_numeric(
            remainder.group("left")
        ) and _formula_operand_is_numeric(remainder.group("right"))

    aggregate = re.search(
        r"\b(?P<operation>addition|average|greater|greatest|higher|highest|largest|least|"
        r"larger|lesser|lower|lowest|max|maximum|mean|median|min|minimum|product|"
        r"remainder|smaller|smallest|sum|total|difference)\s+of\s+"
        r"(?P<operands>.+)$",
        clause,
        flags=re.IGNORECASE,
    )
    if aggregate is not None:
        operands = aggregate.group("operands")
        if _formula_aggregate_operands_are_numeric(operands):
            return True
        return aggregate.group("operation").lower() in {
            "addition",
            "average",
            "mean",
            "median",
            "sum",
            "total",
        } and _formula_operand_is_numeric(operands)

    patterns = (
        r"\bdifference\s+between\s+(?P<left>.+?)\s+and\s+(?P<right>.+)$",
        r"\b(?:quotient|ratio)\s+between\s+(?P<left>.+?)\s+and\s+"
        r"(?P<right>.+)$",
        r"\b(?:quotient|ratio)\s+of\s+(?P<left>.+?)\s+"
        r"(?:and|by|to)\s+(?P<right>.+)$",
        r"\b(?:decrease|division|increase|multiplication|reduction)\s+of\s+"
        r"(?P<left>.+?)\s+by\s+(?P<right>.+)$",
        r"\b(?:deduction|subtraction)\s+of\s+(?P<left>.+?)\s+from\s+"
        r"(?P<right>.+)$",
    )
    for pattern in patterns:
        operation = re.search(pattern, clause, flags=re.IGNORECASE)
        if operation is not None:
            right = operation.group("right").split(",", maxsplit=1)[0]
            return _formula_operand_is_numeric(
                operation.group("left")
            ) and _formula_operand_is_numeric(right)
    unary = re.search(
        r"\b(?:percent|percentage)\s+of\s+(?P<operand>.+)$",
        clause,
        flags=re.IGNORECASE,
    )
    if unary is not None:
        operand = unary.group("operand").split(",", maxsplit=1)[0]
        return _formula_operand_is_numeric(operand)
    return False


def _formula_states_contextual_operator(
    source_text: str,
    *,
    include_unconditional: bool = True,
) -> bool:
    """Recognize result operations only when their grammatical subject is numeric."""

    if include_unconditional and _FORMULA_UNCONDITIONAL_OPERATOR_LANGUAGE.search(
        source_text
    ):
        return True
    for pattern in (
        _FORMULA_RESULT_OPERATION_LANGUAGE,
        _FORMULA_PARTICIPIAL_RESULT_LANGUAGE,
    ):
        for match in pattern.finditer(source_text):
            prefix = source_text[: match.start()]
            if pattern is _FORMULA_PARTICIPIAL_RESULT_LANGUAGE and re.search(
                r"\bbefore(?:\s+\w+){0,3}\s*$",
                prefix,
                flags=re.IGNORECASE,
            ):
                continue
            head = _normalize_formula_result_head(_formula_result_subject_head(prefix))
            if head and (
                pattern is _FORMULA_PARTICIPIAL_RESULT_LANGUAGE
                or _formula_operation_has_numeric_operands(source_text[match.start() :])
            ):
                return True
    return False


def _english_fraction_target_is_duration(
    target: str, *, numeric_result_subject: bool = False
) -> bool:
    """Recognize a short duration noun phrase without swallowing its later base."""

    duration = re.match(
        r"^\s*(?:(?!(?:as|for|from|in|of|over|under)\b)"
        r"[A-Za-z0-9$€£'’.-]+\s+){0,8}"
        r"(?P<unit>seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b"
        r"(?P<tail>.*)$",
        target,
        flags=re.IGNORECASE,
    )
    if duration is None:
        return False
    tail = duration.group("tail").strip()
    if not tail:
        return True
    if re.match(r"^of\b", tail, flags=re.IGNORECASE) is None:
        return False
    return not (numeric_result_subject and duration.group("unit").lower().endswith("s"))


def _english_fraction_of_is_computational(source_text: str) -> bool:
    """Require a numeric target and reject ordinary fractional durations."""

    for match in _ENGLISH_FRACTION_OF.finditer(source_text):
        target = re.split(
            _FORMULA_INDEPENDENT_COORDINATE,
            match.group("target"),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        target = target.split(",", maxsplit=1)[0].strip()
        prefix = source_text[: match.start()]
        result_predicates = tuple(
            re.finditer(rf"\b{_FORMULA_RESULT_PREDICATE}\b", prefix, re.IGNORECASE)
        )
        result_predicate = result_predicates[-1] if result_predicates else None
        intervening = (
            prefix[result_predicate.end() :].strip() if result_predicate else ""
        )
        predicate_directly_governs_fraction = result_predicate is not None and (
            not intervening
            or bool(
                re.fullmatch(
                    rf"(?:a|an|{_ENGLISH_CARDINAL_PHRASE})",
                    intervening,
                    flags=re.IGNORECASE,
                )
            )
        )
        subject = prefix[: result_predicate.start()] if result_predicate else prefix
        numeric_result_subject = predicate_directly_governs_fraction and bool(
            _formula_result_subject_head(subject)
        )
        if _english_fraction_target_is_duration(
            target, numeric_result_subject=numeric_result_subject
        ):
            continue
        if _formula_operand_is_numeric(target):
            return True
    return False


def source_states_explicit_computation(source_text: str) -> bool:
    """Return whether text states a computation rather than only a scalar."""

    computation_text = _without_unproven_applied_operations(
        _without_stated_conversion_results(source_text)
    )
    return bool(
        _has_substantive_arithmetic_expression(computation_text)
        or _COMPUTATION_LANGUAGE.search(computation_text)
        or _EXCESS_BOUND_TAX_RATE_LANGUAGE.search(computation_text)
        or _ENGLISH_WORDED_PERCENTAGE_OF.search(computation_text)
        or _EXPLICIT_NUMERIC_PERCENTAGE_OF.search(computation_text)
        or _english_fraction_of_is_computational(computation_text)
        or _ENGLISH_FRACTIONAL_PERCENTAGE_OF.search(computation_text)
        or _FORMULA_COMPUTED_OPERATION_LANGUAGE.search(computation_text)
        or _formula_states_applied_operation(computation_text)
        or _rounding_language_is_computational(computation_text)
        or _formula_states_contextual_operator(computation_text)
    )


def _source_states_nonrounding_computation(source_text: str) -> bool:
    """Return whether text states a computation other than rounding."""

    computation_text = _without_unproven_applied_operations(
        _without_stated_conversion_results(source_text)
    )
    return bool(
        _has_substantive_arithmetic_expression(computation_text)
        or _COMPUTATION_LANGUAGE.search(computation_text)
        or _EXCESS_BOUND_TAX_RATE_LANGUAGE.search(computation_text)
        or _ENGLISH_WORDED_PERCENTAGE_OF.search(computation_text)
        or _EXPLICIT_NUMERIC_PERCENTAGE_OF.search(computation_text)
        or _english_fraction_of_is_computational(computation_text)
        or _ENGLISH_FRACTIONAL_PERCENTAGE_OF.search(computation_text)
        or _FORMULA_COMPUTED_OPERATION_LANGUAGE.search(computation_text)
        or _formula_states_applied_operation(computation_text)
        or _formula_states_contextual_operator(computation_text)
    )


def source_states_stated_conversion_result(source_text: str) -> bool:
    """Return whether text states both a base scalar and its converted result."""

    return bool(_stated_conversion_result_spans(source_text))


def _stated_conversion_result_spans(source_text: str) -> tuple[tuple[int, int], ...]:
    """Locate result clauses such as ``Umgerechnet ... ergibt sich 6 150``."""

    spans: list[tuple[int, int]] = []
    for cue in _STATED_CONVERSION_CUE.finditer(source_text):
        result = _STATED_CONVERSION_RESULT.search(
            source_text,
            cue.end(),
            min(len(source_text), cue.end() + 180),
        )
        if result is None:
            continue
        if _stated_conversion_candidate_contains_formula(source_text, cue, result):
            continue
        base_window = source_text[max(0, cue.start() - 300) : cue.start()]
        paragraph_break = base_window.rfind("\n\n")
        if paragraph_break >= 0:
            base_window = base_window[paragraph_break + 2 :]
        base_values = list(_STATED_CONVERSION_BASE_VALUE.finditer(base_window))
        substantive_base = next(
            (
                match
                for match in reversed(base_values)
                if not _looks_like_temporal_or_structural_number(base_window, match)
            ),
            None,
        )
        if substantive_base is None:
            continue
        spans.append((cue.start(), result.end("value")))
    return tuple(spans)


def _stated_conversion_candidate_contains_formula(
    source_text: str,
    cue: re.Match[str],
    result: re.Match[str],
) -> bool:
    """Keep arithmetic inside a conversion clause out of the scalar exemption."""

    start = cue.start()
    suffix_limit = min(len(source_text), result.end("value") + 160)
    clause_suffix = source_text[result.end("value") : suffix_limit]
    clause_boundary = re.search(r"[.;!?\n]", clause_suffix)
    end = (
        result.end("value") + clause_boundary.start()
        if clause_boundary is not None
        else suffix_limit
    )
    characters = list(source_text[start:end])
    verb_start = result.start("verb") - start
    verb_end = result.end("verb") - start
    characters[verb_start:verb_end] = " " * (verb_end - verb_start)
    candidate = "".join(characters)
    return bool(
        _source_states_nonrounding_computation(candidate)
        or _ROUNDING_LANGUAGE.search(candidate)
    )


def _looks_like_temporal_or_structural_number(
    text: str,
    match: re.Match[str],
) -> bool:
    """Return whether a prospective base value is only metadata or structure."""

    for pattern in (
        _STATED_CONVERSION_DATE,
        _GERMAN_LEGAL_CITATION,
        _ENGLISH_LEGAL_CITATION,
        _STRUCTURAL_REFERENCE,
    ):
        if any(
            metadata.start() <= match.start() and match.end() <= metadata.end()
            for metadata in pattern.finditer(text)
        ):
            return True

    raw = match.group(0).replace(" ", "").replace(".", "").replace(",", ".")
    with contextlib.suppress(ValueError):
        value = float(raw)
        if value.is_integer() and 1900 <= value <= 2100:
            prefix = text[max(0, match.start() - 24) : match.start()]
            if re.search(
                r"\b(?:jahr(?:es)?|year|in|for|für(?:\s+das)?)\s*$",
                prefix,
                re.IGNORECASE,
            ):
                return True

    prefix = text[max(0, match.start() - 32) : match.start()]
    suffix = text[match.end() : min(len(text), match.end() + 8)]
    if re.search(
        r"(?:§+|artikel|art\.?|absatz|abs\.?|satz|nummer|nr\.?|"
        r"section|sec\.?|subsection|paragraph|clause|item)\s*$",
        prefix,
        re.IGNORECASE,
    ):
        return True
    if prefix.endswith("(") and re.match(r"\s*\)", suffix):
        return True
    line_prefix = prefix.rsplit("\n", 1)[-1]
    return bool(not line_prefix.strip() and re.match(r"\s*\.", suffix))


def _without_stated_conversion_results(source_text: str) -> str:
    """Blank stated-result clauses without hiding other source computations."""

    spans = _stated_conversion_result_spans(source_text)
    if not spans:
        return source_text
    characters = list(source_text)
    for start, end in spans:
        characters[start:end] = " " * (end - start)
    return "".join(characters)


def _has_substantive_arithmetic_expression(source_text: str) -> bool:
    """Ignore slash-separated year spans while recognizing actual arithmetic."""

    arithmetic_text = list(source_text)
    for date_match in _STATED_CONVERSION_DATE.finditer(source_text):
        arithmetic_text[date_match.start() : date_match.end()] = " " * (
            date_match.end() - date_match.start()
        )
    masked_source_text = "".join(arithmetic_text)
    if _WORDED_ARITHMETIC_EXPRESSION.search(masked_source_text):
        return True
    for match in _ARITHMETIC_EXPRESSION.finditer(masked_source_text):
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
    extract_numeric_grounding_occurrences: NumericOccurrenceExtractor | None = None,
    extract_named_scalars: NamedScalarExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
    artifact_numeric_values: Sequence[float] | None = None,
    artifact_numeric_bindings: Sequence[tuple[str, float]] | None = None,
    authenticated_same_act_aliases: Sequence[str] = (),
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
                extract_numeric_grounding_occurrences=(
                    extract_numeric_grounding_occurrences or extract_numeric_occurrences
                ),
                extract_named_scalars=extract_named_scalars,
                numeric_value_is_grounded=numeric_value_is_grounded,
                artifact_numeric_values=artifact_numeric_values,
                artifact_numeric_bindings=artifact_numeric_bindings,
                authenticated_same_act_aliases=authenticated_same_act_aliases,
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
    extract_numeric_grounding_occurrences: NumericOccurrenceExtractor,
    extract_named_scalars: NamedScalarExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
    artifact_numeric_values: Sequence[float] | None,
    artifact_numeric_bindings: Sequence[tuple[str, float]] | None,
    authenticated_same_act_aliases: Sequence[str],
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
    named_rules = {
        str(rule.get("name") or "").strip(): rule
        for rule in payload.get("rules", [])
        if isinstance(rule, dict) and str(rule.get("name") or "").strip()
    }
    deferred_paths, imprecise_deferrals = _deferred_coverage(
        payload,
        corpus_citation_path=corpus_citation_path,
        source_text=source_text,
        branches=branches,
        authenticated_same_act_aliases=authenticated_same_act_aliases,
    )
    issues: list[str] = []
    issues.extend(imprecise_deferrals)
    issues.extend(
        _opaque_same_source_condition_input_issues(
            payload,
            source_text=source_text,
            branches=branches,
            principal_rules=principal_rules,
            corpus_citation_path=corpus_citation_path,
        )
    )

    for branch in branches:
        if _path_covered(
            branch.path,
            all_covered_paths,
            deferred_paths,
        ) or _is_marker_only_container(
            branch,
            branches=branches,
            source_text=source_text,
        ):
            continue
        descendants = tuple(
            candidate
            for candidate in branches
            if len(candidate.path) > len(branch.path)
            and candidate.path[: len(branch.path)] == branch.path
        )
        covered_descendant_detail = ""
        if descendants and all(
            _path_covered(
                candidate.path,
                all_covered_paths,
                deferred_paths,
            )
            or _is_marker_only_container(
                candidate,
                branches=branches,
                source_text=source_text,
            )
            for candidate in descendants
        ):
            direct_source_text = _source_branch_direct_text(
                branch,
                branches=branches,
            )
            direct_text, direct_text_truncated = _bounded_source_feedback_preview(
                direct_source_text
            )
            if direct_text:
                direct_end = branch.start + len(direct_source_text)
                truncation_detail = (
                    " The ellipsis is a locator only; copy a short contiguous "
                    "verbatim proof excerpt from that source span, not the "
                    "ellipsis."
                    if direct_text_truncated
                    else ""
                )
                covered_descendant_detail = (
                    " Descendant `source:` citations and proof excerpts cover only "
                    "the descendants, not this substantive parent. Exact parent "
                    f"chapeau (characters {branch.start}:{direct_end}): "
                    f"`{direct_text}`.{truncation_detail} Attach a separate exact "
                    "contiguous "
                    "parent-chapeau excerpt to the source-faithful rule that "
                    "implements or scopes it (use a `versions[N].formula` proof "
                    "atom on the affected principal formula when applicable), or "
                    "precisely defer the parent; do not invent a dummy output."
                )
        issues.append(
            "[complete-source-unit:structure] "
            f"Source branch {branch.label} at "
            f"{_branch_citation(corpus_citation_path, branch)} is neither "
            "encoded nor precisely deferred."
            f"{covered_descendant_detail}"
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
    control_branches = _source_control_branches(
        source_text,
        branches=branches,
        active_branches=active_branches,
        deferred_paths=deferred_paths,
        formula_branches=formula_branches,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    principal_formula_clause_rules = _principal_formula_clause_rules(
        formula_branches,
        principal_rules=principal_rules,
        principal_rule_paths=principal_rule_paths,
        named_rules=named_rules,
        corpus_citation_path=corpus_citation_path,
    )
    source_has_computation = source_states_explicit_computation(source_text)
    if source_has_computation:
        if formula_branches:
            missing_formula_branches = tuple(
                branch
                for branch in formula_branches
                if not principal_formula_clause_rules[branch]
            )
            missing_formula_branches_by_path: dict[
                tuple[str, ...], list[SourceStructureBranch]
            ] = {}
            for missing_branch in missing_formula_branches:
                missing_formula_branches_by_path.setdefault(
                    missing_branch.path,
                    [],
                ).append(missing_branch)
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
                    + _formula_output_binding_feedback(
                        branch,
                        corpus_citation_path=corpus_citation_path,
                        has_path_covering_principal=bool(
                            _rules_covering_branch(
                                branch,
                                principal_rule_paths,
                            )
                        ),
                        same_owner_missing=missing_formula_branches_by_path[
                            branch.path
                        ],
                    )
                )
        elif not all_formula_branches and not _path_covered(
            (), principal_paths, deferred_paths
        ):
            issues.append(
                "[complete-source-unit:formula-output] Explicit source computation "
                "has no principal derived/relation output (`derived` or "
                "`derived_relation`); "
                "parameter-only representation is invalid."
            )
    for branch in control_branches:
        if _rules_covering_branch(branch, principal_rule_paths):
            continue
        issues.append(
            "[complete-source-unit:formula-output] Source-stated control "
            f"{branch.label} in "
            f"{_branch_citation(corpus_citation_path, branch)} has no "
            "principal derived/relation output (`derived` or "
            "`derived_relation`) and is not precisely deferred; "
            "parameter-only representation is invalid."
        )

    numeric_recall_text = authoritative_numeric_recall_text(source_text)
    source_occurrences = tuple(
        occurrence
        for occurrence in extract_numeric_occurrences(numeric_recall_text)
        if not _temporal_occurrence_is_formula_applicability_preface(
            occurrence,
            numeric_recall_text,
        )
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
        formula_environment = _constant_rule_environment(payload)
        if artifact_numeric_bindings is not None:
            formula_environment = _merge_unambiguous_numeric_bindings(
                formula_environment,
                artifact_numeric_bindings,
            )
        issues.extend(
            _companion_test_issues(
                principal_rules,
                parameter_rules={
                    name: rule
                    for name, rule in named_rules.items()
                    if str(rule.get("kind") or "").strip().lower() == "parameter"
                },
                principal_rule_paths=principal_rule_paths,
                principal_formula_clause_rules=principal_formula_clause_rules,
                formula_branches=formula_branches,
                branches=branches,
                source_text=source_text,
                corpus_citation_path=corpus_citation_path,
                deferred_paths=deferred_paths,
                test_cases=test_cases,
                extract_numeric_occurrences=extract_numeric_grounding_occurrences,
                numeric_value_is_grounded=numeric_value_is_grounded,
                formula_environment=formula_environment,
                source_bound_constant_occurrences=(
                    _source_bound_constant_numeric_occurrences(
                        payload,
                        corpus_citation_path=corpus_citation_path,
                        extract_numeric_occurrences=extract_numeric_occurrences,
                    )
                ),
                declared_input_names={
                    str(item.get("name") or "").strip()
                    for item in payload.get("inputs", [])
                    if isinstance(item, dict) and str(item.get("name") or "").strip()
                },
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
            elif _collapse_text(excerpt) in _collapse_text(source_text):
                paths.add(())
        for path in paths:
            all_paths.add(path)
        if kind in {"derived", "derived_relation"} and name:
            principal_rules[name] = rule
            principal_rule_paths.setdefault(name, set()).update(paths)
            for path in paths:
                principal_paths.add(path)
    return all_paths, principal_paths, principal_rules, principal_rule_paths


def _rule_source_excerpts(rule: dict[str, Any]) -> Iterable[tuple[str, str]]:
    return tuple(
        (citation_path, excerpt)
        for _path, citation_path, excerpt in _rule_source_excerpt_atoms(rule)
    )


_PROOF_ATOM_PATH_INDEX_PATTERN = re.compile(r"\[\s*(\d+)\s*\]")


def _formula_proof_version_index(path: str) -> int | None:
    """Normalize a formula proof path the same way as proof validation."""

    normalized = re.sub(r"\s+", "", str(path)).strip().rstrip(".")
    normalized = _PROOF_ATOM_PATH_INDEX_PATTERN.sub(r"[\1]", normalized)
    normalized = re.sub(r"^versions\.", "versions[0].", normalized)
    match = re.match(r"^versions\[(\d+)\]\.formula\b", normalized)
    return int(match.group(1)) if match is not None else None


def _rule_formula_source_excerpts(
    rule: dict[str, Any],
) -> Iterable[tuple[str, str]]:
    return tuple(
        (citation_path, excerpt)
        for path, citation_path, excerpt in _rule_source_excerpt_atoms(rule)
        if _formula_proof_version_index(path) is not None
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
        path = (
            _normalize_atom_path(str(atom.get("path") or ""))
            if isinstance(atom, dict)
            else ""
        )
        excerpt = source.get("excerpt") if isinstance(source, dict) else None
        citation_path = (
            str(source.get("corpus_citation_path") or "").strip()
            if isinstance(source, dict)
            else ""
        )
        if isinstance(excerpt, str) and excerpt.strip() and citation_path and path:
            excerpts.append((path, citation_path, excerpt.strip()))
    return excerpts


def _source_bound_constant_numeric_occurrences(
    payload: dict[str, Any],
    *,
    corpus_citation_path: str,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> dict[str, tuple[NumericOccurrenceLike, ...]]:
    """Index constant-rule numerics with formula proof from this source unit."""

    authoritative_path = corpus_citation_path.strip("/").casefold()
    bound: dict[str, tuple[NumericOccurrenceLike, ...]] = {}
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return bound
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        name = str(rule.get("name") or "").strip()
        if not name or rule.get("kind") != "parameter":
            continue
        occurrences = tuple(
            occurrence
            for path, citation_path, excerpt in _rule_source_excerpt_atoms(rule)
            if re.fullmatch(r"versions(?:\[\d+\])?\.formula", path)
            and citation_path.strip("/").casefold() == authoritative_path
            for occurrence in extract_numeric_occurrences(excerpt)
        )
        if occurrences:
            bound[name] = occurrences
    return bound


def _principal_formula_clause_rules(
    formula_branches: Sequence[SourceStructureBranch],
    *,
    principal_rules: dict[str, dict[str, Any]],
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
    named_rules: dict[str, dict[str, Any]],
    corpus_citation_path: str,
) -> dict[SourceStructureBranch, set[str]]:
    """Bind each computation clause to principal output evidence.

    A direct source path is sufficient when its structural branch contains one
    computation. When several computations share that path, each one needs a
    source-verbatim proof excerpt on the principal rule that claims it.
    """

    clause_count_by_path: dict[tuple[str, ...], int] = {}
    for clause in formula_branches:
        clause_count_by_path[clause.path] = clause_count_by_path.get(clause.path, 0) + 1

    clause_rules: dict[SourceStructureBranch, set[str]] = {}
    normalized_citation_path = corpus_citation_path.strip("/").lower()
    for clause in formula_branches:
        path_rules = set(_rules_covering_branch(clause, principal_rule_paths))
        if clause_count_by_path[clause.path] == 1:
            if path_rules:
                clause_rules[clause] = path_rules
                continue
        candidate_rules = path_rules or {
            rule_name
            for rule_name, paths in principal_rule_paths.items()
            if any(
                path
                and len(path) < len(clause.path)
                and clause.path[: len(path)] == path
                for path in paths
            )
        }
        uses_ancestor_binding = not path_rules
        clause_text = _normalized_formula_clause_text(clause.text)
        rounding_direction = _rounding_only_direction(clause.text)
        clause_rules[clause] = {
            rule_name
            for rule_name in candidate_rules
            if (
                rounding_direction is not None
                and _rule_implements_rounding(
                    principal_rules[rule_name],
                    rounding_direction,
                )
            )
            or any(
                (excerpt_text := _normalized_formula_clause_text(excerpt))
                and excerpt_citation_path.strip("/").lower() == normalized_citation_path
                and source_states_explicit_computation(excerpt)
                and (excerpt_text in clause_text or clause_text in excerpt_text)
                for excerpt_citation_path, excerpt in (
                    _formula_dependency_source_excerpts(
                        principal_rules[rule_name],
                        named_rules=named_rules,
                    )
                    if uses_ancestor_binding
                    else _rule_formula_source_excerpts(principal_rules[rule_name])
                )
            )
        }
    return clause_rules


def _formula_dependency_source_excerpts(
    rule: dict[str, Any],
    *,
    named_rules: dict[str, dict[str, Any]],
) -> tuple[tuple[str, str], ...]:
    """Include direct parameter evidence consumed by a principal formula."""

    excerpts = list(_rule_formula_source_excerpts(rule))
    versions = rule.get("versions")
    if not isinstance(versions, list):
        return tuple(excerpts)
    dependency_names = {
        identifier
        for version in versions
        if isinstance(version, dict) and isinstance(version.get("formula"), str)
        for identifier in _FORMULA_IDENTIFIER.findall(
            _mask_formula_strings_and_comments(version["formula"])
        )
    }
    for dependency_name in dependency_names:
        dependency = named_rules.get(dependency_name)
        if (
            dependency is None
            or str(dependency.get("kind") or "").lower() != "parameter"
        ):
            continue
        excerpts.extend(_rule_formula_source_excerpts(dependency))
    return tuple(excerpts)


def _normalized_formula_clause_text(text: str) -> str:
    """Normalize a clause while ignoring its structural marker."""

    unmarked = _strip_source_clause_marker(text)
    return _collapse_text(unmarked)


def _strip_source_clause_marker(text: str) -> str:
    """Remove a leading paragraph/list/Satz marker, including glued statutes."""

    compound = re.match(
        r"^\s*(?:(?-i:[A-Z])\.)?"
        r"(?P<chain>(?:\([A-Za-z0-9]+\))+)",
        text,
        flags=re.IGNORECASE,
    )
    if compound is not None:
        labels = re.findall(r"\(([A-Za-z0-9]+)\)", compound.group("chain"))
        if labels and all(_is_parenthesized_outline_label(label) for label in labels):
            return text[compound.end() :].lstrip()
    return re.sub(
        r"^\s*(?:"
        r"(?-i:[A-Z])\.(?!\s*(?-i:[A-Z])\.)|"
        r"\d+[a-z]?\.|"
        r"[a-z]\)|"
        r"satz\s+\d+[a-z]?\s*:?\s*|"
        r"\d+(?=(?-i:[A-ZÄÖÜ]))"
        r")\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )


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
    if _FORMULA_ROUNDED_OPERATION_LANGUAGE.search(text):
        return None
    without_rounding = _ROUNDING_LANGUAGE.sub("", text)
    if _source_states_nonrounding_computation(without_rounding):
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
    reference_aliases = _authoritative_source_unit_aliases(corpus_citation_path)
    for reference in reference_aliases:
        escaped_reference = re.escape(reference)
        for match in re.finditer(
            rf"(?<![A-Za-z0-9_]){escaped_reference}"
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
                    next_reference=reference,
                )
                paths.add(
                    (
                        *base_components,
                        *_keyword_path_components(trailing, base_components),
                    )
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

    candidates = _authoritative_source_unit_aliases(corpus_citation_path)
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


def _authoritative_source_unit_aliases(
    corpus_citation_path: str,
) -> tuple[str, ...]:
    """Return canonical and conventional citations for one source unit."""

    canonical = corpus_citation_path.rstrip("/")
    aliases = [canonical, _rulespec_target_base(canonical)]
    parts = canonical.split("/")
    if (
        len(parts) == 5
        and parts[:2] == ["us", "regulation"]
        and all(re.fullmatch(r"[A-Za-z0-9-]+", part) for part in parts[2:])
    ):
        title, part, section = parts[2:]
        aliases.append(f"{title} CFR {part}.{section}")
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _deferred_coverage(
    payload: dict[str, Any],
    *,
    corpus_citation_path: str,
    source_text: str,
    branches: Sequence[SourceStructureBranch],
    authenticated_same_act_aliases: Sequence[str] = (),
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
        display_path: tuple[str, ...] | None = None
        if output_path == base_target:
            path = ()
            display_path = ()
        elif output_path.startswith(f"{base_target}/"):
            display_path = tuple(
                part for part in output_path[len(base_target) + 1 :].split("/") if part
            )
            path = tuple(part.lower() for part in display_path)
        if path is None:
            fragment = output.partition("#")[2]
            jurisdiction = base_target.partition(":")[0].lower()
            uses_local_policy_root = output_path.lower().startswith(
                f"{jurisdiction}:policies/"
            )
            output_parts = tuple(
                part.lower() for part in output_path.rstrip("/").split("/") if part
            )
            matching_branches = [
                branch
                for branch in branches
                if branch.path
                and len(output_parts) >= len(branch.path)
                and output_parts[-len(branch.path) :] == branch.path
            ]
            if fragment and uses_local_policy_root and matching_branches:
                matching_branch = max(
                    matching_branches, key=lambda branch: len(branch.path)
                )
                display_branch = _deferred_branch_display_path(
                    matching_branch.path,
                    corpus_citation_path=corpus_citation_path,
                    branches=branches,
                )
                corrected_output = (
                    f"{base_target}/{'/'.join(display_branch)}#{fragment}"
                )
                issues.append(
                    "[complete-source-unit:deferral] "
                    f"`module.deferred_outputs[{index}].output` uses non-source "
                    f"root `{output_path}` for source branch "
                    f"(`{'/'.join(display_branch)}`). Use the canonical source "
                    f"anchor, for example `{corrected_output}`."
                )
            continue
        reason = str(record.get("reason") or "").strip()
        blocked_by = record.get("blocked_by")
        normalized_base_target = base_target.lower()
        blocker_targets = (
            tuple(item.strip() for item in blocked_by if isinstance(item, str))
            if isinstance(blocked_by, list)
            else ()
        )
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
                    item.lower().split("#", 1)[0] == normalized_base_target
                    or item.lower()
                    .split("#", 1)[0]
                    .startswith(f"{normalized_base_target}/")
                )
                for item in blocker_targets
            )
        )
        source_scope_text = _deferred_source_scope_text(
            path,
            source_text=source_text,
            branches=branches,
        )

        def reason_is_precise_for(
            candidate_scope_text: str,
            candidate_path: tuple[str, ...],
        ) -> bool:
            if "blocked_by" in record:
                return (
                    exact_blockers
                    and bool(_MISSING_DEPENDENCY_LANGUAGE.search(reason))
                    and all(
                        _reason_identifies_blocker(
                            reason,
                            blocker,
                            corpus_citation_path=corpus_citation_path,
                        )
                        for blocker in blocker_targets
                    )
                    and all(
                        _source_scope_identifies_blocker(
                            candidate_scope_text,
                            blocker,
                            corpus_citation_path=corpus_citation_path,
                        )
                        for blocker in blocker_targets
                    )
                )
            return (
                _reason_dependency_is_source_bound(
                    reason,
                    candidate_scope_text,
                    corpus_citation_path=corpus_citation_path,
                    path=candidate_path,
                    authenticated_same_act_aliases=authenticated_same_act_aliases,
                )
                or _reason_names_source_bound_runtime_gap(
                    reason,
                    candidate_scope_text,
                    path=candidate_path,
                    corpus_citation_path=corpus_citation_path,
                )
                or _reason_covers_source_authenticated_repeal(
                    reason,
                    candidate_scope_text,
                    path=candidate_path,
                    corpus_citation_path=corpus_citation_path,
                )
            )

        valid_root_symbol = (
            not path
            and output.count("#") == 1
            and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", output.partition("#")[2]))
        )
        if valid_root_symbol:
            cited_branches = {
                branch.path
                for branch in branches
                if branch.path
                and _reason_cites_exact_current_statute_branch(
                    reason,
                    corpus_citation_path=corpus_citation_path,
                    path=branch.path,
                    strict_terminal=True,
                )
            }
            most_specific_cited_branches = {
                candidate
                for candidate in cited_branches
                if not any(
                    len(other) > len(candidate) and other[: len(candidate)] == candidate
                    for other in cited_branches
                )
            }
            most_specific_source_scopes = {
                candidate: _deferred_source_scope_text(
                    candidate,
                    source_text=source_text,
                    branches=branches,
                )
                for candidate in most_specific_cited_branches
            }
            attempted_runtime_gap = bool(
                _MISSING_DEPENDENCY_LANGUAGE.search(reason)
                and _SOURCE_BOUND_RUNTIME_GAP_LANGUAGE.search(reason)
            )
            common_cited_branches = {
                candidate
                for candidate in cited_branches
                if all(other[: len(candidate)] == candidate for other in cited_branches)
            }
            precise_external_dependency = any(
                (
                    "blocked_by" in record
                    and reason_is_precise_for(candidate_scope_text, candidate)
                )
                or _reason_dependency_is_source_bound(
                    reason,
                    candidate_scope_text,
                    corpus_citation_path=corpus_citation_path,
                    path=candidate,
                    authenticated_same_act_aliases=authenticated_same_act_aliases,
                )
                for candidate, candidate_scope_text in most_specific_source_scopes.items()
            )
            cited_scope_states_explicit_computation = any(
                source_states_explicit_computation(candidate_scope_text)
                for candidate_scope_text in most_specific_source_scopes.values()
            )
            if (
                attempted_runtime_gap
                and not precise_external_dependency
                and cited_scope_states_explicit_computation
                and common_cited_branches
            ):
                explicit_computation_branch = max(common_cited_branches, key=len)
                rendered_branch = "/".join(
                    _deferred_branch_display_path(
                        explicit_computation_branch,
                        corpus_citation_path=corpus_citation_path,
                        branches=branches,
                    )
                )
                issues.append(
                    "[complete-source-unit:deferral] "
                    f"`module.deferred_outputs[{index}]` cites exact source branch "
                    f"(`{rendered_branch}`), but that branch states an explicit "
                    "computation and cannot be deferred as a runtime gap. Declare "
                    "the source-stated operative facts as local RuleSpec inputs and "
                    "encode the principal derived output."
                )
                continue
            cited_branch = (
                next(iter(most_specific_cited_branches))
                if len(most_specific_cited_branches) == 1
                else None
            )
            if cited_branch is not None:
                cited_source_scope_text = _deferred_source_scope_text(
                    cited_branch,
                    source_text=source_text,
                    branches=branches,
                )
                display_cited_branch = _deferred_branch_display_path(
                    cited_branch,
                    corpus_citation_path=corpus_citation_path,
                    branches=branches,
                )
                rendered_branch = "/".join(display_cited_branch)
                if reason_is_precise_for(cited_source_scope_text, cited_branch):
                    fragment = output.partition("#")[2]
                    corrected_output = f"{base_target}/{rendered_branch}#{fragment}"
                    retry_shape = _imprecise_deferral_retry_shape(
                        corpus_citation_path=corpus_citation_path,
                        path=display_cited_branch,
                        reason=reason,
                    )
                    issues.append(
                        "[complete-source-unit:deferral] "
                        f"`module.deferred_outputs[{index}].output` is rooted at the "
                        f"source unit (`{output}`), but its otherwise precise reason "
                        f"cites exact source branch (`{rendered_branch}`). Preserve "
                        f"that branch in the output path, for example "
                        f"`{corrected_output}`.\n{retry_shape}"
                    )
                    continue
        precise = reason_is_precise_for(source_scope_text, path)
        if precise:
            covered.add(path)
        else:
            branch_label = path[0] if path else "source unit"
            rendered_path = "/".join(path) or "<source-unit>"
            retry_shape = _imprecise_deferral_retry_shape(
                corpus_citation_path=corpus_citation_path,
                path=display_path or path,
                reason=reason,
            )
            issues.append(
                "[complete-source-unit:deferral] "
                f"`module.deferred_outputs[{index}]` identifies source branch "
                f"({branch_label}) (`{rendered_path}`) but its deferral does not "
                "name an exact missing dependency, input, or runtime capability.\n"
                f"{retry_shape}"
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


def _deferred_branch_display_path(
    path: tuple[str, ...],
    *,
    corpus_citation_path: str,
    branches: Sequence[SourceStructureBranch],
) -> tuple[str, ...]:
    """Restore source-marker case for federal branch-path retry advice."""

    if not corpus_citation_path.startswith("us/statute/"):
        return path
    display_path: list[str] = []
    for depth, segment in enumerate(path, start=1):
        prefix = path[:depth]
        candidate_texts = (branch.text for branch in branches if branch.path == prefix)
        source_scope_text = max(candidate_texts, key=len, default="")
        marker_match = re.match(
            r"\s*(?P<markers>(?:\(\s*[A-Za-z0-9]+\s*\)\s*)+)",
            source_scope_text,
        )
        if marker_match is None:
            display_path.append(segment)
            continue
        marker_labels = re.findall(
            r"\(\s*([A-Za-z0-9]+)\s*\)",
            marker_match.group("markers"),
        )
        display_segment = segment
        for width in range(min(len(prefix), len(marker_labels)), 0, -1):
            if tuple(label.lower() for label in marker_labels[:width]) == tuple(
                part.lower() for part in prefix[-width:]
            ):
                display_segment = marker_labels[width - 1]
                break
        display_path.append(display_segment)
    return tuple(display_path)


def _reason_identifies_blocker(
    reason: str,
    blocker: str,
    *,
    corpus_citation_path: str | None = None,
) -> bool:
    """Bind a typed blocker to the legal section or concept named in prose."""

    target_path, _separator, symbol = blocker.partition("#")
    section = target_path.rstrip("/").rsplit("/", 1)[-1]
    target_instrument = _citation_instrument_identity(target_path)
    corpus_instrument = (
        _citation_instrument_identity(corpus_citation_path)
        if corpus_citation_path
        else None
    )
    normalized_reason = re.sub(
        r"[^a-z0-9äöüß]+",
        " ",
        reason.lower(),
    ).strip()
    if target_instrument is not None and corpus_instrument is not None:
        if target_instrument[:2] != corpus_instrument[:2]:
            if not _reason_names_absolute_target(
                normalized_reason,
                target_instrument,
            ):
                return False
        elif target_instrument[2] != corpus_instrument[
            2
        ] and not _reason_names_target_instrument(
            normalized_reason,
            target_instrument,
        ):
            return False
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
    normalized_symbol = re.sub(r"[^a-z0-9äöüß]+", " ", symbol.lower()).strip()
    return bool(
        normalized_symbol
        and re.search(
            rf"(?<![a-z0-9äöüß]){re.escape(normalized_symbol)}"
            r"(?![a-z0-9äöüß])",
            normalized_reason,
        )
    )


def _source_scope_identifies_blocker(
    source_scope_text: str,
    blocker: str,
    *,
    corpus_citation_path: str,
) -> bool:
    """Reject citations expressly described as unrelated to the source rule."""

    if not _reason_identifies_blocker(
        source_scope_text,
        blocker,
        corpus_citation_path=corpus_citation_path,
    ):
        return False
    target_path = blocker.partition("#")[0]
    section = target_path.rstrip("/").rsplit("/", 1)[-1]
    references = tuple(
        match
        for match in _EXPLICIT_LEGAL_SECTION_REFERENCE.finditer(source_scope_text)
        if match.group("section").lower() == section.lower()
    )
    if not references:
        return True
    for reference in references:
        clause_start = (
            max(
                source_scope_text.rfind(separator, 0, reference.start())
                for separator in (".", ";", "\n")
            )
            + 1
        )
        following_stops = [
            position
            for separator in (".", ";", "\n")
            if (position := source_scope_text.find(separator, reference.end())) >= 0
        ]
        clause_end = min(following_stops, default=len(source_scope_text))
        clause = source_scope_text[clause_start:clause_end]
        if re.search(
            r"\b(?:"
            r"unberührt\s+bleib\w*|bleib\w*\s+unberührt|"
            r"without\s+prejudice\s+to|remain\w*\s+unaffected|"
            r"does\s+not\s+affect"
            r")\b",
            clause,
            flags=re.IGNORECASE,
        ):
            continue
        if _source_clause_links_dependency(
            clause,
            reference_start=reference.start() - clause_start,
            reference_end=reference.end() - clause_start,
        ):
            return True
    return False


def _source_clause_links_dependency(
    clause: str,
    *,
    reference_start: int,
    reference_end: int,
) -> bool:
    """Require syntax that makes the cited provision operative or required."""

    before = clause[:reference_start]
    after = clause[reference_end:]
    dependency_link = re.search(
        r"\b(?:"
        r"nach|gemäß|laut|entsprechend|under|according\s+to|pursuant\s+to|"
        r"in\s+accordance\s+with(?:\s+the\s+provisions?\s+of)?|"
        r"abhängig\s+von|depends?\s+on|setzt|requires?|benötigt"
        r")\s*$",
        before,
        flags=re.IGNORECASE,
    )
    if dependency_link:
        preceding_link = before[: dependency_link.start()]
        is_in_accordance_link = bool(
            re.fullmatch(
                r"in\s+accordance\s+with(?:\s+the\s+provisions?\s+of)?",
                dependency_link.group(0).strip(),
                flags=re.IGNORECASE,
            )
        )
        negated_in_accordance_link = (
            is_in_accordance_link
            and _source_link_scope_has_unreset_negation(preceding_link)
        )
        if not negated_in_accordance_link:
            return True
    if re.search(
        r"\b(?:voraussetzung\w*|bedingung\w*|conditions?)"
        r"[^.;]{0,100}\b(?:des|der|nach|under)\s*$",
        before,
        flags=re.IGNORECASE,
    ):
        return True
    if re.match(
        r"\s*(?:ist|sind|wird|werden|is|are)?\s*"
        r"(?:erforderlich|maßgeblich|vorausgesetzt|benötigt|required|needed)\b",
        after,
        flags=re.IGNORECASE,
    ):
        return True
    return (
        re.search(
            r"\b(?:voraussetzung\w*|bedingung\w*|conditions?)"
            r"[^.;]{0,160}\b(?:vorliegen|erfüllt|bestehen|apply|hold)\b",
            clause,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _source_scope_cites_usc_dependency(
    source_scope_text: str,
    *,
    title: str,
    section: str,
    fragments: tuple[str, ...],
    allow_relative_reference: bool,
) -> bool:
    """Return whether an operative source clause cites the same USC provision."""

    normalized_section = normalize_rulespec_path_segment(section.lower())
    qualified_references = _qualified_usc_dependencies(source_scope_text)
    references: list[re.Match[str]] = list(qualified_references)
    if allow_relative_reference:
        references.extend(
            reference
            for reference in _RELATIVE_USC_DEFERRAL_DEPENDENCY.finditer(
                source_scope_text
            )
            if not any(
                qualified.start() <= reference.start()
                and reference.end() <= qualified.end()
                for qualified in qualified_references
            )
        )
    for reference in references:
        reference_title = reference.groupdict().get("title")
        if (
            (reference_title is not None and reference_title.lower() != title.lower())
            or normalize_rulespec_path_segment(reference.group("section").lower())
            != normalized_section
            or _usc_dependency_fragments(reference) != fragments
        ):
            continue
        clause_start = (
            max(
                source_scope_text.rfind(separator, 0, reference.start())
                for separator in (".", ";", "\n")
            )
            + 1
        )
        following_stops = [
            position
            for separator in (".", ";", "\n")
            if (position := source_scope_text.find(separator, reference.end())) >= 0
        ]
        clause_end = min(following_stops, default=len(source_scope_text))
        if _source_clause_links_dependency(
            source_scope_text[clause_start:clause_end],
            reference_start=reference.start() - clause_start,
            reference_end=reference.end() - clause_start,
        ):
            return True
    return False


def _source_scope_cites_louisiana_rs_dependency(
    source_scope_text: str,
    *,
    title: str,
    section: str,
    fragments: tuple[str, ...],
    tail_identity: str,
) -> bool:
    """Return whether an operative Louisiana clause cites one exact R.S. dependency."""

    # Official Louisiana source prose is hard-wrapped within paragraphs. Preserve
    # blank-line paragraph boundaries while preventing a visual line wrap from
    # splitting one operative predicate such as ``shall be\ndetermined``.
    source_scope_text = re.sub(r"(?<!\n)\n(?![ \t]*\n)", " ", source_scope_text)
    normalized_section = normalize_rulespec_path_segment(section.lower())
    references = tuple(
        reference
        for reference in _qualified_louisiana_rs_dependencies(source_scope_text)
        if not _louisiana_rs_match_has_detached_structural_subject(
            source_scope_text, reference
        )
    )
    for reference in references:
        if (
            reference.group("title").lower() != title.lower()
            or normalize_rulespec_path_segment(reference.group("section").lower())
            != normalized_section
            or _usc_dependency_fragments(reference) != fragments
            or _louisiana_rs_tail_identity(reference) != tail_identity
        ):
            continue
        clause_start, clause_end = _reason_clause_bounds(source_scope_text, reference)
        clause = source_scope_text[clause_start:clause_end]
        clause_references = tuple(
            candidate
            for candidate in references
            if clause_start <= candidate.start() and candidate.end() <= clause_end
        )
        syntax_reference = reference
        reference_index = clause_references.index(reference)
        starts_finite_clause = _louisiana_reference_starts_finite_clause(
            source_scope_text[reference.end() : clause_end]
        )
        outer_predicate_follows_citation_list = (
            starts_finite_clause
            and _louisiana_citation_list_completes_outer_subject(
                source_scope_text[clause_start : clause_references[0].start()]
            )
        )
        detached_coordinated_relative = bool(
            starts_finite_clause
            and re.match(
                r"\s*,?\s*(?:along|together)\s+with\b[^,.;]{0,120}"
                r"\b(?:that|which|who)\b",
                source_scope_text[reference.end() : clause_end],
                flags=re.IGNORECASE,
            )
        )
        if detached_coordinated_relative and not outer_predicate_follows_citation_list:
            continue
        while reference_index > 0:
            previous_reference = clause_references[reference_index - 1]
            bridge = source_scope_text[
                previous_reference.end() : syntax_reference.start()
            ]
            if not re.fullmatch(
                r"\s*(?:,\s*(?:(?:and/or|and|or)\s*)?|"
                r"(?:and/or|and|or|as\s+well\s+as|together\s+with|along\s+with|"
                r"combined\s+with|plus|&)\s*)",
                bridge,
                re.IGNORECASE,
            ):
                break
            # Finite-clause evidence wins over optional comma punctuation.  The
            # exception is a citation list completing a reduced-passive outer
            # subject (``the amount determined under A, B, and C is rounded``).
            if starts_finite_clause and not outer_predicate_follows_citation_list:
                break
            syntax_reference = previous_reference
            reference_index -= 1
        previous_reference_end = max(
            (
                candidate.end()
                for candidate in references
                if clause_start <= candidate.start()
                and candidate.end() <= syntax_reference.start()
            ),
            default=clause_start,
        )
        local_before = source_scope_text[
            previous_reference_end : syntax_reference.start()
        ]
        if _louisiana_disclaimer_governs_reference(local_before):
            continue
        reference_start = syntax_reference.start() - clause_start
        if not _louisiana_source_link_is_operative(
            clause,
            reference_start=reference_start,
            reference_end=reference.end() - clause_start,
        ):
            continue
        return True
    return False


_LOUISIANA_CITATION_FINITE_PREDICATE = re.compile(
    r"\s*(?:,\s*)?(?:(?:however|merely|only),?\s+)*(?:"
    r"(?:(?:and|together\s+with)\s+(?:its|the)\s+"
    r"[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,3},?\s+)?"
    r"(?!(?:along|as|by|combined|for|from|if|in|of|or|plus|through|to|"
    r"together|under|when|where|with)\b)(?![A-Za-z][A-Za-z-]*ly\b)"
    r"[A-Za-z][A-Za-z-]*\b)",
    flags=re.IGNORECASE,
)


def _louisiana_reference_starts_finite_clause(after_reference: str) -> bool:
    """Reject citation-list inheritance when the citation is a new clause subject."""

    suffix = re.sub(
        r"^\s*,?\s*(?:(?:however|merely|only),?\s+)*",
        "",
        after_reference,
        flags=re.IGNORECASE,
    )
    bounded_modifier = re.match(
        r"(?:in\s+its\s+(?:current|amended|revised)\s+form|"
        r"as\s+(?:currently|further|subsequently)\s+amended),?\s*",
        suffix,
        flags=re.IGNORECASE,
    )
    if bounded_modifier:
        return _louisiana_reference_starts_finite_clause(
            suffix[bounded_modifier.end() :]
        )
    modal_predicate = re.match(
        r"(?:along|together)\s+with\s+(?P<premodal>[^,.;]{1,80}?),?\s+"
        r"(?:(?:[A-Za-z][A-Za-z-]*ly)\s+)?"
        r"(?:shall|should|must|will|would|can|could|may|might|does?|did)\s+"
        r"(?:not\s+)?[A-Za-z][A-Za-z-]*\b",
        suffix,
        flags=re.IGNORECASE,
    )
    if modal_predicate and not re.search(
        r"\b(?:that|which|who)\b",
        modal_predicate.group("premodal"),
        flags=re.IGNORECASE,
    ):
        return True
    coordinated_body = re.match(
        r"(?:along|together)\s+with\s+(?P<body>[^,.;]{1,120})",
        suffix,
        flags=re.IGNORECASE,
    )
    if coordinated_body:
        body = coordinated_body.group("body")
        relative = re.search(r"\b(?:that|which|who)\b", body, flags=re.IGNORECASE)
        if relative is not None:
            relative_tail = body[relative.end() :]
            relative_nouns = {
                "agencies",
                "analysis",
                "authors",
                "codes",
                "guidelines",
                "instructions",
                "irs",
                "ordinances",
                "policies",
                "process",
                "standards",
                "forms",
                "provisions",
                "regulations",
                "rules",
                "sections",
                "series",
                "statutes",
                "taxpayers",
                "terms",
            }
            relative_predicate = next(
                (
                    token
                    for token in re.finditer(r"[A-Za-z][A-Za-z-]*", relative_tail)
                    if token.group(0).lower() in {"was", "were"}
                    or token.group(0).lower()
                    in {
                        "administer",
                        "address",
                        "apply",
                        "cover",
                        "consult",
                        "enforce",
                        "follow",
                        "govern",
                        "issues",
                        "made",
                        "obey",
                        "publishes",
                        "rely",
                        "review",
                        "retain",
                        "use",
                        "wrote",
                    }
                    or token.group(0).lower().endswith("ed")
                    or (
                        token.group(0).lower().endswith("s")
                        and not token.group(0).isupper()
                        and not token.group(0).lower().endswith(("is", "ss", "us"))
                        and token.group(0).lower() not in relative_nouns
                    )
                ),
                None,
            )
            body = (
                relative_tail[relative_predicate.end() :] if relative_predicate else ""
            )
        if re.search(
            r"\b(?:(?:already|currently|now|presently|still)\s+"
            r"(?:effective|in\s+(?:effect|force))|"
            r"effective\s+(?:today|on\s+[A-Za-z]+\s+\d{1,2}))\s*$",
            body,
            flags=re.IGNORECASE,
        ):
            return False
        words = re.findall(r"[A-Za-z][A-Za-z-]*", body)
        if relative is not None and re.search(
            r"\b(?:can|could|may|might|must|shall|should|will|would)\s+"
            r"[A-Za-z][A-Za-z-]*\b|\b[A-Za-z][A-Za-z-]*ed\b",
            body,
            flags=re.IGNORECASE,
        ):
            return True
        for index, word in enumerate(words):
            if (
                (index > 0 or relative is not None)
                and word.lower().endswith("s")
                and word.lower()
                not in {
                    "codes",
                    "cases",
                    "guidelines",
                    "instructions",
                    "irs",
                    "notices",
                    "ordinances",
                    "policies",
                    "requirements",
                    "schedules",
                    "standards",
                    "forms",
                    "provisions",
                    "regulations",
                    "rules",
                    "sections",
                    "statutes",
                    "tables",
                    "taxpayers",
                    "terms",
                    "this",
                    "those",
                    "directives",
                }
                and (
                    relative is not None
                    or any(
                        previous.lower() not in {"a", "an", "its", "the"}
                        for previous in words[:index]
                    )
                )
            ):
                return True
        if relative is not None and not body.strip():
            return False
    if re.match(
        r"(?:along|together)\s+with\s+[^,.;]{1,80}?,?\s+"
        r"(?:(?:[A-Za-z][A-Za-z-]*ly)\s+)?"
        r"(?:is|are|was|were|became|becomes?|remained|remains?|"
        r"applies?|controls?|governs?|"
        r"sets?|determines?|calculates?|computes?|establishes?|imposes?)\b",
        suffix,
        flags=re.IGNORECASE,
    ):
        return True
    if re.match(
        r"(?:along|together)\s+with\s+"
        r"(?:[A-Za-z][A-Za-z-]*\s+){1,6}?"
        r"(?:adopt(?:s|ed)|amend(?:s|ed)|chang(?:es|ed)|creat(?:es|ed)|"
        r"enact(?:s|ed)|fil(?:es|ed)|gives?|gave|has|have|had|holds?|held|"
        r"issu(?:es|ed)|makes?|made|puts?|put|requir(?:es|ed)|seeks?|sought|"
        r"takes?|took|writes?|wrote|brought|built|bought|caught|chose|drew|"
        r"found|got|heard|hid|kept|laid|left|lost|met|paid|ran|read|saw|sent|"
        r"set|sold|taught|told|won)\b"
        r"\s+(?:a\s+(?!(?:(?:calendar|fiscal|taxable)\s+)?"
        r"(?:century|day|decade|generation|month|period|quarter|time|week|year)\b)|"
        r"an|no|"
        r"the\s+(?!(?:(?:current|following|last|next|prior|"
        r"previous|same)\s+)?(?:day|month|quarter|week|year)\b)|filing|tax|effect\b|"
        r"(?:this|that)\s+(?!(?:day|month|quarter|week|year)\b))",
        suffix,
        flags=re.IGNORECASE,
    ):
        return True
    list_modifier = re.match(
        r"(?:(?:taken|read|considered)\s+together|together|along)"
        r"(?:\s+with\s+(?:its|the|two|three|multiple)\s+"
        r"[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,3})?|"
        r"(?:alone|both|collectively|equally|jointly|respectively)\b|"
        r"(?:if|when|whenever|where)\s+(?:applicable|required)\b",
        suffix,
        flags=re.IGNORECASE,
    )
    if list_modifier:
        remainder = suffix[list_modifier.end() :].lstrip(" ,")
        return bool(remainder) and _louisiana_reference_starts_finite_clause(remainder)
    suffix = re.sub(
        r"^(?:as\s+amended|where\s+applicable),?\s*",
        "",
        suffix,
        flags=re.IGNORECASE,
    )
    return bool(
        re.match(
            r"(?:(?:and|together\s+with)\s+(?:its|the)\s+"
            r"[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,3},?\s+)?"
            r"(?:[A-Za-z][A-Za-z-]*ly\s+)?"
            r"(?!(?:along|as|by|combined|for|from|if|in|of|or|plus|through|to|"
            r"together|under|when|whenever|where|with)\b)"
            r"[A-Za-z][A-Za-z-]*\b",
            suffix,
            flags=re.IGNORECASE,
        )
    )


def _louisiana_citation_list_completes_outer_subject(
    before_first_reference: str,
) -> bool:
    """Recognize a citation list embedded in a reduced-passive subject."""

    reduced = re.search(
        r"(?P<subject>\b[^,.;]{1,100}?)\s+"
        r"(?:(?:[A-Za-z][A-Za-z-]*ly|hereby|then)\s+)*"
        r"(?:allowed|assessed|calculated|computed|derived|determined|established|"
        r"fixed|imposed|levied|prescribed|set|specified)\s+"
        r"(?:according\s+to|pursuant\s+to|under|in\s+accordance\s+with)\s*$",
        before_first_reference,
        flags=re.IGNORECASE,
    )
    if reduced is None:
        return False
    return not re.search(
        r"\b(?:am|are|be|been|being|becomes?|is|remains?|was|were)\s*$",
        reduced.group("subject"),
        flags=re.IGNORECASE,
    )


_LOUISIANA_POSITIVE_OPERATIVE_PREDICATE = re.compile(
    r"\b(?:"
    r"(?:is|are|was|were|becomes?|remains?)\s+"
    r"(?:(?:[A-Za-z][A-Za-z-]*ly|hereby|then)\s+)*"
    r"(?:allowed|assessed|available|calculated|computed|derived|determined|rounded|"
    r"established|fixed|imposed|levied|prescribed|set|specified)|"
    r"(?:shall|may|must|will|would|can|could)\s+be\s+"
    r"(?:(?:[A-Za-z][A-Za-z-]*ly|hereby|then)\s+)*"
    r"(?:allowed|assessed|available|calculated|computed|derived|determined|rounded|"
    r"established|fixed|imposed|levied|prescribed|set|specified)|"
    r"(?:shall|may|must|will|would|can|could|does|do|did)\s+"
    r"(?:(?:[A-Za-z][A-Za-z-]*ly|hereby|then)\s+)*"
    r"(?:appl(?:y|ies)|arise[sd]?|calculates?|computes?|derives?|determines?|"
    r"establishes?|imposes?)|"
    r"(?:is|are|was|were)\s+to\s+be\s+(?:allowed|assessed|available|calculated|"
    r"computed|derived|determined|established|fixed|imposed|levied|prescribed|rounded|set|specified)|"
    r"except\s+as\s+(?:(?:[A-Za-z][A-Za-z-]*ly|hereby|then)\s+)*"
    r"(?:allowed|assessed|available|calculated|computed|derived|determined|"
    r"established|fixed|imposed|levied|prescribed|set|specified)|"
    r"(?:amount|base|calculation|computation|credit|deduction|formula|income|liability|"
    r"percentage|rate|result|rule|schedule|statute|table|tax|threshold)\s+"
    r"(?:(?:[A-Za-z][A-Za-z-]*ly|hereby|then)\s+)*"
    r"(?:allowed|assessed|available|calculated|computed|derived|determined|"
    r"established|fixed|imposed|levied|prescribed|set|specified)|"
    r"(?:agency|commission|commissioner|court|department|employer|rule|secretary|"
    r"statute|taxpayer)\s+"
    r"(?:(?:[A-Za-z][A-Za-z-]*ly|hereby|then)\s+)*"
    r"(?:allows?|assesses?|calculates?|computes?|derives?|determines?|"
    r"establishes?|fixes?|imposes?|levies?|prescribes?|sets?|specifies?)\s+"
    r"(?:the\s+)?(?:[A-Za-z][A-Za-z-]*\s+){0,3}(?:amount|base|calculation|computation|credit|deduction|formula|"
    r"liability|rate|rule|tax)|"
    r"(?:but|yet)\s+(?:allowed|assessed|available|calculated|computed|derived|"
    r"determined|established|fixed|imposed|levied|prescribed|rounded|set|specified)|"
    r"(?:amount|base|calculation|computation|credit|deduction|income|liability|rate|"
    r"rule|statute|tax)\s+"
    r"(?:(?:[A-Za-z][A-Za-z-]*ly|hereby|then)\s+)*"
    r"(?:appl(?:y|ies)|arise[sd]?|calculates?|computes?|derives?|determines?|"
    r"establishes?|imposes?)"
    r")\b",
    flags=re.IGNORECASE,
)


_LOUISIANA_REPORTING_PREDICATE = re.compile(
    r"\b(?:alleg\w*|answer\w*|assert\w*|aver(?:s|red|ring)?|believ\w*|"
    r"claim\w*|conclud\w*|"
    r"conjectur\w*|contend\w*|declar\w*|estimat\w*|infer\w*|maintain\w*|"
    r"explain\w*|not(?:e|es|ed|ing)|observ\w*|opin\w*|posit\w*|postulat\w*|"
    r"acknowledg\w*|confirm\w*|conclusion|predict\w*|presum\w*|purport\w*|"
    r"recall\w*|recit\w*|recount\w*|remark\w*|report\w*|"
    r"said|say\w*|speculat\w*|tell(?:s|ing)?|told|writ(?:e|es|ing|ten)|wrote|"
    r"(?<!-)stat(?:e|es|ed|ing)|surmis\w*|"
    r"testif\w*|underst(?:and|ands|anding|ood))\b",
    flags=re.IGNORECASE,
)

_LOUISIANA_NONOPERATIVE_LANGUAGE = re.compile(
    r"\b(?:alleged(?:ly)?|apparently|assumed|candidate|conceivable|conceivably|"
    r"assumption|conditionally|conceptual|contrived|demonstration|draft|dubious|"
    r"example|fictional(?:ly)?|"
    r"hypothetical(?:ly)?|"
    r"illustration|illustrative|imagined|mock|nonbinding|notional|ostensibly|perhaps|"
    r"nominally|possibly|potentially|proposed|prototype|purported(?:ly)?|"
    r"putative(?:ly)?|questionable|"
    r"questionably|reportedly|sample|scenario|seemingly|specimen|supposedly|"
    r"theoretical(?:ly)?|"
    r"unlikely|unproven|unsupported|unverified)\b",
    flags=re.IGNORECASE,
)


def _louisiana_text_has_nonoperative_framing(text: str) -> bool:
    """Classify bounded hypothetical, evidential, or reported language."""

    if _LOUISIANA_NONOPERATIVE_LANGUAGE.search(text):
        return True
    if re.search(
        r"\b(?:affidavit|testimony)\b[^.;]{0,80}\b"
        r"(?:provides?(?:\s*,?\s*in\s+(?:relevant|pertinent)\s+part\s*,?)?\s+that|"
        r"reads?(?:\s*,?\s*in\s+(?:relevant|pertinent)\s+part\s*,?)?\s+"
        r"(?:substantially\s+|in\s+substance\s+)?as\s+follows|"
        r"sets?\s+forth(?:\s*,?\s*in\s+(?:relevant|pertinent)\s+part\s*,?)?"
        r"\s+that)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\bit\s+(?:(?:has|had)\s+been|is|was)\s+provided\s+in\s+(?:the\s+)?"
        r"(?:affidavit|testimony)\s+that\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:affidavit|agency|analysis|analyst|article|author|commentator|"
        r"document|expert|narrative|professor|report|source|statement|testimony|"
        r"witness)\b[^.;]{0,120}" + _LOUISIANA_REPORTING_PREDICATE.pattern,
        text,
        flags=re.IGNORECASE,
    ):
        return True
    for reporting in _LOUISIANA_REPORTING_PREDICATE.finditer(text):
        report_tail = text[reporting.end() :]
        preceding_state = text[: reporting.start()]
        state_tail_is_executable = bool(
            re.match(
                r"(?:-|\s)+(?:[A-Za-z][A-Za-z-]*\s+){0,4}"
                r"(?:amount|base|calculation|computation|credit|deduction|formula|"
                r"income|liability|percentage|rate|result|rule|schedule|table|tax|"
                r"threshold)\b",
                report_tail,
                flags=re.IGNORECASE,
            )
        )
        preceding_word = re.search(r"([A-Za-z][A-Za-z-]*)\s*$", preceding_state)
        state_descriptors = {
            "adjusted",
            "applicable",
            "excess",
            "federal",
            "final",
            "gross",
            "louisiana",
            "local",
            "net",
            "taxable",
            "various",
        }
        if reporting.group(0).lower() == "state" and (
            not preceding_state.strip()
            or re.search(
                r"\b(?:a|an|in|the|this|that)\s*$",
                preceding_state,
                flags=re.IGNORECASE,
            )
            or (
                state_tail_is_executable
                and preceding_word is not None
                and preceding_word.group(1).lower() in state_descriptors
            )
        ):
            continue
        if (
            reporting.group(0).lower() == "said"
            and (
                not preceding_state.strip()
                or re.search(
                    r"\b(?:the|this|that)\s*$",
                    preceding_state,
                    flags=re.IGNORECASE,
                )
            )
            and re.match(
                r"\s+(?:[A-Za-z][A-Za-z-]*\s+){0,4}"
                r"(?:amount|base|calculation|computation|credit|deduction|formula|"
                r"income|liability|percentage|rate|result|rule|schedule|table|tax|"
                r"threshold)\b",
                report_tail,
                flags=re.IGNORECASE,
            )
        ):
            continue
        if (
            reporting.group(0).lower() == "understood"
            and re.search(
                r"(?:^|[.;])\s*(?:(?:a|an|the|this|that)\s+)?"
                r"(?:[A-Za-z][A-Za-z-]*ly\s+)?$",
                preceding_state,
                flags=re.IGNORECASE,
            )
            and re.match(
                r"\s+(?:[A-Za-z][A-Za-z-]*\s+){0,4}"
                r"(?:amount|base|calculation|computation|credit|deduction|formula|"
                r"income|liability|percentage|rate|result|rule|schedule|table|tax|"
                r"threshold)\b",
                report_tail,
                flags=re.IGNORECASE,
            )
        ):
            continue
        preceding_compound_head = (
            preceding_word.group(1).lower() if preceding_word is not None else ""
        )
        if (
            preceding_compound_head
            in {
                "amount",
                "base",
                "calculation",
                "computation",
                "credit",
                "deduction",
                "formula",
                "income",
                "liability",
                "rate",
                "rule",
                "tax",
            }
            and reporting.group(0).lower().endswith("ing")
            and re.match(
                r"\s+(?:[A-Za-z][A-Za-z-]*\s+){0,2}"
                r"(?:amount|base|calculation|computation|credit|deduction|formula|"
                r"income|liability|rate|rule|tax)\s+"
                r"(?:(?:depends?|requires?|relies?)\b|$)|"
                r"\s+(?:calculation|computation|formula)\s*$",
                report_tail,
                flags=re.IGNORECASE,
            )
        ):
            continue
        report_tail = re.sub(
            r"^\s*(?:(?:[:,;]|\(|\[|[—–])\s*){0,2}"
            r"in\s+(?:(?:relevant|pertinent)\s+)?(?:parts?|portions?)\s*"
            r"(?:(?:[,;]|\)|\]|[—–])\s*){0,2}"
            r"(?:as\s+follows)?",
            " as follows",
            report_tail,
            flags=re.IGNORECASE,
        )
        has_complement = (
            re.match(
                r"\s*(?:(?:(?:in\s+(?:(?:relevant|pertinent)\s+)?"
                r"(?:parts?|portions?)"
                r"(?:\s+as\s+follows)?|(?:substantially\s+|in\s+substance\s+)?"
                r"as\s+follows)|"
                r"the\s+following"
                r"(?:\s+[A-Za-z][A-Za-z-]*){0,2})\s*)?"
                r"(?:[:,]|[—–-])?\s*[\"'“”‘’]?\s*"
                r"(?:(?:[A-Za-z][A-Za-z-]*ly)\s+)*"
                r"(?:,?\s*(?:with|without)\s+"
                r"(?:[A-Za-z][A-Za-z-]*\s+){0,3}[A-Za-z][A-Za-z-]*,?\s*)?"
                r"(?:(?:it|this|that)\s+"
                r"(?:is|are|was|were|shall|may|must|will|would|can|could)\b|"
                r"(?:that\s+)?(?:the\s+)?(?:[A-Za-z][A-Za-z-]*\s+){0,3}"
                r"(?:amount|base|calculation|computation|credit|deduction|formula|"
                r"income|liability|rate|rule|tax)\b)",
                report_tail,
                flags=re.IGNORECASE,
            )
            or re.match(
                r"\s*,\s*[^,.;]{1,60},\s*that\s+(?:the\s+)?"
                r"(?:[A-Za-z][A-Za-z-]*\s+){0,3}"
                r"(?:amount|base|calculation|computation|credit|deduction|formula|"
                r"income|liability|rate|rule|tax)\b",
                report_tail,
                flags=re.IGNORECASE,
            )
            or re.match(
                r"\s*,?\s*(?:[A-Za-z][A-Za-z-]*\s+){1,5}"
                r"that\s+(?:the\s+)?(?:[A-Za-z][A-Za-z-]*\s+){0,3}"
                r"(?:amount|base|calculation|computation|credit|deduction|formula|"
                r"income|liability|rate|rule|tax)\b",
                report_tail,
                flags=re.IGNORECASE,
            )
        )
        if not has_complement:
            continue
        local_subject = re.split(r"[,.;]", text[: reporting.start()])[-1]
        subject_core = re.split(
            r"\bof\b", local_subject, maxsplit=1, flags=re.IGNORECASE
        )[0]
        subject_tokens = re.findall(r"[A-Za-z]+", subject_core.lower())
        legal_speaker = bool(
            subject_tokens
            and subject_tokens[-1]
            in {
                "act",
                "code",
                "court",
                "department",
                "law",
                "legislature",
                "ordinance",
                "provision",
                "regulation",
                "rule",
                "section",
                "statute",
            }
        )
        authoritative_speech = re.fullmatch(
            r"(?:assert\w*|conclud\w*|declar\w*|presum\w*|report\w*|say\w*|"
            r"stat(?:e|es|ed|ing))",
            reporting.group(0),
            flags=re.IGNORECASE,
        )
        if legal_speaker and authoritative_speech:
            continue
        return True
    return False


def _louisiana_source_link_is_operative(
    clause: str,
    *,
    reference_start: int,
    reference_end: int,
) -> bool:
    """Require a local positive executable predicate governing the R.S. link."""

    before = clause[:reference_start]
    if _louisiana_reference_has_nonoperative_postfix(clause[reference_end:]):
        return False
    dependency_link = re.search(
        r"\b(?:under(?:\s+(?:both|either))?|according\s+to|pursuant\s+to|"
        r"in\s+accordance\s+with(?:\s+the\s+provisions?\s+of)?|"
        r"at\s+(?:the\s+)?(?:amounts?|percentages?|rates?)\s+"
        r"provided(?:\s+for)?\s+in|"
        r"depends?(?:\s*,?\s*(?:directly|entirely|necessarily|primarily|solely|"
        r"ultimately|in\s+part)\s*,?)*\s+(?:on|upon)|"
        r"is\s+(?:(?:directly|entirely|necessarily|primarily|solely|ultimately)\s+)*"
        r"dependent(?:\s*,?\s*(?:directly|entirely|necessarily|primarily|solely|"
        r"ultimately|in\s+part)\s*,?)*\s+(?:on|upon)|"
        r"relies?(?:\s*,?\s*(?:directly|entirely|necessarily|primarily|solely|"
        r"ultimately|in\s+part)\s*,?)*\s+"
        r"(?:on|upon)|requires?)\s*$",
        before,
        flags=re.IGNORECASE,
    )
    if dependency_link is None:
        return False
    linker_text = dependency_link.group(0).strip().lower()
    governing_scope = before[: dependency_link.start()]
    if _louisiana_scope_has_nonoperative_framing(governing_scope):
        return False
    if re.fullmatch(
        r"(?:depends?(?:\s*,?\s*(?:directly|entirely|necessarily|primarily|solely|"
        r"ultimately|in\s+part)\s*,?)*\s+(?:on|upon)|"
        r"is\s+(?:(?:directly|entirely|necessarily|primarily|solely|ultimately)\s+)*"
        r"dependent(?:\s*,?\s*(?:directly|entirely|necessarily|primarily|solely|"
        r"ultimately|in\s+part)\s*,?)*\s+(?:on|upon)|"
        r"relies?(?:\s*,?\s*(?:directly|entirely|necessarily|primarily|solely|"
        r"ultimately|in\s+part)\s*,?)*\s+"
        r"(?:on|upon)|requires?)",
        linker_text,
    ):
        return _louisiana_direct_dependency_subject_is_executable(governing_scope)

    governing_scope = _louisiana_strip_inline_context_preamble(governing_scope)
    operative_predicates = tuple(
        _LOUISIANA_POSITIVE_OPERATIVE_PREDICATE.finditer(governing_scope)
    )
    if not operative_predicates:
        return False
    operative = operative_predicates[-1]
    if _louisiana_text_has_nonoperative_framing(operative.group(0)):
        return False
    # The operative predicate must be the expression immediately attached to
    # the dependency linker.  This prevents an earlier computation from lending
    # operativeness to later metadata, commentary, examples, or uncertainty.
    if not _louisiana_operative_suffix_is_bounded(governing_scope[operative.end() :]):
        return False
    local_before_operative = _louisiana_local_finite_clause(
        governing_scope[: operative.start()]
    )
    if re.search(
        r"\b(?:appendix|audit|background|chart|commentary|comparison|description|"
        r"documentation|example|exhibit|explanation|illustration|label|memo|"
        r"memorandum|metadata|nonbinding|note|record|report|sample|scenario|summary)\b|"
        r"\b(?:alleg\w*|assum\w*|believ\w*|claim\w*|describ\w*|discuss\w*|"
        r"indicat\w*|illustrat\w*|list\w*|mention\w*|purport\w*|quot\w*|"
        r"report\w*|say\w*|show\w*|suggest\w*|suppos\w*|testif\w*|"
        r"summariz\w*|treat\w*)\b",
        local_before_operative,
        flags=re.IGNORECASE,
    ):
        return False
    local_clause = local_before_operative + operative.group(0)
    if re.match(r"\s*no\b", local_clause, flags=re.IGNORECASE):
        return False
    return not _louisiana_predicate_is_in_negative_complement(
        governing_scope, predicate_start=operative.start()
    )


def _louisiana_reference_has_nonoperative_postfix(after_reference: str) -> bool:
    """Reject a qualifier after the citation that retracts its operative use."""

    suffix = after_reference.strip()
    legal_authority = (
        r"(?:act|code|court|department|language|law|legislature|ordinance|"
        r"provisions?|regulation|rule|section|statute|table|terms?|text)"
    )
    if re.match(
        r"^,?\s*(?:in|for|during|using|under)\b",
        suffix,
        flags=re.IGNORECASE,
    ):
        return _louisiana_text_has_nonoperative_framing(suffix)
    if re.match(
        r"^,?\s*(?:but\s+)?(?:this|that|the\s+(?:statement|proposition))\b"
        r"[^.;]{0,80}\b(?:contrived|draft|dubious|fictional|hypothetical|"
        r"illustrative|questionable|unproven)\b",
        suffix,
        flags=re.IGNORECASE,
    ):
        return True
    independent = re.match(
        r"^,?\s*(?:although|and|but|though|while|whereas|yet)\s+"
        r"(?:the\s+|a\s+|an\s+)?(?:[A-Za-z][A-Za-z-]*\s+){0,5}"
        r"(?:is|are|was|were|has|have|had|can|could|may|might|must|shall|"
        r"should|will|would|alleges?|asserts?|believes?|claims?|declares?|"
        r"discusses?|reports?|says?|states?|speculates?)\b",
        suffix,
        flags=re.IGNORECASE,
    )
    if independent:
        return False
    attribution = re.match(
        r"^,?\s*(?:at\s+least\s+)?(?:purportedly\s+)?according\s+to\s+"
        r"(?P<object>[^.;]{1,100})",
        suffix,
        flags=re.IGNORECASE,
    )
    if attribution:
        return not _louisiana_speaker_is_legal_authority(attribution.group("object"))
    conditional = re.match(
        r"^,?\s*if\s+(?P<condition>[^.;]{1,100})",
        suffix,
        flags=re.IGNORECASE,
    )
    if conditional:
        condition_core = re.split(
            r"\b(?:under|according\s+to|pursuant\s+to)\b",
            conditional.group("condition"),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        return not re.search(
            rf"\b(?:statutory|legal|applicable|{legal_authority})\b",
            condition_core,
            flags=re.IGNORECASE,
        )
    speech = re.match(
        r"^,?\s*(?:(?:as|or\s+so)\s+)?(?:the\s+)?"
        r"(?P<speaker>[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,9}?)\s+"
        r"(?:(?:[A-Za-z][A-Za-z-]*ly)\s+)*"
        r"(?:alleges?|asserts?|believes?|claims?|concludes?|contends?|declares?|"
        r"maintains?|opines?|reports?|says?|states?|testif(?:y|ies|ied))\b",
        suffix,
        flags=re.IGNORECASE,
    )
    if speech:
        return not _louisiana_speaker_is_legal_authority(speech.group("speaker"))
    return bool(
        re.match(
            r"^,?\s*(?:"
            r"(?:but\s+)?(?:merely|only)\b[^.;]{0,100}\b(?:contrived|example|"
            r"fictional|hypothetical|illustrati\w*|sample|scenario)\b|"
            r"though\s+(?:allegedly|apparently|hypothetically|purportedly|"
            r"reportedly|supposedly|theoretically)\b|"
            r"(?:but\s+)?(?:this|that|which|the\s+(?:statement|proposition))\b"
            r"[^.;]{0,80}\b(?:contrived|draft|dubious|fictional|hypothetical|"
            r"illustrative|questionable|unproven)\b"
            r")",
            suffix,
            flags=re.IGNORECASE,
        )
    )


def _louisiana_speaker_is_legal_authority(speaker: str) -> bool:
    """Resolve the grammatical authority head and any required complement."""

    intrinsic_heads = {
        "act",
        "code",
        "constitution",
        "court",
        "department",
        "law",
        "legislature",
        "ordinance",
        "regulation",
        "rule",
        "section",
        "statute",
        "table",
    }
    derivative_heads = {
        "article",
        "decision",
        "language",
        "opinion",
        "order",
        "provision",
        "provisions",
        "term",
        "terms",
        "text",
    }
    relation = re.search(r"\b(?:for|of)\b", speaker, flags=re.IGNORECASE)
    head_scope = speaker[: relation.start()] if relation else speaker
    head_tokens = re.findall(r"[A-Za-z]+", head_scope.lower())
    if not head_tokens:
        return False
    head = (
        "article"
        if re.fullmatch(
            r"articles?\s+(?:\d+|[ivxlcdm]+)(?:-[a-z0-9]+|\([a-z0-9]+\))?"
            r"(?:\s+and\s+(?:\d+|[ivxlcdm]+))?"
            r"(?:\s*,\s*(?:(?:§+|sections?)\s*\d+(?:[.-]\d+)*"
            r"(?:\([a-z0-9]+\))*(?:\s*(?:-|–)\s*\([a-z0-9]+\)|"
            r"\s+and\s+(?:\d+(?:\([a-z0-9]+\))*|\([a-z0-9]+\)))?"
            r"(?:\s*,\s*paragraph\s+(?:[a-z0-9]+|\([a-z0-9]+\)))?|"
            r"paragraphs?\s+(?:[a-z0-9]+|\([a-z0-9]+\))"
            r"(?:\s+and\s+(?:[a-z0-9]+|\([a-z0-9]+\)))?))?",
            head_scope.strip(),
            flags=re.IGNORECASE,
        )
        else head_tokens[-1]
    )
    governmental_department_modifiers = {
        "federal",
        "health",
        "justice",
        "labor",
        "louisiana",
        "revenue",
        "state",
        "tax",
        "taxation",
        "treasury",
    }
    if (
        head == "department"
        and len(head_tokens) > 1
        and not any(
            token in governmental_department_modifiers for token in head_tokens[:-1]
        )
    ):
        return False
    if head in intrinsic_heads:
        return True
    if head == "order" and re.search(
        r"\b(?:administrative|court|executive|judicial|restraining)\s+order\s*$|"
        r"\bcease-and-desist\s+order\s*$",
        head_scope.strip(),
        flags=re.IGNORECASE,
    ):
        return True
    if head not in derivative_heads:
        return False
    if re.search(r"\b(?:legal|statutory)\b", head_scope, flags=re.IGNORECASE):
        return True
    if relation is None:
        return head in {"provision", "provisions", "term", "terms"}
    complement_source = speaker[relation.end() :]
    documentary_person = re.search(
        r"\b(?:analyst|author|commentator|consultant|expert|researcher|witness)\b",
        complement_source,
        flags=re.IGNORECASE,
    )
    preceding_authority = (
        re.search(
            r"\b(?:act|code|constitution|court|department|law|legislature|"
            r"ordinance|order|regulation|rule|section|statute|table)\b",
            complement_source[: documentary_person.start()],
            flags=re.IGNORECASE,
        )
        if documentary_person is not None
        else None
    )
    if documentary_person is not None and preceding_authority is None:
        return False
    participle = re.search(
        r"\b[A-Za-z][A-Za-z-]*ing\b", complement_source, flags=re.IGNORECASE
    )
    authority_modifiers = {
        "administrative",
        "agency",
        "federal",
        "governing",
        "implementing",
        "local",
        "louisiana",
        "outside",
        "reviewing",
        "state",
        "tax",
    }
    prefix_tokens = (
        tuple(
            token
            for token in re.findall(
                r"[A-Za-z]+", complement_source[: participle.start()].lower()
            )
            if token not in {"a", "an", "the"}
        )
        if participle is not None
        else ()
    )
    if participle is not None and any(
        token not in authority_modifiers for token in prefix_tokens
    ):
        complement_source = complement_source[: participle.start()]
    complement_scope = re.split(
        r"\b(?:about|concerning|in|on|regarding|under|with)\b",
        complement_source,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    complement_tokens = re.findall(r"[A-Za-z]+", complement_scope.lower())
    if (
        complement_tokens
        and complement_tokens[-1] == "department"
        and len(complement_tokens) > 1
        and not any(
            token in governmental_department_modifiers
            for token in complement_tokens[:-1]
        )
    ):
        return False
    return bool(complement_tokens and complement_tokens[-1] in intrinsic_heads)


def _louisiana_scope_has_nonoperative_framing(scope: str) -> bool:
    """Reject hypothetical or documentary embeddings before syntax branches."""

    if re.match(
        r"\s*(?:\(|\[)?\s*in\s+(?:(?:relevant|pertinent)\s+)?"
        r"(?:parts?|portions?)\s*"
        r"(?:\)|\])?\s*,?\s*(?:as\s+follows\b)?",
        scope,
        flags=re.IGNORECASE,
    ):
        return True
    if re.match(
        r"\s*in\s+(?:(?:the\s+)?(?:author|commentator|expert|witness)"
        r"(?:'s|’s)?\s+(?:own\s+)?words|the\s+words\s+of\s+(?:the\s+)?"
        r"(?:author|commentator|expert|witness))\s*,|"
        r"\s*as\s+(?:the\s+)?(?:affidavit|testimony)\s+puts?\s+it\s*,",
        scope,
        flags=re.IGNORECASE,
    ):
        return True
    if re.match(
        r"\s*as\s+set\s+forth\s+in\s+(?:the\s+)?(?:affidavit|testimony)\s*,",
        scope,
        flags=re.IGNORECASE,
    ):
        return True

    prefix_attribution = re.match(
        r"\s*according\s+to\s+(?:the\s+)?(?P<speaker>[^.;]{1,80})\s*,",
        scope,
        flags=re.IGNORECASE,
    )
    if prefix_attribution:
        if not _louisiana_speaker_is_legal_authority(
            prefix_attribution.group("speaker")
        ):
            return True

    # A fronted subordinate supplies context for, but does not grammatically
    # govern, the comma-delimited main clause that carries the dependency.
    if (
        re.match(
            r"\s*(?:although|because|since|when|while|whereas)\b",
            scope,
            flags=re.IGNORECASE,
        )
        and (fronted_end := scope.find(",")) >= 0
        and re.match(
            r"\s*(?:the|a|an|this|final|applicable)\s+[A-Za-z]",
            scope[fronted_end + 1 :],
            flags=re.IGNORECASE,
        )
    ):
        scope = scope[fronted_end + 1 :]

    independent_resets = tuple(
        re.finditer(
            r"(?:\b(?:although|and|because|but|or|while|whereas)\s+|"
            r"\byet\s+(?=(?:the|a|an|this|that|final|applicable)\b))"
            r"(?=(?:(?:the|a|an|this|that|final|applicable)\s+)?"
            r"(?:(?:[A-Za-z][A-Za-z-]*\s+){1,7}"
            r"(?:is|are|was|were|shall|may|must|will|would|can|could|does|do|did|"
            r"depends?|requires?|relies?|determines?|calculates?|computes?|"
            r"establishes?|applies?|imposes?|allows?|arises?)\b|"
            r"(?:[A-Za-z][A-Za-z-]*\s*){1,5}$))",
            scope,
            flags=re.IGNORECASE,
        )
    )
    if independent_resets:
        reset = independent_resets[-1]
        embedding_prefix = scope[: reset.start()]
        if not re.search(
            r"\b(?:alleges?|asserts?|avers?|claims?|contends?|declares?|imagines?|"
            r"maintains?|reports?|states?|suppose[sd]?|testif(?:y|ies|ied))\s+"
            r"(?:that\s+)?[^.;]{0,120}$",
            embedding_prefix,
            flags=re.IGNORECASE,
        ):
            scope = scope[reset.end() :]
    return _louisiana_text_has_nonoperative_framing(scope) or bool(
        re.search(
            r"\b(?:assumed|candidate|conceivable|conceivably|conceptual|draft|"
            r"example|fictional|fictionally|hypothetical(?:ly)?|illustrative|"
            r"imagined|mock|nonbinding|notional|perhaps|possibly|proposed|prototype|"
            r"purported|putative|sample|scenario|specimen|suppose[sd]?|theoretical)\b|"
            r"\b(?:for\s+purposes\s+of\s+argument|in\s+theory)\b|"
            r"\b(?:chance|could\s+be|might\s+be|possibility)\b[^.;]{0,40}\bthat\b|"
            r"^\s*if\s+(?:the\s+)?(?:account|description|narrative|report|source)\s+"
            r"is\s+(?:accurate|correct|true)\b|"
            r"\b(?:agency|analysis|analyst|article|commentator|document|narrative|"
            r"requirement|source|table|witness)\b[^.;]{0,100}"
            r"\b(?:alleges?|asserts?|avers?|believes?|cannot\s+confirm|concludes?|"
            r"contends?|declares?|estimates?|has\s+yet\s+to\s+confirm|indicates?|"
            r"infers?|maintains?|observes?|opines?|predicts?|recites?|surmises?|"
            r"testif(?:y|ies|ied))\b",
            scope,
            flags=re.IGNORECASE,
        )
    )


def _louisiana_direct_dependency_subject_is_executable(scope: str) -> bool:
    """Bind ``depends on`` and ``requires`` to the direct grammatical subject."""

    scope = re.sub(
        r"\b(directly|entirely|necessarily|ordinarily|solely|ultimately)\s+and\s+"
        r"(?=(?:directly|entirely|necessarily|ordinarily|solely|ultimately)\b)",
        r"\1 ",
        scope,
        flags=re.IGNORECASE,
    )
    local_clause = _louisiana_local_finite_clause(scope).strip(" ,")
    if re.search(
        r"\b(?:after|before)\s+(?:the|a|an)\s+"
        r"[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,4}\s*$",
        scope,
        flags=re.IGNORECASE,
    ):
        local_clause = scope.strip(" ,")
    if re.search(
        r"\b(?:line|paragraph|part|schedule)\s+[A-Za-z0-9]+\s+of\s+(?:the\s+)?"
        r"(?:calculation|report|return|tax\s+return|worksheet)\b",
        local_clause,
        flags=re.IGNORECASE,
    ):
        return False
    if _LOUISIANA_DOCUMENTARY_HEAD.search(local_clause) or re.search(
        r"^\s*(?:no|neither|nothing)\b|"
        r"\b(?:can|could|did|does|do|is|are|was|were|will|would)\s+"
        r"(?:never|not)\b|n['’]t\b|"
        r"\b(?:uncertain|unknown|whether)\b|"
        r"\b(?:assertion|availability|citation|claim|date|discussion|email|filing|"
        r"footnote|guide|historical|hypothetical|identifier|outline|proposal|"
        r"reference|rumor|status|title|website)\b",
        local_clause,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\b(?:ceases?|fails?|refuses?)\s+to\s*$",
        local_clause,
        flags=re.IGNORECASE,
    ):
        return False
    subject = local_clause
    while True:
        stripped = re.sub(
            r"\s*,?\s*\b(?:directly|entirely|necessarily|ordinarily|solely|"
            r"ultimately)\b(?:\s+and\s+\b(?:directly|entirely|necessarily|"
            r"ordinarily|solely|ultimately)\b)*\s*$",
            "",
            subject,
            flags=re.IGNORECASE,
        ).strip(" ,")
        if stripped == subject:
            break
        subject = stripped
    if re.match(
        r"which\s+(?:amount|base|calculation|computation|rate|tax)\s+applies$",
        subject,
        re.IGNORECASE,
    ):
        return True
    if re.match(
        r"(?:application|determination)\s+of\s+(?:the\s+)?(?:amount|base|rate|tax)$",
        subject,
        re.IGNORECASE,
    ):
        return True
    if re.match(
        r"(?:the\s+)?(?:lesser|greater)\s+of\s+(?:the\s+)?(?:two|multiple)?\s*rates?$",
        subject,
        re.IGNORECASE,
    ):
        return True
    executable_heads = {
        "amount",
        "base",
        "calculation",
        "computation",
        "credit",
        "deduction",
        "formula",
        "income",
        "liability",
        "percentage",
        "rate",
        "result",
        "rule",
        "schedule",
        "statute",
        "table",
        "tax",
        "taxes",
        "threshold",
    }
    relation = re.search(
        r",|(?<!-)\b(?:about|accompanying|as|beneath|beside|by|concerning|"
        r"after|against|before|containing|covering|describing|displaying|during|"
        r"comprising|excluding|for|in|including|incorporating|listing|of|on|"
        r"outside|under|using|"
        r"presenting|reciting|"
        r"regarding|setting\s+out|matching|reflecting|to|with|without|that|which|who|"
        r"(?:arising|derived|emerging|flowing|resulting|stemming)\s+from|"
        r"(?:based|dependent|predicated|reliant)\s+(?:on|upon)|"
        r"(?:linked|related|tied)\s+to)\b(?!-)",
        subject,
        flags=re.IGNORECASE,
    )
    relation_start = relation.start() if relation else None
    if relation is None:
        for candidate in re.finditer(
            r"\b(?P<head>[A-Za-z][A-Za-z-]*)\s+"
            r"(?P<relation>(?:[A-Za-z]+ed|drawn|taken)\s+from)\b",
            subject,
            flags=re.IGNORECASE,
        ):
            if candidate.group("head").lower() in {"a", "an", "the"}:
                continue
            relation_start = candidate.start("relation")
            break
    if relation_start is None:
        participial_clause = re.search(
            r"\b[A-Za-z][A-Za-z-]*ing\s+(?=(?:a|an|the)\s+"
            r"(?:[A-Za-z][A-Za-z-]*\s+){0,3}"
            r"(?:amount|base|calculation|computation|credit|deduction|formula|"
            r"income|liability|rate|rule|tax)\b)",
            subject,
            flags=re.IGNORECASE,
        )
        if participial_clause is not None:
            relation_start = participial_clause.start()
    grammatical_head_scope = (
        subject[:relation_start] if relation_start is not None else subject
    )
    tokens = re.findall(r"[A-Za-z]+", grammatical_head_scope.lower())
    if not tokens:
        return False
    if tokens[-1] in executable_heads:
        return True
    executable_positions = [
        index for index, token in enumerate(tokens) if token in executable_heads
    ]
    if not executable_positions:
        return False
    modifiers = {
        "allowed",
        "alone",
        "applicable",
        "adopted",
        "attached",
        "allowable",
        "calculated",
        "computed",
        "determined",
        "due",
        "final",
        "imposed",
        "itself",
        "named",
        "owed",
        "otherwise",
        "outstanding",
        "owing",
        "payable",
        "prescribed",
        "remaining",
        "required",
        "selected",
        "shown",
        "specified",
    }
    return all(
        token in modifiers or token.endswith(("able", "ible", "ly"))
        for token in tokens[executable_positions[-1] + 1 :]
    )


def _louisiana_operative_suffix_is_bounded(suffix: str) -> bool:
    """Allow statutory modifiers without admitting documentary framing."""

    suffix = re.sub(
        r"\b(and|or|but)\s*,\s*where\s+applicable\s*,\s*",
        r"\1 ",
        suffix,
        flags=re.IGNORECASE,
    )
    positive_participle = (
        r"(?:allowed|assessed|available|calculated|computed|derived|determined|"
        r"established|fixed|imposed|levied|prescribed|rounded|set|specified)"
    )
    bounded_modifier = (
        r"(?:annually|directly|entirely|exclusively|finally|separately|solely|"
        r"statutorily|provisionally|annually|monthly|quarterly|"
        r"for\s+(?:each|every)\s+(?:taxpayer|return|filing\s+unit)|each\s+year|"
        r"for\s+(?:the\s+)?taxable\s+year|per\s+taxpayer|on\s+(?:the\s+)?return|"
        r"on\s+an\s+annual\s+basis|at\s+the\s+taxpayer\s+level|"
        r"for\s+the\s+applicable\s+taxable\s+year|to\s+the\s+nearest\s+dollar|"
        r"for\s+each\s+(?:taxable\s+year|filing\s+period)|"
        r"to\s+the\s+nearest\s+whole\s+dollar|"
        r"as\s+of\s+the\s+close\s+of\s+the\s+taxable\s+year|"
        r"in\s+all\s+cases|where\s+applicable|"
        r"for\s+purposes\s+of\s+(?:this|the)\s+(?:section|subsection)|"
        r"as\s+applicable|with\s+no\s+(?:adjustment|cap))"
    )
    return bool(
        re.fullmatch(
            rf"[\s,]*(?:(?:(?:and(?:\s+then)?|or|but|then)\s+)?"
            rf"(?:(?:provisionally|finally|directly)\s+)*"
            rf"{positive_participle}[\s,]*)*(?:{bounded_modifier}[\s,]*)*",
            suffix,
            flags=re.IGNORECASE,
        )
    )


def _louisiana_local_finite_clause(scope: str) -> str:
    """Return text after the last structural clause connector."""

    connectors = tuple(
        re.finditer(
            r"\b(?:and|although|because|before|but|if|once|or|since|then|"
            r"unless|when|whenever|where|whereas|while|yet)\b",
            scope,
            flags=re.IGNORECASE,
        )
    )
    boundary = connectors[-1].end() if connectors else 0
    # A comma after a fronted subordinate clause starts a fresh finite main
    # clause; its negation must not contaminate the later operative predicate.
    comma_boundaries = tuple(
        match.end()
        for match in re.finditer(r",", scope)
        if re.match(
            r"\s*(?:(?:the|a|an|this|that|final|applicable)\s+)?"
            r"(?:[A-Za-z][A-Za-z-]*\s+){1,6}"
            r"(?:is|are|was|were|shall|may|must|will|would|can|could|does|do|did)\b",
            scope[match.end() :],
            flags=re.IGNORECASE,
        )
    )
    if (
        re.match(
            r"\s*(?:although|because|since|when|while)\b",
            scope,
            flags=re.IGNORECASE,
        )
        and "," in scope
    ):
        comma_boundaries += (scope.rfind(",") + 1,)
    if comma_boundaries:
        boundary = max(boundary, comma_boundaries[-1])
    return scope[boundary:]


def _louisiana_predicate_is_in_negative_complement(
    scope: str,
    *,
    predicate_start: int,
) -> bool:
    """Reject a positive-looking predicate inside a negated ``that`` complement."""

    prefix = scope[:predicate_start]
    complement_markers = tuple(
        re.finditer(r"\b(?:if|that|whether)\b", prefix, re.IGNORECASE)
    )
    if not complement_markers:
        return bool(
            re.search(
                r"\b(?:uncertain|unknown|unclear|undetermined)\b",
                _louisiana_local_finite_clause(prefix),
                flags=re.IGNORECASE,
            )
        )
    complement = complement_markers[-1]
    marker = complement.group(0).lower()
    introducer = prefix[: complement.start()]
    if marker == "whether":
        return True
    if marker == "if":
        return bool(
            re.search(
                r"\b(?:possible|uncertain|unclear|unknown|undetermined|questions?|"
                r"asks?|doubts?)\b",
                _louisiana_local_finite_clause(introducer),
                flags=re.IGNORECASE,
            )
        )
    preceding_word = re.search(r"([A-Za-z]+)\s*$", introducer)
    if preceding_word and preceding_word.group(1).lower() in {
        "except",
        "provided",
        "so",
    }:
        return False
    local_introducer = _louisiana_local_finite_clause(introducer)
    return bool(
        re.search(
            r"\b(?:no|not|nothing|never|cannot|denies?|rejects?|disclaims?|fails?|"
            r"refuses?)\b|"
            r"\b(?:assumes?|claims?|reports?|says?|purports?|suggests?)\b|"
            r"\b(?:doubtful|evidence|false|untrue|impossible|possible|uncertain|"
            r"unclear|unknown)\b|"
            r"\bby\s+no\s+means\b|n['’]t\b",
            local_introducer,
            flags=re.IGNORECASE,
        )
    )


def _source_link_scope_has_unreset_negation(governing_scope: str) -> bool:
    """Bind negation to a linker unless a later finite predicate resets it."""

    negative_markers = tuple(
        re.finditer(
            r"\b(?:not|never|neither|nor|cannot|nowhere|prohibit\w*|barred|"
            r"forbidden|precluded|prevented|excluded|denied|except)\b|"
            r"\b(?:denies?|rejects?|disclaims?)\b|"
            r"\b(?:false|untrue|impossible)\s+that\b|"
            r"(?:^|[,;:]\s*|\b(?:and|but|or|that|while|whereas)\s+)"
            r"no\s+(?:amount|tax|liability|computation|calculation|"
            r"determination|rate|base|statute|rule|provision|requirement)\b"
            r"(?:\s+[A-Za-z-]+){0,3}\s+(?:is|are|was|were|shall|may|must|will|"
            r"would|can|could|does|do|did|states?|provides?|requires?|"
            r"determines?|calculates?|computes?|applies?|arises?)\b|"
            r"\b(?:ca|do|does|did|is|are|was|were|"
            r"shall|will|would|could|should|must|need)n['’]t\b|"
            r"\b(?:fails?|refuses?|unable)\s+to\b|"
            r"\bwithout\s+(?:being\s+)?(?:determin\w*|comput\w*|calculat\w*|"
            r"appl\w*|impos\w*|aris\w*)\b|"
            r"\b(?:in\s+no\s+(?:case|event)|under\s+no\s+circumstances|"
            r"at\s+no\s+time|by\s+no\s+means|in\s+no\s+way|"
            r"on\s+no\s+account|under\s+no\s+conditions?|"
            r"in\s+no\s+(?:manner|respect))\b|"
            r"\b(?:rather|other)\s+than\b",
            governing_scope,
            flags=re.IGNORECASE,
        )
    )
    if not negative_markers:
        return False

    last_negative = negative_markers[-1]
    after_last_negative = governing_scope[last_negative.end() :]

    # A finite predicate inside an expressly negated complement does not escape
    # merely because the complement coordinates another subject and verb.
    if re.search(
        r"(?:\bnot\s+the\s+case|\b(?:does?|did|is|are|was|were|has|have|had)"
        r"\s+not\s+(?:provide|state|say|specify|indicate|require|establish|"
        r"declare|permit|allow|assert|show)|\bnever\s+(?:provides?|states?|"
        r"says?|specifies?|indicates?|requires?|establishes?|declares?|permits?|"
        r"allows?|asserts?|shows?)|\b(?:denies?|rejects?|disclaims?)|"
        r"\b(?:false|untrue|impossible))\s+that\b",
        governing_scope,
        flags=re.IGNORECASE,
    ) and not re.search(
        r"\b(?:but|yet|however|nevertheless|nonetheless)\b",
        after_last_negative,
        flags=re.IGNORECASE,
    ):
        return True

    reset = re.search(
        r"(?:\b(?:provided\s+that|even\s+though|so\s+long\s+as|and|but|yet|"
        r"while|whereas|or|then|because|since|if|when|whenever|although|unless|"
        r"once|after|before|where|as)\b|,)\s+"
        r"(?:instead\s+)?(?:"
        # A repeated auxiliary/copula makes a coordinated predicate finite.
        r"(?:is|are|was|were|shall|may|must|will|would|can|could|does|do|did|"
        r"becomes?|remains?)\b|"
        # Otherwise require an explicit subject before the finite predicate;
        # this keeps shared-negation forms such as "not apply or arise" bound.
        r"(?:(?:the|a|an|this|that|final|preliminary|separate)\s+)?"
        r"(?:[A-Za-z][A-Za-z-]*\s+){1,7}?"
        r"(?:is|are|was|were|shall|may|must|will|would|can|could|does|do|did|"
        r"becomes?|remains?|depends?|requires?|determines?|calculates?|computes?|"
        r"establishes?|applies?|imposes?|allows?|arises?|supplies?|provides?)\b"
        r")",
        after_last_negative,
        flags=re.IGNORECASE,
    )
    if reset is None:
        return True
    reset_scope = after_last_negative[reset.start() :]
    # A new negative construction after the reset governs the dependency.
    return bool(
        re.search(
            r"\b(?:not|never|neither|nor|cannot|nowhere|prohibit\w*|barred|"
            r"forbidden|precluded|prevented|excluded|denied|except)\b|"
            r"\b(?:denies?|rejects?|disclaims?)\b|"
            r"\b(?:false|untrue|impossible)\s+that\b|"
            r"\b(?:fails?|refuses?|unable)\s+to\b|"
            r"\bwithout\s+(?:being\s+)?(?:determin\w*|comput\w*|calculat\w*|"
            r"appl\w*|impos\w*|aris\w*)\b",
            reset_scope,
            flags=re.IGNORECASE,
        )
    )


def _louisiana_disclaimer_governs_reference(local_before: str) -> bool:
    """Bind a contextual disclaimer to its citation, not an earlier preamble."""

    disclaimers = tuple(
        re.finditer(
            r"\b(?:without\s+prejudice\s+to|remain\w*\s+unaffected|"
            r"unaffected\s+by|does\s+not\s+affect|notwithstanding)\b",
            local_before,
            flags=re.IGNORECASE,
        )
    )
    if not disclaimers:
        return False
    disclaimer = disclaimers[-1]
    after_disclaimer = local_before[disclaimer.end() :]
    separator = re.search(r"(?:,|:|[-\u2013\u2014])", after_disclaimer)
    if separator is None:
        return True
    disclaimer_object = after_disclaimer[: separator.start()]
    reset_scope = after_disclaimer[separator.end() :]
    disclaimer_is_context_preamble = _louisiana_disclaimer_object_is_context_preamble(
        disclaimer_object
    )
    relative_reset = re.match(
        r"\s*(?:(?:by|to|through|under|pursuant\s+to|in|according\s+to)\s+which|"
        r"which|who|whom|whose|where|whereby|wherein|that)\b",
        reset_scope,
        flags=re.IGNORECASE,
    )
    if relative_reset:
        independent_boundaries = []
        for boundary in re.finditer(
            r"\b(?:although|and|because|but|or|while|whereas|yet)\b|,",
            reset_scope,
            re.IGNORECASE,
        ):
            candidate = reset_scope[boundary.end() :]
            if not _louisiana_scope_starts_independent_finite_clause(candidate):
                continue
            if boundary.group(0) == "," and not _louisiana_relative_prefix_is_closed(
                reset_scope[: boundary.start()]
            ):
                continue
            independent_boundaries.append(boundary)
        if not independent_boundaries:
            return True
        reset_candidates = [
            reset_scope[boundary.end() :] for boundary in independent_boundaries
        ]
    else:
        reset_candidates = [reset_scope]
        reset_candidates.extend(
            after_disclaimer[boundary.end() :]
            for boundary in re.finditer(r"(?:,|:|[-\u2013\u2014])", after_disclaimer)
            if boundary.end() > separator.end()
        )
    for candidate in reversed(reset_candidates):
        if re.match(
            r"\s*(?:(?:although|and|because|but|or|since|when|while|whereas|yet)\s+)?"
            r"(?:under\s+which|in\s+which|which|who|whom|whose|where|that)\b",
            candidate,
            flags=re.IGNORECASE,
        ):
            continue
        operative = _LOUISIANA_POSITIVE_OPERATIVE_PREDICATE.search(candidate)
        if operative is None:
            continue
        has_independent_connector = bool(
            re.match(
                r"\s*(?:although|and|because|but|or|since|when|while|whereas|yet)\b",
                candidate,
                flags=re.IGNORECASE,
            )
        )
        if not disclaimer_is_context_preamble and not has_independent_connector:
            continue
        introduction = candidate[: operative.start()]
        if re.search(
            r"\b(?:alleges?|asserts?|claims?|declares?|indicates?|notes?|observes?|"
            r"opines?|proclaims?|recites?|reports?|says?|states?|stipulates?|"
            r"testifies?)"
            r"\b",
            introduction,
            flags=re.IGNORECASE,
        ):
            continue
        if re.match(
            r"\s*(?:(?:although|and|because|but|or|since|when|while|whereas|yet)\s+)?"
            r"(?:(?:the|a|an|this|that|final|applicable)\s+)?"
            r"(?:[A-Za-z][A-Za-z-]*\s+){0,7}"
            r"(?:is|are|was|were|shall|may|must|will|would|can|could|does|do|did|"
            r"becomes?|remains?|allows?|assesses?|calculates?|computes?|derives?|"
            r"determines?|establishes?|fixes?|imposes?|levies?|prescribes?|sets?|"
            r"specifies?)\b",
            candidate,
            flags=re.IGNORECASE,
        ):
            return False
    if not disclaimer_is_context_preamble:
        return True
    before_disclaimer = local_before[: disclaimer.start()]
    reset_is_relative = bool(
        re.match(
            r"\s*(?:(?:although|and|because|but|or|since|when|while|whereas|yet)\s+)?"
            r"(?:(?:by|to|through|under|pursuant\s+to|in|according\s+to)\s+which|"
            r"which|who|whom|whose|where|whereby|wherein|that)\b",
            reset_scope,
            flags=re.IGNORECASE,
        )
    )
    if reset_is_relative:
        return True
    return not bool(
        (
            not reset_is_relative
            and _LOUISIANA_POSITIVE_OPERATIVE_PREDICATE.search(reset_scope)
        )
        or _LOUISIANA_POSITIVE_OPERATIVE_PREDICATE.search(before_disclaimer)
    )


def _louisiana_scope_starts_independent_finite_clause(scope: str) -> bool:
    """Require an explicit subject before a disclaimer reset predicate."""

    return bool(
        re.match(
            r"\s*(?:(?:the|a|an|this|that|final|applicable)\s+)?"
            r"(?:[A-Za-z][A-Za-z-]*\s+){1,7}"
            r"(?:is|are|was|were|shall|may|must|will|would|can|could|does|do|did|"
            r"becomes?|remains?|allows?|assesses?|calculates?|computes?|derives?|"
            r"determines?|establishes?|fixes?|imposes?|levies?|prescribes?|sets?|"
            r"specifies?)\b",
            scope,
            flags=re.IGNORECASE,
        )
    )


def _louisiana_relative_prefix_is_closed(prefix: str) -> bool:
    """Return whether a relative clause has a finite predicate before a comma."""

    relative = re.match(
        r"\s*(?:(?:by|to|through|under|pursuant\s+to|in|according\s+to)\s+which|"
        r"which|who|whom|whose|where|whereby|wherein|that)\b",
        prefix,
        flags=re.IGNORECASE,
    )
    if relative is None:
        return False
    return bool(
        re.search(
            r"\b(?:is|are|was|were|has|have|had|does|do|did|applies?|continues?|"
            r"exists?|remains?|governs?|controls?|requires?|provides?|states?)\b",
            prefix[relative.end() :],
            flags=re.IGNORECASE,
        )
    )


def _louisiana_disclaimer_object_is_context_preamble(disclaimer_object: str) -> bool:
    """Recognize a self-contained external-law preamble object."""

    return bool(
        re.search(
            r"(?:\b(?:applicable|federal|louisiana|other|state)\s+"
            r"(?:law|rights?|return\s+rule|statute)|"
            r"\bthe\s+law\s+of\s+this\s+state|"
            r"\bany\s+law\s+(?:to\s+the\s+)?contrary|"
            r"\bany\s+other\s+provision\s+of\s+law|"
            r"\b\d+\s+U\.?\s*S\.?\s*C\.?\s+\d+[A-Za-z0-9.\-]*|"
            r"\bR\.?\s*S\.?\s*\d+[A-Za-z]?\s*:\s*\d+[A-Za-z0-9.\-]*)\s*$",
            disclaimer_object,
            flags=re.IGNORECASE,
        )
    )


def _louisiana_strip_inline_context_preamble(scope: str) -> str:
    """Remove a bounded inline external-law preamble before attachment checks."""

    return re.sub(
        r"(?:,\s*|[-\u2013\u2014]\s*)"
        r"(?:notwithstanding|without\s+prejudice\s+to)\s+"
        r"(?:(?:applicable|federal|other|state)\s+"
        r"(?:law|rights?|return\s+rule|statute)|"
        r"any\s+other\s+provision\s+of\s+law|"
        r"\d+\s+U\.?\s*S\.?\s*C\.?\s+\d+[A-Za-z0-9.\-]*|"
        r"R\.?\s*S\.?\s*\d+[A-Za-z]?\s*:\s*\d+[A-Za-z0-9.\-]*)\s*"
        r"(?:,|[-\u2013\u2014])\s*",
        " ",
        scope,
        flags=re.IGNORECASE,
    )


def _usc_dependency_fragments(match: re.Match[str]) -> tuple[str, ...]:
    return tuple(
        fragment.lower()
        for fragment in re.findall(
            r"\(\s*([A-Za-z0-9]+(?:[-\u2013][A-Za-z0-9]+)*)\s*\)",
            match.group("tail") or "",
        )
    )


def _louisiana_rs_tail_identity(match: re.Match[str]) -> str:
    """Preserve connector and legal-label semantics in an R.S. citation tail."""

    tail = (match.group("tail") or "").lower()
    tail = re.sub(r"[\u2010-\u2015\u2212\ufe58\ufe63\uff0d]", "-", tail)
    tail = re.sub(r"\s+", " ", tail)
    tail = re.sub(r",\s*(?:and\s+)?(?=\()", " and ", tail)
    tail = re.sub(r"&", " and ", tail)
    tail = re.sub(
        r"\b(?:clauses?|divisions?|items?|lines?|paragraphs?|parts?|"
        r"schedules?|subclauses?|subdivisions?|subitems?|subparagraphs?|"
        r"subparts?)\b",
        lambda item: item.group(0).removesuffix("s"),
        tail,
    )
    tail = re.sub(r"\bsubsections?\s+(?=\()", "", tail)
    tail = re.sub(r"\bread\s+together\s+with\b", "read with", tail)
    tail = re.sub(r"\b(?:through|to)\b|-(?=\s*\()", " range ", tail)
    tail = re.sub(r"\s*([(),])\s*", r"\1", tail)
    tail = re.sub(r"\s+", " ", tail)
    return tail.strip()


def _qualified_usc_dependencies(text: str) -> tuple[re.Match[str], ...]:
    return tuple(
        sorted(
            (
                *_USC_DEFERRAL_DEPENDENCY.finditer(text),
                *_REVERSED_USC_DEFERRAL_DEPENDENCY.finditer(text),
                *_TITLE_FIRST_USC_DEFERRAL_DEPENDENCY.finditer(text),
            ),
            key=lambda match: (match.start(), match.end()),
        )
    )


def _qualified_louisiana_rs_dependencies(text: str) -> tuple[re.Match[str], ...]:
    """Return exact R.S. citations, excluding natural fragment continuations."""

    return tuple(
        match
        for match in _LOUISIANA_RS_DEFERRAL_DEPENDENCY.finditer(text)
        if not _louisiana_rs_match_has_unsupported_fragment_tail(text, match)
    )


def _louisiana_rs_match_has_detached_structural_subject(
    text: str,
    match: re.Match[str],
) -> bool:
    """Reject a labeled parenthetical that is the subject of following prose."""

    tail = match.group("tail") or ""
    if not re.search(
        r"\b(?:clause|division|item|line|paragraph|part|schedule|subsection)s?\s+"
        r"\(",
        tail,
        flags=re.IGNORECASE,
    ):
        return False
    if _louisiana_citation_list_completes_outer_subject(text[: match.start()]):
        return False
    return _louisiana_reference_starts_finite_clause(text[match.end() :])


def _louisiana_rs_match_has_unsupported_fragment_tail(
    text: str,
    match: re.Match[str],
) -> bool:
    """Reject only parenthetical fragments structurally continuing a citation."""

    same_clause_tail = re.split(r"[.;\n]", text[match.end() :], maxsplit=1)[0]
    previous_end = 0
    for parenthetical in re.finditer(
        r"\(\s*[A-Za-z0-9]+(?:[-\u2013][A-Za-z0-9]+)*\s*\)",
        same_clause_tail,
        flags=re.IGNORECASE,
    ):
        prefix = same_clause_tail[previous_end : parenthetical.start()]
        following = same_clause_tail[parenthetical.end() :]
        previous_end = parenthetical.end()
        if re.match(
            r"\s+(?:filers?\b|of\s+(?:the\s+)?(?:calculation|explanatory\s+report|"
            r"form|report|return|tax\s+return|worksheet)\b)",
            following,
            flags=re.IGNORECASE,
        ):
            continue
        if re.fullmatch(
            r"\s*,?\s*(?:"
            r"(?:and|or|plus|as\s+well\s+as|together\s+with|as\s+supplemented\s+by|"
            r"combined\s+with|in\s+addition\s+to|as\s+amended\s+by|including|"
            r"through|to|&|[-\u2013\u2014])|"
            r"followed\s+by(?:\s+(?:clause|division|item|line|paragraph|schedule|"
            r"subclause|subdivision|subitem|subparagraph|subpart|subsection))?|"
            r"as\s+enumerated\s+in|as\s+also\s+set\s+forth\s+at|alongside|"
            r"as\s+extended\s+from|"
            r"(?:as\s+[^,.;]{0,100}|together\s+with\s+[^,.;]{0,100})|"
            r"(?:clause|division|item|paragraph|schedule|subclause|subdivision|"
            r"subitem|subparagraph|subpart|subsection))\s*",
            prefix,
            flags=re.IGNORECASE,
        ):
            return True
        if re.fullmatch(r"\s*,\s*", prefix):
            return True
    return False


def _citation_instrument_identity(
    citation: str,
) -> tuple[str, str, str] | None:
    normalized = citation.split("#", 1)[0].replace(":", "/", 1).strip("/")
    parts = [part.lower() for part in normalized.split("/") if part]
    if len(parts) < 3:
        return None
    collection = parts[1].removesuffix("s")
    return parts[0], collection, parts[2]


def _reason_names_target_instrument(
    normalized_reason: str,
    target_instrument: tuple[str, str, str],
) -> bool:
    _jurisdiction, _collection, instrument = target_instrument
    normalized_instrument = re.sub(
        r"[^a-z0-9äöüß]+",
        " ",
        instrument,
    ).strip()
    return bool(
        normalized_instrument
        and re.search(
            rf"(?<![a-z0-9äöüß]){re.escape(normalized_instrument)}"
            r"(?![a-z0-9äöüß])",
            normalized_reason,
        )
    )


def _reason_names_absolute_target(
    normalized_reason: str,
    target_instrument: tuple[str, str, str],
) -> bool:
    jurisdiction, collection, instrument = target_instrument
    absolute_target = " ".join((jurisdiction, collection, instrument))
    plural_absolute_target = " ".join((jurisdiction, f"{collection}s", instrument))
    return any(
        re.search(
            rf"(?<![a-z0-9äöüß]){re.escape(candidate)}"
            r"(?![a-z0-9äöüß])",
            normalized_reason,
        )
        for candidate in (absolute_target, plural_absolute_target)
    )


def _reason_dependency_is_source_bound(
    reason: str,
    source_scope_text: str,
    *,
    corpus_citation_path: str,
    path: tuple[str, ...] | None = None,
    authenticated_same_act_aliases: Sequence[str] = (),
) -> bool:
    """Require one external dependency citation to bind to the deferred source."""

    if _reason_names_missing_same_act_dependency(
        reason,
        source_scope_text,
        corpus_citation_path=corpus_citation_path,
        path=path,
        authenticated_same_act_aliases=authenticated_same_act_aliases,
    ):
        return True

    louisiana_dependencies = _qualified_louisiana_rs_dependencies(reason)
    if not _MISSING_DEPENDENCY_LANGUAGE.search(reason) and not any(
        _reason_match_has_bounded_insufficiency(
            reason,
            dependency,
            source_scope_text=source_scope_text,
        )
        for dependency in louisiana_dependencies
    ):
        return False
    try:
        current_citation = parse_usc_citation(corpus_citation_path)
    except ValueError:
        current_citation = None
    usc_dependencies = _qualified_usc_dependencies(reason)
    for match in usc_dependencies:
        if not _reason_match_names_missing_dependency(
            reason,
            match,
            source_scope_text=source_scope_text,
            current_usc_title=(current_citation.title if current_citation else None),
        ) or not _usc_dependency_is_external(
            match,
            current_citation=current_citation,
        ):
            continue
        section = normalize_rulespec_path_segment(match.group("section"))
        if _source_scope_cites_usc_dependency(
            source_scope_text,
            title=match.group("title"),
            section=section,
            fragments=_usc_dependency_fragments(match),
            allow_relative_reference=(
                current_citation is not None
                and current_citation.title.lower() == match.group("title").lower()
            ),
        ):
            return True

    for match in louisiana_dependencies:
        if not _reason_match_names_missing_dependency(
            reason,
            match,
            source_scope_text=source_scope_text,
            current_usc_title=None,
            allow_bounded_insufficiency=True,
        ) or not _louisiana_rs_dependency_is_external(
            match,
            corpus_citation_path=corpus_citation_path,
        ):
            continue
        if _source_scope_cites_louisiana_rs_dependency(
            source_scope_text,
            title=match.group("title"),
            section=match.group("section"),
            fragments=_usc_dependency_fragments(match),
            tail_identity=_louisiana_rs_tail_identity(match),
        ):
            return True

    current_section = normalize_rulespec_path_segment(
        corpus_citation_path.rstrip("/").rsplit("/", 1)[-1].lower()
    )
    for match in _PRECISE_DEFERRAL_DEPENDENCY.finditer(reason):
        if any(
            dependency.start() <= match.start() and match.end() <= dependency.end()
            for dependency in usc_dependencies
        ):
            continue
        dependency = match.group(0).strip()
        if not _reason_match_names_missing_dependency(
            reason,
            match,
            source_scope_text=source_scope_text,
            current_usc_title=(current_citation.title if current_citation else None),
        ) or not _prose_dependency_is_external(
            dependency, current_section=current_section
        ):
            continue
        if "#" in dependency and ":" in dependency:
            if _source_scope_identifies_blocker(
                source_scope_text,
                dependency,
                corpus_citation_path=corpus_citation_path,
            ):
                return True
            continue
        section_match = re.search(r"\d+[a-z]?", dependency, flags=re.IGNORECASE)
        corpus_target = _rulespec_target_base(corpus_citation_path)
        corpus_instrument_target = corpus_target.rsplit("/", 1)[0]
        if section_match:
            if _source_scope_identifies_blocker(
                source_scope_text,
                f"{corpus_instrument_target}/{section_match.group(0)}#dependency",
                corpus_citation_path=corpus_citation_path,
            ):
                return True
            continue
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


def _reason_names_missing_same_act_dependency(
    reason: str,
    source_scope_text: str,
    *,
    corpus_citation_path: str,
    path: tuple[str, ...] | None,
    authenticated_same_act_aliases: Sequence[str] = (),
) -> bool:
    """Recognize an exact missing session-law section named by one source branch."""

    if not path:
        return False
    exact_current_branch = _reason_cites_exact_current_statute_branch(
        reason,
        corpus_citation_path=corpus_citation_path,
        path=path,
        strict_terminal=True,
    )
    if not exact_current_branch:
        current_section = corpus_citation_path.rstrip("/").rsplit("/", 1)[-1]
        branch_pattern = r"\s*".join(
            rf"\(\s*{re.escape(normalize_rulespec_path_segment(part))}\s*\)"
            for part in path
        )
        exact_current_branch = bool(
            current_section
            and re.search(
                rf"\bsection\s+{re.escape(current_section)}\s*{branch_pattern}"
                r"(?!\s*\()(?![A-Za-z0-9_])",
                reason,
                flags=re.IGNORECASE,
            )
        )
    if not exact_current_branch:
        return False
    source_sections = {
        match.group("section").lower()
        for match in _SAME_ACT_SECTION_DEPENDENCY.finditer(source_scope_text)
        if re.search(
            r"\b(?:except|unless|subject\s+to)\b",
            source_scope_text[max(0, match.start() - 80) : match.start()],
            flags=re.IGNORECASE,
        )
    }
    reason_matches = _same_act_section_dependencies(
        reason,
        authenticated_same_act_aliases=authenticated_same_act_aliases,
    )
    reason_sections = {section.lower() for section, _start, _end in reason_matches}
    matching_sections = reason_sections.intersection(source_sections)
    if not matching_sections:
        return False
    clauses = tuple(re.finditer(r"[^.;\n]+", reason))
    seen_sections: set[str] = set()
    for dependency_section, dependency_start, _dependency_end in reason_matches:
        identity = dependency_section.lower()
        if identity not in matching_sections or identity in seen_sections:
            continue
        seen_sections.add(identity)
        clause_index = next(
            (
                index
                for index, clause in enumerate(clauses)
                if clause.start() <= dependency_start < clause.end()
            ),
            None,
        )
        if clause_index is None:
            continue
        section = re.escape(dependency_section)
        # Keep the authenticated reference clause in view while checking the
        # next missing-state claim and any immediately following reversal.
        clause_end = clauses[min(clause_index + 2, len(clauses) - 1)].end()
        clause_text = reason[clauses[clause_index].start() : clause_end]
        if _same_act_clause_names_missing_dependency(
            clause_text,
            section=section,
            authenticated_same_act_aliases=authenticated_same_act_aliases,
        ):
            return True
    return False


def _same_act_section_dependencies(
    text: str,
    *,
    authenticated_same_act_aliases: Sequence[str],
) -> tuple[tuple[str, int, int], ...]:
    """Return canonical and provenance-authenticated same-act section references."""

    matches = [
        (match.group("section"), match.start(), match.end())
        for match in _SAME_ACT_SECTION_DEPENDENCY.finditer(text)
    ]
    for alias in dict.fromkeys(authenticated_same_act_aliases):
        normalized_alias = " ".join(str(alias).split())
        if not normalized_alias:
            continue
        pattern = re.compile(
            r"\bsection\s+(?P<section>\d+[a-z]?)\s+of\s+"
            + re.escape(normalized_alias)
            + r"\b",
            flags=re.IGNORECASE,
        )
        matches.extend(
            (match.group("section"), match.start(), match.end())
            for match in pattern.finditer(text)
        )
        authenticated_year = re.match(r"(?P<year>\d{4})\b", normalized_alias)
        if authenticated_year:
            # A year-only "the 2025 act" reference is meaningful only in the
            # context of a provenance-authenticated current-row act alias. It
            # is never accepted without that metadata-bound alias.
            year_pattern = re.compile(
                r"\bsection\s+(?P<section>\d+[a-z]?)\s+of\s+(?:the\s+)?"
                + re.escape(authenticated_year.group("year"))
                + r"\s+act\b",
                flags=re.IGNORECASE,
            )
            matches.extend(
                (match.group("section"), match.start(), match.end())
                for match in year_pattern.finditer(text)
            )
    return tuple(sorted(dict.fromkeys(matches), key=lambda match: (match[1], match[2])))


_SAME_ACT_DEPENDENCY_OBJECT = (
    r"(?:(?:the|that|those)\s+)?(?:(?:exact|external|legal|operative|same-act|"
    r"source-(?:bound|stated)|runtime)\s+){0,3}"
    r"(?:(?:displacement|dependency|implementation|rate)\s+){0,2}"
    r"(?:capabilit(?:y|ies)|conditions?|dependenc(?:y|ies)|requirements?)"
    r"(?:\s+(?:needed|required)\s+to\s+"
    r"(?:apply|compute|determine|encode|implement|resolve)\s+"
    r"(?:(?:the|that)\s+)?(?:controlling\s+)?"
    r"(?:amount|conditions?|effect|rate|result|rule))?"
)
_SAME_ACT_EXECUTABLE_OBJECT = (
    r"executable\s+(?:(?:exact|same-act|source-bound|rulespec)\s+){0,2}"
    r"(?:capabilit(?:y|ies)|exports?|implementations?|outputs?|rules?)"
)


def _same_act_identity_pattern(
    authenticated_same_act_aliases: Sequence[str],
) -> str:
    """Build canonical and provenance-authenticated current-act identities."""

    identities = [r"(?:this|the)\s+act"]
    for alias in dict.fromkeys(authenticated_same_act_aliases):
        normalized_alias = " ".join(str(alias).split())
        if not normalized_alias:
            continue
        identities.append(re.escape(normalized_alias))
        authenticated_year = re.match(r"(?P<year>\d{4})\b", normalized_alias)
        if authenticated_year:
            identities.append(
                rf"(?:the\s+)?{re.escape(authenticated_year.group('year'))}\s+act"
            )
    return rf"(?:{'|'.join(identities)})"


def _same_act_clause_names_missing_dependency(
    clause_text: str,
    *,
    section: str,
    authenticated_same_act_aliases: Sequence[str] = (),
) -> bool:
    act_identity = _same_act_identity_pattern(
        authenticated_same_act_aliases,
    )
    reference = rf"\bsection\s+{section}(?:\s+of\s+{act_identity})?\b"
    direct_state = (
        r"\s*(?:,\s*)?(?:(?:which|that)\s+|whose\s+(?:text|body)\s+)?"
        r"(?:is|remains)\s+"
        r"(?:missing|unavailable|not\s+(?:available|encoded|implemented|supplied))\b"
    )
    direct_match = re.search(reference + direct_state, clause_text, flags=re.IGNORECASE)
    if direct_match and not _same_act_dependency_state_is_reversed(
        clause_text,
        direct_match,
        section=section,
        authenticated_same_act_aliases=authenticated_same_act_aliases,
    ):
        return True
    named_object_state = re.search(
        reference + rf"(?P<object>\s+{_SAME_ACT_DEPENDENCY_OBJECT})\s+"
        r"(?:is|are|remains)\s+"
        r"(?:missing|unavailable|not\s+(?:available|encoded|implemented|supplied))\b",
        clause_text,
        flags=re.IGNORECASE,
    )
    if (
        named_object_state
        and not _ADVERSATIVE_LANGUAGE.search(named_object_state.group("object"))
        and not _same_act_dependency_state_is_reversed(
            clause_text,
            named_object_state,
            section=section,
            authenticated_same_act_aliases=authenticated_same_act_aliases,
        )
    ):
        return True
    no_executable_for_section = (
        rf"\bno\s+{_SAME_ACT_EXECUTABLE_OBJECT}\b[^,;.\n]{{0,100}}"
        r"\b(?:for|from|under)\s+(?:the\s+)?" + reference
    )
    no_executable_match = re.search(
        no_executable_for_section,
        clause_text,
        flags=re.IGNORECASE,
    )
    if no_executable_match and not _same_act_dependency_state_is_reversed(
        clause_text,
        no_executable_match,
        section=section,
        authenticated_same_act_aliases=authenticated_same_act_aliases,
    ):
        return True
    same_act_text_state = re.search(
        reference + r"[^.;\n]{0,180}\b(?:text|body)\s+of\s+(?:that|the)\s+"
        r"(?:exact\s+)?same[- ]act\s+section\s+"
        + section
        + r"\s+dependency\s+(?:is|remains)\s+"
        r"(?:missing|unavailable|not\s+(?:available|encoded|implemented|supplied))\b",
        clause_text,
        flags=re.IGNORECASE,
    )
    if same_act_text_state and not _same_act_dependency_state_is_reversed(
        clause_text,
        same_act_text_state,
        section=section,
        authenticated_same_act_aliases=authenticated_same_act_aliases,
    ):
        return True
    dependency_bridge = (
        r"\b(?:requires?|depends?\s+on|until|without)\s+"
        r"(?:an?\s+|the\s+)?(?:executable\s+)?" + reference
    )
    bridge_match = re.search(dependency_bridge, clause_text, flags=re.IGNORECASE)
    return bool(
        bridge_match
        and not _ADVERSATIVE_LANGUAGE.search(bridge_match.group(0))
        and not _same_act_dependency_state_is_reversed(
            clause_text,
            bridge_match,
            section=section,
            authenticated_same_act_aliases=authenticated_same_act_aliases,
        )
    )


def _same_act_dependency_state_is_reversed(
    clause_text: str,
    dependency_match: re.Match[str],
    *,
    section: str,
    authenticated_same_act_aliases: Sequence[str] = (),
) -> bool:
    """Reject bounded negation or a coordinated reversal of this dependency."""

    prefix = clause_text[: dependency_match.start()]
    if re.search(
        r"\b(?:"
        r"(?:it\s+is\s+)?(?:false|untrue|not\s+true)\s+that"
        r"(?:\s+there\s+is)?|"
        r"no\s+(?:basis|evidence|indication|support)\s+that|"
        r"not\s+(?:because|due\s+to)"
        r")\s*$",
        prefix,
        flags=re.IGNORECASE,
    ):
        return True
    act_identity = _same_act_identity_pattern(authenticated_same_act_aliases)
    same_section_identity = (
        rf"(?:(?:that|the)\s+|(?:the\s+)?same[- ]act\s+)?section\s+{section}"
        rf"(?:\s+of\s+{act_identity})?"
    )
    same_section_subject = (
        same_section_identity + rf"(?:\s*,?\s*(?:whose\s+)?(?:body|text|"
        rf"{_SAME_ACT_DEPENDENCY_OBJECT}))?"
    )
    state_link = r"(?:(?:is|are|remains)|(?:has|have)\s+been)"
    positive_state = (
        state_link + r"\s+(?:(?:actually|fully|in\s+fact)\s+)?"
        r"(?:accessible|available|encoded|implemented|present|provided|supplied)\b"
        r"(?:\s+in\s+full)?"
    )
    no_executable_claim = bool(
        re.match(r"\s*no\s+executable\b", dependency_match.group(0), re.IGNORECASE)
    )
    executable_subject = (
        rf"(?:(?:a|an|the|that|this)\s+)?{_SAME_ACT_EXECUTABLE_OBJECT}"
        rf"(?:\s+(?:for|from|under)\s+(?:the\s+)?{same_section_identity})?"
        r"(?!\s+(?:for|from|under)\b)"
    )
    possessive_executable_subject = (
        rf"(?:its|{same_section_identity}'s|(?:the|that)\s+section's)\s+"
        rf"{_SAME_ACT_EXECUTABLE_OBJECT}"
    )
    prior_positive_subject = (
        executable_subject if no_executable_claim else same_section_subject
    )
    if re.search(
        rf"\b{prior_positive_subject}\s*,?\s*(?:which\s+)?{positive_state}",
        prefix,
        flags=re.IGNORECASE,
    ):
        return True
    implementation_state = (
        state_link + r"\s+(?:(?:actually|fully|in\s+fact)\s+)?"
        r"(?:encoded|implemented)\b"
    )
    if no_executable_claim and re.search(
        rf"\b{same_section_identity}\s+{implementation_state}",
        prefix,
        flags=re.IGNORECASE,
    ):
        return True
    suffix = clause_text[dependency_match.end() :]
    reversal_subject = (
        rf"(?:it|they|{executable_subject}|{possessive_executable_subject})"
    )
    if not no_executable_claim:
        reversal_subject = (
            rf"(?:it|they|that\s+section|the\s+section|{same_section_subject}|"
            rf"{executable_subject}|{possessive_executable_subject})"
        )
    reversal_state = (
        state_link + r"\s+(?:(?:actually|fully|in\s+fact)\s+)?"
        r"(?:accessible|available|encoded|implemented|present|provided|supplied|"
        r"not\s+(?:missing|required|unavailable))|"
        r"(?:is\s+)?not\s+required|(?:does\s+)?exists?"
    )
    explicit_reversal = re.search(
        rf"\b(?:although|but|however|nevertheless|nonetheless|though|yet|and)\b"
        rf"[^.;\n]{{0,100}}?\b{reversal_subject}\s+{reversal_state}\b",
        suffix,
        flags=re.IGNORECASE,
    )
    section_implementation_reversal = None
    executable_existence_reversal = None
    if no_executable_claim:
        reversal_boundary = (
            r"(?:\b(?:although|but|however|nevertheless|nonetheless|though|yet|and)\b"
            r"[^.;\n]{0,100}?|[.;]\s*)"
        )
        section_implementation_reversal = re.search(
            rf"{reversal_boundary}\b{same_section_identity}\s+"
            rf"{implementation_state}\b",
            suffix,
            flags=re.IGNORECASE,
        )
        executable_existence_reversal = re.search(
            rf"{reversal_boundary}\b(?:there\s+(?:is|are)\s+{executable_subject}|"
            rf"{same_section_identity}\s+(?:has|have)\s+{executable_subject})\b",
            suffix,
            flags=re.IGNORECASE,
        )
    elliptical_reversal = re.match(
        rf"\s*[^.;\n]{{0,100}}?\b"
        rf"(?:although|but|however|nevertheless|nonetheless|though|yet|and)\b"
        rf"\s+(?:(?:it|they)\s+)?{reversal_state}\b",
        suffix,
        flags=re.IGNORECASE,
    )
    adjacent_clause_reversal = re.match(
        rf"\s*[^.;\n]{{0,100}}?[.;]\s*"
        rf"{reversal_subject}\s+{reversal_state}\b",
        suffix,
        flags=re.IGNORECASE,
    )
    return bool(
        explicit_reversal
        or section_implementation_reversal
        or executable_existence_reversal
        or elliptical_reversal
        or adjacent_clause_reversal
    )


def _reason_match_names_missing_dependency(
    reason: str,
    match: re.Match[str],
    *,
    source_scope_text: str,
    current_usc_title: str | None,
    allow_bounded_insufficiency: bool = False,
) -> bool:
    clause_start, clause_end = _reason_clause_bounds(reason, match)
    clause = reason[clause_start:clause_end]
    reference_start = match.start() - clause_start
    reference_end = match.end() - clause_start
    before = clause[:reference_start]
    after = clause[reference_end:]
    if _reason_dependency_occurrence_is_contextual(
        reason,
        match,
        current_usc_title=current_usc_title,
    ):
        return False
    if not _reason_named_instruments_are_source_bound(
        before,
        source_scope_text,
        dependency_match=match,
    ):
        return False
    direct_missing_state = re.match(
        r"\s*(?:"
        r"(?:is|are)\s+(?:missing|unavailable|not\s+(?:yet\s+)?(?:encoded|implemented|available))|"
        r"(?:has|have)\s+not\s+been\s+(?:encoded|implemented|made\s+available)|"
        r"fehlt|fehlen|ist\s+nicht\s+codiert|sind\s+nicht\s+codiert"
        r")\b",
        after,
        flags=re.IGNORECASE,
    )
    if allow_bounded_insufficiency and _reason_match_has_bounded_insufficiency(
        reason,
        match,
        source_scope_text=source_scope_text,
    ):
        return True
    direct_signals = list(_MISSING_DEPENDENCY_LANGUAGE.finditer(before))
    direct_signal_text = direct_signals[-1].group(0) if direct_signals else ""
    if (
        direct_missing_state
        and direct_signals
        and not re.fullmatch(
            r"depends?\s+on|requires?",
            direct_signal_text,
            flags=re.IGNORECASE,
        )
        and _reason_direct_missing_introduction_is_bounded(
            before[direct_signals[-1].end() :],
            signal=direct_signal_text,
        )
        and not (
            _qualified_usc_dependencies(before)
            or _PRECISE_DEFERRAL_DEPENDENCY.search(before)
        )
        and _reason_state_tail_is_bounded(after[direct_missing_state.end() :])
    ):
        return True

    signals = direct_signals
    if not signals:
        return False
    signal = signals[-1]
    if re.fullmatch(
        r"cannot\s+be\s+(?:computed|encoded|resolved)",
        signal.group(0),
        flags=re.IGNORECASE,
    ):
        return False
    bridge = before[signal.end() :]
    if len(bridge) > 240 or _has_adversative_language(bridge):
        return False

    signal_text = signal.group(0).lower()
    if re.fullmatch(r"depends?\s+on|requires?", signal_text):
        introduction = before[: signal.end()]
        introduction = re.sub(
            r"^\s*cannot\s+be\s+(?:computed|encoded|resolved)\s+"
            r"(?:until|because|since)\s+",
            "",
            introduction,
            flags=re.IGNORECASE,
        )
        return _reason_dependency_introduction_is_bounded(
            introduction
        ) and _reason_suffix_has_dependency_state(
            after,
            allow_descriptive_list=False,
        )
    if signal_text in {
        "missing",
        "unavailable",
        "not yet encoded",
        "fehlt",
        "nicht codiert",
    }:
        prior_scope = before[: signal.start()]
        prior_dependencies = sorted(
            (
                *_qualified_usc_dependencies(prior_scope),
                *_PRECISE_DEFERRAL_DEPENDENCY.finditer(prior_scope),
            ),
            key=lambda dependency: (dependency.end(), dependency.start()),
            reverse=True,
        )
        if prior_dependencies:
            return False

    if signal_text == "until":
        before_reference = clause[:reference_start]
        executable_bridge = bool(
            match.re is _LOUISIANA_RS_DEFERRAL_DEPENDENCY
            and re.fullmatch(
                r"\s*(?:(?:a|an|the)\s+)?executable\s*",
                bridge,
                flags=re.IGNORECASE,
            )
        )
        if not (_reason_reference_introduction_is_bounded(bridge) or executable_bridge):
            return False
        operative_reference = _source_clause_links_dependency(
            clause,
            reference_start=reference_start,
            reference_end=reference_end,
        ) or bool(
            re.search(
                r"\b(?:cited|defined|described|provided|required|set|specified)"
                r"\s+(?:by|in)\s*$",
                before_reference,
                flags=re.IGNORECASE,
            )
        )
        return _reason_suffix_has_dependency_state(
            after,
            allow_descriptive_list=operative_reference,
            allow_executable_object=executable_bridge,
        )
    return True


def _reason_match_has_bounded_insufficiency(
    reason: str,
    match: re.Match[str],
    *,
    source_scope_text: str,
) -> bool:
    """Recognize one citation-local statement that a supplied dependency is incomplete."""

    _clause_start, clause_end = _reason_clause_bounds(reason, match)
    if clause_end < len(reason) and clause_end - match.end() <= 240:
        later_stop = re.search(r"[.;\n]", reason[clause_end + 1 :])
        clause_end = (
            min(len(reason), clause_end + 1 + later_stop.end())
            if later_stop is not None
            else len(reason)
        )
    after = reason[match.end() : clause_end]
    unavailable_export = re.fullmatch(
        r"\s*(?:[,;:]\s*)?(?:but\s+)?no\s+"
        r"(?P<missing>[^.;\n]{1,200}?)\s+"
        r"(?:is|are)\s+(?:available|encoded|implemented|provided|supplied)"
        r"(?:\s+in\s+(?:the\s+)?(?:available|provided|supplied)\s+context)?"
        r"\s*\.?\s*",
        after,
        flags=re.IGNORECASE,
    )
    if unavailable_export is not None:
        missing_scope = unavailable_export.group("missing")
        return bool(
            _missing_scope_starts_with_executable_object(missing_scope)
            and _unavailable_export_names_source_concept(
                missing_scope,
                source_scope_text=source_scope_text,
            )
            and not _executable_object_scope_claims_ready(missing_scope)
            and not _missing_scope_reverses_insufficiency(missing_scope)
            and not tuple(_reason_dependencies(missing_scope))
        )
    insufficiency = re.match(
        r"\s*(?:(?:dependency|context|module|output|provision|rulespec)\s+)?"
        r"(?:(?:provides?|exports?)\s+only\b(?P<limited>[^.;\n]{1,200}?)"
        r"\s+and\s+)?(?P<verb>lacks?|does\s+not\s+provide)\s+",
        after,
        flags=re.IGNORECASE,
    )
    if insufficiency is None:
        return False
    limited = insufficiency.group("limited") or ""
    if _has_adversative_language(limited):
        return False
    dependencies_after = tuple(
        dependency
        for dependency in _reason_dependencies(after)
        if not (
            re.search(
                r"\b(?:both|those)\s*$",
                after[: dependency.start()],
                flags=re.IGNORECASE,
            )
            and re.match(
                r"\s+[A-Za-z][A-Za-z-]*\b",
                after[dependency.end() :],
            )
        )
    )
    if any(
        dependency.start() < insufficiency.end() for dependency in dependencies_after
    ):
        return False
    missing_tail = after[insufficiency.end() :]
    if re.match(r"\s*(?:no|not|none|without)\b", missing_tail, flags=re.IGNORECASE):
        return False
    current_family = _dependency_citation_family(match)
    current_identity = (
        match.groupdict().get("title", "").lower(),
        normalize_rulespec_path_segment(match.group("section").lower()),
        _usc_dependency_fragments(match),
    )
    foreign_dependency_starts = []
    for dependency in dependencies_after:
        dependency_groups = dependency.groupdict()
        if not dependency_groups.get("section"):
            foreign_dependency_starts.append(dependency.start())
            continue
        dependency_identity = (
            (dependency_groups.get("title") or "").lower(),
            normalize_rulespec_path_segment(dependency_groups["section"].lower()),
            _usc_dependency_fragments(dependency),
        )
        if (
            _dependency_citation_family(dependency) != current_family
            or dependency_identity != current_identity
        ):
            foreign_dependency_starts.append(dependency.start())
    missing_end = min(foreign_dependency_starts, default=len(after))
    missing_scope = after[insufficiency.end() : missing_end]
    missing_scope = _louisiana_normalize_parenthetical_aspect(missing_scope)
    missing_scope = re.sub(
        r"\b(?:(?:although|even\s+though|though|while)\s+"
        r"(?:it|this|that)\s+(?:is|was)\s+(?:[A-Za-z][A-Za-z-]*ly\s+)?|"
        r"(?:despite|notwithstanding)\s+(?:the\s+fact\s+)?that\s+"
        r"(?:it|this|that)\s+(?:is|was)\s+|despite\s+(?:itself\s+)?)"
        r"not\s+(?:being\s+)?(?=(?:merely\s+)?(?:an?\s+)?"
        r"(?:hypothetical|proposed)\b)",
        "not ",
        missing_scope,
        flags=re.IGNORECASE,
    )
    missing_scope = re.sub(
        r"\b(?:although|even\s+though|though|while)\s+"
        r"(?:[A-Za-z][A-Za-z-]*ly\s+)?"
        r"(?=not\s+(?:merely\s+)?(?:an?\s+)?"
        r"(?:hypothetical|proposed)\b)",
        "",
        missing_scope,
        flags=re.IGNORECASE,
    )
    return bool(
        len(missing_scope) <= 240
        and _missing_scope_starts_with_executable_object(missing_scope)
        and not _missing_scope_reverses_insufficiency(missing_scope)
    )


def _unavailable_export_names_source_concept(
    missing_scope: str,
    *,
    source_scope_text: str,
) -> bool:
    """Bind a missing executable export to the source-linked object family."""

    source_link = re.search(
        r"\bat\s+(?:the\s+)?(?P<object>amounts?|percentages?|rates?)\s+"
        r"provided(?:\s+for)?\s+in\s+" + _LOUISIANA_RS_DEFERRAL_DEPENDENCY.pattern,
        source_scope_text,
        flags=re.IGNORECASE,
    )
    if source_link is None:
        return False
    source_signature = _louisiana_object_signature(source_link.group("object"))
    if source_signature is None:
        return False

    wrapper = re.match(
        r"\s*(?:(?:an?|the)\s+)?(?:executable\s+)?(?:RuleSpec\s+)?"
        r"(?:exports?|outputs?)\s+for\s+(?P<concept>.+)",
        missing_scope,
        flags=re.IGNORECASE,
    )
    concept_scope = wrapper.group("concept") if wrapper is not None else missing_scope
    if re.fullmatch(
        r"\s*(?:the\s+)?(?:"
        r"rates?\s+applicable\s+to\s+"
        r"(?:corporations?|(?:corporation|corporate)\s+taxable\s+income)|"
        r"rates?\s+used\s+to\s+compute\s+corporation\s+tax|"
        r"rates?\s+that\s+apply\s+to\s+corporations?"
        r")\s*",
        concept_scope,
        flags=re.IGNORECASE,
    ):
        concept_scope = "corporate rates"
    elif re.search(r"\b(?:that|which)\b", concept_scope, flags=re.IGNORECASE):
        return False
    concept_signature = _louisiana_object_signature(concept_scope)
    if concept_signature is None or concept_signature[0] != source_signature[0]:
        return False

    source_modifiers = {
        _louisiana_normalize_modifier_token(
            {"corporation": "corporate", "corporations": "corporate"}.get(
                token,
                token,
            )
        )
        for token in re.findall(r"[a-z][a-z0-9]+", source_scope_text.lower())
    }
    generic_modifiers = {"applicable"}
    return concept_signature[1] <= source_modifiers | generic_modifiers


_LOUISIANA_DOCUMENTARY_HEAD = re.compile(
    r"\b(?:abstracts?|annotations?|appendices|appendix|articles?|audits?|background|"
    r"bibliograph(?:y|ies)|briefs?|catalogs?|charts?|commentary|comparisons?|"
    r"descriptions?|digests?|documents?|documentation|examples?|exhibits?|"
    r"explanations?|guidance|illustrations?|index(?:es)?|indices|labels?|legends?|manuals?|"
    r"materials?|memos?|memoranda|memorandums?|metadata|narratives?|notes?|overviews?|"
    r"reports?|summaries|summary|synopses|synopsis|transcripts?|worksheets?)\b",
    flags=re.IGNORECASE,
)

_LOUISIANA_SUPPLY_VERB_PATTERN = (
    r"(?:suppl(?:y|ies|ied|ying)|provid(?:e[sd]?|ing)|export(?:s|ed|ing)?|"
    r"contain(?:s|ed|ing)?|includ(?:e[sd]?|ing)|furnish(?:es|ed|ing)?|"
    r"implement(?:s|ed|ing)?|deliver(?:s|ed|ing)?|return(?:s|ed|ing)?|"
    r"produc(?:e[sd]?|ing)|yield(?:s|ed|ing)?|deriv(?:e[sd]?|ing)|"
    r"expos(?:e[sd]?|ing)|generat(?:e[sd]?|ing)|possess(?:es|ed|ing)?|"
    r"retain(?:s|ed|ing)?|offers?|lists?|reproduces?|computes?|calculates?|"
    r"determines?|defines?|encodes?|publishes?|gives?|"
    r"outputs(?=\s+(?:(?:a|an|the|this|that)\s+)?[A-Za-z])|stores?|holds?|"
    r"specifies?|displays?|emits?|retrieves?|access(?:es|ed|ing)?|uses?|reads?|"
    r"loads?|imports?|keeps?|ship(?:s|ped|ping)?|has|have|makes?|renders?|"
    r"gave|sent|made|kept|had)"
)


def _louisiana_supply_verb_is_completed(verb: str) -> bool:
    """Recognize regular and common irregular completed supply predicates."""

    normalized = verb.lower()
    return normalized.endswith("ed") or normalized in {
        "gave",
        "had",
        "kept",
        "made",
        "sent",
    }


def _missing_scope_starts_with_executable_object(missing_scope: str) -> bool:
    """Require the direct missing object's semantic head to be executable."""

    primary_object_scope = re.split(
        r"\b(?:and|but|while|yet)\b", missing_scope, maxsplit=1, flags=re.IGNORECASE
    )[0]
    if _LOUISIANA_DOCUMENTARY_HEAD.search(primary_object_scope) or re.search(
        r"\b(?:candidate|conceptual|draft|fictional|hypothetical|illustrative|mock|"
        r"notional|placeholder|proposed|prototype|purported|sample|specimen|suggested)\b|"
        r"\b(?:provisional|tentative)\s+(?:amount|calculation|computation|formula|"
        r"method)\b",
        primary_object_scope,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\brecords?\b[^.;\n]{0,100}\b(?:containing|detailing|discussing|"
        r"documenting|that|which|whose)\b",
        primary_object_scope,
        flags=re.IGNORECASE,
    ):
        return False
    direct_object = re.split(
        r"\b(?:and|but|while|yet|that|which|who|whose|about|against|among|around|at|"
        r"before|behind|below|beneath|beside|between|beyond|by|concerning|"
        r"considering|covering|despite|down|during|except|following|for|from|in|inside|"
        r"into|like|near|of|off|on|onto|opposite|outside|over|past|regarding|"
        r"round|since|through|throughout|to|toward|under|underneath|unlike|"
        r"until|up|upon|via|with|within|without|contain\w*|detail\w*|describ\w*|"
        r"discuss\w*|document\w*|explain\w*|illustrat\w*|outline\w*|"
        r"demonstrat\w*|summariz\w*|"
        r"includ\w*|support\w*|needed|used|intended|required\s+to|"
        r"necessary\s+to)\b|,",
        missing_scope,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    direct_object = _LOUISIANA_RS_DEFERRAL_DEPENDENCY.sub(" ", direct_object)
    tokens = re.findall(r"[A-Za-z]+", direct_object.lower())
    while tokens and tokens[0] in {"a", "an", "the"}:
        tokens.pop(0)
    if not tokens:
        return False
    if _louisiana_object_signature(tokens[-1]) is not None:
        return True
    return bool(
        tokens[-1] in {"set", "sets"}
        and len(tokens) > 1
        and _louisiana_object_signature(tokens[-2]) is not None
    )


def _executable_object_scope_claims_ready(missing_scope: str) -> bool:
    """Reject a relative or appositive assertion that the object already exists."""

    if re.fullmatch(
        r"\s*(?:(?:an?|the)\s+)?(?:executable\s+)?(?:RuleSpec\s+)?"
        r"(?:exports?|outputs?)\s+for\s+(?:the\s+)?(?:"
        r"rates?\s+used\s+to\s+compute\s+corporation\s+tax"
        r")\s*",
        missing_scope,
        flags=re.IGNORECASE,
    ):
        return False

    ready_state = (
        r"(?:accessible|accessed|available|complete|computed|calculated|determined|"
        r"defined|deployed|downloadable|downloaded|encoded|executed|extant|found|"
        r"furnished|functional|generated|implemented|in\s+place|installed|live|loaded|"
        r"made\s+(?:available|obtainable)|obtainable|obtained|on\s+file|on\s+hand|"
        r"operational|present|produced|provided|published|ready|ready\s+for\s+use|"
        r"recoverable|recovered|released|rendered\s+(?:available|accessible)|"
        r"retrievable|retrieved|returned|run|set|stored|supplied|usable|used|verified|"
        r"working|at\s+(?:hand|the\s+ready))"
    )
    affirmative_adverb = (
        r"(?:actively|actually|already|always|currently|directly|fully|immediately|"
        r"just|now|presently|previously|readily|still)"
    )
    direct_readiness = re.search(
        rf"\b(?:"
        rf"(?:{affirmative_adverb}\s+)*{ready_state}"
        r"(?:\s+(?:here|in\s+(?:this|the)\s+context))?|"
        r"(?:this|the)\s+context\s+"
        r"(?:exports?|furnishes?|has|holds?|implements?|provides?|supplies?)"
        r")\b",
        missing_scope,
        flags=re.IGNORECASE,
    )
    if direct_readiness is not None:
        readiness_prefix = missing_scope[: direct_readiness.start()]
        readiness_tail = missing_scope[direct_readiness.end() :]
        epistemically_limited = re.search(
            r"\b(?:barely|conditionally|hardly|not|supposedly|theoretically)\s*$",
            readiness_prefix,
            flags=re.IGNORECASE,
        )
        conditionally_later = re.match(
            r"\s*(?:(?:only\s+)?(?:after|if|once|unless|upon|when)|"
            r"provided\s+that|in\s+theory(?:\s+only)?|on\s+paper)\b",
            readiness_tail,
            flags=re.IGNORECASE,
        )
        if epistemically_limited is None and conditionally_later is None:
            return True

    qualifier = re.search(
        r"(?P<marker>,|\b(?:that|which)\b)(?P<scope>[^.;\n]{1,180})",
        missing_scope,
        flags=re.IGNORECASE,
    )
    if qualifier is None:
        return False
    scope = qualifier.group("scope")
    if qualifier.group("marker") == ",":
        bare_ready = re.match(
            rf"\s*(?!not\b)"
            rf"(?:{affirmative_adverb}\s+)*"
            rf"{ready_state}\b",
            scope,
            flags=re.IGNORECASE,
        )
        if bare_ready is not None and not re.match(
            r"\s*(?:(?:only\s+)?(?:after|if|once|unless|upon|when)|provided\s+that)\b",
            scope[bare_ready.end() :],
            flags=re.IGNORECASE,
        ):
            return True
    readiness = re.search(
        rf"\b(?:"
        rf"(?:is|are|was|were|becomes?|remains?)\s+(?!not\b)"
        rf"(?:{affirmative_adverb}\s+)*{ready_state}|"
        rf"(?:has|have|had)\s+"
        rf"(?:{affirmative_adverb}\s+)*been\s+"
        rf"(?!not\b)(?:{affirmative_adverb}\s+)*{ready_state}|"
        rf"(?:can|could|may|might)\s+"
        rf"(?:{affirmative_adverb}\s+)*be\s+"
        rf"(?!not\b)(?:{affirmative_adverb}\s+)*{ready_state}|"
        r"exists?\s+(?:already\s+)?(?:in|within)"
        r")\b",
        scope,
        flags=re.IGNORECASE,
    )
    if readiness is not None:
        readiness_tail = scope[readiness.end() :]
        if not re.match(
            r"\s*(?:(?:only\s+)?(?:after|if|once|unless|upon|when)|"
            r"provided\s+that|in\s+theory(?:\s+only)?|on\s+paper)\b",
            readiness_tail,
            flags=re.IGNORECASE,
        ):
            return True
    for assertion in re.finditer(
        rf"\b{_LOUISIANA_SUPPLY_VERB_PATTERN}\b",
        scope,
        flags=re.IGNORECASE,
    ):
        supplier = scope[: assertion.start()]
        supplier = re.sub(
            r"^\s*,\s*(?:(?:after|if|once|unless|upon|when)\b|provided\s+that\b)"
            r"[^,.;\n]{1,100},\s*",
            " ",
            supplier,
            flags=re.IGNORECASE,
        )
        supplier = re.sub(
            r"\b(?:[A-Za-z][A-Za-z-]*ly|already|always|directly|itself|just|now|"
            r"still)\b",
            " ",
            supplier,
            flags=re.IGNORECASE,
        )
        implicit_subject = re.fullmatch(
            r"\s*(?:(?:can|could|may|might|must|shall|should|will|would|to)\s+|"
            r"(?:is|are|was|were)\s+designed\s+to\s+|"
            r"(?:accepts?|accesses?|reads?|receives?|takes?|uses?)\b"
            r"[^,.;\n]{0,100}\b(?:and|then)\s+)*",
            supplier,
            flags=re.IGNORECASE,
        )
        if (
            implicit_subject is None
            and not _louisiana_assertion_is_negative(scope, assertion.start())
            and not _louisiana_supply_assertion_has_nonoperative_framing(
                scope, assertion
            )
        ):
            return True
    return False


def _missing_scope_reverses_insufficiency(missing_scope: str) -> bool:
    """Reject a positive coordinated assertion supplying the missing object."""

    missing_scope = re.sub(
        r",\s*which\s*,\s*(?:"
        r"(?:although|even\s+though|though|while)\s+"
        r"(?:(?:[A-Za-z][A-Za-z-]*ly)(?:\s+and\s+"
        r"[A-Za-z][A-Za-z-]*ly)*\s+)?not\s+(?:merely\s+)?(?:an?\s+)?"
        r"(?:hypothetical|proposed)|"
        r"despite\s+[^,.;]{0,80}\bnot\s+being\s+(?:merely\s+)?(?:an?\s+)?"
        r"(?:hypothetical|proposed)|"
        r"(?:far\s+from|nowhere\s+near|anything\s+but|the\s+opposite\s+of)\s+"
        r"(?:merely\s+)?"
        r"(?:hypothetical|proposed)|"
        r"being\s+(?:actual|real)\s+rather\s+than\s+"
        r"(?:hypothetical|proposed))\s*,\s*",
        " ",
        missing_scope,
        flags=re.IGNORECASE,
    )
    missing_scope = re.sub(
        r"(?P<relative>\b(?:that|which)\b[^,.;]{1,100}),\s*"
        r"(?:if|unless|provided\s+that|so\s+long\s+as)\b[^,.;]{1,80},"
        r"(?=\s*(?:and|as\s+well\s+as|but|or|yet)\s+"
        r"[A-Za-z][A-Za-z-]*\b)",
        r"\g<relative>",
        missing_scope,
        flags=re.IGNORECASE,
    )

    coordinators = tuple(
        re.finditer(
            r"\b(?:although|and|despite|however|notwithstanding|since|then|while)\b|"
            r"(?<!anything )(?<!nothing )\bbut\b|"
            r"(?<!not )\byet\b|"
            r"\bin\s+spite\s+of\b|\beven\s+though\b|[.;]\s*|"
            rf",(?!\s*(?:because|so)\b)\s*(?=(?:then\s+)?[^,.;]{{0,80}}\b"
            rf"(?:(?:(?:can|could|does|may|might|must|shall|should|will|would)\s+)?"
            rf"{_LOUISIANA_SUPPLY_VERB_PATTERN}|"
            r"(?:is|are|was|were|becomes?|remains?)\s+(?:available|complete|present|"
            r"supplied|provided|implemented|accessible))\b)",
            missing_scope,
            re.IGNORECASE,
        )
    )
    if not coordinators:
        return False
    missing_head = _missing_scope_semantic_head(
        missing_scope[: coordinators[0].start()]
    )
    if missing_head is None:
        return False
    missing_object = missing_scope[: coordinators[0].start()]
    coordinated_list_supplies_missing = _louisiana_coordinated_list_supplies_object(
        missing_scope, missing_object=missing_object
    )
    last_supply_assertion = False
    last_explicit_signature: tuple[str, frozenset[str]] | None = None
    numbered_explicit_signatures: dict[str, tuple[str, frozenset[str]]] = {}
    pending_conditional_context = False
    for index, coordinator in enumerate(coordinators):
        segment_end = (
            coordinators[index + 1].start()
            if index + 1 < len(coordinators)
            else len(missing_scope)
        )
        segment = missing_scope[coordinator.end() : segment_end]
        assertions = tuple(
            assertion
            for assertion in re.finditer(
                rf"\b(?P<verb>{_LOUISIANA_SUPPLY_VERB_PATTERN})\b",
                segment,
                flags=re.IGNORECASE,
            )
            if not (
                assertion.group("verb").lower().startswith("return")
                and re.search(
                    r"\b(?:for|of)\s*$",
                    segment[: assertion.start()],
                    flags=re.IGNORECASE,
                )
            )
            and not (
                assertion.group("verb").lower() == "provided"
                and not segment[: assertion.start()].strip()
                and re.match(
                    r"\s+that\b",
                    segment[assertion.end() :],
                    flags=re.IGNORECASE,
                )
            )
            and not (
                _louisiana_supply_verb_is_completed(assertion.group("verb"))
                and (
                    re.search(
                        r"\b(?:is|are|was|were|be|been|being)\s*$",
                        segment[: assertion.start()],
                        flags=re.IGNORECASE,
                    )
                    or not segment[assertion.end() :].strip()
                    or (
                        not re.search(
                            r",\s*(?:because|so)\b",
                            segment[assertion.end() :],
                            flags=re.IGNORECASE,
                        )
                        and re.match(
                            rf"\s+[A-Za-z][A-Za-z-]*\s+[^,.;]{{0,40}}\b"
                            rf"{_LOUISIANA_SUPPLY_VERB_PATTERN}\b",
                            segment[assertion.end() :],
                            flags=re.IGNORECASE,
                        )
                    )
                )
            )
            and not (
                assertion.group("verb").lower().startswith(("contain", "includ"))
                and re.search(
                    r"\b[A-Za-z][A-Za-z-]*(?:s|ed)\s+(?:a|an|the)\s+"
                    r"[^,.;]{1,60}\s+$",
                    segment[: assertion.start()],
                    flags=re.IGNORECASE,
                )
            )
        )
        if not assertions and last_supply_assertion:
            shared_object = re.sub(r"^\s*(?:then\s+)?", "", segment)
            demonstrative = re.match(
                r"\s*(?:this|that|those|it|them|both|the\s+same|same)\b",
                shared_object,
                flags=re.IGNORECASE,
            )
            antecedent_signature = _louisiana_demonstrative_antecedent_signature(
                shared_object,
                last_explicit_signature=last_explicit_signature,
                numbered_signatures=numbered_explicit_signatures,
            )
            if not (
                demonstrative
                and antecedent_signature is not None
                and not _louisiana_signatures_corefer(
                    antecedent_signature,
                    _louisiana_object_signature(missing_object),
                )
            ) and _louisiana_assertion_object_corefers(
                shared_object,
                missing_head=missing_head,
                missing_object=missing_object,
            ):
                return True
        if not assertions and re.match(
            r"\s*(?:if|unless|provided(?:\s+that)?|so\s+long\s+as|in\s+case|"
            r"subject\s+to|dependent\s+on|contingent\s+(?:on|upon)|"
            r"conditioned\s+(?:on|upon)|pending(?:\s+receipt)?|"
            r"(?:on|upon|following)\s+(?:approval|receipt))\b",
            segment,
            flags=re.IGNORECASE,
        ):
            pending_conditional_context = True
        for assertion_index, assertion in enumerate(assertions):
            prefix_signature = _louisiana_pre_supply_antecedent_signature(
                segment[: assertion.start()]
            )
            if prefix_signature is not None:
                last_explicit_signature = prefix_signature
                prefix_number = _louisiana_object_number(segment[: assertion.start()])
                if prefix_number is not None:
                    numbered_explicit_signatures[prefix_number] = prefix_signature
            assertion_end = (
                assertions[assertion_index + 1].start()
                if assertion_index + 1 < len(assertions)
                else len(segment)
            )
            object_scope = segment[assertion.end() : assertion_end]
            if re.match(
                r"\s*(?:this|that|those|it|them|both|the\s+same|same)\b",
                object_scope,
                flags=re.IGNORECASE,
            ):
                direct_object = object_scope
            else:
                direct_object = re.split(
                    r",\s*(?:then\s+)?|\b(?:after|and|because|before|but|"
                    r"concerning|describing|once|that|then|to|used|when|which|"
                    r"while|whose|yet)\b",
                    object_scope,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0]
            if _louisiana_supply_assertion_has_nonoperative_framing(
                segment, assertion
            ) or _louisiana_assertion_is_negative(segment, assertion.start()):
                last_supply_assertion = False
                continue
            if pending_conditional_context and not _louisiana_supply_verb_is_completed(
                assertion.group("verb")
            ):
                pending_conditional_context = False
                last_supply_assertion = False
                continue
            pending_conditional_context = False
            if re.match(
                r"\s*(?:absolutely\s+)?(?:(?:no|none|not)\b|an?\s+incomplete\b|"
                r"only\s+(?:part\s+of|an?\s+(?:draft|incomplete|partial))|"
                r"(?:hardly|barely|scarcely)\s+any|almost\s+no|"
                r"(?:(?:merely\s+)?(?:some|half(?:\s+of)?)|"
                r"an?\s+(?:draft|fragment|partial|subset|fraction)|"
                r"(?:an?\s+few|merely)?\s*(?:pieces?|portions?|fragments?|excerpt)\s+of|"
                r"less\s+than\s+all(?:\s+of)?|most\b))",
                direct_object,
                flags=re.IGNORECASE,
            ):
                last_supply_assertion = False
                continue
            direct_signature = _louisiana_nearest_object_signature(
                direct_object, allow_opaque=True
            )
            demonstrative = bool(
                re.match(
                    r"\s*(?:this|that|those|it|them|both|the\s+same|same)\b",
                    direct_object,
                    flags=re.IGNORECASE,
                )
            )
            antecedent_signature = _louisiana_demonstrative_antecedent_signature(
                direct_object,
                last_explicit_signature=last_explicit_signature,
                numbered_signatures=numbered_explicit_signatures,
            )
            if (
                demonstrative
                and re.match(
                    r"\s*(?:them|those|both)\b",
                    direct_object,
                    flags=re.IGNORECASE,
                )
                and coordinated_list_supplies_missing
            ):
                return True
            if (
                demonstrative
                and antecedent_signature is not None
                and not _louisiana_signatures_corefer(
                    antecedent_signature,
                    _louisiana_object_signature(missing_object),
                )
            ):
                last_supply_assertion = True
                continue
            if _louisiana_assertion_object_corefers(
                direct_object,
                missing_head=missing_head,
                missing_object=missing_object,
            ):
                return True
            if direct_signature is not None:
                if not re.search(
                    r"\b(?:choice|alternative|option)\b",
                    direct_object,
                    flags=re.IGNORECASE,
                ):
                    last_explicit_signature = direct_signature
                    direct_number = _louisiana_object_number(direct_object)
                    if direct_number is not None:
                        numbered_explicit_signatures[direct_number] = direct_signature
            last_supply_assertion = True
            if (
                assertion.group("verb").lower().startswith("return")
                and missing_head in {"calculation", "computation", "formula"}
                and re.search(r"\bamount\b", direct_object, flags=re.IGNORECASE)
            ):
                return True

        precontrast_states = re.finditer(
            r"\b(?P<object>(?:the\s+)?[A-Za-z][A-Za-z -]{0,60}?)\s*"
            r"(?:,\s*(?:which|that)\s*)?"
            r"(?:,\s*|\(\s*|[—–]\s*)"
            r"(?:(?:although|even\s+though|though|while)\s+)?not\s+"
            r"(?:merely\s+)?(?:an?\s+)?"
            r"(?:hypothetical|proposed)(?:\s+[A-Za-z-]+)?"
            r"(?:,\s*|\)\s*|[—–]\s*)"
            r"(?:is|are|was|were|became|becomes?|remains?)\s+"
            r"(?:available|present|complete|accessible|usable|ready|extant|installed|"
            r"released|posted|deployed|in\s+place|on\s+file|on\s+hand|at\s+hand)\b",
            segment,
            flags=re.IGNORECASE,
        )
        for assertion in precontrast_states:
            if _louisiana_assertion_object_corefers(
                assertion.group("object"),
                missing_head=missing_head,
                missing_object=missing_object,
            ):
                return True

        state_assertions = tuple(
            re.finditer(
                r"\b(?P<object>this|that|those|it|them|both|the\s+same|"
                r"(?:the\s+)?[A-Za-z][A-Za-z -]{0,60}?)\s*"
                r"(?:(?:,\s*|\(\s*|[—–]\s*)"
                r"(?:(?:although|even\s+though|though|while)\s+)?not\s+"
                r"(?:merely\s+)?"
                r"(?:an?\s+)?(?:hypothetical|proposed)(?:\s+[A-Za-z-]+)?"
                r"(?:,\s*|\)\s*|[—–]\s*))?"
                r"(?P<state>exists?|(?:(?:can|could|may|might|must|shall|should|will|would)\s+"
                r"(?:(?:[A-Za-z][A-Za-z-]*ly|already|always|just|now|still)\s+)*be|"
                r"(?:has|have|had)\s+"
                r"(?:(?:[A-Za-z][A-Za-z-]*ly|already|always|just|now|still)\s+)*been|"
                r"(?:has|have|had)\s+become|came\s+to\s+be|"
                r"(?:is|are|was|were)\s+(?:being|made)|"
                r"(?:is|are|was|were|became|becomes?|remains?))\s+"
                r"(?:(?:not\s+(?:just|merely)|[A-Za-z][A-Za-z-]*ly|already|always|"
                r"just|now|still)\s+)*"
                r"(?:available|present|supplied|provided|implemented|"
                r"computed|calculated|determined|encoded|defined|accessible|accessed|"
                r"displayed|emitted|retrieved|complete|obtainable|retained|included|"
                r"loaded|imported|stored|obtained|furnished|delivered|published|"
                r"returned|produced|generated|exported|held|given|"
                r"made\s+(?:available|obtainable)|rendered\s+(?:available|accessible)|"
                r"put\s+on\s+hand|ready\s+for\s+use|retrievable|downloadable|"
                r"recoverable|operational|usable|ready|extant|installed|released|posted|"
                r"deployed|downloaded|recovered|found|in\s+place|on\s+file|"
                r"at\s+the\s+ready|on\s+hand|at\s+hand))\b",
                segment,
                flags=re.IGNORECASE,
            )
        )
        for assertion in state_assertions:
            if (
                _louisiana_assertion_is_negative(segment, assertion.start("state"))
                or re.match(
                    r"(?:may|might|should|could|would)\b",
                    assertion.group("state"),
                    flags=re.IGNORECASE,
                )
                or _louisiana_state_assertion_has_nonoperative_framing(
                    segment, assertion
                )
                or pending_conditional_context
                or re.match(
                    r"\s*(?:if|unless|provided\s+that|so\s+long\s+as|in\s+case|"
                    r"subject\s+to|dependent\s+on|contingent\s+(?:on|upon)|"
                    r"conditioned\s+(?:on|upon)|pending(?:\s+receipt)?|"
                    r"(?:on|upon|following)\s+(?:approval|receipt))\b",
                    segment[: assertion.start()],
                    flags=re.IGNORECASE,
                )
                or _louisiana_suffix_has_clause_level_condition(
                    segment[assertion.end() :], include_temporal=False
                )
                or (
                    not re.match(
                        r"(?:was|were|became|has|have|had|came)\b",
                        assertion.group("state"),
                        flags=re.IGNORECASE,
                    )
                    and re.search(
                        r"\bwhenever\b|\bas\s+soon\s+as\b|"
                        r"\bonly\s+(?:after|when)\b|\bafter\b",
                        segment[assertion.end() :],
                        flags=re.IGNORECASE,
                    )
                )
            ):
                continue
            state_object = assertion.group("object")
            if state_object.lower() in {"that", "which"}:
                relative_antecedent = re.search(
                    r"(?P<object>(?:the\s+)?[A-Za-z][A-Za-z -]{0,60}?)\s*,\s*$",
                    segment[: assertion.start()],
                    flags=re.IGNORECASE,
                )
                if relative_antecedent is not None:
                    state_object = relative_antecedent.group("object")
            antecedent_signature = _louisiana_demonstrative_antecedent_signature(
                state_object,
                last_explicit_signature=last_explicit_signature,
                numbered_signatures=numbered_explicit_signatures,
            )
            if (
                re.fullmatch(
                    r"(?:this|that|those|it|them|both|the\s+same)",
                    state_object.strip(),
                    flags=re.IGNORECASE,
                )
                and antecedent_signature is not None
                and not _louisiana_signatures_corefer(
                    antecedent_signature,
                    _louisiana_object_signature(missing_object),
                )
            ):
                continue
            if _louisiana_assertion_object_corefers(
                state_object,
                missing_head=missing_head,
                missing_object=missing_object,
            ):
                return True
        if state_assertions:
            pending_conditional_context = False

        actual_presence = re.finditer(
            r"\b(?P<object>this|that|those|it|them|the\s+same|"
            r"(?:the\s+)?[A-Za-z][A-Za-z -]{0,60}?)\s+"
            r"(?:has|have|had)\s+arrived\b|"
            r"\b(?P<located_object>this|that|those|it|them|the\s+same|"
            r"(?:the\s+)?[A-Za-z][A-Za-z -]{0,60}?)\s+resides?\s+in\s+"
            r"(?:the\s+)?repository\b",
            segment,
            flags=re.IGNORECASE,
        )
        for assertion in actual_presence:
            state_object = assertion.group("object") or assertion.group(
                "located_object"
            )
            if _louisiana_assertion_object_corefers(
                state_object,
                missing_head=missing_head,
                missing_object=missing_object,
            ):
                return True

        if not assertions:
            explicit_signature = _louisiana_coordinated_antecedent_signature(segment)
            if (
                explicit_signature is None
                and re.match(
                    r"\s*(?:a|an|the|this|that)\s+[A-Za-z]",
                    segment,
                    flags=re.IGNORECASE,
                )
                and not re.match(
                    r"\s*(?:the|this|that)?\s*(?:encoder|implementation|module|"
                    r"package|runtime|service|source|system)\b",
                    segment,
                    flags=re.IGNORECASE,
                )
            ):
                explicit_signature = _louisiana_nearest_object_signature(
                    segment, allow_opaque=True
                )
            if explicit_signature is not None and not re.fullmatch(
                r"\s*(?:the|this|that)?\s*(?:encoder|implementation|module|"
                r"package|runtime|service|source|system)\s*",
                segment,
                flags=re.IGNORECASE,
            ):
                previous_signature = last_explicit_signature
                last_explicit_signature = explicit_signature
                explicit_number = _louisiana_object_number(segment)
                if explicit_number is not None:
                    numbered_explicit_signatures[explicit_number] = explicit_signature
                if (
                    coordinator.group(0).strip().lower() == "and"
                    and previous_signature is not None
                    and explicit_number == "singular"
                    and "plural" not in numbered_explicit_signatures
                ):
                    missing_signature = _louisiana_object_signature(missing_object)
                    if _louisiana_signatures_corefer(
                        previous_signature, missing_signature
                    ) or _louisiana_signatures_corefer(
                        explicit_signature, missing_signature
                    ):
                        numbered_explicit_signatures["plural"] = missing_signature

        existential = re.finditer(
            r"\bthere\s+(?:is|are|was|were)\s+(?P<object>[^,.;]{1,80})",
            segment,
            flags=re.IGNORECASE,
        )
        for assertion in existential:
            if _louisiana_assertion_is_negative(segment, assertion.start()):
                continue
            if _louisiana_assertion_object_corefers(
                assertion.group("object"),
                missing_head=missing_head,
                missing_object=missing_object,
            ):
                return True

        capability = re.search(
            r"\b(?:is|are)\s+(?:able\s+to|capable\s+of)\s+"
            r"(?:provid\w*|suppl\w*|comput\w*|calculat\w*|determin\w*)"
            r"(?P<object>[^,.;]{0,80})",
            segment,
            flags=re.IGNORECASE,
        )
        if capability and _louisiana_assertion_object_corefers(
            capability.group("object"),
            missing_head=missing_head,
            missing_object=missing_object,
        ):
            return True
    return False


def _louisiana_coordinated_list_supplies_object(
    missing_scope: str, *, missing_object: str
) -> bool:
    """Recognize a plural pronoun that supplies every member of an object list."""

    missing_scope = re.sub(
        r"\b(?P<pronoun>both|those)\s*[\(\[]\s*(?:"
        r"[A-Za-z][A-Za-z-]*ly|in\s+fact|as\s+expected|with\s+success)\s*[\)\]]",
        r"\g<pronoun>",
        missing_scope,
        flags=re.IGNORECASE,
    )
    missing_scope = re.sub(
        r"\b(?P<pronoun>both|those)\s+in\s+fact(?=\s*[,.;])",
        r"\g<pronoun>",
        missing_scope,
        flags=re.IGNORECASE,
    )
    listed = re.search(
        r"\b(?:identif(?:y|ies|ied)|lists?|finds?|creates?|records?|selects?)\s+"
        r"(?P<objects>(?:not\s+only\s+[^.;]{1,100}\bbut\s+also\b[^.;]{1,100}|"
        r"[^:.;]{1,40}:\s*[^.;]{1,160},[^.;]{1,100}?|"
        r"[^.;]{1,80}\b(?:along\s+with|as\s+well\s+as|in\s+addition\s+to|"
        r"plus|together\s+with)\b"
        r"[^.;]{1,80}?(?:\band\b[^.;]{1,80}?)?|"
        r"(?:[^,.;]{1,80},){2}[^.;]{1,100}?))"
        r",?\s*(?:then\s+)?(?:provid(?:e[sd]?|ing)|suppl(?:y|ies|ied|ying)|"
        r"giv(?:e[sd]?|ing)|gave)\s+"
        r"(?P<pronoun>them|"
        r"those(?!\s*(?:[^\w\s,.;:!?—–]\s*)?[A-Za-z0-9])|"
        r"both(?!\s*(?:[^\w\s,.;:!?—–]\s*)?[A-Za-z0-9]))\b",
        missing_scope,
        flags=re.IGNORECASE,
    )
    if listed is None:
        return False
    objects = re.sub(
        r"^\s*(?:not\s+only\s+|[^:]{1,40}:\s*)",
        "",
        listed.group("objects"),
        flags=re.IGNORECASE,
    )
    additive_coordination = bool(
        re.search(
            r"\b(?:along\s+with|as\s+well\s+as|but\s+also|"
            r"in\s+addition\s+to|plus|together\s+with)\b",
            objects,
            flags=re.IGNORECASE,
        )
    )
    objects = re.sub(r",?\s*then\s*$", "", objects, flags=re.IGNORECASE)
    objects = re.sub(
        r"\bprofit\s+and\s+loss\b", "profit-and-loss", objects, flags=re.IGNORECASE
    )
    objects = re.sub(
        r"\b(?:along\s+with|as\s+well\s+as|but\s+also|in\s+addition\s+to|"
        r"plus|together\s+with)\b",
        ",",
        objects,
        flags=re.IGNORECASE,
    )
    missing_signature = _louisiana_object_signature(missing_object)
    member_separator = r"\s*,\s*" if additive_coordination else r"\s*,\s*|\s+and\s+"
    members = tuple(
        member for member in re.split(member_separator, objects) if member.strip()
    )
    if listed.group("pronoun").lower() == "both" and len(members) != 2:
        return False
    return any(
        _louisiana_signatures_corefer(
            _louisiana_object_signature(member), missing_signature
        )
        for member in members
    )


def _louisiana_supply_assertion_has_nonoperative_framing(
    segment: str, assertion: re.Match[str]
) -> bool:
    """Bind evidential framing to the supply clause it actually governs."""

    prefix = _louisiana_normalize_parenthetical_aspect(segment[: assertion.end()])
    prefix = re.sub(
        r"\bno\s+(?:fewer|less)\s+than\b", "at least", prefix, flags=re.IGNORECASE
    )
    local_prefix = re.split(r"[,.;]", prefix)[-1]
    main_subjects = tuple(
        re.finditer(
            r"\b(?:the|this|that)\s+(?:encoder|implementation|module|package|"
            r"runtime|service|source|system)\b",
            local_prefix,
            flags=re.IGNORECASE,
        )
    )
    if main_subjects:
        local_prefix = local_prefix[main_subjects[-1].start() :]
    if _louisiana_text_has_nonoperative_framing(local_prefix):
        return True
    verb = assertion.group(0).lower()
    habitual = bool(
        re.search(
            r"\b(?:always|regularly|routinely|usually)\b",
            local_prefix[: -len(assertion.group(0))],
            flags=re.IGNORECASE,
        )
    )
    zero_duration = bool(
        re.search(
            r"\bfor\s+(?:(?:an?\s+)?(?:[A-Za-z][A-Za-z-]*\s+){0,3}"
            r"(?:duration|period|total)\s+"
            r"(?:of|totaling|equal\s+to|amounting\s+to)\s+)?"
            r"(?:exactly\s+|precisely\s+)?"
            r"(?:0|nil|no(?!\s+(?:fewer|less)\s+than\b)|none\s+of|zero)\b"
            r"\s+(?:the\s+)?(?:(?:aggregate|calendar|cumulative|elapsed|filing|"
            r"fiscal|taxable|total)\s+){0,2}"
            r"(?:centur(?:y|ies)|days?|decades?|durations?|generations?|hours?|"
            r"minutes?|months?|periods?|quarters?|seconds?|time|weeks?|years?)\b",
            local_prefix[: -len(assertion.group(0))],
            flags=re.IGNORECASE,
        )
    )
    actual_progressive = bool(
        verb.endswith("ing")
        and not zero_duration
        and re.search(
            r"\b(?:"
            r"(?:am|is|are|was|were|will)\s+"
            r"(?:(?:[A-Za-z][A-Za-z-]*ly|already|always|just|now|still|"
            r"in\s+fact|at\s+the\s+time|for\s+(?!(?:0|no|zero)\b)"
            r"[A-Za-z0-9-]+(?:\s+[A-Za-z0-9-]+)*)\s+)*"
            r"(?:be\s+)?|"
            r"(?:has|have|had)\s+"
            r"(?:(?:[A-Za-z][A-Za-z-]*ly|already|always|just|now|still|"
            r"in\s+fact|at\s+the\s+time|for\s+(?!(?:0|no|zero)\b)"
            r"[A-Za-z0-9-]+(?:\s+[A-Za-z0-9-]+)*)\s+)*"
            r"been\s+"
            r"(?:(?:[A-Za-z][A-Za-z-]*ly|already|always|just|now|still|"
            r"in\s+fact|at\s+the\s+time|for\s+(?!(?:0|no|zero)\b)"
            r"[A-Za-z0-9-]+(?:\s+[A-Za-z0-9-]+)*)\s+)*"
            r")$",
            local_prefix[: -len(assertion.group(0))],
            flags=re.IGNORECASE,
        )
    )
    if (
        not _louisiana_supply_verb_is_completed(verb)
        and not habitual
        and not actual_progressive
    ):
        suffix = segment[assertion.end() :]
        if _louisiana_suffix_has_clause_level_condition(suffix):
            return True
    return bool(
        re.search(
            r"\b(?:appears?|appeared|seems?|seemed)\s+to(?:\s+have)?\s+$|"
            r"\b(?:is|was)\s+(?:believed|expected|likely|projected|reported|"
            r"said|scheduled|supposed)\s+to(?:\s+have)?\s+$|"
            r"\b(?:aims?|aimed|attempts?|attempted|claims?|claimed|declines?|"
            r"declined|endeavors?|endeavored|expects?|expected|forgets?|forgot|"
            r"hopes?|hoped|intends?|intended|neglects?|neglected|plans?|planned|"
            r"pretends?|pretended|promises?|promised|seeks?|sought|struggles?|"
            r"struggled|threatens?|threatened|tries?|tried)\s+"
            r"(?:(?:[A-Za-z][A-Za-z-]*ly|in\s+vain|without\s+success)\s+)*"
            r"to(?:\s+have)?\s+$|"
            r"\b(?:(?:am|is|are|was|were|will)\s+"
            r"(?:(?:[A-Za-z][A-Za-z-]*ly)\s+)*(?:be\s+)?|"
            r"(?:has|have|had)\s+(?:(?:[A-Za-z][A-Za-z-]*ly)\s+)*been\s+)"
            r"(?:(?:[A-Za-z][A-Za-z-]*ly)\s+)*(?:aiming|attempting|claiming|declining|"
            r"endeavoring|expecting|forgetting|hoping|intending|neglecting|planning|"
            r"pretending|promising|seeking|struggling|threatening|trying)\s+"
            r"(?:(?:[A-Za-z][A-Za-z-]*ly|in\s+vain|without\s+success)\s+)*to"
            r"(?:\s+have)?\s+$|"
            r"\b(?:has|have|had)\s+(?:aimed|attempted|claimed|declined|endeavored|"
            r"expected|forgotten|hoped|intended|neglected|planned|pretended|"
            r"promised|sought|struggled|threatened|tried)\s+to(?:\s+have)?\s+$|"
            r"\bwould\s+prefer\s+to\s+$|"
            r"\b(?:is|was)\s+(?:about|preparing)\s+to\s+$|"
            r"\b(?:is|was)\s+hardly\s+able\s+to\s+$",
            local_prefix[: -len(assertion.group(0))],
            flags=re.IGNORECASE,
        )
    )


def _louisiana_state_assertion_has_nonoperative_framing(
    segment: str, assertion: re.Match[str]
) -> bool:
    """Keep an entailed availability state across explicit contrast."""

    scope = segment[assertion.start() :]
    if re.search(
        r"\b(?:actually\s+)?(?:available|present|supplied|provided|implemented|"
        r"computed|calculated|determined|encoded|defined|accessible|complete|"
        r"obtainable|retained|included|loaded|imported|stored|furnished|delivered|"
        r"published|returned|produced|generated|exported|held|given|retrievable|"
        r"downloadable|recoverable|operational|usable|ready|extant|installed|"
        r"released|posted|deployed|downloaded|recovered|found|in\s+place|"
        r"on\s+file|at\s+the\s+ready|on\s+hand|at\s+hand|ready\s+for\s+use|"
        r"made\s+(?:available|obtainable)|rendered\s+(?:available|accessible)|"
        r"put\s+on\s+hand)\b[^.;]{0,80}"
        r"(?:(?:,\s*|\(\s*|[—–]\s*)"
        r"(?:(?:although|even\s+though|though|while)\s+)?not\s+"
        r"(?:merely\s+)?(?:an?\s+)?"
        r"(?:hypothetical|proposed)"
        r"(?:\s+[A-Za-z][A-Za-z-]*)?|rather\s+than\s+"
        r"(?:merely\s+)?(?:hypothetical|proposed))\b",
        scope,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(
        r"(?:(?:,\s*|\(\s*|[—–]\s*)"
        r"(?:(?:although|even\s+though|though|while)\s+)?not\s+"
        r"(?:merely\s+)?(?:an?\s+)?"
        r"(?:hypothetical|proposed)(?:\s+[A-Za-z-]+)?"
        r"(?:,\s*|\)\s*|[—–]\s*))[^.;]{0,40}\b"
        r"(?:is|are|was|were|became|becomes?|remains?)\s+"
        r"(?:available|present|complete|accessible|usable|ready|extant|installed|"
        r"released|posted|deployed|in\s+place|on\s+file|on\s+hand|at\s+hand)\b",
        scope,
        flags=re.IGNORECASE,
    ):
        return False
    return _louisiana_text_has_nonoperative_framing(scope)


def _louisiana_normalize_parenthetical_aspect(text: str) -> str:
    """Remove delimiters around bounded aspect and attempt adjuncts."""

    def normalized(match: re.Match[str]) -> str:
        content = match.group("content")
        if re.search(
            r"\b(?:anything\s+but|in\s+no\s+(?:respect|sense)|no\s+longer|"
            r"not(?:\s+(?:at\s+all|ever))?)\s+"
            r"in\s+vain\b",
            content,
            flags=re.IGNORECASE,
        ):
            return " with success "
        if re.search(
            r"\b(?:at\s+no\s+point\s+|by\s+no\s+means\s*,?\s+|"
            r"in\s+no\s+way\s*,?\s+|under\s+no\s+circumstances\s+|"
            r"never\s*,?\s+|no\s+longer\s*,?\s+|"
            r"not\s+(?:at\s+all|[A-Za-z][A-Za-z-]*ly)\s+)"
            r"in\s+fact\b",
            content,
            flags=re.IGNORECASE,
        ):
            return " never "
        if re.search(r"\bno\s+longer\s+in\s+vain\b", content, flags=re.IGNORECASE):
            return " with success "
        if re.search(
            r"\b(?:never\s*,?\s+|no\s+longer\s+)in\s+fact\b",
            content,
            flags=re.IGNORECASE,
        ):
            return " never "
        if re.search(r"\bnever\s+in\s+vain\b", content, flags=re.IGNORECASE):
            return " with success "
        if re.search(r"\bnot\s+without\s+success\b", content, flags=re.IGNORECASE):
            return " with success "
        if re.search(r"\bnot\s+in\s+vain\b", content, flags=re.IGNORECASE):
            return " with success "
        if re.search(r"\bnot\s*,?\s+in\s+fact\b", content, flags=re.IGNORECASE):
            return " not "
        if re.search(r"\bin\s+vain\b", content, flags=re.IGNORECASE):
            return " in vain "
        if re.search(r"\bwithout\s+success\b", content, flags=re.IGNORECASE):
            return " without success "
        if re.search(r"\bin\s+fact\b", content, flags=re.IGNORECASE):
            return " in fact "
        if re.search(r"\bfor\s+", content, flags=re.IGNORECASE):
            return " " + re.sub(r"\s*[:,;—–]\s*", " ", content) + " "
        return f" {content} "

    text = re.sub(
        r"\(\s*without\s+interruption\s*\)",
        " without interruption ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\(\s*(?P<content>[^()]{0,120}(?:in\s+vain|without\s+success|"
        r"in\s+fact|at\s+the\s+time|for\s+[^()]{1,100})[^()]{0,60})\s*\)",
        normalized,
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"(?:,|\(|\[|[—–])\s*(?P<content>[^()\[\]—–]{0,60}"
        r"(?:in\s+vain|without\s+success|in\s+fact|at\s+the\s+time|"
        r"for\s+[^()\[\]—–,]{1,100})[^()\[\]—–]{0,60})\s*"
        r"(?:,|\)|\]|[—–])",
        normalized,
        text,
        flags=re.IGNORECASE,
    )


def _louisiana_suffix_has_clause_level_condition(
    suffix: str, *, include_temporal: bool = True
) -> bool:
    """Recognize a condition on the assertion, not one inside its object NP."""

    temporal = (
        r"when|after|whenever|as\s+soon\s+as|"
        r"once(?!\s+(?:each|every|per)\b)|"
        if include_temporal
        else ""
    )
    condition = re.search(
        rf"\b(?:{temporal}unless|if|provided\s+that|"
        r"so\s+long\s+as|in\s+case|"
        r"(?:on|upon|pending|following)\s+(?:approval|receipt)|"
        r"subject\s+to|dependent\s+on|contingent\s+(?:on|upon)|"
        r"conditioned\s+(?:on|upon)|to\s+the\s+extent|on\s+paper)\b",
        suffix,
        flags=re.IGNORECASE,
    )
    if condition is None:
        return False
    object_prefix = suffix[: condition.start()]
    relative_before_comma = re.search(
        r"\b(?:that|which)\b(?P<body>[^,]*)[,]\s*$",
        object_prefix,
        flags=re.IGNORECASE,
    )
    relative_is_completed = bool(
        relative_before_comma
        and re.search(
            r"(?:\b(?:am|are|is|was|were|be|been|being)\s+"
            r"[A-Za-z][A-Za-z-]*(?:ed|en)|\b[A-Za-z][A-Za-z-]*ed|"
            r"\b(?:began|bought|brought|built|caught|chose|came|did|drew|felt|"
            r"found|gave|got|had|heard|held|kept|knew|left|lost|made|met|paid|"
            r"ran|read|said|saw|sent|set|sold|spoke|taught|told|took|understood|"
            r"went|won|wrote))"
            r"(?:\s+(?:at|by|for|from|in|on|to|under|with)\s+"
            r"(?:(?:a|an|the)\s+)?(?:[A-Za-z][A-Za-z-]*\s+){0,4}"
            r"[A-Za-z][A-Za-z-]*)?"
            r"(?:\s+(?:earlier|last\s+(?:day|month|week|year)|today|yesterday))?\s*$",
            relative_before_comma.group("body"),
            flags=re.IGNORECASE,
        )
    )
    if (
        relative_before_comma
        and not relative_is_completed
        and not re.search(
            r"\b(?:am|are|is|was|were|becomes?|remains?|applies?|uses?|"
            r"requested|completed|selected|specified)\b",
            relative_before_comma.group("body"),
            flags=re.IGNORECASE,
        )
    ):
        return False
    if relative_before_comma and re.search(
        r"^[^.;]{0,80},\s*(?:and|as\s+well\s+as|both|but|or|yet)\s+"
        r"[A-Za-z][A-Za-z-]*\b",
        suffix[condition.end() :],
        flags=re.IGNORECASE,
    ):
        return False
    if relative_before_comma and re.search(
        r"^[^.;]{0,80},\s*(?:after|at|before|by|during|for|in|on|"
        r"pursuant\s+to|"
        r"through|throughout|under|until|using|via|while|with|without)\b",
        suffix[condition.end() :],
        flags=re.IGNORECASE,
    ):
        return False
    if (
        relative_before_comma
        and not relative_is_completed
        and re.search(
            r"^[^.;]{0,80},\s*(?:because|so)\b",
            suffix[condition.end() :],
            flags=re.IGNORECASE,
        )
    ):
        return False
    if re.search(r",\s*$", object_prefix):
        return True
    return not re.search(
        r"\b(?:for\s+use|note\b[^,.;]{0,50}\b(?:saying|stating)|that|which)\b",
        object_prefix,
        flags=re.IGNORECASE,
    )


def _louisiana_pre_supply_antecedent_signature(
    prefix: str,
) -> tuple[str, frozenset[str]] | None:
    """Return an object introduced by a finite predicate before a supply verb."""

    return _louisiana_explicit_transitive_object_signature(
        prefix, terminal=r"(?:before|prior\s+to)"
    )


def _louisiana_coordinated_antecedent_signature(
    segment: str,
) -> tuple[str, frozenset[str]] | None:
    """Track an explicit object introduced in an ordinary coordinated clause."""

    return _louisiana_explicit_transitive_object_signature(segment, terminal=None)


def _louisiana_explicit_transitive_object_signature(
    text: str,
    *,
    terminal: str | None,
) -> tuple[str, frozenset[str]] | None:
    """Extract the direct object of a bounded finite transitive predicate."""

    ending = rf"\s+{terminal}\s*$" if terminal is not None else r"\s*$"
    antecedent = re.search(
        r"\b(?:(?:can|could|may|might|must|shall|should|will|would)\s+)?"
        r"[A-Za-z][A-Za-z-]*\s+"
        r"(?P<object>(?:a|an|the|this|that)\s+[^,.;]{1,60}?)" + ending,
        text,
        flags=re.IGNORECASE,
    )
    if antecedent is None:
        antecedent = re.search(
            r"\b(?:(?:can|could|may|might|must|shall|should|will|would)\s+"
            r"[A-Za-z][A-Za-z-]*|[A-Za-z][A-Za-z-]*(?:s|ed)|brought|built|"
            r"bought|caught|chose|drew|found|gave|got|heard|held|hid|kept|laid|"
            r"left|lost|made|met|paid|put|ran|read|saw|sent|set|sold|taught|told|"
            r"took|won|wrote)\s+"
            r"(?P<object>(?:[A-Za-z][A-Za-z-]*\s+){0,5}[A-Za-z][A-Za-z-]*?)" + ending,
            text,
            flags=re.IGNORECASE,
        )
    if antecedent is None:
        return None
    object_text = re.sub(
        r"\s+(?:aside|later)\s*$", "", antecedent.group("object"), flags=re.IGNORECASE
    )
    object_text = re.split(
        r"\b(?:containing|covering|describing|displaying|including|with)\b",
        object_text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return _louisiana_nearest_object_signature(object_text, allow_opaque=True)


def _louisiana_assertion_is_negative(segment: str, assertion_start: int) -> bool:
    """Return the scoped polarity of one supply assertion."""

    prefix = re.split(r"[,.;]", segment[:assertion_start])[-1]
    prefix = re.sub(r"\bwithout\s+delay\b", " ", prefix, flags=re.IGNORECASE)
    prefix = re.sub(
        r"\bno\s+(?:fewer|less)\s+than\b", "at least", prefix, flags=re.IGNORECASE
    )
    prefix = re.sub(
        r"\b(?:no\s+interruptions?|none\s+of\s+(?:the\s+)?"
        r"(?:[A-Za-z][A-Za-z-]*\s+){0,3}interruptions?)\b",
        " ",
        prefix,
        flags=re.IGNORECASE,
    )
    if re.search(r"\b(?:before|prior\s+to)\s*$", prefix, flags=re.IGNORECASE):
        return False
    if re.search(
        r"\bnot\s+(?:surprisingly|unexpectedly)\s*$",
        prefix,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\b(?:may|might|should|could|would)\s+"
        r"(?:(?:[A-Za-z][A-Za-z-]*ly)\s+)*$",
        prefix,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:unable|incapable)\b[^,.;]{0,40}\bnot\s+(?:to\s+)?$|"
        r"\b(?:fails?|refuses?|unable|incapable|hesitates?)\b[^,.;]{0,40}\b"
        r"(?:avoid|stop|refrain)\w*\s+(?:from\s+|to\s+)?$",
        prefix,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\b(?:can|does?)\s+(?:do\s+)?nothing\s+(?:but|except)\s*$|"
        r"\bcannot\s+do\s+otherwise\s+than\s*$|"
        r"\b(?:is|are|was|were)\s+powerless\s+to\s+avoid\s*$|"
        r"\b(?:has|have|had|is|are|was|were)\b[^,.;]{0,40}"
        r"\bno\s+(?:alternative|option)\s+(?:except\s+to|other\s+than)\s*$|"
        r"\b(?:has|have|had)\s+little\s+choice\s+other\s+than\s*$",
        prefix,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\b(?:cannot|can['’]t)\s+help\s*$|"
        r"\b(?:has|have|had)\s+no\s+(?:difficulty|trouble)\s*$|"
        r"\bno\s+doubt\s*$",
        prefix,
        flags=re.IGNORECASE,
    ):
        return False

    negative_complement = re.search(
        r"\b(?:lacks?\s+(?:the\s+)?ability|fails?|refuses?|avoids?|delays?|"
        r"hesitates?|ceases?|stops?|refrains?|unable|incapable)\b"
        r"(?P<tail>[^,.;]{0,80}?)(?:to|before)?\s*$",
        prefix,
        flags=re.IGNORECASE,
    )
    if negative_complement:
        if re.search(
            r"\b(?:anything\s+but|far\s+from)\s*$",
            prefix[: negative_complement.start()],
            flags=re.IGNORECASE,
        ):
            return False
        reversals = tuple(
            re.finditer(
                r"\b(?:cannot|never|nothing|neither|no|not)\b|n['’]t\b",
                prefix[: negative_complement.start()],
                flags=re.IGNORECASE,
            )
        )
        if not reversals:
            return True
        if reversals[-1].group(0).lower() == "not" and re.search(
            r"\b(?:may|might)\b[^,.;]*$",
            prefix[: reversals[-1].start()],
            flags=re.IGNORECASE,
        ):
            return True
        between = prefix[reversals[-1].end() : negative_complement.start()]
        if re.search(
            r"\b(?:necessarily|always|ordinarily|inevitably|consistently|reliably)\b",
            between,
            flags=re.IGNORECASE,
        ):
            return True
        return False

    direct_negators = tuple(
        re.finditer(
            r"\b(?:cannot|never|nothing|neither|not|no(?!\s+later\b))\b|n['’]t\b",
            prefix,
            flags=re.IGNORECASE,
        )
    )
    return len(direct_negators) % 2 == 1


def _louisiana_assertion_object_corefers(
    object_text: str,
    *,
    missing_head: str,
    missing_object: str,
) -> bool:
    """Match a direct assertion object to the missing grammatical object."""

    normalized = object_text.strip().lower()
    if re.match(
        r"(?:(?:it|them)\b|both\s*$|"
        r"(?:this|that(?:\s+same)?|those|the\s+same|same)\s*$)",
        normalized,
    ):
        if _louisiana_text_has_nonoperative_framing(normalized) or re.search(
            r"\b(?:if|conditionally)\b|\bin\s+theory\b|"
            r"\bnot\s+in\s+practice\b|\baccording\s+to\b|"
            r"\bmerely\s+as\b|\b(?:only\s+)?in\s+the\s+event\b|"
            r"\bonly\s+upon\b|\bsubject\s+to\b|\bassuming\b|"
            r"\bon\s+(?:the\s+)?condition(?:\s+that)?\b|\bcontingently\b|"
            r"\bwere\b[^.;]{0,40}\bsupplied\b|\bonly\s+after\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return False
        return not re.search(
            r"\b(?:different|draft|excerpt|fraction|fragment|incomplete|outline|"
            r"part|partial|piece|portion|provisional|separate|subset|unrelated)\b|"
            r"\b(?:barely|hardly|scarcely)\s+any\b|\balmost\s+no\b|"
            r"\b(?:less\s+than\s+all|most\s+but\s+not\s+all)\b",
            normalized,
        )
    if re.search(
        r"\b(?:different|draft|excerpt|fraction|fragment|incomplete|outline|part|"
        r"partial|piece|portion|provisional|separate|subset|unrelated)\b|"
        r"\b(?:barely|hardly|scarcely)\s+any\b|\balmost\s+no\b|"
        r"\b(?:less\s+than\s+all|most\s+but\s+not\s+all)\b|"
        r"\b(?:a\s+few|merely)\s+(?:fragments?|pieces?|portions?)\b",
        normalized,
    ):
        return False
    object_signature = _louisiana_object_signature(normalized)
    missing_signature = _louisiana_object_signature(missing_object)
    if object_signature is None or missing_signature is None:
        return False
    object_family, object_modifiers = object_signature
    missing_family, missing_modifiers = missing_signature
    expected_family = _louisiana_object_family(missing_head)
    if object_family != missing_family or missing_family != expected_family:
        return False
    return object_modifiers == missing_modifiers


def _louisiana_object_number(text: str) -> str | None:
    """Infer the grammatical number of the nearest explicit object head."""

    outer_np = re.split(
        r"\b(?:accompanied\s+by|containing|of|that|which|with)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    determiners = tuple(
        re.finditer(
            r"\b(?P<determiner>a|an|one|each|every|this|that|these|those|"
            r"many|several|two|three|four|five|six|seven|eight|nine|ten)\s+",
            outer_np,
            flags=re.IGNORECASE,
        )
    )
    if determiners:
        determiner = determiners[0].group("determiner").lower()
        if determiner in {"a", "an", "one", "each", "every", "this", "that"}:
            return "singular"
        return "plural"
    tokens = re.findall(r"[A-Za-z][A-Za-z-]*", outer_np.lower())
    while tokens and tokens[-1] in {
        "aside",
        "later",
        "same",
        "successfully",
    }:
        tokens.pop()
    if not tokens:
        return None
    head = tokens[-1]
    if head in {"them", "those", "these"}:
        return "plural"
    if head in {"it", "this", "that"}:
        return "singular"
    if head in {"children", "data", "men", "people", "women"}:
        return "plural"
    if head in {"analysis", "basis", "means", "process", "series", "status"}:
        return "singular"
    if head.endswith("s") and not head.endswith(("ss", "us", "is")):
        return "plural"
    return "singular"


def _louisiana_demonstrative_antecedent_signature(
    text: str,
    *,
    last_explicit_signature: tuple[str, frozenset[str]] | None,
    numbered_signatures: dict[str, tuple[str, frozenset[str]]],
) -> tuple[str, frozenset[str]] | None:
    """Resolve a demonstrative to the nearest antecedent of matching number."""

    pronoun = re.match(
        r"\s*(?P<pronoun>this|that|those|it|them|both|the\s+same|same)\b",
        text,
        flags=re.IGNORECASE,
    )
    if pronoun is None:
        return last_explicit_signature
    normalized = pronoun.group("pronoun").lower()
    number = (
        "plural"
        if normalized in {"those", "them", "both"}
        else "singular"
        if normalized in {"this", "that", "it"}
        else None
    )
    if number is not None and number in numbered_signatures:
        return numbered_signatures[number]
    return last_explicit_signature


def _louisiana_object_family(head: str) -> str:
    """Normalize synonymous executable object heads."""

    signature = _louisiana_object_signature(head)
    return signature[0] if signature is not None else head


def _louisiana_object_signature(text: str) -> tuple[str, frozenset[str]] | None:
    """Return an executable synonym family and its meaningful modifiers."""

    text = re.split(
        r"\bwith\s+(?:a|an|the)\s+(?:annotation|comment|description|note|report)\b|"
        r"\baccompanied\s+by\s+(?:a|an|the)\s+"
        r"(?:annotation|comment|description|note|report)\b|"
        r"(?<=\w)\s+(?:that|which)\b|\bfor\s+use\s+if\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    text = re.sub(r"\btwice\s+per\s+year\b", "semiannual", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:each|every)\s+two\s+weeks\b", "biweekly", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\b(?:(?:for\s+)?(?:each|every)|per)\s+year\b|\bper\s+annum\b",
        "annual",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:(?:for\s+)?(?:each|every)|per)\s+month\b",
        "monthly",
        text,
        flags=re.IGNORECASE,
    )
    for period, normalized in (
        ("quarter", "quarterly"),
        ("week", "weekly"),
        ("day", "daily"),
    ):
        text = re.sub(
            rf"\b(?:(?:for\s+)?(?:each|every)|per)\s+{period}\b",
            normalized,
            text,
            flags=re.IGNORECASE,
        )
    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("’s", "").replace("'s", ""))
    ignored = {
        "a",
        "an",
        "available",
        "complete",
        "completely",
        "directly",
        "delay",
        "fully",
        "hand",
        "immediately",
        "on",
        "only",
        "same",
        "successfully",
        "with",
        "that",
        "the",
        "this",
        "those",
    }
    tokens = [token for token in tokens if token not in ignored]
    if "resident" in tokens and "taxpayer" in tokens:
        tokens.remove("taxpayer")
    if "individual" in tokens and "income" in tokens:
        tokens.remove("income")
    modifier_normalization = {
        "corporation": "corporate",
        "corporations": "corporate",
        "criteria": "criterion",
        "data": "datum",
        "individuals": "individual",
        "annually": "annual",
        "media": "medium",
        "residents": "resident",
        "spousal": "spouse",
        "taxpayers": "taxpayer",
        "yearly": "annual",
    }
    tokens = [modifier_normalization.get(token, token) for token in tokens]
    family_members = {
        "algorithm": "algorithm",
        "algorithms": "algorithm",
        "amount": "amount",
        "amounts": "amount",
        "base": "base",
        "bases": "base",
        "calculation": "calculation",
        "calculations": "calculation",
        "capability": "capability",
        "capabilities": "capability",
        "classification": "classification",
        "classifications": "classification",
        "computation": "calculation",
        "computations": "calculation",
        "condition": "condition",
        "conditions": "condition",
        "credit": "credit",
        "credits": "credit",
        "data": "data",
        "datum": "data",
        "deduction": "deduction",
        "deductions": "deduction",
        "definition": "definition",
        "definitions": "definition",
        "determination": "determination",
        "determinations": "determination",
        "document": "document",
        "documents": "document",
        "equation": "formula",
        "equations": "formula",
        "export": "output",
        "exports": "output",
        "fact": "fact",
        "facts": "fact",
        "formula": "formula",
        "formulas": "formula",
        "implementation": "implementation",
        "implementations": "implementation",
        "input": "input",
        "inputs": "input",
        "instruction": "instruction",
        "instructions": "instruction",
        "liability": "liability",
        "liabilities": "liability",
        "logic": "logic",
        "lookup": "lookup",
        "lookups": "lookup",
        "mapping": "mapping",
        "mappings": "mapping",
        "matrix": "matrix",
        "matrices": "matrix",
        "mechanics": "mechanics",
        "mechanism": "mechanism",
        "mechanisms": "mechanism",
        "method": "method",
        "methods": "method",
        "output": "output",
        "outputs": "output",
        "parameter": "parameter",
        "parameters": "parameter",
        "percentage": "percentage",
        "percentages": "percentage",
        "procedure": "procedure",
        "procedures": "procedure",
        "process": "process",
        "processes": "process",
        "protocol": "protocol",
        "protocols": "protocol",
        "rate": "rate",
        "rates": "rate",
        "record": "record",
        "records": "record",
        "requirement": "requirement",
        "requirements": "requirement",
        "rule": "rule",
        "rules": "rule",
        "schedule": "rate",
        "schedules": "rate",
        "sequence": "sequence",
        "sequences": "sequence",
        "specification": "specification",
        "specifications": "specification",
        "standard": "standard",
        "standards": "standard",
        "status": "status",
        "statuses": "status",
        "step": "step",
        "steps": "step",
        "structure": "structure",
        "structures": "structure",
        "table": "rate",
        "tables": "rate",
        "tax": "tax",
        "taxes": "tax",
        "threshold": "threshold",
        "thresholds": "threshold",
        "workflow": "workflow",
        "workflows": "workflow",
    }
    family_positions = [
        (index, family_members[token])
        for index, token in enumerate(tokens)
        if token in family_members
    ]
    if not family_positions:
        return None
    relation_positions = [
        index for index, token in enumerate(tokens) if token in {"for", "of"}
    ]
    if relation_positions:
        before_relation = [
            item for item in family_positions if item[0] < relation_positions[0]
        ]
        head_position, family = (before_relation or family_positions)[-1]
    else:
        head_position, family = family_positions[-1]
    modifier_sequence = tuple(
        _louisiana_normalize_modifier_token(token)
        for index, token in enumerate(tokens)
        if index != head_position
        and token not in {"for", "of", "tax"}
        and (token not in family_members or family_members[token] != family)
    )
    modifier_tokens = set(modifier_sequence)
    relational_modifiers = {
        "appeal",
        "beneficiary",
        "borrower",
        "buyer",
        "creditor",
        "debtor",
        "denial",
        "destination",
        "donee",
        "donor",
        "employee",
        "employer",
        "executor",
        "grantee",
        "grantor",
        "guardian",
        "heir",
        "landlord",
        "lender",
        "lessee",
        "lessor",
        "licensee",
        "licensor",
        "owner",
        "origin",
        "parent",
        "partner",
        "partnership",
        "payer",
        "principal",
        "recipient",
        "renter",
        "representative",
        "seller",
        "subsidiary",
        "taxpayer",
        "tenant",
        "testator",
        "trustee",
        "ward",
        "trust",
        "decedent",
    }

    ordered_sequence = tuple(
        token
        for token in modifier_sequence
        if not re.fullmatch(r"(?:19|20)\d{2}", token)
    )
    inverse_role_stems: dict[str, set[str]] = {}
    for token in ordered_sequence:
        suffix = next(
            (ending for ending in ("ee", "or", "er", "ed") if token.endswith(ending)),
            None,
        )
        if suffix is not None and len(token) > len(suffix) + 2:
            inverse_role_stems.setdefault(token[: -len(suffix)], set()).add(suffix)
    inverse_stems = {
        stem
        for stem, suffixes in inverse_role_stems.items()
        if bool(suffixes & {"ee", "ed"}) and bool(suffixes & {"or", "er"})
    }
    relational_sequence = tuple(
        token for token in ordered_sequence if token in relational_modifiers
    )
    person_descriptor_sequence = tuple(
        token
        for token in ordered_sequence
        if token in relational_modifiers or token in {"dependent", "student", "widow"}
    )
    inverse_sequence = tuple(
        token
        for token in ordered_sequence
        if any(
            token.startswith(stem) and token[len(stem) :] in {"ee", "or", "er", "ed"}
            for stem in inverse_stems
        )
    )
    order_sensitive = set(inverse_sequence)
    if relational_sequence and len(person_descriptor_sequence) >= 2:
        order_sensitive.update(person_descriptor_sequence)
    ordered_roles = tuple(
        token for token in ordered_sequence if token in order_sensitive
    )
    if len(ordered_roles) >= 2:
        modifier_tokens.add("order:" + ">".join(ordered_roles))
    return family, frozenset(modifier_tokens)


def _louisiana_normalize_modifier_token(token: str) -> str:
    """Normalize ordinary singular/plural variation in object modifiers."""

    irregular = {
        "analyses": "analysis",
        "bonuses": "bonus",
        "children": "child",
        "leaves": "leaf",
        "matrices": "matrix",
        "means": "means",
        "monies": "money",
        "people": "person",
        "men": "man",
        "wives": "wife",
        "women": "woman",
    }
    if token in irregular:
        return irregular[token]
    if token.endswith("statuses"):
        return token[:-2]
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith(("sses", "shes", "ches", "xes", "zes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _louisiana_nearest_object_signature(
    text: str,
    *,
    allow_opaque: bool = False,
) -> tuple[str, frozenset[str]] | None:
    """Return the nearest executable or documentary antecedent signature."""

    executable = _louisiana_object_signature(text)
    documentary_heads = tuple(_LOUISIANA_DOCUMENTARY_HEAD.finditer(text))
    if not documentary_heads:
        if executable is not None or not allow_opaque:
            return executable
        opaque_tokens = re.findall(r"[A-Za-z][A-Za-z-]*", text.lower())
        if not opaque_tokens:
            return None
        return (
            f"other:{_louisiana_normalize_modifier_token(opaque_tokens[-1])}",
            frozenset(),
        )
    executable_heads = tuple(
        match
        for match in re.finditer(r"\b[A-Za-z]+\b", text)
        if _louisiana_object_signature(match.group(0)) is not None
    )
    if (
        executable_heads
        and executable_heads[-1].start() > documentary_heads[-1].start()
    ):
        return executable
    return (f"other:{documentary_heads[-1].group(0).lower()}", frozenset())


def _louisiana_signatures_corefer(
    first: tuple[str, frozenset[str]] | None,
    second: tuple[str, frozenset[str]] | None,
) -> bool:
    """Compare executable identity without discarding meaningful modifiers."""

    return first is not None and second is not None and first == second


def _missing_scope_semantic_head(scope: str) -> str | None:
    """Return the final executable semantic head of a direct missing object."""

    signature = _louisiana_object_signature(scope)
    return signature[0] if signature is not None else None


def _reason_dependency_occurrence_is_contextual(
    reason: str,
    match: re.Match[str],
    *,
    current_usc_title: str | None,
) -> bool:
    """Reject context attached to this citation without crossing into another one."""

    groups = match.groupdict()
    title = groups.get("title")
    section = groups.get("section")
    if title and section:
        dependencies = _reason_dependencies(reason)
        candidates = (
            *_qualified_usc_dependencies(reason),
            *_RELATIVE_USC_DEFERRAL_DEPENDENCY.finditer(reason),
            *_qualified_louisiana_rs_dependencies(reason),
        )
        for candidate in candidates:
            if _usc_dependency_occurrences_match(
                reason,
                candidate,
                match,
                current_usc_title=current_usc_title,
            ) and (
                _CONTEXTUAL_AUTHORITY_LANGUAGE.search(
                    _reason_dependency_local_context(
                        reason,
                        candidate,
                        dependencies=dependencies,
                    )
                )
            ):
                return True

    remainder = reason[match.end() :]
    stop = min(
        (
            position
            for separator in (".", ";", "\n")
            if (position := remainder.find(separator)) >= 0
        ),
        default=-1,
    )
    if stop < 0:
        return False
    trailing_clause = re.split(r"[.;\n]", remainder[stop + 1 :], maxsplit=1)[0]
    return bool(
        not _reason_dependencies(trailing_clause)
        and _has_adversative_language(trailing_clause)
    )


def _reason_dependencies(reason: str) -> list[re.Match[str]]:
    return sorted(
        (
            *_qualified_usc_dependencies(reason),
            *_RELATIVE_USC_DEFERRAL_DEPENDENCY.finditer(reason),
            *_qualified_louisiana_rs_dependencies(reason),
            *_PRECISE_DEFERRAL_DEPENDENCY.finditer(reason),
        ),
        key=lambda dependency: (dependency.start(), -dependency.end()),
    )


def _reason_dependency_local_context(
    reason: str,
    match: re.Match[str],
    *,
    dependencies: list[re.Match[str]],
) -> str:
    clause_start, clause_end = _reason_clause_bounds(reason, match)
    previous_end = max(
        (
            dependency.end()
            for dependency in dependencies
            if clause_start <= dependency.start() and dependency.end() <= match.start()
        ),
        default=clause_start,
    )
    next_start = min(
        (
            dependency.start()
            for dependency in dependencies
            if match.end() <= dependency.start() < clause_end
        ),
        default=clause_end,
    )
    before = reason[previous_end : match.start()]
    after = reason[match.end() : next_start]
    preceding_coordinations = list(_DEPENDENCY_CONTEXT_COORDINATION.finditer(before))
    if preceding_coordinations:
        before = before[preceding_coordinations[-1].end() :]
    following_coordination = _DEPENDENCY_CONTEXT_COORDINATION.search(after)
    if following_coordination:
        after = after[: following_coordination.start()]
    return before + after


def _dependency_citation_family(match: re.Match[str]) -> str:
    if match.re is _LOUISIANA_RS_DEFERRAL_DEPENDENCY:
        return "louisiana-rs"
    return "usc"


def _usc_dependencies_match(left: re.Match[str], right: re.Match[str]) -> bool:
    left_groups = left.groupdict()
    right_groups = right.groupdict()
    return bool(
        _dependency_citation_family(left) == _dependency_citation_family(right)
        and left_groups.get("title")
        and right_groups.get("title")
        and left_groups["title"].lower() == right_groups["title"].lower()
        and normalize_rulespec_path_segment(left_groups["section"])
        == normalize_rulespec_path_segment(right_groups["section"])
        and _usc_dependency_fragments(left) == _usc_dependency_fragments(right)
    )


def _usc_dependency_occurrences_match(
    reason: str,
    left: re.Match[str],
    right: re.Match[str],
    *,
    current_usc_title: str | None,
) -> bool:
    if _usc_dependencies_match(left, right):
        return True
    left_groups = left.groupdict()
    right_groups = right.groupdict()
    if (
        left_groups.get("title")
        or not right_groups.get("title")
        or not current_usc_title
        or right_groups["title"].lower() != current_usc_title.lower()
    ):
        return False
    if not re.match(
        r"\s+of\s+this\s+title\b",
        reason[left.end() :],
        flags=re.IGNORECASE,
    ):
        return False
    return bool(
        normalize_rulespec_path_segment(left_groups["section"])
        == normalize_rulespec_path_segment(right_groups["section"])
        and _usc_dependency_fragments(left) == _usc_dependency_fragments(right)
    )


def _reason_clause_bounds(
    reason: str,
    match: re.Match[str],
) -> tuple[int, int]:
    masked = list(reason)
    for dependency in (
        *_qualified_usc_dependencies(reason),
        *_qualified_louisiana_rs_dependencies(reason),
    ):
        for position in range(dependency.start(), dependency.end()):
            if masked[position] in {".", ";"}:
                masked[position] = " "
    masked_reason = "".join(masked)
    preceding_stops = [
        position
        for separator in (".", ";", "\n")
        if (position := masked_reason.rfind(separator, 0, match.start())) >= 0
    ]
    following_stops = [
        position
        for separator in (".", ";", "\n")
        if (position := masked_reason.find(separator, match.end())) >= 0
    ]
    return (
        max(preceding_stops, default=-1) + 1,
        min(following_stops, default=len(reason)),
    )


def _reason_suffix_has_dependency_state(
    after: str,
    *,
    allow_descriptive_list: bool,
    allow_executable_object: bool = False,
) -> bool:
    """Require a state predicate for this citation or its coordinated list."""

    if _has_adversative_language(after):
        return False
    bounded = after
    state_pattern = re.compile(
        r"\b(?:"
        r"(?:is|are|was|were|must\s+be)\s+(?:not\s+(?:yet\s+)?)?"
        + _DEPENDENCY_STATE_VALUE
        + r"|(?:has|have)\s+(?:not\s+)?been\s+"
        + _DEPENDENCY_STATE_VALUE
        + r")\b",
        flags=re.IGNORECASE,
    )
    for state in state_pattern.finditer(bounded):
        prefix = bounded[: state.start()]
        tail = bounded[state.end() :]
        if not _reason_state_tail_is_bounded(tail):
            continue
        if not prefix.strip():
            return True
        masked_prefix = list(prefix)
        dependencies = sorted(
            (
                *_qualified_usc_dependencies(prefix),
                *_PRECISE_DEFERRAL_DEPENDENCY.finditer(prefix),
            ),
            key=lambda dependency: (dependency.start(), -dependency.end()),
        )
        for dependency in dependencies:
            masked_prefix[dependency.start() : dependency.end()] = " " * (
                dependency.end() - dependency.start()
            )
        remainder = "".join(masked_prefix)
        if re.fullmatch(
            r"\s*(?:(?:,|\b(?:and|or|both|either|neither)\b)\s*)*",
            remainder,
            flags=re.IGNORECASE,
        ):
            return True
        if (
            allow_executable_object
            and len(prefix) <= 240
            and re.fullmatch(
                r"(?:is|are)\s+"
                + _EXECUTABLE_DEPENDENCY_READY_STATE_VALUE
                + r"|(?:has|have)\s+been\s+"
                + _EXECUTABLE_DEPENDENCY_READY_STATE_VALUE,
                state.group(0),
                flags=re.IGNORECASE,
            )
            and _missing_scope_starts_with_executable_object(prefix)
            and not _executable_object_scope_claims_ready(prefix)
            and not _missing_scope_reverses_insufficiency(prefix)
        ):
            return True
        if allow_descriptive_list and _reason_descriptive_dependency_list_is_bounded(
            prefix
        ):
            return True
    return False


def _reason_state_tail_is_bounded(tail: str) -> bool:
    if re.fullmatch(
        r"[\s,)]*(?:by\s+(?:the\s+)?"
        r"(?:(?:a|an|the)\s+)?"
        r"(?:(?:administering|federal|local|public\s+housing|state)\s+)?"
        r"(?:agency|administrator|authority|commission|"
        r"commissioner(?:\s+of\s+(?:internal\s+revenue|social\s+security))?|hud|irs|"
        r"internal\s+revenue\s+service|social\s+security\s+administration|"
        r"(?:department|secretary)(?:\s+of\s+(?:the\s+)?(?:agriculture|housing\s+and\s+"
        r"urban\s+development|education|health\s+and\s+human\s+services|labor|"
        r"treasury|veterans\s+affairs))?))?"
        r"[\s,)]*",
        tail,
        flags=re.IGNORECASE,
    ):
        return True
    coordination = re.match(
        r"\s*,?\s+(?:and|or)\s+",
        tail,
        flags=re.IGNORECASE,
    )
    if not coordination:
        return False
    continuation = tail[coordination.end() :]
    dependencies = sorted(
        (
            *_qualified_usc_dependencies(continuation),
            *_PRECISE_DEFERRAL_DEPENDENCY.finditer(continuation),
        ),
        key=lambda dependency: (dependency.start(), -dependency.end()),
    )
    dependency = dependencies[0] if dependencies else None
    if dependency is None:
        return False
    introduction = continuation[: dependency.start()]
    if not _reason_dependency_introduction_is_bounded(introduction):
        return False
    return _reason_suffix_has_dependency_state(
        continuation[dependency.end() :],
        allow_descriptive_list=False,
    )


def _reason_descriptive_dependency_list_is_bounded(prefix: str) -> bool:
    if not re.match(r"\s*,?\s+(?:and|or)\s+", prefix, flags=re.IGNORECASE):
        return False
    masked_prefix = list(prefix)
    for dependency in _qualified_usc_dependencies(prefix):
        for position in range(dependency.start(), dependency.end()):
            if masked_prefix[position] == ",":
                masked_prefix[position] = " "
    items = [
        re.sub(r"^\s*(?:and|or)\s+", "", item, flags=re.IGNORECASE).strip()
        for item in "".join(masked_prefix).split(",")
        if item.strip()
    ]
    if not items:
        return False
    for item in items:
        dependencies = sorted(
            (
                *_qualified_usc_dependencies(item),
                *_PRECISE_DEFERRAL_DEPENDENCY.finditer(item),
            ),
            key=lambda dependency: (dependency.end(), -dependency.start()),
            reverse=True,
        )
        if not dependencies:
            return False
        dependency = dependencies[0]
        if item[dependency.end() :].strip():
            return False
        introduction = item[: dependency.start()]
        if not _reason_dependency_introduction_is_bounded(introduction):
            return False
    return True


def _reason_reference_introduction_is_bounded(bridge: str) -> bool:
    if not bridge.strip() or _reason_dependency_introduction_is_bounded(bridge):
        return True
    masked = list(bridge)
    dependencies = (
        *_qualified_usc_dependencies(bridge),
        *_PRECISE_DEFERRAL_DEPENDENCY.finditer(bridge),
    )
    if not dependencies:
        return False
    for dependency in dependencies:
        masked[dependency.start() : dependency.end()] = " " * (
            dependency.end() - dependency.start()
        )
    remainder = "".join(masked)
    if re.fullmatch(
        r"\s*(?:(?:,|\b(?:and|or|both|either|neither)\b)\s*)*",
        remainder,
        flags=re.IGNORECASE,
    ):
        return True
    return bool(
        re.fullmatch(
            r"\s*(?:(?:is|are|was|were|must\s+be)\s+"
            + _DEPENDENCY_STATE_VALUE
            + r"|(?:has|have)\s+(?:not\s+)?been\s+"
            + _DEPENDENCY_STATE_VALUE
            + r")\s*,?\s*(?:and|or)\s*",
            remainder,
            flags=re.IGNORECASE,
        )
    )


def _reason_dependency_introduction_is_bounded(introduction: str) -> bool:
    if not introduction.strip():
        return True
    strong_linker = re.search(
        r"\b(?:depends?\s+on|requires?)\s*$",
        introduction,
        flags=re.IGNORECASE,
    )
    if strong_linker:
        return _dependency_subject_phrase_is_bounded(
            introduction[: strong_linker.start()]
        )
    linker = re.search(
        r"\b(?:"
        r"under|pursuant\s+to|according\s+to|"
        r"(?:cited|defined|described|provided|required|set|specified|referenced)"
        r"\s+(?:by|in)"
        r")\s*$",
        introduction,
        flags=re.IGNORECASE,
    )
    if not linker:
        return False
    subject_scope = introduction[: linker.start()]
    if _dependency_subject_phrase_is_bounded(subject_scope):
        return True
    chain = re.fullmatch(
        r"(?P<subject>.+?)\s+under\s+(?:the\s+)?"
        r"(?P<instrument>(?:[a-z0-9-]+\s+)*(?:act|code|program|regulations?|statute))\s*",
        subject_scope,
        flags=re.IGNORECASE,
    )
    return bool(
        chain
        and _dependency_subject_phrase_is_bounded(chain.group("subject"))
        and _legal_instrument_phrase_is_bounded(chain.group("instrument"))
        and re.match(
            r"(?:cited|defined|described|provided|required|set|specified|referenced)"
            r"\s+(?:by|in)",
            introduction[linker.start() :],
            flags=re.IGNORECASE,
        )
    )


def _dependency_subject_phrase_is_bounded(phrase: str) -> bool:
    tokens = []
    for raw_token in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", phrase):
        token = raw_token.lower()
        if token not in _DEPENDENCY_MODIFIER_TERMS:
            candidates = []
            if token.endswith("ies"):
                candidates.append(f"{token[:-3]}y")
            if token.endswith("es"):
                candidates.append(token[:-2])
            if token.endswith("s"):
                candidates.append(token[:-1])
            token = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate in _DEPENDENCY_MODIFIER_TERMS
                ),
                token,
            )
        tokens.append(token)
    return bool(
        tokens
        and tokens[-1] in _DEPENDENCY_SUBJECT_TERMS
        and all(token in _DEPENDENCY_MODIFIER_TERMS for token in tokens)
    )


def _legal_instrument_phrase_is_bounded(phrase: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*|\d+", phrase)
    if not tokens or tokens[-1].lower() not in {
        "act",
        "code",
        "program",
        "regulation",
        "regulations",
        "statute",
    }:
        return False
    connectors = {"and", "for", "of", "the", "to"}
    return all(
        token.isdigit()
        or token.lower() in connectors
        or token.lower() in _LEGAL_INSTRUMENT_TERMS
        for token in tokens[:-1]
    )


def _reason_named_instruments_are_source_bound(
    before: str,
    source_scope_text: str,
    *,
    dependency_match: re.Match[str],
) -> bool:
    named_instrument = re.compile(
        r"\bunder\s+(?:the\s+)?"
        r"(?P<instrument>(?:[a-z0-9-]+\s+)*(?:act|code|program|regulations?|statute))"
        r"\s+(?:cited|defined|described|provided|required|set|specified|referenced)"
        r"\s+(?:by|in)\s*$",
        flags=re.IGNORECASE,
    )
    match = named_instrument.search(before)
    if match is None:
        return True
    instrument = re.sub(r"[^a-z0-9]+", " ", match.group("instrument").lower()).strip()
    if not instrument:
        return False

    match_groups = dependency_match.groupdict()
    dependency_title = match_groups.get("title")
    dependency_section = match_groups.get("section")
    if dependency_title and dependency_section:
        dependency_fragments = _usc_dependency_fragments(dependency_match)
        source_matches = (
            *_qualified_usc_dependencies(source_scope_text),
            *_RELATIVE_USC_DEFERRAL_DEPENDENCY.finditer(source_scope_text),
        )
        instrument_pattern = re.compile(
            r"(?<![a-z0-9])"
            + r"[^a-z0-9]+".join(map(re.escape, instrument.split()))
            + r"(?![a-z0-9])",
            flags=re.IGNORECASE,
        )
        instrument_link = re.compile(
            r"\b(?:authorizes?|defines?|establishes?|governs?|provides?|"
            r"regulates?|requires?)\b[^.;\n]{0,200}\b"
            r"(?:according\s+to|pursuant\s+to|under)\s*$",
            flags=re.IGNORECASE,
        )
        for source_match in source_matches:
            source_title = source_match.groupdict().get("title")
            if (
                (source_title and source_title.lower() != dependency_title.lower())
                or normalize_rulespec_path_segment(source_match.group("section"))
                != normalize_rulespec_path_segment(dependency_section)
                or _usc_dependency_fragments(source_match) != dependency_fragments
            ):
                continue
            clause_start, clause_end = _reason_clause_bounds(
                source_scope_text,
                source_match,
            )
            clause = source_scope_text[clause_start:clause_end]
            reference_start = source_match.start() - clause_start
            reference_end = source_match.end() - clause_start
            if not _source_clause_links_dependency(
                clause,
                reference_start=reference_start,
                reference_end=reference_end,
            ):
                continue
            for title_match in instrument_pattern.finditer(clause, 0, reference_start):
                bridge = clause[title_match.end() : reference_start]
                if (
                    not _has_adversative_language(bridge)
                    and not _bridge_crosses_coordinated_finite_clause(bridge)
                    and instrument_link.search(bridge)
                ):
                    return True
        return False

    normalized_source = re.sub(r"[^a-z0-9]+", " ", source_scope_text.lower()).strip()
    return bool(
        re.search(
            rf"(?:^|\s){re.escape(instrument)}(?:$|\s)",
            normalized_source,
        )
    )


def _bridge_crosses_coordinated_finite_clause(bridge: str) -> bool:
    if _COORDINATED_FINITE_CLAUSE.search(bridge):
        return True
    for coordination in re.finditer(
        r"(?:,\s*)?\b(?:and|but|or|yet)\s+",
        bridge,
        flags=re.IGNORECASE,
    ):
        coordinated = bridge[coordination.end() :]
        linker = re.search(
            r"\b(?:according\s+to|pursuant\s+to|under)\s*$",
            coordinated,
            flags=re.IGNORECASE,
        )
        if linker is None:
            continue
        tokens = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", coordinated[: linker.start()])
        if len(tokens) < 2:
            continue
        if _coordinated_phrase_has_finite_predicate(tokens):
            return True
        if _coordinated_dependency_object_phrase_is_bounded(
            tokens
        ) or _coordinated_dependency_object_modifier_is_bounded(tokens):
            continue
        for subject_end in range(1, len(tokens)):
            subject = " ".join(tokens[:subject_end])
            if _dependency_subject_phrase_is_bounded(
                subject
            ) or _legal_actor_subject_phrase_is_bounded(subject):
                return True
    return False


def _coordinated_dependency_object_phrase_is_bounded(tokens: list[str]) -> bool:
    for actor_end in range(1, len(tokens)):
        if not _legal_actor_subject_phrase_is_bounded(" ".join(tokens[:actor_end])):
            continue
        if _actor_nominal_object_is_bounded(tokens, actor_end=actor_end):
            return True
        object_tokens = tokens[actor_end:]
        if not _dependency_subject_phrase_is_bounded(" ".join(object_tokens)):
            continue
        if (
            len(object_tokens) == 1
            and (
                _dependency_token_forms(object_tokens[0])
                & _DEPENDENCY_ACTOR_AMBIGUOUS_OBJECT_TERMS
            )
            and not _legal_actor_acronym_phrase_is_bounded(" ".join(tokens[:actor_end]))
        ):
            return False
        return True
    return _dependency_subject_phrase_is_bounded(" ".join(tokens))


def _coordinated_phrase_has_finite_predicate(tokens: list[str]) -> bool:
    if _actor_acronym_sequence_starts_finite_clause(tokens):
        return True
    if _unlisted_actor_acronym_starts_finite_clause(tokens):
        return True
    for subject_end in range(1, len(tokens)):
        subject = " ".join(tokens[:subject_end])
        subject_is_actor = _legal_actor_subject_phrase_is_bounded(subject)
        if not (subject_is_actor or _dependency_subject_phrase_is_bounded(subject)):
            continue
        predicate = tokens[subject_end]
        trailing_object = tokens[subject_end + 1 :]
        if predicate.lower() in _DEPENDENCY_COMPOUND_NOMINAL_TERMS:
            continue
        if subject_is_actor and _actor_nominal_object_is_bounded(
            tokens,
            actor_end=subject_end,
        ):
            continue
        if trailing_object:
            if not _dependency_subject_phrase_is_bounded(" ".join(trailing_object)):
                continue
            is_input_requirement = (
                predicate.lower() == "input"
                and len(trailing_object) == 1
                and "requirement" in _dependency_token_forms(trailing_object[-1])
                and not {
                    "a",
                    "an",
                    "that",
                    "the",
                    "these",
                    "this",
                    "those",
                }
                & {token.lower() for token in trailing_object}
            )
            if is_input_requirement:
                continue
            if predicate.lower() in _DEPENDENCY_ZERO_MARKED_FINITE_VERB_TERMS:
                return True
        elif not (
            subject_is_actor
            and _dependency_token_forms(predicate)
            & _DEPENDENCY_ACTOR_AMBIGUOUS_OBJECT_TERMS
            and not _legal_actor_acronym_phrase_is_bounded(subject)
        ):
            continue
        if _dependency_subject_and_predicate_agree(tokens[subject_end - 1], predicate):
            return True
    return False


def _actor_acronym_sequence_starts_finite_clause(tokens: list[str]) -> bool:
    predicate_index = _legal_actor_acronym_sequence_end(tokens)
    if predicate_index is None or predicate_index >= len(tokens) - 1:
        return False
    if _actor_nominal_object_is_bounded(tokens, actor_end=predicate_index):
        return False
    predicate = tokens[predicate_index].lower()
    return not (
        _dependency_token_forms(predicate) & _DEPENDENCY_MODIFIER_TERMS
    ) and _dependency_subject_phrase_is_bounded(" ".join(tokens[predicate_index + 1 :]))


def _unlisted_actor_acronym_starts_finite_clause(tokens: list[str]) -> bool:
    recognized_actor_end = _legal_actor_acronym_sequence_end(tokens)
    if (
        len(tokens) < 3
        or (recognized_actor_end is not None and recognized_actor_end > 1)
        or not re.fullmatch(r"[A-Za-z]{2,16}", tokens[0])
        or _dependency_token_forms(tokens[0]) & _DEPENDENCY_MODIFIER_TERMS
    ):
        return False
    if _actor_nominal_object_is_bounded(tokens, actor_end=1):
        return False
    return True


def _legal_actor_acronym_sequence_end(tokens: list[str]) -> int | None:
    actor_count = 0
    index = 0
    while index < len(tokens):
        token = tokens[index].lower()
        if token in _LEGAL_ACTOR_ACRONYMS:
            actor_count += 1
            index += 1
            continue
        if (
            token in {"and", "or"}
            and actor_count
            and index + 1 < len(tokens)
            and tokens[index + 1].lower() in _LEGAL_ACTOR_ACRONYMS
        ):
            index += 1
            continue
        break
    return index if actor_count else None


def _actor_nominal_object_is_bounded(
    tokens: list[str],
    *,
    actor_end: int,
) -> bool:
    object_tokens = tokens[actor_end:]
    predicate_forms = _dependency_token_forms(object_tokens[0])
    if (
        len(object_tokens) < 2
        or object_tokens[0].lower() in _DEPENDENCY_ZERO_MARKED_FINITE_VERB_TERMS
        or predicate_forms & _DEPENDENCY_ACTOR_ALWAYS_FINITE_PREDICATE_TERMS
        or _actor_nominal_object_contains_finite_clause(object_tokens)
        or any(
            token.lower() in _DEPENDENCY_DETERMINER_TERMS for token in object_tokens[1:]
        )
    ):
        return False
    policy_program_subject = (
        actor_end == 1 and tokens[0].lower() in _DEPENDENCY_POLICY_PROGRAM_SUBJECT_TERMS
    )
    if _dependency_token_forms(
        object_tokens[-1]
    ) & _DEPENDENCY_ACTOR_NOMINAL_HEAD_TERMS and all(
        _dependency_token_is_actor_nominal_component(
            token,
            component_index=component_index,
            allow_policy_terms=policy_program_subject,
        )
        for component_index, token in enumerate(object_tokens[:-1])
    ):
        return True
    if (
        len(object_tokens) >= 3
        and predicate_forms & _DEPENDENCY_MODIFIER_TERMS
        and _dependency_subject_phrase_is_bounded(" ".join(object_tokens))
        and _dependency_subject_phrase_is_bounded(" ".join(object_tokens[1:]))
    ):
        return True
    if (
        _dependency_token_is_actor_nominal_prefix(object_tokens[0])
        and _dependency_token_forms(object_tokens[-1])
        & _DEPENDENCY_ACTOR_NOMINAL_HEAD_TERMS
        and _dependency_subject_phrase_is_bounded(" ".join(object_tokens[1:]))
    ):
        return True
    if len(object_tokens) >= 3 and _dependency_subject_phrase_is_bounded(
        " ".join(object_tokens[1:])
    ):
        return bool(predicate_forms & _DEPENDENCY_COMPOUND_NOMINAL_TERMS)
    return False


def _actor_nominal_object_contains_finite_clause(object_tokens: list[str]) -> bool:
    for predicate_index in range(1, len(object_tokens) - 1):
        subject_tokens = object_tokens[:predicate_index]
        if not _dependency_subject_phrase_is_bounded(" ".join(subject_tokens)):
            continue
        predicate = object_tokens[predicate_index].lower()
        predicate_forms = _dependency_token_forms(predicate)
        trailing_object = object_tokens[predicate_index + 1 :]
        if not _dependency_subject_phrase_is_bounded(" ".join(trailing_object)):
            continue
        if predicate_forms & _DEPENDENCY_COMPOUND_NOMINAL_TERMS:
            continue
        if (
            predicate_forms
            & (
                _DEPENDENCY_ACTOR_NOMINAL_HEAD_TERMS
                - _DEPENDENCY_ACTOR_AMBIGUOUS_OBJECT_TERMS
            )
            and not predicate_forms & _DEPENDENCY_ACTOR_ALWAYS_FINITE_PREDICATE_TERMS
        ):
            continue
        if (
            predicate,
            trailing_object[0].lower(),
        ) in _DEPENDENCY_NESTED_NOMINAL_COMPOUNDS:
            continue
        is_input_requirement = (
            predicate == "input"
            and len(trailing_object) == 1
            and "requirement" in _dependency_token_forms(trailing_object[0])
        )
        if is_input_requirement:
            continue
        if _dependency_subject_and_predicate_agree(
            subject_tokens[-1],
            predicate,
        ):
            return True
    return False


def _dependency_token_is_actor_nominal_prefix(token: str) -> bool:
    lowered = token.lower()
    forms = _dependency_token_forms(lowered)
    if forms & _DEPENDENCY_ACTOR_ALWAYS_FINITE_PREDICATE_TERMS:
        return False
    return bool(
        forms
        & (
            _DEPENDENCY_ACTOR_NOMINAL_PREFIX_TERMS
            | _DEPENDENCY_MODIFIER_TERMS
            | _LEGAL_INSTRUMENT_TERMS
        )
        or lowered.endswith(
            (
                "age",
                "al",
                "ance",
                "ence",
                "ful",
                "ic",
                "ing",
                "ion",
                "ity",
                "ive",
                "less",
                "ment",
                "ness",
                "ory",
                "ous",
                "ship",
            )
        )
    )


def _dependency_token_is_actor_nominal_component(
    token: str,
    *,
    component_index: int,
    allow_policy_terms: bool,
) -> bool:
    parts = token.lower().split("-")
    return all(
        (
            allow_policy_terms
            and (
                (
                    part in _DEPENDENCY_POLICY_ACRONYM_TERMS
                    or part in _DEPENDENCY_POLICY_COMPOUND_CONNECTOR_TERMS
                )
                and (
                    part not in _DEPENDENCY_POLICY_LEADING_ONLY_TERMS
                    or component_index == 0
                )
            )
        )
        or _dependency_token_is_actor_nominal_prefix(part)
        for part in parts
    )


def _coordinated_dependency_object_modifier_is_bounded(tokens: list[str]) -> bool:
    for subject_end in range(1, len(tokens)):
        if not _dependency_subject_phrase_is_bounded(" ".join(tokens[:subject_end])):
            continue
        modifier_tokens = tokens[subject_end:]
        while len(modifier_tokens) > 1 and _dependency_token_is_object_modifier_adverb(
            modifier_tokens[0]
        ):
            modifier_tokens = modifier_tokens[1:]
        if modifier_tokens[0].lower() in _DEPENDENCY_OBJECT_MODIFIER_TERMS and all(
            _dependency_token_is_object_modifier_adverb(token)
            for token in modifier_tokens[1:]
        ):
            return True
    return False


def _legal_actor_subject_phrase_is_bounded(phrase: str) -> bool:
    if _legal_actor_acronym_phrase_is_bounded(phrase):
        return True
    if re.fullmatch(
        r"(?:the\s+)?(?:[A-Z][A-Za-z-]*\s+){0,5}"
        r"(?:Administration|Agency|Authority|Board|Bureau|Commission|Corporation|Department|Judiciary|Office|Reserve|Service)s?",
        phrase,
    ):
        return True
    if re.fullmatch(
        r"(?:the\s+)?Office\s+of\s+(?:[A-Z][A-Za-z-]*\s*){1,6}"
        r"(?:and\s+(?:[A-Z][A-Za-z-]*\s*){1,3})?",
        phrase,
    ):
        return True
    if re.fullmatch(
        r"(?:the\s+)?offices?\s+of\s+"
        r"(?:management\s+and\s+budget|personnel\s+management)",
        phrase,
        flags=re.IGNORECASE,
    ):
        return True
    return bool(
        re.fullmatch(
            r"(?:a|an|the|this|that|these|those)?\s*"
            r"(?:(?:administering|federal|local|public(?:-|\s+)housing|state)\s+)?"
            r"(?:"
            r"agenc(?:y|ies)|administrators?|authorit(?:y|ies)|commissions?|"
            r"commissioners?|departments?|secretar(?:y|ies)|services?|"
            r"federal\s+(?:(?:[a-z-]+\s+){0,4}"
            r"(?:administrations?|administrators?|agenc(?:y|ies)|authorit(?:y|ies)|"
            r"boards?|bureaus?|commissions?|commissioners?|corporations?|"
            r"departments?|judiciar(?:y|ies)|offices?|reserves?|"
            r"secretar(?:y|ies)|services?)|"
            r"bureaus?\s+of\s+investigation)|"
            r"internal\s+revenue\s+services?|"
            r"social\s+security\s+administrations?|"
            r"(?:united\s+states\s+)?departments?\s+of\s+(?:the\s+)?"
            r"(?:agriculture|commerce|defense|education|energy|"
            r"health\s+and\s+human\s+services|homeland\s+security|"
            r"housing\s+and\s+urban\s+development|interior|justice|labor|state|"
            r"transportation|treasury|veterans\s+affairs)|"
            r"secretar(?:y|ies)\s+of\s+(?:the\s+)?"
            r"(?:agriculture|commerce|defense|education|energy|"
            r"health\s+and\s+human\s+services|homeland\s+security|"
            r"housing\s+and\s+urban\s+development|interior|labor|state|"
            r"transportation|treasury|veterans\s+affairs)|"
            r"commissioners?\s+of\s+(?:internal\s+revenue|social\s+security)"
            r")",
            phrase,
            flags=re.IGNORECASE,
        )
    )


def _legal_actor_acronym_phrase_is_bounded(phrase: str) -> bool:
    normalized = re.sub(r"^(?:the\s+)", "", phrase.strip(), flags=re.IGNORECASE)
    tokens = re.findall(r"[A-Za-z]+", normalized)
    sequence_end = _legal_actor_acronym_sequence_end(tokens)
    return sequence_end == len(tokens)


def _dependency_token_is_object_modifier_adverb(token: str) -> bool:
    lowered = token.lower()
    return lowered in _DEPENDENCY_OBJECT_MODIFIER_ADVERBS or lowered.endswith("ly")


def _dependency_token_forms(token: str) -> set[str]:
    lowered = token.lower()
    candidates = {lowered}
    if lowered.endswith("ies"):
        candidates.add(f"{lowered[:-3]}y")
    if lowered.endswith("es"):
        candidates.add(lowered[:-2])
    if lowered.endswith("s"):
        candidates.add(lowered[:-1])
    return candidates


def _dependency_subject_and_predicate_agree(
    subject_token: str,
    predicate_token: str,
) -> bool:
    subject = subject_token.lower()
    if subject in _DEPENDENCY_NUMBER_AMBIGUOUS_TERMS:
        return True
    subject_is_plural = subject in _DEPENDENCY_IRREGULAR_PLURAL_TERMS or any(
        candidate in _DEPENDENCY_SUBJECT_TERMS or candidate in _LEGAL_ACTOR_HEAD_TERMS
        for candidate in _dependency_token_forms(subject) - {subject}
    )
    predicate = predicate_token.lower()
    predicate_is_singular = predicate.endswith(
        ("s", "es", "ies")
    ) and not predicate.endswith(("ss", "us", "is"))
    return subject_is_plural != predicate_is_singular


def _reason_direct_missing_introduction_is_bounded(
    introduction: str,
    *,
    signal: str,
) -> bool:
    if re.fullmatch(
        r"cannot\s+be\s+(?:computed|encoded|resolved)",
        signal,
        flags=re.IGNORECASE,
    ):
        dependency_introduction = re.sub(
            r"^\s*(?:because|since|as|when)\s+",
            "",
            introduction,
            flags=re.IGNORECASE,
        )
        return not dependency_introduction.strip() or (
            _reason_dependency_introduction_is_bounded(dependency_introduction)
        )
    return _reason_reference_introduction_is_bounded(introduction)


def _usc_dependency_is_external(
    match: re.Match[str],
    *,
    current_citation: CitationParts | None,
) -> bool:
    if current_citation is None:
        return True
    return match.group(
        "title"
    ).lower() != current_citation.title.lower() or normalize_rulespec_path_segment(
        match.group("section").lower()
    ) != normalize_rulespec_path_segment(current_citation.section.lower())


def _louisiana_rs_dependency_is_external(
    match: re.Match[str],
    *,
    corpus_citation_path: str,
) -> bool:
    """Reject current-section Louisiana citations masquerading as dependencies."""

    citation_parts = [
        part for part in corpus_citation_path.strip("/").split("/") if part
    ]
    if (
        len(citation_parts) < 3
        or citation_parts[0].lower() != "us-la"
        or citation_parts[1].lower() != "statute"
    ):
        return False
    statute_tail = citation_parts[2:]
    if ":" in statute_tail[0]:
        title_section = statute_tail[0].split(":")
        if len(title_section) != 2 or not all(title_section):
            return False
        current_title, current_section = title_section
    elif len(statute_tail) >= 2:
        current_title, current_section = statute_tail[:2]
    else:
        return False
    return match.group(
        "title"
    ).lower() != current_title.lower() or normalize_rulespec_path_segment(
        match.group("section").lower()
    ) != normalize_rulespec_path_segment(current_section.lower())


def _prose_dependency_is_external(
    dependency: str,
    *,
    current_section: str,
) -> bool:
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
        return False
    if section_match := re.fullmatch(
        r"§{1,2}\s*(\d+[a-z]?)",
        dependency,
        flags=re.IGNORECASE,
    ):
        if section_match.group(1).lower() == current_section:
            return False
    return not (
        normalized.rstrip("/").endswith(f"/{current_section}") and "#" not in normalized
    )


def _reason_names_source_bound_runtime_gap(
    reason: str,
    source_scope_text: str,
    *,
    path: tuple[str, ...],
    corpus_citation_path: str,
) -> bool:
    """Accept a concrete runtime gap anchored to the exact current statute branch."""

    if (
        not path
        or not source_scope_text.strip()
        or not _MISSING_DEPENDENCY_LANGUAGE.search(reason)
        or not _SOURCE_BOUND_RUNTIME_GAP_LANGUAGE.search(reason)
        or not _ADMINISTRATIVE_SOURCE_ARTIFACT_LANGUAGE.search(source_scope_text)
        or not _ADMINISTRATIVE_SOURCE_ACTION_LANGUAGE.search(source_scope_text)
        or source_states_explicit_computation(source_scope_text)
    ):
        return False
    citation_parts = [
        part for part in corpus_citation_path.strip("/").split("/") if part
    ]
    if len(citation_parts) < 3 or citation_parts[1] != "statute":
        return False
    if not _reason_cites_exact_current_statute_branch(
        reason,
        corpus_citation_path=corpus_citation_path,
        path=path,
    ):
        return False

    def substantive_tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z][a-z0-9]+", text.lower())
            if len(token) >= 5 and token not in _SOURCE_BOUND_RUNTIME_GAP_STOPWORDS
        }

    # Two shared source terms keep an exact self-citation from laundering a vague
    # or invented capability assertion into complete-source coverage.
    return len(substantive_tokens(reason) & substantive_tokens(source_scope_text)) >= 2


def _reason_covers_source_authenticated_repeal(
    reason: str,
    source_scope_text: str,
    *,
    path: tuple[str, ...],
    corpus_citation_path: str,
) -> bool:
    """Cover only an exact current branch whose authoritative text is a repeal."""

    if not path or not source_scope_text.strip():
        return False
    if not _source_branch_is_explicit_repeal_tombstone(source_scope_text):
        return False
    for affirmative in re.finditer(
        r"\b(?:has\s+been|is|stands?|was)\s+repealed\b",
        reason,
        flags=re.IGNORECASE,
    ):
        clause_start, _clause_end = _reason_clause_bounds(reason, affirmative)
        citation_subject = reason[clause_start : affirmative.start()]
        if not _reason_cites_exact_current_statute_branch(
            citation_subject,
            corpus_citation_path=corpus_citation_path,
            path=path,
            strict_terminal=True,
            citation_must_end=True,
            citation_must_start=True,
        ):
            continue
        return _repeal_reason_tail_is_bounded(
            reason[affirmative.end() :],
            source_scope_text=source_scope_text,
        )
    return False


def _repeal_reason_tail_is_bounded(
    tail: str,
    *,
    source_scope_text: str,
) -> bool:
    """Accept only finite tombstone history and non-operative explanations."""

    bounded = _collapse_text(tail).strip()
    if re.fullmatch(r"\.?", bounded):
        return True
    if re.match(
        r"^(?:by\s+|\.\s*)acts?(?:\s+of)?\b",
        bounded,
        flags=re.IGNORECASE,
    ):
        normalized_history = _normalize_written_section_marker(
            re.sub(r"^(?:by\s+|\.\s*)", "", bounded, flags=re.IGNORECASE)
        )
        history = _LOUISIANA_SESSION_LAW_CITATION.match(normalized_history)
        if history is None or history.start() != 0:
            return False
        reason_history = normalized_history[: history.end()]
        bounded = normalized_history[history.end() :].strip()
        effective_date = re.match(
            r"^,\s*eff(?:ective)?\.?\s+"
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|"
            r"July?|Aug(?:ust)?|Sept(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|"
            r"Dec(?:ember)?)\.?\s+\d{1,2}\s*,\s*\d{4}",
            bounded,
            flags=re.IGNORECASE,
        )
        if effective_date is not None:
            reason_history += " " + effective_date.group(0)
            bounded = bounded[effective_date.end() :].strip()
        reason_identity = _normalized_repeal_history(reason_history)
        source_identity = _source_repeal_history(source_scope_text)
        if (
            reason_identity is None
            or source_identity is None
            or reason_identity != source_identity
        ):
            return False
        if re.fullmatch(r"\.?", bounded):
            return True
    explanation = (
        r"(?:,\s*)?(?:and\s+)?(?:"
        r"supplies?\s+no\s+operative\s+rule(?:\s+to\s+encode)?|"
        r"has\s+no\s+operative\s+(?:effect|rule|text)|"
        r"contains?\s+no\s+operative\s+(?:legal\s+)?"
        r"(?:effect|predicate|provision|requirement|rule|text)|"
        r"contains?\s+no\s+operative\s+legal\s+predicate\s*,\s*amount\s*,\s*"
        r"source-stated\s+input\s*,\s*or\s+runtime\s+consequence\s+from\s+"
        r"which\s+an\s+executable\s+output\s+can\s+be\s+computed"
        r")\s*\.?"
    )
    return bool(re.fullmatch(explanation, bounded, flags=re.IGNORECASE))


def _source_repeal_history(source_scope_text: str) -> tuple[object, ...] | None:
    """Return the normalized authenticated history on a repeal tombstone."""

    source_body = _collapse_text(_strip_source_clause_marker(source_scope_text)).strip()
    if source_body.startswith("[") and source_body.endswith("]"):
        source_body = source_body[1:-1].strip()
    repeal = re.fullmatch(
        r"repealed\s+by\s+(?P<history>.+)",
        source_body,
        flags=re.IGNORECASE,
    )
    if repeal is None:
        return None
    return _normalized_repeal_history(repeal.group("history"))


def _normalized_repeal_history(history: str) -> tuple[object, ...] | None:
    """Parse a session law without collapsing list and range structure."""

    normalized = _normalize_written_section_marker(history)
    citation = _LOUISIANA_SESSION_LAW_CITATION.match(normalized)
    if citation is None or citation.start() != 0:
        return None
    citation_text = citation.group(0)
    year = re.search(r"\b(?P<year>\d{4})\b", citation_text)
    number_marker = re.search(r"\bNos?\.?", citation_text, flags=re.IGNORECASE)
    if year is None or number_marker is None:
        return None

    session_text = citation_text[year.end() : number_marker.start()]
    session_text = re.sub(
        r"\bE\.?\s*S\.?\b",
        "ex sess",
        session_text,
        flags=re.IGNORECASE,
    )
    session_text = re.sub(
        r"\b(?:Ex(?:tra)?\.?|Extraordinary)\b",
        "ex",
        session_text,
        flags=re.IGNORECASE,
    )
    session_text = re.sub(
        r"\bSess(?:ion)?\.?\b",
        "sess",
        session_text,
        flags=re.IGNORECASE,
    )
    session_text = re.sub(
        r"\b(\d+)\s+(st|nd|rd|th|d)\b",
        r"\1\2",
        session_text,
        flags=re.IGNORECASE,
    )
    ordinal_words = {
        "first": "1",
        "second": "2",
        "third": "3",
        "fourth": "4",
        "fifth": "5",
        "sixth": "6",
        "seventh": "7",
        "eighth": "8",
        "ninth": "9",
        "tenth": "10",
    }
    session_tokens = re.findall(r"[a-z0-9]+", session_text.lower())
    ex_index = session_tokens.index("ex") if "ex" in session_tokens else 0
    if ex_index:
        word_ordinal = _english_ordinal_value(session_tokens[:ex_index])
        if word_ordinal is not None:
            session_tokens = [str(word_ordinal), *session_tokens[ex_index:]]
    session_parts: list[str] = []
    for token in session_tokens:
        if token in ordinal_words:
            session_parts.append(ordinal_words[token])
            continue
        ordinal = re.fullmatch(r"(?P<number>\d+)(?P<suffix>st|nd|rd|th|d)", token)
        if ordinal is None:
            session_parts.append(token)
            continue
        number = int(ordinal.group("number"))
        suffix = ordinal.group("suffix")
        final_two = number % 100
        expected_suffix = (
            "th"
            if 11 <= final_two <= 13
            else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
        )
        historical_d = (
            suffix == "d" and not 11 <= final_two <= 13 and number % 10 in {2, 3}
        )
        if suffix != expected_suffix and not historical_d:
            return None
        session_parts.append(str(number))
    session = tuple(session_parts)

    section_marker = re.search(r"§{1,2}", citation_text)
    acts_end = (
        section_marker.start() if section_marker is not None else len(citation_text)
    )
    act_numbers = tuple(
        match.group(0)
        for match in re.finditer(
            r"\d+(?:[-–—]\d+)?",
            citation_text[number_marker.end() : acts_end],
        )
    )
    sections: tuple[str, ...] = ()
    if section_marker is not None:
        section_text = citation_text[section_marker.end() :]
        section_text = re.sub(r"\b(?:through|to)\b", "-", section_text, flags=re.I)
        section_text = re.sub(r"\band\b", ",", section_text, flags=re.I)
        section_text = re.sub(r"[–—]", "-", section_text)
        section_text = re.sub(r"\s+", "", section_text).strip(",")
        sections = tuple(part for part in section_text.split(",") if part)

    effective_date: tuple[str, str, str] | None = None
    date = re.search(
        r"\beff(?:ective)?\.?\s+"
        r"(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|"
        r"July?|Aug(?:ust)?|Sept(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\.?\s+(?P<day>\d{1,2})\s*,\s*(?P<year>\d{4})",
        normalized[citation.end() :],
        flags=re.IGNORECASE,
    )
    if date is not None:
        month = date.group("month").lower()[:3]
        if month == "sep":
            month = "sep"
        effective_date = (month, str(int(date.group("day"))), date.group("year"))

    return (
        year.group("year"),
        session,
        act_numbers,
        sections,
        effective_date,
    )


def _normalize_written_section_marker(text: str) -> str:
    """Preserve singular/list structure for written section markers."""

    return re.sub(
        r"\b(?P<label>sections?)\s+(?=\d)",
        lambda match: "§§" if match.group("label").lower().endswith("s") else "§",
        text,
        flags=re.IGNORECASE,
    )


def _english_ordinal_value(tokens: Sequence[str]) -> int | None:
    """Normalize ordinary English session ordinals admitted by the citation grammar."""

    single = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
        "eleventh": 11,
        "twelfth": 12,
        "thirteenth": 13,
        "fourteenth": 14,
        "fifteenth": 15,
        "sixteenth": 16,
        "seventeenth": 17,
        "eighteenth": 18,
        "nineteenth": 19,
        "twentieth": 20,
        "thirtieth": 30,
        "fortieth": 40,
        "fiftieth": 50,
        "sixtieth": 60,
        "seventieth": 70,
        "eightieth": 80,
        "ninetieth": 90,
    }
    if len(tokens) == 1:
        return single.get(tokens[0])
    tens = {
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    if len(tokens) == 2 and tokens[0] in tens and tokens[1] in single:
        unit = single[tokens[1]]
        if 1 <= unit <= 9:
            return tens[tokens[0]] + unit
    return None


def _source_branch_is_explicit_repeal_tombstone(source_scope_text: str) -> bool:
    """Accept a bare repeal or a finite Louisiana session-law repeal history."""

    source_body = _collapse_text(_strip_source_clause_marker(source_scope_text)).strip()
    if source_body.startswith("[") and source_body.endswith("]"):
        source_body = source_body[1:-1].strip()
    if re.fullmatch(r"repealed\s*[.;]?", source_body, flags=re.IGNORECASE):
        return True
    repeal = re.fullmatch(
        r"repealed\s+by\s+(?P<history>.+)",
        source_body,
        flags=re.IGNORECASE,
    )
    if repeal is None:
        return False
    history = _normalize_written_section_marker(repeal.group("history"))
    citation = _LOUISIANA_SESSION_LAW_CITATION.match(history)
    if citation is None or citation.start() != 0:
        return False
    effective_date_tail = history[citation.end() :]
    return bool(
        _normalized_repeal_history(history) is not None
        and re.fullmatch(
            r"\s*(?:,\s*eff(?:ective)?\.?\s+"
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|"
            r"July?|Aug(?:ust)?|Sept(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|"
            r"Dec(?:ember)?)\.?\s+\d{1,2}\s*,\s*\d{4})?\s*\.?\s*",
            effective_date_tail,
            flags=re.IGNORECASE,
        )
    )


def _reason_cites_exact_current_statute_branch(
    reason: str,
    *,
    corpus_citation_path: str,
    path: tuple[str, ...],
    strict_terminal: bool = False,
    citation_must_end: bool = False,
    citation_must_start: bool = False,
) -> bool:
    """Bind a runtime-gap reason to one complete current-source branch."""

    dash_pattern = (
        "[-\\u2010\\u2011\\u2012\\u2013\\u2014\\u2015\\u2212\\ufe58\\ufe63\\uff0d]"
    )

    def literal_pattern(value: str) -> str:
        normalized = re.sub(
            r"[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]",
            "-",
            value,
        )
        return re.escape(normalized).replace(r"\-", dash_pattern)

    complete_branch = path
    branch_pattern = r"\s*".join(
        rf"\(\s*{re.escape(normalize_rulespec_path_segment(part))}\s*\)"
        for part in complete_branch
    )
    state_branch_terminal_guard = r"(?!\s*\()(?![A-Za-z0-9_])"

    if corpus_citation_path.startswith("us/statute/"):
        try:
            citation = parse_usc_citation(corpus_citation_path)
        except ValueError:
            return False
        complete_usc_branch = (*citation.fragments, *path)
        usc_branch_pattern = r"\s*".join(
            rf"\(\s*{re.escape(normalize_rulespec_path_segment(part))}\s*\)"
            for part in complete_usc_branch
        )
        usc_terminal_guard = (
            state_branch_terminal_guard
            if strict_terminal
            else (r"(?!\s*\()" if len(complete_usc_branch) > 1 else "")
        )
        section_pattern = literal_pattern(
            normalize_rulespec_path_segment(citation.section)
        )
        start_guard = r"^\s*" if citation_must_start else ""
        end_guard = r"\s*$" if citation_must_end else ""
        return bool(
            re.search(
                rf"{start_guard}\b{re.escape(citation.title)}\s+"
                rf"U\.?\s*S\.?\s*C\.?\s*"
                rf"(?:§{{1,2}}\s*)?{section_pattern}\s*{usc_branch_pattern}"
                rf"{usc_terminal_guard}{end_guard}",
                reason,
                flags=re.IGNORECASE,
            )
        )

    start_guard = r"^\s*" if citation_must_start else ""
    end_guard = r"\s*$" if citation_must_end else ""
    canonical_pattern = re.compile(
        rf"{start_guard}(?<![A-Za-z0-9_.-])"
        rf"{literal_pattern(corpus_citation_path.rstrip('/'))}"
        rf"\s*{branch_pattern}{state_branch_terminal_guard}{end_guard}",
        flags=re.IGNORECASE,
    )
    if canonical_pattern.search(reason):
        return True

    citation_parts = [
        part for part in corpus_citation_path.strip("/").split("/") if part
    ]
    if len(citation_parts) < 3 or citation_parts[0].lower() != "us-la":
        return False
    statute_tail = citation_parts[2:]
    if not statute_tail:
        return False
    if ":" in statute_tail[0]:
        title_section = statute_tail[0].split(":")
        if len(title_section) != 2 or not all(title_section):
            return False
        title, section = title_section
        existing_scope = statute_tail[1:]
    else:
        if len(statute_tail) < 2:
            return False
        title, section, *existing_scope = statute_tail
    complete_louisiana_branch = (*existing_scope, *path)
    louisiana_branch_pattern = r"\s*".join(
        rf"\(\s*{re.escape(normalize_rulespec_path_segment(part))}\s*\)"
        for part in complete_louisiana_branch
    )
    louisiana_rs_pattern = re.compile(
        rf"{start_guard}(?<![A-Za-z0-9])(?:La\.?\s+)?R\.?\s*S\.?\s*"
        rf"{literal_pattern(title)}\s*:\s*{literal_pattern(section)}\s*"
        rf"{louisiana_branch_pattern}"
        rf"{state_branch_terminal_guard}{end_guard}",
        flags=re.IGNORECASE,
    )
    return bool(louisiana_rs_pattern.search(reason))


def _rulespec_target_base(corpus_citation_path: str) -> str:
    parts = [
        normalize_rulespec_path_segment(part)
        for part in corpus_citation_path.strip("/").split("/")
        if part
    ]
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
    if jurisdiction == "us-la" and plural == "statutes" and tail:
        title_section = tail[0].split(":")
        if len(title_section) == 2 and all(title_section):
            tail = [*title_section, *tail[1:]]
    if (
        plural == "statutes"
        and tail
        and not _preserve_state_statute_dotted_target_leaf(
            jurisdiction,
            plural,
            tail,
        )
    ):
        leaf_segments = [part for part in tail[-1].split(".") if part]
        if leaf_segments:
            tail = [*tail[:-1], *leaf_segments]
    return f"{jurisdiction}:{plural}/{'/'.join(tail)}"


def _preserve_state_statute_dotted_target_leaf(
    jurisdiction: str,
    root: str,
    tail: list[str],
) -> bool:
    """Mirror canonical state-statute leaves that intentionally retain dots."""

    if jurisdiction != "us-co" or root != "statutes" or len(tail) < 2:
        return False
    if len(tail) == 2 and tail[0].isdigit():
        crs_segments = tail[-1].split("-")
        return len(crs_segments) == 3 and all(
            re.fullmatch(r"\d+(?:\.\d+)*", segment) for segment in crs_segments
        )
    if not re.fullmatch(r"\d+(?:\.\d+)+", tail[-1]):
        return False
    return bool(re.fullmatch(r"\d+(?:-\d+)+(?:\.\d+)?", tail[-2]))


def _is_marker_only_container(
    branch: SourceStructureBranch,
    *,
    branches: Sequence[SourceStructureBranch],
    source_text: str,
) -> bool:
    """Ignore a parent wrapper only when it has no operative chapeau text."""

    direct_child_starts = [
        candidate.start
        for candidate in branches
        if len(candidate.path) == len(branch.path) + 1
        and candidate.path[: len(branch.path)] == branch.path
        and branch.start <= candidate.start < branch.end
    ]
    if not direct_child_starts:
        return False
    chapeau = source_text[branch.start : min(direct_child_starts)]
    return re.search(r"\w", _strip_source_clause_marker(chapeau)) is None


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
        deferred_path and path[: len(deferred_path)] == deferred_path
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


def _strip_terminal_session_law_history(source_text: str) -> str:
    """Remove only a fully validated terminal session-law history chain."""

    terminal_parenthetical = re.search(
        r"\s+\((?P<history>[^()]*)\)\s*$",
        source_text,
    )
    if terminal_parenthetical is not None:
        history_entries = tuple(
            entry.strip()
            for entry in terminal_parenthetical.group("history").split(";")
        )
        if history_entries and all(
            _ALABAMA_TERMINAL_ACT_HISTORY_ENTRY.fullmatch(entry)
            or _ALABAMA_TERMINAL_CODE_HISTORY_ENTRY.fullmatch(entry)
            for entry in history_entries
        ):
            return source_text[: terminal_parenthetical.start()].rstrip()

    blank_lines = tuple(re.finditer(r"\n[ \t]*\n", source_text))
    if blank_lines:
        block_start = blank_lines[-1].end()
        history_block = source_text[block_start:].lstrip()
        normalized_history_block = _SESSION_LAW_SENTENCE_SEPARATOR.sub(
            "; ",
            history_block,
        )
        history_entries = tuple(
            _SESSION_LAW_ENTRY.fullmatch(part)
            for part in normalized_history_block.split(";")
        )
        if (
            _STANDALONE_HISTORY_LEADER.match(history_block)
            and _SESSION_LAW_YEAR_CHAPTER.search(history_block)
            and history_entries
            and all(entry is not None for entry in history_entries)
        ):
            return source_text[: blank_lines[-1].start()].rstrip()

    separators = tuple(match.start() for match in re.finditer(";", source_text))
    segment_starts = (0, *(position + 1 for position in separators))
    segment_ends = (*separators, len(source_text))
    segment_matches = tuple(
        _SESSION_LAW_ENTRY.fullmatch(source_text[start:end])
        for start, end in zip(segment_starts, segment_ends)
    )
    suffix_is_valid = [False] * (len(segment_matches) + 1)
    suffix_has_action = [False] * (len(segment_matches) + 1)
    suffix_is_valid[-1] = True
    for index in range(len(segment_matches) - 1, -1, -1):
        entry = segment_matches[index]
        suffix_is_valid[index] = entry is not None and suffix_is_valid[index + 1]
        suffix_has_action[index] = (
            entry is not None and entry.group("action") is not None
        ) or suffix_has_action[index + 1]

    last_candidate_by_segment: dict[int, re.Match[str]] = {}
    for candidate in _SESSION_LAW_TAIL_START.finditer(source_text):
        segment_index = bisect.bisect_right(separators, candidate.start())
        last_candidate_by_segment[segment_index] = candidate

    for segment_index, candidate in sorted(last_candidate_by_segment.items()):
        first_entry = _SESSION_LAW_ENTRY.fullmatch(
            source_text[candidate.start() : segment_ends[segment_index]]
        )
        if first_entry is None or not suffix_is_valid[segment_index + 1]:
            continue

        explicitly_labeled = first_entry.group("history_label") is not None
        has_action = (
            first_entry.group("action") is not None
            or suffix_has_action[segment_index + 1]
        )
        entry_count = len(segment_matches) - segment_index

        prefix = source_text[: candidate.start()]
        preceding = prefix.rstrip()
        separator = prefix[len(preceding) :]
        strong_layout = not preceding or "\n" in separator
        if (
            not explicitly_labeled
            and not strong_layout
            and (entry_count < 2 or not has_action)
        ):
            continue
        if (
            not explicitly_labeled
            and not strong_layout
            and preceding
            and preceding[-1] not in ".!?"
        ):
            continue
        return source_text[: candidate.start()].rstrip()

    return source_text


def authoritative_numeric_recall_text(source_text: str) -> str:
    """Remove structural/citation ordinals, never substantive source values."""

    cleaned = _strip_terminal_session_law_history(source_text)
    cleaned = _LOUISIANA_SESSION_LAW_CITATION.sub("", cleaned)
    cleaned = _LOUISIANA_RS_NUMERIC_RECALL_CITATION.sub("", cleaned)
    cleaned = _GERMAN_LEGAL_CITATION.sub("", cleaned)
    cleaned = _TITLE_SUFFIX_LEGAL_CITATION.sub("", cleaned)
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
        value
        for _name, value in collect_artifact_numeric_bindings(
            content,
            extract_named_scalars=extract_named_scalars,
            imported_symbol_contents=imported_symbol_contents,
        )
    )
    return tuple(values)


def collect_artifact_numeric_bindings(
    content: str,
    *,
    extract_named_scalars: NamedScalarExtractor,
    imported_symbol_contents: Sequence[tuple[str, str]] = (),
) -> tuple[tuple[str, float], ...]:
    """Collect exact scalar names together with strict recall values."""

    bindings = [
        (str(item.name), float(item.value))
        for item in extract_named_scalars(content)
        if hasattr(item, "name") and hasattr(item, "value")
    ]
    for imported_symbol, artifact_content in imported_symbol_contents:
        bindings.extend(
            (imported_symbol, float(item.value))
            for item in extract_named_scalars(artifact_content)
            if hasattr(item, "value")
            and _named_scalar_base_name(str(getattr(item, "name", "")))
            == imported_symbol
        )
    return tuple(bindings)


def _named_scalar_base_name(name: str) -> str:
    return name.split("[", 1)[0]


def _merge_unambiguous_numeric_bindings(
    environment: dict[str, Any],
    bindings: Sequence[tuple[str, float]],
) -> dict[str, Any]:
    """Merge exact scalar bindings while rejecting conflicting definitions."""

    values_by_name: dict[str, list[int | float]] = {}
    for name, value in environment.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values_by_name.setdefault(name, []).append(value)
    for name, value in bindings:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            values_by_name.setdefault(name, []).append(value)

    merged = dict(environment)
    for name, values in values_by_name.items():
        temporal = environment.get(name)
        if isinstance(temporal, _TemporalFormulaValue):
            temporal_values = [
                float(value)
                for _start, _end, value in temporal.versions
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if _numeric_binding_sequences_match(values, temporal_values):
                continue
            merged.pop(name, None)
            continue
        first = values[0]
        if all(math.isclose(float(value), float(first)) for value in values[1:]):
            merged[name] = first
        else:
            merged.pop(name, None)
    return merged


def _numeric_binding_sequences_match(
    left: Sequence[int | float],
    right: Sequence[int | float],
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        math.isclose(float(left_value), float(right_value))
        for left_value, right_value in zip(
            sorted(left, key=float),
            sorted(right, key=float),
        )
    )


_DIRECT_LOCAL_INPUT_NONNEGATIVE_CLAMP = re.compile(
    r"\bmax\s*\(\s*(?:0+(?:\.0+)?)\s*,\s*"
    r"(?P<input>[A-Za-z_][A-Za-z0-9_]*)\s*\)"
)
_NEGATIVE_LOCAL_INPUT_CLAMP_DIAGNOSTIC_LIMIT = 8


_SOURCE_EXPLICIT_CONDITION_DIAGNOSTIC_LIMIT = 8
_SOURCE_EXPLICIT_CONDITION_EXPANSION_LIMIT = 64
_SOURCE_EXPLICIT_CONDITION_DEPENDENCY_NODE_LIMIT = 128
_SOURCE_EXPLICIT_CONDITION_DEPENDENCY_DEPTH_LIMIT = (
    _SOURCE_EXPLICIT_CONDITION_DEPENDENCY_NODE_LIMIT + 1
)
_SOURCE_EXPLICIT_CONDITION_AST_DEPTH_LIMIT = 32
_SOURCE_GATE_ENTITIES = frozenset(
    {
        "applicant",
        "child",
        "claimant",
        "individual",
        "parent",
        "person",
        "spouse",
        "taxpayer",
        "widow",
        "widower",
    }
)
_SOURCE_GATE_STOPWORDS = frozenset(
    {
        "a",
        "additional",
        "all",
        "an",
        "and",
        "another",
        "any",
        "apply",
        "applies",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "being",
        "by",
        "close",
        "credit",
        "dollar",
        "each",
        "for",
        "from",
        "had",
        "has",
        "have",
        "if",
        "in",
        "is",
        "kentucky",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "or",
        "otherwise",
        "provided",
        "taxable",
        "than",
        "that",
        "the",
        "then",
        "to",
        "under",
        "unless",
        "was",
        "were",
        "when",
        "whose",
        "with",
        "without",
        "year",
    }
)
_SOURCE_GATE_PREDICATE_LANGUAGE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|file|files|filed|filing|made|"
    r"attain(?:ed|s)?|own(?:ed|s)?|employ(?:ed|ment)?|work(?:ed|ing|s)?|"
    r"insur(?:ed|ance)|live(?:d|s)?|resid(?:e|ed|es|ent|ence)|"
    r"receive(?:d|s)?|earn(?:ed|ing|ings|s)?|support(?:ed|ing|s)?|"
    r"qualif(?:y|ied|ies)|meet(?:s)?|exceed(?:ed|s)?)\b",
    flags=re.IGNORECASE,
)
_SOURCE_GATE_NOMINAL_PREDICATES = frozenset(
    {"age", "blind", "dependent", "income", "insurance", "married", "resident"}
)
_SOURCE_GATE_POLARITY_PREDICATES = frozenset({"no_income", "not_dependent"})
_SOURCE_GATE_UNKNOWN_POLARITY_PREDICATES = frozenset(
    {"unknown_income", "unknown_dependent"}
)
_SOURCE_GATE_POLARITY_BASE = {
    "no_income": "income",
    "not_dependent": "dependent",
    "unknown_income": "income",
    "unknown_dependent": "dependent",
}
_SOURCE_GATE_ADMINISTRATIVE_PREDICATES = frozenset(
    {
        "administrative",
        "complete",
        "document",
        "record",
        "review",
        "scan",
        "signoff",
        "status",
        "supervisor",
        "workflow",
    }
)
_SOURCE_INDEPENDENT_CONDITION_BOUNDARY = re.compile(
    r",\s+(?:and|or)\s+(?=(?:(?:the|a|an|each|any|every|no|taxpayer|claimant|"
    r"applicant|person|individual)\b|(?:is|are|was|were|has|have|had)\b)"
    r"[^,;.!?]{0,120}\b(?:if|when|unless|provided\s+that)\b)",
    flags=re.IGNORECASE,
)


def _source_gate_normalized_token(token: str) -> str:
    normalized = re.sub(r"['’]s$", "", token.casefold())
    aliases = {
        "abode": "resident",
        "aged": "age",
        "attained": "attain",
        "blindness": "blind",
        "dependence": "dependent",
        "dependency": "dependent",
        "earning": "income",
        "employed": "employ",
        "employment": "employ",
        "filed": "file",
        "filing": "file",
        "home": "home",
        "house": "home",
        "insured": "insurance",
        "made": "file",
        "owned": "own",
        "ownership": "own",
        "residence": "resident",
        "working": "employ",
    }
    if normalized in aliases:
        return aliases[normalized]
    if (
        len(normalized) > 4
        and normalized.endswith("s")
        and not normalized.endswith("ss")
    ):
        normalized = normalized[:-1]
    return aliases.get(normalized, normalized)


def _source_gate_semantic_tokens(text: str) -> frozenset[str]:
    tokens = {
        normalized
        for token in re.findall(r"[A-Za-z][A-Za-z'-]*", text)
        if (normalized := _source_gate_normalized_token(token.strip("'-")))
        and normalized not in _SOURCE_GATE_STOPWORDS
        and len(normalized) > 1
    }
    for base_predicate, negative in _source_gate_predicate_polarities(text).items():
        if negative is None:
            polarity_predicate = f"unknown_{base_predicate}"
        else:
            polarity_predicate = (
                "no_income" if base_predicate == "income" else "not_dependent"
            )
        tokens.discard(base_predicate)
        tokens.discard(f"unknown_{base_predicate}")
        if negative is None:
            tokens.add(polarity_predicate)
        else:
            tokens.add(polarity_predicate if negative else base_predicate)
    return frozenset(tokens)


_SOURCE_GATE_PREDICATE_SCAN_LIMIT = 16
_SOURCE_GATE_CONTRACTION_REPLACEMENTS = {
    "aren't": "are not",
    "aren’t": "are not",
    "arent": "are not",
    "aren t": "are not",
    "can't": "can not",
    "can’t": "can not",
    "cannot": "can not",
    "cant": "can not",
    "can t": "can not",
    "didn't": "did not",
    "didn’t": "did not",
    "didnt": "did not",
    "didn t": "did not",
    "doesn't": "does not",
    "doesn’t": "does not",
    "doesnt": "does not",
    "doesn t": "does not",
    "don't": "do not",
    "don’t": "do not",
    "dont": "do not",
    "don t": "do not",
    "hadn't": "had not",
    "hadn’t": "had not",
    "hadnt": "had not",
    "hadn t": "had not",
    "hasn't": "has not",
    "hasn’t": "has not",
    "hasnt": "has not",
    "hasn t": "has not",
    "haven't": "have not",
    "haven’t": "have not",
    "havent": "have not",
    "haven t": "have not",
    "isn't": "is not",
    "isn’t": "is not",
    "isnt": "is not",
    "isn t": "is not",
    "wasn't": "was not",
    "wasn’t": "was not",
    "wasnt": "was not",
    "wasn t": "was not",
    "weren't": "were not",
    "weren’t": "were not",
    "werent": "were not",
    "weren t": "were not",
}


def _normalized_source_gate_polarity_text(text: str) -> str:
    """Normalize identifier separators and bounded English contractions."""

    normalized = re.sub(r"_+", " ", text.casefold())
    for contraction, replacement in _SOURCE_GATE_CONTRACTION_REPLACEMENTS.items():
        normalized = re.sub(
            rf"\b{re.escape(contraction)}\b",
            replacement,
            normalized,
        )
    normalized = re.sub(r"\bnot\s+only\b", "only", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


@dataclass(frozen=True)
class _SourceGatePredicateReading:
    base_predicate: str
    negative: bool | None


_SOURCE_GATE_WORD = re.compile(r"[a-z]+|0")
_SOURCE_GATE_CLAUSE_BOUNDARY = re.compile(
    r"[;.!?]+|\b(?:whereas|however|but|although|though)\b",
)
_SOURCE_GATE_ENTITY_PATTERN = (
    "(?:" + "|".join(map(re.escape, sorted(_SOURCE_GATE_ENTITIES))) + ")"
)
_SOURCE_GATE_INDEPENDENT_COORDINATION = re.compile(
    r"\b(?:and|or|nor)\b(?=\s+(?:(?:the|a|an)\s+)?"
    rf"(?:{_SOURCE_GATE_ENTITY_PATTERN}|it)\b)",
)
_SOURCE_GATE_CONJUNCTIVE_SEPARATOR = re.compile(
    r"\b(?:and|nor)\b",
    flags=re.IGNORECASE,
)
_SOURCE_GATE_YET = re.compile(r"\byet\b", flags=re.IGNORECASE)
_SOURCE_GATE_YET_PREFIX_CHARACTER_LIMIT = 32
_SOURCE_GATE_YET_SUBJECT_CHARACTER_LIMIT = 96
_SOURCE_GATE_YET_SUBJECT_TOKEN_LIMIT = 8
_SOURCE_GATE_YET_PRONOUNS = frozenset({"he", "it", "she", "they"})
_SOURCE_GATE_AMBIGUOUS_YET_MARKER = "sourcegateambiguousyet"
_SOURCE_GATE_INCOME_BASES = frozenset({"income", "earning", "earnings"})
_SOURCE_GATE_INCOME_BLOCKERS = frozenset(
    {
        "document",
        "documentation",
        "evidence",
        "proof",
        "record",
        "report",
        "reported",
        "reporting",
        "reports",
        "tax",
        "taxed",
        "taxes",
    }
)
_SOURCE_GATE_DEPENDENT_BLOCKERS = frozenset(
    {
        "live",
        "lived",
        "lives",
        "living",
        "report",
        "reported",
        "reporting",
        "reports",
        "responsible",
        "responsibility",
    }
)
_SOURCE_GATE_INCOME_MODIFIERS = (
    r"(?:(?:adjusted|annual|earned|federal|gross|kentucky|state|taxable|total)\s+)*"
)
_SOURCE_GATE_INCOME_NEGATIVE_PATTERNS = (
    rf"(?:no|zero|0|nil)\s+{_SOURCE_GATE_INCOME_MODIFIERS}(?:income|earnings?)",
    rf"devoid\s+of\s+{_SOURCE_GATE_INCOME_MODIFIERS}(?:income|earnings?)",
    rf"without\s+{_SOURCE_GATE_INCOME_MODIFIERS}(?:income|earnings?)",
    rf"(?:an?\s+)?absence\s+of\s+{_SOURCE_GATE_INCOME_MODIFIERS}"
    rf"(?:income|earnings?)",
    rf"lack(?:s|ed|ing)?\s+{_SOURCE_GATE_INCOME_MODIFIERS}(?:income|earnings?)",
    rf"never\s+(?:has|have|had|receive(?:d|s)?|earn(?:ed|s)?)\s+"
    rf"{_SOURCE_GATE_INCOME_MODIFIERS}(?:income|earnings?)",
    rf"fail(?:s|ed)?\s+to\s+(?:receive|earn)\s+"
    rf"{_SOURCE_GATE_INCOME_MODIFIERS}(?:income|earnings?)",
    rf"(?:do|does|did|has|have|had)\s+not\s+"
    rf"(?:have|receive|earn)\s+{_SOURCE_GATE_INCOME_MODIFIERS}"
    rf"(?:income|earnings?)",
    rf"not\s+in\s+receipt\s+of\s+{_SOURCE_GATE_INCOME_MODIFIERS}"
    rf"(?:income|earnings?)",
)
_SOURCE_GATE_INCOME_POSITIVE_PATTERNS = (
    rf"(?:has|have|had|receive(?:d|s)?|earn(?:ed|s)?)\s+"
    rf"{_SOURCE_GATE_INCOME_MODIFIERS}(?:income|earnings?)",
)
_SOURCE_GATE_INCOME_NEGATION_MARKERS = frozenset(
    {
        "0",
        "absence",
        "devoid",
        "fail",
        "failed",
        "fails",
        "lack",
        "lacked",
        "lacking",
        "lacks",
        "never",
        "nil",
        "no",
        "not",
        "without",
        "zero",
    }
)
_SOURCE_GATE_DEPENDENT_OBJECT = (
    r"(?:(?:a|an|another|any|other|the)\s+)?"
    r"(?:(?:other|taxpayer|taxpayers|taxpayer\s+s)\s+)*dependent"
)
_SOURCE_GATE_DEPENDENT_NEGATIVE_PATTERNS = (
    rf"(?:is|are|was|were)\s+not\s+{_SOURCE_GATE_DEPENDENT_OBJECT}",
    rf"(?:can\s+not|unable\s+to|ineligible\s+to)\s+be\s+claimed\s+"
    rf"(?:as\s+)?"
    rf"{_SOURCE_GATE_DEPENDENT_OBJECT}",
    rf"not\s+eligible\s+to\s+be\s+claimed\s+(?:as\s+)?"
    rf"{_SOURCE_GATE_DEPENDENT_OBJECT}",
)
_SOURCE_GATE_DEPENDENT_POSITIVE_PATTERNS = (
    rf"(?:is|are|was|were)\s+{_SOURCE_GATE_DEPENDENT_OBJECT}",
    rf"(?:(?:is|are|was|were)\s+)?claimed\s+(?:as\s+)?"
    rf"{_SOURCE_GATE_DEPENDENT_OBJECT}",
    rf"(?:can\s+be|eligible\s+to\s+be)\s+claimed\s+(?:as\s+)?"
    rf"{_SOURCE_GATE_DEPENDENT_OBJECT}",
)
_SOURCE_GATE_DEPENDENT_NEGATION_MARKERS = frozenset(
    {"cannot", "ineligible", "not", "unable"}
)


def _source_gate_polarity_clauses(text: str) -> tuple[str, ...]:
    normalized = _normalized_source_gate_polarity_text(text)
    normalized = _SOURCE_GATE_CLAUSE_BOUNDARY.sub("\n", normalized)
    normalized = _SOURCE_GATE_INDEPENDENT_COORDINATION.sub("\n", normalized)
    normalized = _source_gate_replace_guarded_yet(
        normalized,
        independent_replacement="\n",
        ambiguous_replacement=f"\n{_SOURCE_GATE_AMBIGUOUS_YET_MARKER} ",
    )
    return tuple(
        clause.strip(" ,:") for clause in normalized.splitlines() if clause.strip(" ,:")
    )


def _source_gate_yet_classification(text: str, match: re.Match[str]) -> str:
    """Classify one bounded temporal, independent, or ambiguous ``yet``."""

    prefix = text[
        max(0, match.start() - _SOURCE_GATE_YET_PREFIX_CHARACTER_LIMIT) : match.start()
    ]
    prefix_tokens = _SOURCE_GATE_WORD.findall(prefix.casefold())
    suffix = text[match.end() :]
    if prefix_tokens and prefix_tokens[-1] == "not":
        return "temporal"
    if re.match(r"^\s+to\b", suffix, flags=re.IGNORECASE):
        return "temporal"
    subject_tokens = _SOURCE_GATE_WORD.findall(
        suffix[:_SOURCE_GATE_YET_SUBJECT_CHARACTER_LIMIT].casefold()
    )[:_SOURCE_GATE_YET_SUBJECT_TOKEN_LIMIT]
    if (_SOURCE_GATE_ENTITIES | _SOURCE_GATE_YET_PRONOUNS).intersection(subject_tokens):
        return "independent"
    return "ambiguous"


def _source_gate_replace_guarded_yet(
    text: str,
    *,
    independent_replacement: str,
    ambiguous_replacement: str,
) -> str:
    """Replace bounded adversatives and poison ambiguous continuations."""

    pieces: list[str] = []
    prior_end = 0
    for match in _SOURCE_GATE_YET.finditer(text):
        classification = _source_gate_yet_classification(text, match)
        if classification == "temporal":
            continue
        if classification == "ambiguous":
            replacement = ambiguous_replacement
        else:
            replacement = independent_replacement
        pieces.extend((text[prior_end : match.start()], replacement))
        prior_end = match.end()
    pieces.append(text[prior_end:])
    return "".join(pieces)


def _source_gate_split_conjunctive_conditions(text: str) -> list[str]:
    """Split conjunctions and the same bounded adversative boundaries."""

    with_yet_boundaries = _source_gate_replace_guarded_yet(
        text,
        independent_replacement="\n",
        ambiguous_replacement=f"\n{_SOURCE_GATE_AMBIGUOUS_YET_MARKER} ",
    )
    return [
        part
        for line in with_yet_boundaries.splitlines()
        for part in _SOURCE_GATE_CONJUNCTIVE_SEPARATOR.split(line)
    ]


def _source_gate_unwrap_negation(tokens: list[str]) -> tuple[list[str], int]:
    """Remove only clause-leading sentential negations and return their parity."""

    parity = 0
    while tokens:
        wrapper_length = 0
        if tokens[:3] == ["not", "true", "that"]:
            wrapper_length = 3
        elif tokens[:4] == ["not", "the", "case", "that"]:
            wrapper_length = 4
        elif tokens[:5] == ["it", "is", "not", "true", "that"]:
            wrapper_length = 5
        elif tokens[:6] == ["it", "is", "not", "the", "case", "that"]:
            wrapper_length = 6
        if not wrapper_length:
            break
        parity ^= 1
        tokens = tokens[wrapper_length:]
    return tokens, parity


def _source_gate_tail_match_start(
    tokens: Sequence[str],
    *,
    end: int,
    patterns: Sequence[str],
) -> int | None:
    start = max(0, end - _SOURCE_GATE_PREDICATE_SCAN_LIMIT)
    tail = " ".join(tokens[start : end + 1])
    starts: list[int] = []
    for pattern in patterns:
        match = re.search(rf"(?P<construct>{pattern})$", tail)
        if match is None:
            continue
        starts.append(start + tail[: match.start("construct")].count(" "))
    return max(starts, default=None)


def _source_gate_local_blocked(
    tokens: Sequence[str],
    *,
    index: int,
    blockers: frozenset[str],
) -> bool:
    before = tokens[max(0, index - 4) : index]
    after = tokens[index + 1 : index + 4]
    return bool(blockers.intersection(before) or blockers.intersection(after))


def _source_gate_has_temporal_yet_prefix(
    tokens: Sequence[str],
    *,
    construct_start: int,
) -> bool:
    temporal_prefix = " ".join(tokens[max(0, construct_start - 8) : construct_start])
    return bool(
        re.search(
            r"(?:\bnot\s+yet(?:\s+[a-z]+){0,3}|"
            r"\b(?:has|have|had)\s+yet\s+to(?:\s+be)?)\s*$",
            temporal_prefix,
        )
    )


def _source_gate_income_reading(
    tokens: Sequence[str],
    *,
    index: int,
    wrapper_parity: int,
    out_of_window_negation: bool,
    has_temporal_yet: bool,
) -> _SourceGatePredicateReading | None:
    if _source_gate_local_blocked(
        tokens,
        index=index,
        blockers=_SOURCE_GATE_INCOME_BLOCKERS,
    ):
        return _SourceGatePredicateReading("income", None)
    prefix_start = max(0, index - _SOURCE_GATE_PREDICATE_SCAN_LIMIT)
    if out_of_window_negation:
        return _SourceGatePredicateReading("income", None)
    negative_start = _source_gate_tail_match_start(
        tokens,
        end=index,
        patterns=_SOURCE_GATE_INCOME_NEGATIVE_PATTERNS,
    )
    if negative_start is not None:
        direct_outer = negative_start > 0 and tokens[negative_start - 1] == "not"
        return _SourceGatePredicateReading(
            "income",
            not bool(wrapper_parity ^ direct_outer),
        )
    positive_start = _source_gate_tail_match_start(
        tokens,
        end=index,
        patterns=_SOURCE_GATE_INCOME_POSITIVE_PATTERNS,
    )
    if positive_start is not None:
        if (
            _source_gate_has_temporal_yet_prefix(
                tokens,
                construct_start=positive_start,
            )
            or has_temporal_yet
        ):
            return _SourceGatePredicateReading("income", None)
        return _SourceGatePredicateReading("income", bool(wrapper_parity))
    local_prefix = tokens[prefix_start:index]
    if _SOURCE_GATE_INCOME_NEGATION_MARKERS.intersection(local_prefix):
        return _SourceGatePredicateReading("income", None)
    if index + 1 < len(tokens) and tokens[index + 1] == "fact":
        return None
    return _SourceGatePredicateReading("income", None)


def _source_gate_dependent_reading(
    tokens: Sequence[str],
    *,
    index: int,
    wrapper_parity: int,
    out_of_window_negation: bool,
    has_temporal_yet: bool,
) -> _SourceGatePredicateReading | None:
    if _source_gate_local_blocked(
        tokens,
        index=index,
        blockers=_SOURCE_GATE_DEPENDENT_BLOCKERS,
    ):
        return _SourceGatePredicateReading("dependent", None)
    prefix_start = max(0, index - _SOURCE_GATE_PREDICATE_SCAN_LIMIT)
    if out_of_window_negation:
        return _SourceGatePredicateReading("dependent", None)
    negative_start = _source_gate_tail_match_start(
        tokens,
        end=index,
        patterns=_SOURCE_GATE_DEPENDENT_NEGATIVE_PATTERNS,
    )
    if negative_start is not None:
        direct_outer = negative_start > 0 and tokens[negative_start - 1] == "not"
        return _SourceGatePredicateReading(
            "dependent",
            not bool(wrapper_parity ^ direct_outer),
        )
    positive_start = _source_gate_tail_match_start(
        tokens,
        end=index,
        patterns=_SOURCE_GATE_DEPENDENT_POSITIVE_PATTERNS,
    )
    if positive_start is not None:
        if (
            _source_gate_has_temporal_yet_prefix(
                tokens,
                construct_start=positive_start,
            )
            or has_temporal_yet
        ):
            return _SourceGatePredicateReading("dependent", None)
        return _SourceGatePredicateReading("dependent", bool(wrapper_parity))
    local_prefix = tokens[prefix_start:index]
    if _SOURCE_GATE_DEPENDENT_NEGATION_MARKERS.intersection(local_prefix):
        return _SourceGatePredicateReading("dependent", None)
    if index + 1 < len(tokens) and tokens[index + 1] == "fact":
        return None
    return _SourceGatePredicateReading("dependent", None)


def _source_gate_predicate_readings(
    text: str,
) -> tuple[_SourceGatePredicateReading, ...]:
    readings: list[_SourceGatePredicateReading] = []
    for clause in _source_gate_polarity_clauses(text):
        clause_tokens, wrapper_parity = _source_gate_unwrap_negation(
            _SOURCE_GATE_WORD.findall(clause)
        )
        has_ambiguous_yet = _SOURCE_GATE_AMBIGUOUS_YET_MARKER in clause_tokens
        # Independent and ambiguous uses were replaced before tokenization, so
        # every literal ``yet`` remaining in this clause is temporal state.
        has_temporal_yet = "yet" in clause_tokens
        income_negation_prefix = [0]
        dependent_negation_prefix = [0]
        for token in clause_tokens:
            income_negation_prefix.append(
                income_negation_prefix[-1]
                + int(token in _SOURCE_GATE_INCOME_NEGATION_MARKERS)
            )
            dependent_negation_prefix.append(
                dependent_negation_prefix[-1]
                + int(token in _SOURCE_GATE_DEPENDENT_NEGATION_MARKERS)
            )
        for index, token in enumerate(clause_tokens):
            reading: _SourceGatePredicateReading | None = None
            prefix_start = max(0, index - _SOURCE_GATE_PREDICATE_SCAN_LIMIT)
            if token in _SOURCE_GATE_INCOME_BASES:
                reading = _source_gate_income_reading(
                    clause_tokens,
                    index=index,
                    wrapper_parity=wrapper_parity,
                    out_of_window_negation=(income_negation_prefix[prefix_start] > 0),
                    has_temporal_yet=has_temporal_yet,
                )
            elif token in {"dependence", "dependency", "dependent"}:
                reading = _source_gate_dependent_reading(
                    clause_tokens,
                    index=index,
                    wrapper_parity=wrapper_parity,
                    out_of_window_negation=(
                        dependent_negation_prefix[prefix_start] > 0
                    ),
                    has_temporal_yet=has_temporal_yet,
                )
            if reading is not None:
                if has_ambiguous_yet:
                    reading = _SourceGatePredicateReading(
                        reading.base_predicate,
                        None,
                    )
                readings.append(reading)
    return tuple(readings)


def _source_gate_predicate_polarities(text: str) -> dict[str, bool | None]:
    grouped: dict[str, list[bool | None]] = {}
    for reading in _source_gate_predicate_readings(text):
        grouped.setdefault(reading.base_predicate, []).append(reading.negative)
    polarities: dict[str, bool | None] = {}
    for base_predicate, values in grouped.items():
        known = {value for value in values if value is not None}
        polarities[base_predicate] = (
            next(iter(known)) if None not in values and len(known) == 1 else None
        )
    return polarities


def _source_gate_explicit_polarities(text: str) -> dict[str, bool]:
    """Return unambiguous base-predicate polarity, where true means negative."""

    return {
        base_predicate: negative
        for base_predicate, negative in _source_gate_predicate_polarities(text).items()
        if negative is not None
    }


def _source_conjunctive_fact_gates(
    text: str,
) -> tuple[tuple[frozenset[str], frozenset[str]], ...]:
    """Return distinct entity/predicate signatures from one condition."""

    conditional = re.search(
        r"\b(?:if|when|provided\s+that|unless)\b(?P<body>.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if conditional is None:
        return ()
    body = conditional.group("body")
    segments = _source_gate_split_conjunctive_conditions(body)
    if len(segments) < 2:
        return ()
    gates: list[tuple[frozenset[str], frozenset[str]]] = []
    inherited_entities: frozenset[str] = frozenset()
    for segment in segments:
        tokens = _source_gate_semantic_tokens(segment)
        explicit_entities = tokens & _SOURCE_GATE_ENTITIES
        is_subject_continuation = bool(
            re.match(
                r"^[\s,]*(?:if\s+)?(?:is|are|was|were|has|have|had)\b",
                segment,
                flags=re.IGNORECASE,
            )
        )
        if explicit_entities and not is_subject_continuation:
            inherited_entities = explicit_entities
        entities = (
            explicit_entities | inherited_entities
            if is_subject_continuation
            else explicit_entities or inherited_entities
        )
        predicates = tokens - _SOURCE_GATE_ENTITIES
        if not predicates:
            continue
        if not (
            _SOURCE_GATE_PREDICATE_LANGUAGE.search(segment)
            or predicates
            & (
                _SOURCE_GATE_NOMINAL_PREDICATES
                | _SOURCE_GATE_POLARITY_PREDICATES
                | _SOURCE_GATE_UNKNOWN_POLARITY_PREDICATES
            )
        ):
            continue
        gates.append((entities, predicates))
    return tuple(gates) if len(gates) >= 2 else ()


def _source_conjunctive_fact_gate_count(text: str) -> int:
    """Count distinct predicate/entity occurrences in one condition."""

    return len(_source_conjunctive_fact_gates(text))


def _rule_cited_source_paths(
    rule: Mapping[str, Any],
    *,
    branches: Sequence[SourceStructureBranch],
    corpus_citation_path: str,
) -> set[tuple[str, ...]]:
    source = str(rule.get("source") or "").strip()
    paths = _paths_from_source_reference(
        source,
        corpus_citation_path=corpus_citation_path,
    )
    if paths:
        return paths
    krs_section = re.search(r"/krs/(?P<section>[^/]+)", corpus_citation_path, re.I)
    if krs_section is None or not re.search(
        rf"\bKRS\s+{re.escape(krs_section.group('section'))}\b",
        source,
        flags=re.IGNORECASE,
    ):
        return set()
    suffix = source[re.search(r"\bKRS\b", source, re.I).end() :]
    candidates = {
        branch.path
        for branch in branches
        if branch.path
        and all(
            re.search(
                rf"(?:\(\s*{re.escape(part)}\s*\)|\b{re.escape(part)}[.)])", suffix
            )
            for part in branch.path
        )
    }
    if not candidates:
        return {()}
    maximum_depth = max(map(len, candidates))
    return {path for path in candidates if len(path) == maximum_depth}


def _source_proposition_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    boundaries = [0, len(text)]
    boundaries.extend(
        match.end()
        for match in re.finditer(
            r";|[.!?](?=\s+(?:[A-Z(]|\d+\s*[.)]))",
            text,
        )
    )
    boundaries.extend(
        match.start() for match in _SOURCE_INDEPENDENT_CONDITION_BOUNDARY.finditer(text)
    )
    ordered = sorted(set(boundaries))
    return (
        max(position for position in ordered if position <= start),
        min(position for position in ordered if position >= end),
    )


def _flattened_inline_dotted_marker_has_item_boundary(
    source_text: str,
    *,
    start: int,
    container_start: int,
) -> bool:
    """Require list punctuation or a physical-line boundary before a marker."""

    line_start = source_text.rfind("\n", container_start, start) + 1
    if not source_text[line_start:start].strip():
        return True
    prefix = source_text[max(container_start, start - 64) : start]
    return (
        re.search(
            r"(?:[:;.!?]\s*(?:(?:and|or)\s+)?)$",
            prefix,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _flattened_inline_dotted_list_has_structural_chapeau(
    source_text: str,
    *,
    start: int,
    parent: SourceStructureBranch,
) -> bool:
    """Require the entire bounded parent prefix to be an operative chapeau."""

    if (
        start <= parent.start
        or start - parent.start > _INLINE_DOTTED_LIST_CHAPEAU_LIMIT
    ):
        return False
    prefix = source_text[parent.start : start]
    chapeau_candidates = [prefix]

    label = parent.label.strip()
    if label:
        label_match = re.match(
            rf"^\s*{re.escape(label)}\s*",
            prefix,
            flags=re.IGNORECASE,
        )
        if label_match is not None:
            chapeau_candidates.append(prefix[label_match.end() :])

    if parent.path:
        path_pattern = r"\s*".join(
            rf"\(\s*{re.escape(part)}\s*\)" for part in parent.path
        )
        path_match = re.match(
            rf"^\s*{path_pattern}\s*",
            prefix,
            flags=re.IGNORECASE,
        )
        if path_match is not None:
            chapeau_candidates.append(prefix[path_match.end() :])

    return any(
        _INLINE_DOTTED_LIST_STRUCTURAL_CHAPEAU.fullmatch(candidate) is not None
        for candidate in chapeau_candidates
    )


def _flattened_inline_dotted_list_ownership(
    source_text: str,
    *,
    parent: SourceStructureBranch,
) -> tuple[tuple[SourceStructureBranch, ...], bool]:
    """Recover proof-only children from a proven flattened dotted list.

    Corpus extraction can flatten ``1.`` through ``N.`` children onto one
    physical line even when their parenthesized parent remains recognizable.
    Keep these virtual branches local to proof ownership: they disambiguate an
    explicit leaf citation without expanding the source-completeness surface.
    The boolean result marks multiple plausible lists or an over-budget scan.
    """

    if not parent.path:
        return (), False
    matches = tuple(
        itertools.islice(
            (
                match
                for match in _INLINE_DOTTED_NUMBER_MARKER.finditer(
                    source_text,
                    parent.start,
                    parent.end,
                )
                if _flattened_inline_dotted_marker_has_item_boundary(
                    source_text,
                    start=match.start(),
                    container_start=parent.start,
                )
            ),
            _INLINE_DOTTED_LIST_MATCH_LIMIT + 1,
        )
    )
    if len(matches) > _INLINE_DOTTED_LIST_MATCH_LIMIT:
        return (), True

    candidate_sequence_count = 0
    qualified_sequence: tuple[re.Match[str], ...] | None = None
    match_index = 0
    while match_index < len(matches):
        if int(matches[match_index].group("label")) != 1:
            match_index += 1
            continue
        has_structural_chapeau = _flattened_inline_dotted_list_has_structural_chapeau(
            source_text,
            start=matches[match_index].start(),
            parent=parent,
        )
        sequence = [matches[match_index]]
        expected = 2
        next_index = match_index + 1
        while next_index < len(matches):
            label = int(matches[next_index].group("label"))
            if label != expected:
                break
            sequence.append(matches[next_index])
            expected += 1
            next_index += 1
        if len(sequence) < _INLINE_DOTTED_LIST_MINIMUM_LENGTH:
            match_index += 1
            continue
        candidate_sequence_count += 1
        if has_structural_chapeau:
            qualified_sequence = tuple(sequence)
        if candidate_sequence_count > 1 and qualified_sequence is not None:
            return (), True
        match_index = next_index
    if qualified_sequence is None:
        return (), False

    branches: list[SourceStructureBranch] = []
    sequence = qualified_sequence
    next_index = matches.index(sequence[-1]) + 1
    sequence_end = (
        matches[next_index].start() if next_index < len(matches) else parent.end
    )
    for index, match in enumerate(sequence):
        start = match.start()
        end = sequence[index + 1].start() if index + 1 < len(sequence) else sequence_end
        label = match.group("label").lower()
        branches.append(
            SourceStructureBranch(
                (*parent.path, label),
                "number",
                match.group("marker"),
                source_text[start:end].strip(),
                start,
                end,
            )
        )
    return tuple(branches), False


def _source_condition_clauses_owned_by_excerpt(
    excerpt: str,
    *,
    rule: Mapping[str, Any],
    source_text: str,
    branches: Sequence[SourceStructureBranch],
    corpus_citation_path: str,
) -> tuple[tuple[_SourceConditionClause, ...], bool]:
    """Resolve exact proof text to rule-cited propositions, reporting ambiguity."""

    excerpt_text = _collapse_text(excerpt)
    if not excerpt_text:
        return (), False
    pattern = re.compile(
        r"\s+".join(re.escape(part) for part in excerpt_text.split()),
        flags=re.IGNORECASE,
    )
    matches = tuple(pattern.finditer(source_text))
    if not matches:
        return (), False
    virtual_branches: dict[tuple[tuple[str, ...], str, int], SourceStructureBranch] = {}
    ambiguous_inline_ownership = False
    inline_ownership_by_parent: dict[
        tuple[tuple[str, ...], str, int],
        tuple[tuple[SourceStructureBranch, ...], bool],
    ] = {}
    for match in matches:
        containing_parents = sorted(
            (
                branch
                for branch in branches
                if branch.path
                and branch.start <= match.start()
                and match.end() <= branch.end
            ),
            key=lambda branch: (-len(branch.path), branch.end - branch.start),
        )
        for parent in containing_parents:
            parent_key = (parent.path, parent.kind, parent.start)
            ownership = inline_ownership_by_parent.get(parent_key)
            if ownership is None:
                ownership = _flattened_inline_dotted_list_ownership(
                    source_text,
                    parent=parent,
                )
                inline_ownership_by_parent[parent_key] = ownership
            owned_branches, ambiguous = ownership
            if ambiguous:
                ambiguous_inline_ownership = True
                break
            if not any(
                branch.start <= match.start() and match.end() <= branch.end
                for branch in owned_branches
            ):
                continue
            virtual_branches.update(
                {
                    (branch.path, branch.kind, branch.start): branch
                    for branch in owned_branches
                }
            )
            break
    ownership_branches = (*branches, *virtual_branches.values())
    cited_paths = _rule_cited_source_paths(
        rule,
        branches=ownership_branches,
        corpus_citation_path=corpus_citation_path,
    )

    def containing_branch(start: int, end: int) -> SourceStructureBranch | None:
        candidates = [
            branch
            for branch in ownership_branches
            if branch.start <= start and end <= branch.end
        ]
        return max(
            candidates,
            key=lambda branch: (len(branch.path), -(branch.end - branch.start)),
            default=None,
        )

    located = [(match, containing_branch(*match.span())) for match in matches]
    restricted = [
        (match, branch)
        for match, branch in located
        if not cited_paths
        or any(
            path == ()
            or branch is not None
            and len(branch.path) >= len(path)
            and branch.path[: len(path)] == path
            for path in cited_paths
        )
    ]
    citation_mismatch = bool(cited_paths and located and not restricted)
    selected = restricted or located
    clauses: dict[tuple[tuple[str, ...], int, int], _SourceConditionClause] = {}
    for match, branch in selected:
        branch_path = branch.path if branch is not None else ()
        container_start = branch.start if branch is not None else 0
        container_end = branch.end if branch is not None else len(source_text)
        if branch is not None:
            # A parent chapeau commonly ends with a colon rather than sentence
            # punctuation.  Keep its proposition local to the parent instead
            # of absorbing the first structural child into the same condition.
            # A proof excerpt located inside a child already resolves to that
            # more-specific branch above, so this only bounds true chapeaux.
            container_end = min(
                (
                    candidate.start
                    for candidate in ownership_branches
                    if len(candidate.path) == len(branch.path) + 1
                    and candidate.path[: len(branch.path)] == branch.path
                    and match.end() <= candidate.start < branch.end
                ),
                default=container_end,
            )
        container_text = source_text[container_start:container_end]
        local_start = match.start() - container_start
        local_end = match.end() - container_start
        proposition_start, proposition_end = _source_proposition_bounds(
            container_text,
            local_start,
            local_end,
        )
        absolute_start = container_start + proposition_start
        absolute_end = container_start + proposition_end
        identity = (branch_path, absolute_start, absolute_end)
        clauses[identity] = _SourceConditionClause(
            branch_path,
            absolute_start,
            absolute_end,
            container_text[proposition_start:proposition_end].strip(" ;,"),
        )
    ordered = tuple(
        clauses[key]
        for key in sorted(clauses, key=lambda item: (item[1], item[2], item[0]))
    )
    return (
        ordered,
        citation_mismatch or ambiguous_inline_ownership or len(ordered) != 1,
    )


def _effective_formula_version_intervals(
    rule: Mapping[str, Any],
) -> tuple[tuple[int, str, str | None, str | None], ...]:
    """Return unambiguous intervals in which each formula version is selected."""

    versions = rule.get("versions")
    if (
        not isinstance(versions, list)
        or len(versions) > _TEMPORAL_WITNESS_VERSION_LIMIT
    ):
        return ()
    formulas = [
        (index, version, str(version.get("formula")))
        for index, version in enumerate(versions)
        if isinstance(version, dict) and version.get("formula") is not None
    ]
    if not formulas:
        return ()
    has_temporal_metadata = any(
        str(version.get("effective_from") or "").strip()
        or str(version.get("effective_to") or "").strip()
        for _index, version, _formula in formulas
    )
    if not has_temporal_metadata:
        if len(formulas) != 1:
            return ()
        index, _version, formula = formulas[0]
        return ((index, formula, None, None),)

    starts: list[str] = []
    for _index, version, _formula in formulas:
        start = str(version.get("effective_from") or "").strip()
        end = str(version.get("effective_to") or "").strip()
        if (
            not _is_iso_calendar_date(start)
            or (end and not _is_iso_calendar_date(end))
            or (end and end < start)
        ):
            return ()
        starts.append(start)
    if len(set(starts)) != len(starts):
        return ()

    intervals: list[tuple[int, str, str | None, str | None]] = []
    for index, version, formula in formulas:
        start = str(version.get("effective_from") or "").strip()
        explicit_end = str(version.get("effective_to") or "").strip() or None
        next_start = min((value for value in starts if value > start), default=None)
        selected_end = explicit_end
        if next_start is not None:
            superseded_end = _shift_iso_date(next_start, -1)
            if superseded_end is None:
                return ()
            selected_end = min(
                (
                    value
                    for value in (explicit_end, superseded_end)
                    if value is not None
                ),
                default=None,
            )
        if selected_end is not None and selected_end < start:
            continue
        intervals.append((index, formula, start, selected_end))
    return tuple(intervals)


def _formula_intervals_overlap(
    left_start: str | None,
    left_end: str | None,
    right_start: str | None,
    right_end: str | None,
) -> bool:
    return not (
        left_end is not None
        and right_start is not None
        and left_end < right_start
        or right_end is not None
        and left_start is not None
        and right_end < left_start
    )


def _formula_interval_intersection(
    left_start: str | None,
    left_end: str | None,
    right_start: str | None,
    right_end: str | None,
) -> tuple[str | None, str | None] | None:
    if not _formula_intervals_overlap(
        left_start,
        left_end,
        right_start,
        right_end,
    ):
        return None
    starts = [value for value in (left_start, right_start) if value is not None]
    ends = [value for value in (left_end, right_end) if value is not None]
    return (max(starts, default=None), min(ends, default=None))


def _terminal_gate_evidence(
    payload: Mapping[str, Any],
) -> dict[str, _TerminalGateEvidence]:
    evidence: dict[str, _TerminalGateEvidence] = {}

    def build(name: str, description: str = "") -> _TerminalGateEvidence:
        name_text = name.replace("_", " ")
        name_tokens = _source_gate_semantic_tokens(name_text)
        description_tokens = _source_gate_semantic_tokens(description)
        name_polarities = _source_gate_predicate_polarities(name_text)
        description_polarities = _source_gate_predicate_polarities(description)
        polarity_conflict = (
            None in name_polarities.values()
            or None in description_polarities.values()
            or any(
                predicate in description_polarities
                and description_polarities[predicate] != negative
                for predicate, negative in name_polarities.items()
            )
        )
        effective_polarities = description_polarities | name_polarities
        tokens = set(name_tokens | description_tokens)
        for base_predicate, negative in effective_polarities.items():
            if negative is None:
                polarity_predicate = f"unknown_{base_predicate}"
            else:
                polarity_predicate = (
                    "no_income" if base_predicate == "income" else "not_dependent"
                )
            tokens.discard(base_predicate)
            tokens.discard(f"unknown_{base_predicate}")
            if negative is None:
                tokens.add(polarity_predicate)
            else:
                tokens.add(polarity_predicate if negative else base_predicate)
        return _TerminalGateEvidence(
            frozenset(tokens) & _SOURCE_GATE_ENTITIES,
            frozenset(tokens) - _SOURCE_GATE_ENTITIES,
            polarity_conflict,
            frozenset(
                (base_predicate, negative)
                for base_predicate, negative in name_polarities.items()
                if negative is not None
            ),
        )

    for item in payload.get("inputs", ()) or ():
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        evidence[name] = build(name, str(item.get("description") or ""))
    for item in payload.get("imports", ()) or ():
        if not isinstance(item, str) or "#" not in item:
            continue
        name = item.rsplit("#", 1)[-1].strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        evidence[name] = build(name)
    return evidence


def _terminal_names_corroborate_source_gates(
    names: frozenset[str],
    gates: tuple[tuple[frozenset[str], frozenset[str]], ...],
    *,
    terminal_evidence: Mapping[str, _TerminalGateEvidence],
    neutral_polarity_imports: frozenset[str] = frozenset(),
) -> bool:
    """Require a distinct semantically related terminal for every source gate."""

    if not gates:
        return True
    if (
        len(gates) > _SOURCE_EXPLICIT_CONDITION_EXPANSION_LIMIT
        or len(names) > _SOURCE_EXPLICIT_CONDITION_EXPANSION_LIMIT
    ):
        return False
    repeated_predicates = [
        frozenset().union(
            *(
                predicates & other_predicates
                for other_index, (_entities, other_predicates) in enumerate(gates)
                if other_index != index
            )
        )
        for index, (_entities, predicates) in enumerate(gates)
    ]
    candidates: list[tuple[str, ...]] = []
    for index, (entities, predicates) in enumerate(gates):
        if predicates & _SOURCE_GATE_UNKNOWN_POLARITY_PREDICATES:
            candidates.append(())
            continue
        matches: list[str] = []
        for name in sorted(names):
            evidence = terminal_evidence.get(
                name,
                _TerminalGateEvidence(frozenset(), frozenset(), True),
            )
            if evidence.polarity_conflict:
                continue
            terminal_entities = evidence.entities
            terminal_predicates = evidence.predicates
            source_administrative = predicates & _SOURCE_GATE_ADMINISTRATIVE_PREDICATES
            terminal_administrative = (
                terminal_predicates & _SOURCE_GATE_ADMINISTRATIVE_PREDICATES
            )
            if terminal_administrative and not (
                terminal_administrative & source_administrative
            ):
                continue
            is_neutral_import_fact = False
            polarity_mismatch = False
            for base_predicate, negative_predicate in (
                ("income", "no_income"),
                ("dependent", "not_dependent"),
            ):
                source_negative = negative_predicate in predicates
                source_positive = base_predicate in predicates
                if not source_negative and not source_positive:
                    continue
                terminal_negative = negative_predicate in terminal_predicates
                terminal_positive = base_predicate in terminal_predicates
                explicit_name_polarities = dict(evidence.explicit_name_polarities)
                neutral_for_base = (
                    source_negative
                    and name in neutral_polarity_imports
                    and "fact" in terminal_predicates
                    and base_predicate not in explicit_name_polarities
                    and terminal_positive
                )
                is_neutral_import_fact |= neutral_for_base
                if source_negative:
                    polarity_mismatch |= not terminal_negative and not neutral_for_base
                else:
                    polarity_mismatch |= terminal_negative or not terminal_positive
            if polarity_mismatch:
                continue
            predicate_overlap = predicates & terminal_predicates
            if is_neutral_import_fact:
                predicate_overlap |= {
                    _SOURCE_GATE_POLARITY_BASE[predicate]
                    for predicate in predicates & _SOURCE_GATE_POLARITY_PREDICATES
                } & terminal_predicates
            entity_overlap = entities & terminal_entities
            if not predicate_overlap:
                continue
            if len(predicate_overlap) < 2 and not entity_overlap:
                continue
            if repeated_predicates[index] and entities and not entity_overlap:
                continue
            matches.append(name)
        candidates.append(tuple(matches))
    if any(not values for values in candidates):
        return False
    order = sorted(range(len(gates)), key=lambda index: (len(candidates[index]), index))

    matched_gate_by_terminal: dict[str, int] = {}

    def augment(gate_index: int, seen: set[str]) -> bool:
        for name in candidates[gate_index]:
            if name in seen:
                continue
            seen.add(name)
            previous_gate = matched_gate_by_terminal.get(name)
            if previous_gate is None or augment(previous_gate, seen):
                matched_gate_by_terminal[name] = gate_index
                return True
        return False

    return all(augment(gate_index, set()) for gate_index in order)


def _opaque_same_source_condition_input_issues(
    payload: Mapping[str, Any],
    *,
    source_text: str,
    branches: Sequence[SourceStructureBranch],
    principal_rules: Mapping[str, dict[str, Any]],
    corpus_citation_path: str,
) -> list[str]:
    """Reject terminal selectors that collapse explicit same-source fact gates."""

    inputs = payload.get("inputs")
    declared_inputs = {
        str(item.get("name") or "").strip()
        for item in inputs or ()
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    imported_names = {
        str(item).rsplit("#", 1)[-1].strip()
        for item in payload.get("imports", ()) or ()
        if isinstance(item, str)
        and "#" in item
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item.rsplit("#", 1)[-1].strip())
    }
    named_rules = {
        str(rule.get("name") or "").strip(): rule
        for rule in payload.get("rules", ()) or ()
        if isinstance(rule, dict) and str(rule.get("name") or "").strip()
    }
    terminal_evidence = _terminal_gate_evidence(payload)

    authoritative_path = corpus_citation_path.strip("/").casefold()
    numeric_or_judgment_dtypes = {
        "count",
        "decimal",
        "integer",
        "judgment",
        "money",
        "rate",
    }
    interval_cache = {
        name: _effective_formula_version_intervals(rule)
        for name, rule in named_rules.items()
    }
    selector_cache: dict[tuple[str, bool], tuple[frozenset[str], ...]] = {}
    source_gate_cache: dict[
        tuple[tuple[str, ...], int, int],
        tuple[tuple[frozenset[str], frozenset[str]], ...],
    ] = {}
    expansion_cache: dict[
        tuple[str, int, str | None, str | None],
        tuple[_TerminalGateAlternative, ...],
    ] = {}
    dependency_budget_cache: dict[
        tuple[str, str, bool, str | None, str | None], bool
    ] = {}

    def formula_selector_groups(
        formula: str,
        *,
        include_result_leaves: bool,
    ) -> tuple[frozenset[str], ...]:
        key = (formula, include_result_leaves)
        if key not in selector_cache:
            selector_cache[key] = _formula_control_selector_name_groups(
                formula,
                include_result_leaves=include_result_leaves,
            )
        return selector_cache[key]

    def source_clause_gates(
        clause: _SourceConditionClause,
    ) -> tuple[tuple[frozenset[str], frozenset[str]], ...]:
        identity = (clause.branch_path, clause.start, clause.end)
        if identity not in source_gate_cache:
            source_gate_cache[identity] = _source_conjunctive_fact_gates(clause.text)
        return source_gate_cache[identity]

    def dependency_expansion_is_bounded(
        root_name: str,
        formula: str,
        *,
        include_result_leaves: bool,
        start: str | None,
        end: str | None,
    ) -> bool:
        """Bound the temporally reachable dependency graph before recursion."""

        cache_key = (root_name, formula, include_result_leaves, start, end)
        if cache_key in dependency_budget_cache:
            return dependency_budget_cache[cache_key]
        initial_names = frozenset().union(
            *formula_selector_groups(
                formula,
                include_result_leaves=include_result_leaves,
            )
        )
        pending = [
            (False, name, start, end) for name in sorted(initial_names, reverse=True)
        ]
        dependencies: set[str] = set()
        active_names = {root_name}
        completed_states: set[tuple[str, str | None, str | None]] = set()
        while pending:
            exiting, name, dependency_start, dependency_end = pending.pop()
            state = (name, dependency_start, dependency_end)
            if exiting:
                active_names.remove(name)
                completed_states.add(state)
                continue
            if (
                state in completed_states
                or name in declared_inputs
                or name in imported_names
            ):
                continue
            dependency = named_rules.get(name)
            if dependency is None:
                dependency_budget_cache[cache_key] = False
                return False
            if name in active_names:
                dependency_budget_cache[cache_key] = False
                return False
            dependencies.add(name)
            if len(dependencies) > _SOURCE_EXPLICIT_CONDITION_DEPENDENCY_NODE_LIMIT:
                dependency_budget_cache[cache_key] = False
                return False
            dependency_includes_leaves = str(
                dependency.get("dtype") or ""
            ).strip().lower() in {"judgment", "boolean", "bool"}
            child_states: set[tuple[str, str | None, str | None]] = set()
            for (
                _index,
                dependency_formula,
                version_start,
                version_end,
            ) in interval_cache.get(name, ()):
                overlap = _formula_interval_intersection(
                    dependency_start,
                    dependency_end,
                    version_start,
                    version_end,
                )
                if overlap is None:
                    continue
                for group in formula_selector_groups(
                    dependency_formula,
                    include_result_leaves=dependency_includes_leaves,
                ):
                    derived_dependencies = {
                        selector_name
                        for selector_name in group
                        if selector_name not in declared_inputs
                        and selector_name not in imported_names
                    }
                    if any(
                        selector_name not in named_rules
                        for selector_name in derived_dependencies
                    ):
                        dependency_budget_cache[cache_key] = False
                        return False
                    child_states.update(
                        (selector_name, overlap[0], overlap[1])
                        for selector_name in derived_dependencies
                    )
            active_names.add(name)
            pending.append((True, name, dependency_start, dependency_end))
            pending.extend(
                (False, child_name, child_start, child_end)
                for child_name, child_start, child_end in sorted(
                    child_states,
                    key=lambda value: (
                        value[0],
                        value[1] or "",
                        value[2] or "~",
                    ),
                    reverse=True,
                )
            )
        dependency_budget_cache[cache_key] = True
        return True

    def bounded_alternatives(
        values: Iterable[_TerminalGateAlternative],
        *,
        fallback_start: str | None,
        fallback_end: str | None,
    ) -> tuple[_TerminalGateAlternative, ...]:
        unique = sorted(
            set(values),
            key=lambda value: (
                value.resolved,
                value.start or "",
                value.end or "~",
                len(value.names),
                tuple(sorted(value.names)),
            ),
        )
        minimal = [
            value
            for value in unique
            if not any(
                candidate.start == value.start
                and candidate.end == value.end
                and candidate.resolved == value.resolved
                and candidate.names < value.names
                for candidate in unique
            )
        ]
        if len(minimal) <= _SOURCE_EXPLICIT_CONDITION_EXPANSION_LIMIT:
            return tuple(minimal)
        return tuple(
            (
                *minimal[: _SOURCE_EXPLICIT_CONDITION_EXPANSION_LIMIT - 1],
                _TerminalGateAlternative(
                    frozenset(),
                    fallback_start,
                    fallback_end,
                    False,
                ),
            )
        )

    def expand_name(
        name: str,
        *,
        start: str | None,
        end: str | None,
        stack: frozenset[str],
    ) -> tuple[_TerminalGateAlternative, ...]:
        if name in declared_inputs or name in imported_names:
            return (_TerminalGateAlternative(frozenset({name}), start, end),)
        if name in stack:
            return (_TerminalGateAlternative(frozenset(), start, end, False),)
        if len(stack) >= _SOURCE_EXPLICIT_CONDITION_DEPENDENCY_DEPTH_LIMIT:
            return (_TerminalGateAlternative(frozenset(), start, end, False),)
        dependency = named_rules.get(name)
        if dependency is None:
            return (_TerminalGateAlternative(frozenset(), start, end, False),)
        choices: list[_TerminalGateAlternative] = []
        saw_overlap = False
        for version_index, formula, version_start, version_end in interval_cache.get(
            name, ()
        ):
            overlap = _formula_interval_intersection(
                start,
                end,
                version_start,
                version_end,
            )
            if overlap is None:
                continue
            saw_overlap = True
            overlap_start, overlap_end = overlap
            key = (
                name,
                version_index,
                overlap_start,
                overlap_end,
            )
            # Preflight proves the dependency graph acyclic, so an otherwise
            # identical expansion does not depend on its traversal path.
            cached = expansion_cache.get(key)
            if cached is None:
                cached = expand_formula(
                    formula,
                    start=overlap_start,
                    end=overlap_end,
                    stack=stack | {name},
                    include_result_leaves=(
                        str(dependency.get("dtype") or "").strip().lower()
                        in {"judgment", "boolean", "bool"}
                    ),
                )
                expansion_cache[key] = cached
            choices.extend(cached)
        if not saw_overlap:
            return (_TerminalGateAlternative(frozenset(), start, end, False),)
        return bounded_alternatives(
            choices,
            fallback_start=start,
            fallback_end=end,
        )

    def expand_formula(
        formula: str,
        *,
        start: str | None,
        end: str | None,
        stack: frozenset[str],
        include_result_leaves: bool,
    ) -> tuple[_TerminalGateAlternative, ...]:
        expanded_groups: list[_TerminalGateAlternative] = []
        for selector_group in formula_selector_groups(
            formula,
            include_result_leaves=include_result_leaves,
        ):
            choices: tuple[_TerminalGateAlternative, ...] = (
                _TerminalGateAlternative(frozenset(), start, end),
            )
            for selector_name in sorted(selector_group):
                selector_choices = expand_name(
                    selector_name,
                    start=start,
                    end=end,
                    stack=stack,
                )
                combined: list[_TerminalGateAlternative] = []
                for left in choices:
                    for right in selector_choices:
                        overlap = _formula_interval_intersection(
                            left.start,
                            left.end,
                            right.start,
                            right.end,
                        )
                        if overlap is None:
                            continue
                        combined.append(
                            _TerminalGateAlternative(
                                left.names | right.names,
                                overlap[0],
                                overlap[1],
                                left.resolved and right.resolved,
                            )
                        )
                choices = bounded_alternatives(
                    combined,
                    fallback_start=start,
                    fallback_end=end,
                )
            expanded_groups.extend(choices)
        return bounded_alternatives(
            expanded_groups,
            fallback_start=start,
            fallback_end=end,
        )

    proof_excerpts: dict[str, dict[int, tuple[str, ...]]] = {}
    for rule_name, rule in named_rules.items():
        formula_versions = interval_cache.get(rule_name, ())
        formula_indexes = {index for index, _formula, _start, _end in formula_versions}
        by_version: dict[int, list[str]] = {}
        for path, citation_path, excerpt in _rule_source_excerpt_atoms(rule):
            if citation_path.strip("/").casefold() != authoritative_path:
                continue
            indexed = re.fullmatch(r"versions\[(\d+)\]\.formula", path)
            if indexed is not None:
                version_index = int(indexed.group(1))
                if version_index in formula_indexes:
                    by_version.setdefault(version_index, []).append(excerpt)
                continue
            if path == "versions.formula" and len(formula_indexes) == 1:
                by_version.setdefault(next(iter(formula_indexes)), []).append(excerpt)
        proof_excerpts[rule_name] = {
            index: tuple(dict.fromkeys(excerpts))
            for index, excerpts in by_version.items()
        }

    findings: list[tuple[str, int, int, frozenset[str]]] = []
    for rule_name, rule in sorted(principal_rules.items()):
        if (
            str(rule.get("dtype") or "").strip().lower()
            not in numeric_or_judgment_dtypes
        ):
            continue
        for version_index, formula, start, end in interval_cache.get(rule_name, ()):
            excerpts = proof_excerpts.get(rule_name, {}).get(version_index, ())
            gate_sets: dict[
                tuple[tuple[str, ...], int, int],
                tuple[tuple[frozenset[str], frozenset[str]], ...],
            ] = {}
            ambiguous_ownership = False
            for excerpt in excerpts:
                owned_clauses, ambiguous = _source_condition_clauses_owned_by_excerpt(
                    excerpt,
                    rule=rule,
                    source_text=source_text,
                    branches=branches,
                    corpus_citation_path=corpus_citation_path,
                )
                excerpt_has_gates = False
                for clause in owned_clauses:
                    gates = source_clause_gates(clause)
                    if len(gates) < 2:
                        continue
                    excerpt_has_gates = True
                    gate_sets[(clause.branch_path, clause.start, clause.end)] = gates
                ambiguous_ownership |= ambiguous and excerpt_has_gates
            gate_count = max((len(gates) for gates in gate_sets.values()), default=0)
            if gate_count < 2:
                continue
            include_result_leaves = str(rule.get("dtype") or "").strip().lower() in {
                "judgment",
                "boolean",
                "bool",
            }
            selector_groups = formula_selector_groups(
                formula,
                include_result_leaves=include_result_leaves,
            )
            direct_selectors = frozenset().union(*selector_groups)
            if dependency_expansion_is_bounded(
                rule_name,
                formula,
                include_result_leaves=include_result_leaves,
                start=start,
                end=end,
            ):
                terminal_choices = expand_formula(
                    formula,
                    start=start,
                    end=end,
                    stack=frozenset({rule_name}),
                    include_result_leaves=include_result_leaves,
                )
            else:
                terminal_choices = (
                    _TerminalGateAlternative(frozenset(), start, end, False),
                )
            if not terminal_choices:
                # A formula that is provably inactive on every path cannot grant
                # the source benefit while bypassing its factual conditions.
                continue
            failing_choices = [
                choice
                for choice in terminal_choices
                if not choice.resolved
                or any(
                    not _terminal_names_corroborate_source_gates(
                        choice.names,
                        gates,
                        terminal_evidence=terminal_evidence,
                        neutral_polarity_imports=frozenset(imported_names),
                    )
                    for gates in gate_sets.values()
                )
            ]
            if not failing_choices and not ambiguous_ownership:
                continue
            least_supported = min(
                failing_choices or terminal_choices,
                key=lambda value: (
                    value.resolved,
                    len(value.names),
                    tuple(sorted(value.names)),
                ),
            )
            findings.append(
                (
                    rule_name,
                    version_index,
                    gate_count,
                    direct_selectors | least_supported.names,
                )
            )
    if not findings:
        return []

    findings.sort(key=lambda item: (item[0], item[1], tuple(sorted(item[3]))))
    rendered = [
        f"`{rule_name}` versions[{version_index}] "
        f"({_bounded_identifier_feedback(tuple(evidence_names)) or 'no resolved fact selectors'} "
        f"for {gate_count} source gates)"
        for rule_name, version_index, gate_count, evidence_names in findings[
            :_SOURCE_EXPLICIT_CONDITION_DIAGNOSTIC_LIMIT
        ]
    ]
    if len(findings) > _SOURCE_EXPLICIT_CONDITION_DIAGNOSTIC_LIMIT:
        rendered.append(
            f"... ({len(findings) - _SOURCE_EXPLICIT_CONDITION_DIAGNOSTIC_LIMIT} "
            "additional formula versions omitted)"
        )
    return [
        "[complete-source-unit:source-explicit-conditions] Derived formula "
        "version(s) delegate multiple conjunctive factual gates stated in their "
        "exact authoritative source clause to too few terminal local inputs or "
        f"canonical imports: {'; '.join(rendered)}. Encode the source-stated facts "
        "as distinct local inputs or canonical imports, combine them in a "
        "source-proved derived Judgment, and test each gate. Reserve aggregate "
        "boundary statuses for source-atomic or externally determined conditions."
    ]


def _direct_local_input_clamps(
    rule: Mapping[str, Any],
    declared_input_names: set[str],
) -> set[str]:
    clamped: set[str] = set()
    versions = rule.get("versions")
    if not isinstance(versions, list):
        return clamped
    for version in versions:
        formula = version.get("formula") if isinstance(version, dict) else None
        if not isinstance(formula, str):
            continue
        clamped.update(
            match.group("input")
            for match in _DIRECT_LOCAL_INPUT_NONNEGATIVE_CLAMP.finditer(formula)
            if match.group("input") in declared_input_names
        )
    return clamped


def _selected_rule_formula_is_local_rule_alias(
    rule: dict[str, Any],
    *,
    case: dict[str, Any],
    principal_rule_names: set[str],
) -> bool:
    """Return whether the case-selected formula is a bare local-rule alias."""

    formula = _rule_formula_text_for_case(rule, case)
    if formula is None:
        return False
    return (
        isinstance(expression := _parse_formula_expression(formula), ast.Name)
        and expression.id in principal_rule_names
    )


_CLAMP_STRUCTURAL_LEAF_LIMIT = 64
_CLAMP_TEMPORAL_CONTEXT_LIMIT = 128


def _formula_structural_leaves(formula: str) -> tuple[str, ...] | None:
    """Expand bounded control flow into every computational formula leaf.

    Clamp claimant competition must see an inactive computation branch before
    the case selector chooses another branch. Returning ``None`` fails closed
    when a malformed or oversized branch tree cannot be inspected completely.
    """

    leaves: set[str] = set()

    def expand(text: str, *, depth: int = 0) -> bool:
        if depth > 32:
            return False
        selected_text = text.strip()
        node = _first_formula_branch_node(selected_text)
        if node is None:
            if _rule_text_has_branching_formula(selected_text) or not (
                _formula_leaf_has_executable_syntax(selected_text)
            ):
                return False
            leaves.add(selected_text)
            return len(leaves) <= _CLAMP_STRUCTURAL_LEAF_LIMIT
        if not all(
            _formula_text_has_executable_branch_tree(selector)
            for selector in node.selectors
        ) or not all(
            _formula_text_has_executable_branch_tree(choice) for choice in node.choices
        ):
            return False
        for choice in node.choices:
            selected_body = textwrap.dedent(choice).strip()
            if not selected_body or not expand(
                selected_text[: node.start] + selected_body + selected_text[node.end :],
                depth=depth + 1,
            ):
                return False
        return True

    return tuple(sorted(leaves)) if expand(formula) else None


def _selected_formula_structural_source_branches(
    rule: dict[str, Any],
    *,
    case: dict[str, Any],
    formula_branches: Sequence[SourceStructureBranch],
    source_formula_intervals: Mapping[
        SourceStructureBranch,
        _NumericInterval | None,
    ],
    formula_environment: dict[str, Any],
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> tuple[SourceStructureBranch, ...]:
    """Find selected-version source computations without trusting ownership."""

    formula = _rule_formula_text_for_case(rule, case)
    if formula is None:
        return ()
    leaves = _formula_structural_leaves(formula)
    if leaves is None:
        return tuple(formula_branches)
    constant_environment = _formula_environment_for_case(formula_environment, case)
    candidate_executions = tuple(
        _FormulaExecution(
            trace=(),
            leaf=leaf,
            evaluated_value=None,
            evaluates_to_zero=False,
            constant_environment=constant_environment,
        )
        for leaf in leaves
    )
    executions = tuple(
        execution
        for execution in candidate_executions
        if _formula_execution_leaf_is_computational(execution)
    )
    return tuple(
        branch
        for branch in formula_branches
        if any(
            _formula_execution_matches_source_branch(
                execution,
                branch,
                interval=source_formula_intervals[branch],
                formula_environment=formula_environment,
                execution_environment=constant_environment,
                extract_numeric_occurrences=extract_numeric_occurrences,
                numeric_value_is_grounded=numeric_value_is_grounded,
            )
            for execution in executions
        )
    )


def _negative_local_input_clamp_test_issues(
    principal_rules: dict[str, dict[str, Any]],
    *,
    parameter_rules: Mapping[str, dict[str, Any]],
    principal_formula_clause_rules: dict[SourceStructureBranch, set[str]],
    formula_branches: Sequence[SourceStructureBranch],
    cases: Sequence[dict[str, Any]],
    declared_input_names: set[str],
    formula_environment: dict[str, Any],
    corpus_citation_path: str,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> list[str]:
    """Require negative evidence for direct max(0, local_input) clamps."""

    principal_rule_names = set(principal_rules)
    source_formula_intervals = {
        branch: _formula_branch_interval(
            branch,
            extract_numeric_occurrences=extract_numeric_occurrences,
        )
        for branch in formula_branches
    }
    source_formula_branches_by_rule = {
        name: tuple(
            branch
            for branch in formula_branches
            if name in principal_formula_clause_rules.get(branch, set())
        )
        for name in principal_rules
    }
    selected_dependencies_by_period: dict[str, dict[str, set[str]]] = {}
    selected_source_branches_by_period: dict[
        str,
        dict[str, tuple[SourceStructureBranch, ...]],
    ] = {}
    selected_structural_source_branches_by_period: dict[
        str,
        dict[str, tuple[SourceStructureBranch, ...]],
    ] = {}
    asserted_dependencies_by_case: dict[int, dict[str, Any]] = {}
    execution_environments_by_case: dict[int, dict[str, Any] | None] = {}

    def selected_period_context(
        case: dict[str, Any],
    ) -> tuple[
        dict[str, set[str]],
        dict[str, tuple[SourceStructureBranch, ...]],
        dict[str, tuple[SourceStructureBranch, ...]],
    ]:
        """Cache selected dependencies, competitors, and proof owners."""

        period = _normalized_case_period(case)
        if period not in selected_dependencies_by_period:
            selected_dependencies_by_period[period] = {
                name: (
                    set(_FORMULA_IDENTIFIER.findall(formula)) & principal_rule_names
                    if (formula := _rule_formula_text_for_case(rule, case)) is not None
                    else set()
                )
                for name, rule in principal_rules.items()
            }
            selected_source_branches_by_period[period] = {
                name: tuple(
                    branch
                    for branch in owned_branches
                    if _selected_principal_formula_has_exact_source_proof(
                        principal_rules[name],
                        case=case,
                        branch=branch,
                        corpus_citation_path=corpus_citation_path,
                    )
                )
                for name, owned_branches in source_formula_branches_by_rule.items()
                if owned_branches
                and not _selected_rule_formula_is_local_rule_alias(
                    principal_rules[name],
                    case=case,
                    principal_rule_names=principal_rule_names,
                )
            }
            selected_structural_source_branches_by_period[period] = {
                name: tuple(
                    dict.fromkeys(
                        (
                            *_selected_formula_structural_source_branches(
                                rule,
                                case=case,
                                formula_branches=formula_branches,
                                source_formula_intervals=source_formula_intervals,
                                formula_environment=formula_environment,
                                extract_numeric_occurrences=(
                                    extract_numeric_occurrences
                                ),
                                numeric_value_is_grounded=numeric_value_is_grounded,
                            ),
                            *selected_source_branches_by_period[period].get(name, ()),
                        )
                    )
                )
                for name, rule in principal_rules.items()
                if not _selected_rule_formula_is_local_rule_alias(
                    rule,
                    case=case,
                    principal_rule_names=principal_rule_names,
                )
                and _rule_formula_text_for_case(rule, case) is not None
            }
        return (
            selected_dependencies_by_period[period],
            selected_structural_source_branches_by_period[period],
            selected_source_branches_by_period[period],
        )

    all_version_dependencies = {
        name: set(_FORMULA_IDENTIFIER.findall(_rule_formula_text(rule)))
        & principal_rule_names
        for name, rule in principal_rules.items()
    }
    parameter_rule_names = set(parameter_rules)

    def bounded_dependency_closure(
        seeds: Iterable[str],
        *,
        dependencies: Mapping[str, set[str]],
    ) -> set[str] | None:
        """Return a bounded principal dependency closure or fail closed."""

        closure: set[str] = set()
        pending = list(seeds)
        while pending:
            current = pending.pop()
            if current in closure:
                continue
            closure.add(current)
            if len(closure) > _CLAMP_TEMPORAL_CONTEXT_LIMIT:
                return None
            pending.extend(dependencies.get(current, set()) - closure)
        return closure

    def temporal_control_names(
        rule_formulas: Iterable[tuple[dict[str, Any], str]],
    ) -> set[str] | None:
        """Return bounded temporal parameter selectors or fail closed.

        Parameter aliases are already flattened into ``formula_environment``;
        retaining the selected alias version and value therefore also captures
        changes in a transitive parameter dependency.  A referenced parameter
        missing from that environment is ambiguous (including cyclic or
        oversized alias graphs), so it cannot safely share a clamp witness.
        """

        selector_names: set[str] = set()

        def clamp_formula_control_names(
            rule: dict[str, Any],
            formula: str,
        ) -> set[str] | None:
            names: set[str] = set()
            inspect_boolean_leaf = str(rule.get("dtype") or "").strip().lower() in {
                "bool",
                "boolean",
                "judgment",
            }

            def inspect(text: str, *, depth: int = 0) -> bool:
                if depth > 32:
                    return False
                node = _first_formula_branch_node(text)
                if node is None:
                    if inspect_boolean_leaf:
                        names.update(_formula_exception_selector_names(text))
                    return True
                for selector in node.selectors:
                    names.update(_exception_condition_names(selector))
                if not all(
                    inspect(textwrap.dedent(choice).strip(), depth=depth + 1)
                    for choice in node.choices
                ):
                    return False
                remainder = text[: node.start] + text[node.end :]
                return not remainder.strip() or inspect(remainder, depth=depth + 1)

            return names if inspect(formula) else None

        for rule, formula in rule_formulas:
            formula_selector_names = clamp_formula_control_names(rule, formula)
            if formula_selector_names is None:
                return None
            selector_names.update(formula_selector_names)
            if len(selector_names) > _CLAMP_TEMPORAL_CONTEXT_LIMIT:
                return None
        referenced_parameters = selector_names & parameter_rule_names
        if any(name not in formula_environment for name in referenced_parameters):
            return None
        temporal_names = {
            name
            for name in referenced_parameters
            if isinstance(formula_environment[name], _TemporalFormulaValue)
        }
        return (
            temporal_names
            if len(temporal_names) <= _CLAMP_TEMPORAL_CONTEXT_LIMIT
            else None
        )

    def selected_temporal_control_topology(
        case: dict[str, Any],
        *,
        topology_names: set[str],
    ) -> tuple[tuple[str, int, str, str], ...] | None:
        """Identify selected temporal parameter controls for one topology."""

        rule_formulas: list[tuple[dict[str, Any], str]] = []
        for name in topology_names:
            rule = principal_rules[name]
            formula = _rule_formula_text_for_case(rule, case)
            if formula is None:
                return None
            rule_formulas.append((rule, formula))
        control_names = temporal_control_names(rule_formulas)
        if control_names is None:
            return None
        selected: list[tuple[str, int, str, str]] = []
        for name in sorted(control_names):
            temporal_value = formula_environment[name]
            indexes = _selected_temporal_version_indexes(temporal_value, case)
            if len(indexes) != 1:
                return None
            index = indexes[0]
            value = temporal_value.versions[index][2]
            selected.append((name, index, type(value).__name__, repr(value)))
        return tuple(selected)

    def clamp_context(
        case: dict[str, Any],
        *,
        owner: str,
        rule: dict[str, Any],
        input_name: str,
    ) -> (
        tuple[
            tuple[Any, ...],
            set[str],
            dict[str, tuple[SourceStructureBranch, ...]],
            bool,
        ]
        | None
    ):
        """Return one selected temporal claimant topology for a direct clamp."""

        selected_formula = _rule_formula_text_for_case(rule, case)
        if selected_formula is None or not any(
            match.group("input") == input_name
            for match in _DIRECT_LOCAL_INPUT_NONNEGATIVE_CLAMP.finditer(
                selected_formula
            )
        ):
            return None
        (
            selected_dependencies,
            selected_structural_source_branches,
            selected_source_branches,
        ) = selected_period_context(case)
        competing_downstream = {
            candidate
            for candidate, owned_branches in selected_structural_source_branches.items()
            if owned_branches
            and candidate != owner
            and _formula_rule_depends_on(
                candidate,
                owner,
                dependencies=selected_dependencies,
            )
        }
        downstream_branch_owners: dict[SourceStructureBranch, set[str]] = {}
        for candidate in competing_downstream:
            for branch in selected_structural_source_branches[candidate]:
                downstream_branch_owners.setdefault(branch, set()).add(candidate)
        unambiguous_downstream_source_branches = {
            candidate: tuple(
                branch
                for branch in selected_source_branches[candidate]
                if len(downstream_branch_owners.get(branch, ())) == 1
            )
            for candidate in competing_downstream & set(selected_source_branches)
        }
        downstream = {
            candidate
            for candidate, owned_branches in (
                unambiguous_downstream_source_branches.items()
            )
            if owned_branches
        }
        principal_outputs = downstream if competing_downstream else {owner}

        topology_names = bounded_dependency_closure(
            {owner, *competing_downstream},
            dependencies=selected_dependencies,
        )
        if topology_names is None:
            return ((), set(), {}, False)
        temporal_control_topology = selected_temporal_control_topology(
            case,
            topology_names=topology_names,
        )
        if temporal_control_topology is None:
            return ((), set(), {}, False)
        dependency_topology = tuple(
            (
                name,
                _selected_rule_formula_version_index(principal_rules[name], case),
                _rule_formula_text_for_case(principal_rules[name], case),
                tuple(sorted(selected_dependencies.get(name, set()) & topology_names)),
            )
            for name in sorted(topology_names)
        )
        claimant_topology = tuple(
            (
                candidate,
                selected_structural_source_branches[candidate],
                selected_source_branches.get(candidate, ()),
            )
            for candidate in sorted(competing_downstream)
        )
        return (
            (
                dependency_topology,
                claimant_topology,
                temporal_control_topology,
            ),
            principal_outputs,
            unambiguous_downstream_source_branches,
            True,
        )

    def temporal_context_cases(owner: str) -> tuple[dict[str, str], ...] | None:
        """Return bounded representatives for owner/downstream change segments."""

        potential_claimants = {
            name
            for name in principal_rules
            if name == owner
            or _formula_rule_depends_on(
                name,
                owner,
                dependencies=all_version_dependencies,
            )
        }
        potentially_relevant = bounded_dependency_closure(
            potential_claimants,
            dependencies=all_version_dependencies,
        )
        if potentially_relevant is None:
            return None
        all_version_rule_formulas: list[tuple[dict[str, Any], str]] = []
        for name in potentially_relevant:
            rule = principal_rules[name]
            versions = rule.get("versions")
            if not isinstance(versions, list):
                return None
            all_version_rule_formulas.extend(
                (rule, str(version["formula"]))
                for version in versions
                if isinstance(version, dict) and version.get("formula") is not None
            )
        all_version_temporal_controls = temporal_control_names(
            all_version_rule_formulas
        )
        if all_version_temporal_controls is None:
            return None
        change_points = {
            period
            for case in cases
            if _is_iso_calendar_date(period := _normalized_case_period(case))
        }
        for name in potentially_relevant:
            versions = principal_rules[name].get("versions")
            if not isinstance(versions, list):
                continue
            for version in versions:
                if not isinstance(version, dict) or version.get("formula") is None:
                    continue
                start = str(version.get("effective_from") or "").strip()
                end = str(version.get("effective_to") or "").strip()
                if _is_iso_calendar_date(start):
                    change_points.add(start)
                if (
                    end
                    and _is_iso_calendar_date(end)
                    and (after_end := _shift_iso_date(end, 1)) is not None
                ):
                    change_points.add(after_end)
                if len(change_points) > _CLAMP_TEMPORAL_CONTEXT_LIMIT:
                    return None
        for name in all_version_temporal_controls:
            temporal_value = formula_environment[name]
            for start, end, _value in temporal_value.versions:
                if not _is_iso_calendar_date(start) or (
                    end and not _is_iso_calendar_date(end)
                ):
                    return None
                change_points.add(start)
                if end and (after_end := _shift_iso_date(end, 1)) is not None:
                    change_points.add(after_end)
                if len(change_points) > _CLAMP_TEMPORAL_CONTEXT_LIMIT:
                    return None
        return tuple({"period": point} for point in sorted(change_points))

    missing_clamps: set[tuple[str, str]] = set()
    for owner, rule in principal_rules.items():
        clamped_inputs = _direct_local_input_clamps(rule, declared_input_names)
        if not clamped_inputs:
            continue
        context_cases = temporal_context_cases(owner)
        for input_name in sorted(clamped_inputs):
            if context_cases is None:
                missing_clamps.add((owner, input_name))
                continue
            required_contexts: set[tuple[Any, ...]] = set()
            unresolved_context = False
            for context_case in context_cases:
                context = clamp_context(
                    context_case,
                    owner=owner,
                    rule=rule,
                    input_name=input_name,
                )
                if context is None:
                    continue
                context_key, _outputs, _branches, resolved = context
                if not resolved:
                    unresolved_context = True
                    break
                required_contexts.add(context_key)
            if unresolved_context:
                missing_clamps.add((owner, input_name))
                continue
            if not required_contexts:
                missing_clamps.add((owner, input_name))
                continue
            witnessed_contexts: set[tuple[Any, ...]] = set()
            for case in cases:
                inputs = case.get("input")
                if not isinstance(inputs, dict):
                    continue
                has_negative = any(
                    input_name in _input_key_names(key)
                    and any(
                        value < 0 for value in _numeric_test_input_values(raw_value)
                    )
                    for key, raw_value in inputs.items()
                )
                if not has_negative:
                    continue
                if not _is_iso_calendar_date(_normalized_case_period(case)):
                    continue
                context = clamp_context(
                    case,
                    owner=owner,
                    rule=rule,
                    input_name=input_name,
                )
                if context is None:
                    continue
                (
                    context_key,
                    principal_outputs,
                    unambiguous_branches,
                    resolved,
                ) = context
                if not resolved:
                    continue
                outputs = _test_case_output_names(case)
                if owner not in outputs:
                    continue
                case_key = id(case)
                if case_key not in asserted_dependencies_by_case:
                    asserted_dependencies_by_case[case_key] = (
                        _case_asserted_dependency_environment(
                            principal_rules,
                            case,
                            formula_environment=formula_environment,
                        )
                    )
                dependency_environment = asserted_dependencies_by_case[case_key]
                if owner not in dependency_environment:
                    continue
                for principal_output in sorted(outputs & principal_outputs):
                    if principal_output == owner:
                        witnessed_contexts.add(context_key)
                        break
                    asserted_principal_value = dependency_environment.get(
                        principal_output,
                        _UNRESOLVED_CONDITION_VALUE,
                    )
                    if asserted_principal_value is _UNRESOLVED_CONDITION_VALUE:
                        continue
                    execution = _case_formula_execution(
                        principal_rules[principal_output],
                        case,
                        formula_environment=formula_environment,
                        dependency_environment=dependency_environment,
                    )
                    if execution is None:
                        continue
                    executed_principal_value = _formula_execution_runtime_value(
                        execution
                    )
                    if (
                        executed_principal_value is _UNRESOLVED_CONDITION_VALUE
                        or not _formula_runtime_values_equal(
                            executed_principal_value,
                            asserted_principal_value,
                        )
                    ):
                        continue
                    if case_key not in execution_environments_by_case:
                        execution_environments_by_case[case_key] = (
                            _case_formula_identifier_environment(
                                case,
                                formula_environment=formula_environment,
                                dependency_environment=dependency_environment,
                            )
                        )
                    execution_environment = execution_environments_by_case[case_key]
                    if execution_environment is None:
                        continue
                    if not any(
                        _formula_execution_matches_source_branch(
                            execution,
                            branch,
                            interval=source_formula_intervals[branch],
                            formula_environment=formula_environment,
                            execution_environment=execution_environment,
                            extract_numeric_occurrences=extract_numeric_occurrences,
                            numeric_value_is_grounded=numeric_value_is_grounded,
                        )
                        for branch in unambiguous_branches[principal_output]
                    ):
                        continue
                    reached_rules = _asserted_reached_rule_executions(
                        principal_rules[principal_output],
                        execution,
                        case=case,
                        principal_rules=principal_rules,
                        formula_environment=formula_environment,
                        dependency_environment=dependency_environment,
                    )
                    if any(
                        reached_rule is principal_rules[owner]
                        for reached_rule, _reached_execution in reached_rules[1:]
                    ):
                        witnessed_contexts.add(context_key)
                        break
            if required_contexts <= witnessed_contexts:
                continue
            missing_clamps.add((owner, input_name))
    if not missing_clamps:
        return []
    ordered_clamps = sorted(missing_clamps)
    shown_clamps = ordered_clamps[:_NEGATIVE_LOCAL_INPUT_CLAMP_DIAGNOSTIC_LIMIT]
    rendered_clamps = ", ".join(
        f"`max(0, {input_name})` in `{owner}`" for owner, input_name in shown_clamps
    )
    omitted = len(ordered_clamps) - len(shown_clamps)
    omission = f" ({omitted} additional clamps omitted)" if omitted else ""
    return [
        "[complete-source-unit:tests] Direct nonnegative clamp companion "
        f"evidence is missing for {rendered_clamps}{omission}. Each requires "
        "an executed companion case in every distinct selected temporal "
        "owner/downstream claimant context, with the input below zero, that "
        "asserts the clamped rule and, when downstream consumers exist, an "
        "unambiguous authoritative principal output with selected-version "
        "formula proof whose matching assertion and selected formula path "
        "reach it through a resolved dependency chain. A zero-valued case "
        "does not exercise the negative clamp."
    ]


def _formula_rule_depends_on(
    candidate: str,
    dependency: str,
    *,
    dependencies: Mapping[str, set[str]],
) -> bool:
    pending = list(dependencies.get(candidate, set()))
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current == dependency:
            return True
        if current in seen:
            continue
        seen.add(current)
        pending.extend(dependencies.get(current, set()) - seen)
    return False


def _companion_test_issues(
    principal_rules: dict[str, dict[str, Any]],
    *,
    parameter_rules: Mapping[str, dict[str, Any]],
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
    source_bound_constant_occurrences: dict[str, tuple[NumericOccurrenceLike, ...]],
    declared_input_names: set[str],
) -> list[str]:
    issues: list[str] = []
    cases = [case for case in (test_cases or ()) if isinstance(case, dict)]
    if not cases:
        return [
            "[complete-source-unit:tests] Complete source-unit controls require "
            "a companion test suite covering outputs, branches, boundaries, "
            "exceptions, and rounding rules."
        ]

    colliding_cases = [
        case
        for case in cases
        if _case_input_principal_output_collisions(case, set(principal_rules))
    ]
    for case in colliding_cases:
        collisions = ", ".join(
            sorted(
                _case_input_principal_output_collisions(
                    case,
                    set(principal_rules),
                )
            )
        )
        issues.append(
            "[complete-source-unit:tests] Companion case "
            f"`{case.get('name') or '<unnamed>'}` supplies local principal "
            f"output(s) {collisions} as inputs; derived outputs must be "
            "computed and asserted, not shadowed."
        )
    cases = [case for case in cases if case not in colliding_cases]

    asserted_by_rule = {
        name: [case for case in cases if name in _test_case_output_names(case)]
        for name in principal_rules
    }
    issues.extend(
        _negative_local_input_clamp_test_issues(
            principal_rules,
            parameter_rules=parameter_rules,
            principal_formula_clause_rules=principal_formula_clause_rules,
            formula_branches=formula_branches,
            cases=cases,
            declared_input_names=declared_input_names,
            formula_environment=formula_environment,
            corpus_citation_path=corpus_citation_path,
            extract_numeric_occurrences=extract_numeric_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
        )
    )
    for name, asserted_cases in asserted_by_rule.items():
        if not asserted_cases:
            issues.append(
                "[complete-source-unit:tests] Principal output "
                f"`{name}` is never asserted by a companion test."
            )

    formula_dependency_cache: dict[int, dict[str, Any]] = {}
    formula_execution_cache: dict[tuple[str, int], _FormulaExecution | None] = {}
    permissive_dependency_cache: dict[tuple[str, int], dict[str, Any]] = {}
    permissive_execution_cache: dict[tuple[str, int], _FormulaExecution | None] = {}
    active_branches = [
        branch
        for branch in branches
        if not _path_is_deferred(branch.path, deferred_paths)
    ]
    missing_formula_branches = _unwitnessed_formula_branches(
        formula_branches,
        corpus_citation_path=corpus_citation_path,
        principal_rules=principal_rules,
        principal_formula_clause_rules=principal_formula_clause_rules,
        asserted_by_rule=asserted_by_rule,
        extract_numeric_occurrences=extract_numeric_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
        formula_environment=formula_environment,
        dependency_cache=formula_dependency_cache,
        execution_cache=formula_execution_cache,
    )
    for branch in missing_formula_branches:
        source_excerpt = _bounded_source_feedback_excerpt(branch.text)
        source_location = f"characters {branch.start}:{branch.end}"
        unbound_diagnostic = _unbound_matching_formula_rules(
            branch,
            corpus_citation_path=corpus_citation_path,
            principal_rules=principal_rules,
            bound_rule_names=principal_formula_clause_rules[branch],
            asserted_by_rule=asserted_by_rule,
            extract_numeric_occurrences=extract_numeric_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
            formula_environment=formula_environment,
            dependency_cache=formula_dependency_cache,
            execution_cache=formula_execution_cache,
        )
        source_binding_detail = _unbound_formula_binding_feedback(
            unbound_diagnostic,
            corpus_citation_path=corpus_citation_path,
        )
        dependency_diagnostic = _uncorroborated_formula_dependency_diagnostic(
            branch,
            principal_rules=principal_rules,
            bound_rule_names=principal_formula_clause_rules[branch],
            asserted_by_rule=asserted_by_rule,
            extract_numeric_occurrences=extract_numeric_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
            formula_environment=formula_environment,
            strict_dependency_cache=formula_dependency_cache,
            strict_execution_cache=formula_execution_cache,
            permissive_dependency_cache=permissive_dependency_cache,
            permissive_execution_cache=permissive_execution_cache,
        )
        dependency_detail = _uncorroborated_formula_dependency_feedback(
            dependency_diagnostic,
        )
        interval = _formula_branch_interval(
            branch,
            extract_numeric_occurrences=extract_numeric_occurrences,
        )
        witness_requirement = (
            "The asserted principal formula must execute this source computation "
            "in its legally applicable companion-case period."
            if interval is None
            else "The asserted principal formula must execute this source "
            "computation in its legally applicable companion-case period, and "
            "the case must supply a numeric selector inside the source-stated "
            "range."
        )
        observed_periods = sorted(
            {
                period
                for rule_name in principal_formula_clause_rules[branch]
                for case in asserted_by_rule.get(rule_name, ())
                if re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}",
                    period := _normalized_case_period(case),
                )
            }
        )
        observed_period_detail = (
            " Observed asserted candidate-case periods: "
            + _bounded_period_feedback(observed_periods)
            + "."
            if observed_periods
            else " No asserted candidate principal-output case has a usable period."
        )
        issues.append(
            "[complete-source-unit:tests] Companion tests do not demonstrate "
            f"formula branch {branch.label} at "
            f"{_branch_citation(corpus_citation_path, branch)}. Exact source "
            f"computation ({source_location}): `{source_excerpt}`. "
            f"{witness_requirement}{observed_period_detail}"
            f"{source_binding_detail}{dependency_detail} Each formula branch "
            "needs distinct executed "
            "test evidence; an internal formula-clause ordinal is not a legal "
            "paragraph number."
        )

    boundary_branches = _active_or_root_source_branches(
        source_text,
        branches=branches,
        active_branches=active_branches,
        deferred_paths=deferred_paths,
    )
    boundary_branches = tuple(
        branch
        for branch in boundary_branches
        if not _is_marker_only_container(
            branch,
            branches=branches,
            source_text=source_text,
        )
    )
    boundary_obligations = _source_boundary_obligations(
        boundary_branches,
        narrative_formula_branches=formula_branches,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    missing_boundaries: list[tuple[SourceStructureBranch, NumericOccurrenceLike]] = []
    for branch, boundary in boundary_obligations:
        if _branch_boundary_test_witnesses(
            branch,
            boundary,
            principal_rules=principal_rules,
            principal_rule_paths=principal_rule_paths,
            asserted_by_rule=asserted_by_rule,
            numeric_value_is_grounded=numeric_value_is_grounded,
            formula_environment=formula_environment,
            source_bound_constant_occurrences=source_bound_constant_occurrences,
            extract_numeric_occurrences=extract_numeric_occurrences,
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
        formula_branches=formula_branches,
    )
    if exception_branches:
        independently_covered_paths = {
            path
            for paths in principal_rule_paths.values()
            for path in paths
            if any(candidate.path == path for candidate in active_branches)
        }
        paired_exception_branches = tuple(
            branch
            for branch in exception_branches
            if _source_exception_requires_paired_witness(
                branch.text,
                branch_path=branch.path,
                independently_covered_paths=independently_covered_paths,
            )
        )
        unconditional_nonapplicability_branches = tuple(
            branch
            for branch in exception_branches
            if _source_unconditional_nonapplicability(branch.text)
        )
        toggled_exception_selectors = _toggled_formula_boolean_selectors(
            principal_rules,
            asserted_by_rule=asserted_by_rule,
            formula_environment=formula_environment,
        )
        missing_exception_branches = _unwitnessed_exception_branches(
            paired_exception_branches,
            principal_rule_paths=principal_rule_paths,
            toggled_exception_selectors=toggled_exception_selectors,
            extract_numeric_occurrences=extract_numeric_occurrences,
        )
        if missing_exception_branches:
            missing_conditions = "; ".join(
                f"{_branch_citation(corpus_citation_path, branch)} "
                f"[{_source_exception_effect_requirement(branch.text)}]: `"
                f"{_bounded_source_feedback_excerpt(branch.text, limit=220)}`"
                for branch in missing_exception_branches
            )
            issues.append(
                "[complete-source-unit:tests] Source-stated exceptions or "
                "applicability conditions require paired positive/blocking cases "
                "that assert the affected principal output and toggle its "
                "controlling formula selector. Each listed condition needs its "
                "own same-period case pair differing in exactly that one input; "
                f"missing: {missing_conditions}."
            )
        missing_unconditional_branches = _unmatched_evidence_obligations(
            {
                branch: _unconditional_nonapplicability_witnesses(
                    branch,
                    corpus_citation_path=corpus_citation_path,
                    principal_rules=principal_rules,
                    principal_rule_paths=principal_rule_paths,
                    asserted_by_rule=asserted_by_rule,
                    formula_environment=formula_environment,
                )
                for branch in unconditional_nonapplicability_branches
            }
        )
        if missing_unconditional_branches:
            missing_conditions = "; ".join(
                f"{_branch_citation(corpus_citation_path, branch)}: `"
                f"{_bounded_source_feedback_excerpt(branch.text, limit=220)}`"
                for branch in missing_unconditional_branches
            )
            issues.append(
                "[complete-source-unit:tests] Unconditional source-stated "
                "non-applicability requires a source-bound principal Judgment "
                "output whose formula executes to false and is asserted false "
                f"by a companion case; missing: {missing_conditions}."
            )

    rounding_obligations = _source_rounding_obligations(
        source_text,
        branches=branches,
        active_branches=active_branches,
        deferred_paths=deferred_paths,
    )
    missing_rounding_formula: list[tuple[SourceStructureBranch, str]] = []
    rounding_witnesses: dict[
        tuple[SourceStructureBranch, str], set[tuple[str, str, str]]
    ] = {}
    require_rounding_clause_binding = len(rounding_obligations) > 1
    for obligation in rounding_obligations:
        branch, direction = obligation
        source_formula_branches = _rounding_source_formula_branches(
            branch,
            formula_branches=formula_branches,
        )
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
                    principal_rules=principal_rules,
                    asserted_by_rule=asserted_by_rule,
                    direction=direction,
                    formula_environment=formula_environment,
                    source_formula_branches=source_formula_branches,
                    rounding_refers_to_result=(
                        _rounding_text_refers_to_result(branch.text)
                    ),
                    require_clause_binding=require_rounding_clause_binding,
                    extract_numeric_occurrences=extract_numeric_occurrences,
                    numeric_value_is_grounded=numeric_value_is_grounded,
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
        nearest_guidance = (
            " A satisfying nearest/kaufmännisch shape is same-period paired "
            "cases with otherwise-identical inputs that drive the executed "
            "pre-rounding derived amount to 100.49 -> 100 and 100.50 -> 101; "
            "assert both the reached unrounded intermediate and the affected "
            "principal output in each case."
            if any(
                direction == "nearest" for _branch, direction in missing_rounding_tests
            )
            else ""
        )
        issues.append(
            "[complete-source-unit:tests] Companion tests do not demonstrate "
            "every source-stated rounding rule with distinct fractional input "
            f"evidence on its affected principal output; missing at {rendered}."
            f"{nearest_guidance}"
            " Alternative: when the rounded computation depends on outputs "
            "that are not yet available (for example a cross-module base this "
            "module cannot compute), a PRECISE typed deferral of that branch "
            "removes this demand — declare `module.deferred_outputs: "
            "[{output: <namespaced branch output>, reason: <names the exact "
            "missing dependency citations>}]` instead of encoding a "
            "speculative mechanism."
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


def _case_input_principal_output_collisions(
    case: dict[str, Any],
    principal_rule_names: set[str],
) -> set[str]:
    inputs = case.get("input")
    if not isinstance(inputs, dict):
        return set()
    return set().union(
        *(_input_key_names(key) & principal_rule_names for key in inputs),
        set(),
    )


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
        if _formula_clause_has_nonnumeric_inline_following_operands(
            clause,
            end=end,
            owner=owner,
            branches=branches,
            source_text=source_text,
        ):
            continue
        if _formula_clause_is_structural_chapeau(
            clause,
            end=end,
            owner=owner,
            branches=branches,
            source_text=source_text,
        ):
            continue
        obligation = SourceStructureBranch(
            owner.path,
            "formula-clause",
            f"{owner.label} formula clause {clause_index}",
            clause,
            start,
            end,
        )
        if _rounding_clause_refers_to_previous_result(
            obligation,
            previous=obligations[-1] if obligations else None,
            source_text=source_text,
        ):
            continue
        obligations.append(obligation)
    return tuple(obligations)


def _formula_clause_has_nonnumeric_inline_following_operands(
    clause: str,
    *,
    end: int,
    owner: SourceStructureBranch,
    branches: Sequence[SourceStructureBranch],
    source_text: str,
) -> bool:
    """Reject abstract inline lists mislabeled as following numeric values."""

    if not (
        clause.rstrip().endswith(":")
        and _FORMULA_DIRECT_FOLLOWING_OPERANDS.search(clause)
    ):
        return False
    descendants = tuple(
        branch
        for branch in branches
        if branch.start >= end
        and branch.end <= owner.end
        and len(branch.path) > len(owner.path)
        and branch.path[: len(owner.path)] == owner.path
    )
    next_descendant = min(
        (branch.start for branch in descendants),
        default=owner.end,
    )
    inline = source_text[end:next_descendant].strip().strip(".;:")
    if inline:
        return not _formula_inline_operands_are_numeric(inline)
    non_leaf_paths = {
        candidate.path[:depth]
        for candidate in descendants
        for depth in range(len(owner.path) + 1, len(candidate.path))
    }
    leaves = tuple(
        branch for branch in descendants if branch.path not in non_leaf_paths
    )
    leaf_computations = tuple(
        source_states_explicit_computation(branch.text) for branch in leaves
    )
    leaf_numeric_operands = tuple(
        _formula_inline_operands_are_numeric(_strip_source_clause_marker(branch.text))
        for branch in leaves
    )
    return bool(leaves) and not (any(leaf_computations) or all(leaf_numeric_operands))


def _formula_clause_is_structural_chapeau(
    clause: str,
    *,
    end: int,
    owner: SourceStructureBranch,
    branches: Sequence[SourceStructureBranch],
    source_text: str,
) -> bool:
    """Let a colon-ended computation heading delegate to its child rows."""

    heading = _FORMULA_NONOPERATIVE_TABLE_HEADING.search(clause)
    if (
        not clause.rstrip().endswith(":")
        or heading is None
        or _formula_clause_states_substantive_operation(clause[: heading.start()])
        or _formula_clause_states_substantive_operation(clause)
    ):
        return False
    descendants = tuple(
        branch
        for branch in branches
        if (
            len(branch.path) > len(owner.path)
            and branch.path[: len(owner.path)] == owner.path
            and branch.start >= end
            and branch.end <= owner.end
        )
    )
    if (
        not descendants
        or source_text[end : min(branch.start for branch in descendants)].strip()
    ):
        return False
    non_leaf_paths = {
        candidate.path[:depth]
        for candidate in descendants
        for depth in range(len(owner.path) + 1, len(candidate.path))
    }
    leaf_descendants = tuple(
        branch for branch in descendants if branch.path not in non_leaf_paths
    )
    return any(
        source_states_explicit_computation(branch.text) for branch in leaf_descendants
    )


def _formula_clause_states_substantive_operation(clause: str) -> bool:
    """Distinguish an operative formula from a non-operative table heading."""

    coordinate_starts = [
        match.start()
        for match in re.finditer(r",?\s+and\s+", clause, flags=re.IGNORECASE)
        if re.match(
            r"(?:(?![,.;:\n]).){0,80}\b(?:"
            r"(?:shall|must|may)\s+(?:be\s+)?equal|is\s+equal|equals?)\b",
            clause[match.end() :],
            flags=re.IGNORECASE,
        )
    ]
    operative_prefix = clause[: coordinate_starts[-1]] if coordinate_starts else clause
    operative_prefix = _strip_source_clause_marker(operative_prefix)
    return bool(
        _has_substantive_arithmetic_expression(clause)
        or _ENGLISH_WORDED_PERCENTAGE_OF.search(clause)
        or _EXPLICIT_NUMERIC_PERCENTAGE_OF.search(clause)
        or _english_fraction_of_is_computational(clause)
        or _FORMULA_COMPUTED_OPERATION_LANGUAGE.search(clause)
        or _formula_states_contextual_operator(clause, include_unconditional=False)
        or _formula_operation_has_numeric_operands(operative_prefix)
        or re.search(r"\btwice\s+(?:the\s+)?\w+", clause, flags=re.IGNORECASE)
        or re.search(
            r"\b(?:calculated|computed|determined)\s+(?:by|through)\s+"
            r"(?:applying|application|combining|doubling|taking)\b",
            clause,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\b(?:added|divided|multiplied|subtracted)\s+by\b",
            clause,
            flags=re.IGNORECASE,
        )
    )


def _source_control_branches(
    source_text: str,
    *,
    branches: Sequence[SourceStructureBranch],
    active_branches: Sequence[SourceStructureBranch],
    deferred_paths: set[tuple[str, ...]],
    formula_branches: Sequence[SourceStructureBranch],
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> tuple[SourceStructureBranch, ...]:
    """Return active boundary, exception, and applicability control clauses."""

    boundary_branches = _active_or_root_source_branches(
        source_text,
        branches=branches,
        active_branches=active_branches,
        deferred_paths=deferred_paths,
    )
    controlled = [
        branch
        for branch, boundary in _source_boundary_obligations(
            boundary_branches,
            narrative_formula_branches=formula_branches,
            extract_numeric_occurrences=extract_numeric_occurrences,
        )
        if not _source_boundary_is_temporal(branch, boundary)
    ]
    controlled.extend(
        _source_exception_branches(
            source_text,
            branches=branches,
            active_branches=active_branches,
            deferred_paths=deferred_paths,
            formula_branches=formula_branches,
        )
    )
    return tuple(
        {
            (branch.path, branch.start, branch.end): branch
            for branch in controlled
            if not _is_marker_only_container(
                branch,
                branches=branches,
                source_text=source_text,
            )
        }.values()
    )


def _active_or_root_source_branches(
    source_text: str,
    *,
    branches: Sequence[SourceStructureBranch],
    active_branches: Sequence[SourceStructureBranch],
    deferred_paths: set[tuple[str, ...]],
) -> tuple[SourceStructureBranch, ...]:
    """Use a synthetic root only when an unstructured root remains active."""

    if active_branches:
        return tuple(active_branches)
    if branches or _path_is_deferred((), deferred_paths):
        return ()
    return (
        SourceStructureBranch(
            (),
            "source-unit",
            "source unit",
            source_text,
            0,
            len(source_text),
        ),
    )


def _source_boundary_is_temporal(
    _branch: SourceStructureBranch,
    boundary: NumericOccurrenceLike,
) -> bool:
    """Read temporal boundary context from the shared typed occurrence."""
    return boundary.has_temporal_context


def _rounding_clause_refers_to_previous_result(
    clause: SourceStructureBranch,
    *,
    previous: SourceStructureBranch | None,
    source_text: str,
) -> bool:
    """Attach a standalone rounding modifier to its preceding computation."""

    return bool(
        previous is not None
        and _same_top_level_source_path(previous.path, clause.path)
        and _rounding_only_direction(clause.text) is not None
        and _rounding_text_refers_to_result(clause.text)
        and not re.search(r"\w", source_text[previous.end : clause.start])
    )


def _same_top_level_source_path(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> bool:
    if not left or not right:
        return left == right
    return left[0] == right[0]


def _rounding_text_refers_to_result(text: str) -> bool:
    unmarked = _strip_source_clause_marker(text)
    return bool(
        re.match(
            r"\s*(?:"
            r"das\s+ergebnis|"
            r"the\s+result|"
            r"der\s+sich\s+ergebende\s+steuerbetrag"
            r")\b",
            unmarked,
            flags=re.IGNORECASE,
        )
    )


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
    inline_operand_list_spans = _formula_inline_operand_list_spans(source_text)
    boundary_matches = (
        match
        for match in boundary.finditer(source_text)
        if not _source_clause_boundary_splits_state_code_citation(source_text, match)
        and not any(
            start < match.end() < end for start, end in inline_operand_list_spans
        )
        and not (
            match.group() == ";"
            and _FORMULA_DIRECT_FOLLOWING_OPERANDS.search(
                source_text[max(0, match.start() - 300) : match.start()]
            )
            and ":" in source_text[max(0, match.start() - 300) : match.start()]
            and not _FORMULA_FOLLOWING_OPERAND_PROVISO.match(source_text[match.end() :])
        )
    )
    split_points = {
        0,
        len(source_text),
        *(match.end() for match in boundary_matches),
        *(branch.start for branch in branches),
        *(branch.end for branch in branches),
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


def _formula_inline_operand_list_spans(source_text: str) -> tuple[tuple[int, int], ...]:
    """Keep flattened colon-introduced formula operands in one source clause."""

    introduction = re.compile(
        r"\b(?:calculated|computed|determined)\s+(?:by|through)\s+"
        r"(?:adding|subtracting|dividing|multiplying|reducing|deducting|"
        r"increasing|decreasing)\b[^.;:\n]{0,360}:\s*a\.",
        flags=re.IGNORECASE,
    )
    next_numeric_sibling = re.compile(
        r";(?=\s*(?:and\s+)?\d+\.\s+[A-Z\"])",
        flags=re.IGNORECASE,
    )
    later_letter_operand = re.compile(r";\s*(?:and\s+)?b\.", re.IGNORECASE)
    spans: list[tuple[int, int]] = []
    for match in introduction.finditer(source_text):
        sibling = next_numeric_sibling.search(
            source_text,
            match.end(),
            min(len(source_text), match.end() + 2400),
        )
        if (
            sibling is None
            or later_letter_operand.search(
                source_text,
                match.end(),
                sibling.start(),
            )
            is None
        ):
            continue
        spans.append((match.start(), sibling.end()))
    return tuple(spans)


def _source_clause_boundary_splits_state_code_citation(
    source_text: str,
    boundary: re.Match[str],
) -> bool:
    """Keep a line-wrapped ``R.S. 47:...`` citation in one legal clause."""

    return bool(
        boundary.group(0) == "."
        and re.search(r"\bR\.S\.$", source_text[: boundary.end()], re.IGNORECASE)
        and re.match(
            r"\s*\d+[A-Za-z]?:\d+[A-Za-z]?",
            source_text[boundary.end() :],
        )
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
        branch for branch in branches if branch.start <= start and end <= branch.end
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
    corpus_citation_path: str,
    principal_rules: dict[str, dict[str, Any]],
    principal_formula_clause_rules: dict[SourceStructureBranch, set[str]],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
    formula_environment: dict[str, Any],
    dependency_cache: dict[int, dict[str, Any]],
    execution_cache: dict[tuple[str, int], _FormulaExecution | None],
) -> tuple[SourceStructureBranch, ...]:
    """Consume each executed rule/case witness for at most one source formula."""

    candidate_witnesses = {
        branch: _formula_branch_test_witnesses(
            branch,
            corpus_citation_path=corpus_citation_path,
            principal_rules=principal_rules,
            rule_names=principal_formula_clause_rules[branch],
            asserted_by_rule=asserted_by_rule,
            extract_numeric_occurrences=extract_numeric_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
            formula_environment=formula_environment,
            dependency_cache=dependency_cache,
            execution_cache=execution_cache,
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
    corpus_citation_path: str,
    principal_rules: dict[str, dict[str, Any]],
    rule_names: set[str],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
    formula_environment: dict[str, Any],
    dependency_cache: dict[int, dict[str, Any]],
    execution_cache: dict[tuple[str, int], _FormulaExecution | None],
    max_cases_per_rule: int | None = None,
) -> set[tuple[str, str]]:
    interval = _formula_branch_interval(
        branch,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    witnesses: set[tuple[str, str]] = set()
    for rule_name in sorted(rule_names):
        rule = principal_rules[rule_name]
        has_branching_formula = _rule_has_branching_formula(rule)
        asserted_cases = asserted_by_rule.get(rule_name, ())
        if max_cases_per_rule is not None:
            asserted_cases = asserted_cases[:max_cases_per_rule]
        for case in asserted_cases:
            if not _selected_principal_formula_proves_branch(
                rule,
                case=case,
                branch=branch,
                corpus_citation_path=corpus_citation_path,
            ):
                continue
            case_key = id(case)
            if case_key not in dependency_cache:
                dependency_cache[case_key] = _case_asserted_dependency_environment(
                    principal_rules,
                    case,
                    formula_environment=formula_environment,
                )
            dependency_environment = dependency_cache[case_key]
            execution_key = (rule_name, case_key)
            if execution_key not in execution_cache:
                execution_cache[execution_key] = _case_formula_execution(
                    rule,
                    case,
                    formula_environment=formula_environment,
                    dependency_environment=dependency_environment,
                )
            execution = execution_cache[execution_key]
            if not _formula_execution_is_source_branch_witness(
                execution,
                branch,
                rule=rule,
                case=case,
                principal_rules=principal_rules,
                interval=interval,
                dependency_environment=dependency_environment,
                require_corroborated_dependencies=True,
                formula_environment=formula_environment,
                extract_numeric_occurrences=extract_numeric_occurrences,
                numeric_value_is_grounded=numeric_value_is_grounded,
            ):
                continue
            if interval is None:
                evaluation_environment = _case_formula_identifier_environment(
                    case,
                    formula_environment=formula_environment,
                    dependency_environment=dependency_environment,
                )
                if evaluation_environment is None:
                    continue
                temporal_bindings = _formula_leaf_temporal_bindings(
                    execution.leaf,
                    case=case,
                    branch=branch,
                    interval=interval,
                    formula_environment=formula_environment,
                    constant_environment=execution.constant_environment,
                    evaluation_environment=evaluation_environment,
                    extract_numeric_occurrences=extract_numeric_occurrences,
                    numeric_value_is_grounded=numeric_value_is_grounded,
                )
                if temporal_bindings is None:
                    continue
                witness = (
                    "leaf:"
                    + _formula_leaf_semantic_key(
                        execution.leaf,
                        formula_environment=execution.constant_environment,
                    )
                    + (f"|temporal:{temporal_bindings!r}" if temporal_bindings else "")
                    if has_branching_formula
                    else f"case:{id(case)}"
                )
                witnesses.add((rule_name, witness))
                continue
            witnesses.add((rule_name, f"case:{id(case)}"))
    return witnesses


def _selected_principal_formula_proves_branch(
    rule: dict[str, Any],
    *,
    case: dict[str, Any],
    branch: SourceStructureBranch,
    corpus_citation_path: str,
) -> bool:
    """Keep indexed principal proof evidence on its case-selected version."""

    branch_text = _normalized_formula_clause_text(branch.text)
    normalized_citation_path = corpus_citation_path.strip("/").lower()
    matching_atoms = {
        (version_index, citation_path.strip("/").lower())
        for path, citation_path, excerpt in _rule_source_excerpt_atoms(rule)
        if (version_index := _formula_proof_version_index(path)) is not None
        and (excerpt_text := _normalized_formula_clause_text(excerpt))
        and (excerpt_text in branch_text or branch_text in excerpt_text)
    }
    if not matching_atoms:
        return True
    matching_indexes = {
        index
        for index, citation_path in matching_atoms
        if citation_path == normalized_citation_path
    }
    selected_index = _selected_rule_formula_version_index(rule, case)
    return selected_index is not None and selected_index in matching_indexes


def _selected_principal_formula_has_exact_source_proof(
    rule: dict[str, Any],
    *,
    case: dict[str, Any],
    branch: SourceStructureBranch,
    corpus_citation_path: str,
) -> bool:
    """Require normalized same-source formula proof on the selected version."""

    selected_index = _selected_rule_formula_version_index(rule, case)
    if selected_index is None:
        return False
    branch_text = _normalized_formula_clause_text(branch.text)
    normalized_citation_path = corpus_citation_path.strip("/").casefold()
    return any(
        version_index == selected_index
        and citation_path.strip("/").casefold() == normalized_citation_path
        and (excerpt_text := _normalized_formula_clause_text(excerpt))
        and (excerpt_text in branch_text or branch_text in excerpt_text)
        and source_states_explicit_computation(excerpt)
        for path, citation_path, excerpt in _rule_source_excerpt_atoms(rule)
        if (version_index := _formula_proof_version_index(path)) is not None
    )


def _formula_leaf_temporal_bindings(
    leaf: str,
    *,
    case: dict[str, Any],
    branch: SourceStructureBranch,
    interval: _NumericInterval | None,
    formula_environment: dict[str, Any],
    constant_environment: dict[str, Any],
    evaluation_environment: dict[str, Any],
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> tuple[tuple[str, str, str], ...] | None:
    """Identify reached temporal constants grounded by this source computation."""

    expression = _parse_formula_expression(leaf)
    if expression is None:
        return ()
    computation_occurrences = _formula_branch_computation_occurrences(
        branch,
        interval=interval,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    source_factor_occurrences = tuple(
        occurrence
        for occurrence in computation_occurrences
        if occurrence.has_rate_context
        or "multiply" in _formula_operation_kinds(branch.text)
    )
    worded_percentage_rates = _source_worded_percentage_rates(branch.text)
    temporal_names = {
        node.id
        for node in ast.walk(expression)
        if isinstance(node, ast.Name)
        and isinstance(formula_environment.get(node.id), _TemporalFormulaValue)
    }
    if len(temporal_names) > _TEMPORAL_WITNESS_NAME_LIMIT:
        return None
    direct_factor_names = {
        operand.id
        for node in ast.walk(expression)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
        for operand in (node.left, node.right)
        if isinstance(operand, ast.Name) and operand.id in temporal_names
    }
    varying_direct_factor_names = {
        name
        for name in direct_factor_names
        if _temporal_formula_values_vary(formula_environment[name])
    }
    multiplicative_temporal_names = {
        name
        for name in varying_direct_factor_names
        if _temporal_value_has_source_branch_proof(
            formula_environment[name],
            current_value=constant_environment.get(
                name,
                _UNRESOLVED_CONDITION_VALUE,
            ),
            branch=branch,
            case=case,
        )
        and _temporal_name_changes_formula_value(
            expression,
            name=name,
            temporal_value=formula_environment[name],
            environment=evaluation_environment,
        )
    }
    bindings = tuple(
        (name, type(value).__name__, repr(value))
        for name in sorted(multiplicative_temporal_names)
        if (value := constant_environment.get(name, _UNRESOLVED_CONDITION_VALUE))
        is not _UNRESOLVED_CONDITION_VALUE
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (
            any(
                numeric_value_is_grounded(float(value), (source_occurrence,))
                for source_occurrence in source_factor_occurrences
            )
            or any(
                math.isclose(float(value), expected)
                for expected in worded_percentage_rates
            )
        )
    )
    if (
        (source_factor_occurrences or worded_percentage_rates)
        and varying_direct_factor_names
        and not bindings
    ):
        return None
    return bindings


def _source_worded_percentage_rates(source_text: str) -> tuple[float, ...]:
    rates: list[float] = []
    matches = sorted(
        itertools.chain(
            _ENGLISH_WORDED_PERCENTAGE_OF.finditer(source_text),
            _ENGLISH_FRACTIONAL_PERCENTAGE_OF.finditer(source_text),
        ),
        key=lambda match: (match.start(), -(match.end() - match.start())),
    )
    seen_spans: set[tuple[int, int]] = set()
    for match in matches:
        if match.span() in seen_spans:
            continue
        seen_spans.add(match.span())
        phrase = re.sub(
            r"\s+(?:percent|per\s+cent)\s+of\s*$",
            "",
            match.group(0),
            flags=re.IGNORECASE,
        )
        value = _english_fractional_number_value(phrase)
        if value is None:
            value = _english_cardinal_value(phrase)
        if value is not None:
            rates.append(value / 100.0)
    return tuple(rates)


def _english_fractional_number_value(text: str) -> float | None:
    """Parse a bounded simple or mixed English fraction."""

    normalized = re.sub(r"[-\s]+", " ", text.strip().lower())
    whole = 0.0
    fraction_text = normalized
    if " and " in normalized:
        whole_text, fraction_text = normalized.rsplit(" and ", 1)
        whole_value = _english_cardinal_value(whole_text)
        if whole_value is None:
            return None
        whole = whole_value
    tokens = fraction_text.split()
    if tokens and tokens[0] in {"a", "an"}:
        tokens = tokens[1:]
    if not tokens:
        return None
    denominator_words = {
        "half": 2,
        "halves": 2,
        "second": 2,
        "seconds": 2,
        "quarter": 4,
        "quarters": 4,
        "fourth": 4,
        "fourths": 4,
        "fifth": 5,
        "fifths": 5,
        "sixth": 6,
        "sixths": 6,
        "seventh": 7,
        "sevenths": 7,
        "eighth": 8,
        "eighths": 8,
        "ninth": 9,
        "ninths": 9,
        "tenth": 10,
        "tenths": 10,
        "eleventh": 11,
        "elevenths": 11,
        "twelfth": 12,
        "twelfths": 12,
    }
    denominator = denominator_words.get(tokens[-1])
    if denominator is None:
        return None
    numerator = 1.0
    if len(tokens) > 1:
        numerator_value = _english_cardinal_value(" ".join(tokens[:-1]))
        if numerator_value is None:
            return None
        numerator = numerator_value
    if numerator <= 0 or numerator >= denominator:
        return None
    return whole + numerator / denominator


def _english_cardinal_value(text: str) -> float | None:
    units = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    scales = {
        "thousand": 1_000,
        "million": 1_000_000,
        "billion": 1_000_000_000,
        "trillion": 1_000_000_000_000,
    }
    tokens = [token for token in re.split(r"[-\s]+", text.lower()) if token]
    if not tokens or tokens[0] == "and" or tokens[-1] == "and":
        return None
    tokens = [token for token in tokens if token != "and"]
    if not tokens or any(
        token not in units and token != "hundred" and token not in scales
        for token in tokens
    ):
        return None

    def parse_under_thousand(group: list[str]) -> float | None:
        if not group:
            return None
        total = 0.0
        if len(group) >= 2 and group[1] == "hundred":
            hundreds = units.get(group[0])
            if hundreds is None or not 1 <= hundreds <= 9:
                return None
            total = hundreds * 100
            group = group[2:]
            if not group:
                return total
        if len(group) == 1:
            remainder = units.get(group[0])
            return total + remainder if remainder is not None else None
        if len(group) == 2:
            tens = units.get(group[0])
            unit = units.get(group[1])
            if tens in {20, 30, 40, 50, 60, 70, 80, 90} and unit is not None:
                if 1 <= unit <= 9:
                    return total + tens + unit
        return None

    total = 0.0
    group: list[str] = []
    previous_scale = math.inf
    for token in tokens:
        scale = scales.get(token)
        if scale is None:
            group.append(token)
            continue
        if scale >= previous_scale:
            return None
        parsed_group = parse_under_thousand(group)
        if parsed_group is None or parsed_group <= 0:
            return None
        total += parsed_group * scale
        previous_scale = scale
        group = []
    if group:
        parsed_group = parse_under_thousand(group)
        if parsed_group is None or (parsed_group == 0 and total > 0):
            return None
        total += parsed_group
    return total


def _temporal_value_has_source_branch_proof(
    temporal_value: _TemporalFormulaValue,
    *,
    current_value: Any,
    branch: SourceStructureBranch,
    case: dict[str, Any],
) -> bool:
    branch_text = _normalized_formula_clause_text(branch.text)
    for index in _selected_temporal_version_indexes(temporal_value, case):
        _start, _end, value = temporal_value.versions[index]
        if not _formula_runtime_values_equal(value, current_value):
            continue
        excerpts = (
            temporal_value.version_formula_excerpts[index]
            if index < len(temporal_value.version_formula_excerpts)
            else ()
        )
        for excerpt in excerpts:
            excerpt_text = _normalized_formula_clause_text(excerpt)
            if excerpt_text and (
                excerpt_text in branch_text or branch_text in excerpt_text
            ):
                return True
    return False


def _selected_temporal_version_indexes(
    temporal_value: _TemporalFormulaValue,
    case: dict[str, Any],
) -> tuple[int, ...]:
    if len(temporal_value.versions) > _TEMPORAL_WITNESS_VERSION_LIMIT:
        return ()
    period = _normalized_case_period(case)
    if not _is_iso_calendar_date(period) or any(
        not _is_iso_calendar_date(start)
        or (end and not _is_iso_calendar_date(end))
        or (end and end < start)
        for start, end, _value in temporal_value.versions
    ):
        return ()
    candidates = [
        (index, start, value)
        for index, (start, end, value) in enumerate(temporal_value.versions)
        if start <= period and (not end or period <= end)
    ]
    if not candidates:
        return ()
    latest = max(start for _index, start, _value in candidates)
    selected = [(index, value) for index, start, value in candidates if start == latest]
    if len(selected) != 1:
        return ()
    return (selected[0][0],)


def _temporal_formula_values_vary(temporal_value: _TemporalFormulaValue) -> bool:
    if len(temporal_value.versions) > _TEMPORAL_WITNESS_VERSION_LIMIT:
        return False
    values = [value for _start, _end, value in temporal_value.versions]
    return bool(values) and any(
        not _formula_runtime_values_equal(values[0], value) for value in values[1:]
    )


def _temporal_name_changes_formula_value(
    expression: ast.expr,
    *,
    name: str,
    temporal_value: _TemporalFormulaValue,
    environment: dict[str, Any],
) -> bool:
    if len(temporal_value.versions) > _TEMPORAL_WITNESS_VERSION_LIMIT:
        return False
    baseline = _evaluate_condition_expression(expression, environment)
    if baseline is _UNRESOLVED_CONDITION_VALUE:
        return False
    current = environment.get(name, _UNRESOLVED_CONDITION_VALUE)
    for _start, _end, candidate in temporal_value.versions[
        :_TEMPORAL_WITNESS_VERSION_LIMIT
    ]:
        if current is not _UNRESOLVED_CONDITION_VALUE and _formula_runtime_values_equal(
            current,
            candidate,
        ):
            continue
        candidate_environment = {**environment, name: candidate}
        alternative = _evaluate_condition_expression(
            expression,
            candidate_environment,
        )
        if alternative is not _UNRESOLVED_CONDITION_VALUE and not (
            _formula_runtime_values_equal(baseline, alternative)
        ):
            return True
    return False


def _formula_branch_computation_occurrences(
    branch: SourceStructureBranch,
    *,
    interval: _NumericInterval | None,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> tuple[NumericOccurrenceLike, ...]:
    recall_text = authoritative_numeric_recall_text(branch.text)
    fractional_percentages = _source_fractional_percentage_occurrences(recall_text)
    fractional_spans = tuple(
        (occurrence.start, occurrence.end) for occurrence in fractional_percentages
    )
    source_occurrences = (
        tuple(
            occurrence
            for occurrence in extract_numeric_occurrences(recall_text)
            if not any(
                start <= occurrence.start and occurrence.end <= end
                for start, end in fractional_spans
            )
        )
        + fractional_percentages
    )
    boundaries = (
        ()
        if interval is None
        else tuple(
            occurrence
            for occurrence in (interval.lower, interval.upper)
            if occurrence is not None
        )
    )
    return tuple(
        occurrence
        for occurrence in sorted(
            source_occurrences,
            key=lambda occurrence: (occurrence.start, occurrence.end),
        )
        if not _temporal_occurrence_is_formula_applicability_preface(
            occurrence,
            branch.text,
        )
        and not any(
            _numeric_occurrences_are_equivalent(occurrence, boundary)
            for boundary in boundaries
        )
    )


def _source_fractional_percentage_occurrences(
    text: str,
) -> tuple[_NumericOccurrenceView, ...]:
    """Collapse English mixed-percentage components into one rate occurrence."""

    occurrences: list[_NumericOccurrenceView] = []
    for match in _ENGLISH_FRACTIONAL_PERCENTAGE_OF.finditer(text):
        phrase = re.sub(
            r"\s+(?:percent|per\s+cent)\s+of\s*$",
            "",
            match.group(0),
            flags=re.IGNORECASE,
        )
        percentage = _english_fractional_number_value(phrase)
        if percentage is None:
            continue
        occurrences.append(
            _NumericOccurrenceView(
                value=percentage / 100.0,
                start=match.start(),
                end=match.end(),
                raw=match.group(0),
                has_rate_context=True,
                has_temporal_context=False,
                has_structural_context=False,
                source_value=percentage,
                requires_rate_context=False,
                is_word_number=True,
                alternative_values=(),
            )
        )
    return tuple(occurrences)


def _formula_execution_is_source_branch_witness(
    execution: _FormulaExecution | None,
    branch: SourceStructureBranch,
    *,
    rule: dict[str, Any],
    case: dict[str, Any],
    principal_rules: dict[str, dict[str, Any]],
    interval: _NumericInterval | None,
    dependency_environment: dict[str, Any],
    require_corroborated_dependencies: bool,
    formula_environment: dict[str, Any],
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> bool:
    """Return whether one resolved execution proves the exact source branch."""

    execution_environment = _case_formula_identifier_environment(
        case,
        formula_environment=formula_environment,
        dependency_environment=dependency_environment,
    )
    if (
        execution is None
        or execution_environment is None
        or not _formula_execution_leaf_is_computational(execution)
        or not _formula_execution_matches_source_branch(
            execution,
            branch,
            interval=interval,
            formula_environment=formula_environment,
            execution_environment=execution_environment,
            extract_numeric_occurrences=extract_numeric_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
        )
    ):
        return False
    if require_corroborated_dependencies and any(
        name not in dependency_environment
        or _test_case_asserted_output_value(case, name) is _UNRESOLVED_CONDITION_VALUE
        for name in _reached_local_formula_dependency_names(
            execution,
            case,
            principal_rules=principal_rules,
            formula_environment=formula_environment,
            dependency_environment=dependency_environment,
        )
    ):
        return False
    if interval is None:
        return True
    if execution.trace:
        selector_values = _reached_formula_interval_selector_values(
            execution,
            case,
            principal_rules=principal_rules,
            formula_environment=formula_environment,
            dependency_environment=dependency_environment,
            interval=interval,
            source_text=branch.text,
        )
    else:
        selector_names = _rule_numeric_selector_names(rule)
        selector_values = _case_numeric_selector_values(
            case,
            selector_names,
            dependency_environment=dependency_environment,
        )
    return any(_interval_contains(interval, value) for value in selector_values)


def _unbound_matching_formula_rules(
    branch: SourceStructureBranch,
    *,
    corpus_citation_path: str,
    principal_rules: dict[str, dict[str, Any]],
    bound_rule_names: set[str],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
    formula_environment: dict[str, Any],
    dependency_cache: dict[int, dict[str, Any]],
    execution_cache: dict[tuple[str, int], _FormulaExecution | None],
) -> _UnboundFormulaDiagnostic:
    """Find asserted formulas that match a clause but lack its source binding."""

    eligible_rule_names = [
        rule_name
        for rule_name in sorted(principal_rules)
        if rule_name not in bound_rule_names and asserted_by_rule.get(rule_name)
    ]
    scanned_rule_names = eligible_rule_names[:_UNBOUND_FORMULA_DIAGNOSTIC_RULE_LIMIT]
    scan_capped = len(scanned_rule_names) < len(eligible_rule_names) or any(
        len(asserted_by_rule[rule_name]) > _UNBOUND_FORMULA_DIAGNOSTIC_CASE_LIMIT
        for rule_name in scanned_rule_names
    )
    witnesses = _formula_branch_test_witnesses(
        branch,
        corpus_citation_path=corpus_citation_path,
        principal_rules=principal_rules,
        rule_names=set(scanned_rule_names),
        asserted_by_rule=asserted_by_rule,
        extract_numeric_occurrences=extract_numeric_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
        formula_environment=formula_environment,
        dependency_cache=dependency_cache,
        execution_cache=execution_cache,
        max_cases_per_rule=_UNBOUND_FORMULA_DIAGNOSTIC_CASE_LIMIT,
    )
    return _UnboundFormulaDiagnostic(
        rule_names=tuple(sorted({rule_name for rule_name, _witness in witnesses})),
        scan_capped=scan_capped,
    )


def _unbound_formula_binding_feedback(
    diagnostic: _UnboundFormulaDiagnostic,
    *,
    corpus_citation_path: str,
) -> str:
    """Render source-binding repair guidance without overstating scan coverage."""

    detail = ""
    if diagnostic.rule_names:
        detail = (
            " Asserted principal formula(s) already execute this computation "
            "but are excluded from source-bound evidence: "
            + _bounded_identifier_feedback(diagnostic.rule_names)
            + ". To bind each named rule unambiguously, add a "
            "`versions[N].formula` proof atom whose "
            "`source.corpus_citation_path` is exactly "
            f"`{corpus_citation_path}` and whose short `source.excerpt` quotes "
            "this computation. A citation-only proof atom, human-readable "
            "rule-level `source:`, or self-import does not distinguish one "
            "internal formula clause from another."
        )
    if diagnostic.scan_capped:
        detail += (
            " The best-effort unbound-formula diagnostic scan was capped at "
            f"{_UNBOUND_FORMULA_DIAGNOSTIC_RULE_LIMIT} rules and "
            f"{_UNBOUND_FORMULA_DIAGNOSTIC_CASE_LIMIT} cases per rule; "
            "additional matching principal formulas may exist beyond the "
            "reported names."
        )
    return detail


def _uncorroborated_formula_dependency_diagnostic(
    branch: SourceStructureBranch,
    *,
    principal_rules: dict[str, dict[str, Any]],
    bound_rule_names: set[str],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
    formula_environment: dict[str, Any],
    strict_dependency_cache: dict[int, dict[str, Any]],
    strict_execution_cache: dict[tuple[str, int], _FormulaExecution | None],
    permissive_dependency_cache: dict[tuple[str, int], dict[str, Any]],
    permissive_execution_cache: dict[tuple[str, int], _FormulaExecution | None],
) -> _UncorroboratedFormulaDependencyDiagnostic:
    """Find reached derived dependencies omitted from candidate assertions."""

    eligible_rule_names = [
        rule_name
        for rule_name in sorted(bound_rule_names)
        if asserted_by_rule.get(rule_name)
    ]
    scanned_rule_names = eligible_rule_names[:_UNBOUND_FORMULA_DIAGNOSTIC_RULE_LIMIT]
    scan_capped = len(scanned_rule_names) < len(eligible_rule_names) or any(
        len(asserted_by_rule[rule_name]) > _UNBOUND_FORMULA_DIAGNOSTIC_CASE_LIMIT
        for rule_name in scanned_rule_names
    )
    interval = _formula_branch_interval(
        branch,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    dependency_names: set[str] = set()
    case_names: set[str] = set()
    for rule_name in scanned_rule_names:
        rule = principal_rules[rule_name]
        rule_dependency_names = _local_rule_dependency_closure(
            rule,
            principal_rules=principal_rules,
        )
        for case in asserted_by_rule[rule_name][
            :_UNBOUND_FORMULA_DIAGNOSTIC_CASE_LIMIT
        ]:
            case_key = id(case)
            if case_key not in strict_dependency_cache:
                strict_dependency_cache[case_key] = (
                    _case_asserted_dependency_environment(
                        principal_rules,
                        case,
                        formula_environment=formula_environment,
                    )
                )
            strict_dependencies = strict_dependency_cache[case_key]
            execution_key = (rule_name, case_key)
            if execution_key not in strict_execution_cache:
                strict_execution_cache[execution_key] = _case_formula_execution(
                    rule,
                    case,
                    formula_environment=formula_environment,
                    dependency_environment=strict_dependencies,
                )
            if _formula_execution_is_source_branch_witness(
                strict_execution_cache[execution_key],
                branch,
                rule=rule,
                case=case,
                principal_rules=principal_rules,
                interval=interval,
                dependency_environment=strict_dependencies,
                require_corroborated_dependencies=True,
                formula_environment=formula_environment,
                extract_numeric_occurrences=extract_numeric_occurrences,
                numeric_value_is_grounded=numeric_value_is_grounded,
            ):
                continue
            if execution_key not in permissive_dependency_cache:
                permissive_dependency_cache[execution_key] = (
                    _case_dependency_environment(
                        principal_rules,
                        case,
                        formula_environment=formula_environment,
                        require_asserted_value=False,
                        allowed_names=rule_dependency_names,
                    )
                )
            permissive_dependencies = permissive_dependency_cache[execution_key]
            if execution_key not in permissive_execution_cache:
                permissive_execution_cache[execution_key] = _case_formula_execution(
                    rule,
                    case,
                    formula_environment=formula_environment,
                    dependency_environment=permissive_dependencies,
                )
            permissive_execution = permissive_execution_cache[execution_key]
            if not _formula_execution_is_source_branch_witness(
                permissive_execution,
                branch,
                rule=rule,
                case=case,
                principal_rules=principal_rules,
                interval=interval,
                dependency_environment=permissive_dependencies,
                require_corroborated_dependencies=False,
                formula_environment=formula_environment,
                extract_numeric_occurrences=extract_numeric_occurrences,
                numeric_value_is_grounded=numeric_value_is_grounded,
            ):
                continue
            missing = _unasserted_reached_formula_dependency_names(
                case,
                execution=permissive_execution,
                principal_rules=principal_rules,
                formula_environment=formula_environment,
                dependency_environment=permissive_dependencies,
            )
            if not missing:
                continue
            dependency_names.update(missing)
            case_names.add(str(case.get("name") or "<unnamed>"))
    return _UncorroboratedFormulaDependencyDiagnostic(
        dependency_names=tuple(sorted(dependency_names)),
        case_names=tuple(sorted(case_names)),
        scan_capped=scan_capped,
    )


def _local_rule_dependency_closure(
    rule: dict[str, Any],
    *,
    principal_rules: dict[str, dict[str, Any]],
) -> set[str]:
    """Return the transitive local principal rules referenced by one rule."""

    pending = set(_FORMULA_IDENTIFIER.findall(_rule_formula_text(rule)))
    dependencies: set[str] = set()
    while pending:
        name = pending.pop()
        dependency_rule = principal_rules.get(name)
        if dependency_rule is None or name in dependencies:
            continue
        dependencies.add(name)
        pending.update(_FORMULA_IDENTIFIER.findall(_rule_formula_text(dependency_rule)))
    return dependencies


def _unasserted_reached_formula_dependency_names(
    case: dict[str, Any],
    *,
    execution: _FormulaExecution,
    principal_rules: dict[str, dict[str, Any]],
    formula_environment: dict[str, Any],
    dependency_environment: dict[str, Any],
) -> set[str]:
    """Return unasserted local derived rules reached by one execution."""

    return {
        name
        for name in _reached_local_formula_dependency_names(
            execution,
            case,
            principal_rules=principal_rules,
            formula_environment=formula_environment,
            dependency_environment=dependency_environment,
        )
        if _test_case_asserted_output_value(case, name) is _UNRESOLVED_CONDITION_VALUE
    }


def _reached_local_formula_dependency_names(
    execution: _FormulaExecution,
    case: dict[str, Any],
    *,
    principal_rules: dict[str, dict[str, Any]],
    formula_environment: dict[str, Any],
    dependency_environment: dict[str, Any],
) -> set[str]:
    """Return every local derived rule reached by one formula execution."""

    selector_environment = _case_formula_identifier_environment(
        case,
        formula_environment=formula_environment,
        dependency_environment=dependency_environment,
    )
    pending = _reached_formula_expression_identifier_names(
        execution.leaf,
        environment=selector_environment,
    )
    if selector_environment is not None:
        pending.update(
            name
            for step in execution.trace
            for selector in step.selectors
            for name in _reached_formula_expression_identifier_names(
                selector,
                environment=selector_environment,
            )
        )
    visited: set[str] = set()
    dependencies: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        dependency_rule = principal_rules.get(name)
        if dependency_rule is None:
            continue
        dependencies.add(name)
        if name not in dependency_environment:
            continue
        dependency_execution = _case_formula_execution(
            dependency_rule,
            case,
            formula_environment=formula_environment,
            dependency_environment=dependency_environment,
        )
        if dependency_execution is None:
            continue
        pending.update(
            _reached_formula_expression_identifier_names(
                dependency_execution.leaf,
                environment=selector_environment,
            )
        )
        if selector_environment is not None:
            pending.update(
                dependency_name
                for step in dependency_execution.trace
                for selector in step.selectors
                for dependency_name in _reached_formula_expression_identifier_names(
                    selector,
                    environment=selector_environment,
                )
            )
    return dependencies


def _reached_formula_interval_selector_values(
    execution: _FormulaExecution,
    case: dict[str, Any],
    *,
    principal_rules: dict[str, dict[str, Any]],
    formula_environment: dict[str, Any],
    dependency_environment: dict[str, Any],
    interval: _NumericInterval,
    source_text: str,
) -> tuple[float, ...]:
    """Trace values paired with the source bound through derived selectors."""

    environment = _case_formula_identifier_environment(
        case,
        formula_environment=formula_environment,
        dependency_environment=dependency_environment,
    )
    if environment is None:
        return ()
    evidence_names = set(dependency_environment)
    inputs = case.get("input")
    if isinstance(inputs, dict):
        for key in inputs:
            evidence_names.update(_input_key_names(key))
    selectors = [selector for step in execution.trace for selector in step.selectors]
    names = set().union(
        *(
            _reached_formula_expression_identifier_names(
                selector,
                environment=environment,
            )
            for selector in selectors
        ),
        set(),
    )
    pending = {name for name in names if name in principal_rules}
    visited: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        dependency_execution = _case_formula_execution(
            principal_rules[name],
            case,
            formula_environment=formula_environment,
            dependency_environment=dependency_environment,
        )
        if dependency_execution is None:
            continue
        dependency_selectors = [
            selector
            for step in dependency_execution.trace
            for selector in step.selectors
        ]
        dependency_selectors.append(dependency_execution.leaf)
        selectors.extend(dependency_selectors)
        reached = set().union(
            *(
                _reached_formula_expression_identifier_names(
                    selector,
                    environment=environment,
                )
                for selector in dependency_selectors
            ),
            set(),
        )
        names.update(reached)
        pending.update(
            dependency_name
            for dependency_name in reached
            if dependency_name in principal_rules and dependency_name not in visited
        )
    values = tuple(
        value
        for selector in selectors
        for value in _formula_interval_subject_values(
            selector,
            environment=environment,
            evidence_names=evidence_names,
            interval=interval,
        )
    )
    if values:
        return values
    dependency_values = _corroborated_dependency_interval_subject_values(
        execution.leaf,
        environment=environment,
        dependency_names=set(dependency_environment),
        interval=interval,
        source_text=source_text,
    )
    if dependency_values:
        return dependency_values
    return _formula_progressive_clamp_subject_values(
        execution.leaf,
        environment=environment,
        evidence_names=evidence_names.difference(dependency_environment),
        source_text=source_text,
    )


def _corroborated_dependency_interval_subject_values(
    formula_text: str,
    *,
    environment: dict[str, Any],
    dependency_names: set[str],
    interval: _NumericInterval,
    source_text: str,
) -> tuple[float, ...]:
    """Trace source bounds applied to an asserted local derived selector."""

    expression = _parse_formula_expression(formula_text)
    if expression is None:
        return ()
    values: list[float] = []

    def numeric_value(node: ast.expr) -> float | None:
        value = _evaluate_condition_expression(node, environment)
        numeric = _rulespec_runtime_decimal(value)
        return float(numeric) if numeric is not None else None

    counterfactual_environment = dict(environment)
    for environment_name, environment_value in environment.items():
        if isinstance(environment_value, bool) or not isinstance(
            environment_value,
            (int, float, Decimal),
        ):
            continue
        numeric_environment_value = _rulespec_runtime_decimal(environment_value)
        if numeric_environment_value is not None:
            counterfactual_environment[environment_name] = float(
                numeric_environment_value
            )

    def subject_influences_formula(name: str) -> bool:
        baseline = _rulespec_runtime_decimal(
            _evaluate_condition_expression(expression, counterfactual_environment)
        )
        subject_value = _rulespec_runtime_decimal(counterfactual_environment.get(name))
        if baseline is None or subject_value is None:
            return False
        delta = max(abs(subject_value) / Decimal(1000), Decimal("0.000001"))
        for candidate_value in (subject_value - delta, subject_value + delta):
            candidate_environment = dict(counterfactual_environment)
            candidate_environment[name] = float(candidate_value)
            candidate = _rulespec_runtime_decimal(
                _evaluate_condition_expression(expression, candidate_environment)
            )
            if candidate is not None and not math.isclose(
                float(candidate),
                float(baseline),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                return True
        return False

    def operation_influences_formula(operation: ast.expr) -> bool:
        baseline = _rulespec_runtime_decimal(
            _evaluate_condition_expression(expression, counterfactual_environment)
        )
        operation_value = _rulespec_runtime_decimal(
            _evaluate_condition_expression(operation, counterfactual_environment)
        )
        if baseline is None or operation_value is None:
            return False
        target = (
            type(operation),
            getattr(operation, "lineno", None),
            getattr(operation, "col_offset", None),
            getattr(operation, "end_lineno", None),
            getattr(operation, "end_col_offset", None),
            ast.dump(operation, include_attributes=False),
        )
        delta = max(abs(operation_value) / Decimal(1000), Decimal("0.000001"))
        for candidate_value in (operation_value - delta, operation_value + delta):
            candidate_expression = _parse_formula_expression(formula_text)
            if candidate_expression is None:
                return False

            class MatchedOperationReplacement(ast.NodeTransformer):
                replaced = False

                def visit(self, node: ast.AST) -> ast.AST:
                    descriptor = (
                        type(node),
                        getattr(node, "lineno", None),
                        getattr(node, "col_offset", None),
                        getattr(node, "end_lineno", None),
                        getattr(node, "end_col_offset", None),
                        ast.dump(node, include_attributes=False),
                    )
                    if not self.replaced and descriptor == target:
                        self.replaced = True
                        return ast.copy_location(
                            ast.Constant(value=float(candidate_value)),
                            node,
                        )
                    return super().visit(node)

            replacement = MatchedOperationReplacement()
            candidate_expression = replacement.visit(candidate_expression)
            if not replacement.replaced:
                return False
            candidate = _rulespec_runtime_decimal(
                _evaluate_condition_expression(
                    candidate_expression,
                    counterfactual_environment,
                )
            )
            if candidate is not None and not math.isclose(
                float(candidate),
                float(baseline),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                return True
        return False

    def record(
        subject: ast.expr,
        bound: ast.expr,
        *,
        boundary: NumericOccurrenceLike | None,
        operation: ast.expr,
    ) -> None:
        if (
            not isinstance(subject, ast.Name)
            or subject.id not in dependency_names
            or not _formula_subject_matches_source(subject.id, source_text)
            or not subject_influences_formula(subject.id)
            or not operation_influences_formula(operation)
        ):
            return
        subject_value = numeric_value(subject)
        bound_value = numeric_value(bound)
        if (
            subject_value is None
            or bound_value is None
            or boundary is None
            or boundary.value is None
            or not math.isclose(
                bound_value,
                float(boundary.value),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            return
        values.append(subject_value)

    for node in ast.walk(expression):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "min"
            and len(node.args) == 2
            and not node.keywords
        ):
            record(
                node.args[0],
                node.args[1],
                boundary=interval.upper,
                operation=node,
            )
            record(
                node.args[1],
                node.args[0],
                boundary=interval.upper,
                operation=node,
            )
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            record(
                node.left,
                node.right,
                boundary=interval.lower,
                operation=node,
            )
    return tuple(values)


def _formula_progressive_clamp_subject_values(
    formula_text: str,
    *,
    environment: dict[str, Any],
    evidence_names: set[str],
    source_text: str,
) -> tuple[float, ...]:
    """Evaluate the minuend subject in a reached ``min(max(...), cap)`` clamp."""

    expression = _parse_formula_expression(formula_text)
    if expression is None:
        return ()
    values: list[float] = []
    for node in ast.walk(expression):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "min"
            and len(node.args) == 2
            and not node.keywords
        ):
            continue
        for argument, cap in (
            (node.args[0], node.args[1]),
            (node.args[1], node.args[0]),
        ):
            if not (
                isinstance(argument, ast.Call)
                and isinstance(argument.func, ast.Name)
                and argument.func.id == "max"
                and len(argument.args) == 2
                and not argument.keywords
            ):
                continue
            zero_operands = [
                operand
                for operand in argument.args
                if isinstance(operand, ast.Constant) and operand.value == 0
            ]
            if len(zero_operands) != 1:
                continue
            positive_operand = next(
                operand for operand in argument.args if operand is not zero_operands[0]
            )
            if isinstance(positive_operand, ast.Name):
                subject = positive_operand
            elif (
                isinstance(positive_operand, ast.BinOp)
                and isinstance(positive_operand.op, ast.Sub)
                and isinstance(positive_operand.left, ast.Name)
                and isinstance(cap, ast.BinOp)
                and isinstance(cap.op, ast.Sub)
                and ast.dump(positive_operand.right) == ast.dump(cap.right)
            ):
                subject = positive_operand.left
            else:
                continue
            if subject.id not in evidence_names or not _formula_subject_matches_source(
                subject.id,
                source_text,
            ):
                continue
            value = _evaluate_condition_expression(subject, environment)
            if _rulespec_runtime_decimal(value) is not None:
                values.append(float(value))
    return tuple(values)


def _formula_progressive_clamp_subject_names(formula_text: str) -> tuple[str, ...]:
    """Return structurally valid subject names from progressive clamps."""

    expression = _parse_formula_expression(formula_text)
    if expression is None:
        return ()
    names: list[str] = []
    for node in ast.walk(expression):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "min"
            and len(node.args) == 2
            and not node.keywords
        ):
            continue
        for argument, cap in (
            (node.args[0], node.args[1]),
            (node.args[1], node.args[0]),
        ):
            if not (
                isinstance(argument, ast.Call)
                and isinstance(argument.func, ast.Name)
                and argument.func.id == "max"
                and len(argument.args) == 2
                and not argument.keywords
            ):
                continue
            zero_operands = [
                operand
                for operand in argument.args
                if isinstance(operand, ast.Constant) and operand.value == 0
            ]
            if len(zero_operands) != 1:
                continue
            positive_operand = next(
                operand for operand in argument.args if operand is not zero_operands[0]
            )
            if isinstance(positive_operand, ast.Name):
                names.append(positive_operand.id)
            elif (
                isinstance(positive_operand, ast.BinOp)
                and isinstance(positive_operand.op, ast.Sub)
                and isinstance(positive_operand.left, ast.Name)
                and isinstance(cap, ast.BinOp)
                and isinstance(cap.op, ast.Sub)
                and ast.dump(positive_operand.right) == ast.dump(cap.right)
            ):
                names.append(positive_operand.left.id)
    return tuple(names)


def _formula_subject_matches_source(name: str, source_text: str) -> bool:
    """Bind an inferred clamp minuend to the source's named subject."""

    generic_tokens = {
        "amount",
        "base",
        "input",
        "subject",
        "total",
        "value",
    }
    raw_name_tokens = tuple(re.findall(r"[a-z0-9]+", name.lower().replace("_", " ")))
    base_name_candidates = [raw_name_tokens]
    section_indexes = tuple(
        index for index, token in enumerate(raw_name_tokens) if token == "section"
    )
    if section_indexes:
        semantic_start = section_indexes[-1] + 1
        while (
            semantic_start < len(raw_name_tokens)
            and raw_name_tokens[semantic_start].isdigit()
        ):
            semantic_start += 1
        if semantic_start < len(raw_name_tokens):
            base_name_candidates.append(raw_name_tokens[semantic_start:])
    candidate_name_tokens = list(base_name_candidates)
    candidate_name_tokens.extend(
        candidate_tokens[:-1]
        for candidate_tokens in base_name_candidates
        if candidate_tokens and candidate_tokens[-1] == "boundary"
    )
    interval_comparison = (
        r"(?:above|at\s+(?:least|most)|below|between|exceeds?|exceeding|from|"
        r"greater\s+than|in\s+excess\s+of|less\s+than|more\s+than|"
        r"no\s+(?:less|more|greater)\s+than|not\s+(?:in\s+excess\s+of|"
        r"less\s+than|more\s+than|over)|over|through|under|up\s+to)"
    )
    for candidate_tokens in candidate_name_tokens:
        name_tokens = tuple(
            token for token in candidate_tokens if token not in generic_tokens
        )
        if not name_tokens:
            continue
        subject_phrase = r"\s+".join(re.escape(token) for token in name_tokens)
        if re.search(
            rf"\b{subject_phrase}\b\s+"
            rf"(?:(?:is|are|shall\s+be)\s+)?{interval_comparison}\b",
            source_text,
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _formula_interval_subject_values(
    selector: str,
    *,
    environment: dict[str, Any],
    evidence_names: set[str],
    interval: _NumericInterval,
) -> tuple[float, ...]:
    """Return evaluated operands paired with an exact source interval bound."""

    expression = _parse_formula_expression(selector)
    if expression is None:
        return ()
    values: list[float] = []

    def numeric_value(node: ast.expr) -> float | None:
        value = _evaluate_condition_expression(node, environment)
        if _rulespec_runtime_decimal(value) is not None:
            return float(value)
        return None

    def identifier_names(node: ast.expr) -> set[str]:
        return {
            candidate.id
            for candidate in ast.walk(node)
            if isinstance(candidate, ast.Name)
        }

    def bound_matches_source(
        bound_value: float,
        operator: ast.cmpop,
        *,
        bound_is_left: bool,
        comparison_value: bool,
    ) -> bool:
        boundaries = tuple(
            float(boundary.value)
            for boundary in (interval.lower, interval.upper)
            if boundary is not None and boundary.value is not None
        )
        if any(
            math.isclose(bound_value, boundary, rel_tol=1e-12, abs_tol=1e-12)
            for boundary in boundaries
        ):
            return True
        if comparison_value:
            return False
        subject_relation = type(operator)
        if bound_is_left:
            subject_relation = {
                ast.Lt: ast.Gt,
                ast.LtE: ast.GtE,
                ast.Gt: ast.Lt,
                ast.GtE: ast.LtE,
            }.get(subject_relation, subject_relation)
        if (
            interval.lower is not None
            and interval.lower.value is not None
            and interval.lower_inclusive
            and subject_relation is ast.LtE
            and float(bound_value).is_integer()
            and float(interval.lower.value).is_integer()
            and math.isclose(float(interval.lower.value) - bound_value, 1.0)
        ):
            return True
        return bool(
            interval.upper is not None
            and interval.upper.value is not None
            and interval.upper_inclusive
            and subject_relation is ast.GtE
            and float(bound_value).is_integer()
            and float(interval.upper.value).is_integer()
            and math.isclose(bound_value - float(interval.upper.value), 1.0)
        )

    def record_source_subject(
        bound: ast.expr,
        subject: ast.expr,
        operator: ast.cmpop,
        *,
        bound_is_left: bool,
        comparison_value: bool,
    ) -> None:
        bound_value = numeric_value(bound)
        subject_value = numeric_value(subject)
        if (
            bound_value is None
            or subject_value is None
            or identifier_names(bound) & evidence_names
            or not identifier_names(subject) & evidence_names
            or not bound_matches_source(
                bound_value,
                operator,
                bound_is_left=bound_is_left,
                comparison_value=comparison_value,
            )
        ):
            return
        values.append(subject_value)

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                visit(item)
                value = _evaluate_condition_expression(item, environment)
                if value is _UNRESOLVED_CONDITION_VALUE:
                    return
                if isinstance(node.op, ast.And) and not bool(value):
                    return
                if isinstance(node.op, ast.Or) and bool(value):
                    return
            return
        if isinstance(node, ast.Compare):
            left = node.left
            for operator, right in zip(node.ops, node.comparators):
                comparison = ast.Compare(
                    left=left,
                    ops=[operator],
                    comparators=[right],
                )
                value = _evaluate_condition_expression(comparison, environment)
                if not isinstance(value, bool):
                    return
                record_source_subject(
                    left,
                    right,
                    operator,
                    bound_is_left=True,
                    comparison_value=value,
                )
                record_source_subject(
                    right,
                    left,
                    operator,
                    bound_is_left=False,
                    comparison_value=value,
                )
                if value is False:
                    return
                left = right
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(expression)
    return tuple(values)


def _case_formula_identifier_environment(
    case: dict[str, Any],
    *,
    formula_environment: dict[str, Any],
    dependency_environment: dict[str, Any],
) -> dict[str, Any] | None:
    """Rebuild the environment used to resolve a case's formula selectors."""

    environment = _formula_environment_for_case(formula_environment, case)
    inputs = _case_input_formula_environment(case)
    if inputs is None:
        return None
    for name, value in (*dependency_environment.items(), *inputs.items()):
        if name in environment and not _formula_runtime_values_equal(
            environment[name],
            value,
        ):
            return None
        environment[name] = value
    return environment


def _evaluated_formula_identifier_names(
    selector: str,
    *,
    environment: dict[str, Any],
) -> set[str] | None:
    """Trace selector identifiers while honoring boolean short-circuiting."""

    expression = _parse_formula_expression(selector)
    if expression is None:
        return None
    names: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            names.add(node.id)
            return
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                visit(item)
                value = _evaluate_condition_expression(item, environment)
                if value is _UNRESOLVED_CONDITION_VALUE:
                    return
                if isinstance(node.op, ast.And) and not bool(value):
                    return
                if isinstance(node.op, ast.Or) and bool(value):
                    return
            return
        if isinstance(node, ast.Compare):
            visit(node.left)
            for index, comparator in enumerate(node.comparators):
                visit(comparator)
                prefix = ast.Compare(
                    left=node.left,
                    ops=node.ops[: index + 1],
                    comparators=node.comparators[: index + 1],
                )
                value = _evaluate_condition_expression(prefix, environment)
                if value is _UNRESOLVED_CONDITION_VALUE or value is False:
                    return
            return
        if isinstance(node, ast.Call):
            for argument in node.args:
                visit(argument)
            for keyword in node.keywords:
                visit(keyword.value)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(expression)
    return names


def _reached_formula_expression_identifier_names(
    expression: str,
    *,
    environment: dict[str, Any] | None,
) -> set[str]:
    """Return reached names, conservatively falling back for opaque syntax."""

    if environment is not None:
        evaluated = _evaluated_formula_identifier_names(
            expression,
            environment=environment,
        )
        if evaluated is not None:
            return evaluated
    return set(_FORMULA_IDENTIFIER.findall(expression))


def _uncorroborated_formula_dependency_feedback(
    diagnostic: _UncorroboratedFormulaDependencyDiagnostic,
) -> str:
    """Render bounded repair guidance for unasserted reached dependencies."""

    detail = ""
    if diagnostic.dependency_names:
        detail = (
            " Best-effort dependency tracing found asserted principal-output "
            "case(s) that can reach this computation, but those cases omit "
            "output assertions for reached local derived dependency selector(s): "
            f"{_bounded_identifier_feedback(diagnostic.dependency_names)}. "
            "A formula witness treats a local derived dependency as corroborated "
            "only when the same case asserts its expected output; raw inputs alone "
            "do not corroborate that intermediate. Add the expected dependency "
            "output(s) to the same candidate case(s): "
            f"{_bounded_identifier_feedback(diagnostic.case_names)}. Never shadow "
            "a local derived rule under `input:`."
        )
    if diagnostic.scan_capped:
        detail += (
            " The best-effort dependency-corroboration diagnostic scan was capped "
            f"at {_UNBOUND_FORMULA_DIAGNOSTIC_RULE_LIMIT} rules and "
            f"{_UNBOUND_FORMULA_DIAGNOSTIC_CASE_LIMIT} cases per rule; additional "
            "missing dependency assertions may exist beyond the reported names."
        )
    return detail


def _formula_execution_matches_source_branch(
    execution: _FormulaExecution,
    branch: SourceStructureBranch,
    *,
    interval: _NumericInterval | None,
    formula_environment: dict[str, Any],
    execution_environment: dict[str, Any] | None = None,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> bool:
    """Bind a reached leaf to numeric evidence in its exact source formula."""

    source_operations = _formula_operation_kinds(branch.text)
    binding_environment = {
        name: value
        for name, value in formula_environment.items()
        if not isinstance(value, _TemporalFormulaValue)
    }
    binding_environment.update(execution.constant_environment)
    operative_leaf = _simplified_formula_text(
        execution.leaf,
        environment=binding_environment,
    )
    clamp_subject_names = _formula_progressive_clamp_subject_names(operative_leaf)
    if interval is not None and clamp_subject_names:
        if len(clamp_subject_names) != 1 or not _formula_subject_matches_source(
            clamp_subject_names[0],
            branch.text,
        ):
            return False
    source_topology = _explicit_source_arithmetic_topology(
        authoritative_numeric_recall_text(branch.text)
    )
    if source_topology is not None and source_topology != _formula_arithmetic_topology(
        operative_leaf,
        environment=binding_environment,
    ):
        return False
    artifact_operations = _formula_ast_operation_kinds(
        operative_leaf,
        environment=execution_environment,
    )
    if not _formula_operations_are_compatible(
        source_operations,
        artifact_operations,
        source_text=branch.text,
        artifact_formula=operative_leaf,
        artifact_environment=binding_environment,
    ):
        return False
    source_multiplier = _source_named_multiplier(branch.text)
    if source_multiplier is not None and (
        source_multiplier.value is None
        or not (
            _formula_has_numeric_factor(
                operative_leaf,
                binding_environment,
                source_multiplier.value,
            )
            or (
                math.isclose(source_multiplier.value, 2.0)
                and _formula_is_duplicate_addition(operative_leaf)
            )
        )
    ):
        return False
    source_divisor = _source_named_divisor(branch.text)
    if source_divisor is not None:
        if source_divisor.value is None:
            return False
        has_source_divisor = _formula_has_numeric_divisor(
            operative_leaf,
            binding_environment,
            source_divisor.value,
        )
        has_equivalent_half_factor = (
            math.isclose(source_divisor.value, 2.0)
            and _source_describes_half(branch.text)
            and _formula_has_numeric_factor(
                operative_leaf,
                binding_environment,
                0.5,
            )
        )
        if not (has_source_divisor or has_equivalent_half_factor):
            return False
    source_delta = _source_named_delta(branch.text)
    if source_delta is not None:
        if source_delta[1].value is None or not _formula_has_numeric_delta(
            operative_leaf,
            binding_environment,
            operation=source_delta[0],
            expected=source_delta[1].value,
        ):
            return False
    if _source_describes_half(branch.text):
        if not (
            _formula_has_numeric_factor(
                operative_leaf,
                binding_environment,
                0.5,
            )
            or _formula_has_numeric_divisor(
                operative_leaf,
                binding_environment,
                2.0,
            )
        ):
            return False
    computation_occurrences = _formula_branch_computation_occurrences(
        branch,
        interval=interval,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    if not computation_occurrences:
        return True

    leaf_names = set(_FORMULA_IDENTIFIER.findall(operative_leaf))
    candidate_values = [
        float(value)
        for name, value in binding_environment.items()
        if name in leaf_names
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]
    candidate_values.extend(
        float(occurrence.value)
        for occurrence in extract_numeric_occurrences(operative_leaf)
    )
    if (
        "multiply" in source_operations
        and "add" in artifact_operations
        and _source_describes_doubling(branch.text)
        and _formula_is_duplicate_addition(operative_leaf)
    ):
        candidate_values.append(2.0)
    return bool(candidate_values) and all(
        any(
            numeric_value_is_grounded(value, (source_occurrence,))
            for value in candidate_values
        )
        for source_occurrence in computation_occurrences
    )


def _temporal_occurrence_is_formula_applicability_preface(
    occurrence: NumericOccurrenceLike,
    source_text: str,
) -> bool:
    """Separate leading temporal applicability from arithmetic operands."""

    starts = [0]
    starts.extend(
        separator.end()
        for separator in re.finditer(
            r"(?:[;\n]+|(?<=[.!?])\s+)",
            source_text,
        )
    )
    for start in starts:
        preface = _FORMULA_APPLICABILITY_PREFACE.match(source_text[start:])
        if preface is None:
            continue
        if (
            occurrence.start >= start + preface.start()
            and occurrence.end <= start + preface.end()
        ):
            return True
    return False


def _formula_operation_kinds(text: str) -> set[str]:
    """Recognize operations in source prose or an explicit expression."""

    parsed_operations = _formula_ast_operation_kinds(text)
    if parsed_operations:
        return parsed_operations
    operations: set[str] = set()
    lowered_text = text.lower()
    operation_patterns = {
        "add": (
            r"(?:\+|\bplus\b|\bsumme\b|\bsum\s+of\b|\bzuzüglich\b|"
            r"\b(?:summieren|addieren)\b|"
            r"\berhöh\w*\s+(?:sich\s+)?um\b|"
            r"\bum\s+(?:\d+(?:[.,]\d+)?|[a-zäöüß]+)\s+zu\s+"
            r"(?:erhöhen|vermehren)\b)"
        ),
        "subtract": (
            r"(?:\s[−–-]\s|\bminus\b|\bunterschied\b|\bdifferenz\b|"
            r"\bdifference\s+between\b|\babzüglich\b|"
            r"\b(?:vermindern|kürzen)\b|"
            r"\b(?:vermindert|gekürzt|mindert|kürzt)\w*\s+"
            r"(?:sich\s+)?um\b|"
            r"\bum\s+(?:\d+(?:[.,]\d+)?|[a-zäöüß]+)\s+zu\s+"
            r"(?:vermindern|kürzen)\b)"
        ),
        "multiply": (
            r"(?:[*×·•∗∙]|\d+(?:[.,]\d+)?\s*%\s+(?:des|der|von|of)\b|"
            r"\bprodukt\b|\bproduct\s+of\b|"
            r"\b(?:multiplied|multiply|multiplying|multiplication|multipliziert|"
            r"multiplizieren|multiplikation)\b|\bvervielfach\w*\b|\bmal\b|"
            r"\b(?:verfünf|versechs|versieben|veracht|verneun|verzehn)"
            r"fach\w*\b|"
            r"\b(?:doppelte|zweifache|dreifache|twice)\b)"
        ),
        "divide": (
            r"(?:/|\bgeteilt\b|\bteilen\b|\bdivided\b|"
            r"\bhälfte\b|\bhalbier\w*\b|\bhalbierung\b|\bhalf\s+of\b)"
        ),
    }
    operations.update(
        operation
        for operation, pattern in operation_patterns.items()
        if re.search(pattern, lowered_text, flags=re.IGNORECASE)
    )
    return operations


def _explicit_source_arithmetic_topology(text: str) -> Any | None:
    """Parse the largest complete symbolic expression, including grouping."""

    tokens = tuple(_SYMBOLIC_ARITHMETIC_TOKEN.finditer(text))
    candidates: list[tuple[int, int, str]] = []
    for start_index, start_token in enumerate(tokens):
        for end_token in tokens[start_index:]:
            expression_text = _normalized_source_arithmetic_expression(
                text[start_token.start() : end_token.end()]
            )
            with contextlib.suppress(SyntaxError):
                expression = ast.parse(expression_text, mode="eval").body
                if not _is_supported_arithmetic_expression(expression):
                    continue
                operation_count = sum(
                    isinstance(node, ast.BinOp) for node in ast.walk(expression)
                )
                if operation_count >= 2:
                    candidates.append(
                        (
                            operation_count,
                            end_token.end() - start_token.start(),
                            expression_text,
                        )
                    )
    if not candidates:
        return None
    _operation_count, _length, expression_text = max(
        candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    return _formula_arithmetic_topology(expression_text, environment={})


def _normalized_source_arithmetic_expression(text: str) -> str:
    normalized = re.sub(
        r"\d{1,3}(?:[ .]\d{3})+(?:,\d+)?|\d+,\d+",
        lambda number: (
            number.group(0)
            .replace(" ", "")
            .replace(".", "")
            .replace(
                ",",
                ".",
            )
        ),
        text,
    )
    normalized = re.sub(r"\bplus\b", "+", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bminus\b", "-", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bmal\b", "*", normalized, flags=re.IGNORECASE)
    return normalized.translate(
        str.maketrans(
            {
                "×": "*",
                "·": "*",
                "•": "*",
                "∗": "*",
                "∙": "*",
                "−": "-",
                "–": "-",
            }
        )
    )


def _is_supported_arithmetic_expression(expression: ast.expr) -> bool:
    return all(
        isinstance(
            node,
            (
                ast.BinOp,
                ast.UnaryOp,
                ast.Name,
                ast.Constant,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.USub,
                ast.UAdd,
                ast.Load,
            ),
        )
        for node in ast.walk(expression)
    )


def _formula_arithmetic_topology(
    formula: str,
    *,
    environment: dict[str, Any],
) -> Any | None:
    """Canonicalize arithmetic shape with bound scalars and abstract variables."""

    with contextlib.suppress(SyntaxError):
        expression = ast.parse(formula.strip(), mode="eval").body
        expression = _unwrap_formula_result_wrapper(expression)
        topology = _formula_arithmetic_topology_node(expression, environment)
        return _canonicalize_topology_variables(topology)
    return None


def _formula_arithmetic_topology_node(
    node: ast.expr,
    environment: dict[str, Any],
) -> Any:
    if isinstance(node, ast.Name):
        known = _known_numeric_formula_value(node, environment)
        if known is not None:
            return "constant", float(known)
        return "variable", node.id
    if isinstance(node, ast.Constant):
        known = _known_numeric_formula_value(node, environment)
        if known is not None:
            return "constant", float(known)
        return "unsupported", type(node.value).__name__
    if isinstance(node, ast.BinOp):
        operator = type(node.op).__name__
        operands: tuple[Any, ...] = (
            _formula_arithmetic_topology_node(node.left, environment),
            _formula_arithmetic_topology_node(node.right, environment),
        )
        if isinstance(node.op, (ast.Add, ast.Mult)):
            flattened: list[Any] = []
            for operand in operands:
                if (
                    isinstance(operand, tuple)
                    and len(operand) == 3
                    and operand[:2] == ("binop", operator)
                ):
                    flattened.extend(operand[2])
                else:
                    flattened.append(operand)
            operands = tuple(
                sorted(
                    flattened,
                    key=lambda operand: repr(_topology_without_variable_names(operand)),
                )
            )
        return "binop", operator, operands
    if isinstance(node, ast.UnaryOp):
        known = _known_numeric_formula_value(node, environment)
        if known is not None:
            return "constant", float(known)
        return (
            "unary",
            type(node.op).__name__,
            _formula_arithmetic_topology_node(node.operand, environment),
        )
    return ("unsupported", type(node).__name__)


def _topology_without_variable_names(topology: Any) -> Any:
    if isinstance(topology, tuple) and len(topology) == 2 and topology[0] == "variable":
        return ("variable",)
    if isinstance(topology, tuple):
        return tuple(_topology_without_variable_names(item) for item in topology)
    return topology


def _canonicalize_topology_variables(topology: Any) -> Any:
    identities: dict[str, int] = {}

    def canonicalize(item: Any) -> Any:
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "variable":
            name = str(item[1])
            identity = identities.setdefault(name, len(identities))
            return "variable", identity
        if isinstance(item, tuple):
            return tuple(canonicalize(part) for part in item)
        return item

    return canonicalize(topology)


def _formula_ast_operation_kinds(
    text: str,
    *,
    environment: dict[str, Any] | None = None,
) -> set[str]:
    operations: set[str] = set()
    expression = _parse_formula_expression(text)
    if expression is None:
        return operations

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                operations.add("add")
            elif isinstance(node.op, ast.Sub):
                operations.add("subtract")
            elif isinstance(node.op, ast.Mult):
                operations.add("multiply")
            elif isinstance(node.op, ast.Div):
                operations.add("divide")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            lowered = node.func.id.lower()
            if "sum" in lowered:
                operations.add("add")
            elif "product" in lowered:
                operations.add("multiply")
        if isinstance(node, ast.BoolOp) and environment is not None:
            for item in node.values:
                visit(item)
                value = _evaluate_condition_expression(item, environment)
                if value is _UNRESOLVED_CONDITION_VALUE:
                    return
                if isinstance(node.op, ast.And) and not bool(value):
                    return
                if isinstance(node.op, ast.Or) and bool(value):
                    return
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(expression)
    return operations


def _formula_operations_are_compatible(
    source_operations: set[str],
    artifact_operations: set[str],
    *,
    source_text: str,
    artifact_formula: str,
    artifact_environment: dict[str, Any],
) -> bool:
    if not source_operations:
        return True
    for operation in source_operations:
        if operation in artifact_operations:
            continue
        if (
            operation == "divide"
            and "multiply" in artifact_operations
            and _source_describes_half(source_text)
            and _formula_has_numeric_factor(
                artifact_formula,
                artifact_environment,
                0.5,
            )
        ):
            continue
        if (
            operation == "multiply"
            and "add" in artifact_operations
            and _source_describes_doubling(source_text)
            and _formula_is_duplicate_addition(artifact_formula)
        ):
            continue
        return False
    return True


def _source_describes_doubling(text: str) -> bool:
    multiplier = _source_named_multiplier(text)
    return (
        multiplier is not None
        and multiplier.value is not None
        and math.isclose(multiplier.value, 2.0)
    )


def _source_named_multiplier(text: str) -> _NamedSourceNumber | None:
    patterns = (
        (
            2.0,
            r"\b(?:doppelte|zweifache|verdoppelt|verdoppeln|"
            r"twice|double[ds]?)\b",
        ),
        (
            3.0,
            r"\b(?:dreifache|verdreifacht|verdreifachen|"
            r"threefold|triple[ds]?)\b",
        ),
        (
            4.0,
            r"\b(?:vierfache|vervierfacht|vervierfachen|"
            r"fourfold|quadruple[ds]?)\b",
        ),
        (5.0, r"\b(?:fünffache|verfünffachen|fivefold)\b"),
        (6.0, r"\b(?:sechsfache|versechsfachen|sixfold)\b"),
        (7.0, r"\b(?:siebenfache|versiebenfachen|sevenfold)\b"),
        (8.0, r"\b(?:achtfache|verachtfachen|eightfold)\b"),
        (9.0, r"\b(?:neunfache|verneunfachen|ninefold)\b"),
        (10.0, r"\b(?:zehnfache|verzehnfachen|tenfold)\b"),
    )
    named_multiplier = next(
        (
            multiplier
            for multiplier, pattern in patterns
            if re.search(pattern, text, flags=re.IGNORECASE)
        ),
        None,
    )
    if named_multiplier is not None:
        return _NamedSourceNumber(named_multiplier)
    return _source_contextual_number_word(
        text,
        patterns=(
            r"\bmal\s+(?P<number>[a-zäöüß]+)\b",
            r"\bmit\s+(?:(?:dem|einem)\s+faktor\s+)?"
            r"(?:von\s+)?(?P<number>[a-zäöüß]+)\s+zu\s+"
            r"(?:multiplizieren|vervielfachen)\b",
            r"\bdurch\s+multiplikation\s+mit\s+"
            r"(?:(?:(?:dem|einem)\s+)?faktor\s+)?(?:von\s+)?"
            r"(?P<number>[a-zäöüß]+)\b",
            r"\bunter\s+anwendung\s+(?:des|eines)\s+faktors?\s+"
            r"(?P<number>[a-zäöüß]+)\b",
        ),
    )


def _source_named_divisor(text: str) -> _NamedSourceNumber | None:
    return _source_contextual_number_word(
        text,
        patterns=(
            r"\bdurch\s+(?P<number>[a-zäöüß]+)\s+zu\s+teilen\b",
            r"\bgeteilt\s+durch\s+(?P<number>[a-zäöüß]+)\b",
            r"\bdurch\s+(?P<number>[a-zäöüß]+)\s+geteilt\b",
            r"\bin\s+(?P<number>[a-zäöüß]+)\s+gleiche\s+teile\s+"
            r"zu\s+teilen\b",
        ),
    )


def _source_named_delta(
    text: str,
) -> tuple[str, _NamedSourceNumber] | None:
    number = r"(?P<number>\d+(?:[.,]\d+)?|[a-zäöüß]+)"
    patterns = (
        (
            "add",
            rf"\bum\s+{number}\s+zu\s+(?:erhöhen|vermehren)\b",
        ),
        (
            "add",
            rf"\b(?:erhöht|vermehrt)\w*\s+(?:sich\s+)?um\s+{number}\b",
        ),
        (
            "add",
            rf"\bwird\s+um\s+{number}\s+(?:erhöht|vermehrt)\b",
        ),
        (
            "subtract",
            rf"\bum\s+{number}\s+zu\s+(?:vermindern|kürzen)\b",
        ),
        (
            "subtract",
            rf"\b(?:vermindert|gekürzt|mindert|kürzt)\w*\s+"
            rf"(?:sich\s+)?um\s+{number}\b",
        ),
        (
            "subtract",
            rf"\bwird\s+um\s+{number}\s+(?:vermindert|gekürzt)\b",
        ),
    )
    for operation, pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        token = match.group("number")
        value = _source_number_token_value(token)
        if value is None and _source_number_token_is_symbolic(token):
            return None
        return operation, _NamedSourceNumber(value)
    return None


def _source_number_token_value(token: str) -> float | None:
    normalized = token.lower()
    if normalized in _GERMAN_CARDINAL_VALUES:
        return _GERMAN_CARDINAL_VALUES[normalized]
    with contextlib.suppress(ValueError):
        return float(normalized.replace(",", "."))
    return None


def _source_number_token_is_symbolic(token: str) -> bool:
    return token.isalpha() and (len(token) == 1 or token.isupper())


def _source_contextual_number_word(
    text: str,
    *,
    patterns: Sequence[str],
) -> _NamedSourceNumber | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            token = match.group("number")
            if _source_number_token_is_symbolic(token):
                return None
            return _NamedSourceNumber(_GERMAN_CARDINAL_VALUES.get(token.lower()))
    return None


def _source_describes_half(text: str) -> bool:
    text = _ENGLISH_FRACTIONAL_PERCENTAGE_OF.sub(" ", text)
    return bool(
        re.search(
            r"\b(?:hälfte|halb(?:e[nsrm]?)?|halbier\w*|halbierung|half)\b|"
            r"\bin\s+(?:2|zwei)\s+gleiche\s+teile\s+zu\s+teilen\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _formula_has_numeric_factor(
    text: str,
    environment: dict[str, Any],
    expected: float,
) -> bool:
    with contextlib.suppress(SyntaxError):
        expression = ast.parse(text.strip(), mode="eval").body
        for node in ast.walk(expression):
            if not isinstance(node, ast.BinOp) or not isinstance(
                node.op,
                ast.Mult,
            ):
                continue
            for operand in (node.left, node.right):
                value = _known_numeric_formula_value(operand, environment)
                if value is not None and math.isclose(float(value), expected):
                    return True
    return False


def _formula_has_numeric_divisor(
    text: str,
    environment: dict[str, Any],
    expected: float,
) -> bool:
    with contextlib.suppress(SyntaxError):
        expression = ast.parse(text.strip(), mode="eval").body
        for node in ast.walk(expression):
            if not isinstance(node, ast.BinOp) or not isinstance(
                node.op,
                ast.Div,
            ):
                continue
            value = _known_numeric_formula_value(node.right, environment)
            if value is not None and math.isclose(float(value), expected):
                return True
    return False


def _formula_has_numeric_delta(
    text: str,
    environment: dict[str, Any],
    *,
    operation: str,
    expected: float,
) -> bool:
    with contextlib.suppress(SyntaxError):
        expression = ast.parse(text.strip(), mode="eval").body
        expression = _unwrap_formula_result_wrapper(expression)
        if not isinstance(expression, ast.BinOp) or not isinstance(
            expression.op,
            (ast.Add, ast.Sub),
        ):
            return False
        constant, has_dynamic_operand = _formula_additive_constant(
            expression,
            environment,
        )
        signed_expected = expected if operation == "add" else -expected
        return has_dynamic_operand and math.isclose(
            constant,
            signed_expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    return False


def _unwrap_formula_result_wrapper(expression: ast.expr) -> ast.expr:
    while (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id in {"ceil", "floor"}
        and len(expression.args) == 1
        and not expression.keywords
    ):
        expression = expression.args[0]
    return expression


def _formula_additive_constant(
    expression: ast.expr,
    environment: dict[str, Any],
) -> tuple[float, bool]:
    known = _known_numeric_formula_value(expression, environment)
    if known is not None:
        return float(known), False
    if isinstance(expression, ast.BinOp) and isinstance(
        expression.op,
        (ast.Add, ast.Sub),
    ):
        left_constant, left_dynamic = _formula_additive_constant(
            expression.left,
            environment,
        )
        right_constant, right_dynamic = _formula_additive_constant(
            expression.right,
            environment,
        )
        return (
            (
                left_constant + right_constant
                if isinstance(expression.op, ast.Add)
                else left_constant - right_constant
            ),
            left_dynamic or right_dynamic,
        )
    return 0.0, True


def _formula_is_duplicate_addition(text: str) -> bool:
    with contextlib.suppress(SyntaxError):
        expression = ast.parse(text.strip(), mode="eval").body
        if isinstance(expression, ast.BinOp) and isinstance(
            expression.op,
            ast.Add,
        ):
            return _canonical_formula_node(expression.left) == _canonical_formula_node(
                expression.right
            )
    return False


def _formula_leaf_semantic_key(
    leaf: str,
    *,
    formula_environment: dict[str, Any] | None = None,
) -> str:
    leaf = _simplified_formula_text(
        leaf,
        environment=formula_environment or {},
    )
    with contextlib.suppress(SyntaxError):
        expression = ast.parse(leaf.strip(), mode="eval").body
        return repr(_canonical_formula_node(expression))
    return _collapse_text(leaf).lower()


def _simplified_formula_text(
    text: str,
    *,
    environment: dict[str, Any],
) -> str:
    expression = _parse_formula_expression(text)
    if expression is None:
        return text
    simplified = _simplify_formula_expression(
        expression,
        environment=environment,
    )
    return ast.unparse(ast.fix_missing_locations(simplified))


def _simplify_formula_expression(
    expression: ast.expr,
    *,
    environment: dict[str, Any],
) -> ast.expr:
    if isinstance(expression, ast.BinOp):
        left = _simplify_formula_expression(
            expression.left,
            environment=environment,
        )
        right = _simplify_formula_expression(
            expression.right,
            environment=environment,
        )
        left_value = _known_numeric_formula_value(left, environment)
        right_value = _known_numeric_formula_value(right, environment)
        if isinstance(expression.op, ast.Mult):
            if left_value == 0 or right_value == 0:
                return ast.Constant(value=0)
            if left_value == 1:
                return right
            if right_value == 1:
                return left
        elif isinstance(expression.op, ast.Add):
            if left_value == 0:
                return right
            if right_value == 0:
                return left
        elif isinstance(expression.op, ast.Sub) and right_value == 0:
            return left
        elif isinstance(expression.op, ast.Div):
            if right_value == 1:
                return left
            if left_value == 0 and _formula_node_is_provably_nonzero(
                right,
                environment,
            ):
                return ast.Constant(value=0)
        return ast.BinOp(left=left, op=expression.op, right=right)
    if isinstance(expression, ast.UnaryOp):
        return ast.UnaryOp(
            op=expression.op,
            operand=_simplify_formula_expression(
                expression.operand,
                environment=environment,
            ),
        )
    if isinstance(expression, ast.Call):
        return ast.Call(
            func=expression.func,
            args=[
                _simplify_formula_expression(
                    argument,
                    environment=environment,
                )
                for argument in expression.args
            ],
            keywords=expression.keywords,
        )
    if isinstance(expression, ast.Compare):
        return ast.Compare(
            left=_simplify_formula_expression(
                expression.left,
                environment=environment,
            ),
            ops=expression.ops,
            comparators=[
                _simplify_formula_expression(
                    comparator,
                    environment=environment,
                )
                for comparator in expression.comparators
            ],
        )
    if isinstance(expression, ast.BoolOp):
        return ast.BoolOp(
            op=expression.op,
            values=[
                _simplify_formula_expression(
                    value,
                    environment=environment,
                )
                for value in expression.values
            ],
        )
    return expression


def _known_numeric_formula_value(
    expression: ast.expr,
    environment: dict[str, Any],
) -> int | float | Decimal | None:
    value = _evaluate_condition_expression(expression, environment)
    if _rulespec_runtime_decimal(value) is not None:
        return value
    return None


def _formula_node_is_provably_nonzero(
    expression: ast.expr,
    environment: dict[str, Any],
) -> bool:
    value = _known_numeric_formula_value(expression, environment)
    if value is not None:
        return value != 0
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "max"
    ):
        return any(
            (known := _known_numeric_formula_value(argument, environment)) is not None
            and known > 0
            for argument in expression.args
        )
    return False


def _canonical_formula_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Name):
        return "name", node.id.lower()
    if isinstance(node, ast.Constant):
        return "constant", type(node.value).__name__, repr(node.value)
    if isinstance(node, ast.BinOp):
        operator = type(node.op).__name__
        if isinstance(node.op, (ast.Add, ast.Mult)):
            operands = _commutative_formula_operands(node, type(node.op))
            return (
                "binop",
                operator,
                tuple(sorted(repr(operand) for operand in operands)),
            )
        return (
            "binop",
            operator,
            _canonical_formula_node(node.left),
            _canonical_formula_node(node.right),
        )
    if isinstance(node, ast.UnaryOp):
        return (
            "unary",
            type(node.op).__name__,
            _canonical_formula_node(node.operand),
        )
    if isinstance(node, ast.Call):
        return (
            "call",
            _canonical_formula_node(node.func),
            tuple(_canonical_formula_node(argument) for argument in node.args),
        )
    if isinstance(node, ast.Compare):
        return (
            "compare",
            _canonical_formula_node(node.left),
            tuple(type(operator).__name__ for operator in node.ops),
            tuple(
                _canonical_formula_node(comparator) for comparator in node.comparators
            ),
        )
    return ast.dump(node, annotate_fields=True, include_attributes=False).lower()


def _commutative_formula_operands(
    node: ast.BinOp,
    operator_type: type[ast.operator],
) -> tuple[Any, ...]:
    operands: list[Any] = []

    def collect(candidate: ast.AST) -> None:
        if isinstance(candidate, ast.BinOp) and isinstance(candidate.op, operator_type):
            collect(candidate.left)
            collect(candidate.right)
        else:
            operands.append(_canonical_formula_node(candidate))

    collect(node)
    return tuple(operands)


def _numeric_occurrences_are_equivalent(
    left: NumericOccurrenceLike,
    right: NumericOccurrenceLike,
) -> bool:
    return (
        math.isclose(float(left.value), float(right.value))
        and left.has_rate_context == right.has_rate_context
        and left.has_temporal_context == right.has_temporal_context
        and left.has_structural_context == right.has_structural_context
        and left.source_value == right.source_value
        and left.requires_rate_context == right.requires_rate_context
        and left.is_word_number == right.is_word_number
        and left.alternative_values == right.alternative_values
    )


def _rule_has_branching_formula(rule: dict[str, Any]) -> bool:
    return _rule_text_has_branching_formula(_rule_formula_text(rule))


_UNRESOLVED_CONDITION_VALUE = object()


def _case_formula_execution(
    rule: dict[str, Any],
    case: dict[str, Any],
    *,
    formula_environment: dict[str, Any] | None = None,
    dependency_environment: dict[str, Any] | None = None,
) -> _FormulaExecution | None:
    """Resolve the reachable RuleSpec formula for one asserted test case."""

    inputs = case.get("input")
    if not isinstance(inputs, dict):
        return None
    constant_environment = _formula_environment_for_case(
        formula_environment or {},
        case,
    )
    environment: dict[str, Any] = dict(constant_environment)
    for name, value in (dependency_environment or {}).items():
        if name in environment and not _formula_runtime_values_equal(
            environment[name],
            value,
        ):
            return None
        environment[name] = value
    input_environment = _case_input_formula_environment(case)
    if input_environment is None:
        return None
    for name, value in input_environment.items():
        if name in environment and not _formula_runtime_values_equal(
            environment[name],
            value,
        ):
            return None
        environment[name] = value
    formula_text = _rule_formula_text_for_case(rule, case)
    if formula_text is None:
        return None
    return _execute_formula_text(
        formula_text,
        environment=environment,
        constant_environment=constant_environment,
    )


def _case_input_formula_environment(
    case: dict[str, Any],
) -> dict[str, Any] | None:
    inputs = case.get("input")
    if not isinstance(inputs, dict):
        return None
    input_environment: dict[str, Any] = {}
    for key, value in inputs.items():
        boolean_value = _boolean_value(value)
        normalized_value = boolean_value if boolean_value is not None else value
        for name in _input_key_names(key):
            if not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*",
                name,
            ):
                continue
            if name in input_environment and not _same_formula_value(
                input_environment[name],
                normalized_value,
            ):
                return None
            input_environment[name] = normalized_value
    return input_environment


def _case_asserted_dependency_environment(
    principal_rules: dict[str, dict[str, Any]],
    case: dict[str, Any],
    *,
    formula_environment: dict[str, Any],
) -> dict[str, Any]:
    """Resolve only derived values corroborated by this case's assertions."""

    return _case_dependency_environment(
        principal_rules,
        case,
        formula_environment=formula_environment,
        require_asserted_value=True,
    )


def _case_dependency_environment(
    principal_rules: dict[str, dict[str, Any]],
    case: dict[str, Any],
    *,
    formula_environment: dict[str, Any],
    require_asserted_value: bool,
    allowed_names: set[str] | None = None,
) -> dict[str, Any]:
    """Evaluate an acyclic reached subset of local principal outputs."""

    constants = _formula_environment_for_case(formula_environment, case)
    inputs = _case_input_formula_environment(case)
    if inputs is None:
        return {}
    candidates = {
        name: rule
        for name, rule in principal_rules.items()
        if allowed_names is None or name in allowed_names
    }
    environment = dict(constants)
    for name, value in inputs.items():
        if name in candidates:
            continue
        if name in environment and not _formula_runtime_values_equal(
            environment[name],
            value,
        ):
            return {}
        environment[name] = value

    resolved: dict[str, Any] = {}
    for _ in range(len(candidates) + 1):
        changed = False
        for name, rule in candidates.items():
            if name in resolved or name in environment:
                continue
            formula_text = _rule_formula_text_for_case(rule, case)
            if formula_text is None:
                continue
            execution = _execute_formula_text(
                formula_text,
                environment=environment,
                constant_environment=constants,
            )
            if execution is None:
                continue
            value = _evaluate_formula_selector(execution.leaf, environment)
            if value is _UNRESOLVED_CONDITION_VALUE:
                continue
            if require_asserted_value:
                asserted = _test_case_asserted_output_value(case, name)
                if (
                    asserted is _UNRESOLVED_CONDITION_VALUE
                    or not _formula_runtime_values_equal(value, asserted)
                ):
                    continue
            resolved[name] = value
            environment[name] = value
            changed = True
        if not changed:
            break
    return resolved


def _formula_runtime_values_equal(left: Any, right: Any) -> bool:
    """Mirror RuleSpec scalar equality for evidence-bearing runtime values."""

    left_boolean = _boolean_value(left)
    right_boolean = _boolean_value(right)
    if left_boolean is not None or right_boolean is not None:
        return left_boolean is not None and left_boolean == right_boolean
    left_number = _rulespec_runtime_decimal(left)
    right_number = _rulespec_runtime_decimal(right)
    if left_number is not None and right_number is not None:
        return left_number == right_number
    return type(left) is type(right) and left == right


def _execute_formula_text(
    formula_text: str,
    *,
    environment: dict[str, Any],
    constant_environment: dict[str, Any],
) -> _FormulaExecution | None:
    """Interpret formula control flow without evaluating the formula itself."""

    selected_text = formula_text.strip()
    trace: list[_FormulaTraceStep] = []
    for _depth in range(32):
        node = _first_formula_branch_node(selected_text)
        if node is None:
            if _rule_text_has_branching_formula(selected_text):
                return None
            leaf = selected_text.strip()
            if not _formula_leaf_has_executable_syntax(leaf):
                return None
            evaluated = _evaluate_formula_selector(leaf, environment)
            value_signature = (
                None
                if evaluated is _UNRESOLVED_CONDITION_VALUE
                else (type(evaluated).__name__, repr(evaluated))
            )
            evaluates_to_zero = (
                evaluated is None
                or evaluated is False
                or (
                    (number := _rulespec_runtime_decimal(evaluated)) is not None
                    and number == 0
                )
                or _formula_expression_is_definitely_zero(
                    leaf,
                    environment=environment,
                )
            )
            return _FormulaExecution(
                tuple(trace),
                leaf,
                value_signature,
                evaluates_to_zero,
                constant_environment,
            )
        if not all(
            _formula_text_has_executable_branch_tree(selector)
            for selector in node.selectors
        ):
            return None
        if not all(
            _formula_text_has_executable_branch_tree(choice) for choice in node.choices
        ):
            return None
        selected = _select_formula_branch(
            node,
            environment=environment,
            constant_environment=constant_environment,
        )
        if selected is None:
            return None
        choice, evaluated_selectors, selector_trace = selected
        selected_body = textwrap.dedent(node.choices[choice]).strip()
        if not selected_body:
            return None
        trace.extend(selector_trace)
        trace.append(
            _FormulaTraceStep(
                node.kind,
                evaluated_selectors,
                choice,
            )
        )
        selected_text = (
            selected_text[: node.start] + selected_body + selected_text[node.end :]
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
    masked_text = _mask_formula_strings_and_comments(text)
    masked_lines = _formula_line_records(masked_text)
    for index, (start, _content_end, _full_end, _line) in enumerate(lines):
        if not _formula_branch_is_at_expression_entry(masked_text, start):
            continue
        header = _multiline_condition_header(
            lines,
            masked_lines,
            index=index,
            keyword="if",
        )
        if header is None:
            continue
        base_indent, header_end, condition, inline_body = header
        inline_node = _continued_header_inline_if_node(
            inline_body,
            condition=condition,
            start=start,
            end=lines[header_end][1],
            trailing_body=text[lines[header_end][1] :],
            trailing_end=len(text),
        )
        if inline_node is not None:
            return inline_node
        headers: list[tuple[int, int, str, str, str]] = [
            (index, header_end, "if", condition, inline_body)
        ]
        valid_chain = True
        saw_else = False
        cursor = header_end + 1
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
            chain_header = None
            if indent == base_indent:
                chain_header = _multiline_condition_header(
                    lines,
                    masked_lines,
                    index=cursor,
                    keyword="elif",
                )
            if chain_header is not None and chain_header[0] == base_indent:
                if saw_else:
                    valid_chain = False
                    break
                _indent, chain_end, chain_condition, chain_inline_body = chain_header
                headers.append(
                    (cursor, chain_end, "elif", chain_condition, chain_inline_body)
                )
                cursor = chain_end + 1
                continue
            else_header = (
                re.match(
                    r"^[ \t]*else:(?P<body>.*)$",
                    masked_candidate,
                )
                if indent == base_indent
                else None
            )
            if else_header is not None:
                if saw_else:
                    valid_chain = False
                    break
                inline_body = candidate[else_header.start("body") :].strip()
                if not _mask_formula_comments(inline_body).strip():
                    inline_body = ""
                headers.append((cursor, cursor, "else", "", inline_body))
                saw_else = True
                cursor += 1
                continue
            break
        if not valid_chain or not saw_else or headers[-1][2] != "else":
            continue
        chain_end = lines[cursor][0] if cursor < len(lines) else len(text)
        final_body_start = lines[headers[-1][1]][2]
        chain_end = _formula_expression_end(
            text,
            body_start=final_body_start,
            branch_start=start,
            limit=len(text),
        )
        conditions = tuple(
            condition
            for _start_line, _end_line, kind, condition, _inline_body in headers
            if kind != "else"
        )
        choices: list[str] = []
        for header_index, (
            _line_index,
            line_end_index,
            _kind,
            _condition,
            inline_body,
        ) in enumerate(headers):
            body_start = lines[line_end_index][2]
            body_end = (
                lines[headers[header_index + 1][0]][0]
                if header_index + 1 < len(headers)
                else chain_end
            )
            trailing_body = text[body_start:body_end]
            choices.append(
                "\n".join(part for part in (inline_body, trailing_body) if part.strip())
            )
        if (
            not conditions
            or not choices
            or any(
                not _mask_formula_comments(choice).strip()
                or (
                    not _rule_text_has_branching_formula(choice)
                    and not _formula_leaf_has_executable_syntax(
                        textwrap.dedent(choice).strip()
                    )
                )
                for choice in choices
            )
        ):
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


def _continued_header_inline_if_node(
    inline_body: str,
    *,
    condition: str,
    start: int,
    end: int,
    trailing_body: str,
    trailing_end: int,
) -> _FormulaBranchNode | None:
    if not inline_body:
        return None
    masked_body = _mask_formula_strings_and_comments(inline_body)
    chain_headers = _find_inline_chain_headers(
        masked_body,
        body_start=0,
        limit=len(inline_body),
    )
    if not chain_headers or chain_headers[-1][0] != "else":
        return None
    conditions = [condition]
    conditions.extend(
        inline_body[condition_start:condition_end].strip()
        for kind, _header_start, _body_start, condition_start, condition_end in chain_headers
        if kind == "elif"
    )
    body_boundaries = [
        header_start
        for _kind, header_start, _body_start, _condition_start, _condition_end in chain_headers
    ]
    body_starts = [
        0,
        *(
            header_body_start
            for kind, _header_start, header_body_start, _start, _end in chain_headers
            if kind == "elif"
        ),
    ]
    choices = [
        inline_body[body_start:body_end]
        for body_start, body_end in zip(body_starts, body_boundaries)
    ]
    else_body_start = chain_headers[-1][2]
    complete_body = inline_body + trailing_body
    inline_end = _formula_expression_end(
        complete_body,
        body_start=else_body_start,
        branch_start=0,
        limit=len(complete_body),
    )
    choices.append(complete_body[else_body_start:inline_end])
    node_end = end - (len(inline_body) - inline_end)
    if (
        not _mask_formula_comments(choices[-1]).strip()
        and _mask_formula_comments(trailing_body).strip()
    ):
        choices[-1] = trailing_body
        node_end = trailing_end
    if any(not _mask_formula_comments(choice).strip() for choice in choices):
        return None
    return _FormulaBranchNode(
        start,
        node_end,
        "if",
        tuple(conditions),
        (),
        tuple(choices),
    )


def _multiline_condition_header(
    lines: tuple[tuple[int, int, int, str], ...],
    masked_lines: tuple[tuple[int, int, int, str], ...],
    *,
    index: int,
    keyword: str,
) -> tuple[int, int, str, str] | None:
    """Parse one possibly continued ``if`` or ``elif`` header."""

    line = lines[index][3]
    masked_line = masked_lines[index][3]
    header = re.match(
        rf"^(?P<indent>[ \t]*){keyword}[ \t]+",
        line,
    )
    if header is None:
        return None
    base_indent = _formula_indent_width(header.group("indent"))
    source_parts: list[str] = []
    masked_parts: list[str] = []
    cursor = index
    while cursor < len(lines) and cursor - index < 32:
        if cursor == index:
            source_part = line[header.end() :]
            masked_part = masked_line[header.end() :]
        else:
            source_line = lines[cursor][3]
            masked_source_line = masked_lines[cursor][3]
            if not source_line.strip():
                return None
            indent_text = source_line[: len(source_line) - len(source_line.lstrip())]
            if _formula_indent_width(indent_text) < base_indent:
                return None
            content_start = len(source_line) - len(source_line.lstrip())
            source_part = source_line[content_start:]
            masked_part = masked_source_line[content_start:]
            previous = "\n".join(masked_parts).rstrip()
            if not (
                _formula_bracket_stack(previous)
                or re.search(r"\b(?:and|or)\s*$", previous)
                or re.match(r"^(?:and|or)\b", masked_part)
            ):
                return None

        colon = masked_part.find(":")
        if colon >= 0:
            source_parts.append(source_part[:colon].rstrip())
            condition = "\n".join(source_parts).strip()
            if (
                not condition
                or _formula_bracket_stack(condition)
                or (
                    not _rule_text_has_branching_formula(condition)
                    and not _formula_leaf_has_executable_syntax(condition)
                )
            ):
                return None
            inline_body = source_part[colon + 1 :].strip()
            if not _mask_formula_comments(inline_body).strip():
                inline_body = ""
            return base_indent, cursor, condition, inline_body

        source_parts.append(source_part.rstrip())
        masked_parts.append(masked_part.rstrip())
        cursor += 1
    return None


def _first_multiline_match_node(text: str) -> _FormulaBranchNode | None:
    lines = _formula_line_records(text)
    masked_lines = _formula_line_records(_mask_formula_strings(text))
    structural_text = _mask_formula_strings_and_comments(text)
    for index, (start, _content_end, _full_end, line) in enumerate(lines):
        if not _formula_branch_is_at_expression_entry(structural_text, start):
            continue
        masked_line = masked_lines[index][3]
        header = re.match(
            r"^(?P<indent>[ \t]*)match[ \t]+",
            line,
        )
        if header is None:
            continue
        masked_selector = masked_line[header.end() :].rstrip()
        if not masked_selector.endswith(":"):
            continue
        selector_colon = header.end() + len(masked_selector) - 1
        selector = line[header.end() : selector_colon].strip()
        if not selector or _rule_text_has_branching_formula(selector):
            continue
        base_indent = _formula_indent_width(header.group("indent"))
        arm_headers: list[tuple[int, str, str]] = []
        arm_indent: int | None = None
        valid_arms = True
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
            if arrow >= 0 and not _rulespec_match_pattern_has_executable_syntax(
                pattern
            ):
                valid_arms = False
                break
            is_arm = (
                arrow >= 0
                and bool(pattern)
                and _rulespec_match_pattern_has_executable_syntax(pattern)
            )
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
        if not valid_arms or not arm_headers:
            continue
        final_body_start = lines[arm_headers[-1][0]][2]
        chain_end = _formula_expression_end(
            text,
            body_start=final_body_start,
            branch_start=start,
            limit=len(text),
        )
        choices: list[str] = []
        for arm_index, (line_index, _pattern, inline_body) in enumerate(arm_headers):
            body_end = (
                lines[arm_headers[arm_index + 1][0]][0]
                if arm_index + 1 < len(arm_headers)
                else chain_end
            )
            trailing_body = text[lines[line_index][2] : body_end]
            choices.append(
                "\n".join(part for part in (inline_body, trailing_body) if part.strip())
            )
        return _FormulaBranchNode(
            start,
            chain_end,
            "match",
            (selector,),
            tuple(pattern for _line_index, pattern, _body in arm_headers),
            tuple(choices),
        )
    return None


def _first_inline_if_node(text: str) -> _FormulaBranchNode | None:
    masked_text = _mask_formula_strings_and_comments(text)
    tokens = _rulespec_formula_tokens(masked_text)
    if tokens is None:
        return None
    for kind, _source, candidate_start, candidate_end in tokens:
        if kind != "IF":
            continue
        if not _formula_branch_is_at_expression_entry(masked_text, candidate_start):
            continue
        colon = _formula_header_colon(
            masked_text,
            start=candidate_end,
            limit=len(text),
        )
        if colon < 0 or not text[colon + 1 :].strip():
            continue
        if _rule_text_has_branching_formula(text[candidate_end:colon]):
            continue
        chain_headers = _find_inline_chain_headers(
            masked_text,
            body_start=colon + 1,
            limit=len(text),
        )
        if not chain_headers or chain_headers[-1][0] != "else":
            continue
        else_body_start = chain_headers[-1][2]
        end = _formula_expression_end(
            text,
            body_start=else_body_start,
            branch_start=candidate_start,
            limit=len(text),
        )
        conditions = [text[candidate_end:colon].strip()]
        conditions.extend(
            text[condition_start:condition_end].strip()
            for kind, _start, _body_start, condition_start, condition_end in chain_headers
            if kind == "elif"
        )
        body_boundaries = [
            header_start
            for _kind, header_start, _body_start, _condition_start, _condition_end in chain_headers
        ]
        body_starts = [
            colon + 1,
            *(
                header_body_start
                for kind, _header_start, header_body_start, _start, _end in chain_headers
                if kind == "elif"
            ),
        ]
        choices = [
            text[start:stop] for start, stop in zip(body_starts, body_boundaries)
        ]
        choices.append(text[else_body_start:end])
        return _FormulaBranchNode(
            candidate_start,
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
    tokens = _rulespec_formula_tokens(masked_text[body_start:limit])
    if tokens is None:
        return ()
    token_names = {"IF": "if", "ELIF": "elif", "ELSE": "else"}
    for token_kind, _source, relative_start, relative_end in tokens:
        kind = token_names.get(token_kind)
        if kind is None:
            continue
        token_start = body_start + relative_start
        token_end = body_start + relative_end
        if kind == "if":
            colon = _formula_header_colon(
                masked_text,
                start=token_end,
                limit=limit,
            )
            if colon >= 0:
                nested_if_depth += 1
            continue
        colon = _formula_header_colon(
            masked_text,
            start=token_end,
            limit=limit,
        )
        if colon < 0:
            continue
        if kind == "elif":
            if nested_if_depth == 0:
                headers.append(("elif", token_start, colon + 1, token_end, colon))
            continue
        cursor = token_end
        while cursor < limit and masked_text[cursor] in " \t\n\r\x0b\x0c":
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
    masked_text = _mask_formula_strings_and_comments(text)
    tokens = _rulespec_formula_tokens(masked_text)
    if tokens is None:
        return None
    for kind, _source, candidate_start, candidate_end in tokens:
        if kind != "MATCH":
            continue
        if not _formula_branch_is_at_expression_entry(masked_text, candidate_start):
            continue
        colon = _formula_header_colon(
            masked_text,
            start=candidate_end,
            limit=len(text),
        )
        if colon < 0 or not text[colon + 1 :].strip():
            continue
        selector = text[candidate_end:colon].strip()
        if not selector or _rule_text_has_branching_formula(selector):
            continue
        end = _formula_expression_end(
            text,
            body_start=colon + 1,
            branch_start=candidate_start,
            limit=len(text),
            stop_at_semicolon=False,
        )
        arms = _split_inline_formula_match_arms(text[colon + 1 : end])
        if not arms:
            continue
        return _FormulaBranchNode(
            candidate_start,
            end,
            "match",
            (selector,),
            tuple(pattern for pattern, _body in arms),
            tuple(body for _pattern, body in arms),
        )
    return None


def _formula_branch_is_at_expression_entry(masked_text: str, start: int) -> bool:
    prefix = masked_text[:start].rstrip()
    return not prefix or prefix[-1] in "([,"


def _formula_header_colon(masked_text: str, *, start: int, limit: int) -> int:
    stack = list(_formula_bracket_stack(masked_text[:start]))
    initial_depth = len(stack)
    pairs = {")": "(", "]": "[", "}": "{"}
    for index in range(start, limit):
        character = masked_text[index]
        if character in "([{":
            stack.append(character)
        elif character in pairs and stack and stack[-1] == pairs[character]:
            stack.pop()
        elif character == ":" and len(stack) == initial_depth:
            return index
    return -1


def _split_inline_formula_match_arms(text: str) -> tuple[tuple[str, str], ...]:
    tokens = _rulespec_formula_tokens(text)
    if tokens is None:
        return ()
    significant = tokens[:-1]
    depths: list[int] = []
    depth = 0
    for token in significant:
        if token[0] in {"RPAREN", "RBRACKET"}:
            depth -= 1
        if depth < 0:
            return ()
        depths.append(depth)
        if token[0] in {"LPAREN", "LBRACKET"}:
            depth += 1
    if depth != 0:
        return ()
    if any(
        token[0] == "MATCH" and depths[index] == 0
        for index, token in enumerate(significant)
    ):
        return ()
    arrow_indexes = [
        index
        for index, token in enumerate(significant)
        if token[0] == "ARROW" and depths[index] == 0
    ]
    if not arrow_indexes:
        return ()
    pattern_indexes = [index - 1 for index in arrow_indexes]
    if pattern_indexes[0] != 0 or any(index < 0 for index in pattern_indexes):
        return ()
    if not all(
        _rulespec_match_pattern_token_is_executable(significant[index])
        for index in pattern_indexes
    ):
        return ()

    arms: list[tuple[str, str]] = []
    for arm_index, (pattern_index, arrow_index) in enumerate(
        zip(pattern_indexes, arrow_indexes)
    ):
        body_start = significant[arrow_index][3]
        body_end = (
            significant[pattern_indexes[arm_index + 1]][2]
            if arm_index + 1 < len(pattern_indexes)
            else len(text)
        )
        body = text[body_start:body_end].strip()
        if not body:
            return ()
        pattern_token = significant[pattern_index]
        arms.append((text[pattern_token[2] : pattern_token[3]], body))
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
            stop_at_semicolon and character == ";" and len(stack) == len(initial_stack)
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
    constant_environment: dict[str, Any],
) -> tuple[int, tuple[str, ...], tuple[_FormulaTraceStep, ...]] | None:
    if node.kind == "if":
        evaluated: list[str] = []
        selector_trace: list[_FormulaTraceStep] = []
        for index, condition in enumerate(node.selectors):
            evaluated.append(condition)
            reduced_condition = condition
            if _rule_text_has_branching_formula(condition):
                condition_execution = _execute_formula_text(
                    condition,
                    environment=environment,
                    constant_environment=constant_environment,
                )
                if condition_execution is None:
                    return None
                selector_trace.extend(condition_execution.trace)
                reduced_condition = condition_execution.leaf
            value = _evaluate_formula_selector(reduced_condition, environment)
            if not isinstance(value, bool):
                return None
            if value:
                return index, tuple(evaluated), tuple(selector_trace)
        if len(node.choices) > len(node.selectors):
            return len(node.selectors), tuple(evaluated), tuple(selector_trace)
        return None

    selector = _evaluate_formula_selector(node.selectors[0], environment)
    if selector is _UNRESOLVED_CONDITION_VALUE:
        return None
    if not node.patterns:
        return None
    for index, pattern in enumerate(node.patterns[:-1]):
        expected = _match_arm_value(pattern.strip(), environment=environment)
        if expected is _UNRESOLVED_CONDITION_VALUE:
            return None
        if _same_formula_value(selector, expected):
            return index, node.selectors, ()
    return len(node.patterns) - 1, node.selectors, ()


def _evaluate_formula_selector(
    selector: str,
    environment: dict[str, Any],
) -> Any:
    return _evaluate_rulespec_formula(selector, environment=environment)


def _rule_text_has_branching_formula(formula_text: str) -> bool:
    masked_text = _mask_formula_strings_and_comments(formula_text)
    tokens = _rulespec_formula_tokens(masked_text)
    if tokens is not None:
        return any(kind in {"IF", "ELIF", "ELSE", "MATCH"} for kind, *_ in tokens)
    return bool(
        re.search(
            r"(?<![A-Za-z0-9_])(?:if|elif|else|match)(?![A-Za-z0-9_])",
            masked_text,
        )
    )


@functools.lru_cache(maxsize=4096)
def _formula_text_has_executable_branch_tree(formula_text: str) -> bool:
    """Validate every branch subtree without enumerating branch combinations."""

    def validate(candidate: str, *, depth: int) -> bool:
        candidate = textwrap.dedent(candidate).strip()
        if depth > 32:
            return False
        node = _first_formula_branch_node(candidate)
        if node is None:
            return not _rule_text_has_branching_formula(
                candidate
            ) and _formula_leaf_has_executable_syntax(candidate)
        if not all(validate(selector, depth=depth + 1) for selector in node.selectors):
            return False
        if node.kind == "match" and not all(
            _rulespec_match_pattern_has_executable_syntax(pattern)
            for pattern in node.patterns
        ):
            return False
        without_branch = candidate[: node.start] + "0" + candidate[node.end :]
        if not validate(without_branch, depth=depth + 1):
            return False
        return all(validate(choice, depth=depth + 1) for choice in node.choices)

    return validate(formula_text, depth=0)


def _formula_leaf_has_executable_syntax(text: str) -> bool:
    """Parse a reduced leaf with the pinned RuleSpec expression grammar."""

    tokens = _rulespec_formula_tokens(text)
    if tokens is None:
        return False
    position = 0

    def peek() -> str:
        return tokens[position][0]

    def consume(expected: str) -> bool:
        nonlocal position
        if peek() != expected:
            return False
        position += 1
        return True

    def parse_expression() -> bool:
        return parse_or()

    def parse_or() -> bool:
        if not parse_and():
            return False
        while peek() == "OR":
            consume("OR")
            if not parse_and():
                return False
        return True

    def parse_and() -> bool:
        if not parse_comparison():
            return False
        while peek() == "AND":
            consume("AND")
            if not parse_comparison():
                return False
        return True

    def parse_comparison() -> bool:
        if not parse_addition():
            return False
        if peek() in {"LT", "GT", "LE", "GE", "EQ", "NE"}:
            operator = peek()
            consume(operator)
            if not parse_addition():
                return False
        return True

    def parse_addition() -> bool:
        if not parse_multiplication():
            return False
        while peek() in {"PLUS", "MINUS"}:
            operator = peek()
            consume(operator)
            if not parse_multiplication():
                return False
        return True

    def parse_multiplication() -> bool:
        if not parse_unary():
            return False
        while peek() in {"STAR", "SLASH"}:
            operator = peek()
            consume(operator)
            if not parse_unary():
                return False
        return True

    def parse_unary() -> bool:
        if peek() in {"MINUS", "NOT"}:
            operator = peek()
            consume(operator)
            return parse_unary()
        return parse_postfix()

    def parse_postfix() -> bool:
        primary_start = position
        if not parse_primary():
            return False
        primary_end = position
        while (
            primary_end - primary_start >= 3
            and tokens[primary_start][0] == "LPAREN"
            and tokens[primary_end - 1][0] == "RPAREN"
        ):
            primary_start += 1
            primary_end -= 1
        callable_variable = primary_end - primary_start == 1 and tokens[primary_start][
            0
        ] in {"IDENT", "PATH"}
        while True:
            if peek() == "LPAREN":
                if not callable_variable:
                    return False
                consume("LPAREN")
                if peek() != "RPAREN":
                    if not parse_expression():
                        return False
                    while peek() == "COMMA":
                        consume("COMMA")
                        if not parse_expression():
                            return False
                if not consume("RPAREN"):
                    return False
                callable_variable = False
                continue
            if peek() == "DOT":
                consume("DOT")
                if not consume("IDENT"):
                    return False
                callable_variable = False
                continue
            if peek() == "LBRACKET":
                consume("LBRACKET")
                if not parse_expression() or not consume("RBRACKET"):
                    return False
                callable_variable = False
                continue
            break
        return True

    def parse_primary() -> bool:
        nonlocal position
        if peek() in {"INT", "FLOAT"}:
            if not _rulespec_numeric_token_is_executable(tokens[position]):
                return False
            position += 1
            return True
        if peek() == "STRING":
            if not _rulespec_string_token_is_executable(tokens[position][1]):
                return False
            position += 1
            return True
        if peek() in {"TRUE", "FALSE", "IDENT", "PATH"}:
            position += 1
            return True
        if consume("LPAREN"):
            if not parse_expression() or not consume("RPAREN"):
                return False
            return True
        return False

    try:
        return parse_expression() and peek() == "EOF"
    except RecursionError:
        return False


def _evaluate_rulespec_formula(
    text: str,
    *,
    environment: dict[str, Any],
) -> Any:
    """Evaluate the pinned expression subset without Python lexer semantics."""

    if not _formula_leaf_has_executable_syntax(text):
        return _UNRESOLVED_CONDITION_VALUE
    tokens = _rulespec_formula_tokens(text)
    if tokens is None:
        return _UNRESOLVED_CONDITION_VALUE
    position = 0
    reference_marker = object()

    def peek() -> str:
        return tokens[position][0]

    def consume(expected: str) -> tuple[str, str, int, int] | None:
        nonlocal position
        if peek() != expected:
            return None
        token = tokens[position]
        position += 1
        return token

    def reference(name: str) -> tuple[object, str]:
        return reference_marker, name

    def is_reference(value: Any) -> bool:
        return (
            isinstance(value, tuple)
            and len(value) == 2
            and value[0] is reference_marker
        )

    def resolve(value: Any) -> Any:
        if is_reference(value):
            return environment.get(value[1], _UNRESOLVED_CONDITION_VALUE)
        return value

    def parse_expression() -> Any:
        return parse_or()

    def parse_or() -> Any:
        value = parse_and()
        while peek() == "OR":
            consume("OR")
            right = parse_and()
            left_value = resolve(value)
            right_value = resolve(right)
            if left_value is _UNRESOLVED_CONDITION_VALUE:
                value = left_value
            elif not isinstance(left_value, bool):
                value = _UNRESOLVED_CONDITION_VALUE
            elif left_value:
                value = True
            elif right_value is _UNRESOLVED_CONDITION_VALUE:
                value = right_value
            elif not isinstance(right_value, bool):
                value = _UNRESOLVED_CONDITION_VALUE
            else:
                value = right_value
        return value

    def parse_and() -> Any:
        value = parse_comparison()
        while peek() == "AND":
            consume("AND")
            right = parse_comparison()
            left_value = resolve(value)
            right_value = resolve(right)
            if left_value is _UNRESOLVED_CONDITION_VALUE:
                value = left_value
            elif not isinstance(left_value, bool):
                value = _UNRESOLVED_CONDITION_VALUE
            elif not left_value:
                value = False
            elif right_value is _UNRESOLVED_CONDITION_VALUE:
                value = right_value
            elif not isinstance(right_value, bool):
                value = _UNRESOLVED_CONDITION_VALUE
            else:
                value = right_value
        return value

    def parse_comparison() -> Any:
        value = parse_addition()
        if peek() not in {"LT", "GT", "LE", "GE", "EQ", "NE"}:
            return value
        operator = peek()
        consume(operator)
        right = parse_addition()
        left_value = resolve(value)
        right_value = resolve(right)
        if (
            left_value is _UNRESOLVED_CONDITION_VALUE
            or right_value is _UNRESOLVED_CONDITION_VALUE
        ):
            return _UNRESOLVED_CONDITION_VALUE
        left_number = _rulespec_runtime_decimal(left_value)
        right_number = _rulespec_runtime_decimal(right_value)
        if left_number is not None or right_number is not None:
            if left_number is None or right_number is None:
                return _UNRESOLVED_CONDITION_VALUE
            left_value = left_number
            right_value = right_number
        else:
            left_date = _rulespec_runtime_date(left_value)
            right_date = _rulespec_runtime_date(right_value)
            if left_date is not None or right_date is not None:
                if left_date is None or right_date is None:
                    return _UNRESOLVED_CONDITION_VALUE
                left_value = left_date
                right_value = right_date
            elif not (
                type(left_value) is type(right_value)
                and isinstance(left_value, (bool, str))
                and operator in {"EQ", "NE"}
            ):
                return _UNRESOLVED_CONDITION_VALUE
        try:
            if operator == "EQ":
                return left_value == right_value
            if operator == "NE":
                return left_value != right_value
            if operator == "LT":
                return left_value < right_value
            if operator == "GT":
                return left_value > right_value
            if operator == "LE":
                return left_value <= right_value
            return left_value >= right_value
        except (TypeError, ValueError):
            return _UNRESOLVED_CONDITION_VALUE

    def parse_addition() -> Any:
        value = parse_multiplication()
        while peek() in {"PLUS", "MINUS"}:
            operator = peek()
            consume(operator)
            right = parse_multiplication()
            left_value = resolve(value)
            right_value = resolve(right)
            if (
                left_value is _UNRESOLVED_CONDITION_VALUE
                or right_value is _UNRESOLVED_CONDITION_VALUE
            ):
                value = _UNRESOLVED_CONDITION_VALUE
                continue
            left_value = _rulespec_runtime_decimal(left_value)
            right_value = _rulespec_runtime_decimal(right_value)
            if left_value is None or right_value is None:
                value = _UNRESOLVED_CONDITION_VALUE
                continue
            try:
                value = _rulespec_decimal_binary_operation(
                    left_value,
                    right_value,
                    operator,
                )
            except (ArithmeticError, TypeError, ValueError):
                value = _UNRESOLVED_CONDITION_VALUE
        return value

    def parse_multiplication() -> Any:
        value = parse_unary()
        while peek() in {"STAR", "SLASH"}:
            operator = peek()
            consume(operator)
            right = parse_unary()
            left_value = resolve(value)
            right_value = resolve(right)
            if (
                left_value is _UNRESOLVED_CONDITION_VALUE
                or right_value is _UNRESOLVED_CONDITION_VALUE
            ):
                value = _UNRESOLVED_CONDITION_VALUE
                continue
            left_value = _rulespec_runtime_decimal(left_value)
            right_value = _rulespec_runtime_decimal(right_value)
            if left_value is None or right_value is None:
                value = _UNRESOLVED_CONDITION_VALUE
                continue
            try:
                value = _rulespec_decimal_binary_operation(
                    left_value,
                    right_value,
                    operator,
                )
            except (ArithmeticError, TypeError, ValueError):
                value = _UNRESOLVED_CONDITION_VALUE
        return value

    def parse_unary() -> Any:
        if peek() == "MINUS":
            consume("MINUS")
            value = resolve(parse_unary())
            number = _rulespec_runtime_decimal(value)
            if number is not None:
                return -number
            return _UNRESOLVED_CONDITION_VALUE
        if peek() == "NOT":
            consume("NOT")
            value = resolve(parse_unary())
            if value is _UNRESOLVED_CONDITION_VALUE:
                return value
            return not value if isinstance(value, bool) else _UNRESOLVED_CONDITION_VALUE
        return parse_postfix()

    def parse_postfix() -> Any:
        value = parse_primary()
        while True:
            if peek() == "LPAREN":
                consume("LPAREN")
                arguments: list[Any] = []
                if peek() != "RPAREN":
                    arguments.append(resolve(parse_expression()))
                    while peek() == "COMMA":
                        consume("COMMA")
                        arguments.append(resolve(parse_expression()))
                consume("RPAREN")
                function_name = value[1] if is_reference(value) else ""
                value = evaluate_call(function_name, arguments)
                continue
            if peek() == "DOT":
                consume("DOT")
                consume("IDENT")
                value = _UNRESOLVED_CONDITION_VALUE
                continue
            if peek() == "LBRACKET":
                consume("LBRACKET")
                index = resolve(parse_expression())
                consume("RBRACKET")
                base = resolve(value)
                if (
                    base is _UNRESOLVED_CONDITION_VALUE
                    or index is _UNRESOLVED_CONDITION_VALUE
                ):
                    value = _UNRESOLVED_CONDITION_VALUE
                    continue
                try:
                    value = base[index]
                except (IndexError, KeyError, TypeError):
                    value = _UNRESOLVED_CONDITION_VALUE
                continue
            break
        return value

    def parse_primary() -> Any:
        kind = peek()
        if kind == "INT":
            token = consume("INT")
            if token is None:
                return 0
            digits = token[1].replace("_", "").lstrip("0") or "0"
            return int(digits)
        if kind == "FLOAT":
            token = consume("FLOAT")
            return (
                _rulespec_decimal_token_value(token[1])
                if token is not None
                else Decimal(0)
            )
        if kind == "STRING":
            token = consume("STRING")
            return _rulespec_string_token_value(token[1]) if token is not None else ""
        if kind == "TRUE":
            consume("TRUE")
            return True
        if kind == "FALSE":
            consume("FALSE")
            return False
        if kind in {"IDENT", "PATH"}:
            token = consume(kind)
            return reference(token[1]) if token is not None else reference("")
        consume("LPAREN")
        value = parse_expression()
        consume("RPAREN")
        return value

    def evaluate_call(function_name: str, arguments: list[Any]) -> Any:
        if any(value is _UNRESOLVED_CONDITION_VALUE for value in arguments):
            return _UNRESOLVED_CONDITION_VALUE
        numbers = [_rulespec_runtime_decimal(value) for value in arguments]
        try:
            if (
                function_name == "min"
                and numbers
                and all(number is not None for number in numbers)
            ):
                return min(numbers)
            if (
                function_name == "max"
                and numbers
                and all(number is not None for number in numbers)
            ):
                return max(numbers)
            if (
                function_name == "floor"
                and len(numbers) == 1
                and numbers[0] is not None
            ):
                return math.floor(numbers[0])
            if function_name == "ceil" and len(numbers) == 1 and numbers[0] is not None:
                return math.ceil(numbers[0])
        except (ArithmeticError, TypeError, ValueError):
            pass
        return _UNRESOLVED_CONDITION_VALUE

    return resolve(parse_expression())


def _rulespec_runtime_decimal(value: Any) -> Decimal | None:
    """Apply pinned scalar numeric coercion without conflating booleans."""

    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        candidate = value
    elif isinstance(value, int):
        candidate = Decimal(value)
    elif isinstance(value, float) and math.isfinite(value):
        candidate = Decimal(str(value))
    else:
        return None
    if not candidate.is_finite():
        return None
    components = candidate.as_tuple()
    exponent = components.exponent
    if not isinstance(exponent, int):
        return None
    coefficient = components.digits
    if not any(coefficient):
        return Decimal(0)
    trailing_zeros = 0
    if exponent < 0:
        trailing_zeros = next(
            (index for index, digit in enumerate(reversed(coefficient)) if digit != 0),
            len(coefficient),
        )
        if trailing_zeros:
            coefficient = coefficient[:-trailing_zeros]
            exponent += trailing_zeros
            candidate = Decimal((components.sign, coefficient, exponent))
    if exponent < -28:
        return None
    maximum = Decimal("79228162514264337593543950335")
    if candidate.copy_abs() > maximum:
        return None
    maximum_coefficient = maximum.as_tuple().digits
    if exponent < 0:
        if len(coefficient) > len(maximum_coefficient) or (
            len(coefficient) == len(maximum_coefficient)
            and coefficient > maximum_coefficient
        ):
            return None
    return candidate


def _rulespec_runtime_date(value: Any) -> date | None:
    """Preserve typed RuleSpec dates without retyping ordinary text."""

    if type(value) is date:
        return value
    return None


def _rulespec_decimal_binary_operation(
    left: Decimal,
    right: Decimal,
    operator: str,
) -> Decimal:
    """Evaluate one rust_decimal-style operation without context truncation."""

    with localcontext() as context:
        context.prec = 64
        if operator == "PLUS":
            value = left + right
        elif operator == "MINUS":
            value = left - right
        elif operator == "STAR":
            value = left * right
        else:
            value = left / right
        return _rulespec_decimal_result_value(value)


def _rulespec_decimal_result_value(value: Decimal) -> Decimal:
    """Fit an arithmetic result into the pinned 96-bit Decimal envelope."""

    maximum = Decimal("79228162514264337593543950335")
    if not value.is_finite():
        raise InvalidOperation
    with localcontext() as context:
        context.prec = 64
        for scale in range(28, -1, -1):
            quantum = Decimal(1).scaleb(-scale)
            rounded = value.quantize(quantum, rounding=ROUND_HALF_EVEN)
            if abs(rounded.scaleb(scale)) <= maximum:
                return rounded
    raise InvalidOperation


def _rulespec_decimal_token_value(source: str) -> Decimal:
    """Round a valid literal as pinned rust_decimal ``FromStr`` does."""

    normalized = source.replace("_", "")
    precision = max(64, len(normalized) + 2)
    maximum = Decimal("79228162514264337593543950335")
    with localcontext() as context:
        context.prec = precision
        value = Decimal(normalized)
        for scale in range(min(28, len(normalized.partition(".")[2])), -1, -1):
            quantum = Decimal(1).scaleb(-scale)
            rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
            coefficient = rounded.scaleb(scale)
            if coefficient <= maximum:
                return rounded
    return value


def _rulespec_string_token_value(source: str) -> str:
    """Decode exactly the escape set recognized by the pinned lexer."""

    escapes = {
        "\\": "\\",
        '"': '"',
        "'": "'",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    value: list[str] = []
    cursor = 1
    while cursor < len(source) - 1:
        character = source[cursor]
        if character == "\\" and cursor + 1 < len(source) - 1:
            cursor += 1
            escaped = source[cursor]
            character = escapes.get(escaped, escaped)
        value.append(character)
        cursor += 1
    return "".join(value)


def _rulespec_string_token_is_executable(source: str) -> bool:
    """Fail closed on the pinned lexer's escaped non-ASCII panic edge."""

    return not any(
        source[index] == "\\" and not source[index + 1].isascii()
        for index in range(1, len(source) - 2)
    )


def _rulespec_formula_tokens(
    text: str,
) -> tuple[tuple[str, str, int, int], ...] | None:
    """Tokenize the pinned RuleSpec formula language while preserving offsets."""

    keywords = {
        "entity": "ENTITY",
        "amend": "AMEND",
        "from": "FROM",
        "to": "TO",
        "match": "MATCH",
        "if": "IF",
        "elif": "ELIF",
        "else": "ELSE",
        "and": "AND",
        "or": "OR",
        "not": "NOT",
        "true": "TRUE",
        "True": "TRUE",
        "false": "FALSE",
        "False": "FALSE",
    }
    multi = {
        "=>": "ARROW",
        "<=": "LE",
        ">=": "GE",
        "==": "EQ",
        "!=": "NE",
        "->": "FK",
    }
    single = {
        ":": "COLON",
        ",": "COMMA",
        ".": "DOT",
        "=": "ASSIGN",
        "+": "PLUS",
        "-": "MINUS",
        "*": "STAR",
        "/": "SLASH",
        "<": "LT",
        ">": "GT",
        "(": "LPAREN",
        ")": "RPAREN",
        "[": "LBRACKET",
        "]": "RBRACKET",
    }
    tokens: list[tuple[str, str, int, int]] = []
    cursor = 0
    while cursor < len(text):
        if text.startswith('"""', cursor):
            end = text.find('"""', cursor + 3)
            cursor = len(text) if end < 0 else end + 3
            continue
        character = text[cursor]
        if character == "#":
            newline = text.find("\n", cursor)
            cursor = len(text) if newline < 0 else newline
            continue
        if character in " \t\n\r\x0b\x0c":
            cursor += 1
            continue
        if (
            character.isascii()
            and character.isdigit()
            and cursor + 10 <= len(text)
            and re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
                text[cursor : cursor + 10],
            )
        ):
            tokens.append(("DATE", text[cursor : cursor + 10], cursor, cursor + 10))
            cursor += 10
            continue
        if character.isascii() and character.isdigit():
            end = cursor
            while end < len(text) and (
                (text[end].isascii() and text[end].isdigit()) or text[end] == "_"
            ):
                end += 1
            kind = "INT"
            if end + 1 < len(text) and text[end] == "." and text[end + 1].isdigit():
                kind = "FLOAT"
                end += 1
                while end < len(text) and (
                    (text[end].isascii() and text[end].isdigit()) or text[end] == "_"
                ):
                    end += 1
            tokens.append((kind, text[cursor:end], cursor, end))
            cursor = end
            continue
        if character in {'"', "'"}:
            quote = character
            end = cursor + 1
            escaped = False
            while end < len(text):
                candidate = text[end]
                if escaped:
                    escaped = False
                elif candidate == "\\":
                    escaped = True
                elif candidate == quote:
                    end += 1
                    break
                end += 1
            else:
                return None
            tokens.append(("STRING", text[cursor:end], cursor, end))
            cursor = end
            continue
        if character.isascii() and (character.isalpha() or character == "_"):
            end = cursor + 1
            while (
                end < len(text)
                and text[end].isascii()
                and (text[end].isalnum() or text[end] == "_")
            ):
                end += 1
            path_end = end
            is_path = False
            while path_end < len(text) and text[path_end] == "/":
                part_start = path_end + 1
                if (
                    part_start >= len(text)
                    or not text[part_start].isascii()
                    or not (text[part_start].isalpha() or text[part_start] == "_")
                ):
                    break
                is_path = True
                path_end = part_start + 1
                while (
                    path_end < len(text)
                    and text[path_end].isascii()
                    and (text[path_end].isalnum() or text[path_end] == "_")
                ):
                    path_end += 1
            end = path_end if is_path else end
            value = text[cursor:end]
            tokens.append(
                (
                    "PATH" if is_path else keywords.get(value, "IDENT"),
                    value,
                    cursor,
                    end,
                )
            )
            cursor = end
            continue
        pair = text[cursor : cursor + 2]
        if pair in multi:
            tokens.append((multi[pair], pair, cursor, cursor + 2))
            cursor += 2
            continue
        if character not in single:
            return None
        tokens.append((single[character], character, cursor, cursor + 1))
        cursor += 1
    tokens.append(("EOF", "", len(text), len(text)))
    return tuple(tokens)


def _rulespec_match_pattern_has_executable_syntax(text: str) -> bool:
    tokens = _rulespec_formula_tokens(text)
    return (
        tokens is not None
        and len(tokens) == 2
        and _rulespec_match_pattern_token_is_executable(tokens[0])
    )


def _rulespec_match_pattern_token_is_executable(
    token: tuple[str, str, int, int],
) -> bool:
    if token[0] in {"INT", "FLOAT"}:
        return _rulespec_numeric_token_is_executable(token)
    if token[0] == "STRING":
        return _rulespec_string_token_is_executable(token[1])
    return token[0] in {"TRUE", "FALSE", "IDENT"}


def _rulespec_numeric_token_is_executable(
    token: tuple[str, str, int, int],
) -> bool:
    """Check pinned i64/rust_decimal bounds without unbounded int parsing."""

    kind, source, _start, _end = token
    value = source.replace("_", "")
    if kind == "INT":
        digits = value.lstrip("0") or "0"
        maximum = str(2**63 - 1)
        return len(digits) < len(maximum) or (
            len(digits) == len(maximum) and digits <= maximum
        )
    if kind != "FLOAT":
        return False
    whole, separator, fraction = value.partition(".")
    if not separator:
        return False
    maximum = "79228162514264337593543950335"
    whole = whole.lstrip("0") or "0"
    if len(whole) != len(maximum):
        return len(whole) < len(maximum)
    if whole != maximum:
        return whole < maximum
    return not fraction or fraction[0] < "5"


def _formula_execution_outcome(execution: _FormulaExecution) -> str:
    return "/".join(f"{step.kind}:{step.choice}" for step in execution.trace)


def _formula_execution_leaf_is_computational(
    execution: _FormulaExecution,
) -> bool:
    if execution.evaluates_to_zero:
        return False
    leaf = execution.leaf.strip()
    with contextlib.suppress(SyntaxError, ValueError):
        literal = ast.literal_eval(leaf)
        if literal is False or (
            isinstance(literal, (int, float, complex))
            and not isinstance(literal, bool)
            and literal == 0
        ):
            return False
    return leaf not in {"true", "false", "True", "False", "null", "none"}


def _formula_execution_references_names(
    execution: _FormulaExecution,
    names: set[str],
    *,
    selectors_only: bool = False,
) -> bool:
    texts = [selector for step in execution.trace for selector in step.selectors]
    if not selectors_only:
        texts.append(execution.leaf)
    return any(set(_FORMULA_IDENTIFIER.findall(text)) & names for text in texts)


def _same_formula_value(left: Any, right: Any) -> bool:
    left_number = _rulespec_runtime_decimal(left)
    right_number = _rulespec_runtime_decimal(right)
    if left_number is not None or right_number is not None:
        return (
            left_number is not None
            and right_number is not None
            and left_number == right_number
        )
    return type(left) is type(right) and left == right


def _mask_formula_strings(text: str) -> str:
    """Blank quoted content while preserving offsets and newlines."""

    masked = list(text)
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(text):
        if quote is None and text.startswith('"""', index):
            end = text.find('"""', index + 3)
            stop = len(text) if end < 0 else end + 3
            for comment_index in range(index, stop):
                if text[comment_index] != "\n":
                    masked[comment_index] = " "
            index = stop
            continue
        character = text[index]
        if quote is None:
            if character in {'"', "'"}:
                quote = character
                masked[index] = " "
            index += 1
            continue
        if character != "\n":
            masked[index] = " "
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == quote:
            quote = None
        index += 1
    return "".join(masked)


def _mask_formula_strings_and_comments(text: str) -> str:
    """Blank quoted content and comments while preserving source offsets."""

    masked = list(_mask_formula_strings(text))
    in_comment = False
    for index, character in enumerate(masked):
        if character == "\n":
            in_comment = False
        elif in_comment:
            masked[index] = " "
        elif character == "#":
            in_comment = True
            masked[index] = " "
    return "".join(masked)


def _mask_formula_comments(text: str) -> str:
    """Blank comments but preserve quoted content, offsets, and newlines."""

    masked = list(text)
    quote: str | None = None
    escaped = False
    in_comment = False
    index = 0
    while index < len(text):
        if quote is None and not in_comment and text.startswith('"""', index):
            end = text.find('"""', index + 3)
            stop = len(text) if end < 0 else end + 3
            for comment_index in range(index, stop):
                if text[comment_index] != "\n":
                    masked[comment_index] = " "
            index = stop
            continue
        character = text[index]
        if character == "\n":
            in_comment = False
            escaped = False
            index += 1
            continue
        if in_comment:
            masked[index] = " "
            index += 1
            continue
        if quote is None:
            if character in {'"', "'"}:
                quote = character
            elif character == "#":
                in_comment = True
                masked[index] = " "
            index += 1
            continue
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == quote:
            quote = None
        index += 1
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


def _match_arm_value(
    value: str,
    *,
    environment: dict[str, Any],
) -> Any:
    tokens = _rulespec_formula_tokens(value)
    if (
        tokens is None
        or len(tokens) != 2
        or not _rulespec_match_pattern_token_is_executable(tokens[0])
    ):
        return _UNRESOLVED_CONDITION_VALUE
    kind, source, _start, _end = tokens[0]
    if kind == "TRUE":
        return True
    if kind == "FALSE":
        return False
    if kind == "INT":
        digits = source.replace("_", "").lstrip("0") or "0"
        return int(digits)
    if kind == "FLOAT":
        return _rulespec_decimal_token_value(source)
    if kind == "STRING":
        return _rulespec_string_token_value(source)
    if kind == "IDENT":
        return environment.get(source, _UNRESOLVED_CONDITION_VALUE)
    return _UNRESOLVED_CONDITION_VALUE


def _formula_expression_is_definitely_zero(
    text: str,
    *,
    environment: dict[str, Any],
) -> bool:
    with contextlib.suppress(SyntaxError):
        expression = ast.parse(text.strip(), mode="eval").body
        simplified = _simplify_formula_expression(
            expression,
            environment=environment,
        )
        value = _evaluate_condition_expression(simplified, environment)
        number = _rulespec_runtime_decimal(value)
        return number is not None and number == 0
    return False


def _evaluate_condition_expression(
    expression: ast.expr,
    environment: dict[str, Any],
) -> Any:
    """Evaluate a small side-effect-free subset of RuleSpec conditions."""

    if isinstance(expression, ast.Constant):
        return expression.value
    if isinstance(expression, ast.Name):
        if expression.id == "true":
            return True
        if expression.id == "false":
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
        if left is _UNRESOLVED_CONDITION_VALUE or right is _UNRESOLVED_CONDITION_VALUE:
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
        and not expression.keywords
    ):
        function_name = expression.func.id
        values = [
            _evaluate_condition_expression(argument, environment)
            for argument in expression.args
        ]
        if any(value is _UNRESOLVED_CONDITION_VALUE for value in values):
            return _UNRESOLVED_CONDITION_VALUE
        with contextlib.suppress(ArithmeticError, TypeError, ValueError):
            if function_name in {"holds", "not_holds"} and len(values) == 1:
                return (
                    bool(values[0]) if function_name == "holds" else not bool(values[0])
                )
            if function_name == "min" and values:
                return min(values)
            if function_name == "max" and values:
                return max(values)
            if function_name == "floor" and len(values) == 1:
                return math.floor(values[0])
            if function_name == "ceil" and len(values) == 1:
                return math.ceil(values[0])
            if function_name == "abs" and len(values) == 1:
                return abs(values[0])
    return _UNRESOLVED_CONDITION_VALUE


def _branch_boundary_test_witnesses(
    branch: SourceStructureBranch,
    boundary: NumericOccurrenceLike,
    *,
    principal_rules: dict[str, dict[str, Any]],
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    numeric_value_is_grounded: NumericGroundingPredicate,
    formula_environment: dict[str, Any] | None = None,
    source_bound_constant_occurrences: dict[str, tuple[NumericOccurrenceLike, ...]]
    | None = None,
    extract_numeric_occurrences: NumericOccurrenceExtractor | None = None,
) -> set[tuple[str, str]]:
    witnesses: set[tuple[str, str]] = set()
    source_interval, source_boolean_polarity = (
        _source_interval_and_polarity_for_boundary(
            branch,
            boundary,
            extract_numeric_occurrences=extract_numeric_occurrences,
        )
        if extract_numeric_occurrences is not None
        else (None, 0)
    )
    for rule_name in _rules_covering_branch(branch, principal_rule_paths):
        rule = principal_rules[rule_name]
        for case in asserted_by_rule.get(rule_name, ()):
            dependency_environment = _case_asserted_dependency_environment(
                principal_rules,
                case,
                formula_environment=formula_environment or {},
            )
            execution = _case_formula_execution(
                rule,
                case,
                formula_environment=formula_environment,
                dependency_environment=dependency_environment,
            )
            if execution is None:
                continue
            for (
                controller_rule,
                controller_execution,
            ) in _asserted_reached_rule_executions(
                rule,
                execution,
                principal_rules=principal_rules,
                case=case,
                dependency_environment=dependency_environment,
                formula_environment=formula_environment or {},
            ):
                selector_names = _rule_numeric_selector_names(controller_rule)
                if selector_names and any(
                    numeric_value_is_grounded(value, (boundary,))
                    and _formula_execution_references_names(
                        controller_execution,
                        input_names,
                    )
                    and _formula_execution_binds_boundary(
                        controller_execution,
                        boundary,
                        input_names=input_names,
                        formula_environment=(controller_execution.constant_environment),
                        source_bound_constant_occurrences=(
                            source_bound_constant_occurrences or {}
                        ),
                        source_interval=source_interval,
                        source_boolean_polarity=source_boolean_polarity,
                        extract_numeric_occurrences=(extract_numeric_occurrences),
                        numeric_value_is_grounded=(numeric_value_is_grounded),
                    )
                    and _boundary_case_changes_formula_effect(
                        controller_rule,
                        case,
                        input_key=input_key,
                        selector_names=input_names,
                        boundary_value=value,
                        execution=controller_execution,
                        principal_rules=principal_rules,
                        dependency_names=set(dependency_environment),
                        formula_environment=formula_environment or {},
                        source_interval=source_interval,
                        source_boolean_polarity=source_boolean_polarity,
                        root_rule=rule,
                        root_execution=execution,
                    )
                    for input_key, input_names, value in (
                        _case_numeric_selector_evidence(
                            case,
                            selector_names,
                            dependency_environment=dependency_environment,
                        )
                    )
                ):
                    witnesses.add((rule_name, f"case:{id(case)}"))
                    break
    return witnesses


def _asserted_reached_rule_executions(
    root_rule: dict[str, Any],
    root_execution: _FormulaExecution,
    *,
    principal_rules: dict[str, dict[str, Any]],
    case: dict[str, Any],
    dependency_environment: dict[str, Any],
    formula_environment: dict[str, Any],
) -> tuple[tuple[dict[str, Any], _FormulaExecution], ...]:
    """Return the root and asserted local dependencies its reached path uses."""

    reached: list[tuple[dict[str, Any], _FormulaExecution]] = [
        (root_rule, root_execution)
    ]
    pending = [root_execution]
    seen_names: set[str] = set()
    while pending:
        parent_execution = pending.pop()
        for name in sorted(set(dependency_environment) - seen_names):
            if not _formula_execution_references_names(parent_execution, {name}):
                continue
            dependency_rule = principal_rules.get(name)
            if dependency_rule is None:
                continue
            dependency_execution = _case_formula_execution(
                dependency_rule,
                case,
                formula_environment=formula_environment,
                dependency_environment=dependency_environment,
            )
            if dependency_execution is None or not _formula_runtime_values_equal(
                _formula_execution_runtime_value(dependency_execution),
                dependency_environment[name],
            ):
                continue
            seen_names.add(name)
            reached.append((dependency_rule, dependency_execution))
            pending.append(dependency_execution)
    return tuple(reached)


def _branch_boundary_has_test_evidence(
    branch: SourceStructureBranch,
    boundary: NumericOccurrenceLike,
    *,
    principal_rules: dict[str, dict[str, Any]],
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    numeric_value_is_grounded: NumericGroundingPredicate,
    formula_environment: dict[str, Any] | None = None,
    source_bound_constant_occurrences: dict[str, tuple[NumericOccurrenceLike, ...]]
    | None = None,
    extract_numeric_occurrences: NumericOccurrenceExtractor | None = None,
) -> bool:
    """Compatibility predicate for callers that do not allocate witnesses."""

    return bool(
        _branch_boundary_test_witnesses(
            branch,
            boundary,
            principal_rules=principal_rules,
            principal_rule_paths=principal_rule_paths,
            asserted_by_rule=asserted_by_rule,
            numeric_value_is_grounded=numeric_value_is_grounded,
            formula_environment=formula_environment,
            source_bound_constant_occurrences=source_bound_constant_occurrences,
            extract_numeric_occurrences=extract_numeric_occurrences,
        )
    )


def _formula_execution_binds_boundary(
    execution: _FormulaExecution,
    boundary: NumericOccurrenceLike,
    *,
    input_names: set[str],
    formula_environment: dict[str, Any],
    source_bound_constant_occurrences: dict[str, tuple[NumericOccurrenceLike, ...]],
    source_interval: _NumericInterval | None,
    source_boolean_polarity: int,
    extract_numeric_occurrences: NumericOccurrenceExtractor | None,
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> bool:
    boundary_names = {
        name
        for name, value in formula_environment.items()
        if _rulespec_runtime_decimal(value) is not None
        and (
            numeric_value_is_grounded(float(value), (boundary,))
            or _is_adjacent_integral_boundary(float(value), boundary)
        )
    }
    if input_names & boundary_names:
        return False
    source_bound_boundary_names = {
        name
        for name in boundary_names
        if any(
            numeric_value_is_grounded(float(boundary.value), (occurrence,))
            for occurrence in source_bound_constant_occurrences.get(name, ())
        )
    }
    checks: list[tuple[str, bool]] = []
    for step in execution.trace:
        if step.kind not in {"if", "match"}:
            continue
        checks.extend((selector, True) for selector in step.selectors)
    checks.append((execution.leaf, source_boolean_polarity < 0))
    return any(
        _formula_text_has_boundary_comparison(
            text,
            allow_complement_relation=allow_complement_relation,
            input_names=input_names,
            boundary_names=boundary_names,
            boundary=boundary,
            formula_environment=formula_environment,
            source_bound_boundary_names=source_bound_boundary_names,
            source_bound_constant_occurrences=source_bound_constant_occurrences,
            source_interval=source_interval,
            extract_numeric_occurrences=extract_numeric_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
        )
        for text, allow_complement_relation in checks
    )


def _formula_text_has_boundary_comparison(
    text: str,
    *,
    allow_complement_relation: bool,
    input_names: set[str],
    boundary_names: set[str],
    boundary: NumericOccurrenceLike,
    formula_environment: dict[str, Any],
    source_interval: _NumericInterval | None,
    extract_numeric_occurrences: NumericOccurrenceExtractor | None,
    numeric_value_is_grounded: NumericGroundingPredicate,
    source_bound_boundary_names: set[str] | None = None,
    source_bound_constant_occurrences: (
        dict[str, tuple[NumericOccurrenceLike, ...]] | None
    ) = None,
) -> bool:
    expression = _parse_formula_expression(text)
    if expression is not None:
        for comparison, negated in _formula_comparisons_with_polarity(expression):
            operands = (comparison.left, *comparison.comparators)
            for operator, left, right in zip(
                comparison.ops,
                operands,
                operands[1:],
            ):
                left_is_input = _formula_node_references_names(
                    left,
                    input_names,
                )
                right_is_input = _formula_node_references_names(
                    right,
                    input_names,
                )
                boundary_node = right if left_is_input and not right_is_input else left
                if not (
                    (left_is_input and not right_is_input)
                    or (right_is_input and not left_is_input)
                ):
                    continue
                boundary_value = _formula_node_boundary_value(
                    boundary_node,
                    boundary_names=boundary_names,
                    boundary=boundary,
                    formula_environment=formula_environment,
                    extract_numeric_occurrences=extract_numeric_occurrences,
                    numeric_value_is_grounded=numeric_value_is_grounded,
                )
                if boundary_value is not None and (
                    source_interval is None
                    or _comparison_matches_source_interval(
                        operator,
                        input_on_left=left_is_input,
                        compared_boundary_value=boundary_value,
                        boundary_names=boundary_names,
                        boundary=boundary,
                        numeric_value_is_grounded=numeric_value_is_grounded,
                        source_interval=source_interval,
                        allow_complement_relation=allow_complement_relation,
                        negated=negated,
                    )
                ):
                    return True
        if source_interval is not None and _formula_expression_has_boundary_clamp(
            expression,
            input_names=input_names,
            boundary_names=boundary_names,
            boundary=boundary,
            formula_environment=formula_environment,
            source_bound_boundary_names=source_bound_boundary_names or set(),
            source_bound_constant_occurrences=(source_bound_constant_occurrences or {}),
            source_interval=source_interval,
            extract_numeric_occurrences=extract_numeric_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
        ):
            return True
    return False


def _formula_expression_has_boundary_clamp(
    expression: ast.expr,
    *,
    input_names: set[str],
    boundary_names: set[str],
    boundary: NumericOccurrenceLike,
    formula_environment: dict[str, Any],
    source_bound_boundary_names: set[str],
    source_bound_constant_occurrences: dict[str, tuple[NumericOccurrenceLike, ...]],
    source_interval: _NumericInterval,
    extract_numeric_occurrences: NumericOccurrenceExtractor | None,
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> bool:
    """Recognize exact upper and lower clamps at a source-stated boundary."""

    boundary_is_strict_lower = bool(
        source_interval.lower is not None
        and not source_interval.lower_inclusive
        and _numeric_occurrences_are_equivalent(source_interval.lower, boundary)
    )

    for node in ast.walk(expression):
        if boundary_is_strict_lower and _formula_lower_boundary_clamp_matches(
            node,
            expression=expression,
            input_names=input_names,
            source_bound_boundary_names=source_bound_boundary_names,
            source_bound_constant_occurrences=source_bound_constant_occurrences,
            source_interval=source_interval,
            boundary=boundary,
            formula_environment=formula_environment,
            extract_numeric_occurrences=extract_numeric_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
        ):
            return True
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "min"
            and len(node.args) == 2
            and not node.keywords
        ):
            continue
        for subject, cap in (
            (node.args[0], node.args[1]),
            (node.args[1], node.args[0]),
        ):
            if not (
                _formula_node_references_names(subject, input_names)
                and not _formula_node_references_names(cap, input_names)
            ):
                continue
            cap_names = {
                candidate.id
                for candidate in ast.walk(cap)
                if isinstance(candidate, ast.Name)
            }
            if not cap_names or cap_names.issubset(boundary_names):
                cap_value = _evaluate_condition_expression(cap, formula_environment)
                if _rulespec_runtime_decimal(
                    cap_value
                ) is not None and numeric_value_is_grounded(
                    float(cap_value), (boundary,)
                ):
                    return True
            if _formula_clamp_preserves_boundary_offset(
                subject,
                cap,
                boundary_names=boundary_names,
                boundary=boundary,
                formula_environment=formula_environment,
                numeric_value_is_grounded=numeric_value_is_grounded,
            ):
                return True
    return False


def _formula_lower_boundary_clamp_matches(
    node: ast.AST,
    *,
    expression: ast.expr,
    input_names: set[str],
    source_bound_boundary_names: set[str],
    source_bound_constant_occurrences: dict[str, tuple[NumericOccurrenceLike, ...]],
    source_interval: _NumericInterval,
    boundary: NumericOccurrenceLike,
    formula_environment: dict[str, Any],
    extract_numeric_occurrences: NumericOccurrenceExtractor | None,
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> bool:
    """Accept an exact, source-bound lower clamp that causally contributes."""

    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "max"
        and len(node.args) == 2
        and not node.keywords
    ):
        return False
    zero, positive_subject = node.args
    if not (
        isinstance(zero, ast.Constant)
        and isinstance(zero.value, (int, float))
        and not isinstance(zero.value, bool)
        and zero.value == 0
    ):
        zero, positive_subject = positive_subject, zero
    if not (
        isinstance(zero, ast.Constant)
        and isinstance(zero.value, (int, float))
        and not isinstance(zero.value, bool)
        and zero.value == 0
        and isinstance(positive_subject, ast.BinOp)
        and isinstance(positive_subject.op, ast.Sub)
        and not _formula_node_references_names(positive_subject.right, input_names)
    ):
        return False
    subject = positive_subject.left
    source_bound_cap: ast.Name | None = None
    if isinstance(subject, ast.Name) and subject.id in input_names:
        pass
    elif (
        isinstance(subject, ast.Call)
        and isinstance(subject.func, ast.Name)
        and subject.func.id == "min"
        and len(subject.args) == 2
        and not subject.keywords
    ):
        for candidate_input, candidate_cap in (
            (subject.args[0], subject.args[1]),
            (subject.args[1], subject.args[0]),
        ):
            if (
                isinstance(candidate_input, ast.Name)
                and candidate_input.id in input_names
                and isinstance(candidate_cap, ast.Name)
                and candidate_cap.id in source_bound_constant_occurrences
                and not _formula_node_references_names(candidate_cap, input_names)
            ):
                source_bound_cap = candidate_cap
                break
        if source_bound_cap is None:
            return False
    else:
        return False
    floor = positive_subject.right
    named_floor = (
        isinstance(floor, ast.Name) and floor.id in source_bound_boundary_names
    )
    literal_floor = (
        isinstance(floor, ast.Constant)
        and isinstance(floor.value, (int, float))
        and not isinstance(floor.value, bool)
    )
    if not (named_floor or literal_floor):
        return False
    floor_value = _formula_node_boundary_value(
        floor,
        boundary_names=source_bound_boundary_names,
        boundary=boundary,
        formula_environment=formula_environment,
        extract_numeric_occurrences=extract_numeric_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    if source_bound_cap is not None:
        cap_value = _rulespec_runtime_decimal(
            _evaluate_condition_expression(source_bound_cap, formula_environment)
        )
        floor_decimal = _rulespec_runtime_decimal(floor_value)
        cap_occurrences = source_bound_constant_occurrences.get(
            source_bound_cap.id,
            (),
        )
        if (
            cap_value is None
            or floor_decimal is None
            or cap_value <= floor_decimal
            or source_interval.upper is None
            or not source_interval.upper_directly_conjoined
            or not numeric_value_is_grounded(
                float(cap_value),
                cap_occurrences,
            )
            or not numeric_value_is_grounded(
                float(cap_value),
                (source_interval.upper,),
            )
        ):
            return False
    return bool(
        floor_value is not None
        and numeric_value_is_grounded(float(floor_value), (boundary,))
        and _formula_node_causally_contributes(
            expression,
            node,
            formula_environment=formula_environment,
        )
    )


def _formula_node_causally_contributes(
    expression: ast.expr,
    target: ast.AST,
    *,
    formula_environment: dict[str, Any],
) -> bool:
    """Require changing one clamp result to change the reached formula value."""

    target_location = (
        type(target),
        getattr(target, "lineno", None),
        getattr(target, "col_offset", None),
        getattr(target, "end_lineno", None),
        getattr(target, "end_col_offset", None),
    )

    def evaluate_with_replacement(replacement: int) -> Decimal | None:
        replaced = False

        class ReplaceTarget(ast.NodeTransformer):
            def generic_visit(self, candidate: ast.AST):
                nonlocal replaced
                candidate_location = (
                    type(candidate),
                    getattr(candidate, "lineno", None),
                    getattr(candidate, "col_offset", None),
                    getattr(candidate, "end_lineno", None),
                    getattr(candidate, "end_col_offset", None),
                )
                if not replaced and candidate_location == target_location:
                    replaced = True
                    return ast.copy_location(ast.Constant(replacement), candidate)
                return super().generic_visit(candidate)

        candidate = ReplaceTarget().visit(copy.deepcopy(expression))
        if not replaced:
            return None
        return _rulespec_runtime_decimal(
            _evaluate_condition_expression(candidate, formula_environment)
        )

    zero_value = evaluate_with_replacement(0)
    one_value = evaluate_with_replacement(1)
    return bool(
        zero_value is not None and one_value is not None and zero_value != one_value
    )


def _formula_clamp_preserves_boundary_offset(
    subject: ast.expr,
    cap: ast.expr,
    *,
    boundary_names: set[str],
    boundary: NumericOccurrenceLike,
    formula_environment: dict[str, Any],
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> bool:
    """Accept ``min(max(0, x - a), ceiling - a)`` as an exact ceiling clamp."""

    if not (isinstance(cap, ast.BinOp) and isinstance(cap.op, ast.Sub)):
        return False
    ceiling_names = {
        candidate.id
        for candidate in ast.walk(cap.left)
        if isinstance(candidate, ast.Name)
    }
    if ceiling_names and not ceiling_names.issubset(boundary_names):
        return False
    positive_subject = next(
        (
            operand
            for node in ast.walk(subject)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "max"
            for operand in node.args
            if not (isinstance(operand, ast.Constant) and operand.value == 0)
        ),
        None,
    )
    if not (
        isinstance(positive_subject, ast.BinOp)
        and isinstance(positive_subject.op, ast.Sub)
        and ast.dump(positive_subject.right) == ast.dump(cap.right)
    ):
        return False
    ceiling_value = _evaluate_condition_expression(cap.left, formula_environment)
    return bool(
        _rulespec_runtime_decimal(ceiling_value) is not None
        and numeric_value_is_grounded(float(ceiling_value), (boundary,))
    )


def _parse_formula_expression(text: str) -> ast.expr | None:
    """Parse one expression, including Axiom's multiline continuations."""

    stripped = text.strip()
    try:
        return ast.parse(stripped, mode="eval").body
    except SyntaxError:
        try:
            return ast.parse(f"(\n{stripped}\n)", mode="eval").body
        except SyntaxError:
            return None


def _formula_comparisons_with_polarity(
    node: ast.AST,
    *,
    negated: bool = False,
) -> Iterable[tuple[ast.Compare, bool]]:
    """Yield comparisons with the boolean negation parity that contains them."""

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        yield from _formula_comparisons_with_polarity(
            node.operand,
            negated=not negated,
        )
        return
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and len(node.args) == 1
        and not node.keywords
        and node.func.id in {"holds", "not_holds"}
    ):
        yield from _formula_comparisons_with_polarity(
            node.args[0],
            negated=negated != (node.func.id == "not_holds"),
        )
        return
    if isinstance(node, ast.Compare):
        yield node, negated
        return
    for child in ast.iter_child_nodes(node):
        yield from _formula_comparisons_with_polarity(
            child,
            negated=negated,
        )


def _formula_node_references_names(
    node: ast.AST,
    names: set[str],
) -> bool:
    return bool(
        {
            candidate.id
            for candidate in ast.walk(node)
            if isinstance(candidate, ast.Name)
        }
        & names
    )


def _formula_node_boundary_value(
    node: ast.AST,
    *,
    boundary_names: set[str],
    boundary: NumericOccurrenceLike,
    formula_environment: dict[str, Any],
    extract_numeric_occurrences: NumericOccurrenceExtractor | None,
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> float | None:
    if isinstance(node, ast.Name) and node.id in boundary_names:
        value = formula_environment.get(node.id)
        if _rulespec_runtime_decimal(value) is not None:
            return float(value)
    if (
        isinstance(node, ast.Constant)
        and isinstance(
            node.value,
            (int, float),
        )
        and not isinstance(node.value, bool)
    ):
        value = float(node.value)
        if numeric_value_is_grounded(
            value,
            (boundary,),
        ) or _is_adjacent_integral_boundary(value, boundary):
            return value
        return None
    if extract_numeric_occurrences is None:
        return None
    text = ast.unparse(node)
    return next(
        (
            float(occurrence.value)
            for occurrence in extract_numeric_occurrences(text)
            if numeric_value_is_grounded(float(occurrence.value), (boundary,))
            or _is_adjacent_integral_boundary(
                float(occurrence.value),
                boundary,
            )
        ),
        None,
    )


def _source_interval_and_polarity_for_boundary(
    branch: SourceStructureBranch,
    boundary: NumericOccurrenceLike,
    *,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> tuple[_NumericInterval | None, int]:
    direct_text = authoritative_numeric_recall_text(branch.text)
    for fragment_start, fragment in _source_boundary_fragments(direct_text):
        interval = _shift_numeric_interval(
            _formula_interval_from_text(
                fragment.split(":", 1)[0],
                extract_numeric_occurrences=extract_numeric_occurrences,
            ),
            fragment_start,
        )
        if interval is None:
            continue
        if any(
            occurrence is not None
            and occurrence.start == boundary.start
            and occurrence.end == boundary.end
            and _numeric_occurrences_are_equivalent(occurrence, boundary)
            for occurrence in (interval.lower, interval.upper)
        ):
            return interval, _source_boundary_boolean_polarity(fragment)
    return None, 0


def _source_boundary_fragments(text: str) -> Iterable[tuple[int, str]]:
    """Yield nonempty boundary fragments with exact offsets in ``text``."""

    cursor = 0
    for separator in re.finditer(
        r"(?:[;\n]+|(?<=[.!?])\s+)",
        text,
    ):
        raw_fragment = text[cursor : separator.start()]
        left_trim = len(raw_fragment) - len(raw_fragment.lstrip())
        right_end = len(raw_fragment.rstrip())
        if left_trim < right_end:
            yield cursor + left_trim, raw_fragment[left_trim:right_end]
        cursor = separator.end()
    raw_fragment = text[cursor:]
    left_trim = len(raw_fragment) - len(raw_fragment.lstrip())
    right_end = len(raw_fragment.rstrip())
    if left_trim < right_end:
        yield cursor + left_trim, raw_fragment[left_trim:right_end]


def _shift_numeric_occurrence(
    occurrence: NumericOccurrenceLike | None,
    offset: int,
) -> NumericOccurrenceLike | None:
    if occurrence is None:
        return None
    return _NumericOccurrenceView(
        value=float(occurrence.value),
        start=occurrence.start + offset,
        end=occurrence.end + offset,
        raw=occurrence.raw,
        has_rate_context=occurrence.has_rate_context,
        has_temporal_context=occurrence.has_temporal_context,
        has_structural_context=occurrence.has_structural_context,
        source_value=occurrence.source_value,
        requires_rate_context=occurrence.requires_rate_context,
        is_word_number=occurrence.is_word_number,
        alternative_values=occurrence.alternative_values,
    )


def _shift_numeric_interval(
    interval: _NumericInterval | None,
    offset: int,
) -> _NumericInterval | None:
    if interval is None:
        return None
    return _NumericInterval(
        lower=_shift_numeric_occurrence(interval.lower, offset),
        lower_inclusive=interval.lower_inclusive,
        upper=_shift_numeric_occurrence(interval.upper, offset),
        upper_inclusive=interval.upper_inclusive,
        upper_directly_conjoined=interval.upper_directly_conjoined,
    )


def _comparison_matches_source_interval(
    operator: ast.cmpop,
    *,
    input_on_left: bool,
    compared_boundary_value: float,
    boundary_names: set[str],
    boundary: NumericOccurrenceLike,
    numeric_value_is_grounded: NumericGroundingPredicate,
    source_interval: _NumericInterval,
    allow_complement_relation: bool,
    negated: bool,
) -> bool:
    del boundary_names
    relation = _input_comparison_relation(operator, input_on_left=input_on_left)
    if relation is None:
        return False
    if negated:
        relation = _complement_comparison_relation(relation)
    relations = {relation}
    if allow_complement_relation:
        relations.add(_complement_comparison_relation(relation))
    is_exact = numeric_value_is_grounded(
        compared_boundary_value,
        (boundary,),
    )
    boundary_value = float(boundary.value)
    is_lower = (
        source_interval.lower is not None
        and _numeric_occurrences_are_equivalent(
            source_interval.lower,
            boundary,
        )
    )
    is_upper = (
        source_interval.upper is not None
        and _numeric_occurrences_are_equivalent(
            source_interval.upper,
            boundary,
        )
    )
    if is_lower:
        expected = ">=" if source_interval.lower_inclusive else ">"
        if is_exact:
            return expected in relations
        if (
            _is_adjacent_integral_boundary(
                compared_boundary_value,
                boundary,
            )
            and source_interval.lower_inclusive
            and math.isclose(compared_boundary_value, boundary_value - 1)
        ):
            return ">" in relations
        if (
            _is_adjacent_integral_boundary(
                compared_boundary_value,
                boundary,
            )
            and not source_interval.lower_inclusive
            and math.isclose(compared_boundary_value, boundary_value + 1)
        ):
            return ">=" in relations
    if is_upper:
        expected = "<=" if source_interval.upper_inclusive else "<"
        if is_exact:
            return expected in relations
        if (
            _is_adjacent_integral_boundary(
                compared_boundary_value,
                boundary,
            )
            and source_interval.upper_inclusive
            and math.isclose(compared_boundary_value, boundary_value + 1)
        ):
            return "<" in relations
        if (
            _is_adjacent_integral_boundary(
                compared_boundary_value,
                boundary,
            )
            and not source_interval.upper_inclusive
            and math.isclose(compared_boundary_value, boundary_value - 1)
        ):
            return "<=" in relations
    return False


def _input_comparison_relation(
    operator: ast.cmpop,
    *,
    input_on_left: bool,
) -> str | None:
    relation = {
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.Eq: "==",
        ast.NotEq: "!=",
    }.get(type(operator))
    if relation is None or input_on_left:
        return relation
    return {
        "<": ">",
        "<=": ">=",
        ">": "<",
        ">=": "<=",
        "==": "==",
        "!=": "!=",
    }[relation]


def _complement_comparison_relation(relation: str) -> str:
    return {
        "<": ">=",
        "<=": ">",
        ">": "<=",
        ">=": "<",
        "==": "!=",
        "!=": "==",
    }[relation]


def _boundary_case_changes_formula_effect(
    rule: dict[str, Any],
    case: dict[str, Any],
    *,
    input_key: object,
    selector_names: set[str],
    boundary_value: float,
    execution: _FormulaExecution,
    principal_rules: dict[str, dict[str, Any]],
    dependency_names: set[str],
    formula_environment: dict[str, Any],
    source_interval: _NumericInterval | None,
    source_boolean_polarity: int,
    root_rule: dict[str, Any] | None = None,
    root_execution: _FormulaExecution | None = None,
) -> bool:
    inputs = case.get("input")
    if not isinstance(inputs, dict):
        return False
    boundary_step = (
        1.0
        if float(boundary_value).is_integer()
        else max(abs(boundary_value) * 1e-6, 1e-9)
    )
    controller_signature = _formula_execution_effect_signature(execution)
    effect_rule = root_rule or rule
    effect_execution = root_execution or execution
    root_signature = _formula_execution_effect_signature(effect_execution)
    boundary_boolean = _formula_execution_boolean_value(execution)
    if (
        boundary_boolean is not None
        and source_interval is not None
        and source_boolean_polarity
        and boundary_boolean
        != (
            _interval_contains(source_interval, float(boundary_value))
            if source_boolean_polarity > 0
            else not _interval_contains(
                source_interval,
                float(boundary_value),
            )
        )
    ):
        return False
    for raw_key, raw_value in inputs.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            continue
        directly_controls_selector = raw_key == input_key and bool(
            _input_key_names(raw_key) & selector_names
        )
        raw_step = (
            boundary_step
            if directly_controls_selector
            else (
                1.0
                if float(raw_value).is_integer()
                else max(abs(float(raw_value)) * 1e-6, 1e-9)
            )
        )
        base_value = boundary_value if directly_controls_selector else float(raw_value)
        for candidate_value in (
            base_value - raw_step,
            base_value + raw_step,
        ):
            candidate_inputs = dict(inputs)
            candidate_inputs[raw_key] = candidate_value
            candidate_case = dict(case)
            candidate_case["input"] = candidate_inputs
            candidate_dependencies = _case_dependency_environment(
                principal_rules,
                candidate_case,
                formula_environment=formula_environment,
                require_asserted_value=False,
                allowed_names=dependency_names,
            )
            candidate_selector_values = tuple(
                value
                for _key, names, value in _case_numeric_selector_evidence(
                    candidate_case,
                    selector_names,
                    dependency_environment=candidate_dependencies,
                )
                if names & selector_names
            )
            if not any(
                not math.isclose(value, float(boundary_value))
                for value in candidate_selector_values
            ):
                continue
            candidate_execution = _case_formula_execution(
                rule,
                candidate_case,
                formula_environment=formula_environment,
                dependency_environment=candidate_dependencies,
            )
            candidate_boolean = (
                _formula_execution_boolean_value(candidate_execution)
                if candidate_execution is not None
                else None
            )
            candidate_root_execution = _case_formula_execution(
                effect_rule,
                candidate_case,
                formula_environment=formula_environment,
                dependency_environment=candidate_dependencies,
            )
            if (
                candidate_execution is not None
                and candidate_root_execution is not None
                and (
                    candidate_boolean is None
                    or source_interval is None
                    or not source_boolean_polarity
                    or any(
                        candidate_boolean
                        == (
                            _interval_contains(source_interval, value)
                            if source_boolean_polarity > 0
                            else not _interval_contains(
                                source_interval,
                                value,
                            )
                        )
                        for value in candidate_selector_values
                    )
                )
                and _formula_execution_effect_signature(candidate_execution)
                != controller_signature
                and _formula_execution_effect_signature(candidate_root_execution)
                != root_signature
            ):
                return True
    return False


def _formula_execution_boolean_value(
    execution: _FormulaExecution | None,
) -> bool | None:
    if execution is None:
        return None
    return _boolean_value(_formula_execution_runtime_value(execution))


def _source_boundary_boolean_polarity(text: str) -> int:
    lowered = _collapse_text(text).lower()
    if re.search(
        r"\b(?:keine?\s+berechtigung|keinen?\s+anspruch|"
        r"nicht\s+berechtigt|ausgeschlossen|gilt\s+nicht|"
        r"does\s+not\s+apply|not\s+applicable|not\s+eligible|"
        r"ineligible|excluded)\b",
        lowered,
    ):
        return -1
    if re.search(
        r"\b(?:gilt(?:\s+für)?|anspruch\s+besteht|berechtigt|"
        r"appl(?:y|ies)|eligible|qualif(?:y|ies))\b",
        lowered,
    ):
        return 1
    return 0


def _is_adjacent_integral_boundary(
    value: float,
    boundary: NumericOccurrenceLike,
) -> bool:
    boundary_value = float(boundary.value)
    return (
        not boundary.has_rate_context
        and value.is_integer()
        and boundary_value.is_integer()
        and math.isclose(abs(value - boundary_value), 1.0)
    )


def _rules_covering_branch(
    branch: SourceStructureBranch,
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
) -> tuple[str, ...]:
    return tuple(
        name for name, paths in principal_rule_paths.items() if branch.path in paths
    )


def _formula_branch_interval(
    branch: SourceStructureBranch,
    *,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> _NumericInterval | None:
    first_line = branch.text.splitlines()[0] if branch.text.splitlines() else ""
    range_text = authoritative_numeric_recall_text(first_line)
    return _formula_interval_from_text(
        range_text,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )


def _formula_conjoined_bound(
    text: str,
    occurrences: tuple[NumericOccurrenceLike, ...],
) -> tuple[NumericOccurrenceLike, bool, str] | None:
    """Parse a bounded second constraint following the first numeric threshold."""

    if len(occurrences) < 2:
        return None
    upper_gap = text[occurrences[0].end : occurrences[1].start]
    normalized_upper_gap = " ".join(upper_gap.replace(",", " , ").split())
    gap_match = re.fullmatch(
        r"(?:(?:dollars?|usd|euros?|eur) )?(?:, )?"
        r"(?:and|but|und) (?P<body>.+)",
        normalized_upper_gap,
        flags=re.IGNORECASE,
    )
    if gap_match is None:
        return (
            (occurrences[1], True, "upper")
            if _formula_has_direct_connectorless_upper_bound(text, occurrences)
            else None
        )
    return _formula_bound_from_comparison(
        gap_match.group("body"),
        occurrences[1],
    )


def _formula_has_direct_connectorless_upper_bound(
    text: str,
    occurrences: tuple[NumericOccurrenceLike, ...],
) -> bool:
    """Recognize only the source-continuation form used by a bounded band."""

    if len(occurrences) < 2:
        return False
    gap = " ".join(
        text[occurrences[0].end : occurrences[1].start].replace(",", " , ").split()
    )
    if (
        re.fullmatch(
            r"(?:(?:dollars?|usd|euros?|eur) )?"
            r"(?:\) )?up\s+to(?:\s+and\s+including)?"
            rf"(?: {_ENGLISH_CARDINAL_PHRASE} dollars? \()?"
            r"\s*(?:\$|€|£|usd|eur|gbp)?",
            gap,
            flags=re.IGNORECASE,
        )
        is None
    ):
        return False
    trailing = text[occurrences[1].end :]
    return (
        re.match(
            r"\s+(?:of|for)\s+(?:(?:a|an|the|this|that)\s+)?"
            r"(?:[a-z-]+\s+){0,5}(?:amount|credit|deduction|income|limit|total|value)\b",
            trailing,
            flags=re.IGNORECASE,
        )
        is None
    )


def _formula_bound_from_comparison(
    comparison: str,
    occurrence: NumericOccurrenceLike,
) -> tuple[NumericOccurrenceLike, bool, str] | None:
    """Parse one comparison while preserving modal and negative control."""

    comparison = comparison.lower().strip()
    comparison = re.sub(r"\s*,\s*", " ", comparison)
    comparison = re.sub(r"\s*[;:]\s*$", "", comparison)
    comparison = re.sub(
        r"\s*(?:\$|€|£|dollars?|euros?|usd|eur|gbp)\s*$",
        "",
        comparison,
        flags=re.IGNORECASE,
    )
    comparison = re.sub(
        r"\s+(?:an?|the)\s+"
        r"(?:[a-z][a-z-]*\s+){0,6}"
        r"(?:amount|income|limit|threshold(?:\s+amount)?|total|value)"
        r"(?:\s+(?:of|equal\s+to))?\s*$",
        "",
        comparison,
    )

    modal = r"(?:can|could|may|might|must|shall|should|will|would)"
    control = (
        r"(?:at\s+no\s+time|in\s+no\s+(?:event|case)|not\s+ever|"
        r"under\s+no\s+(?:conditions?|circumstances)|never)"
    )
    has_negative_control = False
    modal_seen: str | None = None
    subject = (
        r"(?:it|(?:(?:the|this|that|such)\s+)?"
        r"(?:[a-z][a-z-]*\s+){0,5}(?:amount|income|limit|total|value))"
    )
    for _ in range(8):
        control_match = re.match(rf"{control}\s+", comparison)
        if control_match is not None:
            has_negative_control = True
            comparison = comparison[control_match.end() :]
            continue
        modal_match = re.match(rf"{modal}\s+", comparison)
        if modal_match is not None:
            modal_seen = modal_match.group().strip()
            comparison = comparison[modal_match.end() :]
            comparison = re.sub(
                r"^(?:(?:always|at\s+all\s+times|in\s+(?:all|each)\s+cases?|"
                r"in\s+every\s+(?:case|instance)|under\s+all\s+circumstances|"
                r"without\s+exception|[a-z]+ly)\s+){1,3}",
                "",
                comparison,
                flags=re.IGNORECASE,
            )
            continue
        need_not_match = re.match(r"need\s+not\s+", comparison)
        if need_not_match is not None:
            modal_seen = "need not"
            comparison = comparison[need_not_match.end() :]
            continue
        cannot_match = re.match(
            r"(?:can\s+not|cannot)(?:\s+(?:ever|[a-z]+ly)){0,3}\s+",
            comparison,
        )
        if cannot_match is not None:
            has_negative_control = True
            comparison = comparison[cannot_match.end() :]
            continue
        prohibited_match = re.match(
            r"(?:is|are|be)\s+(?:[a-z]+ly\s+){0,2}"
            r"(?:not\s+(?:allowed|authorized|permitted)\s+to|"
            r"unauthorized\s+to|bound\s+not\s+to|"
            r"(?:barred|disallowed|forbidden|precluded|prevented|prohibited)"
            r"(?:\s+(?:ever|[a-z]+ly)){0,2}\s+(?:from|to))\s+",
            comparison,
        )
        if prohibited_match is not None:
            has_negative_control = True
            comparison = comparison[prohibited_match.end() :]
            continue
        permissive_predicate_match = re.match(
            r"(?:is|are|be)\s+(?:[a-z]+ly\s+){0,2}"
            r"(?:allowed|authorized|permitted)\s+to\s+",
            comparison,
        )
        if permissive_predicate_match is not None:
            modal_seen = "may"
            comparison = comparison[permissive_predicate_match.end() :]
            continue
        required_predicate_match = re.match(
            r"(?:is|are|be)\s+(?:[a-z]+ly\s+){0,2}"
            r"(?:compelled|directed|mandated|obliged|ordered|required)\s+"
            r"(?:(?:by|under)\s+(?:(?:this|that|the)\s+)?"
            r"[a-z][a-z0-9 .():§'-]{0,50}?\s+)?"
            r"(?P<negative>not\s+)?to\s+",
            comparison,
        )
        if required_predicate_match is not None:
            modal_seen = "must"
            comparison = comparison[required_predicate_match.end() :]
            if required_predicate_match.group("negative"):
                comparison = f"not {comparison}"
            continue
        bound_predicate_match = re.match(
            r"(?:is|are|be)\s+(?:[a-z]+ly\s+){0,2}"
            r"(?:bound|duty-bound)\s+"
            r"(?:(?:by|under)\s+(?:(?:this|that|the)\s+)?"
            r"[a-z][a-z0-9 .():§'-]{0,50}?\s+)?"
            r"(?P<negative>not\s+)?to\s+",
            comparison,
        )
        if bound_predicate_match is not None:
            modal_seen = "must"
            comparison = comparison[bound_predicate_match.end() :]
            if bound_predicate_match.group("negative"):
                comparison = f"not {comparison}"
            continue
        descriptive_predicate_match = re.match(
            r"(?:(?:is|are|be)\s+(?:anticipated\s+to|apt\s+to|capable\s+of|"
            r"designed\s+to|estimated\s+to|expected\s+to|forecast\s+to|"
            r"free\s+to|liable\s+to|likely\s+to|presumed\s+to|prone\s+to|"
            r"projected\s+to|supposed\s+to|unlikely\s+to)|"
            r"(?:appears?|seems?|tends?)\s+to)\s+",
            comparison,
        )
        if descriptive_predicate_match is not None:
            modal_seen = "may"
            comparison = comparison[descriptive_predicate_match.end() :]
            continue
        subject_match = re.match(rf"{subject}\s+", comparison)
        if subject_match is not None:
            comparison = comparison[subject_match.end() :]
            continue
        descriptive_adverb_match = re.match(
            r"(?:commonly|frequently|generally|normally|occasionally|often|"
            r"ordinarily|possibly|rarely|sometimes|typically|usually)\s+",
            comparison,
        )
        if descriptive_adverb_match is not None:
            modal_seen = "may"
            comparison = comparison[descriptive_adverb_match.end() :]
            continue
        break
    comparison = comparison.strip()
    copula_match = re.match(r"^(?:is|are|be)\s+", comparison)
    copula_stripped = (
        comparison[copula_match.end() :] if copula_match is not None else comparison
    )
    negated_symbol = re.fullmatch(r"not\s*(?P<operator><=|>=|[<>≤≥])", copula_stripped)
    if negated_symbol is not None:
        has_negative_control = True
        copula_stripped = negated_symbol.group("operator")

    upper_exclusive_comparisons = {
        "<",
        "below",
        "cannot equal or exceed",
        "cannot equal or be greater than",
        "cannot be at least",
        "cannot be greater than or equal to",
        "less than",
        "not equal or exceed",
        "not equal or be greater than",
        "not at least",
        "not be at least",
        "not be greater than or equal to",
        "not greater than or equal to",
        "remain below",
        "under",
        "up to but not including",
        "up to but excluding",
        "von weniger als",
        "weniger als",
    }
    upper_inclusive_comparisons = {
        "<=",
        "≤",
        "at or below",
        "at or under",
        "at most",
        "cannot be above",
        "cannot be greater than",
        "cannot be higher than",
        "cannot be larger than",
        "cannot be more than",
        "höchstens",
        "equal to or less than",
        "less than or equal to",
        "no greater than",
        "no higher than",
        "no larger than",
        "no more than",
        "not above",
        "not be above",
        "not be greater than",
        "not be higher than",
        "not be larger than",
        "not be more than",
        "not greater than",
        "not higher than",
        "not over",
        "not in excess of",
        "not larger than",
        "not more than",
        "bis",
        "höchstens",
        "nicht mehr als",
        "up to",
        "up to and including",
        "weniger als oder gleich",
    }
    negative_exceed_comparisons = {
        "cannot exceed",
        "cannot ever exceed",
        "did not exceed",
        "do not exceed",
        "does not exceed",
        "not exceed",
        "not exceeding",
        "not to exceed",
    }
    lower_exclusive_comparisons = {
        ">",
        "above",
        "cannot be at most",
        "cannot be less than or equal to",
        "exceed",
        "exceeds",
        "exceeding",
        "greater than",
        "higher than",
        "in excess of",
        "larger than",
        "more than",
        "not equal or be less than",
        "not equal or less than",
        "not at most",
        "not be at most",
        "not be less than or equal to",
        "not less than or equal to",
        "over",
        "remain above",
        "von mehr als",
        "über",
    }
    lower_inclusive_comparisons = {
        ">=",
        "≥",
        "ab",
        "at or above",
        "at or over",
        "at least",
        "cannot be below",
        "cannot be less than",
        "cannot be lower than",
        "cannot be under",
        "cannot fall below",
        "did not fall below",
        "do not fall below",
        "does not fall below",
        "equal to or greater than",
        "equal or exceed",
        "equal or be greater than",
        "greater than or equal to",
        "no less than",
        "no lower than",
        "not be below",
        "not be less than",
        "not be lower than",
        "not be under",
        "not below",
        "not fall below",
        "not less than",
        "not lower than",
        "not under",
        "remain at or above",
        "remain at least",
        "from",
        "mindestens",
        "von",
    }
    if has_negative_control and copula_stripped in upper_exclusive_comparisons:
        return occurrence, True, "lower"
    if has_negative_control and copula_stripped in upper_inclusive_comparisons:
        return occurrence, False, "lower"
    if has_negative_control and copula_stripped in lower_exclusive_comparisons:
        return occurrence, True, "upper"
    if has_negative_control and copula_stripped in lower_inclusive_comparisons:
        return occurrence, False, "upper"
    permissive_modal = modal_seen in {
        "can",
        "could",
        "may",
        "might",
        "need not",
        "would",
    }
    prohibitive_comparison = bool(
        re.match(r"^(?:can\s+not|cannot|not)\b", copula_stripped)
        or comparison in negative_exceed_comparisons
        or copula_stripped in negative_exceed_comparisons
    )
    recognized_comparison = bool(
        copula_stripped
        in (
            upper_exclusive_comparisons
            | upper_inclusive_comparisons
            | negative_exceed_comparisons
            | lower_exclusive_comparisons
            | lower_inclusive_comparisons
        )
        or comparison in negative_exceed_comparisons
    )
    if permissive_modal and not prohibitive_comparison and recognized_comparison:
        # A sentinel prevents a later suffix retry from hardening the optional
        # comparison after its permissive modal has been discarded.
        return occurrence, False, "permissive"
    if copula_stripped in upper_exclusive_comparisons:
        return occurrence, False, "upper"
    if (
        copula_stripped in upper_inclusive_comparisons
        or comparison in negative_exceed_comparisons
        or copula_stripped in negative_exceed_comparisons
    ):
        return occurrence, True, "upper"
    if copula_stripped in lower_exclusive_comparisons:
        return occurrence, False, "lower"
    if copula_stripped in lower_inclusive_comparisons:
        return occurrence, True, "lower"
    return None


def _formula_first_bound(
    text: str,
    occurrence: NumericOccurrenceLike,
) -> tuple[NumericOccurrenceLike, bool, str] | None:
    """Parse the first threshold without dropping its modal prefix."""

    clause_start = max(
        text.rfind(".", 0, occurrence.start),
        text.rfind(";", 0, occurrence.start),
        text.rfind(":", 0, occurrence.start),
        text.rfind("\n", 0, occurrence.start),
    )
    prefix = _strip_source_clause_marker(text[clause_start + 1 : occurrence.start])
    prefix = re.sub(
        rf"\b{_ENGLISH_CARDINAL_PHRASE}\s+"
        r"(?:dollars?|usd|euros?|eur|pounds?|gbp)\s*"
        r"\(\s*(?:\$|€|£|usd|eur|gbp)?\s*$",
        "",
        prefix,
        flags=re.IGNORECASE,
    )
    starts = [match.start() for match in re.finditer(r"(?<![\w-])(?=\S)", prefix)]
    for start in starts[-18:]:
        bound = _formula_bound_from_comparison(prefix[start:], occurrence)
        if bound is not None:
            discarded_prefix = prefix[:start]
            if re.search(
                r"\b(?:allowed|anticipated|appears?|apt|authorized|can|capable|"
                r"could|designed|estimated|expected|forecast|free|liable|likely|"
                r"may|might|need|occasionally|permitted|possibly|presumed|prone|"
                r"projected|rarely|seems?|supposed|tends?|typically|unlikely|"
                r"would)\b",
                discarded_prefix,
                flags=re.IGNORECASE,
            ):
                return occurrence, False, "permissive"
            if re.search(
                r"\b(?:barred|bound|cannot|compelled|directed|disallowed|"
                r"forbidden|mandated|must|never|not|obliged|ordered|precluded|"
                r"prevented|prohibited|required|shall|should|unauthorized|will)\b",
                discarded_prefix,
                flags=re.IGNORECASE,
            ):
                continue
            return bound
    return None


def _formula_interval_with_conjoined_bound(
    first: NumericOccurrenceLike,
    first_inclusive: bool,
    first_kind: str,
    bound: tuple[NumericOccurrenceLike, bool, str] | None,
    *,
    upper_directly_conjoined: bool = False,
) -> _NumericInterval | None:
    """Compose either first threshold with an optional second constraint."""

    lower = first if first_kind == "lower" else None
    lower_inclusive = first_inclusive if lower is not None else False
    upper = first if first_kind == "upper" else None
    upper_inclusive = first_inclusive if upper is not None else False
    if bound is not None:
        occurrence, inclusive, kind = bound
        if kind == "lower":
            if lower is None or float(occurrence.value) > float(lower.value):
                lower = occurrence
                lower_inclusive = inclusive
            elif math.isclose(
                float(occurrence.value),
                float(lower.value),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                lower_inclusive = lower_inclusive and inclusive
        elif kind == "upper" and (
            upper is None or float(occurrence.value) < float(upper.value)
        ):
            upper = occurrence
            upper_inclusive = inclusive
        elif kind == "upper" and math.isclose(
            float(occurrence.value),
            float(upper.value),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            upper_inclusive = upper_inclusive and inclusive
    if lower is None and upper is None:
        return None
    if lower is not None and upper is not None:
        lower_value = float(lower.value)
        upper_value = float(upper.value)
        if lower_value > upper_value or (
            math.isclose(lower_value, upper_value, rel_tol=0.0, abs_tol=1e-9)
            and not (lower_inclusive and upper_inclusive)
        ):
            return None
    return _NumericInterval(
        lower,
        lower_inclusive,
        upper,
        upper_inclusive,
        upper_directly_conjoined=(
            upper_directly_conjoined
            and bound is not None
            and bound[2] == "upper"
            and upper is bound[0]
        ),
    )


def _formula_has_unequal_english_parenthetical_amount(text: str) -> bool:
    """Reject contradictory spelled and parenthetical English money amounts."""

    pattern = re.compile(
        rf"\b(?P<words>{_ENGLISH_CARDINAL_PHRASE})\s+"
        r"(?P<unit>dollars?|usd|euros?|eur|pounds?|gbp)\s*"
        r"\(\s*(?P<marker>\$|€|£|usd|eur|gbp)?\s*"
        r"(?P<number>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
        r"\s*\)",
        flags=re.IGNORECASE,
    )
    currency_family = {
        "$": "usd",
        "dollar": "usd",
        "dollars": "usd",
        "usd": "usd",
        "€": "eur",
        "euro": "eur",
        "euros": "eur",
        "eur": "eur",
        "£": "gbp",
        "pound": "gbp",
        "pounds": "gbp",
        "gbp": "gbp",
    }
    for match in pattern.finditer(text):
        marker = match.group("marker")
        if (
            marker is not None
            and currency_family[match.group("unit").casefold()]
            != currency_family[marker.casefold()]
        ):
            return True
        word_value = _english_cardinal_value(match.group("words"))
        if word_value is None:
            return True
        with contextlib.suppress(InvalidOperation):
            numeric_value = Decimal(match.group("number").replace(",", ""))
            if Decimal(str(word_value)) != numeric_value:
                return True
    return False


def _formula_interval_from_text(
    text: str,
    *,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> _NumericInterval | None:
    if _formula_has_unequal_english_parenthetical_amount(text):
        return None
    lowered = text.lower()
    clause = _strip_source_clause_marker(text).lower()
    if re.match(
        r"\s*ab\s+(?:(?:dem\s+)?\d{1,2}\.\s+)?(?:"
        r"januar|februar|märz|april|mai|juni|juli|august|september|"
        r"oktober|november|dezember|"
        r"january|february|march|april|may|june|july|august|"
        r"september|october|november|december"
        r")\b",
        clause,
        flags=re.IGNORECASE,
    ):
        return None
    keyword_matches = tuple(
        re.finditer(
            r"\b(?:zwischen|between|von\s+(?:mehr|weniger)\s+als|"
            r"mehr\s+als|weniger\s+als|"
            r"von(?!\s+(?:mehr\s+als|weniger\s+als|höchstens|mindestens|"
            r"nicht\s+mehr\s+als|über|unter))|"
            r"from|unter|"
            r"less\s+than\s+or\s+equal\s+to|"
            r"equal\s+to\s+or\s+less\s+than|no\s+(?:greater|higher|larger|more)\s+than|"
            r"not\s+(?:greater|higher|larger|more)\s+than|not\s+(?:in\s+excess\s+of|over)|"
            r"in\s+excess\s+of|"
            r"at\s+or\s+(?:below|under)|"
            r"no\s+less\s+than|not\s+less\s+than|less\s+than|below|"
            r"bis|up\s+to|höchstens|nicht\s+mehr\s+als|at\s+most|"
            r"greater\s+than\s+or\s+equal\s+to|equal\s+to\s+or\s+greater\s+than|"
            r"at\s+or\s+(?:above|over)|über|more\s+than|greater\s+than|"
            r"exceeds?|exceeding|above|ab|at\s+least|mindestens)\b",
            lowered,
        )
    )
    symbol_keywords = tuple(re.finditer(r"(?:<=|>=|[<>≤≥])", lowered))
    occurrence_text = re.sub(
        r"(?i)(?<=\bexceed):(?=\s)",
        " ",
        text,
    )
    extracted_occurrences = tuple(extract_numeric_occurrences(occurrence_text))
    numeric_occurrences_list: list[NumericOccurrenceLike] = []
    parenthetical_restatement_spans: list[tuple[int, int]] = []
    # Stable source ordering retains the extractor's preferred interpretation
    # when one numeric phrase emits multiple overlapping candidates. A spelled
    # amount followed by its equal parenthetical currency restatement is also
    # one legal threshold, not two consecutive interval bounds.
    for occurrence in sorted(extracted_occurrences, key=lambda item: item.start):
        if (
            numeric_occurrences_list
            and occurrence.start < numeric_occurrences_list[-1].end
        ):
            continue
        if numeric_occurrences_list:
            previous = numeric_occurrences_list[-1]
            restatement_gap = text[previous.end : occurrence.start]
            restatement_suffix = text[occurrence.end :]
            restatement_close = re.match(r"\s*\)", restatement_suffix)
            strict_parenthetical_restatement = (
                re.fullmatch(
                    r"\s*(?:dollars?|usd|euros?|eur|pounds?|gbp)?\s*"
                    r"\(\s*(?:\$|€|£|usd|eur|gbp)?\s*",
                    restatement_gap,
                    flags=re.IGNORECASE,
                )
                and restatement_close is not None
            )
            if strict_parenthetical_restatement:
                if not _numeric_occurrences_are_equivalent(previous, occurrence):
                    return None
                parenthetical_restatement_spans.append(
                    (
                        previous.end,
                        occurrence.end + restatement_close.end(),
                    )
                )
                continue
        numeric_occurrences_list.append(occurrence)
    numeric_occurrences = tuple(numeric_occurrences_list)
    comparison_characters = list(text)
    for start, end in parenthetical_restatement_spans:
        comparison_characters[start:end] = " " * (end - start)
    comparison_text = "".join(comparison_characters)
    keyword = None
    occurrences: tuple[NumericOccurrenceLike, ...] = ()
    for candidate in sorted(
        (*keyword_matches, *symbol_keywords),
        key=lambda match: match.start(),
    ):
        if candidate.group().lower() == "from":
            clause_start = max(
                lowered.rfind(".", 0, candidate.start()),
                lowered.rfind(";", 0, candidate.start()),
                lowered.rfind(":", 0, candidate.start()),
                lowered.rfind("\n", 0, candidate.start()),
            )
            candidate_prefix = _strip_source_clause_marker(
                lowered[clause_start + 1 : candidate.start()]
            ).strip()
            if candidate_prefix:
                subject_modifiers = (
                    r"(?:(?:adjusted|aggregate|annual|applicable|combined|"
                    r"federal|gross|household|individual|resident|state|taxable)"
                    r"\s+){0,8}"
                )
                subject = rf"(?:{_FORMULA_NUMERIC_RESULT_HEAD_PATTERN})"
                direct_subject = re.fullmatch(
                    rf"(?:(?:a|an|that|the|this)\s+)?{subject_modifiers}{subject}",
                    candidate_prefix,
                    flags=re.IGNORECASE,
                )
                embedded_subject = re.search(
                    rf"\b(?:for|whose|with)\s+"
                    rf"(?:(?:a|an|that|the|this)\s+)?"
                    rf"{subject_modifiers}{subject}$",
                    candidate_prefix,
                    flags=re.IGNORECASE,
                )
                introduced_subject = re.search(
                    rf"\b{subject}\s+(?:are|be|is|must\s+(?:be|range)|ranges?|"
                    r"shall\s+(?:be|range))$",
                    candidate_prefix,
                    flags=re.IGNORECASE,
                )
                if not (direct_subject or embedded_subject or introduced_subject):
                    continue
        candidate_occurrences = tuple(
            occurrence
            for occurrence in numeric_occurrences
            if occurrence.start >= candidate.start()
        )
        if not candidate_occurrences:
            continue
        first_gap = text[candidate.end() : candidate_occurrences[0].start]
        spelled_parenthetical_gap = re.fullmatch(
            rf"\s*{_ENGLISH_CARDINAL_PHRASE}\s+"
            r"(?:dollars?|usd|euros?|eur|pounds?|gbp)\s*"
            r"\(\s*(?:\$|€|£|usd|eur|gbp)?\s*",
            first_gap,
            flags=re.IGNORECASE,
        )
        if spelled_parenthetical_gap is not None or re.fullmatch(
            r"\s*(?:\$|€|£|usd|eur|gbp)?\s*(?:(?:zu|bis)\s+)?"
            r"(?:(?:einschließlich|maximal|inklusive|including|maximum)\s+)?"
            r"(?:(?:einem?|einer|dem|der|das)\s+)?"
            r"(?:(?:zu\s+versteuernd\w*|maßgeblich\w*)\s+)?"
            r"(?:(?:einkommen|betrag|wert|income|amount)\s+)?"
            r"(?:(?:von|of)\s+)?"
            r"(?:(?:einschließlich|maximal|inklusive|including|maximum)\s+)?",
            first_gap,
            flags=re.IGNORECASE,
        ):
            keyword = candidate
            occurrences = candidate_occurrences
            break
    if keyword is None:
        return None
    range_text = text[keyword.start() :]
    if re.match(
        r"ab\s+(?:dem\s+)?(?:veranlagungszeitraum|tax\s+year|calendar\s+year)\b",
        range_text,
        flags=re.IGNORECASE,
    ):
        return None
    lowered_range = range_text.lower()
    if re.match(r"(?:zwischen|between)\b", lowered_range) and re.search(
        r"\b(?:und|and)\b",
        lowered_range,
    ):
        if len(occurrences) < 2:
            return None
        if float(occurrences[0].value) > float(occurrences[1].value):
            return None
        return _NumericInterval(occurrences[0], True, occurrences[1], True)
    if re.match(r"(?:von|from)\b", lowered_range) and re.search(
        r"\b(?:bis|to|through)\b",
        lowered_range,
    ):
        if len(occurrences) < 2:
            return None
        if float(occurrences[0].value) > float(occurrences[1].value):
            return None
        return _NumericInterval(occurrences[0], True, occurrences[1], True)
    upper_directly_conjoined = _formula_has_direct_connectorless_upper_bound(
        comparison_text,
        occurrences,
    )
    first_bound = _formula_first_bound(comparison_text, occurrences[0])
    if first_bound is not None:
        first, first_inclusive, first_kind = first_bound
        return _formula_interval_with_conjoined_bound(
            first,
            first_inclusive,
            first_kind,
            _formula_conjoined_bound(comparison_text, occurrences),
            upper_directly_conjoined=upper_directly_conjoined,
        )
    clause_start = max(
        lowered.rfind(".", 0, keyword.start()),
        lowered.rfind(";", 0, keyword.start()),
        lowered.rfind(":", 0, keyword.start()),
        lowered.rfind("\n", 0, keyword.start()),
    )
    unparsed_prefix = lowered[clause_start + 1 : keyword.start()]
    if re.search(
        r"\b(?:allowed|anticipated|appears?|apt|authorized|barred|bound|can|cannot|"
        r"capable|compelled|could|designed|directed|disallowed|estimated|"
        r"expected|forbidden|forecast|free|liable|likely|mandated|may|might|"
        r"must|need|never|not|obliged|ordered|permitted|possibly|precluded|"
        r"prevented|"
        r"presumed|prone|prohibited|projected|rarely|required|seems?|shall|"
        r"should|supposed|tends?|typically|unauthorized|unlikely|will|would)\b",
        unparsed_prefix,
        flags=re.IGNORECASE,
    ):
        second_bound = _formula_conjoined_bound(comparison_text, occurrences)
        if second_bound is None or second_bound[2] == "permissive":
            return None
        second, inclusive, kind = second_bound
        if kind == "lower":
            return _NumericInterval(second, inclusive, None, False)
        return _NumericInterval(None, False, second, inclusive)
    if re.match(
        r"(?:unter|less\s+than(?!\s+or\s+equal\s+to)|below|"
        r"(?:von\s+)?weniger\s+als)\b",
        lowered_range,
    ):
        return _formula_interval_with_conjoined_bound(
            occurrences[0],
            False,
            "upper",
            _formula_conjoined_bound(comparison_text, occurrences),
            upper_directly_conjoined=upper_directly_conjoined,
        )
    if re.match(
        r"(?:bis|up\s+to|höchstens|nicht\s+mehr\s+als|at\s+most|"
        r"less\s+than\s+or\s+equal\s+to|equal\s+to\s+or\s+less\s+than|"
        r"no\s+(?:greater|higher|larger|more)\s+than|"
        r"not\s+(?:greater|higher|larger|more)\s+than|"
        r"at\s+or\s+(?:below|under))\b",
        lowered_range,
    ):
        return _formula_interval_with_conjoined_bound(
            occurrences[0],
            True,
            "upper",
            _formula_conjoined_bound(comparison_text, occurrences),
            upper_directly_conjoined=upper_directly_conjoined,
        )
    if re.match(
        r"(?:(?:von\s+)?mehr\s+als|über|more\s+than|"
        r"greater\s+than(?!\s+or\s+equal\s+to)|in\s+excess\s+of|"
        r"exceeds?|exceeding|above)\b",
        lowered_range,
    ):
        return _formula_interval_with_conjoined_bound(
            occurrences[0],
            False,
            "lower",
            _formula_conjoined_bound(comparison_text, occurrences),
            upper_directly_conjoined=upper_directly_conjoined,
        )
    if re.match(
        r"(?:von|ab|from|at\s+least|no\s+less\s+than|"
        r"not\s+less\s+than|greater\s+than\s+or\s+equal\s+to|"
        r"equal\s+to\s+or\s+greater\s+than|at\s+or\s+(?:above|over)|"
        r"mindestens)\b",
        lowered_range,
    ):
        return _formula_interval_with_conjoined_bound(
            occurrences[0],
            True,
            "lower",
            _formula_conjoined_bound(comparison_text, occurrences),
            upper_directly_conjoined=upper_directly_conjoined,
        )
    return None


def _interval_contains(
    interval: _NumericInterval,
    value: float,
) -> bool:
    lower = interval.lower.value if interval.lower is not None else None
    upper = interval.upper.value if interval.upper is not None else None
    if lower is not None and (
        value < lower
        or (
            not interval.lower_inclusive
            and math.isclose(value, lower, rel_tol=0.0, abs_tol=1e-9)
        )
    ):
        return False
    if upper is not None and (
        value > upper
        or (
            not interval.upper_inclusive
            and math.isclose(value, upper, rel_tol=0.0, abs_tol=1e-9)
        )
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
    for operands_text in _balanced_call_operands(formula_text, "max"):
        expression = _parse_formula_expression(f"max({operands_text})")
        if expression is None:
            continue
        names.update(
            node.id
            for operands in _formula_positive_part_operands(expression)
            for operand in operands
            for node in ast.walk(operand)
            if isinstance(node, ast.Name)
        )
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
    *,
    dependency_environment: dict[str, Any] | None = None,
) -> tuple[float, ...]:
    return tuple(
        value
        for _key, _names, value in _case_numeric_selector_evidence(
            case,
            selector_names,
            dependency_environment=dependency_environment,
        )
    )


def _case_numeric_selector_evidence(
    case: dict[str, Any],
    selector_names: set[str],
    *,
    dependency_environment: dict[str, Any] | None = None,
) -> tuple[tuple[object, set[str], float], ...]:
    inputs = case.get("input")
    if not isinstance(inputs, dict):
        return ()
    evidence: list[tuple[object, set[str], float]] = []
    for key, value in inputs.items():
        matched_names = _input_key_names(key) & selector_names
        if not matched_names:
            continue
        evidence.extend(
            (key, matched_names, numeric_value)
            for numeric_value in _numeric_test_input_values(value)
        )
    directly_matched_names = (
        set().union(
            *(names for _key, names, _value in evidence),
        )
        if evidence
        else set()
    )
    for name, value in (dependency_environment or {}).items():
        if name not in selector_names or name in directly_matched_names:
            continue
        evidence.extend(
            (name, {name}, numeric_value)
            for numeric_value in _numeric_test_input_values(value)
        )
    return tuple(evidence)


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
    formula_branches: Sequence[SourceStructureBranch] = (),
) -> tuple[SourceStructureBranch, ...]:
    obligations: list[SourceStructureBranch] = []
    source_clauses = tuple(_source_clause_spans(source_text, branches=branches))
    matches_by_clause: dict[
        tuple[int, int, str],
        list[re.Match[str]],
    ] = {}
    for match in _source_exception_or_applicability_matches(source_text):
        if _span_is_deferred(
            match.start(),
            match.end(),
            branches=branches,
            deferred_paths=deferred_paths,
        ):
            continue
        clause = next(
            (
                (start, end, text)
                for start, end, text in source_clauses
                if start <= match.start() and match.end() <= end
            ),
            (match.start(), match.end(), match.group(0)),
        )
        matches_by_clause.setdefault(clause, []).append(match)

    for (
        clause_start,
        clause_end,
        clause_text,
    ), clause_matches in matches_by_clause.items():
        groups: list[list[re.Match[str]]] = []
        for match in clause_matches:
            if groups and _exception_cues_have_distinct_conditions(
                source_text[groups[-1][-1].end() : match.start()]
            ):
                groups.append([match])
            elif groups:
                groups[-1].append(match)
            else:
                groups.append([match])
        for group_index, group in enumerate(groups):
            match = group[0]
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
            if len(groups) == 1:
                branch_start = clause_start
                branch_end = clause_end
                branch_text = clause_text
            else:
                branch_start = match.start()
                branch_end = (
                    groups[group_index + 1][0].start()
                    if group_index + 1 < len(groups)
                    else clause_end
                )
                branch_text = source_text[branch_start:branch_end]
            if (
                _EXCEPTION_LANGUAGE.search(branch_text) is None
                and _source_has_arithmetic_comparison_condition(branch_text)
                and any(
                    formula_branch.start == branch_start
                    and formula_branch.end == branch_end
                    for formula_branch in formula_branches
                )
            ):
                continue
            obligations.append(
                SourceStructureBranch(
                    owner.path,
                    "exception-clause",
                    owner.label,
                    branch_text,
                    branch_start,
                    branch_end,
                )
            )
    return tuple(
        {
            (branch.path, branch.start, branch.end): branch for branch in obligations
        }.values()
    )


def _source_has_arithmetic_comparison_condition(text: str) -> bool:
    """Identify an arithmetic ``if/when`` already owned by formula coverage."""

    collapsed = _collapse_text(text)
    condition = re.search(
        r"\b(?:if|when|wenn|falls)\b(?P<body>[^,;.]*)",
        collapsed,
        flags=re.IGNORECASE,
    )
    if condition is None:
        return False
    return bool(
        re.search(
            r"\b(?:exceeds?|exceeded|greater\s+than|less\s+than|more\s+than|"
            r"at\s+least|at\s+most|above|below|over|under|"
            r"übersteigt|überschreitet|mehr\s+als|weniger\s+als|"
            r"mindestens|höchstens|über|unter)\b|[<>]=?",
            condition.group("body"),
            flags=re.IGNORECASE,
        )
    )


def _source_exception_or_applicability_matches(
    source_text: str,
) -> tuple[re.Match[str], ...]:
    """Return ordered, de-duplicated negative and positive condition cues."""

    matches = tuple(
        match
        for match in (
            *_EXCEPTION_LANGUAGE.finditer(source_text),
            *_APPLICABILITY_LANGUAGE.finditer(source_text),
        )
        if not _is_concessive_condition_cue(source_text, match)
    )
    return tuple(
        {
            (match.start(), match.end()): match
            for match in sorted(matches, key=lambda item: (item.start(), item.end()))
        }.values()
    )


def _is_concessive_condition_cue(
    source_text: str,
    match: re.Match[str],
) -> bool:
    """Exclude ``auch wenn`` / ``even if`` clauses that preserve the rule."""

    if match.group(0).strip().lower() not in {"wenn", "if"}:
        return False
    prefix = source_text[max(0, match.start() - 16) : match.start()]
    return (
        re.search(
            r"\b(?:auch|selbst|even)\s+$",
            prefix,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _exception_cues_have_distinct_conditions(between: str) -> bool:
    return (
        re.search(
            r"(?:;|\b(?:und|oder|and|or)\b)\s*$",
            between,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _source_exception_requires_paired_witness(
    text: str,
    *,
    branch_path: tuple[str, ...] = (),
    independently_covered_paths: set[tuple[str, ...]] | frozenset[tuple[str, ...]] = (
        frozenset()
    ),
) -> bool:
    """Return whether a source condition has two locally testable states.

    Cross-reference reservations and unconditional non-applicability rules are
    still validated as source structure and formula/output coverage.  They do
    not, however, describe a local condition that companion cases can toggle.
    """

    collapsed = _collapse_text(text)
    notwithstanding_tail = _louisiana_notwithstanding_reference_tail(collapsed)
    if notwithstanding_tail is not None:
        return _notwithstanding_reference_tail_requires_paired_witness(
            notwithstanding_tail
        )
    if _leading_reference_reservation_is_nonlocal(
        collapsed,
        branch_path=branch_path,
        independently_covered_paths=independently_covered_paths,
    ):
        return False
    if re.search(
        r"\bexcept\s+as\s+[^.;]{0,100}\bprovided\b",
        collapsed,
        flags=re.IGNORECASE,
    ) and _source_reference_reservation_is_only_references(collapsed):
        return False
    if _source_unconditional_nonapplicability(collapsed):
        return False
    if re.search(
        r"\b(?:subject\s+to|vorbehaltlich)\b",
        collapsed,
        flags=re.IGNORECASE,
    ) and _source_reference_reservation_is_only_references(collapsed):
        return False
    return True


def _leading_reference_reservation_is_nonlocal(
    text: str,
    *,
    branch_path: tuple[str, ...],
    independently_covered_paths: set[tuple[str, ...]] | frozenset[tuple[str, ...]],
) -> bool:
    """Exclude one resolved same-subsection override with no local selector."""

    clause = _strip_source_clause_marker(_collapse_text(text)).strip()
    marker = re.match(
        r"(?:subject\s+to|vorbehaltlich|except\s+as)\b",
        clause,
        flags=re.IGNORECASE,
    )
    if marker is None:
        return False
    delimiter = re.search(r"[,;]", clause[marker.end() :])
    if delimiter is None:
        return False
    delimiter_start = marker.end() + delimiter.start()
    reservation = clause[:delimiter_start]
    tail = clause[delimiter_start + 1 :]
    target_match = re.search(
        r"\bparagraph\s*\(?(?P<label>\d+[a-z]?)\)?\s+of\s+this\s+"
        r"subsection\b",
        reservation,
        flags=re.IGNORECASE,
    )
    if target_match is None or not branch_path:
        return False
    target_path = (*branch_path[:-1], target_match.group("label").lower())
    return bool(
        target_path != branch_path
        and target_path in independently_covered_paths
        and _source_reference_reservation_is_only_references(reservation)
        and source_states_explicit_computation(tail)
        and not _source_exception_or_applicability_matches(tail)
    )


def _louisiana_notwithstanding_reference_tail(text: str) -> str | None:
    """Return the rule tail after a leading Louisiana R.S. override reference."""

    match = _LOUISIANA_NOTWITHSTANDING_RS_REFERENCE_PREFIX.match(
        _strip_source_clause_marker(_collapse_text(text))
    )
    return match.group("tail").strip() if match is not None else None


def _notwithstanding_reference_tail_requires_paired_witness(text: str) -> bool:
    """Keep locally testable conditions after a non-toggleable override citation."""

    for match in _source_exception_or_applicability_matches(text):
        if _subject_to_is_administrative_effect(text, match):
            continue
        return True
    return bool(
        re.search(
            r"\b(?:exceeds?|exceeding|greater\s+than|less\s+than|more\s+than|"
            r"at\s+least|at\s+most|above|below)\b|[<>]=?",
            text,
            flags=re.IGNORECASE,
        )
    )


def _subject_to_is_administrative_effect(
    text: str,
    match: re.Match[str],
) -> bool:
    """Distinguish an oversight result from a ``subject to`` condition."""

    if match.group(0).strip().lower() != "subject to":
        return False
    prefix = text[max(0, match.start() - 32) : match.start()]
    suffix = text[match.end() : match.end() + 32]
    return bool(
        re.search(
            r"\b(?:(?:shall|must|will|may)\s+be|is|are|was|were)\s+$",
            prefix,
            flags=re.IGNORECASE,
        )
        and re.match(
            r"\s+(?:oversight|review|approval|audit|reporting)\b",
            suffix,
            flags=re.IGNORECASE,
        )
    )


def _source_has_formal_cross_reference(text: str) -> bool:
    target = (
        r"(?:chapter|title|article|part|subchapter|subtitle|division|section|"
        r"subsection|paragraph|act|law|code|gesetz|absatz|absätze)"
    )
    label = r"(?:\(?\d+[A-Za-z0-9.-]*\)?|\(?[A-Za-z]\)?|[IVXLCDM]+)"
    deictic = r"(?:this|that|the|these|those|dies(?:e[rmns]?))"
    if re.search(
        r"\b(?:all\s+)?(?:provisions?|restrictions?|vorschriften?|bestimmungen?)"
        rf"\s+(?:of|in|under|nach|gemäß)\s+(?:{deictic}\s+{target}"
        rf"(?:\s+{label})?|{target}\s+{label})(?=\W|$)|"
        rf"\b{deictic}\s+{target}\b|"
        rf"\b{target}\s+{label}(?=\W|$)|§",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    if any(
        pattern.search(text)
        for pattern in (
            _PRECISE_DEFERRAL_DEPENDENCY,
            _GERMAN_LEGAL_CITATION,
            _ENGLISH_LEGAL_CITATION,
            _STRUCTURAL_REFERENCE,
        )
    ):
        return True
    return bool(
        re.search(
            r"\b\d+\s+U\.?\s*S\.?\s*C\.?\s*(?:§|s\.)?\s*\d+[a-z0-9.-]*\b",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\b[A-Z][A-Z.]{1,8}\s*\d+[A-Za-z0-9:.-]*",
            text,
        )
    )


def _source_is_simple_main_proposition(text: str) -> bool:
    proposition = text.strip(" \t,;:.")
    determiner = r"(?:(?:a|an|the|this|that|such)\s+)?"
    program_modifier = (
        r"(?:(?:income|earned|child|dependent|property|sales|use|tax|"
        r"refundable|nonrefundable|standard|itemized|personal|business|"
        r"corporate|individual|state|federal|local|earned-income|child-tax|"
        r"income-tax|property-tax|sales-tax|use-tax)\s+)"
    )
    program_subject = (
        rf"(?:{program_modifier})*"
        r"(?:credit|claim|benefit|deduction|exemption|allowance|provision|"
        r"section|subsection|paragraph|rule)"
    )
    program_predicate = (
        r"(?:applies|shall\s+apply|is\s+applicable|is\s+allowed|is\s+available)"
    )
    eligible_subject = r"(?:claimant|taxpayer|person|individual)"
    german_subject = r"(?:der|die|das)\s+(?:anspruch|leistung|regel|vorschrift)"
    return bool(
        re.fullmatch(
            rf"(?:{determiner}{program_subject}\s+{program_predicate}|"
            rf"{determiner}{eligible_subject}\s+is\s+eligible|"
            rf"{german_subject}\s+(?:gilt|besteht)|"
            rf"(?:gilt|besteht)\s+{german_subject})",
            proposition,
            flags=re.IGNORECASE,
        )
    )


def _source_reference_reservation_is_only_references(text: str) -> bool:
    clause = _strip_source_clause_marker(_collapse_text(text)).strip()
    marker = re.search(
        r"\b(?P<marker>subject\s+to|vorbehaltlich|except\s+as)\b",
        clause,
        flags=re.IGNORECASE,
    )
    if marker is None:
        return False
    complement = clause[marker.end() :].strip(" \t,;:.")
    if marker.start() == 0:
        tail = clause[marker.end() :]
        delimiter = re.search(r"[,;]", tail)
        if delimiter is not None and _source_is_simple_main_proposition(
            tail[delimiter.end() :]
        ):
            complement = tail[: delimiter.start()].strip(" \t,;:.")
        elif delimiter is None:
            for proposition_start in re.finditer(
                r"\b(?:a|an|the|this|that|such|der|die|das|gilt|besteht)\b",
                tail,
                flags=re.IGNORECASE,
            ):
                if _source_is_simple_main_proposition(
                    tail[proposition_start.start() :]
                ):
                    complement = tail[: proposition_start.start()].strip(" \t,;:.")
                    break
    if marker.group("marker").lower().startswith("except") and re.fullmatch(
        r"(?:(?:may|might)\s+)?(?:otherwise\s+)?(?:specifically\s+)?"
        r"(?:be\s+)?provided(?:\s+by\s+law)?",
        complement,
        flags=re.IGNORECASE,
    ):
        return True
    if not _source_has_formal_cross_reference(complement):
        return False
    target = (
        r"(?:chapters?|titles?|articles?|parts?|subchapters?|subtitles?|"
        r"divisions?|sections?|subsections?|paragraphs?|acts?|laws?|codes?|"
        r"gesetz|absatz|absätze)"
    )
    label = r"(?:\(?\d+[A-Za-z0-9.-]*\)?|\(?[A-Za-z]\)?|[IVXLCDM]+)"
    deictic = r"(?:this|that|the|these|those|dies(?:e[rmns]?))"
    remainder = complement
    for pattern in (
        _GERMAN_LEGAL_CITATION,
        _ENGLISH_LEGAL_CITATION,
        _STRUCTURAL_REFERENCE,
    ):
        remainder = pattern.sub(" ", remainder)
    for pattern in (
        rf"\b{deictic}\s+{target}(?:\s+{label})?\b",
        rf"\b{target}\s+{label}(?=\W|$)",
        r"\b\d+\s+U\.?\s*S\.?\s*C\.?\s*(?:§|s\.)?\s*\d+[a-z0-9.-]*\b",
    ):
        remainder = re.sub(pattern, " ", remainder, flags=re.IGNORECASE)
    remainder = re.sub(
        r"\b[A-Z][A-Z.]{1,8}\s*\d+[A-Za-z0-9:.-]*",
        " ",
        remainder,
    )
    allowed_words = (
        r"all|the|this|that|these|those|provisions?|restrictions?|"
        r"modifications?|requirements?|limitations?|of|in|under|and|or|except|"
        r"as|may|might|be|otherwise|specifically|provided|by|law|code|act|et|al|"
        r"seq|chapters?|titles?|articles?|parts?|subchapters?|subtitles?|"
        r"divisions?|sections?|subsections?|paragraphs?|vorschriften?|"
        r"bestimmungen?|dies(?:e[rmns]?)|gemäß|nach|und|oder|gesetz|absatz|absätze"
    )
    remainder = re.sub(
        rf"\b(?:{allowed_words})\b",
        " ",
        remainder,
        flags=re.IGNORECASE,
    )
    remainder = re.sub(
        r"(?<![A-Za-z0-9])(?:\(?\d+[A-Za-z0-9.-]*\)?|\(?[A-Za-z]\)?|"
        r"[IVXLCDM]+)(?![A-Za-z0-9])",
        " ",
        remainder,
    )
    return re.fullmatch(r"[\s.,;:()§'/-]*", remainder) is not None


def _source_unconditional_nonapplicability(text: str) -> bool:
    clause = _strip_source_clause_marker(_collapse_text(text)).strip()
    clause = re.sub(
        r"^provided\s+however\s*,?\s+(?:that\s+)?",
        "",
        clause,
        count=1,
        flags=re.IGNORECASE,
    )
    negative_matches = tuple(_NEGATIVE_NONAPPLICABILITY_LANGUAGE.finditer(clause))
    if not negative_matches:
        return False
    negative = negative_matches[-1]
    if (
        re.match(r"gilt\b", negative.group(0), flags=re.IGNORECASE)
        and re.fullmatch(
            r"gilt\s+nicht",
            negative.group(0),
            flags=re.IGNORECASE,
        )
        is None
    ):
        return False
    if clause[negative.end() :].strip(" \t,;:."):
        return False
    subject = clause[: negative.start()].strip(" \t,;:")
    return _source_is_unconditional_subject(subject)


def _source_is_unconditional_subject(text: str) -> bool:
    """Accept only bare subjects or explicit legal-reference noun phrases."""

    subject = re.sub(
        r"^(?:the|this|that|such|der|die|das|dies(?:e[rmns]?))\s+",
        "",
        text.strip(),
        count=1,
        flags=re.IGNORECASE,
    )
    bare_subjects = (
        r"claimants?|persons?|individuals?|taxpayers?|residents?|claims?|credits?|"
        r"benefits?|payments?|deductions?|exemptions?|rules?|anspr(?:uch|üche)|"
        r"leistungen?|regeln?"
    )
    if re.fullmatch(bare_subjects, subject, flags=re.IGNORECASE):
        return True
    if re.fullmatch(
        r"(?:preceding|previous|following|vorherige[rmns]?|folgende[rmns]?)\s+"
        r"(?:sentence|paragraph|clause|satz|absatz)",
        subject,
        flags=re.IGNORECASE,
    ):
        return True
    reference_head = (
        r"provisions?|requirements?|limitations?|subsections?|sections?|"
        r"paragraphs?|clauses?|sentences?|chapters?|titles?|articles?|parts?|"
        r"subchapters?|subtitles?|divisions?|absatz|absätze|satz|sätze|nummern?"
    )
    head = re.match(rf"^(?P<head>{reference_head})\b", subject, flags=re.IGNORECASE)
    if head is None:
        return False
    tail = subject[head.end() :].strip()
    if not tail:
        return True
    reference_split = re.fullmatch(
        r"(?:(?P<labels>.*?)\s+)?(?:of|under|des|der)\s+(?P<reference>.+)",
        tail,
        flags=re.IGNORECASE,
    )
    if reference_split is not None:
        labels = (reference_split.group("labels") or "").strip()
        return (
            not labels or _source_reference_labels_are_structural(labels)
        ) and _source_has_formal_cross_reference(reference_split.group("reference"))
    return _source_reference_labels_are_structural(tail)


def _source_reference_labels_are_structural(text: str) -> bool:
    if re.fullmatch(r"[A-Za-z0-9()., -]+", text) is None:
        return False
    connectors = {"and", "or", "through", "to", "und", "oder", "bis"}
    for token in re.findall(r"[A-Za-z0-9]+", text):
        if (
            token.lower() in connectors
            or len(token) == 1
            or any(character.isdigit() for character in token)
            or re.fullmatch(r"[IVXLCDM]+", token) is not None
        ):
            continue
        return False
    return True


def _source_rounding_obligations(
    source_text: str,
    *,
    branches: Sequence[SourceStructureBranch],
    active_branches: Sequence[SourceStructureBranch],
    deferred_paths: set[tuple[str, ...]],
) -> tuple[tuple[SourceStructureBranch, str], ...]:
    obligations: list[tuple[SourceStructureBranch, str]] = []
    source_clauses = tuple(_source_clause_spans(source_text, branches=branches))
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
        matched_language = match.group(0)
        clause_start, clause_end, clause_text = next(
            (
                (start, end, text)
                for start, end, text in source_clauses
                if start <= match.start() and match.end() <= end
            ),
            (match.start(), match.end(), matched_language),
        )
        if _NEAREST_ROUNDING_LANGUAGE.search(matched_language):
            direction = "nearest"
        elif _UP_ROUNDING_LANGUAGE.search(matched_language):
            direction = "upward"
        else:
            direction = "downward"
        obligation_branch = SourceStructureBranch(
            owner.path,
            "rounding-clause",
            owner.label,
            clause_text,
            clause_start,
            clause_end,
        )
        obligations.append((obligation_branch, direction))
    return tuple(obligations)


def _rounding_source_formula_branches(
    rounding_branch: SourceStructureBranch,
    *,
    formula_branches: Sequence[SourceStructureBranch],
) -> tuple[SourceStructureBranch, ...]:
    containing = [
        branch
        for branch in formula_branches
        if branch.path == rounding_branch.path
        and branch.start <= rounding_branch.start
        and rounding_branch.end <= branch.end
    ]
    if containing:
        return (
            min(
                containing,
                key=lambda branch: branch.end - branch.start,
            ),
        )
    if not _rounding_text_refers_to_result(rounding_branch.text):
        return ()
    preceding = tuple(
        branch
        for branch in formula_branches
        if branch.end <= rounding_branch.start
        and _same_top_level_source_path(branch.path, rounding_branch.path)
    )
    if not preceding:
        return ()
    if re.match(
        r"\s*der\s+sich\s+ergebende\s+steuerbetrag\b",
        _strip_source_clause_marker(rounding_branch.text),
        flags=re.IGNORECASE,
    ):
        interval_branches = tuple(
            branch
            for branch in preceding
            if re.search(
                r"\b(?:bis|von|ab|unter|über|between|from|through|"
                r"up\s+to|above|below)\b",
                branch.text.split(":", 1)[0],
                flags=re.IGNORECASE,
            )
        )
        if interval_branches:
            return interval_branches
    return (max(preceding, key=lambda branch: branch.end),)


def _unwitnessed_exception_branches(
    exception_branches: Sequence[SourceStructureBranch],
    *,
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
    toggled_exception_selectors: set[_ExceptionWitness],
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> tuple[SourceStructureBranch, ...]:
    candidate_witnesses = {
        branch: _exception_witnesses_for_branch(
            branch,
            principal_rule_paths=principal_rule_paths,
            toggled_exception_selectors=toggled_exception_selectors,
            extract_numeric_occurrences=extract_numeric_occurrences,
        )
        for branch in exception_branches
    }
    return _unmatched_evidence_obligations(candidate_witnesses)


def _unconditional_nonapplicability_witnesses(
    branch: SourceStructureBranch,
    *,
    corpus_citation_path: str,
    principal_rules: dict[str, dict[str, Any]],
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    formula_environment: dict[str, Any],
) -> set[tuple[str, int]]:
    """Return unique executable-false evidence for an unconditional rule."""

    branch_text = _collapse_text(branch.text).lower()
    witnesses: set[tuple[str, int]] = set()
    for rule_name in _rules_covering_branch(branch, principal_rule_paths):
        rule = principal_rules[rule_name]
        if str(rule.get("dtype", "")).strip().lower() not in {
            "bool",
            "judgment",
        }:
            continue
        if not any(
            citation_path == corpus_citation_path
            and (excerpt_text := _collapse_text(excerpt).lower())
            and excerpt_text in branch_text
            for citation_path, excerpt in _rule_source_excerpts(rule)
        ):
            continue
        for case_index, case in enumerate(asserted_by_rule.get(rule_name, ())):
            asserted = _test_case_asserted_output_value(case, rule_name)
            if _boolean_value(asserted) is not False:
                continue
            dependencies = _case_asserted_dependency_environment(
                principal_rules,
                case,
                formula_environment=formula_environment,
            )
            execution = _case_formula_execution(
                rule,
                case,
                formula_environment=formula_environment,
                dependency_environment=dependencies,
            )
            if execution is None:
                continue
            runtime = _formula_execution_runtime_value(execution)
            if _boolean_value(runtime) is False and _formula_runtime_values_equal(
                runtime,
                asserted,
            ):
                witnesses.add((rule_name, case_index))
    return witnesses


def _exception_witnesses_for_branch(
    branch: SourceStructureBranch,
    *,
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
    toggled_exception_selectors: set[_ExceptionWitness],
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> set[_ExceptionWitness]:
    affecting_rules = {
        rule_name
        for rule_name, paths in principal_rule_paths.items()
        if not branch.path
        or any(
            path == branch.path[: len(path)] or branch.path == path[: len(branch.path)]
            for path in paths
            if path
        )
    }
    requirement = _source_exception_effect_requirement(branch.text)
    condition_text = _source_exception_condition_text(branch.text)
    numeric_interval = _formula_interval_from_text(
        authoritative_numeric_recall_text(branch.text),
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    return {
        witness
        for witness in toggled_exception_selectors
        if witness.rule_name in affecting_rules
        and (
            _numeric_exception_witness_matches_source(
                branch,
                witness,
                extract_numeric_occurrences=extract_numeric_occurrences,
            )
            if witness.numeric_transition is not None
            else (
                numeric_interval is None
                and witness.active_value
                == _source_exception_selector_active_value(
                    condition_text,
                    witness.selector_name,
                )
            )
        )
        and _source_exception_selector_is_relevant(
            condition_text,
            witness.selector_name,
        )
        and _exception_witness_satisfies_requirement(witness, requirement)
    }


def _source_exception_condition_text(text: str) -> str:
    """Return the condition region without the ordinary claim subject."""

    clause = _strip_source_clause_marker(text)
    notwithstanding_tail = _louisiana_notwithstanding_reference_tail(clause)
    if notwithstanding_tail is not None:
        return notwithstanding_tail
    marker = next(iter(_source_exception_or_applicability_matches(clause)), None)
    if marker is None:
        return clause
    suffix = clause[marker.start() :]
    if _source_qualification_exception_idiom(clause, marker):
        return clause
    condition_cue = re.search(
        r"\b(?:vorausgesetzt\s*,?\s+dass|"
        r"unter\s+der\s+voraussetzung\s*,?\s+dass|"
        r"wenn|falls|sofern|soweit|when|if|bei|ohne|mangels)\b",
        suffix,
        flags=re.IGNORECASE,
    )
    if condition_cue is not None:
        return suffix[condition_cue.start() :]
    prefix = clause[: marker.start()]
    preposed = re.match(
        r"\s*(?P<condition>(?:"
        r"(?:vorausgesetzt\s*,?\s+dass|"
        r"unter\s+der\s+voraussetzung\s*,?\s+dass|"
        r"wenn|falls|sofern|soweit|when|if)\b[^,;]*|"
        r"(?:bei|ohne|mangels)\b[^,;]*|"
        r"im\s+falle\b[^,;]*"
        r"))\s*,?\s*$",
        prefix,
        flags=re.IGNORECASE,
    )
    if preposed is not None:
        return preposed.group("condition")
    return suffix


def _source_qualification_exception_idiom(
    text: str,
    marker: re.Match[str],
) -> bool:
    """Recognize qualifications preserved except for one stated criterion."""

    return bool(
        marker.group(0).strip().lower().startswith("except")
        and re.search(
            r"\b(?:qualifications?|requirements?|conditions?)\b",
            text[: marker.start()],
            flags=re.IGNORECASE,
        )
        and re.search(
            r"\b(?:eligible|qualif(?:y|ied|ication))\b",
            text[marker.end() :],
            flags=re.IGNORECASE,
        )
    )


def _source_exception_effect_requirement(text: str) -> str:
    """Classify the minimum isolated effect expressly required by one clause."""

    collapsed = _collapse_text(text)
    numeric_zero = r"(?<![\d,.])0+(?:[,.]0+)?(?![\d,.])"
    if re.search(
        rf"\b(?:"
        rf"(?:beträgt|ist|wird)[^.;]{{0,120}}\b(?:null|{numeric_zero})|"
        rf"auf\s+(?:null|{numeric_zero})|"
        rf"(?:equals?|is|becomes?|set\s+to)[^.;]{{0,120}}"
        rf"\b(?:zero|{numeric_zero})"
        rf")\b",
        collapsed,
        flags=re.IGNORECASE,
    ):
        return "zero"
    notwithstanding_requirement = _notwithstanding_exemption_effect_requirement(
        collapsed
    )
    if notwithstanding_requirement is not None:
        return notwithstanding_requirement
    if _exception_reverses_negative_proposition(collapsed):
        return "enable"
    condition = next(
        iter(_source_exception_or_applicability_matches(collapsed)),
        None,
    )
    if condition is not None:
        cue = condition.group(0).strip().lower()
        proposition = collapsed[: condition.start()]
        negative_proposition = _source_negative_effect_matches(proposition)
        positive_proposition = _source_positive_effect_matches(proposition)
        if (
            not negative_proposition
            and positive_proposition
            and re.fullmatch(
                r"(?:if|when|wenn|falls|sofern|soweit|"
                r"vorausgesetzt\s*,?\s*dass|"
                r"unter\s+der\s+voraussetzung\s*,?\s+dass)",
                cue,
                flags=re.IGNORECASE,
            )
        ):
            return "enable"
        if (
            negative_proposition
            and not positive_proposition
            and re.fullmatch(
                r"(?:if|when|wenn|falls|sofern|soweit|"
                r"vorausgesetzt\s*,?\s*dass|"
                r"unter\s+der\s+voraussetzung\s*,?\s+dass)",
                cue,
                flags=re.IGNORECASE,
            )
        ):
            return "exclude"
        if (
            not negative_proposition
            and not positive_proposition
            and re.fullmatch(
                r"(?:if|when|wenn|falls|sofern|soweit|"
                r"vorausgesetzt\s*,?\s*dass|"
                r"unter\s+der\s+voraussetzung\s*,?\s+dass)",
                cue,
                flags=re.IGNORECASE,
            )
        ):
            tail = collapsed[condition.end() :]
            positive_tail = _source_positive_effect_matches(tail)
            negative_tail = _source_negative_effect_matches(tail)
            latest_positive = max(
                (match.start() for match in positive_tail),
                default=-1,
            )
            latest_negative = max(
                (match.start() for match in negative_tail),
                default=-1,
            )
            if latest_positive > latest_negative:
                return "enable"
            if latest_negative > latest_positive:
                return "exclude"
        if _source_qualification_exception_idiom(collapsed, condition):
            return "enable"
    if re.search(
        r"\b(?:"
        r"gilt\b[^.;]{0,80}\bnicht|"
        r"findet\s+keine\s+anwendung|"
        r"keine?\s+berechtigung|"
        r"keinen?\s+anspruch|"
        r"nicht\s+berechtigt|"
        r"ausgeschlossen|"
        r"ausgenommen|"
        r"außer|"
        r"ausser|"
        r"es\s+sei\s+denn|"
        r"soweit\s+nicht|"
        r"jedoch\s+nicht|"
        r"shall\s+not\s+apply|"
        r"does\s+not\s+apply|"
        r"not\s+eligible|"
        r"ineligible|"
        r"excluded|"
        r"unless|"
        r"except"
        r")\b",
        collapsed,
        flags=re.IGNORECASE,
    ):
        return "exclude"
    return "change"


def _source_states_negative_refund_effect(text: str) -> bool:
    """Recognize express prohibitions on a refund or refundable excess."""

    return bool(
        re.search(
            r"\b(?:"
            r"no\s+refund\b|"
            r"refund\w*\b[^.;]{0,48}\b(?:disallowed|forbidden|prohibited|"
            r"barred|not\s+(?:allowed|permitted))\b|"
            r"refund\w*\b[^.;]{0,48}\b(?:shall|may|does?|is|are)\s+not\s+"
            r"(?:be\s+)?(?:allowed|paid|permitted|refund\w*)\b|"
            r"(?:shall|may|does?)\s+not\s+(?:be\s+)?(?:paid|refund\w*)\b|"
            r"not\s+refundable\b"
            r")",
            text,
            flags=re.IGNORECASE,
        )
    )


def _notwithstanding_exemption_effect_requirement(text: str) -> str | None:
    """Recognize the direct certification duty in an exemption override."""

    match = re.search(
        r"\bnotwithstanding\s+that\s+qualified\s+public\s+housing\s+agencies\s+"
        r"are\s+exempt\s+under\s+subparagraph\s+\(A\)\s+from\s+the\s+"
        r"requirement\s+under\s+this\s+section\s+to\s+prepare\s+and\s+submit\s+"
        r"an\s+annual\s+public\s+housing\s+plan,\s*"
        r"(?P<effect>[^.;]{0,640})",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    if re.match(
        r"^each\s+qualified\s+public\s+housing\s+agency\s+shall,\s*"
        r"on\s+an\s+annual\s+basis,\s*make\s+the\s+certification\s+"
        r"described\s+in\s+paragraph\s+\(16\)\s+of\s+subsection\s+\(d\),\s*"
        r"except\s+that\b",
        match.group("effect"),
        flags=re.IGNORECASE,
    ):
        return "enable"
    return None


def _source_positive_effect_matches(text: str) -> tuple[re.Match[str], ...]:
    return tuple(
        re.finditer(
            r"\bto\s+qualify\b|"
            r"\b(?:shall|is|are|will|may)\s+be\s+"
            r"(?:eligible|qualified|allowed|entitled)\b|"
            r"\b(?:is|are)\s+(?:eligible|qualified|allowed|entitled)\b|"
            r"\b(?:claim|credit|benefit|provision)\s+applies\b|"
            r"\b(?:ist|sind|wird|werden)\s+(?:\w+\s+){0,2}berechtigt\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _source_negative_effect_matches(text: str) -> tuple[re.Match[str], ...]:
    return tuple(
        re.finditer(
            r"\b(?:shall|does|is|are)\s+not\s+(?:apply|eligible|qualified|"
            r"allowed|entitled)\b|"
            r"\b(?:ineligible|excluded|disqualified|unqualified)\b|"
            r"\bnicht\s+berechtigt\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _exception_reverses_negative_proposition(text: str) -> bool:
    reversal = re.search(
        r"\b(?:außer|ausser|es\s+sei\s+denn|unless|except)\b",
        text,
        flags=re.IGNORECASE,
    )
    if reversal is None:
        return False
    proposition = text[: reversal.start()]
    return (
        re.search(
            r"\b(?:besteht|gilt)\b[^.;]{0,80}\bnicht\b|"
            r"\bfindet\s+keine\s+anwendung\b|"
            r"\b(?:does|shall)\s+not\s+apply\b|"
            r"\b(?:is|are)\s+not\s+(?:eligible|qualified|entitled)\b|"
            r"\b(?:ineligible|excluded|disqualified|unqualified)\b|"
            r"\bkein(?:e|en|em|er|es)?\s+(?:anspruch|berechtigung)\b",
            proposition,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _source_exception_selector_is_relevant(text: str, name: str) -> bool:
    """Reject formula toggles with no semantic link to the source exception."""

    normalized_name = _normalized_selector_name(name)
    collapsed = _collapse_text(text).lower()
    if _source_selector_concept_matches(collapsed, normalized_name):
        return True
    return any(
        len(token) >= 4
        and token not in _SOURCE_SELECTOR_TOKEN_STOPWORDS
        and re.search(rf"\b{re.escape(token)}\w*", collapsed) is not None
        for token in normalized_name.split("_")
    )


def _source_exception_selector_active_value(text: str, name: str) -> bool:
    """Resolve exception orientation from source wording, not formula output."""

    normalized_name = _normalized_selector_name(name)
    source_polarity = _source_selector_concept_polarity(
        _collapse_text(text).lower(),
        normalized_name,
    )
    if source_polarity is not None:
        selector_polarity = (
            -1 if _selector_identifier_negation_count(normalized_name) % 2 else 1
        )
        return selector_polarity == source_polarity
    return _exception_selector_semantic_active_value(normalized_name)


def _numeric_exception_witness_matches_source(
    branch: SourceStructureBranch,
    witness: _ExceptionWitness,
    *,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> bool:
    source_text = _collapse_text(_strip_source_clause_marker(branch.text)).lower()
    matched_source_relation = False
    for left_name, relation, right_name in witness.relational_transitions:
        relation_indices = _source_relational_exception_match_indices(
            source_text,
            left_name=left_name,
            relation=relation,
            right_name=right_name,
        )
        if not relation_indices:
            continue
        matched_source_relation = True
        effect_matches = tuple(
            _relational_exception_witness_has_source_effect(
                source_text,
                witness,
                relation_index=relation_index,
            )
            for relation_index in relation_indices
        )
        if len(set(effect_matches)) > 1:
            return False
        if effect_matches[0]:
            return True
    if matched_source_relation:
        return False
    transition = witness.numeric_transition
    if transition is None:
        return False
    interval = _formula_interval_from_text(
        authoritative_numeric_recall_text(branch.text),
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    if interval is None:
        return False
    ordinary_value, exception_value = transition
    return not _interval_contains(interval, ordinary_value) and _interval_contains(
        interval, exception_value
    )


def _source_relational_exception_matches(
    text: str,
    *,
    left_name: str,
    relation: str,
    right_name: str,
) -> bool:
    """Match a reached formula relation to a source-stated relational trigger."""

    return bool(
        _source_relational_exception_match_indices(
            text,
            left_name=left_name,
            relation=relation,
            right_name=right_name,
        )
    )


def _source_relational_exception_match_indices(
    text: str,
    *,
    left_name: str,
    relation: str,
    right_name: str,
) -> tuple[int, ...]:
    """Return source relation indices matching one reached formula transition."""

    if relation == "<":
        left_name, relation, right_name = right_name, ">", left_name
    elif relation == "<=":
        left_name, relation, right_name = right_name, ">=", left_name
    collapsed = _collapse_text(text).lower()
    left_matches = _source_relational_operand_matches(collapsed, left_name)
    right_matches = _source_relational_operand_matches(collapsed, right_name)
    relation_matches = tuple(
        re.finditer(
            r"\b(?P<inclusive>exceeds?\s+or\s+equals?|"
            r"is\s+(?:greater|more)\s+than\s+or\s+equal\s+to)\b|"
            r"\b(?P<strict>exceeds?|is\s+(?:greater|more)\s+than|übersteigt)\b",
            collapsed,
        )
    )
    matched_indices: list[int] = []
    for index, relation_match in enumerate(relation_matches):
        source_relation = ">=" if relation_match.group("inclusive") else ">"
        if source_relation != relation:
            continue
        left_bound, right_bound = _source_relation_operand_bounds(
            collapsed,
            relation_matches,
            index,
        )
        if any(
            left_bound <= left.start()
            and left.end() <= relation_match.start()
            and relation_match.end() <= right.start()
            and right.end() <= right_bound
            and relation_match.start() - left.end() <= 240
            and right.start() - relation_match.end() <= 240
            for left in left_matches
            for right in right_matches
        ):
            matched_indices.append(index)
    return tuple(matched_indices)


def _source_relation_operand_bounds(
    text: str,
    relation_matches: tuple[re.Match[str], ...],
    index: int,
) -> tuple[int, int]:
    """Bound relation operands to their immediate clause or conjunct."""

    relation_match = relation_matches[index]
    left_bound = max(
        (
            match.end()
            for match in re.finditer(r"[.;:]", text[: relation_match.start()])
        ),
        default=0,
    )
    right_delimiter = re.search(r"[.;:]", text[relation_match.end() :])
    right_bound = (
        relation_match.end() + right_delimiter.start()
        if right_delimiter is not None
        else len(text)
    )
    if index:
        previous = relation_matches[index - 1]
        separator = _source_relation_clause_separator(
            text[previous.end() : relation_match.start()]
        )
        if separator is not None:
            left_bound = max(
                left_bound,
                previous.end() + separator.end(),
            )
    if index + 1 < len(relation_matches):
        following = relation_matches[index + 1]
        separator = _source_relation_clause_separator(
            text[relation_match.end() : following.start()],
        )
        if separator is not None:
            right_bound = min(
                right_bound,
                relation_match.end() + separator.start(),
            )
    return left_bound, right_bound


def _source_relation_clause_separator(text: str) -> re.Match[str] | None:
    """Choose the clause break without splitting compound relation operands."""

    for pattern in (
        r"\b(?:although|but|though|while|whereas|yet)\b",
        r",",
        r"\b(?:and|or|und|oder)\b",
    ):
        matches = tuple(re.finditer(pattern, text))
        if matches:
            return matches[-1]
    return None


def _relational_exception_witness_has_source_effect(
    text: str,
    witness: _ExceptionWitness,
    *,
    relation_index: int,
) -> bool:
    """Require the relation-active side to have the source-directed effect."""

    effect_text = _source_relational_effect_text(text, relation_index=relation_index)
    if _source_relational_exception_reverses_negative_refund(
        text,
        relation_index=relation_index,
    ):
        requirement = "enable"
    elif _source_states_negative_refund_effect(effect_text):
        requirement = "exclude"
    else:
        requirement = _source_exception_effect_requirement(effect_text)
    if requirement == "zero":
        return witness.zeroes
    if requirement == "exclude":
        return witness.blocks or witness.zeroes
    if requirement == "enable":
        return not witness.blocks and not witness.zeroes
    return bool(
        re.search(
            r"\b(?:excess\w*|overpayment\w*|refund\w*)\b",
            effect_text,
            flags=re.IGNORECASE,
        )
        and not witness.blocks
        and not witness.zeroes
    )


def _source_relational_exception_reverses_negative_refund(
    text: str,
    *,
    relation_index: int,
) -> bool:
    """Recognize an exception condition that lifts a stated refund prohibition."""

    relation_matches = tuple(
        re.finditer(
            r"\b(?:exceeds?\s+or\s+equals?|"
            r"is\s+(?:greater|more)\s+than(?:\s+or\s+equal\s+to)?|"
            r"exceeds?|übersteigt)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    if not 0 <= relation_index < len(relation_matches):
        return False
    condition_prefix = text[: relation_matches[relation_index].start()]
    condition_cues = tuple(
        re.finditer(
            r"\b(?P<reversal>unless|except(?:\s+(?:if|when))?)\b|"
            r"\b(?P<ordinary>falls|if|sofern|soweit|wenn|when)\b",
            condition_prefix,
            flags=re.IGNORECASE,
        )
    )
    if not condition_cues or condition_cues[-1].group("reversal") is None:
        return False
    reversal = condition_cues[-1]
    return _source_states_negative_refund_effect(condition_prefix[: reversal.start()])


def _source_relational_effect_text(text: str, *, relation_index: int) -> str:
    """Return the effect associated with one matched relational condition."""

    relation_matches = tuple(
        re.finditer(
            r"\b(?:exceeds?\s+or\s+equals?|"
            r"is\s+(?:greater|more)\s+than(?:\s+or\s+equal\s+to)?|"
            r"exceeds?|übersteigt)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    if not 0 <= relation_index < len(relation_matches):
        return text
    relation_match = relation_matches[relation_index]
    condition_cues = tuple(
        re.finditer(
            r"\b(?:except\s+(?:if|when)|unless|falls|if|sofern|soweit|wenn|when)\b",
            text[: relation_match.start()],
            flags=re.IGNORECASE,
        )
    )
    if condition_cues:
        condition_cue = condition_cues[-1]
        proposition_start = 0
        if relation_index:
            strong_boundary = tuple(
                re.finditer(r"[.;:]", text[: condition_cue.start()])
            )
            if strong_boundary:
                proposition_start = strong_boundary[-1].end()
            elif relation_matches[relation_index - 1].end() < condition_cue.start():
                between_start = relation_matches[relation_index - 1].end()
                coordinate_boundaries = tuple(
                    re.finditer(
                        r"\b(?:although|but|though|while|whereas|yet)\b",
                        text[between_start : condition_cue.start()],
                        flags=re.IGNORECASE,
                    )
                )
                if not coordinate_boundaries:
                    coordinate_boundaries = tuple(
                        re.finditer(
                            r"\b(?:and|or)\b(?=[^,;:.]{0,120}"
                            r"\b(?:excess\w*|overpayment\w*|refund\w*)\b)",
                            text[between_start : condition_cue.start()],
                            flags=re.IGNORECASE,
                        )
                    )
                proposition_start = (
                    between_start + coordinate_boundaries[-1].end()
                    if coordinate_boundaries
                    else condition_cue.start()
                )
        proposition = text[proposition_start : condition_cue.start()].strip(" ,;:")
        if proposition and re.search(
            r"\b(?:excess\w*|overpayment\w*|refund\w*)\b",
            proposition,
            flags=re.IGNORECASE,
        ):
            return proposition
    tail_start = relation_match.end()
    tail = text[tail_start:]
    then_cue = re.search(r"\bthen\b", tail, flags=re.IGNORECASE)
    if then_cue is not None:
        return tail[then_cue.end() :]
    consequence = re.search(r"[,;:]", tail)
    if consequence is None:
        return tail
    effect = tail[consequence.end() :]
    secondary_boundary = re.search(r"[,;:]", effect)
    if secondary_boundary is not None and re.match(
        r"\s*(?:(?:and|or)\s+)?(?:if|when|unless|provided\b|no\b|without\b)",
        effect[: secondary_boundary.start()],
        flags=re.IGNORECASE,
    ):
        return effect[secondary_boundary.end() :]
    return effect


def _source_relational_operand_matches(
    text: str,
    name: str,
) -> tuple[re.Match[str], ...]:
    normalized = _normalized_selector_name(name)
    aliases = {
        "credit": r"\bcredit\b",
        "liability": r"\b(?:liabilit\w*|tax(?:es)?\s+(?:due|owed)|amount\s+of\s+(?:the\s+)?tax)\b",
    }
    matches = list(_source_selector_concept_matches(text, normalized))
    matches.extend(
        match
        for concept, pattern in aliases.items()
        for match in re.finditer(pattern, text)
        if concept in normalized
    )
    unique = {(match.start(), match.end(), match.group(0)): match for match in matches}
    return tuple(unique[key] for key in sorted(unique))


def _normalized_selector_name(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _source_selector_concept_matches(
    text: str,
    normalized_name: str,
) -> tuple[re.Match[str], ...]:
    patterns: list[str] = []
    concept_patterns = {
        "certificate": r"\b(?:certificate|bescheinigung|nachweis|zertifikat)\w*",
        "child": r"\b(?:child|kind(?:er)?)\w*",
        "kind": r"\bkind(?:er)?\w*",
        "condition": r"\b(?:condition|voraussetzung)\w*",
        "eligible": r"\b(?:eligible|berechtig|anspruch)\w*",
        "income": r"\b(?:income|einkommen)\w*",
        "penalty": r"\b(?:penalt(?:y|ies))\b",
        "qualified": r"\b(?:qualified|berechtig|anspruch)\w*",
        "exempt": r"\b(?:exempt|befrei)\w*",
        "exception": r"\b(?:exception|ausnahme|abweich|befrei)\w*",
        "barred": r"\b(?:barred|sperr)\w*",
        "excluded": r"\b(?:excluded|ausgeschlossen|ausgenommen)\w*",
        "remarried": r"\b(?:remarri|wiederverheirat)\w*",
        "waiver": r"\b(?:waiver|waived|verzicht)\w*",
        "surcharge": r"\b(?:surcharge|zuschlag)\w*",
        "status": r"\bstatus\w*",
    }
    for concept, pattern in concept_patterns.items():
        if concept in normalized_name:
            patterns.append(pattern)
    patterns.extend(
        rf"\b{re.escape(token)}\w*"
        for token in normalized_name.split("_")
        if len(token) >= 4 and token not in _SOURCE_SELECTOR_TOKEN_STOPWORDS
    )
    matches = {
        (match.start(), match.end(), match.group(0)): match
        for pattern in patterns
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    }
    return tuple(matches[key] for key in sorted(matches))


def _source_selector_concept_polarity(
    text: str,
    normalized_name: str,
) -> int | None:
    polarities: set[int] = set()
    for match in _source_selector_concept_matches(text, normalized_name):
        before = text[max(0, match.start() - 48) : match.start()]
        after = text[match.end() : min(len(text), match.end() + 48)]
        negative = bool(
            re.fullmatch(
                r"(?:ineligible|disqualified|unqualified)",
                match.group(0),
                flags=re.IGNORECASE,
            )
            or re.search(
                r"\b(?:kein(?:e|en|em|er|es)?|fehlend\w*|ohne|mangels|"
                r"no|without|lack(?:ing)?(?:\s+of)?|absence\s+of)"
                r"\b[^.;]{0,40}$|"
                r"\b(?:nicht|not)\s+(?:\w+\s+){0,1}$",
                before,
                flags=re.IGNORECASE,
            )
            or re.match(
                r"[^.;]{0,40}\b(?:fehlt|fehlen|nicht\s+(?:vorhanden|"
                r"gegeben)|liegt\s+nicht\s+vor|is\s+not\s+present|"
                r"is\s+absent)\b",
                after,
                flags=re.IGNORECASE,
            )
        )
        polarities.add(-1 if negative else 1)
    return next(iter(polarities)) if len(polarities) == 1 else None


def _selector_identifier_negation_count(normalized_name: str) -> int:
    count = sum(
        token
        in {
            "absent",
            "lacking",
            "missing",
            "no",
            "non",
            "not",
            "without",
        }
        for token in normalized_name.split("_")
    )
    count += sum(
        marker in normalized_name.split("_")
        for marker in {"disqualified", "ineligible", "unqualified"}
    )
    return count


def _exception_witness_satisfies_requirement(
    witness: _ExceptionWitness,
    requirement: str,
) -> bool:
    if requirement == "zero":
        return witness.zeroes
    if requirement == "enable":
        return (witness.boolean_effect and not witness.blocks) or (
            not witness.boolean_effect
        )
    if requirement == "exclude":
        return witness.blocks or not witness.boolean_effect
    return True


def _toggled_formula_boolean_selectors(
    principal_rules: dict[str, dict[str, Any]],
    *,
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    formula_environment: dict[str, Any],
) -> set[_ExceptionWitness]:
    toggled: set[_ExceptionWitness] = set()
    for rule_name, rule in principal_rules.items():
        selector_names = _rule_exception_selector_names(rule)
        if not selector_names:
            continue
        records = [
            (
                case,
                (
                    dependency_environment := _case_asserted_dependency_environment(
                        principal_rules,
                        case,
                        formula_environment=formula_environment,
                    )
                ),
                _case_boolean_selector_environment(
                    case,
                    selector_names,
                    dependency_environment=dependency_environment,
                ),
            )
            for case in asserted_by_rule.get(rule_name, ())
        ]
        for left_index, (
            left_case,
            left_dependencies,
            left_selectors,
        ) in enumerate(records):
            for (
                right_case,
                right_dependencies,
                right_selectors,
            ) in records[left_index + 1 :]:
                if _normalized_case_period(left_case) != _normalized_case_period(
                    right_case
                ) or not _cases_differ_by_one_input(
                    left_case,
                    right_case,
                ):
                    continue
                if set(left_selectors) != set(right_selectors):
                    continue
                changed_names = {
                    name
                    for name in left_selectors
                    if left_selectors[name] != right_selectors[name]
                }
                if len(changed_names) != 1:
                    continue
                selector_name = next(iter(changed_names))
                for active_value in (False, True):
                    witness = _exception_witness_for_case_pair(
                        rule_name,
                        rule,
                        selector_name=selector_name,
                        active_value=active_value,
                        left_case=left_case,
                        left_dependencies=left_dependencies,
                        left_selector_value=left_selectors[selector_name],
                        right_case=right_case,
                        right_dependencies=right_dependencies,
                        formula_environment=formula_environment,
                    )
                    if witness is not None:
                        toggled.add(witness)
    numeric_witnesses = _toggled_formula_numeric_selectors(
        principal_rules,
        asserted_by_rule=asserted_by_rule,
        formula_environment=formula_environment,
    )
    toggled.update(_composed_numeric_dependency_witnesses(toggled, numeric_witnesses))
    toggled.update(numeric_witnesses)
    return toggled


def _composed_numeric_dependency_witnesses(
    boolean_witnesses: Iterable[_ExceptionWitness],
    numeric_witnesses: Iterable[_ExceptionWitness],
) -> set[_ExceptionWitness]:
    activating_by_rule_and_pair: dict[
        tuple[str, tuple[int, ...]], list[_ExceptionWitness]
    ] = {}
    for witness in numeric_witnesses:
        if (
            witness.numeric_transition is None
            or not witness.boolean_effect
            or witness.blocks
            or not witness.case_pair_identity
        ):
            continue
        activating_by_rule_and_pair.setdefault(
            (witness.rule_name, witness.case_pair_identity), []
        ).append(witness)
    composed: set[_ExceptionWitness] = set()
    for witness in boolean_witnesses:
        if (
            witness.numeric_transition is not None
            or not witness.active_value
            or not witness.case_pair_identity
        ):
            continue
        for dependency_witness in activating_by_rule_and_pair.get(
            (witness.selector_name, witness.case_pair_identity), ()
        ):
            composed.add(
                _ExceptionWitness(
                    witness.rule_name,
                    dependency_witness.selector_name,
                    True,
                    witness.blocks,
                    witness.boolean_effect,
                    witness.zeroes,
                    dependency_witness.numeric_transition,
                    dependency_witness.relational_transitions,
                    witness.case_pair_identity,
                )
            )
    return composed


def _case_pair_identity(
    left_case: dict[str, Any],
    right_case: dict[str, Any],
) -> tuple[int, int]:
    return tuple(sorted((id(left_case), id(right_case))))


def _toggled_formula_numeric_selectors(
    principal_rules: dict[str, dict[str, Any]],
    *,
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    formula_environment: dict[str, Any],
) -> set[_ExceptionWitness]:
    witnesses: set[_ExceptionWitness] = set()
    for rule_name, rule in principal_rules.items():
        selector_names = _rule_numeric_selector_names(rule)
        for left_index, left_case in enumerate(asserted_by_rule.get(rule_name, ())):
            for right_case in asserted_by_rule[rule_name][left_index + 1 :]:
                if (
                    _normalized_case_period(left_case)
                    != _normalized_case_period(right_case)
                    or not _cases_differ_by_one_input(left_case, right_case)
                    or not _cases_have_same_output_keys(left_case, right_case)
                ):
                    continue
                left_inputs = left_case.get("input")
                right_inputs = right_case.get("input")
                if not isinstance(left_inputs, dict) or not isinstance(
                    right_inputs,
                    dict,
                ):
                    continue
                changed_key = next(
                    key
                    for key in left_inputs
                    if not _formula_runtime_values_equal(
                        left_inputs[key],
                        right_inputs[key],
                    )
                )
                left_value = left_inputs[changed_key]
                right_value = right_inputs[changed_key]
                if (
                    isinstance(left_value, bool)
                    or not isinstance(left_value, (int, float))
                    or isinstance(right_value, bool)
                    or not isinstance(right_value, (int, float))
                ):
                    continue
                changed_names = _input_key_names(changed_key) & selector_names
                if not changed_names:
                    continue
                left_dependencies = _case_asserted_dependency_environment(
                    principal_rules,
                    left_case,
                    formula_environment=formula_environment,
                )
                right_dependencies = _case_asserted_dependency_environment(
                    principal_rules,
                    right_case,
                    formula_environment=formula_environment,
                )
                stable_asserted_dependencies = (
                    _case_pair_stable_asserted_formula_dependencies(
                        rule_name,
                        rule,
                        principal_rules=principal_rules,
                        left_case=left_case,
                        right_case=right_case,
                        formula_environment=formula_environment,
                    )
                )
                left_dependencies = {
                    **stable_asserted_dependencies,
                    **left_dependencies,
                }
                right_dependencies = {
                    **stable_asserted_dependencies,
                    **right_dependencies,
                }
                left_execution = _case_formula_execution(
                    rule,
                    left_case,
                    formula_environment=formula_environment,
                    dependency_environment=left_dependencies,
                )
                right_execution = _case_formula_execution(
                    rule,
                    right_case,
                    formula_environment=formula_environment,
                    dependency_environment=right_dependencies,
                )
                if left_execution is None or right_execution is None:
                    continue
                (
                    left_to_right_relations,
                    right_to_left_relations,
                ) = _formula_relational_transitions(
                    left_execution,
                    right_execution,
                    left_case=left_case,
                    right_case=right_case,
                    left_dependencies=left_dependencies,
                    right_dependencies=right_dependencies,
                    formula_environment=formula_environment,
                    changed_names=changed_names,
                )
                positive_part_relations = _formula_positive_part_relation_descriptors(
                    left_execution.leaf
                ) | _formula_positive_part_relation_descriptors(right_execution.leaf)
                if (
                    (left_execution.trace or right_execution.trace)
                    and _formula_leaf_semantic_key(
                        left_execution.leaf,
                        formula_environment=left_execution.constant_environment,
                    )
                    == _formula_leaf_semantic_key(
                        right_execution.leaf,
                        formula_environment=right_execution.constant_environment,
                    )
                    and not positive_part_relations.intersection(
                        (*left_to_right_relations, *right_to_left_relations)
                    )
                ):
                    continue
                left_runtime = _formula_execution_runtime_value(left_execution)
                right_runtime = _formula_execution_runtime_value(right_execution)
                left_asserted = _test_case_asserted_output_value(
                    left_case,
                    rule_name,
                )
                right_asserted = _test_case_asserted_output_value(
                    right_case,
                    rule_name,
                )
                if not (
                    left_asserted is not _UNRESOLVED_CONDITION_VALUE
                    and right_asserted is not _UNRESOLVED_CONDITION_VALUE
                    and _formula_runtime_values_equal(
                        left_runtime,
                        left_asserted,
                    )
                    and _formula_runtime_values_equal(
                        right_runtime,
                        right_asserted,
                    )
                    and _exception_effect_changes(
                        left_runtime,
                        right_runtime,
                    )
                ):
                    continue
                for selector_name in changed_names:
                    if not (
                        (
                            _formula_execution_reaches_selector(
                                left_execution,
                                selector_name,
                            )
                            and _formula_execution_reaches_selector(
                                right_execution,
                                selector_name,
                            )
                        )
                        or any(
                            selector_name in {left_name, right_name}
                            for left_name, _relation, right_name in (
                                *left_to_right_relations,
                                *right_to_left_relations,
                            )
                        )
                    ):
                        continue
                    for (
                        ordinary_runtime,
                        exception_runtime,
                        ordinary_value,
                        exception_value,
                        relational_transitions,
                    ) in (
                        (
                            left_runtime,
                            right_runtime,
                            float(left_value),
                            float(right_value),
                            left_to_right_relations,
                        ),
                        (
                            right_runtime,
                            left_runtime,
                            float(right_value),
                            float(left_value),
                            right_to_left_relations,
                        ),
                    ):
                        witnesses.add(
                            _ExceptionWitness(
                                rule_name,
                                selector_name,
                                True,
                                _exception_effect_is_blocking(
                                    ordinary_runtime,
                                    exception_runtime,
                                ),
                                (
                                    _boolean_value(ordinary_runtime) is not None
                                    and _boolean_value(exception_runtime) is not None
                                ),
                                _exception_effect_is_zero(exception_runtime),
                                (ordinary_value, exception_value),
                                relational_transitions,
                                _case_pair_identity(left_case, right_case),
                            )
                        )
    return witnesses


def _case_pair_stable_asserted_formula_dependencies(
    rule_name: str,
    rule: dict[str, Any],
    *,
    principal_rules: dict[str, dict[str, Any]],
    left_case: dict[str, Any],
    right_case: dict[str, Any],
    formula_environment: dict[str, Any],
) -> dict[str, Any]:
    """Return reached upstream outputs asserted identically by both cases."""

    dependencies: dict[str, Any] = {}
    referenced_names = set(_FORMULA_IDENTIFIER.findall(_rule_formula_text(rule)))
    candidate_names = referenced_names & set(principal_rules) - {rule_name}
    left_runtime_dependencies = _case_dependency_environment(
        principal_rules,
        left_case,
        formula_environment=formula_environment,
        require_asserted_value=False,
    )
    right_runtime_dependencies = _case_dependency_environment(
        principal_rules,
        right_case,
        formula_environment=formula_environment,
        require_asserted_value=False,
    )
    for name in sorted(candidate_names):
        left_value = _test_case_asserted_output_value(left_case, name)
        right_value = _test_case_asserted_output_value(right_case, name)
        if (
            left_value is _UNRESOLVED_CONDITION_VALUE
            or right_value is _UNRESOLVED_CONDITION_VALUE
            or not _formula_runtime_values_equal(left_value, right_value)
            or name not in left_runtime_dependencies
            or name not in right_runtime_dependencies
            or not _formula_runtime_values_equal(
                left_value,
                left_runtime_dependencies[name],
            )
            or not _formula_runtime_values_equal(
                right_value,
                right_runtime_dependencies[name],
            )
        ):
            continue
        boolean_value = _boolean_value(left_value)
        dependencies[name] = boolean_value if boolean_value is not None else left_value
    return dependencies


def _formula_relational_transitions(
    left_execution: _FormulaExecution,
    right_execution: _FormulaExecution,
    *,
    left_case: dict[str, Any],
    right_case: dict[str, Any],
    left_dependencies: dict[str, Any],
    right_dependencies: dict[str, Any],
    formula_environment: dict[str, Any],
    changed_names: set[str],
) -> tuple[
    tuple[tuple[str, str, str], ...],
    tuple[tuple[str, str, str], ...],
]:
    """Return reached relations that cross false-to-true in each direction."""

    left_environment = _formula_case_runtime_environment(
        left_case,
        dependency_environment=left_dependencies,
        formula_environment=formula_environment,
    )
    right_environment = _formula_case_runtime_environment(
        right_case,
        dependency_environment=right_dependencies,
        formula_environment=formula_environment,
    )
    if left_environment is None or right_environment is None:
        return (), ()
    left_relations = _formula_execution_relational_values(
        left_execution,
        environment=left_environment,
        changed_names=changed_names,
    )
    right_relations = _formula_execution_relational_values(
        right_execution,
        environment=right_environment,
        changed_names=changed_names,
    )
    shared = set(left_relations) & set(right_relations)
    return (
        tuple(
            sorted(
                relation
                for relation in shared
                if left_relations[relation] is False
                and right_relations[relation] is True
            )
        ),
        tuple(
            sorted(
                relation
                for relation in shared
                if right_relations[relation] is False
                and left_relations[relation] is True
            )
        ),
    )


def _formula_case_runtime_environment(
    case: dict[str, Any],
    *,
    dependency_environment: dict[str, Any],
    formula_environment: dict[str, Any],
) -> dict[str, Any] | None:
    environment = _formula_environment_for_case(formula_environment, case)
    inputs = _case_input_formula_environment(case)
    if inputs is None:
        return None
    for name, value in (*dependency_environment.items(), *inputs.items()):
        if name in environment and not _formula_runtime_values_equal(
            environment[name], value
        ):
            return None
        environment[name] = value
    return environment


def _formula_execution_relational_values(
    execution: _FormulaExecution,
    *,
    environment: dict[str, Any],
    changed_names: set[str],
) -> dict[tuple[str, str, str], bool]:
    values: dict[tuple[str, str, str], bool] = {}
    texts = [selector for step in execution.trace for selector in step.selectors]
    texts.append(execution.leaf)
    for text in texts:
        expression = _parse_formula_expression(text)
        if expression is None:
            continue
        for left, relation, right in _formula_relational_expressions(expression):
            referenced_names = {
                node.id
                for operand in (left, right)
                for node in ast.walk(operand)
                if isinstance(node, ast.Name)
            }
            if not referenced_names & changed_names:
                continue
            left_name = _formula_operand_concept_name(left)
            right_name = _formula_operand_concept_name(right)
            if left_name is None or right_name is None:
                continue
            comparison = ast.Compare(
                left=left,
                ops=[
                    {
                        ">": ast.Gt(),
                        ">=": ast.GtE(),
                        "<": ast.Lt(),
                        "<=": ast.LtE(),
                    }[relation]
                ],
                comparators=[right],
            )
            value = _evaluate_condition_expression(comparison, environment)
            if isinstance(value, bool):
                values[(left_name, relation, right_name)] = value
    return values


def _formula_relational_expressions(
    expression: ast.AST,
) -> Iterable[tuple[ast.expr, str, ast.expr]]:
    for node in ast.walk(expression):
        if isinstance(node, ast.Compare):
            operands = (node.left, *node.comparators)
            for operator, left, right in zip(node.ops, operands, operands[1:]):
                relation = {
                    ast.Gt: ">",
                    ast.GtE: ">=",
                    ast.Lt: "<",
                    ast.LtE: "<=",
                }.get(type(operator))
                if relation is not None:
                    yield left, relation, right
    for left, right in _formula_positive_part_operands(expression):
        yield left, ">", right


def _formula_positive_part_operands(
    expression: ast.AST,
) -> Iterable[tuple[ast.expr, ast.expr]]:
    """Yield operands from reached ``max(0, left - right)`` forms."""

    for node in ast.walk(expression):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "max"
            and len(node.args) == 2
            and not node.keywords
        ):
            continue
        zero_index = next(
            (
                index
                for index, argument in enumerate(node.args)
                if _formula_ast_numeric_literal(argument) == 0
            ),
            None,
        )
        if zero_index is None:
            continue
        positive_part = node.args[1 - zero_index]
        if isinstance(positive_part, ast.BinOp) and isinstance(
            positive_part.op, ast.Sub
        ):
            yield positive_part.left, positive_part.right


def _formula_positive_part_relation_descriptors(
    text: str,
) -> set[tuple[str, str, str]]:
    expression = _parse_formula_expression(text)
    if expression is None:
        return set()
    descriptors: set[tuple[str, str, str]] = set()
    for left, right in _formula_positive_part_operands(expression):
        left_name = _formula_operand_concept_name(left)
        right_name = _formula_operand_concept_name(right)
        if left_name is not None and right_name is not None:
            descriptors.add((left_name, ">", right_name))
    return descriptors


def _formula_ast_numeric_literal(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -float(node.operand.value)
    return None


def _formula_operand_concept_name(node: ast.AST) -> str | None:
    names = sorted(
        {
            candidate.id
            for candidate in ast.walk(node)
            if isinstance(candidate, ast.Name)
        }
    )
    return "_".join(names) if names else None


def _exception_witness_for_case_pair(
    rule_name: str,
    rule: dict[str, Any],
    *,
    selector_name: str,
    active_value: bool,
    left_case: dict[str, Any],
    left_dependencies: dict[str, Any],
    left_selector_value: bool,
    right_case: dict[str, Any],
    right_dependencies: dict[str, Any],
    formula_environment: dict[str, Any],
) -> _ExceptionWitness | None:
    if left_selector_value == active_value:
        exception_case = left_case
        exception_dependencies = left_dependencies
        ordinary_case = right_case
        ordinary_dependencies = right_dependencies
    else:
        ordinary_case = left_case
        ordinary_dependencies = left_dependencies
        exception_case = right_case
        exception_dependencies = right_dependencies
    ordinary_execution = _case_formula_execution(
        rule,
        ordinary_case,
        formula_environment=formula_environment,
        dependency_environment=ordinary_dependencies,
    )
    exception_execution = _case_formula_execution(
        rule,
        exception_case,
        formula_environment=formula_environment,
        dependency_environment=exception_dependencies,
    )
    counterfactual_execution = _case_formula_execution_with_boolean_selector(
        rule,
        ordinary_case,
        selector_name=selector_name,
        selector_value=active_value,
        formula_environment=formula_environment,
        dependency_environment=ordinary_dependencies,
    )
    if (
        ordinary_execution is None
        or exception_execution is None
        or counterfactual_execution is None
    ):
        return None
    ordinary_asserted = _test_case_asserted_output_value(
        ordinary_case,
        rule_name,
    )
    exception_asserted = _test_case_asserted_output_value(
        exception_case,
        rule_name,
    )
    if (
        ordinary_asserted is _UNRESOLVED_CONDITION_VALUE
        or exception_asserted is _UNRESOLVED_CONDITION_VALUE
        or not _exception_effect_changes(
            ordinary_asserted,
            exception_asserted,
        )
    ):
        return None
    ordinary_runtime = _formula_execution_runtime_value(ordinary_execution)
    exception_runtime = _formula_execution_runtime_value(exception_execution)
    counterfactual_runtime = _formula_execution_runtime_value(counterfactual_execution)
    if not (
        all(
            _formula_execution_reaches_selector(execution, selector_name)
            for execution in (
                ordinary_execution,
                exception_execution,
                counterfactual_execution,
            )
        )
        and _formula_runtime_values_equal(
            ordinary_runtime,
            ordinary_asserted,
        )
        and _formula_runtime_values_equal(
            exception_runtime,
            exception_asserted,
        )
        and _formula_runtime_values_equal(
            counterfactual_runtime,
            exception_runtime,
        )
        and _exception_effect_changes(
            ordinary_runtime,
            counterfactual_runtime,
        )
    ):
        return None
    return _ExceptionWitness(
        rule_name,
        selector_name,
        active_value,
        _exception_effect_is_blocking(
            ordinary_runtime,
            counterfactual_runtime,
        ),
        (
            _boolean_value(ordinary_runtime) is not None
            and _boolean_value(counterfactual_runtime) is not None
        ),
        _exception_effect_is_zero(counterfactual_runtime),
        None,
        case_pair_identity=_case_pair_identity(left_case, right_case),
    )


def _cases_differ_by_one_input(
    left_case: dict[str, Any],
    right_case: dict[str, Any],
) -> bool:
    left_inputs = left_case.get("input")
    right_inputs = right_case.get("input")
    if not isinstance(left_inputs, dict) or not isinstance(right_inputs, dict):
        return False
    if set(left_inputs) != set(right_inputs):
        return False
    return (
        sum(
            not _formula_runtime_values_equal(
                left_inputs[key],
                right_inputs[key],
            )
            for key in left_inputs
        )
        == 1
    )


def _cases_have_same_output_keys(
    left_case: dict[str, Any],
    right_case: dict[str, Any],
) -> bool:
    left_outputs = left_case.get("output")
    right_outputs = right_case.get("output")
    return (
        isinstance(left_outputs, dict)
        and isinstance(right_outputs, dict)
        and set(left_outputs) == set(right_outputs)
    )


def _exception_effect_is_zero(value: Any) -> bool:
    number = _rulespec_runtime_decimal(value)
    return number is not None and math.isclose(float(number), 0.0, abs_tol=1e-12)


def _case_formula_execution_with_boolean_selector(
    rule: dict[str, Any],
    case: dict[str, Any],
    *,
    selector_name: str,
    selector_value: bool,
    formula_environment: dict[str, Any],
    dependency_environment: dict[str, Any],
) -> _FormulaExecution | None:
    dependencies = dict(dependency_environment)
    candidate_case = case
    if selector_name in dependencies:
        dependencies[selector_name] = selector_value
    else:
        inputs = case.get("input")
        if not isinstance(inputs, dict):
            return None
        matching_keys = [
            key for key in inputs if selector_name in _input_key_names(key)
        ]
        if len(matching_keys) != 1:
            return None
        candidate_inputs = dict(inputs)
        candidate_inputs[matching_keys[0]] = selector_value
        candidate_case = dict(case)
        candidate_case["input"] = candidate_inputs
    return _case_formula_execution(
        rule,
        candidate_case,
        formula_environment=formula_environment,
        dependency_environment=dependencies,
    )


def _formula_execution_runtime_value(execution: _FormulaExecution) -> Any:
    if execution.evaluated_value is not None:
        value_type, raw_value = execution.evaluated_value
        if value_type == "Decimal" and raw_value.startswith("Decimal('"):
            with contextlib.suppress(InvalidOperation, ValueError):
                return Decimal(raw_value[9:-2])
        with contextlib.suppress(SyntaxError, ValueError):
            return ast.literal_eval(raw_value)
    leaf_boolean = _boolean_value(execution.leaf.strip())
    if leaf_boolean is not None:
        return leaf_boolean
    return _UNRESOLVED_CONDITION_VALUE


def _exception_effect_changes(ordinary: Any, exception: Any) -> bool:
    if (
        ordinary is _UNRESOLVED_CONDITION_VALUE
        or exception is _UNRESOLVED_CONDITION_VALUE
    ):
        return False
    return not _formula_runtime_values_equal(ordinary, exception)


def _exception_effect_is_blocking(ordinary: Any, blocking: Any) -> bool:
    if (
        ordinary is _UNRESOLVED_CONDITION_VALUE
        or blocking is _UNRESOLVED_CONDITION_VALUE
    ):
        return False
    ordinary_boolean = _boolean_value(ordinary)
    blocking_boolean = _boolean_value(blocking)
    if ordinary_boolean is not None or blocking_boolean is not None:
        return ordinary_boolean is True and blocking_boolean is False
    if (
        _rulespec_runtime_decimal(ordinary) is not None
        and _rulespec_runtime_decimal(blocking) is not None
    ):
        return float(blocking) < float(ordinary)
    return False


def _case_boolean_selector_environment(
    case: dict[str, Any],
    selector_names: set[str],
    *,
    dependency_environment: dict[str, Any],
) -> dict[str, bool]:
    values: dict[str, bool] = {}
    inputs = case.get("input")
    if not isinstance(inputs, dict):
        return values
    for key, value in inputs.items():
        boolean = _boolean_value(value)
        if boolean is None:
            continue
        for name in _input_key_names(key) & selector_names:
            if name in values and values[name] != boolean:
                return {}
            values[name] = boolean
    for name, value in dependency_environment.items():
        if name not in selector_names:
            continue
        boolean = _boolean_value(value)
        if boolean is None:
            continue
        if name in values and values[name] != boolean:
            return {}
        values[name] = boolean
    return values


def _formula_execution_effect_signature(
    execution: _FormulaExecution,
) -> tuple[str, Any]:
    if execution.evaluated_value is not None:
        runtime_value = _formula_execution_runtime_value(execution)
        numeric_value = _rulespec_runtime_decimal(runtime_value)
        if numeric_value is not None:
            return "evaluated-number", numeric_value
        return "evaluated", execution.evaluated_value
    return "leaf", _collapse_text(execution.leaf).lower()


def _test_case_asserted_output_value(
    case: dict[str, Any],
    rule_name: str,
) -> Any:
    outputs = case.get("output")
    if not isinstance(outputs, dict):
        return _UNRESOLVED_CONDITION_VALUE
    values = [
        value
        for key, value in outputs.items()
        if str(key).rsplit("#", 1)[-1] == rule_name
    ]
    if len(values) != 1:
        return _UNRESOLVED_CONDITION_VALUE
    return values[0]


def _formula_execution_reaches_selector(
    execution: _FormulaExecution,
    selector_name: str,
) -> bool:
    return _formula_execution_references_names(
        execution,
        {selector_name},
        selectors_only=bool(execution.trace),
    )


def _rule_exception_selector_names(rule: dict[str, Any]) -> set[str]:
    """Return formula identifiers used as boolean control selectors."""

    return _formula_exception_selector_names(_rule_formula_text(rule))


def _formula_exception_selector_names(formula_text: str) -> set[str]:
    """Return boolean-control identifiers from one formula text."""

    selector_names: set[str] = set()

    def record(name: str) -> None:
        selector_names.add(name)

    def inspect_expression(text: str) -> None:
        expression = _parse_formula_expression(text)
        if not isinstance(
            expression,
            (ast.BoolOp, ast.Compare, ast.Name, ast.UnaryOp),
        ):
            return
        function_names = {
            node.func.id
            for node in ast.walk(expression)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for name in {
            node.id
            for node in ast.walk(expression)
            if isinstance(node, ast.Name)
            and node.id.lower() not in {"true", "false", "holds", "not_holds"}
            and node.id not in function_names
        }:
            record(name)

    def inspect(text: str, *, depth: int = 0) -> None:
        if depth > 32:
            return
        node = _first_formula_branch_node(text)
        if node is None:
            inspect_expression(text)
            return
        if node.kind == "if":
            for condition in node.selectors:
                for name in _exception_condition_names(condition):
                    record(name)
        elif node.kind == "match":
            for selector in node.selectors:
                for name in _exception_condition_names(selector):
                    record(name)
        for choice in node.choices:
            inspect(choice, depth=depth + 1)
        remainder = text[: node.start] + text[node.end :]
        if remainder.strip():
            inspect(remainder, depth=depth + 1)

    inspect(formula_text)
    return selector_names


def _formula_control_selector_name_groups(
    formula_text: str,
    *,
    include_result_leaves: bool = False,
) -> tuple[frozenset[str], ...]:
    """Return every active control path, omitting constant-inactive leaves."""

    def bounded_groups(
        values: Iterable[frozenset[str]],
    ) -> tuple[frozenset[str], ...]:
        unique = sorted(
            set(values), key=lambda value: (len(value), tuple(sorted(value)))
        )
        minimal = [
            value
            for value in unique
            if not any(candidate < value for candidate in unique)
        ]
        if len(minimal) <= _SOURCE_EXPLICIT_CONDITION_EXPANSION_LIMIT:
            return tuple(minimal)
        return tuple(
            dict.fromkeys(
                (
                    *minimal[: _SOURCE_EXPLICIT_CONDITION_EXPANSION_LIMIT - 1],
                    frozenset(),
                )
            )
        )

    def node_names(expression: ast.AST) -> frozenset[str]:
        function_names = {
            node.func.id
            for node in ast.walk(expression)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        return frozenset(
            node.id
            for node in ast.walk(expression)
            if isinstance(node, ast.Name)
            and node.id.lower() not in {"true", "false", "holds", "not_holds"}
            and node.id not in function_names
        )

    def combine(
        left: tuple[frozenset[str], ...],
        right: tuple[frozenset[str], ...],
    ) -> tuple[frozenset[str], ...]:
        if not left or not right:
            return ()
        return bounded_groups(a | b for a in left for b in right)

    def outcome_groups(
        expression: ast.expr,
        *,
        truth: bool,
        depth: int = 0,
    ) -> tuple[frozenset[str], ...]:
        if depth > _SOURCE_EXPLICIT_CONDITION_AST_DEPTH_LIMIT:
            return (frozenset(),)
        if isinstance(expression, ast.Constant):
            return (frozenset(),) if bool(expression.value) is truth else ()
        if isinstance(expression, ast.Name):
            if expression.id.lower() in {"true", "false"}:
                value = expression.id.lower() == "true"
                return (frozenset(),) if value is truth else ()
            return (frozenset({expression.id}),)
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
            return outcome_groups(
                expression.operand,
                truth=not truth,
                depth=depth + 1,
            )
        if isinstance(expression, ast.BoolOp):
            values = tuple(expression.values)
            if isinstance(expression.op, ast.And):
                if truth:
                    groups: tuple[frozenset[str], ...] = (frozenset(),)
                    for value in values:
                        groups = combine(
                            groups,
                            outcome_groups(value, truth=True, depth=depth + 1),
                        )
                    return groups
                alternatives: list[frozenset[str]] = []
                prefix: tuple[frozenset[str], ...] = (frozenset(),)
                for value in values:
                    alternatives.extend(
                        combine(
                            prefix,
                            outcome_groups(value, truth=False, depth=depth + 1),
                        )
                    )
                    prefix = combine(
                        prefix,
                        outcome_groups(value, truth=True, depth=depth + 1),
                    )
                    if not prefix:
                        break
                return bounded_groups(alternatives)
            if isinstance(expression.op, ast.Or):
                if not truth:
                    groups = (frozenset(),)
                    for value in values:
                        groups = combine(
                            groups,
                            outcome_groups(value, truth=False, depth=depth + 1),
                        )
                    return groups
                alternatives = []
                prefix = (frozenset(),)
                for value in values:
                    alternatives.extend(
                        combine(
                            prefix,
                            outcome_groups(value, truth=True, depth=depth + 1),
                        )
                    )
                    prefix = combine(
                        prefix,
                        outcome_groups(value, truth=False, depth=depth + 1),
                    )
                    if not prefix:
                        break
                return bounded_groups(alternatives)
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and len(expression.args) == 1
            and not expression.keywords
            and expression.func.id in {"holds", "not_holds"}
        ):
            return outcome_groups(
                expression.args[0],
                truth=truth if expression.func.id == "holds" else not truth,
                depth=depth + 1,
            )
        return (node_names(expression),)

    def condition_groups(
        condition: str,
        *,
        truth: bool,
    ) -> tuple[frozenset[str], ...]:
        expression = _parse_formula_expression(condition)
        if expression is None:
            names = frozenset(_exception_condition_names(condition))
            return (names,)
        return outcome_groups(expression, truth=truth)

    def inspect(
        text: str,
        *,
        include_leaves: bool,
        depth: int = 0,
    ) -> tuple[frozenset[str], ...]:
        if depth > 32:
            return (frozenset(),)
        node = _first_formula_branch_node(text)
        if node is None:
            value = _evaluate_formula_selector(text, {})
            number = _rulespec_runtime_decimal(value)
            if value is False or number is not None and number == 0:
                return ()
            if not include_leaves:
                return (frozenset(),)
            expression = _parse_formula_expression(text)
            return (
                outcome_groups(expression, truth=True)
                if expression is not None
                else (frozenset(),)
            )

        choice_groups: list[frozenset[str]] = []
        for choice_index, choice in enumerate(node.choices):
            if node.kind == "if":
                selector_groups: tuple[frozenset[str], ...] = (frozenset(),)
                for selector_index, selector in enumerate(node.selectors):
                    selector_groups = combine(
                        selector_groups,
                        condition_groups(
                            selector,
                            truth=(
                                choice_index < len(node.selectors)
                                and selector_index == choice_index
                            ),
                        ),
                    )
                    if not selector_groups or selector_index >= choice_index:
                        break
            else:
                selector_groups = (
                    frozenset().union(
                        *(
                            _exception_condition_names(selector)
                            for selector in node.selectors
                        )
                    ),
                )
            selected_text = (
                text[: node.start] + textwrap.dedent(choice).strip() + text[node.end :]
            )
            nested_groups = inspect(
                selected_text,
                include_leaves=include_leaves,
                depth=depth + 1,
            )
            choice_groups.extend(combine(selector_groups, nested_groups))
        return bounded_groups(choice_groups)

    return inspect(
        formula_text,
        include_leaves=include_result_leaves,
    )


def _exception_condition_names(condition: str) -> set[str]:
    return {
        identifier
        for identifier in _FORMULA_IDENTIFIER.findall(condition)
        if identifier.lower()
        not in {"and", "or", "not", "true", "false", "holds", "not_holds"}
    }


def _exception_selector_semantic_active_value(name: str) -> bool:
    normalized = _normalized_selector_name(name)
    negated = bool(_selector_identifier_negation_count(normalized) % 2)
    if re.search(
        r"(?:exception|exempt|befrei|ausnahme|barred|blocking|waiver)",
        normalized,
        flags=re.IGNORECASE,
    ):
        return not negated
    if _ordinary_semantic_identifier(normalized) or re.search(
        r"(?:ineligib|disqualif|unqualif)",
        normalized,
        flags=re.IGNORECASE,
    ):
        return negated
    return True


def _formula_indent_width(value: str) -> int:
    return len(value.expandtabs(4))


def _ordinary_semantic_identifier(identifier: str) -> bool:
    return bool(
        re.search(
            r"(?:^|_)(?:ordinary|regular|default|eligible|qualified)"
            r"(?:_|$)",
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
    principal_rules: dict[str, dict[str, Any]],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    direction: str,
    formula_environment: dict[str, Any],
    source_formula_branches: Sequence[SourceStructureBranch],
    rounding_refers_to_result: bool,
    require_clause_binding: bool,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> set[tuple[str, str, str]]:
    evidence: set[tuple[str, str, str]] = set()
    functions = {
        "nearest"
        if direction == "nearest"
        else ("ceil" if direction == "upward" else "floor")
    }
    for case in asserted_by_rule.get(rule_name, ()):
        inputs = case.get("input")
        if not isinstance(inputs, dict):
            continue
        dependency_environment = _case_asserted_dependency_environment(
            principal_rules,
            case,
            formula_environment=formula_environment,
        )
        execution = _case_formula_execution(
            rule,
            case,
            formula_environment=formula_environment,
            dependency_environment=dependency_environment,
        )
        if execution is None or not _formula_execution_implements_rounding(
            execution,
            direction,
        ):
            continue
        operative_leaf = _simplified_formula_text(
            execution.leaf,
            environment=execution.constant_environment,
        )
        input_environment = _case_input_formula_environment(case)
        if input_environment is None:
            continue
        evaluation_environment = dict(execution.constant_environment)
        evaluation_environment.update(dependency_environment)
        evaluation_environment.update(input_environment)
        for function_name, operand in _rounding_call_operands(
            operative_leaf,
            functions=functions,
            root_only=rounding_refers_to_result,
        ):
            effective_operand = _rounding_demonstrated_operand(
                operand,
                direction=direction,
            )
            if effective_operand is None:
                continue
            source_binding_operand = _expand_reached_formula_dependencies(
                effective_operand,
                principal_rules=principal_rules,
                case=case,
                formula_environment=formula_environment,
                dependency_environment=dependency_environment,
            )
            operand_value = _evaluate_formula_selector(
                effective_operand,
                evaluation_environment,
            )
            if (
                _rulespec_runtime_decimal(operand_value) is None
                or float(operand_value).is_integer()
                or not _fractional_input_materially_affects_operand(
                    case,
                    effective_operand,
                    evaluation_environment=evaluation_environment,
                    operand_value=float(operand_value),
                    principal_rules=principal_rules,
                    formula_environment=formula_environment,
                    dependency_names=set(dependency_environment),
                )
            ):
                continue
            if source_formula_branches and not any(
                _rounding_call_binds_source_clause(
                    source_binding_operand,
                    original_operand=effective_operand,
                    operand_value=float(operand_value),
                    direction=direction,
                    rule_name=rule_name,
                    execution=execution,
                    source_formula_branch=source_formula_branch,
                    formula_environment=formula_environment,
                    rounding_refers_to_result=rounding_refers_to_result,
                    require_clause_binding=require_clause_binding,
                    extract_numeric_occurrences=extract_numeric_occurrences,
                    numeric_value_is_grounded=numeric_value_is_grounded,
                )
                for source_formula_branch in source_formula_branches
            ):
                continue
            call_key = f"{function_name}:" + _formula_leaf_semantic_key(
                effective_operand,
                formula_environment=execution.constant_environment,
            )
            evidence.add(
                (
                    rule_name,
                    _formula_execution_outcome(execution),
                    call_key,
                )
            )
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
    return (
        re.search(
            rf"\b{function_name}\s*\(",
            execution.leaf,
        )
        is not None
    )


def _rounding_call_operands(
    formula_text: str,
    *,
    functions: set[str],
    root_only: bool = False,
) -> tuple[tuple[str, str], ...]:
    calls: list[tuple[str, str]] = []
    function_names = {"floor", "ceil"} & functions
    if "nearest" in functions:
        function_names.add("floor")
    if root_only:
        with contextlib.suppress(SyntaxError):
            expression = ast.parse(formula_text.strip(), mode="eval").body
            if (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Name)
                and expression.func.id in function_names
                and len(expression.args) == 1
                and not expression.keywords
            ):
                operand = ast.unparse(expression.args[0])
                if functions != {"nearest"} or re.search(r"\+\s*0?\.5\b", operand):
                    return ((expression.func.id, operand),)
        return ()
    for function_name in function_names:
        for operand in _balanced_call_operands(formula_text, function_name):
            if functions == {"nearest"} and not re.search(r"\+\s*0?\.5\b", operand):
                continue
            calls.append((function_name, operand))
    return tuple(calls)


def _rounding_demonstrated_operand(
    operand: str,
    *,
    direction: str,
) -> str | None:
    if direction != "nearest":
        return operand
    with contextlib.suppress(SyntaxError):
        expression = ast.parse(operand.strip(), mode="eval").body
        if isinstance(expression, ast.BinOp) and isinstance(
            expression.op,
            ast.Add,
        ):
            left_value = _known_numeric_formula_value(expression.left, {})
            right_value = _known_numeric_formula_value(expression.right, {})
            if left_value is not None and math.isclose(float(left_value), 0.5):
                return ast.unparse(expression.right)
            if right_value is not None and math.isclose(float(right_value), 0.5):
                return ast.unparse(expression.left)
    return None


def _expand_reached_formula_dependencies(
    expression_text: str,
    *,
    principal_rules: dict[str, dict[str, Any]],
    case: dict[str, Any],
    formula_environment: dict[str, Any],
    dependency_environment: dict[str, Any],
) -> str:
    """Inline reached, assertion-corroborated intermediates for source matching."""

    try:
        expression = ast.parse(expression_text.strip(), mode="eval").body
    except SyntaxError:
        return expression_text

    resolving: set[str] = set()

    class DependencyExpander(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            name = node.id
            rule = principal_rules.get(name)
            if rule is None or name not in dependency_environment or name in resolving:
                return node
            execution = _case_formula_execution(
                rule,
                case,
                formula_environment=formula_environment,
                dependency_environment=dependency_environment,
            )
            if execution is None:
                return node
            try:
                replacement = ast.parse(
                    execution.leaf.strip(),
                    mode="eval",
                ).body
            except SyntaxError:
                return node
            resolving.add(name)
            expanded = self.visit(replacement)
            resolving.remove(name)
            return ast.copy_location(expanded, node)

    expanded = DependencyExpander().visit(expression)
    ast.fix_missing_locations(expanded)
    return ast.unparse(expanded)


def _fractional_input_materially_affects_operand(
    case: dict[str, Any],
    operand: str,
    *,
    evaluation_environment: dict[str, Any],
    operand_value: float,
    principal_rules: dict[str, dict[str, Any]],
    formula_environment: dict[str, Any],
    dependency_names: set[str],
) -> bool:
    inputs = case.get("input")
    if not isinstance(inputs, dict):
        return False
    operand_names = set(_FORMULA_IDENTIFIER.findall(operand))
    uses_derived_operand = bool(operand_names & dependency_names)
    for key, value in inputs.items():
        if _rulespec_runtime_decimal(value) is None:
            continue
        aliases = _input_key_names(key) & operand_names
        if not aliases and not uses_derived_operand:
            continue
        replacement: int | float
        if float(value).is_integer():
            replacement = int(value) + 1
        else:
            replacement = float(math.floor(float(value)))
        candidate_inputs = dict(inputs)
        candidate_inputs[key] = replacement
        candidate_case = dict(case)
        candidate_case["input"] = candidate_inputs
        candidate_dependencies = _case_dependency_environment(
            principal_rules,
            candidate_case,
            formula_environment=formula_environment,
            require_asserted_value=False,
            allowed_names=dependency_names,
        )
        changed_environment = _formula_environment_for_case(
            formula_environment,
            candidate_case,
        )
        changed_environment.update(candidate_dependencies)
        changed_inputs = _case_input_formula_environment(candidate_case)
        if changed_inputs is None:
            continue
        changed_environment.update(changed_inputs)
        changed_value = _evaluate_formula_selector(
            operand,
            changed_environment,
        )
        if _rulespec_runtime_decimal(changed_value) is not None and not math.isclose(
            float(changed_value), operand_value
        ):
            return True
    return False


def _rounding_call_matches_source_formula(
    operand: str,
    *,
    operand_value: float,
    execution: _FormulaExecution,
    source_formula_branch: SourceStructureBranch,
    formula_environment: dict[str, Any],
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> bool:
    interval = _formula_branch_interval(
        source_formula_branch,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    operand_execution = _FormulaExecution(
        (),
        operand,
        (type(operand_value).__name__, repr(operand_value)),
        operand_value == 0,
        execution.constant_environment,
    )
    return _formula_execution_matches_source_branch(
        operand_execution,
        source_formula_branch,
        interval=interval,
        formula_environment=formula_environment,
        extract_numeric_occurrences=extract_numeric_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _rounding_call_binds_source_clause(
    operand: str,
    *,
    original_operand: str,
    operand_value: float,
    direction: str,
    rule_name: str,
    execution: _FormulaExecution,
    source_formula_branch: SourceStructureBranch,
    formula_environment: dict[str, Any],
    rounding_refers_to_result: bool,
    require_clause_binding: bool,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> bool:
    formula_match = _rounding_call_matches_source_formula(
        operand,
        operand_value=operand_value,
        execution=execution,
        source_formula_branch=source_formula_branch,
        formula_environment=formula_environment,
        extract_numeric_occurrences=extract_numeric_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    source_operations = _formula_operation_kinds(source_formula_branch.text)
    if (
        direction == "nearest"
        and (_formula_ast_operation_kinds(original_operand) & {"add", "subtract"})
        - source_operations
    ):
        return False
    if rounding_refers_to_result:
        return formula_match
    source_has_distinguishing_computation = bool(
        source_operations
        or extract_numeric_occurrences(
            authoritative_numeric_recall_text(source_formula_branch.text)
        )
    )
    if source_has_distinguishing_computation:
        return formula_match
    if require_clause_binding:
        return True
    return _rounding_operand_is_output_stage(
        original_operand,
        rule_name=rule_name,
    )


def _rounding_operand_is_output_stage(
    operand: str,
    *,
    rule_name: str,
) -> bool:
    output_subject = _rounding_stage_subject(rule_name)
    return bool(output_subject) and any(
        _rounding_stage_subject_matches(
            _rounding_stage_subject(identifier),
            output_subject,
        )
        for identifier in _FORMULA_IDENTIFIER.findall(operand)
    )


def _rounding_stage_subject_matches(
    operand_subject: tuple[str, ...],
    output_subject: tuple[str, ...],
) -> bool:
    return bool(operand_subject) and operand_subject == output_subject


def _rounding_stage_subject(identifier: str) -> tuple[str, ...]:
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", identifier)
    stage_tokens = {
        "round",
        "rounded",
        "rounding",
        "unrounded",
        "raw",
        "pre",
        "post",
        "before",
        "after",
        "floor",
        "floored",
        "ceil",
        "ceiled",
        "down",
        "downward",
        "up",
        "upward",
        "nearest",
        "final",
    }
    subject = tuple(
        token
        for token in re.findall(r"[A-Za-z0-9]+", normalized.lower())
        if token not in stage_tokens
    )
    if len(subject) > 1 and subject[-1] in {"amount", "value", "result"}:
        return subject[:-1]
    return subject


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
        direct_text = authoritative_numeric_recall_text(
            _source_branch_direct_text(branch, branches=branches)
        )
        direct_inventory = tuple(extract_numeric_occurrences(direct_text))
        for fragment_start, fragment in _source_boundary_fragments(direct_text):
            range_fragment = fragment.split(":", 1)[0]
            if not re.search(
                r"\b(?:zwischen|between|bis|von|ab|unter|über|"
                r"mehr\s+als|weniger\s+als|from|to|"
                r"through|less\s+than|more\s+than|at\s+least|up\s+to|"
                r"at\s+most|above|below|höchstens|mindestens|"
                r"nicht\s+mehr\s+als)\b",
                range_fragment,
                flags=re.IGNORECASE,
            ):
                continue
            interval = _shift_numeric_interval(
                _formula_interval_from_text(
                    range_fragment,
                    extract_numeric_occurrences=extract_numeric_occurrences,
                ),
                fragment_start,
            )
            if interval is None:
                continue
            obligations.extend(
                (branch, boundary)
                for boundary in (interval.lower, interval.upper)
                if boundary is not None
                and any(
                    boundary.start == occurrence.start
                    and boundary.end == occurrence.end
                    and _numeric_occurrences_are_equivalent(boundary, occurrence)
                    for occurrence in direct_inventory
                )
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
                occurrence.has_temporal_context,
                occurrence.has_structural_context,
                occurrence.source_value,
                occurrence.requires_rate_context,
                occurrence.is_word_number,
                occurrence.alternative_values,
                occurrence.start,
                occurrence.end,
            ): (branch, occurrence)
            for branch, occurrence in obligations
        }.values()
    )


def _source_branch_direct_text(
    branch: SourceStructureBranch,
    *,
    branches: Sequence[SourceStructureBranch],
) -> str:
    """Return one branch's own chapeau without nested child bodies.

    ``SourceStructureBranch.text`` intentionally spans descendants so structural
    and formula analysis can retain parent context. Boundary inventory is
    different: counting descendant thresholds again on every ancestor creates
    impossible parent-level obligations and obscures the child rule that owns
    each comparator. Preserve the original prefix offsets while stopping at the
    first descendant; leaf branches continue to use their complete text. Using
    any descendant is necessary because callers may already have removed
    marker-only intermediate containers from the candidate branch sequence.
    """

    descendant_starts = [
        candidate.start
        for candidate in branches
        if len(candidate.path) > len(branch.path)
        and candidate.path[: len(branch.path)] == branch.path
        and branch.start <= candidate.start < branch.end
    ]
    if not descendant_starts:
        return branch.text
    direct_end = max(0, min(descendant_starts) - branch.start)
    return branch.text[:direct_end]


def _numeric_test_input_values(value: Any) -> tuple[float, ...]:
    values: list[float] = []
    if isinstance(value, bool):
        return ()
    if _rulespec_runtime_decimal(value) is not None:
        return (float(value),)
    if isinstance(value, dict):
        for item in value.values():
            values.extend(_numeric_test_input_values(item))
    elif isinstance(value, list):
        for item in value:
            values.extend(_numeric_test_input_values(item))
    return tuple(values)


def _paired_boolean_toggle_case_pairs(
    cases: Iterable[dict[str, Any]],
) -> Iterable[tuple[str, dict[str, Any], dict[str, Any]]]:
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
            yield changed[0], left_case, right_case


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


def _rule_formula_text_for_case(
    rule: dict[str, Any],
    case: dict[str, Any],
) -> str | None:
    """Resolve one temporal formula when the companion case identifies a period."""

    versions = rule.get("versions")
    if (
        not isinstance(versions, list)
        or len(versions) > _TEMPORAL_WITNESS_VERSION_LIMIT
    ):
        return None
    unambiguous = _unambiguous_rule_formula_text(rule)
    formula_versions = [
        version
        for version in versions
        if isinstance(version, dict) and version.get("formula") is not None
    ]
    has_temporal_metadata = any(
        str(version.get("effective_from") or "").strip()
        or str(version.get("effective_to") or "").strip()
        for version in formula_versions
    )
    if has_temporal_metadata:
        for version in formula_versions:
            effective_from = str(version.get("effective_from") or "").strip()
            effective_to = str(version.get("effective_to") or "").strip()
            if (
                not _is_iso_calendar_date(effective_from)
                or (effective_to and not _is_iso_calendar_date(effective_to))
                or (effective_to and effective_to < effective_from)
            ):
                return None
    period = _normalized_case_period(case)
    if not _is_iso_calendar_date(period):
        if "period" not in case:
            return unambiguous
        return None if has_temporal_metadata else unambiguous
    if not has_temporal_metadata:
        return unambiguous
    candidates: list[tuple[str, str]] = []
    for version in formula_versions:
        effective_from = str(version.get("effective_from") or "").strip()
        effective_to = str(version.get("effective_to") or "").strip()
        if effective_from > period or (effective_to and effective_to < period):
            continue
        candidates.append((effective_from, str(version["formula"])))
    if not candidates:
        return None
    latest = max(effective_from for effective_from, _formula in candidates)
    selected = [
        formula for effective_from, formula in candidates if effective_from == latest
    ]
    return selected[0] if len(selected) == 1 else None


def _selected_rule_formula_version_index(
    rule: dict[str, Any],
    case: dict[str, Any],
) -> int | None:
    versions = rule.get("versions")
    if (
        not isinstance(versions, list)
        or len(versions) > _TEMPORAL_WITNESS_VERSION_LIMIT
    ):
        return None
    formula_versions = [
        (index, version)
        for index, version in enumerate(versions)
        if isinstance(version, dict) and version.get("formula") is not None
    ]
    if len(formula_versions) == 1:
        index, version = formula_versions[0]
        effective_from = str(version.get("effective_from") or "").strip()
        effective_to = str(version.get("effective_to") or "").strip()
        if not effective_from and not effective_to:
            return index
        if "period" not in case and (
            _is_iso_calendar_date(effective_from)
            and (not effective_to or _is_iso_calendar_date(effective_to))
            and (not effective_to or effective_to >= effective_from)
        ):
            return index
        if "period" not in case:
            return None
    period = _normalized_case_period(case)
    if not _is_iso_calendar_date(period):
        return None
    candidates: list[tuple[int, str]] = []
    for index, version in formula_versions:
        effective_from = str(version.get("effective_from") or "").strip()
        effective_to = str(version.get("effective_to") or "").strip()
        if (
            not _is_iso_calendar_date(effective_from)
            or (effective_to and not _is_iso_calendar_date(effective_to))
            or (effective_to and effective_to < effective_from)
        ):
            return None
        if effective_from <= period and (not effective_to or period <= effective_to):
            candidates.append((index, effective_from))
    if not candidates:
        return None
    latest = max(effective_from for _index, effective_from in candidates)
    selected = [index for index, start in candidates if start == latest]
    return selected[0] if len(selected) == 1 else None


def _constant_rule_environment(payload: dict[str, Any]) -> dict[str, Any]:
    """Return bounded local parameter values and temporal aliases."""

    environment: dict[str, Any] = {}
    module = payload.get("module")
    source_verification = (
        module.get("source_verification") if isinstance(module, dict) else None
    )
    authoritative_citation_path = (
        str(source_verification.get("corpus_citation_path") or "").strip("/").lower()
        if isinstance(source_verification, dict)
        else ""
    )
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return environment
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        name = str(rule.get("name") or "").strip()
        versions = rule.get("versions")
        if (
            not name
            or not isinstance(versions, list)
            or len(versions) > _TEMPORAL_WITNESS_VERSION_LIMIT
        ):
            continue
        formula_version_count = sum(
            1
            for version in versions
            if isinstance(version, dict) and "formula" in version
        )
        entries: list[tuple[str, str, Any]] = []
        entry_formula_excerpts: list[tuple[str, ...]] = []
        formula_excerpts_by_version: dict[int, list[str]] = {}
        for path, citation_path, excerpt in _rule_source_excerpt_atoms(rule):
            if (
                not authoritative_citation_path
                or citation_path.strip("/").lower() != authoritative_citation_path
            ):
                continue
            match = re.fullmatch(r"versions\[(\d+)\]\.formula", path)
            if match is not None:
                formula_excerpts_by_version.setdefault(int(match.group(1)), []).append(
                    excerpt
                )
        for version_index, version in enumerate(versions):
            if not isinstance(version, dict) or "formula" not in version:
                continue
            formula = version["formula"]
            with contextlib.suppress(SyntaxError, ValueError):
                value = ast.literal_eval(str(formula))
                if isinstance(value, (str, int, float, bool)) and not isinstance(
                    value,
                    complex,
                ):
                    entries.append(
                        (
                            str(version.get("effective_from") or "").strip(),
                            str(version.get("effective_to") or "").strip(),
                            value,
                        )
                    )
                    entry_formula_excerpts.append(
                        tuple(formula_excerpts_by_version.get(version_index, ()))
                    )
        if len(entries) != formula_version_count:
            continue
        has_temporal_metadata = any(start or end for start, end, _value in entries)
        if (
            has_temporal_metadata
            and entries
            and all(
                _is_iso_calendar_date(start)
                and (not end or _is_iso_calendar_date(end))
                and (not end or end >= start)
                for start, end, _value in entries
            )
        ):
            environment[name] = _TemporalFormulaValue(
                tuple(entries),
                tuple(entry_formula_excerpts),
            )
            continue
        if has_temporal_metadata:
            continue
        values = [value for _start, _end, value in entries]
        if values and all(
            type(value) is type(values[0]) and value == values[0]
            for value in values[1:]
        ):
            environment[name] = values[0]
    pending_aliases = {
        str(rule.get("name") or "").strip(): rule
        for rule in rules
        if isinstance(rule, dict)
        and rule.get("kind") == "parameter"
        and str(rule.get("name") or "").strip()
        and str(rule.get("name") or "").strip() not in environment
    }
    if len(pending_aliases) > _TEMPORAL_WITNESS_NAME_LIMIT:
        return environment
    for _ in range(_TEMPORAL_WITNESS_NAME_LIMIT):
        changed = False
        prior_environment = dict(environment)
        resolved_round: dict[str, Any] = {}
        for name, rule in sorted(pending_aliases.items()):
            value = _resolved_parameter_formula_value(
                rule,
                environment=prior_environment,
                authoritative_citation_path=authoritative_citation_path,
            )
            if value is _UNRESOLVED_CONDITION_VALUE:
                continue
            resolved_round[name] = value
            changed = True
        environment.update(resolved_round)
        for name in resolved_round:
            del pending_aliases[name]
        if not changed:
            break
    return environment


def _resolved_parameter_formula_value(
    rule: dict[str, Any],
    *,
    environment: dict[str, Any],
    authoritative_citation_path: str,
) -> Any:
    """Resolve one input-free parameter alias, failing closed on ambiguity."""

    versions = rule.get("versions")
    if (
        not isinstance(versions, list)
        or not versions
        or len(versions) > _TEMPORAL_WITNESS_VERSION_LIMIT
    ):
        return _UNRESOLVED_CONDITION_VALUE
    formula_versions = [
        (index, version)
        for index, version in enumerate(versions)
        if isinstance(version, dict) and version.get("formula") is not None
    ]
    if not formula_versions:
        return _UNRESOLVED_CONDITION_VALUE
    formulas = [str(version["formula"]) for _index, version in formula_versions]
    referenced_names = set().union(
        *(_FORMULA_IDENTIFIER.findall(formula) for formula in formulas),
        set(),
    )
    unresolved_names = {
        name
        for name in referenced_names
        if name not in environment
        and name.lower() not in _PARAMETER_ALIAS_FORMULA_KEYWORDS
    }
    if unresolved_names:
        return _UNRESOLVED_CONDITION_VALUE
    local_dependencies = {
        name: environment[name] for name in referenced_names if name in environment
    }
    temporal_dependencies = {
        name: value
        for name, value in local_dependencies.items()
        if isinstance(value, _TemporalFormulaValue)
    }
    change_points: set[str] = set()
    has_temporal_metadata = False
    for _index, version in formula_versions:
        start = str(version.get("effective_from") or "").strip()
        end = str(version.get("effective_to") or "").strip()
        if start or end:
            has_temporal_metadata = True
            if (
                not _is_iso_calendar_date(start)
                or (end and not _is_iso_calendar_date(end))
                or (end and end < start)
            ):
                return _UNRESOLVED_CONDITION_VALUE
            change_points.add(start)
            if end and (after_end := _shift_iso_date(end, 1)) is not None:
                change_points.add(after_end)
    for temporal in temporal_dependencies.values():
        for start, end, _value in temporal.versions:
            if not _is_iso_calendar_date(start) or (
                end and not _is_iso_calendar_date(end)
            ):
                return _UNRESOLVED_CONDITION_VALUE
            change_points.add(start)
            if end and (after_end := _shift_iso_date(end, 1)) is not None:
                change_points.add(after_end)

    excerpts_by_version: dict[int, list[str]] = {}
    for path, citation_path, excerpt in _rule_source_excerpt_atoms(rule):
        if (
            not authoritative_citation_path
            or citation_path.strip("/").lower() != authoritative_citation_path
        ):
            continue
        match = re.fullmatch(r"versions\[(\d+)\]\.formula", path)
        if match is not None:
            excerpts_by_version.setdefault(int(match.group(1)), []).append(excerpt)

    if not has_temporal_metadata and not temporal_dependencies:
        formula = _unambiguous_rule_formula_text(rule)
        if formula is None:
            return _UNRESOLVED_CONDITION_VALUE
        resolved = _evaluate_parameter_alias_formula(
            formula,
            environment=_formula_environment_for_case(environment, {}),
        )
        return resolved

    ordered_points = sorted(change_points)
    if not ordered_points or len(ordered_points) > _TEMPORAL_WITNESS_VERSION_LIMIT:
        return _UNRESOLVED_CONDITION_VALUE
    entries: list[tuple[str, str, Any]] = []
    entry_excerpts: list[tuple[str, ...]] = []
    for point_index, point in enumerate(ordered_points):
        case = {"period": point}
        version_index = _selected_rule_formula_version_index(rule, case)
        formula = _rule_formula_text_for_case(rule, case)
        if version_index is None or formula is None:
            continue
        case_environment = _formula_environment_for_case(environment, case)
        resolved = _evaluate_parameter_alias_formula(
            formula,
            environment=case_environment,
        )
        if resolved is _UNRESOLVED_CONDITION_VALUE:
            continue
        next_point = (
            ordered_points[point_index + 1]
            if point_index + 1 < len(ordered_points)
            else None
        )
        if next_point is not None:
            end = _shift_iso_date(next_point, -1)
            if end is None:
                return _UNRESOLVED_CONDITION_VALUE
        else:
            selected_version = versions[version_index]
            end = str(selected_version.get("effective_to") or "").strip()
        propagated_excerpts = list(excerpts_by_version.get(version_index, ()))
        execution = _execute_formula_text(
            formula,
            environment=case_environment,
            constant_environment=case_environment,
        )
        if execution is None:
            return _UNRESOLVED_CONDITION_VALUE
        reached_names = _reached_formula_expression_identifier_names(
            execution.leaf,
            environment=case_environment,
        )
        reached_names.update(
            name
            for step in execution.trace
            for selector in step.selectors
            for name in _reached_formula_expression_identifier_names(
                selector,
                environment=case_environment,
            )
        )
        for dependency_name in sorted(reached_names):
            temporal = temporal_dependencies.get(dependency_name)
            if temporal is None:
                continue
            for selected_index in _selected_temporal_version_indexes(temporal, case):
                if selected_index < len(temporal.version_formula_excerpts):
                    propagated_excerpts.extend(
                        temporal.version_formula_excerpts[selected_index]
                    )
        entries.append((point, end, resolved))
        entry_excerpts.append(tuple(dict.fromkeys(propagated_excerpts)))
    if not entries or len(entries) > _TEMPORAL_WITNESS_VERSION_LIMIT:
        return _UNRESOLVED_CONDITION_VALUE
    return _TemporalFormulaValue(tuple(entries), tuple(entry_excerpts))


def _evaluate_parameter_alias_formula(
    formula: str,
    *,
    environment: dict[str, Any],
) -> Any:
    """Evaluate one bounded parameter formula without inventing inputs."""

    execution = _execute_formula_text(
        formula,
        environment=environment,
        constant_environment=environment,
    )
    if execution is None:
        return _UNRESOLVED_CONDITION_VALUE
    value = _evaluate_formula_selector(execution.leaf, environment)
    if (
        value is _UNRESOLVED_CONDITION_VALUE
        or not isinstance(value, (str, int, float, bool))
        or isinstance(value, complex)
    ):
        return _UNRESOLVED_CONDITION_VALUE
    return value


def _shift_iso_date(value: str, days: int) -> str | None:
    """Return one bounded calendar-date shift used for temporal segments."""

    with contextlib.suppress(ValueError, OverflowError):
        parsed = date.fromisoformat(value)
        return date.fromordinal(parsed.toordinal() + days).isoformat()
    return None


def _formula_environment_for_case(
    environment: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    period = _normalized_case_period(case)
    resolved: dict[str, Any] = {}
    for name, value in environment.items():
        if not isinstance(value, _TemporalFormulaValue):
            resolved[name] = value
            continue
        if not _is_iso_calendar_date(period):
            if "period" not in case:
                candidates = [candidate for _start, _end, candidate in value.versions]
                if candidates and all(
                    type(candidate) is type(candidates[0])
                    and candidate == candidates[0]
                    for candidate in candidates[1:]
                ):
                    resolved[name] = candidates[0]
            continue
        selected_indexes = _selected_temporal_version_indexes(value, case)
        if len(selected_indexes) != 1:
            continue
        resolved[name] = value.versions[selected_indexes[0]][2]
    return resolved


def _normalized_case_period(case: dict[str, Any]) -> str:
    raw_period = case.get("period")
    if isinstance(raw_period, dict):
        start = raw_period.get("start")
        period = str(start or "").strip()
    else:
        period = str(raw_period or "").strip()
    if re.fullmatch(r"\d{4}", period):
        return f"{period}-01-01"
    if re.fullmatch(r"\d{4}-\d{2}", period):
        return f"{period}-01"
    return period


def _is_iso_calendar_date(value: str) -> bool:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    with contextlib.suppress(ValueError):
        date.fromisoformat(value)
        return True
    return False


def _collapse_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _bounded_source_feedback_excerpt(value: str, *, limit: int = 360) -> str:
    """Render bounded source-identifying feedback without changing its words."""

    return _bounded_source_feedback_preview(value, limit=limit)[0]


def _bounded_source_feedback_preview(
    value: str,
    *,
    limit: int = 360,
) -> tuple[str, bool]:
    """Render a source preview and report whether it contains an ellipsis."""

    collapsed = _collapse_text(value).replace("`", "\\`")
    if len(collapsed) <= limit:
        return collapsed, False
    marker = " ... "
    available = limit - len(marker)
    head_length = (available * 2) // 3
    tail_length = available - head_length
    return (
        collapsed[:head_length].rstrip() + marker + collapsed[-tail_length:].lstrip(),
        True,
    )


def _formula_output_binding_feedback(
    branch: SourceStructureBranch,
    *,
    corpus_citation_path: str,
    has_path_covering_principal: bool,
    same_owner_missing: Sequence[SourceStructureBranch] = (),
) -> str:
    """Give a repair model the exact proof binding missing from a formula."""

    source_excerpt, source_excerpt_was_truncated = _bounded_source_feedback_preview(
        branch.text
    )
    principal_repair = (
        "If an existing path-covering principal output's formula already "
        "implements this computation, add the following proof atom to it. "
        "Otherwise create a "
        "principal `derived`/`derived_relation` output or precisely defer the "
        "computation. The required binding is a"
        if has_path_covering_principal
        else "Create a principal `derived`/`derived_relation` output (or precisely "
        "defer this computation); when adding the output, bind it with a"
    )
    bounded_preview_warning = (
        " The displayed ` ... ` is only a bounded locator and is not source "
        "text; copy one contiguous verbatim computation-bearing excerpt from "
        "the cited character span instead."
        if source_excerpt_was_truncated
        else ""
    )
    same_owner_detail = _same_owner_formula_clause_feedback(same_owner_missing)
    return (
        " The formula-clause number is an internal punctuation-span ordinal, "
        "not a statutory paragraph number. "
        f"{principal_repair} "
        "`versions[N].formula` proof atom whose "
        "`source.corpus_citation_path` is exactly "
        f"`{corpus_citation_path}` and whose `source.excerpt` is one contiguous "
        "verbatim excerpt that itself states the computation. Source locator at "
        f"characters {branch.start}:{branch.end}: `{source_excerpt}`. A parameter "
        "rule, a citation-only proof atom, a non-formula proof path, or a shorter "
        "excerpt that does not itself state the computation cannot bind a "
        f"principal output to this clause.{bounded_preview_warning}"
        f"{same_owner_detail}"
    )


def _same_owner_formula_clause_feedback(
    branches: Sequence[SourceStructureBranch],
    *,
    limit: int = 32,
) -> str:
    """Give retries the complete bounded missing set for one structural owner."""

    ordered = sorted({(branch.start, branch.end, branch.label) for branch in branches})
    if len(ordered) <= 1:
        return ""
    rendered = [
        f"`{label}` at characters {start}:{end}"
        for start, end, label in ordered[:limit]
    ]
    if len(ordered) > limit:
        rendered.append(f"... ({len(ordered) - limit} additional clauses omitted)")
    return (
        f" Same structural owner has {len(ordered)} missing formula clauses; "
        "repair or precisely defer the complete set in this attempt: "
        + "; ".join(rendered)
        + "."
    )


def _bounded_period_feedback(periods: Sequence[str], *, limit: int = 8) -> str:
    """Render bounded candidate periods while retaining both temporal extremes."""

    if len(periods) <= limit:
        return ", ".join(periods)
    head_count = limit // 2
    tail_count = limit - head_count
    omitted = len(periods) - limit
    return (
        ", ".join(periods[:head_count])
        + f", ... ({omitted} omitted) ..., "
        + ", ".join(periods[-tail_count:])
    )


def _bounded_identifier_feedback(values: Sequence[str], *, limit: int = 6) -> str:
    """Render a bounded, deterministic list of diagnostic identifiers."""

    identifiers = sorted(
        {_collapse_text(value).replace("`", "\\`") for value in values}
    )
    rendered = [f"`{value}`" for value in identifiers[:limit]]
    if len(identifiers) > limit:
        rendered.append(f"... ({len(identifiers) - limit} omitted)")
    return ", ".join(rendered)
