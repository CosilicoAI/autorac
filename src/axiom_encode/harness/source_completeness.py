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
import math
import re
import textwrap
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

import yaml

from axiom_encode.statute import (
    CitationParts,
    normalize_rulespec_path_segment,
    parse_usc_citation,
)


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
    evaluated_value: tuple[str, str] | None
    evaluates_to_zero: bool
    constant_environment: dict[str, Any]


@dataclass(frozen=True)
class _TemporalFormulaValue:
    """Literal parameter values selectable by a companion case period."""

    versions: tuple[tuple[str, str, Any], ...]


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


@dataclass(frozen=True)
class _NamedSourceNumber:
    """A matched source number whose word value may be unsupported."""

    value: float | None


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
_GLUED_SENTENCE_MARKER = re.compile(r"(?<![\w])(?P<label>[1-9]\d?)(?=[A-ZÄÖÜ](?!:))")
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
    r"(?:weggefallen|aufgehoben|repealed|omitted|\.{3,})"
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
    r"sum\s+of|difference\s+between|product\s+of|twice|half\s+of|"
    r"\d+(?:[.,]\d+)?\s+times\b|"
    r"percentage\s+of|in\s+excess\s+of|"
    r"equals?[^.;]{0,100}\b(?:plus|minus|times)\b"
    r")\b",
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
        "income",
        "information",
        "input",
        "limit",
        "notice",
        "policy",
        "process",
        "procedure",
        "rate",
        "receipt",
        "record",
        "requirement",
        "rule",
        "size",
        "standard",
        "status",
        "submission",
        "threshold",
        "timing",
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
    r"document|filing|hearing|inspection|notice|plan|procedure|recommendation|"
    r"record|report|submission|waiver"
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
input or runtime capability; a generic claim that the branch is unavailable is invalid."""
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
    r"§{1,2}\s*\d+[a-z]?"
    r"(?:\s*,\s*\d+[a-z]?)*"
    r"(?:\s*(?:und|bis|[-–—])\s*\d+[a-z]?)*",
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
            SourceStructureBranch(
                path, "paragraph", match.group("marker"), text, start, end
            )
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


def _is_editorial_omission(text: str) -> bool:
    return bool(_EDITORIAL_OMISSION_ONLY.fullmatch(text))


def source_states_explicit_computation(source_text: str) -> bool:
    """Return whether text states a computation rather than only a scalar."""

    computation_text = _without_stated_conversion_results(source_text)
    return bool(
        _has_substantive_arithmetic_expression(computation_text)
        or _COMPUTATION_LANGUAGE.search(computation_text)
        or _ROUNDING_LANGUAGE.search(computation_text)
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
        _has_substantive_arithmetic_expression(candidate)
        or _COMPUTATION_LANGUAGE.search(candidate)
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

    if _WORDED_ARITHMETIC_EXPRESSION.search(source_text):
        return True
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
    artifact_numeric_bindings: Sequence[tuple[str, float]] | None = None,
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
                artifact_numeric_bindings=artifact_numeric_bindings,
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
    artifact_numeric_bindings: Sequence[tuple[str, float]] | None,
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
        formula_environment = _constant_rule_environment(payload)
        if artifact_numeric_bindings is not None:
            formula_environment = _merge_unambiguous_numeric_bindings(
                formula_environment,
                artifact_numeric_bindings,
            )
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
                formula_environment=formula_environment,
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
        clause_count_by_path[clause.path] = clause_count_by_path.get(clause.path, 0) + 1

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
                (excerpt_text := _normalized_formula_clause_text(excerpt))
                and excerpt_citation_path.strip("/").lower() == normalized_citation_path
                and source_states_explicit_computation(excerpt)
                and (excerpt_text in clause_text or clause_text in excerpt_text)
                for excerpt_citation_path, excerpt in _rule_formula_source_excerpts(
                    principal_rules[rule_name]
                )
            )
        }
    return clause_rules


def _normalized_formula_clause_text(text: str) -> str:
    """Normalize a clause while ignoring its structural marker."""

    unmarked = _strip_source_clause_marker(text)
    return _collapse_text(unmarked)


def _strip_source_clause_marker(text: str) -> str:
    """Remove a leading paragraph/list/Satz marker, including glued statutes."""

    return re.sub(
        r"^\s*(?:"
        r"\((?:\d+[a-z]?|[a-z])\)|"
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
    without_rounding = _ROUNDING_LANGUAGE.sub("", text)
    if _has_substantive_arithmetic_expression(
        without_rounding
    ) or _COMPUTATION_LANGUAGE.search(without_rounding):
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
        if "blocked_by" in record:
            precise = (
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
                        source_scope_text,
                        blocker,
                        corpus_citation_path=corpus_citation_path,
                    )
                    for blocker in blocker_targets
                )
            )
        else:
            precise = _reason_dependency_is_source_bound(
                reason,
                source_scope_text,
                corpus_citation_path=corpus_citation_path,
            ) or _reason_names_source_bound_runtime_gap(
                reason,
                source_scope_text,
                path=path,
                corpus_citation_path=corpus_citation_path,
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
                "name an exact missing dependency, input, or runtime capability.\n"
                f"{_IMPRECISE_DEFERRAL_RETRY_SHAPE}"
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
    if re.search(
        r"\b(?:"
        r"nach|gemäß|laut|entsprechend|under|according\s+to|pursuant\s+to|"
        r"abhängig\s+von|depends?\s+on|setzt|requires?|benötigt"
        r")\s*$",
        before,
        flags=re.IGNORECASE,
    ):
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


def _usc_dependency_fragments(match: re.Match[str]) -> tuple[str, ...]:
    return tuple(
        fragment.lower()
        for fragment in re.findall(
            r"\(\s*([A-Za-z0-9]+)\s*\)",
            match.group("tail") or "",
        )
    )


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
) -> bool:
    """Require one external dependency citation to bind to the deferred source."""

    if not _MISSING_DEPENDENCY_LANGUAGE.search(reason):
        return False
    try:
        current_citation = parse_usc_citation(corpus_citation_path)
    except ValueError:
        current_citation = None
    usc_dependencies = _qualified_usc_dependencies(reason)
    for match in usc_dependencies:
        if not _reason_match_names_missing_dependency(
            reason, match, source_scope_text=source_scope_text
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


def _reason_match_names_missing_dependency(
    reason: str,
    match: re.Match[str],
    *,
    source_scope_text: str,
) -> bool:
    clause_start, clause_end = _reason_clause_bounds(reason, match)
    clause = reason[clause_start:clause_end]
    reference_start = match.start() - clause_start
    reference_end = match.end() - clause_start
    before = clause[:reference_start]
    after = clause[reference_end:]
    if _has_adversative_language(reason[match.end() :]):
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
        if not _reason_reference_introduction_is_bounded(bridge):
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
        )
    return True


def _reason_clause_bounds(
    reason: str,
    match: re.Match[str],
) -> tuple[int, int]:
    masked = list(reason)
    for dependency in _qualified_usc_dependencies(reason):
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
                if not _has_adversative_language(bridge) and instrument_link.search(
                    bridge
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
    """Accept a concrete runtime gap anchored to the exact current USC branch."""

    if (
        not path
        or not source_scope_text.strip()
        or not _MISSING_DEPENDENCY_LANGUAGE.search(reason)
        or not _SOURCE_BOUND_RUNTIME_GAP_LANGUAGE.search(reason)
        or not _ADMINISTRATIVE_SOURCE_ARTIFACT_LANGUAGE.search(source_scope_text)
        or not _ADMINISTRATIVE_SOURCE_ACTION_LANGUAGE.search(source_scope_text)
        or source_states_explicit_computation(source_scope_text)
        or not corpus_citation_path.startswith("us/statute/")
    ):
        return False
    try:
        citation = parse_usc_citation(corpus_citation_path)
    except ValueError:
        return False

    dash_pattern = (
        "[-\\u2010\\u2011\\u2012\\u2013\\u2014\\u2015\\u2212\\ufe58\\ufe63\\uff0d]"
    )
    section_pattern = re.escape(
        normalize_rulespec_path_segment(citation.section)
    ).replace(r"\-", dash_pattern)
    branch_pattern = r"\s*".join(
        rf"\(\s*{re.escape(normalize_rulespec_path_segment(part))}\s*\)"
        for part in path
    )
    descendant_guard = r"(?!\s*\()" if len(path) > 1 else ""
    exact_branch_citation = re.compile(
        rf"\b{re.escape(citation.title)}\s+U\.?\s*S\.?\s*C\.?\s*"
        rf"(?:§{{1,2}}\s*)?{section_pattern}\s*{branch_pattern}{descendant_guard}",
        flags=re.IGNORECASE,
    )
    if not exact_branch_citation.search(reason):
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
    return f"{jurisdiction}:{plural}/{'/'.join(tail)}"


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
    cleaned = _GERMAN_LEGAL_CITATION.sub("", cleaned)
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
        numeric_value_is_grounded=numeric_value_is_grounded,
        formula_environment=formula_environment,
    )
    for branch in missing_formula_branches:
        source_excerpt = _bounded_source_feedback_excerpt(branch.text)
        source_location = f"characters {branch.start}:{branch.end}"
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
            f"{witness_requirement}{observed_period_detail} Each formula branch "
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
            extract_numeric_occurrences=extract_numeric_occurrences,
        )
        if missing_exception_branches:
            missing_citations = ", ".join(
                dict.fromkeys(
                    _branch_citation(corpus_citation_path, branch)
                    for branch in missing_exception_branches
                )
            )
            issues.append(
                "[complete-source-unit:tests] Source-stated exceptions or "
                "applicability conditions require paired positive/blocking cases "
                "that assert the affected principal output and toggle its "
                "controlling formula selector; missing at "
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
        )
    )
    return tuple(
        {
            (branch.path, branch.start, branch.end): branch for branch in controlled
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
    principal_rules: dict[str, dict[str, Any]],
    principal_formula_clause_rules: dict[SourceStructureBranch, set[str]],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
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
            numeric_value_is_grounded=numeric_value_is_grounded,
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
    numeric_value_is_grounded: NumericGroundingPredicate,
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
            if (
                execution is None
                or not _formula_execution_leaf_is_computational(execution)
                or not _formula_execution_matches_source_branch(
                    execution,
                    branch,
                    interval=interval,
                    formula_environment=formula_environment,
                    extract_numeric_occurrences=extract_numeric_occurrences,
                    numeric_value_is_grounded=numeric_value_is_grounded,
                )
            ):
                continue
            if interval is None:
                witness = (
                    "leaf:"
                    + _formula_leaf_semantic_key(
                        execution.leaf,
                        formula_environment=execution.constant_environment,
                    )
                    if has_branching_formula
                    else f"case:{id(case)}"
                )
                witnesses.add((rule_name, witness))
                continue
            if (
                _formula_execution_references_names(
                    execution,
                    selector_names,
                )
                and selector_names
                and any(
                    _interval_contains(interval, value)
                    for value in _case_numeric_selector_values(
                        case,
                        selector_names,
                        dependency_environment=dependency_environment,
                    )
                )
            ):
                witnesses.add((rule_name, f"case:{id(case)}"))
    return witnesses


def _formula_execution_matches_source_branch(
    execution: _FormulaExecution,
    branch: SourceStructureBranch,
    *,
    interval: _NumericInterval | None,
    formula_environment: dict[str, Any],
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
    source_topology = _explicit_source_arithmetic_topology(branch.text)
    if source_topology is not None and source_topology != _formula_arithmetic_topology(
        operative_leaf,
        environment=binding_environment,
    ):
        return False
    artifact_operations = _formula_ast_operation_kinds(operative_leaf)
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
    source_occurrences = tuple(
        extract_numeric_occurrences(authoritative_numeric_recall_text(branch.text))
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
    computation_occurrences = tuple(
        occurrence
        for occurrence in source_occurrences
        if not any(
            _numeric_occurrences_are_equivalent(occurrence, boundary)
            for boundary in boundaries
        )
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
            r"\b(?:multiplied|multiply|multiplication|multipliziert|"
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


def _formula_ast_operation_kinds(text: str) -> set[str]:
    operations: set[str] = set()
    try:
        expression = ast.parse(text.strip(), mode="eval").body
    except SyntaxError:
        return operations
    for node in ast.walk(expression):
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
    with contextlib.suppress(SyntaxError):
        expression = ast.parse(text.strip(), mode="eval").body
        simplified = _simplify_formula_expression(
            expression,
            environment=environment,
        )
        return ast.unparse(ast.fix_missing_locations(simplified))
    return text


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
) -> int | float | None:
    value = _evaluate_condition_expression(expression, environment)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
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
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
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
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(
            float(left),
            float(right),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
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
                    isinstance(evaluated, (int, float))
                    and not isinstance(evaluated, bool)
                    and evaluated == 0
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
                            chain_header.start("condition") : chain_header.end(
                                "condition"
                            )
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
            condition for _line_index, kind, condition in headers if kind != "else"
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
            (line[header.start("selector") : header.end("selector")].strip(),),
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
        expected = _match_arm_value(pattern.strip(), environment=environment)
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
        expression = ast.parse(selector.strip(), mode="eval").body
    except SyntaxError:
        try:
            expression = ast.parse(f"(\n{selector.strip()}\n)", mode="eval").body
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


def _match_arm_value(
    value: str,
    *,
    environment: dict[str, Any],
) -> Any:
    if value == "true":
        return True
    if value == "false":
        return False
    with contextlib.suppress(SyntaxError, ValueError):
        return ast.literal_eval(value)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        return environment.get(value, _UNRESOLVED_CONDITION_VALUE)
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
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == 0
        )
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
        selector_names = _rule_numeric_selector_names(rule)
        if not selector_names:
            continue
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
            if execution is not None and any(
                numeric_value_is_grounded(value, (boundary,))
                and _formula_execution_references_names(
                    execution,
                    input_names,
                )
                and _formula_execution_binds_boundary(
                    execution,
                    boundary,
                    input_names=input_names,
                    formula_environment=execution.constant_environment,
                    source_interval=source_interval,
                    source_boolean_polarity=source_boolean_polarity,
                    extract_numeric_occurrences=(extract_numeric_occurrences),
                    numeric_value_is_grounded=(numeric_value_is_grounded),
                )
                and _boundary_case_changes_formula_effect(
                    rule,
                    case,
                    input_key=input_key,
                    selector_names=input_names,
                    boundary_value=value,
                    execution=execution,
                    principal_rules=principal_rules,
                    dependency_names=set(dependency_environment),
                    formula_environment=formula_environment or {},
                    source_interval=source_interval,
                    source_boolean_polarity=source_boolean_polarity,
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
    return witnesses


def _branch_boundary_has_test_evidence(
    branch: SourceStructureBranch,
    boundary: NumericOccurrenceLike,
    *,
    principal_rules: dict[str, dict[str, Any]],
    principal_rule_paths: dict[str, set[tuple[str, ...]]],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    numeric_value_is_grounded: NumericGroundingPredicate,
    formula_environment: dict[str, Any] | None = None,
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
            extract_numeric_occurrences=extract_numeric_occurrences,
        )
    )


def _formula_execution_binds_boundary(
    execution: _FormulaExecution,
    boundary: NumericOccurrenceLike,
    *,
    input_names: set[str],
    formula_environment: dict[str, Any],
    source_interval: _NumericInterval | None,
    source_boolean_polarity: int,
    extract_numeric_occurrences: NumericOccurrenceExtractor | None,
    numeric_value_is_grounded: NumericGroundingPredicate,
) -> bool:
    boundary_names = {
        name
        for name, value in formula_environment.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (
            numeric_value_is_grounded(float(value), (boundary,))
            or _is_adjacent_integral_boundary(float(value), boundary)
        )
    }
    if input_names & boundary_names:
        return False
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
) -> bool:
    with contextlib.suppress(SyntaxError):
        expression = ast.parse(text.strip(), mode="eval").body
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
    return False


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
        if isinstance(value, (int, float)) and not isinstance(value, bool):
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
) -> bool:
    inputs = case.get("input")
    if not isinstance(inputs, dict):
        return False
    boundary_step = (
        1.0
        if float(boundary_value).is_integer()
        else max(abs(boundary_value) * 1e-6, 1e-9)
    )
    signature = _formula_execution_effect_signature(execution)
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
            if (
                candidate_execution is not None
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
                != signature
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


def _formula_interval_from_text(
    text: str,
    *,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
) -> _NumericInterval | None:
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
    keyword = re.search(
        r"\b(?:zwischen|between|von\s+(?:mehr|weniger)\s+als|"
        r"mehr\s+als|weniger\s+als|"
        r"von(?!\s+(?:mehr\s+als|weniger\s+als|höchstens|mindestens|"
        r"nicht\s+mehr\s+als|über|unter))|"
        r"from|unter|less\s+than|below|"
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
    occurrences = tuple(
        occurrence
        for occurrence in extract_numeric_occurrences(text)
        if occurrence.start >= keyword.start()
    )
    if not occurrences:
        return None
    first_gap = text[keyword.end() : occurrences[0].start]
    if not re.fullmatch(
        r"\s*(?:(?:zu|bis)\s+)?"
        r"(?:(?:einschließlich|maximal|inklusive|including|maximum)\s+)?"
        r"(?:(?:einem?|einer|dem|der|das)\s+)?"
        r"(?:(?:zu\s+versteuernd\w*|maßgeblich\w*)\s+)?"
        r"(?:(?:einkommen|betrag|wert|income|amount)\s+)?"
        r"(?:(?:von|of)\s+)?"
        r"(?:(?:einschließlich|maximal|inklusive|including|maximum)\s+)?",
        first_gap,
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
        return _NumericInterval(occurrences[0], True, occurrences[1], True)
    if re.match(r"(?:von|from)\b", lowered_range) and re.search(
        r"\b(?:bis|to|through)\b",
        lowered_range,
    ):
        if len(occurrences) < 2:
            return None
        return _NumericInterval(occurrences[0], True, occurrences[1], True)
    if re.match(
        r"(?:unter|less\s+than|below|(?:von\s+)?weniger\s+als)\b",
        lowered_range,
    ):
        return _NumericInterval(None, False, occurrences[0], False)
    if re.match(
        r"(?:bis|up\s+to|höchstens|nicht\s+mehr\s+als|at\s+most)\b",
        lowered_range,
    ):
        return _NumericInterval(None, False, occurrences[0], True)
    if re.match(
        r"(?:(?:von\s+)?mehr\s+als|über|more\s+than|above)\b",
        lowered_range,
    ):
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
        value < lower or (not interval.lower_inclusive and math.isclose(value, lower))
    ):
        return False
    if upper is not None and (
        value > upper or (not interval.upper_inclusive and math.isclose(value, upper))
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
        or any(path == branch.path[: len(path)] for path in paths if path)
    }
    requirement = _source_exception_effect_requirement(branch.text)
    condition_text = _source_exception_condition_text(branch.text)
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
                witness.active_value
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
    marker = next(iter(_source_exception_or_applicability_matches(clause)), None)
    if marker is None:
        return clause
    suffix = clause[marker.start() :]
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
    if _exception_reverses_negative_proposition(collapsed):
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
            r"\bkein(?:e|en|em|er|es)?\s+(?:anspruch|berechtigung)\b",
            proposition,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _source_exception_selector_is_relevant(text: str, name: str) -> bool:
    """Reject formula toggles with no semantic link to the source exception."""

    normalized_name = _normalized_selector_name(name)
    if _exception_semantic_identifier(normalized_name):
        return True
    collapsed = _collapse_text(text).lower()
    if _source_selector_concept_matches(collapsed, normalized_name):
        return True
    ignored = {
        "applies",
        "apply",
        "case",
        "condition",
        "flag",
        "has",
        "is",
        "status",
        "the",
    }
    return any(
        len(token) >= 4
        and token not in ignored
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
        "qualified": r"\b(?:qualified|berechtig|anspruch)\w*",
        "exempt": r"\b(?:exempt|befrei)\w*",
        "exception": r"\b(?:exception|ausnahme)\w*",
        "surcharge": r"\b(?:surcharge|zuschlag)\w*",
        "status": r"\bstatus\w*",
    }
    for concept, pattern in concept_patterns.items():
        if concept in normalized_name:
            patterns.append(pattern)
    ignored = {
        "applies",
        "apply",
        "case",
        "condition",
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
    patterns.extend(
        rf"\b{re.escape(token)}\w*"
        for token in normalized_name.split("_")
        if len(token) >= 4 and token not in ignored
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
            re.search(
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
    toggled.update(
        _toggled_formula_numeric_selectors(
            principal_rules,
            asserted_by_rule=asserted_by_rule,
            formula_environment=formula_environment,
        )
    )
    return toggled


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
                if _normalized_case_period(left_case) != _normalized_case_period(
                    right_case
                ) or not _cases_differ_by_one_input(left_case, right_case):
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
                if (
                    left_execution.trace or right_execution.trace
                ) and _formula_leaf_semantic_key(
                    left_execution.leaf,
                    formula_environment=left_execution.constant_environment,
                ) == _formula_leaf_semantic_key(
                    right_execution.leaf,
                    formula_environment=right_execution.constant_environment,
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
                        _formula_execution_reaches_selector(
                            left_execution,
                            selector_name,
                        )
                        and _formula_execution_reaches_selector(
                            right_execution,
                            selector_name,
                        )
                    ):
                        continue
                    for (
                        ordinary_runtime,
                        exception_runtime,
                        ordinary_value,
                        exception_value,
                    ) in (
                        (
                            left_runtime,
                            right_runtime,
                            float(left_value),
                            float(right_value),
                        ),
                        (
                            right_runtime,
                            left_runtime,
                            float(right_value),
                            float(left_value),
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
                            )
                        )
    return witnesses


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


def _exception_effect_is_zero(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isclose(float(value), 0.0, abs_tol=1e-12)
    )


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
        _value_type, raw_value = execution.evaluated_value
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
        isinstance(ordinary, (int, float))
        and not isinstance(ordinary, bool)
        and isinstance(blocking, (int, float))
        and not isinstance(blocking, bool)
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
        value_type, raw_value = execution.evaluated_value
        if value_type in {"int", "float"}:
            with contextlib.suppress(InvalidOperation):
                return "evaluated-number", Decimal(raw_value)
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

    formula_text = _rule_formula_text(rule)
    selector_names: set[str] = set()

    def record(name: str) -> None:
        selector_names.add(name)

    def inspect_expression(text: str) -> None:
        with contextlib.suppress(SyntaxError):
            expression = ast.parse(text.strip(), mode="eval").body
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


def _exception_semantic_identifier(identifier: str) -> bool:
    return bool(
        re.search(
            r"(?:exception|exempt|exclud|disqual|inelig|remarri|"
            r"barred|blocking|waiver|befrei|ausnahme)",
            identifier,
            flags=re.IGNORECASE,
        )
    )


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
                not isinstance(operand_value, (int, float))
                or isinstance(operand_value, bool)
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
        if isinstance(value, bool) or not isinstance(value, (int, float)):
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
        if (
            isinstance(changed_value, (int, float))
            and not isinstance(changed_value, bool)
            and not math.isclose(float(changed_value), operand_value)
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
        direct_text = authoritative_numeric_recall_text(branch.text)
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
    if not isinstance(versions, list):
        return None
    unambiguous = _unambiguous_rule_formula_text(rule)
    formula_versions = [
        version
        for version in versions
        if isinstance(version, dict) and version.get("formula") is not None
    ]
    dated_versions = [
        version
        for version in formula_versions
        if re.fullmatch(
            r"\d{4}-\d{2}-\d{2}",
            str(version.get("effective_from") or "").strip(),
        )
    ]
    has_temporal_metadata = any(
        str(version.get("effective_from") or "").strip()
        or str(version.get("effective_to") or "").strip()
        for version in formula_versions
    )
    if has_temporal_metadata and (
        len(dated_versions) != len(formula_versions)
        or any(
            effective_to and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", effective_to)
            for version in formula_versions
            if (effective_to := str(version.get("effective_to") or "").strip())
        )
    ):
        return None
    period = _normalized_case_period(case)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", period):
        if "period" not in case:
            return unambiguous
        return None if has_temporal_metadata else unambiguous
    if not dated_versions:
        return unambiguous
    candidates: list[tuple[str, str]] = []
    for version in dated_versions:
        effective_from = str(version.get("effective_from") or "").strip()
        effective_to = str(version.get("effective_to") or "").strip()
        if effective_from > period or (effective_to and effective_to < period):
            continue
        candidates.append((effective_from, str(version["formula"])))
    if not candidates:
        return None
    latest = max(effective_from for effective_from, _formula in candidates)
    formulas = {
        formula for effective_from, formula in candidates if effective_from == latest
    }
    return next(iter(formulas)) if len(formulas) == 1 else None


def _constant_rule_environment(payload: dict[str, Any]) -> dict[str, Any]:
    """Return literal rule values, retaining resolvable temporal variants."""

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
        formula_version_count = sum(
            1
            for version in versions
            if isinstance(version, dict) and "formula" in version
        )
        entries: list[tuple[str, str, Any]] = []
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
                    entries.append(
                        (
                            str(version.get("effective_from") or "").strip(),
                            str(version.get("effective_to") or "").strip(),
                            value,
                        )
                    )
        if len(entries) != formula_version_count:
            continue
        values = [value for _start, _end, value in entries]
        if values and all(
            type(value) is type(values[0]) and value == values[0]
            for value in values[1:]
        ):
            environment[name] = values[0]
        elif entries and all(
            re.fullmatch(r"\d{4}-\d{2}-\d{2}", start)
            and (not end or re.fullmatch(r"\d{4}-\d{2}-\d{2}", end))
            for start, end, _value in entries
        ):
            environment[name] = _TemporalFormulaValue(tuple(entries))
    return environment


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
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", period):
            continue
        candidates = [
            (start, candidate)
            for start, end, candidate in value.versions
            if start <= period and (not end or period <= end)
        ]
        if not candidates:
            continue
        latest = max(start for start, _candidate in candidates)
        latest_values = [
            candidate for start, candidate in candidates if start == latest
        ]
        if latest_values and all(
            type(candidate) is type(latest_values[0]) and candidate == latest_values[0]
            for candidate in latest_values[1:]
        ):
            resolved[name] = latest_values[0]
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


def _collapse_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _bounded_source_feedback_excerpt(value: str, *, limit: int = 360) -> str:
    """Render bounded source-identifying feedback without changing its words."""

    collapsed = _collapse_text(value).replace("`", "\\`")
    if len(collapsed) <= limit:
        return collapsed
    marker = " ... "
    available = limit - len(marker)
    head_length = (available * 2) // 3
    tail_length = available - head_length
    return collapsed[:head_length].rstrip() + marker + collapsed[-tail_length:].lstrip()


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
