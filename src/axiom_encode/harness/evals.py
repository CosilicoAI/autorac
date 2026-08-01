"""Model comparison evals for statute and corpus-backed policy encoding."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Iterator, Literal, Sequence

import requests
import yaml

from axiom_encode.codex_cli import resolve_codex_cli
from axiom_encode.concepts.jurisdiction import jurisdiction_prefix
from axiom_encode.concepts.registry import (
    Concept,
    ConceptRegistry,
    load_concept_registry,
)
from axiom_encode.constants import DEFAULT_OPENAI_MODEL
from axiom_encode.prompts.encoder import SOURCE_SCOPE_PROTOCOL
from axiom_encode.repo_routing import (
    canonical_rulespec_repo_name,
    find_policy_repo_root,
    jurisdiction_content_dir,
    jurisdiction_subdir_names,
    monorepo_alternative_path,
)
from axiom_encode.statute import (
    CitationParts,
    citation_to_citation_path,
    citation_to_relative_rulespec_path,
    parse_usc_citation,
)

from .dependency_stubs import (
    ResolvedCanonicalConcept,
    ResolvedDefinedTerm,
    import_target_to_relative_rulespec_path,
    materialize_registered_stub,
    resolve_canonical_concepts_from_text,
    resolve_defined_terms_from_text,
)
from .encoding_db import TokenUsage
from .eval_prompt_surface import (
    render_date_silent_scaffold_guidance,
    render_single_amount_row_guidance,
    render_uk_legislation_guidance,
)
from .observability import emit_eval_result, extract_reasoning_output_tokens
from .pricing import estimate_usage_cost_usd
from .validator_pipeline import (
    ValidationResult,
    ValidatorPipeline,
    _candidate_local_corpus_provision_files,
    _fetch_supabase_corpus_source_text,
    _local_corpus_record_text,
    _read_local_corpus_provision_file,
    _source_text_looks_like_table,
    extract_embedded_source_text,
    extract_grounding_values,
    extract_named_scalar_occurrences,
    extract_numbers_from_text,
    extract_numeric_grounding_source_text,
    extract_numeric_occurrences_from_text,
    find_deferred_output_issues,
    find_ungrounded_numeric_issues,
    find_unused_import_issues,
    find_unused_modifier_parameter_issues,
    numeric_value_is_grounded,
    repair_copied_cross_reference_summary,
    repair_formula_let_bindings,
    repair_source_table_band_scalar_parameters,
    repair_source_table_interval_row_alignment,
    repair_source_table_interval_tests,
    repair_source_table_open_ended_bound_sentinels,
    repair_unsupported_chained_conditionals,
)

EvalMode = Literal["cold", "repo-augmented"]
EvalOracleMode = Literal["none", "policyengine", "all"]
IMPORT_ITEM_PATTERN = re.compile(r"^\s*-\s*(['\"]?)([^'\"]+?)\1\s*$")
TABLE_BOUND_COMPARATOR_NUMBER_PATTERN = re.compile(
    r"(?:(?:<=|>=|<|>|==)\s*(-?[\d,]+(?:\.\d+)?)"
    r"|(-?[\d,]+(?:\.\d+)?)\s*(?:<=|>=|<|>|==))"
)
SUPPORTED_EVAL_ENTITIES = (
    "Payment",
    "Person",
    "TaxUnit",
    "Household",
    "Family",
    "TanfUnit",
    "SnapUnit",
    "SPMUnit",
    "Corporation",
    "Business",
    "Employer",
    "Asset",
    "StateAgency",
)
ADMINISTRATIVE_EVAL_ENTITIES = {
    "stateagency",
    "snapqualitycontrolfiscalyear",
    "administrativeclaim",
    "stateagencyappeal",
    "bonusaward",
}
SUPPORTED_EVAL_PERIODS = ("Year", "Month", "Week", "Day")
SUPPORTED_EVAL_DTYPES = (
    "Money",
    "Rate",
    "Boolean",
    "Integer",
    "Count",
    "String",
    "Decimal",
    "Float",
)
_PURE_NUMERIC_EXPRESSION_PATTERN = re.compile(r"^[\d\s()+\-*/.,]+$")
_ISO_WEEK_PERIOD_PATTERN = re.compile(r"^\d{4}-W\d{2}(?:-\d)?$")
_ADMIN_AGENCY_AGGREGATE_SUBJECT_PATTERN = re.compile(
    r"\b(?:FNS|State\s+agenc(?:y|ies)|State(?:'s)?\s+administration|"
    r"Federal\s+(?:reviewer|case\s+reviews?|subsample)|"
    r"national\s+performance\s+measure)\b",
    flags=re.IGNORECASE,
)
_ADMIN_AGENCY_AGGREGATE_MEASURE_PATTERN = re.compile(
    r"\b(?:active\s+case|negative\s+case|payment\s+error|negative\s+error|"
    r"error\s+rates?|quality\s+control\s+sample|rereview\s+sample|"
    r"regress(?:ed|ion)|subsample|sample\s+size|liabilit(?:y|ies)|"
    r"waiver\s+of\s+liability|at-risk|new\s+investment|caseload\s+growth|"
    r"high\s+performance\s+bonuses?|bonus\s+payments?|program\s+access\s+index|"
    r"application\s+processing\s+timeliness)\b",
    flags=re.IGNORECASE,
)
_ADMIN_AGENCY_BONUS_USE_RESTRICTION_PATTERN = re.compile(
    r"\b(?:bonus\s+award\s+money|bonus\s+payments?)\b[\s\S]{0,160}"
    r"\b(?:shall\s+be\s+used\s+only|shall\s+not\s+be\s+used|"
    r"household\s+benefits?|incentive\s+payments?|"
    r"SNAP-related\s+expenses?)\b",
    flags=re.IGNORECASE,
)
_ADMIN_AGENCY_AGGREGATE_CONTEXT_WINDOW = 500
_CONDITIONAL_AMOUNT_SLICE_PATTERN = re.compile(
    r"\b(?:if|where|unless|except|subject to|treated as paid)\b",
    re.IGNORECASE,
)
_LOCAL_IMPORT_ROOT_TOKENS = {"legislation", "statutes", "regulation"}
_RULESPEC_SOURCE_ROOT_TOKENS = {
    "form",
    "forms",
    "guidance",
    "manual",
    "manuals",
    "policies",
    "policy",
    "regulation",
    "regulations",
    "statute",
    "statutes",
}
_CORPUS_DOCUMENT_CLASS_BY_SOURCE_TOKEN = {
    "form": "form",
    "forms": "form",
    "guidance": "guidance",
    "manual": "manual",
    "manuals": "manual",
    "policies": "policy",
    "policy": "policy",
    "regulation": "regulation",
    "regulations": "regulation",
    "statute": "statute",
    "statutes": "statute",
}
_RULESPEC_OUTPUT_ROOT_BY_SOURCE_TOKEN = {
    "form": "policies",
    "forms": "policies",
    "guidance": "policies",
    "manual": "policies",
    "manuals": "policies",
    "policies": "policies",
    "policy": "policies",
    "regulation": "regulations",
    "regulations": "regulations",
    "statute": "statutes",
    "statutes": "statutes",
}
_UK_LEGISLATION_DOMAIN_TOKEN = "legislation.gov.uk"
_UK_LEGISLATION_SECTION_TOKENS = {"article", "regulation", "section"}
_CODEX_DEFAULT_TIMEOUT_SECONDS = 600
_CODEX_DEFAULT_IDLE_TIMEOUT_SECONDS = 300
_CODEX_LONG_SOURCE_CHAR_THRESHOLD = 40_000
_CODEX_LONG_SOURCE_TIMEOUT_SECONDS = 1800
_CODEX_LONG_SOURCE_IDLE_TIMEOUT_SECONDS = 900
_POLICYENGINE_HINT_BROAD_PLACEHOLDER_RE = re.compile(
    r"\b(?:"
    r"person_is_described_in_[A-Za-z0-9_]*"
    r"|person_is_not_described_in_or_enrolled_under_[A-Za-z0-9_]*"
    r"|person_is_in_[A-Za-z0-9_]*category[A-Za-z0-9_]*"
    r"|person_[A-Za-z0-9_]*listed_state_plan[A-Za-z0-9_]*"
    r"|person_[A-Za-z0-9_]*(?:mandatory|optional|cost_sharing|ssi_related)"
    r"[A-Za-z0-9_]*group[A-Za-z0-9_]*"
    r"|person_[A-Za-z0-9_]*(?:subparagraph|subclause|subsection|section)"
    r"[A-Za-z0-9_]*"
    r"|person_[A-Za-z0-9_]*as_defined_in_[A-Za-z0-9_]*"
    r"|person_covered_by_[A-Za-z0-9_]*(?:category|subparagraph)[A-Za-z0-9_]*"
    r"|person_is_qualified_[A-Za-z0-9_]*group[A-Za-z0-9_]*"
    r"|income_(?:as_)?determined_(?:under|for)_[A-Za-z0-9_]*"
    r"|[A-Za-z0-9_]*subsection_[A-Za-z0-9_]*requirements"
    r"(?:_[A-Za-z0-9_]+)?_satisfied"
    r"|[A-Za-z0-9_]*subject_to_subsections?_[A-Za-z0-9_]*_satisfied"
    r")\b"
)
_RULESPEC_FORMULA_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_RULESPEC_FORMULA_KEYWORDS = frozenset(
    {
        "and",
        "else",
        "false",
        "if",
        "in",
        "not",
        "or",
        "true",
    }
)


def _is_rulespec_local_identifier(value: str | None) -> bool:
    return bool(value and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value))


def _matching_numeric_occurrence_count(
    occurrences: Counter[float],
    value: float,
) -> int:
    """Return whether a named scalar covers the source value.

    This is a value-coverage gate, not a duplicate-definition gate. One named
    scalar should cover repeated source mentions of the same amount or deadline.
    """
    return int(
        any(
            numeric_value_is_grounded(occurrence_value, {value})
            for occurrence_value in occurrences
        )
    )


_DECIMAL_PERCENT_CONCEPT_NAME_RE = re.compile(
    r"(?<!\d)(?P<integer>\d{1,3})_(?P<fraction>\d{1,4})_percent\b",
    re.IGNORECASE,
)
_CONCEPT_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _numeric_occurrences_from_concept_text(text: str) -> Counter[float]:
    """Extract numeric semantics from concept names and labels.

    This credits names such as ``under_18`` or ``130_percent`` when a source
    number is represented by an imported predicate/table rather than a local
    scalar formula.
    """
    normalized = _DECIMAL_PERCENT_CONCEPT_NAME_RE.sub(
        lambda match: f"{match.group('integer')}.{match.group('fraction')} percent",
        text,
    )
    normalized = normalized.replace("_", " ").replace("-", " ")
    return Counter(extract_numeric_occurrences_from_text(normalized))


def _numeric_concept_name_occurrences(content: str) -> Counter[float]:
    """Count source numeric values represented in local rule/input names."""
    occurrences: Counter[float] = Counter()
    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if not (
            isinstance(payload, dict)
            and payload.get("format") == "rulespec/v1"
            and isinstance(payload.get("rules"), list)
        ):
            return occurrences
        for rule in payload["rules"]:
            if not isinstance(rule, dict):
                continue
            name = str(rule.get("name") or "").strip()
            if name:
                occurrences.update(_numeric_occurrences_from_concept_text(name))
            versions = rule.get("versions")
            if not isinstance(versions, list):
                continue
            for version in versions:
                if not isinstance(version, dict):
                    continue
                formula = version.get("formula")
                if not isinstance(formula, str):
                    continue
                for identifier in set(_CONCEPT_IDENTIFIER_RE.findall(formula)):
                    occurrences.update(
                        _numeric_occurrences_from_concept_text(identifier)
                    )
    return occurrences


def _verification_value_numeric_occurrences(content: str) -> Counter[float]:
    """Count source values recorded in RuleSpec verification payloads."""
    occurrences: Counter[float] = Counter()

    def visit(value: object) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            occurrences.update([float(value)])
            return
        if isinstance(value, str):
            stripped = value.strip().replace(",", "")
            with contextlib.suppress(ValueError):
                occurrences.update([float(stripped)])
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if isinstance(value, list):
            for item in value:
                visit(item)

    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if not (
            isinstance(payload, dict)
            and payload.get("format") == "rulespec/v1"
            and isinstance(payload.get("rules"), list)
        ):
            return occurrences
        for rule in payload["rules"]:
            if not isinstance(rule, dict):
                continue
            verification = rule.get("verification")
            if not isinstance(verification, dict):
                continue
            visit(verification.get("values"))
    return occurrences


def _deferred_output_numeric_occurrences(content: str) -> Counter[float]:
    """Count source values explicitly scoped to deferred output reasons."""
    occurrences: Counter[float] = Counter()
    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if not (
            isinstance(payload, dict)
            and payload.get("format") == "rulespec/v1"
            and isinstance(payload.get("module"), dict)
            and isinstance(payload["module"].get("deferred_outputs"), list)
        ):
            return occurrences
        for deferred_output in payload["module"]["deferred_outputs"]:
            if not isinstance(deferred_output, dict):
                continue
            reason = deferred_output.get("reason")
            if not isinstance(reason, str):
                continue
            occurrences.update(
                extract_numeric_occurrences_from_text(
                    _numeric_occurrence_source_text(reason)
                )
            )
    return occurrences


_SECTION_CROSS_REFERENCE_PATTERN = re.compile(
    r"\b(?:sections?|secs?\.?|regs?\.?|regulations?|paragraphs?)\s+"
    r"\d+(?:\.\d+)+(?:\s*(?:through|to|-|and|,)\s*\d+(?:\.\d+)+)*",
    re.IGNORECASE,
)
_LEADING_ZERO_MANUAL_SECTION_PATTERN = re.compile(r"\b0\d{3}\.\d{2}(?:\.\d{2})?\b")


def _numeric_occurrence_source_text(source_text: str) -> str:
    """Drop citation-like cross-references before source numeric coverage checks."""
    without_cross_references = _SECTION_CROSS_REFERENCE_PATTERN.sub("", source_text)
    return _LEADING_ZERO_MANUAL_SECTION_PATTERN.sub("", without_cross_references)


_UNREFERENCED_PROOF_IMPORT_RE = re.compile(
    r"Proof import not referenced:\s*`(?P<rule>[^`]+)`\s+"
    r"proof imports\s+`(?P<symbol>[^`]+)`"
)
_UNUSED_IMPORT_RE = re.compile(r"Unused import `(?P<target>[^`]+)`")


def _strip_yaml_scalar_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _yaml_list_item_block_end(
    lines: list[str],
    *,
    start: int,
    start_indent: int,
) -> int:
    index = start + 1
    while index < len(lines):
        line = lines[index]
        if line.strip():
            indent = len(line) - len(line.lstrip())
            if indent <= start_indent:
                break
        index += 1
    return index


def _proof_import_atom_block_imported_symbol(block: str) -> str:
    if not re.search(r"(?m)^\s*kind:\s*import\s*$", block):
        return ""
    output_match = re.search(r"(?m)^\s*output:\s*(.+?)\s*$", block)
    if output_match:
        return _strip_yaml_scalar_quotes(output_match.group(1).strip())
    target_match = re.search(r"(?m)^\s*target:\s*(.+?)\s*$", block)
    if not target_match:
        return ""
    target = _strip_yaml_scalar_quotes(target_match.group(1).strip())
    if "#" not in target:
        return ""
    return target.rsplit("#", 1)[1].strip()


def _remove_unreferenced_proof_import_atom_blocks(
    content: str,
    *,
    stale_pairs: set[tuple[str, str]],
) -> tuple[str, list[str]]:
    lines = content.splitlines(keepends=True)
    repaired_lines: list[str] = []
    removed: list[str] = []
    current_rule = ""
    index = 0

    while index < len(lines):
        line = lines[index]
        rule_match = re.match(r"^\s*-\s+name:\s*(.+?)\s*$", line)
        if rule_match:
            current_rule = _strip_yaml_scalar_quotes(rule_match.group(1).strip())

        item_match = re.match(r"^(\s*)-\s+path:\s*", line)
        if current_rule and item_match:
            block_end = _yaml_list_item_block_end(
                lines,
                start=index,
                start_indent=len(item_match.group(1)),
            )
            block = "".join(lines[index:block_end])
            imported_symbol = _proof_import_atom_block_imported_symbol(block)
            if imported_symbol and (current_rule, imported_symbol) in stale_pairs:
                removed.append(f"{current_rule}:{imported_symbol}")
                index = block_end
                continue

        repaired_lines.append(line)
        index += 1

    return "".join(repaired_lines), removed


def _unused_import_items(content: str) -> list[str]:
    items: list[str] = []
    for issue in find_unused_import_issues(content):
        match = re.search(r"Unused import `([^`]+)`", issue)
        if match:
            items.append(match.group(1))
    return items


def _prune_unused_imports(content: str) -> tuple[str, list[str]]:
    unused_imports = _unused_import_items(content)
    if not unused_imports:
        return content, []

    unused = set(unused_imports)
    repaired_lines: list[str] = []
    in_imports = False
    imports_indent = 0
    removed: list[str] = []

    for line in content.splitlines(keepends=True):
        imports_match = re.match(r"^(\s*)imports:\s*$", line)
        if imports_match:
            in_imports = True
            imports_indent = len(imports_match.group(1))
            repaired_lines.append(line)
            continue

        if in_imports:
            stripped = line.strip()
            if stripped:
                indent = len(line) - len(line.lstrip())
                item_match = re.match(r"^\s*-\s+(.+?)\s*$", line)
                if item_match and indent >= imports_indent:
                    item = _strip_yaml_scalar_quotes(item_match.group(1).strip())
                    if item in unused:
                        removed.append(item)
                        continue
                elif indent <= imports_indent:
                    in_imports = False

        repaired_lines.append(line)

    return "".join(repaired_lines), removed


def _remove_unreferenced_proof_import_atoms(
    rules_file: Path,
    issues: Sequence[str],
) -> list[str]:
    if not rules_file.exists():
        return []
    stale_pairs: set[tuple[str, str]] = set()
    for issue in issues:
        for match in _UNREFERENCED_PROOF_IMPORT_RE.finditer(str(issue)):
            stale_pairs.add((match["rule"].strip(), match["symbol"].strip()))
    if not stale_pairs:
        return []

    original_content = rules_file.read_text()
    try:
        payload = yaml.safe_load(original_content) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return []
    if not isinstance(payload, dict):
        return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    repaired_content, removed = _remove_unreferenced_proof_import_atom_blocks(
        original_content,
        stale_pairs=stale_pairs,
    )
    if not removed:
        return []
    repaired_content, pruned_imports = _prune_unused_imports(repaired_content)
    rules_file.write_text(repaired_content)
    return sorted([*removed, *[f"unused_import:{item}" for item in pruned_imports]])


def _prune_unused_imports_from_file(
    rules_file: Path,
    issues: Sequence[str],
) -> list[str]:
    if not rules_file.exists():
        return []
    if not any(_UNUSED_IMPORT_RE.search(str(issue)) for issue in issues):
        return []

    original_content = rules_file.read_text()
    repaired_content, removed = _prune_unused_imports(original_content)
    if not removed or repaired_content == original_content:
        return []
    rules_file.write_text(repaired_content)
    return sorted(removed)


def _is_empty_nonassertable_artifact(content: str) -> bool:
    """Return true for intentionally non-executable artifacts with no rules."""
    try:
        payload = yaml.safe_load(content)
    except (ValueError, yaml.YAMLError):
        return False
    if not isinstance(payload, dict):
        return False
    module = payload.get("module")
    status = (
        str(module.get("status", "")).strip()
        if isinstance(module, dict)
        else str(payload.get("status", "")).strip()
    )
    return status in {"deferred", "entity_not_supported"} and not payload.get("rules")


def _extract_proof_source_excerpt_text(content: str) -> str:
    """Return source excerpts from proof atoms for numeric grounding."""
    with contextlib.suppress(ValueError, TypeError, yaml.YAMLError):
        payload = yaml.safe_load(content)
        if not isinstance(payload, dict):
            return ""
        rules = payload.get("rules")
        if not isinstance(rules, list):
            return ""
        excerpts: list[str] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            metadata = rule.get("metadata")
            if not isinstance(metadata, dict):
                continue
            proof = metadata.get("proof")
            if not isinstance(proof, dict):
                continue
            atoms = proof.get("atoms")
            if not isinstance(atoms, list):
                continue
            for atom in atoms:
                if not isinstance(atom, dict):
                    continue
                source = atom.get("source")
                if not isinstance(source, dict):
                    continue
                excerpt = source.get("excerpt")
                if isinstance(excerpt, str) and excerpt.strip():
                    excerpts.append(excerpt.strip())
        return "\n".join(excerpts)
    return ""


def _has_parameter_table_proof_atom(content: str) -> bool:
    """Return true when a RuleSpec artifact grounds values in a source table."""
    with contextlib.suppress(ValueError, TypeError, yaml.YAMLError):
        payload = yaml.safe_load(content)
        if not isinstance(payload, dict):
            return False
        rules = payload.get("rules")
        if not isinstance(rules, list):
            return False
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            metadata = rule.get("metadata")
            if not isinstance(metadata, dict):
                continue
            proof = metadata.get("proof")
            if not isinstance(proof, dict):
                continue
            atoms = proof.get("atoms")
            if not isinstance(atoms, list):
                continue
            if any(
                isinstance(atom, dict) and atom.get("kind") == "parameter_table"
                for atom in atoms
            ):
                return True
    return False


@dataclass(frozen=True)
class EvalRunnerSpec:
    """How to invoke a model in an eval."""

    name: str
    backend: str
    model: str


@dataclass
class GroundingMetric:
    """A numeric grounding decision."""

    line: int
    raw: str
    value: float
    grounded: bool


@dataclass
class EvalArtifactMetrics:
    """Deterministic checks over a produced RuleSpec artifact."""

    compile_pass: bool
    compile_issues: list[str]
    ci_pass: bool
    ci_issues: list[str]
    embedded_source_present: bool
    grounded_numeric_count: int
    ungrounded_numeric_count: int
    grounding: list[GroundingMetric]
    source_numeric_occurrence_count: int = 0
    covered_source_numeric_occurrence_count: int = 0
    missing_source_numeric_occurrence_count: int = 0
    numeric_occurrence_issues: list[str] = field(default_factory=list)
    generalist_review_pass: bool | None = None
    generalist_review_score: float | None = None
    generalist_review_issues: list[str] = field(default_factory=list)
    generalist_review_prompt_sha256: str | None = None
    policyengine_pass: bool | None = None
    policyengine_score: float | None = None
    policyengine_issues: list[str] = field(default_factory=list)
    taxsim_pass: bool | None = None
    taxsim_score: float | None = None
    taxsim_issues: list[str] = field(default_factory=list)


@dataclass
class EvalContextFile:
    """A context file copied into the eval workspace."""

    source_path: str
    workspace_path: str
    import_path: str
    kind: str
    label: str | None = None


@dataclass
class EvalWorkspace:
    """Prepared workspace bundle for an eval run."""

    root: Path
    source_text_file: Path
    manifest_file: Path
    source_metadata_file: Path | None = None
    source_metadata: dict[str, object] | None = None
    context_files: list[EvalContextFile] = field(default_factory=list)


@dataclass(frozen=True)
class CorpusSourceUnit:
    """A normalized source unit resolved from corpus.provisions."""

    requested: str
    citation_path: str
    body: str
    source: Literal["local", "supabase"]


@dataclass(frozen=True)
class PrimarySourceContinuation:
    """A context file that explicitly continues the primary source text."""

    source_path: Path
    corpus_citation_path: str | None
    body: str


@dataclass
class EvalPromptResponse:
    """Raw model output and trace from a prompt-only eval run."""

    text: str
    duration_ms: int
    tokens: TokenUsage | None = None
    estimated_cost_usd: float | None = None
    actual_cost_usd: float | None = None
    trace: dict | None = None
    unexpected_accesses: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class EvalResult:
    """One citation x runner result."""

    citation: str
    runner: str
    backend: str
    model: str
    mode: EvalMode
    output_file: str
    trace_file: str
    context_manifest_file: str
    duration_ms: int
    success: bool
    error: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    reasoning_output_tokens: int
    estimated_cost_usd: float | None
    actual_cost_usd: float | None
    retrieved_files: list[str]
    unexpected_accesses: list[str]
    metrics: EvalArtifactMetrics | None
    generation_prompt_sha256: str | None = None
    retry_count: int = 0

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.metrics is not None:
            data["metrics"] = asdict(self.metrics)
        return data


@dataclass(frozen=True)
class EvalReadinessGates:
    """Thresholds that determine whether a benchmark suite is bulk-ready."""

    min_cases: int = 1
    min_success_rate: float | None = None
    min_compile_pass_rate: float | None = None
    min_ci_pass_rate: float | None = None
    min_zero_ungrounded_rate: float | None = None
    min_generalist_review_pass_rate: float | None = 1.0
    min_policyengine_pass_rate: float | None = None
    max_mean_estimated_cost_usd: float | None = None


@dataclass
class EvalSuiteCase:
    """One manifest entry in an eval suite."""

    kind: Literal["citation", "source"]
    name: str
    mode: EvalMode
    allow_context: list[Path] = field(default_factory=list)
    citation: str | None = None
    source_id: str | None = None
    corpus_citation_path: str | None = None
    policyengine_rule_hint: str | None = None
    oracle: EvalOracleMode = "none"
    policyengine_country: str = "auto"


@dataclass
class EvalSuiteManifest:
    """Manifest describing a benchmark suite and its readiness gates."""

    name: str
    path: Path
    runners: list[str]
    mode: EvalMode
    allow_context: list[Path]
    gates: EvalReadinessGates
    cases: list[EvalSuiteCase]


@dataclass(frozen=True)
class EvalReadinessGateResult:
    """Outcome of one readiness threshold."""

    name: str
    comparator: Literal["min", "max"]
    threshold: float | int
    actual: float | int | None
    passed: bool


@dataclass
class EvalReadinessSummary:
    """Aggregated readiness summary for one runner across a suite."""

    total_cases: int
    success_rate: float
    compile_pass_rate: float
    ci_pass_rate: float
    zero_ungrounded_rate: float
    generalist_review_pass_rate: float
    mean_generalist_review_score: float | None
    policyengine_case_count: int
    policyengine_pass_rate: float | None
    mean_policyengine_score: float | None
    mean_estimated_cost_usd: float | None
    gate_results: list[EvalReadinessGateResult]
    ready: bool


def parse_runner_spec(spec: str) -> EvalRunnerSpec:
    """Parse `[name=]backend:model` into a structured runner spec."""
    alias = ""
    target = spec
    if "=" in spec:
        alias, target = spec.split("=", 1)

    if ":" not in target:
        raise ValueError(
            f"Invalid runner spec '{spec}'. Expected [name=]backend:model."
        )

    backend, model = target.split(":", 1)
    backend = backend.strip()
    model = model.strip()
    name = alias.strip() or re.sub(r"[^a-zA-Z0-9._-]+", "-", f"{backend}-{model}")

    if backend not in {"claude", "codex", "openai"}:
        raise ValueError(f"Unsupported backend '{backend}' in runner spec '{spec}'")

    return EvalRunnerSpec(name=name, backend=backend, model=model)


def run_model_eval(
    citations: list[str],
    runner_specs: list[str],
    output_root: Path,
    policy_path: Path,
    runtime_axiom_rules_path: Path,
    corpus_path: Path,
    mode: EvalMode = "repo-augmented",
    extra_context_paths: list[Path] | None = None,
    include_tests: bool = False,
    skip_reviewers: bool = False,
    policyengine_rule_hint: str | None = None,
) -> list[EvalResult]:
    """Run a deterministic comparison over one or more citations."""
    results: list[EvalResult] = []

    for runner in [parse_runner_spec(spec) for spec in runner_specs]:
        for citation in citations:
            results.append(
                _run_single_eval(
                    citation=citation,
                    runner=runner,
                    output_root=output_root,
                    policy_path=policy_path,
                    runtime_axiom_rules_path=runtime_axiom_rules_path,
                    corpus_path=corpus_path,
                    mode=mode,
                    extra_context_paths=extra_context_paths or [],
                    include_tests=include_tests,
                    skip_reviewers=skip_reviewers,
                    policyengine_rule_hint=policyengine_rule_hint,
                )
            )

    return results


def run_source_eval(
    source_id: str,
    source_text: str,
    runner_specs: list[str],
    output_root: Path,
    policy_path: Path,
    source_metadata_payload: dict[str, object] | None = None,
    runtime_axiom_rules_path: Path | None = None,
    mode: EvalMode = "repo-augmented",
    extra_context_paths: list[Path] | None = None,
    oracle: EvalOracleMode = "none",
    policyengine_country: str = "auto",
    policyengine_rule_hint: str | None = None,
    skip_reviewers: bool = False,
) -> list[EvalResult]:
    """Run a deterministic comparison over one corpus-backed source unit."""
    results: list[EvalResult] = []
    continuations = _primary_source_continuations_from_context_paths(
        extra_context_paths or []
    )
    source_text = _append_primary_source_continuations(source_text, continuations)
    source_metadata_payload = _source_metadata_with_continuations(
        source_metadata_payload,
        continuations,
    )

    for runner in [parse_runner_spec(spec) for spec in runner_specs]:
        results.append(
            _run_single_source_eval(
                source_id=source_id,
                source_text=source_text,
                runner=runner,
                output_root=output_root,
                policy_path=policy_path,
                source_metadata_payload=source_metadata_payload,
                runtime_axiom_rules_path=runtime_axiom_rules_path or policy_path,
                mode=mode,
                extra_context_paths=extra_context_paths or [],
                oracle=oracle,
                policyengine_country=policyengine_country,
                policyengine_rule_hint=policyengine_rule_hint,
                skip_reviewers=skip_reviewers,
            )
        )

    return results


def _sha256_text(text: str | None) -> str | None:
    """Return a stable digest for a prompt or prompt-derived text blob."""
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sum_optional_float(left: float | None, right: float | None) -> float | None:
    if left is None and right is None:
        return None
    return (left or 0.0) + (right or 0.0)


def _sum_token_usage(
    left: TokenUsage | None,
    right: TokenUsage | None,
) -> TokenUsage | None:
    if left is None and right is None:
        return None
    return TokenUsage(
        input_tokens=(left.input_tokens if left else 0)
        + (right.input_tokens if right else 0),
        output_tokens=(left.output_tokens if left else 0)
        + (right.output_tokens if right else 0),
        cache_read_tokens=(left.cache_read_tokens if left else 0)
        + (right.cache_read_tokens if right else 0),
        cache_creation_tokens=(left.cache_creation_tokens if left else 0)
        + (right.cache_creation_tokens if right else 0),
        reasoning_output_tokens=(left.reasoning_output_tokens if left else 0)
        + (right.reasoning_output_tokens if right else 0),
    )


_PRIMARY_SOURCE_CONTINUATION_HEADER_PATTERN = re.compile(
    r"^\s*Primary source continuation\b",
    flags=re.IGNORECASE,
)
_CORPUS_CITATION_PATH_LINE_PATTERN = re.compile(
    r"^\s*Corpus citation path:\s*(?P<path>\S+)\s*$",
    flags=re.IGNORECASE,
)


def _primary_source_continuations_from_context_paths(
    extra_context_paths: Sequence[Path],
) -> list[PrimarySourceContinuation]:
    """Return explicit primary-source continuations supplied as context files."""
    continuations: list[PrimarySourceContinuation] = []
    for raw_path in extra_context_paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        try:
            raw_text = path.read_text()
        except OSError:
            continue
        lines = raw_text.splitlines()
        first_nonempty = next((line for line in lines if line.strip()), "")
        if not _PRIMARY_SOURCE_CONTINUATION_HEADER_PATTERN.match(first_nonempty):
            continue

        corpus_citation_path: str | None = None
        body_lines: list[str] = []
        for line in lines:
            if _PRIMARY_SOURCE_CONTINUATION_HEADER_PATTERN.match(line):
                continue
            citation_match = _CORPUS_CITATION_PATH_LINE_PATTERN.match(line)
            if citation_match:
                corpus_citation_path = citation_match.group("path").strip()
                continue
            body_lines.append(line)
        body = "\n".join(body_lines).strip()
        if body:
            continuations.append(
                PrimarySourceContinuation(
                    source_path=path,
                    corpus_citation_path=corpus_citation_path,
                    body=body,
                )
            )
    return continuations


def _append_primary_source_continuations(
    source_text: str,
    continuations: Sequence[PrimarySourceContinuation],
) -> str:
    if not continuations:
        return source_text
    parts = [source_text.strip()]
    for continuation in continuations:
        label = continuation.corpus_citation_path or continuation.source_path.as_posix()
        parts.append(f"[Primary source continuation: {label}]\n{continuation.body}")
    return "\n\n".join(part for part in parts if part).strip()


def _source_metadata_with_continuations(
    source_metadata_payload: dict[str, object] | None,
    continuations: Sequence[PrimarySourceContinuation],
) -> dict[str, object] | None:
    if not continuations:
        return source_metadata_payload

    metadata = dict(source_metadata_payload or {})
    citation_paths: list[str] = []
    raw_paths = metadata.get("corpus_citation_paths")
    if isinstance(raw_paths, list):
        citation_paths.extend(str(item) for item in raw_paths if str(item).strip())
    raw_path = metadata.get("corpus_citation_path")
    if isinstance(raw_path, str) and raw_path.strip():
        citation_paths.append(raw_path.strip())

    continuation_records: list[dict[str, str]] = []
    for continuation in continuations:
        record = {"context_path": str(continuation.source_path)}
        if continuation.corpus_citation_path:
            citation_paths.append(continuation.corpus_citation_path)
            record["corpus_citation_path"] = continuation.corpus_citation_path
        continuation_records.append(record)

    deduped_paths = list(dict.fromkeys(citation_paths))
    if deduped_paths:
        metadata["corpus_citation_paths"] = deduped_paths
    metadata["primary_source_continuations"] = continuation_records
    return metadata


def _combine_retry_response(
    initial: EvalPromptResponse,
    retry: EvalPromptResponse,
    retry_prompt: str,
) -> EvalPromptResponse:
    """Return the retry response while preserving aggregate accounting."""
    return EvalPromptResponse(
        text=retry.text,
        duration_ms=initial.duration_ms + retry.duration_ms,
        tokens=_sum_token_usage(initial.tokens, retry.tokens),
        estimated_cost_usd=_sum_optional_float(
            initial.estimated_cost_usd, retry.estimated_cost_usd
        ),
        actual_cost_usd=_sum_optional_float(
            initial.actual_cost_usd, retry.actual_cost_usd
        ),
        trace={
            "retry_count": 1,
            "retry_reason": "empty_rulespec_artifact",
            "retry_prompt_sha256": _sha256_text(retry_prompt),
            "attempts": [
                {"attempt": 0, "trace": initial.trace or {}},
                {"attempt": 1, "trace": retry.trace or {}},
            ],
        },
        unexpected_accesses=[
            *initial.unexpected_accesses,
            *retry.unexpected_accesses,
        ],
        error=retry.error,
    )


def _response_allows_empty_artifact_retry(response: EvalPromptResponse) -> bool:
    """Return true when a missing artifact should get one forced retry."""
    if response.error is None:
        return True
    return not response.text.strip() and "timed out" in response.error.lower()


def load_eval_suite_manifest(path: Path) -> EvalSuiteManifest:
    """Load a manifest describing a benchmark suite and readiness gates."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Eval suite manifest must be a mapping: {path}")

    base_dir = Path(path).resolve().parent
    default_mode = _coerce_eval_mode(raw.get("mode", "repo-augmented"))
    default_context = [
        _resolve_manifest_path(base_dir, entry)
        for entry in raw.get("allow_context", []) or []
    ]
    runners = [
        str(item) for item in (raw.get("runners") or [f"codex:{DEFAULT_OPENAI_MODEL}"])
    ]

    gates_raw = raw.get("gates") or {}
    gates = EvalReadinessGates(
        min_cases=int(gates_raw.get("min_cases", 1)),
        min_success_rate=_optional_float(gates_raw.get("min_success_rate")),
        min_compile_pass_rate=_optional_float(gates_raw.get("min_compile_pass_rate")),
        min_ci_pass_rate=_optional_float(gates_raw.get("min_ci_pass_rate")),
        min_zero_ungrounded_rate=_optional_float(
            gates_raw.get("min_zero_ungrounded_rate")
        ),
        min_generalist_review_pass_rate=_optional_float(
            gates_raw.get("min_generalist_review_pass_rate", 1.0)
        ),
        min_policyengine_pass_rate=_optional_float(
            gates_raw.get("min_policyengine_pass_rate")
        ),
        max_mean_estimated_cost_usd=_optional_float(
            gates_raw.get("max_mean_estimated_cost_usd")
        ),
    )

    cases_raw = raw.get("cases") or []
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError(f"Eval suite manifest has no cases: {path}")

    cases: list[EvalSuiteCase] = []
    for index, item in enumerate(cases_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Eval suite case #{index} must be a mapping")
        kind = str(item.get("kind", "")).strip()
        if kind not in {"citation", "source"}:
            raise ValueError(f"Unsupported eval suite case kind '{kind}'")

        case_mode = _coerce_eval_mode(item.get("mode", default_mode))
        name = str(item.get("name", "")).strip() or str(
            item.get("citation") or item.get("source_id") or f"case-{index}"
        )
        if "source_file" in item:
            raise ValueError(
                "Eval suite source cases must use 'corpus_citation_path'; "
                f"'source_file' is no longer supported in {path}"
            )
        if "metadata_file" in item:
            raise ValueError(
                "Eval suite source metadata now comes from corpus.provisions; "
                f"'metadata_file' is no longer supported in {path}"
            )

        case = EvalSuiteCase(
            kind=kind,
            name=name,
            mode=case_mode,
            allow_context=[
                _resolve_manifest_path(base_dir, entry)
                for entry in item.get("allow_context", []) or []
            ],
            citation=item.get("citation"),
            source_id=item.get("source_id"),
            corpus_citation_path=(
                str(item.get("corpus_citation_path")).strip()
                if item.get("corpus_citation_path") is not None
                else None
            ),
            policyengine_rule_hint=(
                str(item.get("policyengine_rule_hint")).strip()
                if item.get("policyengine_rule_hint") is not None
                else None
            ),
            oracle=str(item.get("oracle", "none")),
            policyengine_country=str(item.get("policyengine_country", "auto")),
        )
        _validate_eval_suite_case(case, index)
        cases.append(case)

    return EvalSuiteManifest(
        name=str(raw.get("name") or Path(path).stem),
        path=Path(path).resolve(),
        runners=runners,
        mode=default_mode,
        allow_context=default_context,
        gates=gates,
        cases=cases,
    )


def run_eval_suite(
    manifest: EvalSuiteManifest,
    output_root: Path,
    axiom_rules_path: Path,
    corpus_path: Path | None = None,
    runner_specs: list[str] | None = None,
    suite_retry_attempts: int = 2,
    resume_existing: bool = False,
) -> list[EvalResult]:
    """Run every case in a benchmark suite manifest."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_runners = runner_specs or manifest.runners
    parsed_runners = [parse_runner_spec(spec) for spec in resolved_runners]
    results: list[EvalResult] = []
    started_at = _utc_now_iso()
    completed_case_indexes: set[int] = set()
    completed_cases = 0
    last_case_name: str | None = None
    active_case_index: int | None = None
    active_case_name: str | None = None
    active_case_started_at: str | None = None
    active_case_output_root: Path | None = None
    if resume_existing:
        (
            started_at,
            results,
            completed_case_indexes,
        ) = _load_eval_suite_resume_state(
            output_root=output_root,
            manifest=manifest,
            resolved_runners=resolved_runners,
            runner_count=len(parsed_runners),
        )
        completed_cases = _contiguous_completed_case_count(
            completed_case_indexes, len(manifest.cases)
        )
        if completed_cases > 0:
            last_case_name = manifest.cases[completed_cases - 1].name
    _write_eval_suite_run_state(
        output_root=output_root,
        manifest=manifest,
        resolved_runners=resolved_runners,
        status="running",
        started_at=started_at,
        completed_cases=completed_cases,
        result_count=len(results),
        last_case_name=last_case_name,
    )
    try:
        for index, case in enumerate(manifest.cases, start=1):
            if index in completed_case_indexes:
                continue
            case_output_root = output_root / f"{index:02d}-{_slugify(case.name)}"
            extra_context = [*manifest.allow_context, *case.allow_context]
            attempts = max(suite_retry_attempts, 0) + 1
            active_case_index = index
            active_case_name = case.name
            active_case_started_at = _utc_now_iso()
            active_case_output_root = case_output_root
            _write_eval_suite_run_state(
                output_root=output_root,
                manifest=manifest,
                resolved_runners=resolved_runners,
                status="running",
                started_at=started_at,
                completed_cases=completed_cases,
                result_count=len(results),
                last_case_name=last_case_name,
                active_case_index=active_case_index,
                active_case_name=active_case_name,
                active_case_started_at=active_case_started_at,
                active_case_output_root=active_case_output_root,
            )
            for attempt_index in range(attempts):
                try:
                    if case.kind == "citation":
                        if corpus_path is None:
                            raise ValueError(
                                "corpus_path is required for citation eval suite cases"
                            )
                        case_results = run_model_eval(
                            citations=[case.citation or ""],
                            runner_specs=resolved_runners,
                            output_root=case_output_root,
                            policy_path=(
                                axiom_rules_path.parent / "rulespec-us"
                                if axiom_rules_path.name == "axiom-rules-engine"
                                else axiom_rules_path
                            ),
                            runtime_axiom_rules_path=axiom_rules_path,
                            corpus_path=corpus_path,
                            mode=case.mode,
                            extra_context_paths=extra_context,
                        )
                    elif case.kind == "source":
                        if corpus_path is None:
                            raise ValueError(
                                "corpus_path is required for corpus-backed "
                                "source eval suite cases"
                            )
                        source_unit = resolve_corpus_source_unit(
                            case.corpus_citation_path or "",
                            corpus_path,
                        )
                        source_text = source_unit.body
                        policy_repo_root = _policy_repo_root_for_corpus_source(
                            source_unit.citation_path,
                            axiom_rules_path,
                        )
                        source_metadata_payload = {
                            "corpus_citation_path": source_unit.citation_path,
                            "corpus_source": source_unit.source,
                            "requested_source": source_unit.requested,
                        }
                        case_results = run_source_eval(
                            source_id=case.source_id or case.name,
                            source_text=source_text,
                            runner_specs=resolved_runners,
                            output_root=case_output_root,
                            policy_path=policy_repo_root,
                            source_metadata_payload=source_metadata_payload,
                            runtime_axiom_rules_path=axiom_rules_path,
                            mode=case.mode,
                            extra_context_paths=extra_context,
                            oracle=case.oracle,
                            policyengine_country=case.policyengine_country,
                            policyengine_rule_hint=case.policyengine_rule_hint,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported eval suite case kind '{case.kind}'"
                        )
                except Exception as exc:
                    case_results = _suite_case_failure_results(
                        case, parsed_runners, exc
                    )

                if (
                    attempt_index >= attempts - 1
                    or not _suite_case_results_should_retry(case_results)
                ):
                    break

            for result in case_results:
                if case.name and case.name != result.citation:
                    result.citation = f"{case.name} ({result.citation})"
            results.extend(case_results)
            completed_case_indexes.add(index)
            completed_cases = index
            last_case_name = case.name
            active_case_index = None
            active_case_name = None
            active_case_started_at = None
            active_case_output_root = None
            _append_eval_suite_case_results(output_root, index, case, case_results)
            if _suite_case_results_hit_usage_limit(case_results):
                _write_eval_suite_run_state(
                    output_root=output_root,
                    manifest=manifest,
                    resolved_runners=resolved_runners,
                    status="failed",
                    started_at=started_at,
                    completed_cases=completed_cases,
                    result_count=len(results),
                    last_case_name=last_case_name,
                    error=(
                        "Usage limit reached while running "
                        f"case '{case.name}'. Stop the suite and retry after quota resets."
                    ),
                )
                return results
            _write_eval_suite_run_state(
                output_root=output_root,
                manifest=manifest,
                resolved_runners=resolved_runners,
                status="running",
                started_at=started_at,
                completed_cases=completed_cases,
                result_count=len(results),
                last_case_name=last_case_name,
            )
    except BaseException as exc:
        _write_eval_suite_run_state(
            output_root=output_root,
            manifest=manifest,
            resolved_runners=resolved_runners,
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            started_at=started_at,
            completed_cases=completed_cases,
            result_count=len(results),
            last_case_name=last_case_name,
            error=_format_suite_exception(exc),
            active_case_index=active_case_index,
            active_case_name=active_case_name,
            active_case_started_at=active_case_started_at,
            active_case_output_root=active_case_output_root,
        )
        raise

    _write_eval_suite_run_state(
        output_root=output_root,
        manifest=manifest,
        resolved_runners=resolved_runners,
        status="completed",
        started_at=started_at,
        completed_cases=completed_cases,
        result_count=len(results),
        last_case_name=last_case_name,
    )
    return results


def _contiguous_completed_case_count(
    completed_case_indexes: set[int],
    total_cases: int,
) -> int:
    """Return the largest completed case prefix represented in the ledger."""
    completed = 0
    for index in range(1, total_cases + 1):
        if index not in completed_case_indexes:
            break
        completed = index
    return completed


def _utc_now_iso() -> str:
    """Return the current UTC time in a stable JSON-friendly format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_suite_exception(exc: BaseException) -> str:
    """Return a non-empty error string for suite state logs."""
    message = str(exc).strip()
    return message or exc.__class__.__name__


def _load_eval_suite_resume_state(
    output_root: Path,
    manifest: EvalSuiteManifest,
    resolved_runners: list[str],
    runner_count: int,
) -> tuple[str, list[EvalResult], set[int]]:
    """Load prior suite state and completed case results for resumption."""
    state_path = output_root / "suite-run.json"
    ledger_path = output_root / "suite-results.jsonl"
    started_at = _utc_now_iso()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        manifest_payload = state.get("manifest") or {}
        existing_path = manifest_payload.get("path")
        if existing_path and existing_path != str(manifest.path):
            raise ValueError(
                "Cannot resume eval suite with a different manifest path: "
                f"{existing_path}"
            )
        existing_runners = manifest_payload.get("effective_runners")
        if existing_runners and list(existing_runners) != list(resolved_runners):
            raise ValueError(
                "Cannot resume eval suite with different effective runners: "
                f"{existing_runners}"
            )
        started_at = state.get("started_at") or started_at

    if not ledger_path.exists():
        return started_at, [], set()

    rows_by_case: dict[int, list[dict]] = defaultdict(list)
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        case_index = int(payload.get("case_index", 0) or 0)
        if case_index <= 0:
            continue
        rows_by_case[case_index].append(payload)

    completed_case_indexes: set[int] = set()
    results: list[EvalResult] = []
    for case_index in sorted(rows_by_case):
        rows = rows_by_case[case_index]
        if len(rows) < runner_count:
            continue
        completed_case_indexes.add(case_index)
        for payload in rows[:runner_count]:
            results.append(_eval_result_from_payload(payload.get("result") or {}))

    return started_at, results, completed_case_indexes


def _write_eval_suite_run_state(
    output_root: Path,
    manifest: EvalSuiteManifest,
    resolved_runners: list[str],
    status: str,
    started_at: str,
    completed_cases: int,
    result_count: int,
    last_case_name: str | None = None,
    error: str | None = None,
    active_case_index: int | None = None,
    active_case_name: str | None = None,
    active_case_started_at: str | None = None,
    active_case_output_root: Path | None = None,
) -> None:
    """Persist suite lifecycle state so interrupted runs remain inspectable."""
    payload = {
        "manifest": {
            "name": manifest.name,
            "path": str(manifest.path),
            "runners": manifest.runners,
            "effective_runners": resolved_runners,
        },
        "status": status,
        "started_at": started_at,
        "updated_at": _utc_now_iso(),
        "total_cases": len(manifest.cases),
        "completed_cases": completed_cases,
        "result_count": result_count,
    }
    if last_case_name:
        payload["last_case_name"] = last_case_name
    if error:
        payload["error"] = error
    if active_case_index is not None and active_case_name:
        payload["active_case"] = {
            "index": active_case_index,
            "name": active_case_name,
            "started_at": active_case_started_at,
            "output_root": str(active_case_output_root)
            if active_case_output_root is not None
            else None,
        }
    if status != "running":
        payload["finished_at"] = payload["updated_at"]
    (output_root / "suite-run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _append_eval_suite_case_results(
    output_root: Path,
    case_index: int,
    case: EvalSuiteCase,
    case_results: list[EvalResult],
) -> None:
    """Append finalized case results to a durable JSONL ledger."""
    ledger_path = output_root / "suite-results.jsonl"
    with ledger_path.open("a", encoding="utf-8") as handle:
        for result in case_results:
            payload = {
                "case_index": case_index,
                "case_name": case.name,
                "case_kind": case.kind,
                "result": result.to_dict(),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _eval_result_from_payload(payload: dict) -> EvalResult:
    """Rehydrate an EvalResult from a persisted JSON payload."""
    metrics_payload = payload.get("metrics")
    metrics = None
    if isinstance(metrics_payload, dict):
        grounding = [
            GroundingMetric(
                line=int(item.get("line", 0) or 0),
                raw=str(item.get("raw", "")),
                value=float(item.get("value", 0.0) or 0.0),
                grounded=bool(item.get("grounded", False)),
            )
            for item in metrics_payload.get("grounding") or []
        ]
        metrics = EvalArtifactMetrics(
            compile_pass=bool(metrics_payload.get("compile_pass", False)),
            compile_issues=list(metrics_payload.get("compile_issues") or []),
            ci_pass=bool(metrics_payload.get("ci_pass", False)),
            ci_issues=list(metrics_payload.get("ci_issues") or []),
            embedded_source_present=bool(
                metrics_payload.get("embedded_source_present", False)
            ),
            grounded_numeric_count=int(
                metrics_payload.get("grounded_numeric_count", 0) or 0
            ),
            ungrounded_numeric_count=int(
                metrics_payload.get("ungrounded_numeric_count", 0) or 0
            ),
            grounding=grounding,
            source_numeric_occurrence_count=int(
                metrics_payload.get("source_numeric_occurrence_count", 0) or 0
            ),
            covered_source_numeric_occurrence_count=int(
                metrics_payload.get("covered_source_numeric_occurrence_count", 0) or 0
            ),
            missing_source_numeric_occurrence_count=int(
                metrics_payload.get("missing_source_numeric_occurrence_count", 0) or 0
            ),
            numeric_occurrence_issues=list(
                metrics_payload.get("numeric_occurrence_issues") or []
            ),
            generalist_review_pass=metrics_payload.get("generalist_review_pass"),
            generalist_review_score=metrics_payload.get("generalist_review_score"),
            generalist_review_issues=list(
                metrics_payload.get("generalist_review_issues") or []
            ),
            generalist_review_prompt_sha256=metrics_payload.get(
                "generalist_review_prompt_sha256"
            ),
            policyengine_pass=metrics_payload.get("policyengine_pass"),
            policyengine_score=metrics_payload.get("policyengine_score"),
            policyengine_issues=list(metrics_payload.get("policyengine_issues") or []),
            taxsim_pass=metrics_payload.get("taxsim_pass"),
            taxsim_score=metrics_payload.get("taxsim_score"),
            taxsim_issues=list(metrics_payload.get("taxsim_issues") or []),
        )

    return EvalResult(
        citation=str(payload.get("citation", "")),
        runner=str(payload.get("runner", "")),
        backend=str(payload.get("backend", "")),
        model=str(payload.get("model", "")),
        mode=payload.get("mode", "cold"),
        output_file=str(payload.get("output_file", "")),
        trace_file=str(payload.get("trace_file", "")),
        context_manifest_file=str(payload.get("context_manifest_file", "")),
        duration_ms=int(payload.get("duration_ms", 0) or 0),
        success=bool(payload.get("success", False)),
        error=payload.get("error"),
        generation_prompt_sha256=payload.get("generation_prompt_sha256"),
        input_tokens=int(payload.get("input_tokens", 0) or 0),
        output_tokens=int(payload.get("output_tokens", 0) or 0),
        cache_read_tokens=int(payload.get("cache_read_tokens", 0) or 0),
        cache_creation_tokens=int(payload.get("cache_creation_tokens", 0) or 0),
        reasoning_output_tokens=int(payload.get("reasoning_output_tokens", 0) or 0),
        estimated_cost_usd=payload.get("estimated_cost_usd"),
        actual_cost_usd=payload.get("actual_cost_usd"),
        retrieved_files=list(payload.get("retrieved_files") or []),
        unexpected_accesses=list(payload.get("unexpected_accesses") or []),
        metrics=metrics,
        retry_count=int(payload.get("retry_count", 0) or 0),
    )


def _suite_case_failure_results(
    case: EvalSuiteCase,
    runners: list[EvalRunnerSpec],
    exc: Exception,
) -> list[EvalResult]:
    """Convert an exception into explicit failed results for each runner."""
    return [
        EvalResult(
            citation=case.name,
            runner=runner.name,
            backend=runner.backend,
            model=runner.model,
            mode=case.mode,
            output_file="",
            trace_file="",
            context_manifest_file="",
            duration_ms=0,
            success=False,
            error=str(exc),
            generation_prompt_sha256=None,
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            reasoning_output_tokens=0,
            estimated_cost_usd=None,
            actual_cost_usd=None,
            retrieved_files=[],
            unexpected_accesses=[],
            metrics=None,
        )
        for runner in runners
    ]


def _suite_case_results_should_retry(case_results: list[EvalResult]) -> bool:
    """Return True when a suite case likely failed for a transient reason."""
    if _suite_case_results_hit_usage_limit(case_results):
        return False
    return any(
        result.error is not None
        or result.metrics is None
        or _eval_result_indicates_retryable_timeout(result)
        for result in case_results
    )


def _suite_case_results_hit_usage_limit(case_results: list[EvalResult]) -> bool:
    """Return True when a case result indicates hard quota exhaustion."""
    return any(_eval_result_indicates_usage_limit(result) for result in case_results)


def _eval_result_indicates_usage_limit(result: EvalResult) -> bool:
    """Return True when one result contains a non-retryable usage-limit error."""
    texts: list[str] = []
    if result.error:
        texts.append(result.error)
    if result.metrics is not None:
        texts.extend(result.metrics.compile_issues)
        texts.extend(result.metrics.ci_issues)
        texts.extend(result.metrics.generalist_review_issues)
        texts.extend(result.metrics.policyengine_issues)
        texts.extend(result.metrics.taxsim_issues)

    return any("usage limit" in text.lower() for text in texts)


def _eval_result_indicates_retryable_timeout(result: EvalResult) -> bool:
    """Return True when one result failed due to a transient timeout."""
    texts: list[str] = []
    if result.error:
        texts.append(result.error)
    if result.metrics is not None:
        texts.extend(result.metrics.compile_issues)
        texts.extend(result.metrics.ci_issues)
        texts.extend(result.metrics.generalist_review_issues)
        texts.extend(result.metrics.policyengine_issues)
        texts.extend(result.metrics.taxsim_issues)

    lowered = [text.lower() for text in texts]
    return any("timeout after" in text or "timed out" in text for text in lowered)


def summarize_readiness(
    results: list[EvalResult],
    gates: EvalReadinessGates,
) -> EvalReadinessSummary:
    """Summarize suite readiness for one runner."""
    total_cases = len(results)
    success_rate = _fraction(
        sum(1 for result in results if result.success), total_cases
    )
    compile_pass_rate = _fraction(
        sum(
            1
            for result in results
            if result.metrics is not None and result.metrics.compile_pass
        ),
        total_cases,
    )
    ci_pass_rate = _fraction(
        sum(
            1
            for result in results
            if result.metrics is not None and result.metrics.ci_pass
        ),
        total_cases,
    )
    zero_ungrounded_rate = _fraction(
        sum(
            1
            for result in results
            if result.metrics is not None
            and result.metrics.ungrounded_numeric_count == 0
        ),
        total_cases,
    )
    generalist_review_pass_rate = _fraction(
        sum(
            1
            for result in results
            if result.metrics is not None and result.metrics.generalist_review_pass
        ),
        total_cases,
    )
    generalist_scores = [
        result.metrics.generalist_review_score
        for result in results
        if result.metrics is not None
        and result.metrics.generalist_review_score is not None
    ]
    mean_generalist_review_score = (
        round(mean(generalist_scores), 6) if generalist_scores else None
    )

    policyengine_results = [
        result
        for result in results
        if result.metrics is not None and result.metrics.policyengine_score is not None
    ]
    policyengine_case_count = len(policyengine_results)
    policyengine_pass_rate = (
        _fraction(
            sum(
                1
                for result in policyengine_results
                if result.metrics is not None and result.metrics.policyengine_pass
            ),
            policyengine_case_count,
        )
        if policyengine_case_count
        else None
    )
    mean_policyengine_score = (
        round(
            mean(
                result.metrics.policyengine_score
                for result in policyengine_results
                if result.metrics is not None
                and result.metrics.policyengine_score is not None
            ),
            6,
        )
        if policyengine_case_count
        else None
    )

    costs = [
        result.estimated_cost_usd
        for result in results
        if result.estimated_cost_usd is not None
    ]
    mean_estimated_cost_usd = round(mean(costs), 6) if costs else None

    gate_results: list[EvalReadinessGateResult] = [
        _min_gate("min_cases", total_cases, gates.min_cases),
    ]
    if gates.min_success_rate is not None:
        gate_results.append(
            _min_gate("min_success_rate", success_rate, gates.min_success_rate)
        )
    if gates.min_compile_pass_rate is not None:
        gate_results.append(
            _min_gate(
                "min_compile_pass_rate",
                compile_pass_rate,
                gates.min_compile_pass_rate,
            )
        )
    if gates.min_ci_pass_rate is not None:
        gate_results.append(
            _min_gate("min_ci_pass_rate", ci_pass_rate, gates.min_ci_pass_rate)
        )
    if gates.min_zero_ungrounded_rate is not None:
        gate_results.append(
            _min_gate(
                "min_zero_ungrounded_rate",
                zero_ungrounded_rate,
                gates.min_zero_ungrounded_rate,
            )
        )
    if gates.min_generalist_review_pass_rate is not None:
        gate_results.append(
            _min_gate(
                "min_generalist_review_pass_rate",
                generalist_review_pass_rate,
                gates.min_generalist_review_pass_rate,
            )
        )
    if gates.min_policyengine_pass_rate is not None:
        gate_results.append(
            _min_gate(
                "min_policyengine_pass_rate",
                policyengine_pass_rate,
                gates.min_policyengine_pass_rate,
            )
        )
    if gates.max_mean_estimated_cost_usd is not None:
        gate_results.append(
            _max_gate(
                "max_mean_estimated_cost_usd",
                mean_estimated_cost_usd,
                gates.max_mean_estimated_cost_usd,
            )
        )

    return EvalReadinessSummary(
        total_cases=total_cases,
        success_rate=success_rate,
        compile_pass_rate=compile_pass_rate,
        ci_pass_rate=ci_pass_rate,
        zero_ungrounded_rate=zero_ungrounded_rate,
        generalist_review_pass_rate=generalist_review_pass_rate,
        mean_generalist_review_score=mean_generalist_review_score,
        policyengine_case_count=policyengine_case_count,
        policyengine_pass_rate=policyengine_pass_rate,
        mean_policyengine_score=mean_policyengine_score,
        mean_estimated_cost_usd=mean_estimated_cost_usd,
        gate_results=gate_results,
        ready=all(result.passed for result in gate_results),
    )


def _coerce_eval_mode(value: str) -> EvalMode:
    """Validate a manifest eval mode."""
    normalized = str(value).strip()
    if normalized not in {"cold", "repo-augmented"}:
        raise ValueError(f"Unsupported eval mode '{value}'")
    return normalized  # type: ignore[return-value]


def _optional_float(value: object) -> float | None:
    """Convert optional numeric manifest values to float."""
    if value is None:
        return None
    return float(value)


def _resolve_manifest_path(base_dir: Path, value: object) -> Path:
    """Resolve a manifest path entry relative to the manifest file.

    Manifest entries are written against the legacy sibling-checkout layout
    (``../rulespec-us-co/...``). When the literal path is missing but the
    content has moved into a country monorepo
    (``../rulespec-us/us-co/...``), the monorepo equivalent resolves instead
    — benchmark YAML stays unchanged across the consolidation.
    """
    path = Path(str(value))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.exists():
        monorepo_path = monorepo_alternative_path(path)
        if monorepo_path is not None and monorepo_path.exists():
            return monorepo_path
    return path


def _validate_eval_suite_case(case: EvalSuiteCase, index: int) -> None:
    """Validate one suite case after parsing."""
    if case.kind == "citation" and not case.citation:
        raise ValueError(f"Eval suite case #{index} is missing 'citation'")
    if case.kind == "source":
        if not case.source_id:
            raise ValueError(f"Eval suite case #{index} is missing 'source_id'")
        if not case.corpus_citation_path:
            raise ValueError(
                f"Eval suite case #{index} is missing 'corpus_citation_path'"
            )


def _fraction(numerator: int, denominator: int) -> float:
    """Return a rounded fraction or 0 when the denominator is empty."""
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _min_gate(
    name: str,
    actual: float | int | None,
    threshold: float | int,
) -> EvalReadinessGateResult:
    """Evaluate a lower-bound readiness gate."""
    return EvalReadinessGateResult(
        name=name,
        comparator="min",
        threshold=threshold,
        actual=actual,
        passed=actual is not None and actual >= threshold,
    )


def _max_gate(
    name: str,
    actual: float | int | None,
    threshold: float | int,
) -> EvalReadinessGateResult:
    """Evaluate an upper-bound readiness gate."""
    return EvalReadinessGateResult(
        name=name,
        comparator="max",
        threshold=threshold,
        actual=actual,
        passed=actual is not None and actual <= threshold,
    )


def select_context_files(
    citation: str | CitationParts,
    policy_root: Path,
    max_files: int = 6,
) -> list[Path]:
    """Select canonical implementation precedent files for repo-augmented evals."""
    parts = (
        citation
        if isinstance(citation, CitationParts)
        else parse_usc_citation(citation)
    )
    repo_root = Path(policy_root)
    statutes_root = repo_root / "statutes"
    section_root = statutes_root / parts.title / parts.section
    target_rel = citation_to_relative_rulespec_path(parts)
    target_path = repo_root / target_rel

    candidates: list[Path] = []
    if section_root.exists():
        candidates.extend(
            sorted(
                path
                for path in section_root.rglob("*.yaml")
                if path.resolve() != target_path.resolve()
                and not path.name.endswith(".test.yaml")
            )
        )

    if not candidates:
        title_root = statutes_root / parts.title
        candidates.extend(
            sorted(
                path
                for path in title_root.rglob("*.yaml")
                if path != target_path and not path.name.endswith(".test.yaml")
            )
        )

    # Bias toward nearby files first, then shallower paths for readability.
    candidates.sort(
        key=lambda path: (
            0 if path.parent == section_root else 1,
            len(path.relative_to(statutes_root).parts),
            str(path),
        )
    )

    selected: list[Path] = []
    for candidate in candidates:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= max_files:
            break
    return selected


def prepare_eval_workspace(
    citation: str,
    runner: EvalRunnerSpec,
    output_root: Path,
    source_text: str,
    axiom_rules_path: Path,
    mode: EvalMode,
    source_metadata_payload: dict[str, object] | None = None,
    extra_context_paths: list[Path] | None = None,
) -> EvalWorkspace:
    """Create an isolated workspace bundle for a single eval."""
    slug = _slugify(citation)
    workspace_root = (
        Path(output_root) / "_eval_workspaces" / runner.name / slug / "workspace"
    )
    if workspace_root.parent.exists():
        shutil.rmtree(workspace_root.parent)
    workspace_root.mkdir(parents=True, exist_ok=True)

    source_text_file = workspace_root / "source.txt"
    source_text_file.write_text(source_text.strip() + "\n")
    source_metadata = dict(source_metadata_payload or {})
    if not source_metadata:
        source_metadata = None
    source_metadata_file: Path | None = None
    if source_metadata is not None:
        source_metadata_file = workspace_root / "source-metadata.json"
        source_metadata_file.write_text(
            json.dumps(source_metadata, indent=2, sort_keys=True) + "\n"
        )

    context_files: list[EvalContextFile] = []
    context_root = workspace_root / "context"
    target_rel = _target_rel_for_eval_identifier(citation)
    current_file = axiom_rules_path / target_rel if target_rel is not None else None
    for resolved_term in resolve_defined_terms_from_text(source_text):
        context_files.append(
            _materialize_resolved_definition_stub(
                context_root=context_root,
                resolved_term=resolved_term,
                workspace_root=workspace_root,
            )
        )
    for resolved_concept in resolve_canonical_concepts_from_text(
        source_text,
        axiom_rules_path,
        current_file=current_file,
    ):
        context_item = _materialize_resolved_canonical_concept(
            context_root=context_root,
            resolved_concept=resolved_concept,
            workspace_root=workspace_root,
        )
        context_files.append(context_item)
        companion_item = _materialize_resolved_canonical_concept_companion_test(
            context_item=context_item,
            resolved_concept=resolved_concept,
            workspace_root=workspace_root,
        )
        if companion_item is not None:
            context_files.append(companion_item)

    context_corpus_root = _repo_augmented_context_root(axiom_rules_path)

    if mode == "repo-augmented":
        selected = _auto_select_context_files(citation, context_corpus_root)
        selected.extend(
            _select_child_fragment_context_files(citation, context_corpus_root)
        )
        selected.extend(
            _select_same_section_subsection_context_files(
                citation,
                source_text,
                context_corpus_root,
            )
        )
        selected.extend(
            _select_cross_section_context_files(
                citation,
                source_text,
                context_corpus_root,
            )
        )
        for extra_path in extra_context_paths or []:
            path = Path(extra_path)
            if path.exists():
                selected.append(path)

        expanded_context = _expand_context_files(
            selected, context_corpus_root, target_rel
        )

        for source_path, kind in expanded_context:
            relative_target = _context_import_relative_target(
                source_path, axiom_rules_path
            )

            workspace_relative_path = Path("context") / relative_target
            workspace_path = workspace_root / workspace_relative_path
            workspace_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, workspace_path)
            context_files.append(
                EvalContextFile(
                    source_path=str(source_path),
                    workspace_path=str(workspace_relative_path),
                    import_path=_context_import_target(source_path, relative_target),
                    kind=kind,
                )
            )

    manifest_file = workspace_root / "context-manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "citation": citation,
                "mode": mode,
                "source_text_file": str(source_text_file.relative_to(workspace_root)),
                "source_metadata_file": (
                    str(source_metadata_file.relative_to(workspace_root))
                    if source_metadata_file is not None
                    else None
                ),
                "source_metadata": source_metadata,
                "context_files": [asdict(item) for item in context_files],
            },
            indent=2,
            sort_keys=True,
        )
    )

    return EvalWorkspace(
        root=workspace_root,
        source_text_file=source_text_file,
        manifest_file=manifest_file,
        source_metadata_file=source_metadata_file,
        source_metadata=source_metadata,
        context_files=context_files,
    )


def resolve_corpus_source_unit(
    identifier: str,
    corpus_path: Path,
) -> CorpusSourceUnit:
    """Resolve an encode target to normalized corpus.provisions text.

    The identifier may already be a corpus citation path, or it may be a USC
    citation that can be normalized to one. For USC child fragments, the
    resolver falls back to the nearest available section-level corpus provision
    and slices that body to the requested fragment when the parent text carries
    structural markers such as ``(a)``, ``(2)``, or ``(C)``.
    """
    candidates = _candidate_corpus_citation_paths(identifier)
    primary = candidates[0] if candidates else ""
    for citation_path in candidates:
        local_text = _fetch_local_corpus_source_text_from_repo(
            citation_path,
            corpus_path,
        )
        if local_text is not None:
            body = _slice_parent_corpus_text_for_requested_path(
                local_text,
                requested_path=primary,
                resolved_path=citation_path,
            )
            return CorpusSourceUnit(
                requested=identifier,
                citation_path=citation_path,
                body=body,
                source="local",
            )
        supabase_text = _fetch_supabase_corpus_source_text(citation_path)
        if supabase_text is not None:
            body = _slice_parent_corpus_text_for_requested_path(
                supabase_text,
                requested_path=primary,
                resolved_path=citation_path,
            )
            return CorpusSourceUnit(
                requested=identifier,
                citation_path=citation_path,
                body=body,
                source="supabase",
            )

    candidates = ", ".join(_candidate_corpus_citation_paths(identifier)[:4])
    raise ValueError(
        "No corpus.provisions source text found for "
        f"{identifier!r}. Tried: {candidates}"
    )


def _slice_parent_corpus_text_for_requested_path(
    text: str,
    *,
    requested_path: str,
    resolved_path: str,
) -> str:
    """Slice section-granular source text to the requested child fragment."""
    requested_parts = requested_path.strip("/").split("/")
    resolved_parts = resolved_path.strip("/").split("/")
    if (
        len(requested_parts) <= len(resolved_parts)
        or requested_parts[: len(resolved_parts)] != resolved_parts
        or not _citation_path_supports_parenthetical_slicing(resolved_parts)
    ):
        return text
    missing_fragments = tuple(requested_parts[len(resolved_parts) :])
    if _corpus_citation_path_is_us_cfr(resolved_parts):
        cfr_sliced = _target_source_scope_by_cfr_hierarchy(
            text,
            list(missing_fragments),
        )
        if cfr_sliced is not None:
            return cfr_sliced.strip()
    sliced = _slice_legal_text_by_parenthetical_fragments(text, missing_fragments)
    return sliced if sliced is not None else text


def _citation_path_supports_parenthetical_slicing(parts: list[str]) -> bool:
    return len(parts) >= 2 and parts[1] in {"statute", "regulation"}


def _corpus_citation_path_is_us_cfr(parts: list[str]) -> bool:
    return (
        len(parts) >= 5
        and parts[0] == "us"
        and parts[1] == "regulation"
        and parts[2].isdigit()
    )


def _slice_legal_text_by_parenthetical_fragments(
    text: str,
    fragments: tuple[str, ...],
) -> str | None:
    sliced = _slice_legal_text_by_parenthetical_fragments_from(
        text,
        fragments,
        depth=0,
    )
    return sliced.strip() if sliced is not None else None


def _slice_legal_text_by_parenthetical_fragments_from(
    text: str,
    fragments: tuple[str, ...],
    *,
    depth: int,
) -> str | None:
    if not fragments:
        return text

    fragment = fragments[0]
    simple_slice = _slice_legal_text_by_parenthetical_fragment(
        text,
        fragment,
        top_level=depth == 0,
    )
    if simple_slice is not None:
        simple_result = _slice_legal_text_by_parenthetical_fragments_from(
            simple_slice,
            fragments[1:],
            depth=depth + 1,
        )
        if simple_result is not None:
            return simple_result

    if len(fragments) >= 2:
        combined = _combined_dotted_parenthetical_fragment(fragment, fragments[1])
        if combined is not None:
            combined_slice = _slice_legal_text_by_parenthetical_fragment(
                text,
                combined,
                top_level=depth == 0,
            )
            if combined_slice is not None:
                return _slice_legal_text_by_parenthetical_fragments_from(
                    combined_slice,
                    fragments[2:],
                    depth=depth + 2,
                )

    return None


def _combined_dotted_parenthetical_fragment(
    fragment: str, next_fragment: str
) -> str | None:
    if not next_fragment.isdigit():
        return None
    if re.fullmatch(r"(?:[A-Za-z]|\d+)(?:\.\d+)*", fragment):
        return f"{fragment}.{next_fragment}"
    return None


def _slice_legal_text_by_parenthetical_fragment(
    text: str,
    fragment: str,
    *,
    top_level: bool,
) -> str | None:
    escaped = re.escape(fragment)
    if top_level:
        marker_pattern = re.compile(rf"(?:^|\n\s*)(\[?\({escaped}\)\s+)")
    else:
        marker_pattern = re.compile(rf"(?<![A-Za-z0-9])(\({escaped}\)\s+)")
    marker_match = next(
        (
            match
            for match in marker_pattern.finditer(text)
            if _parenthetical_marker_context_is_structural(text, match.start(1))
        ),
        None,
    )
    if marker_match is None:
        return None

    start = marker_match.start(1)
    body_start = marker_match.end(1)
    sibling_pattern = _sibling_parenthetical_marker_pattern(fragment, top_level)
    end = len(text)
    for sibling_match in sibling_pattern.finditer(text, body_start):
        if sibling_match.start(
            1
        ) > start and _parenthetical_marker_context_is_structural(
            text,
            sibling_match.start(1),
        ):
            end = sibling_match.start(1)
            break
    return text[start:end]


_NONSTRUCTURAL_PARENTHETICAL_REFERENCE_LABEL = (
    r"(?:paragraphs?|subparagraphs?|clauses?|subclauses?|sections?|"
    r"subsections?|chapters?|titles?|parts?|items?|sentences?|regulations?)"
)
_NONSTRUCTURAL_PARENTHETICAL_REFERENCE_PREFIX = re.compile(
    rf"\b{_NONSTRUCTURAL_PARENTHETICAL_REFERENCE_LABEL}\s+$",
    re.IGNORECASE,
)


def _parenthetical_marker_context_is_structural(text: str, marker_start: int) -> bool:
    prefix = text[max(0, marker_start - 60) : marker_start]
    previous = prefix.rstrip()[-1:] if prefix.rstrip() else ""
    if previous == ")" and not re.search(
        r"(?:^|\n)\s*\([A-Za-z0-9]+(?:\.[0-9]+)*\)\s+$",
        prefix,
    ):
        return False
    if _NONSTRUCTURAL_PARENTHETICAL_REFERENCE_PREFIX.search(prefix):
        return False
    return not _parenthetical_marker_is_in_reference_list(prefix)


def _parenthetical_marker_is_in_reference_list(prefix: str) -> bool:
    segment = re.split(r"(?:[.;]\s+|\n+)", prefix)[-1]
    if not re.search(
        rf"\b{_NONSTRUCTURAL_PARENTHETICAL_REFERENCE_LABEL}\b",
        segment,
        flags=re.IGNORECASE,
    ):
        return False
    if not re.search(r"\([A-Za-z0-9]+\)", segment):
        return False
    return (
        re.search(
            r"(?:,\s*|\b(?:or|and|through|to)\s+|[-–—]\s*)$",
            segment,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _sibling_parenthetical_marker_pattern(
    fragment: str,
    top_level: bool,
) -> re.Pattern[str]:
    if re.fullmatch(r"\d+(?:\.\d+)*", fragment):
        marker = _numeric_sibling_parenthetical_marker(fragment)
    elif re.fullmatch(r"[A-Z](?:\.\d+)*", fragment):
        marker = _alpha_sibling_parenthetical_marker(fragment, top_level=top_level)
    elif re.fullmatch(r"[a-z](?:\.\d+)*", fragment):
        marker = _alpha_sibling_parenthetical_marker(fragment, top_level=top_level)
    elif len(fragment) == 1 and fragment.isalpha() and fragment.isupper():
        marker = (
            _next_alpha_parenthetical_marker(fragment) if top_level else r"\([A-Z]\)"
        )
    elif len(fragment) == 1 and fragment.isalpha() and fragment.islower():
        marker = (
            _next_alpha_parenthetical_marker(fragment) if top_level else r"\([a-z]\)"
        )
    elif re.fullmatch(r"[ivxlcdm]+", fragment, re.IGNORECASE):
        marker = r"\([ivxlcdm]+\)"
    else:
        marker = r"\([A-Za-z0-9]+\)"
    if top_level:
        return re.compile(rf"\n\s*(\[?{marker}\s+)")
    return re.compile(rf"(?<![A-Za-z0-9])({marker}\s+)")


def _numeric_sibling_parenthetical_marker(fragment: str) -> str:
    stem = fragment.split(".", 1)[0]
    same_stem = _same_stem_dotted_sibling_marker(stem, fragment)
    following_stem = _following_numeric_parenthetical_stem_marker(stem)
    return rf"\((?:{same_stem}|{following_stem}(?:\.[0-9]+)*)\)"


def _following_numeric_parenthetical_stem_marker(stem: str) -> str:
    digits = str(int(stem))
    same_width_patterns: list[str] = []
    for idx, digit in enumerate(digits):
        lower = int(digit) + 1
        if idx == 0:
            lower = max(lower, 1)
        if lower > 9:
            continue
        prefix = re.escape(digits[:idx])
        suffix_len = len(digits) - idx - 1
        suffix = rf"[0-9]{{{suffix_len}}}" if suffix_len else ""
        same_width_patterns.append(rf"{prefix}[{lower}-9]{suffix}")
    longer_width = rf"[1-9][0-9]{{{len(digits)},}}"
    return "(?:" + "|".join([*same_width_patterns, longer_width]) + ")"


def _alpha_sibling_parenthetical_marker(fragment: str, *, top_level: bool) -> str:
    stem = fragment[0]
    same_stem = _same_stem_dotted_sibling_marker(stem, fragment)
    following_stem = (
        _next_alpha_parenthetical_stem_marker(stem)
        if top_level
        else _following_alpha_parenthetical_stem_marker(stem)
    )
    return rf"\((?:{same_stem}|{following_stem}(?:\.[0-9]+)*)\)"


def _next_alpha_parenthetical_stem_marker(stem: str) -> str:
    next_codepoint = ord(stem) + 1
    if stem.islower() and next_codepoint <= ord("z"):
        return chr(next_codepoint)
    if stem.isupper() and next_codepoint <= ord("Z"):
        return chr(next_codepoint)
    return r"\b\B"


def _following_alpha_parenthetical_stem_marker(stem: str) -> str:
    next_codepoint = ord(stem) + 1
    if stem.islower() and next_codepoint <= ord("z"):
        return f"[{chr(next_codepoint)}-z]"
    if stem.isupper() and next_codepoint <= ord("Z"):
        return f"[{chr(next_codepoint)}-Z]"
    return r"\b\B"


def _same_stem_dotted_sibling_marker(stem: str, fragment: str) -> str:
    escaped_stem = re.escape(stem)
    if "." not in fragment:
        return rf"{escaped_stem}(?:\.[0-9]+)+"
    suffix = re.escape(fragment.split(".", 1)[1])
    return rf"{escaped_stem}\.(?!{suffix}(?:\.|\)))[0-9]+(?:\.[0-9]+)*"


def _next_alpha_parenthetical_marker(fragment: str) -> str:
    next_codepoint = ord(fragment) + 1
    if fragment.islower() and next_codepoint <= ord("z"):
        return rf"\({chr(next_codepoint)}\)"
    if fragment.isupper() and next_codepoint <= ord("Z"):
        return rf"\({chr(next_codepoint)}\)"
    return r"\b\B"


def _candidate_corpus_citation_paths(identifier: str) -> tuple[str, ...]:
    """Return exact and nearest-parent corpus citation path candidates."""
    normalized = identifier.strip().strip("/")
    if not normalized:
        return ()

    try:
        if _looks_like_corpus_citation_path(normalized):
            primary = normalized
        else:
            primary = _citation_to_corpus_citation_path(normalized)
    except ValueError:
        primary = normalized

    candidates: list[str] = []

    def add(candidate: str) -> None:
        cleaned = candidate.strip().strip("/")
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    primary = _normalize_rulespec_source_id_to_corpus_path(primary)
    add(primary)
    parts = primary.split("/")
    for end in range(len(parts) - 1, 2, -1):
        add("/".join(parts[:end]))
    return tuple(candidates)


def _normalize_rulespec_source_id_to_corpus_path(identifier: str) -> str:
    """Convert ``us-ca:regulation/...`` source ids to corpus paths.

    RuleSpec source ids use a jurisdiction-prefixed root to keep generated file
    paths repository-relative, while corpus.provisions stores the same source as
    ``jurisdiction/document_class/...`` with singular document classes.
    """
    parts = [part for part in identifier.strip().strip("/").split("/") if part]
    if not parts or ":" not in parts[0]:
        return identifier
    jurisdiction, source_root = parts[0].split(":", 1)
    document_class = _CORPUS_DOCUMENT_CLASS_BY_SOURCE_TOKEN.get(source_root)
    if not jurisdiction or document_class is None:
        return identifier
    return "/".join((jurisdiction, document_class, *parts[1:]))


_CFR_CITATION_RE = re.compile(
    r"^(?P<title>\d+)\s+C\.?\s*F\.?\s*R\.?\s+"
    r"(?:(?:part|pt\.?)\s+)?(?:§+\s*)?"
    r"(?P<section>[0-9A-Za-z.-]+)"
    r"(?P<tail>(?:\([^)]+\))*)$",
    re.IGNORECASE,
)


def _citation_to_corpus_citation_path(citation: str) -> str:
    """Convert a bare legal citation to a corpus citation path."""
    cfr_path = _cfr_citation_to_corpus_citation_path(citation)
    if cfr_path is not None:
        return cfr_path
    if re.search(r"\bC\.?\s*F\.?\s*R\.?\b", citation, flags=re.IGNORECASE):
        raise ValueError(f"Could not parse CFR citation: {citation}")
    return citation_to_citation_path(citation)


def _cfr_citation_to_corpus_citation_path(citation: str) -> str | None:
    match = _CFR_CITATION_RE.match(citation.strip().replace("§", "§ "))
    if match is None:
        return None
    section_parts = [part for part in match.group("section").split(".") if part]
    if not section_parts:
        return None
    fragments = re.findall(r"\(([^)]+)\)", match.group("tail"))
    return "/".join(
        (
            "us",
            "regulation",
            match.group("title"),
            *section_parts,
            *fragments,
        )
    )


def _looks_like_corpus_citation_path(identifier: str) -> bool:
    parts = [part for part in identifier.strip().strip("/").split("/") if part]
    if parts and ":" in parts[0]:
        _jurisdiction, source_root = parts[0].split(":", 1)
        if source_root in _RULESPEC_SOURCE_ROOT_TOKENS:
            return True
    if len(parts) >= 2 and parts[0] in _RULESPEC_SOURCE_ROOT_TOKENS:
        return True
    return len(parts) >= 3 and parts[1] in {
        "form",
        "forms",
        "guidance",
        "manual",
        "manuals",
        "policies",
        "policy",
        "regulation",
        "regulations",
        # Lane-local T0 tariff addition: corpus stores Federal Register
        # instruments under us/rulemaking/...; without this token the CLI
        # mangles the citation via citation_to_citation_path into
        # us/statute/us/rulemaking/... and the corpus lookup fails.
        "rulemaking",
        "statute",
        "statutes",
    }


def _fetch_local_corpus_source_text_from_repo(
    citation_path: str,
    corpus_path: Path,
) -> str | None:
    normalized_path = citation_path.strip().strip("/")
    if not normalized_path:
        return None
    provisions_root = _corpus_provisions_root(corpus_path)
    if provisions_root is None:
        return None
    for provision_file in _candidate_local_corpus_provision_files(
        provisions_root,
        normalized_path,
    ):
        source_text = _read_local_corpus_provision_file(
            provision_file,
            normalized_path,
        )
        if source_text is not None:
            return source_text
        source_text = _read_local_corpus_descendant_text(
            provision_file,
            normalized_path,
        )
        if source_text is not None:
            return source_text
    return None


def _read_local_corpus_descendant_text(
    provision_file: Path,
    citation_path: str,
) -> str | None:
    """Read body-bearing child provisions for a metadata-only source document."""
    try:
        lines = provision_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    descendants: list[tuple[int, int, str, str, str]] = []
    child_prefix = f"{citation_path}/"
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        record_path = str(record.get("citation_path") or "")
        if not record_path.startswith(child_prefix):
            continue
        body = _local_corpus_record_text(record)
        if body is None:
            continue
        descendants.append(
            (
                int(record.get("level") or 0),
                int(record.get("ordinal") or 0),
                str(record.get("heading") or ""),
                record_path,
                body,
            )
        )

    if not descendants:
        return None
    chunks: list[str] = []
    for _, _, heading, _, body in sorted(descendants):
        if heading:
            chunks.append(f"{heading}\n\n{body}")
        else:
            chunks.append(body)
    return "\n\n".join(chunks)


def _corpus_provisions_root(corpus_path: Path) -> Path | None:
    root = Path(corpus_path).expanduser()
    candidates = (
        root / "data" / "corpus" / "provisions",
        root / "data" / "corpus",
        root / "provisions",
        root,
    )
    for candidate in candidates:
        provisions_root = (
            candidate if candidate.name == "provisions" else candidate / "provisions"
        )
        with contextlib.suppress(OSError):
            resolved = provisions_root.resolve()
            if resolved.is_dir():
                return resolved
    return None


def _materialize_resolved_definition_stub(
    *,
    context_root: Path,
    resolved_term: ResolvedDefinedTerm,
    workspace_root: Path,
) -> EvalContextFile:
    """Write one resolved definition stub into the eval workspace context."""
    relative_target = Path("context") / import_target_to_relative_rulespec_path(
        resolved_term.import_target
    )
    materialize_registered_stub(
        workspace_root,
        [resolved_term],
        prefix=Path("context"),
    )
    return EvalContextFile(
        source_path=resolved_term.citation,
        workspace_path=str(relative_target),
        import_path=_relative_rulespec_path_to_import_target(
            relative_target.relative_to("context")
        ),
        kind="definition_stub",
        label=resolved_term.label,
    )


def _materialize_resolved_canonical_concept(
    *,
    context_root: Path,
    resolved_concept: ResolvedCanonicalConcept,
    workspace_root: Path,
) -> EvalContextFile:
    """Copy one resolved canonical concept file into the eval workspace context."""
    relative_target = Path("context") / import_target_to_relative_rulespec_path(
        resolved_concept.import_target
    )
    target = workspace_root / relative_target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved_concept.rulespec_file, target)
    import_path = resolved_concept.import_target.split("#", 1)[0]
    return EvalContextFile(
        source_path=str(resolved_concept.rulespec_file),
        workspace_path=str(relative_target),
        import_path=import_path,
        kind="canonical_concept",
        label=resolved_concept.label,
    )


def _materialize_resolved_canonical_concept_companion_test(
    *,
    context_item: EvalContextFile,
    resolved_concept: ResolvedCanonicalConcept,
    workspace_root: Path,
) -> EvalContextFile | None:
    """Copy the companion test for a resolved canonical concept when available."""
    test_source = _rulespec_test_path(resolved_concept.rulespec_file)
    if not test_source.exists():
        return None

    test_import_path = f"{context_item.import_path}.test"
    relative_target = Path("context") / import_target_to_relative_rulespec_path(
        test_import_path
    )
    target = workspace_root / relative_target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(test_source, target)
    return EvalContextFile(
        source_path=str(test_source),
        workspace_path=str(relative_target),
        import_path=test_import_path,
        kind="implementation_test_context",
    )


def _auto_select_context_files(citation: str, policy_root: Path) -> list[Path]:
    """Best-effort auto-context selection for target files and nearby precedent."""
    selected: list[Path] = []
    target_rel = _target_rel_for_eval_identifier(citation)
    if target_rel is not None:
        target_path = policy_root / target_rel
        if target_path.exists():
            selected.append(target_path)
        if target_path.parent.exists():
            for sibling in sorted(target_path.parent.glob("*.yaml")):
                if sibling.name.endswith(".test.yaml") or sibling == target_path:
                    continue
                selected.append(sibling)
                if len(selected) >= 6:
                    break
        if selected:
            return selected[:6]

    try:
        return select_context_files(citation, policy_root)
    except Exception:
        return []


def _select_same_section_subsection_context_files(
    citation: str,
    source_text: str,
    policy_root: Path,
) -> list[Path]:
    """Select existing RuleSpecs for same-section subsections cited by source text."""
    parts = _citation_parts_for_eval_identifier(citation)
    if parts is None:
        return []

    target_rel = citation_to_relative_rulespec_path(parts)
    selected: list[Path] = []
    seen: set[Path] = set()
    for referenced_fragments in _iter_same_section_subsection_references(source_text):
        if referenced_fragments[0] in parts.fragments:
            continue
        for length in range(len(referenced_fragments), 0, -1):
            candidate_rel = citation_to_relative_rulespec_path(
                CitationParts(
                    parts.title,
                    parts.section,
                    referenced_fragments[:length],
                )
            )
            if candidate_rel == target_rel:
                continue
            candidates = _cited_context_candidates(
                policy_root,
                candidate_rel,
                max_child_candidates=None,
                include_exporting_candidate_children=True,
            )
            if not candidates:
                continue
            for candidate in candidates:
                resolved = candidate.resolve()
                if resolved not in seen:
                    selected.append(candidate)
                    seen.add(resolved)
            break
    return selected


def _citation_parts_for_eval_identifier(citation: str) -> CitationParts | None:
    """Return citation parts for either USC text or corpus-backed eval identifiers."""
    target_rel = _target_rel_for_eval_identifier(citation)
    if target_rel is None:
        return None
    path_parts = list(target_rel.parts)
    if len(path_parts) < 3 or path_parts[0] != "statutes":
        return None
    if len(path_parts) == 3:
        return CitationParts(
            title=path_parts[1],
            section=Path(path_parts[2]).stem,
            fragments=(),
        )
    return CitationParts(
        title=path_parts[1],
        section=path_parts[2],
        fragments=tuple((*path_parts[3:-1], Path(path_parts[-1]).stem)),
    )


_SUBSECTION_REFERENCE_START_RE = re.compile(r"\bsubsections?\s+", re.IGNORECASE)
_PARENTHETICAL_FRAGMENT_RE = re.compile(r"\s*\((?P<fragment>[A-Za-z0-9.]+)\)")
_SUBSECTION_REFERENCE_SEPARATOR_RE = re.compile(
    r"\s*(?:,\s*(?:(?:and|or)\b)?|\band\b|\bor\b)\s*",
    re.IGNORECASE,
)


def _iter_same_section_subsection_references(
    source_text: str,
) -> Iterator[tuple[str, ...]]:
    """Yield subsection fragment paths cited after `subsection(s)` in source text."""
    for match in _SUBSECTION_REFERENCE_START_RE.finditer(source_text):
        position = match.end()
        while True:
            fragments: list[str] = []
            while fragment_match := _PARENTHETICAL_FRAGMENT_RE.match(
                source_text, position
            ):
                fragment = fragment_match.group("fragment").strip()
                if not fragment:
                    break
                fragments.append(fragment)
                position = fragment_match.end()

            if not fragments:
                break
            yield tuple(fragments)

            separator = _SUBSECTION_REFERENCE_SEPARATOR_RE.match(source_text, position)
            if separator is None:
                break
            next_position = separator.end()
            if _PARENTHETICAL_FRAGMENT_RE.match(source_text, next_position) is None:
                break
            position = next_position


def _select_cross_section_context_files(
    citation: str,
    source_text: str,
    policy_root: Path,
) -> list[Path]:
    """Select existing RuleSpecs for cited sections outside this section."""
    current_section = _section_for_eval_identifier(citation)
    if current_section is None:
        return []

    target_rel = _target_rel_for_eval_identifier(citation)
    if target_rel is None:
        return []
    selected: list[Path] = []
    seen: set[Path] = set()
    for cited_parts, _start, _end in _iter_cited_usc_sections(source_text):
        if cited_parts.section == current_section:
            continue
        for candidate_rel in _cited_context_candidate_paths_for_eval_identifier(
            citation,
            cited_parts,
        ):
            if candidate_rel == target_rel:
                continue
            candidates = _cited_context_candidates(
                policy_root,
                candidate_rel,
                include_exporting_candidate_children=(
                    _source_text_requests_cited_subsection_rates(source_text)
                ),
            )
            if not candidates:
                continue
            for candidate in candidates:
                resolved = candidate.resolve()
                if resolved not in seen:
                    selected.append(candidate)
                    seen.add(resolved)
            break
    return selected


def _section_for_eval_identifier(citation: str) -> str | None:
    """Return the current legal section for USC or corpus-backed identifiers."""
    if _looks_like_corpus_citation_path(citation):
        target_rel = _target_rel_for_eval_identifier(citation)
        if target_rel is None:
            return None
        parts = list(target_rel.parts)
        if len(parts) >= 3 and parts[0] in {"regulations", "statutes"}:
            return Path(parts[2]).stem
        return None

    try:
        return parse_usc_citation(citation).section
    except Exception:
        return None


def _cited_context_candidate_paths_for_eval_identifier(
    citation: str,
    cited_parts: CitationParts,
) -> list[Path]:
    """Return candidate RuleSpec paths for a source-text section citation."""
    if not _looks_like_corpus_citation_path(citation):
        try:
            current_parts = parse_usc_citation(citation)
        except Exception:
            current_parts = None
        if current_parts is None:
            return []
        return _cited_context_candidate_paths(
            CitationParts(
                current_parts.title,
                cited_parts.section,
                cited_parts.fragments,
            )
        )

    target_rel = _target_rel_for_eval_identifier(citation)
    if target_rel is None:
        return []
    return _same_source_cited_context_candidate_paths(target_rel, cited_parts)


def _same_source_cited_context_candidate_paths(
    target_rel: Path,
    cited_parts: CitationParts,
) -> list[Path]:
    """Return same-instrument candidates for non-USC corpus-backed citations."""
    parts = list(target_rel.parts)
    if len(parts) >= 3 and parts[0] == "regulations":
        base = Path(parts[0]) / parts[1]
        return _dotted_section_context_candidate_paths(base, cited_parts)
    if len(parts) >= 3 and parts[0] == "statutes":
        return _cited_context_candidate_paths(
            CitationParts(parts[1], cited_parts.section, cited_parts.fragments)
        )
    return []


def _dotted_section_context_candidate_paths(
    base: Path,
    cited_parts: CitationParts,
) -> list[Path]:
    """Return exact and dotted-ancestor candidates for state regulation sections."""
    paths: list[Path] = []
    section = cited_parts.section
    for length in range(len(cited_parts.fragments), -1, -1):
        paths.append(
            _dotted_section_context_path(
                base,
                section,
                cited_parts.fragments[:length],
            )
        )
    if re.fullmatch(r"\d+(?:\.\d+)+", section):
        segments = section.split(".")
        for length in range(len(segments) - 1, 0, -1):
            paths.append(base / f"{'.'.join(segments[:length])}.yaml")

    unique: list[Path] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def _dotted_section_context_path(
    base: Path,
    section: str,
    fragments: tuple[str, ...],
) -> Path:
    if not fragments:
        return base / f"{section}.yaml"
    return base / section / Path(*fragments[:-1]) / f"{fragments[-1]}.yaml"


def _source_text_requests_cited_subsection_rates(source_text: str) -> bool:
    """Return whether source asks for rates under cited section subsections."""
    return bool(
        re.search(
            r"\b(?:rate|rates|percentage|percentages)\b.{0,120}"
            r"\bsubsections?\b.{0,120}\bsection\b",
            source_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        or re.search(
            r"\bsection\b.{0,120}\bsubsections?\b.{0,120}"
            r"\b(?:rate|rates|percentage|percentages)\b",
            source_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )


def _cited_context_candidate_paths(cited_parts: CitationParts) -> list[Path]:
    """Return exact and ancestor RuleSpec paths for a cited USC provision."""
    paths: list[Path] = []
    for length in range(len(cited_parts.fragments), -1, -1):
        candidate_parts = CitationParts(
            cited_parts.title,
            cited_parts.section,
            cited_parts.fragments[:length],
        )
        paths.append(citation_to_relative_rulespec_path(candidate_parts))
    return paths


def _iter_cited_usc_sections(
    source_text: str,
) -> Iterator[tuple[CitationParts, int, int]]:
    """Yield USC section citations, including list tails like `sections 911, 931, or 933`."""
    for match in re.finditer(
        r"\bsections?\s+"
        r"(?P<section>[0-9][A-Za-z0-9.-]*)"
        r"(?P<fragments>(?:\([A-Za-z0-9]+\))*)",
        source_text,
        flags=re.IGNORECASE,
    ):
        if _citation_match_points_to_other_act(source_text, match.end()):
            continue
        yield _citation_parts_from_match(match), match.start(), match.end()
        position = match.end()
        while True:
            tail = re.match(
                r"\s*(?:,\s*(?:(?:and|or)\b)?|\band\b|\bor\b)\s*"
                r"(?:sections?\s+)?"
                r"(?P<section>[0-9][A-Za-z0-9.-]*)"
                r"(?P<fragments>(?:\([A-Za-z0-9]+\))*)",
                source_text[position:],
                flags=re.IGNORECASE,
            )
            if tail is None:
                break
            tail_start = position + tail.start()
            tail_end = position + tail.end()
            if _citation_match_points_to_other_act(source_text, tail_end):
                break
            yield _citation_parts_from_match(tail), tail_start, tail_end
            position = tail_end


def _citation_parts_from_match(match: re.Match[str]) -> CitationParts:
    section = match.group("section").rstrip(".")
    fragments = tuple(re.findall(r"\(([A-Za-z0-9]+)\)", match.group("fragments")))
    return CitationParts("", section, fragments)


def _cited_context_candidates(
    policy_root: Path,
    candidate_rel: Path,
    *,
    max_child_candidates: int | None = 8,
    include_exporting_candidate_children: bool = False,
) -> list[Path]:
    """Return an exact cited RuleSpec file or child fragments when only a section exists."""
    candidate = policy_root / candidate_rel
    child_root = policy_root / candidate_rel.with_suffix("")
    child_candidates: list[Path] = []
    if child_root.is_dir():
        child_candidates = [
            path
            for path in child_root.rglob("*.yaml")
            if not path.name.endswith(".test.yaml")
        ]
        child_candidates.sort(
            key=lambda path: (len(path.relative_to(child_root).parts), str(path))
        )
        if max_child_candidates is not None:
            child_candidates = child_candidates[:max_child_candidates]

    if candidate.exists():
        if include_exporting_candidate_children and child_candidates:
            return [candidate, *child_candidates]
        if _context_file_exports(str(candidate)) or not child_candidates:
            return [candidate]
        return [candidate, *child_candidates]

    return child_candidates


def _select_child_fragment_context_files(
    citation: str, policy_root: Path
) -> list[Path]:
    """Select existing child RuleSpecs when encoding an aggregate parent fragment."""
    target_rel = _target_rel_for_eval_identifier(citation)
    if target_rel is None:
        return []
    child_root = policy_root / target_rel.with_suffix("")
    if not child_root.is_dir():
        return []
    return sorted(
        path
        for path in child_root.rglob("*.yaml")
        if not path.name.endswith(".test.yaml")
    )


def _repo_augmented_context_root(policy_path: Path) -> Path:
    """Resolve the corpus root used for automatic repo-augmented context selection."""
    resolved = Path(policy_path).resolve()
    if resolved.name == "axiom-rules-engine":
        fallback = resolved.parent / "rulespec-us"
        if fallback.exists():
            return jurisdiction_content_dir(fallback, jurisdiction_prefix(fallback))
    return jurisdiction_content_dir(resolved, jurisdiction_prefix(resolved))


def _context_import_relative_target(source_path: Path, policy_path: Path) -> Path:
    """Prefer canonical repo-relative import targets for copied precedent files."""
    repo_parent = policy_path.parent.resolve()
    resolved_source = source_path.resolve()
    source_policy_root = find_policy_repo_root(resolved_source)
    if source_policy_root is not None:
        with contextlib.suppress(ValueError):
            return resolved_source.relative_to(source_policy_root.resolve())

    for candidate in sorted(repo_parent.glob("rulespec-*")):
        if not candidate.is_dir():
            continue
        resolved_candidate = candidate.resolve()
        with contextlib.suppress(ValueError):
            relative = resolved_source.relative_to(resolved_candidate)
            return relative

    return Path("external") / resolved_source.name


def _context_import_target(source_path: Path, relative_target: Path) -> str:
    """Return the canonical RuleSpec import target for a copied context file."""
    prefix = _rulespec_repo_import_prefix(source_path)
    return _relative_rulespec_path_to_import_target(relative_target, prefix=prefix)


def _rulespec_repo_import_prefix(source_path: Path) -> str | None:
    """Infer the absolute RuleSpec import prefix from a `rulespec-*` repo path."""
    repo_name = canonical_rulespec_repo_name(source_path)
    if repo_name and len(repo_name) > len("rulespec-"):
        return repo_name.removeprefix("rulespec-")
    return None


def _relative_rulespec_path_to_import_target(
    path: Path,
    *,
    prefix: str | None = None,
) -> str:
    """Convert a relative RuleSpec file path into an import target."""
    normalized = path.with_suffix("") if path.suffix in {".yaml", ".yml"} else path
    target = normalized.as_posix()
    return f"{prefix}:{target}" if prefix else target


def _target_rel_for_eval_identifier(citation: str) -> Path | None:
    """Return the canonical RuleSpec target path for corpus or bare citations."""
    normalized = citation.strip().strip("/")
    first_part = normalized.split("/", 1)[0]
    if ":" in first_part:
        _jurisdiction, source_root = first_part.split(":", 1)
        if source_root in _RULESPEC_SOURCE_ROOT_TOKENS:
            rest = normalized.split(":", 1)[1]
            return _source_identifier_to_relative_rulespec_path(rest)
    if _looks_like_corpus_citation_path(normalized):
        return _source_identifier_to_relative_rulespec_path(normalized)
    try:
        return _source_identifier_to_relative_rulespec_path(
            _citation_to_corpus_citation_path(citation)
        )
    except Exception:
        return None


def _policy_repo_root_for_corpus_source(
    corpus_citation_path: str,
    axiom_rules_path: Path,
) -> Path:
    """Choose the jurisdiction RuleSpec repo for a corpus citation path."""
    jurisdiction = corpus_citation_path.strip().split("/", 1)[0] or "us"
    repo_name = "rulespec-us" if jurisdiction == "us" else f"rulespec-{jurisdiction}"
    if axiom_rules_path.name == repo_name:
        return axiom_rules_path
    candidate = axiom_rules_path.parent / repo_name
    return candidate if candidate.exists() else axiom_rules_path


def evaluate_artifact(
    rulespec_file: Path,
    policy_repo_root: Path,
    axiom_rules_path: Path,
    source_text: str,
    oracle: EvalOracleMode = "none",
    policyengine_country: str = "auto",
    policyengine_rule_hint: str | None = None,
    skip_reviewers: bool = False,
) -> EvalArtifactMetrics:
    """Evaluate one RuleSpec artifact with deterministic checks plus optional oracles."""
    with _rulespec_validation_target(
        rulespec_file, policy_repo_root
    ) as validation_file:
        validation_policy_repo_root = _validation_policy_repo_root(
            validation_file, policy_repo_root
        )
        pipeline = ValidatorPipeline(
            policy_repo_path=validation_policy_repo_root,
            axiom_rules_path=axiom_rules_path,
            enable_oracles=oracle != "none",
            policyengine_country=policyengine_country,
            policyengine_rule_hint=policyengine_rule_hint,
            require_policy_proofs=True,
            source_text=source_text,
        )
        compile_result = pipeline._run_compile_check(validation_file)
        ci_result = pipeline._run_ci(validation_file)

        policyengine_result = None
        taxsim_result = None
        if oracle in ("policyengine", "all"):
            try:
                policyengine_result = pipeline._run_policyengine(validation_file)
            except Exception as exc:
                policyengine_result = ValidationResult(
                    validator_name="policyengine",
                    passed=False,
                    error=str(exc),
                    issues=[str(exc)],
                )
        if oracle == "all":
            try:
                taxsim_result = pipeline._run_taxsim(validation_file)
            except Exception as exc:
                taxsim_result = ValidationResult(
                    validator_name="taxsim",
                    passed=False,
                    error=str(exc),
                    issues=[str(exc)],
                )

        oracle_context: dict[str, dict[str, object]] = {}
        if policyengine_result is not None:
            oracle_context["policyengine"] = {
                "score": policyengine_result.score,
                "passed": policyengine_result.passed,
                "issues": policyengine_result.issues,
                "duration_ms": policyengine_result.duration_ms,
            }
        if taxsim_result is not None:
            oracle_context["taxsim"] = {
                "score": taxsim_result.score,
                "passed": taxsim_result.passed,
                "issues": taxsim_result.issues,
                "duration_ms": taxsim_result.duration_ms,
            }
        review_context = (
            "This review is running inside an eval-suite benchmark workspace. "
            "The artifact file path is generic benchmark output and is not itself the legal citation. "
            "Benchmark directory labels may be stale, generic, or misleading and must be ignored as legal cues. "
            "The benchmark target is an atomic source slice/unit, so judge fidelity to exactly this source text rather than demanding omitted sibling limbs or parent consequences unless the RuleSpec claims to encode them. "
            "Judge citation fidelity against the embedded source-text docstring and this authoritative source excerpt:\n\n"
            f"{source_text.strip()[:4000]}"
        )
        if re.search(
            r"\bon the first day\b|\bnext benefit week\b|\bon or after the day\b",
            source_text,
            flags=re.IGNORECASE,
        ):
            review_context += (
                "\n\nThis is a temporal timing clause. The RuleSpec eval path does not expose a native date-valued output here, "
                "so a boolean day-predicate helper on `period: Day`, plus explicit trigger preconditions from the source text, "
                "is an acceptable representation."
            )
        if skip_reviewers:
            generalist_review_result = ValidationResult(
                validator_name="generalist-reviewer",
                passed=True,
                issues=[],
            )
        else:
            try:
                generalist_review_result = pipeline._run_reviewer(
                    "generalist-reviewer",
                    validation_file,
                    oracle_context or None,
                    review_context=review_context,
                )
            except Exception as exc:
                generalist_review_result = ValidationResult(
                    validator_name="generalist-reviewer",
                    passed=False,
                    error=str(exc),
                    issues=[f"Reviewer error: {exc}"],
                )

    content = rulespec_file.read_text()
    embedded_source = extract_embedded_source_text(content)
    proof_excerpt_text = _extract_proof_source_excerpt_text(content)
    numeric_source_text = extract_numeric_grounding_source_text(content)
    if not numeric_source_text and source_text:
        numeric_source_text = source_text
    numeric_validation_source_text = embedded_source or numeric_source_text
    numeric_grounding_validation_source_text = numeric_validation_source_text
    if source_text and _has_parameter_table_proof_atom(content):
        numeric_grounding_validation_source_text = "\n".join(
            part for part in (source_text, numeric_validation_source_text) if part
        )
    numeric_grounding_source_text = "\n".join(
        part
        for part in (numeric_grounding_validation_source_text, proof_excerpt_text)
        if part
    )
    source_numbers = extract_numbers_from_text(numeric_grounding_source_text or "")
    source_numeric_occurrences = Counter(
        extract_numeric_occurrences_from_text(
            _numeric_occurrence_source_text(numeric_validation_source_text or "")
        )
    )
    if _is_empty_nonassertable_artifact(content):
        source_numeric_occurrences = Counter()
    named_scalar_occurrences = Counter(
        item.value for item in extract_named_scalar_occurrences(content)
    )
    named_scalar_occurrences.update(_numeric_concept_name_occurrences(content))
    named_scalar_occurrences.update(_verification_value_numeric_occurrences(content))
    named_scalar_occurrences.update(_deferred_output_numeric_occurrences(content))
    named_scalar_occurrences.update(
        _imported_named_scalar_occurrences(content, policy_repo_root)
    )
    source_is_table = _source_text_looks_like_table(
        numeric_grounding_validation_source_text or ""
    )
    inline_table_formula_occurrences = (
        _inline_table_formula_numeric_occurrences(content)
        if source_is_table
        else Counter()
    )

    grounding_metrics: list[GroundingMetric] = []
    for line, raw, value in extract_grounding_values(content):
        grounding_metrics.append(
            GroundingMetric(
                line=line,
                raw=raw,
                value=value,
                grounded=numeric_value_is_grounded(value, source_numbers),
            )
        )

    numeric_occurrence_issues: list[str] = []
    covered_source_numeric_occurrence_count = 0
    missing_source_numeric_occurrence_count = 0
    for value, expected_count in sorted(source_numeric_occurrences.items()):
        covered_count = (
            expected_count
            if _matching_numeric_occurrence_count(named_scalar_occurrences, value)
            else 0
        )
        if inline_table_formula_occurrences.get(value):
            covered_count = max(covered_count, expected_count)
        covered_source_numeric_occurrence_count += covered_count
        if covered_count < expected_count:
            missing_count = expected_count - covered_count
            missing_source_numeric_occurrence_count += missing_count
            numeric_occurrence_issues.append(
                f"Source numeric value {value:g} appears {expected_count} time(s), "
                f"but only {covered_count} named scalar definition(s) with that value were found."
            )

    ungrounded_numeric_issues = find_ungrounded_numeric_issues(
        content,
        numeric_grounding_source_text,
    )
    admin_agency_aggregate_issues = find_admin_agency_aggregate_entity_issues(
        content,
        source_text,
    )
    policyengine_hint_upstream_issues = _policyengine_hint_upstream_composition_issues(
        content,
        policyengine_rule_hint,
    )
    ci_issues = []
    seen_ci_issues: set[str] = set()
    for issue in (
        list(ci_result.issues)
        + ungrounded_numeric_issues
        + numeric_occurrence_issues
        + admin_agency_aggregate_issues
        + policyengine_hint_upstream_issues
    ):
        if issue in seen_ci_issues:
            continue
        ci_issues.append(issue)
        seen_ci_issues.add(issue)
    ci_pass = (
        ci_result.passed
        and not ungrounded_numeric_issues
        and not numeric_occurrence_issues
        and not admin_agency_aggregate_issues
        and not policyengine_hint_upstream_issues
    )

    return EvalArtifactMetrics(
        compile_pass=compile_result.passed,
        compile_issues=compile_result.issues,
        ci_pass=ci_pass,
        ci_issues=ci_issues,
        embedded_source_present=bool(embedded_source),
        grounded_numeric_count=sum(1 for item in grounding_metrics if item.grounded),
        ungrounded_numeric_count=sum(
            1 for item in grounding_metrics if not item.grounded
        ),
        grounding=grounding_metrics,
        source_numeric_occurrence_count=sum(source_numeric_occurrences.values()),
        covered_source_numeric_occurrence_count=covered_source_numeric_occurrence_count,
        missing_source_numeric_occurrence_count=missing_source_numeric_occurrence_count,
        numeric_occurrence_issues=numeric_occurrence_issues,
        generalist_review_pass=generalist_review_result.passed,
        generalist_review_score=generalist_review_result.score,
        generalist_review_issues=generalist_review_result.issues,
        generalist_review_prompt_sha256=generalist_review_result.prompt_sha256,
        policyengine_pass=(
            policyengine_result.passed if policyengine_result is not None else None
        ),
        policyengine_score=(
            policyengine_result.score if policyengine_result is not None else None
        ),
        policyengine_issues=(
            policyengine_result.issues if policyengine_result is not None else []
        ),
        taxsim_pass=taxsim_result.passed if taxsim_result is not None else None,
        taxsim_score=taxsim_result.score if taxsim_result is not None else None,
        taxsim_issues=taxsim_result.issues if taxsim_result is not None else [],
    )


def _evaluate_generated_artifact_with_repairs(
    rulespec_file: Path,
    policy_repo_root: Path,
    axiom_rules_path: Path,
    source_text: str,
    oracle: EvalOracleMode = "none",
    policyengine_country: str = "auto",
    policyengine_rule_hint: str | None = None,
    skip_reviewers: bool = False,
) -> EvalArtifactMetrics | None:
    metrics = evaluate_artifact(
        rulespec_file=rulespec_file,
        policy_repo_root=policy_repo_root,
        axiom_rules_path=axiom_rules_path,
        source_text=source_text,
        oracle=oracle,
        policyengine_country=policyengine_country,
        policyengine_rule_hint=policyengine_rule_hint,
        skip_reviewers=skip_reviewers,
    )
    if metrics is None:
        return None
    repairs = _apply_generated_eval_repairs(
        rulespec_file=rulespec_file,
        policy_repo_root=policy_repo_root,
        axiom_rules_path=axiom_rules_path,
        issues=metrics.ci_issues,
    )
    if not repairs:
        return metrics
    return evaluate_artifact(
        rulespec_file=rulespec_file,
        policy_repo_root=policy_repo_root,
        axiom_rules_path=axiom_rules_path,
        source_text=source_text,
        oracle=oracle,
        policyengine_country=policyengine_country,
        policyengine_rule_hint=policyengine_rule_hint,
        skip_reviewers=skip_reviewers,
    )


_EVAL_COMPANION_REPAIR_MARKERS = (
    "Judgment rule missing positive companion output coverage:",
    "Derived rule missing companion output coverage:",
    "Zero branch test coverage missing:",
    "mixes derived output entities",
    "input invalid: relation ",
)
_EVAL_SCALAR_REPAIR_MARKERS = (
    "Proof import not referenced:",
    "Unused import `",
)


def _apply_generated_eval_repairs(
    *,
    rulespec_file: Path,
    policy_repo_root: Path,
    axiom_rules_path: Path,
    issues: list[str],
) -> list[str]:
    """Apply deterministic generated-artifact repairs before final eval scoring."""
    repairs: list[str] = []
    proof_repairs = _remove_unreferenced_proof_import_atoms(rulespec_file, issues)
    repairs.extend(f"proof_import:{name}" for name in proof_repairs)
    unused_import_repairs = _prune_unused_imports_from_file(rulespec_file, issues)
    repairs.extend(f"unused_import:{name}" for name in unused_import_repairs)

    companion_issues = [
        issue
        for issue in issues
        if any(marker in str(issue) for marker in _EVAL_COMPANION_REPAIR_MARKERS)
    ]
    if not companion_issues:
        return repairs

    relative_output = _relative_rulespec_source_path(rulespec_file)
    test_file = _rulespec_test_path(rulespec_file)
    if relative_output is None or not test_file.exists():
        return repairs

    # Reuse the CLI's deterministic companion-test repair helpers lazily to
    # avoid an import cycle: cli imports this module during startup.
    from axiom_encode import cli as cli_helpers

    scalar_relation_issues = []
    for issue in companion_issues:
        parsed = cli_helpers._parse_scalar_relation_row_issue(str(issue))
        if parsed is not None:
            scalar_relation_issues.append(parsed)
    if scalar_relation_issues:
        repairs.extend(
            f"relation_row:{name}"
            for name in cli_helpers._repair_scalar_relation_rows(
                rules_file=rulespec_file,
                test_file=test_file,
                policy_repo_path=policy_repo_root,
                parsed_issues=scalar_relation_issues,
            )
        )

    if any("mixes derived output entities" in str(issue) for issue in companion_issues):
        repairs.extend(
            f"mixed_entity:{name}"
            for name in cli_helpers._repair_mixed_derived_entity_output_tests(
                rules_file=rulespec_file,
                test_file=test_file,
                repo_path=policy_repo_root,
                relative_output=relative_output,
            )
        )
    repairs.extend(
        f"derived_output:{name}"
        for name in cli_helpers._append_generated_derived_output_tests_if_missing(
            rules_file=rulespec_file,
            test_file=test_file,
            repo_path=policy_repo_root,
            relative_output=relative_output,
            issues=companion_issues,
        )
    )
    repairs.extend(
        f"judgment_positive:{name}"
        for name in cli_helpers._append_generated_judgment_positive_tests_if_missing(
            rules_file=rulespec_file,
            test_file=test_file,
            repo_path=policy_repo_root,
            axiom_rules_path=axiom_rules_path,
            relative_output=relative_output,
            issues=companion_issues,
            test_failure_checker=cli_helpers._rulespec_companion_test_failures,
        )
    )
    repairs.extend(
        f"zero_branch:{name}"
        for name in cli_helpers._append_generated_zero_branch_tests_if_missing(
            rules_file=rulespec_file,
            test_file=test_file,
            repo_path=policy_repo_root,
            relative_output=relative_output,
            issues=companion_issues,
        )
    )
    return repairs


def _validation_policy_repo_root(validation_file: Path, policy_repo_root: Path) -> Path:
    """Return the repo root that contains the validation copy."""
    repo_names = {
        name
        for name in (
            policy_repo_root.name,
            canonical_rulespec_repo_name(policy_repo_root),
        )
        if name
    }
    jurisdiction = jurisdiction_prefix(policy_repo_root)
    for parent in validation_file.parents:
        if parent.name in repo_names:
            content_root = jurisdiction_content_dir(parent, jurisdiction)
            if _is_under_root(validation_file, content_root):
                return content_root
            return parent
    discovered = find_policy_repo_root(validation_file)
    if discovered is not None:
        return discovered
    return policy_repo_root


def _imported_named_scalar_occurrences(
    content: str,
    policy_repo_root: Path,
) -> Counter[float]:
    """Count numeric values from imported RuleSpec files and import symbols."""
    occurrences: Counter[float] = Counter()
    seen: set[Path] = set()
    for import_target in _extract_import_targets(content):
        if "#" in import_target:
            occurrences.update(
                _numeric_occurrences_from_concept_text(import_target.rsplit("#", 1)[1])
            )
        for path in _candidate_import_rule_files(import_target, policy_repo_root):
            resolved = path.resolve()
            if resolved in seen or not path.exists():
                continue
            seen.add(resolved)
            with contextlib.suppress(OSError):
                imported_content = path.read_text()
                occurrences.update(
                    item.value
                    for item in extract_named_scalar_occurrences(imported_content)
                )
                occurrences.update(_numeric_concept_name_occurrences(imported_content))
            break
    return occurrences


def _inline_table_formula_numeric_occurrences(content: str) -> Counter[float]:
    """Count inline formula literals that are allowed as structural table bounds."""
    occurrences: Counter[float] = Counter()
    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if not (
            isinstance(payload, dict)
            and payload.get("format") == "rulespec/v1"
            and isinstance(payload.get("rules"), list)
        ):
            return occurrences
        for rule in payload["rules"]:
            if not isinstance(rule, dict):
                continue
            versions = rule.get("versions")
            if not isinstance(versions, list):
                continue
            for version in versions:
                if not isinstance(version, dict):
                    continue
                formula = version.get("formula")
                if not isinstance(formula, str):
                    continue
                occurrences.update(_formula_comparator_numeric_values(formula))
    return occurrences


def _formula_comparator_numeric_values(formula_text: str) -> list[float]:
    """Return numeric literals used as comparison bounds in a formula."""
    values: list[float] = []
    for line in formula_text.splitlines():
        cleaned = line.split("#", 1)[0]
        for match in TABLE_BOUND_COMPARATOR_NUMBER_PATTERN.finditer(cleaned):
            raw = (match.group(1) or match.group(2) or "").replace(",", "")
            with contextlib.suppress(ValueError):
                values.append(float(raw))
    return values


def _candidate_import_rule_files(
    import_target: str,
    policy_repo_root: Path,
) -> list[Path]:
    """Return possible local files for an import target."""
    target_path = _import_target_to_path(import_target)
    candidates = [policy_repo_root / target_path]
    import_prefix = _import_target_prefix(import_target)
    if import_prefix:
        candidates.append(
            policy_repo_root.parent / f"rulespec-{import_prefix}" / target_path
        )
    return candidates


@contextlib.contextmanager
def _rulespec_validation_target(
    rulespec_file: Path, policy_repo_root: Path
) -> Iterator[Path]:
    """Yield a validation path whose ancestors expose canonical repo identity."""
    if _is_under_root(rulespec_file, policy_repo_root):
        yield rulespec_file
        return
    relative = _relative_rulespec_source_path(rulespec_file)
    if relative is None:
        yield rulespec_file
        return

    source_repo_root = policy_repo_root
    if not source_repo_root.name.startswith("rulespec-"):
        maybe_monorepo_root = source_repo_root.parent
        if maybe_monorepo_root.name.startswith(
            "rulespec-"
        ) and source_repo_root.name in jurisdiction_subdir_names(maybe_monorepo_root):
            source_repo_root = maybe_monorepo_root
        else:
            yield rulespec_file
            return

    with tempfile.TemporaryDirectory() as tmpdir:
        overlay_parent = Path(tmpdir)
        for sibling in source_repo_root.parent.glob("rulespec-*"):
            if sibling.resolve() == source_repo_root.resolve() or not sibling.is_dir():
                continue
            sibling_target = overlay_parent / sibling.name
            try:
                sibling_target.symlink_to(sibling.resolve(), target_is_directory=True)
            except OSError:
                shutil.copytree(sibling, sibling_target, dirs_exist_ok=True)
        overlay_repo_name = (
            canonical_rulespec_repo_name(source_repo_root) or source_repo_root.name
        )
        overlay_repo = overlay_parent / overlay_repo_name
        shutil.copytree(
            source_repo_root,
            overlay_repo,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", ".pytest_cache"
            ),
        )
        eval_workspaces_root = _nearby_eval_workspaces_root(rulespec_file)
        if eval_workspaces_root is not None:
            eval_overlay = overlay_parent / "_eval_workspaces"
            try:
                eval_overlay.symlink_to(
                    eval_workspaces_root.resolve(),
                    target_is_directory=True,
                )
            except OSError:
                shutil.copytree(eval_workspaces_root, eval_overlay)
        validation_content_root = jurisdiction_content_dir(
            overlay_repo,
            jurisdiction_prefix(policy_repo_root),
        )
        validation_file = validation_content_root / relative
        validation_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rulespec_file, validation_file)
        companion_test = rulespec_file.with_name(f"{rulespec_file.stem}.test.yaml")
        if companion_test.exists():
            validation_test = validation_file.with_name(
                f"{validation_file.stem}.test.yaml"
            )
            shutil.copy2(companion_test, validation_test)
        yield validation_file


def _nearby_eval_workspaces_root(path: Path) -> Path | None:
    for ancestor in path.parents:
        candidate = ancestor / "_eval_workspaces"
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _relative_rulespec_source_path(path: Path) -> Path | None:
    """Return the path beginning at the RuleSpec source-root directory."""
    parts = path.parts
    for index, part in enumerate(parts):
        if part in {"policies", "regulations", "statutes"}:
            return Path(*parts[index:])
    return None


_MINIMAL_SCOPE_HINT = """Source-scope protocol (minimal):
- Match each executable rule's `entity:` to the legal subject stated by the supplied source text.
- If the source uses the word "household" or "household's", use `entity: Household`. Prefer `Household` over `SnapUnit` for plain household-level SNAP rules.
- If the source says "households in which all members", "households with a member",
  or another household-level condition expressed through member facts, encode the
  executable eligibility/result on `Household`. Use person/member facts only as
  inputs or relation children needed to evaluate that household rule.
- If the source's legal subject is an "individual", "person", "member", "claimant",
  "child", "dependent", "spouse", or "employee" rather than a household/unit
  described through those people, use `entity: Person` (or `Employer` for employer amounts).
- Any rule that uses `entity: SnapUnit` (or another filtered entity) requires the same file to also declare that entity via a `kind: derived_relation` rule. The declaration shape is:

```yaml
- name: snap_unit
  kind: derived_relation
  derived_relation:
    arity: 2
    source_relation: us:statutes/7/2012/j#relation.member_of_household
    entity: SnapUnit
    member_relation: members
    slot_entities:
      - Person
      - Household
  source: us:statutes/7/2012/m
  versions:
    - effective_from: '<earliest applicable date>'
      formula: <member-eligibility predicate>
```

Either declare `snap_unit` inline like that (and define a real per-member eligibility predicate as its formula) or switch the executable rule's entity to `Household` and drop the SnapUnit reference. Do not leave a bare `entity: SnapUnit` rule without the matching declaration.

"""


def _strip_source_scope_protocol(prompt: str) -> str:
    """Replace the multi-page SOURCE_SCOPE_PROTOCOL block with a minimal hint.

    The full protocol guides entity-scope decisions for tax/credit/employer-style
    sources but adds ~600 lines of guidance that does not apply to simple
    single-entity judgment rules. For short SNAP/benefit sources, the bulk of
    the protocol pushes the model into emitting no artifact at all. On retry we
    swap the full protocol for a tight 4-bullet version that preserves the
    important entity-scope discipline (especially "do not invent SnapUnit") so
    the retry can succeed without regressing entity choices.
    """
    start_marker = "Source-scope protocol:"
    start = prompt.find(start_marker)
    if start == -1:
        return prompt
    end_marker = "Additional encoding guidance:"
    end = prompt.find(end_marker, start)
    if end == -1:
        return prompt
    return prompt[:start] + _MINIMAL_SCOPE_HINT + prompt[end:]


def _build_empty_artifact_retry_prompt(
    original_prompt: str,
    target_file_name: str,
    include_tests: bool,
) -> str:
    """Build the one-shot repair prompt for narrative-only eval responses."""
    output_contract = (
        f"Return exactly this two-file bundle and nothing else, beginning with "
        f"`=== FILE: {target_file_name} ===`."
        if include_tests
        else (
            f"Return only raw RuleSpec YAML for `{target_file_name}` beginning "
            "with `format: rulespec/v1`."
        )
    )
    trimmed_prompt = _strip_source_scope_protocol(original_prompt)
    return f"""The previous response did not contain a RuleSpec artifact, so the harness could not parse or write `{target_file_name}`.

Emit the artifact now.
- Do not narrate your plan.
- Do not explain what you will do.
- Do not include markdown prose, analysis, or file-write confirmations.
- {output_contract}

Use the same source, context, schema, and validation constraints from the original task below.

=== BEGIN ORIGINAL TASK ===
{trimmed_prompt}
=== END ORIGINAL TASK ===
"""


def _run_prompt_eval_with_empty_artifact_retry(
    runner: EvalRunnerSpec,
    workspace: EvalWorkspace,
    prompt: str,
    output_file: Path,
    source_text: str,
    target_file_name: str,
    include_tests: bool,
    policyengine_rule_hint: str | None = None,
) -> tuple[EvalPromptResponse, bool, int]:
    """Run an eval and retry once if no RuleSpec artifact can be materialized."""
    response = _run_prompt_eval(runner, workspace, prompt)
    wrote_artifact = _materialize_eval_artifact(
        response.text,
        output_file,
        source_text=source_text,
        workspace_root=workspace.root,
        policyengine_rule_hint=policyengine_rule_hint,
    )
    if wrote_artifact or not _response_allows_empty_artifact_retry(response):
        return response, wrote_artifact, 0

    retry_prompt = _build_empty_artifact_retry_prompt(
        prompt,
        target_file_name=target_file_name,
        include_tests=include_tests,
    )
    retry_response = _run_prompt_eval(runner, workspace, retry_prompt)
    retry_wrote_artifact = _materialize_eval_artifact(
        retry_response.text,
        output_file,
        source_text=source_text,
        workspace_root=workspace.root,
        policyengine_rule_hint=policyengine_rule_hint,
    )
    return (
        _combine_retry_response(response, retry_response, retry_prompt),
        retry_wrote_artifact,
        1,
    )


def _run_single_eval(
    citation: str,
    runner: EvalRunnerSpec,
    output_root: Path,
    policy_path: Path,
    runtime_axiom_rules_path: Path,
    corpus_path: Path,
    mode: EvalMode,
    extra_context_paths: list[Path],
    include_tests: bool = False,
    skip_reviewers: bool = False,
    policyengine_rule_hint: str | None = None,
) -> EvalResult:
    source_unit = resolve_corpus_source_unit(citation, corpus_path)
    continuations = _primary_source_continuations_from_context_paths(
        extra_context_paths
    )
    source_text = source_unit.body
    source_text = _append_primary_source_continuations(source_text, continuations)
    prompt_corpus_citation_path = _prompt_corpus_citation_path(source_unit)
    source_metadata_payload = _source_metadata_with_continuations(
        {
            "corpus_citation_path": prompt_corpus_citation_path,
            "corpus_source": source_unit.source,
            "requested_source": source_unit.requested,
            "resolved_corpus_citation_path": source_unit.citation_path,
        },
        continuations,
    )

    workspace = prepare_eval_workspace(
        citation=citation,
        runner=runner,
        output_root=output_root,
        source_text=source_text,
        axiom_rules_path=policy_path,
        mode=mode,
        source_metadata_payload=source_metadata_payload,
        extra_context_paths=extra_context_paths,
    )

    # Derive the output path from the *requested* identifier rather than the
    # *resolved* corpus citation_path. When the resolver falls back to a parent
    # provision (subsection-level corpus row missing), the resolved path drops
    # the subsection identity and every sibling subsection would collapse onto
    # the parent's output file. Issue #71 documents the observable bug; using
    # the requested identifier keeps each subsection in its own file.
    #
    # When the benchmark uses a free-text ``source_id`` ("USDA SNAP FY2026
    # maximum asset limits") alongside a path-like ``corpus_citation_path``,
    # the citation alone can't be parsed as a path — fall through to
    # ``source_unit.requested`` so the artifact lands inside a rulespec
    # source-root directory instead of ``source/<slug>.yaml``.
    is_corpus_path = _looks_like_corpus_citation_path(citation)
    relative_output = _resolve_eval_output_path(
        citation,
        requested_source=source_unit.requested,
        fallback=citation_to_relative_rulespec_path,
    )
    target_ref_source = citation if is_corpus_path else source_unit.citation_path
    prompt = _build_eval_prompt(
        citation,
        mode,
        workspace,
        workspace.context_files,
        target_file_name=relative_output.name,
        target_ref_prefix=_canonical_target_ref_prefix(
            target_ref_source,
            relative_output,
            policy_repo_path=policy_path,
        ),
        include_tests=include_tests,
        runner_backend=runner.backend,
        policyengine_rule_hint=policyengine_rule_hint,
    )
    generation_prompt_sha256 = _sha256_text(prompt)
    output_file = Path(output_root) / runner.name / relative_output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    response, wrote_artifact, retry_count = _run_prompt_eval_with_empty_artifact_retry(
        runner=runner,
        workspace=workspace,
        prompt=prompt,
        output_file=output_file,
        source_text=source_text,
        target_file_name=relative_output.name,
        include_tests=include_tests,
        policyengine_rule_hint=policyengine_rule_hint,
    )
    if wrote_artifact:
        eval_root = Path(output_root) / runner.name
        _hydrate_eval_root(eval_root, workspace)

    trace_file = (
        Path(output_root) / "traces" / runner.name / f"{_slugify(citation)}.json"
    )
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text(json.dumps(response.trace or {}, indent=2, sort_keys=True))

    metrics = None
    if output_file.exists():
        metrics = _evaluate_generated_artifact_with_repairs(
            rulespec_file=output_file,
            policy_repo_root=policy_path,
            axiom_rules_path=runtime_axiom_rules_path,
            source_text=source_text,
            policyengine_rule_hint=policyengine_rule_hint,
            skip_reviewers=skip_reviewers,
        )
    validation_error = _eval_artifact_validation_error(metrics)

    tokens = response.tokens
    result = EvalResult(
        citation=citation,
        runner=runner.name,
        backend=runner.backend,
        model=runner.model,
        mode=mode,
        output_file=str(output_file),
        trace_file=str(trace_file),
        context_manifest_file=str(workspace.manifest_file),
        duration_ms=response.duration_ms,
        success=wrote_artifact and response.error is None and validation_error is None,
        error=response.error
        or (None if wrote_artifact else "No RuleSpec content returned")
        or validation_error,
        generation_prompt_sha256=generation_prompt_sha256,
        input_tokens=tokens.input_tokens if tokens else 0,
        output_tokens=tokens.output_tokens if tokens else 0,
        cache_read_tokens=tokens.cache_read_tokens if tokens else 0,
        cache_creation_tokens=tokens.cache_creation_tokens if tokens else 0,
        reasoning_output_tokens=tokens.reasoning_output_tokens if tokens else 0,
        estimated_cost_usd=response.estimated_cost_usd,
        actual_cost_usd=response.actual_cost_usd,
        retrieved_files=[item.source_path for item in workspace.context_files],
        unexpected_accesses=response.unexpected_accesses,
        metrics=metrics,
        retry_count=retry_count,
    )
    emit_eval_result(result, response.trace)
    return result


def _eval_artifact_validation_error(metrics: EvalArtifactMetrics | None) -> str | None:
    if metrics is None:
        return None
    if not metrics.compile_pass:
        return "Generated RuleSpec failed compile validation"
    if not metrics.ci_pass:
        return "Generated RuleSpec failed CI validation"
    return None


def _run_single_source_eval(
    source_id: str,
    source_text: str,
    runner: EvalRunnerSpec,
    output_root: Path,
    policy_path: Path,
    source_metadata_payload: dict[str, object] | None,
    runtime_axiom_rules_path: Path,
    mode: EvalMode,
    extra_context_paths: list[Path],
    oracle: EvalOracleMode,
    policyengine_country: str,
    policyengine_rule_hint: str | None,
    skip_reviewers: bool = False,
) -> EvalResult:
    """Run one eval on a corpus-backed source unit rather than a USC citation."""
    workspace = prepare_eval_workspace(
        citation=source_id,
        runner=runner,
        output_root=output_root,
        source_text=source_text,
        axiom_rules_path=policy_path,
        mode=mode,
        source_metadata_payload=source_metadata_payload,
        extra_context_paths=extra_context_paths,
    )

    # Source-kind benchmarks often pass a human-readable source_id
    # ("USDA SNAP FY2026 maximum asset limits") alongside a path-like
    # corpus_citation_path in source_metadata_payload. Without consulting
    # the requested target, free-text source_ids land the artifact at
    # ``source/<slug>.yaml`` — outside the rulespec source-root, so the
    # compile validator can't find it. `_resolve_eval_output_path` falls
    # through to the metadata's `requested_source` when the source_id
    # isn't itself path-like.
    requested_source = None
    if isinstance(source_metadata_payload, dict):
        candidate = source_metadata_payload.get("requested_source")
        if isinstance(candidate, str) and candidate:
            requested_source = candidate
    relative_output = _resolve_eval_output_path(
        source_id, requested_source=requested_source
    )
    target_ref_source = _resolve_eval_reference_source_id(
        source_id, requested_source=requested_source
    )
    prompt = _build_eval_prompt(
        source_id,
        mode,
        workspace,
        workspace.context_files,
        target_file_name=relative_output.name,
        target_ref_prefix=_canonical_target_ref_prefix(
            target_ref_source,
            relative_output,
            policy_repo_path=policy_path,
        ),
        include_tests=True,
        runner_backend=runner.backend,
        policyengine_rule_hint=policyengine_rule_hint,
    )
    generation_prompt_sha256 = _sha256_text(prompt)
    output_file = Path(output_root) / runner.name / relative_output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    response, wrote_artifact, retry_count = _run_prompt_eval_with_empty_artifact_retry(
        runner=runner,
        workspace=workspace,
        prompt=prompt,
        output_file=output_file,
        source_text=source_text,
        target_file_name=relative_output.name,
        include_tests=True,
        policyengine_rule_hint=policyengine_rule_hint,
    )
    if wrote_artifact:
        eval_root = Path(output_root) / runner.name
        _hydrate_eval_root(eval_root, workspace)

    trace_file = (
        Path(output_root) / "traces" / runner.name / f"{_slugify(source_id)}.json"
    )
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text(json.dumps(response.trace or {}, indent=2, sort_keys=True))

    metrics = None
    if output_file.exists():
        metrics = _evaluate_generated_artifact_with_repairs(
            rulespec_file=output_file,
            policy_repo_root=policy_path,
            axiom_rules_path=runtime_axiom_rules_path,
            source_text=source_text,
            oracle=oracle,
            policyengine_country=policyengine_country,
            policyengine_rule_hint=policyengine_rule_hint,
            skip_reviewers=skip_reviewers,
        )
    validation_error = _eval_artifact_validation_error(metrics)

    tokens = response.tokens
    result = EvalResult(
        citation=source_id,
        runner=runner.name,
        backend=runner.backend,
        model=runner.model,
        mode=mode,
        output_file=str(output_file),
        trace_file=str(trace_file),
        context_manifest_file=str(workspace.manifest_file),
        duration_ms=response.duration_ms,
        success=wrote_artifact and response.error is None and validation_error is None,
        error=response.error
        or (None if wrote_artifact else "No RuleSpec content returned")
        or validation_error,
        generation_prompt_sha256=generation_prompt_sha256,
        input_tokens=tokens.input_tokens if tokens else 0,
        output_tokens=tokens.output_tokens if tokens else 0,
        cache_read_tokens=tokens.cache_read_tokens if tokens else 0,
        cache_creation_tokens=tokens.cache_creation_tokens if tokens else 0,
        reasoning_output_tokens=tokens.reasoning_output_tokens if tokens else 0,
        estimated_cost_usd=response.estimated_cost_usd,
        actual_cost_usd=response.actual_cost_usd,
        retrieved_files=[item.source_path for item in workspace.context_files],
        unexpected_accesses=response.unexpected_accesses,
        metrics=metrics,
        retry_count=retry_count,
    )
    emit_eval_result(result, response.trace)
    return result


def _resolve_eval_output_path(
    citation: str,
    *,
    requested_source: str | None = None,
    fallback: Callable[[str], Path] | None = None,
) -> Path:
    """Resolve the rulespec-relative output path for an encode target.

    Strategy:
      1. If ``citation`` already looks like a corpus citation path, use it
         directly (preserves subsection identity per issue #71).
      2. Otherwise, if ``requested_source`` is supplied AND looks like a
         corpus citation path, use that. Handles benchmarks that supply
         a free-text ``source_id`` ("USDA SNAP FY2026 maximum asset
         limits") alongside a path-like ``corpus_citation_path`` —
         without this branch, free-text source_ids land at
         ``source/<slug>.yaml`` outside any rulespec source root and
         the compile validator can't find them.
      3. Otherwise, defer to ``fallback`` (the caller decides whether
         the existing site's USC-citation parser or the path-identifier
         helper is appropriate). Defaults to the path-identifier helper,
         which preserves prior behaviour for source-kind eval call
         sites that always rendered free-text source_ids to
         ``source/<slug>.yaml``.
    """
    source_identifier = _resolve_eval_reference_source_id(
        citation,
        requested_source=requested_source,
    )
    if source_identifier != citation or _looks_like_corpus_citation_path(citation):
        return _source_identifier_to_relative_rulespec_path(source_identifier)
    try:
        return _source_identifier_to_relative_rulespec_path(
            _citation_to_corpus_citation_path(citation)
        )
    except ValueError:
        pass
    if fallback is not None:
        return fallback(citation)
    return _source_identifier_to_relative_rulespec_path(citation)


def _prompt_corpus_citation_path(source_unit: CorpusSourceUnit) -> str:
    """Return the citation path the prompt should ask RuleSpec to verify.

    `CorpusSourceUnit.citation_path` intentionally records the row that supplied
    text. For child-fragment requests resolved by slicing a parent row, prompts
    and validators should use the requested child path so source coverage is
    scoped to the encoded subtree instead of the whole parent section.
    """

    resolved = source_unit.citation_path.strip().strip("/")
    requested = source_unit.requested.strip().strip("/")
    if not resolved or not requested:
        return resolved
    try:
        primary_requested = _candidate_corpus_citation_paths(requested)[0]
    except (IndexError, ValueError):
        return resolved
    primary_requested = primary_requested.strip().strip("/")
    if (
        primary_requested
        and primary_requested != resolved
        and primary_requested.startswith(f"{resolved}/")
    ):
        return primary_requested
    return resolved


def _resolve_eval_reference_source_id(
    citation: str,
    *,
    requested_source: str | None = None,
) -> str:
    """Return the source identifier that should anchor output and test refs."""
    if _looks_like_corpus_citation_path(citation):
        return citation
    if requested_source and _looks_like_corpus_citation_path(requested_source):
        return requested_source
    return citation


def _source_identifier_to_relative_rulespec_path(source_id: str) -> Path:
    """Map an arbitrary source identifier to a stable eval artifact path.

    Each path segment is treated as a directory and a dot inside the leaf
    segment is treated as a further nesting separator (CDSS-style numbering
    like ``63-503.132`` becomes ``63-503/132``). This avoids the pathlib
    ``with_suffix`` pitfall where a dotted leaf would be silently truncated
    to a section-level path and collide with sibling subsections at apply
    time (issue #71).
    """
    parts = [part for part in source_id.strip().strip("/").split("/") if part]
    if parts and ":" in parts[0]:
        jurisdiction, source_root = parts[0].split(":", 1)
        if source_root in _RULESPEC_SOURCE_ROOT_TOKENS:
            tail = parts[1:]
            root = _RULESPEC_OUTPUT_ROOT_BY_SOURCE_TOKEN.get(source_root, source_root)
            if (
                tail
                and jurisdiction == "us"
                and source_root
                in {
                    "regulation",
                    "regulations",
                }
            ):
                tail = _canonical_us_regulation_tail(tail)
            if tail and jurisdiction == "uk":
                tail = _canonical_uk_legislation_tail(tail)
            if tail:
                if _preserve_state_statute_dotted_leaf(jurisdiction, root, tail):
                    return Path(root) / Path(*tail[:-1]) / f"{tail[-1]}.yaml"
                return Path(root) / _dotted_leaf_to_nested_yaml_path(tail)
    if len(parts) >= 2 and parts[0] in _RULESPEC_SOURCE_ROOT_TOKENS:
        tail = parts[1:]
        if tail:
            root = _RULESPEC_OUTPUT_ROOT_BY_SOURCE_TOKEN.get(parts[0], parts[0])
            if tail and tail[0] == _UK_LEGISLATION_DOMAIN_TOKEN:
                tail = _canonical_uk_legislation_tail(tail)
            return Path(root) / _dotted_leaf_to_nested_yaml_path(tail)
    if len(parts) >= 3:
        root = _RULESPEC_OUTPUT_ROOT_BY_SOURCE_TOKEN.get(parts[1])
        if root is not None:
            tail = parts[2:]
            if parts[0] == "us" and parts[1] in {"regulation", "regulations"}:
                tail = _canonical_us_regulation_tail(tail)
            if parts[0] == "uk":
                tail = _canonical_uk_legislation_tail(tail)
            if tail:
                if _preserve_state_statute_dotted_leaf(parts[0], root, tail):
                    return Path(root) / Path(*tail[:-1]) / f"{tail[-1]}.yaml"
                return Path(root) / _dotted_leaf_to_nested_yaml_path(tail)
    return Path("source") / f"{_slugify(source_id)}.yaml"


def _dotted_leaf_to_nested_yaml_path(tail: list[str]) -> Path:
    """Join tail segments, splitting the leaf on ``.`` into further nesting.

    ``["mpp", "63-503"]`` → ``mpp/63-503.yaml``
    ``["mpp", "63-503.132"]`` → ``mpp/63-503/132.yaml``
    ``["mpp", "63-503.131.a"]`` → ``mpp/63-503/131/a.yaml``
    ``["10-ccr-2506-1", "4.207.2"]`` → ``10-ccr-2506-1/4.207.2.yaml``
    """
    if _preserve_dotted_leaf_filename(tail):
        return Path(*tail[:-1]) / f"{tail[-1]}.yaml"
    leaf_segments = tail[-1].split(".")
    directory_parts = list(tail[:-1]) + leaf_segments[:-1]
    leaf_file = f"{leaf_segments[-1]}.yaml"
    if directory_parts:
        return Path(*directory_parts) / leaf_file
    return Path(leaf_file)


def _preserve_dotted_leaf_filename(tail: list[str]) -> bool:
    """Return true for source families whose dotted section is the file stem."""
    if len(tail) < 2:
        return False
    return bool(
        re.fullmatch(r"\d+-ccr-\d+-\d+", tail[0], flags=re.IGNORECASE)
        and re.fullmatch(r"\d+(?:\.\d+)+", tail[-1])
    )


def _preserve_state_statute_dotted_leaf(
    jurisdiction: str,
    root: str,
    tail: list[str],
) -> bool:
    """Return true when a dotted state statute segment is a legal citation label."""
    if jurisdiction != "us-co" or root != "statutes" or len(tail) < 2:
        return False
    if len(tail) == 2 and tail[0].isdigit():
        return bool(re.fullmatch(r"\d+(?:-\d+)+(?:\.\d+)+", tail[-1]))
    if not re.fullmatch(r"\d+(?:\.\d+)+", tail[-1]):
        return False
    return bool(re.fullmatch(r"\d+(?:-\d+)+(?:\.\d+)?", tail[-2]))


def _canonical_us_regulation_tail(tail: list[str]) -> list[str]:
    """Map federal regulation corpus paths to canonical RuleSpec repo paths."""
    if not tail:
        return tail
    title = tail[0].strip()
    if title.isdigit():
        return [f"{title}-cfr", *tail[1:]]
    return tail


def _canonical_uk_legislation_tail(tail: list[str]) -> list[str]:
    """Map official legislation.gov.uk corpus paths to canonical UK RuleSpec paths."""
    normalized = list(tail)
    if normalized and normalized[0] == _UK_LEGISLATION_DOMAIN_TOKEN:
        normalized = normalized[1:]
    if (
        len(normalized) >= 2
        and normalized[-2].lower() in _UK_LEGISLATION_SECTION_TOKENS
    ):
        normalized = [*normalized[:-2], normalized[-1]]
    return normalized


def _canonical_target_ref_prefix(
    source_id: str,
    relative_path: Path,
    *,
    policy_repo_path: Path | None = None,
) -> str | None:
    parts = [part for part in source_id.strip().strip("/").split("/") if part]
    if len(parts) < 3:
        return None
    if relative_path.parts and relative_path.parts[0] == "source":
        return None
    jurisdiction = _canonical_target_ref_jurisdiction(
        parts, policy_repo_path=policy_repo_path
    )
    if not jurisdiction:
        return None
    return f"{jurisdiction}:{_relative_rulespec_path_to_import_target(relative_path)}"


def _canonical_target_ref_jurisdiction(
    source_id_parts: list[str],
    *,
    policy_repo_path: Path | None = None,
) -> str | None:
    """Return the jurisdiction prefix for generated test references."""
    first = source_id_parts[0].split(":", 1)[0]
    if ":" in source_id_parts[0]:
        return first
    if first in _RULESPEC_SOURCE_ROOT_TOKENS:
        return jurisdiction_prefix(policy_repo_path) if policy_repo_path else None
    return first


def _rulespec_test_path(path: Path) -> Path:
    """Return the companion RuleSpec test path for a RuleSpec file."""
    return path.with_name(f"{path.stem}.test.yaml")


def _build_rulespec_eval_prompt(
    citation: str,
    mode: EvalMode,
    workspace: EvalWorkspace,
    context_files: list[EvalContextFile],
    target_file_name: str,
    target_ref_prefix: str | None,
    include_tests: bool,
    runner_backend: str,
    policyengine_rule_hint: str | None,
) -> str:
    """Build the RuleSpec authoring prompt used by current evals."""
    source_text = workspace.source_text_file.read_text().strip()
    corpus_citation_path = _workspace_corpus_citation_path(workspace)
    backend_section = ""
    if runner_backend == "openai":
        backend_section = (
            "You do not have filesystem tool access in this eval; rely on the "
            "inline source and any inline context copies in this prompt.\n"
        )
    scaffold_dates = _collect_scaffold_dates(workspace, context_files)
    scaffold_dates_section = ""
    if scaffold_dates:
        scaffold_dates_section = f"""
Temporal scaffold dates visible in copied context:
{", ".join(f"`{item}`" for item in scaffold_dates)}
Prefer the earliest scaffold date that is relevant to the copied precedent when `./source.txt` lacks its own effective date.
"""

    source_metadata_section = ""
    if workspace.source_metadata is not None:
        source_metadata_section = f"""
Structured source metadata is available in `./source-metadata.json` and copied below.
If a metadata relation says this source `sets`, `amends`, `implements`, or `restates` a canonical target, record that legal/provenance edge as a separate `kind: source_relation` rule with `source_relation.type` and the absolute target path under `source_relation.target`. This is not an `amends` relationship unless the source itself amends another source.
For state option/source-slice metadata, do not add a top-level `imports:` entry to the absolute canonical target path such as `us:regulation/...#...` or `us:statutes/...#...` unless a copied context file actually provides that import target.
If the canonical target is an option/applies/uses-style slot such as `...#*_applies` or `...#*_uses_*`, encode the canonical boolean slot as a direct dated constant `true` or `false` when the source text itself sets that option.
Do not invent jurisdiction guards like `*_is_in_state` or `*_is_in_jurisdiction` unless `./source.txt` states them; for a jurisdiction-specific source slice, use only positive/continuity cases rather than a fabricated out-of-jurisdiction false case.
For a jurisdiction-specific setting slice, omit an inapplicable false test unless `./source.txt` itself states a narrower in-jurisdiction condition.

=== BEGIN SOURCE-METADATA.JSON ===
{json.dumps(workspace.source_metadata, indent=2, sort_keys=True)}
=== END SOURCE-METADATA.JSON ===
"""

    context_section = ""
    if context_files:
        listings = "\n".join(
            _format_context_file_listing(item) for item in context_files
        )
        inline_context = ""
        if runner_backend == "openai":
            inline_context = f"""

You do not have filesystem tool access in this eval, so the relevant context files are also copied inline below.
Inline context copies:
{_format_inline_context_snippets(workspace, context_files)}
"""
        definition_items = [
            item for item in context_files if item.kind == "definition_stub"
        ]
        canonical_items = [
            item for item in context_files if item.kind == "canonical_concept"
        ]
        resolved_guidance = ""
        if definition_items:
            labels = "\n".join(
                f"- {item.label or item.import_path}" for item in definition_items
            )
            resolved_guidance += f"""
Resolved definition files are available below.
{labels}
import that canonical definition instead of inventing a leaf-local helper. Do not replace that import with a local deferred stub.
Do not encode such local factual predicates as placeholder constants like `true` or `false`.
Do not encode such local factual predicates as `status: deferred`.
"""
        if canonical_items:
            labels = "\n".join(
                f"- {item.label or item.import_path}" for item in canonical_items
            )
            resolved_guidance += f"""
Resolved canonical concept files from this corpus are available below.
{labels}
import or re-export that exact canonical concept instead of duplicating it locally.
"""
        branch_child_naming_section = _format_branch_child_naming_guidance(
            context_files,
            target_file_name=target_file_name,
            target_ref_prefix=target_ref_prefix,
        )
        cited_context_imports_section = _format_cited_context_import_guidance(
            source_text,
            context_files,
        )
        excluded_child_context_section = _format_excluded_child_context_guidance(
            source_text,
            context_files,
        )
        unavailable_cited_context_section = _format_unavailable_cited_context_guidance(
            source_text, context_files
        )
        partial_extent_child_schema_section = (
            _format_partial_extent_child_schema_limit_guidance(
                source_text,
                context_files,
                target_ref_prefix=target_ref_prefix,
            )
        )
        parent_child_terminal_section = _format_parent_child_terminal_output_guidance(
            context_files,
            target_ref_prefix=target_ref_prefix,
        )
        child_exception_import_section = _format_child_exception_import_guidance(
            source_text,
            context_files,
            target_ref_prefix=target_ref_prefix,
        )
        cycle_prone_context_import_section = (
            _format_cycle_prone_context_import_guidance(
                context_files,
                target_ref_prefix=target_ref_prefix,
            )
        )
        existing_target_contract_section = _format_existing_target_contract_guidance(
            context_files
        )
        existing_target_invalid_input_section = (
            _format_existing_target_invalid_input_guidance(context_files)
        )
        existing_target_validation_section = (
            _format_existing_target_validation_guidance(context_files)
        )
        existing_target_valid_input_section = (
            _format_existing_target_valid_input_guidance(context_files)
        )
        context_section = f"""
Context mode: `{mode}`.
Context files are precedent and dependency context, not independent legal authority for new values:
{listings}
{inline_context}
{resolved_guidance}
{existing_target_contract_section}
{existing_target_invalid_input_section}
{existing_target_validation_section}
{existing_target_valid_input_section}
{branch_child_naming_section}
{cited_context_imports_section}
{excluded_child_context_section}
{unavailable_cited_context_section}
{partial_extent_child_schema_section}
{parent_child_terminal_section}
{child_exception_import_section}
{cycle_prone_context_import_section}
Import and context rules:
- Use the listed import target rather than the `./context/...` inspection path.
- do not wrap import targets in quotes.
- Every import path must point to a file that is actually copied into the workspace.
- Top-level `imports:` entries must be scalar strings, never map entries like
  `- target:` plus `symbols:`. Import a copied export as one exact string such
  as `<jurisdiction>:<repo-path>#<exported_symbol>`.
- If a copied context file already defines the exact symbol you need, import that exact symbol instead of inventing renamed locals that overlap with the copied file.
- Copied context listings include exported symbols as `import_target#name`; use
  those exact references in `imports:` and proof atoms when composing from context.
- Copied context listings include local input slots as `import_target#input.name`.
  When assigning inputs for an imported file, use only the input slots listed
  for that imported file or listed for another imported dependency that is
  actually needed by the compiled import graph. Never copy a `#input` key from
  a sibling context test merely because that sibling imports the same file.
- Never drop the jurisdiction prefix from copied context imports. If context
  lists `us:statutes/26/24/h#some_output`, the top-level import and any proof
  import target must use exactly `us:statutes/26/24/h#some_output`, not
  `statutes/26/24/h#some_output`.
- In formulas, reference imported exports by their bare local rule name after adding an `imports:` entry; never write an absolute `us:...#rule_name` reference inside a formula.
- Treat copied current target files as context, not as backward compatibility contracts. You may drop, rename, rebuild, or defer existing executable rules, tests, imports, and local factual inputs when the source text, schema, canonical imports, or validation guardrails require a cleaner encoding.
- Do not preserve legacy executable surfaces merely because downstream tests or oracle mappings used them. Source-faithful RuleSpec with canonical legal pointers is more important than compatibility with old local names.
- Never preserve, rename, or recreate a legacy local input if it conflicts with the current no-placeholder, no-bare-friendly-name, filing-status, temporal, import, or source-grounding rules. If an existing output cannot be represented faithfully without such a local input, defer that executable surface or leave it out of executable formulas.
- When source text cites a section or subsection and a copied context file for
  that citation is listed, import and use the listed exported symbol from that
  context instead of creating a local `section_...` or `subsection_...`
  placeholder.
- If this target is an aggregate parent provision and copied child-fragment files
  already encode subparagraphs, import those child outputs and compose them.
  Do not redefine the child parameters, helper rules, or copied executable
  outputs in the parent file.
- Do not manufacture a parent-level `Judgment` output whose formula is only a
  pass-through, conjunction, or disjunction of imported child `Judgment`
  predicates. If a child paragraph already exports the exact predicate and the
  parent source adds no distinct parent-level condition, keep that child text
  documentary in `module.summary`, import the child predicate only where a real
  parent formula consumes it, or defer the parent surface. Compose imported
  child Judgments only when the requested source itself states a new named
  parent condition or result that genuinely combines those child predicates with
  source-stated local conditions.
- If context contains a more specific child file under the current target path
  that exports the exact scalar needed by this source, such as a `/rate`,
  `/threshold`, `/amount`, `/cap`, or `/limit` file, treat that child file as
  the canonical home for the scalar. Import the exact child export and use it
  in the current formula; do not emit a duplicate local `parameter` with the
  same value or name in the parent/composition file.
- Before using any imported output in arithmetic, check the copied context
  export's `dtype:`. An imported `dtype: Judgment` is a predicate, not a scalar
  amount, rate, or base. Never multiply, add, subtract, divide, `min`, or `max`
  a Judgment import as if it were Money, Rate, Count, or another numeric value.
  If the current source states a numeric base such as wages, remuneration,
  payments, or amounts attributable to a category and the copied import only
  identifies whether an item is attributable to that category, encode the source-stated numeric base as a local amount fact, or as a relation-filtered
  aggregate only when a compatible relation and numeric amount field are
  present. If neither is available, defer the numeric output instead of using
  the Judgment import as a placeholder scalar.
- If a copied context file for this target or a same-program sibling contains a `kind: source_relation` record, preserve the legal/provenance edge unless `./source.txt` proves it wrong; executable formula changes are not a reason to drop source graph context.
- Do not fabricate same-instrument imports or `statutes/...#symbol` paths unless that exact `path#symbol` import target is listed.
- do not fabricate sibling-file imports for uncopied same-instrument provisions.
- When a copied chart or parameter file supplies values, keep `.test.yaml` inputs and expected outputs consistent with the rows visible in that imported file; do not guess contradictory expectations for those imported values.
- Do not invent degenerate placeholder rows like `number_of_children_in_assistance_unit: 0` plus `number_of_caretakers_in_assistance_unit: 0` unless that row is visible in the copied chart file.
- Do not assert an exact zero imported standard, grant, or threshold unless that exact imported row is visible in the copied chart file.
{scaffold_dates_section}
"""

    missing_cited_source_section = _format_missing_cited_source_guidance(
        citation,
        source_text,
        context_files,
    )

    canonical_concept_section = _format_canonical_concept_registry_guidance(
        source_text,
        workspace,
        context_files,
    )

    test_file_name = _rulespec_test_path(Path(target_file_name)).name
    policyengine_hint_is_rulespec_identifier = _is_rulespec_local_identifier(
        policyengine_rule_hint
    )
    if include_tests:
        oracle_rule = ""
        if policyengine_rule_hint and policyengine_hint_is_rulespec_identifier:
            oracle_rule = (
                "- Every non-empty test `output:` mapping must assert the "
                f"canonical RuleSpec output whose local name is "
                f"`{policyengine_rule_hint}`; use its full "
                "`jurisdiction:path#rule` id when the artifact has a legal "
                "pointer.\n"
            )
        elif policyengine_rule_hint:
            oracle_rule = (
                "- Because the PolicyEngine hint is not a valid local RuleSpec "
                "identifier, tests must assert the source-faithful RuleSpec "
                "output generated for this file rather than the dotted "
                f"PolicyEngine path `{policyengine_rule_hint}`.\n"
            )
        canonical_target_rule = ""
        if target_ref_prefix:
            canonical_target_rule = f"""
	- The canonical RuleSpec reference prefix for `{target_file_name}` is `{target_ref_prefix}`.
	- In tests for this file, use `{target_ref_prefix}#input.<fact>` for local factual inputs and `{target_ref_prefix}#<rule>` for outputs. Never use `{target_file_name}#...` keys.
	"""
        proration_test_guidance = _format_proration_test_guidance(source_text)
        output_rules = f"""
Return exactly this two-file bundle and nothing else:
=== FILE: {target_file_name} ===
<RuleSpec YAML>
=== FILE: {test_file_name} ===
<YAML list of test cases>

Test file rules:
- `{test_file_name}` must be a YAML list beginning with `- name:` entries.
- Use `period`, `input`, and `output` keys. Use concrete scalar values, not formula strings.
- Do not use bare year periods like `2024`; they are ambiguous across jurisdictions.
- For monthly outputs, use `period: YYYY-MM`.
- Supported mapping `period_kind` values are `tax_year` and `custom`; never use `period_kind: calendar_year`.
- For annual tax tests, use an explicit tax-year mapping such as `period: {{period_kind: tax_year, start: '2024-01-01', end: '2024-12-31'}}`.
- For non-tax annual periods, use `period: {{period_kind: custom, name: calendar_year, start: '2024-01-01', end: '2024-12-31'}}`.
- For `period: Day` outputs, use a custom day mapping such as `period: {{period_kind: custom, name: day, start: '2024-01-15', end: '2024-01-15'}}`; never use bare `YYYY-MM-DD` shorthand.
- If an external oracle uses different public scenario inputs from the legal
  RuleSpec facts needed under `input:`, keep the legal facts in `input:` and add
  `oracle_inputs.<oracle>` with equivalent oracle-native inputs. Do not put
  oracle-only scenario keys directly in `input:`.
- Emit 1-4 cases unless a source-driven coverage rule below requires more. If
  `module.status` is `deferred` or `entity_not_supported`, the test file may be
  empty.
- Never emit a concrete test case with `output: {{}}` or an empty `output` map.
  If no executable output can be asserted, leave the test file empty instead of
  adding placeholder cases.
- The test file must contain YAML only; do not put prose or markdown fences in it.
- Use factual predicates or quantities in `input:`, not the output variable being asserted.
- Never assign an imported module's computed `#rule_name` output in `input:`. If this file imports that rule, the compiled program computes it. To make an imported output true, false, or equal a value, mirror the imported file's companion test pattern by setting its underlying `#input.<fact>` and `#relation.<name>` keys.
- When a test is meant to exercise a threshold, cap, or boundary on an imported derived output, do not assume one upstream raw input equals that imported output. First compute the imported formula from the upstream inputs you set; if the upstream formula has deductions, rates, or offsets that make the exact boundary awkward, choose clearly below/above-boundary inputs instead of an exact boundary case.
- Never turn an imported derived rule into a fabricated `#input.<same_rule_name>` key. For example, use `<jurisdiction>:<repo-path>#imported_judgment: holds` or `not_holds`, not `<jurisdiction>:<repo-path>#input.imported_judgment`.
- Do not invent `#input` keys for imported files. Use only the bare fact names that the imported file's formulas actually reference, or mirror the imported file's companion `.test.yaml` input pattern when it is supplied in context. If that imported output is driven by an upstream structural relation, set the upstream `#relation.<name>` rows used by the companion test instead of creating a local input under the imported file.
- A `#relation.<name>` input value must be a YAML list of row mappings. Never use a scalar row such as `- true`. Bad: `<jurisdiction>:<repo-path>#relation.member_of_household: [- true]`. Good: `<jurisdiction>:<repo-path>#relation.member_of_household:` followed by `- <jurisdiction>:<repo-path>#input.member_has_required_status: true`.
- Put `#relation.<name>` test inputs under the test case's top-level `input:`, not inside `tables.<Entity>` rows. If a table-row entity output depends on a relation, write separate scalar cases with the row's scalar facts and relation list under `input:` instead of a row-ordered `tables` case.
- Each `.test.yaml` case may assert derived outputs for only one entity type. If a module defines outputs on multiple entities, create separate cases for each entity pair, such as `Person`/`TaxUnit`, `Person`/`Employer`, or `Employer`/`Payment`. For example: `Person` cases set person facts at the top level and assert person outputs; `TaxUnit` cases use relation rows to supply person facts and assert only tax-unit outputs. Do not assert relation-child outputs in the parent entity's case.
- Use `holds` and `not_holds` for actual `dtype: Judgment` rule keys in test inputs and outputs; do not use YAML booleans for Judgment rule values.
- Use YAML booleans `true` and `false` for local factual `#input.<fact>` keys referenced directly by formulas.
- Compute each expected `output:` value by evaluating the emitted RuleSpec
  formula against the case inputs step by step. Do not guess expected outputs
  from nearby statutory thresholds or caps. For formulas that combine a flat
  threshold with a percentage of excess income, include the threshold amount,
  the excess amount, and the percentage amount in the calculation reflected by
  the scalar expected output.
- For positive tests that expect a nonzero amount, `holds` Judgment, or other
  affirmative result from a formula with source-stated age, income, resource,
  duration, date, status, or other threshold gates, set every gate input on the
  qualifying side of the threshold. For example, if the formula requires
  `age >= age_threshold`, a case expecting the positive amount must set `age`
  at least to `age_threshold`; use a separate negative case for below-threshold
  inputs.
- In mixed-output test cases, do not assert an output's affirmative or nonzero
  result when any input in that same case intentionally falls on the
  nonqualifying side of that output's threshold gate. Split the case instead:
  one case for the blocked output and a separate all-gates-positive case for
  unrelated affirmative outputs.
- For proration, average, ratio, or percentage tests with a source-stated denominator, choose input amounts divisible by that denominator so expected outputs are exact decimals, not rounded approximations. For example, if the denominator is 365, use a base amount like 36500 so `36500 * 182 / 365 = 18200`; if an average divides by 6, use totals like 600 or 1800, not 700. Avoid exact equality boundaries for ratios or percentages; choose clearly below/above-boundary values so decimal precision cannot decide the test outcome.
- Every test case for a local derived formula must assign every local factual
  `#input.<fact>` referenced by that formula, including facts that are false in
  the case. Missing false inputs make the executable test invalid.
- For every encoded `except`, `unless`, or `notwithstanding` carve-out,
  including `subject to` carve-outs, include companion tests for the positive
  path and the carve-out path so exclusions and override conditions cannot be
  silently dropped.
- When a source says a subsection, paragraph, payment, credit, benefit,
  eligibility path, or other output "shall not apply" or "does not apply",
  the exported rule that says that target applies, is allowed, is included, or
  is eligible must negate the exception. Do not expose the exception only as a
  standalone helper while leaving the affected `*_applies`, eligibility,
  inclusion, exclusion, or amount output true under the exception.
- When that exception is encoded as a local derived helper, include a blocking
  companion test asserting both that helper as `holds` and the directly affected
  Judgment output as `not_holds`.
- For scoped exceptions, include a control case proving a non-excepted
  qualifying item is not reduced or blocked even when the exception amount or
  exception fact is positive/nonzero, plus a case where the same exception
  applies to the source-stated excepted category.
- Preserve anaphoric scope in source predicates. If the source says "such
  account", "such instrument", "through such account", "with respect to such
  payment", or similar same-object language, the predicate name and companion
  tests must keep that same-object relationship. Do not shorten it to broad
  activity for the person, broker, household, or entity generally.
- When a local formula has five or fewer independent source-stated boolean
  gates joined by `and`, include one all-gates-positive case and enough negative
  cases to toggle each gate at least once. Do not leave a source-stated gate
  untested just because another negative case toggles a different gate.
- If a formula negates multiple exception predicates, include a separate companion test for each predicate that sets that exception input true and expects the directly affected Judgment rule to be `not_holds`.
- For any negated exception predicate, include a paired positive case with the same output rule where only the exception input changes from `false` to `true`; do not combine the exception test with another branch change.
- Validation fails if a direct local `#input.*_exception_applies` or
  `#input.*_exception_*` predicate is negated by an exported Judgment rule
  without this paired positive/negative companion.
- Do not collapse a list of cited exceptions or cross-reference carve-outs into one aggregate fact such as `sections_..._do_not_preclude...`. Encode or import each cited exception separately, then combine them in a helper if useful.
- If context files import this target file or reference this target file's outputs, use that as a signal to repair the dependency graph, not as a requirement to preserve old names. Keep an old output only when it remains the cleanest source-faithful RuleSpec surface.
- Do not preserve existing factual input slots referenced by copied formulas or companion tests when a cleaner source-faithful encoding removes them. For names listed under invalid copied local inputs, do not preserve, rename, or recreate them.
- When this source text itself names the operative factual disqualification,
  exception, or eligibility condition, encode that named condition as a local
  factual input even if the sentence cites another section for definitions or
  compliance procedures. Do not defer the target output solely because the
  cited section is absent when the source gives enough facts to evaluate the
  branch as true or false.
- For repo-backed artifacts, every `input:` and `output:` key must be a canonical
  legal RuleSpec reference that resolves to an actual file and fragment; do not
  use bare friendly keys or absolute-looking placeholders.
- Do not add speculative future-period tests that rely on uprating or amendments not stated in `./source.txt`.
{proration_test_guidance.rstrip()}
{canonical_target_rule.rstrip()}
{oracle_rule.rstrip()}
"""
    else:
        output_rules = f"""
Return ONLY raw RuleSpec YAML for `{target_file_name}`. Do not include fences or explanation.
"""

    target_hint = ""
    if policyengine_rule_hint:
        hinted_output_reference = (
            f"`{policyengine_rule_hint}`"
            if policyengine_hint_is_rulespec_identifier
            else "the source-faithful oracle-facing output"
        )
        oracle_test_guidance = (
            "assert the canonical RuleSpec output whose local name is "
            f"`{policyengine_rule_hint}` in every non-empty `output:` mapping"
            if policyengine_hint_is_rulespec_identifier
            else "assert the generated source-faithful output in every non-empty `output:` mapping"
        )
        naming_guidance = (
            f"- Name the main derived rule `{policyengine_rule_hint}` unless the source clearly defines a different canonical concept."
            if policyengine_hint_is_rulespec_identifier
            else (
                "- Choose a source-faithful snake_case RuleSpec concept name "
                "for the main output; the provided PolicyEngine hint is not a "
                "valid local RuleSpec identifier."
            )
        )
        policyengine_context_exports_section = (
            _format_policyengine_hint_context_exports(
                context_files,
            )
        )
        target_hint = f"""
Preferred principal output:
- Treat the PolicyEngine hint as an oracle-semantic hint, not as a license to
  copy PolicyEngine internals into RuleSpec. Name the main derived rule
  `{policyengine_rule_hint}` only when that value is already a valid local
  RuleSpec identifier. If the hint is a dotted PolicyEngine parameter path,
  slash path, or other non-RuleSpec identifier, choose a source-faithful
  snake_case RuleSpec concept name and keep the hinted PolicyEngine path only
  as oracle/comparison context. Never emit a RuleSpec rule whose `name:` is a
  dotted path such as `gov.states...`.
{naming_guidance}
- Treat the hinted policy surface as a required oracle-facing surface. Do not
  put the source-faithful output under `module.deferred_outputs[]` merely
  because the source is broad or cites many sibling provisions when an
  executable formula can be composed from source-stated facts, scalar
  parameters, or imported primary-source RuleSpec exports supplied in context.
- When the source slice contains multiple independent table columns or
  sibling figures but the PolicyEngine hint names one oracle-facing output,
  encode only the column, parameters, and boundary facts needed to compute
  that hinted output. Do not create executable rules for unrelated sibling
  columns, recoupment schedules, deemed-income amounts, administrative counts,
  or documentary table fields unless the formula for the hinted output depends
  on them or the requested source identifier explicitly names that sibling
  surface.
- If the hinted output would otherwise depend on broad placeholders such as
  `person_is_described_in_*`, `person_is_in_*_category`, `*_determined_for_*`,
  `*_mandatory_subclauses*`, or `person_is_qualified_*_group`, first look for
  the primary statute/regulation export in copied context and import that
  concrete output instead of leaving the broad phrase as a local boundary
  input. Defer only the specific unavailable branch, not the whole hinted
  output, when the remaining branches are executable and source-grounded.
- For an aggregate/composite hinted output, first enumerate the executable
  `dtype: Judgment` exports already visible in copied context that match the
  source's listed categories or branches. Compose {hinted_output_reference} as a
  disjunction/conjunction of those imported exports plus only the local
  conditions the current source itself newly states. Do not create broad local
  inputs such as `person_covered_by_*category`,
  `person_covered_by_*subparagraph`, `person_is_described_in_previous_*`,
  `person_is_not_described_in_or_enrolled_under_*`, or
  `income_determined_under_*` when a copied primary-source RuleSpec file
  exports the corresponding concrete category, income methodology, or branch.
- If a listed branch has no executable primary-source export in copied context,
  split that branch out: either encode the upstream source first, or omit/defer
  only that branch with exact provenance. Do not let the oracle-facing hinted
  rule depend on an aggregate leaf input merely to make the formula executable.
{policyengine_context_exports_section.rstrip()}
- Keep oracle-comparable tests at that named semantic level; do not assert only helper parameters or documentary scalars.
- Keep `.test.yaml` inputs oracle-comparable: prefer the oracle's direct component facts over inverted household proxy inputs, preserve direct component surfaces when available, and {oracle_test_guidance}.
- When PolicyEngine can compare the output but cannot consume the source-level
  legal facts directly, add `oracle_inputs.policyengine` with equivalent
  PolicyEngine-native scenario inputs instead of weakening the RuleSpec `input:`
  coverage.
- Prefer a contemporary monthly `.test.yaml` period like `2022-01` or `2024-01` when the source is current-effective and lacks a better effective date; avoid pre-2015 historical periods that PolicyEngine US cannot evaluate.
- If that output has a durable `jurisdiction:path#rule` id, key the test by that id rather than the friendly local name.
- Key inputs by their resolving legal RuleSpec target too, e.g. `jurisdiction:path#input.fact`, `jurisdiction:path#relation.name`, or `jurisdiction:path#upstream_rule`.
- If a copied downstream output with the oracle hint's local name is available, assert that canonical copied output rather than replacing it with a helper-only local test.
"""

    guidance_parts = [render_uk_legislation_guidance()]
    if _source_identifier_requests_percentage_rate_boundary(citation, source_text):
        guidance_parts.append(render_rate_only_source_boundary_guidance())
    if _is_single_amount_table_slice(source_text):
        guidance_parts.append(render_single_amount_row_guidance())
    if not re.search(r"\b\d{4}-\d{2}-\d{2}\b", source_text) and not scaffold_dates:
        guidance_parts.append(render_date_silent_scaffold_guidance())
    additional_guidance = "\n".join(
        part.strip() for part in guidance_parts if part.strip()
    )

    inline_source = f"""
=== BEGIN SOURCE.TXT ===
{source_text}
=== END SOURCE.TXT ===
{_format_subparagraph_coverage_checklist(source_text, corpus_citation_path)}"""
    corpus_source_section = ""
    corpus_rulespec_requirement = ""
    if corpus_citation_path:
        corpus_source_section = f"""
- This source text was read from `corpus.provisions` at `{corpus_citation_path}`.
"""
        corpus_rulespec_requirement = f"""
- Include `module.source_verification.corpus_citation_path: {corpus_citation_path}` exactly.
"""

    return f"""You are participating in an encoding eval for {citation}.

Author the output in Axiom RuleSpec YAML.
Do not narrate your plan or describe what you will do before emitting the artifact.
The response must begin with the requested RuleSpec artifact, not with prose.
This is Axiom encoding work. Do not read, load, or apply PolicyEngine skills,
PolicyEngine workflow docs, or PolicyEngine implementation guidance. PolicyEngine
may be mentioned only as oracle/comparison data when explicitly supplied by this
prompt. Use only the inline source, copied context files, RuleSpec/Axiom
instructions in this prompt, and local Axiom/RuleSpec files.

Primary legal authority:
- `./source.txt` contains the complete source text for this target source unit.
- Treat that source text as the only source of legal truth for this artifact.
{corpus_source_section.rstrip()}
{inline_source}
{source_metadata_section}{context_section}{missing_cited_source_section}
{backend_section}
{canonical_concept_section}
RuleSpec requirements:
- The RuleSpec file must begin with `format: rulespec/v1`.
- Include `module.summary: |-` with a concise exact audit excerpt, not the full source text when the source is more than a short paragraph. Corpus-backed validation reads the authoritative source from `corpus.provisions`; use the summary only to orient reviewers to the encoded provisions.
- Do not emit `source_url`; RuleSpec source verification reads `corpus.provisions`, not raw PDFs or web pages.
{corpus_rulespec_requirement.rstrip()}
- If the corpus source path is below statute/regulation authority (for example
  a `policy`, `manual`, `guidance`, `form`, table, CMS summary, or state plan),
  do not treat it as adequate by default. First check statute/regulation
  authority when provided in context. If the lower source remains the correct
  source for the encoded value or rule, include
  `module.source_verification.upstream_source_check` with `status`
  (`checked_higher_authority`, `official_parameter_source`,
  `delegated_parameter_source`, or `no_higher_authority_found`),
  `checked_paths` listing at least one statute/regulation corpus path or
  RuleSpec target that was checked, and `rationale` explaining why the lower
  source is still used. If no higher-authority check is available, stop and
  emit a typed request `upstream_source_check_required` instead of encoding.
- When higher-authority context is supplied through copied JSONL, inventory, or
  ingest-run files, cite the embedded corpus `citation_path` values in
  `checked_paths` and proof atoms. Do not cite the copied `external/...`
  workspace filename as legal authority.
- Include `module.proof_validation.required: true` and add
  `metadata.proof.atoms` to every `parameter`, `derived`, and
  `derived_relation` rule. Each atom
  must point to the corpus source, an accepted claim, or an explicit imported
  RuleSpec export supporting that rule's formula/value.
- For source-backed proof atoms, `source.corpus_citation_path` is sufficient.
  Add `source.excerpt` only for numeric amounts, rates, dates, or necessary
  disambiguation; keep excerpts short and do not quote long definitions or
  institutional descriptions.
- For imported proof support, put `import:` at the proof atom top level
  (for example `kind: import` plus `import.target: us:statutes/...#symbol`);
  do not put imported RuleSpec targets under `source:`. Import proof atoms must
  include `import.target`, `import.output`, and `import.hash` with the listed
  context file `sha256:` hash. If no `sha256:` hash is listed for that import,
  do not emit an import proof atom. When the imported proof target is in the
  same RuleSpec file, use `hash: sha256:local`; never use `sha256:self`.
- Proof atom `kind` must be one of: `amount`, `condition`, `definition`,
  `default`, `effective_period`, `exception`, `formula`, `import`, `ordering`,
  `parameter`, `parameter_table`, `predicate`, `table_cell`, or `unit`.
- Use `rules:` as a list of rule objects. The filepath is the ID; do not add an `id:` field.
- Do not invent schema keys like `namespace:`, `parameter`, `variable`, or `rule:`.
- Rule kinds are `parameter`, `derived`, `derived_relation`, `data_relation`, or `source_relation`. Use `parameter` for named source scalars when it fits the local schema, `derived` for entity-scoped outputs, `derived_relation` when source text defines a filtered legal membership relation, `data_relation` for runtime predicates, and `source_relation` for non-executable legal/provenance edges. The numeric invariant is the named concept: source-stated amounts, rates, thresholds, caps, limits, and table/grid cells should be named so consuming formulas can reference names instead of embedding literals.
- This numeric invariant applies in mixed deferred or entity-unsupported provisions too. Defer the unsupported output under `module.deferred_outputs[]`, but keep any independent source-stated scalar legal values as `kind: parameter` rules rather than dropping them into prose.
- Use `kind: parameter` with `indexed_by` and versioned `values` for source-stated numeric tables/scales keyed by household size, family size, income band, age band, or another row key. Do not encode those cells as `match` arms or numeric literals inside a derived formula. For source tables with interval/range row labels such as "at least / but less than" bands, do not create one scalar parameter per row, bound, or cell with names like `*_row_0_upper_*`, `*_row_3_rate`, or `*_lower_bound_band_9`. Define a source-backed band selector as a `derived` rule, store each substantive output column as a `kind: parameter` with `indexed_by: <band_selector>` and versioned `values`, and have the exported outputs look up the indexed table. Indexed table keys must be integer band ids such as `0`, `1`, and `2`; do not use decimal row thresholds like `1.33`, `2.5`, or strings such as `2_5_to_less_than_3_0` as lookup keys. Store source-stated row bounds as private named bound concepts or private indexed table/grid bound columns, and have the selector reference those names while returning integer band ids. If interpolation or clamping needs the active row bounds before native interval-table support exists, store lower/upper bounds as private indexed parameter columns and reference those names in derived formulas; do not repeat bound literals outside the selector. Preserve source row identity: open lower or upper interval cells are real rows, not defaults and not dropped rows; omit only the open side of the predicate.
- Indexed parameter `values` keys must be integers. If a source table maps textual labels such as county names, program names, payment codes, provider classes, or other strings to an amount, rate, or numeric classification, do not put those strings under `values:` and do not set `indexed_by` to a text input. Instead encode a `derived` selector/result using string equality or `match` arms over the source-stated text labels, or encode source-stated boolean predicates for listed categories and combine them. Keep the text labels in proof excerpts/tests, not as parameter table keys. Do not use an `indexed_by` table with numeric keys merely to encode rows whose source keys are text labels; integer table keys are for real numeric bands or explicit source row numbers, not invented positions for code pairs like Federal Code/State OS Code.
- For long source-stated text-label lists, do not emit one giant `or` chain or one giant `match` with every label. Large nested formula trees can exceed the compiled artifact parser's recursion limit. If a category list has more than about 25 text labels, split it into private source-backed `dtype: Judgment` helper predicates for chunks of that category (for example `zone_3_county_group_1`, `zone_3_county_group_2`), keep each helper formula to at most 25 text comparisons, and make the exported selector combine the helpers with a short conditional.
- Do not treat the final interval row as open-ended unless the source row is actually open-ended. If the last source row has an upper bound, the selector must return an out-of-table sentinel above that bound and the principal output must handle that sentinel. Include a companion test above the final bounded row so the generated artifact cannot silently extend the table.
- The out-of-table sentinel is not itself a source table row. Do not add sentinel entries to indexed parameter tables and do not clamp sentinel cases to the final table row's values. Handle the sentinel before table lookups, using the existing target's source-grounded out-of-range branch when repairing an existing artifact. Use a negative sentinel such as `-1`; do not use the next positive band id such as `6` merely because there are six legal rows, because that positive id is not source-stated and will fail numeric grounding.
- Do not hard-code the final real band id in non-selector formulas merely to make the final row constant. If the final row's initial and final table values are the same, let the indexed interpolation formula produce that constant value; branch only on the out-of-table sentinel and on genuinely distinct source-stated first-row behavior.
- For percentage interval row labels, bounds, rates, and ratio inputs, encode percent values as decimal ratios. For example, source text `133%` should be represented as `1.33`, and `60%` as `0.60`, not as percent-point values like `133` or `60`. When repairing an existing artifact, update companion tests to the same ratio scale instead of preserving old percent-point test inputs.
- For interval-table repair of an existing target, keep the executable surface narrow: add indexed bound columns and update the existing source-faithful principal formula, but do not add extra exported derived rules that merely project table columns such as `initial_*` or `final_*` unless the source text makes those projections legal outputs in their own right. Reference indexed table columns directly from the principal formula when they are only helpers for interpolation.
- Structural interval bounds that are only used by the selector should still be private implementation concepts, not public outputs. Prefer indexed bound columns or narrowly named private bound concepts over embedded selector literals; table/grid key indexes may remain as structural literals.
- For source-stated rate or percentage tables whose column header names a legal
  application such as "applicable percentage for section 3201(b)" or
  "applicable percentage for sections 3211(b) and 3221(b)", name the exported
  output after that statutory application. Do not append a consumer entity
  suffix like `_for_tax_unit`, `_for_person`, or `_for_employer` unless the
  source header itself states that entity.
- A `kind: table_cell` proof atom must include `source.table.header`, `source.table.row`, and `source.table.column`. A `kind: parameter_table` proof atom with `source.table` must include `source.table.header`, `source.table.row_key`, and `source.table.column_key`; header-only `parameter_table` proof atoms are invalid. Example: `source: {{table: {{header: "credit percentage table", row_key: "qualifying_child_count", column_key: "credit_percentage"}}}}`. If you cannot identify table coordinates, use a direct proof kind such as `amount`, `parameter`, or `formula` instead of `table_cell` or `parameter_table`.
- Every executable `parameter`, `derived`, and `derived_relation` rule must include a `source:`
  field with the legal citation/span that directly supports that rule. Keep
  `source:` short and local to the rule; use `module.source_verification` for
  the corpus locator.
- Use `kind: derived_relation` only when the source text explicitly defines
  membership in a derived legal unit by filtering a source relation through a
  stated predicate. "This source is about SNAP" is not enough. If the source
  uses an existing structural entity such as `Household`, `TaxUnit`, `Employer`,
  or `Person`, and merely references a program-specific concept without defining
  who belongs to it, stay on the source-stated structural entity.
- For source text that imposes an amount, tax, credit, or limitation on each,
  every, or any employer, use `entity: Employer`. Do not default to `TaxUnit`
  merely because the output is tax-related.
- Keep the membership predicate as an ordinary source-backed rule, then define
  the filtered entity under `derived_relation:` with `arity`, `source_relation`,
  `entity`, `member_relation`, `slot_entities`, and a `versions[].formula` that
  names the predicate.
- Any rule that uses `entity: <filtered-entity>` such as `SnapUnit`, a MAGI
  household, or a qualifying-child set requires the same file to either declare
  that entity with a `kind: derived_relation` rule or import a RuleSpec file
  that declares it. Filtered entities have no structural existence without that
  dependency.
{SOURCE_SCOPE_PROTOCOL}
- If `./source.txt` is a broad application, furnishing, administrative duty, or purpose clause without a computable policy condition, preserve it in `module.summary` but do not create an executable derived output just to paraphrase it. Encode only the concrete conditions, exceptions, parameters, and relations that affect computation.
- Do not create an output for administrative clauses like "assistance shall be furnished to all eligible households who make application." Unless the source defines a calculable benefit, amount, condition, or exception, keep that text documentary in `module.summary`.
- Do not encode a pure pass-through rule whose formula is only one local fact. If the source only names a preexisting fact without changing it, reference the upstream rule when available or leave the phrase documentary.
- If a copied child-fragment file encodes a limitation, branch, amount, or
  predicate needed by the requested parent provision, import the child output
  and compose it. Do not copy the child formula or its factual inputs into the
  parent file.
- Do not create standalone small-number parameters just to restate prose such as "one-time" or "more than one consecutive month" when the number only qualifies a local factual condition. Encode the whole source-stated condition as a fact predicate or derived condition unless the scalar is an independent reusable amount, rate, threshold, cap, or limit.
- Do not append citation or file suffixes like `_2014_a` to new local rule names; the file path is already the legal ID. Keep names concise and semantic unless a copied public interface must be preserved.
- Rule names ending in the current path fragments, such as `_2_C`, `_b_1`,
  `_d_2_C`, or `_2014_a`, are invalid.
- If an existing copied output name violates the no-citation/path-suffix rule,
  do not preserve it. Rename it to a concise semantic name and update the
  companion tests.
- Rule names must not collide with copied sibling files. For subparagraph/list
  item child files, make the principal output name semantic to that branch
  (for example `care_responsibility_exemption_applies`), not only the shared
  parent consequence like `person_exempt_from_paragraph_1_work_requirements`.
- If a source provision is headed "Definition of X", a successful executable
  artifact must expose the final source-backed output `x` (normalized to the
  local naming style, such as `surviving_spouse` or `head_of_household`). Helper,
  limitation, and prerequisite predicates may support that output, but they do
  not replace it. If X cannot be computed faithfully because upstream legal
  definitions are missing, mark the module deferred with no executable rules
  rather than applying a helper-only definition. This overrides the usual
  mixed-provision instruction to keep independent helpers: a `Definition of X`
  artifact without executable X is misleading and must be cleanly deferred.
- If a definition applies an exclusion, cap, threshold, exception, or special
  rule only "for purposes of", "in the case of", "with respect to", or "for the
  tax imposed by" a specific downstream provision, do not collapse that
  purpose-specific branch into one generic output for all downstream uses. Emit
  source-backed purpose-specific outputs such as `x_for_section_1234_a` and keep
  any broader `x` output free of that limited branch when the source supports a
  broader meaning. Downstream provisions must import the output that matches
  their cited purpose rather than using a stale local input or an over-broad
  generic output.
- If one purpose-specific exception, rate portion, or base branch is not
  executable yet, do not export a generic `x_after_cap`, `x_included`,
  `x_excluded`, `taxable_x`, or similar output that silently applies the
  non-excepted branch to every downstream purpose. Either split the executable
  surface into concrete purpose-scoped outputs and defer only the unresolved
  purpose, or defer the generic surface entirely.
- Do not use boundary inputs named like `applicable_base_for_current_purpose`,
  `amount_for_current_context`, or `rate_under_current_use`. Those hide
  purpose-specific legal mechanics from downstream importers. Use the concrete
  source-stated purpose in the rule name and formula input, such as
  `applicable_base_for_section_3201_a_non_hospital_insurance_rate_portion`, or
  defer that purpose-specific surface.
- When a child provision substitutes, increases, caps, or otherwise modifies a
  sibling or parent output, give the replacement a branch-specific name such as
  `_under_subsection_h`, `_after_temporary_amendment`, or another source-stated
  modifier. Do not reuse sibling output names when the requested branch changes
  the meaning.
- Choose structural relations at the narrow legal subject stated by the source.
  If the source grants an amount to the taxpayer, spouse, claimant, child, or
  other role-limited person, do not aggregate over a broader household/tax-unit
  relation unless the source says every member counts. Name the relation for the
  role set that is legally counted, such as `taxpayer_or_spouse`, not merely for
  the container entity. If a copied relation is legally too broad for the
  requested source, rename it; relation names are not stable public outputs.
  Never preserve or create `*_member_of_tax_unit` or `member_of_tax_unit` for a
  source that counts only the taxpayer, spouse, qualified individual, claimant,
  child, or dependent.
- For any source that says "qualifying child", "dependent of the taxpayer", or
  "with respect to such child", do not use `member_of_tax_unit`. Define a
  role-scoped relation such as `dependent_of_tax_unit`,
  `qualifying_child_of_tax_unit`, or `child_or_dependent_of_tax_unit`, and
  aggregate over that relation.
- If a generic role-scoped relation name is already exported by a copied
  sibling file, do not reuse it. Make the relation source-specific instead of
  reusing the sibling's generic relation name.
- If the source computes an amount by reference to an entitlement, status,
  amount, or test "under" another section, subsection, paragraph, regulation, or
  document, do not inline that cross-reference's mechanics into this file unless
  that cross-referenced source text is included and this file is the canonical
  home for those mechanics. Import the existing RuleSpec target when present. If
  the cross-reference is not yet encoded, expose a semantic input/count named
  for the cross-reference itself, such as
  `additional_standard_deduction_entitlement_count_under_subsection_f`, rather
  than inventing the cross-referenced age, blindness, household, or membership
  tests locally.
- When a state, local, or downstream tax source consumes a completed federal
  return amount, such as a deduction, credit, federal adjusted gross income,
  federal taxable income, or itemized deductions already claimed on the federal
  return, keep the current source executable from a neutral federal-return
  amount input if no same-period imported RuleSpec output directly exports that
  completed return line. Name that input for the completed return amount, not
  for the legal citation pointer: for example use a name like
  `federal_return_deduction_amount` or
  `itemized_deductions_claimed_on_federal_return`, not
  `section_<section>_*`, `*_under_section_<section>`, or
  `*_allowed_under_section_<section>`. This rule applies only when the current
  source merely adds, subtracts, caps, gates, or otherwise consumes the
  completed upstream return amount; do not use it to restate upstream mechanics
  that the current source does not provide.
- When an unencoded cross-reference must be represented as a semantic local
  input, name it after the legal status with an `_under_section_<section>` or
  `_under_subsection_<subsection>` suffix. Do not start a local input with
  `section_<section>_` or `subsection_<subsection>_`; those names are reserved
  for imported legal outputs and will be treated as missing imports.
  Cross-reference local inputs such as `_under_section_<section>`,
  `_provided_in_section_<section>`, `_allowed_under_section_<section>`,
  `_deduction_under_section_<section>`, or `_credit_allowed_under_section_<section>`
  are only allowed for non-exception factual interfaces when the cited source is
  not available as RuleSpec. If the citation appears in definition,
  same-meaning, treated-as, rules-similar, exception, exclusion, `unless`,
  `notwithstanding`, shall-not-apply, or not-treated-as logic and the cited
  source is unavailable, do not invent a local cross-reference fact for the
  cited mechanics. If the requested source itself states the operative effect
  and only uses the citation to label a category, encode a source-named boundary
  predicate for that category instead of deferring. This includes `within the
  meaning of section ...` carve-outs and `described in section ...` category
  labels where the current source states that the category is included,
  excluded, or not treated in a specified way. Otherwise, if the dependency is
  essential to the only requested executable concept, emit
  `module.status: deferred` or `module.status: entity_not_supported` with
  `rules: []`. In a mixed provision, omit or defer only the affected executable
  surface and still encode independent source-backed outputs that do not require
  the unavailable dependency. For each omitted/deferred executable output in a
  mixed provision, add `module.deferred_outputs[]` with absolute RuleSpec
  targets for `output`, a plain-language `reason`, and `source_values` entries
  for any source-stated local parameters retained only for that deferred output.
  If those scalar legal values are independently encoded as `kind: parameter`
  rules in the same file, do not also demote them to prose or rely on
  `source_values` instead.
  Treat category membership phrases such as `person described in section X`,
  `organization described in section X`, or `service described in section X` as
  factual boundary predicates when the current source states the legal effect.
  Use a source-named predicate like `organization_described_in_section_509_a_3`
  plus any conditions stated in the current source. Do not defer merely because
  the cited section is unavailable unless the current source requires computing
  numeric amounts or legal mechanics from that section rather than testing
  membership in the described category.
  If a source says a rate, percentage, amount, applicable percentage, or
  similar numeric term is determined under, in effect under, or equal to rates
  from another section or subsection, do not model that numeric term as a local
  input such as `tier_1_applicable_percentage`. Import the upstream output when
  it exists; otherwise defer the affected output and name the cited legal
  dependency in `reason`.
  Before applying any imported rate to the current source's whole base, check
  whether the cited source makes that rate thresholded, capped, base-limited, or
  part of an amount formula that applies only above or below a specified amount.
  If so, do not flatten the cited mechanics into `current_base * imported_rate`
  or into a combined percentage that is later multiplied by the whole current
  base. Import and compose the cited executable amount or the cited base,
  threshold, cap, and excess-amount outputs faithfully. If the current schema
  cannot pass the correct adjusted base into those cited mechanics, defer the
  affected executable output and name the missing cited computation rather than
  approximating it with a flat rate.
  When the omitted output covers a specific subsection or subparagraph, the
  `output` target path must include that source path segment, e.g.
  `us:statutes/26/3201/a#tier_1_employee_tax`, not
  `us:statutes/26/3201#tier_1_employee_tax`.
  Only include `blocked_by` entries when you know the exact RuleSpec output with
  a `#rule_fragment`. Do not list bare legal provisions, corpus paths, statute
  sections, or guessed pseudo-targets in `blocked_by`; for example,
  `us:statutes/us-ca/17000` is invalid. If the exact upstream RuleSpec output is
  unknown, omit `blocked_by` and name the legal dependency in `reason`. Do not create
  tests for deferred outputs. If a source-grounded overriding rule makes the
  unavailable branch zero or unreachable for the encoded effective period,
  encode that overriding branch instead of deferring the whole module. If that
  section is present in repo context, import it and use its exported output
  instead.
- An import only resolves a cross-reference when the imported file exports the
  actual referenced legal concept, amount, status, test, or definition needed by
  the formula. Importing an adjacent upstream output only as proof, while the
  formula still depends on a local `_under_section_...` or
  `_in_effect_under_section_...` fact, does not satisfy the dependency. If the
  listed context file for a cited source does not export the needed concept, do
  not import an unrelated output from that file as a stand-in; encode the proper
  upstream source slice first, split the unresolved branch, or emit a deferred
  status when the requested file cannot compute faithfully without it.
- When the requested source imposes a rate, tax, deduction, credit, cap, or
  threshold on a legal term that is defined by an available upstream RuleSpec
  file, import that upstream definition and use it in the formula. Do not leave
  a same-named local input such as `x` merely because a copied target file used
  `#input.x`. If the upstream definition has purpose-specific exports, select
  the export matching the requested source's clause.
- Every proof import must correspond to a symbol actually used by that rule's
  formula. Do not add an import atom merely because the source text mentions an
  exception or cross-reference that the formula excludes, subtracts around, or
  otherwise handles without the imported output.
- A cited context file with `module.status: entity_not_supported`,
  `module.status: deferred`, or `rules: []` is not an executable dependency.
  Do not preserve, rename, or recreate a local cross-reference input for that
  cited source. If the current provision cannot compute faithfully without that
  cited source, defer the affected executable surface; if a source-grounded
  overriding rule makes the cited branch unreachable for the encoded effective
  period, encode only that overriding branch and leave the unresolved branch out
  of executable formulas.
- Never introduce an import cycle. If a cited source directly or transitively
  imports the current target module, do not import that source back into the
  same module; keep a source-named boundary predicate or numeric boundary input
  for that cyclic condition until the sources are split into acyclic subsection
  modules. This applies to cross-referenced rates or parameters as well as
  eligibility predicates: if importing a rate-bearing source would complete a
  cycle with a foundational base definition, keep the rate as a source-named
  boundary input and continue encoding the non-cyclic base formula.
- Never create a derived rule whose formula references that same rule's name.
  The derived rule name must be the legal conclusion or compliance output, while
  required facts inside the formula must use distinct source-named local inputs.
  For example, do not define `x_has_bona_fide_need` as
  `x_has_bona_fide_need and other_conditions`; instead name the derived output
  `x_arrangement_valid` and reference a separate factual input such as
  `bona_fide_need_for_x_arrangement`.
- When the requested source defines a base, net amount, includable amount, wage
  base, income base, deduction base, or similar amount that a tax, contribution,
  credit, or deduction section will consume, do not import that consumer section
  only to use its rate parameters. If the requested source merely cites the
  consumer section's rates for an adjustment to the base, keep that rate or rate
  sum as a source-named numeric boundary input unless a non-cyclic standalone
  rate table/source is available.
- When a copied context file encodes a cited upstream source on a different
  entity, import that upstream output and bridge entities with a structural
  relation instead of replacing the import with a local cross-reference amount.
  Do not replace a specific upstream output with a broad local input for all
  amounts described by that upstream source.
- If an upstream output is already executable, do not replace it with a local
  placeholder fact or compatibility alias.
- Do not encode simple unary factual inputs as `kind: data_relation` rules. If a formula needs a local true/false fact, reference a descriptive bare fact name in the formula and put that fact in tests as `{target_ref_prefix + "#input.<fact>" if target_ref_prefix else "<jurisdiction>:<path>#input.<fact>"}`.
- Use `kind: data_relation` only for structural runtime predicates with explicit `data_relation.predicate`, `data_relation.arity`, and `data_relation.arguments`.
- If the requested source text includes a limitation, cap, exception, or
  cross-referenced subparagraph that changes the final exported amount, the
  final exported amount must apply that limitation. If a copied sibling/context
  file already encodes the limitation, import it and compose with it instead of
  duplicating or ignoring it.
- When an exception or carve-out applies only to a source-stated category of an
  otherwise qualifying payment, person, household, expense, or amount, gate that
  exception with a predicate for the excepted category or use an input amount
  whose name is explicitly scoped to that category. Do not subtract, disallow,
  or reduce all qualifying branches merely because an exception amount input is
  nonzero.
- If the requested source itself enumerates qualifying or exception categories
  and cites other laws only to define those category labels, encode each
  source-stated category as its own boundary predicate and combine them into the
  final rule. Do not defer the final exported output solely because cited title,
  chapter, schedule, appointment, office, retirement-system, election,
  covered-service, section-described supporting organization,
  treated-as-trade-or-business, unrelated-trade-or-business, or other
  within-meaning/described-in definitions are not encoded when the requested
  source states the operative effect.
- If the copied target file is already executable, do not let its old surface
  force local placeholders or compatibility names. Rebuild, drop, or defer
  individual outputs as needed. Prefer retaining or replacing source-backed
  independent outputs that can be encoded without unresolved dependencies; use
  top-level `module.status: deferred` or `module.status: entity_not_supported`
  only when no executable rule in the requested source can be represented
  faithfully.
- If the requested source itself defines a legal status or test through
  relationship, age, abode/residence, support, filing, income, or tie-breaker
  conditions, encode those conditions as executable predicates with boundary
  inputs for facts not defined in the source. Do not emit
  `module.status: deferred` merely because some facts must be supplied by the
  caller or because tie-breaker facts require relation inputs.
- If the requested source defines an exclusion, inclusion, deduction, or credit
  amount but depends on externally determined classifications, official
  designations, statuses, event facts, or source-document categories, encode
  the amount with boundary inputs for those facts instead of deferring.
- If the source says an actor may not request something, is not entitled to
  something, or is otherwise categorically prohibited, do not create a local
  authorization escape-hatch input such as
  `*_has_source_authorized_*_entitlement` to make the positive entitlement
  hold. Encode the entitlement as a constant false Judgment or encode the
  source-stated prohibition/not-entitled rule directly.
- This includes exclusions conditioned on a reasonable belief that an item can
  be excluded from income under another section. Do not defer solely because
  the cited exclusion section is not encoded; model the source-stated
  reasonable-belief condition as a local factual predicate and gate the
  source-stated excluded amount with it.
- If the requested source itself states a cap, threshold, exclusion, or base
  formula that uses an externally determined official base, wage amount,
  compensation amount, rate, status, or special-case fact that is not available
  as copied RuleSpec context, keep the source-stated formula executable with
  semantic local boundary inputs named for those legal values or facts instead
  of deferring the whole output.
- If a missing special rule or unavailable cited definition affects only one
  subtype, carve-out, or branch, defer only that branch or expose a
  source-named boundary input for that branch. Do not defer an unrelated
  source-stated cap/base computation that can be executed from the source text.
- For cross-reference boundary facts that remain local because the cited source
  is not present in context at all, keep the legal pointer in the identifier.
  If context for the cited source is present but unsupported, deferred, empty,
  or missing the needed export, do not preserve, rename, or recreate the local
  cross-reference fact; import a real export, defer the affected executable
  surface, or encode a source-grounded overriding branch that avoids it.
- When the requested source states its own amount, cap, threshold, or formula
  but begins with an exception such as `except as otherwise provided in section
  X` or `except as otherwise provided in subsection X` and the cited external
  or parent source is not present in copied context at all, do not defer the
  requested formula merely because section X may have unresolved effective
  versions, ballot triggers, repeal facts, or program conditions. Encode the
  requested source's own formula and represent the absent cross-reference
  boundary with a source-named predicate such as
  `subsection_x_does_not_displace_this_subsection`. Include a companion case
  where the cross-reference boundary blocks the local output. If copied context
  for the cited source is present but lacks the exact displacement predicate,
  follow the copied-context rule above instead. This applies to cited external
  or parent sources, not to uncopied sibling clauses; for sibling clause
  exception phrases, follow the sibling-clause rule below and do not invent
  local `clause_*` booleans.
- When that output also composes an imported child or sibling result, check
  that the imported file does not defer another branch, period, or purpose that
  can affect the same final amount. Do not treat a missing deferred child branch
  as zero by importing only the available branch result. Either scope the
  executable output to the branch where the deferred child branch is impossible,
  or defer the composite output and list the child deferred dependency.
- Importing a child rate or threshold is not enough when the child file already
  exports the executable tax, benefit, deduction, or eligibility result. For
  aggregate parent sections, import the child result output itself and sum,
  cap, select, or otherwise compose those imported results. Do not recompute a
  child result locally from the child rate and the child factual inputs.
- Do not create parallel statutory-dollar executable parameters when a copied
  current-year authority already provides the applicable inflation-adjusted
  parameter. Import the current-year authority unless the task is to encode the
  inflation adjustment formula itself.
- If a copied current-year authority exports the same concept or output name
  that the requested statute formula would otherwise create, do not emit a
  local executable duplicate with that name. Import and use the current-year
  authority's output, keeping only statute-specific conditions or non-executable
  `source_relation` records in the statute file.
- If a current-year authority provides a directly rounded final amount table,
  use that table for the final amount instead of recomputing the amount from
  related rates and thresholds.
- When source text says an exemption, exclusion, or adjustment applies
  `to the extent` of an amount, do not model it as all-or-nothing zeroing such as
  `if exempt_amount > 0: 0 else: tax`. Subtract or apportion the stated amount.
  Emit `module.status: entity_not_supported` or `deferred` only when the current
  requested source changes the basis that would need to be passed into an
  already-imported child result and the schema cannot wire that adjusted basis
  into the child. Do not defer a parent merely because an imported terminal
  child output internally handled its own `to the extent` exclusion; import and
  compose that terminal child output at the parent scope instead.
- If the source has a cross-reference such as `For application of different
  contribution bases ... see section X`, do not repair it by keeping tax,
  amount, or rate-times-compensation formulas on the raw wage, compensation,
  remuneration, or payment base. Import and compose the cited
  base/cap/exclusion/excess outputs; if those cited base mechanics are missing,
  purpose-specific, or deferred, add `module.deferred_outputs[]` for each
  affected source subsection output.
- Do not repair that case by importing child rates or thresholds and rebuilding
  the child branch locally with an adjusted basis. That still re-encodes the
  child branch and is invalid unless the schema can explicitly wire the
  adjusted basis into the imported child result.
- When the statute states pre-inflation base dollars that a current-year
  authority adjusts, any local statute output must be named as a statutory/base
  concept, not as the current-year value.
- When the source rounds an inflation or cost-of-living increase, round the
  increase before adding it to the base amount unless the source explicitly
  says to round the final total. Companion tests must assert the rounded
  increase plus the base, not the unrounded total. For example, with base
  15750, adjustment 0.1, and a next-lower $50 multiple, the increase is 1550
  and the total is 17300, not 17325.
- Do not invent arbitrary entities. Use existing standard entities when they
  match the source-stated legal subject. If the source states a legal subject
  outside the standard benefit/tax/person ontology, introduce a narrow singular
  PascalCase entity for that subject, such as `StateAgency` for State-agency
  SNAP administration or `SnapQualityControlFiscalYear` for a national SNAP QC
  fiscal-year aggregate. Do not use a generic `State`, row-index, or
  household/person/tax-unit entity for administrative aggregates.
- Standard `entity:` examples are {", ".join(f"`{entity}`" for entity in SUPPORTED_EVAL_ENTITIES)}.
- Allowed `period:` values are {", ".join(f"`{period}`" for period in SUPPORTED_EVAL_PERIODS)}.
- Allowed `dtype:` values are {", ".join(f"`{dtype}`" for dtype in SUPPORTED_EVAL_DTYPES)}, or `Enum[Name]`.
- Use `dtype: Judgment`, not `dtype: Boolean`, for legal eligibility, availability, applicability, entitlement, and other holds/not-holds style outputs, especially when the formula contains `not`.
- Do not use `kind: parameter`, `dtype: Boolean` for date-versioned applicability flags. Use a numeric 1/0 indicator parameter with explicit comparisons, or fold the date effect into the rate, amount, or threshold parameter that consumes it.
- Do not emit top-level `values` lookup tables outside `versions`. Source-backed indexed parameter tables may use `indexed_by` with versioned `values`; every executable derived output must use `versions[].formula`, and formulas must consume those tables with `table_name[index_expr]`, never bare `table_name`. Small non-reusable statutory bands can be expressed as explicit derived conditionals.
- Do not create derived `dtype: Boolean` helper rules with logical formulas. Use `dtype: Judgment` for derived legal predicates, or leave simple local facts as factual `{target_ref_prefix + "#input.<fact>" if target_ref_prefix else "<jurisdiction>:<path>#input.<fact>"}` keys consumed by formulas and tests.
- Use `unit: USD`, `unit: GBP`, or another explicit unit for money outputs when the source states a currency.
- Put each rule's formulas under `versions: - effective_from: 'YYYY-MM-DD'` and `formula: |-`.
- Do not encode legal effective dates as `dtype: String` parameters or date
  literal formulas such as `2025-01-01`. Axiom formulas have no date literal type.
  Use `effective_from` metadata for version timing, or use a
  source-stated semantic boolean predicate when a date window is a runtime
  condition. Do not put the date or year value in the fact name; use names like
  `taxable_year_begins_after_termination_date` or
  `taxable_year_is_in_temporary_effective_window`, not
  `taxable_year_begins_after_2024_and_before_2029` or
  `taxable_year_begins_after_december_31_2021`.
  Never use `post_YYYY`, `pre_YYYY`, `after_YYYY`, `before_YYYY`, or any
  four-digit year in a runtime date-window fact name.
  This overrides preservation of existing local input names: if a copied
  formula uses a date-valued fact name, rename that fact consistently to a
  semantic date-window predicate in formulas and tests.
- Do not emit more than one `versions:` entry for `kind: derived`; the runtime does not yet support period-selecting versioned formulas. Use a single source-faithful conditional formula when the provision itself defines a temporal branch, or encode only the currently applicable provision after resolving the source context. When a derived result changes only because a base rate, threshold, cap, or additive adjustment changes over time, put the dated changes on named `parameter` or helper rules and keep the consuming `kind: derived` rule to one formula, such as `base_rate + temporary_adjustment + later_adjustment`.
- Formula strings use Axiom formula syntax: `if condition: value else: other`, `==` for equality, `and`/`or` for booleans, decimal ratios for percentages, and no Python inline ternary syntax.
  Do not write `else if` or `elif`; chain branches as `if condition: value else: if next_condition: next_value else: fallback`.
- Function calls in formulas are expression syntax, not Python syntax. Do not
  include trailing commas in calls such as `min(a, b)` or `max(0, x)`, and do
  not write tuple-style expressions.
- Supported scalar functions are `min(...)`, `max(...)`, `floor(x)`, and `ceil(x)`. Do not use Python-only functions such as `round(...)`; express nearest-multiple rounding as `floor((x / multiple) + 0.5) * multiple` for nonnegative amounts.
- Benefit, allotment, credit, deduction, allowance, and subsidy formulas must never emit negative money. When subtracting an income, contribution, or other reduction from a maximum amount, floor the result with `max(0, ...)` before applying downstream minimum-benefit or issuance branches. When a nonnegative credit, deduction, allowance, subsidy, or benefit is a percentage of `min(income, cap)` or similar, floor the income base at zero: use `rate * min(max(0, earned_income), cap)`, not `rate * min(earned_income, cap)`.
- Outputs named `taxable_income` or ending in `_taxable_income` must also never be negative. Wrap the final selected branch at zero, including both sides of conditionals: use `if condition: max(0, branch_a) else: max(0, branch_b)`, not `if condition: branch_a else: branch_b`.
- Taxpayer elections such as electing to itemize deductions are legitimate
  election-state inputs to the legal computation. Do not mark a taxable-income
  rule unsupported merely because the taxpayer could optimize that election
  outside the core RuleSpec runtime; encode the statutory branches keyed by the
  election fact, and let oracle/comparison harnesses run multiple scenarios when
  they need optimization.
- If that reduction has rounding alternatives, every branch must be floored: use `if round_up: max(0, maximum - ceil(reduction)) else: max(0, floor(maximum - reduction))`, never `if round_up: maximum - ceil(reduction) else: floor(maximum - reduction)`.
- US tax filing status is a derived legal classification, not a downstream
  boundary fact. Do not create local `#input.filing_status` facts in a rule or
  test. Encode the upstream filing-status source first, then import its absolute
  RuleSpec output into downstream threshold, phaseout, deduction, and credit
  rules. If an already-encoded upstream filing-status output is unavailable,
  stop and encode that upstream source rather than synthesizing a local input.
  Do not preserve existing `#input.filing_status` or `#input.tax_filing_status`
  surfaces from copied target files; migrate them to upstream imports or
  source-backed non-status leaf facts such as whether a joint or separate return
  was actually made.
- The shared US tax filing-status output remains a structural enum: 0 single,
  1 joint return, 2 married filing separately, 3 head of household, and
  4 surviving spouse / qualifying widow(er). Never encode US tax filing status
  as string literals such as `"married_filing_jointly"` or as unbacked local
  boolean facts such as `married_filing_jointly`, `head_of_household`, or
  `surviving_spouse`. If a source provision itself defines a legal status or
  return category, encode that source-backed output at its absolute RuleSpec
  path and import it downstream. If a shared numeric filing-status output is
  available, import it and use the structural enum in formulas when the source
  supports that grouping, e.g.
  `match filing_status: 1 => joint_amount; 4 => joint_amount; ...`. If the source
  groups surviving spouse with joint return, every branch or match that handles
  status 1 must also handle status 4 in that same branch with the same result.
  If the source says only "joint return" without also naming surviving spouse
  or qualifying widow(er), do not route status 4 to the joint-return branch;
  status 4 falls under any "other case" branch unless the source states
  otherwise.
- Do not replace filing-status components with local status inputs such as
  `taxpayer_is_surviving_spouse`, `surviving_spouse`, or `head_of_household`.
  This also prohibits compound status predicates such as
  `individual_is_not_married_and_is_not_surviving_spouse`.
  Those are derived legal classifications; import their source-backed RuleSpec
  outputs or defer the affected output until those upstream definitions exist.
- If the source states a substitution, higher amount, increase, cap, or other
  modifier amount, do not define the modifier as an unused scalar while
  computing the affected numeric output without it. Use the modifier in the
  affected formula, or defer that affected output until the upstream branch
  condition can be encoded/imported. If you defer the affected output, list the
  deferred output under `module.deferred_outputs[]` and list the absolute target
  for the retained modifier parameter under that record's `source_values`. Include
  `blocked_by` only for exact upstream RuleSpec outputs with `#rule_fragment`;
  otherwise explain the unknown blocker in `reason`.
  Do not solve this by deleting the affected numeric output while leaving the
  modifier parameter stranded.
- When the source says a value or amount "in excess of" a stated limit is
  counted, included, deemed, or added, that is an executable excess formula.
  Encode the limit as a named scalar and encode the affected numeric output as
  `max(0, measured_value - limit)` or the source-stated equivalent, using a
  local factual input for `measured_value` if no source-backed upstream measure
  exists. Do not defer that excess output merely because a later aggregate
  resource, income, or liability calculation is outside the source slice; defer
  only the later aggregate if necessary.
- When the source states a final effective legal amount and also explains that
  amount as an increase by a percentage, inflation index, or cost-of-living
  adjustment, do not encode the explanatory percentage or index as a standalone
  scalar unless the source also supplies the prior base and the target formula
  uses that calculation. Encode the final effective amount as the operative
  scalar; keep the explanatory increase text in the summary or proof excerpt
  rather than as an unused modifier parameter.
- Supported relation aggregators are `len(relation)`,
  `count_where(relation, predicate_fact)`, `sum(relation.amount_fact)`, and
  `sum_where(relation, amount_fact_or_derived, predicate_fact)`. Do not write
  `sum(relation, expression)` or put arithmetic inside a relation field access.
  Use `sum(relation.amount_fact)` only when `amount_fact` is a raw scalar fact
  supplied directly on each relation row. Do not use `sum(relation.local_output)`
  for a `parameter` or `derived` rule defined in the same file; for a computed
  per-related-entity amount, write
  `sum_where(relation, local_output, source_stated_predicate_fact)` instead.
  To count two boolean conditions over the same relation, write two
  `count_where(...)` calls and add them.
- If a conditional is embedded inside arithmetic or another larger expression, wrap the whole conditional in parentheses, such as `amount + (if condition: extra else: 0)`. Do not write `amount + if condition: extra else: 0`.
- When source text says an amount "shall not include" or excludes "the part in
  excess of" a cap, the included amount is capped at that limit:
  `min(source_amount, cap)`. The excluded excess is
  `max(0, source_amount - cap)`, but do not return `source_amount - cap` or
  `source_amount - remaining_cap` as the included amount.
- Formula strings must use bare identifiers only. If an imported rule is listed
  as `us:statutes/...#example_rule`, add that exact target to `imports:` but
  reference `example_rule` inside formula text.
- Axiom conditionals are expression syntax, not YAML syntax. Money/scalar formulas may use `if condition: value else: other`; do not use Python ternary syntax, `else if`, or `elif`.
- `dtype: Judgment` formulas must not use `if ... else ...`. Write them as boolean expressions using `and`, `or`, `not`, comparisons, and parentheses. For example, encode `if exempt: net_ok else: net_ok and gross_ok` as `net_ok and (exempt or gross_ok)`.
- When using negated conjuncts, write them as a multiline formula with each `not <predicate>` term on its own line joined by `and`, rather than one compact `not A and not B` line.
- Do not use Python inline ternaries like `x if cond else y`.
- Use chained `if condition: value else: other_value` expressions; do not use YAML-style `if:` / `then:` / `else:` blocks, `else if`, or `elif`.
- Do not append a multiline conditional directly onto another expression, and do not use inline assignment syntax like `:=` inside formula blocks.
- For `dtype: Rate`, encode percentages as decimal ratios like `0.60` or `0.40`, never as `%` literals and never as arithmetic like `25 / 100` unless the source itself states both numerator and denominator.
- Do not simplify source-stated ratios or fractions into new decimal literals.
  If the source states `20/200`, encode grounded numerator and denominator
  parameters and compare with `20 / 200` or with those named parameters; do not
  emit an ungrounded decimal such as `0.10`.
- Use concrete ISO calendar dates like `2025-03-21` for day-level tests; do not use ISO week strings like `2025-W13`.
- Any substantive numeric literal in a formula must either appear in `./source.txt` or be one of -1, 0, 1, 2, or 3.
- Every substantive numeric occurrence in `./source.txt` must be represented by a named scalar definition in RuleSpec when it is a legal amount, rate, threshold, cap, or limit.
- If you encode a substantive numeric literal, `module.summary` or the rule's proof excerpt
  must include the exact source phrase containing that number. Do not omit a
  subsection, table row, or clause that grounds an encoded
  numeric amount, rate, threshold, cap, or limit.
- Represent every substantive source amount, rate, threshold, cap, or limit as a named `parameter` rule, then reference that parameter from derived formulas.
- If the same numeric value appears twice in materially different legal roles, including separate numbered exceptions or subparagraphs, give those roles distinct named scalars; otherwise reuse that named scalar everywhere the rule compares against or computes with that number.
- Adjacent bracket thresholds repeated as both an upper bound and the next bracket's lower bound are separate source-stated legal roles; define distinct semantic scalars for those occurrences and use them in the branch conditions.
- When a source says a subsection, paragraph, payment, credit, benefit,
  eligibility path, or other output "shall not apply" or "does not apply",
  the exported rule that says that target applies, is allowed, is included, or
  is eligible must negate the exception. Do not expose the exception only as a
  standalone helper while leaving the affected `*_applies`, eligibility,
  inclusion, exclusion, or amount output true under the exception.
- When that exception is encoded as a local derived helper, include a blocking
  companion test asserting both that helper as `holds` and the directly affected
  Judgment output as `not_holds`.
- For scoped exceptions, include a control case proving a non-excepted
  qualifying item is not reduced or blocked even when the exception amount or
  exception fact is positive/nonzero, plus a case where the same exception
  applies to the source-stated excepted category.
- Preserve anaphoric scope in source predicates. If the source says "such
  account", "such instrument", "through such account", "with respect to such
  payment", or similar same-object language, the predicate name and companion
  tests must keep that same-object relationship. Do not shorten it to broad
  activity for the person, broker, household, or entity generally.
- When a local formula has five or fewer independent source-stated boolean
  gates joined by `and`, include one all-gates-positive case and enough negative
  cases to toggle each gate at least once. Do not leave a source-stated gate
  untested just because another negative case toggles a different gate.
- If a formula negates multiple exception predicates, include a separate companion test for each predicate that sets that exception input true and expects the directly affected Judgment rule to be `not_holds`.
- For any negated exception predicate, include a paired positive case with the same output rule where only the exception input changes from `false` to `true`; do not combine the exception test with another branch change.
- Every local executable `kind: derived` or `kind: derived_relation` rule must
  appear at least once under an `output:` block in the companion `.test.yaml`;
  do not leave helper derived rules unasserted.
- Do not assert raw `kind: parameter` rules directly in companion test
  `output:` blocks. Cover parameters through derived outputs that consume them.
  If a module only contains parameters and has no derived output to assert,
  leave the companion test file empty.
- Each `.test.yaml` case may assert derived outputs for only one entity type. If
  a module defines outputs on multiple entities, create separate cases for each
  entity pair, such as `Person`/`TaxUnit`, `Person`/`Employer`, or
  `Employer`/`Payment`. For example:
  `Person` cases set person facts at the top level and assert person outputs;
  `TaxUnit` cases use relation rows to supply person facts and assert only
  tax-unit outputs. Do not assert relation-child outputs in the parent entity's
  case.
- Do not collapse a list of cited exceptions or cross-reference carve-outs into one aggregate fact such as `sections_..._do_not_preclude...`. Encode or import each cited exception separately, then combine them in a helper if useful.
- If `./source.txt` says someone is "aged 18 or over", "under 25", or similar, model the legal age predicate instead of inventing documentary age constants.
- When source text uses amendment markup like `[old] new`, treat the bracketed value as superseded text. Encode the current unbracketed value/effective date unless the task explicitly asks for historical text.
- If `./source.txt` makes an allowance, deduction, exemption, or eligibility branch conditional on billed, paid, incurred, anticipated, or other cost/expense facts, encode a positive fact predicate for that source-stated condition. Do not model availability solely as `not` other categories.
- When the cost/expense fact only matters after exclusion predicates, exported amount/quantity formulas consumed by dependent modules must guard the exclusions before referencing the branch-specific fact, so excluded cases do not require that fact as an input. For example, the amount should use `if other_allowance_eligible: 0 else: if household_has_telephone_cost: amount else: 0` rather than `if telephone_eligible: amount else: 0` when `telephone_eligible` itself references the branch-specific telephone-cost input.
- Phrases like `consists of the cost for X` or `available to households with X costs` require a positive fact for that cost/service. For example, a telephone allowance must depend on a fact for the household having or incurring the basic telephone-service cost before applying exclusions for other allowances.
- In a jurisdiction-specific repo, phrases that merely identify the target jurisdiction usually describe the document's scope, not a new input variable. Do not add a state-residency input unless the provision itself is encoding a residency eligibility test.
- If an encoded child paragraph depends on an operative parent condition, include the parent condition in `module.summary` and include both child and parent corpus paths under `module.source_verification.corpus_citation_paths` when those corpus paths are available.
- Do not create scalar variables for citation numbers, paragraph numbers, branch numbers, or source line labels.
- Do not invent `dtype: String` variables just to restate the effective date.
- Do not decompose legal dates into numeric `year`, `month`, or `day` scalar variables.
- Do not create public outputs for structural table row labels, household-size row indexes, or branch numbers. When those labels are source-stated numeric bounds or grid cells, keep them as private named numeric concepts or indexed table/grid values; consuming formulas should reference names rather than embedding the policy literal.
- If a source-stated scalar is needed to compute another local scalar, reference
  the named scalar concept instead of repeating its literal value in a formula.
  For example, if the source states a five-year period and a one-fifth fraction,
  encode the period as `benefit_cost_rate_compensation_lookback_years = 5` and
  the fraction as `1 / benefit_cost_rate_compensation_lookback_years`, not
  `1 / 5`.
- In a state-specific RuleSpec repository, if the source is a multi-state or
  multi-jurisdiction table, encode only the row(s) for the target repository's
  jurisdiction. Do not invent a fake `State` entity, row-index input, or
  all-state table surface just to preserve every row. Defer any broader
  all-state table output that cannot be represented faithfully.
- If the source cannot be represented faithfully with the supported schema, emit `module.status: deferred` or `module.status: entity_not_supported` with `rules: []`; do not invent unsupported ontology.
- Never emit `rules: []` without an explicit non-executable `module.status`. If the source has operative text, encode at least one source-backed rule instead of silently returning an empty module.
- For deferred or entity-not-supported artifacts, leave the companion `.test.yaml` empty. Do not create tests for deferred outputs or assertions against deferred symbols.
- If metadata or context names an absolute canonical target that this source `sets`, `amends`, `implements`, or `restates`, add a separate `kind: source_relation` record with `source_relation.type` and `source_relation.target`. Do not put source graph edges in executable rule metadata.
- Preserve existing or copied `kind: source_relation` records unless `./source.txt` proves the legal/provenance edge is wrong.
- For state-set standards, allowances, thresholds, or options implementing federal delegation, include `source_relation.value` pointing to the local executable RuleSpec output and `source_relation.basis.delegation` when context identifies the upstream delegated slot.
- Federal provisions that authorize state agencies to set a value create the delegated slot; encode those source graph records with `source_relation.type: delegates`. Reserve `source_relation.type: sets` or `implements` for the state or implementing authority that fills that slot, and always include `source_relation.basis.delegation` for `sets` or `implements`.
- When the source says a value is determined `in accordance with section X`, emit the upstream import instead of restating the concept locally when that import target is available.
- When the source uses `except`, `unless`, or `subject to` with cited sections
  or same-section subsections, do not create local `section_...` or
  `subsection_...` inputs for those cited sources. Import the cited RuleSpec
  source when it exists; if the target source is needed but unavailable, stop
  with an explicit missing-upstream/dependency request instead of encoding an
  opaque placeholder.
- For opening scope phrases such as `except as provided in clause (ii)` that
  point to a sibling clause outside the requested target and no copied context
  supplies that sibling's executable output, do not invent a local boolean like
  `clause_ii_provides_otherwise`. Keep the current target scoped to the
  source-stated positive calculation, or defer only the final affected surface
  if the sibling exception is essential to the requested output.
- A pure `notwithstanding subsection ...` override does not require importing
  the overridden subsection unless the formula actually needs that cited
  subsection's computed output.
- If the cited same-section subsection or sibling paragraph is supplied in context as a RuleSpec file, do not summarize it into a local fact like `person_meets_...requirements`. For operative `except`, `unless`, or `subject to` carve-outs that can change the requested output, a bare file-level import is not enough: import the exact `#rule_name` exported by the cited file and reference that bare symbol in the affected formula, usually negated or used as a branch guard. Validation rejects file-level imports for operative sibling carve-outs when the formula never uses a cited output.
- If the cited sibling file is deferred, empty, unsupported, or missing a usable exported rule and the carve-out changes the result, defer the affected executable output or encode a source-grounded overriding branch that avoids the dependency. Do not emit a formula that ignores the carve-out, and do not invent a local boolean for the cited sibling source.
- File-level imports without a `#symbol` fragment are acceptable only for non-operative provenance or boundary context, such as a pure `notwithstanding` override or a local source-stated override where the formula does not depend on the cited output. They are not acceptable for `except`, `unless`, or `subject to` formula carve-outs.
- Example: if the requested source says `Subject to paragraph (c)` and copied context contains `regulations/.../c.yaml` exporting `cash_assistance_less_restrictive_methodologies_may_be_applied`, import `us:regulations/.../c#cash_assistance_less_restrictive_methodologies_may_be_applied` and include `cash_assistance_less_restrictive_methodologies_may_be_applied` in the affected formula and proof atoms. A formula that repeats only the positive paragraph requirements while omitting the cited paragraph's symbol is invalid.
- Do not copy the body of a cited cross-reference provision into this module's `summary` or re-encode that cited provision locally. Keep this module scoped to the requested citation and import the cited provision instead.
- Do not fabricate sibling-file imports, do not guess unavailable import targets, and do not invent `import` statements or `imports:` blocks for uncopied same-instrument provisions.
- Before using any imported output in arithmetic, check the copied context
  export's `dtype:`. An imported `dtype: Judgment` is a predicate, not a scalar
  amount, rate, or base. Never multiply, add, subtract, divide, `min`, or `max`
  a Judgment import as if it were Money, Rate, Count, or another numeric value.
  If the current source states a numeric base such as wages, remuneration,
  payments, or amounts attributable to a category and the copied import only
  identifies whether an item is attributable to that category, encode the source-stated numeric base as a local amount fact, or as a relation-filtered
  aggregate only when a compatible relation and numeric amount field are
  present. If neither is available, defer the numeric output instead of using
  the Judgment import as a placeholder scalar.
- Before finalizing, do this self-check:
  1. Numeric inventory: every source-stated legal amount, rate, threshold, cap,
     or limit has a named local numeric concept or an exact imported concept
     from context, and derived formulas reference that local or imported name
     rather than an inline literal. For tables and grids, bounds and cells are
     indexed numeric concepts; formulas may use structural integer keys only to
     select from those concepts. If an exact same-path child scalar is available
     in context, import it instead of duplicating it locally.
  1a. Dependency inventory: no local derived rule formula references its own
      rule name. If a legal phrase is both a required fact and a desired output,
      rename the output to the conclusion and keep the required fact as a
      distinct local input.
  2. Test input inventory: for every local factual identifier referenced by a
     local derived formula, every companion test case assigns the corresponding
     `#input.<fact>` explicitly, including false facts. Do not rely on implicit
     defaults. Explicit rate-only source-boundary artifacts that contain only
     scalar parameters may assert those canonical parameter outputs directly.
     Do not assert raw `kind: parameter` rules directly in companion test `output:` blocks for other artifacts; assert derived outputs that consume the parameters instead.
     For imported modules, only assign imported `#input` or `#relation` keys
     that exist in the current imported RuleSpec context. Do not preserve stale
     imported test inputs from copied files. Do not stub imported derived
     outputs as test inputs; imported programs are computed. If the downstream
     rule depends on an imported output, assign all current upstream factual
     inputs and relations needed by that imported output, including false facts.
     This does not override no-input guardrails: never assign prohibited derived
     classifications such as any imported or local `#input.filing_status` or
     `#input.tax_filing_status`. This prohibition is absolute even when the
     value is a numeric enum and even when the key belongs to an imported
     module. If an imported output cannot be exercised without those prohibited
     test inputs, omit that assertion or encode the upstream filing-status
     sources first.
  3. Proof inventory: every proof atom uses only an allowed `kind`; imported
     proof atoms include `import.target`, `import.output`, and `import.hash`;
     textual claim support is either direct corpus source support or a claim ID
     listed under `module.source_claims`.
  4. Import inventory: every `imports:` entry is an exact copied/importable
     RuleSpec target. Top-level `imports:` entries must be scalar strings; never
     map entries like `- target:` plus `symbols:`. Do not guess sibling paths; if
     required upstream context is missing, emit a typed missing-upstream/dependency
     request instead.
     Never drop the jurisdiction prefix from copied context imports: use
     `us:statutes/...#symbol`, not `statutes/...#symbol`.
{target_hint}
Additional encoding guidance:
{additional_guidance}

Minimal RuleSpec shape:
```yaml
format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: <corpus citation path from this prompt>
  summary: |-
    <concise exact audit excerpt from the source text>
rules:
  - name: example_amount
    kind: parameter
    dtype: Money
    unit: USD
    source: <legal citation/span>
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: <corpus citation path from this prompt>
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          451
  - name: example_output
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    source: <legal citation/span>
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          if example_condition: example_amount else: 0
```

Derived membership shape:
```yaml
rules:
  - name: snap_member_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 7 CFR 273.1(a)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          member_has_required_status
          and not member_is_excluded_student
  - name: snap_unit
    kind: derived_relation
    derived_relation:
      arity: 2
      source_relation: member_of_household
      entity: SnapUnit
      member_relation: members
      slot_entities: [Person, Household]
    source: 7 CFR 273.1(a)
    versions:
      - effective_from: '2026-01-01'
        formula: snap_member_eligible
```

{output_rules}
Do not respond with summaries, markdown prose, or file-write confirmations.
"""


def _workspace_corpus_citation_path(workspace: EvalWorkspace) -> str | None:
    source_metadata = workspace.source_metadata
    if not isinstance(source_metadata, dict):
        return None
    raw_value = source_metadata.get("corpus_citation_path")
    if not isinstance(raw_value, str):
        return None
    citation_path = raw_value.strip()
    return citation_path or None


def _build_eval_prompt(
    citation: str,
    mode: EvalMode,
    workspace: EvalWorkspace,
    context_files: list[EvalContextFile],
    target_file_name: str,
    target_ref_prefix: str | None = None,
    include_tests: bool = False,
    runner_backend: str = "codex",
    policyengine_rule_hint: str | None = None,
) -> str:
    """Build a prompt-only eval request with explicit provenance rules."""
    return _build_rulespec_eval_prompt(
        citation=citation,
        mode=mode,
        workspace=workspace,
        context_files=context_files,
        target_file_name=target_file_name,
        target_ref_prefix=target_ref_prefix,
        include_tests=include_tests,
        runner_backend=runner_backend,
        policyengine_rule_hint=policyengine_rule_hint,
    )


def _is_single_amount_table_slice(source_text: str) -> bool:
    """Return True when source text is a single amount-bearing table row slice."""
    marker = "Structured table:"
    if marker in source_text:
        table_section = source_text.split(marker, 1)[1]
        lines = [line.strip() for line in table_section.splitlines() if line.strip()]
        amount_rows = [line for line in lines if "|" in line and re.search(r"\d", line)]
        if len(amount_rows) == 1:
            return True

    money_matches = re.findall(r"[£$€]\s*\d[\d,]*(?:\.\d+)?", source_text)
    if len(money_matches) != 1:
        return False
    if _CONDITIONAL_AMOUNT_SLICE_PATTERN.search(source_text):
        return False

    money_lines = [
        line.strip()
        for line in source_text.splitlines()
        if re.search(r"[£$€]\s*\d[\d,]*(?:\.\d+)?", line)
    ]
    return len(money_lines) == 1


def _source_identifier_requests_percentage_rate_boundary(
    citation: str, source_text: str
) -> bool:
    parts = [part for part in citation.strip().strip("/").split("/") if part]
    if not parts or parts[-1].lower() != "rate":
        return False
    return bool(re.search(r"(?i)(?:\bper\s+cent\b|\bpercent(?:age)?\b|%)", source_text))


def _format_subparagraph_coverage_checklist(
    source_text: str, corpus_citation_path: str | None
) -> str:
    """Pre-compute the validator's subparagraph coverage list and inject it as a
    structured checklist the model must satisfy.

    The source-subparagraph-coverage validator (validator_pipeline.py) flags
    every high-signal top-level subparagraph that the encoded artifact neither
    cites in a rule's ``source:`` nor lists in ``module.deferred_outputs``.
    Adding more abstract instructions to the prompt failed to make gpt-class
    models comply (two prior iterations either ignored the directive or
    omitted more subparagraphs). The structural fix is to do the enumeration
    deterministically here, render it next to the source text, and force the
    model to acknowledge each subparagraph by name. Same data the validator
    uses, same expectation, no model-side enumeration discretion.
    """
    if not corpus_citation_path:
        return ""
    try:
        from .validator_pipeline import (
            _compact_source_excerpt,
            _format_source_subparagraph_citation,
            _high_signal_top_level_subparagraphs,
        )
    except ImportError:
        return ""
    subparagraphs = _high_signal_top_level_subparagraphs(source_text)
    if not subparagraphs:
        return ""
    rows = []
    for label, text in subparagraphs:
        citation = _format_source_subparagraph_citation(corpus_citation_path, label)
        excerpt = _compact_source_excerpt(text, limit=100)
        rows.append(f"  - {citation}: {excerpt}")
    rows_text = "\n".join(rows)
    return f"""
=== BEGIN SUBPARAGRAPH COVERAGE CHECKLIST ===
The source text above contains {len(subparagraphs)} high-signal top-level
subparagraphs. Your output MUST account for every one. For each subparagraph
below, the encoded artifact must contain EITHER:
  (a) An executable rule whose `source:` field cites that subparagraph
      (e.g. `source: 7 USC 2014(d)`), OR
  (b) An entry under `module.deferred_outputs[]` whose `output:` target path
      includes that subparagraph segment (e.g.
      `us:statutes/7/2014/d#unspecified_output`) and whose `reason:` names
      the legal dependency or scope reason blocking encoding.

Legislative findings, preambles, intent clauses, and purpose clauses are still
source coverage items. If the text is non-operative, satisfy this checklist with
`module.deferred_outputs[]` rather than inventing an executable formula.

Subparagraphs without (a) or (b) are an automatic CI failure under the
source-subparagraph-coverage validator. There is no implicit "covered by
the umbrella rule" path — composite rules cite the parent section, not
the children.

The citation strings below are exact validator keys, not examples. To satisfy
coverage with an executable rule, copy the relevant string exactly into that
rule's `source:` field, such as `source: us-ny/regulation/.../5(a)`. A
human-readable source like `18 NYCRR 387.14(a)(5)(i)(a)` may be useful in prose
but does not satisfy this checklist. If a top-level checklist item has nested
legal clauses, at least one rule for that top-level item must cite the exact
top-level checklist string, or the item must be listed under
`module.deferred_outputs`.

Subparagraphs requiring action:
{rows_text}

When in doubt, prefer (b): a `deferred_outputs` entry with a clear
`reason` is honest and cheap to refine later. Silent omission is not.
=== END SUBPARAGRAPH COVERAGE CHECKLIST ===
"""


def render_rate_only_source_boundary_guidance() -> str:
    return """
Rate-only source boundary:
- The requested source id ends in `/rate`, and this source states a percentage
  rate, so this artifact must expose only source-stated rate or percentage
  parameters anchored in `./source.txt` for that branch.
- Do not encode the downstream tax, contribution, credit, deduction, wage
  base, income base, exemption, exception, or other non-rate output merely
  because the broader source text mentions it.
- Prefer `kind: parameter`, `dtype: Rate`, and one scalar output per
  source-stated rate. Use a `derived` rate only when the source itself states
  the rate as an expression of other rates.
- Name each output after the legal application stated in the source text, such
  as `<tax_or_application>_rate`, not after the path fragment alone.
- Do not import a consumer or base source solely to compute a rate. A rate-only
  boundary must stay acyclic and reusable by base definitions that cite the
  rate.
- Because this is a pure rate boundary, companion tests may assert the
  canonical parameter output directly when there is no derived output to
  exercise.
"""


def _format_inline_context_snippets(
    workspace: EvalWorkspace,
    context_files: list[EvalContextFile],
    max_chars_per_file: int = 6000,
) -> str:
    """Inline copied precedent files for non-tool backends like Responses API."""
    snippets: list[str] = []
    for item in context_files:
        path = workspace.root / item.workspace_path
        try:
            content = path.read_text().strip()
        except OSError:
            continue
        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file].rstrip() + "\n... [truncated]"
        snippets.append(f"=== FILE: {item.workspace_path} ===\n{content}")
    return "\n\n".join(snippets)


def _format_context_file_listing(
    item: EvalContextFile,
    *,
    include_label: bool = False,
) -> str:
    """Format one copied context file for prompt display."""
    details = f": {item.label or item.source_path}" if include_label else ""
    kind = f" (kind: {item.kind})"
    context_hash = _context_file_hash(item.source_path)
    hash_detail = f"; context hash `{context_hash}`" if context_hash else ""
    export_detail = _context_file_export_detail(item)
    if item.workspace_path == item.import_path:
        return f"- `{item.workspace_path}`{hash_detail}{export_detail}{details}{kind}"
    return (
        f"- inspect `{item.workspace_path}`; import target `{item.import_path}`"
        f"{hash_detail}{export_detail}{details}{kind}"
    )


def _format_policyengine_hint_context_exports(
    context_files: list[EvalContextFile],
    *,
    max_exports: int = 40,
) -> str:
    """Return PE-hint scoped executable context exports for prompt guidance."""
    export_lines: list[str] = []
    for item in context_files:
        if item.kind.endswith("_test_context"):
            continue
        surfaces = _context_file_executable_surfaces(item.source_path)
        for name, surface in sorted(surfaces.items()):
            dtype = str(surface.get("dtype") or "").strip()
            if dtype != "Judgment":
                continue
            kind = str(surface.get("kind") or "").strip()
            entity = str(surface.get("entity") or "").strip()
            details = ", ".join(
                part
                for part in [
                    f"kind={kind}" if kind else "",
                    f"entity={entity}" if entity else "",
                    f"dtype={dtype}",
                ]
                if part
            )
            export_lines.append(f"- `{item.import_path}#{name}` ({details})")
    if not export_lines:
        return ""
    rendered = export_lines[:max_exports]
    if len(export_lines) > max_exports:
        rendered.append(
            f"- ... {len(export_lines) - max_exports} additional Judgment exports omitted"
        )
    return """
Executable Judgment exports visible in copied context for the hinted output.
Prefer these exact imports before creating any local branch/category/status
placeholder for the oracle-facing output:
{lines}
""".format(lines="\n".join(rendered))


_CANONICAL_CONCEPT_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]*")


def _canonical_concept_token_index(registry: ConceptRegistry) -> dict[str, Concept]:
    """Return a name → concept index covering canonicals and blocked synonyms."""
    index: dict[str, Concept] = {}
    for concept in registry.concepts_by_id.values():
        index[concept.canonical_name] = concept
        for synonym in concept.blocked_synonyms:
            index[synonym] = concept
    return index


def _format_canonical_concept_registry_guidance(
    source_text: str,
    workspace: EvalWorkspace,
    context_files: list[EvalContextFile],
    *,
    registry: ConceptRegistry | None = None,
) -> str:
    """Inject canonical-concept registry directives scoped to mentioned concepts.

    Scans source text plus copied context files for any canonical name or
    blocked synonym in the registry; emits a terse "use these exact names"
    block for the matched concepts only. Concepts that never appear in any
    text are omitted so the prompt does not pay tokens for irrelevant rules.
    """
    if registry is None:
        try:
            registry = load_concept_registry()
        except (OSError, ValueError):
            return ""
    if not registry.concepts_by_id:
        return ""

    haystack_parts: list[str] = [source_text]
    for item in context_files:
        path = workspace.root / item.workspace_path
        try:
            haystack_parts.append(path.read_text())
        except OSError:
            continue
    haystack_tokens = set(
        _CANONICAL_CONCEPT_TOKEN_RE.findall("\n".join(haystack_parts))
    )
    if not haystack_tokens:
        return ""

    token_index = _canonical_concept_token_index(registry)
    matched: list[Concept] = []
    seen_ids: set[str] = set()
    for token in haystack_tokens:
        concept = token_index.get(token)
        if concept is None or concept.id in seen_ids:
            continue
        seen_ids.add(concept.id)
        matched.append(concept)

    if not matched:
        return ""

    matched.sort(key=lambda c: c.id)
    lines: list[str] = []
    for concept in matched:
        parts: list[str] = [f"`{concept.canonical_name}`"]
        if concept.has_producer:
            parts.append(f"producer `{concept.producer_anchor}`")
        if concept.blocked_synonyms:
            blocked = ", ".join(f"`{s}`" for s in concept.blocked_synonyms)
            parts.append(f"do not use: {blocked}")
        lines.append("- " + " — ".join(parts))

    return """
Canonical concept names:
Use these exact identifiers for the listed legal concepts; never introduce the blocked synonyms. The post-apply validator rejects drift, so picking the canonical name on the first pass avoids wasted re-encodes:
{lines}
""".format(lines="\n".join(lines))


def _format_existing_target_contract_guidance(
    context_files: list[EvalContextFile],
) -> str:
    """Return explicit public-surface contracts for copied target files."""
    contract_lines: list[str] = []
    for item in context_files:
        if item.kind != "existing_target":
            continue
        surfaces = _context_file_executable_surfaces(item.source_path)
        for name, surface in surfaces.items():
            details = [
                f"kind={surface.get('kind') or ''}",
                f"entity={surface.get('entity') or ''}",
                f"dtype={surface.get('dtype') or ''}",
                f"period={surface.get('period') or ''}",
            ]
            unit = surface.get("unit")
            if unit:
                details.append(f"unit={unit}")
            indexed_by = surface.get("indexed_by") or ()
            if indexed_by:
                details.append(f"indexed_by={','.join(indexed_by)}")
            effective_dates = surface.get("effective_dates") or ()
            if effective_dates:
                details.append(f"effective_from={','.join(effective_dates)}")
            contract_lines.append(
                f"- `{item.import_path}#{name}` ({'; '.join(details)})"
            )
    if not contract_lines:
        return ""
    return """
Existing target executable surfaces:
The copied current target exports these executable names for inspection. They
are not compatibility contracts. Preserve a name only when it remains the
cleanest source-faithful surface under current validation; otherwise rename,
rebuild, drop, or defer it:
{lines}
""".format(lines="\n".join(contract_lines))


def _format_existing_target_invalid_input_guidance(
    context_files: list[EvalContextFile],
) -> str:
    """Return copied target input names that current invariants must reject."""
    invalid_lines: list[str] = []
    for item in context_files:
        if item.kind != "existing_target":
            continue
        invalid_inputs = _context_file_invalid_local_inputs(
            item.source_path,
            context_files=context_files,
        )
        for name, reason in invalid_inputs.items():
            invalid_lines.append(f"- `{item.import_path}#input.{name}`: {reason}")
    if not invalid_lines:
        return ""
    return """
Invalid copied local input names:
The copied current target uses these local factual input names, but current RuleSpec invariants reject them. Do not preserve these exact names in formulas or tests:
{lines}
""".format(lines="\n".join(invalid_lines))


def _format_existing_target_validation_guidance(
    context_files: list[EvalContextFile],
) -> str:
    """Return current validation failures for copied target files."""
    issue_lines: list[str] = []
    for item in context_files:
        if item.kind != "existing_target":
            continue
        issues = _context_file_current_validation_issues(item.source_path)
        for issue in issues:
            issue_lines.append(f"- `{item.import_path}`: {issue}")
    if not issue_lines:
        return ""
    return """
Copied existing target fails current RuleSpec validation:
The copied current target is stale under the current encoder invariants. Do not
preserve the failing shape. Repair the generated target so these validation
issues are gone. If a copied import no longer resolves in the clean repo
context, remove that import and defer the affected executable surface instead of
preserving a dirty-tree dependency. For deferred executable surfaces, record
exact provenance under `module.deferred_outputs[]` with absolute `output`,
`source_values`, and exact `blocked_by` targets as applicable. If an exact
upstream RuleSpec output is unknown, omit `blocked_by` and explain the legal
dependency in `reason` instead of inventing a bare provision target:
{lines}
""".format(lines="\n".join(issue_lines))


def _context_file_current_validation_issues(source_path: str) -> list[str]:
    """Run lightweight current-generation validators against a copied file."""
    try:
        content = Path(source_path).read_text()
    except OSError:
        return []

    issues: list[str] = []
    issues.extend(find_deferred_output_issues(content))
    issues.extend(find_unused_modifier_parameter_issues(content))
    issues.extend(_context_file_unresolved_import_issues(source_path, content))
    return issues


def _context_file_unresolved_import_issues(source_path: str, content: str) -> list[str]:
    """Return imports from copied files that do not resolve in the clean repo."""
    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if not isinstance(payload, dict):
            return []
        imports = payload.get("imports")
        if not isinstance(imports, list):
            return []

        policy_repo_root = _rulespec_repo_root_for_context_path(Path(source_path))
        issues: list[str] = []
        for raw_item in imports:
            if not isinstance(raw_item, str):
                continue
            reference = raw_item.strip().strip("\"'")
            import_path, separator, symbol = reference.partition("#")
            import_path = import_path.strip()
            symbol = symbol.strip()
            if not import_path:
                continue

            target_file = _first_existing_context_import_file(
                import_path, policy_repo_root
            )
            if target_file is None:
                issues.append(
                    f"Import `{reference}` does not resolve to a RuleSpec file "
                    "in the clean repo context."
                )
                continue
            if separator and symbol:
                exports = set(_context_file_exports(str(target_file)))
                if symbol not in exports:
                    issues.append(
                        f"Import `{reference}` resolves to `{target_file}` but "
                        f"that file does not export `{symbol}`."
                    )
        return issues
    return []


def _first_existing_context_import_file(
    import_path: str,
    policy_repo_root: Path,
) -> Path | None:
    """Resolve one copied context import to the first visible RuleSpec file."""
    for candidate in _candidate_import_rule_files(import_path, policy_repo_root):
        if candidate.exists():
            return candidate
    return None


def _rulespec_repo_root_for_context_path(path: Path) -> Path:
    """Infer the nearest `rulespec-*` checkout root containing a context file."""
    resolved = path.resolve()
    start = resolved if resolved.is_dir() else resolved.parent
    for parent in (start, *start.parents):
        if parent.name.startswith("rulespec-"):
            return parent
    return start


def _format_existing_target_valid_input_guidance(
    context_files: list[EvalContextFile],
) -> str:
    """Return copied target factual input names that should be preserved."""
    valid_lines: list[str] = []
    for item in context_files:
        if item.kind != "existing_target":
            continue
        invalid_inputs = set(
            _context_file_invalid_local_inputs(
                item.source_path,
                context_files=context_files,
            )
        )
        valid_inputs = sorted(
            name
            for name in _context_file_local_inputs(item.source_path)
            if name not in invalid_inputs
        )
        for name in valid_inputs:
            valid_lines.append(f"- `{item.import_path}#input.{name}`")
    if not valid_lines:
        return ""
    return """
Existing target local factual inputs:
These copied target factual input names are not rejected by the current prompt
preflight. Reuse one only when it remains a clean source-stated fact for the new
encoding; otherwise replace it with the source-faithful surface:
{lines}
""".format(lines="\n".join(valid_lines))


def _format_cycle_prone_context_import_guidance(
    context_files: list[EvalContextFile],
    *,
    target_ref_prefix: str | None,
) -> str:
    """Return context imports that would cycle back into the current target."""
    if not target_ref_prefix:
        return ""
    blocked: list[str] = []
    for item in context_files:
        if item.kind == "existing_target":
            continue
        if _context_file_imports_target(item.source_path, target_ref_prefix):
            blocked.append(
                f"- `{item.import_path}` already imports `{target_ref_prefix}`; "
                "do not import it from this target"
            )
    if not blocked:
        return ""
    return """
Cycle-prone context imports:
These copied context files already depend on the current target. Importing them from this target would create a RuleSpec import cycle:
{lines}
Use only source-stated local predicates, existing non-cyclic imports, or an explicit missing-upstream/dependency status instead.
""".format(lines="\n".join(blocked))


def _format_partial_extent_child_schema_limit_guidance(
    source_text: str,
    context_files: list[EvalContextFile],
    *,
    target_ref_prefix: str | None,
) -> str:
    """Return target-specific guidance for unsupported partial child rewiring."""
    if not target_ref_prefix:
        return ""
    scoped_source_text = _target_source_scope_for_heuristics(
        source_text, target_ref_prefix
    )
    if _source_opens_amount_adjustment_list(scoped_source_text):
        return ""
    if not _source_has_partial_extent_child_rewiring_limit(scoped_source_text):
        return ""
    child_prefix = f"{target_ref_prefix.rstrip('/')}/"
    child_terminal_refs: list[str] = []
    for item in context_files:
        if not item.import_path.startswith(child_prefix):
            continue
        for export in _context_file_terminal_exports(item.source_path):
            child_terminal_refs.append(f"{item.import_path}#{export}")
    if not child_terminal_refs:
        return ""

    refs = ", ".join(f"`{ref}`" for ref in child_terminal_refs[:6])
    if len(child_terminal_refs) > 6:
        refs += ", ..."
    return f"""
Target-specific schema limit:
- `./source.txt` uses `to the extent` and copied child-fragment files already
  export executable results ({refs}). Under the current executable schema, this
  parent cannot faithfully recompute those child results using a locally
  adjusted basis, and it cannot wire that adjusted basis into imported child
  results. Emit `module.status: entity_not_supported` or `deferred` with
  `rules: []` and an empty companion test file. Do not create a
  `*_before_exemption` executable output or an adjusted local wage/base helper
  for this parent.
"""


def _format_parent_child_terminal_output_guidance(
    context_files: list[EvalContextFile],
    *,
    target_ref_prefix: str | None,
) -> str:
    """Return concrete child-output imports for aggregate parent encodings."""
    if not target_ref_prefix:
        return ""
    child_prefix = f"{target_ref_prefix.rstrip('/')}/"
    lines: list[str] = []
    local_inputs: set[str] = set()
    for item in context_files:
        if item.kind.endswith("_test_context"):
            continue
        if not item.import_path.startswith(child_prefix):
            continue
        terminal_exports = _context_file_terminal_exports(item.source_path)
        if not terminal_exports:
            continue
        surfaces = _context_file_executable_surfaces(item.source_path)
        export_details: list[str] = []
        for name in terminal_exports:
            surface = surfaces.get(name) or {}
            dtype = str(surface.get("dtype") or "").strip()
            kind = str(surface.get("kind") or "").strip()
            entity = str(surface.get("entity") or "").strip()
            annotations = ", ".join(part for part in (kind, dtype, entity) if part)
            suffix = f" ({annotations})" if annotations else ""
            export_details.append(f"`{item.import_path}#{name}`{suffix}")
        inputs = sorted(_context_file_local_inputs(item.source_path))
        local_inputs.update(inputs)
        input_note = ""
        if inputs:
            visible_inputs = ", ".join(f"`{name}`" for name in inputs[:8])
            if len(inputs) > 8:
                visible_inputs += ", ..."
            input_note = f"; child-local inputs: {visible_inputs}"
        lines.append(
            f"- `{item.import_path}` terminal exports: "
            + ", ".join(export_details)
            + input_note
        )
    if not lines:
        return ""

    input_guidance = ""
    if local_inputs:
        visible_inputs = ", ".join(f"`{name}`" for name in sorted(local_inputs)[:12])
        if len(local_inputs) > 12:
            visible_inputs += ", ..."
        input_guidance = (
            "\n- These child-local input names are already owned by child files: "
            f"{visible_inputs}. Do not recreate a child branch by copying those "
            "inputs into the parent when a terminal child output above is "
            "available."
        )

    return f"""
Aggregate parent child outputs detected:
{chr(10).join(lines)}
- For this aggregate parent, start from the terminal child outputs above.
  Import each terminal child output needed by the parent and compose those
  imported bare names directly. For numeric parent outputs, prefer terminal
  `Money`, `Rate`, or `Count` child outputs over child predicates, rates, or raw
  factual inputs.
- Do not rebuild a child branch in the parent from the child's local facts,
  helper predicates, or numeric literals. If a copied child already exports a
  terminal numeric result, the parent formula should add, cap, select, subtract,
  or otherwise compose that imported result.
- If a copied child only exports a terminal `Judgment` and the source requires a
  numeric parent amount for that branch, you may use a source-stated local
  amount fact for that judgment-only branch, but still import all available
  terminal numeric outputs for sibling child branches instead of recomputing
  them locally.{input_guidance}
"""


def _target_source_scope_for_heuristics(
    source_text: str, target_ref_prefix: str
) -> str:
    """Return the apparent target paragraph slice for prompt heuristics.

    Some corpus pages contain a whole section even when the requested target is
    a child paragraph. Prompt heuristics should not let unrelated sibling text
    trigger target-specific instructions.
    """
    fragments = _legal_fragments_from_rulespec_ref(target_ref_prefix)
    if not fragments:
        return source_text

    hierarchical_scope = _target_source_scope_by_cfr_hierarchy(source_text, fragments)
    if hierarchical_scope is not None:
        return hierarchical_scope

    start = 0
    end = len(source_text)
    search_from = 0
    for depth, fragment in enumerate(fragments):
        match = _find_structural_parenthetical_marker(
            source_text,
            fragment,
            search_from,
            end,
            depth=depth,
        )
        if not match:
            return source_text
        start = match.start("marker")
        search_from = match.end("marker")

        sibling = _find_structural_sibling_marker(
            source_text,
            fragment,
            search_from,
            end,
            depth=depth,
        )
        if sibling:
            end = sibling.start("marker")
    return source_text[start:end]


def _target_source_scope_by_cfr_hierarchy(
    source_text: str,
    fragments: list[str],
) -> str | None:
    """Slice CFR-style parenthetical hierarchy by legal marker level.

    CFR paragraphs reuse marker spellings at deeper levels, such as top-level
    numeric ``(3)`` paragraphs and nested upper-alpha list item ``(3)`` markers.
    A simple regex search for ``(3)`` can therefore select the wrong branch. This
    scanner tracks the conventional CFR marker sequence:
    lower-alpha, then numeric/lower-roman/upper-alpha repeating.
    """
    if any(not _cfr_marker_kind_ordinals(fragment) for fragment in fragments):
        return None

    stack: list[tuple[str, str, int]] = []
    target = tuple(fragments)
    target_start: int | None = None
    target_level: int | None = None

    for match in _iter_cfr_structural_markers(source_text):
        token = match.group("token")
        assigned = _assign_cfr_marker_level(stack, token)
        if assigned is None:
            continue
        level, kind, ordinal = assigned
        stack = stack[:level]
        stack.append((kind, token, ordinal))
        path = tuple(item[1] for item in stack)

        if target_start is None:
            if path == target:
                target_start = match.start("marker")
                target_level = level
            continue

        if target_level is not None and level <= target_level:
            return source_text[target_start : match.start("marker")]

    if target_start is not None:
        return source_text[target_start:]
    return None


def _iter_cfr_structural_markers(source_text: str) -> Iterable[re.Match[str]]:
    """Yield parenthetical markers that appear in structural positions."""
    marker_pattern = re.compile(
        r"(?P<marker>\((?P<token>[A-Za-z0-9]+)\))"
        r"(?=\s+|\([A-Za-z0-9]+\))"
    )
    last_yielded_marker_end: int | None = None
    for match in marker_pattern.finditer(source_text):
        marker_start = match.start("marker")
        line_start = source_text.rfind("\n", 0, marker_start) + 1
        if not source_text[line_start:marker_start].strip():
            last_yielded_marker_end = match.end("marker")
            yield match
            continue

        if last_yielded_marker_end == marker_start:
            last_yielded_marker_end = match.end("marker")
            yield match
            continue

        previous = match.start("marker") - 1
        while previous >= 0 and source_text[previous].isspace():
            previous -= 1
        follows_spaced_marker = (
            previous >= 0
            and source_text[previous] == ")"
            and marker_start > 0
            and source_text[marker_start - 1].isspace()
        )
        if previous < 0 or source_text[previous] in "\n.;:" or follows_spaced_marker:
            last_yielded_marker_end = match.end("marker")
            yield match


def _assign_cfr_marker_level(
    stack: list[tuple[str, str, int]],
    token: str,
) -> tuple[int, str, int] | None:
    possible = _cfr_marker_kind_ordinals(token)
    if not possible:
        return None

    child_level = len(stack)
    expected_child_kind = _cfr_marker_kind_for_level(child_level)
    if expected_child_kind == "lower_roman":
        for level in range(len(stack) - 1, -1, -1):
            expected_kind = _cfr_marker_kind_for_level(level)
            for kind, ordinal in possible:
                if (
                    kind == expected_kind
                    and expected_kind == "lower_alpha"
                    and ordinal == stack[level][2] + 1
                ):
                    return (level, kind, ordinal)
        for kind, ordinal in possible:
            if kind == expected_child_kind:
                return (child_level, kind, ordinal)

    for level in range(len(stack) - 1, -1, -1):
        expected_kind = _cfr_marker_kind_for_level(level)
        for kind, ordinal in possible:
            if kind == expected_kind and ordinal > stack[level][2]:
                return (level, kind, ordinal)

    for kind, ordinal in possible:
        if kind == expected_child_kind:
            return (child_level, kind, ordinal)
    return None


def _cfr_marker_kind_for_level(level: int) -> str:
    if level == 0:
        return "lower_alpha"
    return ("numeric", "lower_roman", "upper_alpha")[(level - 1) % 3]


def _cfr_marker_kind_ordinals(token: str) -> list[tuple[str, int]]:
    kinds: list[tuple[str, int]] = []
    if re.fullmatch(r"\d+", token):
        kinds.append(("numeric", int(token)))
    if re.fullmatch(r"[a-z]", token):
        kinds.append(("lower_alpha", ord(token) - ord("a") + 1))
    if re.fullmatch(r"[A-Z]", token):
        kinds.append(("upper_alpha", ord(token) - ord("A") + 1))
    if re.fullmatch(r"[ivxlcdm]+", token):
        roman = _lower_roman_to_int(token)
        if roman is not None:
            kinds.append(("lower_roman", roman))
    return kinds


def _lower_roman_to_int(token: str) -> int | None:
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    previous = 0
    for char in reversed(token):
        value = values.get(char)
        if value is None:
            return None
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total or None


def _find_structural_parenthetical_marker(
    source_text: str,
    fragment: str,
    start: int,
    end: int,
    *,
    depth: int,
) -> re.Match[str] | None:
    """Find a paragraph marker while ignoring inline cross-references.

    eCFR section text often contains references such as ``paragraph (b)(4)``
    before the actual ``(b)`` paragraph. For target-scope heuristics, prefer
    markers that look structural: top-level markers must begin a line; nested
    markers may also appear inline after punctuation, as in ``(2) Heading.
    (i) Text``.
    """
    pattern = _structural_parenthetical_marker_pattern(
        rf"\({re.escape(fragment)}\)",
        allow_inline=False,
    )
    match = pattern.search(source_text, start, end)
    if match or depth == 0:
        return match
    inline_pattern = _structural_parenthetical_marker_pattern(
        rf"\({re.escape(fragment)}\)",
        allow_inline=True,
    )
    return inline_pattern.search(source_text, start, end)


def _find_structural_sibling_marker(
    source_text: str,
    fragment: str,
    start: int,
    end: int,
    *,
    depth: int,
) -> re.Match[str] | None:
    marker = _structural_sibling_marker_fragment(fragment, depth=depth)
    if not marker:
        return None
    pattern = _structural_parenthetical_marker_pattern(
        marker,
        allow_inline=False,
    )
    match = pattern.search(source_text, start, end)
    if match or depth == 0:
        return match
    inline_pattern = _structural_parenthetical_marker_pattern(
        marker,
        allow_inline=True,
    )
    return inline_pattern.search(source_text, start, end)


def _structural_parenthetical_marker_pattern(
    marker_pattern: str,
    *,
    allow_inline: bool,
) -> re.Pattern[str]:
    boundary = r"(?P<prefix>\A|\n"
    if allow_inline:
        boundary += r"|[.;:]\s+"
    boundary += r")"
    return re.compile(
        rf"{boundary}\s*(?P<marker>{marker_pattern})\s+",
    )


def _structural_sibling_marker_fragment(fragment: str, *, depth: int) -> str | None:
    """Return a marker regex for the next same-level structural sibling."""
    if re.fullmatch(r"\d+(?:\.\d+)*", fragment):
        return _numeric_sibling_parenthetical_marker(fragment)
    if _is_roman_parenthetical_fragment(fragment, depth=depth):
        return r"\([ivxlcdm]+\)"
    if re.fullmatch(r"[a-z](?:\.\d+)*", fragment, flags=re.IGNORECASE):
        return _alpha_sibling_parenthetical_marker(fragment, top_level=depth == 0)
    return None


def _is_roman_parenthetical_fragment(fragment: str, *, depth: int) -> bool:
    """Treat nested lower-case roman markers as roman, not alpha letters."""
    return depth > 0 and re.fullmatch(r"[ivxlcdm]+", fragment) is not None


def _legal_fragments_from_rulespec_ref(target_ref_prefix: str) -> list[str]:
    """Return legal child fragments after the title/section prefix."""
    path = target_ref_prefix.split(":", 1)[-1]
    parts = [part for part in path.split("/") if part]
    if "statutes" in parts:
        idx = parts.index("statutes")
        if len(parts) > idx + 3:
            return parts[idx + 3 :]
    if "regulations" in parts:
        idx = parts.index("regulations")
        if len(parts) > idx + 4 and re.fullmatch(
            r"\d+-cfr", parts[idx + 1], flags=re.IGNORECASE
        ):
            return parts[idx + 4 :]
        if len(parts) > idx + 3:
            return parts[idx + 3 :]
    return []


def _sibling_marker_pattern(fragment: str) -> str | None:
    """Return a parenthetical marker pattern for the next same-level sibling."""
    if re.fullmatch(r"\d+(?:\.\d+)*", fragment):
        return _numeric_sibling_parenthetical_marker(fragment)
    if re.fullmatch(r"[a-z](?:\.\d+)*", fragment, flags=re.IGNORECASE):
        return _alpha_sibling_parenthetical_marker(fragment, top_level=False)
    return None


def _source_has_partial_extent_child_rewiring_limit(source_text: str) -> bool:
    """Return true for partial exemptions that cannot be rewired through children."""
    normalized = " ".join(source_text.split())
    for match in re.finditer(r"\bto\s+the\s+extent\b", normalized, re.IGNORECASE):
        window = normalized[max(0, match.start() - 180) : match.end() + 180]
        if re.search(
            r"\b(?:exempt|exemption|excluded|exclusion|not\s+subject|"
            r"subject\s+exclusively|taxes?\s+imposed)\b",
            window,
            re.IGNORECASE,
        ):
            return True
    return False


def _source_opens_amount_adjustment_list(source_text: str) -> bool:
    """Return true when the source opens an aggregate list of amount adjustments."""
    normalized = " ".join(source_text.split())
    return bool(
        re.search(
            r"\bthere\s+shall\s+be\s+"
            r"(?:added|subtracted|deducted)\s+"
            r"(?:to|from)\b",
            normalized[:500],
            re.IGNORECASE,
        )
    )


def _format_child_exception_import_guidance(
    source_text: str,
    context_files: list[EvalContextFile],
    *,
    target_ref_prefix: str | None,
) -> str:
    """Require parent exception-list slices to compose existing child exceptions."""
    if not target_ref_prefix:
        return ""
    if not _source_text_opens_exception_list(source_text):
        return ""

    child_prefix = f"{target_ref_prefix.rstrip('/')}/"
    child_exception_refs: list[str] = []
    for item in context_files:
        if not item.import_path.startswith(child_prefix):
            continue
        for export in _context_file_exception_exports(item.source_path):
            child_exception_refs.append(f"{item.import_path}#{export}")
    if not child_exception_refs:
        return ""

    refs = ", ".join(f"`{ref}`" for ref in child_exception_refs[:8])
    if len(child_exception_refs) > 8:
        refs += ", ..."
    return f"""
Parent exception-list child fragments detected:
- `./source.txt` opens an exception or exclusion list, and copied child-fragment
  files already export executable exception outputs ({refs}). Import each listed child exception output and negate it in the affected parent definition or
  composition. Do not leave the parent definition as only positive conditions,
  and do not replace the child exceptions with local `*_exception` inputs.
- This overrides the usual small-test-count preference. Include one positive
  companion case with no listed exception active and one blocking companion test
  for each listed child exception output, changing only that imported child
  module's underlying facts from the positive case and expecting the affected
  parent Judgment to be `not_holds`.
"""


def _format_branch_child_naming_guidance(
    context_files: list[EvalContextFile],
    *,
    target_file_name: str,
    target_ref_prefix: str | None,
) -> str:
    """Warn child-fragment encoders about reserved sibling export names."""
    target_key = _context_import_path_key(target_ref_prefix or target_file_name)
    if target_key is None:
        return ""
    target_parent = _context_import_parent_key(target_key)

    target_exports: set[str] = set()
    sibling_exports: dict[str, str] = {}
    sibling_count = 0
    for item in context_files:
        item_key = _context_import_path_key(item.import_path)
        if item_key is None or _context_import_parent_key(item_key) != target_parent:
            continue
        exports = _context_file_exports(item.source_path)
        if not exports:
            continue
        if item_key == target_key:
            target_exports.update(exports)
            continue
        sibling_count += 1
        for name in exports:
            sibling_exports.setdefault(name, item.import_path)

    if not sibling_exports or sibling_count == 0:
        return ""

    reserved = _format_rule_name_list(sorted(sibling_exports))
    colliding_exports = target_exports & sibling_exports.keys()
    colliding = _format_rule_name_list(sorted(colliding_exports))
    collision_note = ""
    if colliding_exports:
        collision_note = (
            "\n- The copied target currently exports invalid colliding names: "
            f"{colliding}; do not preserve those names."
        )

    return f"""
Sibling export naming for this target:
- Copied sibling files already reserve these exported names: {reserved}.
{collision_note}
- Do not export any local rule with a copied sibling's name. If a suggested
  generic relation name is already reserved, use a source-specific semantic
  name rather than the sibling's generic relation name.
- If this target is a child branch and the source states a shared parent
  consequence such as "shall be exempt if (A) ...", define the condition in this
  branch, not the shared parent consequence. Use a concise semantic output
  name based on this branch's condition.
- If the copied target exports a name that mainly describes the shared parent outcome rather than this branch's source-stated condition, treat that name as stale and rename it even when no sibling currently collides with it.
"""


def _format_cited_context_import_guidance(
    source_text: str,
    context_files: list[EvalContextFile],
) -> str:
    """Highlight copied context files directly cited by the source text."""
    lines: list[str] = []
    for item in context_files:
        citation = _import_target_to_statute_citation(item.import_path)
        if citation is None:
            continue
        if not _source_text_cites_statute(source_text, citation):
            continue
        exports = _context_file_exports(item.source_path)
        if not exports:
            continue
        references = ", ".join(f"`{item.import_path}#{name}`" for name in exports[:8])
        if len(exports) > 8:
            references += ", ..."
        preferred_exports: list[str] = []
        preferred_note = ""
        if _source_text_citation_is_displacement_reference(source_text, citation):
            preferred_note = (
                " This citation appears in a displacement or replacement phrase; "
                "use the cited file only as context for what is being displaced. "
                "Do not import the cited final amount solely because it is named."
            )
        else:
            preferred_exports = _preferred_exports_for_cited_reference(
                source_text,
                citation,
                exports,
            )
        if preferred_exports:
            preferred = ", ".join(
                f"`{item.import_path}#{name}`" for name in preferred_exports
            )
            preferred_note = (
                " For the cited deduction/exemption/credit reference, prefer "
                f"the final imported output {preferred} over any local "
                "`*_provided_in_section_*`, `*_under_section_*`, or similar "
                "placeholder."
            )
        lines.append(
            f"- Source cites `{citation.label}`; copied context target "
            f"`{item.import_path}` exports {references}.{preferred_note}"
        )
    if not lines:
        return ""
    return f"""
Mandatory cited RuleSpec imports detected from source text:
{chr(10).join(lines)}
When this source computes by reference to one of these cited targets, add the
appropriate listed `imports:` entry and use the imported bare rule name in the
formula. Do not keep a local `_under_section_...` or `_under_subsection_...`
input, or a local `_provided_in_section_...`, `_allowed_under_section_...`,
`_deduction_under_section_...`, or `_credit_allowed_under_section_...` input,
for an already copied RuleSpec context target.
If a cited target appears in a displacement phrase such as `in lieu of`,
`instead of`, or `in place of`, do not import the cited target's final amount
solely because it is cited; encode the current source's replacement amount or
rate from the current source text.
"""


def _format_excluded_child_context_guidance(
    source_text: str,
    context_files: list[EvalContextFile],
) -> str:
    """Guide ancestor/child import choice when the source excludes a child branch."""
    if not re.search(
        r"\b(?:other\s+than|except(?:ing)?|excluding)\b",
        source_text,
        flags=re.IGNORECASE,
    ):
        return ""

    contexts: list[tuple[EvalContextFile, _StatuteCitation, list[str]]] = []
    for item in context_files:
        if item.kind.endswith("_test_context"):
            continue
        citation = _import_target_to_statute_citation(item.import_path)
        if citation is None or not _source_text_cites_statute(source_text, citation):
            continue
        exports = _context_file_exports(item.source_path)
        if not exports:
            continue
        contexts.append((item, citation, exports))

    parent_by_section = {
        citation.section: (item, exports)
        for item, citation, exports in contexts
        if not citation.fragments
    }
    child_by_section: dict[
        str, list[tuple[EvalContextFile, _StatuteCitation, list[str]]]
    ] = defaultdict(list)
    for item, citation, exports in contexts:
        if citation.fragments:
            child_by_section[citation.section].append((item, citation, exports))

    lines: list[str] = []
    for excluded in _excluded_child_citations(source_text):
        parent = parent_by_section.get(excluded.section)
        children = child_by_section.get(excluded.section, [])
        if parent is None or not children:
            continue
        included_children = [
            (item, citation, exports)
            for item, citation, exports in children
            if not _citation_is_same_or_descendant(citation, excluded)
        ]
        if not included_children:
            continue

        parent_item, _parent_exports = parent
        child_summary = "; ".join(
            f"`{item.import_path}` exports "
            + ", ".join(f"`{item.import_path}#{name}`" for name in exports[:4])
            for item, _citation, exports in included_children[:4]
        )
        lines.append(
            f"- Source cites ancestor section `{excluded.section}` but excludes "
            f"`{excluded.label}`. Copied context includes ancestor "
            f"`{parent_item.import_path}` and more specific included child "
            f"targets: {child_summary}."
        )

    if not lines:
        return ""

    return f"""
Excluded cited child branch guidance:
{chr(10).join(lines)}
For an amount that excludes a child branch, do not import an ancestor aggregate
output from the cited section when more specific copied child outputs are
available for the included branches. Import the exact included child outputs and
compose them directly; omit the excluded child branch unless the current source
adds it back.
"""


def _excluded_child_citations(source_text: str) -> list[_StatuteCitation]:
    """Return child citations that appear in an exclusion phrase."""
    excluded: list[_StatuteCitation] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for cited_parts, start, end in _iter_cited_usc_sections(source_text):
        if not cited_parts.fragments:
            continue
        window_start = max(0, start - 120)
        window_end = min(len(source_text), end + 40)
        window = source_text[window_start:window_end]
        if not re.search(
            r"\b(?:other\s+than|except(?:ing)?|excluding)\b",
            window,
            flags=re.IGNORECASE,
        ):
            continue
        key = (cited_parts.section, cited_parts.fragments)
        if key in seen:
            continue
        seen.add(key)
        label = cited_parts.section + "".join(
            f"({fragment})" for fragment in cited_parts.fragments
        )
        excluded.append(
            _StatuteCitation(
                label=label,
                section=cited_parts.section,
                fragments=cited_parts.fragments,
            )
        )
    return excluded


def _citation_is_same_or_descendant(
    citation: _StatuteCitation,
    ancestor: _StatuteCitation,
) -> bool:
    """Return whether citation is the excluded citation or nested under it."""
    return (
        citation.section == ancestor.section
        and citation.fragments[: len(ancestor.fragments)] == ancestor.fragments
    )


def _format_unavailable_cited_context_guidance(
    source_text: str,
    context_files: list[EvalContextFile],
) -> str:
    """Warn when a cited context file is present but unavailable as an executable dependency."""
    lines: list[str] = []
    seen: set[str] = set()
    for item in context_files:
        citation = _import_target_to_statute_citation(item.import_path)
        if citation is None or not _source_text_cites_statute(source_text, citation):
            continue
        exports = _context_file_exports(item.source_path)
        reason = _context_file_unavailable_reason(item.source_path, exports)
        if reason is None:
            continue
        if item.import_path in seen:
            continue
        seen.add(item.import_path)
        suffixes = _citation_example_suffixes(item.import_path)
        suffix_list = ", ".join(f"`{suffix}`" for suffix in suffixes)
        example_suffix = suffixes[-1]
        lines.append(
            f"- Source cites `{citation.label}`; copied context target "
            f"`{item.import_path}` has {reason}. For this cited target, do not "
            f"create local cross-reference facts using suffixes {suffix_list}. "
            f"That includes `_under_section_{example_suffix}`, "
            f"`section_{example_suffix}_...`, "
            f"`*_provided_in_section_{example_suffix}`, "
            f"`*_to_which_section_{example_suffix}_applies`, or similar facts."
        )
    if not lines:
        return ""
    return f"""
Unavailable cited RuleSpec context detected:
{chr(10).join(lines)}
These copied targets are present but not executable dependencies. If an output
would depend on one of them, omit or defer only the affected executable surface
and keep encoding independent rules from this source. Do not synthesize local
facts to stand in for unavailable cited definitions, exceptions, exclusions, or
computed amounts.
"""


def _format_missing_cited_source_guidance(
    citation: str,
    source_text: str,
    context_files: list[EvalContextFile],
) -> str:
    """Warn when exception-like logic cites upstream RuleSpec not in context."""
    if not _source_text_has_cross_reference_dependency(source_text):
        return ""

    missing_targets = _missing_cited_statute_targets(
        citation,
        source_text,
        context_files,
    )
    if not missing_targets:
        return ""

    lines = [
        f"- Source cites `{label}`, but no copied context provides `{target}`."
        for label, target in missing_targets
    ]
    example_suffix = _citation_example_suffix(missing_targets[0][1])
    return f"""
Missing cited RuleSpec sources detected:
{chr(10).join(lines)}
For definition, same-meaning, treated-as, rules-similar, exception, exclusion,
`unless`, `notwithstanding`, shall-not-apply, not-treated-as,
carryback/carryover, or special-rule logic, these citations are upstream legal
dependencies. Do not create local facts such as
`section_{example_suffix}...`, `transaction_to_which_section_{example_suffix}_applies`,
`*_under_section_{example_suffix}`, or `*_provided_in_section_{example_suffix}`.
If an executable output would depend on any missing target above, emit
`module.status: deferred` or `module.status: entity_not_supported` for that
surface, or add a `module.deferred_outputs[]` record for that output when the
module has independent executable rules. Preserve the source text in
`module.summary`, use absolute `output` and `blocked_by` targets in deferred
records, and leave any tests for that deferred surface empty. Do not use
top-level `module.status` merely because some other branch in the same source
has a missing citation. If an independent output can be encoded using available
imports or source-grounded local facts, encode it and omit or defer only the
blocked surface. If a copied child output
already covers the subsection containing the missing citation, import that child
output instead of treating the child's internal missing citation as a blocker for
the parent composition. Encode the upstream cited source first, then retry the
blocked surface.
"""


def _source_text_has_cross_reference_dependency(source_text: str) -> bool:
    """Return whether text has dependency phrasing that should not be stubbed."""
    return bool(
        re.search(
            r"\b(?:except|unless|notwithstanding)\b"
            r"|shall\s+not\s+apply"
            r"|not\s+be\s+treated"
            r"|carrybacks?\s+and\s+carryovers?\s+under\s+section"
            r"|tax\s+imposed\s+by\s+section",
            source_text,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\bsame\s+meaning\b.*\bsection\s+[0-9]"
            r"|\btreated\s+as\b.*\bunder\s+section\s+[0-9]"
            r"|\brules\s+similar\s+to\b.*\bsection\s+[0-9]"
            r"|\bin\s+accordance\s+with\s+section\s+[0-9]",
            source_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )


def _missing_cited_statute_targets(
    citation: str,
    source_text: str,
    context_files: list[EvalContextFile],
) -> list[tuple[str, str]]:
    """Return cited statute targets that are not available as copied context."""
    current_section = _section_for_eval_identifier(citation)
    if current_section is None:
        return []
    import_prefix = _import_prefix_for_eval_identifier(citation)

    available = {
        _normalize_prompt_import_target(item.import_path) for item in context_files
    }
    missing: list[tuple[str, str]] = []
    seen: set[str] = set()
    for cited_parts, _start, _end in _iter_cited_usc_sections(source_text):
        section = cited_parts.section
        if section == current_section:
            continue
        candidate_paths = _cited_context_candidate_paths_for_eval_identifier(
            citation,
            cited_parts,
        )
        if not candidate_paths:
            continue
        candidate_targets = [
            _relative_rulespec_path_to_import_target(path, prefix=import_prefix)
            for path in candidate_paths
        ]
        normalized_candidates = [
            _normalize_prompt_import_target(target) for target in candidate_targets
        ]
        if any(
            _prompt_import_covers(available_target, normalized)
            for available_target in available
            for normalized in normalized_candidates
        ):
            continue
        target = candidate_targets[0]
        normalized = normalized_candidates[0]
        if normalized in seen:
            continue
        seen.add(normalized)
        label = (
            "section "
            + section
            + "".join(f"({fragment})" for fragment in cited_parts.fragments)
        )
        missing.append((label, target))
    return missing


def _import_prefix_for_eval_identifier(citation: str) -> str | None:
    """Return the canonical import prefix implied by an eval identifier."""
    normalized = citation.strip().strip("/")
    first_part = normalized.split("/", 1)[0]
    if ":" in first_part:
        prefix, _rest = first_part.split(":", 1)
        return prefix or None
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 3 and parts[0] not in _RULESPEC_SOURCE_ROOT_TOKENS:
        return parts[0]
    try:
        parse_usc_citation(citation)
    except Exception:
        return None
    return "us"


def _normalize_prompt_import_target(import_target: str) -> str:
    """Normalize an import target for prompt-context availability checks."""
    normalized = import_target.strip().strip("\"'")
    normalized = normalized.split("#", 1)[0]
    if ":" in normalized:
        _, normalized = normalized.split(":", 1)
    return normalized.strip("/")


def _prompt_import_covers(available: str, expected: str) -> bool:
    """Return whether an available prompt import covers an expected target."""
    if (
        available == expected
        or available.startswith(expected + "/")
        or expected.startswith(available + "/")
    ):
        return True
    available_parts = available.split("/")
    expected_parts = expected.split("/")
    return (
        len(available_parts) == len(expected_parts)
        and available_parts[:-1] == expected_parts[:-1]
        and expected_parts[-1].startswith(available_parts[-1] + ".")
    )


def _citation_match_points_to_other_act(source_text: str, match_end: int) -> bool:
    """Heuristically skip `section X of the Other Act` references."""
    following = source_text[match_end : match_end + 80]
    return bool(
        re.match(
            r"\s+of\s+(?!this\s+(?:section|title)\b)(?!the\s+Internal\s+Revenue\s+Code\b)",
            following,
            flags=re.IGNORECASE,
        )
    )


def _citation_example_suffix(import_target: str) -> str:
    """Return identifier-style suffix for a cited import target."""
    normalized = _normalize_prompt_import_target(import_target)
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 3 and parts[0] == "statutes":
        return "_".join(parts[2:])
    return "_".join(parts[-2:]) if len(parts) >= 2 else "cited"


def _citation_example_suffixes(import_target: str) -> list[str]:
    """Return exact and ancestor identifier suffixes for a cited import target."""
    normalized = _normalize_prompt_import_target(import_target)
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 3 or parts[0] != "statutes":
        return [_citation_example_suffix(import_target)]

    citation_parts = parts[2:]
    suffixes: list[str] = []
    # A bare section ancestor is usually too broad for prompt guidance, but
    # subsection ancestors are exactly how models phrase local placeholders.
    minimum_length = 2 if len(citation_parts) > 1 else 1
    for length in range(len(citation_parts), minimum_length - 1, -1):
        suffix = "_".join(citation_parts[:length])
        if suffix not in suffixes:
            suffixes.append(suffix)
    return suffixes or [_citation_example_suffix(import_target)]


def _format_proration_test_guidance(source_text: str) -> str:
    """Return source-specific guidance for exact proration companion tests."""
    if not re.search(
        r"\bfraction\b.*\bdenominator\b.*\b[0-9]",
        source_text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return ""
    return """
Source-specific proration test guidance:
- For proration tests with a source-stated denominator, choose input amounts
  divisible by that denominator so expected outputs are exact decimals, not
  rounded approximations. For denominator 365, prefer values like 36500 with
  182 days, so `36500 * 182 / 365 = 18200`, rather than 100000 with 182 days.
"""


@dataclass(frozen=True)
class _StatuteCitation:
    label: str
    section: str
    fragments: tuple[str, ...]


def _import_target_to_statute_citation(import_path: str) -> _StatuteCitation | None:
    """Infer a statute section citation from a RuleSpec import target."""
    normalized = import_path.strip().strip("\"'")
    normalized = normalized.split("#", 1)[0].strip().strip("/")
    if ":" in normalized:
        maybe_prefix, rest = normalized.split(":", 1)
        if re.fullmatch(r"[a-z][a-z0-9-]*", maybe_prefix) and rest:
            normalized = rest.strip("/")
    if normalized.endswith((".yaml", ".yml")):
        normalized = normalized.rsplit(".", 1)[0]
    parts = [part for part in normalized.split("/") if part]
    try:
        statutes_index = parts.index("statutes")
    except ValueError:
        statutes_index = None
    if statutes_index is not None:
        if len(parts) <= statutes_index + 2:
            return None
        section = parts[statutes_index + 2]
        fragments = tuple(parts[statutes_index + 3 :])
    else:
        try:
            regulations_index = parts.index("regulations")
        except ValueError:
            return None
        if len(parts) <= regulations_index + 2:
            return None
        regulation_root = parts[regulations_index + 1]
        if re.fullmatch(r"\d+-ccr-\d+-\d+", regulation_root, flags=re.IGNORECASE):
            section = parts[regulations_index + 2]
            fragments = tuple(parts[regulations_index + 3 :])
        elif (
            re.fullmatch(r"\d+-cfr", regulation_root, flags=re.IGNORECASE)
            and len(parts) > regulations_index + 3
        ):
            section = f"{parts[regulations_index + 2]}.{parts[regulations_index + 3]}"
            fragments = tuple(parts[regulations_index + 4 :])
        else:
            return None
    if not re.fullmatch(r"[0-9][A-Za-z0-9.-]*", section):
        return None
    if not all(_is_citation_fragment(fragment) for fragment in fragments):
        return None
    label = section + "".join(f"({fragment})" for fragment in fragments)
    return _StatuteCitation(label=label, section=section, fragments=fragments)


def _source_text_cites_statute(
    source_text: str,
    citation: _StatuteCitation,
) -> bool:
    """Return whether source text cites the statute section or fragment."""
    citation_pattern = re.escape(citation.section)
    for fragment in citation.fragments:
        citation_pattern += rf"\s*\(\s*{re.escape(fragment)}\s*\)"
    if re.search(
        rf"\bsection\s+{citation_pattern}(?:\b|\s|\)|,|;|\.)",
        source_text,
        flags=re.IGNORECASE,
    ):
        return True
    if citation.fragments:
        if re.search(
            rf"\b{citation_pattern}(?:\b|\s|\)|,|;|\.)",
            source_text,
            flags=re.IGNORECASE,
        ):
            return True
    for cited_parts, _start, _end in _iter_cited_usc_sections(source_text):
        if _cited_parts_cover_import(cited_parts, citation):
            return True
    return False


def _cited_parts_cover_import(
    cited_parts: CitationParts,
    citation: _StatuteCitation,
) -> bool:
    """Return whether a source citation covers a copied import target."""
    if cited_parts.section != citation.section:
        return False
    if cited_parts.fragments == citation.fragments:
        return True
    if not cited_parts.fragments or not citation.fragments:
        return True
    return (
        citation.fragments[: len(cited_parts.fragments)] == cited_parts.fragments
        or cited_parts.fragments[: len(citation.fragments)] == citation.fragments
    )


def _preferred_exports_for_cited_reference(
    source_text: str,
    citation: _StatuteCitation,
    exports: list[str],
) -> list[str]:
    """Suggest likely final outputs for source references to cited deductions."""
    window = _source_text_citation_window(source_text, citation).lower()
    if not any(
        token in window
        for token in (
            "deduction",
            "deduct",
            "exemption",
            "credit",
            "allowance",
            "allowed",
            "allowable",
        )
    ):
        return []

    scored: list[tuple[tuple[int, int, int, str], str]] = []
    for export in exports:
        name = export.lower()
        score = 0
        if "exemption" in window and "exemption" in name:
            score += 50
        if "credit" in window and "credit" in name:
            score += 50
        if "deduction" in window and "deduction" in name:
            score += 50
        if "allowance" in window and "allowance" in name:
            score += 50
        if name.endswith(("_deduction", "_credit", "_allowance")):
            score += 35
        if name.startswith("section_"):
            score += 10
        if any(
            marker in name
            for marker in (
                "_before_",
                "_cap",
                "_eligible",
                "_eligibility",
                "_increment",
                "_phaseout",
                "_rate",
                "_threshold",
                "_base",
                "_amount_per",
                "_modified_adjusted",
            )
        ):
            score -= 30
        if score <= 0:
            continue
        scored.append(((-score, len(export), exports.index(export), export), export))
    return [export for _sort_key, export in sorted(scored)[:3]]


def _source_text_citation_is_displacement_reference(
    source_text: str,
    citation: _StatuteCitation,
) -> bool:
    """Return whether a cited provision is being displaced, not composed."""
    window = _source_text_citation_window(source_text, citation).lower()
    if not window:
        return False
    if not re.search(r"\b(?:in\s+lieu\s+of|instead\s+of|in\s+place\s+of)\b", window):
        return False
    return bool(
        re.search(
            r"\b(?:deduction|credit|allowance|exemption|rate|tax|amount)\b",
            window,
        )
    )


def _source_text_citation_window(
    source_text: str,
    citation: _StatuteCitation,
) -> str:
    citation_pattern = re.escape(citation.section)
    for fragment in citation.fragments:
        citation_pattern += rf"\s*\(\s*{re.escape(fragment)}\s*\)"
    match = re.search(
        rf"\bsection\s+{citation_pattern}(?:\b|\s|\)|,|;|\.)",
        source_text,
        flags=re.IGNORECASE,
    )
    if match is None and citation.fragments:
        match = re.search(
            rf"\b{citation_pattern}(?:\b|\s|\)|,|;|\.)",
            source_text,
            flags=re.IGNORECASE,
        )
    if match is None:
        for cited_parts, start, end in _iter_cited_usc_sections(source_text):
            if _cited_parts_cover_import(cited_parts, citation):
                start = max(0, start - 120)
                end = min(len(source_text), end + 80)
                return source_text[start:end]
        return ""
    start = max(0, match.start() - 120)
    end = min(len(source_text), match.end() + 80)
    return source_text[start:end]


def _is_citation_fragment(fragment: str) -> bool:
    """Return whether an import path fragment can represent a citation part."""
    return bool(re.fullmatch(r"[A-Za-z0-9]+", fragment))


def _context_import_path_key(import_path: str) -> tuple[str | None, str] | None:
    """Normalize an import target for sibling/target comparisons."""
    normalized = import_path.strip().strip("\"'")
    if not normalized:
        return None
    normalized = normalized.split("#", 1)[0].strip().strip("/")
    prefix: str | None = None
    if ":" in normalized:
        maybe_prefix, rest = normalized.split(":", 1)
        if re.fullmatch(r"[a-z][a-z0-9-]*", maybe_prefix) and rest:
            prefix = maybe_prefix
            normalized = rest.strip("/")
    if normalized.endswith((".yaml", ".yml")):
        normalized = normalized.rsplit(".", 1)[0]
    normalized = normalized.strip("/")
    return (prefix, normalized) if normalized else None


def _context_import_parent_key(
    key: tuple[str | None, str],
) -> tuple[str | None, str]:
    prefix, target = key
    return prefix, Path(target).parent.as_posix()


def _format_rule_name_list(names: list[str], *, limit: int = 12) -> str:
    """Format a compact backticked list of rule names."""
    visible = ", ".join(f"`{name}`" for name in names[:limit])
    if len(names) > limit:
        visible += ", ..."
    return visible or "`<none>`"


def _context_file_hash(source_path: str) -> str | None:
    """Return the sha256 hash for a context file when it is readable."""
    try:
        digest = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    except OSError:
        return None
    return f"sha256:{digest}"


def find_admin_agency_aggregate_entity_issues(
    content: str,
    source_text: str,
) -> list[str]:
    """Reject executable entity rules for State-agency/FNS aggregate measures."""
    is_admin_agency_aggregate = _has_admin_agency_aggregate_source_context(source_text)
    if not is_admin_agency_aggregate:
        return []
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    module = payload.get("module")
    if isinstance(module, dict):
        status = str(module.get("status") or "").strip().lower()
        if status in {"deferred", "entity_not_supported"}:
            return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    issues: list[str] = []
    standard_entities = {entity.lower(): entity for entity in SUPPORTED_EVAL_ENTITIES}
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        entity = str(rule.get("entity") or "").strip()
        if not entity:
            continue
        normalized_entity = entity.lower()
        if normalized_entity in ADMINISTRATIVE_EVAL_ENTITIES:
            continue
        if normalized_entity not in standard_entities:
            continue
        name = str(rule.get("name") or f"rules[{index}]").strip()
        issues.append(
            "Unsupported administrative aggregate entity: "
            f"`{name}` is declared on `{entity}`, but the authoritative source "
            "defines a State agency/FNS aggregate performance, sampling, "
            "liability, waiver, or bonus measure. Use a source-stated "
            "administrative entity such as `StateAgency` instead of a "
            "household/person/tax/payment entity, or defer only if the "
            "administrative surface still cannot be represented faithfully."
        )
    return issues


def _has_admin_agency_aggregate_source_context(source_text: str) -> bool:
    """Return true only when administrative subjects and measures are local."""
    if _ADMIN_AGENCY_BONUS_USE_RESTRICTION_PATTERN.search(source_text):
        return True

    subject_matches = list(
        _ADMIN_AGENCY_AGGREGATE_SUBJECT_PATTERN.finditer(source_text)
    )
    if not subject_matches:
        return False
    measure_matches = list(
        _ADMIN_AGENCY_AGGREGATE_MEASURE_PATTERN.finditer(source_text)
    )
    if not measure_matches:
        return False

    for subject_match in subject_matches:
        subject_center = (subject_match.start() + subject_match.end()) // 2
        for measure_match in measure_matches:
            measure_center = (measure_match.start() + measure_match.end()) // 2
            if (
                abs(subject_center - measure_center)
                <= _ADMIN_AGENCY_AGGREGATE_CONTEXT_WINDOW
            ):
                return True
    return False


def _context_file_export_detail(item: EvalContextFile) -> str:
    """Return a compact list of exported symbols for a context RuleSpec file."""
    exports = _context_file_exports(item.source_path)
    local_inputs = sorted(_context_file_local_inputs(item.source_path))
    if not exports and not local_inputs:
        return ""
    details: list[str] = []
    if exports:
        references = ", ".join(f"`{item.import_path}#{name}`" for name in exports[:8])
        if len(exports) > 8:
            references += ", ..."
        details.append(f"exports {references}")
    terminal_exports = _context_file_terminal_exports(item.source_path)
    if terminal_exports:
        terminal_references = ", ".join(
            f"`{item.import_path}#{name}`" for name in terminal_exports[:5]
        )
        if len(terminal_exports) > 5:
            terminal_references += ", ..."
        details.append(f"terminal exports {terminal_references}")
    if local_inputs:
        input_references = ", ".join(
            f"`{item.import_path}#input.{name}`" for name in local_inputs[:8]
        )
        if len(local_inputs) > 8:
            input_references += ", ..."
        details.append(f"local input slots {input_references}")
    return "; " + "; ".join(details)


def _context_file_exports(source_path: str) -> list[str]:
    """Extract RuleSpec rule names exported by a copied context file."""
    try:
        payload = yaml.safe_load(Path(source_path).read_text())
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []
    exports: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        name = str(rule.get("name") or "").strip()
        if name:
            exports.append(name)
    return exports


def _context_file_unavailable_reason(
    source_path: str,
    exports: list[str],
) -> str | None:
    """Return a prompt-friendly reason when a context file cannot be imported."""
    try:
        payload = yaml.safe_load(Path(source_path).read_text())
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return "no readable RuleSpec payload"
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return "no RuleSpec payload"
    module = payload.get("module")
    status = ""
    if isinstance(module, dict):
        status = str(module.get("status") or "").strip()
    if status in {"deferred", "entity_not_supported"}:
        return f"`module.status: {status}` and no executable exports"
    rules = payload.get("rules")
    if rules == [] or not exports:
        return "no executable exports"
    return None


def _source_text_opens_exception_list(source_text: str) -> bool:
    """Return true when source text introduces a following exception list."""
    normalized = " ".join(source_text.split())
    return bool(
        re.search(
            r"\b(?:exceptions?|exclusions?)\b[^.:\n]{0,180}\b(?:following|below)\b\s*:?"
            r"|\b(?:such\s+term\s+)?shall\s+not\s+include\b[^.:\n]{0,220}\b(?:following|below)\b\s*:?"
            r"|\bnot\s+(?:be\s+)?treated\b[^.:\n]{0,220}\b(?:unless|following|below)\b\s*:?",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _context_file_exception_exports(source_path: str) -> list[str]:
    """Return terminal exports from a copied child file that encode exceptions."""
    try:
        payload = yaml.safe_load(Path(source_path).read_text())
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    terminal_exports = set(_context_file_terminal_exports(source_path))
    exports: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        name = str(rule.get("name") or "").strip()
        if not name or name not in terminal_exports:
            continue
        if _is_exception_like_export_name(name):
            exports.append(name)
    return exports


def _is_exception_like_export_name(name: str) -> bool:
    normalized = name.lower()
    return any(
        token in normalized
        for token in (
            "exception",
            "except",
            "exclusion",
            "excluded",
            "disqualified",
            "disqualifying",
            "carve_out",
            "not_treated",
        )
    )


def _rule_has_exception_proof(rule: dict[str, object]) -> bool:
    try:
        atoms = rule["metadata"]["proof"]["atoms"]  # type: ignore[index]
    except (KeyError, TypeError):
        return False
    if not isinstance(atoms, list):
        return False
    return any(
        isinstance(atom, dict)
        and str(atom.get("kind") or "").strip().lower() in {"exception", "exclusion"}
        for atom in atoms
    )


def _context_file_executable_surfaces(source_path: str) -> dict[str, dict[str, object]]:
    """Extract public executable surfaces from a copied context RuleSpec file."""
    try:
        payload = yaml.safe_load(Path(source_path).read_text())
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return {}
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return {}
    surfaces: dict[str, dict[str, object]] = {}
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        kind = str(rule.get("kind") or "").strip().lower()
        if kind not in {"parameter", "derived", "derived_relation", "data_relation"}:
            continue
        name = str(rule.get("name") or "").strip()
        if not name:
            continue
        versions = rule.get("versions")
        effective_dates: list[str] = []
        if isinstance(versions, list):
            for version in versions:
                if isinstance(version, dict) and "effective_from" in version:
                    effective_dates.append(
                        str(version.get("effective_from") or "").strip()
                    )
        derived_relation = rule.get("derived_relation")
        relation_entity = ""
        if kind == "derived_relation" and isinstance(derived_relation, dict):
            relation_entity = str(derived_relation.get("entity") or "").strip()
        surfaces[name] = {
            "kind": kind,
            "entity": relation_entity or str(rule.get("entity") or "").strip(),
            "dtype": str(rule.get("dtype") or "").strip(),
            "period": str(rule.get("period") or "").strip(),
            "unit": str(rule.get("unit") or "").strip(),
            "indexed_by": _context_surface_sequence(rule.get("indexed_by")),
            "effective_dates": tuple(effective_dates),
        }
    return surfaces


def _context_surface_sequence(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ()
    text = str(value).strip()
    return (text,) if text else ()


_CONTEXT_FORMULA_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_CONTEXT_TEMPORAL_VALUE_FACT_YEAR_PATTERN = re.compile(r"(?:^|_)(?:19|20)\d{2}(?:_|$)")
_CONTEXT_FORMULA_BUILTINS = {
    "and",
    "ceil",
    "else",
    "false",
    "floor",
    "if",
    "match",
    "max",
    "min",
    "not",
    "or",
    "true",
}


def _context_file_invalid_local_inputs(
    source_path: str,
    *,
    context_files: list[EvalContextFile] | None = None,
) -> dict[str, str]:
    """Return local input names in copied context that current rules reject."""
    invalid: dict[str, str] = {}
    for identifier in _context_file_local_inputs(source_path):
        reason = _invalid_local_input_reason(identifier)
        if not reason:
            reason = _encoded_cross_reference_local_input_reason(
                identifier,
                source_path=source_path,
                context_files=context_files or [],
            )
        if reason:
            invalid[identifier] = reason
    return dict(sorted(invalid.items()))


_CONTEXT_SEMANTIC_SECTION_INPUT = re.compile(
    r"(?:^|_)(?:"
    r"under"
    r"|provided_in"
    r"|provided_by"
    r"|allowed_under"
    r"|allowable_under"
    r"|allowed_by"
    r"|allowable_by"
    r"|excluded_under"
    r"|excludable_under"
    r"|deduction_under"
    r"|credit_allowed_under"
    r")_section_(?P<section>[0-9][A-Za-z0-9.-]*)"
    r"(?P<tail>(?:_[A-Za-z0-9]+)*)"
)


def _encoded_cross_reference_local_input_reason(
    identifier: str,
    *,
    source_path: str,
    context_files: list[EvalContextFile],
) -> str:
    """Return a reason when a copied input shadows available cited context."""
    target = _context_cross_reference_input_target(identifier, source_path)
    if not target:
        return ""
    for item in context_files:
        if item.kind.endswith("_test_context") or item.kind == "existing_target":
            continue
        available = _normalize_prompt_import_target(item.import_path)
        if _prompt_import_covers(available, target):
            return (
                "encoded cross-reference placeholder; a context file for "
                f"`{item.import_path}` is available, so do not preserve this "
                "local input unless the generated file imports an actual "
                "exported output for that same referenced legal concept"
            )
    return ""


def _context_cross_reference_input_target(
    identifier: str,
    source_path: str,
) -> str | None:
    """Infer a repo-relative cited target from a copied local input name."""
    match = _CONTEXT_SEMANTIC_SECTION_INPUT.search(identifier)
    if not match:
        return None
    title = _context_file_title(source_path)
    if not title:
        return None
    fragments: list[str] = []
    for fragment in (match.group("tail") or "").split("_"):
        if not fragment:
            continue
        if _context_cross_reference_path_fragment(fragment):
            fragments.append(fragment)
            continue
        break
    return "/".join(["statutes", title, match.group("section"), *fragments])


def _context_file_title(source_path: str) -> str | None:
    parts = Path(source_path).parts
    for index, part in enumerate(parts):
        if part == "statutes" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _context_cross_reference_path_fragment(fragment: str) -> bool:
    return bool(
        re.fullmatch(r"\d+[A-Za-z]?|[A-Za-z]|[ivxlcdm]+", fragment)
        and fragment.lower() not in {"and"}
    )


def _context_file_local_inputs(source_path: str) -> set[str]:
    """Return local factual input identifiers used by copied context formulas."""
    try:
        payload = yaml.safe_load(Path(source_path).read_text())
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return set()
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return set()
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return set()

    defined = {
        str(rule.get("name") or "").strip()
        for rule in rules
        if isinstance(rule, dict) and str(rule.get("name") or "").strip()
    }
    imported = _context_file_imported_symbols(payload)
    inputs: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        versions = rule.get("versions")
        if not isinstance(versions, list):
            continue
        for version in versions:
            if not isinstance(version, dict):
                continue
            formula = version.get("formula")
            if isinstance(formula, (int, float)) and not isinstance(formula, bool):
                formula_text = str(formula)
            elif isinstance(formula, str):
                formula_text = formula
            else:
                continue
            for identifier in _CONTEXT_FORMULA_IDENTIFIER.findall(formula_text):
                if (
                    identifier in defined
                    or identifier in imported
                    or identifier in _CONTEXT_FORMULA_BUILTINS
                ):
                    continue
                inputs.add(identifier)
    return inputs


def _context_file_imported_symbols(payload: dict[str, object]) -> set[str]:
    imports = payload.get("imports")
    if not isinstance(imports, list):
        return set()
    symbols: set[str] = set()
    for item in imports:
        if not isinstance(item, str) or "#" not in item:
            continue
        fragment = item.rsplit("#", 1)[1].strip()
        if fragment and "." not in fragment:
            symbols.add(fragment)
    return symbols


def _context_file_imports_target(source_path: str, target_ref_prefix: str) -> bool:
    try:
        payload = yaml.safe_load(Path(source_path).read_text())
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return False
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return False
    imports = payload.get("imports")
    if not isinstance(imports, list):
        return False
    target_prefix = f"{target_ref_prefix}#"
    return any(
        isinstance(item, str)
        and (item == target_ref_prefix or item.startswith(target_prefix))
        for item in imports
    )


def _invalid_local_input_reason(identifier: str) -> str:
    if identifier in {"filing_status", "tax_filing_status"}:
        return (
            "filing status is a derived legal classification; import an "
            "upstream filing-status output when available, or use source-stated "
            "non-status predicates such as whether a joint or separate return "
            "was actually made"
        )
    if identifier.startswith("taxable_year") and (
        _CONTEXT_TEMPORAL_VALUE_FACT_YEAR_PATTERN.search(identifier) is not None
    ):
        return (
            "date/year-valued temporal fact; rename consistently to a semantic "
            "date-window predicate such as "
            "`taxable_year_begins_after_termination_date`; never use "
            "`post_YYYY`, `pre_YYYY`, or any four-digit year"
        )
    return ""


def _context_file_terminal_exports(source_path: str) -> list[str]:
    """Return executable exports not consumed by another local executable rule."""
    try:
        payload = yaml.safe_load(Path(source_path).read_text())
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    exports: list[str] = []
    formulas: list[tuple[str, str]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("kind") or "").strip().lower() not in {
            "parameter",
            "derived",
        }:
            continue
        name = str(rule.get("name") or "").strip()
        if not name:
            continue
        exports.append(name)
        versions = rule.get("versions")
        if not isinstance(versions, list):
            continue
        for version in versions:
            if not isinstance(version, dict):
                continue
            formula = version.get("formula")
            if isinstance(formula, (int, float)) and not isinstance(formula, bool):
                formulas.append((name, str(formula)))
            elif isinstance(formula, str) and formula.strip():
                formulas.append((name, formula))
    if not exports:
        return []

    export_names = set(exports)
    referenced: set[str] = set()
    for owner, formula in formulas:
        referenced.update(
            identifier
            for identifier in _CONTEXT_FORMULA_IDENTIFIER.findall(formula)
            if identifier in export_names
            and identifier != owner
            and identifier not in _CONTEXT_FORMULA_BUILTINS
        )
    terminal = [name for name in exports if name not in referenced]
    return terminal or exports


def _collect_scaffold_dates(
    workspace: EvalWorkspace,
    context_files: list[EvalContextFile],
) -> list[str]:
    """Collect usable temporal scaffold dates from copied precedent files."""
    dates: set[str] = set()
    for item in context_files:
        path = workspace.root / item.workspace_path
        try:
            text = path.read_text()
        except OSError:
            continue
        dates.update(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text))
    return sorted(dates)


def _expand_context_files(
    selected_paths: list[Path],
    policy_root: Path,
    target_rel: Path | None,
) -> list[tuple[Path, str]]:
    """Expand selected precedent files with their transitive canonical imports."""
    expanded: list[tuple[Path, str]] = []
    pending: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    for path in selected_paths:
        is_target = (
            target_rel is not None
            and _relative_to_root(path, policy_root) == target_rel
        )
        kind = (
            "existing_target"
            if is_target
            else (
                "implementation_precedent"
                if _is_under_root(path, policy_root)
                else "implementation_external"
            )
        )
        pending.append((path, kind))

    while pending:
        source_path, kind = pending.pop(0)
        resolved = source_path.resolve()
        if resolved in seen:
            continue
        if (
            target_rel is not None
            and _relative_to_root(source_path, policy_root) == target_rel
            and kind != "existing_target"
        ):
            continue
        seen.add(resolved)
        expanded.append((source_path, kind))
        if source_path.suffix in {".yaml", ".yml"} and not source_path.name.endswith(
            ".test.yaml"
        ):
            test_path = _rulespec_test_path(source_path)
            resolved_test = test_path.resolve()
            if test_path.exists() and resolved_test not in seen:
                seen.add(resolved_test)
                test_kind = (
                    "existing_target_test_context"
                    if kind == "existing_target"
                    else "implementation_test_context"
                )
                expanded.append((test_path, test_kind))

        if not _is_under_root(source_path, policy_root):
            continue

        for dependency in _resolve_context_imports(source_path, policy_root):
            if dependency.resolve() in seen:
                continue
            pending.append((dependency, "implementation_dependency"))

    return expanded


def _resolve_context_imports(source_path: Path, policy_root: Path) -> list[Path]:
    """Resolve canonical import targets for one copied precedent file."""
    dependencies: list[Path] = []
    for import_target in _extract_import_targets(source_path.read_text()):
        import_prefix = _import_target_prefix(import_target)
        target_path = _import_target_to_path(import_target)
        candidates = [policy_root / target_path]
        if import_prefix:
            candidates.append(
                policy_root.parent / f"rulespec-{import_prefix}" / target_path
            )
        if target_path.parts:
            first = target_path.parts[0]
            if first == policy_root.name:
                candidates.append(policy_root / Path(*target_path.parts[1:]))
            if first in _LOCAL_IMPORT_ROOT_TOKENS:
                candidates.append(policy_root.parent / target_path)

        for candidate in candidates:
            if candidate.exists():
                dependencies.append(candidate)
                break
    return dependencies


def _extract_import_targets(content: str) -> list[str]:
    """Extract file-level import targets from RuleSpec imports blocks."""
    targets: list[str] = []
    in_imports = False
    imports_indent = 0

    for line in content.splitlines():
        imports_match = re.match(r"^(\s*)imports:\s*$", line)
        if imports_match:
            in_imports = True
            imports_indent = len(imports_match.group(1))
            continue

        if not in_imports:
            continue

        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip())
        if indent <= imports_indent:
            in_imports = False
            continue

        item_match = IMPORT_ITEM_PATTERN.match(line)
        if not item_match:
            continue

        item = item_match.group(2).strip()
        import_target = item.split("#", 1)[0].strip()
        if import_target:
            targets.append(import_target)

    return targets


def _import_target_to_path(import_target: str) -> Path:
    """Convert an import target like 26/24/c#name into 26/24/c.yaml."""
    normalized = import_target.strip().strip('"').strip("'")
    normalized = normalized.split("#", 1)[0].strip()
    if ":" in normalized:
        prefix, rest = normalized.split(":", 1)
        if re.fullmatch(r"[a-z][a-z0-9-]*", prefix) and rest:
            normalized = rest
    if normalized.endswith((".yaml", ".yml")):
        return Path(normalized)
    return Path(f"{normalized}.yaml")


def _import_target_prefix(import_target: str) -> str | None:
    """Return the repo prefix from an absolute RuleSpec import target."""
    normalized = import_target.strip().strip('"').strip("'")
    normalized = normalized.split("#", 1)[0].strip()
    if ":" not in normalized:
        return None
    prefix, rest = normalized.split(":", 1)
    if re.fullmatch(r"[a-z][a-z0-9-]*", prefix) and rest:
        return prefix
    return None


def _is_under_root(path: Path, root: Path) -> bool:
    """Return True when a path is within the given root."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _relative_to_root(path: Path, root: Path) -> Path | None:
    """Return the path relative to the root when possible."""
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return None


def _hydrate_eval_root(eval_root: Path, workspace: EvalWorkspace) -> None:
    """Copy allowed precedent files into the eval root so imports resolve."""
    for item in workspace.context_files:
        workspace_path = Path(item.workspace_path)
        if not workspace_path.parts or workspace_path.parts[0] != "context":
            continue

        target_relative = _import_target_to_path(item.import_path)
        target = eval_root / target_relative
        if target.exists():
            continue

        source = workspace.root / workspace_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _run_prompt_eval(
    runner: EvalRunnerSpec,
    workspace: EvalWorkspace,
    prompt: str,
) -> EvalPromptResponse:
    """Run one prompt-only eval through the selected local CLI."""
    if runner.backend == "claude":
        return _run_claude_prompt_eval(runner, workspace, prompt)
    if runner.backend == "codex":
        return _run_codex_prompt_eval(runner, workspace, prompt)
    if runner.backend == "openai":
        return _run_openai_prompt_eval(runner, workspace, prompt)
    raise ValueError(f"Unsupported backend: {runner.backend}")


def _run_claude_prompt_eval(
    runner: EvalRunnerSpec,
    workspace: EvalWorkspace,
    prompt: str,
) -> EvalPromptResponse:
    """Run prompt-only eval via Claude CLI."""
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        runner.model,
        "-p",
        prompt,
    ]

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=workspace.root,
        timeout=600,
    )
    duration_ms = int((time.time() - start) * 1000)

    trace: dict = {}
    text = result.stdout + result.stderr
    tokens = None
    actual_cost = None
    error = None

    try:
        payload = json.loads(text)
        trace = {
            "provider": "anthropic",
            "backend": "claude-print",
            "model": runner.model,
            "json_result": payload,
        }
        usage = payload.get("usage", {}) or {}
        tokens = TokenUsage(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            cache_creation_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
        )
        actual_cost = payload.get("total_cost_usd")
        text = payload.get("result", "") or ""
        if payload.get("is_error"):
            error = text or "Claude eval returned an error"
    except json.JSONDecodeError:
        trace = {
            "provider": "anthropic",
            "backend": "claude-print",
            "model": runner.model,
            "raw_output": result.stdout + result.stderr,
        }
        if result.returncode != 0:
            error = (result.stdout + result.stderr).strip() or "Claude eval failed"

    if result.returncode != 0 and not error:
        error = (result.stdout + result.stderr).strip() or "Claude eval failed"

    return EvalPromptResponse(
        text=text,
        duration_ms=duration_ms,
        tokens=tokens,
        estimated_cost_usd=estimate_usage_cost_usd(runner.model, tokens),
        actual_cost_usd=actual_cost,
        trace=trace,
        error=error,
    )


def _run_codex_prompt_eval(
    runner: EvalRunnerSpec,
    workspace: EvalWorkspace,
    prompt: str,
) -> EvalPromptResponse:
    """Run prompt-only eval via Codex CLI."""
    codex_timeout_seconds, codex_idle_timeout_seconds = _codex_prompt_timeouts(
        workspace
    )
    last_message_file = workspace.root / ".codex-last-message.txt"
    if last_message_file.exists():
        last_message_file.unlink()

    cmd = [
        resolve_codex_cli(),
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "-o",
        str(last_message_file),
        "-m",
        runner.model,
        "-c",
        'reasoning_effort="low"',
        "-C",
        str(workspace.root),
        "-s",
        "read-only",
        prompt,
    ]

    start = time.time()
    terminated_after_output = False
    timed_out = False
    timeout_reason = None
    with (
        tempfile.TemporaryDirectory(prefix="axiom-codex-home-") as codex_home_dir,
        tempfile.NamedTemporaryFile(mode="w+", delete=False) as stdout_file,
        tempfile.NamedTemporaryFile(mode="w+", delete=False) as stderr_file,
    ):
        codex_home = _prepare_codex_eval_home(Path(codex_home_dir))
        codex_env = os.environ.copy()
        codex_env["CODEX_HOME"] = str(codex_home)
        stdout_path = Path(stdout_file.name)
        stderr_path = Path(stderr_file.name)
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            cwd=workspace.root,
            env=codex_env,
        )
        try:
            terminated_after_output = _wait_for_codex_process(
                process,
                last_message_file=last_message_file,
                timeout=codex_timeout_seconds,
                heartbeat_paths=[stdout_path, stderr_path],
                max_idle_seconds=codex_idle_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            timeout_reason = (
                "idle" if exc.timeout == codex_idle_timeout_seconds else "wall"
            )
            process.kill()
            process.wait()

    stdout_text = stdout_path.read_text()
    stderr_text = stderr_path.read_text()
    stdout_path.unlink(missing_ok=True)
    stderr_path.unlink(missing_ok=True)
    duration_ms = int((time.time() - start) * 1000)

    events: list[dict] = []
    assistant_messages: list[str] = []
    usage_payload: dict | None = None
    unexpected_accesses: list[str] = []
    error = None

    for line in (stdout_text + stderr_text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        events.append(payload)
        if payload.get("type") == "item.completed":
            item = payload.get("item", {}) or {}
            if item.get("type") == "agent_message" and item.get("text"):
                assistant_messages.append(item["text"])
            if item.get("type") == "command_execution":
                command = item.get("command", "")
                if _command_uses_policyengine_skill(command):
                    unexpected_accesses.append(command)
                    if not error:
                        error = (
                            "Codex eval attempted to use PolicyEngine skills, "
                            "which are disallowed for Axiom encoding"
                        )
                if _command_looks_out_of_bounds(command, workspace.root):
                    unexpected_accesses.append(command)
        elif payload.get("type") == "turn.completed":
            usage_payload = payload.get("usage") or {}
        elif payload.get("type") == "error":
            error = payload.get("message") or "Codex eval error"

    tokens = None
    if usage_payload is not None:
        tokens = TokenUsage(
            input_tokens=int(usage_payload.get("input_tokens", 0) or 0),
            output_tokens=int(usage_payload.get("output_tokens", 0) or 0),
            cache_read_tokens=int(usage_payload.get("cached_input_tokens", 0) or 0),
            reasoning_output_tokens=extract_reasoning_output_tokens(
                {
                    "provider": "openai",
                    "events": events,
                }
            ),
        )

    final_text = "\n".join(assistant_messages).strip()
    if last_message_file.exists():
        file_text = last_message_file.read_text().strip()
        if file_text:
            final_text = file_text

    if timed_out and not error and not final_text:
        error = "Codex eval timed out"

    if (
        process.returncode != 0
        and not error
        and not ((terminated_after_output and final_text) or (timed_out and final_text))
    ):
        error = (stdout_text + stderr_text).strip() or "Codex eval failed"

    return EvalPromptResponse(
        text=final_text,
        duration_ms=duration_ms,
        tokens=tokens,
        estimated_cost_usd=estimate_usage_cost_usd(runner.model, tokens),
        trace={
            "provider": "openai",
            "backend": "codex-exec",
            "model": runner.model,
            "timed_out": timed_out,
            "timeout_reason": timeout_reason,
            "timeout_seconds": codex_timeout_seconds,
            "idle_timeout_seconds": codex_idle_timeout_seconds,
            "events": events,
        },
        unexpected_accesses=unexpected_accesses,
        error=error,
    )


def _prepare_codex_eval_home(codex_home: Path) -> Path:
    """Create a minimal CODEX_HOME for eval subprocesses without user skills."""
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "skills").mkdir(exist_ok=True)
    source_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    for filename in ("auth.json", "installation_id"):
        source = source_home / filename
        target = codex_home / filename
        if target.exists() or target.is_symlink() or not source.exists():
            continue
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)
    return codex_home


def _codex_prompt_timeouts(workspace: EvalWorkspace) -> tuple[int, int]:
    """Return wall and idle timeouts for a Codex eval workspace."""
    try:
        source_length = len(workspace.source_text_file.read_text())
    except OSError:
        source_length = 0
    if source_length >= _CODEX_LONG_SOURCE_CHAR_THRESHOLD:
        return (
            _positive_int_env(
                "AXIOM_ENCODE_CODEX_LONG_TIMEOUT_SECONDS",
                _CODEX_LONG_SOURCE_TIMEOUT_SECONDS,
            ),
            _positive_int_env(
                "AXIOM_ENCODE_CODEX_LONG_IDLE_TIMEOUT_SECONDS",
                _CODEX_LONG_SOURCE_IDLE_TIMEOUT_SECONDS,
            ),
        )
    return (
        _positive_int_env(
            "AXIOM_ENCODE_CODEX_TIMEOUT_SECONDS",
            _CODEX_DEFAULT_TIMEOUT_SECONDS,
        ),
        _positive_int_env(
            "AXIOM_ENCODE_CODEX_IDLE_TIMEOUT_SECONDS",
            _CODEX_DEFAULT_IDLE_TIMEOUT_SECONDS,
        ),
    )


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _wait_for_codex_process(
    process: subprocess.Popen[str],
    last_message_file: Path,
    timeout: int,
    *,
    heartbeat_paths: Sequence[Path] | None = None,
    settle_seconds: float = 5.0,
    max_output_wait_seconds: float = 30.0,
    max_idle_seconds: float = 120.0,
    poll_interval: float = 0.5,
) -> bool:
    """Wait for Codex CLI, terminating it once output is stable or persistent."""
    start = time.time()
    last_snapshot: tuple[int, int] | None = None
    stable_since: float | None = None
    output_seen_at: float | None = None
    last_activity_at = start
    heartbeat_snapshot: tuple[tuple[int, int, int], ...] | None = None

    def _snapshot_activity() -> tuple[tuple[int, int, int], ...]:
        files = [last_message_file, *(heartbeat_paths or [])]
        snapshot: list[tuple[int, int, int]] = []
        for path in files:
            if not path.exists():
                snapshot.append((0, 0, 0))
                continue
            try:
                stat = path.stat()
            except OSError:
                snapshot.append((0, 0, 0))
                continue
            snapshot.append((1, stat.st_size, stat.st_mtime_ns))
        return tuple(snapshot)

    while True:
        if process.poll() is not None:
            return False

        now = time.time()
        if now - start > timeout:
            raise subprocess.TimeoutExpired(process.args, timeout)

        current_heartbeat_snapshot = _snapshot_activity()
        if current_heartbeat_snapshot != heartbeat_snapshot:
            heartbeat_snapshot = current_heartbeat_snapshot
            last_activity_at = now
        elif now - last_activity_at >= max_idle_seconds:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise subprocess.TimeoutExpired(process.args, max_idle_seconds)

        if last_message_file.exists():
            try:
                text = last_message_file.read_text().strip()
                stat = last_message_file.stat()
            except OSError:
                text = ""
                stat = None

            if text and stat is not None:
                output_seen_at = output_seen_at or now
                snapshot = (stat.st_size, stat.st_mtime_ns)
                if snapshot == last_snapshot:
                    stable_since = stable_since or now
                    if now - stable_since >= settle_seconds:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        return True
                else:
                    last_snapshot = snapshot
                    stable_since = None
                    if (
                        output_seen_at is not None
                        and now - output_seen_at >= max_output_wait_seconds
                    ):
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        return True

        time.sleep(poll_interval)


def _extract_openai_response_text(payload: dict) -> str:
    """Flatten a Responses API payload into assistant text."""
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            continue
        if item.get("type") == "message":
            for content_item in item.get("content", []) or []:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") in {"output_text", "text"}:
                    text = content_item.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

    return "\n\n".join(texts).strip()


def _run_openai_prompt_eval(
    runner: EvalRunnerSpec,
    workspace: EvalWorkspace,
    prompt: str,
) -> EvalPromptResponse:
    """Run prompt-only eval via the OpenAI Responses API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return EvalPromptResponse(
            text="",
            duration_ms=0,
            trace={
                "provider": "openai",
                "backend": "responses",
                "model": runner.model,
            },
            error="OPENAI_API_KEY is not set",
        )

    body = {
        "model": runner.model,
        "input": prompt,
        "max_output_tokens": 16384,
        "reasoning": {
            "effort": "low",
            "summary": "auto",
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start = time.time()
    try:
        response = _post_openai_eval_request(headers=headers, body=body)
    except requests.RequestException as exc:
        duration_ms = int((time.time() - start) * 1000)
        return EvalPromptResponse(
            text="",
            duration_ms=duration_ms,
            trace={
                "provider": "openai",
                "backend": "responses",
                "model": runner.model,
                "request_body": body,
            },
            error=str(exc),
        )
    duration_ms = int((time.time() - start) * 1000)

    request_id = response.headers.get("x-request-id")
    try:
        payload = response.json()
    except ValueError:
        payload = {
            "error": {
                "message": response.text or f"HTTP {response.status_code}",
            }
        }

    trace = {
        "provider": "openai",
        "backend": "responses",
        "model": runner.model,
        "request_id": request_id,
        "request_body": body,
        "json_result": payload,
        "status_code": response.status_code,
    }

    if response.status_code >= 400:
        error = payload.get("error") or {}
        return EvalPromptResponse(
            text="",
            duration_ms=duration_ms,
            trace=trace,
            error=error.get("message") or response.text or "OpenAI eval failed",
        )

    usage = payload.get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    tokens = TokenUsage(
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cache_read_tokens=int(input_details.get("cached_tokens", 0) or 0),
    )
    tokens.reasoning_output_tokens = int(
        ((usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
    )

    return EvalPromptResponse(
        text=_extract_openai_response_text(payload),
        duration_ms=duration_ms,
        tokens=tokens,
        estimated_cost_usd=estimate_usage_cost_usd(runner.model, tokens),
        trace=trace,
    )


def _post_openai_eval_request(
    headers: dict[str, str],
    body: dict[str, object],
    attempts: int = 6,
) -> requests.Response:
    """POST a Responses API eval request with transient retry handling."""
    last_response: requests.Response | None = None
    last_error: requests.RequestException | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=body,
                timeout=(30, 180),
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt == attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 10))
            continue

        last_response = response
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == attempts:
            return response
        time.sleep(min(2 ** (attempt - 1), 10))

    if last_response is not None:
        return last_response
    if last_error is not None:
        raise last_error
    raise requests.RequestException("OpenAI eval request failed without response")


def _command_looks_out_of_bounds(command: str, workspace_root: Path) -> bool:
    """Heuristic for whether a Codex shell command accessed paths outside workspace."""
    if not command:
        return False

    if re.search(r"(^|[\s'\"`])\.\.(?:/|[\s'\"`]|$)", command):
        return True

    abs_paths = re.findall(r"(?:(?<=^)|(?<=[\s'\"`]))(/[^\s'\"`]+)", command)
    for raw_path in abs_paths:
        path = Path(raw_path)
        try:
            path.resolve().relative_to(workspace_root.resolve())
        except ValueError:
            return True
        except FileNotFoundError:
            return True

    return False


def _command_uses_policyengine_skill(command: str) -> bool:
    """Return True when a Codex shell command reads PolicyEngine skill material."""
    if not command:
        return False
    normalized = command.lower()
    policyengine_skill_markers = (
        "/policyengine-skills/",
        "policyengine-skills/",
        "/skills/workflows/encode-policy",
        "encode-policy-v2-skill",
    )
    return any(marker in normalized for marker in policyengine_skill_markers)


def _extract_rulespec_content(llm_response: str) -> str | None:
    """Extract raw RuleSpec content from a model response."""
    if not llm_response or not llm_response.strip():
        return None

    cleaned = re.sub(r"\x1b\[[0-9;]*m", "", llm_response)

    fence_match = re.search(r"```(?:yaml|text)?\s*\n(.*?)\n```", cleaned, re.DOTALL)
    if fence_match:
        content = fence_match.group(1).strip()
        if content.startswith("format: rulespec/v1"):
            return content + "\n"

    stripped = cleaned.strip()
    if re.match(r"^(format:\s*rulespec/v1|schema:\s*axiom\.rules\.)", stripped):
        return stripped + ("\n" if not stripped.endswith("\n") else "")

    lines = stripped.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^(format:\s*rulespec/v1|schema:\s*axiom\.rules\.)", line):
            return "\n".join(lines[i:]).strip() + "\n"

    return None


def _normalize_rulespec_content(content: str) -> str:
    """Normalize generated RuleSpec without rewriting embedded source prose."""
    stripped = content.strip()
    if stripped and not stripped.startswith("format: rulespec/v1"):
        raise ValueError("RuleSpec content must start with `format: rulespec/v1`")
    return stripped + ("\n" if stripped else "")


def _normalize_main_eval_content(
    content: str,
    *,
    target_path: Path,
    single_amount_table_slice: bool,
    source_text: str | None = None,
) -> str:
    """Normalize generated main artifacts according to their format."""
    if target_path.suffix not in {".yaml", ".yml"}:
        raise ValueError("RuleSpec artifacts must use .yaml or .yml paths")
    content = _clean_generated_file_content(content)
    normalized = _normalize_rulespec_content(content)
    normalized, _repaired_rules = repair_source_table_band_scalar_parameters(
        normalized,
        source_text=source_text,
    )
    normalized, _bound_repairs = repair_source_table_open_ended_bound_sentinels(
        normalized,
        source_text=source_text,
    )
    normalized, _interval_repairs = repair_source_table_interval_row_alignment(
        normalized,
        source_text=source_text,
    )
    normalized, _summary_repairs = repair_copied_cross_reference_summary(
        normalized,
        rules_file=target_path,
    )
    normalized, _let_binding_repairs = repair_formula_let_bindings(normalized)
    normalized, _conditional_repairs = repair_unsupported_chained_conditionals(
        normalized
    )
    return normalized


def _rulespec_rule_names(content: str) -> set[str]:
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return set()
    if not isinstance(payload, dict):
        return set()
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return set()
    names: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        name = str(rule.get("name") or "").strip()
        if name:
            names.add(name)
    return names


def _removed_rulespec_rule_names(original: str, normalized: str) -> set[str]:
    return _rulespec_rule_names(original) - _rulespec_rule_names(normalized)


def _indexed_parameter_rule_names(content: str) -> set[str]:
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return set()
    if not isinstance(payload, dict):
        return set()
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return set()
    names: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("kind") or "").strip().lower() != "parameter":
            continue
        if rule.get("indexed_by") is None:
            continue
        name = str(rule.get("name") or "").strip()
        if name:
            names.add(name)
    return names


def _strip_test_outputs_for_rule_names(
    content: str,
    *,
    rule_names: set[str],
) -> str:
    if not rule_names:
        return content
    try:
        cases = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return content
    if not isinstance(cases, list):
        return content

    changed = False
    for case in cases:
        if not isinstance(case, dict):
            continue
        output = case.get("output")
        if not isinstance(output, dict):
            continue
        filtered = {
            key: value
            for key, value in output.items()
            if _rulespec_output_key_name(key) not in rule_names
        }
        if len(filtered) != len(output):
            case["output"] = filtered
            changed = True

    if not changed:
        return content
    return yaml.safe_dump(cases, sort_keys=False).strip() + "\n"


def _rulespec_output_key_name(key: object) -> str | None:
    if not isinstance(key, str):
        return None
    if "#" in key:
        return key.rsplit("#", 1)[1].strip() or None
    return key.strip() or None


def _extract_generated_file_bundle(llm_response: str) -> dict[str, str]:
    """Extract a small multi-file bundle emitted by eval backends."""
    if not llm_response or "=== FILE:" not in llm_response:
        return {}

    pattern = re.compile(r"^=== FILE:\s*(?P<name>.+?)\s*===\s*$", re.MULTILINE)
    matches = list(pattern.finditer(llm_response))
    if not matches:
        return {}

    files: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(llm_response)
        )
        content = llm_response[start:end].strip()
        if content:
            files[match.group("name").strip()] = _clean_generated_file_content(content)
    return files


def _clean_generated_file_content(content: str) -> str:
    """Strip common wrapper noise from bundled file content."""
    stripped = content.strip()
    fenced = re.match(
        r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n```(?:\s|$)",
        stripped,
        re.DOTALL,
    )
    if fenced:
        stripped = fenced.group(1).strip()
    stripped = re.sub(
        r"(?<![\d.])(-?\d+(?:,\d{3})*(?:\.\d+)?)\s+(GBP|USD|EUR)\b",
        lambda match: match.group(1),
        stripped,
    )
    stripped = _normalize_backslash_escaped_yaml_apostrophes(stripped)
    stripped = _normalize_semicolon_separated_yaml_excerpts(stripped)
    return stripped + ("\n" if stripped and not stripped.endswith("\n") else "")


def _normalize_semicolon_separated_yaml_excerpts(content: str) -> str:
    """Repair excerpt lines emitted as multiple adjacent quoted scalars."""
    if "excerpt:" not in content or not re.search(
        r"['\"]\s*(?:;|\b(?:and|or)\b)\s*['\"]",
        content,
    ):
        return content

    repaired_lines: list[str] = []
    for line in content.splitlines(keepends=True):
        repaired_lines.append(_normalize_semicolon_separated_yaml_excerpt_line(line))
    return "".join(repaired_lines)


def _normalize_semicolon_separated_yaml_excerpt_line(line: str) -> str:
    body = line.rstrip("\r\n")
    newline = line[len(body) :]
    match = re.match(r"^(?P<prefix>\s*excerpt:\s*)(?P<rest>.+?)\s*$", body)
    if not match:
        return line
    parsed = _parse_semicolon_separated_quoted_scalars(match.group("rest").strip())
    if parsed is None or len(parsed[0]) < 2:
        return line
    values, separators = parsed
    joined = values[0] + "".join(
        separator + value for separator, value in zip(separators, values[1:])
    )
    escaped = joined.replace("\\", "\\\\").replace('"', '\\"')
    return f'{match.group("prefix")}"{escaped}"{newline}'


def _parse_semicolon_separated_quoted_scalars(
    text: str,
) -> tuple[list[str], list[str]] | None:
    values: list[str] = []
    separators: list[str] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] not in {"'", '"'}:
            return None
        quote = text[index]
        index += 1
        value: list[str] = []
        while index < len(text):
            char = text[index]
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if char == "\\" and next_char:
                value.append(next_char)
                index += 2
                continue
            if quote == "'" and char == "'" and next_char == "'":
                value.append("'")
                index += 2
                continue
            if char == quote:
                index += 1
                break
            value.append(char)
            index += 1
        else:
            return None
        values.append("".join(value))
        while index < len(text) and text[index].isspace():
            index += 1
        if index == len(text):
            return values, separators
        if text[index] == ";":
            separators.append("; ")
            index += 1
            continue
        separator_match = re.match(r"(and|or)\b", text[index:], flags=re.IGNORECASE)
        if separator_match is None:
            return None
        separators.append(f" {separator_match.group(1).lower()} ")
        index += len(separator_match.group(1))
    return values, separators


def _normalize_backslash_escaped_yaml_apostrophes(content: str) -> str:
    """Repair LLM-emitted YAML quote escapes that PyYAML cannot parse."""
    if "\\'" not in content:
        return content

    repaired_lines: list[str] = []
    for line in content.splitlines(keepends=True):
        if "\\'" not in line:
            repaired_lines.append(line)
            continue
        repaired_lines.append(
            _normalize_backslash_escaped_yaml_apostrophes_in_line(line)
        )
    return "".join(repaired_lines)


def _normalize_backslash_escaped_yaml_apostrophes_in_line(line: str) -> str:
    output: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(line):
        char = line[index]
        next_char = line[index + 1] if index + 1 < len(line) else ""
        if quote is None:
            if char in {"'", '"'}:
                quote = char
            output.append(char)
            index += 1
            continue

        if quote == "'":
            if char == "\\" and next_char == "'":
                output.append("''")
                index += 2
                continue
            if char == "'" and next_char == "'":
                output.append("''")
                index += 2
                continue
            if char == "'":
                quote = None
            output.append(char)
            index += 1
            continue

        if char == "\\" and next_char == "'":
            output.append("'")
            index += 2
            continue
        if char == '"':
            quote = None
        output.append(char)
        index += 1
    return "".join(output)


def _normalize_comma_numeric_literals(content: str) -> str:
    """Strip thousands separators from numeric literals without touching prose."""
    return re.sub(
        r"(?<![\d.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d.])",
        lambda match: match.group(0).replace(",", ""),
        content,
    )


def _format_safe_numeric_expression(expression: str) -> str | None:
    """Evaluate a numeric literal or simple arithmetic expression safely."""
    try:
        value = _evaluate_safe_numeric_expression(expression.strip())
    except (ValueError, SyntaxError, ZeroDivisionError):
        return None

    if float(value).is_integer():
        return str(int(value))
    return format(value, "g")


def _evaluate_safe_numeric_expression(expression: str) -> float:
    """Return the numeric value of a simple arithmetic expression."""
    node = ast.parse(expression, mode="eval")

    def visit(current: ast.AST) -> float:
        if isinstance(current, ast.Expression):
            return visit(current.body)
        if isinstance(current, ast.Constant) and isinstance(
            current.value, (int, float)
        ):
            return float(current.value)
        if isinstance(current, ast.UnaryOp) and isinstance(
            current.op, (ast.UAdd, ast.USub)
        ):
            operand = visit(current.operand)
            return operand if isinstance(current.op, ast.UAdd) else -operand
        if isinstance(current, ast.BinOp) and isinstance(
            current.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = visit(current.left)
            right = visit(current.right)
            if isinstance(current.op, ast.Add):
                return left + right
            if isinstance(current.op, ast.Sub):
                return left - right
            if isinstance(current.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError(f"Unsupported numeric expression: {expression}")

    return visit(node)


def _normalize_single_amount_row_test_content(
    content: str,
    rulespec_content: str | None = None,
    source_text: str | None = None,
) -> str:
    """Drop alternate-branch tests for one-row fixed-amount slices."""
    normalized = _normalize_comma_numeric_literals(content)
    try:
        payload = yaml.safe_load(normalized)
    except yaml.YAMLError:
        return normalized

    if payload is None:
        return normalized

    annual_period = bool(
        rulespec_content
        and re.search(r"^\s*period:\s*Year\s*$", rulespec_content, flags=re.MULTILINE)
    )
    effective_date = _extract_effective_date_for_tests(
        rulespec_content=rulespec_content,
        source_text=source_text,
    )

    def should_keep(case_name: str | None) -> bool:
        if not case_name:
            return True
        lowered = case_name.lower()
        if "alternate" in lowered:
            return False
        if annual_period and "effective_date_boundary" in lowered:
            return False
        return True

    def normalize_case(case: object) -> object:
        if not isinstance(case, dict):
            return case
        normalized_case = dict(case)
        if annual_period and effective_date is not None:
            normalized_case["period"] = _normalize_annual_test_period_value(
                normalized_case.get("period"),
                effective_date,
            )
        for key in ("input", "inputs", "output"):
            if key in normalized_case and isinstance(normalized_case[key], dict):
                normalized_case[key] = {
                    child_key: _normalize_test_case_value(child_value)
                    for child_key, child_value in normalized_case[key].items()
                }
        output = normalized_case.get("output")
        if isinstance(output, dict) and len(output) > 1:
            numeric_output = {
                key: value
                for key, value in output.items()
                if value is None
                or (isinstance(value, (int, float)) and not isinstance(value, bool))
            }
            if numeric_output:
                normalized_case["output"] = numeric_output
        return normalized_case

    cases = _coerce_test_payload_to_case_list(payload)
    if cases is not None:
        filtered = [
            normalize_case(case)
            for case in cases
            if not isinstance(case, dict) or should_keep(case.get("name"))
        ]
        return yaml.safe_dump(filtered, sort_keys=False).strip() + "\n"

    return normalized


def _extract_effective_date_for_tests(
    rulespec_content: str | None,
    source_text: str | None,
) -> date | None:
    """Return the earliest explicit effective date available for test normalization."""
    if rulespec_content:
        if from_match := re.search(
            r"\beffective_from:\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?",
            rulespec_content,
        ):
            parsed = date.fromisoformat(from_match.group(1))
            if parsed != date(1, 1, 1):
                return parsed
    if source_text and (
        source_match := re.search(
            r"\b(?:text|current text)\s+valid\s+from\s+(\d{4}-\d{2}-\d{2})\b",
            source_text,
            flags=re.IGNORECASE,
        )
    ):
        parsed = date.fromisoformat(source_match.group(1))
        if parsed != date(1, 1, 1):
            return parsed
    return None


def _normalize_annual_test_period_value(
    period: object,
    effective_date: date,
) -> object:
    """Convert bare annual periods into concrete dates on or after the effective date."""
    year: int | None = None
    if period is None:
        year = effective_date.year
    elif isinstance(period, int):
        year = period
    elif isinstance(period, str) and re.fullmatch(r"\d{4}", period):
        year = int(period)

    if year is None:
        return period
    if year < effective_date.year:
        return period
    if year == effective_date.year:
        return effective_date.isoformat()
    return f"{year}-01-01"


def _extract_rulespec_period_granularity(rulespec_content: str | None) -> str | None:
    """Return the first declared RuleSpec period name."""
    if rulespec_content is None:
        return None
    match = re.search(
        r"^\s*period:\s*(Year|Month|Week|Day)\s*$",
        rulespec_content,
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def _default_test_period_for_granularity(
    granularity: str | None,
    effective_date: date,
) -> str:
    """Return a concrete test period compatible with the test runner."""
    return effective_date.isoformat()


def _normalize_week_test_period_value(period: object) -> object:
    """Convert ISO week shorthands into explicit benefit-week periods."""
    if not isinstance(period, str) or not _ISO_WEEK_PERIOD_PATTERN.fullmatch(period):
        return period
    year = int(period[:4])
    week = int(period[-2:])
    try:
        start = date.fromisocalendar(year, week, 1)
    except ValueError:
        return period
    end = date.fromordinal(start.toordinal() + 6)
    return {
        "period_kind": "benefit_week",
        "start": start.isoformat(),
        "end": end.isoformat(),
    }


def _normalize_nonannual_test_period_value(
    period: object,
    effective_date: date,
    granularity: str | None = None,
) -> object:
    """Normalize non-annual test periods while preserving explicit monthly boundary tests."""
    if granularity == "Month":
        effective_month = effective_date.strftime("%Y-%m")
        if period is None:
            return effective_month
        if isinstance(period, date):
            period_month = period.strftime("%Y-%m")
            return period_month
        if isinstance(period, int):
            if period == effective_date.year:
                return effective_month
            if period > effective_date.year:
                return f"{period}-01"
            return period
        if isinstance(period, str):
            if re.fullmatch(r"\d{4}", period):
                year = int(period)
                if year == effective_date.year:
                    return effective_month
                if year > effective_date.year:
                    return f"{year}-01"
                return period
            if _ISO_WEEK_PERIOD_PATTERN.fullmatch(period):
                week_year = int(period[:4])
                if week_year == effective_date.year:
                    return effective_month
                if week_year > effective_date.year:
                    return f"{week_year}-01"
                return period
            if re.fullmatch(r"\d{4}-\d{2}", period):
                return period
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", period):
                try:
                    parsed = date.fromisoformat(period)
                except ValueError:
                    return period
                return parsed.strftime("%Y-%m")

    if granularity == "Week":
        normalized_week = _normalize_week_test_period_value(period)
        if normalized_week != period:
            return normalized_week

    if period is None:
        return effective_date.isoformat()
    if isinstance(period, date):
        if period < effective_date:
            return effective_date.isoformat()
        return period.isoformat()
    if isinstance(period, int):
        if period == effective_date.year:
            return effective_date.isoformat()
        if period > effective_date.year:
            return f"{period}-01-01"
        return period
    if isinstance(period, str):
        if re.fullmatch(r"\d{4}", period):
            year = int(period)
            if year == effective_date.year:
                return effective_date.isoformat()
            if year > effective_date.year:
                return f"{year}-01-01"
            return period
        if _ISO_WEEK_PERIOD_PATTERN.fullmatch(period):
            week_year = int(period[:4])
            if week_year == effective_date.year:
                return effective_date.isoformat()
            if week_year > effective_date.year:
                return f"{week_year}-01-01"
            return period
        if re.fullmatch(r"\d{4}-\d{2}", period):
            if period == effective_date.strftime("%Y-%m"):
                return effective_date.isoformat()
            return period
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", period):
            try:
                parsed = date.fromisoformat(period)
            except ValueError:
                return period
            if parsed < effective_date:
                return effective_date.isoformat()
            return period
    return period


def _normalize_placeholder_monthly_test_period_value(period: object) -> object:
    """Replace placeholder month periods with a contemporary comparable month."""
    contemporary_month = "2024-01"
    if period is None:
        return contemporary_month
    if isinstance(period, date) and period.year <= 1:
        return contemporary_month
    if isinstance(period, int) and period <= 1:
        return contemporary_month
    if isinstance(period, str):
        stripped = period.strip()
        if stripped in {"", "1", "0001"}:
            return contemporary_month
        if re.fullmatch(r"0001-\d{2}", stripped):
            return contemporary_month
        if re.fullmatch(r"0001-\d{2}-\d{2}", stripped):
            return contemporary_month
    return period


def _coerce_test_payload_to_case_list(payload: object) -> list[object] | None:
    """Return test payloads as a plain list of case objects when recognizable."""
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return None

    tests_payload = payload.get("tests")
    if isinstance(tests_payload, list):
        return tests_payload

    case_like_keys = {"name", "period", "input", "inputs", "output", "expect"}
    if case_like_keys & set(payload):
        return [payload]

    if not payload or not all(isinstance(value, dict) for value in payload.values()):
        return None

    cases: list[object] = []
    for key, value in payload.items():
        case = dict(value)
        if isinstance(key, str) and "name" not in case:
            case["name"] = key
        cases.append(case)
    return cases


def _normalize_test_case_value(value: object) -> object:
    """Collapse entity/time wrappers in generated RuleSpec test values to scalars."""
    if isinstance(value, list):
        return [_normalize_test_case_value(item) for item in value]
    if isinstance(value, str):
        expression = value.strip()
        if _PURE_NUMERIC_EXPRESSION_PATTERN.fullmatch(expression):
            try:
                formatted = _format_safe_numeric_expression(expression)
                if formatted is None:
                    return value
                return yaml.safe_load(formatted)
            except (TypeError, ValueError, yaml.YAMLError):
                return value
        return value
    if not isinstance(value, dict):
        return value

    normalized = {
        key: _normalize_test_case_value(inner) for key, inner in value.items()
    }
    lowered_keys = {str(key).lower() for key in normalized.keys()}
    metadata_keys = {
        "entity",
        "period",
        "dtype",
        "unit",
        "label",
        "description",
        "default",
    }

    values_entries = [
        inner for key, inner in normalized.items() if str(key).lower() == "values"
    ]
    if len(values_entries) == 1 and lowered_keys.issubset(metadata_keys | {"values"}):
        values_entry = values_entries[0]
        if isinstance(values_entry, dict) and len(values_entry) == 1:
            return _normalize_test_case_value(next(iter(values_entry.values())))
        return _normalize_test_case_value(values_entry)

    from_entries = [
        inner
        for key, inner in normalized.items()
        if str(key).lower().startswith("from ")
    ]
    if from_entries and lowered_keys.issubset(
        metadata_keys
        | {
            str(key).lower()
            for key in normalized.keys()
            if str(key).lower().startswith("from ")
        }
    ):
        if len(from_entries) == 1:
            return _normalize_test_case_value(from_entries[0])

    if len(normalized) == 1:
        only_key, only_value = next(iter(normalized.items()))
        if re.fullmatch(r"\d{4}(?:-\d{2})?(?:-\d{2})?", str(only_key)):
            return _normalize_test_case_value(only_value)
        if not isinstance(only_value, (dict, list)):
            return only_value

    return normalized


def _normalize_test_case_output_value(value: object) -> object:
    """Normalize generated expected outputs that need YAML-native scalar types."""
    if isinstance(value, list):
        return [_normalize_test_case_output_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_test_case_output_value(inner)
            for key, inner in value.items()
        }
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return value
    return _normalize_test_case_value(value)


def _period_precedes_effective_month(period: object, effective_date: date) -> bool:
    """Return True when an explicit period falls before the effective month."""
    effective_month = effective_date.strftime("%Y-%m")
    if isinstance(period, date):
        return period.strftime("%Y-%m") < effective_month
    if isinstance(period, str):
        if re.fullmatch(r"\d{4}-\d{2}", period):
            return period < effective_month
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", period):
            with contextlib.suppress(ValueError):
                return date.fromisoformat(period).strftime("%Y-%m") < effective_month
    return False


def _case_outputs_only_zero_values(case: object) -> bool:
    """Return True when a test case only asserts zero/false-like outputs."""
    if not isinstance(case, dict):
        return False
    output = case.get("output")
    if not isinstance(output, dict) or not output:
        return False
    saw_value = False
    for value in output.values():
        if value is None:
            continue
        saw_value = True
        if isinstance(value, bool):
            if value:
                return False
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if float(value) != 0.0:
                return False
            continue
        return False
    return saw_value


def _case_output_keys(case: object) -> set[str]:
    if not isinstance(case, dict):
        return set()
    output = case.get("output")
    if not isinstance(output, dict):
        return set()
    return {str(key) for key in output.keys()}


def _normalize_test_periods_to_effective_dates(
    content: str,
    rulespec_content: str | None = None,
    source_text: str | None = None,
) -> str:
    """Normalize annual test periods to concrete dates that survive effective-date compilation."""
    normalized = _normalize_comma_numeric_literals(content)
    granularity = _extract_rulespec_period_granularity(rulespec_content)
    effective_date = _extract_effective_date_for_tests(
        rulespec_content=rulespec_content,
        source_text=source_text,
    )

    try:
        payload = yaml.safe_load(normalized)
    except yaml.YAMLError:
        return normalized

    if payload is None:
        return normalized

    cases = _coerce_test_payload_to_case_list(payload)
    positive_output_keys: set[str] = set()
    if cases is not None:
        for case in cases:
            if not isinstance(case, dict):
                continue
            output = case.get("output")
            if not isinstance(output, dict):
                continue
            if any(
                (isinstance(value, bool) and value)
                or (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and float(value) != 0.0
                )
                for value in output.values()
                if value is not None
            ):
                positive_output_keys.update(_case_output_keys(case))

    def normalize_case(case: object) -> object:
        if not isinstance(case, dict):
            return case
        normalized_case = _repair_misindented_period_mapping_fields(case)
        if granularity == "Year" and effective_date is not None:
            normalized_case["period"] = _normalize_annual_test_period_value(
                normalized_case.get("period"),
                effective_date,
            )
        elif effective_date is not None:
            normalized_case["period"] = _normalize_nonannual_test_period_value(
                normalized_case.get("period"),
                effective_date,
                granularity=granularity,
            )
        elif granularity == "Month":
            normalized_case["period"] = (
                _normalize_placeholder_monthly_test_period_value(
                    normalized_case.get("period")
                )
            )
        elif granularity == "Week":
            normalized_case["period"] = _normalize_week_test_period_value(
                normalized_case.get("period")
            )

        for key in ("input", "inputs"):
            if key in normalized_case and isinstance(normalized_case[key], dict):
                normalized_case[key] = {
                    child_key: _normalize_test_case_value(child_value)
                    for child_key, child_value in normalized_case[key].items()
                }
        if "output" in normalized_case and isinstance(normalized_case["output"], dict):
            normalized_case["output"] = {
                child_key: _normalize_test_case_output_value(child_value)
                for child_key, child_value in normalized_case["output"].items()
            }
        if "expect" in normalized_case:
            normalized_case["expect"] = _normalize_test_case_output_value(
                normalized_case["expect"]
            )
        return normalized_case

    if cases is not None:
        filtered_cases: list[object] = []
        for case in cases:
            if (
                granularity == "Month"
                and effective_date is not None
                and isinstance(case, dict)
                and "pre_effective" in str(case.get("name", "")).lower()
                and _period_precedes_effective_month(case.get("period"), effective_date)
                and _case_outputs_only_zero_values(case)
                and (_case_output_keys(case) & positive_output_keys)
            ):
                continue
            filtered_cases.append(case)
        return (
            yaml.safe_dump(
                [normalize_case(case) for case in filtered_cases],
                sort_keys=False,
            ).strip()
            + "\n"
        )

    return normalized


def _repair_misindented_period_mapping_fields(case: dict[str, Any]) -> dict[str, Any]:
    """Move generated top-level period fields back under `period` when unambiguous."""
    normalized_case = dict(case)
    period = normalized_case.get("period")
    if not isinstance(period, dict):
        return normalized_case

    repaired_period = dict(period)
    for key in ("period_kind", "start", "end"):
        if key not in normalized_case:
            continue
        if key not in repaired_period:
            repaired_period[key] = normalized_case.pop(key)
            continue
        if normalized_case[key] == repaired_period[key]:
            normalized_case.pop(key)
    normalized_case["period"] = repaired_period
    return normalized_case


def _parse_simple_rulespec_literal(value: str) -> object | None:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped in {"true", "false"}:
        return stripped == "true"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
        return yaml.safe_load(stripped)
    return None


def _extract_simple_rulespec_constant(
    rulespec_content: str | None,
    var_name: str | None,
) -> object | None:
    if not rulespec_content or not var_name:
        return None
    with contextlib.suppress(yaml.YAMLError, TypeError):
        payload = yaml.safe_load(rulespec_content)
        if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
            for rule in payload["rules"]:
                if not isinstance(rule, dict) or rule.get("name") != var_name:
                    continue
                versions = rule.get("versions")
                if not isinstance(versions, list):
                    continue
                for version in versions:
                    if not isinstance(version, dict):
                        continue
                    formula = version.get("formula")
                    if isinstance(formula, (int, float, bool)):
                        return formula
                    if isinstance(formula, str):
                        literal = _parse_simple_rulespec_literal(formula)
                        if literal is not None:
                            return literal
    return None


def _rulespec_declares_rule(
    rulespec_content: str | None,
    var_name: str | None,
) -> bool:
    if not rulespec_content or not var_name:
        return False
    with contextlib.suppress(yaml.YAMLError, TypeError):
        payload = yaml.safe_load(rulespec_content)
        if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
            return any(
                isinstance(rule, dict) and rule.get("name") == var_name
                for rule in payload["rules"]
            )
    return False


def _canonical_rulespec_target_for_path(rulespec_path: Path | None) -> str | None:
    if rulespec_path is None:
        return None
    components = [
        str(component)
        for component in rulespec_path.expanduser().resolve(strict=False).parts
    ]
    repo_index = next(
        (
            index
            for index in range(len(components) - 1, -1, -1)
            if components[index].startswith("rulespec-")
        ),
        None,
    )
    if repo_index is None or repo_index + 1 >= len(components):
        return None
    prefix = components[repo_index].removeprefix("rulespec-")
    if not prefix:
        return None
    relative = Path(*components[repo_index + 1 :])
    if relative.suffix in {".yaml", ".yml"}:
        relative = relative.with_suffix("")
    return f"{prefix}:{relative.as_posix()}"


def _policyengine_hint_upstream_composition_issues(
    rulespec_content: str,
    policyengine_rule_hint: str | None,
) -> list[str]:
    """Flag PE-hinted surfaces that still rely on broad upstream placeholders."""
    if not policyengine_rule_hint:
        return []
    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(rulespec_content)
        if not isinstance(payload, dict):
            return []
        issues: list[str] = []
        module = payload.get("module")
        deferred_outputs = (
            module.get("deferred_outputs") if isinstance(module, dict) else None
        )
        if isinstance(deferred_outputs, list):
            for deferred_output in deferred_outputs:
                if not isinstance(deferred_output, dict):
                    continue
                output = str(deferred_output.get("output", "")).strip()
                if output == policyengine_rule_hint or output.endswith(
                    f"#{policyengine_rule_hint}"
                ):
                    issues.append(
                        "PolicyEngine hinted output "
                        f"`{policyengine_rule_hint}` is deferred; compose it from "
                        "available primary-source RuleSpec exports, or defer only "
                        "specific unavailable branches."
                    )
        rules = payload.get("rules")
        if not isinstance(rules, list):
            return issues
        rules_by_name = {
            rule.get("name"): rule
            for rule in rules
            if isinstance(rule, dict) and isinstance(rule.get("name"), str)
        }
        rule = rules_by_name.get(policyengine_rule_hint)
        if not isinstance(rule, dict):
            return issues

        placeholders: set[str] = set()
        visited: set[str] = set()
        pending = [policyengine_rule_hint]
        while pending:
            rule_name = pending.pop()
            if rule_name in visited:
                continue
            visited.add(rule_name)
            rule = rules_by_name.get(rule_name)
            if not isinstance(rule, dict):
                continue
            versions = rule.get("versions")
            if not isinstance(versions, list):
                continue
            for version in versions:
                if not isinstance(version, dict):
                    continue
                formula = version.get("formula")
                if not isinstance(formula, str):
                    continue
                placeholders.update(
                    match.group(0)
                    for match in _POLICYENGINE_HINT_BROAD_PLACEHOLDER_RE.finditer(
                        formula
                    )
                )
                for identifier in _RULESPEC_FORMULA_IDENTIFIER_RE.findall(formula):
                    if identifier in _RULESPEC_FORMULA_KEYWORDS:
                        continue
                    if identifier not in rules_by_name or identifier in visited:
                        continue
                    pending.append(identifier)
        if placeholders:
            issues.append(
                "PolicyEngine hinted output "
                f"`{policyengine_rule_hint}` uses broad upstream placeholder(s) "
                f"{', '.join(f'`{item}`' for item in sorted(placeholders))}. "
                "Import concrete primary-source RuleSpec outputs already present "
                "in context, encode the upstream source first, or defer only the "
                "specific unavailable branch instead of making the PE surface "
                "depend on aggregate leaf inputs."
            )
        return issues
    return []


def _policyengine_hint_test_output_key(
    rulespec_content: str | None,
    policyengine_rule_hint: str | None,
    rulespec_path: Path | None,
) -> str | None:
    if not _rulespec_declares_rule(rulespec_content, policyengine_rule_hint):
        return None
    canonical_target = _canonical_rulespec_target_for_path(rulespec_path)
    if canonical_target and policyengine_rule_hint:
        return f"{canonical_target}#{policyengine_rule_hint}"
    return policyengine_rule_hint


def _complete_oracle_hint_test_outputs(
    content: str,
    rulespec_content: str | None,
    policyengine_rule_hint: str | None,
    rulespec_path: Path | None = None,
) -> str:
    if not policyengine_rule_hint:
        return content
    output_key = _policyengine_hint_test_output_key(
        rulespec_content, policyengine_rule_hint, rulespec_path
    )
    if not output_key:
        return content
    hint_value = _extract_simple_rulespec_constant(
        rulespec_content, policyengine_rule_hint
    )
    try:
        payload = yaml.safe_load(content)
    except yaml.YAMLError:
        return content
    if not isinstance(payload, list):
        return content

    changed = False
    normalized_cases: list[object] = []
    for case in payload:
        if not isinstance(case, dict):
            normalized_cases.append(case)
            continue
        output = case.get("output")
        if not isinstance(output, dict) or output_key in output:
            normalized_cases.append(case)
            continue
        normalized_case = dict(case)
        normalized_output = dict(output)
        if output_key != policyengine_rule_hint and policyengine_rule_hint in output:
            normalized_output[output_key] = normalized_output.pop(
                policyengine_rule_hint
            )
        elif hint_value is None:
            normalized_cases.append(case)
            continue
        else:
            normalized_output[output_key] = hint_value
        normalized_case["output"] = normalized_output
        normalized_cases.append(normalized_case)
        changed = True
    if not changed:
        return content
    return yaml.safe_dump(normalized_cases, sort_keys=False).strip() + "\n"


def _materialize_eval_artifact(
    llm_response: str,
    expected_path: Path,
    source_text: str | None = None,
    workspace_root: Path | None = None,
    policyengine_rule_hint: str | None = None,
) -> bool:
    """Write an eval artifact and optional companion test file from model output."""
    single_amount_table_slice = bool(
        source_text and _is_single_amount_table_slice(source_text)
    )
    expected_test_path = _rulespec_test_path(expected_path)

    if workspace_root is not None:
        wrote_from_workspace = _materialize_workspace_artifacts(
            expected_path=expected_path,
            expected_test_path=expected_test_path,
            workspace_root=workspace_root,
            single_amount_table_slice=single_amount_table_slice,
            source_text=source_text,
            policyengine_rule_hint=policyengine_rule_hint,
        )
        if wrote_from_workspace:
            return True

    bundle = _extract_generated_file_bundle(llm_response)
    if bundle:
        wrote_main = False
        bundle_by_candidate_name = {
            Path(file_name).name: content for file_name, content in bundle.items()
        }
        main_bundle_name = expected_path.name
        test_bundle_name = expected_test_path.name
        if main_bundle_name not in bundle_by_candidate_name:
            generated_main_names = [
                Path(file_name).name
                for file_name in bundle
                if Path(file_name).suffix in {".yaml", ".yml"}
                and not Path(file_name).name.endswith(".test.yaml")
            ]
            if len(generated_main_names) == 1:
                main_bundle_name = generated_main_names[0]
                generated_test_name = (
                    Path(main_bundle_name).with_suffix(".test.yaml").name
                )
                if generated_test_name in bundle_by_candidate_name:
                    test_bundle_name = generated_test_name
                else:
                    generated_test_names = [
                        Path(file_name).name
                        for file_name in bundle
                        if Path(file_name).name.endswith(".test.yaml")
                    ]
                    if len(generated_test_names) == 1:
                        test_bundle_name = generated_test_names[0]
        normalized_main_content: str | None = None
        stripped_test_output_names: set[str] = set()
        raw_main_content = bundle_by_candidate_name.get(main_bundle_name)
        if raw_main_content is not None:
            try:
                normalized_main_content = _normalize_main_eval_content(
                    raw_main_content,
                    target_path=expected_path,
                    single_amount_table_slice=single_amount_table_slice,
                    source_text=source_text,
                )
                stripped_test_output_names = _removed_rulespec_rule_names(
                    raw_main_content,
                    normalized_main_content,
                )
                stripped_test_output_names.update(
                    _indexed_parameter_rule_names(normalized_main_content)
                )
            except ValueError:
                normalized_main_content = None
                stripped_test_output_names = set()
        for file_name, content in bundle.items():
            candidate_name = Path(file_name).name
            if candidate_name == main_bundle_name:
                target_path = expected_path
            elif candidate_name == test_bundle_name:
                target_path = expected_test_path
            else:
                continue
            if target_path == expected_path:
                if normalized_main_content is None:
                    continue
                content = normalized_main_content
            elif target_path == expected_test_path:
                if single_amount_table_slice:
                    content = _normalize_single_amount_row_test_content(
                        content,
                        rulespec_content=normalized_main_content or raw_main_content,
                        source_text=source_text,
                    )
                else:
                    content = _normalize_test_periods_to_effective_dates(
                        content,
                        rulespec_content=normalized_main_content or raw_main_content,
                        source_text=source_text,
                    )
                    content = _complete_oracle_hint_test_outputs(
                        content,
                        rulespec_content=normalized_main_content or raw_main_content,
                        policyengine_rule_hint=policyengine_rule_hint,
                        rulespec_path=expected_path,
                    )
                    content, _interval_test_repairs = (
                        repair_source_table_interval_tests(
                            content,
                            rulespec_content=normalized_main_content
                            or raw_main_content,
                            source_text=source_text,
                        )
                    )
                content = _strip_test_outputs_for_rule_names(
                    content,
                    rule_names=stripped_test_output_names,
                )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content)
            if target_path == expected_path:
                wrote_main = True
        if wrote_main or expected_path.exists():
            return True

    rulespec_content = _extract_rulespec_content(llm_response)
    if not rulespec_content:
        return False
    try:
        rulespec_content = _normalize_main_eval_content(
            rulespec_content,
            target_path=expected_path,
            single_amount_table_slice=single_amount_table_slice,
            source_text=source_text,
        )
    except ValueError:
        return False

    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_text(rulespec_content)
    return True


def _materialize_workspace_artifacts(
    expected_path: Path,
    expected_test_path: Path,
    workspace_root: Path,
    single_amount_table_slice: bool,
    source_text: str | None,
    policyengine_rule_hint: str | None = None,
) -> bool:
    """Salvage eval artifacts that a model wrote directly into the workspace."""
    workspace_main = workspace_root / expected_path.name
    workspace_test = workspace_root / expected_test_path.name
    if not workspace_main.exists():
        return False

    raw_main_content = workspace_main.read_text()
    try:
        main_content = _normalize_main_eval_content(
            raw_main_content,
            target_path=expected_path,
            single_amount_table_slice=single_amount_table_slice,
            source_text=source_text,
        )
    except ValueError:
        return False
    stripped_test_output_names = _removed_rulespec_rule_names(
        raw_main_content,
        main_content,
    )
    stripped_test_output_names.update(_indexed_parameter_rule_names(main_content))

    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_text(main_content)

    if workspace_test.exists():
        test_content = workspace_test.read_text()
        if single_amount_table_slice:
            test_content = _normalize_single_amount_row_test_content(
                test_content,
                rulespec_content=main_content,
                source_text=source_text,
            )
        else:
            test_content = _normalize_test_periods_to_effective_dates(
                test_content,
                rulespec_content=main_content,
                source_text=source_text,
            )
            test_content = _complete_oracle_hint_test_outputs(
                test_content,
                rulespec_content=main_content,
                policyengine_rule_hint=policyengine_rule_hint,
                rulespec_path=expected_path,
            )
            test_content, _interval_test_repairs = repair_source_table_interval_tests(
                test_content,
                rulespec_content=main_content,
                source_text=source_text,
            )
        test_content = _strip_test_outputs_for_rule_names(
            test_content,
            rule_names=stripped_test_output_names,
        )
        expected_test_path.write_text(test_content)

    return True


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "eval"
