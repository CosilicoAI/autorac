"""Fold eval-suite results into an N-runner model-capability board.

`eval-suite-report` compares exactly two runners from one suite output. A
capability board compares an open roster: runs happen per runner (often in
parallel, on different days, from single-runner manifest variants), and new
models join without re-running incumbents. This module folds any number of
`results.json` suite payloads into one board, refusing to mix runs that are
not comparable.

Comparability contract: every folded payload must carry the same suite name,
the same ordered case identities, the same corpus release identity, and the
same score-affecting execution identity (encoder, rules engine, RuleSpec
content/toolchain/waivers, backend timeout policy, timeout retry policy,
PolicyEngine runtime) — compared after dropping location-only fields, so the
same toolchain checked out at different paths still folds. The manifest
content hash may differ (single-runner variants of one suite differ byte-wise
but share case identities), and runner sets may differ — that is the
add-a-model path. Duplicate runner names across payloads are refused rather
than merged: two runs of one runner are two boards, not one.

The board consumes canonical v6 suite payloads and refuses anything else:
unknown schema versions, rows for runners a payload never declared, rows
whose case identity does not match the manifest, coverage claims the result
matrix contradicts, and malformed metric types are all hard errors rather
than silent reinterpretations.

The headline metric is the deterministic gate-pass rate: a case passes for a
runner when the encode succeeded and the artifact compiled, passed CI, and
contains zero ungrounded numeric literals — the eval-workspace analogue of
the drain's first-pass gate battery. Reviewer and oracle columns are
reported alongside but never fold into the headline: the generalist reviewer
is an LLM judgment, and oracle coverage varies by case.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Literal

BoardCellState = Literal["pass", "fail", "timeout", "error", "missing"]

_RESULTS_FILE_NAME = "results.json"

# The one producer schema this consumer understands. A new producer version
# must be reviewed here before boards fold it; test_eval_board locks this to
# the producer constant in cli.py.
SUPPORTED_RESULTS_SCHEMA = "axiom-encode/eval-suite-results/v6"

# The one execution-identity schema whose field semantics the normalizer
# below understands; test_eval_board locks this to the producer constant.
SUPPORTED_EXECUTION_IDENTITY_SCHEMA = "axiom-encode/eval-execution-identity/v3"

# The evidence schema this consumer understands; locked to the producer
# constant by test_eval_board.
SUPPORTED_EVIDENCE_SCHEMA = "axiom-encode/eval-suite-evidence/v5"
EVAL_BOARD_SCHEMA = "axiom-encode/eval-board/v2"

# Every persisted result row carries this self-binding digest.
_RESULT_SHA256_FIELD = "result_sha256"

# Location-only identity fields: where a checkout lives never affects scores,
# so normalized execution identities drop these before comparison.
_LOCATION_ONLY_IDENTITY_KEYS = frozenset({"path", "toolchain_root", "repository_root"})

# The PolicyEngine runtime identity additionally embeds its sealed
# environment's absolute locations (see policyengine_runtime's canonical
# identity + probe payload). These are dropped only inside the
# `policyengine_runtime` subtree; every remaining field there — versions,
# tree digests, locked_versions, probe flags — is score-affecting.
_POLICYENGINE_LOCATION_ONLY_KEYS = _LOCATION_ONLY_IDENTITY_KEYS | frozenset(
    {
        "rulespec_runtime_pin_path",
        "venv_root",
        "stdlib_root",
        "site_packages_root",
        "python_executable",
        "python_prefix",
        "python_base_prefix",
        "python_exec_prefix",
        "python_base_exec_prefix",
        "initial_sys_path",
        "effective_sys_path",
        "module_origin",
        "metadata_root",
    }
)

# Identity digests computed over path-bearing structures (the PolicyEngine
# runtime wrapper digest is the only producer key named exactly `sha256`;
# content digests use distinct names such as `content_sha256` and
# `working_tree_sha256`). The normalized structural comparison replaces them.
_LOCATION_DEPENDENT_DIGEST_KEYS = frozenset({"sha256"})


class EvalBoardError(ValueError):
    """A board input is unreadable, malformed, incomplete, or not comparable."""


@dataclass(frozen=True)
class BoardCase:
    """One suite case, in manifest order."""

    index: int
    name: str
    kind: str
    corpus_citation_path: str | None
    sha256: str | None


@dataclass(frozen=True)
class BoardCell:
    """One case x runner outcome."""

    state: BoardCellState
    duration_ms: int | None = None
    detail: str | None = None


@dataclass
class BoardRunnerStats:
    """Aggregated rates for one runner across the folded cases."""

    runner: str
    backend: str
    model: str
    source: str
    cases_run: int
    artifact_case_count: int
    timeout_count: int
    gate_pass_count: int
    compile_pass_count: int
    ci_pass_count: int
    zero_ungrounded_count: int
    success_count: int
    source_numeric_occurrences: int
    covered_source_numeric_occurrences: int
    generalist_review_pass_count: int
    generalist_review_scores: list[float] = field(default_factory=list)
    policyengine_case_count: int = 0
    policyengine_pass_count: int = 0
    durations_ms: list[int] = field(default_factory=list)
    costs_usd: list[float] = field(default_factory=list)

    @property
    def gate_pass_rate(self) -> float:
        return _rate(self.gate_pass_count, self.cases_run)

    @property
    def compile_pass_rate(self) -> float | None:
        return _optional_rate(self.compile_pass_count, self.artifact_case_count)

    @property
    def ci_pass_rate(self) -> float | None:
        return _optional_rate(self.ci_pass_count, self.artifact_case_count)

    @property
    def zero_ungrounded_rate(self) -> float | None:
        return _optional_rate(self.zero_ungrounded_count, self.artifact_case_count)

    @property
    def source_numeric_coverage_rate(self) -> float | None:
        if self.source_numeric_occurrences <= 0:
            return None
        return round(
            self.covered_source_numeric_occurrences / self.source_numeric_occurrences,
            6,
        )

    @property
    def generalist_review_pass_rate(self) -> float | None:
        return _optional_rate(
            self.generalist_review_pass_count,
            self.artifact_case_count,
        )

    @property
    def mean_generalist_review_score(self) -> float | None:
        if not self.generalist_review_scores:
            return None
        return round(mean(self.generalist_review_scores), 6)

    @property
    def policyengine_pass_rate(self) -> float | None:
        if self.policyengine_case_count <= 0:
            return None
        return _rate(self.policyengine_pass_count, self.policyengine_case_count)

    @property
    def median_duration_seconds(self) -> float | None:
        if not self.durations_ms:
            return None
        return round(median(self.durations_ms) / 1000.0, 3)

    @property
    def mean_cost_usd(self) -> float | None:
        if not self.costs_usd:
            return None
        return round(mean(self.costs_usd), 6)


@dataclass
class EvalBoard:
    """A folded model-capability board."""

    suite_name: str
    corpus_identity: dict[str, object]
    cases: list[BoardCase]
    runners: list[BoardRunnerStats]
    cells: dict[tuple[int, str], BoardCell]
    sources: dict[str, str]
    incomplete_sources: list[str] = field(default_factory=list)
    mixed_toolchain_sources: list[str] = field(default_factory=list)
    execution_identity_sha256s: dict[str, str] = field(default_factory=dict)

    def ordered_runners(self) -> list[BoardRunnerStats]:
        """Runners by gate-pass rate, then passes, then speed, then name.

        Rate leads so a complete runner is never outranked by a partial
        runner's raw pass count under ``--allow-partial``; the count breaks
        rate ties between runners with different case counts.
        """
        return sorted(
            self.runners,
            key=lambda stats: (
                -stats.gate_pass_rate,
                -stats.gate_pass_count,
                stats.median_duration_seconds
                if stats.median_duration_seconds is not None
                else float("inf"),
                stats.runner,
            ),
        )


def resolve_board_input_path(raw: Path) -> Path:
    """Accept a results.json file or a suite output directory."""
    path = Path(raw)
    if path.is_dir():
        candidate = path / _RESULTS_FILE_NAME
        if not candidate.is_file():
            raise EvalBoardError(
                f"Suite output directory has no {_RESULTS_FILE_NAME}: {path}"
            )
        return candidate
    if not path.is_file():
        raise EvalBoardError(f"Suite results file not found: {path}")
    return path


def load_eval_suite_results(path: Path) -> dict:
    """Load one results.json payload with structural checks."""
    resolved = resolve_board_input_path(path)
    try:
        payload = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalBoardError(f"Could not read suite results {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvalBoardError(f"Suite results must be a JSON object: {resolved}")
    schema = payload.get("schema")
    if schema != SUPPORTED_RESULTS_SCHEMA:
        raise EvalBoardError(
            f"Suite results {resolved} carry schema {schema!r}; eval-board "
            f"folds only {SUPPORTED_RESULTS_SCHEMA!r} payloads"
        )
    for key in ("evidence", "results", "coverage"):
        if key not in payload:
            raise EvalBoardError(
                f"Suite results are missing the '{key}' section: {resolved}"
            )
    evidence = payload["evidence"]
    if not isinstance(evidence, dict) or not isinstance(evidence.get("manifest"), dict):
        raise EvalBoardError(f"Suite results carry no manifest evidence: {resolved}")
    evidence_schema = evidence.get("schema")
    if evidence_schema != SUPPORTED_EVIDENCE_SCHEMA:
        raise EvalBoardError(
            f"Suite results {resolved} carry evidence schema "
            f"{evidence_schema!r}; eval-board folds only "
            f"{SUPPORTED_EVIDENCE_SCHEMA!r} evidence"
        )
    evidence_sha256 = evidence.get("sha256")
    unsigned_evidence = dict(evidence)
    unsigned_evidence.pop("sha256", None)
    if not isinstance(
        evidence_sha256, str
    ) or evidence_sha256 != _canonical_json_sha256(unsigned_evidence):
        raise EvalBoardError(
            f"Suite results evidence digest is missing or does not match its "
            f"evidence payload: {resolved}"
        )
    return payload


def normalized_execution_identity(
    identity: object,
    *,
    location_keys: frozenset[str] = _LOCATION_ONLY_IDENTITY_KEYS,
) -> object:
    """Drop location-only fields so identical toolchains compare equal.

    Checkout paths (and digests computed over structures that embed them)
    differ across machines and directories without affecting scores; every
    other field — commits, content hashes, waiver digests, versions, the case
    budget, and runner timeouts — is score-affecting and must match exactly. The
    `policyengine_runtime`
    subtree uses the extended location-key set because its sealed-runtime
    identity embeds venv/stdlib/interpreter locations.
    """
    if isinstance(identity, dict):
        return {
            key: normalized_execution_identity(
                value,
                location_keys=(
                    _POLICYENGINE_LOCATION_ONLY_KEYS
                    if key == "policyengine_runtime"
                    else location_keys
                ),
            )
            for key, value in identity.items()
            if key not in location_keys and key not in _LOCATION_DEPENDENT_DIGEST_KEYS
        }
    if isinstance(identity, list):
        return [
            normalized_execution_identity(item, location_keys=location_keys)
            for item in identity
        ]
    return identity


def _canonical_json_sha256(payload: object) -> str:
    """Mirror the producer's canonical JSON digest.

    Byte-identical to `_canonical_json_sha256` in harness/evals.py and
    `_eval_suite_json_sha256` in cli.py; test_eval_board locks all three
    together.
    """
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _optional_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return _rate(numerator, denominator)


def _payload_case_identities(payload: dict, source: str) -> list[dict]:
    identities = payload["evidence"]["manifest"].get("case_identities")
    if not isinstance(identities, list) or not identities:
        raise EvalBoardError(f"Suite results carry no case identities: {source}")
    for position, identity in enumerate(identities, start=1):
        if not isinstance(identity, dict) or not (
            type(identity.get("index")) is int and identity["index"] == position
        ):
            raise EvalBoardError(
                f"Suite results case identities are malformed at position "
                f"{position}: {source}"
            )
    return identities


def _payload_suite_name(payload: dict, source: str) -> str:
    name = payload["evidence"]["manifest"].get("name")
    if not isinstance(name, str) or not name:
        raise EvalBoardError(f"Suite results carry no suite name: {source}")
    return name


_SHA256_HEX = frozenset("0123456789abcdef")


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _is_positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _payload_corpus_identity(payload: dict, source: str) -> dict:
    corpus = payload["evidence"].get("corpus")
    if (
        not isinstance(corpus, dict)
        or set(corpus)
        != {
            "corpus_release",
            "corpus_release_content_sha256",
            "corpus_release_selector_sha256",
        }
        or not isinstance(corpus.get("corpus_release"), str)
        or not corpus.get("corpus_release")
        or not _is_sha256_hex(corpus.get("corpus_release_content_sha256"))
        or not _is_sha256_hex(corpus.get("corpus_release_selector_sha256"))
    ):
        raise EvalBoardError(
            f"Suite results corpus release identity is missing or incomplete "
            f"(expected release name + content and selector digests): {source}"
        )
    return corpus


def _payload_runner_identities(payload: dict, source: str) -> list[dict]:
    identities = payload["evidence"].get("effective_runner_identities")
    if not isinstance(identities, list) or not identities:
        raise EvalBoardError(f"Suite results carry no runner identities: {source}")
    return identities


def _payload_execution_identity(payload: dict, source: str) -> tuple[dict, str]:
    identity = payload["evidence"].get("execution_identity")
    digest = payload["evidence"].get("execution_identity_sha256")
    if not isinstance(identity, dict) or not identity:
        raise EvalBoardError(f"Suite results carry no execution identity: {source}")
    schema = identity.get("schema")
    if schema != SUPPORTED_EXECUTION_IDENTITY_SCHEMA:
        raise EvalBoardError(
            f"Suite results execution identity carries schema {schema!r}; "
            f"eval-board understands only "
            f"{SUPPORTED_EXECUTION_IDENTITY_SCHEMA!r}: {source}"
        )
    runner_timeouts = identity.get("runner_timeouts")
    claude_timeout = (
        runner_timeouts.get("claude") if isinstance(runner_timeouts, dict) else None
    )
    claude_wall_seconds = (
        claude_timeout.get("wall_seconds") if isinstance(claude_timeout, dict) else None
    )
    codex_timeout = (
        runner_timeouts.get("codex") if isinstance(runner_timeouts, dict) else None
    )
    codex_short = (
        codex_timeout.get("short_source") if isinstance(codex_timeout, dict) else None
    )
    codex_long = (
        codex_timeout.get("long_source") if isinstance(codex_timeout, dict) else None
    )
    openai_timeout = (
        runner_timeouts.get("openai") if isinstance(runner_timeouts, dict) else None
    )
    if (
        not isinstance(runner_timeouts, dict)
        or set(runner_timeouts) != {"claude", "codex", "openai"}
        or not isinstance(claude_timeout, dict)
        or set(claude_timeout) != {"wall_seconds"}
        or not _is_positive_int(claude_wall_seconds)
        or not isinstance(codex_timeout, dict)
        or set(codex_timeout)
        != {"short_source", "long_source", "long_source_char_threshold"}
        or not isinstance(codex_short, dict)
        or set(codex_short) != {"wall_seconds", "idle_seconds"}
        or not _is_positive_int(codex_short.get("wall_seconds"))
        or not _is_positive_int(codex_short.get("idle_seconds"))
        or codex_short["idle_seconds"] > codex_short["wall_seconds"]
        or not isinstance(codex_long, dict)
        or set(codex_long) != {"wall_seconds", "idle_seconds"}
        or not _is_positive_int(codex_long.get("wall_seconds"))
        or not _is_positive_int(codex_long.get("idle_seconds"))
        or codex_long["idle_seconds"] > codex_long["wall_seconds"]
        or not _is_positive_int(codex_timeout.get("long_source_char_threshold"))
        or not isinstance(openai_timeout, dict)
        or set(openai_timeout) != {"request_connect_seconds", "request_read_seconds"}
        or not _is_positive_int(openai_timeout.get("request_connect_seconds"))
        or not _is_positive_int(openai_timeout.get("request_read_seconds"))
    ):
        raise EvalBoardError(
            "Suite results execution identity has a missing or malformed "
            f"runner timeout policy: {source}"
        )
    retry_policy = identity.get("timeout_retry_policy")
    if (
        not isinstance(retry_policy, dict)
        or set(retry_policy)
        != {
            "empty_artifact_max_attempts",
            "suite_max_attempts",
            "suite_retries_after_timeout",
            "openai_request_max_attempts",
            "openai_request_backoff_seconds",
        }
        or retry_policy.get("empty_artifact_max_attempts") != 2
        or not _is_positive_int(retry_policy.get("suite_max_attempts"))
        or retry_policy.get("suite_retries_after_timeout") is not False
        or retry_policy.get("openai_request_max_attempts") != 6
        or retry_policy.get("openai_request_backoff_seconds") != [1, 2, 4, 8, 10]
    ):
        raise EvalBoardError(
            "Suite results execution identity has a missing or malformed "
            f"timeout retry policy: {source}"
        )
    if not isinstance(digest, str) or not digest:
        raise EvalBoardError(
            f"Suite results carry no execution identity digest: {source}"
        )
    recomputed = _canonical_json_sha256(identity)
    if digest != recomputed:
        raise EvalBoardError(
            f"Suite results execution identity digest does not match its "
            f"identity payload: {source}"
        )
    return identity, digest


def _payload_completeness(
    payload: dict,
    source: str,
    *,
    case_count: int,
    runner_count: int,
    results: list,
) -> bool:
    """Verify the coverage section against the payload's own result rows."""
    coverage = payload.get("coverage")
    if not isinstance(coverage, dict):
        raise EvalBoardError(f"Suite results carry no coverage section: {source}")
    complete = coverage.get("complete")
    if not isinstance(complete, bool):
        raise EvalBoardError(
            f"Suite results coverage.complete must be a boolean: {source}"
        )
    completed_case_indexes = {
        result["eval_case"]["index"]
        for result in results
        if isinstance(result, dict)
        and isinstance(result.get("eval_case"), dict)
        and type(result["eval_case"].get("index")) is int
    }
    expectations = {
        "expected_case_count": case_count,
        "completed_case_count": len(completed_case_indexes),
        "expected_runner_count": runner_count,
        "expected_result_count": case_count * runner_count,
        "actual_result_count": len(results),
    }
    for key, expected in expectations.items():
        value = coverage.get(key)
        if type(value) is not int or value != expected:
            raise EvalBoardError(
                f"Suite results coverage.{key} is {value!r} but the payload "
                f"implies {expected}: {source}"
            )
    recorded_results_sha256 = coverage.get("results_sha256")
    if recorded_results_sha256 != _canonical_json_sha256(results):
        raise EvalBoardError(
            f"Suite results coverage.results_sha256 does not match the "
            f"result rows: {source}"
        )
    return complete


def _require_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise EvalBoardError(f"{context} must be a boolean, got {value!r}")
    return value


def _require_optional_bool(value: object, *, context: str) -> bool | None:
    if value is None:
        return None
    return _require_bool(value, context=context)


def _require_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvalBoardError(f"{context} must be an integer, got {value!r}")
    return value


def _require_optional_number(value: object, *, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvalBoardError(f"{context} must be a number, got {value!r}")
    numeric = float(value)
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        raise EvalBoardError(f"{context} must be finite, got {value!r}")
    return numeric


def _require_optional_nonnegative_number(
    value: object, *, context: str
) -> float | None:
    numeric = _require_optional_number(value, context=context)
    if numeric is not None and numeric < 0:
        raise EvalBoardError(f"{context} must be nonnegative, got {value!r}")
    return numeric


def _require_nonnegative_int(value: object, *, context: str) -> int:
    numeric = _require_int(value, context=context)
    if numeric < 0:
        raise EvalBoardError(f"{context} must be nonnegative, got {value!r}")
    return numeric


def _result_metrics(result: dict) -> dict | None:
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return None


def result_gate_pass(result: dict) -> bool:
    """The deterministic gate battery for one case x runner result."""
    if (
        result.get("success") is not True
        or result.get("error")
        or result.get("timed_out") is True
    ):
        return False
    metrics = _result_metrics(result)
    if metrics is None:
        return False
    return bool(
        metrics.get("compile_pass") is True
        and metrics.get("ci_pass") is True
        and metrics.get("ungrounded_numeric_count") == 0
    )


def _validate_result_types(result: dict, *, context: str) -> None:
    """Refuse malformed rows instead of reinterpreting them."""
    _require_bool(result.get("success"), context=f"{context} success")
    error = result.get("error")
    if error is not None and not isinstance(error, str):
        raise EvalBoardError(f"{context} error must be null or a string, got {error!r}")
    failure_kind = result.get("failure_kind")
    if failure_kind not in {None, "timeout", "validation", "error"}:
        raise EvalBoardError(
            f"{context} failure_kind must be null, timeout, validation, or error"
        )
    timed_out = result.get("timed_out")
    if not isinstance(timed_out, bool):
        raise EvalBoardError(f"{context} timed_out must be a boolean")
    if timed_out is not (failure_kind == "timeout"):
        raise EvalBoardError(f"{context} has inconsistent timeout classification")
    timeout_attempts = _require_nonnegative_int(
        result.get("timeout_attempts"),
        context=f"{context} timeout_attempts",
    )
    timeout_stage = result.get("timeout_stage")
    timeout_reason = result.get("timeout_reason")
    if timeout_stage is not None and not isinstance(timeout_stage, str):
        raise EvalBoardError(
            f"{context} timeout_stage must be null or a string, got {timeout_stage!r}"
        )
    if timeout_reason is not None and not isinstance(timeout_reason, str):
        raise EvalBoardError(
            f"{context} timeout_reason must be null or a string, got {timeout_reason!r}"
        )
    timeout_seconds = _require_optional_nonnegative_number(
        result.get("timeout_seconds"),
        context=f"{context} timeout_seconds",
    )
    if timeout_seconds == 0:
        raise EvalBoardError(f"{context} timeout_seconds must be positive when set")
    if timeout_attempts == 0 and any(
        value is not None for value in (timeout_stage, timeout_reason, timeout_seconds)
    ):
        raise EvalBoardError(f"{context} has timeout details without timeout attempts")
    if result.get("success") is True:
        if failure_kind is not None:
            raise EvalBoardError(f"{context} marks success with a failure_kind")
    elif failure_kind is None:
        raise EvalBoardError(f"{context} failure row has no failure_kind")
    _require_nonnegative_int(
        result.get("duration_ms"), context=f"{context} duration_ms"
    )
    _require_optional_nonnegative_number(
        result.get("estimated_cost_usd"),
        context=f"{context} estimated_cost_usd",
    )
    raw_metrics = result.get("metrics")
    if raw_metrics is not None and not isinstance(raw_metrics, dict):
        raise EvalBoardError(
            f"{context} metrics must be null or an object, got {raw_metrics!r}"
        )
    metrics = _result_metrics(result)
    if failure_kind == "timeout" and (timeout_attempts == 0 or metrics is not None):
        raise EvalBoardError(
            f"{context} timeout row must have attempts and no artifact metrics"
        )
    if failure_kind == "validation" and metrics is None:
        raise EvalBoardError(f"{context} validation failure has no artifact metrics")
    if metrics is None:
        return
    _require_bool(metrics.get("compile_pass"), context=f"{context} compile_pass")
    _require_bool(metrics.get("ci_pass"), context=f"{context} ci_pass")
    _require_nonnegative_int(
        metrics.get("ungrounded_numeric_count"),
        context=f"{context} ungrounded_numeric_count",
    )
    occurrences = _require_nonnegative_int(
        metrics.get("source_numeric_occurrence_count"),
        context=f"{context} source_numeric_occurrence_count",
    )
    covered = _require_nonnegative_int(
        metrics.get("covered_source_numeric_occurrence_count"),
        context=f"{context} covered_source_numeric_occurrence_count",
    )
    if covered > occurrences:
        raise EvalBoardError(
            f"{context} covers {covered} source numeric occurrences out of "
            f"only {occurrences}"
        )
    _require_optional_bool(
        metrics.get("generalist_review_pass"),
        context=f"{context} generalist_review_pass",
    )
    _require_optional_number(
        metrics.get("generalist_review_score"),
        context=f"{context} generalist_review_score",
    )
    _require_optional_bool(
        metrics.get("policyengine_pass"),
        context=f"{context} policyengine_pass",
    )
    _require_optional_number(
        metrics.get("policyengine_score"),
        context=f"{context} policyengine_score",
    )


def _validate_result_case_binding(
    eval_case: dict,
    reference_cases: list[dict],
    *,
    context: str,
) -> int:
    """Bind one result row to the manifest case identity it claims."""
    case_index = eval_case.get("index")
    if (
        isinstance(case_index, bool)
        or not isinstance(case_index, int)
        or not 1 <= case_index <= len(reference_cases)
    ):
        raise EvalBoardError(
            f"{context} names case index {case_index!r}, outside the manifest's "
            f"1..{len(reference_cases)} cases"
        )
    reference = reference_cases[case_index - 1]
    for field_name in ("name", "kind", "corpus_citation_path", "sha256"):
        if eval_case.get(field_name) != reference.get(field_name):
            raise EvalBoardError(
                f"{context} case identity field {field_name!r} "
                f"({eval_case.get(field_name)!r}) does not match the manifest "
                f"case at index {case_index} ({reference.get(field_name)!r})"
            )
    return case_index


def _cell_for_result(result: dict) -> BoardCell:
    duration_ms = result.get("duration_ms")
    if not isinstance(duration_ms, int) or isinstance(duration_ms, bool):
        duration_ms = None
    failure_kind = result.get("failure_kind")
    if failure_kind == "timeout" or result.get("timed_out") is True:
        error = result.get("error")
        detail = str(error)[:200] if error else "encoder or case budget timed out"
        return BoardCell(state="timeout", duration_ms=duration_ms, detail=detail)
    metrics = _result_metrics(result)
    failed: list[str] = []
    if metrics is not None:
        if metrics.get("compile_pass") is not True:
            failed.append("compile")
        if metrics.get("ci_pass") is not True:
            failed.append("ci")
        if metrics.get("ungrounded_numeric_count") != 0:
            failed.append(f"ungrounded={metrics.get('ungrounded_numeric_count')}")
    if failure_kind == "validation":
        return BoardCell(
            state="fail",
            duration_ms=duration_ms,
            detail=", ".join(failed)
            or str(result.get("error") or "validation failure"),
        )
    if failed:
        return BoardCell(
            state="fail",
            duration_ms=duration_ms,
            detail=", ".join(failed),
        )
    if (
        failure_kind == "error"
        or result.get("success") is not True
        or result.get("error")
    ):
        error = result.get("error")
        detail = str(error)[:200] if error else "encode did not succeed"
        return BoardCell(state="error", duration_ms=duration_ms, detail=detail)
    if metrics is None:
        return BoardCell(
            state="error",
            duration_ms=duration_ms,
            detail="no artifact metrics recorded",
        )
    if result_gate_pass(result):
        return BoardCell(state="pass", duration_ms=duration_ms)
    return BoardCell(
        state="fail",
        duration_ms=duration_ms,
        detail=", ".join(failed) or "gate failure",
    )


def fold_eval_board(
    inputs: list[Path],
    *,
    allow_partial: bool = False,
    allow_mixed_toolchains: bool = False,
) -> EvalBoard:
    """Fold one or more suite results payloads into a capability board."""
    if not inputs:
        raise EvalBoardError("eval-board needs at least one suite results input")

    reference_cases: list[dict] | None = None
    reference_suite: str | None = None
    reference_corpus: dict | None = None
    reference_execution: object = None
    reference_source = ""
    runner_sources: dict[str, str] = {}
    runner_identities: dict[str, dict] = {}
    runner_results: dict[str, dict[int, dict]] = {}
    sources: dict[str, str] = {}
    incomplete_sources: list[str] = []
    mixed_toolchain_sources: list[str] = []
    execution_identity_sha256s: dict[str, str] = {}

    for raw_path in inputs:
        resolved = resolve_board_input_path(Path(raw_path))
        source = str(resolved)
        payload = load_eval_suite_results(resolved)
        suite_name = _payload_suite_name(payload, source)
        case_identities = _payload_case_identities(payload, source)
        corpus_identity = _payload_corpus_identity(payload, source)
        execution_identity, execution_digest = _payload_execution_identity(
            payload, source
        )
        normalized_execution = normalized_execution_identity(execution_identity)
        execution_identity_sha256s[source] = execution_digest

        if reference_cases is None:
            reference_cases = case_identities
            reference_suite = suite_name
            reference_corpus = corpus_identity
            reference_execution = normalized_execution
            reference_source = source
        else:
            if suite_name != reference_suite:
                raise EvalBoardError(
                    "Suite results are not comparable: suite name "
                    f"{suite_name!r} in {source} does not match "
                    f"{reference_suite!r} in {reference_source}"
                )
            if case_identities != reference_cases:
                raise EvalBoardError(
                    "Suite results are not comparable: case identities in "
                    f"{source} do not match {reference_source}; boards fold "
                    "only runs of the identical case set"
                )
            if corpus_identity != reference_corpus:
                raise EvalBoardError(
                    "Suite results are not comparable: corpus release "
                    f"identity in {source} does not match {reference_source}"
                )
            if normalized_execution != reference_execution:
                if not allow_mixed_toolchains:
                    raise EvalBoardError(
                        "Suite results are not comparable: score-affecting "
                        f"execution identity in {source} does not match "
                        f"{reference_source} (encoder, rules engine, RuleSpec "
                        "content/toolchain/waivers, runner timeouts, or "
                        "PolicyEngine runtime differ; checkout locations are "
                        "ignored). Re-run on "
                        "one toolchain, or pass --allow-mixed-toolchains to "
                        "fold anyway with the mismatch recorded."
                    )
                mixed_toolchain_sources.append(source)

        payload_runner_names: set[str] = set()
        for identity in _payload_runner_identities(payload, source):
            if not isinstance(identity, dict):
                raise EvalBoardError(
                    f"Suite results carry a malformed runner identity: {source}"
                )
            name = identity.get("name")
            if not isinstance(name, str) or not name:
                raise EvalBoardError(
                    f"Suite results carry a malformed runner identity: {source}"
                )
            for identity_field in ("backend", "model"):
                declared_value = identity.get(identity_field)
                if not isinstance(declared_value, str) or not declared_value:
                    raise EvalBoardError(
                        f"Suite results declare runner {name!r} without a "
                        f"valid {identity_field}: {source}"
                    )
            if name in runner_sources:
                raise EvalBoardError(
                    f"Runner {name!r} appears in both {runner_sources[name]} "
                    f"and {source}; two runs of one runner are two boards, "
                    "not one — drop one input or rename the runner"
                )
            runner_sources[name] = source
            runner_identities[name] = identity
            runner_results[name] = {}
            payload_runner_names.add(name)

        results = payload.get("results")
        if not isinstance(results, list):
            raise EvalBoardError(f"Suite results carry no result rows: {source}")
        for position, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                raise EvalBoardError(f"Malformed result row in {source}")
            bound_digest = result.get(_RESULT_SHA256_FIELD)
            unsigned_row = dict(result)
            unsigned_row.pop(_RESULT_SHA256_FIELD, None)
            if not isinstance(
                bound_digest, str
            ) or bound_digest != _canonical_json_sha256(unsigned_row):
                raise EvalBoardError(
                    f"Result row #{position} in {source} is missing its "
                    f"{_RESULT_SHA256_FIELD} binding or does not match it"
                )
            runner = result.get("runner")
            if not isinstance(runner, str) or runner not in payload_runner_names:
                raise EvalBoardError(
                    f"Result row #{position} in {source} names runner "
                    f"{runner!r}, which this payload never declared"
                )
            declared = runner_identities[runner]
            for identity_field in ("backend", "model"):
                if result.get(identity_field) != declared.get(identity_field):
                    raise EvalBoardError(
                        f"Result row #{position} in {source} carries "
                        f"{identity_field} {result.get(identity_field)!r} but "
                        f"runner {runner!r} is declared as "
                        f"{declared.get(identity_field)!r}"
                    )
            eval_case = result.get("eval_case")
            if not isinstance(eval_case, dict):
                raise EvalBoardError(
                    f"Result row #{position} in {source} carries no case identity"
                )
            context = f"Result row #{position} in {source}"
            case_index = _validate_result_case_binding(
                eval_case,
                case_identities,
                context=context,
            )
            if case_index in runner_results[runner]:
                raise EvalBoardError(
                    f"Duplicate result for runner {runner!r} case "
                    f"#{case_index} in {source}"
                )
            _validate_result_types(result, context=context)
            runner_results[runner][case_index] = result

        complete = _payload_completeness(
            payload,
            source,
            case_count=len(case_identities),
            runner_count=len(payload_runner_names),
            results=results,
        )
        matrix_gaps: list[str] = []
        for name in sorted(payload_runner_names):
            missing = [
                str(index)
                for index in range(1, len(case_identities) + 1)
                if index not in runner_results[name]
            ]
            if missing:
                matrix_gaps.append(f"runner {name!r} case(s) {', '.join(missing)}")
        if complete and matrix_gaps:
            raise EvalBoardError(
                f"Suite results claim coverage.complete but the result matrix "
                f"is missing {'; '.join(matrix_gaps)}: {source}"
            )
        if not complete and not matrix_gaps:
            raise EvalBoardError(
                f"Suite results claim an incomplete run but carry a full "
                f"result matrix: {source}. Re-emit the payload through "
                "eval-suite rather than folding a contradictory coverage "
                "claim."
            )
        if not complete:
            if not allow_partial:
                raise EvalBoardError(
                    f"Suite results are incomplete: {source}. Finish the run "
                    "(eval-suite --resume) or pass --allow-partial to fold "
                    "what exists."
                )
            incomplete_sources.append(source)
        sources[source] = suite_name

    assert reference_cases is not None and reference_suite is not None
    assert reference_corpus is not None

    cases = [
        BoardCase(
            index=identity["index"],
            name=str(identity.get("name") or f"case-{position}"),
            kind=str(identity.get("kind") or "source"),
            corpus_citation_path=identity.get("corpus_citation_path"),
            sha256=identity.get("sha256"),
        )
        for position, identity in enumerate(reference_cases, start=1)
    ]

    cells: dict[tuple[int, str], BoardCell] = {}
    runner_stats: list[BoardRunnerStats] = []
    for name in sorted(runner_results):
        identity = runner_identities[name]
        stats = BoardRunnerStats(
            runner=name,
            backend=identity["backend"],
            model=identity["model"],
            source=runner_sources[name],
            cases_run=0,
            artifact_case_count=0,
            timeout_count=0,
            gate_pass_count=0,
            compile_pass_count=0,
            ci_pass_count=0,
            zero_ungrounded_count=0,
            success_count=0,
            source_numeric_occurrences=0,
            covered_source_numeric_occurrences=0,
            generalist_review_pass_count=0,
        )
        for case in cases:
            result = runner_results[name].get(case.index)
            if result is None:
                cells[(case.index, name)] = BoardCell(state="missing")
                continue
            cell = _cell_for_result(result)
            cells[(case.index, name)] = cell
            stats.cases_run += 1
            if cell.duration_ms is not None:
                stats.durations_ms.append(cell.duration_ms)
            cost = result.get("estimated_cost_usd")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                stats.costs_usd.append(float(cost))
            if result.get("success") is True and not result.get("error"):
                stats.success_count += 1
            if cell.state == "pass":
                stats.gate_pass_count += 1
            if cell.state == "timeout":
                stats.timeout_count += 1
            metrics = _result_metrics(result)
            if metrics is None:
                continue
            stats.artifact_case_count += 1
            if metrics.get("compile_pass") is True:
                stats.compile_pass_count += 1
            if metrics.get("ci_pass") is True:
                stats.ci_pass_count += 1
            if metrics.get("ungrounded_numeric_count") == 0:
                stats.zero_ungrounded_count += 1
            occurrences = metrics.get("source_numeric_occurrence_count")
            covered = metrics.get("covered_source_numeric_occurrence_count")
            if isinstance(occurrences, int) and isinstance(covered, int):
                stats.source_numeric_occurrences += occurrences
                stats.covered_source_numeric_occurrences += covered
            if metrics.get("generalist_review_pass") is True:
                stats.generalist_review_pass_count += 1
            score = metrics.get("generalist_review_score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                stats.generalist_review_scores.append(float(score))
            pe_pass = metrics.get("policyengine_pass")
            pe_score = metrics.get("policyengine_score")
            if pe_pass is not None or pe_score is not None:
                stats.policyengine_case_count += 1
                if pe_pass is True:
                    stats.policyengine_pass_count += 1
        runner_stats.append(stats)

    return EvalBoard(
        suite_name=reference_suite,
        corpus_identity=dict(reference_corpus),
        cases=cases,
        runners=runner_stats,
        cells=cells,
        sources=sources,
        incomplete_sources=incomplete_sources,
        mixed_toolchain_sources=mixed_toolchain_sources,
        execution_identity_sha256s=execution_identity_sha256s,
    )


_CELL_GLYPHS: dict[BoardCellState, str] = {
    "pass": "P",
    "fail": "F",
    "timeout": "T",
    "error": "E",
    "missing": "·",
}


def _format_percent(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}"


def _format_optional(value: float | None, template: str) -> str:
    if value is None:
        return "—"
    return template.format(value)


def eval_board_to_json(board: EvalBoard) -> dict:
    """A machine-readable board payload."""
    return {
        "schema": EVAL_BOARD_SCHEMA,
        "suite": board.suite_name,
        "corpus": board.corpus_identity,
        "sources": board.sources,
        "incomplete_sources": board.incomplete_sources,
        "mixed_toolchain_sources": board.mixed_toolchain_sources,
        "execution_identity_sha256s": board.execution_identity_sha256s,
        "cases": [
            {
                "index": case.index,
                "name": case.name,
                "kind": case.kind,
                "corpus_citation_path": case.corpus_citation_path,
                "sha256": case.sha256,
            }
            for case in board.cases
        ],
        "runners": [
            {
                "runner": stats.runner,
                "backend": stats.backend,
                "model": stats.model,
                "source": stats.source,
                "cases_run": stats.cases_run,
                "artifact_case_count": stats.artifact_case_count,
                "timeout_count": stats.timeout_count,
                "gate_pass_count": stats.gate_pass_count,
                "gate_pass_rate": stats.gate_pass_rate,
                "success_count": stats.success_count,
                "compile_pass_rate": stats.compile_pass_rate,
                "ci_pass_rate": stats.ci_pass_rate,
                "zero_ungrounded_rate": stats.zero_ungrounded_rate,
                "source_numeric_coverage_rate": stats.source_numeric_coverage_rate,
                "generalist_review_pass_rate": stats.generalist_review_pass_rate,
                "mean_generalist_review_score": stats.mean_generalist_review_score,
                "policyengine_case_count": stats.policyengine_case_count,
                "policyengine_pass_rate": stats.policyengine_pass_rate,
                "median_duration_seconds": stats.median_duration_seconds,
                "mean_cost_usd": stats.mean_cost_usd,
            }
            for stats in board.ordered_runners()
        ],
        "cells": [
            {
                "case_index": case.index,
                "case_name": case.name,
                "runner": stats.runner,
                "state": board.cells[(case.index, stats.runner)].state,
                "duration_ms": board.cells[(case.index, stats.runner)].duration_ms,
                "detail": board.cells[(case.index, stats.runner)].detail,
            }
            for case in board.cases
            for stats in board.ordered_runners()
        ],
    }


def eval_board_case_rows(board: EvalBoard) -> list[dict]:
    """Per-case grid rows for CSV export."""
    ordered = board.ordered_runners()
    rows: list[dict] = []
    for case in board.cases:
        row: dict[str, object] = {
            "case_index": case.index,
            "case_name": case.name,
            "corpus_citation_path": case.corpus_citation_path or "",
        }
        for stats in ordered:
            cell = board.cells[(case.index, stats.runner)]
            row[stats.runner] = cell.state
            row[f"{stats.runner}_seconds"] = (
                round(cell.duration_ms / 1000.0, 1)
                if cell.duration_ms is not None
                else ""
            )
            row[f"{stats.runner}_detail"] = cell.detail or ""
        rows.append(row)
    return rows


def render_eval_board_markdown(board: EvalBoard) -> str:
    """Render the leaderboard and per-case grid as markdown."""
    ordered = board.ordered_runners()
    lines: list[str] = []
    lines.append(f"# Eval board — {board.suite_name}")
    lines.append("")
    corpus_release = board.corpus_identity.get("corpus_release")
    if corpus_release:
        lines.append(f"Corpus release: `{corpus_release}`")
        lines.append("")
    if board.incomplete_sources:
        lines.append(
            "> Partial fold: incomplete suite runs were included with "
            "--allow-partial; missing cells render as `·` and rates cover "
            "only the cases each runner ran."
        )
        lines.append("")
    if board.mixed_toolchain_sources:
        lines.append(
            "> Mixed toolchains: these runs used a different score-affecting "
            "execution identity and were folded with "
            "--allow-mixed-toolchains: "
            + ", ".join(f"`{source}`" for source in board.mixed_toolchain_sources)
        )
        lines.append("")
    lines.append(
        "Gate pass = encode success + compile + CI + zero ungrounded "
        "numerics, per case. Compile/CI/grounded rates cover produced "
        "artifacts only; reviewer and oracle columns are advisory."
    )
    lines.append("")
    header = (
        "| runner | model | gate pass | timeouts | artifacts | compile | ci | grounded | "
        "src coverage | review | review score | oracle | median s | mean $ |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 14)
    for stats in ordered:
        oracle = (
            f"{stats.policyengine_pass_count}/{stats.policyengine_case_count}"
            if stats.policyengine_case_count
            else "—"
        )
        lines.append(
            "| {runner} | {model} | {gate} | {timeouts} | {artifacts} | "
            "{compile} | {ci} | {grounded} | "
            "{coverage} | {review} | {review_score} | {oracle} | {median} | "
            "{cost} |".format(
                runner=stats.runner,
                model=stats.model,
                gate=f"{stats.gate_pass_count}/{stats.cases_run} "
                f"({_format_percent(stats.gate_pass_rate)})",
                timeouts=stats.timeout_count,
                artifacts=stats.artifact_case_count,
                compile=_format_percent(stats.compile_pass_rate),
                ci=_format_percent(stats.ci_pass_rate),
                grounded=_format_percent(stats.zero_ungrounded_rate),
                coverage=_format_percent(stats.source_numeric_coverage_rate),
                review=_format_percent(stats.generalist_review_pass_rate),
                review_score=_format_optional(
                    stats.mean_generalist_review_score, "{:.2f}/10"
                ),
                oracle=oracle,
                median=_format_optional(stats.median_duration_seconds, "{:.0f}"),
                cost=_format_optional(stats.mean_cost_usd, "${:.4f}"),
            )
        )
    lines.append("")
    lines.append("## Per-case grid")
    lines.append("")
    lines.append(
        "P = gate pass, F = validation/gate fail, T = encoder/case timeout, "
        "E = encode error, · = not run."
    )
    lines.append("")
    grid_header = "| case | " + " | ".join(stats.runner for stats in ordered) + " |"
    lines.append(grid_header)
    lines.append("|" + "---|" * (len(ordered) + 1))
    for case in board.cases:
        cells = [
            _CELL_GLYPHS[board.cells[(case.index, stats.runner)].state]
            for stats in ordered
        ]
        lines.append(f"| {case.index:02d} {case.name} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_eval_board_text(board: EvalBoard) -> str:
    """Render a console summary."""
    ordered = board.ordered_runners()
    lines: list[str] = []
    lines.append(f"Suite: {board.suite_name}")
    corpus_release = board.corpus_identity.get("corpus_release")
    if corpus_release:
        lines.append(f"Corpus release: {corpus_release}")
    lines.append(f"Cases: {len(board.cases)}  Runners: {len(ordered)}")
    if board.incomplete_sources:
        lines.append(
            f"Partial fold: {len(board.incomplete_sources)} incomplete run(s) included"
        )
    if board.mixed_toolchain_sources:
        lines.append(
            f"Mixed toolchains: {len(board.mixed_toolchain_sources)} run(s) "
            "folded with --allow-mixed-toolchains"
        )
    lines.append("")
    name_width = max((len(stats.runner) for stats in ordered), default=6)
    for stats in ordered:
        lines.append(
            f"{stats.runner:<{name_width}}  "
            f"gate {stats.gate_pass_count}/{stats.cases_run} "
            f"({_format_percent(stats.gate_pass_rate)})  "
            f"T timeout {stats.timeout_count}  "
            f"artifacts {stats.artifact_case_count}  "
            f"compile {_format_percent(stats.compile_pass_rate)}  "
            f"ci {_format_percent(stats.ci_pass_rate)}  "
            f"grounded {_format_percent(stats.zero_ungrounded_rate)}  "
            f"median {_format_optional(stats.median_duration_seconds, '{:.0f}s')}"
        )
    lines.append("")
    lines.append("Grid (P pass / F fail / T timeout / E error / · not run):")
    for case in board.cases:
        cells = " ".join(
            _CELL_GLYPHS[board.cells[(case.index, stats.runner)].state]
            for stats in ordered
        )
        lines.append(f"  {case.index:02d} {case.name:<32} {cells}")
    lines.append("")
    lines.append("Runners: " + ", ".join(stats.runner for stats in ordered))
    return "\n".join(lines)
