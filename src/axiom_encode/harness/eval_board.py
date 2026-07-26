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
content/toolchain/waivers, per-case-runner generation/retry budget, backend
timeout policy, timeout retry policy, PolicyEngine runtime) — compared after
dropping location-only fields, so the same toolchain checked out at different
paths still folds. Requested effort may differ across distinct runner names,
while same-name runs must match. The manifest content hash may differ
(single-runner variants of one suite differ byte-wise but share case
identities), and runner sets may differ — that is the add-a-model path.
Duplicate runner names across payloads are refused rather than merged: two
runs of one runner are two boards, not one.

The board consumes canonical v8 suite payloads and refuses anything else:
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
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from statistics import mean, median
from typing import Literal

from axiom_encode.harness.policyengine_runtime import (
    POLICYENGINE_RUNTIME_PIN_SCHEMA,
    POLICYENGINE_RUNTIME_SCHEMA,
)

BoardCellState = Literal[
    "pass",
    "fail",
    "timeout",
    "context_overflow",
    "output_truncated",
    "integrity",
    "error",
    "missing",
]

_RESULTS_FILE_NAME = "results.json"

# The one producer schema this consumer understands. A new producer version
# must be reviewed here before boards fold it; test_eval_board locks this to
# the producer constant in cli.py.
SUPPORTED_RESULTS_SCHEMA = "axiom-encode/eval-suite-results/v8"

# The one execution-identity schema whose field semantics the normalizer
# below understands; test_eval_board locks this to the producer constant.
SUPPORTED_EXECUTION_IDENTITY_SCHEMA = "axiom-encode/eval-execution-identity/v5"

# The evidence schema this consumer understands; locked to the producer
# constant by test_eval_board.
SUPPORTED_EVIDENCE_SCHEMA = "axiom-encode/eval-suite-evidence/v5"
EVAL_BOARD_SCHEMA = "axiom-encode/eval-board/v3"
_RUNNER_EFFORTS_BY_BACKEND = {
    "claude": frozenset({"low", "medium", "high", "max"}),
    "codex": frozenset({"low", "medium", "high", "xhigh", "ultra"}),
    "openai": frozenset({"none", "low", "medium", "high", "xhigh", "max"}),
}
_OPENAI_REASONING_EFFORTS_BY_MODEL_PREFIX = (
    ("gpt-5.6", frozenset({"none", "low", "medium", "high", "xhigh", "max"})),
    ("gpt-5.5-pro", frozenset({"medium", "high", "xhigh"})),
    ("gpt-5.5", frozenset({"none", "low", "medium", "high", "xhigh"})),
    ("gpt-5.4-pro", frozenset({"medium", "high", "xhigh"})),
    ("gpt-5.4", frozenset({"none", "low", "medium", "high", "xhigh"})),
)
_INFRA_FAILURE_KINDS = frozenset({"context_overflow", "output_truncated", "integrity"})

# Every persisted result row carries this self-binding digest.
_RESULT_SHA256_FIELD = "result_sha256"
_RESULT_ADMISSION_SCHEMA = "axiom-encode/eval-result-admission/v2"

# These exact scopes are part of the v5 producer contract. Admission keeps
# independent literals so a producer scope change cannot silently widen or
# narrow what an existing board consumer accepts.
_ENCODER_GIT_PATHSPECS = ("src/axiom_encode", "pyproject.toml", "uv.lock")
_RULESPEC_TOOLCHAIN_PATHSPEC = ".axiom/toolchain.toml"
_RULESPEC_RUNTIME_PIN_PATHSPEC = ".axiom/policyengine-runtime.toml"
_RULESPEC_WAIVER_PATHSPEC = "known-validation-gaps.yaml"
_GITHUB_ORIGIN_REPOSITORY_RE = re.compile(r"github[.]com/[^/\s]+/[^/\s]+")

# Location-only identity fields: where a checkout lives never affects scores,
# so normalized execution identities drop these before comparison.
_LOCATION_ONLY_IDENTITY_KEYS = frozenset({"path", "toolchain_root", "repository_root"})

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
    requested_effort: str | None
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
    PolicyEngine's sealed-runtime paths are reduced to stable semantic anchors
    so import order and relative module topology remain score-affecting even
    when two equivalent runtimes live in different checkout directories.
    """
    if isinstance(identity, dict):
        return {
            key: (
                _normalized_policyengine_runtime(value)
                if key == "policyengine_runtime"
                else normalized_execution_identity(value, location_keys=location_keys)
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


def _normalized_policyengine_runtime(value: object) -> object:
    """Preserve sealed import topology while removing host-specific roots."""

    if value is None or not isinstance(value, dict):
        return value
    runtime = value.get("identity")
    if not isinstance(runtime, dict):
        return {
            key: normalized_execution_identity(item)
            for key, item in value.items()
            if key not in _LOCATION_DEPENDENT_DIGEST_KEYS
        }

    repository_root = runtime.get("repository_root")
    venv_root = runtime.get("venv_root")
    stdlib_root = runtime.get("stdlib_root")
    site_packages_root = runtime.get("site_packages_root")
    anchors = (
        ("<policyengine-site-packages>", site_packages_root),
        ("<policyengine-stdlib>", stdlib_root),
        ("<policyengine-venv>", venv_root),
        ("<policyengine-checkout>", repository_root),
    )

    def anchored_path(path: object) -> object:
        if not isinstance(path, str):
            return path
        for label, root in anchors:
            if not isinstance(root, str):
                continue
            relative = _relative_identity_path(path, root)
            if relative is None:
                continue
            return label if relative == "." else f"{label}/{relative}"
        return "<outside-policyengine-runtime>"

    normalized_runtime: dict[str, object] = {}
    for key, item in runtime.items():
        if key == "rulespec_runtime_pin_path":
            continue
        if key == "repository_root":
            normalized_runtime[key] = "<policyengine-checkout>"
        elif key == "venv_root":
            normalized_runtime[key] = _normalized_child_layout(
                item,
                repository_root,
                "<policyengine-checkout>",
            )
        elif key == "stdlib_root":
            normalized_runtime[key] = _normalized_child_layout(
                item,
                venv_root,
                "<policyengine-venv>",
            )
        elif key == "site_packages_root":
            normalized_runtime[key] = _normalized_child_layout(
                item,
                stdlib_root,
                "<policyengine-stdlib>",
            )
        elif key in {
            "python_executable",
            "python_prefix",
            "python_base_prefix",
            "python_exec_prefix",
            "python_base_exec_prefix",
        }:
            normalized_runtime[key] = anchored_path(item)
        elif key in {"initial_sys_path", "effective_sys_path"} and isinstance(
            item, list
        ):
            normalized_runtime[key] = [anchored_path(path) for path in item]
        elif key == "packages" and isinstance(item, dict):
            normalized_runtime[key] = {
                distribution: {
                    package_key: (
                        anchored_path(package_value)
                        if package_key in {"module_origin", "metadata_root"}
                        else normalized_execution_identity(package_value)
                    )
                    for package_key, package_value in package.items()
                }
                if isinstance(package, dict)
                else normalized_execution_identity(package)
                for distribution, package in item.items()
            }
        else:
            normalized_runtime[key] = normalized_execution_identity(item)
    return {"identity": normalized_runtime}


def _normalized_child_layout(
    path: object,
    root: object,
    root_label: str,
) -> object:
    """Return one stable path relative to its declared sealed-runtime parent."""

    if not isinstance(path, str) or not isinstance(root, str):
        return path
    relative = _relative_identity_path(path, root)
    if relative is None:
        return "<outside-policyengine-runtime>"
    return root_label if relative == "." else f"{root_label}/{relative}"


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
        if (
            not isinstance(identity, dict)
            or set(identity)
            != {
                "index",
                "name",
                "kind",
                "corpus_citation_path",
                "oracle",
                "sha256",
            }
            or type(identity.get("index")) is not int
            or identity["index"] != position
            or not _is_nonempty_string(identity.get("name"))
            or identity.get("kind") not in {"citation", "source"}
            or not _is_nonempty_string(identity.get("corpus_citation_path"))
            or identity.get("oracle") not in {"none", "policyengine"}
            or not _is_sha256_hex(identity.get("sha256"))
        ):
            raise EvalBoardError(
                f"Suite results case identities are malformed at position "
                f"{position}: {source}"
            )
    return identities


def _payload_suite_name(payload: dict, source: str) -> str:
    manifest = payload["evidence"]["manifest"]
    name = manifest.get("name")
    if (
        set(manifest) != {"name", "path", "content_sha256", "case_identities"}
        or not _is_nonempty_string(name)
        or not _is_nonempty_string(manifest.get("path"))
        or not _is_sha256_hex(manifest.get("content_sha256"))
    ):
        raise EvalBoardError(f"Suite results carry no suite name: {source}")
    return name


def _payload_run_identity(payload: dict, source: str) -> dict:
    """Require the immutable producer run identity bound into every row."""

    run = payload["evidence"].get("run")
    if not isinstance(run, dict) or set(run) != {"id", "started_at"}:
        raise EvalBoardError(f"Suite results carry a malformed run identity: {source}")
    run_id = run.get("id")
    started_at = run.get("started_at")
    try:
        parsed_run_id = uuid.UUID(run_id) if isinstance(run_id, str) else None
    except ValueError:
        parsed_run_id = None
    try:
        parsed_started_at = (
            datetime.fromisoformat(started_at)
            if isinstance(started_at, str) and started_at
            else None
        )
    except ValueError:
        parsed_started_at = None
    if (
        parsed_run_id is None
        or parsed_run_id.version != 4
        or str(parsed_run_id) != run_id
        or parsed_started_at is None
        or parsed_started_at.tzinfo is None
        or parsed_started_at.utcoffset() is None
    ):
        raise EvalBoardError(f"Suite results carry a malformed run identity: {source}")
    return run


_SHA256_HEX = frozenset("0123456789abcdef")


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _is_positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _is_nonnegative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _is_git_object_hex(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) in {40, 64} and set(value) <= _SHA256_HEX
    )


def _valid_origin_repository(value: object) -> bool:
    """Accept only the producer's normalized GitHub repository spelling."""

    return (
        isinstance(value, str)
        and _GITHUB_ORIGIN_REPOSITORY_RE.fullmatch(value) is not None
    )


def _valid_checkout_execution_identity(
    value: object,
    *,
    require_version: bool,
    expected_git_pathspecs: tuple[str, ...] | None,
) -> bool:
    """Accept exactly the git/tree identity union emitted by the producer."""

    if not isinstance(value, dict):
        return False
    version_keys = {"version"} if require_version else set()
    if require_version and not _is_nonempty_string(value.get("version")):
        return False
    kind = value.get("kind")
    if kind == "git":
        required_keys = {
            "kind",
            "path",
            "commit",
            "origin_repository",
            "dirty",
            "working_tree_sha256",
            *version_keys,
        }
        if expected_git_pathspecs is not None:
            required_keys.add("pathspecs")
        if set(value) != required_keys:
            return False
        origin_repository = value.get("origin_repository")
        return (
            _is_nonempty_string(value.get("path"))
            and _is_git_object_hex(value.get("commit"))
            and (
                origin_repository is None or _valid_origin_repository(origin_repository)
            )
            and type(value.get("dirty")) is bool
            and _is_sha256_hex(value.get("working_tree_sha256"))
            and (
                expected_git_pathspecs is None
                or value.get("pathspecs") == list(expected_git_pathspecs)
            )
        )
    if kind == "tree":
        required_keys = {
            "kind",
            "path",
            "state",
            "tree_sha256",
            "file_count",
            *version_keys,
        }
        if set(value) != required_keys:
            return False
        state = value.get("state")
        file_count = value.get("file_count")
        return (
            _is_nonempty_string(value.get("path"))
            and state in {"missing", "file", "directory"}
            and _is_sha256_hex(value.get("tree_sha256"))
            and _is_nonnegative_int(file_count)
            and (state != "missing" or file_count == 0)
            and (state != "file" or file_count == 1)
        )
    return False


def _relative_identity_path(path: str, root: str) -> str | None:
    """Return a lexical child path for either POSIX or Windows identities."""

    path_classes = (
        (PureWindowsPath, PurePosixPath)
        if "\\" in path or "\\" in root
        else (PurePosixPath, PureWindowsPath)
    )
    for path_class in path_classes:
        candidate = path_class(path)
        parent = path_class(root)
        try:
            relative = candidate.relative_to(parent)
        except ValueError:
            continue
        if ".." in relative.parts:
            continue
        return relative.as_posix()
    return None


def _rulespec_root_topology(path: object, toolchain_root: object) -> str | None:
    """Return the jurisdiction for one canonical direct checkout child."""

    if not isinstance(path, str) or not isinstance(toolchain_root, str):
        return None
    windows_paths = "\\" in path or "\\" in toolchain_root
    path_class = PureWindowsPath if windows_paths else PurePosixPath
    candidate = path_class(path)
    checkout = path_class(toolchain_root)
    canonical_candidate = str(candidate) if windows_paths else candidate.as_posix()
    canonical_checkout = str(checkout) if windows_paths else checkout.as_posix()
    if (
        not candidate.is_absolute()
        or not checkout.is_absolute()
        or canonical_candidate != path
        or canonical_checkout != toolchain_root
        or (
            not windows_paths
            and (path.startswith("//") or toolchain_root.startswith("//"))
        )
        or ".." in candidate.parts
        or ".." in checkout.parts
        or candidate.parent != checkout
    ):
        return None
    checkout_match = re.fullmatch(r"rulespec-([a-z]{2})", checkout.name)
    if checkout_match is None:
        return None
    country = checkout_match.group(1)
    jurisdiction = candidate.name
    if re.fullmatch(rf"{re.escape(country)}(?:-[a-z0-9]+)*", jurisdiction) is None:
        return None
    return jurisdiction


def _valid_rulespec_root_execution_identity(value: object) -> bool:
    """Validate the complete RuleSpec root identity before path normalization."""

    if not isinstance(value, dict) or set(value) != {
        "path",
        "content_state",
        "content_sha256",
        "file_count",
        "toolchain_root",
        "checkout_identity",
        "toolchain_contract_sha256",
        "policyengine_runtime_pin_sha256",
        "validation_waiver_set_sha256",
    }:
        return False
    file_count = value.get("file_count")
    path = value.get("path")
    toolchain_root = value.get("toolchain_root")
    if not _is_nonempty_string(path) or not _is_nonempty_string(toolchain_root):
        return False
    jurisdiction = _rulespec_root_topology(path, toolchain_root)
    if jurisdiction is None:
        return False
    expected_pathspecs = tuple(
        dict.fromkeys(
            (
                jurisdiction,
                _RULESPEC_TOOLCHAIN_PATHSPEC,
                _RULESPEC_RUNTIME_PIN_PATHSPEC,
                _RULESPEC_WAIVER_PATHSPEC,
            )
        )
    )
    checkout_identity = value.get("checkout_identity")
    runtime_pin_digest = value.get("policyengine_runtime_pin_sha256")
    return (
        value.get("content_state") == "directory"
        and _is_sha256_hex(value.get("content_sha256"))
        and _is_nonnegative_int(file_count)
        and isinstance(checkout_identity, dict)
        and checkout_identity.get("path") == toolchain_root
        and _valid_checkout_execution_identity(
            checkout_identity,
            require_version=False,
            expected_git_pathspecs=expected_pathspecs,
        )
        and _is_sha256_hex(value.get("toolchain_contract_sha256"))
        and (runtime_pin_digest is None or _is_sha256_hex(runtime_pin_digest))
        and _is_sha256_hex(value.get("validation_waiver_set_sha256"))
    )


def _valid_policyengine_package_identity(
    value: object,
    *,
    distribution: str,
    version: str,
) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"distribution", "version", "module_origin", "metadata_root"}
        and value.get("distribution") == distribution
        and value.get("version") == version
        and _is_nonempty_string(value.get("module_origin"))
        and _is_nonempty_string(value.get("metadata_root"))
    )


def _posix_identity_relative(path: object, root: object) -> str | None:
    """Return a canonical lexical relative path for producer-emitted POSIX paths."""

    if not isinstance(path, str) or not isinstance(root, str):
        return None
    candidate = PurePosixPath(path)
    parent = PurePosixPath(root)
    if (
        not candidate.is_absolute()
        or not parent.is_absolute()
        or candidate.as_posix() != path
        or parent.as_posix() != root
        or ".." in candidate.parts
        or ".." in parent.parts
    ):
        return None
    try:
        relative = candidate.relative_to(parent)
    except ValueError:
        return None
    if ".." in relative.parts:
        return None
    return relative.as_posix()


def _valid_policyengine_runtime_path_topology(
    value: dict,
    *,
    country: str,
) -> bool:
    """Mirror the producer's sealed-root and trusted-import-path invariants."""

    repository_root = value.get("repository_root")
    venv_root = value.get("venv_root")
    stdlib_root = value.get("stdlib_root")
    site_packages_root = value.get("site_packages_root")
    python_executable = value.get("python_executable")
    python_version = value.get("python_version")
    pin_path = value.get("rulespec_runtime_pin_path")
    python_components = (
        python_version.split(".") if isinstance(python_version, str) else []
    )
    expected_stdlib_path = (
        f"lib/python{python_components[0]}.{python_components[1]}"
        if len(python_components) == 3
        else None
    )
    pin = PurePosixPath(pin_path) if isinstance(pin_path, str) else None
    if (
        not isinstance(repository_root, str)
        or PurePosixPath(repository_root).name != f"policyengine-{country}"
        or _posix_identity_relative(venv_root, repository_root) != ".venv"
        or _posix_identity_relative(stdlib_root, venv_root) != expected_stdlib_path
        or _posix_identity_relative(site_packages_root, stdlib_root) != "site-packages"
        or _posix_identity_relative(python_executable, venv_root) != "bin/python"
        or pin is None
        or not pin.is_absolute()
        or pin.parts[-3:]
        != (
            f"rulespec-{country}",
            ".axiom",
            "policyengine-runtime.toml",
        )
    ):
        return False

    for field_name in (
        "python_prefix",
        "python_base_prefix",
        "python_exec_prefix",
        "python_base_exec_prefix",
    ):
        if _posix_identity_relative(value.get(field_name), venv_root) is None:
            return False

    initial_sys_path = value.get("initial_sys_path")
    effective_sys_path = value.get("effective_sys_path")
    if (
        not isinstance(initial_sys_path, list)
        or not initial_sys_path
        or any(
            _posix_identity_relative(path, venv_root) is None
            for path in initial_sys_path
        )
        or effective_sys_path
        != [repository_root, site_packages_root, *initial_sys_path]
    ):
        return False

    packages = value.get("packages")
    country_distribution = f"policyengine-{country}"
    if not isinstance(packages, dict):
        return False
    country_package = packages.get(country_distribution)
    core_package = packages.get("policyengine-core")
    if not isinstance(country_package, dict) or not isinstance(core_package, dict):
        return False
    return (
        _posix_identity_relative(
            country_package.get("module_origin"),
            repository_root,
        )
        is not None
        and _posix_identity_relative(
            core_package.get("module_origin"),
            site_packages_root,
        )
        is not None
        and all(
            _posix_identity_relative(
                package.get("metadata_root"),
                site_packages_root,
            )
            is not None
            for package in (country_package, core_package)
        )
    )


def _valid_policyengine_runtime_identity(value: object) -> bool:
    """Accept exactly the sealed runtime-v2 identity emitted by the producer."""

    if not isinstance(value, dict) or set(value) != {
        "schema",
        "country",
        "official_repository_url",
        "trusted_git_commit",
        "official_tree_sha256",
        "official_tree_file_count",
        "official_tree_byte_count",
        "rulespec_runtime_pin_path",
        "rulespec_runtime_pin_schema",
        "rulespec_runtime_pin_sha256",
        "repository_root",
        "checkout_execution_tree_sha256",
        "checkout_execution_file_count",
        "checkout_execution_byte_count",
        "venv_root",
        "venv_execution_tree_sha256",
        "venv_execution_file_count",
        "venv_execution_byte_count",
        "stdlib_root",
        "site_packages_root",
        "pyproject_sha256",
        "uv_lock_sha256",
        "locked_versions",
        "python_version",
        "python_implementation",
        "python_executable",
        "python_prefix",
        "python_base_prefix",
        "python_exec_prefix",
        "python_base_exec_prefix",
        "initial_sys_path",
        "effective_sys_path",
        "isolated",
        "no_site",
        "packages",
    }:
        return False
    country = value.get("country")
    country_package = f"policyengine-{country}"
    locked_versions = value.get("locked_versions")
    packages = value.get("packages")
    python_version = value.get("python_version")
    initial_sys_path = value.get("initial_sys_path")
    effective_sys_path = value.get("effective_sys_path")
    location_fields = (
        "rulespec_runtime_pin_path",
        "repository_root",
        "venv_root",
        "stdlib_root",
        "site_packages_root",
        "python_executable",
        "python_prefix",
        "python_base_prefix",
        "python_exec_prefix",
        "python_base_exec_prefix",
    )
    digest_fields = (
        "official_tree_sha256",
        "rulespec_runtime_pin_sha256",
        "checkout_execution_tree_sha256",
        "venv_execution_tree_sha256",
        "pyproject_sha256",
        "uv_lock_sha256",
    )
    count_fields = (
        "official_tree_file_count",
        "official_tree_byte_count",
        "checkout_execution_file_count",
        "checkout_execution_byte_count",
        "venv_execution_file_count",
        "venv_execution_byte_count",
    )
    if (
        value.get("schema") != POLICYENGINE_RUNTIME_SCHEMA
        or country not in {"us", "uk"}
        or value.get("official_repository_url")
        != f"https://github.com/PolicyEngine/policyengine-{country}.git"
        or not (
            isinstance(value.get("trusted_git_commit"), str)
            and len(value["trusted_git_commit"]) == 40
            and set(value["trusted_git_commit"]) <= _SHA256_HEX
        )
        or value.get("rulespec_runtime_pin_schema") != POLICYENGINE_RUNTIME_PIN_SCHEMA
        or any(not _is_sha256_hex(value.get(field)) for field in digest_fields)
        or any(not _is_positive_int(value.get(field)) for field in count_fields)
        or value.get("official_tree_file_count")
        != value.get("checkout_execution_file_count")
        or value.get("official_tree_byte_count")
        != value.get("checkout_execution_byte_count")
        or any(not _is_nonempty_string(value.get(field)) for field in location_fields)
        or not isinstance(locked_versions, dict)
        or set(locked_versions) != {"policyengine-core", country_package}
        or any(not _is_nonempty_string(version) for version in locked_versions.values())
        or not isinstance(python_version, str)
        or len(python_version.split(".")) != 3
        or any(
            not component or not component.isdigit()
            for component in python_version.split(".")
        )
        or value.get("python_implementation") != "cpython"
        or type(value.get("isolated")) is not int
        or value["isolated"] != 1
        or type(value.get("no_site")) is not int
        or value["no_site"] != 1
        or not isinstance(initial_sys_path, list)
        or not initial_sys_path
        or any(not _is_nonempty_string(path) for path in initial_sys_path)
        or not isinstance(effective_sys_path, list)
        or any(not _is_nonempty_string(path) for path in effective_sys_path)
        or not isinstance(packages, dict)
        or set(packages) != {"policyengine-core", country_package}
        or not _valid_policyengine_runtime_path_topology(value, country=country)
    ):
        return False
    return all(
        _valid_policyengine_package_identity(
            packages[distribution],
            distribution=distribution,
            version=locked_versions[distribution],
        )
        for distribution in ("policyengine-core", country_package)
    )


def _valid_policyengine_runtime_wrapper(value: object) -> bool:
    """Validate both the wrapper shape and its binding to the runtime identity."""

    if value is None:
        return True
    if (
        not isinstance(value, dict)
        or set(value) != {"identity", "sha256"}
        or not _valid_policyengine_runtime_identity(value.get("identity"))
        or not _is_sha256_hex(value.get("sha256"))
    ):
        return False
    return value["sha256"] == _canonical_json_sha256(value["identity"])


def _valid_policyengine_rulespec_binding(
    runtime_wrapper: object,
    rulespec_roots: list[object],
) -> bool:
    """Bind a sealed runtime pin to an exposed producer-owned RuleSpec checkout."""

    if runtime_wrapper is None:
        return True
    if not isinstance(runtime_wrapper, dict):
        return False
    runtime = runtime_wrapper.get("identity")
    if not isinstance(runtime, dict):
        return False
    country = runtime.get("country")
    pin_path = runtime.get("rulespec_runtime_pin_path")
    if not isinstance(country, str) or not isinstance(pin_path, str):
        return False
    for root in rulespec_roots:
        if not isinstance(root, dict):
            continue
        toolchain_root = root.get("toolchain_root")
        if (
            not isinstance(toolchain_root, str)
            or PurePosixPath(toolchain_root).name != f"rulespec-{country}"
        ):
            continue
        expected_pin = (
            PurePosixPath(toolchain_root) / ".axiom" / "policyengine-runtime.toml"
        ).as_posix()
        if (
            pin_path == expected_pin
            and root.get("policyengine_runtime_pin_sha256")
            == runtime.get("rulespec_runtime_pin_sha256")
            and _posix_identity_relative(pin_path, toolchain_root)
            == ".axiom/policyengine-runtime.toml"
        ):
            return True
    return False


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
    names: set[str] = set()
    for identity in identities:
        name = identity.get("name") if isinstance(identity, dict) else None
        if _is_nonempty_string(name):
            backend = identity.get("backend")
            if backend not in {"claude", "codex", "openai"}:
                raise EvalBoardError(
                    f"Suite results declare runner {name!r} without a "
                    f"valid backend: {source}"
                )
            if not _is_nonempty_string(identity.get("model")):
                raise EvalBoardError(
                    f"Suite results declare runner {name!r} without a "
                    f"valid model: {source}"
                )
        if (
            not isinstance(identity, dict)
            or set(identity) != {"name", "backend", "model"}
            or not _is_nonempty_string(name)
            or identity["name"] in names
        ):
            raise EvalBoardError(
                "Suite results carry a malformed runner identity without a "
                f"valid backend, model, or unique name: {source}"
            )
        names.add(identity["name"])
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
    axiom_encode = identity.get("axiom_encode")
    axiom_rules_engine = identity.get("axiom_rules_engine")
    rulespec_roots = identity.get("rulespec_roots")
    policyengine_runtime = identity.get("policyengine_runtime")
    if (
        "policyengine_runtime" not in identity
        or not _valid_checkout_execution_identity(
            axiom_encode,
            require_version=True,
            expected_git_pathspecs=_ENCODER_GIT_PATHSPECS,
        )
        or not _valid_checkout_execution_identity(
            axiom_rules_engine,
            require_version=False,
            expected_git_pathspecs=None,
        )
        or not isinstance(rulespec_roots, list)
        or not rulespec_roots
        or any(
            not _valid_rulespec_root_execution_identity(root) for root in rulespec_roots
        )
        or not _valid_policyengine_runtime_wrapper(policyengine_runtime)
        or not _valid_policyengine_rulespec_binding(
            policyengine_runtime,
            rulespec_roots,
        )
    ):
        raise EvalBoardError(
            "Suite results execution identity has missing or malformed core "
            f"toolchain fields: {source}"
        )
    case_timeout_seconds = identity.get("case_timeout_seconds")
    if not _is_positive_int(case_timeout_seconds):
        raise EvalBoardError(
            "Suite results execution identity has a missing or malformed "
            f"generation/retry case timeout: {source}"
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
    runner_identities = _payload_runner_identities(payload, source)
    runner_efforts = identity.get("runner_efforts")
    if not isinstance(runner_efforts, list) or len(runner_efforts) != len(
        runner_identities
    ):
        raise EvalBoardError(
            "Suite results execution identity has missing or malformed "
            f"requested effort declarations: {source}"
        )
    for runner_effort, runner_identity in zip(
        runner_efforts,
        runner_identities,
        strict=True,
    ):
        backend = runner_identity["backend"]
        model = runner_identity["model"]
        accepted_efforts = _runner_efforts_for_backend_model(backend, model)
        requested_effort = (
            runner_effort.get("requested_effort")
            if isinstance(runner_effort, dict)
            else None
        )
        uses_receiver_default = (
            runner_effort.get("uses_receiver_default")
            if isinstance(runner_effort, dict)
            else None
        )
        if (
            not isinstance(runner_effort, dict)
            or set(runner_effort)
            != {"name", "requested_effort", "uses_receiver_default"}
            or runner_effort.get("name") != runner_identity["name"]
            or (
                requested_effort is not None
                and (
                    not isinstance(requested_effort, str)
                    or requested_effort not in accepted_efforts
                )
            )
            or uses_receiver_default is not (requested_effort is None)
        ):
            raise EvalBoardError(
                "Suite results execution identity has a malformed requested "
                f"effort for runner {runner_identity['name']!r}: {source}"
            )
    receiver_environments = identity.get("receiver_environments")
    expected_local_backends = {
        runner_identity["backend"]
        for runner_identity in runner_identities
        if runner_identity["backend"] in {"claude", "codex"}
    }
    if (
        not isinstance(receiver_environments, dict)
        or set(receiver_environments) != expected_local_backends
    ):
        raise EvalBoardError(
            "Suite results execution identity has missing, extra, or mismatched "
            f"receiver environments: {source}"
        )
    for backend, environment in receiver_environments.items():
        if (
            not isinstance(environment, dict)
            or set(environment) != {"cli_version", "launcher_sha256", "native_sha256"}
            or not _is_nonempty_string(environment.get("cli_version"))
            or not _is_sha256_hex(environment.get("launcher_sha256"))
            or not _is_sha256_hex(environment.get("native_sha256"))
        ):
            raise EvalBoardError(
                "Suite results execution identity has a malformed receiver "
                f"environment for {backend!r}: {source}"
            )
    if set(identity) != {
        "schema",
        "runner_efforts",
        "receiver_environments",
        "case_timeout_seconds",
        "runner_timeouts",
        "timeout_retry_policy",
        "axiom_encode",
        "axiom_rules_engine",
        "policyengine_runtime",
        "rulespec_roots",
    }:
        raise EvalBoardError(
            f"Suite results execution identity has unexpected v5 fields: {source}"
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


def _runner_efforts_for_backend_model(
    backend: str,
    model: str,
) -> frozenset[str]:
    """Return receiver-supported explicit effort values for a runner."""

    if backend != "openai":
        return _RUNNER_EFFORTS_BY_BACKEND.get(backend, frozenset())
    for prefix, efforts in _OPENAI_REASONING_EFFORTS_BY_MODEL_PREFIX:
        if model == prefix or model.startswith(f"{prefix}-"):
            return efforts
    return frozenset()


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


def _validate_result_artifact_bindings(result: dict, *, context: str) -> None:
    """Mirror the producer's path/digest and generated-artifact invariants."""

    bound_fields: set[str] = set()
    for path_field, digest_field, label in (
        ("output_file", "generated_output_sha256", "generated RuleSpec"),
        ("trace_file", "trace_sha256", "model trace"),
        (
            "context_manifest_file",
            "context_manifest_sha256",
            "context manifest",
        ),
        (
            "verdict_file",
            "verdict_sha256",
            "validator verdict evidence",
        ),
    ):
        if digest_field not in result:
            if path_field == "verdict_file" and "verdict_file" not in result:
                continue
            raise EvalBoardError(
                f"{context} is missing immutable {label} digest {digest_field!r}"
            )
        raw_path = result.get(path_field)
        digest = result.get(digest_field)
        if not isinstance(raw_path, str):
            raise EvalBoardError(f"{context} has a malformed {label} path")
        if digest is None:
            if raw_path:
                raise EvalBoardError(
                    f"{context} has a {label} path without its SHA-256 digest"
                )
            continue
        if not _is_sha256_hex(digest):
            raise EvalBoardError(f"{context} has a malformed {label} SHA-256 digest")
        if not raw_path:
            raise EvalBoardError(
                f"{context} has a {label} SHA-256 digest without its path"
            )
        bound_fields.add(path_field)

    if (
        result.get("success") is True or isinstance(result.get("metrics"), dict)
    ) and "output_file" not in bound_fields:
        raise EvalBoardError(f"{context} has no content-bound generated RuleSpec")
    generation_bound_fields = bound_fields - {"verdict_file"}
    if generation_bound_fields and not {
        "trace_file",
        "context_manifest_file",
    }.issubset(bound_fields):
        raise EvalBoardError(
            f"{context} is missing its content-bound trace or context manifest"
        )


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


def _validate_result_effective_environment(result: dict, *, context: str) -> None:
    """Bind receiver metadata to the backend that produced this v8 row."""

    string_fields = (
        "claude_cli_version",
        "codex_cli_version",
        "openai_endpoint",
        "openai_response_model_id",
        "openai_service_tier",
    )
    for field_name in string_fields:
        value = result.get(field_name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise EvalBoardError(
                f"{context} {field_name} must be null or a nonempty string"
            )

    local_digest_fields = (
        "claude_cli_launcher_sha256",
        "claude_cli_native_sha256",
        "codex_cli_launcher_sha256",
        "codex_cli_native_sha256",
    )
    for field_name in local_digest_fields:
        value = result.get(field_name)
        if value is not None and not _is_sha256_hex(value):
            raise EvalBoardError(
                f"{context} {field_name} must be null or 64 lowercase hex characters"
            )
    if "codex_cli_sha256" in result:
        raise EvalBoardError(
            f"{context} carries legacy codex_cli_sha256 in a v8 result"
        )
    openai_max_output_tokens = result.get("openai_max_output_tokens")
    if openai_max_output_tokens is not None and not _is_positive_int(
        openai_max_output_tokens
    ):
        raise EvalBoardError(
            f"{context} openai_max_output_tokens must be null or a positive integer"
        )

    backend = result.get("backend")
    claude_fields_present = any(
        result.get(field_name) is not None
        for field_name in (
            "claude_cli_version",
            "claude_cli_launcher_sha256",
            "claude_cli_native_sha256",
        )
    )
    codex_fields_present = any(
        result.get(field_name) is not None
        for field_name in (
            "codex_cli_version",
            "codex_cli_launcher_sha256",
            "codex_cli_native_sha256",
        )
    )
    openai_fields_present = any(
        result.get(field_name) is not None
        for field_name in (
            "openai_endpoint",
            "openai_response_model_id",
            "openai_service_tier",
            "openai_max_output_tokens",
        )
    )
    if (
        (claude_fields_present and backend != "claude")
        or (codex_fields_present and backend != "codex")
        or (openai_fields_present and backend != "openai")
    ):
        raise EvalBoardError(
            f"{context} effective-environment fields do not match its backend"
        )

    if backend in {"claude", "codex"}:
        version_field = f"{backend}_cli_version"
        if not _is_nonempty_string(result.get(version_field)):
            raise EvalBoardError(f"{context} requires {version_field}")
        for digest_field in (
            f"{backend}_cli_launcher_sha256",
            f"{backend}_cli_native_sha256",
        ):
            if not _is_sha256_hex(result.get(digest_field)):
                raise EvalBoardError(f"{context} requires {digest_field}")
    if backend == "openai":
        if not _is_nonempty_string(result.get("openai_endpoint")):
            raise EvalBoardError(f"{context} requires openai_endpoint")
        if not _is_positive_int(result.get("openai_max_output_tokens")):
            raise EvalBoardError(f"{context} requires openai_max_output_tokens")
        has_generated_artifact = bool(result.get("output_file")) or isinstance(
            result.get("metrics"), dict
        )
        if has_generated_artifact and not _is_nonempty_string(
            result.get("openai_response_model_id")
        ):
            raise EvalBoardError(f"{context} requires openai_response_model_id")


def _validate_result_receiver_identity_binding(
    result: dict,
    *,
    execution_identity: dict,
    context: str,
) -> None:
    """Require each local row to match its suite-preflighted receiver."""

    backend = result.get("backend")
    if backend not in {"claude", "codex"}:
        return
    receiver_environments = execution_identity.get("receiver_environments")
    environment = (
        receiver_environments.get(backend)
        if isinstance(receiver_environments, dict)
        else None
    )
    expected = {
        f"{backend}_cli_version": (
            environment.get("cli_version") if isinstance(environment, dict) else None
        ),
        f"{backend}_cli_launcher_sha256": (
            environment.get("launcher_sha256")
            if isinstance(environment, dict)
            else None
        ),
        f"{backend}_cli_native_sha256": (
            environment.get("native_sha256") if isinstance(environment, dict) else None
        ),
    }
    for field_name, expected_value in expected.items():
        if result.get(field_name) != expected_value:
            raise EvalBoardError(
                f"{context} {field_name} does not match its execution identity"
            )


def _validate_result_types(result: dict, *, context: str) -> None:
    """Refuse malformed rows instead of reinterpreting them."""
    _require_bool(result.get("success"), context=f"{context} success")
    error = result.get("error")
    if error is not None and not isinstance(error, str):
        raise EvalBoardError(f"{context} error must be null or a string, got {error!r}")
    failure_kind = result.get("failure_kind")
    if failure_kind not in {
        None,
        "timeout",
        "validation",
        "error",
        *_INFRA_FAILURE_KINDS,
    }:
        raise EvalBoardError(
            f"{context} failure_kind must be null, timeout, validation, error, "
            "context_overflow, output_truncated, or integrity"
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
    unexpected_accesses = result.get("unexpected_accesses")
    if not isinstance(unexpected_accesses, list) or any(
        not isinstance(access, str) or not access.strip()
        for access in unexpected_accesses
    ):
        raise EvalBoardError(
            f"{context} unexpected_accesses must be a list of nonempty strings"
        )
    if unexpected_accesses and failure_kind != "integrity":
        raise EvalBoardError(
            f"{context} has unexpected_accesses without an integrity failure"
        )
    if failure_kind == "integrity" and not unexpected_accesses:
        raise EvalBoardError(
            f"{context} integrity failure must record unexpected_accesses"
        )
    _require_nonnegative_int(
        result.get("duration_ms"), context=f"{context} duration_ms"
    )
    _require_optional_nonnegative_number(
        result.get("estimated_cost_usd"),
        context=f"{context} estimated_cost_usd",
    )
    _validate_result_effective_environment(result, context=context)
    raw_metrics = result.get("metrics")
    if raw_metrics is not None and not isinstance(raw_metrics, dict):
        raise EvalBoardError(
            f"{context} metrics must be null or an object, got {raw_metrics!r}"
        )
    _validate_result_artifact_bindings(result, context=context)
    metrics = _result_metrics(result)
    if failure_kind == "timeout" and (
        timeout_attempts == 0
        or metrics is not None
        or bool(result.get("output_file"))
        or result.get("generated_output_sha256") is not None
    ):
        raise EvalBoardError(
            f"{context} timeout row must have attempts and no generated artifact "
            "or artifact metrics"
        )
    if failure_kind in _INFRA_FAILURE_KINDS and (
        metrics is not None
        or bool(result.get("output_file"))
        or result.get("generated_output_sha256") is not None
    ):
        raise EvalBoardError(
            f"{context} {failure_kind} row must have no generated artifact "
            "or artifact metrics"
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


def _validate_result_execution_admission(
    result: dict,
    *,
    run_identity: dict,
    suite_name: str,
    manifest_identity: dict,
    case_identity: dict,
    corpus_identity: dict,
    runner_identities: list[dict],
    execution_identity: dict,
    execution_identity_sha256: str,
    context: str,
) -> None:
    """Bind each durable row to its complete producer admission context."""

    admission = result.get("admission")
    admitted_execution = (
        admission.get("execution") if isinstance(admission, dict) else None
    )
    if (
        not isinstance(admission, dict)
        or set(admission)
        != {
            "schema",
            "run",
            "suite",
            "case",
            "corpus",
            "execution",
            "rulespec",
        }
        or admission.get("schema") != _RESULT_ADMISSION_SCHEMA
        or not isinstance(admitted_execution, dict)
        or set(admitted_execution) != {"identity", "sha256"}
        or admitted_execution.get("identity") != execution_identity
        or admitted_execution.get("sha256") != execution_identity_sha256
    ):
        raise EvalBoardError(
            f"{context} admission execution identity does not match the suite evidence"
        )
    expected_suite = {
        "name": suite_name,
        "manifest_path": manifest_identity.get("path"),
        "manifest_content_sha256": manifest_identity.get("content_sha256"),
        "manifest_case_identities": manifest_identity.get("case_identities"),
        "effective_runner_identities": runner_identities,
    }
    if (
        admission.get("run") != run_identity
        or admission.get("suite") != expected_suite
        or admission.get("case") != case_identity
        or admission.get("corpus") != corpus_identity
    ):
        raise EvalBoardError(
            f"{context} admission does not match its run, manifest, case, "
            "corpus, or runner evidence"
        )

    rulespec_admission = admission.get("rulespec")
    if not isinstance(rulespec_admission, dict):
        raise EvalBoardError(f"{context} admission has malformed RuleSpec evidence")
    admitted_policy_root = rulespec_admission.get("policy_repo_root")
    roots = execution_identity.get("rulespec_roots")
    matching_roots = (
        [
            root
            for root in roots
            if isinstance(root, dict) and root.get("path") == admitted_policy_root
        ]
        if isinstance(roots, list)
        else []
    )
    if len(matching_roots) != 1:
        raise EvalBoardError(
            f"{context} admission RuleSpec root is not unique in its execution identity"
        )
    root_identity = matching_roots[0]
    expected_rulespec = {
        "policy_repo_root": root_identity.get("path"),
        "root_content_sha256": root_identity.get("content_sha256"),
        "toolchain_contract_sha256": root_identity.get("toolchain_contract_sha256"),
        "validation_waiver_set_sha256": root_identity.get(
            "validation_waiver_set_sha256"
        ),
    }
    if rulespec_admission != expected_rulespec:
        raise EvalBoardError(
            f"{context} admission RuleSpec evidence does not match its "
            "execution identity"
        )
    citation_path = case_identity.get("corpus_citation_path")
    case_jurisdiction = (
        citation_path.split("/", 1)[0] if isinstance(citation_path, str) else None
    )
    root_jurisdiction = _rulespec_root_topology(
        root_identity.get("path"),
        root_identity.get("toolchain_root"),
    )
    if root_jurisdiction != case_jurisdiction:
        raise EvalBoardError(
            f"{context} admission RuleSpec root does not match its case "
            "citation jurisdiction"
        )
    if case_identity.get("oracle") != "policyengine":
        return
    runtime_wrapper = execution_identity.get("policyengine_runtime")
    runtime = (
        runtime_wrapper.get("identity") if isinstance(runtime_wrapper, dict) else None
    )
    if not isinstance(runtime, dict):
        return
    toolchain_root = root_identity.get("toolchain_root")
    expected_pin_path = (
        (PurePosixPath(toolchain_root) / _RULESPEC_RUNTIME_PIN_PATHSPEC).as_posix()
        if isinstance(toolchain_root, str) and "\\" not in toolchain_root
        else None
    )
    if (
        runtime.get("country") != case_jurisdiction.split("-", 1)[0]
        or runtime.get("rulespec_runtime_pin_path") != expected_pin_path
        or runtime.get("rulespec_runtime_pin_sha256")
        != root_identity.get("policyengine_runtime_pin_sha256")
    ):
        raise EvalBoardError(
            f"{context} PolicyEngine runtime is not bound to its admitted RuleSpec root"
        )


def _validate_result_policyengine_runtime_evidence(
    result: dict,
    *,
    case_identity: dict,
    execution_identity: dict,
    context: str,
) -> None:
    """Bind oracle metrics to the exact sealed runtime admitted for the suite."""

    metrics = _result_metrics(result)
    metric_identity = (
        metrics.get("policyengine_runtime_identity")
        if isinstance(metrics, dict)
        else None
    )
    metric_digest = (
        metrics.get("policyengine_runtime_identity_sha256")
        if isinstance(metrics, dict)
        else None
    )
    has_policyengine_evidence = any(
        value is not None
        for value in (
            metrics.get("policyengine_pass") if isinstance(metrics, dict) else None,
            metrics.get("policyengine_score") if isinstance(metrics, dict) else None,
            metric_identity,
            metric_digest,
        )
    )
    if case_identity.get("oracle") == "none":
        if has_policyengine_evidence:
            raise EvalBoardError(
                f"{context} has undeclared PolicyEngine oracle evidence"
            )
        return
    expected_runtime = execution_identity.get("policyengine_runtime")
    if not isinstance(expected_runtime, dict):
        raise EvalBoardError(
            f"{context} PolicyEngine runtime is not admitted for this case"
        )
    if metrics is None:
        if result.get("success") is True:
            raise EvalBoardError(
                f"{context} PolicyEngine case succeeded without oracle evidence"
            )
        if (
            bool(result.get("output_file"))
            or result.get("generated_output_sha256") is not None
        ):
            raise EvalBoardError(
                f"{context} PolicyEngine artifact has no oracle evidence"
            )
        return
    if not has_policyengine_evidence or metrics.get("policyengine_pass") is None:
        raise EvalBoardError(
            f"{context} PolicyEngine oracle evidence has no pass outcome"
        )
    if metric_identity != expected_runtime.get(
        "identity"
    ) or metric_digest != expected_runtime.get("sha256"):
        raise EvalBoardError(
            f"{context} PolicyEngine runtime evidence does not match the "
            "suite execution identity"
        )
    if result.get("success") is True and metrics.get("policyengine_pass") is not True:
        raise EvalBoardError(
            f"{context} succeeded although its PolicyEngine oracle did not pass"
        )


def _validate_result_row_admission(
    result: dict,
    *,
    run_identity: dict,
    suite_name: str,
    manifest_identity: dict,
    case_identity: dict,
    corpus_identity: dict,
    runner_identities: list[dict],
    execution_identity: dict,
    execution_identity_sha256: str,
    context: str,
) -> None:
    """Apply the complete shared admission policy for one durable result row."""

    canonical_citation = case_identity.get("corpus_citation_path")
    if (
        not isinstance(canonical_citation, str)
        or not canonical_citation
        or result.get("citation") != canonical_citation
    ):
        raise EvalBoardError(
            f"{context} citation does not match its canonical case path"
        )
    _validate_result_execution_admission(
        result,
        run_identity=run_identity,
        suite_name=suite_name,
        manifest_identity=manifest_identity,
        case_identity=case_identity,
        corpus_identity=corpus_identity,
        runner_identities=runner_identities,
        execution_identity=execution_identity,
        execution_identity_sha256=execution_identity_sha256,
        context=context,
    )
    _validate_result_types(result, context=context)
    _validate_result_receiver_identity_binding(
        result,
        execution_identity=execution_identity,
        context=context,
    )
    _validate_result_policyengine_runtime_evidence(
        result,
        case_identity=case_identity,
        execution_identity=execution_identity,
        context=context,
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
    for field_name in ("name", "kind", "corpus_citation_path", "oracle", "sha256"):
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
    if failure_kind in _INFRA_FAILURE_KINDS:
        error = result.get("error")
        detail = str(error)[:200] if error else failure_kind.replace("_", " ")
        return BoardCell(
            state=failure_kind,
            duration_ms=duration_ms,
            detail=detail,
        )
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
    runner_effort_identities: dict[str, dict] = {}
    receiver_environment_identities: dict[str, dict] = {}
    receiver_environment_sources: dict[str, str] = {}
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
        manifest_identity = payload["evidence"]["manifest"]
        run_identity = _payload_run_identity(payload, source)
        corpus_identity = _payload_corpus_identity(payload, source)
        payload_runner_identities = _payload_runner_identities(payload, source)
        execution_identity, execution_digest = _payload_execution_identity(
            payload, source
        )
        common_execution_identity = dict(execution_identity)
        common_execution_identity.pop("runner_efforts")
        common_execution_identity.pop("receiver_environments")
        normalized_execution = normalized_execution_identity(common_execution_identity)
        payload_runner_efforts = {
            effort["name"]: effort for effort in execution_identity["runner_efforts"]
        }
        for backend, environment in execution_identity["receiver_environments"].items():
            prior_environment = receiver_environment_identities.get(backend)
            if (
                prior_environment is not None
                and prior_environment != environment
                and not allow_mixed_toolchains
            ):
                raise EvalBoardError(
                    "Suite results are not comparable: receiver environment "
                    f"for {backend!r} in {source} does not match "
                    f"{receiver_environment_sources[backend]}"
                )
            if prior_environment is not None and prior_environment != environment:
                if source not in mixed_toolchain_sources:
                    mixed_toolchain_sources.append(source)
            else:
                receiver_environment_identities[backend] = environment
                receiver_environment_sources[backend] = source
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
                        "content/toolchain/waivers, generation/retry case "
                        "budget, runner "
                        "timeouts/retries, or PolicyEngine runtime differ; "
                        "checkout locations are ignored). Re-run on "
                        "one toolchain, or pass --allow-mixed-toolchains to "
                        "fold anyway with the mismatch recorded."
                    )
                mixed_toolchain_sources.append(source)

        payload_runner_names: set[str] = set()
        for identity in payload_runner_identities:
            name = identity.get("name")
            assert isinstance(name, str)
            effort_identity = payload_runner_efforts[name]
            prior_effort_identity = runner_effort_identities.get(name)
            if (
                prior_effort_identity is not None
                and prior_effort_identity != effort_identity
            ):
                raise EvalBoardError(
                    f"Runner {name!r} requests a different effort in "
                    f"{runner_sources[name]} and {source}; same-name runs must "
                    "use the same requested effort, including receiver default"
                )
            if name in runner_sources:
                raise EvalBoardError(
                    f"Runner {name!r} appears in both {runner_sources[name]} "
                    f"and {source}; two runs of one runner are two boards, "
                    "not one — drop one input or rename the runner"
                )
            runner_sources[name] = source
            runner_identities[name] = identity
            runner_effort_identities[name] = effort_identity
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
            context = f"Result row #{position} in {source}"
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
            case_index = _validate_result_case_binding(
                eval_case,
                case_identities,
                context=context,
            )
            _validate_result_row_admission(
                result,
                run_identity=run_identity,
                suite_name=suite_name,
                manifest_identity=manifest_identity,
                case_identity=case_identities[case_index - 1],
                corpus_identity=corpus_identity,
                runner_identities=payload_runner_identities,
                execution_identity=execution_identity,
                execution_identity_sha256=execution_digest,
                context=context,
            )
            if case_index in runner_results[runner]:
                raise EvalBoardError(
                    f"Duplicate result for runner {runner!r} case "
                    f"#{case_index} in {source}"
                )
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
            requested_effort=runner_effort_identities[name]["requested_effort"],
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
    "context_overflow": "C",
    "output_truncated": "X",
    "integrity": "I",
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
                "requested_effort": stats.requested_effort,
                "uses_receiver_default": stats.requested_effort is None,
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
        "| runner | model | requested effort | gate pass | timeouts | artifacts | "
        "compile | ci | grounded | src coverage | review | review score | oracle | "
        "median s | mean $ |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 15)
    for stats in ordered:
        oracle = (
            f"{stats.policyengine_pass_count}/{stats.policyengine_case_count}"
            if stats.policyengine_case_count
            else "—"
        )
        lines.append(
            "| {runner} | {model} | {effort} | {gate} | {timeouts} | {artifacts} | "
            "{compile} | {ci} | {grounded} | "
            "{coverage} | {review} | {review_score} | {oracle} | {median} | "
            "{cost} |".format(
                runner=stats.runner,
                model=stats.model,
                effort=stats.requested_effort or "default (receiver)",
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
        "C = context overflow, X = output truncated, I = integrity error, "
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
            f"requested effort {stats.requested_effort or 'default (receiver)'}  "
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
    lines.append(
        "Grid (P pass / F fail / T timeout / C context overflow / "
        "X output truncated / I integrity error / E error / · not run):"
    )
    for case in board.cases:
        cells = " ".join(
            _CELL_GLYPHS[board.cells[(case.index, stats.runner)].state]
            for stats in ordered
        )
        lines.append(f"  {case.index:02d} {case.name:<32} {cells}")
    lines.append("")
    lines.append("Runners: " + ", ".join(stats.runner for stats in ordered))
    return "\n".join(lines)
