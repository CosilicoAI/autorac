#!/usr/bin/env python3
"""Run the local SNAP queue until idle.

This is the canonical repo-backed copy of the local queue runner that powers the
event-driven Codex automation. The live automation may invoke a thin wrapper from
`~/.codex`, but logic changes should land here first.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from axiom_encode.harness.evals import resolve_corpus_source_unit
from axiom_encode.toolchain import load_rulespec_local_corpus_release

CODEX_HOME = (
    Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    .expanduser()
    .resolve()
)
AXIOM_ENCODE_ROOT = (
    Path(os.environ.get("AXIOM_ENCODE_ROOT", str(Path(__file__).resolve().parents[1])))
    .expanduser()
    .resolve()
)
AUTOMATION_DIR = (
    Path(
        os.environ.get(
            "AXIOM_ENCODE_SNAP_AUTOMATION_DIR",
            str(CODEX_HOME / "automations" / "hourly-snap-encode"),
        )
    )
    .expanduser()
    .resolve()
)
QUEUE_PATH = AUTOMATION_DIR / "queue.json"
MEMORY_PATH = AUTOMATION_DIR / "memory.md"
RUN_LEDGER_PATH = AUTOMATION_DIR / "run_ledger.ndjson"
LOCK_PATH = AUTOMATION_DIR / "runner.lock"
TMP_ROOT = (
    Path(os.environ.get("AXIOM_ENCODE_TMP_ROOT", str(Path.home() / "tmp")))
    .expanduser()
    .resolve()
)
DEFAULT_ARCHIVE_ROOT = (
    Path(
        os.environ.get(
            "AXIOM_ENCODE_EVAL_ARCHIVE_ROOT",
            str(AXIOM_ENCODE_ROOT / "artifacts" / "eval-suites"),
        )
    )
    .expanduser()
    .resolve()
)
BENCHMARK_GLOB = "us_snap_*_refresh.yaml"
SOURCE_TRACKING_VERSION = 2
RUN_LEDGER_SCHEMA_VERSION = 2
EVAL_SUITE_RESULTS_SCHEMA = "axiom-encode/eval-suite-results/v8"
EVAL_SUITE_SUMMARY_SCHEMA = "axiom-encode/eval-suite-summary/v8"
_EVAL_SUITE_SCHEMAS = {
    "results": EVAL_SUITE_RESULTS_SCHEMA,
    "summary": EVAL_SUITE_SUMMARY_SCHEMA,
}
_EVAL_SUITE_SCHEMA_RE = re.compile(
    r"^axiom-encode/eval-suite-(results|summary)/v([1-9][0-9]*)$"
)
REQUIRED_PATH_ENTRIES = [
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    str(Path.home() / ".bun" / "bin"),
    str(Path.home() / ".local" / "bin"),
    str(Path.home() / "bin"),
]
UV_CANDIDATES = [
    Path("/opt/homebrew/bin/uv"),
    Path("/usr/local/bin/uv"),
]
RETRYABLE_PATTERNS = (
    "usage limit",
    "rate limit",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "temporary failure",
    "dns",
    "could not resolve",
    "connection reset",
    "connection aborted",
    "connection refused",
    "502",
    "503",
    "504",
)


@dataclass
class ActiveState:
    status: str
    action: str
    manifest: str = "none"
    target: str = "none"
    source_repo: str = "none"
    corpus_citation_path: str = "none"
    output_dir: str = "none"
    archive_path: str = "none"
    started_at: str = "none"
    finished_at: str = "none"
    progress: str = "none"
    outcome: str = "none"


@dataclass
class ConsumedSchemaTracker:
    """Keep every artifact consumed by one queue invocation on one generation."""

    generation: int | None = None

    def admit(self, *, kind: str, schema: object, source: str) -> str:
        if kind not in _EVAL_SUITE_SCHEMAS:
            raise ValueError(f"Unknown eval-suite artifact kind: {kind}")
        if not isinstance(schema, str):
            raise ValueError(f"{source} has a malformed {kind} schema")
        match = _EVAL_SUITE_SCHEMA_RE.fullmatch(schema)
        if match is None or match.group(1) != kind:
            raise ValueError(f"{source} has a malformed {kind} schema: {schema!r}")
        generation = int(match.group(2))
        if self.generation is not None and generation != self.generation:
            raise ValueError(
                "Refusing mixed eval-suite schema generations in one SNAP ledger "
                f"invocation: already consumed v{self.generation}, but {source} "
                f"uses v{generation}"
            )
        expected = _EVAL_SUITE_SCHEMAS[kind]
        if schema != expected:
            raise ValueError(
                f"{source} uses unsupported {kind} schema {schema!r}; "
                f"expected {expected!r}"
            )
        self.generation = generation
        return schema


def now_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def now_local() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "snap-task"


def load_queue(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_queue(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_paths(paths: list[Path | None]) -> str | None:
    existing = [
        path.resolve()
        for path in paths
        if path is not None and path.exists() and path.is_file()
    ]
    if not existing:
        return None

    digest = hashlib.sha256()
    for resolved in existing:
        digest.update(b"file\0")
        file_digest = sha256_file(resolved)
        if file_digest is not None:
            digest.update(file_digest.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def git_head(repo: Path | None) -> str | None:
    if repo is None or not repo.exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_root_for_path(path: Path | None) -> Path | None:
    if path is None or not path.exists():
        return None
    target = path if path.is_dir() else path.parent
    result = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()) if result.stdout.strip() else None


def append_event(data: dict[str, Any], message: str) -> None:
    log = data.setdefault("event_log", [])
    log.append({"timestamp": now_utc(), "message": message})
    if len(log) > 50:
        del log[:-50]


def build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("AXIOM_ENCODE_POLICYENGINE_US_PYTHON", None)
    env.pop("AXIOM_ENCODE_POLICYENGINE_UK_PYTHON", None)
    existing = [entry for entry in env.get("PATH", "").split(":") if entry]
    merged: list[str] = []
    for entry in [*REQUIRED_PATH_ENTRIES, *existing]:
        if entry not in merged:
            merged.append(entry)
    env["PATH"] = ":".join(merged)
    return env


def resolve_uv_bin() -> str:
    uv_from_path = shutil.which("uv")
    if uv_from_path:
        return uv_from_path
    for candidate in UV_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return "uv"


def infer_repo(corpus_citation_path: str | None) -> str:
    if not corpus_citation_path:
        return "none"
    jurisdiction = corpus_citation_path.strip().split("/", 1)[0]
    if not jurisdiction:
        return "none"
    if jurisdiction == "us" or jurisdiction.startswith("us-"):
        return "rulespec-us"
    return f"rulespec-{jurisdiction}"


def resolve_manifest_case_corpus_citation_path(
    case: dict[str, Any],
) -> str | None:
    kind = str(case.get("kind") or "")
    if kind == "source":
        corpus_citation_path = str(case.get("corpus_citation_path") or "").strip()
        return corpus_citation_path or None
    return None


def sha256_corpus_source(
    corpus_citation_path: str | None,
    corpus_root: Path,
    policy_repo_root: Path,
) -> str | None:
    if not corpus_citation_path:
        return None
    corpus_release = load_rulespec_local_corpus_release(
        policy_repo_root,
        corpus_root,
    )
    source_unit = resolve_corpus_source_unit(
        corpus_citation_path,
        corpus_release,
    )
    digest = hashlib.sha256()
    digest.update(b"corpus\0")
    digest.update(corpus_release.name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(corpus_release.content_sha256.encode("ascii"))
    digest.update(b"\0")
    digest.update(source_unit.citation_path.encode("utf-8"))
    digest.update(b"\0")
    digest.update(source_unit.body.encode("utf-8"))
    return digest.hexdigest()


def iter_manifest_queue_candidates(
    corpus_root: Path,
    policy_repo_root: Path,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for manifest_path in sorted(
        (AXIOM_ENCODE_ROOT / "benchmarks").glob(BENCHMARK_GLOB)
    ):
        try:
            manifest = yaml.safe_load(manifest_path.read_text()) or {}
        except yaml.YAMLError:
            continue
        cases = manifest.get("cases") or []
        if not isinstance(cases, list):
            continue
        for case in cases:
            if not isinstance(case, dict):
                continue
            name = case.get("name")
            corpus_citation_path = resolve_manifest_case_corpus_citation_path(case)
            if not name or corpus_citation_path is None:
                continue
            source_repo = infer_repo(corpus_citation_path)
            if source_repo in {"none", "unknown"}:
                continue
            source_sha = sha256_corpus_source(
                corpus_citation_path,
                corpus_root,
                policy_repo_root,
            )
            if source_sha is None:
                continue
            candidates.append(
                {
                    "name": str(name),
                    "manifest": str(manifest_path),
                    "corpus_citation_path": corpus_citation_path,
                    "corpus_source_sha256": source_sha,
                    "source_repo": source_repo,
                }
            )
    return candidates


def sync_queue_with_manifests(
    data: dict[str, Any],
    corpus_root: Path,
    policy_repo_root: Path,
) -> tuple[bool, list[str], list[str], list[str]]:
    items = data.setdefault("items", [])
    candidates = iter_manifest_queue_candidates(corpus_root, policy_repo_root)
    candidate_names = {candidate["name"] for candidate in candidates}
    by_name = {
        item.get("name"): item
        for item in items
        if isinstance(item, dict) and item.get("name")
    }
    added: list[str] = []
    retired: list[str] = []
    refreshed: list[str] = []
    changed = False
    for candidate in candidates:
        manifest_sha = sha256_file(Path(candidate["manifest"]))
        source_sha = candidate.get("corpus_source_sha256")
        existing = by_name.get(candidate["name"])
        if existing is None:
            items.append(
                {
                    "name": candidate["name"],
                    "status": "queued",
                    "manifest": candidate["manifest"],
                    "corpus_citation_path": candidate["corpus_citation_path"],
                    "manifest_sha256": manifest_sha,
                    "corpus_source_sha256": source_sha,
                    "source_tracking_version": SOURCE_TRACKING_VERSION,
                    "note": "queued from manifest sync",
                }
            )
            added.append(candidate["name"])
            changed = True
            continue
        previous_manifest_sha = existing.get("manifest_sha256")
        previous_source_sha = existing.get("corpus_source_sha256")
        previous_tracking_version = existing.get("source_tracking_version")
        identity_changed = (
            previous_tracking_version != SOURCE_TRACKING_VERSION
            or previous_manifest_sha != manifest_sha
            or previous_source_sha != source_sha
        )
        if existing.get("status") == "running" and identity_changed:
            # Preserve the identity an in-flight run actually started with.
            # Once it stops, a later sync will requeue it against the new one.
            continue
        for key in ("manifest", "corpus_citation_path"):
            if existing.get(key) != candidate[key]:
                existing[key] = candidate[key]
                changed = True
        if existing.get("manifest_sha256") != manifest_sha:
            existing["manifest_sha256"] = manifest_sha
            changed = True
        if existing.get("corpus_source_sha256") != source_sha:
            existing["corpus_source_sha256"] = source_sha
            changed = True
        if existing.get("source_tracking_version") != SOURCE_TRACKING_VERSION:
            existing["source_tracking_version"] = SOURCE_TRACKING_VERSION
            changed = True
        if identity_changed and existing.get("status") in {
            "done",
            "blocked",
            "retryable",
        }:
            existing["status"] = "queued"
            existing["started_at"] = None
            existing["finished_at"] = None
            existing["output_dir"] = None
            existing["archive_path"] = None
            existing["note"] = "requeued after manifest/source identity change"
            refreshed.append(candidate["name"])
            changed = True
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        manifest = item.get("manifest")
        status = item.get("status")
        if not name or name in candidate_names or not manifest:
            continue
        if status in {"running", "done", "retired"}:
            continue
        if Path(manifest).exists():
            continue
        item["status"] = "retired"
        item["finished_at"] = now_utc()
        item["note"] = (
            "manifest removed from Axiom Encode benchmarks; retired from queue"
        )
        retired.append(str(name))
        changed = True
    return changed, added, retired, refreshed


def find_active_eval_processes() -> list[str]:
    result = subprocess.run(
        ["ps", "-Ao", "pid=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    active: list[str] = []
    for line in lines:
        if "run_queue_until_idle.py" in line:
            continue
        command = line.split(maxsplit=1)[1] if " " in line else ""
        if re.search(
            r"(^|\\s)uv run axiom-encode eval-suite(\\s|$)", command
        ) or re.search(
            r"(^|\\s)\\S+/axiom-encode eval-suite(\\s|$)",
            command,
        ):
            active.append(line)
    return active


def build_output_dir(name: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    return TMP_ROOT / f"axiom-encode-{slugify(name)}-{timestamp}"


def classify_status(
    returncode: int, summary: dict[str, Any] | None, results: dict[str, Any] | None
) -> tuple[str, str]:
    if summary and summary.get("all_ready"):
        return "done", "closed fully ready"

    combined_parts: list[str] = []
    if returncode != 0:
        combined_parts.append(f"eval-suite exited with code {returncode}")
    if results:
        for result in results.get("results", []):
            error = result.get("error")
            if error:
                combined_parts.append(str(error))
            metrics = result.get("metrics") or {}
            combined_parts.extend(metrics.get("compile_issues") or [])
            combined_parts.extend(metrics.get("ci_issues") or [])
            combined_parts.extend(metrics.get("generalist_review_issues") or [])
            combined_parts.extend(metrics.get("policyengine_issues") or [])
    combined = " | ".join(part for part in combined_parts if part).lower()
    if any(pattern in combined for pattern in RETRYABLE_PATTERNS):
        return "retryable", combined_parts[
            0
        ] if combined_parts else "retryable infrastructure failure"
    return "blocked", combined_parts[0] if combined_parts else "not ready"


def run_eval_item(
    item: dict[str, Any],
    reviewer_cli: str,
    output_dir: Path,
    corpus_root: Path,
    axiom_rules_root: Path,
    policy_repo_root: Path,
    policyengine_runtime_root: Path,
) -> int:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    env = build_subprocess_env()
    env["AXIOM_ENCODE_REVIEWER_CLI"] = reviewer_cli
    cmd = [
        resolve_uv_bin(),
        "run",
        "axiom-encode",
        "eval-suite",
        item["manifest"],
        "--corpus-path",
        str(corpus_root),
        "--axiom-rules-engine-path",
        str(axiom_rules_root),
        "--policy-repo-path",
        str(policy_repo_root),
        "--policyengine-runtime-root",
        str(policyengine_runtime_root),
        "--output",
        str(output_dir),
    ]
    try:
        result = subprocess.run(cmd, cwd=AXIOM_ENCODE_ROOT, env=env, check=False)
    except FileNotFoundError:
        return 127
    return result.returncode


def archive_eval(
    output_dir: Path,
    *,
    corpus_root: Path,
    axiom_rules_root: Path,
    policy_repo_root: Path,
    policyengine_runtime_root: Path,
) -> Path | None:
    cmd = [
        resolve_uv_bin(),
        "run",
        "axiom-encode",
        "eval-suite-archive",
        str(output_dir),
        "--corpus-path",
        str(corpus_root),
        "--axiom-rules-engine-path",
        str(axiom_rules_root),
        "--policy-repo-path",
        str(policy_repo_root),
        "--policyengine-runtime-root",
        str(policyengine_runtime_root),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=AXIOM_ENCODE_ROOT,
            env=build_subprocess_env(),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"Archived eval suite to (.+)", result.stdout)
    return Path(match.group(1).strip()) if match else None


def _validate_eval_artifact_payload(
    payload: object,
    *,
    kind: str,
    schema_tracker: ConsumedSchemaTracker,
    source: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object")
    if "schema" not in payload:
        raise ValueError(f"{source} is missing schema")
    schema_tracker.admit(kind=kind, schema=payload["schema"], source=source)
    return payload


def load_eval_artifact(
    path: Path,
    *,
    kind: str,
    schema_tracker: ConsumedSchemaTracker,
) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    return _validate_eval_artifact_payload(
        payload,
        kind=kind,
        schema_tracker=schema_tracker,
        source=str(path),
    )


def _validate_consumed_schemas(
    consumed_schemas: object,
    *,
    schema_tracker: ConsumedSchemaTracker,
    source: str,
) -> dict[str, str | None]:
    if not isinstance(consumed_schemas, dict) or set(consumed_schemas) != {
        "results",
        "summary",
    }:
        raise ValueError(
            f"{source} must contain exact results and summary consumed schemas"
        )
    validated: dict[str, str | None] = {}
    for kind in ("summary", "results"):
        schema = consumed_schemas[kind]
        if schema is None:
            validated[kind] = None
            continue
        validated[kind] = schema_tracker.admit(
            kind=kind,
            schema=schema,
            source=source,
        )
    return validated


def load_run_ledger_ids(
    path: Path,
    *,
    schema_tracker: ConsumedSchemaTracker,
) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"SNAP run ledger has malformed JSON at line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"SNAP run ledger has a non-object row at line {line_number}"
            )
        schema_version = record.get("schema_version")
        if type(schema_version) is not int:
            raise ValueError(
                "SNAP run ledger has a malformed record schema at line "
                f"{line_number}: {schema_version!r}"
            )
        if schema_version == RUN_LEDGER_SCHEMA_VERSION:
            _validate_consumed_schemas(
                record.get("consumed_schemas"),
                schema_tracker=schema_tracker,
                source=f"SNAP run ledger line {line_number}",
            )
        elif schema_version != 1:
            raise ValueError(
                "SNAP run ledger has an unsupported record schema at line "
                f"{line_number}: {schema_version!r}"
            )
        run_id = record.get("run_id")
        if schema_version == RUN_LEDGER_SCHEMA_VERSION and (
            not isinstance(run_id, str) or not run_id
        ):
            raise ValueError(
                f"SNAP run ledger v2 row at line {line_number} has no run_id"
            )
        if isinstance(run_id, str) and run_id:
            ids.add(run_id)
    return ids


def compute_run_id(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("name") or "unknown"),
        str(item.get("started_at") or "none"),
        str(item.get("output_dir") or item.get("archive_path") or "none"),
    ]
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:24]


def classify_failure_class(
    status: str,
    returncode: int | None,
    summary: dict[str, Any] | None,
    results: dict[str, Any] | None,
    note: str | None,
) -> str:
    if summary and summary.get("all_ready"):
        return "ready"
    lowered = " | ".join(
        str(part).lower() for part in [note, returncode] if part not in (None, "")
    )
    if "usage limit" in lowered or "rate limit" in lowered:
        return "retryable_quota"
    if any(
        token in lowered
        for token in ("dns", "timeout", "temporarily unavailable", "connection")
    ):
        return "retryable_transport"
    for result in (results or {}).get("results", []):
        error = str(result.get("error") or "").lower()
        metrics = result.get("metrics") or {}
        if "no rulespec content returned" in error:
            return "generation_no_content"
        if metrics.get("compile_pass") is False:
            return "compile"
        if metrics.get("ci_pass") is False:
            return "ci"
        if metrics.get("generalist_review_pass") is False:
            return "review"
        if metrics.get("policyengine_pass") is False:
            return "policyengine"
    if status == "retryable":
        return "retryable_unknown"
    if status == "blocked":
        return "blocked_unknown"
    return "completed_unknown"


def build_run_record(
    item: dict[str, Any],
    *,
    reviewer_cli: str,
    returncode: int | None,
    summary: dict[str, Any] | None,
    results: dict[str, Any] | None,
    archive_path: Path | None,
    status: str,
    note: str,
    policy_repo_root: Path,
    policyengine_runtime_root: Path,
    schema_tracker: ConsumedSchemaTracker,
    backfilled: bool = False,
) -> dict[str, Any]:
    validated_summary = (
        _validate_eval_artifact_payload(
            summary,
            kind="summary",
            schema_tracker=schema_tracker,
            source=f"SNAP run record for {item.get('name') or 'unknown'}",
        )
        if summary is not None
        else None
    )
    validated_results = (
        _validate_eval_artifact_payload(
            results,
            kind="results",
            schema_tracker=schema_tracker,
            source=f"SNAP run record for {item.get('name') or 'unknown'}",
        )
        if results is not None
        else None
    )
    consumed_schemas = _validate_consumed_schemas(
        {
            "summary": (
                validated_summary.get("schema")
                if validated_summary is not None
                else None
            ),
            "results": (
                validated_results.get("schema")
                if validated_results is not None
                else None
            ),
        },
        schema_tracker=schema_tracker,
        source=f"SNAP run record for {item.get('name') or 'unknown'}",
    )
    manifest_path = Path(item["manifest"]).resolve() if item.get("manifest") else None
    corpus_citation_path = item.get("corpus_citation_path")
    output_dir = Path(item["output_dir"]).resolve() if item.get("output_dir") else None
    source_repo = infer_repo(corpus_citation_path)
    effective_runner = None
    readiness_block = None
    if summary:
        effective_runners = (summary.get("manifest") or {}).get(
            "effective_runners"
        ) or []
        effective_runner = effective_runners[0] if effective_runners else None
        readiness = summary.get("readiness") or {}
        if effective_runner and isinstance(readiness, dict):
            readiness_block = readiness.get(effective_runner)
        elif isinstance(readiness, dict) and len(readiness) == 1:
            readiness_block = next(iter(readiness.values()))

    result_rows = (results or {}).get("results") or []
    first_result = result_rows[0] if result_rows else {}
    metrics = first_result.get("metrics") or {}
    token_fields = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    estimated_cost_usd = 0.0
    actual_cost_usd: float | None = None
    duration_ms = 0
    issue_counts = {
        "compile_issue_count": 0,
        "ci_issue_count": 0,
        "generalist_review_issue_count": 0,
        "policyengine_issue_count": 0,
    }
    errors: list[str] = []
    for row in result_rows:
        for key in token_fields:
            token_fields[key] += int(row.get(key) or 0)
        estimated_cost_usd += float(row.get("estimated_cost_usd") or 0.0)
        if row.get("actual_cost_usd") is not None:
            actual_cost_usd = (actual_cost_usd or 0.0) + float(row["actual_cost_usd"])
        duration_ms += int(row.get("duration_ms") or 0)
        row_metrics = row.get("metrics") or {}
        issue_counts["compile_issue_count"] += len(
            row_metrics.get("compile_issues") or []
        )
        issue_counts["ci_issue_count"] += len(row_metrics.get("ci_issues") or [])
        issue_counts["generalist_review_issue_count"] += len(
            row_metrics.get("generalist_review_issues") or []
        )
        issue_counts["policyengine_issue_count"] += len(
            row_metrics.get("policyengine_issues") or []
        )
        if row.get("error"):
            errors.append(str(row["error"]))

    return {
        "schema_version": RUN_LEDGER_SCHEMA_VERSION,
        "consumed_schemas": consumed_schemas,
        "recorded_at": now_utc(),
        "run_id": compute_run_id(item),
        "target": item.get("name"),
        "status": status,
        "failure_class": classify_failure_class(
            status, returncode, summary, results, note
        ),
        "backfilled_from_queue": backfilled,
        "backend": first_result.get("backend") or None,
        "effective_runner": effective_runner,
        "reviewer_cli": reviewer_cli,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_sha256": sha256_file(manifest_path),
        "corpus_citation_path": corpus_citation_path,
        "corpus_source_sha256": item["corpus_source_sha256"],
        "source_tracking_version": item.get(
            "source_tracking_version", SOURCE_TRACKING_VERSION
        ),
        "source_repo": source_repo,
        "axiom_encode_sha": git_head(AXIOM_ENCODE_ROOT),
        "policy_repo_sha": git_head(policy_repo_root),
        "policyengine_sha": git_head(policyengine_runtime_root),
        "started_at": item.get("started_at"),
        "finished_at": item.get("finished_at"),
        "returncode": returncode,
        "duration_ms": duration_ms or None,
        "output_dir": str(output_dir) if output_dir else None,
        "archive_path": str(archive_path.resolve())
        if archive_path
        else item.get("archive_path"),
        "note": note,
        "all_ready": bool(summary.get("all_ready")) if summary else False,
        "readiness": readiness_block,
        "result_count": len(result_rows),
        "first_error": errors[0] if errors else None,
        "generation_prompt_sha256": first_result.get("generation_prompt_sha256"),
        "generalist_review_prompt_sha256": metrics.get(
            "generalist_review_prompt_sha256"
        ),
        "metrics": {
            "compile_pass": metrics.get("compile_pass"),
            "ci_pass": metrics.get("ci_pass"),
            "generalist_review_pass": metrics.get("generalist_review_pass"),
            "policyengine_pass": metrics.get("policyengine_pass"),
            "success": first_result.get("success"),
            "zero_ungrounded": (metrics.get("ungrounded_numeric_count") == 0)
            if "ungrounded_numeric_count" in metrics
            else None,
            "generalist_review_score": metrics.get("generalist_review_score"),
            "policyengine_score": metrics.get("policyengine_score"),
            "estimated_cost_usd": estimated_cost_usd if result_rows else None,
            "actual_cost_usd": actual_cost_usd,
            **token_fields,
            **issue_counts,
        },
    }


def append_run_record(path: Path, record: dict[str, Any], known_ids: set[str]) -> None:
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id or run_id in known_ids:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    known_ids.add(run_id)


def backfill_run_ledger(
    data: dict[str, Any],
    known_ids: set[str],
    policy_repo_root: Path,
    policyengine_runtime_root: Path,
    *,
    schema_tracker: ConsumedSchemaTracker,
) -> None:
    for item in data.get("items", []):
        if item.get("status") not in {"done", "blocked", "retryable"}:
            continue
        if not item.get("output_dir") and not item.get("archive_path"):
            continue
        output_dir = (
            Path(item["output_dir"]).resolve() if item.get("output_dir") else None
        )
        archive_path = (
            Path(item["archive_path"]).resolve() if item.get("archive_path") else None
        )
        summary = (
            load_eval_artifact(
                output_dir / "summary.json",
                kind="summary",
                schema_tracker=schema_tracker,
            )
            if output_dir
            else None
        )
        results = (
            load_eval_artifact(
                output_dir / "results.json",
                kind="results",
                schema_tracker=schema_tracker,
            )
            if output_dir
            else None
        )
        if summary is None and archive_path:
            summary = load_eval_artifact(
                archive_path / "summary.json",
                kind="summary",
                schema_tracker=schema_tracker,
            )
        if results is None and archive_path:
            results = load_eval_artifact(
                archive_path / "results.json",
                kind="results",
                schema_tracker=schema_tracker,
            )
        record = build_run_record(
            item,
            reviewer_cli=str(data.get("default_reviewer_cli") or "claude"),
            returncode=None,
            summary=summary,
            results=results,
            archive_path=archive_path,
            status=str(item.get("status") or "unknown"),
            note=str(item.get("note") or ""),
            policy_repo_root=policy_repo_root,
            policyengine_runtime_root=policyengine_runtime_root,
            schema_tracker=schema_tracker,
            backfilled=True,
        )
        append_run_record(RUN_LEDGER_PATH, record, known_ids)


def render_memory(data: dict[str, Any], active: ActiveState) -> str:
    items = data.get("items", [])
    next_item = next(
        (
            item["name"]
            for item in items
            if item.get("status") in {"queued", "retryable"}
        ),
        "none queued or retryable",
    )
    lines = [
        "# Hourly SNAP Encode Memory",
        "",
        f"Last refreshed: {now_utc()}",
        f"Run ledger: `{RUN_LEDGER_PATH}`",
        "Queue seeding: auto-sync from checked-in `axiom-encode` SNAP refresh manifests on runner wakeup",
        "",
        "## Active SNAP eval",
        "",
        f"- status: {active.status}",
        f"- action: {active.action}",
        f"- manifest: {active.manifest}",
        f"- target: {active.target}",
        f"- source repo: {active.source_repo}",
        f"- corpus citation path: {active.corpus_citation_path}",
        f"- output dir: {active.output_dir}",
        f"- archive path: {active.archive_path}",
        f"- suite started_at: {active.started_at}",
        f"- suite finished_at: {active.finished_at}",
        f"- suite progress: {active.progress}",
        f"- outcome: {active.outcome}",
        "",
        "## Queue state",
        "",
    ]
    for item in items:
        lines.append(f"- `{item['name']}`: {item.get('status', 'unknown')}")
        if item.get("manifest"):
            lines.append(f"  manifest: `{item['manifest']}`")
        if item.get("corpus_citation_path"):
            lines.append(f"  corpus citation path: `{item['corpus_citation_path']}`")
        if item.get("output_dir"):
            lines.append(f"  output dir: `{item['output_dir']}`")
        if item.get("archive_path"):
            lines.append(f"  archive path: `{item['archive_path']}`")
        if item.get("note"):
            lines.append(f"  note: {item['note']}")
    lines.append(f"- next eligible target: {next_item}")
    lines.extend(["", "## Notes", ""])
    for event in data.get("event_log", [])[-25:]:
        lines.append(f"- {event['timestamp']}: {event['message']}")
    lines.append("")
    return "\n".join(lines)


def write_memory(data: dict[str, Any], active: ActiveState) -> None:
    MEMORY_PATH.write_text(render_memory(data, active))


def reconcile_stale_running_items(
    data: dict[str, Any],
    active_processes: list[str],
    *,
    schema_tracker: ConsumedSchemaTracker,
) -> bool:
    if active_processes:
        return False

    changed = False
    for item in data.get("items", []):
        if item.get("status") != "running":
            continue
        output_dir = (
            Path(item["output_dir"]).resolve() if item.get("output_dir") else None
        )
        archive_path = (
            Path(item["archive_path"]).resolve() if item.get("archive_path") else None
        )
        summary = (
            load_eval_artifact(
                output_dir / "summary.json",
                kind="summary",
                schema_tracker=schema_tracker,
            )
            if output_dir
            else None
        )
        if (not summary or not summary.get("all_ready")) and archive_path:
            summary = load_eval_artifact(
                archive_path / "summary.json",
                kind="summary",
                schema_tracker=schema_tracker,
            )
        if summary and summary.get("all_ready"):
            item["status"] = "done"
            item["finished_at"] = now_utc()
            item["note"] = "closed fully ready after orphaned eval completed"
            changed = True
            append_event(
                data,
                f"{item['name']} was left in `running`, but its output is ready; marked done.",
            )
            continue
        item["status"] = "retryable"
        item["finished_at"] = now_utc()
        item["note"] = (
            "runner exited before this queued eval finished; marked retryable for relaunch"
        )
        changed = True
        append_event(
            data,
            f"{item['name']} was left in `running` without a live eval process; marked retryable for relaunch.",
        )
    return changed


def reconcile_ready_output_items(
    data: dict[str, Any],
    *,
    schema_tracker: ConsumedSchemaTracker,
) -> tuple[bool, list[str]]:
    changed = False
    reconciled: list[str] = []
    for item in data.get("items", []):
        if item.get("status") not in {"blocked", "retryable"}:
            continue
        output_dir = (
            Path(item["output_dir"]).resolve() if item.get("output_dir") else None
        )
        archive_path = (
            Path(item["archive_path"]).resolve() if item.get("archive_path") else None
        )
        summary = (
            load_eval_artifact(
                output_dir / "summary.json",
                kind="summary",
                schema_tracker=schema_tracker,
            )
            if output_dir
            else None
        )
        if (not summary or not summary.get("all_ready")) and archive_path:
            summary = load_eval_artifact(
                archive_path / "summary.json",
                kind="summary",
                schema_tracker=schema_tracker,
            )
        if not summary or not summary.get("all_ready"):
            continue

        item["status"] = "done"
        item["finished_at"] = now_utc()
        item["note"] = "closed fully ready after revalidation"
        changed = True
        reconciled.append(str(item.get("name") or "unknown"))
    return changed, reconciled


def _resolve_required_checkout(path: Path, *, label: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"invalid {label} checkout {path}: {exc}") from exc
    if not resolved.is_dir():
        raise RuntimeError(f"invalid {label} checkout (not a directory): {resolved}")
    return resolved


def process_queue(
    queue_path: Path,
    corpus_root: Path,
    axiom_rules_root: Path,
    policy_repo_root: Path,
    policyengine_runtime_root: Path,
) -> int:
    corpus_root = _resolve_required_checkout(corpus_root, label="axiom-corpus")
    axiom_rules_root = _resolve_required_checkout(
        axiom_rules_root,
        label="axiom-rules-engine",
    )
    policy_repo_root = _resolve_required_checkout(
        policy_repo_root,
        label="RuleSpec policy",
    )
    policyengine_runtime_root = _resolve_required_checkout(
        policyengine_runtime_root,
        label="PolicyEngine runtime",
    )
    data = load_queue(queue_path)
    sync_changed, added_items, retired_items, refreshed_items = (
        sync_queue_with_manifests(data, corpus_root, policy_repo_root)
    )
    if sync_changed:
        save_queue(queue_path, data)
    if added_items:
        append_event(
            data,
            "Queued newly discovered SNAP manifests: " + ", ".join(added_items),
        )
        save_queue(queue_path, data)
    if retired_items:
        append_event(
            data,
            "Retired queue items whose manifests were removed: "
            + ", ".join(retired_items),
        )
        save_queue(queue_path, data)
    if refreshed_items:
        append_event(
            data,
            "Requeued queue items after manifest/source updates: "
            + ", ".join(refreshed_items),
        )
        save_queue(queue_path, data)
    reviewer_cli = str(data.get("default_reviewer_cli") or "claude")
    schema_tracker = ConsumedSchemaTracker()
    known_run_ids = load_run_ledger_ids(
        RUN_LEDGER_PATH,
        schema_tracker=schema_tracker,
    )
    ready_changed, ready_items = reconcile_ready_output_items(
        data,
        schema_tracker=schema_tracker,
    )
    if ready_changed:
        append_event(
            data,
            "Marked revalidated queue items done: " + ", ".join(ready_items),
        )
        save_queue(queue_path, data)
    backfill_run_ledger(
        data,
        known_run_ids,
        policy_repo_root,
        policyengine_runtime_root,
        schema_tracker=schema_tracker,
    )

    while True:
        active_processes = find_active_eval_processes()
        if reconcile_stale_running_items(
            data,
            active_processes,
            schema_tracker=schema_tracker,
        ):
            save_queue(queue_path, data)
        if active_processes:
            active = ActiveState(
                status="waiting",
                action="another SNAP-focused `axiom-encode eval-suite` is already running; waiting for it to finish before starting the next queued item",
                outcome=f"external eval already active at {now_local()}",
            )
            write_memory(data, active)
            time.sleep(30)
            data = load_queue(queue_path)
            sync_changed, added_items, retired_items, refreshed_items = (
                sync_queue_with_manifests(data, corpus_root, policy_repo_root)
            )
            if sync_changed:
                if added_items:
                    append_event(
                        data,
                        "Queued newly discovered SNAP manifests: "
                        + ", ".join(added_items),
                    )
                if retired_items:
                    append_event(
                        data,
                        "Retired queue items whose manifests were removed: "
                        + ", ".join(retired_items),
                    )
                if refreshed_items:
                    append_event(
                        data,
                        "Requeued queue items after manifest/source updates: "
                        + ", ".join(refreshed_items),
                    )
                save_queue(queue_path, data)
            continue

        item = next(
            (
                candidate
                for candidate in data.get("items", [])
                if candidate.get("status") in {"queued", "retryable"}
            ),
            None,
        )
        if item is None:
            active = ActiveState(
                status="idle",
                action="no active SNAP eval at refresh time; the queue has no queued or retryable target to launch",
                outcome=f"queue is idle at {now_local()}",
            )
            write_memory(data, active)
            return 0

        output_dir = build_output_dir(item["name"])
        item["status"] = "running"
        item["started_at"] = now_utc()
        item["output_dir"] = str(output_dir)
        item["archive_path"] = None
        item.pop("finished_at", None)
        item["note"] = "started with manifest-declared runners"
        save_queue(queue_path, data)
        active = ActiveState(
            status="running",
            action="running the next queued SNAP eval until completion",
            manifest=item.get("manifest", "none"),
            target=item["name"],
            source_repo=infer_repo(item.get("corpus_citation_path")),
            corpus_citation_path=item.get("corpus_citation_path", "none"),
            output_dir=str(output_dir),
            started_at=item["started_at"],
            outcome=f"started at {now_local()} with manifest-declared runners",
        )
        write_memory(data, active)

        returncode = run_eval_item(
            item,
            reviewer_cli=reviewer_cli,
            output_dir=output_dir,
            corpus_root=corpus_root,
            axiom_rules_root=axiom_rules_root,
            policy_repo_root=policy_repo_root,
            policyengine_runtime_root=policyengine_runtime_root,
        )
        summary = load_eval_artifact(
            output_dir / "summary.json",
            kind="summary",
            schema_tracker=schema_tracker,
        )
        results = load_eval_artifact(
            output_dir / "results.json",
            kind="results",
            schema_tracker=schema_tracker,
        )
        archive_path = (
            archive_eval(
                output_dir,
                corpus_root=corpus_root,
                axiom_rules_root=axiom_rules_root,
                policy_repo_root=policy_repo_root,
                policyengine_runtime_root=policyengine_runtime_root,
            )
            if (output_dir / "suite-run.json").exists()
            else None
        )
        new_status, reason = classify_status(returncode, summary, results)
        item["status"] = new_status
        item["output_dir"] = str(output_dir)
        item["archive_path"] = (
            str(archive_path) if archive_path else item.get("archive_path")
        )
        item["finished_at"] = now_utc()
        item["source_tracking_version"] = SOURCE_TRACKING_VERSION
        item["note"] = reason
        append_event(
            data,
            f"{item['name']} finished with status `{new_status}`; output `{output_dir}`"
            + (f" and archive `{archive_path}`" if archive_path else ""),
        )
        record = build_run_record(
            item,
            reviewer_cli=reviewer_cli,
            returncode=returncode,
            summary=summary,
            results=results,
            archive_path=archive_path,
            status=new_status,
            note=reason,
            policy_repo_root=policy_repo_root,
            policyengine_runtime_root=policyengine_runtime_root,
            schema_tracker=schema_tracker,
        )
        append_run_record(RUN_LEDGER_PATH, record, known_run_ids)
        save_queue(queue_path, data)
        active = ActiveState(
            status="completed",
            action="last queued SNAP eval finished",
            manifest=item.get("manifest", "none"),
            target=item["name"],
            source_repo=infer_repo(item.get("corpus_citation_path")),
            corpus_citation_path=item.get("corpus_citation_path", "none"),
            output_dir=str(output_dir),
            archive_path=str(archive_path) if archive_path else "none",
            started_at=item.get("started_at", "none"),
            finished_at=item["finished_at"],
            progress="1 case complete",
            outcome=f"{item['name']} finished `{new_status}` at {now_local()}",
        )
        write_memory(data, active)

        data = load_queue(queue_path)
        sync_changed, added_items, retired_items, refreshed_items = (
            sync_queue_with_manifests(data, corpus_root, policy_repo_root)
        )
        if sync_changed:
            if added_items:
                append_event(
                    data,
                    "Queued newly discovered SNAP manifests: " + ", ".join(added_items),
                )
            if retired_items:
                append_event(
                    data,
                    "Retired queue items whose manifests were removed: "
                    + ", ".join(retired_items),
                )
            if refreshed_items:
                append_event(
                    data,
                    "Requeued queue items after manifest/source updates: "
                    + ", ".join(refreshed_items),
                )
            save_queue(queue_path, data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SNAP queue until idle.")
    parser.add_argument("--queue", default=str(QUEUE_PATH))
    parser.add_argument(
        "--corpus-path",
        required=True,
        type=Path,
        help="Canonical axiom-corpus checkout bound by RuleSpec toolchains",
    )
    parser.add_argument(
        "--axiom-rules-engine-path",
        required=True,
        type=Path,
        help="Exact axiom-rules-engine checkout used by eval validation",
    )
    parser.add_argument(
        "--policy-repo-path",
        required=True,
        type=Path,
        help="Exact rulespec-us checkout used by the SNAP eval suite",
    )
    parser.add_argument(
        "--policyengine-runtime-root",
        required=True,
        type=Path,
        help="Exact clean official policyengine-us checkout used by oracle cases",
    )
    args = parser.parse_args()

    queue_path = Path(args.queue).resolve()
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        return process_queue(
            queue_path,
            args.corpus_path,
            args.axiom_rules_engine_path,
            args.policy_repo_path,
            args.policyengine_runtime_root,
        )


if __name__ == "__main__":
    sys.exit(main())
