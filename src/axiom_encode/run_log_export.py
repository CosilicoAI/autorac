"""Emission, backfill, publication, and staleness checks for the run log.

This module holds all axiom-encode-specific mapping between existing encoder
data structures and :mod:`axiom_encode.run_log` events. :mod:`run_log` stays a
pure schema/writer core; everything that knows about ``EvalResult``,
``EncodingRun``, encodings.db, and apply-manifests lives here.

Three producers, one schema:

* :func:`emit_live_encode_events` - called from the encode flow with the live
  ``EvalResult`` (whose ``metrics`` carry the *real* per-gate verdicts:
  compile/CI/grounding/oracle/review). This is the going-forward path that closes
  the "no durable per-gate verdict" gap.
* :func:`export_backfill` - reconstructs events for historical runs from
  encodings.db + signed apply-manifests. Only source-backed fields are set;
  everything else is explicitly ``null`` (never fabricated). Stages that were
  never captured historically (judge/pr/ci_run/merge/oracle_at_merge) are simply
  absent.
* :func:`publish` - folds every run's JSONL into the committed dashboard
  artifacts, mirroring how axiom-corpus publishes; :func:`check_staleness` is the
  freshness guard analog so publication cannot silently stop.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import ValidationError

from .repo_routing import is_composition_policy_repo_root
from .run_log import (
    FUNNEL_STEPS,
    SCHEMA_VERSION,
    Finding,
    RunLogEvent,
    RunLogWriter,
    Severity,
    StageStatus,
    fold_run,
    iter_events,
    pipeline_spec_dict,
    run_log_dir,
)

APPLY_MANIFEST_GLOB = ".axiom/encoding-manifests"

# Published artifact filenames (committed into the dashboard data dir).
PIPELINE_SPEC_FILE = "run_log_pipeline.json"
RUNS_INDEX_FILE = "run_log_runs.json"
EVENTS_FILE = "run_log_events.jsonl"
PUBLICATION_META_FILE = "run_log_publication.json"

# The wishlist fields the audit tracks, for coverage reporting.
COVERAGE_FIELDS = (
    "run_id",
    "citation",
    "backend",
    "model",
    "generation_prompt_sha256",
    "trace_ref",
    "corpus_context_ref",
    "gate.compile",
    "gate.ci",
    "gate.oracle",
    "gate.review",
    "judge",
    "apply",
    "manifest_sha_chain",
    "pr",
    "merge",
    "oracle_at_merge",
)


def _severity_for(kind: str) -> Severity:
    return {
        "critical": Severity.critical,
        "important": Severity.important,
        "minor": Severity.minor,
    }.get(kind, Severity.info)


# ---------------------------------------------------------------------------
# Live emission (rich: real per-gate verdicts from EvalArtifactMetrics)
# ---------------------------------------------------------------------------


def _gate_events_from_metrics(metrics: Any) -> list[dict[str, Any]]:
    """Map a live ``EvalArtifactMetrics`` into gate-stage event kwargs.

    Only gates whose verdict the metrics actually recorded are emitted; a gate
    that did not run (``None``) is omitted, not guessed.
    """
    events: list[dict[str, Any]] = []
    if metrics is None:
        return events

    compile_pass = getattr(metrics, "compile_pass", None)
    if compile_pass is not None:
        events.append(
            {
                "stage": "gate.compile",
                "status": StageStatus.passed if compile_pass else StageStatus.failed,
                "reason_code": None if compile_pass else "compile_error",
                "findings": [
                    Finding(code="compile_error", severity=Severity.critical, message=m)
                    for m in (getattr(metrics, "compile_issues", None) or [])
                ],
            }
        )

    ungrounded = getattr(metrics, "ungrounded_numeric_count", None)
    if ungrounded is not None:
        grounded = getattr(metrics, "grounded_numeric_count", 0)
        events.append(
            {
                "stage": "gate.grounding",
                "status": StageStatus.passed if not ungrounded else StageStatus.failed,
                "reason_code": None if not ungrounded else "ungrounded_numeric",
                "attrs": {
                    "grounded_numeric_count": grounded,
                    "ungrounded_numeric_count": ungrounded,
                    "missing_source_numeric_occurrence_count": getattr(
                        metrics, "missing_source_numeric_occurrence_count", None
                    ),
                },
                "findings": [
                    Finding(
                        code="ungrounded_numeric",
                        severity=Severity.important,
                        message=m,
                    )
                    for m in (getattr(metrics, "numeric_occurrence_issues", None) or [])
                ],
            }
        )

    ci_pass = getattr(metrics, "ci_pass", None)
    if ci_pass is not None:
        events.append(
            {
                "stage": "gate.ci",
                "status": StageStatus.passed if ci_pass else StageStatus.failed,
                "reason_code": None if ci_pass else "ci_failure",
                "findings": [
                    Finding(code="ci_failure", severity=Severity.critical, message=m)
                    for m in (getattr(metrics, "ci_issues", None) or [])
                ],
            }
        )

    review_pass = getattr(metrics, "generalist_review_pass", None)
    if review_pass is not None:
        events.append(
            {
                "stage": "gate.review",
                "status": StageStatus.passed if review_pass else StageStatus.failed,
                "reason_code": None if review_pass else "reviewer_rejected",
                "attrs": {
                    "score": getattr(metrics, "generalist_review_score", None),
                    "prompt_sha256": getattr(
                        metrics, "generalist_review_prompt_sha256", None
                    ),
                },
                "findings": [
                    Finding(
                        code="reviewer_issue", severity=Severity.important, message=m
                    )
                    for m in (getattr(metrics, "generalist_review_issues", None) or [])
                ],
            }
        )

    for oracle in ("policyengine",):
        passed = getattr(metrics, f"{oracle}_pass", None)
        if passed is None:
            continue
        events.append(
            {
                "stage": "gate.oracle",
                "status": StageStatus.passed if passed else StageStatus.failed,
                "reason_code": None if passed else "oracle_mismatch",
                "attrs": {
                    "oracle": oracle,
                    "score": getattr(metrics, f"{oracle}_score", None),
                },
                "findings": [
                    Finding(
                        code="oracle_mismatch", severity=Severity.important, message=m
                    )
                    for m in (getattr(metrics, f"{oracle}_issues", None) or [])
                ],
            }
        )
    return events


def _apply_event_from_outcome(outcome: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Map an encode ``outcome`` dict to an apply-stage event, or ``None``."""
    if not outcome:
        return None
    final_success = outcome.get("final_success")
    status = outcome.get("status")
    if final_success is True:
        stage_status = StageStatus.passed
    elif final_success is False:
        stage_status = StageStatus.failed
    else:
        stage_status = StageStatus.skipped
    return {
        "stage": "apply",
        "status": stage_status,
        "reason_code": status if stage_status is not StageStatus.passed else None,
        "reason": outcome.get("apply_error"),
        "attrs": {
            "status": status,
            "applied_files": outcome.get("applied_files"),
            "standalone_validation_success": outcome.get(
                "standalone_validation_success"
            ),
        },
    }


def emit_live_encode_events(
    result: Any,
    run_id: str,
    outcome: dict[str, Any],
    *,
    total_duration_ms: int | None = None,
    log_dir: Optional[Path] = None,
) -> RunLogWriter:
    """Emit generate + gate + apply events for a just-completed encode run.

    Duck-typed on ``result`` (an ``EvalResult``) so this module does not import
    harness types. Never raises: the writer swallows its own errors.
    """
    writer = RunLogWriter(run_id, log_dir=log_dir)
    generation_ok = bool(getattr(result, "success", False)) or getattr(
        result, "output_file", None
    )
    writer.emit(
        "generate",
        StageStatus.passed if generation_ok else StageStatus.failed,
        reason=getattr(result, "error", None),
        duration_ms=(
            total_duration_ms
            if total_duration_ms is not None
            else getattr(result, "duration_ms", None)
        ),
        attrs={
            "citation": getattr(result, "citation", None),
            "legal_id": None,
            "backend": getattr(result, "backend", None),
            "model": getattr(result, "model", None),
            "runner": getattr(result, "runner", None),
            "generation_prompt_sha256": getattr(
                result, "generation_prompt_sha256", None
            ),
            "trace_ref": getattr(result, "trace_file", None),
            "corpus_context_ref": getattr(result, "context_manifest_file", None),
            "embedded_source_present": getattr(
                getattr(result, "metrics", None), "embedded_source_present", None
            ),
            "retry_count": getattr(result, "retry_count", None),
            "generation_escalation": (outcome or {}).get("generation_escalation"),
        },
    )
    for kwargs in _gate_events_from_metrics(getattr(result, "metrics", None)):
        writer.emit(**kwargs)
    apply_event = _apply_event_from_outcome(outcome or {})
    if apply_event is not None:
        writer.emit(**apply_event)
    return writer


# ---------------------------------------------------------------------------
# Backfill (source-backed only; absent -> null, never fabricated)
# ---------------------------------------------------------------------------


def build_manifest_index(repo_paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Index signed apply-manifests by ``run_id`` across rulespec repo checkouts."""
    index: dict[str, dict[str, Any]] = {}
    invalid: list[str] = []
    seen_repos: set[Path] = set()
    for raw_repo_path in repo_paths:
        raw_repo_path = Path(raw_repo_path).expanduser()
        if not is_composition_policy_repo_root(raw_repo_path):
            raise ValueError(
                "Run-log manifest indexing requires an explicit exact canonical "
                f"rulespec-<country> checkout: {raw_repo_path}"
            )
        repo_path = raw_repo_path.resolve(strict=True)
        if repo_path in seen_repos:
            raise ValueError(f"Duplicate RuleSpec checkout: {repo_path}")
        seen_repos.add(repo_path)
        root = repo_path / APPLY_MANIFEST_GLOB
        if not root.is_dir():
            continue
        path_migration_receipt_proof_cache: dict[tuple[str, str], tuple[str, ...]] = {}
        for manifest_path in root.rglob("*.json"):
            from .cli import _load_verified_applied_encoding_manifest_payload

            payload, _root_prefix, manifest_sha256, issues = (
                _load_verified_applied_encoding_manifest_payload(
                    repo_path,
                    manifest_path.relative_to(repo_path).as_posix(),
                    path_migration_receipt_proof_cache=(
                        path_migration_receipt_proof_cache
                    ),
                )
            )
            if issues or payload is None:
                invalid.append(
                    f"{manifest_path}: "
                    + ("; ".join(issues) or "manifest is not canonical")
                )
                continue
            payload = dict(payload)
            run_id = payload.get("run_id")
            if isinstance(run_id, str) and run_id:
                if run_id in index:
                    raise ValueError(
                        f"Duplicate apply-manifest run_id {run_id!r}: "
                        f"{index[run_id]['_manifest_path']} and {manifest_path}"
                    )
                assert manifest_sha256 is not None
                payload["_manifest_sha256"] = manifest_sha256
                payload["_manifest_path"] = str(manifest_path)
                index[run_id] = payload
            else:
                invalid.append(f"{manifest_path}: manifest has no run_id")
    if invalid:
        raise ValueError("Invalid apply manifests:\n" + "\n".join(invalid))
    return index


def _manifest_sha_chain(payload: Optional[dict[str, Any]]) -> Optional[list[str]]:
    if not payload:
        return None
    manifest_sha256 = payload.get("_manifest_sha256") or ""
    return [manifest_sha256] if manifest_sha256 else None


def synthesize_backfill_events(
    run: Any, manifest: Optional[dict[str, Any]]
) -> list[RunLogEvent]:
    """Reconstruct run-log events for one historical ``EncodingRun``.

    Only source-backed verdicts are emitted. The oracle stage uses the recorded
    match score with the exact-match convention (score == 1.0 => passed); the raw
    score is always preserved in ``attrs`` so nothing is hidden.
    """
    ts = run.timestamp.isoformat() if getattr(run, "timestamp", None) else None
    seq = 0
    events: list[RunLogEvent] = []

    def _event(stage: str, status: StageStatus, **kw: Any) -> None:
        nonlocal seq
        findings = kw.pop("findings", []) or []
        events.append(
            RunLogEvent(
                run_id=run.id,
                seq=seq,
                ts=ts or datetime.now(timezone.utc).isoformat(),
                stage=stage,
                status=status,
                reason_code=kw.pop("reason_code", None),
                reason=kw.pop("reason", None),
                duration_ms=kw.pop("duration_ms", None),
                attrs=kw.pop("attrs", {}),
                findings=findings,
            )
        )
        seq += 1

    outcome = getattr(run, "outcome", None) or {}
    # Only claim generate=passed when it is sourced. The recorded outcome status
    # `apply_blocked_generation` is the one historical signal that generation
    # itself failed; every other status (validated/applied/blocked-at-validation
    # /blocked-at-manifest) implies a generated artifact reached a later stage.
    generation_failed = outcome.get("status") == "apply_blocked_generation"
    _event(
        "generate",
        StageStatus.failed if generation_failed else StageStatus.passed,
        reason_code="generation_failed" if generation_failed else None,
        reason=outcome.get("apply_error") if generation_failed else None,
        duration_ms=getattr(run, "total_duration_ms", None) or None,
        attrs={
            "citation": getattr(run, "citation", None),
            "legal_id": None,
            "backend": (manifest or {}).get("backend")
            or getattr(run, "agent_type", None),
            "model": (manifest or {}).get("model") or getattr(run, "agent_model", None),
            "generation_prompt_sha256": (manifest or {}).get(
                "generation_prompt_sha256"
            ),
            "trace_ref": (manifest or {}).get("trace_file"),
            "trace_sha256": (manifest or {}).get("trace_sha256"),
            "corpus_context_ref": (manifest or {}).get("context_manifest_file"),
            "corpus_context_sha256": (manifest or {}).get("context_manifest_sha256"),
            "generation_escalation": outcome.get("generation_escalation"),
            "backfilled": True,
        },
    )

    # gate.ci from the recorded standalone-validation signal, when present.
    standalone = outcome.get("standalone_validation_success")
    if standalone is not None:
        _event(
            "gate.ci",
            StageStatus.passed if standalone else StageStatus.failed,
            reason_code=None if standalone else "ci_failure",
        )

    review_results = getattr(run, "review_results", None)
    if review_results is not None:
        # gate.review only when reviewers actually ran.
        if getattr(review_results, "reviews", None):
            findings: list[Finding] = []
            for review in review_results.reviews:
                for kind in ("critical_issues", "important_issues", "minor_issues"):
                    for msg in getattr(review, kind, None) or []:
                        findings.append(
                            Finding(
                                code=f"{getattr(review, 'reviewer', 'review')}",
                                severity=_severity_for(kind.split("_")[0]),
                                message=str(msg),
                            )
                        )
            _event(
                "gate.review",
                StageStatus.passed if review_results.passed else StageStatus.failed,
                reason_code=None if review_results.passed else "reviewer_rejected",
                findings=findings,
            )
        # gate.oracle whenever a match score was recorded (independent of
        # reviewers). Exact-match convention; raw score preserved in attrs.
        for oracle, score in (
            ("policyengine", getattr(review_results, "policyengine_match", None)),
        ):
            if score is None:
                continue
            passed = score >= 1.0
            _event(
                "gate.oracle",
                StageStatus.passed if passed else StageStatus.failed,
                reason_code=None if passed else "oracle_mismatch",
                attrs={"oracle": oracle, "score": score, "convention": "exact_match"},
            )

    # apply from the recorded outcome, or from manifest presence (a manifest is
    # only written on a clean apply).
    final_success = outcome.get("final_success")
    if final_success is not None or manifest is not None:
        if final_success is True or (final_success is None and manifest is not None):
            status = StageStatus.passed
        elif final_success is False:
            status = StageStatus.failed
        else:
            status = StageStatus.skipped
        _event(
            "apply",
            status,
            reason_code=outcome.get("status")
            if status is not StageStatus.passed
            else None,
            reason=outcome.get("apply_error"),
            attrs={
                "status": outcome.get("status"),
                "manifest_path": (manifest or {}).get("_manifest_path"),
                "manifest_sha_chain": _manifest_sha_chain(manifest),
                "signature_key_id": ((manifest or {}).get("signature") or {}).get(
                    "key_id"
                ),
                "applied_files": [
                    f.get("path") for f in (manifest or {}).get("applied_files", [])
                ]
                or outcome.get("applied_files"),
            },
        )
    return events


def _parse_review_results(review_results_json: Optional[str]) -> Any:
    """Defensively parse ``review_results_json`` into a ReviewResults-shaped object.

    Old or malformed result shapes are rejected rather than translated.
    """
    if not review_results_json:
        return None
    from .harness.encoding_db import ReviewResult, ReviewResults

    try:
        data = json.loads(review_results_json)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    allowed_fields = {"reviews", "policyengine_match", "oracle_context", "lessons"}
    if set(data) - allowed_fields:
        return None

    def _as_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str) and value:
            return [value]
        return []

    reviews = []
    for r in data.get("reviews", []) if isinstance(data.get("reviews"), list) else []:
        if not isinstance(r, dict):
            continue
        reviews.append(
            ReviewResult(
                reviewer=str(r.get("reviewer", "")),
                passed=bool(r.get("passed", False)),
                critical_issues=_as_list(r.get("critical_issues")),
                important_issues=_as_list(r.get("important_issues")),
                minor_issues=_as_list(r.get("minor_issues")),
            )
        )
    return ReviewResults(
        reviews=reviews,
        policyengine_match=data.get("policyengine_match"),
    )


def _iter_db_runs(db_path: Path, limit: int) -> Iterable[Any]:
    """Yield lightweight run objects straight from encodings.db.

    Reads only the columns the run-log needs (skips the large ``rulespec_content``
    /``source_text`` blobs) and is robust to malformed ``iterations_json`` rows
    that break the full ``EncodingDB.get_recent_runs`` deserialiser.
    """
    import sqlite3
    from types import SimpleNamespace

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            "SELECT id, timestamp, citation, agent_type, agent_model, "
            "total_duration_ms, review_results_json, outcome_json "
            "FROM encoding_runs ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        for row in cursor:
            try:
                outcome = json.loads(row[7]) if row[7] else {}
            except (TypeError, json.JSONDecodeError):
                outcome = {}
            if not isinstance(outcome, dict):
                outcome = {}
            # A malformed/legacy timestamp must not abort the whole backfill; this
            # runs inside the generator, outside the caller's per-run try/except.
            try:
                ts = datetime.fromisoformat(row[1]) if row[1] else None
            except (ValueError, TypeError):
                ts = None
            yield SimpleNamespace(
                id=row[0],
                timestamp=ts,
                citation=row[2],
                agent_type=row[3],
                agent_model=row[4],
                total_duration_ms=row[5],
                review_results=_parse_review_results(row[6]),
                outcome=outcome,
            )
    finally:
        conn.close()


def export_backfill(
    db_path: Path,
    repo_paths: Iterable[Path],
    *,
    log_dir: Optional[Path] = None,
    limit: int = 100000,
) -> dict[str, Any]:
    """Backfill run-log JSONL for every historical run in ``db_path``.

    Returns a coverage report: overall counts plus per-field non-null coverage.
    """
    manifest_index = build_manifest_index(repo_paths)
    target_dir = run_log_dir(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    field_hits = {field: 0 for field in COVERAGE_FIELDS}
    exported = 0
    total = 0
    with_manifest = 0
    for run in _iter_db_runs(Path(db_path), limit):
        total += 1
        manifest = manifest_index.get(run.id)
        if manifest is not None:
            with_manifest += 1
        path = target_dir / f"{run.id}.jsonl"
        try:
            if path.exists() and not _log_is_backfill(path):
                # A richer live log already exists for this run; never clobber it
                # with a poorer backfill. Still tally its coverage.
                events = list(iter_events(path))
            else:
                events = synthesize_backfill_events(run, manifest)
                path.write_text(
                    "".join(event.to_json_line() + "\n" for event in events),
                    encoding="utf-8",
                )
        except Exception:  # noqa: BLE001 - one bad row must not stop the backfill
            continue
        exported += 1
        _tally_coverage(field_hits, run, manifest, events)

    return {
        "total_runs": total,
        "exported": exported,
        "with_manifest": with_manifest,
        "coverage_pct": {
            field: round(100.0 * hits / total, 1) if total else 0.0
            for field, hits in field_hits.items()
        },
        "log_dir": str(target_dir),
    }


def _tally_coverage(
    field_hits: dict[str, int],
    run: Any,
    manifest: Optional[dict[str, Any]],
    events: list[RunLogEvent],
) -> None:
    stages = {e.stage: e for e in events}
    generate_attrs = stages["generate"].attrs if "generate" in stages else {}
    checks = {
        "run_id": bool(getattr(run, "id", None)),
        "citation": bool(getattr(run, "citation", None)),
        "backend": bool(generate_attrs.get("backend")),
        "model": bool(generate_attrs.get("model")),
        "generation_prompt_sha256": bool(
            generate_attrs.get("generation_prompt_sha256")
        ),
        "trace_ref": bool(generate_attrs.get("trace_ref")),
        "corpus_context_ref": bool(generate_attrs.get("corpus_context_ref")),
        "gate.compile": "gate.compile" in stages,
        "gate.ci": "gate.ci" in stages,
        "gate.oracle": "gate.oracle" in stages,
        "gate.review": "gate.review" in stages,
        "judge": "judge" in stages,
        "apply": "apply" in stages,
        "manifest_sha_chain": bool(manifest),
        "pr": "pr" in stages,
        "merge": "merge" in stages,
        "oracle_at_merge": "oracle_at_merge" in stages,
    }
    for field, hit in checks.items():
        if hit:
            field_hits[field] += 1


# ---------------------------------------------------------------------------
# Publication (committed dashboard artifacts) + staleness guard
# ---------------------------------------------------------------------------


def _iter_run_log_files(log_dir: Path) -> list[Path]:
    return sorted(Path(log_dir).glob("*.jsonl"))


def _log_is_backfill(path: Path) -> bool:
    """True when a run-log file was produced by backfill (safe to overwrite).

    A live log's ``generate`` event has no ``backfilled`` flag; a backfill log
    sets ``attrs.backfilled = true``. Unreadable/empty files are treated as
    backfill (overwritable).
    """
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            first = handle.readline().strip()
        if not first:
            return True
        return bool(json.loads(first).get("attrs", {}).get("backfilled"))
    except (OSError, json.JSONDecodeError):
        return True


def _compute_aggregates(records: list[Any]) -> dict[str, Any]:
    """Precompute the funnel + failure Pareto so the renderer is trivially fed."""
    funnel = {bucket: 0 for bucket, _ in FUNNEL_STEPS}
    stage_status_counts: dict[str, dict[str, int]] = {}
    failure_pareto: dict[str, int] = {}
    for record in records:
        for bucket in funnel:
            if record.funnel.get(bucket):
                funnel[bucket] += 1
        for stage, status in record.stage_status.items():
            stage_status_counts.setdefault(stage, {})
            stage_status_counts[stage][status] = (
                stage_status_counts[stage].get(status, 0) + 1
            )
        if record.first_failure_stage:
            key = (
                f"{record.first_failure_stage}:"
                f"{record.first_failure_reason_code or 'unspecified'}"
            )
            failure_pareto[key] = failure_pareto.get(key, 0) + 1
    pareto = [
        {
            "key": k,
            "stage": k.split(":", 1)[0],
            "reason_code": k.split(":", 1)[1],
            "count": v,
        }
        for k, v in sorted(failure_pareto.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return {
        "funnel": [
            {"bucket": b, "stage": s, "count": funnel[b]} for b, s in FUNNEL_STEPS
        ],
        "stage_status_counts": stage_status_counts,
        "failure_pareto": pareto,
    }


def _read_events_tolerant(path: Path) -> tuple[list[RunLogEvent], int]:
    """Read a run-log JSONL file, skipping (and counting) malformed lines.

    Used by :func:`publish` and :func:`check_staleness` so a single corrupt or
    truncated line (e.g. a crash mid-append) can't abort publication or the
    freshness guard - the guard whose whole purpose is to notice publication
    stopping must not itself crash. The strict :func:`iter_events` is retained
    for correctness-sensitive callers.
    """
    events: list[RunLogEvent] = []
    skipped = 0
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(RunLogEvent.from_json_line(line))
                except (json.JSONDecodeError, ValidationError, ValueError):
                    skipped += 1
    except OSError:
        pass
    return events, skipped


def publish(
    log_dir: Path, out_dir: Path, *, max_detail_runs: int = 800
) -> dict[str, Any]:
    """Fold every run-log JSONL into the committed dashboard artifacts.

    Writes, into ``out_dir`` (the dashboard ``public/data`` dir):
      * ``run_log_pipeline.json`` - the machine-readable stage DAG (renderer input)
      * ``run_log_runs.json`` - folded per-run summaries + precomputed aggregates
        (funnel + failure Pareto); this is all the renderer needs
      * ``run_log_events.jsonl`` - raw events for the most recent ``max_detail_runs``
        runs (per-run findings drill-down); capped so the committed file stays small
      * ``run_log_publication.json`` - freshness metadata for the staleness guard
    """
    log_dir = Path(log_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[Any] = []
    per_run_events: dict[str, list[RunLogEvent]] = {}
    newest_event_ts: Optional[str] = None
    total_events = 0
    skipped_lines = 0
    for path in _iter_run_log_files(log_dir):
        events, skipped = _read_events_tolerant(path)
        skipped_lines += skipped
        if not events:
            continue
        record = fold_run(events)
        records.append(record)
        per_run_events[record.run_id] = events
        total_events += len(events)
        for event in events:
            if newest_event_ts is None or event.ts > newest_event_ts:
                newest_event_ts = event.ts

    records.sort(key=lambda r: r.started_at or "", reverse=True)

    (out_dir / RUNS_INDEX_FILE).write_text(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "run_count": len(records),
                "funnel_steps": [{"bucket": b, "stage": s} for b, s in FUNNEL_STEPS],
                "aggregates": _compute_aggregates(records),
                "runs": [r.to_dict() for r in records],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    detail_events = 0
    with (out_dir / EVENTS_FILE).open("w", encoding="utf-8") as events_out:
        for record in records[:max_detail_runs]:
            for event in per_run_events.get(record.run_id, []):
                events_out.write(event.to_json_line() + "\n")
                detail_events += 1

    (out_dir / PIPELINE_SPEC_FILE).write_text(
        json.dumps(pipeline_spec_dict(), indent=2, sort_keys=True) + "\n"
    )
    meta = {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(records),
        "event_count": total_events,
        "detail_event_count": detail_events,
        "detail_run_cap": max_detail_runs,
        "skipped_malformed_lines": skipped_lines,
        "newest_event_ts": newest_event_ts,
        "source_log_dir": str(log_dir),
    }
    (out_dir / PUBLICATION_META_FILE).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    return meta


def check_staleness(
    out_dir: Path,
    log_dir: Path,
    *,
    max_lag_hours: float = 24.0,
) -> dict[str, Any]:
    """Freshness guard analog to corpus's publication staleness check.

    Fails (``ok=False``) when the newest source run-log event is materially newer
    than the newest *published* event - i.e. publication has silently stopped -
    beyond ``max_lag_hours``. Also fails when nothing has ever been published but
    source logs exist.
    """
    out_dir = Path(out_dir)
    log_dir = Path(log_dir)

    def _newest_source_ts() -> Optional[str]:
        newest: Optional[str] = None
        for path in _iter_run_log_files(log_dir):
            events, _skipped = _read_events_tolerant(path)
            for event in events:
                if newest is None or event.ts > newest:
                    newest = event.ts
        return newest

    source_ts = _newest_source_ts()
    meta_path = out_dir / PUBLICATION_META_FILE
    if not meta_path.exists():
        return {
            "ok": source_ts is None,
            "reason": "never_published" if source_ts else "no_source",
            "source_newest": source_ts,
            "published_newest": None,
            "lag_hours": None,
        }

    meta = json.loads(meta_path.read_text())
    published_ts = meta.get("newest_event_ts")
    if source_ts is None:
        return {"ok": True, "reason": "no_source", "published_newest": published_ts}

    lag_hours = None
    if published_ts:
        lag = _parse_iso(source_ts) - _parse_iso(published_ts)
        lag_hours = lag.total_seconds() / 3600.0
    ok = published_ts is not None and (lag_hours or 0) <= max_lag_hours
    return {
        "ok": ok,
        "reason": "fresh" if ok else "stale",
        "source_newest": source_ts,
        "published_newest": published_ts,
        "lag_hours": round(lag_hours, 2) if lag_hours is not None else None,
        "max_lag_hours": max_lag_hours,
    }


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
