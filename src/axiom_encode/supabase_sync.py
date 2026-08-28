"""
Supabase sync for axiom_encode encoding runs.

Syncs local SQLite encoding DB to Supabase for the public dashboard.
"""

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from supabase import Client, create_client

from .harness.encoding_db import RUN_COST_COLUMNS, TokenUsage

ENCODINGS_SCHEMA = "encodings"
TELEMETRY_SCHEMA = "telemetry"

# Where a run's scores/telemetry came from. Kept as a hard allowlist to
# prevent fake data from reaching the dashboard:
#   reviewer_agent - scores from actual reviewer agent runs
#   ci_only        - only CI tests ran, no reviewer scores
#   mock           - fake/placeholder data for testing
#   manual_estimate- human-estimated scores (NOT from agents)
#   apply_manifest - reconstructed from a signed applied-rulespec manifest
#                    committed in a rulespec repo (real applied encoding;
#                    session-level telemetry was not captured)
VALID_DATA_SOURCES = {
    "reviewer_agent",
    "ci_only",
    "mock",
    "manual_estimate",
    "apply_manifest",
}

APPLY_MANIFEST_DIR = Path(".axiom") / "encoding-manifests"


def _review_results_to_scores(review_results) -> dict:
    """Convert checklist review results to the dashboard score shape."""
    scores = {
        "policyengine_match": review_results.policyengine_match,
    }
    reviewer_keys = {
        "rulespec": "rulespec",
        "rulespec_reviewer": "rulespec",
        "formula": "formula",
        "formula_reviewer": "formula",
        "parameter": "parameter",
        "parameter_reviewer": "parameter",
        "integration": "integration",
        "integration_reviewer": "integration",
    }
    for review in review_results.reviews:
        key = reviewer_keys.get(review.reviewer)
        if not key:
            continue
        if review.items_checked:
            scores[key] = review.items_passed / review.items_checked * 10
        else:
            scores[key] = 10 if review.passed else 0
    return scores


def get_supabase_client(*, require_write: bool = True) -> Client:
    """Get Supabase client using environment variables."""
    url = os.environ.get("AXIOM_ENCODE_SUPABASE_URL")
    if require_write:
        key = os.environ.get("AXIOM_ENCODE_SUPABASE_SECRET_KEY")
    else:
        key = os.environ.get("AXIOM_ENCODE_SUPABASE_SECRET_KEY") or os.environ.get(
            "AXIOM_ENCODE_SUPABASE_ANON_KEY"
        )

    if not url or not key:
        if require_write:
            raise ValueError(
                "Missing Supabase write credentials. Set AXIOM_ENCODE_SUPABASE_URL "
                "and AXIOM_ENCODE_SUPABASE_SECRET_KEY."
            )
        raise ValueError(
            "Missing Supabase credentials. Set AXIOM_ENCODE_SUPABASE_URL and "
            "AXIOM_ENCODE_SUPABASE_SECRET_KEY "
            "(or AXIOM_ENCODE_SUPABASE_ANON_KEY for read-only)."
        )

    return create_client(url, key)


def sync_run_to_supabase(
    run: "EncodingRun",
    data_source: str,  # REQUIRED: 'reviewer_agent', 'ci_only', 'mock', 'manual_estimate'
    client: Optional[Client] = None,
) -> bool:
    """
    Sync a single encoding run to Supabase.

    Args:
        run: The EncodingRun to sync
        data_source: REQUIRED - one of 'reviewer_agent', 'ci_only', 'mock', 'manual_estimate'
        client: Optional Supabase client (creates one if not provided)

    Returns:
        True if sync succeeded
    """

    # Validate data_source - this is a hard requirement to prevent fake data
    valid_sources = VALID_DATA_SOURCES
    if data_source not in valid_sources:
        raise ValueError(
            f"data_source must be one of {valid_sources}, got: {data_source}"
        )

    if client is None:
        client = get_supabase_client()

    # Convert to Supabase format
    encoder_version = getattr(run, "axiom_encode_version", "") or ""
    if not isinstance(encoder_version, str):
        encoder_version = ""
    outcome = _run_outcome(run)

    data = {
        "id": run.id,
        "timestamp": run.timestamp.isoformat(),
        "citation": run.citation,
        "file_path": run.file_path,
        "complexity": {
            "cross_references": run.complexity.cross_references,
            "has_nested_structure": run.complexity.has_nested_structure,
            "has_numeric_thresholds": run.complexity.has_numeric_thresholds,
            "has_phase_in_out": run.complexity.has_phase_in_out,
            "estimated_variables": run.complexity.estimated_variables,
            "estimated_parameters": run.complexity.estimated_parameters,
        },
        "iterations": [
            {
                "attempt": it.attempt,
                "duration_ms": it.duration_ms,
                "success": it.success,
                "errors": [
                    {
                        "error_type": e.error_type,
                        "message": e.message,
                        "variable": e.variable,
                        "fix_applied": e.fix_applied,
                    }
                    for e in it.errors
                ],
            }
            for it in run.iterations
        ],
        "total_duration_ms": run.total_duration_ms,
        "agent_type": run.agent_type,
        "agent_model": run.agent_model,
        "rulespec_content": run.rulespec_content,
        "session_id": run.session_id,
        "encoder_version": encoder_version or None,
        "has_issues": _run_has_issues(run),
        "note": _run_issue_note(run),
        "synced_at": datetime.now().isoformat(),
        "data_source": data_source,
    }
    # Ship only ledger fields that were actually measured; omitted fields stay
    # NULL remotely, so "unknown" never reads as a real zero (manifest-only
    # runs, for example, carry no telemetry at all).
    tokens = getattr(run, "tokens", None)
    if isinstance(tokens, TokenUsage) and tokens.has_recorded_usage:
        data.update(
            {
                "input_tokens": tokens.input_tokens,
                "output_tokens": tokens.output_tokens,
                "cache_read_tokens": tokens.cache_read_tokens,
                "cache_creation_tokens": tokens.cache_creation_tokens,
                "reasoning_output_tokens": tokens.reasoning_output_tokens,
            }
        )
    estimated_cost_usd = getattr(run, "estimated_cost_usd", None)
    if isinstance(estimated_cost_usd, (int, float)) and not isinstance(
        estimated_cost_usd, bool
    ):
        data["estimated_cost_usd"] = float(estimated_cost_usd)
    actual_cost_usd = getattr(run, "actual_cost_usd", None)
    if isinstance(actual_cost_usd, (int, float)) and not isinstance(
        actual_cost_usd, bool
    ):
        data["actual_cost_usd"] = float(actual_cost_usd)
    generation_attempt_count = getattr(run, "generation_attempt_count", 0)
    if isinstance(generation_attempt_count, int) and generation_attempt_count > 0:
        data["generation_attempt_count"] = generation_attempt_count
    if outcome:
        data["outcome"] = outcome

    if run.review_results:
        data["scores"] = _review_results_to_scores(run.review_results)

    # Retry with column groups removed only when Supabase reports an unknown
    # column (a schema that predates a migration). Each tier starts from the
    # full payload so a missing "outcome" column cannot cost the run its
    # ledger data, and any non-schema error fails the sync loudly.
    last_error: Exception | None = None
    attempted: list[dict] = []
    for dropped in (
        (),
        RUN_COST_COLUMNS,
        ("outcome",),
        (*RUN_COST_COLUMNS, "outcome"),
    ):
        payload = {key: value for key, value in data.items() if key not in dropped}
        if any(payload == prior for prior in attempted):
            continue
        attempted.append(payload)
        try:
            result = _upsert_encoding_run(client, payload)
        except Exception as e:
            last_error = e
            if _is_missing_column_error(e):
                continue
            break
        if dropped:
            print(
                f"Synced run {run.id} without {'/'.join(dropped)} "
                f"after Supabase rejected them: {last_error}"
            )
        return len(result.data) > 0
    print(f"Error syncing run {run.id}: {last_error}")
    return False


def _is_missing_column_error(error: Exception) -> bool:
    """True when a Supabase/PostgREST error reports an unknown column."""
    text = str(error)
    return (
        "PGRST204" in text
        or "Could not find the" in text
        or "does not exist" in text
    )


def _upsert_encoding_run(client: Client, data: dict):
    # Upsert to handle both new and updated runs.
    return client.schema(ENCODINGS_SCHEMA).table("encoding_runs").upsert(data).execute()


def _run_outcome(run: "EncodingRun") -> dict:
    outcome = getattr(run, "outcome", None)
    if isinstance(outcome, dict):
        return outcome
    return {}


def _run_has_issues(run: "EncodingRun") -> bool:
    outcome = _run_outcome(run)
    if outcome.get("final_success") is False:
        return True
    skip_iteration_issues = outcome.get("final_success") is True

    iterations = getattr(run, "iterations", None)
    if isinstance(iterations, list) and not skip_iteration_issues:
        if iterations and not getattr(iterations[-1], "success", False):
            return True
        for iteration in iterations:
            errors = getattr(iteration, "errors", None)
            if isinstance(errors, list) and errors:
                return True

    review_results = getattr(run, "review_results", None)
    reviews = getattr(review_results, "reviews", None)
    if isinstance(reviews, list):
        for review in reviews:
            critical_issues = getattr(review, "critical_issues", None)
            if isinstance(critical_issues, list) and critical_issues:
                return True
            important_issues = getattr(review, "important_issues", None)
            if isinstance(important_issues, list):
                if any(
                    not _is_non_blocking_review_note(issue)
                    for issue in important_issues
                ):
                    return True
            if getattr(review, "passed", True) is False:
                return True

    return False


def _is_non_blocking_review_note(issue: object) -> bool:
    return isinstance(issue, str) and issue.lstrip().startswith("[non-blocking]")


def _run_issue_note(run: "EncodingRun") -> str | None:
    notes: list[str] = []
    outcome = _run_outcome(run)
    final_success = outcome.get("final_success")
    if final_success is True:
        if (
            outcome.get("standalone_validation_success") is False
            and outcome.get("apply_success") is True
        ):
            notes.append("Standalone validation failed; overlay apply succeeded.")
    else:
        status = outcome.get("status")
        if isinstance(status, str) and status:
            notes.append(f"status={status}")
        apply_error = outcome.get("apply_error")
        if isinstance(apply_error, str) and apply_error:
            notes.append(apply_error)

    iterations = getattr(run, "iterations", None)
    if isinstance(iterations, list) and final_success is not True:
        for iteration in iterations:
            errors = getattr(iteration, "errors", None)
            if not isinstance(errors, list):
                continue
            for error in errors:
                message = getattr(error, "message", "")
                if isinstance(message, str) and message:
                    notes.append(message)

    review_results = getattr(run, "review_results", None)
    reviews = getattr(review_results, "reviews", None)
    if isinstance(reviews, list):
        for review in reviews:
            for field in ("critical_issues", "important_issues"):
                issues = getattr(review, field, None)
                if isinstance(issues, list):
                    notes.extend(str(issue) for issue in issues if issue)

    if not notes:
        return None
    return "; ".join(notes)[:2000]


def sync_all_runs(
    db_path: Path, data_source: str, client: Optional[Client] = None
) -> dict:
    """
    Sync all runs from local SQLite to Supabase.

    Args:
        db_path: Path to local encodings.db
        data_source: REQUIRED - one of 'reviewer_agent', 'ci_only', 'mock', 'manual_estimate'
        client: Optional Supabase client

    Returns:
        Dict with sync stats
    """
    from .harness.encoding_db import EncodingDB

    if client is None:
        client = get_supabase_client()

    db = EncodingDB(db_path)
    runs = db.get_recent_runs(limit=1000)  # Get all runs

    synced = 0
    failed = 0

    for run in runs:
        if sync_run_to_supabase(run, data_source, client):
            synced += 1
        else:
            failed += 1

    return {
        "total": len(runs),
        "synced": synced,
        "failed": failed,
    }


def _manifest_rel_key(manifest_path: Path) -> str:
    """Machine-independent identity for a manifest: its path below the
    encoding-manifests directory (absolute checkout paths differ per host)."""
    parts = list(Path(manifest_path).parts)
    if "encoding-manifests" in parts:
        index = parts.index("encoding-manifests")
        return "/".join(parts[index + 1 :])
    return Path(manifest_path).name


def find_apply_manifests(repo_path: Path) -> list[Path]:
    """List applied-rulespec manifests committed in a rulespec repo checkout."""
    manifest_root = Path(repo_path) / APPLY_MANIFEST_DIR
    if not manifest_root.is_dir():
        return []
    return sorted(manifest_root.rglob("*.json"))


def run_from_apply_manifest(manifest_path: Path, payload: dict) -> "EncodingRun":
    """
    Reconstruct an EncodingRun from a signed applied-rulespec manifest.

    Apply manifests are the durable record of encodings that landed in a
    rulespec repo. Reusing the manifest's original run_id keeps the sync
    idempotent and lets a later, richer sync of the same run (from a local
    encodings.db) upsert over the reconstruction.
    """
    from .harness.encoding_db import EncodingRun, Iteration

    if payload.get("schema_version") != "axiom-encode/applied-rulespec/v5":
        raise ValueError(f"{manifest_path}: manifest is not canonical v3")

    citation = payload.get("citation")
    generated_at = payload.get("generated_at")
    if not isinstance(citation, str) or not citation:
        raise ValueError(f"{manifest_path}: manifest has no citation")
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError(f"{manifest_path}: manifest has no generated_at")

    # Only trust run_id as an upsert key when it looks like a real per-run id
    # (uuid4 prefix). Missing, malformed, or shared values would make unrelated
    # manifest rows overwrite each other.
    run_id = payload.get("run_id")
    if not (isinstance(run_id, str) and re.fullmatch(r"[0-9a-f]{8}", run_id)):
        digest_source = (
            f"apply-manifest:{citation}:{generated_at}:"
            f"{_manifest_rel_key(manifest_path)}:{run_id or ''}"
        )
        run_id = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:8]

    applied_files = payload.get("applied_files")
    file_path = ""
    if isinstance(applied_files, list) and applied_files:
        first = applied_files[0]
        if isinstance(first, dict):
            file_path = str(first.get("path") or "")

    backend = payload.get("backend")
    agent_type = (
        f"{backend}:encoder" if isinstance(backend, str) and backend else "encoder"
    )

    encoder_version = payload.get("axiom_encode_version")
    if not isinstance(encoder_version, str):
        encoder_version = ""

    return EncodingRun(
        id=run_id,
        timestamp=datetime.fromisoformat(generated_at),
        citation=citation,
        file_path=file_path,
        iterations=[Iteration(attempt=1, duration_ms=0, errors=[], success=True)],
        total_duration_ms=0,
        agent_type=agent_type,
        agent_model=str(payload.get("model") or ""),
        axiom_encode_version=encoder_version,
        # The manifest is only written for encodings that applied cleanly.
        outcome={"final_success": True, "status": "applied"},
        session_id=None,
    )


def sync_applied_manifest_runs(
    repo_paths: list[Path],
    *,
    client: Optional[Client] = None,
    dry_run: bool = False,
) -> dict:
    """
    Backfill encodings.encoding_runs from applied-rulespec manifests.

    Scans each rulespec repo checkout for .axiom/encoding-manifests/**/*.json
    and upserts one run per manifest with data_source="apply_manifest".
    Idempotent: rows are keyed by the manifest's original run_id.
    """
    stats = {"total": 0, "synced": 0, "failed": 0, "skipped": 0, "preserved": 0}
    runs: list["EncodingRun"] = []
    for repo_path in repo_paths:
        repo_path = Path(repo_path).resolve()
        path_migration_receipt_proof_cache: dict[tuple[str, str], tuple[str, ...]] = {}
        for manifest_path in find_apply_manifests(repo_path):
            stats["total"] += 1
            try:
                from .cli import _load_verified_applied_encoding_manifest_payload

                payload, _root_prefix, _manifest_sha256, issues = (
                    _load_verified_applied_encoding_manifest_payload(
                        repo_path,
                        manifest_path.relative_to(repo_path).as_posix(),
                        path_migration_receipt_proof_cache=(
                            path_migration_receipt_proof_cache
                        ),
                    )
                )
                if issues or payload is None:
                    raise ValueError("; ".join(issues) or "manifest is not canonical")
                runs.append(run_from_apply_manifest(manifest_path, payload))
            except Exception as e:
                print(f"Invalid apply manifest {manifest_path}: {e}")
                stats["failed"] += 1

    if dry_run:
        for run in runs:
            print(
                f"would sync {run.id} {run.timestamp.isoformat()} "
                f"{run.citation} [{run.agent_type} {run.agent_model}]"
            )
        return stats

    if client is None:
        client = get_supabase_client()
    # Never overwrite rows that carry richer telemetry than a manifest
    # reconstruction (e.g. a reviewer_agent run with durations and a session).
    protected_ids = _non_manifest_run_ids(client)
    for run in runs:
        if run.id in protected_ids:
            stats["preserved"] += 1
        elif sync_run_to_supabase(run, "apply_manifest", client=client):
            stats["synced"] += 1
        else:
            stats["failed"] += 1
    return stats


def _non_manifest_run_ids(client: Client) -> set[str]:
    """Ids of encoding_runs whose data_source is anything but apply_manifest."""
    ids: set[str] = set()
    page_size = 1000
    offset = 0
    while True:
        result = (
            client.schema(ENCODINGS_SCHEMA)
            .table("encoding_runs")
            .select("id")
            .neq("data_source", "apply_manifest")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = result.data or []
        ids.update(row["id"] for row in rows if isinstance(row.get("id"), str))
        if len(rows) < page_size:
            return ids
        offset += page_size


def fetch_runs_from_supabase(
    limit: int = 20,
    citation: Optional[str] = None,
    client: Optional[Client] = None,
) -> list[dict]:
    """
    Fetch encoding runs from Supabase.

    Args:
        limit: Maximum runs to fetch
        citation: Optional filter by citation
        client: Optional Supabase client

    Returns:
        List of run records
    """
    if client is None:
        client = get_supabase_client(require_write=False)

    query = client.schema(ENCODINGS_SCHEMA).table("encoding_runs").select("*")

    if citation:
        query = query.eq("citation", citation)

    query = query.order("timestamp", desc=True).limit(limit)

    result = query.execute()
    return result.data


# ============================================================================
# Transcript Sync (from PostToolUse hook local DB to Supabase)
# ============================================================================

TRANSCRIPT_DB = Path.home() / "TheAxiomFoundation" / "axiom-encode" / "transcripts.db"


def sync_transcripts_to_supabase(
    session_id: Optional[str] = None,
    client: Optional[Client] = None,
) -> dict:
    """
    Sync agent transcripts from local SQLite to Supabase.

    Args:
        session_id: Optional filter by session (syncs all unsynced if None)
        client: Optional Supabase client

    Returns:
        Dict with sync stats
    """
    import sqlite3

    if not TRANSCRIPT_DB.exists():
        return {"total": 0, "synced": 0, "failed": 0, "error": "No local transcript DB"}

    if client is None:
        client = get_supabase_client()

    conn = sqlite3.connect(str(TRANSCRIPT_DB))
    conn.row_factory = sqlite3.Row

    # Get unsynced transcripts
    query = "SELECT * FROM agent_transcripts WHERE uploaded_at IS NULL"
    params = []
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)

    rows = conn.execute(query, params).fetchall()

    synced = 0
    failed = 0

    for row in rows:
        try:
            data = {
                "session_id": row["session_id"],
                "tool_use_id": row["tool_use_id"],
                "subagent_type": row["subagent_type"],
                "prompt": row["prompt"],
                "description": row["description"],
                "response_summary": row["response_summary"],
                "transcript": json.loads(row["transcript"])
                if row["transcript"]
                else None,
                "message_count": row["message_count"],
                "created_at": row["created_at"],
            }

            result = (
                client.schema(TELEMETRY_SCHEMA)
                .table("agent_transcripts")
                .upsert(data)
                .execute()
            )

            if result.data:
                # Mark as uploaded
                conn.execute(
                    "UPDATE agent_transcripts SET uploaded_at = ? WHERE id = ?",
                    (datetime.now().isoformat(), row["id"]),
                )
                conn.commit()
                synced += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error syncing transcript {row['tool_use_id']}: {e}")
            failed += 1

    conn.close()

    return {
        "total": len(rows),
        "synced": synced,
        "failed": failed,
    }


ENCODINGS_DB = Path.home() / "TheAxiomFoundation" / "axiom-encode" / "encodings.db"


def sync_agent_sessions_to_supabase(
    session_id: Optional[str] = None,
    client: Optional[Client] = None,
    include_all: bool = False,
    db_path: Optional[Path] = None,
) -> dict:
    """
    Sync agent sessions from encodings.db to Supabase.

    Args:
        session_id: Optional filter by session ID
        client: Optional Supabase client
        include_all: Sync every local session. By default only sessions that look
            tied to Axiom Encode are synced, to avoid uploading unrelated global
            agent transcript history.
        db_path: Optional path to an encodings.db override.

    Returns:
        Dict with sync stats
    """
    import sqlite3

    source_db = Path(db_path) if db_path is not None else ENCODINGS_DB
    if not source_db.exists():
        return {"total": 0, "synced": 0, "failed": 0, "error": "No encodings.db"}

    if client is None:
        client = get_supabase_client()

    conn = sqlite3.connect(str(source_db))
    conn.row_factory = sqlite3.Row

    session_columns = _sqlite_columns(conn, "sessions")

    # Get sessions tied to Axiom Encode. Older hook databases may include broad
    # global agent activity, so the default deliberately avoids syncing every row.
    query = "SELECT * FROM sessions"
    params = []
    if session_id:
        query = "SELECT * FROM sessions WHERE id = ?"
        params.append(session_id)
    elif not include_all:
        filters = []
        if "run_id" in session_columns:
            filters.append("COALESCE(run_id, '') != ''")
        if "axiom_encode_version" in session_columns:
            filters.append("COALESCE(axiom_encode_version, '') != ''")
        if "autorac_version" in session_columns:
            filters.append("COALESCE(autorac_version, '') != ''")
        if "cwd" in session_columns:
            filters.append("cwd LIKE '%/axiom-encode%'")
        query += f" WHERE {' OR '.join(filters) if filters else '0 = 1'}"

    sessions = conn.execute(query, params).fetchall()

    synced = 0
    failed = 0

    for session in sessions:
        try:
            # Get events for this session
            events = conn.execute(
                "SELECT * FROM session_events WHERE session_id = ? ORDER BY sequence",
                (session["id"],),
            ).fetchall()

            # Build session data
            session_data = {
                "id": session["id"],
                "started_at": session["started_at"],
                "ended_at": session["ended_at"],
                "model": session["model"],
                "cwd": session["cwd"],
                "event_count": session["event_count"],
                "input_tokens": session["input_tokens"],
                "output_tokens": session["output_tokens"],
                "cache_read_tokens": session["cache_read_tokens"],
                "estimated_cost_usd": (
                    float(session["estimated_cost_usd"])
                    if session["estimated_cost_usd"] is not None
                    else None
                ),
                "encoder_version": _row_value(
                    session, "axiom_encode_version", "autorac_version"
                )
                or None,
            }
            for token_column in ("cache_creation_tokens", "reasoning_output_tokens"):
                if token_column in session_columns:
                    session_data[token_column] = session[token_column]

            # Build events data
            events_data = [
                {
                    "id": e["id"],
                    "session_id": e["session_id"],
                    "sequence": e["sequence"],
                    "timestamp": e["timestamp"],
                    "event_type": e["event_type"],
                    "tool_name": e["tool_name"],
                    "content": e["content"][:10000]
                    if e["content"]
                    else None,  # Truncate large content
                    "metadata": json.loads(e["metadata_json"])
                    if e["metadata_json"]
                    else None,
                }
                for e in events
            ]

            # Upsert session. Retry without the newest token columns only when
            # Supabase reports an unknown column (a schema that predates
            # migration 007); any other failure counts the session as failed.
            try:
                client.schema(TELEMETRY_SCHEMA).table("sdk_sessions").upsert(
                    session_data
                ).execute()
            except Exception as session_error:
                fallback_session_data = {
                    key: value
                    for key, value in session_data.items()
                    if key
                    not in ("cache_creation_tokens", "reasoning_output_tokens")
                }
                if fallback_session_data == session_data or not (
                    _is_missing_column_error(session_error)
                ):
                    raise
                client.schema(TELEMETRY_SCHEMA).table("sdk_sessions").upsert(
                    fallback_session_data
                ).execute()

            # Upsert events (in batches of 100)
            for i in range(0, len(events_data), 100):
                batch = events_data[i : i + 100]
                client.schema(TELEMETRY_SCHEMA).table("sdk_session_events").upsert(
                    batch
                ).execute()

            synced += 1
            print(f"  Synced session {session['id']} ({len(events)} events)")

        except Exception as e:
            print(f"  Error syncing session {session['id']}: {e}")
            failed += 1

    conn.close()

    return {
        "total": len(sessions),
        "synced": synced,
        "failed": failed,
    }


def _sqlite_columns(conn, table_name: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")}


def _row_value(row, *keys: str):
    available = set(row.keys())
    for key in keys:
        if key in available:
            value = row[key]
            if value not in (None, ""):
                return value
    return None


def get_local_transcript_stats() -> dict:
    """Get stats about local transcript database."""
    import sqlite3

    if not TRANSCRIPT_DB.exists():
        return {"exists": False}

    conn = sqlite3.connect(str(TRANSCRIPT_DB))

    total = conn.execute("SELECT COUNT(*) FROM agent_transcripts").fetchone()[0]
    unsynced = conn.execute(
        "SELECT COUNT(*) FROM agent_transcripts WHERE uploaded_at IS NULL"
    ).fetchone()[0]
    by_type = conn.execute(
        "SELECT subagent_type, COUNT(*) FROM agent_transcripts GROUP BY subagent_type"
    ).fetchall()

    conn.close()

    return {
        "exists": True,
        "total": total,
        "unsynced": unsynced,
        "synced": total - unsynced,
        "by_type": dict(by_type),
    }


if __name__ == "__main__":  # pragma: no cover
    # CLI usage: python -m axiom_encode.supabase_sync <db_path> <data_source>
    # data_source is REQUIRED to prevent syncing fake data
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m axiom_encode.supabase_sync runs <db_path> <data_source>")
        print("  python -m axiom_encode.supabase_sync transcripts [session_id]")
        print("  python -m axiom_encode.supabase_sync stats")
        print("")
        print("data_source for runs must be one of:")
        print("  reviewer_agent  - Scores from actual reviewer agent runs")
        print("  ci_only         - Only CI tests ran, no reviewer scores")
        print("  mock            - Fake/placeholder data for testing")
        print("  manual_estimate - Human-estimated scores (NOT from agents)")
        print("  apply_manifest  - Reconstructed from signed apply manifests")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "runs":
        if len(sys.argv) < 4:
            print(
                "Usage: python -m axiom_encode.supabase_sync runs <db_path> <data_source>"
            )
            sys.exit(1)
        db_path = Path(sys.argv[2])
        data_source = sys.argv[3]
        print(f"Syncing runs from {db_path} with data_source={data_source}...")
        stats = sync_all_runs(db_path, data_source)
        print(
            f"Done! {stats['synced']} synced, {stats['failed']} failed of {stats['total']} total"
        )

    elif cmd == "transcripts":
        session_id = sys.argv[2] if len(sys.argv) > 2 else None
        print(
            f"Syncing transcripts{f' for session {session_id}' if session_id else ''}..."
        )
        stats = sync_transcripts_to_supabase(session_id)
        print(
            f"Done! {stats['synced']} synced, {stats['failed']} failed of {stats['total']} total"
        )

    elif cmd == "stats":
        stats = get_local_transcript_stats()
        if not stats.get("exists"):
            print("No local transcript database found")
        else:
            print(
                f"Local transcripts: {stats['total']} total, {stats['unsynced']} unsynced"
            )
            print(f"By agent type: {stats['by_type']}")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
