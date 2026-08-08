"""
Live encode-run presence for the ops dashboard.

Maintains one row in encodings.live_encoding_runs per in-flight
`axiom-encode encode` invocation: inserted when the run starts, heartbeated
by a daemon thread while the run is active, and closed with a pointer to the
final encodings.encoding_runs row. The dashboard treats a 'running' row with
a stale heartbeat as a dead encoder.

Telemetry is strictly best-effort: every network failure is swallowed so a
Supabase outage can never fail an encode.
"""

import getpass
import os
import platform
import socket
import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

HEARTBEAT_INTERVAL_SECONDS = 30.0

_CI_ENV_VARS = ("CI", "GITHUB_ACTIONS", "BUILDKITE", "CIRCLECI")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def runner_identity() -> dict:
    """Machine identity attached to a live run (shown on the ops dashboard)."""
    try:
        hostname = socket.gethostname()
    except OSError:
        hostname = ""
    try:
        username = getpass.getuser()
    except (KeyError, OSError):
        username = ""
    return {
        "hostname": hostname,
        "username": username,
        "platform": platform.system().lower(),
        "pid": os.getpid(),
        "is_ci": any(os.environ.get(var) for var in _CI_ENV_VARS),
    }


def _live_telemetry_configured() -> bool:
    # Test-suite invocations of the encode path must never reach the real
    # dashboard: developer machines carry write credentials in their shell
    # environment, so credential presence alone cannot distinguish a real
    # encode from a pytest fixture run.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return bool(
        os.environ.get("AXIOM_ENCODE_SUPABASE_URL")
        and os.environ.get("AXIOM_ENCODE_SUPABASE_SECRET_KEY")
    )


class LiveRunTelemetry:
    """Context manager owning one live_encoding_runs row and its heartbeat.

    Constructed unconditionally; becomes a no-op when Supabase write
    credentials are absent or the initial insert fails.
    """

    def __init__(
        self,
        *,
        citation: str,
        backend: str,
        model: str,
        encoder_version: str,
        enabled: bool = True,
    ):
        self.id = f"live-{uuid.uuid4().hex[:12]}"
        self.citation = citation
        self.backend = backend
        self.model = model
        self.encoder_version = encoder_version
        self._enabled = enabled and _live_telemetry_configured()
        self._client = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._finished = False

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "LiveRunTelemetry":
        if not self._enabled:
            return self
        try:
            from .supabase_sync import ENCODINGS_SCHEMA, get_supabase_client

            self._client = get_supabase_client()
            now = _utcnow()
            self._client.schema(ENCODINGS_SCHEMA).table("live_encoding_runs").insert(
                {
                    "id": self.id,
                    "citation": self.citation,
                    "status": "running",
                    "started_at": now,
                    "last_heartbeat_at": now,
                    "backend": self.backend,
                    "model": self.model,
                    "attempt": 1,
                    "encoder_version": self.encoder_version or None,
                    "runner": runner_identity(),
                }
            ).execute()
        except Exception as exc:
            print(f"live-run telemetry disabled: {exc}", file=sys.stderr)
            self._enabled = False
            self._client = None
            return self
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="live-run-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Interrupted or crashed runs finish as 'failed'; normal completion
        # already called finish() with the real outcome.
        self.finish("failed" if exc_type is not None else "completed")

    # -- updates -------------------------------------------------------------

    def set_attempt(self, attempt: int, model: str) -> None:
        """Record a retry/escalation so the dashboard shows current state."""
        self.model = model
        self._update({"attempt": attempt, "model": model})

    def set_phase(self, phase: str) -> None:
        self._update({"phase": phase})

    def finish(self, status: str, run_id: Optional[str] = None) -> None:
        """Close the live row; idempotent, first call wins."""
        if self._finished:
            return
        self._finished = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        data = {
            "status": status,
            "finished_at": _utcnow(),
            "last_heartbeat_at": _utcnow(),
        }
        if run_id:
            data["run_id"] = run_id
        self._update(data, force=True)

    # -- internals -----------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            self._update({"last_heartbeat_at": _utcnow()})

    def _update(self, data: dict, *, force: bool = False) -> None:
        if self._client is None or (self._finished and not force):
            return
        try:
            from .supabase_sync import ENCODINGS_SCHEMA

            self._client.schema(ENCODINGS_SCHEMA).table("live_encoding_runs").update(
                data
            ).eq("id", self.id).execute()
        except Exception:
            # Best-effort: a missed heartbeat shows as staleness, nothing more.
            pass
