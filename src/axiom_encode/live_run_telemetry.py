"""
Live encode-run presence for the ops dashboard.

Maintains one row in encodings.live_encoding_runs per in-flight
`axiom-encode encode` invocation: inserted when the run starts, heartbeated
by a daemon thread while the run is active, and closed with a pointer to the
final encodings.encoding_runs row. The dashboard treats a 'running' row with
a stale heartbeat as a dead encoder.

Two transports, chosen automatically:

- **direct**: environments holding Supabase write credentials
  (`AXIOM_ENCODE_SUPABASE_URL` + `AXIOM_ENCODE_SUPABASE_SECRET_KEY`)
  write rows directly, as trusted telemetry.
- **ingest**: everyone else — including third-party encoders — reports
  credential-free to the public ops ingest endpoint, which stamps rows
  as self-reported. This is the default: no setup, no tokens.

Set `AXIOM_ENCODE_TELEMETRY=off` (or pass `--no-sync`) to opt out.
Telemetry is strictly best-effort: every network failure is swallowed so
neither a Supabase outage nor an unreachable ingest endpoint can ever
fail an encode.
"""

import getpass
import json
import os
import platform
import socket
import sys
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Optional

HEARTBEAT_INTERVAL_SECONDS = 30.0
DEFAULT_INGEST_URL = "https://axiom.org/api/ops/encoding/ingest"
INGEST_TIMEOUT_SECONDS = 10.0

_CI_ENV_VARS = ("CI", "GITHUB_ACTIONS", "BUILDKITE", "CIRCLECI")
_TELEMETRY_OFF_VALUES = {"off", "0", "false", "disabled"}
_TELEMETRY_ON_VALUES = {"on", "1", "true"}


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


def running_under_tests() -> bool:
    """Test-suite invocations of the encode path must never reach the real
    dashboard. The env marker alone is not enough: hermetic tests clear
    os.environ (in-process) or spawn the CLI with scrubbed envs, so the
    in-process signal is the pytest module itself.
    """
    return "pytest" in sys.modules or bool(os.environ.get("PYTEST_CURRENT_TEST"))


def telemetry_blocked_for_tests() -> bool:
    """True when test detection should suppress telemetry. The explicit
    `AXIOM_ENCODE_TELEMETRY=on` override bypasses detection — used by
    telemetry tests exercising the (mocked) transports, never in
    production config."""
    override = os.environ.get("AXIOM_ENCODE_TELEMETRY", "").strip().lower()
    return override not in _TELEMETRY_ON_VALUES and running_under_tests()


def telemetry_mode() -> str:
    """Resolve the transport: 'direct', 'ingest', or 'off'."""
    override = os.environ.get("AXIOM_ENCODE_TELEMETRY", "").strip().lower()
    if override in _TELEMETRY_OFF_VALUES:
        return "off"
    if telemetry_blocked_for_tests():
        return "off"
    if os.environ.get("AXIOM_ENCODE_SUPABASE_URL") and os.environ.get(
        "AXIOM_ENCODE_SUPABASE_SECRET_KEY"
    ):
        return "direct"
    return "ingest"


def _ingest_url() -> str:
    return (
        os.environ.get("AXIOM_ENCODE_TELEMETRY_INGEST_URL", "").strip()
        or DEFAULT_INGEST_URL
    )


class LiveRunTelemetry:
    """Context manager owning one live_encoding_runs row and its heartbeat.

    Constructed unconditionally; becomes a no-op when telemetry is off or
    the initial announce fails.
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
        self._mode = telemetry_mode() if enabled else "off"
        self._client = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._finished = False

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "LiveRunTelemetry":
        if self._mode == "off":
            return self
        started = (
            self._start_direct() if self._mode == "direct" else self._start_ingest()
        )
        if not started:
            self._mode = "off"
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
        self._send("heartbeat", {"attempt": attempt, "model": model})

    def set_phase(self, phase: str) -> None:
        self._send("heartbeat", {"phase": phase})

    def finish(self, status: str, run_id: Optional[str] = None) -> None:
        """Close the live row; idempotent, first call wins."""
        if self._finished:
            return
        self._finished = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        fields: dict = {"status": status}
        if run_id:
            fields["run_id"] = run_id
        self._send("finish", fields, force=True)

    # -- internals -----------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            self._send("heartbeat", {})

    def _send(self, kind: str, fields: dict, *, force: bool = False) -> None:
        if self._mode == "off" or (self._finished and not force):
            return
        try:
            if self._mode == "direct":
                self._send_direct(kind, fields)
            else:
                self._send_ingest(kind, fields)
        except Exception:
            # Best-effort: a missed heartbeat shows as staleness, nothing more.
            pass

    # -- direct transport (trusted environments) -----------------------------

    def _start_direct(self) -> bool:
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
            return True
        except Exception as exc:
            print(f"live-run telemetry disabled: {exc}", file=sys.stderr)
            return False

    def _send_direct(self, kind: str, fields: dict) -> None:
        if self._client is None:
            return
        from .supabase_sync import ENCODINGS_SCHEMA

        data = dict(fields)
        data["last_heartbeat_at"] = _utcnow()
        if kind == "finish":
            data["finished_at"] = _utcnow()
        self._client.schema(ENCODINGS_SCHEMA).table("live_encoding_runs").update(
            data
        ).eq("id", self.id).execute()

    # -- ingest transport (credential-free default) --------------------------

    def _start_ingest(self) -> bool:
        ok = self._post_ingest(
            {
                "op": "start",
                "id": self.id,
                "citation": self.citation,
                "backend": self.backend,
                "model": self.model,
                "attempt": 1,
                "encoder_version": self.encoder_version or None,
                "runner": runner_identity(),
            }
        )
        if not ok:
            print(
                "live-run telemetry disabled: ops ingest endpoint unreachable",
                file=sys.stderr,
            )
        return ok

    def _send_ingest(self, kind: str, fields: dict) -> None:
        self._post_ingest({"op": kind, "id": self.id, **fields})

    def _post_ingest(self, payload: dict) -> bool:
        try:
            request = urllib.request.Request(
                _ingest_url(),
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(
                request, timeout=INGEST_TIMEOUT_SECONDS
            ) as response:
                return 200 <= response.status < 300
        except Exception:
            return False
