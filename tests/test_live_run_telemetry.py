"""Tests for live encode-run presence telemetry."""

import json
from unittest.mock import MagicMock, patch

from axiom_encode.live_run_telemetry import (
    LiveRunTelemetry,
    runner_identity,
    telemetry_mode,
)


def _mock_client():
    client = MagicMock()
    table = client.schema.return_value.table.return_value
    table.insert.return_value.execute.return_value = MagicMock()
    table.update.return_value.eq.return_value.execute.return_value = MagicMock()
    return client, table


def _configured_env(monkeypatch):
    monkeypatch.setenv("AXIOM_ENCODE_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("AXIOM_ENCODE_SUPABASE_SECRET_KEY", "secret")
    # These tests exercise the enabled path with mocked transports; the
    # explicit "on" override is the only way past the in-test detection.
    monkeypatch.setenv("AXIOM_ENCODE_TELEMETRY", "on")


def _ingest_env(monkeypatch):
    monkeypatch.delenv("AXIOM_ENCODE_SUPABASE_URL", raising=False)
    monkeypatch.delenv("AXIOM_ENCODE_SUPABASE_SECRET_KEY", raising=False)
    monkeypatch.setenv("AXIOM_ENCODE_TELEMETRY", "on")


def _mock_urlopen(status=204):
    response = MagicMock()
    response.status = status
    response.__enter__ = lambda self: self
    response.__exit__ = lambda self, *args: None
    return patch(
        "axiom_encode.live_run_telemetry.urllib.request.urlopen",
        return_value=response,
    )


def _ingest_payloads(urlopen_mock):
    return [
        json.loads(call.args[0].data.decode("utf-8"))
        for call in urlopen_mock.call_args_list
    ]


class TestRunnerIdentity:
    def test_contains_machine_fields(self):
        identity = runner_identity()
        assert set(identity) == {"hostname", "username", "platform", "pid", "is_ci"}
        assert isinstance(identity["pid"], int)
        assert isinstance(identity["is_ci"], bool)


class TestTelemetryMode:
    def test_defaults_to_ingest_without_credentials(self, monkeypatch):
        _ingest_env(monkeypatch)
        assert telemetry_mode() == "ingest"

    def test_direct_with_credentials(self, monkeypatch):
        _configured_env(monkeypatch)
        assert telemetry_mode() == "direct"

    def test_off_under_pytest_or_explicit_optout(self, monkeypatch):
        _ingest_env(monkeypatch)
        monkeypatch.setenv("AXIOM_ENCODE_TELEMETRY", "off")
        assert telemetry_mode() == "off"
        monkeypatch.setenv("AXIOM_ENCODE_TELEMETRY", "false")
        assert telemetry_mode() == "off"
        # Without the explicit "on" override, in-process test detection wins
        # even when hermetic tests have scrubbed every env marker: the
        # pytest module itself is the signal.
        monkeypatch.delenv("AXIOM_ENCODE_TELEMETRY")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        assert telemetry_mode() == "off"

    def test_explicit_off_beats_explicit_on_semantics(self, monkeypatch):
        _ingest_env(monkeypatch)
        assert telemetry_mode() == "ingest"
        monkeypatch.setenv("AXIOM_ENCODE_TELEMETRY", "disabled")
        assert telemetry_mode() == "off"


class TestLiveRunTelemetry:
    def test_ingest_mode_reports_lifecycle_without_credentials(self, monkeypatch):
        _ingest_env(monkeypatch)
        with _mock_urlopen() as urlopen_mock:
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.2.1670",
            ) as live:
                live.set_attempt(2, "gpt-5.5-max")
                live.finish("completed", run_id="abc12345")

        payloads = _ingest_payloads(urlopen_mock)
        assert payloads[0]["op"] == "start"
        assert payloads[0]["citation"] == "us/statute/26/32"
        assert payloads[0]["runner"]["hostname"] == runner_identity()["hostname"]
        assert payloads[1] == {
            "op": "heartbeat",
            "id": live.id,
            "attempt": 2,
            "model": "gpt-5.5-max",
        }
        assert payloads[2]["op"] == "finish"
        assert payloads[2]["status"] == "completed"
        assert payloads[2]["run_id"] == "abc12345"
        request = urlopen_mock.call_args_list[0].args[0]
        assert request.full_url.startswith("https://axiom.org/")

    def test_ingest_url_override(self, monkeypatch):
        _ingest_env(monkeypatch)
        monkeypatch.setenv(
            "AXIOM_ENCODE_TELEMETRY_INGEST_URL", "https://staging.example/ingest"
        )
        with _mock_urlopen() as urlopen_mock:
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.0.0",
            ):
                pass
        assert urlopen_mock.call_args_list[0].args[0].full_url == (
            "https://staging.example/ingest"
        )

    def test_ingest_start_failure_disables_telemetry(self, monkeypatch):
        _ingest_env(monkeypatch)
        with patch(
            "axiom_encode.live_run_telemetry.urllib.request.urlopen",
            side_effect=OSError("unreachable"),
        ) as urlopen_mock:
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.0.0",
            ) as live:
                live.finish("completed")
        # Only the failed start attempt — no heartbeat or finish posts after.
        assert urlopen_mock.call_count == 1

    def test_noop_under_pytest_even_with_credentials(self, monkeypatch):
        monkeypatch.setenv("AXIOM_ENCODE_SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("AXIOM_ENCODE_SUPABASE_SECRET_KEY", "secret")
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_x.py::test_y")
        with patch("axiom_encode.supabase_sync.get_supabase_client") as mock_get:
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.0.0",
            ) as live:
                assert live._client is None
        mock_get.assert_not_called()

    def test_noop_when_disabled(self, monkeypatch):
        _configured_env(monkeypatch)
        with patch("axiom_encode.supabase_sync.get_supabase_client") as mock_get:
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.0.0",
                enabled=False,
            ) as live:
                assert live._client is None
        mock_get.assert_not_called()

    def test_inserts_running_row_and_finishes_completed(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.1.0",
            ) as live:
                live.finish("completed", run_id="abc12345")

        inserted = table.insert.call_args[0][0]
        assert inserted["citation"] == "us/statute/26/32"
        assert inserted["status"] == "running"
        assert inserted["backend"] == "codex"
        assert inserted["model"] == "gpt-5.5"
        assert inserted["runner"]["hostname"] == runner_identity()["hostname"]
        assert inserted["id"].startswith("live-")

        finished = table.update.call_args[0][0]
        assert finished["status"] == "completed"
        assert finished["run_id"] == "abc12345"
        assert finished["finished_at"]

    def test_exit_without_finish_marks_failed(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            try:
                with LiveRunTelemetry(
                    citation="us/statute/26/32",
                    backend="codex",
                    model="gpt-5.5",
                    encoder_version="0.1.0",
                ):
                    raise KeyboardInterrupt
            except KeyboardInterrupt:
                pass

        finished = table.update.call_args[0][0]
        assert finished["status"] == "failed"
        assert "run_id" not in finished

    def test_finish_is_idempotent(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.1.0",
            ) as live:
                live.finish("failed")
        # __exit__ must not overwrite the explicit finish.
        assert table.update.call_count == 1
        assert table.update.call_args[0][0]["status"] == "failed"

    def test_set_attempt_updates_row(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.1.0",
            ) as live:
                live.set_attempt(2, "gpt-5.5-max")
                live.finish("completed")
        attempt_update = table.update.call_args_list[0][0][0]
        assert attempt_update["attempt"] == 2
        assert attempt_update["model"] == "gpt-5.5-max"
        assert attempt_update["last_heartbeat_at"]

    def test_insert_failure_disables_telemetry(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        table.insert.return_value.execute.side_effect = RuntimeError("supabase down")
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.1.0",
            ) as live:
                assert live._client is None
                live.finish("completed")
        table.update.assert_not_called()
