"""Tests for live encode-run presence telemetry."""

from unittest.mock import MagicMock, patch

from axiom_encode.live_run_telemetry import LiveRunTelemetry, runner_identity


def _mock_client():
    client = MagicMock()
    table = client.schema.return_value.table.return_value
    table.insert.return_value.execute.return_value = MagicMock()
    table.update.return_value.eq.return_value.execute.return_value = MagicMock()
    return client, table


def _configured_env(monkeypatch):
    monkeypatch.setenv("AXIOM_ENCODE_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("AXIOM_ENCODE_SUPABASE_SECRET_KEY", "secret")


class TestRunnerIdentity:
    def test_contains_machine_fields(self):
        identity = runner_identity()
        assert set(identity) == {"hostname", "username", "platform", "pid", "is_ci"}
        assert isinstance(identity["pid"], int)
        assert isinstance(identity["is_ci"], bool)


class TestLiveRunTelemetry:
    def test_noop_without_credentials(self, monkeypatch):
        monkeypatch.delenv("AXIOM_ENCODE_SUPABASE_URL", raising=False)
        monkeypatch.delenv("AXIOM_ENCODE_SUPABASE_SECRET_KEY", raising=False)
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
        assert attempt_update == {"attempt": 2, "model": "gpt-5.5-max"}

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
