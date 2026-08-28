"""
Tests for supabase_sync module.

All external dependencies (supabase, sqlite DBs, env vars) are mocked.
"""

import hashlib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from axiom_encode import __version__
from axiom_encode.supabase_sync import (
    ENCODINGS_SCHEMA,
    TELEMETRY_SCHEMA,
    fetch_runs_from_supabase,
    get_local_transcript_stats,
    get_supabase_client,
    sync_agent_sessions_to_supabase,
    sync_all_runs,
    sync_run_to_supabase,
    sync_transcripts_to_supabase,
)
from tests.eval_evidence_fixtures import (
    TEST_APPLY_PRIVATE_KEY_B64,
    TEST_APPLY_PUBLIC_KEY_B64,
)
from tests.signing_broker_fixtures import SigningBrokerFixture

TEST_APPLY_SIGNING_BROKER = SigningBrokerFixture(
    apply_private_key=TEST_APPLY_PRIVATE_KEY_B64,
    apply_public_key=TEST_APPLY_PUBLIC_KEY_B64,
)


@pytest.fixture(autouse=True)
def _apply_manifest_signing_key(monkeypatch):
    monkeypatch.setenv(
        "AXIOM_ENCODE_APPLY_SIGNING_PUBLIC_KEY",
        TEST_APPLY_PUBLIC_KEY_B64,
    )
    monkeypatch.setattr(
        "axiom_encode.signing_broker._active_broker",
        TEST_APPLY_SIGNING_BROKER,
    )
    monkeypatch.setattr("axiom_encode.signing_broker._active_broker_pid", None)


# =========================================================================
# get_supabase_client
# =========================================================================


class TestGetSupabaseClient:
    def test_missing_url(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing Supabase write credentials"):
                get_supabase_client()

    def test_missing_key(self):
        with patch.dict(
            os.environ,
            {"AXIOM_ENCODE_SUPABASE_URL": "https://example.supabase.co"},
            clear=True,
        ):
            with pytest.raises(ValueError, match="Missing Supabase write credentials"):
                get_supabase_client()

    def test_with_secret_key(self):
        with patch.dict(
            os.environ,
            {
                "AXIOM_ENCODE_SUPABASE_URL": "https://example.supabase.co",
                "AXIOM_ENCODE_SUPABASE_SECRET_KEY": "secret-key",
            },
            clear=True,
        ):
            with patch("axiom_encode.supabase_sync.create_client") as mock_create:
                mock_create.return_value = MagicMock()
                get_supabase_client()
                mock_create.assert_called_once_with(
                    "https://example.supabase.co", "secret-key"
                )

    def test_with_anon_key_fallback_for_reads(self):
        with patch.dict(
            os.environ,
            {
                "AXIOM_ENCODE_SUPABASE_URL": "https://example.supabase.co",
                "AXIOM_ENCODE_SUPABASE_ANON_KEY": "anon-key",
            },
            clear=True,
        ):
            with patch("axiom_encode.supabase_sync.create_client") as mock_create:
                mock_create.return_value = MagicMock()
                get_supabase_client(require_write=False)
                mock_create.assert_called_once_with(
                    "https://example.supabase.co", "anon-key"
                )

    def test_write_client_requires_secret_key(self):
        with patch.dict(
            os.environ,
            {
                "AXIOM_ENCODE_SUPABASE_URL": "https://example.supabase.co",
                "AXIOM_ENCODE_SUPABASE_ANON_KEY": "anon-key",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="Missing Supabase write credentials"):
                get_supabase_client()


# =========================================================================
# sync_run_to_supabase
# =========================================================================


class TestSyncRunToSupabase:
    def test_invalid_data_source(self):
        mock_run = MagicMock()
        with pytest.raises(ValueError, match="data_source must be one of"):
            sync_run_to_supabase(mock_run, "invalid_source")

    def test_valid_data_sources(self):
        for source in [
            "reviewer_agent",
            "ci_only",
            "mock",
            "manual_estimate",
            "apply_manifest",
        ]:
            mock_run = MagicMock()
            mock_run.id = "test-123"
            mock_run.timestamp = datetime.now()
            mock_run.citation = "26 USC 1"
            mock_run.file_path = "test.yaml"
            mock_run.source_text = "source text"
            mock_run.rulespec_content = ""
            mock_run.review_results = None

            mock_client = MagicMock()
            mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
                data=[{"id": "test-123"}]
            )

            result = sync_run_to_supabase(mock_run, source, client=mock_client)
            assert result is True

    def test_with_review_results(self):
        mock_run = MagicMock()
        mock_run.id = "test-123"
        mock_run.timestamp = datetime.now()
        mock_run.citation = "26 USC 1"
        mock_run.file_path = "test.yaml"
        mock_run.source_text = "source text"
        mock_run.rulespec_content = "format: rulespec/v1"
        mock_run.review_results = MagicMock(
            reviews=[
                MagicMock(
                    reviewer="rulespec_reviewer",
                    items_checked=10,
                    items_passed=8,
                    passed=True,
                ),
                MagicMock(
                    reviewer="formula_reviewer",
                    items_checked=10,
                    items_passed=7,
                    passed=True,
                ),
            ],
            policyengine_match=0.95,
        )

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": "test-123"}]
        )

        result = sync_run_to_supabase(mock_run, "reviewer_agent", client=mock_client)
        assert result is True
        mock_client.schema.assert_called_once_with(ENCODINGS_SCHEMA)
        mock_client.schema.return_value.table.assert_called_once_with("encoding_runs")
        upsert_payload = (
            mock_client.schema.return_value.table.return_value.upsert.call_args.args[0]
        )
        assert upsert_payload["rulespec_content"] == "format: rulespec/v1"
        assert upsert_payload["encoder_version"] is None
        assert upsert_payload["has_issues"] is False
        assert upsert_payload["note"] is None
        assert upsert_payload["scores"]["rulespec"] == 8.0
        assert upsert_payload["scores"]["formula"] == 7.0
        assert set(upsert_payload["scores"]) == {
            "policyengine_match",
            "rulespec",
            "formula",
        }

    def test_final_apply_outcome_overrides_raw_iteration_issue(self):
        mock_run = MagicMock()
        mock_run.id = "test-123"
        mock_run.timestamp = datetime.now()
        mock_run.citation = "NY SNAP telephone standard"
        mock_run.file_path = "policies/example.yaml"
        mock_run.source_text = "source text"
        mock_run.rulespec_content = "format: rulespec/v1"
        mock_run.review_results = None
        mock_run.outcome = {
            "standalone_validation_success": False,
            "apply_requested": True,
            "overlay_validation_success": True,
            "apply_success": True,
            "final_success": True,
            "status": "apply_applied",
        }
        error = MagicMock()
        error.error_type = "validation"
        error.message = "Generated RuleSpec failed compile validation."
        error.variable = None
        error.fix_applied = None
        iteration = MagicMock()
        iteration.attempt = 1
        iteration.duration_ms = 1000
        iteration.success = False
        iteration.errors = [error]
        mock_run.iterations = [iteration]

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": "test-123"}]
        )

        result = sync_run_to_supabase(mock_run, "reviewer_agent", client=mock_client)

        assert result is True
        upsert_payload = (
            mock_client.schema.return_value.table.return_value.upsert.call_args.args[0]
        )
        assert upsert_payload["outcome"]["final_success"] is True
        assert upsert_payload["has_issues"] is False
        assert (
            upsert_payload["note"]
            == "Standalone validation failed; overlay apply succeeded."
        )

    def test_final_apply_with_only_non_blocking_review_notes_has_no_issues(self):
        mock_run = MagicMock()
        mock_run.id = "test-123"
        mock_run.timestamp = datetime.now()
        mock_run.citation = "NY SNAP telephone standard"
        mock_run.file_path = "policies/example.yaml"
        mock_run.source_text = "source text"
        mock_run.rulespec_content = "format: rulespec/v1"
        mock_run.outcome = {
            "standalone_validation_success": True,
            "apply_requested": True,
            "overlay_validation_success": True,
            "apply_success": True,
            "final_success": True,
            "status": "apply_applied",
        }
        review = MagicMock()
        review.critical_issues = []
        review.important_issues = ["[non-blocking] add one more edge case"]
        review.passed = True
        mock_run.review_results = MagicMock(
            reviews=[review],
            policyengine_match=None,
        )
        mock_run.iterations = []

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": "test-123"}]
        )

        result = sync_run_to_supabase(mock_run, "reviewer_agent", client=mock_client)

        assert result is True
        upsert_payload = (
            mock_client.schema.return_value.table.return_value.upsert.call_args.args[0]
        )
        assert upsert_payload["has_issues"] is False
        assert "[non-blocking] add one more edge case" in upsert_payload["note"]

    def test_retries_without_outcome_when_remote_schema_rejects_it(self):
        mock_run = MagicMock()
        mock_run.id = "test-123"
        mock_run.timestamp = datetime.now()
        mock_run.citation = "26 USC 1"
        mock_run.file_path = "test.yaml"
        mock_run.source_text = "source text"
        mock_run.rulespec_content = ""
        mock_run.review_results = None
        mock_run.outcome = {"final_success": True, "status": "apply_applied"}

        mock_client = MagicMock()
        execute = mock_client.schema.return_value.table.return_value.upsert.return_value.execute
        execute.side_effect = [
            Exception("Could not find the 'outcome' column"),
            MagicMock(data=[{"id": "test-123"}]),
        ]

        result = sync_run_to_supabase(mock_run, "ci_only", client=mock_client)

        assert result is True
        upsert_calls = (
            mock_client.schema.return_value.table.return_value.upsert.call_args_list
        )
        assert "outcome" in upsert_calls[0].args[0]
        assert "outcome" not in upsert_calls[1].args[0]

    def test_upsert_failure(self, capsys):
        mock_run = MagicMock()
        mock_run.id = "test-123"
        mock_run.timestamp = datetime.now()
        mock_run.citation = "26 USC 1"
        mock_run.file_path = "test.yaml"
        mock_run.source_text = "source text"
        mock_run.rulespec_content = ""
        mock_run.review_results = None

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.side_effect = Exception(
            "Connection error"
        )

        result = sync_run_to_supabase(mock_run, "ci_only", client=mock_client)
        assert result is False

    def test_upsert_empty_result(self, capsys):
        mock_run = MagicMock()
        mock_run.id = "test-123"
        mock_run.timestamp = datetime.now()
        mock_run.citation = "26 USC 1"
        mock_run.file_path = "test.yaml"
        mock_run.source_text = "source text"
        mock_run.rulespec_content = ""
        mock_run.review_results = None

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[]
        )

        result = sync_run_to_supabase(mock_run, "ci_only", client=mock_client)
        assert result is False

    def test_creates_client_if_not_provided(self):
        mock_run = MagicMock()
        mock_run.id = "test-123"
        mock_run.timestamp = datetime.now()
        mock_run.citation = "26 USC 1"
        mock_run.file_path = "test.yaml"
        mock_run.source_text = "source text"
        mock_run.rulespec_content = ""
        mock_run.review_results = None

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": "test-123"}]
        )

        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=mock_client
        ) as mock_get_client:
            result = sync_run_to_supabase(mock_run, "ci_only")
            assert result is True
            mock_get_client.assert_called_once_with()


# =========================================================================
# sync_all_runs
# =========================================================================


class TestSyncAllRuns:
    def test_sync_all_runs(self, tmp_path):
        from axiom_encode.harness.encoding_db import (
            EncodingDB,
            EncodingRun,
            Iteration,
        )

        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        run = EncodingRun(
            citation="26 USC 1",
            file_path="test.yaml",
            iterations=[Iteration(attempt=1, duration_ms=1000, success=True)],
        )
        db.log_run(run)

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": run.id}]
        )

        stats = sync_all_runs(db_path, "ci_only", client=mock_client)
        assert stats["total"] >= 1
        assert stats["synced"] >= 1

    def test_sync_all_runs_with_failure(self, tmp_path):
        from axiom_encode.harness.encoding_db import (
            EncodingDB,
            EncodingRun,
            Iteration,
        )

        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        run = EncodingRun(
            citation="26 USC 1",
            file_path="test.yaml",
            iterations=[Iteration(attempt=1, duration_ms=1000, success=True)],
        )
        db.log_run(run)

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.side_effect = Exception(
            "Error"
        )

        stats = sync_all_runs(db_path, "ci_only", client=mock_client)
        assert stats["failed"] >= 1

    def test_creates_client_if_not_provided(self, tmp_path):
        from axiom_encode.harness.encoding_db import EncodingDB

        db_path = tmp_path / "test.db"
        EncodingDB(db_path)

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[]
        )

        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=mock_client
        ):
            stats = sync_all_runs(db_path, "ci_only")
            assert stats["total"] == 0


# =========================================================================
# fetch_runs_from_supabase
# =========================================================================


class TestFetchRunsFromSupabase:
    def test_fetch_without_citation(self):
        mock_client = MagicMock()
        mock_result = MagicMock(data=[{"id": "1"}, {"id": "2"}])
        mock_client.schema.return_value.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        results = fetch_runs_from_supabase(limit=20, client=mock_client)
        assert len(results) == 2
        mock_client.schema.assert_called_once_with(ENCODINGS_SCHEMA)
        mock_client.schema.return_value.table.assert_called_once_with("encoding_runs")

    def test_fetch_with_citation(self):
        mock_client = MagicMock()
        mock_result = MagicMock(data=[{"id": "1", "citation": "26 USC 1"}])
        mock_client.schema.return_value.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        results = fetch_runs_from_supabase(citation="26 USC 1", client=mock_client)
        assert len(results) == 1

    def test_creates_client_if_not_provided(self):
        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[]
        )

        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=mock_client
        ) as mock_get_client:
            results = fetch_runs_from_supabase()
            assert results == []
            mock_get_client.assert_called_once_with(require_write=False)


# =========================================================================
# sync_transcripts_to_supabase
# =========================================================================


class TestSyncTranscriptsToSupabase:
    def test_no_transcript_db(self):
        with patch.object(Path, "exists", return_value=False):
            result = sync_transcripts_to_supabase()
            assert result["total"] == 0
            assert "error" in result

    def test_sync_all_unsynced(self, tmp_path):
        # Create a temporary transcript DB
        db_path = tmp_path / "transcripts.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE agent_transcripts (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                tool_use_id TEXT,
                subagent_type TEXT,
                prompt TEXT,
                description TEXT,
                response_summary TEXT,
                transcript TEXT,
                message_count INTEGER,
                created_at TEXT,
                uploaded_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO agent_transcripts VALUES (1, 'sess-1', 'tu-1', 'encoder', 'prompt', 'desc', 'summary', '[]', 5, '2024-01-01', NULL)"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": 1}]
        )

        with patch("axiom_encode.supabase_sync.TRANSCRIPT_DB", db_path):
            result = sync_transcripts_to_supabase(client=mock_client)
            assert result["synced"] == 1
            assert result["total"] == 1
            mock_client.schema.assert_called_once_with(TELEMETRY_SCHEMA)
            mock_client.schema.return_value.table.assert_called_once_with(
                "agent_transcripts"
            )

    def test_sync_with_session_filter(self, tmp_path):
        db_path = tmp_path / "transcripts.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE agent_transcripts (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                tool_use_id TEXT,
                subagent_type TEXT,
                prompt TEXT,
                description TEXT,
                response_summary TEXT,
                transcript TEXT,
                message_count INTEGER,
                created_at TEXT,
                uploaded_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO agent_transcripts VALUES (1, 'sess-1', 'tu-1', 'encoder', 'prompt', 'desc', 'summary', NULL, 5, '2024-01-01', NULL)"
        )
        conn.execute(
            "INSERT INTO agent_transcripts VALUES (2, 'sess-2', 'tu-2', 'encoder', 'prompt', 'desc', 'summary', NULL, 3, '2024-01-01', NULL)"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": 1}]
        )

        with patch("axiom_encode.supabase_sync.TRANSCRIPT_DB", db_path):
            result = sync_transcripts_to_supabase(
                session_id="sess-1", client=mock_client
            )
            assert result["synced"] == 1
            assert result["total"] == 1

    def test_sync_failure_during_upsert(self, tmp_path):
        db_path = tmp_path / "transcripts.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE agent_transcripts (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                tool_use_id TEXT,
                subagent_type TEXT,
                prompt TEXT,
                description TEXT,
                response_summary TEXT,
                transcript TEXT,
                message_count INTEGER,
                created_at TEXT,
                uploaded_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO agent_transcripts VALUES (1, 'sess-1', 'tu-1', 'encoder', 'prompt', 'desc', 'summary', '[]', 5, '2024-01-01', NULL)"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.side_effect = Exception(
            "Connection error"
        )

        with patch("axiom_encode.supabase_sync.TRANSCRIPT_DB", db_path):
            result = sync_transcripts_to_supabase(client=mock_client)
            assert result["failed"] == 1

    def test_sync_empty_result(self, tmp_path):
        db_path = tmp_path / "transcripts.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE agent_transcripts (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                tool_use_id TEXT,
                subagent_type TEXT,
                prompt TEXT,
                description TEXT,
                response_summary TEXT,
                transcript TEXT,
                message_count INTEGER,
                created_at TEXT,
                uploaded_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO agent_transcripts VALUES (1, 'sess-1', 'tu-1', 'encoder', 'prompt', 'desc', 'summary', '[]', 5, '2024-01-01', NULL)"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[]
        )

        with patch("axiom_encode.supabase_sync.TRANSCRIPT_DB", db_path):
            result = sync_transcripts_to_supabase(client=mock_client)
            assert result["failed"] == 1

    def test_creates_client_if_not_provided(self, tmp_path):
        db_path = tmp_path / "transcripts.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE agent_transcripts (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                tool_use_id TEXT,
                subagent_type TEXT,
                prompt TEXT,
                description TEXT,
                response_summary TEXT,
                transcript TEXT,
                message_count INTEGER,
                created_at TEXT,
                uploaded_at TEXT
            )
        """)
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        with patch("axiom_encode.supabase_sync.TRANSCRIPT_DB", db_path):
            with patch(
                "axiom_encode.supabase_sync.get_supabase_client",
                return_value=mock_client,
            ):
                result = sync_transcripts_to_supabase()
                assert result["total"] == 0


# =========================================================================
# sync_agent_sessions_to_supabase
# =========================================================================


class TestSyncAgentSessionsToSupabase:
    def test_no_experiments_db(self):
        with patch(
            "axiom_encode.supabase_sync.ENCODINGS_DB", Path("/nonexistent/path")
        ):
            result = sync_agent_sessions_to_supabase()
            assert result["total"] == 0

    def test_sync_session(self, tmp_path):
        db_path = tmp_path / "encodings.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE session_events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                sequence INTEGER,
                timestamp TEXT,
                event_type TEXT,
                tool_name TEXT,
                content TEXT,
                metadata_json TEXT
            )
        """)
        conn.execute(
            "INSERT INTO sessions VALUES ('agent-test-1', '2024-01-01', '2024-01-01', 'opus', '/tmp', 2, 100, 50, 10, 0.01)"
        )
        conn.execute(
            "INSERT INTO session_events VALUES ('ev-1', 'agent-test-1', 1, '2024-01-01', 'tool_call', 'Read', 'content', '{}')"
        )
        conn.execute(
            "INSERT INTO session_events VALUES ('ev-2', 'agent-test-1', 2, '2024-01-01', 'tool_result', NULL, 'result', NULL)"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": "test"}]
        )

        with patch("axiom_encode.supabase_sync.ENCODINGS_DB", db_path):
            result = sync_agent_sessions_to_supabase(
                client=mock_client, include_all=True
            )
            assert result["synced"] == 1
            assert result["total"] == 1
            mock_client.schema.assert_any_call(TELEMETRY_SCHEMA)
            mock_client.schema.return_value.table.assert_any_call("sdk_sessions")
            mock_client.schema.return_value.table.assert_any_call("sdk_session_events")

    def test_sync_session_preserves_unknown_estimated_cost(self, tmp_path):
        db_path = tmp_path / "encodings.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL
            )
        """)
        conn.execute("""
            CREATE TABLE session_events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                sequence INTEGER,
                timestamp TEXT,
                event_type TEXT,
                tool_name TEXT,
                content TEXT,
                metadata_json TEXT
            )
        """)
        conn.execute(
            "INSERT INTO sessions VALUES "
            "('agent-test-unknown', '2024-01-01', '2024-01-01', "
            "'gpt-5.6-terra', '/tmp', 0, 272001, 1, 0, NULL)"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": "test"}]
        )

        with patch("axiom_encode.supabase_sync.ENCODINGS_DB", db_path):
            result = sync_agent_sessions_to_supabase(
                client=mock_client, include_all=True
            )

        assert result["synced"] == 1
        session_payload = (
            mock_client.schema.return_value.table.return_value.upsert.call_args_list[
                0
            ].args[0]
        )
        assert session_payload["estimated_cost_usd"] is None

    def test_sync_with_session_filter(self, tmp_path):
        db_path = tmp_path / "encodings.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE session_events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                sequence INTEGER,
                timestamp TEXT,
                event_type TEXT,
                tool_name TEXT,
                content TEXT,
                metadata_json TEXT
            )
        """)
        conn.execute(
            "INSERT INTO sessions VALUES ('agent-test-1', '2024-01-01', '2024-01-01', 'opus', '/tmp', 0, 0, 0, 0, 0)"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": "test"}]
        )

        with patch("axiom_encode.supabase_sync.ENCODINGS_DB", db_path):
            result = sync_agent_sessions_to_supabase(
                session_id="agent-test-1", client=mock_client
            )
            assert result["synced"] == 1

    def test_sync_failure(self, tmp_path):
        db_path = tmp_path / "encodings.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE session_events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                sequence INTEGER,
                timestamp TEXT,
                event_type TEXT,
                tool_name TEXT,
                content TEXT,
                metadata_json TEXT
            )
        """)
        conn.execute(
            "INSERT INTO sessions VALUES ('agent-test-1', '2024-01-01', '2024-01-01', 'opus', '/tmp', 0, 0, 0, 0, 0)"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.side_effect = Exception(
            "Error"
        )

        with patch("axiom_encode.supabase_sync.ENCODINGS_DB", db_path):
            result = sync_agent_sessions_to_supabase(
                client=mock_client, include_all=True
            )
            assert result["failed"] == 1

    def test_sync_default_filters_to_axiom_encode_sessions(self, tmp_path):
        db_path = tmp_path / "encodings.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0,
                axiom_encode_version TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE session_events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                sequence INTEGER,
                timestamp TEXT,
                event_type TEXT,
                tool_name TEXT,
                content TEXT,
                metadata_json TEXT
            )
        """)
        conn.execute(
            "INSERT INTO sessions VALUES ('general-1', '', '2024-01-01', '2024-01-01', 'opus', '/Users/maxghenis', 0, 0, 0, 0, 0, '')"
        )
        conn.execute(
            "INSERT INTO sessions VALUES ('axiom-1', '', '2024-01-02', '2024-01-02', 'opus', '/Users/maxghenis/TheAxiomFoundation/axiom-encode', 0, 0, 0, 0, 0, '0.4.2')"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": "test"}]
        )

        with patch("axiom_encode.supabase_sync.ENCODINGS_DB", db_path):
            result = sync_agent_sessions_to_supabase(client=mock_client)

        assert result["total"] == 1
        session_payload = (
            mock_client.schema.return_value.table.return_value.upsert.call_args_list[
                0
            ].args[0]
        )
        assert session_payload["id"] == "axiom-1"
        assert session_payload["encoder_version"] == "0.4.2"

    def test_creates_client_if_not_provided(self, tmp_path):
        db_path = tmp_path / "encodings.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE session_events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                sequence INTEGER,
                timestamp TEXT,
                event_type TEXT,
                tool_name TEXT,
                content TEXT,
                metadata_json TEXT
            )
        """)
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        with patch("axiom_encode.supabase_sync.ENCODINGS_DB", db_path):
            with patch(
                "axiom_encode.supabase_sync.get_supabase_client",
                return_value=mock_client,
            ):
                result = sync_agent_sessions_to_supabase()
                assert result["total"] == 0


# =========================================================================
# get_local_transcript_stats
# =========================================================================


class TestGetLocalTranscriptStats:
    def test_no_db(self):
        with patch(
            "axiom_encode.supabase_sync.TRANSCRIPT_DB", Path("/nonexistent/path")
        ):
            result = get_local_transcript_stats()
            assert result["exists"] is False

    def test_with_data(self, tmp_path):
        db_path = tmp_path / "transcripts.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE agent_transcripts (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                tool_use_id TEXT,
                subagent_type TEXT,
                prompt TEXT,
                description TEXT,
                response_summary TEXT,
                transcript TEXT,
                message_count INTEGER,
                created_at TEXT,
                uploaded_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO agent_transcripts VALUES (1, 'sess-1', 'tu-1', 'encoder', 'p', 'd', 's', '[]', 5, '2024-01-01', NULL)"
        )
        conn.execute(
            "INSERT INTO agent_transcripts VALUES (2, 'sess-1', 'tu-2', 'reviewer', 'p', 'd', 's', '[]', 3, '2024-01-01', '2024-01-02')"
        )
        conn.commit()
        conn.close()

        with patch("axiom_encode.supabase_sync.TRANSCRIPT_DB", db_path):
            result = get_local_transcript_stats()
            assert result["exists"] is True
            assert result["total"] == 2
            assert result["unsynced"] == 1
            assert result["synced"] == 1
            assert "encoder" in result["by_type"]
            assert "reviewer" in result["by_type"]


# =========================================================================
# apply-manifest backfill
# =========================================================================


def _apply_manifest_payload(**overrides) -> dict:
    payload = {
        "schema_version": "axiom-encode/applied-rulespec/v5",
        "generated_at": "2026-05-26T23:53:33.747249+00:00",
        "tool": "axiom-encode encode --apply",
        "axiom_encode_version": "0.2.301",
        "axiom_encode_git": {
            "root": "/repo/axiom-encode",
            "commit": "a" * 40,
            "dirty_tracked": False,
            "version": "0.2.301",
            "version_commit": "b" * 40,
            "identity_source": "git",
        },
        "generation_prompt_sha256": None,
        "run_id": "5c8705fb",
        "citation": "26 USC 3127",
        "runner": "openai",
        "backend": "openai",
        "model": "gpt-5.5",
        "validation_waiver_set_sha256": "a" * 64,
        "generated_output_root": "/tmp/axiom-generated",
        "generated_output_file": None,
        "generated_output_sha256": None,
        "trace_file": None,
        "trace_sha256": None,
        "context_manifest_file": None,
        "context_manifest_sha256": None,
        "applied_files": [
            {"path": "statutes/26/3127.yaml", "sha256": "abc"},
            {"path": "statutes/26/3127.test.yaml", "sha256": "def"},
        ],
        "source_attestation": {},
        "validation_execution": {},
        "signature": {},
    }
    payload.update(overrides)
    return payload


class TestRunFromApplyManifest:
    def test_rejects_pre_v3_manifest(self):
        from axiom_encode.supabase_sync import run_from_apply_manifest

        with pytest.raises(ValueError, match="not canonical v3"):
            run_from_apply_manifest(
                Path("a.json"),
                _apply_manifest_payload(
                    schema_version="axiom-encode/applied-rulespec/v2"
                ),
            )

    def test_reconstructs_run_from_manifest(self):
        from axiom_encode.supabase_sync import run_from_apply_manifest

        run = run_from_apply_manifest(
            Path("repo/.axiom/encoding-manifests/statutes/26/3127.json"),
            _apply_manifest_payload(),
        )

        assert run.id == "5c8705fb"
        assert run.citation == "26 USC 3127"
        assert run.timestamp == datetime.fromisoformat(
            "2026-05-26T23:53:33.747249+00:00"
        )
        assert run.file_path == "statutes/26/3127.yaml"
        assert run.agent_type == "openai:encoder"
        assert run.agent_model == "gpt-5.5"
        assert run.axiom_encode_version == "0.2.301"
        assert run.session_id is None
        assert run.success is True

    def test_derives_deterministic_id_when_run_id_missing(self):
        from axiom_encode.supabase_sync import run_from_apply_manifest

        payload = _apply_manifest_payload(run_id=None)
        run_a = run_from_apply_manifest(Path("a.json"), dict(payload))
        run_b = run_from_apply_manifest(Path("a.json"), dict(payload))
        assert run_a.id == run_b.id
        assert len(run_a.id) == 8

    def test_rejects_manifest_without_citation(self):
        from axiom_encode.supabase_sync import run_from_apply_manifest

        with pytest.raises(ValueError, match="no citation"):
            run_from_apply_manifest(
                Path("a.json"), _apply_manifest_payload(citation="")
            )

    def test_rejects_manifest_without_generated_at(self):
        from axiom_encode.supabase_sync import run_from_apply_manifest

        with pytest.raises(ValueError, match="no generated_at"):
            run_from_apply_manifest(
                Path("a.json"), _apply_manifest_payload(generated_at=None)
            )


class TestSyncAppliedManifestRuns:
    def _write_repo(self, tmp_path: Path) -> Path:
        repo = tmp_path / "rulespec-us"
        waiver = repo / "known-validation-gaps.yaml"
        waiver.parent.mkdir(parents=True)
        waiver.write_text("validate_failures: {}\n")
        waiver_sha256 = hashlib.sha256(waiver.read_bytes()).hexdigest()
        toolchain = repo / ".axiom/toolchain.toml"
        toolchain.parent.mkdir(parents=True)
        toolchain.write_text(
            "[toolchain]\n"
            'axiom_corpus_release = "supabase-sync-test"\n'
            f'axiom_corpus_release_content_sha256 = "{"e" * 64}"\n'
            f'validation_waiver_set_sha256 = "{waiver_sha256}"\n'
        )
        manifest_dir = repo / ".axiom" / "encoding-manifests" / "us" / "statutes" / "26"
        manifest_dir.mkdir(parents=True)

        def write_manifest(
            section: str,
            *,
            run_id: str,
            citation: str,
            generated_at: str,
        ) -> None:
            corpus_path = f"us/statute/26/{section}"
            rule_rel = f"us/statutes/26/{section}.yaml"
            rule = repo / rule_rel
            rule.parent.mkdir(parents=True, exist_ok=True)
            source_sha256 = "a" * 64
            rule.write_text(
                "format: rulespec/v1\n"
                "module:\n"
                "  source_verification:\n"
                f"    corpus_citation_path: {corpus_path}\n"
                f"    source_sha256: {source_sha256}\n"
                "rules: []\n"
            )
            provision_file = "data/corpus/provisions/us/statute/test.jsonl"
            source_attestation = {
                "requested_corpus_citation_path": corpus_path,
                "resolved_corpus_citation_path": corpus_path,
                "corpus_source": "local",
                "corpus_release": "supabase-sync-test",
                "corpus_release_content_sha256": "e" * 64,
                "corpus_release_selector_sha256": "b" * 64,
                "provision_file": provision_file,
                "provision_file_sha256": "c" * 64,
                "row": {
                    "provision_file": provision_file,
                    "provision_file_sha256": "c" * 64,
                    "line_number": 1,
                    "record_id": f"test:{section}",
                    "citation_path": corpus_path,
                    "jurisdiction": "us",
                    "document_class": "statute",
                    "version": "test",
                    "source_path": f"sources/us/statute/{section}",
                    "source_as_of": "2026-01-01",
                    "expression_date": "2026-01-01",
                    "body_sha256": source_sha256,
                },
                "component_rows": [],
                "source_sha256": source_sha256,
                "resolved_text_sha256": source_sha256,
                "generation_input_sha256": "d" * 64,
                "rulespec_root": "rulespec-us/us",
                "source_as_of": "2026-01-01",
                "expression_date": "2026-01-01",
            }
            payload = _apply_manifest_payload(
                run_id=run_id,
                citation=citation,
                generated_at=generated_at,
                axiom_encode_version=__version__,
                axiom_encode_git={
                    "root": "/repo/axiom-encode",
                    "commit": "a" * 40,
                    "dirty_tracked": False,
                    "version": __version__,
                    "version_commit": "b" * 40,
                    "identity_source": "git",
                },
                validation_waiver_set_sha256=waiver_sha256,
                source_attestation=source_attestation,
                validation_execution={
                    "schema": "axiom-encode/apply-validation-execution/v1",
                    "axiom_encode": {
                        "repository": ("github.com/TheAxiomFoundation/axiom-encode"),
                        "commit": "a" * 40,
                        "version": __version__,
                        "identity_source": "git",
                    },
                    "axiom_rules_engine": {
                        "repository": (
                            "github.com/TheAxiomFoundation/axiom-rules-engine"
                        ),
                        "commit": "e" * 40,
                    },
                    "policy_pre_apply": {
                        "rulespec_root": "rulespec-us/us",
                        "pre_apply_content_sha256": "f" * 64,
                        "pre_apply_file_count": 1,
                        "toolchain_contract_sha256": "d" * 64,
                        "validation_waiver_set_sha256": waiver_sha256,
                    },
                    "rulespec_dependencies": [],
                },
                applied_files=[
                    {
                        "path": rule_rel,
                        "sha256": hashlib.sha256(rule.read_bytes()).hexdigest(),
                    }
                ],
            )
            from axiom_encode.cli import _sign_applied_encoding_manifest

            _sign_applied_encoding_manifest(payload, TEST_APPLY_SIGNING_BROKER)
            (manifest_dir / f"{section}.json").write_text(json.dumps(payload))

        write_manifest(
            "3127",
            run_id="5c8705fb",
            citation="26 USC 3127",
            generated_at="2026-05-26T23:53:33.747249+00:00",
        )
        write_manifest(
            "3111",
            run_id="aa11bb22",
            citation="26 USC 3111",
            generated_at="2026-06-01T10:00:00+00:00",
        )
        return repo

    def test_dry_run_counts_without_client(self, tmp_path, capsys):
        from axiom_encode.supabase_sync import sync_applied_manifest_runs

        repo = self._write_repo(tmp_path)
        stats = sync_applied_manifest_runs([repo], dry_run=True)

        assert stats == {
            "total": 2,
            "synced": 0,
            "failed": 0,
            "skipped": 0,
            "preserved": 0,
        }
        output = capsys.readouterr().out
        assert "5c8705fb" in output
        assert "aa11bb22" in output

    def test_scan_reuses_one_receipt_proof_cache(self, tmp_path, monkeypatch):
        import axiom_encode.cli as cli_module
        from axiom_encode.supabase_sync import sync_applied_manifest_runs

        repo = self._write_repo(tmp_path)
        original = cli_module._load_verified_applied_encoding_manifest_payload
        proof_calls = 0

        def counted_loader(*args, **kwargs):
            nonlocal proof_calls
            cache = kwargs["path_migration_receipt_proof_cache"]
            proof_key = (str(repo.resolve()), "shared-receipt")
            if proof_key not in cache:
                proof_calls += 1
                cache[proof_key] = ()
            return original(*args, **kwargs)

        monkeypatch.setattr(
            cli_module,
            "_load_verified_applied_encoding_manifest_payload",
            counted_loader,
        )

        stats = sync_applied_manifest_runs([repo], dry_run=True)

        assert stats["total"] == 2
        assert stats["failed"] == 0
        assert proof_calls == 1

    def test_syncs_manifest_runs_with_apply_manifest_source(self, tmp_path):
        from axiom_encode.supabase_sync import sync_applied_manifest_runs

        repo = self._write_repo(tmp_path)
        mock_client = MagicMock()
        table = mock_client.schema.return_value.table.return_value
        table.upsert.return_value.execute.return_value = MagicMock(data=[{"id": "x"}])
        table.select.return_value.neq.return_value.range.return_value.execute.return_value = MagicMock(
            data=[]
        )

        stats = sync_applied_manifest_runs([repo], client=mock_client)

        assert stats == {
            "total": 2,
            "synced": 2,
            "failed": 0,
            "skipped": 0,
            "preserved": 0,
        }
        upsert_calls = (
            mock_client.schema.return_value.table.return_value.upsert.call_args_list
        )
        payloads = [call.args[0] for call in upsert_calls]
        assert {payload["id"] for payload in payloads} == {"5c8705fb", "aa11bb22"}
        assert all(payload["data_source"] == "apply_manifest" for payload in payloads)
        assert all(payload["has_issues"] is False for payload in payloads)

    def test_fails_unreadable_manifest(self, tmp_path, capsys):
        from axiom_encode.supabase_sync import sync_applied_manifest_runs

        repo = self._write_repo(tmp_path)
        bad = repo / ".axiom" / "encoding-manifests" / "bad.json"
        bad.write_text("{not json")

        stats = sync_applied_manifest_runs([repo], dry_run=True)

        assert stats["total"] == 3
        assert stats["failed"] == 1
        assert stats["skipped"] == 0
        assert "Invalid apply manifest" in capsys.readouterr().out

    def test_preserves_rows_with_richer_data_sources(self, tmp_path):
        from axiom_encode.supabase_sync import sync_applied_manifest_runs

        repo = self._write_repo(tmp_path)
        mock_client = MagicMock()
        table = mock_client.schema.return_value.table.return_value
        table.upsert.return_value.execute.return_value = MagicMock(data=[{"id": "x"}])
        # 5c8705fb already exists as a reviewer_agent run - must not be touched.
        table.select.return_value.neq.return_value.range.return_value.execute.return_value = MagicMock(
            data=[{"id": "5c8705fb"}]
        )

        stats = sync_applied_manifest_runs([repo], client=mock_client)

        assert stats == {
            "total": 2,
            "synced": 1,
            "failed": 0,
            "skipped": 0,
            "preserved": 1,
        }
        upsert_ids = {call.args[0]["id"] for call in table.upsert.call_args_list}
        assert upsert_ids == {"aa11bb22"}

    def test_missing_manifest_dir_is_empty(self, tmp_path):
        from axiom_encode.supabase_sync import find_apply_manifests

        assert find_apply_manifests(tmp_path) == []

    def test_noncanonical_run_ids_get_unique_derived_ids(self):
        from axiom_encode.supabase_sync import run_from_apply_manifest

        run_a = run_from_apply_manifest(
            Path("repo/.axiom/encoding-manifests/regulations/14/a/1.json"),
            _apply_manifest_payload(run_id="shared-run"),
        )
        run_b = run_from_apply_manifest(
            Path("repo/.axiom/encoding-manifests/regulations/14/a/5.json"),
            _apply_manifest_payload(run_id="shared-run"),
        )

        assert run_a.id != "shared-run"
        assert run_a.id != run_b.id
        assert len(run_a.id) == 8
        # Same manifest re-synced later keeps the same id.
        run_a_again = run_from_apply_manifest(
            Path(
                "/other/host/checkout/.axiom/encoding-manifests/regulations/14/a/1.json"
            ),
            _apply_manifest_payload(run_id="shared-run"),
        )
        assert run_a_again.id == run_a.id


class TestSyncRunCostLedger:
    """Per-run token/cost fields in the encodings.encoding_runs payload."""

    def _make_run_with_ledger(self):
        from axiom_encode.harness.encoding_db import TokenUsage

        mock_run = MagicMock()
        mock_run.id = "cost-123"
        mock_run.timestamp = datetime.now()
        mock_run.citation = "26 USC 1"
        mock_run.file_path = "test.yaml"
        mock_run.source_text = "source text"
        mock_run.rulespec_content = ""
        mock_run.review_results = None
        mock_run.iterations = []
        mock_run.tokens = TokenUsage(
            input_tokens=76_000,
            output_tokens=3_700,
            cache_read_tokens=1_200,
            cache_creation_tokens=800,
            reasoning_output_tokens=400,
        )
        mock_run.estimated_cost_usd = 0.3322
        mock_run.actual_cost_usd = None
        mock_run.generation_attempt_count = 2
        return mock_run

    def test_payload_includes_token_ledger(self):
        mock_run = self._make_run_with_ledger()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": "cost-123"}]
        )

        assert sync_run_to_supabase(mock_run, "ci_only", client=mock_client) is True
        payload = (
            mock_client.schema.return_value.table.return_value.upsert.call_args.args[0]
        )
        assert payload["input_tokens"] == 76_000
        assert payload["output_tokens"] == 3_700
        assert payload["cache_read_tokens"] == 1_200
        assert payload["cache_creation_tokens"] == 800
        assert payload["reasoning_output_tokens"] == 400
        assert payload["estimated_cost_usd"] == 0.3322
        assert payload["actual_cost_usd"] is None
        assert payload["generation_attempt_count"] == 2

    def test_retries_without_cost_columns_when_remote_schema_rejects_them(self):
        mock_run = self._make_run_with_ledger()
        mock_run.outcome = {"final_success": True, "status": "apply_applied"}

        mock_client = MagicMock()
        execute = (
            mock_client.schema.return_value.table.return_value.upsert.return_value.execute
        )
        execute.side_effect = [
            Exception("Could not find the 'input_tokens' column"),
            MagicMock(data=[{"id": "cost-123"}]),
        ]

        assert sync_run_to_supabase(mock_run, "ci_only", client=mock_client) is True
        upsert_calls = (
            mock_client.schema.return_value.table.return_value.upsert.call_args_list
        )
        assert "input_tokens" in upsert_calls[0].args[0]
        assert "input_tokens" not in upsert_calls[1].args[0]
        assert "estimated_cost_usd" not in upsert_calls[1].args[0]
        # The outcome column is only stripped by the next fallback tier.
        assert "outcome" in upsert_calls[1].args[0]

    def test_retries_without_cost_columns_then_outcome(self):
        mock_run = self._make_run_with_ledger()
        mock_run.outcome = {"final_success": True, "status": "apply_applied"}

        mock_client = MagicMock()
        execute = (
            mock_client.schema.return_value.table.return_value.upsert.return_value.execute
        )
        execute.side_effect = [
            Exception("Could not find the 'input_tokens' column"),
            Exception("Could not find the 'outcome' column"),
            MagicMock(data=[{"id": "cost-123"}]),
        ]

        assert sync_run_to_supabase(mock_run, "ci_only", client=mock_client) is True
        upsert_calls = (
            mock_client.schema.return_value.table.return_value.upsert.call_args_list
        )
        assert "input_tokens" not in upsert_calls[2].args[0]
        assert "outcome" not in upsert_calls[2].args[0]


class TestSyncSessionTokenColumns:
    """cache_creation/reasoning token columns in the sdk_sessions payload."""

    def _make_session_db(self, tmp_path):
        db_path = tmp_path / "encodings.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_creation_tokens INTEGER DEFAULT 0,
                reasoning_output_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE session_events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                sequence INTEGER,
                timestamp TEXT,
                event_type TEXT,
                tool_name TEXT,
                content TEXT,
                metadata_json TEXT
            )
        """)
        conn.execute(
            "INSERT INTO sessions VALUES "
            "('agent-test-1', '2024-01-01', '2024-01-01', 'opus', '/tmp', 0, "
            "100, 50, 10, 20, 5, 0.01)"
        )
        conn.commit()
        conn.close()
        return db_path

    def test_payload_includes_new_token_columns(self, tmp_path):
        db_path = self._make_session_db(tmp_path)

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": "agent-test-1"}]
        )

        with patch("axiom_encode.supabase_sync.ENCODINGS_DB", db_path):
            result = sync_agent_sessions_to_supabase(
                client=mock_client, include_all=True
            )
        assert result["synced"] == 1
        payload = (
            mock_client.schema.return_value.table.return_value.upsert.call_args.args[0]
        )
        assert payload["cache_creation_tokens"] == 20
        assert payload["reasoning_output_tokens"] == 5

    def test_retries_without_new_token_columns(self, tmp_path):
        db_path = self._make_session_db(tmp_path)

        mock_client = MagicMock()
        execute = (
            mock_client.schema.return_value.table.return_value.upsert.return_value.execute
        )
        execute.side_effect = [
            Exception("Could not find the 'cache_creation_tokens' column"),
            MagicMock(data=[{"id": "agent-test-1"}]),
        ]

        with patch("axiom_encode.supabase_sync.ENCODINGS_DB", db_path):
            result = sync_agent_sessions_to_supabase(
                client=mock_client, include_all=True
            )
        assert result["synced"] == 1
        assert result["failed"] == 0
        upsert_calls = (
            mock_client.schema.return_value.table.return_value.upsert.call_args_list
        )
        assert "cache_creation_tokens" in upsert_calls[0].args[0]
        assert "cache_creation_tokens" not in upsert_calls[1].args[0]
        assert "reasoning_output_tokens" not in upsert_calls[1].args[0]
