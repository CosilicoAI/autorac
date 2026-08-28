"""
Tests for the experiment database.
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from axiom_encode import (
    EncodingDB,
    Iteration,
    ReviewResult,
    ReviewResults,
    create_run,
)


class TestCreateRun:
    """Tests for the create_run factory function."""

    def test_create_run_generates_id(self):
        """Test that create_run generates a unique ID."""
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
        )
        assert run.id is not None
        assert len(run.id) == 8  # UUID[:8]

    def test_create_run_sets_timestamp(self):
        """Test that create_run sets current timestamp."""
        before = datetime.now()
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
        )
        after = datetime.now()
        assert before <= run.timestamp <= after

    def test_create_run_sets_iteration_1_for_new_run(self):
        """Test that new runs have iteration=1."""
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
        )
        assert run.iteration == 1
        assert run.parent_run_id is None

    def test_create_run_sets_iteration_2_for_revision(self):
        """Test that revisions have iteration=2."""
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
            parent_run_id="abc12345",
        )
        assert run.iteration == 2
        assert run.parent_run_id == "abc12345"


class TestEncodingDBInit:
    """Tests for EncodingDB initialization."""

    def test_creates_database_file(self, temp_db_path):
        """Test that database file is created."""
        EncodingDB(temp_db_path)
        assert temp_db_path.exists()

    def test_creates_tables(self, experiment_db, temp_db_path):
        """Test that required tables are created."""
        import sqlite3

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Check encoding_runs table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='encoding_runs'"
        )
        assert cursor.fetchone() is not None

        # Check calibration_snapshots table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='calibration_snapshots'"
        )
        assert cursor.fetchone() is not None

        conn.close()


class TestLogAndRetrieveRuns:
    """Tests for logging and retrieving encoding runs."""

    def test_log_run_and_retrieve(self, experiment_db, sample_encoding_run):
        """Test logging a run and retrieving it."""
        run_id = experiment_db.log_run(sample_encoding_run)

        retrieved = experiment_db.get_run(run_id)

        assert retrieved is not None
        assert retrieved.id == sample_encoding_run.id
        assert retrieved.file_path == sample_encoding_run.file_path
        assert retrieved.citation == sample_encoding_run.citation
        assert retrieved.agent_type == sample_encoding_run.agent_type
        assert retrieved.agent_model == sample_encoding_run.agent_model
        assert retrieved.rulespec_content == sample_encoding_run.rulespec_content

    def test_log_run_with_review_results(self, experiment_db, sample_review_results):
        """Test logging a run with review results."""
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
            review_results=sample_review_results,
        )

        experiment_db.log_run(run)
        retrieved = experiment_db.get_run(run.id)

        assert retrieved.review_results is not None
        assert len(retrieved.review_results.reviews) == 4
        assert retrieved.review_results.reviews[0].reviewer == "rulespec_reviewer"
        assert retrieved.review_results.reviews[0].passed is True
        assert retrieved.review_results.policyengine_match == 0.90

    def test_log_run_with_lessons(self, experiment_db):
        """Test logging a run with lessons."""
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
            lessons="Learned that bracket syntax needs special handling.",
        )

        experiment_db.log_run(run)
        retrieved = experiment_db.get_run(run.id)

        assert (
            retrieved.lessons == "Learned that bracket syntax needs special handling."
        )

    def test_log_run_with_final_outcome(self, experiment_db, sample_encoding_run):
        """Test final encode/apply outcomes are persisted and drive run success."""
        sample_encoding_run.iterations = [
            Iteration(attempt=1, duration_ms=1000, success=False)
        ]
        sample_encoding_run.outcome = {
            "standalone_validation_success": False,
            "apply_requested": True,
            "overlay_validation_success": True,
            "apply_success": True,
            "final_success": True,
            "status": "apply_applied",
        }

        experiment_db.log_run(sample_encoding_run)
        retrieved = experiment_db.get_run(sample_encoding_run.id)

        assert retrieved.outcome["status"] == "apply_applied"
        assert retrieved.iterations[0].success is False
        assert retrieved.success is True

    def test_log_run_with_review_issues(self, experiment_db):
        """Test logging a run with review issues at different severity levels."""
        review_results = ReviewResults(
            reviews=[
                ReviewResult(
                    reviewer="rulespec_reviewer",
                    passed=False,
                    items_checked=5,
                    items_passed=3,
                    critical_issues=["Missing entity declaration"],
                    important_issues=["Citation format incorrect"],
                    minor_issues=["Style: prefer lowercase"],
                ),
            ],
        )

        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
            review_results=review_results,
        )

        experiment_db.log_run(run)
        retrieved = experiment_db.get_run(run.id)

        review = retrieved.review_results.reviews[0]
        assert review.passed is False
        assert review.critical_issues == ["Missing entity declaration"]
        assert review.important_issues == ["Citation format incorrect"]
        assert review.minor_issues == ["Style: prefer lowercase"]

    def test_get_nonexistent_run_returns_none(self, experiment_db):
        """Test that getting a nonexistent run returns None."""
        result = experiment_db.get_run("nonexistent-id")
        assert result is None


class TestUpdateReviewResults:
    """Tests for updating review results after validation."""

    def test_update_review_results(self, experiment_db, sample_review_results):
        """Test updating a run with review results."""
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
        )
        experiment_db.log_run(run)

        # Initially no review results
        retrieved = experiment_db.get_run(run.id)
        assert retrieved.review_results is None

        # Update with review results
        experiment_db.update_review_results(run.id, sample_review_results)

        # Now has review results
        retrieved = experiment_db.get_run(run.id)
        assert retrieved.review_results is not None
        assert len(retrieved.review_results.reviews) == 4
        assert retrieved.review_results.reviews[0].passed is True
        assert retrieved.review_results.policyengine_match == 0.90


class TestListRunsWithFilters:
    """Tests for listing runs with various filters."""

    def test_get_runs_for_citation(self, experiment_db):
        """Test getting all runs for a specific citation."""
        # Create runs for different citations
        for i in range(3):
            run = create_run(
                file_path=f"/path/to/file{i}.yaml",
                citation="26 USC 32",
                agent_type="axiom_encode:encoder",
                agent_model="claude-opus-4-6",
                rulespec_content=f"# content {i}",
            )
            experiment_db.log_run(run)

        run_other = create_run(
            file_path="/path/to/other.yaml",
            citation="26 USC 24",  # Different citation
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# other content",
        )
        experiment_db.log_run(run_other)

        # Get runs for 26 USC 32
        runs = experiment_db.get_runs_for_citation("26 USC 32")
        assert len(runs) == 3
        assert all(r.citation == "26 USC 32" for r in runs)

        # Get runs for 26 USC 24
        runs = experiment_db.get_runs_for_citation("26 USC 24")
        assert len(runs) == 1
        assert runs[0].citation == "26 USC 24"

    def test_get_recent_runs(self, experiment_db):
        """Test getting most recent runs with limit."""
        # Create 5 runs
        for i in range(5):
            run = create_run(
                file_path=f"/path/to/file{i}.yaml",
                citation=f"26 USC {i}",
                agent_type="axiom_encode:encoder",
                agent_model="claude-opus-4-6",
                rulespec_content=f"# content {i}",
            )
            experiment_db.log_run(run)

        # Get last 3
        runs = experiment_db.get_recent_runs(limit=3)
        assert len(runs) == 3

        # Most recent should be first (DESC order)
        # Since timestamps are very close, just verify we get 3 runs

    def test_get_recent_runs_default_limit(self, experiment_db):
        """Test getting recent runs with default limit."""
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
        )
        experiment_db.log_run(run)

        runs = experiment_db.get_recent_runs()
        assert len(runs) == 1


class TestReviewResultsProperties:
    """Tests for ReviewResults properties."""

    def test_passed_all_pass(self):
        """Test passed property when all reviews pass."""
        rr = ReviewResults(
            reviews=[
                ReviewResult(reviewer="rulespec_reviewer", passed=True),
                ReviewResult(reviewer="formula_reviewer", passed=True),
            ]
        )
        assert rr.passed is True

    def test_passed_one_fails(self):
        """Test passed property when one review fails."""
        rr = ReviewResults(
            reviews=[
                ReviewResult(reviewer="rulespec_reviewer", passed=True),
                ReviewResult(reviewer="formula_reviewer", passed=False),
            ]
        )
        assert rr.passed is False

    def test_passed_empty_reviews(self):
        """Test passed property with no reviews."""
        rr = ReviewResults(reviews=[])
        assert rr.passed is False

    def test_total_critical_issues(self):
        """Test total_critical_issues counts across all reviews."""
        rr = ReviewResults(
            reviews=[
                ReviewResult(
                    reviewer="rulespec",
                    critical_issues=["issue1", "issue2"],
                ),
                ReviewResult(
                    reviewer="formula",
                    critical_issues=["issue3"],
                ),
            ]
        )
        assert rr.total_critical_issues == 3


class TestSessionLogging:
    """Tests for session logging (used by SDK orchestrator)."""

    def test_start_session_generates_id(self, experiment_db):
        """Test that start_session generates a unique ID."""
        session = experiment_db.start_session(
            model="test-model", cwd="/tmp", axiom_encode_version="0.2.1"
        )
        assert session.id is not None
        assert len(session.id) == 8
        assert session.axiom_encode_version == "0.2.1"

    def test_start_session_with_custom_id(self, experiment_db):
        """Test that start_session accepts custom session_id."""
        session = experiment_db.start_session(
            model="test-model", cwd="/tmp", session_id="custom-123"
        )
        assert session.id == "custom-123"

    def test_start_session_links_run_id(self, experiment_db, sample_encoding_run):
        """Test that start_session can link SDK telemetry to an encoding run."""
        experiment_db.log_run(sample_encoding_run)
        session = experiment_db.start_session(
            model="test-model",
            cwd="/tmp",
            session_id="linked-session",
            run_id=sample_encoding_run.id,
        )

        assert session.run_id == sample_encoding_run.id
        retrieved = experiment_db.get_session("linked-session")
        assert retrieved is not None
        assert retrieved.run_id == sample_encoding_run.id

    def test_get_session_retrieves_by_id(self, experiment_db):
        """Test that get_session retrieves session by ID."""
        experiment_db.start_session(
            model="opus-4.5", cwd="/workspace", session_id="retrieve-test"
        )

        retrieved = experiment_db.get_session("retrieve-test")
        assert retrieved is not None
        assert retrieved.id == "retrieve-test"
        assert retrieved.model == "opus-4.5"
        assert retrieved.cwd == "/workspace"
        assert retrieved.axiom_encode_version == ""

    def test_get_session_returns_none_for_unknown(self, experiment_db):
        """Test that get_session returns None for unknown ID."""
        retrieved = experiment_db.get_session("nonexistent-id")
        assert retrieved is None

    def test_log_event_to_session(self, experiment_db):
        """Test logging events to a session."""
        experiment_db.start_session(session_id="event-test")

        event = experiment_db.log_event(
            session_id="event-test",
            event_type="agent_start",
            content="Test prompt",
            metadata={"agent_type": "encoder"},
        )

        assert event.sequence == 1
        assert event.event_type == "agent_start"

    def test_get_session_events(self, experiment_db):
        """Test retrieving all events for a session."""
        experiment_db.start_session(session_id="events-test")

        experiment_db.log_event(
            session_id="events-test", event_type="agent_start", content="Starting"
        )
        experiment_db.log_event(
            session_id="events-test", event_type="agent_end", content="Done"
        )

        events = experiment_db.get_session_events("events-test")
        assert len(events) == 2
        assert events[0].event_type == "agent_start"
        assert events[1].event_type == "agent_end"

    def test_session_event_count_updates(self, experiment_db):
        """Test that session event_count is tracked."""
        experiment_db.start_session(session_id="count-test")

        for i in range(3):
            experiment_db.log_event(session_id="count-test", event_type=f"event_{i}")

        session = experiment_db.get_session("count-test")
        assert session.event_count == 3


class TestRowToRun:
    """Tests for _row_to_run with the current schema."""

    def test_row_to_run(self, experiment_db):
        """Test _row_to_run parses current review result rows."""
        import json

        review_results = json.dumps(
            {
                "reviews": [
                    {
                        "reviewer": "rulespec_reviewer",
                        "passed": True,
                        "items_checked": 10,
                        "items_passed": 8,
                        "critical_issues": [],
                        "important_issues": [],
                        "minor_issues": [],
                        "lessons": "",
                    }
                ],
                "policyengine_match": 0.95,
                "oracle_context": {},
                "lessons": "Some lessons",
            }
        )
        row = (
            "test-id",
            "2024-01-01T00:00:00",
            "26 USC 32",
            "/path/file.yaml",
            "source text",
            "{}",
            "[]",
            3000,
            "encoder",
            "opus",
            "content",
            "sess-456",
            1,
            None,
            review_results,
            "Some lessons",
            "0.2.0",
            "{}",
            100,
            50,
            10,
            5,
            2,
            0.0123,
            0.0111,
            1,
        )
        run = experiment_db._row_to_run(row)
        assert run.id == "test-id"
        assert run.review_results is not None
        assert len(run.review_results.reviews) == 1
        assert run.review_results.reviews[0].passed is True
        assert run.review_results.policyengine_match == 0.95
        assert run.lessons == "Some lessons"
        assert run.rulespec_content == "content"
        assert run.source_text == "source text"
        assert run.axiom_encode_version == "0.2.0"
        assert run.outcome == {}

    def test_row_to_run_rejects_removed_taxsim_field(self, experiment_db):
        import json

        review_results = json.dumps(
            {
                "reviews": [],
                "policyengine_match": None,
                "taxsim_match": None,
                "oracle_context": {},
                "lessons": "",
            }
        )
        row = (
            "test-id",
            "2024-01-01T00:00:00",
            "26 USC 32",
            "/path/file.yaml",
            "source text",
            "{}",
            "[]",
            3000,
            "encoder",
            "opus",
            "content",
            None,
            1,
            None,
            review_results,
            "",
            "0.2.0",
            "{}",
            0,
            0,
            0,
            0,
            0,
            None,
            None,
            0,
        )

        with pytest.raises(ValueError, match="unsupported schema"):
            experiment_db._row_to_run(row)


class TestAxiomEncodeVersion:
    """Tests for axiom_encode version tracking on encoding runs."""

    def test_create_run_sets_version(self):
        """Test that create_run auto-populates axiom_encode_version."""
        from axiom_encode import __version__
        from axiom_encode.harness.encoding_db import create_run

        run = create_run(
            file_path="/tmp/test.yaml",
            citation="26 USC 21",
            agent_type="encoder",
            agent_model="opus",
            rulespec_content="test",
        )
        assert run.axiom_encode_version == __version__

    def test_version_persisted_in_db(self, experiment_db, sample_encoding_run):
        """Test that axiom_encode_version is persisted and retrieved from DB."""
        sample_encoding_run.axiom_encode_version = "0.2.0"
        experiment_db.log_run(sample_encoding_run)

        retrieved = experiment_db.get_run(sample_encoding_run.id)
        assert retrieved.axiom_encode_version == "0.2.0"

    def test_version_defaults_empty_for_old_runs(
        self, experiment_db, sample_encoding_run
    ):
        """Test that runs without axiom_encode_version default to empty string."""
        sample_encoding_run.axiom_encode_version = ""
        experiment_db.log_run(sample_encoding_run)

        retrieved = experiment_db.get_run(sample_encoding_run.id)
        assert retrieved.axiom_encode_version == ""


class TestUpdateSessionTokens:
    """Tests for updating session tokens."""

    def test_update_session_tokens(self, experiment_db):
        """Test updating token usage for a session."""
        experiment_db.start_session(
            model="test-model", cwd="/tmp", session_id="token-test"
        )

        experiment_db.update_session_tokens(
            session_id="token-test",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=200,
            estimated_cost_usd=0.0123,
        )

        session = experiment_db.get_session("token-test")
        # Session.total_tokens = input_tokens + output_tokens = 1500
        assert session.total_tokens == 1500
        with sqlite3.connect(experiment_db.db_path) as conn:
            cost = conn.execute(
                "SELECT estimated_cost_usd FROM sessions WHERE id = 'token-test'"
            ).fetchone()[0]
        assert cost == 0.0123

    def test_accumulates_across_writes(self, experiment_db):
        """Each stage's write adds onto the session totals."""
        experiment_db.start_session(
            model="test-model", cwd="/tmp", session_id="accumulate-test"
        )

        experiment_db.update_session_tokens(
            session_id="accumulate-test",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=200,
            cache_creation_tokens=100,
            reasoning_output_tokens=50,
            estimated_cost_usd=0.01,
        )
        experiment_db.update_session_tokens(
            session_id="accumulate-test",
            input_tokens=2000,
            output_tokens=1000,
            cache_read_tokens=400,
            cache_creation_tokens=200,
            reasoning_output_tokens=100,
            estimated_cost_usd=0.02,
        )

        session = experiment_db.get_session("accumulate-test")
        assert session.input_tokens == 3000
        assert session.output_tokens == 1500
        assert session.cache_read_tokens == 600
        assert session.cache_creation_tokens == 300
        assert session.reasoning_output_tokens == 150
        assert session.total_tokens == 4500
        assert session.estimated_cost_usd == pytest.approx(0.03)

    def test_unknown_cost_increment_poisons_total(self, experiment_db):
        """A token-spending write without a cost keeps the total unknown."""
        experiment_db.start_session(
            model="test-model", cwd="/tmp", session_id="poison-test"
        )

        experiment_db.update_session_tokens(
            session_id="poison-test",
            input_tokens=1000,
            output_tokens=500,
            estimated_cost_usd=None,
        )
        experiment_db.update_session_tokens(
            session_id="poison-test",
            input_tokens=2000,
            output_tokens=1000,
            estimated_cost_usd=0.02,
        )

        session = experiment_db.get_session("poison-test")
        assert session.input_tokens == 3000
        assert session.estimated_cost_usd is None

    def test_zero_token_costless_write_preserves_known_cost(self, experiment_db):
        """A bookkeeping write that spends nothing must not wipe a known cost."""
        experiment_db.start_session(
            model="test-model", cwd="/tmp", session_id="preserve-test"
        )

        experiment_db.update_session_tokens(
            session_id="preserve-test",
            input_tokens=1000,
            output_tokens=500,
            estimated_cost_usd=0.05,
        )
        experiment_db.update_session_tokens(session_id="preserve-test")

        session = experiment_db.get_session("preserve-test")
        assert session.input_tokens == 1000
        assert session.estimated_cost_usd == pytest.approx(0.05)

    def test_new_session_cost_starts_unknown(self, experiment_db):
        """A session that never records usage reports unknown cost, not $0."""
        experiment_db.start_session(
            model="test-model", cwd="/tmp", session_id="fresh-test"
        )

        session = experiment_db.get_session("fresh-test")
        assert session.estimated_cost_usd is None

    def test_missing_session_is_a_no_op(self, experiment_db):
        """Writing to an unknown session id must not raise."""
        experiment_db.update_session_tokens(
            session_id="does-not-exist", input_tokens=100
        )


class TestRunCostLedger:
    """Tests for per-run token usage and cost persistence."""

    def test_run_token_ledger_round_trips(self, experiment_db, sample_encoding_run):
        from axiom_encode.harness.encoding_db import TokenUsage

        sample_encoding_run.tokens = TokenUsage(
            input_tokens=76_000,
            output_tokens=3_700,
            cache_read_tokens=1_200,
            cache_creation_tokens=800,
            reasoning_output_tokens=400,
        )
        sample_encoding_run.estimated_cost_usd = 0.3322
        sample_encoding_run.actual_cost_usd = 0.31
        sample_encoding_run.generation_attempt_count = 2
        experiment_db.log_run(sample_encoding_run)

        retrieved = experiment_db.get_run(sample_encoding_run.id)
        assert retrieved.tokens.input_tokens == 76_000
        assert retrieved.tokens.output_tokens == 3_700
        assert retrieved.tokens.cache_read_tokens == 1_200
        assert retrieved.tokens.cache_creation_tokens == 800
        assert retrieved.tokens.reasoning_output_tokens == 400
        assert retrieved.estimated_cost_usd == pytest.approx(0.3322)
        assert retrieved.actual_cost_usd == pytest.approx(0.31)
        assert retrieved.generation_attempt_count == 2

    def test_run_cost_defaults_to_unknown(self, experiment_db, sample_encoding_run):
        """Runs without usage keep cost as None (unknown), never zero."""
        experiment_db.log_run(sample_encoding_run)

        retrieved = experiment_db.get_run(sample_encoding_run.id)
        assert retrieved.tokens.input_tokens == 0
        assert retrieved.estimated_cost_usd is None
        assert retrieved.actual_cost_usd is None
        assert retrieved.generation_attempt_count == 0
