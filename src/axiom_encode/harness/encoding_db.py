"""
Encoding Database - tracks encoding runs for continuous improvement.

Key insight: We learn from the JOURNEY (errors, fixes, iterations),
not from comparing predictions to actuals.

Now also tracks full session transcripts for replay and analysis.
"""

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

# Per-run token/cost ledger columns, shared with the Supabase sync so the
# local schema, the payload, and the fallback ladder cannot drift apart.
RUN_COST_COLUMNS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_output_tokens",
    "estimated_cost_usd",
    "actual_cost_usd",
    "generation_attempt_count",
)

RUN_COLUMNS = (
    "id",
    "timestamp",
    "citation",
    "file_path",
    "source_text",
    "complexity_json",
    "iterations_json",
    "total_duration_ms",
    "agent_type",
    "agent_model",
    "rulespec_content",
    "session_id",
    "iteration",
    "parent_run_id",
    "review_results_json",
    "lessons",
    "axiom_encode_version",
    "outcome_json",
) + RUN_COST_COLUMNS

SESSION_COLUMNS = (
    "id",
    "run_id",
    "started_at",
    "ended_at",
    "model",
    "cwd",
    "event_count",
    "total_tokens",
    "axiom_encode_version",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_output_tokens",
    "estimated_cost_usd",
)


@dataclass
class TokenUsage:
    """Token usage for a session or run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def has_recorded_usage(self) -> bool:
        """True when any token counter is non-zero (usage was measured)."""
        return any(
            (
                self.input_tokens,
                self.output_tokens,
                self.cache_read_tokens,
                self.cache_creation_tokens,
                self.reasoning_output_tokens,
            )
        )


# Session event types
EventType = Literal[
    "session_start",
    "session_end",
    "user_prompt",
    "assistant_response",
    "tool_call",
    "tool_result",
    "subagent_start",
    "subagent_end",
    "agent_start",
    "agent_assistant",
    "agent_end",
    "provenance_plan",
    "provenance_decision",
    "provenance_reasoning",
    "provenance_artifact",
    "provenance_sidecar",
    "provenance_validation",
    "provenance_review",
    # Validation events (3-tier pipeline)
    "validation_ci_start",
    "validation_ci_end",
    "validation_oracle_start",
    "validation_oracle_end",
    "validation_llm_start",
    "validation_llm_end",
    # Eval-backed encode events
    "encode_request",
    "encode_result",
    "encode_outcome",
    "encode_issue",
]


@dataclass
class SessionEvent:
    """A single event in a session transcript."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    session_id: str = ""
    sequence: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""  # EventType
    tool_name: Optional[str] = None
    content: str = ""  # Main content (prompt, response, tool input/output)
    metadata: dict = field(default_factory=dict)  # Extra data (tokens, duration, etc.)


@dataclass
class Session:
    """A full Claude Code session transcript."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    run_id: Optional[str] = None  # FK to EncodingRun if this is an encoding session
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    model: str = ""
    cwd: str = ""
    axiom_encode_version: str = ""
    event_count: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_output_tokens: int = 0
    estimated_cost_usd: Optional[float] = None


@dataclass
class ComplexityFactors:
    """Upfront analysis of statute complexity."""

    cross_references: list[str] = field(default_factory=list)  # ["1402(a)", "164(f)"]
    has_nested_structure: bool = False
    has_numeric_thresholds: bool = False
    has_phase_in_out: bool = False
    estimated_variables: int = 1
    estimated_parameters: int = 0


@dataclass
class IterationError:
    """An error encountered during encoding."""

    error_type: str  # "parse", "test", "import", "style", "other"
    message: str
    variable: Optional[str] = None  # Which variable failed, if applicable
    fix_applied: Optional[str] = None  # What fix was attempted


@dataclass
class Iteration:
    """A single encoding attempt."""

    attempt: int
    duration_ms: int
    errors: list[IterationError] = field(default_factory=list)
    success: bool = False


@dataclass
class OracleResult:
    """Detailed result from an oracle validator."""

    name: str
    score: Optional[float] = None  # Match rate 0-1
    passed: bool = False
    issues: list[str] = field(default_factory=list)
    duration_ms: int = 0
    test_cases_run: int = 0
    test_cases_passed: int = 0


@dataclass
class ReviewResult:
    """Result from a single reviewer (checklist-based)."""

    reviewer: str = ""  # "rulespec", "formula", "parameter", "integration"
    passed: bool = False
    items_checked: int = 0
    items_passed: int = 0
    critical_issues: list[str] = field(default_factory=list)
    important_issues: list[str] = field(default_factory=list)
    minor_issues: list[str] = field(default_factory=list)
    lessons: str = ""


@dataclass
class ReviewResults:
    """Aggregated results from all reviewers."""

    reviews: list[ReviewResult] = field(default_factory=list)
    policyengine_match: Optional[float] = None
    oracle_context: dict = field(default_factory=dict)
    lessons: str = ""

    @property
    def passed(self) -> bool:
        """All reviews passed."""
        return all(r.passed for r in self.reviews) if self.reviews else False

    @property
    def total_critical_issues(self) -> int:
        """Total critical issues across all reviews."""
        return sum(len(r.critical_issues) for r in self.reviews)


@dataclass
class EncodingRun:
    """A complete encoding run from start to finish."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)

    # What we're encoding
    citation: str = ""
    file_path: str = ""
    source_text: Optional[str] = None

    # Upfront analysis
    complexity: ComplexityFactors = field(default_factory=ComplexityFactors)

    # Review results
    review_results: Optional[ReviewResults] = None
    lessons: str = ""

    # Iteration tracking
    iteration: int = 1
    parent_run_id: Optional[str] = None

    # The journey
    iterations: list[Iteration] = field(default_factory=list)

    # Final result
    total_duration_ms: int = 0
    rulespec_content: str = ""

    # Agent info
    agent_type: str = "encoder"
    agent_model: str = ""
    axiom_encode_version: str = ""
    outcome: dict = field(default_factory=dict)

    # Model usage across all generation attempts (cache-aware)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    # None means "unknown" (some attempt lacked usage/pricing), never 0.
    estimated_cost_usd: Optional[float] = None
    actual_cost_usd: Optional[float] = None
    generation_attempt_count: int = 0

    # Session linkage
    session_id: Optional[str] = None

    @property
    def iterations_needed(self) -> int:
        return len(self.iterations)

    @property
    def success(self) -> bool:
        if isinstance(self.outcome, dict) and "final_success" in self.outcome:
            return bool(self.outcome["final_success"])
        return self.iterations and self.iterations[-1].success

    @property
    def all_errors(self) -> list[IterationError]:
        errors = []
        for it in self.iterations:
            errors.extend(it.errors)
        return errors


def create_run(
    file_path: str,
    citation: str,
    agent_type: str,
    agent_model: str,
    rulespec_content: str,
    source_text: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    review_results: Optional[ReviewResults] = None,
    lessons: str = "",
) -> EncodingRun:
    """Factory function to create an EncodingRun with defaults."""
    from axiom_encode import __version__

    return EncodingRun(
        file_path=file_path,
        citation=citation,
        agent_type=agent_type,
        agent_model=agent_model,
        rulespec_content=rulespec_content,
        source_text=source_text,
        iteration=2 if parent_run_id else 1,
        parent_run_id=parent_run_id,
        review_results=review_results,
        lessons=lessons,
        axiom_encode_version=__version__,
    )


class EncodingDB:
    """SQLite-based encoding database."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Encoding runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS encoding_runs (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                citation TEXT,
                file_path TEXT,
                source_text TEXT,
                complexity_json TEXT,
                iterations_json TEXT,
                total_duration_ms INTEGER,
                agent_type TEXT,
                agent_model TEXT,
                rulespec_content TEXT,
                session_id TEXT,
                iteration INTEGER DEFAULT 1,
                parent_run_id TEXT,
                review_results_json TEXT,
                lessons TEXT DEFAULT '',
                axiom_encode_version TEXT DEFAULT '',
                outcome_json TEXT DEFAULT '{}',
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_creation_tokens INTEGER DEFAULT 0,
                reasoning_output_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                generation_attempt_count INTEGER DEFAULT 0
            )
        """)

        # Keep active developer databases usable when the schema grows.
        for col, col_type, default in [
            ("source_text", "TEXT", None),
            ("rulespec_content", "TEXT", None),
            ("session_id", "TEXT", None),
            ("iteration", "INTEGER", "1"),
            ("parent_run_id", "TEXT", None),
            ("review_results_json", "TEXT", None),
            ("lessons", "TEXT", "''"),
            ("axiom_encode_version", "TEXT", "''"),
            ("outcome_json", "TEXT", "'{}'"),
            ("input_tokens", "INTEGER", "0"),
            ("output_tokens", "INTEGER", "0"),
            ("cache_read_tokens", "INTEGER", "0"),
            ("cache_creation_tokens", "INTEGER", "0"),
            ("reasoning_output_tokens", "INTEGER", "0"),
            ("estimated_cost_usd", "REAL", None),
            ("actual_cost_usd", "REAL", None),
            ("generation_attempt_count", "INTEGER", "0"),
        ]:
            try:
                stmt = f"ALTER TABLE encoding_runs ADD COLUMN {col} {col_type}"
                if default is not None:
                    stmt += f" DEFAULT {default}"
                cursor.execute(stmt)
            except sqlite3.OperationalError:
                pass  # Column already exists

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_citation ON encoding_runs(citation)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON encoding_runs(timestamp)
        """)

        # Calibration snapshots table (per-metric rows for trend analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_snapshots (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                predicted_mean REAL,
                actual_mean REAL,
                mse REAL,
                n_samples INTEGER
            )
        """)

        # Sessions table - full Claude Code session transcripts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES encoding_runs(id)
            )
        """)

        try:
            cursor.execute("ALTER TABLE sessions ADD COLUMN run_id TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_run ON sessions(run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_started ON sessions(started_at)
        """)

        # Session events table - individual events within a session
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                sequence INTEGER,
                timestamp TEXT,
                event_type TEXT,
                tool_name TEXT,
                content TEXT,
                metadata_json TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_session ON session_events(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_type ON session_events(event_type)
        """)

        # =====================================================================
        # Add token usage columns to sessions (migration)
        # =====================================================================
        for col in [
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
            "reasoning_output_tokens",
            "estimated_cost_usd",
            "axiom_encode_version",
        ]:
            try:
                if col == "axiom_encode_version":
                    col_type = "TEXT DEFAULT ''"
                elif col == "estimated_cost_usd":
                    # No default: NULL means "cost unknown", which must stay
                    # distinguishable from a measured $0.
                    col_type = "REAL"
                else:
                    col_type = "INTEGER DEFAULT 0"
                cursor.execute(f"ALTER TABLE sessions ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists

        conn.commit()
        conn.close()

    def log_run(self, run: EncodingRun) -> str:
        """Log a completed encoding run. Returns the run ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Convert dataclasses to JSON
        complexity_json = json.dumps(
            {
                "cross_references": run.complexity.cross_references,
                "has_nested_structure": run.complexity.has_nested_structure,
                "has_numeric_thresholds": run.complexity.has_numeric_thresholds,
                "has_phase_in_out": run.complexity.has_phase_in_out,
                "estimated_variables": run.complexity.estimated_variables,
                "estimated_parameters": run.complexity.estimated_parameters,
            }
        )

        iterations_json = json.dumps(
            [
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
            ]
        )

        # Serialize review_results (new checklist-based format)
        review_results_json = None
        if run.review_results:
            review_results_json = json.dumps(
                {
                    "reviews": [
                        {
                            "reviewer": r.reviewer,
                            "passed": r.passed,
                            "items_checked": r.items_checked,
                            "items_passed": r.items_passed,
                            "critical_issues": r.critical_issues,
                            "important_issues": r.important_issues,
                            "minor_issues": r.minor_issues,
                            "lessons": r.lessons,
                        }
                        for r in run.review_results.reviews
                    ],
                    "policyengine_match": run.review_results.policyengine_match,
                    "oracle_context": run.review_results.oracle_context,
                    "lessons": run.review_results.lessons,
                }
            )

        cursor.execute(
            """
            INSERT OR REPLACE INTO encoding_runs
            (id, timestamp, citation, file_path, source_text, complexity_json,
             iterations_json, total_duration_ms, agent_type, agent_model,
             rulespec_content, session_id, iteration, parent_run_id,
             review_results_json, lessons, axiom_encode_version, outcome_json,
             input_tokens, output_tokens, cache_read_tokens,
             cache_creation_tokens, reasoning_output_tokens,
             estimated_cost_usd, actual_cost_usd, generation_attempt_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                run.id,
                run.timestamp.isoformat(),
                run.citation,
                run.file_path,
                run.source_text,
                complexity_json,
                iterations_json,
                run.total_duration_ms,
                run.agent_type,
                run.agent_model,
                run.rulespec_content,
                run.session_id,
                run.iteration,
                run.parent_run_id,
                review_results_json,
                run.lessons,
                run.axiom_encode_version,
                json.dumps(run.outcome or {}),
                run.tokens.input_tokens,
                run.tokens.output_tokens,
                run.tokens.cache_read_tokens,
                run.tokens.cache_creation_tokens,
                run.tokens.reasoning_output_tokens,
                run.estimated_cost_usd,
                run.actual_cost_usd,
                run.generation_attempt_count,
            ),
        )

        conn.commit()
        conn.close()

        return run.id

    def update_run_outcome(self, run_id: str, outcome: dict) -> None:
        """Update final encode/apply outcome metadata for a run."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE encoding_runs SET outcome_json = ? WHERE id = ?",
            (json.dumps(outcome or {}, sort_keys=True), run_id),
        )

        conn.commit()
        conn.close()

    def get_run(self, run_id: str) -> Optional[EncodingRun]:
        """Get a specific run by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT {', '.join(RUN_COLUMNS)} FROM encoding_runs WHERE id = ?",
            (run_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_run(row)

    def get_recent_runs(self, limit: int = 20) -> list[EncodingRun]:
        """Get recent runs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT {', '.join(RUN_COLUMNS)} FROM encoding_runs "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_run(row) for row in rows]

    def get_runs_for_citation(self, citation: str) -> list[EncodingRun]:
        """Get all runs for a specific citation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT {', '.join(RUN_COLUMNS)} FROM encoding_runs "
            "WHERE citation = ? ORDER BY timestamp DESC",
            (citation,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_run(row) for row in rows]

    def update_review_results(self, run_id: str, review_results: ReviewResults) -> None:
        """Update a run with review results after validation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        review_results_json = json.dumps(
            {
                "reviews": [
                    {
                        "reviewer": r.reviewer,
                        "passed": r.passed,
                        "items_checked": r.items_checked,
                        "items_passed": r.items_passed,
                        "critical_issues": r.critical_issues,
                        "important_issues": r.important_issues,
                        "minor_issues": r.minor_issues,
                        "lessons": r.lessons,
                    }
                    for r in review_results.reviews
                ],
                "policyengine_match": review_results.policyengine_match,
                "oracle_context": review_results.oracle_context,
                "lessons": review_results.lessons,
            }
        )

        cursor.execute(
            "UPDATE encoding_runs SET review_results_json = ? WHERE id = ?",
            (review_results_json, run_id),
        )

        conn.commit()
        conn.close()

    def get_error_stats(self) -> dict:
        """Get error type distribution."""
        runs = self.get_recent_runs(limit=100)

        error_counts = {}
        for run in runs:
            for err in run.all_errors:
                error_counts[err.error_type] = error_counts.get(err.error_type, 0) + 1

        total = sum(error_counts.values())
        return {
            "counts": error_counts,
            "percentages": {
                k: v / total * 100 if total > 0 else 0 for k, v in error_counts.items()
            },
            "total_runs": len(runs),
            "total_errors": total,
        }

    def get_iteration_stats(self) -> dict:
        """Get iteration distribution."""
        runs = self.get_recent_runs(limit=100)

        iteration_counts = {}
        for run in runs:
            n = run.iterations_needed
            iteration_counts[n] = iteration_counts.get(n, 0) + 1

        total = len(runs)
        avg = (
            sum(n * c for n, c in iteration_counts.items()) / total if total > 0 else 0
        )

        return {
            "distribution": iteration_counts,
            "average": avg,
            "first_try_rate": iteration_counts.get(1, 0) / total * 100
            if total > 0
            else 0,
            "total_runs": total,
        }

    def _row_to_run(self, row) -> EncodingRun:
        """Convert a current-schema database row to EncodingRun."""
        if len(row) != len(RUN_COLUMNS):
            raise ValueError(
                f"Expected {len(RUN_COLUMNS)} encoding run columns, got {len(row)}"
            )

        values = dict(zip(RUN_COLUMNS, row))
        id = values["id"]
        timestamp = values["timestamp"]
        citation = values["citation"]
        file_path = values["file_path"]
        source_text = values["source_text"]
        complexity_json = values["complexity_json"]
        iterations_json = values["iterations_json"]
        total_duration_ms = values["total_duration_ms"]
        agent_type = values["agent_type"]
        agent_model = values["agent_model"]
        rulespec_content = values["rulespec_content"]
        session_id = values["session_id"]
        iteration = values["iteration"]
        parent_run_id = values["parent_run_id"]
        review_results_json = values["review_results_json"]
        lessons = values["lessons"]
        axiom_encode_version = values["axiom_encode_version"]
        outcome_json = values["outcome_json"]
        tokens = TokenUsage(
            input_tokens=int(values["input_tokens"] or 0),
            output_tokens=int(values["output_tokens"] or 0),
            cache_read_tokens=int(values["cache_read_tokens"] or 0),
            cache_creation_tokens=int(values["cache_creation_tokens"] or 0),
            reasoning_output_tokens=int(values["reasoning_output_tokens"] or 0),
        )
        estimated_cost_usd = (
            float(values["estimated_cost_usd"])
            if values["estimated_cost_usd"] is not None
            else None
        )
        actual_cost_usd = (
            float(values["actual_cost_usd"])
            if values["actual_cost_usd"] is not None
            else None
        )
        generation_attempt_count = int(values["generation_attempt_count"] or 0)

        # Parse complexity
        c = json.loads(complexity_json) if complexity_json else {}
        complexity = ComplexityFactors(
            cross_references=c.get("cross_references", []),
            has_nested_structure=c.get("has_nested_structure", False),
            has_numeric_thresholds=c.get("has_numeric_thresholds", False),
            has_phase_in_out=c.get("has_phase_in_out", False),
            estimated_variables=c.get("estimated_variables", 1),
            estimated_parameters=c.get("estimated_parameters", 0),
        )

        # Parse iterations
        iterations = []
        if iterations_json:
            for it_data in json.loads(iterations_json):
                errors = [
                    IterationError(
                        error_type=e["error_type"],
                        message=e["message"],
                        variable=e.get("variable"),
                        fix_applied=e.get("fix_applied"),
                    )
                    for e in it_data.get("errors", [])
                ]
                iterations.append(
                    Iteration(
                        attempt=it_data["attempt"],
                        duration_ms=it_data["duration_ms"],
                        errors=errors,
                        success=it_data.get("success", False),
                    )
                )

        # Parse review_results.
        review_results = None
        if review_results_json:
            rr = json.loads(review_results_json)
            allowed_fields = {
                "reviews",
                "policyengine_match",
                "oracle_context",
                "lessons",
            }
            if not isinstance(rr, dict) or set(rr) - allowed_fields:
                raise ValueError(
                    "encoding run review_results_json uses an unsupported schema"
                )
            review_results = ReviewResults(
                reviews=[
                    ReviewResult(
                        reviewer=r.get("reviewer", ""),
                        passed=r.get("passed", False),
                        items_checked=r.get("items_checked", 0),
                        items_passed=r.get("items_passed", 0),
                        critical_issues=r.get("critical_issues", []),
                        important_issues=r.get("important_issues", []),
                        minor_issues=r.get("minor_issues", []),
                        lessons=r.get("lessons", ""),
                    )
                    for r in rr.get("reviews", [])
                ],
                policyengine_match=rr.get("policyengine_match"),
                oracle_context=rr.get("oracle_context", {}),
                lessons=rr.get("lessons", ""),
            )

        return EncodingRun(
            id=id,
            timestamp=datetime.fromisoformat(timestamp),
            citation=citation,
            file_path=file_path,
            source_text=source_text,
            complexity=complexity,
            review_results=review_results,
            lessons=lessons or "",
            iteration=iteration or 1,
            parent_run_id=parent_run_id,
            iterations=iterations,
            total_duration_ms=total_duration_ms or 0,
            agent_type=agent_type or "encoder",
            agent_model=agent_model or "",
            axiom_encode_version=axiom_encode_version or "",
            outcome=json.loads(outcome_json) if outcome_json else {},
            rulespec_content=rulespec_content or "",
            tokens=tokens,
            estimated_cost_usd=estimated_cost_usd,
            actual_cost_usd=actual_cost_usd,
            generation_attempt_count=generation_attempt_count,
            session_id=session_id,
        )

    # =========================================================================
    # Session Logging Methods
    # =========================================================================

    def start_session(
        self,
        model: str = "",
        cwd: str = "",
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        axiom_encode_version: str = "",
    ) -> Session:
        """Start a new session and return it."""
        session = Session(
            run_id=run_id,
            model=model,
            cwd=cwd or os.getcwd(),
            axiom_encode_version=axiom_encode_version,
        )
        # Allow custom session_id for SDK orchestrator
        if session_id:
            session.id = session_id

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO sessions (
                id, run_id, started_at, model, cwd, event_count, total_tokens,
                estimated_cost_usd, axiom_encode_version
            )
            VALUES (?, ?, ?, ?, ?, 0, 0, NULL, ?)
        """,
            (
                session.id,
                session.run_id,
                session.started_at.isoformat(),
                session.model,
                session.cwd,
                session.axiom_encode_version,
            ),
        )

        conn.commit()
        conn.close()

        return session

    def end_session(self, session_id: str) -> None:
        """Mark a session as ended."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE sessions SET ended_at = ? WHERE id = ?
        """,
            (datetime.now().isoformat(), session_id),
        )

        conn.commit()
        conn.close()

    def log_event(
        self,
        session_id: str,
        event_type: str,
        content: str = "",
        tool_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> SessionEvent:
        """Log an event to a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get next sequence number
        cursor.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM session_events WHERE session_id = ?",
            (session_id,),
        )
        sequence = cursor.fetchone()[0]

        event = SessionEvent(
            session_id=session_id,
            sequence=sequence,
            event_type=event_type,
            tool_name=tool_name,
            content=content,
            metadata=metadata or {},
        )

        cursor.execute(
            """
            INSERT INTO session_events (id, session_id, sequence, timestamp, event_type, tool_name, content, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                event.id,
                event.session_id,
                event.sequence,
                event.timestamp.isoformat(),
                event.event_type,
                event.tool_name,
                event.content,
                json.dumps(event.metadata),
            ),
        )

        # Update event count
        cursor.execute(
            """
            UPDATE sessions SET event_count = event_count + 1 WHERE id = ?
        """,
            (session_id,),
        )

        conn.commit()
        conn.close()

        return event

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT {", ".join(SESSION_COLUMNS)}
            FROM sessions
            WHERE id = ?
        """,
            (session_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_session(row)

    @staticmethod
    def _row_to_session(row) -> Session:
        """Convert a SESSION_COLUMNS-ordered row to a Session."""
        if len(row) != len(SESSION_COLUMNS):
            raise ValueError(
                f"Expected {len(SESSION_COLUMNS)} session columns, got {len(row)}"
            )
        values = dict(zip(SESSION_COLUMNS, row))
        return Session(
            id=values["id"],
            run_id=values["run_id"],
            started_at=datetime.fromisoformat(values["started_at"])
            if values["started_at"]
            else datetime.now(),
            ended_at=datetime.fromisoformat(values["ended_at"])
            if values["ended_at"]
            else None,
            model=values["model"] or "",
            cwd=values["cwd"] or "",
            axiom_encode_version=values["axiom_encode_version"] or "",
            event_count=values["event_count"] or 0,
            total_tokens=values["total_tokens"] or 0,
            input_tokens=values["input_tokens"] or 0,
            output_tokens=values["output_tokens"] or 0,
            cache_read_tokens=values["cache_read_tokens"] or 0,
            cache_creation_tokens=values["cache_creation_tokens"] or 0,
            reasoning_output_tokens=values["reasoning_output_tokens"] or 0,
            estimated_cost_usd=(
                float(values["estimated_cost_usd"])
                if values["estimated_cost_usd"] is not None
                else None
            ),
        )

    def get_session_events(self, session_id: str) -> list[SessionEvent]:
        """Get all events for a session, ordered by sequence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, session_id, sequence, timestamp, event_type, tool_name, content, metadata_json
            FROM session_events
            WHERE session_id = ?
            ORDER BY sequence
        """,
            (session_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        events = []
        for row in rows:
            events.append(
                SessionEvent(
                    id=row[0],
                    session_id=row[1],
                    sequence=row[2],
                    timestamp=datetime.fromisoformat(row[3])
                    if row[3]
                    else datetime.now(),
                    event_type=row[4] or "",
                    tool_name=row[5],
                    content=row[6] or "",
                    metadata=json.loads(row[7]) if row[7] else {},
                )
            )

        return events

    def get_recent_sessions(self, limit: int = 20) -> list[Session]:
        """Get recent sessions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT {", ".join(SESSION_COLUMNS)}
            FROM sessions
            ORDER BY started_at DESC
            LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_session(row) for row in rows]

    def get_session_stats(self) -> dict:
        """Get session statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total sessions
        cursor.execute("SELECT COUNT(*) FROM sessions")
        total = cursor.fetchone()[0]

        # Event type distribution
        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM session_events
            GROUP BY event_type
            ORDER BY count DESC
        """)
        event_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Tool usage
        cursor.execute("""
            SELECT tool_name, COUNT(*) as count
            FROM session_events
            WHERE tool_name IS NOT NULL
            GROUP BY tool_name
            ORDER BY count DESC
            LIMIT 20
        """)
        tool_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Average events per session
        cursor.execute("SELECT AVG(event_count) FROM sessions")
        avg_events = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_sessions": total,
            "event_type_counts": event_counts,
            "tool_usage": tool_counts,
            "avg_events_per_session": round(avg_events, 1),
        }

    def update_session_tokens(
        self,
        session_id: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        reasoning_output_tokens: int = 0,
        estimated_cost_usd: float | None = None,
    ) -> None:
        """Accumulate token usage and its provider/model-aware cost estimate.

        Token counts add onto whatever the session has already recorded, so
        each pipeline stage (generation, repair, review) can report its own
        usage without clobbering earlier writes. The cost estimate follows
        the same rule, except that a token-spending increment with an unknown
        cost (``None``) poisons the session total to ``None``: a partial sum
        would read as a real, too-low figure. A zero-token increment without
        a cost leaves the total untouched, and once poisoned while tokens are
        recorded the total stays ``None``. The whole write is one UPDATE so
        concurrent stages cannot lose each other's cost contribution.
        """
        increment_token_sum = (
            input_tokens
            + output_tokens
            + cache_read_tokens
            + cache_creation_tokens
            + reasoning_output_tokens
        )
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE sessions
            SET estimated_cost_usd = CASE
                    WHEN :has_cost = 0 AND :increment_tokens > 0 THEN NULL
                    WHEN :has_cost = 0 THEN estimated_cost_usd
                    WHEN estimated_cost_usd IS NULL
                         AND COALESCE(input_tokens, 0)
                             + COALESCE(output_tokens, 0)
                             + COALESCE(cache_read_tokens, 0)
                             + COALESCE(cache_creation_tokens, 0)
                             + COALESCE(reasoning_output_tokens, 0) > 0
                        THEN NULL
                    ELSE COALESCE(estimated_cost_usd, 0.0) + :cost
                END,
                input_tokens = COALESCE(input_tokens, 0) + :input,
                output_tokens = COALESCE(output_tokens, 0) + :output,
                cache_read_tokens = COALESCE(cache_read_tokens, 0) + :cache_read,
                cache_creation_tokens =
                    COALESCE(cache_creation_tokens, 0) + :cache_creation,
                reasoning_output_tokens =
                    COALESCE(reasoning_output_tokens, 0) + :reasoning,
                total_tokens = COALESCE(total_tokens, 0) + :input + :output
            WHERE id = :session_id
        """,
            {
                "session_id": session_id,
                "input": input_tokens,
                "output": output_tokens,
                "cache_read": cache_read_tokens,
                "cache_creation": cache_creation_tokens,
                "reasoning": reasoning_output_tokens,
                "increment_tokens": increment_token_sum,
                "has_cost": 0 if estimated_cost_usd is None else 1,
                "cost": float(estimated_cost_usd or 0.0),
            },
        )

        conn.commit()
        conn.close()
