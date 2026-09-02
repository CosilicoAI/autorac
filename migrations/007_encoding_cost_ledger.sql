-- Unified encoding cost ledger: per-run token usage and spend, plus the
-- session token columns previously dropped by the sync.
--
-- Every ledger column is nullable with no default: NULL means "not measured"
-- (pre-ledger rows, manifest-only reconstructions), which must stay
-- distinguishable from a measured zero.

ALTER TABLE encodings.encoding_runs
    ADD COLUMN IF NOT EXISTS input_tokens BIGINT,
    ADD COLUMN IF NOT EXISTS output_tokens BIGINT,
    ADD COLUMN IF NOT EXISTS cache_read_tokens BIGINT,
    ADD COLUMN IF NOT EXISTS cache_creation_tokens BIGINT,
    ADD COLUMN IF NOT EXISTS reasoning_output_tokens BIGINT,
    ADD COLUMN IF NOT EXISTS estimated_cost_usd NUMERIC,
    ADD COLUMN IF NOT EXISTS actual_cost_usd NUMERIC,
    ADD COLUMN IF NOT EXISTS generation_attempt_count INTEGER;

ALTER TABLE IF EXISTS telemetry.sdk_sessions
    ADD COLUMN IF NOT EXISTS cache_creation_tokens BIGINT,
    ADD COLUMN IF NOT EXISTS reasoning_output_tokens BIGINT;

DROP FUNCTION IF EXISTS encodings.get_encoding_runs(INTEGER, INTEGER);

CREATE OR REPLACE FUNCTION encodings.get_encoding_runs(
    limit_count INTEGER DEFAULT 100,
    offset_count INTEGER DEFAULT 0
)
RETURNS TABLE (
    id TEXT,
    "timestamp" TIMESTAMPTZ,
    citation TEXT,
    iterations JSONB,
    outcome JSONB,
    scores JSONB,
    has_issues BOOLEAN,
    note TEXT,
    total_duration_ms INTEGER,
    agent_type TEXT,
    agent_model TEXT,
    data_source TEXT,
    session_id TEXT,
    input_tokens BIGINT,
    output_tokens BIGINT,
    cache_read_tokens BIGINT,
    cache_creation_tokens BIGINT,
    reasoning_output_tokens BIGINT,
    estimated_cost_usd NUMERIC,
    actual_cost_usd NUMERIC,
    generation_attempt_count INTEGER
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = encodings
AS $$
    SELECT
        encoding_runs.id,
        encoding_runs.timestamp,
        encoding_runs.citation,
        encoding_runs.iterations,
        encoding_runs.outcome,
        COALESCE(encoding_runs.scores, encoding_runs.final_scores, '{}'::jsonb) AS scores,
        encoding_runs.has_issues,
        encoding_runs.note,
        encoding_runs.total_duration_ms,
        encoding_runs.agent_type,
        encoding_runs.agent_model,
        encoding_runs.data_source,
        encoding_runs.session_id,
        encoding_runs.input_tokens,
        encoding_runs.output_tokens,
        encoding_runs.cache_read_tokens,
        encoding_runs.cache_creation_tokens,
        encoding_runs.reasoning_output_tokens,
        encoding_runs.estimated_cost_usd,
        encoding_runs.actual_cost_usd,
        encoding_runs.generation_attempt_count
    FROM encodings.encoding_runs
    ORDER BY encoding_runs.timestamp DESC
    LIMIT GREATEST(1, LEAST(limit_count, 500))
    OFFSET GREATEST(0, offset_count);
$$;

GRANT EXECUTE ON FUNCTION encodings.get_encoding_runs(INTEGER, INTEGER) TO anon;
GRANT EXECUTE ON FUNCTION encodings.get_encoding_runs(INTEGER, INTEGER) TO authenticated;
