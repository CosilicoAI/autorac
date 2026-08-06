-- Live encode-run presence: one row per in-flight `axiom-encode encode`
-- invocation, heartbeated while the run is active, closed with a pointer to
-- the final encodings.encoding_runs row. Lets the ops dashboard surface
-- concurrent runs across machines in real time; a stale heartbeat on a
-- 'running' row means the encoder died mid-run.

CREATE TABLE IF NOT EXISTS encodings.live_encoding_runs (
    id TEXT PRIMARY KEY,
    citation TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    phase TEXT,
    attempt INTEGER,
    backend TEXT,
    model TEXT,
    encoder_version TEXT,
    -- Final encodings.encoding_runs id once the run is logged.
    run_id TEXT,
    -- Machine identity: hostname, username, platform, pid, is_ci.
    runner JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_live_encoding_runs_status_heartbeat
    ON encodings.live_encoding_runs(status, last_heartbeat_at DESC);
CREATE INDEX IF NOT EXISTS idx_live_encoding_runs_started
    ON encodings.live_encoding_runs(started_at DESC);

ALTER TABLE encodings.live_encoding_runs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS anon_read ON encodings.live_encoding_runs;
CREATE POLICY anon_read ON encodings.live_encoding_runs
    FOR SELECT TO anon USING (true);

DROP POLICY IF EXISTS authenticated_read ON encodings.live_encoding_runs;
CREATE POLICY authenticated_read ON encodings.live_encoding_runs
    FOR SELECT TO authenticated USING (true);

GRANT SELECT ON encodings.live_encoding_runs TO anon, authenticated;
GRANT ALL ON encodings.live_encoding_runs TO postgres, service_role;
