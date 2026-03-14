-- PostgreSQL init script: run once on first startup (or re-run safely — all idempotent).
--
-- pgvector and pg_trgm are OPTIONAL. If the hosted DB doesn't have them,
-- comment out the CREATE EXTENSION lines below. The agent will work without them.
-- The agent's postgres_store.py also tries to enable these at runtime with a fallback.

-- Optional extensions (require superuser; comment out if unavailable)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Agent registry
CREATE TABLE IF NOT EXISTS agents (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    status      TEXT NOT NULL DEFAULT 'offline',  -- online|offline|busy
    model       TEXT,
    last_seen   TIMESTAMPTZ,
    metadata    JSONB DEFAULT '{}'
);

-- Shared task queue
CREATE TABLE IF NOT EXISTS shared_tasks (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    from_agent  TEXT REFERENCES agents(id),
    to_agent    TEXT REFERENCES agents(id),
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed
    description TEXT NOT NULL,
    result      TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Audit / event log (also used for agent broadcasts)
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL PRIMARY KEY,
    agent_id    TEXT REFERENCES agents(id),
    event_type  TEXT NOT NULL,
    payload     JSONB DEFAULT '{}',
    ts          TIMESTAMPTZ DEFAULT NOW()
);

-- Shared long-term memory (cross-agent knowledge base)
-- The embedding column is added automatically if pgvector is available.
CREATE TABLE IF NOT EXISTS shared_memory (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    agent_id    TEXT REFERENCES agents(id),
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Add vector embedding column if pgvector is available (safe to run multiple times)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'shared_memory' AND column_name = 'embedding'
        ) THEN
            ALTER TABLE shared_memory ADD COLUMN embedding vector(1536);
        END IF;
    END IF;
END$$;

-- Indexes
CREATE INDEX IF NOT EXISTS audit_log_agent_ts_idx    ON audit_log (agent_id, ts DESC);
CREATE INDEX IF NOT EXISTS audit_log_type_ts_idx     ON audit_log (event_type, ts DESC);
CREATE INDEX IF NOT EXISTS audit_log_ts_idx          ON audit_log (ts DESC);  -- for age-based cleanup
CREATE INDEX IF NOT EXISTS shared_tasks_to_agent_idx ON shared_tasks (to_agent, status);
CREATE INDEX IF NOT EXISTS shared_tasks_updated_idx  ON shared_tasks (updated_at DESC);  -- for cleanup
CREATE INDEX IF NOT EXISTS shared_memory_agent_idx   ON shared_memory (agent_id, created_at DESC);

-- Vector index (only if the embedding column exists)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'shared_memory' AND column_name = 'embedding'
    ) THEN
        IF NOT EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'shared_memory' AND indexname = 'shared_memory_embedding_idx'
        ) THEN
            CREATE INDEX shared_memory_embedding_idx
                ON shared_memory USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
        END IF;
    END IF;
END$$;
