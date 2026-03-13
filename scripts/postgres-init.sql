-- PostgreSQL init script: run once on first startup
-- Enable pgvector extension
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

-- Audit / event log
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL PRIMARY KEY,
    agent_id    TEXT REFERENCES agents(id),
    event_type  TEXT NOT NULL,
    payload     JSONB DEFAULT '{}',
    ts          TIMESTAMPTZ DEFAULT NOW()
);

-- Shared long-term memory with vector embeddings
CREATE TABLE IF NOT EXISTS shared_memory (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    agent_id    TEXT REFERENCES agents(id),
    content     TEXT NOT NULL,
    embedding   vector(1536),
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS shared_memory_embedding_idx
    ON shared_memory USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS audit_log_agent_ts_idx ON audit_log (agent_id, ts DESC);
CREATE INDEX IF NOT EXISTS shared_tasks_to_agent_idx ON shared_tasks (to_agent, status);
