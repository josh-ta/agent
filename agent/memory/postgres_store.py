"""
PostgreSQL store: shared state for multi-agent coordination.

Optional — only used when POSTGRES_URL is set.
Uses asyncpg for async, non-blocking access.
Tables are created automatically on init() via CREATE TABLE IF NOT EXISTS.
pgvector/pg_trgm extensions are attempted but not required.

When pgvector is available AND an OpenAI API key is configured, share_memory()
generates embeddings via text-embedding-3-small and stores them in the
`embedding` column. search_shared_memory() then uses cosine distance (<=>)
for semantic similarity search, falling back to ILIKE keyword search otherwise.
"""

from __future__ import annotations

import json
import time

import asyncpg
import structlog

from agent.config import settings
from agent.memory.postgres_components import (
    AgentRegistryRepository,
    AuditLogRepository,
    PostgresMaintenance,
    SharedMemoryRepository,
    SharedTaskRepository,
)

log = structlog.get_logger()

# ── Embeddings ────────────────────────────────────────────────────────────────

async def _embed(text: str) -> list[float] | None:
    """Generate an embedding vector via OpenAI.  Returns None on any failure."""
    if not settings.has_embeddings:
        return None
    try:
        from openai import AsyncOpenAI  # soft import — already a dep via pydantic-ai
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        resp = await client.embeddings.create(
            model=settings.embedding_model,
            input=text,
            encoding_format="float",
        )
        return resp.data[0].embedding
    except Exception as exc:
        log.warning("embedding_failed", error=str(exc))
        return None


# ── Schema ────────────────────────────────────────────────────────────────────

_EXTENSIONS = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
]

SCHEMA = """
-- Agent registry
CREATE TABLE IF NOT EXISTS agents (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    status      TEXT NOT NULL DEFAULT 'offline',
    model       TEXT,
    last_seen   TIMESTAMPTZ,
    metadata    JSONB DEFAULT '{}'
);

-- Shared task queue
CREATE TABLE IF NOT EXISTS shared_tasks (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    from_agent  TEXT REFERENCES agents(id),
    to_agent    TEXT REFERENCES agents(id),
    status      TEXT NOT NULL DEFAULT 'pending',
    description TEXT NOT NULL,
    result      TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Audit / event log (also used for broadcasts between agents)
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL PRIMARY KEY,
    agent_id    TEXT REFERENCES agents(id),
    event_type  TEXT NOT NULL,
    payload     JSONB DEFAULT '{}',
    ts          TIMESTAMPTZ DEFAULT NOW()
);

-- Shared long-term memory (text only; vector column added separately if pgvector available)
CREATE TABLE IF NOT EXISTS shared_memory (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    agent_id    TEXT REFERENCES agents(id),
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS audit_log_agent_ts_idx  ON audit_log (agent_id, ts DESC);
CREATE INDEX IF NOT EXISTS audit_log_type_ts_idx   ON audit_log (event_type, ts DESC);
CREATE INDEX IF NOT EXISTS shared_tasks_to_agent_idx ON shared_tasks (to_agent, status);
CREATE INDEX IF NOT EXISTS shared_memory_agent_idx ON shared_memory (agent_id, created_at DESC);
"""

# Attempt to add the vector column if pgvector was successfully enabled.
_ADD_VECTOR_COLUMN = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='shared_memory' AND column_name='embedding'
    ) THEN
        ALTER TABLE shared_memory ADD COLUMN embedding vector(1536);
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS shared_memory_embedding_idx
    ON shared_memory USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


_PG_CLEANUP_INTERVAL_S = 3600  # run cleanup at most once per hour


class PostgresStore:
    def __init__(self, url: str) -> None:
        # asyncpg uses postgres:// or postgresql:// — strip the +asyncpg driver hint
        self._url = url.replace("postgresql+asyncpg://", "postgresql://").replace(
            "postgres+asyncpg://", "postgresql://"
        )
        self._pool: asyncpg.Pool | None = None  # type: ignore[type-arg]
        self._has_vector: bool = False
        self._has_embeddings: bool = False  # True when vector + OpenAI key both available
        self._last_cleanup_ts: float = 0.0
        self.registry = AgentRegistryRepository(self)
        self.tasks = SharedTaskRepository(self)
        self.audit = AuditLogRepository(self)
        self.shared_memory_repo = SharedMemoryRepository(self)
        self.maintenance = PostgresMaintenance(self)

    async def init(self) -> None:
        """Create connection pool and ensure schema exists."""
        self._pool = await asyncpg.create_pool(
            self._url,
            min_size=1,
            max_size=5,
            command_timeout=10,
        )
        log.info("postgres_connected")

        async with self._pool.acquire() as conn:
            # Try to enable optional extensions (requires superuser; skip on failure)
            for ext_sql in _EXTENSIONS:
                try:
                    await conn.execute(ext_sql)
                    if "vector" in ext_sql:
                        self._has_vector = True
                except Exception as exc:
                    log.warning("postgres_extension_skipped", sql=ext_sql, reason=str(exc))

            # Create tables (idempotent)
            await conn.execute(SCHEMA)

            # If pgvector loaded, try adding embedding column + index
            if self._has_vector:
                try:
                    await conn.execute(_ADD_VECTOR_COLUMN)
                    self._has_embeddings = settings.has_embeddings
                except Exception as exc:
                    log.warning("postgres_vector_index_skipped", reason=str(exc))

        log.info(
            "postgres_schema_ready",
            vector=self._has_vector,
            embeddings=self._has_embeddings,
            embedding_model=settings.embedding_model if self._has_embeddings else "n/a",
        )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    # ── Agent registry ────────────────────────────────────────────────────────

    async def register_agent(self) -> None:
        """Upsert this agent's presence in the agents table."""
        await self.registry.register_agent()
        log.info("agent_registered", name=settings.agent_name)

    async def set_offline(self) -> None:
        await self.registry.set_offline()

    async def list_agents(self) -> str:
        """Return a formatted list of all registered agents."""
        return await self.registry.list_agents()

    # ── Shared task queue ─────────────────────────────────────────────────────

    async def create_task(self, to_agent: str, description: str) -> str:
        """Create a task for another agent."""
        result = await self.tasks.create_task(to_agent, description)
        log.info("shared_task_created", to=to_agent)
        return result

    async def get_my_tasks(self) -> str:
        """Get pending tasks assigned to this agent (formatted for tool output)."""
        return await self.tasks.get_my_tasks()

    async def get_pending_task_rows(self) -> list[dict]:
        """Fetch pending tasks for the heartbeat A2A poller (returns raw dicts)."""
        return await self.tasks.get_pending_task_rows()

    async def mark_task_running(self, task_id: str) -> None:
        """Mark a shared task as running so it's not picked up twice."""
        await self.tasks.mark_task_running(task_id)

    async def complete_task(self, task_id: str, result: str) -> str:
        """Mark a shared task as done with a result."""
        result_text = await self.tasks.complete_task(task_id, result)
        log.info("shared_task_completed", id=task_id)
        return result_text

    # ── Broadcasts (audit_log) ────────────────────────────────────────────────

    async def broadcast_message(self, message: str, event_type: str = "broadcast") -> str:
        """Write a broadcast message to the audit log, visible to all agents."""
        return await self.audit.broadcast_message(message, event_type)

    async def read_broadcasts(self, limit: int = 20, event_type: str = "broadcast") -> str:
        """Read recent broadcast messages from the audit log."""
        return await self.audit.read_broadcasts(limit, event_type)

    # ── Shared memory ─────────────────────────────────────────────────────────

    async def share_memory(self, content: str, metadata: dict | None = None) -> str:
        """Write a fact to shared memory — visible to all agents."""
        return await self.shared_memory_repo.share_memory(content, metadata)

    async def search_shared_memory(self, query: str, limit: int = 5) -> str:
        """Search shared memory."""
        return await self.shared_memory_repo.search_shared_memory(query, limit)

    # ── Audit log ─────────────────────────────────────────────────────────────

    async def log_event(self, event_type: str, payload: dict) -> None:
        await self.audit.log_event(event_type, payload)

    async def log_task_start(self, task_content: str, source: str, tier: str) -> None:
        """Auto-write a task_start event to the shared audit log."""
        await self.audit.log_task_start(task_content, source, tier)

    async def log_task_done(self, task_content: str, success: bool, elapsed_ms: float, tool_calls: int) -> None:
        """Auto-write a task_done event to the shared audit log."""
        await self.audit.log_task_done(task_content, success, elapsed_ms, tool_calls)

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def heartbeat(self) -> None:
        """Update last_seen timestamp and run cleanup if due."""
        await self.maintenance.heartbeat()

    async def _cleanup(self) -> None:
        """Prune Postgres tables to their retention limits."""
        await self.maintenance.cleanup()

    async def get_stats(self) -> dict:
        """Return row counts for diagnostics."""
        return await self.maintenance.get_stats()

    async def _embed(self, text: str) -> list[float] | None:
        return await _embed(text)
