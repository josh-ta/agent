"""
PostgreSQL store: shared state for multi-agent coordination.

Optional — only used when POSTGRES_URL is set.
Uses asyncpg for async, non-blocking access.
Tables are created automatically on init() via CREATE TABLE IF NOT EXISTS.
pgvector/pg_trgm extensions are attempted but not required.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import asyncpg
import structlog

from agent.config import settings

log = structlog.get_logger()

# ── Schema ────────────────────────────────────────────────────────────────────
# pgvector / pg_trgm are optional; we try to enable them but continue without.
# shared_memory stores content as text; vector search uses ILIKE fallback.

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


class PostgresStore:
    def __init__(self, url: str) -> None:
        # asyncpg uses postgres:// or postgresql:// — strip the +asyncpg driver hint
        self._url = url.replace("postgresql+asyncpg://", "postgresql://").replace(
            "postgres+asyncpg://", "postgresql://"
        )
        self._pool: asyncpg.Pool | None = None  # type: ignore[type-arg]
        self._has_vector: bool = False

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
                except Exception as exc:
                    log.warning("postgres_vector_index_skipped", reason=str(exc))

        log.info("postgres_schema_ready", vector=self._has_vector)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    # ── Agent registry ────────────────────────────────────────────────────────

    async def register_agent(self) -> None:
        """Upsert this agent's presence in the agents table."""
        assert self._pool
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO agents (id, name, status, model, last_seen)
                   VALUES ($1, $2, 'online', $3, NOW())
                   ON CONFLICT (id) DO UPDATE
                   SET status='online', last_seen=NOW(), model=EXCLUDED.model""",
                settings.agent_name,
                settings.agent_name,
                settings.agent_model,
            )
        log.info("agent_registered", name=settings.agent_name)

    async def set_offline(self) -> None:
        assert self._pool
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE agents SET status='offline', last_seen=NOW() WHERE id=$1",
                settings.agent_name,
            )

    async def list_agents(self) -> str:
        """Return a formatted list of all registered agents."""
        assert self._pool
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT name, status, model, last_seen FROM agents ORDER BY last_seen DESC"
            )
        if not rows:
            return "(no agents registered)"
        lines = ["Registered agents:", ""]
        for r in rows:
            last = r["last_seen"].strftime("%Y-%m-%d %H:%M") if r["last_seen"] else "never"
            lines.append(f"  • {r['name']} [{r['status']}] model={r['model']} last_seen={last}")
        return "\n".join(lines)

    # ── Shared task queue ─────────────────────────────────────────────────────

    async def create_task(self, to_agent: str, description: str) -> str:
        """Create a task for another agent."""
        assert self._pool
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO shared_tasks (from_agent, to_agent, description)
                   VALUES ($1, $2, $3) RETURNING id""",
                settings.agent_name,
                to_agent,
                description,
            )
        task_id = row["id"] if row else "unknown"
        log.info("shared_task_created", to=to_agent, id=task_id)
        return f"Task {task_id} created for {to_agent}."

    async def get_my_tasks(self) -> str:
        """Get pending tasks assigned to this agent."""
        assert self._pool
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, from_agent, description, status, created_at
                   FROM shared_tasks
                   WHERE to_agent=$1 AND status='pending'
                   ORDER BY created_at ASC""",
                settings.agent_name,
            )
        if not rows:
            return "(no pending tasks)"
        lines = ["My pending tasks:", ""]
        for r in rows:
            ts = r["created_at"].strftime("%Y-%m-%d %H:%M")
            lines.append(f"  [{r['id'][:8]}] from={r['from_agent']} at={ts}: {r['description']}")
        return "\n".join(lines)

    async def complete_task(self, task_id: str, result: str) -> str:
        """Mark a shared task as done with a result."""
        assert self._pool
        async with self._pool.acquire() as conn:
            tag = await conn.execute(
                """UPDATE shared_tasks
                   SET status='done', result=$1, updated_at=NOW()
                   WHERE id=$2""",
                result,
                task_id,
            )
        updated = int(tag.split()[-1]) if tag else 0
        if updated == 0:
            return f"Task {task_id[:8]} not found or already completed."
        log.info("shared_task_completed", id=task_id)
        return f"Task {task_id[:8]} marked done."

    # ── Broadcasts (audit_log) ────────────────────────────────────────────────

    async def broadcast_message(self, message: str, event_type: str = "broadcast") -> str:
        """Write a broadcast message to the audit log, visible to all agents."""
        assert self._pool
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO audit_log (agent_id, event_type, payload) VALUES ($1,$2,$3)",
                settings.agent_name,
                event_type,
                json.dumps({"message": message}),
            )
        return f"Broadcast sent [{event_type}]: {message[:80]}"

    async def read_broadcasts(self, limit: int = 20, event_type: str = "broadcast") -> str:
        """Read recent broadcast messages from the audit log."""
        assert self._pool
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT agent_id, payload, ts FROM audit_log
                   WHERE event_type=$1
                   ORDER BY ts DESC LIMIT $2""",
                event_type,
                limit,
            )
        if not rows:
            return "(no broadcasts)"
        lines = [f"Recent broadcasts (last {limit}):", ""]
        for r in rows:
            ts = r["ts"].strftime("%Y-%m-%d %H:%M")
            payload = r["payload"] or {}
            msg = payload.get("message", str(payload))
            lines.append(f"  [{ts}] {r['agent_id']}: {msg}")
        return "\n".join(lines)

    # ── Shared memory ─────────────────────────────────────────────────────────

    async def share_memory(self, content: str, metadata: dict | None = None) -> str:
        """Write a fact to shared memory — visible to all agents."""
        assert self._pool
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO shared_memory (agent_id, content, metadata)
                   VALUES ($1, $2, $3) RETURNING id""",
                settings.agent_name,
                content,
                json.dumps(metadata or {}),
            )
        mem_id = row["id"][:8] if row else "unknown"
        return f"Shared memory saved [{mem_id}]: {content[:80]}"

    async def search_shared_memory(self, query: str, limit: int = 5) -> str:
        """Search shared memory by keyword (case-insensitive)."""
        assert self._pool
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT agent_id, content, created_at FROM shared_memory
                   WHERE content ILIKE $1
                   ORDER BY created_at DESC LIMIT $2""",
                f"%{query}%",
                limit,
            )
        if not rows:
            return f"(no shared memory matches for: {query})"
        lines = [f"Shared memory results for '{query}':", ""]
        for r in rows:
            ts = r["created_at"].strftime("%Y-%m-%d %H:%M")
            lines.append(f"  [{ts}] {r['agent_id']}: {r['content']}")
        return "\n".join(lines)

    # ── Audit log ─────────────────────────────────────────────────────────────

    async def log_event(self, event_type: str, payload: dict) -> None:
        assert self._pool
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO audit_log (agent_id, event_type, payload) VALUES ($1,$2,$3)",
                settings.agent_name,
                event_type,
                json.dumps(payload),
            )

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def heartbeat(self) -> None:
        """Update last_seen timestamp."""
        if self._pool:
            try:
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE agents SET last_seen=NOW() WHERE id=$1",
                        settings.agent_name,
                    )
            except Exception as exc:
                log.warning("postgres_heartbeat_failed", error=str(exc))
