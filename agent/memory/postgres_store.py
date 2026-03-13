"""
PostgreSQL store: shared state for multi-agent coordination.

Optional — only used when POSTGRES_URL is set.
Uses asyncpg for async, non-blocking access.
Schema is created by scripts/postgres-init.sql on first startup.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import asyncpg
import structlog

from agent.config import settings

log = structlog.get_logger()


class PostgresStore:
    def __init__(self, url: str) -> None:
        self._url = url
        self._pool: asyncpg.Pool | None = None  # type: ignore[type-arg]

    async def init(self) -> None:
        """Create connection pool."""
        self._pool = await asyncpg.create_pool(
            self._url,
            min_size=1,
            max_size=5,
            command_timeout=10,
        )
        log.info("postgres_connected")

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

    async def complete_task(self, task_id: str, result: str) -> None:
        assert self._pool
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE shared_tasks
                   SET status='done', result=$1, updated_at=NOW()
                   WHERE id=$2""",
                result,
                task_id,
            )

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
