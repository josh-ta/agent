from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import structlog

from agent.config import settings

if TYPE_CHECKING:
    from agent.memory.postgres_store import PostgresStore

log = structlog.get_logger()
_PG_CLEANUP_INTERVAL_S = 3600


class AgentRegistryRepository:
    def __init__(self, store: "PostgresStore") -> None:
        self._store = store

    async def register_agent(self) -> None:
        assert self._store._pool
        async with self._store._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO agents (id, name, status, model, last_seen)
                   VALUES ($1, $2, 'online', $3, NOW())
                   ON CONFLICT (id) DO UPDATE
                   SET status='online', last_seen=NOW(), model=EXCLUDED.model""",
                settings.agent_name,
                settings.agent_name,
                settings.agent_model,
            )

    async def set_offline(self) -> None:
        assert self._store._pool
        async with self._store._pool.acquire() as conn:
            await conn.execute(
                "UPDATE agents SET status='offline', last_seen=NOW() WHERE id=$1",
                settings.agent_name,
            )

    async def list_agents(self) -> str:
        assert self._store._pool
        async with self._store._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT name, status, model, last_seen FROM agents ORDER BY last_seen DESC"
            )
        if not rows:
            return "(no agents registered)"
        lines = ["Registered agents:", ""]
        for row in rows:
            last = row["last_seen"].strftime("%Y-%m-%d %H:%M") if row["last_seen"] else "never"
            lines.append(f"  • {row['name']} [{row['status']}] model={row['model']} last_seen={last}")
        return "\n".join(lines)


class SharedTaskRepository:
    def __init__(self, store: "PostgresStore") -> None:
        self._store = store

    async def create_task(self, to_agent: str, description: str) -> str:
        assert self._store._pool
        async with self._store._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO shared_tasks (from_agent, to_agent, description)
                   VALUES ($1, $2, $3) RETURNING id""",
                settings.agent_name,
                to_agent,
                description,
            )
        task_id = row["id"] if row else "unknown"
        return f"Task {task_id} created for {to_agent}."

    async def get_my_tasks(self) -> str:
        assert self._store._pool
        async with self._store._pool.acquire() as conn:
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
        for row in rows:
            lines.append(
                f"  [{row['id'][:8]}] from={row['from_agent']} at={row['created_at'].strftime('%Y-%m-%d %H:%M')}: {row['description']}"
            )
        return "\n".join(lines)

    async def get_pending_task_rows(self) -> list[dict]:
        if not self._store._pool:
            return []
        async with self._store._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, from_agent, description
                   FROM shared_tasks
                   WHERE to_agent=$1 AND status='pending'
                   ORDER BY created_at ASC
                   LIMIT 5""",
                settings.agent_name,
            )
        return [dict(row) for row in rows]

    async def mark_task_running(self, task_id: str) -> bool:
        if not self._store._pool:
            return False
        async with self._store._pool.acquire() as conn:
            row = await conn.fetchrow(
                """UPDATE shared_tasks
                   SET status='running', updated_at=NOW()
                   WHERE id=$1 AND status='pending'
                   RETURNING id""",
                task_id,
            )
        return row is not None

    async def complete_task(self, task_id: str, result: str) -> str:
        assert self._store._pool
        async with self._store._pool.acquire() as conn:
            tag = await conn.execute(
                """UPDATE shared_tasks
                   SET status='done', result=$1, updated_at=NOW()
                   WHERE id=$2""",
                result,
                task_id,
            )
        updated = int(tag.split()[-1]) if tag else 0
        return f"Task {task_id[:8]} marked done." if updated else f"Task {task_id[:8]} not found or already completed."


class AuditLogRepository:
    def __init__(self, store: "PostgresStore") -> None:
        self._store = store

    async def broadcast_message(self, message: str, event_type: str = "broadcast") -> str:
        assert self._store._pool
        async with self._store._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO audit_log (agent_id, event_type, payload) VALUES ($1,$2,$3)",
                settings.agent_name,
                event_type,
                json.dumps({"message": message}),
            )
        return f"Broadcast sent [{event_type}]: {message[:80]}"

    async def read_broadcasts(self, limit: int = 20, event_type: str = "broadcast") -> str:
        assert self._store._pool
        async with self._store._pool.acquire() as conn:
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
        for row in rows:
            payload = row["payload"] or {}
            lines.append(
                f"  [{row['ts'].strftime('%Y-%m-%d %H:%M')}] {row['agent_id']}: {payload.get('message', str(payload))}"
            )
        return "\n".join(lines)

    async def log_event(self, event_type: str, payload: dict) -> None:
        assert self._store._pool
        async with self._store._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO audit_log (agent_id, event_type, payload) VALUES ($1,$2,$3)",
                settings.agent_name,
                event_type,
                json.dumps(payload),
            )

    async def log_task_start(self, task_content: str, source: str, tier: str) -> None:
        if not self._store._pool:
            return
        try:
            await self.log_event("task_start", {"content": task_content[:300], "source": source, "tier": tier})
        except Exception as exc:
            log.warning("audit_log_task_start_failed", error=str(exc))

    async def log_task_done(self, task_content: str, success: bool, elapsed_ms: float, tool_calls: int) -> None:
        if not self._store._pool:
            return
        try:
            await self.log_event(
                "task_done",
                {
                    "content": task_content[:300],
                    "success": success,
                    "elapsed_ms": round(elapsed_ms),
                    "tool_calls": tool_calls,
                },
            )
        except Exception as exc:
            log.warning("audit_log_task_done_failed", error=str(exc))


class SharedMemoryRepository:
    def __init__(self, store: "PostgresStore") -> None:
        self._store = store

    async def share_memory(self, content: str, metadata: dict | None = None) -> str:
        assert self._store._pool
        embedding = await self._store._embed(content) if self._store._has_embeddings else None
        async with self._store._pool.acquire() as conn:
            if embedding is not None:
                row = await conn.fetchrow(
                    """INSERT INTO shared_memory (agent_id, content, metadata, embedding)
                       VALUES ($1, $2, $3, $4::vector) RETURNING id""",
                    settings.agent_name,
                    content,
                    json.dumps(metadata or {}),
                    "[" + ",".join(str(v) for v in embedding) + "]",
                )
            else:
                row = await conn.fetchrow(
                    """INSERT INTO shared_memory (agent_id, content, metadata)
                       VALUES ($1, $2, $3) RETURNING id""",
                    settings.agent_name,
                    content,
                    json.dumps(metadata or {}),
                )
        return f"Shared memory saved [{(row['id'][:8] if row else 'unknown')}]: {content[:80]}"

    async def search_shared_memory(self, query: str, limit: int = 5) -> str:
        assert self._store._pool
        async with self._store._pool.acquire() as conn:
            if self._store._has_embeddings:
                query_vec = await self._store._embed(query)
                if query_vec is not None:
                    rows = await conn.fetch(
                        """SELECT agent_id, content, created_at, 1 - (embedding <=> $1::vector) AS similarity
                           FROM shared_memory
                           WHERE embedding IS NOT NULL
                           ORDER BY embedding <=> $1::vector
                           LIMIT $2""",
                        "[" + ",".join(str(v) for v in query_vec) + "]",
                        limit,
                    )
                    if rows:
                        lines = [f"Shared memory — semantic search for '{query}':", ""]
                        for row in rows:
                            lines.append(
                                f"  [{row['created_at'].strftime('%Y-%m-%d %H:%M')}] {row['agent_id']} (sim={row['similarity']:.2f}): {row['content']}"
                            )
                        return "\n".join(lines)
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
        for row in rows:
            lines.append(f"  [{row['created_at'].strftime('%Y-%m-%d %H:%M')}] {row['agent_id']}: {row['content']}")
        return "\n".join(lines)


class PostgresMaintenance:
    def __init__(self, store: "PostgresStore") -> None:
        self._store = store

    async def heartbeat(self) -> None:
        if self._store._pool:
            try:
                async with self._store._pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE agents SET last_seen=NOW() WHERE id=$1",
                        settings.agent_name,
                    )
            except Exception as exc:
                log.warning("postgres_heartbeat_failed", error=str(exc))

        if time.time() - self._store._last_cleanup_ts >= _PG_CLEANUP_INTERVAL_S:
            try:
                await self.cleanup()
            except Exception as exc:
                log.warning("postgres_cleanup_error", error=str(exc))

    async def cleanup(self) -> None:
        if not self._store._pool:
            return
        log.info("postgres_cleanup_start")
        now_ts = time.time()
        async with self._store._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM audit_log WHERE ts < NOW() - ($1 || ' days')::interval",
                str(settings.retention_audit_log_days),
            )
            await conn.execute(
                """DELETE FROM shared_tasks
                   WHERE status IN ('done','failed')
                     AND updated_at < NOW() - ($1 || ' days')::interval""",
                str(settings.retention_shared_tasks_days),
            )
            await conn.execute(
                """DELETE FROM shared_memory
                   WHERE id NOT IN (
                       SELECT id FROM shared_memory
                       WHERE agent_id = $1
                       ORDER BY created_at DESC
                       LIMIT $2
                   ) AND agent_id = $1""",
                settings.agent_name,
                settings.retention_shared_memory_max,
            )
        self._store._last_cleanup_ts = now_ts
        log.info("postgres_cleanup_done")

    async def get_stats(self) -> dict:
        if not self._store._pool:
            return {}
        stats: dict[str, Any] = {}
        async with self._store._pool.acquire() as conn:
            for table in ("agents", "shared_tasks", "audit_log", "shared_memory"):
                row = await conn.fetchrow(f"SELECT COUNT(*) as n FROM {table}")
                stats[table] = row["n"] if row else 0
        stats["last_cleanup"] = (
            time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(self._store._last_cleanup_ts))
            if self._store._last_cleanup_ts
            else "never"
        )
        return stats
