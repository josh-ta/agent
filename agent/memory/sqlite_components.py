from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import structlog

from agent.config import settings

if TYPE_CHECKING:
    from agent.loop import Task, TaskResult
    from agent.memory.sqlite_store import SQLiteStore

log = structlog.get_logger()

_CLEANUP_INTERVAL_S = 3600
_VACUUM_INTERVAL_S = 30 * 24 * 3600


class SQLiteConversationRepository:
    def __init__(self, store: "SQLiteStore") -> None:
        self._store = store

    async def save_message(
        self,
        role: str,
        content: str,
        channel_id: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._store._check()
        assert self._store._db
        await self._store._db.execute(
            "INSERT INTO conversations (channel_id, role, content, metadata, ts) VALUES (?,?,?,?,?)",
            (channel_id, role, content, json.dumps(metadata or {}), time.time()),
        )
        await self._store._db.commit()

    async def get_history(self, channel_id: int = 0, limit: int = 20) -> list[dict[str, Any]]:
        self._store._check()
        assert self._store._db
        async with self._store._db.execute(
            "SELECT role, content, ts FROM conversations WHERE channel_id=? ORDER BY ts DESC LIMIT ?",
            (channel_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in reversed(rows)]


class SQLiteTaskRepository:
    def __init__(self, store: "SQLiteStore") -> None:
        self._store = store

    async def migrate(self) -> None:
        self._store._check()
        assert self._store._db
        async with self._store._db.execute("PRAGMA table_info(tasks)") as cur:
            columns = {row["name"] for row in await cur.fetchall()}

        migrations: list[tuple[str, tuple[Any, ...] | None]] = []
        if "task_id" not in columns:
            migrations.append(("ALTER TABLE tasks ADD COLUMN task_id TEXT", None))
        if "status" not in columns:
            migrations.append(("ALTER TABLE tasks ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'", None))
        if "metadata" not in columns:
            migrations.append(("ALTER TABLE tasks ADD COLUMN metadata TEXT DEFAULT '{}'", None))
        if "error" not in columns:
            migrations.append(("ALTER TABLE tasks ADD COLUMN error TEXT", None))
        if "created_ts" not in columns:
            migrations.append(("ALTER TABLE tasks ADD COLUMN created_ts REAL", None))
        if "started_ts" not in columns:
            migrations.append(("ALTER TABLE tasks ADD COLUMN started_ts REAL", None))
        if "finished_ts" not in columns:
            migrations.append(("ALTER TABLE tasks ADD COLUMN finished_ts REAL", None))
        if "updated_ts" not in columns:
            migrations.append(("ALTER TABLE tasks ADD COLUMN updated_ts REAL", None))

        for sql, params in migrations:
            assert params is None
            await self._store._db.execute(sql)

        await self._store._db.execute(
            """
            UPDATE tasks
            SET
                created_ts = COALESCE(created_ts, ts),
                updated_ts = COALESCE(updated_ts, ts),
                metadata = COALESCE(metadata, '{}'),
                finished_ts = CASE
                    WHEN status IN ('succeeded', 'failed') THEN COALESCE(finished_ts, ts)
                    ELSE finished_ts
                END,
                status = CASE
                    WHEN status IS NULL OR status = '' OR (task_id IS NULL AND status = 'completed') THEN
                        CASE
                            WHEN success = 1 THEN 'succeeded'
                            ELSE 'failed'
                        END
                    ELSE status
                END,
                error = CASE
                    WHEN success = 0 AND result IS NOT NULL AND (error IS NULL OR error = '') THEN result
                    ELSE error
                END
            """
        )
        await self._store._db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS tasks_task_id_idx ON tasks (task_id)"
        )
        await self._store._db.commit()

    async def create_task_record(
        self,
        *,
        task_id: str,
        source: str,
        author: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._store._check()
        assert self._store._db
        now = time.time()
        await self._store._db.execute(
            """INSERT INTO tasks
               (task_id, source, author, content, status, metadata, success, created_ts, updated_ts, ts)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (task_id, source, author, content, "queued", json.dumps(metadata or {}), 0, now, now, now),
        )
        await self._store._db.commit()

    async def mark_task_running(self, task_id: str) -> None:
        self._store._check()
        assert self._store._db
        now = time.time()
        await self._store._db.execute(
            """UPDATE tasks
               SET status='running',
                   started_ts=COALESCE(started_ts, ?),
                   updated_ts=?
               WHERE task_id=?""",
            (now, now, task_id),
        )
        await self._store._db.commit()

    async def record_task(self, task: "Task", result: "TaskResult") -> None:
        self._store._check()
        assert self._store._db
        task_id = str(task.metadata.get("task_id", "")).strip() if task.metadata else ""
        now = time.time()
        if task_id:
            await self._store._db.execute(
                """UPDATE tasks
                   SET source=?,
                       author=?,
                       content=?,
                       status=?,
                       metadata=?,
                       result=?,
                       error=?,
                       success=?,
                       elapsed_ms=?,
                       tool_calls=?,
                       started_ts=COALESCE(started_ts, ?),
                       finished_ts=?,
                       updated_ts=?
                   WHERE task_id=?""",
                (
                    task.source,
                    task.author,
                    task.content,
                    result.status,
                    json.dumps(task.metadata or {}),
                    result.output if result.status == "succeeded" else None,
                    result.output if result.status == "failed" else None,
                    1 if result.status == "succeeded" else 0,
                    result.elapsed_ms,
                    result.tool_calls,
                    now,
                    now if result.status in {"succeeded", "failed"} else None,
                    now,
                    task_id,
                ),
            )
        else:
            await self._store._db.execute(
                """INSERT INTO tasks
                   (source, author, content, status, metadata, result, error, success, elapsed_ms, tool_calls, created_ts, finished_ts, updated_ts, ts)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    task.source,
                    task.author,
                    task.content,
                    result.status,
                    json.dumps(task.metadata or {}),
                    result.output if result.status == "succeeded" else None,
                    result.output if result.status == "failed" else None,
                    1 if result.status == "succeeded" else 0,
                    result.elapsed_ms,
                    result.tool_calls,
                    now,
                    now if result.status in {"succeeded", "failed"} else None,
                    now,
                    now,
                ),
            )
        await self._store._db.commit()

    async def mark_task_waiting(self, task_id: str, *, metadata: dict[str, Any], question: str) -> None:
        self._store._check()
        assert self._store._db
        now = time.time()
        await self._store._db.execute(
            """UPDATE tasks
               SET status='waiting_for_user',
                   metadata=?,
                   result=NULL,
                   error=?,
                   success=0,
                   finished_ts=NULL,
                   updated_ts=?
               WHERE task_id=?""",
            (json.dumps(metadata), question, now, task_id),
        )
        await self._store._db.commit()

    async def mark_task_queued(self, task_id: str, *, metadata: dict[str, Any] | None = None) -> None:
        self._store._check()
        assert self._store._db
        now = time.time()
        if metadata is None:
            await self._store._db.execute(
                """UPDATE tasks
                   SET status='queued',
                       updated_ts=?
                   WHERE task_id=?""",
                (now, task_id),
            )
        else:
            await self._store._db.execute(
                """UPDATE tasks
                   SET status='queued',
                       metadata=?,
                       updated_ts=?
                   WHERE task_id=?""",
                (json.dumps(metadata), now, task_id),
            )
        await self._store._db.commit()

    async def get_task_record(self, task_id: str) -> dict[str, Any] | None:
        self._store._check()
        assert self._store._db
        async with self._store._db.execute(
            """SELECT task_id, source, author, content, status, result, error, success,
                      metadata, elapsed_ms, tool_calls, created_ts, started_ts, finished_ts, updated_ts
               FROM tasks
               WHERE task_id=?
               LIMIT 1""",
            (task_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        data = dict(row)
        try:
            data["metadata"] = json.loads(data.get("metadata") or "{}")
        except Exception:
            data["metadata"] = {}
        return data

    async def list_waiting_task_records(self) -> list[dict[str, Any]]:
        self._store._check()
        assert self._store._db
        async with self._store._db.execute(
            """SELECT task_id, source, author, content, status, result, error, success,
                      metadata, elapsed_ms, tool_calls, created_ts, started_ts, finished_ts, updated_ts
               FROM tasks
               WHERE status='waiting_for_user'
               ORDER BY updated_ts ASC, created_ts ASC"""
        ) as cur:
            rows = await cur.fetchall()

        records: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            try:
                data["metadata"] = json.loads(data.get("metadata") or "{}")
            except Exception:
                data["metadata"] = {}
            records.append(data)
        return records


class SQLiteMemoryRepository:
    def __init__(self, store: "SQLiteStore") -> None:
        self._store = store

    async def save_memory_fact(self, content: str, metadata: dict[str, Any] | None = None) -> int:
        self._store._check()
        assert self._store._db
        db = self._store._db
        cur = await self._store._db.execute(
            "INSERT INTO memory_facts (content, metadata, ts) VALUES (?,?,?)",
            (content, json.dumps(metadata or {}), time.time()),
        )
        await self._store._db.commit()
        fact_id = cur.lastrowid or 0

        if self._store._has_vec and fact_id:
            try:
                embedding = await embed_text(content)
                if embedding is not None:
                    await db.execute(
                        "INSERT OR REPLACE INTO memory_vec(fact_id, embedding) VALUES (?, ?)",
                        (fact_id, json.dumps(embedding)),
                    )
                    await db.commit()
            except Exception as exc:
                try:
                    await db.rollback()
                except Exception as rollback_exc:
                    log.debug("memory_vec_insert_rollback_failed", error=str(rollback_exc))
                log.warning("memory_vec_insert_failed", error=str(exc))

        return fact_id

    async def search_memory(self, query: str, limit: int = 5) -> str:
        self._store._check()
        assert self._store._db

        if self._store._has_vec:
            try:
                embedding = await embed_text(query)
                if embedding is not None:
                    async with self._store._db.execute(
                        """SELECT mf.content, mf.ts, mv.distance
                           FROM memory_vec mv
                           JOIN memory_facts mf ON mv.fact_id = mf.id
                           WHERE mv.embedding MATCH ?
                             AND k = ?
                           ORDER BY mv.distance""",
                        (json.dumps(embedding), limit),
                    ) as cur:
                        rows = await cur.fetchall()
                    if rows:
                        import datetime

                        return "\n".join(
                            f"[{datetime.datetime.fromtimestamp(row['ts']).strftime('%Y-%m-%d')}] {row['content']}"
                            for row in rows
                        )
            except Exception as exc:
                log.debug("memory_vec_search_failed", error=str(exc))

        safe_query = query.replace('"', '""')
        try:
            async with self._store._db.execute(
                """SELECT mf.content, mf.ts, rank
                   FROM memory_fts
                   JOIN memory_facts mf ON memory_fts.rowid = mf.id
                   WHERE memory_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (f'"{safe_query}"', limit),
            ) as cur:
                rows = await cur.fetchall()
        except Exception:
            async with self._store._db.execute(
                "SELECT content, ts, 0 as rank FROM memory_facts WHERE content LIKE ? ORDER BY ts DESC LIMIT ?",
                (f"%{query}%", limit),
            ) as cur:
                rows = await cur.fetchall()

        if not rows:
            return f"(no memory matches for: {query})"

        import datetime

        return "\n".join(
            f"[{datetime.datetime.fromtimestamp(row['ts']).strftime('%Y-%m-%d')}] {row['content']}"
            for row in rows
        )


class SQLiteLessonRepository:
    def __init__(self, store: "SQLiteStore") -> None:
        self._store = store

    async def save_lesson(self, summary: str, kind: str = "lesson", context: str = "") -> int:
        self._store._check()
        assert self._store._db
        cur = await self._store._db.execute(
            "INSERT INTO lessons (kind, summary, context, ts) VALUES (?,?,?,?)",
            (kind, summary, context, time.time()),
        )
        await self._store._db.commit()
        return cur.lastrowid or 0

    async def search_lessons(self, query: str, limit: int = 5) -> str:
        self._store._check()
        assert self._store._db
        safe_query = query.replace('"', '""')
        try:
            async with self._store._db.execute(
                """SELECT l.id, l.kind, l.summary, l.ts
                   FROM lessons_fts
                   JOIN lessons l ON lessons_fts.rowid = l.id
                   WHERE lessons_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (f'"{safe_query}"', limit),
            ) as cur:
                rows = await cur.fetchall()
        except Exception:
            async with self._store._db.execute(
                "SELECT id, kind, summary, ts FROM lessons WHERE summary LIKE ? ORDER BY ts DESC LIMIT ?",
                (f"%{query}%", limit),
            ) as cur:
                rows = await cur.fetchall()

        if not rows:
            return ""

        matched_ids = [row["id"] for row in rows]
        placeholders = ",".join("?" * len(matched_ids))
        await self._store._db.execute(
            f"UPDATE lessons SET applied=applied+1 WHERE id IN ({placeholders})",
            matched_ids,
        )
        await self._store._db.commit()

        import datetime

        lines = ["## Relevant past lessons:"]
        for row in rows:
            lines.append(
                f"- [{row['kind'].upper()} {datetime.datetime.fromtimestamp(row['ts']).strftime('%Y-%m-%d')}] {row['summary']}"
            )
            if len("\n".join(lines)) > 1200:
                break
        return "\n".join(lines)

    async def get_recent_lessons(self, limit: int = 20) -> str:
        self._store._check()
        assert self._store._db
        async with self._store._db.execute(
            "SELECT kind, summary, ts FROM lessons ORDER BY ts DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        if not rows:
            return "(no lessons recorded yet)"

        import datetime

        return "\n".join(
            f"- [{row['kind'].upper()} {datetime.datetime.fromtimestamp(row['ts']).strftime('%Y-%m-%d')}] {row['summary']}"
            for row in rows
        )


class SQLiteMaintenance:
    def __init__(self, store: "SQLiteStore") -> None:
        self._store = store

    async def get_stats(self) -> dict[str, Any]:
        self._store._check()
        assert self._store._db
        stats: dict[str, Any] = {}
        for table in ("conversations", "tasks", "memory_facts", "lessons"):
            async with self._store._db.execute(f"SELECT COUNT(*) as n FROM {table}") as cur:
                row = await cur.fetchone()
                stats[table] = row["n"] if row else 0
        try:
            stats["db_size_mb"] = round(self._store._path.stat().st_size / 1_048_576, 2)
        except Exception:
            stats["db_size_mb"] = "unknown"
        stats["last_cleanup"] = (
            time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(self._store._last_cleanup_ts))
            if self._store._last_cleanup_ts
            else "never"
        )
        stats["vec_enabled"] = self._store._has_vec
        return stats

    async def cleanup(self) -> None:
        self._store._check()
        assert self._store._db
        now = time.time()
        log.info("sqlite_cleanup_start")

        await self._store._db.execute(
            "DELETE FROM conversations WHERE ts < ?",
            (now - settings.retention_conversations_days * 86400,),
        )
        async with self._store._db.execute("SELECT DISTINCT channel_id FROM conversations") as cur:
            channels = [row[0] for row in await cur.fetchall()]
        for channel_id in channels:
            await self._store._db.execute(
                """DELETE FROM conversations WHERE channel_id=? AND id NOT IN (
                       SELECT id FROM conversations WHERE channel_id=? ORDER BY ts DESC LIMIT 1000
                   )""",
                (channel_id, channel_id),
            )

        await self._store._db.execute(
            "DELETE FROM tasks WHERE ts < ?",
            (now - settings.retention_tasks_days * 86400,),
        )
        await self._store._db.execute(
            """DELETE FROM memory_facts WHERE id NOT IN (
                   SELECT id FROM memory_facts ORDER BY ts DESC LIMIT ?
               )""",
            (settings.retention_memory_facts_max,),
        )
        if self._store._has_vec:
            await self._store._db.execute(
                "DELETE FROM memory_vec WHERE fact_id NOT IN (SELECT id FROM memory_facts)"
            )

        await self._store._db.execute(
            """DELETE FROM lessons WHERE id NOT IN (
                   SELECT id FROM lessons ORDER BY applied DESC, ts DESC LIMIT ?
               )""",
            (settings.retention_lessons_max,),
        )
        await self._store._db.commit()

        try:
            await self._store._db.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
            await self._store._db.execute("INSERT INTO lessons_fts(lessons_fts) VALUES ('rebuild')")
            await self._store._db.commit()
        except Exception as exc:
            try:
                await self._store._db.rollback()
            except Exception as rollback_exc:
                log.debug("fts_rebuild_rollback_failed", error=str(rollback_exc))
            log.warning("fts_rebuild_failed", error=str(exc))

        await self._store._db.execute("PRAGMA wal_checkpoint(PASSIVE)")

        async with self._store._db.execute("SELECT value FROM _meta WHERE key='last_vacuum_ts'") as cur:
            row = await cur.fetchone()
        last_vacuum = float(row[0]) if row else 0.0
        if now - last_vacuum > _VACUUM_INTERVAL_S:
            await self._store._db.execute("VACUUM")
            await self._store._db.execute(
                "INSERT OR REPLACE INTO _meta(key,value) VALUES('last_vacuum_ts',?)",
                (str(now),),
            )
            await self._store._db.commit()
            log.info("sqlite_vacuumed")

        self._store._last_cleanup_ts = now
        await self._store._db.execute(
            "INSERT OR REPLACE INTO _meta(key,value) VALUES('last_cleanup_ts',?)",
            (str(now),),
        )
        await self._store._db.commit()
        log.info("sqlite_cleanup_done")

    async def heartbeat(self) -> None:
        if not self._store._db:
            return
        await self._store._db.execute("PRAGMA wal_checkpoint(PASSIVE)")
        if time.time() - self._store._last_cleanup_ts >= _CLEANUP_INTERVAL_S:
            try:
                await self.cleanup()
            except Exception as exc:
                log.warning("sqlite_cleanup_error", error=str(exc))


async def embed_text(text: str) -> list[float] | None:
    if not settings.has_embeddings:
        return None
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        resp = await client.embeddings.create(
            model=settings.embedding_model,
            input=text,
            encoding_format="float",
        )
        return resp.data[0].embedding
    except Exception as exc:
        log.warning("sqlite_embed_failed", error=str(exc))
        return None
