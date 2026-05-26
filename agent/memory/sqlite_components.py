from __future__ import annotations

import json
import re
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


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _json_load(value: str | None, default: Any) -> Any:
    try:
        return json.loads(value or "")
    except Exception:  # pragma: no cover - defensive parse fallback
        return default


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def _fts_query(value: str) -> str:
    tokens = [token for token in re.findall(r"[A-Za-z0-9_./:-]+", value.lower()) if len(token) >= 3]
    if not tokens:  # pragma: no cover - fallback for punctuation-only input
        escaped = value.replace('"', '""').strip()
        return f'"{escaped}"' if escaped else '""'
    return " OR ".join(f'{token.replace("\"", "\"\"")}*' for token in tokens[:8])


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
        if "input_tokens_est" not in columns:
            migrations.append(("ALTER TABLE tasks ADD COLUMN input_tokens_est INTEGER", None))
        if "output_tokens_est" not in columns:
            migrations.append(("ALTER TABLE tasks ADD COLUMN output_tokens_est INTEGER", None))

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
            in_tok = getattr(result, "input_tokens_est", None)
            out_tok = getattr(result, "output_tokens_est", None)
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
                       updated_ts=?,
                       input_tokens_est=?,
                       output_tokens_est=?
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
                    in_tok if in_tok is not None else None,
                    out_tok if out_tok is not None else None,
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

        if fact_id:
            source = str((metadata or {}).get("source", "task"))
            scope = str((metadata or {}).get("scope", "task"))
            await self._store.learning.save_memory_item(
                kind="fact",
                content=content,
                scope=scope,
                source=source,
                confidence=0.72,
                salience=0.6,
                pinned=bool((metadata or {}).get("pinned", False)),
                sensitive=bool((metadata or {}).get("sensitive", False)),
                metadata=metadata,
            )

        return fact_id

    async def search_memory(self, query: str, limit: int = 5) -> str:
        self._store._check()
        assert self._store._db

        ranked_items = await self._store.learning.search_memory_items(query, limit=limit)
        if ranked_items:
            import datetime

            return "\n".join(
                f"[{datetime.datetime.fromtimestamp(float(row['ts'])).strftime('%Y-%m-%d')}] {row['content']}"
                for row in ranked_items
            )

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
            except Exception as exc:  # pragma: no cover - defensive vec fallback
                log.debug("memory_vec_search_failed", error=str(exc))

        safe_query = _fts_query(query)
        try:
            async with self._store._db.execute(
                """SELECT mf.content, mf.ts, rank
                   FROM memory_fts
                   JOIN memory_facts mf ON memory_fts.rowid = mf.id
                   WHERE memory_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (safe_query, limit),
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


class SQLiteLearningRepository:
    def __init__(self, store: "SQLiteStore") -> None:
        self._store = store

    async def save_memory_item(
        self,
        *,
        kind: str,
        content: str,
        scope: str = "task",
        source: str = "agent",
        confidence: float = 0.5,
        salience: float = 0.5,
        pinned: bool = False,
        sensitive: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        self._store._check()
        assert self._store._db
        normalized = _normalize_text(content)
        if not normalized:  # pragma: no cover - trivial empty-input guard
            return 0
        now = time.time()
        meta = metadata or {}
        async with self._store._db.execute(
            """SELECT id, confidence, salience, success_credit, failure_credit, use_count, pinned, sensitive
               FROM memory_items
               WHERE kind=? AND content=?
               LIMIT 1""",
            (kind, normalized),
        ) as cur:
            existing = await cur.fetchone()
        if existing is not None:
            await self._store._db.execute(
                """UPDATE memory_items
                   SET scope=?,
                       source=?,
                       confidence=?,
                       salience=?,
                       pinned=?,
                       sensitive=?,
                       metadata=?,
                       updated_ts=?
                   WHERE id=?""",
                (
                    scope,
                    source,
                    max(float(existing["confidence"]), _clamp(confidence)),
                    max(float(existing["salience"]), _clamp(salience)),
                    1 if pinned or int(existing["pinned"]) else 0,
                    1 if sensitive or int(existing["sensitive"]) else 0,
                    json.dumps(meta),
                    now,
                    int(existing["id"]),
                ),
            )
            await self._store._db.commit()
            return int(existing["id"])
        cur = await self._store._db.execute(
            """INSERT INTO memory_items
               (kind, scope, content, source, confidence, salience, pinned, sensitive, metadata, ts, updated_ts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                kind,
                scope,
                normalized,
                source,
                _clamp(confidence),
                _clamp(salience),
                1 if pinned else 0,
                1 if sensitive else 0,
                json.dumps(meta),
                now,
                now,
            ),
        )
        await self._store._db.commit()
        return cur.lastrowid or 0

    async def search_memory_items(
        self,
        query: str,
        *,
        limit: int = 5,
        kinds: tuple[str, ...] | None = None,
        include_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        self._store._check()
        assert self._store._db
        filters = ["mi.sensitive=0"] if not include_sensitive else []
        params: list[Any] = []
        if kinds:
            filters.append(f"mi.kind IN ({','.join('?' * len(kinds))})")
            params.extend(kinds)
        where_clause = ""
        if filters:
            where_clause = "WHERE " + " AND ".join(filters) + " AND "
        else:
            where_clause = "WHERE "
        safe_query = _fts_query(query)
        sql = (
            """SELECT mi.id, mi.kind, mi.content, mi.source, mi.salience, mi.success_credit,
                      mi.failure_credit, mi.use_count, mi.ts, mi.metadata, rank
               FROM memory_items_fts
               JOIN memory_items mi ON memory_items_fts.rowid = mi.id
            """
            + where_clause
            + """
               memory_items_fts MATCH ?
               ORDER BY rank,
                        (mi.pinned * 10 + mi.salience + mi.success_credit - mi.failure_credit) DESC,
                        mi.updated_ts DESC
               LIMIT ?"""
        )
        rows: list[Any]
        try:
            async with self._store._db.execute(sql, (*params, safe_query, limit)) as cur:
                rows = await cur.fetchall()
        except Exception:
            fallback = (
                "SELECT id, kind, content, source, salience, success_credit, failure_credit, use_count, ts, metadata, 0 as rank "
                "FROM memory_items "
                + ("WHERE " + " AND ".join(filter.replace("mi.", "") for filter in filters) if filters else "")
            )
            fallback += (" AND " if filters else " WHERE ") + "content LIKE ? "
            fallback += (
                "ORDER BY (pinned * 10 + salience + success_credit - failure_credit) DESC, updated_ts DESC LIMIT ?"
            )
            async with self._store._db.execute(fallback, (*params, f"%{query}%", limit)) as cur:
                rows = await cur.fetchall()
        items = [dict(row) for row in rows]
        for item in items:
            item["metadata"] = _json_load(item.get("metadata"), {})
        if items:
            ids = [int(item["id"]) for item in items]
            placeholders = ",".join("?" * len(ids))
            now = time.time()
            await self._store._db.execute(
                f"""UPDATE memory_items
                    SET use_count=use_count+1,
                        last_used_ts=?,
                        salience=MIN(1.0, salience + 0.02)
                    WHERE id IN ({placeholders})""",
                (now, *ids),
            )
            await self._store._db.commit()
        return items

    async def record_episodic_event(
        self,
        *,
        task_id: str = "",
        session_id: str = "",
        event_kind: str,
        summary: str,
        reward: float = 0.0,
        details: dict[str, Any] | None = None,
    ) -> int:
        self._store._check()
        assert self._store._db
        cur = await self._store._db.execute(
            """INSERT INTO episodic_events (task_id, session_id, event_kind, summary, reward, details, ts)
               VALUES (?,?,?,?,?,?,?)""",
            (task_id, session_id, event_kind, summary, reward, json.dumps(details or {}), time.time()),
        )
        await self._store._db.commit()
        return cur.lastrowid or 0

    async def search_learning_context(self, query: str, limit: int = 3) -> dict[str, Any]:
        facts = await self.search_memory_items(
            query,
            limit=limit,
            kinds=("fact", "insight", "preference", "resource", "pattern", "lesson", "mistake", "warning"),
        )
        pitfalls = [item for item in facts if item["kind"] in {"mistake", "warning"}]
        facts_only = [item for item in facts if item["kind"] not in {"mistake", "warning"}]
        procedures = await self._store.procedures.search_procedures(query, limit=limit)

        sections: list[str] = []
        if facts_only:
            sections.append(
                "## Relevant facts\n"
                + "\n".join(f"- {item['content']}" for item in facts_only[:limit])
            )
        if pitfalls:
            sections.append(
                "## Known pitfalls\n"
                + "\n".join(f"- {item['content']}" for item in pitfalls[:limit])
            )
        if procedures:
            sections.append(
                "## Preferred procedures\n"
                + "\n".join(
                    f"- Trigger: {item['trigger_text']} | Checklist: {item['checklist']}"
                    for item in procedures[:limit]
                )
            )
        return {
            "text": "\n\n".join(sections),
            "memory_ids": [int(item["id"]) for item in facts[:limit]],
            "procedure_ids": [int(item["id"]) for item in procedures[:limit]],
        }

    async def pin_memory_item(self, memory_item_id: int, *, pinned: bool = True) -> None:
        self._store._check()
        assert self._store._db
        await self._store._db.execute(
            "UPDATE memory_items SET pinned=?, updated_ts=? WHERE id=?",
            (1 if pinned else 0, time.time(), memory_item_id),
        )
        await self._store._db.commit()

    async def apply_outcome_to_memory(
        self,
        memory_ids: list[int],
        *,
        success_delta: float = 0.0,
        failure_delta: float = 0.0,
    ) -> None:
        self._store._check()
        assert self._store._db
        if not memory_ids:  # pragma: no cover - trivial empty-input guard
            return
        placeholders = ",".join("?" * len(memory_ids))
        await self._store._db.execute(
            f"""UPDATE memory_items
                SET success_credit=success_credit+?,
                    failure_credit=failure_credit+?,
                    salience=MIN(1.0, MAX(0.0, salience + ? - ?)),
                    updated_ts=?
                WHERE id IN ({placeholders})""",
            (
                success_delta,
                failure_delta,
                success_delta * 0.05,
                failure_delta * 0.05,
                time.time(),
                *memory_ids,
            ),
        )
        await self._store._db.commit()


class SQLiteProcedureRepository:
    def __init__(self, store: "SQLiteStore") -> None:
        self._store = store

    async def save_procedure(
        self,
        *,
        trigger_text: str,
        checklist: str,
        kind: str = "procedure",
        confidence: float = 0.5,
        salience: float = 0.5,
        pinned: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        self._store._check()
        assert self._store._db
        trigger = _normalize_text(trigger_text)
        steps = _normalize_text(checklist)
        if not trigger or not steps:  # pragma: no cover - trivial empty-input guard
            return 0
        now = time.time()
        async with self._store._db.execute(
            "SELECT id, confidence, salience, pinned FROM procedures WHERE trigger_text=? AND checklist=? LIMIT 1",
            (trigger, steps),
        ) as cur:
            existing = await cur.fetchone()
        if existing is not None:
            await self._store._db.execute(
                """UPDATE procedures
                   SET confidence=?,
                       salience=?,
                       pinned=?,
                       metadata=?,
                       updated_ts=?
                   WHERE id=?""",
                (
                    max(float(existing["confidence"]), _clamp(confidence)),
                    max(float(existing["salience"]), _clamp(salience)),
                    1 if pinned or int(existing["pinned"]) else 0,
                    json.dumps(metadata or {}),
                    now,
                    int(existing["id"]),
                ),
            )
            await self._store._db.commit()
            return int(existing["id"])
        cur = await self._store._db.execute(
            """INSERT INTO procedures
               (kind, trigger_text, checklist, confidence, salience, pinned, metadata, ts, updated_ts)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                kind,
                trigger,
                steps,
                _clamp(confidence),
                _clamp(salience),
                1 if pinned else 0,
                json.dumps(metadata or {}),
                now,
                now,
            ),
        )
        await self._store._db.commit()
        return cur.lastrowid or 0

    async def search_procedures(self, query: str, *, limit: int = 3) -> list[dict[str, Any]]:
        self._store._check()
        assert self._store._db
        safe_query = _fts_query(query)
        try:
            async with self._store._db.execute(
                """SELECT p.id, p.kind, p.trigger_text, p.checklist, p.salience, p.success_credit,
                          p.failure_credit, p.use_count, p.ts, rank
                   FROM procedures_fts
                   JOIN procedures p ON procedures_fts.rowid = p.id
                   WHERE procedures_fts MATCH ?
                   ORDER BY rank,
                            (p.pinned * 10 + p.salience + p.success_credit - p.failure_credit) DESC,
                            p.updated_ts DESC
                   LIMIT ?""",
                (safe_query, limit),
            ) as cur:
                rows = await cur.fetchall()
        except Exception:
            async with self._store._db.execute(
                """SELECT id, kind, trigger_text, checklist, salience, success_credit, failure_credit,
                          use_count, ts, 0 as rank
                   FROM procedures
                   WHERE trigger_text LIKE ? OR checklist LIKE ?
                   ORDER BY (pinned * 10 + salience + success_credit - failure_credit) DESC, updated_ts DESC
                   LIMIT ?""",
                (f"%{query}%", f"%{query}%", limit),
            ) as cur:
                rows = await cur.fetchall()
        items = [dict(row) for row in rows]
        if items:
            ids = [int(item["id"]) for item in items]
            placeholders = ",".join("?" * len(ids))
            await self._store._db.execute(
                f"""UPDATE procedures
                    SET use_count=use_count+1,
                        last_used_ts=?,
                        salience=MIN(1.0, salience + 0.02)
                    WHERE id IN ({placeholders})""",
                (time.time(), *ids),
            )
            await self._store._db.commit()
        return items

    async def pin_procedure(self, procedure_id: int, *, pinned: bool = True) -> None:
        self._store._check()
        assert self._store._db
        await self._store._db.execute(
            "UPDATE procedures SET pinned=?, updated_ts=? WHERE id=?",
            (1 if pinned else 0, time.time(), procedure_id),
        )
        await self._store._db.commit()

    async def apply_outcome_to_procedures(
        self,
        procedure_ids: list[int],
        *,
        success_delta: float = 0.0,
        failure_delta: float = 0.0,
    ) -> None:
        self._store._check()
        assert self._store._db
        if not procedure_ids:  # pragma: no cover - trivial empty-input guard
            return
        placeholders = ",".join("?" * len(procedure_ids))
        await self._store._db.execute(
            f"""UPDATE procedures
                SET success_credit=success_credit+?,
                    failure_credit=failure_credit+?,
                    salience=MIN(1.0, MAX(0.0, salience + ? - ?)),
                    updated_ts=?
                WHERE id IN ({placeholders})""",
            (
                success_delta,
                failure_delta,
                success_delta * 0.05,
                failure_delta * 0.05,
                time.time(),
                *procedure_ids,
            ),
        )
        await self._store._db.commit()


class SQLiteFeedbackRepository:
    def __init__(self, store: "SQLiteStore") -> None:
        self._store = store

    async def record_feedback(
        self,
        *,
        task_id: str = "",
        feedback_kind: str,
        score: float = 0.0,
        memory_item_id: int | None = None,
        procedure_id: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> int:
        self._store._check()
        assert self._store._db
        cur = await self._store._db.execute(
            """INSERT INTO feedback_events
               (task_id, memory_item_id, procedure_id, feedback_kind, score, details, ts)
               VALUES (?,?,?,?,?,?,?)""",
            (
                task_id,
                memory_item_id,
                procedure_id,
                feedback_kind,
                score,
                json.dumps(details or {}),
                time.time(),
            ),
        )
        await self._store._db.commit()
        if memory_item_id is not None:
            await self._store.learning.apply_outcome_to_memory(
                [memory_item_id],
                success_delta=max(score, 0.0),
                failure_delta=abs(min(score, 0.0)),
            )
        if procedure_id is not None:
            await self._store.procedures.apply_outcome_to_procedures(
                [procedure_id],
                success_delta=max(score, 0.0),
                failure_delta=abs(min(score, 0.0)),
            )
        return cur.lastrowid or 0


class SQLiteLessonRepository:
    def __init__(self, store: "SQLiteStore") -> None:
        self._store = store

    async def save_lesson(self, summary: str, kind: str = "lesson", context: str = "") -> int:
        self._store._check()
        assert self._store._db
        normalized = _normalize_text(summary)
        if not normalized:  # pragma: no cover - trivial empty-input guard
            return 0
        cur = await self._store._db.execute(
            "INSERT INTO lessons (kind, summary, context, ts) VALUES (?,?,?,?)",
            (kind, normalized, context, time.time()),
        )
        await self._store._db.commit()
        lesson_id = cur.lastrowid or 0
        await self._store.learning.save_memory_item(
            kind="warning" if kind == "mistake" else kind,
            content=normalized,
            scope="task",
            source="lesson",
            confidence=0.78 if kind in {"mistake", "pattern"} else 0.65,
            salience=0.75 if kind == "mistake" else 0.65,
            metadata={"context": context, "legacy_lesson_id": lesson_id},
        )
        if kind in {"pattern", "procedure"}:
            await self._store.procedures.save_procedure(
                trigger_text=context or normalized,
                checklist=normalized,
                kind="pattern",
                confidence=0.72,
                salience=0.68,
                metadata={"legacy_lesson_id": lesson_id},
            )
        return lesson_id

    async def search_lessons(self, query: str, limit: int = 5) -> str:
        self._store._check()
        assert self._store._db
        learning_matches = await self._store.learning.search_memory_items(
            query,
            limit=limit,
            kinds=("pattern", "insight", "lesson", "warning", "mistake"),
        )
        if learning_matches:
            import datetime

            legacy_ids = [
                int(item["metadata"].get("legacy_lesson_id", 0))
                for item in learning_matches
                if int(item["metadata"].get("legacy_lesson_id", 0))
            ]
            if legacy_ids:
                placeholders = ",".join("?" * len(legacy_ids))
                await self._store._db.execute(
                    f"UPDATE lessons SET applied=applied+1 WHERE id IN ({placeholders})",
                    legacy_ids,
                )
                await self._store._db.commit()
            lines = ["## Relevant past lessons:"]
            for row in learning_matches:
                kind = str(row["kind"]).replace("warning", "mistake")
                lines.append(
                    f"- [{kind.upper()} {datetime.datetime.fromtimestamp(row['ts']).strftime('%Y-%m-%d')}] {row['content']}"
                )
                if len("\n".join(lines)) > 1200:
                    break
            return "\n".join(lines)

        safe_query = _fts_query(query)
        try:
            async with self._store._db.execute(
                """SELECT l.id, l.kind, l.summary, l.ts
                   FROM lessons_fts
                   JOIN lessons l ON lessons_fts.rowid = l.id
                   WHERE lessons_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (safe_query, limit),
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
            """SELECT kind, summary, ts FROM lessons
               UNION ALL
               SELECT kind, content as summary, ts FROM memory_items
               WHERE kind IN ('pattern', 'insight', 'warning', 'lesson')
               ORDER BY ts DESC LIMIT ?""",
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
        for table in (
            "conversations",
            "tasks",
            "memory_facts",
            "lessons",
            "episodic_events",
            "memory_items",
            "procedures",
            "feedback_events",
            "scheduled_tasks",
        ):
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
        await self._store._db.execute(
            "DELETE FROM episodic_events WHERE ts < ?",
            (now - settings.retention_episodic_events_days * 86400,),
        )
        await self._store._db.execute(
            "DELETE FROM feedback_events WHERE ts < ?",
            (now - settings.retention_feedback_events_days * 86400,),
        )
        await self._store._db.execute(
            """DELETE FROM memory_items
               WHERE pinned=0 AND id NOT IN (
                   SELECT id FROM memory_items
                   ORDER BY (salience + success_credit - failure_credit) DESC, updated_ts DESC
                   LIMIT ?
               )""",
            (settings.retention_memory_items_max,),
        )
        await self._store._db.execute(
            """DELETE FROM procedures
               WHERE pinned=0 AND id NOT IN (
                   SELECT id FROM procedures
                   ORDER BY (salience + success_credit - failure_credit) DESC, updated_ts DESC
                   LIMIT ?
               )""",
            (settings.retention_procedures_max,),
        )
        await self._store._db.execute(
            """
            DELETE FROM memory_items
            WHERE pinned=0
              AND id NOT IN (
                  SELECT MIN(id)
                  FROM memory_items
                  GROUP BY kind, content
              )
            """
        )
        await self._store._db.execute(
            """
            DELETE FROM procedures
            WHERE pinned=0
              AND id NOT IN (
                  SELECT MIN(id)
                  FROM procedures
                  GROUP BY trigger_text, checklist
              )
            """
        )
        await self._store._db.execute(
            """
            UPDATE memory_items
            SET salience=MAX(0.05, MIN(1.0, salience * 0.995))
            WHERE pinned=0
            """
        )
        await self._store._db.execute(
            """
            UPDATE procedures
            SET salience=MAX(0.05, MIN(1.0, salience * 0.995))
            WHERE pinned=0
            """
        )
        await self._store._db.commit()

        try:
            await self._store._db.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
            await self._store._db.execute("INSERT INTO lessons_fts(lessons_fts) VALUES ('rebuild')")
            await self._store._db.execute("INSERT INTO memory_items_fts(memory_items_fts) VALUES ('rebuild')")
            await self._store._db.execute("INSERT INTO procedures_fts(procedures_fts) VALUES ('rebuild')")
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
    from agent.embeddings import embed_text as _embed_text

    return await _embed_text(text)
