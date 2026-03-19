"""
SQLite memory store: conversations, tasks, memory facts, and lessons.

Features:
- WAL mode for better concurrent read performance
- FTS5 virtual tables for full-text search over memory facts and lessons
- sqlite-vec for optional vector similarity search (when available + embedding key set)
- aiosqlite for non-blocking async access
- Automatic retention/cleanup: runs hourly from heartbeat(), keeps tables bounded
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
import structlog

from agent.memory.sqlite_components import (
    SQLiteConversationRepository,
    SQLiteLessonRepository,
    SQLiteMaintenance,
    SQLiteMemoryRepository,
    SQLiteTaskRepository,
)

if TYPE_CHECKING:
    from agent.loop import Task, TaskResult

log = structlog.get_logger()

SCHEMA = """
-- Conversation history
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id  INTEGER NOT NULL DEFAULT 0,
    role        TEXT NOT NULL,       -- user|assistant|system|tool
    content     TEXT NOT NULL,
    metadata    TEXT DEFAULT '{}',   -- JSON
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS conv_channel_ts ON conversations (channel_id, ts DESC);

-- Durable conversation sessions + turns
CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id       TEXT PRIMARY KEY,
    source           TEXT NOT NULL,
    channel_id       INTEGER NOT NULL DEFAULT 0,
    status           TEXT NOT NULL DEFAULT 'active',
    title            TEXT DEFAULT '',
    summary          TEXT DEFAULT '',
    last_user_message TEXT DEFAULT '',
    last_agent_message TEXT DEFAULT '',
    pending_task_id  TEXT DEFAULT '',
    open_questions   TEXT DEFAULT '[]',
    metadata         TEXT DEFAULT '{}',
    created_ts       REAL NOT NULL,
    updated_ts       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS conv_sessions_source_channel_idx
    ON conversation_sessions (source, channel_id, updated_ts DESC);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    task_id         TEXT DEFAULT '',
    role            TEXT NOT NULL,
    turn_kind       TEXT NOT NULL DEFAULT 'message',
    content         TEXT NOT NULL,
    metadata        TEXT DEFAULT '{}',
    ts              REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES conversation_sessions(session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS conv_turns_session_ts_idx
    ON conversation_turns (session_id, ts DESC);

CREATE TABLE IF NOT EXISTS task_checkpoints (
    task_id         TEXT PRIMARY KEY,
    session_id      TEXT DEFAULT '',
    summary         TEXT DEFAULT '',
    draft           TEXT DEFAULT '',
    notes           TEXT DEFAULT '',
    metadata        TEXT DEFAULT '{}',
    updated_ts      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS task_checkpoints_session_idx
    ON task_checkpoints (session_id, updated_ts DESC);

-- Task log
CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT UNIQUE,
    source      TEXT NOT NULL,
    author      TEXT NOT NULL DEFAULT 'system',
    content     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'completed',
    metadata    TEXT DEFAULT '{}',
    result      TEXT,
    error       TEXT,
    success     INTEGER NOT NULL DEFAULT 0,
    elapsed_ms  REAL,
    tool_calls  INTEGER DEFAULT 0,
    created_ts  REAL,
    started_ts  REAL,
    finished_ts REAL,
    updated_ts  REAL,
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS tasks_ts ON tasks (ts DESC);

-- Long-term memory facts (FTS5 for keyword search)
CREATE TABLE IF NOT EXISTS memory_facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,
    metadata    TEXT DEFAULT '{}',
    ts          REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    content='memory_facts',
    content_rowid='id'
);

-- Keep FTS in sync
CREATE TRIGGER IF NOT EXISTS memory_facts_ai AFTER INSERT ON memory_facts BEGIN
    INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memory_facts_ad AFTER DELETE ON memory_facts BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memory_facts_au AFTER UPDATE ON memory_facts BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
END;

-- Lessons learned from failures and observations
CREATE TABLE IF NOT EXISTS lessons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL DEFAULT 'lesson',  -- lesson|mistake|insight|pattern
    summary     TEXT NOT NULL,
    context     TEXT DEFAULT '',                 -- what triggered it
    applied     INTEGER NOT NULL DEFAULT 0,      -- how many times retrieved/used
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS lessons_ts ON lessons (ts DESC);
CREATE INDEX IF NOT EXISTS lessons_applied ON lessons (applied DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS lessons_fts USING fts5(
    summary,
    context,
    content='lessons',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS lessons_ai AFTER INSERT ON lessons BEGIN
    INSERT INTO lessons_fts(rowid, summary, context) VALUES (new.id, new.summary, new.context);
END;
CREATE TRIGGER IF NOT EXISTS lessons_ad AFTER DELETE ON lessons BEGIN
    INSERT INTO lessons_fts(lessons_fts, rowid, summary, context)
    VALUES ('delete', old.id, old.summary, old.context);
END;
CREATE TRIGGER IF NOT EXISTS lessons_au AFTER UPDATE ON lessons BEGIN
    INSERT INTO lessons_fts(lessons_fts, rowid, summary, context)
    VALUES ('delete', old.id, old.summary, old.context);
    INSERT INTO lessons_fts(rowid, summary, context) VALUES (new.id, new.summary, new.context);
END;

-- Internal metadata (cleanup timestamps, etc.)
CREATE TABLE IF NOT EXISTS _meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

PRAGMAS = """
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-65536;
PRAGMA foreign_keys=ON;
PRAGMA temp_store=MEMORY;
"""

# Cleanup interval — run at most once per hour
_CLEANUP_INTERVAL_S = 3600
# VACUUM interval — run at most once per 30 days
_VACUUM_INTERVAL_S = 30 * 24 * 3600


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._db: aiosqlite.Connection | None = None
        self._has_vec: bool = False
        self._last_cleanup_ts: float = 0.0
        self.conversations = SQLiteConversationRepository(self)
        self.tasks = SQLiteTaskRepository(self)
        self.memory_facts = SQLiteMemoryRepository(self)
        self.lessons = SQLiteLessonRepository(self)
        self.maintenance = SQLiteMaintenance(self)

    def _check(self) -> None:
        if not self._db:
            raise RuntimeError("SQLiteStore not initialized — call await init() first")

    async def init(self) -> None:
        """Open the database, apply pragmas, and run migrations."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._path))
        self._db.row_factory = aiosqlite.Row

        for pragma in PRAGMAS.strip().splitlines():
            pragma = pragma.strip()
            if pragma:
                await self._db.execute(pragma)

        # Run schema (idempotent CREATE IF NOT EXISTS)
        await self._db.executescript(SCHEMA)
        await self.tasks.migrate()
        await self._db.commit()

        # Attempt to load sqlite-vec for vector similarity search
        try:
            import sqlite_vec
            await self._db.enable_load_extension(True)
            await self._db.load_extension(sqlite_vec.loadable_path())
            await self._db.enable_load_extension(False)
            self._has_vec = True
            # Create vec0 virtual table for memory_facts embeddings
            await self._db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
                    fact_id INTEGER PRIMARY KEY,
                    embedding float[1536]
                )
            """)
            await self._db.commit()
            log.info("sqlite_vec_loaded")
        except Exception as exc:
            log.info("sqlite_vec_unavailable", reason=str(exc))

        # Read last cleanup time from _meta
        try:
            async with self._db.execute(
                "SELECT value FROM _meta WHERE key='last_cleanup_ts'"
            ) as cur:
                row = await cur.fetchone()
                if row:
                    self._last_cleanup_ts = float(row[0])
        except Exception:
            pass

        log.info("sqlite_ready", path=str(self._path), vec=self._has_vec)

    async def close(self) -> None:
        if self._db:
            db = self._db
            try:
                try:
                    await db.rollback()
                except Exception as exc:
                    log.debug("sqlite_rollback_on_close_failed", error=str(exc))
                try:
                    await db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception as exc:
                    log.debug("sqlite_checkpoint_on_close_failed", error=str(exc))
            finally:
                await db.close()
                self._db = None

    # ── Conversations ─────────────────────────────────────────────────────────

    async def save_message(
        self,
        role: str,
        content: str,
        channel_id: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.conversations.save_message(role, content, channel_id, metadata)

    async def get_history(self, channel_id: int = 0, limit: int = 20) -> list[dict[str, Any]]:
        return await self.conversations.get_history(channel_id, limit)

    async def ensure_session(
        self,
        *,
        session_id: str,
        source: str,
        channel_id: int = 0,
        title: str = "",
        status: str = "active",
        pending_task_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._check()
        assert self._db
        now = time.time()
        await self._db.execute(
            """INSERT INTO conversation_sessions
               (session_id, source, channel_id, status, title, pending_task_id, metadata, created_ts, updated_ts)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(session_id) DO UPDATE SET
                   source=excluded.source,
                   channel_id=excluded.channel_id,
                   status=excluded.status,
                   title=CASE WHEN excluded.title != '' THEN excluded.title ELSE conversation_sessions.title END,
                   pending_task_id=CASE
                       WHEN excluded.pending_task_id != '' THEN excluded.pending_task_id
                       ELSE conversation_sessions.pending_task_id
                   END,
                   metadata=excluded.metadata,
                   updated_ts=excluded.updated_ts""",
            (
                session_id,
                source,
                channel_id,
                status,
                title,
                pending_task_id,
                json.dumps(metadata or {}),
                now,
                now,
            ),
        )
        await self._db.commit()

    async def append_session_turn(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        turn_kind: str = "message",
        task_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._check()
        assert self._db
        if not session_id or not content.strip():
            return
        now = time.time()
        await self._db.execute(
            """INSERT INTO conversation_turns
               (session_id, task_id, role, turn_kind, content, metadata, ts)
               VALUES (?,?,?,?,?,?,?)""",
            (session_id, task_id, role, turn_kind, content, json.dumps(metadata or {}), now),
        )
        session = await self.get_session(session_id)
        last_user = content if role == "user" else str((session or {}).get("last_user_message", ""))
        last_agent = content if role == "assistant" else str((session or {}).get("last_agent_message", ""))
        open_questions = []
        if session is not None:
            open_questions = list((session.get("open_questions") or []))
        if turn_kind == "question":
            open_questions.append(content[:300])
        elif turn_kind in {"answer", "assistant"} and open_questions:
            open_questions = open_questions[-3:]
        summary_parts = []
        if last_user:
            summary_parts.append(f"Latest user goal: {last_user[:280]}")
        if open_questions:
            summary_parts.append("Open questions: " + " | ".join(item[:120] for item in open_questions[-3:]))
        if last_agent:
            summary_parts.append(f"Latest agent reply: {last_agent[:280]}")
        await self._db.execute(
            """UPDATE conversation_sessions
               SET last_user_message=?,
                   last_agent_message=?,
                   open_questions=?,
                   summary=?,
                   updated_ts=?
               WHERE session_id=?""",
            (
                last_user,
                last_agent,
                json.dumps(open_questions[-5:]),
                "\n".join(summary_parts)[:2000],
                now,
                session_id,
            ),
        )
        await self._db.commit()

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        self._check()
        assert self._db
        async with self._db.execute(
            """SELECT session_id, source, channel_id, status, title, summary, last_user_message,
                      last_agent_message, pending_task_id, open_questions, metadata, created_ts, updated_ts
               FROM conversation_sessions
               WHERE session_id=?
               LIMIT 1""",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        data = dict(row)
        for key in ("open_questions", "metadata"):
            try:
                data[key] = json.loads(data.get(key) or ("[]" if key == "open_questions" else "{}"))
            except Exception:
                data[key] = [] if key == "open_questions" else {}
        return data

    async def list_session_turns(self, session_id: str, limit: int = 12) -> list[dict[str, Any]]:
        self._check()
        assert self._db
        async with self._db.execute(
            """SELECT session_id, task_id, role, turn_kind, content, metadata, ts
               FROM conversation_turns
               WHERE session_id=?
               ORDER BY ts DESC
               LIMIT ?""",
            (session_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        turns = [dict(row) for row in reversed(rows)]
        for row in turns:
            try:
                row["metadata"] = json.loads(row.get("metadata") or "{}")
            except Exception:
                row["metadata"] = {}
        return turns

    async def set_session_status(
        self,
        session_id: str,
        *,
        status: str | None = None,
        pending_task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._check()
        assert self._db
        session = await self.get_session(session_id)
        if session is None:
            return
        next_status = status or str(session.get("status", "active"))
        next_pending = (
            str(pending_task_id)
            if pending_task_id is not None
            else str(session.get("pending_task_id", ""))
        )
        next_metadata = metadata if metadata is not None else dict(session.get("metadata") or {})
        await self._db.execute(
            """UPDATE conversation_sessions
               SET status=?, pending_task_id=?, metadata=?, updated_ts=?
               WHERE session_id=?""",
            (next_status, next_pending, json.dumps(next_metadata), time.time(), session_id),
        )
        await self._db.commit()

    async def get_session_context(self, session_id: str, *, limit: int = 10, char_cap: int = 2500) -> str:
        session = await self.get_session(session_id)
        if session is None:
            return ""
        turns = await self.list_session_turns(session_id, limit=limit)
        parts: list[str] = []
        summary = str(session.get("summary", "")).strip()
        if summary:
            parts.append("## Session summary\n" + summary[:1200])
        if turns:
            rendered: list[str] = []
            total = 0
            for row in turns:
                line = f"{row['role'].capitalize()} ({row['turn_kind']}): {str(row['content'])[:320]}"
                if total + len(line) > char_cap:
                    break
                rendered.append(line)
                total += len(line)
            if rendered:
                parts.append("## Recent session turns\n" + "\n".join(rendered))
        return "\n\n".join(parts)

    async def save_task_checkpoint(
        self,
        *,
        task_id: str,
        session_id: str = "",
        summary: str = "",
        draft: str = "",
        notes: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._check()
        assert self._db
        if not task_id:
            return
        await self._db.execute(
            """INSERT INTO task_checkpoints (task_id, session_id, summary, draft, notes, metadata, updated_ts)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(task_id) DO UPDATE SET
                   session_id=excluded.session_id,
                   summary=excluded.summary,
                   draft=excluded.draft,
                   notes=excluded.notes,
                   metadata=excluded.metadata,
                   updated_ts=excluded.updated_ts""",
            (task_id, session_id, summary, draft, notes, json.dumps(metadata or {}), time.time()),
        )
        await self._db.commit()

    async def append_task_note(self, task_id: str, note: str, *, session_id: str = "") -> None:
        checkpoint = await self.get_task_checkpoint(task_id)
        existing_notes = str((checkpoint or {}).get("notes", ""))
        merged = (existing_notes + ("\n" if existing_notes else "") + note.strip()).strip()
        await self.save_task_checkpoint(
            task_id=task_id,
            session_id=session_id or str((checkpoint or {}).get("session_id", "")),
            summary=str((checkpoint or {}).get("summary", "")),
            draft=str((checkpoint or {}).get("draft", "")),
            notes=merged[-8000:],
            metadata=dict((checkpoint or {}).get("metadata") or {}),
        )

    async def get_task_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        self._check()
        assert self._db
        async with self._db.execute(
            """SELECT task_id, session_id, summary, draft, notes, metadata, updated_ts
               FROM task_checkpoints
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

    async def clear_task_checkpoint(self, task_id: str) -> None:
        self._check()
        assert self._db
        await self._db.execute("DELETE FROM task_checkpoints WHERE task_id=?", (task_id,))
        await self._db.commit()

    # ── Tasks ─────────────────────────────────────────────────────────────────

    async def record_task(self, task: Task, result: TaskResult) -> None:
        await self.tasks.record_task(task, result)

    async def create_task_record(
        self,
        *,
        task_id: str,
        source: str,
        author: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.tasks.create_task_record(
            task_id=task_id,
            source=source,
            author=author,
            content=content,
            metadata=metadata,
        )

    async def mark_task_running(self, task_id: str) -> None:
        await self.tasks.mark_task_running(task_id)

    async def mark_task_waiting(self, task_id: str, *, metadata: dict[str, Any], question: str) -> None:
        await self.tasks.mark_task_waiting(task_id, metadata=metadata, question=question)

    async def mark_task_queued(self, task_id: str, *, metadata: dict[str, Any] | None = None) -> None:
        await self.tasks.mark_task_queued(task_id, metadata=metadata)

    async def get_task_record(self, task_id: str) -> dict[str, Any] | None:
        return await self.tasks.get_task_record(task_id)

    async def list_waiting_task_records(self) -> list[dict[str, Any]]:
        return await self.tasks.list_waiting_task_records()

    async def list_pending_task_records(self) -> list[dict[str, Any]]:
        self._check()
        assert self._db
        async with self._db.execute(
            """SELECT task_id, source, author, content, status, result, error, success,
                      metadata, elapsed_ms, tool_calls, created_ts, started_ts, finished_ts, updated_ts
               FROM tasks
               WHERE status IN ('queued', 'running')
               ORDER BY updated_ts ASC, created_ts ASC"""
        ) as cur:
            rows = await cur.fetchall()
        items = [dict(row) for row in rows]
        for row in items:
            try:
                row["metadata"] = json.loads(row.get("metadata") or "{}")
            except Exception:
                row["metadata"] = {}
        return items

    async def fail_task(self, task_id: str, *, error: str, metadata: dict[str, Any] | None = None) -> None:
        self._check()
        assert self._db
        now = time.time()
        params = [error, now, now, task_id]
        if metadata is None:
            await self._db.execute(
                """UPDATE tasks
                   SET status='failed', error=?, success=0, finished_ts=?, updated_ts=?
                   WHERE task_id=?""",
                params,
            )
        else:
            await self._db.execute(
                """UPDATE tasks
                   SET status='failed', error=?, success=0, finished_ts=?, updated_ts=?, metadata=?
                   WHERE task_id=?""",
                (error, now, now, json.dumps(metadata), task_id),
            )
        await self._db.commit()

    # ── Memory facts ──────────────────────────────────────────────────────────

    async def save_memory_fact(self, content: str, metadata: dict[str, Any] | None = None) -> int:
        return await self.memory_facts.save_memory_fact(content, metadata)

    async def search_memory(self, query: str, limit: int = 5) -> str:
        """Search memory facts — tries vector similarity first, falls back to FTS5."""
        return await self.memory_facts.search_memory(query, limit)

    # ── Lessons ───────────────────────────────────────────────────────────────

    async def save_lesson(
        self,
        summary: str,
        kind: str = "lesson",
        context: str = "",
    ) -> int:
        """Record a lesson, mistake, or insight for future reference."""
        return await self.lessons.save_lesson(summary, kind, context)

    async def search_lessons(self, query: str, limit: int = 5) -> str:
        """Search lessons relevant to a query and increment their applied counters."""
        return await self.lessons.search_lessons(query, limit)

    async def get_recent_lessons(self, limit: int = 20) -> str:
        """Return the most recent lessons for MEMORY.md updates."""
        return await self.lessons.get_recent_lessons(limit)

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        """Return row counts and file size for diagnostics."""
        return await self.maintenance.get_stats()

    # ── Cleanup / Retention ───────────────────────────────────────────────────

    async def _cleanup(self) -> None:
        """Prune tables to their retention limits."""
        await self.maintenance.cleanup()

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def heartbeat(self) -> None:
        """Periodic maintenance: checkpoint WAL, run cleanup if due."""
        await self.maintenance.heartbeat()

    async def healthcheck(self) -> bool:
        self._check()
        assert self._db
        try:
            async with self._db.execute("SELECT 1") as cur:
                row = await cur.fetchone()
            return bool(row and row[0] == 1)
        except Exception:
            return False


# ── Embedding helper (mirrors postgres_store pattern) ─────────────────────────

async def _embed(text: str) -> list[float] | None:
    """Generate an embedding via OpenAI. Returns None on any failure."""
    from agent.config import settings
    if not settings.has_embeddings:
        return None
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.secret_value(settings.openai_api_key))
        resp = await client.embeddings.create(
            model=settings.embedding_model,
            input=text,
            encoding_format="float",
        )
        return resp.data[0].embedding
    except Exception as exc:
        log.warning("sqlite_embed_failed", error=str(exc))
        return None
