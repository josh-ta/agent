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
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
import structlog

from agent.config import settings
from agent.memory.sqlite_components import (
    SQLiteConversationRepository,
    SQLiteFeedbackRepository,
    SQLiteLessonRepository,
    SQLiteLearningRepository,
    SQLiteMaintenance,
    SQLiteMemoryRepository,
    SQLiteProcedureRepository,
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

-- Episodic learning events used for reliability scoring and replay
CREATE TABLE IF NOT EXISTS episodic_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT DEFAULT '',
    session_id   TEXT DEFAULT '',
    event_kind   TEXT NOT NULL DEFAULT 'observation',
    summary      TEXT NOT NULL,
    reward       REAL NOT NULL DEFAULT 0,
    details      TEXT DEFAULT '{}',
    ts           REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS episodic_events_task_ts_idx
    ON episodic_events (task_id, ts DESC);
CREATE INDEX IF NOT EXISTS episodic_events_kind_ts_idx
    ON episodic_events (event_kind, ts DESC);

-- Normalized long-term memory with usefulness metadata
CREATE TABLE IF NOT EXISTS memory_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT NOT NULL DEFAULT 'fact',
    scope           TEXT NOT NULL DEFAULT 'task',
    content         TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'agent',
    confidence      REAL NOT NULL DEFAULT 0.5,
    salience        REAL NOT NULL DEFAULT 0.5,
    success_credit  REAL NOT NULL DEFAULT 0,
    failure_credit  REAL NOT NULL DEFAULT 0,
    use_count       INTEGER NOT NULL DEFAULT 0,
    pinned          INTEGER NOT NULL DEFAULT 0,
    sensitive       INTEGER NOT NULL DEFAULT 0,
    last_used_ts    REAL,
    metadata        TEXT DEFAULT '{}',
    ts              REAL NOT NULL,
    updated_ts      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_items_kind_updated_idx
    ON memory_items (kind, updated_ts DESC);
CREATE INDEX IF NOT EXISTS memory_items_salience_idx
    ON memory_items (pinned DESC, salience DESC, updated_ts DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts USING fts5(
    content,
    source,
    content='memory_items',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS memory_items_ai AFTER INSERT ON memory_items BEGIN
    INSERT INTO memory_items_fts(rowid, content, source) VALUES (new.id, new.content, new.source);
END;
CREATE TRIGGER IF NOT EXISTS memory_items_ad AFTER DELETE ON memory_items BEGIN
    INSERT INTO memory_items_fts(memory_items_fts, rowid, content, source)
    VALUES ('delete', old.id, old.content, old.source);
END;
CREATE TRIGGER IF NOT EXISTS memory_items_au AFTER UPDATE ON memory_items BEGIN
    INSERT INTO memory_items_fts(memory_items_fts, rowid, content, source)
    VALUES ('delete', old.id, old.content, old.source);
    INSERT INTO memory_items_fts(rowid, content, source) VALUES (new.id, new.content, new.source);
END;

-- Reusable procedures and checklists
CREATE TABLE IF NOT EXISTS procedures (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT NOT NULL DEFAULT 'procedure',
    trigger_text    TEXT NOT NULL,
    checklist       TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.5,
    salience        REAL NOT NULL DEFAULT 0.5,
    success_credit  REAL NOT NULL DEFAULT 0,
    failure_credit  REAL NOT NULL DEFAULT 0,
    use_count       INTEGER NOT NULL DEFAULT 0,
    pinned          INTEGER NOT NULL DEFAULT 0,
    last_used_ts    REAL,
    metadata        TEXT DEFAULT '{}',
    ts              REAL NOT NULL,
    updated_ts      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS procedures_salience_idx
    ON procedures (pinned DESC, salience DESC, updated_ts DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS procedures_fts USING fts5(
    trigger_text,
    checklist,
    content='procedures',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS procedures_ai AFTER INSERT ON procedures BEGIN
    INSERT INTO procedures_fts(rowid, trigger_text, checklist)
    VALUES (new.id, new.trigger_text, new.checklist);
END;
CREATE TRIGGER IF NOT EXISTS procedures_ad AFTER DELETE ON procedures BEGIN
    INSERT INTO procedures_fts(procedures_fts, rowid, trigger_text, checklist)
    VALUES ('delete', old.id, old.trigger_text, old.checklist);
END;
CREATE TRIGGER IF NOT EXISTS procedures_au AFTER UPDATE ON procedures BEGIN
    INSERT INTO procedures_fts(procedures_fts, rowid, trigger_text, checklist)
    VALUES ('delete', old.id, old.trigger_text, old.checklist);
    INSERT INTO procedures_fts(rowid, trigger_text, checklist)
    VALUES (new.id, new.trigger_text, new.checklist);
END;

-- Explicit human/operator feedback on tasks and memories
CREATE TABLE IF NOT EXISTS feedback_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         TEXT DEFAULT '',
    memory_item_id  INTEGER,
    procedure_id    INTEGER,
    feedback_kind   TEXT NOT NULL,
    score           REAL NOT NULL DEFAULT 0,
    details         TEXT DEFAULT '{}',
    ts              REAL NOT NULL,
    FOREIGN KEY(memory_item_id) REFERENCES memory_items(id) ON DELETE SET NULL,
    FOREIGN KEY(procedure_id) REFERENCES procedures(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS feedback_events_task_ts_idx
    ON feedback_events (task_id, ts DESC);
CREATE INDEX IF NOT EXISTS feedback_events_kind_ts_idx
    ON feedback_events (feedback_kind, ts DESC);

-- Tool permissions (CC-style rules)
CREATE TABLE IF NOT EXISTS permission_rules (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name       TEXT NOT NULL,
    rule_behavior   TEXT NOT NULL,
    rule_content    TEXT DEFAULT '',
    source          TEXT DEFAULT 'projectSettings',
    created_ts      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS permission_rules_tool_idx ON permission_rules (tool_name);

-- Versioned per-task transcript (for resume / audit)
CREATE TABLE IF NOT EXISTS transcript_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    schema_version  INTEGER NOT NULL DEFAULT 1,
    role            TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'message',
    content         TEXT NOT NULL,
    metadata        TEXT DEFAULT '{}',
    ts              REAL NOT NULL,
    UNIQUE(task_id, seq)
);
CREATE INDEX IF NOT EXISTS transcript_task_seq_idx ON transcript_entries (task_id, seq);

-- Background / scheduled prompts (heartbeat enqueues into agent task queue)
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id              TEXT PRIMARY KEY,
    prompt          TEXT NOT NULL,
    next_run_ts     REAL NOT NULL,
    interval_seconds REAL,
    enabled         INTEGER NOT NULL DEFAULT 1,
    metadata        TEXT NOT NULL DEFAULT '{}',
    created_ts      REAL NOT NULL,
    last_run_ts     REAL
);
CREATE INDEX IF NOT EXISTS scheduled_tasks_due_idx
    ON scheduled_tasks (enabled, next_run_ts);

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
        self.learning = SQLiteLearningRepository(self)
        self.procedures = SQLiteProcedureRepository(self)
        self.feedback = SQLiteFeedbackRepository(self)
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
        from agent.migrations.runner import run_sqlite_migrations

        await run_sqlite_migrations(self._db)
        await self._db.commit()

        # Attempt to load sqlite-vec for vector similarity search
        try:
            import sqlite_vec
            await self._db.enable_load_extension(True)
            await self._db.load_extension(sqlite_vec.loadable_path())
            await self._db.enable_load_extension(False)
            self._has_vec = True
            dim = max(1, int(settings.embedding_dimensions))
            # Create vec0 virtual table for memory_facts embeddings
            await self._db.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
                    fact_id INTEGER PRIMARY KEY,
                    embedding float[{dim}]
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

    # ── Permissions (rules only; mode comes from PERMISSION_MODE env) ─────────

    async def permission_list_rules(self) -> list[dict[str, Any]]:
        self._check()
        assert self._db
        async with self._db.execute(
            """SELECT tool_name, rule_behavior, rule_content, source
               FROM permission_rules ORDER BY id ASC"""
        ) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in rows]

    # ── Transcripts ───────────────────────────────────────────────────────────

    async def append_transcript_entry(
        self,
        *,
        task_id: str,
        role: str,
        content: str,
        kind: str = "message",
        metadata: dict[str, Any] | None = None,
        schema_version: int = 1,
    ) -> None:
        self._check()
        assert self._db
        tid = task_id.strip()
        if not tid:
            return
        async with self._db.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS n FROM transcript_entries WHERE task_id=?",
            (tid,),
        ) as cur:
            row = await cur.fetchone()
        seq = int(row[0]) if row else 0
        now = time.time()
        await self._db.execute(
            """INSERT INTO transcript_entries
               (task_id, seq, schema_version, role, kind, content, metadata, ts)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                tid,
                seq,
                schema_version,
                role,
                kind,
                content,
                json.dumps(metadata or {}),
                now,
            ),
        )
        await self._db.commit()

    async def list_transcript_entries(self, task_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        self._check()
        assert self._db
        async with self._db.execute(
            """SELECT task_id, seq, schema_version, role, kind, content, metadata, ts
               FROM transcript_entries WHERE task_id=? ORDER BY seq ASC LIMIT ?""",
            (task_id.strip(), max(1, min(limit, 500))),
        ) as cur:
            rows = await cur.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["metadata"] = json.loads(item.get("metadata") or "{}")
            except Exception:
                item["metadata"] = {}
            out.append(item)
        return out

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

    async def search_learning_context(self, query: str, limit: int = 3) -> dict[str, Any]:
        return await self.learning.search_learning_context(query, limit=limit)

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
        return await self.learning.save_memory_item(
            kind=kind,
            content=content,
            scope=scope,
            source=source,
            confidence=confidence,
            salience=salience,
            pinned=pinned,
            sensitive=sensitive,
            metadata=metadata,
        )

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
        return await self.procedures.save_procedure(
            trigger_text=trigger_text,
            checklist=checklist,
            kind=kind,
            confidence=confidence,
            salience=salience,
            pinned=pinned,
            metadata=metadata,
        )

    async def search_procedures(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        return await self.procedures.search_procedures(query, limit=limit)

    async def pin_memory_item(self, memory_item_id: int, *, pinned: bool = True) -> None:
        await self.learning.pin_memory_item(memory_item_id, pinned=pinned)

    async def pin_procedure(self, procedure_id: int, *, pinned: bool = True) -> None:
        await self.procedures.pin_procedure(procedure_id, pinned=pinned)

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
        return await self.learning.record_episodic_event(
            task_id=task_id,
            session_id=session_id,
            event_kind=event_kind,
            summary=summary,
            reward=reward,
            details=details,
        )

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
        return await self.feedback.record_feedback(
            task_id=task_id,
            feedback_kind=feedback_kind,
            score=score,
            memory_item_id=memory_item_id,
            procedure_id=procedure_id,
            details=details,
        )

    # ── Scheduled tasks (background enqueue) ──────────────────────────────────

    async def scheduled_task_count(self) -> int:
        self._check()
        assert self._db
        async with self._db.execute("SELECT COUNT(*) AS n FROM scheduled_tasks WHERE enabled=1") as cur:
            row = await cur.fetchone()
        return int(row["n"] if row else 0)

    async def scheduled_task_create(
        self,
        *,
        prompt: str,
        delay_seconds: float,
        interval_seconds: float | None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self._check()
        assert self._db
        now = time.time()
        task_id = str(uuid.uuid4())
        next_ts = now + max(0.0, float(delay_seconds))
        meta = json.dumps(metadata or {})
        await self._db.execute(
            """INSERT INTO scheduled_tasks
               (id, prompt, next_run_ts, interval_seconds, enabled, metadata, created_ts, last_run_ts)
               VALUES (?,?,?,?,1,?,?,NULL)""",
            (task_id, prompt, next_ts, interval_seconds, meta, now),
        )
        await self._db.commit()
        return task_id

    async def scheduled_task_list(self, *, include_disabled: bool = False) -> list[dict[str, Any]]:
        self._check()
        assert self._db
        q = "SELECT * FROM scheduled_tasks"
        if not include_disabled:
            q += " WHERE enabled=1"
        q += " ORDER BY next_run_ts ASC LIMIT 200"
        async with self._db.execute(q) as cur:
            rows = await cur.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            try:
                d["metadata"] = json.loads(d.get("metadata") or "{}")
            except json.JSONDecodeError:
                d["metadata"] = {}
            out.append(d)
        return out

    async def scheduled_task_cancel(self, task_id: str) -> bool:
        self._check()
        assert self._db
        async with self._db.execute(
            "SELECT 1 FROM scheduled_tasks WHERE id=? AND enabled=1",
            (task_id,),
        ) as cur:
            if await cur.fetchone() is None:
                return False
        await self._db.execute("UPDATE scheduled_tasks SET enabled=0 WHERE id=?", (task_id,))
        await self._db.commit()
        return True

    async def scheduled_tasks_claim_due(self, *, now: float, limit: int) -> list[dict[str, Any]]:
        """Atomically advance next_run / disable one-shots for due rows; returns rows to enqueue."""
        self._check()
        assert self._db
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            async with self._db.execute(
                """SELECT id, prompt, interval_seconds, metadata FROM scheduled_tasks
                   WHERE enabled=1 AND next_run_ts <= ?
                   ORDER BY next_run_ts ASC LIMIT ?""",
                (now, limit),
            ) as cur:
                raw = await cur.fetchall()
            claimed: list[dict[str, Any]] = []
            for row in raw:
                sid = str(row["id"])
                prompt = str(row["prompt"])
                interval = row["interval_seconds"]
                try:
                    meta = json.loads(row["metadata"] or "{}")
                except json.JSONDecodeError:
                    meta = {}
                if interval is not None and float(interval) > 0:
                    nxt = now + float(interval)
                    await self._db.execute(
                        "UPDATE scheduled_tasks SET last_run_ts=?, next_run_ts=? WHERE id=?",
                        (now, nxt, sid),
                    )
                else:
                    await self._db.execute(
                        "UPDATE scheduled_tasks SET enabled=0, last_run_ts=? WHERE id=?",
                        (now, sid),
                    )
                claimed.append({"id": sid, "prompt": prompt, "metadata": meta})
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise
        return claimed

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
    from agent.embeddings import embed_text

    return await embed_text(text)
