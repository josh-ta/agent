"""
SQLite memory store: conversations, tasks, and memory facts.

Features:
- WAL mode for better concurrent read performance
- FTS5 virtual table for full-text search over memory facts
- sqlite-vec for optional vector similarity search (if available)
- aiosqlite for non-blocking async access
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
import structlog

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

-- Task log
CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    author      TEXT NOT NULL DEFAULT 'system',
    content     TEXT NOT NULL,
    result      TEXT,
    success     INTEGER NOT NULL DEFAULT 0,
    elapsed_ms  REAL,
    tool_calls  INTEGER DEFAULT 0,
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
"""

PRAGMAS = """
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-65536;
PRAGMA foreign_keys=ON;
PRAGMA temp_store=MEMORY;
"""


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._db: aiosqlite.Connection | None = None

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
        await self._db.commit()

        log.info("sqlite_ready", path=str(self._path))

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    # ── Conversations ─────────────────────────────────────────────────────────

    async def save_message(
        self,
        role: str,
        content: str,
        channel_id: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        assert self._db
        await self._db.execute(
            "INSERT INTO conversations (channel_id, role, content, metadata, ts) VALUES (?,?,?,?,?)",
            (channel_id, role, content, json.dumps(metadata or {}), time.time()),
        )
        await self._db.commit()

    async def get_history(self, channel_id: int = 0, limit: int = 50) -> list[dict[str, Any]]:
        assert self._db
        async with self._db.execute(
            "SELECT role, content, ts FROM conversations "
            "WHERE channel_id=? ORDER BY ts DESC LIMIT ?",
            (channel_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in reversed(rows)]

    # ── Tasks ─────────────────────────────────────────────────────────────────

    async def record_task(self, task: "Task", result: "TaskResult") -> None:
        assert self._db
        await self._db.execute(
            """INSERT INTO tasks
               (source, author, content, result, success, elapsed_ms, tool_calls, ts)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                task.source,
                task.author,
                task.content,
                result.output,
                int(result.success),
                result.elapsed_ms,
                result.tool_calls,
                time.time(),
            ),
        )
        await self._db.commit()

    # ── Memory facts ──────────────────────────────────────────────────────────

    async def save_memory_fact(self, content: str, metadata: dict[str, Any] | None = None) -> int:
        assert self._db
        cur = await self._db.execute(
            "INSERT INTO memory_facts (content, metadata, ts) VALUES (?,?,?)",
            (content, json.dumps(metadata or {}), time.time()),
        )
        await self._db.commit()
        return cur.lastrowid or 0

    async def search_memory(self, query: str, limit: int = 5) -> str:
        """Full-text search over memory facts."""
        assert self._db
        # Escape FTS5 special characters
        safe_query = query.replace('"', '""')
        try:
            async with self._db.execute(
                """SELECT mf.content, mf.ts,
                          rank
                   FROM memory_fts
                   JOIN memory_facts mf ON memory_fts.rowid = mf.id
                   WHERE memory_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (f'"{safe_query}"', limit),
            ) as cur:
                rows = await cur.fetchall()
        except Exception:
            # Fall back to LIKE search if FTS query is malformed
            async with self._db.execute(
                "SELECT content, ts, 0 as rank FROM memory_facts WHERE content LIKE ? ORDER BY ts DESC LIMIT ?",
                (f"%{query}%", limit),
            ) as cur:
                rows = await cur.fetchall()

        if not rows:
            return f"(no memory matches for: {query})"

        import datetime
        lines = []
        for row in rows:
            ts_str = datetime.datetime.fromtimestamp(row["ts"]).strftime("%Y-%m-%d")
            lines.append(f"[{ts_str}] {row['content']}")
        return "\n".join(lines)

    # ── Lessons ───────────────────────────────────────────────────────────────

    async def save_lesson(
        self,
        summary: str,
        kind: str = "lesson",
        context: str = "",
    ) -> int:
        """Record a lesson, mistake, or insight for future reference."""
        assert self._db
        cur = await self._db.execute(
            "INSERT INTO lessons (kind, summary, context, ts) VALUES (?,?,?,?)",
            (kind, summary, context, time.time()),
        )
        await self._db.commit()
        return cur.lastrowid or 0

    async def search_lessons(self, query: str, limit: int = 5) -> str:
        """Search lessons relevant to a query — surfaced before each task."""
        assert self._db
        safe_query = query.replace('"', '""')
        try:
            async with self._db.execute(
                """SELECT l.kind, l.summary, l.ts
                   FROM lessons_fts
                   JOIN lessons l ON lessons_fts.rowid = l.id
                   WHERE lessons_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (f'"{safe_query}"', limit),
            ) as cur:
                rows = await cur.fetchall()
        except Exception:
            async with self._db.execute(
                "SELECT kind, summary, ts FROM lessons WHERE summary LIKE ? ORDER BY ts DESC LIMIT ?",
                (f"%{query}%", limit),
            ) as cur:
                rows = await cur.fetchall()

        if not rows:
            return ""

        import datetime
        lines = ["## Relevant past lessons:"]
        for row in rows:
            ts_str = datetime.datetime.fromtimestamp(row["ts"]).strftime("%Y-%m-%d")
            lines.append(f"- [{row['kind'].upper()} {ts_str}] {row['summary']}")
        result = "\n".join(lines)
        # Hard cap — lessons context must not blow the prompt budget
        return result[:500]

    async def get_recent_lessons(self, limit: int = 20) -> str:
        """Return the most recent lessons for MEMORY.md updates."""
        assert self._db
        async with self._db.execute(
            "SELECT kind, summary, ts FROM lessons ORDER BY ts DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()

        if not rows:
            return "(no lessons recorded yet)"

        import datetime
        lines = []
        for row in rows:
            ts_str = datetime.datetime.fromtimestamp(row["ts"]).strftime("%Y-%m-%d")
            lines.append(f"- [{row['kind'].upper()} {ts_str}] {row['summary']}")
        return "\n".join(lines)

    async def get_failed_tasks(self, limit: int = 5) -> list[dict[str, Any]]:
        """Return recent failed tasks for post-task reflection."""
        assert self._db
        async with self._db.execute(
            "SELECT content, result, ts FROM tasks WHERE success=0 ORDER BY ts DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def heartbeat(self) -> None:
        """Periodic maintenance: checkpoint WAL."""
        if self._db:
            await self._db.execute("PRAGMA wal_checkpoint(PASSIVE)")
