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
            await self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            await self._db.close()
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


# ── Embedding helper (mirrors postgres_store pattern) ─────────────────────────

async def _embed(text: str) -> list[float] | None:
    """Generate an embedding via OpenAI. Returns None on any failure."""
    from agent.config import settings
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
