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
        self._check()
        assert self._db
        await self._db.execute(
            "INSERT INTO conversations (channel_id, role, content, metadata, ts) VALUES (?,?,?,?,?)",
            (channel_id, role, content, json.dumps(metadata or {}), time.time()),
        )
        await self._db.commit()

    async def get_history(self, channel_id: int = 0, limit: int = 20) -> list[dict[str, Any]]:
        self._check()
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
        self._check()
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
        self._check()
        assert self._db
        cur = await self._db.execute(
            "INSERT INTO memory_facts (content, metadata, ts) VALUES (?,?,?)",
            (content, json.dumps(metadata or {}), time.time()),
        )
        await self._db.commit()
        fact_id = cur.lastrowid or 0

        # Store embedding in vec table if sqlite-vec is available
        if self._has_vec and fact_id:
            try:
                embedding = await _embed(content)
                if embedding is not None:
                    vec_str = json.dumps(embedding)
                    await self._db.execute(
                        "INSERT OR REPLACE INTO memory_vec(fact_id, embedding) VALUES (?, ?)",
                        (fact_id, vec_str),
                    )
                    await self._db.commit()
            except Exception as exc:
                log.warning("memory_vec_insert_failed", error=str(exc))

        return fact_id

    async def search_memory(self, query: str, limit: int = 5) -> str:
        """Search memory facts — tries vector similarity first, falls back to FTS5."""
        self._check()
        assert self._db

        # Try semantic vector search first
        if self._has_vec:
            try:
                embedding = await _embed(query)
                if embedding is not None:
                    vec_str = json.dumps(embedding)
                    async with self._db.execute(
                        """SELECT mf.content, mf.ts, mv.distance
                           FROM memory_vec mv
                           JOIN memory_facts mf ON mv.fact_id = mf.id
                           WHERE mv.embedding MATCH ?
                             AND k = ?
                           ORDER BY mv.distance""",
                        (vec_str, limit),
                    ) as cur:
                        rows = await cur.fetchall()
                    if rows:
                        import datetime
                        lines = []
                        for row in rows:
                            ts_str = datetime.datetime.fromtimestamp(row["ts"]).strftime("%Y-%m-%d")
                            lines.append(f"[{ts_str}] {row['content']}")
                        return "\n".join(lines)
            except Exception as exc:
                log.debug("memory_vec_search_failed", error=str(exc))

        # FTS5 keyword search
        safe_query = query.replace('"', '""')
        try:
            async with self._db.execute(
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
        self._check()
        assert self._db
        cur = await self._db.execute(
            "INSERT INTO lessons (kind, summary, context, ts) VALUES (?,?,?,?)",
            (kind, summary, context, time.time()),
        )
        await self._db.commit()
        return cur.lastrowid or 0

    async def search_lessons(self, query: str, limit: int = 5) -> str:
        """Search lessons relevant to a query — surfaced before each task.
        Increments `applied` counter for each matched lesson so useful lessons
        accumulate higher scores and survive retention longer.
        """
        self._check()
        assert self._db
        safe_query = query.replace('"', '""')
        matched_ids: list[int] = []
        try:
            async with self._db.execute(
                """SELECT l.id, l.kind, l.summary, l.ts
                   FROM lessons_fts
                   JOIN lessons l ON lessons_fts.rowid = l.id
                   WHERE lessons_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (f'"{safe_query}"', limit),
            ) as cur:
                rows = await cur.fetchall()
            matched_ids = [r["id"] for r in rows]
        except Exception:
            async with self._db.execute(
                "SELECT id, kind, summary, ts FROM lessons WHERE summary LIKE ? ORDER BY ts DESC LIMIT ?",
                (f"%{query}%", limit),
            ) as cur:
                rows = await cur.fetchall()
            matched_ids = [r["id"] for r in rows]

        if not rows:
            return ""

        # Increment applied counter for retrieved lessons
        if matched_ids:
            placeholders = ",".join("?" * len(matched_ids))
            await self._db.execute(
                f"UPDATE lessons SET applied=applied+1 WHERE id IN ({placeholders})",
                matched_ids,
            )
            await self._db.commit()

        import datetime
        lines = ["## Relevant past lessons:"]
        for row in rows:
            ts_str = datetime.datetime.fromtimestamp(row["ts"]).strftime("%Y-%m-%d")
            lines.append(f"- [{row['kind'].upper()} {ts_str}] {row['summary']}")
            # Cap at whole-lesson boundary — stop before this line would overflow
            candidate = "\n".join(lines)
            if len(candidate) > 1200:
                break
        return "\n".join(lines)

    async def get_recent_lessons(self, limit: int = 20) -> str:
        """Return the most recent lessons for MEMORY.md updates."""
        self._check()
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
        self._check()
        assert self._db
        async with self._db.execute(
            "SELECT content, result, ts FROM tasks WHERE success=0 ORDER BY ts DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        """Return row counts and file size for diagnostics."""
        self._check()
        assert self._db
        stats: dict[str, Any] = {}
        for table in ("conversations", "tasks", "memory_facts", "lessons"):
            async with self._db.execute(f"SELECT COUNT(*) as n FROM {table}") as cur:
                row = await cur.fetchone()
                stats[table] = row["n"] if row else 0
        try:
            stats["db_size_mb"] = round(self._path.stat().st_size / 1_048_576, 2)
        except Exception:
            stats["db_size_mb"] = "unknown"
        stats["last_cleanup"] = (
            time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(self._last_cleanup_ts))
            if self._last_cleanup_ts else "never"
        )
        stats["vec_enabled"] = self._has_vec
        return stats

    # ── Cleanup / Retention ───────────────────────────────────────────────────

    async def _cleanup(self) -> None:
        """Prune tables to their retention limits. Runs at most once per hour."""
        from agent.config import settings
        self._check()
        assert self._db
        now = time.time()
        log.info("sqlite_cleanup_start")

        # conversations: keep last RETENTION_CONVERSATIONS_DAYS per channel,
        # also hard-cap at 1000 rows per channel
        days = settings.retention_conversations_days
        cutoff = now - days * 86400
        await self._db.execute(
            "DELETE FROM conversations WHERE ts < ?", (cutoff,)
        )
        # Per-channel hard cap: delete oldest beyond 1000
        async with self._db.execute(
            "SELECT DISTINCT channel_id FROM conversations"
        ) as cur:
            channels = [r[0] for r in await cur.fetchall()]
        for ch in channels:
            await self._db.execute(
                """DELETE FROM conversations WHERE channel_id=? AND id NOT IN (
                       SELECT id FROM conversations WHERE channel_id=? ORDER BY ts DESC LIMIT 1000
                   )""",
                (ch, ch),
            )

        # tasks: keep last RETENTION_TASKS_DAYS
        days_t = settings.retention_tasks_days
        await self._db.execute(
            "DELETE FROM tasks WHERE ts < ?", (now - days_t * 86400,)
        )

        # memory_facts: keep newest RETENTION_MEMORY_FACTS_MAX rows
        max_facts = settings.retention_memory_facts_max
        await self._db.execute(
            """DELETE FROM memory_facts WHERE id NOT IN (
                   SELECT id FROM memory_facts ORDER BY ts DESC LIMIT ?
               )""",
            (max_facts,),
        )
        # Also clean up orphaned vec entries
        if self._has_vec:
            await self._db.execute(
                "DELETE FROM memory_vec WHERE fact_id NOT IN (SELECT id FROM memory_facts)"
            )

        # lessons: keep top RETENTION_LESSONS_MAX by applied DESC, ts DESC
        max_lessons = settings.retention_lessons_max
        await self._db.execute(
            """DELETE FROM lessons WHERE id NOT IN (
                   SELECT id FROM lessons ORDER BY applied DESC, ts DESC LIMIT ?
               )""",
            (max_lessons,),
        )

        await self._db.commit()

        # Rebuild FTS indexes after deletes
        try:
            await self._db.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
            await self._db.execute("INSERT INTO lessons_fts(lessons_fts) VALUES ('rebuild')")
            await self._db.commit()
        except Exception as exc:
            log.warning("fts_rebuild_failed", error=str(exc))

        # WAL checkpoint
        await self._db.execute("PRAGMA wal_checkpoint(PASSIVE)")

        # Monthly VACUUM
        async with self._db.execute(
            "SELECT value FROM _meta WHERE key='last_vacuum_ts'"
        ) as cur:
            row = await cur.fetchone()
        last_vacuum = float(row[0]) if row else 0.0
        if now - last_vacuum > _VACUUM_INTERVAL_S:
            await self._db.execute("VACUUM")
            await self._db.execute(
                "INSERT OR REPLACE INTO _meta(key,value) VALUES('last_vacuum_ts',?)",
                (str(now),),
            )
            await self._db.commit()
            log.info("sqlite_vacuumed")

        # Record cleanup time
        self._last_cleanup_ts = now
        await self._db.execute(
            "INSERT OR REPLACE INTO _meta(key,value) VALUES('last_cleanup_ts',?)",
            (str(now),),
        )
        await self._db.commit()
        log.info("sqlite_cleanup_done")

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def heartbeat(self) -> None:
        """Periodic maintenance: checkpoint WAL, run cleanup if due."""
        if not self._db:
            return
        await self._db.execute("PRAGMA wal_checkpoint(PASSIVE)")

        now = time.time()
        if now - self._last_cleanup_ts >= _CLEANUP_INTERVAL_S:
            try:
                await self._cleanup()
            except Exception as exc:
                log.warning("sqlite_cleanup_error", error=str(exc))


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
