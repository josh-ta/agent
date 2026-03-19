from __future__ import annotations

import json
import sys
import time
from types import ModuleType
from types import SimpleNamespace

import pytest

import agent.memory.sqlite_components as sqlite_components
from agent.loop import Task, TaskResult


class _ExecuteFailProxy:
    def __init__(self, db, *, needle: str) -> None:
        self._db = db
        self._needle = needle
        self._failed = False

    def execute(self, sql: str, *args, **kwargs):
        if self._needle in sql and not self._failed:
            self._failed = True
            raise RuntimeError(f"forced failure for {self._needle}")
        return self._db.execute(sql, *args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._db, name)


class _ExecuteRollbackFailProxy:
    def __init__(self, db, *, needle: str) -> None:
        self._db = db
        self._needle = needle
        self._failed = False

    def execute(self, sql: str, *args, **kwargs):
        if self._needle in sql and not self._failed:
            self._failed = True
            raise RuntimeError(f"forced failure for {self._needle}")
        return self._db.execute(sql, *args, **kwargs)

    async def rollback(self) -> None:
        raise RuntimeError("rollback failed")

    def __getattr__(self, name: str):
        return getattr(self._db, name)


class _RowsCursor:
    def __init__(self, rows) -> None:
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def fetchall(self):
        return self._rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None


class _ExecuteRowsProxy:
    def __init__(self, db, *, needle: str, rows=None, error: str | None = None) -> None:
        self._db = db
        self._needle = needle
        self._rows = rows or []
        self._error = error

    def execute(self, sql: str, *args, **kwargs):
        if self._needle in sql:
            if self._error is not None:
                raise RuntimeError(self._error)
            return _RowsCursor(self._rows)
        return self._db.execute(sql, *args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._db, name)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_task_repository_record_task_without_task_id_stores_failed_error(sqlite_store) -> None:
    task = Task(content="broken task", source="api", author="tester", metadata={})
    result = TaskResult(
        task=task,
        output="boom",
        success=False,
        elapsed_ms=5.0,
        status="failed",
        tool_calls=2,
    )

    await sqlite_store.tasks.record_task(task, result)

    assert sqlite_store._db is not None
    async with sqlite_store._db.execute(
        "SELECT status, result, error, success, tool_calls FROM tasks ORDER BY id DESC LIMIT 1"
    ) as cur:
        row = await cur.fetchone()

    assert row["status"] == "failed"
    assert row["result"] is None
    assert row["error"] == "boom"
    assert row["success"] == 0
    assert row["tool_calls"] == 2


@pytest.mark.asyncio
@pytest.mark.integration
async def test_mark_task_queued_preserves_existing_metadata_when_none_passed(sqlite_store) -> None:
    await sqlite_store.create_task_record(
        task_id="task-1",
        source="api",
        author="tester",
        content="do thing",
        metadata={"custom": "value"},
    )
    await sqlite_store.mark_task_running("task-1")

    await sqlite_store.tasks.mark_task_queued("task-1", metadata=None)
    row = await sqlite_store.get_task_record("task-1")

    assert row is not None
    assert row["status"] == "queued"
    assert row["metadata"] == {"custom": "value"}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_task_repository_handles_malformed_metadata_json(sqlite_store) -> None:
    assert sqlite_store._db is not None
    now = time.time()
    await sqlite_store._db.execute(
        """INSERT INTO tasks
           (task_id, source, author, content, status, metadata, success, created_ts, updated_ts, ts)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        ("task-bad", "api", "tester", "do thing", "waiting_for_user", "{bad", 0, now, now, now),
    )
    await sqlite_store._db.commit()

    row = await sqlite_store.tasks.get_task_record("task-bad")
    waiting = await sqlite_store.tasks.list_waiting_task_records()

    assert row is not None
    assert row["metadata"] == {}
    assert waiting[0]["metadata"] == {}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_save_memory_fact_tolerates_embedding_failures(monkeypatch: pytest.MonkeyPatch, sqlite_store) -> None:
    sqlite_store._has_vec = True

    async def fail_embed(_text: str):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(sqlite_components, "embed_text", fail_embed)

    fact_id = await sqlite_store.memory_facts.save_memory_fact("Remember this fact")

    assert fact_id > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_memory_falls_back_to_like_when_fts_unavailable(sqlite_store) -> None:
    await sqlite_store.save_memory_fact("Parser needs fixture")
    original_db = sqlite_store._db
    assert original_db is not None
    sqlite_store._db = _ExecuteFailProxy(original_db, needle="FROM memory_fts")

    try:
        result = await sqlite_store.memory_facts.search_memory("Parser")
    finally:
        sqlite_store._db = original_db

    assert "Parser needs fixture" in result


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_memory_returns_no_matches_message(sqlite_store) -> None:
    result = await sqlite_store.memory_facts.search_memory("missing")

    assert result == "(no memory matches for: missing)"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_memory_vector_paths(monkeypatch: pytest.MonkeyPatch, sqlite_store) -> None:
    sqlite_store._has_vec = True

    async def fake_embed(_text: str):
        return [0.1, 0.2]

    monkeypatch.setattr(sqlite_components, "embed_text", fake_embed)
    original_db = sqlite_store._db
    assert original_db is not None

    sqlite_store._db = _ExecuteRowsProxy(
        original_db,
        needle="FROM memory_vec",
        rows=[{"content": "Vector fact", "ts": time.time(), "distance": 0.1}],
    )
    try:
        result = await sqlite_store.memory_facts.search_memory("Vector")
    finally:
        sqlite_store._db = original_db

    assert "Vector fact" in result


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_memory_vector_fallbacks_to_fts_and_like(monkeypatch: pytest.MonkeyPatch, sqlite_store) -> None:
    sqlite_store._has_vec = True
    await sqlite_store.save_memory_fact("Parser needs fixture")

    async def fake_embed(_text: str):
        return [0.1, 0.2]

    monkeypatch.setattr(sqlite_components, "embed_text", fake_embed)
    original_db = sqlite_store._db
    assert original_db is not None

    sqlite_store._db = _ExecuteRowsProxy(original_db, needle="FROM memory_vec", rows=[])
    try:
        fts_result = await sqlite_store.memory_facts.search_memory("Parser")
    finally:
        sqlite_store._db = original_db
    assert "Parser needs fixture" in fts_result

    sqlite_store._db = _ExecuteRowsProxy(original_db, needle="FROM memory_vec", error="vec failed")
    try:
        like_result = await sqlite_store.memory_facts.search_memory("Parser")
    finally:
        sqlite_store._db = original_db
    assert "Parser needs fixture" in like_result


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_memory_falls_back_when_embedding_is_none(monkeypatch: pytest.MonkeyPatch, sqlite_store) -> None:
    sqlite_store._has_vec = True
    await sqlite_store.save_memory_fact("Parser needs fixture")

    async def fake_embed(_text: str):
        return None

    monkeypatch.setattr(sqlite_components, "embed_text", fake_embed)

    result = await sqlite_store.memory_facts.search_memory("Parser")

    assert "Parser needs fixture" in result


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_lessons_falls_back_and_increments_applied(sqlite_store) -> None:
    lesson_id = await sqlite_store.lessons.save_lesson("Use routing fixtures", kind="pattern")
    original_db = sqlite_store._db
    assert original_db is not None
    sqlite_store._db = _ExecuteFailProxy(original_db, needle="FROM lessons_fts")

    try:
        result = await sqlite_store.lessons.search_lessons("routing")
    finally:
        sqlite_store._db = original_db

    assert "Relevant past lessons" in result
    async with original_db.execute("SELECT applied FROM lessons WHERE id=?", (lesson_id,)) as cur:
        row = await cur.fetchone()
    assert row["applied"] == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_lessons_returns_empty_string_and_truncates(sqlite_store) -> None:
    assert await sqlite_store.lessons.search_lessons("missing") == ""

    for idx in range(10):
        await sqlite_store.lessons.save_lesson(f"Use routing fixtures {'x' * 200} {idx}", kind="pattern")

    result = await sqlite_store.lessons.search_lessons("routing", limit=10)

    assert result.startswith("## Relevant past lessons:")
    assert len(result) < 1500


@pytest.mark.asyncio
@pytest.mark.integration
async def test_save_memory_fact_vector_success_and_stats_fallback(monkeypatch: pytest.MonkeyPatch, sqlite_store) -> None:
    sqlite_store._has_vec = True
    assert sqlite_store._db is not None
    await sqlite_store._db.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_vec (
            fact_id INTEGER PRIMARY KEY,
            embedding TEXT
        )
        """
    )
    await sqlite_store._db.commit()

    async def fake_embed(_text: str):
        return [0.1] * 1536

    monkeypatch.setattr(sqlite_components, "embed_text", fake_embed)
    fact_id = await sqlite_store.memory_facts.save_memory_fact("Remember this fact")

    async with sqlite_store._db.execute("SELECT embedding FROM memory_vec WHERE fact_id=?", (fact_id,)) as cur:
        row = await cur.fetchone()
    assert row is not None

    original_stat = type(sqlite_store._path).stat

    def _broken_stat(self):
        if self == sqlite_store._path:
            raise OSError("stat failed")
        return original_stat(self)

    monkeypatch.setattr(type(sqlite_store._path), "stat", _broken_stat)
    stats = await sqlite_store.maintenance.get_stats()
    assert stats["db_size_mb"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_save_memory_fact_skips_vector_insert_when_embedding_none(monkeypatch: pytest.MonkeyPatch, sqlite_store) -> None:
    sqlite_store._has_vec = True
    assert sqlite_store._db is not None
    await sqlite_store._db.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_vec (
            fact_id INTEGER PRIMARY KEY,
            embedding TEXT
        )
        """
    )
    await sqlite_store._db.commit()

    async def fake_embed(_text: str):
        return None

    monkeypatch.setattr(sqlite_components, "embed_text", fake_embed)
    fact_id = await sqlite_store.memory_facts.save_memory_fact("Remember this fact")

    async with sqlite_store._db.execute("SELECT embedding FROM memory_vec WHERE fact_id=?", (fact_id,)) as cur:
        row = await cur.fetchone()
    assert row is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_save_memory_fact_swallows_vector_insert_and_rollback_failures(
    monkeypatch: pytest.MonkeyPatch, sqlite_store
) -> None:
    original_db = sqlite_store._db
    assert original_db is not None
    sqlite_store._has_vec = True
    sqlite_store._db = _ExecuteRollbackFailProxy(original_db, needle="INSERT OR REPLACE INTO memory_vec")

    async def fake_embed(_text: str):
        return [0.1] * 1536

    monkeypatch.setattr(sqlite_components, "embed_text", fake_embed)
    try:
        fact_id = await sqlite_store.memory_facts.save_memory_fact("Remember this fact")
    finally:
        sqlite_store._db = original_db

    async with original_db.execute("SELECT content FROM memory_facts WHERE id=?", (fact_id,)) as cur:
        row = await cur.fetchone()
    assert row["content"] == "Remember this fact"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cleanup_updates_meta_even_when_fts_rebuild_fails(sqlite_store) -> None:
    original_db = sqlite_store._db
    assert original_db is not None
    sqlite_store._db = _ExecuteFailProxy(original_db, needle="INSERT INTO memory_fts(memory_fts)")

    try:
        await sqlite_store.maintenance.cleanup()
    finally:
        sqlite_store._db = original_db

    assert sqlite_store._last_cleanup_ts > 0
    async with original_db.execute(
        "SELECT value FROM _meta WHERE key='last_cleanup_ts'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cleanup_swallows_fts_rebuild_and_rollback_failures(sqlite_store) -> None:
    original_db = sqlite_store._db
    assert original_db is not None
    sqlite_store._db = _ExecuteRollbackFailProxy(original_db, needle="INSERT INTO memory_fts(memory_fts)")

    try:
        await sqlite_store.maintenance.cleanup()
    finally:
        sqlite_store._db = original_db

    assert sqlite_store._last_cleanup_ts > 0
    async with original_db.execute(
        "SELECT value FROM _meta WHERE key='last_cleanup_ts'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_heartbeat_swallows_cleanup_errors(monkeypatch: pytest.MonkeyPatch, sqlite_store) -> None:
    async def boom() -> None:
        raise RuntimeError("cleanup failed")

    sqlite_store._last_cleanup_ts = 0
    monkeypatch.setattr(sqlite_store.maintenance, "cleanup", boom)

    await sqlite_store.maintenance.heartbeat()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_maintenance_heartbeat_handles_uninitialized_and_not_due_paths(monkeypatch: pytest.MonkeyPatch, isolated_paths) -> None:
    from agent.memory.sqlite_store import SQLiteStore

    store = SQLiteStore(isolated_paths["workspace"] / "temp.db")
    await store.maintenance.heartbeat()

    await store.init()
    try:
        store._last_cleanup_ts = time.time()
        calls: list[str] = []

        async def fake_cleanup() -> None:
            calls.append("cleanup")

        monkeypatch.setattr(store.maintenance, "cleanup", fake_cleanup)
        await store.maintenance.heartbeat()
        assert calls == []
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cleanup_prunes_orphan_vectors_and_skips_recent_vacuum(monkeypatch: pytest.MonkeyPatch, sqlite_store) -> None:
    sqlite_store._has_vec = True
    assert sqlite_store._db is not None
    await sqlite_store._db.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_vec (
            fact_id INTEGER PRIMARY KEY,
            embedding TEXT
        )
        """
    )
    await sqlite_store._db.execute(
        "INSERT OR REPLACE INTO memory_vec(fact_id, embedding) VALUES (?, ?)",
        (999, json.dumps([0.1] * 1536)),
    )
    now = time.time()
    await sqlite_store._db.execute(
        "INSERT OR REPLACE INTO _meta(key,value) VALUES('last_vacuum_ts',?)",
        (str(now),),
    )
    await sqlite_store._db.commit()

    await sqlite_store.maintenance.cleanup()

    async with sqlite_store._db.execute("SELECT COUNT(*) as n FROM memory_vec WHERE fact_id=999") as cur:
        row = await cur.fetchone()
    assert row["n"] == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cleanup_enforces_per_channel_conversation_cap(monkeypatch: pytest.MonkeyPatch, sqlite_store) -> None:
    monkeypatch.setattr("agent.memory.sqlite_components.settings.retention_conversations_days", 365)

    for idx in range(1002):
        await sqlite_store.save_message("user", f"message-{idx}", channel_id=7)

    await sqlite_store.maintenance.cleanup()

    assert sqlite_store._db is not None
    async with sqlite_store._db.execute(
        "SELECT COUNT(*) as n FROM conversations WHERE channel_id=?",
        (7,),
    ) as cur:
        row = await cur.fetchone()
    assert row["n"] == 1000


@pytest.mark.asyncio
async def test_embed_text_returns_none_when_disabled_or_openai_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sqlite_components.settings, "openai_api_key", "")
    assert await sqlite_components.embed_text("hello") is None

    monkeypatch.setattr(sqlite_components.settings, "openai_api_key", "token")
    module = ModuleType("openai")

    class _Client:
        def __init__(self, api_key: str) -> None:
            self.embeddings = self

        async def create(self, *, model: str, input: str, encoding_format: str):
            raise RuntimeError("boom")

    module.AsyncOpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", module)

    assert await sqlite_components.embed_text("hello") is None


@pytest.mark.asyncio
async def test_embed_text_returns_embedding_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sqlite_components.settings, "openai_api_key", "token")
    module = ModuleType("openai")

    class _Client:
        def __init__(self, api_key: str) -> None:
            self.embeddings = self

        async def create(self, *, model: str, input: str, encoding_format: str):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

    module.AsyncOpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", module)

    assert await sqlite_components.embed_text("hello") == [0.1, 0.2]
