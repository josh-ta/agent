from __future__ import annotations

from contextlib import asynccontextmanager
from types import ModuleType
from types import SimpleNamespace
import sys

import pytest

from agent.config import settings
from agent.memory.postgres_store import PostgresStore
from agent.memory.postgres_store import _embed


class _FakeConn:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, tuple]] = []
        self.execute_results: list[str] = []
        self.fetch_results: list[list] = []
        self.fetchrow_results: list[dict | None] = []

    async def execute(self, query: str, *args):
        self.execute_calls.append((query, args))
        return self.execute_results.pop(0) if self.execute_results else "UPDATE 1"

    async def fetch(self, query: str, *args):
        return self.fetch_results.pop(0) if self.fetch_results else []

    async def fetchrow(self, query: str, *args):
        return self.fetchrow_results.pop(0) if self.fetchrow_results else {"n": 1}


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn
        self.closed = False

    @asynccontextmanager
    async def acquire(self):
        yield self._conn

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_embed_returns_none_when_embeddings_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "")

    assert await _embed("hello") is None


@pytest.mark.asyncio
async def test_embed_returns_embedding_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "token")
    module = ModuleType("openai")

    class _Client:
        def __init__(self, api_key: str) -> None:
            assert api_key == "token"
            self.embeddings = self

        async def create(self, *, model: str, input: str, encoding_format: str):
            assert input == "hello"
            assert encoding_format == "float"
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

    module.AsyncOpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", module)

    assert await _embed("hello") == [0.1, 0.2]


@pytest.mark.asyncio
async def test_embed_returns_none_on_openai_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "token")
    module = ModuleType("openai")

    class _Client:
        def __init__(self, api_key: str) -> None:
            self.embeddings = self

        async def create(self, *, model: str, input: str, encoding_format: str):
            raise RuntimeError("boom")

    module.AsyncOpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", module)

    assert await _embed("hello") is None


@pytest.mark.asyncio
async def test_postgres_store_init_enables_vector_and_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn()
    pool = _FakePool(conn)
    store = PostgresStore("postgresql://example")

    async def fake_create_pool(url: str, **kwargs):
        assert url == "postgresql://example"
        return pool

    monkeypatch.setattr(settings, "openai_api_key", "token")
    monkeypatch.setattr("agent.memory.postgres_store.asyncpg.create_pool", fake_create_pool)

    await store.init()

    assert store._pool is pool
    assert store._has_vector is True
    assert store._has_embeddings is True
    assert conn.execute_calls[0][0] == "CREATE EXTENSION IF NOT EXISTS vector"
    assert conn.execute_calls[1][0] == "CREATE EXTENSION IF NOT EXISTS pg_trgm"


@pytest.mark.asyncio
async def test_postgres_store_init_skips_vector_when_extension_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Conn(_FakeConn):
        async def execute(self, query: str, *args):
            self.execute_calls.append((query, args))
            if query == "CREATE EXTENSION IF NOT EXISTS vector":
                raise RuntimeError("no permission")
            return "OK"

    conn = _Conn()
    pool = _FakePool(conn)
    store = PostgresStore("postgresql://example")

    monkeypatch.setattr(settings, "openai_api_key", "token")
    async def fake_create_pool(*args, **kwargs):
        return pool

    monkeypatch.setattr("agent.memory.postgres_store.asyncpg.create_pool", fake_create_pool)

    await store.init()

    assert store._has_vector is False
    assert store._has_embeddings is False


@pytest.mark.asyncio
async def test_postgres_store_init_skips_embedding_index_when_column_setup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Conn(_FakeConn):
        async def execute(self, query: str, *args):
            self.execute_calls.append((query, args))
            if "ALTER TABLE shared_memory ADD COLUMN embedding" in query:
                raise RuntimeError("vector ddl failed")
            return "OK"

    conn = _Conn()
    pool = _FakePool(conn)
    store = PostgresStore("postgresql://example")

    monkeypatch.setattr(settings, "openai_api_key", "token")
    async def fake_create_pool(*args, **kwargs):
        return pool

    monkeypatch.setattr("agent.memory.postgres_store.asyncpg.create_pool", fake_create_pool)

    await store.init()

    assert store._has_vector is True
    assert store._has_embeddings is False


@pytest.mark.asyncio
async def test_postgres_store_close_closes_pool() -> None:
    conn = _FakeConn()
    pool = _FakePool(conn)
    store = PostgresStore("postgresql://example")
    store._pool = pool  # type: ignore[assignment]

    await store.close()

    assert pool.closed is True


@pytest.mark.asyncio
async def test_postgres_store_formats_agent_list() -> None:
    conn = _FakeConn()
    conn.fetch_results.append(
        [{"name": "helper", "status": "online", "model": "gpt-4o", "last_seen": SimpleNamespace(strftime=lambda _: "2026-03-16 10:00")}]
    )
    store = PostgresStore("postgresql://example")
    store._pool = _FakePool(conn)  # type: ignore[assignment]

    rendered = await store.list_agents()

    assert "helper [online]" in rendered


@pytest.mark.asyncio
async def test_postgres_store_complete_task_handles_missing_row() -> None:
    conn = _FakeConn()
    conn.execute_results.append("UPDATE 0")
    store = PostgresStore("postgresql://example")
    store._pool = _FakePool(conn)  # type: ignore[assignment]

    rendered = await store.complete_task("task-12345678", "done")

    assert "not found" in rendered


@pytest.mark.asyncio
async def test_postgres_store_search_shared_memory_falls_back_to_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn()
    conn.fetch_results.append(
        [{"agent_id": "helper", "content": "Parser fix note", "created_at": SimpleNamespace(strftime=lambda _: "2026-03-16 10:00")}]
    )
    store = PostgresStore("postgresql://example")
    store._pool = _FakePool(conn)  # type: ignore[assignment]
    store._has_embeddings = False

    rendered = await store.search_shared_memory("Parser")

    assert "Parser fix note" in rendered


@pytest.mark.asyncio
async def test_postgres_store_log_task_start_delegates_to_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    store = PostgresStore("postgresql://example")
    store._pool = object()  # type: ignore[assignment]
    captured = {}

    async def _log_event(event_type: str, payload: dict) -> None:
        captured["event_type"] = event_type
        captured["payload"] = payload

    monkeypatch.setattr(store.audit, "log_event", _log_event)

    await store.log_task_start("Investigate failures", "discord", "smart")

    assert captured["event_type"] == "task_start"
    assert captured["payload"]["tier"] == "smart"


@pytest.mark.asyncio
async def test_postgres_store_wrappers_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    store = PostgresStore("postgresql://example")
    store._pool = object()  # type: ignore[assignment]
    calls: list[tuple[str, tuple]] = []

    async def record(name: str, *args):
        calls.append((name, args))
        if name in {"list_agents", "create_task", "get_my_tasks", "complete_task", "broadcast_message", "read_broadcasts", "share_memory", "search_shared_memory"}:
            return f"{name}-ok"
        if name == "get_pending_task_rows":
            return [{"id": "task-1"}]
        if name == "get_stats":
            return {"agents": 1}
        if name == "_embed":
            return [0.5]
        return None

    monkeypatch.setattr(store.registry, "register_agent", lambda: record("register_agent"))
    monkeypatch.setattr(store.registry, "set_offline", lambda: record("set_offline"))
    monkeypatch.setattr(store.registry, "list_agents", lambda: record("list_agents"))
    monkeypatch.setattr(store.tasks, "create_task", lambda *args: record("create_task", *args))
    monkeypatch.setattr(store.tasks, "get_my_tasks", lambda: record("get_my_tasks"))
    monkeypatch.setattr(store.tasks, "get_pending_task_rows", lambda: record("get_pending_task_rows"))
    monkeypatch.setattr(store.tasks, "mark_task_running", lambda *args: record("mark_task_running", *args))
    monkeypatch.setattr(store.tasks, "complete_task", lambda *args: record("complete_task", *args))
    monkeypatch.setattr(store.audit, "broadcast_message", lambda *args: record("broadcast_message", *args))
    monkeypatch.setattr(store.audit, "read_broadcasts", lambda *args: record("read_broadcasts", *args))
    monkeypatch.setattr(store.audit, "log_event", lambda *args: record("log_event", *args))
    monkeypatch.setattr(store.audit, "log_task_done", lambda *args: record("log_task_done", *args))
    monkeypatch.setattr(store.shared_memory_repo, "share_memory", lambda *args: record("share_memory", *args))
    monkeypatch.setattr(store.shared_memory_repo, "search_shared_memory", lambda *args: record("search_shared_memory", *args))
    monkeypatch.setattr(store.maintenance, "heartbeat", lambda: record("heartbeat"))
    monkeypatch.setattr(store.maintenance, "cleanup", lambda: record("cleanup"))
    monkeypatch.setattr(store.maintenance, "get_stats", lambda: record("get_stats"))
    monkeypatch.setattr("agent.memory.postgres_store._embed", lambda text: record("_embed", text))

    assert await store.list_agents() == "list_agents-ok"
    assert await store.create_task("peer", "check build") == "create_task-ok"
    assert await store.get_my_tasks() == "get_my_tasks-ok"
    assert await store.get_pending_task_rows() == [{"id": "task-1"}]
    await store.mark_task_running("task-1")
    assert await store.complete_task("task-1", "done") == "complete_task-ok"
    assert await store.broadcast_message("hello") == "broadcast_message-ok"
    assert await store.read_broadcasts() == "read_broadcasts-ok"
    assert await store.share_memory("fact") == "share_memory-ok"
    assert await store.search_shared_memory("fact") == "search_shared_memory-ok"
    await store.log_event("evt", {"ok": True})
    await store.log_task_done("fact", True, 1.2, 3)
    await store.heartbeat()
    await store._cleanup()
    assert await store.get_stats() == {"agents": 1}
    assert await store._embed("hello") == [0.5]

    method_names = [name for name, _ in calls]
    assert method_names == [
        "list_agents",
        "create_task",
        "get_my_tasks",
        "get_pending_task_rows",
        "mark_task_running",
        "complete_task",
        "broadcast_message",
        "read_broadcasts",
        "share_memory",
        "search_shared_memory",
        "log_event",
        "log_task_done",
        "heartbeat",
        "cleanup",
        "get_stats",
        "_embed",
    ]


@pytest.mark.asyncio
async def test_postgres_store_register_and_set_offline_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    store = PostgresStore("postgresql://example")
    calls: list[str] = []

    async def _register() -> None:
        calls.append("register")

    async def _offline() -> None:
        calls.append("offline")

    monkeypatch.setattr(store.registry, "register_agent", _register)
    monkeypatch.setattr(store.registry, "set_offline", _offline)

    await store.register_agent()
    await store.set_offline()

    assert calls == ["register", "offline"]
