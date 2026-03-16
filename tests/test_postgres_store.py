from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from agent.memory.postgres_store import PostgresStore


class _FakeConn:
    def __init__(self) -> None:
        self.execute_results: list[str] = []
        self.fetch_results: list[list] = []
        self.fetchrow_results: list[dict | None] = []

    async def execute(self, query: str, *args):
        return self.execute_results.pop(0) if self.execute_results else "UPDATE 1"

    async def fetch(self, query: str, *args):
        return self.fetch_results.pop(0) if self.fetch_results else []

    async def fetchrow(self, query: str, *args):
        return self.fetchrow_results.pop(0) if self.fetchrow_results else {"n": 1}


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self._conn


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
