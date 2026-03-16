from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from agent.memory.postgres_components import (
    AgentRegistryRepository,
    AuditLogRepository,
    PostgresMaintenance,
    SharedMemoryRepository,
    SharedTaskRepository,
)
from agent.memory.postgres_store import PostgresStore


class _Conn:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, tuple]] = []
        self.fetch_results: list[list] = []
        self.fetchrow_results: list[dict | None] = []

    async def execute(self, query: str, *args):
        self.execute_calls.append((query, args))
        return "UPDATE 1"

    async def fetch(self, query: str, *args):
        self.execute_calls.append((query, args))
        return self.fetch_results.pop(0) if self.fetch_results else []

    async def fetchrow(self, query: str, *args):
        self.execute_calls.append((query, args))
        return self.fetchrow_results.pop(0) if self.fetchrow_results else {"n": 1}


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


@pytest.mark.asyncio
async def test_agent_registry_registers_and_sets_offline() -> None:
    conn = _Conn()
    store = PostgresStore("postgresql://example")
    store._pool = _Pool(conn)  # type: ignore[assignment]
    repo = AgentRegistryRepository(store)

    await repo.register_agent()
    await repo.set_offline()

    assert "INSERT INTO agents" in conn.execute_calls[0][0]
    assert "UPDATE agents SET status='offline'" in conn.execute_calls[1][0]


@pytest.mark.asyncio
async def test_shared_task_repository_formats_pending_tasks() -> None:
    conn = _Conn()
    conn.fetch_results.append(
        [
            {
                "id": "1234567890",
                "from_agent": "peer",
                "description": "check build",
                "status": "pending",
                "created_at": SimpleNamespace(strftime=lambda fmt: "2026-03-16 10:00"),
            }
        ]
    )
    store = PostgresStore("postgresql://example")
    store._pool = _Pool(conn)  # type: ignore[assignment]
    repo = SharedTaskRepository(store)

    rendered = await repo.get_my_tasks()

    assert "check build" in rendered
    assert "[12345678]" in rendered


@pytest.mark.asyncio
async def test_audit_log_repository_reads_broadcasts_and_task_done() -> None:
    conn = _Conn()
    conn.fetch_results.append(
        [
            {
                "agent_id": "peer",
                "payload": {"message": "hi"},
                "ts": SimpleNamespace(strftime=lambda fmt: "2026-03-16 10:00"),
            }
        ]
    )
    store = PostgresStore("postgresql://example")
    store._pool = _Pool(conn)  # type: ignore[assignment]
    repo = AuditLogRepository(store)

    rendered = await repo.read_broadcasts()
    await repo.log_task_done("task", True, 12.7, 3)

    assert "peer: hi" in rendered
    assert any("INSERT INTO audit_log" in call[0] for call in conn.execute_calls)


@pytest.mark.asyncio
async def test_shared_memory_repository_semantic_search_and_share(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _Conn()
    conn.fetchrow_results.append({"id": "abcdef1234"})
    conn.fetch_results.append(
        [
            {
                "agent_id": "peer",
                "content": "important fact",
                "created_at": SimpleNamespace(strftime=lambda fmt: "2026-03-16 10:00"),
                "similarity": 0.95,
            }
        ]
    )
    store = PostgresStore("postgresql://example")
    store._pool = _Pool(conn)  # type: ignore[assignment]
    store._has_embeddings = True

    async def fake_embed(text: str):
        return [0.1, 0.2]

    monkeypatch.setattr(store, "_embed", fake_embed)
    repo = SharedMemoryRepository(store)

    saved = await repo.share_memory("important fact")
    rendered = await repo.search_shared_memory("important")

    assert "abcdef12" in saved
    assert "sim=0.95" in rendered


@pytest.mark.asyncio
async def test_postgres_maintenance_heartbeat_and_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _Conn()
    conn.fetchrow_results.extend([{"n": 1}, {"n": 2}, {"n": 3}, {"n": 4}])
    store = PostgresStore("postgresql://example")
    store._pool = _Pool(conn)  # type: ignore[assignment]
    maintenance = PostgresMaintenance(store)

    await maintenance.cleanup()
    stats = await maintenance.get_stats()

    assert stats["agents"] == 1
    assert stats["shared_memory"] == 4
    assert store._last_cleanup_ts > 0
