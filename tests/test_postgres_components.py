from __future__ import annotations

from contextlib import asynccontextmanager
import time
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
        self.fetchval_result = 1

    async def execute(self, query: str, *args):
        self.execute_calls.append((query, args))
        return "UPDATE 1"

    async def fetch(self, query: str, *args):
        self.execute_calls.append((query, args))
        return self.fetch_results.pop(0) if self.fetch_results else []

    async def fetchrow(self, query: str, *args):
        self.execute_calls.append((query, args))
        return self.fetchrow_results.pop(0) if self.fetchrow_results else {"n": 1}

    async def fetchval(self, query: str, *args):
        self.execute_calls.append((query, args))
        return self.fetchval_result


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
async def test_shared_task_repository_handles_empty_and_no_pool_paths() -> None:
    conn = _Conn()
    conn.fetchrow_results.append(None)
    store = PostgresStore("postgresql://example")
    repo = SharedTaskRepository(store)

    assert await repo.get_pending_task_rows() == []
    await repo.mark_task_running("task-1")

    store._pool = _Pool(conn)  # type: ignore[assignment]
    assert await repo.create_task("peer", "check build") == "Task unknown created for peer."
    assert await repo.get_my_tasks() == "(no pending tasks)"


@pytest.mark.asyncio
async def test_shared_task_repository_returns_pending_rows_and_running_status() -> None:
    conn = _Conn()
    conn.fetch_results.append([{"id": "task-1", "from_agent": "peer", "description": "check build"}])
    conn.fetchrow_results.extend([{"id": "task-1"}, None])
    store = PostgresStore("postgresql://example")
    store._pool = _Pool(conn)  # type: ignore[assignment]
    repo = SharedTaskRepository(store)

    rows = await repo.get_pending_task_rows()
    first = await repo.mark_task_running("task-1")
    second = await repo.mark_task_running("task-1")

    assert rows == [{"id": "task-1", "from_agent": "peer", "description": "check build"}]
    assert first is True
    assert second is False


@pytest.mark.asyncio
async def test_shared_task_repository_completes_task_when_updated() -> None:
    conn = _Conn()
    store = PostgresStore("postgresql://example")
    store._pool = _Pool(conn)  # type: ignore[assignment]
    repo = SharedTaskRepository(store)

    rendered = await repo.complete_task("1234567890", "done")

    assert rendered == "Task 12345678 marked done."


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
async def test_audit_log_repository_handles_empty_and_error_tolerant_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    store = PostgresStore("postgresql://example")
    store._pool = _Pool(_Conn())  # type: ignore[assignment]
    repo = AuditLogRepository(store)

    assert await repo.read_broadcasts() == "(no broadcasts)"
    await repo.log_task_start("task", "discord", "smart")
    await repo.log_task_done("task", True, 1.2, 3)

    store._pool = object()  # type: ignore[assignment]

    async def _explode(*args, **kwargs) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(repo, "log_event", _explode)

    await repo.log_task_start("task", "discord", "smart")
    await repo.log_task_done("task", True, 1.2, 3)


@pytest.mark.asyncio
async def test_audit_log_repository_no_pool_short_circuits() -> None:
    repo = AuditLogRepository(PostgresStore("postgresql://example"))

    await repo.log_task_start("task", "discord", "smart")
    await repo.log_task_done("task", True, 1.2, 3)


@pytest.mark.asyncio
async def test_agent_registry_list_agents_empty_and_broadcast_message() -> None:
    conn = _Conn()
    store = PostgresStore("postgresql://example")
    store._pool = _Pool(conn)  # type: ignore[assignment]

    assert await AgentRegistryRepository(store).list_agents() == "(no agents registered)"
    result = await AuditLogRepository(store).broadcast_message("hello", event_type="status")
    assert result == "Broadcast sent [status]: hello"


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
async def test_shared_memory_repository_keyword_and_no_match_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _Conn()
    conn.fetchrow_results.append({"id": "abcdef1234"})
    conn.fetch_results.extend([
        [],
        [],
    ])
    store = PostgresStore("postgresql://example")
    store._pool = _Pool(conn)  # type: ignore[assignment]
    store._has_embeddings = True
    repo = SharedMemoryRepository(store)

    async def fake_embed(text: str):
        return None

    monkeypatch.setattr(store, "_embed", fake_embed)

    saved = await repo.share_memory("important fact", {"tag": "note"})
    rendered = await repo.search_shared_memory("missing")

    assert "abcdef12" in saved
    assert rendered == "(no shared memory matches for: missing)"


@pytest.mark.asyncio
async def test_shared_memory_repository_semantic_miss_falls_back_to_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _Conn()
    conn.fetchrow_results.append({"id": "abcdef1234"})
    conn.fetch_results.extend([
        [],
        [{"agent_id": "peer", "content": "important fact", "created_at": SimpleNamespace(strftime=lambda fmt: "2026-03-16 10:00")}],
    ])
    store = PostgresStore("postgresql://example")
    store._pool = _Pool(conn)  # type: ignore[assignment]
    store._has_embeddings = True

    async def fake_embed(text: str):
        return [0.1, 0.2]

    monkeypatch.setattr(store, "_embed", fake_embed)
    rendered = await SharedMemoryRepository(store).search_shared_memory("important")

    assert "important fact" in rendered


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


@pytest.mark.asyncio
async def test_postgres_maintenance_handles_no_pool_and_cleanup_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    store = PostgresStore("postgresql://example")
    maintenance = PostgresMaintenance(store)

    assert await maintenance.get_stats() == {}
    await maintenance.cleanup()
    await maintenance.heartbeat()

    conn = _Conn()
    store._pool = _Pool(conn)  # type: ignore[assignment]
    store._last_cleanup_ts = 0

    async def _explode() -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(maintenance, "cleanup", _explode)

    await maintenance.heartbeat()


@pytest.mark.asyncio
async def test_postgres_maintenance_heartbeat_logs_when_agent_update_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenConn(_Conn):
        async def execute(self, query: str, *args):
            if "UPDATE agents SET last_seen=NOW()" in query:
                raise RuntimeError("boom")
            return await super().execute(query, *args)

    conn = _BrokenConn()
    store = PostgresStore("postgresql://example")
    store._pool = _Pool(conn)  # type: ignore[assignment]
    maintenance = PostgresMaintenance(store)
    store._last_cleanup_ts = time.time()

    await maintenance.heartbeat()
