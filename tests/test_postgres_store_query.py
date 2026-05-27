"""Tests for PostgresStore read-only query methods."""

from __future__ import annotations

import pytest

from agent.memory.postgres_store import PostgresStore


class _FakeConn:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = rows or []

    async def fetch(self, sql: str, *args: object):
        if "information_schema.tables" in sql:
            return [{"table_schema": "public", "table_name": "events", "table_type": "BASE TABLE"}]
        return self.rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def acquire(self):
        return self._conn


@pytest.mark.asyncio
async def test_query_readonly_formats_rows() -> None:
    store = PostgresStore("postgresql://u:p@localhost/db")
    store._pool = _FakePool(_FakeConn([{"id": 1, "title": "Show"}]))  # type: ignore[assignment]

    out = await store.query_readonly("SELECT id, title FROM events", output_format="csv")
    assert "id,title" in out
    assert "1,Show" in out
    assert "(1 row(s)" in out


@pytest.mark.asyncio
async def test_query_readonly_writes_csv_to_output_path(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.config import settings

    monkeypatch.setattr(settings, "workspace_path", tmp_path)
    store = PostgresStore("postgresql://u:p@localhost/db")
    store._pool = _FakePool(_FakeConn([{"id": 1, "title": "Show"}]))  # type: ignore[assignment]

    out = await store.query_readonly(
        "SELECT id, title FROM events",
        output_format="csv",
        output_path="export.csv",
    )

    export = tmp_path / "export.csv"
    assert export.exists()
    assert export.read_text(encoding="utf-8") == "id,title\n1,Show"
    assert "Exported CSV to export.csv" in out
    assert "discord_attachment:export.csv" in out
    assert "Written" in out


@pytest.mark.asyncio
async def test_query_readonly_rejects_mutating_sql() -> None:
    store = PostgresStore("postgresql://u:p@localhost/db")
    store._pool = _FakePool(_FakeConn())  # type: ignore[assignment]

    out = await store.query_readonly("DELETE FROM events")
    assert out.startswith("[ERROR:")


@pytest.mark.asyncio
async def test_list_tables() -> None:
    store = PostgresStore("postgresql://u:p@localhost/db")
    store._pool = _FakePool(_FakeConn())  # type: ignore[assignment]

    out = await store.list_tables()
    assert "events" in out
