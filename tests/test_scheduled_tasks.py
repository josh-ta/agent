"""SQLite scheduled_tasks persistence and claim semantics."""

from __future__ import annotations

import time

import pytest

from agent.memory.sqlite_store import SQLiteStore


@pytest.mark.asyncio
async def test_scheduled_task_lifecycle_and_claim(tmp_path) -> None:
    db = tmp_path / "s.db"
    store = SQLiteStore(db)
    await store.init()

    assert await store.scheduled_task_count() == 0
    tid = await store.scheduled_task_create(
        prompt="hello",
        delay_seconds=0,
        interval_seconds=None,
        metadata={"x": 1},
    )
    assert await store.scheduled_task_count() == 1

    now = time.time()
    claimed = await store.scheduled_tasks_claim_due(now=now + 1, limit=10)
    assert len(claimed) == 1
    assert claimed[0]["id"] == tid
    assert claimed[0]["prompt"] == "hello"
    assert claimed[0]["metadata"]["x"] == 1
    assert await store.scheduled_task_count() == 0

    rows = await store.scheduled_task_list(include_disabled=True)
    assert len(rows) == 1
    assert rows[0]["enabled"] == 0


@pytest.mark.asyncio
async def test_scheduled_task_recurring_advances_next_run(tmp_path) -> None:
    db = tmp_path / "r.db"
    store = SQLiteStore(db)
    await store.init()

    tid = await store.scheduled_task_create(
        prompt="beat",
        delay_seconds=0,
        interval_seconds=120.0,
    )
    t0 = time.time()
    claimed = await store.scheduled_tasks_claim_due(now=t0 + 5, limit=5)
    assert len(claimed) == 1
    rows = await store.scheduled_task_list(include_disabled=False)
    assert len(rows) == 1
    assert rows[0]["id"] == tid
    assert float(rows[0]["next_run_ts"]) >= t0 + 115


@pytest.mark.asyncio
async def test_scheduled_task_cancel(tmp_path) -> None:
    db = tmp_path / "c.db"
    store = SQLiteStore(db)
    await store.init()
    tid = await store.scheduled_task_create(prompt="x", delay_seconds=3600, interval_seconds=None)
    assert await store.scheduled_task_cancel(tid) is True
    assert await store.scheduled_task_cancel(tid) is False
    assert await store.scheduled_task_count() == 0
