from __future__ import annotations

import aiosqlite
import pytest

from agent.loop import Task, TaskResult
from agent.memory.sqlite_store import SQLiteStore


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_saves_messages_and_returns_history(sqlite_store) -> None:
    await sqlite_store.save_message("user", "hello", channel_id=7)
    await sqlite_store.save_message("assistant", "hi there", channel_id=7)

    history = await sqlite_store.get_history(channel_id=7)

    assert [row["content"] for row in history] == ["hello", "hi there"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_records_tasks_memory_and_lessons(sqlite_store) -> None:
    task = Task(content="Fix tests", source="discord", author="Josh")
    result = TaskResult(task=task, output="done", success=True, elapsed_ms=12.0, tool_calls=3)

    await sqlite_store.record_task(task, result)
    await sqlite_store.save_memory_fact("The parser needs a fixture.")
    await sqlite_store.save_lesson("Use parametrized tests for routing.", kind="pattern")

    memory_results = await sqlite_store.search_memory("parser")
    lessons = await sqlite_store.search_lessons("routing")
    stats = await sqlite_store.get_stats()

    assert "parser" in memory_results.lower()
    assert "routing" in lessons.lower()
    assert stats["tasks"] == 1
    assert stats["memory_facts"] == 1
    assert stats["lessons"] == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_persists_sessions_turns_and_checkpoints(sqlite_store) -> None:
    await sqlite_store.ensure_session(
        session_id="discord:101:55",
        source="discord",
        channel_id=101,
        title="Investigate flaky test",
        pending_task_id="task-55",
        metadata={"author": "Josh"},
    )
    await sqlite_store.append_session_turn(
        session_id="discord:101:55",
        role="user",
        content="Investigate the flaky integration test",
        task_id="task-55",
    )
    await sqlite_store.append_session_turn(
        session_id="discord:101:55",
        role="assistant",
        content="Which environment should I use?",
        turn_kind="question",
        task_id="task-55",
    )
    await sqlite_store.save_task_checkpoint(
        task_id="task-55",
        session_id="discord:101:55",
        summary="Waiting on environment clarification.",
        notes="Collected initial failures.",
    )

    session = await sqlite_store.get_session("discord:101:55")
    turns = await sqlite_store.list_session_turns("discord:101:55")
    context = await sqlite_store.get_session_context("discord:101:55")
    checkpoint = await sqlite_store.get_task_checkpoint("task-55")

    assert session is not None
    assert session["pending_task_id"] == "task-55"
    assert len(turns) == 2
    assert "Latest user goal" in session["summary"]
    assert "Session summary" in context
    assert checkpoint is not None
    assert checkpoint["summary"] == "Waiting on environment clarification."


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_persists_api_task_lifecycle(sqlite_store) -> None:
    await sqlite_store.create_task_record(
        task_id="task-123",
        source="api",
        author="tester",
        content="Ship the API",
    )
    queued = await sqlite_store.get_task_record("task-123")
    assert queued is not None
    assert queued["status"] == "queued"

    await sqlite_store.mark_task_running("task-123")
    running = await sqlite_store.get_task_record("task-123")
    assert running is not None
    assert running["status"] == "running"
    assert running["started_ts"] is not None

    task = Task(content="Ship the API", source="api", author="tester", metadata={"task_id": "task-123"})
    result = TaskResult(task=task, output="done", success=True, elapsed_ms=42.0, tool_calls=4)
    await sqlite_store.record_task(task, result)

    finished = await sqlite_store.get_task_record("task-123")
    assert finished is not None
    assert finished["status"] == "succeeded"
    assert finished["result"] == "done"
    assert finished["error"] is None
    assert finished["tool_calls"] == 4
    assert finished["finished_ts"] is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_lists_waiting_tasks(sqlite_store) -> None:
    await sqlite_store.create_task_record(
        task_id="task-wait",
        source="discord",
        author="tester",
        content="Need clarification",
        metadata={"task_id": "task-wait"},
    )
    await sqlite_store.mark_task_waiting(
        "task-wait",
        metadata={
            "task_id": "task-wait",
            "wait_state": {
                "question": "Which environment?",
                "timeout_s": 90,
                "channel_id": 101,
                "message_id": 55,
                "prompt_message_id": 66,
            },
        },
        question="Which environment?",
    )

    rows = await sqlite_store.list_waiting_task_records()

    assert len(rows) == 1
    assert rows[0]["task_id"] == "task-wait"
    assert rows[0]["metadata"]["wait_state"]["prompt_message_id"] == 66


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_cleanup_applies_retention(monkeypatch: pytest.MonkeyPatch, sqlite_store) -> None:
    monkeypatch.setattr("agent.memory.sqlite_components.settings.retention_memory_facts_max", 1)
    monkeypatch.setattr("agent.memory.sqlite_components.settings.retention_lessons_max", 1)
    monkeypatch.setattr("agent.memory.sqlite_components.settings.retention_tasks_days", 0)
    monkeypatch.setattr("agent.memory.sqlite_components.settings.retention_conversations_days", 0)

    await sqlite_store.save_message("user", "first", channel_id=1)
    await sqlite_store.save_message("assistant", "second", channel_id=1)
    await sqlite_store.record_task(
        Task(content="old"),
        TaskResult(task=Task(content="old"), output="done", success=True, elapsed_ms=1.0),
    )
    await sqlite_store.save_memory_fact("fact one")
    await sqlite_store.save_memory_fact("fact two")
    await sqlite_store.save_lesson("lesson one")
    await sqlite_store.save_lesson("lesson two")

    await sqlite_store._cleanup()
    stats = await sqlite_store.get_stats()

    assert stats["memory_facts"] == 1
    assert stats["lessons"] == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_init_migrates_legacy_tasks_before_creating_task_id_index(
    isolated_paths,
) -> None:
    db_path = isolated_paths["workspace"] / "legacy.db"
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(
            """
            CREATE TABLE tasks (
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
            """
        )
        await db.commit()

    store = SQLiteStore(db_path)
    await store.init()
    try:
        await store.create_task_record(
            task_id="task-legacy",
            source="api",
            author="tester",
            content="Verify migration",
        )
        row = await store.get_task_record("task-legacy")
        assert row is not None
        assert row["status"] == "queued"
    finally:
        await store.close()
