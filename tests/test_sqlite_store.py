from __future__ import annotations

import pytest

from agent.loop import Task, TaskResult


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
