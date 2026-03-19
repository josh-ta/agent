from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import agent.task_waits as task_waits
from agent.task_waits import TaskWaitRegistry, current_task_wait_context, task_wait_context


def _suspend(
    registry: TaskWaitRegistry,
    *,
    task_id: str,
    channel_id: int = 10,
    prompt_message_id: int | None = None,
    created_at: datetime | None = None,
) -> None:
    suspended = registry.suspend(
        task_id=task_id,
        source="discord",
        author="tester",
        content="do thing",
        channel_id=channel_id,
        message_id=1,
        metadata={"task_id": task_id, "wait_state": {"question": "Which env?"}},
        question="Which env?",
        timeout_s=60,
        base_prompt="prompt",
        tier="smart",
    )
    if prompt_message_id is not None:
        registry.bind_prompt_message(task_id, prompt_message_id)
    if created_at is not None:
        suspended.created_at = created_at


def test_task_wait_context_sets_and_resets_on_success() -> None:
    assert current_task_wait_context() is None

    with task_wait_context(task_id="task-1", source="discord", channel_id=99):
        context = current_task_wait_context()
        assert context is not None
        assert context.task_id == "task-1"
        assert context.source == "discord"
        assert context.channel_id == 99

    assert current_task_wait_context() is None


def test_task_wait_context_resets_after_exception() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with task_wait_context(task_id="task-1", source="api", channel_id=0):
            raise RuntimeError("boom")

    assert current_task_wait_context() is None


def test_ensure_task_id_reuses_existing_value() -> None:
    metadata = {"task_id": "existing"}

    task_id = TaskWaitRegistry.ensure_task_id(metadata)

    assert task_id == "existing"
    assert metadata["task_id"] == "existing"


def test_ensure_task_id_creates_and_mutates_missing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task_waits, "uuid4", lambda: "generated-id")
    metadata: dict[str, str] = {}

    task_id = TaskWaitRegistry.ensure_task_id(metadata)

    assert task_id == "generated-id"
    assert metadata["task_id"] == "generated-id"


def test_find_and_pop_for_discord_reply_prefers_prompt_message_match() -> None:
    registry = TaskWaitRegistry()
    _suspend(registry, task_id="task-1", prompt_message_id=111)
    _suspend(registry, task_id="task-2", prompt_message_id=222)

    found = registry.find_for_discord_reply(channel_id=10, reference_message_id=222)
    popped = registry.pop_for_discord_reply(channel_id=10, reference_message_id=222)

    assert found is not None
    assert found.task_id == "task-2"
    assert popped is not None
    assert popped.task_id == "task-2"
    assert registry.get("task-2") is None


def test_find_for_discord_reply_uses_singleton_pending_fallback() -> None:
    registry = TaskWaitRegistry()
    _suspend(registry, task_id="task-1")

    found = registry.find_for_discord_reply(channel_id=10, reference_message_id=None)

    assert found is not None
    assert found.task_id == "task-1"


def test_find_for_discord_reply_returns_none_when_pending_waits_are_ambiguous() -> None:
    registry = TaskWaitRegistry()
    _suspend(registry, task_id="task-1")
    _suspend(registry, task_id="task-2")

    found = registry.find_for_discord_reply(channel_id=10, reference_message_id=None)

    assert found is None


def test_bind_prompt_message_missing_task_and_nonexpired_wait() -> None:
    registry = TaskWaitRegistry()
    registry.bind_prompt_message("missing", 123)

    now = datetime.now(UTC)
    _suspend(registry, task_id="task-1", created_at=now)

    assert registry.list_expired(now=now) == []


def test_list_expired_uses_minimum_timeout_floor_of_one_second() -> None:
    registry = TaskWaitRegistry()
    now = datetime.now(UTC)
    _suspend(registry, task_id="task-1", created_at=now - timedelta(seconds=2))
    registry.get("task-1").timeout_s = 0  # type: ignore[union-attr]

    expired = registry.list_expired(now=now)

    assert [item.task_id for item in expired] == ["task-1"]


def test_build_resumed_metadata_preserves_context_and_clears_wait_state() -> None:
    registry = TaskWaitRegistry()
    _suspend(registry, task_id="task-1")
    suspended = registry.get("task-1")
    assert suspended is not None
    suspended.metadata["resume_context"] = {"prior": "value"}

    metadata = registry.build_resumed_metadata(
        suspended,
        answer="production",
        resumed_from="api",
    )

    assert metadata["task_id"] == "task-1"
    assert "wait_state" not in metadata
    assert metadata["resume_context"] == {
        "prior": "value",
        "question": "Which env?",
        "answer": "production",
        "resumed_from": "api",
        "suspended_task_id": "task-1",
    }
