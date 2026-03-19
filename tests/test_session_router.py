from __future__ import annotations

from agent.session_router import SessionRouter, TurnIntent
from agent.task_waits import TaskWaitRegistry


def _suspend(
    registry: TaskWaitRegistry,
    *,
    task_id: str,
    channel_id: int = 10,
    prompt_message_id: int | None = None,
) -> None:
    registry.suspend(
        task_id=task_id,
        source="discord",
        author="tester",
        content="do thing",
        channel_id=channel_id,
        message_id=1,
        metadata={"task_id": task_id},
        question="Which env?",
        timeout_s=60,
        base_prompt="prompt",
        tier="smart",
    )
    if prompt_message_id is not None:
        registry.bind_prompt_message(task_id, prompt_message_id)


def test_build_session_prefers_existing_session_id() -> None:
    router = SessionRouter()

    session = router.build_session(
        source="discord",
        channel_id=123,
        message_id=456,
        metadata={"session_id": "existing-session", "thread_key": "saved-thread"},
    )

    assert session.session_id == "existing-session"
    assert session.thread_key == "saved-thread"
    assert session.channel_id == 123


def test_build_session_uses_discord_reference_anchor() -> None:
    router = SessionRouter()

    session = router.build_session(
        source="discord",
        channel_id=123,
        message_id=999,
        reference_message_id=555,
    )

    assert session.session_id == "discord:123:555"
    assert session.thread_key == "123:555"


def test_build_session_uses_task_id_for_non_discord_sources() -> None:
    router = SessionRouter()

    session = router.build_session(
        source="api",
        channel_id=0,
        message_id=12,
        metadata={"task_id": "task-1"},
    )

    assert session.session_id == "api:task-1"
    assert session.thread_key == "task-1"


def test_build_session_falls_back_when_no_existing_or_task_id() -> None:
    router = SessionRouter()

    session = router.build_session(
        source="api",
        channel_id=7,
        message_id=0,
        reference_message_id=None,
        metadata={},
    )

    assert session.session_id == "api:7:0"
    assert session.thread_key == "0"


def test_build_metadata_merges_session_fields() -> None:
    router = SessionRouter()

    metadata = router.build_metadata(
        source="discord",
        channel_id=42,
        message_id=77,
        reference_message_id=66,
        metadata={"task_id": "task-1", "custom": "value"},
    )

    assert metadata["task_id"] == "task-1"
    assert metadata["custom"] == "value"
    assert metadata["source"] == "discord"
    assert metadata["session_id"] == "discord:42:66"
    assert metadata["thread_key"] == "42:66"


def test_classify_turn_uses_reply_match_for_pending_question() -> None:
    router = SessionRouter()
    registry = TaskWaitRegistry()
    _suspend(registry, task_id="task-1", prompt_message_id=900)

    decision = router.classify_turn(
        source="discord",
        channel_id=10,
        message_id=11,
        reference_message_id=900,
        content="production",
        wait_registry=registry,
    )

    assert decision.intent is TurnIntent.ANSWER_PENDING_QUESTION


def test_classify_turn_reply_without_matching_wait_falls_through() -> None:
    router = SessionRouter()
    registry = TaskWaitRegistry()

    decision = router.classify_turn(
        source="discord",
        channel_id=10,
        message_id=11,
        reference_message_id=900,
        content="production",
        wait_registry=registry,
        has_active_task=True,
    )

    assert decision.intent is TurnIntent.CONTINUE_SAME_TASK


def test_classify_turn_uses_single_pending_short_message_as_answer() -> None:
    router = SessionRouter()
    registry = TaskWaitRegistry()
    _suspend(registry, task_id="task-1")

    decision = router.classify_turn(
        source="discord",
        channel_id=10,
        message_id=11,
        content="production",
        wait_registry=registry,
    )

    assert decision.intent is TurnIntent.ANSWER_PENDING_QUESTION


def test_classify_turn_single_pending_long_message_is_not_auto_answer() -> None:
    router = SessionRouter()
    registry = TaskWaitRegistry()
    _suspend(registry, task_id="task-1")

    decision = router.classify_turn(
        source="discord",
        channel_id=10,
        message_id=11,
        content=" ".join(["word"] * 50),
        wait_registry=registry,
    )

    assert decision.intent is TurnIntent.START_NEW_TASK


def test_classify_turn_uses_cancel_pause_when_no_pending_wait() -> None:
    router = SessionRouter()

    decision = router.classify_turn(
        source="discord",
        channel_id=10,
        message_id=11,
        content="cancel this",
    )

    assert decision.intent is TurnIntent.CANCEL_OR_PAUSE


def test_classify_turn_treats_forget_it_as_cancel() -> None:
    router = SessionRouter()

    decision = router.classify_turn(
        source="discord",
        channel_id=10,
        message_id=11,
        content="forget it",
    )

    assert decision.intent is TurnIntent.CANCEL_OR_PAUSE


def test_classify_turn_prefers_clarification_for_active_task_prefix() -> None:
    router = SessionRouter()

    decision = router.classify_turn(
        source="discord",
        channel_id=10,
        message_id=11,
        content="actually use postgres instead",
        has_active_task=True,
    )

    assert decision.intent is TurnIntent.CLARIFICATION_OR_NEW_CONSTRAINT


def test_classify_turn_continues_same_task_for_short_active_followup() -> None:
    router = SessionRouter()

    decision = router.classify_turn(
        source="discord",
        channel_id=10,
        message_id=11,
        content="sounds good",
        has_active_task=True,
    )

    assert decision.intent is TurnIntent.CONTINUE_SAME_TASK


def test_classify_turn_starts_new_task_for_short_imperative_followup() -> None:
    router = SessionRouter()

    decision = router.classify_turn(
        source="discord",
        channel_id=10,
        message_id=11,
        content="restart my docker containers",
        has_active_task=True,
    )

    assert decision.intent is TurnIntent.START_NEW_TASK


def test_classify_turn_starts_new_task_for_long_active_followup_without_reference() -> None:
    router = SessionRouter()

    decision = router.classify_turn(
        source="discord",
        channel_id=10,
        message_id=11,
        content=" ".join(["word"] * 25),
        has_active_task=True,
    )

    assert decision.intent is TurnIntent.START_NEW_TASK


def test_classify_turn_starts_new_task_when_pending_waits_are_ambiguous() -> None:
    router = SessionRouter()
    registry = TaskWaitRegistry()
    _suspend(registry, task_id="task-1")
    _suspend(registry, task_id="task-2")

    decision = router.classify_turn(
        source="discord",
        channel_id=10,
        message_id=11,
        content="production",
        wait_registry=registry,
    )

    assert decision.intent is TurnIntent.START_NEW_TASK
