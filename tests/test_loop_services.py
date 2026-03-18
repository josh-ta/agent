from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)

from agent.loop import Task, TaskResult
from agent.loop_services import HeartbeatService, ReflectionService, RunExecutor, TaskContextBuilder, TaskJournal
from agent.events import (
    ProgressEvent,
    TextDeltaEvent,
    TextTurnEndEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolResultEvent,
)


class _MemoryStore:
    def __init__(self) -> None:
        self.saved_lessons: list[tuple[str, str, str]] = []

    async def search_lessons(self, query: str, limit: int = 3) -> str:
        return "## Relevant past lessons:\n- Use fixtures."

    async def get_history(self, channel_id: int, limit: int = 10) -> list[dict]:
        return [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "current"},
        ]

    async def save_lesson(self, summary: str, kind: str, context: str) -> None:
        self.saved_lessons.append((summary, kind, context))

    async def get_recent_lessons(self, limit: int = 20) -> str:
        return "- [PATTERN 2026-03-16] Reuse fixtures."


class _NullContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _RetryAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run_mcp_servers(self) -> _NullContext:
        return _NullContext()

    async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("429 too many requests")
        if False:
            yield None


class _ReflectAgent:
    async def run(self, prompt: str, usage_limits=None):
        return SimpleNamespace(output="Cache setup details after you confirm the environment.")


class _Bridge:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def emit(self, event: object) -> None:
        self.events.append(event)


class _StreamingAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def run_mcp_servers(self) -> _NullContext:
        return _NullContext()

    async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            final = FinalResultEvent(tool_name=None, tool_call_id=None)
            final.output = "draft"
            yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="reasoning"))
            yield PartEndEvent(index=0, part=ThinkingPart(content="reasoning"))
            yield PartDeltaEvent(index=1, delta=TextPartDelta(content_delta="draft"))
            yield PartEndEvent(index=1, part=TextPart(content="draft"))
            yield FunctionToolCallEvent(
                ToolCallPart(tool_name="run_shell", args='{"command": "pytest"}', tool_call_id="call-1")
            )
            yield FunctionToolResultEvent(
                ToolReturnPart(tool_name="run_shell", content="ok", tool_call_id="call-1")
            )
            yield final
            return

        final = FinalResultEvent(tool_name=None, tool_call_id=None)
        final.output = "revised"
        yield final


class _StreamingAgentWithEmptyThinkingDelta:
    def run_mcp_servers(self) -> _NullContext:
        return _NullContext()

    async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
        del prompt, message_history, usage_limits
        final = FinalResultEvent(tool_name=None, tool_call_id=None)
        final.output = "done"
        yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="reason"))
        yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=None))
        yield PartEndEvent(index=0, part=ThinkingPart(content="reason"))
        yield final


class _ContextOverflowAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def run_mcp_servers(self) -> _NullContext:
        return _NullContext()

    async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            raise RuntimeError("prompt is too long")
        final = FinalResultEvent(tool_name=None, tool_call_id=None)
        final.output = "done"
        yield final


class _BadArgsAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run_mcp_servers(self) -> _NullContext:
        return _NullContext()

    async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
        self.calls += 1
        raise RuntimeError("EOF while parsing")
        if False:
            yield None


@pytest.mark.asyncio
async def test_task_context_builder_adds_history_and_lessons() -> None:
    memory = _MemoryStore()
    builder = TaskContextBuilder(memory)

    task, tier, prompt = await builder.build(Task(content="/best please refactor this module", source="discord", channel_id=123))

    assert task.content == "please refactor this module"
    assert tier == "best"
    assert "Recent conversation history" in prompt
    assert "## Relevant past lessons" in prompt


@pytest.mark.asyncio
async def test_task_journal_expires_appends_and_clears(tmp_path: Path) -> None:
    now = 4_000.0
    journal = TaskJournal(tmp_path, now_fn=lambda: now)
    journal.path.write_text("stale", encoding="utf-8")
    journal.path.touch()
    old_mtime = now - 4_000
    journal.path.chmod(0o644)
    Path(journal.path).touch()
    import os

    os.utime(journal.path, (old_mtime, old_mtime))

    journal.expire_stale(max_age_s=100)
    assert not journal.path.exists()

    journal.append("RUN", "body")
    assert "RUN" in journal.path.read_text(encoding="utf-8")

    journal.clear()
    assert not journal.path.exists()


@pytest.mark.asyncio
async def test_task_context_builder_uses_history_reader_and_truncates() -> None:
    async def fake_history_reader(channel_id: int, limit: int) -> str:
        assert channel_id == 321
        return "\n".join(
            [
                "User: " + ("a" * 700),
                "Assistant: " + ("b" * 700),
                "User: current message",
            ]
        )

    builder = TaskContextBuilder(None, history_reader=fake_history_reader)
    _, _, prompt = await builder.build(Task(content="investigate", source="discord", channel_id=321))

    assert "Recent conversation history" in prompt
    assert "current message" not in prompt
    assert len(prompt) < 1700


@pytest.mark.asyncio
async def test_task_context_builder_handles_non_discord_and_loader_failures() -> None:
    class _FailingMemory:
        async def search_lessons(self, query: str, limit: int = 3) -> str:
            raise RuntimeError("boom")

        async def get_history(self, channel_id: int, limit: int = 10) -> list[dict]:
            raise RuntimeError("boom")

    builder = TaskContextBuilder(_FailingMemory())
    task, tier, prompt = await builder.build(Task(content="hello there", source="shell"))

    assert task.content == "hello there"
    assert tier == "fast"
    assert prompt == "hello there"


@pytest.mark.asyncio
async def test_run_executor_retries_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    executor = RunExecutor(sleep=fake_sleep)
    agent = _RetryAgent()

    result = await executor.run(
        task=Task(content="retry this"),
        agent=agent,  # type: ignore[arg-type]
        base_prompt="retry this",
        tier="smart",
    )

    assert agent.calls == 2
    assert sleeps == [5.0]
    assert result == ("", 0, False)


@pytest.mark.asyncio
async def test_run_executor_emits_events_and_folds_injected_messages() -> None:
    bridge = _Bridge()
    agent = _StreamingAgent()
    executor = RunExecutor(event_bridge=bridge)
    inject_q: asyncio.Queue[str] = asyncio.Queue()
    inject_q.put_nowait("also include this")
    task = Task(content="start", inject_queue=inject_q)

    result = await executor.run(
        task=task,
        agent=agent,  # type: ignore[arg-type]
        base_prompt="start",
        tier="smart",
    )

    assert result == ("revised", 1, False)
    assert agent.calls == 2
    assert "New message received while you were working" in agent.prompts[1]
    assert [type(event) for event in bridge.events] == [
        ThinkingDeltaEvent,
        ThinkingEndEvent,
        TextDeltaEvent,
        TextTurnEndEvent,
        ToolCallStartEvent,
        ToolResultEvent,
        ProgressEvent,
    ]


@pytest.mark.asyncio
async def test_run_executor_ignores_empty_thinking_deltas() -> None:
    bridge = _Bridge()
    agent = _StreamingAgentWithEmptyThinkingDelta()
    executor = RunExecutor(event_bridge=bridge)

    result = await executor.run(
        task=Task(content="start"),
        agent=agent,  # type: ignore[arg-type]
        base_prompt="start",
        tier="smart",
    )

    assert result == ("done", 0, False)
    assert [type(event) for event in bridge.events] == [
        ThinkingDeltaEvent,
        ThinkingEndEvent,
    ]
    assert bridge.events[0].delta == "reason"
    assert bridge.events[1].text == "reason"


@pytest.mark.asyncio
async def test_run_executor_compresses_context_and_retries(isolated_paths) -> None:
    bridge = _Bridge()
    journal = TaskJournal(isolated_paths["workspace"])
    agent = _ContextOverflowAgent()

    async def summarize_context(task: Task, prompt: str) -> str:
        assert task.content == "summarize"
        return "summary"

    executor = RunExecutor(
        event_bridge=bridge,
        journal=journal,
        summarize_context=summarize_context,
    )

    result = await executor.run(
        task=Task(content="summarize"),
        agent=agent,  # type: ignore[arg-type]
        base_prompt="summarize",
        tier="smart",
    )

    assert result == ("done", 0, False)
    assert "summary" in agent.prompts[1]
    assert "CONTEXT COMPRESSED" in journal.path.read_text(encoding="utf-8")
    assert isinstance(bridge.events[0], ProgressEvent)


@pytest.mark.asyncio
async def test_run_executor_raises_after_exhausting_bad_args_retries() -> None:
    bridge = _Bridge()
    executor = RunExecutor(event_bridge=bridge)
    agent = _BadArgsAgent()

    with pytest.raises(RuntimeError, match="EOF while parsing"):
        await executor.run(
            task=Task(content="broken"),
            agent=agent,  # type: ignore[arg-type]
            base_prompt="broken",
            tier="smart",
            message_history=["keep"],
        )

    assert agent.calls == 3
    assert len(bridge.events) == 2
    assert all(isinstance(event, ProgressEvent) for event in bridge.events)


@pytest.mark.asyncio
async def test_reflection_service_updates_memory_md_on_failure(isolated_paths) -> None:
    memory = _MemoryStore()
    service = ReflectionService(agents={"fast": _ReflectAgent()}, memory_store=memory)

    await service.reflect(
        Task(content="broken task"),
        TaskResult(task=Task(content="broken task"), output="boom", success=False, elapsed_ms=1.0),
        success_count=1,
        memory_update_interval=10,
    )

    memory_md = (isolated_paths["identity"] / "MEMORY.md").read_text(encoding="utf-8")
    assert memory.saved_lessons
    assert memory.saved_lessons[0][1] == "mistake"
    assert "## Recent Lessons" in memory_md


@pytest.mark.asyncio
async def test_reflection_service_saves_success_patterns_and_updates_memory(isolated_paths) -> None:
    memory = _MemoryStore()
    service = ReflectionService(agents={"fast": _ReflectAgent()}, memory_store=memory)

    await service.reflect(
        Task(content="optimize cache"),
        TaskResult(
            task=Task(content="optimize cache"),
            output="done",
            success=True,
            elapsed_ms=1.0,
            tool_calls=8,
        ),
        success_count=10,
        memory_update_interval=10,
    )

    memory_md = (isolated_paths["identity"] / "MEMORY.md").read_text(encoding="utf-8")
    assert memory.saved_lessons[0][1] == "pattern"
    assert "Recent Lessons" in memory_md


@pytest.mark.asyncio
async def test_reflection_service_tolerates_memory_update_errors(isolated_paths) -> None:
    class _FailingMemory(_MemoryStore):
        async def get_recent_lessons(self, limit: int = 20) -> str:
            raise RuntimeError("boom")

    service = ReflectionService(agents={"fast": _ReflectAgent()}, memory_store=_FailingMemory())
    await service.update_memory_md()
    await service.reflect(
        Task(content="optimize cache"),
        TaskResult(
            task=Task(content="optimize cache"),
            output="done",
            success=True,
            elapsed_ms=1.0,
            tool_calls=8,
        ),
        success_count=10,
        memory_update_interval=10,
    )


@pytest.mark.asyncio
async def test_heartbeat_service_enqueues_pending_a2a_tasks() -> None:
    enqueued: list[Task] = []

    async def _enqueue(task: Task) -> None:
        enqueued.append(task)

    class _Postgres:
        async def get_pending_task_rows(self) -> list[dict]:
            return [{"id": "task-1", "description": "Review tests", "from_agent": "peer-1"}]

        async def mark_task_running(self, task_id: str) -> None:
            assert task_id == "task-1"

    service = HeartbeatService(
        memory_store=None,
        postgres_store=_Postgres(),
        enqueue=_enqueue,
    )

    await service.heartbeat(is_busy=False)

    assert len(enqueued) == 1
    assert enqueued[0].source == "a2a"
    assert enqueued[0].metadata["task_id"] == "task-1"
