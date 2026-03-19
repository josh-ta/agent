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

from agent.config import settings
from agent.loop import Task, TaskResult
from agent.loop_services import HeartbeatService, ReflectionService, RunExecutor, RunResult, TaskContextBuilder, TaskJournal
from agent.events import (
    ProgressEvent,
    TextDeltaEvent,
    TextTurnEndEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolResultEvent,
)
from agent.task_waits import UserInputRequired


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


class _StreamingAgentWithDiscordSend:
    def __init__(self, args: str) -> None:
        self._args = args

    def run_mcp_servers(self) -> _NullContext:
        return _NullContext()

    async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
        del prompt, message_history, usage_limits
        yield FunctionToolCallEvent(
            ToolCallPart(tool_name="send_discord", args=self._args, tool_call_id="call-1")
        )
        final = FinalResultEvent(tool_name=None, tool_call_id=None)
        final.output = "done"
        yield final


class _StreamingAgentWithDiscordSendResult:
    def __init__(self, args: str, result_text: str) -> None:
        self._args = args
        self._result_text = result_text

    def run_mcp_servers(self) -> _NullContext:
        return _NullContext()

    async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
        del prompt, message_history, usage_limits
        yield FunctionToolCallEvent(
            ToolCallPart(tool_name="send_discord", args=self._args, tool_call_id="call-1")
        )
        yield FunctionToolResultEvent(
            ToolReturnPart(tool_name="send_discord", content=self._result_text, tool_call_id="call-1")
        )
        final = FinalResultEvent(tool_name=None, tool_call_id=None)
        final.output = "done"
        yield final


class _StreamingAgentWithBrowserScreenshot:
    def run_mcp_servers(self) -> _NullContext:
        return _NullContext()

    async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
        del prompt, message_history, usage_limits
        yield FunctionToolCallEvent(
            ToolCallPart(tool_name="browser_screenshot", args="{}", tool_call_id="call-1")
        )
        yield FunctionToolResultEvent(
            ToolReturnPart(
                tool_name="browser_screenshot",
                content="data:image/png;base64,cG5n",
                tool_call_id="call-1",
            )
        )
        final = FinalResultEvent(tool_name=None, tool_call_id=None)
        final.output = "done"
        yield final


class _StreamingAgentWithUserQuestion:
    def run_mcp_servers(self) -> _NullContext:
        return _NullContext()

    async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
        del prompt, message_history, usage_limits
        raise UserInputRequired("Which environment should I use?", timeout_s=120)
        if False:
            yield None


class _SlowFinalAgent:
    def run_mcp_servers(self) -> _NullContext:
        return _NullContext()

    async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
        del prompt, message_history, usage_limits
        await asyncio.sleep(1.05)
        final = FinalResultEvent(tool_name=None, tool_call_id=None)
        final.output = "done"
        yield final


class _HungStreamAgent:
    def run_mcp_servers(self) -> _NullContext:
        return _NullContext()

    async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
        del prompt, message_history, usage_limits
        await asyncio.sleep(60)
        if False:
            yield None


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
async def test_task_context_builder_prefers_session_context_and_adds_resume_checkpoint_and_facts() -> None:
    class _Memory:
        async def search_lessons(self, query: str, limit: int = 3) -> str:
            return "## Relevant past lessons:\n- Reuse staging."

        async def search_memory(self, query: str, limit: int = 3) -> str:
            return "- Staging deploys need approval."

        async def get_session_context(self, session_id: str, limit: int = 10, char_cap: int = 2200) -> str:
            assert session_id == "discord:101:1"
            return "## Recent session turns\nUser: continue"

        async def get_task_checkpoint(self, task_id: str) -> dict:
            assert task_id == "task-1"
            return {
                "summary": "Already inspected the deploy script.",
                "notes": "Need to confirm environment variables.",
                "draft": "Current draft reply.",
            }

        async def get_history(self, channel_id: int, limit: int = 10) -> list[dict]:
            raise AssertionError("channel history should be skipped when session context exists")

    builder = TaskContextBuilder(_Memory())
    _, _, prompt = await builder.build(
        Task(
            content="deploy this",
            source="discord",
            channel_id=101,
            metadata={
                "session_id": "discord:101:1",
                "task_id": "task-1",
                "resume_context": {"question": "Which environment?", "answer": "staging"},
            },
        )
    )

    assert "## Recent session turns" in prompt
    assert "## Relevant stored facts" in prompt
    assert "## Previous checkpoint summary" in prompt
    assert "## Previous task notes" in prompt
    assert "## Partial draft" in prompt
    assert "## Resume context" in prompt
    assert "Recent conversation history" not in prompt


@pytest.mark.asyncio
async def test_task_context_builder_includes_project_memory(isolated_paths) -> None:
    (isolated_paths["workspace"] / ".agent-project-memory.md").write_text(
        "# Project Memory\n\n- App host is root@example\n",
        encoding="utf-8",
    )
    builder = TaskContextBuilder(None)

    _, _, prompt = await builder.build(Task(content="deploy", source="discord", channel_id=101))

    assert "## Project memory" in prompt
    assert "App host is root@example" in prompt


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
async def test_task_journal_swallows_io_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal = TaskJournal(tmp_path)
    journal.path.write_text("data", encoding="utf-8")

    original_stat = Path.stat
    original_open = Path.open
    original_unlink = Path.unlink

    def _broken_stat(self: Path, *args, **kwargs):
        if self == journal.path:
            raise OSError("stat failed")
        return original_stat(self, *args, **kwargs)

    def _broken_open(self: Path, *args, **kwargs):
        if self == journal.path:
            raise OSError("open failed")
        return original_open(self, *args, **kwargs)

    def _broken_unlink(self: Path, *args, **kwargs):
        if self == journal.path:
            raise OSError("unlink failed")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _broken_stat)
    journal.expire_stale(max_age_s=0)

    monkeypatch.setattr(Path, "open", _broken_open)
    journal.append("RUN", "body")

    monkeypatch.setattr(Path, "unlink", _broken_unlink)
    journal.clear()


def test_task_journal_keeps_fresh_entries(tmp_path: Path) -> None:
    journal = TaskJournal(tmp_path, now_fn=lambda: 100.0)
    journal.path.write_text("fresh", encoding="utf-8")
    journal.path.touch()

    journal.expire_stale(max_age_s=1000)

    assert journal.path.exists()


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
async def test_task_context_builder_uses_default_discord_reader_and_handles_history_reader_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_discord_read(channel_id: int, limit: int = 20) -> str:
        assert channel_id == 321
        assert limit == 7
        return "\n".join(
            [
                "User: " + ("a" * 700),
                "Assistant: " + ("b" * 700),
                "User: current message",
            ]
        )

    async def broken_history_reader(channel_id: int, limit: int) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr("agent.tools.discord_tools.discord_read", fake_discord_read)
    builder = TaskContextBuilder(None)
    _, _, prompt = await builder.build(Task(content="investigate", source="discord", channel_id=321))

    assert builder._history_reader is fake_discord_read
    assert "Recent conversation history" in prompt
    assert "current message" not in prompt

    broken = TaskContextBuilder(None, history_reader=broken_history_reader)
    _, _, prompt = await broken.build(Task(content="investigate", source="discord", channel_id=321))
    assert prompt == "investigate"


@pytest.mark.asyncio
async def test_task_context_builder_covers_history_break_paths() -> None:
    class _Memory:
        async def search_lessons(self, query: str, limit: int = 3) -> str:
            return ""

        async def get_history(self, channel_id: int, limit: int = 10) -> list[dict]:
            return [
                {"role": "user", "content": "a" * 700},
                {"role": "assistant", "content": "b" * 700},
                {"role": "assistant", "content": "current"},
            ]

    async def empty_reader(channel_id: int, limit: int) -> str:
        return ""

    builder = TaskContextBuilder(_Memory())
    _, _, prompt = await builder.build(Task(content="investigate", source="discord", channel_id=321))
    assert "Recent conversation history" in prompt
    assert "current" not in prompt

    async def long_reader(channel_id: int, limit: int) -> str:
        return "\n".join(
            [
                "User: " + ("a" * 700),
                "Assistant: " + ("b" * 700),
                "User: " + ("c" * 700),
                "User: current message",
            ]
        )

    truncated = TaskContextBuilder(None, history_reader=long_reader)
    _, _, prompt = await truncated.build(Task(content="investigate", source="discord", channel_id=321))
    assert "Recent conversation history" in prompt
    assert "current message" not in prompt

    empty = TaskContextBuilder(None, history_reader=empty_reader)
    _, _, prompt = await empty.build(Task(content="investigate", source="discord", channel_id=321))
    assert prompt == "investigate"


@pytest.mark.asyncio
async def test_task_context_builder_ignores_no_memory_matches_and_empty_resume_context() -> None:
    class _Memory:
        async def search_lessons(self, query: str, limit: int = 3) -> str:
            return ""

        async def search_memory(self, query: str, limit: int = 3) -> str:
            return "(no memory matches for: deploy)"

    builder = TaskContextBuilder(_Memory())
    _, _, prompt = await builder.build(
        Task(content="deploy", metadata={"resume_context": {}})
    )

    assert "## Relevant stored facts" not in prompt
    assert "## Resume context" not in prompt


@pytest.mark.asyncio
async def test_task_context_builder_handles_session_checkpoint_failures_and_large_history() -> None:
    class _Memory:
        async def search_lessons(self, query: str, limit: int = 3) -> str:
            return ""

        async def get_session_context(self, session_id: str, limit: int = 10, char_cap: int = 2200) -> str:
            raise RuntimeError("session boom")

        async def get_task_checkpoint(self, task_id: str) -> dict | None:
            raise RuntimeError("checkpoint boom")

        async def get_history(self, channel_id: int, limit: int = 10) -> list[dict]:
            return [
                {"role": "user", "content": "x" * 2000},
                {"role": "assistant", "content": "current"},
            ]

    builder = TaskContextBuilder(_Memory())
    _, _, prompt = await builder.build(
        Task(
            content="deploy",
            source="discord",
            channel_id=101,
            metadata={"session_id": "discord:101:1", "task_id": "task-1"},
        )
    )

    assert "## Recent conversation history" in prompt
    assert "## Previous checkpoint summary" not in prompt


@pytest.mark.asyncio
async def test_task_context_builder_ignores_empty_checkpoint() -> None:
    class _Memory:
        async def search_lessons(self, query: str, limit: int = 3) -> str:
            return ""

        async def get_task_checkpoint(self, task_id: str) -> dict | None:
            return None

    builder = TaskContextBuilder(_Memory())
    _, _, prompt = await builder.build(Task(content="deploy", metadata={"task_id": "task-1"}))

    assert "## Previous checkpoint summary" not in prompt


@pytest.mark.asyncio
async def test_task_context_builder_skips_blank_checkpoint_resume_and_memory_fact_sections() -> None:
    class _Memory:
        async def search_lessons(self, query: str, limit: int = 3) -> str:
            return ""

        async def search_memory(self, query: str, limit: int = 3) -> str:
            raise RuntimeError("memory boom")

        async def get_task_checkpoint(self, task_id: str) -> dict | None:
            return {"summary": "", "notes": "", "draft": ""}

    builder = TaskContextBuilder(_Memory())

    _, _, question_only = await builder.build(
        Task(content="deploy", metadata={"task_id": "task-1", "resume_context": {"question": "Which env?"}})
    )
    _, _, answer_only = await builder.build(
        Task(content="deploy", metadata={"task_id": "task-1", "resume_context": {"answer": "staging"}})
    )

    assert "## Relevant stored facts" not in question_only
    assert "## Previous checkpoint summary" not in question_only
    assert "Question asked: Which env?" in question_only
    assert "User answer:" not in question_only

    assert "## Relevant stored facts" not in answer_only
    assert "## Previous task notes" not in answer_only
    assert "Question asked:" not in answer_only
    assert "User answer: staging" in answer_only


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
    assert result == RunResult()


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

    assert result.output == "revised"
    assert result.tool_calls == 1
    assert result.user_visible_reply_sent is False
    assert result.attachments == []
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

    assert result.output == "done"
    assert result.tool_calls == 0
    assert result.user_visible_reply_sent is False
    assert result.attachments == []
    assert [type(event) for event in bridge.events] == [
        ThinkingDeltaEvent,
        ThinkingEndEvent,
    ]
    assert bridge.events[0].delta == "reason"
    assert bridge.events[1].text == "reason"


@pytest.mark.asyncio
async def test_run_executor_does_not_treat_comms_send_as_user_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "discord_comms_channel_id", 303)
    monkeypatch.setattr(settings, "discord_bus_channel_id", 202)
    monkeypatch.setattr(settings, "discord_agent_channel_id", 101)
    executor = RunExecutor(event_bridge=_Bridge())
    agent = _StreamingAgentWithDiscordSend('{"channel_id": 303, "message": "done"}')

    result = await executor.run(
        task=Task(content="start", channel_id=101),
        agent=agent,  # type: ignore[arg-type]
        base_prompt="start",
        tier="smart",
    )

    assert result.output == "done"
    assert result.tool_calls == 1
    assert result.user_visible_reply_sent is False
    assert result.attachments == []


@pytest.mark.asyncio
async def test_run_executor_treats_private_send_as_user_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "discord_comms_channel_id", 303)
    monkeypatch.setattr(settings, "discord_bus_channel_id", 202)
    monkeypatch.setattr(settings, "discord_agent_channel_id", 101)
    executor = RunExecutor(event_bridge=_Bridge())
    agent = _StreamingAgentWithDiscordSend('{"channel_id": 101, "message": "done"}')

    result = await executor.run(
        task=Task(content="start", channel_id=404),
        agent=agent,  # type: ignore[arg-type]
        base_prompt="start",
        tier="smart",
    )

    assert result.output == "done"
    assert result.tool_calls == 1
    assert result.user_visible_reply_sent is False
    assert result.attachments == []


@pytest.mark.asyncio
async def test_run_executor_marks_visible_send_discord_result_as_user_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "discord_comms_channel_id", 303)
    monkeypatch.setattr(settings, "discord_bus_channel_id", 202)
    monkeypatch.setattr(settings, "discord_agent_channel_id", 101)
    executor = RunExecutor(event_bridge=_Bridge())
    agent = _StreamingAgentWithDiscordSendResult(
        '{"channel_id": 101, "message": "done"}',
        "Sent message to Discord channel 101",
    )

    result = await executor.run(
        task=Task(content="start", channel_id=404),
        agent=agent,  # type: ignore[arg-type]
        base_prompt="start",
        tier="smart",
    )

    assert result.output == "done"
    assert result.tool_calls == 1
    assert result.user_visible_reply_sent is True


@pytest.mark.asyncio
async def test_run_executor_collects_browser_screenshot_attachments() -> None:
    executor = RunExecutor(event_bridge=_Bridge())
    agent = _StreamingAgentWithBrowserScreenshot()

    result = await executor.run(
        task=Task(content="show screenshot", source="discord", channel_id=101),
        agent=agent,  # type: ignore[arg-type]
        base_prompt="show screenshot",
        tier="smart",
    )

    assert result.output == "done"
    assert result.tool_calls == 1
    assert result.user_visible_reply_sent is False
    assert len(result.attachments) == 1
    assert result.attachments[0].filename == "browser-screenshot-1.png"
    assert result.attachments[0].data == b"png"


@pytest.mark.asyncio
async def test_run_executor_emits_idle_progress_when_task_goes_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "progress_heartbeat_seconds", 1)
    bridge = _Bridge()
    executor = RunExecutor(event_bridge=bridge)
    agent = _SlowFinalAgent()

    result = await executor.run(
        task=Task(content="wait for it"),
        agent=agent,  # type: ignore[arg-type]
        base_prompt="wait for it",
        tier="smart",
    )

    assert result.output == "done"
    assert result.tool_calls == 0
    assert result.user_visible_reply_sent is False
    assert result.attachments == []
    assert any(
        isinstance(event, ProgressEvent) and "Still working" in event.message
        for event in bridge.events
    )


@pytest.mark.asyncio
async def test_run_executor_times_out_stalled_model_turn() -> None:
    bridge = _Bridge()
    executor = RunExecutor(event_bridge=bridge, model_event_idle_timeout_s=1)
    agent = _HungStreamAgent()

    with pytest.raises(RuntimeError, match="Model turn timed out"):
        await executor.run(
            task=Task(content="hang forever"),
            agent=agent,  # type: ignore[arg-type]
            base_prompt="hang forever",
            tier="smart",
        )

    assert any(
        isinstance(event, ProgressEvent) and "Model turn timed out" in event.message
        for event in bridge.events
    )


@pytest.mark.asyncio
async def test_run_executor_progress_watchdog_skips_recent_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "progress_heartbeat_seconds", 1)
    bridge = _Bridge()
    sleep_calls = {"count": 0}

    class _Loop:
        def __init__(self) -> None:
            self.values = iter([0.5])

        def time(self) -> float:
            return next(self.values)

    async def fake_sleep(delay: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: _Loop())
    executor = RunExecutor(event_bridge=bridge, progress_sleep=fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await executor._progress_watchdog({"last_activity_at": 0.0, "activity": "thinking"})

    assert bridge.events == []


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

    assert result.output == "done"
    assert result.tool_calls == 0
    assert result.user_visible_reply_sent is False
    assert result.attachments == []
    assert "summary" in agent.prompts[1]
    assert "CONTEXT COMPRESSED" in journal.path.read_text(encoding="utf-8")
    assert isinstance(bridge.events[0], ProgressEvent)


@pytest.mark.asyncio
async def test_run_executor_returns_waiting_result_for_user_question() -> None:
    executor = RunExecutor(event_bridge=_Bridge())
    agent = _StreamingAgentWithUserQuestion()

    result = await executor.run(
        task=Task(content="deploy it", source="discord", channel_id=101),
        agent=agent,  # type: ignore[arg-type]
        base_prompt="deploy it",
        tier="smart",
    )

    assert result.waiting_for_user is True
    assert result.question == "Which environment should I use?"
    assert result.timeout_s == 120


@pytest.mark.asyncio
async def test_run_executor_followup_injected_message_can_pause_for_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "discord_comms_channel_id", 303)
    monkeypatch.setattr(settings, "discord_bus_channel_id", 202)
    monkeypatch.setattr(settings, "discord_agent_channel_id", 101)

    class _Agent:
        def __init__(self) -> None:
            self.calls = 0

        def run_mcp_servers(self) -> _NullContext:
            return _NullContext()

        async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
            self.calls += 1
            if self.calls == 1:
                yield FunctionToolCallEvent(
                    ToolCallPart(tool_name="send_discord", args='{"channel_id": 101, "message": "done"}', tool_call_id="call-1")
                )
                yield FunctionToolResultEvent(
                    ToolReturnPart(tool_name="send_discord", content="Sent message to Discord channel 101", tool_call_id="call-1")
                )
                yield FunctionToolCallEvent(
                    ToolCallPart(tool_name="browser_screenshot", args="{}", tool_call_id="call-2")
                )
                yield FunctionToolResultEvent(
                    ToolReturnPart(
                        tool_name="browser_screenshot",
                        content="data:image/png;base64,cG5n",
                        tool_call_id="call-2",
                    )
                )
                final = FinalResultEvent(tool_name=None, tool_call_id=None)
                final.output = "draft"
                yield final
                return
            raise UserInputRequired("Which environment should I use?", timeout_s=45)
            if False:
                yield None

    executor = RunExecutor(event_bridge=_Bridge())
    inject_q: asyncio.Queue[str] = asyncio.Queue()
    inject_q.put_nowait("also include this")
    result = await executor.run(
        task=Task(content="start", channel_id=101, inject_queue=inject_q),
        agent=_Agent(),  # type: ignore[arg-type]
        base_prompt="start",
        tier="smart",
    )

    assert result.waiting_for_user is True
    assert result.question == "Which environment should I use?"
    assert result.timeout_s == 45
    assert result.tool_calls == 2
    assert result.user_visible_reply_sent is True
    assert len(result.attachments) == 1


def test_user_input_required_allows_traceback_assignment() -> None:
    exc = UserInputRequired("Which environment should I use?", timeout_s=120)

    try:
        raise RuntimeError("boom")
    except RuntimeError as err:
        exc.__traceback__ = err.__traceback__

    assert str(exc) == "Which environment should I use?"
    assert exc.timeout_s == 120


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
async def test_task_context_builder_includes_attachment_context() -> None:
    builder = TaskContextBuilder(None)
    task = Task(
        content="summarize the upload",
        metadata={
            "attachments": [
                {
                    "filename": "report.csv",
                    "content_type": "text/csv",
                    "size_bytes": 22,
                    "saved_path": "/tmp/report.csv",
                    "summary": "CSV preview:\nname | value",
                    "inline_part": None,
                }
            ]
        },
    )

    _, _, prompt = await builder.build(task)

    assert "## Attachments" in prompt
    assert "report.csv" in prompt
    assert "/tmp/report.csv" in prompt


def test_run_executor_helper_methods_cover_parsing_visibility_and_attachment_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "discord_comms_channel_id", 303)
    monkeypatch.setattr(settings, "discord_bus_channel_id", 202)
    monkeypatch.setattr(settings, "discord_agent_channel_id", 101)
    task = Task(content="start", channel_id=101)

    assert RunExecutor._parse_tool_args('{"channel_id": 101}') == {"channel_id": 101}
    assert RunExecutor._parse_tool_args("{bad json") == "{bad json"
    assert RunExecutor._is_user_visible_discord_send(task, "send_discord", {"channel_id": 101}) is True
    assert RunExecutor._is_user_visible_discord_send(task, "send_discord", {"channel_id": "oops"}) is False
    assert RunExecutor._is_successful_send_discord_result("Sent message") is True
    assert RunExecutor._is_successful_send_discord_result("[ERROR] nope") is False
    assert RunExecutor._tool_name_from_result_event(SimpleNamespace(tool_name="from-event"), SimpleNamespace()) == "from-event"
    assert RunExecutor._tool_name_from_result_event(SimpleNamespace(tool_name=""), SimpleNamespace(tool_name="from-result")) == "from-result"
    assert RunExecutor._sanitize_tool_args("secret_set", {"name": "LOGIN_PASSWORD", "value": "hunter2"}) == {
        "name": "LOGIN_PASSWORD",
        "value": "[REDACTED]",
    }
    assert RunExecutor._sanitize_tool_result("secret_get", "hunter2") == "[REDACTED secret value]"
    assert RunExecutor._detect_shell_failure("ok\n[exit code: 0]") is None
    assert "Host key verification failed" in RunExecutor._detect_shell_failure(
        "Host key verification failed.\n[exit code: 255]"
    )
    assert RunExecutor._iter_text_values(
        {
            "a": "plain",
            "b": [{"text": "nested"}, {"content": ["deep", {"text": "leaf"}]}],
        }
    ) == ["plain", "nested", "deep", "leaf"]

    attachments = RunExecutor._extract_discord_attachments(
        "browser_screenshot",
        {"content": ["data:image/png;base64,cG5n", {"text": "data:image/png;base64,cG5n"}]},
    )
    assert [attachment.filename for attachment in attachments] == [
        "browser-screenshot-1.png",
        "browser-screenshot-2.png",
    ]
    assert RunExecutor._drain_queue(asyncio.Queue()) == []

    class _SelfContent:
        def __init__(self) -> None:
            self.content = self

    class _NoText:
        pass

    assert RunExecutor._iter_text_values(_SelfContent()) == []
    assert RunExecutor._iter_text_values(_NoText()) == []


def test_run_executor_helper_methods_cover_more_edge_cases() -> None:
    class _TextValue:
        text = "plain"

    class _QueueRace:
        def empty(self) -> bool:
            return False

        def get_nowait(self):
            raise asyncio.QueueEmpty

    assert RunExecutor._parse_tool_args({"channel_id": 101}) == {"channel_id": 101}
    assert RunExecutor._extract_discord_attachments("send_discord", "data:image/png;base64,cG5n") == []
    assert RunExecutor._iter_text_values(None) == []
    assert RunExecutor._iter_text_values(_TextValue()) == ["plain"]
    assert RunExecutor._iter_text_values(SimpleNamespace(content=["deep"])) == ["deep"]
    assert RunExecutor._drain_queue(_QueueRace()) == []


def test_run_executor_compose_user_prompt_loads_inline_attachment(tmp_path: Path) -> None:
    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    task = Task(
        content="look at the image",
        metadata={
            "attachments": [
                {
                    "filename": "shot.png",
                    "content_type": "image/png",
                    "size_bytes": 8,
                    "saved_path": str(image_path),
                    "summary": "Image metadata: PNG 1x1 (RGBA).",
                    "inline_part": {
                        "path": str(image_path),
                        "media_type": "image/png",
                        "identifier": "shot.png",
                        "vendor_metadata": {"detail": "high"},
                    },
                }
            ]
        },
    )

    prompt = RunExecutor._compose_user_prompt("look at the image", task)

    assert isinstance(prompt, list)
    assert prompt[0] == "look at the image"
    assert prompt[1].media_type == "image/png"
    assert prompt[1].data == b"\x89PNG\r\n\x1a\n"


@pytest.mark.asyncio
async def test_run_executor_handles_part_end_whitespace_and_invalid_attachment(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Agent:
        def run_mcp_servers(self) -> _NullContext:
            return _NullContext()

        async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
            yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="   "))
            yield PartEndEvent(index=0, part=SimpleNamespace())
            yield PartEndEvent(index=1, part=SimpleNamespace())
            yield PartDeltaEvent(index=2, delta=TextPartDelta(content_delta="   "))
            yield PartEndEvent(index=2, part=SimpleNamespace())
            yield FunctionToolResultEvent(
                ToolReturnPart(
                    tool_name="browser_screenshot",
                    content="not-a-data-url",
                    tool_call_id="call-1",
                )
            )
            final = FinalResultEvent(tool_name=None, tool_call_id=None)
            final.output = "done"
            yield final

    bridge = _Bridge()
    executor = RunExecutor(event_bridge=bridge)
    result = await executor.run(
        task=Task(content="start"),
        agent=_Agent(),  # type: ignore[arg-type]
        base_prompt="start",
        tier="smart",
    )

    assert result.output == "done"
    assert result.attachments == []


@pytest.mark.asyncio
async def test_run_executor_ignores_unhandled_result_and_empty_injected_drain() -> None:
    class _QueueRace:
        def empty(self) -> bool:
            return False

        def get_nowait(self):
            raise asyncio.QueueEmpty

    class _Agent:
        def run_mcp_servers(self) -> _NullContext:
            return _NullContext()

        async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
            yield FunctionToolResultEvent(SimpleNamespace())
            yield SimpleNamespace()
            final = FinalResultEvent(tool_name=None, tool_call_id=None)
            final.output = "done"
            yield final

    bridge = _Bridge()
    executor = RunExecutor(event_bridge=bridge)
    result = await executor.run(
        task=Task(content="start", inject_queue=_QueueRace()),  # type: ignore[arg-type]
        agent=_Agent(),  # type: ignore[arg-type]
        base_prompt="start",
        tier="smart",
    )

    assert result.output == "done"
    assert [type(event) for event in bridge.events] == [ToolResultEvent]


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
async def test_reflection_service_skips_non_recordable_success_and_missing_memory_md(isolated_paths) -> None:
    class _NothingAgent:
        async def run(self, prompt: str, usage_limits=None):
            return SimpleNamespace(output="NOTHING_TO_RECORD")

    memory = _MemoryStore()
    (isolated_paths["identity"] / "MEMORY.md").unlink()
    service = ReflectionService(agents={"fast": _NothingAgent()}, memory_store=memory)

    await service.reflect(
        Task(content="small success"),
        TaskResult(
            task=Task(content="small success"),
            output="done",
            success=True,
            elapsed_ms=1.0,
            tool_calls=8,
        ),
        success_count=10,
        memory_update_interval=10,
    )

    assert memory.saved_lessons == []
    assert not (isolated_paths["identity"] / "MEMORY.md").exists()


@pytest.mark.asyncio
async def test_reflection_service_returns_early_without_memory() -> None:
    service = ReflectionService(agents={"fast": _ReflectAgent()}, memory_store=None)

    await service.reflect(
        Task(content="small success"),
        TaskResult(task=Task(content="small success"), output="done", success=True, elapsed_ms=1.0),
        success_count=1,
        memory_update_interval=10,
    )


@pytest.mark.asyncio
async def test_reflection_service_skips_low_signal_success_and_replaces_recent_lessons_section(isolated_paths, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _MemoryStore()
    service = ReflectionService(agents={"fast": _ReflectAgent()}, memory_store=memory)
    memory_md_path = isolated_paths["identity"] / "MEMORY.md"
    memory_md_path.write_text("# Memory\n\n## Recent Lessons\nold stuff\n", encoding="utf-8")

    await service.reflect(
        Task(content="small success"),
        TaskResult(task=Task(content="small success"), output="done", success=True, elapsed_ms=1.0, tool_calls=7),
        success_count=1,
        memory_update_interval=10,
    )
    assert memory.saved_lessons == []

    await service.update_memory_md()
    content = memory_md_path.read_text(encoding="utf-8")
    assert content.count("## Recent Lessons") == 1
    assert "old stuff" not in content


@pytest.mark.asyncio
async def test_reflection_service_tolerates_outer_reflect_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    service = ReflectionService(agents={"fast": _ReflectAgent()}, memory_store=_MemoryStore())

    async def boom(*args, **kwargs) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "_reflect_on_success", boom)

    await service.reflect(
        Task(content="optimize cache"),
        TaskResult(task=Task(content="optimize cache"), output="done", success=True, elapsed_ms=1.0, tool_calls=8),
        success_count=1,
        memory_update_interval=10,
    )


@pytest.mark.asyncio
async def test_reflection_service_tolerates_agent_failures(isolated_paths) -> None:
    class _FailAgent:
        async def run(self, prompt: str, usage_limits=None):
            raise RuntimeError("boom")

    memory = _MemoryStore()
    service = ReflectionService(agents={"fast": _FailAgent()}, memory_store=memory)

    await service._reflect_on_success(
        Task(content="optimize cache"),
        TaskResult(task=Task(content="optimize cache"), output="done", success=True, elapsed_ms=1.0, tool_calls=8),
    )
    await service._reflect_on_failure(
        Task(content="broken"),
        TaskResult(task=Task(content="broken"), output="boom", success=False, elapsed_ms=1.0),
    )

    assert memory.saved_lessons == []


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


@pytest.mark.asyncio
async def test_heartbeat_service_calls_memory_heartbeat_and_skips_unclaimed_rows() -> None:
    enqueued: list[Task] = []

    async def _enqueue(task: Task) -> None:
        enqueued.append(task)

    class _Memory:
        def __init__(self) -> None:
            self.heartbeat_calls = 0

        async def heartbeat(self) -> None:
            self.heartbeat_calls += 1

    class _Postgres:
        async def get_pending_task_rows(self) -> list[dict]:
            return [{"id": "task-1", "description": "Review tests", "from_agent": "peer-1"}]

        async def mark_task_running(self, task_id: str) -> bool:
            return False

    memory = _Memory()
    service = HeartbeatService(
        memory_store=memory,
        postgres_store=_Postgres(),
        enqueue=_enqueue,
    )

    await service.heartbeat(is_busy=False)

    assert memory.heartbeat_calls == 1
    assert enqueued == []


@pytest.mark.asyncio
async def test_heartbeat_service_skips_unexpired_waiting_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    notifications: list[str] = []

    class _Memory:
        async def list_waiting_task_records(self) -> list[dict]:
            return [
                {
                    "task_id": "task-wait",
                    "updated_ts": 25.0,
                    "metadata": {
                        "wait_state": {
                            "question": "Which environment?",
                            "timeout_s": 30,
                            "channel_id": 101,
                            "created_ts": 25.0,
                        },
                    },
                }
            ]

    async def fake_discord_send(channel_id: int, message: str) -> str:
        notifications.append(message)
        return "ok"

    monkeypatch.setattr("agent.loop_services.time.time", lambda: 30.0)
    monkeypatch.setattr("agent.loop_services.discord_send", fake_discord_send)

    service = HeartbeatService(
        memory_store=_Memory(),
        postgres_store=None,
        enqueue=lambda task: asyncio.sleep(0),
    )

    await service.heartbeat(is_busy=False)

    assert notifications == []


@pytest.mark.asyncio
async def test_heartbeat_service_expires_stale_waiting_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    notifications: list[tuple[int, str]] = []

    class _Memory:
        def __init__(self) -> None:
            self.failed: list[str] = []
            self.sessions: list[tuple[str, str]] = []

        async def list_waiting_task_records(self) -> list[dict]:
            return [
                {
                    "task_id": "task-wait",
                    "updated_ts": 1.0,
                    "metadata": {
                        "session_id": "discord:101:1",
                        "wait_state": {
                            "question": "Which environment?",
                            "timeout_s": 10,
                            "channel_id": 101,
                            "created_ts": 1.0,
                        },
                    },
                }
            ]

        async def fail_task(self, task_id: str, *, error: str, metadata=None) -> None:
            self.failed.append(task_id)

        async def set_session_status(self, session_id: str, *, status: str | None = None, pending_task_id: str | None = None, metadata=None) -> None:
            self.sessions.append((session_id, status or ""))

        async def save_task_checkpoint(self, *, task_id: str, session_id: str = "", summary: str = "", draft: str = "", notes: str = "", metadata=None) -> None:
            return None

    class _WaitRegistry:
        def __init__(self) -> None:
            self.popped: list[str] = []

        def pop(self, task_id: str):
            self.popped.append(task_id)
            return None

    async def fake_discord_send(channel_id: int, message: str) -> str:
        notifications.append((channel_id, message))
        return "ok"

    monkeypatch.setattr("agent.loop_services.time.time", lambda: 30.0)
    monkeypatch.setattr("agent.loop_services.discord_send", fake_discord_send)

    service = HeartbeatService(
        memory_store=_Memory(),
        postgres_store=None,
        enqueue=lambda task: asyncio.sleep(0),
        wait_registry=_WaitRegistry(),
    )

    await service.heartbeat(is_busy=False)

    assert notifications
    assert notifications[0][0] == 101


@pytest.mark.asyncio
async def test_heartbeat_service_continues_after_persist_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _Bridge()

    class _Memory:
        async def list_waiting_task_records(self) -> list[dict]:
            return [
                {
                    "task_id": "task-wait",
                    "updated_ts": 1.0,
                    "metadata": {
                        "session_id": "discord:101:1",
                        "wait_state": {
                            "question": "Which environment?",
                            "timeout_s": 10,
                            "channel_id": 101,
                            "created_ts": 1.0,
                        },
                    },
                }
            ]

        async def fail_task(self, task_id: str, *, error: str, metadata=None) -> None:
            raise RuntimeError("persist failed")

    class _WaitRegistry:
        def __init__(self) -> None:
            self.popped: list[str] = []

        def pop(self, task_id: str):
            self.popped.append(task_id)
            return None

    notifications: list[tuple[int, str]] = []

    async def fake_discord_send(channel_id: int, message: str) -> str:
        notifications.append((channel_id, message))
        return "ok"

    monkeypatch.setattr("agent.loop_services.time.time", lambda: 30.0)
    monkeypatch.setattr("agent.loop_services.discord_send", fake_discord_send)

    wait_registry = _WaitRegistry()
    service = HeartbeatService(
        memory_store=_Memory(),
        postgres_store=None,
        enqueue=lambda task: asyncio.sleep(0),
        wait_registry=wait_registry,
        event_bridge=bridge,
    )

    await service.heartbeat(is_busy=False)

    assert wait_registry.popped == ["task-wait"]
    assert notifications[0][0] == 101
    assert any(isinstance(event, ProgressEvent) for event in bridge.events)


@pytest.mark.asyncio
async def test_heartbeat_service_logs_and_continues_on_a2a_poll_error() -> None:
    enqueued: list[Task] = []

    async def _enqueue(task: Task) -> None:
        enqueued.append(task)

    class _Postgres:
        async def get_pending_task_rows(self) -> list[dict]:
            raise RuntimeError("db boom")

    service = HeartbeatService(
        memory_store=None,
        postgres_store=_Postgres(),
        enqueue=_enqueue,
    )

    await service.heartbeat(is_busy=False)

    assert enqueued == []


@pytest.mark.asyncio
async def test_heartbeat_service_skips_invalid_wait_records_and_tolerates_notify_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _Bridge()

    class _Memory:
        async def list_waiting_task_records(self) -> list[dict]:
            return [
                {"task_id": "", "metadata": {"wait_state": {"timeout_s": 1}}, "updated_ts": 1.0},
                {
                    "task_id": "task-wait",
                    "updated_ts": 1.0,
                    "metadata": {
                        "session_id": "discord:101:1",
                        "wait_state": {
                            "question": "Which environment?",
                            "timeout_s": 10,
                            "channel_id": 101,
                            "created_ts": 1.0,
                        },
                    },
                },
            ]

        async def fail_task(self, task_id: str, *, error: str, metadata=None) -> None:
            return None

        async def set_session_status(self, session_id: str, *, status: str | None = None, pending_task_id: str | None = None, metadata=None) -> None:
            return None

        async def save_task_checkpoint(self, *, task_id: str, session_id: str = "", summary: str = "", draft: str = "", notes: str = "", metadata=None) -> None:
            return None

    class _WaitRegistry:
        def __init__(self) -> None:
            self.popped: list[str] = []

        def pop(self, task_id: str):
            self.popped.append(task_id)
            return None

    async def fake_discord_send(channel_id: int, message: str) -> str:
        raise RuntimeError("send failed")

    monkeypatch.setattr("agent.loop_services.time.time", lambda: 30.0)
    monkeypatch.setattr("agent.loop_services.discord_send", fake_discord_send)
    wait_registry = _WaitRegistry()
    service = HeartbeatService(
        memory_store=_Memory(),
        postgres_store=None,
        enqueue=lambda task: asyncio.sleep(0),
        wait_registry=wait_registry,
        event_bridge=bridge,
    )

    await service.heartbeat(is_busy=False)

    assert wait_registry.popped == ["task-wait"]
    assert any(isinstance(event, ProgressEvent) for event in bridge.events)


@pytest.mark.asyncio
async def test_heartbeat_service_expires_waiting_tasks_without_optional_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _Bridge()

    class _Memory:
        def __init__(self) -> None:
            self.checkpoints: list[dict] = []
            self.sessions: list[str] = []

        async def list_waiting_task_records(self) -> list[dict]:
            return [
                {
                    "task_id": "task-wait",
                    "updated_ts": 1.0,
                    "metadata": {
                        "wait_state": {
                            "question": "Which environment?",
                            "timeout_s": 10,
                            "channel_id": 0,
                            "created_ts": 1.0,
                        },
                    },
                }
            ]

        async def set_session_status(self, session_id: str, *, status: str | None = None, pending_task_id: str | None = None, metadata=None) -> None:
            self.sessions.append(session_id)

        async def save_task_checkpoint(self, *, task_id: str, session_id: str = "", summary: str = "", draft: str = "", notes: str = "", metadata=None) -> None:
            self.checkpoints.append({"task_id": task_id, "session_id": session_id, "summary": summary})

    monkeypatch.setattr("agent.loop_services.time.time", lambda: 30.0)
    memory = _Memory()
    service = HeartbeatService(
        memory_store=memory,
        postgres_store=None,
        enqueue=lambda task: asyncio.sleep(0),
        wait_registry=None,
        event_bridge=bridge,
    )

    await service.heartbeat(is_busy=False)

    assert memory.sessions == []
    assert memory.checkpoints[0]["task_id"] == "task-wait"
    assert any(isinstance(event, ProgressEvent) for event in bridge.events)


@pytest.mark.asyncio
async def test_heartbeat_service_expires_waiting_tasks_without_session_or_checkpoint_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _Bridge()

    class _Memory:
        async def list_waiting_task_records(self) -> list[dict]:
            return [
                {
                    "task_id": "task-wait",
                    "updated_ts": 1.0,
                    "metadata": {
                        "wait_state": {
                            "question": "Which environment?",
                            "timeout_s": 10,
                            "channel_id": 0,
                            "created_ts": 1.0,
                        },
                    },
                }
            ]

        async def fail_task(self, task_id: str, *, error: str, metadata=None) -> None:
            return None

    monkeypatch.setattr("agent.loop_services.time.time", lambda: 30.0)
    service = HeartbeatService(
        memory_store=_Memory(),
        postgres_store=None,
        enqueue=lambda task: asyncio.sleep(0),
        wait_registry=None,
        event_bridge=bridge,
    )

    await service.heartbeat(is_busy=False)

    assert any(isinstance(event, ProgressEvent) for event in bridge.events)


def test_heartbeat_service_build_a2a_task_uses_unknown_default() -> None:
    task = HeartbeatService._build_a2a_task({"id": "task-1", "description": "Review tests"})

    assert task.source == "a2a"
    assert task.author == "unknown"
    assert task.metadata["from_agent"] == "unknown"
