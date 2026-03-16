from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.loop import Task, TaskResult
from agent.loop_services import HeartbeatService, ReflectionService, RunExecutor, TaskContextBuilder


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
