from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import agent.loop as loop_module
from agent.events import TaskDoneEvent, TaskErrorEvent, TaskStartEvent, TaskWaitingEvent
from agent.loop import AgentLoop, Task, TaskResult
from agent.loop_services import RunResult


class _StubAgent:
    async def run(self, prompt: str, usage_limits=None):
        return SimpleNamespace(output="summary")


class _Memory:
    def __init__(self) -> None:
        self.saved_messages: list[tuple[str, str, int, dict | None]] = []
        self.recorded_tasks: list[TaskResult] = []
        self.rows: dict[str, dict] = {}

    async def save_message(self, role: str, content: str, channel_id: int = 0, metadata=None) -> None:
        self.saved_messages.append((role, content, channel_id, metadata))

    async def record_task(self, task: Task, result: TaskResult) -> None:
        self.recorded_tasks.append(result)

    async def create_task_record(self, *, task_id: str, source: str, author: str, content: str, metadata=None) -> None:
        self.rows[task_id] = {
            "task_id": task_id,
            "source": source,
            "author": author,
            "content": content,
            "status": "queued",
            "metadata": metadata or {},
        }

    async def get_task_record(self, task_id: str):
        return self.rows.get(task_id)

    async def mark_task_waiting(self, task_id: str, *, metadata: dict, question: str) -> None:
        row = self.rows.setdefault(task_id, {"task_id": task_id})
        row["status"] = "waiting_for_user"
        row["metadata"] = metadata
        row["question"] = question

    async def list_waiting_task_records(self) -> list[dict]:
        return [row for row in self.rows.values() if row.get("status") == "waiting_for_user"]


class _Postgres:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def log_task_start(self, *args) -> None:
        self.calls.append(("start",) + args)

    async def log_task_done(self, *args) -> None:
        self.calls.append(("done",) + args)


@pytest.mark.asyncio
async def test_execute_task_records_result_and_resolves_future(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _Memory()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)
    created: list[asyncio.coroutines] = []

    async def fake_process(task: Task) -> TaskResult:
        return TaskResult(task=task, output="ok", success=True, elapsed_ms=1.0)

    monkeypatch.setattr(loop, "_process", fake_process)
    monkeypatch.setattr(asyncio, "create_task", lambda coro: created.append(coro))

    future: asyncio.Future[TaskResult] = asyncio.get_running_loop().create_future()
    task = Task(content="do thing", response_future=future)
    result = await loop._execute_task(task)

    assert result.output == "ok"
    assert memory.recorded_tasks[0].output == "ok"
    assert future.result().output == "ok"
    assert created
    for coro in created:
        coro.close()


@pytest.mark.asyncio
async def test_run_forever_processes_queued_task(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})
    processed: list[str] = []

    async def fake_execute(task: Task) -> TaskResult:
        processed.append(task.content)
        loop.stop()
        return TaskResult(task=task, output="done", success=True, elapsed_ms=1.0)

    monkeypatch.setattr(loop, "_execute_task", fake_execute)
    await loop.enqueue(Task(content="queued"))
    await loop.run_forever()

    assert processed == ["queued"]
    assert loop.is_busy is False


@pytest.mark.asyncio
async def test_run_forever_sets_exception_on_response_future(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})
    future: asyncio.Future[TaskResult] = asyncio.get_running_loop().create_future()

    async def fake_execute(task: Task) -> TaskResult:
        loop.stop()
        raise RuntimeError("boom")

    monkeypatch.setattr(loop, "_execute_task", fake_execute)
    await loop.enqueue(Task(content="queued", response_future=future))
    await loop.run_forever()

    assert future.done()
    assert isinstance(future.exception(), RuntimeError)


@pytest.mark.asyncio
async def test_run_forever_heartbeats_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})
    calls = {"wait": 0, "heartbeat": 0}

    async def fake_wait_for(awaitable, timeout):
        calls["wait"] += 1
        awaitable.close()
        raise TimeoutError

    async def fake_heartbeat() -> None:
        calls["heartbeat"] += 1
        loop.stop()

    monkeypatch.setattr(loop_module.asyncio, "wait_for", fake_wait_for)
    monkeypatch.setattr(loop, "_heartbeat", fake_heartbeat)

    await loop.run_forever()

    assert calls == {"wait": 1, "heartbeat": 1}


@pytest.mark.asyncio
async def test_process_success_emits_events_and_persists(event_collector, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _Memory()
    postgres = _Postgres()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory, postgres_store=postgres)
    normalized_task = Task(content="Fix parser", source="discord", author="Josh", channel_id=7)

    async def fake_build(task: Task):
        return normalized_task, "smart", "prompt"

    async def fake_run(**kwargs):
        return RunResult(output="done", tool_calls=2)

    monkeypatch.setattr(loop._context_builder, "build", fake_build)
    monkeypatch.setattr(loop._run_executor, "run", fake_run)
    monkeypatch.setattr(loop, "_ensure_answer_required", lambda **kwargs: asyncio.sleep(0, result=("done", True)))

    result = await loop._process(Task(content="Fix parser", source="discord", author="Josh", channel_id=7))

    assert result.success is True
    assert loop._success_count == 1
    assert memory.saved_messages[0][0] == "user"
    assert memory.saved_messages[1][0] == "assistant"
    assert postgres.calls[0][0] == "start"
    assert postgres.calls[1][0] == "done"
    assert [type(event) for event in event_collector] == [TaskStartEvent, TaskDoneEvent]


@pytest.mark.asyncio
async def test_process_waiting_marks_task_as_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _Memory()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)
    normalized_task = Task(content="Fix parser", source="discord", author="Josh", channel_id=7)

    async def fake_build(task: Task):
        return normalized_task, "smart", "prompt"

    async def fake_run(**kwargs):
        return RunResult(waiting_for_user=True, question="What environment?", timeout_s=90)

    monkeypatch.setattr(loop._context_builder, "build", fake_build)
    monkeypatch.setattr(loop._run_executor, "run", fake_run)

    result = await loop._process(Task(content="Fix parser", source="discord", author="Josh", channel_id=7))

    assert result.status == "waiting_for_user"
    assert result.success is None
    assert result.question == "What environment?"
    wait_state = memory.rows[result.task.metadata["task_id"]]["metadata"]["wait_state"]
    assert wait_state["channel_id"] == 7
    assert wait_state["prompt_message_id"] is None


@pytest.mark.asyncio
async def test_restore_waiting_tasks_rebuilds_registry() -> None:
    memory = _Memory()
    memory.rows["task-1"] = {
        "task_id": "task-1",
        "source": "discord",
        "author": "Josh",
        "content": "Fix parser",
        "status": "waiting_for_user",
        "metadata": {
            "task_id": "task-1",
            "wait_state": {
                "question": "What environment?",
                "timeout_s": 90,
                "channel_id": 7,
                "message_id": 11,
                "prompt_message_id": 42,
            },
        },
    }
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)

    restored = await loop.restore_waiting_tasks()

    assert restored == 1
    suspended = loop.wait_registry.get("task-1")
    assert suspended is not None
    assert suspended.question == "What environment?"
    assert suspended.channel_id == 7
    assert suspended.prompt_message_id == 42


@pytest.mark.asyncio
async def test_process_failure_appends_rate_limit_journal(event_collector, monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})
    calls: list[tuple[str, str, str] | str] = []

    async def fake_build(task: Task):
        return task, "smart", "prompt"

    async def fake_run(**kwargs):
        raise RuntimeError("429 limited")

    monkeypatch.setattr(loop._context_builder, "build", fake_build)
    monkeypatch.setattr(loop._run_executor, "run", fake_run)
    monkeypatch.setattr(loop._journal, "append", lambda title, body: calls.append(("append", title, body)))
    monkeypatch.setattr(loop._journal, "clear", lambda: calls.append("clear"))

    result = await loop._process(Task(content="Fix parser"))

    assert result.success is False
    assert calls and calls[0][0] == "append"  # type: ignore[index]
    assert isinstance(event_collector[1], TaskErrorEvent)


@pytest.mark.asyncio
async def test_summarize_context_returns_fallback_on_error() -> None:
    class _FailAgent(_StubAgent):
        async def run(self, prompt: str, usage_limits=None):
            raise RuntimeError("nope")

    loop = AgentLoop({"smart": _FailAgent(), "fast": _FailAgent(), "best": _FailAgent()})

    summary = await loop._summarize_context(Task(content="Long task"), "history")

    assert "Context was compressed" in summary
