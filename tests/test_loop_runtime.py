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
        self.sessions: dict[str, dict] = {}
        self.turns: list[tuple[str, str, str]] = []
        self.checkpoints: dict[str, dict] = {}
        self.running: list[str] = []

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

    async def mark_task_queued(self, task_id: str, *, metadata=None) -> None:
        row = self.rows.setdefault(task_id, {"task_id": task_id})
        row["status"] = "queued"
        if metadata is not None:
            row["metadata"] = metadata

    async def mark_task_running(self, task_id: str) -> None:
        self.running.append(task_id)
        row = self.rows.setdefault(task_id, {"task_id": task_id})
        row["status"] = "running"

    async def list_waiting_task_records(self) -> list[dict]:
        return [row for row in self.rows.values() if row.get("status") == "waiting_for_user"]

    async def list_pending_task_records(self) -> list[dict]:
        return [row for row in self.rows.values() if row.get("status") in {"queued", "running"}]

    async def fail_task(self, task_id: str, *, error: str, metadata=None) -> None:
        row = self.rows.setdefault(task_id, {"task_id": task_id})
        row["status"] = "failed"
        row["error"] = error
        if metadata is not None:
            row["metadata"] = metadata

    async def ensure_session(self, *, session_id: str, source: str, channel_id: int = 0, title: str = "", status: str = "active", pending_task_id: str = "", metadata=None) -> None:
        self.sessions[session_id] = {
            "session_id": session_id,
            "source": source,
            "channel_id": channel_id,
            "status": status,
            "title": title,
            "pending_task_id": pending_task_id,
            "metadata": metadata or {},
        }

    async def set_session_status(self, session_id: str, *, status: str | None = None, pending_task_id: str | None = None, metadata=None) -> None:
        session = self.sessions.setdefault(session_id, {"session_id": session_id})
        if status is not None:
            session["status"] = status
        if pending_task_id is not None:
            session["pending_task_id"] = pending_task_id
        if metadata is not None:
            session["metadata"] = metadata

    async def append_session_turn(self, *, session_id: str, role: str, content: str, turn_kind: str = "message", task_id: str = "", metadata=None) -> None:
        self.turns.append((session_id, role, content))

    async def save_task_checkpoint(self, *, task_id: str, session_id: str = "", summary: str = "", draft: str = "", notes: str = "", metadata=None) -> None:
        self.checkpoints[task_id] = {
            "task_id": task_id,
            "session_id": session_id,
            "summary": summary,
            "draft": draft,
            "notes": notes,
            "metadata": metadata or {},
        }

    async def save_memory_fact(self, content: str, metadata=None) -> None:
        self.checkpoints["memory_fact"] = {"content": content, "metadata": metadata or {}}


class _Postgres:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def log_task_start(self, *args) -> None:
        self.calls.append(("start",) + args)

    async def log_task_done(self, *args) -> None:
        self.calls.append(("done",) + args)

    async def complete_task(self, *args) -> None:
        self.calls.append(("complete",) + args)


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
async def test_execute_task_preserves_existing_ids_and_does_not_override_done_future(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})
    created: list[asyncio.coroutines] = []

    async def fake_process(task: Task) -> TaskResult:
        return TaskResult(task=task, output="ok", success=True, elapsed_ms=1.0)

    monkeypatch.setattr(loop, "_process", fake_process)
    monkeypatch.setattr(asyncio, "create_task", lambda coro: created.append(coro))

    future: asyncio.Future[TaskResult] = asyncio.get_running_loop().create_future()
    sentinel = TaskResult(task=Task(content="done"), output="sentinel", success=True, elapsed_ms=1.0)
    future.set_result(sentinel)
    task = Task(
        content="do thing",
        metadata={"task_id": "task-1", "session_id": "session-1"},
        response_future=future,
    )
    result = await loop._execute_task(task)

    assert result.output == "ok"
    assert task.metadata["task_id"] == "task-1"
    assert task.metadata["session_id"] == "session-1"
    assert future.result().output == "sentinel"
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
async def test_has_pending_work_run_once_and_resumed_task_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})
    assert loop.has_pending_work is False

    await loop.enqueue(Task(content="queued"))
    assert loop.has_pending_work is True
    await loop.queue.get()
    loop.is_busy = True
    assert loop.has_pending_work is True
    loop.is_busy = False

    async def fake_process(task: Task) -> TaskResult:
        return TaskResult(task=task, output=f"ran:{task.source}", success=True, elapsed_ms=1.0)

    monkeypatch.setattr(loop, "_process", fake_process)
    result = await loop.run_once("ship it", source="api")
    assert result.output == "ran:api"

    suspended = loop.wait_registry.suspend(
        task_id="task-1",
        source="discord",
        author="Josh",
        content="original",
        channel_id=7,
        message_id=11,
        metadata={"task_id": "task-1"},
        question="Which env?",
        timeout_s=30,
        base_prompt="prompt",
        tier="smart",
    )
    resumed = loop.build_resumed_task(suspended=suspended, answer="staging", author="Josh", source="discord")
    assert resumed.source == "discord"
    assert resumed.metadata["resume_context"]["answer"] == "staging"

    api_suspended = loop.wait_registry.suspend(
        task_id="task-2",
        source="api",
        author="Josh",
        content="original",
        channel_id=0,
        message_id=0,
        metadata={"task_id": "task-2"},
        question="Which env?",
        timeout_s=30,
        base_prompt="prompt",
        tier="smart",
    )
    api_resumed = loop.build_resumed_task(suspended=api_suspended, answer="prod", author="Josh", source="discord")
    assert api_resumed.source == "discord"


@pytest.mark.asyncio
async def test_clear_queued_tasks_fails_removed_items_and_resolves_futures() -> None:
    memory = _Memory()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)
    future: asyncio.Future[TaskResult] = asyncio.get_running_loop().create_future()
    kept = Task(content="keep me", source="api", metadata={"task_id": "task-keep"})
    removed = Task(
        content="drop me",
        source="discord",
        channel_id=7,
        metadata={"task_id": "task-drop"},
        response_future=future,
    )
    await memory.create_task_record(
        task_id="task-drop",
        source="discord",
        author="Josh",
        content="drop me",
        metadata={"task_id": "task-drop"},
    )
    await loop.enqueue(kept)
    await loop.enqueue(removed)

    cleared = await loop.clear_queued_tasks(source="discord", channel_id=7, reason="Cancelled by operator.")

    assert [task.content for task in cleared] == ["drop me"]
    assert [task.content for task in loop.queued_tasks()] == ["keep me"]
    assert memory.rows["task-drop"]["status"] == "failed"
    assert future.result().output == "Cancelled by operator."


@pytest.mark.asyncio
async def test_restore_pending_tasks_skips_discord_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _Memory()
    memory.rows["task-1"] = {
        "task_id": "task-1",
        "source": "discord",
        "author": "Josh",
        "content": "old discord task",
        "status": "queued",
        "metadata": {"task_id": "task-1"},
    }
    memory.rows["task-2"] = {
        "task_id": "task-2",
        "source": "api",
        "author": "Josh",
        "content": "api task",
        "status": "queued",
        "metadata": {"task_id": "task-2"},
    }
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)

    monkeypatch.setattr(loop_module.settings, "restore_pending_discord_tasks", False)
    restored = await loop.restore_pending_tasks()

    assert restored == 1
    assert [task.content for task in loop.queued_tasks()] == ["api task"]


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
async def test_run_forever_handles_execute_errors_without_response_future(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})

    async def fake_execute(task: Task) -> TaskResult:
        loop.stop()
        raise RuntimeError("boom")

    monkeypatch.setattr(loop, "_execute_task", fake_execute)
    await loop.enqueue(Task(content="queued"))

    await loop.run_forever()

    assert loop.is_busy is False


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
async def test_process_succeeds_despite_initial_best_effort_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FlakyMemory(_Memory):
        async def ensure_session(self, **kwargs) -> None:
            raise RuntimeError("ensure boom")

        async def save_message(self, role: str, content: str, channel_id: int = 0, metadata=None) -> None:
            if role == "user":
                raise RuntimeError("save boom")
            await super().save_message(role, content, channel_id, metadata)

    class _FlakyPostgres(_Postgres):
        async def log_task_start(self, *args) -> None:
            raise RuntimeError("start boom")

    memory = _FlakyMemory()
    postgres = _FlakyPostgres()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory, postgres_store=postgres)

    async def fake_build(task: Task):
        return Task(content="Fix parser", source="discord", author="Josh", channel_id=7), "smart", "prompt"

    async def fake_run(**kwargs):
        return RunResult(output="done", tool_calls=2)

    monkeypatch.setattr(loop._context_builder, "build", fake_build)
    monkeypatch.setattr(loop._run_executor, "run", fake_run)
    monkeypatch.setattr(loop, "_ensure_answer_required", lambda **kwargs: asyncio.sleep(0, result=("done", True)))

    result = await loop._process(Task(content="Fix parser", source="discord", author="Josh", channel_id=7))

    assert result.success is True
    assert result.output == "done"
    assert memory.saved_messages[-1][0] == "assistant"


@pytest.mark.asyncio
async def test_process_discord_message_saves_without_session_turn_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MessageOnlyMemory:
        def __init__(self) -> None:
            self.saved_messages: list[tuple[str, str]] = []

        async def save_message(self, role: str, content: str, channel_id: int = 0, metadata=None) -> None:
            self.saved_messages.append((role, content))

    memory = _MessageOnlyMemory()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)
    promoted: list[str] = []

    async def fake_build(task: Task):
        return Task(content="Fix parser", source="discord", author="Josh", channel_id=7), "smart", "prompt"

    async def fake_run(**kwargs):
        return RunResult(output="done", tool_calls=1, user_visible_reply_sent=True)

    async def fake_promote(*, task: Task) -> None:
        promoted.append(task.content)

    monkeypatch.setattr(loop._context_builder, "build", fake_build)
    monkeypatch.setattr(loop._run_executor, "run", fake_run)
    monkeypatch.setattr(loop, "_maybe_promote_memory_fact", fake_promote)

    result = await loop._process(Task(content="Fix parser", source="discord", author="Josh", channel_id=7))

    assert result.success is True
    assert memory.saved_messages == [("user", "Fix parser"), ("assistant", "done")]
    assert promoted == ["Fix parser"]


@pytest.mark.asyncio
async def test_process_succeeds_despite_completion_best_effort_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FlakyMemory(_Memory):
        async def set_session_status(self, *args, **kwargs) -> None:
            raise RuntimeError("status boom")

        async def save_task_checkpoint(self, *args, **kwargs) -> None:
            raise RuntimeError("checkpoint boom")

        async def append_session_turn(self, *args, **kwargs) -> None:
            raise RuntimeError("turn boom")

        async def save_message(self, role: str, content: str, channel_id: int = 0, metadata=None) -> None:
            if role == "assistant":
                raise RuntimeError("save boom")
            await super().save_message(role, content, channel_id, metadata)

    class _FlakyPostgres(_Postgres):
        async def log_task_done(self, *args) -> None:
            raise RuntimeError("done boom")

    memory = _FlakyMemory()
    postgres = _FlakyPostgres()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory, postgres_store=postgres)

    async def fake_build(task: Task):
        return Task(content="Fix parser", source="discord", author="Josh", channel_id=7), "smart", "prompt"

    async def fake_run(**kwargs):
        return RunResult(output="done", tool_calls=2)

    monkeypatch.setattr(loop._context_builder, "build", fake_build)
    monkeypatch.setattr(loop._run_executor, "run", fake_run)
    monkeypatch.setattr(loop, "_ensure_answer_required", lambda **kwargs: asyncio.sleep(0, result=("done", True)))

    result = await loop._process(Task(content="Fix parser", source="discord", author="Josh", channel_id=7))

    assert result.success is True
    assert result.output == "done"


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
async def test_process_waiting_uses_existing_task_record_and_minimal_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MinimalMemory:
        def __init__(self) -> None:
            self.created = 0
            self.waiting: list[tuple[str, dict, str]] = []

        async def get_task_record(self, task_id: str):
            return {"task_id": task_id}

        async def create_task_record(self, **kwargs) -> None:
            self.created += 1

        async def mark_task_waiting(self, task_id: str, *, metadata: dict, question: str) -> None:
            self.waiting.append((task_id, dict(metadata), question))

    memory = _MinimalMemory()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)
    normalized_task = Task(content="Fix parser", source="api", author="Josh", metadata={"task_id": "task-1"})

    async def fake_build(task: Task):
        return normalized_task, "smart", "prompt"

    async def fake_run(**kwargs):
        return RunResult(waiting_for_user=True, question="", timeout_s=90)

    monkeypatch.setattr(loop._context_builder, "build", fake_build)
    monkeypatch.setattr(loop._run_executor, "run", fake_run)

    result = await loop._process(Task(content="Fix parser", source="api", author="Josh", metadata={"task_id": "task-1"}))

    assert result.status == "waiting_for_user"
    assert memory.created == 0
    assert memory.waiting == [("task-1", result.task.metadata, "")]


@pytest.mark.asyncio
async def test_process_waiting_skips_optional_persistence_without_waiting_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ReadOnlyMemory:
        def __init__(self) -> None:
            self.lookups: list[str] = []

        async def get_task_record(self, task_id: str):
            self.lookups.append(task_id)
            return None

    memory = _ReadOnlyMemory()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)

    async def fake_build(task: Task):
        return Task(content="Fix parser", source="api", author="Josh", metadata={"task_id": "task-1"}), "smart", "prompt"

    async def fake_run(**kwargs):
        return RunResult(waiting_for_user=True, question="What environment?", timeout_s=90)

    monkeypatch.setattr(loop._context_builder, "build", fake_build)
    monkeypatch.setattr(loop._run_executor, "run", fake_run)

    result = await loop._process(Task(content="Fix parser", source="api", author="Josh", metadata={"task_id": "task-1"}))

    assert result.status == "waiting_for_user"
    assert memory.lookups == []


@pytest.mark.asyncio
async def test_process_waiting_for_a2a_task_fails_without_suspending(event_collector, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _Memory()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)
    normalized_task = Task(content="Fix parser", source="a2a", author="peer-1")

    async def fake_build(task: Task):
        return normalized_task, "smart", "prompt"

    async def fake_run(**kwargs):
        return RunResult(waiting_for_user=True, question="What environment?", timeout_s=90)

    monkeypatch.setattr(loop._context_builder, "build", fake_build)
    monkeypatch.setattr(loop._run_executor, "run", fake_run)

    result = await loop._process(Task(content="Fix parser", source="a2a", author="peer-1"))

    assert result.status == "failed"
    assert result.success is False
    assert result.question == "What environment?"
    assert loop.wait_registry.get(result.task.metadata["task_id"]) is None
    assert result.task.metadata["task_id"] not in memory.rows
    assert [type(event) for event in event_collector] == [TaskStartEvent, TaskErrorEvent]


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
async def test_restore_waiting_tasks_skips_invalid_rows_and_normalizes_timeout() -> None:
    memory = _Memory()
    memory.rows = {
        "missing-question": {
            "task_id": "missing-question",
            "source": "discord",
            "author": "Josh",
            "content": "Fix parser",
            "status": "waiting_for_user",
            "metadata": {"task_id": "missing-question", "wait_state": {"timeout_s": 0}},
        },
        "valid": {
            "task_id": "valid",
            "source": "discord",
            "author": "Josh",
            "content": "Fix parser",
            "status": "waiting_for_user",
            "metadata": {
                "task_id": "valid",
                "wait_state": {
                    "question": "What environment?",
                    "timeout_s": 0,
                    "channel_id": "7",
                    "message_id": "11",
                    "prompt_message_id": "42",
                },
            },
        },
    }
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)

    restored = await loop.restore_waiting_tasks()

    assert restored == 1
    suspended = loop.wait_registry.get("valid")
    assert suspended is not None
    assert suspended.timeout_s == 1
    assert suspended.channel_id == 7
    assert suspended.message_id == 11
    assert suspended.prompt_message_id == 42


@pytest.mark.asyncio
async def test_restore_waiting_tasks_handles_non_dict_wait_state_and_missing_prompt_message() -> None:
    memory = _Memory()
    memory.rows = {
        "bad": {
            "task_id": "bad",
            "source": "discord",
            "author": "Josh",
            "content": "Fix parser",
            "status": "waiting_for_user",
            "metadata": {"task_id": "bad", "wait_state": "not-a-dict"},
        },
        "valid": {
            "task_id": "valid",
            "source": "discord",
            "author": "Josh",
            "content": "Fix parser",
            "status": "waiting_for_user",
            "metadata": {
                "task_id": "valid",
                "wait_state": {
                    "question": "What environment?",
                    "timeout_s": 90,
                    "channel_id": 7,
                    "message_id": 11,
                    "prompt_message_id": None,
                },
            },
        },
    }
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)

    restored = await loop.restore_waiting_tasks()

    assert restored == 1
    suspended = loop.wait_registry.get("valid")
    assert suspended is not None
    assert suspended.prompt_message_id is None


@pytest.mark.asyncio
async def test_restore_waiting_tasks_returns_zero_when_rows_do_not_restore() -> None:
    memory = _Memory()
    memory.rows = {
        "missing-question": {
            "task_id": "missing-question",
            "source": "discord",
            "author": "Josh",
            "content": "Fix parser",
            "status": "waiting_for_user",
            "metadata": {"task_id": "missing-question", "wait_state": {"timeout_s": 30}},
        }
    }
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)

    assert await loop.restore_waiting_tasks() == 0


@pytest.mark.asyncio
async def test_restore_pending_tasks_requeues_running_and_queued_rows() -> None:
    memory = _Memory()
    memory.rows["task-queued"] = {
        "task_id": "task-queued",
        "source": "api",
        "author": "Josh",
        "content": "Queued work",
        "status": "queued",
        "metadata": {"task_id": "task-queued", "session_id": "api:task-queued"},
    }
    memory.rows["task-running"] = {
        "task_id": "task-running",
        "source": "api",
        "author": "Josh",
        "content": "Running work",
        "status": "running",
        "metadata": {"task_id": "task-running", "session_id": "api:task-running"},
    }
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)

    restored = await loop.restore_pending_tasks()

    assert restored == 2
    assert loop.queue.qsize() == 2
    assert memory.rows["task-running"]["status"] == "queued"


@pytest.mark.asyncio
async def test_restore_pending_tasks_skips_waiting_entries_and_builds_fallback_session_ids() -> None:
    memory = _Memory()
    memory.rows = {
        "task-waiting": {
            "task_id": "task-waiting",
            "source": "api",
            "author": "Josh",
            "content": "Waiting work",
            "status": "waiting_for_user",
            "metadata": {"task_id": "task-waiting", "wait_state": {"question": "?", "timeout_s": 10}},
        },
        "task-running": {
            "task_id": "task-running",
            "source": "api",
            "author": "Josh",
            "content": "Running work",
            "status": "running",
            "metadata": {"task_id": "task-running"},
        },
    }
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)
    loop.wait_registry.suspend(
        task_id="task-waiting",
        source="api",
        author="Josh",
        content="Waiting work",
        channel_id=0,
        message_id=0,
        metadata=memory.rows["task-waiting"]["metadata"],
        question="?",
        timeout_s=10,
        base_prompt="",
        tier="smart",
    )

    restored = await loop.restore_pending_tasks()
    restored_task = await loop.queue.get()

    assert restored == 1
    assert restored_task.metadata["session_id"] == "api:task-running"
    assert memory.rows["task-running"]["status"] == "queued"


@pytest.mark.asyncio
async def test_restore_pending_tasks_skips_blank_and_already_pending_ids() -> None:
    memory = _Memory()
    memory.rows = {
        "blank": {
            "task_id": "   ",
            "source": "api",
            "author": "Josh",
            "content": "Blank work",
            "status": "queued",
            "metadata": {},
        },
        "task-pending": {
            "task_id": "task-pending",
            "source": "api",
            "author": "Josh",
            "content": "Pending work",
            "status": "queued",
            "metadata": {"task_id": "task-pending"},
        },
    }
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)
    loop.wait_registry.suspend(
        task_id="task-pending",
        source="api",
        author="Josh",
        content="Pending work",
        channel_id=0,
        message_id=0,
        metadata={"task_id": "task-pending"},
        question="?",
        timeout_s=10,
        base_prompt="",
        tier="smart",
    )

    restored = await loop.restore_pending_tasks()

    assert restored == 0
    assert loop.queue.qsize() == 0


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
async def test_process_non_rate_limit_failure_clears_journal_and_marks_session_failed(
    event_collector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _Memory()
    postgres = _Postgres()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory, postgres_store=postgres)
    calls: list[str] = []

    async def fake_build(task: Task):
        return task, "smart", "prompt"

    async def fake_run(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(loop._context_builder, "build", fake_build)
    monkeypatch.setattr(loop._run_executor, "run", fake_run)
    monkeypatch.setattr(loop._journal, "clear", lambda: calls.append("clear"))

    result = await loop._process(Task(content="Fix parser", source="discord", author="Josh", metadata={"session_id": "discord:1:2", "task_id": "task-1"}))

    assert result.status == "failed"
    assert result.output == "Error: boom"
    assert calls == ["clear"]
    assert memory.sessions["discord:1:2"]["status"] == "failed"
    assert memory.checkpoints["task-1"]["summary"].startswith("Task failed: boom")
    assert isinstance(event_collector[1], TaskErrorEvent)


@pytest.mark.asyncio
async def test_process_failure_tolerates_checkpoint_write_error(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _Memory()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)

    async def fake_build(task: Task):
        return task, "smart", "prompt"

    async def fake_run(**kwargs):
        raise RuntimeError("429 limited")

    async def broken_checkpoint(**kwargs) -> None:
        raise RuntimeError("checkpoint boom")

    monkeypatch.setattr(loop._context_builder, "build", fake_build)
    monkeypatch.setattr(loop._run_executor, "run", fake_run)
    monkeypatch.setattr(memory, "save_task_checkpoint", broken_checkpoint)

    result = await loop._process(Task(content="Fix parser", metadata={"task_id": "task-1", "session_id": "session-1"}))

    assert result.status == "failed"
    assert result.output == "Error: 429 limited"


@pytest.mark.asyncio
async def test_process_failure_tolerates_session_status_error(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _Memory()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)

    async def fake_build(task: Task):
        return task, "smart", "prompt"

    async def fake_run(**kwargs):
        raise RuntimeError("boom")

    checkpoints: list[str] = []

    async def broken_status(*args, **kwargs) -> None:
        raise RuntimeError("status boom")

    async def track_checkpoint(**kwargs) -> None:
        checkpoints.append(kwargs["task_id"])
        await _Memory.save_task_checkpoint(memory, **kwargs)

    monkeypatch.setattr(loop._context_builder, "build", fake_build)
    monkeypatch.setattr(loop._run_executor, "run", fake_run)
    monkeypatch.setattr(memory, "set_session_status", broken_status)
    monkeypatch.setattr(memory, "save_task_checkpoint", track_checkpoint)

    result = await loop._process(Task(content="Fix parser", metadata={"task_id": "task-1", "session_id": "session-1"}))

    assert result.status == "failed"
    assert checkpoints == ["task-1", "task-1"]
    assert memory.checkpoints["task-1"]["summary"].startswith("Task failed: boom")


@pytest.mark.asyncio
async def test_summarize_context_returns_fallback_on_error() -> None:
    class _FailAgent(_StubAgent):
        async def run(self, prompt: str, usage_limits=None):
            raise RuntimeError("nope")

    loop = AgentLoop({"smart": _FailAgent(), "fast": _FailAgent(), "best": _FailAgent()})

    summary = await loop._summarize_context(Task(content="Long task"), "history")

    assert "Context was compressed" in summary


@pytest.mark.asyncio
async def test_summarize_context_returns_stripped_summary_on_success() -> None:
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})

    summary = await loop._summarize_context(Task(content="Long task"), "history")

    assert summary == "summary"


@pytest.mark.asyncio
async def test_loop_wrapper_methods_delegate_to_services(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})
    calls: list[str] = []

    async def fake_reflect(*args) -> None:
        calls.append("reflect")

    async def fake_success(*args) -> None:
        calls.append("success")

    async def fake_failure(*args) -> None:
        calls.append("failure")

    async def fake_update() -> None:
        calls.append("update")

    async def fake_heartbeat(*, is_busy) -> None:
        calls.append(f"heartbeat:{is_busy}")

    monkeypatch.setattr(loop._reflection_service, "reflect", fake_reflect)
    monkeypatch.setattr(loop._reflection_service, "_reflect_on_success", fake_success)
    monkeypatch.setattr(loop._reflection_service, "_reflect_on_failure", fake_failure)
    monkeypatch.setattr(loop._reflection_service, "update_memory_md", fake_update)
    monkeypatch.setattr(loop._heartbeat_service, "heartbeat", fake_heartbeat)

    task = Task(content="Fix parser")
    result = TaskResult(task=task, output="done", success=True, elapsed_ms=1.0)
    await loop._reflect(task, result)
    await loop._reflect_on_success(task, result)
    await loop._reflect_on_failure(task, result)
    await loop._update_memory_md()
    await loop._heartbeat()

    assert calls == ["reflect", "success", "failure", "update", "heartbeat:False"]


@pytest.mark.asyncio
async def test_is_answer_acceptable_rejects_empty_and_status_like_outputs() -> None:
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})
    task = Task(content="Fix parser")

    assert await loop._is_answer_acceptable(task=task, output="", tool_calls=0) is False
    assert await loop._is_answer_acceptable(task=task, output="[ERROR] nope", tool_calls=1) is False
    assert await loop._is_answer_acceptable(task=task, output="Looks good", tool_calls=0) is True


@pytest.mark.asyncio
async def test_is_answer_acceptable_uses_validator_and_fallback_word_count() -> None:
    class _AnsweringAgent:
        def __init__(self, output: str) -> None:
            self.output = output

        async def run(self, prompt: str, usage_limits=None):
            return SimpleNamespace(output=self.output)

    task = Task(content="Fix parser")
    positive = AgentLoop({"smart": _StubAgent(), "fast": _AnsweringAgent("ANSWERED"), "best": _StubAgent()})
    negative = AgentLoop({"smart": _StubAgent(), "fast": _AnsweringAgent("NOT_ANSWERED"), "best": _StubAgent()})

    assert await positive._is_answer_acceptable(task=task, output="draft answer", tool_calls=2) is True
    assert await negative._is_answer_acceptable(task=task, output="draft answer", tool_calls=2) is False

    class _FailAgent(_StubAgent):
        async def run(self, prompt: str, usage_limits=None):
            raise RuntimeError("validator failed")

    fallback = AgentLoop({"smart": _StubAgent(), "fast": _FailAgent(), "best": _StubAgent()})
    assert (
        await fallback._is_answer_acceptable(
            task=task,
            output="this draft has enough words to count",
            tool_calls=2,
        )
        is True
    )


@pytest.mark.asyncio
async def test_repair_user_answer_returns_agent_output_or_original_on_error() -> None:
    class _RepairAgent:
        async def run(self, prompt: str, usage_limits=None):
            return SimpleNamespace(output="Repaired answer")

    repaired_loop = AgentLoop({"smart": _StubAgent(), "fast": _RepairAgent(), "best": _StubAgent()})
    original_loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})

    async def fail_run(prompt: str, usage_limits=None):
        raise RuntimeError("repair failed")

    original_loop.agents["fast"].run = fail_run  # type: ignore[method-assign]

    assert await repaired_loop._repair_user_answer(task=Task(content="Fix parser"), output="notes") == "Repaired answer"
    assert await original_loop._repair_user_answer(task=Task(content="Fix parser"), output="notes") == "notes"


@pytest.mark.asyncio
async def test_ensure_answer_required_uses_default_fallback_when_both_outputs_are_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})

    async def reject(**kwargs):
        return False

    async def blank_repair(**kwargs):
        return "   "

    monkeypatch.setattr(loop, "_is_answer_acceptable", reject)
    monkeypatch.setattr(loop, "_repair_user_answer", blank_repair)

    output, answered = await loop._ensure_answer_required(
        task=Task(content="Fix parser"),
        output="",
        tool_calls=2,
    )

    assert answered is False
    assert "I could not produce a reliable final answer" in output


@pytest.mark.asyncio
async def test_ensure_answer_required_prefers_repaired_then_repair_then_original_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})
    verdicts = iter([False, True])

    async def accept_repaired(**kwargs):
        return next(verdicts)

    async def repaired(**kwargs):
        return "repaired answer"

    monkeypatch.setattr(loop, "_is_answer_acceptable", accept_repaired)
    monkeypatch.setattr(loop, "_repair_user_answer", repaired)
    output, answered = await loop._ensure_answer_required(
        task=Task(content="Fix parser"),
        output="draft",
        tool_calls=2,
    )
    assert (output, answered) == ("repaired answer", True)

    async def reject(**kwargs):
        return False

    monkeypatch.setattr(loop, "_is_answer_acceptable", reject)
    monkeypatch.setattr(loop, "_repair_user_answer", repaired)
    output, answered = await loop._ensure_answer_required(
        task=Task(content="Fix parser"),
        output="draft",
        tool_calls=2,
    )
    assert (output, answered) == ("repaired answer", False)

    async def blank_repair(**kwargs):
        return "   "

    monkeypatch.setattr(loop, "_repair_user_answer", blank_repair)
    output, answered = await loop._ensure_answer_required(
        task=Task(content="Fix parser"),
        output="original answer",
        tool_calls=2,
    )
    assert (output, answered) == ("original answer", False)


@pytest.mark.asyncio
async def test_maybe_promote_memory_fact_saves_preferences_and_operator_corrections() -> None:
    memory = _Memory()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory)

    await loop._maybe_promote_memory_fact(
        task=Task(
            content="remember to use pytest",
            source="discord",
            author="Josh",
            metadata={"session_id": "discord:1:2"},
        )
    )
    assert memory.checkpoints["memory_fact"]["content"] == "remember to use pytest"
    assert memory.checkpoints["memory_fact"]["metadata"]["session_id"] == "discord:1:2"

    memory.checkpoints.clear()
    await loop._maybe_promote_memory_fact(task=Task(content="investigate parser bug", source="discord"))
    assert "memory_fact" not in memory.checkpoints

    no_memory_loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})
    await no_memory_loop._maybe_promote_memory_fact(task=Task(content="remember this", source="discord"))

    await loop._maybe_promote_memory_fact(
        task=Task(
            content="the repo is available inside the /workspace directory so check the file system first and do not guess the repo or hostname",
            source="discord",
            author="Josh",
        )
    )
    assert "file system" in memory.checkpoints["memory_fact"]["content"].lower() or "guess" in memory.checkpoints["memory_fact"]["content"].lower()

    memory.checkpoints.clear()
    await loop._maybe_promote_memory_fact(task=Task(content="remember " + ("x" * 1300), source="discord"))
    assert "memory_fact" not in memory.checkpoints


@pytest.mark.asyncio
async def test_process_fails_when_answer_requirement_not_met(event_collector, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _Memory()
    postgres = _Postgres()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory, postgres_store=postgres)
    normalized_task = Task(content="Fix parser", source="discord", author="Josh", channel_id=7)

    async def fake_build(task: Task):
        return normalized_task, "smart", "prompt"

    async def fake_run(**kwargs):
        return RunResult(output="notes", tool_calls=2)

    monkeypatch.setattr(loop._context_builder, "build", fake_build)
    monkeypatch.setattr(loop._run_executor, "run", fake_run)
    monkeypatch.setattr(loop, "_ensure_answer_required", lambda **kwargs: asyncio.sleep(0, result=("not enough evidence", False)))

    result = await loop._process(Task(content="Fix parser", source="discord", author="Josh", channel_id=7))

    assert result.status == "failed"
    assert result.answered_user is False
    assert loop._success_count == 0
    assert memory.sessions[result.task.metadata["session_id"]]["status"] == "failed"
    assert postgres.calls[1][0] == "done"
    assert postgres.calls[1][2] is False
    assert [type(event) for event in event_collector] == [TaskStartEvent, TaskErrorEvent]


@pytest.mark.asyncio
async def test_ensure_answer_required_returns_original_when_already_acceptable(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})

    async def accept(**kwargs):
        return True

    monkeypatch.setattr(loop, "_is_answer_acceptable", accept)

    output, answered = await loop._ensure_answer_required(
        task=Task(content="Fix parser"),
        output="final answer",
        tool_calls=0,
    )

    assert output == "final answer"
    assert answered is True


def test_coerce_int_helpers_handle_invalid_values() -> None:
    assert AgentLoop._coerce_int("5") == 5
    assert AgentLoop._coerce_int("bad", default=9) == 9
    assert AgentLoop._coerce_int_or_none("7") == 7
    assert AgentLoop._coerce_int_or_none("bad") is None


@pytest.mark.asyncio
async def test_execute_task_marks_running_completes_a2a_and_skips_completion_for_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _Memory()
    postgres = _Postgres()
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, memory_store=memory, postgres_store=postgres)
    created: list[asyncio.coroutines] = []

    async def fake_process(task: Task) -> TaskResult:
        return TaskResult(task=task, output="ok", success=True, elapsed_ms=1.0, status="succeeded")

    monkeypatch.setattr(loop, "_process", fake_process)
    monkeypatch.setattr(asyncio, "create_task", lambda coro: created.append(coro))

    task = Task(content="delegated", source="a2a", metadata={})
    result = await loop._execute_task(task)

    assert result.status == "succeeded"
    assert memory.running == [task.metadata["task_id"]]
    assert task.metadata["session_id"].startswith("a2a:")
    assert ("complete", task.metadata["task_id"], "ok") in postgres.calls
    assert created
    for coro in created:
        coro.close()

    created.clear()

    async def fake_waiting(task: Task) -> TaskResult:
        return TaskResult(task=task, output="", success=None, elapsed_ms=1.0, status="waiting_for_user")

    monkeypatch.setattr(loop, "_process", fake_waiting)
    waiting_task = Task(content="delegated", source="a2a", metadata={})
    waiting = await loop._execute_task(waiting_task)

    assert waiting.status == "waiting_for_user"
    assert not any(call[0] == "complete" and call[1] == waiting_task.metadata["task_id"] for call in postgres.calls)


@pytest.mark.asyncio
async def test_execute_task_handles_none_metadata_complete_failure_and_no_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingPostgres(_Postgres):
        async def complete_task(self, *args) -> None:
            raise RuntimeError("complete boom")

    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()}, postgres_store=_FailingPostgres())
    created: list[asyncio.coroutines] = []

    async def fake_process(task: Task) -> TaskResult:
        return TaskResult(task=task, output="ok", success=True, elapsed_ms=1.0, status="succeeded")

    monkeypatch.setattr(loop, "_process", fake_process)
    monkeypatch.setattr(asyncio, "create_task", lambda coro: created.append(coro))

    task = Task(content="delegated", source="a2a", metadata=None)  # type: ignore[arg-type]
    result = await loop._execute_task(task)

    assert result.status == "succeeded"
    assert task.metadata["session_id"].startswith("a2a:")
    for coro in created:
        coro.close()


@pytest.mark.asyncio
async def test_restore_helpers_handle_missing_memory_and_run_forever_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop({"smart": _StubAgent(), "fast": _StubAgent(), "best": _StubAgent()})
    assert await loop.restore_waiting_tasks() == 0
    assert await loop.restore_pending_tasks() == 0

    async def fake_wait_for(awaitable, timeout):
        awaitable.close()
        raise asyncio.CancelledError

    monkeypatch.setattr(loop_module.asyncio, "wait_for", fake_wait_for)
    await loop.run_forever()
