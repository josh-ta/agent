from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import agent.control_plane.app as app_module
from agent.control_plane.app import SseBroker, _is_runtime_ready, _require_runtime, _task_response_from_row, build_app, create_app
from agent.events import ProgressEvent, TaskQueuedEvent, bridge
from agent.loop import Task
from agent.task_waits import SuspendedTask, TaskWaitRegistry


class _FakeSQLite:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.rows: dict[str, dict[str, object]] = {}
        self.sessions: dict[str, dict[str, object]] = {}
        self.turns: dict[str, list[dict[str, object]]] = {}

    async def create_task_record(
        self,
        *,
        task_id: str,
        source: str,
        author: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        row = {
            "task_id": task_id,
            "source": source,
            "author": author,
            "content": content,
            "status": "queued",
            "result": None,
            "error": None,
            "success": 0,
            "elapsed_ms": None,
            "tool_calls": None,
            "created_ts": 1.0,
            "started_ts": None,
            "finished_ts": None,
            "updated_ts": 1.0,
            "metadata": metadata or {},
        }
        self.created.append(row)
        self.rows[task_id] = row

    async def get_task_record(self, task_id: str) -> dict[str, object] | None:
        return self.rows.get(task_id)

    async def mark_task_queued(self, task_id: str, *, metadata=None) -> None:
        row = self.rows[task_id]
        row["status"] = "queued"
        if metadata is not None:
            row["metadata"] = metadata

    async def ensure_session(
        self,
        *,
        session_id: str,
        source: str,
        channel_id: int = 0,
        title: str = "",
        status: str = "active",
        pending_task_id: str = "",
        metadata=None,
    ) -> None:
        self.sessions[session_id] = {
            "session_id": session_id,
            "source": source,
            "channel_id": channel_id,
            "status": status,
            "title": title,
            "summary": "",
            "pending_task_id": pending_task_id,
            "metadata": metadata or {},
        }

    async def append_session_turn(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        turn_kind: str = "message",
        task_id: str = "",
        metadata=None,
    ) -> None:
        self.turns.setdefault(session_id, []).append(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "turn_kind": turn_kind,
                "task_id": task_id,
                "metadata": metadata or {},
            }
        )
        session = self.sessions.setdefault(
            session_id,
            {
                "session_id": session_id,
                "source": "api",
                "channel_id": 0,
                "status": "active",
                "title": "",
                "summary": "",
                "pending_task_id": "",
                "metadata": {},
            },
        )
        session["summary"] = f"Latest turn: {content[:80]}"

    async def get_session(self, session_id: str) -> dict[str, object] | None:
        return self.sessions.get(session_id)

    async def list_session_turns(self, session_id: str, limit: int = 20) -> list[dict[str, object]]:
        return self.turns.get(session_id, [])[-limit:]


class _FakeLoop:
    def __init__(self) -> None:
        self.enqueued: list[Task] = []
        self.wait_registry = TaskWaitRegistry()

    async def enqueue(self, task: Task) -> None:
        self.enqueued.append(task)

    def build_resumed_task(
        self,
        *,
        suspended: SuspendedTask,
        answer: str,
        author: str,
        source: str,
        metadata_overrides=None,
    ) -> Task:
        metadata = self.wait_registry.build_resumed_metadata(suspended, answer=answer, resumed_from=source)
        if metadata_overrides:
            metadata.update(metadata_overrides)
        return Task(content=suspended.content, source=source, author=author, metadata=metadata)


def _build_app() -> tuple[TestClient, _FakeSQLite, _FakeLoop]:
    sqlite = _FakeSQLite()
    loop = _FakeLoop()
    runtime = SimpleNamespace(sqlite=sqlite, loop=loop, bot=None)
    app = create_app(runtime, start_background_runtime=False, shutdown_runtime=False)
    return TestClient(app), sqlite, loop


def test_health_and_readiness_endpoints() -> None:
    client, _, _ = _build_app()
    with client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json() == {"status": "ready"}


def test_readyz_returns_service_unavailable_when_runtime_not_ready() -> None:
    client, _, _ = _build_app()
    with client:
        client.app.state.ready = False
        response = client.get("/readyz")

        assert response.status_code == 503
        assert response.json()["error_code"] == "service_unavailable"


def test_post_tasks_persists_and_enqueues() -> None:
    client, sqlite, loop = _build_app()
    with client:
        response = client.post("/tasks", json={"content": "  summarize repo  ", "author": "tester"})

        assert response.status_code == 202
        payload = response.json()
        assert payload["status"] == "queued"
        assert len(sqlite.created) == 1
        assert sqlite.created[0]["task_id"] == payload["id"]
        assert len(loop.enqueued) == 1
        assert loop.enqueued[0].content == "summarize repo"
        assert loop.enqueued[0].metadata["task_id"] == payload["id"]
        assert "session_id" in loop.enqueued[0].metadata


def test_post_tasks_still_works_without_optional_session_storage() -> None:
    class _MinimalSQLite:
        def __init__(self) -> None:
            self.created: list[dict[str, object]] = []

        async def create_task_record(self, **kwargs) -> None:
            self.created.append(kwargs)

    class _Loop:
        def __init__(self) -> None:
            self.enqueued: list[Task] = []

        async def enqueue(self, task: Task) -> None:
            self.enqueued.append(task)

    sqlite = _MinimalSQLite()
    loop = _Loop()
    runtime = SimpleNamespace(sqlite=sqlite, loop=loop, bot=None)
    client = TestClient(create_app(runtime, start_background_runtime=False, shutdown_runtime=False))

    with client:
        response = client.post("/tasks", json={"content": "summarize repo", "author": "tester"})

    assert response.status_code == 202
    assert len(sqlite.created) == 1
    assert len(loop.enqueued) == 1


def test_post_tasks_rejects_blank_content() -> None:
    client, _, _ = _build_app()
    with client:
        response = client.post("/tasks", json={"content": "   "})

        assert response.status_code == 400
        assert response.json()["error_code"] == "invalid_request"


def test_get_task_returns_persisted_status() -> None:
    client, sqlite, _ = _build_app()
    sqlite.rows["task-1"] = {
        "task_id": "task-1",
        "source": "api",
        "author": "tester",
        "content": "do thing",
        "status": "succeeded",
        "result": "done",
        "error": None,
        "success": 1,
        "elapsed_ms": 12.5,
        "tool_calls": 2,
        "created_ts": 1.0,
        "started_ts": 2.0,
        "finished_ts": 3.0,
        "updated_ts": 3.0,
    }

    with client:
        response = client.get("/tasks/task-1")

        assert response.status_code == 200
        assert response.json()["status"] == "succeeded"
        assert response.json()["result"] == "done"


def test_get_task_returns_typed_not_found_error() -> None:
    client, _, _ = _build_app()
    with client:
        response = client.get("/tasks/missing")

        assert response.status_code == 404
        assert response.json()["error_code"] == "task_not_found"


def test_resume_waiting_task_enqueues_followup_input() -> None:
    client, sqlite, loop = _build_app()
    sqlite.rows["task-1"] = {
        "task_id": "task-1",
        "source": "api",
        "author": "tester",
        "content": "do thing",
        "status": "waiting_for_user",
        "result": None,
        "error": None,
        "success": 0,
        "elapsed_ms": None,
        "tool_calls": None,
        "created_ts": 1.0,
        "started_ts": 2.0,
        "finished_ts": None,
        "updated_ts": 3.0,
        "metadata": {"task_id": "task-1", "wait_state": {"question": "Which env?"}},
    }
    loop.wait_registry.suspend(
        task_id="task-1",
        source="api",
        author="tester",
        content="do thing",
        channel_id=0,
        message_id=0,
        metadata={"task_id": "task-1"},
        question="Which env?",
        timeout_s=300,
        base_prompt="prompt",
        tier="smart",
    )

    with client:
        response = client.post("/tasks/task-1/input", json={"content": "production", "author": "tester"})

        assert response.status_code == 202
        assert response.json()["status"] == "queued"
        assert loop.enqueued[0].metadata["resume_context"]["answer"] == "production"


def test_resume_waiting_task_still_works_without_optional_session_turn_append() -> None:
    class _SQLite:
        def __init__(self) -> None:
            self.rows = {
                "task-1": {
                    "task_id": "task-1",
                    "source": "api",
                    "author": "tester",
                    "content": "do thing",
                    "status": "waiting_for_user",
                    "metadata": {"task_id": "task-1", "wait_state": {"question": "Which env?"}},
                }
            }
            self.marked: list[tuple[str, dict]] = []

        async def get_task_record(self, task_id: str) -> dict[str, object] | None:
            return self.rows.get(task_id)

        async def mark_task_queued(self, task_id: str, *, metadata=None) -> None:
            self.marked.append((task_id, metadata or {}))

    sqlite = _SQLite()
    loop = _FakeLoop()
    loop.wait_registry.suspend(
        task_id="task-1",
        source="api",
        author="tester",
        content="do thing",
        channel_id=0,
        message_id=0,
        metadata={"task_id": "task-1"},
        question="Which env?",
        timeout_s=300,
        base_prompt="prompt",
        tier="smart",
    )
    runtime = SimpleNamespace(sqlite=sqlite, loop=loop, bot=None)
    client = TestClient(create_app(runtime, start_background_runtime=False, shutdown_runtime=False))

    with client:
        response = client.post("/tasks/task-1/input", json={"content": "production", "author": "tester"})

    assert response.status_code == 202
    assert sqlite.marked
    assert loop.enqueued[0].metadata["resume_context"]["answer"] == "production"


def test_resume_waiting_task_rejects_missing_task() -> None:
    client, _, _ = _build_app()
    with client:
        response = client.post("/tasks/missing/input", json={"content": "production", "author": "tester"})

        assert response.status_code == 404
        assert response.json()["error_code"] == "task_not_found"


def test_resume_waiting_task_rejects_non_waiting_status() -> None:
    client, sqlite, _ = _build_app()
    sqlite.rows["task-1"] = {
        "task_id": "task-1",
        "source": "api",
        "author": "tester",
        "content": "do thing",
        "status": "running",
        "result": None,
        "error": None,
        "success": 0,
        "elapsed_ms": None,
        "tool_calls": None,
        "created_ts": 1.0,
        "started_ts": 2.0,
        "finished_ts": None,
        "updated_ts": 3.0,
        "metadata": {"task_id": "task-1"},
    }

    with client:
        response = client.post("/tasks/task-1/input", json={"content": "production", "author": "tester"})

        assert response.status_code == 400
        assert response.json()["error_code"] == "invalid_request"


def test_resume_waiting_task_rejects_missing_loaded_suspend_state() -> None:
    client, sqlite, _ = _build_app()
    sqlite.rows["task-1"] = {
        "task_id": "task-1",
        "source": "api",
        "author": "tester",
        "content": "do thing",
        "status": "waiting_for_user",
        "result": None,
        "error": None,
        "success": 0,
        "elapsed_ms": None,
        "tool_calls": None,
        "created_ts": 1.0,
        "started_ts": 2.0,
        "finished_ts": None,
        "updated_ts": 3.0,
        "metadata": {"task_id": "task-1", "wait_state": {"question": "Which env?"}},
    }

    with client:
        response = client.post("/tasks/task-1/input", json={"content": "production", "author": "tester"})

        assert response.status_code == 503
        assert response.json()["error_code"] == "service_unavailable"


def test_resume_waiting_task_rejects_blank_trimmed_content() -> None:
    client, sqlite, loop = _build_app()
    sqlite.rows["task-1"] = {
        "task_id": "task-1",
        "source": "api",
        "author": "tester",
        "content": "do thing",
        "status": "waiting_for_user",
        "result": None,
        "error": None,
        "success": 0,
        "elapsed_ms": None,
        "tool_calls": None,
        "created_ts": 1.0,
        "started_ts": 2.0,
        "finished_ts": None,
        "updated_ts": 3.0,
        "metadata": {"task_id": "task-1", "wait_state": {"question": "Which env?"}},
    }
    loop.wait_registry.suspend(
        task_id="task-1",
        source="api",
        author="tester",
        content="do thing",
        channel_id=0,
        message_id=0,
        metadata={"task_id": "task-1"},
        question="Which env?",
        timeout_s=300,
        base_prompt="prompt",
        tier="smart",
    )

    with client:
        response = client.post("/tasks/task-1/input", json={"content": "   ", "author": "tester"})

        assert response.status_code == 400
        assert response.json()["error_code"] == "invalid_request"


def test_conversation_endpoints_return_session_and_turns() -> None:
    client, sqlite, _ = _build_app()
    sqlite.sessions["api:task-1"] = {
        "session_id": "api:task-1",
        "source": "api",
        "channel_id": 0,
        "status": "active",
        "title": "Investigate issue",
        "summary": "Latest turn: investigate issue",
        "pending_task_id": "task-1",
        "metadata": {},
    }
    sqlite.turns["api:task-1"] = [
        {"session_id": "api:task-1", "role": "user", "content": "Investigate issue", "turn_kind": "message"}
    ]

    with client:
        session_response = client.get("/conversations/api:task-1")
        turns_response = client.get("/conversations/api:task-1/turns")

        assert session_response.status_code == 200
        assert session_response.json()["id"] == "api:task-1"
        assert turns_response.status_code == 200
        assert turns_response.json()[0]["content"] == "Investigate issue"


def test_conversation_endpoint_returns_not_found() -> None:
    client, _, _ = _build_app()
    with client:
        response = client.get("/conversations/api:missing")

        assert response.status_code == 404
        assert response.json()["error_code"] == "task_not_found"


def test_conversation_endpoints_return_service_unavailable_without_storage_methods() -> None:
    client, _, _ = _build_app()
    with client:
        client.app.state.runtime.sqlite = SimpleNamespace()

        session_response = client.get("/conversations/api:task-1")
        turns_response = client.get("/conversations/api:task-1/turns")

        assert session_response.status_code == 503
        assert turns_response.status_code == 503
        assert session_response.json()["error_code"] == "service_unavailable"
        assert turns_response.json()["error_code"] == "service_unavailable"


def test_validation_errors_return_typed_invalid_request() -> None:
    client, _, _ = _build_app()
    with client:
        response = client.post("/tasks", json={"content": ""})

        assert response.status_code == 422
        assert response.json()["error_code"] == "invalid_request"


def test_sse_broker_filters_and_serializes_by_task_id() -> None:
    broker = SseBroker()
    subscriber_id, queue = broker.subscribe("task-1")
    try:
        asyncio.run(broker.publish(ProgressEvent(message="ignore", task_id="other-task")))
        asyncio.run(broker.publish(TaskQueuedEvent(task_id="task-1", content="do thing", source="api")))
        payload = asyncio.run(asyncio.wait_for(queue.get(), timeout=1))

        assert "event: task_queued" in payload
        assert '"task_id":"task-1"' in payload
        assert "other-task" not in payload
    finally:
        broker.unsubscribe(subscriber_id)


def test_sse_broker_drops_oldest_when_queue_is_full_and_handles_plain_dict_events() -> None:
    broker = SseBroker()
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
    queue.put_nowait("old")
    broker._subscribers["sub-1"] = (queue, None)

    asyncio.run(broker.publish({"kind": "custom", "task_id": "task-1"}))

    payload = asyncio.run(asyncio.wait_for(queue.get(), timeout=1))
    assert "event: custom" in payload
    assert "old" not in payload


def test_events_route_streams_event_and_unregisters_subscriber() -> None:
    client, _, _ = _build_app()

    class _FakeRequest:
        def __init__(self, app) -> None:
            self.app = app
            self._checks = 0

        async def is_disconnected(self) -> bool:
            self._checks += 1
            return False

    async def _exercise_stream():
        with client:
            route = next(route for route in client.app.routes if getattr(route, "path", "") == "/events")
            request = _FakeRequest(client.app)
            response = await route.endpoint(request, task_id="task-1")
            broker = client.app.state.sse_broker
            body = response.body_iterator

            connected = await anext(body)
            assert broker._subscribers

            await broker.publish(TaskQueuedEvent(task_id="task-1", content="do thing", source="api"))
            event = await anext(body)

            await body.aclose()
            return connected, event, broker._subscribers

    connected, event, subscribers = asyncio.run(_exercise_stream())

    assert connected == ": connected\n\n"
    assert "event: task_queued" in event
    assert '"task_id":"task-1"' in event
    assert subscribers == {}


def test_events_route_emits_ping_then_disconnect(monkeypatch) -> None:
    client, _, _ = _build_app()

    class _FakeRequest:
        def __init__(self, app) -> None:
            self.app = app
            self._checks = 0

        async def is_disconnected(self) -> bool:
            self._checks += 1
            return self._checks >= 2

    async def fake_wait_for(awaitable, timeout):
        awaitable.close()
        raise TimeoutError

    monkeypatch.setattr(app_module.asyncio, "wait_for", fake_wait_for)

    async def _exercise_stream():
        with client:
            route = next(route for route in client.app.routes if getattr(route, "path", "") == "/events")
            request = _FakeRequest(client.app)
            response = await route.endpoint(request, task_id="task-1")
            body = response.body_iterator

            connected = await anext(body)
            ping = await anext(body)
            with pytest.raises(StopAsyncIteration):
                await anext(body)
            await body.aclose()
            return connected, ping

    connected, ping = asyncio.run(_exercise_stream())

    assert connected == ": connected\n\n"
    assert ping == ": ping\n\n"


def test_create_app_builds_runtime_when_not_provided_and_falls_back_to_unknown_version(monkeypatch) -> None:
    import agent.main as main_module

    runtime = SimpleNamespace(sqlite=_FakeSQLite(), loop=_FakeLoop(), bot=None)
    built: list[bool] = []

    async def fake_build_runtime(*, start_discord: bool):
        built.append(start_discord)
        return runtime

    monkeypatch.setattr(main_module, "_build_runtime", fake_build_runtime)
    monkeypatch.setattr(app_module, "package_version", lambda name: (_ for _ in ()).throw(RuntimeError("missing package")))
    client = TestClient(create_app(None, start_background_runtime=False, shutdown_runtime=False))

    with client:
        response = client.get("/openapi.json")

    assert built == [False]
    assert response.status_code == 200
    assert response.json()["info"]["version"] == "unknown"


def test_task_response_from_row_uses_wait_state_question_and_nonterminal_success() -> None:
    response = _task_response_from_row(
        {
            "task_id": "task-1",
            "source": "api",
            "author": "tester",
            "content": "do thing",
            "status": "waiting_for_user",
            "result": None,
            "error": "Which env?",
            "success": 0,
            "elapsed_ms": None,
            "tool_calls": None,
            "created_ts": 1.0,
            "started_ts": 2.0,
            "finished_ts": None,
            "metadata": {"wait_state": {"question": "Which env?"}},
        }
    )

    assert response.question == "Which env?"
    assert response.success is None
    assert response.started_at is not None


def test_openapi_includes_control_plane_routes() -> None:
    client, _, _ = _build_app()
    with client:
        response = client.get("/openapi.json")

        assert response.status_code == 200
        paths = response.json()["paths"]
        assert "/healthz" in paths
        assert "/readyz" in paths
        assert "/tasks" in paths
        assert "/tasks/{task_id}" in paths
        assert "/tasks/{task_id}/input" in paths
        assert "/conversations/{session_id}" in paths
        assert "/conversations/{session_id}/turns" in paths
        assert "/events" in paths


def test_require_runtime_and_readiness_helpers_cover_degraded_states() -> None:
    app = SimpleNamespace(state=SimpleNamespace(ready=False, runtime=None))

    try:
        _require_runtime(app)  # type: ignore[arg-type]
    except Exception as exc:
        assert exc.error_code == "service_unavailable"

    ready_app = SimpleNamespace(state=SimpleNamespace(ready=True, runtime=SimpleNamespace(sqlite=None, postgres=None), loop_task=None))
    assert _is_runtime_ready(ready_app) is True  # type: ignore[arg-type]

    missing_runtime = SimpleNamespace(state=SimpleNamespace(ready=True, runtime=None, loop_task=None))
    assert _is_runtime_ready(missing_runtime) is False  # type: ignore[arg-type]

    done_task = SimpleNamespace(done=lambda: True)
    loop_down = SimpleNamespace(state=SimpleNamespace(ready=True, runtime=SimpleNamespace(sqlite=None, postgres=None), loop_task=done_task))
    assert _is_runtime_ready(loop_down) is False  # type: ignore[arg-type]

    sqlite_down = SimpleNamespace(
        state=SimpleNamespace(
            ready=True,
            runtime=SimpleNamespace(sqlite=SimpleNamespace(_db=None), postgres=None),
            loop_task=None,
        )
    )
    assert _is_runtime_ready(sqlite_down) is False  # type: ignore[arg-type]

    postgres_down = SimpleNamespace(
        state=SimpleNamespace(
            ready=True,
            runtime=SimpleNamespace(sqlite=None, postgres=SimpleNamespace(_pool=None)),
            loop_task=None,
        )
    )
    assert _is_runtime_ready(postgres_down) is False  # type: ignore[arg-type]


def test_build_app_delegates_to_create_app(monkeypatch) -> None:
    marker = object()
    monkeypatch.setattr(app_module, "create_app", lambda: marker)

    assert build_app() is marker


def test_create_app_background_lifespan_starts_and_stops_runtime(monkeypatch) -> None:
    import agent.main as main_module

    class _LoopRunner:
        def __init__(self) -> None:
            self.started = False
            self.cancelled = False

        async def run_forever(self) -> None:
            self.started = True
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    class _Bot:
        def __init__(self) -> None:
            self.started = False
            self.cancelled = False

        async def start_bot(self) -> None:
            self.started = True
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    runtime = SimpleNamespace(sqlite=_FakeSQLite(), loop=_LoopRunner(), bot=_Bot())
    shutdown_calls: list[object] = []

    async def fake_shutdown(passed_runtime) -> None:
        shutdown_calls.append(passed_runtime)

    monkeypatch.setattr(main_module, "_shutdown_runtime", fake_shutdown)
    client = TestClient(create_app(runtime, start_background_runtime=True, shutdown_runtime=True))

    with client:
        assert client.app.state.loop_task is not None
        assert client.app.state.bot_task is not None

    assert shutdown_calls == [runtime]
    assert runtime.loop.cancelled is True
    assert runtime.bot.cancelled is True


def test_create_app_background_runtime_with_no_bot(monkeypatch) -> None:
    import agent.main as main_module

    class _LoopRunner:
        def __init__(self) -> None:
            self.cancelled = False

        async def run_forever(self) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    runtime = SimpleNamespace(sqlite=_FakeSQLite(), loop=_LoopRunner(), bot=None)

    async def fake_shutdown(passed_runtime) -> None:
        return None

    monkeypatch.setattr(main_module, "_shutdown_runtime", fake_shutdown)
    client = TestClient(create_app(runtime, start_background_runtime=True, shutdown_runtime=True))

    with client:
        assert client.app.state.loop_task is not None
        assert client.app.state.bot_task is None

    assert runtime.loop.cancelled is True


def test_get_task_unexpected_error_returns_internal_error() -> None:
    class _BrokenSQLite:
        async def get_task_record(self, task_id: str):
            raise RuntimeError("db exploded")

    runtime = SimpleNamespace(sqlite=_BrokenSQLite(), loop=_FakeLoop(), bot=None)
    client = TestClient(
        create_app(runtime, start_background_runtime=False, shutdown_runtime=False),
        raise_server_exceptions=False,
    )

    with client:
        response = client.get("/tasks/task-1")

    assert response.status_code == 500
    assert response.json()["error_code"] == "internal_error"
