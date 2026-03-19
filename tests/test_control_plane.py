from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent.control_plane.app import SseBroker, create_app
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

    def build_resumed_task(self, *, suspended: SuspendedTask, answer: str, author: str, source: str) -> Task:
        metadata = self.wait_registry.build_resumed_metadata(suspended, answer=answer, resumed_from=source)
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
