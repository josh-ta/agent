from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent.control_plane.app import SseBroker, create_app
from agent.events import ProgressEvent, TaskQueuedEvent, bridge
from agent.loop import Task


class _FakeSQLite:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.rows: dict[str, dict[str, object]] = {}

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


class _FakeLoop:
    def __init__(self) -> None:
        self.enqueued: list[Task] = []

    async def enqueue(self, task: Task) -> None:
        self.enqueued.append(task)


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
        assert "/events" in paths
