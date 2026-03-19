from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from typing import Any, AsyncIterator, Literal
from uuid import uuid4

from fastapi import FastAPI, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from agent.config import settings
from agent.events import AgentEvent, TaskQueuedEvent, bridge
from agent.loop import Task
from agent.session_router import SessionRouter


TaskState = Literal["queued", "running", "waiting_for_user", "succeeded", "failed"]
ErrorCode = Literal[
    "invalid_request",
    "task_not_found",
    "service_unavailable",
    "internal_error",
]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "ready"]


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_code: ErrorCode
    message: str
    details: dict[str, Any] | None = None


class CreateTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=20000)
    author: str = Field(default="api", min_length=1, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateTaskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    status: TaskState


class ResumeTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=20000)
    author: str = Field(default="api", min_length=1, max_length=200)


class TaskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    author: str
    content: str
    status: TaskState
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: str | None = None
    error: str | None = None
    success: bool | None = None
    elapsed_ms: float | None = None
    tool_calls: int | None = None
    question: str | None = None


class ConversationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    channel_id: int
    status: str
    title: str
    summary: str
    pending_task_id: str


class ApiError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        error_code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.details = details


class SseBroker:
    def __init__(self) -> None:
        self._subscribers: dict[str, tuple[asyncio.Queue[str], str | None]] = {}

    async def publish(self, event: AgentEvent) -> None:
        event_name, payload = self._serialize_event(event)
        task_id = payload.get("task_id")
        frame = self._encode_frame(event_name, payload)

        for queue, filter_task_id in list(self._subscribers.values()):
            if filter_task_id is not None and task_id != filter_task_id:
                continue
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with suppress(asyncio.QueueFull):
                queue.put_nowait(frame)

    async def sink(self, event: AgentEvent) -> None:
        await self.publish(event)

    def subscribe(self, task_id: str | None) -> tuple[str, asyncio.Queue[str]]:
        subscriber_id = str(uuid4())
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
        self._subscribers[subscriber_id] = (queue, task_id)
        return subscriber_id, queue

    def unsubscribe(self, subscriber_id: str) -> None:
        self._subscribers.pop(subscriber_id, None)

    def _serialize_event(self, event: AgentEvent) -> tuple[str, dict[str, Any]]:
        if is_dataclass(event):
            payload = jsonable_encoder(asdict(event))
        else:
            payload = jsonable_encoder(event)
        event_name = payload.get("kind", "message")
        return event_name, payload

    def _encode_frame(self, event_name: str, payload: dict[str, Any]) -> str:
        return f"event: {event_name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


ERROR_RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
    status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
}


def create_app(
    runtime_override: Any | None = None,
    *,
    start_background_runtime: bool = True,
    shutdown_runtime: bool = True,
) -> FastAPI:
    session_router = SessionRouter()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from agent.main import _build_runtime, _shutdown_runtime

        runtime = runtime_override
        loop_task: asyncio.Task[None] | None = None
        bot_task: asyncio.Task[None] | None = None

        if runtime is None:
            runtime = await _build_runtime(start_discord=settings.has_discord)

        broker = SseBroker()
        app.state.runtime = runtime
        app.state.loop_task = None
        app.state.bot_task = None
        app.state.sse_broker = broker
        app.state.ready = True
        bridge.register("control_plane_sse", broker.sink)

        if start_background_runtime:
            loop_task = asyncio.create_task(runtime.loop.run_forever())
            app.state.loop_task = loop_task
            if runtime.bot is not None:
                bot_task = asyncio.create_task(runtime.bot.start_bot())
                app.state.bot_task = bot_task

        try:
            yield
        finally:
            app.state.ready = False
            bridge.unregister("control_plane_sse")
            if shutdown_runtime:
                await _shutdown_runtime(runtime)
            for task in (bot_task, loop_task):
                if task is not None:
                    task.cancel()
            for task in (bot_task, loop_task):
                if task is not None:
                    with suppress(asyncio.CancelledError):
                        await task

    try:
        app_version = package_version("agent")
    except Exception:
        app_version = "unknown"

    app = FastAPI(
        title="Agent Control Plane",
        version=app_version,
        description="Minimal FastAPI control plane for submitting tasks and streaming task events.",
        lifespan=lifespan,
    )

    @app.exception_handler(ApiError)
    async def handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        payload = ErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump(mode="json"))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        payload = ErrorResponse(
            error_code="invalid_request",
            message="Request validation failed.",
            details={"errors": exc.errors()},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=payload.model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, _exc: Exception) -> JSONResponse:
        payload = ErrorResponse(
            error_code="internal_error",
            message="An unexpected internal error occurred.",
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=payload.model_dump(mode="json"),
        )

    @app.get("/healthz", response_model=HealthResponse, tags=["system"])
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get(
        "/readyz",
        response_model=HealthResponse,
        responses=ERROR_RESPONSES,
        tags=["system"],
    )
    async def readyz(request: Request) -> HealthResponse:
        if not _is_runtime_ready(request.app):
            raise ApiError(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                error_code="service_unavailable",
                message="The control plane runtime is not ready.",
            )
        return HealthResponse(status="ready")

    @app.post(
        "/tasks",
        response_model=CreateTaskResponse,
        status_code=status.HTTP_202_ACCEPTED,
        responses=ERROR_RESPONSES,
        tags=["tasks"],
    )
    async def create_task(request: Request, payload: CreateTaskRequest) -> CreateTaskResponse:
        runtime = _require_runtime(request.app)
        content = payload.content.strip()
        if not content:
            raise ApiError(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code="invalid_request",
                message="Task content must not be blank.",
            )

        task_id = str(uuid4())
        metadata = dict(payload.metadata)
        metadata["task_id"] = task_id
        metadata = session_router.build_metadata(
            source="api",
            message_id=0,
            reference_message_id=None,
            metadata=metadata,
        )

        await runtime.sqlite.create_task_record(
            task_id=task_id,
            source="api",
            author=payload.author,
            content=content,
            metadata=metadata,
        )
        if hasattr(runtime.sqlite, "ensure_session"):
            await runtime.sqlite.ensure_session(
                session_id=metadata["session_id"],
                source="api",
                title=content[:120],
                status="queued",
                pending_task_id=task_id,
                metadata={"author": payload.author},
            )
            await runtime.sqlite.append_session_turn(
                session_id=metadata["session_id"],
                role="user",
                content=content,
                task_id=task_id,
                metadata={"author": payload.author},
            )

        task = Task(
            content=content,
            source="api",
            author=payload.author,
            metadata=metadata,
        )
        await runtime.loop.enqueue(task)
        await bridge.emit(TaskQueuedEvent(task_id=task_id, content=content, source="api"))
        return CreateTaskResponse(id=task_id, status="queued")

    @app.get(
        "/tasks/{task_id}",
        response_model=TaskResponse,
        responses=ERROR_RESPONSES,
        tags=["tasks"],
    )
    async def get_task(request: Request, task_id: str) -> TaskResponse:
        runtime = _require_runtime(request.app)
        row = await runtime.sqlite.get_task_record(task_id)
        if row is None:
            raise ApiError(
                status_code=status.HTTP_404_NOT_FOUND,
                error_code="task_not_found",
                message=f"Task '{task_id}' was not found.",
            )
        return _task_response_from_row(row)

    @app.post(
        "/tasks/{task_id}/input",
        response_model=CreateTaskResponse,
        status_code=status.HTTP_202_ACCEPTED,
        responses=ERROR_RESPONSES,
        tags=["tasks"],
    )
    async def resume_task(request: Request, task_id: str, payload: ResumeTaskRequest) -> CreateTaskResponse:
        runtime = _require_runtime(request.app)
        row = await runtime.sqlite.get_task_record(task_id)
        if row is None:
            raise ApiError(
                status_code=status.HTTP_404_NOT_FOUND,
                error_code="task_not_found",
                message=f"Task '{task_id}' was not found.",
            )
        if row["status"] != "waiting_for_user":
            raise ApiError(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code="invalid_request",
                message=f"Task '{task_id}' is not waiting for user input.",
            )

        suspended = runtime.loop.wait_registry.get(task_id)
        if suspended is None:
            raise ApiError(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                error_code="service_unavailable",
                message=f"Task '{task_id}' is waiting, but no resumable state is currently loaded.",
            )

        answer = payload.content.strip()
        if not answer:
            raise ApiError(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code="invalid_request",
                message="Resume content must not be blank.",
            )

        resumed = runtime.loop.build_resumed_task(
            suspended=runtime.loop.wait_registry.pop(task_id) or suspended,
            answer=answer,
            author=payload.author,
            source="api",
        )
        await runtime.sqlite.mark_task_queued(task_id, metadata=resumed.metadata)
        if hasattr(runtime.sqlite, "append_session_turn"):
            await runtime.sqlite.append_session_turn(
                session_id=str((resumed.metadata or {}).get("session_id", "")),
                role="user",
                content=answer,
                turn_kind="answer",
                task_id=task_id,
                metadata={"author": payload.author},
            )
        await runtime.loop.enqueue(resumed)
        await bridge.emit(TaskQueuedEvent(task_id=task_id, content=resumed.content, source="api"))
        return CreateTaskResponse(id=task_id, status="queued")

    @app.get(
        "/conversations/{session_id}",
        response_model=ConversationResponse,
        responses=ERROR_RESPONSES,
        tags=["conversations"],
    )
    async def get_conversation(request: Request, session_id: str) -> ConversationResponse:
        runtime = _require_runtime(request.app)
        if not hasattr(runtime.sqlite, "get_session"):
            raise ApiError(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                error_code="service_unavailable",
                message="Conversation storage is not available.",
            )
        row = await runtime.sqlite.get_session(session_id)
        if row is None:
            raise ApiError(
                status_code=status.HTTP_404_NOT_FOUND,
                error_code="task_not_found",
                message=f"Conversation '{session_id}' was not found.",
            )
        return ConversationResponse(
            id=row["session_id"],
            source=row["source"],
            channel_id=int(row.get("channel_id") or 0),
            status=row["status"],
            title=row.get("title", ""),
            summary=row.get("summary", ""),
            pending_task_id=row.get("pending_task_id", ""),
        )

    @app.get(
        "/conversations/{session_id}/turns",
        responses=ERROR_RESPONSES,
        tags=["conversations"],
    )
    async def get_conversation_turns(
        request: Request,
        session_id: str,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> list[dict[str, Any]]:
        runtime = _require_runtime(request.app)
        if not hasattr(runtime.sqlite, "list_session_turns"):
            raise ApiError(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                error_code="service_unavailable",
                message="Conversation storage is not available.",
            )
        return await runtime.sqlite.list_session_turns(session_id, limit=limit)

    @app.get(
        "/events",
        responses={
            status.HTTP_200_OK: {"content": {"text/event-stream": {}}},
            **ERROR_RESPONSES,
        },
        tags=["tasks"],
        summary="Stream task events over SSE",
    )
    async def events(
        request: Request,
        task_id: str | None = Query(default=None, description="Optional task ID filter."),
    ) -> StreamingResponse:
        _require_runtime(request.app)
        broker: SseBroker = request.app.state.sse_broker
        subscriber_id, queue = broker.subscribe(task_id)

        async def stream() -> AsyncIterator[str]:
            try:
                yield ": connected\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        message = await asyncio.wait_for(
                            queue.get(),
                            timeout=settings.control_plane_sse_ping_seconds,
                        )
                        yield message
                    except TimeoutError:
                        yield ": ping\n\n"
            finally:
                broker.unsubscribe(subscriber_id)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def build_app() -> FastAPI:
    return create_app()


def _require_runtime(app: FastAPI) -> Any:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None or not getattr(app.state, "ready", False):
        raise ApiError(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="service_unavailable",
            message="The control plane runtime is not ready.",
        )
    return runtime


def _is_runtime_ready(app: FastAPI) -> bool:
    if not getattr(app.state, "ready", False):
        return False
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return False
    loop_task = getattr(app.state, "loop_task", None)
    if loop_task is not None and loop_task.done():
        return False
    sqlite = getattr(runtime, "sqlite", None)
    if sqlite is not None and hasattr(sqlite, "_db") and getattr(sqlite, "_db", None) is None:
        return False
    postgres = getattr(runtime, "postgres", None)
    if postgres is not None and hasattr(postgres, "_pool") and getattr(postgres, "_pool", None) is None:
        return False
    return True


def _task_response_from_row(row: dict[str, Any]) -> TaskResponse:
    return TaskResponse(
        id=row["task_id"],
        source=row["source"],
        author=row["author"],
        content=row["content"],
        status=row["status"],
        created_at=_ts_to_datetime(row["created_ts"]),
        started_at=_ts_to_datetime(row.get("started_ts")),
        finished_at=_ts_to_datetime(row.get("finished_ts")),
        result=row.get("result"),
        error=row.get("error"),
        success=(
            None
            if row["status"] in {"queued", "running", "waiting_for_user"}
            else bool(row["success"])
        ),
        elapsed_ms=row.get("elapsed_ms"),
        tool_calls=row.get("tool_calls"),
        question=((row.get("metadata") or {}).get("wait_state") or {}).get("question"),
    )


def _ts_to_datetime(value: float | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=UTC)
