from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterator
from uuid import uuid4

import structlog

log = structlog.get_logger()


@dataclass
class UserInputRequired(Exception):
    question: str
    timeout_s: int = 300

    def __str__(self) -> str:
        return self.question


@dataclass(frozen=True)
class TaskWaitContext:
    task_id: str
    source: str
    channel_id: int


@dataclass
class SuspendedTask:
    task_id: str
    source: str
    author: str
    content: str
    channel_id: int
    message_id: int
    metadata: dict[str, Any]
    question: str
    timeout_s: int
    base_prompt: str
    tier: str
    prompt_message_id: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


_task_wait_context_var: contextvars.ContextVar[TaskWaitContext | None] = contextvars.ContextVar(
    "task_wait_context",
    default=None,
)


@contextlib.contextmanager
def task_wait_context(*, task_id: str, source: str, channel_id: int) -> Iterator[None]:
    token = _task_wait_context_var.set(TaskWaitContext(task_id=task_id, source=source, channel_id=channel_id))
    try:
        yield
    finally:
        _task_wait_context_var.reset(token)


def current_task_wait_context() -> TaskWaitContext | None:
    return _task_wait_context_var.get()


class TaskWaitRegistry:
    def __init__(self) -> None:
        self._by_task_id: dict[str, SuspendedTask] = {}

    @staticmethod
    def ensure_task_id(metadata: dict[str, Any] | None) -> str:
        metadata = metadata if metadata is not None else {}
        task_id = str(metadata.get("task_id", "")).strip()
        if not task_id:
            task_id = str(uuid4())
            metadata["task_id"] = task_id
        return task_id

    def suspend(
        self,
        *,
        task_id: str,
        source: str,
        author: str,
        content: str,
        channel_id: int,
        message_id: int,
        metadata: dict[str, Any],
        question: str,
        timeout_s: int,
        base_prompt: str,
        tier: str,
    ) -> SuspendedTask:
        suspended = SuspendedTask(
            task_id=task_id,
            source=source,
            author=author,
            content=content,
            channel_id=channel_id,
            message_id=message_id,
            metadata=dict(metadata),
            question=question,
            timeout_s=timeout_s,
            base_prompt=base_prompt,
            tier=tier,
        )
        self._by_task_id[task_id] = suspended
        return suspended

    def bind_prompt_message(self, task_id: str, prompt_message_id: int) -> None:
        suspended = self._by_task_id.get(task_id)
        if suspended is None:
            return
        suspended.prompt_message_id = prompt_message_id

    def get(self, task_id: str) -> SuspendedTask | None:
        return self._by_task_id.get(task_id)

    def pop(self, task_id: str) -> SuspendedTask | None:
        return self._by_task_id.pop(task_id, None)

    def pending_for_channel(self, channel_id: int) -> list[SuspendedTask]:
        return [item for item in self._by_task_id.values() if item.channel_id == channel_id and item.source == "discord"]

    def pop_for_discord_reply(self, *, channel_id: int, reference_message_id: int | None) -> SuspendedTask | None:
        suspended = self.find_for_discord_reply(
            channel_id=channel_id,
            reference_message_id=reference_message_id,
        )
        if suspended is None:
            return None
        return self._by_task_id.pop(suspended.task_id, None)

    def find_for_discord_reply(self, *, channel_id: int, reference_message_id: int | None) -> SuspendedTask | None:
        if reference_message_id is not None:
            for suspended in self._by_task_id.values():
                if suspended.channel_id == channel_id and suspended.prompt_message_id == reference_message_id:
                    return suspended
            return None

        candidates = self.pending_for_channel(channel_id)
        if len(candidates) == 1:
            return candidates[0]
        return None

    def list_expired(self, *, now: datetime | None = None) -> list[SuspendedTask]:
        now = now or datetime.now(UTC)
        expired: list[SuspendedTask] = []
        for item in self._by_task_id.values():
            age = (now - item.created_at).total_seconds()
            if age >= max(1, item.timeout_s):
                expired.append(item)
        return expired

    def build_resumed_metadata(self, suspended: SuspendedTask, *, answer: str, resumed_from: str) -> dict[str, Any]:
        metadata = dict(suspended.metadata)
        resume_context = dict(metadata.get("resume_context", {}))
        resume_context.update(
            {
                "question": suspended.question,
                "answer": answer,
                "resumed_from": resumed_from,
                "suspended_task_id": suspended.task_id,
            }
        )
        metadata["resume_context"] = resume_context
        metadata["task_id"] = suspended.task_id
        metadata.pop("wait_state", None)
        return metadata

    def has_pending(self, task_id: str) -> bool:
        return task_id in self._by_task_id
