from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from agent.task_waits import TaskWaitRegistry


class TurnIntent(str, Enum):
    ANSWER_PENDING_QUESTION = "answer_pending_question"
    CLARIFICATION_OR_NEW_CONSTRAINT = "clarification_or_new_constraint"
    CANCEL_OR_PAUSE = "cancel_or_pause"
    CONTINUE_SAME_TASK = "continue_same_task"
    START_NEW_TASK = "start_new_task"


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    source: str
    channel_id: int = 0
    thread_key: str = ""


@dataclass(frozen=True)
class TurnDecision:
    session: SessionInfo
    intent: TurnIntent


class SessionRouter:
    """Derive durable session IDs and lightweight turn intent decisions."""

    def build_session(
        self,
        *,
        source: str,
        channel_id: int = 0,
        message_id: int = 0,
        reference_message_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionInfo:
        metadata = metadata or {}
        existing = str(metadata.get("session_id", "")).strip()
        if existing:
            return SessionInfo(
                session_id=existing,
                source=source,
                channel_id=channel_id,
                thread_key=str(metadata.get("thread_key", "")).strip(),
            )

        if source == "discord":
            anchor = reference_message_id or message_id or channel_id
            thread_key = f"{channel_id}:{anchor}"
            return SessionInfo(
                session_id=f"discord:{thread_key}",
                source=source,
                channel_id=channel_id,
                thread_key=thread_key,
            )

        task_id = str(metadata.get("task_id", "")).strip()
        if task_id:
            return SessionInfo(
                session_id=f"{source}:{task_id}",
                source=source,
                channel_id=channel_id,
                thread_key=task_id,
            )

        fallback = reference_message_id or message_id or 0
        return SessionInfo(
            session_id=f"{source}:{channel_id}:{fallback}",
            source=source,
            channel_id=channel_id,
            thread_key=str(fallback),
        )

    def build_metadata(
        self,
        *,
        source: str,
        channel_id: int = 0,
        message_id: int = 0,
        reference_message_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged = dict(metadata or {})
        session = self.build_session(
            source=source,
            channel_id=channel_id,
            message_id=message_id,
            reference_message_id=reference_message_id,
            metadata=merged,
        )
        merged["session_id"] = session.session_id
        merged["thread_key"] = session.thread_key
        merged["source"] = source
        return merged

    def classify_turn(
        self,
        *,
        source: str,
        channel_id: int = 0,
        message_id: int = 0,
        reference_message_id: int | None = None,
        content: str,
        metadata: dict[str, Any] | None = None,
        has_active_task: bool = False,
        wait_registry: TaskWaitRegistry | None = None,
    ) -> TurnDecision:
        session = self.build_session(
            source=source,
            channel_id=channel_id,
            message_id=message_id,
            reference_message_id=reference_message_id,
            metadata=metadata,
        )
        text = content.strip().lower()
        pending_count = len(wait_registry.pending_for_channel(channel_id)) if wait_registry else 0

        if reference_message_id is not None and wait_registry is not None:
            suspended = wait_registry.find_for_discord_reply(
                channel_id=channel_id,
                reference_message_id=reference_message_id,
            )
            if suspended is not None:
                return TurnDecision(session=session, intent=TurnIntent.ANSWER_PENDING_QUESTION)

        if pending_count == 1 and reference_message_id is None and text:
            if len(text.split()) <= 40:
                return TurnDecision(session=session, intent=TurnIntent.ANSWER_PENDING_QUESTION)

        if any(text.startswith(prefix) for prefix in ("cancel", "stop", "pause", "hold off", "never mind")):
            return TurnDecision(session=session, intent=TurnIntent.CANCEL_OR_PAUSE)

        if has_active_task:
            if any(
                text.startswith(prefix)
                for prefix in ("actually", "instead", "update:", "change:", "one more thing", "also", "constraint:")
            ):
                return TurnDecision(session=session, intent=TurnIntent.CLARIFICATION_OR_NEW_CONSTRAINT)
            if reference_message_id is not None or len(text.split()) <= 20:
                return TurnDecision(session=session, intent=TurnIntent.CONTINUE_SAME_TASK)

        return TurnDecision(session=session, intent=TurnIntent.START_NEW_TASK)
