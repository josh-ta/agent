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

    _CANCEL_PREFIXES = (
        "cancel",
        "stop",
        "pause",
        "hold off",
        "never mind",
        "nevermind",
        "forget it",
        "forget that",
        "drop it",
        "scratch that",
        "/cancel",
        "/forget",
    )
    _CONSTRAINT_PREFIXES = (
        "actually",
        "update:",
        "change:",
        "one more thing",
        "also",
        "constraint:",
        "make sure",
        "don't",
        "do not",
        "without ",
        "with ",
    )
    _NEW_TASK_PREFIXES = (
        "please ",
        "now ",
        "go ahead and ",
        "start ",
        "restart ",
        "deploy ",
        "fix ",
        "check ",
        "investigate ",
        "run ",
        "pull ",
        "ship ",
        "build ",
        "update ",
        "review ",
        "summarize ",
        "search ",
        "find ",
        "look into ",
        "ssh ",
    )
    _SAME_TASK_PHRASES = {
        "ok",
        "okay",
        "sounds good",
        "keep going",
        "continue",
        "continue please",
        "yep",
        "yes",
        "no",
        "thanks",
        "thank you",
    }

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

        if any(text.startswith(prefix) for prefix in self._CANCEL_PREFIXES):
            return TurnDecision(session=session, intent=TurnIntent.CANCEL_OR_PAUSE)

        if has_active_task:
            if any(text.startswith(prefix) for prefix in self._CONSTRAINT_PREFIXES):
                return TurnDecision(session=session, intent=TurnIntent.CLARIFICATION_OR_NEW_CONSTRAINT)
            if self._looks_like_new_task(text):
                return TurnDecision(session=session, intent=TurnIntent.START_NEW_TASK)
            if reference_message_id is not None or self._looks_like_same_task_followup(text):
                return TurnDecision(session=session, intent=TurnIntent.CONTINUE_SAME_TASK)

        return TurnDecision(session=session, intent=TurnIntent.START_NEW_TASK)

    def _looks_like_new_task(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if any(stripped.startswith(prefix) for prefix in self._NEW_TASK_PREFIXES):
            return True
        words = stripped.split()
        if not words:
            return False
        if words[0] in {
            "restart",
            "deploy",
            "fix",
            "check",
            "investigate",
            "run",
            "pull",
            "ship",
            "build",
            "update",
            "review",
            "summarize",
            "search",
            "find",
            "ssh",
        }:
            return True
        return len(words) > 20

    def _looks_like_same_task_followup(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if stripped in self._SAME_TASK_PHRASES:
            return True
        return len(stripped.split()) <= 8 and not self._looks_like_new_task(stripped)
