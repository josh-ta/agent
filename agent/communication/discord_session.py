"""Per-channel session state for Discord task routing."""

from __future__ import annotations

import asyncio
from typing import Any


class DiscordSessionState:
    """Tracks inject queues, sticky sessions, and cancel state per channel."""

    def __init__(self) -> None:
        self._inject_queues: dict[int, asyncio.Queue[str]] = {}
        self._active_sessions: dict[int, str] = {}
        self._sticky_sessions: dict[int, str] = {}
        self._cancelling_channels: set[int] = set()

    def get_inject_queue(self, channel_id: int) -> asyncio.Queue[str] | None:
        return self._inject_queues.get(channel_id)

    def set_inject_queue(self, channel_id: int, queue: asyncio.Queue[str]) -> None:
        self._inject_queues[channel_id] = queue

    def pop_inject_queue(self, channel_id: int) -> asyncio.Queue[str] | None:
        return self._inject_queues.pop(channel_id, None)

    def has_active_task(self, channel_id: int) -> bool:
        return channel_id in self._inject_queues

    def get_active_session(self, channel_id: int) -> str:
        return self._active_sessions.get(channel_id, "")

    def set_active_session(self, channel_id: int, session_id: str) -> None:
        self._active_sessions[channel_id] = session_id

    def pop_active_session(self, channel_id: int) -> str | None:
        return self._active_sessions.pop(channel_id, None)

    def get_sticky_session(self, channel_id: int) -> str:
        return self._sticky_sessions.get(channel_id, "")

    def set_sticky_session(self, channel_id: int, session_id: str) -> None:
        self._sticky_sessions[channel_id] = session_id

    def pop_sticky_session(self, channel_id: int) -> str | None:
        return self._sticky_sessions.pop(channel_id, None)

    def mark_cancelling(self, channel_id: int) -> None:
        self._cancelling_channels.add(channel_id)

    def clear_cancelling(self, channel_id: int) -> None:
        self._cancelling_channels.discard(channel_id)

    def is_cancelling(self, channel_id: int) -> bool:
        return channel_id in self._cancelling_channels

    def describe(self) -> dict[str, Any]:
        return {
            "active_channels": list(self._inject_queues.keys()),
            "sticky_sessions": dict(self._sticky_sessions),
        }
