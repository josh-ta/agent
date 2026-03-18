from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import agent.tools.discord_tools as discord_tools_module
from agent.config import settings
from agent.tools.discord_tools import DiscordAttachment, ask_user, discord_read, discord_read_named, discord_send, send_attachments


@dataclass
class _ReplyRef:
    message_id: int


@dataclass
class _HistoryMessage:
    author: object
    content: str
    created_at: datetime
    reference: _ReplyRef | None = None


class _Channel:
    def __init__(self, channel_id: int) -> None:
        self.id = channel_id
        self.sent: list[str] = []
        self.sent_files: list[str] = []
        self._history_items: list[_HistoryMessage] = []

    async def send(self, content: str = "", *, file=None):
        self.sent.append(content)
        if file is not None:
            self.sent_files.append(getattr(file, "filename", "attachment"))
        return SimpleNamespace(id=len(self.sent))

    async def history(self, limit: int = 20, after=None, oldest_first: bool = False):
        items = self._history_items[:limit]
        if oldest_first:
            ordered = items
        else:
            ordered = list(reversed(items))
        for item in ordered:
            yield item


class _Client:
    def __init__(self, channels: dict[int, _Channel]) -> None:
        self._channels = channels

    def get_channel(self, channel_id: int):
        return self._channels.get(channel_id)


@pytest.mark.asyncio
async def test_discord_send_splits_long_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    channel = _Channel(1)
    monkeypatch.setattr(discord_tools_module, "_discord_client", _Client({1: channel}))

    result = await discord_send(1, "x" * 2500)

    assert "Sent 2 message(s)" in result
    assert len(channel.sent) == 2


@pytest.mark.asyncio
async def test_send_attachments_uploads_png_files() -> None:
    channel = _Channel(1)

    sent = await send_attachments(
        channel,
        [DiscordAttachment(filename="browser-screenshot-1.png", data=b"png-bytes")],
        message="here you go",
    )

    assert sent == 2
    assert channel.sent == ["here you go", ""]
    assert channel.sent_files == ["browser-screenshot-1.png"]


@pytest.mark.asyncio
async def test_discord_read_and_named_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    channel = _Channel(7)
    channel._history_items = [
        _HistoryMessage(
            author=SimpleNamespace(display_name="Josh", bot=False),
            content="hello",
            created_at=datetime(2026, 3, 16, 12, 0, tzinfo=UTC),
        )
    ]
    monkeypatch.setattr(discord_tools_module, "_discord_client", _Client({7: channel}))
    monkeypatch.setattr(settings, "discord_agent_channel_id", 7)

    direct = await discord_read(7)
    named = await discord_read_named("private")

    assert "Josh: hello" in direct
    assert named == direct


@pytest.mark.asyncio
async def test_ask_user_returns_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    channel = _Channel(9)
    client = _Client({9: channel})
    monkeypatch.setattr(discord_tools_module, "_discord_client", client)
    monkeypatch.setattr(settings, "discord_agent_channel_id", 9)

    async def fake_sleep(seconds: float) -> None:
        if not channel._history_items:
            channel._history_items.extend(
                [
                    _HistoryMessage(
                        author=SimpleNamespace(id=10, display_name="Josh", bot=False),
                        content="placeholder",
                        created_at=datetime.now(UTC),
                    ),
                    _HistoryMessage(
                        author=SimpleNamespace(id=10, display_name="Josh", bot=False),
                        content="yes, proceed",
                        created_at=datetime.now(UTC),
                        reference=_ReplyRef(message_id=1),
                    ),
                ]
            )

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = await ask_user("Proceed?", timeout=3)

    assert result == "yes, proceed"
    assert discord_tools_module.has_pending_question(9) is False
