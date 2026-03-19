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


def test_discord_attachment_decode_helpers() -> None:
    attachment = discord_tools_module.decode_data_url_attachment(
        "data:image/png;base64,cG5n",
        filename="shot.png",
    )
    invalid = discord_tools_module.decode_data_url_attachment("not-a-data-url")
    corrupt = discord_tools_module.decode_data_url_attachment("data:image/png;base64,@@@")

    assert attachment is not None
    assert attachment.filename == "shot.png"
    assert attachment.data == b"png"
    assert invalid is None
    assert corrupt is None


def test_pending_question_helpers_detect_replies() -> None:
    discord_tools_module._pending_question_ids.clear()
    discord_tools_module._pending_question_ids[9] = {42}
    message = SimpleNamespace(channel=SimpleNamespace(id=9), reference=_ReplyRef(message_id=42))

    assert discord_tools_module.has_pending_question(9) is True
    assert discord_tools_module.is_pending_question_reply(message) is True
    assert discord_tools_module.is_pending_question_reply(
        SimpleNamespace(channel=SimpleNamespace(id=9), reference=None)
    ) is False
    assert discord_tools_module.is_pending_question_reply(
        SimpleNamespace(channel=SimpleNamespace(id=99), reference=_ReplyRef(message_id=42))
    ) is False


@pytest.mark.asyncio
async def test_discord_send_splits_long_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    channel = _Channel(1)
    monkeypatch.setattr(discord_tools_module, "_discord_client", _Client({1: channel}))

    result = await discord_send(1, "x" * 2500)

    assert "Sent 2 message(s)" in result
    assert len(channel.sent) == 2


@pytest.mark.asyncio
async def test_discord_send_handles_missing_client_channel_and_send_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discord_tools_module, "_discord_client", None)
    assert await discord_send(1, "hello") == "[ERROR: Discord client not initialised]"

    monkeypatch.setattr(discord_tools_module, "_discord_client", _Client({}))
    assert "not found or not accessible" in await discord_send(1, "hello")

    class _BrokenChannel(_Channel):
        async def send(self, content: str = "", *, file=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(discord_tools_module, "_discord_client", _Client({1: _BrokenChannel(1)}))
    assert await discord_send(1, "hello") == "[ERROR: boom]"


@pytest.mark.asyncio
async def test_send_text_returns_zero_for_empty_message() -> None:
    channel = _Channel(1)

    sent = await discord_tools_module.send_text(channel, "")

    assert sent == 0
    assert channel.sent == []


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
async def test_discord_read_handles_missing_client_channel_empty_history_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discord_tools_module, "_discord_client", None)
    assert await discord_read(7) == "[ERROR: Discord client not initialised]"

    monkeypatch.setattr(discord_tools_module, "_discord_client", _Client({}))
    assert await discord_read(7) == "[ERROR: channel 7 not found]"

    empty = _Channel(7)
    monkeypatch.setattr(discord_tools_module, "_discord_client", _Client({7: empty}))
    assert await discord_read(7) == "(no messages)"

    class _BrokenChannel(_Channel):
        async def history(self, limit: int = 20, after=None, oldest_first: bool = False):
            raise RuntimeError("history failed")
            yield None

    monkeypatch.setattr(discord_tools_module, "_discord_client", _Client({7: _BrokenChannel(7)}))
    assert await discord_read(7) == "[ERROR: history failed]"


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
async def test_discord_read_named_rejects_unknown_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "discord_agent_channel_id", 7)
    assert "unknown channel name" in await discord_read_named("unknown")


@pytest.mark.asyncio
async def test_ask_user_returns_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    channel = _Channel(9)
    client = _Client({9: channel})
    discord_tools_module._pending_question_ids.clear()
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


@pytest.mark.asyncio
async def test_ask_user_handles_wait_context_and_common_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    discord_tools_module._pending_question_ids.clear()
    monkeypatch.setattr(discord_tools_module, "current_task_wait_context", lambda: object())
    with pytest.raises(discord_tools_module.UserInputRequired):
        await ask_user("Proceed?", timeout=3)

    monkeypatch.setattr(discord_tools_module, "current_task_wait_context", lambda: None)
    monkeypatch.setattr(discord_tools_module, "_discord_client", None)
    assert await ask_user("Proceed?", timeout=3) == "[ERROR: Discord client not initialised]"

    monkeypatch.setattr(discord_tools_module, "_discord_client", _Client({}))
    monkeypatch.setattr(settings, "discord_agent_channel_id", 0)
    assert await ask_user("Proceed?", timeout=3) == "[ERROR: DISCORD_AGENT_CHANNEL_ID not configured]"

    monkeypatch.setattr(settings, "discord_agent_channel_id", 9)
    assert await ask_user("Proceed?", timeout=3) == "[ERROR: private channel 9 not found]"

    class _BrokenSendChannel(_Channel):
        async def send(self, content: str = "", *, file=None):
            raise RuntimeError("send failed")

    monkeypatch.setattr(discord_tools_module, "_discord_client", _Client({9: _BrokenSendChannel(9)}))
    assert await ask_user("Proceed?", timeout=3) == "[ERROR sending question: send failed]"


@pytest.mark.asyncio
async def test_ask_user_handles_history_errors_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    channel = _Channel(9)
    client = _Client({9: channel})
    discord_tools_module._pending_question_ids.clear()
    monkeypatch.setattr(discord_tools_module, "_discord_client", client)
    monkeypatch.setattr(settings, "discord_agent_channel_id", 9)
    monkeypatch.setattr(discord_tools_module, "current_task_wait_context", lambda: None)

    class _BrokenHistoryChannel(_Channel):
        async def history(self, limit: int = 20, after=None, oldest_first: bool = False):
            raise RuntimeError("history failed")
            yield None

    broken = _BrokenHistoryChannel(9)
    monkeypatch.setattr(discord_tools_module, "_discord_client", _Client({9: broken}))
    assert await ask_user("Proceed?", timeout=3) == "[ERROR reading reply: history failed]"

    monkeypatch.setattr(discord_tools_module, "_discord_client", client)

    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    result = await ask_user("Proceed?", timeout=2)
    assert "No reply received after 2s" in result


@pytest.mark.asyncio
async def test_ask_user_ignores_bot_and_blank_messages_then_uses_single_author_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    channel = _Channel(9)
    client = _Client({9: channel})
    discord_tools_module._pending_question_ids.clear()
    monkeypatch.setattr(discord_tools_module, "_discord_client", client)
    monkeypatch.setattr(settings, "discord_agent_channel_id", 9)
    monkeypatch.setattr(discord_tools_module, "current_task_wait_context", lambda: None)

    async def fake_sleep(seconds: float) -> None:
        if not channel._history_items:
            channel._history_items.extend(
                [
                    _HistoryMessage(
                        author=SimpleNamespace(id=10, display_name="Bot", bot=True),
                        content="ignore me",
                        created_at=datetime.now(UTC),
                    ),
                    _HistoryMessage(
                        author=SimpleNamespace(id=11, display_name="Josh", bot=False),
                        content="   ",
                        created_at=datetime.now(UTC),
                    ),
                    _HistoryMessage(
                        author=SimpleNamespace(id=11, display_name="Josh", bot=False),
                        content="use staging",
                        created_at=datetime.now(UTC),
                    ),
                ]
            )

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = await ask_user("Proceed?", timeout=3)

    assert result == "use staging"
