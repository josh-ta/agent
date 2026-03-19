from __future__ import annotations

import asyncio
from types import SimpleNamespace

import discord
import pytest

import agent.communication.discord_bot as discord_bot_module
import agent.communication.discord_services as discord_services_module
from agent.communication.discord_bot import DiscordBot
from agent.communication.message_router import MessageKind, ParsedMessage
from agent.loop import TaskResult
from agent.task_waits import TaskWaitRegistry


class _NullTyping:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeChannel:
    def __init__(self, channel_id: int) -> None:
        self.id = channel_id
        self.sent: list[str] = []

    def typing(self) -> _NullTyping:
        return _NullTyping()

    async def send(self, content: str = "", *, file=None) -> None:
        self.sent.append(content)


class _FakeMessage:
    def __init__(self, channel: _FakeChannel) -> None:
        self.channel = channel
        self.id = 999
        self.content = "please fix it"
        self.author = SimpleNamespace(display_name="Josh", bot=False)
        self.replies: list[str] = []
        self.reactions: list[str] = []

    async def reply(self, content: str, mention_author: bool = False) -> None:
        self.replies.append(content)

    async def add_reaction(self, emoji: str) -> None:
        self.reactions.append(emoji)


class _FakeLoop:
    def __init__(self) -> None:
        self.has_pending_work = False
        self.queue = SimpleNamespace(qsize=lambda: 0)
        self.enqueued = None
        self.wait_registry = TaskWaitRegistry()
        self.memory = None

    async def enqueue(self, task) -> None:
        self.enqueued = task
        assert task.response_future is not None
        task.response_future.set_result(
            TaskResult(
                task=task,
                output="done",
                success=True,
                elapsed_ms=1.0,
            )
        )

    async def _process(self, task) -> TaskResult:  # pragma: no cover - should never be called
        raise AssertionError("DiscordBot should enqueue work instead of calling _process directly")

    def build_resumed_task(self, *, suspended, answer: str, author: str, source: str, metadata_overrides=None):
        metadata = self.wait_registry.build_resumed_metadata(suspended, answer=answer, resumed_from=source)
        if metadata_overrides:
            metadata.update(metadata_overrides)
        return SimpleNamespace(content=suspended.content, source=source, author=author, metadata=metadata)


class _EventClient:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}
        self.user = SimpleNamespace(id=1, display_name="agent-1", bot=True)
        self.guilds = [SimpleNamespace(name="Guild One")]

    def event(self, fn):
        self.handlers[fn.__name__] = fn
        return fn


@pytest.mark.asyncio
async def test_free_discord_message_uses_queue_path(monkeypatch: pytest.MonkeyPatch) -> None:
    channel_id = 123
    loop = _FakeLoop()
    bot = DiscordBot(loop)  # type: ignore[arg-type]
    fake_channel = _FakeChannel(channel_id)
    fake_client = SimpleNamespace(
        user=SimpleNamespace(id=1, bot=True),
        get_channel=lambda _channel_id: fake_channel,
    )
    bot._client = fake_client  # type: ignore[assignment]
    bot._presenter._client = fake_client  # type: ignore[assignment]
    bot._messages._client = fake_client  # type: ignore[assignment]

    monkeypatch.setattr(discord_bot_module.settings, "discord_agent_channel_id", channel_id)
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda message, user: ParsedMessage(
            kind=MessageKind.TASK,
            content="Fix the bug",
            author="Josh",
            channel_id=channel_id,
            message_id=message.id,
        ),
    )

    monkeypatch.setattr(discord_bot_module.bridge, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(discord_bot_module.bridge, "unregister", lambda *args, **kwargs: None)

    message = _FakeMessage(fake_channel)
    await bot._handle_message(message)  # type: ignore[arg-type]

    assert loop.enqueued is not None
    assert loop.enqueued.content == "Fix the bug"
    assert message.reactions == ["👀", "🏁"]
    assert message.replies == ["done"]


@pytest.mark.asyncio
async def test_start_bot_handles_login_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = DiscordBot(_FakeLoop())  # type: ignore[arg-type]

    async def _start(_token: str) -> None:
        raise discord.LoginFailure("bad token")

    monkeypatch.setattr(discord_bot_module.settings, "discord_bot_token", "token")
    monkeypatch.setattr(bot._client, "start", _start)

    await bot.start_bot()


@pytest.mark.asyncio
async def test_start_bot_returns_early_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = DiscordBot(_FakeLoop())  # type: ignore[arg-type]
    started: list[str] = []

    async def _start(_token: str) -> None:
        started.append("started")

    monkeypatch.setattr(discord_bot_module.settings, "discord_bot_token", "")
    monkeypatch.setattr(bot._client, "start", _start)

    await bot.start_bot()

    assert started == []


@pytest.mark.asyncio
async def test_start_bot_handles_cancelled_error_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = DiscordBot(_FakeLoop())  # type: ignore[arg-type]
    closed: list[str] = []

    async def _start(_token: str) -> None:
        raise asyncio.CancelledError

    async def _close() -> None:
        closed.append("closed")

    monkeypatch.setattr(discord_bot_module.settings, "discord_bot_token", "token")
    monkeypatch.setattr(bot._client, "start", _start)
    monkeypatch.setattr(bot._client, "close", _close)

    await bot.start_bot()

    assert closed == ["closed"]


@pytest.mark.asyncio
async def test_start_bot_handles_unexpected_error_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = DiscordBot(_FakeLoop())  # type: ignore[arg-type]
    closed: list[str] = []

    async def _start(_token: str) -> None:
        raise RuntimeError("boom")

    async def _close() -> None:
        closed.append("closed")

    monkeypatch.setattr(discord_bot_module.settings, "discord_bot_token", "token")
    monkeypatch.setattr(bot._client, "start", _start)
    monkeypatch.setattr(bot._client, "close", _close)

    await bot.start_bot()

    assert closed == ["closed"]


@pytest.mark.asyncio
async def test_announce_online_posts_status(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = DiscordBot(_FakeLoop())  # type: ignore[arg-type]
    channel = _FakeChannel(55)
    bot._client = SimpleNamespace(get_channel=lambda channel_id: channel)  # type: ignore[assignment]

    monkeypatch.setattr(discord_bot_module.settings, "discord_bus_channel_id", 55)
    monkeypatch.setattr(discord_bot_module.settings, "agent_name", "agent-1")
    monkeypatch.setattr(discord_bot_module.settings, "agent_model", "gpt-4o")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.2.3")

    await bot._announce_online()

    assert "agent-1" in channel.sent[0]
    assert "1.2.3" in channel.sent[0]


@pytest.mark.asyncio
async def test_announce_online_uses_unknown_version_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = DiscordBot(_FakeLoop())  # type: ignore[arg-type]
    channel = _FakeChannel(55)
    bot._client = SimpleNamespace(get_channel=lambda channel_id: channel)  # type: ignore[assignment]

    monkeypatch.setattr(discord_bot_module.settings, "discord_bus_channel_id", 55)
    monkeypatch.setattr(discord_bot_module.settings, "agent_name", "agent-1")
    monkeypatch.setattr(discord_bot_module.settings, "agent_model", "gpt-4o")

    def _raise(name: str) -> str:
        raise RuntimeError("missing")

    monkeypatch.setattr("importlib.metadata.version", _raise)

    await bot._announce_online()

    assert "unknown" in channel.sent[0]


@pytest.mark.asyncio
async def test_announce_online_swallows_http_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenChannel(_FakeChannel):
        async def send(self, content: str = "", *, file=None) -> None:
            raise RuntimeError("send failed")

    bot = DiscordBot(_FakeLoop())  # type: ignore[arg-type]
    bot._client = SimpleNamespace(get_channel=lambda channel_id: _BrokenChannel(55))  # type: ignore[assignment]

    monkeypatch.setattr(discord_bot_module.settings, "discord_bus_channel_id", 55)
    monkeypatch.setattr(discord_bot_module.settings, "agent_name", "agent-1")
    monkeypatch.setattr(discord_bot_module.settings, "agent_model", "gpt-4o")
    monkeypatch.setattr(discord_bot_module.discord, "HTTPException", RuntimeError)

    await bot._announce_online()


@pytest.mark.asyncio
async def test_announce_online_returns_when_bus_channel_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = DiscordBot(_FakeLoop())  # type: ignore[arg-type]
    channel = _FakeChannel(55)
    bot._client = SimpleNamespace(get_channel=lambda channel_id: channel)  # type: ignore[assignment]

    monkeypatch.setattr(discord_bot_module.settings, "discord_bus_channel_id", 0)

    await bot._announce_online()

    assert channel.sent == []


@pytest.mark.asyncio
async def test_announce_online_returns_when_channel_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = DiscordBot(_FakeLoop())  # type: ignore[arg-type]
    bot._client = SimpleNamespace(get_channel=lambda channel_id: None)  # type: ignore[assignment]

    monkeypatch.setattr(discord_bot_module.settings, "discord_bus_channel_id", 55)

    await bot._announce_online()


@pytest.mark.asyncio
async def test_discord_bot_wrapper_methods_delegate_to_services(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = DiscordBot(_FakeLoop())  # type: ignore[arg-type]
    calls: list[tuple[str, object]] = []

    async def fake_send_reply(parsed, output, original_message) -> None:
        calls.append(("reply", output))

    async def fake_send_chunked(channel, text) -> None:
        calls.append(("chunked", text))

    async def fake_post_bus_status(message: str) -> None:
        calls.append(("bus", message))

    monkeypatch.setattr(bot._messages, "send_reply", fake_send_reply)
    monkeypatch.setattr(bot._presenter, "send_chunked", fake_send_chunked)
    monkeypatch.setattr(bot._messages, "post_bus_status", fake_post_bus_status)
    monkeypatch.setattr(bot._presenter, "make_sink", lambda channel: ("sink", channel.id))

    channel = _FakeChannel(55)
    message = _FakeMessage(channel)

    await bot._send_reply(None, "done", message)  # type: ignore[arg-type]
    await bot._send_chunked(channel, "long text")
    await bot._post_bus_status("ready")
    sink = bot._make_discord_sink(channel)

    assert calls == [("reply", "done"), ("chunked", "long text"), ("bus", "ready")]
    assert sink == ("sink", 55)


@pytest.mark.asyncio
async def test_setup_events_registers_handlers_and_routes_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = object.__new__(DiscordBot)
    bot._client = _EventClient()
    seen: list[tuple[str, object]] = []

    async def _announce_online() -> None:
        seen.append(("ready", None))

    async def _handle_message(message) -> None:
        seen.append(("message", message.id))

    bot._announce_online = _announce_online  # type: ignore[assignment]
    bot._handle_message = _handle_message  # type: ignore[assignment]

    warnings: list[str] = []
    infos: list[str] = []
    monkeypatch.setattr(discord_bot_module.log, "warning", lambda event, **kwargs: warnings.append(event))
    monkeypatch.setattr(discord_bot_module.log, "info", lambda event, **kwargs: infos.append(event))

    bot._setup_events()

    await bot._client.handlers["on_ready"]()
    await bot._client.handlers["on_message"](SimpleNamespace(id=7))
    bot._client.user = None
    await bot._client.handlers["on_message"](SimpleNamespace(id=8))
    await bot._client.handlers["on_disconnect"]()
    await bot._client.handlers["on_resumed"]()

    assert seen == [("ready", None), ("message", 7)]
    assert infos == ["discord_ready", "discord_resumed"]
    assert warnings == ["discord_disconnected"]
