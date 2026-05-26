from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.communication.discord_config import ConfigCommandHandler
from agent.communication.discord_commands import NativeCommand
from agent.communication.message_router import MessageKind, ParsedMessage
from tests.conftest import FakeChannel, FakeDiscordMessage


@pytest.mark.asyncio
async def test_config_command_direct_set(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from agent.config import settings as base_settings

    cfg = base_settings.model_copy(
        update={
            "runtime_overrides_path": tmp_path / "overrides.json",
            "agent_model": "claude-haiku-4-5",
        }
    )
    monkeypatch.setattr("agent.runtime_config.settings", cfg)

    channel = FakeChannel(id=42)
    message = FakeDiscordMessage(channel=channel)
    replies: list[str] = []
    reloaded: list[bool] = []

    async def reply_safe(_message, text: str) -> None:
        replies.append(text)

    loop = SimpleNamespace(reload_agents=lambda: reloaded.append(True))
    service = SimpleNamespace(_reply_safe=reply_safe, _agent_loop=loop)
    handler = ConfigCommandHandler(service=service)  # type: ignore[arg-type]

    parsed = ParsedMessage(
        kind=MessageKind.TASK,
        content="/config AGENT_MODEL:claude-sonnet-4-5",
        author="Test User",
        channel_id=42,
        message_id=1,
    )
    handled = await handler.handle_command(
        message=message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="config", argument="AGENT_MODEL:claude-sonnet-4-5"),
    )

    assert handled is True
    assert "Updated" in replies[0]
    assert reloaded == [True]
    assert cfg.agent_model == "claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_config_wizard_flow(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from agent.config import settings as base_settings

    cfg = base_settings.model_copy(
        update={
            "runtime_overrides_path": tmp_path / "overrides.json",
            "thinking_enabled": False,
        }
    )
    monkeypatch.setattr("agent.runtime_config.settings", cfg)

    channel = FakeChannel(id=7)
    message = FakeDiscordMessage(channel=channel)
    replies: list[str] = []

    async def reply_safe(_message, text: str) -> None:
        replies.append(text)

    loop = SimpleNamespace(reload_agents=lambda: None)
    service = SimpleNamespace(_reply_safe=reply_safe, _agent_loop=loop)
    handler = ConfigCommandHandler(service=service)  # type: ignore[arg-type]
    parsed = ParsedMessage(
        kind=MessageKind.TASK,
        content="/config",
        author="Test User",
        channel_id=7,
        message_id=1,
    )

    await handler.handle_command(
        message=message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="config"),
    )
    assert "Config wizard" in replies[0]

    pick = ParsedMessage(
        kind=MessageKind.TASK,
        content="THINKING_ENABLED",
        author="Test User",
        channel_id=7,
        message_id=2,
    )
    pick_message = FakeDiscordMessage(channel=channel, content="THINKING_ENABLED")
    await handler.maybe_handle_wizard(pick_message, pick)  # type: ignore[arg-type]
    assert "Current `THINKING_ENABLED`" in replies[1]

    value = ParsedMessage(
        kind=MessageKind.TASK,
        content="true",
        author="Test User",
        channel_id=7,
        message_id=3,
    )
    value_message = FakeDiscordMessage(channel=channel, content="true")
    await handler.maybe_handle_wizard(value_message, value)  # type: ignore[arg-type]
    assert cfg.thinking_enabled is True
    assert "Updated" in replies[2]


@pytest.mark.asyncio
async def test_config_command_list_and_cancel(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from agent.config import settings as base_settings

    cfg = base_settings.model_copy(update={"runtime_overrides_path": tmp_path / "overrides.json"})
    monkeypatch.setattr("agent.runtime_config.settings", cfg)

    channel = FakeChannel(id=9)
    message = FakeDiscordMessage(channel=channel)
    replies: list[str] = []

    async def reply_safe(_message, text: str) -> None:
        replies.append(text)

    service = SimpleNamespace(_reply_safe=reply_safe, _agent_loop=SimpleNamespace(reload_agents=lambda: None))
    handler = ConfigCommandHandler(service=service)  # type: ignore[arg-type]
    parsed = ParsedMessage(
        kind=MessageKind.TASK,
        content="/config list",
        author="Test User",
        channel_id=9,
        message_id=1,
    )

    await handler.handle_command(
        message=message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="config", argument="list"),
    )
    assert "AGENT_MODEL" in replies[0]

    await handler.handle_command(
        message=message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="config"),
    )
    assert handler.has_wizard(9)

    await handler.handle_command(
        message=message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="config", argument="cancel"),
    )
    assert not handler.has_wizard(9)
    assert "cancelled" in replies[-1].lower()


@pytest.mark.asyncio
async def test_config_command_rejects_missing_colon(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from agent.config import settings as base_settings

    cfg = base_settings.model_copy(update={"runtime_overrides_path": tmp_path / "overrides.json"})
    monkeypatch.setattr("agent.runtime_config.settings", cfg)

    channel = FakeChannel(id=11)
    message = FakeDiscordMessage(channel=channel)
    replies: list[str] = []

    async def reply_safe(_message, text: str) -> None:
        replies.append(text)

    service = SimpleNamespace(_reply_safe=reply_safe, _agent_loop=SimpleNamespace(reload_agents=lambda: None))
    handler = ConfigCommandHandler(service=service)  # type: ignore[arg-type]
    parsed = ParsedMessage(kind=MessageKind.TASK, content="/config AGENT_MODEL", author="u", channel_id=11, message_id=1)

    await handler.handle_command(
        message=message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="config", argument="AGENT_MODEL"),
    )
    assert "Usage" in replies[0]


@pytest.mark.asyncio
async def test_config_wizard_accepts_numeric_selection(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from agent.config import settings as base_settings

    cfg = base_settings.model_copy(
        update={
            "runtime_overrides_path": tmp_path / "overrides.json",
            "permission_mode": "default",
        }
    )
    monkeypatch.setattr("agent.runtime_config.settings", cfg)

    channel = FakeChannel(id=12)
    message = FakeDiscordMessage(channel=channel)
    replies: list[str] = []

    async def reply_safe(_message, text: str) -> None:
        replies.append(text)

    service = SimpleNamespace(_reply_safe=reply_safe, _agent_loop=SimpleNamespace(reload_agents=lambda: None))
    handler = ConfigCommandHandler(service=service)  # type: ignore[arg-type]
    parsed = ParsedMessage(kind=MessageKind.TASK, content="/config", author="u", channel_id=12, message_id=1)

    await handler.handle_command(
        message=message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="config"),
    )

    pick_message = FakeDiscordMessage(channel=channel, content="8")
    pick = ParsedMessage(kind=MessageKind.TASK, content="8", author="u", channel_id=12, message_id=2)
    await handler.maybe_handle_wizard(pick_message, pick)  # type: ignore[arg-type]
    assert "PERMISSION_MODE" in replies[1]

    value_message = FakeDiscordMessage(channel=channel, content="plan")
    value = ParsedMessage(kind=MessageKind.TASK, content="plan", author="u", channel_id=12, message_id=3)
    await handler.maybe_handle_wizard(value_message, value)  # type: ignore[arg-type]
    assert cfg.permission_mode == "plan"


@pytest.mark.asyncio
async def test_config_wizard_rejects_invalid_selection(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from agent.config import settings as base_settings

    cfg = base_settings.model_copy(update={"runtime_overrides_path": tmp_path / "overrides.json"})
    monkeypatch.setattr("agent.runtime_config.settings", cfg)

    channel = FakeChannel(id=13)
    message = FakeDiscordMessage(channel=channel)
    replies: list[str] = []

    async def reply_safe(_message, text: str) -> None:
        replies.append(text)

    service = SimpleNamespace(_reply_safe=reply_safe, _agent_loop=SimpleNamespace(reload_agents=lambda: None))
    handler = ConfigCommandHandler(service=service)  # type: ignore[arg-type]
    parsed = ParsedMessage(kind=MessageKind.TASK, content="/config", author="u", channel_id=13, message_id=1)

    await handler.handle_command(
        message=message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="config"),
    )

    bad_message = FakeDiscordMessage(channel=channel, content="999")
    bad = ParsedMessage(kind=MessageKind.TASK, content="999", author="u", channel_id=13, message_id=2)
    await handler.maybe_handle_wizard(bad_message, bad)  # type: ignore[arg-type]
    assert "Reply with a number" in replies[1]
