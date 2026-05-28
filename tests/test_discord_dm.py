from __future__ import annotations

from types import SimpleNamespace

import discord
import pytest

from agent.communication.discord_constants import (
    dm_allowed_user_ids,
    dm_user_allowed,
    is_dm_channel,
)
from agent.communication.discord_services import MessageHandlingService
from agent.communication.message_router import MessageKind, ParsedMessage
from tests.conftest import FakeChannel


def test_is_dm_channel_detects_private_type() -> None:
    assert is_dm_channel(FakeChannel(id=1, type=discord.ChannelType.private)) is True
    assert is_dm_channel(FakeChannel(id=1, type=discord.ChannelType.text)) is False


def test_dm_user_allowed_respects_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.communication.discord_constants.settings.discord_dm_enabled", True)
    monkeypatch.setattr(
        "agent.communication.discord_constants.settings.discord_dm_allowed_user_ids",
        "7, 9",
    )
    assert dm_allowed_user_ids() == frozenset({7, 9})
    assert dm_user_allowed(7) is True
    assert dm_user_allowed(8) is False


@pytest.mark.asyncio
async def test_resolve_stream_target_uses_message_channel_for_dm(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.communication.discord_services import MessageHandlingService
    from agent.communication.message_router import MessageKind, ParsedMessage

    monkeypatch.setattr("agent.communication.discord_services.settings.discord_agent_channel_id", 101)
    guild_channel = FakeChannel(id=101)
    dm_channel = FakeChannel(id=999)
    client = SimpleNamespace(get_channel=lambda _cid: guild_channel)
    service = MessageHandlingService(
        agent_loop=SimpleNamespace(),
        client=client,  # type: ignore[arg-type]
        presenter=SimpleNamespace(),
    )
    parsed = ParsedMessage(MessageKind.TASK, "hi", "josh", 999, 1, is_direct_message=True)

    work, reply = await service._resolve_stream_target(
        parsed=parsed,
        task_content="analyze sales",
        private_channel=guild_channel,  # type: ignore[arg-type]
        message_channel=dm_channel,  # type: ignore[arg-type]
    )

    assert work is dm_channel
    assert reply is dm_channel


def test_is_operator_surface_treats_dm_and_private_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.communication.discord_services.settings.discord_agent_channel_id", 101)
    service = MessageHandlingService(
        agent_loop=SimpleNamespace(),
        client=SimpleNamespace(),
        presenter=SimpleNamespace(),
    )
    dm = ParsedMessage(MessageKind.TASK, "hi", "josh", 404, 1, is_direct_message=True)
    private = ParsedMessage(MessageKind.TASK, "hi", "josh", 101, 1)
    bus = ParsedMessage(MessageKind.BUS, "hi", "josh", 202, 1)

    assert service._is_operator_surface(dm) is True
    assert service._is_operator_surface(private) is True
    assert service._is_operator_surface(bus) is False
