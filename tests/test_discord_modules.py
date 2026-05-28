from __future__ import annotations

import pytest

from agent.communication.discord_commands import CommandHandler, command_help_text, parse_native_command
from agent.communication.discord_constants import format_args
from agent.communication.discord_session import DiscordSessionState
from agent.core_services import ModelFactory


def test_parse_native_command_force_cancel() -> None:
    cmd = parse_native_command("/force-cancel")
    assert cmd is not None
    assert cmd.name == "force-cancel"


def test_command_help_mentions_threads_and_force_cancel() -> None:
    text = command_help_text()
    assert "/force-cancel" in text
    assert "/config" in text


def test_discord_session_state_cancel_tracking() -> None:
    state = DiscordSessionState()
    assert not state.is_cancelling(42)
    state.mark_cancelling(42)
    assert state.is_cancelling(42)
    state.clear_cancelling(42)
    assert not state.is_cancelling(42)


def test_format_args_truncates_long_values() -> None:
    rendered = format_args({"key": "x" * 100})
    assert "…" in rendered


def test_model_factory_ignores_invalid_mcp_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.config import settings

    monkeypatch.setattr(settings, "mcp_servers_json", "{not-json")
    monkeypatch.setattr(settings, "browser_mcp_url", "")
    assert ModelFactory().mcp_servers() == []


@pytest.mark.asyncio
async def test_command_handler_clear_reports_empty_queue() -> None:
    from types import SimpleNamespace

    from agent.communication.discord_commands import NativeCommand
    from agent.communication.message_router import MessageKind, ParsedMessage
    from tests.conftest import FakeChannel, FakeDiscordMessage

    channel = FakeChannel(id=42)
    message = FakeDiscordMessage(channel=channel)
    replies: list[str] = []

    async def reply_safe(_message, text: str) -> None:
        replies.append(text)

    async def clear_tasks(**_kwargs) -> int:
        return 0

    async def acknowledge(_message) -> None:
        return None

    service = SimpleNamespace(
        _agent_loop=SimpleNamespace(
            wait_registry=SimpleNamespace(pending_for_channel=lambda _cid: []),
            queue=SimpleNamespace(qsize=lambda: 0),
            has_pending_work=False,
        ),
        _is_operator_surface=lambda _parsed: True,
        _acknowledge_message=acknowledge,
        _reply_safe=reply_safe,
        _clear_queued_channel_tasks=clear_tasks,
    )
    handler = CommandHandler(service=service)  # type: ignore[arg-type]
    parsed = ParsedMessage(
        kind=MessageKind.TASK,
        content="/clear",
        author="Test User",
        channel_id=42,
        message_id=1,
    )
    handled = await handler.handle(
        message=message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="clear"),
        task_content="",
        attachment_metadata={},
    )
    assert handled is True
    assert "no queued tasks" in replies[0].lower()


@pytest.mark.asyncio
async def test_command_handler_status_uses_fallback_format() -> None:
    from types import SimpleNamespace

    from agent.communication.discord_commands import NativeCommand
    from agent.communication.message_router import MessageKind, ParsedMessage
    from tests.conftest import FakeChannel, FakeDiscordMessage

    channel = FakeChannel(id=42)
    message = FakeDiscordMessage(channel=channel)
    replies: list[str] = []

    async def reply_safe(_message, text: str) -> None:
        replies.append(text)

    service = SimpleNamespace(
        _agent_loop=SimpleNamespace(
            wait_registry=SimpleNamespace(pending_for_channel=lambda _cid: ["q1"]),
            queue=SimpleNamespace(qsize=lambda: 2),
            has_pending_work=True,
        ),
        _is_operator_surface=lambda _parsed: True,
        _acknowledge_message=lambda _message: __import__("asyncio").sleep(0),
        _reply_safe=reply_safe,
    )
    handler = CommandHandler(service=service)  # type: ignore[arg-type]
    parsed = ParsedMessage(
        kind=MessageKind.TASK,
        content="/status",
        author="Test User",
        channel_id=42,
        message_id=1,
    )
    handled = await handler.handle(
        message=message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="status"),
        task_content="",
        attachment_metadata={},
    )
    assert handled is True
    assert "Active: yes" in replies[0]
    assert "Queued: 2" in replies[0]


@pytest.mark.asyncio
async def test_command_handler_force_cancel_marks_cancelling() -> None:
    from types import SimpleNamespace

    from agent.communication.discord_commands import NativeCommand
    from agent.communication.discord_session import DiscordSessionState
    from agent.communication.message_router import MessageKind, ParsedMessage
    from tests.conftest import FakeChannel, FakeDiscordMessage

    channel = FakeChannel(id=42)
    message = FakeDiscordMessage(channel=channel)
    replies: list[str] = []
    session_state = DiscordSessionState()

    async def reply_safe(_message, text: str) -> None:
        replies.append(text)

    async def clear_tasks(**_kwargs) -> int:
        return 0

    async def cancel_active(**_kwargs) -> bool:
        return True

    service = SimpleNamespace(
        _agent_loop=SimpleNamespace(),
        _is_operator_surface=lambda _parsed: True,
        _acknowledge_message=lambda _message: __import__("asyncio").sleep(0),
        _reply_safe=reply_safe,
        _clear_queued_channel_tasks=clear_tasks,
        _request_cancel_active_task=cancel_active,
        _session_state=session_state,
        _sticky_sessions={},
    )
    handler = CommandHandler(service=service)  # type: ignore[arg-type]
    parsed = ParsedMessage(
        kind=MessageKind.TASK,
        content="/force-cancel",
        author="Test User",
        channel_id=42,
        message_id=1,
    )
    handled = await handler.handle(
        message=message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="force-cancel"),
        task_content="",
        attachment_metadata={},
    )
    assert handled is True
    assert "Force cancelling" in replies[0]
    assert session_state.is_cancelling(42)
