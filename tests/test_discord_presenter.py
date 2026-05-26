from __future__ import annotations

import asyncio
from types import SimpleNamespace

import discord
import pytest

from agent.communication.discord_presenter import (
    DiscordEventPresenter,
    StatusEmbedManager,
    edit_with_retry,
    send_with_retry,
)
from agent.communication.discord_session import DiscordSessionState
from agent.events import TextDeltaEvent, TextTurnEndEvent
from tests.conftest import FakeChannel, FakeSentMessage


@pytest.mark.asyncio
async def test_send_with_retry_empty_content() -> None:
    channel = FakeChannel(id=1)
    sent = await send_with_retry(channel, content="")
    assert sent is None


@pytest.mark.asyncio
async def test_status_embed_manager_set_stopped() -> None:
    channel = FakeChannel(id=1)
    manager = StatusEmbedManager(channel, debounce_seconds=0)
    manager.set_stopped()
    await manager.flush()
    assert "Stopped" in channel.sent_messages[0].embed.description


@pytest.mark.asyncio
async def test_presenter_send_chunked_reports_failure(fake_client) -> None:
    channel = FakeChannel(id=1)

    async def fail_send(content: str = "", *, file=None, embed=None):
        response = SimpleNamespace(status=500, reason="Error")
        raise discord.HTTPException(response, "fail")

    channel.send = fail_send  # type: ignore[method-assign]
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    ok = await presenter.send_chunked(channel, "hello")
    assert ok is False


@pytest.mark.asyncio
async def test_status_embed_manager_flush_and_finalize() -> None:
    channel = FakeChannel(id=1)
    manager = StatusEmbedManager(channel, debounce_seconds=0)
    await manager.handle_tool("run_shell", {"command": "pytest"})
    await manager.flush()
    assert channel.sent_messages
    await manager.finalize(success=True)
    assert channel.sent_messages[0].deleted


@pytest.mark.asyncio
async def test_presenter_streams_text_deltas(fake_client, fake_message_factory, discord_channels) -> None:
    channel = FakeChannel(id=1)
    message = fake_message_factory(channel=discord_channels["private"], content="hello")
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    sink = presenter.make_sink(
        channel,
        debounce_seconds=0,
        main_channel=channel,
        channel_id=7,
        session_state=DiscordSessionState(),
        reply_to=message,  # type: ignore[arg-type]
    )
    await sink(TextDeltaEvent(delta="Hello! How can I assist "))
    await sink(TextDeltaEvent(delta="you today? 😊"))
    assert message.replies == []
    await sink.finalize_reply("Hello! How can I assist you today? 😊")  # type: ignore[attr-defined]
    assert message.replies == ["Hello! How can I assist you today? 😊"]
    assert sink.reply_delivered()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_presenter_keeps_longer_stream_buffer_on_short_final_turn(fake_client, fake_message_factory, discord_channels) -> None:
    channel = FakeChannel(id=1)
    message = fake_message_factory(channel=discord_channels["private"], content="hello")
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    sink = presenter.make_sink(
        channel,
        debounce_seconds=0,
        reply_to=message,  # type: ignore[arg-type]
    )
    await sink(TextDeltaEvent(delta="Hello! How can I assist "))
    await sink(TextTurnEndEvent(text="you today? 😊", is_final=True))
    assert message.replies == []
    await sink.finalize_reply("Hello! How can I assist you today? 😊")  # type: ignore[attr-defined]
    assert message.replies == ["Hello! How can I assist you today? 😊"]


@pytest.mark.asyncio
async def test_presenter_sink_filters_stale_run_generation(fake_client) -> None:
    channel = FakeChannel(id=1)
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    sink = presenter.make_sink(channel, expected_run_generation=2, debounce_seconds=0)

    from agent.events import ProgressEvent

    event = ProgressEvent(message="tick")
    event.run_generation = 1  # type: ignore[attr-defined]
    await sink(event)
    await asyncio.sleep(0.01)
    assert channel.sent == []


@pytest.mark.asyncio
async def test_presenter_sink_covers_lifecycle_events(fake_client) -> None:
    channel = FakeChannel(id=1)
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    sink = presenter.make_sink(channel, debounce_seconds=0)

    from agent.events import (
        ShellDoneEvent,
        ShellOutputEvent,
        ShellStartEvent,
        TaskErrorEvent,
        TaskStartEvent,
        ThinkingEndEvent,
        ToolCallStartEvent,
    )

    await sink(TaskStartEvent(content="task", tier="fast"))
    await sink(ThinkingEndEvent(text="thought"))
    await sink(ToolCallStartEvent(tool_name="run_shell", call_id="1", args={"command": "ls"}))
    await sink(ShellStartEvent(command="ls", cwd="/tmp"))
    await sink(ShellOutputEvent(chunk="ok\n"))
    await sink(ShellDoneEvent(exit_code=0, elapsed_s=0.1))
    await sink(TaskErrorEvent(error="boom"))
    await asyncio.sleep(0.01)

    assert channel.sent_messages
    assert channel.sent_messages[0].deleted
    assert not any("thought" in item for item in channel.sent)


@pytest.mark.asyncio
async def test_presenter_create_task_thread_without_support(fake_client) -> None:
    channel = FakeChannel(id=1)
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    result = await presenter.create_task_thread(channel, task_summary="deploy")
    assert result is None


@pytest.mark.asyncio
async def test_send_with_retry_recovers_from_rate_limit() -> None:
    channel = FakeChannel(id=1)
    attempts = 0

    async def flaky_send(content: str = "", *, file=None, embed=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            response = SimpleNamespace(status=429, reason="Rate limited")
            exc = discord.HTTPException(response, "rate limited")
            exc.retry_after = 0.01
            raise exc
        return await FakeChannel.send(channel, content, file=file, embed=embed)

    channel.send = flaky_send  # type: ignore[method-assign]
    sent = await send_with_retry(channel, content="hello")
    assert sent is not None
    assert attempts == 2


@pytest.mark.asyncio
async def test_edit_with_retry_recovers_from_rate_limit() -> None:
    message = FakeSentMessage(content="old")
    attempts = 0

    async def flaky_edit(*, content: str | None = None, embed=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            response = SimpleNamespace(status=429, reason="Rate limited")
            exc = discord.HTTPException(response, "rate limited")
            exc.retry_after = 0.01
            raise exc
        await FakeSentMessage.edit(message, content=content, embed=embed)

    message.edit = flaky_edit  # type: ignore[method-assign]
    assert await edit_with_retry(message, content="new") is True
    assert attempts == 2


@pytest.mark.asyncio
async def test_edit_with_retry_returns_false_on_failure() -> None:
    message = FakeSentMessage(content="old")

    async def fail_edit(*, content: str | None = None, embed=None):
        response = SimpleNamespace(status=500, reason="Error")
        raise discord.HTTPException(response, "fail")

    message.edit = fail_edit  # type: ignore[method-assign]
    assert await edit_with_retry(message, content="new") is False


@pytest.mark.asyncio
async def test_status_embed_manager_debounced_flush() -> None:
    channel = FakeChannel(id=1)
    manager = StatusEmbedManager(channel, debounce_seconds=0.02)
    manager.set_cancelling()
    await asyncio.sleep(0.05)
    assert channel.sent_messages


@pytest.mark.asyncio
async def test_status_embed_manager_shell_failure_embed() -> None:
    channel = FakeChannel(id=1)
    manager = StatusEmbedManager(channel, debounce_seconds=0)
    await manager.handle_shell_output("stderr line")
    await manager.handle_shell_done(exit_code=1, elapsed_s=2.5)
    await manager.flush()
    assert "Shell failed" in channel.sent_messages[0].embed.description


@pytest.mark.asyncio
async def test_status_embed_manager_progress_with_cancel_hint() -> None:
    channel = FakeChannel(id=1)
    manager = StatusEmbedManager(channel, debounce_seconds=0)
    await manager.handle_progress("Cancel requested by operator")
    await manager.flush()
    assert "Cancel" in channel.sent_messages[0].embed.description


@pytest.mark.asyncio
async def test_status_embed_manager_edits_existing_message() -> None:
    channel = FakeChannel(id=1)
    manager = StatusEmbedManager(channel, debounce_seconds=0)
    await manager.handle_tool("run_shell", {"command": "ls"})
    await manager.flush()
    await manager.handle_tool("browser_navigate", {"url": "https://example.com"})
    await manager.flush()
    assert channel.sent_messages[0].embed_edits


@pytest.mark.asyncio
async def test_presenter_send_chunked_uses_fallback_channel(fake_client) -> None:
    primary = FakeChannel(id=1)
    fallback = FakeChannel(id=2)

    async def fail_send(content: str = "", *, file=None, embed=None):
        response = SimpleNamespace(status=500, reason="Error")
        raise discord.HTTPException(response, "fail")

    primary.send = fail_send  # type: ignore[method-assign]
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    ok = await presenter.send_chunked(primary, "hello", fallback_channel=fallback)
    assert ok is True
    assert fallback.sent_messages


@pytest.mark.asyncio
async def test_presenter_create_task_thread_success(fake_client) -> None:
    channel = FakeChannel(id=1)

    async def create_thread(*, name: str, auto_archive_duration: int = 60):
        return FakeChannel(id=99)

    channel.create_thread = create_thread  # type: ignore[attr-defined]
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    thread = await presenter.create_task_thread(channel, task_summary="deploy app")
    assert thread is not None
    assert thread.id == 99


@pytest.mark.asyncio
async def test_presenter_create_task_thread_handles_http_error(fake_client) -> None:
    channel = FakeChannel(id=1)

    async def create_thread(*, name: str, auto_archive_duration: int = 60):
        response = SimpleNamespace(status=403, reason="Forbidden")
        raise discord.HTTPException(response, "forbidden")

    channel.create_thread = create_thread  # type: ignore[attr-defined]
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    assert await presenter.create_task_thread(channel, task_summary="deploy") is None


@pytest.mark.asyncio
async def test_presenter_sink_shows_cancelling_state(fake_client) -> None:
    channel = FakeChannel(id=1)
    state = DiscordSessionState()
    state.mark_cancelling(7)
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    sink = presenter.make_sink(channel, channel_id=7, session_state=state, debounce_seconds=0)

    from agent.events import ProgressEvent

    await sink(ProgressEvent(message="still working"))
    await asyncio.sleep(0.01)
    assert "Cancelling" in channel.sent_messages[0].embed.description


@pytest.mark.asyncio
async def test_presenter_sink_non_final_text_turn(fake_client) -> None:
    channel = FakeChannel(id=1)
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    sink = presenter.make_sink(channel, debounce_seconds=0)
    await sink(TextTurnEndEvent(text="intermediate answer", is_final=False))
    await asyncio.sleep(0.01)
    assert channel.sent_messages
    assert "intermediate answer" in channel.sent_messages[0].embed.description


@pytest.mark.asyncio
async def test_presenter_send_attachments_logs_http_error(fake_client, monkeypatch) -> None:
    channel = FakeChannel(id=1)
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]

    async def fail_send(*_args, **_kwargs):
        response = SimpleNamespace(status=500, reason="Error")
        raise discord.HTTPException(response, "fail")

    monkeypatch.setattr(
        "agent.communication.discord_presenter.discord_tools_module.send_attachments",
        fail_send,
    )
    await presenter.send_attachments(channel, [])
