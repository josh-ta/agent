from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.communication.discord_services import DiscordEventPresenter, MessageHandlingService
from agent.communication.message_router import MessageKind, ParsedMessage
from agent.events import ShellDoneEvent, ShellOutputEvent, ShellStartEvent
from agent.loop import TaskResult


class _BusyLoop:
    def __init__(self) -> None:
        self.has_pending_work = True
        self.queue = SimpleNamespace(qsize=lambda: 2)
        self.enqueued = None

    async def enqueue(self, task) -> None:
        self.enqueued = task


class _ReadyLoop:
    def __init__(self) -> None:
        self.has_pending_work = False
        self.queue = SimpleNamespace(qsize=lambda: 0)

    async def enqueue(self, task) -> None:
        assert task.response_future is not None
        task.response_future.set_result(
            TaskResult(task=task, output="finished", success=True, elapsed_ms=1.0)
        )


@pytest.mark.asyncio
async def test_message_service_queues_when_busy(fake_client, discord_channels, fake_message_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    service = MessageHandlingService(
        agent_loop=_BusyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    message = fake_message_factory(
        channel=discord_channels["private"],
        content="please help",
    )

    await service.handle_message(message)  # type: ignore[arg-type]

    assert "queued yours" in message.replies[0]


@pytest.mark.asyncio
async def test_message_service_sends_a2a_result_to_comms_and_bus(fake_client, discord_channels, fake_message_factory) -> None:
    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    parsed = ParsedMessage(
        kind=MessageKind.A2A,
        content="",
        author="peer-1",
        channel_id=discord_channels["comms"].id,
        message_id=1,
        a2a_payload={"from": "peer-1"},
    )
    message = fake_message_factory(channel=discord_channels["private"], content="hello")

    await service.send_reply(parsed, "result payload", message)  # type: ignore[arg-type]

    assert '"task": "result"' in discord_channels["comms"].sent[0]
    assert "completed task from peer-1" in discord_channels["bus"].sent[0]


@pytest.mark.asyncio
async def test_discord_event_presenter_buffers_shell_output(fake_client, discord_channels) -> None:
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    sink = presenter.make_sink(discord_channels["private"])  # type: ignore[arg-type]

    await sink(ShellStartEvent(command="pytest", cwd="/tmp"))
    await sink(ShellOutputEvent(chunk="line 1\n"))
    await sink(ShellDoneEvent(exit_code=1, elapsed_s=1.2))

    sent_message = discord_channels["private"].sent_messages[0]
    assert discord_channels["private"].sent[0] == "$ `pytest`"
    assert "line 1" in sent_message.edits[0]
    assert "exit 1 (1.2s)" in sent_message.edits[0]
