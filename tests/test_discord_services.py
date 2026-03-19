from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import agent.communication.discord_services as discord_services_module
import agent.tools.discord_tools as discord_tools_module
from agent.communication.discord_services import DiscordEventPresenter, MAX_REPLY_LEN, MessageHandlingService
from agent.communication.message_router import MessageKind, ParsedMessage
from agent.config import settings
from agent.events import (
    ProgressEvent,
    ShellDoneEvent,
    ShellOutputEvent,
    ShellStartEvent,
    TaskErrorEvent,
    TaskStartEvent,
    TextTurnEndEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
)
from agent.loop import TaskResult
from agent.task_waits import TaskWaitRegistry
from tests.conftest import FakeChannel, FakeSentMessage


class _BusyLoop:
    def __init__(self) -> None:
        self.has_pending_work = True
        self.queue = SimpleNamespace(qsize=lambda: 2)
        self.enqueued = None
        self.wait_registry = TaskWaitRegistry()
        self.memory = None

    async def enqueue(self, task) -> None:
        self.enqueued = task

    def build_resumed_task(self, *, suspended, answer: str, author: str, source: str):
        metadata = self.wait_registry.build_resumed_metadata(suspended, answer=answer, resumed_from=source)
        return SimpleNamespace(content=suspended.content, source=source, author=author, metadata=metadata)


class _ReadyLoop:
    def __init__(self) -> None:
        self.has_pending_work = False
        self.queue = SimpleNamespace(qsize=lambda: 0)
        self.wait_registry = TaskWaitRegistry()
        self.memory = None

    async def enqueue(self, task) -> None:
        assert task.response_future is not None
        task.response_future.set_result(
            TaskResult(task=task, output="finished", success=True, elapsed_ms=1.0)
        )

    def build_resumed_task(self, *, suspended, answer: str, author: str, source: str):
        metadata = self.wait_registry.build_resumed_metadata(suspended, answer=answer, resumed_from=source)
        return SimpleNamespace(content=suspended.content, source=source, author=author, metadata=metadata)


def _parsed(
    kind: MessageKind,
    *,
    content: str = "please help",
    channel_id: int = 101,
    payload: dict | None = None,
) -> ParsedMessage:
    return ParsedMessage(
        kind=kind,
        content=content,
        author="Test User",
        channel_id=channel_id,
        message_id=1,
        a2a_payload=payload,
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

    assert message.reactions == ["👀"]
    assert "queued yours" in message.replies[0]


@pytest.mark.asyncio
async def test_message_service_uses_next_position_when_single_item_ahead(fake_client, discord_channels, fake_message_factory) -> None:
    loop = _BusyLoop()
    loop.queue = SimpleNamespace(qsize=lambda: 1)
    service = MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    message = fake_message_factory(channel=discord_channels["private"], content="please help")

    await service.handle_message(message)  # type: ignore[arg-type]

    assert "next up" in message.replies[0]


@pytest.mark.asyncio
async def test_message_service_resumes_suspended_task_from_reply(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _BusyLoop()
    service = MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    loop.wait_registry.suspend(
        task_id="task-1",
        source="discord",
        author="Josh",
        content="skip it",
        channel_id=discord_channels["private"].id,
        message_id=1,
        metadata={"task_id": "task-1"},
        question="Which environment?",
        timeout_s=300,
        base_prompt="prompt",
        tier="smart",
    )
    loop.wait_registry.bind_prompt_message("task-1", 42)
    message = fake_message_factory(channel=discord_channels["private"], content="skip it")
    message.reference = SimpleNamespace(message_id=42)

    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="skip it", channel_id=discord_channels["private"].id),
    )

    await service.handle_message(message)  # type: ignore[arg-type]

    assert message.reactions == ["👀"]
    assert message.replies == ["💬 Got it — resuming from your answer now."]
    assert service._agent_loop.enqueued is not None


@pytest.mark.asyncio
async def test_message_service_ignores_bus_and_blank_content(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    message = fake_message_factory(channel=discord_channels["private"], content="hello")

    monkeypatch.setattr(discord_services_module, "classify", lambda *_args: _parsed(MessageKind.BUS, channel_id=202))
    await service.handle_message(message)  # type: ignore[arg-type]

    monkeypatch.setattr(discord_services_module, "classify", lambda *_args: _parsed(MessageKind.TASK, content="   "))
    await service.handle_message(message)  # type: ignore[arg-type]

    assert message.replies == []
    assert discord_channels["private"].sent == []


@pytest.mark.asyncio
async def test_message_service_injects_followup_and_replies_inline(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    inject_q: asyncio.Queue[str] = asyncio.Queue()
    service._inject_queues[discord_channels["private"].id] = inject_q
    message = fake_message_factory(channel=discord_channels["private"], content="new detail")
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="new detail", channel_id=discord_channels["private"].id),
    )

    await service.handle_message(message)  # type: ignore[arg-type]

    assert await inject_q.get() == "new detail"
    assert message.reactions == ["👀"]
    assert "fold that into what I'm working on" in message.replies[0]


@pytest.mark.asyncio
async def test_message_service_injects_followup_without_inline_reply_in_comms(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    inject_q: asyncio.Queue[str] = asyncio.Queue()
    service._inject_queues[discord_channels["comms"].id] = inject_q
    message = fake_message_factory(channel=discord_channels["comms"], content="new detail")
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="new detail", channel_id=discord_channels["comms"].id),
    )

    await service.handle_message(message)  # type: ignore[arg-type]

    assert await inject_q.get() == "new detail"
    assert message.reactions == ["👀"]
    assert message.replies == []


@pytest.mark.asyncio
async def test_message_service_reacts_when_accepting_new_task(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="please help", channel_id=discord_channels["private"].id),
    )
    message = fake_message_factory(channel=discord_channels["private"], content="please help")

    await service.handle_message(message)  # type: ignore[arg-type]

    assert message.reactions == ["👀", "🏁"]


@pytest.mark.asyncio
async def test_message_service_prompts_and_marks_waiting_for_user(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _WaitingLoop(_ReadyLoop):
        async def enqueue(self, task) -> None:
            assert task.response_future is not None
            task.metadata["task_id"] = "task-wait"
            task.metadata["wait_state"] = {
                "question": "Which environment?",
                "timeout_s": 90,
                "channel_id": task.channel_id,
                "message_id": task.message_id,
                "prompt_message_id": None,
            }
            self.wait_registry.suspend(
                task_id="task-wait",
                source=task.source,
                author=task.author,
                content=task.content,
                channel_id=task.channel_id,
                message_id=task.message_id,
                metadata=task.metadata,
                question="Which environment?",
                timeout_s=90,
                base_prompt="",
                tier="smart",
            )
            task.response_future.set_result(
                TaskResult(
                    task=task,
                    output="",
                    success=None,
                    status="waiting_for_user",
                    elapsed_ms=1.0,
                    waiting_for_user=True,
                    question="Which environment?",
                    timeout_s=90,
                )
            )

    loop = _WaitingLoop()
    service = MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="please help", channel_id=discord_channels["private"].id),
    )
    message = fake_message_factory(channel=discord_channels["private"], content="please help")

    await service.handle_message(message)  # type: ignore[arg-type]

    assert "Which environment?" in message.replies[0]
    assert message.reactions == ["👀", "⏸️"]
    pending = loop.wait_registry.pending_for_channel(discord_channels["private"].id)
    assert len(pending) == 1
    assert pending[0].prompt_message_id == 1


@pytest.mark.asyncio
async def test_message_service_skips_sending_reply_for_empty_or_already_sent_results(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        TaskResult(task=SimpleNamespace(), output="", success=True, elapsed_ms=1.0),
        TaskResult(task=SimpleNamespace(), output="done", success=True, elapsed_ms=1.0, user_visible_reply_sent=True),
        TaskResult(
            task=SimpleNamespace(),
            output="done",
            success=True,
            elapsed_ms=1.0,
            user_visible_reply_sent=True,
            attachments=[discord_tools_module.DiscordAttachment(filename="browser-screenshot-1.png", data=b"png")],
        ),
    ]
    sent: list[str] = []

    class _Loop:
        has_pending_work = False
        queue = SimpleNamespace(qsize=lambda: 0)
        wait_registry = TaskWaitRegistry()
        memory = None

        async def enqueue(self, task) -> None:
            task.response_future.set_result(results.pop(0))

        def build_resumed_task(self, *, suspended, answer: str, author: str, source: str):
            metadata = self.wait_registry.build_resumed_metadata(suspended, answer=answer, resumed_from=source)
            return SimpleNamespace(content=suspended.content, source=source, author=author, metadata=metadata)

    service = MessageHandlingService(
        agent_loop=_Loop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    async def fake_send_reply(*args, **kwargs) -> bool:
        sent.append("reply")
        return True

    monkeypatch.setattr(
        service,
        "send_reply",
        fake_send_reply,
    )
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="please help", channel_id=discord_channels["private"].id),
    )
    message = fake_message_factory(channel=discord_channels["private"], content="please help")

    await service.handle_message(message)  # type: ignore[arg-type]
    await service.handle_message(message)  # type: ignore[arg-type]
    await service.handle_message(message)  # type: ignore[arg-type]

    assert sent == ["reply"]


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

    delivered = await service.send_reply(parsed, "result payload", message)  # type: ignore[arg-type]

    assert delivered is True
    assert '"task": "result"' in discord_channels["comms"].sent[0]
    assert "completed task from peer-1" in discord_channels["bus"].sent[0]


@pytest.mark.asyncio
async def test_message_service_send_reply_handles_a2a_failure(fake_client, discord_channels, fake_message_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    parsed = _parsed(
        MessageKind.A2A,
        channel_id=discord_channels["comms"].id,
        payload={"from": "peer-1"},
    )
    message = fake_message_factory(channel=discord_channels["private"], content="hello")

    class _ExplodingChannel(FakeChannel):
        async def send(self, content: str = "", *, file=None):
            raise RuntimeError("send failed")

    fake_client.channels[discord_channels["comms"].id] = _ExplodingChannel(id=discord_channels["comms"].id)
    monkeypatch.setattr(discord_services_module.discord, "HTTPException", RuntimeError)

    delivered = await service.send_reply(parsed, "result payload", message)  # type: ignore[arg-type]

    assert delivered is False
    assert "completed task from peer-1" in discord_channels["bus"].sent[0]


@pytest.mark.asyncio
async def test_message_service_send_reply_for_bus_posts_status_and_private_copy(
    fake_client,
    discord_channels,
    fake_message_factory,
) -> None:
    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    parsed = _parsed(MessageKind.TASK, channel_id=discord_channels["bus"].id)
    message = fake_message_factory(channel=discord_channels["bus"], content="hello")

    delivered = await service.send_reply(parsed, "bus output", message)  # type: ignore[arg-type]

    assert delivered is True
    assert discord_channels["bus"].sent[0].startswith(f"**{settings.agent_name}**:")
    assert discord_channels["private"].sent[0] == "bus output"


@pytest.mark.asyncio
async def test_message_service_send_reply_splits_private_reply(
    fake_client,
    discord_channels,
    fake_message_factory,
) -> None:
    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    parsed = _parsed(MessageKind.TASK, channel_id=discord_channels["private"].id)
    message = fake_message_factory(channel=discord_channels["private"], content="hello")
    output = "x" * (MAX_REPLY_LEN + 20)

    delivered = await service.send_reply(parsed, output, message)  # type: ignore[arg-type]

    assert delivered is True
    assert message.replies == ["x" * MAX_REPLY_LEN]
    assert discord_channels["private"].sent == ["x" * 20]


@pytest.mark.asyncio
async def test_message_service_send_reply_uploads_screenshot_attachments(
    fake_client,
    discord_channels,
    fake_message_factory,
) -> None:
    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    parsed = _parsed(MessageKind.TASK, channel_id=discord_channels["private"].id)
    message = fake_message_factory(channel=discord_channels["private"], content="hello")

    delivered = await service.send_reply(
        parsed,
        "screenshot attached",
        message,  # type: ignore[arg-type]
        attachments=[discord_tools_module.DiscordAttachment(filename="browser-screenshot-1.png", data=b"png")],
    )

    assert delivered is True
    assert message.replies == ["screenshot attached"]
    assert discord_channels["private"].sent_files == ["browser-screenshot-1.png"]


@pytest.mark.asyncio
async def test_post_bus_status_handles_noop_and_failure(fake_client, discord_channels, monkeypatch: pytest.MonkeyPatch) -> None:
    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )

    monkeypatch.setattr(settings, "discord_bus_channel_id", 0)
    await service.post_bus_status("ignored")

    monkeypatch.setattr(settings, "discord_bus_channel_id", discord_channels["bus"].id)
    monkeypatch.setattr(discord_services_module.discord, "HTTPException", RuntimeError)

    class _ExplodingChannel(FakeChannel):
        async def send(self, content: str = "", *, file=None):
            raise RuntimeError("send failed")

    fake_client.channels[discord_channels["bus"].id] = _ExplodingChannel(id=discord_channels["bus"].id)
    await service.post_bus_status("hello")


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


@pytest.mark.asyncio
async def test_discord_event_presenter_renders_text_tool_progress_and_error(fake_client, discord_channels) -> None:
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    sink = presenter.make_sink(discord_channels["private"])  # type: ignore[arg-type]

    await sink(TaskStartEvent(content="Investigate the failing deployment workflow", tier="smart"))
    await sink(ThinkingEndEvent(text="first *idea*"))
    await sink(TextTurnEndEvent(text="working", is_final=False))
    await sink(ToolCallStartEvent(tool_name="run_shell", call_id="1", args={"command": "pytest"}))
    await sink(ToolCallStartEvent(tool_name="read_file", call_id="2", args={"path": "README.md"}))
    await sink(ProgressEvent(message="still working"))
    await sink(TaskErrorEvent(error="boom"))

    assert discord_channels["private"].sent == [
        "🟢 Starting: Investigate the failing deployment workflow",
        "🧠 *first \\*idea\\**",
        "💭 working",
        "🔧 `run_shell(command=pytest)`",
        "still working",
        "❌ boom",
    ]


@pytest.mark.asyncio
async def test_discord_event_presenter_falls_back_when_shell_edit_fails(fake_client, monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenSentMessage(FakeSentMessage):
        async def edit(self, *, content: str) -> None:
            raise RuntimeError("edit failed")

    class _BrokenChannel(FakeChannel):
        async def send(self, content: str = "", *, file=None) -> FakeSentMessage:
            self.sent.append(content)
            if file is not None:
                self.sent_files.append(getattr(file, "filename", "attachment"))
            sent = _BrokenSentMessage(content=content, id=len(self.sent_messages) + 1)
            self.sent_messages.append(sent)
            return sent

    channel = _BrokenChannel(id=101)
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    sink = presenter.make_sink(channel)  # type: ignore[arg-type]

    monkeypatch.setattr(discord_services_module.discord, "HTTPException", RuntimeError)
    await sink(ShellStartEvent(command="pytest", cwd="/tmp"))
    await sink(ShellOutputEvent(chunk="line 1\n"))
    await sink(ShellDoneEvent(exit_code=1, elapsed_s=1.2))

    assert channel.sent[0] == "$ `pytest`"
    assert "line 1" in channel.sent[1]
    assert "exit 1 (1.2s)" in channel.sent[1]
