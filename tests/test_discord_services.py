from __future__ import annotations

import asyncio
from types import SimpleNamespace

import discord
import pytest

import agent.communication.discord_services as discord_services_module
import agent.tools.discord_tools as discord_tools_module
from agent.communication.discord_services import DiscordEventPresenter, MAX_REPLY_LEN, MessageHandlingService, NativeCommand
from agent.communication.message_router import MessageKind, ParsedMessage
from agent.config import settings
from agent.events import (
    ProgressEvent,
    ShellDoneEvent,
    ShellOutputEvent,
    ShellStartEvent,
    TaskErrorEvent,
    TaskStartEvent,
    TaskWaitingEvent,
    TextTurnEndEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    bridge,
)
from agent.loop import TaskResult
from agent.task_waits import TaskWaitRegistry
from tests.conftest import FakeChannel, FakeDiscordAttachment, FakeSentMessage


class _BusyLoop:
    def __init__(self) -> None:
        self.has_pending_work = True
        self.queue = SimpleNamespace(qsize=lambda: 2)
        self.enqueued = None
        self.front_enqueued = None
        self.cleared: list[tuple[str | None, int | None, str]] = []
        self.cancelled: list[tuple[int | None, str]] = []
        self.wait_registry = TaskWaitRegistry()
        self.memory = None

    async def enqueue(self, task) -> None:
        self.enqueued = task
        response_future = getattr(task, "response_future", None)
        if response_future is not None and not response_future.done():
            response_future.set_result(
                TaskResult(task=task, output="finished", success=True, elapsed_ms=1.0)
            )

    async def enqueue_front(self, task) -> None:
        self.front_enqueued = task
        response_future = getattr(task, "response_future", None)
        if response_future is not None and not response_future.done():
            response_future.set_result(
                TaskResult(task=task, output="finished", success=True, elapsed_ms=1.0)
            )

    async def clear_queued_tasks(self, *, source=None, channel_id=None, reason="") -> list[object]:
        self.cleared.append((source, channel_id, reason))
        return [SimpleNamespace()] if channel_id is not None else []

    async def request_cancel_active_task(self, *, channel_id=None, reason: str) -> bool:
        self.cancelled.append((channel_id, reason))
        return True

    def describe_work(self, *, channel_id: int | None = None) -> str:
        return "Active: deploy\nQueued: 2"

    def build_resumed_task(self, *, suspended, answer: str, author: str, source: str, metadata_overrides=None):
        metadata = self.wait_registry.build_resumed_metadata(suspended, answer=answer, resumed_from=source)
        if metadata_overrides:
            metadata.update(metadata_overrides)
        return SimpleNamespace(content=suspended.content, source=source, author=author, metadata=metadata)


class _ReadyLoop:
    def __init__(self) -> None:
        self.has_pending_work = False
        self.queue = SimpleNamespace(qsize=lambda: 0)
        self.wait_registry = TaskWaitRegistry()
        self.memory = None
        self.enqueued = None
        self.front_enqueued = None
        self.cleared: list[tuple[str | None, int | None, str]] = []
        self.cancelled: list[tuple[int | None, str]] = []

    async def enqueue(self, task) -> None:
        self.enqueued = task
        assert task.response_future is not None
        task.response_future.set_result(
            TaskResult(task=task, output="finished", success=True, elapsed_ms=1.0)
        )

    async def enqueue_front(self, task) -> None:
        self.front_enqueued = task
        assert task.response_future is not None
        task.response_future.set_result(
            TaskResult(task=task, output="finished", success=True, elapsed_ms=1.0)
        )

    async def clear_queued_tasks(self, *, source=None, channel_id=None, reason="") -> list[object]:
        self.cleared.append((source, channel_id, reason))
        return []

    async def request_cancel_active_task(self, *, channel_id=None, reason: str) -> bool:
        self.cancelled.append((channel_id, reason))
        return True

    def describe_work(self, *, channel_id: int | None = None) -> str:
        return "No active or queued work."

    def build_resumed_task(self, *, suspended, answer: str, author: str, source: str, metadata_overrides=None):
        metadata = self.wait_registry.build_resumed_metadata(suspended, answer=answer, resumed_from=source)
        if metadata_overrides:
            metadata.update(metadata_overrides)
        return SimpleNamespace(content=suspended.content, source=source, author=author, metadata=metadata)


class _IntentRouter:
    def __init__(self, intent) -> None:
        self._intent = intent

    def classify_turn(self, **kwargs):
        return SimpleNamespace(intent=self._intent)


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
    assert "queued next" in message.replies[0]


@pytest.mark.asyncio
async def test_message_service_busy_queue_preserves_existing_task_id_and_persists_record(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MemorySpy:
        def __init__(self) -> None:
            self.created: list[dict] = []

        async def create_task_record(self, **kwargs) -> None:
            self.created.append(kwargs)

    loop = _BusyLoop()
    loop.memory = _MemorySpy()
    service = MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        service._session_router,
        "build_metadata",
        lambda **kwargs: {"task_id": "prebuilt-id", "session_id": "discord:101:1"},
    )
    message = fake_message_factory(channel=discord_channels["private"], content="please help")

    await service.handle_message(message)  # type: ignore[arg-type]

    assert loop.memory.created == [
        {
            "task_id": "prebuilt-id",
            "source": "discord",
            "author": "Test User",
            "content": "please help",
            "metadata": {"task_id": "prebuilt-id", "session_id": "discord:101:1"},
        }
    ]
    assert loop.front_enqueued.metadata["task_id"] == "prebuilt-id"


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

    assert "queued next" in message.replies[0]


@pytest.mark.asyncio
async def test_message_service_status_command_uses_loop_snapshot(
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
    message = fake_message_factory(channel=discord_channels["private"], content="/status")

    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="/status", channel_id=discord_channels["private"].id),
    )

    await service.handle_message(message)  # type: ignore[arg-type]

    assert message.reactions == ["👀"]
    assert message.replies == ["Active: deploy\nQueued: 2"]


def test_parse_native_command_rejects_blank_and_unknown_values() -> None:
    assert discord_services_module.parse_native_command("/   ") is None
    assert discord_services_module.parse_native_command("/wat") is None
    assert discord_services_module.parse_native_command("status") is None


@pytest.mark.asyncio
async def test_message_service_replace_command_cancels_and_queues_front(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _ReadyLoop()
    loop.has_pending_work = True
    service = MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    message = fake_message_factory(channel=discord_channels["private"], content="/replace restart the containers")

    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(
            MessageKind.TASK,
            content="/replace restart the containers",
            channel_id=discord_channels["private"].id,
        ),
    )

    await service.handle_message(message)  # type: ignore[arg-type]
    await asyncio.sleep(0)

    assert loop.front_enqueued is not None
    assert loop.front_enqueued.content == "restart the containers"
    assert loop.cancelled
    assert loop.cleared
    assert "Replacing the current task" in message.replies[0]
    assert "finished" in message.replies[-1]


@pytest.mark.asyncio
async def test_message_service_replaces_active_private_task_for_new_imperative_followup(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _ReadyLoop()
    service = MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    inject_q: asyncio.Queue[str] = asyncio.Queue()
    service._inject_queues[discord_channels["private"].id] = inject_q
    message = fake_message_factory(channel=discord_channels["private"], content="restart the containers")

    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="restart the containers", channel_id=discord_channels["private"].id),
    )

    await service.handle_message(message)  # type: ignore[arg-type]
    await asyncio.sleep(0)

    assert loop.front_enqueued is not None
    assert loop.front_enqueued.content == "restart the containers"
    assert inject_q.empty()
    assert loop.cancelled
    assert "Replacing the current task" in message.replies[0]


@pytest.mark.asyncio
async def test_message_service_enqueues_attachment_only_message(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(settings, "attachments_path", tmp_path)
    service = MessageHandlingService(
        agent_loop=_BusyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    message = fake_message_factory(
        channel=discord_channels["private"],
        content="",
        attachments=[
            FakeDiscordAttachment(
                filename="report.csv",
                data=b"name,value\nfoo,1\nbar,2\n",
                content_type="text/csv",
            )
        ],
    )

    await service.handle_message(message)  # type: ignore[arg-type]

    assert service._agent_loop.front_enqueued is not None
    assert service._agent_loop.front_enqueued.content == "Please inspect the attached file(s) and help with them."
    attachments = service._agent_loop.front_enqueued.metadata["attachments"]
    assert len(attachments) == 1
    assert attachments[0]["filename"] == "report.csv"
    assert "CSV preview" in attachments[0]["summary"]
    assert message.reactions == ["👀"]


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
async def test_is_answering_pending_question_true_and_false(
    fake_client,
    discord_channels,
    fake_message_factory,
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
        content="deploy it",
        channel_id=discord_channels["private"].id,
        message_id=1,
        metadata={"task_id": "task-1"},
        question="Which environment?",
        timeout_s=300,
        base_prompt="prompt",
        tier="smart",
    )
    loop.wait_registry.bind_prompt_message("task-1", 42)

    answering = fake_message_factory(channel=discord_channels["private"], content="staging")
    answering.reference = SimpleNamespace(message_id=42)
    normal = fake_message_factory(channel=discord_channels["bus"], content="new task")

    assert service._is_answering_pending_question(
        _parsed(MessageKind.TASK, content="staging", channel_id=discord_channels["private"].id),
        answering,  # type: ignore[arg-type]
    ) is True
    assert service._is_answering_pending_question(
        _parsed(MessageKind.TASK, content="new task", channel_id=discord_channels["bus"].id),
        normal,  # type: ignore[arg-type]
    ) is False


@pytest.mark.asyncio
async def test_message_service_resumes_suspended_task_updates_memory_side_effects(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MemorySpy:
        def __init__(self) -> None:
            self.marked: list[tuple[str, dict]] = []
            self.turns: list[dict] = []

        async def mark_task_queued(self, task_id: str, metadata: dict) -> None:
            self.marked.append((task_id, dict(metadata)))

        async def append_session_turn(self, **kwargs) -> None:
            self.turns.append(kwargs)

    loop = _BusyLoop()
    loop.memory = _MemorySpy()
    service = MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    loop.wait_registry.suspend(
        task_id="task-2",
        source="discord",
        author="Josh",
        content="deploy it",
        channel_id=discord_channels["private"].id,
        message_id=1,
        metadata={"task_id": "task-2", "session_id": "discord:101:1"},
        question="Which environment?",
        timeout_s=300,
        base_prompt="prompt",
        tier="smart",
    )
    loop.wait_registry.bind_prompt_message("task-2", 42)
    message = fake_message_factory(channel=discord_channels["private"], content="staging")
    message.reference = SimpleNamespace(message_id=42)

    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="staging", channel_id=discord_channels["private"].id),
    )

    await service.handle_message(message)  # type: ignore[arg-type]

    assert loop.memory.marked == [("task-2", service._agent_loop.enqueued.metadata)]
    assert loop.memory.turns[0]["session_id"] == "discord:101:1"
    assert loop.memory.turns[0]["content"] == "staging"
    assert loop.memory.turns[0]["turn_kind"] == "answer"


@pytest.mark.asyncio
async def test_message_service_ignores_bus_and_blank_content(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _ReadyLoop()
    service = MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
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
async def test_message_service_requests_disambiguation_for_multiple_waiting_tasks(
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
    for index, task_id in enumerate(("task-a", "task-b"), start=1):
        loop.wait_registry.suspend(
            task_id=task_id,
            source="discord",
            author="Josh",
            content=f"question {index}",
            channel_id=discord_channels["private"].id,
            message_id=index,
            metadata={"task_id": task_id},
            question="Which environment?",
            timeout_s=300,
            base_prompt="prompt",
            tier="smart",
        )
        loop.wait_registry.bind_prompt_message(task_id, 40 + index)
    message = fake_message_factory(channel=discord_channels["private"], content="answer")

    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="answer", channel_id=discord_channels["private"].id),
    )

    await service.handle_message(message)  # type: ignore[arg-type]

    assert message.reactions == ["👀"]
    assert "more than one suspended question" in message.replies[0]
    assert loop.enqueued is None


@pytest.mark.asyncio
async def test_message_service_injects_followup_and_replies_inline(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _ReadyLoop()
    service = MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
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
async def test_message_service_injects_pause_request_for_active_task(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _ReadyLoop()
    service = MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    inject_q: asyncio.Queue[str] = asyncio.Queue()
    service._inject_queues[discord_channels["private"].id] = inject_q
    service._session_router = _IntentRouter(discord_services_module.TurnIntent.CANCEL_OR_PAUSE)
    message = fake_message_factory(channel=discord_channels["private"], content="pause this")

    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="pause this", channel_id=discord_channels["private"].id),
    )

    await service.handle_message(message)  # type: ignore[arg-type]

    assert loop.cancelled
    assert message.reactions == ["👀"]
    assert "stop after the current step" in message.replies[0]


@pytest.mark.asyncio
async def test_message_service_native_command_helper_paths(
    fake_client,
    discord_channels,
    fake_message_factory,
    isolated_paths,
) -> None:
    loop = _ReadyLoop()
    service = MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    parsed = _parsed(MessageKind.TASK, channel_id=discord_channels["private"].id)

    help_message = fake_message_factory(channel=discord_channels["private"], content="/help")
    handled = await service._handle_native_command(
        message=help_message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="help"),
        task_content="",
        attachment_metadata={},
    )
    assert handled is True
    assert "/replace <task>" in help_message.replies[0]

    memory_message = fake_message_factory(channel=discord_channels["private"], content="/memory")
    handled = await service._handle_native_command(
        message=memory_message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="memory"),
        task_content="",
        attachment_metadata={},
    )
    assert handled is True
    assert "(no project memory recorded yet)" in memory_message.replies[0]

    remember_message = fake_message_factory(channel=discord_channels["private"], content="/remember app host is root@example")
    handled = await service._handle_native_command(
        message=remember_message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="remember", argument="app host is root@example"),
        task_content="app host is root@example",
        attachment_metadata={},
    )
    assert handled is True
    assert "Saved that to project memory" in remember_message.replies[0]

    memory_message = fake_message_factory(channel=discord_channels["private"], content="/memory")
    handled = await service._handle_native_command(
        message=memory_message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="memory"),
        task_content="",
        attachment_metadata={},
    )
    assert "root@example" in memory_message.replies[0]

    unremember_message = fake_message_factory(channel=discord_channels["private"], content="/unremember root@example")
    handled = await service._handle_native_command(
        message=unremember_message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="unremember", argument="root@example"),
        task_content="root@example",
        attachment_metadata={},
    )
    assert handled is True
    assert "Removed 1 matching project-memory entry" in unremember_message.replies[0]

    queue_message = fake_message_factory(channel=discord_channels["private"], content="/queue run tests")
    handled = await service._handle_native_command(
        message=queue_message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="queue", argument="run tests"),
        task_content="run tests",
        attachment_metadata={},
    )
    await asyncio.sleep(0)
    assert handled is True
    assert loop.enqueued is not None
    assert queue_message.replies[0] == "📝 Queued that task."
    assert queue_message.replies[-1] == "finished"

    usage_message = fake_message_factory(channel=discord_channels["private"], content="/queue")
    handled = await service._handle_native_command(
        message=usage_message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="queue"),
        task_content="",
        attachment_metadata={},
    )
    assert handled is True
    assert usage_message.replies == ["Usage: `/queue <task>`"]

    remember_usage = fake_message_factory(channel=discord_channels["private"], content="/remember")
    handled = await service._handle_native_command(
        message=remember_usage,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="remember"),
        task_content="",
        attachment_metadata={},
    )
    assert handled is True
    assert remember_usage.replies == ["Usage: `/remember <text>`"]

    public_message = fake_message_factory(channel=discord_channels["bus"], content="/help")
    public_parsed = _parsed(MessageKind.TASK, channel_id=discord_channels["bus"].id)
    handled = await service._handle_native_command(
        message=public_message,  # type: ignore[arg-type]
        parsed=public_parsed,
        command=NativeCommand(name="help"),
        task_content="",
        attachment_metadata={},
    )
    assert handled is False


@pytest.mark.asyncio
async def test_message_service_resume_and_cancel_command_branches(
    fake_client,
    discord_channels,
    fake_message_factory,
) -> None:
    class _LoopNoCancel(_ReadyLoop):
        async def clear_queued_tasks(self, *, source=None, channel_id=None, reason="") -> list[object]:
            self.cleared.append((source, channel_id, reason))
            return []

        async def request_cancel_active_task(self, *, channel_id=None, reason: str) -> bool:
            self.cancelled.append((channel_id, reason))
            return False

    loop = _LoopNoCancel()
    service = MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    parsed = _parsed(MessageKind.TASK, channel_id=discord_channels["private"].id)

    none_waiting = fake_message_factory(channel=discord_channels["private"], content="/resume")
    await service._handle_native_command(
        message=none_waiting,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="resume"),
        task_content="",
        attachment_metadata={},
    )
    assert "no suspended question" in none_waiting.replies[0]

    loop.wait_registry.suspend(
        task_id="task-one",
        source="discord",
        author="Josh",
        content="deploy",
        channel_id=discord_channels["private"].id,
        message_id=1,
        metadata={"task_id": "task-one"},
        question="Which environment?",
        timeout_s=300,
        base_prompt="prompt",
        tier="smart",
    )
    one_waiting = fake_message_factory(channel=discord_channels["private"], content="/resume")
    await service._handle_native_command(
        message=one_waiting,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="resume"),
        task_content="",
        attachment_metadata={},
    )
    assert one_waiting.replies == ["❓ Which environment?"]

    loop.wait_registry.suspend(
        task_id="task-two",
        source="discord",
        author="Josh",
        content="deploy",
        channel_id=discord_channels["private"].id,
        message_id=2,
        metadata={"task_id": "task-two"},
        question="Which cluster?",
        timeout_s=300,
        base_prompt="prompt",
        tier="smart",
    )
    many_waiting = fake_message_factory(channel=discord_channels["private"], content="/resume")
    await service._handle_native_command(
        message=many_waiting,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="resume"),
        task_content="",
        attachment_metadata={},
    )
    assert "more than one suspended question" in many_waiting.replies[0]

    cancel_message = fake_message_factory(channel=discord_channels["private"], content="/cancel")
    await service._handle_native_command(
        message=cancel_message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="cancel"),
        task_content="",
        attachment_metadata={},
    )
    assert "no active or queued task" in cancel_message.replies[0]

    forget_message = fake_message_factory(channel=discord_channels["private"], content="/forget")
    await service._handle_native_command(
        message=forget_message,  # type: ignore[arg-type]
        parsed=parsed,
        command=NativeCommand(name="forget"),
        task_content="",
        attachment_metadata={},
    )
    assert "no active or queued task" in forget_message.replies[0]


@pytest.mark.asyncio
async def test_message_service_helper_fallbacks_and_exception_paths(
    fake_client,
    discord_channels,
    fake_message_factory,
) -> None:
    loop = _ReadyLoop()
    service = MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )

    service._agent_loop = SimpleNamespace(wait_registry=TaskWaitRegistry(), queue=SimpleNamespace(qsize=lambda: 0), has_pending_work=False)
    assert await service._clear_queued_channel_tasks(channel_id=101, reason="x") == 0

    service._inject_queues[101] = asyncio.Queue()
    assert await service._request_cancel_active_task(channel_id=101, reason="stop") is True
    assert await service._inject_queues[101].get() == "stop"
    assert await service._request_cancel_active_task(channel_id=999, reason="stop") is False

    comms_message = fake_message_factory(channel=discord_channels["comms"], content="hello")
    await service._reply_safe(comms_message, "ignored")  # type: ignore[arg-type]
    assert comms_message.replies == []

    class _RaisingMessage:
        channel = SimpleNamespace(id=discord_channels["private"].id)

        async def reply(self, *_args, **_kwargs):
            response = SimpleNamespace(status=400, reason="bad")
            raise discord.HTTPException(response, "bad")

    await service._reply_safe(_RaisingMessage(), "ignored")  # type: ignore[arg-type]

    message = fake_message_factory(channel=discord_channels["private"], content="hello")
    future: asyncio.Future[TaskResult] = asyncio.get_running_loop().create_future()
    future.set_exception(RuntimeError("boom"))
    await service._wait_for_deferred_result(
        parsed=_parsed(MessageKind.TASK, channel_id=discord_channels["private"].id),
        message=message,  # type: ignore[arg-type]
        response_future=future,
    )
    assert message.replies == ["❌ boom"]


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
async def test_message_service_resume_and_pause_paths_skip_inline_reply_in_comms(
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
        content="deploy it",
        channel_id=discord_channels["comms"].id,
        message_id=1,
        metadata={"task_id": "task-1"},
        question="Which environment?",
        timeout_s=300,
        base_prompt="prompt",
        tier="smart",
    )
    loop.wait_registry.bind_prompt_message("task-1", 42)
    resume_message = fake_message_factory(channel=discord_channels["comms"], content="staging")
    resume_message.reference = SimpleNamespace(message_id=42)
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="staging", channel_id=discord_channels["comms"].id),
    )
    await service.handle_message(resume_message)  # type: ignore[arg-type]
    assert resume_message.replies == []

    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    inject_q: asyncio.Queue[str] = asyncio.Queue()
    service._inject_queues[discord_channels["comms"].id] = inject_q
    service._session_router = _IntentRouter(discord_services_module.TurnIntent.CANCEL_OR_PAUSE)
    pause_message = fake_message_factory(channel=discord_channels["comms"], content="pause this")
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="pause this", channel_id=discord_channels["comms"].id),
    )
    await service.handle_message(pause_message)  # type: ignore[arg-type]

    assert service._agent_loop.cancelled  # type: ignore[attr-defined]
    assert pause_message.replies == []


@pytest.mark.asyncio
async def test_message_service_swallow_inline_reply_failures_for_resume_pause_inject_and_busy(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ReplyFailMessage:
        def __init__(self, base) -> None:
            self._base = base
            self.__dict__.update(base.__dict__)

        async def reply(self, content: str, mention_author: bool = False):
            raise RuntimeError("reply failed")

        def __getattr__(self, name: str):
            return getattr(self._base, name)

    monkeypatch.setattr(discord_services_module.discord, "HTTPException", RuntimeError)

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
        content="deploy it",
        channel_id=discord_channels["private"].id,
        message_id=1,
        metadata={"task_id": "task-1"},
        question="Which environment?",
        timeout_s=300,
        base_prompt="prompt",
        tier="smart",
    )
    loop.wait_registry.bind_prompt_message("task-1", 42)
    resume_message = _ReplyFailMessage(fake_message_factory(channel=discord_channels["private"], content="staging"))
    resume_message.reference = SimpleNamespace(message_id=42)
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="staging", channel_id=discord_channels["private"].id),
    )
    await service.handle_message(resume_message)  # type: ignore[arg-type]
    assert loop.enqueued is not None

    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    inject_q: asyncio.Queue[str] = asyncio.Queue()
    service._inject_queues[discord_channels["private"].id] = inject_q
    service._session_router = _IntentRouter(discord_services_module.TurnIntent.CANCEL_OR_PAUSE)
    pause_message = _ReplyFailMessage(fake_message_factory(channel=discord_channels["private"], content="pause this"))
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="pause this", channel_id=discord_channels["private"].id),
    )
    await service.handle_message(pause_message)  # type: ignore[arg-type]
    assert service._agent_loop.cancelled  # type: ignore[attr-defined]

    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    inject_q = asyncio.Queue()
    service._inject_queues[discord_channels["private"].id] = inject_q
    followup_message = _ReplyFailMessage(fake_message_factory(channel=discord_channels["private"], content="new detail"))
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="new detail", channel_id=discord_channels["private"].id),
    )
    await service.handle_message(followup_message)  # type: ignore[arg-type]
    assert await inject_q.get() == "new detail"

    service = MessageHandlingService(
        agent_loop=_BusyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    busy_message = _ReplyFailMessage(fake_message_factory(channel=discord_channels["private"], content="please help"))
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="please help", channel_id=discord_channels["private"].id),
    )
    await service.handle_message(busy_message)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_message_service_swallow_ack_and_disambiguation_failures_and_skip_busy_inline_reply_in_comms(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ReactionFailMessage:
        def __init__(self, base) -> None:
            self._base = base
            self.__dict__.update(base.__dict__)

        async def add_reaction(self, emoji: str) -> None:
            raise RuntimeError("reaction failed")

        async def reply(self, content: str, mention_author: bool = False):
            raise RuntimeError("reply failed")

        def __getattr__(self, name: str):
            return getattr(self._base, name)

    monkeypatch.setattr(discord_services_module.discord, "HTTPException", RuntimeError)
    service = MessageHandlingService(
        agent_loop=_BusyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    message = _ReactionFailMessage(fake_message_factory(channel=discord_channels["private"], content="please help"))
    await service._acknowledge_message(message)  # type: ignore[arg-type]

    loop = _BusyLoop()
    service = MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    for index, task_id in enumerate(("task-a", "task-b"), start=1):
        loop.wait_registry.suspend(
            task_id=task_id,
            source="discord",
            author="Josh",
            content=f"question {index}",
            channel_id=discord_channels["private"].id,
            message_id=index,
            metadata={"task_id": task_id},
            question="Which environment?",
            timeout_s=300,
            base_prompt="prompt",
            tier="smart",
        )
        loop.wait_registry.bind_prompt_message(task_id, 40 + index)
    message = _ReactionFailMessage(fake_message_factory(channel=discord_channels["private"], content="answer"))
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="answer", channel_id=discord_channels["private"].id),
    )
    await service.handle_message(message)  # type: ignore[arg-type]
    assert loop.enqueued is None

    service = MessageHandlingService(
        agent_loop=_BusyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    message = fake_message_factory(channel=discord_channels["comms"], content="please help")
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="please help", channel_id=discord_channels["comms"].id),
    )
    await service.handle_message(message)  # type: ignore[arg-type]
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
async def test_message_service_reuses_sticky_private_session_between_turns(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _ReadyLoop()
    service = MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, channel_id=discord_channels["private"].id),
    )

    first = fake_message_factory(channel=discord_channels["private"], content="check the deploy logs", message_id=1)
    await service.handle_message(first)  # type: ignore[arg-type]
    first_session = loop.enqueued.metadata["session_id"]

    second = fake_message_factory(channel=discord_channels["private"], content="also verify the nginx config", message_id=2)
    await service.handle_message(second)  # type: ignore[arg-type]
    second_session = loop.enqueued.metadata["session_id"]

    assert first_session == second_session


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
async def test_message_service_waiting_path_persists_task_and_session_memory(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MemorySpy:
        def __init__(self) -> None:
            self.created: list[dict] = []
            self.sessions: list[dict] = []
            self.waiting: list[dict] = []

        async def create_task_record(self, **kwargs) -> None:
            self.created.append(kwargs)

        async def ensure_session(self, **kwargs) -> None:
            self.sessions.append(kwargs)

        async def mark_task_waiting(self, task_id: str, *, metadata: dict, question: str) -> None:
            self.waiting.append({"task_id": task_id, "metadata": dict(metadata), "question": question})

    class _WaitingLoop(_ReadyLoop):
        def __init__(self) -> None:
            super().__init__()
            self.memory = _MemorySpy()

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

    assert loop.memory.created[0]["task_id"] == "discord-101-1"
    assert loop.memory.sessions[0]["pending_task_id"] == "discord-101-1"
    assert loop.memory.waiting[0]["task_id"] == "task-wait"
    assert loop.memory.waiting[0]["question"] == "Which environment?"
    assert loop.memory.waiting[0]["metadata"]["wait_state"]["prompt_message_id"] == 1


@pytest.mark.asyncio
async def test_message_service_waiting_without_prompt_id_skips_binding_and_memory_wait_mark(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MemorySpy:
        def __init__(self) -> None:
            self.waiting_calls = 0

        async def create_task_record(self, **kwargs) -> None:
            return None

        async def ensure_session(self, **kwargs) -> None:
            return None

        async def mark_task_waiting(self, task_id: str, *, metadata: dict, question: str) -> None:
            self.waiting_calls += 1

    class _WaitingLoop(_ReadyLoop):
        def __init__(self) -> None:
            super().__init__()
            self.memory = _MemorySpy()

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
    monkeypatch.setattr(service, "_send_waiting_prompt", lambda *args, **kwargs: asyncio.sleep(0, result=None))
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="please help", channel_id=discord_channels["private"].id),
    )
    message = fake_message_factory(channel=discord_channels["private"], content="please help")

    await service.handle_message(message)  # type: ignore[arg-type]

    pending = loop.wait_registry.pending_for_channel(discord_channels["private"].id)
    assert pending[0].prompt_message_id is None
    assert loop.memory.waiting_calls == 0
    assert message.reactions == ["👀", "⏸️"]


@pytest.mark.asyncio
async def test_background_waiting_event_prompts_channel_and_persists_prompt_id(
    fake_client,
    discord_channels,
) -> None:
    class _MemorySpy:
        def __init__(self) -> None:
            self.waiting: list[dict] = []

        async def mark_task_waiting(self, task_id: str, *, metadata: dict, question: str) -> None:
            self.waiting.append({"task_id": task_id, "metadata": dict(metadata), "question": question})

    loop = _ReadyLoop()
    loop.memory = _MemorySpy()
    MessageHandlingService(
        agent_loop=loop,  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    loop.wait_registry.suspend(
        task_id="task-wait",
        source="discord",
        author="Josh",
        content="deploy it",
        channel_id=discord_channels["private"].id,
        message_id=7,
        metadata={
            "task_id": "task-wait",
            "wait_state": {
                "question": "Which environment?",
                "timeout_s": 90,
                "channel_id": discord_channels["private"].id,
                "message_id": 7,
                "prompt_message_id": None,
            },
        },
        question="Which environment?",
        timeout_s=90,
        base_prompt="",
        tier="smart",
    )

    await bridge.emit(
        TaskWaitingEvent(
            question="Which environment?",
            timeout_s=90,
            task_id="task-wait",
            source="discord",
            channel_id=discord_channels["private"].id,
            deliver_inline_reply=False,
        )
    )

    pending = loop.wait_registry.pending_for_channel(discord_channels["private"].id)
    assert discord_channels["private"].sent == ["❓ Which environment?"]
    assert pending[0].prompt_message_id == 1
    assert loop.memory.waiting == [
        {
            "task_id": "task-wait",
            "metadata": {
                "task_id": "task-wait",
                "wait_state": {
                    "question": "Which environment?",
                    "timeout_s": 90,
                    "channel_id": discord_channels["private"].id,
                    "message_id": 7,
                    "prompt_message_id": 1,
                },
            },
            "question": "Which environment?",
        }
    ]


@pytest.mark.asyncio
async def test_send_waiting_prompt_falls_back_to_channel_send_and_handles_total_failure(
    fake_client,
    discord_channels,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )

    class _FallbackMessage:
        def __init__(self, channel) -> None:
            self.channel = channel

        async def reply(self, content: str, mention_author: bool = False):
            raise RuntimeError("reply failed")

    class _BrokenChannel(FakeChannel):
        async def send(self, content: str = "", *, file=None):
            raise RuntimeError("send failed")

    monkeypatch.setattr(discord_services_module.discord, "HTTPException", RuntimeError)

    fallback = await service._send_waiting_prompt(
        _FallbackMessage(discord_channels["private"]),  # type: ignore[arg-type]
        "Which environment?",
    )
    failed = await service._send_waiting_prompt(
        _FallbackMessage(_BrokenChannel(id=999)),  # type: ignore[arg-type]
        "Which environment?",
    )

    assert fallback == 1
    assert failed is None


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

        def build_resumed_task(self, *, suspended, answer: str, author: str, source: str, metadata_overrides=None):
            metadata = self.wait_registry.build_resumed_metadata(suspended, answer=answer, resumed_from=source)
            if metadata_overrides:
                metadata.update(metadata_overrides)
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
async def test_message_service_sends_attachment_only_results_through_send_reply(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        TaskResult(
            task=SimpleNamespace(),
            output="",
            success=True,
            elapsed_ms=1.0,
            attachments=[discord_tools_module.DiscordAttachment(filename="browser-screenshot-1.png", data=b"png")],
        )
    ]
    sent: list[tuple[str, int]] = []

    class _Loop:
        has_pending_work = False
        queue = SimpleNamespace(qsize=lambda: 0)
        wait_registry = TaskWaitRegistry()
        memory = None

        async def enqueue(self, task) -> None:
            task.response_future.set_result(results.pop(0))

        def build_resumed_task(self, *, suspended, answer: str, author: str, source: str, metadata_overrides=None):
            metadata = self.wait_registry.build_resumed_metadata(suspended, answer=answer, resumed_from=source)
            if metadata_overrides:
                metadata.update(metadata_overrides)
            return SimpleNamespace(content=suspended.content, source=source, author=author, metadata=metadata)

    service = MessageHandlingService(
        agent_loop=_Loop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )

    async def fake_send_reply(parsed, output: str, original_message, attachments=None) -> bool:
        sent.append((output, len(attachments or [])))
        return True

    monkeypatch.setattr(service, "send_reply", fake_send_reply)
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="please help", channel_id=discord_channels["private"].id),
    )
    message = fake_message_factory(channel=discord_channels["private"], content="please help")

    await service.handle_message(message)  # type: ignore[arg-type]

    assert sent == [("", 1)]


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
async def test_message_service_send_reply_handles_a2a_missing_sender_or_missing_comms_channel(
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

    no_sender = _parsed(MessageKind.A2A, channel_id=discord_channels["comms"].id, payload={})
    delivered = await service.send_reply(no_sender, "result payload", message)  # type: ignore[arg-type]
    assert delivered is True

    with_sender = _parsed(
        MessageKind.A2A,
        channel_id=discord_channels["comms"].id,
        payload={"from": "peer-1"},
    )
    fake_client.channels.pop(discord_channels["comms"].id)
    monkeypatch.setattr(settings, "discord_comms_channel_id", discord_channels["comms"].id)
    delivered = await service.send_reply(with_sender, "result payload", message)  # type: ignore[arg-type]
    assert delivered is False
    assert discord_channels["bus"].sent


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
async def test_message_service_send_reply_for_bus_handles_attachments_and_missing_private_channel(
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

    delivered = await service.send_reply(
        parsed,
        "bus output",
        message,  # type: ignore[arg-type]
        attachments=[discord_tools_module.DiscordAttachment(filename="browser-screenshot-1.png", data=b"png")],
    )
    assert delivered is True
    assert discord_channels["private"].sent_files == ["browser-screenshot-1.png"]

    fake_client.channels.pop(discord_channels["private"].id)
    delivered = await service.send_reply(parsed, "bus output", message)  # type: ignore[arg-type]
    assert delivered is True


@pytest.mark.asyncio
async def test_message_service_send_reply_uses_presenter_for_non_private_channel(
    fake_client,
    discord_channels,
    fake_message_factory,
) -> None:
    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=DiscordEventPresenter(fake_client),  # type: ignore[arg-type]
    )
    parsed = _parsed(MessageKind.TASK, channel_id=999)
    message = fake_message_factory(channel=FakeChannel(id=999), content="hello")

    delivered = await service.send_reply(parsed, "normal output", message)  # type: ignore[arg-type]

    assert delivered is True
    assert message.replies == []
    assert discord_channels["private"].sent[0] == "normal output"


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
async def test_message_service_send_reply_returns_false_on_presenter_failure(
    fake_client,
    discord_channels,
    fake_message_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenPresenter(DiscordEventPresenter):
        async def send_chunked(self, channel, text: str) -> None:
            raise RuntimeError("send failed")

    service = MessageHandlingService(
        agent_loop=_ReadyLoop(),  # type: ignore[arg-type]
        client=fake_client,  # type: ignore[arg-type]
        presenter=_BrokenPresenter(fake_client),  # type: ignore[arg-type]
    )
    parsed = _parsed(MessageKind.TASK, channel_id=999)
    message = fake_message_factory(channel=FakeChannel(id=999), content="hello")
    monkeypatch.setattr(discord_services_module.discord, "HTTPException", RuntimeError)

    delivered = await service.send_reply(parsed, "normal output", message)  # type: ignore[arg-type]

    assert delivered is False


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

    fake_client.channels.pop(discord_channels["bus"].id, None)
    await service.post_bus_status("hello again")


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
async def test_discord_event_presenter_shell_success_paths_and_noop_cases(fake_client, discord_channels) -> None:
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]

    direct_channel = FakeChannel(id=101)
    sink = presenter.make_sink(direct_channel)  # type: ignore[arg-type]
    await sink(ShellOutputEvent(chunk="line 1\n"))
    await sink(ShellDoneEvent(exit_code=0, elapsed_s=1.2))
    assert "line 1" in direct_channel.sent[0]
    assert "exit 0" not in direct_channel.sent[0]

    started_channel = FakeChannel(id=101)
    sink = presenter.make_sink(started_channel)  # type: ignore[arg-type]
    await sink(ShellStartEvent(command="pytest", cwd="/tmp"))
    await sink(ShellDoneEvent(exit_code=0, elapsed_s=1.2))
    assert started_channel.sent == ["$ `pytest`"]
    assert started_channel.sent_messages[0].edits == []


@pytest.mark.asyncio
async def test_discord_event_presenter_status_only_falls_back_when_edit_fails(fake_client, monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenSentMessage(FakeSentMessage):
        async def edit(self, *, content: str) -> None:
            raise RuntimeError("edit failed")

    class _BrokenChannel(FakeChannel):
        async def send(self, content: str = "", *, file=None):
            self.sent.append(content)
            sent = _BrokenSentMessage(content=content, id=len(self.sent_messages) + 1)
            self.sent_messages.append(sent)
            return sent

    channel = _BrokenChannel(id=101)
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    sink = presenter.make_sink(channel)  # type: ignore[arg-type]
    monkeypatch.setattr(discord_services_module.discord, "HTTPException", RuntimeError)

    await sink(ShellStartEvent(command="pytest", cwd="/tmp"))
    await sink(ShellDoneEvent(exit_code=1, elapsed_s=1.2))

    assert channel.sent == ["$ `pytest`", "exit 1 (1.2s)"]


@pytest.mark.asyncio
async def test_discord_event_presenter_handles_shell_failure_without_output_or_start_message(
    fake_client,
    discord_channels,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    sink = presenter.make_sink(discord_channels["private"])  # type: ignore[arg-type]

    await sink(ShellStartEvent(command="pytest", cwd="/tmp"))
    await sink(ShellDoneEvent(exit_code=1, elapsed_s=1.2))

    sent_message = discord_channels["private"].sent_messages[0]
    assert "exit 1 (1.2s)" in sent_message.edits[0]

    class _BrokenStartChannel(FakeChannel):
        async def send(self, content: str = "", *, file=None):
            if content.startswith("$ `"):
                raise RuntimeError("send failed")
            return await super().send(content, file=file)

    channel = _BrokenStartChannel(id=101)
    sink = presenter.make_sink(channel)  # type: ignore[arg-type]
    monkeypatch.setattr(discord_services_module.discord, "HTTPException", RuntimeError)
    await sink(ShellStartEvent(command="pytest", cwd="/tmp"))
    await sink(ShellDoneEvent(exit_code=1, elapsed_s=1.2))

    assert channel.sent == ["exit 1 (1.2s)"]


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
        "🟢 Working on it.",
        "🧠 *first \\*idea\\**",
        "💭 working",
        "🔧 `run_shell(command=pytest)`",
        "still working",
        "❌ boom",
    ]


@pytest.mark.asyncio
async def test_discord_event_presenter_ignores_empty_and_unhandled_events(fake_client) -> None:
    channel = FakeChannel(id=101)
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    sink = presenter.make_sink(channel)  # type: ignore[arg-type]

    await sink(ThinkingEndEvent(text=""))
    await sink(ThinkingEndEvent(text="   "))
    await sink(TextTurnEndEvent(text="done", is_final=True))
    await sink(ProgressEvent(message=""))
    await sink(SimpleNamespace())

    assert channel.sent == []


@pytest.mark.asyncio
async def test_discord_event_presenter_truncates_long_start_and_formats_non_dict_tool_args(fake_client) -> None:
    channel = FakeChannel(id=101)
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    sink = presenter.make_sink(channel)  # type: ignore[arg-type]

    await sink(TaskStartEvent(content="x" * 200, tier="smart"))
    await sink(ToolCallStartEvent(tool_name="run_shell", call_id="1", args="y" * 300))

    assert channel.sent[0] == "🟢 Working on it."
    assert channel.sent[1].startswith("🔧 `run_shell(")
    assert len(channel.sent[1]) < 240


@pytest.mark.asyncio
async def test_discord_event_presenter_send_helpers_swallow_http_exception(
    fake_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    presenter = DiscordEventPresenter(fake_client)  # type: ignore[arg-type]
    channel = FakeChannel(id=101)

    async def fail_send_text(*args, **kwargs):
        raise RuntimeError("send failed")

    async def fail_send_attachments(*args, **kwargs):
        raise RuntimeError("attach failed")

    monkeypatch.setattr(discord_services_module.discord, "HTTPException", RuntimeError)
    monkeypatch.setattr(discord_tools_module, "send_text", fail_send_text)
    monkeypatch.setattr(discord_tools_module, "send_attachments", fail_send_attachments)

    await presenter.send_chunked(channel, "hello")  # type: ignore[arg-type]
    await presenter.send_attachments(
        channel,  # type: ignore[arg-type]
        [discord_tools_module.DiscordAttachment(filename="file.txt", data=b"data")],
    )


@pytest.mark.asyncio
async def test_message_service_handles_missing_private_channel_when_running_task(
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
    fake_client.channels.pop(discord_channels["private"].id)
    monkeypatch.setattr(
        discord_services_module,
        "classify",
        lambda *_args: _parsed(MessageKind.TASK, content="please help", channel_id=discord_channels["private"].id),
    )
    message = fake_message_factory(channel=discord_channels["private"], content="please help")

    await service.handle_message(message)  # type: ignore[arg-type]

    assert message.reactions == ["👀", "🏁"]


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
