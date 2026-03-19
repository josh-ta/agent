from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from agent.config import settings
from agent.events import AgentEvent, bridge
from agent.memory.sqlite_store import SQLiteStore


class NullAsyncContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


@dataclass
class FakeAuthor:
    id: int = 1
    display_name: str = "Test User"
    bot: bool = False


@dataclass
class FakeMessageReference:
    message_id: int


@dataclass
class FakeSentMessage:
    content: str
    id: int = 1
    edits: list[str] = field(default_factory=list)

    async def edit(self, *, content: str) -> None:
        self.content = content
        self.edits.append(content)


@dataclass
class FakeHistoryMessage:
    author: FakeAuthor
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    reference: FakeMessageReference | None = None


@dataclass
class FakeDiscordAttachment:
    filename: str
    data: bytes
    content_type: str = "application/octet-stream"

    @property
    def size(self) -> int:
        return len(self.data)

    async def read(self) -> bytes:
        return self.data


@dataclass
class FakeChannel:
    id: int
    sent: list[str] = field(default_factory=list)
    sent_files: list[str] = field(default_factory=list)
    sent_messages: list[FakeSentMessage] = field(default_factory=list)
    history_messages: list[FakeHistoryMessage] = field(default_factory=list)

    def typing(self) -> NullAsyncContext:
        return NullAsyncContext()

    async def send(self, content: str = "", *, file=None) -> FakeSentMessage:
        self.sent.append(content)
        if file is not None:
            self.sent_files.append(getattr(file, "filename", "attachment"))
        sent = FakeSentMessage(content=content, id=len(self.sent_messages) + 1)
        self.sent_messages.append(sent)
        return sent

    async def history(
        self,
        limit: int = 20,
        after: FakeSentMessage | None = None,
        oldest_first: bool = False,
    ) -> AsyncIterator[FakeHistoryMessage]:
        messages = self.history_messages
        if after is not None:
            messages = messages[after.id:]
        selected = list(messages[-limit:])
        if not oldest_first:
            selected = list(reversed(selected))
        for item in selected:
            yield item


@dataclass
class FakeDiscordMessage:
    channel: FakeChannel
    content: str = ""
    author: FakeAuthor = field(default_factory=FakeAuthor)
    id: int = 1
    mentions: list[Any] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    reply_messages: list[FakeSentMessage] = field(default_factory=list)
    reactions: list[str] = field(default_factory=list)
    reference: FakeMessageReference | None = None
    attachments: list[FakeDiscordAttachment] = field(default_factory=list)

    async def reply(self, content: str, mention_author: bool = False) -> FakeSentMessage:
        self.replies.append(content)
        sent = FakeSentMessage(content=content, id=len(self.reply_messages) + 1)
        self.reply_messages.append(sent)
        return sent

    async def add_reaction(self, emoji: str) -> None:
        self.reactions.append(emoji)


class FakeClientUser(FakeAuthor):
    def mentioned_in(self, message: FakeDiscordMessage) -> bool:
        return any(getattr(user, "id", None) == self.id for user in message.mentions)


@dataclass
class FakeDiscordClient:
    user: Any
    channels: dict[int, FakeChannel]

    def get_channel(self, channel_id: int) -> FakeChannel | None:
        return self.channels.get(channel_id)


@pytest.fixture(autouse=True)
def clear_bridge_sinks() -> None:
    bridge._sinks.clear()


@pytest.fixture
def isolated_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    workspace = tmp_path / "workspace"
    identity = tmp_path / "identity"
    skills = tmp_path / "skills"
    workspace.mkdir()
    identity.mkdir()
    skills.mkdir()
    (identity / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")
    (identity / "IDENTITY.md").write_text("# Identity\n", encoding="utf-8")
    (skills / "example.md").write_text("# Example\nUseful skill.\n", encoding="utf-8")

    monkeypatch.setattr(settings, "workspace_path", workspace)
    monkeypatch.setattr(settings, "identity_path", identity)
    monkeypatch.setattr(settings, "skills_path", skills)
    monkeypatch.setattr(settings, "sqlite_path", workspace / "agent.db")

    return {"workspace": workspace, "identity": identity, "skills": skills}


@pytest.fixture
async def sqlite_store(isolated_paths: dict[str, Path]) -> AsyncIterator[SQLiteStore]:
    store = SQLiteStore(isolated_paths["workspace"] / "test.db")
    await store.init()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture
def event_collector() -> list[AgentEvent]:
    events: list[AgentEvent] = []

    async def sink(event: AgentEvent) -> None:
        events.append(event)

    bridge.register("test_collector", sink)
    return events


@pytest.fixture
def discord_channels(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeChannel]:
    private = FakeChannel(id=101)
    bus = FakeChannel(id=202)
    comms = FakeChannel(id=303)

    monkeypatch.setattr(settings, "discord_agent_channel_id", private.id)
    monkeypatch.setattr(settings, "discord_bus_channel_id", bus.id)
    monkeypatch.setattr(settings, "discord_comms_channel_id", comms.id)

    return {"private": private, "bus": bus, "comms": comms}


@pytest.fixture
def fake_client(discord_channels: dict[str, FakeChannel]) -> FakeDiscordClient:
    user = FakeClientUser(id=999, display_name="agent-1", bot=True)
    channels = {channel.id: channel for channel in discord_channels.values()}
    return FakeDiscordClient(user=user, channels=channels)


@pytest.fixture
def fake_message_factory() -> Any:
    def _factory(
        *,
        channel: FakeChannel,
        content: str,
        author: FakeAuthor | None = None,
        mentions: list[Any] | None = None,
        attachments: list[FakeDiscordAttachment] | None = None,
        message_id: int = 1,
    ) -> FakeDiscordMessage:
        return FakeDiscordMessage(
            channel=channel,
            content=content,
            author=author or FakeAuthor(),
            id=message_id,
            mentions=mentions or [],
            attachments=attachments or [],
        )

    return _factory
