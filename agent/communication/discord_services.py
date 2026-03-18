"""
Discord-facing services used by `discord_bot`.

The gateway stays responsible for discord.py lifecycle, while these classes hold
message policy and event rendering.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import discord
import structlog

from agent.communication.message_router import (
    MessageKind,
    ParsedMessage,
    a2a_to_task_content,
    classify,
)
from agent.config import settings
from agent.events import (
    AgentEvent,
    ProgressEvent,
    ShellDoneEvent,
    ShellOutputEvent,
    ShellStartEvent,
    TaskErrorEvent,
    TaskStartEvent,
    TextTurnEndEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    bridge,
)
from agent.loop import Task

if TYPE_CHECKING:
    from agent.loop import AgentLoop

log = structlog.get_logger()

MAX_REPLY_LEN = 1990
SILENT_TOOLS = frozenset(
    {
        "read_file",
        "list_dir",
        "memory_save",
        "lesson_search",
        "task_resume",
        "task_journal_clear",
        "task_note",
        "memory_search",
        "lessons_recent",
        "lesson_save",
        "read_channel",
        "read_discord",
        "identity_read",
        "skill_list",
        "skill_read",
        "db_stats",
    }
)


def escape_md_italics(text: str) -> str:
    return text.replace("*", "\\*")


def escape_codeblock(text: str) -> str:
    return text.replace("```", "`` `")


def format_args(args: object) -> str:
    if isinstance(args, dict):
        parts = []
        for key, value in args.items():
            text = str(value)
            parts.append(f"{key}={text[:60] + '…' if len(text) > 60 else text}")
        return ", ".join(parts)[:200]
    return str(args)[:200]


def allows_inline_reply(channel_id: int) -> bool:
    return channel_id != settings.discord_comms_channel_id


class DiscordEventPresenter:
    def __init__(self, client: discord.Client) -> None:
        self._client = client

    async def send_chunked(self, channel: discord.abc.Messageable, text: str) -> None:
        chunks = [text[i:i + MAX_REPLY_LEN] for i in range(0, len(text), MAX_REPLY_LEN)]
        for chunk in chunks:
            try:
                await channel.send(chunk)
            except discord.HTTPException as exc:
                log.warning("send_chunked_failed", error=str(exc))

    def make_sink(self, channel: discord.abc.Messageable):  # type: ignore[return]
        shell_lines: list[str] = []
        shell_msg: discord.Message | None = None

        async def _send(text: str) -> None:
            await self.send_chunked(channel, text)

        async def sink(event: AgentEvent) -> None:
            nonlocal shell_lines, shell_msg

            if isinstance(event, TaskStartEvent):
                return

            if isinstance(event, ThinkingEndEvent):
                if event.text:
                    for i in range(0, len(event.text), 1800):
                        chunk = event.text[i:i + 1800].strip()
                        if chunk:
                            await _send(f"🧠 *{escape_md_italics(chunk)}*")
                return

            if isinstance(event, TextTurnEndEvent):
                if not event.is_final and event.text:
                    await _send(f"💭 {event.text[:1900]}")
                return

            if isinstance(event, ToolCallStartEvent):
                if event.tool_name not in SILENT_TOOLS:
                    await _send(f"🔧 `{event.tool_name}({format_args(event.args)})`")
                return

            if isinstance(event, ShellStartEvent):
                shell_lines = []
                shell_msg = None
                try:
                    shell_msg = await channel.send(f"$ `{event.command[:200]}`")  # type: ignore[union-attr]
                except discord.HTTPException:
                    pass
                return

            if isinstance(event, ShellOutputEvent):
                shell_lines.append(event.chunk)
                return

            if isinstance(event, ShellDoneEvent):
                output = "".join(shell_lines).strip()
                shell_lines = []
                failed = event.exit_code != 0
                status = f"exit {event.exit_code} ({event.elapsed_s:.1f}s)" if failed else None
                if output:
                    body = f"```\n{escape_codeblock(output[-1400:])}\n```"
                    if status:
                        body += f"\n{status}"
                    try:
                        if shell_msg is not None:
                            await shell_msg.edit(content=f"{shell_msg.content}\n{body}")
                        else:
                            await _send(body)
                    except discord.HTTPException:
                        await _send(body)
                elif status:
                    try:
                        if shell_msg is not None:
                            await shell_msg.edit(content=f"{shell_msg.content} → {status}")
                        else:
                            await _send(status)
                    except discord.HTTPException:
                        await _send(status)
                shell_msg = None
                return

            if isinstance(event, ProgressEvent):
                if event.message:
                    await _send(event.message)
                return

            if isinstance(event, TaskErrorEvent):
                await _send(f"❌ {event.error[:400]}")

        return sink


class MessageHandlingService:
    def __init__(
        self,
        *,
        agent_loop: "AgentLoop",
        client: discord.Client,
        presenter: DiscordEventPresenter,
        event_bridge: Any = bridge,
    ) -> None:
        self._agent_loop = agent_loop
        self._client = client
        self._presenter = presenter
        self._bridge = event_bridge
        self._inject_queues: dict[int, asyncio.Queue[str]] = {}

    async def _acknowledge_message(self, message: discord.Message, emoji: str = "✅") -> None:
        try:
            await message.add_reaction(emoji)
        except discord.HTTPException:
            pass

    async def handle_message(self, message: discord.Message) -> None:
        assert self._client.user
        parsed = classify(message, self._client.user)

        if parsed.kind in {MessageKind.IGNORE, MessageKind.BUS}:
            return

        task_content = (
            a2a_to_task_content(parsed.a2a_payload)
            if parsed.kind == MessageKind.A2A and parsed.a2a_payload
            else parsed.content
        )
        if not task_content.strip():
            return

        private_channel = self._client.get_channel(settings.discord_agent_channel_id)
        inject_q = self._inject_queues.get(parsed.channel_id)
        if inject_q is not None:
            await inject_q.put(task_content)
            await self._acknowledge_message(message)
            if allows_inline_reply(parsed.channel_id):
                try:
                    await message.reply(
                        "💬 Got it — I'll fold that into what I'm working on.",
                        mention_author=False,
                    )
                except discord.HTTPException:
                    pass
            return

        if self._agent_loop.has_pending_work:
            await self._agent_loop.enqueue(
                Task(
                    content=task_content,
                    source="discord",
                    author=parsed.author,
                    channel_id=parsed.channel_id,
                    message_id=parsed.message_id,
                    inject_queue=asyncio.Queue(),
                )
            )
            await self._acknowledge_message(message)
            if allows_inline_reply(parsed.channel_id):
                position = f"#{self._agent_loop.queue.qsize()}" if self._agent_loop.queue.qsize() > 1 else "next"
                try:
                    await message.reply(
                        f"⏸️ I'm still working on the previous task — queued yours ({position} up).",
                        mention_author=False,
                    )
                except discord.HTTPException:
                    pass
            return

        inject_q = asyncio.Queue()
        self._inject_queues[parsed.channel_id] = inject_q
        response_future = asyncio.get_running_loop().create_future()
        task = Task(
            content=task_content,
            source="discord",
            author=parsed.author,
            channel_id=parsed.channel_id,
            message_id=parsed.message_id,
            inject_queue=inject_q,
            response_future=response_future,
        )

        sink_tag = f"discord_{parsed.channel_id}_{id(task)}"
        if private_channel is not None:
            self._bridge.register(sink_tag, self._presenter.make_sink(private_channel))  # type: ignore[arg-type]

        typing_ctx = private_channel.typing() if private_channel is not None else message.channel.typing()  # type: ignore[union-attr]
        try:
            async with typing_ctx:
                await self._agent_loop.enqueue(task)
                await self._acknowledge_message(message)
                result = await response_future
        finally:
            self._inject_queues.pop(parsed.channel_id, None)
            self._bridge.unregister(sink_tag)

        if result.discord_replied or not result.output:
            return

        await self.send_reply(parsed, result.output, message)

    async def send_reply(
        self,
        parsed: ParsedMessage,
        output: str,
        original_message: discord.Message,
    ) -> None:
        is_a2a = parsed.kind == MessageKind.A2A and parsed.a2a_payload is not None
        is_bus = parsed.channel_id == settings.discord_bus_channel_id

        if is_a2a:
            from_agent = parsed.a2a_payload.get("from", "")
            if from_agent and settings.discord_comms_channel_id:
                comms = self._client.get_channel(settings.discord_comms_channel_id)
                if comms is not None:
                    try:
                        await comms.send(
                            json.dumps(
                                {
                                    "from": settings.agent_name,
                                    "to": from_agent,
                                    "task": "result",
                                    "payload": output[:1800],
                                }
                            )
                        )  # type: ignore[union-attr]
                    except discord.HTTPException as exc:
                        log.warning("a2a_reply_failed", error=str(exc))
            await self.post_bus_status(f"**{settings.agent_name}** completed task from {from_agent or 'unknown'}.")
            return

        if is_bus:
            await self.post_bus_status(f"**{settings.agent_name}**: {output[:300]}")
            private_channel = self._client.get_channel(settings.discord_agent_channel_id)
            if private_channel is not None:
                await self._presenter.send_chunked(private_channel, output)  # type: ignore[arg-type]
            return

        private_channel = self._client.get_channel(settings.discord_agent_channel_id)
        target = private_channel or original_message.channel
        try:
            chunks = [output[i:i + MAX_REPLY_LEN] for i in range(0, len(output), MAX_REPLY_LEN)]
            if original_message.channel.id == settings.discord_agent_channel_id and chunks:
                await original_message.reply(chunks[0], mention_author=False)
                for chunk in chunks[1:]:
                    await target.send(chunk)  # type: ignore[union-attr]
            else:
                await self._presenter.send_chunked(target, output)  # type: ignore[arg-type]
        except discord.HTTPException as exc:
            log.error("discord_send_failed", error=str(exc))

    async def post_bus_status(self, message: str) -> None:
        if not settings.discord_bus_channel_id:
            return
        bus = self._client.get_channel(settings.discord_bus_channel_id)
        if bus is None:
            return
        try:
            await bus.send(message[:MAX_REPLY_LEN])  # type: ignore[union-attr]
        except discord.HTTPException as exc:
            log.warning("bus_status_failed", error=str(exc))
