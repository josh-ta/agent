"""
Discord bot: receives messages, routes them to the agent loop, sends replies.

Uses discord.py with the following intents:
  - message_content  (required to read message text)
  - guilds
  - guild_messages

Event bridge integration
------------------------
Instead of using Task.progress_callback, this bot registers a per-task sink on
the module-level EventBridge (agent.events.bridge). The sink maps typed AgentEvent
objects to Discord messages, giving the bot a clean, exhaustive view of everything
the agent is doing — thinking, tool calls, shell output, checkpoints, errors.

The sink is registered with a unique tag before the task starts and unregistered
in the finally block so it never leaks across tasks.
"""

from __future__ import annotations

import asyncio
import traceback

import discord
import structlog

from agent.communication.message_router import MessageKind, a2a_to_task_content, classify
from agent.config import settings
from agent.events import (
    bridge,
    AgentEvent,
    TextTurnEndEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolResultEvent,
    ShellStartEvent,
    ShellOutputEvent,
    ShellDoneEvent,
    TaskStartEvent,
    TaskDoneEvent,
    TaskErrorEvent,
    ProgressEvent,
)
from agent.loop import AgentLoop, Task
from agent.tools.discord_tools import set_discord_client

log = structlog.get_logger()

MAX_REPLY_LEN = 1990  # Discord limit minus a small buffer

# Tool names whose args/results are too noisy to show in Discord
_SILENT_TOOLS = frozenset({
    "read_file", "list_dir", "memory_save", "lesson_search",
    "task_resume", "task_journal_clear",
})

# How many shell output chunks to buffer before flushing to Discord.
# We don't want one message per line — batch them into blocks.
_SHELL_OUTPUT_BATCH = 20


def _fmt_args(args: object) -> str:
    """Format tool args for display — truncate to keep Discord messages readable."""
    if isinstance(args, dict):
        # Show key=value pairs, skip large values
        parts = []
        for k, v in args.items():
            vs = str(v)
            parts.append(f"{k}={vs[:60] + '…' if len(vs) > 60 else vs}")
        return ", ".join(parts)[:200]
    return str(args)[:200]


class DiscordBot:
    """Wraps a discord.py Client and connects it to the AgentLoop."""

    def __init__(self, loop: AgentLoop) -> None:
        self._agent_loop = loop
        # Per-channel inject queues: while a task is running, new messages for
        # the same channel are pushed here instead of starting a new task.
        self._inject_queues: dict[int, asyncio.Queue[str]] = {}
        # The channel_id of the task currently running (0 if none)
        self._active_channel: int = 0

        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.guild_messages = True

        self._client = discord.Client(intents=intents)
        self._setup_events()

        # Give tools access to the client for discord_send/discord_read
        set_discord_client(self._client)

    def _setup_events(self) -> None:
        client = self._client

        @client.event
        async def on_ready() -> None:
            log.info(
                "discord_ready",
                user=str(client.user),
                agent=settings.agent_name,
                guilds=[g.name for g in client.guilds],
            )
            await self._announce_online()

        @client.event
        async def on_message(message: discord.Message) -> None:
            if client.user is None:
                return
            await self._handle_message(message)

        @client.event
        async def on_disconnect() -> None:
            log.warning("discord_disconnected")

        @client.event
        async def on_resumed() -> None:
            log.info("discord_resumed")

    async def start_bot(self) -> None:
        """Connect to Discord and run the bot indefinitely."""
        if not settings.discord_bot_token:
            log.error("no_discord_token")
            return

        try:
            await self._client.start(settings.discord_bot_token)
        except discord.LoginFailure:
            log.error("discord_login_failed")
        except asyncio.CancelledError:
            await self._client.close()
        except Exception:
            log.error("discord_fatal", exc=traceback.format_exc())
            await self._client.close()

    async def _handle_message(self, message: discord.Message) -> None:
        """Route a Discord message to the agent loop and reply with the result."""
        assert self._client.user

        parsed = classify(message, self._client.user)

        if parsed.kind == MessageKind.IGNORE:
            return

        if parsed.kind == MessageKind.BUS:
            return

        if parsed.kind == MessageKind.A2A and parsed.a2a_payload:
            task_content = a2a_to_task_content(parsed.a2a_payload)
        else:
            task_content = parsed.content

        if not task_content.strip():
            return

        channel = message.channel

        if self._agent_loop.is_busy:
            if self._active_channel == parsed.channel_id:
                inject_q = self._inject_queues.get(parsed.channel_id)
                if inject_q is not None:
                    await inject_q.put(task_content)
                    try:
                        await message.reply(
                            "💬 Got it — I'll fold that into what I'm working on.",
                            mention_author=False,
                        )
                    except discord.HTTPException:
                        pass
                    return

            await self._agent_loop.enqueue(Task(
                content=task_content,
                source="discord",
                author=parsed.author,
                channel_id=parsed.channel_id,
                message_id=parsed.message_id,
                inject_queue=asyncio.Queue(),
            ))
            queue_depth = self._agent_loop.queue.qsize()
            position = f"#{queue_depth}" if queue_depth > 1 else "next"
            try:
                await message.reply(
                    f"⏸️ I'm still working on the previous task — queued yours ({position} up).",
                    mention_author=False,
                )
            except discord.HTTPException:
                pass
            return

        # Agent is free — build inject queue and run directly
        inject_q: asyncio.Queue[str] = asyncio.Queue()
        self._inject_queues[parsed.channel_id] = inject_q
        self._active_channel = parsed.channel_id

        task = Task(
            content=task_content,
            source="discord",
            author=parsed.author,
            channel_id=parsed.channel_id,
            message_id=parsed.message_id,
            inject_queue=inject_q,
        )

        # Register a per-task Discord sink on the bridge
        sink_tag = f"discord_{parsed.channel_id}_{id(task)}"
        discord_sink = self._make_discord_sink(channel)  # type: ignore[arg-type]
        bridge.register(sink_tag, discord_sink)

        self._agent_loop.is_busy = True
        try:
            async with channel.typing():  # type: ignore[union-attr]
                result = await self._agent_loop._process(task)
        finally:
            self._agent_loop.is_busy = False
            self._active_channel = 0
            self._inject_queues.pop(parsed.channel_id, None)
            bridge.unregister(sink_tag)

        # Don't send a second reply if the agent already called send_discord
        if result.discord_replied:
            return

        reply = result.output
        if not reply:
            return

        try:
            chunks = [reply[i:i+MAX_REPLY_LEN] for i in range(0, len(reply), MAX_REPLY_LEN)]
            await message.reply(chunks[0], mention_author=False)
            for chunk in chunks[1:]:
                await channel.send(chunk)  # type: ignore[union-attr]
        except discord.HTTPException as exc:
            log.error("discord_send_failed", error=str(exc))

    def _make_discord_sink(
        self,
        channel: discord.abc.Messageable,
    ):
        """
        Return an async sink function that maps typed AgentEvents to Discord messages.

        Per-event mapping:
          TaskStartEvent      → 🔍 Working on: *<content>*
          ThinkingEndEvent    → 🧠 *<thinking>*
          TextTurnEndEvent    → 💭 <text>  (intermediate only; final is sent as reply)
          ToolCallStartEvent  → 🔧 `tool_name(args)`  (skip noisy read/memory tools)
          ShellStartEvent     → $ `command`
          ShellOutputEvent    → batched into code blocks (flushed every N lines or on done)
          ShellDoneEvent      → exit N (X.Xs)
          ProgressEvent       → message as-is  (ticker, rate-limit, inject acks, etc.)
          TaskErrorEvent      → ❌ error
          TaskDoneEvent       → (silent — the final reply is sent by _handle_message)
          ToolResultEvent     → (silent — too verbose; surface via ThinkingEndEvent)
          TextDeltaEvent      → (silent — individual tokens, too noisy for Discord)
          ThinkingDeltaEvent  → (silent — individual tokens, too noisy for Discord)
        """
        shell_output_buf: list[str] = []

        async def _send(text: str) -> None:
            chunks = [text[i:i+MAX_REPLY_LEN] for i in range(0, len(text), MAX_REPLY_LEN)]
            for chunk in chunks:
                try:
                    await channel.send(chunk)
                except discord.HTTPException as exc:
                    log.warning("discord_sink_send_failed", error=str(exc))

        async def _flush_shell_buf() -> None:
            if not shell_output_buf:
                return
            combined = "".join(shell_output_buf)
            shell_output_buf.clear()
            # Wrap in a code block for readability; truncate if huge
            display = combined[:1800]
            await _send(f"```\n{display}\n```")

        async def sink(event: AgentEvent) -> None:
            nonlocal shell_output_buf

            if isinstance(event, TaskStartEvent):
                await _send(f"🔍 Working on: *{event.content}*")

            elif isinstance(event, ThinkingEndEvent):
                if event.text:
                    await _send(f"🧠 *{event.text[:1900]}*")

            elif isinstance(event, TextTurnEndEvent):
                # is_final=True → this is the reply, sent by _handle_message; skip here
                if not event.is_final and event.text:
                    await _send(f"💭 {event.text[:1900]}")

            elif isinstance(event, ToolCallStartEvent):
                # Skip noisy tools that don't add value in the chat
                if event.tool_name not in _SILENT_TOOLS:
                    await _send(f"🔧 `{event.tool_name}({_fmt_args(event.args)})`")

            elif isinstance(event, ShellStartEvent):
                await _send(f"$ `{event.command[:200]}`")

            elif isinstance(event, ShellOutputEvent):
                shell_output_buf.append(event.chunk)
                if len(shell_output_buf) >= _SHELL_OUTPUT_BATCH:
                    await _flush_shell_buf()

            elif isinstance(event, ShellDoneEvent):
                await _flush_shell_buf()
                await _send(f"exit {event.exit_code} ({event.elapsed_s:.1f}s)")

            elif isinstance(event, ProgressEvent):
                if event.message:
                    await _send(event.message)

            elif isinstance(event, TaskErrorEvent):
                await _send(f"❌ {event.error[:400]}")

            # TaskDoneEvent, ToolResultEvent, TextDeltaEvent,
            # ThinkingDeltaEvent — intentionally silent in Discord

        return sink

    async def _announce_online(self) -> None:
        """Post an online announcement to the bus channel."""
        if not settings.discord_bus_channel_id:
            return

        channel = self._client.get_channel(settings.discord_bus_channel_id)
        if channel is None:
            log.warning("bus_channel_not_found", id=settings.discord_bus_channel_id)
            return

        try:
            await channel.send(  # type: ignore[union-attr]
                f"**{settings.agent_name}** is online. Model: `{settings.agent_model}`. "
                f"Type `@{settings.agent_name} <task>` to assign work."
            )
        except discord.HTTPException as exc:
            log.warning("announce_failed", error=str(exc))
