"""
Discord bot: receives messages, routes them to the agent loop, sends replies.

Uses discord.py with the following intents:
  - message_content  (required to read message text)
  - guilds
  - guild_messages

Channel roles
-------------
  #bob / #barbara (DISCORD_AGENT_CHANNEL_ID)
      The user's real-time window into the agent. ALL streaming — thinking,
      tool calls, shell output, progress events — goes here regardless of
      where the triggering message came from. Also used for direct user↔agent
      conversation and full task replies.

  #agent-comms (DISCORD_COMMS_CHANNEL_ID)
      Machine channel. Structured JSON A2A task routing only. When an agent
      finishes an A2A task, it posts the result back here as JSON so the
      requesting agent can read it programmatically. No reasoning, no streaming.

  #agent-bus (DISCORD_BUS_CHANNEL_ID)
      Broadcast channel. Brief one-line status announcements only. No streaming.

Event bridge integration
------------------------
A per-task sink is registered on the EventBridge against the PRIVATE channel
(DISCORD_AGENT_CHANNEL_ID), not the channel the triggering message came from.
This ensures thinking/tool streaming always appears in the private channel,
keeping agent-comms and agent-bus clean.
"""

from __future__ import annotations

import asyncio
import json
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
_SILENT_TOOLS = frozenset({
    "read_file", "list_dir", "memory_save", "lesson_search",
    "task_resume", "task_journal_clear", "task_note", "memory_search",
    "lessons_recent", "lesson_save", "read_channel", "read_discord",
    "identity_read", "skill_list", "skill_read", "db_stats",
})


def _escape_md_italics(text: str) -> str:
    """Escape bare asterisks so they don't break Discord italic/bold markdown."""
    return text.replace("*", "\\*")


def _escape_codeblock(text: str) -> str:
    """Prevent triple-backtick sequences inside a code block from breaking it."""
    return text.replace("```", "`` `")


def _fmt_args(args: object) -> str:
    """Format tool args for display — truncate to keep Discord messages readable."""
    if isinstance(args, dict):
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
        # Per-channel inject queues for zipper-merge
        self._inject_queues: dict[int, asyncio.Queue[str]] = {}
        # The channel_id of the task currently running (0 if none)
        self._active_channel: int = 0

        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.guild_messages = True

        self._client = discord.Client(intents=intents)
        self._setup_events()

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
        """Route a Discord message to the agent loop and send replies."""
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

        # The private channel is ALWAYS used for streaming, regardless of where
        # the message came from. This keeps agent-comms and agent-bus clean.
        private_channel = self._client.get_channel(settings.discord_agent_channel_id)

        # Zipper-merge: if agent is busy, inject into running task or queue
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

        # Agent is free — set up inject queue and run
        inject_q = asyncio.Queue()
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

        # Always register the EventBridge sink against the PRIVATE channel so
        # that all thinking/tool/shell streaming goes there, never to comms/bus.
        sink_tag = f"discord_{parsed.channel_id}_{id(task)}"
        if private_channel is not None:
            discord_sink = self._make_discord_sink(private_channel)  # type: ignore[arg-type]
            bridge.register(sink_tag, discord_sink)

        # Show typing indicator in the private channel during processing
        typing_ctx = (
            private_channel.typing()  # type: ignore[union-attr]
            if private_channel is not None
            else message.channel.typing()  # type: ignore[union-attr]
        )

        self._agent_loop.is_busy = True
        try:
            async with typing_ctx:
                result = await self._agent_loop._process(task)
        finally:
            self._agent_loop.is_busy = False
            self._active_channel = 0
            self._inject_queues.pop(parsed.channel_id, None)
            bridge.unregister(sink_tag)

        if result.discord_replied:
            return

        reply = result.output
        if not reply:
            return

        await self._send_reply(parsed, result.output, message)

    async def _send_reply(
        self,
        parsed: object,
        output: str,
        original_message: discord.Message,
    ) -> None:
        """
        Route the final task result to the appropriate channel(s).

          A2A task  → post structured JSON result back to agent-comms (so the
                       requesting agent can read it) + brief status to agent-bus
          Bus task  → post brief status to agent-bus; full result to private
          Direct    → reply in the private channel
        """
        from agent.communication.message_router import ParsedMessage

        # Narrow the type for attribute access
        parsed_msg = parsed  # type: ignore[assignment]

        is_a2a = (
            hasattr(parsed_msg, "kind")
            and parsed_msg.kind == MessageKind.A2A  # type: ignore[attr-defined]
            and getattr(parsed_msg, "a2a_payload", None)
        )
        is_bus = (
            hasattr(parsed_msg, "channel_id")
            and parsed_msg.channel_id == settings.discord_bus_channel_id  # type: ignore[attr-defined]
        )

        if is_a2a:
            from_agent = parsed_msg.a2a_payload.get("from", "")  # type: ignore[attr-defined]
            # Post JSON result back to comms for the requesting agent to read
            if from_agent and settings.discord_comms_channel_id:
                comms = self._client.get_channel(settings.discord_comms_channel_id)
                if comms is not None:
                    reply_json = json.dumps({
                        "from": settings.agent_name,
                        "to": from_agent,
                        "task": "result",
                        "payload": output[:1800],
                    })
                    try:
                        await comms.send(reply_json)  # type: ignore[union-attr]
                    except discord.HTTPException as exc:
                        log.warning("a2a_reply_failed", error=str(exc))
            # Also post a brief status to the bus
            await self._post_bus_status(
                f"**{settings.agent_name}** completed task from {from_agent or 'unknown'}."
            )

        elif is_bus:
            # Mentioned in bus — short status to bus, full result to private
            await self._post_bus_status(f"**{settings.agent_name}**: {output[:300]}")
            private_channel = self._client.get_channel(settings.discord_agent_channel_id)
            if private_channel is not None:
                await self._send_chunked(private_channel, output)  # type: ignore[arg-type]

        else:
            # Direct private-channel task — reply in private
            private_channel = self._client.get_channel(settings.discord_agent_channel_id)
            target = private_channel or original_message.channel
            try:
                chunks = [output[i:i+MAX_REPLY_LEN] for i in range(0, len(output), MAX_REPLY_LEN)]
                # Use message.reply only if the original message was in the private channel
                if (
                    original_message.channel.id == settings.discord_agent_channel_id
                    and chunks
                ):
                    await original_message.reply(chunks[0], mention_author=False)
                    for chunk in chunks[1:]:
                        await target.send(chunk)  # type: ignore[union-attr]
                else:
                    await self._send_chunked(target, output)  # type: ignore[arg-type]
            except discord.HTTPException as exc:
                log.error("discord_send_failed", error=str(exc))

    async def _send_chunked(self, channel: discord.abc.Messageable, text: str) -> None:
        """Send a long text to a channel, splitting into chunks as needed."""
        chunks = [text[i:i+MAX_REPLY_LEN] for i in range(0, len(text), MAX_REPLY_LEN)]
        for chunk in chunks:
            try:
                await channel.send(chunk)
            except discord.HTTPException as exc:
                log.warning("send_chunked_failed", error=str(exc))

    async def _post_bus_status(self, message: str) -> None:
        """Post a brief status message to agent-bus. Silent if not configured."""
        if not settings.discord_bus_channel_id:
            return
        bus = self._client.get_channel(settings.discord_bus_channel_id)
        if bus is None:
            return
        try:
            await bus.send(message[:MAX_REPLY_LEN])  # type: ignore[union-attr]
        except discord.HTTPException as exc:
            log.warning("bus_status_failed", error=str(exc))

    def _make_discord_sink(self, channel: discord.abc.Messageable):  # type: ignore[return]
        """
        Return an async sink function bound to the given channel (always private).

        Event → Discord mapping:
          TaskStartEvent      → 🔍 Working on: *<content>*
          ThinkingEndEvent    → 🧠 *<thinking>*  (chunked)
          TextTurnEndEvent    → 💭 <text>  (intermediate turns only)
          ToolCallStartEvent  → 🔧 `tool_name(args)`  (silent for read/memory tools)
          ShellStartEvent     → $ `command`
          ShellOutputEvent    → buffered, flushed as a single code block on ShellDoneEvent
          ShellDoneEvent      → appended to the shell output block as exit N (X.Xs)
          ProgressEvent       → message as-is
          TaskErrorEvent      → ❌ error
          (all others silent)
        """
        # Per-command buffer: (message_obj, lines)
        # We edit the original message to append the exit code rather than sending a new one.
        shell_lines: list[str] = []
        shell_msg: discord.Message | None = None

        async def _send(text: str) -> None:
            chunks = [text[i:i+MAX_REPLY_LEN] for i in range(0, len(text), MAX_REPLY_LEN)]
            for chunk in chunks:
                try:
                    await channel.send(chunk)
                except discord.HTTPException as exc:
                    log.warning("discord_sink_send_failed", error=str(exc))

        async def sink(event: AgentEvent) -> None:
            nonlocal shell_lines, shell_msg

            if isinstance(event, TaskStartEvent):
                pass  # don't echo the task back — user already knows what they asked

            elif isinstance(event, ThinkingEndEvent):
                if event.text:
                    chunk_size = 1800
                    for i in range(0, len(event.text), chunk_size):
                        chunk = event.text[i:i+chunk_size].strip()
                        if chunk:
                            await _send(f"🧠 *{_escape_md_italics(chunk)}*")

            elif isinstance(event, TextTurnEndEvent):
                if not event.is_final and event.text:
                    await _send(f"💭 {event.text[:1900]}")

            elif isinstance(event, ToolCallStartEvent):
                if event.tool_name not in _SILENT_TOOLS:
                    await _send(f"🔧 `{event.tool_name}({_fmt_args(event.args)})`")

            elif isinstance(event, ShellStartEvent):
                # Show the command; buffer output to append on done
                shell_lines = []
                shell_msg = None
                try:
                    shell_msg = await channel.send(f"$ `{event.command[:200]}`")  # type: ignore[union-attr]
                except discord.HTTPException:
                    pass

            elif isinstance(event, ShellOutputEvent):
                # Buffer output — only flush to Discord on done to avoid message spam
                shell_lines.append(event.chunk)

            elif isinstance(event, ShellDoneEvent):
                output = "".join(shell_lines).strip()
                shell_lines = []
                # Only annotate with exit status when it's a failure or there's output
                failed = event.exit_code != 0
                status = f"exit {event.exit_code} ({event.elapsed_s:.1f}s)" if failed else None
                if output:
                    display = _escape_codeblock(output[-1400:])  # tail — most relevant part
                    body = f"```\n{display}\n```"
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
                    # Failed with no output — show the exit code
                    try:
                        if shell_msg is not None:
                            await shell_msg.edit(content=f"{shell_msg.content} → {status}")
                        else:
                            await _send(status)
                    except discord.HTTPException:
                        await _send(status)
                # Success with no output — silently discard (no edit needed)
                shell_msg = None

            elif isinstance(event, ProgressEvent):
                if event.message:
                    await _send(event.message)

            elif isinstance(event, TaskErrorEvent):
                await _send(f"❌ {event.error[:400]}")

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
            from importlib.metadata import version as _pkg_version
            agent_version = _pkg_version("agent")
        except Exception:
            agent_version = "unknown"

        try:
            await channel.send(  # type: ignore[union-attr]
                f"**{settings.agent_name}** v{agent_version} is online. "
                f"Model: `{settings.agent_model}`. "
                f"Type `@{settings.agent_name} <task>` to assign work."
            )
        except discord.HTTPException as exc:
            log.warning("announce_failed", error=str(exc))
