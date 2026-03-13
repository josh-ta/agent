"""
Discord bot: receives messages, routes them to the agent loop, sends replies.

Uses discord.py with the following intents:
  - message_content  (required to read message text)
  - guilds
  - guild_messages
"""

from __future__ import annotations

import asyncio
import traceback

import discord
import structlog

from agent.communication.message_router import MessageKind, a2a_to_task_content, classify
from agent.config import settings
from agent.loop import AgentLoop, Task
from agent.tools.discord_tools import set_discord_client

log = structlog.get_logger()

MAX_REPLY_LEN = 1990  # Discord limit minus a small buffer


class DiscordBot:
    """Wraps a discord.py Client and connects it to the AgentLoop."""

    def __init__(self, loop: AgentLoop) -> None:
        self._agent_loop = loop

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
            # Announce presence on the bus channel
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
            # Log bus traffic to SQLite but don't respond unless relevant
            return

        # Build task content
        if parsed.kind == MessageKind.A2A and parsed.a2a_payload:
            task_content = a2a_to_task_content(parsed.a2a_payload)
        else:
            task_content = parsed.content

        if not task_content.strip():
            return

        task = Task(
            content=task_content,
            source="discord",
            author=parsed.author,
            channel_id=parsed.channel_id,
            message_id=parsed.message_id,
        )

        # Show typing indicator while the agent thinks
        channel = message.channel
        async with channel.typing():  # type: ignore[union-attr]
            result = await self._agent_loop._process(task)

        # Send reply
        reply = result.output
        if not reply:
            return

        try:
            # Split into Discord-sized chunks
            chunks = [reply[i:i+MAX_REPLY_LEN] for i in range(0, len(reply), MAX_REPLY_LEN)]
            # Reply to the original message for the first chunk
            await message.reply(chunks[0], mention_author=False)
            for chunk in chunks[1:]:
                await channel.send(chunk)  # type: ignore[union-attr]
        except discord.HTTPException as exc:
            log.error("discord_send_failed", error=str(exc))

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
