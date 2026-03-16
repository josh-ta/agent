"""
Discord bot integration.

The bot classifies inbound messages, routes tasks into `AgentLoop`, and sends
final replies. Streaming is always bound to the private agent channel so
agent-comms and agent-bus stay clean.
"""

from __future__ import annotations

import asyncio
import traceback

import discord
import structlog

from agent.communication.discord_services import (
    DiscordEventPresenter,
    MAX_REPLY_LEN,
    MessageHandlingService,
)
from agent.config import settings
from agent.events import bridge
from agent.loop import AgentLoop
from agent.tools.discord_tools import set_discord_client

log = structlog.get_logger()


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
        self._presenter = DiscordEventPresenter(self._client)
        self._messages = MessageHandlingService(
            agent_loop=loop,
            client=self._client,
            presenter=self._presenter,
            event_bridge=bridge,
        )

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
        """Route a Discord message to the message handling service."""
        await self._messages.handle_message(message)

    async def _send_reply(
        self,
        parsed,
        output: str,
        original_message: discord.Message,
    ) -> None:
        await self._messages.send_reply(parsed, output, original_message)

    async def _send_chunked(self, channel: discord.abc.Messageable, text: str) -> None:
        """Send a long text to a channel, splitting into chunks as needed."""
        await self._presenter.send_chunked(channel, text)

    async def _post_bus_status(self, message: str) -> None:
        """Post a brief status message to agent-bus. Silent if not configured."""
        await self._messages.post_bus_status(message)

    def _make_discord_sink(self, channel: discord.abc.Messageable):  # type: ignore[return]
        return self._presenter.make_sink(channel)

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
