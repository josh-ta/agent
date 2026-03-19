"""
Discord bot integration.

The bot classifies inbound messages, routes tasks into `AgentLoop`, and sends
final replies. Streaming is always bound to the private agent channel so
agent-comms and agent-bus stay clean.
"""

from __future__ import annotations

import asyncio
import traceback
from types import SimpleNamespace
from typing import Any

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
        self._tree = discord.app_commands.CommandTree(self._client)
        self._commands_synced = False
        self._setup_events()
        self._register_app_commands()
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
            await self._sync_app_commands()
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

    def _register_app_commands(self) -> None:
        @self._tree.command(name="status", description="Show active, queued, and waiting work")
        async def status(interaction: discord.Interaction) -> None:
            await self._handle_slash_command(interaction, "status")

        @self._tree.command(name="cancel", description="Cancel the current task after the next safe step")
        async def cancel(interaction: discord.Interaction) -> None:
            await self._handle_slash_command(interaction, "cancel")

        @self._tree.command(name="forget", description="Discard the current task and queued stale work")
        async def forget(interaction: discord.Interaction) -> None:
            await self._handle_slash_command(interaction, "forget")

        @self._tree.command(name="clear", description="Clear queued tasks in this channel")
        async def clear(interaction: discord.Interaction) -> None:
            await self._handle_slash_command(interaction, "clear")

        @self._tree.command(name="resume", description="Repeat the current waiting question")
        async def resume(interaction: discord.Interaction) -> None:
            await self._handle_slash_command(interaction, "resume")

        @self._tree.command(name="help", description="Show the available agent commands")
        async def help_cmd(interaction: discord.Interaction) -> None:
            await self._handle_slash_command(interaction, "help")

        @self._tree.command(name="queue", description="Queue a new task")
        @discord.app_commands.describe(task="Task to add to the back of the queue")
        async def queue(interaction: discord.Interaction, task: str) -> None:
            await self._handle_slash_command(interaction, "queue", task)

        @self._tree.command(name="replace", description="Replace current work with a new task")
        @discord.app_commands.describe(task="Task to run after cancelling the current one")
        async def replace(interaction: discord.Interaction, task: str) -> None:
            await self._handle_slash_command(interaction, "replace", task)

    async def _sync_app_commands(self) -> None:
        if self._commands_synced:
            return
        try:
            guild_id = settings.discord_guild_id
            if guild_id:
                guild = discord.Object(id=guild_id)
                self._tree.copy_global_to(guild=guild)
                await self._tree.sync(guild=guild)
            else:
                await self._tree.sync()
            self._commands_synced = True
        except Exception:
            log.warning("discord_command_sync_failed", exc=traceback.format_exc())

    async def _handle_slash_command(
        self,
        interaction: discord.Interaction,
        name: str,
        argument: str = "",
    ) -> None:
        if interaction.channel is None or interaction.user is None:
            return
        content = f"/{name} {argument}".strip()
        message = _SlashCommandMessage(interaction=interaction, content=content)
        await self._messages.handle_message(message)  # type: ignore[arg-type]

    async def start_bot(self) -> None:
        """Connect to Discord and run the bot indefinitely."""
        token = settings.secret_value(settings.discord_bot_token)
        if not token:
            log.error("no_discord_token")
            return

        try:
            await self._client.start(token)
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
                f"Type `@{settings.agent_name} <task>` or use `/help`."
            )
        except discord.HTTPException as exc:
            log.warning("announce_failed", error=str(exc))


class _SlashCommandMessage:
    def __init__(self, *, interaction: discord.Interaction, content: str) -> None:
        self._interaction = interaction
        self.channel = interaction.channel
        self.id = int(getattr(interaction, "id", 0) or 0)
        self.content = content
        self.author = getattr(interaction, "user", SimpleNamespace(display_name="user", bot=False))
        self.replies: list[str] = []
        self.reactions: list[str] = []
        self.attachments: list[Any] = []
        self.reference = None

    async def reply(self, content: str, mention_author: bool = False) -> None:
        del mention_author
        self.replies.append(content)
        response = getattr(self._interaction, "response", None)
        if response is not None and hasattr(response, "is_done") and not response.is_done():
            await response.send_message(content)
            return
        followup = getattr(self._interaction, "followup", None)
        if followup is not None:
            await followup.send(content)
            return
        if self.channel is not None:
            await self.channel.send(content)  # type: ignore[union-attr]

    async def add_reaction(self, emoji: str) -> None:
        self.reactions.append(emoji)
