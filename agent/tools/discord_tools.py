"""
Discord tools: send messages and read channel history.

These are thin wrappers called from within a running Pydantic AI tool.
The actual discord.py Client lives in communication/discord_bot.py;
these functions receive it as a dependency or via a module-level ref.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import discord

log = structlog.get_logger()

# Module-level reference to the discord client (set by discord_bot.py on startup)
_discord_client: "discord.Client | None" = None


def set_discord_client(client: "discord.Client") -> None:
    global _discord_client
    _discord_client = client


async def discord_send(channel_id: int, message: str) -> str:
    """
    Send a message to a Discord channel.

    Args:
        channel_id: Numeric Discord channel ID.
        message: Text to send (max 2000 chars; longer messages are split).

    Returns:
        Confirmation or error.
    """
    if _discord_client is None:
        return "[ERROR: Discord client not initialised]"

    channel = _discord_client.get_channel(channel_id)
    if channel is None:
        return f"[ERROR: channel {channel_id} not found or not accessible]"

    try:
        # Split messages longer than Discord's 2000-char limit
        chunks = [message[i:i+1990] for i in range(0, len(message), 1990)]
        for chunk in chunks:
            await channel.send(chunk)  # type: ignore[union-attr]
        log.info("discord_sent", channel=channel_id, chunks=len(chunks))
        return f"Sent {len(chunks)} message(s) to channel {channel_id}."
    except Exception as exc:
        return f"[ERROR: {exc}]"


async def discord_read(channel_id: int, limit: int = 20) -> str:
    """
    Read recent messages from a Discord channel by ID.

    Args:
        channel_id: Numeric Discord channel ID.
        limit: Number of recent messages to fetch (max 50).

    Returns:
        Formatted message history or error.
    """
    if _discord_client is None:
        return "[ERROR: Discord client not initialised]"

    limit = min(limit, 50)
    channel = _discord_client.get_channel(channel_id)
    if channel is None:
        return f"[ERROR: channel {channel_id} not found]"

    try:
        messages = []
        async for msg in channel.history(limit=limit):  # type: ignore[union-attr]
            ts = msg.created_at.strftime("%Y-%m-%d %H:%M")
            messages.append(f"[{ts}] {msg.author.display_name}: {msg.content}")

        messages.reverse()  # chronological order
        return "\n".join(messages) if messages else "(no messages)"
    except Exception as exc:
        return f"[ERROR: {exc}]"


async def ask_user(question: str, timeout: int = 300) -> str:
    """
    Post a question to the agent's private channel and wait for the user's reply.

    Sends the question with a ❓ prefix so the user knows a response is needed,
    then polls for a new message from a non-bot user for up to `timeout` seconds.

    Args:
        question: The question to ask the user.
        timeout:  Seconds to wait for a reply (default 300 = 5 minutes).

    Returns:
        The user's reply text, or a timeout notice if no reply arrives.
    """
    from agent.config import settings

    if _discord_client is None:
        return "[ERROR: Discord client not initialised]"

    channel_id = settings.discord_agent_channel_id
    if not channel_id:
        return "[ERROR: DISCORD_AGENT_CHANNEL_ID not configured]"

    channel = _discord_client.get_channel(channel_id)
    if channel is None:
        return f"[ERROR: private channel {channel_id} not found]"

    # Record the message ID we're about to send so we only read replies after it
    try:
        sent = await channel.send(f"❓ {question}")  # type: ignore[union-attr]
        log.info("ask_user_sent", channel=channel_id, question=question[:80])
    except Exception as exc:
        return f"[ERROR sending question: {exc}]"

    # Poll for a reply from a non-bot user
    import asyncio as _asyncio
    poll_interval = 3  # seconds between checks
    elapsed = 0
    while elapsed < timeout:
        await _asyncio.sleep(poll_interval)
        elapsed += poll_interval
        try:
            async for msg in channel.history(limit=10, after=sent):  # type: ignore[union-attr]
                if not msg.author.bot:
                    log.info("ask_user_replied", elapsed_s=elapsed, reply=msg.content[:80])
                    return msg.content
        except Exception as exc:
            return f"[ERROR reading reply: {exc}]"

    log.warning("ask_user_timeout", timeout=timeout, question=question[:80])
    return f"[No reply received after {timeout}s — proceeding with best judgment]"


async def discord_read_named(name: str, limit: int = 20) -> str:
    """
    Read recent messages from a named channel: 'private', 'bus', or 'comms'.

    Args:
        name: One of 'private' (this agent's channel), 'bus' (#agent-bus),
              or 'comms' (#agent-comms).
        limit: Number of recent messages to fetch (max 50).
    """
    from agent.config import settings

    channel_map = {
        "private": settings.discord_agent_channel_id,
        "bus": settings.discord_bus_channel_id,
        "comms": settings.discord_comms_channel_id,
    }
    channel_id = channel_map.get(name.lower())
    if not channel_id:
        return f"[ERROR: unknown channel name '{name}'. Use: private, bus, comms]"
    return await discord_read(channel_id, limit)
