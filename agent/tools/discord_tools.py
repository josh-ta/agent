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
    Read recent messages from a Discord channel.

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
