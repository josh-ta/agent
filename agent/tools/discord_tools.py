"""
Discord tools: send messages, uploads, and read channel history.

These are thin wrappers called from within a running Pydantic AI tool.
The actual discord.py Client lives in communication/discord_bot.py;
these functions receive it as a dependency or via a module-level ref.
"""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import discord
import structlog

from agent.task_waits import UserInputRequired, current_task_wait_context

if TYPE_CHECKING:
    from discord.abc import Messageable

log = structlog.get_logger()

# Module-level reference to the discord client (set by discord_bot.py on startup)
_discord_client: discord.Client | None = None
_pending_question_ids: dict[int, set[int]] = {}


@dataclass(frozen=True)
class DiscordAttachment:
    filename: str
    data: bytes

    def to_file(self) -> discord.File:
        return discord.File(io.BytesIO(self.data), filename=self.filename)


def set_discord_client(client: discord.Client) -> None:
    global _discord_client
    _discord_client = client


def has_pending_question(channel_id: int) -> bool:
    return bool(_pending_question_ids.get(channel_id))


def is_pending_question_reply(message: "discord.Message") -> bool:
    pending_ids = _pending_question_ids.get(message.channel.id)
    if not pending_ids:
        return False
    ref = getattr(message, "reference", None)
    return getattr(ref, "message_id", None) in pending_ids


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
        sent_messages = await send_text(channel, message)
        log.info("discord_sent", channel=channel_id, chunks=sent_messages)
        return f"Sent {sent_messages} message(s) to channel {channel_id}."
    except Exception as exc:
        return f"[ERROR: {exc}]"


async def send_text(channel: "Messageable", message: str, *, max_len: int = 1990) -> int:
    if not message:
        return 0

    chunks = [message[i:i + max_len] for i in range(0, len(message), max_len)]
    for chunk in chunks:
        await channel.send(chunk)
    return len(chunks)


async def send_attachments(
    channel: "Messageable",
    attachments: Sequence[DiscordAttachment],
    *,
    message: str = "",
) -> int:
    sent_messages = 0
    if message:
        sent_messages += await send_text(channel, message)

    for attachment in attachments:
        await channel.send(file=attachment.to_file())
        sent_messages += 1
    return sent_messages


def decode_data_url_attachment(data_url: str, *, filename: str = "browser-screenshot.png") -> DiscordAttachment | None:
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        return None

    try:
        return DiscordAttachment(
            filename=filename,
            data=base64.b64decode(data_url[len(prefix):], validate=True),
        )
    except (ValueError, binascii.Error):
        log.warning("discord_attachment_decode_failed")
        return None


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
    if current_task_wait_context() is not None:
        raise UserInputRequired(question=question, timeout_s=timeout)

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
        _pending_question_ids.setdefault(channel_id, set()).add(sent.id)
        log.info("ask_user_sent", channel=channel_id, question=question[:80])
    except Exception as exc:
        return f"[ERROR sending question: {exc}]"

    # Poll for a reply from a human. Prefer an explicit reply to our question;
    # otherwise only accept a lone human response to avoid grabbing chatter.
    import asyncio as _asyncio
    poll_interval = 3  # seconds between checks
    elapsed = 0
    try:
        while elapsed < timeout:
            await _asyncio.sleep(poll_interval)
            elapsed += poll_interval
            try:
                candidates = []
                async for msg in channel.history(limit=50, after=sent, oldest_first=True):  # type: ignore[union-attr]
                    if not msg.author.bot and msg.content.strip():
                        candidates.append(msg)

                for msg in candidates:
                    ref = getattr(msg, "reference", None)
                    if ref and getattr(ref, "message_id", None) == sent.id:
                        log.info("ask_user_replied", elapsed_s=elapsed)
                        return msg.content

                distinct_authors = {msg.author.id for msg in candidates}
                if len(distinct_authors) == 1 and candidates:
                    reply = candidates[-1]
                    log.info("ask_user_replied", elapsed_s=elapsed)
                    return reply.content
            except Exception as exc:
                return f"[ERROR reading reply: {exc}]"

        log.warning("ask_user_timeout", timeout=timeout, question=question[:80])
        return f"[No reply received after {timeout}s — proceeding with best judgment]"
    finally:
        pending_ids = _pending_question_ids.get(channel_id)
        if pending_ids is not None:
            pending_ids.discard(sent.id)
            if not pending_ids:
                _pending_question_ids.pop(channel_id, None)


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
