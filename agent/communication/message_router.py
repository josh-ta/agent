"""
Message router: parse and classify inbound Discord messages.

Classifies messages into:
  - TASK: addressed to this agent (mention or in private channel)
  - A2A:  structured JSON agent-to-agent message
  - BUS:  broadcast to all agents
  - IGNORE: not relevant to this agent
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import discord
import structlog

from agent.config import settings

log = structlog.get_logger()

A2A_RE = re.compile(r'^\s*\{.*"from"\s*:.*\}', re.DOTALL)


class MessageKind(Enum):
    TASK = auto()    # Agent should respond
    A2A = auto()     # Structured agent-to-agent payload
    BUS = auto()     # Broadcast, agent may monitor but need not respond
    IGNORE = auto()  # Not for this agent


@dataclass
class ParsedMessage:
    kind: MessageKind
    content: str
    author: str
    channel_id: int
    message_id: int
    a2a_payload: dict[str, Any] | None = None


def classify(message: discord.Message, bot_user: discord.ClientUser) -> ParsedMessage:
    """Classify an incoming Discord message."""
    channel_id = message.channel.id
    author = message.author.display_name
    content = message.content.strip()
    message_id = message.id

    # Ignore own messages
    if message.author == bot_user:
        return ParsedMessage(MessageKind.IGNORE, content, author, channel_id, message_id)

    # Ignore bots (other bots) unless in comms channel
    if message.author.bot and channel_id != settings.discord_comms_channel_id:
        return ParsedMessage(MessageKind.IGNORE, content, author, channel_id, message_id)

    # A2A structured messages (comms channel)
    if channel_id == settings.discord_comms_channel_id:
        if A2A_RE.match(content):
            try:
                payload = json.loads(content)
                to = payload.get("to", "")
                # Only process if addressed to this agent or broadcast
                if to in ("*", settings.agent_name, ""):
                    return ParsedMessage(
                        MessageKind.A2A, content, author, channel_id, message_id, a2a_payload=payload
                    )
                return ParsedMessage(MessageKind.IGNORE, content, author, channel_id, message_id)
            except json.JSONDecodeError:
                pass
        # Non-JSON message in comms from a human — treat as a task so both agents act on it.
        # Bot posts that are non-JSON (e.g. malformed JSON) remain IGNORE.
        if not message.author.bot:
            clean = re.sub(r"<@!?\d+>", "", content).strip()
            if clean:
                return ParsedMessage(MessageKind.TASK, clean, author, channel_id, message_id)

    # Private channel for this agent → always a task
    if channel_id == settings.discord_agent_channel_id:
        return ParsedMessage(MessageKind.TASK, content, author, channel_id, message_id)

    # Bus channel → task only if mentioned
    if channel_id == settings.discord_bus_channel_id:
        if bot_user.mentioned_in(message):
            # Strip mention from content
            clean = re.sub(r"<@!?\d+>", "", content).strip()
            return ParsedMessage(MessageKind.TASK, clean, author, channel_id, message_id)
        return ParsedMessage(MessageKind.BUS, content, author, channel_id, message_id)

    # Any other channel: only respond if mentioned
    if bot_user.mentioned_in(message):
        clean = re.sub(r"<@!?\d+>", "", content).strip()
        return ParsedMessage(MessageKind.TASK, clean, author, channel_id, message_id)

    return ParsedMessage(MessageKind.IGNORE, content, author, channel_id, message_id)


def a2a_to_task_content(payload: dict[str, Any]) -> str:
    """Convert an A2A JSON payload to a natural-language task string."""
    from_agent = payload.get("from", "unknown")
    task = payload.get("task", payload.get("content", ""))
    extra = payload.get("payload", "")
    parts = [f"[A2A from {from_agent}] {task}"]
    if extra:
        parts.append(f"Additional context: {extra}")
    return "\n".join(parts)
