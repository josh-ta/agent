"""Shared Discord constants and formatting helpers."""

from __future__ import annotations

import re

from agent.config import settings

MAX_REPLY_LEN = 1990
_PARAGRAPH_BREAK = re.compile(r"\n\n+")
STATUS_EMBED_DEBOUNCE_SECONDS = 1.5

SILENT_TOOLS = frozenset(
    {
        "read_file",
        "list_dir",
        "run_shell_read_only",
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
        "secret_list",
        "secret_set",
        "secret_get",
        "secret_delete",
    }
)


def split_message_chunks(text: str, *, max_len: int = MAX_REPLY_LEN) -> list[str]:
    """
    Split text into Discord-sized messages without breaking paragraphs when possible.

    Packing prefers whole paragraphs (blocks separated by blank lines). Oversized
    paragraphs are split on line breaks, then spaces, then a hard cut at max_len.
    """
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    paragraphs = [p for p in _PARAGRAPH_BREAK.split(text) if p]
    if not paragraphs:
        return _split_oversized(text, max_len=max_len)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_len:
            flush()
            chunks.extend(_split_oversized(paragraph, max_len=max_len))
            continue

        separator_len = 2 if current else 0  # "\n\n" between packed paragraphs
        needed = current_len + separator_len + len(paragraph)
        if current and needed > max_len:
            flush()
        current.append(paragraph)
        current_len = sum(len(p) for p in current) + 2 * (len(current) - 1)

    flush()
    return chunks


def _split_oversized(text: str, *, max_len: int) -> list[str]:
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break

        window = remaining[:max_len]
        split_at = window.rfind("\n\n")
        if split_at <= 0:
            split_at = window.rfind("\n")
        if split_at <= 0:
            split_at = window.rfind(" ")
        if split_at <= 0:
            split_at = max_len

        chunk = remaining[:split_at].rstrip()
        if not chunk:
            chunk = remaining[:max_len]
            split_at = max_len

        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip("\n")

    return chunks


def escape_md_italics(text: str) -> str:
    return text.replace("*", "\\*")


def escape_codeblock(text: str) -> str:
    return text.replace("```", "`` `")


def summarize_tool_activity(tool_name: str, args: object) -> str:
    """Short user-facing label for status updates — not raw tool internals."""
    labels = {
        "run_shell": "Running a command",
        "query_postgres": "Querying the database",
        "browser_navigate": "Browsing the web",
        "browser_screenshot": "Capturing a screenshot",
        "web_search": "Searching the web",
        "http_request": "Fetching a URL",
        "write_file": "Writing a file",
        "str_replace": "Editing a file",
        "send_discord": "Sending a message",
        "ask_user_question": "Waiting for your answer",
    }
    if tool_name in labels:
        return labels[tool_name]
    if tool_name in SILENT_TOOLS:
        return "Working…"
    return f"Using `{tool_name}`"


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
