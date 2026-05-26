"""Shared Discord constants and formatting helpers."""

from __future__ import annotations

from agent.config import settings

MAX_REPLY_LEN = 1990
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


def escape_md_italics(text: str) -> str:
    return text.replace("*", "\\*")


def escape_codeblock(text: str) -> str:
    return text.replace("```", "`` `")


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
