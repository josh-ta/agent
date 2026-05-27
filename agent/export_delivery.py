"""Track workspace export files for Discord attachment delivery."""

from __future__ import annotations

import contextvars
import re
from pathlib import Path

import structlog

from agent.config import settings
from agent.tools.discord_tools import DiscordAttachment

log = structlog.get_logger()

_pending_exports: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "pending_export_paths",
    default=None,
)

_BARE_EXPORT_FILENAME_RE = re.compile(
    r"\b([\w.-]+\.(?:csv|tsv|json|txt|md))\b",
    re.IGNORECASE,
)


def register_export_path(path: str) -> None:
    """Record a workspace export path for attachment delivery on this task."""
    text = str(path or "").strip()
    if not text:
        return
    current = _pending_exports.get()
    paths = [] if current is None else list(current)
    if text not in paths:
        paths.append(text)
    _pending_exports.set(paths)


def take_export_paths() -> list[str]:
    """Return and clear registered export paths for the current task."""
    paths = _pending_exports.get()
    _pending_exports.set(None)
    return [] if paths is None else list(paths)


def resolve_export_path(path: str) -> Path:
    raw = Path(path.strip())
    if raw.is_absolute():
        return raw.resolve()
    return (settings.workspace_path / raw).resolve()


def attachments_for_paths(paths: list[str]) -> list[DiscordAttachment]:
    from agent.tools.filesystem import read_workspace_attachment

    attachments: list[DiscordAttachment] = []
    seen: set[str] = set()
    for path in paths:
        payload = read_workspace_attachment(path)
        if payload is None:
            log.warning("discord_attachment_skipped", path=path, reason="unreadable")
            continue
        filename, data = payload
        if filename in seen:
            continue
        seen.add(filename)
        log.info("discord_attachment_collected", filename=filename, bytes=len(data), path=path)
        attachments.append(DiscordAttachment(filename=filename, data=data))
    return attachments


def attachments_from_registered_exports() -> list[DiscordAttachment]:
    return attachments_for_paths(take_export_paths())


_IGNORE_BARE_EXPORT_FILENAMES = frozenset({"task_journal.md", "your-file.csv"})


def bare_export_filenames_in_text(text: str) -> list[str]:
    if not text.strip():
        return []
    seen: set[str] = set()
    paths: list[str] = []
    for match in _BARE_EXPORT_FILENAME_RE.finditer(text):
        name = match.group(1)
        if name in seen or name.lower() in _IGNORE_BARE_EXPORT_FILENAMES:
            continue
        seen.add(name)
        paths.append(name)
    return paths
