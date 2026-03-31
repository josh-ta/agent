"""
Shell command validation (CC BashTool-style, lightweight).

Conservative checks: cwd must stay under workspace, deny obvious hostiles,
optional read-only mode for a restricted tool.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

_READ_ONLY_DENY = re.compile(
    r"(?i)(\brm\b|\bmv\b|\bcp\b|>>|> |\btee\b|\bchmod\b|\bchown\b|"
    r"\bmkfs\b|\bdd\b|\bwget\b|\bcurl\b.*(-X|--request)|\bgit\s+push\b|"
    r"\bdocker\s+(run|exec|rm|kill)\b|\bkubectl\b)"
)


def resolve_shell_cwd(working_dir: str | None, workspace: Path) -> tuple[Path | None, str | None]:
    """Return (cwd, error_message)."""
    workspace = workspace.resolve()
    if not working_dir:
        cwd = workspace
    else:
        requested = Path(working_dir)
        cwd = requested if requested.is_absolute() else workspace / requested
    try:
        cwd = cwd.resolve()
    except OSError as exc:
        return None, f"[ERROR: working_dir invalid: {exc}]"
    if not cwd.exists():
        return None, f"[ERROR: working_dir not found: {cwd}]"
    if not cwd.is_dir():
        return None, f"[ERROR: working_dir is not a directory: {cwd}]"
    try:
        cwd.relative_to(workspace)
    except ValueError:
        return None, f"[ERROR: working_dir must be under workspace {workspace}]"
    return cwd, None


def validate_shell_command(
    command: str,
    *,
    read_only: bool = False,
) -> str | None:
    """
    Return an error string to block execution, or None if allowed.

    Uses shlex.split for a light parse; fails closed on unparseable input.
    """
    text = command.strip()
    if not text:
        return "[ERROR: empty command]"
    if "\x00" in text:
        return "[ERROR: null byte in command]"
    try:
        parts = shlex.split(text)
    except ValueError as exc:
        return f"[ERROR: could not parse command safely: {exc}]"
    if not parts:
        return "[ERROR: empty command after parse]"
    lowered = text.lower()
    if "&&" in text or "||" in text or ";" in text:
        if read_only:
            return "[ERROR: read-only shell: compound commands are not allowed]"
    dangerous = (
        ":(){ ",  # fork bomb pattern start
        "$(rm ",
        "`rm ",
    )
    for needle in dangerous:
        if needle in lowered:
            return f"[ERROR: blocked pattern in command: {needle.strip()}]"
    if read_only and _READ_ONLY_DENY.search(text):
        return "[ERROR: read-only shell: mutating or network write patterns are not allowed]"
    return None
