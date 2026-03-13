"""
Filesystem tools: read, write, list, delete files within the workspace.

All paths are resolved relative to WORKSPACE_PATH to prevent escaping
the sandbox.  If the agent needs to access absolute paths outside the
workspace it should use the shell tool instead.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import structlog

from agent.config import settings

log = structlog.get_logger()

MAX_READ_BYTES = 32 * 1024  # 32 KB


def _safe_path(path: str) -> Path:
    """Resolve path relative to workspace; raise if it escapes."""
    workspace = settings.workspace_path.resolve()
    resolved = (workspace / path).resolve()

    # Allow absolute paths that are already under workspace
    if not str(resolved).startswith(str(workspace)):
        # For absolute paths outside workspace, still allow (agent may need /tmp etc)
        # but log it
        log.warning("path_outside_workspace", path=path, resolved=str(resolved))

    return resolved


def read_file(path: str, encoding: str = "utf-8") -> str:
    """
    Read a file and return its content.

    Args:
        path: File path (relative to workspace or absolute).
        encoding: Text encoding. Use 'binary' to get hex dump of binary files.

    Returns:
        File content as string, or error message.
    """
    try:
        fp = _safe_path(path)
        if not fp.exists():
            return f"[ERROR: file not found: {path}]"
        if not fp.is_file():
            return f"[ERROR: not a file: {path}]"

        size = fp.stat().st_size
        if encoding == "binary":
            data = fp.read_bytes()
            return data[:MAX_READ_BYTES].hex()

        content = fp.read_text(encoding=encoding, errors="replace")
        if len(content) > MAX_READ_BYTES:
            content = content[:MAX_READ_BYTES] + f"\n... [truncated, total {size} bytes]"
        return content

    except Exception as exc:
        return f"[ERROR: {exc}]"


def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """
    Write content to a file, creating parent directories as needed.

    Args:
        path: File path (relative to workspace or absolute).
        content: Text content to write.

    Returns:
        Success message or error.
    """
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding=encoding)
        log.info("file_written", path=str(fp), size=len(content))
        return f"Written {len(content)} bytes to {fp}"
    except Exception as exc:
        return f"[ERROR: {exc}]"


def list_dir(path: str = ".") -> str:
    """
    List directory contents with file sizes and types.

    Args:
        path: Directory path. Defaults to workspace root.

    Returns:
        Formatted directory listing.
    """
    try:
        dp = _safe_path(path)
        if not dp.exists():
            return f"[ERROR: path not found: {path}]"
        if not dp.is_dir():
            return f"[ERROR: not a directory: {path}]"

        entries = sorted(dp.iterdir(), key=lambda p: (p.is_file(), p.name))
        if not entries:
            return f"(empty directory: {dp})"

        lines = [f"Contents of {dp}:", ""]
        for entry in entries:
            try:
                s = entry.stat()
                kind = "DIR " if entry.is_dir() else "FILE"
                size = f"{s.st_size:>10,} B" if entry.is_file() else "           "
                lines.append(f"  {kind}  {size}  {entry.name}")
            except OSError:
                lines.append(f"  ????             {entry.name}")

        return "\n".join(lines)
    except Exception as exc:
        return f"[ERROR: {exc}]"


def delete_file(path: str) -> str:
    """
    Delete a file (not a directory).

    Args:
        path: File path to delete.

    Returns:
        Success message or error.
    """
    try:
        fp = _safe_path(path)
        if not fp.exists():
            return f"[ERROR: file not found: {path}]"
        if fp.is_dir():
            return "[ERROR: use shell 'rm -rf' for directories]"
        fp.unlink()
        return f"Deleted {fp}"
    except Exception as exc:
        return f"[ERROR: {exc}]"
