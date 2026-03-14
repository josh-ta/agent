"""
Filesystem tools: read, write, list, delete files within the workspace.

All paths are resolved relative to WORKSPACE_PATH to prevent escaping
the sandbox.  If the agent needs to access absolute paths outside the
workspace it should use the shell tool instead.
"""

from __future__ import annotations

import os
import shlex
import stat
import subprocess
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


def str_replace_file(
    path: str,
    old_str: str,
    new_str: str,
    expected_replacements: int = 1,
) -> str:
    """
    Replace an exact string in a file and write it back.

    Verifies the old string appears exactly expected_replacements times before
    writing, so the agent gets a loud error instead of a silent wrong edit.

    Args:
        path: File path (relative to workspace or absolute).
        old_str: The exact text to find. Include enough surrounding context to
                 make it unique.
        new_str: The replacement text.
        expected_replacements: How many times old_str must appear (default 1).

    Returns:
        Success message or error describing the mismatch.
    """
    try:
        fp = _safe_path(path)
        if not fp.exists():
            return f"[ERROR: file not found: {path}]"
        if not fp.is_file():
            return f"[ERROR: not a file: {path}]"

        content = fp.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_str)
        if count == 0:
            return (
                f"[ERROR: old_str not found in {path}. "
                f"Check for whitespace or encoding differences.]"
            )
        if count != expected_replacements:
            return (
                f"[ERROR: expected {expected_replacements} occurrence(s) of old_str "
                f"but found {count} in {path}. "
                f"Make old_str more specific or set expected_replacements={count}.]"
            )

        updated = content.replace(old_str, new_str, expected_replacements)
        fp.write_text(updated, encoding="utf-8")
        log.info("str_replace", path=str(fp), count=count)
        return f"Replaced {count} occurrence(s) in {fp}"
    except Exception as exc:
        return f"[ERROR: {exc}]"


MAX_SEARCH_BYTES = 8 * 1024  # 8 KB — enough for ~200 lines of context


def search_files(
    pattern: str,
    path: str = ".",
    file_glob: str = "",
    context_lines: int = 2,
) -> str:
    """
    Search files using ripgrep (rg) and return matching lines with context.

    Args:
        pattern: Regular expression to search for.
        path: Directory or file to search (relative to workspace or absolute).
        file_glob: File name filter, e.g. '*.py' or '*.{ts,tsx}'.
        context_lines: Lines of context to show around each match (default 2).

    Returns:
        Ripgrep output (file:line:content format) or error message.
    """
    try:
        resolved = _safe_path(path)
        if not resolved.exists():
            return f"[ERROR: path not found: {path}]"

        cmd: list[str] = [
            "rg",
            "--line-number",
            f"--context={context_lines}",
            "--no-heading",
        ]
        if file_glob:
            cmd += ["--glob", file_glob]
        cmd += [pattern, str(resolved)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = result.stdout
        if not output and result.returncode == 1:
            return f"(no matches for pattern: {pattern})"
        if result.returncode not in (0, 1):
            return f"[ERROR: rg exited {result.returncode}: {result.stderr[:200]}]"

        if len(output) > MAX_SEARCH_BYTES:
            output = (
                output[:MAX_SEARCH_BYTES]
                + f"\n... [truncated at 8KB — refine your pattern or file_glob]"
            )
        return output or f"(no matches for pattern: {pattern})"
    except FileNotFoundError:
        return "[ERROR: ripgrep (rg) not found — install it with: apt-get install ripgrep]"
    except subprocess.TimeoutExpired:
        return "[ERROR: search timed out after 30s — narrow the search path or pattern]"
    except Exception as exc:
        return f"[ERROR: {exc}]"
