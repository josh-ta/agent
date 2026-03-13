"""
Shell tool: run arbitrary commands on the host system.

Security notes:
- Commands run as the container user (non-root by default)
- Working directory defaults to WORKSPACE_PATH
- Output is capped at 8 KB to avoid flooding the context window
- Timeout defaults to 30 s; agent can override per call
"""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path

import structlog

from agent.config import settings

log = structlog.get_logger()

MAX_OUTPUT_BYTES = 2 * 1024  # 2 KB — use read_file for large files, not cat


async def shell_run(
    command: str,
    working_dir: str | None = None,
    timeout: int = 30,
) -> str:
    """
    Execute a shell command and return combined stdout + stderr.

    Args:
        command: The shell command to run (passed to bash -c).
        working_dir: Directory to run in. Defaults to /workspace.
        timeout: Max seconds to wait. Defaults to 30.

    Returns:
        Combined output string with exit code appended.
    """
    cwd = Path(working_dir) if working_dir else settings.workspace_path
    # Only auto-create the default workspace — don't create arbitrary dirs,
    # as they may be owned by a different user (e.g. after a root-created clone).
    if not working_dir:
        cwd.mkdir(parents=True, exist_ok=True)
    elif not cwd.exists():
        # Fall back to workspace root so the shell command can report the error
        # cleanly (e.g. "cd: no such file") rather than crashing the tool call.
        cwd = settings.workspace_path
        cwd.mkdir(parents=True, exist_ok=True)

    log.info("shell_run", command=command[:200], cwd=str(cwd), timeout=timeout)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd),
            executable="/bin/bash",
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return f"[TIMEOUT after {timeout}s]\n"

        output = stdout.decode("utf-8", errors="replace")

        # Truncate if too large — use read_file tool for reading large files
        if len(output) > MAX_OUTPUT_BYTES:
            output = output[:MAX_OUTPUT_BYTES] + f"\n... [truncated at 2KB, total {len(output)} bytes — use read_file for large files]"

        result = f"{output}\n[exit code: {proc.returncode}]"
        log.info("shell_done", exit_code=proc.returncode, output_len=len(output))
        return result

    except Exception as exc:
        log.error("shell_error", error=str(exc))
        return f"[ERROR: {exc}]"
