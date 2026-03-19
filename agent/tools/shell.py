"""
Shell tool: run arbitrary commands on the host system.

Security notes:
- Commands run as the container user (non-root by default)
- Working directory defaults to WORKSPACE_PATH
- Output is capped at 10 KB to avoid flooding the context window; use read_file for large files
- Pass tail_lines=N to get the last N lines instead (useful for test runners)
- Timeout defaults to 120 s; agent can pass a higher value for long-running commands
  (e.g. timeout=3600 for docker build, npm install, etc.) — there is no upper limit

Streaming:
- stdout/stderr are read line-by-line and emitted as ShellOutputEvent via the bridge
  so Discord (and other sinks) can show live output as the command runs.
- The 10KB cap only applies to what is returned to the model — the bridge receives
  every line regardless of size.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import structlog

from agent.config import settings
from agent.events import bridge, ShellStartEvent, ShellOutputEvent, ShellDoneEvent

log = structlog.get_logger()

MAX_OUTPUT_BYTES = 10 * 1024  # 10 KB — use read_file for large files, not cat
REMOTE_PREFLIGHT_RE = re.compile(r"(?mi)^\s*#\s*remote-preflight:\s*(?P<evidence>.+)$")
SSH_RE = re.compile(r"(^|[\s;&(])ssh(\s|$)")
REMOTE_GUESS_RE = re.compile(r"(?i)(/root/theticketactionapp|<your-server-ip>|example\.com)")


def _validate_remote_command(command: str) -> str | None:
    if not SSH_RE.search(command):
        return None
    if "github.com" in command:
        return None
    if REMOTE_GUESS_RE.search(command):
        return "[ERROR: remote command blocked: guessed or placeholder deploy target detected.]"

    match = REMOTE_PREFLIGHT_RE.search(command)
    if match is None:
        return (
            "[ERROR: remote command blocked: add a first-line comment like "
            "'# remote-preflight: workspace=/workspace/<repo>; basis=user-provided host' "
            "after verifying the repo locally.]"
        )

    evidence = match.group("evidence").strip()
    workspace_match = re.search(r"workspace=([^\s;]+)", evidence)
    if workspace_match:
        workspace_path = Path(workspace_match.group(1))
        if not workspace_path.is_absolute():
            workspace_path = settings.workspace_path / workspace_path
        if not workspace_path.exists():
            return f"[ERROR: remote command blocked: preflight workspace path not found: {workspace_path}]"
        return None

    if "/workspace" not in evidence and "user" not in evidence and "explicit" not in evidence:
        return (
            "[ERROR: remote command blocked: preflight comment must cite workspace evidence "
            "or say the host/path came explicitly from the user.]"
        )
    return None


async def shell_run(
    command: str,
    working_dir: str | None = None,
    timeout: int = 120,
    tail_lines: int = 0,
) -> str:
    """
    Execute a shell command and return combined stdout + stderr.

    stdout/stderr are streamed line-by-line to the event bridge (ShellOutputEvent)
    as the process runs, so consumers see live output without waiting for completion.
    The return value (passed back to the model) is capped at 10KB by default.

    Args:
        command: The shell command to run (passed to bash -c).
        working_dir: Directory to run in. Defaults to /workspace.
        timeout: Max seconds to wait. Defaults to 120. Pass a higher value
                 for long-running commands (docker build, npm install, etc.).
                 There is no upper limit — the agent controls this.
        tail_lines: When > 0, return the last N lines instead of the first 10KB.
                    Use this for test runners where failures appear at the end.

    Returns:
        Combined output string with exit code appended.
    """
    if not working_dir:
        cwd = settings.workspace_path
        cwd.mkdir(parents=True, exist_ok=True)
    else:
        requested = Path(working_dir)
        cwd = requested if requested.is_absolute() else settings.workspace_path / requested
        if not cwd.exists():
            return f"[ERROR: working_dir not found: {cwd}]"
        if not cwd.is_dir():
            return f"[ERROR: working_dir is not a directory: {cwd}]"

    remote_validation_error = _validate_remote_command(command)
    if remote_validation_error is not None:
        return remote_validation_error

    log.info("shell_run", command=command[:200], cwd=str(cwd), timeout=timeout)
    await bridge.emit(ShellStartEvent(command=command, cwd=str(cwd)))

    start_time = time.monotonic()

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd),
            executable="/bin/bash",
        )

        output_chunks: list[bytes] = []
        total_bytes = 0
        timed_out = False

        async def _read_stream() -> None:
            """Read stdout line-by-line, emit each line, accumulate for return value."""
            nonlocal total_bytes
            assert proc.stdout is not None
            async for raw_line in proc.stdout:
                await bridge.emit(ShellOutputEvent(chunk=raw_line.decode("utf-8", errors="replace")))
                output_chunks.append(raw_line)
                total_bytes += len(raw_line)

        try:
            await asyncio.wait_for(_read_stream(), timeout=timeout)
            await proc.wait()
        except asyncio.TimeoutError:
            timed_out = True
            proc.kill()
            # Drain any remaining output after kill — bounded to 5s to avoid hanging
            async def _drain_after_kill() -> None:
                try:
                    assert proc.stdout is not None
                    async for raw_line in proc.stdout:
                        await bridge.emit(ShellOutputEvent(chunk=raw_line.decode("utf-8", errors="replace")))
                        output_chunks.append(raw_line)
                        total_bytes += len(raw_line)
                except Exception:
                    pass
            try:
                await asyncio.wait_for(_drain_after_kill(), timeout=5)
            except asyncio.TimeoutError:
                pass
            try:
                await asyncio.wait_for(proc.communicate(), timeout=3)
            except asyncio.TimeoutError:
                pass

        elapsed_s = time.monotonic() - start_time
        exit_code = proc.returncode if proc.returncode is not None else -1

        await bridge.emit(ShellDoneEvent(exit_code=exit_code, elapsed_s=elapsed_s))

        if timed_out:
            log.warning("shell_timeout", timeout=timeout)
            return f"[TIMEOUT after {timeout}s]\n"

        output = b"".join(output_chunks).decode("utf-8", errors="replace")

        # tail_lines mode: return the last N lines (ideal for test runners)
        if tail_lines > 0:
            lines = output.splitlines()
            if len(lines) > tail_lines:
                output = f"... [{len(lines) - tail_lines} earlier lines omitted]\n" + "\n".join(lines[-tail_lines:])
            # else output fits entirely — no truncation needed
        elif len(output) > MAX_OUTPUT_BYTES:
            output = (
                output[:MAX_OUTPUT_BYTES]
                + f"\n... [truncated at 10KB, total {total_bytes} bytes — use read_file for large files or tail_lines=N for test output]"
            )

        result = f"{output}\n[exit code: {exit_code}]"
        log.info("shell_done", exit_code=exit_code, output_len=total_bytes)
        return result

    except Exception as exc:
        elapsed_s = time.monotonic() - start_time
        await bridge.emit(ShellDoneEvent(exit_code=-1, elapsed_s=elapsed_s))
        log.error("shell_error", error=str(exc))
        return f"[ERROR: {exc}]"
