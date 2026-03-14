"""
Shell tool: run arbitrary commands on the host system.

Security notes:
- Commands run as the container user (non-root by default)
- Working directory defaults to WORKSPACE_PATH
- Output is capped at 2 KB to avoid flooding the context window; use read_file for large files
- Timeout defaults to 120 s; agent can pass a higher value for long-running commands
  (e.g. timeout=3600 for docker build, npm install, etc.) — there is no upper limit

Streaming:
- stdout/stderr are read line-by-line and emitted as ShellOutputEvent via the bridge
  so Discord (and other sinks) can show live output as the command runs.
- The 2KB cap only applies to what is returned to the model — the bridge receives
  every line regardless of size.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import structlog

from agent.config import settings
from agent.events import bridge, ShellStartEvent, ShellOutputEvent, ShellDoneEvent

log = structlog.get_logger()

MAX_OUTPUT_BYTES = 2 * 1024  # 2 KB — use read_file for large files, not cat


async def shell_run(
    command: str,
    working_dir: str | None = None,
    timeout: int = 120,
) -> str:
    """
    Execute a shell command and return combined stdout + stderr.

    stdout/stderr are streamed line-by-line to the event bridge (ShellOutputEvent)
    as the process runs, so consumers see live output without waiting for completion.
    The return value (passed back to the model) is still capped at 2KB.

    Args:
        command: The shell command to run (passed to bash -c).
        working_dir: Directory to run in. Defaults to /workspace.
        timeout: Max seconds to wait. Defaults to 120. Pass a higher value
                 for long-running commands (docker build, npm install, etc.).
                 There is no upper limit — the agent controls this.

    Returns:
        Combined output string with exit code appended.
    """
    cwd = Path(working_dir) if working_dir else settings.workspace_path
    if not working_dir:
        cwd.mkdir(parents=True, exist_ok=True)
    elif not cwd.exists():
        cwd = settings.workspace_path
        cwd.mkdir(parents=True, exist_ok=True)

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
            # Drain any remaining output after kill
            try:
                assert proc.stdout is not None
                async for raw_line in proc.stdout:
                    await bridge.emit(ShellOutputEvent(chunk=raw_line.decode("utf-8", errors="replace")))
                    output_chunks.append(raw_line)
                    total_bytes += len(raw_line)
            except Exception:
                pass
            await proc.communicate()

        elapsed_s = time.monotonic() - start_time
        exit_code = proc.returncode if proc.returncode is not None else -1

        await bridge.emit(ShellDoneEvent(exit_code=exit_code, elapsed_s=elapsed_s))

        if timed_out:
            log.warning("shell_timeout", timeout=timeout)
            return f"[TIMEOUT after {timeout}s]\n"

        output = b"".join(output_chunks).decode("utf-8", errors="replace")

        # Truncate what is returned to the model — bridge already got everything
        if len(output) > MAX_OUTPUT_BYTES:
            output = (
                output[:MAX_OUTPUT_BYTES]
                + f"\n... [truncated at 2KB, total {total_bytes} bytes — use read_file for large files]"
            )

        result = f"{output}\n[exit code: {exit_code}]"
        log.info("shell_done", exit_code=exit_code, output_len=total_bytes)
        return result

    except Exception as exc:
        elapsed_s = time.monotonic() - start_time
        await bridge.emit(ShellDoneEvent(exit_code=-1, elapsed_s=elapsed_s))
        log.error("shell_error", error=str(exc))
        return f"[ERROR: {exc}]"
