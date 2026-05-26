"""Discord event rendering: status embeds, streaming replies, and retries."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, cast

import discord
import structlog

from agent.communication.discord_constants import (
    MAX_REPLY_LEN,
    SILENT_TOOLS,
    STATUS_EMBED_DEBOUNCE_SECONDS,
    escape_codeblock,
    summarize_tool_activity,
)
from agent.events import (
    AgentEvent,
    ProgressEvent,
    ShellDoneEvent,
    ShellOutputEvent,
    ShellStartEvent,
    TaskErrorEvent,
    TaskStartEvent,
    TextDeltaEvent,
    TextTurnEndEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
)
from agent.tools import discord_tools as discord_tools_module

log = structlog.get_logger()


async def send_with_retry(
    channel: discord.abc.Messageable,
    *,
    content: str = "",
    embed: discord.Embed | None = None,
    max_len: int = MAX_REPLY_LEN,
    retries: int = 2,
) -> discord.Message | None:
    """Send or chunk-send with one retry on rate limits."""
    attempt = 0
    while attempt <= retries:
        try:
            if embed is not None:
                return await channel.send(embed=embed)
            if not content:
                return None
            chunks = [content[i : i + max_len] for i in range(0, len(content), max_len)]
            sent: discord.Message | None = None
            for chunk in chunks:
                sent = await channel.send(chunk)
            return sent
        except discord.HTTPException as exc:
            status = getattr(exc, "status", None)
            if status == 429 and attempt < retries:
                retry_after = getattr(exc, "retry_after", 1.0) or 1.0
                log.warning("discord_rate_limited", retry_after=retry_after, attempt=attempt)
                await asyncio.sleep(float(retry_after))
                attempt += 1
                continue
            log.warning(
                "discord_send_failed",
                error=str(exc),
                status=status,
                task_id=getattr(exc, "task_id", None),
            )
            return None
    return None


async def edit_with_retry(
    message: discord.Message,
    *,
    content: str | None = None,
    embed: discord.Embed | None = None,
    retries: int = 2,
) -> bool:
    attempt = 0
    while attempt <= retries:
        try:
            await message.edit(content=content, embed=embed)
            return True
        except discord.HTTPException as exc:
            status = getattr(exc, "status", None)
            if status == 429 and attempt < retries:
                retry_after = getattr(exc, "retry_after", 1.0) or 1.0
                await asyncio.sleep(float(retry_after))
                attempt += 1
                continue
            log.warning("discord_edit_failed", error=str(exc), status=status)
            return False
    return False


async def delete_with_retry(message: discord.Message, *, retries: int = 2) -> bool:
    attempt = 0
    while attempt <= retries:
        try:
            await message.delete()
            return True
        except discord.HTTPException as exc:
            status = getattr(exc, "status", None)
            if status == 429 and attempt < retries:
                retry_after = getattr(exc, "retry_after", 1.0) or 1.0
                await asyncio.sleep(float(retry_after))
                attempt += 1
                continue
            log.warning("discord_delete_failed", error=str(exc), status=status)
            return False
    return False


class StatusEmbedManager:
    """Debounced single-message status embed for tool/shell/progress updates."""

    def __init__(
        self,
        channel: discord.abc.Messageable,
        *,
        debounce_seconds: float = STATUS_EMBED_DEBOUNCE_SECONDS,
    ) -> None:
        self._channel = channel
        self._debounce_seconds = debounce_seconds
        self._message: discord.Message | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._started_at = time.monotonic()
        self._current_tool = ""
        self._last_shell = ""
        self._progress = ""
        self._cancel_state = ""
        self._failed_shell_output = ""
        self._active = False

    def set_channel(self, channel: discord.abc.Messageable) -> None:
        self._channel = channel

    def set_cancelling(self) -> None:
        self._active = True
        self._cancel_state = "Cancelling…"
        self._schedule_flush()

    def set_stopped(self) -> None:
        self._active = True
        self._cancel_state = "Stopped"
        self._schedule_flush()

    async def handle_tool(self, tool_name: str, args: object) -> None:
        if tool_name in SILENT_TOOLS:
            return
        self._active = True
        self._current_tool = summarize_tool_activity(tool_name, args)
        self._schedule_flush()

    async def handle_progress(self, message: str) -> None:
        if not message:
            return
        self._active = True
        text = message.strip()
        if text.startswith("⏳ Still working — "):
            text = text.removeprefix("⏳ Still working — ").rstrip(".")
            text = f"Still working — {text}…"
        elif text.startswith("⌛ "):
            text = text.removeprefix("⌛ ")
        self._progress = text[:500]
        if "cancel" in message.lower():
            self._cancel_state = message[:120]
        self._schedule_flush()

    async def handle_shell_start(self, command: str) -> None:
        self._active = True
        self._last_shell = command[:200]
        self._failed_shell_output = ""
        self._schedule_flush()

    async def handle_shell_output(self, chunk: str) -> None:
        self._active = True
        self._failed_shell_output = (self._failed_shell_output + chunk)[-1400:]

    async def handle_shell_done(self, *, exit_code: int, elapsed_s: float) -> None:
        self._active = True
        if exit_code != 0 and self._failed_shell_output.strip():
            body = escape_codeblock(self._failed_shell_output.strip())
            self._progress = f"Shell failed (exit {exit_code}, {elapsed_s:.1f}s):\n```{body}```"[:900]
        self._schedule_flush()

    def _schedule_flush(self) -> None:
        if self._flush_task is not None and not self._flush_task.done():
            return
        self._flush_task = asyncio.create_task(self._debounced_flush())

    async def _debounced_flush(self) -> None:
        if self._debounce_seconds <= 0:
            await self.flush()
            return
        await asyncio.sleep(self._debounce_seconds)
        await self.flush()

    async def flush(self) -> None:
        if not self._active:
            return
        elapsed = int(time.monotonic() - self._started_at)
        lines: list[str] = []
        if self._cancel_state:
            lines.append(self._cancel_state)
        if self._current_tool:
            lines.append(self._current_tool)
        if self._last_shell:
            lines.append(f"Shell: `{self._last_shell}`")
        if self._progress:
            lines.append(self._progress)
        lines.append(f"Elapsed: {elapsed}s")

        embed = discord.Embed(
            title="Working…",
            description="\n".join(lines)[:4000] or "Starting…",
            color=discord.Color.blurple(),
        )
        if self._message is None:
            self._message = await send_with_retry(self._channel, embed=embed)
        else:
            await edit_with_retry(self._message, embed=embed)

    async def dismiss(self) -> None:
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._flush_task
        if self._message is not None:
            await delete_with_retry(self._message)
            self._message = None

    async def finalize(self, *, success: bool = True) -> None:
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._flush_task
        if self._message is None and self._active:
            await self.flush()
        await self.dismiss()


class DiscordEventPresenter:
    def __init__(self, client: discord.Client) -> None:
        self._client = client

    async def send_chunked(
        self,
        channel: discord.abc.Messageable,
        text: str,
        *,
        fallback_channel: discord.abc.Messageable | None = None,
    ) -> bool:
        sent = await send_with_retry(channel, content=text)
        if sent is not None or not text:
            return True
        if fallback_channel is not None and fallback_channel is not channel:
            sent = await send_with_retry(fallback_channel, content="⚠️ Could not deliver full output to the primary channel.")
            return sent is not None
        return False

    async def send_attachments(
        self,
        channel: discord.abc.Messageable,
        attachments: list[discord_tools_module.DiscordAttachment],
    ) -> None:
        try:
            await discord_tools_module.send_attachments(channel, attachments)
        except discord.HTTPException as exc:
            log.warning("send_attachment_failed", error=str(exc))

    async def create_task_thread(
        self,
        parent_channel: discord.abc.Messageable,
        *,
        task_summary: str,
    ) -> discord.abc.Messageable | None:
        create_thread = getattr(parent_channel, "create_thread", None)
        if create_thread is None:
            return None
        from agent.config import settings

        name = f"{settings.agent_name}: {task_summary[:60]}".strip()
        try:
            thread = await create_thread(name=name[:100], auto_archive_duration=60)
            return cast(discord.abc.Messageable, thread)
        except discord.HTTPException as exc:
            log.warning("discord_thread_create_failed", error=str(exc))
            return None

    def make_sink(
        self,
        channel: discord.abc.Messageable,
        *,
        expected_run_generation: int | None = None,
        main_channel: discord.abc.Messageable | None = None,
        channel_id: int | None = None,
        session_state: Any | None = None,
        debounce_seconds: float | None = None,
        reply_to: discord.Message | None = None,
        create_thread_fn: Callable[[], Awaitable[discord.abc.Messageable | None]] | None = None,
    ) -> Callable[[AgentEvent], Awaitable[None]]:
        debounce = STATUS_EMBED_DEBOUNCE_SECONDS if debounce_seconds is None else debounce_seconds
        reply_channel = main_channel or channel
        work_channel = channel
        status = StatusEmbedManager(work_channel, debounce_seconds=debounce)
        reply_message: discord.Message | None = None
        reply_buffer = ""
        reply_delivered = False
        shell_lines: list[str] = []
        thread_promoted = False

        async def _promote_to_thread() -> None:
            nonlocal work_channel, thread_promoted
            if thread_promoted or create_thread_fn is None:
                return
            thread = await create_thread_fn()
            if thread is None:
                return
            thread_promoted = True
            work_channel = thread
            status.set_channel(work_channel)

        async def _update_reply(text: str, *, final: bool = False) -> None:
            nonlocal reply_message, reply_delivered
            preview = text.strip()
            if not preview and not final:
                return
            body = preview[:MAX_REPLY_LEN] if preview else "…"
            if reply_message is None:
                if reply_to is not None:
                    try:
                        reply_message = await reply_to.reply(body, mention_author=False)
                    except discord.HTTPException:
                        reply_message = await send_with_retry(reply_channel, content=body)
                else:
                    reply_message = await send_with_retry(reply_channel, content=body)
            else:
                await edit_with_retry(reply_message, content=body)
            if final and preview:
                reply_delivered = True

        async def sink(event: AgentEvent) -> None:
            nonlocal reply_buffer, shell_lines

            if expected_run_generation is not None:
                rg = getattr(event, "run_generation", None)
                if rg is not None and rg != expected_run_generation:
                    return

            if channel_id is not None and session_state is not None and session_state.is_cancelling(channel_id):
                status.set_cancelling()

            if isinstance(event, TaskStartEvent):
                return

            if isinstance(event, TextDeltaEvent):
                reply_buffer += event.delta
                return

            if isinstance(event, ThinkingEndEvent):
                if event.text:
                    log.info(
                        "model_thinking",
                        chars=len(event.text),
                        preview=event.text[:240],
                    )
                return

            if isinstance(event, TextTurnEndEvent):
                if event.is_final and event.text:
                    # PartEndEvent text can be a trailing segment only; keep the fuller buffer.
                    if len(event.text.strip()) > len(reply_buffer.strip()):
                        reply_buffer = event.text
                    await _update_reply(reply_buffer, final=True)
                elif not event.is_final and event.text:
                    await status.handle_progress(event.text[:240])
                return

            if isinstance(event, ToolCallStartEvent):
                await _promote_to_thread()
                await status.handle_tool(event.tool_name, event.args)
                return

            if isinstance(event, ShellStartEvent):
                await _promote_to_thread()
                shell_lines = []
                await status.handle_shell_start(event.command)
                return

            if isinstance(event, ShellOutputEvent):
                shell_lines.append(event.chunk)
                await status.handle_shell_output(event.chunk)
                return

            if isinstance(event, ShellDoneEvent):
                await status.handle_shell_done(exit_code=event.exit_code, elapsed_s=event.elapsed_s)
                shell_lines = []
                return

            if isinstance(event, ProgressEvent):
                await status.handle_progress(event.message or "")
                return

            if isinstance(event, TaskErrorEvent):
                await status.handle_progress(f"❌ {event.error[:400]}")
                await status.finalize(success=False)
                if channel_id is not None and session_state is not None:
                    session_state.clear_cancelling(channel_id)

        async def finalize_reply(text: str) -> None:
            """Apply the authoritative task output to the user-visible reply."""
            nonlocal reply_buffer
            cleaned = text.strip()
            if not cleaned:
                return
            reply_buffer = cleaned
            await _update_reply(reply_buffer, final=True)

        sink.finalize_status = status.finalize  # type: ignore[attr-defined]
        sink.mark_stopped = status.set_stopped  # type: ignore[attr-defined]
        sink.finalize_reply = finalize_reply  # type: ignore[attr-defined]
        sink.reply_delivered = lambda: reply_delivered  # type: ignore[attr-defined]
        sink.reply_text = lambda: reply_buffer  # type: ignore[attr-defined]
        return sink
