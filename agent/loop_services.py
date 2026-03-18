"""
Supporting services for `agent.loop`.

These classes keep side effects and branching logic out of the queue coordinator
so they can be tested with fakes.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import structlog
from pydantic_ai import Agent
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    TextPartDelta,
    ThinkingPartDelta,
)
from pydantic_ai.usage import UsageLimits

from agent.config import settings
from agent.events import (
    ProgressEvent,
    TaskDoneEvent,
    TaskErrorEvent,
    TaskStartEvent,
    TextDeltaEvent,
    TextTurnEndEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolResultEvent,
    bridge,
)

if TYPE_CHECKING:
    from agent.loop import Task, TaskResult

log = structlog.get_logger()


class TaskJournal:
    def __init__(self, root: Path, *, now_fn: Callable[[], float] = time.time) -> None:
        self._path = root / ".task_journal.md"
        self._now = now_fn

    @property
    def path(self) -> Path:
        return self._path

    def expire_stale(self, *, max_age_s: int = 1800) -> None:
        try:
            if self._path.exists():
                age_s = self._now() - self._path.stat().st_mtime
                if age_s > max_age_s:
                    self._path.unlink()
                    log.info("task_journal_expired", age_s=round(age_s))
        except Exception:
            pass

    def append(self, title: str, body: str) -> None:
        try:
            ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n### [{ts}] — {title}\n{body}\n")
        except Exception:
            pass

    def clear(self) -> None:
        try:
            if self._path.exists():
                self._path.unlink()
        except Exception:
            pass


class TaskContextBuilder:
    def __init__(self, memory_store: Any, history_reader: Callable[[int, int], Awaitable[str]] | None = None) -> None:
        self._memory = memory_store
        self._history_reader = history_reader

    async def build(self, task: "Task") -> tuple["Task", str, str]:
        from agent.loop import Task as LoopTask
        from agent.loop import _classify_tier, _parse_override

        content, forced_tier = _parse_override(task.content)
        tier = forced_tier or _classify_tier(content)

        normalized_task = LoopTask(
            content=content,
            source=task.source,
            author=task.author,
            channel_id=task.channel_id,
            message_id=task.message_id,
            metadata=task.metadata,
            created_at=task.created_at,
            inject_queue=task.inject_queue,
        )

        lessons_context = await self._load_lessons(task.content)
        channel_context = await self._load_channel_history(normalized_task)

        parts: list[str] = []
        if channel_context:
            parts.append(channel_context)
        if lessons_context:
            parts.append(lessons_context)
        parts.append(normalized_task.content)
        return normalized_task, tier, "\n\n---\n\n".join(parts)

    async def _load_lessons(self, content: str) -> str:
        if self._memory and hasattr(self._memory, "search_lessons"):
            try:
                return await self._memory.search_lessons(content[:200], limit=3)
            except Exception:
                return ""
        return ""

    async def _load_channel_history(self, task: "Task") -> str:
        if task.source != "discord" or not task.channel_id:
            return ""

        try:
            if self._memory and hasattr(self._memory, "get_history"):
                history_rows = await self._memory.get_history(task.channel_id, limit=10)
                history_rows = history_rows[:-1] if history_rows else []
                lines: list[str] = []
                total_chars = 0
                for row in history_rows:
                    role_label = "You" if row["role"] == "assistant" else row["role"].capitalize()
                    text = row["content"][:300] + ("…" if len(row["content"]) > 300 else "")
                    line = f"{role_label}: {text}"
                    if total_chars + len(line) > 1500:
                        break
                    lines.append(line)
                    total_chars += len(line)
                return "## Recent conversation history\n" + "\n".join(lines) if lines else ""

            if self._history_reader is None:
                from agent.tools.discord_tools import discord_read

                self._history_reader = discord_read

            raw = await self._history_reader(task.channel_id, 7)
            lines = raw.splitlines()
            if lines:
                lines = lines[:-1]
            truncated: list[str] = []
            total_chars = 0
            for line in lines:
                short = line[:300] + ("…" if len(line) > 300 else "")
                if total_chars + len(short) > 1500:
                    break
                truncated.append(short)
                total_chars += len(short)
            return "## Recent conversation history\n" + "\n".join(truncated) if truncated else ""
        except Exception:
            return ""


class RunExecutor:
    def __init__(
        self,
        *,
        event_bridge: Any = bridge,
        journal: TaskJournal | None = None,
        summarize_context: Callable[["Task", str], Awaitable[str]] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._bridge = event_bridge
        self._journal = journal or TaskJournal(settings.workspace_path)
        self._summarize_context = summarize_context
        self._sleep = sleep

    async def run(
        self,
        *,
        task: "Task",
        agent: Agent,  # type: ignore[type-arg]
        base_prompt: str,
        tier: str,
        message_history: list | None = None,
    ) -> tuple[str, int, bool]:
        prompt = base_prompt
        context_retries = 0
        max_context_retries = 2
        rate_retries = 0
        max_rate_retries = 3
        rate_delay = 5.0
        bad_args_retries = 0
        max_bad_args_retries = 2

        while True:
            discord_replied = False
            tool_calls = 0
            result_output = ""

            try:
                async with agent.run_mcp_servers():
                    thinking_buf: list[str] = []
                    text_buf: list[str] = []
                    is_final = False

                    async for event in agent.run_stream_events(
                        prompt,
                        message_history=message_history or [],
                        usage_limits=UsageLimits(request_limit=None),
                    ):
                        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
                            delta = event.delta.content_delta
                            if delta:
                                thinking_buf.append(delta)
                                await self._bridge.emit(ThinkingDeltaEvent(delta=delta))
                        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                            delta = event.delta.content_delta
                            text_buf.append(delta)
                            await self._bridge.emit(TextDeltaEvent(delta=delta))
                        elif isinstance(event, FinalResultEvent):
                            is_final = True
                            result_output = str(event.output) if hasattr(event, "output") else ""
                        elif isinstance(event, PartEndEvent):
                            if thinking_buf:
                                full_thinking = "".join(thinking_buf).strip()
                                thinking_buf = []
                                if full_thinking:
                                    await self._bridge.emit(ThinkingEndEvent(text=full_thinking))
                            elif text_buf:
                                full_text = "".join(text_buf).strip()
                                text_buf = []
                                if full_text:
                                    await self._bridge.emit(TextTurnEndEvent(text=full_text, is_final=is_final))
                        elif isinstance(event, FunctionToolCallEvent):
                            tool_calls += 1
                            tool_name = event.part.tool_name
                            if tool_name == "send_discord":
                                discord_replied = True
                            await self._bridge.emit(
                                ToolCallStartEvent(
                                    tool_name=tool_name,
                                    call_id=event.part.tool_call_id,
                                    args=self._parse_tool_args(event.part.args),
                                )
                            )
                        elif isinstance(event, FunctionToolResultEvent):
                            ret = getattr(event, "result", None)
                            result_str = str(getattr(ret, "content", ret)) if ret is not None else ""
                            await self._bridge.emit(
                                ToolResultEvent(
                                    tool_name=getattr(event, "tool_name", ""),
                                    call_id=getattr(ret, "tool_call_id", ""),
                                    result=result_str[:500],
                                )
                            )

                if task.inject_queue and not task.inject_queue.empty():
                    injected = self._drain_queue(task.inject_queue)
                    if injected:
                        count = len(injected)
                        combined = "\n".join(f"[{i + 1}] {msg}" for i, msg in enumerate(injected))
                        await self._bridge.emit(
                            ProgressEvent(
                                message=(
                                    f"💬 Got {'your messages' if count > 1 else 'your message'} — "
                                    f"{'folding them' if count > 1 else 'folding it'} in now."
                                )
                            )
                        )
                        followup_output, followup_calls, followup_replied = await self.run(
                            task=task,
                            agent=agent,
                            base_prompt=(
                                f"{base_prompt}\n\n---\n\n"
                                f"## New message{'s' if count > 1 else ''} received while you were working:\n"
                                f"{combined}\n\n"
                                f"Please incorporate {'these' if count > 1 else 'this'} into your work and continue."
                            ),
                            tier=tier,
                            message_history=message_history,
                        )
                        result_output = followup_output
                        tool_calls += followup_calls
                        discord_replied = discord_replied or followup_replied

                return result_output, tool_calls, discord_replied

            except Exception as exc:
                exc_str = str(exc)
                is_context_overflow = "prompt is too long" in exc_str or ("400" in exc_str and "maximum" in exc_str)
                is_rate_limit = "429" in exc_str
                is_bad_args = "EOF while parsing" in exc_str or "args_as_dict" in exc_str

                if is_context_overflow and context_retries < max_context_retries and self._summarize_context:
                    context_retries += 1
                    await self._bridge.emit(
                        ProgressEvent(
                            message=(
                                f"📦 Context window full — compressing and continuing… "
                                f"(attempt {context_retries}/{max_context_retries})"
                            )
                        )
                    )
                    summary = await self._summarize_context(task, prompt)
                    self._journal.append(
                        f"CONTEXT COMPRESSED (retry {context_retries})",
                        summary,
                    )
                    prompt = (
                        f"{base_prompt}\n\n"
                        f"## Progress so far (summarized — context was compressed):\n{summary}"
                    )
                    continue

                if is_rate_limit and rate_retries < max_rate_retries:
                    rate_retries += 1
                    await self._bridge.emit(
                        ProgressEvent(message=f"⏸️ Rate limited — retrying in {int(rate_delay)}s…")
                    )
                    await self._sleep(rate_delay)
                    rate_delay *= 2
                    continue

                if is_bad_args and bad_args_retries < max_bad_args_retries:
                    bad_args_retries += 1
                    await self._bridge.emit(
                        ProgressEvent(
                            message=(
                                "⚠️ Tool args got truncated — retrying from last checkpoint… "
                                f"({bad_args_retries}/{max_bad_args_retries})"
                            )
                        )
                    )
                    message_history = None
                    continue

                raise

    @staticmethod
    def _parse_tool_args(raw_args: object) -> object:
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args)
            except Exception:
                return raw_args
        return raw_args

    @staticmethod
    def _drain_queue(queue: asyncio.Queue[str]) -> list[str]:
        items: list[str] = []
        while not queue.empty():
            try:
                items.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items


class ReflectionService:
    def __init__(self, *, agents: dict[str, Agent], memory_store: Any, journal: TaskJournal | None = None) -> None:  # type: ignore[type-arg]
        self._agents = agents
        self._memory = memory_store
        self._journal = journal or TaskJournal(settings.workspace_path)

    @property
    def _fast_agent(self) -> Agent:  # type: ignore[type-arg]
        return self._agents.get("fast") or self._agents.get("smart") or next(iter(self._agents.values()))

    async def reflect(self, task: "Task", result: "TaskResult", success_count: int, memory_update_interval: int) -> None:
        if not self._memory:
            return

        try:
            if not result.success:
                await self._reflect_on_failure(task, result)
            else:
                if result.tool_calls > 7:
                    await self._reflect_on_success(task, result)
                if success_count % memory_update_interval == 0:
                    await self.update_memory_md()
        except Exception:
            log.warning("reflect_error", exc=traceback.format_exc())

    async def _reflect_on_success(self, task: "Task", result: "TaskResult") -> None:
        reflection_prompt = (
            f"You just successfully completed the following task using {result.tool_calls} tool calls.\n\n"
            f"Task: {task.content}\n\n"
            f"Result summary: {result.output[:500]}\n\n"
            "In one concise sentence (max 200 chars), extract ONE reusable pattern, shortcut, or insight "
            "from this task that would help you do similar tasks better or faster in the future. "
            "Focus on what was surprising, efficient, or non-obvious. "
            "If there is nothing genuinely useful to record, reply with exactly: NOTHING_TO_RECORD"
        )
        try:
            insight_result = await self._fast_agent.run(
                reflection_prompt,
                usage_limits=UsageLimits(request_limit=None),
            )
            insight = str(insight_result.output).strip()
            if insight and "NOTHING_TO_RECORD" not in insight:
                await self._memory.save_lesson(
                    summary=insight[:300],
                    kind="pattern",
                    context=task.content[:300],
                )
        except Exception:
            log.warning("reflect_on_success_error", exc=traceback.format_exc())

    async def _reflect_on_failure(self, task: "Task", result: "TaskResult") -> None:
        reflection_prompt = (
            "You just attempted the following task and it FAILED.\n\n"
            f"Task: {task.content}\n\n"
            f"Error/output: {result.output}\n\n"
            "In 1-2 sentences, diagnose what went wrong and state what you should do differently next time. "
            "Be concise and specific."
        )
        try:
            reflection = await self._fast_agent.run(
                reflection_prompt,
                usage_limits=UsageLimits(request_limit=None),
            )
            lesson = str(reflection.output).strip()
            await self._memory.save_lesson(
                summary=lesson,
                kind="mistake",
                context=task.content[:300],
            )
            await self.update_memory_md()
        except Exception:
            log.warning("reflect_on_failure_error", exc=traceback.format_exc())

    async def update_memory_md(self) -> None:
        try:
            recent = await self._memory.get_recent_lessons(limit=20)
            memory_path = settings.identity_path / "MEMORY.md"
            if not memory_path.exists():
                return

            content = memory_path.read_text(encoding="utf-8")
            marker = "## Recent Lessons"
            if marker in content:
                content = content[:content.index(marker)].rstrip()
            else:
                content = content.rstrip()

            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            content += f"\n\n{marker}\n_Last updated: {now}_\n\n{recent}\n"
            memory_path.write_text(content, encoding="utf-8")
        except Exception:
            log.warning("update_memory_md_error", exc=traceback.format_exc())


@dataclass
class HeartbeatTask:
    id: str
    description: str
    from_agent: str


class HeartbeatService:
    def __init__(self, *, memory_store: Any, postgres_store: Any, enqueue: Callable[["Task"], Awaitable[None]]) -> None:
        self._memory = memory_store
        self._postgres = postgres_store
        self._enqueue = enqueue

    async def heartbeat(self, *, is_busy: bool) -> None:
        if self._memory and hasattr(self._memory, "heartbeat"):
            await self._memory.heartbeat()

        if self._postgres is not None and not is_busy:
            try:
                rows = await self._postgres.get_pending_task_rows()
                for row in rows:
                    await self._postgres.mark_task_running(row["id"])
                    await self._enqueue(self._build_a2a_task(row))
            except Exception as exc:
                log.warning("a2a_poll_error", error=str(exc))

    @staticmethod
    def _build_a2a_task(row: dict[str, Any]) -> "Task":
        from agent.loop import Task

        return Task(
            content=row["description"],
            source="a2a",
            author=row.get("from_agent", "unknown"),
            metadata={"task_id": row["id"], "from_agent": row.get("from_agent", "unknown")},
        )
