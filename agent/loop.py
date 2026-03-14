"""
Main agent reasoning loop: Observe → Think → Act → Remember → Reflect.

The loop processes tasks from an asyncio.Queue populated by:
  - Discord messages
  - Scheduled heartbeats
  - Direct API calls (when used as a library)

After every task the agent runs a brief reflection pass:
  - On failure: diagnoses what went wrong, saves a lesson, updates MEMORY.md
  - On success with tool calls: extracts any reusable insight
  - Periodically: rewrites MEMORY.md with latest lessons summary

Model routing
-------------
Tasks are classified into three tiers before execution:
  fast  → haiku   (simple Q&A, greetings, status checks)
  smart → sonnet  (code, research, multi-step tasks)
  best  → opus    (architecture, complex reasoning, long tasks)

Users can override per-message with a prefix:
  /fast  <task>   force haiku
  /smart <task>   force sonnet
  /best  <task>   force opus

Streaming
---------
Uses agent.run_stream_events() for a single flat async stream of all events
across the entire run (multiple model turns + all tool calls). Raw pydantic-ai
events are translated into typed AgentEvent objects and emitted via the module-
level EventBridge singleton (agent.events.bridge). Sinks registered on the bridge
(Discord, logs, etc.) receive every event without the loop knowing about them.

Context management
------------------
When the context window overflows (400 "prompt is too long"), the loop asks
the fast agent to compress history into a brief summary, saves it, and retries.

Zipper-merge injection
-----------------------
New messages arriving while the agent is running are held in Task.inject_queue.
After each run completes, the queue is drained and a follow-up run is issued
with the injected messages appended so the model incorporates them.
"""

from __future__ import annotations

import asyncio
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_ai.messages import (
    FinalResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    TextPartDelta,
    ThinkingPartDelta,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
)

from agent.config import settings
from agent.events import (
    bridge,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    TextTurnEndEvent,
    ToolCallStartEvent,
    ToolResultEvent,
    TaskStartEvent,
    TaskDoneEvent,
    TaskErrorEvent,
    ProgressEvent,
)

log = structlog.get_logger()

# Reflect on MEMORY.md every N successful tasks
MEMORY_UPDATE_INTERVAL = 10


# ── Model routing ──────────────────────────────────────────────────────────────

# User override prefixes (case-insensitive)
_OVERRIDE_RE = re.compile(r"^/(fast|smart|best)\s+", re.IGNORECASE)

# Keywords that signal a complex task needing a smarter model
_BEST_KEYWORDS = re.compile(
    r"\b(architect|design|refactor|review|audit|security|"
    r"production|deploy|pipeline|ci/cd|complex|analysis|"
    r"deep\s+dive|explain\s+why|compare\s+tradeoffs)\b",
    re.IGNORECASE,
)
_SMART_KEYWORDS = re.compile(
    r"\b(code|implement|write|create|fix|debug|test|pr|"
    r"pull\s+request|commit|clone|install|setup|configure|"
    r"research|summarize|search|find|build|run|script|sql)\b",
    re.IGNORECASE,
)


def _parse_override(content: str) -> tuple[str, str | None]:
    """
    Strip a /fast|/smart|/best prefix from the message.
    Returns (cleaned_content, tier_override | None).
    """
    m = _OVERRIDE_RE.match(content)
    if m:
        tier = m.group(1).lower()
        return content[m.end():].strip(), tier
    return content, None


def _classify_tier(content: str) -> str:
    """
    Classify task complexity into fast | smart | best.
    Simple heuristic based on length and keyword matching.
    """
    words = len(content.split())
    if words < 8 and not _SMART_KEYWORDS.search(content):
        return "fast"
    if _BEST_KEYWORDS.search(content):
        return "best"
    if _SMART_KEYWORDS.search(content) or words > 40:
        return "smart"
    return "fast"


@dataclass
class Task:
    """A unit of work for the agent."""

    content: str
    source: str = "system"          # discord|system|heartbeat|api
    author: str = "system"
    channel_id: int = 0
    message_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Queue for zipper-merge injection: new user messages arriving mid-task
    # are pushed here by DiscordBot and drained after each run completes.
    inject_queue: asyncio.Queue[str] | None = field(default=None, repr=False)


@dataclass
class TaskResult:
    task: Task
    output: str
    success: bool
    elapsed_ms: float
    tool_calls: int = 0
    discord_replied: bool = False  # True if agent called send_discord during this task


class AgentLoop:
    """
    Async loop that processes tasks using the Pydantic AI agent.
    Maintains conversation history per Discord channel (or context window).
    Dynamically routes each task to the appropriate model tier.

    All streaming events are emitted via agent.events.bridge — the loop itself
    has no knowledge of Discord or any other output channel.
    """

    def __init__(self, agents: dict[str, Agent], memory_store: Any = None, postgres_store: Any = None) -> None:  # type: ignore[type-arg]
        # agents dict: {"fast": Agent, "smart": Agent, "best": Agent}
        self.agents = agents
        # Fallback: if only one agent passed (legacy), use it for all tiers
        if isinstance(agents, Agent):  # type: ignore[arg-type]
            self.agents = {"fast": agents, "smart": agents, "best": agents}
        self.memory = memory_store
        self._postgres = postgres_store
        self.queue: asyncio.Queue[Task] = asyncio.Queue()
        self._running = False
        self._task_count = 0
        self._success_count = 0
        # True while _process() is running — used by DiscordBot to detect concurrency
        self.is_busy = False

    @property
    def agent(self) -> Agent:  # type: ignore[type-arg]
        """Default agent (smart tier) — used by reflection passes."""
        return self.agents.get("smart") or next(iter(self.agents.values()))

    async def enqueue(self, task: Task) -> None:
        """Add a task to the processing queue."""
        await self.queue.put(task)

    async def run_forever(self) -> None:
        """Process tasks from the queue indefinitely."""
        self._running = True
        log.info("loop_started", agent=settings.agent_name)

        while self._running:
            try:
                try:
                    task = await asyncio.wait_for(self.queue.get(), timeout=settings.heartbeat_seconds)
                except asyncio.TimeoutError:
                    await self._heartbeat()
                    continue

                self.is_busy = True
                try:
                    result = await self._process(task)
                finally:
                    self.is_busy = False
                self.queue.task_done()

                if self.memory:
                    await self.memory.record_task(task, result)

                # Post-task reflection (non-blocking, best-effort)
                asyncio.create_task(self._reflect(task, result))

            except asyncio.CancelledError:
                break
            except Exception:
                log.error("loop_unhandled_exception", exc=traceback.format_exc())

        log.info("loop_stopped")

    def stop(self) -> None:
        self._running = False

    async def run_once(self, content: str, source: str = "api") -> TaskResult:
        """Run a single task synchronously (useful for testing/CLI)."""
        task = Task(content=content, source=source)
        return await self._process(task)

    async def _summarize_context(self, task: Task, accumulated_prompt: str) -> str:
        """
        Ask the fast agent to compress accumulated tool-call history into a
        brief checkpoint summary. Called when the context window overflows.
        Returns the summary string (never raises).

        NOTE: runs WITHOUT opening MCP servers — summarization is text-only
        and we're already inside an outer run_mcp_servers() context.
        """
        log.warning("context_overflow_summarizing", task=task.content[:80])
        try:
            summarize_prompt = (
                "The following is the accumulated progress log for an ongoing task. "
                "Compress it into a concise bullet-point summary (max 1500 chars) that captures: "
                "what has been done, what was found/discovered, and what still needs to happen. "
                "Keep all key facts, file paths, error messages, and decisions. "
                "Discard verbose tool output and redundant steps.\n\n"
                f"TASK: {task.content[:300]}\n\n"
                f"ACCUMULATED CONTEXT (truncated for summarization):\n{accumulated_prompt[-20_000:]}"
            )
            fast_agent = self.agents.get("fast", self.agent)
            # Run without MCP servers — summarization needs no browser tools
            # and we're already inside an outer run_mcp_servers() context.
            summary_result = await fast_agent.run(summarize_prompt)
            summary = str(summary_result.output).strip()
            log.info("context_summarized", summary_len=len(summary))
            return summary
        except Exception:
            log.warning("context_summarize_failed", exc=traceback.format_exc())
            return f"Task in progress: {task.content[:300]}\n(Context was compressed due to length; some history may be lost.)"

    async def _process(self, task: Task) -> TaskResult:
        """
        Core observe→think→act cycle using agent.run_stream_events() for full streaming.

        All events are emitted via the EventBridge singleton. The loop itself has
        no knowledge of Discord — sinks registered on the bridge handle delivery.
        """
        self._task_count += 1
        start = asyncio.get_event_loop().time()

        # Parse user model override (/fast, /smart, /best) and classify tier
        content, forced_tier = _parse_override(task.content)
        tier = forced_tier or _classify_tier(content)
        agent = self.agents.get(tier, self.agent)

        # Rebuild task with cleaned content
        task = Task(
            content=content,
            source=task.source,
            author=task.author,
            channel_id=task.channel_id,
            message_id=task.message_id,
            metadata=task.metadata,
            created_at=task.created_at,
            inject_queue=task.inject_queue,
        )

        log.info(
            "task_start",
            n=self._task_count,
            tier=tier,
            forced=forced_tier is not None,
            source=task.source,
            author=task.author,
            content=task.content[:120],
        )

        # Expire stale task journal — clear if last modified more than 30 minutes ago.
        # Short window prevents old context bleeding into unrelated tasks.
        _journal_path = settings.workspace_path / ".task_journal.md"
        try:
            if _journal_path.exists():
                import os as _os
                age_s = time.time() - _journal_path.stat().st_mtime
                if age_s > 1800:  # 30 minutes
                    _journal_path.unlink()
                    log.info("task_journal_expired", age_s=round(age_s))
        except Exception:
            pass

        await bridge.emit(TaskStartEvent(content=task.content, tier=tier))

        # Broadcast task start to Postgres audit log (shared agent activity stream)
        if self._postgres is not None:
            try:
                await self._postgres.log_task_start(task.content, task.source, tier)
            except Exception:
                pass

        # Save the incoming user message to conversation history
        if self.memory and hasattr(self.memory, "save_message") and task.source == "discord":
            try:
                await self.memory.save_message(
                    role="user",
                    content=task.content,
                    channel_id=task.channel_id,
                    metadata={"author": task.author},
                )
            except Exception:
                pass

        # Surface relevant past lessons
        lessons_context = ""
        if self.memory and hasattr(self.memory, "search_lessons"):
            try:
                lessons_context = await self.memory.search_lessons(task.content[:200], limit=3)
            except Exception:
                pass

        # Inject recent channel history — use SQLite DB when available, else fall back to Discord API
        channel_context = ""
        if task.source == "discord" and task.channel_id:
            try:
                # Prefer structured DB history (already saved on each task)
                if self.memory and hasattr(self.memory, "get_history"):
                    history_rows = await self.memory.get_history(task.channel_id, limit=10)
                    # Exclude the current message (last row) — already in task.content
                    history_rows = history_rows[:-1] if history_rows else []
                    if history_rows:
                        lines = []
                        total_chars = 0
                        for row in history_rows:
                            role_label = "You" if row["role"] == "assistant" else row["role"].capitalize()
                            text = row["content"][:300] + ("…" if len(row["content"]) > 300 else "")
                            line = f"{role_label}: {text}"
                            if total_chars + len(line) > 1500:
                                break
                            lines.append(line)
                            total_chars += len(line)
                        if lines:
                            channel_context = "## Recent conversation history\n" + "\n".join(lines)
                else:
                    # Fallback: live Discord read
                    from agent.tools.discord_tools import discord_read
                    raw = await discord_read(task.channel_id, limit=7)
                    lines = raw.splitlines()
                    if lines:
                        lines = lines[:-1]
                    truncated = []
                    total_chars = 0
                    for line in lines:
                        short = line[:300] + ("…" if len(line) > 300 else "")
                        if total_chars + len(short) > 1500:
                            break
                        truncated.append(short)
                        total_chars += len(short)
                    if truncated:
                        channel_context = "## Recent conversation history\n" + "\n".join(truncated)
            except Exception:
                pass

        parts: list[str] = []
        if channel_context:
            parts.append(channel_context)
        if lessons_context:
            parts.append(lessons_context)
        parts.append(task.content)
        base_prompt = "\n\n---\n\n".join(parts)

        try:
            result_output, tool_calls, discord_replied = await self._run_with_streaming(
                task=task,
                agent=agent,
                base_prompt=base_prompt,
                tier=tier,
            )

            elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
            elapsed_s = elapsed_ms / 1000

            log.info(
                "task_done",
                n=self._task_count,
                tier=tier,
                elapsed_ms=round(elapsed_ms),
                output_len=len(result_output),
                tool_calls=tool_calls,
            )

            await bridge.emit(TaskDoneEvent(
                output=result_output,
                elapsed_s=elapsed_s,
                tool_calls=tool_calls,
            ))

            # Broadcast task done to Postgres audit log
            if self._postgres is not None:
                try:
                    await self._postgres.log_task_done(task.content, True, elapsed_ms, tool_calls)
                except Exception:
                    pass

            self._success_count += 1

            # Save the assistant's reply to conversation history
            if self.memory and hasattr(self.memory, "save_message") and task.source == "discord":
                try:
                    await self.memory.save_message(
                        role="assistant",
                        content=result_output,
                        channel_id=task.channel_id,
                    )
                except Exception:
                    pass

            return TaskResult(
                task=task,
                output=result_output,
                discord_replied=discord_replied,
                success=True,
                elapsed_ms=elapsed_ms,
                tool_calls=tool_calls,
            )

        except Exception as exc:
            elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
            log.error("task_failed", error=str(exc), exc=traceback.format_exc())

            await bridge.emit(TaskErrorEvent(error=str(exc)[:400]))

            exc_str = str(exc)
            if "429" in exc_str:
                # Rate limit: preserve journal so the agent can resume
                try:
                    from datetime import datetime as _dt
                    journal = settings.workspace_path / ".task_journal.md"
                    ts = _dt.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                    note = (
                        f"\n### [{ts}] — INTERRUPTED BY RATE LIMIT\n"
                        f"Task: {task.content[:300]}\n"
                        f"Error: {str(exc)[:200]}\n"
                        f"Resume this task by calling task_resume() to see progress so far.\n"
                    )
                    journal.parent.mkdir(parents=True, exist_ok=True)
                    with journal.open("a", encoding="utf-8") as f:
                        f.write(note)
                except Exception:
                    pass
            else:
                # Hard crash (tool limit, EOF, etc.) — clear the journal so it
                # doesn't bleed stale context into the next unrelated task.
                try:
                    journal = settings.workspace_path / ".task_journal.md"
                    if journal.exists():
                        journal.unlink()
                        log.info("task_journal_cleared_on_crash", error=exc_str[:80])
                except Exception:
                    pass

            return TaskResult(
                task=task,
                output=f"Error: {exc}",
                success=False,
                elapsed_ms=elapsed_ms,
            )

        finally:
            pass

    async def _run_with_streaming(
        self,
        task: Task,
        agent: Agent,  # type: ignore[type-arg]
        base_prompt: str,
        tier: str,
        message_history: list | None = None,
    ) -> tuple[str, int, bool]:
        """
        Run the agent with full event streaming via the EventBridge.

        Returns (final_output_str, tool_call_count, discord_replied).

        Translates raw pydantic-ai events into typed AgentEvents:
          - ThinkingPartDelta → ThinkingDeltaEvent (per token)
          - PartEndEvent      → ThinkingEndEvent / TextTurnEndEvent
          - FinalResultEvent  → TextTurnEndEvent(is_final=True)
          - FunctionToolCallEvent  → ToolCallStartEvent
          - FunctionToolResultEvent → ToolResultEvent

        Handles context overflow (max 2 retries with summarization) and
        rate-limit 429s (max 3 retries with exponential backoff).

        After the run, drains inject_queue and recursively runs a follow-up
        if messages arrived while the agent was working (zipper-merge).
        """
        import json as _json

        prompt = base_prompt
        _context_retries = 0
        _max_context_retries = 2
        _rate_retries = 0
        _max_rate_retries = 3
        _rate_delay = 5.0
        _tool_call_total = 0  # cumulative across retries for iteration cap

        while True:  # retry loop for context overflow and rate limits
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
                        # ── Thinking delta ─────────────────────────────────────
                        if isinstance(event, PartDeltaEvent) and isinstance(
                            event.delta, ThinkingPartDelta
                        ):
                            delta = event.delta.content_delta
                            thinking_buf.append(delta)
                            await bridge.emit(ThinkingDeltaEvent(delta=delta))

                        # ── Text delta ──────────────────────────────────────────
                        elif isinstance(event, PartDeltaEvent) and isinstance(
                            event.delta, TextPartDelta
                        ):
                            delta = event.delta.content_delta
                            text_buf.append(delta)
                            await bridge.emit(TextDeltaEvent(delta=delta))

                        # ── Final result marker ─────────────────────────────────
                        elif isinstance(event, FinalResultEvent):
                            is_final = True
                            result_output = str(event.output) if hasattr(event, "output") else ""

                        # ── Part ended — flush buffers ──────────────────────────
                        elif isinstance(event, PartEndEvent):
                            if thinking_buf:
                                full_thinking = "".join(thinking_buf).strip()
                                thinking_buf = []
                                if full_thinking:
                                    await bridge.emit(ThinkingEndEvent(text=full_thinking))
                            elif text_buf:
                                full_text = "".join(text_buf).strip()
                                text_buf = []
                                if full_text:
                                    await bridge.emit(TextTurnEndEvent(
                                        text=full_text,
                                        is_final=is_final,
                                    ))

                        # ── Tool call started ───────────────────────────────────
                        elif isinstance(event, FunctionToolCallEvent):
                            tool_name = event.part.tool_name
                            tool_calls += 1
                            _tool_call_total += 1
                            if tool_name == "send_discord":
                                discord_replied = True

                            # Guard against infinite tool-call loops
                            if _tool_call_total > settings.max_loop_iterations:
                                raise RuntimeError(
                                    f"Tool call limit reached ({settings.max_loop_iterations}). "
                                    "Task aborted to prevent infinite loop."
                                )

                            # Parse args — may be dict or JSON string
                            raw_args = event.part.args
                            if isinstance(raw_args, str):
                                try:
                                    parsed_args = _json.loads(raw_args)
                                except Exception:
                                    parsed_args = raw_args
                            else:
                                parsed_args = raw_args

                            await bridge.emit(ToolCallStartEvent(
                                tool_name=tool_name,
                                call_id=event.part.tool_call_id,
                                args=parsed_args,
                            ))

                        # ── Tool result returned ────────────────────────────────
                        elif isinstance(event, FunctionToolResultEvent):
                            result_str = ""
                            # FunctionToolResultEvent.result is a ToolReturnPart
                            ret = getattr(event, "result", None)
                            if ret is not None:
                                result_str = str(getattr(ret, "content", ret))
                            await bridge.emit(ToolResultEvent(
                                tool_name=getattr(event, "tool_name", ""),
                                call_id=getattr(ret, "tool_call_id", ""),
                                result=result_str[:500],
                            ))

                # ── Drain inject_queue: zipper-merge ────────────────────────────
                if task.inject_queue and not task.inject_queue.empty():
                    injected: list[str] = []
                    while not task.inject_queue.empty():
                        try:
                            injected.append(task.inject_queue.get_nowait())
                        except asyncio.QueueEmpty:
                            break

                    if injected:
                        count = len(injected)
                        combined = "\n".join(f"[{i+1}] {m}" for i, m in enumerate(injected))
                        injection_text = (
                            f"## New message{'s' if count > 1 else ''} received while you were working:\n"
                            f"{combined}\n\n"
                            f"Please incorporate {'these' if count > 1 else 'this'} into your work "
                            f"and continue."
                        )
                        log.info("messages_injected", count=count)
                        await bridge.emit(ProgressEvent(
                            message=(
                                f"💬 Got {'your messages' if count > 1 else 'your message'} — "
                                f"{'folding them' if count > 1 else 'folding it'} in now."
                            )
                        ))
                        followup_output, followup_tool_calls, followup_replied = await self._run_with_streaming(
                            task=task,
                            agent=agent,
                            base_prompt=injection_text,
                            tier=tier,
                            message_history=None,
                        )
                        result_output = followup_output
                        tool_calls += followup_tool_calls
                        discord_replied = discord_replied or followup_replied

                return result_output, tool_calls, discord_replied

            except Exception as _exc:
                exc_str = str(_exc)
                is_context_overflow = (
                    "prompt is too long" in exc_str
                    or ("400" in exc_str and "maximum" in exc_str)
                )
                is_rate_limit = "429" in exc_str
                # Truncated tool-call args in message history: pydantic-ai can't
                # re-serialize a ToolCallPart whose args JSON was cut off mid-stream.
                # Fix: drop the message history and retry from scratch.
                is_bad_args = (
                    "EOF while parsing" in exc_str
                    or "args_as_dict" in exc_str
                )

                if is_context_overflow and _context_retries < _max_context_retries:
                    _context_retries += 1
                    log.warning("context_overflow", context_retry=_context_retries, tier=tier)
                    await bridge.emit(ProgressEvent(
                        message=(
                            f"📦 Context window full — compressing and continuing… "
                            f"(attempt {_context_retries}/{_max_context_retries})"
                        )
                    ))
                    summary = await self._summarize_context(task, prompt)
                    try:
                        journal = settings.workspace_path / ".task_journal.md"
                        from datetime import datetime as _dt
                        ts = _dt.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                        journal.parent.mkdir(parents=True, exist_ok=True)
                        with journal.open("a", encoding="utf-8") as f:
                            f.write(
                                f"\n### [{ts}] — CONTEXT COMPRESSED (retry {_context_retries})\n"
                                f"{summary}\n"
                            )
                    except Exception:
                        pass
                    prompt = (
                        f"{base_prompt}\n\n"
                        f"## Progress so far (summarized — context was compressed):\n{summary}"
                    )
                    continue

                elif is_rate_limit and _rate_retries < _max_rate_retries:
                    _rate_retries += 1
                    log.warning("rate_limit_retry", attempt=_rate_retries, wait_s=_rate_delay)
                    await bridge.emit(ProgressEvent(
                        message=f"⏸️ Rate limited — retrying in {int(_rate_delay)}s…"
                    ))
                    await asyncio.sleep(_rate_delay)
                    _rate_delay *= 2
                    continue

                elif is_bad_args:
                    log.warning("truncated_tool_args_retry", error=exc_str[:200])
                    await bridge.emit(ProgressEvent(
                        message="⚠️ Tool args got truncated — retrying from last checkpoint…"
                    ))
                    # Drop message history so pydantic-ai starts fresh on the next turn
                    message_history = None
                    continue

                else:
                    raise

    async def _reflect(self, task: Task, result: TaskResult) -> None:
        """
        Post-task reflection: learn from failures and extract insights from successes.

        On failure  → ask the agent what went wrong + save a MISTAKE lesson
        On success with >3 tool calls → extract a reusable pattern/insight
        Every MEMORY_UPDATE_INTERVAL successes → rewrite MEMORY.md
        """
        if not self.memory:
            return

        try:
            if not result.success:
                await self._reflect_on_failure(task, result)
            else:
                # Extract insight from complex successful tasks (>3 tool calls)
                if result.tool_calls > 3:
                    await self._reflect_on_success(task, result)
                if self._success_count % MEMORY_UPDATE_INTERVAL == 0:
                    await self._update_memory_md()
        except Exception:
            log.warning("reflect_error", exc=traceback.format_exc())

    async def _reflect_on_success(self, task: Task, result: TaskResult) -> None:
        """Extract a reusable pattern or insight from a successful complex task."""
        log.info("reflecting_on_success", task=task.content[:80], tool_calls=result.tool_calls)

        reflection_prompt = (
            f"You just successfully completed the following task using {result.tool_calls} tool calls.\n\n"
            f"Task: {task.content}\n\n"
            f"Result summary: {result.output[:500]}\n\n"
            f"In one concise sentence (max 200 chars), extract ONE reusable pattern, shortcut, or insight "
            f"from this task that would help you do similar tasks better or faster in the future. "
            f"Focus on what was surprising, efficient, or non-obvious. "
            f"If there is nothing genuinely useful to record, reply with exactly: NOTHING_TO_RECORD"
        )

        try:
            fast_agent = self.agents.get("fast", self.agent)
            insight_result = await fast_agent.run(reflection_prompt)
            insight = str(insight_result.output).strip()

            if insight and "NOTHING_TO_RECORD" not in insight:
                await self.memory.save_lesson(
                    summary=insight[:300],
                    kind="pattern",
                    context=task.content[:300],
                )
                log.info("pattern_saved", insight=insight[:100])
        except Exception:
            log.warning("reflect_on_success_error", exc=traceback.format_exc())

    async def _reflect_on_failure(self, task: Task, result: TaskResult) -> None:
        """Ask the agent to diagnose a failure and record the lesson."""
        log.info("reflecting_on_failure", task=task.content[:80])

        reflection_prompt = (
            f"You just attempted the following task and it FAILED.\n\n"
            f"Task: {task.content}\n\n"
            f"Error/output: {result.output}\n\n"
            f"In 1-2 sentences, diagnose what went wrong and state what you should do differently next time. "
            f"Then call `memory_save` with the lesson prefixed with 'MISTAKE: ' and call `skill_edit` "
            f"to update any relevant skill if the mistake was procedural. "
            f"Be concise and specific."
        )

        try:
            # Run without MCP servers — reflection is text+memory only, no browser needed
            reflection = await self.agent.run(reflection_prompt)
            lesson = str(reflection.output).strip()

            await self.memory.save_lesson(
                summary=lesson,
                kind="mistake",
                context=task.content[:300],
            )

            await self._update_memory_md()

            log.info("lesson_saved", kind="mistake", lesson=lesson[:100])

        except Exception:
            log.warning("reflect_on_failure_error", exc=traceback.format_exc())

    async def _update_memory_md(self) -> None:
        """Periodically rewrite the Lessons section of MEMORY.md with recent lessons."""
        log.info("updating_memory_md")
        try:
            recent = await self.memory.get_recent_lessons(limit=20)
            memory_path = settings.identity_path / "MEMORY.md"
            if not memory_path.exists():
                return

            content = memory_path.read_text(encoding="utf-8")

            MARKER = "## Recent Lessons"
            if MARKER in content:
                before = content[:content.index(MARKER)]
                content = before.rstrip()
            else:
                content = content.rstrip()

            from datetime import datetime as dt
            now = dt.now().strftime("%Y-%m-%d %H:%M")
            content += f"\n\n{MARKER}\n_Last updated: {now}_\n\n{recent}\n"
            memory_path.write_text(content, encoding="utf-8")
            log.info("memory_md_updated")

        except Exception:
            log.warning("update_memory_md_error", exc=traceback.format_exc())

    async def _heartbeat(self) -> None:
        """Periodic background work: update Postgres presence, checkpoint SQLite, poll A2A tasks."""
        log.debug("heartbeat", agent=settings.agent_name, queue_size=self.queue.qsize())
        if self.memory and hasattr(self.memory, "heartbeat"):
            await self.memory.heartbeat()

        # ── A2A task polling ────────────────────────────────────────────────────
        # If Postgres is available, check for tasks delegated to us by other agents.
        # Enqueue any pending tasks so they're processed without requiring a human ping.
        if self._postgres is not None and not self.is_busy:
            try:
                rows = await self._postgres.get_pending_task_rows()
                for row in rows:
                    task_id = row["id"]
                    description = row["description"]
                    from_agent = row.get("from_agent", "unknown")
                    log.info("a2a_task_received", id=task_id[:8], from_agent=from_agent)
                    # Mark as running so we don't pick it up again
                    await self._postgres.mark_task_running(task_id)
                    await self.enqueue(Task(
                        content=description,
                        source="a2a",
                        author=from_agent,
                        metadata={"task_id": task_id, "from_agent": from_agent},
                    ))
            except Exception as exc:
                log.warning("a2a_poll_error", error=str(exc))
