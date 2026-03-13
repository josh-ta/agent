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
across the entire run (multiple model turns + all tool calls). This gives us:
  - ThinkingPartDelta  → 🧠 thinking blocks streamed to Discord in real time
  - TextPartDelta      → 💭 intermediate reasoning turns (not the final reply)
  - FunctionToolCallEvent → 🔧 tool call notifications
  - FinalResultEvent   → marks the last text block (suppressed — sent as reply)

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
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

import structlog
from pydantic_ai import Agent
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

log = structlog.get_logger()

# Reflect on MEMORY.md every N successful tasks
MEMORY_UPDATE_INTERVAL = 10

# How often to send a "still working" ping for long tasks (seconds)
PROGRESS_PING_INTERVAL = 60


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
    # Optional callback for mid-task progress messages (set by discord_bot)
    progress_callback: Callable[[str], Awaitable[None]] | None = field(default=None, repr=False)
    # Queue for zipper-merge injection: new user messages arriving mid-task
    # are pushed here by DiscordBot and drained between tool-call rounds.
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
    """

    def __init__(self, agents: dict[str, Agent], memory_store: Any = None) -> None:  # type: ignore[type-arg]
        # agents dict: {"fast": Agent, "smart": Agent, "best": Agent}
        self.agents = agents
        # Fallback: if only one agent passed (legacy), use it for all tiers
        if isinstance(agents, Agent):  # type: ignore[arg-type]
            self.agents = {"fast": agents, "smart": agents, "best": agents}
        self.memory = memory_store
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

    async def _send_progress(self, task: Task, message: str) -> None:
        """Send a progress message via the task's callback (no-op if not set)."""
        if task.progress_callback:
            try:
                await task.progress_callback(message)
            except Exception:
                pass

    async def _progress_ticker(self, task: Task, start: float) -> None:
        """
        Coroutine that sends periodic status pings after the agent has been running a while.
        Sleeps first, then fires — so the first ping only appears if the task is genuinely slow.
        start is the asyncio loop time when the task began, used to compute real elapsed time.
        """
        while True:
            await asyncio.sleep(PROGRESS_PING_INTERVAL)
            elapsed_s = asyncio.get_event_loop().time() - start
            mins = int(elapsed_s // 60)
            secs = int(elapsed_s % 60)
            time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
            await self._send_progress(task, f"⏳ Still working… ({time_str} elapsed)")

    async def _summarize_context(self, task: Task, accumulated_prompt: str) -> str:
        """
        Ask the fast agent to compress accumulated tool-call history into a
        brief checkpoint summary. Called when the context window overflows.
        Returns the summary string (never raises).
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
            async with fast_agent.run_mcp_servers():
                summary_result = await fast_agent.run(summarize_prompt)
            summary = str(summary_result.output).strip()
            log.info("context_summarized", summary_len=len(summary))
            return summary
        except Exception:
            log.warning("context_summarize_failed", exc=traceback.format_exc())
            # Return a minimal fallback summary so the task can continue
            return f"Task in progress: {task.content[:300]}\n(Context was compressed due to length; some history may be lost.)"

    async def _process(self, task: Task) -> TaskResult:
        """
        Core observe→think→act cycle using agent.run_stream_events() for full streaming.

        run_stream_events() gives a single flat async iterator over the entire run
        (multiple model turns + all tool calls), including:
          - ThinkingPartDelta  → streamed to Discord as 🧠 blocks
          - TextPartDelta      → intermediate turns sent as 💭, final reply suppressed
          - FunctionToolCallEvent → tool call notifications + task_note forwarding
          - FinalResultEvent   → marks the final answer (sent as Discord reply)

        After the run, inject_queue is drained; if messages arrived mid-task they
        are appended and a follow-up run is issued (zipper-merge).
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
            progress_callback=task.progress_callback,
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

        # Immediate acknowledgment
        if task.source == "discord" and task.channel_id:
            await self._send_progress(task, f"🔍 Working on: *{task.content}*")

        # Surface relevant past lessons
        lessons_context = ""
        if self.memory and hasattr(self.memory, "search_lessons"):
            try:
                lessons_context = await self.memory.search_lessons(task.content[:200], limit=3)
            except Exception:
                pass

        # Inject recent channel history
        channel_context = ""
        if task.source == "discord" and task.channel_id:
            try:
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

        # Progress ticker — fires every PROGRESS_PING_INTERVAL if task is still running
        ticker: asyncio.Task | None = None  # type: ignore[type-arg]
        if task.progress_callback:
            ticker = asyncio.create_task(self._progress_ticker(task, start))

        try:
            result_output = await self._run_with_streaming(
                task=task,
                agent=agent,
                base_prompt=base_prompt,
                tier=tier,
            )

            elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
            discord_replied = getattr(result_output, "_discord_replied", False)
            tool_calls = getattr(result_output, "_tool_calls", 0)
            output_str = result_output if isinstance(result_output, str) else str(result_output)

            log.info(
                "task_done",
                n=self._task_count,
                tier=tier,
                elapsed_ms=round(elapsed_ms),
                output_len=len(output_str),
            )

            self._success_count += 1
            return TaskResult(
                task=task,
                output=output_str,
                discord_replied=False,
                success=True,
                elapsed_ms=elapsed_ms,
                tool_calls=0,
            )

        except Exception as exc:
            elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
            log.error("task_failed", error=str(exc), exc=traceback.format_exc())
            await self._send_progress(task, f"❌ Task failed: {str(exc)[:200]}")

            if "429" in str(exc):
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

            return TaskResult(
                task=task,
                output=f"Error: {exc}",
                success=False,
                elapsed_ms=elapsed_ms,
            )

        finally:
            if ticker and not ticker.done():
                ticker.cancel()
                try:
                    await ticker
                except asyncio.CancelledError:
                    pass

    async def _run_with_streaming(
        self,
        task: Task,
        agent: Agent,  # type: ignore[type-arg]
        base_prompt: str,
        tier: str,
        message_history: list | None = None,
    ) -> str:
        """
        Run the agent with full event streaming, returning the final output string.

        Handles context overflow and rate-limit retries. After the run, drains
        inject_queue and recursively issues a follow-up run if messages arrived.
        """
        import json as _json

        prompt = base_prompt
        _context_retries = 0
        _max_context_retries = 2
        _rate_retries = 0
        _max_rate_retries = 3
        _rate_delay = 5.0

        while True:  # retry loop for context overflow and rate limits
            discord_replied = False
            tool_calls = 0
            result_output = ""

            try:
                async with agent.run_mcp_servers():
                    # Streaming: one flat event stream for the entire run
                    thinking_buf: list[str] = []
                    text_buf: list[str] = []
                    is_final = False

                    async for event in agent.run_stream_events(
                        prompt,
                        message_history=message_history or [],
                    ):
                        # ── Thinking delta — buffer until PartEndEvent ──────────
                        if isinstance(event, PartDeltaEvent) and isinstance(
                            event.delta, ThinkingPartDelta
                        ):
                            thinking_buf.append(event.delta.content_delta)

                        # ── Text delta — buffer until PartEndEvent ──────────────
                        elif isinstance(event, PartDeltaEvent) and isinstance(
                            event.delta, TextPartDelta
                        ):
                            text_buf.append(event.delta.content_delta)

                        # ── Final result marker — suppress last text block ───────
                        elif isinstance(event, FinalResultEvent):
                            is_final = True
                            result_output = str(event.output) if hasattr(event, "output") else ""

                        # ── Part ended — flush buffers ──────────────────────────
                        elif isinstance(event, PartEndEvent):
                            if thinking_buf:
                                _thinking = "".join(thinking_buf).strip()
                                thinking_buf = []
                                if _thinking:
                                    # Send each thinking block as its own message
                                    await self._send_progress(task, f"🧠 *{_thinking[:1900]}*")
                            elif text_buf:
                                _text = "".join(text_buf).strip()
                                text_buf = []
                                if _text and not is_final:
                                    # Intermediate reasoning turn — send as 💭
                                    await self._send_progress(task, f"💭 {_text[:1900]}")
                                # If is_final, this is the reply — discord_bot sends it

                        # ── Tool call started ───────────────────────────────────
                        elif isinstance(event, FunctionToolCallEvent):
                            tool_name = event.part.tool_name
                            tool_calls += 1
                            if tool_name == "send_discord":
                                discord_replied = True
                            elif tool_name == "task_note":
                                # Forward task_note to Discord immediately
                                raw_args = event.part.args
                                note = ""
                                if isinstance(raw_args, dict):
                                    note = raw_args.get("note", "")
                                elif isinstance(raw_args, str):
                                    try:
                                        parsed = _json.loads(raw_args)
                                        note = parsed.get("note", raw_args) if isinstance(parsed, dict) else raw_args
                                    except Exception:
                                        note = raw_args
                                if note:
                                    await self._send_progress(task, f"📝 **Checkpoint:** {note[:400]}")

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
                        await self._send_progress(
                            task,
                            f"💬 Got {'your messages' if count > 1 else 'your message'} — "
                            f"{'folding them' if count > 1 else 'folding it'} in now.",
                        )
                        # Recursively run with the injected messages as the next prompt.
                        # Pass current result as context so the model knows what was done.
                        followup_prompt = injection_text
                        followup_history = None  # run_stream_events builds history internally
                        result_output = await self._run_with_streaming(
                            task=task,
                            agent=agent,
                            base_prompt=followup_prompt,
                            tier=tier,
                            message_history=followup_history,
                        )

                return result_output

            except Exception as _exc:
                exc_str = str(_exc)
                is_context_overflow = (
                    "prompt is too long" in exc_str
                    or ("400" in exc_str and "maximum" in exc_str)
                )
                is_rate_limit = "429" in exc_str

                if is_context_overflow and _context_retries < _max_context_retries:
                    _context_retries += 1
                    log.warning("context_overflow", context_retry=_context_retries, tier=tier)
                    await self._send_progress(
                        task,
                        f"📦 Context window full — compressing and continuing… "
                        f"(attempt {_context_retries}/{_max_context_retries})",
                    )
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
                    await self._send_progress(
                        task, f"⏸️ Rate limited — retrying in {int(_rate_delay)}s…"
                    )
                    await asyncio.sleep(_rate_delay)
                    _rate_delay *= 2
                    continue

                else:
                    raise

    async def _reflect(self, task: Task, result: TaskResult) -> None:
        """
        Post-task reflection: learn from failures and extract insights.

        On failure  → ask the agent what went wrong + save a MISTAKE lesson
        On success  → every MEMORY_UPDATE_INTERVAL tasks, update MEMORY.md
        """
        if not self.memory:
            return

        try:
            if not result.success:
                await self._reflect_on_failure(task, result)
            elif self._success_count % MEMORY_UPDATE_INTERVAL == 0:
                await self._update_memory_md()
        except Exception:
            log.warning("reflect_error", exc=traceback.format_exc())

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
            async with self.agent.run_mcp_servers():
                reflection = await self.agent.run(reflection_prompt)
            lesson = str(reflection.output).strip()

            # Save to SQLite — this is the durable append-only log
            await self.memory.save_lesson(
                summary=lesson,
                kind="mistake",
                context=task.content[:300],
            )

            # Update MEMORY.md as a rolling summary (not a raw append)
            # SQLite is the log; MEMORY.md is the distilled snapshot
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

            # Replace or append the ## Lessons section
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
        """Periodic background work: update Postgres presence, checkpoint SQLite."""
        log.debug("heartbeat", agent=settings.agent_name, queue_size=self.queue.qsize())
        if self.memory and hasattr(self.memory, "heartbeat"):
            await self.memory.heartbeat()
