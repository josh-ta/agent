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

Context management
------------------
When a task's accumulated tool call history exceeds the model's context window
(~200k tokens for Claude), the loop catches the 400 "prompt is too long" error,
asks the fast agent to compress all progress into a brief summary, saves it to
the task journal, then retries with that compressed context instead of raw history.
This mirrors how Cursor/Claude Code handle long tasks: periodic summarization.

Progress updates
----------------
For Discord tasks, the loop sends an immediate acknowledgment when the task
starts, periodic "still working" pings every 60 s, and a note whenever the
agent calls task_note() (forwarded from the journal write hook).

Zipper-merge injection (Option C)
----------------------------------
While the agent is mid-task, new messages from the user are held in
Task.inject_queue. Between each tool-call round-trip (after CallToolsNode
completes, before ModelRequestNode sends its next request), the loop drains
inject_queue and appends the messages as a UserPromptPart so the model sees
them at exactly the right moment — between thoughts, not before or after.
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
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai import FinalResultEvent, PartDeltaEvent, PartEndEvent, TextPartDelta, ThinkingPartDelta

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
        Core observe→think→act cycle using agent.iter() for step-by-step control.

        Drives the pydantic-ai graph node-by-node so we can:
          - Inject queued user messages between tool-call rounds (zipper merge)
          - Detect context overflow at each ModelRequestNode and compress
          - Forward task_note tool calls to Discord as they happen (not post-hoc)
        """
        self._task_count += 1
        start = asyncio.get_event_loop().time()

        # Parse user model override (/fast, /smart, /best) and classify tier
        content, forced_tier = _parse_override(task.content)
        tier = forced_tier or _classify_tier(content)
        agent = self.agents.get(tier, self.agent)

        # Preserve progress_callback and inject_queue across the Task rebuild
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
        prompt = base_prompt

        # Progress ticker
        ticker: asyncio.Task | None = None  # type: ignore[type-arg]
        if task.progress_callback:
            ticker = asyncio.create_task(self._progress_ticker(task, start))

        try:
            _context_retries = 0
            _max_context_retries = 2
            _rate_retries = 0
            _max_rate_retries = 3
            _rate_delay = 5.0

            while True:  # outer retry loop for context overflow + rate limits
                discord_replied = False
                tool_calls = 0

                try:
                    async with agent.run_mcp_servers():
                        async with agent.iter(prompt) as agent_run:
                            node = agent_run.next_node

                            while not Agent.is_end_node(node):

                                # ── ModelRequestNode: stream reasoning to Discord ──
                                if Agent.is_model_request_node(node):
                                    async with node.stream(agent_run.ctx) as request_stream:
                                        thinking_buf: list[str] = []
                                        text_buf: list[str] = []
                                        is_final = False
                                        async for event in request_stream:
                                            if isinstance(event, PartDeltaEvent):
                                                if isinstance(event.delta, ThinkingPartDelta):
                                                    thinking_buf.append(event.delta.content_delta)
                                                elif isinstance(event.delta, TextPartDelta):
                                                    text_buf.append(event.delta.content_delta)
                                            elif isinstance(event, FinalResultEvent):
                                                # Mark final *before* PartEndEvent so the flush
                                                # below correctly suppresses the last text block
                                                # (that block becomes the Discord reply).
                                                is_final = True
                                            elif isinstance(event, PartEndEvent):
                                                # Flush completed thinking block immediately
                                                if thinking_buf:
                                                    _thinking = "".join(thinking_buf).strip()
                                                    if _thinking:
                                                        await self._send_progress(
                                                            task, f"🧠 *{_thinking[:1800]}*"
                                                        )
                                                    thinking_buf = []
                                                # Flush completed intermediate text turns.
                                                # Skip the final answer — it's sent as the reply.
                                                elif text_buf and not is_final:
                                                    _text = "".join(text_buf).strip()
                                                    if _text:
                                                        await self._send_progress(
                                                            task, f"💭 {_text[:1800]}"
                                                        )
                                                    text_buf = []

                                # Drive the node forward
                                next_node = await agent_run.next(node)

                                # ── After CallToolsNode finishes: inspect results ──
                                if Agent.is_call_tools_node(node):
                                    tool_calls += 1

                                    # Scan the model response parts for tool calls
                                    # we care about (send_discord, task_note)
                                    for part in getattr(node.model_response, "parts", []):
                                        tool_name = getattr(part, "tool_name", None)
                                        if tool_name == "send_discord":
                                            discord_replied = True
                                        elif tool_name == "task_note":
                                            args = getattr(part, "args", {})
                                            note = (
                                                args.get("note", "")
                                                if isinstance(args, dict)
                                                else str(args)
                                            )
                                            if note:
                                                # Strip JSON wrapper if agent passed a JSON string
                                                import json as _json
                                                try:
                                                    parsed = _json.loads(note)
                                                    if isinstance(parsed, dict):
                                                        note = parsed.get("note", note)
                                                    elif isinstance(parsed, str):
                                                        note = parsed
                                                except Exception:
                                                    pass
                                                await self._send_progress(
                                                    task, f"📝 **Checkpoint:** {note[:400]}"
                                                )

                                    # Drain inject_queue and fold pending messages
                                    # into the run's message history as a UserPromptPart
                                    if task.inject_queue and not task.inject_queue.empty():
                                        injected: list[str] = []
                                        while not task.inject_queue.empty():
                                            try:
                                                injected.append(task.inject_queue.get_nowait())
                                            except asyncio.QueueEmpty:
                                                break

                                        if injected:
                                            combined = "\n".join(
                                                f"[{i+1}] {m}" for i, m in enumerate(injected)
                                            )
                                            injection_text = (
                                                f"\n\n---\n## New instructions received while working:\n"
                                                f"{combined}\n"
                                                f"Please incorporate these into your ongoing work."
                                            )
                                            log.info(
                                                "messages_injected",
                                                count=len(injected),
                                                chars=len(injection_text),
                                            )
                                            await self._send_progress(
                                                task,
                                                f"💬 Got your message"
                                                f"{' (×'+str(len(injected))+')' if len(injected) > 1 else ''}"
                                                f" — incorporating into current work.",
                                            )
                                            # Append as a user-turn message so the model
                                            # sees it on its next request
                                            agent_run.ctx.state.message_history.append(
                                                ModelRequest(parts=[
                                                    UserPromptPart(content=injection_text)
                                                ])
                                            )

                                node = next_node

                            # Ran to End — grab result
                            result_output = str(agent_run.result.output) if agent_run.result else ""
                            break  # success, exit outer retry loop

                except Exception as _exc:
                    exc_str = str(_exc)
                    is_context_overflow = (
                        "prompt is too long" in exc_str
                        or ("400" in exc_str and "maximum" in exc_str)
                    )
                    is_rate_limit = "429" in exc_str

                    if is_context_overflow and _context_retries < _max_context_retries:
                        _context_retries += 1
                        log.warning(
                            "context_overflow",
                            context_retry=_context_retries,
                            tier=tier,
                        )
                        await self._send_progress(
                            task,
                            f"📦 Context window full — compressing progress and continuing… "
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
                        continue  # retry outer loop with compressed prompt

                    elif is_rate_limit and _rate_retries < _max_rate_retries:
                        _rate_retries += 1
                        log.warning(
                            "rate_limit_retry",
                            attempt=_rate_retries,
                            wait_s=_rate_delay,
                            tier=tier,
                        )
                        await self._send_progress(
                            task, f"⏸️ Rate limited — retrying in {int(_rate_delay)}s…"
                        )
                        await asyncio.sleep(_rate_delay)
                        _rate_delay *= 2
                        continue  # retry outer loop

                    else:
                        raise

            elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
            log.info(
                "task_done",
                n=self._task_count,
                tier=tier,
                elapsed_ms=round(elapsed_ms),
                output_len=len(result_output),
                discord_replied=discord_replied,
                tool_calls=tool_calls,
            )

            self._success_count += 1
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
