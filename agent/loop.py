"""
Main agent loop: queue tasks, stream events, persist results, and reflect.

Tasks can come from Discord, heartbeats, or direct API calls. The loop routes
each task to a model tier, emits typed events through `agent.events.bridge`,
and folds in queued follow-up messages via `Task.inject_queue`.
"""

from __future__ import annotations

import asyncio
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from agent.config import settings
from agent.events import (
    TaskDoneEvent,
    TaskErrorEvent,
    TaskStartEvent,
    bridge,
)
from agent.loop_services import (
    HeartbeatService,
    ReflectionService,
    RunExecutor,
    TaskContextBuilder,
    TaskJournal,
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
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # Queue for zipper-merge injection: new user messages arriving mid-task
    # are pushed here by DiscordBot and drained after each run completes.
    inject_queue: asyncio.Queue[str] | None = field(default=None, repr=False)
    response_future: asyncio.Future[TaskResult] | None = field(default=None, repr=False)


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
        self._journal = TaskJournal(settings.workspace_path)
        self._context_builder = TaskContextBuilder(self.memory)
        self._run_executor = RunExecutor(
            event_bridge=bridge,
            journal=self._journal,
            summarize_context=self._summarize_context,
        )
        self._reflection_service = ReflectionService(
            agents=self.agents,
            memory_store=self.memory,
            journal=self._journal,
        )
        self._heartbeat_service = HeartbeatService(
            memory_store=self.memory,
            postgres_store=self._postgres,
            enqueue=self.enqueue,
        )

    @property
    def agent(self) -> Agent:  # type: ignore[type-arg]
        """Default agent (smart tier) — used by reflection passes."""
        return self.agents.get("smart") or next(iter(self.agents.values()))

    async def enqueue(self, task: Task) -> None:
        """Add a task to the processing queue."""
        await self.queue.put(task)

    @property
    def has_pending_work(self) -> bool:
        """True when work is running or already queued."""
        return self.is_busy or not self.queue.empty()

    async def _execute_task(self, task: Task) -> TaskResult:
        result = await self._process(task)

        if self.memory:
            await self.memory.record_task(task, result)

        # Post-task reflection (non-blocking, best-effort)
        asyncio.create_task(self._reflect(task, result))

        if task.response_future and not task.response_future.done():
            task.response_future.set_result(result)

        return result

    async def run_forever(self) -> None:
        """Process tasks from the queue indefinitely."""
        self._running = True
        log.info("loop_started", agent=settings.agent_name)

        while self._running:
            try:
                try:
                    task = await asyncio.wait_for(self.queue.get(), timeout=settings.heartbeat_seconds)
                except TimeoutError:
                    await self._heartbeat()
                    continue

                self.is_busy = True
                try:
                    await self._execute_task(task)
                finally:
                    self.is_busy = False
                self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                if 'task' in locals() and task.response_future and not task.response_future.done():
                    task.response_future.set_exception(exc)
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
            summary_result = await fast_agent.run(summarize_prompt, usage_limits=UsageLimits(request_limit=None))
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
        start = asyncio.get_running_loop().time()

        # Parse user model override (/fast, /smart, /best) and classify tier
        content, forced_tier = _parse_override(task.content)
        tier = forced_tier or _classify_tier(content)
        agent = self.agents.get(tier, self.agent)
        task, tier, base_prompt = await self._context_builder.build(task)

        log.info(
            "task_start",
            n=self._task_count,
            tier=tier,
            forced=forced_tier is not None,
            source=task.source,
            author=task.author,
            content=task.content[:120],
        )

        # Expire stale task journal — short window prevents old context bleeding across tasks.
        self._journal.expire_stale()

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

        try:
            result_output, tool_calls, discord_replied = await self._run_executor.run(
                task=task,
                agent=agent,
                base_prompt=base_prompt,
                tier=tier,
            )

            elapsed_ms = (asyncio.get_running_loop().time() - start) * 1000
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
            elapsed_ms = (asyncio.get_running_loop().time() - start) * 1000
            log.error("task_failed", error=str(exc), exc=traceback.format_exc())

            await bridge.emit(TaskErrorEvent(error=str(exc)[:400]))

            exc_str = str(exc)
            if "429" in exc_str:
                self._journal.append(
                    "INTERRUPTED BY RATE LIMIT",
                    (
                        f"Task: {task.content[:300]}\n"
                        f"Error: {str(exc)[:200]}\n"
                        "Resume this task by calling task_resume() to see progress so far."
                    ),
                )
            else:
                self._journal.clear()
                log.info("task_journal_cleared_on_crash", error=exc_str[:80])

            return TaskResult(
                task=task,
                output=f"Error: {exc}",
                success=False,
                elapsed_ms=elapsed_ms,
            )

    async def _run_with_streaming(
        self,
        task: Task,
        agent: Agent,  # type: ignore[type-arg]
        base_prompt: str,
        tier: str,
        message_history: list | None = None,
    ) -> tuple[str, int, bool]:
        return await self._run_executor.run(
            task=task,
            agent=agent,
            base_prompt=base_prompt,
            tier=tier,
            message_history=message_history,
        )

    async def _reflect(self, task: Task, result: TaskResult) -> None:
        await self._reflection_service.reflect(
            task,
            result,
            self._success_count,
            MEMORY_UPDATE_INTERVAL,
        )

    async def _reflect_on_success(self, task: Task, result: TaskResult) -> None:
        await self._reflection_service._reflect_on_success(task, result)

    async def _reflect_on_failure(self, task: Task, result: TaskResult) -> None:
        await self._reflection_service._reflect_on_failure(task, result)

    async def _update_memory_md(self) -> None:
        await self._reflection_service.update_memory_md()

    async def _heartbeat(self) -> None:
        """Periodic background work: update presence, checkpoint SQLite, poll A2A tasks."""
        log.debug("heartbeat", agent=settings.agent_name, queue_size=self.queue.qsize())
        await self._heartbeat_service.heartbeat(is_busy=self.is_busy)
