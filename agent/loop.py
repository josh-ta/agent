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
from typing import Any, Literal
from uuid import uuid4

import structlog
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from agent.config import settings
from agent.events import (
    TaskDoneEvent,
    TaskErrorEvent,
    TaskStartEvent,
    TaskWaitingEvent,
    bridge,
)
from agent.loop_services import (
    HeartbeatService,
    ReflectionService,
    RunResult,
    RunExecutor,
    TaskContextBuilder,
    TaskJournal,
)
from agent.tools.discord_tools import DiscordAttachment
from agent.task_waits import SuspendedTask, TaskWaitRegistry, task_wait_context

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


TaskStatus = Literal["queued", "running", "waiting_for_user", "succeeded", "failed"]


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
    success: bool | None
    elapsed_ms: float
    status: TaskStatus = "succeeded"
    tool_calls: int = 0
    answered_user: bool = False
    user_visible_reply_sent: bool = False
    waiting_for_user: bool = False
    question: str | None = None
    timeout_s: int = 300
    attachments: list[DiscordAttachment] = field(default_factory=list)


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
        self.wait_registry = TaskWaitRegistry()
        self._heartbeat_service = HeartbeatService(
            memory_store=self.memory,
            postgres_store=self._postgres,
            enqueue=self.enqueue,
            wait_registry=self.wait_registry,
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
        if task.metadata is None:
            task.metadata = {}
        task_id = str(task.metadata.get("task_id", "")).strip() if task.metadata else ""
        if not task_id:
            task_id = str(uuid4())
            task.metadata["task_id"] = task_id
        session_id = str(task.metadata.get("session_id", "")).strip()
        if not session_id:
            fallback = f"{task.source}:{task.channel_id}:{task.message_id or task_id}"
            task.metadata["session_id"] = fallback

        with bridge.task_context(task_id or None), task_wait_context(
            task_id=task_id,
            source=task.source,
            channel_id=task.channel_id,
        ):
            if self.memory and task_id and hasattr(self.memory, "mark_task_running"):
                await self.memory.mark_task_running(task_id)

            result = await self._process(task)

            if (
                self._postgres is not None
                and task.source == "a2a"
                and result.status in {"succeeded", "failed"}
            ):
                try:
                    await self._postgres.complete_task(task_id, result.output[:4000])
                except Exception:
                    log.warning("a2a_complete_failed", task_id=task_id)

            if self.memory:
                await self.memory.record_task(task, result)

            # Post-task reflection (non-blocking, best-effort)
            if result.status in {"succeeded", "failed"}:
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

    def build_resumed_task(self, *, suspended: SuspendedTask, answer: str, author: str, source: str) -> Task:
        metadata = self.wait_registry.build_resumed_metadata(
            suspended,
            answer=answer,
            resumed_from=source,
        )
        return Task(
            content=suspended.content,
            source=suspended.source if suspended.source != "api" else source,
            author=author,
            channel_id=suspended.channel_id,
            message_id=0,
            metadata=metadata,
        )

    async def restore_waiting_tasks(self) -> int:
        if self.memory is None or not hasattr(self.memory, "list_waiting_task_records"):
            return 0

        restored = 0
        rows = await self.memory.list_waiting_task_records()
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            task_id = str(row.get("task_id", "")).strip()
            wait_state = metadata.get("wait_state")
            if not task_id or not isinstance(wait_state, dict):
                continue
            question = str(wait_state.get("question", "")).strip()
            if not question:
                continue
            channel_id = self._coerce_int(wait_state.get("channel_id"))
            message_id = self._coerce_int(wait_state.get("message_id"))
            prompt_message_id = self._coerce_int_or_none(wait_state.get("prompt_message_id"))
            timeout_s = max(1, self._coerce_int(wait_state.get("timeout_s"), default=300))
            suspended = self.wait_registry.suspend(
                task_id=task_id,
                source=str(row.get("source", metadata.get("source", "api"))),
                author=str(row.get("author", metadata.get("author", "system"))),
                content=str(row.get("content", "")),
                channel_id=channel_id,
                message_id=message_id,
                metadata=metadata,
                question=question,
                timeout_s=timeout_s,
                base_prompt="",
                tier=str(metadata.get("tier", "")),
            )
            if prompt_message_id is not None:
                suspended.prompt_message_id = prompt_message_id
            restored += 1
        if restored:
            log.info("waiting_tasks_restored", count=restored)
        return restored

    async def restore_pending_tasks(self) -> int:
        if self.memory is None or not hasattr(self.memory, "list_pending_task_records"):
            return 0
        restored = 0
        rows = await self.memory.list_pending_task_records()
        for row in rows:
            task_id = str(row.get("task_id", "")).strip()
            if not task_id or self.wait_registry.has_pending(task_id):
                continue
            metadata = dict(row.get("metadata") or {})
            metadata["task_id"] = task_id
            session_id = str(metadata.get("session_id", "")).strip()
            if not session_id:
                metadata["session_id"] = f"{row.get('source', 'api')}:{task_id}"
            task = Task(
                content=str(row.get("content", "")),
                source=str(row.get("source", "api")),
                author=str(row.get("author", "system")),
                channel_id=self._coerce_int(metadata.get("channel_id")),
                message_id=self._coerce_int(metadata.get("message_id")),
                metadata=metadata,
            )
            if row.get("status") == "running" and hasattr(self.memory, "mark_task_queued"):
                await self.memory.mark_task_queued(task_id, metadata=metadata)
            await self.enqueue(task)
            restored += 1
        if restored:
            log.info("pending_tasks_restored", count=restored)
        return restored

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
        task_id = self.wait_registry.ensure_task_id(task.metadata)
        session_id = str(task.metadata.get("session_id", "")).strip()
        if not session_id:
            session_id = f"{task.source}:{task.channel_id}:{task.message_id or task_id}"
            task.metadata["session_id"] = session_id

        log.info(
            "task_start",
            n=self._task_count,
            tier=tier,
            forced=forced_tier is not None,
            source=task.source,
            author=task.author,
            content=task.content[:120],
        )

        if self.memory and hasattr(self.memory, "ensure_session"):
            try:
                await self.memory.ensure_session(
                    session_id=session_id,
                    source=task.source,
                    channel_id=task.channel_id,
                    title=task.content[:120],
                    status="active",
                    pending_task_id=task_id,
                    metadata={"author": task.author},
                )
                await self.memory.save_task_checkpoint(
                    task_id=task_id,
                    session_id=session_id,
                    summary=f"Task started: {task.content[:300]}",
                    metadata={"tier": tier, "source": task.source},
                )
            except Exception:
                pass

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
                    metadata={"author": task.author, "session_id": session_id, "task_id": task_id},
                )
                if hasattr(self.memory, "append_session_turn"):
                    await self.memory.append_session_turn(
                        session_id=session_id,
                        role="user",
                        content=task.content,
                        task_id=task_id,
                        metadata={"author": task.author},
                    )
                await self._maybe_promote_memory_fact(task=task)
            except Exception:
                pass

        try:
            run_result = await self._run_executor.run(
                task=task,
                agent=agent,
                base_prompt=base_prompt,
                tier=tier,
            )
            result_output = run_result.output
            tool_calls = run_result.tool_calls
            attachments = run_result.attachments

            if run_result.waiting_for_user:
                if task.source == "a2a":
                    blocker = (
                        "I need clarification to continue, but this delegated task cannot pause for interactive input. "
                        f"Question: {run_result.question or 'Additional context required.'}"
                    )
                    await bridge.emit(TaskErrorEvent(error=blocker[:400]))
                    return TaskResult(
                        task=task,
                        output=blocker,
                        success=False,
                        status="failed",
                        elapsed_ms=(asyncio.get_running_loop().time() - start) * 1000,
                        tool_calls=tool_calls,
                        question=run_result.question,
                        timeout_s=run_result.timeout_s,
                        attachments=attachments,
                    )
                task_id = self.wait_registry.ensure_task_id(task.metadata)
                task.metadata["wait_state"] = {
                    "question": run_result.question,
                    "timeout_s": run_result.timeout_s,
                    "channel_id": task.channel_id,
                    "message_id": task.message_id,
                    "prompt_message_id": None,
                    "created_ts": time.time(),
                }
                if self.memory and task_id and hasattr(self.memory, "get_task_record") and hasattr(self.memory, "create_task_record"):
                    existing = await self.memory.get_task_record(task_id)
                    if existing is None:
                        await self.memory.create_task_record(
                            task_id=task_id,
                            source=task.source,
                            author=task.author,
                            content=task.content,
                            metadata=task.metadata,
                        )
                if self.memory and task_id and hasattr(self.memory, "mark_task_waiting"):
                    await self.memory.mark_task_waiting(
                        task_id,
                        metadata=task.metadata,
                        question=run_result.question or "",
                    )
                    if hasattr(self.memory, "set_session_status"):
                        await self.memory.set_session_status(
                            session_id,
                            status="waiting_for_user",
                            pending_task_id=task_id,
                        )
                    if hasattr(self.memory, "append_session_turn") and run_result.question:
                        await self.memory.append_session_turn(
                            session_id=session_id,
                            role="assistant",
                            content=run_result.question,
                            turn_kind="question",
                            task_id=task_id,
                        )
                    if hasattr(self.memory, "save_task_checkpoint"):
                        await self.memory.save_task_checkpoint(
                            task_id=task_id,
                            session_id=session_id,
                            summary=f"Waiting for user input: {run_result.question or ''}".strip(),
                            draft=result_output,
                            metadata=task.metadata,
                        )
                self.wait_registry.suspend(
                    task_id=task_id,
                    source=task.source,
                    author=task.author,
                    content=task.content,
                    channel_id=task.channel_id,
                    message_id=task.message_id,
                    metadata=task.metadata,
                    question=run_result.question or "",
                    timeout_s=run_result.timeout_s,
                    base_prompt=base_prompt,
                    tier=tier,
                )
                await bridge.emit(
                    TaskWaitingEvent(
                        question=run_result.question or "",
                        timeout_s=run_result.timeout_s,
                    )
                )
                return TaskResult(
                    task=task,
                    output="",
                    success=None,
                    status="waiting_for_user",
                    elapsed_ms=(asyncio.get_running_loop().time() - start) * 1000,
                    tool_calls=tool_calls,
                    waiting_for_user=True,
                    question=run_result.question,
                    timeout_s=run_result.timeout_s,
                    attachments=attachments,
                )

            answered_user = run_result.user_visible_reply_sent
            final_output = result_output
            if not answered_user:
                final_output, answered_user = await self._ensure_answer_required(
                    task=task,
                    output=result_output,
                    tool_calls=tool_calls,
                )

            elapsed_ms = (asyncio.get_running_loop().time() - start) * 1000
            elapsed_s = elapsed_ms / 1000

            log.info(
                "task_done",
                n=self._task_count,
                tier=tier,
                elapsed_ms=round(elapsed_ms),
                output_len=len(final_output),
                tool_calls=tool_calls,
            )

            status_value: TaskStatus = "succeeded" if answered_user else "failed"
            if answered_user:
                await bridge.emit(TaskDoneEvent(
                    output=final_output,
                    elapsed_s=elapsed_s,
                    tool_calls=tool_calls,
                ))
            else:
                await bridge.emit(TaskErrorEvent(error=final_output[:400]))

            # Broadcast task done to Postgres audit log
            if self._postgres is not None:
                try:
                    await self._postgres.log_task_done(task.content, answered_user, elapsed_ms, tool_calls)
                except Exception:
                    pass

            if answered_user:
                self._success_count += 1

            if self.memory and hasattr(self.memory, "set_session_status"):
                try:
                    await self.memory.set_session_status(
                        session_id,
                        status="completed" if answered_user else "failed",
                        pending_task_id="",
                    )
                except Exception:
                    pass
            if self.memory and hasattr(self.memory, "save_task_checkpoint"):
                try:
                    await self.memory.save_task_checkpoint(
                        task_id=task_id,
                        session_id=session_id,
                        summary=final_output[:1200],
                        draft=final_output[:2000],
                        metadata=task.metadata,
                    )
                except Exception:
                    pass
            if self.memory and hasattr(self.memory, "append_session_turn") and final_output:
                try:
                    await self.memory.append_session_turn(
                        session_id=session_id,
                        role="assistant",
                        content=final_output,
                        turn_kind="assistant",
                        task_id=task_id,
                    )
                except Exception:
                    pass

            # Save the assistant's reply to conversation history
            if self.memory and hasattr(self.memory, "save_message") and task.source == "discord" and final_output:
                try:
                    await self.memory.save_message(
                        role="assistant",
                        content=final_output,
                        channel_id=task.channel_id,
                        metadata={"session_id": session_id, "task_id": task_id},
                    )
                except Exception:
                    pass

            return TaskResult(
                task=task,
                output=final_output,
                status=status_value,
                answered_user=answered_user,
                user_visible_reply_sent=run_result.user_visible_reply_sent,
                attachments=attachments,
                success=answered_user,
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
                if self.memory and hasattr(self.memory, "save_task_checkpoint"):
                    try:
                        await self.memory.save_task_checkpoint(
                            task_id=task.metadata.get("task_id", ""),
                            session_id=str((task.metadata or {}).get("session_id", "")),
                            summary="Interrupted by rate limit.",
                            draft="",
                            metadata=task.metadata,
                        )
                    except Exception:
                        pass
            else:
                self._journal.clear()
                log.info("task_journal_cleared_on_crash", error=exc_str[:80])
            if self.memory and hasattr(self.memory, "set_session_status"):
                try:
                    await self.memory.set_session_status(
                        str((task.metadata or {}).get("session_id", "")),
                        status="failed",
                        pending_task_id="",
                    )
                except Exception:
                    pass
            if self.memory and hasattr(self.memory, "save_task_checkpoint"):
                try:
                    await self.memory.save_task_checkpoint(
                        task_id=str((task.metadata or {}).get("task_id", "")),
                        session_id=str((task.metadata or {}).get("session_id", "")),
                        summary=f"Task failed: {exc_str[:500]}",
                        metadata=task.metadata,
                    )
                except Exception:
                    pass

            return TaskResult(
                task=task,
                output=f"Error: {exc}",
                status="failed",
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
    ) -> RunResult:
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

    async def _ensure_answer_required(self, *, task: Task, output: str, tool_calls: int) -> tuple[str, bool]:
        if await self._is_answer_acceptable(task=task, output=output, tool_calls=tool_calls):
            return output, True

        repaired = await self._repair_user_answer(task=task, output=output)
        if await self._is_answer_acceptable(task=task, output=repaired, tool_calls=tool_calls):
            return repaired, True

        fallback = (
            repaired.strip()
            or output.strip()
            or (
                "I could not produce a reliable final answer from the work completed. "
                "Please retry or ask me to continue from a narrower checkpoint."
            )
        )
        return fallback, False

    async def _is_answer_acceptable(self, *, task: Task, output: str, tool_calls: int) -> bool:
        text = output.strip()
        if not text:
            return False
        if text.startswith(("[ERROR", "Error: [No reply", "⏳ Still working", "🔧", "🟢", "💭")):
            return False
        if tool_calls == 0 and len(text.split()) >= 2:
            return True
        validator_prompt = (
            "Decide whether the assistant's draft directly answers the user's request.\n"
            "Reply with exactly one line: ANSWERED or NOT_ANSWERED.\n\n"
            f"User request:\n{task.content[:2000]}\n\n"
            f"Assistant draft:\n{text[:4000]}"
        )
        try:
            validator = self.agents.get("fast", self.agent)
            result = await validator.run(validator_prompt, usage_limits=UsageLimits(request_limit=None))
            verdict = str(result.output).strip().splitlines()[0].strip().upper()
            return verdict == "ANSWERED"
        except Exception:
            return len(text.split()) >= 6

    async def _repair_user_answer(self, *, task: Task, output: str) -> str:
        repair_prompt = (
            "Write the final user-facing answer now.\n"
            "Use only the evidence already gathered.\n"
            "Answer the user's request directly in 3-8 sentences.\n"
            "If information is missing, say exactly what is missing.\n"
            "Do not call tools.\n\n"
            f"User request:\n{task.content[:2000]}\n\n"
            f"Existing draft or notes:\n{output[:4000]}"
        )
        try:
            repair_agent = self.agents.get("fast", self.agent)
            result = await repair_agent.run(repair_prompt, usage_limits=UsageLimits(request_limit=None))
            return str(result.output).strip()
        except Exception:
            return output.strip()

    @staticmethod
    def _coerce_int(value: object, *, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_int_or_none(value: object) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    async def _maybe_promote_memory_fact(self, *, task: Task) -> None:
        if self.memory is None or not hasattr(self.memory, "save_memory_fact"):
            return
        text = task.content.strip()
        lowered = text.lower()
        if not text or len(text) > 240:
            return
        fact_prefixes = (
            "remember ",
            "use ",
            "always ",
            "never ",
            "my ",
            "for this repo",
            "for this project",
            "prefer ",
        )
        if not any(lowered.startswith(prefix) for prefix in fact_prefixes):
            return
        await self.memory.save_memory_fact(
            text,
            metadata={
                "source": task.source,
                "author": task.author,
                "session_id": str((task.metadata or {}).get("session_id", "")),
            },
        )
