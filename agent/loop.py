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
"""

from __future__ import annotations

import asyncio
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog
from pydantic_ai import Agent

from agent.config import settings


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


@dataclass
class TaskResult:
    task: Task
    output: str
    success: bool
    elapsed_ms: float
    tool_calls: int = 0


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
        # Per-channel message history for multi-turn conversations
        self._histories: dict[int, list[Any]] = {}

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

                result = await self._process(task)
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

    async def _process(self, task: Task) -> TaskResult:
        """Core observe→think→act cycle."""
        self._task_count += 1
        start = asyncio.get_event_loop().time()

        # Parse user model override (/fast, /smart, /best) and classify tier
        content, forced_tier = _parse_override(task.content)
        tier = forced_tier or _classify_tier(content)
        agent = self.agents.get(tier, self.agent)
        # Update task content if prefix was stripped
        task = Task(
            content=content,
            source=task.source,
            author=task.author,
            channel_id=task.channel_id,
            message_id=task.message_id,
            metadata=task.metadata,
            created_at=task.created_at,
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

        # Surface relevant past lessons before processing
        lessons_context = ""
        if self.memory and hasattr(self.memory, "search_lessons"):
            try:
                lessons_context = await self.memory.search_lessons(task.content[:200])
            except Exception:
                pass

        prompt = task.content
        if lessons_context:
            prompt = f"{lessons_context}\n\n---\n\n{task.content}"

        try:
            history = self._histories.get(task.channel_id, [])

            async with agent.run_mcp_servers():
                result = await agent.run(
                    prompt,
                    message_history=history if history else None,
                )

            output = str(result.output)
            tool_calls = len([m for m in result.new_messages() if hasattr(m, "parts")])

            # Update rolling history for this channel (keep last 10 messages)
            self._histories[task.channel_id] = list(result.all_messages())[-10:]

            elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
            log.info(
                "task_done",
                n=self._task_count,
                tier=tier,
                elapsed_ms=round(elapsed_ms),
                output_len=len(output),
            )

            self._success_count += 1
            return TaskResult(
                task=task,
                output=output,
                success=True,
                elapsed_ms=elapsed_ms,
                tool_calls=tool_calls,
            )

        except Exception as exc:
            elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
            log.error("task_failed", error=str(exc), exc=traceback.format_exc())
            return TaskResult(
                task=task,
                output=f"Error: {exc}",
                success=False,
                elapsed_ms=elapsed_ms,
            )

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

            # Save to lessons table
            await self.memory.save_lesson(
                summary=lesson,
                kind="mistake",
                context=task.content[:300],
            )

            # Append to MEMORY.md
            memory_path = settings.identity_path / "MEMORY.md"
            if memory_path.exists():
                existing = memory_path.read_text(encoding="utf-8")
                from datetime import date
                today = date.today().isoformat()
                updated = existing.rstrip() + f"\n\n### Mistake — {today}\n{lesson}\n"
                memory_path.write_text(updated, encoding="utf-8")

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
