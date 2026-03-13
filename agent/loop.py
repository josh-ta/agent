"""
Main agent reasoning loop: Observe → Think → Act → Remember.

The loop processes tasks from an asyncio.Queue populated by:
  - Discord messages
  - Scheduled heartbeats
  - Direct API calls (when used as a library)
"""

from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog
from pydantic_ai import Agent

from agent.config import settings

log = structlog.get_logger()


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
    """

    def __init__(self, agent: Agent, memory_store: Any = None) -> None:  # type: ignore[type-arg]
        self.agent = agent
        self.memory = memory_store
        self.queue: asyncio.Queue[Task] = asyncio.Queue()
        self._running = False
        self._task_count = 0
        # Per-channel message history for multi-turn conversations
        self._histories: dict[int, list[Any]] = {}

    async def enqueue(self, task: Task) -> None:
        """Add a task to the processing queue."""
        await self.queue.put(task)

    async def run_forever(self) -> None:
        """Process tasks from the queue indefinitely."""
        self._running = True
        log.info("loop_started", agent=settings.agent_name)

        while self._running:
            try:
                # Wait for next task (with timeout to allow heartbeat tasks)
                try:
                    task = await asyncio.wait_for(self.queue.get(), timeout=settings.heartbeat_seconds)
                except asyncio.TimeoutError:
                    # Heartbeat
                    await self._heartbeat()
                    continue

                result = await self._process(task)
                self.queue.task_done()

                # Store result in memory
                if self.memory:
                    await self.memory.record_task(task, result)

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

        log.info(
            "task_start",
            n=self._task_count,
            source=task.source,
            author=task.author,
            content=task.content[:120],
        )

        try:
            # Retrieve channel history for multi-turn context
            history = self._histories.get(task.channel_id, [])

            # Run the Pydantic AI agent
            async with self.agent.run_mcp_servers():
                result = await self.agent.run(
                    task.content,
                    message_history=history if history else None,
                )

            output = str(result.output)
            tool_calls = len([m for m in result.new_messages() if hasattr(m, "parts")])

            # Update rolling history for this channel (keep last 40 messages)
            self._histories[task.channel_id] = list(result.all_messages())[-40:]

            elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
            log.info(
                "task_done",
                n=self._task_count,
                elapsed_ms=round(elapsed_ms),
                output_len=len(output),
            )

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

    async def _heartbeat(self) -> None:
        """Periodic background work: update Postgres presence, check for updates."""
        log.debug("heartbeat", agent=settings.agent_name, queue_size=self.queue.qsize())
        # Memory store handles its own heartbeat registration
        if self.memory and hasattr(self.memory, "heartbeat"):
            await self.memory.heartbeat()
