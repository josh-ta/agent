"""
Event bridge: typed events + fan-out to registered sinks.

All agent activity — model streaming, tool calls, shell execution, task lifecycle —
is emitted as a structured AgentEvent through the module-level `bridge` singleton.
Consumers (Discord, logs, future WebSocket) register async sink functions once and
receive every event without the loop needing to know about any specific output channel.

Event taxonomy
--------------
Streaming (per-character / per-chunk):
  TextDeltaEvent      — one text token from the model (every chunk)
  ThinkingDeltaEvent  — one thinking token (only when provider exposes reasoning)

Turn completion (buffered):
  ThinkingEndEvent    — full thinking block assembled, ready to display
  TextTurnEndEvent    — full text turn; is_final=True means it's the reply

Tool lifecycle:
  ToolCallStartEvent  — model invoked a tool (name, args, call_id)
  ToolResultEvent     — tool returned a result

Shell lifecycle:
  ShellStartEvent     — command started (command string, cwd)
  ShellOutputEvent    — one stdout/stderr chunk as it arrives
  ShellDoneEvent      — process exited (exit_code, elapsed seconds)

Task lifecycle:
  TaskStartEvent      — new task begun (content, tier)
  TaskDoneEvent       — task completed successfully
  TaskErrorEvent      — task failed with an error

Housekeeping:
  ProgressEvent       — ticker pings, rate-limit notices, context-compression alerts
"""

from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Union

import structlog

log = structlog.get_logger()

SinkFn = Callable[["AgentEvent"], Awaitable[None]]


# ── Streaming deltas ───────────────────────────────────────────────────────────

@dataclass
class TextDeltaEvent:
    """One text token from the model. Emitted for every chunk."""
    delta: str
    kind: str = field(default="text_delta", init=False)


@dataclass
class ThinkingDeltaEvent:
    """One thinking/reasoning token. Only emitted when the provider exposes reasoning."""
    delta: str
    kind: str = field(default="thinking_delta", init=False)


# ── Turn completion ────────────────────────────────────────────────────────────

@dataclass
class ThinkingEndEvent:
    """Full thinking block assembled from deltas, ready to display."""
    text: str
    kind: str = field(default="thinking_end", init=False)


@dataclass
class TextTurnEndEvent:
    """Full text turn assembled from deltas.

    is_final=True  → this is the agent's final answer (sent as Discord reply)
    is_final=False → this is an intermediate reasoning step (sent as 💭)
    """
    text: str
    is_final: bool
    kind: str = field(default="text_turn_end", init=False)


# ── Tool lifecycle ─────────────────────────────────────────────────────────────

@dataclass
class ToolCallStartEvent:
    """Model invoked a tool. Emitted before the tool runs."""
    tool_name: str
    call_id: str
    args: Any
    kind: str = field(default="tool_call_start", init=False)


@dataclass
class ToolResultEvent:
    """Tool execution completed and returned a result to the model."""
    tool_name: str
    call_id: str
    result: str
    kind: str = field(default="tool_result", init=False)


# ── Shell lifecycle ────────────────────────────────────────────────────────────

@dataclass
class ShellStartEvent:
    """A shell command started executing."""
    command: str
    cwd: str
    kind: str = field(default="shell_start", init=False)


@dataclass
class ShellOutputEvent:
    """One stdout/stderr chunk from a running command."""
    chunk: str
    kind: str = field(default="shell_output", init=False)


@dataclass
class ShellDoneEvent:
    """Shell command finished."""
    exit_code: int
    elapsed_s: float
    kind: str = field(default="shell_done", init=False)


# ── Task lifecycle ─────────────────────────────────────────────────────────────

@dataclass
class TaskStartEvent:
    """A new task has begun processing."""
    content: str
    tier: str
    kind: str = field(default="task_start", init=False)


@dataclass
class TaskDoneEvent:
    """Task completed successfully."""
    output: str
    elapsed_s: float
    tool_calls: int
    kind: str = field(default="task_done", init=False)


@dataclass
class TaskErrorEvent:
    """Task failed with an error."""
    error: str
    kind: str = field(default="task_error", init=False)


# ── Housekeeping ───────────────────────────────────────────────────────────────

@dataclass
class ProgressEvent:
    """Ticker pings, rate-limit notices, context-compression alerts, injection acks."""
    message: str
    kind: str = field(default="progress", init=False)


# ── Union type ─────────────────────────────────────────────────────────────────

AgentEvent = Union[
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    TextTurnEndEvent,
    ToolCallStartEvent,
    ToolResultEvent,
    ShellStartEvent,
    ShellOutputEvent,
    ShellDoneEvent,
    TaskStartEvent,
    TaskDoneEvent,
    TaskErrorEvent,
    ProgressEvent,
]


# ── EventBridge ────────────────────────────────────────────────────────────────

class EventBridge:
    """
    Fan-out bus: emit typed events, deliver them to all registered sinks.

    Sinks are registered with an optional tag so they can be removed later
    (e.g. when a task finishes and the Discord channel-specific sink should stop).

    Thread/task safety: sinks are called concurrently via asyncio.gather so a
    slow sink cannot block others. Exceptions in any individual sink are caught
    and logged — they never propagate back to the emitter.
    """

    def __init__(self) -> None:
        self._sinks: dict[str, SinkFn] = {}

    def register(self, tag: str, sink: SinkFn) -> None:
        """Register a sink under a unique tag. Replaces any existing sink with that tag."""
        self._sinks[tag] = sink

    def unregister(self, tag: str) -> None:
        """Remove a sink by tag. Silent no-op if tag not found."""
        self._sinks.pop(tag, None)

    async def emit(self, event: AgentEvent) -> None:
        """Deliver an event to all currently registered sinks concurrently."""
        if not self._sinks:
            return
        sinks = list(self._sinks.values())
        results = await asyncio.gather(
            *(s(event) for s in sinks),
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.warning(
                    "event_sink_error",
                    event_kind=event.kind,
                    sink_index=i,
                    error=str(result),
                )


# Module-level singleton — import and use directly:
#   from agent.events import bridge, TextDeltaEvent
#   await bridge.emit(TextDeltaEvent(delta="hello"))
bridge = EventBridge()
