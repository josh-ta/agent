"""
Run lifecycle guard (CC QueryGuard-style).

Pairs with an explicit monotonic run_generation on each task so event sinks
can ignore stale deliveries and end_run() can detect superseded completions.
"""

from __future__ import annotations


class RunGuard:
    """At most one run with a known generation may be active at a time."""

    __slots__ = ("_current_gen", "_status")

    def __init__(self) -> None:
        self._status: str = "idle"
        self._current_gen: int | None = None

    def begin_run(self, generation: int) -> bool:
        if self._status == "running":
            return False
        self._status = "running"
        self._current_gen = generation
        return True

    def end_run(self, generation: int) -> bool:
        if self._status != "running" or self._current_gen != generation:
            return False
        self._status = "idle"
        self._current_gen = None
        return True

    def force_idle(self) -> None:
        self._status = "idle"
        self._current_gen = None
