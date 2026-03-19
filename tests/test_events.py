from __future__ import annotations

import asyncio

import pytest

from agent.events import EventBridge, ProgressEvent


@pytest.mark.asyncio
async def test_event_bridge_delivers_to_registered_sinks() -> None:
    bridge = EventBridge()
    seen: list[str] = []

    async def sink(event) -> None:
        seen.append(event.kind)

    bridge.register("sink", sink)
    await bridge.emit(ProgressEvent(message="hello"))

    assert seen == ["progress"]


@pytest.mark.asyncio
async def test_event_bridge_isolates_sink_failures() -> None:
    bridge = EventBridge()
    seen: list[str] = []

    async def bad_sink(event) -> None:
        raise RuntimeError("boom")

    async def good_sink(event) -> None:
        seen.append(event.message)

    bridge.register("bad", bad_sink)
    bridge.register("good", good_sink)
    await bridge.emit(ProgressEvent(message="still works"))

    assert seen == ["still works"]


@pytest.mark.asyncio
async def test_event_bridge_unregister_removes_sink() -> None:
    bridge = EventBridge()
    seen: list[str] = []

    async def sink(event) -> None:
        seen.append(event.kind)

    bridge.register("sink", sink)
    bridge.unregister("sink")
    await bridge.emit(ProgressEvent(message="ignored"))

    assert seen == []


@pytest.mark.asyncio
async def test_event_bridge_applies_task_context_to_events() -> None:
    bridge = EventBridge()
    seen: list[str | None] = []

    async def sink(event) -> None:
        seen.append(event.task_id)

    bridge.register("sink", sink)
    with bridge.task_context("task-123"):
        await bridge.emit(ProgressEvent(message="working"))

    assert seen == ["task-123"]


@pytest.mark.asyncio
async def test_event_bridge_times_out_slow_sink_without_blocking_fast_sink() -> None:
    bridge = EventBridge(sink_timeout_s=0.1)
    seen: list[str] = []
    slow_started = asyncio.Event()

    async def slow_sink(event) -> None:
        slow_started.set()
        await asyncio.sleep(60)

    async def fast_sink(event) -> None:
        seen.append(event.message)

    bridge.register("slow", slow_sink)
    bridge.register("fast", fast_sink)

    await bridge.emit(ProgressEvent(message="still works"))

    assert slow_started.is_set()
    assert seen == ["still works"]
