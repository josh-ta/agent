from __future__ import annotations

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
