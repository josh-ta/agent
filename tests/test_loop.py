from __future__ import annotations

import asyncio

import pytest

import agent.loop as loop_module
from agent.loop import AgentLoop, Task


class _NullContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _RecordingAgent:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.histories: list[list | None] = []

    def run_mcp_servers(self) -> _NullContext:
        return _NullContext()

    async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
        self.prompts.append(prompt)
        self.histories.append(message_history)
        if False:
            yield None


class _BadArgsAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run_mcp_servers(self) -> _NullContext:
        return _NullContext()

    async def run_stream_events(self, prompt: str, message_history=None, usage_limits=None):
        self.calls += 1
        if False:
            yield None
        raise RuntimeError("EOF while parsing tool args")


@pytest.mark.asyncio
async def test_injected_messages_keep_original_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _emit(event) -> None:
        return None

    monkeypatch.setattr(loop_module.bridge, "emit", _emit)

    agent = _RecordingAgent()
    loop = AgentLoop({"smart": agent, "fast": agent, "best": agent})
    inject_queue: asyncio.Queue[str] = asyncio.Queue()
    await inject_queue.put("Please also update the docs")
    task = Task(content="Fix the parser", inject_queue=inject_queue)

    await loop._run_with_streaming(
        task=task,
        agent=agent,
        base_prompt="Original task context",
        tier="smart",
    )

    assert len(agent.prompts) == 2
    assert agent.prompts[0] == "Original task context"
    assert "Original task context" in agent.prompts[1]
    assert "Please also update the docs" in agent.prompts[1]


@pytest.mark.asyncio
async def test_bad_args_retry_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _emit(event) -> None:
        return None

    monkeypatch.setattr(loop_module.bridge, "emit", _emit)

    agent = _BadArgsAgent()
    loop = AgentLoop({"smart": agent, "fast": agent, "best": agent})
    task = Task(content="Run the task")

    with pytest.raises(RuntimeError, match="EOF while parsing tool args"):
        await loop._run_with_streaming(
            task=task,
            agent=agent,
            base_prompt="Run the task",
            tier="smart",
        )

    assert agent.calls == 3
