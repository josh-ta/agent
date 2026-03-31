"""Subagent runner and restricted tool attachment."""

from __future__ import annotations

from typing import Any

import pytest

from agent.subagent_runner import SubagentRunner, attach_run_subagent_tool
from agent.tools.subagent_attach import attach_subagent_tools


class _FakeAgent:
    def __init__(self) -> None:
        self.tool_names: list[str] = []
        self._tools: dict[str, Any] = {}

    def tool_plain(self, fn: Any) -> Any:
        self.tool_names.append(fn.__name__)
        self._tools[fn.__name__] = fn
        return fn


def test_attach_subagent_tools_minimal_registers_reads_only() -> None:
    agent = _FakeAgent()
    attach_subagent_tools(agent, sqlite=None, postgres=None, profile="minimal")
    assert "read_file" in agent.tool_names
    assert "list_dir" in agent.tool_names
    assert "search_files" in agent.tool_names
    assert "run_shell_read_only" not in agent.tool_names
    assert "task_resume" not in agent.tool_names


def test_attach_subagent_tools_explore_adds_journal_read() -> None:
    agent = _FakeAgent()
    attach_subagent_tools(agent, sqlite=None, postgres=None, profile="explore")
    assert "task_resume" in agent.tool_names
    assert "run_shell_read_only" not in agent.tool_names


def test_attach_subagent_tools_verify_adds_shell_ro() -> None:
    agent = _FakeAgent()
    attach_subagent_tools(agent, sqlite=None, postgres=None, profile="verify")
    assert "run_shell_read_only" in agent.tool_names


def test_attach_subagent_tools_with_sqlite_adds_memory_tools() -> None:
    class _S:
        async def search_memory(self, q: str, limit: int = 5) -> str:
            return ""

        async def search_lessons(self, q: str, limit: int = 5) -> str:
            return ""

        async def search_procedures(self, q: str, limit: int = 5) -> list:
            return []

    agent = _FakeAgent()
    attach_subagent_tools(agent, sqlite=_S(), postgres=None, profile="minimal")
    assert "memory_search" in agent.tool_names


def test_attach_subagent_tools_with_postgres_adds_list_agents() -> None:
    class _P:
        async def list_agents(self) -> str:
            return ""

    agent = _FakeAgent()
    attach_subagent_tools(agent, sqlite=None, postgres=_P(), profile="minimal")
    assert "list_agents" in agent.tool_names


@pytest.mark.asyncio
async def test_subagent_runner_unknown_profile() -> None:
    runner = SubagentRunner(sqlite=None, postgres=None)
    out = await runner.run(instruction="x", profile="not-a-profile")
    assert "unknown profile" in out


@pytest.mark.asyncio
async def test_subagent_runner_success_via_build_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = SubagentRunner(sqlite=None, postgres=None)

    class _Sub:
        async def run(self, *a: Any, **kw: Any) -> Any:
            class _R:
                output = "  ok  "

            return _R()

    monkeypatch.setattr(runner, "_build_subagent", lambda _p: _Sub())
    out = await runner.run(instruction="task", profile="explore", max_tool_calls=5)
    assert out == "ok"


@pytest.mark.asyncio
async def test_subagent_runner_run_handles_subagent_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = SubagentRunner(sqlite=None, postgres=None)

    class _Boom:
        async def run(self, *a: Any, **kw: Any) -> Any:
            raise RuntimeError("model down")

    monkeypatch.setattr(runner, "_build_subagent", lambda _p: _Boom())
    out = await runner.run(instruction="x", profile="minimal")
    assert "subagent failed" in out
    assert "model down" in out


@pytest.mark.asyncio
async def test_attach_run_subagent_tool_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _FakeAgent()
    runner = SubagentRunner(sqlite=None, postgres=None)

    async def fake_run(
        self: Any,
        *,
        instruction: str,
        profile: str = "explore",
        max_tool_calls: int | None = None,
    ) -> str:
        return f"{instruction}|{profile}|{max_tool_calls}"

    monkeypatch.setattr(SubagentRunner, "run", fake_run)
    attach_run_subagent_tool(agent, runner)
    assert "run_agent_subtask" in agent.tool_names
    fn = agent._tools["run_agent_subtask"]
    out = await fn("hello", "verify", 12)
    assert out == "hello|verify|12"
