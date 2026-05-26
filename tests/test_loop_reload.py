from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.loop import AgentLoop


def test_reload_agents_rebuilds_tier_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, object]] = []

    class _Registry:
        pass

    def fake_create_agents(registry):
        created.append({"registry": registry, "count": len(created)})
        return {"fast": f"agent-{len(created)}", "smart": f"agent-{len(created)}", "best": f"agent-{len(created)}"}

    def fake_create_chat_agent():
        return "chat-agent"

    monkeypatch.setattr("agent.core.create_agents", fake_create_agents)
    monkeypatch.setattr("agent.core.create_chat_agent", fake_create_chat_agent)

    registry = _Registry()
    loop = AgentLoop({"fast": "old"}, tool_registry=registry)
    loop._reflection_service = SimpleNamespace(_agents=loop.agents)

    loop.reload_agents()

    assert loop.agents["fast"] == "agent-1"
    assert loop.chat_agent == "chat-agent"
    assert loop._reflection_service._agents == loop.agents
    assert created[0]["registry"] is registry


def test_reload_agents_without_registry_logs_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr("agent.loop.log.warning", lambda event, **kwargs: warnings.append(event))

    loop = AgentLoop({"fast": "old"}, tool_registry=None)
    loop.reload_agents()

    assert warnings == ["agent_reload_skipped"]
