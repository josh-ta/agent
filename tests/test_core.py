from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.config import settings
from agent.core import build_system_prompt, create_agents
from agent.core_services import ModelFactory, PeerAgentProvider


def test_build_system_prompt_includes_identity_skills_and_peers(isolated_paths, monkeypatch: pytest.MonkeyPatch) -> None:
    (isolated_paths["identity"] / "GOALS.md").write_text("# Goals\nShip tests.\n", encoding="utf-8")
    monkeypatch.setattr(settings, "agent_name", "agent-1")

    prompt = build_system_prompt(other_agents=["agent-2"])

    assert "You are agent-1" in prompt
    assert "Peer Agents (online now)" in prompt
    assert "agent-2" in prompt
    assert "**example**" in prompt


@pytest.mark.asyncio
async def test_peer_agent_provider_parses_other_agents() -> None:
    async def _list_agents() -> str:
        return "Registered agents:\n\n  • agent-1 [online]\n  • helper [online]"

    provider = PeerAgentProvider(SimpleNamespace(list_agents=_list_agents))
    others = await provider.list_other_agents()

    assert others == ["helper"]


def test_model_factory_sets_reasoning_for_openai_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "thinking_enabled", True)
    monkeypatch.setattr(settings, "agent_name", "agent-1")

    model_settings = ModelFactory().model_settings("openai:o3-mini")

    assert model_settings is not None
    assert model_settings["openai_prompt_cache_key"] == "agent-1"
    assert model_settings["openai_reasoning_effort"] == "high"


def test_model_factory_sets_safe_max_tokens_for_claude_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "thinking_enabled", True)
    monkeypatch.setattr(settings, "thinking_budget_tokens", 5000)

    model_settings = ModelFactory().model_settings("anthropic:claude-sonnet-4-5")

    assert model_settings is not None
    assert model_settings["anthropic_thinking"] == {
        "type": "enabled",
        "budget_tokens": 5000,
    }
    assert model_settings["max_tokens"] == 6024


def test_model_factory_builds_openai_compatible_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_base_url", "http://localhost:1234/v1")
    monkeypatch.setattr(settings, "openai_api_key", "test-key")

    model = ModelFactory().build_model("openai:gpt-4o")

    assert getattr(model, "model_name", "") == "gpt-4o"


def test_create_agents_deduplicates_identical_models(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    monkeypatch.setattr(settings, "model_fast", "same-model")
    monkeypatch.setattr(settings, "model_smart", "same-model")
    monkeypatch.setattr(settings, "model_best", "other-model")
    monkeypatch.setattr(settings, "agent_model", "same-model")

    def fake_create_agent(registry, model_string: str):
        created.append(model_string)
        return object()

    monkeypatch.setattr("agent.core.create_agent", fake_create_agent)

    agents = create_agents(SimpleNamespace())

    assert set(agents.keys()) == {"fast", "smart", "best"}
    assert created.count("same-model") == 1
    assert created.count("other-model") == 1
