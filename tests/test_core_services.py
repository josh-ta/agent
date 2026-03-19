from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.core_services as core_services
from agent.config import settings
from agent.core_services import ModelFactory, PeerAgentProvider, PromptSources, SystemPromptBuilder


def test_load_skills_compact_uses_first_non_heading_line(isolated_paths) -> None:
    skills_path = isolated_paths["skills"]
    (skills_path / "_private.md").write_text("# Ignore\nHidden\n", encoding="utf-8")
    (skills_path / "deploy.md").write_text("# Deploy\n\nShip it safely.\nMore text.\n", encoding="utf-8")

    output = PromptSources().load_skills_compact(skills_path)

    assert "**deploy**: Ship it safely." in output
    assert "_private" not in output


def test_load_skills_compact_allows_empty_description(isolated_paths) -> None:
    skills_path = isolated_paths["skills"]
    (skills_path / "blank.md").write_text("# Blank\n## Still blank\n", encoding="utf-8")

    output = PromptSources().load_skills_compact(skills_path)

    assert "**blank**: " in output


def test_load_identity_truncates_large_files(isolated_paths) -> None:
    identity_path = isolated_paths["identity"]
    long_identity = "A" * 2105
    long_memory = "B" * 4010
    (identity_path / "IDENTITY.md").write_text(long_identity, encoding="utf-8")
    (identity_path / "MEMORY.md").write_text(long_memory, encoding="utf-8")

    output = PromptSources().load_identity(identity_path)

    assert "[...truncated...]" in output
    assert ("A" * 2000) in output
    assert ("B" * 4000) in output


def test_prompt_sources_return_empty_for_missing_paths(tmp_path) -> None:
    sources = PromptSources()

    assert sources.load_skills_compact(tmp_path / "missing-skills") == ""
    assert sources.load_identity(tmp_path / "missing-identity") == ""


def test_system_prompt_builder_includes_peers_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = SimpleNamespace(
        load_identity=lambda _path: "Identity block",
        load_skills_compact=lambda _path: "Skills block",
    )
    builder = SystemPromptBuilder(prompt_sources=sources)
    monkeypatch.setattr(settings, "agent_name", "agent-1")
    monkeypatch.setattr(settings, "discord_comms_channel_id", 11)
    monkeypatch.setattr(settings, "discord_bus_channel_id", 22)
    monkeypatch.setattr(settings, "discord_agent_channel_id", 33)

    prompt = builder.build(other_agents=["helper"])

    assert "You are agent-1" in prompt
    assert "Peer Agents (online now)" in prompt
    assert "- helper" in prompt


def test_system_prompt_builder_truncates_oversized_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = SimpleNamespace(
        load_identity=lambda _path: "A" * 45000,
        load_skills_compact=lambda _path: "",
    )
    builder = SystemPromptBuilder(prompt_sources=sources)
    monkeypatch.setattr(settings, "agent_name", "agent-1")

    prompt = builder.build()

    assert "system prompt truncated" in prompt
    assert len(prompt) > 40000


@pytest.mark.asyncio
async def test_peer_agent_provider_returns_empty_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _list_agents() -> str:
        return "Registered agents:\n- helper [online]"

    async def _raise_timeout(awaitable, timeout):
        awaitable.close()
        raise TimeoutError

    provider = PeerAgentProvider(SimpleNamespace(list_agents=_list_agents))
    monkeypatch.setattr(asyncio, "wait_for", _raise_timeout)

    assert await provider.list_other_agents() == []


@pytest.mark.asyncio
async def test_peer_agent_provider_returns_empty_when_store_missing() -> None:
    assert await PeerAgentProvider(None).list_other_agents() == []


@pytest.mark.asyncio
async def test_peer_agent_provider_filters_self_and_malformed_lines() -> None:
    async def _list_agents() -> str:
        return "\n".join(
            [
                "Registered agents:",
                "- agent-1 [online]",
                "- helper [online]",
                "- note without bracket",
                "- : [online]",
                "status:",
            ]
        )

    provider = PeerAgentProvider(SimpleNamespace(list_agents=_list_agents))

    assert await provider.list_other_agents() == ["helper"]


def test_model_factory_returns_mcp_server_when_url_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "browser_mcp_url", "http://localhost:3080/sse")
    monkeypatch.setattr(core_services, "MCPServerHTTP", lambda url: {"url": url})

    servers = ModelFactory().mcp_servers()

    assert servers == [{"url": "http://localhost:3080/sse"}]


def test_model_factory_returns_no_mcp_servers_without_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "browser_mcp_url", "")
    assert ModelFactory().mcp_servers() == []


def test_model_factory_swallow_mcp_server_constructor_error_and_plain_openai_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "browser_mcp_url", "http://localhost:3080/sse")

    def _boom(url: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(core_services, "MCPServerHTTP", _boom)
    monkeypatch.setattr(settings, "openai_base_url", "")
    factory = ModelFactory()

    assert factory.mcp_servers() == []
    assert factory.build_model("openai:gpt-4o") == "openai:gpt-4o"
    assert factory.build_model("plain-model") == "plain-model"


def test_model_factory_build_model_uses_provider_specific_base_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, str, str]] = []

    def fake_build(model_name: str, *, base_url: str, api_key: str) -> str:
        captured.append((model_name, base_url, api_key))
        return f"built:{model_name}"

    monkeypatch.setattr(ModelFactory, "_build_openai_compatible_model", staticmethod(fake_build))
    monkeypatch.setattr(settings, "xai_base_url", "https://xai.example")
    monkeypatch.setattr(settings, "xai_api_key", "xai-key")
    monkeypatch.setattr(settings, "mistral_base_url", "https://mistral.example")
    monkeypatch.setattr(settings, "mistral_api_key", "mistral-key")
    monkeypatch.setattr(settings, "openai_base_url", "https://openai.example")
    monkeypatch.setattr(settings, "openai_api_key", "openai-key")

    factory = ModelFactory()

    assert factory.build_model("xai:grok-4") == "built:grok-4"
    assert factory.build_model("mistral:large") == "built:large"
    assert factory.build_model("openai:gpt-4o") == "built:gpt-4o"
    assert captured == [
        ("grok-4", "https://xai.example", "xai-key"),
        ("large", "https://mistral.example", "mistral-key"),
        ("gpt-4o", "https://openai.example", "openai-key"),
    ]


def test_model_factory_model_settings_cover_haiku_gemini_and_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "thinking_enabled", True)
    monkeypatch.setattr(settings, "thinking_budget_tokens", 321)
    monkeypatch.setattr(settings, "agent_name", "agent-1")
    factory = ModelFactory()

    haiku = factory.model_settings("anthropic:claude-3-5-haiku")
    gemini = factory.model_settings("google:gemini-2.5-pro")
    groq = factory.model_settings("groq:deepseek-r1")

    assert haiku is None
    assert gemini == {"google_thinking_config": {"thinking_budget": 321}}
    assert groq == {"groq_reasoning_format": "parsed"}


def test_model_factory_model_settings_cover_non_reasoning_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "thinking_enabled", False)
    monkeypatch.setattr(settings, "agent_name", "agent-1")

    settings_map = ModelFactory().model_settings("openai:gpt-4o")

    assert settings_map == {
        "openai_prompt_cache_key": "agent-1",
        "openai_prompt_cache_retention": "24h",
    }
