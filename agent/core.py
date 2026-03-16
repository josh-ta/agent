"""
Agent core: builds the Pydantic AI agent, loads skills, wires tools + MCP servers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic_ai import Agent

from agent.config import settings
from agent.core_services import ModelFactory, PeerAgentProvider, PromptSources, SystemPromptBuilder

if TYPE_CHECKING:
    from agent.tools.registry import ToolRegistry

log = structlog.get_logger()

# Module-level Postgres reference — set by main.py after init so the dynamic
# system prompt can look up online peer agents on every task.
_postgres: Any | None = None
_prompt_sources = PromptSources()
_prompt_builder = SystemPromptBuilder(_prompt_sources)
_model_factory = ModelFactory()


def set_postgres(store: Any) -> None:
    global _postgres
    _postgres = store


def build_system_prompt(other_agents: list[str] | None = None) -> str:
    """Compose the system prompt from identity, peers, and the skill index."""
    return _prompt_builder.build(other_agents=other_agents)


def create_agent(registry: ToolRegistry, model_string: str) -> Agent:  # type: ignore[type-arg]
    """Create a single agent for the given model string."""
    mcp_servers = _model_factory.mcp_servers()
    model = _model_factory.build_model(model_string)
    model_settings = _model_factory.model_settings(model_string)

    agent: Agent = Agent(  # type: ignore[type-arg]
        model=model,
        mcp_servers=mcp_servers,
        model_settings=model_settings,
        retries=2,
    )

    # Dynamic system prompt: re-read identity files on every task so MEMORY.md
    # updates (and the 4k cap) are always reflected without a container restart.
    # Also fetches online peer agents from Postgres so Rule 13 (delegation) works.
    @agent.system_prompt(dynamic=True)
    async def _system_prompt() -> str:
        provider = PeerAgentProvider(_postgres)
        other_agents = await provider.list_other_agents()
        return build_system_prompt(other_agents=other_agents or None)

    registry.attach_to_agent(agent)
    return agent


def create_agents(registry: ToolRegistry) -> dict[str, Agent]:  # type: ignore[type-arg]
    """Create fast/smart/best agent tiers, reusing the same MCP + tool registry."""
    tiers = {
        "fast": settings.model_string_for("fast"),
        "smart": settings.model_string_for("smart"),
        "best": settings.model_string_for("best"),
    }

    # Deduplicate — if tiers point to the same model, share the Agent instance
    seen: dict[str, Agent] = {}  # type: ignore[type-arg]
    agents: dict[str, Agent] = {}  # type: ignore[type-arg]
    for tier, model_str in tiers.items():
        if model_str not in seen:
            seen[model_str] = create_agent(registry, model_str)
            log.info("agent_created", tier=tier, model=model_str)
        agents[tier] = seen[model_str]

    return agents
