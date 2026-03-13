"""
Agent core: builds the Pydantic AI agent, loads skills, wires tools + MCP servers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP

from agent.config import settings

if TYPE_CHECKING:
    from agent.tools.registry import ToolRegistry

log = structlog.get_logger()


def _load_skills(skills_path: Path) -> str:
    """Load all .md skill files and return them as a combined system prompt section."""
    if not skills_path.exists():
        return ""

    parts: list[str] = ["## Available Skills\n"]
    for skill_file in sorted(skills_path.glob("*.md")):
        if skill_file.name.startswith("_"):
            continue
        content = skill_file.read_text(encoding="utf-8").strip()
        parts.append(f"### Skill: {skill_file.stem}\n{content}\n")

    return "\n".join(parts)


def _load_identity(identity_path: Path) -> str:
    """Load IDENTITY.md, GOALS.md, MEMORY.md into the system prompt."""
    if not identity_path.exists():
        return ""

    sections: list[str] = []
    for filename in ("IDENTITY.md", "GOALS.md", "MEMORY.md"):
        fp = identity_path / filename
        if fp.exists():
            sections.append(fp.read_text(encoding="utf-8").strip())

    return "\n\n".join(sections)


def build_system_prompt() -> str:
    """Compose the full system prompt from identity + skills."""
    identity = _load_identity(settings.identity_path)
    skills = _load_skills(settings.skills_path)

    base = f"""You are {settings.agent_name}, an autonomous AI agent running in a Docker container.

{identity}

## Capabilities
You have access to the following tools:
- **shell**: Execute any shell command on the host system
- **read_file** / **write_file** / **list_dir**: Read and write files in your workspace
- **browser**: Control a real browser (navigate, click, fill forms, screenshot)
- **discord_send**: Send messages to Discord channels
- **discord_read**: Read recent messages from a Discord channel
- **edit_skill**: Create or update a skill file (self-improvement)
- **edit_identity**: Update your IDENTITY.md, GOALS.md, or MEMORY.md
- **self_restart**: Restart your own container after code-level changes

## Behaviour Guidelines
1. Think step by step before acting.
2. Use the shell for any system tasks; prefer small, safe commands.
3. Always read a file before writing it.
4. Record significant learnings by updating MEMORY.md or creating a new skill.
5. Communicate clearly in Discord: mention @agent-name when addressing a specific agent.
6. When given a task, break it into steps and execute them sequentially.
7. If unsure, ask the user via Discord rather than guessing.

{skills}
"""
    return base.strip()


def create_agent(registry: "ToolRegistry") -> Agent:  # type: ignore[type-arg]
    """Create and return the configured Pydantic AI agent."""
    mcp_servers = []

    # Browser MCP sidecar
    if settings.browser_mcp_url:
        try:
            mcp_servers.append(MCPServerHTTP(url=settings.browser_mcp_url))
            log.info("browser_mcp_registered", url=settings.browser_mcp_url)
        except Exception as exc:
            log.warning("browser_mcp_unavailable", error=str(exc))

    agent: Agent = Agent(  # type: ignore[type-arg]
        model=settings.model_string,
        system_prompt=build_system_prompt(),
        mcp_servers=mcp_servers,
        retries=3,
    )

    # Register all tools from the registry
    registry.attach_to_agent(agent)

    log.info(
        "agent_created",
        name=settings.agent_name,
        model=settings.model_string,
        mcp_count=len(mcp_servers),
    )
    return agent
