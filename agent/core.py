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


def _load_skills_compact(skills_path: Path) -> str:
    """Return a compact skill index — just names and one-line descriptions."""
    if not skills_path.exists():
        return ""

    lines = ["## Skills (use read_skill <name> to load full content)"]
    for skill_file in sorted(skills_path.glob("*.md")):
        if skill_file.name.startswith("_"):
            continue
        # Grab first non-empty, non-heading line as description
        desc = ""
        for line in skill_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                desc = line[:80]
                break
        lines.append(f"- **{skill_file.stem}**: {desc}")

    return "\n".join(lines)


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
    """Compose a compact system prompt from identity + skill index."""
    identity = _load_identity(settings.identity_path)
    skills = _load_skills_compact(settings.skills_path)

    base = f"""You are {settings.agent_name}, an autonomous AI agent.

{identity}

## Tools
shell, read_file, write_file, list_dir, browser_navigate/screenshot/click/type, discord_send, read_discord, read_channel(name), edit_skill, edit_identity, self_restart, memory_save, lesson_search, skill_read

## Rules
1. Think before acting. Use shell for system tasks.
2. Read files before writing. Ask if unsure.
3. After failures: call lesson_save(kind="mistake"). Before complex tasks: call lesson_search.
4. Use skill_read <name> to load a skill's full procedure before following it.
5. Use read_channel('private') to catch up on recent conversation history when context is needed.
6. Each mistake happens only once — record it and move on.
7. NEVER call send_discord to reply to the user. Your text response IS the reply — it is sent automatically. Only call send_discord to proactively message a *different* channel than the one you were addressed in.
8. Give one clear response. Do not send multiple messages saying the same thing.
9. If the same approach fails twice, STOP and report what you tried and what's blocking you. Do not keep retrying variations of the same broken approach.

## Git / GitHub
- Clone repos to /workspace/<repo-name> (NOT /tmp).
- `gh` CLI is pre-authenticated via GH_TOKEN env var — NEVER run `gh auth login`.
- If `gh auth status` exits 0, you ARE authenticated — proceed directly to `gh pr create`.
- Create PRs with: `gh pr create --title "..." --body "..." --base main --repo owner/repo`
- SSH key is at /data/ssh/id_ed25519 (or id_rsa). GIT_SSH_COMMAND env var is pre-configured — plain `git clone git@github.com:...` just works.
- Always set git user inside cloned repos: `git config user.name "bob-agent" && git config user.email "bob@agent.local"`

{skills}
"""
    return base.strip()


def create_agent(registry: "ToolRegistry", model_string: str) -> Agent:  # type: ignore[type-arg]
    """Create a single agent for the given model string."""
    mcp_servers = []
    if settings.browser_mcp_url:
        try:
            mcp_servers.append(MCPServerHTTP(url=settings.browser_mcp_url))
        except Exception as exc:
            log.warning("browser_mcp_unavailable", error=str(exc))

    agent: Agent = Agent(  # type: ignore[type-arg]
        model=model_string,
        system_prompt=build_system_prompt(),
        mcp_servers=mcp_servers,
        retries=2,
    )
    registry.attach_to_agent(agent)
    return agent


def create_agents(registry: "ToolRegistry") -> dict[str, "Agent"]:  # type: ignore[type-arg]
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
