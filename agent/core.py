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
    """Load IDENTITY.md, GOALS.md, MEMORY.md — with a size cap on MEMORY.md."""
    if not identity_path.exists():
        return ""

    # Character budget per file (rough: 1 token ≈ 4 chars)
    MEMORY_CHAR_CAP = 4_000   # ~1k tokens — most recent lessons only
    IDENTITY_CHAR_CAP = 2_000 # ~500 tokens each for identity/goals

    sections: list[str] = []
    for filename in ("IDENTITY.md", "GOALS.md", "MEMORY.md"):
        fp = identity_path / filename
        if not fp.exists():
            continue
        text = fp.read_text(encoding="utf-8").strip()
        cap = MEMORY_CHAR_CAP if filename == "MEMORY.md" else IDENTITY_CHAR_CAP
        if len(text) > cap:
            # Keep the tail (most recent content is at the bottom)
            text = "[...truncated...]\n\n" + text[-cap:]
            log.warning("identity_file_truncated", file=filename, cap=cap)
        sections.append(text)

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
10. For long-running commands (docker build, npm install, git clone large repos), pass timeout=3600 or higher to run_shell — there is no task-level timeout.

## Git / GitHub
- Clone repos to /workspace/<repo-name> (NOT /tmp).
- `gh` CLI is pre-authenticated via GH_TOKEN env var — NEVER run `gh auth login`.
- If `gh auth status` exits 0, you ARE authenticated — proceed directly to `gh pr create`.
- Create PRs with: `gh pr create --title "..." --body "..." --base main --repo owner/repo`
- SSH key is at /data/ssh/id_ed25519 (or id_rsa). GIT_SSH_COMMAND env var is pre-configured — plain `git clone git@github.com:...` just works.
- Always set git user inside cloned repos: `git config user.name "bob-agent" && git config user.email "bob@agent.local"`

{skills}
"""
    prompt = base.strip()

    # Hard cap: ~40k chars ≈ 10k tokens. Keep system prompt small so there's
    # plenty of room for tool call output within the 200k token window.
    MAX_CHARS = 40_000
    if len(prompt) > MAX_CHARS:
        prompt = prompt[:MAX_CHARS] + "\n[...system prompt truncated — MEMORY.md too large, use identity_edit to trim it...]"
        log.warning("system_prompt_truncated", chars=len(base), cap=MAX_CHARS)

    log.info("system_prompt_built", chars=len(prompt))
    return prompt


def create_agent(registry: "ToolRegistry", model_string: str) -> Agent:  # type: ignore[type-arg]
    """Create a single agent for the given model string."""
    mcp_servers = []
    if settings.browser_mcp_url:
        try:
            mcp_servers.append(MCPServerHTTP(url=settings.browser_mcp_url))
        except Exception as exc:
            log.warning("browser_mcp_unavailable", error=str(exc))

    # No static system_prompt here — we register a dynamic one below so it's
    # re-evaluated on every run, picking up MEMORY.md changes mid-session.
    agent: Agent = Agent(  # type: ignore[type-arg]
        model=model_string,
        mcp_servers=mcp_servers,
        retries=2,
    )

    # Dynamic system prompt: re-read identity files on every task so MEMORY.md
    # updates (and the 4k cap) are always reflected without a container restart.
    @agent.system_prompt(dynamic=True)
    def _system_prompt() -> str:
        return build_system_prompt()

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
