"""
Agent core: builds the Pydantic AI agent, loads skills, wires tools + MCP servers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from agent.config import settings

if TYPE_CHECKING:
    from agent.tools.registry import ToolRegistry

log = structlog.get_logger()

# Module-level Postgres reference — set by main.py after init so the dynamic
# system prompt can look up online peer agents on every task.
_postgres: Any | None = None


def set_postgres(store: Any) -> None:
    global _postgres
    _postgres = store


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


def build_system_prompt(other_agents: list[str] | None = None) -> str:
    """Compose the system prompt from identity, peers, and the skill index."""
    identity = _load_identity(settings.identity_path)
    skills = _load_skills_compact(settings.skills_path)

    # Channel IDs injected as concrete constants so the prompt can reference them directly.
    comms_id = settings.discord_comms_channel_id
    bus_id = settings.discord_bus_channel_id
    private_id = settings.discord_agent_channel_id

    # Only show the peer section when other agents are online.
    if other_agents:
        peers_block = (
            "\n## Peer Agents (online now)\n"
            + "\n".join(f"- {name}" for name in other_agents)
            + "\n"
        )
    else:
        peers_block = ""

    base = f"""You are {settings.agent_name}, an autonomous AI agent.

{identity}
{peers_block}
## Discord Channels
- Your private channel (streaming + direct chat): {private_id}
- Agent comms — structured JSON A2A only: {comms_id}
- Agent bus — brief broadcast announcements: {bus_id}

## Rules
1. Think before acting. Use shell for system tasks.
2. Read files before writing. Ask if unsure.
3. Prefer surgical edits:
   - Use `str_replace` for targeted edits.
   - Use `write_file` for new files or true rewrites.
   - Use `search_files` to locate code before editing.
   - Use `tail_lines=100` with `run_shell` for verbose test output.
4. After failures: call lesson_save(kind="mistake"). Before complex tasks: call lesson_search.
5. Use skill_read <name> to load a skill's full procedure before following it.
6. Use read_channel('private') to catch up on recent conversation history when context is needed.
7. Each mistake happens only once — record it and move on.
8. For multi-step tasks (>2 tool calls), call task_note() after each major step. Do not use send_discord to restate your final answer.
9. Give one clear response. Do not send multiple messages saying the same thing.
10. If the same approach fails twice, STOP and report what you tried and what is blocking you.
11. For long-running commands, pass timeout=3600 or higher to run_shell.
12. When you receive a task prefixed [A2A from X], complete it and send the result back to agent-comms:
    send_discord({comms_id}, '{{"from": "{settings.agent_name}", "to": "X", "task": "result", "payload": "your answer"}}')
    When delegating, poll read_discord({comms_id}) every few tool calls for replies.
13. If uncertainty would change the approach, call ask_user_question() and ask one clear question.
14. Delegate only when the work splits cleanly and other agents are online:
    a. Identify parallelizable sub-tasks.
    b. Delegate one sub-task via agent-comms JSON: send_discord({comms_id}, '{{"from": "{settings.agent_name}", "to": "PEER_NAME", "task": "DESCRIPTION", "payload": ""}}')
    c. Work your own sub-task simultaneously.
    d. Poll read_discord({comms_id}) periodically until you see their result ("task": "result").
    e. Combine both results in your final reply.
    Do not delegate trivial work, solo work, or work you have already started.

## Long tasks — checkpointing (IMPORTANT)
For tasks with more than ~5 tool calls, use the task journal:
- Call task_resume() FIRST — check if this task was previously attempted.
- Call task_note(note) after EVERY significant step: what you did, what you found, what comes next.
- Write notes in plain English, not JSON.
- Call task_journal_clear() only after success.
- After a rate limit or interruption, resume with task_resume().

## Git / GitHub
- Clone repos to /workspace/<repo-name> (NOT /tmp).
- `gh` CLI is pre-authenticated via GH_TOKEN env var — NEVER run `gh auth login`.
- If `gh auth status` exits 0, you ARE authenticated — proceed directly to `gh pr create`.
- Create PRs with: `gh pr create --title "..." --body "..." --base main --repo owner/repo`
- SSH key is at /data/ssh/id_ed25519 (or id_rsa). Plain `git clone git@github.com:...` should work.
- Always set git user inside cloned repos: `git config user.name "bob-agent" && git config user.email "bob@agent.local"`

{skills}
"""
    prompt = base.strip()

    # Keep the prompt capped so tool output still has room in the model context.
    MAX_CHARS = 40_000
    if len(prompt) > MAX_CHARS:
        prompt = (
            prompt[:MAX_CHARS]
            + "\n[...system prompt truncated — MEMORY.md too large, use identity_edit to trim it...]"
        )
        log.warning("system_prompt_truncated", chars=len(base), cap=MAX_CHARS)

    log.info("system_prompt_built", chars=len(prompt))
    return prompt


def _build_openai_compatible_model(model_name: str, *, base_url: str, api_key: str) -> OpenAIModel:
    provider = OpenAIProvider(
        base_url=base_url or None,
        api_key=api_key or None,
    )
    return OpenAIModel(model_name, provider=provider)


def create_agent(registry: ToolRegistry, model_string: str) -> Agent:  # type: ignore[type-arg]
    """Create a single agent for the given model string."""
    mcp_servers = []
    if settings.browser_mcp_url:
        try:
            mcp_servers.append(MCPServerHTTP(url=settings.browser_mcp_url))
        except Exception as exc:
            log.warning("browser_mcp_unavailable", error=str(exc))

    model: str | OpenAIModel = model_string
    if model_string.startswith("xai:"):
        model = _build_openai_compatible_model(
            model_string.split(":", 1)[1],
            base_url=settings.xai_base_url,
            api_key=settings.xai_api_key,
        )
    elif model_string.startswith("mistral:"):
        model = _build_openai_compatible_model(
            model_string.split(":", 1)[1],
            base_url=settings.mistral_base_url,
            api_key=settings.mistral_api_key,
        )
    elif model_string.startswith("openai:") and settings.openai_base_url:
        model = _build_openai_compatible_model(
            model_string.split(":", 1)[1],
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )

    # Build model settings with prompt caching and optional extended thinking.
    # Cache TTL is 1h (vs default 5m) — sporadic Discord usage means 5m caches
    # frequently expire; 1h gives far more cache hits at only a slightly higher
    # write cost (2x vs 1.25x surcharge, same 90% discount on reads).
    is_claude = model_string.startswith("anthropic:") or "claude" in model_string
    is_haiku = "haiku" in model_string
    is_openai = model_string.startswith(("openai:", "xai:", "mistral:"))
    is_reasoning_model = any(x in model_string for x in ("o1", "o3", "o4"))
    is_gemini = "google" in model_string or "gemini" in model_string
    is_groq = "groq" in model_string

    model_settings: dict = {}

    # Anthropic — explicit cache markers required; Haiku does not support TTL control.
    if is_claude and not is_haiku:
        model_settings["anthropic_cache_instructions"] = "1h"
        model_settings["anthropic_cache_tool_definitions"] = "1h"
        # Cache the last user message so accumulated tool-call turns stay cached.
        model_settings["anthropic_cache_messages"] = "1h"

    if settings.thinking_enabled and is_claude and not is_haiku:
        model_settings["anthropic_thinking"] = {
            "type": "enabled",
            "budget_tokens": settings.thinking_budget_tokens,
        }
        log.info(
            "thinking_enabled",
            model=model_string,
            budget_tokens=settings.thinking_budget_tokens,
        )

    # OpenAI — caching is automatic; these settings improve routing hit rates
    # and extend retention from the default 5–60 min to 24h.
    if is_openai:
        model_settings["openai_prompt_cache_key"] = settings.agent_name
        model_settings["openai_prompt_cache_retention"] = "24h"
        if is_reasoning_model and settings.thinking_enabled:
            model_settings["openai_reasoning_effort"] = "high"
            log.info(
                "thinking_enabled",
                model=model_string,
                budget_tokens=settings.thinking_budget_tokens,
            )

    # Gemini — caching is fully automatic (implicit); only wire thinking here.
    if is_gemini and settings.thinking_enabled:
        model_settings["google_thinking_config"] = {
            "thinking_budget": settings.thinking_budget_tokens,
        }
        log.info(
            "thinking_enabled",
            model=model_string,
            budget_tokens=settings.thinking_budget_tokens,
        )

    # Groq — no caching available; wire reasoning format for capable models.
    if is_groq and settings.thinking_enabled:
        model_settings["groq_reasoning_format"] = "parsed"
        log.info(
            "thinking_enabled",
            model=model_string,
            budget_tokens=settings.thinking_budget_tokens,
        )

    model_settings = model_settings or None

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
        other_agents: list[str] = []
        if _postgres is not None:
            try:
                import asyncio as _asyncio
                raw = await _asyncio.wait_for(_postgres.list_agents(), timeout=2.0)
                # list_agents() returns a formatted string; parse out agent names
                # excluding ourselves
                for line in raw.splitlines():
                    line = line.strip("- •\t ")
                    if line and settings.agent_name.lower() not in line.lower():
                        # Extract just the name before any status/model info
                        name = line.split()[0].rstrip(":")
                        if name:
                            other_agents.append(name)
            except Exception:
                pass
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
