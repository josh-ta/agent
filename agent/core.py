"""
Agent core: builds the Pydantic AI agent, loads skills, wires tools + MCP servers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP

from agent.config import settings

if TYPE_CHECKING:
    from agent.tools.registry import ToolRegistry

log = structlog.get_logger()

# Module-level Postgres reference — set by main.py after init so the dynamic
# system prompt can look up online peer agents on every task.
_postgres: "Any | None" = None


def set_postgres(store: "Any") -> None:
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
    """Compose a compact system prompt from identity + skill index.

    other_agents: names of other online agents (from Postgres registry).
                  Injected so the agent knows who it can delegate to.
    """
    identity = _load_identity(settings.identity_path)
    skills = _load_skills_compact(settings.skills_path)

    # Channel IDs as usable constants
    comms_id = settings.discord_comms_channel_id
    bus_id = settings.discord_bus_channel_id
    private_id = settings.discord_agent_channel_id

    # Peer agent section — only shown when there are peers
    if other_agents:
        peers_block = (
            f"\n## Peer Agents (online now)\n"
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

## Tools
shell, read_file, write_file, list_dir, browser_navigate/screenshot/click/type, discord_send, read_discord, read_channel(name), ask_user_question, edit_skill, edit_identity, self_restart, memory_save, lesson_search, skill_read, task_note, task_resume, task_journal_clear, gh_pr_view, gh_pr_diff, gh_pr_comment, gh_pr_review, gh_pr_review_inline, gh_pr_checks, gh_pr_merge, gh_issue_view, gh_issue_comment, gh_issue_create, gh_ci_list, gh_ci_logs_failed, gh_ci_rerun

## Rules
1. Think before acting. Use shell for system tasks.
2. Read files before writing. Ask if unsure.
3. After failures: call lesson_save(kind="mistake"). Before complex tasks: call lesson_search.
4. Use skill_read <name> to load a skill's full procedure before following it.
5. Use read_channel('private') to catch up on recent conversation history when context is needed.
6. Each mistake happens only once — record it and move on.
7. For multi-step tasks (>2 tool calls), call task_note() after each major step — notes are forwarded to Discord automatically. Your final text response is the reply; don't call send_discord to summarize.
8. Give one clear response. Do not send multiple messages saying the same thing.
9. If the same approach fails twice, STOP and report what you tried and what's blocking you. Do not keep retrying variations of the same broken approach.
10. For long-running commands (docker build, npm install, git clone large repos), pass timeout=3600 or higher to run_shell — there is no task-level timeout.
11. When you receive a task prefixed [A2A from X], another agent has delegated work to you. Complete the task, then send your result back to agent-comms:
    send_discord({comms_id}, '{{"from": "{settings.agent_name}", "to": "X", "task": "result", "payload": "your answer"}}')
    When delegating, poll read_discord({comms_id}) every few tool calls to check for their reply.
12. When genuinely uncertain about something that would change your approach — ambiguous requirements, a destructive/irreversible action, missing credentials, or a fork between two valid paths — call ask_user_question() to pause and get clarification. Ask one clear question at a time.
13. DELEGATION — When a task has clearly separable sub-tasks AND other agents are online, split the work:
    a. Identify what can run in parallel (e.g. "fix frontend tests" vs "fix backend tests").
    b. Delegate one sub-task via agent-comms JSON: send_discord({comms_id}, '{{"from": "{settings.agent_name}", "to": "PEER_NAME", "task": "DESCRIPTION", "payload": ""}}')
    c. Work your own sub-task simultaneously.
    d. Poll read_discord({comms_id}) periodically until you see their result ("task": "result").
    e. Combine both results in your final reply.
    Do NOT delegate if you are the only agent online, if the task is trivially small, or if you have already started doing the work yourself.

## Long tasks — checkpointing (IMPORTANT)
For any task with more than ~5 tool calls, use the task journal to avoid losing work:
- Call task_resume() FIRST — check if this task was previously attempted.
- Call task_note(note) after EVERY significant step: what you did, what you found, what comes next.
- Write notes in plain English, NOT JSON. Example: "Checked CI run 23057002906. Backend and security jobs failed — root cause is PYTHONPATH missing in ci-cd.yml line 186. Fixed that. Next: fix security-scan permissions."
- NEVER pass a JSON object or dict as the note — just write a sentence or two.
- Call task_journal_clear() only after the task is fully and successfully complete.
- If you hit a rate limit or error mid-task, your next run should start with task_resume() to pick up from the last note.

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

    # Extended thinking: on by default for capable Claude models.
    # Disabled when: THINKING_ENABLED=false, non-Claude model, or Haiku
    # (Haiku does not support the thinking API).
    model_settings = None
    is_claude = "claude" in model_string
    is_haiku = "haiku" in model_string
    if settings.thinking_enabled and is_claude and not is_haiku:
        model_settings = {
            "thinking": {
                "type": "enabled",
                "budget_tokens": settings.thinking_budget_tokens,
            }
        }
        log.info(
            "thinking_enabled",
            model=model_string,
            budget_tokens=settings.thinking_budget_tokens,
        )

    agent: Agent = Agent(  # type: ignore[type-arg]
        model=model_string,
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
