"""
Helper services for prompt/model construction in `agent.core`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agent.config import settings

log = structlog.get_logger()


class PromptSources:
    def load_skills_compact(self, skills_path: Path) -> str:
        if not skills_path.exists():
            return ""

        lines = ["## Skills (use read_skill <name> to load full content)"]
        for skill_file in sorted(skills_path.glob("*.md")):
            if skill_file.name.startswith("_"):
                continue
            desc = ""
            for line in skill_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    desc = line[:80]
                    break
            lines.append(f"- **{skill_file.stem}**: {desc}")
        return "\n".join(lines)

    def load_identity(self, identity_path: Path) -> str:
        if not identity_path.exists():
            return ""

        memory_char_cap = 4_000
        identity_char_cap = 2_000
        sections: list[str] = []
        for filename in ("IDENTITY.md", "GOALS.md", "MEMORY.md"):
            fp = identity_path / filename
            if not fp.exists():
                continue
            text = fp.read_text(encoding="utf-8").strip()
            cap = memory_char_cap if filename == "MEMORY.md" else identity_char_cap
            if len(text) > cap:
                text = "[...truncated...]\n\n" + text[-cap:]
                log.warning("identity_file_truncated", file=filename, cap=cap)
            sections.append(text)
        return "\n\n".join(sections)


class SystemPromptBuilder:
    def __init__(self, prompt_sources: PromptSources | None = None) -> None:
        self._sources = prompt_sources or PromptSources()

    def build(self, *, other_agents: list[str] | None = None) -> str:
        identity = self._sources.load_identity(settings.identity_path)
        skills = self._sources.load_skills_compact(settings.skills_path)

        comms_id = settings.discord_comms_channel_id
        bus_id = settings.discord_bus_channel_id
        private_id = settings.discord_agent_channel_id

        peers_block = ""
        if other_agents:
            peers_block = (
                "\n## Peer Agents (online now)\n"
                + "\n".join(f"- {name}" for name in other_agents)
                + "\n"
            )

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
8. For multi-step tasks (>2 tool calls), narrate your progress with task_note():
   - Call it before the first tool with a short plan.
   - Call it after each major step with: what you did, what you found, and what comes next.
   - If the task is still running after a noticeable pause, send another task_note so the user never sees a silent hang.
   Do not use send_discord to restate your final answer.
9. Give one clear response. Do not send multiple messages saying the same thing.
10. If the same approach fails twice, STOP and report what you tried and what is blocking you.
11. For long-running commands, pass timeout=3600 or higher to run_shell.
11a. In your private Discord channel, prefer native control commands when appropriate:
    `/status`, `/cancel`, `/replace <task>`, `/queue <task>`, `/clear`, `/resume`, `/forget`, `/help`.
12. Agent-comms is for actionable task handoffs and final results only. Never send receipt acks, thank-you notes, or status chatter there.
13. When you receive a task prefixed [A2A from X], complete it and send the result back to agent-comms:
    send_discord({comms_id}, '{{"from": "{settings.agent_name}", "to": "X", "task": "result", "payload": "your answer"}}')
    When delegating, poll read_discord({comms_id}) every few tool calls for replies. Treat incoming "result", "update", and "status" messages as information to consume silently, not messages to answer.
14. If uncertainty would change the approach, call ask_user_question() and ask one clear question.
15. Delegate only when the work splits cleanly and other agents are online:
    a. Identify parallelizable sub-tasks.
    b. Delegate one sub-task via agent-comms JSON: send_discord({comms_id}, '{{"from": "{settings.agent_name}", "to": "PEER_NAME", "task": "DESCRIPTION", "payload": ""}}')
    c. Work your own sub-task simultaneously.
    d. Poll read_discord({comms_id}) periodically until you see their result ("task": "result").
    e. Combine both results in your final reply.
    Do not delegate trivial work, solo work, or work you have already started.
16. Secrets are available via secret tools and secret-aware browser tools:
    - Never reveal secret values in Discord, task notes, memory, files, or final answers.
    - Never paste secrets into shell commands when a secret-aware tool can do the job.
    - Use secret_list() to discover names, secret_get(name) only when raw text is unavoidable,
      and browser_fill_secret/browser_type_secret for login forms.
17. If the user message includes attachments, inspect the attachment summary/path context first.
    Use read_file on saved attachment paths when you need deeper inspection. Images and PDFs may
    already be attached to the model input, so reason over them directly when helpful.

## Long tasks — checkpointing (IMPORTANT)
For tasks with more than ~5 tool calls, use the task journal:
- Call task_resume() FIRST — check if this task was previously attempted.
- Call task_note(note) before the first tool and after EVERY significant step: what you did, what you found, what comes next.
- Never stay silent for long stretches on an active task; add another task_note when progress is slow or blocked.
- Write notes in plain English, not JSON.
- Call task_journal_clear() only after success.
- After a rate limit or interruption, resume with task_resume().

## Git / GitHub
- Use the existing checkout in WORKSPACE_PATH when one is already present.
- Before repo-specific git/gh commands, discover the repo root with `git rev-parse --show-toplevel`, `pwd`, or `ls`.
- Only use `/workspace/<repo-name>` after you clone that repo yourself.
- Before any SSH or remote deploy command, inspect the local workspace first and ground the target repo/path there when possible.
- Risky remote shell commands are blocked unless `run_shell` starts with a first-line comment like:
  `# remote-preflight: workspace=/workspace/<repo>; basis=user-provided host`
- `gh` CLI is pre-authenticated via GH_TOKEN env var — NEVER run `gh auth login`.
- If `gh auth status` exits 0, you ARE authenticated — proceed directly to `gh pr create`.
- Never guess a GitHub repo slug from memory or narrative context. Use the checked-out repo, confirm the slug with `gh repo view`, or ask the user.
- Never guess VPS hostnames, IPs, repo roots, or deploy paths. If they are not explicit in the workspace or from the user, ask.
- Create PRs with: `gh pr create --title "..." --body "..." --base main --repo owner/repo`
- SSH key is at /data/ssh/id_ed25519 (or id_rsa). Plain `git clone git@github.com:...` should work.
- Always set git user inside cloned repos: `git config user.name "bob-agent" && git config user.email "bob@agent.local"`

{skills}
"""
        prompt = base.strip()
        max_chars = 40_000
        if len(prompt) > max_chars:
            prompt = (
                prompt[:max_chars]
                + "\n[...system prompt truncated — MEMORY.md too large, use identity_edit to trim it...]"
            )
            log.warning("system_prompt_truncated", chars=len(base), cap=max_chars)
        log.info("system_prompt_built", chars=len(prompt))
        return prompt


class PeerAgentProvider:
    def __init__(self, store: Any | None) -> None:
        self._store = store

    async def list_other_agents(self) -> list[str]:
        if self._store is None:
            return []
        try:
            import asyncio

            raw = await asyncio.wait_for(self._store.list_agents(), timeout=2.0)
            others: list[str] = []
            for line in raw.splitlines():
                line = line.strip("- •\t ")
                if not line or line.endswith(":") or "[" not in line:
                    continue
                if settings.agent_name.lower() not in line.lower():
                    name = line.split()[0].rstrip(":")
                    if name:
                        others.append(name)
            return others
        except Exception:
            return []


class ModelFactory:
    def mcp_servers(self) -> list[MCPServerHTTP]:
        servers: list[MCPServerHTTP] = []
        if settings.browser_mcp_url:
            try:
                servers.append(MCPServerHTTP(url=settings.browser_mcp_url))
            except Exception as exc:
                log.warning("browser_mcp_unavailable", error=str(exc))
        return servers

    def build_model(self, model_string: str) -> str | OpenAIChatModel:
        if model_string.startswith("xai:"):
            return self._build_openai_compatible_model(
                model_string.split(":", 1)[1],
                base_url=settings.xai_base_url,
                api_key=settings.secret_value(settings.xai_api_key),
            )
        if model_string.startswith("mistral:"):
            return self._build_openai_compatible_model(
                model_string.split(":", 1)[1],
                base_url=settings.mistral_base_url,
                api_key=settings.secret_value(settings.mistral_api_key),
            )
        if model_string.startswith("openai:") and settings.openai_base_url:
            return self._build_openai_compatible_model(
                model_string.split(":", 1)[1],
                base_url=settings.openai_base_url,
                api_key=settings.secret_value(settings.openai_api_key),
            )
        return model_string

    def model_settings(self, model_string: str) -> dict | None:
        is_claude = model_string.startswith("anthropic:") or "claude" in model_string
        is_haiku = "haiku" in model_string
        is_openai = model_string.startswith(("openai:", "xai:", "mistral:"))
        is_reasoning_model = any(x in model_string for x in ("o1", "o3", "o4"))
        is_gemini = "google" in model_string or "gemini" in model_string
        is_groq = "groq" in model_string

        model_settings: dict[str, Any] = {}
        if is_claude and not is_haiku:
            model_settings["anthropic_cache_instructions"] = "1h"
            model_settings["anthropic_cache_tool_definitions"] = "1h"
            model_settings["anthropic_cache_messages"] = "1h"

        if settings.thinking_enabled and is_claude and not is_haiku:
            model_settings["anthropic_thinking"] = {
                "type": "enabled",
                "budget_tokens": settings.thinking_budget_tokens,
            }
            # Anthropic requires max_tokens to be greater than thinking.budget_tokens.
            # Reserve additional room for the visible answer instead of just adding 1.
            model_settings["max_tokens"] = max(4096, settings.thinking_budget_tokens + 1024)

        if is_openai:
            model_settings["openai_prompt_cache_key"] = settings.agent_name
            model_settings["openai_prompt_cache_retention"] = "24h"
            if is_reasoning_model and settings.thinking_enabled:
                model_settings["openai_reasoning_effort"] = "high"

        if is_gemini and settings.thinking_enabled:
            model_settings["google_thinking_config"] = {
                "thinking_budget": settings.thinking_budget_tokens,
            }

        if is_groq and settings.thinking_enabled:
            model_settings["groq_reasoning_format"] = "parsed"

        return model_settings or None

    @staticmethod
    def _build_openai_compatible_model(model_name: str, *, base_url: str, api_key: str) -> OpenAIChatModel:
        provider = OpenAIProvider(base_url=base_url or None, api_key=api_key or None)
        return OpenAIChatModel(model_name, provider=provider)
