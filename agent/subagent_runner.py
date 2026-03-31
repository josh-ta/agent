"""
Nested agent runs (CC-style subagents): isolated history, capped tools, summary back to parent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from agent.config import settings
from agent.core_services import ModelFactory, PeerAgentProvider
from agent.metrics import Metrics
from agent.permissions import get_permission_engine
from agent.tools.subagent_attach import attach_subagent_tools

if TYPE_CHECKING:
    from agent.memory.postgres_store import PostgresStore
    from agent.memory.sqlite_store import SQLiteStore

log = structlog.get_logger()

VALID_PROFILES = frozenset({"minimal", "explore", "verify"})

_subagent_prompts = {
    "minimal": (
        "You are a sub-agent inside a larger task. Answer one focused question.\n"
        "Use only tools you have; report concrete paths, symbols, facts.\n"
        "Return a concise summary for the parent. No Discord."
    ),
    "explore": (
        "You explore the codebase: find relevant files, symbols, patterns.\n"
        "Read before concluding; prefer search_files and read_file; "
        "use task_resume if a journal exists.\n"
        "Summarize: bullet paths + one-line relevance each. No Discord."
    ),
    "verify": (
        "You verify behavior via read_file, search_files, read-only shell (tests, linters).\n"
        "Do not modify files. Report pass/fail with evidence excerpts. No Discord."
    ),
}


class SubagentRunner:
    """Builds short-lived nested agents with a restricted tool surface."""

    def __init__(
        self,
        *,
        sqlite: SQLiteStore | None,
        postgres: PostgresStore | None,
    ) -> None:
        self._sqlite = sqlite
        self._postgres = postgres
        self._model_factory = ModelFactory()

    def _build_subagent(self, profile: str) -> Agent:  # type: ignore[type-arg]
        model_string = settings.model_string_for("fast")
        mcp_servers = self._model_factory.mcp_servers()
        model = self._model_factory.build_model(model_string)
        model_settings = self._model_factory.model_settings(model_string)

        agent: Agent = Agent(  # type: ignore[type-arg]
            model=model,
            mcp_servers=mcp_servers,
            model_settings=model_settings,
            retries=1,
        )

        prompt_body = _subagent_prompts.get(profile, _subagent_prompts["explore"])

        @agent.system_prompt(dynamic=True)
        async def _system_prompt() -> str:
            provider = PeerAgentProvider(self._postgres)
            peers = await provider.list_other_agents()
            peer_line = ""
            if peers:
                lines = "\n".join(f"- {n}" for n in peers)
                peer_line = f"\n## Peers (read-only context)\n{lines}\n"
            return f"{prompt_body}{peer_line}"

        attach_subagent_tools(
            agent,
            sqlite=self._sqlite,
            postgres=self._postgres,
            profile=profile,
        )
        return agent

    async def run(
        self,
        *,
        instruction: str,
        profile: str = "explore",
        max_tool_calls: int | None = None,
    ) -> str:
        prof = (profile or "explore").strip().lower()
        if prof not in VALID_PROFILES:
            return f"[subagent error: unknown profile {profile!r}; use {sorted(VALID_PROFILES)}]"

        cap = settings.subagent_instruction_char_cap
        instr = instruction.strip()
        if len(instr) > cap:
            instr = instr[:cap] + "\n[...truncated instruction...]"

        mtc = max_tool_calls if max_tool_calls is not None else settings.subagent_max_tool_calls
        mtc = max(1, min(mtc, 80))

        sub = self._build_subagent(prof)
        log.info("subagent_run_start", profile=prof, max_tool_calls=mtc)
        try:
            result = await sub.run(
                instr,
                usage_limits=UsageLimits(
                    request_limit=40,
                    tool_calls_limit=mtc,
                ),
            )
        except Exception as exc:
            log.warning("subagent_run_failed", exc=str(exc))
            return f"[subagent failed: {exc}]"

        out = str(result.output).strip()
        cap_out = settings.subagent_output_char_cap
        if len(out) > cap_out:
            out = out[:cap_out] + "\n[...subagent output truncated...]"
        log.info("subagent_run_done", profile=prof, chars=len(out))
        return out or "(subagent returned empty output)"


def _parent_tool_perm_block(tool_name: str, **kwargs: Any) -> str | None:
    eng = get_permission_engine()
    if eng is None:
        return None
    result = eng.check_sync(tool_name, kwargs)
    if result.ok:
        return None
    Metrics.inc_permission_denied()
    return result.message


def attach_run_subagent_tool(
    agent: Agent,  # type: ignore[type-arg]
    runner: SubagentRunner,
) -> None:
    """Parent-only tool: delegates to SubagentRunner (child agents must not register this)."""

    @agent.tool_plain
    async def run_agent_subtask(
        instruction: str,
        profile: str = "explore",
        max_tool_calls: int = 0,
    ) -> str:
        """Isolated sub-agent (explore|verify|minimal); returns a text summary."""
        if msg := _parent_tool_perm_block(
            "run_agent_subtask",
            instruction=instruction,
            profile=profile,
            max_tool_calls=max_tool_calls,
        ):
            return msg
        mtc = max_tool_calls if max_tool_calls and max_tool_calls > 0 else None
        return await runner.run(instruction=instruction, profile=profile, max_tool_calls=mtc)
