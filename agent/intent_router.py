"""LLM intent router: classify user requests against available tools before execution."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Literal

import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from agent.config import settings
from agent.tool_catalog import build_tool_catalog, format_tool_catalog

if TYPE_CHECKING:
    from agent.loop import Task

log = structlog.get_logger()

IntentKind = Literal[
    "social_chat",
    "database_analytics",
    "database_csv_export",
    "code_work",
    "shell_ops",
    "research",
    "general_work",
]

ExecutionMode = Literal["chat", "agent"]
Tier = Literal["fast", "smart", "best"]

_CORRECTION_RE = re.compile(
    r"\b(you have|i told you|use the|access to my|check the)\b.*\b(database|postgres|db)\b",
    re.IGNORECASE,
)


class RoutingDecision(BaseModel):
    intent: IntentKind = "general_work"
    execution_mode: ExecutionMode = "agent"
    tier: Tier = "smart"
    needs_tools: bool = True
    export_csv: bool = False
    fold_with_previous: bool = False
    effective_request: str = Field(
        default="",
        description="Full user request to execute. When fold_with_previous is true, merge prior context into one clear ask.",
    )
    suggested_tools: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="", max_length=400)

    def to_metadata(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_metadata(cls, raw: object) -> RoutingDecision | None:
        if not isinstance(raw, dict):
            return None
        try:
            return cls.model_validate(raw)
        except Exception:
            return None


def _extract_last_user_turn(session_context: str) -> str:
    if not session_context.strip():
        return ""
    for line in reversed(session_context.splitlines()):
        stripped = line.strip()
        if stripped.lower().startswith("user:"):
            return stripped.split(":", 1)[-1].strip()
        if stripped.startswith("- **user**"):
            return stripped.split(":", 1)[-1].strip()
    return ""


def heuristic_route(
    *,
    content: str,
    session_context: str = "",
    postgres_available: bool = False,
) -> RoutingDecision:
    """Regex/heuristic fallback when the LLM router is disabled or fails."""
    from agent.task_router import (
        classify_execution_mode,
        requires_database_csv_export,
        requires_database_query,
        requires_tool_use,
    )

    text = content.strip()
    mode = classify_execution_mode(text, source="discord")
    needs_tools = requires_tool_use(text)
    db_query = requires_database_query(text)
    export_csv = requires_database_csv_export(text)

    if mode == "chat" and not db_query:
        return RoutingDecision(
            intent="social_chat",
            execution_mode="chat",
            tier="fast",
            needs_tools=False,
            export_csv=False,
            effective_request=text,
            reasoning="heuristic: social/chat message",
        )

    intent: IntentKind = "general_work"
    if export_csv:
        intent = "database_csv_export"
    elif db_query:
        intent = "database_analytics"
    elif re.search(r"\b(code|implement|fix|debug|refactor|\.py\b)", text, re.I):
        intent = "code_work"
    elif re.search(r"\b(run|shell|deploy|docker|ssh)\b", text, re.I):
        intent = "shell_ops"
    elif re.search(r"\b(research|search the web|look up)\b", text, re.I):
        intent = "research"

    tier: Tier = "smart"
    from agent.loop import _classify_tier

    tier = _classify_tier(text)  # type: ignore[assignment]

    suggested: list[str] = []
    if db_query and postgres_available:
        suggested = ["list_postgres_tables", "query_postgres"]

    effective = text
    fold = False
    if _CORRECTION_RE.search(text) and session_context.strip():
        prior = _extract_last_user_turn(session_context)
        if prior:
            fold = True
            effective = f"{prior}\n\n(Operator clarification: {text})"

    return RoutingDecision(
        intent=intent,
        execution_mode="agent",
        tier=tier,
        needs_tools=needs_tools or db_query,
        export_csv=export_csv,
        fold_with_previous=fold,
        effective_request=effective,
        suggested_tools=suggested,
        reasoning="heuristic fallback",
    )


_ROUTER_SYSTEM = """\
You are a routing classifier for an autonomous Discord agent named Bob.

Given the user's message, recent session context, and available tools, output a routing decision.

Rules:
- social_chat: greetings, thanks, acknowledgments only — no tools needed.
- database_analytics: questions about events, sales, tickets, venues, rankings, reports from Postgres — needs query_postgres (NOT csv unless user asks for a file).
- database_csv_export: user explicitly wants a CSV/file/download/export/resend attachment.
- code_work: implement, fix, debug, refactor code.
- shell_ops: deploy, docker, ssh, run commands.
- research: web lookup when database is not the source.
- general_work: everything else that needs tools.

- needs_tools=false ONLY for pure social_chat.
- export_csv=true ONLY when the user wants a file attachment (csv, export, download, resend file).
- fold_with_previous=true when the current message is a short clarification/correction that completes an earlier unanswered request in session context (e.g. "you have access to my database" after a data question). Set effective_request to one combined clear instruction.
- tier: fast for trivial chat; smart for database/code/shell; best for architecture/security/deep analysis.
- suggested_tools: list tool names likely needed (from the catalog only).

Never tell the user you lack database access when list_postgres_tables/query_postgres are available.
"""


class IntentRouter:
    def __init__(
        self,
        *,
        router_agent: Agent | None = None,  # type: ignore[type-arg]
        postgres_available: bool | None = None,
    ) -> None:
        self._agent = router_agent
        self._postgres_available = (
            postgres_available
            if postgres_available is not None
            else bool(settings.postgres_url.strip())
        )

    @staticmethod
    def create_router_agent() -> Agent:  # type: ignore[type-arg]
        from agent.core_services import ModelFactory

        factory = ModelFactory()
        model_string = settings.model_string_for("fast")
        model = factory.build_model(model_string)
        model_settings = factory.model_settings(model_string)
        return Agent(
            model=model,
            model_settings=model_settings,
            output_type=RoutingDecision,
            system_prompt=_ROUTER_SYSTEM,
            retries=1,
        )

    @staticmethod
    def create_router_agent_or_none() -> Agent | None:  # type: ignore[type-arg]
        try:
            return IntentRouter.create_router_agent()
        except Exception as exc:
            log.warning("intent_router_agent_unavailable", error=str(exc))
            return None

    async def route(
        self,
        *,
        task: Task,
        session_context: str = "",
    ) -> RoutingDecision:
        content = task.content.strip()
        if not content:
            return heuristic_route(content=content, session_context=session_context)

        if not settings.intent_router_enabled:
            return merge_routing_with_heuristics(
                heuristic_route(
                    content=content,
                    session_context=session_context,
                    postgres_available=self._postgres_available,
                ),
                content=content,
                session_context=session_context,
                postgres_available=self._postgres_available,
            )

        if self._agent is None:
            return merge_routing_with_heuristics(
                heuristic_route(
                    content=content,
                    session_context=session_context,
                    postgres_available=self._postgres_available,
                ),
                content=content,
                session_context=session_context,
                postgres_available=self._postgres_available,
            )

        catalog = build_tool_catalog(
            postgres_available=self._postgres_available,
            sqlite_available=True,
        )
        prompt_parts = [
            format_tool_catalog(catalog),
            "",
            f"Postgres connected: {self._postgres_available}",
            "",
        ]
        if session_context.strip():
            prompt_parts.extend(["Recent session:", session_context.strip()[:2200], ""])
        meta = task.metadata or {}
        if meta.get("attachments"):
            prompt_parts.append("Note: user attached files — likely needs agent mode with read_file.")
        prompt_parts.extend(["Current user message:", content[:4000]])
        prompt = "\n".join(prompt_parts)

        try:
            result = await self._agent.run(prompt, usage_limits=UsageLimits(request_limit=5))
            decision = result.output
            if not isinstance(decision, RoutingDecision):
                decision = RoutingDecision.model_validate(decision)
            decision = self._normalize(decision, content=content, session_context=session_context)
            decision = merge_routing_with_heuristics(
                decision,
                content=content,
                session_context=session_context,
                postgres_available=self._postgres_available,
            )
            log.info(
                "intent_routed",
                intent=decision.intent,
                mode=decision.execution_mode,
                tier=decision.tier,
                needs_tools=decision.needs_tools,
                export_csv=decision.export_csv,
                fold=decision.fold_with_previous,
                reasoning=decision.reasoning[:120],
            )
            return decision
        except Exception as exc:
            log.warning("intent_router_failed", error=str(exc))
            return merge_routing_with_heuristics(
                heuristic_route(
                    content=content,
                    session_context=session_context,
                    postgres_available=self._postgres_available,
                ),
                content=content,
                session_context=session_context,
                postgres_available=self._postgres_available,
            )

    def _normalize(
        self,
        decision: RoutingDecision,
        *,
        content: str,
        session_context: str,
    ) -> RoutingDecision:
        effective = (decision.effective_request or content).strip()
        if decision.fold_with_previous and not decision.effective_request.strip():
            prior = _extract_last_user_turn(session_context)
            if prior:
                effective = f"{prior}\n\n(Operator clarification: {content})"

        if decision.intent == "database_csv_export":
            decision.export_csv = True
            decision.needs_tools = True
            decision.execution_mode = "agent"
        elif decision.intent == "database_analytics":
            decision.export_csv = False
            decision.needs_tools = True
            decision.execution_mode = "agent"
        elif decision.intent == "social_chat":
            decision.needs_tools = False
            decision.export_csv = False
            decision.execution_mode = "chat"

        if decision.export_csv:
            decision.needs_tools = True
            decision.execution_mode = "agent"
            decision.intent = "database_csv_export"

        if decision.needs_tools and decision.execution_mode == "chat":
            decision.execution_mode = "agent"

        if not effective:
            effective = content

        return decision.model_copy(update={"effective_request": effective})


def merge_routing_with_heuristics(
    decision: RoutingDecision,
    *,
    content: str,
    session_context: str = "",
    postgres_available: bool = False,
) -> RoutingDecision:
    """Ensure the LLM router cannot disable tools for obvious database/work requests."""
    floor = heuristic_route(
        content=content,
        session_context=session_context,
        postgres_available=postgres_available,
    )
    updates: dict[str, object] = {}
    if floor.needs_tools and not decision.needs_tools:
        updates["needs_tools"] = True
        updates["execution_mode"] = "agent"
    if floor.export_csv and not decision.export_csv:
        updates["export_csv"] = True
        updates["intent"] = "database_csv_export"
        updates["needs_tools"] = True
        updates["execution_mode"] = "agent"
    elif floor.intent == "database_analytics" and decision.intent in {"social_chat", "general_work"}:
        updates["intent"] = "database_analytics"
        updates["needs_tools"] = True
        updates["execution_mode"] = "agent"
    if floor.suggested_tools and not decision.suggested_tools:
        updates["suggested_tools"] = floor.suggested_tools
    if floor.fold_with_previous and not decision.fold_with_previous:
        updates["fold_with_previous"] = True
        updates["effective_request"] = floor.effective_request
    if updates:
        decision = decision.model_copy(update=updates)
    return decision


def parse_routing_json(text: str) -> RoutingDecision | None:
    try:
        payload = json.loads(text)
        return RoutingDecision.model_validate(payload)
    except Exception:
        return None
