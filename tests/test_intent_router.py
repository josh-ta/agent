from __future__ import annotations

import pytest

from agent.intent_router import IntentRouter, RoutingDecision, heuristic_route
from agent.task_router import (
    requires_database_csv_export,
    requires_database_query,
    requires_tool_use,
)


def test_heuristic_route_event_analytics_needs_tools() -> None:
    decision = heuristic_route(
        content="Of all the events with a sale starting today which 5 events should I focus on buying?",
        postgres_available=True,
    )
    assert decision.needs_tools is True
    assert decision.export_csv is False
    assert decision.execution_mode == "agent"
    assert decision.intent == "database_analytics"
    assert "query_postgres" in decision.suggested_tools


def test_heuristic_route_csv_export() -> None:
    decision = heuristic_route(
        content="Give me a csv file of arena events with ticket limit 4",
        postgres_available=True,
    )
    assert decision.export_csv is True
    assert decision.intent == "database_csv_export"


def test_heuristic_route_social_chat() -> None:
    decision = heuristic_route(content="thanks!", postgres_available=True)
    assert decision.execution_mode == "chat"
    assert decision.needs_tools is False


def test_heuristic_fold_database_correction() -> None:
    session = "Recent turns:\n- **user**: Which 5 events should I buy today?\n"
    decision = heuristic_route(
        content="you have access to my database",
        session_context=session,
        postgres_available=True,
    )
    assert decision.fold_with_previous is True
    assert "Which 5 events" in decision.effective_request
    assert "database" in decision.effective_request.lower()


def test_routing_metadata_overrides_requires_tool_use() -> None:
    meta = RoutingDecision(
        intent="database_analytics",
        needs_tools=True,
        export_csv=False,
        effective_request="query events",
    ).to_metadata()
    assert requires_tool_use("hello", metadata={"routing": meta}) is True
    assert requires_database_query("hello", metadata={"routing": meta}) is True
    assert requires_database_csv_export("hello", metadata={"routing": meta}) is False


def test_routing_metadata_csv_export_flag() -> None:
    meta = RoutingDecision(
        intent="database_csv_export",
        needs_tools=True,
        export_csv=True,
        effective_request="export csv",
    ).to_metadata()
    assert requires_database_csv_export("hello", metadata={"routing": meta}) is True


@pytest.mark.asyncio
async def test_intent_router_uses_heuristic_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.config import settings

    monkeypatch.setattr(settings, "intent_router_enabled", False)
    router = IntentRouter(router_agent=None, postgres_available=True)

    class _Task:
        content = "Which 5 events with sales starting today?"
        metadata: dict = {}

    decision = await router.route(task=_Task())  # type: ignore[arg-type]
    assert decision.needs_tools is True
    assert decision.intent == "database_analytics"


@pytest.mark.asyncio
async def test_intent_router_llm_path_with_stub_agent() -> None:
    class _StubAgent:
        async def run(self, prompt: str, usage_limits=None):
            del prompt, usage_limits
            return type(
                "R",
                (),
                {
                    "output": RoutingDecision(
                        intent="database_analytics",
                        execution_mode="agent",
                        tier="smart",
                        needs_tools=True,
                        export_csv=False,
                        effective_request="Rank top 5 events with sales starting today",
                        suggested_tools=["query_postgres"],
                        reasoning="analytics query",
                    )
                },
            )()

    router = IntentRouter(router_agent=_StubAgent(), postgres_available=True)  # type: ignore[arg-type]

    class _Task:
        content = "you have access to my database"
        metadata: dict = {}

    decision = await router.route(
        task=_Task(),  # type: ignore[arg-type]
        session_context="- **user**: Which 5 events should I focus on?",
    )
    assert decision.needs_tools is True
    assert decision.export_csv is False
    assert decision.intent == "database_analytics"
