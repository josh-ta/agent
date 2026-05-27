from __future__ import annotations

import pytest
from types import SimpleNamespace

from agent.intent_router import IntentRouter, RoutingDecision
from agent.loop import AgentLoop, Task
from agent.loop_services import TaskContextBuilder


@pytest.mark.asyncio
async def test_process_applies_routing_before_build(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class _StubRouter:
        async def route(self, *, task, session_context=""):
            del session_context
            return RoutingDecision(
                intent="database_analytics",
                execution_mode="agent",
                tier="smart",
                needs_tools=True,
                export_csv=False,
                effective_request="Rank top 5 events with sales starting today from Postgres",
                suggested_tools=["query_postgres"],
                reasoning="test",
            )

    class _StubExecutor:
        async def run(self, *, task, agent, base_prompt, tier, message_history=None):
            captured["content"] = task.content
            captured["metadata"] = task.metadata
            captured["tier"] = tier
            return SimpleNamespace(
                output="done",
                tool_calls=2,
                user_visible_reply_sent=False,
                attachments=[],
                shell_failures=[],
                input_chars=1,
                output_chars=4,
                cancelled=False,
                waiting_for_user=False,
            )

    loop = AgentLoop({"smart": object(), "fast": object(), "best": object()})
    loop._intent_router = _StubRouter()  # type: ignore[assignment]
    loop._run_executor = _StubExecutor()  # type: ignore[assignment]

    async def _ensure_answer_required(**kw):
        return "done", True

    monkeypatch.setattr(loop, "_ensure_answer_required", _ensure_answer_required)

    task = Task(content="you have access to my database", source="discord", channel_id=1)
    task.metadata = {"session_id": "discord:1:1", "task_id": "t-1"}

    result = await loop._process(task)

    assert "Rank top 5 events" in captured["content"]
    assert captured["metadata"]["routing"]["intent"] == "database_analytics"
    assert captured["tier"] == "smart"
    assert result.output == "done"


def test_task_context_builder_includes_routing_hint() -> None:
    builder = TaskContextBuilder(None)
    hint = builder._load_routing_hint(
        {
            "routing": RoutingDecision(
                intent="database_analytics",
                needs_tools=True,
                suggested_tools=["query_postgres"],
                reasoning="query the events table",
            ).to_metadata()
        }
    )
    assert "Routing plan" in hint
    assert "query_postgres" in hint
