from __future__ import annotations

from agent.skill_loader import analytics_first_action, build_event_analysis_blocks, read_skill
from agent.task_router import (
    matching_event_analysis_skills,
    primary_event_analysis_skill,
    requires_event_hold_analysis,
    requires_price_history_analysis,
    requires_sale_day_focus,
)


def _write_skills(skills_dir) -> None:
    for name in (
        "sale-day-focus",
        "event-spec-analysis",
        "event-hold-analysis",
        "price-history-analysis",
        "events-schema-reference",
        "discord-analyst-replies",
        "query-database",
    ):
        skills_dir.joinpath(f"{name}.md").write_text(f"# {name}\n\nBody for {name}.\n", encoding="utf-8")


def test_matching_skills_sale_day_before_spec(isolated_paths: dict) -> None:
    _write_skills(isolated_paths["skills"])
    q = "Of all the events with a sale starting today which 5 events should I focus on buying?"
    assert requires_sale_day_focus(q)
    assert matching_event_analysis_skills(q) == ["sale-day-focus"]
    assert primary_event_analysis_skill(q) == "sale-day-focus"


def test_matching_skills_spec() -> None:
    q = "Which events with upcoming public sales should we spec?"
    assert matching_event_analysis_skills(q) == ["event-spec-analysis"]


def test_matching_skills_hold() -> None:
    q = "Which arena events should I hold because they won't drop on secondary?"
    assert requires_event_hold_analysis(q)
    assert "event-hold-analysis" in matching_event_analysis_skills(q)


def test_matching_skills_price_history() -> None:
    q = "Show me price history drops since onsale for stadium events"
    assert requires_price_history_analysis(q)
    assert matching_event_analysis_skills(q) == ["price-history-analysis"]


def test_build_event_analysis_blocks_includes_schema_and_discord(isolated_paths: dict) -> None:
    _write_skills(isolated_paths["skills"])
    blocks = build_event_analysis_blocks("Which events should we spec?")
    assert "sale-day-focus" not in blocks.lower() or "event-spec" in blocks.lower()
    assert "events-schema-reference" in blocks or "Events schema reference" in blocks
    assert "discord-analyst-replies" in blocks or "Discord analyst" in blocks


def test_analytics_first_action(isolated_paths: dict) -> None:
    _write_skills(isolated_paths["skills"])
    action = analytics_first_action(
        "Of all the events with a sale starting today which 5 events should I focus on buying?"
    )
    assert "FIRST ACTION" in action
    assert "today" in action.lower() or "buy focus" in action.lower()


def test_read_skill_missing_returns_empty(isolated_paths: dict) -> None:
    assert read_skill("nonexistent-skill") == ""
