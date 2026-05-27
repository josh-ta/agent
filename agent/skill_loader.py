"""Load and compose event-analysis skill blocks for task prompts."""

from __future__ import annotations

from agent.config import settings
from agent.task_router import (
    matching_event_analysis_skills,
    primary_event_analysis_skill,
    requires_database_analytics,
    requires_database_csv_export,
    requires_database_query,
)

_SKILL_HEADERS: dict[str, str] = {
    "sale-day-focus": "## Sale-day focus — rank today's buy priorities\n",
    "event-spec-analysis": "## Event spec / price-drop prediction\n",
    "event-hold-analysis": "## Event hold analysis — sticky demand\n",
    "price-history-analysis": "## Price history analysis\n",
    "events-schema-reference": "## Events schema reference\n",
    "discord-analyst-replies": "## Discord analyst reply format\n",
}

_SKILL_MAX_CHARS: dict[str, int] = {
    "sale-day-focus": 2800,
    "event-spec-analysis": 3500,
    "event-hold-analysis": 2800,
    "price-history-analysis": 2800,
    "events-schema-reference": 2200,
    "discord-analyst-replies": 1500,
}

_FIRST_ACTION: dict[str, str] = {
    "sale-day-focus": (
        "Query `events` for **sales starting today**, rank top N for **buy focus**, explain each pick."
    ),
    "event-spec-analysis": (
        "Query `events`, rank top N **spec** candidates (expect price drop before show), explain each."
    ),
    "event-hold-analysis": (
        "Query `events`, rank top N **hold** candidates (sticky demand / won't drop much), explain each."
    ),
    "price-history-analysis": (
        "Discover price history tables if needed, analyze drops over time, then recommend."
    ),
}


def read_skill(stem: str, *, max_chars: int | None = None) -> str:
    path = settings.skills_path / f"{stem}.md"
    if not path.exists():
        return ""
    cap = max_chars if max_chars is not None else _SKILL_MAX_CHARS.get(stem, 3000)
    return path.read_text(encoding="utf-8").strip()[:cap]


def build_event_analysis_blocks(content: str, metadata: dict | None = None) -> str:
    """Skill bodies to inject for Postgres event analytics tasks."""
    if not requires_database_query(content, metadata=metadata):
        return ""

    parts: list[str] = []
    for stem in matching_event_analysis_skills(content, metadata=metadata):
        body = read_skill(stem)
        if body:
            header = _SKILL_HEADERS.get(stem, f"## {stem}\n")
            parts.append(f"{header}{body}")

    schema = read_skill("events-schema-reference")
    if schema:
        parts.append(f"{_SKILL_HEADERS['events-schema-reference']}{schema}")

    if requires_database_analytics(content, metadata=metadata) and not requires_database_csv_export(
        content, metadata=metadata
    ):
        discord = read_skill("discord-analyst-replies")
        if discord:
            parts.append(f"{_SKILL_HEADERS['discord-analyst-replies']}{discord}")

    return "\n\n".join(parts)


def analytics_first_action(content: str, metadata: dict | None = None) -> str:
    """One-line FIRST ACTION instruction for non-streaming analytics runs."""
    primary = primary_event_analysis_skill(content, metadata=metadata)
    if primary and primary in _FIRST_ACTION:
        return (
            f"\n## FIRST ACTION REQUIRED\n"
            f"{_FIRST_ACTION[primary]}\n"
            "Use `query_postgres()` — do NOT ask for criteria or claim you lack database access.\n"
        )
    return (
        "\n## FIRST ACTION REQUIRED\n"
        "This is a Postgres analytics question. Your FIRST step MUST be `query_postgres()` "
        "against the `events` table (see schema reference). "
        "Do NOT answer from memory or claim you lack database access.\n"
    )
