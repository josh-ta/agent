from __future__ import annotations

import pytest

from agent.task_router import (
    classify_execution_mode,
    requires_database_csv_export,
    requires_database_query,
    requires_database_tools,
    requires_tool_use,
)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("hello", "chat"),
        ("Hi chat", "chat"),
        ("thanks", "chat"),
        ("no problem", "chat"),
        (
            "Of all the events with a sale starting today which 5 events should I focus on buying?",
            "agent",
        ),
        ("Can you export that query result as a csv file?", "agent"),
        ("Fix the login bug in auth.py", "agent"),
        ("```python\nprint(1)\n```", "agent"),
    ],
)
def test_classify_execution_mode(content: str, expected: str) -> None:
    assert classify_execution_mode(content, source="discord") == expected


def test_non_discord_sources_always_agent() -> None:
    assert classify_execution_mode("hello", source="system") == "agent"


def test_attachments_force_agent() -> None:
    assert (
        classify_execution_mode(
            "hello",
            source="discord",
            metadata={"attachments": [{"filename": "x.png"}]},
        )
        == "agent"
    )


def test_requires_database_tools() -> None:
    assert requires_database_tools(
        "Give me a csv file of all upcoming arena/stadium events with a ticket limit of 4"
    )
    assert requires_database_tools("export postgres query as csv") is True
    assert requires_database_tools(
        "Of all the events with a sale starting today which 5 events should I focus on buying?"
    )
    assert requires_database_tools("you have access to my database") is True
    assert requires_database_tools("fix the login bug") is False
    assert requires_database_tools("hello") is False


def test_requires_database_csv_export() -> None:
    assert requires_database_csv_export(
        "Give me a csv file of all upcoming arena/stadium events with a ticket limit of 4"
    )
    assert requires_database_csv_export("can you resend me the csv?") is True
    assert requires_database_csv_export(
        "Of all the events with a sale starting today which 5 events should I focus on buying?"
    ) is False
    assert requires_database_csv_export("you have access to my database") is False


def test_requires_tool_use_for_event_analytics() -> None:
    assert requires_tool_use(
        "Of all the events with a sale starting today which 5 events should I focus on buying?"
    )
    assert requires_tool_use("hello") is False
