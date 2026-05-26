from __future__ import annotations

import pytest

from agent.task_router import classify_execution_mode


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("hello", "chat"),
        ("Hi chat", "chat"),
        ("thanks", "chat"),
        ("no problem", "chat"),
        (
            "Pull me a list of all upcoming events in stadiums and arenas with a ticket limit of 4.",
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
