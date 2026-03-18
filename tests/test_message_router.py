from __future__ import annotations

from types import SimpleNamespace

from agent.communication.message_router import MessageKind, a2a_to_task_content, classify


def test_classify_ignores_own_messages(discord_channels, fake_client, fake_message_factory) -> None:
    message = fake_message_factory(
        channel=discord_channels["private"],
        content="hello",
        author=fake_client.user,
    )

    parsed = classify(message, fake_client.user)

    assert parsed.kind == MessageKind.IGNORE


def test_classify_accepts_targeted_a2a_payload(discord_channels, fake_client, fake_message_factory) -> None:
    message = fake_message_factory(
        channel=discord_channels["comms"],
        content='{"from":"peer-1","to":"agent-1","task":"review","payload":"check tests"}',
        author=SimpleNamespace(display_name="peer-1", bot=True),
    )

    parsed = classify(message, fake_client.user)

    assert parsed.kind == MessageKind.A2A
    assert parsed.a2a_payload is not None
    assert parsed.a2a_payload["task"] == "review"


def test_classify_ignores_non_actionable_a2a_payloads(discord_channels, fake_client, fake_message_factory) -> None:
    message = fake_message_factory(
        channel=discord_channels["comms"],
        content='{"from":"peer-1","to":"agent-1","task":"result","payload":"done"}',
        author=SimpleNamespace(display_name="peer-1", bot=True),
    )

    parsed = classify(message, fake_client.user)

    assert parsed.kind == MessageKind.IGNORE


def test_classify_human_broadcast_in_comms_becomes_task(discord_channels, fake_client, fake_message_factory) -> None:
    message = fake_message_factory(
        channel=discord_channels["comms"],
        content="Can anyone summarize the failure?",
    )

    parsed = classify(message, fake_client.user)

    assert parsed.kind == MessageKind.TASK
    assert parsed.content == "Can anyone summarize the failure?"


def test_classify_other_bot_mention_in_comms_is_ignored(discord_channels, fake_client, fake_message_factory) -> None:
    other_bot = SimpleNamespace(id=123, bot=True, display_name="other-bot")
    message = fake_message_factory(
        channel=discord_channels["comms"],
        content="<@123> only you should answer",
        mentions=[other_bot],
    )

    parsed = classify(message, fake_client.user)

    assert parsed.kind == MessageKind.IGNORE


def test_classify_bus_mention_becomes_task_and_strips_mention(discord_channels, fake_client, fake_message_factory) -> None:
    message = fake_message_factory(
        channel=discord_channels["bus"],
        content="<@999> please investigate",
        mentions=[fake_client.user],
    )

    parsed = classify(message, fake_client.user)

    assert parsed.kind == MessageKind.TASK
    assert parsed.content == "please investigate"


def test_a2a_to_task_content_includes_extra_context() -> None:
    rendered = a2a_to_task_content(
        {"from": "peer-1", "task": "review this change", "payload": "look at startup code"}
    )

    assert "[A2A from peer-1] review this change" in rendered
    assert "Additional context: look at startup code" in rendered
