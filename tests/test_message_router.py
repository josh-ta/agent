from __future__ import annotations

from types import SimpleNamespace

from agent.communication.message_router import (
    MessageKind,
    _normalize_a2a_task,
    _strip_mentions,
    a2a_to_task_content,
    classify,
)


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


def test_classify_human_targeted_mention_in_comms_becomes_task(discord_channels, fake_client, fake_message_factory) -> None:
    message = fake_message_factory(
        channel=discord_channels["comms"],
        content="<@999> investigate this failure",
        mentions=[fake_client.user],
    )

    parsed = classify(message, fake_client.user)

    assert parsed.kind == MessageKind.TASK
    assert parsed.content == "investigate this failure"


def test_classify_other_bot_mention_in_comms_is_ignored(discord_channels, fake_client, fake_message_factory) -> None:
    other_bot = SimpleNamespace(id=123, bot=True, display_name="other-bot")
    message = fake_message_factory(
        channel=discord_channels["comms"],
        content="<@123> only you should answer",
        mentions=[other_bot],
    )

    parsed = classify(message, fake_client.user)

    assert parsed.kind == MessageKind.IGNORE


def test_classify_ignores_a2a_payload_for_other_agent_and_invalid_json(discord_channels, fake_client, fake_message_factory) -> None:
    other_agent = fake_message_factory(
        channel=discord_channels["comms"],
        content='{"from":"peer-1","to":"other-agent","task":"review"}',
        author=SimpleNamespace(display_name="peer-1", bot=True),
    )
    invalid_json = fake_message_factory(
        channel=discord_channels["comms"],
        content='{"from":"peer-1","to":"agent-1","task":"review"',
        author=SimpleNamespace(display_name="peer-1", bot=False),
    )

    assert classify(other_agent, fake_client.user).kind == MessageKind.IGNORE
    parsed = classify(invalid_json, fake_client.user)
    assert parsed.kind == MessageKind.TASK
    assert parsed.content == invalid_json.content


def test_classify_comms_invalid_json_from_bot_and_mentions_only_from_human(
    discord_channels,
    fake_client,
    fake_message_factory,
) -> None:
    invalid_bot_json = fake_message_factory(
        channel=discord_channels["comms"],
        content='{"from":"peer-1","to":"agent-1","task":"review"',
        author=SimpleNamespace(display_name="peer-1", bot=True),
    )
    mentions_only = fake_message_factory(
        channel=discord_channels["comms"],
        content="<@999>",
        mentions=[fake_client.user],
    )

    assert classify(invalid_bot_json, fake_client.user).kind == MessageKind.IGNORE
    parsed = classify(mentions_only, fake_client.user)
    assert parsed.kind == MessageKind.TASK
    assert parsed.content == ""


def test_classify_comms_jsondecodeerror_from_bot_is_ignored(discord_channels, fake_client, fake_message_factory) -> None:
    invalid_json = fake_message_factory(
        channel=discord_channels["comms"],
        content='{"from":"peer-1","to":"agent-1","task":"review",}',
        author=SimpleNamespace(display_name="peer-1", bot=True),
    )

    assert classify(invalid_json, fake_client.user).kind == MessageKind.IGNORE


def test_classify_bus_mention_becomes_task_and_strips_mention(discord_channels, fake_client, fake_message_factory) -> None:
    message = fake_message_factory(
        channel=discord_channels["bus"],
        content="<@999> please investigate",
        mentions=[fake_client.user],
    )

    parsed = classify(message, fake_client.user)

    assert parsed.kind == MessageKind.TASK
    assert parsed.content == "please investigate"


def test_classify_bus_without_mention_becomes_bus_and_private_channel_always_task(
    discord_channels,
    fake_client,
    fake_message_factory,
) -> None:
    bus_message = fake_message_factory(channel=discord_channels["bus"], content="status update")
    private_message = fake_message_factory(channel=discord_channels["private"], content="please help")

    assert classify(bus_message, fake_client.user).kind == MessageKind.BUS
    assert classify(private_message, fake_client.user).kind == MessageKind.TASK


def test_classify_other_channels_require_mention_and_ignore_bots(
    discord_channels,
    fake_client,
    fake_message_factory,
) -> None:
    general = SimpleNamespace(id=404)
    mentioned = fake_message_factory(
        channel=general,
        content="<@999> investigate this",
        mentions=[fake_client.user],
    )
    ignored = fake_message_factory(channel=general, content="just chatting")
    other_bot = fake_message_factory(
        channel=general,
        content="hello",
        author=SimpleNamespace(display_name="other-bot", bot=True),
    )

    assert classify(mentioned, fake_client.user).content == "investigate this"
    assert classify(ignored, fake_client.user).kind == MessageKind.IGNORE
    assert classify(other_bot, fake_client.user).kind == MessageKind.IGNORE


def test_strip_mentions_and_normalize_a2a_task_helpers() -> None:
    assert _strip_mentions("<@999> hello <@!123> there") == "hello  there"
    assert _normalize_a2a_task(" Review ") == "review"
    assert _normalize_a2a_task(None) == ""


def test_a2a_to_task_content_includes_extra_context() -> None:
    rendered = a2a_to_task_content(
        {"from": "peer-1", "task": "review this change", "payload": "look at startup code"}
    )

    assert "[A2A from peer-1] review this change" in rendered
    assert "Additional context: look at startup code" in rendered


def test_a2a_to_task_content_falls_back_to_content_and_unknown_sender() -> None:
    rendered = a2a_to_task_content({"content": "do the thing"})

    assert rendered == "[A2A from unknown] do the thing"
