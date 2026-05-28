from __future__ import annotations

from agent.communication.discord_constants import MAX_REPLY_LEN, split_message_chunks


def _event_block(name: str) -> str:
    return (
        f"{name} · Venue · Sale May 28, 2026 · Show Oct 15, 2026\n"
        "Signal: popularity 95, momentum 28.\n"
        "Take: Buy focus.\n"
        "Risk: More legs might announce."
    )


def test_split_empty_returns_empty_list() -> None:
    assert split_message_chunks("") == []


def test_split_short_message_is_single_chunk() -> None:
    text = "Summary — Top 10 events."
    assert split_message_chunks(text) == [text]


def test_split_keeps_paragraphs_intact() -> None:
    intro = "Summary — Top 10 events with sales starting today."
    events = "\n\n".join(_event_block(f"Artist {i}") for i in range(12))
    text = f"{intro}\n\n{events}"
    chunks = split_message_chunks(text, max_len=600)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= MAX_REPLY_LEN
        if "Signal:" in chunk:
            assert "Take:" in chunk
            assert "Risk:" in chunk
    assert intro in chunks[0]
    joined = "\n\n".join(chunks)
    assert joined == text


def test_split_does_not_cut_mid_line_when_possible() -> None:
    line_a = "A" * 100
    line_b = "B" * (MAX_REPLY_LEN - 50)
    text = f"{line_a}\n{line_b}"
    chunks = split_message_chunks(text, max_len=MAX_REPLY_LEN)

    assert chunks[0] == line_a
    assert chunks[1] == line_b


def test_split_hard_cut_only_when_unavoidable() -> None:
    text = "x" * (MAX_REPLY_LEN + 100)
    chunks = split_message_chunks(text, max_len=MAX_REPLY_LEN)

    assert len(chunks) == 2
    assert len(chunks[0]) == MAX_REPLY_LEN
    assert chunks[1] == "x" * 100
