"""Per-message routing: quick conversational turns vs tool-using work turns.

One Discord conversation mixes both — greetings and thanks stay lightweight;
requests that need SQL, files, shell, or research get the full agent stack.
"""

from __future__ import annotations

import re
from typing import Any, Literal

ExecutionMode = Literal["chat", "agent"]

# Greetings, acknowledgments, sign-offs — reply in a few words, no tools.
_SOCIAL_RE = re.compile(
    r"^\s*(?:"
    r"(?:hi|hello|hey|yo|sup|hiya|howdy)(?:\s+(?:chat|bob|there))?|"
    r"good\s+(?:morning|afternoon|evening|night)|"
    r"thanks(?:\s+you)?|thank\s+you|thx|ty|cheers|"
    r"(?:no\s+)?problem|you(?:'re|\s+are)\s+welcome|"
    r"ok(?:ay)?|k|cool|nice|great|awesome|perfect|"
    r"what'?s\s+up|how\s+(?:are|r)\s+you(?:\s+doing)?|"
    r"yes|no|yep|nope|sure|got\s+it|understood|"
    r"lol|lmao|haha|"
    r"bye|goodbye|see\s+ya|cya"
    r")[\s!.?]*$",
    re.IGNORECASE,
)

# User wants data pulled, exported, queried, built, checked, etc.
_WORK_RE = re.compile(
    r"\b("
    r"pull|export|download|upload|generate|create|write|build|implement|fix|debug|"
    r"run|execute|ssh|deploy|install|update|change|edit|modify|refactor|"
    r"clone|push|commit|merge|restart|configure|setup|browse|open|scrape|"
    r"query|sql|select|insert|grep|curl|docker|"
    r"list\s+of|get\s+me|give\s+me|show\s+me|send\s+me|fetch|retrieve|"
    r"help\s+me|can\s+you|could\s+you|please|need\s+you\s+to|i\s+need|"
    r"research|summarize|analyze|compare|review|audit|"
    r"ticket\s+limit|csv|spreadsheet|report|database|postgres|"
    r"/status"
    r")\b",
    re.IGNORECASE,
)

_COMPLEX_RE = re.compile(
    r"```|\bhttps?://|/[\w.-]+\.(?:py|js|ts|md|json|yml|yaml|sh|sql|csv)\b",
    re.IGNORECASE,
)


def _has_attachments(metadata: dict[str, Any]) -> bool:
    attachments = metadata.get("attachments")
    if isinstance(attachments, list) and attachments:
        return True
    if metadata.get("attachment_count"):
        return True
    return bool(metadata.get("attachment_names"))


def classify_execution_mode(
    content: str,
    *,
    source: str = "discord",
    metadata: dict[str, Any] | None = None,
) -> ExecutionMode:
    """
    Classify this message as a conversational turn (chat) or work turn (agent).

    Same user, same session — only the per-message depth changes.
    """
    if source != "discord":
        return "agent"

    meta = metadata or {}
    text = content.strip()
    if not text:
        return "agent"

    if _has_attachments(meta):
        return "agent"

    if meta.get("resume_context") or meta.get("checkpoint_context"):
        return "agent"

    if _COMPLEX_RE.search(text):
        return "agent"

    if _WORK_RE.search(text):
        return "agent"

    from agent.loop import _BEST_KEYWORDS, _SMART_KEYWORDS

    if _SMART_KEYWORDS.search(text) or _BEST_KEYWORDS.search(text):
        return "agent"

    words = len(text.split())
    if words > 12 or len(text) > 180:
        return "agent"

    if "?" in text and words > 4:
        return "agent"

    if _SOCIAL_RE.match(text):
        return "chat"

    if words <= 6 and len(text) <= 80 and "?" not in text:
        return "chat"

    return "agent"
