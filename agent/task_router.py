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
    r"research|summarize|analyze|compare|review|audit|recommend|suggest|"
    r"ticket\s+limit|csv|spreadsheet|report|database|postgres|"
    r"/status"
    r")\b",
    re.IGNORECASE,
)

_COMPLEX_RE = re.compile(
    r"```|\bhttps?://|/[\w.-]+\.(?:py|js|ts|md|json|yml|yaml|sh|sql|csv)\b",
    re.IGNORECASE,
)

# Postgres / ticket-inventory data — answer via query_postgres, not from memory.
_DATABASE_WORK_RE = re.compile(
    r"\b("
    r"csv|postgres|database|sql|ticket\s+limit|export|spreadsheet|"
    r"information_schema|query_postgres|list_postgres|arena|stadium|events?"
    r")\b",
    re.IGNORECASE,
)

# Event/sale analytics phrasing that implies the events database even without "database".
_EVENT_DATA_RE = re.compile(
    r"\b("
    r"events?|sale?s?|ticket|tickets|venue|venues|arena|stadium|onsale|presale|"
    r"chartmetric|buying|focus\s+on|which\s+\d+|top\s+\d+|starting\s+today|"
    r"on\s+sale|ticketmaster"
    r")\b",
    re.IGNORECASE,
)

_DATABASE_CSV_RE = re.compile(
    r"\b(csv|spreadsheet|export|download|attach|resend)\b|\.csv\b",
    re.IGNORECASE,
)


def _has_attachments(metadata: dict[str, Any]) -> bool:
    attachments = metadata.get("attachments")
    if isinstance(attachments, list) and attachments:
        return True
    if metadata.get("attachment_count"):
        return True
    return bool(metadata.get("attachment_names"))


def _routing_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    raw = metadata.get("routing")
    return raw if isinstance(raw, dict) else {}


def _content_requires_tools(text: str) -> bool:
    """Heuristic tool-use detection from message text alone (ignores routing metadata)."""
    stripped = text.strip()
    if not stripped:
        return False
    if bool(_DATABASE_WORK_RE.search(stripped)) or bool(_EVENT_DATA_RE.search(stripped)):
        return True
    return bool(_WORK_RE.search(stripped))


def requires_database_query(content: str, *, metadata: dict[str, Any] | None = None) -> bool:
    """True when the request needs Postgres query tools (analytics or export)."""
    text = content.strip()
    if _content_requires_database(text):
        return True
    routing = _routing_metadata(metadata)
    intent = str(routing.get("intent", ""))
    if intent in {"database_analytics", "database_csv_export"}:
        return True
    if routing.get("needs_tools") and routing.get("export_csv"):
        return True
    return False


def _content_requires_database(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(_DATABASE_WORK_RE.search(stripped)) or bool(_EVENT_DATA_RE.search(stripped))


def requires_database_csv_export(content: str, *, metadata: dict[str, Any] | None = None) -> bool:
    """True when the user wants a CSV/file export (use non-streaming multi-step run)."""
    text = content.strip()
    if text and requires_database_query(text, metadata=None) and bool(_DATABASE_CSV_RE.search(text)):
        return True
    routing = _routing_metadata(metadata)
    if routing.get("export_csv"):
        return True
    if str(routing.get("intent", "")) == "database_csv_export":
        return True
    if not requires_database_query(text, metadata=metadata):
        return False
    return bool(_DATABASE_CSV_RE.search(text))


def requires_database_tools(content: str, *, metadata: dict[str, Any] | None = None) -> bool:
    """Alias for requires_database_query (Postgres tools required)."""
    return requires_database_query(content, metadata=metadata)


def requires_tool_use(content: str, *, metadata: dict[str, Any] | None = None) -> bool:
    """True when the user request clearly needs tools (SQL, files, shell, etc.)."""
    text = content.strip()
    if _content_requires_tools(text):
        return True
    routing = _routing_metadata(metadata)
    if routing.get("needs_tools"):
        return True
    return False


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
    meta = metadata or {}
    routing = _routing_metadata(meta)
    text = content.strip()

    if routing.get("execution_mode") == "agent":
        return "agent"

    if source != "discord":
        return "agent"

    if not text:
        return "agent"

    if _has_attachments(meta):
        return "agent"

    if meta.get("resume_context") or meta.get("checkpoint_context"):
        return "agent"

    if _COMPLEX_RE.search(text):
        return "agent"

    if _content_requires_tools(text):
        return "agent"

    if routing.get("execution_mode") == "chat":
        return "chat"

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
