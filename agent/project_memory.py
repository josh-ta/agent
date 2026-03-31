from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

from agent.config import settings

PROJECT_MEMORY_FILENAME = ".agent-project-memory.md"
# CC memdir-style caps for markdown entrypoints loaded into context
MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000
_HEADER = (
    "# Project Memory\n\n"
    "_This file is loaded on every turn. Keep durable repo-specific preferences, "
    "deploy facts, paths, and operating constraints here._\n"
)
_HOST_FACT_RE = re.compile(
    r"\b((?:app|data|deploy|db|database|redis|api|frontend)\s+host\s+(?:is|=)\s+([^\s,;]+))",
    re.IGNORECASE,
)
_PATH_FACT_RE = re.compile(
    r"\b((?:workspace|repo root|project root|deploy path)\s+(?:is|=)\s+([^\s,;]+))",
    re.IGNORECASE,
)
_SCRIPT_FACT_RE = re.compile(
    r"\b(use\s+(scripts/[^\s,;]+|[^\s,;]+\.sh)\b[^.]*)",
    re.IGNORECASE,
)


def project_memory_path() -> Path:
    return settings.workspace_path / PROJECT_MEMORY_FILENAME


def truncate_markdown_entrypoint(
    raw: str,
    *,
    max_lines: int = MAX_ENTRYPOINT_LINES,
    max_bytes: int = MAX_ENTRYPOINT_BYTES,
) -> str:
    """Line- then byte-cap long markdown (aligned with CC MEMORY.md policy)."""
    trimmed = raw.strip()
    if not trimmed:
        return ""
    lines = trimmed.split("\n")
    line_count = len(lines)
    byte_count = len(trimmed.encode("utf-8"))
    was_line = line_count > max_lines
    was_byte = byte_count > max_bytes
    if not was_line and not was_byte:
        return trimmed

    truncated = "\n".join(lines[:max_lines]) if was_line else trimmed
    tb = truncated.encode("utf-8")
    if len(tb) > max_bytes:
        cut = truncated.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
        last_nl = cut.rfind("\n")
        truncated = cut[:last_nl] if last_nl > 0 else cut
    if was_line and not was_byte:
        reason = f"{line_count} lines (limit {max_lines})"
    elif was_byte and not was_line:
        reason = f"{byte_count} bytes (limit {max_bytes})"
    else:
        reason = f"{line_count} lines and {byte_count} bytes"
    return (
        truncated
        + f"\n\n> WARNING: content exceeded limits ({reason}). "
        "Keep index entries short; move detail into topic files.\n"
    )


def load_project_memory(*, char_cap: int = 2500) -> str:
    path = project_memory_path()
    try:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8").strip()
    except Exception:  # pragma: no cover - defensive filesystem fallback
        return ""

    if not text:
        return ""

    text = truncate_markdown_entrypoint(text)
    if len(text) > char_cap:
        text = "[...truncated...]\n\n" + text[-char_cap:]
    return "## Project memory\n" + text


def save_project_memory_facts(facts: Iterable[str]) -> int:
    path = project_memory_path()
    normalized = [_normalize_fact(fact) for fact in facts]
    normalized = [fact for fact in normalized if fact]
    if not normalized:
        return 0

    existing = _read_existing_facts(path)
    ordered = list(existing)
    added = 0
    seen = {fact.casefold() for fact in existing}
    for fact in normalized:
        key = fact.casefold()
        if key in seen:
            continue
        ordered.append(fact)
        seen.add(key)
        added += 1

    if added == 0:  # pragma: no cover - simple no-op guard
        return 0

    _write_project_memory(path, ordered)
    return added


def remove_project_memory_facts(query: str) -> int:
    normalized_query = _normalize_fact(query)
    if not normalized_query:  # pragma: no cover - trivial guard
        return 0

    path = project_memory_path()
    existing = _read_existing_facts(path)
    if not existing:
        return 0

    kept = [fact for fact in existing if normalized_query.casefold() not in fact.casefold()]
    removed = len(existing) - len(kept)
    if removed == 0:  # pragma: no cover - simple no-op guard
        return 0
    _write_project_memory(path, kept)
    return removed


def render_project_memory() -> str:
    loaded = load_project_memory(char_cap=4000)
    return loaded or "## Project memory\n(no project memory recorded yet)"


def extract_project_memory_facts(text: str) -> set[str]:
    content = " ".join(str(text).strip().split())
    lowered = content.lower()
    if not content or len(content) > 1200:
        return set()

    facts: set[str] = set()
    fact_prefixes = (
        "remember ",
        "remember permanently ",
        "use ",
        "always ",
        "never ",
        "my ",
        "for this repo",
        "for this project",
        "in this repo",
        "in this project",
        "prefer ",
    )
    if any(lowered.startswith(prefix) for prefix in fact_prefixes):
        facts.add(content[:500])
    if any(
        phrase in lowered
        for phrase in (
            "app host",
            "data host",
            "deploy host",
            "workspace is",
            "repo root",
            "project root",
            "use scripts/",
            "deploy script",
            "do not claim deploy success",
            "never claim deploy success",
        )
    ):
        facts.add(content[:500])
    for pattern in (_HOST_FACT_RE, _PATH_FACT_RE, _SCRIPT_FACT_RE):
        for match in pattern.finditer(content):
            facts.add(match.group(1)[:500])
    if "/workspace" in content:
        facts.add("Use the existing checkout in /workspace before guessing repo names, paths, or hosts.")
    if any(phrase in lowered for phrase in ("hallucinating a repo", "guess the repo", "guess a repo", "guess the host", "guess the hostname")):
        facts.add("Do not guess repo names, repo paths, hostnames, or VPS IPs. Inspect local files first and ask if missing.")
    if "starting:" in lowered and "repeat" in lowered and "prompt" in lowered:
        facts.add("Do not echo the user's full prompt back as a start message.")
    if "check the file system first" in lowered or "read and check the file system first" in lowered:
        facts.add("Check the local file system before attempting deploy, SSH, or repo-specific actions.")
    if any(
        phrase in lowered for phrase in ("do not guess", "don't guess", "stop guessing")
    ):
        facts.add(content[:500])
    return facts


def _normalize_fact(fact: str) -> str:
    text = " ".join(str(fact).strip().split())
    return text[:500]


def _read_existing_facts(path: Path) -> list[str]:
    try:
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:  # pragma: no cover - defensive filesystem fallback
        return []

    facts: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            facts.append(stripped[2:].strip())
    return facts


def _write_project_memory(path: Path, facts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not facts:
        if path.exists():  # pragma: no cover - tiny filesystem cleanup branch
            path.unlink()
        return
    body = "\n".join(f"- {fact}" for fact in facts[-100:])
    path.write_text(f"{_HEADER}\n\n{body}\n", encoding="utf-8")
