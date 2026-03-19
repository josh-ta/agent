from __future__ import annotations

from pathlib import Path
from typing import Iterable

from agent.config import settings

PROJECT_MEMORY_FILENAME = ".agent-project-memory.md"
_HEADER = (
    "# Project Memory\n\n"
    "_This file is loaded on every turn. Keep durable repo-specific preferences, "
    "deploy facts, paths, and operating constraints here._\n"
)


def project_memory_path() -> Path:
    return settings.workspace_path / PROJECT_MEMORY_FILENAME


def load_project_memory(*, char_cap: int = 2500) -> str:
    path = project_memory_path()
    try:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

    if not text:
        return ""

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

    if added == 0:
        return 0

    _write_project_memory(path, ordered)
    return added


def remove_project_memory_facts(query: str) -> int:
    normalized_query = _normalize_fact(query)
    if not normalized_query:
        return 0

    path = project_memory_path()
    existing = _read_existing_facts(path)
    if not existing:
        return 0

    kept = [fact for fact in existing if normalized_query.casefold() not in fact.casefold()]
    removed = len(existing) - len(kept)
    if removed == 0:
        return 0
    _write_project_memory(path, kept)
    return removed


def render_project_memory() -> str:
    loaded = load_project_memory(char_cap=4000)
    return loaded or "## Project memory\n(no project memory recorded yet)"


def _normalize_fact(fact: str) -> str:
    text = " ".join(str(fact).strip().split())
    return text[:500]


def _read_existing_facts(path: Path) -> list[str]:
    try:
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
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
        if path.exists():
            path.unlink()
        return
    body = "\n".join(f"- {fact}" for fact in facts[-100:])
    path.write_text(f"{_HEADER}\n\n{body}\n", encoding="utf-8")
