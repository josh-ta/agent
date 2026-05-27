"""Human-readable catalog of tools for the LLM intent router."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCatalogEntry:
    name: str
    description: str
    category: str


def build_tool_catalog(*, postgres_available: bool, sqlite_available: bool = True) -> list[ToolCatalogEntry]:
    entries: list[ToolCatalogEntry] = [
        ToolCatalogEntry("run_shell", "Run shell commands in the workspace", "shell"),
        ToolCatalogEntry("run_shell_read_only", "Run read-only shell commands (ls, head, wc, etc.)", "shell"),
        ToolCatalogEntry("read_file", "Read a file from the workspace", "filesystem"),
        ToolCatalogEntry("write_file", "Write or create a file in the workspace", "filesystem"),
        ToolCatalogEntry("list_dir", "List directory contents", "filesystem"),
        ToolCatalogEntry("search_files", "Search file contents under the workspace", "filesystem"),
        ToolCatalogEntry("web_search", "Search the web for current information", "research"),
        ToolCatalogEntry("task_note", "Record progress notes for the current task", "journal"),
        ToolCatalogEntry("send_discord", "Send a message to a Discord channel", "discord"),
        ToolCatalogEntry("read_discord", "Read recent Discord channel messages", "discord"),
        ToolCatalogEntry("memory_search", "Search local agent memory", "memory"),
        ToolCatalogEntry("lesson_search", "Search stored lessons and procedures", "memory"),
    ]
    if sqlite_available:
        entries.extend(
            [
                ToolCatalogEntry("db_stats", "SQLite database statistics", "database"),
            ]
        )
    if postgres_available:
        entries.extend(
            [
                ToolCatalogEntry(
                    "list_postgres_tables",
                    "List tables in the connected Postgres database (schema discovery)",
                    "database",
                ),
                ToolCatalogEntry(
                    "query_postgres",
                    "Run read-only SQL (SELECT/WITH) against Postgres; use output_format='csv' and output_path for file exports",
                    "database",
                ),
            ]
        )
    return entries


def format_tool_catalog(entries: list[ToolCatalogEntry]) -> str:
    lines = ["Available tools (use names exactly when suggesting tools):"]
    by_cat: dict[str, list[ToolCatalogEntry]] = {}
    for entry in entries:
        by_cat.setdefault(entry.category, []).append(entry)
    for category in sorted(by_cat):
        lines.append(f"\n[{category}]")
        for entry in by_cat[category]:
            lines.append(f"- {entry.name}: {entry.description}")
    return "\n".join(lines)
