"""
Restricted toolsets for nested sub-agents (isolated runs, no run_agent_subtask).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from pydantic_ai import Agent

from agent.config import settings
from agent.metrics import Metrics
from agent.permissions import get_permission_engine
from agent.tools import filesystem, shell

if TYPE_CHECKING:
    from agent.memory.postgres_store import PostgresStore
    from agent.memory.sqlite_store import SQLiteStore

log = structlog.get_logger()


def _tool_perm_block(tool_name: str, **kwargs: Any) -> str | None:
    eng = get_permission_engine()
    if eng is None:
        return None
    result = eng.check_sync(tool_name, kwargs)
    if result.ok:
        return None
    Metrics.inc_permission_denied()
    return result.message


def attach_subagent_tools(
    agent: Agent,  # type: ignore[type-arg]
    *,
    sqlite: SQLiteStore | None,
    postgres: PostgresStore | None,
    profile: str,
) -> None:
    """Attach a read-focused tool subset. Profiles: minimal | explore | verify."""

    @agent.tool_plain
    def read_file(
        path: str,
        encoding: str = "utf-8",
        start_line: int = 1,
        end_line: int | None = None,
        max_lines: int = 2000,
    ) -> str:
        return filesystem.read_file(
            path,
            encoding,
            start_line=start_line,
            end_line=end_line,
            max_lines=max_lines,
        )

    @agent.tool_plain
    def list_dir(path: str = ".") -> str:
        return filesystem.list_dir(path)

    @agent.tool_plain
    def search_files(
        pattern: str,
        path: str = ".",
        file_glob: str = "",
        context_lines: int = 2,
        fixed_string: bool = False,
        case_sensitive: bool = True,
        max_matches_per_file: int | None = None,
        max_total_matches: int | None = None,
        output_mode: str = "text",
    ) -> str:
        return filesystem.search_files(
            pattern,
            path,
            file_glob,
            context_lines,
            fixed_string=fixed_string,
            case_sensitive=case_sensitive,
            max_matches_per_file=max_matches_per_file,
            max_total_matches=max_total_matches,
            output_mode=output_mode,
        )

    if profile in {"explore", "verify"}:
        journal_path = settings.workspace_path / ".task_journal.md"

        @agent.tool_plain
        def task_resume() -> str:
            try:
                if not journal_path.exists():
                    return "(no task journal)"
                content = journal_path.read_text(encoding="utf-8").strip()
                if len(content) > 8_000:
                    content = "[...truncated...]\n\n" + content[-8_000:]
                return f"## Task Journal\n\n{content}"
            except Exception as exc:
                return f"[journal read error: {exc}]"

    if sqlite:
        @agent.tool_plain
        async def memory_search(query: str, limit: int = 5) -> str:
            return await sqlite.search_memory(query, limit)

        @agent.tool_plain
        async def lesson_search(query: str, limit: int = 5) -> str:
            result = await sqlite.search_lessons(query, limit)
            return result or f"(no lessons for: {query})"

        @agent.tool_plain
        async def procedure_search(query: str, limit: int = 5) -> str:
            procedures = await sqlite.search_procedures(query, limit=limit)
            if not procedures:
                return f"(no procedures for: {query})"
            return "## Procedures\n" + "\n".join(
                f"- #{row['id']} {row['trigger_text']} => {row['checklist']}" for row in procedures
            )

    if postgres:

        @agent.tool_plain
        async def list_agents() -> str:
            return await postgres.list_agents()

    if profile == "verify":

        @agent.tool_plain
        async def run_shell_read_only(
            command: str, working_dir: str = "", timeout: int = 30, tail_lines: int = 0
        ) -> str:
            if msg := _tool_perm_block(
                "run_shell_read_only",
                command=command,
                working_dir=working_dir,
                timeout=timeout,
                tail_lines=tail_lines,
            ):
                return msg
            return await shell.shell_run(
                command, working_dir or None, timeout, tail_lines, read_only=True
            )

    log.info("subagent_tools_attached", profile=profile)
