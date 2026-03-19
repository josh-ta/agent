from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from pydantic_ai import Agent

from agent.config import settings
from agent.tools import filesystem, github, self_edit, shell
from agent.events import current_task_id
from agent.tools.discord_tools import ask_user, discord_read, discord_read_named, discord_send

if TYPE_CHECKING:
    from agent.memory.postgres_store import PostgresStore
    from agent.memory.sqlite_store import SQLiteStore

log = structlog.get_logger()


def attach_all_tools(
    agent: Agent,  # type: ignore[type-arg]
    *,
    sqlite: "SQLiteStore | None",
    postgres: "PostgresStore | None",
) -> None:
    attach_shell_tools(agent)
    attach_filesystem_tools(agent)
    attach_journal_tools(agent, sqlite=sqlite)
    attach_self_edit_tools(agent)
    attach_discord_tools(agent)
    attach_github_tools(agent)
    attach_database_tools(agent, sqlite=sqlite, postgres=postgres)
    log.info("tools_attached_to_agent")


def attach_shell_tools(agent: Agent) -> None:  # type: ignore[type-arg]
    @agent.tool_plain
    async def run_shell(command: str, working_dir: str = "", timeout: int = 30, tail_lines: int = 0) -> str:
        """Run a shell command. Returns combined stdout+stderr + exit code."""
        return await shell.shell_run(command, working_dir or None, timeout, tail_lines)


def attach_filesystem_tools(agent: Agent) -> None:  # type: ignore[type-arg]
    @agent.tool_plain
    def read_file(path: str) -> str:
        return filesystem.read_file(path)

    @agent.tool_plain
    def write_file(path: str, content: str) -> str:
        return filesystem.write_file(path, content)

    @agent.tool_plain
    def list_dir(path: str = ".") -> str:
        return filesystem.list_dir(path)

    @agent.tool_plain
    def delete_file(path: str) -> str:
        return filesystem.delete_file(path)

    @agent.tool_plain
    def str_replace(path: str, old_str: str, new_str: str, expected_replacements: int = 1) -> str:
        return filesystem.str_replace_file(path, old_str, new_str, expected_replacements)

    @agent.tool_plain
    def search_files(pattern: str, path: str = ".", file_glob: str = "", context_lines: int = 2) -> str:
        return filesystem.search_files(pattern, path, file_glob, context_lines)


def attach_journal_tools(agent: Agent, *, sqlite: "SQLiteStore | None") -> None:  # type: ignore[type-arg]
    journal_path = settings.workspace_path / ".task_journal.md"

    @agent.tool_plain
    def task_note(note: str) -> str:
        import asyncio as _asyncio
        from datetime import UTC as _UTC
        from datetime import datetime as _dt

        from agent.events import ProgressEvent as _ProgressEvent
        from agent.events import bridge as _bridge

        ts = _dt.now(_UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        entry = f"\n### [{ts}]\n{note.strip()}\n"
        try:
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            with journal_path.open("a", encoding="utf-8") as handle:
                handle.write(entry)
            task_id = current_task_id()
            if sqlite is not None and task_id:
                loop = _asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(sqlite.append_task_note(task_id, f"[{ts}] {note.strip()}"))
            try:
                loop = _asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(_bridge.emit(_ProgressEvent(message=f"📝 {note.strip()}")))
            except Exception:
                pass
            return f"Journal updated ({journal_path})."
        except Exception as exc:
            return f"[journal write error: {exc}]"

    @agent.tool_plain
    def task_resume() -> str:
        try:
            if not journal_path.exists():
                return "(no task journal found — this is a fresh start)"
            content = journal_path.read_text(encoding="utf-8").strip()
            if not content:
                return "(task journal is empty)"
            if len(content) > 8_000:
                content = "[...older entries truncated...]\n\n" + content[-8_000:]
            return f"## Task Journal\n\n{content}"
        except Exception as exc:
            return f"[journal read error: {exc}]"

    @agent.tool_plain
    def task_journal_clear() -> str:
        try:
            task_id = current_task_id()
            if sqlite is not None and task_id:
                import asyncio as _asyncio

                try:
                    loop = _asyncio.get_running_loop()
                    if loop.is_running():
                        loop.create_task(sqlite.clear_task_checkpoint(task_id))
                except Exception:
                    pass
            if journal_path.exists():
                journal_path.unlink()
            return "Task journal cleared."
        except Exception as exc:
            return f"[journal clear error: {exc}]"


def attach_self_edit_tools(agent: Agent) -> None:  # type: ignore[type-arg]
    @agent.tool_plain
    def skill_list() -> str:
        return self_edit.list_skills()

    @agent.tool_plain
    def skill_read(name: str) -> str:
        return self_edit.read_skill(name)

    @agent.tool_plain
    def skill_edit(name: str, content: str) -> str:
        return self_edit.edit_skill(name, content)

    @agent.tool_plain
    def identity_read(filename: str) -> str:
        return self_edit.read_identity(filename)

    @agent.tool_plain
    def identity_edit(filename: str, content: str) -> str:
        return self_edit.edit_identity(filename, content)

    @agent.tool_plain
    async def agent_restart(reason: str = "manual restart") -> str:
        return await self_edit.self_restart(reason)


def attach_discord_tools(agent: Agent) -> None:  # type: ignore[type-arg]
    @agent.tool_plain
    async def send_discord(channel_id: int, message: str) -> str:
        return await discord_send(channel_id, message)

    @agent.tool_plain
    async def read_discord(channel_id: int, limit: int = 20) -> str:
        return await discord_read(channel_id, limit)

    @agent.tool_plain
    async def read_channel(name: str, limit: int = 20) -> str:
        return await discord_read_named(name, limit)

    @agent.tool_plain
    async def ask_user_question(question: str, timeout: int = 300) -> str:
        return await ask_user(question, timeout)


def attach_github_tools(agent: Agent) -> None:  # type: ignore[type-arg]
    @agent.tool_plain
    async def gh_pr_view(pr: int, repo: str = "") -> str:
        return await github.pr_view(pr, repo or None)

    @agent.tool_plain
    async def gh_pr_list(repo: str = "", state: str = "open", limit: int = 20) -> str:
        return await github.pr_list(repo or None, state, limit)

    @agent.tool_plain
    async def gh_pr_diff(pr: int, repo: str = "") -> str:
        return await github.pr_diff(pr, repo or None)

    @agent.tool_plain
    async def gh_pr_comment(pr: int, body: str, repo: str = "") -> str:
        return await github.pr_comment(pr, body, repo or None)

    @agent.tool_plain
    async def gh_pr_review(pr: int, action: str, body: str = "", repo: str = "") -> str:
        return await github.pr_review(pr, action, body, repo or None)

    @agent.tool_plain
    async def gh_pr_review_inline(pr: int, action: str, body: str, comments: list[dict], repo: str = "") -> str:
        return await github.pr_review_with_comments(pr, action, body, comments, repo or None)

    @agent.tool_plain
    async def gh_pr_checks(pr: int, repo: str = "") -> str:
        return await github.pr_checks(pr, repo or None)

    @agent.tool_plain
    async def gh_pr_merge(pr: int, method: str = "squash", repo: str = "") -> str:
        return await github.pr_merge(pr, method, repo or None)

    @agent.tool_plain
    async def gh_issue_view(issue: int, repo: str = "") -> str:
        return await github.issue_view(issue, repo or None)

    @agent.tool_plain
    async def gh_issue_list(repo: str = "", state: str = "open", limit: int = 20) -> str:
        return await github.issue_list(repo or None, state, limit)

    @agent.tool_plain
    async def gh_issue_comment(issue: int, body: str, repo: str = "") -> str:
        return await github.issue_comment(issue, body, repo or None)

    @agent.tool_plain
    async def gh_issue_create(title: str, body: str, labels: list[str] | None = None, repo: str = "") -> str:
        return await github.issue_create(title, body, labels, repo or None)

    @agent.tool_plain
    async def gh_issue_close(issue: int, reason: str = "completed", repo: str = "") -> str:
        return await github.issue_close(issue, reason, repo or None)

    @agent.tool_plain
    async def gh_ci_list(repo: str = "", branch: str = "", limit: int = 5) -> str:
        return await github.ci_list(repo or None, branch or None, limit)

    @agent.tool_plain
    async def gh_ci_view(run_id: str, repo: str = "") -> str:
        return await github.ci_view(run_id, repo or None)

    @agent.tool_plain
    async def gh_ci_logs_failed(run_id: str, repo: str = "") -> str:
        return await github.ci_logs_failed(run_id, repo or None)

    @agent.tool_plain
    async def gh_ci_rerun(run_id: str, failed_only: bool = True, repo: str = "") -> str:
        return await github.ci_rerun(run_id, failed_only, repo or None)


def attach_database_tools(
    agent: Agent,  # type: ignore[type-arg]
    *,
    sqlite: "SQLiteStore | None",
    postgres: "PostgresStore | None",
) -> None:
    if sqlite or postgres:
        @agent.tool_plain
        async def db_stats() -> str:
            lines = ["## Database Stats", ""]
            if sqlite:
                try:
                    stats = await sqlite.get_stats()
                    lines.append("**SQLite (local):**")
                    lines.append(f"  conversations: {stats.get('conversations', '?')} rows")
                    lines.append(f"  tasks: {stats.get('tasks', '?')} rows")
                    lines.append(f"  memory_facts: {stats.get('memory_facts', '?')} rows")
                    lines.append(f"  lessons: {stats.get('lessons', '?')} rows")
                    lines.append(f"  db size: {stats.get('db_size_mb', '?')} MB")
                    lines.append(f"  vec search: {stats.get('vec_enabled', False)}")
                    lines.append(f"  last cleanup: {stats.get('last_cleanup', 'never')}")
                except Exception as exc:
                    lines.append(f"  SQLite error: {exc}")
            if postgres:
                try:
                    stats = await postgres.get_stats()
                    lines.append("\n**Postgres (shared):**")
                    lines.append(f"  agents: {stats.get('agents', '?')} rows")
                    lines.append(f"  shared_tasks: {stats.get('shared_tasks', '?')} rows")
                    lines.append(f"  audit_log: {stats.get('audit_log', '?')} rows")
                    lines.append(f"  shared_memory: {stats.get('shared_memory', '?')} rows")
                    lines.append(f"  last cleanup: {stats.get('last_cleanup', 'never')}")
                except Exception as exc:
                    lines.append(f"  Postgres error: {exc}")
            return "\n".join(lines)

    if sqlite:
        @agent.tool_plain
        async def memory_search(query: str, limit: int = 5) -> str:
            return await sqlite.search_memory(query, limit)

        @agent.tool_plain
        async def memory_save(content: str) -> str:
            await sqlite.save_memory_fact(content)
            return f"Saved to memory: {content[:80]}"

        @agent.tool_plain
        async def lesson_save(summary: str, kind: str = "lesson") -> str:
            await sqlite.save_lesson(summary, kind=kind)
            return f"Lesson recorded [{kind}]: {summary[:80]}"

        @agent.tool_plain
        async def lesson_search(query: str, limit: int = 5) -> str:
            result = await sqlite.search_lessons(query, limit)
            return result or f"(no lessons found for: {query})"

        @agent.tool_plain
        async def lessons_recent(limit: int = 10) -> str:
            return await sqlite.get_recent_lessons(limit)

    if postgres:
        @agent.tool_plain
        async def list_agents() -> str:
            return await postgres.list_agents()

        @agent.tool_plain
        async def create_shared_task(to_agent: str, description: str) -> str:
            return await postgres.create_task(to_agent, description)

        @agent.tool_plain
        async def my_tasks() -> str:
            return await postgres.get_my_tasks()

        @agent.tool_plain
        async def complete_task(task_id: str, result: str) -> str:
            return await postgres.complete_task(task_id, result)

        @agent.tool_plain
        async def broadcast_message(message: str) -> str:
            return await postgres.broadcast_message(message)

        @agent.tool_plain
        async def read_broadcasts(limit: int = 20) -> str:
            return await postgres.read_broadcasts(limit)

        @agent.tool_plain
        async def share_memory(content: str) -> str:
            return await postgres.share_memory(content)

        @agent.tool_plain
        async def search_shared_memory(query: str, limit: int = 5) -> str:
            return await postgres.search_shared_memory(query, limit)
