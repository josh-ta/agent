"""
Tool registry: registers all agent tools with Pydantic AI.

Pydantic AI tools are plain async/sync functions decorated with @agent.tool.
We use a registry pattern so tools can be conditionally included and
the agent is assembled after all dependencies are available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from pydantic_ai import Agent, RunContext

from agent.config import settings
from agent.tools import filesystem, github, self_edit, shell
from agent.tools.discord_tools import ask_user, discord_read, discord_read_named, discord_send

if TYPE_CHECKING:
    from agent.memory.sqlite_store import SQLiteStore
    from agent.memory.postgres_store import PostgresStore

log = structlog.get_logger()


class ToolRegistry:
    """Collects tool functions and attaches them to a Pydantic AI agent."""

    def __init__(self) -> None:
        self._sqlite: SQLiteStore | None = None
        self._postgres: PostgresStore | None = None

    def register_all(
        self,
        sqlite: "SQLiteStore | None" = None,
        postgres: "PostgresStore | None" = None,
    ) -> None:
        self._sqlite = sqlite
        self._postgres = postgres
        log.info("tools_registered", sqlite=sqlite is not None, postgres=postgres is not None)

    def attach_to_agent(self, agent: Agent) -> None:  # type: ignore[type-arg]
        """Attach all tool functions to the given Pydantic AI agent."""
        sqlite = self._sqlite
        postgres = self._postgres

        # ── Shell ─────────────────────────────────────────────────────────────
        @agent.tool_plain
        async def run_shell(command: str, working_dir: str = "", timeout: int = 30, tail_lines: int = 0) -> str:
            """
            Run a shell command. Returns combined stdout+stderr + exit code.
            tail_lines: when > 0, return only the last N lines (useful for test runners
            where failures appear at the end and output exceeds the 10KB cap).
            """
            return await shell.shell_run(command, working_dir or None, timeout, tail_lines)

        # ── Filesystem ────────────────────────────────────────────────────────
        @agent.tool_plain
        def read_file(path: str) -> str:
            """Read a file. Path is relative to workspace or absolute."""
            return filesystem.read_file(path)

        @agent.tool_plain
        def write_file(path: str, content: str) -> str:
            """Write content to a file (creates parents as needed)."""
            return filesystem.write_file(path, content)

        @agent.tool_plain
        def list_dir(path: str = ".") -> str:
            """List directory contents with sizes. Defaults to workspace root."""
            return filesystem.list_dir(path)

        @agent.tool_plain
        def delete_file(path: str) -> str:
            """Delete a file (not directories)."""
            return filesystem.delete_file(path)

        @agent.tool_plain
        def str_replace(path: str, old_str: str, new_str: str, expected_replacements: int = 1) -> str:
            """
            Replace an exact string in a file (surgical edit — preferred over write_file).

            Fails loudly if old_str is not found exactly expected_replacements times,
            so mismatches are caught before any write happens.

            old_str: include enough surrounding context to make it unique in the file.
            expected_replacements: how many occurrences to replace (default 1).
            """
            return filesystem.str_replace_file(path, old_str, new_str, expected_replacements)

        @agent.tool_plain
        def search_files(pattern: str, path: str = ".", file_glob: str = "", context_lines: int = 2) -> str:
            """
            Search files with ripgrep (rg). Returns matching lines with context.

            pattern: regular expression.
            path: directory or file to search (default: workspace root).
            file_glob: filename filter, e.g. '*.py' or '*.{ts,tsx}'.
            context_lines: lines of context around each match (default 2).
            """
            return filesystem.search_files(pattern, path, file_glob, context_lines)

        # ── Task journal (checkpointing for long tasks) ───────────────────────
        _journal_path = settings.workspace_path / ".task_journal.md"

        @agent.tool_plain
        def task_note(note: str) -> str:
            """
            Write a progress note to the persistent task journal.

            Call this frequently during long tasks — after each major step,
            discovery, or decision. If the task is interrupted (rate limit,
            restart, error), you can call task_resume() to read back all notes
            and pick up exactly where you left off without losing any work.

            note: what you just did, what you found, and what the next step is.
            """
            import asyncio as _asyncio
            from agent.events import bridge as _bridge, ProgressEvent as _ProgressEvent
            from datetime import datetime as _dt
            ts = _dt.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            entry = f"\n### [{ts}]\n{note.strip()}\n"
            try:
                _journal_path.parent.mkdir(parents=True, exist_ok=True)
                with _journal_path.open("a", encoding="utf-8") as f:
                    f.write(entry)
                # Surface the checkpoint in Discord
                try:
                    loop = _asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(_bridge.emit(_ProgressEvent(message=f"📝 {note.strip()}")))
                except Exception:
                    pass
                return f"Journal updated ({_journal_path})."
            except Exception as exc:
                return f"[journal write error: {exc}]"

        @agent.tool_plain
        def task_resume() -> str:
            """
            Read the task journal to resume an interrupted long task.

            Returns all progress notes written by task_note(). Use this at the
            start of a task if you suspect it was previously attempted, or
            whenever you need to recall what has already been done.
            """
            try:
                if not _journal_path.exists():
                    return "(no task journal found — this is a fresh start)"
                content = _journal_path.read_text(encoding="utf-8").strip()
                if not content:
                    return "(task journal is empty)"
                # Cap to avoid flooding context
                if len(content) > 8_000:
                    content = "[...older entries truncated...]\n\n" + content[-8_000:]
                return f"## Task Journal\n\n{content}"
            except Exception as exc:
                return f"[journal read error: {exc}]"

        @agent.tool_plain
        def task_journal_clear() -> str:
            """
            Clear the task journal once a task is fully complete.
            Call this after successfully finishing a long task so the next
            task starts with a clean journal.
            """
            try:
                if _journal_path.exists():
                    _journal_path.unlink()
                return "Task journal cleared."
            except Exception as exc:
                return f"[journal clear error: {exc}]"

        # ── Self-edit ─────────────────────────────────────────────────────────
        @agent.tool_plain
        def skill_list() -> str:
            """List all available skills."""
            return self_edit.list_skills()

        @agent.tool_plain
        def skill_read(name: str) -> str:
            """Read a skill file by name (slug, e.g. 'web-research')."""
            return self_edit.read_skill(name)

        @agent.tool_plain
        def skill_edit(name: str, content: str) -> str:
            """Create or update a skill file. Name must be a lowercase slug."""
            return self_edit.edit_skill(name, content)

        @agent.tool_plain
        def identity_read(filename: str) -> str:
            """Read IDENTITY.md, GOALS.md, or MEMORY.md."""
            return self_edit.read_identity(filename)

        @agent.tool_plain
        def identity_edit(filename: str, content: str) -> str:
            """Update IDENTITY.md, GOALS.md, or MEMORY.md with new content."""
            return self_edit.edit_identity(filename, content)

        @agent.tool_plain
        async def agent_restart(reason: str = "manual restart") -> str:
            """Restart this agent container (use after code changes)."""
            return await self_edit.self_restart(reason)

        # ── Discord ───────────────────────────────────────────────────────────
        @agent.tool_plain
        async def send_discord(channel_id: int, message: str) -> str:
            """Send a message to a Discord channel by its numeric ID."""
            return await discord_send(channel_id, message)

        @agent.tool_plain
        async def read_discord(channel_id: int, limit: int = 20) -> str:
            """Read recent messages from a Discord channel by numeric ID."""
            return await discord_read(channel_id, limit)

        @agent.tool_plain
        async def read_channel(name: str, limit: int = 20) -> str:
            """
            Read recent messages from a named channel.
            name: 'private' (your channel), 'bus' (#agent-bus), 'comms' (#agent-comms).
            Use this to catch up on conversation history or check what other agents said.
            """
            return await discord_read_named(name, limit)

        @agent.tool_plain
        async def ask_user_question(question: str, timeout: int = 300) -> str:
            """
            Ask the user a question and wait for their reply (up to timeout seconds).

            Use this when you are genuinely uncertain about something that could
            change what you do: ambiguous requirements, risky/irreversible actions,
            missing credentials, or a fork in the approach. Do NOT ask for things
            you can figure out yourself.

            The question is posted to your private channel with a ❓ prefix.
            This tool pauses your work until the user replies or the timeout expires.

            question: A clear, specific question. Prefer yes/no or short-answer questions.
            timeout:  Seconds to wait (default 300 = 5 min). Use longer for low-urgency asks.
            """
            return await ask_user(question, timeout)

        # ── GitHub ────────────────────────────────────────────────────────────

        @agent.tool_plain
        async def gh_pr_view(pr: int, repo: str = "") -> str:
            """Read a PR: title, body, status, CI checks, and all comments."""
            return await github.pr_view(pr, repo or None)

        @agent.tool_plain
        async def gh_pr_list(repo: str = "", state: str = "open", limit: int = 20) -> str:
            """List PRs. state: open | closed | merged | all."""
            return await github.pr_list(repo or None, state, limit)

        @agent.tool_plain
        async def gh_pr_diff(pr: int, repo: str = "") -> str:
            """Get the diff for a PR (capped at 300 lines)."""
            return await github.pr_diff(pr, repo or None)

        @agent.tool_plain
        async def gh_pr_comment(pr: int, body: str, repo: str = "") -> str:
            """Post a general comment on a PR."""
            return await github.pr_comment(pr, body, repo or None)

        @agent.tool_plain
        async def gh_pr_review(pr: int, action: str, body: str = "", repo: str = "") -> str:
            """
            Submit a PR review.
            action: 'approve' | 'request-changes' | 'comment'
            body: review summary (required for request-changes).
            """
            return await github.pr_review(pr, action, body, repo or None)

        @agent.tool_plain
        async def gh_pr_review_inline(
            pr: int,
            action: str,
            body: str,
            comments: list[dict],
            repo: str = "",
        ) -> str:
            """
            Submit a PR review with inline line-level comments.
            action: 'APPROVE' | 'REQUEST_CHANGES' | 'COMMENT'
            body: top-level review summary.
            comments: list of {path, line, message} dicts.
            """
            return await github.pr_review_with_comments(pr, action, body, comments, repo or None)

        @agent.tool_plain
        async def gh_pr_checks(pr: int, repo: str = "") -> str:
            """Show CI check status for a PR."""
            return await github.pr_checks(pr, repo or None)

        @agent.tool_plain
        async def gh_pr_merge(pr: int, method: str = "squash", repo: str = "") -> str:
            """Merge a PR. method: 'merge' | 'squash' | 'rebase'."""
            return await github.pr_merge(pr, method, repo or None)

        @agent.tool_plain
        async def gh_issue_view(issue: int, repo: str = "") -> str:
            """Read an issue and all its comments."""
            return await github.issue_view(issue, repo or None)

        @agent.tool_plain
        async def gh_issue_list(repo: str = "", state: str = "open", limit: int = 20) -> str:
            """List issues. state: open | closed | all."""
            return await github.issue_list(repo or None, state, limit)

        @agent.tool_plain
        async def gh_issue_comment(issue: int, body: str, repo: str = "") -> str:
            """Post a comment on an issue."""
            return await github.issue_comment(issue, body, repo or None)

        @agent.tool_plain
        async def gh_issue_create(title: str, body: str, labels: list[str] | None = None, repo: str = "") -> str:
            """Create a new GitHub issue."""
            return await github.issue_create(title, body, labels, repo or None)

        @agent.tool_plain
        async def gh_issue_close(issue: int, reason: str = "completed", repo: str = "") -> str:
            """Close an issue. reason: 'completed' | 'not planned'."""
            return await github.issue_close(issue, reason, repo or None)

        @agent.tool_plain
        async def gh_ci_list(repo: str = "", branch: str = "", limit: int = 5) -> str:
            """List recent CI workflow runs."""
            return await github.ci_list(repo or None, branch or None, limit)

        @agent.tool_plain
        async def gh_ci_view(run_id: str, repo: str = "") -> str:
            """Show job status for a CI run."""
            return await github.ci_view(run_id, repo or None)

        @agent.tool_plain
        async def gh_ci_logs_failed(run_id: str, repo: str = "") -> str:
            """Fetch failed step logs from a CI run (capped at 200 lines / ~4KB)."""
            return await github.ci_logs_failed(run_id, repo or None)

        @agent.tool_plain
        async def gh_ci_rerun(run_id: str, failed_only: bool = True, repo: str = "") -> str:
            """Re-trigger a CI run (or only its failed jobs)."""
            return await github.ci_rerun(run_id, failed_only, repo or None)

        # ── Database stats + diagnostics ─────────────────────────────────────
        if sqlite or postgres:
            @agent.tool_plain
            async def db_stats() -> str:
                """
                Show database row counts, file sizes, and last cleanup times.
                Use this to check database health, see how many memories/lessons are stored,
                or verify that cleanup is running.
                """
                lines = ["## Database Stats", ""]
                if sqlite:
                    try:
                        s = await sqlite.get_stats()
                        lines.append("**SQLite (local):**")
                        lines.append(f"  conversations: {s.get('conversations', '?')} rows")
                        lines.append(f"  tasks: {s.get('tasks', '?')} rows")
                        lines.append(f"  memory_facts: {s.get('memory_facts', '?')} rows")
                        lines.append(f"  lessons: {s.get('lessons', '?')} rows")
                        lines.append(f"  db size: {s.get('db_size_mb', '?')} MB")
                        lines.append(f"  vec search: {s.get('vec_enabled', False)}")
                        lines.append(f"  last cleanup: {s.get('last_cleanup', 'never')}")
                    except Exception as exc:
                        lines.append(f"  SQLite error: {exc}")
                if postgres:
                    try:
                        p = await postgres.get_stats()
                        lines.append("\n**Postgres (shared):**")
                        lines.append(f"  agents: {p.get('agents', '?')} rows")
                        lines.append(f"  shared_tasks: {p.get('shared_tasks', '?')} rows")
                        lines.append(f"  audit_log: {p.get('audit_log', '?')} rows")
                        lines.append(f"  shared_memory: {p.get('shared_memory', '?')} rows")
                        lines.append(f"  last cleanup: {p.get('last_cleanup', 'never')}")
                    except Exception as exc:
                        lines.append(f"  Postgres error: {exc}")
                return "\n".join(lines)

        # ── Memory search ─────────────────────────────────────────────────────
        if sqlite:
            @agent.tool_plain
            async def memory_search(query: str, limit: int = 5) -> str:
                """Search past conversations and memory facts by keyword."""
                return await sqlite.search_memory(query, limit)

            @agent.tool_plain
            async def memory_save(content: str) -> str:
                """Save a fact or insight to long-term memory."""
                await sqlite.save_memory_fact(content)
                return f"Saved to memory: {content[:80]}"

            @agent.tool_plain
            async def lesson_save(summary: str, kind: str = "lesson") -> str:
                """
                Record a lesson, mistake, or insight so it is never forgotten.
                kind: 'lesson' | 'mistake' | 'insight' | 'pattern'
                Use 'mistake' when recording something that went wrong.
                """
                await sqlite.save_lesson(summary, kind=kind)
                return f"Lesson recorded [{kind}]: {summary[:80]}"

            @agent.tool_plain
            async def lesson_search(query: str, limit: int = 5) -> str:
                """Search past lessons and mistakes relevant to a topic."""
                result = await sqlite.search_lessons(query, limit)
                return result or f"(no lessons found for: {query})"

            @agent.tool_plain
            async def lessons_recent(limit: int = 10) -> str:
                """Show the most recent lessons and mistakes."""
                return await sqlite.get_recent_lessons(limit)

        # ── Collaboration API (Postgres — multi-agent) ───────────────────────
        if postgres:
            # -- Agent registry --

            @agent.tool_plain
            async def list_agents() -> str:
                """List all agents registered in the shared database with their status."""
                return await postgres.list_agents()

            # -- Task queue --

            @agent.tool_plain
            async def create_shared_task(to_agent: str, description: str) -> str:
                """Assign a task to another agent in the shared task queue."""
                return await postgres.create_task(to_agent, description)

            @agent.tool_plain
            async def my_tasks() -> str:
                """List pending tasks assigned to this agent from the shared queue."""
                return await postgres.get_my_tasks()

            @agent.tool_plain
            async def complete_task(task_id: str, result: str) -> str:
                """
                Mark a shared task as done.
                task_id: the full task UUID (from my_tasks output).
                result: summary of what was accomplished.
                """
                return await postgres.complete_task(task_id, result)

            # -- Broadcasts --

            @agent.tool_plain
            async def broadcast_message(message: str) -> str:
                """
                Broadcast a message to all agents via the shared audit log.
                Use this to announce discoveries, status updates, or requests
                that any agent should be able to see.
                """
                return await postgres.broadcast_message(message)

            @agent.tool_plain
            async def read_broadcasts(limit: int = 20) -> str:
                """Read recent broadcast messages from all agents."""
                return await postgres.read_broadcasts(limit)

            # -- Shared memory --

            @agent.tool_plain
            async def share_memory(content: str) -> str:
                """
                Write a fact or insight to shared memory — visible to all agents.
                Use this to share knowledge that other agents should be able to find.
                """
                return await postgres.share_memory(content)

            @agent.tool_plain
            async def search_shared_memory(query: str, limit: int = 5) -> str:
                """Search shared memory from all agents by keyword."""
                return await postgres.search_shared_memory(query, limit)

        log.info("tools_attached_to_agent")
