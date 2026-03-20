from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from pydantic_ai import Agent

from agent.config import settings
from agent.secret_store import SecretNotFoundError, SecretStore, SecretStoreError, mask_secret
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
    attach_secret_tools(agent)
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


def attach_secret_tools(agent: Agent) -> None:  # type: ignore[type-arg]
    def _store() -> SecretStore:
        return SecretStore(settings.agent_secrets_path)

    @agent.tool_plain
    def secret_list() -> str:
        store = _store()
        if hasattr(store, "list_entries"):
            entries = store.list_entries()
        else:
            entries = [{"name": name, "purpose": "", "scope": "", "allowed_tools": []} for name in store.list_names()]
        if not entries:
            return "(no secrets stored)"
        lines = ["## Stored secrets"]
        for entry in entries:
            meta = []
            if entry["purpose"]:
                meta.append(f"purpose={entry['purpose']}")
            if entry["scope"]:
                meta.append(f"scope={entry['scope']}")
            if entry["allowed_tools"]:
                meta.append(f"tools={','.join(entry['allowed_tools'])}")
            suffix = f" ({'; '.join(meta)})" if meta else ""
            lines.append(f"- {entry['name']}{suffix}")
        return "\n".join(lines)

    @agent.tool_plain
    def secret_set(
        name: str,
        value: str,
        purpose: str = "",
        scope: str = "",
        allowed_tools: list[str] | None = None,
        rotation_hint: str = "",
    ) -> str:
        try:
            store = _store()
            try:
                store.set(
                    name,
                    value,
                    purpose=purpose,
                    scope=scope,
                    allowed_tools=allowed_tools,
                    rotation_hint=rotation_hint,
                )
            except TypeError:
                store.set(name, value)
            return f"Stored secret `{name}` = {mask_secret(value)}"
        except (SecretStoreError, ValueError) as exc:
            return f"[ERROR: {exc}]"

    @agent.tool_plain
    def secret_search(query: str, limit: int = 5) -> str:
        try:
            store = _store()
            if hasattr(store, "search"):
                matches = store.search(query, limit=limit)
            else:
                names = [name for name in store.list_names() if query.lower() in name.lower()]
                matches = [{"name": name, "purpose": "", "scope": "", "allowed_tools": []} for name in names[:limit]]
            if not matches:
                return f"(no stored secrets match: {query})"
            lines = ["## Matching secrets"]
            for entry in matches:
                meta = []
                if entry["purpose"]:
                    meta.append(entry["purpose"])
                if entry["scope"]:
                    meta.append(f"scope={entry['scope']}")
                lines.append(f"- {entry['name']}" + (f" ({'; '.join(meta)})" if meta else ""))
            return "\n".join(lines)
        except (SecretStoreError, ValueError) as exc:
            return f"[ERROR: {exc}]"

    @agent.tool_plain
    def secret_get(name: str) -> str:
        try:
            return _store().get(name)
        except SecretNotFoundError:
            return f"[ERROR: secret not found: {name}]"
        except (SecretStoreError, ValueError) as exc:
            return f"[ERROR: {exc}]"

    @agent.tool_plain
    def secret_delete(name: str) -> str:
        try:
            removed = _store().delete(name)
            return f"Deleted secret `{name}`." if removed else f"[ERROR: secret not found: {name}]"
        except (SecretStoreError, ValueError) as exc:
            return f"[ERROR: {exc}]"


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
                    lines.append(f"  episodic_events: {stats.get('episodic_events', '?')} rows")
                    lines.append(f"  memory_items: {stats.get('memory_items', '?')} rows")
                    lines.append(f"  procedures: {stats.get('procedures', '?')} rows")
                    lines.append(f"  feedback_events: {stats.get('feedback_events', '?')} rows")
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
        async def procedure_save(trigger_text: str, checklist: str, kind: str = "procedure") -> str:
            procedure_id = await sqlite.save_procedure(
                trigger_text=trigger_text,
                checklist=checklist,
                kind=kind,
            )
            return f"Saved procedure #{procedure_id}: {trigger_text[:60]}"

        @agent.tool_plain
        async def procedure_search(query: str, limit: int = 5) -> str:
            procedures = await sqlite.search_procedures(query, limit=limit)
            if not procedures:
                return f"(no procedures found for: {query})"
            return "## Procedures\n" + "\n".join(
                f"- #{row['id']} {row['trigger_text']} => {row['checklist']}" for row in procedures
            )

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

        @agent.tool_plain
        async def memory_feedback(
            feedback_kind: str,
            score: float = 1.0,
            task_id: str = "",
            memory_item_id: int | None = None,
            procedure_id: int | None = None,
            details: str = "",
        ) -> str:
            feedback_id = await sqlite.record_feedback(
                task_id=task_id,
                feedback_kind=feedback_kind,
                score=score,
                memory_item_id=memory_item_id,
                procedure_id=procedure_id,
                details={"note": details} if details else None,
            )
            return f"Recorded feedback #{feedback_id} [{feedback_kind}] score={score}"

        @agent.tool_plain
        async def memory_pin(memory_item_id: int, pinned: bool = True) -> str:
            await sqlite.pin_memory_item(memory_item_id, pinned=pinned)
            return f"{'Pinned' if pinned else 'Unpinned'} memory item #{memory_item_id}"

        @agent.tool_plain
        async def procedure_pin(procedure_id: int, pinned: bool = True) -> str:
            await sqlite.pin_procedure(procedure_id, pinned=pinned)
            return f"{'Pinned' if pinned else 'Unpinned'} procedure #{procedure_id}"

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
