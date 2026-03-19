from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.tools.toolsets as toolsets


class _Agent:
    def __init__(self) -> None:
        self.funcs: dict[str, object] = {}

    def tool_plain(self, fn):
        self.funcs[fn.__name__] = fn
        return fn


class _SQLite:
    async def get_stats(self):
        return {
            "conversations": 1,
            "tasks": 2,
            "memory_facts": 3,
            "lessons": 4,
            "db_size_mb": 1.2,
            "vec_enabled": False,
            "last_cleanup": "never",
        }

    async def search_memory(self, query: str, limit: int = 5) -> str:
        return f"memory:{query}:{limit}"

    async def save_memory_fact(self, content: str) -> None:
        self.saved_memory = content

    async def save_lesson(self, summary: str, kind: str = "lesson") -> None:
        self.saved_lesson = (summary, kind)

    async def search_lessons(self, query: str, limit: int = 5) -> str:
        return ""

    async def get_recent_lessons(self, limit: int = 10) -> str:
        return f"recent:{limit}"


class _Postgres:
    async def get_stats(self):
        return {
            "agents": 1,
            "shared_tasks": 2,
            "audit_log": 3,
            "shared_memory": 4,
            "last_cleanup": "never",
        }

    async def list_agents(self) -> str:
        return "agents"

    async def create_task(self, to_agent: str, description: str) -> str:
        return f"{to_agent}:{description}"

    async def get_my_tasks(self) -> str:
        return "tasks"

    async def complete_task(self, task_id: str, result: str) -> str:
        return f"{task_id}:{result}"

    async def broadcast_message(self, message: str) -> str:
        return message

    async def read_broadcasts(self, limit: int = 20) -> str:
        return f"broadcasts:{limit}"

    async def share_memory(self, content: str) -> str:
        return content

    async def search_shared_memory(self, query: str, limit: int = 5) -> str:
        return f"shared:{query}:{limit}"


@pytest.mark.asyncio
async def test_toolsets_attach_and_invoke_wrapped_tools(monkeypatch: pytest.MonkeyPatch, isolated_paths) -> None:
    agent = _Agent()
    sqlite = _SQLite()
    postgres = _Postgres()

    monkeypatch.setattr(toolsets.shell, "shell_run", lambda *args, **kwargs: _async_result("shell"))
    monkeypatch.setattr(toolsets.filesystem, "read_file", lambda path: f"read:{path}")
    monkeypatch.setattr(toolsets.filesystem, "write_file", lambda path, content: f"write:{path}:{content}")
    monkeypatch.setattr(toolsets.filesystem, "list_dir", lambda path=".": f"list:{path}")
    monkeypatch.setattr(toolsets.filesystem, "delete_file", lambda path: f"delete:{path}")
    monkeypatch.setattr(toolsets.filesystem, "str_replace_file", lambda *args: "replace")
    monkeypatch.setattr(toolsets.filesystem, "search_files", lambda *args: "search")
    monkeypatch.setattr(toolsets.self_edit, "list_skills", lambda: "skills")
    monkeypatch.setattr(toolsets.self_edit, "read_skill", lambda name: f"skill:{name}")
    monkeypatch.setattr(toolsets.self_edit, "edit_skill", lambda name, content: f"edit-skill:{name}")
    monkeypatch.setattr(toolsets.self_edit, "read_identity", lambda filename: f"identity:{filename}")
    monkeypatch.setattr(toolsets.self_edit, "edit_identity", lambda filename, content: f"edit-identity:{filename}")
    monkeypatch.setattr(toolsets.self_edit, "self_restart", lambda reason="manual": _async_result(f"restart:{reason}"))
    monkeypatch.setattr(toolsets, "discord_send", lambda channel_id, message: _async_result(f"send:{channel_id}:{message}"))
    monkeypatch.setattr(toolsets, "discord_read", lambda channel_id, limit=20: _async_result(f"read:{channel_id}:{limit}"))
    monkeypatch.setattr(toolsets, "discord_read_named", lambda name, limit=20: _async_result(f"named:{name}:{limit}"))
    monkeypatch.setattr(toolsets, "ask_user", lambda question, timeout=300: _async_result(f"ask:{question}:{timeout}"))
    monkeypatch.setattr(toolsets.github, "pr_view", lambda pr, repo=None: _async_result(f"pr-view:{pr}:{repo}"))
    monkeypatch.setattr(toolsets.github, "pr_list", lambda repo=None, state="open", limit=20: _async_result(f"pr-list:{state}:{limit}"))
    monkeypatch.setattr(toolsets.github, "pr_diff", lambda pr, repo=None: _async_result("pr-diff"))
    monkeypatch.setattr(toolsets.github, "pr_comment", lambda pr, body, repo=None: _async_result("pr-comment"))
    monkeypatch.setattr(toolsets.github, "pr_review", lambda pr, action, body="", repo=None: _async_result("pr-review"))
    monkeypatch.setattr(toolsets.github, "pr_review_with_comments", lambda *args, **kwargs: _async_result("pr-review-inline"))
    monkeypatch.setattr(toolsets.github, "pr_checks", lambda pr, repo=None: _async_result("pr-checks"))
    monkeypatch.setattr(toolsets.github, "pr_merge", lambda pr, method="squash", repo=None: _async_result("pr-merge"))
    monkeypatch.setattr(toolsets.github, "issue_view", lambda issue, repo=None: _async_result("issue-view"))
    monkeypatch.setattr(toolsets.github, "issue_list", lambda repo=None, state="open", limit=20: _async_result("issue-list"))
    monkeypatch.setattr(toolsets.github, "issue_comment", lambda issue, body, repo=None: _async_result("issue-comment"))
    monkeypatch.setattr(toolsets.github, "issue_create", lambda title, body, labels=None, repo=None: _async_result("issue-create"))
    monkeypatch.setattr(toolsets.github, "issue_close", lambda issue, reason="completed", repo=None: _async_result("issue-close"))
    monkeypatch.setattr(toolsets.github, "ci_list", lambda repo=None, branch=None, limit=5: _async_result("ci-list"))
    monkeypatch.setattr(toolsets.github, "ci_view", lambda run_id, repo=None: _async_result("ci-view"))
    monkeypatch.setattr(toolsets.github, "ci_logs_failed", lambda run_id, repo=None: _async_result("ci-logs"))
    monkeypatch.setattr(toolsets.github, "ci_rerun", lambda run_id, failed_only=True, repo=None: _async_result("ci-rerun"))

    toolsets.attach_all_tools(agent, sqlite=sqlite, postgres=postgres)  # type: ignore[arg-type]

    assert await agent.funcs["run_shell"]("ls") == "shell"
    assert agent.funcs["read_file"]("foo.txt") == "read:foo.txt"
    assert agent.funcs["write_file"]("foo.txt", "bar") == "write:foo.txt:bar"
    assert agent.funcs["list_dir"]("src") == "list:src"
    assert agent.funcs["delete_file"]("foo.txt") == "delete:foo.txt"
    assert agent.funcs["str_replace"]("a", "b", "c") == "replace"
    assert agent.funcs["search_files"]("x") == "search"
    assert "Journal updated" in agent.funcs["task_note"]("hello")
    assert "Task Journal" in agent.funcs["task_resume"]()
    assert agent.funcs["task_journal_clear"]() == "Task journal cleared."
    assert agent.funcs["skill_list"]() == "skills"
    assert agent.funcs["skill_read"]("demo") == "skill:demo"
    assert agent.funcs["skill_edit"]("demo", "content") == "edit-skill:demo"
    assert agent.funcs["identity_read"]("IDENTITY.md") == "identity:IDENTITY.md"
    assert agent.funcs["identity_edit"]("IDENTITY.md", "x") == "edit-identity:IDENTITY.md"
    assert await agent.funcs["agent_restart"]("because") == "restart:because"
    assert await agent.funcs["send_discord"](1, "hi") == "send:1:hi"
    assert await agent.funcs["read_discord"](1, 2) == "read:1:2"
    assert await agent.funcs["read_channel"]("private", 3) == "named:private:3"
    assert await agent.funcs["ask_user_question"]("Proceed?", 10) == "ask:Proceed?:10"
    assert await agent.funcs["gh_pr_view"](1) == "pr-view:1:None"
    assert await agent.funcs["gh_pr_list"]() == "pr-list:open:20"
    assert await agent.funcs["gh_pr_diff"](1) == "pr-diff"
    assert await agent.funcs["gh_pr_comment"](1, "body") == "pr-comment"
    assert await agent.funcs["gh_pr_review"](1, "approve") == "pr-review"
    assert await agent.funcs["gh_pr_review_inline"](1, "COMMENT", "body", []) == "pr-review-inline"
    assert await agent.funcs["gh_pr_checks"](1) == "pr-checks"
    assert await agent.funcs["gh_pr_merge"](1) == "pr-merge"
    assert await agent.funcs["gh_issue_view"](1) == "issue-view"
    assert await agent.funcs["gh_issue_list"]() == "issue-list"
    assert await agent.funcs["gh_issue_comment"](1, "body") == "issue-comment"
    assert await agent.funcs["gh_issue_create"]("title", "body") == "issue-create"
    assert await agent.funcs["gh_issue_close"](1) == "issue-close"
    assert await agent.funcs["gh_ci_list"]() == "ci-list"
    assert await agent.funcs["gh_ci_view"]("1") == "ci-view"
    assert await agent.funcs["gh_ci_logs_failed"]("1") == "ci-logs"
    assert await agent.funcs["gh_ci_rerun"]("1") == "ci-rerun"
    assert "SQLite (local)" in await agent.funcs["db_stats"]()
    assert await agent.funcs["memory_search"]("parser") == "memory:parser:5"
    assert await agent.funcs["memory_save"]("fact") == "Saved to memory: fact"
    assert await agent.funcs["lesson_save"]("lesson") == "Lesson recorded [lesson]: lesson"
    assert await agent.funcs["lesson_search"]("nothing") == "(no lessons found for: nothing)"
    assert await agent.funcs["lessons_recent"]() == "recent:10"
    assert await agent.funcs["list_agents"]() == "agents"
    assert await agent.funcs["create_shared_task"]("peer", "work") == "peer:work"
    assert await agent.funcs["my_tasks"]() == "tasks"
    assert await agent.funcs["complete_task"]("id", "result") == "id:result"
    assert await agent.funcs["broadcast_message"]("hello") == "hello"
    assert await agent.funcs["read_broadcasts"]() == "broadcasts:20"
    assert await agent.funcs["share_memory"]("fact") == "fact"
    assert await agent.funcs["search_shared_memory"]("fact") == "shared:fact:5"


async def _async_result(value):
    return value


def test_task_resume_handles_missing_empty_truncated_and_read_error(
    monkeypatch: pytest.MonkeyPatch,
    isolated_paths,
) -> None:
    agent = _Agent()
    monkeypatch.setattr(toolsets.settings, "workspace_path", isolated_paths["workspace"])
    toolsets.attach_journal_tools(agent, sqlite=None)  # type: ignore[arg-type]
    journal_path = isolated_paths["workspace"] / ".task_journal.md"

    assert agent.funcs["task_resume"]() == "(no task journal found — this is a fresh start)"

    journal_path.write_text("", encoding="utf-8")
    assert agent.funcs["task_resume"]() == "(task journal is empty)"

    journal_path.write_text("x" * 9000, encoding="utf-8")
    truncated = agent.funcs["task_resume"]()
    assert "[...older entries truncated...]" in truncated

    original_read_text = Path.read_text

    def _broken_read_text(self: Path, *args, **kwargs):
        if self == journal_path:
            raise OSError("boom")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _broken_read_text)
    assert agent.funcs["task_resume"]() == "[journal read error: boom]"


@pytest.mark.asyncio
async def test_task_note_and_clear_schedule_sqlite_work(monkeypatch: pytest.MonkeyPatch, isolated_paths) -> None:
    class _SQLiteSpy:
        def __init__(self) -> None:
            self.notes: list[tuple[str, str]] = []
            self.cleared: list[str] = []

        async def append_task_note(self, task_id: str, note: str) -> None:
            self.notes.append((task_id, note))

        async def clear_task_checkpoint(self, task_id: str) -> None:
            self.cleared.append(task_id)

    sqlite = _SQLiteSpy()
    agent = _Agent()
    monkeypatch.setattr(toolsets.settings, "workspace_path", isolated_paths["workspace"])
    monkeypatch.setattr(toolsets, "current_task_id", lambda: "task-1")
    toolsets.attach_journal_tools(agent, sqlite=sqlite)  # type: ignore[arg-type]

    result = agent.funcs["task_note"]("hello")
    await asyncio.sleep(0)
    assert "Journal updated" in result
    assert sqlite.notes and sqlite.notes[0][0] == "task-1"

    clear_result = agent.funcs["task_journal_clear"]()
    await asyncio.sleep(0)
    assert clear_result == "Task journal cleared."
    assert sqlite.cleared == ["task-1"]


def test_task_note_and_clear_handle_missing_task_context(monkeypatch: pytest.MonkeyPatch, isolated_paths) -> None:
    agent = _Agent()
    monkeypatch.setattr(toolsets.settings, "workspace_path", isolated_paths["workspace"])
    monkeypatch.setattr(toolsets, "current_task_id", lambda: "")
    toolsets.attach_journal_tools(agent, sqlite=_SQLite())  # type: ignore[arg-type]

    result = agent.funcs["task_note"]("hello")
    clear_result = agent.funcs["task_journal_clear"]()

    assert "Journal updated" in result
    assert clear_result == "Task journal cleared."


def test_task_note_and_clear_report_write_errors(monkeypatch: pytest.MonkeyPatch, isolated_paths) -> None:
    agent = _Agent()
    monkeypatch.setattr(toolsets.settings, "workspace_path", isolated_paths["workspace"])
    toolsets.attach_journal_tools(agent, sqlite=None)  # type: ignore[arg-type]

    original_open = Path.open

    def _broken_open(self: Path, *args, **kwargs):
        raise OSError("open failed")

    def _broken_unlink(self: Path, *args, **kwargs):
        raise OSError("unlink failed")

    monkeypatch.setattr(Path, "open", _broken_open)
    assert agent.funcs["task_note"]("hello") == "[journal write error: open failed]"

    journal_path = isolated_paths["workspace"] / ".task_journal.md"
    monkeypatch.setattr(Path, "open", original_open)
    journal_path.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(Path, "unlink", _broken_unlink)
    assert agent.funcs["task_journal_clear"]() == "[journal clear error: unlink failed]"


def test_task_note_and_clear_skip_async_scheduling_when_loop_not_running(monkeypatch: pytest.MonkeyPatch, isolated_paths) -> None:
    class _Loop:
        def is_running(self) -> bool:
            return False

        def create_task(self, coro) -> None:
            raise AssertionError("create_task should not be called")

    agent = _Agent()
    monkeypatch.setattr(toolsets.settings, "workspace_path", isolated_paths["workspace"])
    monkeypatch.setattr(toolsets, "current_task_id", lambda: "task-1")
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: _Loop())
    toolsets.attach_journal_tools(agent, sqlite=_SQLite())  # type: ignore[arg-type]

    assert "Journal updated" in agent.funcs["task_note"]("hello")
    assert agent.funcs["task_journal_clear"]() == "Task journal cleared."


def test_task_journal_clear_ignores_running_loop_lookup_errors(monkeypatch: pytest.MonkeyPatch, isolated_paths) -> None:
    agent = _Agent()
    monkeypatch.setattr(toolsets.settings, "workspace_path", isolated_paths["workspace"])
    monkeypatch.setattr(toolsets, "current_task_id", lambda: "task-1")
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: (_ for _ in ()).throw(RuntimeError("no loop")))
    toolsets.attach_journal_tools(agent, sqlite=_SQLite())  # type: ignore[arg-type]

    assert agent.funcs["task_journal_clear"]() == "Task journal cleared."


@pytest.mark.asyncio
async def test_db_stats_reports_sqlite_and_postgres_errors() -> None:
    class _BrokenSQLite:
        async def get_stats(self):
            raise RuntimeError("sqlite down")

    class _BrokenPostgres:
        async def get_stats(self):
            raise RuntimeError("postgres down")

    agent = _Agent()
    toolsets.attach_database_tools(agent, sqlite=_BrokenSQLite(), postgres=_BrokenPostgres())  # type: ignore[arg-type]

    result = await agent.funcs["db_stats"]()

    assert "SQLite error: sqlite down" in result
    assert "Postgres error: postgres down" in result


@pytest.mark.asyncio
async def test_attach_database_tools_with_single_backend() -> None:
    sqlite_agent = _Agent()
    toolsets.attach_database_tools(sqlite_agent, sqlite=_SQLite(), postgres=None)  # type: ignore[arg-type]
    assert "memory_search" in sqlite_agent.funcs
    assert "list_agents" not in sqlite_agent.funcs
    assert "SQLite (local)" in await sqlite_agent.funcs["db_stats"]()
    assert "Postgres (shared)" not in await sqlite_agent.funcs["db_stats"]()

    postgres_agent = _Agent()
    toolsets.attach_database_tools(postgres_agent, sqlite=None, postgres=_Postgres())  # type: ignore[arg-type]
    assert "list_agents" in postgres_agent.funcs
    assert "memory_search" not in postgres_agent.funcs
    assert "Postgres (shared)" in await postgres_agent.funcs["db_stats"]()
    assert "SQLite (local)" not in await postgres_agent.funcs["db_stats"]()


def test_attach_secret_tools_covers_success_empty_and_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _Agent()
    state = {"names": [], "values": {}, "mode": "ok"}

    class _SecretStore:
        def __init__(self, path) -> None:
            self.path = path

        def list_names(self):
            if state["mode"] == "list_error":
                raise toolsets.SecretStoreError("list failed")
            return list(state["names"])

        def set(self, name: str, value: str) -> None:
            if state["mode"] == "set_error":
                raise toolsets.SecretStoreError("set failed")
            if state["mode"] == "set_value_error":
                raise ValueError("bad name")
            state["values"][name] = value
            if name not in state["names"]:
                state["names"].append(name)

        def get(self, name: str) -> str:
            if state["mode"] == "get_error":
                raise toolsets.SecretStoreError("get failed")
            if name not in state["values"]:
                raise toolsets.SecretNotFoundError(name)
            return state["values"][name]

        def delete(self, name: str) -> bool:
            if state["mode"] == "delete_error":
                raise toolsets.SecretStoreError("delete failed")
            return state["values"].pop(name, None) is not None

    monkeypatch.setattr(toolsets, "SecretStore", _SecretStore)
    toolsets.attach_secret_tools(agent)  # type: ignore[arg-type]

    assert agent.funcs["secret_list"]() == "(no secrets stored)"
    assert "Stored secret `TOKEN`" in agent.funcs["secret_set"]("TOKEN", "hunter2")
    assert "## Stored secrets" in agent.funcs["secret_list"]()
    assert agent.funcs["secret_get"]("TOKEN") == "hunter2"
    assert agent.funcs["secret_delete"]("TOKEN") == "Deleted secret `TOKEN`."
    assert agent.funcs["secret_delete"]("TOKEN") == "[ERROR: secret not found: TOKEN]"
    assert agent.funcs["secret_get"]("TOKEN") == "[ERROR: secret not found: TOKEN]"

    state["mode"] = "set_error"
    assert "set failed" in agent.funcs["secret_set"]("TOKEN", "value")
    state["mode"] = "set_value_error"
    assert "bad name" in agent.funcs["secret_set"]("TOKEN", "value")
    state["mode"] = "get_error"
    assert "get failed" in agent.funcs["secret_get"]("TOKEN")
    state["mode"] = "delete_error"
    assert "delete failed" in agent.funcs["secret_delete"]("TOKEN")
