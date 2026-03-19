from __future__ import annotations

import asyncio
import aiosqlite
import sys
from types import ModuleType
from types import SimpleNamespace
import pytest

import agent.memory.sqlite_store as sqlite_store_module
from agent.config import settings
from agent.loop import Task, TaskResult
from agent.memory.sqlite_store import SQLiteStore, _embed


class _FakeExecResult:
    def __init__(self, rows=None) -> None:
        self._rows = rows or []

    def __await__(self):
        async def _done():
            return self

        return _done().__await__()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def fetchall(self):
        return self._rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeInitDb:
    def __init__(self) -> None:
        self.row_factory = None
        self.executed: list[str] = []
        self.loaded_extensions: list[str] = []
        self.enable_calls: list[bool] = []

    def execute(self, sql: str, *args, **kwargs):
        self.executed.append(sql)
        if "PRAGMA table_info(tasks)" in sql:
            return _FakeExecResult(
                [
                    {"name": "task_id"},
                    {"name": "status"},
                    {"name": "metadata"},
                    {"name": "error"},
                    {"name": "created_ts"},
                    {"name": "started_ts"},
                    {"name": "finished_ts"},
                    {"name": "updated_ts"},
                ]
            )
        if "SELECT value FROM _meta" in sql:
            return _FakeExecResult([])
        return _FakeExecResult([])

    async def executescript(self, script: str) -> None:
        self.executed.append("executescript")

    async def commit(self) -> None:
        return None

    async def enable_load_extension(self, enabled: bool) -> None:
        self.enable_calls.append(enabled)

    async def load_extension(self, path: str) -> None:
        self.loaded_extensions.append(path)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_saves_messages_and_returns_history(sqlite_store) -> None:
    await sqlite_store.save_message("user", "hello", channel_id=7)
    await sqlite_store.save_message("assistant", "hi there", channel_id=7)

    history = await sqlite_store.get_history(channel_id=7)

    assert [row["content"] for row in history] == ["hello", "hi there"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_records_tasks_memory_and_lessons(sqlite_store) -> None:
    task = Task(content="Fix tests", source="discord", author="Josh")
    result = TaskResult(task=task, output="done", success=True, elapsed_ms=12.0, tool_calls=3)

    await sqlite_store.record_task(task, result)
    await sqlite_store.save_memory_fact("The parser needs a fixture.")
    await sqlite_store.save_lesson("Use parametrized tests for routing.", kind="pattern")

    memory_results = await sqlite_store.search_memory("parser")
    lessons = await sqlite_store.search_lessons("routing")
    stats = await sqlite_store.get_stats()

    assert "parser" in memory_results.lower()
    assert "routing" in lessons.lower()
    assert stats["tasks"] == 1
    assert stats["memory_facts"] == 1
    assert stats["lessons"] == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_persists_sessions_turns_and_checkpoints(sqlite_store) -> None:
    await sqlite_store.ensure_session(
        session_id="discord:101:55",
        source="discord",
        channel_id=101,
        title="Investigate flaky test",
        pending_task_id="task-55",
        metadata={"author": "Josh"},
    )
    await sqlite_store.append_session_turn(
        session_id="discord:101:55",
        role="user",
        content="Investigate the flaky integration test",
        task_id="task-55",
    )
    await sqlite_store.append_session_turn(
        session_id="discord:101:55",
        role="assistant",
        content="Which environment should I use?",
        turn_kind="question",
        task_id="task-55",
    )
    await sqlite_store.save_task_checkpoint(
        task_id="task-55",
        session_id="discord:101:55",
        summary="Waiting on environment clarification.",
        notes="Collected initial failures.",
    )

    session = await sqlite_store.get_session("discord:101:55")
    turns = await sqlite_store.list_session_turns("discord:101:55")
    context = await sqlite_store.get_session_context("discord:101:55")
    checkpoint = await sqlite_store.get_task_checkpoint("task-55")

    assert session is not None
    assert session["pending_task_id"] == "task-55"
    assert len(turns) == 2
    assert "Latest user goal" in session["summary"]
    assert "Session summary" in context
    assert checkpoint is not None
    assert checkpoint["summary"] == "Waiting on environment clarification."


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_updates_session_status_and_truncates_context(sqlite_store) -> None:
    await sqlite_store.ensure_session(
        session_id="discord:101:55",
        source="discord",
        channel_id=101,
        title="Investigate flaky test",
        pending_task_id="task-55",
        metadata={"author": "Josh"},
    )
    await sqlite_store.append_session_turn(
        session_id="discord:101:55",
        role="user",
        content="A" * 400,
        turn_kind="message",
        task_id="task-55",
    )
    await sqlite_store.set_session_status(
        "discord:101:55",
        status="paused",
        pending_task_id="",
        metadata={"state": "paused"},
    )

    session = await sqlite_store.get_session("discord:101:55")
    context = await sqlite_store.get_session_context("discord:101:55", char_cap=400)

    assert session is not None
    assert session["status"] == "paused"
    assert session["pending_task_id"] == ""
    assert session["metadata"] == {"state": "paused"}
    assert "Recent session turns" in context


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_session_context_handles_small_caps_and_empty_sessions(sqlite_store) -> None:
    await sqlite_store.ensure_session(
        session_id="discord:101:small",
        source="discord",
        channel_id=101,
        title="Short context",
    )
    await sqlite_store.append_session_turn(
        session_id="discord:101:small",
        role="user",
        content="A" * 400,
        task_id="task-55",
    )

    context = await sqlite_store.get_session_context("discord:101:small", char_cap=20)
    empty_session = await sqlite_store.get_session_context("discord:101:empty")

    assert "## Recent session turns" not in context
    assert empty_session == ""


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_existing_empty_session_has_empty_context(sqlite_store) -> None:
    await sqlite_store.ensure_session(
        session_id="discord:101:empty-existing",
        source="discord",
        channel_id=101,
        title="Empty context",
    )

    assert await sqlite_store.get_session_context("discord:101:empty-existing") == ""


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_session_helpers_handle_missing_rows_and_empty_inputs(sqlite_store) -> None:
    await sqlite_store.append_session_turn(
        session_id="",
        role="user",
        content="",
        task_id="task-1",
    )
    await sqlite_store.set_session_status("missing-session", status="paused")

    assert await sqlite_store.get_session("missing-session") is None
    assert await sqlite_store.get_session_context("missing-session") == ""


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_append_session_turn_trims_open_questions_and_updates_summary(sqlite_store) -> None:
    await sqlite_store.ensure_session(
        session_id="discord:101:55",
        source="discord",
        channel_id=101,
        title="Investigate flaky test",
        pending_task_id="task-55",
    )
    for index in range(4):
        await sqlite_store.append_session_turn(
            session_id="discord:101:55",
            role="assistant",
            content=f"Question {index}",
            turn_kind="question",
            task_id="task-55",
        )
    await sqlite_store.append_session_turn(
        session_id="discord:101:55",
        role="assistant",
        content="Answer received",
        turn_kind="answer",
        task_id="task-55",
    )

    session = await sqlite_store.get_session("discord:101:55")

    assert session is not None
    assert session["open_questions"] == ["Question 1", "Question 2", "Question 3"]
    assert "Latest agent reply: Answer received" in session["summary"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_json_recovery_paths_fall_back_to_empty_structures(sqlite_store) -> None:
    await sqlite_store.ensure_session(
        session_id="discord:101:55",
        source="discord",
        channel_id=101,
        title="Investigate flaky test",
        pending_task_id="task-55",
    )
    await sqlite_store.append_session_turn(
        session_id="discord:101:55",
        role="user",
        content="Investigate the flaky integration test",
        task_id="task-55",
        metadata={"author": "Josh"},
    )
    await sqlite_store.save_task_checkpoint(
        task_id="task-55",
        session_id="discord:101:55",
        summary="Waiting",
        metadata={"state": "ok"},
    )
    assert sqlite_store._db is not None
    await sqlite_store._db.execute(
        "UPDATE conversation_sessions SET open_questions=?, metadata=? WHERE session_id=?",
        ("{bad", "{bad", "discord:101:55"),
    )
    await sqlite_store._db.execute(
        "UPDATE conversation_turns SET metadata=? WHERE session_id=?",
        ("{bad", "discord:101:55"),
    )
    await sqlite_store._db.execute(
        "UPDATE task_checkpoints SET metadata=? WHERE task_id=?",
        ("{bad", "task-55"),
    )
    await sqlite_store._db.commit()

    session = await sqlite_store.get_session("discord:101:55")
    turns = await sqlite_store.list_session_turns("discord:101:55")
    checkpoint = await sqlite_store.get_task_checkpoint("task-55")

    assert session is not None and session["open_questions"] == []
    assert session["metadata"] == {}
    assert turns[0]["metadata"] == {}
    assert checkpoint is not None and checkpoint["metadata"] == {}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_appends_and_clears_task_notes(sqlite_store) -> None:
    await sqlite_store.save_task_checkpoint(
        task_id="task-55",
        session_id="discord:101:55",
        summary="Waiting on environment clarification.",
        notes="Collected initial failures.",
    )

    await sqlite_store.append_task_note("task-55", "Need one more repro case.")
    checkpoint = await sqlite_store.get_task_checkpoint("task-55")
    assert checkpoint is not None
    assert "Collected initial failures." in checkpoint["notes"]
    assert "Need one more repro case." in checkpoint["notes"]

    await sqlite_store.clear_task_checkpoint("task-55")
    assert await sqlite_store.get_task_checkpoint("task-55") is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_save_task_checkpoint_ignores_blank_task_id(sqlite_store) -> None:
    await sqlite_store.save_task_checkpoint(task_id="", summary="ignored")

    assert await sqlite_store.get_task_checkpoint("") is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_persists_api_task_lifecycle(sqlite_store) -> None:
    await sqlite_store.create_task_record(
        task_id="task-123",
        source="api",
        author="tester",
        content="Ship the API",
    )
    queued = await sqlite_store.get_task_record("task-123")
    assert queued is not None
    assert queued["status"] == "queued"

    await sqlite_store.mark_task_running("task-123")
    running = await sqlite_store.get_task_record("task-123")
    assert running is not None
    assert running["status"] == "running"
    assert running["started_ts"] is not None

    task = Task(content="Ship the API", source="api", author="tester", metadata={"task_id": "task-123"})
    result = TaskResult(task=task, output="done", success=True, elapsed_ms=42.0, tool_calls=4)
    await sqlite_store.record_task(task, result)

    finished = await sqlite_store.get_task_record("task-123")
    assert finished is not None
    assert finished["status"] == "succeeded"
    assert finished["result"] == "done"
    assert finished["error"] is None
    assert finished["tool_calls"] == 4
    assert finished["finished_ts"] is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_lists_pending_and_fails_task(sqlite_store) -> None:
    await sqlite_store.create_task_record(
        task_id="task-queued",
        source="api",
        author="tester",
        content="Ship the API",
        metadata={"task_id": "task-queued"},
    )
    await sqlite_store.create_task_record(
        task_id="task-running",
        source="api",
        author="tester",
        content="Run the API",
        metadata={"task_id": "task-running"},
    )
    await sqlite_store.mark_task_running("task-running")

    pending = await sqlite_store.list_pending_task_records()
    assert {row["task_id"] for row in pending} == {"task-queued", "task-running"}

    await sqlite_store.fail_task("task-running", error="boom", metadata={"state": "failed"})
    failed = await sqlite_store.get_task_record("task-running")
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["error"] == "boom"
    assert failed["metadata"] == {"state": "failed"}

    await sqlite_store.fail_task("task-queued", error="oops", metadata=None)
    missing = await sqlite_store.get_task_record("missing-task")
    queued = await sqlite_store.get_task_record("task-queued")
    assert missing is None
    assert queued is not None
    assert queued["status"] == "failed"
    assert queued["error"] == "oops"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_mark_task_queued_with_metadata_and_pending_json_recovery(sqlite_store) -> None:
    await sqlite_store.create_task_record(
        task_id="task-queued",
        source="api",
        author="tester",
        content="Ship the API",
        metadata={"task_id": "task-queued"},
    )
    await sqlite_store.mark_task_queued("task-queued", metadata={"state": "queued"})
    row = await sqlite_store.get_task_record("task-queued")
    assert row is not None
    assert row["metadata"] == {"state": "queued"}

    assert sqlite_store._db is not None
    await sqlite_store._db.execute("UPDATE tasks SET metadata=? WHERE task_id=?", ("{bad", "task-queued"))
    await sqlite_store._db.commit()
    pending = await sqlite_store.list_pending_task_records()
    assert pending[0]["metadata"] == {}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_lists_waiting_tasks(sqlite_store) -> None:
    await sqlite_store.create_task_record(
        task_id="task-wait",
        source="discord",
        author="tester",
        content="Need clarification",
        metadata={"task_id": "task-wait"},
    )
    await sqlite_store.mark_task_waiting(
        "task-wait",
        metadata={
            "task_id": "task-wait",
            "wait_state": {
                "question": "Which environment?",
                "timeout_s": 90,
                "channel_id": 101,
                "message_id": 55,
                "prompt_message_id": 66,
            },
        },
        question="Which environment?",
    )

    rows = await sqlite_store.list_waiting_task_records()

    assert len(rows) == 1
    assert rows[0]["task_id"] == "task-wait"
    assert rows[0]["metadata"]["wait_state"]["prompt_message_id"] == 66


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_cleanup_applies_retention(monkeypatch: pytest.MonkeyPatch, sqlite_store) -> None:
    monkeypatch.setattr("agent.memory.sqlite_components.settings.retention_memory_facts_max", 1)
    monkeypatch.setattr("agent.memory.sqlite_components.settings.retention_lessons_max", 1)
    monkeypatch.setattr("agent.memory.sqlite_components.settings.retention_tasks_days", 0)
    monkeypatch.setattr("agent.memory.sqlite_components.settings.retention_conversations_days", 0)

    await sqlite_store.save_message("user", "first", channel_id=1)
    await sqlite_store.save_message("assistant", "second", channel_id=1)
    await sqlite_store.record_task(
        Task(content="old"),
        TaskResult(task=Task(content="old"), output="done", success=True, elapsed_ms=1.0),
    )
    await sqlite_store.save_memory_fact("fact one")
    await sqlite_store.save_memory_fact("fact two")
    await sqlite_store.save_lesson("lesson one")
    await sqlite_store.save_lesson("lesson two")

    await sqlite_store._cleanup()
    stats = await sqlite_store.get_stats()

    assert stats["memory_facts"] == 1
    assert stats["lessons"] == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_recent_lessons_and_close_edge_paths(sqlite_store, isolated_paths) -> None:
    assert await sqlite_store.get_recent_lessons() == "(no lessons recorded yet)"

    await sqlite_store.save_lesson("lesson one")
    recent = await sqlite_store.get_recent_lessons()
    assert "lesson one" in recent

    await sqlite_store.close()
    await sqlite_store.close()

    fresh = SQLiteStore(isolated_paths["workspace"] / "fresh.db")
    with pytest.raises(RuntimeError, match="call await init"):
        await fresh.get_stats()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_heartbeat_delegates_and_init_restores_cleanup_meta(isolated_paths, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = isolated_paths["workspace"] / "heartbeat.db"
    store = SQLiteStore(db_path)
    await store.init()
    try:
        calls: list[str] = []

        async def fake_heartbeat() -> None:
            calls.append("heartbeat")

        monkeypatch.setattr(store.maintenance, "heartbeat", fake_heartbeat)
        await store.heartbeat()
        assert calls == ["heartbeat"]

        assert store._db is not None
        await store._db.execute(
            "INSERT OR REPLACE INTO _meta(key,value) VALUES('last_cleanup_ts',?)",
            ("123.5",),
        )
        await store._db.commit()
    finally:
        await store.close()

    reopened = SQLiteStore(db_path)
    await reopened.init()
    try:
        assert reopened._last_cleanup_ts == 123.5

        assert reopened._db is not None
        await reopened._db.execute(
            "INSERT OR REPLACE INTO _meta(key,value) VALUES('last_cleanup_ts',?)",
            ("not-a-float",),
        )
        await reopened._db.commit()
    finally:
        await reopened.close()

    invalid = SQLiteStore(db_path)
    await invalid.init()
    try:
        assert invalid._last_cleanup_ts == 0.0
    finally:
        await invalid.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_init_migrates_legacy_tasks_before_creating_task_id_index(
    isolated_paths,
) -> None:
    db_path = isolated_paths["workspace"] / "legacy.db"
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(
            """
            CREATE TABLE tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT NOT NULL,
                author      TEXT NOT NULL DEFAULT 'system',
                content     TEXT NOT NULL,
                result      TEXT,
                success     INTEGER NOT NULL DEFAULT 0,
                elapsed_ms  REAL,
                tool_calls  INTEGER DEFAULT 0,
                ts          REAL NOT NULL
            );
            """
        )
        await db.commit()

    store = SQLiteStore(db_path)
    await store.init()
    try:
        await store.create_task_record(
            task_id="task-legacy",
            source="api",
            author="tester",
            content="Verify migration",
        )
        row = await store.get_task_record("task-legacy")
        assert row is not None
        assert row["status"] == "queued"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_store_init_loads_sqlite_vec_and_skips_blank_pragmas(monkeypatch: pytest.MonkeyPatch, isolated_paths) -> None:
    fake_db = _FakeInitDb()
    module = ModuleType("sqlite_vec")
    module.loadable_path = lambda: "/tmp/sqlite_vec"

    async def fake_connect(path: str):
        return fake_db

    monkeypatch.setitem(sys.modules, "sqlite_vec", module)
    monkeypatch.setattr(sqlite_store_module.aiosqlite, "connect", fake_connect)
    monkeypatch.setattr(sqlite_store_module, "PRAGMAS", "\nPRAGMA one;\n\nPRAGMA two;\n")

    store = SQLiteStore(isolated_paths["workspace"] / "vec.db")
    await store.init()

    assert store._has_vec is True
    assert fake_db.enable_calls == [True, False]
    assert fake_db.loaded_extensions == ["/tmp/sqlite_vec"]
    assert "PRAGMA one;" in fake_db.executed
    assert "PRAGMA two;" in fake_db.executed
    assert any("CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0" in sql for sql in fake_db.executed)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_healthcheck_reports_query_failures(sqlite_store, monkeypatch: pytest.MonkeyPatch) -> None:
    assert await sqlite_store.healthcheck() is True

    class _FailingDb:
        def __init__(self, db) -> None:
            self._db = db

        def execute(self, sql, *args, **kwargs):
            raise RuntimeError("db broken")

        def __getattr__(self, name):
            return getattr(self._db, name)

    original_db = sqlite_store._db
    assert original_db is not None
    sqlite_store._db = _FailingDb(original_db)
    try:
        assert await sqlite_store.healthcheck() is False
    finally:
        sqlite_store._db = original_db


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sqlite_store_append_session_turn_handles_missing_cached_session(sqlite_store, monkeypatch: pytest.MonkeyPatch) -> None:
    await sqlite_store.ensure_session(
        session_id="discord:101:missing-cache",
        source="discord",
        channel_id=101,
        title="Cached session",
    )

    monkeypatch.setattr(sqlite_store, "get_session", lambda session_id: asyncio.sleep(0, result=None))

    await sqlite_store.append_session_turn(
        session_id="discord:101:missing-cache",
        role="assistant",
        content="hello",
    )


@pytest.mark.asyncio
async def test_sqlite_store_embed_helper_returns_none_when_disabled_and_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "")
    assert await _embed("hello") is None

    monkeypatch.setattr(settings, "openai_api_key", "token")
    module = ModuleType("openai")

    class _Client:
        def __init__(self, api_key: str) -> None:
            self.embeddings = self

        async def create(self, *, model: str, input: str, encoding_format: str):
            raise RuntimeError("boom")

    module.AsyncOpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", module)

    assert await _embed("hello") is None


@pytest.mark.asyncio
async def test_sqlite_store_embed_helper_returns_embedding_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "token")
    module = ModuleType("openai")

    class _Client:
        def __init__(self, api_key: str) -> None:
            self.embeddings = self

        async def create(self, *, model: str, input: str, encoding_format: str):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.3, 0.4])])

    module.AsyncOpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", module)

    assert await _embed("hello") == [0.3, 0.4]
