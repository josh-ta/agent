from __future__ import annotations

import asyncio
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from agent.config import settings
from agent.loop import Task, TaskResult
from agent.main import RuntimeServices, _build_runtime, _install_signal_handlers, _run_once, _shutdown_runtime, _start, run, serve_api, start


def test_start_and_run_commands_delegate_to_asyncio(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    monkeypatch.setattr("agent.main.asyncio.run", lambda coro: seen.append(coro))

    start()
    run("ship it")

    assert len(seen) == 2
    for coro in seen:
        coro.close()


@pytest.mark.asyncio
async def test_run_once_exits_nonzero_on_failure(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    class _Loop:
        async def run_once(self, task: str):
            return TaskResult(task=Task(content=task), output="failed", success=False, elapsed_ms=1.0)

        def stop(self) -> None:
            return None

    runtime = RuntimeServices(
        sqlite=SimpleNamespace(close=lambda: None),
        postgres=None,
        loop=_Loop(),
        bot=None,
    )

    async def _build_runtime(*, start_discord: bool):
        return runtime

    async def _shutdown_runtime_stub(runtime: RuntimeServices) -> None:
        return None

    def _exit(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr("agent.main._build_runtime", _build_runtime)
    monkeypatch.setattr("agent.main._shutdown_runtime", _shutdown_runtime_stub)
    monkeypatch.setattr("agent.main.sys.exit", _exit)

    with pytest.raises(SystemExit, match="1"):
        await _run_once("do the thing")

    assert "failed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_run_once_prints_success_and_shuts_down(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    calls: list[str] = []

    class _Loop:
        async def run_once(self, task: str):
            return TaskResult(task=Task(content=task), output="done", success=True, elapsed_ms=1.0)

        def stop(self) -> None:
            return None

    runtime = RuntimeServices(
        sqlite=SimpleNamespace(close=lambda: None),
        postgres=None,
        loop=_Loop(),
        bot=None,
    )

    async def _build_runtime(*, start_discord: bool):
        return runtime

    async def _shutdown_runtime_stub(runtime: RuntimeServices) -> None:
        calls.append("shutdown")

    monkeypatch.setattr("agent.main._build_runtime", _build_runtime)
    monkeypatch.setattr("agent.main._shutdown_runtime", _shutdown_runtime_stub)

    await _run_once("do the thing")

    assert capsys.readouterr().out.strip() == "done"
    assert calls == ["shutdown"]


@pytest.mark.asyncio
async def test_shutdown_runtime_marks_offline_closes_sqlite_and_stops_loop() -> None:
    calls: list[str] = []

    async def _set_offline() -> None:
        calls.append("offline")

    async def _close() -> None:
        calls.append("close")

    class _Loop:
        def stop(self) -> None:
            calls.append("stop")

    runtime = RuntimeServices(
        sqlite=SimpleNamespace(close=_close),
        postgres=SimpleNamespace(set_offline=_set_offline),
        loop=_Loop(),
        bot=None,
    )

    await _shutdown_runtime(runtime)

    assert calls == ["offline", "close", "stop"]


@pytest.mark.asyncio
async def test_shutdown_runtime_tolerates_offline_and_close_failures() -> None:
    calls: list[str] = []

    async def _set_offline() -> None:
        calls.append("offline")
        raise RuntimeError("boom")

    async def _close() -> None:
        calls.append("close")
        raise RuntimeError("boom")

    class _Loop:
        def stop(self) -> None:
            calls.append("stop")

    runtime = RuntimeServices(
        sqlite=SimpleNamespace(close=_close),
        postgres=SimpleNamespace(set_offline=_set_offline),
        loop=_Loop(),
        bot=None,
    )

    await _shutdown_runtime(runtime)

    assert calls == ["offline", "close", "stop"]


@pytest.mark.asyncio
async def test_shutdown_runtime_handles_missing_postgres() -> None:
    calls: list[str] = []

    async def _close() -> None:
        calls.append("close")

    class _Loop:
        def stop(self) -> None:
            calls.append("stop")

    runtime = RuntimeServices(
        sqlite=SimpleNamespace(close=_close),
        postgres=None,
        loop=_Loop(),
        bot=None,
    )

    await _shutdown_runtime(runtime)

    assert calls == ["close", "stop"]


@pytest.mark.asyncio
async def test_build_runtime_wires_sqlite_only(monkeypatch: pytest.MonkeyPatch, isolated_paths) -> None:
    import agent.communication.discord_bot as discord_bot_module
    import agent.core as core_module
    import agent.loop as loop_module
    import agent.memory.sqlite_store as sqlite_store_module
    import agent.tools.registry as registry_module

    calls: list[tuple[str, object]] = []

    class _SQLiteStore:
        def __init__(self, path) -> None:
            calls.append(("sqlite_path", path))

        async def init(self) -> None:
            calls.append(("sqlite_init", None))

    class _Registry:
        def register_all(self, sqlite, postgres) -> None:
            calls.append(("register_all", (sqlite.__class__.__name__, postgres)))

    class _Loop:
        def __init__(self, agents, *, memory_store, postgres_store) -> None:
            self.agents = agents
            self.memory_store = memory_store
            self.postgres_store = postgres_store
            self.restored = 0

        async def restore_waiting_tasks(self) -> int:
            self.restored += 1
            return 0

    monkeypatch.setattr(settings, "postgres_url", "")
    monkeypatch.setattr(sqlite_store_module, "SQLiteStore", _SQLiteStore)
    monkeypatch.setattr(registry_module, "ToolRegistry", _Registry)
    monkeypatch.setattr(core_module, "create_agents", lambda registry: {"fast": "agent", "registry": registry})
    monkeypatch.setattr(core_module, "set_postgres", lambda postgres: calls.append(("set_postgres", postgres)))
    monkeypatch.setattr(loop_module, "AgentLoop", _Loop)
    monkeypatch.setattr(discord_bot_module, "DiscordBot", lambda loop: ("bot", loop))

    runtime = await _build_runtime(start_discord=False)

    assert runtime.postgres is None
    assert runtime.bot is None
    assert runtime.loop.postgres_store is None
    assert runtime.loop.memory_store.__class__.__name__ == "_SQLiteStore"
    assert runtime.loop.restored == 1
    assert ("sqlite_path", isolated_paths["workspace"] / "agent.db") in calls
    assert ("sqlite_init", None) in calls
    assert ("register_all", ("_SQLiteStore", None)) in calls
    assert not any(name == "set_postgres" for name, _ in calls)


@pytest.mark.asyncio
async def test_build_runtime_wires_postgres_and_bot(monkeypatch: pytest.MonkeyPatch, isolated_paths) -> None:
    import agent.communication.discord_bot as discord_bot_module
    import agent.core as core_module
    import agent.loop as loop_module
    import agent.memory.postgres_store as postgres_store_module
    import agent.memory.sqlite_store as sqlite_store_module
    import agent.tools.registry as registry_module

    calls: list[tuple[str, object]] = []

    class _SQLiteStore:
        def __init__(self, path) -> None:
            self.path = path

        async def init(self) -> None:
            calls.append(("sqlite_init", self.path))

    class _PostgresStore:
        def __init__(self, url: str) -> None:
            self.url = url

        async def init(self) -> None:
            calls.append(("postgres_init", self.url))

        async def register_agent(self) -> None:
            calls.append(("register_agent", self.url))

    class _Registry:
        def register_all(self, sqlite, postgres) -> None:
            calls.append(("register_all", (sqlite.__class__.__name__, postgres.__class__.__name__)))

    class _Loop:
        def __init__(self, agents, *, memory_store, postgres_store) -> None:
            self.agents = agents
            self.memory_store = memory_store
            self.postgres_store = postgres_store
            self.restored = 0

        async def restore_waiting_tasks(self) -> int:
            self.restored += 1
            return 0

    class _Bot:
        def __init__(self, loop) -> None:
            self.loop = loop

    monkeypatch.setattr(settings, "postgres_url", "postgresql://example")
    monkeypatch.setattr(sqlite_store_module, "SQLiteStore", _SQLiteStore)
    monkeypatch.setattr(postgres_store_module, "PostgresStore", _PostgresStore)
    monkeypatch.setattr(registry_module, "ToolRegistry", _Registry)
    monkeypatch.setattr(core_module, "create_agents", lambda registry: {"smart": "agent", "registry": registry})
    monkeypatch.setattr(core_module, "set_postgres", lambda postgres: calls.append(("set_postgres", postgres.url)))
    monkeypatch.setattr(loop_module, "AgentLoop", _Loop)
    monkeypatch.setattr(discord_bot_module, "DiscordBot", _Bot)

    runtime = await _build_runtime(start_discord=True)

    assert runtime.postgres is not None
    assert runtime.bot is not None
    assert runtime.bot.loop is runtime.loop
    assert runtime.loop.postgres_store is runtime.postgres
    assert runtime.loop.restored == 1
    assert ("sqlite_init", isolated_paths["workspace"] / "agent.db") in calls
    assert ("postgres_init", "postgresql://example") in calls
    assert ("register_agent", "postgresql://example") in calls
    assert ("register_all", ("_SQLiteStore", "_PostgresStore")) in calls
    assert ("set_postgres", "postgresql://example") in calls


@pytest.mark.asyncio
async def test_build_runtime_restores_pending_tasks_when_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent.core as core_module
    import agent.loop as loop_module
    import agent.memory.sqlite_store as sqlite_store_module
    import agent.tools.registry as registry_module

    class _SQLiteStore:
        def __init__(self, path) -> None:
            self.path = path

        async def init(self) -> None:
            return None

    class _Registry:
        def register_all(self, sqlite, postgres) -> None:
            return None

    class _Loop:
        def __init__(self, agents, *, memory_store, postgres_store) -> None:
            self.waiting = 0
            self.pending = 0

        async def restore_waiting_tasks(self) -> int:
            self.waiting += 1
            return 0

        async def restore_pending_tasks(self) -> int:
            self.pending += 1
            return 0

    monkeypatch.setattr(settings, "postgres_url", "")
    monkeypatch.setattr(sqlite_store_module, "SQLiteStore", _SQLiteStore)
    monkeypatch.setattr(registry_module, "ToolRegistry", _Registry)
    monkeypatch.setattr(core_module, "create_agents", lambda registry: {"fast": "agent"})
    monkeypatch.setattr(loop_module, "AgentLoop", _Loop)

    runtime = await _build_runtime(start_discord=False)

    assert runtime.loop.waiting == 1
    assert runtime.loop.pending == 1


def test_install_signal_handlers_ignores_unsupported_platforms() -> None:
    class _Loop:
        def add_signal_handler(self, sig, handler) -> None:
            raise NotImplementedError

    _install_signal_handlers(_Loop(), lambda sig_name: None)  # type: ignore[arg-type]


def test_install_signal_handlers_registers_supported_signals() -> None:
    handlers: dict[str, object] = {}

    class _Loop:
        def add_signal_handler(self, sig, handler) -> None:
            handlers[sig.name] = handler

    seen: list[str] = []
    _install_signal_handlers(_Loop(), lambda sig_name: seen.append(sig_name))  # type: ignore[arg-type]

    assert set(handlers) == {"SIGTERM", "SIGINT"}
    handlers["SIGTERM"]()
    handlers["SIGINT"]()
    assert seen == ["SIGTERM", "SIGINT"]


@pytest.mark.asyncio
async def test_start_runs_headless_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _Loop:
        async def run_forever(self) -> None:
            calls.append("run_forever")

        def stop(self) -> None:
            calls.append("stop")

    runtime = RuntimeServices(
        sqlite=SimpleNamespace(close=lambda: None),
        postgres=None,
        loop=_Loop(),
        bot=None,
    )

    async def fake_build_runtime(*, start_discord: bool):
        return runtime

    monkeypatch.setattr("agent.main._build_runtime", fake_build_runtime)
    monkeypatch.setattr("agent.main.settings.discord_bot_token", "")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.0.0")

    await _start()

    assert calls == ["run_forever"]


@pytest.mark.asyncio
async def test_start_runs_discord_and_loop_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _Loop:
        async def run_forever(self) -> None:
            calls.append("loop")

        def stop(self) -> None:
            calls.append("stop")

    class _Bot:
        async def start_bot(self) -> None:
            calls.append("bot")

    runtime = RuntimeServices(
        sqlite=SimpleNamespace(close=lambda: None),
        postgres=None,
        loop=_Loop(),
        bot=_Bot(),
    )

    async def fake_build_runtime(*, start_discord: bool):
        return runtime

    monkeypatch.setattr("agent.main._build_runtime", fake_build_runtime)
    monkeypatch.setattr("agent.main.settings.discord_bot_token", "token")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.0.0")

    await _start()

    assert calls == ["loop", "bot"]


@pytest.mark.asyncio
async def test_start_handles_signal_shutdown_and_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    captured_handler = {}
    shutdown_complete = asyncio.Event()

    class _Loop:
        async def run_forever(self) -> None:
            calls.append("loop")
            await asyncio.sleep(0)
            captured_handler["handler"]("SIGINT")
            await asyncio.sleep(10)

        def stop(self) -> None:
            calls.append("stop")

    class _Bot:
        async def start_bot(self) -> None:
            calls.append("bot")
            await asyncio.sleep(10)

    runtime = RuntimeServices(
        sqlite=SimpleNamespace(close=lambda: None),
        postgres=None,
        loop=_Loop(),
        bot=_Bot(),
    )

    async def fake_build_runtime(*, start_discord: bool):
        return runtime

    async def fake_shutdown_runtime(_runtime: RuntimeServices) -> None:
        calls.append("shutdown")
        shutdown_complete.set()

    def fake_install_signal_handlers(_loop, handler) -> None:
        captured_handler["handler"] = handler

    monkeypatch.setattr("agent.main._build_runtime", fake_build_runtime)
    monkeypatch.setattr("agent.main._shutdown_runtime", fake_shutdown_runtime)
    monkeypatch.setattr("agent.main._install_signal_handlers", fake_install_signal_handlers)
    monkeypatch.setattr("agent.main.settings.discord_bot_token", "token")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.0.0")

    await _start()
    await asyncio.wait_for(shutdown_complete.wait(), timeout=1)

    assert calls[:3] == ["loop", "bot", "shutdown"]


@pytest.mark.asyncio
async def test_start_handles_signal_shutdown_without_main_task(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    captured_handler = {}
    shutdown_complete = asyncio.Event()

    class _Loop:
        async def run_forever(self) -> None:
            calls.append("loop")
            captured_handler["handler"]("SIGTERM")

        def stop(self) -> None:
            calls.append("stop")

    runtime = RuntimeServices(
        sqlite=SimpleNamespace(close=lambda: None),
        postgres=None,
        loop=_Loop(),
        bot=None,
    )

    async def fake_build_runtime(*, start_discord: bool):
        return runtime

    async def fake_shutdown_runtime(_runtime: RuntimeServices) -> None:
        calls.append("shutdown")
        shutdown_complete.set()

    def fake_install_signal_handlers(_loop, handler) -> None:
        captured_handler["handler"] = handler

    monkeypatch.setattr("agent.main._build_runtime", fake_build_runtime)
    monkeypatch.setattr("agent.main._shutdown_runtime", fake_shutdown_runtime)
    monkeypatch.setattr("agent.main._install_signal_handlers", fake_install_signal_handlers)
    monkeypatch.setattr("agent.main.settings.discord_bot_token", "")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.0.0")

    await _start()
    await asyncio.wait_for(shutdown_complete.wait(), timeout=1)

    assert calls == ["loop", "shutdown"]


def test_serve_api_runs_uvicorn_with_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(app, **kwargs) -> None:
        seen["app"] = app
        seen["kwargs"] = kwargs

    monkeypatch.setattr("agent.main.uvicorn.run", fake_run)
    monkeypatch.setattr("agent.main.settings.control_plane_host", "127.0.0.1")
    monkeypatch.setattr("agent.main.settings.control_plane_port", 9000)
    monkeypatch.setattr("agent.main.settings.log_level", "WARNING")

    serve_api()

    assert seen["app"] == "agent.control_plane.app:build_app"
    assert seen["kwargs"] == {
        "factory": True,
        "host": "127.0.0.1",
        "port": 9000,
        "log_level": "warning",
    }


def test_main_module_cli_guard_invokes_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _FakeTyper:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def command(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def __call__(self) -> None:
            calls.append("cli")

    monkeypatch.setattr(typer, "Typer", _FakeTyper)

    runpy.run_path(str(Path(__file__).resolve().parents[1] / "agent" / "main.py"), run_name="__main__")

    assert calls == ["cli"]
