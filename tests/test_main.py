from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.loop import Task, TaskResult
from agent.main import RuntimeServices, _build_runtime, _install_signal_handlers, _run_once, _shutdown_runtime, _start


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


def test_install_signal_handlers_ignores_unsupported_platforms() -> None:
    class _Loop:
        def add_signal_handler(self, sig, handler) -> None:
            raise NotImplementedError

    _install_signal_handlers(_Loop(), lambda sig_name: None)  # type: ignore[arg-type]


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
