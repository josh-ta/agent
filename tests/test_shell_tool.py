from __future__ import annotations

import asyncio

import pytest

import agent.tools.shell as shell_module
from agent.tools.shell import shell_run


@pytest.mark.asyncio
async def test_shell_run_returns_output_and_emits_events(event_collector, isolated_paths) -> None:
    result = await shell_run(
        "python - <<'PY'\nprint('hello from shell')\nPY",
        working_dir=str(isolated_paths["workspace"]),
        timeout=5,
    )

    assert "hello from shell" in result
    assert [event.kind for event in event_collector] == [
        "shell_start",
        "shell_output",
        "shell_done",
    ]


@pytest.mark.asyncio
async def test_shell_run_tail_lines_mode(isolated_paths) -> None:
    result = await shell_run(
        "python - <<'PY'\nfor i in range(5):\n    print(f'line-{i}')\nPY",
        working_dir=str(isolated_paths["workspace"]),
        timeout=5,
        tail_lines=2,
    )

    assert "line-3" in result
    assert "line-4" in result
    assert "line-0" not in result


@pytest.mark.asyncio
async def test_shell_run_tail_lines_returns_full_output_when_shorter_than_tail(isolated_paths) -> None:
    result = await shell_run(
        "python - <<'PY'\nprint('line-1')\nprint('line-2')\nPY",
        working_dir=str(isolated_paths["workspace"]),
        timeout=5,
        tail_lines=5,
    )

    assert "line-1" in result
    assert "line-2" in result
    assert "earlier lines omitted" not in result


@pytest.mark.asyncio
async def test_shell_run_resolves_relative_working_dir_against_workspace(isolated_paths) -> None:
    subdir = isolated_paths["workspace"] / "repo"
    subdir.mkdir()

    result = await shell_run("pwd", working_dir="repo", timeout=5)

    assert str(subdir) in result
    assert "[exit code: 0]" in result


@pytest.mark.asyncio
async def test_shell_run_errors_for_missing_explicit_working_dir(isolated_paths) -> None:
    result = await shell_run("pwd", working_dir="missing-repo", timeout=5)

    assert result == f"[ERROR: working_dir not found: {isolated_paths['workspace'] / 'missing-repo'}]"


@pytest.mark.asyncio
async def test_shell_run_errors_for_non_directory_working_dir(isolated_paths) -> None:
    file_path = isolated_paths["workspace"] / "file.txt"
    file_path.write_text("hello", encoding="utf-8")

    result = await shell_run("pwd", working_dir=str(file_path), timeout=5)

    assert result == f"[ERROR: working_dir is not a directory: {file_path}]"


@pytest.mark.asyncio
async def test_shell_run_blocks_remote_ssh_without_preflight(isolated_paths) -> None:
    result = await shell_run(
        "ssh root@89.167.14.150 'hostname'",
        working_dir=str(isolated_paths["workspace"]),
        timeout=5,
    )

    assert "remote command blocked" in result
    assert "remote-preflight" in result


@pytest.mark.asyncio
async def test_shell_run_blocks_guessed_remote_target_even_with_preflight(isolated_paths) -> None:
    result = await shell_run(
        "# remote-preflight: workspace=/workspace/app; basis=user-provided host\nssh root@89.167.14.150 'cd /root/theticketactionapp && git pull'",
        working_dir=str(isolated_paths["workspace"]),
        timeout=5,
    )

    assert "guessed or placeholder deploy target" in result


@pytest.mark.asyncio
async def test_shell_run_validates_remote_preflight_workspace_path(isolated_paths) -> None:
    result = await shell_run(
        "# remote-preflight: workspace=/workspace/missing; basis=user-provided host\nssh root@89.167.14.150 'hostname'",
        working_dir=str(isolated_paths["workspace"]),
        timeout=5,
    )

    assert "preflight workspace path not found" in result


@pytest.mark.asyncio
async def test_shell_run_allows_remote_ssh_with_valid_preflight(monkeypatch: pytest.MonkeyPatch, isolated_paths) -> None:
    repo = isolated_paths["workspace"] / "repo"
    repo.mkdir()
    calls: list[str] = []

    async def fake_create_subprocess_shell(command, **kwargs):
        calls.append(command)

        class _Stdout:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        class _Proc:
            stdout = _Stdout()
            returncode = 0

            async def wait(self) -> None:
                return None

        return _Proc()

    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_shell", fake_create_subprocess_shell)

    result = await shell_run(
        f"# remote-preflight: workspace={repo}; basis=user-provided host\nssh root@89.167.14.150 'hostname'",
        working_dir=str(isolated_paths["workspace"]),
        timeout=5,
    )

    assert calls
    assert "[exit code: 0]" in result


@pytest.mark.asyncio
async def test_shell_run_returns_timeout_message(isolated_paths) -> None:
    result = await shell_run(
        "python - <<'PY'\nimport time\ntime.sleep(2)\nPY",
        working_dir=str(isolated_paths["workspace"]),
        timeout=1,
    )

    assert result.startswith("[TIMEOUT after 1s]")


@pytest.mark.asyncio
async def test_shell_run_truncates_large_output(isolated_paths) -> None:
    result = await shell_run(
        "python - <<'PY'\nprint('x' * 12000)\nPY",
        working_dir=str(isolated_paths["workspace"]),
        timeout=5,
    )

    assert "truncated at 10KB" in result


@pytest.mark.asyncio
async def test_shell_run_reports_unexpected_subprocess_error(
    monkeypatch: pytest.MonkeyPatch,
    isolated_paths,
) -> None:
    async def fail_subprocess(*args, **kwargs):
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(shell_module.settings, "workspace_path", isolated_paths["workspace"])
    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_shell", fail_subprocess)

    result = await shell_run("echo hi", timeout=5)

    assert result == "[ERROR: spawn failed]"


@pytest.mark.asyncio
async def test_shell_run_timeout_drain_and_communicate_timeouts(monkeypatch: pytest.MonkeyPatch, isolated_paths) -> None:
    class _Stdout:
        def __init__(self) -> None:
            self._chunks = [b"late line\n"]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._chunks:
                return self._chunks.pop(0)
            raise StopAsyncIteration

    class _Proc:
        def __init__(self) -> None:
            self.stdout = _Stdout()
            self.returncode = None

        def kill(self) -> None:
            self.returncode = 143

        async def wait(self) -> None:
            return None

        async def communicate(self):
            return (b"", b"")

    calls = {"wait_for": 0}

    async def fake_create_subprocess_shell(*args, **kwargs):
        return _Proc()

    async def fake_wait_for(awaitable, timeout):
        calls["wait_for"] += 1
        if calls["wait_for"] == 1:
            awaitable.close()
            raise asyncio.TimeoutError
        if calls["wait_for"] == 2:
            return await awaitable
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(shell_module.settings, "workspace_path", isolated_paths["workspace"])
    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_shell", fake_create_subprocess_shell)
    monkeypatch.setattr(shell_module.asyncio, "wait_for", fake_wait_for)

    result = await shell_run("echo hi", timeout=1)

    assert result.startswith("[TIMEOUT after 1s]")


@pytest.mark.asyncio
async def test_shell_run_timeout_handles_drain_timeout(monkeypatch: pytest.MonkeyPatch, isolated_paths) -> None:
    class _Proc:
        def __init__(self) -> None:
            self.stdout = None
            self.returncode = None

        def kill(self) -> None:
            self.returncode = 143

        async def wait(self) -> None:
            return None

        async def communicate(self):
            return (b"", b"")

    calls = {"wait_for": 0}

    async def fake_create_subprocess_shell(*args, **kwargs):
        return _Proc()

    async def fake_wait_for(awaitable, timeout):
        calls["wait_for"] += 1
        if calls["wait_for"] == 1:
            awaitable.close()
            raise asyncio.TimeoutError
        if calls["wait_for"] == 2:
            awaitable.close()
            raise asyncio.TimeoutError
        return await awaitable

    monkeypatch.setattr(shell_module.settings, "workspace_path", isolated_paths["workspace"])
    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_shell", fake_create_subprocess_shell)
    monkeypatch.setattr(shell_module.asyncio, "wait_for", fake_wait_for)

    result = await shell_run("echo hi", timeout=1)

    assert result.startswith("[TIMEOUT after 1s]")
