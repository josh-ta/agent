from __future__ import annotations

import pytest

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
