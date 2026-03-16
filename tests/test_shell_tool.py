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
