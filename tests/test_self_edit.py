from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import agent.tools.self_edit as self_edit_module
from agent.tools import self_edit


def test_self_edit_skill_round_trip_and_listing(isolated_paths) -> None:
    save_result = self_edit.edit_skill("new-skill", "# New Skill\nDo the thing.\n")
    read_result = self_edit.read_skill("new-skill")
    list_result = self_edit.list_skills()

    assert "saved" in save_result.lower()
    assert "Do the thing." in read_result
    assert "new-skill" in list_result


def test_self_edit_identity_round_trip(isolated_paths) -> None:
    save_result = self_edit.edit_identity("GOALS.md", "# Goals\nShip.\n")
    read_result = self_edit.read_identity("GOALS.md")

    assert "updated" in save_result.lower()
    assert "Ship." in read_result


def test_self_edit_rejects_invalid_skill_name() -> None:
    result = self_edit.edit_skill("Bad Skill Name", "content")

    assert result.startswith("[ERROR:")


def test_self_edit_rejects_invalid_identity_filename() -> None:
    result = self_edit.edit_identity("NOTES.md", "content")

    assert result.startswith("[ERROR:")


def test_self_edit_read_skill_reports_invalid_and_missing() -> None:
    assert self_edit.read_skill("Bad Skill Name").startswith("[ERROR:")
    assert self_edit.read_skill("missing-skill").startswith("[ERROR: skill 'missing-skill' not found]")


def test_self_edit_read_identity_reports_missing_file() -> None:
    result = self_edit.read_identity("GOALS.md")

    assert "does not exist yet" in result


def test_edit_skill_returns_error_when_write_fails(
    isolated_paths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broken_write_text(self: Path, content: str, encoding: str = "utf-8") -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _broken_write_text)

    result = self_edit.edit_skill("new-skill", "content")

    assert result == "[ERROR: disk full]"


def test_list_skills_reports_missing_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing-skills"
    monkeypatch.setattr(self_edit_module.settings, "skills_path", missing)

    assert self_edit.list_skills() == "(no skills directory found)"


def test_self_edit_list_skills_handles_missing_empty_and_unreadable(
    isolated_paths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skills_dir = isolated_paths["skills"]
    for file in skills_dir.glob("*.md"):
        file.unlink()
    assert self_edit.list_skills() == "(no skills found)"

    broken = skills_dir / "broken.md"
    broken.write_text("# Broken\n", encoding="utf-8")

    original_read_text = Path.read_text

    def _broken_read_text(self: Path, *args, **kwargs):
        if self == broken:
            raise RuntimeError("boom")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _broken_read_text)

    assert "(unreadable)" in self_edit.list_skills()


def test_edit_identity_returns_error_when_write_fails(
    isolated_paths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broken_write_text(self: Path, content: str, encoding: str = "utf-8") -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _broken_write_text)

    result = self_edit.edit_identity("GOALS.md", "content")

    assert result == "[ERROR: disk full]"


def test_read_identity_rejects_invalid_filename() -> None:
    result = self_edit.read_identity("NOTES.md")

    assert result.startswith("[ERROR:")

@pytest.mark.asyncio
async def test_self_restart_returns_disabled_when_configured_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(self_edit_module.settings, "docker_restart_self", False)

    result = await self_edit.self_restart("test")

    assert result == "[DISABLED: DOCKER_RESTART_SELF=false]"


@pytest.mark.asyncio
async def test_self_restart_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Proc:
        returncode = 0

        async def communicate(self):
            return (b"", b"")

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _Proc()

    async def fake_wait_for(awaitable, timeout):
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(self_edit_module.settings, "docker_restart_self", True)
    monkeypatch.setattr(self_edit_module.settings, "agent_container_name", "agent-test")
    monkeypatch.setattr(self_edit_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(self_edit_module.asyncio, "wait_for", fake_wait_for)

    result = await self_edit.self_restart("test")

    assert "timed out" in result


@pytest.mark.asyncio
async def test_self_restart_reports_subprocess_failure_and_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailedProc:
        returncode = 1

        async def communicate(self):
            return (b"", b"container missing")

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _FailedProc()

    monkeypatch.setattr(self_edit_module.settings, "docker_restart_self", True)
    monkeypatch.setattr(self_edit_module.settings, "agent_container_name", "agent-test")
    monkeypatch.setattr(self_edit_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(self_edit_module.asyncio, "wait_for", lambda awaitable, timeout: awaitable)

    failed = await self_edit.self_restart("test")
    assert "ERROR restarting container" in failed

    async def fake_raise(*args, **kwargs):
        raise RuntimeError("docker missing")

    monkeypatch.setattr(self_edit_module.asyncio, "create_subprocess_exec", fake_raise)
    errored = await self_edit.self_restart("test")
    assert errored == "[ERROR: docker missing]"


@pytest.mark.asyncio
async def test_self_restart_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Proc:
        returncode = 0

        async def communicate(self):
            return (b"", b"")

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _Proc()

    monkeypatch.setattr(self_edit_module.settings, "docker_restart_self", True)
    monkeypatch.setattr(self_edit_module.settings, "agent_container_name", "agent-test")
    monkeypatch.setattr(self_edit_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(self_edit_module.asyncio, "wait_for", lambda awaitable, timeout: awaitable)

    result = await self_edit.self_restart("test")

    assert result == "Container 'agent-test' restart initiated. Reason: test"
