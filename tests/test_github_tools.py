from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent.tools.github as github


@pytest.mark.asyncio
async def test_gh_adds_detected_repo_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_detect_repo() -> str:
        return "owner/repo"

    async def fake_shell_run(command: str, timeout: int = 30, working_dir: str | None = None) -> str:
        captured["command"] = command
        captured["timeout"] = timeout
        return "ok"

    monkeypatch.setattr(github, "_detect_repo", fake_detect_repo)
    monkeypatch.setattr(github, "shell_run", fake_shell_run)

    result = await github._gh("pr list")

    assert result == "ok"
    assert captured["command"] == "gh pr list --repo owner/repo"


@pytest.mark.asyncio
async def test_pr_review_rejects_invalid_action() -> None:
    result = await github.pr_review(12, "ship-it")

    assert "action must be" in result


@pytest.mark.asyncio
async def test_pr_diff_builds_capped_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_shell_run(command: str, timeout: int = 30, working_dir: str | None = None) -> str:
        captured["command"] = command
        captured["timeout"] = timeout
        return "diff"

    monkeypatch.setattr(github, "shell_run", fake_shell_run)

    result = await github.pr_diff(55, repo="owner/repo")

    assert result == "diff"
    assert captured["command"] == "gh pr diff 55 --repo owner/repo | head -300"


@pytest.mark.asyncio
async def test_pr_review_with_comments_requires_repo_when_detection_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_detect_repo():
        return None

    monkeypatch.setattr(github, "_detect_repo", fake_detect_repo)

    result = await github.pr_review_with_comments(
        99,
        "COMMENT",
        "summary",
        [{"path": "agent/main.py", "line": 10, "message": "note"}],
    )

    assert "repo is required" in result


@pytest.mark.asyncio
async def test_issue_create_adds_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_gh(args: str, repo: str | None = None, timeout: int = 30) -> str:
        captured["args"] = args
        return "created"

    monkeypatch.setattr(github, "_gh", fake_gh)

    result = await github.issue_create("Bug", "Fix it", labels=["bug", "urgent"])

    assert result == "created"
    assert "--label bug,urgent" in str(captured["args"])
