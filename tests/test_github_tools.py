from __future__ import annotations

import asyncio
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


def test_looks_like_repo_slug_validates_expected_shapes() -> None:
    assert github._looks_like_repo_slug("owner/repo") is True
    assert github._looks_like_repo_slug("owner/repo.with-dots") is True
    assert github._looks_like_repo_slug("not a slug") is False


def test_split_shell_result_parses_exit_code() -> None:
    body, exit_code = github._split_shell_result("owner/repo\n[exit code: 0]")

    assert body == "owner/repo"
    assert exit_code == 0


def test_split_shell_result_without_marker_returns_none_exit_code() -> None:
    body, exit_code = github._split_shell_result("plain output")

    assert body == "plain output"
    assert exit_code is None


def test_split_shell_result_invalid_exit_code_returns_none() -> None:
    body, exit_code = github._split_shell_result("owner/repo\n[exit code: nope]")

    assert body == "owner/repo"
    assert exit_code is None


@pytest.mark.asyncio
async def test_detect_repo_strips_shell_exit_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    github._detected_repo = None

    async def fake_shell_run(command: str, timeout: int = 30, working_dir: str | None = None) -> str:
        return "owner/repo\n[exit code: 0]"

    monkeypatch.setattr(github, "shell_run", fake_shell_run)

    result = await github._detect_repo()

    assert result == "owner/repo"


@pytest.mark.asyncio
async def test_detect_repo_returns_cached_none_and_handles_shell_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    github._detected_repo = None

    async def fake_shell_run(command: str, timeout: int = 30, working_dir: str | None = None) -> str:
        return "not-a-slug\n[exit code: 1]"

    monkeypatch.setattr(github, "shell_run", fake_shell_run)
    assert await github._detect_repo() is None

    async def fake_raise(command: str, timeout: int = 30, working_dir: str | None = None) -> str:
        raise RuntimeError("gh missing")

    monkeypatch.setattr(github, "shell_run", fake_raise)
    assert await github._detect_repo() is None


@pytest.mark.asyncio
async def test_detect_repo_returns_cached_slug_without_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    github._detected_repo = "owner/repo"

    async def fail_shell_run(command: str, timeout: int = 30, working_dir: str | None = None) -> str:
        raise AssertionError("shell_run should not be called for cached repo")

    monkeypatch.setattr(github, "shell_run", fail_shell_run)

    assert await github._detect_repo() == "owner/repo"


@pytest.mark.asyncio
async def test_gh_rejects_invalid_repo_slug() -> None:
    result = await github._gh("pr list", repo="not a slug")

    assert "invalid repo" in result


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

    async def fake_verify_repo(repo: str) -> str | None:
        return repo

    monkeypatch.setattr(github, "shell_run", fake_shell_run)
    monkeypatch.setattr(github, "_verify_repo", fake_verify_repo)

    result = await github.pr_diff(55, repo="owner/repo")

    assert result == "diff"
    assert captured["command"] == "gh pr diff 55 --repo owner/repo | head -300"


@pytest.mark.asyncio
async def test_pr_diff_returns_resolve_repo_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(github, "_resolve_repo", lambda repo=None: asyncio.sleep(0, result=(None, "bad repo")))

    assert await github.pr_diff(55, repo="owner/repo") == "bad repo"


@pytest.mark.asyncio
async def test_gh_rejects_unverified_explicit_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_verify_repo(repo: str) -> str | None:
        return None

    async def fake_detect_repo() -> str | None:
        return "actual/repo"

    monkeypatch.setattr(github, "_verify_repo", fake_verify_repo)
    monkeypatch.setattr(github, "_detect_repo", fake_detect_repo)

    result = await github._gh("pr checks 1", repo="made/up")

    assert "could not verify repo 'made/up'" in result
    assert "actual/repo" in result


@pytest.mark.asyncio
async def test_gh_accepts_verified_explicit_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_verify_repo(repo: str) -> str | None:
        return "owner/repo"

    async def fake_shell_run(command: str, timeout: int = 30, working_dir: str | None = None) -> str:
        captured["command"] = command
        return "ok"

    monkeypatch.setattr(github, "_verify_repo", fake_verify_repo)
    monkeypatch.setattr(github, "shell_run", fake_shell_run)

    result = await github._gh("pr view 1", repo="owner/repo")

    assert result == "ok"
    assert captured["command"] == "gh pr view 1 --repo owner/repo"


@pytest.mark.asyncio
async def test_resolve_repo_without_explicit_value_uses_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_detect_repo() -> str | None:
        return "owner/repo"

    monkeypatch.setattr(github, "_detect_repo", fake_detect_repo)

    assert await github._resolve_repo(None) == ("owner/repo", None)


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
async def test_pr_review_with_comments_rejects_invalid_action() -> None:
    result = await github.pr_review_with_comments(99, "BAD", "summary", [], repo="owner/repo")

    assert "action must be" in result


@pytest.mark.asyncio
async def test_pr_view_and_pr_list_delegate_to_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str | None]] = []

    async def fake_gh(args: str, repo: str | None = None, timeout: int = 30) -> str:
        seen.append((args, repo))
        return "ok"

    monkeypatch.setattr(github, "_gh", fake_gh)

    assert await github.pr_view(12, repo="owner/repo") == "ok"
    assert await github.pr_list(repo="owner/repo", state="closed", limit=7) == "ok"
    assert seen == [
        ("pr view 12 --comments", "owner/repo"),
        ("pr list --state closed --limit 7", "owner/repo"),
    ]


@pytest.mark.asyncio
async def test_pr_review_valid_action_builds_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    async def fake_gh(args: str, repo: str | None = None, timeout: int = 30) -> str:
        captured.append(args)
        return "reviewed"

    monkeypatch.setattr(github, "_gh", fake_gh)

    assert await github.pr_review(12, "approve", repo="owner/repo") == "reviewed"
    assert await github.pr_review(12, "comment", "it's fine", repo="owner/repo") == "reviewed"
    assert captured == [
        "pr review 12 --approve",
        "pr review 12 --comment --body 'it'\\''s fine'",
    ]


@pytest.mark.asyncio
async def test_pr_review_with_comments_builds_payload_and_tolerates_unlink_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_shell_run(command: str, timeout: int = 30, working_dir: str | None = None) -> str:
        captured["command"] = command
        return "reviewed"

    monkeypatch.setattr(github, "_resolve_repo", lambda repo=None: asyncio.sleep(0, result=("owner/repo", None)))
    monkeypatch.setattr(github, "shell_run", fake_shell_run)

    import os

    def _broken_unlink(path: str) -> None:
        raise OSError("busy")

    monkeypatch.setattr(os, "unlink", _broken_unlink)

    result = await github.pr_review_with_comments(
        12,
        "COMMENT",
        "summary",
        [{"path": "agent/main.py", "line": 10, "message": "note"}],
        repo="owner/repo",
    )

    assert result == "reviewed"
    assert "repos/owner/repo/pulls/12/reviews" in str(captured["command"])


@pytest.mark.asyncio
async def test_pr_review_with_comments_returns_resolve_repo_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(github, "_resolve_repo", lambda repo=None: asyncio.sleep(0, result=(None, "bad repo")))

    assert await github.pr_review_with_comments(12, "COMMENT", "summary", [], repo="owner/repo") == "bad repo"


@pytest.mark.asyncio
async def test_verify_repo_checks_explicit_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    github._verified_repos.clear()
    captured: dict[str, object] = {}

    async def fake_shell_run(command: str, timeout: int = 30, working_dir: str | None = None) -> str:
        captured["command"] = command
        return "owner/repo\n[exit code: 0]"

    monkeypatch.setattr(github, "shell_run", fake_shell_run)

    result = await github._verify_repo("owner/repo")

    assert result == "owner/repo"
    assert captured["command"] == "gh repo view owner/repo --json nameWithOwner -q .nameWithOwner"


@pytest.mark.asyncio
async def test_verify_repo_returns_none_when_shell_output_is_not_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    github._verified_repos.clear()

    async def fake_shell_run(command: str, timeout: int = 30, working_dir: str | None = None) -> str:
        return "not-a-slug\n[exit code: 0]"

    monkeypatch.setattr(github, "shell_run", fake_shell_run)

    assert await github._verify_repo("owner/repo") is None


@pytest.mark.asyncio
async def test_verify_repo_uses_cache_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    github._verified_repos.clear()
    github._verified_repos["owner/repo"] = "owner/repo"

    async def fake_shell_run(command: str, timeout: int = 30, working_dir: str | None = None) -> str:
        raise AssertionError("shell_run should not be called for cached repo")

    monkeypatch.setattr(github, "shell_run", fake_shell_run)

    assert await github._verify_repo("owner/repo") == "owner/repo"


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


@pytest.mark.asyncio
async def test_issue_create_escapes_title_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_gh(args: str, repo: str | None = None, timeout: int = 30) -> str:
        captured["args"] = args
        return "created"

    monkeypatch.setattr(github, "_gh", fake_gh)

    result = await github.issue_create("it's broken", "can't deploy")

    assert result == "created"
    assert "it'\\''s broken" in str(captured["args"])
    assert "can'\\''t deploy" in str(captured["args"])


@pytest.mark.asyncio
async def test_pr_comment_escapes_single_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_gh(args: str, repo: str | None = None, timeout: int = 30) -> str:
        captured["args"] = args
        return "commented"

    monkeypatch.setattr(github, "_gh", fake_gh)

    result = await github.pr_comment(7, "it's fixed")

    assert result == "commented"
    assert "it'\\''s fixed" in str(captured["args"])


@pytest.mark.asyncio
async def test_pr_merge_rejects_invalid_method() -> None:
    result = await github.pr_merge(7, method="fast-forward")

    assert "method must be" in result


@pytest.mark.asyncio
async def test_pr_merge_valid_method_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_gh(args: str, repo: str | None = None, timeout: int = 30) -> str:
        captured["args"] = args
        return "merged"

    monkeypatch.setattr(github, "_gh", fake_gh)

    result = await github.pr_merge(7, method="squash", repo="owner/repo")

    assert result == "merged"
    assert captured["args"] == "pr merge 7 --squash --auto"


@pytest.mark.asyncio
async def test_ci_list_adds_branch_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_gh(args: str, repo: str | None = None, timeout: int = 30) -> str:
        captured["args"] = args
        return "listed"

    monkeypatch.setattr(github, "_gh", fake_gh)

    result = await github.ci_list(branch="main", limit=10)

    assert result == "listed"
    assert captured["args"] == "run list --branch main --limit 10"


@pytest.mark.asyncio
async def test_wrapper_commands_delegate_to_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    async def fake_gh(args: str, repo: str | None = None, timeout: int = 30) -> str:
        seen.append(args)
        return "ok"

    monkeypatch.setattr(github, "_gh", fake_gh)

    assert await github.pr_checks(5) == "ok"
    assert await github.issue_view(6) == "ok"
    assert await github.issue_list(state="closed", limit=7) == "ok"
    assert await github.issue_comment(8, "done") == "ok"
    assert await github.issue_close(9, reason="not planned") == "ok"
    assert await github.ci_view(10) == "ok"

    assert seen == [
        "pr checks 5",
        "issue view 6 --comments",
        "issue list --state closed --limit 7",
        "issue comment 8 --body 'done'",
        "issue close 9 --reason 'not planned'",
        "run view 10",
    ]


@pytest.mark.asyncio
async def test_ci_rerun_uses_failed_flag_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_gh(args: str, repo: str | None = None, timeout: int = 30) -> str:
        captured["args"] = args
        return "rerun"

    monkeypatch.setattr(github, "_gh", fake_gh)

    result = await github.ci_rerun(123)

    assert result == "rerun"
    assert captured["args"] == "run rerun 123 --failed"


@pytest.mark.asyncio
async def test_ci_rerun_omits_failed_flag_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_gh(args: str, repo: str | None = None, timeout: int = 30) -> str:
        captured["args"] = args
        return "rerun"

    monkeypatch.setattr(github, "_gh", fake_gh)

    result = await github.ci_rerun(123, failed_only=False)

    assert result == "rerun"
    assert captured["args"] == "run rerun 123"


@pytest.mark.asyncio
async def test_ci_logs_failed_builds_repo_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_verify_repo(repo: str) -> str | None:
        return repo

    async def fake_shell_run(command: str, timeout: int = 30, working_dir: str | None = None) -> str:
        captured["command"] = command
        captured["timeout"] = timeout
        return "logs"

    monkeypatch.setattr(github, "_verify_repo", fake_verify_repo)
    monkeypatch.setattr(github, "shell_run", fake_shell_run)

    result = await github.ci_logs_failed(123, repo="owner/repo")

    assert result == "logs"
    assert captured["command"] == "gh run view 123 --repo owner/repo --log-failed 2>&1 | head -200"
    assert captured["timeout"] == 60


@pytest.mark.asyncio
async def test_ci_logs_failed_returns_resolve_repo_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(github, "_resolve_repo", lambda repo=None: asyncio.sleep(0, result=(None, "bad repo")))

    assert await github.ci_logs_failed(123, repo="owner/repo") == "bad repo"
