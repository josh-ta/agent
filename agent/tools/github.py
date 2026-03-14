"""
GitHub tools: structured PR review, comments, issues, and CI operations.

All operations use the `gh` CLI (pre-authenticated via GH_TOKEN).
These wrap common gh invocations with clean return values so the agent
doesn't have to remember exact flag syntax or parse raw CLI output.
"""

from __future__ import annotations

import json

from agent.tools.shell import shell_run

_detected_repo: str | None = None  # cached within a process run


async def _detect_repo() -> str | None:
    """
    Auto-detect the GitHub repo slug (owner/repo) from the workspace git remote.
    Returns None if detection fails. Result is cached after first successful call.
    """
    global _detected_repo
    if _detected_repo:
        return _detected_repo
    from agent.config import settings
    try:
        result = await shell_run(
            "gh repo view --json nameWithOwner -q .nameWithOwner",
            working_dir=str(settings.workspace_path),
            timeout=10,
        )
        slug = result.strip()
        if slug and "/" in slug and not slug.startswith("["):
            _detected_repo = slug
            return slug
    except Exception:
        pass
    return None


async def _gh(args: str, repo: str | None = None, timeout: int = 30) -> str:
    """Run a gh command, optionally scoped to a repo. Auto-detects repo if not given."""
    if not repo:
        repo = await _detect_repo()
    repo_flag = f" --repo {repo}" if repo else ""
    return await shell_run(f"gh {args}{repo_flag}", timeout=timeout)


# ── Pull requests ─────────────────────────────────────────────────────────────

async def pr_view(pr: int | str, repo: str | None = None) -> str:
    """Return title, body, status, checks, and recent comments for a PR."""
    return await _gh(f"pr view {pr} --comments", repo)


async def pr_list(repo: str | None = None, state: str = "open", limit: int = 20) -> str:
    """List PRs. state: open | closed | merged | all."""
    return await _gh(f"pr list --state {state} --limit {limit}", repo)


async def pr_diff(pr: int | str, repo: str | None = None) -> str:
    """Return the full diff for a PR (piped through head to cap size)."""
    repo_flag = f" --repo {repo}" if repo else ""
    # Cap diff at ~300 lines to keep context manageable
    return await shell_run(
        f"gh pr diff {pr}{repo_flag} | head -300",
        timeout=30,
    )


async def pr_comment(pr: int | str, body: str, repo: str | None = None) -> str:
    """Post a general (non-review) comment on a PR."""
    safe_body = body.replace("'", "'\\''")
    return await _gh(f"pr comment {pr} --body '{safe_body}'", repo)


async def pr_review(
    pr: int | str,
    action: str,
    body: str = "",
    repo: str | None = None,
) -> str:
    """
    Submit a PR review.

    action: 'approve' | 'request-changes' | 'comment'
    body:   Review summary comment (required for request-changes, optional otherwise).
    """
    if action not in ("approve", "request-changes", "comment"):
        return f"[error] action must be 'approve', 'request-changes', or 'comment' — got '{action}'"
    safe_body = body.replace("'", "'\\''")
    body_flag = f" --body '{safe_body}'" if body else ""
    return await _gh(f"pr review {pr} --{action}{body_flag}", repo)


async def pr_review_with_comments(
    pr: int | str,
    action: str,
    body: str,
    comments: list[dict],
    repo: str | None = None,
) -> str:
    """
    Submit a PR review with inline line comments via the GitHub REST API.

    comments: list of dicts with keys:
        path     — file path relative to repo root (e.g. 'src/main.py')
        line     — line number in the diff to comment on
        message  — the comment text

    action: 'APPROVE' | 'REQUEST_CHANGES' | 'COMMENT'
    body:   top-level review summary

    Uses `gh api` so the agent doesn't need to manage auth headers.
    The repo slug (owner/repo) is required for the API path.
    """
    if action not in ("APPROVE", "REQUEST_CHANGES", "COMMENT"):
        return "[error] action must be 'APPROVE', 'REQUEST_CHANGES', or 'COMMENT'"

    if not repo:
        repo = await _detect_repo()
        if not repo:
            return "[error] repo is required for inline review comments (e.g. 'owner/repo')"

    review_comments = [
        {"path": c["path"], "line": c["line"], "body": c["message"]}
        for c in comments
    ]
    payload = json.dumps({
        "body": body,
        "event": action,
        "comments": review_comments,
    })

    # Write payload to a temp file to avoid shell quoting issues entirely
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        tf.write(payload)
        tmp_path = tf.name

    try:
        cmd = (
            f"gh api repos/{repo}/pulls/{pr}/reviews "
            f"--method POST "
            f"--input {tmp_path}"
        )
        return await shell_run(cmd, timeout=30)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def pr_checks(pr: int | str, repo: str | None = None) -> str:
    """Show CI check status for a PR."""
    return await _gh(f"pr checks {pr}", repo)


async def pr_merge(
    pr: int | str,
    method: str = "squash",
    repo: str | None = None,
) -> str:
    """
    Merge a PR.
    method: 'merge' | 'squash' | 'rebase'
    """
    if method not in ("merge", "squash", "rebase"):
        return f"[error] method must be 'merge', 'squash', or 'rebase' — got '{method}'"
    return await _gh(f"pr merge {pr} --{method} --auto", repo)


# ── Issues ────────────────────────────────────────────────────────────────────

async def issue_view(issue: int | str, repo: str | None = None) -> str:
    """Return full issue body and comments."""
    return await _gh(f"issue view {issue} --comments", repo)


async def issue_list(repo: str | None = None, state: str = "open", limit: int = 20) -> str:
    """List issues. state: open | closed | all."""
    return await _gh(f"issue list --state {state} --limit {limit}", repo)


async def issue_comment(issue: int | str, body: str, repo: str | None = None) -> str:
    """Post a comment on an issue."""
    safe_body = body.replace("'", "'\\''")
    return await _gh(f"issue comment {issue} --body '{safe_body}'", repo)


async def issue_create(
    title: str,
    body: str,
    labels: list[str] | None = None,
    repo: str | None = None,
) -> str:
    """Create a new issue."""
    safe_title = title.replace("'", "'\\''")
    safe_body = body.replace("'", "'\\''")
    labels_flag = f" --label {','.join(labels)}" if labels else ""
    return await _gh(
        f"issue create --title '{safe_title}' --body '{safe_body}'{labels_flag}",
        repo,
    )


async def issue_close(issue: int | str, reason: str = "completed", repo: str | None = None) -> str:
    """Close an issue. reason: 'completed' | 'not planned'."""
    return await _gh(f"issue close {issue} --reason '{reason}'", repo)


# ── CI / Workflow runs ────────────────────────────────────────────────────────

async def ci_list(repo: str | None = None, branch: str | None = None, limit: int = 5) -> str:
    """List recent CI workflow runs."""
    branch_flag = f" --branch {branch}" if branch else ""
    return await _gh(f"run list{branch_flag} --limit {limit}", repo)


async def ci_view(run_id: str | int, repo: str | None = None) -> str:
    """Show job status for a CI run."""
    return await _gh(f"run view {run_id}", repo)


async def ci_logs_failed(run_id: str | int, repo: str | None = None) -> str:
    """Fetch only the failed step logs from a CI run (capped at 4KB)."""
    repo_flag = f" --repo {repo}" if repo else ""
    return await shell_run(
        f"gh run view {run_id}{repo_flag} --log-failed 2>&1 | head -200",
        timeout=60,
    )


async def ci_rerun(run_id: str | int, failed_only: bool = True, repo: str | None = None) -> str:
    """Re-trigger a CI run (or only its failed jobs)."""
    failed_flag = " --failed" if failed_only else ""
    return await _gh(f"run rerun {run_id}{failed_flag}", repo)
