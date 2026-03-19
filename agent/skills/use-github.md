# Use GitHub

Use this skill when working with GitHub repositories: cloning, branching, committing, opening PRs, reviewing PRs, managing issues, or checking CI.

## Tools Available

Dedicated structured tools (use these — they're more reliable than raw shell):

| Tool | What it does |
|---|---|
| `gh_pr_view(pr, repo?)` | Read PR: title, body, status, checks, all comments |
| `gh_pr_list(repo?, state?, limit?)` | List PRs (state: open/closed/merged/all) |
| `gh_pr_diff(pr, repo?)` | Get the PR diff (capped at 300 lines) |
| `gh_pr_comment(pr, body, repo?)` | Post a general comment on a PR |
| `gh_pr_review(pr, action, body?, repo?)` | Submit a review: action = approve / request-changes / comment |
| `gh_pr_review_inline(pr, action, body, comments, repo?)` | Review with inline line comments: comments = [{path, line, message}] |
| `gh_pr_checks(pr, repo?)` | Show CI check results for a PR |
| `gh_pr_merge(pr, method?, repo?)` | Merge a PR (method: merge/squash/rebase) |
| `gh_issue_view(issue, repo?)` | Read an issue and all comments |
| `gh_issue_list(repo?, state?, limit?)` | List issues |
| `gh_issue_comment(issue, body, repo?)` | Comment on an issue |
| `gh_issue_create(title, body, labels?, repo?)` | Create a new issue |
| `gh_issue_close(issue, reason?, repo?)` | Close an issue |
| `gh_ci_list(repo?, branch?, limit?)` | List recent CI runs |
| `gh_ci_view(run_id, repo?)` | Show CI run job status |
| `gh_ci_logs_failed(run_id, repo?)` | Fetch failed step logs (capped at 200 lines) |
| `gh_ci_rerun(run_id, failed_only?, repo?)` | Re-trigger a CI run |

The `repo` argument is `owner/repo` (e.g. `josh-ta/TicketActionApp`). Omit it when running from inside a cloned repo directory — `gh` infers it automatically.
Never guess `owner/repo` from memory or conversation context. If you cannot verify the slug from the checked-out repo or with `gh repo view`, ask the user.

## Workflows

### Review a PR and request changes
```
gh_pr_view(1, "owner/repo")          # read the PR
gh_pr_diff(1, "owner/repo")          # read the diff
gh_pr_review(1, "request-changes",
    body="Please fix X before merging:\n- Line 42: ...\n- Missing tests for Y",
    repo="owner/repo")
```

### Review with inline comments
```
gh_pr_review_inline(
    pr=1,
    action="REQUEST_CHANGES",
    body="A few things to address:",
    comments=[
        {"path": "src/api/main.py", "line": 42, "message": "This will throw if `user` is None"},
        {"path": "src/api/main.py", "line": 87, "message": "Missing error handling for 404"},
    ],
    repo="owner/repo"
)
```

### Approve a PR
```
gh_pr_review(1, "approve", body="LGTM — tests pass and logic looks solid.", repo="owner/repo")
```

### Investigate and fix failing CI
```
gh_ci_list(repo="owner/repo", branch="fix/my-branch")   # get run ID
gh_ci_view("23057002906", "owner/repo")                 # see which jobs failed
gh_ci_logs_failed("23057002906", "owner/repo")          # read the error output
# ... fix the code ...
gh_ci_rerun("23057002906", failed_only=True, repo="owner/repo")
```

### Comment on an issue
```
gh_issue_view(42, "owner/repo")
gh_issue_comment(42, "I looked into this — the root cause is X. PR incoming.", "owner/repo")
```

## Shell fallback
For anything not covered above, use `run_shell` with `gh` or `git` directly. Discover the real repo root first unless you just cloned the repo yourself:
```
run_shell('ROOT="$(git rev-parse --show-toplevel)" && cd "$ROOT" && gh pr create --title "..." --body "..."')
run_shell('ROOT="$(git rev-parse --show-toplevel)" && cd "$ROOT" && git log --oneline -10')
```

## Auth Check
```
run_shell("gh auth status")
```

## Tips
- Always `cd` into the repo dir before raw git commands.
- Prefer the existing checkout in `WORKSPACE_PATH` when present.
- Discover the repo root with `git rev-parse --show-toplevel` before repo-specific commands.
- Only use `/workspace/<repo-name>` immediately after cloning that repo yourself.
- Set git identity once per clone: `git config user.name "bob-agent" && git config user.email "bob@agent.local"`
- `gh` CLI is pre-authenticated via `GH_TOKEN` — never run `gh auth login`.
- Follow conventional commits: `type(scope): description`
- After opening a PR or resolving a significant issue, call `memory_save` with the PR URL and outcome.
