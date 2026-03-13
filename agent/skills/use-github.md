# Use GitHub

Use this skill when working with GitHub repositories: cloning, branching, committing, opening PRs, managing issues, or querying the API.

## Tools Available

- **`gh` CLI** — full GitHub CLI, authenticated via `GH_TOKEN` at container startup
- **`git`** — authenticated for HTTPS via `GITHUB_TOKEN` (no SSH key needed)
- **`run_shell`** — all git/gh commands run through the shell tool

## Common Operations

### Clone a repository
```bash
run_shell("gh repo clone owner/repo /workspace/repo")
# or with HTTPS:
run_shell("git clone https://github.com/owner/repo /workspace/repo")
```

### Create a branch and commit
```bash
run_shell("cd /workspace/repo && git checkout -b feature/my-change")
# ... make changes with write_file ...
run_shell("cd /workspace/repo && git add -A && git commit -m 'feat: description'")
```

### Open a Pull Request
```bash
run_shell("cd /workspace/repo && git push origin feature/my-change")
run_shell('cd /workspace/repo && gh pr create --title "My change" --body "Description of what and why"')
```

### List / read issues
```bash
run_shell("gh issue list --repo owner/repo --limit 20")
run_shell("gh issue view 42 --repo owner/repo")
```

### Create an issue
```bash
run_shell('gh issue create --repo owner/repo --title "Bug: X" --body "Steps to reproduce..."')
```

### Read PR comments / reviews
```bash
run_shell("gh pr view 123 --repo owner/repo --comments")
```

### Check workflow / CI status
```bash
run_shell("gh run list --repo owner/repo --limit 5")
run_shell("gh run view <run-id> --log-failed")
```

### Search code on GitHub
```bash
run_shell("gh search code 'function parseConfig' --repo owner/repo")
```

## Auth Check

If you're unsure whether auth is working:
```bash
run_shell("gh auth status")
run_shell("git config --global --list | grep user")
```

## Tips

- Always `cd` into the repo directory before running git commands.
- Use `gh repo fork` to fork before contributing to repos you don't own.
- For large file changes, write them with `write_file` then `git add` — don't try to echo large content in shell.
- Commit messages should follow conventional commits: `type(scope): description`
- After opening a PR, save the PR URL to memory with `memory_save`.
