# Memory

_This file is maintained by the agent. It is rewritten periodically to reflect current knowledge, lessons, and preferences. The agent appends mistakes and insights here automatically._

## Setup

- First deployed: (agent updates this on first run)
- Operator contact: (agent updates with Discord username of owner)
- Workspace: `/workspace`
- Skills directory: `/app/agent/skills`

## Preferences

- Preferred shell: `/bin/bash`
- Always read a file before writing it
- Commit style: conventional commits (`type(scope): description`)
- Test Python files with `python -m py_compile <file>` before committing

## Known Environment Details

- Docker container name matches `AGENT_NAME` env var
- SSH key for GitHub is at `/tmp/agent_ssh/agent_ed25519` (mounted from host `.ssh/`)
- noVNC browser viewer available at port 6080
- Playwright MCP server at `http://browser:3080`

## Recent Lessons

_(Automatically populated by the agent after each task cycle)_
