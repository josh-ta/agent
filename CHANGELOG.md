# Changelog

## 1.0.0 — 2026-05-26

First production-ready release focused on Discord reliability, core agent capabilities, and deployment hardening.

### Discord
- Refactored Discord layer into presenter, commands, and session modules
- Debounced status embed replaces per-tool message spam
- Streams final answers via `TextDeltaEvent` preview messages
- Optional per-task threads in the private channel (`DISCORD_USE_TASK_THREADS`, default on)
- Clearer cancel UX with `/force-cancel` and accurate cancellation feedback
- **`/config` command** — change models and safe settings from Discord without redeploying
- Restores pending Discord tasks on restart by default
- Retries Discord sends on rate limits; surfaces delivery failures

### Agent capabilities
- `web_search` tool (Tavily or Brave via env)
- `http_request` tool with `HTTP_ALLOWED_HOSTS`
- `apply_patch` for structured file edits
- `MCP_SERVERS` JSON env for additional MCP sidecars

### Security & ops
- Control plane API key auth (`CONTROL_PLANE_API_KEY`)
- Non-root agent process in Docker (`gosu agent`)
- Docker socket mount moved to optional `self-restart` compose profile
- CI lint workflow (ruff + mypy)
- Architecture documentation

## 0.8.4

Prior stable release.
