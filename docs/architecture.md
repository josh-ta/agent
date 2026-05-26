# Agent Platform Architecture

## Overview

The agent is an autonomous Docker-deployed AI platform. Operators interact primarily through Discord; automation can submit work via a FastAPI control plane.

```text
Discord / HTTP API
        │
        ▼
  MessageHandlingService / AgentLoop
        │
        ▼
  Pydantic AI (fast / smart / best tiers)
        │
        ├── Native tools (shell, filesystem, GitHub, secrets, memory)
        ├── Browser MCP sidecar (Playwright)
        └── Optional MCP servers (MCP_SERVERS JSON)
        │
        ▼
  EventBridge → Discord status embed / SSE / metrics
```

## Runtime components

| Component | Role |
|-----------|------|
| `agent/main.py` | CLI entry; wires SQLite, permissions, agents, loop, Discord |
| `agent/loop.py` | Single-task queue, model routing, cancellation, restore |
| `agent/communication/` | Discord gateway, command handling, streaming presenter |
| `agent/control_plane/app.py` | HTTP task API + SSE when `CONTROL_PLANE_ENABLED=true` |
| `mcps/browser/` | Playwright MCP + noVNC sidecar |
| `agent/memory/sqlite_store.py` | Local persistence, transcripts, scheduled tasks |

## Discord model

1. Messages in the private agent channel become tasks.
2. Tool/shell/progress events stream to a **task thread** (when enabled) via a **single debounced status embed**.
3. Final answers reply in the private channel (with live preview while generating).
4. Native commands (`/status`, `/cancel`, `/queue`, …) and slash commands share one handler.

Configure:
- `DISCORD_USE_TASK_THREADS=true` (default)
- `RESTORE_PENDING_DISCORD_TASKS=true` (default)

## Control plane

- `POST /tasks` — enqueue work
- `GET /tasks/{id}` — task status
- `GET /events` — SSE stream
- `GET /healthz`, `/readyz`, `/metrics` — public probes

When `CONTROL_PLANE_API_KEY` is set, other routes require `Authorization: Bearer <key>`.

## Deployment (Hetzner)

See [README.md](../README.md) for the full wizard. Typical stack:

```bash
docker compose up -d          # agent + browser
docker compose --profile self-restart up -d   # if agent_restart needs docker.sock
```

Data volumes: `/data` (SQLite, secrets), `/workspace` (repos and artifacts).

## Extension points

- **Skills** — markdown procedures in `agent/skills/`
- **Identity** — `agent/identity/*.md`
- **MCP** — add sidecars and register URLs in `MCP_SERVERS`
- **Permissions** — `PERMISSION_MODE` + SQLite permission rules

TicketAction native integration is planned post-1.0.
