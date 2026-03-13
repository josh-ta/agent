# Agent Platform

A fully autonomous, self-modifying AI agent that runs in Docker. Each agent has:

- Full shell + filesystem access
- Browser automation (Playwright + noVNC for human observation)
- Discord integration for agent-to-agent communication
- SQLite local memory + optional PostgreSQL for multi-agent coordination
- A skill system: agents learn and evolve by editing their own `.md` skill files
- Self-modification: the agent can edit its own skills, identity, and code

## Quick Start

### 1. Prerequisites

- Docker + Docker Compose
- A Discord bot token ([create one here](https://discord.com/developers/applications))
- An Anthropic API key

### 2. Install on a VPS / server

```bash
git clone <this-repo> agent && cd agent
bash scripts/install.sh
```

Or manually:

```bash
cp .env.example .env
# Edit .env with your keys
docker compose up -d
```

### 3. Watch the agent in action

Visit `http://<your-server>:6080` to see the live browser (noVNC).

Check logs:
```bash
docker compose logs -f agent
```

## Configuration

All configuration is in `.env`. Key variables:

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key (required) |
| `AGENT_NAME` | Unique name for this agent instance |
| `DISCORD_BOT_TOKEN` | Discord bot token |
| `DISCORD_AGENT_CHANNEL_ID` | Channel ID for this agent's private channel |
| `DISCORD_BUS_CHANNEL_ID` | Shared `#agent-bus` channel for all agents |
| `POSTGRES_URL` | Optional shared PostgreSQL for multi-agent coordination |

## Discord Channel Setup

Create the following channels in your Discord server:

- `#agent-bus` — broadcast channel all agents monitor
- `#agent-comms` — structured JSON A2A messages
- `#agent-<name>` — one per agent (private task/log channel)

Invite the bot to the server with scopes: `bot`, `applications.commands`  
Bot permissions: `Send Messages`, `Read Message History`, `View Channels`

## Skills

Skills live in `agent/skills/`. Each is a Markdown file describing a procedure the agent can follow. The agent can create and edit skills — changes persist across restarts since the directory is bind-mounted.

To add a skill, create a `.md` file in `agent/skills/` and restart the agent (or the agent will pick it up on its next heartbeat).

## Multi-Agent Setup

1. Clone this repo on each server/VPS
2. Each gets a unique `AGENT_NAME` and `DISCORD_AGENT_CHANNEL_ID`
3. All share the same `DISCORD_BUS_CHANNEL_ID` and `DISCORD_COMMS_CHANNEL_ID`
4. Optionally share a single `POSTGRES_URL` for coordination

Agents communicate by posting structured JSON to `#agent-comms`:
```json
{"from": "agent-1", "to": "agent-2", "task": "research X", "payload": "..."}
```

## Architecture

```
Docker Host
├── agent container        ← Python agent (Claude + tools)
│   ├── Shell tool         ← run any command
│   ├── Filesystem tool    ← read/write /workspace
│   ├── Browser tool       ← Playwright via MCP
│   ├── Discord tool       ← send/read messages
│   └── Self-edit tool     ← modify own skills/identity
├── browser container      ← Playwright MCP + noVNC :6080
└── postgres container     ← optional, profile: postgres
```

## Updating

```bash
bash scripts/update-agent.sh
```

## Self-modification

The agent can edit its own `agent/skills/` and `agent/identity/` files at runtime. For code-level changes, it commits to git and triggers a container restart via the Docker socket.
