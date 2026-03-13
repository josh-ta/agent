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

## Discord Bot Setup

### 1. Create the bot

1. Go to [discord.com/developers/applications](https://discord.com/developers/applications) → **New Application** → give it your agent's name
2. Open **Bot** in the left sidebar → **Add Bot**
3. Under **Privileged Gateway Intents**, enable all three:
   - Presence Intent
   - Server Members Intent
   - **Message Content Intent** ← required to read messages
4. Click **Reset Token** and copy it — this is your `DISCORD_BOT_TOKEN`

### 2. Invite the bot to your server

Replace `YOUR_CLIENT_ID` with the value from the **General Information** page:

```
https://discord.com/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=8515702525261888&scope=bot
```

Open the URL in your browser, select your server, and authorize.

### 3. Create the required channels

In your Discord server, create:

| Channel | Purpose | How to talk to the agent |
|---|---|---|
| `#<agent-name>` (e.g. `#bob`) | Private channel for this agent | Send any message — agent always responds |
| `#agent-bus` | Broadcast visible to all agents | Must `@mention` the bot to get a response |
| `#agent-comms` | Structured agent-to-agent JSON | Post raw JSON `{"from":"you","to":"bob","task":"..."}` |

### 4. Get your IDs

In Discord, go to **User Settings → Advanced** and enable **Developer Mode**, then:

- Right-click your **server name** → Copy Server ID → `DISCORD_GUILD_ID`
- Right-click `#<agent-name>` → Copy Channel ID → `DISCORD_AGENT_CHANNEL_ID`
- Right-click `#agent-bus` → Copy Channel ID → `DISCORD_BUS_CHANNEL_ID`
- Right-click `#agent-comms` → Copy Channel ID → `DISCORD_COMMS_CHANNEL_ID`

### 5. Update .env and restart

```bash
nano .env
# Set DISCORD_BOT_TOKEN, DISCORD_GUILD_ID, and the four channel IDs
docker compose up -d
```

The agent will post an online announcement to `#agent-bus` on startup to confirm it connected.

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
