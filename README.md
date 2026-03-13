# Agent Platform

A fully autonomous, self-modifying AI agent that runs in Docker. Give it a name, connect it to Discord, and chat with it. It can browse the web, run shell commands, write code, manage files, and improve itself over time.

---

## Deploy in 5 steps

### 1. Create a VPS on Hetzner

1. Go to [hetzner.com/cloud](https://www.hetzner.com/cloud) → **New Project** → **Add Server**
2. Choose:
   - **Location**: closest to you
   - **Image**: Ubuntu 24.04
   - **Type**: CX22 (2 vCPU, 4 GB RAM) or larger
   - **SSH key**: add yours so you can log in
3. Click **Create & Buy** — note the server's IP address

```bash
ssh root@<your-server-ip>
```

---

### 2. Create a Discord bot

1. Go to [discord.com/developers/applications](https://discord.com/developers/applications) → **New Application** → name it after your agent (e.g. `bob`)
2. Open **Bot** → enable all three **Privileged Gateway Intents** (especially **Message Content Intent**)
3. Click **Reset Token** → copy it (your `DISCORD_BOT_TOKEN`)
4. Copy your **Application ID** from General Information (your `CLIENT_ID`)
5. Invite the bot to your server by opening this URL in a browser:
   ```
   https://discord.com/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=8515702525261888&scope=bot
   ```

**Create these channels in your Discord server:**

| Channel | Purpose |
|---|---|
| `#<agent-name>` (e.g. `#bob`) | Chat directly with the agent |
| `#agent-bus` | Broadcast channel — `@mention` the bot to get a response |
| `#agent-comms` | Structured JSON messages between agents |

**Get channel and server IDs:** In Discord → User Settings → Advanced → enable **Developer Mode**, then right-click any channel or server name → **Copy ID**.

---

### 3. Install

On your server:

```bash
git clone https://github.com/josh-ta/agent.git bob && cd bob
bash scripts/install.sh
```

The wizard will ask for:
- Agent name
- LLM provider (Anthropic or OpenAI) and API key
- Discord bot token and channel IDs

Everything is saved to `.env`. The wizard then builds and starts the containers automatically.

---

### 4. Talk to your agent

Send a message in your agent's private Discord channel (e.g. `#bob`). The agent responds to every message there. In `#agent-bus` you must `@mention` it.

Watch it work live at `http://<your-server-ip>:6080` (browser via noVNC).

---

### 5. Staying up to date

If you want to pull template updates:

```bash
git remote add template https://github.com/josh-ta/agent.git
git fetch template
git merge template/main --allow-unrelated-histories  # first time only
# subsequent updates:
git fetch template && git merge template/main
docker compose up --build -d
```

---

## Configuration

All settings live in `.env`. Re-run the wizard any time:

```bash
bash scripts/install.sh
```

Key variables:

| Variable | Description |
|---|---|
| `AGENT_NAME` | Agent's name — used as Docker container name and identity |
| `ANTHROPIC_API_KEY` | Claude API key |
| `OPENAI_API_KEY` | OpenAI API key (alternative) |
| `AGENT_MODEL` | Model string, e.g. `claude-sonnet-4-5` or `gpt-4o` |
| `DISCORD_BOT_TOKEN` | Bot token from Discord Developer Portal |
| `DISCORD_GUILD_ID` | Your Discord server ID |
| `DISCORD_AGENT_CHANNEL_ID` | This agent's private channel ID |
| `DISCORD_BUS_CHANNEL_ID` | `#agent-bus` channel ID |
| `DISCORD_COMMS_CHANNEL_ID` | `#agent-comms` channel ID |

---

## Useful commands

```bash
docker compose logs -f agent        # live agent logs
docker compose logs -f browser      # browser sidecar logs
docker compose down                 # stop everything
docker compose up --build -d        # rebuild and restart
```

---

## Multi-agent setup

Each agent runs on its own server. They share the same Discord server.

1. Clone and run `install.sh` on each server — give each a unique `AGENT_NAME`
2. All agents share the same `DISCORD_BUS_CHANNEL_ID` and `DISCORD_COMMS_CHANNEL_ID`
3. Each gets its own private Discord channel and `DISCORD_AGENT_CHANNEL_ID`

Agents talk to each other by posting JSON in `#agent-comms`:
```json
{"from": "bob", "to": "alice", "task": "research X and report back"}
```

---

## Architecture

```
VPS (Docker host)
├── agent container        ← Python + Claude (or GPT-4o)
│   ├── shell tool         ← run commands
│   ├── filesystem tool    ← read/write /workspace
│   ├── browser tool       ← Playwright via MCP sidecar
│   ├── discord tool       ← send/read Discord messages
│   └── self-edit tool     ← update own skills and identity
└── browser container      ← Playwright + noVNC (view at :6080)
```

Skills live in `agent/skills/` as Markdown files. The agent can create and edit them — changes persist across restarts because the directory is bind-mounted from the host.
