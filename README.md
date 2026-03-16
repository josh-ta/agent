# Agent Platform

A fully autonomous, self-modifying AI agent that runs in Docker. Give it a name, connect it to Discord, and chat with it. It can browse the web, run shell commands, write code, manage files, and improve itself over time.

Supported model providers include Anthropic, OpenAI, Google, Groq, xAI, and Mistral. OpenAI-compatible endpoints such as Ollama or vLLM can also be used via `OPENAI_BASE_URL`.

---

## Full Setup Guide

### Step 1 — Create a VPS on Hetzner

1. Go to [hetzner.com/cloud](https://www.hetzner.com/cloud) → **New Project** → **Add Server**
2. Choose:
   - **Location**: closest to you
   - **Image**: Ubuntu 24.04
   - **Type**: CX22 (2 vCPU, 4 GB RAM) minimum — CX32 recommended
   - **SSH key**: paste your public key so you can log in without a password
3. Click **Create & Buy** — note the server IP

SSH in:
```bash
ssh root@<your-server-ip>
```

---

### Step 2 — Set up GitHub SSH on the server

Generate an SSH key on the server and add it to your GitHub account so you can clone private repos and push commits:

```bash
ssh-keygen -t ed25519 -C "root@<your-server-ip>" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
```

Copy the output, then go to [github.com/settings/ssh/new](https://github.com/settings/ssh/new), paste it, and save.

Test it:
```bash
ssh -T git@github.com
# Hi <username>! You've successfully authenticated...
```

---

### Step 3 — Create a repo from the template

1. Go to [github.com/josh-ta/agent](https://github.com/josh-ta/agent)
2. Click **Use this template** → **Create a new repository**
3. Name the repo after your agent (e.g. `bob`) — keep it public or private, your choice
4. Click **Create repository**

---

### Step 4 — Clone onto the server

Replace `<your-github-username>` and `<agent-name>` (e.g. `bob`):

```bash
git clone git@github.com:<your-github-username>/<agent-name>.git <agent-name>
cd <agent-name>
```

---

### Step 5 — Create the Discord bot

Do this before running the wizard — you'll need the token and IDs ready.

#### 5a. Create the application

1. Go to [discord.com/developers/applications](https://discord.com/developers/applications)
2. Click **New Application** → name it after your agent (e.g. `bob`)
3. Go to **General Information** → copy the **Application ID** (you'll need it for the invite URL)

#### 5b. Configure the bot

1. Click **Bot** in the left sidebar
2. Under **Privileged Gateway Intents**, enable all three:
   - **Presence Intent**
   - **Server Members Intent**
   - **Message Content Intent** ← required to read messages
3. Click **Reset Token** → copy the token (this is your `DISCORD_BOT_TOKEN`)

#### 5c. Build the OAuth invite link

Replace `YOUR_APPLICATION_ID`:

```
https://discord.com/oauth2/authorize?client_id=YOUR_APPLICATION_ID&permissions=8515702525261888&scope=bot
```

#### 5d. Create channels in your Discord server

Create these three channels:

| Channel | Purpose |
|---|---|
| `#<agent-name>` (e.g. `#bob`) | Private chat — agent responds to every message |
| `#agent-bus` | Broadcast — must `@mention` the bot to get a response |
| `#agent-comms` | Structured JSON agent-to-agent messages |

#### 5e. Invite the bot

Open the OAuth URL from step 5c in your browser. Select your server and authorize. The bot will appear in your member list.

To restrict the bot to only its private channel: after inviting, go to your server settings and configure channel permissions so the bot role only has access to `#<agent-name>`, `#agent-bus`, and `#agent-comms`.

#### 5f. Collect the IDs

Enable Developer Mode: Discord → **User Settings → Advanced → Developer Mode**.

Then right-click to copy:

| What to copy | Where to right-click |
|---|---|
| Server ID | The server name in the sidebar |
| `#<agent-name>` channel ID | The agent's private channel |
| `#agent-bus` channel ID | The agent-bus channel |
| `#agent-comms` channel ID | The agent-comms channel |

Have these four IDs and your bot token ready for the wizard.

---

### Step 6 — Run the install wizard

```bash
bash scripts/install.sh
```

The wizard will ask for:
1. **Agent name** — must match the repo name and Discord channel (e.g. `bob`)
2. **LLM providers** — choose one or more of Anthropic, OpenAI, Google, Groq, Mistral, and xAI
3. **API keys** for the providers you selected
4. **Optional OpenAI-compatible base URL** if you want to use Ollama, vLLM, or another compatible endpoint
5. **Discord bot token** and the four IDs from step 5f

The wizard saves everything to `.env`, then builds and starts the containers automatically. The build takes about 2 minutes on first run.

---

### Step 7 — Chat with your agent

Open Discord and send a message in `#<agent-name>`. The agent replies to every message there.

Watch the browser live at `http://<your-server-ip>:6080`.
The control plane is available at `http://<your-server-ip>:8000` with interactive docs at `/docs`.

Check logs:
```bash
docker compose logs -f agent
```

---

## Staying up to date

Your repo was created from the template, so it has its own independent history. To pull in future template updates:

```bash
# One-time setup
git remote add template https://github.com/josh-ta/agent.git
git fetch template
git merge template/main --allow-unrelated-histories

# Future updates
git fetch template && git merge template/main
docker compose up --build -d
```

---

## Local development

Install the project and dev tools locally:

```bash
pip install -e ".[dev]"
```

Run a one-off task without Discord:

```bash
python -m agent.main run "summarize the repository"
```

Run the FastAPI control plane locally:

```bash
python -m agent.main serve-api
```

Then open:

- `http://127.0.0.1:8000/docs` for Swagger UI
- `http://127.0.0.1:8000/openapi.json` for the generated OpenAPI schema

Run tests:

```bash
pytest
```

## HTTP control plane

The control plane exposes a minimal HTTP API for health checks, task submission, task lookup, and server-sent task events.

- `GET /healthz` returns liveness status
- `GET /readyz` returns readiness status for the runtime
- `POST /tasks` accepts a task and returns a stable task ID
- `GET /tasks/{id}` returns the persisted task state from SQLite
- `GET /events?task_id=<id>` streams task events over SSE, or omit `task_id` to watch all events

By default the Docker entrypoint serves the control plane on port `8000`. Set `CONTROL_PLANE_ENABLED=false` to disable it, or change `CONTROL_PLANE_PORT` to move it to another port.

See [`docs/api.md`](docs/api.md) for a short reference and example requests.

## Useful commands

```bash
docker compose logs -f agent        # live agent logs
docker compose logs -f browser      # browser sidecar logs
docker compose down                 # stop everything
docker compose up --build -d        # rebuild and restart
docker compose --profile postgres up -d  # start optional shared Postgres too
bash scripts/install.sh             # re-run configuration wizard
```

---

## Multi-agent setup

Each agent runs on its own server. They share the same Discord server.

1. Create a new VPS, create a new repo from the template, repeat steps 1–6
2. Give each agent a unique name (`bob`, `alice`, `researcher`, etc.)
3. Each agent gets its own private Discord channel
4. All agents share the same `#agent-bus` and `#agent-comms` channels

Agents communicate by posting JSON in `#agent-comms`:
```json
{"from": "bob", "to": "alice", "task": "research X and report back"}
```

---

## Architecture

```
VPS (Docker host)
├── agent container        ← Python + your configured model provider(s)
│   ├── shell tool         ← run commands on the server
│   ├── filesystem tool    ← read/write /workspace (and explicit external paths when needed)
│   ├── browser tool       ← Playwright via MCP sidecar
│   ├── discord tool       ← send/read Discord messages
│   └── self-edit tool     ← update own skills and identity
└── browser container      ← Playwright + noVNC (view at :6080)
```

Skills live in `agent/skills/` as Markdown files. The agent can create and edit them — changes persist across restarts because the directory is bind-mounted from the host.
