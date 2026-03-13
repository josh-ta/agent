#!/usr/bin/env bash
# One-command install script for the agent platform.
# Run on a fresh VPS: bash scripts/install.sh
set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${CYAN}[agent]${NC} $*"; }
success() { echo -e "${GREEN}[agent]${NC} $*"; }
warn()    { echo -e "${YELLOW}[agent]${NC} $*"; }
error()   { echo -e "${RED}[agent] ERROR:${NC} $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 1. Check Docker ────────────────────────────────────────────────────────────
info "Checking Docker..."
if ! command -v docker &>/dev/null; then
    warn "Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    # Add current user to docker group
    if [[ "$EUID" -ne 0 ]]; then
        sudo usermod -aG docker "$USER"
        warn "Added $USER to docker group. You may need to log out and back in."
    fi
fi

DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+' | head -1)
info "Docker version: ${DOCKER_VERSION}"

# Check for docker compose (v2)
if ! docker compose version &>/dev/null; then
    error "Docker Compose v2 is required. Install via: sudo apt install docker-compose-plugin"
fi

# ── 2. Environment file ────────────────────────────────────────────────────────
info "Setting up environment..."
cd "$ROOT_DIR"

# Read a value for a .env variable interactively.
# Usage: prompt_var VAR_NAME "Description" "default_value" secret
#   secret=1  → input is hidden (for API keys/tokens)
prompt_var() {
    local var="$1"
    local desc="$2"
    local default="$3"
    local secret="${4:-0}"

    echo ""
    echo -e "  ${CYAN}${var}${NC}"
    echo -e "  ${desc}"
    if [[ -n "$default" ]]; then
        if [[ "$secret" == "1" ]]; then
            echo -e "  Current/default: ${YELLOW}(set)${NC}"
        else
            echo -e "  Current/default: ${YELLOW}${default}${NC}"
        fi
        local prompt_str="  New value (Enter to keep): "
    else
        local prompt_str="  Value: "
    fi

    if [[ "$secret" == "1" ]]; then
        read -rsp "$prompt_str" input
        echo ""   # newline after hidden input
    else
        read -rp "$prompt_str" input
    fi

    # Use new value if provided, otherwise keep default
    local value="${input:-$default}"

    # Write/update the variable in .env
    if grep -q "^${var}=" .env 2>/dev/null; then
        # Replace existing line (macOS + Linux compatible)
        sed -i.bak "s|^${var}=.*|${var}=${value}|" .env && rm -f .env.bak
    else
        echo "${var}=${value}" >> .env
    fi
}

# Helper: read current value from .env (strips inline comments)
current_val() {
    local var="$1"
    grep "^${var}=" .env 2>/dev/null \
        | head -1 \
        | sed "s/^${var}=//" \
        | sed 's/[[:space:]]*#.*//' \
        | tr -d '"' \
        || true
}

configure_env() {
    echo ""
    echo -e "${CYAN}┌─────────────────────────────────────────────────┐${NC}"
    echo -e "${CYAN}│        Agent Configuration Wizard               │${NC}"
    echo -e "${CYAN}│  Press Enter to accept defaults / current value │${NC}"
    echo -e "${CYAN}└─────────────────────────────────────────────────┘${NC}"

    # ── LLM ─────────────────────────────────────────────────────────────────
    echo ""
    echo -e "${CYAN}── LLM Provider ──────────────────────────────────────${NC}"

    prompt_var "ANTHROPIC_API_KEY" \
        "Anthropic Claude API key (required). Get one at https://console.anthropic.com" \
        "$(current_val ANTHROPIC_API_KEY)" 1

    prompt_var "AGENT_MODEL" \
        "Model to use (e.g. claude-opus-4-5, claude-sonnet-4-5, gpt-4o)" \
        "$(current_val AGENT_MODEL)"

    prompt_var "OPENAI_API_KEY" \
        "OpenAI API key (optional fallback model support)" \
        "$(current_val OPENAI_API_KEY)" 1

    # ── Agent Identity ───────────────────────────────────────────────────────
    echo ""
    echo -e "${CYAN}── Agent Identity ────────────────────────────────────${NC}"

    prompt_var "AGENT_NAME" \
        "Unique name for this agent instance (e.g. agent-1, researcher, coder)" \
        "$(current_val AGENT_NAME)"

    # ── Discord ──────────────────────────────────────────────────────────────
    echo ""
    echo -e "${CYAN}── Discord ───────────────────────────────────────────${NC}"
    echo -e "  ${YELLOW}Tip:${NC} Enable Developer Mode in Discord → right-click any channel/server to copy IDs."

    prompt_var "DISCORD_BOT_TOKEN" \
        "Discord bot token. Create a bot at https://discord.com/developers/applications" \
        "$(current_val DISCORD_BOT_TOKEN)" 1

    prompt_var "DISCORD_GUILD_ID" \
        "Your Discord server (guild) numeric ID" \
        "$(current_val DISCORD_GUILD_ID)"

    prompt_var "DISCORD_AGENT_CHANNEL_ID" \
        "Channel ID for THIS agent's private channel (e.g. #agent-${AGENT_NAME:-1})" \
        "$(current_val DISCORD_AGENT_CHANNEL_ID)"

    prompt_var "DISCORD_BUS_CHANNEL_ID" \
        "Channel ID for #agent-bus (shared broadcast channel, all agents)" \
        "$(current_val DISCORD_BUS_CHANNEL_ID)"

    prompt_var "DISCORD_COMMS_CHANNEL_ID" \
        "Channel ID for #agent-comms (structured A2A JSON messages)" \
        "$(current_val DISCORD_COMMS_CHANNEL_ID)"

    # ── Databases ────────────────────────────────────────────────────────────
    echo ""
    echo -e "${CYAN}── Databases ─────────────────────────────────────────${NC}"

    prompt_var "POSTGRES_URL" \
        "Shared PostgreSQL URL for multi-agent coordination (optional, leave blank to skip)" \
        "$(current_val POSTGRES_URL)"

    # ── Misc ─────────────────────────────────────────────────────────────────
    echo ""
    echo -e "${CYAN}── Container ─────────────────────────────────────────${NC}"

    prompt_var "AGENT_CONTAINER_NAME" \
        "Docker container name for self-restart (should match AGENT_NAME)" \
        "$(current_val AGENT_CONTAINER_NAME)"

    # ── GitHub ───────────────────────────────────────────────────────────────
    echo ""
    echo -e "${CYAN}── GitHub (optional) ─────────────────────────────────${NC}"
    echo -e "  ${YELLOW}Tip:${NC} Create a fine-grained PAT at https://github.com/settings/tokens"
    echo -e "       Scopes needed: Contents (read/write), Pull requests, Issues"

    prompt_var "GITHUB_TOKEN" \
        "GitHub personal access token (enables gh CLI + private repo access)" \
        "$(current_val GITHUB_TOKEN)" 1

    prompt_var "GITHUB_USERNAME" \
        "Your GitHub username (used for git commit authorship)" \
        "$(current_val GITHUB_USERNAME)"

    echo ""
    success "Configuration saved to .env"
}

if [[ ! -f .env ]]; then
    cp .env.example .env
    configure_env
else
    info ".env already exists."
    echo -ne "  ${YELLOW}Reconfigure interactively?${NC} [y/N] "
    read -r reconfigure
    if [[ "${reconfigure,,}" == "y" ]]; then
        configure_env
    fi
fi

# ── 3. Set up git ──────────────────────────────────────────────────────────────
if [[ ! -d .git ]]; then
    info "Initialising git repository..."
    git init
    git add .
    git commit -m "chore: initial agent setup"
fi

# ── 4. Create workspace ────────────────────────────────────────────────────────
mkdir -p workspace
chmod 755 workspace

# ── 5. Build + start containers ───────────────────────────────────────────────
info "Building and starting agent containers..."
docker compose build --no-cache
docker compose up -d

# ── 6. Health check ────────────────────────────────────────────────────────────
info "Waiting for services to be healthy (up to 60s)..."
TIMEOUT=60
ELAPSED=0
while [[ $ELAPSED -lt $TIMEOUT ]]; do
    STATUS=$(docker compose ps --format json 2>/dev/null | python3 -c "
import sys, json
data = sys.stdin.read().strip()
if not data:
    print('starting')
    exit(0)
lines = data.splitlines()
for line in lines:
    try:
        svc = json.loads(line)
        health = svc.get('Health', 'starting')
        if health == 'unhealthy':
            print('unhealthy')
            exit(0)
    except Exception:
        pass
print('healthy')
" 2>/dev/null || echo "starting")

    if [[ "$STATUS" == "healthy" ]]; then
        break
    fi
    sleep 3
    ELAPSED=$((ELAPSED + 3))
done

# ── 7. Summary ─────────────────────────────────────────────────────────────────
echo ""
success "════════════════════════════════════════"
success "  Agent platform is up!"
success "════════════════════════════════════════"
echo ""
info "Services:"
docker compose ps
echo ""
info "Live browser view (noVNC):  http://$(hostname -I | awk '{print $1}'):6080"
info "Agent logs:                 docker compose logs -f agent"
info "Stop:                       docker compose down"
echo ""
success "Done! Check Discord for the agent's online announcement."
