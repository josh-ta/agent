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
    echo -e "${CYAN}── GitHub SSH ────────────────────────────────────────${NC}"
    echo -e "  ${YELLOW}Tip:${NC} We'll generate an SSH key for this agent and walk you through"
    echo -e "       adding it to GitHub so the agent can clone/push repos."

    prompt_var "GITHUB_USERNAME" \
        "Your GitHub username (used for git commit authorship)" \
        "$(current_val GITHUB_USERNAME)"

    prompt_var "GITHUB_TOKEN" \
        "GitHub personal access token (for gh CLI — API calls, PRs, issues). Leave blank to skip." \
        "$(current_val GITHUB_TOKEN)" 1

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

# ── 3. GitHub SSH key ──────────────────────────────────────────────────────────
setup_github_ssh() {
    local ssh_dir="${ROOT_DIR}/.ssh"
    local key_file="${ssh_dir}/agent_ed25519"

    mkdir -p "$ssh_dir"
    chmod 700 "$ssh_dir"

    # Generate key if it doesn't exist yet
    if [[ ! -f "$key_file" ]]; then
        info "Generating SSH key for this agent..."
        local label="agent@$(hostname)"
        ssh-keygen -t ed25519 -C "$label" -f "$key_file" -N ""
        success "SSH key generated: ${key_file}"
    else
        info "SSH key already exists: ${key_file}"
    fi

    # Show the public key and wait for the user to add it to GitHub
    echo ""
    echo -e "${CYAN}┌──────────────────────────────────────────────────────────┐${NC}"
    echo -e "${CYAN}│  Add this public key to GitHub as a Deploy Key or        │${NC}"
    echo -e "${CYAN}│  go to: https://github.com/settings/ssh/new              │${NC}"
    echo -e "${CYAN}└──────────────────────────────────────────────────────────┘${NC}"
    echo ""
    cat "${key_file}.pub"
    echo ""
    echo -e "  ${YELLOW}→ Copy the key above, add it to GitHub, then press Enter to verify.${NC}"
    read -rp "  Press Enter when done (or Ctrl+C to skip SSH setup): "

    # Test the connection
    info "Testing SSH connection to github.com..."
    if ssh -i "$key_file" \
           -o StrictHostKeyChecking=accept-new \
           -o BatchMode=yes \
           -o ConnectTimeout=10 \
           -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        success "SSH connection to GitHub verified!"
    else
        # ssh returns exit 1 even on success (GitHub closes the shell), so
        # check the output text rather than the exit code
        local result
        result=$(ssh -i "$key_file" \
                     -o StrictHostKeyChecking=accept-new \
                     -o BatchMode=yes \
                     -o ConnectTimeout=10 \
                     -T git@github.com 2>&1 || true)
        if echo "$result" | grep -q "successfully authenticated"; then
            success "SSH connection to GitHub verified!"
        else
            warn "Could not verify SSH connection. Output: ${result}"
            warn "You can test manually later: ssh -i ${key_file} -T git@github.com"
        fi
    fi

    # Write the SSH key path to .env so the entrypoint can use it
    if grep -q "^GITHUB_SSH_KEY=" .env 2>/dev/null; then
        sed -i.bak "s|^GITHUB_SSH_KEY=.*|GITHUB_SSH_KEY=/root/.ssh/agent_ed25519|" .env && rm -f .env.bak
    else
        echo "GITHUB_SSH_KEY=/root/.ssh/agent_ed25519" >> .env
    fi
}

echo ""
echo -ne "  ${YELLOW}Set up GitHub SSH key?${NC} [Y/n] "
read -r do_ssh
if [[ "${do_ssh,,}" != "n" ]]; then
    setup_github_ssh
fi

# ── 4. Wire up GitHub remotes (origin = agent fork, upstream = base repo) ──────
setup_github_remotes() {
    local github_user
    github_user=$(grep "^GITHUB_USERNAME=" .env 2>/dev/null | sed 's/^GITHUB_USERNAME=//' | sed 's/[[:space:]]*#.*//' | tr -d '"' || echo "")

    local github_token
    github_token=$(grep "^GITHUB_TOKEN=" .env 2>/dev/null | sed 's/^GITHUB_TOKEN=//' | sed 's/[[:space:]]*#.*//' | tr -d '"' || echo "")

    if [[ -z "$github_user" || -z "$github_token" ]]; then
        warn "Skipping GitHub remote setup — GITHUB_USERNAME or GITHUB_TOKEN not set."
        return
    fi

    export GH_TOKEN="$github_token"

    # Detect what origin currently points at
    local origin_repo=""
    if git remote get-url origin &>/dev/null 2>&1; then
        origin_repo=$(git remote get-url origin \
            | sed 's|https://github.com/||' \
            | sed 's|git@github.com:||' \
            | sed 's|\.git$||')
    fi

    # ── Case 1: Already cloned the fork (most common / recommended flow) ──────
    # origin is already josh-ta/bob — just add upstream and we're done.
    local agent_name
    agent_name=$(grep "^AGENT_NAME=" .env 2>/dev/null | sed 's/^AGENT_NAME=//' | sed 's/[[:space:]]*#.*//' | tr -d '"' || echo "agent-1")

    local expected_fork="${github_user}/${agent_name}"

    if [[ "$origin_repo" == "$expected_fork" ]]; then
        info "origin is already the agent fork (${expected_fork}) — good."

        # Add upstream → base repo if missing
        if ! git remote get-url upstream &>/dev/null 2>&1; then
            echo ""
            read -rp "  Base repo to track as upstream (e.g. josh-ta/agent, Enter to skip): " base_repo
            if [[ -n "$base_repo" ]]; then
                git remote add upstream "git@github.com:${base_repo}.git"
                success "Added upstream → ${base_repo}"
            fi
        else
            info "upstream already set: $(git remote get-url upstream)"
        fi

        # Write AGENT_REPO to .env
        _write_agent_repo "$expected_fork"
        return
    fi

    # ── Case 2: Cloned the base repo directly — need to fork + re-point ──────
    echo ""
    echo -e "  ${YELLOW}origin${NC} is currently: ${origin_repo:-"(none)"}"
    echo -e "  Expected agent fork:  ${expected_fork}"
    echo ""
    echo -e "  Options:"
    echo -e "   [1] Fork ${origin_repo} → ${expected_fork} on GitHub and re-point origin (recommended)"
    echo -e "   [2] I already have a fork — enter its name"
    echo -e "   [3] Skip"
    read -rp "  Choice [1]: " fork_choice
    fork_choice="${fork_choice:-1}"

    local base_repo="$origin_repo"
    local fork_full="$expected_fork"

    case "$fork_choice" in
        2)
            read -rp "  Fork repo (owner/name): " fork_full
            ;;
        3)
            warn "Skipping fork setup."
            return
            ;;
        *)
            # Option 1: create the fork
            if gh repo view "$fork_full" &>/dev/null 2>&1; then
                info "Fork ${fork_full} already exists on GitHub."
            else
                info "Forking ${base_repo} → ${fork_full}..."
                gh repo fork "$base_repo" \
                    --fork-name "$agent_name" \
                    --clone=false \
                    --default-branch-only
                success "Forked to ${fork_full}"
            fi
            ;;
    esac

    # Re-point origin to the fork via SSH
    local fork_ssh="git@github.com:${fork_full}.git"
    git remote set-url origin "$fork_ssh" 2>/dev/null || git remote add origin "$fork_ssh"
    info "origin → ${fork_ssh}"

    # Add upstream → base
    if [[ -n "$base_repo" ]] && ! git remote get-url upstream &>/dev/null 2>&1; then
        git remote add upstream "git@github.com:${base_repo}.git"
        info "upstream → ${base_repo}"
    fi

    _write_agent_repo "$fork_full"
    success "GitHub remotes configured: origin=${fork_full}  upstream=${base_repo}"
    echo -e "  ${YELLOW}Tip:${NC} Pull base updates later: git fetch upstream && git merge upstream/main"
}

_write_agent_repo() {
    local repo="$1"
    if grep -q "^AGENT_REPO=" .env 2>/dev/null; then
        sed -i.bak "s|^AGENT_REPO=.*|AGENT_REPO=${repo}|" .env && rm -f .env.bak
    else
        echo "AGENT_REPO=${repo}" >> .env
    fi
}

echo ""
echo -ne "  ${YELLOW}Set up GitHub remotes (origin=fork, upstream=base)?${NC} [Y/n] "
read -r do_fork
if [[ "${do_fork,,}" != "n" ]]; then
    setup_github_remotes
fi

# ── 4. Set up git ──────────────────────────────────────────────────────────────
if [[ ! -d .git ]]; then
    info "Initialising git repository..."
    git init
    git add .
    git commit -m "chore: initial agent setup"
fi

# ── 5. Create workspace ────────────────────────────────────────────────────────
mkdir -p workspace
chmod 755 workspace

# ── 6. Build + start containers ───────────────────────────────────────────────
info "Building and starting agent containers..."
docker compose build --no-cache
docker compose up -d

# ── 7. Health check ────────────────────────────────────────────────────────────
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
