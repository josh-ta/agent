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

if [[ ! -f .env ]]; then
    cp .env.example .env
    warn ".env created from .env.example — PLEASE EDIT IT with your API keys!"
    warn ""
    warn "  Required:"
    warn "    ANTHROPIC_API_KEY  — your Claude API key"
    warn "    DISCORD_BOT_TOKEN  — your Discord bot token"
    warn "    DISCORD_AGENT_CHANNEL_ID"
    warn "    DISCORD_BUS_CHANNEL_ID"
    warn ""
    warn "Press Enter to open .env in nano, or Ctrl+C to edit manually later."
    read -r
    nano .env
else
    info ".env already exists — skipping."
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
