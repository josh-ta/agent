#!/usr/bin/env bash
# Pull latest code and restart the agent with zero-downtime approach.
set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

info()    { echo -e "${CYAN}[update]${NC} $*"; }
success() { echo -e "${GREEN}[update]${NC} $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

info "Pulling latest changes..."
git pull --ff-only

info "Rebuilding containers..."
docker compose build

info "Restarting agent (browser stays running for speed)..."
docker compose up -d --no-deps agent

info "Waiting for agent to be healthy..."
sleep 5
docker compose ps agent

success "Update complete."
