#!/usr/bin/env bash
# One-command install script for the agent platform.
# Run on a fresh VPS: bash scripts/install.sh
set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[agent]${NC} $*"; }
success() { echo -e "${GREEN}[agent]${NC} $*"; }
warn()    { echo -e "${YELLOW}[agent]${NC} $*"; }
error()   { echo -e "${RED}[agent] ERROR:${NC} $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 1. Check / install Docker ──────────────────────────────────────────────────
info "Checking Docker..."
if ! command -v docker &>/dev/null; then
    warn "Docker not found. Installing..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
fi
if ! docker compose version &>/dev/null; then
    error "Docker Compose v2 is required. Install: sudo apt install docker-compose-plugin"
fi
info "Docker $(docker --version | grep -oP '\d+\.\d+' | head -1) ready"

# ── Helpers ────────────────────────────────────────────────────────────────────
cd "$ROOT_DIR"

# Write/update a single variable in .env
set_var() {
    local var="$1" value="$2"
    if grep -q "^${var}=" .env 2>/dev/null; then
        sed -i.bak "s|^${var}=.*|${var}=${value}|" .env && rm -f .env.bak
    else
        echo "${var}=${value}" >> .env
    fi
}

# Read current value from .env (strips inline comments + quotes)
current_val() {
    grep "^${1}=" .env 2>/dev/null \
        | head -1 | sed "s/^${1}=//" \
        | sed 's/[[:space:]]*#.*//' | tr -d '"' || true
}

# Prompt for a value, hiding input when secret=1
prompt_var() {
    local var="$1" desc="$2" default="$3" secret="${4:-0}"
    echo ""
    echo -e "  ${CYAN}${var}${NC}  ${desc}"
    local prompt_str="  Value: "
    if [[ -n "$default" ]]; then
        if [[ "$secret" == "1" ]]; then
            echo -e "  Current: ${YELLOW}(set)${NC}"
        else
            echo -e "  Current: ${YELLOW}${default}${NC}"
        fi
        prompt_str="  New value (Enter to keep): "
    fi
    local input=""
    if [[ "$secret" == "1" ]]; then
        read -rsp "$prompt_str" input; echo ""
    else
        read -rp "$prompt_str" input
    fi
    set_var "$var" "${input:-$default}"
}

# Arrow-key / spacebar multi-select menu
# Usage: multiselect result_var "Option 1" "Option 2" ...
# Sets result_var to a space-separated list of selected indices (0-based)
multiselect() {
    local result_var="$1"; shift
    local options=("$@")
    local selected=()
    local cursor=0
    local n=${#options[@]}

    # Default: none selected
    for ((i=0; i<n; i++)); do selected+=("false"); done

    # Hide cursor
    tput civis 2>/dev/null || true
    trap 'tput cnorm 2>/dev/null || true' EXIT INT TERM

    _draw() {
        for ((i=0; i<n; i++)); do
            local mark="[ ]"
            [[ "${selected[$i]}" == "true" ]] && mark="[x]"
            if [[ "$i" == "$cursor" ]]; then
                echo -e "  ${YELLOW}▶ ${mark} ${options[$i]}${NC}"
            else
                echo -e "    ${mark} ${options[$i]}"
            fi
        done
    }

    while true; do
        _draw
        # Read a key (including escape sequences for arrows)
        local key
        IFS= read -rsn1 key
        if [[ "$key" == $'\x1b' ]]; then
            local seq
            read -rsn2 -t 0.1 seq || true
            key="${key}${seq}"
        fi
        case "$key" in
            $'\x1b[A'|k) # up
                ((cursor > 0)) && ((cursor--)) || true
                ;;
            $'\x1b[B'|j) # down
                ((cursor < n-1)) && ((cursor++)) || true
                ;;
            ' ') # spacebar — toggle
                if [[ "${selected[$cursor]}" == "true" ]]; then
                    selected[$cursor]="false"
                else
                    selected[$cursor]="true"
                fi
                ;;
            ''|$'\n') # enter — confirm
                break
                ;;
        esac
        # Move cursor up to redraw
        for ((i=0; i<n; i++)); do tput cuu1 2>/dev/null || true; done
    done

    tput cnorm 2>/dev/null || true

    local result_indices=()
    for ((i=0; i<n; i++)); do
        [[ "${selected[$i]}" == "true" ]] && result_indices+=("$i")
    done
    printf -v "$result_var" '%s' "${result_indices[*]}"
}

# ── 2. Configuration wizard ────────────────────────────────────────────────────
[[ ! -f .env ]] && cp .env.example .env

configure_env() {
    echo ""
    echo -e "${BOLD}${CYAN}┌─────────────────────────────────────────────┐${NC}"
    echo -e "${BOLD}${CYAN}│         Agent Configuration Wizard          │${NC}"
    echo -e "${BOLD}${CYAN}│  ↑↓ arrows, Space to select, Enter confirm  │${NC}"
    echo -e "${BOLD}${CYAN}└─────────────────────────────────────────────┘${NC}"

    # ── Agent name ──────────────────────────────────────────────────────────
    echo ""
    echo -e "${CYAN}── Identity ──────────────────────────────────────────${NC}"
    prompt_var "AGENT_NAME" \
        "Unique name for this agent (e.g. bob, researcher, coder)" \
        "$(current_val AGENT_NAME)"

    # ── LLM provider selection ───────────────────────────────────────────────
    echo ""
    echo -e "${CYAN}── LLM Providers ─────────────────────────────────────${NC}"
    echo -e "  Use ${YELLOW}Space${NC} to select providers, ${YELLOW}Enter${NC} to confirm:"
    echo ""

    selected_providers=""
    multiselect selected_providers \
        "Anthropic (Claude — recommended)" \
        "OpenAI (GPT-4o / o-series)" \
        "Google (Gemini)" \
        "Groq (fast open-source models)" \
        "Mistral" \
        "xAI (Grok)"

    # Parse selections
    want_anthropic=false; want_openai=false; want_google=false
    want_groq=false; want_mistral=false; want_xai=false
    for idx in $selected_providers; do
        [[ "$idx" == "0" ]] && want_anthropic=true
        [[ "$idx" == "1" ]] && want_openai=true
        [[ "$idx" == "2" ]] && want_google=true
        [[ "$idx" == "3" ]] && want_groq=true
        [[ "$idx" == "4" ]] && want_mistral=true
        [[ "$idx" == "5" ]] && want_xai=true
    done

    if [[ "$want_anthropic" == "true" ]]; then
        prompt_var "ANTHROPIC_API_KEY" \
            "Anthropic API key — console.anthropic.com" \
            "$(current_val ANTHROPIC_API_KEY)" 1

        # Set default model if not already set
        current_model="$(current_val AGENT_MODEL)"
        if [[ -z "$current_model" || "$current_model" == "claude-opus-4-5" ]]; then
            echo ""
            echo -e "${CYAN}── Default Claude Model ──────────────────────────────${NC}"
            echo -e "  Choose default model (Space to select, Enter to confirm):"
            echo ""
            selected_model=""
            multiselect selected_model \
                "claude-opus-4-5   (most capable, slower)" \
                "claude-sonnet-4-5 (balanced — recommended)" \
                "claude-haiku-4-5  (fastest, lightest)"
            for idx in $selected_model; do
                case "$idx" in
                    0) set_var "AGENT_MODEL" "claude-opus-4-5" ;;
                    1) set_var "AGENT_MODEL" "claude-sonnet-4-5" ;;
                    2) set_var "AGENT_MODEL" "claude-haiku-4-5" ;;
                esac
            done
        fi
    fi

    if [[ "$want_openai" == "true" ]]; then
        prompt_var "OPENAI_API_KEY" \
            "OpenAI API key — platform.openai.com/api-keys" \
            "$(current_val OPENAI_API_KEY)" 1
        if [[ "$want_anthropic" == "false" ]]; then
            set_var "AGENT_MODEL" "gpt-4o"
        fi
    fi

    if [[ "$want_google" == "true" ]]; then
        prompt_var "GOOGLE_API_KEY" \
            "Google AI API key — aistudio.google.com/app/apikey" \
            "$(current_val GOOGLE_API_KEY)" 1
        if [[ "$want_anthropic" == "false" && "$want_openai" == "false" ]]; then
            set_var "AGENT_MODEL" "gemini-2.0-flash"
        fi
    fi

    if [[ "$want_groq" == "true" ]]; then
        prompt_var "GROQ_API_KEY" \
            "Groq API key — console.groq.com/keys" \
            "$(current_val GROQ_API_KEY)" 1
    fi

    if [[ "$want_mistral" == "true" ]]; then
        prompt_var "MISTRAL_API_KEY" \
            "Mistral API key — console.mistral.ai" \
            "$(current_val MISTRAL_API_KEY)" 1
    fi

    if [[ "$want_xai" == "true" ]]; then
        prompt_var "XAI_API_KEY" \
            "xAI API key — console.x.ai" \
            "$(current_val XAI_API_KEY)" 1
        if [[ "$want_anthropic" == "false" && "$want_openai" == "false" && "$want_google" == "false" ]]; then
            set_var "AGENT_MODEL" "grok-3"
        fi
    fi

    # ── Model tiers ──────────────────────────────────────────────────────────
    echo ""
    echo -e "${CYAN}── Model Tiers (optional) ────────────────────────────${NC}"
    echo -e "  ${YELLOW}Tip:${NC} Users can prefix messages with /fast, /smart, /best"
    echo -e "       to route to different model tiers. Press Enter to keep defaults."
    prompt_var "MODEL_FAST" \
        "Fast tier — simple Q&A, greetings" \
        "$(current_val MODEL_FAST)"
    prompt_var "MODEL_SMART" \
        "Smart tier — code, research, multi-step tasks" \
        "$(current_val MODEL_SMART)"
    prompt_var "MODEL_BEST" \
        "Best tier — complex reasoning, architecture" \
        "$(current_val MODEL_BEST)"

    # ── GitHub ───────────────────────────────────────────────────────────────
    echo ""
    echo -e "${CYAN}── GitHub (optional) ─────────────────────────────────${NC}"
    echo -e "  ${YELLOW}Tip:${NC} Required for the agent to push code, open PRs, or use gh CLI."
    prompt_var "GITHUB_TOKEN" \
        "Personal access token — github.com/settings/tokens (scopes: repo, read:org)" \
        "$(current_val GITHUB_TOKEN)" 1
    prompt_var "GITHUB_USERNAME" \
        "GitHub username (for git commit identity inside the container)" \
        "$(current_val GITHUB_USERNAME)"

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    echo ""
    echo -e "${CYAN}── PostgreSQL (optional) ─────────────────────────────${NC}"
    echo -e "  ${YELLOW}Tip:${NC} Connect to an external Postgres server for your own data/tools."
    echo -e "       (SQLite handles agent-internal state automatically.)"
    echo -ne "  Connect to an external PostgreSQL database? [y/N] "
    read -r use_postgres
    if [[ "${use_postgres,,}" == "y" ]]; then
        prompt_var "PG_HOST" \
            "Postgres host (IP or hostname)" \
            "$(current_val PG_HOST)"
        prompt_var "PG_PORT" \
            "Postgres port" \
            "$(current_val PG_PORT)"
        prompt_var "PG_DATABASE" \
            "Database name" \
            "$(current_val PG_DATABASE)"
        prompt_var "PG_USER" \
            "Database user" \
            "$(current_val PG_USER)"
        prompt_var "PG_PASSWORD" \
            "Database password" \
            "$(current_val PG_PASSWORD)" 1
        # Assemble POSTGRES_URL from the individual fields
        _pg_host="$(current_val PG_HOST)"
        _pg_port="$(current_val PG_PORT)"
        _pg_db="$(current_val PG_DATABASE)"
        _pg_user="$(current_val PG_USER)"
        _pg_pass="$(current_val PG_PASSWORD)"
        set_var "POSTGRES_URL" "postgresql+asyncpg://${_pg_user}:${_pg_pass}@${_pg_host}:${_pg_port}/${_pg_db}"
    fi

    # ── Proxy ────────────────────────────────────────────────────────────────
    echo ""
    echo -e "${CYAN}── Proxy & Browser Identity (optional) ──────────────${NC}"
    echo -e "  ${YELLOW}Tip:${NC} Routes browser traffic through a residential/ISP proxy so the"
    echo -e "       agent appears to come from your target country, not a datacenter."
    echo -ne "  Configure a proxy? [y/N] "
    read -r use_proxy
    if [[ "${use_proxy,,}" == "y" ]]; then
        echo ""
        echo -e "  ${YELLOW}Option A${NC} — single URL (recommended for most providers):"
        echo -e "    Format: http://user:pass@host:port  or  socks5://user:pass@host:port"
        prompt_var "PROXY_URL" \
            "Full proxy URL (leave blank to use split fields below)" \
            "$(current_val PROXY_URL)"

        current_proxy_url="$(current_val PROXY_URL)"
        if [[ -z "$current_proxy_url" ]]; then
            echo ""
            echo -e "  ${YELLOW}Option B${NC} — split fields (some providers supply these separately):"
            prompt_var "PROXY_SERVER" \
                "Proxy host:port (e.g. proxy.example.com:8080)" \
                "$(current_val PROXY_SERVER)"
            prompt_var "PROXY_USERNAME" \
                "Proxy username" \
                "$(current_val PROXY_USERNAME)"
            prompt_var "PROXY_PASSWORD" \
                "Proxy password" \
                "$(current_val PROXY_PASSWORD)" 1
        fi

        echo ""
        echo -e "  ${CYAN}── Browser locale / geolocation ──────────────────────${NC}"
        echo -e "  ${YELLOW}Tip:${NC} Should match your proxy's exit country so locale, timezone,"
        echo -e "       and geolocation all agree with the IP address."
        prompt_var "BROWSER_LOCALE" \
            "Browser locale (e.g. en-US, en-GB, de-DE)" \
            "$(current_val BROWSER_LOCALE)"
        prompt_var "BROWSER_TIMEZONE" \
            "Browser timezone (e.g. America/New_York, Europe/London)" \
            "$(current_val BROWSER_TIMEZONE)"
        prompt_var "BROWSER_GEO_LAT" \
            "Geolocation latitude  (e.g. 40.7128 for New York)" \
            "$(current_val BROWSER_GEO_LAT)"
        prompt_var "BROWSER_GEO_LON" \
            "Geolocation longitude (e.g. -74.0060 for New York)" \
            "$(current_val BROWSER_GEO_LON)"
    fi

    # ── Discord ──────────────────────────────────────────────────────────────
    echo ""
    echo -e "${CYAN}── Discord ───────────────────────────────────────────${NC}"
    echo -e "  ${YELLOW}Tip:${NC} In Discord → User Settings → Advanced → enable Developer Mode"
    echo -e "       then right-click any channel/server to copy its ID."

    prompt_var "DISCORD_BOT_TOKEN" \
        "Bot token — discord.com/developers/applications" \
        "$(current_val DISCORD_BOT_TOKEN)" 1

    prompt_var "DISCORD_GUILD_ID" \
        "Server (guild) ID — right-click server name → Copy Server ID" \
        "$(current_val DISCORD_GUILD_ID)"

    AGENT_NAME_VAL="$(current_val AGENT_NAME)"
    prompt_var "DISCORD_AGENT_CHANNEL_ID" \
        "Channel ID for #${AGENT_NAME_VAL:-agent} (this agent's private channel)" \
        "$(current_val DISCORD_AGENT_CHANNEL_ID)"

    prompt_var "DISCORD_BUS_CHANNEL_ID" \
        "Channel ID for #agent-bus (broadcast, all agents)" \
        "$(current_val DISCORD_BUS_CHANNEL_ID)"

    prompt_var "DISCORD_COMMS_CHANNEL_ID" \
        "Channel ID for #agent-comms (structured A2A messages)" \
        "$(current_val DISCORD_COMMS_CHANNEL_ID)"

    echo ""
    success "Configuration saved to .env"
}

if [[ "$(current_val AGENT_NAME)" == "" ]] || [[ "$(current_val DISCORD_BOT_TOKEN)" == "" ]]; then
    configure_env
else
    info ".env already configured."
    echo -ne "  ${YELLOW}Reconfigure interactively?${NC} [y/N] "
    read -r reconfigure
    [[ "${reconfigure,,}" == "y" ]] && configure_env
fi

# Set AGENT_CONTAINER_NAME to match AGENT_NAME automatically
AGENT_NAME_VAL="$(current_val AGENT_NAME)"
if [[ -n "$AGENT_NAME_VAL" ]]; then
    set_var "AGENT_CONTAINER_NAME" "$AGENT_NAME_VAL"
fi

# ── 3. Workspace ───────────────────────────────────────────────────────────────
mkdir -p workspace
chmod 755 workspace

# ── 4. Build + start ───────────────────────────────────────────────────────────
info "Building and starting containers (this takes ~2 min on first run)..."
docker compose build --no-cache
docker compose up -d

# ── 5. Wait for healthy ────────────────────────────────────────────────────────
info "Waiting for services to be healthy (up to 120s)..."
for i in $(seq 1 40); do
    sleep 3
    STATUS=$(docker compose ps --format json 2>/dev/null | python3 -c "
import sys, json
lines = sys.stdin.read().strip().splitlines()
states = []
for l in lines:
    try:
        s = json.loads(l)
        h = s.get('Health','')
        if h: states.append(h)
    except: pass
if any(h == 'unhealthy' for h in states): print('unhealthy')
elif all(h == 'healthy' for h in states) and states: print('healthy')
else: print('starting')
" 2>/dev/null || echo "starting")
    [[ "$STATUS" == "healthy" ]] && break
    [[ "$STATUS" == "unhealthy" ]] && { warn "A container is unhealthy — run: docker compose logs"; break; }
done

# ── 6. Summary ─────────────────────────────────────────────────────────────────
echo ""
success "════════════════════════════════════════"
success "  Agent is up!"
success "════════════════════════════════════════"
echo ""
docker compose ps
echo ""
info "Live browser:  http://$(hostname -I | awk '{print $1}'):6080/vnc.html"
info "Logs:          docker compose logs -f agent"
info "Stop:          docker compose down"
echo ""
success "Done! Check Discord — the agent will announce itself in #agent-bus."
