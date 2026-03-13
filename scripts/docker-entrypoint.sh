#!/usr/bin/env bash
# Runtime entrypoint: configures git identity and gh CLI auth from env vars,
# then hands off to the Python agent.
set -euo pipefail

# ── Git identity ───────────────────────────────────────────────────────────────
if [[ -n "${GITHUB_USERNAME:-}" ]]; then
    git config --global user.name  "${GITHUB_USERNAME}"
    git config --global user.email "${GITHUB_USERNAME}@users.noreply.github.com"
else
    # Fallback so git commits don't fail
    git config --global user.name  "${AGENT_NAME:-agent}"
    git config --global user.email "${AGENT_NAME:-agent}@localhost"
fi

# ── gh CLI auth ────────────────────────────────────────────────────────────────
# gh reads GH_TOKEN automatically; also set GITHUB_TOKEN as the legacy name.
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    export GH_TOKEN="${GITHUB_TOKEN}"
fi

# ── Start agent ────────────────────────────────────────────────────────────────
exec python -m agent.main "$@"
