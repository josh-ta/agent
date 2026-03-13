#!/usr/bin/env bash
# Runtime entrypoint: configures git identity, SSH, and gh CLI auth,
# then hands off to the Python agent.
set -euo pipefail

# ── Git identity ───────────────────────────────────────────────────────────────
if [[ -n "${GITHUB_USERNAME:-}" ]]; then
    git config --global user.name  "${GITHUB_USERNAME}"
    git config --global user.email "${GITHUB_USERNAME}@users.noreply.github.com"
else
    git config --global user.name  "${AGENT_NAME:-agent}"
    git config --global user.email "${AGENT_NAME:-agent}@localhost"
fi

# ── SSH setup ─────────────────────────────────────────────────────────────────
# The .ssh directory is bind-mounted from the host at /root/.ssh (read-only).
# We need correct permissions — SSH refuses keys that are group/world readable.
if [[ -d /root/.ssh ]]; then
    # Copy to a writable location so we can chmod (bind mount may be read-only)
    cp -r /root/.ssh /tmp/agent_ssh
    chmod 700 /tmp/agent_ssh
    chmod 600 /tmp/agent_ssh/* 2>/dev/null || true

    # Point SSH at the agent key
    KEY="/tmp/agent_ssh/agent_ed25519"
    if [[ -f "$KEY" ]]; then
        mkdir -p ~/.ssh
        chmod 700 ~/.ssh

        # Write an SSH config that uses this key for github.com
        cat > ~/.ssh/config <<EOF
Host github.com
    IdentityFile ${KEY}
    StrictHostKeyChecking accept-new
    AddKeysToAgent no
EOF
        chmod 600 ~/.ssh/config

        # Accept github.com host key silently on first connect
        ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null || true
    fi
fi

# ── gh CLI auth ────────────────────────────────────────────────────────────────
# gh reads GH_TOKEN / GITHUB_TOKEN automatically for API calls.
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    export GH_TOKEN="${GITHUB_TOKEN}"
fi

# ── Patch identity files with runtime agent name ───────────────────────────────
# IDENTITY.md uses ${AGENT_NAME} as a placeholder — substitute it now.
# Uses Python to avoid sed -i needing a writable temp file in the same directory
# (bind-mounted directories may not allow temp file creation).
IDENTITY_FILE="/app/agent/identity/IDENTITY.md"
if [[ -f "$IDENTITY_FILE" ]]; then
    python3 - "$IDENTITY_FILE" "${AGENT_NAME:-agent}" <<'PYEOF'
import sys, pathlib
path, name = pathlib.Path(sys.argv[1]), sys.argv[2]
text = path.read_text()
if '${AGENT_NAME}' in text:
    path.write_text(text.replace('${AGENT_NAME}', name))
PYEOF
fi


exec python -m agent.main start "$@"
