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
    # Copy to /data/ssh (persistent volume) so keys survive container restarts.
    # Fall back to /tmp/agent_ssh if /data isn't writable yet.
    if [[ -d /data ]] && touch /data/.writable 2>/dev/null; then
        rm -f /data/.writable
        SSH_DEST=/data/ssh
    else
        SSH_DEST=/tmp/agent_ssh
    fi

    mkdir -p "$SSH_DEST"
    cp -r /root/.ssh/. "$SSH_DEST/"
    chmod 700 "$SSH_DEST"
    chmod 600 "$SSH_DEST"/* 2>/dev/null || true

    # Prefer agent_ed25519, fall back to id_ed25519 or id_rsa (server's own key)
    KEY=""
    for candidate in \
        "$SSH_DEST/agent_ed25519" \
        "$SSH_DEST/id_ed25519" \
        "$SSH_DEST/id_rsa"; do
        if [[ -f "$candidate" ]]; then
            KEY="$candidate"
            break
        fi
    done

    if [[ -n "$KEY" ]]; then
        mkdir -p ~/.ssh
        chmod 700 ~/.ssh
        cat > ~/.ssh/config <<EOF
Host github.com
    IdentityFile ${KEY}
    StrictHostKeyChecking accept-new
    AddKeysToAgent no
Host *
    IdentityFile ${KEY}
    StrictHostKeyChecking accept-new
    AddKeysToAgent no
EOF
        chmod 600 ~/.ssh/config
        ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null || true
    fi
fi

# ── gh CLI auth ────────────────────────────────────────────────────────────────
# gh reads GH_TOKEN / GITHUB_TOKEN automatically for API calls.
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    export GH_TOKEN="${GITHUB_TOKEN}"
fi

# ── Start agent ────────────────────────────────────────────────────────────────
exec python -m agent.main start "$@"
