#!/usr/bin/env bash
# Runtime entrypoint: fixes workspace permissions, configures git/SSH/gh,
# then starts the Python agent.
set -euo pipefail

# ── Workspace permissions ──────────────────────────────────────────────────────
# ./workspace is bind-mounted from the host and may be owned by root.
# Make it fully writable so the agent can clone repos and create files there.
chmod -R a+rwX /workspace 2>/dev/null || true

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
    chmod 700 "$SSH_DEST" 2>/dev/null || true
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
        # Write SSH config to a writable location (not the read-only bind mount)
        mkdir -p /root/.ssh_rw
        chmod 700 /root/.ssh_rw
        cat > /root/.ssh_rw/config <<EOF
Host github.com
    IdentityFile ${KEY}
    StrictHostKeyChecking accept-new
    AddKeysToAgent no
Host *
    IdentityFile ${KEY}
    StrictHostKeyChecking accept-new
    AddKeysToAgent no
EOF
        chmod 600 /root/.ssh_rw/config
        # Copy known_hosts from read-only mount if present, then add github
        cp /root/.ssh/known_hosts /root/.ssh_rw/known_hosts 2>/dev/null || true
        ssh-keyscan -t ed25519 github.com >> /root/.ssh_rw/known_hosts 2>/dev/null || true
        # Point SSH at this writable config dir
        export GIT_SSH_COMMAND="ssh -F /root/.ssh_rw/config -o UserKnownHostsFile=/root/.ssh_rw/known_hosts"
    fi
fi

# ── gh CLI auth ────────────────────────────────────────────────────────────────
# gh reads GH_TOKEN / GITHUB_TOKEN automatically for API calls.
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    export GH_TOKEN="${GITHUB_TOKEN}"
fi

# ── Trim oversized MEMORY.md ──────────────────────────────────────────────────
# Bob appends lessons after every failure. If MEMORY.md grows too large it
# blows the context window. Keep only the last 4000 chars (most recent lessons).
MEMORY_FILE="${IDENTITY_PATH:-/app/agent/identity}/MEMORY.md"
if [[ -f "$MEMORY_FILE" ]]; then
    SIZE=$(wc -c < "$MEMORY_FILE")
    if [[ "$SIZE" -gt 8000 ]]; then
        echo "[entrypoint] MEMORY.md is ${SIZE} bytes — trimming to last 4000 chars"
        TAIL=$(tail -c 4000 "$MEMORY_FILE")
        printf '[...older entries trimmed at startup...]\n\n%s' "$TAIL" > "$MEMORY_FILE"
    fi
fi

# ── Start agent ────────────────────────────────────────────────────────────────
exec python -m agent.main start "$@"
