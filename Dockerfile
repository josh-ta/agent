FROM python:3.12-slim

# System dependencies: git, gh CLI, openssh, curl, docker CLI (for self-restart)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    curl \
    ca-certificates \
    docker.io \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
       | gpg --dearmor -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
       > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application source before installing the package so editable installs
# always see the `agent/` package.
COPY pyproject.toml ./
COPY README.md ./
COPY agent/ ./agent/

RUN pip install --no-cache-dir -e .

# Create data/workspace directories (container runs as root — simplest for bind mounts)
RUN mkdir -p /data /workspace

# Entrypoint wrapper: configures git identity + gh auth at runtime from env vars
COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD sh -c 'if [ "${CONTROL_PLANE_ENABLED:-true}" = "true" ]; then curl -sf "http://localhost:${CONTROL_PLANE_PORT:-8000}/healthz" >/dev/null; else python -c "import agent; print(\"ok\")"; fi' || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
