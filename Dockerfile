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

# Install Python deps first (cache layer)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || pip install --no-cache-dir -e . || true

# Copy application source
COPY agent/ ./agent/

# Install the package properly
RUN pip install --no-cache-dir -e .

# Create data directory for SQLite
RUN mkdir -p /data /workspace

# Entrypoint wrapper: configures git identity + gh auth at runtime from env vars
COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Non-root user for safety (still has docker group access via mounted socket)
# -m creates the home directory so git config --global has somewhere to write
RUN groupadd -r agentuser && useradd -r -m -g agentuser -G root agentuser \
    && chown -R agentuser:agentuser /app /data /workspace

USER agentuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import agent; print('ok')" || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
