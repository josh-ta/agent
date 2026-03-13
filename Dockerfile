FROM python:3.12-slim

# System dependencies: git, curl, docker CLI (for self-restart)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    docker.io \
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

# Non-root user for safety (still has docker group access via mounted socket)
RUN groupadd -r agentuser && useradd -r -g agentuser -G root agentuser \
    && chown -R agentuser:agentuser /app /data /workspace

USER agentuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import agent; print('ok')" || exit 1

ENTRYPOINT ["python", "-m", "agent.main"]
