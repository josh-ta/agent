FROM python:3.12-slim

# Install system dependencies: VNC, noVNC, X11, browsers runtime deps, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    fluxbox \
    curl \
    # Chromium runtime dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages — all in one pip call, same interpreter
RUN pip install --no-cache-dir \
    "playwright" \
    "mcp[cli]" \
    "uvicorn[standard]" \
    "starlette" \
    "markdownify>=0.13" \
    && python -m playwright install chromium \
    && python -m playwright install-deps chromium \
    && python -c "import playwright, uvicorn, starlette, mcp; print('deps ok')"

# Startup script + fallback MCP bridge
COPY start.sh /start.sh
COPY mcp_bridge.py /mcp_bridge.py
RUN chmod +x /start.sh

EXPOSE 3080 5900 6080

HEALTHCHECK --interval=20s --timeout=5s --start-period=90s --retries=6 \
    CMD python3 -c "import socket; s=socket.socket(); s.settimeout(3); s.connect(('localhost',3080)); s.close()" || exit 1

CMD ["/start.sh"]
