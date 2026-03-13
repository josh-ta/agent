FROM mcr.microsoft.com/playwright/python:v1.50.0-noble

# Install VNC + noVNC + websockify for live browser observation
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    fluxbox \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright MCP server
# --ignore-installed needed because Ubuntu Noble's apt-installed packages
# have no pip RECORD file and block normal upgrades
RUN pip install --no-cache-dir --break-system-packages --ignore-installed \
    "mcp[cli]" "playwright-mcp>=0.0.6" 2>/dev/null || \
    pip install --no-cache-dir --break-system-packages --ignore-installed "mcp[cli]"

# Startup script + fallback MCP bridge
COPY start.sh /start.sh
COPY mcp_bridge.py /mcp_bridge.py
RUN chmod +x /start.sh

EXPOSE 3080 5900 6080

HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=8 \
    CMD curl -sf http://localhost:3080/health || curl -sf http://localhost:3080/ || exit 1

CMD ["/start.sh"]
