FROM mcr.microsoft.com/playwright/python:v1.50.0-noble

# Install VNC + noVNC + websockify for live browser observation
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    fluxbox \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright MCP server
RUN pip install --no-cache-dir "mcp[cli]" "playwright-mcp>=0.0.6" 2>/dev/null || \
    pip install --no-cache-dir "mcp[cli]"

# Startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 3080 5900 6080

HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -sf http://localhost:3080/ || exit 1

CMD ["/start.sh"]
