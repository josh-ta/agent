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
    "mcp[cli]" "uvicorn" "starlette" 2>/dev/null || \
    pip install --no-cache-dir --break-system-packages --ignore-installed "mcp[cli]" "uvicorn" "starlette"

# Startup script + fallback MCP bridge
COPY start.sh /start.sh
COPY mcp_bridge.py /mcp_bridge.py
RUN chmod +x /start.sh

EXPOSE 3080 5900 6080

HEALTHCHECK --interval=20s --timeout=5s --start-period=90s --retries=6 \
    CMD python3 -c "import socket; s=socket.socket(); s.settimeout(3); s.connect(('localhost',3080)); s.close()" || exit 1

CMD ["/start.sh"]
