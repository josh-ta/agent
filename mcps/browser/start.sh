#!/usr/bin/env bash
# Browser sidecar startup: Xvfb → fluxbox → x11vnc → noVNC → Playwright MCP
set -euo pipefail

DISPLAY_NUM=${DISPLAY_NUM:-99}
VNC_PORT=${VNC_PORT:-5900}
NOVNC_PORT=${NOVNC_PORT:-6080}
MCP_PORT=${MCP_PORT:-3080}

export DISPLAY=":${DISPLAY_NUM}"

echo "[browser] Starting Xvfb on :${DISPLAY_NUM}"
Xvfb ":${DISPLAY_NUM}" -screen 0 1280x900x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Wait for Xvfb to be ready
sleep 1

echo "[browser] Starting fluxbox window manager"
fluxbox -display ":${DISPLAY_NUM}" &>/dev/null &

echo "[browser] Starting x11vnc on port ${VNC_PORT}"
x11vnc \
    -display ":${DISPLAY_NUM}" \
    -nopw \
    -listen 0.0.0.0 \
    -port "${VNC_PORT}" \
    -xkb \
    -forever \
    -shared \
    -quiet \
    &

echo "[browser] Starting noVNC websockify on port ${NOVNC_PORT}"
websockify \
    --web /usr/share/novnc \
    "${NOVNC_PORT}" \
    "localhost:${VNC_PORT}" \
    &

echo "[browser] Starting Playwright MCP server on port ${MCP_PORT}"
# Try playwright-mcp package first, fall back to simple HTTP health endpoint
if python -c "import playwright_mcp" 2>/dev/null; then
    python -m playwright_mcp.server \
        --port "${MCP_PORT}" \
        --host "0.0.0.0" \
        --browser chromium \
        --headless false \
        &
else
    echo "[browser] playwright-mcp not found; starting minimal MCP bridge"
    python /mcp_bridge.py "${MCP_PORT}" &
fi

echo "[browser] All services started. VNC=${VNC_PORT} noVNC=${NOVNC_PORT} MCP=${MCP_PORT}"

# Wait for background processes
wait $XVFB_PID
