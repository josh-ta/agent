#!/usr/bin/env bash
# Browser sidecar startup: Xvfb → fluxbox → x11vnc → noVNC → Playwright MCP
# Note: no set -e here — background processes may exit/restart without killing the script

DISPLAY_NUM=${DISPLAY_NUM:-99}
VNC_PORT=${VNC_PORT:-5900}
NOVNC_PORT=${NOVNC_PORT:-6080}
MCP_PORT=${MCP_PORT:-3080}

export DISPLAY=":${DISPLAY_NUM}"

echo "[browser] Starting Xvfb on :${DISPLAY_NUM}"
Xvfb ":${DISPLAY_NUM}" -screen 0 1280x900x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Wait for Xvfb to be ready
sleep 2

echo "[browser] Starting fluxbox window manager"
fluxbox -display ":${DISPLAY_NUM}" &>/dev/null &

echo "[browser] Starting x11vnc on port ${VNC_PORT}"
x11vnc \
    -display ":${DISPLAY_NUM}" \
    -nopw \
    -listen 0.0.0.0 \
    -rfbport "${VNC_PORT}" \
    -xkb \
    -forever \
    -shared \
    -quiet \
    &

echo "[browser] Starting noVNC websockify on port ${NOVNC_PORT}"
# noVNC static files live at /usr/share/novnc; the entry point is vnc.html.
# Create a symlink so that hitting / redirects to vnc.html automatically.
if [[ ! -f /usr/share/novnc/index.html ]]; then
    ln -sf /usr/share/novnc/vnc.html /usr/share/novnc/index.html
fi
websockify \
    --web /usr/share/novnc \
    "${NOVNC_PORT}" \
    "localhost:${VNC_PORT}" \
    &

echo "[browser] Starting MCP HTTP bridge on port ${MCP_PORT}"
python /mcp_bridge.py "${MCP_PORT}" 2>&1 &
MCP_PID=$!

# Give the bridge a moment to start and verify it bound the port
sleep 3
if ! kill -0 $MCP_PID 2>/dev/null; then
    echo "[browser] ERROR: mcp_bridge.py exited immediately — check logs above"
else
    echo "[browser] MCP bridge running (pid=$MCP_PID)"
fi

echo "[browser] All services started. VNC=${VNC_PORT} noVNC=${NOVNC_PORT} MCP=${MCP_PORT}"

# Keep container alive — wait on Xvfb, restart it if it dies
while true; do
    if ! kill -0 $XVFB_PID 2>/dev/null; then
        echo "[browser] Xvfb died, restarting..."
        Xvfb ":${DISPLAY_NUM}" -screen 0 1280x900x24 -ac +extension GLX +render -noreset &
        XVFB_PID=$!
    fi
    sleep 10
done
