"""
Minimal Playwright MCP bridge (fallback when playwright-mcp package is unavailable).

Exposes a tiny HTTP server on the given port that:
  - GET  /         → health check (200 OK)
  - GET  /health   → health check (200 OK)
  - POST /tools    → execute a Playwright command (JSON body)

This is a simplified SSE-less version; real deployments should use the
official playwright-mcp package.
"""

from __future__ import annotations

import asyncio
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from playwright.async_api import async_playwright


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        pass  # suppress default HTTP logs

    def do_GET(self) -> None:
        if self.path in ("/", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
            result = asyncio.run(_execute_playwright(payload))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"result": result}).encode())
        except Exception as exc:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode())


async def _execute_playwright(payload: dict) -> str:
    """Execute a single Playwright action."""
    action = payload.get("action", "")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        if action == "navigate":
            await page.goto(payload.get("url", "about:blank"))
            return f"Navigated to {payload.get('url')}"
        elif action == "screenshot":
            img = await page.screenshot()
            return f"Screenshot taken ({len(img)} bytes)"
        elif action == "content":
            return await page.content()
        else:
            return f"Unknown action: {action}"

        await browser.close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3080
    print(f"[mcp_bridge] Listening on :{port}")
    server = HTTPServer(("0.0.0.0", port), _Handler)
    server.serve_forever()
