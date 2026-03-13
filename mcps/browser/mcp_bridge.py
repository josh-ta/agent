"""
Playwright MCP bridge — MCP server over HTTP/SSE.

Uses only mcp + anyio (already present via mcp[cli]) plus the stdlib
http.server for the /health endpoint on a separate thread.

Architecture:
  - Thread 1: stdlib HTTPServer on /health (for Docker healthcheck)
  - Thread 2: asyncio event loop running the MCP SSE server on /sse

Both share the same port via a single uvicorn/starlette app if those
packages are available; otherwise falls back to mcp's built-in runner.
"""

from __future__ import annotations

import asyncio
import base64
import re
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── Attempt to import optional deps ───────────────────────────────────────────
try:
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route
    HAS_STARLETTE = True
except ImportError:
    HAS_STARLETTE = False

try:
    from markdownify import markdownify as _md
    HAS_MARKDOWNIFY = True
except ImportError:
    HAS_MARKDOWNIFY = False

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from playwright.async_api import async_playwright

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3080

print(f"[mcp_bridge] Starting on port {PORT} (starlette={'yes' if HAS_STARLETTE else 'no'}, markdownify={'yes' if HAS_MARKDOWNIFY else 'no'})", flush=True)

# ── HTML → Markdown conversion ─────────────────────────────────────────────────
# Tags that add zero information when converted — strip them entirely before md
_STRIP_TAGS = ["script", "style", "noscript", "iframe", "svg", "head"]
_STRIP_RE = re.compile(
    r"<(" + "|".join(_STRIP_TAGS) + r")[\s>].*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Caps — tuned to stay well under 8k tokens even for dense pages
_MD_CHAR_CAP = 12_000   # ~3k tokens after conversion
_HTML_CHAR_CAP = 80_000  # pre-strip cap to avoid huge regex on multi-MB pages


def _html_to_markdown(html: str) -> str:
    """Strip boilerplate and convert HTML to compact Markdown."""
    if len(html) > _HTML_CHAR_CAP:
        html = html[:_HTML_CHAR_CAP]

    # Remove script/style/svg noise before any conversion
    html = _STRIP_RE.sub("", html)

    if HAS_MARKDOWNIFY:
        md = _md(
            html,
            heading_style="ATX",
            bullets="-",
            strip=["a"],        # drop hyperlinks — keep link text, lose URLs
            convert=["p", "h1", "h2", "h3", "h4", "h5", "h6",
                     "li", "ul", "ol", "table", "tr", "td", "th",
                     "strong", "em", "br", "hr"],
        )
    else:
        # Minimal fallback: strip all remaining tags
        md = re.sub(r"<[^>]+>", " ", html)

    # Collapse runs of whitespace / blank lines
    md = _BLANK_LINES_RE.sub("\n\n", md).strip()

    if len(md) > _MD_CHAR_CAP:
        md = md[:_MD_CHAR_CAP] + "\n\n... [page truncated]"

    return md

# ── Playwright singleton ───────────────────────────────────────────────────────

_browser = None
_page = None
_pw = None


async def get_page():
    global _browser, _page, _pw
    try:
        if _pw is None:
            _pw = await async_playwright().start()
        if _browser is None or not _browser.is_connected():
            _browser = await _pw.chromium.launch(headless=False)
            _page = await _browser.new_page()
        elif _page is None:
            _page = await _browser.new_page()
        return _page
    except Exception as e:
        print(f"[mcp_bridge] get_page error: {e}", flush=True)
        raise


# ── MCP Server ─────────────────────────────────────────────────────────────────

mcp = Server("playwright-bridge")


@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="browser_navigate",
            description="Navigate the browser to a URL",
            inputSchema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        ),
        Tool(
            name="browser_screenshot",
            description="Take a screenshot of the current page, returns base64 PNG",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="browser_content",
            description="Get the page content as clean Markdown (scripts/styles stripped). Much more token-efficient than raw HTML. Use this to read page text, listings, tables, etc.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="browser_click",
            description="Click an element by CSS selector",
            inputSchema={
                "type": "object",
                "properties": {"selector": {"type": "string"}},
                "required": ["selector"],
            },
        ),
        Tool(
            name="browser_type",
            description="Type text into an element by CSS selector",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["selector", "text"],
            },
        ),
        Tool(
            name="browser_evaluate",
            description="Execute JavaScript in the browser and return the result",
            inputSchema={
                "type": "object",
                "properties": {"script": {"type": "string"}},
                "required": ["script"],
            },
        ),
    ]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        page = await get_page()
    except Exception as e:
        return [TextContent(type="text", text=f"Browser error: {e}")]

    try:
        if name == "browser_navigate":
            url = arguments["url"]
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return [TextContent(type="text", text=f"Navigated to {url}")]

        elif name == "browser_screenshot":
            png = await page.screenshot(full_page=False)
            b64 = base64.b64encode(png).decode()
            return [TextContent(type="text", text=f"data:image/png;base64,{b64}")]

        elif name == "browser_content":
            html = await page.content()
            md = _html_to_markdown(html)
            return [TextContent(type="text", text=md)]

        elif name == "browser_click":
            await page.click(arguments["selector"], timeout=10000)
            return [TextContent(type="text", text=f"Clicked {arguments['selector']}")]

        elif name == "browser_type":
            await page.fill(arguments["selector"], arguments["text"], timeout=10000)
            return [TextContent(type="text", text=f"Typed into {arguments['selector']}")]

        elif name == "browser_evaluate":
            result = await page.evaluate(arguments["script"])
            return [TextContent(type="text", text=str(result))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Tool error ({name}): {e}\n{traceback.format_exc()}")]


# ── Health endpoint (always available via stdlib, no extra deps) ───────────────

class _HealthHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/health"):
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)


# ── Server startup ─────────────────────────────────────────────────────────────

if HAS_STARLETTE:
    # Full Starlette app: /health + /sse + /messages on one port
    sse_transport = SseServerTransport("/messages")

    async def _handle_sse(request: Request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp.run(streams[0], streams[1], mcp.create_initialization_options())

    async def _handle_health(request: Request):
        return JSONResponse({"status": "ok"})

    app = Starlette(
        routes=[
            Route("/health", _handle_health),
            Route("/sse", _handle_sse),
            Mount("/messages", app=sse_transport.handle_post_message),
        ]
    )

    if __name__ == "__main__":
        print(f"[mcp_bridge] Running full Starlette+uvicorn MCP server on :{PORT}", flush=True)
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")

else:
    # Fallback: run health on PORT, MCP SSE on PORT+1, tell the agent to use PORT+1
    # But we still need /sse on PORT for the agent — run a minimal asyncio HTTP server
    async def _run_mcp_sse():
        """Run MCP over SSE using mcp's built-in asyncio HTTP handler."""
        from mcp.server.sse import SseServerTransport

        transport = SseServerTransport("/messages")

        async def app(scope, receive, send):
            if scope["type"] == "http":
                path = scope.get("path", "")
                if path == "/health":
                    body = b'{"status":"ok"}'
                    await send({"type": "http.response.start", "status": 200,
                                "headers": [[b"content-type", b"application/json"],
                                            [b"content-length", str(len(body)).encode()]]})
                    await send({"type": "http.response.body", "body": body})
                elif path == "/sse":
                    async with transport.connect_sse(scope, receive, send) as streams:
                        await mcp.run(streams[0], streams[1], mcp.create_initialization_options())
                elif path.startswith("/messages"):
                    await transport.handle_post_message(scope, receive, send)
                else:
                    await send({"type": "http.response.start", "status": 404, "headers": []})
                    await send({"type": "http.response.body", "body": b"not found"})

        # Try to use hypercorn or fall back to a basic asyncio server
        try:
            import hypercorn.asyncio
            import hypercorn.config
            config = hypercorn.config.Config()
            config.bind = [f"0.0.0.0:{PORT}"]
            config.loglevel = "WARNING"
            print(f"[mcp_bridge] Running hypercorn MCP server on :{PORT}", flush=True)
            await hypercorn.asyncio.serve(app, config)
        except ImportError:
            # Last resort: just run the health server on PORT and log a warning
            print(f"[mcp_bridge] WARNING: neither uvicorn nor hypercorn available; /sse will not work", flush=True)
            print(f"[mcp_bridge] Only /health will be served (healthcheck will pass but browser tools unavailable)", flush=True)
            server = HTTPServer(("0.0.0.0", PORT), _HealthHandler)
            await asyncio.get_event_loop().run_in_executor(None, server.serve_forever)

    if __name__ == "__main__":
        print(f"[mcp_bridge] Running fallback MCP server on :{PORT}", flush=True)
        asyncio.run(_run_mcp_sse())
