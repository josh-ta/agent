"""
Playwright MCP bridge — real MCP server over HTTP/SSE.

Exposes Playwright browser automation as MCP tools:
  - browser_navigate   : go to a URL
  - browser_screenshot : capture the current page as a base64 PNG
  - browser_content    : get the full page HTML
  - browser_click      : click an element by CSS selector
  - browser_type       : type text into an element
  - browser_evaluate   : run JavaScript on the page

Listens on 0.0.0.0:<port> (default 3080).
GET /health  → {"status":"ok"}  (used by the Docker healthcheck)
GET /sse     → MCP SSE stream   (used by pydantic-ai MCPServerHTTP)
POST /messages → MCP message endpoint
"""

from __future__ import annotations

import asyncio
import base64
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from playwright.async_api import Browser, Page, async_playwright
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route


# ── Playwright singleton ───────────────────────────────────────────────────────

_browser: Browser | None = None
_page: Page | None = None
_playwright_ctx = None


async def get_page() -> Page:
    global _browser, _page, _playwright_ctx
    if _page is None or _browser is None or not _browser.is_connected():
        if _playwright_ctx is None:
            _playwright_ctx = await async_playwright().start()
        _browser = await _playwright_ctx.chromium.launch(headless=False)
        _page = await _browser.new_page()
    return _page


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
                "properties": {"url": {"type": "string", "description": "URL to navigate to"}},
                "required": ["url"],
            },
        ),
        Tool(
            name="browser_screenshot",
            description="Take a screenshot of the current page. Returns a base64-encoded PNG.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="browser_content",
            description="Get the full HTML content of the current page",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="browser_click",
            description="Click an element on the page by CSS selector",
            inputSchema={
                "type": "object",
                "properties": {"selector": {"type": "string", "description": "CSS selector"}},
                "required": ["selector"],
            },
        ),
        Tool(
            name="browser_type",
            description="Type text into an input element identified by CSS selector",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector"},
                    "text": {"type": "string", "description": "Text to type"},
                },
                "required": ["selector", "text"],
            },
        ),
        Tool(
            name="browser_evaluate",
            description="Execute JavaScript in the browser and return the result",
            inputSchema={
                "type": "object",
                "properties": {"script": {"type": "string", "description": "JavaScript to execute"}},
                "required": ["script"],
            },
        ),
    ]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    page = await get_page()

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
        # Truncate very large pages
        if len(html) > 50_000:
            html = html[:50_000] + "\n... [truncated]"
        return [TextContent(type="text", text=html)]

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


# ── Starlette app (SSE transport + health endpoint) ───────────────────────────

sse_transport = SseServerTransport("/messages")


async def handle_sse(request: Request):
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await mcp.run(streams[0], streams[1], mcp.create_initialization_options())


async def handle_health(request: Request):
    return JSONResponse({"status": "ok"})


app = Starlette(
    routes=[
        Route("/health", handle_health),
        Route("/sse", handle_sse),
        Mount("/messages", app=sse_transport.handle_post_message),
    ]
)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3080
    print(f"[mcp_bridge] MCP SSE server listening on :{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
