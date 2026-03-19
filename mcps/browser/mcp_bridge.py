"""
Playwright MCP bridge exposed over HTTP/SSE.

When Starlette/uvicorn are available, a single ASGI app serves both `/health`
and the MCP SSE endpoints on one port. Otherwise it falls back to a smaller
async HTTP server that still serves `/health`, `/sse`, and `/messages`.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import os as _os
import re
import sys
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from playwright.async_api import async_playwright

try:
    from agent.secret_store import SecretNotFoundError, SecretStore
except ImportError:
    _SECRET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")

    class SecretNotFoundError(KeyError):
        """Raised when a named secret is missing."""

    class SecretStore:
        """Minimal secret reader fallback for the standalone browser image."""

        def __init__(self, path: Path) -> None:
            self._path = path

        def get(self, name: str) -> str:
            normalized = name.strip()
            if not normalized or not _SECRET_NAME_RE.fullmatch(normalized):
                raise ValueError("Secret name is invalid.")
            if not self._path.exists():
                raise SecretNotFoundError(normalized)
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("Secret store must contain a JSON object.")
            value = raw.get(normalized)
            if not isinstance(value, str):
                raise SecretNotFoundError(normalized)
            return value


@dataclass(frozen=True)
class RuntimeDeps:
    uvicorn: Any | None = None
    Starlette: Any | None = None
    Request: Any | None = None
    JSONResponse: Any | None = None
    Mount: Any | None = None
    Route: Any | None = None
    markdownify: Callable[..., str] | None = None

    @property
    def has_starlette(self) -> bool:
        return all(
            item is not None
            for item in (self.uvicorn, self.Starlette, self.Request, self.JSONResponse, self.Mount, self.Route)
        )

    @property
    def has_markdownify(self) -> bool:
        return self.markdownify is not None


@dataclass(frozen=True)
class BridgeConfig:
    port: int
    proxy_settings: dict[str, str] | None
    locale: str
    timezone: str
    latitude: float
    longitude: float
    viewport: dict[str, int]
    user_agent: str
    agent_secrets_path: str = "/data/agent-secrets.json"


def parse_proxy_settings(
    *,
    proxy_url: str = "",
    proxy_server: str = "",
    proxy_user: str = "",
    proxy_pass: str = "",
) -> dict[str, str] | None:
    proxy_url = proxy_url.strip()
    proxy_server = proxy_server.strip()
    proxy_user = proxy_user.strip()
    proxy_pass = proxy_pass.strip()
    if proxy_url:
        parsed = urlparse(proxy_url)
        server_url = proxy_url
        if parsed.username or parsed.password:
            if parsed.hostname is None:
                raise ValueError(f"Invalid PROXY_URL: missing hostname in {proxy_url!r}")
            netloc = parsed.hostname
            if parsed.port:
                netloc += f":{parsed.port}"
            server_url = parsed._replace(netloc=netloc).geturl()
        settings = {"server": server_url}
        if parsed.username:
            settings["username"] = parsed.username
        if parsed.password:
            settings["password"] = parsed.password
        return settings
    if proxy_server:
        settings = {"server": proxy_server}
        if proxy_user:
            settings["username"] = proxy_user
        if proxy_pass:
            settings["password"] = proxy_pass
        return settings
    return None


def build_config(env: Mapping[str, str], argv: Sequence[str]) -> BridgeConfig:
    port = int(argv[1]) if len(argv) > 1 else 3080
    return BridgeConfig(
        port=port,
        proxy_settings=parse_proxy_settings(
            proxy_url=env.get("PROXY_URL", ""),
            proxy_server=env.get("PROXY_SERVER", ""),
            proxy_user=env.get("PROXY_USERNAME", ""),
            proxy_pass=env.get("PROXY_PASSWORD", ""),
        ),
        agent_secrets_path=env.get("AGENT_SECRETS_PATH", "/data/agent-secrets.json"),
        locale=env.get("BROWSER_LOCALE", "en-US"),
        timezone=env.get("BROWSER_TIMEZONE", "America/New_York"),
        latitude=float(env.get("BROWSER_GEO_LAT", "40.7128")),
        longitude=float(env.get("BROWSER_GEO_LON", "-74.0060")),
        viewport={"width": 1920, "height": 1080},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    )


def load_optional_deps(
    import_module: Callable[[str], Any] = importlib.import_module,
) -> RuntimeDeps:
    uvicorn = Starlette = Request = JSONResponse = Mount = Route = markdownify = None
    try:
        uvicorn = import_module("uvicorn")
        Starlette = import_module("starlette.applications").Starlette
        Request = import_module("starlette.requests").Request
        JSONResponse = import_module("starlette.responses").JSONResponse
        routes = import_module("starlette.routing")
        Mount = routes.Mount
        Route = routes.Route
    except ImportError:
        pass
    try:
        markdownify = import_module("markdownify").markdownify
    except ImportError:
        pass
    return RuntimeDeps(
        uvicorn=uvicorn,
        Starlette=Starlette,
        Request=Request,
        JSONResponse=JSONResponse,
        Mount=Mount,
        Route=Route,
        markdownify=markdownify,
    )


CONFIG = build_config(_os.environ, sys.argv)
DEPS = load_optional_deps()

PORT = CONFIG.port
_PROXY_SETTINGS = CONFIG.proxy_settings
_SECRETS_PATH = CONFIG.agent_secrets_path
_LOCALE = CONFIG.locale
_TIMEZONE = CONFIG.timezone
_LAT = CONFIG.latitude
_LON = CONFIG.longitude
_VIEWPORT = CONFIG.viewport
_USER_AGENT = CONFIG.user_agent
HAS_STARLETTE = DEPS.has_starlette
HAS_MARKDOWNIFY = DEPS.has_markdownify

print(
    f"[mcp_bridge] Starting on port {PORT} "
    f"(starlette={'yes' if HAS_STARLETTE else 'no'}, "
    f"markdownify={'yes' if HAS_MARKDOWNIFY else 'no'}, "
    f"proxy={'yes' if _PROXY_SETTINGS else 'no'}, "
    f"locale={_LOCALE}, tz={_TIMEZONE})",
    flush=True,
)

_STRIP_TAGS = ["script", "style", "noscript", "iframe", "svg", "head"]
_STRIP_RE = re.compile(
    r"<(" + "|".join(_STRIP_TAGS) + r")[\s>].*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_MD_CHAR_CAP = 12_000
_HTML_CHAR_CAP = 80_000


def _html_to_markdown(
    html: str,
    *,
    markdownify_fn: Callable[..., str] | None = None,
    md_char_cap: int = _MD_CHAR_CAP,
    html_char_cap: int = _HTML_CHAR_CAP,
) -> str:
    if len(html) > html_char_cap:
        html = html[:html_char_cap]
    html = _STRIP_RE.sub("", html)
    renderer = DEPS.markdownify if markdownify_fn is None else markdownify_fn
    if renderer is not None:
        md = renderer(
            html,
            heading_style="ATX",
            bullets="-",
            strip=["a"],
        )
    else:
        md = re.sub(r"<[^>]+>", " ", html)
    md = _BLANK_LINES_RE.sub("\n\n", md).strip()
    if len(md) > md_char_cap:
        md = md[:md_char_cap] + "\n\n... [page truncated]"
    return md


def _context_kwargs(config: BridgeConfig) -> dict[str, Any]:
    return {
        "proxy": config.proxy_settings,
        "locale": config.locale,
        "timezone_id": config.timezone,
        "geolocation": {"latitude": config.latitude, "longitude": config.longitude},
        "permissions": ["geolocation"],
        "viewport": config.viewport,
        "user_agent": config.user_agent,
        "color_scheme": "light",
        "extra_http_headers": {
            "Accept-Language": f"{config.locale},{config.locale.split('-')[0]};q=0.9,en;q=0.8",
        },
    }


def _init_script(config: BridgeConfig) -> str:
    return f"""
                Object.defineProperty(navigator, 'webdriver', {{ get: () => undefined }});
                Object.defineProperty(navigator, 'plugins', {{ get: () => [1,2,3,4,5] }});
                Object.defineProperty(navigator, 'languages', {{
                    get: () => ['{config.locale}', '{config.locale.split("-")[0]}']
                }});
            """


_browser = None
_context = None
_page = None
_pw = None


async def get_page(
    *,
    config: BridgeConfig | None = None,
    playwright_factory: Callable[[], Any] = async_playwright,
):
    global _browser, _context, _page, _pw
    config = config or CONFIG
    try:
        if _pw is None:
            _pw = await playwright_factory().start()
        if _browser is None or not _browser.is_connected():
            _browser = await _pw.chromium.launch(
                headless=False,
                proxy=config.proxy_settings,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
            )
            _context = None
            _page = None
        if _context is None:
            _context = await _browser.new_context(**_context_kwargs(config))
            _page = None
        if _page is None:
            _page = await _context.new_page()
            await _page.add_init_script(_init_script(config))
        return _page
    except Exception as exc:
        print(f"[mcp_bridge] get_page error: {exc}", flush=True)
        raise


mcp = Server("playwright-bridge")


def _tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "browser_navigate",
            "description": "Navigate the browser to a URL",
            "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        },
        {
            "name": "browser_screenshot",
            "description": "Take a screenshot of the current page, returns base64 PNG",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "browser_content",
            "description": "Get the page content as clean Markdown (scripts/styles stripped). Much more token-efficient than raw HTML. Use this to read page text, listings, tables, etc.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "browser_click",
            "description": "Click an element by CSS selector",
            "inputSchema": {
                "type": "object",
                "properties": {"selector": {"type": "string"}},
                "required": ["selector"],
            },
        },
        {
            "name": "browser_type",
            "description": "Type text into an element by CSS selector",
            "inputSchema": {
                "type": "object",
                "properties": {"selector": {"type": "string"}, "text": {"type": "string"}},
                "required": ["selector", "text"],
            },
        },
        {
            "name": "browser_evaluate",
            "description": "Execute JavaScript in the browser and return the result",
            "inputSchema": {
                "type": "object",
                "properties": {"script": {"type": "string"}},
                "required": ["script"],
            },
        },
        {
            "name": "browser_snapshot",
            "description": (
                "Return a compact accessibility tree of the current page — element roles, "
                "names, and ref IDs. Much cheaper than screenshots for understanding page "
                "structure before interacting with it. Use this instead of browser_screenshot "
                "when you need to find selectors or understand layout."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "browser_fill",
            "description": "Clear a form field and type new text into it (by CSS selector). Equivalent to clearing then typing.",
            "inputSchema": {
                "type": "object",
                "properties": {"selector": {"type": "string"}, "value": {"type": "string"}},
                "required": ["selector", "value"],
            },
        },
        {
            "name": "browser_fill_secret",
            "description": "Clear a form field and fill it from a stored secret by name.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "secret_name": {"type": "string"},
                },
                "required": ["selector", "secret_name"],
            },
        },
        {
            "name": "browser_type_secret",
            "description": "Type into a form field using a stored secret by name.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "secret_name": {"type": "string"},
                },
                "required": ["selector", "secret_name"],
            },
        },
        {
            "name": "browser_scroll",
            "description": "Scroll the page or a specific element. Use direction='down'/'up'/'left'/'right' and amount in pixels, or supply a selector to scroll an element into view.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["down", "up", "left", "right"],
                        "default": "down",
                    },
                    "amount": {"type": "integer", "description": "Pixels to scroll", "default": 500},
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector — scroll this element into view instead",
                    },
                },
            },
        },
    ]


def _text_result(text: str) -> list[TextContent]:
    return [TextContent(type="text", text=text)]


def _read_secret(secret_name: str) -> str:
    store = SecretStore(Path(_SECRETS_PATH))
    try:
        return store.get(secret_name)
    except SecretNotFoundError as exc:
        raise ValueError(f"Secret not found: {secret_name}") from exc


def format_accessibility_tree(snapshot: dict[str, Any], limit: int = 8000) -> str:
    def _compact(node: dict[str, Any], depth: int = 0) -> str:
        indent = "  " * depth
        role = node.get("role", "")
        name = node.get("name", "")
        label = f"{role}: {name}" if name else role
        lines = [f"{indent}{label}"]
        for child in node.get("children", []):
            lines.append(_compact(child, depth + 1))
        return "\n".join(lines)

    tree = _compact(snapshot)
    if len(tree) > limit:
        tree = tree[:limit] + "\n... [truncated]"
    return tree


def scroll_args_to_delta(direction: str, amount: int) -> tuple[int, int]:
    axis_map = {"down": (0, amount), "up": (0, -amount), "right": (amount, 0), "left": (-amount, 0)}
    return axis_map.get(direction, (0, amount))


async def _tool_navigate(page: Any, arguments: dict[str, Any]) -> str:
    url = arguments["url"]
    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    return f"Navigated to {url}"


async def _tool_screenshot(page: Any, arguments: dict[str, Any]) -> str:
    png = await page.screenshot(full_page=False)
    return f"data:image/png;base64,{base64.b64encode(png).decode()}"


async def _tool_content(page: Any, arguments: dict[str, Any]) -> str:
    return _html_to_markdown(await page.content())


async def _tool_click(page: Any, arguments: dict[str, Any]) -> str:
    await page.click(arguments["selector"], timeout=10000)
    return f"Clicked {arguments['selector']}"


async def _tool_type(page: Any, arguments: dict[str, Any]) -> str:
    await page.fill(arguments["selector"], arguments["text"], timeout=10000)
    return f"Typed into {arguments['selector']}"


async def _tool_evaluate(page: Any, arguments: dict[str, Any]) -> str:
    return str(await page.evaluate(arguments["script"]))


async def _tool_snapshot(page: Any, arguments: dict[str, Any]) -> str:
    snapshot = await page.accessibility.snapshot()
    if snapshot is None:
        return "(accessibility tree is empty)"
    return format_accessibility_tree(snapshot)


async def _tool_fill(page: Any, arguments: dict[str, Any]) -> str:
    selector = arguments["selector"]
    value = arguments.get("value", "")
    await page.fill(selector, value, timeout=10000)
    return f"Filled '{selector}' with provided value"


async def _tool_fill_secret(page: Any, arguments: dict[str, Any]) -> str:
    selector = arguments["selector"]
    secret_name = arguments["secret_name"]
    await page.fill(selector, _read_secret(secret_name), timeout=10000)
    return f"Filled '{selector}' from secret '{secret_name}'"


async def _tool_type_secret(page: Any, arguments: dict[str, Any]) -> str:
    selector = arguments["selector"]
    secret_name = arguments["secret_name"]
    await page.fill(selector, _read_secret(secret_name), timeout=10000)
    return f"Typed into '{selector}' from secret '{secret_name}'"


async def _tool_scroll(page: Any, arguments: dict[str, Any]) -> str:
    selector = arguments.get("selector", "")
    if selector:
        await page.eval_on_selector(
            selector,
            "el => el.scrollIntoView({behavior: 'smooth', block: 'center'})",
        )
        return f"Scrolled '{selector}' into view"
    direction = arguments.get("direction", "down")
    amount = int(arguments.get("amount", 500))
    dx, dy = scroll_args_to_delta(direction, amount)
    await page.evaluate(f"window.scrollBy({dx}, {dy})")
    return f"Scrolled {direction} by {amount}px"


TOOL_HANDLERS: dict[str, Callable[[Any, dict[str, Any]], Any]] = {
    "browser_navigate": _tool_navigate,
    "browser_screenshot": _tool_screenshot,
    "browser_content": _tool_content,
    "browser_click": _tool_click,
    "browser_type": _tool_type,
    "browser_evaluate": _tool_evaluate,
    "browser_snapshot": _tool_snapshot,
    "browser_fill": _tool_fill,
    "browser_fill_secret": _tool_fill_secret,
    "browser_type_secret": _tool_type_secret,
    "browser_scroll": _tool_scroll,
}


@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [Tool(**spec) for spec in _tool_specs()]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        page = await get_page()
    except Exception as exc:
        return _text_result(f"Browser error: {exc}")
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return _text_result(f"Unknown tool: {name}")
    try:
        return _text_result(await handler(page, arguments))
    except Exception as exc:
        return _text_result(f"Tool error ({name}): {exc}\n{traceback.format_exc()}")


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


def build_starlette_app(mcp_server: Server, transport: Any, deps: RuntimeDeps) -> Any:
    async def _handle_sse(request: Any):
        async with transport.connect_sse(request.scope, request.receive, request._send) as streams:
            await mcp_server.run(streams[0], streams[1], mcp_server.create_initialization_options())

    async def _handle_health(request: Any):
        return deps.JSONResponse({"status": "ok"})

    return deps.Starlette(
        routes=[
            deps.Route("/health", _handle_health),
            deps.Route("/sse", _handle_sse),
            deps.Mount("/messages", app=transport.handle_post_message),
        ]
    )


def build_fallback_asgi_app(mcp_server: Server, transport: Any):
    async def app(scope, receive, send):
        if scope["type"] != "http":
            return
        path = scope.get("path", "")
        if path == "/health":
            body = b'{"status":"ok"}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        [b"content-type", b"application/json"],
                        [b"content-length", str(len(body)).encode()],
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
        elif path == "/sse":
            async with transport.connect_sse(scope, receive, send) as streams:
                await mcp_server.run(streams[0], streams[1], mcp_server.create_initialization_options())
        elif path.startswith("/messages"):
            await transport.handle_post_message(scope, receive, send)
        else:
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"not found"})

    return app


def resolve_hypercorn(import_module: Callable[[str], Any] = importlib.import_module) -> tuple[Any | None, Any | None]:
    try:
        serve = import_module("hypercorn.asyncio").serve
        config_cls = import_module("hypercorn.config").Config
        return serve, config_cls
    except ImportError:
        return None, None


async def run_fallback_server(
    *,
    port: int = PORT,
    mcp_server: Server = mcp,
    transport_factory: Callable[[str], Any] = SseServerTransport,
    hypercorn_resolver: Callable[[], tuple[Any | None, Any | None]] = resolve_hypercorn,
    http_server_cls: type[HTTPServer] = HTTPServer,
) -> str:
    transport = transport_factory("/messages")
    app = build_fallback_asgi_app(mcp_server, transport)
    serve, config_cls = hypercorn_resolver()
    if serve is not None and config_cls is not None:
        config = config_cls()
        config.bind = [f"0.0.0.0:{port}"]
        config.loglevel = "WARNING"
        print(f"[mcp_bridge] Running hypercorn MCP server on :{port}", flush=True)
        await serve(app, config)
        return "hypercorn"
    print("[mcp_bridge] WARNING: neither uvicorn nor hypercorn available; /sse will not work", flush=True)
    print("[mcp_bridge] Only /health will be served (healthcheck will pass but browser tools unavailable)", flush=True)
    server = http_server_cls(("0.0.0.0", port), _HealthHandler)
    await asyncio.get_running_loop().run_in_executor(None, server.serve_forever)
    return "health_only"


def build_default_app() -> Any | None:
    if not DEPS.has_starlette:
        return None
    return build_starlette_app(mcp, SseServerTransport("/messages"), DEPS)


app = build_default_app()


def main() -> str:
    if app is not None:
        print(f"[mcp_bridge] Running full Starlette+uvicorn MCP server on :{PORT}", flush=True)
        DEPS.uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
        return "starlette"
    print(f"[mcp_bridge] Running fallback MCP server on :{PORT}", flush=True)
    asyncio.run(run_fallback_server())
    return "fallback"


if __name__ == "__main__":
    main()
