from __future__ import annotations

import asyncio
import importlib.util
import itertools
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace


_COUNTER = itertools.count()
_BRIDGE_PATH = Path("/Users/josh/Projects/agent/mcps/browser/mcp_bridge.py")


@dataclass
class _FakeTextContent:
    type: str
    text: str


@dataclass
class _FakeTool:
    name: str
    description: str
    inputSchema: dict


class _FakeServer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.run_calls: list[tuple[object, object, dict]] = []

    def list_tools(self):
        def decorator(fn):
            return fn

        return decorator

    def call_tool(self):
        def decorator(fn):
            return fn

        return decorator

    async def run(self, *args, **kwargs) -> None:
        self.run_calls.append((args[0], args[1], args[2]))

    def create_initialization_options(self) -> dict:
        return {"initialized": True}


class _FakeSseServerTransport:
    def __init__(self, path: str) -> None:
        self.path = path
        self.post_calls: list[tuple[object, object, object]] = []
        self.handle_post_message = self._handle_post_message
        self.connect_calls: list[tuple[object, object, object]] = []

    class _Context:
        def __init__(self, outer) -> None:
            self.outer = outer

        async def __aenter__(self):
            return ("read-stream", "write-stream")

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    def connect_sse(self, *args, **kwargs):
        self.connect_calls.append((args[0], args[1], args[2]))
        return self._Context(self)

    async def _handle_post_message(self, scope, receive, send) -> None:
        self.post_calls.append((scope, receive, send))


def _load_bridge(monkeypatch, *, env: dict[str, str] | None = None):
    env = env or {}
    for key in (
        "PROXY_URL",
        "PROXY_SERVER",
        "PROXY_USERNAME",
        "PROXY_PASSWORD",
        "BROWSER_LOCALE",
        "BROWSER_TIMEZONE",
        "BROWSER_GEO_LAT",
        "BROWSER_GEO_LON",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    monkeypatch.setattr(sys, "argv", ["mcp_bridge.py"])

    server_module = ModuleType("mcp.server")
    server_module.Server = _FakeServer

    sse_module = ModuleType("mcp.server.sse")
    sse_module.SseServerTransport = _FakeSseServerTransport

    types_module = ModuleType("mcp.types")
    types_module.TextContent = _FakeTextContent
    types_module.Tool = _FakeTool

    mcp_module = ModuleType("mcp")
    mcp_module.server = server_module
    mcp_module.types = types_module

    playwright_async_api = ModuleType("playwright.async_api")
    playwright_async_api.async_playwright = lambda: SimpleNamespace(start=lambda: None)
    playwright_module = ModuleType("playwright")
    playwright_module.async_api = playwright_async_api

    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)
    monkeypatch.setitem(sys.modules, "mcp.server.sse", sse_module)
    monkeypatch.setitem(sys.modules, "mcp.types", types_module)
    monkeypatch.setitem(sys.modules, "playwright", playwright_module)
    monkeypatch.setitem(sys.modules, "playwright.async_api", playwright_async_api)

    module_name = f"test_mcp_bridge_{next(_COUNTER)}"
    spec = importlib.util.spec_from_file_location(module_name, _BRIDGE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_proxy_settings_and_build_config(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)

    assert bridge.parse_proxy_settings(proxy_url="http://user:pass@example.com:8080") == {
        "server": "http://example.com:8080",
        "username": "user",
        "password": "pass",
    }
    assert bridge.parse_proxy_settings(
        proxy_server="proxy.local:9000",
        proxy_user="alice",
        proxy_pass="secret",
    ) == {"server": "proxy.local:9000", "username": "alice", "password": "secret"}
    assert bridge.parse_proxy_settings(proxy_url="http://proxy.example:8080") == {
        "server": "http://proxy.example:8080"
    }
    assert bridge.parse_proxy_settings(proxy_url="http://user@proxy.example") == {
        "server": "http://proxy.example",
        "username": "user",
    }

    config = bridge.build_config(
        {
            "PROXY_SERVER": "proxy.local:9000",
            "BROWSER_LOCALE": "fr-FR",
            "BROWSER_TIMEZONE": "Europe/Paris",
            "BROWSER_GEO_LAT": "48.8",
            "BROWSER_GEO_LON": "2.3",
        },
        ["mcp_bridge.py", "4001"],
    )

    assert config.port == 4001
    assert config.proxy_settings == {"server": "proxy.local:9000"}
    assert config.locale == "fr-FR"
    assert config.timezone == "Europe/Paris"
    assert config.latitude == 48.8
    assert config.longitude == 2.3

    try:
        bridge.parse_proxy_settings(proxy_url="http://user:pass@:8080")
    except ValueError as exc:
        assert "missing hostname" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid proxy URL")


def test_load_optional_deps_handles_present_and_missing_modules(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)

    class _Routes:
        Mount = object()
        Route = object()

    modules = {
        "uvicorn": object(),
        "starlette.applications": SimpleNamespace(Starlette=object()),
        "starlette.requests": SimpleNamespace(Request=object()),
        "starlette.responses": SimpleNamespace(JSONResponse=object()),
        "starlette.routing": _Routes,
        "markdownify": SimpleNamespace(markdownify=lambda *args, **kwargs: "ok"),
    }

    deps = bridge.load_optional_deps(import_module=modules.__getitem__)
    assert deps.has_starlette is True
    assert deps.has_markdownify is True

    def _missing(name: str):
        raise ImportError(name)

    missing = bridge.load_optional_deps(import_module=_missing)
    assert missing.has_starlette is False
    assert missing.has_markdownify is False


def test_html_markdown_and_format_helpers(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)

    text = bridge._html_to_markdown(
        "<head>secret</head><script>bad()</script><p>Hello world</p>",
        markdownify_fn=lambda html, **kwargs: "Hello world " * 3,
        md_char_cap=10,
    )
    tree = bridge.format_accessibility_tree(
        {"role": "document", "children": [{"role": "button", "name": "Run"}]},
        limit=20,
    )

    assert "secret" not in text
    assert "bad()" not in text
    assert text.endswith("... [page truncated]")
    assert tree.endswith("... [truncated]")
    assert bridge.scroll_args_to_delta("left", 20) == (-20, 0)
    assert bridge.scroll_args_to_delta("weird", 20) == (0, 20)
    assert "Hello" in bridge._html_to_markdown("<p>Hello</p>", markdownify_fn=None)


def test_html_markdown_uses_regex_fallback_when_markdownify_missing(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)
    monkeypatch.setattr(bridge, "DEPS", SimpleNamespace(markdownify=None))

    text = bridge._html_to_markdown(
        "<p>Hello</p><script>bad()</script>",
        html_char_cap=7,
        markdownify_fn=None,
    )

    assert text == "Hell"
    assert "bad()" not in text


def test_context_and_init_helpers(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)
    config = bridge.BridgeConfig(
        port=1,
        proxy_settings={"server": "proxy.local"},
        locale="fr-FR",
        timezone="Europe/Paris",
        latitude=48.8,
        longitude=2.3,
        viewport={"width": 100, "height": 200},
        user_agent="UA",
    )

    kwargs = bridge._context_kwargs(config)
    script = bridge._init_script(config)

    assert kwargs["proxy"] == {"server": "proxy.local"}
    assert kwargs["extra_http_headers"]["Accept-Language"].startswith("fr-FR,fr")
    assert kwargs["viewport"] == {"width": 100, "height": 200}
    assert "'fr-FR'" in script
    assert "'fr'" in script


def test_list_tools_exposes_expected_browser_tool_names(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)

    tools = asyncio.run(bridge.list_tools())
    names = {tool.name for tool in tools}

    assert {
        "browser_navigate",
        "browser_screenshot",
        "browser_content",
        "browser_click",
        "browser_type",
        "browser_evaluate",
        "browser_snapshot",
        "browser_fill",
        "browser_scroll",
    } <= names


def test_call_tool_covers_dispatch_unknown_and_errors(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)

    class _FakeAccessibility:
        async def snapshot(self):
            return {"role": "document", "children": [{"role": "button", "name": "Run"}]}

    class _FakePage:
        def __init__(self) -> None:
            self.accessibility = _FakeAccessibility()
            self.goto_calls: list[tuple[str, str, int]] = []
            self.fill_calls: list[tuple[str, str, int]] = []
            self.evaluate_calls: list[str] = []
            self.selector_calls: list[tuple[str, str]] = []

        async def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
            self.goto_calls.append((url, wait_until, timeout))

        async def screenshot(self, *, full_page: bool) -> bytes:
            assert full_page is False
            return b"png"

        async def content(self) -> str:
            return "<p>Hello</p>"

        async def click(self, selector: str, *, timeout: int) -> None:
            self.fill_calls.append((selector, "click", timeout))

        async def fill(self, selector: str, value: str, *, timeout: int) -> None:
            self.fill_calls.append((selector, value, timeout))

        async def evaluate(self, script: str):
            self.evaluate_calls.append(script)
            return "eval-result"

        async def eval_on_selector(self, selector: str, script: str) -> None:
            self.selector_calls.append((selector, script))

    page = _FakePage()

    async def get_page():
        return page

    async def fail_get_page():
        raise RuntimeError("browser unavailable")

    monkeypatch.setattr(bridge, "get_page", get_page)

    navigate = asyncio.run(bridge.call_tool("browser_navigate", {"url": "https://example.com"}))
    screenshot = asyncio.run(bridge.call_tool("browser_screenshot", {}))
    content = asyncio.run(bridge.call_tool("browser_content", {}))
    click = asyncio.run(bridge.call_tool("browser_click", {"selector": "#run"}))
    type_result = asyncio.run(bridge.call_tool("browser_type", {"selector": "#name", "text": "Josh"}))
    evaluate = asyncio.run(bridge.call_tool("browser_evaluate", {"script": "1+1"}))
    snapshot = asyncio.run(bridge.call_tool("browser_snapshot", {}))
    fill = asyncio.run(bridge.call_tool("browser_fill", {"selector": "#name", "value": "Updated"}))
    scroll_selector = asyncio.run(bridge.call_tool("browser_scroll", {"selector": "#target"}))
    scroll_direction = asyncio.run(bridge.call_tool("browser_scroll", {"direction": "up", "amount": 20}))
    unknown = asyncio.run(bridge.call_tool("unknown_tool", {}))

    monkeypatch.setattr(bridge, "get_page", fail_get_page)
    browser_error = asyncio.run(bridge.call_tool("browser_content", {}))

    assert navigate[0].text == "Navigated to https://example.com"
    assert screenshot[0].text.startswith("data:image/png;base64,")
    assert "Hello" in content[0].text
    assert click[0].text == "Clicked #run"
    assert type_result[0].text == "Typed into #name"
    assert evaluate[0].text == "eval-result"
    assert "button: Run" in snapshot[0].text
    assert fill[0].text == "Filled '#name' with provided value"
    assert scroll_selector[0].text == "Scrolled '#target' into view"
    assert scroll_direction[0].text == "Scrolled up by 20px"
    assert unknown[0].text == "Unknown tool: unknown_tool"
    assert browser_error[0].text == "Browser error: browser unavailable"
    assert page.goto_calls == [("https://example.com", "domcontentloaded", 30000)]
    assert page.selector_calls == [("#target", "el => el.scrollIntoView({behavior: 'smooth', block: 'center'})")]
    assert page.evaluate_calls[-1] == "window.scrollBy(0, -20)"


def test_call_tool_snapshot_handles_empty_accessibility_tree(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)

    class _FakeAccessibility:
        async def snapshot(self):
            return None

    class _FakePage:
        def __init__(self) -> None:
            self.accessibility = _FakeAccessibility()

    async def get_page():
        return _FakePage()

    monkeypatch.setattr(bridge, "get_page", get_page)

    result = asyncio.run(bridge.call_tool("browser_snapshot", {}))

    assert result[0].text == "(accessibility tree is empty)"


def test_call_tool_returns_tool_error_when_handler_raises(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)

    class _FakePage:
        async def content(self) -> str:
            raise RuntimeError("content failed")

    async def get_page():
        return _FakePage()

    monkeypatch.setattr(bridge, "get_page", get_page)

    result = asyncio.run(bridge.call_tool("browser_content", {}))

    assert "Tool error (browser_content): content failed" in result[0].text


def test_get_page_initializes_and_reuses_singletons(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)
    calls = {"launch": 0, "context": 0, "page": 0, "scripts": 0}

    class _FakePage:
        async def add_init_script(self, script: str) -> None:
            calls["scripts"] += 1
            assert "Object.defineProperty(navigator, 'webdriver'" in script
            assert "'en-US'" in script

    class _FakeContext:
        async def new_page(self):
            calls["page"] += 1
            return _FakePage()

    class _FakeBrowser:
        def is_connected(self) -> bool:
            return True

        async def new_context(self, **kwargs):
            calls["context"] += 1
            assert kwargs["locale"] == "en-US"
            assert kwargs["timezone_id"] == "America/New_York"
            return _FakeContext()

    class _FakePlaywright:
        def __init__(self) -> None:
            self.chromium = self

        async def launch(self, **kwargs):
            calls["launch"] += 1
            assert kwargs["headless"] is False
            assert kwargs["proxy"] is None
            return _FakeBrowser()

    class _FakeStarter:
        async def start(self):
            return _FakePlaywright()

    first = asyncio.run(bridge.get_page(playwright_factory=lambda: _FakeStarter()))
    second = asyncio.run(bridge.get_page(playwright_factory=lambda: _FakeStarter()))

    assert first is second
    assert calls == {"launch": 1, "context": 1, "page": 1, "scripts": 1}


def test_get_page_logs_and_raises_when_playwright_start_fails(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)
    seen: list[str] = []

    class _BrokenStarter:
        async def start(self):
            raise RuntimeError("start failed")

    monkeypatch.setattr("builtins.print", lambda message, flush=True: seen.append(message))

    try:
        asyncio.run(bridge.get_page(playwright_factory=lambda: _BrokenStarter()))
    except RuntimeError as exc:
        assert str(exc) == "start failed"
    else:
        raise AssertionError("expected get_page to raise")

    assert any("get_page error: start failed" in line for line in seen)


def test_build_fallback_asgi_app_routes_health_sse_messages_and_404(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)
    server = _FakeServer("bridge")
    transport = _FakeSseServerTransport("/messages")
    app = bridge.build_fallback_asgi_app(server, transport)

    async def _invoke(path: str) -> list[dict]:
        sent: list[dict] = []

        async def _send(message: dict) -> None:
            sent.append(message)

        async def _receive() -> dict:
            return {}

        await app({"type": "http", "path": path}, _receive, _send)
        return sent

    health = asyncio.run(_invoke("/health"))
    missing = asyncio.run(_invoke("/missing"))
    asyncio.run(_invoke("/sse"))
    asyncio.run(_invoke("/messages/1"))

    assert health[0]["status"] == 200
    assert health[1]["body"] == b'{"status":"ok"}'
    assert missing[0]["status"] == 404
    assert missing[1]["body"] == b"not found"
    assert server.run_calls == [("read-stream", "write-stream", {"initialized": True})]
    assert len(transport.post_calls) == 1


def test_build_fallback_asgi_app_ignores_non_http_scope(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)
    server = _FakeServer("bridge")
    transport = _FakeSseServerTransport("/messages")
    app = bridge.build_fallback_asgi_app(server, transport)
    sent: list[dict] = []

    async def _send(message: dict) -> None:
        sent.append(message)

    async def _receive() -> dict:
        return {}

    asyncio.run(app({"type": "websocket", "path": "/sse"}, _receive, _send))

    assert sent == []
    assert transport.connect_calls == []
    assert transport.post_calls == []
    assert server.run_calls == []


def test_build_starlette_app_uses_routes_and_mount(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)
    server = _FakeServer("bridge")
    transport = _FakeSseServerTransport("/messages")
    recorded: dict[str, object] = {}

    class _FakeRoute:
        def __init__(self, path: str, endpoint) -> None:
            recorded.setdefault("routes", []).append((path, endpoint))

    class _FakeMount:
        def __init__(self, path: str, *, app) -> None:
            recorded["mount"] = (path, app)

    class _FakeStarlette:
        def __init__(self, *, routes) -> None:
            recorded["starlette_routes"] = routes

    deps = bridge.RuntimeDeps(
        uvicorn=SimpleNamespace(run=lambda *args, **kwargs: None),
        Starlette=_FakeStarlette,
        Request=object(),
        JSONResponse=lambda payload: payload,
        Mount=_FakeMount,
        Route=_FakeRoute,
        markdownify=None,
    )

    app = bridge.build_starlette_app(server, transport, deps)

    assert isinstance(app, _FakeStarlette)
    assert [path for path, _ in recorded["routes"]] == ["/health", "/sse"]
    assert recorded["mount"] == ("/messages", transport.handle_post_message)

    health_endpoint = recorded["routes"][0][1]
    sse_endpoint = recorded["routes"][1][1]
    request = SimpleNamespace(
        scope={"type": "http", "path": "/sse"},
        receive=lambda: None,
        _send=lambda message=None: None,
    )

    assert asyncio.run(health_endpoint(SimpleNamespace())) == {"status": "ok"}
    asyncio.run(sse_endpoint(request))
    assert transport.connect_calls == [(
        {"type": "http", "path": "/sse"},
        request.receive,
        request._send,
    )]
    assert server.run_calls == [("read-stream", "write-stream", {"initialized": True})]


def test_build_default_app_and_resolve_hypercorn(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)
    monkeypatch.setattr(bridge, "DEPS", bridge.RuntimeDeps())
    assert bridge.build_default_app() is None

    monkeypatch.setattr(
        bridge,
        "DEPS",
        bridge.RuntimeDeps(
            uvicorn=object(),
            Starlette=lambda **kwargs: ("starlette", kwargs),
            Request=object(),
            JSONResponse=lambda payload: payload,
            Mount=lambda path, app=None: ("mount", path, app),
            Route=lambda path, endpoint: ("route", path, endpoint),
            markdownify=None,
        ),
    )
    app = bridge.build_default_app()
    assert app[0] == "starlette"

    modules = {
        "hypercorn.asyncio": SimpleNamespace(serve="serve"),
        "hypercorn.config": SimpleNamespace(Config="Config"),
    }
    assert bridge.resolve_hypercorn(import_module=modules.__getitem__) == ("serve", "Config")

    def _missing(name: str):
        raise ImportError(name)

    assert bridge.resolve_hypercorn(import_module=_missing) == (None, None)


def test_health_handler_serves_health_and_404(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)
    events: list[tuple[str, object]] = []

    handler = object.__new__(bridge._HealthHandler)
    handler.path = "/health"
    handler.wfile = SimpleNamespace(write=lambda body: events.append(("body", body)))
    handler.send_response = lambda status: events.append(("status", status))
    handler.send_header = lambda key, value: events.append((key, value))
    handler.end_headers = lambda: events.append(("end", None))
    handler.send_error = lambda status: events.append(("error", status))
    handler.log_message("ignored")
    handler.do_GET()

    missing = object.__new__(bridge._HealthHandler)
    missing.path = "/missing"
    missing.wfile = SimpleNamespace(write=lambda body: None)
    missing.send_response = lambda status: None
    missing.send_header = lambda key, value: None
    missing.end_headers = lambda: None
    missing.send_error = lambda status: events.append(("error", status))
    missing.do_GET()

    assert ("status", 200) in events
    assert ("body", b'{"status":"ok"}') in events
    assert ("error", 404) in events


def test_run_fallback_server_uses_hypercorn_or_health_only(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)
    events: list[tuple[str, object]] = []

    async def _serve(app, config) -> None:
        events.append(("serve", config.bind))

    class _Config:
        def __init__(self) -> None:
            self.bind = []
            self.loglevel = ""

    mode = asyncio.run(
        bridge.run_fallback_server(
            port=4567,
            transport_factory=_FakeSseServerTransport,
            hypercorn_resolver=lambda: (_serve, _Config),
        )
    )

    assert mode == "hypercorn"
    assert events == [("serve", ["0.0.0.0:4567"])]

    class _FakeHealthServer:
        def __init__(self, address, handler) -> None:
            self.address = address
            self.handler = handler

        def serve_forever(self) -> None:
            events.append(("health", self.address))

    async def _run_health_only() -> str:
        loop = asyncio.get_running_loop()
        original = loop.run_in_executor

        async def _wrapped(executor, fn, *args):
            fn(*args)
            return None

        loop.run_in_executor = _wrapped  # type: ignore[assignment]
        try:
            return await bridge.run_fallback_server(
                port=7654,
                transport_factory=_FakeSseServerTransport,
                hypercorn_resolver=lambda: (None, None),
                http_server_cls=_FakeHealthServer,
            )
        finally:
            loop.run_in_executor = original  # type: ignore[assignment]

    health_mode = asyncio.run(_run_health_only())
    assert health_mode == "health_only"
    assert events[-1] == ("health", ("0.0.0.0", 7654))


def test_main_selects_starlette_or_fallback(monkeypatch) -> None:
    bridge = _load_bridge(monkeypatch)
    calls: list[str] = []

    class _Uvicorn:
        @staticmethod
        def run(app, **kwargs) -> None:
            calls.append(f"uvicorn:{kwargs['port']}")

    monkeypatch.setattr(bridge, "app", object())
    monkeypatch.setattr(
        bridge,
        "DEPS",
        bridge.RuntimeDeps(
            uvicorn=_Uvicorn,
            Starlette=object(),
            Request=object(),
            JSONResponse=object(),
            Mount=object(),
            Route=object(),
            markdownify=None,
        ),
    )
    monkeypatch.setattr(bridge, "PORT", 4321)
    assert bridge.main() == "starlette"
    assert calls == ["uvicorn:4321"]

    monkeypatch.setattr(bridge, "app", None)

    def _fake_asyncio_run(coro) -> None:
        coro.close()
        calls.append("fallback-run")

    monkeypatch.setattr(bridge.asyncio, "run", _fake_asyncio_run)
    assert bridge.main() == "fallback"
    assert calls[-1] == "fallback-run"


def test_mcp_bridge_main_guard_invokes_main(monkeypatch) -> None:
    server_module = ModuleType("mcp.server")
    server_module.Server = _FakeServer

    sse_module = ModuleType("mcp.server.sse")
    sse_module.SseServerTransport = _FakeSseServerTransport

    types_module = ModuleType("mcp.types")
    types_module.TextContent = _FakeTextContent
    types_module.Tool = _FakeTool

    mcp_module = ModuleType("mcp")
    mcp_module.server = server_module
    mcp_module.types = types_module

    playwright_async_api = ModuleType("playwright.async_api")
    playwright_async_api.async_playwright = lambda: SimpleNamespace(start=lambda: None)
    playwright_module = ModuleType("playwright")
    playwright_module.async_api = playwright_async_api

    calls: list[tuple[str, object, int]] = []
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)
    monkeypatch.setitem(sys.modules, "mcp.server.sse", sse_module)
    monkeypatch.setitem(sys.modules, "mcp.types", types_module)
    monkeypatch.setitem(sys.modules, "playwright", playwright_module)
    monkeypatch.setitem(sys.modules, "playwright.async_api", playwright_async_api)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda app, **kwargs: calls.append(("uvicorn", app, kwargs["port"]))))
    monkeypatch.setitem(
        sys.modules,
        "starlette.applications",
        SimpleNamespace(Starlette=lambda **kwargs: ("starlette", kwargs)),
    )
    monkeypatch.setitem(sys.modules, "starlette.requests", SimpleNamespace(Request=object()))
    monkeypatch.setitem(sys.modules, "starlette.responses", SimpleNamespace(JSONResponse=lambda payload: payload))
    monkeypatch.setitem(
        sys.modules,
        "starlette.routing",
        SimpleNamespace(
            Mount=lambda *args, **kwargs: ("mount", args, kwargs),
            Route=lambda *args, **kwargs: ("route", args, kwargs),
        ),
    )
    monkeypatch.setattr(sys, "argv", ["mcp_bridge.py"])

    runpy.run_path(str(_BRIDGE_PATH), run_name="__main__")

    assert calls and calls[0][0] == "uvicorn"
