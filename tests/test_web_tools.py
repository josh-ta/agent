from __future__ import annotations

import json
from types import SimpleNamespace

import asyncio

import pytest

from agent.config import settings
from agent.core_services import ModelFactory
from agent.tools import filesystem, http_client, web_search


@pytest.mark.asyncio
async def test_web_search_requires_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "web_search_provider", "")
    result = await web_search.web_search("hello")
    assert "WEB_SEARCH_PROVIDER" in result


@pytest.mark.asyncio
async def test_http_request_requires_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "http_allowed_hosts", "")
    result = await http_client.http_request("GET", "https://example.com/")
    assert "HTTP_ALLOWED_HOSTS" in result


@pytest.mark.asyncio
async def test_http_request_invalid_method(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "http_allowed_hosts", "api.example.com")
    result = await http_client.http_request("NOPE", "https://api.example.com/")
    assert "unsupported method" in result


@pytest.mark.asyncio
async def test_http_request_rejects_non_http_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "http_allowed_hosts", "example.com")
    result = await http_client.http_request("GET", "ftp://example.com/file")
    assert "only http/https" in result


@pytest.mark.asyncio
async def test_http_request_rejects_disallowed_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "http_allowed_hosts", "api.example.com")
    result = await http_client.http_request("GET", "https://evil.test/")
    assert "host not allowed" in result


@pytest.mark.asyncio
async def test_http_request_handles_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "http_allowed_hosts", "api.example.com")

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def request(self, *args, **kwargs):
            raise http_client.httpx.HTTPError("boom")

    monkeypatch.setattr(http_client.httpx, "AsyncClient", lambda **kwargs: _Client())
    result = await http_client.http_request("GET", "https://api.example.com/x")
    assert "HTTP request failed" in result


@pytest.mark.asyncio
async def test_web_search_handles_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "web_search_provider", "tavily")
    monkeypatch.setattr(settings, "web_search_api_key", "key")

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            raise web_search.httpx.HTTPError("boom")

    monkeypatch.setattr(web_search.httpx, "AsyncClient", lambda **kwargs: _Client())
    result = await web_search.web_search("hello")
    assert "web search request failed" in result


def test_model_factory_rejects_non_object_mcp_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "mcp_servers_json", '["not-a-map"]')
    monkeypatch.setattr(settings, "browser_mcp_url", "")
    assert ModelFactory().mcp_servers() == []


@pytest.mark.asyncio
async def test_web_search_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "web_search_provider", "tavily")
    monkeypatch.setattr(settings, "web_search_api_key", "")
    result = await web_search.web_search("hello")
    assert "WEB_SEARCH_API_KEY" in result


@pytest.mark.asyncio
async def test_http_request_truncates_large_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "http_allowed_hosts", "api.example.com")

    class _Response:
        status_code = 200
        reason_phrase = "OK"
        headers = {"content-type": "application/json"}
        text = "x" * 20000

        def json(self):
            return {"data": "x" * 20000}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def request(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(http_client.httpx, "AsyncClient", lambda **kwargs: _Client())
    result = await http_client.http_request("GET", "https://api.example.com/big")
    assert "truncated" in result


def test_apply_patch_with_conflict_markers(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "app.py"
    target.write_text("print('old')\n", encoding="utf-8")
    monkeypatch.setattr(settings, "workspace_path", workspace)
    monkeypatch.setattr(settings, "filesystem_strict_workspace", True)

    patch = """<<<<<<<
print('old')
=======
print('new')
>>>>>>>
"""
    result = filesystem.apply_patch("app.py", patch)
    assert result.startswith("Patched")
    assert target.read_text(encoding="utf-8").strip() == "print('new')"


@pytest.mark.asyncio
async def test_web_search_tavily_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "web_search_provider", "tavily")
    monkeypatch.setattr(settings, "web_search_api_key", "key")

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"results": [{"title": "A", "url": "https://a.test", "content": "snippet"}]}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json=None):
            assert url.endswith("/search")
            return _Response()

    monkeypatch.setattr(web_search.httpx, "AsyncClient", lambda **kwargs: _Client())
    result = await web_search.web_search("hello", max_results=3)
    assert "Web search: hello" in result
    assert "https://a.test" in result


@pytest.mark.asyncio
async def test_web_search_brave_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "web_search_provider", "brave")
    monkeypatch.setattr(settings, "web_search_api_key", "key")

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"web": {"results": [{"title": "B", "url": "https://b.test", "description": "desc"}]}}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, params=None, headers=None):
            return _Response()

    monkeypatch.setattr(web_search.httpx, "AsyncClient", lambda **kwargs: _Client())
    result = await web_search.web_search("hello")
    assert "https://b.test" in result


@pytest.mark.asyncio
async def test_http_request_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "http_allowed_hosts", "api.example.com")

    class _Response:
        status_code = 200
        reason_phrase = "OK"
        headers = {"content-type": "text/plain"}
        text = "ok"

        def json(self):
            return {"ok": True}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def request(self, method, url, headers=None, content=None):
            assert method == "GET"
            return _Response()

    monkeypatch.setattr(http_client.httpx, "AsyncClient", lambda **kwargs: _Client())
    result = await http_client.http_request("GET", "https://api.example.com/v1/ping")
    assert "HTTP 200" in result
    assert "ok" in result


def test_model_factory_loads_extra_mcp_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        settings,
        "mcp_servers_json",
        json.dumps({"custom": "http://custom:3081/sse"}),
    )
    monkeypatch.setattr(settings, "browser_mcp_url", "")
    servers = ModelFactory().mcp_servers()
    assert len(servers) == 1


@pytest.mark.asyncio
async def test_web_search_rejects_empty_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "web_search_provider", "tavily")
    monkeypatch.setattr(settings, "web_search_api_key", "key")
    result = await web_search.web_search("   ")
    assert "query is required" in result


@pytest.mark.asyncio
async def test_web_search_unsupported_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "web_search_provider", "unknown")
    monkeypatch.setattr(settings, "web_search_api_key", "key")
    result = await web_search.web_search("hello")
    assert "unsupported" in result


def test_apply_patch_missing_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(settings, "workspace_path", workspace)
    monkeypatch.setattr(settings, "filesystem_strict_workspace", True)
    result = filesystem.apply_patch("missing.py", "patch")
    assert "file not found" in result


def test_apply_unified_patch_hunk(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "app.py"
    target.write_text("line1\nline2\nline3\n", encoding="utf-8")
    monkeypatch.setattr(settings, "workspace_path", workspace)
    monkeypatch.setattr(settings, "filesystem_strict_workspace", True)

    patch = """--- a/app.py
+++ b/app.py
@@ -2,1 +2,1 @@
-line2
+line2changed
"""
    result = filesystem.apply_patch("app.py", patch)
    assert result.startswith("Patched")
    assert "line2changed" in target.read_text(encoding="utf-8")
