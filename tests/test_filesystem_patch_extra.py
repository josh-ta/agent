from __future__ import annotations

import pytest

from agent.config import settings
from agent.tools import filesystem


def test_apply_patch_empty_patch(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "app.py"
    target.write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(settings, "workspace_path", workspace)
    result = filesystem.apply_patch("app.py", "   ")
    assert "patch is empty" in result


def test_apply_patch_unsupported_format(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "app.py"
    target.write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(settings, "workspace_path", workspace)
    result = filesystem.apply_patch("app.py", "not a valid patch")
    assert "unsupported patch format" in result


@pytest.mark.asyncio
async def test_web_search_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.tools import web_search as web_search_module

    monkeypatch.setattr(settings, "web_search_provider", "tavily")
    monkeypatch.setattr(settings, "web_search_api_key", "key")

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"results": []}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json=None):
            return _Response()

    monkeypatch.setattr(web_search_module.httpx, "AsyncClient", lambda **kwargs: _Client())
    result = await web_search_module.web_search("missing")
    assert "no web results" in result
