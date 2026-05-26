"""Web search via Tavily or Brave Search API."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from agent.config import settings

DEFAULT_TIMEOUT = 30.0


def _provider() -> str:
    return settings.web_search_provider.strip().lower()


def _api_key() -> str:
    return settings.secret_value(settings.web_search_api_key)


async def web_search(query: str, max_results: int = 5) -> str:
    provider = _provider()
    api_key = _api_key()
    if not provider:
        return "[ERROR: WEB_SEARCH_PROVIDER is not set (tavily or brave)]"
    if not api_key:
        return "[ERROR: WEB_SEARCH_API_KEY is not set]"
    if not query.strip():
        return "[ERROR: query is required]"
    max_results = max(1, min(max_results, 10))

    try:
        if provider == "tavily":
            return await _search_tavily(query.strip(), max_results, api_key)
        if provider == "brave":
            return await _search_brave(query.strip(), max_results, api_key)
        return f"[ERROR: unsupported WEB_SEARCH_PROVIDER: {provider}]"
    except httpx.HTTPError as exc:
        return f"[ERROR: web search request failed: {exc}]"
    except Exception as exc:
        return f"[ERROR: {exc}]"


async def _search_tavily(query: str, max_results: int, api_key: str) -> str:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
    return _format_results(query, _normalize_tavily(payload))


async def _search_brave(query: str, max_results: int, api_key: str) -> str:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()
    return _format_results(query, _normalize_brave(payload))


def _normalize_tavily(payload: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("content", ""))[:500],
            }
        )
    return rows


def _normalize_brave(payload: dict[str, Any]) -> list[dict[str, str]]:
    web = payload.get("web") or {}
    rows: list[dict[str, str]] = []
    for item in web.get("results") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("description", ""))[:500],
            }
        )
    return rows


def _format_results(query: str, rows: list[dict[str, str]]) -> str:
    if not rows:
        return f"(no web results for: {query})"
    lines = [f"## Web search: {query}", ""]
    for idx, row in enumerate(rows, start=1):
        host = urlparse(row["url"]).netloc or row["url"]
        lines.append(f"{idx}. **{row['title'] or host}**")
        if row["url"]:
            lines.append(f"   {row['url']}")
        if row["snippet"]:
            lines.append(f"   {row['snippet']}")
        lines.append("")
    return "\n".join(lines).strip()
