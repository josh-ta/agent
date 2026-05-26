"""Generic HTTP client with host allowlisting."""

from __future__ import annotations

import json
from urllib.parse import urlparse

import httpx

from agent.config import settings

MAX_RESPONSE_CHARS = 12_000
DEFAULT_TIMEOUT = 30.0


def _allowed_hosts() -> set[str]:
    raw = settings.http_allowed_hosts.strip()
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _validate_url(url: str) -> str | None:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        return "[ERROR: only http/https URLs are allowed]"
    host = (parsed.hostname or "").lower()
    if not host:
        return "[ERROR: URL must include a host]"
    allowed = _allowed_hosts()
    if not allowed:
        return "[ERROR: HTTP_ALLOWED_HOSTS is empty — add allowed hostnames to .env]"
    if host not in allowed and not any(host.endswith(f".{entry}") for entry in allowed):
        return f"[ERROR: host not allowed: {host}]"
    return None


async def http_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: str = "",
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    err = _validate_url(url)
    if err:
        return err
    verb = method.strip().upper()
    if verb not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
        return f"[ERROR: unsupported method: {method}]"

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.request(
                verb,
                url.strip(),
                headers=headers or None,
                content=body if body else None,
            )
    except httpx.HTTPError as exc:
        return f"[ERROR: HTTP request failed: {exc}]"

    content_type = response.headers.get("content-type", "")
    text = response.text
    if "application/json" in content_type:
        try:
            text = json.dumps(response.json(), indent=2)
        except json.JSONDecodeError:
            pass
    if len(text) > MAX_RESPONSE_CHARS:
        text = text[:MAX_RESPONSE_CHARS] + "\n... [truncated]"
    header_lines = [f"{key}: {value}" for key, value in list(response.headers.items())[:12]]
    return (
        f"HTTP {response.status_code} {response.reason_phrase}\n"
        + "\n".join(header_lines)
        + "\n\n"
        + text
    )
