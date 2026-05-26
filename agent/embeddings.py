"""Shared embedding client for OpenAI and OpenAI-compatible local servers (Ollama)."""

from __future__ import annotations

import structlog

from agent.config import settings

log = structlog.get_logger()


def openai_compatible_api_key() -> str:
    """Return an API key suitable for OpenAI-compatible endpoints (Ollama ignores it)."""
    key = settings.secret_value(settings.openai_api_key)
    if key:
        return key
    if settings.openai_base_url.strip():
        return "ollama"
    return ""


def embedding_client_kwargs() -> dict[str, str]:
    kwargs: dict[str, str] = {}
    base = settings.openai_base_url.strip()
    if base:
        kwargs["base_url"] = base
    api_key = openai_compatible_api_key()
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


async def embed_text(text: str) -> list[float] | None:
    """Generate an embedding vector. Returns None when disabled or on failure."""
    if not settings.has_embeddings:
        return None
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(**embedding_client_kwargs())
        resp = await client.embeddings.create(
            model=settings.embedding_model,
            input=text,
            encoding_format="float",
        )
        return resp.data[0].embedding
    except Exception as exc:
        log.warning("embedding_failed", error=str(exc))
        return None
