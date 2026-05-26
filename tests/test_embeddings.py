from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.config import settings
from agent.embeddings import embed_text, embedding_client_kwargs, openai_compatible_api_key


def test_openai_compatible_api_key_uses_placeholder_for_local_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "openai_base_url", "http://10.0.0.5:11434/v1")
    assert openai_compatible_api_key() == "ollama"


def test_has_embeddings_true_for_local_base_url_without_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "openai_base_url", "http://10.0.0.5:11434/v1")
    monkeypatch.setattr(settings, "embedding_model", "nomic-embed-text")
    assert settings.has_embeddings is True


def test_embedding_client_kwargs_include_local_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "openai_base_url", "http://10.0.0.5:11434/v1")
    assert embedding_client_kwargs() == {
        "base_url": "http://10.0.0.5:11434/v1",
        "api_key": "ollama",
    }


@pytest.mark.asyncio
async def test_embed_text_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "openai_base_url", "")
    monkeypatch.setattr(settings, "embedding_model", "")
    assert await embed_text("hello") is None
