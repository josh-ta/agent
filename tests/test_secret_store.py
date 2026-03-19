from __future__ import annotations

import pytest

from agent.secret_store import SecretNotFoundError, SecretStore, mask_secret


def test_secret_store_round_trip(tmp_path) -> None:
    store = SecretStore(tmp_path / "agent-secrets.json")

    store.set("LOGIN_PASSWORD", "hunter2")

    assert store.list_names() == ["LOGIN_PASSWORD"]
    assert store.get("LOGIN_PASSWORD") == "hunter2"
    assert mask_secret("hunter2") == "hu***r2"
    assert store.delete("LOGIN_PASSWORD") is True
    assert store.list_names() == []


def test_secret_store_rejects_bad_names_and_missing_values(tmp_path) -> None:
    store = SecretStore(tmp_path / "agent-secrets.json")

    with pytest.raises(ValueError):
        store.set("bad secret", "value")

    with pytest.raises(SecretNotFoundError):
        store.get("MISSING_SECRET")
