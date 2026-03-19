from __future__ import annotations

from pathlib import Path

import pytest

import agent.secret_store as secret_store_module
from agent.secret_store import SecretNotFoundError, SecretStore, SecretStoreError, mask_secret


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


def test_secret_store_covers_mask_normalization_and_false_delete(tmp_path) -> None:
    store = SecretStore(tmp_path / "agent-secrets.json")

    assert mask_secret("") == "(empty)"
    assert mask_secret("abc") == "***"
    assert secret_store_module._normalize_name("  API_TOKEN  ") == "API_TOKEN"

    with pytest.raises(ValueError):
        secret_store_module._normalize_name("   ")

    assert store.delete("MISSING_SECRET") is False


def test_secret_store_load_and_write_errors(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = tmp_path / "agent-secrets.json"
    store = SecretStore(path)

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SecretStoreError):
        store.list_names()

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(SecretStoreError):
        store.list_names()

    path.write_text('{"ok": 1}', encoding="utf-8")
    with pytest.raises(SecretStoreError):
        store.list_names()

    original_read_text = Path.read_text

    def _broken_read_text(self: Path, *args, **kwargs):
        if self == path:
            raise OSError("read failed")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _broken_read_text)
    with pytest.raises(SecretStoreError):
        store.list_names()

    monkeypatch.setattr(Path, "read_text", original_read_text)

    original_write_text = Path.write_text

    def _broken_write_text(self: Path, *args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(Path, "write_text", _broken_write_text)
    with pytest.raises(SecretStoreError):
        store.set("TOKEN", "value")

    monkeypatch.setattr(Path, "write_text", original_write_text)
    path.unlink()
    store.set("TOKEN", "value")

    original_replace = Path.replace

    def _broken_replace(self: Path, target, *args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", _broken_replace)
    with pytest.raises(SecretStoreError):
        store.set("TOKEN", "new-value")

    monkeypatch.setattr(Path, "replace", original_replace)
