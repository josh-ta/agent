from __future__ import annotations

from pathlib import Path

import pytest

import agent.secret_store as secret_store_module
from agent.secret_store import SecretNotFoundError, SecretStore, SecretStoreError, mask_secret


def test_secret_store_round_trip(tmp_path) -> None:
    store = SecretStore(tmp_path / "agent-secrets.json")

    store.set(
        "LOGIN_PASSWORD",
        "hunter2",
        purpose="dashboard login",
        scope="staging",
        allowed_tools=["browser_fill_secret"],
    )

    assert store.list_names() == ["LOGIN_PASSWORD"]
    assert store.get("LOGIN_PASSWORD") == "hunter2"
    assert store.get_metadata("LOGIN_PASSWORD")["purpose"] == "dashboard login"
    assert store.search("LOGIN")[0]["scope"] == "staging"
    assert store.list_entries()[0]["allowed_tools"] == ["browser_fill_secret"]
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


def test_secret_store_legacy_plaintext_format_and_redaction(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secret_store_module.settings, "agent_secrets_master_key", "unit-test-key")
    path = tmp_path / "agent-secrets.json"
    path.write_text('{"TOKEN":"hunter2"}', encoding="utf-8")
    store = SecretStore(path)

    assert store.get("TOKEN") == "hunter2"
    assert mask_secret("hunter2") in store.redact_text("token=hunter2")
    assert "hunter2" not in store.redact_text("token=hunter2")


def test_secret_store_generates_key_file_when_master_key_missing(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secret_store_module.settings, "agent_secrets_master_key", "")
    store = SecretStore(tmp_path / "agent-secrets.json")

    store.set("TOKEN", "value")

    assert store.key_path.exists()
    assert store.get("TOKEN") == "value"


def test_secret_store_path_empty_search_metadata_merge_and_missing_metadata(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secret_store_module.settings, "agent_secrets_master_key", "unit-test-key")
    store = SecretStore(tmp_path / "agent-secrets.json")
    store.set("TOKEN", "value", metadata={"team": "ops"})

    assert store.path == tmp_path / "agent-secrets.json"
    assert store.search("") == store.list_entries()
    assert store.get_metadata("TOKEN")["team"] == "ops"
    with pytest.raises(SecretNotFoundError):
        store.get_metadata("MISSING")


def test_secret_store_rejects_records_missing_ciphertext_and_key_write_failures(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secret_store_module.settings, "agent_secrets_master_key", "unit-test-key")
    store = SecretStore(tmp_path / "agent-secrets.json")
    with pytest.raises(SecretStoreError):
        store._decrypt_record({})

    monkeypatch.setattr(secret_store_module.settings, "agent_secrets_master_key", "")
    key_store = SecretStore(tmp_path / "needs-key.json")
    original_write_text = Path.write_text

    def _broken_write_text(self: Path, *args, **kwargs):
        if self == key_store.key_path:
            raise OSError("key write failed")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _broken_write_text)
    with pytest.raises(SecretStoreError):
        key_store._resolve_key()


def test_secret_store_rejects_unsupported_version_and_wrong_key(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "agent-secrets.json"
    path.write_text('{"version":99,"secrets":{}}', encoding="utf-8")
    with pytest.raises(SecretStoreError):
        SecretStore(path).list_names()

    monkeypatch.setattr(secret_store_module.settings, "agent_secrets_master_key", "key-one")
    store = SecretStore(tmp_path / "encrypted.json")
    store.set("TOKEN", "value")
    monkeypatch.setattr(secret_store_module.settings, "agent_secrets_master_key", "key-two")
    with pytest.raises(SecretStoreError):
        SecretStore(tmp_path / "encrypted.json").get("TOKEN")


def test_secret_store_rejects_malformed_versioned_records(tmp_path) -> None:
    path = tmp_path / "agent-secrets.json"

    path.write_text('{"version":2,"secrets":[]}', encoding="utf-8")
    with pytest.raises(SecretStoreError):
        SecretStore(path).list_names()

    path.write_text('{"version":2,"secrets":{"TOKEN":"value"}}', encoding="utf-8")
    with pytest.raises(SecretStoreError):
        SecretStore(path).list_names()

    path.write_text('{"version":2,"secrets":{"TOKEN":{"ciphertext":1}}}', encoding="utf-8")
    with pytest.raises(SecretStoreError):
        SecretStore(path).list_names()

    path.write_text('{"version":2,"secrets":{"TOKEN":{"ciphertext":"abc","meta":"bad"}}}', encoding="utf-8")
    with pytest.raises(SecretStoreError):
        SecretStore(path).list_names()


def test_secret_store_raises_when_key_file_cannot_be_read(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secret_store_module.settings, "agent_secrets_master_key", "")
    store = SecretStore(tmp_path / "agent-secrets.json")
    store.key_path.write_text("abc", encoding="utf-8")

    original_read_text = Path.read_text

    def _broken_read_text(self: Path, *args, **kwargs):
        if self == store.key_path:
            raise OSError("key read failed")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _broken_read_text)
    with pytest.raises(SecretStoreError):
        store._resolve_key()
