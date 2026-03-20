"""Persistent encrypted named secrets for agent-only access."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from agent.config import settings

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_VERSION = 2


class SecretStoreError(RuntimeError):
    """Raised when the secret store cannot be read or written safely."""


class SecretNotFoundError(KeyError):
    """Raised when a named secret is missing."""


def mask_secret(value: str) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def _normalize_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise ValueError("Secret name cannot be empty.")
    if not _NAME_RE.fullmatch(normalized):
        raise ValueError(
            "Secret name may only contain letters, numbers, dots, underscores, colons, slashes, and dashes."
        )
    return normalized


def _normalize_key(raw_key: str) -> bytes:
    candidate = raw_key.strip().encode("utf-8")
    try:
        decoded = base64.urlsafe_b64decode(candidate)
        if len(decoded) == 32:  # pragma: no cover - simple fast-path for already-normalized keys
            return candidate
    except Exception:
        pass
    digest = hashlib.sha256(raw_key.strip().encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


class SecretStore:
    def __init__(self, path: Path, master_key: str | None = None) -> None:
        self._path = path
        self._master_key = master_key

    @property
    def path(self) -> Path:
        return self._path

    @property
    def key_path(self) -> Path:
        return self._path.with_suffix(f"{self._path.suffix}.key")

    def list_names(self) -> list[str]:
        return sorted(self._load_records().keys())

    def list_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for name, record in sorted(self._load_records().items()):
            meta = dict(record.get("meta") or {})
            entries.append(
                {
                    "name": name,
                    "purpose": str(meta.get("purpose", "")),
                    "scope": str(meta.get("scope", "")),
                    "allowed_tools": list(meta.get("allowed_tools") or []),
                    "rotation_hint": str(meta.get("rotation_hint", "")),
                    "last_used_ts": meta.get("last_used_ts"),
                    "updated_ts": record.get("updated_ts"),
                }
            )
        return entries

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        needle = query.strip().lower()
        if not needle:
            return self.list_entries()[:limit]
        terms = [part for part in re.findall(r"[A-Za-z0-9_./:-]+", needle) if len(part) >= 3]
        matches: list[dict[str, Any]] = []
        for entry in self.list_entries():
            haystack = " ".join(
                [
                    entry["name"],
                    entry["purpose"],
                    entry["scope"],
                    " ".join(entry["allowed_tools"]),
                    entry["rotation_hint"],
                ]
            ).lower()
            if needle in haystack or any(term in haystack for term in terms):  # pragma: no branch - straightforward membership check
                matches.append(entry)
        return matches[:limit]

    def redact_text(self, text: str) -> str:
        redacted = text
        for record in self._load_records().values():
            value = self._decrypt_record(record)
            if value:  # pragma: no branch - replacement only when a stored value is present
                redacted = redacted.replace(value, mask_secret(value))
        return redacted

    def get(self, name: str) -> str:
        normalized = _normalize_name(name)
        records = self._load_records()
        record = records.get(normalized)
        if record is None:
            raise SecretNotFoundError(normalized)
        value = self._decrypt_record(record)
        meta = dict(record.get("meta") or {})
        meta["last_used_ts"] = time.time()
        record["meta"] = meta
        records[normalized] = record
        self._write_records(records)
        return value

    def get_metadata(self, name: str) -> dict[str, Any]:
        normalized = _normalize_name(name)
        records = self._load_records()
        record = records.get(normalized)
        if record is None:
            raise SecretNotFoundError(normalized)
        return dict(record.get("meta") or {})

    def set(
        self,
        name: str,
        value: str,
        *,
        purpose: str = "",
        scope: str = "",
        allowed_tools: list[str] | None = None,
        rotation_hint: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        normalized = _normalize_name(name)
        records = self._load_records()
        now = time.time()
        meta = {
            "purpose": purpose.strip(),
            "scope": scope.strip(),
            "allowed_tools": list(allowed_tools or []),
            "rotation_hint": rotation_hint.strip(),
            "last_used_ts": None,
        }
        if metadata:
            meta.update(metadata)
        records[normalized] = {
            "ciphertext": self._fernet().encrypt(value.encode("utf-8")).decode("utf-8"),
            "meta": meta,
            "updated_ts": now,
        }
        self._write_records(records)

    def delete(self, name: str) -> bool:
        normalized = _normalize_name(name)
        records = self._load_records()
        removed = records.pop(normalized, None)
        if removed is None:
            return False
        self._write_records(records)
        return True

    def _load_records(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SecretStoreError(f"Invalid JSON in secret store: {exc}") from exc
        except OSError as exc:
            raise SecretStoreError(f"Failed to read secret store: {exc}") from exc

        if not isinstance(raw, dict):
            raise SecretStoreError("Secret store must contain a JSON object.")

        if "version" in raw or "secrets" in raw:
            version = raw.get("version")
            if version != _VERSION:
                raise SecretStoreError(f"Unsupported secret store version: {version!r}")
            payload = raw.get("secrets")
            if not isinstance(payload, dict):
                raise SecretStoreError("Secret store 'secrets' field must be an object.")
            records: dict[str, dict[str, Any]] = {}
            for key, value in payload.items():
                normalized = _normalize_name(key)
                if not isinstance(value, dict):
                    raise SecretStoreError("Each secret record must be an object.")
                ciphertext = value.get("ciphertext")
                if not isinstance(ciphertext, str):
                    raise SecretStoreError("Secret record ciphertext must be a string.")
                meta = value.get("meta") or {}
                if not isinstance(meta, dict):
                    raise SecretStoreError("Secret metadata must be an object.")
                records[normalized] = {
                    "ciphertext": ciphertext,
                    "meta": meta,
                    "updated_ts": float(value.get("updated_ts") or 0.0),
                }
            return records

        # Backward-compatible plaintext format: {"NAME": "value"}.
        records = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise SecretStoreError("Secret store keys and values must all be strings.")
            records[_normalize_name(key)] = {
                "plaintext": value,
                "meta": {},
                "updated_ts": 0.0,
            }
        return records

    def _decrypt_record(self, record: dict[str, Any]) -> str:
        plaintext = record.get("plaintext")
        if isinstance(plaintext, str):
            return plaintext
        ciphertext = record.get("ciphertext")
        if not isinstance(ciphertext, str):
            raise SecretStoreError("Secret record is missing ciphertext.")
        try:
            return self._fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:
            raise SecretStoreError("Secret decryption failed. Check the master key.") from exc

    def _write_records(self, records: dict[str, dict[str, Any]]) -> None:
        payload: dict[str, Any] = {"version": _VERSION, "secrets": {}}
        for name, record in sorted(records.items()):
            value = self._decrypt_record(record)
            payload["secrets"][name] = {
                "ciphertext": self._fernet().encrypt(value.encode("utf-8")).decode("utf-8"),
                "meta": dict(record.get("meta") or {}),
                "updated_ts": float(record.get("updated_ts") or time.time()),
            }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(f"{self._path.suffix}.tmp")
            tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            tmp_path.replace(self._path)
        except OSError as exc:
            raise SecretStoreError(f"Failed to write secret store: {exc}") from exc

    def _fernet(self) -> Fernet:
        return Fernet(self._resolve_key())

    def _resolve_key(self) -> bytes:
        explicit = (self._master_key or "").strip()
        configured = settings.secret_value(settings.agent_secrets_master_key).strip()
        chosen = explicit or configured
        if chosen:
            return _normalize_key(chosen)
        if self.key_path.exists():
            try:
                return _normalize_key(self.key_path.read_text(encoding="utf-8"))
            except OSError as exc:
                raise SecretStoreError(f"Failed to read secret key: {exc}") from exc
        generated = Fernet.generate_key()
        try:
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
            self.key_path.write_text(generated.decode("utf-8"), encoding="utf-8")
        except OSError as exc:
            raise SecretStoreError(f"Failed to persist generated secret key: {exc}") from exc
        return generated
