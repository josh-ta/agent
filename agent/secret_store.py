"""Persistent named secrets for agent-only access."""

from __future__ import annotations

import json
import re
from pathlib import Path

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")


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


class SecretStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def list_names(self) -> list[str]:
        return sorted(self._load().keys())

    def get(self, name: str) -> str:
        normalized = _normalize_name(name)
        data = self._load()
        if normalized not in data:
            raise SecretNotFoundError(normalized)
        return data[normalized]

    def set(self, name: str, value: str) -> None:
        normalized = _normalize_name(name)
        data = self._load()
        data[normalized] = value
        self._write(data)

    def delete(self, name: str) -> bool:
        normalized = _normalize_name(name)
        data = self._load()
        removed = data.pop(normalized, None)
        if removed is None:
            return False
        self._write(data)
        return True

    def _load(self) -> dict[str, str]:
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

        secrets: dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise SecretStoreError("Secret store keys and values must all be strings.")
            secrets[_normalize_name(key)] = value
        return secrets

    def _write(self, data: dict[str, str]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(f"{self._path.suffix}.tmp")
            tmp_path.write_text(
                json.dumps(dict(sorted(data.items())), indent=2) + "\n",
                encoding="utf-8",
            )
            tmp_path.replace(self._path)
        except OSError as exc:
            raise SecretStoreError(f"Failed to write secret store: {exc}") from exc
