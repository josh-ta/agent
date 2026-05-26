from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.config import Settings
from agent.runtime_config import (
    apply_stored_overrides,
    clear_config_override,
    format_config_list,
    load_stored_overrides,
    resolve_config_key,
    set_config_value,
)


@pytest.fixture
def temp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    from agent.config import settings as base_settings

    overrides = tmp_path / "overrides.json"
    cfg = base_settings.model_copy(
        update={
            "runtime_overrides_path": overrides,
            "agent_model": "claude-haiku-4-5",
            "model_fast": "claude-haiku-4-5",
            "model_smart": "claude-sonnet-4-5",
            "model_best": "claude-opus-4-5",
        }
    )
    monkeypatch.setattr("agent.runtime_config.settings", cfg)
    return cfg


def test_resolve_config_key_accepts_env_and_field_names() -> None:
    assert resolve_config_key("AGENT_MODEL") is not None
    assert resolve_config_key("agent_model") is not None
    assert resolve_config_key("unknown") is None


def test_set_config_value_persists_and_applies(temp_settings: Settings) -> None:
    ok, message, reload = set_config_value("AGENT_MODEL", "claude-sonnet-4-5", target=temp_settings)

    assert ok is True
    assert reload is True
    assert temp_settings.agent_model == "claude-sonnet-4-5"
    assert "Updated" in message
    stored = load_stored_overrides(temp_settings)
    assert stored["AGENT_MODEL"] == "claude-sonnet-4-5"


def test_apply_stored_overrides_on_startup(temp_settings: Settings) -> None:
    temp_settings.runtime_overrides_path.write_text(
        json.dumps({"MODEL_FAST": "gpt-4o-mini", "THINKING_ENABLED": "true"}),
        encoding="utf-8",
    )
    temp_settings.model_fast = "claude-haiku-4-5"
    temp_settings.thinking_enabled = False

    applied = apply_stored_overrides(temp_settings)

    assert "MODEL_FAST" in applied
    assert temp_settings.model_fast == "gpt-4o-mini"
    assert temp_settings.thinking_enabled is True


def test_set_config_value_rejects_empty_string(temp_settings: Settings) -> None:
    ok, message, reload = set_config_value("AGENT_MODEL", "   ", target=temp_settings)
    assert ok is False
    assert reload is False


def test_set_config_value_rejects_invalid_int(temp_settings: Settings) -> None:
    ok, message, reload = set_config_value("THINKING_BUDGET_TOKENS", "-1", target=temp_settings)
    assert ok is False
    assert reload is False

    ok, message, reload = set_config_value("THINKING_ENABLED", "maybe", target=temp_settings)

    assert ok is False
    assert reload is False
    assert "Invalid value" in message


def test_format_config_list_includes_model_keys() -> None:
    text = format_config_list()
    assert "AGENT_MODEL" in text
    assert "/config AGENT_MODEL:" in text


def test_set_config_value_rejects_unknown_key(temp_settings: Settings) -> None:
    ok, message, reload = set_config_value("NOT_A_KEY", "value", target=temp_settings)
    assert ok is False
    assert reload is False
    assert "Unknown setting" in message


def test_load_stored_overrides_ignores_invalid_json(temp_settings: Settings) -> None:
    temp_settings.runtime_overrides_path.write_text("{bad", encoding="utf-8")
    assert load_stored_overrides(temp_settings) == {}


def test_load_stored_overrides_ignores_non_dict_payload(temp_settings: Settings) -> None:
    temp_settings.runtime_overrides_path.write_text("[]", encoding="utf-8")
    assert load_stored_overrides(temp_settings) == {}


def test_apply_stored_overrides_skips_unknown_and_invalid_values(temp_settings: Settings) -> None:
    temp_settings.runtime_overrides_path.write_text(
        json.dumps({"UNKNOWN": "x", "THINKING_ENABLED": "maybe"}),
        encoding="utf-8",
    )
    applied = apply_stored_overrides(temp_settings)
    assert applied == []


def test_clear_config_override_unknown_key(temp_settings: Settings) -> None:
    ok, message = clear_config_override("NOPE", target=temp_settings)
    assert ok is False
    assert "Unknown setting" in message


def test_set_config_value_no_reload_when_unchanged(temp_settings: Settings) -> None:
    ok, message, reload = set_config_value(
        "AGENT_MODEL",
        temp_settings.agent_model,
        target=temp_settings,
    )
    assert ok is True
    assert reload is False

    set_config_value("AGENT_MODEL", "claude-sonnet-4-5", target=temp_settings)
    ok, message = clear_config_override(
        "AGENT_MODEL",
        target=temp_settings,
    )
    assert ok is True
    assert "Cleared override" in message


def test_set_config_value_updates_log_level(temp_settings: Settings) -> None:
    ok, message, reload = set_config_value("LOG_LEVEL", "ERROR", target=temp_settings)
    assert ok is True
    assert reload is False
    assert temp_settings.log_level == "ERROR"
