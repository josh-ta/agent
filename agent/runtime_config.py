"""Runtime configuration overrides persisted on disk and applied without redeploy."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import structlog

from agent.config import Settings, settings

log = structlog.get_logger()


@dataclass(frozen=True)
class ConfigFieldSpec:
    field_name: str
    env_key: str
    description: str
    value_type: Literal["str", "bool", "int"]
    reload_agents: bool = False


CONFIG_FIELDS: dict[str, ConfigFieldSpec] = {
    "agent_model": ConfigFieldSpec(
        "agent_model",
        "AGENT_MODEL",
        "Default model when no tier matches",
        "str",
        reload_agents=True,
    ),
    "model_fast": ConfigFieldSpec(
        "model_fast",
        "MODEL_FAST",
        "Fast tier model (/fast prefix)",
        "str",
        reload_agents=True,
    ),
    "model_smart": ConfigFieldSpec(
        "model_smart",
        "MODEL_SMART",
        "Smart tier model (/smart prefix)",
        "str",
        reload_agents=True,
    ),
    "model_best": ConfigFieldSpec(
        "model_best",
        "MODEL_BEST",
        "Best tier model (/best prefix)",
        "str",
        reload_agents=True,
    ),
    "thinking_enabled": ConfigFieldSpec(
        "thinking_enabled",
        "THINKING_ENABLED",
        "Enable Claude extended thinking",
        "bool",
    ),
    "thinking_budget_tokens": ConfigFieldSpec(
        "thinking_budget_tokens",
        "THINKING_BUDGET_TOKENS",
        "Extended thinking token budget",
        "int",
    ),
    "discord_use_task_threads": ConfigFieldSpec(
        "discord_use_task_threads",
        "DISCORD_USE_TASK_THREADS",
        "Route task detail to per-task threads",
        "bool",
    ),
    "permission_mode": ConfigFieldSpec(
        "permission_mode",
        "PERMISSION_MODE",
        "Permission mode (default, plan, etc.)",
        "str",
    ),
    "log_level": ConfigFieldSpec(
        "log_level",
        "LOG_LEVEL",
        "Log level (DEBUG, INFO, WARNING, ERROR)",
        "str",
    ),
}

_ENV_TO_FIELD = {spec.env_key: spec.field_name for spec in CONFIG_FIELDS.values()}
_FIELD_TO_ENV = {spec.field_name: spec.env_key for spec in CONFIG_FIELDS.values()}


def resolve_config_key(raw: str) -> ConfigFieldSpec | None:
    key = raw.strip().lower().replace("-", "_")
    if key in CONFIG_FIELDS:
        return CONFIG_FIELDS[key]
    upper = raw.strip().upper()
    field_name = _ENV_TO_FIELD.get(upper)
    if field_name is not None:
        return CONFIG_FIELDS[field_name]
    return None


def overrides_path(target: Settings | None = None) -> Path:
    cfg = target or settings
    return cfg.runtime_overrides_path


def load_stored_overrides(target: Settings | None = None) -> dict[str, str]:
    path = overrides_path(target)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("runtime_config_load_failed", path=str(path), error=str(exc))
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def apply_stored_overrides(target: Settings | None = None) -> list[str]:
    cfg = target or settings
    applied: list[str] = []
    for env_key, raw_value in load_stored_overrides(cfg).items():
        spec = resolve_config_key(env_key)
        if spec is None:
            continue
        try:
            parsed = _coerce_value(spec, raw_value)
        except ValueError:
            log.warning("runtime_config_skip_invalid", key=env_key, value=raw_value)
            continue
        setattr(cfg, spec.field_name, parsed)
        applied.append(spec.env_key)
    if applied:
        log.info("runtime_config_loaded", keys=applied)
    return applied


def _coerce_value(spec: ConfigFieldSpec, raw: str) -> Any:
    text = raw.strip()
    if spec.value_type == "str":
        if not text:
            raise ValueError("empty string")
        return text
    if spec.value_type == "bool":
        lowered = text.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"invalid bool: {raw}")
    if spec.value_type == "int":
        value = int(text)
        if value < 0:
            raise ValueError("must be non-negative")
        return value
    raise ValueError(f"unsupported type: {spec.value_type}")


def _serialize_value(spec: ConfigFieldSpec, value: Any) -> str:
    if spec.value_type == "bool":
        return "true" if bool(value) else "false"
    return str(value)


def get_field_value(spec: ConfigFieldSpec, target: Settings | None = None) -> Any:
    cfg = target or settings
    return getattr(cfg, spec.field_name)


def format_field_value(spec: ConfigFieldSpec, target: Settings | None = None) -> str:
    return _serialize_value(spec, get_field_value(spec, target))


def format_config_list(target: Settings | None = None) -> str:
    lines = ["**Runtime configuration** (change without redeploying)", ""]
    for index, spec in enumerate(CONFIG_FIELDS.values(), start=1):
        value = _serialize_value(spec, get_field_value(spec, target))
        reload_note = " · reloads agents" if spec.reload_agents else ""
        lines.append(f"{index}. `{spec.env_key}` = `{value}` — {spec.description}{reload_note}")
    lines.extend(
        [
            "",
            "Direct set: `/config AGENT_MODEL:claude-sonnet-4-5`",
            "Wizard: `/config` then reply with a number or key name",
            "Cancel wizard: `/config cancel`",
        ]
    )
    return "\n".join(lines)


def format_wizard_prompt() -> str:
    return (
        "⚙️ **Config wizard** — pick a setting to change.\n\n"
        f"{format_config_list()}"
    )


def _write_overrides(data: dict[str, str], target: Settings | None = None) -> None:
    path = overrides_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def set_config_value(
    key: str,
    raw_value: str,
    *,
    target: Settings | None = None,
) -> tuple[bool, str, bool]:
    """Apply one override. Returns (ok, message, reload_agents)."""
    spec = resolve_config_key(key)
    if spec is None:
        known = ", ".join(sorted(spec.env_key for spec in CONFIG_FIELDS.values()))
        return False, f"Unknown setting `{key}`. Allowed: {known}", False

    try:
        parsed = _coerce_value(spec, raw_value)
    except ValueError as exc:
        return False, f"Invalid value for `{spec.env_key}`: {exc}", False

    cfg = target or settings
    previous = get_field_value(spec, cfg)
    setattr(cfg, spec.field_name, parsed)

    stored = load_stored_overrides(cfg)
    stored[spec.env_key] = _serialize_value(spec, parsed)
    _write_overrides(stored, cfg)

    if spec.field_name == "log_level":
        logging.getLogger().setLevel(getattr(logging, str(parsed).upper(), logging.INFO))

    changed = previous != parsed
    reload = spec.reload_agents and changed
    message = (
        f"✅ Updated `{spec.env_key}` → `{_serialize_value(spec, parsed)}`."
        + (" Agents reloaded for the next task." if reload else "")
    )
    log.info(
        "runtime_config_updated",
        key=spec.env_key,
        value=_serialize_value(spec, parsed),
        reload_agents=reload,
    )
    return True, message, reload


def clear_config_override(key: str, *, target: Settings | None = None) -> tuple[bool, str]:
    spec = resolve_config_key(key)
    if spec is None:
        return False, f"Unknown setting `{key}`."
    cfg = target or settings
    stored = load_stored_overrides(cfg)
    stored.pop(spec.env_key, None)
    _write_overrides(stored, cfg)
    return True, f"🧹 Cleared override for `{spec.env_key}`. Restart or edit `.env` to revert fully."
