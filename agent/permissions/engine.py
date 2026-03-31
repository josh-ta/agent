from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from typing import Any

import structlog

from agent.permissions.models import PermissionMode

log = structlog.get_logger()

_engine: PermissionEngine | None = None


def set_permission_engine(engine: PermissionEngine | None) -> None:
    global _engine
    _engine = engine


def get_permission_engine() -> PermissionEngine | None:
    return _engine


@dataclass(frozen=True)
class PermissionResult:
    ok: bool
    message: str = ""


# Tools that mutate external state or the workspace (blocked in plan mode).
MUTATING_TOOLS: frozenset[str] = frozenset(
    {
        "run_shell",
        "write_file",
        "delete_file",
        "str_replace",
        "skill_edit",
        "identity_edit",
        "agent_restart",
        "send_discord",
        "ask_user_question",
        "gh_pr_comment",
        "gh_pr_review",
        "gh_pr_review_inline",
        "gh_pr_merge",
        "gh_issue_comment",
        "gh_issue_create",
        "gh_issue_close",
        "gh_ci_rerun",
        "secret_set",
        "secret_delete",
        "memory_save",
        "lesson_save",
        "procedure_save",
        "schedule_background_task",
        "cancel_scheduled_task",
        "run_agent_subtask",
    }
)


class PermissionEngine:
    """Evaluate tool use from persisted mode + rules."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self._mode: PermissionMode = "default"
        self._rules: list[dict[str, Any]] = []

    async def load(self) -> None:
        from agent.config import settings

        valid_modes = {"default", "plan", "dontAsk", "bypassPermissions", "acceptEdits"}
        raw = (settings.permission_mode or "default").strip()
        self._mode = raw if raw in valid_modes else "default"
        self._rules = []
        if self._store is None or not hasattr(self._store, "permission_list_rules"):
            return
        try:
            self._rules = await self._store.permission_list_rules()
        except Exception as exc:
            log.warning("permission_load_failed", error=str(exc))

    def check_sync(self, tool_name: str, payload: dict[str, Any] | None = None) -> PermissionResult:
        payload = payload or {}
        mode = self._mode

        if mode in ("bypassPermissions", "dontAsk", "acceptEdits"):
            return PermissionResult(ok=True)

        if mode == "plan" and tool_name in MUTATING_TOOLS:
            return PermissionResult(
                ok=False,
                message=(
                    f"[Permission denied: `{tool_name}` is blocked in plan mode. "
                    "Switch PERMISSION_MODE to default or acceptEdits.]"
                ),
            )

        tn = tool_name.lower()
        for rule in self._rules:
            pat = (rule.get("tool_name") or "*").strip().lower() or "*"
            if not fnmatch.fnmatch(tn, pat):
                continue
            behavior = rule.get("rule_behavior", "deny")
            content_pat = (rule.get("rule_content") or "").strip()
            if content_pat:
                payload_text = json.dumps(payload, sort_keys=True, default=str)
                if content_pat not in payload_text and not fnmatch.fnmatch(
                    payload_text.lower(), content_pat.lower()
                ):
                    continue
            if behavior == "deny":
                return PermissionResult(
                    ok=False,
                    message=f"[Permission denied by rule for `{tool_name}`.]",
                )
            if behavior == "ask":
                return PermissionResult(
                    ok=False,
                    message=(
                        f"[Permission: tool `{tool_name}` requires approval — "
                        "add an allow rule or use PERMISSION_MODE=dontAsk.]"
                    ),
                )
            if behavior == "allow":
                return PermissionResult(ok=True)

        return PermissionResult(ok=True)
