"""Tool permission layer (CC-style modes + rules)."""

from __future__ import annotations

from agent.permissions.engine import (
    PermissionEngine,
    PermissionResult,
    get_permission_engine,
    set_permission_engine,
)
from agent.permissions.models import PermissionBehavior, PermissionMode

__all__ = [
    "PermissionBehavior",
    "PermissionEngine",
    "PermissionMode",
    "PermissionResult",
    "get_permission_engine",
    "set_permission_engine",
]
