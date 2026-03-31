from __future__ import annotations

from typing import Literal

PermissionMode = Literal[
    "default",
    "plan",
    "dontAsk",
    "bypassPermissions",
    "acceptEdits",
]

PermissionBehavior = Literal["allow", "deny", "ask"]
