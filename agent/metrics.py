"""Lightweight in-process counters (Prometheus text exposition)."""

from __future__ import annotations

import threading
from typing import ClassVar

_lock = threading.Lock()
_tasks_completed_ok = 0
_tasks_completed_fail = 0
_permission_denied = 0
_shell_blocked = 0
_context_warns = 0


def inc_task_completed(*, success: bool) -> None:
    global _tasks_completed_ok, _tasks_completed_fail
    with _lock:
        if success:
            _tasks_completed_ok += 1
        else:
            _tasks_completed_fail += 1


def inc_permission_denied() -> None:
    global _permission_denied
    with _lock:
        _permission_denied += 1


def inc_shell_blocked() -> None:
    global _shell_blocked
    with _lock:
        _shell_blocked += 1


def inc_context_warn() -> None:
    global _context_warns
    with _lock:
        _context_warns += 1


def prometheus_text() -> str:
    with _lock:
        ok = _tasks_completed_ok
        fail = _tasks_completed_fail
        pd = _permission_denied
        sb = _shell_blocked
        cw = _context_warns
    lines = [
        "# HELP agent_tasks_completed_total Tasks finished by outcome.",
        "# TYPE agent_tasks_completed_total counter",
        f'agent_tasks_completed_total{{outcome="success"}} {ok}',
        f'agent_tasks_completed_total{{outcome="failure"}} {fail}',
        "# HELP agent_permission_denied_total Tool calls blocked by permission engine.",
        "# TYPE agent_permission_denied_total counter",
        f"agent_permission_denied_total {pd}",
        "# HELP agent_shell_blocked_total Shell commands blocked by policy.",
        "# TYPE agent_shell_blocked_total counter",
        f"agent_shell_blocked_total {sb}",
        "# HELP agent_context_token_warn_total Context budget warnings emitted.",
        "# TYPE agent_context_token_warn_total counter",
        f"agent_context_token_warn_total {cw}",
        "",
    ]
    return "\n".join(lines)


class Metrics:
    """Class wrapper for tests / future DI."""

    inc_task_completed: ClassVar = staticmethod(inc_task_completed)
    inc_permission_denied: ClassVar = staticmethod(inc_permission_denied)
    inc_shell_blocked: ClassVar = staticmethod(inc_shell_blocked)
    inc_context_warn: ClassVar = staticmethod(inc_context_warn)
    prometheus_text: ClassVar = staticmethod(prometheus_text)
