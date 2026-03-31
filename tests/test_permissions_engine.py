import pytest

import agent.config as agent_config
from agent.permissions.engine import PermissionEngine, MUTATING_TOOLS


@pytest.mark.asyncio
async def test_plan_mode_denies_run_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_config.settings, "permission_mode", "plan")
    eng = PermissionEngine(None)
    await eng.load()
    r = eng.check_sync("run_shell", {"command": "ls"})
    assert r.ok is False


@pytest.mark.asyncio
async def test_plan_mode_allows_run_shell_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_config.settings, "permission_mode", "plan")
    eng = PermissionEngine(None)
    await eng.load()
    r = eng.check_sync("run_shell_read_only", {"command": "ls"})
    assert r.ok is True


@pytest.mark.asyncio
async def test_rules_deny_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_config.settings, "permission_mode", "default")

    class _Store:
        async def permission_list_rules(self):
            return [{"tool_name": "write_*", "rule_behavior": "deny", "rule_content": ""}]

    eng = PermissionEngine(_Store())
    await eng.load()
    assert eng.check_sync("write_file", {"path": "x"}).ok is False
    assert eng.check_sync("read_file", {"path": "x"}).ok is True


def test_mutating_tools_includes_run_shell() -> None:
    assert "run_shell" in MUTATING_TOOLS
    assert "run_shell_read_only" not in MUTATING_TOOLS
