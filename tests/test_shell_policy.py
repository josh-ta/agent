from pathlib import Path

from agent.tools.shell_policy import resolve_shell_cwd, validate_shell_command


def test_validate_shell_blocks_read_only_redirection() -> None:
    err = validate_shell_command("echo hi > /tmp/x", read_only=True)
    assert err is not None


def test_validate_shell_allows_echo_read_only() -> None:
    assert validate_shell_command("echo hi", read_only=True) is None


def test_resolve_shell_cwd_rejects_escape(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    cwd, err = resolve_shell_cwd(str(outside), ws)
    assert cwd is None
    assert err is not None
