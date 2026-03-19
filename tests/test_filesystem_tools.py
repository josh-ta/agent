from __future__ import annotations

from pathlib import Path

from agent.tools import filesystem


def test_filesystem_write_read_list_and_delete(isolated_paths) -> None:
    rel_path = "notes/example.txt"

    write_result = filesystem.write_file(rel_path, "hello world")
    read_result = filesystem.read_file(rel_path)
    list_result = filesystem.list_dir("notes")
    delete_result = filesystem.delete_file(rel_path)

    assert "Written" in write_result
    assert read_result == "hello world"
    assert "example.txt" in list_result
    assert "Deleted" in delete_result


def test_filesystem_str_replace_and_search(isolated_paths, monkeypatch) -> None:
    file_path = Path(isolated_paths["workspace"]) / "sample.py"
    file_path.write_text("value = 1\nprint(value)\n", encoding="utf-8")

    class _Result:
        returncode = 0
        stdout = f"{file_path}:1:value = 2\n"
        stderr = ""

    monkeypatch.setattr(filesystem.subprocess, "run", lambda *args, **kwargs: _Result())

    replace_result = filesystem.str_replace_file(
        str(file_path),
        "value = 1",
        "value = 2",
    )
    search_result = filesystem.search_files("value", str(file_path))

    assert "Replaced 1 occurrence" in replace_result
    assert "value = 2" in search_result


def test_filesystem_read_missing_file_returns_error() -> None:
    result = filesystem.read_file("does/not/exist.txt")

    assert result.startswith("[ERROR: file not found")


def test_filesystem_str_replace_reports_missing_and_mismatch(isolated_paths) -> None:
    file_path = Path(isolated_paths["workspace"]) / "sample.py"
    file_path.write_text("value = 1\nvalue = 1\n", encoding="utf-8")

    missing = filesystem.str_replace_file(str(file_path), "missing", "value = 2")
    mismatch = filesystem.str_replace_file(str(file_path), "value = 1", "value = 2")

    assert "old_str not found" in missing
    assert "expected 1 occurrence" in mismatch


def test_filesystem_delete_file_rejects_directories(isolated_paths) -> None:
    notes_dir = Path(isolated_paths["workspace"]) / "notes"
    notes_dir.mkdir()

    result = filesystem.delete_file(str(notes_dir))

    assert "use shell 'rm -rf' for directories" in result


def test_filesystem_reports_missing_delete_and_non_directory_list_and_replace(isolated_paths) -> None:
    file_path = Path(isolated_paths["workspace"]) / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    assert filesystem.delete_file("missing.txt").startswith("[ERROR: file not found:")
    assert filesystem.list_dir(str(file_path)).startswith("[ERROR: not a directory:")
    assert filesystem.str_replace_file("missing.txt", "a", "b").startswith("[ERROR: file not found:")


def test_filesystem_list_dir_and_search_report_missing_paths() -> None:
    list_result = filesystem.list_dir("missing")
    search_result = filesystem.search_files("value", "missing")

    assert list_result.startswith("[ERROR: path not found")
    assert search_result.startswith("[ERROR: path not found")


def test_filesystem_safe_path_warns_for_external_absolute_paths(isolated_paths, monkeypatch) -> None:
    warnings: list[tuple[str, str]] = []

    def _warn(event: str, *, path: str, resolved: str) -> None:
        warnings.append((path, resolved))

    monkeypatch.setattr(filesystem.log, "warning", _warn)

    resolved = filesystem._safe_path("/tmp/outside.txt")

    assert str(resolved).endswith("/tmp/outside.txt")
    assert warnings == [("/tmp/outside.txt", str(resolved))]


def test_filesystem_read_file_supports_binary_and_truncation(isolated_paths) -> None:
    binary_path = Path(isolated_paths["workspace"]) / "data.bin"
    text_path = Path(isolated_paths["workspace"]) / "big.txt"
    binary_path.write_bytes(b"\x00\x01png")
    text_path.write_text("a" * (filesystem.MAX_READ_BYTES + 5), encoding="utf-8")

    binary = filesystem.read_file(str(binary_path), encoding="binary")
    truncated = filesystem.read_file(str(text_path))

    assert binary.startswith("0001706e67")
    assert truncated.endswith(f"... [truncated, total {filesystem.MAX_READ_BYTES + 5} bytes]")


def test_filesystem_search_handles_no_matches_errors_truncation_and_timeouts(isolated_paths, monkeypatch) -> None:
    file_path = Path(isolated_paths["workspace"]) / "sample.py"
    file_path.write_text("value = 1\n", encoding="utf-8")

    class _NoMatches:
        returncode = 1
        stdout = ""
        stderr = ""

    monkeypatch.setattr(filesystem.subprocess, "run", lambda *args, **kwargs: _NoMatches())
    assert filesystem.search_files("missing", str(file_path)) == "(no matches for pattern: missing)"

    class _Error:
        returncode = 2
        stdout = ""
        stderr = "bad regex"

    monkeypatch.setattr(filesystem.subprocess, "run", lambda *args, **kwargs: _Error())
    assert "rg exited 2" in filesystem.search_files("bad", str(file_path))

    class _Large:
        returncode = 0
        stdout = "x" * (filesystem.MAX_SEARCH_BYTES + 10)
        stderr = ""

    monkeypatch.setattr(filesystem.subprocess, "run", lambda *args, **kwargs: _Large())
    assert "truncated at 8KB" in filesystem.search_files("value", str(file_path))

    def _missing_rg(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(filesystem.subprocess, "run", _missing_rg)
    assert "ripgrep (rg) not found" in filesystem.search_files("value", str(file_path))

    def _timeout(*args, **kwargs):
        raise filesystem.subprocess.TimeoutExpired(cmd="rg", timeout=30)

    monkeypatch.setattr(filesystem.subprocess, "run", _timeout)
    assert "timed out after 30s" in filesystem.search_files("value", str(file_path))


def test_filesystem_search_passes_file_glob(monkeypatch, isolated_paths) -> None:
    captured: dict[str, object] = {}
    file_path = Path(isolated_paths["workspace"]) / "sample.py"
    file_path.write_text("value = 1\n", encoding="utf-8")

    class _Result:
        returncode = 0
        stdout = "match\n"
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(filesystem.subprocess, "run", fake_run)

    result = filesystem.search_files("value", str(isolated_paths["workspace"]), file_glob="*.py")

    assert result == "match\n"
    assert "--glob" in captured["cmd"]


def test_filesystem_helpers_handle_general_exceptions_and_edge_directory_cases(isolated_paths, monkeypatch) -> None:
    file_path = Path(isolated_paths["workspace"]) / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")
    empty_dir = Path(isolated_paths["workspace"]) / "empty"
    empty_dir.mkdir()

    assert filesystem.list_dir(str(empty_dir)).startswith("(empty directory:")
    assert filesystem.read_file(str(empty_dir)).startswith("[ERROR: not a file:")
    assert filesystem.str_replace_file(str(empty_dir), "a", "b").startswith("[ERROR: not a file:")

    original_safe_path = filesystem._safe_path

    def _boom(path: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(filesystem, "_safe_path", _boom)
    assert filesystem.read_file("x") == "[ERROR: boom]"
    assert filesystem.write_file("x", "y") == "[ERROR: boom]"
    assert filesystem.list_dir("x") == "[ERROR: boom]"
    assert filesystem.delete_file("x") == "[ERROR: boom]"
    assert filesystem.str_replace_file("x", "a", "b") == "[ERROR: boom]"
    assert filesystem.search_files("x", "y") == "[ERROR: boom]"

    monkeypatch.setattr(filesystem, "_safe_path", original_safe_path)


def test_filesystem_list_dir_handles_entry_stat_errors(isolated_paths, monkeypatch) -> None:
    file_path = Path(isolated_paths["workspace"]) / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    original_stat = Path.stat

    def _broken_stat(self: Path, *args, **kwargs):
        if self.name == "sample.txt":
            raise OSError("stat failed")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _broken_stat)

    listing = filesystem.list_dir(".")

    assert listing == "[ERROR: stat failed]"


def test_filesystem_list_dir_renders_unknown_entry_when_entry_stat_fails(monkeypatch) -> None:
    class _Entry:
        name = "broken.txt"

        def is_file(self) -> bool:
            return True

        def is_dir(self) -> bool:
            return False

        def stat(self):
            raise OSError("stat failed")

    class _Dir:
        def exists(self) -> bool:
            return True

        def is_dir(self) -> bool:
            return True

        def iterdir(self):
            return [_Entry()]

        def __str__(self) -> str:
            return "/fake"

    monkeypatch.setattr(filesystem, "_safe_path", lambda path: _Dir())

    listing = filesystem.list_dir("fake")

    assert "????" in listing
    assert "broken.txt" in listing
