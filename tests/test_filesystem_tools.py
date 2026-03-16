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
