from __future__ import annotations

from pathlib import Path

import pytest

from agent.export_delivery import (
    attachments_for_paths,
    attachments_from_registered_exports,
    bare_export_filenames_in_text,
    register_export_path,
    take_export_paths,
)
from agent.config import settings


def test_register_and_take_export_paths() -> None:
    take_export_paths()
    register_export_path("/workspace/a.csv")
    register_export_path("/workspace/a.csv")
    register_export_path("/workspace/b.csv")
    assert take_export_paths() == ["/workspace/a.csv", "/workspace/b.csv"]
    assert take_export_paths() == []


def test_bare_export_filenames_in_text() -> None:
    names = bare_export_filenames_in_text(
        "File: upcoming_arena_stadium_events_ticket_limit_4.csv with 251 rows"
    )
    assert names == ["upcoming_arena_stadium_events_ticket_limit_4.csv"]


def test_attachments_for_paths_reads_workspace_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "workspace_path", tmp_path)
    export = tmp_path / "export.csv"
    export.write_text("id,name\n1,Show", encoding="utf-8")

    attachments = attachments_for_paths(["export.csv"])
    assert len(attachments) == 1
    assert attachments[0].filename == "export.csv"
    assert attachments[0].data == b"id,name\n1,Show"


def test_attachments_from_registered_exports(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "workspace_path", tmp_path)
    export = tmp_path / "registered.csv"
    export.write_text("a,b\n1,2", encoding="utf-8")
    take_export_paths()
    register_export_path(str(export))

    attachments = attachments_from_registered_exports()
    assert len(attachments) == 1
    assert attachments[0].filename == "registered.csv"
    assert take_export_paths() == []
