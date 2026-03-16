from __future__ import annotations

from agent.tools import self_edit


def test_self_edit_skill_round_trip_and_listing(isolated_paths) -> None:
    save_result = self_edit.edit_skill("new-skill", "# New Skill\nDo the thing.\n")
    read_result = self_edit.read_skill("new-skill")
    list_result = self_edit.list_skills()

    assert "saved" in save_result.lower()
    assert "Do the thing." in read_result
    assert "new-skill" in list_result


def test_self_edit_identity_round_trip(isolated_paths) -> None:
    save_result = self_edit.edit_identity("GOALS.md", "# Goals\nShip.\n")
    read_result = self_edit.read_identity("GOALS.md")

    assert "updated" in save_result.lower()
    assert "Ship." in read_result


def test_self_edit_rejects_invalid_skill_name() -> None:
    result = self_edit.edit_skill("Bad Skill Name", "content")

    assert result.startswith("[ERROR:")
