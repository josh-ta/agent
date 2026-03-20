from __future__ import annotations

from agent.project_memory import (
    extract_project_memory_facts,
    load_project_memory,
    project_memory_path,
    remove_project_memory_facts,
    render_project_memory,
    save_project_memory_facts,
)


def test_project_memory_save_load_remove_and_render(isolated_paths) -> None:
    assert project_memory_path().parent == isolated_paths["workspace"]
    assert render_project_memory() == "## Project memory\n(no project memory recorded yet)"

    added = save_project_memory_facts(["Use pytest", "Use pytest", "Prefer staged deploys"])
    assert added == 2
    assert "Project memory" in load_project_memory()
    assert "Use pytest" in render_project_memory()

    removed = remove_project_memory_facts("pytest")
    assert removed == 1
    assert "Prefer staged deploys" in render_project_memory()

    removed_all = remove_project_memory_facts("staged deploys")
    assert removed_all == 1
    assert render_project_memory() == "## Project memory\n(no project memory recorded yet)"


def test_project_memory_extracts_operational_facts() -> None:
    facts = extract_project_memory_facts(
        "Workspace is /workspace/app. App host is root@example. Use scripts/deploy.sh. "
        "Starting: don't repeat the prompt. Check the file system first and do not guess."
    )

    lowered = {fact.lower() for fact in facts}
    assert any("/workspace" in fact for fact in lowered)
    assert any("app host is root@example" in fact for fact in lowered)
    assert any("scripts/deploy.sh" in fact for fact in lowered)
    assert any("do not echo the user's full prompt back" in fact for fact in lowered)
    assert any("file system" in fact for fact in lowered)
    assert any("do not guess" in fact for fact in lowered)


def test_project_memory_handles_empty_missing_and_truncated_cases(isolated_paths) -> None:
    path = project_memory_path()

    assert load_project_memory() == ""
    assert save_project_memory_facts(["   "]) == 0
    assert remove_project_memory_facts("missing") == 0

    path.write_text("", encoding="utf-8")
    assert load_project_memory() == ""

    path.write_text("# Project Memory\n\n- " + ("x" * 5000), encoding="utf-8")
    loaded = load_project_memory(char_cap=100)
    assert loaded.startswith("## Project memory\n[...truncated...]")

    assert extract_project_memory_facts("x" * 1300) == set()

