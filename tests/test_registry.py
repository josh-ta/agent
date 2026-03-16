from __future__ import annotations

from agent.tools.registry import ToolRegistry


class _FakeAgent:
    def __init__(self) -> None:
        self.tool_names: list[str] = []

    def tool_plain(self, fn):
        self.tool_names.append(fn.__name__)
        return fn


class _FakeSQLite:
    async def get_stats(self):
        return {}

    async def search_memory(self, query: str, limit: int = 5) -> str:
        return ""

    async def save_memory_fact(self, content: str) -> None:
        return None

    async def save_lesson(self, summary: str, kind: str = "lesson") -> None:
        return None

    async def search_lessons(self, query: str, limit: int = 5) -> str:
        return ""

    async def get_recent_lessons(self, limit: int = 10) -> str:
        return ""


class _FakePostgres:
    async def get_stats(self):
        return {}

    async def list_agents(self) -> str:
        return ""

    async def create_task(self, to_agent: str, description: str) -> str:
        return ""

    async def get_my_tasks(self) -> str:
        return ""

    async def complete_task(self, task_id: str, result: str) -> str:
        return ""

    async def broadcast_message(self, message: str) -> str:
        return ""

    async def read_broadcasts(self, limit: int = 20) -> str:
        return ""

    async def share_memory(self, content: str) -> str:
        return ""

    async def search_shared_memory(self, query: str, limit: int = 5) -> str:
        return ""


def test_registry_attaches_base_toolsets() -> None:
    registry = ToolRegistry()
    agent = _FakeAgent()

    registry.attach_to_agent(agent)  # type: ignore[arg-type]

    assert "run_shell" in agent.tool_names
    assert "task_note" in agent.tool_names
    assert "skill_read" in agent.tool_names
    assert "send_discord" in agent.tool_names
    assert "gh_pr_view" in agent.tool_names


def test_registry_attaches_database_tools_when_stores_exist() -> None:
    registry = ToolRegistry()
    registry.register_all(_FakeSQLite(), _FakePostgres())  # type: ignore[arg-type]
    agent = _FakeAgent()

    registry.attach_to_agent(agent)  # type: ignore[arg-type]

    assert "memory_search" in agent.tool_names
    assert "lessons_recent" in agent.tool_names
    assert "list_agents" in agent.tool_names
    assert "search_shared_memory" in agent.tool_names
