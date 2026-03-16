"""
Tool registry: registers all agent tools with Pydantic AI.

Pydantic AI tools are plain async/sync functions decorated with @agent.tool.
We use a registry pattern so tools can be conditionally included and
the agent is assembled after all dependencies are available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from pydantic_ai import Agent

from agent.tools.toolsets import attach_all_tools

if TYPE_CHECKING:
    from agent.memory.postgres_store import PostgresStore
    from agent.memory.sqlite_store import SQLiteStore

log = structlog.get_logger()


class ToolRegistry:
    """Collects tool functions and attaches them to a Pydantic AI agent."""

    def __init__(self) -> None:
        self._sqlite: SQLiteStore | None = None
        self._postgres: PostgresStore | None = None

    def register_all(
        self,
        sqlite: SQLiteStore | None = None,
        postgres: PostgresStore | None = None,
    ) -> None:
        self._sqlite = sqlite
        self._postgres = postgres
        log.info("tools_registered", sqlite=sqlite is not None, postgres=postgres is not None)

    def attach_to_agent(self, agent: Agent) -> None:  # type: ignore[type-arg]
        """Attach all tool functions to the given Pydantic AI agent."""
        attach_all_tools(agent, sqlite=self._sqlite, postgres=self._postgres)
