"""
Tool registry: registers all agent tools with Pydantic AI.

Pydantic AI tools are plain async/sync functions decorated with @agent.tool.
We use a registry pattern so tools can be conditionally included and
the agent is assembled after all dependencies are available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from pydantic_ai import Agent, RunContext

from agent.config import settings
from agent.tools import filesystem, self_edit, shell
from agent.tools.discord_tools import discord_read, discord_read_named, discord_send

if TYPE_CHECKING:
    from agent.memory.sqlite_store import SQLiteStore
    from agent.memory.postgres_store import PostgresStore

log = structlog.get_logger()


class ToolRegistry:
    """Collects tool functions and attaches them to a Pydantic AI agent."""

    def __init__(self) -> None:
        self._sqlite: SQLiteStore | None = None
        self._postgres: PostgresStore | None = None

    def register_all(
        self,
        sqlite: "SQLiteStore | None" = None,
        postgres: "PostgresStore | None" = None,
    ) -> None:
        self._sqlite = sqlite
        self._postgres = postgres
        log.info("tools_registered", sqlite=sqlite is not None, postgres=postgres is not None)

    def attach_to_agent(self, agent: Agent) -> None:  # type: ignore[type-arg]
        """Attach all tool functions to the given Pydantic AI agent."""
        sqlite = self._sqlite
        postgres = self._postgres

        # ── Shell ─────────────────────────────────────────────────────────────
        @agent.tool_plain
        async def run_shell(command: str, working_dir: str = "", timeout: int = 30) -> str:
            """Run a shell command. Returns combined stdout+stderr + exit code."""
            return await shell.shell_run(command, working_dir or None, timeout)

        # ── Filesystem ────────────────────────────────────────────────────────
        @agent.tool_plain
        def read_file(path: str) -> str:
            """Read a file. Path is relative to workspace or absolute."""
            return filesystem.read_file(path)

        @agent.tool_plain
        def write_file(path: str, content: str) -> str:
            """Write content to a file (creates parents as needed)."""
            return filesystem.write_file(path, content)

        @agent.tool_plain
        def list_dir(path: str = ".") -> str:
            """List directory contents with sizes. Defaults to workspace root."""
            return filesystem.list_dir(path)

        @agent.tool_plain
        def delete_file(path: str) -> str:
            """Delete a file (not directories)."""
            return filesystem.delete_file(path)

        # ── Self-edit ─────────────────────────────────────────────────────────
        @agent.tool_plain
        def skill_list() -> str:
            """List all available skills."""
            return self_edit.list_skills()

        @agent.tool_plain
        def skill_read(name: str) -> str:
            """Read a skill file by name (slug, e.g. 'web-research')."""
            return self_edit.read_skill(name)

        @agent.tool_plain
        def skill_edit(name: str, content: str) -> str:
            """Create or update a skill file. Name must be a lowercase slug."""
            return self_edit.edit_skill(name, content)

        @agent.tool_plain
        def identity_read(filename: str) -> str:
            """Read IDENTITY.md, GOALS.md, or MEMORY.md."""
            return self_edit.read_identity(filename)

        @agent.tool_plain
        def identity_edit(filename: str, content: str) -> str:
            """Update IDENTITY.md, GOALS.md, or MEMORY.md with new content."""
            return self_edit.edit_identity(filename, content)

        @agent.tool_plain
        async def agent_restart(reason: str = "manual restart") -> str:
            """Restart this agent container (use after code changes)."""
            return await self_edit.self_restart(reason)

        # ── Discord ───────────────────────────────────────────────────────────
        @agent.tool_plain
        async def send_discord(channel_id: int, message: str) -> str:
            """Send a message to a Discord channel by its numeric ID."""
            return await discord_send(channel_id, message)

        @agent.tool_plain
        async def read_discord(channel_id: int, limit: int = 20) -> str:
            """Read recent messages from a Discord channel by numeric ID."""
            return await discord_read(channel_id, limit)

        @agent.tool_plain
        async def read_channel(name: str, limit: int = 20) -> str:
            """
            Read recent messages from a named channel.
            name: 'private' (your channel), 'bus' (#agent-bus), 'comms' (#agent-comms).
            Use this to catch up on conversation history or check what other agents said.
            """
            return await discord_read_named(name, limit)

        # ── Memory search ─────────────────────────────────────────────────────
        if sqlite:
            @agent.tool_plain
            async def memory_search(query: str, limit: int = 5) -> str:
                """Search past conversations and memory facts by keyword."""
                return await sqlite.search_memory(query, limit)

            @agent.tool_plain
            async def memory_save(content: str) -> str:
                """Save a fact or insight to long-term memory."""
                await sqlite.save_memory_fact(content)
                return f"Saved to memory: {content[:80]}"

            @agent.tool_plain
            async def lesson_save(summary: str, kind: str = "lesson") -> str:
                """
                Record a lesson, mistake, or insight so it is never forgotten.
                kind: 'lesson' | 'mistake' | 'insight' | 'pattern'
                Use 'mistake' when recording something that went wrong.
                """
                await sqlite.save_lesson(summary, kind=kind)
                return f"Lesson recorded [{kind}]: {summary[:80]}"

            @agent.tool_plain
            async def lesson_search(query: str, limit: int = 5) -> str:
                """Search past lessons and mistakes relevant to a topic."""
                result = await sqlite.search_lessons(query, limit)
                return result or f"(no lessons found for: {query})"

            @agent.tool_plain
            async def lessons_recent(limit: int = 10) -> str:
                """Show the most recent lessons and mistakes."""
                return await sqlite.get_recent_lessons(limit)

        # ── Postgres (multi-agent) ────────────────────────────────────────────
        if postgres:
            @agent.tool_plain
            async def list_agents() -> str:
                """List all agents registered in the shared database."""
                return await postgres.list_agents()

            @agent.tool_plain
            async def create_shared_task(to_agent: str, description: str) -> str:
                """Create a task for another agent in the shared task queue."""
                return await postgres.create_task(to_agent, description)

            @agent.tool_plain
            async def my_tasks() -> str:
                """List tasks assigned to this agent from the shared queue."""
                return await postgres.get_my_tasks()

        log.info("tools_attached_to_agent")
