"""
Main entrypoint. Wires everything together and starts the agent.

Usage:
  python -m agent.main          # Start Discord bot + agent loop
  python -m agent.main run "task text"  # Run a single task and exit
"""

from __future__ import annotations

import asyncio
import logging
import sys

import structlog
import typer

from agent.config import settings

# ── Logging setup ─────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    ),
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger()

cli = typer.Typer(name="agent", help="Autonomous AI agent")


@cli.command()
def start() -> None:
    """Start the agent (Discord bot + reasoning loop)."""
    asyncio.run(_start())


@cli.command()
def run(task: str = typer.Argument(..., help="Task text to run once and exit")) -> None:
    """Run a single task and print the result."""
    asyncio.run(_run_once(task))


async def _start() -> None:
    from agent.memory.sqlite_store import SQLiteStore
    from agent.memory.postgres_store import PostgresStore
    from agent.tools.registry import ToolRegistry
    from agent.core import create_agent
    from agent.loop import AgentLoop
    from agent.communication.discord_bot import DiscordBot

    log.info("agent_starting", name=settings.agent_name, model=settings.model_string)

    # Memory
    sqlite = SQLiteStore(settings.sqlite_path)
    await sqlite.init()

    postgres: PostgresStore | None = None
    if settings.has_postgres:
        postgres = PostgresStore(settings.postgres_url)
        await postgres.init()
        await postgres.register_agent()

    # Tools + Agent
    registry = ToolRegistry()
    registry.register_all(sqlite, postgres)

    agent = create_agent(registry)
    loop = AgentLoop(agent, memory_store=sqlite)

    if settings.has_discord:
        bot = DiscordBot(loop)
        # Run bot + loop concurrently
        await asyncio.gather(
            loop.run_forever(),
            bot.start_bot(),
        )
    else:
        log.warning("no_discord_token_running_headless")
        await loop.run_forever()


async def _run_once(task: str) -> None:
    from agent.memory.sqlite_store import SQLiteStore
    from agent.tools.registry import ToolRegistry
    from agent.core import create_agent
    from agent.loop import AgentLoop

    sqlite = SQLiteStore(settings.sqlite_path)
    await sqlite.init()

    registry = ToolRegistry()
    registry.register_all(sqlite, None)

    agent = create_agent(registry)
    loop = AgentLoop(agent, memory_store=sqlite)

    result = await loop.run_once(task)
    print(result.output)
    if not result.success:
        sys.exit(1)


if __name__ == "__main__":
    cli()
