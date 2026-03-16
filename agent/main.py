"""
Main entrypoint. Wires everything together and starts the agent.

Usage:
  python -m agent.main          # Start Discord bot + agent loop
  python -m agent.main run "task text"  # Run a single task and exit
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass

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


@dataclass
class RuntimeServices:
    sqlite: object
    postgres: object | None
    loop: object
    bot: object | None


@cli.command()
def start() -> None:
    """Start the agent (Discord bot + reasoning loop)."""
    asyncio.run(_start())


@cli.command()
def run(task: str = typer.Argument(..., help="Task text to run once and exit")) -> None:
    """Run a single task and print the result."""
    asyncio.run(_run_once(task))


async def _start() -> None:
    from importlib.metadata import version as _pkg_version

    _agent_version = _pkg_version("agent")

    log.info("agent_starting", name=settings.agent_name, model=settings.agent_model, version=_agent_version)
    runtime = await _build_runtime(start_discord=settings.has_discord)

    ev_loop = asyncio.get_running_loop()
    main_task: asyncio.Task | None = None

    async def _shutdown(sig_name: str) -> None:
        log.info("shutdown_signal", signal=sig_name)
        await _shutdown_runtime(runtime)
        if main_task is not None:
            main_task.cancel()

    def _handle_signal(sig_name: str) -> None:
        ev_loop.create_task(_shutdown(sig_name))

    _install_signal_handlers(ev_loop, _handle_signal)

    # ── Run ───────────────────────────────────────────────────────────────────
    if settings.has_discord:
        main_task = asyncio.ensure_future(asyncio.gather(
            runtime.loop.run_forever(),
            runtime.bot.start_bot(),
        ))
        try:
            await main_task
        except asyncio.CancelledError:
            pass
    else:
        log.warning("no_discord_token_running_headless")
        await runtime.loop.run_forever()


async def _run_once(task: str) -> None:
    runtime = await _build_runtime(start_discord=False)
    try:
        result = await runtime.loop.run_once(task)
        print(result.output)
        if not result.success:
            sys.exit(1)
    finally:
        await _shutdown_runtime(runtime)


async def _build_runtime(*, start_discord: bool) -> RuntimeServices:
    from agent.communication.discord_bot import DiscordBot
    from agent.core import create_agents, set_postgres
    from agent.loop import AgentLoop
    from agent.memory.postgres_store import PostgresStore
    from agent.memory.sqlite_store import SQLiteStore
    from agent.tools.registry import ToolRegistry

    sqlite = SQLiteStore(settings.sqlite_path)
    await sqlite.init()

    postgres: PostgresStore | None = None
    if settings.has_postgres:
        postgres = PostgresStore(settings.postgres_url)
        await postgres.init()
        await postgres.register_agent()
        set_postgres(postgres)

    registry = ToolRegistry()
    registry.register_all(sqlite, postgres)
    agents = create_agents(registry)
    loop = AgentLoop(agents, memory_store=sqlite, postgres_store=postgres)
    bot = DiscordBot(loop) if start_discord else None
    return RuntimeServices(sqlite=sqlite, postgres=postgres, loop=loop, bot=bot)


async def _shutdown_runtime(runtime: RuntimeServices) -> None:
    postgres = runtime.postgres
    if postgres is not None:
        try:
            await postgres.set_offline()
            log.info("postgres_set_offline")
        except Exception:
            pass
    try:
        await runtime.sqlite.close()
    except Exception:
        pass
    runtime.loop.stop()


def _install_signal_handlers(
    ev_loop: asyncio.AbstractEventLoop,
    handler: callable,
) -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            ev_loop.add_signal_handler(sig, lambda s=sig.name: handler(s))
        except NotImplementedError:
            pass


if __name__ == "__main__":
    cli()
