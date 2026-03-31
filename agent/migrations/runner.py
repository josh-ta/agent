"""
Ordered migrations after base SCHEMA is applied.

Schema DDL lives in sqlite_store.SCHEMA; this runner advances agent_migration_version
for future data backfills and ordered ALTER steps.
"""

from __future__ import annotations

import aiosqlite
import structlog

log = structlog.get_logger()

CURRENT_MIGRATION_VERSION = 1

_META_KEY = "agent_migration_version"


async def run_sqlite_migrations(db: aiosqlite.Connection) -> None:
    async with db.execute("SELECT value FROM _meta WHERE key=?", (_META_KEY,)) as cur:
        row = await cur.fetchone()
    version = int(row[0]) if row and row[0] is not None else 0

    while version < CURRENT_MIGRATION_VERSION:
        version += 1
        await _apply_migration(db, version)
        await db.execute(
            "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
            (_META_KEY, str(version)),
        )
        log.info("sqlite_migration_applied", version=version)


async def _apply_migration(db: aiosqlite.Connection, version: int) -> None:
    if version == 1:
        # Reserved: initial migration slot (schema is created via SCHEMA script).
        return
    raise RuntimeError(f"Unknown migration version {version}")
