"""
Self-edit tools: allow the agent to modify its own skills, identity files,
and optionally restart its own container for code-level changes.

Self-modification is *bounded*:
  - Skills (agent/skills/*.md) — freely editable
  - Identity (agent/identity/*.md) — freely editable
  - Core Python source — editable via git commit + container restart
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import structlog

from agent.config import settings

log = structlog.get_logger()

_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*$")


def _resolve_skill_path(name: str) -> Path | None:
    """Return path for a skill file. Name must be slug-form (no .md extension needed)."""
    clean = name.lower().rstrip(".md")
    if not _SKILL_NAME_RE.match(clean):
        return None
    return settings.skills_path / f"{clean}.md"


def edit_skill(name: str, content: str) -> str:
    """
    Create or overwrite a skill file.

    Args:
        name: Skill name (slug, e.g. 'web-research').  .md extension optional.
        content: Full Markdown content for the skill.

    Returns:
        Confirmation message or error.
    """
    path = _resolve_skill_path(name)
    if path is None:
        return "[ERROR: skill name must be lowercase alphanumeric with hyphens]"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        log.info("skill_edited", skill=name, size=len(content))
        return f"Skill '{name}' saved to {path} ({len(content)} chars)."
    except Exception as exc:
        return f"[ERROR: {exc}]"


def read_skill(name: str) -> str:
    """
    Read a skill file.

    Args:
        name: Skill name (slug).

    Returns:
        Skill content or error.
    """
    path = _resolve_skill_path(name)
    if path is None:
        return "[ERROR: invalid skill name]"
    if not path.exists():
        return f"[ERROR: skill '{name}' not found]"
    return path.read_text(encoding="utf-8")


def list_skills() -> str:
    """List all available skills with their first line (title)."""
    skills_dir = settings.skills_path
    if not skills_dir.exists():
        return "(no skills directory found)"

    skills = sorted(p for p in skills_dir.glob("*.md") if not p.name.startswith("_"))
    if not skills:
        return "(no skills found)"

    lines = ["Available skills:", ""]
    for sk in skills:
        try:
            first_line = sk.read_text(encoding="utf-8").splitlines()[0].lstrip("#").strip()
        except Exception:
            first_line = "(unreadable)"
        lines.append(f"  • {sk.stem}: {first_line}")
    return "\n".join(lines)


def edit_identity(filename: str, content: str) -> str:
    """
    Update an identity file (IDENTITY.md, GOALS.md, or MEMORY.md).

    Args:
        filename: Must be one of IDENTITY.md, GOALS.md, MEMORY.md.
        content: New Markdown content.

    Returns:
        Confirmation or error.
    """
    allowed = {"IDENTITY.md", "GOALS.md", "MEMORY.md"}
    if filename not in allowed:
        return f"[ERROR: filename must be one of {allowed}]"

    path = settings.identity_path / filename
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        log.info("identity_edited", file=filename, size=len(content))
        return f"Identity file '{filename}' updated ({len(content)} chars)."
    except Exception as exc:
        return f"[ERROR: {exc}]"


def read_identity(filename: str) -> str:
    """
    Read an identity file.

    Args:
        filename: IDENTITY.md, GOALS.md, or MEMORY.md.
    """
    allowed = {"IDENTITY.md", "GOALS.md", "MEMORY.md"}
    if filename not in allowed:
        return f"[ERROR: filename must be one of {allowed}]"
    path = settings.identity_path / filename
    if not path.exists():
        return f"(file '{filename}' does not exist yet)"
    return path.read_text(encoding="utf-8")


async def self_restart(reason: str = "code update") -> str:
    """
    Restart this agent's own Docker container.

    This is used after code-level self-modifications (git commit + restart).
    The agent will be unreachable for ~10–30 seconds during restart.

    Args:
        reason: Human-readable reason for the restart (logged).

    Returns:
        Confirmation that restart was initiated.
    """
    if not settings.docker_restart_self:
        return "[DISABLED: DOCKER_RESTART_SELF=false]"

    container = settings.agent_container_name
    log.info("self_restart_initiated", container=container, reason=reason)

    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "restart", container,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        if proc.returncode == 0:
            return f"Container '{container}' restart initiated. Reason: {reason}"
        return f"[ERROR restarting container: {stderr.decode().strip()}]"
    except asyncio.TimeoutError:
        return f"[Restart command timed out — container '{container}' may still restart]"
    except Exception as exc:
        return f"[ERROR: {exc}]"
