"""Native and slash Discord command parsing and handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import discord

from agent.communication.message_router import ParsedMessage
from agent.config import settings
from agent.project_memory import (
    remove_project_memory_facts,
    render_project_memory,
    save_project_memory_facts,
)

if TYPE_CHECKING:
    from agent.communication.discord_services import MessageHandlingService
    from agent.loop import AgentLoop


@dataclass(frozen=True)
class NativeCommand:
    name: str
    argument: str = ""

    @property
    def expects_task_text(self) -> bool:
        return self.name in {"replace", "queue", "remember", "unremember"}


def parse_native_command(content: str) -> NativeCommand | None:
    text = content.strip()
    if not text.startswith("/"):
        return None
    parts = text[1:].split(None, 1)
    if not parts:
        return None
    name = parts[0].strip().lower()
    argument = parts[1].strip() if len(parts) > 1 else ""
    if name not in {
        "status",
        "cancel",
        "force-cancel",
        "replace",
        "queue",
        "clear",
        "resume",
        "forget",
        "help",
        "memory",
        "remember",
        "unremember",
    }:
        return None
    return NativeCommand(name=name, argument=argument)


def command_help_text() -> str:
    return (
        "## Commands\n"
        "- `/status` — show active, queued, and waiting work\n"
        "- `/memory` — show saved project memory for this repo\n"
        "- `/remember <fact>` — save a repo-specific fact or preference\n"
        "- `/unremember <text>` — remove saved project-memory entries matching text\n"
        "- `/cancel` — stop the current task after the next safe step\n"
        "- `/force-cancel` — request immediate stop (same safe-step semantics, clearer intent)\n"
        "- `/replace <task>` — cancel current work and run a new task next\n"
        "- `/queue <task>` — add a task to the back of the queue\n"
        "- `/clear` — drop queued tasks in this channel\n"
        "- `/resume` — repeat the current waiting question\n"
        "- `/forget` — discard the current task and queued stale work\n"
        "- `/help` — show this help"
    )


class CommandHandler:
    """Handles native `/command` messages in the agent private channel."""

    def __init__(self, service: MessageHandlingService) -> None:
        self._service = service

    @property
    def _loop(self) -> AgentLoop:
        return self._service._agent_loop  # noqa: SLF001

    async def handle(
        self,
        *,
        message: discord.Message,
        parsed: ParsedMessage,
        command: NativeCommand,
        task_content: str,
        attachment_metadata: dict[str, Any],
    ) -> bool:
        if not self._service._is_private_channel(parsed.channel_id):  # noqa: SLF001
            return False

        await self._service._acknowledge_message(message)  # noqa: SLF001

        if command.name == "help":
            await self._service._reply_safe(message, command_help_text())  # noqa: SLF001
            return True

        if command.name == "status":
            await self._service._reply_safe(message, self._format_status(parsed.channel_id))  # noqa: SLF001
            return True

        if command.name == "memory":
            await self._service._reply_safe(message, render_project_memory())  # noqa: SLF001
            return True

        if command.name in {"remember", "unremember"} and not task_content.strip():
            await self._service._reply_safe(message, f"Usage: `/{command.name} <text>`")  # noqa: SLF001
            return True

        if command.name == "remember":
            added = save_project_memory_facts([task_content])
            await self._service._reply_safe(  # noqa: SLF001
                message,
                "🧠 Saved that to project memory."
                if added
                else "💬 That project-memory fact was already saved.",
            )
            return True

        if command.name == "unremember":
            removed = remove_project_memory_facts(task_content)
            await self._service._reply_safe(  # noqa: SLF001
                message,
                f"🧹 Removed {removed} matching project-memory entr{'y' if removed == 1 else 'ies'}."
                if removed
                else "💬 I couldn't find a matching project-memory entry to remove.",
            )
            return True

        if command.name == "resume":
            waiting = self._loop.wait_registry.pending_for_channel(parsed.channel_id)
            if len(waiting) == 1:
                await self._service._reply_safe(message, f"❓ {waiting[0].question}")  # noqa: SLF001
            elif len(waiting) > 1:
                await self._service._reply_safe(  # noqa: SLF001
                    message,
                    "💬 I have more than one suspended question. Reply directly to the specific one you mean.",
                )
            else:
                await self._service._reply_safe(message, "💬 There is no suspended question to resume right now.")  # noqa: SLF001
            return True

        if command.name == "clear":
            removed = await self._service._clear_queued_channel_tasks(  # noqa: SLF001
                channel_id=parsed.channel_id,
                reason="Cleared by operator command.",
            )
            await self._service._reply_safe(  # noqa: SLF001
                message,
                f"🧹 Cleared {removed} queued task{'s' if removed != 1 else ''}."
                if removed
                else "💬 There were no queued tasks to clear.",
            )
            return True

        if command.name in {"cancel", "forget", "force-cancel"}:
            return await self._handle_cancel(message, parsed, command)

        if command.name in {"replace", "queue"} and not task_content.strip():
            await self._service._reply_safe(message, f"Usage: `/{command.name} <task>`")  # noqa: SLF001
            return True

        if command.name == "queue":
            await self._service._enqueue_deferred_task(  # noqa: SLF001
                parsed=parsed,
                message=message,
                task_content=task_content,
                attachment_metadata=attachment_metadata,
                front=False,
                session_id_seed="",
            )
            await self._service._reply_safe(message, "📝 Queued that task.")  # noqa: SLF001
            return True

        if command.name == "replace":
            await self._service._clear_queued_channel_tasks(  # noqa: SLF001
                channel_id=parsed.channel_id,
                reason="Replaced by a newer operator task.",
            )
            await self._service._request_cancel_active_task(  # noqa: SLF001
                channel_id=parsed.channel_id,
                reason="Operator issued /replace. Stop after the current safe step and hand off to the replacement task.",
            )
            await self._service._enqueue_deferred_task(  # noqa: SLF001
                parsed=parsed,
                message=message,
                task_content=task_content,
                attachment_metadata=attachment_metadata,
                front=True,
                session_id_seed="",
            )
            await self._service._reply_safe(message, "⏭️ Replacing the current task. Your new task is next.")  # noqa: SLF001
            return True

        return False

    def _format_status(self, channel_id: int) -> str:
        if hasattr(self._loop, "describe_work"):
            base = self._loop.describe_work(channel_id=channel_id)
        else:
            waiting = len(self._loop.wait_registry.pending_for_channel(channel_id))
            queue_size = self._loop.queue.qsize() if hasattr(self._loop, "queue") else 0
            active = "yes" if getattr(self._loop, "has_pending_work", False) else "no"
            base = f"Active: {active}\nQueued: {queue_size}\nWaiting for user: {waiting}"
        tier_hint = f"Default tier routing: fast=`{settings.model_fast}`, smart=`{settings.model_smart}`, best=`{settings.model_best}`"
        threads = "on" if settings.discord_use_task_threads else "off"
        return f"{base}\n\nModel default: `{settings.agent_model}`\nTask threads: {threads}\n{tier_hint}"

    async def _handle_cancel(
        self,
        message: discord.Message,
        parsed: ParsedMessage,
        command: NativeCommand,
    ) -> bool:
        if command.name == "forget":
            self._service._sticky_sessions.pop(parsed.channel_id, None)  # noqa: SLF001
        removed = await self._service._clear_queued_channel_tasks(  # noqa: SLF001
            channel_id=parsed.channel_id,
            reason="Cancelled by operator command.",
        )
        hard = command.name == "force-cancel"
        cancel_reason = (
            "Operator issued /forget. Stop after the current safe step, discard the old task, and acknowledge cancellation."
            if command.name == "forget"
            else (
                "Operator issued /force-cancel. Stop immediately after the current safe step and acknowledge cancellation."
                if hard
                else "Operator issued /cancel. Stop after the current safe step and acknowledge cancellation."
            )
        )
        cancelled = await self._service._request_cancel_active_task(  # noqa: SLF001
            channel_id=parsed.channel_id,
            reason=cancel_reason,
        )
        if cancelled:
            self._service._session_state.mark_cancelling(parsed.channel_id)  # noqa: SLF001
            if command.name == "forget":
                text = "🛑 Cancelling — I'll stop after the current step and discard stale work."
            elif hard:
                text = "🛑 Force cancelling — I'll stop after the current step."
            else:
                text = "⏸️ Cancelling — I'll stop after the current step."
            await self._service._reply_safe(message, text)  # noqa: SLF001
        elif removed:
            await self._service._reply_safe(  # noqa: SLF001
                message,
                f"🧹 Cleared {removed} queued task{'s' if removed != 1 else ''}.",
            )
        else:
            await self._service._reply_safe(message, "💬 There is no active or queued task to cancel.")  # noqa: SLF001
        return True
