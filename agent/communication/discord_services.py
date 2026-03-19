"""
Discord-facing services used by `discord_bot`.

The gateway stays responsible for discord.py lifecycle, while these classes hold
message policy and event rendering.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import discord
import structlog

from agent.attachment_ingest import ingest_discord_attachments
from agent.communication.message_router import (
    MessageKind,
    ParsedMessage,
    a2a_to_task_content,
    classify,
)
from agent.config import settings
from agent.events import (
    AgentEvent,
    ProgressEvent,
    ShellDoneEvent,
    ShellOutputEvent,
    ShellStartEvent,
    TaskErrorEvent,
    TaskWaitingEvent,
    TaskStartEvent,
    TextTurnEndEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    bridge,
)
from agent.loop import Task
from agent.project_memory import (
    remove_project_memory_facts,
    render_project_memory,
    save_project_memory_facts,
)
from agent.session_router import SessionRouter, TurnIntent
from agent.tools import discord_tools as discord_tools_module

if TYPE_CHECKING:
    from agent.loop import AgentLoop

log = structlog.get_logger()

MAX_REPLY_LEN = 1990
SILENT_TOOLS = frozenset(
    {
        "read_file",
        "list_dir",
        "memory_save",
        "lesson_search",
        "task_resume",
        "task_journal_clear",
        "task_note",
        "memory_search",
        "lessons_recent",
        "lesson_save",
        "read_channel",
        "read_discord",
        "identity_read",
        "skill_list",
        "skill_read",
        "db_stats",
        "secret_list",
        "secret_set",
        "secret_get",
        "secret_delete",
    }
)


def escape_md_italics(text: str) -> str:
    return text.replace("*", "\\*")


def escape_codeblock(text: str) -> str:
    return text.replace("```", "`` `")


def format_args(args: object) -> str:
    if isinstance(args, dict):
        parts = []
        for key, value in args.items():
            text = str(value)
            parts.append(f"{key}={text[:60] + '…' if len(text) > 60 else text}")
        return ", ".join(parts)[:200]
    return str(args)[:200]


def allows_inline_reply(channel_id: int) -> bool:
    return channel_id != settings.discord_comms_channel_id


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


class DiscordEventPresenter:
    def __init__(self, client: discord.Client) -> None:
        self._client = client

    async def send_chunked(self, channel: discord.abc.Messageable, text: str) -> None:
        try:
            await discord_tools_module.send_text(channel, text, max_len=MAX_REPLY_LEN)
        except discord.HTTPException as exc:
            log.warning("send_chunked_failed", error=str(exc))

    async def send_attachments(
        self,
        channel: discord.abc.Messageable,
        attachments: list[discord_tools_module.DiscordAttachment],
    ) -> None:
        try:
            await discord_tools_module.send_attachments(channel, attachments)
        except discord.HTTPException as exc:
            log.warning("send_attachment_failed", error=str(exc))

    def make_sink(self, channel: discord.abc.Messageable):  # type: ignore[return]
        shell_lines: list[str] = []
        shell_msg: discord.Message | None = None

        async def _send(text: str) -> None:
            await self.send_chunked(channel, text)

        async def sink(event: AgentEvent) -> None:
            nonlocal shell_lines, shell_msg

            if isinstance(event, TaskStartEvent):
                await _send("🟢 Working on it.")
                return

            if isinstance(event, ThinkingEndEvent):
                if event.text:
                    for i in range(0, len(event.text), 1800):
                        chunk = event.text[i:i + 1800].strip()
                        if chunk:
                            await _send(f"🧠 *{escape_md_italics(chunk)}*")
                return

            if isinstance(event, TextTurnEndEvent):
                if not event.is_final and event.text:
                    await _send(f"💭 {event.text[:1900]}")
                return

            if isinstance(event, ToolCallStartEvent):
                if event.tool_name not in SILENT_TOOLS:
                    await _send(f"🔧 `{event.tool_name}({format_args(event.args)})`")
                return

            if isinstance(event, ShellStartEvent):
                shell_lines = []
                shell_msg = None
                try:
                    shell_msg = await channel.send(f"$ `{event.command[:200]}`")  # type: ignore[union-attr]
                except discord.HTTPException:
                    pass
                return

            if isinstance(event, ShellOutputEvent):
                shell_lines.append(event.chunk)
                return

            if isinstance(event, ShellDoneEvent):
                output = "".join(shell_lines).strip()
                shell_lines = []
                failed = event.exit_code != 0
                status = f"exit {event.exit_code} ({event.elapsed_s:.1f}s)" if failed else None
                if output:
                    body = f"```\n{escape_codeblock(output[-1400:])}\n```"
                    if status:
                        body += f"\n{status}"
                    try:
                        if shell_msg is not None:
                            await shell_msg.edit(content=f"{shell_msg.content}\n{body}")
                        else:
                            await _send(body)
                    except discord.HTTPException:
                        await _send(body)
                elif status:
                    try:
                        if shell_msg is not None:
                            await shell_msg.edit(content=f"{shell_msg.content} → {status}")
                        else:
                            await _send(status)
                    except discord.HTTPException:
                        await _send(status)
                shell_msg = None
                return

            if isinstance(event, ProgressEvent):
                if event.message:
                    await _send(event.message)
                return

            if isinstance(event, TaskErrorEvent):
                await _send(f"❌ {event.error[:400]}")

        return sink


class MessageHandlingService:
    def __init__(
        self,
        *,
        agent_loop: "AgentLoop",
        client: discord.Client,
        presenter: DiscordEventPresenter,
        event_bridge: Any = bridge,
    ) -> None:
        self._agent_loop = agent_loop
        self._client = client
        self._presenter = presenter
        self._bridge = event_bridge
        self._waiting_sink_tag = f"discord_waiting_prompt_{id(self)}"
        self._inject_queues: dict[int, asyncio.Queue[str]] = {}
        self._active_sessions: dict[int, str] = {}
        self._sticky_sessions: dict[int, str] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._session_router = SessionRouter()
        self._bridge.register(self._waiting_sink_tag, self._make_waiting_prompt_sink())

    @staticmethod
    def _command_help_text() -> str:
        return (
            "## Commands\n"
            "- `/status` — show active, queued, and waiting work\n"
            "- `/memory` — show saved project memory for this repo\n"
            "- `/remember <fact>` — save a repo-specific fact or preference\n"
            "- `/unremember <text>` — remove saved project-memory entries matching text\n"
            "- `/cancel` — stop the current task after the next safe step\n"
            "- `/replace <task>` — cancel current work and run a new task next\n"
            "- `/queue <task>` — add a task to the back of the queue\n"
            "- `/clear` — drop queued tasks in this channel\n"
            "- `/resume` — repeat the current waiting question\n"
            "- `/forget` — discard the current task and queued stale work\n"
            "- `/help` — show this help"
        )

    @staticmethod
    def _is_forget_request(text: str) -> bool:
        lowered = text.strip().lower()
        return lowered.startswith(("forget it", "forget that", "never mind", "nevermind", "scratch that", "/forget"))

    def _is_private_channel(self, channel_id: int) -> bool:
        return channel_id == settings.discord_agent_channel_id

    def _build_task_metadata(
        self,
        *,
        parsed: ParsedMessage,
        message: discord.Message,
        attachment_metadata: dict[str, Any],
        session_id_seed: str = "",
    ) -> dict[str, Any]:
        metadata_seed = dict(attachment_metadata)
        if session_id_seed:
            metadata_seed["session_id"] = session_id_seed
        metadata = self._session_router.build_metadata(
            source="discord",
            channel_id=parsed.channel_id,
            message_id=parsed.message_id,
            reference_message_id=getattr(getattr(message, "reference", None), "message_id", None),
            metadata=metadata_seed,
        )
        metadata["task_id"] = metadata.get("task_id") or f"discord-{parsed.channel_id}-{parsed.message_id}"
        return metadata

    async def _persist_task_record(
        self,
        *,
        metadata: dict[str, Any],
        parsed: ParsedMessage,
        task_content: str,
    ) -> None:
        if self._agent_loop.memory is not None and hasattr(self._agent_loop.memory, "create_task_record"):
            await self._agent_loop.memory.create_task_record(
                task_id=metadata["task_id"],
                source="discord",
                author=parsed.author,
                content=task_content,
                metadata=metadata,
            )

    async def _enqueue_deferred_task(
        self,
        *,
        parsed: ParsedMessage,
        message: discord.Message,
        task_content: str,
        attachment_metadata: dict[str, Any],
        front: bool = False,
        session_id_seed: str = "",
    ) -> Task:
        metadata = self._build_task_metadata(
            parsed=parsed,
            message=message,
            attachment_metadata=attachment_metadata,
            session_id_seed=session_id_seed,
        )
        self._sticky_sessions[parsed.channel_id] = metadata["session_id"]
        await self._persist_task_record(
            metadata=metadata,
            parsed=parsed,
            task_content=task_content,
        )
        response_future = asyncio.get_running_loop().create_future()
        task = Task(
            content=task_content,
            source="discord",
            author=parsed.author,
            channel_id=parsed.channel_id,
            message_id=parsed.message_id,
            metadata=metadata,
            inject_queue=asyncio.Queue(),
            response_future=response_future,
        )
        if front and hasattr(self._agent_loop, "enqueue_front"):
            await self._agent_loop.enqueue_front(task)
        else:
            await self._agent_loop.enqueue(task)
        if self._is_private_channel(parsed.channel_id):
            waiter = asyncio.create_task(
                self._wait_for_deferred_result(
                    parsed=parsed,
                    message=message,
                    response_future=response_future,
                )
            )
            self._background_tasks.add(waiter)
            waiter.add_done_callback(self._background_tasks.discard)
        return task

    async def _clear_queued_channel_tasks(self, *, channel_id: int, reason: str) -> int:
        if not hasattr(self._agent_loop, "clear_queued_tasks"):
            return 0
        removed = await self._agent_loop.clear_queued_tasks(
            source="discord",
            channel_id=channel_id,
            reason=reason,
        )
        return len(removed)

    async def _request_cancel_active_task(self, *, channel_id: int, reason: str) -> bool:
        if hasattr(self._agent_loop, "request_cancel_active_task"):
            return await self._agent_loop.request_cancel_active_task(
                channel_id=channel_id,
                reason=reason,
            )
        inject_q = self._inject_queues.get(channel_id)
        if inject_q is None:
            return False
        await inject_q.put(reason)
        return True

    async def _reply_safe(self, message: discord.Message, content: str) -> None:
        if not allows_inline_reply(message.channel.id):
            return
        try:
            await message.reply(content, mention_author=False)
        except discord.HTTPException:
            pass

    async def _wait_for_deferred_result(
        self,
        *,
        parsed: ParsedMessage,
        message: discord.Message,
        response_future: asyncio.Future[TaskResult],
    ) -> None:
        try:
            result = await response_future
        except Exception as exc:
            await self._reply_safe(message, f"❌ {exc}")
            return
        await self._handle_task_result(parsed=parsed, message=message, result=result)

    async def _handle_task_result(
        self,
        *,
        parsed: ParsedMessage,
        message: discord.Message,
        result: TaskResult,
    ) -> None:
        if result.waiting_for_user:
            task_id = str(result.task.metadata.get("task_id", "")).strip()
            suspended = self._agent_loop.wait_registry.get(task_id) if task_id else None
            prompt_message_id = suspended.prompt_message_id if suspended is not None else None
            if prompt_message_id is None:
                prompt_message_id = await self._send_waiting_prompt(
                    message,
                    result.question or "I need more information.",
                )
            if task_id and prompt_message_id is not None and (suspended is None or suspended.prompt_message_id is None):
                await self._record_waiting_prompt(
                    task_id=task_id,
                    metadata=result.task.metadata,
                    question=result.question or "",
                    prompt_message_id=prompt_message_id,
                )
            await self._mark_task_waiting(message)
            return

        delivery_confirmed = False
        if result.user_visible_reply_sent:
            if result.attachments:
                delivery_confirmed = await self.send_reply(parsed, "", message, attachments=result.attachments)
            else:
                delivery_confirmed = True
        elif not result.output:
            if result.attachments:
                delivery_confirmed = await self.send_reply(parsed, "", message, attachments=result.attachments)
        else:
            delivery_confirmed = await self.send_reply(parsed, result.output, message, attachments=result.attachments)

        await self._mark_task_finished(message, success=bool(result.success and delivery_confirmed))

    async def _handle_native_command(
        self,
        *,
        message: discord.Message,
        parsed: ParsedMessage,
        command: NativeCommand,
        task_content: str,
        attachment_metadata: dict[str, Any],
    ) -> bool:
        if not self._is_private_channel(parsed.channel_id):
            return False

        await self._acknowledge_message(message)

        if command.name == "help":
            await self._reply_safe(message, self._command_help_text())
            return True

        if command.name == "status":
            if hasattr(self._agent_loop, "describe_work"):
                status = self._agent_loop.describe_work(channel_id=parsed.channel_id)
            else:
                waiting = len(self._agent_loop.wait_registry.pending_for_channel(parsed.channel_id))
                queue_size = self._agent_loop.queue.qsize() if hasattr(self._agent_loop, "queue") else 0
                active = "yes" if getattr(self._agent_loop, "has_pending_work", False) else "no"
                status = f"Active: {active}\nQueued: {queue_size}\nWaiting for user: {waiting}"
            await self._reply_safe(message, status)
            return True

        if command.name == "memory":
            await self._reply_safe(message, render_project_memory())
            return True

        if command.name in {"remember", "unremember"} and not task_content.strip():
            await self._reply_safe(
                message,
                f"Usage: `/{command.name} <text>`",
            )
            return True

        if command.name == "remember":
            added = save_project_memory_facts([task_content])
            await self._reply_safe(
                message,
                "🧠 Saved that to project memory."
                if added
                else "💬 That project-memory fact was already saved.",
            )
            return True

        if command.name == "unremember":
            removed = remove_project_memory_facts(task_content)
            await self._reply_safe(
                message,
                f"🧹 Removed {removed} matching project-memory entr{'y' if removed == 1 else 'ies'}."
                if removed
                else "💬 I couldn't find a matching project-memory entry to remove.",
            )
            return True

        if command.name == "resume":
            waiting = self._agent_loop.wait_registry.pending_for_channel(parsed.channel_id)
            if len(waiting) == 1:
                await self._reply_safe(message, f"❓ {waiting[0].question}")
            elif len(waiting) > 1:
                await self._reply_safe(message, "💬 I have more than one suspended question. Reply directly to the specific one you mean.")
            else:
                await self._reply_safe(message, "💬 There is no suspended question to resume right now.")
            return True

        if command.name == "clear":
            removed = await self._clear_queued_channel_tasks(
                channel_id=parsed.channel_id,
                reason="Cleared by operator command.",
            )
            await self._reply_safe(
                message,
                f"🧹 Cleared {removed} queued task{'s' if removed != 1 else ''}."
                if removed
                else "💬 There were no queued tasks to clear.",
            )
            return True

        if command.name in {"cancel", "forget"}:
            if command.name == "forget":
                self._sticky_sessions.pop(parsed.channel_id, None)
            removed = await self._clear_queued_channel_tasks(
                channel_id=parsed.channel_id,
                reason="Cancelled by operator command.",
            )
            cancel_reason = (
                "Operator issued /forget. Stop after the current safe step, discard the old task, and acknowledge cancellation."
                if command.name == "forget"
                else "Operator issued /cancel. Stop after the current safe step and acknowledge cancellation."
            )
            cancelled = await self._request_cancel_active_task(
                channel_id=parsed.channel_id,
                reason=cancel_reason,
            )
            if cancelled:
                await self._reply_safe(
                    message,
                    (
                        "🛑 I’ll stop after the current safe step and discard the stale work."
                        if command.name == "forget"
                        else "⏸️ I’ll stop after the current safe step."
                    ),
                )
            elif removed:
                await self._reply_safe(message, f"🧹 Cleared {removed} queued task{'s' if removed != 1 else ''}.")
            else:
                await self._reply_safe(message, "💬 There is no active or queued task to cancel.")
            return True

        if command.name in {"replace", "queue"} and not task_content.strip():
            await self._reply_safe(message, f"Usage: `/{command.name} <task>`")
            return True

        if command.name == "queue":
            await self._enqueue_deferred_task(
                parsed=parsed,
                message=message,
                task_content=task_content,
                attachment_metadata=attachment_metadata,
                front=False,
                session_id_seed="",
            )
            await self._reply_safe(message, "📝 Queued that task.")
            return True

        if command.name == "replace":
            await self._clear_queued_channel_tasks(
                channel_id=parsed.channel_id,
                reason="Replaced by a newer operator task.",
            )
            await self._request_cancel_active_task(
                channel_id=parsed.channel_id,
                reason="Operator issued /replace. Stop after the current safe step and hand off to the replacement task.",
            )
            await self._enqueue_deferred_task(
                parsed=parsed,
                message=message,
                task_content=task_content,
                attachment_metadata=attachment_metadata,
                front=True,
                session_id_seed="",
            )
            await self._reply_safe(message, "⏭️ Replacing the current task. Your new task is next.")
            return True

        return False

    def _is_answering_pending_question(
        self,
        parsed: ParsedMessage,
        message: discord.Message,
    ) -> bool:
        decision = self._session_router.classify_turn(
            source="discord",
            channel_id=parsed.channel_id,
            message_id=parsed.message_id,
            reference_message_id=getattr(getattr(message, "reference", None), "message_id", None),
            content=parsed.content,
            wait_registry=self._agent_loop.wait_registry,
        )
        return decision.intent == TurnIntent.ANSWER_PENDING_QUESTION

    async def _acknowledge_message(self, message: discord.Message, emoji: str = "👀") -> None:
        try:
            await message.add_reaction(emoji)
        except discord.HTTPException:
            pass

    async def _mark_task_finished(self, message: discord.Message, *, success: bool) -> None:
        await self._acknowledge_message(message, "🏁" if success else "❌")

    async def _mark_task_waiting(self, message: discord.Message) -> None:
        await self._acknowledge_message(message, "⏸️")

    async def _send_waiting_prompt_to_channel(
        self,
        channel: discord.abc.Messageable,
        question: str,
    ) -> int | None:
        try:
            sent = await channel.send(f"❓ {question}")  # type: ignore[union-attr]
            return getattr(sent, "id", None)
        except discord.HTTPException:
            return None

    async def _send_waiting_prompt(self, message: discord.Message, question: str) -> int | None:
        try:
            sent = await message.reply(f"❓ {question}", mention_author=False)
            return getattr(sent, "id", None)
        except discord.HTTPException:
            return await self._send_waiting_prompt_to_channel(message.channel, question)

    async def _record_waiting_prompt(
        self,
        *,
        task_id: str,
        metadata: dict[str, Any] | None,
        question: str,
        prompt_message_id: int,
    ) -> None:
        self._agent_loop.wait_registry.bind_prompt_message(task_id, prompt_message_id)
        wait_state = dict((metadata or {}).get("wait_state") or {})
        wait_state["prompt_message_id"] = prompt_message_id
        if metadata is not None:
            metadata["wait_state"] = wait_state

        suspended = self._agent_loop.wait_registry.get(task_id)
        if suspended is not None:
            suspended.prompt_message_id = prompt_message_id
            suspended_wait_state = dict(suspended.metadata.get("wait_state") or {})
            suspended_wait_state["prompt_message_id"] = prompt_message_id
            suspended.metadata["wait_state"] = suspended_wait_state

        if self._agent_loop.memory is not None and hasattr(self._agent_loop.memory, "mark_task_waiting"):
            await self._agent_loop.memory.mark_task_waiting(
                task_id,
                metadata=metadata or (suspended.metadata if suspended is not None else {"wait_state": wait_state}),
                question=question,
            )

    def _make_waiting_prompt_sink(self):  # type: ignore[return]
        async def sink(event: AgentEvent) -> None:
            if not isinstance(event, TaskWaitingEvent):
                return
            if event.deliver_inline_reply or event.source != "discord":
                return

            task_id = str(event.task_id or "").strip()
            if not task_id:
                return
            suspended = self._agent_loop.wait_registry.get(task_id)
            if suspended is None or suspended.prompt_message_id is not None:
                return

            channel = self._client.get_channel(event.channel_id or suspended.channel_id)
            if channel is None:
                return

            question = event.question or suspended.question or "I need more information."
            prompt_message_id = await self._send_waiting_prompt_to_channel(channel, question)
            if prompt_message_id is None:
                return

            await self._record_waiting_prompt(
                task_id=task_id,
                metadata=suspended.metadata,
                question=question,
                prompt_message_id=prompt_message_id,
            )

        return sink

    async def _build_task_input(
        self,
        message: discord.Message,
        *,
        base_content: str,
    ) -> tuple[str, dict[str, Any], str]:
        task_content = base_content.strip()
        attachment_metadata: dict[str, Any] = {}
        combined_content = task_content
        attachments = getattr(message, "attachments", None) or []
        if attachments:
            bundle = await ingest_discord_attachments(
                attachments,
                root=settings.attachments_path,
                storage_key=f"discord-{message.channel.id}-{message.id}",
                max_bytes=settings.attachment_max_bytes,
                text_char_cap=settings.attachment_text_char_cap,
            )
            if bundle.metadata:
                attachment_metadata["attachments"] = bundle.metadata
                if not task_content:
                    task_content = "Please inspect the attached file(s) and help with them."
                combined_content = (
                    f"{task_content}\n\n---\n\n{bundle.prompt_text}"
                    if bundle.prompt_text
                    else task_content
                )
        return task_content, attachment_metadata, combined_content

    async def _maybe_resume_waiting_discord_task(self, message: discord.Message) -> Task | None:
        reference = getattr(getattr(message, "reference", None), "message_id", None)
        suspended = self._agent_loop.wait_registry.pop_for_discord_reply(
            channel_id=message.channel.id,
            reference_message_id=reference,
        )
        if suspended is None:
            return None
        answer, attachment_metadata, combined_answer = await self._build_task_input(
            message,
            base_content=message.content,
        )
        return self._agent_loop.build_resumed_task(
            suspended=suspended,
            answer=combined_answer or answer,
            author=getattr(message.author, "display_name", "user"),
            source="discord",
            metadata_overrides=attachment_metadata,
        )

    async def handle_message(self, message: discord.Message) -> None:
        assert self._client.user
        parsed = classify(message, self._client.user)

        if parsed.kind in {MessageKind.IGNORE, MessageKind.BUS}:
            return

        raw_task_content = (
            a2a_to_task_content(parsed.a2a_payload)
            if parsed.kind == MessageKind.A2A and parsed.a2a_payload
            else parsed.content
        )
        native_command = (
            parse_native_command(raw_task_content)
            if parsed.kind == MessageKind.TASK and self._is_private_channel(parsed.channel_id)
            else None
        )
        sticky_session_id = (
            self._sticky_sessions.get(parsed.channel_id, "")
            if self._is_private_channel(parsed.channel_id) and native_command is None
            else ""
        )
        base_content = native_command.argument if native_command and native_command.expects_task_text else raw_task_content
        task_content, attachment_metadata, combined_content = await self._build_task_input(
            message,
            base_content=base_content,
        )
        if native_command is not None:
            handled = await self._handle_native_command(
                message=message,
                parsed=parsed,
                command=native_command,
                task_content=task_content,
                attachment_metadata=attachment_metadata,
            )
            if handled:
                return

        if not task_content.strip():
            return

        private_channel = self._client.get_channel(settings.discord_agent_channel_id)
        decision = self._session_router.classify_turn(
            source="discord",
            channel_id=parsed.channel_id,
            message_id=parsed.message_id,
            reference_message_id=getattr(getattr(message, "reference", None), "message_id", None),
            content=combined_content,
            metadata={
                "session_id": self._active_sessions.get(parsed.channel_id, "") or sticky_session_id,
                **attachment_metadata,
            },
            has_active_task=parsed.channel_id in self._inject_queues,
            wait_registry=self._agent_loop.wait_registry,
        )
        resumed_task = await self._maybe_resume_waiting_discord_task(message)
        if resumed_task is not None:
            await self._acknowledge_message(message)
            task_id = str(resumed_task.metadata.get("task_id", "")).strip()
            if task_id and hasattr(self._agent_loop.memory, "mark_task_queued"):
                await self._agent_loop.memory.mark_task_queued(task_id, metadata=resumed_task.metadata)
            if self._agent_loop.memory is not None and hasattr(self._agent_loop.memory, "append_session_turn"):
                await self._agent_loop.memory.append_session_turn(
                    session_id=str((resumed_task.metadata or {}).get("session_id", "")),
                    role="user",
                    content=combined_content,
                    turn_kind="answer",
                    task_id=task_id,
                    metadata={"author": getattr(message.author, "display_name", "user")},
                )
            if allows_inline_reply(parsed.channel_id):
                try:
                    await message.reply(
                        "💬 Got it — resuming from your answer now.",
                        mention_author=False,
                    )
                except discord.HTTPException:
                    pass
            self._sticky_sessions[parsed.channel_id] = str((resumed_task.metadata or {}).get("session_id", "")).strip()
            await self._agent_loop.enqueue(resumed_task)
            return

        if (
            parsed.channel_id == settings.discord_agent_channel_id
            and getattr(message, "reference", None) is None
            and len(self._agent_loop.wait_registry.pending_for_channel(parsed.channel_id)) > 1
        ):
            await self._acknowledge_message(message)
            try:
                await message.reply(
                    "💬 I have more than one suspended question. Reply directly to the specific question you mean.",
                    mention_author=False,
                )
            except discord.HTTPException:
                pass
            return

        inject_q = self._inject_queues.get(parsed.channel_id)
        if inject_q is not None:
            if decision.intent == TurnIntent.CANCEL_OR_PAUSE:
                await self._acknowledge_message(message)
                forget_request = self._is_forget_request(combined_content)
                if forget_request:
                    self._sticky_sessions.pop(parsed.channel_id, None)
                    await self._clear_queued_channel_tasks(
                        channel_id=parsed.channel_id,
                        reason="Discarded after operator said to forget the task.",
                    )
                await self._request_cancel_active_task(
                    channel_id=parsed.channel_id,
                    reason=(
                        "Operator said to forget the current task. Stop after the current safe step, discard stale work, and acknowledge cancellation."
                        if forget_request
                        else "User asked to pause/cancel after the current step."
                    ),
                )
                await self._reply_safe(
                    message,
                    (
                        "🛑 I’ll stop after the current step and discard the stale work."
                        if forget_request
                        else "⏸️ I can't safely interrupt the running tool chain yet, but I'll stop after the current step."
                    ),
                )
                return
            if self._is_private_channel(parsed.channel_id) and decision.intent == TurnIntent.START_NEW_TASK:
                await self._clear_queued_channel_tasks(
                    channel_id=parsed.channel_id,
                    reason="Replaced by a newer operator task.",
                )
                await self._request_cancel_active_task(
                    channel_id=parsed.channel_id,
                    reason="Operator sent a new task. Stop after the current safe step and hand off to the replacement task.",
                )
                await self._enqueue_deferred_task(
                    parsed=parsed,
                    message=message,
                    task_content=task_content,
                    attachment_metadata=attachment_metadata,
                    front=True,
                    session_id_seed="",
                )
                await self._acknowledge_message(message)
                await self._reply_safe(message, "⏭️ Replacing the current task. Your new task is next.")
                return
            await inject_q.put(combined_content)
            await self._acknowledge_message(message)
            await self._reply_safe(
                message,
                (
                    "💬 Got it — I'll update the current task with that new constraint."
                    if decision.intent == TurnIntent.CLARIFICATION_OR_NEW_CONSTRAINT
                    else "💬 Got it — I'll fold that into what I'm working on."
                ),
            )
            return

        if self._agent_loop.has_pending_work:
            if self._is_private_channel(parsed.channel_id) and decision.intent == TurnIntent.START_NEW_TASK:
                await self._clear_queued_channel_tasks(
                    channel_id=parsed.channel_id,
                    reason="Replaced by a newer operator task.",
                )
                await self._enqueue_deferred_task(
                    parsed=parsed,
                    message=message,
                    task_content=task_content,
                    attachment_metadata=attachment_metadata,
                    front=True,
                    session_id_seed="",
                )
                await self._acknowledge_message(message)
                await self._reply_safe(message, "⏭️ Your new task is queued next.")
                return
            await self._enqueue_deferred_task(
                parsed=parsed,
                message=message,
                task_content=task_content,
                attachment_metadata=attachment_metadata,
                front=False,
                session_id_seed=sticky_session_id,
            )
            await self._acknowledge_message(message)
            position = f"#{self._agent_loop.queue.qsize()}" if self._agent_loop.queue.qsize() > 1 else "next"
            await self._reply_safe(
                message,
                f"⏸️ I'm still working on the previous task — queued yours ({position} up).",
            )
            return

        inject_q = asyncio.Queue()
        self._inject_queues[parsed.channel_id] = inject_q
        response_future = asyncio.get_running_loop().create_future()
        metadata = self._build_task_metadata(
            parsed=parsed,
            message=message,
            attachment_metadata=attachment_metadata,
            session_id_seed=sticky_session_id,
        )
        self._active_sessions[parsed.channel_id] = metadata["session_id"]
        self._sticky_sessions[parsed.channel_id] = metadata["session_id"]
        await self._persist_task_record(
            metadata=metadata,
            parsed=parsed,
            task_content=task_content,
        )
        if self._agent_loop.memory is not None and hasattr(self._agent_loop.memory, "ensure_session"):
            await self._agent_loop.memory.ensure_session(
                session_id=metadata["session_id"],
                source="discord",
                channel_id=parsed.channel_id,
                title=task_content[:120],
                status="active",
                pending_task_id=metadata["task_id"],
                metadata={"author": parsed.author},
            )
        task = Task(
            content=task_content,
            source="discord",
            author=parsed.author,
            channel_id=parsed.channel_id,
            message_id=parsed.message_id,
            metadata=metadata,
            inject_queue=inject_q,
            response_future=response_future,
        )

        sink_tag = f"discord_{parsed.channel_id}_{id(task)}"
        if private_channel is not None:
            self._bridge.register(sink_tag, self._presenter.make_sink(private_channel))  # type: ignore[arg-type]

        typing_ctx = private_channel.typing() if private_channel is not None else message.channel.typing()  # type: ignore[union-attr]
        try:
            async with typing_ctx:
                await self._agent_loop.enqueue(task)
                await self._acknowledge_message(message)
                result = await response_future
        finally:
            self._inject_queues.pop(parsed.channel_id, None)
            self._active_sessions.pop(parsed.channel_id, None)
            self._bridge.unregister(sink_tag)

        await self._handle_task_result(parsed=parsed, message=message, result=result)

    async def send_reply(
        self,
        parsed: ParsedMessage,
        output: str,
        original_message: discord.Message,
        attachments: list[discord_tools_module.DiscordAttachment] | None = None,
    ) -> bool:
        is_a2a = parsed.kind == MessageKind.A2A and parsed.a2a_payload is not None
        is_bus = parsed.channel_id == settings.discord_bus_channel_id
        attachments = attachments or []

        if is_a2a:
            from_agent = parsed.a2a_payload.get("from", "")
            delivered = False
            if from_agent and settings.discord_comms_channel_id:
                comms = self._client.get_channel(settings.discord_comms_channel_id)
                if comms is not None:
                    try:
                        await comms.send(
                            json.dumps(
                                {
                                    "from": settings.agent_name,
                                    "to": from_agent,
                                    "task": "result",
                                    "payload": output[:1800],
                                }
                            )
                        )  # type: ignore[union-attr]
                        delivered = True
                    except discord.HTTPException as exc:
                        log.warning("a2a_reply_failed", error=str(exc))
            await self.post_bus_status(f"**{settings.agent_name}** completed task from {from_agent or 'unknown'}.")
            return delivered or not from_agent

        if is_bus:
            await self.post_bus_status(f"**{settings.agent_name}**: {output[:300]}")
            private_channel = self._client.get_channel(settings.discord_agent_channel_id)
            if private_channel is not None:
                await self._presenter.send_chunked(private_channel, output)  # type: ignore[arg-type]
                if attachments:
                    await self._presenter.send_attachments(private_channel, attachments)  # type: ignore[arg-type]
            return True

        private_channel = self._client.get_channel(settings.discord_agent_channel_id)
        target = private_channel or original_message.channel
        try:
            chunks = [output[i:i + MAX_REPLY_LEN] for i in range(0, len(output), MAX_REPLY_LEN)]
            if original_message.channel.id == settings.discord_agent_channel_id and chunks:
                await original_message.reply(chunks[0], mention_author=False)
                for chunk in chunks[1:]:
                    await target.send(chunk)  # type: ignore[union-attr]
            else:
                await self._presenter.send_chunked(target, output)  # type: ignore[arg-type]
            if attachments:
                await self._presenter.send_attachments(target, attachments)  # type: ignore[arg-type]
            return True
        except discord.HTTPException as exc:
            log.error("discord_send_failed", error=str(exc))
            return False

    async def post_bus_status(self, message: str) -> None:
        if not settings.discord_bus_channel_id:
            return
        bus = self._client.get_channel(settings.discord_bus_channel_id)
        if bus is None:
            return
        try:
            await bus.send(message[:MAX_REPLY_LEN])  # type: ignore[union-attr]
        except discord.HTTPException as exc:
            log.warning("bus_status_failed", error=str(exc))
