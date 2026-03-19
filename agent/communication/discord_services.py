"""
Discord-facing services used by `discord_bot`.

The gateway stays responsible for discord.py lifecycle, while these classes hold
message policy and event rendering.
"""

from __future__ import annotations

import asyncio
import json
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
                summary = event.content.strip().replace("\n", " ")
                if len(summary) > 160:
                    summary = summary[:157] + "..."
                await _send(f"🟢 Starting: {summary}")
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
        self._session_router = SessionRouter()
        self._bridge.register(self._waiting_sink_tag, self._make_waiting_prompt_sink())

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

        task_content = (
            a2a_to_task_content(parsed.a2a_payload)
            if parsed.kind == MessageKind.A2A and parsed.a2a_payload
            else parsed.content
        )
        task_content, attachment_metadata, combined_content = await self._build_task_input(
            message,
            base_content=task_content,
        )
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
                "session_id": self._active_sessions.get(parsed.channel_id, ""),
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
                if allows_inline_reply(parsed.channel_id):
                    try:
                        await message.reply(
                            "⏸️ I can't safely interrupt the running tool chain yet, but I'll stop after the current step and treat this as a pause request.",
                            mention_author=False,
                        )
                    except discord.HTTPException:
                        pass
                await inject_q.put("User asked to pause/cancel after the current step.")
                return
            await inject_q.put(combined_content)
            await self._acknowledge_message(message)
            if allows_inline_reply(parsed.channel_id):
                try:
                    await message.reply(
                        (
                            "💬 Got it — I'll update the current task with that new constraint."
                            if decision.intent == TurnIntent.CLARIFICATION_OR_NEW_CONSTRAINT
                            else "💬 Got it — I'll fold that into what I'm working on."
                        ),
                        mention_author=False,
                    )
                except discord.HTTPException:
                    pass
            return

        if self._agent_loop.has_pending_work:
            metadata = self._session_router.build_metadata(
                source="discord",
                channel_id=parsed.channel_id,
                message_id=parsed.message_id,
                reference_message_id=getattr(getattr(message, "reference", None), "message_id", None),
                metadata=attachment_metadata,
            )
            if "task_id" not in metadata:
                metadata["task_id"] = f"discord-{parsed.channel_id}-{parsed.message_id}"
            if self._agent_loop.memory is not None and hasattr(self._agent_loop.memory, "create_task_record"):
                await self._agent_loop.memory.create_task_record(
                    task_id=metadata["task_id"],
                    source="discord",
                    author=parsed.author,
                    content=task_content,
                    metadata=metadata,
                )
            await self._agent_loop.enqueue(
                Task(
                    content=task_content,
                    source="discord",
                    author=parsed.author,
                    channel_id=parsed.channel_id,
                    message_id=parsed.message_id,
                    metadata=metadata,
                    inject_queue=asyncio.Queue(),
                )
            )
            await self._acknowledge_message(message)
            if allows_inline_reply(parsed.channel_id):
                position = f"#{self._agent_loop.queue.qsize()}" if self._agent_loop.queue.qsize() > 1 else "next"
                try:
                    await message.reply(
                        f"⏸️ I'm still working on the previous task — queued yours ({position} up).",
                        mention_author=False,
                    )
                except discord.HTTPException:
                    pass
            return

        inject_q = asyncio.Queue()
        self._inject_queues[parsed.channel_id] = inject_q
        response_future = asyncio.get_running_loop().create_future()
        metadata = self._session_router.build_metadata(
            source="discord",
            channel_id=parsed.channel_id,
            message_id=parsed.message_id,
            reference_message_id=getattr(getattr(message, "reference", None), "message_id", None),
                metadata=attachment_metadata,
        )
        metadata["task_id"] = metadata.get("task_id") or f"discord-{parsed.channel_id}-{parsed.message_id}"
        self._active_sessions[parsed.channel_id] = metadata["session_id"]
        if self._agent_loop.memory is not None and hasattr(self._agent_loop.memory, "create_task_record"):
            await self._agent_loop.memory.create_task_record(
                task_id=metadata["task_id"],
                source="discord",
                author=parsed.author,
                content=task_content,
                metadata=metadata,
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
