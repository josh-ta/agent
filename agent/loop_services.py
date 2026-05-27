"""
Supporting services for `agent.loop`.

These classes keep side effects and branching logic out of the queue coordinator
so they can be tested with fakes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import structlog
from pydantic_ai import Agent


@contextlib.asynccontextmanager
async def _agent_mcp_context(agent: object):
    run_mcp_servers = getattr(agent, "run_mcp_servers", None)
    if run_mcp_servers is None:
        yield
        return
    async with run_mcp_servers():
        yield
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    TextPartDelta,
    ThinkingPartDelta,
)
from pydantic_ai.usage import UsageLimits

from agent.attachment_ingest import inline_prompt_parts_from_metadata, render_attachment_context
from agent.config import settings
from agent.metrics import Metrics
from agent.events import (
    ProgressEvent,
    ShellDoneEvent,
    ShellOutputEvent,
    ShellStartEvent,
    TextDeltaEvent,
    TextTurnEndEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolResultEvent,
    bridge,
)
from agent.memory.learning_service import LearningService
from agent.project_memory import load_project_memory
from agent.secret_store import SecretStore
from agent.tools.discord_tools import DiscordAttachment, decode_data_url_attachment, discord_send
from agent.task_waits import UserInputRequired

if TYPE_CHECKING:
    from agent.loop import Task, TaskResult

log = structlog.get_logger()
_SHELL_EXIT_CODE_RE = re.compile(r"\[exit code:\s*(-?\d+)\]", re.IGNORECASE)
_SHELL_CRITICAL_FAILURE_RE = re.compile(
    r"(?i)(\[timeout|\[error:|host key verification failed|permission denied|command not found|"
    r"module(?:not)?founderror|pull access denied|no module named|traceback )"
)
_MAX_STREAMING_TOOL_USE_RETRIES = 2
_WORKSPACE_EXPORT_PATH_RE = re.compile(
    r"(?:Written \d+ bytes to|Exported CSV to)\s+(\S+\.(?:csv|tsv|json|txt|md))\b",
    re.IGNORECASE,
)
_ATTACHABLE_EXTENSIONS = {".csv", ".tsv", ".json", ".txt", ".md"}


@dataclass
class RunResult:
    output: str = ""
    tool_calls: int = 0
    user_visible_reply_sent: bool = False
    waiting_for_user: bool = False
    question: str | None = None
    timeout_s: int = 300
    attachments: list[DiscordAttachment] = field(default_factory=list)
    shell_failures: list[str] = field(default_factory=list)
    input_chars: int = 0
    output_chars: int = 0


class TaskJournal:
    def __init__(self, root: Path, *, now_fn: Callable[[], float] = time.time) -> None:
        self._path = root / ".task_journal.md"
        self._now = now_fn
        self._secret_store = SecretStore(settings.agent_secrets_path)

    @property
    def path(self) -> Path:
        return self._path

    def expire_stale(self, *, max_age_s: int = 1800) -> None:
        try:
            if self._path.exists():
                age_s = self._now() - self._path.stat().st_mtime
                if age_s > max_age_s:
                    self._path.unlink()
                    log.info("task_journal_expired", age_s=round(age_s))
        except Exception:
            pass

    def append(self, title: str, body: str) -> None:
        try:
            ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            safe_body = self._secret_store.redact_text(body)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n### [{ts}] — {title}\n{safe_body}\n")
        except Exception:
            pass

    def clear(self) -> None:
        try:
            if self._path.exists():
                self._path.unlink()
        except Exception:
            pass


class TaskContextBuilder:
    def __init__(
        self,
        memory_store: Any,
        history_reader: Callable[[int, int], Awaitable[str]] | None = None,
        secret_store: SecretStore | None = None,
    ) -> None:
        self._memory = memory_store
        self._history_reader = history_reader
        self._secret_store = secret_store

    async def build(self, task: "Task") -> tuple["Task", str, str]:
        from agent.loop import Task as LoopTask
        from agent.loop import _classify_tier, _parse_override

        content, forced_tier = _parse_override(task.content)
        tier = forced_tier or _classify_tier(content)

        normalized_task = LoopTask(
            content=content,
            source=task.source,
            author=task.author,
            channel_id=task.channel_id,
            message_id=task.message_id,
            metadata=task.metadata,
            created_at=task.created_at,
            inject_queue=task.inject_queue,
            response_future=task.response_future,
        )

        learning_context, retrieved_memory_ids, retrieved_procedure_ids = await self._load_learning_context(
            task.content
        )
        lessons_context = "" if learning_context else await self._load_lessons(task.content)
        facts_context = "" if learning_context else await self._load_memory_facts(task.content)
        session_context = await self._load_session_context(normalized_task)
        project_context = self._load_project_context()
        channel_context = await self._load_channel_history(normalized_task) if not session_context else ""
        resume_context = self._load_resume_context(normalized_task)
        checkpoint_context = await self._load_checkpoint_context(normalized_task)
        transcript_context = await self._load_transcript_context(normalized_task)
        attachment_context = self._load_attachment_context(normalized_task)
        secret_context = self._load_secret_context(normalized_task.content)

        parts: list[str] = []
        if session_context:
            parts.append(session_context)
        elif channel_context:
            parts.append(channel_context)
        if project_context:
            parts.append(project_context)
        if learning_context:
            parts.append(learning_context)
        if facts_context:
            parts.append(facts_context)
        if lessons_context:
            parts.append(lessons_context)
        if checkpoint_context:
            parts.append(checkpoint_context)
        if transcript_context:
            parts.append(transcript_context)
        if resume_context:
            parts.append(resume_context)
        if attachment_context:
            parts.append(attachment_context)
        if secret_context:
            parts.append(secret_context)
        database_hint = self._load_database_workflow_hint(normalized_task.content)
        if database_hint:
            parts.append(database_hint)
        if retrieved_memory_ids:
            normalized_task.metadata["retrieved_memory_ids"] = retrieved_memory_ids
        if retrieved_procedure_ids:
            normalized_task.metadata["retrieved_procedure_ids"] = retrieved_procedure_ids
        parts.append(normalized_task.content)
        return normalized_task, tier, "\n\n---\n\n".join(parts)

    async def build_chat(self, task: "Task") -> tuple["Task", str, str]:
        """Conversational turn: recent session + user message (no tools/memory search)."""
        from agent.loop import Task as LoopTask
        from agent.loop import _parse_override

        content, forced_tier = _parse_override(task.content)
        tier = forced_tier or "fast"
        normalized_task = LoopTask(
            content=content,
            source=task.source,
            author=task.author,
            channel_id=task.channel_id,
            message_id=task.message_id,
            metadata=task.metadata,
            created_at=task.created_at,
            inject_queue=task.inject_queue,
            response_future=task.response_future,
        )
        session_context = await self._load_session_context(normalized_task)
        parts: list[str] = []
        if session_context:
            parts.append(session_context)
        parts.append(normalized_task.content)
        return normalized_task, tier, "\n\n---\n\n".join(parts)

    @staticmethod
    def _load_project_context() -> str:
        return load_project_memory()

    @staticmethod
    def _load_database_workflow_hint(content: str) -> str:
        from agent.task_router import requires_database_tools

        if not requires_database_tools(content):
            return ""
        skill_path = settings.skills_path / "query-database.md"
        if not skill_path.exists():
            return ""
        body = skill_path.read_text(encoding="utf-8").strip()[:2500]
        return (
            "## Database / CSV workflow (use tools — do not answer from memory)\n"
            f"{body}\n"
            "For CSV exports use query_postgres(..., output_format='csv', output_path='/workspace/export.csv') "
            "so the file is written directly without piping rows through write_file."
        )

    async def _load_lessons(self, content: str) -> str:
        if self._memory and hasattr(self._memory, "search_lessons"):
            try:
                return await self._memory.search_lessons(content[:200], limit=3)
            except Exception:
                return ""
        return ""

    async def _load_learning_context(self, content: str) -> tuple[str, list[int], list[int]]:
        if self._memory and hasattr(self._memory, "search_learning_context"):
            try:
                payload = await self._memory.search_learning_context(content[:200], limit=3)
                return (
                    str(payload.get("text", "")).strip(),
                    [int(item) for item in payload.get("memory_ids", [])],
                    [int(item) for item in payload.get("procedure_ids", [])],
                )
            except Exception:
                return "", [], []
        return "", [], []

    async def _load_memory_facts(self, content: str) -> str:
        if self._memory and hasattr(self._memory, "search_memory"):
            try:
                facts = await self._memory.search_memory(content[:200], limit=3)
                if facts and not facts.startswith("(no memory matches"):
                    return "## Relevant stored facts\n" + facts
            except Exception:
                return ""
        return ""

    def _load_secret_context(self, content: str) -> str:
        if self._secret_store is None:
            return ""
        try:
            matches = self._secret_store.search(content[:200], limit=3)
        except Exception:
            return ""
        if not matches:
            return ""
        return "## Sensitive resources available by explicit request\n" + "\n".join(
            f"- {entry['name']}" + (f" ({entry['purpose']})" if entry["purpose"] else "")
            for entry in matches
        )

    async def _load_session_context(self, task: "Task") -> str:
        session_id = str((task.metadata or {}).get("session_id", "")).strip()
        if not session_id or not self._memory or not hasattr(self._memory, "get_session_context"):
            return ""
        try:
            return await self._memory.get_session_context(session_id, limit=10, char_cap=2200)
        except Exception:
            return ""

    async def _load_transcript_context(self, task: "Task") -> str:
        limit = max(0, int(getattr(settings, "restore_transcript_turns", 0) or 0))
        if limit <= 0:
            return ""
        task_id = str((task.metadata or {}).get("task_id", "")).strip()
        if not task_id or not self._memory or not hasattr(self._memory, "list_transcript_entries"):
            return ""
        try:
            rows = await self._memory.list_transcript_entries(task_id, limit=limit)
        except Exception:
            return ""
        if not rows:
            return ""
        lines = ["## Transcript checkpoint (persisted)"]
        for row in rows:
            role = str(row.get("role", ""))
            kind = str(row.get("kind", ""))
            content = str(row.get("content", ""))[:2000]
            lines.append(f"- **{role}** ({kind}): {content}")
        return "\n".join(lines)

    async def _load_checkpoint_context(self, task: "Task") -> str:
        task_id = str((task.metadata or {}).get("task_id", "")).strip()
        if not task_id or not self._memory or not hasattr(self._memory, "get_task_checkpoint"):
            return ""
        try:
            checkpoint = await self._memory.get_task_checkpoint(task_id)
        except Exception:
            return ""
        if not checkpoint:
            return ""
        parts: list[str] = []
        summary = str(checkpoint.get("summary", "")).strip()
        notes = str(checkpoint.get("notes", "")).strip()
        draft = str(checkpoint.get("draft", "")).strip()
        if summary:
            parts.append("## Previous checkpoint summary\n" + summary[:1200])
        if notes:
            parts.append("## Previous task notes\n" + notes[-1800:])
        if draft:
            parts.append("## Partial draft\n" + draft[:800])
        return "\n\n".join(parts)

    async def _load_channel_history(self, task: "Task") -> str:
        if task.source != "discord" or not task.channel_id:
            return ""

        try:
            if self._memory and hasattr(self._memory, "get_history"):
                history_rows = await self._memory.get_history(task.channel_id, limit=10)
                history_rows = history_rows[:-1] if history_rows else []
                lines: list[str] = []
                total_chars = 0
                for row in history_rows:
                    role_label = "You" if row["role"] == "assistant" else row["role"].capitalize()
                    text = row["content"][:300] + ("…" if len(row["content"]) > 300 else "")
                    line = f"{role_label}: {text}"
                    if total_chars + len(line) > 1500:
                        break
                    lines.append(line)
                    total_chars += len(line)
                return "## Recent conversation history\n" + "\n".join(lines) if lines else ""

            if self._history_reader is None:
                from agent.tools.discord_tools import discord_read

                self._history_reader = discord_read

            raw = await self._history_reader(task.channel_id, 7)
            lines = raw.splitlines()
            if lines:
                lines = lines[:-1]
            truncated: list[str] = []
            total_chars = 0
            for line in lines:
                short = line[:300] + ("…" if len(line) > 300 else "")
                if total_chars + len(short) > 1500:
                    break
                truncated.append(short)
                total_chars += len(short)
            return "## Recent conversation history\n" + "\n".join(truncated) if truncated else ""
        except Exception:
            return ""

    @staticmethod
    def _load_resume_context(task: "Task") -> str:
        metadata = task.metadata or {}
        resume_context = metadata.get("resume_context")
        if not isinstance(resume_context, dict):
            return ""
        question = str(resume_context.get("question", "")).strip()
        answer = str(resume_context.get("answer", "")).strip()
        if not question and not answer:
            return ""
        parts = ["## Resume context", "This task was paused for clarification and is now resuming."]
        if question:
            parts.append(f"Question asked: {question}")
        if answer:
            parts.append(f"User answer: {answer}")
        return "\n".join(parts)

    @staticmethod
    def _load_attachment_context(task: "Task") -> str:
        attachments = (task.metadata or {}).get("attachments")
        if not isinstance(attachments, list):
            return ""
        return render_attachment_context(attachments)


class RunExecutor:
    def __init__(
        self,
        *,
        event_bridge: Any = bridge,
        journal: TaskJournal | None = None,
        summarize_context: Callable[["Task", str], Awaitable[str]] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        progress_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        model_event_idle_timeout_s: float | None = None,
    ) -> None:
        self._bridge = event_bridge
        self._journal = journal or TaskJournal(settings.workspace_path)
        self._summarize_context = summarize_context
        self._sleep = sleep
        self._progress_sleep = progress_sleep
        configured_timeout = (
            settings.model_event_idle_timeout_seconds
            if model_event_idle_timeout_s is None
            else model_event_idle_timeout_s
        )
        self._model_event_idle_timeout_s = max(1.0, float(configured_timeout))

    @staticmethod
    def _load_query_database_skill() -> str:
        skill_path = settings.skills_path / "query-database.md"
        if not skill_path.exists():
            return ""
        return skill_path.read_text(encoding="utf-8").strip()[:2500]

    def _build_tool_retry_prompt(self, *, base_prompt: str, task: "Task", retry_index: int) -> str:
        from agent.task_router import requires_database_tools

        skill_block = ""
        if requires_database_tools(task.content):
            skill = self._load_query_database_skill()
            if skill:
                skill_block = f"\n## Database workflow (follow exactly)\n{skill}\n"

        if retry_index == 0:
            mandatory = (
                "## MANDATORY — tool use required\n"
                "Your previous turn finished without calling any tools. That is not acceptable.\n"
                "You MUST call at least one tool before giving a final answer.\n"
                "- Database/CSV: list_postgres_tables() → query_postgres(output_format='csv', output_path='/workspace/export.csv') → attach file\n"
                "- Other work: run_shell, read_file, write_file, or the relevant tool\n"
                "Do not reply with prose only. Invoke a tool in your next step."
            )
        else:
            mandatory = (
                "## MANDATORY — still no tools called\n"
                "You have repeatedly finished without calling any tools. Prose-only answers are rejected.\n"
                "Your very next action MUST be a tool call — not text.\n"
                "- Database/CSV: call list_postgres_tables() first, then query_postgres(output_format='csv', "
                "output_path='/workspace/export.csv') — do not paste CSV into write_file\n"
                "- Other work: run_shell, read_file, write_file, or the relevant tool\n"
                "Do not explain what you would do — call the tool."
            )

        return f"{base_prompt}\n\n---\n\n{mandatory}{skill_block}"

    async def _run_non_streaming_tool_recovery(
        self,
        *,
        task: "Task",
        agent: Agent,  # type: ignore[type-arg]
        base_prompt: str,
    ) -> RunResult:
        from agent.task_router import requires_database_tools

        skill_block = ""
        if requires_database_tools(task.content):
            skill = self._load_query_database_skill()
            if skill:
                skill_block = f"\n## Database workflow\n{skill}\n"

        prompt = (
            f"{base_prompt}\n\n---\n\n"
            "## FINAL ATTEMPT — complete this with tools\n"
            "Streaming attempts returned text without calling any tools. That failed.\n"
            "Use tools to finish the user's request. Do not answer from memory.\n"
            f"{skill_block}"
            "Your first action MUST be a tool call. Keep calling tools until the work is done."
        )
        await self._bridge.emit(
            ProgressEvent(message="🔧 Running a forced tool pass to complete this request…")
        )
        composed = self._compose_user_prompt(prompt, task)
        async with _agent_mcp_context(agent):
            result = await agent.run(
                composed,
                usage_limits=UsageLimits(request_limit=50, tool_calls_limit=40),
            )
        tool_calls = int(getattr(result.usage(), "tool_calls", 0) or 0)
        output = str(result.output).strip()
        ic, oc = len(composed), len(output)
        await self._maybe_warn_context(ic + oc)
        return RunResult(
            output=output,
            tool_calls=tool_calls,
            input_chars=ic,
            output_chars=oc,
        )

    async def _maybe_warn_context(self, total_chars: int) -> None:
        est = max(0, total_chars) // 4
        threshold = max(1000, int(settings.context_token_warn_threshold))
        if est < threshold:
            return
        Metrics.inc_context_warn()
        await self._bridge.emit(
            ProgressEvent(
                message=(
                    f"⚠️ Estimated context usage ~{est} tokens (threshold {threshold}). "
                    "Consider narrowing scope or summarizing."
                )
            )
        )

    async def run(
        self,
        *,
        task: "Task",
        agent: Agent,  # type: ignore[type-arg]
        base_prompt: str,
        tier: str,
        message_history: list | None = None,
    ) -> RunResult:
        prompt = base_prompt
        context_retries = 0
        max_context_retries = 2
        rate_retries = 0
        max_rate_retries = 3
        rate_delay = 5.0
        bad_args_retries = 0
        max_bad_args_retries = 2

        while True:
            user_visible_reply_sent = False
            tool_calls = 0
            result_output = ""
            attachments: list[DiscordAttachment] = []
            shell_failures: list[str] = []
            pending_visible_discord_sends: set[str] = set()
            injection_restart_requested = False
            injected_messages: list[str] = []
            loop = asyncio.get_running_loop()
            progress_state = {
                "last_activity_at": loop.time(),
                "activity": "thinking about your request",
            }
            progress_watchdog: asyncio.Task[None] | None = None

            try:
                progress_watchdog = loop.create_task(self._progress_watchdog(progress_state))
                activity_sink_tag = f"run_executor_activity_{id(progress_state)}"
                _rg = (task.metadata or {}).get("run_generation")
                _expected_rg = int(_rg) if _rg is not None else None
                if hasattr(self._bridge, "register"):
                    self._bridge.register(
                        activity_sink_tag,
                        self._make_activity_sink(
                            progress_state=progress_state,
                            task_id=str((task.metadata or {}).get("task_id", "")).strip() or None,
                            expected_run_generation=_expected_rg,
                        ),
                    )
                async with _agent_mcp_context(agent):
                    thinking_buf: list[str] = []
                    text_buf: list[str] = []
                    is_final = False
                    event_stream = agent.run_stream_events(
                        self._compose_user_prompt(prompt, task),
                        message_history=message_history or [],
                        usage_limits=UsageLimits(request_limit=None),
                    )
                    event_iter = event_stream.__aiter__()

                    while True:
                        try:
                            event = await self._await_next_stream_event(event_iter, progress_state)
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError as exc:
                            await self._close_event_stream(event_iter)
                            timeout_message = self._model_timeout_message(progress_state)
                            self._journal.append("MODEL TURN TIMEOUT", timeout_message)
                            await self._bridge.emit(ProgressEvent(message=f"⌛ {timeout_message}"))
                            raise RuntimeError(timeout_message) from exc

                        self._mark_progress_activity(progress_state)
                        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
                            delta = event.delta.content_delta
                            if delta:
                                thinking_buf.append(delta)
                                await self._bridge.emit(ThinkingDeltaEvent(delta=delta))
                        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                            delta = event.delta.content_delta
                            text_buf.append(delta)
                            await self._bridge.emit(TextDeltaEvent(delta=delta))
                        elif isinstance(event, FinalResultEvent):
                            is_final = True
                            result_output = str(event.output) if hasattr(event, "output") else ""
                        elif isinstance(event, PartEndEvent):
                            if thinking_buf:
                                full_thinking = "".join(thinking_buf).strip()
                                thinking_buf = []
                                if full_thinking:
                                    await self._bridge.emit(ThinkingEndEvent(text=full_thinking))
                            elif text_buf:
                                full_text = "".join(text_buf).strip()
                                text_buf = []
                                if full_text:
                                    await self._bridge.emit(TextTurnEndEvent(text=full_text, is_final=is_final))
                        elif isinstance(event, FunctionToolCallEvent):
                            tool_calls += 1
                            tool_name = event.part.tool_name
                            self._set_progress_activity(progress_state, f"running `{tool_name}`")
                            parsed_args = self._parse_tool_args(event.part.args)
                            if self._is_user_visible_discord_send(task, tool_name, parsed_args):
                                pending_visible_discord_sends.add(event.part.tool_call_id)
                            await self._bridge.emit(
                                ToolCallStartEvent(
                                    tool_name=tool_name,
                                    call_id=event.part.tool_call_id,
                                    args=self._sanitize_tool_args(tool_name, parsed_args),
                                )
                            )
                        elif isinstance(event, FunctionToolResultEvent):
                            ret = getattr(event, "result", None)
                            tool_name = self._tool_name_from_result_event(event, ret)
                            if tool_name:
                                self._set_progress_activity(progress_state, f"reviewing results from `{tool_name}`")
                            result_str = str(getattr(ret, "content", ret)) if ret is not None else ""
                            if (
                                tool_name == "send_discord"
                                and getattr(ret, "tool_call_id", "") in pending_visible_discord_sends
                                and self._is_successful_send_discord_result(result_str)
                            ):
                                user_visible_reply_sent = True
                            if tool_name == "run_shell":
                                failure = self._detect_shell_failure(result_str)
                                if failure:
                                    shell_failures.append(failure)
                            attachments.extend(self._extract_discord_attachments(tool_name, getattr(ret, "content", ret)))
                            await self._bridge.emit(
                                ToolResultEvent(
                                    tool_name=tool_name,
                                    call_id=getattr(ret, "tool_call_id", ""),
                                    result=self._sanitize_tool_result(tool_name, result_str),
                                )
                            )

                        if task.inject_queue and isinstance(event, (FunctionToolResultEvent, FinalResultEvent)):
                            injected_messages = self._drain_queue(task.inject_queue)
                            if injected_messages:
                                injection_restart_requested = True
                                await self._close_event_stream(event_iter)
                                break

                if injection_restart_requested or (task.inject_queue and not task.inject_queue.empty()):
                    injected = injected_messages or self._drain_queue(task.inject_queue)
                    if injected:
                        count = len(injected)
                        combined = "\n".join(f"[{i + 1}] {msg}" for i, msg in enumerate(injected))
                        await self._bridge.emit(
                            ProgressEvent(
                                message=(
                                    f"💬 Got {'your messages' if count > 1 else 'your message'} — "
                                    f"{'folding them' if count > 1 else 'folding it'} in now."
                                )
                            )
                        )
                        followup_result = await self.run(
                            task=task,
                            agent=agent,
                            base_prompt=(
                                f"{base_prompt}\n\n---\n\n"
                                f"## New message{'s' if count > 1 else ''} received while you were working:\n"
                                f"{combined}\n\n"
                                f"Please incorporate {'these' if count > 1 else 'this'} into your work and continue."
                            ),
                            tier=tier,
                            message_history=message_history,
                        )
                        result_output = followup_result.output
                        tool_calls += followup_result.tool_calls
                        user_visible_reply_sent = user_visible_reply_sent or followup_result.user_visible_reply_sent
                        attachments.extend(followup_result.attachments)
                        if followup_result.waiting_for_user:
                            up = self._compose_user_prompt(prompt, task)
                            ic, oc = len(up), len(result_output)
                            await self._maybe_warn_context(ic + oc)
                            return RunResult(
                                output=result_output,
                                tool_calls=tool_calls,
                                user_visible_reply_sent=user_visible_reply_sent,
                                waiting_for_user=True,
                                question=followup_result.question,
                                timeout_s=followup_result.timeout_s,
                                attachments=attachments,
                                shell_failures=shell_failures + followup_result.shell_failures,
                                input_chars=ic,
                                output_chars=oc,
                            )
                        shell_failures.extend(followup_result.shell_failures)

                tool_retries = int((task.metadata or {}).get("_tool_use_retries", 0))
                from agent.task_router import requires_tool_use

                if tool_calls == 0 and requires_tool_use(task.content) and tool_retries < _MAX_STREAMING_TOOL_USE_RETRIES:
                    task.metadata["_tool_use_retries"] = tool_retries + 1
                    await self._bridge.emit(
                        ProgressEvent(
                            message=(
                                "🔧 Retrying — this task needs tools "
                                "(database query, shell, files, etc.)."
                            )
                        )
                    )
                    return await self.run(
                        task=task,
                        agent=agent,
                        base_prompt=self._build_tool_retry_prompt(
                            base_prompt=base_prompt,
                            task=task,
                            retry_index=tool_retries,
                        ),
                        tier=tier,
                        message_history=message_history,
                    )

                if (
                    tool_calls == 0
                    and requires_tool_use(task.content)
                    and tool_retries >= _MAX_STREAMING_TOOL_USE_RETRIES
                    and hasattr(agent, "run")
                ):
                    recovery = await self._run_non_streaming_tool_recovery(
                        task=task,
                        agent=agent,
                        base_prompt=base_prompt,
                    )
                    if recovery.tool_calls > 0:
                        return recovery
                    tool_calls = recovery.tool_calls
                    result_output = recovery.output or result_output

                up = self._compose_user_prompt(prompt, task)
                ic, oc = len(up), len(result_output)
                await self._maybe_warn_context(ic + oc)
                return RunResult(
                    output=result_output,
                    tool_calls=tool_calls,
                    user_visible_reply_sent=user_visible_reply_sent,
                    attachments=attachments,
                    shell_failures=shell_failures,
                    input_chars=ic,
                    output_chars=oc,
                )

            except Exception as exc:
                if isinstance(exc, UserInputRequired):
                    up = self._compose_user_prompt(prompt, task)
                    ic = len(up)
                    await self._maybe_warn_context(ic)
                    return RunResult(
                        tool_calls=tool_calls,
                        user_visible_reply_sent=user_visible_reply_sent,
                        waiting_for_user=True,
                        question=exc.question,
                        timeout_s=exc.timeout_s,
                        attachments=attachments,
                        shell_failures=shell_failures,
                        input_chars=ic,
                        output_chars=0,
                    )
                exc_str = str(exc)
                is_context_overflow = "prompt is too long" in exc_str or ("400" in exc_str and "maximum" in exc_str)
                is_rate_limit = "429" in exc_str
                is_bad_args = "EOF while parsing" in exc_str or "args_as_dict" in exc_str

                if is_context_overflow and context_retries < max_context_retries and self._summarize_context:
                    context_retries += 1
                    await self._bridge.emit(
                        ProgressEvent(
                            message=(
                                f"📦 Context window full — compressing and continuing… "
                                f"(attempt {context_retries}/{max_context_retries})"
                            )
                        )
                    )
                    summary = await self._summarize_context(task, prompt)
                    self._journal.append(
                        f"CONTEXT COMPRESSED (retry {context_retries})",
                        summary,
                    )
                    prompt = (
                        f"{base_prompt}\n\n"
                        f"## Progress so far (summarized — context was compressed):\n{summary}"
                    )
                    continue

                if is_rate_limit and rate_retries < max_rate_retries:
                    rate_retries += 1
                    await self._bridge.emit(
                        ProgressEvent(message=f"⏸️ Rate limited — retrying in {int(rate_delay)}s…")
                    )
                    await self._sleep(rate_delay)
                    rate_delay *= 2
                    continue

                if is_bad_args and bad_args_retries < max_bad_args_retries:
                    bad_args_retries += 1
                    await self._bridge.emit(
                        ProgressEvent(
                            message=(
                                "⚠️ Tool args got truncated — retrying from last checkpoint… "
                                f"({bad_args_retries}/{max_bad_args_retries})"
                            )
                        )
                    )
                    message_history = None
                    continue

                raise
            finally:
                if 'activity_sink_tag' in locals() and hasattr(self._bridge, "unregister"):
                    self._bridge.unregister(activity_sink_tag)
                if progress_watchdog is not None:
                    progress_watchdog.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await progress_watchdog

    async def _await_next_stream_event(
        self,
        event_iter: object,
        progress_state: dict[str, object],
    ) -> object:
        """Wait for the next model event, but only fail after real inactivity."""
        poll_s = 1.0
        loop = asyncio.get_running_loop()
        pending: asyncio.Task[object] | None = None
        try:
            while True:
                now = loop.time()
                idle_s = now - float(progress_state["last_activity_at"])
                if idle_s >= self._model_event_idle_timeout_s:
                    raise asyncio.TimeoutError()
                wait_s = min(poll_s, self._model_event_idle_timeout_s - idle_s)
                if pending is None:
                    pending = asyncio.create_task(anext(event_iter))
                if pending.done():
                    return pending.result()
                try:
                    await asyncio.wait_for(asyncio.shield(pending), timeout=wait_s)
                except asyncio.TimeoutError:
                    continue
                return pending.result()
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                    await pending

    @staticmethod
    async def _close_event_stream(event_iter: object) -> None:
        aclose = getattr(event_iter, "aclose", None)
        if aclose is None:
            return
        with contextlib.suppress(Exception):
            await aclose()

    def _model_timeout_message(self, progress_state: dict[str, object]) -> str:
        activity = str(progress_state.get("activity") or "waiting for the model")
        return (
            "Task stalled after "
            f"{self._model_event_idle_timeout_s:.0f}s with no activity while {activity}."
        )

    def _make_activity_sink(
        self,
        *,
        progress_state: dict[str, object],
        task_id: str | None,
        expected_run_generation: int | None = None,
    ):
        async def sink(event: object) -> None:
            if expected_run_generation is not None:
                rg = getattr(event, "run_generation", None)
                if rg is not None and rg != expected_run_generation:
                    return
            event_task_id = getattr(event, "task_id", None)
            if task_id is not None and event_task_id not in {None, task_id}:
                return
            if isinstance(event, ShellStartEvent):
                self._set_progress_activity(progress_state, "waiting on shell command output")
            elif isinstance(event, ShellOutputEvent):
                self._set_progress_activity(progress_state, "streaming shell output")
            elif isinstance(event, ShellDoneEvent):
                self._set_progress_activity(progress_state, "reviewing shell results")
            elif isinstance(event, ToolCallStartEvent):
                self._set_progress_activity(progress_state, f"running `{event.tool_name}`")
            elif isinstance(event, ToolResultEvent):
                self._set_progress_activity(progress_state, f"reviewing results from `{event.tool_name}`")
            elif isinstance(event, ThinkingDeltaEvent):
                self._set_progress_activity(progress_state, "reasoning about your request")

        return sink

    @staticmethod
    def _parse_tool_args(raw_args: object) -> object:
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args)
            except Exception:
                return raw_args
        return raw_args

    @staticmethod
    def _compose_user_prompt(base_prompt: str, task: "Task") -> str | list[object]:
        attachment_parts = inline_prompt_parts_from_metadata((task.metadata or {}).get("attachments", []))
        if not attachment_parts:
            return base_prompt
        return [base_prompt, *attachment_parts]

    @staticmethod
    def _sanitize_tool_args(tool_name: str, args: object) -> object:
        if tool_name == "secret_set" and isinstance(args, dict):
            return {"name": args.get("name", ""), "value": "[REDACTED]"}
        if tool_name in {"secret_get", "secret_delete"} and isinstance(args, dict):
            return {"name": args.get("name", "")}
        return args

    @staticmethod
    def _sanitize_tool_result(tool_name: str, result: str) -> str:
        if tool_name == "secret_get":
            return "[REDACTED secret value]"
        if tool_name == "secret_set":
            return "Stored secret [REDACTED]"
        return result[:500]

    @staticmethod
    def _is_user_visible_discord_send(task: "Task", tool_name: str, args: object) -> bool:
        if tool_name != "send_discord" or not isinstance(args, dict):
            return False
        try:
            channel_id = int(args.get("channel_id", 0))
        except (TypeError, ValueError):
            return False
        if channel_id in {settings.discord_comms_channel_id, settings.discord_bus_channel_id}:
            return False
        visible_channels = {cid for cid in (task.channel_id, settings.discord_agent_channel_id) if cid}
        return channel_id in visible_channels

    @staticmethod
    def _is_successful_send_discord_result(result: str) -> bool:
        text = result.strip()
        return bool(text) and not text.startswith("[ERROR") and text.startswith("Sent ")

    @staticmethod
    def _tool_name_from_result_event(event: FunctionToolResultEvent, result: object) -> str:
        return str(getattr(event, "tool_name", "") or getattr(result, "tool_name", "") or "")

    @staticmethod
    def _extract_discord_attachments(tool_name: str, result: object) -> list[DiscordAttachment]:
        if tool_name == "browser_screenshot":
            attachments: list[DiscordAttachment] = []
            for text in RunExecutor._iter_text_values(result):
                attachment = decode_data_url_attachment(
                    text,
                    filename=f"browser-screenshot-{len(attachments) + 1}.png",
                )
                if attachment is not None:
                    attachments.append(attachment)
            return attachments

        if tool_name not in {"write_file", "query_postgres"}:
            return []

        attachments: list[DiscordAttachment] = []
        for text in RunExecutor._iter_text_values(result):
            attachments.extend(RunExecutor._attachments_from_export_result(text))
        return attachments

    @staticmethod
    def _attachments_from_export_result(result: str) -> list[DiscordAttachment]:
        from agent.tools.filesystem import read_workspace_attachment

        attachments: list[DiscordAttachment] = []
        seen: set[str] = set()
        for match in _WORKSPACE_EXPORT_PATH_RE.finditer(result):
            path = match.group(1)
            if Path(path).suffix.lower() not in _ATTACHABLE_EXTENSIONS:
                continue
            payload = read_workspace_attachment(path)
            if payload is None:
                continue
            filename, data = payload
            if filename in seen:
                continue
            seen.add(filename)
            attachments.append(DiscordAttachment(filename=filename, data=data))
        return attachments

    @staticmethod
    def _iter_text_values(value: object) -> list[str]:
        texts: list[str] = []
        if value is None:
            return texts
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            for item in value:
                texts.extend(RunExecutor._iter_text_values(item))
            return texts
        if isinstance(value, dict):
            for item in value.values():
                texts.extend(RunExecutor._iter_text_values(item))
            return texts

        text = getattr(value, "text", None)
        if isinstance(text, str):
            texts.append(text)
        else:
            content = getattr(value, "content", None)
            if content is not None and content is not value:
                texts.extend(RunExecutor._iter_text_values(content))
        return texts

    @staticmethod
    def _drain_queue(queue: asyncio.Queue[str]) -> list[str]:
        items: list[str] = []
        while not queue.empty():
            try:
                items.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    @staticmethod
    def _detect_shell_failure(result: str) -> str | None:
        text = result.strip()
        if not text:
            return None

        match = _SHELL_EXIT_CODE_RE.search(text)
        if match is not None:
            try:
                if int(match.group(1)) == 0:
                    return None
            except ValueError:
                pass
            return RunExecutor._summarize_shell_failure(text)

        if _SHELL_CRITICAL_FAILURE_RE.search(text):
            return RunExecutor._summarize_shell_failure(text)
        return None

    @staticmethod
    def _summarize_shell_failure(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return "Shell command failed."
        return " | ".join(lines[-3:])[:400]

    async def _progress_watchdog(self, progress_state: dict[str, object]) -> None:
        interval_s = max(1, int(settings.progress_heartbeat_seconds))
        while True:
            await self._progress_sleep(interval_s)
            now = asyncio.get_running_loop().time()
            last_activity_at = float(progress_state["last_activity_at"])
            if now - last_activity_at < interval_s:
                continue
            activity = str(progress_state.get("activity") or "working on the task")
            await self._bridge.emit(ProgressEvent(message=f"⏳ Still working — {activity}"))

    @staticmethod
    def _mark_progress_activity(progress_state: dict[str, object]) -> None:
        progress_state["last_activity_at"] = asyncio.get_running_loop().time()

    @staticmethod
    def _set_progress_activity(progress_state: dict[str, object], activity: str) -> None:
        progress_state["activity"] = activity
        RunExecutor._mark_progress_activity(progress_state)


class ReflectionService:
    def __init__(self, *, agents: dict[str, Agent], memory_store: Any, journal: TaskJournal | None = None) -> None:  # type: ignore[type-arg]
        self._agents = agents
        self._memory = memory_store
        self._journal = journal or TaskJournal(settings.workspace_path)
        self._learning = LearningService()
        self._secret_store = SecretStore(settings.agent_secrets_path)

    @property
    def _fast_agent(self) -> Agent:  # type: ignore[type-arg]
        return self._agents.get("fast") or self._agents.get("smart") or next(iter(self._agents.values()))

    async def _run_fast_agent(self, prompt: str) -> str:
        agent = self._fast_agent
        async with _agent_mcp_context(agent):
            result = await agent.run(prompt, usage_limits=UsageLimits(request_limit=None))
        return str(result.output).strip()

    async def reflect(self, task: "Task", result: "TaskResult", success_count: int, memory_update_interval: int) -> None:
        if not self._memory:
            return

        try:
            episode = self._learning.summarize_episode(task, result)
            task_id = str((task.metadata or {}).get("task_id", "")).strip()
            session_id = str((task.metadata or {}).get("session_id", "")).strip()
            if hasattr(self._memory, "record_episodic_event"):
                await self._memory.record_episodic_event(
                    task_id=task_id,
                    session_id=session_id,
                    event_kind=episode.event_kind,
                    summary=self._secret_store.redact_text(episode.summary),
                    reward=episode.reward.score,
                    details={
                        **{
                            key: self._secret_store.redact_text(value)
                            if isinstance(value, str)
                            else value
                            for key, value in episode.details.items()
                        },
                        "reward_reasons": episode.reward.reasons,
                    },
                )
            if hasattr(self._memory, "learning") and hasattr(self._memory.learning, "apply_outcome_to_memory"):
                await self._memory.learning.apply_outcome_to_memory(
                    [int(item) for item in result.retrieved_memory_ids],
                    success_delta=max(episode.reward.score, 0.0),
                    failure_delta=abs(min(episode.reward.score, 0.0)),
                )
            if hasattr(self._memory, "procedures") and hasattr(self._memory.procedures, "apply_outcome_to_procedures"):
                await self._memory.procedures.apply_outcome_to_procedures(
                    [int(item) for item in result.retrieved_procedure_ids],
                    success_delta=max(episode.reward.score, 0.0),
                    failure_delta=abs(min(episode.reward.score, 0.0)),
                )
            if not result.success:
                if not self._learning.should_promote_failure(result, episode):
                    return
                await self._reflect_on_failure(task, result, episode.reward.score, episode.reward.reasons)
            else:
                if not self._learning.should_promote_success(result, episode):
                    if success_count % memory_update_interval == 0:
                        await self.update_memory_md()
                    return
                await self._reflect_on_success(task, result, episode.reward.score, episode.reward.reasons)
                if success_count % memory_update_interval == 0:
                    await self.update_memory_md()
        except Exception:
            log.warning("reflect_error", exc=traceback.format_exc())

    async def _reflect_on_success(
        self,
        task: "Task",
        result: "TaskResult",
        reward_score: float = 0.0,
        reward_reasons: list[str] | None = None,
    ) -> None:
        reward_reasons = reward_reasons or []
        reflection_prompt = (
            f"You just completed the following task successfully using {result.tool_calls} tool calls.\n\n"
            f"Task: {task.content}\n\n"
            f"Result summary: {result.output[:500]}\n\n"
            f"Reward score: {reward_score:.2f}\n"
            f"Reward reasons: {', '.join(reward_reasons) or 'none'}\n\n"
            "Return 1 or 2 lines only. Each line must start with one of:\n"
            "PATTERN: <reusable success pattern>\n"
            "PROCEDURE: <trigger> => <checklist>\n"
            "If there is nothing genuinely reusable to record, reply with exactly: NOTHING_TO_RECORD"
        )
        try:
            insight = await self._run_fast_agent(reflection_prompt)
            if not insight or "NOTHING_TO_RECORD" in insight:
                return
            for raw_line in insight.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("PROCEDURE:"):
                    payload = line.removeprefix("PROCEDURE:").strip()
                    trigger, _, checklist = payload.partition("=>")
                    if hasattr(self._memory, "save_procedure"):
                        await self._memory.save_procedure(
                            trigger_text=self._secret_store.redact_text((trigger or task.content[:240]).strip()),
                            checklist=self._secret_store.redact_text((checklist or payload).strip())[:400],
                            kind="pattern",
                            confidence=0.72,
                            salience=0.7,
                            metadata={"context": self._secret_store.redact_text(task.content[:300]), "reward_score": reward_score},
                        )
                    continue
                summary = line.removeprefix("PATTERN:").strip() if line.startswith("PATTERN:") else line
                await self._memory.save_lesson(
                    summary=self._secret_store.redact_text(summary)[:300],
                    kind="pattern",
                    context=self._secret_store.redact_text(task.content[:300]),
                )
        except Exception:
            log.warning("reflect_on_success_error", exc=traceback.format_exc())

    async def _reflect_on_failure(
        self,
        task: "Task",
        result: "TaskResult",
        reward_score: float = 0.0,
        reward_reasons: list[str] | None = None,
    ) -> None:
        reward_reasons = reward_reasons or []
        reflection_prompt = (
            "You just attempted the following task and it FAILED.\n\n"
            f"Task: {task.content}\n\n"
            f"Error/output: {result.output}\n\n"
            f"Reward score: {reward_score:.2f}\n"
            f"Reward reasons: {', '.join(reward_reasons) or 'none'}\n\n"
            "Return 1 or 2 lines only. Each line must start with one of:\n"
            "MISTAKE: <what failed and what to do differently>\n"
            "PROCEDURE: <trigger> => <corrective checklist>\n"
            "Be concise and specific."
        )
        try:
            lesson = await self._run_fast_agent(reflection_prompt)
            for raw_line in lesson.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("PROCEDURE:"):
                    payload = line.removeprefix("PROCEDURE:").strip()
                    trigger, _, checklist = payload.partition("=>")
                    if hasattr(self._memory, "save_procedure"):
                        await self._memory.save_procedure(
                            trigger_text=self._secret_store.redact_text((trigger or task.content[:240]).strip()),
                            checklist=self._secret_store.redact_text((checklist or payload).strip())[:400],
                            kind="recovery",
                            confidence=0.8,
                            salience=0.8,
                            metadata={"context": self._secret_store.redact_text(task.content[:300]), "reward_score": reward_score},
                        )
                    continue
                summary = line.removeprefix("MISTAKE:").strip() if line.startswith("MISTAKE:") else line
                await self._memory.save_lesson(
                    summary=self._secret_store.redact_text(summary),
                    kind="mistake",
                    context=self._secret_store.redact_text(task.content[:300]),
                )
            await self.update_memory_md()
        except Exception:
            log.warning("reflect_on_failure_error", exc=traceback.format_exc())

    async def update_memory_md(self) -> None:
        try:
            recent = await self._memory.get_recent_lessons(limit=20)
            memory_path = settings.identity_path / "MEMORY.md"
            if not memory_path.exists():
                return

            content = memory_path.read_text(encoding="utf-8")
            marker = "## Recent Lessons"
            if marker in content:
                content = content[:content.index(marker)].rstrip()
            else:
                content = content.rstrip()

            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            content += f"\n\n{marker}\n_Last updated: {now}_\n\n{recent}\n"
            memory_path.write_text(content, encoding="utf-8")
        except Exception:
            log.warning("update_memory_md_error", exc=traceback.format_exc())


@dataclass
class HeartbeatTask:
    id: str
    description: str
    from_agent: str


class HeartbeatService:
    def __init__(
        self,
        *,
        memory_store: Any,
        postgres_store: Any,
        enqueue: Callable[["Task"], Awaitable[None]],
        wait_registry: Any = None,
        event_bridge: Any = bridge,
    ) -> None:
        self._memory = memory_store
        self._postgres = postgres_store
        self._enqueue = enqueue
        self._wait_registry = wait_registry
        self._bridge = event_bridge

    async def heartbeat(self, *, is_busy: bool) -> None:
        if self._memory and hasattr(self._memory, "heartbeat"):
            await self._memory.heartbeat()

        await self._expire_waiting_tasks()
        await self._dispatch_scheduled_tasks(is_busy=is_busy)

        if self._postgres is not None and not is_busy:
            try:
                rows = await self._postgres.get_pending_task_rows()
                for row in rows:
                    claimed = await self._postgres.mark_task_running(row["id"])
                    if claimed is False:
                        continue
                    await self._enqueue(self._build_a2a_task(row))
            except Exception as exc:
                log.warning("a2a_poll_error", error=str(exc))

    async def _dispatch_scheduled_tasks(self, *, is_busy: bool) -> None:
        """When idle, claim due SQLite scheduled rows and enqueue tasks."""
        if is_busy or self._memory is None or not hasattr(self._memory, "scheduled_tasks_claim_due"):
            return
        from agent.config import settings
        from agent.loop import Task

        now = time.time()
        limit = max(1, int(settings.scheduled_dispatch_per_heartbeat))
        try:
            rows = await self._memory.scheduled_tasks_claim_due(now=now, limit=limit)
        except Exception as exc:
            log.warning("scheduled_dispatch_error", error=str(exc))
            return
        for row in rows:
            meta = dict(row.get("metadata") or {})
            meta["scheduled_task_id"] = row["id"]
            task = Task(
                content=row["prompt"],
                source="scheduled",
                author="scheduler",
                metadata=meta,
            )
            await self._enqueue(task)
            log.info("scheduled_task_enqueued", scheduled_task_id=row["id"][:8])

    async def _expire_waiting_tasks(self) -> None:
        if self._memory is None or not hasattr(self._memory, "list_waiting_task_records"):
            return
        rows = await self._memory.list_waiting_task_records()
        now = time.time()
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            wait_state = dict(metadata.get("wait_state") or {})
            task_id = str(row.get("task_id", "")).strip()
            if not task_id or not wait_state:
                continue
            created_ts = float(wait_state.get("created_ts") or row.get("updated_ts") or now)
            timeout_s = max(1, int(wait_state.get("timeout_s") or 300))
            if now - created_ts < timeout_s:
                continue
            question = str(wait_state.get("question", "")).strip()
            timeout_message = (
                "Timed out waiting for user input."
                + (f" Question was: {question}" if question else "")
            )
            try:
                if hasattr(self._memory, "fail_task"):
                    await self._memory.fail_task(task_id, error=timeout_message, metadata=metadata)
                if hasattr(self._memory, "set_session_status"):
                    session_id = str(metadata.get("session_id", "")).strip()
                    if session_id:
                        await self._memory.set_session_status(session_id, status="timed_out", pending_task_id="")
                if hasattr(self._memory, "save_task_checkpoint"):
                    await self._memory.save_task_checkpoint(
                        task_id=task_id,
                        session_id=str(metadata.get("session_id", "")),
                        summary=timeout_message,
                        metadata=metadata,
                    )
            except Exception:
                log.warning("wait_timeout_persist_failed", task_id=task_id)
            if self._wait_registry is not None:
                self._wait_registry.pop(task_id)
            channel_id = int(wait_state.get("channel_id") or 0)
            if channel_id:
                try:
                    await discord_send(channel_id, f"⌛ {timeout_message}")
                except Exception:
                    log.warning("wait_timeout_notify_failed", task_id=task_id, channel_id=channel_id)
            await self._bridge.emit(ProgressEvent(message=f"Timed out waiting on task {task_id[:8]}."))

    @staticmethod
    def _build_a2a_task(row: dict[str, Any]) -> "Task":
        from agent.loop import Task

        return Task(
            content=row["description"],
            source="a2a",
            author=row.get("from_agent", "unknown"),
            metadata={"task_id": row["id"], "from_agent": row.get("from_agent", "unknown")},
        )
