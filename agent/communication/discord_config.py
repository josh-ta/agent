"""Discord `/config` command and interactive wizard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import discord

from agent.communication.discord_commands import NativeCommand
from agent.communication.message_router import ParsedMessage
from agent.runtime_config import (
    CONFIG_FIELDS,
    ConfigFieldSpec,
    format_config_list,
    format_wizard_prompt,
    resolve_config_key,
    set_config_value,
)

if TYPE_CHECKING:
    from agent.communication.discord_services import MessageHandlingService


WizardStep = Literal["pick_key", "enter_value"]


@dataclass
class ConfigWizardState:
    step: WizardStep
    selected_env_key: str = ""


class ConfigCommandHandler:
    def __init__(self, service: MessageHandlingService) -> None:
        self._service = service
        self._wizard: dict[int, ConfigWizardState] = {}

    def cancel_wizard(self, channel_id: int) -> None:
        self._wizard.pop(channel_id, None)

    def has_wizard(self, channel_id: int) -> bool:
        return channel_id in self._wizard

    async def handle_command(
        self,
        *,
        message: discord.Message,
        parsed: ParsedMessage,
        command: NativeCommand,
    ) -> bool:
        argument = command.argument.strip()
        if argument.lower() in {"cancel", "abort", "exit"}:
            self.cancel_wizard(parsed.channel_id)
            await self._service._reply_safe(message, "⚙️ Config wizard cancelled.")  # noqa: SLF001
            return True

        if argument.lower() in {"list", "show", "help"} or not argument:
            if not argument:
                self._wizard[parsed.channel_id] = ConfigWizardState(step="pick_key")
                await self._service._reply_safe(message, format_wizard_prompt())  # noqa: SLF001
                return True
            await self._service._reply_safe(message, format_config_list())  # noqa: SLF001
            return True

        if ":" not in argument:
            await self._service._reply_safe(  # noqa: SLF001
                message,
                "Usage: `/config KEY:VALUE` or `/config` for the wizard.\n"
                "Example: `/config AGENT_MODEL:claude-sonnet-4-5`",
            )
            return True

        key, value = argument.split(":", 1)
        ok, text, reload_agents = set_config_value(key, value)
        if ok and reload_agents:
            self._service._agent_loop.reload_agents()
        await self._service._reply_safe(message, text)  # noqa: SLF001
        return True

    async def maybe_handle_wizard(self, message: discord.Message, parsed: ParsedMessage) -> bool:
        state = self._wizard.get(parsed.channel_id)
        if state is None:
            return False

        text = (getattr(message, "content", None) or parsed.content or "").strip()
        if text.lower() in {"/config cancel", "/config abort", "/config exit"}:
            self.cancel_wizard(parsed.channel_id)
            await self._service._reply_safe(message, "⚙️ Config wizard cancelled.")  # noqa: SLF001
            return True

        if state.step == "pick_key":
            spec = _resolve_wizard_selection(text)
            if spec is None:
                await self._service._reply_safe(  # noqa: SLF001
                    message,
                    "Reply with a number (1–"
                    f"{len(CONFIG_FIELDS)}) or a setting name like `AGENT_MODEL`. "
                    "Send `/config cancel` to stop.",
                )
                return True
            state.step = "enter_value"
            state.selected_env_key = spec.env_key
            from agent.runtime_config import format_field_value

            current = format_field_value(spec)
            await self._service._reply_safe(  # noqa: SLF001
                message,
                f"Current `{spec.env_key}` = `{current}`.\n"
                f"Send the new value for **{spec.description}**.",
            )
            return True

        spec = resolve_config_key(state.selected_env_key)
        if spec is None:
            self.cancel_wizard(parsed.channel_id)
            await self._service._reply_safe(message, "⚙️ Wizard state lost — run `/config` again.")  # noqa: SLF001
            return True

        ok, reply, reload_agents = set_config_value(spec.env_key, text)
        self.cancel_wizard(parsed.channel_id)
        if ok and reload_agents:
            self._service._agent_loop.reload_agents()
        await self._service._reply_safe(message, reply)  # noqa: SLF001
        return True


def _resolve_wizard_selection(text: str) -> ConfigFieldSpec | None:
    stripped = text.strip()
    if stripped.isdigit():
        index = int(stripped)
        specs = list(CONFIG_FIELDS.values())
        if 1 <= index <= len(specs):
            return specs[index - 1]
        return None
    return resolve_config_key(stripped)
