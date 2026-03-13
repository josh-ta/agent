"""Configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    agent_model: str = Field(default="claude-haiku-4-5", alias="AGENT_MODEL")
    # Model tiers for dynamic routing (override with env vars)
    model_fast: str = Field(default="claude-haiku-4-5", alias="MODEL_FAST")
    model_smart: str = Field(default="claude-sonnet-4-5", alias="MODEL_SMART")
    model_best: str = Field(default="claude-opus-4-5", alias="MODEL_BEST")

    # ── Identity ──────────────────────────────────────────────────────────────
    agent_name: str = Field(default="agent-1", alias="AGENT_NAME")

    # ── Discord ───────────────────────────────────────────────────────────────
    discord_bot_token: str = Field(default="", alias="DISCORD_BOT_TOKEN")
    discord_agent_channel_id: int = Field(default=0, alias="DISCORD_AGENT_CHANNEL_ID")
    discord_bus_channel_id: int = Field(default=0, alias="DISCORD_BUS_CHANNEL_ID")
    discord_comms_channel_id: int = Field(default=0, alias="DISCORD_COMMS_CHANNEL_ID")
    discord_guild_id: int = Field(default=0, alias="DISCORD_GUILD_ID")

    # ── Databases ─────────────────────────────────────────────────────────────
    sqlite_path: Path = Field(default=Path("/data/agent.db"), alias="SQLITE_PATH")
    postgres_url: str = Field(default="", alias="POSTGRES_URL")

    # ── MCP ───────────────────────────────────────────────────────────────────
    browser_mcp_url: str = Field(
        default="http://browser:3080/sse", alias="BROWSER_MCP_URL"
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    workspace_path: Path = Field(default=Path("/workspace"), alias="WORKSPACE_PATH")
    skills_path: Path = Field(
        default=Path("/app/agent/skills"), alias="SKILLS_PATH"
    )
    identity_path: Path = Field(
        default=Path("/app/agent/identity"), alias="IDENTITY_PATH"
    )

    # ── Self-update ───────────────────────────────────────────────────────────
    docker_restart_self: bool = Field(default=True, alias="DOCKER_RESTART_SELF")
    agent_container_name: str = Field(
        default="agent", alias="AGENT_CONTAINER_NAME"
    )

    # ── Behaviour ─────────────────────────────────────────────────────────────
    max_loop_iterations: int = Field(default=50, alias="MAX_LOOP_ITERATIONS")
    heartbeat_seconds: int = Field(default=60, alias="HEARTBEAT_SECONDS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def model_string(self) -> str:
        """Return the pydantic-ai model string for the default model."""
        return self._to_model_string(self.agent_model)

    def _to_model_string(self, model: str) -> str:
        if model.startswith("claude"):
            return f"anthropic:{model}"
        if model.startswith("gpt") or model.startswith("o"):
            return f"openai:{model}"
        return model

    def model_string_for(self, tier: str) -> str:
        """Return pydantic-ai model string for a named tier: fast | smart | best."""
        tier_map = {"fast": self.model_fast, "smart": self.model_smart, "best": self.model_best}
        return self._to_model_string(tier_map.get(tier, self.agent_model))

    @property
    def has_discord(self) -> bool:
        return bool(self.discord_bot_token)

    @property
    def has_postgres(self) -> bool:
        return bool(self.postgres_url)


# Singleton
settings = Settings()
