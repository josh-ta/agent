"""Configuration loaded from environment variables."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    anthropic_api_key: SecretStr = Field(default=SecretStr(""), alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr = Field(default=SecretStr(""), alias="OPENAI_API_KEY")
    mistral_api_key: SecretStr = Field(default=SecretStr(""), alias="MISTRAL_API_KEY")
    xai_api_key: SecretStr = Field(default=SecretStr(""), alias="XAI_API_KEY")
    agent_model: str = Field(default="claude-haiku-4-5", alias="AGENT_MODEL")
    # Embedding model used for semantic memory search (requires OPENAI_API_KEY)
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")
    mistral_base_url: str = Field(default="https://api.mistral.ai/v1", alias="MISTRAL_BASE_URL")
    xai_base_url: str = Field(default="https://api.x.ai/v1", alias="XAI_BASE_URL")
    # ── Model tiers for dynamic routing (override with env vars)
    model_fast: str = Field(default="claude-haiku-4-5", alias="MODEL_FAST")
    model_smart: str = Field(default="claude-sonnet-4-5", alias="MODEL_SMART")
    model_best: str = Field(default="claude-opus-4-5", alias="MODEL_BEST")

    # ── Extended thinking ─────────────────────────────────────────────────────
    # Enables Claude's extended thinking (chain-of-thought) feature.
    # Only supported on non-Haiku Claude models (sonnet, opus, etc.).
    # When enabled, thinking blocks are streamed to Discord as 🧠 messages.
    # Defaults to False — set THINKING_ENABLED=true only when deep reasoning is needed.
    thinking_enabled: bool = Field(default=False, alias="THINKING_ENABLED")
    thinking_budget_tokens: int = Field(default=1000, alias="THINKING_BUDGET_TOKENS")

    # ── Identity ──────────────────────────────────────────────────────────────
    agent_name: str = Field(default="agent-1", alias="AGENT_NAME")

    # ── Discord ───────────────────────────────────────────────────────────────
    discord_bot_token: SecretStr = Field(default=SecretStr(""), alias="DISCORD_BOT_TOKEN")
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

    # ── Control plane ─────────────────────────────────────────────────────────
    control_plane_enabled: bool = Field(default=True, alias="CONTROL_PLANE_ENABLED")
    control_plane_host: str = Field(default="0.0.0.0", alias="CONTROL_PLANE_HOST")
    control_plane_port: int = Field(default=8000, alias="CONTROL_PLANE_PORT")
    control_plane_sse_ping_seconds: int = Field(
        default=15,
        alias="CONTROL_PLANE_SSE_PING_SECONDS",
    )

    # ── Proxy (optional — routes browser traffic through a residential/ISP proxy)
    # Format: http://user:pass@host:port  or  socks5://user:pass@host:port
    proxy_url: str = Field(default="", alias="PROXY_URL")

    # ── Paths ─────────────────────────────────────────────────────────────────
    workspace_path: Path = Field(default=Path("/workspace"), alias="WORKSPACE_PATH")
    attachments_path: Path = Field(default=Path("/data/attachments"), alias="ATTACHMENTS_PATH")
    agent_secrets_path: Path = Field(
        default=Path("/data/agent-secrets.json"),
        alias="AGENT_SECRETS_PATH",
    )
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
    heartbeat_seconds: int = Field(default=60, alias="HEARTBEAT_SECONDS")
    progress_heartbeat_seconds: int = Field(default=20, alias="PROGRESS_HEARTBEAT_SECONDS")
    restore_pending_discord_tasks: bool = Field(
        default=False,
        alias="RESTORE_PENDING_DISCORD_TASKS",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    attachment_max_bytes: int = Field(default=10_000_000, alias="ATTACHMENT_MAX_BYTES")
    attachment_text_char_cap: int = Field(default=12_000, alias="ATTACHMENT_TEXT_CHAR_CAP")

    # ── Retention / cleanup ───────────────────────────────────────────────────
    # SQLite
    retention_conversations_days: int = Field(default=30, alias="RETENTION_CONVERSATIONS_DAYS")
    retention_tasks_days: int = Field(default=90, alias="RETENTION_TASKS_DAYS")
    retention_memory_facts_max: int = Field(default=500, alias="RETENTION_MEMORY_FACTS_MAX")
    retention_lessons_max: int = Field(default=200, alias="RETENTION_LESSONS_MAX")
    # Postgres
    retention_audit_log_days: int = Field(default=30, alias="RETENTION_AUDIT_LOG_DAYS")
    retention_shared_tasks_days: int = Field(default=14, alias="RETENTION_SHARED_TASKS_DAYS")
    retention_shared_memory_max: int = Field(default=1000, alias="RETENTION_SHARED_MEMORY_MAX")

    @property
    def model_string(self) -> str:
        """Return the pydantic-ai model string for the default model."""
        return self._to_model_string(self.agent_model)

    def _to_model_string(self, model: str) -> str:
        model = model.strip()
        if model.startswith("claude"):
            return f"anthropic:{model}"
        if model.startswith("gpt") or re.match(r"^o\d", model):
            return f"openai:{model}"
        if model.startswith("gemini"):
            return f"google-gla:{model}"
        if model.startswith("grok"):
            return f"xai:{model}"
        if model.startswith(("mistral", "ministral", "codestral", "pixtral")):
            return f"mistral:{model}"
        # Already fully-qualified with a provider prefix — pass through as-is.
        if ":" in model:
            return model
        # Bare open-source model names default to Groq inference.
        if model.startswith(("llama", "mixtral", "qwen", "deepseek", "kimi", "moonshotai")):
            return f"groq:{model}"
        return model

    def model_string_for(self, tier: str) -> str:
        """Return pydantic-ai model string for a named tier: fast | smart | best."""
        tier_map = {"fast": self.model_fast, "smart": self.model_smart, "best": self.model_best}
        return self._to_model_string(tier_map.get(tier, self.agent_model))

    @property
    def has_discord(self) -> bool:
        return bool(self.secret_value(self.discord_bot_token))

    @property
    def has_postgres(self) -> bool:
        return bool(self.postgres_url)

    @property
    def has_embeddings(self) -> bool:
        """True when we can generate embeddings (needs OpenAI key)."""
        return bool(self.secret_value(self.openai_api_key))

    @staticmethod
    def secret_value(value: SecretStr | str) -> str:
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        return value


# Singleton
settings = Settings()
