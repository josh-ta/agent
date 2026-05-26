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
    # Embedding model for semantic memory search (OpenAI or Ollama via OPENAI_BASE_URL)
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")
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
    discord_use_task_threads: bool = Field(default=True, alias="DISCORD_USE_TASK_THREADS")

    # ── Databases ─────────────────────────────────────────────────────────────
    sqlite_path: Path = Field(default=Path("/data/agent.db"), alias="SQLITE_PATH")
    postgres_url: str = Field(default="", alias="POSTGRES_URL")

    # ── MCP ───────────────────────────────────────────────────────────────────
    browser_mcp_url: str = Field(
        default="http://browser:3080/sse", alias="BROWSER_MCP_URL"
    )
    # JSON object mapping MCP name -> SSE URL, e.g. {"custom": "http://host:3081/sse"}
    mcp_servers_json: str = Field(default="", alias="MCP_SERVERS")

    # Web search (tavily or brave)
    web_search_provider: str = Field(default="", alias="WEB_SEARCH_PROVIDER")
    web_search_api_key: SecretStr = Field(default=SecretStr(""), alias="WEB_SEARCH_API_KEY")

    # Comma-separated host allowlist for http_request tool (empty = deny all)
    http_allowed_hosts: str = Field(default="", alias="HTTP_ALLOWED_HOSTS")

    # ── Control plane ─────────────────────────────────────────────────────────
    control_plane_enabled: bool = Field(default=True, alias="CONTROL_PLANE_ENABLED")
    control_plane_host: str = Field(default="0.0.0.0", alias="CONTROL_PLANE_HOST")
    control_plane_port: int = Field(default=8000, alias="CONTROL_PLANE_PORT")
    control_plane_sse_ping_seconds: int = Field(
        default=15,
        alias="CONTROL_PLANE_SSE_PING_SECONDS",
    )
    control_plane_api_key: SecretStr = Field(default=SecretStr(""), alias="CONTROL_PLANE_API_KEY")

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
    agent_secrets_master_key: SecretStr = Field(
        default=SecretStr(""),
        alias="AGENT_SECRETS_MASTER_KEY",
    )
    skills_path: Path = Field(
        default=Path("/app/agent/skills"), alias="SKILLS_PATH"
    )
    identity_path: Path = Field(
        default=Path("/app/agent/identity"), alias="IDENTITY_PATH"
    )

    # When true, read_file/write_file/search_files/list_dir/delete_file/str_replace reject
    # paths outside WORKSPACE_PATH (no /tmp, /data, etc. via those tools).
    filesystem_strict_workspace: bool = Field(
        default=False,
        alias="FILESYSTEM_STRICT_WORKSPACE",
    )

    # ── Self-update ───────────────────────────────────────────────────────────
    docker_restart_self: bool = Field(default=True, alias="DOCKER_RESTART_SELF")
    agent_container_name: str = Field(
        default="agent", alias="AGENT_CONTAINER_NAME"
    )

    # ── Behaviour ─────────────────────────────────────────────────────────────
    heartbeat_seconds: int = Field(default=60, alias="HEARTBEAT_SECONDS")
    progress_heartbeat_seconds: int = Field(default=20, alias="PROGRESS_HEARTBEAT_SECONDS")
    event_sink_timeout_seconds: float = Field(default=5.0, alias="EVENT_SINK_TIMEOUT_SECONDS")
    model_event_idle_timeout_seconds: float = Field(
        default=600.0,
        alias="MODEL_EVENT_IDLE_TIMEOUT_SECONDS",
    )
    restore_pending_discord_tasks: bool = Field(
        default=True,
        alias="RESTORE_PENDING_DISCORD_TASKS",
    )
    runtime_overrides_path: Path = Field(
        default=Path("/data/runtime-overrides.json"),
        alias="RUNTIME_OVERRIDES_PATH",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ── Permissions (CC-style) ────────────────────────────────────────────────
    permission_mode: str = Field(default="default", alias="PERMISSION_MODE")

    # ── Context / transcripts ─────────────────────────────────────────────────
    context_token_warn_threshold: int = Field(default=80_000, alias="CONTEXT_TOKEN_WARN_THRESHOLD")
    restore_transcript_turns: int = Field(default=8, alias="RESTORE_TRANSCRIPT_TURNS")
    attachment_max_bytes: int = Field(default=10_000_000, alias="ATTACHMENT_MAX_BYTES")
    attachment_text_char_cap: int = Field(default=12_000, alias="ATTACHMENT_TEXT_CHAR_CAP")

    # ── Subagents / background scheduling ───────────────────────────────────
    subagent_max_tool_calls: int = Field(default=24, alias="SUBAGENT_MAX_TOOL_CALLS")
    subagent_output_char_cap: int = Field(default=12_000, alias="SUBAGENT_OUTPUT_CHAR_CAP")
    subagent_instruction_char_cap: int = Field(
        default=8_000, alias="SUBAGENT_INSTRUCTION_CHAR_CAP"
    )
    scheduled_tasks_max_rows: int = Field(default=32, alias="SCHEDULED_TASKS_MAX_ROWS")
    scheduled_prompt_char_cap: int = Field(default=8_000, alias="SCHEDULED_PROMPT_CHAR_CAP")
    scheduled_dispatch_per_heartbeat: int = Field(
        default=5, alias="SCHEDULED_DISPATCH_PER_HEARTBEAT"
    )

    # ── Retention / cleanup ───────────────────────────────────────────────────
    # SQLite
    retention_conversations_days: int = Field(default=30, alias="RETENTION_CONVERSATIONS_DAYS")
    retention_tasks_days: int = Field(default=90, alias="RETENTION_TASKS_DAYS")
    retention_memory_facts_max: int = Field(default=500, alias="RETENTION_MEMORY_FACTS_MAX")
    retention_lessons_max: int = Field(default=200, alias="RETENTION_LESSONS_MAX")
    retention_episodic_events_days: int = Field(default=180, alias="RETENTION_EPISODIC_EVENTS_DAYS")
    retention_feedback_events_days: int = Field(default=365, alias="RETENTION_FEEDBACK_EVENTS_DAYS")
    retention_memory_items_max: int = Field(default=1000, alias="RETENTION_MEMORY_ITEMS_MAX")
    retention_procedures_max: int = Field(default=300, alias="RETENTION_PROCEDURES_MAX")
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
        """True when embeddings are configured for OpenAI or a compatible local server."""
        if not self.embedding_model.strip():
            return False
        if self.openai_base_url.strip():
            return True
        return bool(self.secret_value(self.openai_api_key))

    @property
    def uses_local_llm(self) -> bool:
        return bool(self.openai_base_url.strip())

    def openai_compatible_api_key(self) -> str:
        """API key for OpenAI-compatible endpoints; Ollama accepts any non-empty placeholder."""
        key = self.secret_value(self.openai_api_key)
        if key:
            return key
        if self.openai_base_url.strip():
            return "ollama"
        return ""

    @staticmethod
    def secret_value(value: SecretStr | str) -> str:
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        return value


# Singleton
settings = Settings()
