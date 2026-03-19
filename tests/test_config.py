from agent.config import Settings


def test_model_string_maps_supported_providers() -> None:
    settings = Settings(_env_file=None)

    assert settings._to_model_string("claude-sonnet-4-5") == "anthropic:claude-sonnet-4-5"
    assert settings._to_model_string("gpt-4o") == "openai:gpt-4o"
    assert settings._to_model_string("gemini-2.5-pro") == "google-gla:gemini-2.5-pro"
    assert settings._to_model_string("grok-3") == "xai:grok-3"
    assert settings._to_model_string("mistral-large-latest") == "mistral:mistral-large-latest"
    assert settings._to_model_string("llama-3.3-70b-versatile") == "groq:llama-3.3-70b-versatile"


def test_model_string_preserves_fully_qualified_models() -> None:
    settings = Settings(_env_file=None)

    assert settings._to_model_string("openai:local-model") == "openai:local-model"
    assert settings._to_model_string("xai:grok-3-mini") == "xai:grok-3-mini"


def test_model_string_property_uses_agent_model() -> None:
    settings = Settings(_env_file=None, AGENT_MODEL="gpt-4o")

    assert settings.model_string == "openai:gpt-4o"
