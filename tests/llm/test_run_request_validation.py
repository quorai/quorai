"""Tests for RunRequest.validate_provider_keys — fail-fast on missing API keys."""

from unittest.mock import MagicMock

import pytest

from src.llm.request import RunRequest


def _make_settings(**overrides):
    s = MagicMock()
    # All keys absent by default
    for attr in (
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "MOONSHOT_API_KEY",
        "KIMI_API_KEY",
        "XAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
    ):
        setattr(s, attr, None)
    for attr, value in overrides.items():
        setattr(s, attr, value)
    return s


def test_valid_when_key_present(monkeypatch):
    monkeypatch.setattr("src.llm.models.get_settings", lambda: _make_settings(OPENROUTER_API_KEY="sk-test"))
    req = RunRequest(agent_models={"*": ("deepseek/v3", "OpenRouter")})
    req.validate_provider_keys(None, global_model="deepseek/v3", global_provider="OpenRouter")


def test_wildcard_entry_missing_key_raises(monkeypatch):
    monkeypatch.setattr("src.llm.models.get_settings", lambda: _make_settings())
    req = RunRequest(agent_models={"*": ("deepseek/v3", "OpenRouter")})
    with pytest.raises(ValueError, match="agent_models entry '\\*'"):
        req.validate_provider_keys(None)


def test_named_entry_missing_key_raises(monkeypatch):
    monkeypatch.setattr("src.llm.models.get_settings", lambda: _make_settings())
    req = RunRequest(agent_models={"warren_buffett_agent": ("claude-3-5-sonnet", "Anthropic")})
    with pytest.raises(ValueError, match="warren_buffett_agent"):
        req.validate_provider_keys(None)


def test_global_provider_missing_key_raises(monkeypatch):
    monkeypatch.setattr("src.llm.models.get_settings", lambda: _make_settings())
    req = RunRequest()
    with pytest.raises(ValueError, match="global model"):
        req.validate_provider_keys(None, global_model="deepseek/v3", global_provider="OpenRouter")


def test_no_agent_models_no_global_passes(monkeypatch):
    monkeypatch.setattr("src.llm.models.get_settings", lambda: _make_settings())
    req = RunRequest()
    req.validate_provider_keys(None)  # no agent_models, no global — nothing to validate


def test_error_message_includes_missing_key_name(monkeypatch):
    monkeypatch.setattr("src.llm.models.get_settings", lambda: _make_settings())
    req = RunRequest(agent_models={"*": ("gpt-4", "OpenAI")})
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        req.validate_provider_keys(None)
