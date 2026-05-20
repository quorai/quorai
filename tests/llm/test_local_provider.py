"""Tests for the LOCAL (Ollama) provider."""

from unittest.mock import MagicMock

from langchain_ollama import ChatOllama

from src.llm.models import ModelProvider, check_provider_api_key, get_model


def _make_settings(**overrides):
    s = MagicMock()
    s.LOCAL_BASE_URL = "http://localhost:11434/v1"
    s.LOCAL_API_KEY = "ollama"
    s.LOCAL_JSON_MODE = False
    s.LOCAL_NUM_CTX = 16384
    s.LOCAL_NUM_PREDICT = 1024
    s.LOCAL_KEEP_ALIVE = "1h"
    for attr, value in overrides.items():
        setattr(s, attr, value)
    return s


def test_local_provider_builds_chat_ollama(monkeypatch):
    monkeypatch.setattr("src.llm.models.get_settings", lambda: _make_settings())
    from src.llm import models

    models._model_cache.clear()

    model = get_model("qwen2.5:14b", ModelProvider.LOCAL)

    assert isinstance(model, ChatOllama)
    assert model.model == "qwen2.5:14b"
    assert "11434" in str(model.base_url)


def test_local_provider_ollama_options(monkeypatch):
    monkeypatch.setattr("src.llm.models.get_settings", lambda: _make_settings())
    from src.llm import models

    models._model_cache.clear()

    model = get_model("granite4.1:3b", ModelProvider.LOCAL)

    assert isinstance(model, ChatOllama)
    assert model.num_ctx == 16384
    assert model.num_predict == 1024
    assert model.keep_alive == "1h"
    assert model.temperature == 0


def test_local_provider_custom_base_url(monkeypatch):
    monkeypatch.setattr("src.llm.models.get_settings", lambda: _make_settings(LOCAL_BASE_URL="http://localhost:8080/v1"))
    from src.llm import models

    models._model_cache.clear()

    model = get_model("llama3.1:8b", ModelProvider.LOCAL)

    assert isinstance(model, ChatOllama)
    assert "8080" in str(model.base_url)
    assert "/v1" not in str(model.base_url)


def test_local_provider_strips_v1_suffix(monkeypatch):
    monkeypatch.setattr("src.llm.models.get_settings", lambda: _make_settings(LOCAL_BASE_URL="http://localhost:11434/v1"))
    from src.llm import models

    models._model_cache.clear()

    model = get_model("llama3.1:8b", ModelProvider.LOCAL)

    assert "/v1" not in str(model.base_url)


def test_has_json_mode_false_by_default(monkeypatch):
    monkeypatch.setattr("src.llm.models.get_settings", lambda: _make_settings(LOCAL_JSON_MODE=False))
    from src.llm.models import LLMModel

    m = LLMModel(display_name="Qwen", model_name="qwen2.5:14b", provider=ModelProvider.LOCAL)
    assert m.has_json_mode() is False


def test_has_json_mode_true_when_opted_in(monkeypatch):
    monkeypatch.setattr("src.llm.models.get_settings", lambda: _make_settings(LOCAL_JSON_MODE=True))
    from src.llm.models import LLMModel

    m = LLMModel(display_name="Qwen", model_name="qwen2.5:14b", provider=ModelProvider.LOCAL)
    assert m.has_json_mode() is True


def test_has_native_structured_output_false(monkeypatch):
    from src.llm.models import LLMModel

    m = LLMModel(display_name="Qwen", model_name="qwen2.5:14b", provider=ModelProvider.LOCAL)
    assert m.has_native_structured_output() is False


def test_check_provider_api_key_passes_without_key(monkeypatch):
    monkeypatch.setattr("src.llm.models.get_settings", lambda: _make_settings())
    check_provider_api_key(ModelProvider.LOCAL.value, {})
