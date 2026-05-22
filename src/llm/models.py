from enum import Enum
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_gigachat import GigaChat
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_xai import ChatXAI
from pydantic import BaseModel

from src.config import get_settings

logger = logging.getLogger(__name__)

_LLM_REQUEST_TIMEOUT = int(os.environ.get("QUORAI_LLM_REQUEST_TIMEOUT", "120"))


class ModelProvider(str, Enum):
    """Enum for supported LLM providers"""

    ALIBABA = "Alibaba"
    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    GOOGLE = "Google"
    GROQ = "Groq"
    KIMI = "Kimi"
    META = "Meta"
    MISTRAL = "Mistral"
    OPENAI = "OpenAI"
    OPENROUTER = "OpenRouter"
    GIGACHAT = "GigaChat"
    AZURE_OPENAI = "Azure OpenAI"
    XAI = "xAI"
    LOCAL = "Local"


class LLMModel(BaseModel):
    """Represents an LLM model configuration"""

    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """Convert to format needed for questionary choices"""
        return (self.display_name, self.model_name, self.provider.value)

    def is_custom(self) -> bool:
        """Check if this is a custom/placeholder model entry."""
        return self.model_name == "-"

    def has_native_structured_output(self) -> bool:
        """True for providers that enforce the Pydantic schema server-side."""
        return self.provider in (ModelProvider.ANTHROPIC, ModelProvider.OPENAI, ModelProvider.AZURE_OPENAI)

    def has_json_mode(self) -> bool:
        """Check if the model supports JSON mode"""
        if self.provider == ModelProvider.OPENROUTER:
            return True  # OpenRouter normalises response_format for all routed models
        if self.provider == ModelProvider.LOCAL:
            return get_settings().LOCAL_JSON_MODE
        if self.is_deepseek() or self.is_gemini():
            return False
        return True

    def is_deepseek(self) -> bool:
        """Check if the model is a DeepSeek model"""
        return self.model_name.startswith("deepseek")

    def is_kimi(self) -> bool:
        """Check if the model is a Kimi (Moonshot) model"""
        return self.provider == ModelProvider.KIMI

    def is_gemini(self) -> bool:
        """Check if the model is a Gemini model"""
        return self.model_name.startswith("gemini") or self.model_name.startswith("google/gemini")


# Load models from JSON file
def load_models_from_json(json_path: str) -> List[LLMModel]:
    """Load models from a JSON file"""
    with open(json_path, "r") as f:
        models_data = json.load(f)

    models = []
    for model_data in models_data:
        # Convert string provider to ModelProvider enum
        provider_enum = ModelProvider(model_data["provider"])
        models.append(LLMModel(display_name=model_data["display_name"], model_name=model_data["model_name"], provider=provider_enum))
    return models


# Get the path to the JSON files
current_dir = Path(__file__).parent
models_json_path = current_dir / "api_models.json"

# Load available models from JSON
AVAILABLE_MODELS = load_models_from_json(str(models_json_path))

# Create LLM_ORDER in the format expected by the UI
LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]


def get_model_info(model_name: str, model_provider: str) -> LLMModel | None:
    """Get model information by model_name"""
    return next((model for model in AVAILABLE_MODELS if model.model_name == model_name and model.provider == model_provider), None)


def find_model_by_name(model_name: str) -> LLMModel | None:
    """Find a model by its name across all available models."""
    return next((model for model in AVAILABLE_MODELS if model.model_name == model_name), None)


def get_models_list():
    """Get the list of models for API responses."""
    return [{"display_name": model.display_name, "model_name": model.model_name, "provider": model.provider.value} for model in AVAILABLE_MODELS]


_model_cache: dict[tuple, Any] = {}


def check_provider_api_key(model_provider: str, api_keys: dict | None = None) -> None:
    """Raise ValueError with a clear message if the required API key is missing.

    Matches the key-lookup logic in _build_model so callers can fail fast before
    the agent graph starts.
    """
    cfg = get_settings()
    p = model_provider.upper()
    if p == ModelProvider.GROQ.value.upper():
        if not ((api_keys or {}).get("GROQ_API_KEY") or cfg.GROQ_API_KEY):
            raise ValueError("Missing GROQ_API_KEY — set it in .env or provide via api_keys")
    elif p == ModelProvider.OPENAI.value.upper():
        if not ((api_keys or {}).get("OPENAI_API_KEY") or cfg.OPENAI_API_KEY):
            raise ValueError("Missing OPENAI_API_KEY — set it in .env or provide via api_keys")
    elif p == ModelProvider.ANTHROPIC.value.upper():
        if not ((api_keys or {}).get("ANTHROPIC_API_KEY") or cfg.ANTHROPIC_API_KEY):
            raise ValueError("Missing ANTHROPIC_API_KEY — set it in .env or provide via api_keys")
    elif p == ModelProvider.DEEPSEEK.value.upper():
        if not ((api_keys or {}).get("DEEPSEEK_API_KEY") or cfg.DEEPSEEK_API_KEY):
            raise ValueError("Missing DEEPSEEK_API_KEY — set it in .env or provide via api_keys")
    elif p == ModelProvider.GOOGLE.value.upper():
        if not ((api_keys or {}).get("GOOGLE_API_KEY") or cfg.GOOGLE_API_KEY):
            raise ValueError("Missing GOOGLE_API_KEY — set it in .env or provide via api_keys")
    elif p == ModelProvider.OPENROUTER.value.upper():
        if not ((api_keys or {}).get("OPENROUTER_API_KEY") or cfg.OPENROUTER_API_KEY):
            raise ValueError("Missing OPENROUTER_API_KEY — set it in .env or provide via api_keys")
    elif p in (ModelProvider.KIMI.value.upper(), "KIMI"):
        if not ((api_keys or {}).get("MOONSHOT_API_KEY") or cfg.MOONSHOT_API_KEY or (api_keys or {}).get("KIMI_API_KEY") or cfg.KIMI_API_KEY):
            raise ValueError("Missing MOONSHOT_API_KEY (or KIMI_API_KEY) — set it in .env or provide via api_keys")
    elif p == ModelProvider.XAI.value.upper():
        if not ((api_keys or {}).get("XAI_API_KEY") or cfg.XAI_API_KEY):
            raise ValueError("Missing XAI_API_KEY — set it in .env or provide via api_keys")
    elif p == ModelProvider.AZURE_OPENAI.value.upper():
        if not cfg.AZURE_OPENAI_API_KEY:
            raise ValueError("Missing AZURE_OPENAI_API_KEY — set it in .env")
    elif p == ModelProvider.LOCAL.value.upper():
        pass  # local servers don't require an API key
    # GigaChat and unknown providers are skipped (GigaChat uses user/password auth).


def get_model(model_name: str, model_provider: ModelProvider, api_keys: dict = None) -> ChatOpenAI | ChatGroq | GigaChat | None:
    cache_key = (model_name, str(model_provider), frozenset((api_keys or {}).items()))
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    instance = _build_model(model_name, model_provider, api_keys)
    _model_cache[cache_key] = instance
    return instance


def _build_model(model_name: str, model_provider: ModelProvider, api_keys: dict = None) -> ChatOpenAI | ChatGroq | GigaChat | None:
    cfg = get_settings()
    if model_provider == ModelProvider.GROQ:
        api_key = (api_keys or {}).get("GROQ_API_KEY") or cfg.GROQ_API_KEY
        if not api_key:
            logger.warning("API Key Error: Please make sure GROQ_API_KEY is set in your .env file or provided via API keys.")
            raise ValueError("Groq API key not found.  Please make sure GROQ_API_KEY is set in your .env file or provided via API keys.")
        return ChatGroq(model=model_name, api_key=api_key, timeout=_LLM_REQUEST_TIMEOUT)
    elif model_provider == ModelProvider.OPENAI:
        api_key = (api_keys or {}).get("OPENAI_API_KEY") or cfg.OPENAI_API_KEY
        base_url = cfg.OPENAI_API_BASE or None
        if not api_key:
            logger.warning("API Key Error: Please make sure OPENAI_API_KEY is set in your .env file or provided via API keys.")
            raise ValueError("OpenAI API key not found.  Please make sure OPENAI_API_KEY is set in your .env file or provided via API keys.")
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, timeout=_LLM_REQUEST_TIMEOUT)
    elif model_provider == ModelProvider.ANTHROPIC:
        api_key = (api_keys or {}).get("ANTHROPIC_API_KEY") or cfg.ANTHROPIC_API_KEY
        if not api_key:
            logger.warning("API Key Error: Please make sure ANTHROPIC_API_KEY is set in your .env file or provided via API keys.")
            raise ValueError("Anthropic API key not found.  Please make sure ANTHROPIC_API_KEY is set in your .env file or provided via API keys.")
        return ChatAnthropic(model=model_name, api_key=api_key, timeout=_LLM_REQUEST_TIMEOUT)
    elif model_provider == ModelProvider.DEEPSEEK:
        api_key = (api_keys or {}).get("DEEPSEEK_API_KEY") or cfg.DEEPSEEK_API_KEY
        if not api_key:
            logger.warning("API Key Error: Please make sure DEEPSEEK_API_KEY is set in your .env file or provided via API keys.")
            raise ValueError("DeepSeek API key not found.  Please make sure DEEPSEEK_API_KEY is set in your .env file or provided via API keys.")
        return ChatDeepSeek(model=model_name, api_key=api_key, timeout=_LLM_REQUEST_TIMEOUT)
    elif model_provider == ModelProvider.GOOGLE:
        api_key = (api_keys or {}).get("GOOGLE_API_KEY") or cfg.GOOGLE_API_KEY
        if not api_key:
            logger.warning("API Key Error: Please make sure GOOGLE_API_KEY is set in your .env file or provided via API keys.")
            raise ValueError("Google API key not found.  Please make sure GOOGLE_API_KEY is set in your .env file or provided via API keys.")
        return ChatGoogleGenerativeAI(model=model_name, api_key=api_key, timeout=_LLM_REQUEST_TIMEOUT)
    elif model_provider == ModelProvider.OPENROUTER:
        api_key = (api_keys or {}).get("OPENROUTER_API_KEY") or cfg.OPENROUTER_API_KEY
        if not api_key:
            logger.warning("API Key Error: Please make sure OPENROUTER_API_KEY is set in your .env file or provided via API keys.")
            raise ValueError("OpenRouter API key not found. Please make sure OPENROUTER_API_KEY is set in your .env file or provided via API keys.")

        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            timeout=_LLM_REQUEST_TIMEOUT,
            max_retries=1,
            model_kwargs={
                "extra_headers": {
                    "HTTP-Referer": cfg.YOUR_SITE_URL,
                    "X-Title": cfg.YOUR_SITE_NAME,
                }
            },
        )
    elif model_provider == ModelProvider.KIMI:
        api_key = (api_keys or {}).get("MOONSHOT_API_KEY") or cfg.MOONSHOT_API_KEY or (api_keys or {}).get("KIMI_API_KEY") or cfg.KIMI_API_KEY
        if not api_key:
            logger.warning("API Key Error: Please make sure MOONSHOT_API_KEY (or KIMI_API_KEY) is set in your .env file or provided via API keys.")
            raise ValueError("Kimi API key not found. Please make sure MOONSHOT_API_KEY (or KIMI_API_KEY) is set in your .env file or provided via API keys.")
        base_url = cfg.MOONSHOT_BASE_URL or cfg.KIMI_BASE_URL or "https://api.moonshot.ai/v1"
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, timeout=_LLM_REQUEST_TIMEOUT)
    elif model_provider == ModelProvider.XAI:
        api_key = (api_keys or {}).get("XAI_API_KEY") or cfg.XAI_API_KEY
        if not api_key:
            logger.warning("API Key Error: Please make sure XAI_API_KEY is set in your .env file or provided via API keys.")
            raise ValueError("xAI API key not found. Please make sure XAI_API_KEY is set in your .env file or provided via API keys.")
        return ChatXAI(model=model_name, api_key=api_key, timeout=_LLM_REQUEST_TIMEOUT)
    elif model_provider == ModelProvider.GIGACHAT:
        if cfg.GIGACHAT_USER or cfg.GIGACHAT_PASSWORD:
            return GigaChat(model=model_name)
        else:
            api_key = (api_keys or {}).get("GIGACHAT_API_KEY") or cfg.GIGACHAT_API_KEY or cfg.GIGACHAT_CREDENTIALS
            if not api_key:
                logger.warning("API Key Error: Please make sure api_keys is set in your .env file or provided via API keys.")
                raise ValueError("GigaChat API key not found. Please make sure GIGACHAT_API_KEY is set in your .env file or provided via API keys.")

            return GigaChat(credentials=api_key, model=model_name)
    elif model_provider == ModelProvider.AZURE_OPENAI:
        api_key = cfg.AZURE_OPENAI_API_KEY
        if not api_key:
            logger.warning("API Key Error: Please make sure AZURE_OPENAI_API_KEY is set in your .env file.")
            raise ValueError("Azure OpenAI API key not found.  Please make sure AZURE_OPENAI_API_KEY is set in your .env file.")
        azure_endpoint = cfg.AZURE_OPENAI_ENDPOINT
        if not azure_endpoint:
            logger.warning("Azure Endpoint Error: Please make sure AZURE_OPENAI_ENDPOINT is set in your .env file.")
            raise ValueError("Azure OpenAI endpoint not found.  Please make sure AZURE_OPENAI_ENDPOINT is set in your .env file.")
        azure_deployment_name = cfg.AZURE_OPENAI_DEPLOYMENT_NAME
        if not azure_deployment_name:
            logger.warning("Azure Deployment Name Error: Please make sure AZURE_OPENAI_DEPLOYMENT_NAME is set in your .env file.")
            raise ValueError("Azure OpenAI deployment name not found.  Please make sure AZURE_OPENAI_DEPLOYMENT_NAME is set in your .env file.")
        return AzureChatOpenAI(azure_endpoint=azure_endpoint, azure_deployment=azure_deployment_name, api_key=api_key, api_version="2024-10-21", timeout=_LLM_REQUEST_TIMEOUT)
    elif model_provider == ModelProvider.LOCAL:
        cfg = get_settings()
        # Strip any /v1 suffix — ChatOllama talks to the native Ollama endpoint, not the OpenAI-compat shim.
        base_url = cfg.LOCAL_BASE_URL.removesuffix("/v1")
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0,
            num_ctx=cfg.LOCAL_NUM_CTX,
            num_predict=cfg.LOCAL_NUM_PREDICT,
            keep_alive=cfg.LOCAL_KEEP_ALIVE,
        )
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}. Supported providers: {', '.join(p.value for p in ModelProvider)}")
