from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Alpaca
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_PAPER: bool = True

    # Finnhub
    FINNHUB_API_KEY: str = ""

    # SEC EDGAR (no key required; User-Agent is mandatory per SEC fair-access policy)
    SEC_USER_AGENT: str = "Quorai Research n.flaschel@gmail.com"

    # LLM providers
    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: str = ""
    ANTHROPIC_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    XAI_API_KEY: str = ""
    MISTRAL_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    YOUR_SITE_URL: str = "https://github.com/nils-fl/quorai"
    YOUR_SITE_NAME: str = "Quorai"
    DASHSCOPE_API_KEY: str = ""

    # Kimi / Moonshot
    MOONSHOT_API_KEY: str = ""
    KIMI_API_KEY: str = ""
    MOONSHOT_BASE_URL: str = ""
    KIMI_BASE_URL: str = ""

    # GigaChat
    GIGACHAT_API_KEY: str = ""
    GIGACHAT_CREDENTIALS: str = ""
    GIGACHAT_USER: str = ""
    GIGACHAT_PASSWORD: str = ""

    # Azure OpenAI
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_DEPLOYMENT_NAME: str = ""

    DEFAULT_MODEL: str = "deepseek/deepseek-v4-flash"
    DEFAULT_PROVIDER: str = "OpenRouter"

    # Local / Ollama
    LOCAL_BASE_URL: str = "http://localhost:11434/v1"  # OpenAI-compat endpoint (legacy/fallback)
    LOCAL_API_KEY: str = "ollama"  # any non-empty string; Ollama requires a bearer token
    LOCAL_JSON_MODE: bool = False  # set True for capable 14B+ models that honour response_format
    LOCAL_NUM_CTX: int = 16384  # KV-cache size; 16384 is comfortable on Apple Silicon for 3B-8B models
    LOCAL_NUM_PREDICT: int = 1024  # max tokens to generate per call
    LOCAL_KEEP_ALIVE: str = "1h"  # keep model resident in VRAM; prevents per-call reload stalls

    # Risk gate caps (used by Phase 4)
    MAX_ORDER_NOTIONAL: float = 10_000.0
    MAX_ORDER_QTY: float = 1_000.0
    DAILY_LOSS_LIMIT_PCT: float = 0.05
    KILL_SWITCH: bool = False
    EQUITY_REFRESH_INTERVAL: int = 0  # re-fetch equity every N submitted orders; 0 = disabled

    # Telegram approval
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    TELEGRAM_APPROVAL_TIMEOUT_SECONDS: int = 1800  # 30 min


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def refresh_settings() -> Settings:
    """Clear the settings cache and reload from environment / .env file.

    Call this before checking KILL_SWITCH or other env-controlled flags to pick
    up changes made after process startup (e.g. toggling KILL_SWITCH=true in .env).
    """
    get_settings.cache_clear()
    return get_settings()
