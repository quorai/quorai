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

    # Risk gate caps (used by Phase 4)
    MAX_ORDER_NOTIONAL: float = 10_000.0
    MAX_ORDER_QTY: float = 1_000.0
    DAILY_LOSS_LIMIT_PCT: float = 0.05
    KILL_SWITCH: bool = False

    # Telegram approval
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    TELEGRAM_APPROVAL_TIMEOUT_SECONDS: int = 1800  # 30 min


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
