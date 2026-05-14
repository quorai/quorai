"""Helper functions for LLM"""

import json
import logging
import os
import random
import threading
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.graph.state import AgentState
from src.llm.models import get_model, get_model_info
from src.utils.progress import progress

logger = logging.getLogger(__name__)

# Semaphore to cap concurrent LLM calls across all threads.
# Defaults to QUORAI_LLM_MAX_CONCURRENCY env var (default 8).
_llm_semaphore = threading.Semaphore(int(os.environ.get("QUORAI_LLM_MAX_CONCURRENCY", "8")))


class _RateLimiter:
    """Sliding-window rate limiter: caps calls per minute across all threads."""

    def __init__(self, rpm: int) -> None:
        self._rpm = rpm
        self._lock = threading.Lock()
        self._call_times: list[float] = []

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - 60.0
                self._call_times = [t for t in self._call_times if t > cutoff]
                if len(self._call_times) < self._rpm:
                    self._call_times.append(now)
                    return
                sleep_until = self._call_times[0] + 60.0 + 0.1
            time.sleep(max(0.05, sleep_until - time.monotonic()))


_rate_limiter = _RateLimiter(int(os.environ.get("QUORAI_LLM_MAX_RPM", "14")))


class _UsageCapture(BaseCallbackHandler):
    """LangChain callback that captures token usage from any LLM invocation."""

    def __init__(self) -> None:
        super().__init__()
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cache_read_tokens: int = 0
        self.cache_creation_tokens: int = 0

    def on_llm_end(self, response, **kwargs) -> None:
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                um = getattr(msg, "usage_metadata", None) or {}
                self.input_tokens += int(um.get("input_tokens", 0))
                self.output_tokens += int(um.get("output_tokens", 0))
                # OpenRouter surfaces cache counts with different keys per provider
                self.cache_read_tokens += int(
                    um.get("cache_read_input_tokens", 0)  # Anthropic-routed
                    or um.get("prompt_cache_hit_tokens", 0)  # DeepSeek-routed
                )
                self.cache_creation_tokens += int(
                    um.get("cache_creation_input_tokens", 0)  # Anthropic-routed
                    or um.get("prompt_cache_miss_tokens", 0)  # DeepSeek-routed
                )


_run_token_log: list[dict] = []
_token_log_lock = threading.Lock()


def reset_token_log() -> None:
    """Clear the per-run token log. Call before each agent graph invocation."""
    with _token_log_lock:
        _run_token_log.clear()


def get_token_log() -> list[dict]:
    """Return a snapshot of per-call token records for the current run."""
    with _token_log_lock:
        return list(_run_token_log)


def get_token_summary() -> dict:
    """Return aggregated token counts and call count for the current run."""
    with _token_log_lock:
        snapshot = list(_run_token_log)
    return {
        "calls": len(snapshot),
        "input_tokens": sum(e["input_tokens"] for e in snapshot),
        "output_tokens": sum(e["output_tokens"] for e in snapshot),
        "total_tokens": sum(e["input_tokens"] + e["output_tokens"] for e in snapshot),
        "cache_read_tokens": sum(e.get("cache_read_tokens", 0) for e in snapshot),
        "cache_creation_tokens": sum(e.get("cache_creation_tokens", 0) for e in snapshot),
    }


def _attach_cache_control(invoke_prompt: Any, model_name: str, model_provider: str) -> None:
    """Attach Anthropic cache_control to the system message for anthropic/* OpenRouter calls.

    Other OpenRouter-routed models (DeepSeek, Gemini, etc.) cache automatically — no marker needed.
    """
    if model_provider.upper() != "OPENROUTER" or not model_name.startswith("anthropic/"):
        return
    messages = getattr(invoke_prompt, "messages", None) or (invoke_prompt if isinstance(invoke_prompt, list) else None)
    if messages and hasattr(messages[0], "additional_kwargs"):
        messages[0].additional_kwargs.setdefault("cache_control", {"type": "ephemeral"})


def call_llm(
    prompt: Any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: AgentState | None = None,
    max_retries: int = 5,
    default_factory=None,
) -> BaseModel:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates and model config extraction
        state: Optional state object to extract agent-specific model configuration
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """

    if state and agent_name:
        model_name, model_provider = get_agent_model_config(state, agent_name)
    else:
        model_name = "gpt-4.1"
        model_provider = "OPENAI"

    api_keys = None
    if state:
        request = state.get("metadata", {}).get("request")
        if request and hasattr(request, "api_keys"):
            api_keys = request.api_keys

    model_info = get_model_info(model_name, model_provider)
    llm = get_model(model_name, model_provider, api_keys)

    # Bind temperature if the caller requested a specific value (e.g. for A/B experiments).
    # Defaults to None which leaves the provider's default in place.
    if state is not None:
        meta = state.get("metadata", {})
        llm_temperature = meta.get("llm_temperature")
        if llm_temperature is not None:
            llm = llm.bind(temperature=llm_temperature)

    # Three-tier structured output dispatch:
    #   1. Anthropic / OpenAI — provider enforces the schema server-side (tool_use / json_schema)
    #   2. Other json-mode providers (Groq, xAI, OpenRouter …) — JSON hint only
    #   3. DeepSeek / Gemini — no json_mode; parse manually from markdown fences
    #
    # When model_info is None (unlisted model), fall back on the name: DeepSeek and Gemini
    # reject the response_format parameter that json_mode sends, so use manual extraction.
    def _name_implies_manual(name: str, provider: str) -> bool:
        if provider.upper() == "OPENROUTER":
            return False  # OpenRouter normalises response_format for all routed models
        n = name.lower()
        return "deepseek" in n or "gemini" in n

    use_manual_extraction = (model_info is not None and not model_info.has_json_mode()) or (model_info is None and _name_implies_manual(model_name, model_provider))
    if not use_manual_extraction:
        if model_info is not None and model_info.has_native_structured_output():
            llm = llm.with_structured_output(pydantic_model)
        else:
            llm = llm.with_structured_output(pydantic_model, method="json_mode")

    hint = HumanMessage(content="IMPORTANT: You MUST respond with ONLY a valid JSON object matching the schema. No prose, no explanation, no markdown fences.")

    _capture = _UsageCapture()

    for attempt in range(max_retries):
        try:
            if attempt > 0 and use_manual_extraction:
                if hasattr(prompt, "messages"):
                    invoke_prompt = list(prompt.messages) + [hint]
                elif isinstance(prompt, list):
                    invoke_prompt = prompt + [hint]
                else:
                    invoke_prompt = str(prompt) + "\n\nIMPORTANT: Respond with ONLY a valid JSON object. No prose."
            else:
                invoke_prompt = prompt

            _attach_cache_control(invoke_prompt, model_name, model_provider)

            _rate_limiter.acquire()
            with _llm_semaphore:
                result = llm.invoke(invoke_prompt, config={"callbacks": [_capture]})

            token_entry = {
                "agent": agent_name or "unknown",
                "model": model_name,
                "input_tokens": _capture.input_tokens,
                "output_tokens": _capture.output_tokens,
                "cache_read_tokens": _capture.cache_read_tokens,
                "cache_creation_tokens": _capture.cache_creation_tokens,
            }

            if use_manual_extraction:
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    output = pydantic_model(**parsed_result)
                    with _token_log_lock:
                        _run_token_log.append(token_entry)
                    return output
                raise ValueError("Could not extract JSON from LLM response")
            else:
                with _token_log_lock:
                    _run_token_log.append(token_entry)
                return result

        except Exception as exc:
            exc_str = str(exc)

            # Switch extraction mode on provider rejection of json_mode
            if not use_manual_extraction and ("400" in exc_str or "response_format" in exc_str.lower() or "bad request" in exc_str.lower()):
                logger.warning("json_mode rejected by provider for %s (attempt %d); switching to manual extraction", agent_name or model_name, attempt + 1)
                use_manual_extraction = True

            # Exponential back-off for rate-limit errors; immediate retry otherwise.
            # Use a longer base for 429s so we can outlast a 60-second window reset.
            is_rate_limit = "429" in exc_str or "rate_limit" in exc_str.lower() or "rate limit" in exc_str.lower() or "too many requests" in exc_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                backoff = min(10 * (2**attempt) + random.uniform(0, 3), 90)
                logger.warning("Rate limit for %s (attempt %d); backing off %.1fs", agent_name or model_name, attempt + 1, backoff)
                time.sleep(backoff)

            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")

            if attempt == max_retries - 1:
                logger.exception("LLM call failed for %s after %d attempts; using default response", agent_name or "unknown", max_retries)
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    if default_factory:
        return default_factory()
    return create_default_response(pydantic_model)


def create_default_response(model_class: type[BaseModel]) -> BaseModel:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation is str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation is float:
            default_values[field_name] = 0.0
        elif field.annotation is int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ is dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> dict | None:
    """Extracts JSON from response, trying multiple formats."""

    def _as_dict(obj) -> dict | None:
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
        return None

    # Strip leading markdown/label noise (e.g. "**JSON:**\n", "JSON:\n")
    stripped = content.strip()
    for prefix in ("**JSON:**", "JSON:", "**Response:**", "Response:"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].strip()
            break

    # Try fenced code blocks (```json or ```)
    for fence in ("```json", "```JSON", "```"):
        start = stripped.find(fence)
        if start != -1:
            text = stripped[start + len(fence) :]
            end = text.find("```")
            if end != -1:
                try:
                    result = _as_dict(json.loads(text[:end].strip()))
                    if result is not None:
                        return result
                except (json.JSONDecodeError, ValueError):
                    pass

    # Try parsing the whole response as JSON
    try:
        result = _as_dict(json.loads(stripped))
        if result is not None:
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Extract the outermost {...} block
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        try:
            result = _as_dict(json.loads(stripped[start : end + 1]))
            if result is not None:
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("extract_json_from_response: could not parse JSON from response (first 500 chars): %r", content[:500])
    return None


def get_agent_model_config(state, agent_name):
    """
    Get model configuration for a specific agent from the state.
    Falls back to global model configuration if agent-specific config is not available.
    Always returns valid model_name and model_provider values.
    """
    request = state.get("metadata", {}).get("request")

    if request and hasattr(request, "get_agent_model_config"):
        model_name, model_provider = request.get_agent_model_config(agent_name)
        if model_name and model_provider:
            return model_name, model_provider.value if hasattr(model_provider, "value") else str(model_provider)

    meta = state.get("metadata", {})
    model_name = meta.get("model_name") or "gpt-4.1"
    model_provider = meta.get("model_provider") or "OPENAI"

    if hasattr(model_provider, "value"):
        model_provider = model_provider.value

    return model_name, model_provider
