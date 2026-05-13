"""Tests for LLM token telemetry (task 20260511-1700)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pydantic import BaseModel

import src.utils.llm as llm_module
from src.utils.llm import (
    _UsageCapture,
    get_token_log,
    get_token_summary,
    reset_token_log,
)


class _SimpleModel(BaseModel):
    value: str = "ok"


# ---------------------------------------------------------------------------
# _UsageCapture callback
# ---------------------------------------------------------------------------


def _make_fake_response(input_tokens: int = 10, output_tokens: int = 5):
    """Build a fake LangChain LLMResult-like object."""
    msg = MagicMock()
    msg.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    gen = MagicMock()
    gen.message = msg
    response = MagicMock()
    response.generations = [[gen]]
    return response


def test_usage_capture_accumulates_tokens():
    capture = _UsageCapture()
    capture.on_llm_end(_make_fake_response(10, 5))
    assert capture.input_tokens == 10
    assert capture.output_tokens == 5


def test_usage_capture_accumulates_across_calls():
    capture = _UsageCapture()
    capture.on_llm_end(_make_fake_response(10, 5))
    capture.on_llm_end(_make_fake_response(20, 8))
    assert capture.input_tokens == 30
    assert capture.output_tokens == 13


def test_usage_capture_handles_missing_metadata():
    capture = _UsageCapture()
    gen = MagicMock()
    gen.message = MagicMock(spec=[])  # no usage_metadata attribute
    response = MagicMock()
    response.generations = [[gen]]
    capture.on_llm_end(response)  # should not raise
    assert capture.input_tokens == 0
    assert capture.output_tokens == 0


# ---------------------------------------------------------------------------
# Module-level log helpers
# ---------------------------------------------------------------------------


def test_reset_clears_log():
    llm_module._run_token_log.append({"agent": "test", "model": "x", "input_tokens": 1, "output_tokens": 1})
    reset_token_log()
    assert get_token_log() == []


def test_get_token_log_returns_snapshot():
    reset_token_log()
    llm_module._run_token_log.append({"agent": "a", "model": "m", "input_tokens": 5, "output_tokens": 3})
    snapshot = get_token_log()
    assert len(snapshot) == 1
    # Snapshot is independent of the live list
    snapshot.append({"agent": "b", "model": "m", "input_tokens": 0, "output_tokens": 0})
    assert len(get_token_log()) == 1


def test_get_token_summary():
    reset_token_log()
    llm_module._run_token_log.extend(
        [
            {"agent": "a", "model": "m", "input_tokens": 10, "output_tokens": 4},
            {"agent": "b", "model": "m", "input_tokens": 20, "output_tokens": 6},
        ]
    )
    summary = get_token_summary()
    assert summary["calls"] == 2
    assert summary["input_tokens"] == 30
    assert summary["output_tokens"] == 10
    assert summary["total_tokens"] == 40


def test_get_token_summary_empty():
    reset_token_log()
    summary = get_token_summary()
    assert summary == {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cache_read_tokens": 0, "cache_creation_tokens": 0}


# ---------------------------------------------------------------------------
# call_llm integration: token log is populated on success
# ---------------------------------------------------------------------------


def test_call_llm_logs_token_usage_on_success():
    """call_llm() should append an entry to _run_token_log after a successful invocation."""
    reset_token_log()

    fake_result = _SimpleModel(value="hello")

    with patch("src.utils.llm.get_model") as mock_get_model, patch("src.utils.llm.get_model_info", return_value=None):
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.bind.return_value = mock_llm

        # Simulate the invoke returning the Pydantic model (structured-output path).
        # The callback won't fire for a mock, but the log entry is still written.
        mock_llm.invoke.return_value = fake_result
        mock_get_model.return_value = mock_llm

        from src.utils.llm import call_llm

        result = call_llm(
            prompt="test prompt",
            pydantic_model=_SimpleModel,
            agent_name="test_agent",
        )

    assert result == fake_result
    log = get_token_log()
    assert len(log) == 1
    assert log[0]["agent"] == "test_agent"
    assert log[0]["input_tokens"] == 0  # callback didn't fire (mock), so 0
    assert log[0]["output_tokens"] == 0
