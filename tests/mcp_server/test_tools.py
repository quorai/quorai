"""Tests for src/mcp_server/tools.py — pure functions, no I/O."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.llm.request import RunRequest
from src.mcp_server.tools import (
    build_panel_result,
    build_portfolio,
    build_request,
    get_analyst_info_impl,
    list_analysts_impl,
    parse_dates,
    validate_analyst_keys,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_MOCK_RAW = {
    "decisions": {
        "AAPL": {"action": "buy", "quantity": 100.0, "confidence": 75, "reasoning": "Strong moat"},
    },
    "analyst_signals": {
        "warren_buffett_agent": {
            "AAPL": {"signal": "bullish", "confidence": 80, "reasoning": "Durable franchise"},
        },
        "risk_management_agent": {
            "AAPL": {
                "reasoning": {"position_limit": 10000.0, "remaining_limit": 8000.0, "correlation_multiplier": 1.0},
                "volatility_metrics": {"annualized_volatility": 0.25},
                "remaining_position_limit": 8000.0,
            },
        },
    },
    "group_signals": {"AAPL": {"quality_compounders": {"stance": "bullish"}}},
    "debate_summaries": {
        "AAPL": {
            "consensus_strength": "strong_agreement",
            "core_disagreement": "Valuation concerns",
            "group_positions": [{"group": "quality_compounders", "stance": "bullish", "key_argument": "Wide moat"}],
        },
    },
    "current_prices": {"AAPL": 185.50},
}


# ── list_analysts_impl ────────────────────────────────────────────────────────


def test_list_analysts_returns_25():
    analysts = list_analysts_impl()
    assert len(analysts) == 25


def test_list_analysts_fields():
    analysts = list_analysts_impl()
    for a in analysts:
        assert a.key
        assert a.display_name
        assert a.description
        assert a.investing_style
        assert a.pull_quote
        assert a.strategy_group
        assert isinstance(a.order, int)


def test_list_analysts_ordered():
    analysts = list_analysts_impl()
    orders = [a.order for a in analysts]
    assert orders == sorted(orders)


def test_list_analysts_unique_keys():
    analysts = list_analysts_impl()
    keys = [a.key for a in analysts]
    assert len(keys) == len(set(keys))


# ── get_analyst_info_impl ─────────────────────────────────────────────────────


def test_get_analyst_info_known_key():
    info = get_analyst_info_impl("warren_buffett")
    assert info.key == "warren_buffett"
    assert info.display_name == "Warren Buffett"
    assert info.strategy_group == "quality_compounders"


def test_get_analyst_info_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown analyst key"):
        get_analyst_info_impl("definitely_not_real")


# ── parse_dates ───────────────────────────────────────────────────────────────


def test_parse_dates_defaults():
    start, end = parse_dates(None, None)
    today = str(date.today())
    expected_start = str(date.today() - timedelta(days=30))
    assert end == today
    assert start == expected_start


def test_parse_dates_explicit():
    start, end = parse_dates("2024-01-01", "2024-01-31")
    assert start == "2024-01-01"
    assert end == "2024-01-31"


def test_parse_dates_only_end():
    start, end = parse_dates(None, "2024-06-30")
    assert end == "2024-06-30"
    assert start == "2024-05-31"


def test_parse_dates_bad_format_raises():
    with pytest.raises(ValueError, match="Invalid end_date"):
        parse_dates(None, "31-01-2024")


def test_parse_dates_start_after_end_raises():
    with pytest.raises(ValueError, match="must be <="):
        parse_dates("2024-02-01", "2024-01-01")


# ── validate_analyst_keys ─────────────────────────────────────────────────────


def test_validate_analyst_keys_valid():
    validate_analyst_keys(["warren_buffett", "ben_graham"])


def test_validate_analyst_keys_invalid_raises():
    with pytest.raises(ValueError, match="Unknown analyst key"):
        validate_analyst_keys(["warren_buffett", "not_real"])


# ── build_portfolio ───────────────────────────────────────────────────────────


def test_build_portfolio_shape():
    port = build_portfolio(["AAPL", "MSFT"], initial_cash=50000.0)
    assert port["cash"] == 50000.0
    assert set(port["positions"]) == {"AAPL", "MSFT"}
    assert port["positions"]["AAPL"]["long"] == 0
    assert port["realized_gains"]["MSFT"]["long"] == 0.0


# ── build_request ─────────────────────────────────────────────────────────────


def test_build_request_none():
    assert build_request(None) is None


def test_build_request_adds_agent_suffix():
    req = build_request({"warren_buffett": ["model-x", "OpenRouter"]})
    assert isinstance(req, RunRequest)
    assert "warren_buffett_agent" in req.agent_models
    assert req.agent_models["warren_buffett_agent"] == ("model-x", "OpenRouter")


def test_build_request_wildcard_unchanged():
    req = build_request({"*": ["fallback-model", "OpenRouter"]})
    assert "*" in req.agent_models
    assert req.agent_models["*"] == ("fallback-model", "OpenRouter")


def test_build_request_mixed():
    req = build_request(
        {
            "warren_buffett": ["big-model", "Anthropic"],
            "*": ["small-model", "OpenRouter"],
        }
    )
    assert req.agent_models["warren_buffett_agent"] == ("big-model", "Anthropic")
    assert req.agent_models["*"] == ("small-model", "OpenRouter")


# ── build_panel_result ────────────────────────────────────────────────────────


def test_build_panel_result_basic():
    result = build_panel_result(_MOCK_RAW, ["AAPL"], "2024-01-01", "2024-01-31")
    assert result.tickers == ["AAPL"]
    assert result.window == {"start": "2024-01-01", "end": "2024-01-31"}


def test_build_panel_result_decisions():
    result = build_panel_result(_MOCK_RAW, ["AAPL"], "2024-01-01", "2024-01-31")
    assert "AAPL" in result.portfolio_decisions
    d = result.portfolio_decisions["AAPL"]
    assert d.action == "buy"
    assert d.quantity == 100.0
    assert d.confidence == 75


def test_build_panel_result_analyst_signals():
    result = build_panel_result(_MOCK_RAW, ["AAPL"], "2024-01-01", "2024-01-31")
    assert "warren_buffett_agent" in result.analyst_signals
    s = result.analyst_signals["warren_buffett_agent"]["AAPL"]
    assert s.signal == "bullish"
    assert s.confidence == 80


def test_build_panel_result_excludes_risk_from_analyst_signals():
    result = build_panel_result(_MOCK_RAW, ["AAPL"], "2024-01-01", "2024-01-31")
    assert "risk_management_agent" not in result.analyst_signals


def test_build_panel_result_risk_assessments():
    result = build_panel_result(_MOCK_RAW, ["AAPL"], "2024-01-01", "2024-01-31")
    assert "AAPL" in result.risk_assessments
    r = result.risk_assessments["AAPL"]
    assert r.position_limit == 10000.0
    assert r.remaining_limit == 8000.0
    assert r.annualized_volatility == 0.25


def test_build_panel_result_debate_summaries():
    result = build_panel_result(_MOCK_RAW, ["AAPL"], "2024-01-01", "2024-01-31")
    assert "AAPL" in result.debate_summaries
    db = result.debate_summaries["AAPL"]
    assert db.consensus_strength == "strong_agreement"
    assert db.group_positions is not None
    assert db.group_positions[0].group == "quality_compounders"


def test_build_panel_result_current_prices():
    result = build_panel_result(_MOCK_RAW, ["AAPL"], "2024-01-01", "2024-01-31")
    assert result.current_prices == {"AAPL": 185.50}


def test_build_panel_result_markdown_summary():
    result = build_panel_result(_MOCK_RAW, ["AAPL"], "2024-01-01", "2024-01-31")
    md = result.markdown_summary
    assert "AAPL" in md
    assert "BUY" in md
    assert "BULLISH" in md
    assert "2024-01-01" in md


def test_build_panel_result_none_decisions():
    raw = dict(_MOCK_RAW, decisions=None)
    result = build_panel_result(raw, ["AAPL"], "2024-01-01", "2024-01-31")
    assert result.portfolio_decisions == {}
