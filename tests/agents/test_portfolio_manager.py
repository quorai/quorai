"""Tests for portfolio manager decision logic: allowed actions and pre-fill/LLM split."""

from unittest.mock import patch

import pytest

from src.agents.portfolio_manager import (
    PortfolioDecision,
    PortfolioManagerOutput,
    compute_allowed_actions,
    generate_trading_decision,
)


def _portfolio(cash: float = 10_000.0, equity: float | None = None, positions: dict | None = None) -> dict:
    p: dict = {
        "cash": cash,
        "positions": positions or {},
        "margin_requirement": 0.5,
        "margin_used": 0.0,
    }
    if equity is not None:
        p["equity"] = equity
    return p


def _state() -> dict:
    return {"messages": [], "data": {}, "metadata": {"show_reasoning": False}}


class TestComputeAllowedActions:
    def test_zero_max_shares_results_in_hold_only(self):
        """max_shares == 0 → no buy/short possible; only hold is returned."""
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 150.0},
            max_shares={"AAPL": 0},
            portfolio=_portfolio(cash=10_000.0, equity=10_000.0),
        )
        assert result["AAPL"] == {"hold": 0}

    def test_fractional_share_capacity_allows_actions(self):
        """max_shares = 0.5 (fractional) → buy and short actions are present (not hold-only)."""
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 150.0},
            max_shares={"AAPL": 0.5},
            portfolio=_portfolio(cash=10_000.0, equity=10_000.0),
        )
        assert "buy" in result["AAPL"]
        assert result["AAPL"]["buy"] == pytest.approx(0.5)

    def test_existing_long_adds_sell_action(self):
        """Holding long shares exposes the sell action equal to held quantity."""
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio(
                cash=5_000.0,
                equity=6_000.0,
                positions={"AAPL": {"long": 5.0, "long_cost_basis": 100.0, "short": 0.0, "short_cost_basis": 0.0}},
            ),
        )
        assert "sell" in result["AAPL"]
        assert result["AAPL"]["sell"] == pytest.approx(5.0)

    def test_max_short_shares_allows_larger_short_than_long(self):
        """
        Regression for F2: when cash is low (low long cap) but margin capacity is high,
        providing a separate max_short_shares allows a larger short than the long cap.
        """
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 5.0},       # long cap: 5 shares ($500 from cash)
            max_short_shares={"AAPL": 20.0},  # short cap: 20 shares (margin-derived)
            portfolio=_portfolio(
                cash=500.0,
                equity=10_000.0,
                positions={},
            ),
        )
        long_cap = result["AAPL"].get("buy", 0)
        short_cap = result["AAPL"].get("short", 0)
        assert long_cap == pytest.approx(5.0), f"Expected buy=5.0, got {long_cap}"
        assert short_cap > long_cap, (
            f"Short capacity ({short_cap}) should exceed long capacity ({long_cap}) "
            "when max_short_shares is higher than max_shares. F2 bug: max_short_shares ignored."
        )

    def test_equity_fallback_uses_cash_when_equity_missing(self):
        """portfolio without 'equity' key falls back to cash for margin calculation (M13 known behaviour)."""
        portfolio_no_equity = {
            "cash": 5_000.0,
            "positions": {},
            "margin_requirement": 0.5,
            "margin_used": 0.0,
            # no 'equity' key
        }
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=portfolio_no_equity,
        )
        # available_margin = (cash / margin_requirement) - margin_used = (5000 / 0.5) - 0 = 10000
        # max_short_margin = 10000 / 100 = 100; capped to max_qty=10
        assert "short" in result["AAPL"]
        assert result["AAPL"]["short"] == pytest.approx(10.0)


class TestGenerateTradingDecision:
    def test_zero_capacity_ticker_prefilled_not_sent_to_llm(self):
        """All-zero max_shares → all tickers pre-filled as hold; LLM is never called."""
        result = generate_trading_decision(
            tickers=["AAPL", "MSFT"],
            signals_by_ticker={"AAPL": {}, "MSFT": {}},
            current_prices={"AAPL": 150.0, "MSFT": 300.0},
            max_shares={"AAPL": 0, "MSFT": 0},
            portfolio=_portfolio(cash=0.0, equity=0.0),
            agent_id="portfolio_manager",
            state=_state(),
        )
        assert result.decisions["AAPL"].action == "hold"
        assert result.decisions["MSFT"].action == "hold"

    def test_nonzero_capacity_ticker_sent_to_llm(self):
        """Ticker with max_shares > 0 is forwarded to the LLM; mock LLM result is used."""
        mock_output = PortfolioManagerOutput(decisions={"AAPL": PortfolioDecision(action="buy", quantity=1.0, confidence=80, reasoning="bullish")})
        with patch("src.agents.portfolio_manager.call_llm", return_value=mock_output):
            result = generate_trading_decision(
                tickers=["AAPL"],
                signals_by_ticker={"AAPL": {"agent_x": {"sig": "bullish", "conf": 75}}},
                current_prices={"AAPL": 150.0},
                max_shares={"AAPL": 5.0},
                portfolio=_portfolio(cash=10_000.0, equity=10_000.0),
                agent_id="portfolio_manager",
                state=_state(),
            )

        assert result.decisions["AAPL"].action == "buy"
        assert result.decisions["AAPL"].quantity == pytest.approx(1.0)
