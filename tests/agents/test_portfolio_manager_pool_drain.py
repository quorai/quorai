"""Tests for R42: compute_allowed_actions drains the shared capital pool across tickers."""

import pytest

from src.agents.portfolio_manager import compute_allowed_actions


def _portfolio(cash=10_000.0, equity=10_000.0, margin_requirement=0.5, margin_used=0.0):
    return {
        "cash": cash,
        "equity": equity,
        "margin_requirement": margin_requirement,
        "margin_used": margin_used,
        "positions": {},
    }


class TestPortfolioManagerPoolDrain:
    def test_second_ticker_sees_residual_cash_after_first_buy(self):
        """
        If AAPL's max_buy reserves $6k of $10k cash, MSFT must see at most $4k, not $10k.
        Pre-fix: both tickers each saw the full $10k pool (double-spend).
        Post-fix: MSFT's max_buy_cash = (10_000 - max_aapl_notional) / price_msft.
        """
        portfolio = _portfolio(cash=10_000.0)
        prices = {"AAPL": 100.0, "MSFT": 100.0}
        max_shares = {"AAPL": 60.0, "MSFT": 60.0}  # AAPL capped at 60 → $6k notional

        allowed = compute_allowed_actions(["AAPL", "MSFT"], prices, max_shares, portfolio)

        aapl_buy = allowed["AAPL"].get("buy", 0)
        msft_buy = allowed["MSFT"].get("buy", 0)

        assert aapl_buy == pytest.approx(60.0)  # AAPL limited by max_shares=60

        # MSFT can only buy with cash remaining after AAPL's max allocation ($6k used → $4k left)
        assert msft_buy <= 40.0, f"MSFT max_buy should be ≤ 40 shares ($4k remaining), got {msft_buy}. Double-spend bug: both tickers saw the full $10k pool."

    def test_combined_buy_notional_does_not_exceed_cash(self):
        """Sum of all max_buy × price must not exceed initial cash."""
        portfolio = _portfolio(cash=10_000.0)
        prices = {"AAPL": 100.0, "MSFT": 50.0, "GOOG": 200.0}
        max_shares = {"AAPL": 200.0, "MSFT": 200.0, "GOOG": 200.0}

        allowed = compute_allowed_actions(["AAPL", "MSFT", "GOOG"], prices, max_shares, portfolio)

        total_notional = sum(allowed[t].get("buy", 0) * prices[t] for t in ["AAPL", "MSFT", "GOOG"])
        assert total_notional <= 10_000.0 + 1e-9, f"Combined buy notional {total_notional:.2f} exceeds cash 10_000. Double-spend bug."

    def test_short_margin_not_double_spent(self):
        """
        equity=10_000, margin_req=0.5 → max short notional = 20_000.
        Two tickers each shorting up to 150 shares @ $100 = $15k each.
        Combined short margin must not exceed 20_000.
        """
        portfolio = _portfolio(cash=10_000.0, equity=10_000.0, margin_requirement=0.5, margin_used=0.0)
        prices = {"AAPL": 100.0, "MSFT": 100.0}
        max_shares = {"AAPL": 150.0, "MSFT": 150.0}

        allowed = compute_allowed_actions(["AAPL", "MSFT"], prices, max_shares, portfolio)

        aapl_short = allowed["AAPL"].get("short", 0)
        msft_short = allowed["MSFT"].get("short", 0)
        combined_margin = (aapl_short + msft_short) * 100.0 * 0.5
        assert combined_margin <= 10_000.0 + 1e-9, f"Combined margin {combined_margin:.2f} exceeds equity 10_000. Short double-spend bug."

    def test_short_capacity_with_existing_margin_used(self):
        """
        Regression for F1 units bug: (equity - margin_used) / margin_req, not (equity/margin_req) - margin_used.

        Setup: equity=10_000, margin_req=0.5, margin_used=5_000 (already shorted $10k notional).
        Correct remaining notional = (10_000 - 5_000) / 0.5 = 10_000.
        Buggy formula would give 10_000/0.5 - 5_000 = 15_000.
        """
        portfolio = {
            "cash": 15_000.0,  # cash inflated by short proceeds
            "equity": 10_000.0,
            "margin_requirement": 0.5,
            "margin_used": 5_000.0,  # $10k notional already shorted
            "positions": {},
        }
        prices = {"AAPL": 100.0}
        max_shares = {"AAPL": 200.0}  # high enough not to be the binding constraint

        allowed = compute_allowed_actions(["AAPL"], prices, max_shares, portfolio)

        aapl_short = allowed["AAPL"].get("short", 0)
        # Correct remaining notional = 10_000; max short shares = 10_000 / 100 = 100.
        # Buggy formula would allow 150 shares (notional 15_000).
        assert aapl_short <= 100.0 + 1e-9, (
            f"Short capacity should be ≤ 100 shares (notional 10_000 remaining) but got {aapl_short}. "
            "Formula bug: (equity - margin_used) / margin_req != equity/margin_req - margin_used."
        )

    def test_first_ticker_unaffected_when_cash_is_ample(self):
        """When cash comfortably covers the first ticker's max, its allocation is unchanged."""
        portfolio = _portfolio(cash=50_000.0)
        prices = {"AAPL": 100.0, "MSFT": 100.0}
        max_shares = {"AAPL": 100.0, "MSFT": 100.0}

        allowed = compute_allowed_actions(["AAPL", "MSFT"], prices, max_shares, portfolio)

        # AAPL's max_buy is 100 shares ($10k), cash =$50k → not restricted
        assert allowed["AAPL"].get("buy", 0) == pytest.approx(100.0)
        # MSFT also 100 shares; $40k remaining after AAPL → not restricted
        assert allowed["MSFT"].get("buy", 0) == pytest.approx(100.0)
