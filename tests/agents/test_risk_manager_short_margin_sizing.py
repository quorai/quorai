"""Tests for R41: risk_manager exposes a separate short-position limit sized by margin."""

from src.agents.risk_manager import risk_management_agent


def _state(cash=10_000.0, equity=10_000.0, margin_requirement=0.5, margin_used=0.0, prices=None):
    """Build a minimal AgentState for the risk_management_agent."""
    tickers = ["AAPL"]
    if prices is None:
        prices = {"AAPL": 100.0}
    return {
        "messages": [],
        "data": {
            "tickers": tickers,
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "portfolio": {
                "cash": cash,
                "equity": equity,
                "margin_requirement": margin_requirement,
                "margin_used": margin_used,
                "positions": {},
            },
            "analyst_signals": {},
            "prices": {t: [{"time": "2024-01-01T00:00:00", "open": p, "close": p, "high": p, "low": p, "volume": 1000}] for t, p in prices.items()},
        },
        "metadata": {
            "show_reasoning": False,
        },
    }


class TestRiskManagerShortMarginSizing:
    def test_max_short_position_size_uses_margin_not_cash(self, monkeypatch):
        """
        With cash=0 (fully invested) but 50% margin and no shorts yet,
        max_short_position_size should be > 0 (margin capacity exists).
        Before the fix, remaining_position_limit was capped by cash=0, so max_short=0.
        """
        import pandas as pd

        # Inject a price series long enough for volatility calc
        price_rows = [{"time": f"2024-{m:02d}-01T00:00:00", "open": 100.0, "close": 100.0, "high": 101.0, "low": 99.0, "volume": 1000} for m in range(1, 7)]

        monkeypatch.setattr("src.agents.risk_manager.get_prices", lambda **kw: price_rows)

        def fake_prices_to_df(prices):
            return pd.DataFrame([{"close": p["close"], "open": p["open"], "high": p["high"], "low": p["low"], "volume": p["volume"]} for p in prices])

        monkeypatch.setattr("src.agents.risk_manager.prices_to_df", fake_prices_to_df)

        state = _state(cash=0.0, equity=10_000.0, margin_requirement=0.5, margin_used=0.0)
        state["data"]["analyst_signals"] = {}

        result = risk_management_agent(state)
        signals = result["data"]["analyst_signals"]["risk_management_agent"]

        aapl = signals["AAPL"]
        assert "max_short_position_size" in aapl, "risk_analysis must include max_short_position_size"
        max_short = aapl["max_short_position_size"]
        assert max_short > 0, f"With cash=0 and equity=10_000 at 50% margin, max_short should be > 0 (margin capacity = 20_000). Got {max_short}."

    def test_long_limit_still_capped_by_cash(self, monkeypatch):
        """remaining_position_limit for longs is still bounded by available cash."""
        price_rows = [{"time": f"2024-{m:02d}-01T00:00:00", "open": 100.0, "close": 100.0, "high": 101.0, "low": 99.0, "volume": 1000} for m in range(1, 7)]
        monkeypatch.setattr("src.agents.risk_manager.get_prices", lambda **kw: price_rows)

        def fake_prices_to_df(prices):
            import pandas as pd

            return pd.DataFrame([{"close": p["close"], "open": p["open"], "high": p["high"], "low": p["low"], "volume": p["volume"]} for p in prices])

        monkeypatch.setattr("src.agents.risk_manager.prices_to_df", fake_prices_to_df)

        state = _state(cash=500.0, equity=10_000.0, margin_requirement=0.5, margin_used=0.0)
        result = risk_management_agent(state)
        signals = result["data"]["analyst_signals"]["risk_management_agent"]
        aapl_long_limit = signals["AAPL"]["remaining_position_limit"]
        assert aapl_long_limit <= 500.0, f"Long limit should be capped by cash=500, got {aapl_long_limit}"
