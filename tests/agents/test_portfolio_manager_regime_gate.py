"""Tests for the regime-aware action gate in compute_allowed_actions."""

from src.agents.portfolio_manager import compute_allowed_actions


def _portfolio(cash: float = 50_000.0, equity: float = 50_000.0) -> dict:
    return {
        "cash": cash,
        "positions": {},
        "margin_requirement": 0.5,
        "margin_used": 0.0,
        "equity": equity,
    }


def _portfolio_with_short(ticker: str, shares: float, cash: float = 50_000.0, equity: float = 50_000.0) -> dict:
    return {
        "cash": cash,
        "positions": {ticker: {"long": 0.0, "short": shares, "long_cost_basis": 0.0, "short_cost_basis": 100.0}},
        "margin_requirement": 0.5,
        "margin_used": 0.0,
        "equity": equity,
    }


def _portfolio_with_long(ticker: str, shares: float, cash: float = 10_000.0, equity: float = 50_000.0) -> dict:
    return {
        "cash": cash,
        "positions": {ticker: {"long": shares, "short": 0.0, "long_cost_basis": 100.0, "short_cost_basis": 0.0}},
        "margin_requirement": 0.5,
        "margin_used": 0.0,
        "equity": equity,
    }


def _group_signals(ticker: str, quant: str = "neutral", growth: str = "neutral", quality: str = "neutral") -> dict:
    return {
        ticker: {
            "quant_systematic": {"signal": quant, "confidence": 70.0, "dissent": 0},
            "growth_and_catalyst": {"signal": growth, "confidence": 60.0, "dissent": 0},
            "quality_compounders": {"signal": quality, "confidence": 65.0, "dissent": 0},
            "deep_value": {"signal": "bearish", "confidence": 80.0, "dissent": 0},
            "macro_and_cycle": {"signal": "bearish", "confidence": 70.0, "dissent": 0},
        }
    }


class TestBullTrendGate:
    def test_short_blocked_when_quant_bullish(self):
        gs = _group_signals("AAPL", quant="bullish", growth="neutral")
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio(),
            regime="bull_trend",
            group_signals=gs,
        )
        assert "short" not in result["AAPL"]
        assert "buy" in result["AAPL"]

    def test_short_blocked_when_growth_bullish(self):
        gs = _group_signals("AAPL", quant="neutral", growth="bullish")
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio(),
            regime="bull_trend",
            group_signals=gs,
        )
        assert "short" not in result["AAPL"]

    def test_short_allowed_when_both_quant_and_growth_bearish(self):
        gs = _group_signals("AAPL", quant="bearish", growth="bearish")
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio(),
            regime="bull_trend",
            group_signals=gs,
        )
        assert "short" in result["AAPL"]

    def test_cover_always_allowed_in_bull_trend(self):
        gs = _group_signals("AAPL", quant="bullish", growth="bullish")
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio_with_short("AAPL", shares=5.0),
            regime="bull_trend",
            group_signals=gs,
        )
        assert "cover" in result["AAPL"]
        assert "short" not in result["AAPL"]

    def test_buy_never_blocked_in_bull_trend(self):
        gs = _group_signals("AAPL", quant="bullish", growth="bullish")
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio(),
            regime="bull_trend",
            group_signals=gs,
        )
        assert "buy" in result["AAPL"]


class TestBearTrendGate:
    def test_buy_blocked_when_quant_bearish(self):
        gs = _group_signals("MSFT", quant="bearish", quality="neutral")
        result = compute_allowed_actions(
            tickers=["MSFT"],
            current_prices={"MSFT": 100.0},
            max_shares={"MSFT": 10.0},
            portfolio=_portfolio(),
            regime="bear_trend",
            group_signals=gs,
        )
        assert "buy" not in result["MSFT"]
        assert "short" in result["MSFT"]

    def test_buy_blocked_when_quality_bearish(self):
        gs = _group_signals("MSFT", quant="neutral", quality="bearish")
        result = compute_allowed_actions(
            tickers=["MSFT"],
            current_prices={"MSFT": 100.0},
            max_shares={"MSFT": 10.0},
            portfolio=_portfolio(),
            regime="bear_trend",
            group_signals=gs,
        )
        assert "buy" not in result["MSFT"]

    def test_buy_allowed_when_both_quant_and_quality_bullish(self):
        gs = _group_signals("MSFT", quant="bullish", quality="bullish")
        result = compute_allowed_actions(
            tickers=["MSFT"],
            current_prices={"MSFT": 100.0},
            max_shares={"MSFT": 10.0},
            portfolio=_portfolio(),
            regime="bear_trend",
            group_signals=gs,
        )
        assert "buy" in result["MSFT"]

    def test_sell_always_allowed_in_bear_trend(self):
        gs = _group_signals("MSFT", quant="bearish", quality="bearish")
        result = compute_allowed_actions(
            tickers=["MSFT"],
            current_prices={"MSFT": 100.0},
            max_shares={"MSFT": 10.0},
            portfolio=_portfolio_with_long("MSFT", shares=5.0),
            regime="bear_trend",
            group_signals=gs,
        )
        assert "sell" in result["MSFT"]
        assert "buy" not in result["MSFT"]


class TestRiskOffGate:
    def test_buy_and_short_both_blocked(self):
        gs = _group_signals("SPY", quant="bullish", growth="bullish")
        result = compute_allowed_actions(
            tickers=["SPY"],
            current_prices={"SPY": 400.0},
            max_shares={"SPY": 10.0},
            portfolio=_portfolio(),
            regime="risk_off",
            group_signals=gs,
        )
        assert "buy" not in result["SPY"]
        assert "short" not in result["SPY"]

    def test_cover_and_sell_allowed_in_risk_off(self):
        gs = _group_signals("SPY")
        result = compute_allowed_actions(
            tickers=["SPY"],
            current_prices={"SPY": 400.0},
            max_shares={"SPY": 10.0},
            portfolio=_portfolio_with_short("SPY", shares=3.0),
            regime="risk_off",
            group_signals=gs,
        )
        assert "cover" in result["SPY"]
        assert "short" not in result["SPY"]
        assert "buy" not in result["SPY"]


class TestNoRegimeGate:
    def test_none_regime_no_filtering(self):
        """regime=None preserves legacy behavior — all capacity-valid actions available."""
        gs = _group_signals("AAPL", quant="bullish", growth="bullish")
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio(),
            regime=None,
            group_signals=gs,
        )
        assert "buy" in result["AAPL"]
        assert "short" in result["AAPL"]

    def test_neutral_regime_no_filtering(self):
        gs = _group_signals("AAPL", quant="bullish")
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio(),
            regime="neutral",
            group_signals=gs,
        )
        assert "short" in result["AAPL"]

    def test_no_group_signals_no_filtering(self):
        """group_signals=None disables gate even with regime set."""
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio(),
            regime="bull_trend",
            group_signals=None,
        )
        assert "short" in result["AAPL"]

    def test_missing_ticker_in_group_signals_no_error(self):
        """Ticker missing from group_signals dict → gate skipped gracefully."""
        result = compute_allowed_actions(
            tickers=["NVDA"],
            current_prices={"NVDA": 500.0},
            max_shares={"NVDA": 5.0},
            portfolio=_portfolio(),
            regime="bull_trend",
            group_signals={"OTHER": {}},
        )
        assert "short" in result["NVDA"]


def _bearish_panel(ticker: str, n_bear: int, n_neut: int, n_bull: int = 0) -> dict:
    """Build a panel_stats dict with given directional counts and computed tilt."""
    dir_conf = n_bear * 60.0  # assume 60 confidence each
    bull_conf = n_bull * 60.0
    total_dir = dir_conf + bull_conf
    tilt = (bull_conf - dir_conf) / total_dir if total_dir > 0 else 0.0
    return {
        ticker: {
            "bullish": n_bull,
            "bearish": n_bear,
            "neutral": n_neut,
            "tilt": round(tilt, 3),
        }
    }


class TestPanelTiltGate:
    def test_buy_blocked_when_panel_net_bearish_and_participation_met(self):
        """Strong bearish tilt with ≥40% directional → buy blocked."""
        # 12 bear, 0 bull, 13 neutral → 48% directional, tilt = -1.0
        ps = _bearish_panel("AAPL", n_bear=12, n_neut=13)
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio(),
            panel_stats=ps,
        )
        assert "buy" not in result["AAPL"]

    def test_buy_allowed_when_participation_below_floor(self):
        """Panel is net-bearish but <40% analysts are directional → gate does NOT fire."""
        # 5 bear, 0 bull, 20 neutral → only 20% directional, tilt = -1.0
        ps = _bearish_panel("AAPL", n_bear=5, n_neut=20)
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio(),
            panel_stats=ps,
        )
        # Gate should NOT fire — mostly neutral panel → buy allowed
        assert "buy" in result["AAPL"]

    def test_sell_always_allowed_despite_bearish_panel(self):
        """sell is never blocked by the panel-tilt gate."""
        ps = _bearish_panel("AAPL", n_bear=15, n_neut=10)
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio_with_long("AAPL", shares=5.0),
            panel_stats=ps,
        )
        assert "sell" in result["AAPL"]

    def test_gate_disarmed_when_panel_stats_none(self):
        """panel_stats=None disables the gate entirely."""
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio(),
            panel_stats=None,
        )
        assert "buy" in result["AAPL"]


class TestMinHoldGate:
    def test_sell_blocked_within_hold_window(self):
        """sell blocked if the last _MIN_HOLD_CYCLES trades include a buy."""
        recent = {"AAPL": [{"action": "buy", "date": "2026-04-08", "quantity": 10}]}
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio_with_long("AAPL", shares=10.0),
            recent_trades=recent,
        )
        assert "sell" not in result["AAPL"]

    def test_sell_allowed_after_hold_window(self):
        """sell allowed once _MIN_HOLD_CYCLES have elapsed (buy is older than the window)."""
        from src.agents.portfolio_manager import _MIN_HOLD_CYCLES

        # The buy happened long ago; the last _MIN_HOLD_CYCLES trades are all holds.
        old_buy = {"action": "buy", "date": "2026-04-01", "quantity": 10}
        hold = {"action": "hold", "date": "2026-04-09", "quantity": 0}
        recent = {"AAPL": [old_buy] + [hold] * _MIN_HOLD_CYCLES}
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio_with_long("AAPL", shares=10.0),
            recent_trades=recent,
        )
        assert "sell" in result["AAPL"]

    def test_cover_blocked_within_hold_window_after_short(self):
        """cover blocked if a recent short was opened."""
        recent = {"AAPL": [{"action": "short", "date": "2026-04-08", "quantity": 5}]}
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            max_short_shares={"AAPL": 0.0},
            portfolio=_portfolio_with_short("AAPL", shares=5.0),
            recent_trades=recent,
        )
        assert "cover" not in result["AAPL"]

    def test_no_recent_trades_no_block(self):
        """Empty recent_trades does not block any action."""
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio_with_long("AAPL", shares=10.0),
            recent_trades={},
        )
        assert "sell" in result["AAPL"]

    def test_recent_trades_none_no_block(self):
        """recent_trades=None disables the gate entirely."""
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio_with_long("AAPL", shares=10.0),
            recent_trades=None,
        )
        assert "sell" in result["AAPL"]
