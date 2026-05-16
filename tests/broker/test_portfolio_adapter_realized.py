"""Tests for portfolio_adapter realized_gains limitation documentation and warning."""

import logging
from unittest.mock import MagicMock

from src.backtesting.types import PortfolioSnapshot, PositionState, TickerRealizedGains
from src.broker.portfolio_adapter import to_snapshot


def _make_account(equity: str = "100000", cash: str = "50000") -> MagicMock:
    account = MagicMock()
    account.equity = equity
    account.cash = cash
    return account


def test_realized_gains_zeroed_for_all_tickers():
    """realized_gains is always zero-filled — Alpaca API does not expose session P&L."""
    snapshot = to_snapshot(_make_account(), [], ["AAPL", "MSFT"])
    assert snapshot["realized_gains"]["AAPL"]["long"] == 0.0
    assert snapshot["realized_gains"]["AAPL"]["short"] == 0.0
    assert snapshot["realized_gains"]["MSFT"]["long"] == 0.0


def test_docstring_documents_limitation():
    """to_snapshot docstring must mention the zero-fill and point to AuditJournal."""
    doc = to_snapshot.__doc__ or ""
    assert "realized" in doc.lower()
    assert "AuditJournal" in doc


def test_warning_logged_when_position_closes(caplog):
    """When a ticker had a position last cycle but now shows zero, a WARNING is logged."""
    prev: PortfolioSnapshot = {
        "cash": 50000.0,
        "margin_used": 0.0,
        "margin_requirement": 0.0,
        "positions": {
            "AAPL": PositionState(
                long=10.0,
                short=0.0,
                long_cost_basis=150.0,
                short_cost_basis=0.0,
                short_margin_used=0.0,
            ),
        },
        "realized_gains": {"AAPL": TickerRealizedGains(long=0.0, short=0.0)},
    }

    with caplog.at_level(logging.WARNING, logger="src.broker.portfolio_adapter"):
        to_snapshot(_make_account(), positions=[], tickers=["AAPL"], prev_snapshot=prev)

    assert any("AAPL" in r.message and "realized" in r.message.lower() for r in caplog.records), "Expected a WARNING mentioning AAPL and realized P&L"


def test_no_warning_when_position_unchanged(caplog):
    """No WARNING if the ticker still has a position in the new snapshot."""

    prev: PortfolioSnapshot = {
        "cash": 50000.0,
        "margin_used": 0.0,
        "margin_requirement": 0.0,
        "positions": {
            "AAPL": PositionState(
                long=5.0,
                short=0.0,
                long_cost_basis=150.0,
                short_cost_basis=0.0,
                short_margin_used=0.0,
            ),
        },
        "realized_gains": {"AAPL": TickerRealizedGains(long=0.0, short=0.0)},
    }

    pos = MagicMock()
    pos.symbol = "AAPL"
    pos.qty = "5"
    pos.avg_entry_price = "150.0"
    pos.market_value = "750.0"

    with caplog.at_level(logging.WARNING, logger="src.broker.portfolio_adapter"):
        to_snapshot(_make_account(), positions=[pos], tickers=["AAPL"], prev_snapshot=prev)

    assert not any("AAPL" in r.message and "realized" in r.message.lower() for r in caplog.records)
