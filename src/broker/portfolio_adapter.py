import logging

from alpaca.trading.models import Position, TradeAccount

from src.backtesting.types import PortfolioSnapshot, PositionState, TickerRealizedGains

logger = logging.getLogger(__name__)


def to_snapshot(
    account: TradeAccount,
    positions: list[Position],
    tickers: list[str],
    margin_requirement: float = 0.0,
    prev_snapshot: PortfolioSnapshot | None = None,
) -> PortfolioSnapshot:
    """Convert Alpaca account + positions into a PortfolioSnapshot.

    Note: ``realized_gains`` is always zero-filled. The Alpaca positions API does
    not expose session-realized P&L. For closed-position attribution, consult the
    ``AuditJournal`` (``logs/trades-<date>.jsonl``).
    """
    """Convert Alpaca account + positions into a PortfolioSnapshot."""
    position_map: dict[str, PositionState] = {}
    total_margin_used = 0.0

    for pos in positions:
        ticker = pos.symbol
        qty = float(pos.qty)
        avg_entry = float(pos.avg_entry_price)

        if qty >= 0:
            position_map[ticker] = PositionState(
                long=qty,
                short=0.0,
                long_cost_basis=avg_entry,
                short_cost_basis=0.0,
                short_margin_used=0.0,
            )
        else:
            # Margin required = absolute market value × margin_requirement fraction
            short_market_value = abs(float(pos.market_value or 0))
            short_margin = short_market_value * margin_requirement
            total_margin_used += short_margin
            position_map[ticker] = PositionState(
                long=0.0,
                short=abs(qty),
                long_cost_basis=0.0,
                short_cost_basis=avg_entry,
                short_margin_used=short_margin,
            )

    # Zero-fill any tickers not present in Alpaca positions
    for ticker in tickers:
        if ticker not in position_map:
            position_map[ticker] = PositionState(
                long=0.0,
                short=0.0,
                long_cost_basis=0.0,
                short_cost_basis=0.0,
                short_margin_used=0.0,
            )

    realized_gains: dict[str, TickerRealizedGains] = {t: TickerRealizedGains(long=0.0, short=0.0) for t in tickers}

    if prev_snapshot is not None:
        prev_positions = prev_snapshot.get("positions", {})
        for tkr, prev_state in prev_positions.items():
            had_position = (prev_state.get("long", 0.0) or 0.0) > 0 or (prev_state.get("short", 0.0) or 0.0) > 0
            now_state = position_map.get(tkr)
            now_zero = now_state is None or (now_state["long"] == 0.0 and now_state["short"] == 0.0)
            if had_position and now_zero:
                logger.warning(
                    "[portfolio_adapter] %s position closed — realized P&L not available from Alpaca; consult AuditJournal (logs/trades-*.jsonl) for attribution",
                    tkr,
                )

    return PortfolioSnapshot(
        cash=float(account.cash or "0"),
        margin_used=total_margin_used,
        margin_requirement=margin_requirement,
        positions=position_map,
        realized_gains=realized_gains,
    )
