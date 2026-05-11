from alpaca.trading.models import Position, TradeAccount

from src.backtesting.types import PortfolioSnapshot, PositionState, TickerRealizedGains


def to_snapshot(
    account: TradeAccount,
    positions: list[Position],
    tickers: list[str],
    margin_requirement: float = 0.0,
) -> PortfolioSnapshot:
    """Convert Alpaca account + positions into a PortfolioSnapshot."""
    position_map: dict[str, PositionState] = {}

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
            position_map[ticker] = PositionState(
                long=0.0,
                short=abs(qty),
                long_cost_basis=0.0,
                short_cost_basis=avg_entry,
                short_margin_used=0.0,
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

    return PortfolioSnapshot(
        cash=float(account.cash or "0"),
        margin_used=0.0,
        margin_requirement=margin_requirement,
        positions=position_map,
        realized_gains=realized_gains,
    )
