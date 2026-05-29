from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Trading-cost parameters for the backtest fill path.

    All three components default to **zero** so the existing test suite, which
    asserts exact cash values, continues to pass without modification.

    Recommended research defaults for liquid large-cap US equities:
        slippage_bps=5, commission_bps=2, borrow_bps_annual=50

    Fields:
        slippage_bps: Per-side price impact in basis points.  Buys and covers
            fill at ``price * (1 + slippage_bps/1e4)``; sells and short-opens
            fill at ``price * (1 - slippage_bps/1e4)``.
        commission_bps: Per-trade commission as basis points of trade notional.
            Charged as a separate cash debit so cost-basis stays at the
            slippage-adjusted execution price and realized-gain accounting
            remains interpretable.
        borrow_bps_annual: Annualised short-borrow cost in basis points, accrued
            daily (÷ 252) on open short notional.  Call
            ``Portfolio.accrue_borrow_cost()`` once per trading day.
    """

    slippage_bps: float = 0.0
    commission_bps: float = 0.0
    borrow_bps_annual: float = 0.0

    @classmethod
    def zero(cls) -> "CostModel":
        """Return the zero-cost model (no friction).  Identical to ``CostModel()``."""
        return cls()

    @classmethod
    def from_args(
        cls,
        slippage_bps: float = 0.0,
        commission_bps: float = 0.0,
        borrow_bps_annual: float = 0.0,
    ) -> "CostModel":
        """Construct from explicit parameter values."""
        return cls(
            slippage_bps=float(slippage_bps),
            commission_bps=float(commission_bps),
            borrow_bps_annual=float(borrow_bps_annual),
        )
