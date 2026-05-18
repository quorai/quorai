from __future__ import annotations

from enum import Enum
import math

import pandas as pd


class MarketRegime(str, Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"


def classify_regime(
    spy_df: pd.DataFrame,
    as_of_date: str,
    sma_window: int = 20,
    vol_window: int = 20,
    vol_long_window: int = 60,
    drawdown_threshold: float = -0.08,
    high_vol_multiplier: float = 1.5,
) -> MarketRegime:
    """Classify the market regime as of `as_of_date` using SPY price history.

    Rules (applied in order):
      RISK_OFF   — current drawdown from rolling peak > drawdown_threshold
                   AND short-term vol > high_vol_multiplier × long-run vol
      BULL_TREND — close > SMA(sma_window) AND short-term vol ≤ high_vol_multiplier × long-run vol
      BEAR_TREND — close < SMA(sma_window) AND short-term vol > 1.0× long-run vol
      NEUTRAL    — otherwise
    """
    regime, _ = classify_regime_with_indicators(
        spy_df,
        as_of_date,
        sma_window=sma_window,
        vol_window=vol_window,
        vol_long_window=vol_long_window,
        drawdown_threshold=drawdown_threshold,
        high_vol_multiplier=high_vol_multiplier,
    )
    return regime


def classify_regime_with_indicators(
    spy_df: pd.DataFrame,
    as_of_date: str,
    sma_window: int = 20,
    vol_window: int = 20,
    vol_long_window: int = 60,
    drawdown_threshold: float = -0.08,
    high_vol_multiplier: float = 1.5,
) -> tuple[MarketRegime, dict]:
    """Like classify_regime but also returns the raw indicator values used for the classification."""
    df = spy_df[spy_df.index <= pd.Timestamp(as_of_date)].copy()
    empty_indicators: dict = {"as_of": as_of_date}
    if len(df) < vol_long_window:
        return MarketRegime.NEUTRAL, empty_indicators

    closes = df["close"]
    if closes.isna().any():
        return MarketRegime.NEUTRAL, empty_indicators

    current = float(closes.iloc[-1])
    sma = float(closes.rolling(sma_window).mean().iloc[-1])
    short_vol = float(closes.pct_change().rolling(vol_window).std().iloc[-1])
    long_vol = float(closes.pct_change().rolling(vol_long_window).std().iloc[-1])
    peak = float(closes.rolling(vol_long_window).max().iloc[-1])
    drawdown = (current - peak) / peak if peak > 0 else 0.0

    if any(math.isnan(x) for x in (sma, short_vol, long_vol, current)):
        return MarketRegime.NEUTRAL, empty_indicators

    if long_vol == 0:
        return MarketRegime.NEUTRAL, empty_indicators

    vol_ratio = short_vol / long_vol

    indicators = {
        "as_of": as_of_date,
        "close": current,
        "sma_20": sma,
        "short_vol_20": short_vol,
        "long_vol_60": long_vol,
        "vol_ratio": vol_ratio,
        "drawdown": drawdown,
        "peak_60": peak,
    }

    if drawdown < drawdown_threshold and vol_ratio > high_vol_multiplier:
        return MarketRegime.RISK_OFF, indicators
    if current > sma and vol_ratio <= high_vol_multiplier:
        return MarketRegime.BULL_TREND, indicators
    if current < sma and vol_ratio > 1.0:
        return MarketRegime.BEAR_TREND, indicators
    return MarketRegime.NEUTRAL, indicators
