import numpy as np
import pandas as pd

from src.regime import MarketRegime, classify_regime


def _make_spy(closes: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame({"close": closes}, index=idx)


def _last_date(df: pd.DataFrame) -> str:
    return str(df.index[-1].date())


VOL_LONG = 60


def test_bull_trend():
    # Steadily rising prices → close > SMA, low vol
    closes = list(np.linspace(100, 130, VOL_LONG + 10))
    spy = _make_spy(closes)
    result = classify_regime(spy, _last_date(spy))
    assert result == MarketRegime.BULL_TREND


def test_bear_trend():
    # 50 flat days then 20 days alternating -1.5%/+1% → close < SMA, short_vol > long_vol,
    # drawdown stays at ~-5% (above -8% threshold) so RISK_OFF is not triggered
    base = [100.0] * 50
    zigzag = []
    price = 100.0
    for i in range(20):
        price *= 0.985 if i % 2 == 0 else 1.01
        zigzag.append(price)
    closes = base + zigzag
    spy = _make_spy(closes)
    result = classify_regime(spy, _last_date(spy))
    assert result == MarketRegime.BEAR_TREND


def test_risk_off():
    # 50 gentle rising days then 20 days alternating -8%/+3% → deep drawdown + high short vol
    stable = [100.0 + i * 0.1 for i in range(50)]
    crash = []
    price = 105.0
    for i in range(20):
        price *= 0.92 if i % 2 == 0 else 1.03
        crash.append(price)
    closes = stable + crash
    spy = _make_spy(closes)
    result = classify_regime(spy, _last_date(spy))
    assert result == MarketRegime.RISK_OFF


def test_insufficient_data_returns_neutral():
    closes = list(np.linspace(100, 110, 30))  # fewer than vol_long_window=60
    spy = _make_spy(closes)
    result = classify_regime(spy, _last_date(spy))
    assert result == MarketRegime.NEUTRAL


def test_neutral_flat():
    # Completely flat prices → vol = 0 → NEUTRAL
    closes = [100.0] * (VOL_LONG + 5)
    spy = _make_spy(closes)
    result = classify_regime(spy, _last_date(spy))
    assert result == MarketRegime.NEUTRAL


def test_as_of_date_filters_future_data():
    closes = list(np.linspace(100, 130, VOL_LONG + 30))
    spy = _make_spy(closes)
    # Slice to first VOL_LONG + 10 rows — should still produce a valid regime
    cutoff = str(spy.index[VOL_LONG + 9].date())
    result = classify_regime(spy, cutoff)
    assert isinstance(result, MarketRegime)
