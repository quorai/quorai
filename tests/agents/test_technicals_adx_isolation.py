"""Tests for R55: calculate_adx must not mutate the caller's DataFrame."""

import numpy as np
import pandas as pd

from src.agents.technicals import calculate_adx


def _make_ohlcv(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    closes = 100.0 + rng.standard_normal(n).cumsum()
    highs = closes + rng.uniform(0, 1, n)
    lows = closes - rng.uniform(0, 1, n)
    return pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes, "volume": 1_000_000})


class TestTechnicalsAdxIsolation:
    def test_calculate_adx_does_not_add_columns_to_original_df(self):
        """
        R55: calculate_adx added 'high_low', 'tr', '+di', 'adx', etc. directly to the
        passed DataFrame. After the fix (df = df.copy()), the original must be unchanged.
        """
        df = _make_ohlcv()
        cols_before = set(df.columns)

        calculate_adx(df)

        cols_after = set(df.columns)
        leaked = cols_after - cols_before
        assert not leaked, f"calculate_adx leaked columns into the caller's DataFrame: {sorted(leaked)}"

    def test_calculate_adx_returns_df_with_adx_column(self):
        """The returned DataFrame must contain the 'adx' column."""
        df = _make_ohlcv()
        result = calculate_adx(df)
        assert "adx" in result.columns, "calculate_adx must return a DataFrame with 'adx' column"

    def test_original_df_values_unchanged_after_call(self):
        """Verify that the close values in the original df are not modified."""
        df = _make_ohlcv()
        closes_before = df["close"].copy()
        calculate_adx(df)
        pd.testing.assert_series_equal(df["close"], closes_before)
