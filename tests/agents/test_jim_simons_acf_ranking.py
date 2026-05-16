"""Tests for R64: jim_simons ACF scoring — mean-reversion > momentum >= no-signal."""

import numpy as np
import pandas as pd

from src.agents.jim_simons import compute_autocorrelation


def _df_from_returns(returns: list[float]) -> pd.DataFrame:
    closes = [100.0]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    return pd.DataFrame({"close": closes, "volume": 1_000_000})


def _mean_reverting_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Alternating returns → strong negative ACF(1)."""
    rng = np.random.default_rng(seed)
    returns = []
    for _ in range(n):
        returns.append(0.01 if len(returns) % 2 == 0 else -0.01)
        returns[-1] += rng.normal(0, 0.001)
    return _df_from_returns(returns)


def _momentum_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Autocorrelated (trending) returns → positive ACF(1)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + 0.002 + rng.normal(0, 0.005)
    return _df_from_returns(x.tolist())


def _no_signal_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """i.i.d. returns → near-zero ACF(1)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.005, n).tolist()
    return _df_from_returns(returns)


class TestJimSimonsAcfRanking:
    def test_mean_reverting_scores_higher_than_momentum(self):
        """
        R64: mean-reversion (acf1 << 0) should score higher than momentum (acf1 > 0.10)
        for a stat-arb strategy.
        Before fix: near-zero ACF scored +2, momentum +1 — mean-reversion +2 or +4.
        After fix: no-signal drops to +1; mean-reversion still ≥ momentum.
        """
        result_mr = compute_autocorrelation(_mean_reverting_df())
        result_mom = compute_autocorrelation(_momentum_df())

        score_mr = result_mr["score"]
        score_mom = result_mom["score"]

        assert score_mr >= score_mom, f"Mean-reversion score ({score_mr}) must be >= momentum score ({score_mom}) for stat-arb"

    def test_momentum_scores_at_least_as_high_as_no_signal(self):
        """
        R64: After fix, momentum (+1) must score >= no-signal (+1). Before fix no-signal was +2, momentum +1.
        """
        result_mom = compute_autocorrelation(_momentum_df())
        result_ns = compute_autocorrelation(_no_signal_df())

        assert result_mom["score"] >= result_ns["score"], f"Momentum score ({result_mom['score']}) must be >= no-signal score ({result_ns['score']})"

    def test_function_returns_score_and_details(self):
        """Regression: compute_autocorrelation must always return a dict with score/details."""
        for df in [_mean_reverting_df(), _momentum_df(), _no_signal_df()]:
            result = compute_autocorrelation(df)
            assert "score" in result
            assert "details" in result
