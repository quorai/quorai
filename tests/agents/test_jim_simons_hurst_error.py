"""Tests for R46: jim_simons Hurst calculation error returns score=0, not score=2."""

from unittest.mock import patch

import pandas as pd

from src.agents.jim_simons import compute_hurst_regime


def _prices_df(n: int = 50, value: float = 100.0) -> pd.DataFrame:
    """Return a constant-value price DataFrame (degenerate — triggers Hurst calc issues)."""
    return pd.DataFrame({"close": [value] * n})


class TestJimSimonsHurstError:
    def test_hurst_error_returns_score_zero(self):
        """
        When calculate_hurst_exponent raises, score must be 0 (not 2).
        Score=2 is a valid 'near random walk, moderate signal' reading; conflating
        calc failures with a valid neutral reading inflates the final tally.
        """
        with patch("src.agents.jim_simons.calculate_hurst_exponent", side_effect=RuntimeError("test")):
            result = compute_hurst_regime(_prices_df())

        assert result["score"] == 0, f"Hurst calc error must score 0, got {result['score']}"

    def test_hurst_error_details_mention_error(self):
        """Error path must indicate an error, not claim 'neutral'."""
        with patch("src.agents.jim_simons.calculate_hurst_exponent", side_effect=ValueError("boom")):
            result = compute_hurst_regime(_prices_df())

        assert "error" in result["details"].lower(), f"Details must mention error. Got: {result['details']}"

    def test_hurst_neutral_range_still_scores_two(self):
        """Normal Hurst in [0.45, 0.55) scores 2 — the error fix must not change the valid path."""
        with patch("src.agents.jim_simons.calculate_hurst_exponent", return_value=0.50):
            result = compute_hurst_regime(_prices_df())

        assert result["score"] == 2, f"Valid neutral Hurst should score 2, got {result['score']}"
