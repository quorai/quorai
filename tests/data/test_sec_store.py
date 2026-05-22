"""Tests for SecStore: point-in-time gating, TTM aggregation, seeded/unseeded distinction."""

from pathlib import Path

import pytest

from src.data.sec_store import FactRow, SecStore


def _make_store(tmp_path: Path) -> SecStore:
    db = tmp_path / "sec_fundamentals_test.db"
    return SecStore(db_path=db)


def _seed_aapl(store: SecStore, rows: list[FactRow]) -> None:
    store.upsert_facts("AAPL", "0000320193", "Apple Inc.", rows)


def _row(fact_key: str, period: str, period_end: str, filed: str, value: float, section: str = "income") -> FactRow:
    return FactRow(
        ticker="AAPL",
        section=section,
        fact_key=fact_key,
        period=period,
        period_end=period_end,
        filed=filed,
        value=value,
    )


class TestIsSeeded:
    def test_unseeded_ticker_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_statements("AAPL", "quarterly", "2023-06-30", 4) is None

    def test_seeded_ticker_returns_list(self, tmp_path):
        store = _make_store(tmp_path)
        _seed_aapl(store, [_row("revenue", "quarterly", "2022-12-31", "2023-02-01", 1e9)])
        result = store.get_statements("AAPL", "quarterly", "2023-03-01", 4)
        assert isinstance(result, list)


class TestPointInTimeGating:
    def test_unfiled_period_excluded(self, tmp_path):
        """A period filed AFTER end_date must not appear."""
        store = _make_store(tmp_path)
        _seed_aapl(
            store,
            [
                _row("revenue", "quarterly", "2022-12-31", "2023-02-01", 1_000.0),  # filed before end_date
                _row("revenue", "quarterly", "2023-03-31", "2023-04-30", 2_000.0),  # filed after end_date
            ],
        )
        result = store.get_statements("AAPL", "quarterly", "2023-03-01", 5)
        assert result is not None
        assert len(result) == 1
        assert result[0].period_end == "2022-12-31"

    def test_both_periods_visible_after_filing(self, tmp_path):
        store = _make_store(tmp_path)
        _seed_aapl(
            store,
            [
                _row("revenue", "quarterly", "2022-12-31", "2023-02-01", 1_000.0),
                _row("revenue", "quarterly", "2023-03-31", "2023-04-30", 2_000.0),
            ],
        )
        result = store.get_statements("AAPL", "quarterly", "2023-05-01", 5)
        assert result is not None
        assert len(result) == 2
        assert result[0].period_end == "2023-03-31"  # newest first

    def test_limit_respected(self, tmp_path):
        store = _make_store(tmp_path)
        rows = [_row("revenue", "quarterly", f"202{y}-{m:02d}-30", "2023-06-01", float(y * 100 + m)) for y, m in [(1, 3), (1, 6), (1, 9), (1, 12), (2, 3)]]
        _seed_aapl(store, rows)
        result = store.get_statements("AAPL", "quarterly", "2023-06-01", 3)
        assert result is not None
        assert len(result) == 3


class TestTTMAggregation:
    def _seed_four_quarters(self, store: SecStore) -> None:
        quarters = [
            ("2022-03-31", "2022-04-29", 10_000.0),
            ("2022-06-30", "2022-07-29", 11_000.0),
            ("2022-09-30", "2022-10-28", 12_000.0),
            ("2022-12-31", "2023-01-27", 13_000.0),
        ]
        rows = []
        for pe, filed, rev in quarters:
            rows.append(_row("revenue", "quarterly", pe, filed, rev))
            rows.append(_row("net_income_loss_attributable_common_shareholders", "quarterly", pe, filed, rev * 0.2))
            # Balance sheet rows (section='balance')
            rows.append(_row("total_assets", "quarterly", pe, filed, rev * 10, section="balance"))
        _seed_aapl(store, rows)

    def test_ttm_sums_four_quarters(self, tmp_path):
        store = _make_store(tmp_path)
        self._seed_four_quarters(store)
        result = store.get_statements("AAPL", "ttm", "2023-02-01", 3)
        assert result is not None
        assert len(result) >= 1
        ttm = result[0]
        # TTM revenue = 10k + 11k + 12k + 13k = 46k
        assert ttm.income.get("revenue") == pytest.approx(46_000.0)

    def test_ttm_uses_most_recent_balance(self, tmp_path):
        store = _make_store(tmp_path)
        self._seed_four_quarters(store)
        result = store.get_statements("AAPL", "ttm", "2023-02-01", 3)
        assert result is not None
        # Balance should be from the most recent quarter (2022-12-31: total_assets = 13k * 10 = 130k)
        ttm = result[0]
        assert ttm.balance.get("total_assets") == pytest.approx(130_000.0)

    def test_ttm_fewer_than_4_quarters_returns_partial(self, tmp_path):
        store = _make_store(tmp_path)
        # Only 2 quarters available
        rows = [
            _row("revenue", "quarterly", "2022-09-30", "2022-10-28", 12_000.0),
            _row("revenue", "quarterly", "2022-12-31", "2023-01-27", 13_000.0),
        ]
        _seed_aapl(store, rows)
        result = store.get_statements("AAPL", "ttm", "2023-02-01", 3)
        # Returns partial result rather than None (caller decides whether to use it)
        assert result is not None


class TestSharesOutstanding:
    def test_returns_most_recent_filed_before_end_date(self, tmp_path):
        store = _make_store(tmp_path)
        rows = [
            FactRow(ticker="AAPL", section="dei", fact_key="shares_outstanding", period="quarterly", period_end="2022-12-31", filed="2023-01-27", value=15_943_425_000.0),
            FactRow(ticker="AAPL", section="dei", fact_key="shares_outstanding", period="quarterly", period_end="2023-03-31", filed="2023-04-28", value=15_788_000_000.0),
        ]
        _seed_aapl(store, rows)
        # Only the first entry is filed before 2023-04-01
        shares = store.get_shares_outstanding("AAPL", "2023-04-01")
        assert shares == pytest.approx(15_943_425_000.0)

    def test_unseeded_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_shares_outstanding("NVDA", "2023-01-01") is None
