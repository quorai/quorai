"""Tests for intraday SOD equity capture protection."""

import json
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

_NY = ZoneInfo("America/New_York")


def test_save_before_market_open_succeeds(tmp_path):
    """Saving SOD equity before 09:30 ET writes the file without error."""
    import src.live.sod_equity as mod

    pre_open = datetime(2024, 1, 15, 8, 0, 0, tzinfo=_NY)

    with patch.object(mod, "datetime") as mock_dt:
        mock_dt.now.return_value = pre_open

        mod.save_sod_equity(100_000.0, log_dir=str(tmp_path))

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["equity"] == 100_000.0


def test_save_at_market_open_raises(tmp_path):
    """Saving SOD equity exactly at 09:30 ET raises RuntimeError."""
    import src.live.sod_equity as mod

    at_open = datetime(2024, 1, 15, 9, 30, 0, tzinfo=_NY)

    with patch.object(mod, "datetime") as mock_dt:
        mock_dt.now.return_value = at_open

        with pytest.raises(RuntimeError, match="intraday"):
            mod.save_sod_equity(98_000.0, log_dir=str(tmp_path))

    assert list(tmp_path.iterdir()) == [], "No file should be written on intraday capture"


def test_save_after_market_open_raises(tmp_path):
    """Saving SOD equity at 11:00 ET (post-open) raises RuntimeError."""
    import src.live.sod_equity as mod

    post_open = datetime(2024, 1, 15, 11, 0, 0, tzinfo=_NY)

    with patch.object(mod, "datetime") as mock_dt:
        mock_dt.now.return_value = post_open

        with pytest.raises(RuntimeError, match="intraday"):
            mod.save_sod_equity(95_000.0, log_dir=str(tmp_path))


def test_error_message_includes_manual_instructions(tmp_path):
    """RuntimeError message must tell operator how to manually set the SOD file."""
    import src.live.sod_equity as mod

    post_open = datetime(2024, 1, 15, 14, 0, 0, tzinfo=_NY)

    with patch.object(mod, "datetime") as mock_dt:
        mock_dt.now.return_value = post_open

        with pytest.raises(RuntimeError, match="manually"):
            mod.save_sod_equity(90_000.0, log_dir=str(tmp_path))
