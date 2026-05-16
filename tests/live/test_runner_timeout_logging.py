"""Tests for R39: runner emits a warning log on reconciler timeout."""

import logging
from unittest.mock import MagicMock, patch


class TestRunnerTimeoutLogging:
    def test_timeout_status_emits_warning(self, caplog):
        """
        When reconciler returns status='timeout', runner must emit a WARNING
        and not raise. The result string should still record the status.
        """
        from src.live.runner import LiveRunner

        runner = LiveRunner(
            tickers=["AAPL"],
            model_name="test",
            model_provider="test",
            selected_analysts=None,
            dry_run=True,
        )
        # Inject a fake broker and fake executor so no real network calls happen.
        fake_broker = MagicMock()
        runner._broker = fake_broker

        fake_executor = MagicMock()
        fake_executor.submitted_orders = {"AAPL": "ord-123"}
        fake_executor.execute_decisions.return_value = {}

        timeout_fills = {
            "ord-123": {
                "status": "timeout",
                "filled_qty": 2.5,
                "filled_avg_price": None,
                "ticker": "AAPL",
            }
        }

        fake_reconciler = MagicMock()
        fake_reconciler.reconcile.return_value = timeout_fills

        with (
            patch("src.live.runner.LiveExecutor", return_value=fake_executor),
            patch("src.live.reconciler.Reconciler", return_value=fake_reconciler),
            caplog.at_level(logging.WARNING, logger="src.live.runner"),
        ):
            runner.dry_run = False
            results = runner.execute({"AAPL": {"action": "buy", "quantity": 10}})

        # Must not crash
        assert "AAPL" in results
        assert "timeout" in results["AAPL"]

        # Must emit a WARNING mentioning timeout and manual review
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("manual review" in t.lower() for t in warning_texts), f"Expected a warning mentioning 'manual review' for reconciler timeout. Got: {warning_texts}"

    def test_filled_status_does_not_emit_timeout_warning(self, caplog):
        """Successful fills must not emit spurious timeout warnings."""
        from src.live.runner import LiveRunner

        runner = LiveRunner(
            tickers=["AAPL"],
            model_name="test",
            model_provider="test",
            selected_analysts=None,
            dry_run=False,
        )
        runner._broker = MagicMock()

        fake_executor = MagicMock()
        fake_executor.submitted_orders = {"AAPL": "ord-456"}
        fake_executor.execute_decisions.return_value = {}

        filled_fills = {
            "ord-456": {
                "status": "filled",
                "filled_qty": 10.0,
                "filled_avg_price": 150.0,
                "ticker": "AAPL",
            }
        }

        fake_reconciler = MagicMock()
        fake_reconciler.reconcile.return_value = filled_fills

        with (
            patch("src.live.runner.LiveExecutor", return_value=fake_executor),
            patch("src.live.reconciler.Reconciler", return_value=fake_reconciler),
            caplog.at_level(logging.WARNING, logger="src.live.runner"),
        ):
            runner.execute({"AAPL": {"action": "buy", "quantity": 10}})

        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("manual review" in t.lower() for t in warning_texts)
