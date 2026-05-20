import json

from src.backtesting.portfolio import Portfolio
from src.orchestration.preflight import PipelineContext, update_run_manifest


def _make_ctx(log_dir: str) -> PipelineContext:
    return PipelineContext(
        agent=lambda **_: {"decisions": {}, "analyst_signals": {}},
        tickers=["AAPL"],
        run_id="test-run",
        mode="backtest",
        model_name="test-model",
        model_provider="test-provider",
        selected_analysts=None,
        llm_temperature=None,
        show_reasoning=False,
        use_regime_selection=False,
        use_conviction_weights=False,
        conviction_weights={},
        signal_logger=None,
        request=None,
        log_dir=log_dir,
    )


def _minimal_output():
    return {"decisions": {}, "analyst_signals": {}}


def test_finalize_cycle_populates_bundle(tmp_path):
    ctx = _make_ctx(str(tmp_path))
    date = "2024-01-15"

    ctx._write_cycle_bundle(
        date=date,
        lookback_start="2023-01-15",
        cycle_started_at="2024-01-15T09:00:00Z",
        cycle_finished_at="2024-01-15T09:01:00Z",
        output=_minimal_output(),
        portfolio_before={"cash": 100_000.0, "positions": {}},
        signal_prices={"AAPL": 150.0},
        regime_info={"classified": None, "indicators": {}, "narrowed_analysts": [], "effective_analysts": []},
    )

    portfolio = Portfolio(tickers=["AAPL"], initial_cash=100_000.0, margin_requirement=0.5)
    ctx.finalize_cycle(
        date=date,
        fill_prices={"AAPL": 151.0},
        executed_trades={"AAPL": 10.0},
        portfolio_after=portfolio,
    )

    bundle_path = tmp_path / "cycles" / "test-run" / f"cycle-{date}.json"
    bundle = json.loads(bundle_path.read_text())

    assert bundle["fill_prices"] == {"AAPL": 151.0}
    assert bundle["trades"] == {"AAPL": 10.0}
    assert bundle["portfolio_after"] is not None
    assert "cash" in bundle["portfolio_after"]


def test_finalize_cycle_noop_when_no_bundle(tmp_path):
    ctx = _make_ctx(str(tmp_path))
    # No prior _write_cycle_bundle call — file does not exist.
    # finalize_cycle must not raise.
    portfolio = Portfolio(tickers=["AAPL"], initial_cash=100_000.0, margin_requirement=0.5)
    ctx.finalize_cycle(
        date="2024-01-15",
        fill_prices={"AAPL": 151.0},
        executed_trades={"AAPL": 10.0},
        portfolio_after=portfolio,
    )


def test_update_run_manifest_merges_patch(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    manifest_path = runs_dir / "test-run.json"
    manifest_path.write_text(json.dumps({"run_id": "test-run", "status": "completed", "inputs": {"tickers": ["AAPL"]}}))

    update_run_manifest("test-run", {"cli_args": {"argv": ["--tickers", "AAPL"], "parsed": {"seed": 42}}, "result": {"total_return_pct": 5.0}}, log_dir=str(tmp_path))

    updated = json.loads(manifest_path.read_text())
    assert updated["run_id"] == "test-run"
    assert updated["status"] == "completed"
    assert updated["inputs"]["tickers"] == ["AAPL"]
    assert updated["cli_args"]["parsed"]["seed"] == 42
    assert updated["result"]["total_return_pct"] == 5.0


def test_update_run_manifest_noop_when_missing(tmp_path):
    # Should not raise even when the manifest file does not exist.
    update_run_manifest("nonexistent-run", {"cli_args": {}}, log_dir=str(tmp_path))
