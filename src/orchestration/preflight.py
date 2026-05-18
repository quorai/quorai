from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Callable

import pandas as pd

from src.backtesting.controller import AgentController
from src.backtesting.signal_log import SignalLogger
from src.backtesting.types import AgentOutput, PortfolioSnapshot
from src.feedback.loader import load_weights
from src.llm.request import RunRequest
from src.regime import classify_regime_with_indicators, select_analysts_for_regime
from src.risk_profiles import RiskProfile

if TYPE_CHECKING:
    from src.backtesting.portfolio import Portfolio

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _token_summary(calls: list[dict]) -> dict:
    if not calls:
        return {}
    return {
        "calls": len(calls),
        "input_tokens": sum(e.get("input_tokens", 0) for e in calls),
        "output_tokens": sum(e.get("output_tokens", 0) for e in calls),
        "total_tokens": sum(e.get("input_tokens", 0) + e.get("output_tokens", 0) for e in calls),
        "cache_read_tokens": sum(e.get("cache_read_tokens", 0) for e in calls),
        "cache_creation_tokens": sum(e.get("cache_creation_tokens", 0) for e in calls),
    }


def _portfolio_to_dict(portfolio: "Portfolio | PortfolioSnapshot") -> dict:
    """Coerce a Portfolio object or snapshot dict to a plain serializable dict."""
    from src.backtesting.portfolio import Portfolio as PortfolioClass  # avoid circular at module level

    if isinstance(portfolio, PortfolioClass):
        return dict(portfolio.get_snapshot())
    return dict(portfolio)


def _extract_risk_manager(analyst_signals: dict) -> dict:
    """Pull risk_management_agent entries out of analyst_signals for the bundle's risk_manager section."""
    result = {}
    for agent_key in ("risk_management_agent", *[k for k in analyst_signals if k.startswith("risk_management_agent_")]):
        ticker_map = analyst_signals.get(agent_key, {})
        for ticker, data in ticker_map.items():
            result[ticker] = data
    return result


def _atomic_json_write(path: Path, obj: object) -> None:
    """Serialize obj to JSON and replace path atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(obj, indent=2, default=str)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class PipelineContext:
    """Stable per-run orchestration state shared by live and backtest flows.

    Encapsulates regime selection, conviction-weights loading, signal logging,
    and AgentController invocation so both flows share a single implementation.

    Callers supply per-cycle data (date, prices, portfolio, spy_df) to
    run_cycle(); all setup state (weights, logger, request) is held here.

    Typical usage::

        with PipelineContext.build(agent=run_quorai, ...) as ctx:
            output = ctx.run_cycle(
                date="2026-01-15",
                lookback_start="2025-12-15",
                portfolio=snapshot,
                signal_prices=prices,
                spy_df=spy_df,
            )
        summary = ctx.token_summary()
    """

    def __init__(
        self,
        *,
        agent: Callable[..., AgentOutput],
        tickers: list[str],
        run_id: str,
        mode: str,
        model_name: str,
        model_provider: str,
        selected_analysts: list[str] | None,
        llm_temperature: float | None,
        show_reasoning: bool,
        use_regime_selection: bool,
        use_conviction_weights: bool,
        conviction_weights: dict[str, float],
        signal_logger: SignalLogger | None,
        request: RunRequest | None,
        risk_profile: RiskProfile | None = None,
        log_dir: str = "logs",
    ) -> None:
        self._agent = agent
        self._tickers = tickers
        self._run_id = run_id
        self._mode = mode
        self._model_name = model_name
        self._model_provider = model_provider
        self._selected_analysts = selected_analysts
        self._llm_temperature = llm_temperature
        self._show_reasoning = show_reasoning
        self._use_regime_selection = use_regime_selection
        self._use_conviction_weights = use_conviction_weights
        self._conviction_weights = conviction_weights
        self._signal_logger = signal_logger
        self._request = request
        self._risk_profile = risk_profile
        self._log_dir = log_dir
        self._controller = AgentController()
        self._token_usage: list[dict] = []
        self._started_at = _utcnow_iso()
        self._cycle_dates: list[str] = []
        self._cycle_files: list[str] = []

    @classmethod
    def build(
        cls,
        *,
        agent: Callable[..., AgentOutput],
        tickers: list[str],
        run_id: str,
        mode: str = "backtest",
        model_name: str,
        model_provider: str,
        selected_analysts: list[str] | None = None,
        llm_temperature: float | None = None,
        show_reasoning: bool = False,
        use_regime_selection: bool = False,
        use_conviction_weights: bool = False,
        enable_signal_log: bool = True,
        request: RunRequest | None = None,
        risk_profile: RiskProfile | None = None,
        log_dir: str = "logs",
    ) -> PipelineContext:
        """Create a PipelineContext, loading weights and opening the signal logger."""
        conviction_weights: dict[str, float] = {}
        if use_conviction_weights:
            conviction_weights = load_weights()
            if not conviction_weights:
                logger.warning("--use-conviction-weights set but weights.json is missing or empty; running with uniform weights")
            else:
                logger.info("Loaded conviction weights for %d agents", len(conviction_weights))

        signal_logger: SignalLogger | None = SignalLogger(run_id, log_dir=log_dir) if enable_signal_log else None

        instance = cls(
            agent=agent,
            tickers=tickers,
            run_id=run_id,
            mode=mode,
            model_name=model_name,
            model_provider=model_provider,
            selected_analysts=selected_analysts,
            llm_temperature=llm_temperature,
            show_reasoning=show_reasoning,
            use_regime_selection=use_regime_selection,
            use_conviction_weights=use_conviction_weights,
            conviction_weights=conviction_weights,
            signal_logger=signal_logger,
            request=request,
            risk_profile=risk_profile,
            log_dir=log_dir,
        )
        instance._write_run_manifest("running")
        return instance

    def run_cycle(
        self,
        *,
        date: str,
        lookback_start: str,
        portfolio: "Portfolio | PortfolioSnapshot",
        signal_prices: dict[str, float],
        spy_df: pd.DataFrame | None = None,
    ) -> AgentOutput:
        """Run one agent cycle and log signals; return the full AgentOutput.

        ``date`` is the as-of date (YYYY-MM-DD); ``lookback_start`` is the
        earliest data date. Regime narrowing uses ``spy_df`` when provided.
        """
        cycle_started_at = _utcnow_iso()
        portfolio_before = _portfolio_to_dict(portfolio)

        effective_analysts = self._selected_analysts
        regime_info: dict = {
            "classified": None,
            "indicators": {},
            "narrowed_analysts": [],
            "effective_analysts": list(effective_analysts or []),
        }

        if self._use_regime_selection and spy_df is not None and not spy_df.empty:
            regime, indicators = classify_regime_with_indicators(spy_df, date)
            regime_analysts = select_analysts_for_regime(regime)
            effective_analysts = regime_analysts if regime_analysts else self._selected_analysts
            regime_info = {
                "classified": regime.value,
                "indicators": indicators,
                "narrowed_analysts": regime_analysts or [],
                "effective_analysts": list(effective_analysts or []),
            }
            logger.debug(
                "Regime %s → %d analysts on %s",
                regime.value,
                len(effective_analysts or []),
                date,
            )

        output = self._controller.run_agent(
            self._agent,
            tickers=self._tickers,
            start_date=lookback_start,
            end_date=date,
            portfolio=portfolio,
            model_name=self._model_name,
            model_provider=self._model_provider,
            selected_analysts=effective_analysts,
            llm_temperature=self._llm_temperature,
            show_reasoning=self._show_reasoning,
            conviction_weights=self._conviction_weights,
            request=self._request,
            risk_profile=self._risk_profile,
        )

        if self._signal_logger is not None:
            self._signal_logger.log_day(date, output.get("analyst_signals", {}), signal_prices)

        self._token_usage.extend(output.get("token_usage") or [])

        cycle_finished_at = _utcnow_iso()
        self._write_cycle_bundle(
            date=date,
            lookback_start=lookback_start,
            cycle_started_at=cycle_started_at,
            cycle_finished_at=cycle_finished_at,
            output=output,
            portfolio_before=portfolio_before,
            signal_prices=signal_prices,
            regime_info=regime_info,
        )

        return output

    def _write_cycle_bundle(
        self,
        *,
        date: str,
        lookback_start: str,
        cycle_started_at: str,
        cycle_finished_at: str,
        output: AgentOutput,
        portfolio_before: dict,
        signal_prices: dict[str, float],
        regime_info: dict,
    ) -> None:
        try:
            token_calls = output.get("token_usage") or []
            bundle: dict = {
                "schema_version": 1,
                "run_id": self._run_id,
                "mode": self._mode,
                "cycle_date": date,
                "lookback_start": lookback_start,
                "started_at": cycle_started_at,
                "finished_at": cycle_finished_at,
                "inputs": {
                    "tickers": self._tickers,
                    "model_name": self._model_name,
                    "model_provider": self._model_provider,
                    "selected_analysts_requested": self._selected_analysts,
                    "use_regime_selection": self._use_regime_selection,
                    "use_conviction_weights": self._use_conviction_weights,
                    "risk_profile": (
                        {
                            "name": self._risk_profile.name,
                            "base_limit": self._risk_profile.base_limit,
                            "max_order_notional": self._risk_profile.max_order_notional,
                            "max_order_qty": self._risk_profile.max_order_qty,
                            "daily_loss_limit_pct": self._risk_profile.daily_loss_limit_pct,
                        }
                        if self._risk_profile is not None
                        else None
                    ),
                    "agent_models": ({agent_id: list(model_cfg) for agent_id, model_cfg in self._request.agent_models.items()} if self._request is not None and self._request.agent_models else {}),
                },
                "regime": regime_info,
                "conviction_weights": self._conviction_weights,
                "analyst_signals": output.get("analyst_signals", {}),
                "group_signals": output.get("group_signals", {}),
                "debate_summaries": output.get("debate_summaries", {}),
                "risk_manager": _extract_risk_manager(output.get("analyst_signals", {})),
                "portfolio_manager": output.get("pm_decisions", {}),
                "portfolio_before": portfolio_before,
                "portfolio_after": None,
                "signal_prices": signal_prices,
                "fill_prices": None,
                "trades": [],
                "token_usage": {
                    "calls": token_calls,
                    "summary": _token_summary(token_calls),
                },
                "llm_io_dir": None,
            }

            cycles_dir = Path(self._log_dir) / "cycles" / self._run_id
            bundle_path = cycles_dir / f"cycle-{date}.json"
            _atomic_json_write(bundle_path, bundle)

            rel_path = str(bundle_path)
            self._cycle_dates.append(date)
            self._cycle_files.append(rel_path)
            self._write_run_manifest("running")
        except Exception:
            logger.exception("Failed to write cycle bundle for run=%s date=%s", self._run_id, date)

    def _write_run_manifest(self, status: str) -> None:
        try:
            manifest: dict = {
                "schema_version": 1,
                "run_id": self._run_id,
                "mode": self._mode,
                "started_at": self._started_at,
                "finished_at": None,
                "inputs": {
                    "tickers": self._tickers,
                    "model_name": self._model_name,
                    "model_provider": self._model_provider,
                    "selected_analysts_requested": self._selected_analysts,
                    "use_regime_selection": self._use_regime_selection,
                    "use_conviction_weights": self._use_conviction_weights,
                    "risk_profile": (
                        {
                            "name": self._risk_profile.name,
                            "base_limit": self._risk_profile.base_limit,
                            "max_order_notional": self._risk_profile.max_order_notional,
                            "max_order_qty": self._risk_profile.max_order_qty,
                            "daily_loss_limit_pct": self._risk_profile.daily_loss_limit_pct,
                        }
                        if self._risk_profile is not None
                        else None
                    ),
                    "agent_models": ({agent_id: list(model_cfg) for agent_id, model_cfg in self._request.agent_models.items()} if self._request is not None and self._request.agent_models else {}),
                },
                "cycle_dates": list(self._cycle_dates),
                "cycle_files": list(self._cycle_files),
                "signal_log_path": self.signal_log_path,
                "token_summary": _token_summary(self._token_usage),
                "status": status,
                "error": None,
            }
            runs_dir = Path(self._log_dir) / "runs"
            manifest_path = runs_dir / f"{self._run_id}.json"
            _atomic_json_write(manifest_path, manifest)
        except Exception:
            logger.exception("Failed to write run manifest for run=%s", self._run_id)

    @property
    def signal_log_path(self) -> str | None:
        return self._signal_logger.path if self._signal_logger is not None else None

    def token_summary(self) -> dict:
        """Return aggregated token-usage stats accumulated across all run_cycle calls."""
        return _token_summary(self._token_usage)

    def close(self) -> None:
        if self._signal_logger is not None:
            self._signal_logger.close()
        try:
            self._write_run_manifest("completed")
            # Stamp finished_at on the manifest
            runs_dir = Path(self._log_dir) / "runs"
            manifest_path = runs_dir / f"{self._run_id}.json"
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as fh:
                    manifest = json.load(fh)
                manifest["finished_at"] = _utcnow_iso()
                _atomic_json_write(manifest_path, manifest)
        except Exception:
            logger.exception("Failed to finalize run manifest for run=%s", self._run_id)

    def __enter__(self) -> PipelineContext:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
