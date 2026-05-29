from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from src.backtesting.costs import CostModel
from src.backtesting.engine import BacktestEngine
from src.backtesting.types import PerformanceMetrics
from src.config import get_settings
from src.main import run_quorai
from src.risk_profiles import RiskProfile


@dataclass
class RunConfig:
    """model_name/model_provider default from Settings so env overrides apply at construction time."""

    label: str
    tickers: list[str]
    start_date: str
    end_date: str
    initial_capital: float
    selected_analysts: list[str] | None = None
    use_regime_selection: bool = False
    use_conviction_weights: bool = False
    initial_margin_requirement: float = 0.0
    model_name: str = field(default_factory=lambda: get_settings().DEFAULT_MODEL)
    model_provider: str = field(default_factory=lambda: get_settings().DEFAULT_PROVIDER)
    seed: int | None = None
    risk_profile: RiskProfile | None = None
    slippage_bps: float = 0.0
    commission_bps: float = 0.0
    borrow_bps_annual: float = 0.0


@dataclass
class RunResult:
    label: str
    metrics: PerformanceMetrics
    token_summary: dict[str, Any]
    final_value: float
    duration_seconds: float
    cost_summary: dict[str, float] = field(default_factory=dict)


def run_comparison(configs: list[RunConfig], output_dir: str = "logs") -> list[RunResult]:
    """Run each config sequentially and return results for comparison."""
    results = []
    for cfg in configs:
        start = time.monotonic()

        slug = re.sub(r"[^a-z0-9]+", "-", cfg.label.lower()).strip("-")
        engine = BacktestEngine(
            agent=run_quorai,
            tickers=cfg.tickers,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            initial_capital=cfg.initial_capital,
            model_name=cfg.model_name,
            model_provider=cfg.model_provider,
            selected_analysts=cfg.selected_analysts,
            initial_margin_requirement=cfg.initial_margin_requirement,
            use_regime_selection=cfg.use_regime_selection,
            use_conviction_weights=cfg.use_conviction_weights,
            run_label=slug,
            seed=cfg.seed,
            risk_profile=cfg.risk_profile,
            cost_model=CostModel.from_args(
                slippage_bps=cfg.slippage_bps,
                commission_bps=cfg.commission_bps,
                borrow_bps_annual=cfg.borrow_bps_annual,
            ),
        )
        metrics = engine.run_backtest()
        pv = engine.get_portfolio_values()
        final_value = float(pv[-1]["Portfolio Value"]) if pv else cfg.initial_capital

        results.append(
            RunResult(
                label=cfg.label,
                metrics=metrics,
                token_summary=engine.get_token_summary(),
                final_value=final_value,
                duration_seconds=time.monotonic() - start,
                cost_summary=engine.get_cost_summary(),
            )
        )

    _print_comparison_table(results, initial_capital=configs[0].initial_capital)

    out_path = Path(output_dir) / f"comparison-{configs[0].start_date}-{configs[0].end_date}.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([_result_to_dict(r) for r in results], f, indent=2)

    return results


def run_ablation(common: dict, output_dir: str = "logs") -> list[RunResult]:
    """Run a baseline plus one config per disabled PM gate and print a comparison table.

    ``common`` should contain all RunConfig fields except ``label`` and the gate
    toggles.  The cost fields (slippage_bps, commission_bps, borrow_bps_annual)
    are forwarded from ``common`` so churn deltas are cost-adjusted.

    Gate env-var toggles (QUORAI_GATE_REGIME / _PANEL / _MIN_HOLD) are set
    temporarily via os.environ for each ablation run and restored afterwards.
    """
    _GATE_CONFIGS: list[tuple[str, dict[str, str]]] = [
        ("Baseline (all gates on)", {}),
        ("No regime gate", {"QUORAI_GATE_REGIME": "0"}),
        ("No panel gate", {"QUORAI_GATE_PANEL": "0"}),
        ("No min-hold gate", {"QUORAI_GATE_MIN_HOLD": "0"}),
    ]

    configs: list[RunConfig] = []
    for label, gate_overrides in _GATE_CONFIGS:
        cfg = RunConfig(label=label, **common)  # type: ignore[arg-type]
        configs.append(cfg)

    results: list[RunResult] = []
    for (label, gate_overrides), cfg in zip(_GATE_CONFIGS, configs):
        # Temporarily override gate env vars for this run.
        saved = {k: os.environ.get(k) for k in gate_overrides}
        os.environ.update(gate_overrides)
        try:
            run_results = run_comparison([cfg], output_dir=output_dir)
            results.extend(run_results)
        finally:
            # Restore original env values.
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    _print_ablation_table(results, initial_capital=common["initial_capital"])
    return results


def _print_comparison_table(results: list[RunResult], initial_capital: float) -> None:
    col_w = 22
    headers = ["Metric"] + [r.label[:col_w] for r in results]
    sep = "-" * (col_w + (col_w + 2) * len(results))
    print("\n" + sep)
    print("  ".join(h.ljust(col_w) for h in headers))
    print(sep)

    def row(label: str, *vals: str) -> None:
        print("  ".join([label.ljust(col_w)] + [v.ljust(col_w) for v in vals]))

    for metric_key, fmt in [
        ("sharpe_ratio", ".3f"),
        ("sortino_ratio", ".3f"),
        ("max_drawdown", ".2%"),
    ]:
        vals = []
        for r in results:
            v = r.metrics.get(metric_key)
            vals.append(f"{v:{fmt}}" if v is not None else "N/A")
        row(metric_key.replace("_", " ").title(), *vals)

    row("Final portfolio", *[f"${r.final_value:,.0f}" for r in results])
    row(
        "Total return",
        *[f"{(r.final_value - initial_capital) / initial_capital:.2%}" for r in results],
    )
    row("Token calls", *[str(r.token_summary.get("calls", "?")) for r in results])
    row("Total tokens", *[str(r.token_summary.get("total_tokens", "?")) for r in results])
    row("Duration (s)", *[f"{r.duration_seconds:.1f}" for r in results])
    # Show cost breakdown if any run had non-zero costs.
    if any(r.cost_summary.get("total_costs", 0) > 0 for r in results):
        row("Total costs", *[f"${r.cost_summary.get('total_costs', 0):,.2f}" for r in results])
    print(sep + "\n")


def _print_ablation_table(results: list[RunResult], initial_capital: float) -> None:
    """Print ablation results with delta columns relative to the baseline."""
    if not results:
        return
    baseline = results[0]
    baseline_return = (baseline.final_value - initial_capital) / initial_capital

    col_w = 26
    print(f"\n{'=' * 70}")
    print("ABLATION SUMMARY  (baseline = all gates on)")
    print(f"{'=' * 70}")
    print(f"{'Run':<{col_w}} {'Return':>8} {'Δ vs base':>10} {'Sharpe':>7} {'Costs':>10}")
    print("-" * 70)
    for r in results:
        ret = (r.final_value - initial_capital) / initial_capital
        delta = ret - baseline_return
        sharpe = r.metrics.get("sharpe_ratio")
        costs = r.cost_summary.get("total_costs", 0)
        sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "N/A"
        delta_str = f"{delta:+.2%}"
        print(f"{r.label:<{col_w}} {ret:>8.2%} {delta_str:>10} {sharpe_str:>7} ${costs:>8,.0f}")
    print("=" * 70 + "\n")


def _result_to_dict(r: RunResult) -> dict:
    return {
        "label": r.label,
        "metrics": dict(r.metrics),
        "token_summary": r.token_summary,
        "final_value": r.final_value,
        "duration_seconds": r.duration_seconds,
        "cost_summary": r.cost_summary,
    }
