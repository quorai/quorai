from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

from src.backtesting.engine import BacktestEngine
from src.backtesting.types import PerformanceMetrics
from src.main import run_quorai


@dataclass
class RunConfig:
    label: str
    tickers: list[str]
    start_date: str
    end_date: str
    initial_capital: float
    selected_analysts: list[str] | None = None
    use_regime_selection: bool = False
    use_conviction_weights: bool = False
    initial_margin_requirement: float = 0.0
    model_name: str = "deepseek/deepseek-v4-flash"
    model_provider: str = "OpenRouter"


@dataclass
class RunResult:
    label: str
    metrics: PerformanceMetrics
    token_summary: dict[str, Any]
    final_value: float
    duration_seconds: float


def run_comparison(configs: list[RunConfig], output_dir: str = "logs") -> list[RunResult]:
    """Run each config sequentially and return results for comparison."""
    results = []
    for cfg in configs:
        start = time.monotonic()

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
            )
        )

    _print_comparison_table(results, initial_capital=configs[0].initial_capital)

    out_path = Path(output_dir) / f"comparison-{configs[0].start_date}-{configs[0].end_date}.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([_result_to_dict(r) for r in results], f, indent=2)

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
    print(sep + "\n")


def _result_to_dict(r: RunResult) -> dict:
    return {
        "label": r.label,
        "metrics": dict(r.metrics),
        "token_summary": r.token_summary,
        "final_value": r.final_value,
        "duration_seconds": r.duration_seconds,
    }
