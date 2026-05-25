from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import logging
import random
import sys

from colorama import Fore, Style, init

from src.cli.input import add_risk_profile_arg, parse_tickers, select_model
from src.llm.models import check_provider_api_key
from src.main import run_quorai
from src.orchestration.preflight import update_run_manifest
from src.risk_profiles import get_profile
from src.utils.analysts import ALL_ANALYST_KEYS
from src.utils.tz import now_ny
from src.utils.validation import validate_ticker

from .comparison import RunConfig, run_comparison
from .engine import BacktestEngine
from .metrics import PerformanceMetricsCalculator


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated tickers, e.g. AAPL,MSFT")
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--calendar-days",
        "--days",
        dest="days",
        type=int,
        default=30,
        help="Number of calendar days to look back from end-date (default: 30)",
    )
    date_group.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (overrides --days)",
    )
    parser.add_argument("--initial-capital", type=float, default=100_000)
    parser.add_argument("--margin-requirement", type=float, default=0.0)
    parser.add_argument("--analysts", type=str, default=None, help="Comma-separated analyst IDs (default: all)")
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g. deepseek/deepseek-chat")
    parser.add_argument(
        "--model-provider",
        type=str,
        default=None,
        dest="model_provider",
        help="Provider string, e.g. OpenRouter (bypasses catalog lookup when set)",
    )
    parser.add_argument("--show-reasoning", action="store_true", dest="show_reasoning")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--use-regime-selection", action="store_true", dest="use_regime_selection", help="Narrow analyst set by daily SPY regime")
    parser.add_argument("--use-conviction-weights", action="store_true", dest="use_conviction_weights", help="Weight agents by rolling hit-rate (requires weights.json)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--run-label", type=str, default="", dest="run_label", help="Tag for this run (embedded in run_id and manifest for later filtering)")
    parser.add_argument("--log-dir", type=str, default=None, dest="log_dir", help="Override artifact directory (default: logs/backtest)")
    add_risk_profile_arg(parser)


def _resolve_dates(args: argparse.Namespace) -> tuple[str, str]:
    end_date = args.end_date or now_ny().strftime("%Y-%m-%d")
    start_date = args.start_date or (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=args.days)).strftime("%Y-%m-%d")
    return start_date, end_date


def _resolve_model(args: argparse.Namespace) -> tuple[str, str]:
    if args.model_provider is not None:
        return args.model, args.model_provider
    return select_model(args.model)


def _resolve_analysts(args: argparse.Namespace) -> list[str]:
    if args.analysts is not None:
        return [a.strip() for a in args.analysts.split(",") if a.strip()]
    return ALL_ANALYST_KEYS


def _compute_baselines(engine: "BacktestEngine", tickers: list[str], start_date: str, end_date: str) -> dict:
    bm = engine.get_benchmark()
    spy_ret = bm.get_return_pct("SPY", start_date, end_date)
    ticker_rets: dict[str, float | None] = {t: bm.get_return_pct(t, start_date, end_date) for t in tickers}
    valid = [r for r in ticker_rets.values() if r is not None]
    basket_ret: float | None = sum(valid) / len(valid) if valid else None
    return {"spy_return_pct": spy_ret, "tickers": ticker_rets, "equal_weight_basket_return_pct": basket_ret}


def _print_baselines(baselines: dict, tickers: list[str], start_date: str, end_date: str) -> None:
    def _fmt(ret: float | None) -> str:
        if ret is None:
            return "N/A"
        color = Fore.GREEN if ret >= 0 else Fore.RED
        return f"{color}{ret:.2f}%{Style.RESET_ALL}"

    print(f"\n{Fore.WHITE}{Style.BRIGHT}BASELINES ({start_date} → {end_date}){Style.RESET_ALL}")
    print(f"  SPY:                  {_fmt(baselines['spy_return_pct'])}")

    for t in tickers:
        print(f"  {t}:{'':>{max(0, 18 - len(t))}}{_fmt(baselines['tickers'].get(t))}")

    label = f"Equal-weight ({','.join(tickers)})"
    print(f"  {label}:  {_fmt(baselines['equal_weight_basket_return_pct'])}")


def _print_active_metrics(
    total_return: float,
    alpha_vs_spy: float | None,
    alpha_vs_basket: float | None,
    ir_vs_spy: float | None,
    ir_vs_basket: float | None,
) -> None:
    def _fmt_pct(v: float | None) -> str:
        if v is None:
            return "N/A"
        color = Fore.GREEN if v >= 0 else Fore.RED
        sign = "+" if v >= 0 else ""
        return f"{color}{sign}{v:.2f}%{Style.RESET_ALL}"

    def _fmt_ir(v: float | None) -> str:
        if v is None:
            return "N/A"
        return f"{v:.2f}"

    print(f"\n{Fore.WHITE}{Style.BRIGHT}ACTIVE PERFORMANCE{Style.RESET_ALL}")
    print(f"  Strategy:                      {_fmt_pct(total_return)}")
    print(f"  Alpha vs SPY:                  {_fmt_pct(alpha_vs_spy)}")
    print(f"  Alpha vs Equal-weight basket:  {_fmt_pct(alpha_vs_basket)}")
    print(f"  IR vs SPY:                     {_fmt_ir(ir_vs_spy)}")
    print(f"  IR vs Equal-weight basket:     {_fmt_ir(ir_vs_basket)}")


def _main_run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run a single backtest")
    _add_common_args(parser)
    args = parser.parse_args(argv)

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)

    start_date, end_date = _resolve_dates(args)

    import pandas as pd

    _trading_days = len(pd.date_range(start_date, end_date, freq="B"))
    logging.getLogger(__name__).info(
        "Backtest window: %s → %s (%d calendar days, ~%d trading days)",
        start_date,
        end_date,
        args.days,
        _trading_days,
    )

    tickers = [validate_ticker(t) for t in parse_tickers(args.tickers)]
    model_name, model_provider = _resolve_model(args)
    check_provider_api_key(model_provider)

    engine = BacktestEngine(
        agent=run_quorai,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.initial_capital,
        model_name=model_name,
        model_provider=model_provider,
        selected_analysts=_resolve_analysts(args),
        initial_margin_requirement=args.margin_requirement,
        llm_temperature=args.temperature,
        show_reasoning=args.show_reasoning,
        use_regime_selection=args.use_regime_selection,
        use_conviction_weights=args.use_conviction_weights,
        risk_profile=get_profile(args.risk_profile),
        seed=args.seed,
        run_label=args.run_label,
        log_dir=args.log_dir,
    )

    cli_args_record = {"argv": sys.argv[:], "parsed": vars(args)}
    result_record: dict | None = None
    try:
        metrics = engine.run_backtest()
        values = engine.get_portfolio_values()

        if values:
            print(f"\n{Fore.WHITE}{Style.BRIGHT}ENGINE RUN COMPLETE{Style.RESET_ALL}")
            last_value = values[-1]["Portfolio Value"]
            start_value = values[0]["Portfolio Value"]
            total_return = (last_value / start_value - 1.0) * 100.0 if start_value else 0.0
            print(f"Total Return: {Fore.GREEN if total_return >= 0 else Fore.RED}{total_return:.2f}%{Style.RESET_ALL}")

        if metrics.get("sharpe_ratio") is not None:
            print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
        if metrics.get("sortino_ratio") is not None:
            print(f"Sortino: {metrics['sortino_ratio']:.2f}")
        if metrics.get("max_drawdown") is not None:
            md = abs(metrics["max_drawdown"]) if metrics["max_drawdown"] is not None else 0.0
            if metrics.get("max_drawdown_date"):
                print(f"Max DD: {md:.2f}% on {metrics['max_drawdown_date']}")
            else:
                print(f"Max DD: {md:.2f}%")

        baselines = _compute_baselines(engine, tickers, start_date, end_date)
        _print_baselines(baselines, tickers, start_date, end_date)

        bm = engine.get_benchmark()
        _calc = PerformanceMetricsCalculator()
        spy_daily = bm.get_daily_returns("SPY", start_date, end_date)
        basket_daily = bm.get_basket_daily_returns(tickers, start_date, end_date)
        spy_rel = _calc.compute_benchmark_relative(values, spy_daily) if spy_daily is not None else {"alpha_pct": None, "information_ratio": None}
        basket_rel = _calc.compute_benchmark_relative(values, basket_daily) if basket_daily is not None else {"alpha_pct": None, "information_ratio": None}

        metrics["alpha_vs_spy_pct"] = spy_rel["alpha_pct"]
        metrics["alpha_vs_basket_pct"] = basket_rel["alpha_pct"]
        metrics["information_ratio_vs_spy"] = spy_rel["information_ratio"]
        metrics["information_ratio_vs_basket"] = basket_rel["information_ratio"]

        _print_active_metrics(
            total_return if values else 0.0,
            spy_rel["alpha_pct"],
            basket_rel["alpha_pct"],
            spy_rel["information_ratio"],
            basket_rel["information_ratio"],
        )

        result_record = {"metrics": dict(metrics), "baselines": baselines}
        if values:
            result_record["total_return_pct"] = total_return
            result_record["initial_portfolio_value"] = start_value
            result_record["final_portfolio_value"] = last_value
    finally:
        patch: dict = {"cli_args": cli_args_record}
        if result_record is not None:
            patch["result"] = result_record
        update_run_manifest(engine.run_id, patch, log_dir=args.log_dir or "logs/backtest")

    return 0


def _main_compare(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run side-by-side A/B backtest comparison")
    _add_common_args(parser)
    parser.add_argument(
        "--mode",
        choices=["regime", "weights", "both"],
        default="both",
        help="Which comparison to run: regime, weights, or both (default: both)",
    )
    args = parser.parse_args(argv)

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)

    start_date, end_date = _resolve_dates(args)
    tickers = [validate_ticker(t) for t in parse_tickers(args.tickers)]
    model_name, model_provider = _resolve_model(args)
    check_provider_api_key(model_provider)

    common = dict(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.initial_capital,
        model_name=model_name,
        model_provider=model_provider,
        selected_analysts=_resolve_analysts(args),
        initial_margin_requirement=args.margin_requirement,
        risk_profile=get_profile(args.risk_profile),
        seed=args.seed,
    )

    if args.mode in ("regime", "both"):
        print("\n=== Full analyst set vs Regime-selected analysts ===")
        run_comparison(
            [
                RunConfig(label="Full analyst set", use_regime_selection=False, **common),
                RunConfig(label="Regime selection", use_regime_selection=True, **common),
            ]
        )

    if args.mode in ("weights", "both"):
        print("\n=== Uniform weights vs Conviction weights ===")
        run_comparison(
            [
                RunConfig(label="Uniform weights", use_conviction_weights=False, **common),
                RunConfig(label="Conviction weights", use_conviction_weights=True, **common),
            ]
        )

    return 0


def _main_feedback(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Label a signal log with forward returns and compute per-agent conviction weights")
    parser.add_argument("--signal-log", required=True, dest="signal_log", help="Path to JSONL signal log (from a backtest or live run)")
    parser.add_argument("--horizon", type=int, default=5, help="Forward-return horizon in trading days (default: 5)")
    parser.add_argument("--window", type=int, default=60, help="Rolling scoring window in trading days (default: 60)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Directory for labeled log and accuracy report (default: same directory as signal log)",
    )
    args = parser.parse_args(argv)

    from datetime import datetime, timedelta
    import json
    from pathlib import Path

    from src.feedback.labeler import label_signals
    from src.feedback.scorer import compute_weights
    from src.tools.api import get_price_data

    signal_path = Path(args.signal_log)

    records = [json.loads(line) for line in signal_path.read_text().splitlines() if line.strip()]
    if not records:
        print("Signal log is empty — nothing to label.")
        return 1

    tickers = sorted({r["ticker"] for r in records})
    dates = [r["date"] for r in records]
    min_date, max_date = min(dates), max(dates)
    buffer_end = (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=45)).strftime("%Y-%m-%d")

    print(f"Fetching price data for {len(tickers)} tickers ({min_date} → {buffer_end}) ...")
    price_data = {ticker: get_price_data(ticker, min_date, buffer_end) for ticker in tickers}

    output_dir = Path(args.output_dir) if args.output_dir else signal_path.parent
    labeled_path = str(output_dir / f"labeled_{signal_path.name}")

    print(f"Labeling {len(records)} signal records ...")
    label_signals(args.signal_log, price_data, output_path=labeled_path)

    weights_path = "src/feedback/weights.json"
    report_path = str(output_dir / "accuracy_report.json")

    print("Computing conviction weights ...")
    weights = compute_weights(
        labeled_path,
        horizon=args.horizon,
        window_days=args.window,
        weights_path=weights_path,
        report_path=report_path,
    )

    if weights:
        print(f"\nWeights for {len(weights)} agents written to {weights_path}")
        print(f"Accuracy report written to {report_path}")
        for agent_id, w in sorted(weights.items(), key=lambda x: -x[1])[:5]:
            print(f"  {agent_id}: {w:.3f}")
        if len(weights) > 5:
            print(f"  ... ({len(weights) - 5} more)")
    else:
        print("No weights computed — insufficient labeled data (need ≥ min_samples per agent).")

    return 0


def main() -> int:
    init(autoreset=True)
    argv = sys.argv[1:]
    if argv and argv[0] == "compare":
        return _main_compare(argv[1:])
    if argv and argv[0] == "feedback":
        return _main_feedback(argv[1:])
    return _main_run(argv)
