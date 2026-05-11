from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

from colorama import Fore, Style, init
from dateutil.relativedelta import relativedelta

from src.cli.input import parse_tickers, select_analysts, select_model
from src.main import run_quorai

from .engine import BacktestEngine

_NY = ZoneInfo("America/New_York")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run backtesting engine (modular)")
    parser.add_argument("--tickers", type=str, required=False, help="Comma-separated tickers")
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now(_NY).strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now(_NY) - relativedelta(months=1)).strftime("%Y-%m-%d"),
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument("--initial-capital", type=float, default=100000)
    parser.add_argument("--margin-requirement", type=float, default=0.0)
    parser.add_argument("--analysts", type=str, required=False)
    parser.add_argument("--analysts-all", action="store_true")
    parser.add_argument("--model", type=str, required=False, help="Model name to use")

    args = parser.parse_args()
    init(autoreset=True)

    tickers = parse_tickers(args.tickers)
    selected_analysts = select_analysts({"analysts_all": args.analysts_all, "analysts": args.analysts})
    model_name, model_provider = select_model(args.model)

    engine = BacktestEngine(
        agent=run_quorai,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        model_name=model_name,
        model_provider=model_provider,
        selected_analysts=selected_analysts,
        initial_margin_requirement=args.margin_requirement,
    )

    metrics = engine.run_backtest()
    values = engine.get_portfolio_values()

    # Minimal terminal output (no plots)
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
