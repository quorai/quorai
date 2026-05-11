from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from colorama import Fore, Style, init

from src.cli.input import parse_tickers, select_model
from src.main import run_quorai
from src.utils.analysts import ALL_ANALYST_KEYS

from .engine import BacktestEngine

_NY = ZoneInfo("America/New_York")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run backtesting engine")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated tickers, e.g. AAPL,MSFT")
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now(_NY).strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD (default: today)",
    )
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--days",
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
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g. deepseek/deepseek-v4-flash")
    parser.add_argument("--model-provider", type=str, default=None, dest="model_provider", help="Provider string, e.g. OpenRouter (bypasses catalog when set)")
    parser.add_argument("--show-reasoning", action="store_true", dest="show_reasoning", help="Print each agent's reasoning")
    parser.add_argument("--temperature", type=float, default=None, help="LLM temperature override")

    args = parser.parse_args()
    init(autoreset=True)

    start_date = args.start_date or (datetime.strptime(args.end_date, "%Y-%m-%d") - timedelta(days=args.days)).strftime("%Y-%m-%d")

    tickers = parse_tickers(args.tickers)

    if args.analysts is not None:
        selected_analysts = [a.strip() for a in args.analysts.split(",") if a.strip()]
    else:
        selected_analysts = ALL_ANALYST_KEYS

    if args.model_provider is not None:
        model_name, model_provider = args.model, args.model_provider
    else:
        model_name, model_provider = select_model(args.model)

    engine = BacktestEngine(
        agent=run_quorai,
        tickers=tickers,
        start_date=start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        model_name=model_name,
        model_provider=model_provider,
        selected_analysts=selected_analysts,
        initial_margin_requirement=args.margin_requirement,
        llm_temperature=args.temperature,
        show_reasoning=args.show_reasoning,
    )

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
