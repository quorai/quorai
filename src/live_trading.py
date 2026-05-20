"""Live paper trading entry point.

Usage:
    uv run python src/live_trading.py \\
        --tickers AAPL,MSFT,NVDA \\
        --model openrouter/anthropic/claude-3.5-sonnet \\
        --model-provider OpenRouter \\
        --dry-run
"""

import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

# Silence chatty HTTP libraries at module load; reconfigured to RichHandler in main()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a live paper trading cycle using the Quorai agent graph.")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. AAPL,MSFT")
    parser.add_argument("--analysts", default=None, help="Comma-separated analyst IDs to include (default: all)")
    parser.add_argument("--model", required=True, help="Model name, e.g. openrouter/anthropic/claude-3.5-sonnet")
    parser.add_argument("--model-provider", required=True, dest="model_provider", help="Provider string, e.g. OpenRouter")
    parser.add_argument("--margin-requirement", type=float, default=0.0, dest="margin_requirement", help="Margin requirement fraction (default: 0.0)")
    parser.add_argument("--temperature", type=float, default=None, help="LLM temperature override")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Print decisions but do not submit orders")
    parser.add_argument("--show-reasoning", action="store_true", dest="show_reasoning", help="Print each agent's reasoning and the debate summaries")
    parser.add_argument("--use-regime-selection", action="store_true", dest="use_regime_selection", help="Narrow analysts to the regime-appropriate set using SPY market data")
    parser.add_argument("--use-conviction-weights", action="store_true", dest="use_conviction_weights", help="Apply per-agent conviction weights from src/feedback/weights.json (requires prior backtest run with signal log)")
    parser.add_argument("--no-signal-log", action="store_false", dest="enable_signal_log", help="Disable writing the per-agent signal JSONL (logs/live/signals/signals-YYYY-MM-DD-live.jsonl)")
    parser.set_defaults(enable_signal_log=True)
    parser.add_argument(
        "--agent-model",
        action="append",
        dest="agent_models",
        metavar="AGENT=model[:PROVIDER]",
        help=(
            "Override model for a specific agent. Repeatable. "
            "Format: AGENT_KEY=model_slug[:PROVIDER] — provider defaults to OpenRouter. "
            "Use '*' as agent key for a wildcard fallback. "
            "Example: --agent-model portfolio_manager=anthropic/claude-sonnet-4:Anthropic "
            "--agent-model '*=deepseek/deepseek-chat-v3.1'. "
            "Also reads QUORAI_AGENT_MODELS_JSON env var (JSON dict)."
        ),
    )
    from src.cli.input import add_risk_profile_arg

    add_risk_profile_arg(parser)
    parser.add_argument("--force", action="store_true", help="Skip market-open check (useful for development/testing)")
    parser.add_argument(
        "--catch-up",
        action="store_true",
        dest="catch_up",
        help=(
            "Missed-cron recovery: if no SOD equity file exists, fetch the prior-close equity "
            "from Alpaca's portfolio history and use it as the loss-limit baseline. "
            "Safe to use intraday; the fetched value is the true start-of-day baseline, "
            "not the (potentially depressed) current equity."
        ),
    )
    parser.add_argument("--confirm", action="store_true", help="Auto-confirm without interactive prompt")
    parser.add_argument(
        "--require-approval",
        action="store_true",
        dest="require_approval",
        help="Send orders to Telegram for approval before submitting (use with --confirm for cron)",
    )
    parser.add_argument(
        "--auto-submit",
        action="store_true",
        dest="auto_submit",
        help="Submit orders immediately without Telegram approval; send execution report to Telegram afterwards",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from src.utils.validation import validate_ticker

    tickers = [validate_ticker(t.strip().upper()) for t in args.tickers.split(",") if t.strip()]
    selected_analysts = [a.strip() for a in args.analysts.split(",")] if args.analysts else None

    # Reconfigure root logger to use Rich so log lines don't break the Live display
    from rich.logging import RichHandler

    from src.utils.progress import console as _rich_console

    _root = logging.getLogger()
    _root.handlers.clear()
    _rich_handler = RichHandler(console=_rich_console, show_path=False, markup=False, rich_tracebacks=False)
    _rich_handler.setFormatter(logging.Formatter("%(name)s — %(message)s", datefmt="[%X]"))
    _root.addHandler(_rich_handler)

    from src.broker.alpaca_client import AlpacaClient
    from src.config import get_settings, refresh_settings
    from src.live.audit_journal import AuditJournal
    from src.live.risk_gate import RiskGate
    from src.live.runner import LiveRunner
    from src.risk_profiles import get_profile

    profile = get_profile(args.risk_profile)
    settings = get_settings().model_copy(
        update={
            "MAX_ORDER_NOTIONAL": profile.max_order_notional,
            "MAX_ORDER_QTY": profile.max_order_qty,
            "DAILY_LOSS_LIMIT_PCT": profile.daily_loss_limit_pct,
        }
    )
    log = logging.getLogger(__name__)
    log.info("Risk profile: %s (base_limit=%.2f, notional_cap=$%.0f)", profile.name, profile.base_limit, profile.max_order_notional)
    broker = AlpacaClient()
    journal = AuditJournal(log_dir="logs/live")
    risk_gate = RiskGate(settings=settings, journal=journal)

    from src.live.idempotency_guard import (
        DenyByDefaultApprover,
        IdempotencyGuard,
        TelegramPriorRunApprover,
    )
    from src.notifications.telegram import TelegramClient

    if settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID:
        _approver = TelegramPriorRunApprover(
            telegram_client=TelegramClient(
                token=settings.TELEGRAM_BOT_TOKEN,
                chat_id=settings.TELEGRAM_CHAT_ID,
            ),
            timeout_seconds=settings.TELEGRAM_APPROVAL_TIMEOUT_SECONDS,
        )
    else:
        _approver = DenyByDefaultApprover()

    idempotency_guard = IdempotencyGuard(journal=journal, approver=_approver)

    from src.llm.request import RunRequest

    run_request = RunRequest.from_agent_model_args(agent_model_args=getattr(args, "agent_models", None) or [])
    run_request.validate_provider_keys(settings, global_model=args.model, global_provider=args.model_provider)

    runner = LiveRunner(
        tickers=tickers,
        model_name=args.model,
        model_provider=args.model_provider,
        selected_analysts=selected_analysts,
        margin_requirement=args.margin_requirement,
        llm_temperature=args.temperature,
        dry_run=args.dry_run,
        show_reasoning=args.show_reasoning,
        use_regime_selection=args.use_regime_selection,
        use_conviction_weights=args.use_conviction_weights,
        enable_signal_log=args.enable_signal_log,
        broker=broker,
        journal=journal,
        risk_gate=risk_gate,
        idempotency_guard=idempotency_guard,
        request=run_request,
        risk_profile=profile,
        catch_up=args.catch_up,
    )

    # Pre-flight: skip on non-trading days
    if not broker.is_market_open_today():
        if args.force:
            logging.getLogger(__name__).warning("Market is closed today — continuing anyway (--force).")
        else:
            logging.getLogger(__name__).info("Market is closed today — skipping run.")
            return

    # Pre-flight: check Telegram command inbox
    from src.notifications.command_store import CommandStore, parse_directive

    command_store = CommandStore()

    if settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID:
        tg_cmd = TelegramClient(
            token=settings.TELEGRAM_BOT_TOKEN,
            chat_id=settings.TELEGRAM_CHAT_ID,
        )
        try:
            for msg_text in tg_cmd.poll_text_messages():
                directive = parse_directive(msg_text)
                if directive is not None:
                    command_store.apply(directive, msg_text)
                    if directive == "none":
                        tg_cmd.send_message("▶️ Resumed — skip_until_continue cleared.")
                    else:
                        labels = {
                            "only_sells": "📉 Got it — will accept only sells next run.",
                            "skip_next": "⏭️ Got it — will skip the next run.",
                            "skip_until_continue": '⏸️ Got it — pausing until you send "continue".',
                        }
                        tg_cmd.send_message(labels[directive])
                    log.info("[commands] directive '%s' set from message: %s", directive, msg_text)
        except Exception as exc:
            log.warning("Failed to poll Telegram commands: %s", exc)

    active = command_store.load()
    if active.directive == "skip_next":
        log.info("Skipping this run (skip_next command active).")
        command_store.consume_one_shot(active)
        return
    if active.directive == "skip_until_continue":
        log.info("Skipping this run (skip_until_continue command active).")
        return

    # Pre-flight: KILL_SWITCH — reload settings so a mid-day .env flip takes effect
    _live_settings = refresh_settings()
    if _live_settings.KILL_SWITCH:
        log.warning("KILL_SWITCH is active — aborting before agent graph.")
        return

    # Step 1: sync portfolio + run agents
    print("Running agent graph…")
    decisions, snapshot = runner.prepare()

    # Step 2: fetch latest prices for display
    from datetime import date, timedelta

    from src.tools.api import get_prices

    today_str = date.today().strftime("%Y-%m-%d")
    lookback_str = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    latest_prices: dict[str, float] = {}
    for ticker in decisions:
        prices = get_prices(ticker, lookback_str, today_str)
        latest_prices[ticker] = prices[-1].close if prices else float("nan")

    # Step 3: print decisions table
    print("\nDecisions:")
    print(f"{'Ticker':<10} {'Action':<8} {'Qty':>10} {'Price':>10}")
    print("-" * 44)
    for ticker, d in decisions.items():
        action = d.get("action", "hold")
        qty = d.get("quantity", 0)
        price = latest_prices.get(ticker, float("nan"))
        print(f"{ticker:<10} {action:<8} {qty:>10.3f} {price:>10.2f}")
    print()

    # Signal log path
    if runner.signal_log_path:
        print(f"Signal log: {runner.signal_log_path}")

    # Token usage summary
    tok = runner.token_summary()
    if tok:
        print(f"Tokens: {tok['calls']} calls, {tok['input_tokens']} in / {tok['output_tokens']} out")

    # Apply only_sells filter (one-shot)
    if active.directive == "only_sells":
        decisions = {t: d for t, d in decisions.items() if d.get("action", "hold") in ("sell", "hold")}
        print("only_sells active — buy orders suppressed.")
        command_store.consume_one_shot(active)

    # Step 4: dry-run short-circuit
    if args.dry_run:
        print("Dry run — no orders submitted.")
        return

    # Step 5: confirmation gate
    _approval_reason: str = "confirmed"
    if args.require_approval:
        if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
            log.error("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — cannot request approval; aborting.")
            return  # fail-closed: missing creds means no human reviewed the orders
        tg = TelegramClient(
            token=settings.TELEGRAM_BOT_TOKEN,
            chat_id=settings.TELEGRAM_CHAT_ID,
        )
        text = tg.format_decisions_table(decisions, latest_prices)
        try:
            msg_id = tg.send_approval_request(text)
            decision = tg.wait_for_decision(msg_id, timeout_seconds=settings.TELEGRAM_APPROVAL_TIMEOUT_SECONDS)
        except Exception as exc:
            log.error("Telegram error: %s — aborting for safety.", exc)
            return  # fail-closed: unknown Telegram state means no human confirmed
        if decision != "approve":
            reason = "telegram_reject" if decision == "reject" else "telegram_timeout"
            print(f"{'Rejected' if decision == 'reject' else 'Timed out'} via Telegram — no orders submitted.")
            for ticker, d in decisions.items():
                journal.record(
                    ticker=ticker,
                    action=d.get("action", "hold"),
                    qty=float(d.get("quantity", 0)),
                    side=d.get("action", "hold"),
                    status="rejected",
                    reason=reason,
                )
            return
        _approval_reason = "telegram_approved"

    elif args.auto_submit:
        _approval_reason = "auto_submitted"

    elif not args.confirm:
        answer = input("Submit these orders? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    # Step 6: execute — executor writes its own journal records (pending → submitted/error)
    execution_results = runner.execute(decisions)

    print("\nExecution results:")
    print(f"{'Ticker':<10} {'Result'}")
    print("-" * 32)
    for ticker, result in execution_results.items():
        print(f"{ticker:<10} {result}")
    print()

    # Post-execution Telegram report for --auto-submit
    if args.auto_submit and settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID:
        tg_notify = TelegramClient(
            token=settings.TELEGRAM_BOT_TOKEN,
            chat_id=settings.TELEGRAM_CHAT_ID,
        )
        lines = ["*Executed Orders*", "", "Ticker | Action | Qty | Price | Result", "------ | ------ | --- | ----- | ------"]
        for ticker, d in decisions.items():
            action = d.get("action", "hold")
            qty = float(d.get("quantity", 0))
            price = latest_prices.get(ticker, float("nan"))
            result = execution_results.get(ticker, "—")
            lines.append(f"{ticker} | {action} | {qty:.3f} | ${price:.2f} | {result}")
        tok = runner.token_summary()
        if tok:
            lines.append("")
            lines.append(f"_Tokens: {tok['calls']} calls · {tok['input_tokens']} in / {tok['output_tokens']} out_")
        try:
            tg_notify.send_message("\n".join(lines))
        except Exception as exc:
            log.warning("Failed to send execution report to Telegram: %s", exc)


if __name__ == "__main__":
    main()
