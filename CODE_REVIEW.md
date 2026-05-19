# Code Review — Quorai Key Modules (2026-05-18)

## Scope, Methodology, and How to Read This Report

**Modules reviewed:** `src/live/` (executor, runner, risk_gate, audit_journal, sod_equity, reconciler, idempotency_guard), `src/broker/` (alpaca_client, portfolio_adapter), `src/backtesting/` (engine, portfolio, output, cli, signal_log, comparison, metrics, benchmarks), `src/orchestration/preflight.py`, `src/config.py`.

**Method:** Two parallel Explore agents read every file in the above modules fully. The top four critical findings were independently verified by reading the current source. No fixes were applied; only the remediation backlog (this file + `.ai/tasks/`) was created.

**Priority labels:** `CRITICAL` — real-money risk or safety bypass; `HIGH` — correctness bug that produces wrong outcomes or crashes; `MEDIUM` — code quality, silent-swallow, or performance issue that degrades reliability; `LOW` — minor cleanup with no runtime impact.

**Task links:** Every MEDIUM/HIGH/CRITICAL finding has a corresponding task file in `.ai/tasks/`. LOW findings are grouped into a single future cleanup PR; no task file is created per-LOW.

---

## Overall Assessment

### Live trading & broker
The live-trading stack is reasonably well-structured with a clean Protocol, deterministic `client_order_id`, and a meaningful test suite (~1400 LoC across 12 files). Recent commits show active correctness work. However, **three CRITICAL issues remain** that can expose real money to unbounded orders or silently skip legitimate trades: the notional cap does not guard against a zero-price default, the kill-switch is not read-refreshed at execution time, and the pending-order dedup collapses `sell` and `short` into the same key. Seven further HIGH issues (journal durability, override-suffix counting, qty validation, SOD equity fallback, etc.) and six MEDIUM issues round out the backlog.

### Backtesting & orchestration
The engine avoids same-bar look-ahead (signal close / next-bar open fill), and `Portfolio` cost-basis math is correct. `PipelineContext` is a genuinely useful abstraction. The dominant correctness problem is that **cycle bundles always record `portfolio_after: None / trades: []`** because `_write_cycle_bundle` fires before the engine executes trades — the feature introduced in commit `3360e4a` is currently inert. A close second is **run_id collision in A/B comparison** that overwrites the losing arm's artifacts. Non-deterministic LLM temperature, duplicate equity-curve seed point, and dead lookback guard are significant; corporate-action blindness is a latent risk.

---

## Findings — Live Trading & Broker

### [#01] CRITICAL — Safety — Notional cap bypassed on missing price
File: `src/live/risk_gate.py:73` / `src/live/executor.py:151`

Issue: `prices.get(ticker, 0.0)` returns `0.0` when signal prices are missing, and the notional check `opening_qty * price > MAX_ORDER_NOTIONAL` then always evaluates to `0 > cap` → False. An unbounded buy or short is allowed for any ticker with a missing price.

Fix: In `RiskGate._evaluate`, add `if price <= 0: return "missing_price"` before the notional check; also raise (not default to 0) when signal prices are absent.

Task: `.ai/tasks/20260518-1500-rv01-notional-missing-price.md`

---

### [#02] CRITICAL — Safety — Kill-switch not refreshed at execution time
File: `src/live/runner.py` (`run()` / `execute()`) / `src/live/risk_gate.py:62`

Issue: `KILL_SWITCH` is read only in `runner.run()`. `LiveExecutor.execute_decisions()` is a public method; callers that invoke it directly bypass the switch entirely. Additionally, `RiskGate` snapshots `self._settings` at construction; toggling `KILL_SWITCH` in `.env` after the process starts is never reflected.

Fix: Re-read `Settings` (via `get_settings.cache_clear()` + `get_settings()`) at the top of `execute_decisions()`; update `RiskGate._evaluate` to pull from a fresh `get_settings()` call rather than `self._settings`.

Task: `.ai/tasks/20260518-1501-rv02-kill-switch-refresh.md`

---

### [#03] CRITICAL — Correctness — Pending-order dedupe collapses sell/short
File: `src/live/executor.py:62-66, 144`

Issue: Open orders are indexed as `(symbol, side.value)` where Alpaca's `side.value` is `"buy"` or `"sell"`. Because both `sell` and `short` produce `side="sell"`, a pending sell on `AAPL` silently suppresses a new short on `AAPL` (and vice-versa). The first order to land wins; the second is skipped without logging.

Fix: Key by `(symbol, action)` using the order's `client_order_id` suffix (last segment after splitting on `-`), or build the pending set from `(order.symbol, order.client_order_id.split("-")[-1])` where action is recoverable.

Task: `.ai/tasks/20260518-1502-rv03-pending-side-collision.md`

---

### [#04] HIGH — Safety — Audit journal not fsync'd on submit
File: `src/live/audit_journal.py:46-48`

Issue: The journal appends JSON via `open(..., "a")` and calls `write()` without `flush()` + `os.fsync()`. A crash between `submit_order` returning (order accepted by Alpaca) and the OS flushing the write buffer can silently lose the "submitted" row — defeating the crash-recovery guarantee that the deterministic `client_order_id` is designed to provide.

Fix: After `f.write(...)`, call `f.flush()` then `os.fsync(f.fileno())` inside the locked block for statuses `"pending"` and `"submitted"`.

Task: `.ai/tasks/20260518-1503-rv04-journal-fsync.md`

---

### [#05] HIGH — Correctness — Override suffix can collide on prior error/rejected attempts
File: `src/live/executor.py:180-186`

Issue: `prior_count = sum(1 for e in list_submitted_today() if ...)` counts only entries with `status == "submitted"`. If a prior override attempt ended with `status == "error"` or `"rejected"`, it is excluded, and a subsequent attempt produces the same `-r{N}` that the failed attempt used — generating a duplicate `client_order_id` the broker may accept or reject unpredictably.

Fix: Count all prior entries for `(ticker, action)` regardless of status; pick `max_index + 1` over any already-used suffix rather than `count + 1`.

Task: `.ai/tasks/20260518-1504-rv05-override-suffix-collision.md`

---

### [#06] HIGH — Correctness — Quantity not validated for sign/finiteness
File: `src/live/executor.py:88`

Issue: `qty = round(float(decision.get("quantity", 0)), 3)` does not guard against negative values or non-finite floats. A decision with `quantity: -5` passes all downstream checks and submits a negative-qty order; `quantity: NaN` passes the `qty == 0` guard and reaches the broker call.

Fix: After rounding, assert `qty > 0 and math.isfinite(qty)`; log and skip with a clear message otherwise.

Task: `.ai/tasks/20260518-1505-rv06-qty-validation.md`

---

### [#07] HIGH — Safety — Masked equity fetch failure blocks emergency liquidations
File: `src/live/executor.py:84` / `src/live/risk_gate.py:77`

Issue: `float(account.equity or "0")` coerces a failed/missing equity fetch to `0.0`. The daily-loss formula `account_equity / sod_equity - 1 <= -DAILY_LOSS_LIMIT_PCT` then evaluates to `-1.0 <= -0.05` → True, rejecting all trades including closing ones. An Alpaca outage therefore silently blocks emergency liquidation with no error surfaced to the caller.

Fix: Raise `RuntimeError("could not fetch account equity")` when equity is None/empty; catch it in the runner and surface it clearly rather than masking it as a loss-limit rejection.

Task: `.ai/tasks/20260518-1506-rv07-equity-fetch-failure.md`

---

### [#08] HIGH — Correctness — SOD equity save aborts run when called post-open
File: `src/live/sod_equity.py:27-30` / `src/live/runner.py:113-114`

Issue: `save_sod_equity` raises `RuntimeError` when called after 09:30 ET, and `runner.prepare()` calls it unconditionally without catching that error. A process that starts (or restarts) after market open crashes the run entirely instead of falling back to the broker's `--catch-up` equity path.

Fix: In `runner.prepare()`, detect `now_ny() >= 09:30` and call the catch-up path automatically; reserve `save_sod_equity` for pre-open starts only.

Task: `.ai/tasks/20260518-1507-rv08-sod-equity-post-open.md`

---

### [#09] HIGH — Safety — Account equity fetched once for entire batch
File: `src/live/executor.py:81-84`

Issue: `account_equity` is fetched once before the decision loop. A flash-crash or large fill during a long batch is not re-checked; the daily-loss gate uses a stale value. Combined with the equity-mask issue (#07), an intraday breach can pass silently.

Fix: Re-fetch `account_equity` every N orders (configurable) or every time `DAILY_LOSS_LIMIT_PCT` is within 10% of being triggered, and always for the first closing trade after a loss-limit near-miss.

Task: `.ai/tasks/20260518-1508-rv09-stale-equity.md`

---

### [#10] HIGH — Correctness — Unknown side string silently becomes SELL
File: `src/broker/alpaca_client.py:118`

Issue: `OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL` maps every non-"buy" string — including `"buy_to_cover"`, typos, or empty strings — to a SELL order. This can submit the wrong side silently.

Fix: Replace with an explicit dict or if/elif chain that raises `ValueError(f"unknown order side: {side!r}")` for any unrecognised input.

Task: `.ai/tasks/20260518-1509-rv10-alpaca-side-mapping.md`

---

### [#11] MEDIUM — Code quality — Unused _TERMINAL_STATUSES constant
File: `src/live/executor.py:21`

Issue: `_TERMINAL_STATUSES` is defined but never referenced in `executor.py`; the same set is used in `reconciler.py:11-22`. This is dead code that will confuse future readers.

Fix: Remove the constant from `executor.py`.

Task: `.ai/tasks/20260518-1510-rv11-dead-terminal-statuses.md`

---

### [#12] MEDIUM — Performance — Journal re-read on every override decision
File: `src/live/executor.py:181-185`

Issue: On override runs, `list_submitted_today()` re-reads and re-parses the full journal file once per decision. For N decisions this is O(N · file_size) and grows with journal history.

Fix: Cache the scan result at the top of `execute_decisions()` and maintain a local counter per `(ticker, action)` as new orders are submitted within the batch.

Task: `.ai/tasks/20260518-1511-rv12-journal-rescan.md`

---

### [#13] MEDIUM — Safety — No abort-on-error for mid-batch failures
File: `src/live/executor.py:200-225`

Issue: If order #3 of 10 fails (after retries), orders #4-10 still attempt to submit. For a coordinated rebalance this produces a half-rotated portfolio with no clear way to recover the intended state.

Fix: Add an `abort_on_error: bool = True` parameter to `execute_decisions()`; break out of the loop and return a summary on first terminal error when enabled.

Task: `.ai/tasks/20260518-1512-rv13-abort-on-error.md`

---

### [#14] MEDIUM — Performance — Fixed retry backoff with no jitter
File: `src/broker/alpaca_client.py:18, 21-50`

Issue: `_RETRY_DELAYS = (0.5, 2.0)` is deterministic. Multiple concurrent retry attempts (e.g. from parallel-ticker live runs) will all fire at the same instant, thundering the Alpaca API.

Fix: Add randomised jitter: `delay * (1 + random.uniform(0, 0.3))`.

Task: `.ai/tasks/20260518-1513-rv14-retry-jitter.md`

---

### [#15] MEDIUM — Safety — cancel_order does not verify transition
File: `src/broker/alpaca_client.py:130-133`

Issue: `cancel_order` returns `None` on HTTP 200 but does not check whether the order actually reached `canceled` status. A cancel during a partial-fill race may return 200 while the residual quantity still fills.

Fix: Follow up with `get_order(order_id)` and surface `filled_qty` + final `status` to the caller; return a `CancelResult` with both.

Task: `.ai/tasks/20260518-1514-rv15-cancel-verify.md`

---

### [#16] MEDIUM — Correctness — Zero fill-price filter misses "0.00" strings
File: `src/live/reconciler.py:64`

Issue: `raw_price not in (None, "", "0")` only catches the exact string `"0"`. Values like `"0.00"` or `"0.000"` pass the filter and produce a `filled_avg_price` of `0.0` in the journal — a phantom zero-price fill.

Fix: Compare numerically: `price_val = float(raw_price) if raw_price not in (None, "") else None; return price_val if price_val and price_val > 0 else None`.

Task: `.ai/tasks/20260518-1515-rv16-zero-fill-price.md`

---

### [#17] LOW — Code quality — Duplicate docstring in portfolio_adapter
File: `src/broker/portfolio_adapter.py:17-23`

Issue: Two consecutive triple-quoted docstrings are present; the second is silently ignored.

Fix: Delete the duplicate `"""Convert Alpaca account + positions into a PortfolioSnapshot."""` on line 23.

---

### [#18] LOW — Code quality — Magic rounding threshold (0.4)
File: `src/live/executor.py:96`

Issue: `qty = float(math.floor(qty + 0.4))` encodes a "round if ≥ 0.6" rule via a magic addend. The comment `# 0.5→0, 0.6→1` explains intent but the expression is fragile.

Fix: Extract `_whole_share_qty(qty, threshold=0.6)` helper with the logic made explicit.

---

### [#19] LOW — Code quality — datetime imported inside method bodies
File: `src/live/audit_journal.py:33, 61`

Issue: `from datetime import datetime, timezone` is repeated inside two methods instead of at module level.

Fix: Move the import to the module top.

---

## Findings — Backtesting & Orchestration

### [#20] HIGH — Correctness — Cycle bundle written before trades execute
File: `src/orchestration/preflight.py:260-269, 322-326`

Issue: `_write_cycle_bundle` is called inside `run_cycle` immediately after the agent runs and before `BacktestEngine.execute_trade` is invoked. The bundle therefore unconditionally records `"portfolio_after": None`, `"fill_prices": None`, `"trades": []` — making commit `3360e4a`'s per-cycle logging feature inert.

Fix: Expose a `ctx.finalize_cycle(date, fill_prices, executed_trades, portfolio_after)` method on `PipelineContext`, call it from the engine after `execute_trade`, and pass real post-trade data to update the bundle file atomically.

Task: `.ai/tasks/20260518-1516-rv20-cycle-bundle-timing.md`

---

### [#21] HIGH — Correctness — run_id collision in A/B comparison
File: `src/backtesting/engine.py:174` / `src/backtesting/comparison.py:42-56`

Issue: `run_id = "-".join(tickers)+start+end` contains no discriminator. Both runs in `run_comparison()` produce the same `run_id`, so the second run's cycle bundles and manifest overwrite the first.

Fix: Include a config hash or label in `run_id` (e.g., append a slugified `cfg.label` or the flag combo `f"-regime{use_regime_selection}-weights{use_conviction_weights}"`); `RunConfig.label` is already available.

Task: `.ai/tasks/20260518-1517-rv21-run-id-collision.md`

---

### [#22] HIGH — Safety — Non-deterministic backtest (no temperature/seed)
File: `src/backtesting/engine.py:73` / `src/backtesting/controller.py:51` / `src/backtesting/cli.py:54`

Issue: `llm_temperature` defaults to `None` throughout, passing the provider's default (often 1.0) to the LLM. Re-running the same backtest produces different decisions. No `random.seed` or `numpy.random.seed` is set.

Fix: Default `llm_temperature` to `0.0` when `mode == "backtest"`; expose a `--seed` CLI flag and apply it to `random.seed()` / `np.random.seed()` at backtest start.

Task: `.ai/tasks/20260518-1518-rv22-backtest-nondeterminism.md`

---

### [#23] HIGH — Correctness — Duplicate equity-curve seed point on day 0
File: `src/backtesting/engine.py:176-180, 287`

Issue: `_portfolio_values` is initialised with `[{"Date": dates[0], "Portfolio Value": initial_capital}]`. The loop then appends a second point for `dates[0]` with the post-signal-bar NAV. `pct_change()` in Sharpe/Sortino sees a spurious zero or tiny return between the two day-0 rows, biasing risk-adjusted metrics.

Fix: Remove the seed initialisation and let the loop produce the first point naturally, or start `_portfolio_values = []` and seed it inside the loop on `i == 0`.

Task: `.ai/tasks/20260518-1519-rv23-equity-curve-seed.md`

---

### [#24] HIGH — Correctness — Dead lookback guard never fires
File: `src/backtesting/engine.py:200-205`

Issue: `if lookback_start == current_date_str: continue` compares a date shifted back 12 months to today's date — they can never be equal. This guard was presumably meant to skip dates with insufficient history but is completely inert.

Fix: Remove the dead guard or replace it with a real check (e.g., ensure at least N price bars exist before `current_date` in `_prefetched_prices`).

Task: `.ai/tasks/20260518-1520-rv24-dead-lookback-guard.md`

---

### [#25] HIGH — Performance/Cost — LLM tokens burned on NYSE holidays
File: `src/backtesting/engine.py:200, 253`

Issue: `pd.date_range(..., freq="B")` iterates Mon–Fri including US exchange holidays. On holiday dates, the missing-data guard fires only *after* `ctx.run_cycle` has already invoked the agent and burned LLM tokens. For a 252-day backtest with ~10 NYSE holidays per year, this wastes ~4% of LLM budget.

Fix: Filter `dates` against the actual trading days present in `_prefetched_prices.index` before entering the loop. No NYSE calendar dependency needed — just skip dates where no ticker has a price bar.

Task: `.ai/tasks/20260518-1521-rv25-holiday-token-waste.md`

---

### [#26] HIGH — Correctness — Portfolio buying-power check relies on implicit invariant
File: `src/backtesting/portfolio.py:77-106`

Issue: `apply_long_buy` checks `cost <= self._portfolio["cash"]` where `cash` has already been decremented by short-open margin. This is correct only because of an implicit invariant (short open debits cash). There is no `available_buying_power()` method; if that invariant ever breaks (e.g., a new `apply_*` path is added), the check silently allows over-allocation.

Fix: Add `Portfolio.available_buying_power() -> float` and use it as the single source of truth in all `apply_*` methods.

Task: `.ai/tasks/20260518-1522-rv26-buying-power-accessor.md`

---

### [#27] HIGH — Correctness — No split/dividend handling
File: `src/backtesting/engine.py:124` (`_prefetch_data`)

Issue: Price data is fetched once without explicit adjustment settings. If yfinance returns adjusted prices but `Portfolio` tracks raw share counts, a 2-for-1 split mid-backtest produces a 50% phantom loss the day shares are not doubled. If prices are unadjusted, the reverse holds. The current code documents neither choice.

Fix: (a) Explicitly request `auto_adjust=True` in `get_price_data` and document that all prices are split-adjusted, OR (b) add a split-detection loop that adjusts `Portfolio` share counts on detected split days. At minimum, add a docstring/warning.

Task: `.ai/tasks/20260518-1523-rv27-split-dividend.md`

---

### [#28] MEDIUM — Correctness — Full-day skip on single-ticker data miss
File: `src/backtesting/engine.py:215-251`

Issue: `missing_data = True; break` exits the per-ticker loop and skips the entire day for all tickers, including ones that had valid data. Open positions in healthy tickers receive no valuation point, creating equity-curve gaps that `pct_change` reads as zero-return days and that bias Sharpe/Sortino upward.

Fix: Skip only the specific ticker that is missing data; continue valuation and decision-making for the rest using their available prices.

Task: `.ai/tasks/20260518-1524-rv28-partial-data-skip.md`

---

### [#29] MEDIUM — Safety — Signal log not fsync'd
File: `src/backtesting/signal_log.py:15, 37-38`

Issue: The signal log is opened with `"a"`, `flush()`ed per record, but never `fsync`ed. A crash mid-day can leave a partial JSON line that `json.loads(line)` in the feedback labeler raises on, silently corrupting the labeling pipeline.

Fix: Call `os.fsync(self._file.fileno())` after `flush()`, or buffer the full day's records in memory and atomically append a complete newline-delimited block at day close.

Task: `.ai/tasks/20260518-1525-rv29-signal-log-fsync.md`

---

### [#30] MEDIUM — Code quality — Duplicated price-fetching logic between engine and runner
File: `src/backtesting/engine.py:217-242` / `src/live/runner.py:135-143`

Issue: Both `BacktestEngine.run_backtest` and `LiveRunner.prepare` independently build lookback windows, fetch SPY for regime, and fetch per-ticker signal prices before entering `PipelineContext`. The logic is slightly different in each, creating a maintenance surface and risk of divergence.

Fix: Add a `PriceFeed` abstraction (or `PipelineContext.fetch_signal_prices(date, lookback_days, tickers)`) that both callers delegate to, with a `BacktestPriceFeed` implementation using `_prefetched_prices` and a `LivePriceFeed` that calls `get_prices` directly.

Task: `.ai/tasks/20260518-1526-rv30-price-feed-dedup.md`

---

### [#31] MEDIUM — Correctness — Sortino +inf breaks JSON serialisation
File: `src/backtesting/metrics.py:55`

Issue: When `downside_dev == 0` and `mean_excess > 0`, Sortino is set to `float("inf")`. Standard `json.dump` raises `ValueError: Out of range float values are not JSON compliant` unless `allow_nan=True`; even when serialised with `default=str`, it becomes the string `"inf"` — silently losing the numeric type.

Fix: Return `None` when `downside_dev == 0`; handle `None` in display code as "N/A".

Task: `.ai/tasks/20260518-1527-rv31-sortino-inf.md`

---

### [#32] MEDIUM — Performance — Metrics recomputed every iteration (O(N²))
File: `src/backtesting/engine.py:303-306`

Issue: `self._perf.compute_metrics(self._portfolio_values)` is called inside the date loop, recomputing Sharpe/Sortino/max-drawdown on the full equity curve every day. For a 252-day backtest this is 252 full pandas computations — the dominant non-LLM cost.

Fix: Move the call outside the loop; compute metrics once after the loop completes. If a live progress row is needed, maintain running stats incrementally (e.g., a rolling cumulative-return tracker).

Task: `.ai/tasks/20260518-1528-rv32-metrics-on2.md`

---

### [#33] MEDIUM — Correctness — Cost-basis drift on buy side
File: `src/backtesting/portfolio.py:118-120, 185-188`

Issue: The `math.isclose(..., abs_tol=1e-9)` zero-out guard is applied on sell/cover close but not on the buy side. Accumulated floating-point errors in `long_cost_basis` across many fractional-share buys are never cleared, creating phantom cost basis that distorts realised-P&L calculations.

Fix: Apply the same `abs_tol` zero-out guard in `apply_long_buy` after decrementing `long`; or accumulate total cost rather than average cost and divide on read.

Task: `.ai/tasks/20260518-1529-rv33-cost-basis-drift.md`

---

### [#34] MEDIUM — Safety — Bare except swallows bundle/manifest write failures
File: `src/orchestration/preflight.py:343, 384, 407`

Issue: `_write_cycle_bundle` and `_write_run_manifest` wrap their entire bodies in `except Exception: logger.exception(...)`. A disk-full error or bug producing malformed JSON silently becomes a log warning; `_cycle_files` continues to grow with paths that were never actually written.

Fix: Re-raise after logging on the first failure (or surface a failure counter on `ctx.token_summary()`) so callers can detect dropped bundles rather than silently missing post-mortem data.

Task: `.ai/tasks/20260518-1530-rv34-swallowed-bundle-write.md`

---

### [#35] MEDIUM — Code quality — --days is calendar days but engine uses business days
File: `src/backtesting/cli.py:60-63`

Issue: `start_date = end - timedelta(days=args.days)` treats `--days` as calendar days, but `pd.date_range(..., freq="B")` then enumerates only business days. A user who passes `--days 30` gets approximately 22 trading days with no warning.

Fix: Rename to `--calendar-days` and add an informational note in the run header printing the resolved trading-day count; or add a `--trading-days` option with explicit business-day arithmetic.

Task: `.ai/tasks/20260518-1531-rv35-days-calendar-vs-business.md`

---

### [#36] MEDIUM — Correctness — RunConfig defaults wrong model
File: `src/backtesting/comparison.py:25-26`

Issue: `RunConfig` has `model_name: str = "deepseek/deepseek-v4-flash"` and `model_provider: str = "OpenRouter"` as defaults. A caller constructing `RunConfig` directly without specifying a model gets these hardcoded defaults silently, regardless of the active `QUORAI_*` env settings.

Fix: Remove the default values or source them from `get_settings()` inside `run_comparison()`; at minimum, document that direct construction bypasses env config.

Task: `.ai/tasks/20260518-1532-rv36-runconfig-default-model.md`

---

### [#37] LOW — Correctness — Benchmark date window misaligned
File: `src/backtesting/benchmarks.py:34`

Issue: The benchmark slice uses `self._start_date` (the backtest config start), not `self._portfolio_values[0]["Date"]` (the actual first equity-curve date). If the equity curve's first day differs, benchmark return covers a different window than strategy return, distorting relative performance.

Fix: Pass `pd.Timestamp(self._portfolio_values[0]["Date"])` as the benchmark start.

---

### [#38] LOW — Code quality — Local re-imports in _main_feedback
File: `src/backtesting/cli.py:191`

Issue: `from datetime import datetime, timedelta`, `import json`, `from pathlib import Path` are re-imported inside `_main_feedback`, shadowing identical module-level imports.

Fix: Remove the inner imports.

---

### [#39] LOW — Correctness — KeyError risk in output.py on partial-day fix
File: `src/backtesting/output.py:43-44` / `src/backtesting/engine.py:267`

Issue: `pos["long"] * current_prices[ticker]` raises `KeyError` if a ticker is absent from `current_prices`. Today this never fires because finding #28's full-day skip ensures all tickers are present; fixing #28 will expose this.

Fix: Use `current_prices.get(ticker, 0.0)` with a warning log, or assert presence at the engine boundary before calling output.

---

### [#40] LOW — Correctness — _extract_risk_manager double-collects canonical key
File: `src/orchestration/preflight.py:56-60`

Issue: `("risk_management_agent", *[k for k in analyst_signals if k.startswith("risk_management_agent_")])` — the `startswith` predicate also matches `"risk_management_agent"` (no trailing underscore) if a variant key has it as a prefix. The canonical key is collected twice; the second iteration overwrites the first in `result[ticker]`.

Fix: Use exact set membership and deduplicate: collect canonical key explicitly, then add strictly-prefixed variants via `k.startswith("risk_management_agent_")` (underscore required at end of prefix).

---

### [#41] LOW — Safety — close() writes manifest twice with inconsistent state
File: `src/orchestration/preflight.py:402-406`

Issue: `close()` calls `_write_run_manifest("completed")` (with `finished_at: None`), then reads the file back and stamps `finished_at`. Between the two writes the manifest is briefly in state `status="completed"` with `finished_at: null` — inconsistent to any concurrent reader.

Fix: Compute `finished_at = _utcnow_iso()` first, then write a single manifest with `status="completed"` and `finished_at` set together.

---

### [#42] LOW — Correctness — _spy_prices assigned in helper, not __init__
File: `src/backtesting/engine.py:163`

Issue: `self._spy_prices` is assigned inside `_prefetch_data`, not in `__init__`. Any code path that accesses the attribute before `_prefetch_data` completes (or via early-exit) will raise `AttributeError`.

Fix: Initialise `self._spy_prices: pd.DataFrame = pd.DataFrame()` in `__init__`.

---

### [#43] LOW — Performance — Run manifest rewritten every cycle (O(N²) bytes)
File: `src/orchestration/preflight.py:341`

Issue: `_write_run_manifest("running")` is called after every cycle bundle write. The manifest includes a growing `cycle_files` list, so total bytes written across a 252-day backtest is O(N²).

Fix: Either write manifest updates every K cycles (always on close), or maintain a separate append-only `cycle_index.jsonl` and only write the manifest header once.

---

### [#44] LOW — Safety — _token_usage.extend not thread-safe
File: `src/orchestration/preflight.py:257` / `src/live/runner.py` (`token_summary`)

Issue: `self._token_usage.extend(...)` is not protected by a lock. Today all live runs are sequential, but if parallel-ticker execution is enabled, concurrent `run_cycle` calls would data-race on the list.

Fix: Protect `_token_usage` mutations with a `threading.Lock`, or replace with `queue.SimpleQueue`.

---

## Test-Coverage Gaps

The following test scenarios are missing and should accompany their respective task fixes:

| Gap | Relevant Finding |
|-----|-----------------|
| `(symbol, side)` pending collision between `sell` and `short` | #03 |
| `price == 0.0` passing the notional cap | #01 |
| `KILL_SWITCH` toggled after process start, checked via `execute_decisions()` directly | #02 |
| Journal fsync survival: "submitted" row persisted despite buffered-write | #04 |
| Override-suffix collision when prior attempt ended `error`/`rejected` | #05 |
| Negative or non-finite `quantity` rejected before reaching broker | #06 |
| Unknown `side` string raises in `alpaca_client.submit_order` | #10 |
| `runner.prepare()` past 09:30 ET falls back gracefully instead of crashing | #08 |
| `run_id` differs between the two arms in `run_comparison()` | #21 |
| Cycle bundle `portfolio_after` / `fill_prices` / `trades` populated after `finalize_cycle()` | #20 |

---

## Suggested Remediation Order

1. **CRITICAL safety fixes** (#01 · #02 · #03) — smallest scope; prevent real-money risk right now.
2. **HIGH live-trading correctness** (#04 · #05 · #06 · #10 · #08) — journal durability, qty guard, side mapping, SOD fallback. Each is a focused, testable change.
3. **HIGH backtest correctness** (#20 · #21 · #22 · #23 · #24 · #27) — fix the inert cycle-bundle feature, prevent A/B overwrite, add determinism.
4. **MEDIUM batch A — live** (#07 · #09 · #13 · #15 · #16) — equity-fetch hardening and execution safety.
5. **MEDIUM batch B — quality** (#11 · #12 · #14 · #30 · #35 · #36) — dead code, retry jitter, price-feed dedup, CLI UX.
6. **MEDIUM batch C — backtest** (#25 · #28 · #29 · #31 · #32 · #33 · #34) — holiday skip, partial-data, signal-log fsync, Sortino inf, O(N²) metrics, cost-basis, swallowed writes.
7. **LOW cleanup sweep** (#17-19 · #37-44) — single PR.

---

## Appendix — Files Reviewed

| File | Findings |
|------|----------|
| `src/live/executor.py` | #03 #05 #06 #09 #11 #12 #13 #18 |
| `src/live/risk_gate.py` | #01 #02 #07 |
| `src/live/audit_journal.py` | #04 #19 |
| `src/live/runner.py` | #02 #08 |
| `src/live/sod_equity.py` | #08 |
| `src/live/reconciler.py` | #16 |
| `src/live/idempotency_guard.py` | — |
| `src/broker/alpaca_client.py` | #10 #14 #15 |
| `src/broker/portfolio_adapter.py` | #17 |
| `src/backtesting/engine.py` | #21 #22 #23 #24 #25 #27 #28 #32 #39 #42 |
| `src/backtesting/portfolio.py` | #26 #33 |
| `src/backtesting/output.py` | #39 |
| `src/backtesting/cli.py` | #22 #35 #38 |
| `src/backtesting/signal_log.py` | #29 |
| `src/backtesting/comparison.py` | #21 #36 |
| `src/backtesting/metrics.py` | #31 |
| `src/backtesting/benchmarks.py` | #37 |
| `src/orchestration/preflight.py` | #20 #34 #40 #41 #43 #44 |
| `src/config.py` | (context only) |

Files **not** reviewed in this pass: `src/agents/` (individual analyst prompts), `src/graph/`, `src/regime/`, `src/feedback/`, `src/tools/`, `src/llm/`, `src/notifications/`, `src/cli/`, `src/utils/`, `tests/`.
