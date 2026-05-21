# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **SEC EDGAR fundamentals data layer** (`src/data/sec_store.py`, `src/tools/_sec_fundamentals.py`,
  `experiments/seed_sec_fundamentals.py`): point-in-time XBRL Company Facts stored in a local
  SQLite database at `.cache/sec_fundamentals.db`. Eliminates yfinance look-ahead bias on
  historical share counts and financial statements. Requires `QUORAI_SEC_USER_AGENT` env var
  (SEC fair-access policy). Seed via:
  `QUORAI_SEC_USER_AGENT="email" uv run python experiments/seed_sec_fundamentals.py --tickers AAPL`
- **Regime-gated portfolio allocation**: `compute_allowed_actions(regime, group_signals)` in
  `src/agents/portfolio_manager.py` deterministically filters the LLM action space before the
  portfolio-manager LLM call. Rules: `bull_trend` removes `short` when quant/growth groups are
  bullish; `bear_trend` removes `buy` when quant/quality groups are bearish; `risk_off` blocks
  both `buy` and `short`. `cover` and `sell` are never blocked.
- **Local LLM provider (Ollama)** via `langchain_ollama.ChatOllama` (`ModelProvider.LOCAL`).
  Requires a running Ollama daemon; no API key needed.
- **Regime evaluation harness** (`experiments/run_scenarios.py`): sweeps 10 curated
  period × ticker-set scenarios across BULL/BEAR/RISK_OFF/NEUTRAL regimes. Streams subprocess
  output live; writes a markdown summary to `experiments/results/eval-<date>.md`. Supports
  `--max-tickers` / `--max-days` smoke truncation and `--no-regime-selection` /
  `--no-conviction-weights` toggles.
- **Backtest CLI flags**: `--log-dir` (override artifact root; default `logs/backtest`) and
  `--run-label` (tag embedded in `run_id` and manifest for later filtering).
- **Performance metrics**: alpha vs SPY, alpha vs equal-weight ticker basket, and information
  ratio vs SPY (`√252 × mean(active_daily_return) / std(active_daily_return)`) reported in
  ENGINE RUN COMPLETE summary and returned from `engine.get_metrics()`.
- **SPY + per-ticker buy-and-hold baselines** printed in ENGINE RUN COMPLETE summary.

### Changed

- `src/tools/_yfinance_fundamentals.py`: `fetch_statements()` and `fetch_market_cap()` now
  consult `SecStore` first; fall through to yfinance only for unseeded tickers. Deduplicates
  repeated TTM-unavailable warnings via a `_ttm_warned` set.
- **Live and backtest logs separated**: backtest artifacts go to `logs/backtest/{cycles,runs,signals}/`
  (was `logs/cycles/`, `logs/runs/`). `run_id` prefixed with execution date and config fingerprint
  (`YYYY-MM-DD-<hash>`) to prevent run overwrites.
- `display()` context-manager replaces ad-hoc `progress.start()` / `progress.stop()` +
  `set_header()` calls throughout the codebase.

### Fixed (RV-23 … RV-36)

- Sortino ratio returns `None` instead of `+inf` when there is no downside deviation.
- NYSE holiday skipping via price-index date filter (backtest engine no longer runs on market
  holidays).
- `SignalLogger` calls `fsync` after each `log_day` to prevent data loss on crash.
- Zero-out guard for cost-basis on buy side (prevents stale cost-basis carry-over).
- Centralised signal-price fetching in `PriceFeed` (removes duplicated per-module price lookups).
- Bundle and manifest write failures are now counted and surfaced rather than silently swallowed.
- `--calendar-days` alias added; resolved trading-day count is logged at run start.
- `RunConfig` defaults sourced from `Settings` instead of hardcoded strings.
- Performance: `compute_metrics` hoisted out of the per-day backtest loop.
- Per-ticker failures skip only the failing ticker rather than the entire trading day.
- Split / dividend-adjustment contract documented and asserted (`auto_adjust=True`).
- Duplicate equity-curve seed point removed from backtest initialisation.
- Dead lookback guard removed (stale defensive code that was no longer reachable).

## [1.0.0] — 2026-05-16

### Added

- 25 analyst agents: 13 investor personas (Buffett, Munger, Ackman, Burry, Wood, Dalio, Simons,
  Druckenmiller, Lynch, Fisher, Greenblatt, Damodaran, Jhunjhunwala, Marks, Pabrai, Seykota,
  Asness, Taleb) + dedicated fundamentals, technical, growth, valuation, sentiment, and
  news-sentiment analysts
- LangGraph `StateGraph` pipeline: `start_node → [25 analysts in parallel] → debate_node →
  risk_management_agent → portfolio_manager → END`
- Backtesting engine with next-day-open fill pricing (no lookahead bias), portfolio metrics
  (Sharpe, Sortino, max drawdown), and SPY benchmark comparison
- Live/paper trading via Alpaca — `--dry-run`, `--require-approval`, `--auto-submit`
- Multi-provider LLM dispatch: OpenAI, Anthropic, Groq, Gemini, DeepSeek, xAI, OpenRouter,
  Azure OpenAI, and more (13 providers via `src/llm/models.py`)
- Market-regime classifier (SPY-based: BULL_TREND / BEAR_TREND / RISK_OFF / NEUTRAL) — narrows
  active analyst set per day
- Conviction-weight feedback loop: per-agent rolling directional hit-rate → `weights.json` →
  debate aggregation weighting
- Signal logging (JSONL per agent+ticker) + forward-return labeler (1d/5d/20d)
- Token-usage telemetry with Anthropic prompt-caching support (cache-read / cache-creation tokens)
- A/B comparison harness (`backtester compare --mode regime|weights|both`)
- Per-agent model routing (`--agent-model AGENT=model/PROVIDER`)
- Parallel per-ticker execution via thread pool (`QUORAI_PARALLEL_TICKERS`)
- Telegram approval gate with inline Approve/Reject buttons and bot command inbox
- `backtester` console script installed by `uv sync`
- CI (GitHub Actions): ruff lint, ruff format check, mypy, pytest on every push/PR

### Fixed (correctness hardening rounds 1–10, R1–R56)

- LangGraph `O(N²)` message-list growth — messages channel now uses concat annotation correctly
- Short-margin double-subtraction in portfolio: margin was deducted twice per short trade
- Missing `_MIN_PRICES=126` guard on thin price series causing `IndexError` in technical agents
- `transaction_shares=None` crash on SEC Form-4 value-only insider-trade filings
- Capex sign normalisation: yfinance returns capex as negative; agents now negate before use
- Damodaran FCFF: net-debt subtracted correctly from enterprise value → equity value
- ROIC positive-invested-capital guard: division by zero on companies with negative book equity
- RSI flat-series guard: avoid divide-by-zero when all price changes are zero
- Spurious `sqrt()` in Hurst exponent calculation removed
- Jhunjhunwala loss-year inclusion fixed: prior logic excluded years with negative earnings
- Broker `Protocol` extended with `get_order`, `list_orders`, `client_order_id` methods
- `AuditJournal.record_reconciliation` added for live-trading post-fill reconciliation
- 56+ additional targeted fixes with accompanying regression tests (see commits `5a23ee6`..`ce2110a`)
