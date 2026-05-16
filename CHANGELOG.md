# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
