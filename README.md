# Quorai

<p align="center">
  <img src="assets/quorai-logo.png" alt="Quorai Logo" width="400"/>
</p>

*A quorum of AI agents deliberating trading decisions. Pronounced "KWOR-eye" (quorum + AI).*

**[GitHub](https://github.com/quorai/quorai)** · **[About](https://quorai.github.io/)**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green)](https://github.com/langchain-ai/langgraph)
[![CI](https://github.com/quorai/quorai/actions/workflows/ci.yml/badge.svg)](https://github.com/quorai/quorai/actions/workflows/ci.yml)

A multi-agent AI trading system where specialized LLM analyst agents deliberate and vote on trading decisions through a portfolio manager. Built on [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain). For **educational purposes only** — not intended for real trading or investment.

## Quickstart

```bash
uv sync
cp .env.example .env   # add OPENROUTER_API_KEY and FINNHUB_API_KEY
uv run backtester --tickers AAPL,MSFT --model deepseek/deepseek-chat --model-provider OpenRouter
```

The `backtester` console script is installed by `uv sync`. For all options see [Usage — Backtesting](#backtesting).

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Analyst roster](#analyst-roster)
- [Architecture](#architecture)
- [Math & quantitative methods](#math--quantitative-methods)
- [Setup](#setup)
- [Usage](#usage)
  - [Backtesting](#backtesting)
  - [Live / Paper Trading](#live--paper-trading)
  - [Telegram approval gate](#telegram-approval-gate)
- [Safety mechanisms](#safety-mechanisms)
- [Project structure](#project-structure)
- [Running tests](#running-tests)
- [Adding an analyst](#adding-an-analyst)
- [Troubleshooting](#troubleshooting)
- [Python version](#python-version)
- [Changelog](#changelog)
- [Disclaimer](#disclaimer)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Features

- **25 analyst agents** — value, growth, macro, technical, fundamentals, sentiment, risk, and more
- **Famous investor personas** — simulations of Buffett, Munger, Ackman, Burry, Wood, Dalio, Simons, Lynch, and others
- **Multi-provider LLM support** — OpenAI, Anthropic, Groq, Gemini, DeepSeek, xAI, OpenRouter, Ollama (local)
- **Backtesting engine** — replay historical data with full agent deliberation and portfolio metrics
- **Live / paper trading** — execute via Alpaca with optional Telegram approval gate
- **Group-level debate node** — collapses 25 analyst signals into 6 strategy groups via confidence-weighted aggregation; an LLM moderator summarises contested tickers
- **Market-regime selection** — classifies the current SPY regime (bull/bear/risk-off/neutral) each day and narrows the active analyst subset accordingly
- **Conviction-weight feedback loop** — tracks each agent's rolling directional hit-rate; high-accuracy agents receive proportionally more weight in the debate aggregation
- **Signal logging + forward-return labeling** — persists every per-agent-per-ticker signal to JSONL during a backtest run; a separate labeler attaches 1d/5d/20d forward returns so hit-rates can be computed
- **Token-usage telemetry** — captures and accumulates LLM token counts per agent across the full backtest run; Anthropic prompt caching is applied automatically and cache-read/creation tokens are surfaced separately
- **A/B comparison harness** — runs two backtest configs back-to-back and prints a side-by-side metrics table (full-vs-regime analysts, uniform-vs-conviction weights)
- **Per-agent model routing** — override model and provider per analyst via `--agent-model AGENT=model/PROVIDER`; handled by `RunRequest` (`src/llm/request.py`)
- **Parallel per-ticker execution** — set `QUORAI_PARALLEL_TICKERS=N` to run N tickers concurrently via a thread pool (`src/utils/concurrency.py`)
- **SEC EDGAR fundamentals** — point-in-time XBRL data via a local SQLite store (`.cache/sec_fundamentals.db`); eliminates yfinance look-ahead bias on historical share counts and financial statements. Seed with `experiments/seed_sec_fundamentals.py`; falls through to yfinance for unseeded tickers.
- **Regime-gated allocation** — the portfolio manager deterministically filters proposed LLM actions by the detected SPY regime: `bull_trend` removes `short` when quant/growth groups are bullish; `bear_trend` removes `buy` when quant/quality groups are bearish; `risk_off` blocks both `buy` and `short`

## How it works

```
Market Data → Analyst Agents → Portfolio Manager → Order Execution
                  ↑                   ↑
           (LangGraph nodes)   (deliberation graph)
```

<p align="center">
  <img src="assets/flow-diagram.png" alt="Quorai pipeline diagram" width="700"/>
</p>

Each trading cycle:
1. Financial data is fetched (price, fundamentals, news, macro indicators)
2. Each analyst agent runs as a LangGraph node and produces a signal with reasoning
3. A portfolio manager agent weighs the signals and issues buy/hold/sell orders
4. Orders are executed via Alpaca (live) or simulated (backtest)

## Analyst roster

| Category | Agents |
|---|---|
| Value | Buffett, Munger, Ackman, Burry, Greenblatt, Pabrai, Damodaran |
| Growth | Cathie Wood, Phil Fisher, Peter Lynch, Jhunjhunwala |
| Macro | Dalio, Druckenmiller, Marks |
| Quant | Simons, Asness, Seykota |
| Sentiment | News sentiment, social sentiment |
| Risk | Risk manager, Taleb (tail-risk) |
| Special | Bull/bear debate node |

<p align="center">
  <img src="assets/agents-groups.png" alt="Analyst strategy groups" width="700"/>
</p>

## Architecture

The pipeline runs as a LangGraph `StateGraph`: `start_node` fans out to all selected analyst nodes in parallel, feeds into a `debate_node` (conviction-weighted group aggregation), then `risk_management_agent` (pure maths, no LLM), then `portfolio_manager` (LLM decision). Regime selection and conviction-weight loading happen in `src/orchestration/preflight.py:PipelineContext` before the graph is invoked each day.

```
start_node → [analyst_1 … analyst_25] → debate_node → risk_management_agent → portfolio_manager → END
```

### Debate node

The debate node (`src/agents/debate_node.py`) runs in two phases:

1. **Group aggregation (deterministic).** The 25 analyst signals are collapsed into 6 strategy groups: `deep_value`, `growth_and_catalyst`, `macro_and_cycle`, `quant_systematic`, `quality_compounders`, `sentiment_and_analytical`. Within each group, signals are **confidence-weighted** (not majority-voted): each agent's stance (`bullish` → +1, `neutral` → 0, `bearish` → −1) is multiplied by its confidence (and optionally by its conviction weight from `weights.json`), then averaged.

   | Weighted stance | Group signal |
   |---|---|
   | ≥ +0.25 | bullish |
   | ≤ −0.25 | bearish |
   | otherwise | neutral |

2. **Moderator synthesis (LLM).** Only for *contested* tickers — at least one bullish group AND at least one bearish group — an LLM moderator receives the group stances and their top-2 arguments, and returns a `DebateSummary` with each group's one-sentence position, the root structural disagreement, and a `consensus_strength` label (`strong_agreement` / `mixed` / `structural_split`).

Individual agents do not argue with each other; the "debate" is between the six group-level positions, and only contested tickers incur the extra LLM call.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design — data layer, LLM dispatch, backtesting internals, regime classifier, conviction-weight feedback loop, token telemetry, live trading layer, and per-ticker parallelism.

## Math & quantitative methods

This section documents the quantitative formulas used throughout the codebase. All annualisation uses 252 trading days.

### Portfolio metrics (`src/backtesting/metrics.py`)

| Metric | Formula |
|---|---|
| Daily return | `(price_t − price_{t−1}) / price_{t−1}` |
| Excess return | `daily_return − risk_free_rate / 252`  (RF = 4.34% annual) |
| Sharpe ratio | `√252 × mean(excess) / std(excess)` |
| Sortino ratio | `√252 × mean(excess) / √mean(min(excess, 0)²)` |
| Max drawdown | `(value_t − max(value_{0..t})) / max(value_{0..t})` — tracked as a running peak |
| Total return | `(final_value / initial_capital − 1) × 100%` |
| Benchmark return | `(SPY_last / SPY_first − 1) × 100%` (buy-and-hold over the same window) |
| Alpha vs SPY | `strategy_total_return − SPY_total_return` |
| Alpha vs basket | `strategy_total_return − equal_weight_basket_total_return` |
| Information ratio vs SPY | `√252 × mean(daily_active_return) / std(daily_active_return)` where `active_return = strategy_return − SPY_return` |

### Portfolio exposure (`src/backtesting/valuation.py`)

| Metric | Formula |
|---|---|
| NAV | `cash + Σ(long_shares × price) − Σ(short_shares × price)` |
| Long exposure | `Σ(long_shares × price)` |
| Short exposure | `Σ(short_shares × price)` |
| Gross exposure | `long + short` |
| Net exposure | `long − short` |
| L/S ratio | `long / short` |
| Weighted-average cost basis | `(old_basis × old_qty + new_price × new_qty) / total_qty` (updated on every fill) |

### Position sizing (`src/agents/risk_manager.py`)

The risk manager derives a per-ticker notional limit in two steps:

1. **Volatility adjustment** — annualised vol `= std(60-day returns) × √252`:

   | Annualised vol | Multiplier |
   |---|---|
   | < 15% | 1.25× |
   | 15–30% | `1.0 − (vol − 0.15) × 0.5` |
   | 30–50% | `0.75 − (vol − 0.30) × 0.5` |
   | > 50% | 0.50× |

2. **Correlation adjustment** — average correlation with existing open positions:

   | Avg correlation | Multiplier |
   |---|---|
   | ≥ 0.80 | 0.70× |
   | 0.60–0.80 | 0.85× |
   | 0.40–0.60 | 1.00× |
   | 0.20–0.40 | 1.05× |
   | < 0.20 | 1.10× |

   Final limit: `base_limit × vol_multiplier × corr_multiplier`

   Cash/margin constraints are applied last: `max_long = min(position_limit, available_cash)`, `max_short = min(position_limit, available_margin)` where `available_margin = equity / margin_requirement − margin_used`.

### Valuation models (`src/agents/valuation.py`)

**Owner earnings (Buffett)**
`owner_earnings = net_income + D&A − capex − Δworking_capital`
Projected forward for 10 years then discounted; terminal value uses a Gordon Growth model. A 25 % margin of safety is applied to the resulting intrinsic value.

**DCF (free cash flow)**
`intrinsic = Σ_{t=1}^{n} FCF_t / (1+r)^t + TV / (1+r)^n`  
Terminal value: `TV = FCF_n × (1 + g_terminal) / (r − g_terminal)`

**Multi-stage DCF** — three growth phases discounted at WACC, with a quality adjustment:
`quality_factor = max(0.7, 1 − fcf_volatility × 0.5)` where `fcf_volatility = std(FCF) / mean(FCF)` (coefficient of variation).
A scenario overlay applies bear/base/bull growth assumptions weighted 20 / 60 / 20 %.

**EV/EBITDA cross-check**
`implied_equity = median_sector_EV/EBITDA × current_EBITDA − net_debt`

**Residual income (Edwards-Bell-Ohlson)**
`RI_t = net_income_t − cost_of_equity × book_value_{t−1}`
`intrinsic = book_value + Σ PV(RI_t) + PV(terminal_RI)`

**WACC**
`cost_of_equity = RF + β × MRP`  (RF = 4.5 %, MRP = 6 %, β from TTM metrics)  
`cost_of_debt = max(RF + 0.01, RF + 10 / interest_coverage)`  
`WACC = (E/V) × CoE + (D/V) × CoD × (1 − 0.25)`, floored at 6 % and capped at 20 %.

**Blended signal**
The four methods are weighted DCF 35 %, Owner Earnings 35 %, EV/EBITDA 20 %, Residual Income 10 %. The resulting valuation gap `= (weighted_intrinsic − market_cap) / market_cap` drives the bullish/neutral/bearish signal (thresholds ±15 %).

### Technical indicators (`src/agents/technicals.py`, `src/agents/jim_simons.py`)

| Indicator | Formula / Definition |
|---|---|
| EMA | `close.ewm(span=N, adjust=False).mean()` |
| RSI | `100 − 100 / (1 + avg_gain / avg_loss)` over 14 periods |
| Bollinger Bands | `SMA(20) ± 2 × σ(20)` |
| Z-score | `(price − MA) / σ` — signals at ±2 |
| ADX | `EWM(DX)` where `DX = 100 × |DI+ − DI−| / (DI+ + DI−)`, `DI± = 100 × smoothed_DM± / smoothed_TR` |
| ATR | `SMA(true_range, 14)` where `TR = max(H−L, |H−C_{prev}|, |L−C_{prev}|)` |
| Momentum (1/3/6 m) | `returns.rolling(21/63/126).sum()`, blended as `0.4×mom_1m + 0.3×mom_3m + 0.3×mom_6m` |
| 12-1 momentum (AQR) | `(price_{−21d} − price_{−252d}) / price_{−252d}` — skips the most recent month to avoid short-term reversal |
| Hurst exponent | OLS slope of `log(lag)` vs `log(std(returns at lag))`; H < 0.5 → mean-reverting, H > 0.5 → trending |
| Lag-1 autocorrelation | `corr(returns[:-1], returns[1:])` — negative ACF supports mean-reversion entry |
| Volume spike | `current_volume / SMA(volume, 21)` — > 2× on a down day flags potential capitulation |

The final technical signal is a weighted sum: Trend 25 %, Mean Reversion 20 %, Momentum 25 %, Volatility 15 %, Stat-Arb 15 %; mapped to bullish/bearish via a ±0.2 threshold.

### AQR multi-factor scoring (`src/agents/cliff_asness.py`)

| Factor | Key sub-signals | Max pts |
|---|---|---|
| Value | P/E, P/B, FCF yield, EV/EBITDA vs thresholds | 8 |
| Momentum | 12-1 momentum vs ±5 % / ±20 % thresholds; −1 if 1-month gain > 15 % | 4 |
| Quality | ROIC, gross margin, earnings stability (% positive years) | 6 |
| Low volatility | 63-day annualised vol bucketed into five tiers | 4 |

Overall signal strength scales with how many factors align simultaneously (max 22 pts → 90–100 % confidence).

### Conviction-weight feedback loop (`src/feedback/`)

After each run, signals are labeled with 1 d / 5 d / 20 d forward returns. The rolling directional hit-rate for each agent is used to upweight high-accuracy agents in the debate aggregation. Weights are persisted in `src/feedback/weights.json` and reloaded at the start of the next run when `--use-conviction-weights` is set.

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```bash
# At least one LLM provider is required
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...
OPENROUTER_API_KEY=...

# Financial data (Finnhub)
FINNHUB_API_KEY=...
```

### 3. Seed SEC fundamentals (recommended for accurate backtests)

The fundamentals tools consult a local SQLite cache of SEC EDGAR XBRL data before falling back to yfinance. Without this cache yfinance's `.info` returns *current* share counts and financial metrics for historical dates, introducing look-ahead bias in backtests.

```bash
# Required env var — SEC fair-access policy requires a contact identifier
export QUORAI_SEC_USER_AGENT="your.email@example.com"

# Seed a specific subset (fast, ~10-30 s)
uv run python experiments/seed_sec_fundamentals.py --tickers AAPL,MSFT,NVDA

# Seed the full US market (~10 000 tickers, 3-4 hours, 5-10 GB)
uv run python experiments/seed_sec_fundamentals.py

# Skip tickers last synced within N days
uv run python experiments/seed_sec_fundamentals.py --refresh-older-than 30

# Dry run — print what would be downloaded without writing
uv run python experiments/seed_sec_fundamentals.py --dry-run --tickers AAPL
```

The seeder respects the SEC's 10 req/s rate limit automatically. Tickers not in the cache are silently fetched from yfinance at run-time.

## Usage

### Backtesting

```bash
uv run backtester \
    --tickers AAPL,MSFT \
    --model deepseek/deepseek-chat \
    --model-provider OpenRouter \
    --show-reasoning
```

You can also invoke the module directly: `uv run python -m src.backtesting`.

Key flags:
- `--tickers` — comma-separated list of tickers (required)
- `--model` — model name (required)
- `--model-provider` — provider string; bypasses catalog, accepts any OpenRouter/provider slug
- `--analysts` — comma-separated analyst IDs (default: all)
- `--start-date` / `--end-date` — YYYY-MM-DD (default: last month → today)
- `--initial-capital` — starting cash (default: 100 000)
- `--show-reasoning` — print each agent's reasoning
- `--temperature` — LLM temperature override
- `--use-regime-selection` — classify SPY regime per day and narrow analysts to the relevant group
- `--use-conviction-weights` — weight agents by rolling directional hit-rate (requires `src/feedback/weights.json` from a prior scored run)
- `--risk-profile` — choose one of five risk presets: `conservative`, `cautious`, `balanced` (default), `aggressive`, `speculative`. Controls per-ticker position sizing and notional/loss-limit caps together.
- `--agent-model AGENT=model/PROVIDER` — override model for a specific analyst; repeatable; use `*=model/PROVIDER` to override all agents
- `--log-dir` — override artifact directory (default: `logs/backtest`)
- `--run-label` — tag embedded in `run_id` and manifest for later filtering

#### A/B comparison

```bash
uv run backtester compare \
    --tickers AAPL,MSFT \
    --model deepseek/deepseek-chat \
    --model-provider OpenRouter \
    --mode regime    # full analysts vs regime subset

# --mode weights    uniform weights vs conviction weights
# --mode both       run both comparisons sequentially
```

#### Reading the backtest output

The engine prints two blocks of output.

**Rolling `PORTFOLIO SUMMARY`** — printed at the end of each trading day:

```
PORTFOLIO SUMMARY:
Cash Balance: $136,342.28        ← uninvested cash
Total Position Value: $-36,520.48  ← market value of open positions (negative = net short)
Total Value: $99,821.80          ← cash + position value = current portfolio NAV
Portfolio Return: -0.18%         ← return vs starting capital since inception
Benchmark Return: +0.77%         ← S&P 500 return over the same period
Sharpe Ratio: -4.58              ← risk-adjusted return (annualised); see below
Sortino Ratio: -6.01             ← like Sharpe but only penalises downside volatility
Max Drawdown: -0.74%             ← largest peak-to-trough decline so far
```

**`ENGINE RUN COMPLETE`** — printed once at the very end, using the final metrics:

```
ENGINE RUN COMPLETE
Total Return: -0.18%
Sharpe: -2.25
Sortino: -3.00
Max DD: 0.74% on 2026-05-08
SPY Return: +1.12%           ← buy-and-hold SPY over the same window
Alpha vs SPY: -1.30%         ← strategy total return − SPY total return
Alpha vs Basket: -0.45%      ← strategy total return − equal-weight ticker basket
IR vs SPY: -0.83             ← (daily active return mean / std) × √252
```

> The Sharpe/Sortino in the final summary may differ slightly from the last rolling figure because the two blocks use marginally different timing for their calculation windows.

**Interpreting the metrics**

| Metric | Good | Acceptable | Poor |
|---|---|---|---|
| Portfolio Return | Beats benchmark | Roughly flat vs benchmark | Lags benchmark |
| Sharpe Ratio | > 1.0 | 0 – 1.0 | < 0 |
| Sortino Ratio | > 1.5 | 0 – 1.5 | < 0 |
| Max Drawdown | < 10% | 10 – 20% | > 20% |
| Alpha vs SPY | > 0% | ~ 0% | < 0% |
| IR vs SPY | > 0.5 | 0 – 0.5 | < 0 |

**Important caveats for short backtests**

- Sharpe and Sortino are annualised from daily returns. With only a few days of data there are too few samples for the figures to be statistically meaningful — treat them as noise until the test window covers at least several months.
- A negative `Total Position Value` means the portfolio manager issued net short orders. This is valid behaviour but unusual; check `--show-reasoning` to understand why.
- Always compare against the benchmark return over the *same* period before drawing conclusions.

### Live / Paper Trading

```bash
uv run python src/live_trading.py \
    --tickers AAPL,MSFT,NVDA \
    --model openrouter/anthropic/claude-3.5-sonnet \
    --model-provider OpenRouter \
    --dry-run
```

Key flags:
- `--tickers` — comma-separated list of tickers (required)
- `--model` — model name (required)
- `--model-provider` — provider string (required)
- `--analysts` — comma-separated analyst IDs to include (default: all)
- `--use-regime-selection` — classify today's SPY regime and narrow analysts to the matching strategy groups (same logic as `BacktestEngine`)
- `--use-conviction-weights` — apply per-agent conviction weights from `src/feedback/weights.json`; warns if the file is absent but does not abort
- `--risk-profile` — choose one of five risk presets: `conservative`, `cautious`, `balanced` (default), `aggressive`, `speculative`. Controls per-ticker position sizing and RiskGate caps (notional, quantity, daily loss limit) together. See the safety table below for the values per preset.
- `--no-signal-log` — disable writing `logs/signals-live-YYYY-MM-DD.jsonl` (signal logging is on by default)
- `--dry-run` — print decisions without submitting orders
- `--confirm` — skip interactive confirmation prompt
- `--require-approval` — send orders to Telegram for human approval before submitting
- `--auto-submit` — submit immediately and send an execution report to Telegram afterwards
- `--margin-requirement` — margin requirement fraction (default: 0.0)
- `--temperature` — LLM temperature override

After each run the console prints:

```
Signal log: logs/signals-live-2026-05-12.jsonl
Tokens: 12 calls, 84 200 in / 3 100 out
```

The signal JSONL feeds the same `feedback/labeler.py → scorer.py → weights.json` pipeline used in backtesting, so conviction weights improve over time as live-run history accumulates.

### Telegram approval gate

When `--require-approval` is set and `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` are configured, each run sends the proposed orders as an inline message. Tap **Approve ✅** or **Reject ❌** within the timeout window (default 30 min) to decide whether orders are submitted.

Required `.env` keys:

```bash
TELEGRAM_BOT_TOKEN=...        # BotFather token
TELEGRAM_CHAT_ID=...          # your chat / group ID
TELEGRAM_APPROVAL_TIMEOUT_SECONDS=1800  # optional, default 30 min
```

#### Bot command inbox

You can send plain-text messages to the bot at any time. They are read at the start of the **next** run and take effect immediately:

| Message (case-insensitive) | Effect |
|---|---|
| `accept only sales` / `only sells` | Suppress all buy orders for the next run only |
| `skip next day` / `skip next` | Skip the next scheduled run entirely |
| `pause` / `stop trading` / `skip until continue` | Pause all runs until you send `continue` |
| `continue` / `resume` | Clear an active pause |

The bot replies with a confirmation message when a command is recognised. Command state is persisted in `logs/command_state.json` so it survives process restarts and cron jobs.

## Safety mechanisms

The Alpaca client (`src/broker/alpaca_client.py:66-67`) refuses to construct a live-trading client
unless `ALPACA_PAPER=True`, making this paper-trading software by construction. Within that sandbox,
multiple caps apply:

| Layer | Source | Default (`balanced`) |
|---|---|---|
| Per-ticker volatility cap | `src/agents/risk_manager.py` | 5–25% of NAV (lower for high-vol or correlated names) |
| Cycle-wide cash guard | `src/agents/portfolio_manager.py:111-149` | Cumulative buys across all tickers cannot exceed available cash |
| Backtest cash invariant | `src/backtesting/portfolio.py:82-106` | Over-budget buys truncated to `cash / price` |
| Per-order notional cap | `src/live/risk_gate.py` (`MAX_ORDER_NOTIONAL`) | $10,000 |
| Per-order quantity cap | `src/live/risk_gate.py` (`MAX_ORDER_QTY`) | 1,000 shares |
| Daily loss limit | `src/live/risk_gate.py` (`DAILY_LOSS_LIMIT_PCT`) | 5% of start-of-day equity |
| Kill switch | `src/config.py` (`KILL_SWITCH`) | Off by default; flip to reject all orders immediately |
| Telegram approval gate (opt-in) | `src/live_trading.py` (`--require-approval`) | Fail-closed: missing creds, Telegram error, reject, or timeout all abort with zero orders submitted |
| Prior-run idempotency re-prompt | `src/live/idempotency_guard.py:34` (`TelegramPriorRunApprover`) | Re-asks via Telegram if today already has submissions; fail-closed if Telegram unreachable |

The three `RiskGate` caps and the position-sizing `base_limit` are bundled into five presets selectable via `--risk-profile`:

| Profile | `base_limit` | Notional cap | Qty cap | Daily loss limit |
|---|---|---|---|---|
| `conservative` | 10% | $5,000 | 500 shares | 2% |
| `cautious` | 15% | $7,500 | 750 shares | 3% |
| `balanced` *(default)* | 20% | $10,000 | 1,000 shares | 5% |
| `aggressive` | 30% | $20,000 | 2,000 shares | 8% |
| `speculative` | 50% | $50,000 | 5,000 shares | 15% |

Individual caps are still overridable via env vars (see `src/config.py`). The `--risk-profile` flag takes precedence over the env defaults for that run only.

### Known limitations

- **Notional cap is per-order, not per-cycle.** With N tickers, up to `N × $10,000` of orders can be submitted in a single cycle before any cap fires.
- **No portfolio-level concentration cap.** Four low-vol uncorrelated names can each hit the 25% per-ticker ceiling and effectively go all-in across the basket.
- **Daily loss limit re-baselines if SOD equity is missing.** If `logs/sod_equity.json` is absent at run-time (e.g. after a crash), `src/live/runner.py:99-103` resets the baseline to current (already drawn-down) equity, defeating the limit for that day.
- **Sub-$1 fractional buys are silently dropped.** `src/live/executor.py:83` rounds `qty` to 3 decimals; a tiny allocation on a high-priced stock rounds to `0.000` and is classified as `skipped` with no warning.
- **No `fractionable` pre-check in the Alpaca client.** Fractional `qty` on a non-fractionable asset fails after order submission rather than being caught early (`src/broker/alpaca_client.py:118-128`).
- **Backtest silent truncation.** Over-budget buys in `src/backtesting/portfolio.py:93-105` partially fill without any log line, which can mask LLM or risk-manager miscalculations in backtest results.

The paper-only hard-stop in `alpaca_client.py` is the base safety net. Running with `--require-approval` adds a human-in-the-loop gate on top of it. The limitations above are documented, not fixed.

## Project structure

| Path | Purpose |
|---|---|
| `src/main.py` | `run_quorai()` — single-run entry point; `create_workflow()` — LangGraph builder |
| `src/live_trading.py` | Live/paper trading CLI entry point |
| `src/agents/` | 25 analyst agents (personality + quant) plus risk manager and portfolio manager |
| `src/backtesting/` | Engine, portfolio, metrics, CLI (`backtester` / `python -m src.backtesting [compare]`), signal log, A/B harness |
| `src/orchestration/` | `PipelineContext` — pre-graph helper shared by live and backtest |
| `src/regime/` | `MarketRegime` classifier + analyst-selection policy |
| `src/feedback/` | Forward-return labeler, rolling per-agent scorer, weights loader |
| `src/broker/` | `Broker` protocol + Alpaca client |
| `src/live/` | Live executor, runner, risk gate, audit journal |
| `src/notifications/` | Telegram approval client + command store |
| `src/data/` | Disk-persisted cache (`cache.py`), Pydantic data models, SEC EDGAR XBRL store (`sec_store.py`) |
| `src/llm/` | Multi-provider LLM dispatch, OpenRouter catalog |
| `src/utils/` | Analyst registry (`ANALYST_CONFIG`), shared helpers |
| `src/config.py` | Centralised env-var config via pydantic-settings |
| `experiments/run_scenarios.py` | Regime evaluation harness — sweeps 10 period × ticker-set scenarios and writes a markdown report |
| `experiments/seed_sec_fundamentals.py` | Seeds `.cache/sec_fundamentals.db` from SEC EDGAR XBRL; requires `QUORAI_SEC_USER_AGENT` |
| `tests/` | Unit and integration tests |

## Running tests

```bash
uv run python -m pytest
```

> Use `python -m pytest`, not `uv run pytest` — the latter invokes a stale venv shebang that resolves to the wrong Python.

## Adding an analyst

1. Create `src/agents/my_analyst.py` with a `my_analyst_agent(state, agent_id)` function.
2. Register it in `src/utils/analysts.py` — add an entry to `ANALYST_CONFIG`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for full contribution guidelines.

## Troubleshooting

**`uv run pytest` fails or runs the wrong Python**  
Use `uv run python -m pytest` instead.

**`--use-conviction-weights` warns about a missing `weights.json`**  
Conviction weights are computed from a prior backtest's signal log. Run a backtest first, then label and score the output:

```python
from src.feedback.labeler import label_signals
from src.feedback.scorer import compute_weights

labeled = label_signals("logs/signals-<run-id>.jsonl", prices_df)
compute_weights(labeled)  # writes src/feedback/weights.json
```

Then re-run with `--use-conviction-weights`.

**Sharpe / Sortino look extreme on a short backtest**  
Both ratios are annualised from daily returns. A handful of data points isn't statistically meaningful — use a test window of at least several months before drawing conclusions.

**Live trading fails to connect to Alpaca**  
Ensure `ALPACA_API_KEY`, `ALPACA_API_SECRET`, and `ALPACA_BASE_URL` are set in `.env`. For paper trading, `ALPACA_BASE_URL` should be `https://paper-api.alpaca.markets`.

**Telegram approval bot doesn't respond**  
Ensure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set. Verify the bot is started (`/start`) and you are in the correct chat. The default timeout is 30 minutes — increase `TELEGRAM_APPROVAL_TIMEOUT_SECONDS` if needed.

## Python version

Python 3.11+ required (`>=3.11` in `pyproject.toml`; `.python-version` pins 3.11). CI runs 3.12.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Creator assumes no liability for financial losses
- Past performance does not indicate future results

The agent modules named after real investors (Buffett, Munger, Ackman, Burry, Wood, Asness, Dalio, Marks, Simons, Druckenmiller, Seykota, Greenblatt, Damodaran, Fisher, Lynch, Jhunjhunwala, Pabrai, Taleb, and others) are **educational simulations** that approximate publicly stated investment philosophies derived from books, interviews, and public writings. They are not affiliated with, endorsed by, or representative of the actual individuals or their organisations.

## Acknowledgements

- **[virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)** — inspiration for the initial persona-agent architecture and LLM prompt design.
- **[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)** — the bull/bear debate concept that inspired `src/agents/debate_node.py`.
- **[Finnhub](https://finnhub.io/)** — insider trades and company news API.
- **[yfinance](https://github.com/ranaroussi/yfinance)** — prices, financial metrics, and fundamental data.
- **[Alpaca](https://alpaca.markets/)** — paper and live trading API.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
Third-party notices (including the upstream MIT license for ai-hedge-fund material) are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
