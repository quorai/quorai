# Quorai

<p align="center">
  <img src="assets/logo-detailed.jpeg" alt="Quorai Logo" width="400"/>
</p>

*A quorum of AI agents deliberating trading decisions. Pronounced "KWOR-eye" (quorum + AI).*

**[GitHub](https://github.com/nils-fl/quorai)** · **[About](https://nils-fl.github.io/quorai/)**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green)](https://github.com/langchain-ai/langgraph)

A multi-agent AI trading system where specialized LLM analyst agents deliberate and vote on trading decisions through a portfolio manager. Built on [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain). For **educational purposes only** — not intended for real trading or investment.

## Features

- **25 analyst agents** — value, growth, macro, technical, fundamentals, sentiment, risk, and more
- **Famous investor personas** — simulations of Buffett, Munger, Ackman, Burry, Wood, Dalio, Simons, Lynch, and others
- **Multi-provider LLM support** — OpenAI, Anthropic, Groq, Gemini, DeepSeek, xAI, OpenRouter
- **Backtesting engine** — replay historical data with full agent deliberation and portfolio metrics
- **Live / paper trading** — execute via Alpaca with optional Telegram approval gate
- **Bull/bear debate node** — agents argue opposing sides before the portfolio manager decides
- **Market-regime selection** — classifies the current SPY regime (bull/bear/risk-off/neutral) each day and narrows the active analyst subset accordingly
- **Conviction-weight feedback loop** — tracks each agent's rolling directional hit-rate; high-accuracy agents receive proportionally more weight in the debate aggregation
- **Signal logging + forward-return labeling** — persists every per-agent-per-ticker signal to JSONL during a backtest run; a separate labeler attaches 1d/5d/20d forward returns so hit-rates can be computed
- **Token-usage telemetry** — captures and accumulates LLM token counts per agent across the full backtest run; Anthropic prompt caching is applied automatically and cache-read/creation tokens are surfaced separately
- **A/B comparison harness** — runs two backtest configs back-to-back and prints a side-by-side metrics table (full-vs-regime analysts, uniform-vs-conviction weights)
- **Per-agent model routing** — override model and provider per analyst via `--agent-model AGENT=model/PROVIDER`; handled by `RunRequest` (`src/llm/request.py`)
- **Parallel per-ticker execution** — set `QUORAI_PARALLEL_TICKERS=N` to run N tickers concurrently via a thread pool (`src/utils/concurrency.py`)

## How it works

```
Market Data → Analyst Agents → Portfolio Manager → Order Execution
                  ↑                   ↑
           (LangGraph nodes)   (deliberation graph)
```

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

## Usage

### Backtesting

Run from the CLI:

```bash
uv run python -m src.backtesting \
    --tickers AAPL,MSFT \
    --model deepseek/deepseek-chat \
    --model-provider OpenRouter \
    --show-reasoning
```

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
- `--agent-model AGENT=model/PROVIDER` — override model for a specific analyst; repeatable; use `*=model/PROVIDER` to override all agents

To run a side-by-side A/B comparison, add the `compare` subcommand:

```bash
uv run python -m src.backtesting compare \
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
```

> The Sharpe/Sortino in the final summary may differ slightly from the last rolling figure because the two blocks use marginally different timing for their calculation windows.

**Interpreting the metrics**

| Metric | Good | Acceptable | Poor |
|---|---|---|---|
| Portfolio Return | Beats benchmark | Roughly flat vs benchmark | Lags benchmark |
| Sharpe Ratio | > 1.0 | 0 – 1.0 | < 0 |
| Sortino Ratio | > 1.5 | 0 – 1.5 | < 0 |
| Max Drawdown | < 10% | 10 – 20% | > 20% |

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

## Project structure

| Path | Purpose |
|---|---|
| `src/` | Core library: agents, backtesting engine, LLM dispatch, data fetching, live trading |
| `src/agents/` | 25 analyst agents (personality + quant) plus risk manager and portfolio manager |
| `src/backtesting/` | Engine, portfolio, metrics, CLI (`python -m src.backtesting [compare]`), signal log, A/B harness |
| `src/regime/` | `MarketRegime` classifier + analyst-selection policy |
| `src/feedback/` | Forward-return labeler, rolling per-agent scorer, weights loader |
| `src/broker/` | `Broker` protocol + Alpaca client |
| `src/live/` | Live executor, runner, risk gate, audit journal |
| `src/notifications/` | Telegram approval client + command store |
| `src/data/` | Disk-persisted cache, Pydantic data models |
| `src/llm/` | Multi-provider LLM dispatch, OpenRouter catalog |
| `src/utils/` | Analyst registry (`ANALYST_CONFIG`), shared helpers |
| `src/config.py` | Centralised env-var config via pydantic-settings |
| `src/orchestration/` | `PipelineContext` — shared pre-graph helper for live and backtest |
| `tests/` | Unit and integration tests |

## Running tests

```bash
uv run pytest
```

## Adding an analyst

1. Create `src/agents/my_analyst.py` with a `my_analyst_agent(state, agent_id)` function.
2. Register it in `src/utils/analysts.py` — add an entry to `ANALYST_CONFIG`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for full contribution guidelines and [ARCHITECTURE.md](ARCHITECTURE.md) for system design.

## Python version

Python 3.12 recommended; 3.11 supported.

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Creator assumes no liability for financial losses
- Past performance does not indicate future results

The agent modules named after real investors (Buffett, Munger, Ackman, Burry, Wood, Asness, Dalio, Marks, Simons, Druckenmiller, Seykota, Greenblatt, Damodaran, Fisher, Lynch, Jhunjhunwala, Pabrai, Taleb, and others) are **educational simulations** that approximate publicly stated investment philosophies derived from books, interviews, and public writings. They are not affiliated with, endorsed by, or representative of the actual individuals or their organisations.

## Acknowledgements

- **[virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)** — persona-agent architecture, LLM prompt design, and orchestration patterns that Quorai is built upon.
- **[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)** — the bull/bear debate concept that inspired `src/agents/debate_node.py`.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
Third-party notices (including the upstream MIT license for ai-hedge-fund material) are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
