# Quorai

<p align="center">
  <img src="assets/logo-detailed.jpeg" alt="Quorai Logo" width="400"/>
</p>

*A quorum of AI agents deliberating trading decisions. Pronounced "KWOR-eye" (quorum + AI).*

A proof of concept for a multi-agent AI trading system. For **educational purposes only** — not intended for real trading or investment.

The system runs multiple analyst agents (value, growth, macro, technical, fundamentals, sentiment, risk) that collaborate through a portfolio manager to produce trading decisions.

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Creator assumes no liability for financial losses
- Past performance does not indicate future results

The agent modules named after real investors (Buffett, Munger, Ackman, Burry, Wood, Asness, Dalio, Marks, Simons, Druckenmiller, Seykota, Greenblatt, Damodaran, Fisher, Lynch, Jhunjhunwala, Pabrai, Taleb, and others) are **educational simulations** that approximate publicly stated investment philosophies derived from books, interviews, and public writings. They are not affiliated with, endorsed by, or representative of the actual individuals or their organisations.

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

# Financial data
FINANCIAL_DATASETS_API_KEY=...
```

## Usage

### Backtesting

Edit `run_backtest.py` to configure tickers, date range, model, and analysts, then run:

```bash
uv run python run_backtest.py
```

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
- `--dry-run` — print decisions without submitting orders
- `--confirm` — skip interactive confirmation prompt
- `--require-approval` — send orders to Telegram for human approval before submitting
- `--margin-requirement` — margin requirement fraction (default: 0.0)
- `--temperature` — LLM temperature override

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
| `src/agents/` | 19 analyst agents (personality + quant) plus risk manager and portfolio manager |
| `src/backtesting/` | Backtesting engine, portfolio, metrics, CLI |
| `src/broker/` | `Broker` protocol + Alpaca client |
| `src/live/` | Live executor, runner, risk gate, audit journal |
| `src/data/` | Disk-persisted cache, Pydantic data models |
| `src/llm/` | Multi-provider LLM dispatch, OpenRouter catalog |
| `src/config.py` | Centralised env-var config via pydantic-settings |
| `tests/` | Unit and integration tests |

## Running tests

```bash
uv run pytest
```

## Adding an analyst

1. Create `src/agents/my_analyst.py` with a `my_analyst_agent(state, agent_id)` function.
2. Register it in `src/utils/analysts.py` — add an entry to `ANALYST_CONFIG`.

## Python version

Python 3.12 recommended; 3.11 supported.

## Acknowledgements

- **[virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)** — persona-agent architecture, LLM prompt design, and orchestration patterns that Quorai is built upon.
- **[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)** — the bull/bear debate concept that inspired `src/agents/debate_node.py`.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
Third-party notices (including the upstream MIT license for ai-hedge-fund material) are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
