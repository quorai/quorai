# Math & quantitative methods

All annualisation uses 252 trading days.

## Portfolio metrics (`src/backtesting/metrics.py`)

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
| Information ratio vs basket | Same formula with `active_return = strategy_return − equal_weight_basket_return` |

## Portfolio exposure (`src/backtesting/valuation.py`)

| Metric | Formula |
|---|---|
| NAV | `cash + Σ(long_shares × price) − Σ(short_shares × price)` |
| Long exposure | `Σ(long_shares × price)` |
| Short exposure | `Σ(short_shares × price)` |
| Gross exposure | `long + short` |
| Net exposure | `long − short` |
| L/S ratio | `long / short` |
| Weighted-average cost basis | `(old_basis × old_qty + new_price × new_qty) / total_qty` (updated on every fill) |

## Position sizing (`src/agents/risk_manager.py`)

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

## Valuation models (`src/agents/valuation.py`)

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

## Technical indicators (`src/agents/technicals.py`, `src/agents/jim_simons.py`)

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

## AQR multi-factor scoring (`src/agents/cliff_asness.py`)

| Factor | Key sub-signals | Max pts |
|---|---|---|
| Value | P/E, P/B, FCF yield, EV/EBITDA vs thresholds | 8 |
| Momentum | 12-1 momentum vs ±5 % / ±20 % thresholds; −1 if 1-month gain > 15 % | 4 |
| Quality | ROIC, gross margin, earnings stability (% positive years) | 6 |
| Low volatility | 63-day annualised vol bucketed into five tiers | 4 |

Overall signal strength scales with how many factors align simultaneously (max 22 pts → 90–100 % confidence).

## Conviction-weight feedback loop (`src/feedback/`)

After each run, signals are labeled with 1 d / 5 d / 20 d forward returns. The rolling directional hit-rate for each agent is used to upweight high-accuracy agents in the debate aggregation. Weights are persisted in `src/feedback/weights.json` and reloaded at the start of the next run when `--use-conviction-weights` is set.
