# Reading the backtest output

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

At the end of the run three final blocks are printed:

**`ENGINE RUN COMPLETE`** — core metrics:

```
ENGINE RUN COMPLETE
Total Return: -0.18%
Sharpe: -2.25
Sortino: -3.00
Max DD: 0.74% on 2026-05-08
```

**`BASELINES`** — buy-and-hold returns over the same window:

```
BASELINES (2026-04-22 → 2026-05-22)
  SPY:                  +1.12%
  AAPL:                 +0.85%
  MSFT:                 +2.31%
  Equal-weight (AAPL,MSFT):  +1.58%
```

**`ACTIVE PERFORMANCE`** — strategy performance vs benchmarks:

```
ACTIVE PERFORMANCE
  Strategy:                      -0.18%
  Alpha vs SPY:                  -1.30%   ← strategy total return − SPY total return
  Alpha vs Equal-weight basket:  -0.45%   ← strategy − equal-weight ticker basket
  IR vs SPY:                     -0.83    ← (mean active daily return / std) × √252
  IR vs Equal-weight basket:     -0.61
```

> The Sharpe/Sortino in the final summary may differ slightly from the last rolling figure because the two blocks use marginally different timing for their calculation windows.

## Interpreting the metrics

| Metric | Good | Acceptable | Poor |
|---|---|---|---|
| Portfolio Return | Beats benchmark | Roughly flat vs benchmark | Lags benchmark |
| Sharpe Ratio | > 1.0 | 0 – 1.0 | < 0 |
| Sortino Ratio | > 1.5 | 0 – 1.5 | < 0 |
| Max Drawdown | < 10% | 10 – 20% | > 20% |
| Alpha vs SPY | > 0% | ~ 0% | < 0% |
| IR vs SPY | > 0.5 | 0 – 0.5 | < 0 |
| IR vs basket | > 0.5 | 0 – 0.5 | < 0 |

## Important caveats for short backtests

- Sharpe and Sortino are annualised from daily returns. With only a few days of data there are too few samples for the figures to be statistically meaningful — treat them as noise until the test window covers at least several months.
- A negative `Total Position Value` means the portfolio manager issued net short orders. This is valid behaviour but unusual; check `--show-reasoning` to understand why.
- Always compare against the benchmark return over the *same* period before drawing conclusions.

For the formulas behind all metrics see [docs/math.md](math.md).
