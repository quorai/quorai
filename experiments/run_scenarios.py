#!/usr/bin/env python3
"""Run a curated set of market-regime evaluation scenarios and produce a summary report.

Usage (from project root):
    uv run python experiments/run_scenarios.py                         # run all 10
    uv run python experiments/run_scenarios.py --scenarios bull-megacap-2024Q1  # smoke test
    uv run python experiments/run_scenarios.py --skip-existing         # skip already-done
    uv run python experiments/run_scenarios.py --summary-only          # regenerate report
    uv run python experiments/run_scenarios.py --no-llm-cache          # fresh LLM calls (after prompt edits)
    uv run python experiments/run_scenarios.py --max-tickers 2 --max-days 10  # smoke: 2 tickers × 10 days (~6% cost)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.regime.classifier import MarketRegime, classify_regime  # noqa: E402
from src.tools.api import get_price_data  # noqa: E402

LOGS_DIR = PROJECT_ROOT / "logs/experiments/runs"
RESULTS_DIR = PROJECT_ROOT / "experiments/results"


@dataclass(frozen=True)
class Scenario:
    label: str
    start: str
    end: str
    tickers: str
    expected_regime: MarketRegime
    shape: str


SCENARIOS: list[Scenario] = [
    Scenario(
        label="bull-megacap-2024Q1",
        start="2024-01-01",
        end="2024-03-31",
        tickers="NVDA,MSFT,META,AVGO,AMD",
        expected_regime=MarketRegime.BULL_TREND,
        shape="Strong AI-led uptrend — can it ride momentum without prematurely shorting?",
    ),
    Scenario(
        label="bear-growth-2022Q2",
        start="2022-04-01",
        end="2022-06-30",
        tickers="NVDA,MSFT,META,NFLX,TSLA",
        expected_regime=MarketRegime.BEAR_TREND,
        shape="Rate-hike growth selloff — does it short, hedge, or just lose less?",
    ),
    Scenario(
        label="crash-recovery-2020-covid",
        start="2020-02-15",
        end="2020-05-15",
        tickers="SPY,AAPL,MSFT,JPM,BA",
        expected_regime=MarketRegime.RISK_OFF,
        shape="COVID crash + V-recovery — regime transition, risk_manager under stress",
    ),
    Scenario(
        label="chop-defensives-2023Q2",
        start="2023-04-01",
        end="2023-06-30",
        tickers="JNJ,PG,KO,WMT,VZ",
        expected_regime=MarketRegime.NEUTRAL,
        shape="Sideways tape with defensives — does it overtrade in a directionless market?",
    ),
    Scenario(
        label="dispersion-2022Q4",
        start="2022-10-01",
        end="2022-12-31",
        tickers="NVDA,AAPL,META,INTC,F",
        expected_regime=MarketRegime.NEUTRAL,
        shape="Clear winner/loser dispersion — can long-short capture both sides?",
    ),
    Scenario(
        label="meme-vol-2021Q1",
        start="2021-01-01",
        end="2021-03-31",
        tickers="GME,AMC,BB",
        expected_regime=MarketRegime.BULL_TREND,
        shape="Meme/extreme vol — does it stay disciplined under irrational price action?",
    ),
    Scenario(
        label="rotation-2022Q1",
        start="2022-01-01",
        end="2022-03-31",
        tickers="XOM,CVX,NVDA,META",
        expected_regime=MarketRegime.BEAR_TREND,
        shape="Textbook cyclicals-up/growth-down rotation — long-short sector signal quality",
    ),
    Scenario(
        label="grind-down-2022-late",
        start="2022-08-01",
        end="2022-10-15",
        tickers="AAPL,MSFT,JPM,BAC,XOM",
        expected_regime=MarketRegime.BEAR_TREND,
        shape="Slow -16% SPY drift — does it avoid catching falling knives?",
    ),
    Scenario(
        label="quiet-bull-2023-summer",
        start="2023-06-01",
        end="2023-08-15",
        tickers="AAPL,MSFT,NVDA,GOOG,AMZN",
        expected_regime=MarketRegime.BULL_TREND,
        shape="Low-vol +6% drift — does it let winners run without overtrading?",
    ),
    Scenario(
        label="tariff-shock-2025Q2",
        start="2025-04-01",
        end="2025-04-30",
        tickers="SPY,AAPL,NVDA,F,XOM",
        expected_regime=MarketRegime.RISK_OFF,
        shape="Apr-2 tariff shock + Apr-9 reversal — regime switch handling under news vol",
    ),
]


def _truncate(s: Scenario, max_tickers: int | None, max_days: int | None) -> Scenario:
    tickers = s.tickers
    end = s.end
    if max_tickers:
        tickers = ",".join(s.tickers.split(",")[:max_tickers])
    if max_days:
        capped = (date.fromisoformat(s.start) + timedelta(days=max_days)).isoformat()
        end = min(capped, s.end)
    if tickers == s.tickers and end == s.end:
        return s
    return replace(s, tickers=tickers, end=end)


def _run_scenario(
    scenario: Scenario,
    model: str,
    provider: str,
    no_llm_cache: bool,
    use_regime_selection: bool = True,
    use_conviction_weights: bool = True,
) -> bool:
    env = os.environ.copy()
    if no_llm_cache:
        env["QUORAI_LLM_CACHE"] = "0"

    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{scenario.label}.log"

    cmd = [
        "uv",
        "run",
        "backtester",
        "--tickers",
        scenario.tickers,
        "--start-date",
        scenario.start,
        "--end-date",
        scenario.end,
        "--model",
        model,
        "--model-provider",
        provider,
        "--initial-capital",
        "100000",
        "--seed",
        "42",
        "--run-label",
        scenario.label,
        "--log-dir",
        "logs/experiments",
    ]
    if use_regime_selection:
        cmd.append("--use-regime-selection")
    if use_conviction_weights:
        cmd.append("--use-conviction-weights")

    print(f"[{scenario.label}] {scenario.start} → {scenario.end} ({scenario.tickers})", flush=True)

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=PROJECT_ROOT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            print(line, end="", flush=True)
        proc.wait()

    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode}) — see {log_file.relative_to(PROJECT_ROOT)}")
        return False

    print("  OK")
    return True


def _find_manifest(label: str) -> dict | None:
    if not LOGS_DIR.exists():
        return None
    for f in sorted(LOGS_DIR.glob("*.json"), reverse=True):
        try:
            m = json.loads(f.read_text())
            if m.get("cli_args", {}).get("parsed", {}).get("run_label") == label:
                return m
        except Exception:
            continue
    return None


def _is_already_done(label: str) -> bool:
    m = _find_manifest(label)
    return m is not None and m.get("status") == "completed"


def _classify_scenario_regime(scenario: Scenario) -> MarketRegime:
    # Fetch SPY with 4-month warmup (classifier needs ≥60 trading days for long-vol window)
    warmup_start = (datetime.strptime(scenario.start, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
    spy_df = get_price_data("SPY", warmup_start, scenario.end)
    return classify_regime(spy_df, scenario.end)


def _fmt(v: float | None, suffix: str = "") -> str:
    if v is None:
        return "N/A"
    return f"{v:.2f}{suffix}"


def _build_report(scenarios: list[Scenario], output_path: Path) -> None:
    rows = []
    missing = []

    for s in scenarios:
        manifest = _find_manifest(s.label)
        if manifest is None or manifest.get("status") != "completed":
            missing.append(s.label)
            continue

        result = manifest.get("result", {})
        metrics = result.get("metrics", {})
        baselines = result.get("baselines", {})

        try:
            regime_observed = _classify_scenario_regime(s)
        except Exception:
            regime_observed = MarketRegime.NEUTRAL

        rows.append(
            {
                "label": s.label,
                "window": f"{s.start} → {s.end}",
                "tickers": s.tickers,
                "expected_regime": s.expected_regime.value,
                "observed_regime": regime_observed.value,
                "total_return_pct": result.get("total_return_pct"),
                "spy_return_pct": baselines.get("spy_return_pct"),
                "alpha_vs_spy": metrics.get("alpha_vs_spy_pct"),
                "alpha_vs_basket": metrics.get("alpha_vs_basket_pct"),
                "ir_vs_spy": metrics.get("information_ratio_vs_spy"),
                "sharpe": metrics.get("sharpe_ratio"),
                "max_dd": metrics.get("max_drawdown"),
                "long_short_ratio": metrics.get("long_short_ratio"),
                "shape": s.shape,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Regime Evaluation Report",
        f"\nGenerated: {date.today()}  |  Scenarios run: {len(rows)}/{len(scenarios)}\n",
    ]

    # Per-scenario table
    lines += [
        "## Per-Scenario Results\n",
        "| Label | Window | Expected | Observed | Return | SPY | α vs SPY | α vs basket | IR vs SPY | Sharpe | Max DD | L/S ratio |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        match = "✓" if r["expected_regime"] == r["observed_regime"] else "✗"
        cols = [
            r["label"],
            r["window"],
            r["expected_regime"],
            f"{r['observed_regime']} {match}",
            _fmt(r["total_return_pct"], "%"),
            _fmt(r["spy_return_pct"], "%"),
            _fmt(r["alpha_vs_spy"], "%"),
            _fmt(r["alpha_vs_basket"], "%"),
            _fmt(r["ir_vs_spy"]),
            _fmt(r["sharpe"]),
            _fmt(r["max_dd"], "%"),
            _fmt(r["long_short_ratio"]),
        ]
        lines.append("| " + " | ".join(cols) + " |")

    if missing:
        lines.append(f"\n_Not yet completed: {', '.join(missing)}_")

    # Regime-grouped summary
    lines += ["", "## Results by Observed Regime\n"]
    grouped: dict[str, list] = {}
    for r in rows:
        grouped.setdefault(r["observed_regime"], []).append(r)

    lines += [
        "| Regime | n | Mean α vs SPY | Median α vs SPY | Win rate (α > 0) | Mean IR |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for regime, group in sorted(grouped.items()):
        alphas = [r["alpha_vs_spy"] for r in group if r["alpha_vs_spy"] is not None]
        irs = [r["ir_vs_spy"] for r in group if r["ir_vs_spy"] is not None]
        mean_alpha = sum(alphas) / len(alphas) if alphas else None
        median_alpha = sorted(alphas)[len(alphas) // 2] if alphas else None
        win_rate = (sum(1 for a in alphas if a > 0) / len(alphas)) if alphas else None
        mean_ir = sum(irs) / len(irs) if irs else None
        lines.append(f"| {regime} | {len(group)} | {_fmt(mean_alpha, '%')} | {_fmt(median_alpha, '%')} | {f'{win_rate:.0%}' if win_rate is not None else 'N/A'} | {_fmt(mean_ir)} |")

    # Notable outliers
    notable = [r for r in rows if r["alpha_vs_spy"] is not None and abs(r["alpha_vs_spy"]) > 5.0]
    if notable:
        lines += ["", "## Notable Outliers (|α vs SPY| > 5pp)\n"]
        for r in notable:
            sign = "+" if r["alpha_vs_spy"] > 0 else ""
            lines.append(f"- **{r['label']}**: α={sign}{r['alpha_vs_spy']:.2f}% — {r['shape']}")

    # Scenario key
    lines += ["", "## Scenario Descriptions\n"]
    for s in scenarios:
        lines.append(f"- **{s.label}** (`{s.tickers}`): {s.shape}")

    output_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport: {output_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run curated regime evaluation scenarios and report results")
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--model-provider", dest="model_provider", default="OpenRouter")
    parser.add_argument("--scenarios", type=str, default=None, help="Comma-separated subset of scenario labels to run")
    parser.add_argument("--skip-existing", action="store_true", dest="skip_existing", help="Skip scenarios with a completed manifest in logs/")
    parser.add_argument("--summary-only", action="store_true", dest="summary_only", help="Skip all runs; regenerate report from existing manifests")
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        dest="max_tickers",
        help="Truncate each scenario's ticker list to the first N (smoke testing).",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        dest="max_days",
        help="Truncate each scenario's date range to N calendar days from start (smoke testing).",
    )
    parser.add_argument(
        "--no-llm-cache",
        action="store_true",
        dest="no_llm_cache",
        help="Set QUORAI_LLM_CACHE=0 for fresh LLM calls (use after editing prompts)",
    )
    parser.add_argument(
        "--no-regime-selection",
        action="store_true",
        dest="no_regime_selection",
        help="Disable --use-regime-selection (runs all analysts regardless of market regime)",
    )
    parser.add_argument(
        "--no-conviction-weights",
        action="store_true",
        dest="no_conviction_weights",
        help="Disable --use-conviction-weights (uniform analyst weighting)",
    )
    args = parser.parse_args()

    label_filter = {s.strip() for s in args.scenarios.split(",")} if args.scenarios else None
    scenarios = [s for s in SCENARIOS if label_filter is None or s.label in label_filter]

    if not scenarios:
        print(f"No matching scenarios for filter: {args.scenarios}", file=sys.stderr)
        sys.exit(1)

    if not args.summary_only:
        if args.max_tickers or args.max_days:
            print(f"SMOKE MODE: max_tickers={args.max_tickers}, max_days={args.max_days} — results are not representative.\n")
        if args.no_llm_cache:
            print("LLM cache disabled — all calls will be fresh.")
        else:
            print("LLM cache active. Pass --no-llm-cache after editing prompts to avoid stale responses.\n")

        for s in scenarios:
            s = _truncate(s, args.max_tickers, args.max_days)
            if args.skip_existing and _is_already_done(s.label):
                print(f"[{s.label}] skipping (already completed)")
                continue
            _run_scenario(
                s,
                args.model,
                args.model_provider,
                args.no_llm_cache,
                use_regime_selection=not args.no_regime_selection,
                use_conviction_weights=not args.no_conviction_weights,
            )

    report_path = RESULTS_DIR / f"eval-{date.today()}.md"
    _build_report(scenarios, report_path)


if __name__ == "__main__":
    main()
