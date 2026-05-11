from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from src.backtesting.controller import AgentController
from src.backtesting.types import AgentDecisions, PortfolioSnapshot
from src.broker import Broker
from src.broker.alpaca_client import AlpacaClient
from src.broker.portfolio_adapter import to_snapshot
from src.live.audit_journal import AuditJournal
from src.live.executor import LiveExecutor
from src.live.risk_gate import RiskGate
from src.live.sod_equity import load_sod_equity, save_sod_equity

if TYPE_CHECKING:
    from src.live.idempotency_guard import IdempotencyGuard

logger = logging.getLogger(__name__)


class LiveRunner:
    def __init__(
        self,
        *,
        tickers: list[str],
        model_name: str,
        model_provider: str,
        selected_analysts: list[str] | None,
        margin_requirement: float = 0.0,
        llm_temperature: float | None = None,
        dry_run: bool = False,
        show_reasoning: bool = False,
        broker: Broker | None = None,
        journal: AuditJournal | None = None,
        risk_gate: RiskGate | None = None,
        idempotency_guard: IdempotencyGuard | None = None,
    ) -> None:
        self.tickers = tickers
        self.model_name = model_name
        self.model_provider = model_provider
        self.selected_analysts = selected_analysts
        self.margin_requirement = margin_requirement
        self.llm_temperature = llm_temperature
        self.dry_run = dry_run
        self.show_reasoning = show_reasoning
        self._broker = broker
        self._journal = journal
        self._risk_gate = risk_gate
        self._idempotency_guard = idempotency_guard
        self._sod_equity: float = 0.0

    def prepare(self) -> tuple[AgentDecisions, PortfolioSnapshot]:
        """Sync portfolio and run the agent graph; return decisions + snapshot.
        Does NOT submit any orders. Call execute() afterwards if desired."""
        from src.main import run_quorai

        # 1. Sync portfolio state
        if self._broker is None:
            self._broker = AlpacaClient()
        account = self._broker.get_account()
        positions = self._broker.get_positions()
        snapshot = to_snapshot(
            account=account,
            positions=positions,
            tickers=self.tickers,
            margin_requirement=self.margin_requirement,
        )
        logger.info("Portfolio synced: cash=%.2f", snapshot["cash"])
        for _tkr, _pos in snapshot.get("positions", {}).items():
            if _pos.get("long", 0) or _pos.get("short", 0):
                logger.info("Position synced: %s long=%.0f short=%.0f", _tkr, _pos.get("long", 0), _pos.get("short", 0))

        # 2. Capture SOD equity (first call today saves it)
        account_equity = float(account.equity or "0")
        sod = load_sod_equity()
        if sod is None:
            save_sod_equity(account_equity)
            sod = account_equity
        self._sod_equity = sod

        # 3. Build lookback window
        today = datetime.now(ZoneInfo("America/New_York")).date()
        start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

        # 4. Run agent graph
        controller = AgentController()
        output = controller.run_agent(
            run_quorai,
            tickers=self.tickers,
            start_date=start_date,
            end_date=end_date,
            portfolio=snapshot,
            model_name=self.model_name,
            model_provider=self.model_provider,
            selected_analysts=self.selected_analysts,
            llm_temperature=self.llm_temperature,
            show_reasoning=self.show_reasoning,
        )
        return output["decisions"], snapshot

    def execute(self, decisions: AgentDecisions) -> dict[str, str]:
        """Submit orders for the given decisions. prepare() must be called first."""
        if self._broker is None:
            raise RuntimeError("Call prepare() before execute()")
        executor = LiveExecutor(
            broker=self._broker,
            risk_gate=self._risk_gate,
            journal=self._journal,
            sod_equity=self._sod_equity,
            idempotency_guard=self._idempotency_guard,
        )
        return executor.execute_decisions(decisions, dry_run=self.dry_run)

    def run(self) -> dict:
        """Full pipeline: prepare → execute. Returns {decisions, execution_results, portfolio_snapshot}."""
        decisions, snapshot = self.prepare()

        print("\nDecisions:")
        print(f"{'Ticker':<8} {'Action':<8} {'Qty':>8}")
        print("-" * 30)
        for ticker, d in decisions.items():
            action = d.get("action", "hold")
            qty = d.get("quantity", 0)
            print(f"{ticker:<8} {action:<8} {qty:>8.3f}")
        print()

        execution_results = self.execute(decisions)
        return {
            "decisions": decisions,
            "execution_results": execution_results,
            "portfolio_snapshot": snapshot,
        }
