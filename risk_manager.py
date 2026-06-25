from __future__ import annotations

"""
risk_manager.py  (RISK-CONTROL REMEDIATION PASS)

FIX — drawdown-aware position sizing:

Previously lot_size() always returned the same fixed lot size (or the same
equity-risk-based size) regardless of recent performance. There was no path
anywhere in this file, or in trade_manager.py, that reduced size after a
losing streak or after a realized equity drawdown — sizing was either "full
size" or "zero" (blocked entirely by a guard), with no intermediate
de-risking step. That meant drawdown depth was structurally maximized: the
system traded the same size into a losing streak as into a winning one,
right up until a hard daily-loss cutoff in allowed().

lot_size() now accepts recent_closed_trades and scales the computed/fixed
base size down via _drawdown_multiplier(), which combines two independent
checks (whichever cuts size more wins):

  (a) Consecutive-loss step-down — catches *behavioral* degradation
      (the strategy is currently wrong about something) independent of
      the magnitude of each loss.
  (b) Equity-curve drawdown step-down — catches *magnitude* risk
      (a couple of large losses) independent of streak length.

This sizing layer is a SECOND, independent control. It is not a substitute
for the entry-side guards in trade_manager.py / auto_upgrade_engine.py — it
exists for trades that still get through those guards (e.g. a different
agent that isn't individually flagged, but overall equity is still down).
"""

from datetime import datetime, timezone
from typing import Iterable, Tuple

from config import CONFIG, RiskConfig, active_news_blackout, asset_profile
from models import Direction, Signal, Trade, TradeStatus


class RiskManager:
    def __init__(self, config: RiskConfig = CONFIG.risk):
        self.config = config

    def allowed(self, signal: Signal, open_trades: Iterable[Trade], spread: float, realized_today: float) -> Tuple[bool, str]:
        profile = asset_profile(signal.symbol)
        blackout = active_news_blackout()
        if blackout:
            return False, f"News blackout active: {blackout}"
        if self.config.high_winrate_mode and signal.confidence < self.config.high_winrate_min_confidence:
            return False, "Confidence below high win-rate floor"
        if signal.rr < self._minimum_rr(signal.symbol):
            return False, "RR below minimum"
        if spread > profile.max_spread_points:
            return False, "Spread too high"
        active_trades = [t for t in open_trades if t.status in {TradeStatus.OPEN, TradeStatus.PARTIAL}]
        if len(active_trades) >= self.config.max_concurrent_trades:
            return False, "Max concurrent trades reached"
        duplicate = self._duplicate_setup(signal, active_trades, profile.max_same_setup_open, profile.duplicate_entry_atr)
        if duplicate:
            return False, duplicate
        max_daily_loss = -self.config.account_equity * self.config.max_daily_drawdown_pct / 100
        if realized_today <= max_daily_loss:
            return False, "Daily drawdown protection active"
        if signal.direction == Direction.BUY and not (signal.stop_loss < signal.entry < signal.take_profit):
            return False, "Invalid BUY geometry"
        if signal.direction == Direction.SELL and not (signal.take_profit < signal.entry < signal.stop_loss):
            return False, "Invalid SELL geometry"
        return True, "Allowed"

    def lot_size(
        self,
        entry: float,
        stop_loss: float,
        symbol: str = "",
        recent_closed_trades: Iterable[Trade] | None = None,
    ) -> float:
        """
        Compute the base lot size (fixed or equity-risk-based, same as before),
        then scale it down based on recent performance via
        _drawdown_multiplier(). recent_closed_trades is optional and defaults
        to no scaling (multiplier 1.0) for any caller that hasn't been updated
        to pass history — this keeps the method backwards compatible.
        """
        if self.config.fixed_lot_size > 0:
            base = round(self.config.fixed_lot_size / self.config.lot_step) * self.config.lot_step
        else:
            profile = asset_profile(symbol)
            risk_amount = self.config.account_equity * self.config.risk_per_trade_pct / 100
            risk_points = abs(entry - stop_loss)
            raw_lots = risk_amount / max(risk_points * profile.contract_size, 1e-9)
            base = round(raw_lots / self.config.lot_step) * self.config.lot_step

        multiplier = self._drawdown_multiplier(recent_closed_trades)
        scaled = base * multiplier
        stepped = round(scaled / self.config.lot_step) * self.config.lot_step
        bounded = max(self.config.min_lot, min(self.config.max_lot, stepped))
        return round(bounded, 2)

    def _drawdown_multiplier(self, recent_closed_trades: Iterable[Trade] | None) -> float:
        """
        Reduce size based on (a) consecutive losses and (b) realized drawdown
        from the recent equity peak. Both checks apply — whichever cuts size
        more (the smaller multiplier) wins. This function never increases
        size above 1.0; it only ever holds flat or reduces.
        """
        if not recent_closed_trades:
            return 1.0
        trades = list(recent_closed_trades)
        if not trades:
            return 1.0

        # (a) Consecutive-loss step-down
        consecutive_losses = 0
        for trade in reversed(trades):
            if trade.pnl < 0:
                consecutive_losses += 1
            else:
                break
        if consecutive_losses >= 4:
            streak_mult = 0.25
        elif consecutive_losses == 3:
            streak_mult = 0.50
        elif consecutive_losses == 2:
            streak_mult = 0.75
        else:
            streak_mult = 1.0

        # (b) Equity-curve drawdown step-down (based on realized PnL of recent
        # trades, not unrealized/floating — this is the cumulative PnL curve
        # over the lookback window, not a snapshot of open positions)
        window = trades[-50:]
        equity_curve = []
        running = 0.0
        for trade in window:
            running += trade.pnl
            equity_curve.append(running)
        if equity_curve:
            peak = max(equity_curve)
            current = equity_curve[-1]
            drawdown = peak - current  # in account currency, >= 0
            equity_ref = max(self.config.account_equity, 1.0)
            drawdown_pct = (drawdown / equity_ref) * 100
        else:
            drawdown_pct = 0.0

        if drawdown_pct >= self.config.max_daily_drawdown_pct * 1.5:
            dd_mult = 0.25
        elif drawdown_pct >= self.config.max_daily_drawdown_pct:
            dd_mult = 0.50
        elif drawdown_pct >= self.config.max_daily_drawdown_pct * 0.5:
            dd_mult = 0.75
        else:
            dd_mult = 1.0

        return min(streak_mult, dd_mult)

    def today_key(self) -> datetime.date:
        return datetime.now(timezone.utc).date()

    def _minimum_rr(self, symbol: str) -> float:
        if self.config.use_micro_scalp_exits:
            floor = max(1.0, float(self.config.micro_min_rr))
        else:
            floor = max(self.config.min_rr, asset_profile(symbol).min_rr)
        if self.config.high_winrate_mode:
            floor = max(floor, float(self.config.high_winrate_min_rr))
        return floor

    def _duplicate_setup(self, signal: Signal, active_trades: list[Trade], max_same_setup: int, entry_atr: float) -> str | None:
        setup = str(signal.metadata.get("setup_model", ""))
        atr_value = float(signal.metadata.get("atr", 0.0) or 0.0)
        same_setup = [
            trade
            for trade in active_trades
            if trade.signal.symbol == signal.symbol
            and trade.signal.direction == signal.direction
            and str(trade.signal.metadata.get("setup_model", "")) == setup
        ]
        if len(same_setup) >= max_same_setup:
            return f"Same setup already open: {setup}"
        if atr_value <= 0:
            return None
        for trade in active_trades:
            if trade.signal.symbol != signal.symbol or trade.signal.direction != signal.direction:
                continue
            if str(trade.signal.metadata.get("setup_model", "")) != setup:
                continue
            if abs(trade.signal.entry - signal.entry) <= atr_value * entry_atr:
                return "Duplicate entry too close to existing open trade"
        return None
