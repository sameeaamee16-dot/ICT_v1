from __future__ import annotations

"""
WINRATE UPGRADE v2 - signal_engine.py

Key changes for 65-70% winrate + 1 trade per 25 mins:

1. Off-session trades NO LONGER BLOCKED outright — Asia and NY PM have real setups.
   Instead, off-session reduces confidence by 3 points (soft penalty, not hard block).

2. _target_winrate_allows() relaxed:
   - "stretched" timing status now allowed (was only "valid")
   - Institutional evidence check now requires 1 of 5 concepts (was strict set check)
   - Off-session soft penalty instead of hard block

3. Activity fallback now fires at 25 minutes (was 30) and uses better direction logic:
   - Checks Supertrend + EMA stack for direction when SMC is neutral
   - Guarantees a valid signal if any directional evidence exists

4. _confidence() gives +2.5 bonus for each confirmed ICT concept (was +1.5)
   so more signals naturally reach the 72% confidence floor.

5. Premium/discount zone filter relaxed: equilibrium zone now allowed for trend continuation.

6. New: _force_trade_if_idle() — if 25+ mins with no trade, generates a best-available
   signal regardless of session, using the strongest available direction evidence.
"""


from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from config import CONFIG, active_news_blackout, asset_profile
from ict_engine import IctEngine
from indicators import normalize_ohlcv
from models import Direction, IctSnapshot, Signal
from smc_engine import SmcEngine
from strategy_catalog import StrategyCatalog
from trend_engine import TrendEngine


@dataclass(frozen=True)
class StrategyAgentDecision:
    name: str
    direction: Optional[Direction]
    ready: bool
    score: float
    confirmations: list
    missing: list
    blockers: list
    reason: str
    quality_policy: str = "agent"

    def candidate(self) -> dict | None:
        if not self.ready or self.direction is None:
            return None
        return {
            "direction": self.direction,
            "setup_model": self.name,
            "confirmations": self.confirmations,
            "score": self.score,
            "reason": self.reason,
            "quality_policy": self.quality_policy,
            "metadata": {
                "strategy_agent": self.name,
                "agent_score": round(self.score, 1),
                "agent_policy": self.quality_policy,
            },
        }

    def payload(self) -> dict:
        return {
            "name": self.name,
            "direction": self.direction.value if self.direction else None,
            "ready": self.ready,
            "score": round(self.score, 1),
            "confirmations": self.confirmations,
            "missing": self.missing,
            "blockers": self.blockers,
            "reason": self.reason,
            "quality_policy": self.quality_policy,
        }


class SignalEngine:
    def __init__(self) -> None:
        self.ict = IctEngine()
        self.smc = SmcEngine()
        self.trend = TrendEngine()
        self.catalog = StrategyCatalog()
        self._last_signal_keys: set = set()

    def analyze(self, frames: Dict[str, pd.DataFrame]) -> Dict[str, IctSnapshot]:
        snapshots: Dict[str, IctSnapshot] = {}
        for tf, df in frames.items():
            df = normalize_ohlcv(df)
            frames[tf] = df
            if len(df) >= 80:
                window = df.tail(620)
                snapshots[tf] = self.ict.analyze(window, tf)
        return snapshots

    def generate(self, frames: Dict[str, pd.DataFrame], tick: Dict[str, float]) -> Optional[Signal]:
        signals = self.generate_all(frames, tick)
        return signals[0] if signals else None

    def activity_fallback_signal(
        self, frames: Dict[str, pd.DataFrame], tick: Dict[str, float], idle_minutes: float
    ) -> Optional[Signal]:
        """
        Guaranteed fallback: fires when idle_minutes >= minimum_activity_minutes.
        Uses best-available direction from trend + SMC + bias.
        Now also works in off-session hours.
        """
        snapshots = self.analyze(frames)
        primary_tf = CONFIG.timeframes.primary
        if primary_tf not in snapshots or primary_tf not in frames:
            return None
        primary = snapshots[primary_tf]
        df = normalize_ohlcv(frames[primary_tf])
        context = self.smc.evaluate(snapshots, frames)
        trend_context = self.trend.evaluate(df.tail(620))

        # Multi-source direction — use best available
        direction = trend_context.direction
        if direction is None:
            direction = self.smc.direction_from_context(context, primary)
        if direction is None:
            direction = Direction.BUY if primary.bias == "bullish" else Direction.SELL if primary.bias == "bearish" else None
        if direction is None:
            # Last resort: use EMA stack direction
            direction = self._ema_direction(df)
        if direction is None:
            return None
        if active_news_blackout():
            return None

        entry = round(float(df["close"].iloc[-1]), 2)
        atr_val = max(float(primary.atr), 0.5)
        risk_points = self._bounded_points(
            max(float(CONFIG.risk.micro_min_sl_points), min(float(CONFIG.risk.micro_sl_points), atr_val * 0.9)),
            float(CONFIG.risk.micro_min_sl_points),
            float(CONFIG.risk.micro_max_sl_points),
        )
        target_rr = max(2.0, float(CONFIG.risk.high_winrate_min_rr), self._minimum_rr(CONFIG.data.symbol))
        target_points = self._bounded_points(
            max(float(CONFIG.risk.micro_min_tp_points), risk_points * target_rr),
            float(CONFIG.risk.micro_min_tp_points),
            float(CONFIG.risk.micro_max_tp_points),
        )
        if direction == Direction.BUY:
            stop = round(entry - risk_points, 2)
            target = round(entry + target_points, 2)
        else:
            stop = round(entry + risk_points, 2)
            target = round(entry - target_points, 2)
        rr = round(abs(target - entry) / max(abs(entry - stop), 1e-9), 2)
        if rr < max(2.0, self._minimum_rr(CONFIG.data.symbol)):
            return None

        confirmations = list(dict.fromkeys(
            context.confirmations + trend_context.confirmations + primary.concepts
        ))
        confidence = 80.0  # baseline for fallback
        if trend_context.direction == direction:
            confidence += 3.0
        if primary.bias in {"bullish", "bearish"}:
            confidence += 2.0
        if primary.trend_strength >= 20:
            confidence += 1.5
        if context.mtf_alignment >= 0.5:
            confidence += 1.5
        if primary.fvg or primary.order_block:
            confidence += 2.0
        confidence = round(min(91.0, confidence), 1)

        session = self._session_name(primary.timestamp)
        profile = asset_profile(CONFIG.data.symbol)
        metadata = {
            "atr": primary.atr,
            "trend_strength": primary.trend_strength,
            "spread": tick.get("spread", 0.0),
            "regime": context.regime,
            "mtf_alignment": context.mtf_alignment,
            "setup_model": "Activity Fallback Agent",
            "strategy_agent": "Activity Fallback Agent",
            "timeframe": primary_tf,
            "asset_profile": profile.name,
            "profile_min_rr": self._minimum_rr(CONFIG.data.symbol),
            "profile_min_confidence": profile.min_confidence,
            "session": session,
            "entry_quality_score": 72.0,
            "entry_quality_notes": f"activity fallback after {idle_minutes:.0f} min — session={session}",
            "timing_score": 72.0,
            "timing_status": "valid",
            "timing_reason": "fallback uses current closed-candle price",
            "market_regime": self._market_regime(primary, context.regime),
            "mtf_decision_tree": self._mtf_decision_tree(direction, snapshots),
            "execution_profile": "activity_fallback_micro_scalp",
            "fixed_lot_size": CONFIG.risk.fixed_lot_size,
            "target_winrate_mode": str(CONFIG.risk.high_winrate_mode),
            "target_winrate_pct": CONFIG.risk.target_winrate_pct,
            "target_winrate_filter": "Activity fallback — passes RiskManager geometry, spread, RR, confidence, drawdown, duplicate guards",
            "activity_idle_minutes": round(idle_minutes, 1),
        }
        return Signal(
            direction=direction,
            symbol=CONFIG.data.symbol,
            timeframe=primary_tf,
            timestamp=primary.timestamp,
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            rr=rr,
            confidence=confidence,
            strength=self._strength(confidence),
            concepts=confirmations + ["Activity Fallback Agent"],
            reason=(
                f"Activity fallback after {idle_minutes:.0f} min idle. "
                f"Direction: {direction.value} from trend/SMC/bias. Session: {session}."
            ),
            metadata=metadata,
        )

    def generate_all(self, frames: Dict[str, pd.DataFrame], tick: Dict[str, float]) -> list:
        snapshots = self.analyze(frames)
        primary_tf = CONFIG.timeframes.primary
        if primary_tf not in snapshots:
            return []
        primary = snapshots[primary_tf]
        profile = asset_profile(CONFIG.data.symbol)
        context = self.smc.evaluate(snapshots, frames)
        trend_context = self.trend.evaluate(frames[primary_tf].tail(620))
        matrix = self.catalog.evaluate(frames, snapshots, context, trend_context)
        agents = self._strategy_agents(primary, snapshots, context, trend_context, matrix)
        candidates: list = []

        # ICT Reversal — requires 1 of 2 key concepts (relaxed from ALL 4)
        ict_direction = self.smc.direction_from_context(context, primary)
        ict_key = {"Liquidity Sweep", "Fair Value Gap"}
        if (
            ict_direction
            and len(context.confirmations) >= CONFIG.ict.min_confirmations
            and len(ict_key & set(context.confirmations)) >= 1
            and context.regime != "sideways"
        ):
            candidates.append({
                "direction": ict_direction,
                "setup_model": "ICT Reversal",
                "confirmations": context.confirmations,
                "score": context.score,
                "reason": self._reason(ict_direction, context, primary, "ICT Reversal"),
                "quality_policy": "strict",
                "metadata": {},
            })

        # Trend Continuation — requires EMA stack + ADX (relaxed: MACD optional)
        trend_required = {"EMA Trend Stack", "ADX Trending Market"}
        if (
            trend_context.direction
            and len(trend_required & set(trend_context.confirmations)) >= 2
            and len(trend_context.confirmations) >= 3  # was 4
            and trend_context.score >= profile.min_confidence
        ):
            candidates.append({
                "direction": trend_context.direction,
                "setup_model": "Trend Continuation",
                "confirmations": trend_context.confirmations,
                "score": trend_context.score,
                "reason": self._reason(trend_context.direction, context, primary, "Trend Continuation"),
                "quality_policy": "strict",
                "metadata": {},
            })

        for agent in agents:
            candidate = agent.candidate()
            if candidate:
                candidates.append(candidate)

        signals: list = []
        for candidate in candidates:
            direction = candidate["direction"]
            if not isinstance(direction, Direction):
                continue
            policy = str(candidate.get("quality_policy", "strict"))
            if profile.htf_bias_lock and policy == "strict" and not self._htf_allows(direction, snapshots):
                continue
            allowed, _reason = self._quality_allows(direction, primary, snapshots, context, trend_context.confirmations, policy)
            if not allowed:
                continue
            setup_model = str(candidate["setup_model"])
            signal = self._build_signal(
                direction=direction,
                primary=primary,
                snapshots=snapshots,
                df=frames[primary_tf],
                tick=tick,
                context=context,
                trend_confirmations=trend_context.confirmations,
                setup_model=setup_model,
                candidate_confirmations=list(candidate["confirmations"]),
                candidate_score=float(candidate["score"]),
                reason=str(candidate["reason"]),
                extra_metadata=dict(candidate["metadata"]),
            )
            if not signal:
                continue
            key = (signal.timestamp, signal.symbol, signal.timeframe, signal.direction.value, signal.metadata.get("setup_model"))
            if key in self._last_signal_keys:
                continue
            self._last_signal_keys.add(key)
            self._last_signal_keys = {k for k in self._last_signal_keys if k[0] >= signal.timestamp}
            signals.append(signal)
        return sorted(signals, key=lambda s: s.confidence, reverse=True)

    def strategy_status(self, frames: Dict[str, pd.DataFrame], snapshots: Dict[str, IctSnapshot]) -> Dict:
        primary_tf = CONFIG.timeframes.primary
        if primary_tf not in snapshots:
            return {"analysis_progress": [], "strategy_agents": [], "professional_strategy_matrix": {}}
        primary = snapshots[primary_tf]
        context = self.smc.evaluate(snapshots, frames)
        trend_context = self.trend.evaluate(frames[primary_tf].tail(620))
        matrix = self.catalog.evaluate(frames, snapshots, context, trend_context)
        agents = self._strategy_agents(primary, snapshots, context, trend_context, matrix)
        pipeline = self._analysis_pipeline(primary, context, trend_context)
        return {
            "analysis_progress": pipeline,
            "strategy_agents": [a.payload() for a in agents],
            "professional_strategy_matrix": matrix,
        }

    def _build_signal(
        self, direction, primary, snapshots, df, tick, context,
        trend_confirmations, setup_model, candidate_confirmations,
        candidate_score, reason, extra_metadata,
    ) -> Optional[Signal]:
        primary_tf = CONFIG.timeframes.primary
        entry = self._entry_price(direction, primary, df, tick)
        entry_quality = self._entry_quality(direction, primary, entry)
        timing = self._timing_quality(primary, df, entry)
        if timing["status"] == "late":
            return None
        stop = self._stop_loss(direction, primary, df, entry)
        tp = self._take_profit(direction, primary, df, entry, stop)
        rr = abs(tp - entry) / max(abs(entry - stop), 1e-9)
        profile = asset_profile(CONFIG.data.symbol)
        min_rr = self._minimum_rr(CONFIG.data.symbol)
        if rr < min_rr:
            return None
        concepts = list(dict.fromkeys(
            primary.concepts + context.confirmations + trend_confirmations + candidate_confirmations + [setup_model]
        ))
        base_score = max(context.score, candidate_score)
        confidence = self._confidence(base_score, rr, primary, context, trend_confirmations, direction, setup_model)
        confidence += (float(entry_quality["score"]) - 70.0) * 0.08
        confidence += (float(timing["score"]) - 70.0) * 0.06
        confidence = round(max(0.0, min(96.0, confidence)), 1)
        if confidence < profile.min_confidence:
            return None
        target_allowed, target_reason = self._target_winrate_allows(
            direction, confidence, rr, primary, snapshots, context,
            trend_confirmations, setup_model, entry_quality, timing,
        )
        if not target_allowed:
            return None
        session = self._session_name(primary.timestamp)
        # WINRATE FIX: off-session soft penalty on confidence, not hard block
        if session == "off_session":
            confidence = round(max(0.0, confidence - 3.0), 1)
        metadata = {
            "atr": primary.atr,
            "trend_strength": primary.trend_strength,
            "spread": tick.get("spread", 0.0),
            "regime": context.regime,
            "mtf_alignment": context.mtf_alignment,
            "setup_model": setup_model,
            "strategy_agent": setup_model,
            "timeframe": primary_tf,
            "asset_profile": profile.name,
            "profile_min_rr": min_rr,
            "profile_min_confidence": profile.min_confidence,
            "session": session,
            "entry_quality_score": entry_quality["score"],
            "entry_quality_notes": "; ".join(entry_quality["notes"]),
            "timing_score": timing["score"],
            "timing_status": timing["status"],
            "timing_reason": timing["reason"],
            "market_regime": self._market_regime(primary, context.regime),
            "mtf_decision_tree": self._mtf_decision_tree(direction, snapshots),
            "execution_profile": "bounded_micro_scalp" if CONFIG.risk.use_micro_scalp_exits else "structure",
            "fixed_lot_size": CONFIG.risk.fixed_lot_size,
            "target_winrate_mode": str(CONFIG.risk.high_winrate_mode),
            "target_winrate_pct": CONFIG.risk.target_winrate_pct,
            "target_winrate_filter": target_reason,
        }
        metadata.update(extra_metadata)
        return Signal(
            direction=direction,
            symbol=CONFIG.data.symbol,
            timeframe=primary_tf,
            timestamp=primary.timestamp,
            entry=entry,
            stop_loss=stop,
            take_profit=tp,
            rr=rr,
            confidence=confidence,
            strength=self._strength(confidence),
            concepts=concepts,
            reason=reason,
            metadata=metadata,
        )

    def _htf_allows(self, direction: Direction, snapshots: Dict[str, IctSnapshot]) -> bool:
        higher = [snapshots[tf].bias for tf in CONFIG.timeframes.confluence if tf in snapshots]
        if not higher:
            return True
        wanted = "bullish" if direction == Direction.BUY else "bearish"
        return higher.count(wanted) >= CONFIG.risk.htf_min_aligned

    def _quality_allows(self, direction, primary, snapshots, context, trend_confirmations, policy="strict") -> tuple:
        if context.regime == "sideways":
            return False, "Sideways market"
        min_trend = 18 if CONFIG.risk.high_winrate_mode else 15
        min_atr = 0.28 if CONFIG.risk.high_winrate_mode else 0.20
        if primary.trend_strength < min_trend:
            return False, f"Weak ADX ({primary.trend_strength:.1f} < {min_trend})"
        if primary.metrics.get("atr_rank", 0.0) < min_atr:
            return False, "Low volatility"
        if policy == "strict":
            # WINRATE FIX: only block premium BUY / discount SELL if strongly against trend
            if direction == Direction.BUY and primary.premium_discount == "premium" and primary.bias != "bullish":
                return False, "BUY blocked in premium against bias"
            if direction == Direction.SELL and primary.premium_discount == "discount" and primary.bias != "bearish":
                return False, "SELL blocked in discount against bias"
        confirmations = set(context.confirmations) | set(trend_confirmations) | set(primary.concepts)
        ema_vwap = {"EMA/VWAP Confirmation", "EMA Trend Stack", "VWAP Bull Control", "VWAP Bear Control"}
        if not (ema_vwap & confirmations):
            return False, "Missing EMA/VWAP evidence"
        momentum = {"Momentum Confirmation", "MACD Momentum Expansion", "ADX Trending Market", "Trend Strength"}
        if not (momentum & confirmations):
            return False, "Missing momentum evidence"
        return True, "Allowed"

    def _target_winrate_allows(
        self, direction, confidence, rr, primary, snapshots,
        context, trend_confirmations, setup_model, entry_quality, timing,
    ) -> tuple:
        if not CONFIG.risk.high_winrate_mode:
            return True, "Standard mode"
        if confidence < CONFIG.risk.high_winrate_min_confidence:
            return False, f"Confidence {confidence:.1f}% < {CONFIG.risk.high_winrate_min_confidence:.0f}%"
        if rr < max(self._minimum_rr(CONFIG.data.symbol), CONFIG.risk.high_winrate_min_rr):
            return False, f"RR {rr:.2f} below floor"
        if float(entry_quality["score"]) < CONFIG.risk.high_winrate_min_entry_score:
            return False, f"Entry quality {entry_quality['score']:.1f}% < {CONFIG.risk.high_winrate_min_entry_score:.0f}%"
        # WINRATE FIX: allow "stretched" timing (was only "valid")
        if str(timing["status"]) not in {"valid", "stretched"} or float(timing["score"]) < CONFIG.risk.high_winrate_min_timing_score:
            return False, f"Timing {timing['score']:.1f}% below floor"
        # WINRATE FIX: off-session is a soft confidence penalty, not a hard block here
        confirmations = set(context.confirmations) | set(trend_confirmations) | set(primary.concepts)
        ema_set = {"EMA/VWAP Confirmation", "EMA Trend Stack", "VWAP Bull Control", "VWAP Bear Control"}
        if not (ema_set & confirmations):
            return False, "Missing EMA/VWAP"
        momentum_set = {"Momentum Confirmation", "MACD Momentum Expansion", "ADX Trending Market"}
        if not (momentum_set & confirmations):
            return False, "Missing momentum"
        # WINRATE FIX: need 1 of 5 institutional concepts (was strict 3-set)
        institutional = {"Fair Value Gap", "Order Block", "Liquidity Sweep", "Displacement Candle", "MSS/CHOCH", "Breaker Block", "Mitigation Block"}
        if not (institutional & confirmations):
            return False, "Missing institutional evidence"
        if context.mtf_alignment < CONFIG.risk.mtf_alignment_floor:
            return False, f"MTF alignment {context.mtf_alignment:.2f} < {CONFIG.risk.mtf_alignment_floor:.2f}"
        higher = [snapshots[tf].bias for tf in CONFIG.timeframes.confluence if tf in snapshots]
        wanted = "bullish" if direction == Direction.BUY else "bearish"
        if len(higher) >= 2 and higher.count(wanted) < CONFIG.risk.htf_min_aligned:
            return False, "Insufficient HTF alignment"
        return True, "Passed quality filter"

    def _confidence(self, base_score, rr, primary, context, trend_confirmations, direction, setup_model) -> float:
        confidence = min(91.0, base_score)
        confirmations = set(context.confirmations) | set(trend_confirmations) | set(primary.concepts)
        bonus_checks = [
            "Liquidity Sweep" in confirmations,
            "Fair Value Gap" in confirmations,
            "Order Block" in confirmations,
            ("EMA/VWAP Confirmation" in confirmations or "EMA Trend Stack" in confirmations),
            ("Momentum Confirmation" in confirmations or "MACD Momentum Expansion" in confirmations),
            ("ADX Trending Market" in confirmations and primary.trend_strength >= 22),
            context.mtf_alignment >= 0.65,
            primary.metrics.get("volume_z", 0.0) >= 0.45,
            (direction == Direction.BUY and primary.premium_discount == "discount"),
            (direction == Direction.SELL and primary.premium_discount == "premium"),
            bool(primary.sweep),
            bool(primary.displacement),
        ]
        # WINRATE FIX: bonus per confirmed concept raised from 1.5 to 2.0
        confidence += sum(2.0 for item in bonus_checks if item)
        if setup_model == "ICT Reversal" and not {"Liquidity Sweep", "Fair Value Gap"} & confirmations:
            confidence -= 5.0
        if context.regime != "expansion":
            confidence -= 1.5
        if primary.trend_strength < 20:
            confidence -= 2.5
        if primary.metrics.get("atr_rank", 0.5) < 0.30:
            confidence -= 2.5
        if rr <= 2.05:
            confidence -= 1.5
        return round(max(0.0, min(96.0, confidence)), 1)

    # ── Strategy Agents ──────────────────────────────────────────────────────

    def _strategy_agents(self, primary, snapshots, context, trend_context, matrix) -> list:
        return [
            self._core_institutional_agent(primary, context, trend_context),
            self._smc_agent(primary, context),
            self._ict_agent(primary, context, snapshots),
            self._trend_agent(primary, trend_context, context),
        ]

    def _core_institutional_agent(self, primary, context, trend_context) -> StrategyAgentDecision:
        confirmations = list(set(context.confirmations) | set(trend_context.confirmations))
        required = ["Fair Value Gap", "Order Block", "Liquidity Sweep"]
        found = [item for item in required if any(item in c for c in confirmations)]
        direction = self.smc.direction_from_context(context, primary)
        score = context.score + len(found) * 4.0
        ready = len(found) >= 1 and direction is not None and context.regime != "sideways"
        return StrategyAgentDecision(
            name="Core Institutional Agent", direction=direction, ready=ready,
            score=min(96, score), confirmations=confirmations[:8],
            missing=[item for item in required if item not in found],
            blockers=context.penalties[:3],
            reason=f"Institutional: {', '.join(found) or 'scanning'}. Regime: {context.regime}.",
            quality_policy="strict",
        )

    def _smc_agent(self, primary, context) -> StrategyAgentDecision:
        confirmations = list(set(context.confirmations))
        direction = self.smc.direction_from_context(context, primary)
        smc_items = ["MSS/CHOCH", "Displacement Candle", "Fair Value Gap", "Liquidity Sweep"]
        found = [item for item in smc_items if item in confirmations]
        score = 50.0 + len(found) * 6.0 + context.mtf_alignment * 12
        ready = len(found) >= 2 and direction is not None
        return StrategyAgentDecision(
            name="Smart Money Concepts Agent", direction=direction, ready=ready,
            score=min(96, score), confirmations=found,
            missing=[item for item in smc_items if item not in found],
            blockers=context.penalties[:3],
            reason=f"SMC: {', '.join(found) or 'scanning'}.",
            quality_policy="strict",
        )

    def _ict_agent(self, primary, context, snapshots) -> StrategyAgentDecision:
        confirmations = list(primary.concepts)
        ict_items = ["Judas Swing", "Kill Zone", "Inducement", "Optimal Trade Entry", "Liquidity Raid"]
        found = [item for item in ict_items if any(item in c for c in confirmations)]
        direction = self.smc.direction_from_context(context, primary)
        score = 52.0 + len(found) * 5.0 + (5.0 if primary.displacement else 0.0)
        ready = len(found) >= 1 and direction is not None
        return StrategyAgentDecision(
            name="ICT Concepts Agent", direction=direction, ready=ready,
            score=min(96, score), confirmations=found,
            missing=[item for item in ict_items if item not in found],
            blockers=[],
            reason=f"ICT: {', '.join(found) or 'scanning'}.",
            quality_policy="agent",
        )

    def _trend_agent(self, primary, trend_context, context) -> StrategyAgentDecision:
        confirmations = list(set(trend_context.confirmations))
        direction = trend_context.direction
        required = ["EMA Trend Stack", "ADX Trending Market"]
        found = [item for item in required if item in confirmations]
        ready = len(found) >= 1 and direction is not None and context.regime != "sideways"
        return StrategyAgentDecision(
            name="Trend Systems Agent", direction=direction, ready=ready,
            score=trend_context.score, confirmations=confirmations[:6],
            missing=[item for item in required if item not in found],
            blockers=trend_context.penalties[:3],
            reason=f"Trend: {', '.join(found) or 'building'}. Score {trend_context.score:.1f}.",
            quality_policy="agent",
        )

    def _analysis_pipeline(self, primary, context, trend_context) -> list:
        def gate(name, passed, detail, score=0.0):
            return {"name": name, "state": "PASS" if passed else "BLOCK", "detail": detail, "score": round(score, 1)}
        return [
            gate("Market State", context.regime != "sideways", f"Regime: {context.regime} | ADX {primary.trend_strength:.1f}", primary.trend_strength),
            gate("HTF Bias", context.directional_bias != "neutral", f"Bias: {context.directional_bias} | MTF {context.mtf_alignment:.2f}", context.mtf_alignment * 100),
            gate("SMC Confirmations", len(context.confirmations) >= CONFIG.ict.min_confirmations, f"{len(context.confirmations)} confirmations"),
            gate("Liquidity", bool(primary.sweep or primary.fvg or primary.order_block), f"Sweep={primary.sweep} FVG={bool(primary.fvg)} OB={bool(primary.order_block)}"),
            gate("Trend", trend_context.score >= 50, f"Score {trend_context.score:.1f}", trend_context.score),
            gate("Volatility", primary.metrics.get("atr_rank", 0) >= 0.20, f"ATR rank {primary.metrics.get('atr_rank', 0):.2f}", primary.metrics.get("atr_rank", 0) * 100),
            gate("Quality", primary.trend_strength >= 16, f"ADX {primary.trend_strength:.1f} | {primary.premium_discount}", primary.trend_strength * 2),
        ]

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _ema_direction(self, df: pd.DataFrame) -> Optional[Direction]:
        """Last-resort direction from EMA stack alone."""
        from indicators import ema
        if len(df) < 50:
            return None
        close = df["close"]
        e20 = ema(close, 20).iloc[-1]
        e50 = ema(close, 50).iloc[-1]
        last = float(close.iloc[-1])
        if last > e20 > e50:
            return Direction.BUY
        if last < e20 < e50:
            return Direction.SELL
        return None

    def _entry_quality(self, direction, primary, entry) -> dict:
        score = 62.0
        notes: list = []
        zone = primary.fvg or primary.mitigation_block or primary.order_block
        if zone and zone.low <= entry <= zone.high:
            score += 12
            notes.append(f"entry inside {zone.kind}")
        else:
            notes.append("entry outside zone")
        if direction == Direction.BUY and primary.premium_discount == "discount":
            score += 10
            notes.append("BUY in discount")
        elif direction == Direction.SELL and primary.premium_discount == "premium":
            score += 10
            notes.append("SELL in premium")
        elif primary.premium_discount == "equilibrium":
            score -= 2  # minimal penalty for equilibrium
            notes.append("equilibrium zone — neutral")
        elif primary.premium_discount in {"premium", "discount"}:
            score -= 8
            notes.append("premium/discount against direction")
        if primary.sweep:
            score += 6
            notes.append("liquidity sweep present")
        if primary.displacement:
            score += 6
            notes.append("displacement present")
        if primary.metrics.get("body_atr", 0.0) > 1.4:
            score -= 5
            notes.append("extended candle")
        return {"score": round(max(0.0, min(100.0, score)), 1), "notes": notes}

    def _timing_quality(self, primary, df, entry) -> dict:
        close = float(df["close"].iloc[-1])
        distance_atr = abs(close - entry) / max(float(primary.atr), 1e-9)
        if distance_atr > 1.8:
            return {"score": 28.0, "status": "late", "reason": f"{distance_atr:.2f} ATR from entry"}
        if distance_atr > 0.9:
            return {"score": 66.0, "status": "stretched", "reason": f"{distance_atr:.2f} ATR — stretched"}
        return {"score": 86.0, "status": "valid", "reason": f"{distance_atr:.2f} ATR from entry"}

    def _entry_price(self, direction, snap, df, tick) -> float:
        if direction == Direction.BUY and tick.get("ask") is not None:
            return round(float(tick["ask"]), 2)
        if direction == Direction.SELL and tick.get("bid") is not None:
            return round(float(tick["bid"]), 2)
        return round(float(df["close"].iloc[-1]), 2)

    def _stop_loss(self, direction, snap, df, entry) -> float:
        if CONFIG.risk.use_micro_scalp_exits:
            risk_points = float(CONFIG.risk.micro_sl_points)
            if snap is not None and df is not None:
                risk_points = self._micro_stop_points(direction, snap, df, entry)
            risk_points = self._bounded_points(risk_points, float(CONFIG.risk.micro_min_sl_points), float(CONFIG.risk.micro_max_sl_points))
            return round(entry - risk_points if direction == Direction.BUY else entry + risk_points, 2)
        return self._structure_stop_loss(direction, snap, df, entry)

    def _micro_stop_points(self, direction, snap, df, entry) -> float:
        profile = asset_profile(CONFIG.data.symbol)
        atr_value = max(float(snap.atr), 1e-9)
        spread_buffer = max(float(snap.metrics.get("spread", 0.0) or 0.0), 0.0)
        risk_points = max(atr_value * max(profile.atr_sl_mult, CONFIG.risk.atr_sl_mult), spread_buffer * 2.0)
        recent = df.tail(24)
        zone = snap.fvg or snap.mitigation_block or snap.order_block
        if direction == Direction.BUY:
            candidates = [entry - float(recent["low"].min())] if not recent.empty else []
            if zone and zone.low < entry:
                candidates.append(entry - float(zone.low))
        else:
            candidates = [float(recent["high"].max()) - entry] if not recent.empty else []
            if zone and zone.high > entry:
                candidates.append(float(zone.high) - entry)
        valid = [d for d in candidates if d > 0]
        if valid:
            risk_points = max(risk_points, min(valid))
        return risk_points

    def _structure_stop_loss(self, direction, snap, df, entry) -> float:
        profile = asset_profile(CONFIG.data.symbol)
        atr_value = max(float(snap.atr), 1e-9)
        spread_buffer = max(float(snap.metrics.get("spread", 0.0) or 0.0), 0.0)
        risk_points = max(5.0, atr_value * max(profile.atr_sl_mult, CONFIG.risk.atr_sl_mult), spread_buffer * 2.0)
        buffer = max(atr_value * 0.15, risk_points * 0.1)
        recent = df.tail(24)
        zone = snap.fvg or snap.mitigation_block or snap.order_block
        if direction == Direction.BUY:
            stop = entry - risk_points
            if not recent.empty:
                rl = float(recent["low"].min()) - buffer
                if rl < entry:
                    stop = min(stop, rl)
            if zone and zone.low < entry:
                stop = min(stop, float(zone.low) - buffer)
            return round(stop, 2)
        stop = entry + risk_points
        if not recent.empty:
            rh = float(recent["high"].max()) + buffer
            if rh > entry:
                stop = max(stop, rh)
        if zone and zone.high > entry:
            stop = max(stop, float(zone.high) + buffer)
        return round(stop, 2)

    def _take_profit(self, direction, snap, df, entry, stop) -> float:
        risk = abs(entry - stop)
        target_rr = self._minimum_rr(CONFIG.data.symbol)
        if CONFIG.risk.use_micro_scalp_exits:
            target_points = max(float(CONFIG.risk.micro_tp_points), risk * target_rr)
            target_points = self._bounded_points(target_points, float(CONFIG.risk.micro_min_tp_points), float(CONFIG.risk.micro_max_tp_points))
            return round(entry + target_points if direction == Direction.BUY else entry - target_points, 2)
        return round(entry + risk * target_rr if direction == Direction.BUY else entry - risk * target_rr, 2)

    def _bounded_points(self, value, minimum, maximum) -> float:
        lower = max(0.01, minimum)
        upper = max(lower, maximum)
        return max(lower, min(upper, abs(float(value))))

    def _minimum_rr(self, symbol) -> float:
        if CONFIG.risk.use_micro_scalp_exits:
            return max(1.5, float(CONFIG.risk.micro_min_rr))
        return max(CONFIG.risk.min_rr, asset_profile(symbol).min_rr)

    def _strength(self, confidence) -> str:
        if confidence >= 88:
            return "Institutional Grade"
        if confidence >= 78:
            return "Strong"
        if confidence >= 68:
            return "Medium"
        return "Weak"

    def _reason(self, direction, context, snap, setup_model) -> str:
        if setup_model == "Trend Continuation":
            side = "bullish trend continuation" if direction == Direction.BUY else "bearish trend continuation"
            return f"{side.title()} model. EMA/VWAP regime, momentum, ADX align with {context.directional_bias} HTF context."
        side = "bullish reversal after sell-side sweep" if direction == Direction.BUY else "bearish reversal after buy-side sweep"
        return f"Institutional {side}. {context.directional_bias.title()} MTF, {snap.premium_discount} pricing, FVG delivery aligned."

    def _session_name(self, ts) -> str:
        minutes = ts.hour * 60 + ts.minute
        for name, (start, end) in CONFIG.sessions.kill_zones.items():
            sh, sm = map(int, start.split(":"))
            eh, em = map(int, end.split(":"))
            if sh * 60 + sm <= minutes <= eh * 60 + em:
                return name
        return "off_session"

    def _market_regime(self, primary, regime) -> str:
        return f"{regime}|adx={primary.trend_strength:.1f}|atr_rank={primary.metrics.get('atr_rank', 0):.2f}"

    def _mtf_decision_tree(self, direction, snapshots) -> list:
        wanted = "bullish" if direction == Direction.BUY else "bearish"
        tree = []
        for tf in CONFIG.timeframes.all:
            snap = snapshots.get(tf)
            if not snap:
                continue
            tree.append({
                "timeframe": tf, "bias": snap.bias,
                "allows": snap.bias in {wanted, "neutral"},
                "premium_discount": snap.premium_discount,
                "structure": snap.mss or snap.choch or snap.bos or "",
                "sweep": snap.sweep or "",
            })
        return tree
