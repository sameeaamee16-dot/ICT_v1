from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from config import CONFIG
from models import Direction, IctSnapshot


@dataclass(frozen=True)
class SmcContext:
    directional_bias: str
    confirmations: List[str]
    penalties: List[str]
    score: float
    regime: str
    mtf_alignment: float


class SmcEngine:
    def evaluate(self, snapshots: Dict[str, IctSnapshot], frames: Dict[str, pd.DataFrame]) -> SmcContext:
        primary = snapshots[CONFIG.timeframes.primary]
        confirmations: List[str] = []
        penalties: List[str] = []

        higher = [snapshots[tf].bias for tf in CONFIG.timeframes.confluence if tf in snapshots]
        bullish = higher.count("bullish")
        bearish = higher.count("bearish")
        if bullish > bearish:
            bias = "bullish"
        elif bearish > bullish:
            bias = "bearish"
        else:
            bias = primary.bias
        mtf_alignment = max(bullish, bearish) / max(len(higher), 1)
        if mtf_alignment >= 0.66 and bias != "neutral":
            confirmations.append("Multi Timeframe Bias")
        else:
            penalties.append("Mixed higher-timeframe bias")

        if primary.sweep:
            confirmations.append("Liquidity Sweep")
        if "Turtle Soup" in primary.concepts:
            confirmations.append("Turtle Soup")
        if primary.displacement:
            confirmations.append("Displacement Candle")
        if primary.fvg:
            confirmations.append("Fair Value Gap")
            confirmations.append("Bullish FVG" if primary.fvg.direction == Direction.BUY else "Bearish FVG")
        if primary.mss or primary.choch:
            confirmations.append("MSS/CHOCH")
        if primary.order_block:
            confirmations.append("Order Block")
            confirmations.append("Bullish OB" if primary.order_block.direction == Direction.BUY else "Bearish OB")
        if primary.breaker_block:
            confirmations.append("Breaker Block")
        if primary.mitigation_block:
            confirmations.append("Mitigation Block")
        if "Judas Swing" in primary.concepts:
            confirmations.append("Judas Swing")
        if "Session Manipulation" in primary.concepts:
            confirmations.append("Session Manipulation")
        if "Inducement" in primary.concepts:
            confirmations.append("Inducement")
        if "New York Am Kill Zone" in primary.concepts or "London Kill Zone" in primary.concepts:
            confirmations.append("Kill Zone")
        for item in ["Power of 3", "Optimal Trade Entry", "Discount Zone", "Premium Zone", "Daily Bias Bullish", "Daily Bias Bearish"]:
            if item in primary.concepts:
                confirmations.append(item)

        if primary.metrics.get("volume_z", 0.0) > 0.55:
            confirmations.append("Volume Expansion")
        if primary.trend_strength >= 20:
            confirmations.append("Trend Strength")
        if primary.metrics.get("atr_rank", 0.5) >= CONFIG.ict.low_atr_percentile:
            confirmations.append("ATR Volatility")
        else:
            penalties.append("Low volatility")
        if primary.bias == bias and bias != "neutral":
            confirmations.append("Execution TF Bias")
        if self._momentum_confirms(primary, bias):
            confirmations.append("Momentum Confirmation")
        if self._ema_vwap_confirms(primary, bias):
            confirmations.append("EMA/VWAP Confirmation")
        if self._candle_imbalance(primary, bias):
            confirmations.append("Candle Imbalance")

        regime = self._regime(primary)
        if regime == "sideways":
            penalties.append("Sideways market")
        if primary.trend_strength < CONFIG.ict.sideways_adx_threshold:
            penalties.append("Weak trend strength")

        score = 48.0 + len(set(confirmations)) * 4.8 - len(set(penalties)) * 6.5
        score += min(12.0, primary.metrics.get("body_atr", 0.0) * 4)
        score += min(7.0, max(0.0, primary.metrics.get("volume_z", 0.0)) * 2)
        score += 6.0 if mtf_alignment >= 0.9 else 0.0
        return SmcContext(bias, list(dict.fromkeys(confirmations)), list(dict.fromkeys(penalties)), max(0, min(100, score)), regime, mtf_alignment)

    def direction_from_context(self, context: SmcContext, primary: IctSnapshot) -> Optional[Direction]:
        bullish_terms = " ".join([primary.sweep or "", primary.displacement or "", primary.mss or "", primary.choch or "", primary.bos or ""]).lower()
        if context.directional_bias == "bullish" and ("sell-side" in bullish_terms or "bullish" in bullish_terms):
            return Direction.BUY
        if context.directional_bias == "bearish" and ("buy-side" in bullish_terms or "bearish" in bullish_terms):
            return Direction.SELL
        return None

    def _momentum_confirms(self, snap: IctSnapshot, bias: str) -> bool:
        rsi = snap.metrics.get("rsi", 50.0)
        return (bias == "bullish" and rsi > 52) or (bias == "bearish" and rsi < 48)

    def _ema_vwap_confirms(self, snap: IctSnapshot, bias: str) -> bool:
        if bias == "bullish":
            return snap.ema_fast > snap.ema_slow and snap.metrics.get("rsi", 50) > 50
        if bias == "bearish":
            return snap.ema_fast < snap.ema_slow and snap.metrics.get("rsi", 50) < 50
        return False

    def _candle_imbalance(self, snap: IctSnapshot, bias: str) -> bool:
        body_atr = snap.metrics.get("body_atr", 0.0)
        if body_atr < 0.65 or not snap.displacement:
            return False
        return (bias == "bullish" and "bullish" in snap.displacement) or (bias == "bearish" and "bearish" in snap.displacement)

    def _regime(self, snap: IctSnapshot) -> str:
        if snap.trend_strength < CONFIG.ict.sideways_adx_threshold or snap.metrics.get("atr_rank", 0.5) < CONFIG.ict.low_atr_percentile:
            return "sideways"
        if snap.trend_strength > 28 and snap.metrics.get("atr_rank", 0.5) > 0.65:
            return "expansion"
        return "balanced"
