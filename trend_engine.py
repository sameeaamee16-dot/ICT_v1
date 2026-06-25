from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from indicators import adx, atr, ema, vwap
from models import Direction


@dataclass(frozen=True)
class TrendContext:
    direction: Optional[Direction]
    confirmations: List[str]
    penalties: List[str]
    score: float


class TrendEngine:
    def evaluate(self, df: pd.DataFrame) -> TrendContext:
        if len(df) < 120:
            return TrendContext(None, [], ["Insufficient trend history"], 0.0)
        work = df.copy()
        work["atr"] = atr(work)
        work["ema_20"] = ema(work["close"], 20)
        work["ema_50"] = ema(work["close"], 50)
        work["ema_200"] = ema(work["close"], 200)
        work["vwap"] = vwap(work)
        work["adx"] = adx(work)
        macd_line, signal_line, hist = self._macd(work["close"])
        work["macd_hist"] = hist
        upper, lower, width = self._bollinger(work["close"])
        work["bb_upper"] = upper
        work["bb_lower"] = lower
        work["bb_width"] = width
        supertrend_dir = self._supertrend_direction(work)

        last = work.iloc[-1]
        prev = work.iloc[-2]
        confirmations: List[str] = []
        penalties: List[str] = []
        bull = 0
        bear = 0

        if last["close"] > last["ema_20"] > last["ema_50"]:
            bull += 1
            confirmations.append("EMA Trend Stack")
        elif last["close"] < last["ema_20"] < last["ema_50"]:
            bear += 1
            confirmations.append("EMA Trend Stack")
        else:
            penalties.append("No clean EMA stack")

        if pd.notna(last["ema_200"]):
            if last["close"] > last["ema_200"]:
                bull += 1
                confirmations.append("200 EMA Bull Regime")
            elif last["close"] < last["ema_200"]:
                bear += 1
                confirmations.append("200 EMA Bear Regime")

        if last["close"] > last["vwap"]:
            bull += 1
            confirmations.append("VWAP Bull Control")
        elif last["close"] < last["vwap"]:
            bear += 1
            confirmations.append("VWAP Bear Control")

        donchian_high = work["high"].shift(1).rolling(55, min_periods=30).max().iloc[-1]
        donchian_low = work["low"].shift(1).rolling(55, min_periods=30).min().iloc[-1]
        if pd.notna(donchian_high) and last["close"] > donchian_high:
            bull += 2
            confirmations.append("Donchian Breakout")
        if pd.notna(donchian_low) and last["close"] < donchian_low:
            bear += 2
            confirmations.append("Donchian Breakdown")

        if last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]:
            bull += 1
            confirmations.append("MACD Momentum Expansion")
        elif last["macd_hist"] < 0 and last["macd_hist"] < prev["macd_hist"]:
            bear += 1
            confirmations.append("MACD Momentum Expansion")

        width_rank = work["bb_width"].tail(120).rank(pct=True).iloc[-1]
        if width_rank > 0.7 and last["close"] > last["bb_upper"]:
            bull += 1
            confirmations.append("Bollinger Expansion Breakout")
        elif width_rank > 0.7 and last["close"] < last["bb_lower"]:
            bear += 1
            confirmations.append("Bollinger Expansion Breakdown")
        elif width_rank < 0.25:
            penalties.append("Bollinger Squeeze No Expansion")

        if supertrend_dir == Direction.BUY:
            bull += 1
            confirmations.append("Supertrend Bullish")
        elif supertrend_dir == Direction.SELL:
            bear += 1
            confirmations.append("Supertrend Bearish")

        if pd.notna(last["adx"]) and last["adx"] >= 22:
            confirmations.append("ADX Trending Market")
        else:
            penalties.append("ADX trend too weak")

        direction = Direction.BUY if bull >= bear + 2 else Direction.SELL if bear >= bull + 2 else None
        score = 45 + max(bull, bear) * 7 - len(penalties) * 5
        if pd.notna(last["adx"]):
            score += min(10, max(0, float(last["adx"]) - 20) * 0.7)
        return TrendContext(direction, list(dict.fromkeys(confirmations)), penalties, max(0, min(100, score)))

    def _macd(self, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        fast = ema(close, 12)
        slow = ema(close, 26)
        line = fast - slow
        signal = ema(line, 9)
        return line, signal, line - signal

    def _bollinger(self, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        mid = close.rolling(20, min_periods=20).mean()
        std = close.rolling(20, min_periods=20).std()
        upper = mid + std * 2
        lower = mid - std * 2
        width = (upper - lower) / mid.replace(0, pd.NA)
        return upper, lower, width

    def _supertrend_direction(self, df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> Optional[Direction]:
        if len(df) < period + 5:
            return None
        hl2 = (df["high"] + df["low"]) / 2
        atr_val = df["atr"].bfill()
        upper = hl2 + mult * atr_val
        lower = hl2 - mult * atr_val
        trend = Direction.BUY
        final_upper = upper.copy()
        final_lower = lower.copy()
        for i in range(1, len(df)):
            if upper.iloc[i] < final_upper.iloc[i - 1] or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]:
                final_upper.iloc[i] = upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i - 1]
            if lower.iloc[i] > final_lower.iloc[i - 1] or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]:
                final_lower.iloc[i] = lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i - 1]
            if trend == Direction.BUY and df["close"].iloc[i] < final_lower.iloc[i]:
                trend = Direction.SELL
            elif trend == Direction.SELL and df["close"].iloc[i] > final_upper.iloc[i]:
                trend = Direction.BUY
        return trend
