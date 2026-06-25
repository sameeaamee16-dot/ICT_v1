from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import CONFIG, IctConfig
from indicators import adx, atr, ema, rsi, vwap
from models import Direction, IctSnapshot, Zone


@dataclass(frozen=True)
class Swing:
    kind: str
    pivot_time: datetime
    confirm_time: datetime
    price: float
    index: int


class IctEngine:
    def __init__(self, config: IctConfig = CONFIG.ict):
        self.config = config

    def analyze(self, df: pd.DataFrame, timeframe: str) -> IctSnapshot:
        if len(df) < 80:
            raise ValueError(f"{timeframe} requires at least 80 closed candles")
        work = df.copy()
        work["atr"] = atr(work)
        work["ema_fast"] = ema(work["close"], 21)
        work["ema_slow"] = ema(work["close"], 55)
        work["vwap"] = vwap(work)
        work["adx"] = adx(work)
        work["rsi"] = rsi(work["close"])
        atr_tail = work["atr"].tail(240).dropna()
        atr_rank = 0.5
        if not atr_tail.empty and pd.notna(atr_tail.iloc[-1]):
            atr_rank = float((atr_tail <= atr_tail.iloc[-1]).mean())
        work["atr_rank"] = atr_rank

        swings = self._confirmed_swings(work)
        last = work.iloc[-1]
        zones: List[Zone] = []
        fvg = self._latest_fvg(work)
        order_block = self._latest_order_block(work)
        breaker = self._latest_breaker_block(work, swings)
        mitigation = self._latest_mitigation_block(work, order_block)
        rejection = self._latest_rejection_block(work)
        for zone in [fvg, order_block, breaker, mitigation, rejection]:
            if zone:
                zones.append(zone)

        liquidity = self._liquidity_levels(work, swings)
        sweep = self._liquidity_sweep(work, liquidity)
        mss, bos, choch = self._structure_events(work, swings)
        displacement = self._displacement(work)
        premium_discount = self._premium_discount(work)
        bias = self._bias(work, mss, choch, bos)

        concepts = self._concepts(work, swings, liquidity, fvg, order_block, breaker, mitigation, rejection, sweep, mss, bos, choch, displacement, premium_discount, bias)
        return IctSnapshot(
            timeframe=timeframe,
            timestamp=work.index[-1].to_pydatetime(),
            bias=bias,
            trend_strength=float(last["adx"]) if pd.notna(last["adx"]) else 0.0,
            atr=float(last["atr"]),
            vwap=float(last["vwap"]),
            ema_fast=float(last["ema_fast"]),
            ema_slow=float(last["ema_slow"]),
            concepts=concepts,
            zones=zones,
            liquidity_levels=liquidity,
            premium_discount=premium_discount,
            displacement=displacement,
            mss=mss,
            choch=choch,
            bos=bos,
            sweep=sweep,
            fvg=fvg,
            order_block=order_block,
            breaker_block=breaker,
            mitigation_block=mitigation,
            rejection_block=rejection,
            metrics={
                "rsi": float(last["rsi"]) if pd.notna(last["rsi"]) else 50.0,
                "atr_rank": float(last["atr_rank"]) if pd.notna(last["atr_rank"]) else 0.5,
                "volume_z": self._volume_z(work),
                "body_atr": abs(float(last["close"] - last["open"])) / max(float(last["atr"]), 1e-9),
            },
        )

    def _confirmed_swings(self, df: pd.DataFrame) -> List[Swing]:
        left, right = self.config.swing_left, self.config.swing_right
        swings: List[Swing] = []
        for i in range(left, len(df) - right):
            window = df.iloc[i - left : i + right + 1]
            high = df["high"].iloc[i]
            low = df["low"].iloc[i]
            confirm_time = df.index[i + right].to_pydatetime()
            if high == window["high"].max() and window["high"].iloc[:left + 1].idxmax() == df.index[i]:
                swings.append(Swing("high", df.index[i].to_pydatetime(), confirm_time, float(high), i + right))
            if low == window["low"].min() and window["low"].iloc[:left + 1].idxmin() == df.index[i]:
                swings.append(Swing("low", df.index[i].to_pydatetime(), confirm_time, float(low), i + right))
        return [s for s in swings if s.index < len(df)]

    def _last_swings(self, swings: List[Swing], kind: str, n: int = 3) -> List[Swing]:
        return [s for s in swings if s.kind == kind][-n:]

    def _structure_events(self, df: pd.DataFrame, swings: List[Swing]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        highs = self._last_swings(swings, "high", 3)
        lows = self._last_swings(swings, "low", 3)
        if len(highs) < 2 or len(lows) < 2:
            return None, None, None
        close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])
        last_high, prior_high = highs[-1], highs[-2]
        last_low, prior_low = lows[-1], lows[-2]
        trend_up = last_high.price > prior_high.price and last_low.price > prior_low.price
        trend_down = last_high.price < prior_high.price and last_low.price < prior_low.price
        bos = None
        choch = None
        mss = None
        if prev_close <= last_high.price < close:
            bos = "bullish BOS" if trend_up else None
            choch = "bullish CHOCH" if trend_down else None
            mss = "bullish MSS"
        if prev_close >= last_low.price > close:
            bos = "bearish BOS" if trend_down else None
            choch = "bearish CHOCH" if trend_up else None
            mss = "bearish MSS"
        return mss, bos, choch

    def _latest_fvg(self, df: pd.DataFrame) -> Optional[Zone]:
        atr_val = float(df["atr"].iloc[-1])
        min_gap = atr_val * self.config.fvg_min_atr
        for i in range(len(df) - 1, max(2, len(df) - 80), -1):
            a, c = df.iloc[i - 2], df.iloc[i]
            if c["low"] > a["high"] and c["low"] - a["high"] >= min_gap:
                touched = bool((df["low"].iloc[i + 1 :] <= c["low"]).any()) if i + 1 < len(df) else False
                return Zone("FVG", Direction.BUY, df.index[i - 2].to_pydatetime(), df.index[i].to_pydatetime(), float(a["high"]), float(c["low"]), min(100, (c["low"] - a["high"]) / max(atr_val, 1e-9) * 60), touched)
            if c["high"] < a["low"] and a["low"] - c["high"] >= min_gap:
                touched = bool((df["high"].iloc[i + 1 :] >= c["high"]).any()) if i + 1 < len(df) else False
                return Zone("FVG", Direction.SELL, df.index[i - 2].to_pydatetime(), df.index[i].to_pydatetime(), float(c["high"]), float(a["low"]), min(100, (a["low"] - c["high"]) / max(atr_val, 1e-9) * 60), touched)
        return None

    def _latest_order_block(self, df: pd.DataFrame) -> Optional[Zone]:
        atr_val = float(df["atr"].iloc[-1])
        for i in range(len(df) - 1, max(5, len(df) - self.config.ob_lookback), -1):
            body = abs(float(df["close"].iloc[i] - df["open"].iloc[i]))
            if body < atr_val * self.config.displacement_atr_mult:
                continue
            direction = Direction.BUY if df["close"].iloc[i] > df["open"].iloc[i] else Direction.SELL
            opposite = df.iloc[max(0, i - 8) : i]
            if direction == Direction.BUY:
                candidates = opposite[opposite["close"] < opposite["open"]]
            else:
                candidates = opposite[opposite["close"] > opposite["open"]]
            if candidates.empty:
                continue
            candle = candidates.iloc[-1]
            start = candidates.index[-1].to_pydatetime()
            return Zone("Order Block", direction, start, df.index[i].to_pydatetime(), float(candle["low"]), float(candle["high"]), min(100, body / max(atr_val, 1e-9) * 55))
        return None

    def _latest_breaker_block(self, df: pd.DataFrame, swings: List[Swing]) -> Optional[Zone]:
        ob = self._latest_order_block(df)
        mss, _, choch = self._structure_events(df, swings)
        if not ob or not (mss or choch):
            return None
        opposite = Direction.SELL if ob.direction == Direction.BUY else Direction.BUY
        close = float(df["close"].iloc[-1])
        broken = close < ob.low if ob.direction == Direction.BUY else close > ob.high
        if broken:
            return Zone("Breaker Block", opposite, ob.start_time, df.index[-1].to_pydatetime(), ob.low, ob.high, ob.strength * 0.9)
        return None

    def _latest_mitigation_block(self, df: pd.DataFrame, ob: Optional[Zone]) -> Optional[Zone]:
        if not ob:
            return None
        recent = df.tail(self.config.mitigation_lookback)
        touched = ((recent["low"] <= ob.high) & (recent["high"] >= ob.low)).iloc[-8:].any()
        rejected_up = float(df["close"].iloc[-1]) > ob.high and ob.direction == Direction.BUY
        rejected_down = float(df["close"].iloc[-1]) < ob.low and ob.direction == Direction.SELL
        if touched and (rejected_up or rejected_down):
            return Zone("Mitigation Block", ob.direction, ob.start_time, df.index[-1].to_pydatetime(), ob.low, ob.high, ob.strength * 0.85, True, True)
        return None

    def _latest_rejection_block(self, df: pd.DataFrame) -> Optional[Zone]:
        last = df.iloc[-1]
        atr_val = max(float(last["atr"]), 1e-9)
        upper = float(last["high"] - max(last["open"], last["close"]))
        lower = float(min(last["open"], last["close"]) - last["low"])
        body = abs(float(last["close"] - last["open"]))
        if lower > atr_val * 0.45 and lower > body * 1.7:
            return Zone("Rejection Block", Direction.BUY, df.index[-1].to_pydatetime(), df.index[-1].to_pydatetime(), float(last["low"]), float(min(last["open"], last["close"])), min(100, lower / atr_val * 45))
        if upper > atr_val * 0.45 and upper > body * 1.7:
            return Zone("Rejection Block", Direction.SELL, df.index[-1].to_pydatetime(), df.index[-1].to_pydatetime(), float(max(last["open"], last["close"])), float(last["high"]), min(100, upper / atr_val * 45))
        return None

    def _liquidity_levels(self, df: pd.DataFrame, swings: List[Swing]) -> List[Zone]:
        atr_val = float(df["atr"].iloc[-1])
        tol = atr_val * self.config.equal_level_atr_tolerance
        levels: List[Zone] = []
        for kind, direction in [("high", Direction.SELL), ("low", Direction.BUY)]:
            selected = self._last_swings(swings, kind, 12)
            for i in range(len(selected) - 1):
                a, b = selected[i], selected[i + 1]
                if abs(a.price - b.price) <= tol:
                    low, high = sorted([a.price, b.price])
                    levels.append(Zone(f"Equal {'Highs' if kind == 'high' else 'Lows'}", direction, a.confirm_time, b.confirm_time, low, high, 75))
        latest_high = self._last_swings(swings, "high", 1)
        latest_low = self._last_swings(swings, "low", 1)
        if latest_high:
            s = latest_high[-1]
            levels.append(Zone("External Buy-Side Liquidity", Direction.SELL, s.confirm_time, df.index[-1].to_pydatetime(), s.price, s.price, 65))
        if latest_low:
            s = latest_low[-1]
            levels.append(Zone("External Sell-Side Liquidity", Direction.BUY, s.confirm_time, df.index[-1].to_pydatetime(), s.price, s.price, 65))
        return levels[-10:]

    def _liquidity_sweep(self, df: pd.DataFrame, levels: List[Zone]) -> Optional[str]:
        last, prev = df.iloc[-1], df.iloc[-2]
        for level in reversed(levels):
            if "High" in level.kind or "Buy-Side" in level.kind:
                if last["high"] > level.high and last["close"] < level.high and prev["close"] <= level.high:
                    return "buy-side liquidity sweep"
            if "Low" in level.kind or "Sell-Side" in level.kind:
                if last["low"] < level.low and last["close"] > level.low and prev["close"] >= level.low:
                    return "sell-side liquidity sweep"
        return None

    def _displacement(self, df: pd.DataFrame) -> Optional[str]:
        last = df.iloc[-1]
        body = abs(float(last["close"] - last["open"]))
        atr_val = max(float(last["atr"]), 1e-9)
        range_val = max(float(last["high"] - last["low"]), 1e-9)
        if body / atr_val >= self.config.displacement_atr_mult and body / range_val > 0.58:
            return "bullish displacement" if last["close"] > last["open"] else "bearish displacement"
        return None

    def _premium_discount(self, df: pd.DataFrame) -> str:
        recent = df.tail(self.config.premium_discount_lookback)
        high, low, close = float(recent["high"].max()), float(recent["low"].min()), float(df["close"].iloc[-1])
        equilibrium = (high + low) / 2.0
        if close < equilibrium * 0.998:
            return "discount"
        if close > equilibrium * 1.002:
            return "premium"
        return "equilibrium"

    def _bias(self, df: pd.DataFrame, mss: Optional[str], choch: Optional[str], bos: Optional[str]) -> str:
        last = df.iloc[-1]
        score = 0
        score += 1 if last["close"] > last["ema_fast"] > last["ema_slow"] else -1 if last["close"] < last["ema_fast"] < last["ema_slow"] else 0
        score += 1 if last["close"] > last["vwap"] else -1
        for event in [mss, choch, bos]:
            if event and "bullish" in event:
                score += 2
            if event and "bearish" in event:
                score -= 2
        return "bullish" if score >= 2 else "bearish" if score <= -2 else "neutral"

    def _volume_z(self, df: pd.DataFrame) -> float:
        vol = df["volume"].tail(80).astype(float)
        std = vol.std()
        if not std or np.isnan(std):
            return 0.0
        return float((vol.iloc[-1] - vol.mean()) / std)

    def _concepts(self, df: pd.DataFrame, swings: List[Swing], liquidity: List[Zone], fvg: Optional[Zone], ob: Optional[Zone], breaker: Optional[Zone], mitigation: Optional[Zone], rejection: Optional[Zone], sweep: Optional[str], mss: Optional[str], bos: Optional[str], choch: Optional[str], displacement: Optional[str], premium_discount: str, bias: str) -> List[str]:
        concepts = ["Market Structure", "Dealing Range", "Risk Management"]
        concepts.extend(self._structure_labels(swings))
        if bias == "bullish":
            concepts.append("Daily Bias Bullish")
        elif bias == "bearish":
            concepts.append("Daily Bias Bearish")
        if premium_discount == "discount":
            concepts.append("Discount Zone")
        elif premium_discount == "premium":
            concepts.append("Premium Zone")
        else:
            concepts.append("Equilibrium")
        for item in [sweep, mss, bos, choch, displacement]:
            if item:
                label = item.title()
                concepts.append(label.replace("Bos", "BOS").replace("Choch", "CHOCH").replace("Mss", "MSS"))
        if sweep:
            concepts.append("Liquidity Sweep")
            concepts.append("Turtle Soup")
            if "buy-side" in sweep:
                concepts.append("Buy Side Liquidity")
            if "sell-side" in sweep:
                concepts.append("Sell Side Liquidity")
        if liquidity:
            concepts.append("Liquidity")
        if fvg:
            concepts.append("Fair Value Gap")
            concepts.append("Bullish FVG" if fvg.direction == Direction.BUY else "Bearish FVG")
        if ob:
            concepts.append("Order Block")
            concepts.append("Bullish OB" if ob.direction == Direction.BUY else "Bearish OB")
        if breaker:
            concepts.append("Breaker Block")
        if mitigation:
            concepts.append("Mitigation Block")
        if rejection:
            concepts.append("Rejection Block")
        if (premium_discount == "discount" and (fvg and fvg.direction == Direction.BUY or ob and ob.direction == Direction.BUY)) or (
            premium_discount == "premium" and (fvg and fvg.direction == Direction.SELL or ob and ob.direction == Direction.SELL)
        ):
            concepts.append("Optimal Trade Entry")
        if any("Equal Highs" == z.kind for z in liquidity):
            concepts.append("Equal Highs")
            concepts.append("Buy Side Liquidity")
        if any("Equal Lows" == z.kind for z in liquidity):
            concepts.append("Equal Lows")
            concepts.append("Sell Side Liquidity")
        if self._inducement(df, swings):
            concepts.append("Inducement")
        concepts.extend(self._session_concepts(df))
        return list(dict.fromkeys(concepts))

    def _structure_labels(self, swings: List[Swing]) -> List[str]:
        labels: List[str] = []
        highs = self._last_swings(swings, "high", 2)
        lows = self._last_swings(swings, "low", 2)
        if len(highs) >= 2:
            labels.append("Higher High" if highs[-1].price > highs[-2].price else "Lower High")
        if len(lows) >= 2:
            labels.append("Higher Low" if lows[-1].price > lows[-2].price else "Lower Low")
        return labels

    def _inducement(self, df: pd.DataFrame, swings: List[Swing]) -> bool:
        recent = swings[-8:]
        if len(recent) < 4:
            return False
        close = float(df["close"].iloc[-1])
        near = [s for s in recent if abs(close - s.price) <= float(df["atr"].iloc[-1]) * 0.35]
        return len(near) >= 1

    def _session_concepts(self, df: pd.DataFrame) -> List[str]:
        ts = df.index[-1]
        minutes = ts.hour * 60 + ts.minute
        concepts = []
        for name, (start, end) in CONFIG.sessions.kill_zones.items():
            sh, sm = map(int, start.split(":"))
            eh, em = map(int, end.split(":"))
            if sh * 60 + sm <= minutes <= eh * 60 + em:
                concepts.append(f"{name.replace('_', ' ').title()} Kill Zone")
        day = df[df.index.date == ts.date()]
        if len(day) > 10:
            early = day.between_time("00:00", "06:30")
            if not early.empty and (df["high"].iloc[-1] > early["high"].max() or df["low"].iloc[-1] < early["low"].min()):
                concepts.append("Judas Swing")
                concepts.append("Session Manipulation")
                concepts.append("Power of 3")
        return concepts
