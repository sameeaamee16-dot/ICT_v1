from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict

import pandas as pd

from config import CONFIG, DataConfig
from indicators import normalize_ohlcv, resample_ohlcv
from logger import get_logger

log = get_logger(__name__)


class MarketDataFeed(ABC):
    @abstractmethod
    def get_closed_candles(self, timeframe: str, bars: int) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_tick(self) -> Dict[str, float]:
        raise NotImplementedError

    def get_candles(self, timeframe: str, bars: int, include_current: bool = False) -> pd.DataFrame:
        return self.get_closed_candles(timeframe, bars)

    def get_multi_timeframe(self, bars: int = 1500, include_current: bool = False) -> Dict[str, pd.DataFrame]:
        base = self.get_candles("1m", bars, include_current=include_current)
        return {tf: (base if tf == "1m" else resample_ohlcv(base, tf)) for tf in CONFIG.timeframes.all}


class MT5Feed(MarketDataFeed):
    TF_MAP = {}

    def __init__(self, config: DataConfig = CONFIG.data):
        self.config = config
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise RuntimeError("MetaTrader5 package is not installed") from exc
        self.mt5 = mt5
        self.TF_MAP = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "1h": mt5.TIMEFRAME_H1,
        }
        if not mt5.initialize():
            raise RuntimeError(
                f"MT5 initialize failed: {mt5.last_error()}. "
                "Open MetaTrader 5, login to your broker, then restart this bot."
            )
        self.symbol = self._select_symbol()
        log.info("MT5 feed initialized for %s", self.symbol)

    def _select_symbol(self) -> str:
        for symbol in self.config.mt5_symbol_candidates:
            info = self.mt5.symbol_info(symbol)
            if info is not None:
                self.mt5.symbol_select(symbol, True)
                return symbol
        discovered = self.discover_symbols()
        suffix = f" Broker matching symbols found: {discovered}" if discovered else " No broker XAU/GOLD matching symbols found."
        raise RuntimeError(f"No valid MT5 XAUUSD symbol found among {self.config.mt5_symbol_candidates}.{suffix}")

    def discover_gold_symbols(self) -> list[str]:
        return self.discover_symbols()

    def discover_symbols(self) -> list[str]:
        symbols = self.mt5.symbols_get()
        if symbols is None:
            return []
        terms = ("XAU", "GOLD")
        return [s.name for s in symbols if any(term in s.name.upper() for term in terms)]

    def get_closed_candles(self, timeframe: str, bars: int) -> pd.DataFrame:
        return self.get_candles(timeframe, bars, include_current=False)

    def get_candles(self, timeframe: str, bars: int, include_current: bool = False) -> pd.DataFrame:
        tf = self.TF_MAP[timeframe.lower()]
        start_pos = 0 if include_current else 1
        rates = self.mt5.copy_rates_from_pos(self.symbol, tf, start_pos, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5 returned no rates for {self.symbol} {timeframe}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = normalize_ohlcv(df)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    def get_multi_timeframe(self, bars: int = 1500, include_current: bool = False) -> Dict[str, pd.DataFrame]:
        return {tf: self.get_candles(tf, bars, include_current=include_current) for tf in CONFIG.timeframes.all}

    def get_tick(self) -> Dict[str, float]:
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            last = self.get_closed_candles("1m", 2)["close"].iloc[-1]
            return {"bid": last, "ask": last, "spread": 0.0, "time": datetime.now(timezone.utc).timestamp()}
        return {
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "spread": float((tick.ask - tick.bid) * 10),
            "time": float(tick.time),
        }


def create_feed() -> MarketDataFeed:
    return MT5Feed()
