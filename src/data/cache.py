from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .binance_client import INTERVAL_MS, KLINE_COLUMNS, BinanceClient

LOG = logging.getLogger(__name__)

FLOAT_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "taker_base", "taker_quote"]
INT_COLUMNS = ["open_time", "close_time", "trades"]


def klines_to_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df = df.drop(columns=["ignore"])
    for col in FLOAT_COLUMNS:
        df[col] = df[col].astype(float)
    for col in INT_COLUMNS:
        df[col] = df[col].astype("int64")
    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    return df


class CandleCache:
    def __init__(self, root: str | Path = "data/cache") -> None:
        self.root = Path(root)

    def path(self, symbol: str, interval: str) -> Path:
        return self.root / interval / f"{symbol}.parquet"

    def load(self, symbol: str, interval: str) -> pd.DataFrame | None:
        p = self.path(symbol, interval)
        if p.exists():
            return pd.read_parquet(p)
        return None

    def save(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        p = self.path(symbol, interval)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)

    def fetch(
        self,
        client: BinanceClient,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        force: bool = False,
    ) -> pd.DataFrame:
        """Return candles for [start_ms, end_ms], using/extending the parquet cache."""
        step = INTERVAL_MS[interval]
        existing = None if force else self.load(symbol, interval)

        if existing is not None and not existing.empty:
            last_open = int(existing["open_time"].iloc[-1])
            if last_open >= end_ms - step:
                return existing
            fetch_start = last_open + step
            base = existing
        else:
            fetch_start = int(start_ms)
            base = None

        rows = list(client.iter_klines(symbol, interval, fetch_start, end_ms))
        if not rows:
            return base if base is not None else pd.DataFrame(columns=KLINE_COLUMNS)

        new = klines_to_frame(rows)
        df = new if base is None else pd.concat([base, new], ignore_index=True)
        df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
        self.save(symbol, interval, df)
        return df
