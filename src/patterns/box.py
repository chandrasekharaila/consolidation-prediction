from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config

UP = "up"
DOWN = "down"


@dataclass
class Box:
    """A consolidation, resolved (side set) or still forming (side None)."""

    symbol: str
    interval: str
    end_idx: int          # last bar of the box — the decision point
    timestamp: int
    box_high: float
    box_low: float
    range_pct: float
    box_age: int          # consecutive bars the range has stayed this tight
    close: float
    side: str | None = None   # "up" / "down" / None if unresolved
    breakout_idx: int | None = None
    entry: float | None = None


def tight_and_age(df: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rolling box extremes, tightness flag, and how long the box has held.

    Everything at bar t uses bars up to and including t, so nothing here can see
    the future — the features built from it are safe to compute at decision time.
    """
    p = cfg.box
    roll_high = df["high"].rolling(p.lookback).max().to_numpy()
    roll_low = df["low"].rolling(p.lookback).min().to_numpy()
    range_pct = (roll_high - roll_low) / roll_low * 100.0
    tight = np.isfinite(range_pct) & (range_pct <= p.max_range_pct)

    age = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        age[i] = age[i - 1] + 1 if (i > 0 and tight[i]) else (1 if tight[i] else 0)
    return roll_high, roll_low, range_pct, age


def detect_boxes(symbol: str, interval: str, df: pd.DataFrame, cfg: Config) -> list[Box]:
    """Walk the series and emit one box per consolidation, non-overlapping.

    A box is anchored at the first bar the tightness condition holds; then we look
    forward `breakout_window` bars for the first close beyond either edge. No
    breakout inside the window leaves the box unresolved (side=None) — we never
    invent a label for it.
    """
    p = cfg.box
    if len(df) < p.lookback + p.breakout_window + 2:
        return []

    roll_high, roll_low, range_pct, age = tight_and_age(df, cfg)
    closes = df["close"].to_numpy()
    times = df["open_time"].to_numpy()
    n = len(df)

    boxes: list[Box] = []
    t = p.lookback
    while t < n - 1:
        if not np.isfinite(range_pct[t]) or range_pct[t] > p.max_range_pct:
            t += 1
            continue
        if age[t] < p.min_box_age:
            t += 1
            continue

        box_high, box_low = float(roll_high[t]), float(roll_low[t])
        side = breakout_idx = entry = None
        for j in range(t + 1, min(t + 1 + p.breakout_window, n)):
            if closes[j] > box_high:
                side, breakout_idx, entry = UP, j, float(closes[j])
                break
            if closes[j] < box_low:
                side, breakout_idx, entry = DOWN, j, float(closes[j])
                break

        boxes.append(
            Box(
                symbol=symbol,
                interval=interval,
                end_idx=t,
                timestamp=int(times[t]),
                box_high=box_high,
                box_low=box_low,
                range_pct=float(range_pct[t]),
                box_age=int(age[t]),
                close=float(closes[t]),
                side=side,
                breakout_idx=breakout_idx,
                entry=entry,
            )
        )
        # move past the resolution so boxes never overlap
        t = (breakout_idx + 1) if breakout_idx is not None else (t + 1 + p.breakout_window)
    return boxes
