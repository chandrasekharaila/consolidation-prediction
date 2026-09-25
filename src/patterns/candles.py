from __future__ import annotations

import numpy as np
import pandas as pd


def bullish_array(df: pd.DataFrame) -> np.ndarray:
    return (df["close"].to_numpy() > df["open"].to_numpy())


def bearish_array(df: pd.DataFrame) -> np.ndarray:
    return (df["close"].to_numpy() < df["open"].to_numpy())


def consecutive_run(arr: np.ndarray, i: int) -> int:
    """Count consecutive True entries ending at index i (inclusive).

    Direction-agnostic: pass the bullish array to count soldiers, the bearish
    array to count the short-side equivalent.
    """
    count = 0
    j = i
    while j >= 0 and arr[j]:
        count += 1
        j -= 1
    return count


# kept for callers that pass the bullish array explicitly
consecutive_bullish = consecutive_run


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).rolling(window=window, min_periods=window).mean()
