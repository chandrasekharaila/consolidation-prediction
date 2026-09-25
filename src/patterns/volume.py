from __future__ import annotations

import pandas as pd

from .candles import sma


def relative_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume relative to the average of the *prior* `window` bars.

    The average is lagged by one bar so the current (signal) candle never
    contributes to its own baseline — no lookahead.
    """
    prior_avg = sma(df["volume"].shift(1), window)
    return df["volume"] / prior_avg
