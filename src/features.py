from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data.binance_client import INTERVAL_MS
from .patterns.box import UP, detect_boxes
from .patterns.candles import atr

# The feature columns the models see. Chosen from an exploratory pass over ~222k
# consolidations; box tightness was dropped because it showed no signal.
FEATURES = [
    "sma_dist",       # distance above/below the 200-bar SMA (%)  — trend location
    "pos_range",      # position within the 200-bar high/low range (0-1)
    "prior_ret",      # return over the prior N bars (%)          — trend momentum
    "box_upvol",      # share of volume on up-bars inside the box — accumulation
    "box_age",        # consecutive bars the range has held tight — coil duration
    "box_range_pct",  # the box's own height (%)                  — tightness
    "atr_ratio",      # short ATR / long ATR                     — volatility compression
]

FEATURE_NOTES = {
    "sma_dist": "above the 200-SMA favours an upward break",
    "pos_range": "top of the 200-bar range favours an upward break",
    "prior_ret": "recent strength favours an upward break",
    "box_upvol": "volume on up-bars inside the box = accumulation",
    "box_age": "longer coils may store more energy",
    "box_range_pct": "tightness — no signal in the exploratory pass",
    "atr_ratio": "compression before expansion",
}


def _simulate(
    side: str, entry: float, box_high: float, box_low: float, highs, lows, closes,
    interval: str, cfg: Config,
) -> float:
    """R multiple for taking `side` at the breakout close.

    Stop at the opposite box edge, target = box height, stop checked before target
    within a bar (pessimistic). Net of fees, slippage and (on shorts) funding.
    """
    e = cfg.evaluation
    if side == UP:
        stop = box_low
        target = entry + (box_high - box_low)
        risk = entry - stop
    else:
        stop = box_high
        target = entry - (box_high - box_low)
        risk = stop - entry
    if risk <= 0:
        return np.nan

    n = min(e.max_bars, len(highs))
    if n <= 0:
        return np.nan
    risk_pct = risk / entry * 100.0
    cost = 2.0 * (e.fee_per_side_pct + e.slippage_pct) / risk_pct
    if side != UP:
        hours = n * INTERVAL_MS[interval] / 3_600_000
        cost += (e.funding_pct_per_8h * hours / 8.0) / risk_pct

    for j in range(n):
        if side == UP:
            if lows[j] <= stop:
                return -1.0 - cost
            if highs[j] >= target:
                return (target - entry) / risk - cost
        else:
            if highs[j] >= stop:
                return -1.0 - cost
            if lows[j] <= target:
                return (entry - target) / risk - cost

    last = float(closes[n - 1])
    gross = (last - entry) / risk if side == UP else (entry - last) / risk
    return gross - cost


def box_features(symbol: str, interval: str, df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per resolved consolidation: features known at the box, plus the label
    and the outcome of trading either side (so models only have to pick a side)."""
    f, bc = cfg.features, cfg.box
    boxes = detect_boxes(symbol, interval, df, cfg)
    if not boxes:
        return pd.DataFrame()

    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    opens = df["open"].to_numpy()
    vols = df["volume"].to_numpy()
    sma = pd.Series(closes).rolling(f.trend_window).mean().to_numpy()
    atr_s = atr(df, f.atr_short).to_numpy()
    atr_l = atr(df, f.atr_long).to_numpy()

    rows = []
    for box in boxes:
        t = box.end_idx
        if box.side is None or t < f.trend_window:
            continue  # unresolved, or not enough history for the trend features

        win = slice(t - f.trend_window + 1, t + 1)
        rng_hi, rng_lo = float(highs[win].max()), float(lows[win].min())
        w = slice(t - bc.lookback + 1, t + 1)
        bv, bcl, bop = vols[w], closes[w], opens[w]
        up_vol = float(bv[bcl > bop].sum())

        atr_ratio = (
            float(atr_s[t] / atr_l[t])
            if np.isfinite(atr_s[t]) and np.isfinite(atr_l[t]) and atr_l[t] > 0
            else np.nan
        )
        b = box.breakout_idx
        r_long = _simulate(UP, box.entry, box.box_high, box.box_low,
                           highs[b + 1:], lows[b + 1:], closes[b + 1:], interval, cfg)
        r_short = _simulate("down", box.entry, box.box_high, box.box_low,
                            highs[b + 1:], lows[b + 1:], closes[b + 1:], interval, cfg)

        rows.append(
            dict(
                symbol=symbol, interval=interval, timestamp=box.timestamp,
                sma_dist=(box.close - sma[t]) / sma[t] * 100.0 if sma[t] > 0 else 0.0,
                pos_range=(box.close - rng_lo) / (rng_hi - rng_lo) if rng_hi > rng_lo else 0.5,
                prior_ret=(box.close - closes[t - f.prior_return_window])
                / closes[t - f.prior_return_window] * 100.0,
                box_upvol=up_vol / bv.sum() if bv.sum() > 0 else 0.5,
                box_age=box.box_age,
                box_range_pct=box.range_pct,
                atr_ratio=atr_ratio,
                label=1 if box.side == UP else 0,
                r_long=r_long, r_short=r_short,
                entry=box.entry, box_high=box.box_high, box_low=box.box_low,
                breakout_idx=box.breakout_idx,
            )
        )
    return pd.DataFrame(rows)


def build_feature_matrix(cfg: Config, cache, symbols, interval: str) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        df = cache.load(symbol, interval)
        if df is None or len(df) < cfg.features.trend_window + cfg.box.breakout_window + 5:
            continue
        out = box_features(symbol, interval, df, cfg)
        if not out.empty:
            frames.append(out)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("timestamp").reset_index(drop=True)
