import numpy as np
import pandas as pd

from src.config import Config
from src.evaluate import _folds
from src.features import FEATURES, box_features
from src.models import build_models
from src.patterns.box import DOWN, UP, detect_boxes

BOX_HI, BOX_LO = 100.5, 99.5


def small_cfg() -> Config:
    """Short windows so the small synthetic frames satisfy the detector's minimum
    length guard (lookback + breakout_window + 2)."""
    cfg = Config()
    cfg.box.breakout_window = 10
    cfg.features.trend_window = 20
    cfg.features.prior_return_window = 10
    cfg.features.atr_short = 5
    cfg.features.atr_long = 20
    return cfg


def make_df(breakout: str | None = "up", flat: int = 30, tail: int = 60) -> pd.DataFrame:
    """`flat` bars of a tight box, then either a breakout bar or more of the same."""
    rows = []
    for i in range(flat):
        rows.append(dict(open=100.0, high=BOX_HI, low=BOX_LO,
                         close=100.2 if i % 2 == 0 else 99.9, volume=100.0))
    if breakout == "up":
        rows.append(dict(open=100.4, high=101.2, low=100.3, close=101.0, volume=500.0))
    elif breakout == "down":
        rows.append(dict(open=99.6, high=99.7, low=98.7, close=99.0, volume=500.0))
    for _ in range(tail):
        rows.append(dict(open=100.0, high=100.4, low=99.6, close=100.0, volume=100.0))
    df = pd.DataFrame(rows)
    df["open_time"] = range(len(df))
    return df


def test_detects_a_box_and_labels_the_breakout_side():
    boxes = detect_boxes("T", "15m", make_df("up"), Config())
    assert boxes, "expected at least one box"
    first = boxes[0]
    assert first.box_high == BOX_HI and first.box_low == BOX_LO
    assert first.side == UP
    assert first.entry == 101.0
    assert first.breakout_idx > first.end_idx


def test_labels_a_downward_break():
    boxes = detect_boxes("T", "15m", make_df("down"), Config())
    assert boxes[0].side == DOWN
    assert boxes[0].entry == 99.0


def test_no_breakout_leaves_the_box_unresolved():
    boxes = detect_boxes("T", "15m", make_df(None), Config())
    assert boxes, "the coil itself should still be reported"
    assert all(b.side is None for b in boxes), "never invent a label"


def test_boxes_do_not_overlap():
    df = make_df("up", flat=30, tail=200)
    boxes = detect_boxes("T", "15m", df, Config())
    # walk the resolved boxes and confirm each starts after the previous resolution
    resolved = [b for b in boxes if b.breakout_idx is not None]
    for earlier, later in zip(resolved, resolved[1:]):
        assert later.end_idx > earlier.breakout_idx


def test_features_do_not_look_ahead():
    cfg = small_cfg()
    df = make_df("up", flat=30, tail=400)
    full = box_features("T", "15m", df, cfg)
    assert not full.empty

    cut = 150  # truncate mid-series; features for earlier boxes must not change
    truncated = box_features("T", "15m", df.iloc[:cut].reset_index(drop=True), cfg)
    assert not truncated.empty

    common = set(full["timestamp"]) & set(truncated["timestamp"])
    assert common, "expected the same box to appear in both frames"
    for ts in common:
        before = full[full["timestamp"] == ts][FEATURES].iloc[0]
        after = truncated[truncated["timestamp"] == ts][FEATURES].iloc[0]
        for col in FEATURES:
            a, b = before[col], after[col]
            if pd.isna(a) or pd.isna(b):
                assert pd.isna(a) and pd.isna(b), f"{col} NaN mismatch"
            else:
                assert abs(float(a) - float(b)) < 1e-9, f"{col} changed when the future was cut"


def test_folds_are_time_ordered():
    cfg = Config()
    folds = _folds(1000, cfg)
    assert folds, "expected folds"
    for train_end, test_end in folds:
        assert train_end < test_end <= 1000
    # expanding window: each fold trains on strictly more than the last
    ends = [tr for tr, _ in folds]
    assert ends == sorted(ends)
    # no fold's test set can include bars from before its training cut
    for i, (_, te) in enumerate(folds):
        if i + 1 < len(folds):
            assert te <= folds[i + 1][0], "test windows must not overlap the next train slice"


def test_economic_simulation_signs():
    """A long that runs to the target pays; an up-breakout has no tradeable short."""
    df = make_df("up", flat=30, tail=5)
    df.loc[31, ["open", "high", "low", "close"]] = [101.0, 103.0, 100.9, 102.8]
    out = box_features("T", "15m", df, small_cfg())
    assert not out.empty
    row = out.iloc[0]
    assert row["label"] == 1
    assert row["r_long"] > 0, "long into strength should pay here"
    # shorting at an up-breakout price would put the stop below the entry — not tradeable
    assert pd.isna(row["r_short"])


def test_scorecard_reads_direction_from_a_predictive_feature():
    cfg = Config()
    rng = np.random.default_rng(0)
    n = 600
    driver = rng.normal(size=n)
    y = (driver + rng.normal(scale=0.5, size=n) > 0).astype(int)
    X = pd.DataFrame({c: rng.normal(size=n) for c in FEATURES})
    X["sma_dist"] = driver          # only this one carries signal

    model = build_models(cfg)[0]    # scorecard
    model.fit(X, y)
    directions = {d["feature"]: d for d in model.directions()}
    assert directions["sma_dist"]["direction"] == "up"
    assert directions["sma_dist"]["strength"] == max(
        d["strength"] for d in model.directions()
    ), "the predictive feature should be the strongest"
