from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import Config
from .features import FEATURES


def _folds(n: int, cfg: Config) -> list[tuple[int, int]]:
    """Time-ordered (train_end, test_end) pairs with an expanding training window.

    Never a random split: a random split would let the model train on bars that come
    after the ones it is scored on, which is the easiest way to fake a result.
    """
    start = int(n * cfg.model.min_train_frac)
    size = max(1, (n - start) // cfg.model.n_folds)
    out = []
    for k in range(cfg.model.n_folds):
        test_end = min(start + (k + 1) * size, n)
        if start + k * size >= test_end:
            break
        out.append((start + k * size, test_end))
    return out


def _econ(label, proba, r_long, r_short, thresholds) -> dict:
    """Per-order expectancy.

    The prediction is used to place a single stop order at one box edge. If the box
    breaks that way, the order fills and we take that side's R; if it breaks the other
    way, the order never fills — no position, no cost. So a wrong call costs nothing
    and a right call pays; accuracy decides which trades you capture.
    """
    up = label == 1
    r_actual = np.where(up, r_long, r_short)

    def block(mask, pnl_mask):
        n = int(mask.sum())
        if n == 0:
            return dict(n=0, exp=float("nan"), hit=float("nan"))
        pnl = np.where(pnl_mask, r_actual, 0.0)[mask]
        return dict(n=n, exp=float(np.nanmean(pnl)), hit=float(np.nanmean(pnl > 0)))

    out = {
        "trade-all-long": block(np.ones(len(label), dtype=bool), up),
        "trade-all-short": block(np.ones(len(label), dtype=bool), ~up),
    }
    for t in thresholds:
        conf = np.abs(proba - 0.5) >= (t - 0.5)
        pred_up = proba >= 0.5
        out[f"conf>={t:.2f}"] = block(conf, pred_up == up)
    return out


def _calibration(y, proba, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append(dict(bin=f"{edges[b]:.1f}-{edges[b+1]:.1f}", n=int(m.sum()),
                         predicted=float(proba[m].mean()), actual=float(y[m].mean())))
    return rows


def walk_forward(df: pd.DataFrame, cfg: Config, models: list) -> dict:
    """Run each model through the same folds and pool the out-of-fold predictions."""
    data = df.dropna(subset=["label"]).sort_values("timestamp")
    # every resolved box has exactly one tradeable side
    data = data.dropna(subset=["r_long", "r_short"], how="all")
    X = data[FEATURES]
    y = data["label"].to_numpy(dtype=int)
    r_long = data["r_long"].to_numpy(dtype=float)
    r_short = data["r_short"].to_numpy(dtype=float)
    n = len(data)
    folds = _folds(n, cfg)

    results: dict[str, dict] = {}
    for model in models:
        accs, aucs, per_fold_econ = [], [], []
        pooled_p = np.full(n, np.nan)
        pooled_mask = np.zeros(n, dtype=bool)

        for tr, te in folds:
            model.fit(X.iloc[:tr], y[:tr])           # expanding window: past only
            proba = model.predict_proba(X.iloc[tr:te])
            yt = y[tr:te]
            accs.append(float(np.mean((proba >= 0.5) == yt)))
            aucs.append(float(roc_auc_score(yt, proba)) if len(set(yt)) > 1 else float("nan"))
            per_fold_econ.append(
                _econ(yt, proba, r_long[tr:te], r_short[tr:te], cfg.evaluation.thresholds)
            )
            pooled_p[tr:te] = proba
            pooled_mask[tr:te] = True

        econ_summary = {}
        for key in (per_fold_econ[0] if per_fold_econ else {}):
            exps = [e[key]["exp"] for e in per_fold_econ if np.isfinite(e[key]["exp"])]
            ns = [e[key]["n"] for e in per_fold_econ]
            econ_summary[key] = dict(
                n_mean=float(np.mean(ns)) if ns else 0.0,
                exp_mean=float(np.mean(exps)) if exps else float("nan"),
                exp_std=float(np.std(exps)) if len(exps) > 1 else 0.0,
                folds=len(exps),
            )

        results[model.name] = dict(
            model=model.name, folds=len(folds),
            accuracy_mean=float(np.mean(accs)) if accs else float("nan"),
            accuracy_std=float(np.std(accs)) if len(accs) > 1 else 0.0,
            auc_mean=float(np.nanmean(aucs)) if aucs else float("nan"),
            auc_std=float(np.nanstd(aucs)) if len(aucs) > 1 else 0.0,
            per_fold_accuracy=accs, per_fold_auc=aucs,
            calibration=_calibration(y[pooled_mask], pooled_p[pooled_mask]),
            econ=econ_summary, per_fold_econ=per_fold_econ,
            direction=model.directions(),
        )

    return dict(
        n_boxes=n,
        n_folds=len(folds),
        base_rate_up=float(y[pooled_mask].mean()) if pooled_mask.any() else float("nan"),
        models=results,
    )
