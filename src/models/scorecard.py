from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import FEATURES


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Scorecard:
    """Bucket each feature into training-set quantiles, read off the conditional
    P(breaks up) per bucket, and average the centred log-odds.

    Fully inspectable, and the averaging (rather than summing) keeps it from
    becoming wildly overconfident when features are correlated — which most of
    these are, since they all partly measure "is it in an uptrend".
    """

    name = "scorecard"

    def __init__(self, n_bins: int = 5, min_bin: int = 20):
        self.n_bins = n_bins
        self.min_bin = min_bin
        self.edges: dict[str, np.ndarray] = {}
        self.tables: dict[str, dict[int, float]] = {}
        self.medians: dict[str, float] = {}
        self.base_p = 0.5

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Scorecard":
        y = np.asarray(y)
        self.base_p = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
        for col in FEATURES:
            x = X[col].to_numpy(dtype=float)
            finite = np.isfinite(x)
            self.medians[col] = float(np.median(x[finite])) if finite.any() else 0.0
            if finite.sum() < self.n_bins * self.min_bin:
                self.edges[col] = np.array([])
                self.tables[col] = {0: self.base_p}
                continue
            qs = np.unique(np.nanquantile(x[finite], np.linspace(0, 1, self.n_bins + 1)))
            edges = qs[1:-1] if len(qs) > 2 else np.array([])
            self.edges[col] = edges
            bins = np.digitize(x, edges, right=False)
            table: dict[int, float] = {}
            for k in range(len(edges) + 1):
                m = finite & (bins == k)
                table[k] = float(np.clip(y[m].mean(), 1e-3, 1 - 1e-3)) if m.sum() >= self.min_bin else self.base_p
            self.tables[col] = table
        return self

    def _centred_logits(self, X: pd.DataFrame) -> np.ndarray:
        base = _logit(self.base_p)
        out = np.zeros((len(X), len(FEATURES)))
        for i, col in enumerate(FEATURES):
            x = X[col].to_numpy(dtype=float)
            x = np.where(np.isfinite(x), x, self.medians[col])
            bins = np.digitize(x, self.edges[col], right=False)
            p = np.array([self.tables[col].get(int(k), self.base_p) for k in bins])
            out[:, i] = _logit(p) - base
        return out

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        centred = self._centred_logits(X)
        combined = _logit(self.base_p) + centred.mean(axis=1)
        return 1.0 / (1.0 + np.exp(-combined))

    def directions(self) -> list[dict]:
        rows = []
        for col in FEATURES:
            table = self.tables[col]
            if len(table) < 2:
                continue
            ks = sorted(table)
            lo, hi = table[ks[0]], table[ks[-1]]
            rows.append(
                dict(feature=col, direction="up" if hi >= lo else "down",
                     strength=abs(hi - lo) * 100.0, detail=f"P(up) {lo:.1%} → {hi:.1%}")
            )
        return rows
