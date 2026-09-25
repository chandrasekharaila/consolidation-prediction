from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from ..features import FEATURES


class Boosted:
    """Gradient boosting. Handles NaNs natively, so no imputation step.

    Direction is derived by pushing each feature from its 25th to its 75th
    percentile (others held at the median) and reading the change in predicted
    probability — a signed effect that pairs with permutation importance for strength.
    """

    name = "boosted"

    def __init__(self, trees: int = 200, learning_rate: float = 0.05, max_depth: int = 3):
        self.model = HistGradientBoostingClassifier(
            max_iter=trees, learning_rate=learning_rate, max_depth=max_depth,
            early_stopping=False, random_state=0,
        )
        self._X: pd.DataFrame | None = None
        self._y: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Boosted":
        self.model.fit(X[FEATURES], y)
        self._X, self._y = X[FEATURES], np.asarray(y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[FEATURES])[:, 1]

    def directions(self) -> list[dict]:
        if self._X is None:
            return []
        p25 = self._X.quantile(0.25)
        p75 = self._X.quantile(0.75)
        median = self._X.median()
        base = median.to_frame().T
        rows = []
        for col in FEATURES:
            low, high = base.copy(), base.copy()
            low[col], high[col] = p25[col], p75[col]
            delta = float(self.model.predict_proba(high)[0, 1] - self.model.predict_proba(low)[0, 1])
            rows.append(dict(feature=col, direction="up" if delta >= 0 else "down",
                             strength=abs(delta) * 100.0,
                             detail=f"P(up) {delta:+.1%} from p25→p75"))
        return sorted(rows, key=lambda r: -r["strength"])
