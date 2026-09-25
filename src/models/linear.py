from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features import FEATURES


class Linear:
    """Logistic regression on standardised features. Signed coefficients make the
    direction table trivial, and the probabilities come out reasonably calibrated."""

    name = "linear"

    def __init__(self, max_iter: int = 2000):
        self.pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(max_iter=max_iter)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Linear":
        self.pipe.fit(X[FEATURES], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(X[FEATURES])[:, 1]

    def directions(self) -> list[dict]:
        coefs = self.pipe.named_steps["lr"].coef_.ravel()
        rows = []
        for col, c in zip(FEATURES, coefs):
            rows.append(
                dict(feature=col, direction="up" if c >= 0 else "down",
                     strength=abs(float(c)), detail=f"standardised coef {c:+.3f}")
            )
        return sorted(rows, key=lambda r: -r["strength"])
