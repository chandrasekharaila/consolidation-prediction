from __future__ import annotations

from .boosted import Boosted
from .linear import Linear
from .scorecard import Scorecard


def build_models(cfg) -> list:
    """All three, each exposing fit / predict_proba / directions."""
    return [
        Scorecard(cfg.model.n_bins),
        Linear(),
        Boosted(cfg.model.trees, cfg.model.learning_rate, cfg.model.max_depth),
    ]


__all__ = ["Scorecard", "Linear", "Boosted", "build_models"]
