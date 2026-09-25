from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class UniverseCfg:
    quote_asset: str = "USDT"
    top_n: int = 500
    exclude_stables: bool = True
    exclude_leveraged: bool = True


@dataclass
class BoxCfg:
    lookback: int = 24          # bars forming the consolidation box
    max_range_pct: float = 8.0  # box must be at least this tight
    breakout_window: int = 50   # bars allowed for the breakout to occur
    min_box_age: int = 6        # box must have held at least this long


@dataclass
class FeatureCfg:
    trend_window: int = 200     # SMA / range window
    prior_return_window: int = 50
    atr_short: int = 14
    atr_long: int = 100


@dataclass
class ModelCfg:
    n_folds: int = 5            # walk-forward folds
    min_train_frac: float = 0.30  # size of the first fold's training slice
    n_bins: int = 5             # scorecard quintiles
    trees: int = 200
    learning_rate: float = 0.05
    max_depth: int = 3


@dataclass
class EvalCfg:
    thresholds: list[float] = field(
        default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7]
    )
    target_mode: str = "box_height"  # or "r_multiple"
    target_r: float = 2.0
    max_bars: int = 100
    fee_per_side_pct: float = 0.1
    slippage_pct: float = 0.05
    funding_pct_per_8h: float = 0.01  # charged on the short side


@dataclass
class Config:
    universe: UniverseCfg = field(default_factory=UniverseCfg)
    box: BoxCfg = field(default_factory=BoxCfg)
    features: FeatureCfg = field(default_factory=FeatureCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    evaluation: EvalCfg = field(default_factory=EvalCfg)
    intervals: list[str] = field(default_factory=lambda: ["15m", "1h"])
    history_days: dict[str, int] = field(
        default_factory=lambda: {"15m": 120, "1h": 365}
    )
    cache_dir: str = "data/cache"
    reports_dir: str = "reports"


def load_config(path: str | Path = "config.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        return Config()
    raw = yaml.safe_load(p.read_text()) or {}
    cfg = Config()

    for section, target in (
        ("universe", cfg.universe),
        ("box", cfg.box),
        ("features", cfg.features),
        ("model", cfg.model),
        ("evaluation", cfg.evaluation),
    ):
        for key, value in (raw.get(section) or {}).items():
            setattr(target, key, value)

    if "intervals" in raw:
        cfg.intervals = [str(i) for i in raw["intervals"]]
    if "history_days" in raw:
        cfg.history_days = {str(k): int(v) for k, v in raw["history_days"].items()}

    bt = raw.get("output") or {}
    cfg.cache_dir = bt.get("cache_dir", cfg.cache_dir)
    cfg.reports_dir = bt.get("reports_dir", cfg.reports_dir)
    return cfg
