from __future__ import annotations

import argparse
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .data.binance_client import BinanceClient
from .data.cache import CandleCache
from .data.universe import build_universe
from .evaluate import walk_forward
from .features import FEATURES, FEATURE_NOTES, build_feature_matrix
from .models import build_models

DAY_MS = 86_400_000
LOG = logging.getLogger("consolidation-predictor")


def now_ms() -> int:
    return int(time.time() * 1000)


def _symbols(cfg: Config, args, client) -> list[str]:
    if getattr(args, "symbols", None):
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    top_n = getattr(args, "top", None) or cfg.universe.top_n
    return build_universe(
        client, top_n=top_n, quote_asset=cfg.universe.quote_asset,
        exclude_stables=cfg.universe.exclude_stables,
        exclude_leveraged=cfg.universe.exclude_leveraged,
    )


def _fetch(cfg, client, cache, symbols, interval, args) -> int:
    days = getattr(args, "days", None) or cfg.history_days.get(interval, 120)
    end = now_ms()
    start = end - days * DAY_MS
    force = getattr(args, "force", False)
    workers = max(1, getattr(args, "workers", 12))

    def load(symbol):
        try:
            return symbol, cache.fetch(client, symbol, interval, start, end, force=force)
        except Exception as exc:
            LOG.warning("%s %s: %s", symbol, interval, exc)
            return symbol, None

    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (symbol, df) in enumerate(pool.map(load, symbols), 1):
            if df is not None:
                ok += 1
            if done % 100 == 0:
                LOG.info("%s: fetched %d/%d", interval, done, len(symbols))
    return ok


def cmd_fetch(cfg: Config, args) -> None:
    client, cache = BinanceClient(), CandleCache(cfg.cache_dir)
    symbols = _symbols(cfg, args, client)
    intervals = cfg.intervals if args.all else [args.interval]
    print(f"Universe: {len(symbols)} symbols | intervals: {', '.join(intervals)}")
    for interval in intervals:
        print(f"[{interval}] cached {_fetch(cfg, client, cache, symbols, interval, args)} symbols")


def _load_matrices(cfg, args) -> dict[str, pd.DataFrame]:
    cache = CandleCache(cfg.cache_dir)
    out = {}
    for interval in (cfg.intervals if args.all else [args.interval]):
        files = sorted((Path(cfg.cache_dir) / interval).glob("*.parquet"))
        if not files:
            print(f"[{interval}] no cached data — run `fetch` first")
            continue
        symbols = [f.stem for f in files]
        matrix = build_feature_matrix(cfg, cache, symbols, interval)
        if matrix.empty:
            print(f"[{interval}] no resolved consolidations")
            continue
        out[interval] = matrix
        print(f"[{interval}] {len(matrix)} resolved consolidations "
              f"({matrix['label'].mean()*100:.1f}% broke up)")
    return out


def cmd_features(cfg: Config, args) -> None:
    matrices = _load_matrices(cfg, args)
    out = Path(cfg.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    for interval, m in matrices.items():
        path = out / f"features_{interval}.csv"
        m.to_csv(path, index=False)
        print(f"wrote {path}")


def cmd_evaluate(cfg: Config, args) -> None:
    matrices = _load_matrices(cfg, args)
    if not matrices:
        print("nothing to evaluate")
        return
    out = Path(cfg.reports_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for interval, matrix in matrices.items():
        print(f"[{interval}] walk-forward over {cfg.model.n_folds} folds…")
        all_results[interval] = walk_forward(matrix, cfg, build_models(cfg))

    (out / "summary.json").write_text(json.dumps(_jsonable(all_results), indent=2))
    (out / "summary.md").write_text(_markdown(all_results))
    (out / "feature_direction.md").write_text(_direction_markdown(all_results))
    print(f"wrote {out/'summary.md'}, {out/'feature_direction.md'}, {out/'summary.json'}")
    _print(all_results)


def cmd_report(cfg: Config, args) -> None:
    path = Path(args.dir or cfg.reports_dir) / "summary.json"
    if not path.exists():
        print(f"No report at {path}. Run `evaluate` first.")
        return
    _print(json.loads(path.read_text()))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isinf(obj) or math.isnan(obj)) else obj
    return obj


def _print(all_results: dict) -> None:
    for interval, res in all_results.items():
        print(f"\n===== {interval} =====")
        print(f"  consolidations {res['n_boxes']} | base rate up {res['base_rate_up']*100:.1f}% "
              f"| folds {res['n_folds']}")
        for name, m in res["models"].items():
            print(f"  {name:<10} acc {m['accuracy_mean']*100:5.1f}% (±{m['accuracy_std']*100:.1f})  "
                  f"auc {m['auc_mean']:.3f} (±{m['auc_std']:.3f})")
            ctrl = m["econ"].get("trade-all-long", {})
            pick = m["econ"].get("conf>=0.55", {})
            print(f"             trade-all-long {ctrl.get('exp_mean', float('nan')):+.3f}R   "
                  f"conf>=0.55 {pick.get('exp_mean', float('nan')):+.3f}R "
                  f"(±{pick.get('exp_std', 0):.3f}, n≈{pick.get('n_mean', 0):.0f})")


def _markdown(all_results: dict) -> str:
    lines = ["# Consolidation predictor — walk-forward results", "",
             "All metrics are **out-of-fold**: each fold trains only on bars before the "
             "bars it is scored on.", ""]
    for interval, res in all_results.items():
        lines += [f"## {interval}", "",
                  f"- consolidations: **{res['n_boxes']}**",
                  f"- base rate breaking up: **{res['base_rate_up']*100:.1f}%**",
                  f"- walk-forward folds: **{res['n_folds']}**", "",
                  "### Classification", "",
                  "| model | accuracy | ± fold sd | AUC | ± fold sd |", "| --- | --- | --- | --- | --- |"]
        for name, m in res["models"].items():
            lines.append(f"| {name} | {m['accuracy_mean']*100:.1f}% | {m['accuracy_std']*100:.1f} | "
                         f"{m['auc_mean']:.3f} | {m['auc_std']:.3f} |")
        lines += ["", "### Economics (per order placed, net of costs)", "",
                  "A prediction places one stop order at a box edge. Wrong side = order never "
                  "fills = no cost. `trade-all-long` / `trade-all-short` are the controls.", "",
                  "| model | set | orders | expectancy | ± fold sd |", "| --- | --- | --- | --- | --- |"]
        for name, m in res["models"].items():
            for key, e in m["econ"].items():
                lines.append(f"| {name} | {key} | {e['n_mean']:.0f} | {e['exp_mean']:+.3f}R | {e['exp_std']:.3f} |")
        first_model = next(iter(res["models"].values()))
        lines += ["", "### Calibration (pooled out-of-fold, " + first_model["model"] + ")", "",
                  "| predicted | n | predicted P(up) | actual P(up) |", "| --- | --- | --- | --- |"]
        for row in first_model["calibration"]:
            lines.append(f"| {row['bin']} | {row['n']} | {row['predicted']:.3f} | {row['actual']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def _direction_markdown(all_results: dict) -> str:
    lines = ["# Which features are in favour — and which are against", "",
             "Signed direction per feature per model. `direction` is what the feature does to "
             "P(breaks up); `strength` is its own scale (percentage points for the scorecard and "
             "boosted, standardised coefficient for the linear model).", ""]
    names = {c: FEATURE_NOTES.get(c, "") for c in FEATURES}
    for interval, res in all_results.items():
        lines += [f"## {interval}", ""]
        for model_name, m in res["models"].items():
            lines += [f"### {model_name}", "",
                      "| feature | direction | strength | what it means |", "| --- | --- | --- | --- |"]
            for row in m["direction"]:
                arrow = "▲ up" if row["direction"] == "up" else "▼ down"
                lines.append(f"| `{row['feature']}` | {arrow} | {row['strength']:.3f} | {row['detail']} |")
            lines.append("")
        lines += ["Feature rationales:", ""]
        for c in FEATURES:
            lines.append(f"- `{c}` — {names[c]}")
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="consolidation-predictor",
                                description="Predict which way a consolidation breaks")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def common(x):
        x.add_argument("--symbols", default=None)
        x.add_argument("--top", type=int, default=None)
        x.add_argument("--interval", default="15m")
        x.add_argument("--days", type=int, default=None)
        x.add_argument("--workers", type=int, default=12)
        x.add_argument("--force", action="store_true")

    fp = sub.add_parser("fetch", help="download and cache candles")
    common(fp); fp.add_argument("--all", action="store_true")
    fep = sub.add_parser("features", help="build the feature matrix")
    common(fep); fep.add_argument("--all", action="store_true", default=True)
    ep = sub.add_parser("evaluate", help="walk-forward train + evaluate")
    common(ep); ep.add_argument("--all", action="store_true", default=True)
    rp = sub.add_parser("report", help="show the latest results")
    rp.add_argument("--dir", default=None)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    {"fetch": cmd_fetch, "features": cmd_features,
     "evaluate": cmd_evaluate, "report": cmd_report}[args.command](cfg, args)


if __name__ == "__main__":
    main()
