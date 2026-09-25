# consolidation-predictor

Predicts **which way a consolidation will break** — before it breaks.

Detects the box, scores it with three different models, and reports both the probability and
**which features are pushing which way**. Then checks whether acting on the prediction actually
makes money, out of sample.

## Method

**1. Detect the consolidation.** The last 24 bars must span a range of ≤ 8%, and the range must
have held tight for at least 6 bars. Boxes are non-overlapping: once one resolves, the scan jumps
past it.

**2. Label it.** The first close beyond either edge within 50 bars → `up` or `down`. A box that
never breaks is **left unresolved and dropped** — no label is ever invented for it.

**3. Features, all known at the box.** Nothing in the matrix can see the future:

| feature | what it captures |
| --- | --- |
| `sma_dist` | distance above/below the 200-bar SMA — trend location |
| `pos_range` | position within the 200-bar high/low range |
| `prior_ret` | return over the prior 50 bars — trend momentum |
| `box_upvol` | share of volume on up-bars **inside the box** — accumulation |
| `box_age` | how long the range has held tight |
| `box_range_pct` | the box's own height |
| `atr_ratio` | short ATR ÷ long ATR — volatility compression |

`box_range_pct` is kept but expected to be useless: an exploratory pass over ~222k consolidations
found tightness had no signal (52.4% → 54.3% across quintiles), while the others were monotonic.

**4. Three models, same folds.** A rule-based **scorecard**, **logistic regression**, and
**gradient boosting** — each exposing `fit` / `predict_proba` / `directions`, so the comparison is
like-for-like and every model produces a signed per-feature table.

**5. Walk-forward validation.** Folds are **time-ordered with an expanding training window**: each
fold trains only on bars *before* the ones it is scored on. Never a random split — that would let
the model train on the future, which is the easiest way to fake a good result. Reported metrics are
**out-of-fold** only, with the **per-fold spread** alongside every mean.

**6. Economics, not just accuracy.** A prediction places a single stop order at one box edge. If
the box breaks that way, the order fills (stop at the opposite edge, target = box height). If it
breaks the other way, **the order never fills — no position, no cost**. So a wrong call is free and
a right call pays; accuracy decides which trades you capture. `trade-all-long` and `trade-all-short`
are the controls the models have to beat.

## Results

Walk-forward, **out-of-fold**, top 489 USDT pairs — 15m over 120 days, 1h over 365 days.

**Direction is highly predictable.** 234,613 resolved consolidations on 15m (base rate 52.4% up):

| model | accuracy | ± fold sd | AUC | ± fold sd |
| --- | --- | --- | --- | --- |
| scorecard | 74.0% | 2.2 | 0.804 | 0.018 |
| linear | 74.8% | 2.0 | 0.805 | 0.018 |
| **boosted** | **76.3%** | 1.7 | **0.821** | 0.017 |

On 1h (123,136 consolidations, base rate 46.2%): boosted **71.2%** / AUC **0.769**. Fold spreads are
tight everywhere (±1–2pp accuracy), so this is stable across folds, not one lucky slice.

**And it makes no money.** Every configuration loses (15m, boosted):

| set | orders | expectancy |
| --- | --- | --- |
| trade-all-long | 32,846 | −0.054R |
| trade-all-short | 32,846 | −0.134R |
| conf ≥ 0.55 | 30,874 | −0.118R |
| conf ≥ 0.70 | 22,800 | −0.100R |

Raising the confidence threshold does not help — flat on 15m, worse on 1h (−0.105R → −0.223R for the
scorecard). The 1h numbers are the same story.

**Why: predicting the direction is a different question from whether the trade pays.**

| | 15m | 1h |
| --- | --- | --- |
| median R:R at entry | 0.91 | 0.91 |
| realised win rate of the long trade | ~45% | ~52% |
| mean R per long trade | −0.142R | −0.077R |
| mean R per short trade | −0.231R | −0.110R |

The target is one box height while the stop sits at the *opposite* box edge (plus the breakout
overshoot), so R:R is under 1 — and the trade reaches that target only ~45% of the time, not the 76%
the model achieves on direction. The model is right about the side and the trade still loses. Filtering
by confidence cannot fix that: it only selects *which* negative-payoff trades you take.

### Which features are in favour — and which are against

15m scorecard, the univariate view (full per-model tables in `reports/feature_direction.md`):

| feature | direction | P(up), bottom → top quintile |
| --- | --- | --- |
| `pos_range` | ▲ up | 16.4% → 83.0% |
| `prior_ret` | ▲ up | 20.0% → 79.8% |
| `sma_dist` | ▲ up | 21.7% → 77.8% |
| `box_upvol` | ▲ up | 24.4% → 77.6% |
| `atr_ratio` | ▲ up | 40.5% → 57.5% |
| `box_age` | ▲ up | 42.4% → 49.5% (weak) |
| `box_range_pct` | ▼ down | 50.3% → 49.9% (**no signal**, as predicted) |

Two things worth taking from this:

- **The four strong features are mostly one factor.** `sma_dist`, `pos_range` and `prior_ret` all
  measure "is it in an uptrend". The direct evidence is a sign conflict: `sma_dist` is strongly
  *positive* on its own (21.7% → 77.8%) but the linear model gives it a **negative** coefficient
  (−0.239), because `pos_range` absorbs the effect. Correlated inputs, not four independent
  predictors. `box_upvol` — volume inside the box — is the one genuinely separate signal, and
  boosting ranks it second (20.1pp).
- **The probabilities are under-confident.** The scorecard's 0.5–0.6 bucket actually breaks up
  **73.6%** of the time. The ordering is right (hence AUC 0.82) but the numbers are compressed
  toward 0.5 — read them as ranks, not probabilities.

### What this implies

The classifier is worth keeping; this trade is not. With a 76%-accurate direction call, the fix is on
the **payoff**, not the model: exit at a fraction of the box height, or stop nearer the box edge, and
sweep the geometry with the direction held fixed.

## Usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.cli fetch --all          # cache 15m (120d) + 1h (365d) candles
python -m src.cli features --all       # write the feature matrix to reports/
python -m src.cli evaluate --all       # walk-forward train + evaluate each model
python -m src.cli report               # print the last result
```

Outputs land in `reports/`: `summary.md`, `feature_direction.md`, `summary.json`, and
`features_{interval}.csv` (the raw matrix, so any number can be traced back to the rows behind it).
Candles cache to `data/cache/` (both gitignored).

## Caveats worth knowing before trusting anything here

- **Most features are proxies for one factor.** `sma_dist`, `pos_range` and `prior_ret` all measure
  "is this instrument in an uptrend". They are correlated, not independent evidence. The one
  genuinely separate signal is `box_upvol` — accumulation visible in the box's own volume.
- **Regime dominates.** In the sibling project the same rules swung from −0.135R to +0.162R between
  the first 70% and last 30% of one dataset. That is why the fold spread is reported next to every
  mean, and why nothing here should be trusted from a single aggregate number.
- **Probability is not profit.** The economic table is the verdict. A model can predict well and
  still lose money if the payoff geometry is bad — that has happened repeatedly in the sibling
  projects, and this README should say so if it happens again.
- **No feature uses future bars**, and a test enforces it: features computed on a truncated frame
  must be identical to those computed on the full one.

## Design

```
src/patterns/box.py   consolidation detection + breakout labelling (no lookahead)
src/features.py       feature matrix + the fixed trade geometry (outcome per side)
src/models/           scorecard / linear / boosted, each with a signed direction table
src/evaluate.py       walk-forward folds, calibration, per-order economics
src/cli.py            fetch | features | evaluate | report
```

The data layer (`binance_client`, `cache`, `universe`) and the candle/volume helpers are shared
with the sibling projects.
