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

### The payoff can't be fixed by tuning — 12-point sweep

Direction held fixed at the model's call; only the exit varies. Stop is expressed as a fraction of
the box height inside the broken edge; target as a multiple of box height.

| stop ↓ / target → | 0.5× | 1.0× | 1.5× | 2.0× |
| --- | --- | --- | --- | --- |
| 0.25 × box | −0.533 | −0.506 | −0.486 | −0.459 |
| 0.5 × box | −0.309 | −0.288 | −0.278 | −0.259 |
| 1.0 × box (opposite edge) | −0.171 | **−0.162** | −0.156 | −0.146 |

*15m, per-order expectancy net of costs. 1h shows the same shape.*

**Every cell loses**, and tighter stops are strictly worse: cost in R scales as `1/risk`, so halving
the stop doubles the cost drag and noise takes the rest. The current geometry (−0.162R) is within
0.02R of the best cell in the entire grid. There is no knob here that turns this positive.

### Does the probability at least predict *how far* it runs?

Conditioning on the boxes that actually broke **up** on 15m:

| | mean p | expectancy | hit rate |
| --- | --- | --- | --- |
| lowest p decile | 0.19 | −0.119R | 49% |
| highest p decile | 0.91 | **+0.073R** | 57% |

So there *is* a magnitude gradient on 15m (+0.191R top-to-bottom). **But it does not replicate.**
The same measurement on 1h runs the other way (−0.063R), and the short side is flat at every
confidence level (−0.030R spread on 15m, +0.021R on 1h). One pocket, one timeframe, one 120-day
window, with the mirror contradicting it — that is a hypothesis, not an edge.

## Verdict

**The direction model is real. Nothing built on it pays.**

- Predicting which way a consolidation breaks works, out of sample, and stable across folds:
  **76.3% / AUC 0.821** on 15m, **71.2% / AUC 0.769** on 1h.
- Every trading configuration tested loses — 12 exit geometries, 5 confidence thresholds, both
  timeframes, both directions. The only positive pocket fails to replicate out of its timeframe.
- The reason is now precise: the model predicts **direction**, but a trade's outcome depends on
  **magnitude**, and the follow-through after a breakout is close to a coin flip (49% hit at
  sub-1 R:R). Being right about the side while being blind to the distance does not pay.

### Bridge test: could the classifier filter the profitable continuation long?

The sibling project's continuation long earns ~+0.17R on 15m and takes *every* breakout, including
the ones sitting at the bottom of a downtrend. The obvious use for this classifier was as a filter
on those entries: match each long trade to a predictor box, and bucket that strategy's expectancy by
`p_up`.

| | 15m | 1h |
| --- | --- | --- |
| trades matched to a box | 1067 (66%) | 329 (68%) |
| baseline expectancy | **+0.083R** ±0.036 | **+0.176R** ±0.074 |
| keep `p_up ≥ 0.6` | +0.163R (66% kept) | +0.135R (40%) |
| keep `p_up ≥ 0.7` | **+0.205R** (56%) | +0.107R (29%) |
| keep `p_up ≥ 0.8` | +0.255R (32%) | −0.118R (13%) |
| quintiles | −0.019, −0.119, +0.066, +0.179, +0.308 | +0.142, +0.556, −0.065, +0.305, −0.063 |
| top minus bottom | **+0.327R** | **−0.205R** |

**On 15m it looks like a win** — a clean monotone gradient, and keeping `p_up ≥ 0.7` lifts the
strategy from +0.083R to +0.205R on 56% of trades (about +0.12R, roughly 2 standard errors).

**On 1h it reverses and makes things worse**: +0.176R → +0.107R, and at `p_up ≥ 0.8` it goes
negative. The 1h quintiles are non-monotonic noise.

**So the filter is a timeframe artifact, not an edge.** This is the third independent test to show
the same 15m-only pattern with the slower timeframe contradicting it — the magnitude gradient above
(+0.191R / −0.063R) and the sibling project's own timeframe comparison behaved identically. Do not
ship it.

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
