# International air passengers — what to ship, and what it will cost you to trust it

**Emad Sulaiman Alwan** · Capstone, SDAIA Time Series Analysis & Forecasting · 2026-09-07

Monthly international air passengers, 1949-01 to 1960-12. 144 months, no gaps, no
duplicates. Every number below comes out of one rolling-origin harness — **8
origins, 12 months each, step 12, 96 scored points** — and every model in the table
faced exactly those origins. A number that could not have come out of that harness
is not in this report.

---

## 1 · The recommendation

**Ship `logARIMA` — an automatically selected seasonal ARIMA fitted on the
logarithm of the series — and the one number that earns it is MASE 0.64 against
the seasonal-naive floor's 1.31: it makes 51% less point error than the benchmark
every other model has to beat.**

| model | MASE ↓ | scaled CRPS ↓ | 80% coverage → 80% | median band width | mean bias | worst fold (MASE) |
|---|---|---|---|---|---|---|
| **Seasonal naive** — the floor | 1.313 | 0.0616 | **51.0%** | 80 | **+34.9** | 1.959 |
| AutoETS | 0.952 | 0.0499 | 49.0% | 49 | +20.2 | 1.554 |
| AutoARIMA (raw scale) | 0.647 | 0.0339 | 69.8% | 41 | +5.9 | 1.594 |
| **logARIMA** — recommended | **0.637** | **0.0327** | **72.9%** | 49 | −4.3 | **1.173** |

**The honest margin.** On the point forecast, `logARIMA` beats raw `AutoARIMA` by
**1.6%** — that is inside the noise you should expect from eight folds, and it is
not the reason to prefer it. The reasons are the other three columns: a better
distribution score, a band three points closer to honest, and above all a **worst
fold of 1.17 against 1.59**. The recommended model is the one whose bad year is
least bad, not the one whose average is prettiest by a hair.

**What the floor already told us.** The floor's own residuals (`y_t − y_{t−12}`)
are nowhere near white noise — Ljung-Box gives p < 1e-40 at both 12 and 24 lags —
and the structure left behind is a **trend, not a season**: the residual
autocorrelation at lag 12 has collapsed to −0.04 while lag 1 is still 0.75, and the
residuals average **+32 passengers rather than 0**. The floor repeats last year into
a series that grew fourfold in twelve years, so it is systematically low. That
diagnosis, made before any model was fitted, is exactly what selected the
shortlist: only models carrying a trend term survived.

---

## 2 · The intervals

**The band is the part of this work I would not sign off as finished.**

| | floor | logARIMA |
|---|---|---|
| nominal level | 80% | 80% |
| **actual coverage** | **51.0%** | **72.9%** |
| median width | 80 passengers | 49 passengers |

**The floor's interval is not honest, and it is wrong in two ways at once.** It is
too narrow, and it is *pointed in the wrong direction* — 95% of its errors fall on
the same side of the forecast. Widening it would not fix it; it needs re-centring,
and re-centring needs a trend term. Split its coverage by horizon and the mechanism
is visible: **at 7 months ahead it covers 12.5%** — the peak summer months, which
grow fastest and are precisely the months it copies from last year.

**The recommended model's interval is better and still not honest.** 72.9% where
80% was promised. In plain terms: *if you plan capacity against this band expecting
to be caught out one month in five, you will be caught out closer to one month in
four.* I would quote it to a planner as "about 73% of months landed inside the 80%
band" and add a margin, rather than quote 80% and be quietly wrong seven months out
of a hundred.

No model in the table reached the nominal level. That is a finding about the whole
family of methods on this series, not a defect of one model, and it is the single
most important sentence in this report for anyone about to act on a forecast.

---

## 3 · The residuals — what the model missed

**The floor's failure was a bias; the pick's failure is spread.**

The recommended model's cross-validated errors average **−4.3 passengers, −1.3% of
the level** — small, and, importantly, *negative on purpose*. The model is fitted on
log(y) and inverted with exp(), and that inversion returns the **median** of a
right-skewed distribution rather than its mean. The theory predicts a slight
systematic under-forecast, and the measurement confirms it: the raw-scale ARIMA's
bias is **+5.9**, the log path's is **−4.3**. The sign flipped exactly where it was
supposed to.

What is left is dispersion: a residual standard deviation of **21.8 passengers**,
scattered rather than structured. The error map in the notebook shows this directly
— 96 cells, one per (fold, horizon). The floor's map is a wall that darkens down
the page as the level rises (mean absolute error **10.4%**); the pick's is noise
with no year and no horizon owning it (**5.3%**).

**One caveat I will not paper over.** A Ljung-Box test run on the *pooled*
cross-validated errors rejects for the recommended model too, but that statistic is
not valid here: errors issued from the same origin at horizons h and h+1 are
correlated by construction, so pooled multi-horizon errors reject almost regardless
of the model. The floor's rejection above *is* meaningful because it was computed on
one-step in-sample residuals. Reporting the second number as if it meant the same
thing as the first would be a mistake, and it is not one I am making here.

**What that suggests is missing.** Nothing seasonal and nothing trending — both are
accounted for. What remains is level shifts the calendar cannot explain: the series
carries visible steps that no month-of-year term will ever predict, because their
causes are outside the series.

---

## 4 · One change

**Correct the back-transform bias, and measure the correction through the same
eight origins before adopting it.**

The −4.3 passenger bias is not random noise — it is a known, signed, theoretically
predicted consequence of forecasting in logs and exponentiating back. A standard
correction adds half the forecast error variance before inverting, converting the
median back into a mean.

**What I expect it to do, and what would make me drop it:** it should remove most of
the −1.3% systematic under-forecast and shift the interval upward with it, which
should push coverage from 72.9% toward the nominal 80% *without* widening the band.
If the harness shows coverage improving while MASE is unchanged, adopt it. If MASE
degrades, drop it — a small unbiased-on-average forecast that is worse month to
month is not an improvement, and the harness is what settles that, not the theory.

This is the cheapest remaining win because it changes one line and touches nothing
else. The expensive alternative — an external driver such as regional fuel price or
economic activity — is the right second step, but it introduces a variable that
itself has to be forecast, and on this evidence the model does not yet need it.

---

## Reproducibility, and one defect caught before it shipped

Every figure and number here regenerates by running `project.ipynb` top to bottom;
it completes in **under a minute**, and the exact package versions are printed by
its first cell.

**The defect.** An earlier draft staged the series through a CSV written with
`to_csv()` and read it back with an `Unnamed: 0` index column attached.
`StatsForecast` treats every column that is not `unique_id`/`ds`/`y` as an
**exogenous regressor**, silently — so the row counter 0…143, a perfect
deterministic time trend, was handed to the model as a feature. It helped: MASE fell
from 0.637 to 0.498, a 22% "improvement" produced by a column nobody chose and
nobody would have had at forecast time.

It was caught by asking why a number improved, not by an error message — nothing
raised, nothing warned. The notebook now carries a guard that asserts the modelling
frame holds exactly those three columns, plus a negative control that proves the
guard fires on the exact case that caused it. **All numbers in this report are the
clean ones.** The tempting figure — 0.498 — appears nowhere except in this
paragraph, where it belongs.
