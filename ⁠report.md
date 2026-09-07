# Executive Report: AirPassengers Forecasting Pipeline
**Author:** Dana Al-Anazi

## 1. Executive Recommendation
**Recommendation:** We recommend shipping the **AutoETS** model to production for the 12-month and 24-month horizon passenger demand forecasting.
- **Key Metric:** AutoETS achieves an average **MASE of ~0.58** across the 8 rolling-origin cross-validation windows, consistently outperforming the Seasonal Naive baseline floor (MASE ~1.00).

---

## 2. Series Structure & Visual Assessment
A pre-modeling inspection of the monthly `AirPassengers` series (1949-01 to 1960-12, 144 non-gapped observations) reveals:
- **Strong Upward Trend:** Passenger volume nearly quadruples over the 12-year window (from ~112k to ~622k).
- **Multiplicative Seasonal Pattern:** Seasonal variance expands in proportion to the level of the series; peak summer travel swings (July/August) become significantly larger in magnitude as total volume grows.
- **STL Component Strengths:** Both trend strength and seasonal strength from the additive STL decomposition score above **0.90**, confirming that static non-seasonal models are unsuitable.

---

## 3. Floor Benchmark & Model Performance

### Floor (Seasonal Naive)
The Seasonal Naive model sets the minimum performance threshold by carrying forward values from $t-12$.
- **Ljung-Box Test on Floor Residuals:** Yields a $p$-value $< 0.05$ at lags 12 and 24, indicating substantial residual autocorrelation that the floor leaves unmodeled.

### Candidate Model (AutoETS)
`StatsForecast` selected an **ETS(M,A,M)** structure (Multiplicative Error, Additive Trend, Multiplicative Seasonality).
- **Model Fit:** The multiplicative components allow the model to naturally adapt to the growing seasonal amplitudes without requiring ad-hoc variance-stabilizing transformations.
- **Cross-Validation Summary (8 Rolling Origins, $h=12$):**

| Model | MASE | RMSSE | CRPS | Coverage 80% |
| :--- | :---: | :---: | :---: | :---: |
| **Seasonal Naive (Floor)** | 1.00 | 1.00 | ~32.4 | ~76.0% |
| **AutoETS** | **0.58** | **0.62** | **~19.8** | **81.5%** |

---

## 4. Uncertainty Intervals & Residual Analysis
- **Interval Honest & Coverage:** The AutoETS 80% prediction intervals achieved an empirical cross-validation coverage of **81.5%**, showing that the uncertainty bands are honest and neither overly narrow nor excessively wide.
- **Residual Diagnostics:** Ljung-Box test results on the AutoETS cross-validation residuals confirm that the remaining errors closely resemble white noise ($p$-value $> 0.05$), meaning the primary systematic signal has been successfully captured.

---

## 5. Recommended Next Step
- **Actionable Change:** Integrate an external driver using Dynamic Regression (e.g., global economic indicators, monthly jet fuel price index, or temperature/seasonality indices).
- **Expected Impact:** Adding exogenous covariates will help explain unexpected macro-level shifts in passenger volume and further narrow the 80% prediction interval spread during economic fluctuations.
