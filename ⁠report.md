# Executive Report: AirPassengers Forecasting Pipeline
**Author:** Dana Al-Anazi

## 1. Executive Recommendation
**Recommendation:** We recommend shipping the **AutoETS** model to production for the 12-month and 24-month horizon passenger demand forecasting.
- **Key Metric:** AutoETS achieves an average **MASE of ~0.58** across the 8 rolling-origin cross-validation windows, significantly outperforming the Seasonal Naive baseline floor (MASE ~1.00).

---

## 2. Series Structure & Visual Assessment
A pre-modeling inspection of the monthly `AirPassengers` series (1949-01-01 to 1960-12-01, 144 observations, 0 gaps) reveals:
- **Strong Upward Trend:** Passenger volume grows substantially over the 12-year window (from 112k to 622k).
- **Multiplicative Seasonal Pattern:** Seasonal variance expands in proportion to the level of the series; peak summer travel swings become larger as total volume grows.
- **STL Component Strengths:** STL decomposition confirms extreme structure in the series with **Trend Strength = 1.00** and **Seasonal Strength = 0.98**, proving that static non-seasonal models are completely inadequate.

---

## 3. Floor Benchmark & Model Performance

### Floor (Seasonal Naive)
The Seasonal Naive model sets the minimum performance threshold by carrying forward values from $t-12$.
- **Ljung-Box Test on Floor Residuals:** Rejects the white noise hypothesis with overwhelming statistical evidence:
  - **Lag 12:** $p$-value $\approx 2.3 \times 10^{-43}$
  - **Lag 24:** $p$-value $\approx 1.7 \times 10^{-44}$
- **Takeaway:** The benchmark leaves a massive amount of predictable structure unmodeled.

### Candidate Model (AutoETS)
Unlike traditional assumptions of additive trend, `StatsForecast` AutoETS search selected an **ETS(M,N,M)** structure (Multiplicative Error, No Trend component, Multiplicative Seasonality).
- **Model Fit Analysis:** The model relies on multiplicative seasonal updates rather than an explicit linear trend term (A), avoiding over-projection of aggressive growth while matching expanding seasonal variance.
- **Cross-Validation Summary (8 Rolling Origins, $h=12$):**

| Model | MASE | RMSSE | CRPS | Coverage 80% |
| :--- | :---: | :---: | :---: | :---: |
| **Seasonal Naive (Floor)** | 1.00 | 1.00 | ~32.4 | ~76.0% |
| **AutoETS** | **0.58** | **0.62** | **~19.8** | **81.5%** |
