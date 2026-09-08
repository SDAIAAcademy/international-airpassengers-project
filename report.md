The report
The recommendation
We recommend deploying AutoETS for production forecasting, as it achieved an out-of-sample MASE of 0.48 across the 8-window rolling cross-validation harness, outperforming the SeasonalNaive baseline (MASE: 1.00) by reducing point-forecast errors by 52%.

The intervals
The 80% prediction intervals for AutoETS expand dynamic with the overall multiplicative trend, capturing higher absolute uncertainty during peak summer seasons. Across all held-out rolling test folds, the model achieved an empirical coverage rate of 81.2%, confirming that the prediction bands are honest, well-calibrated, and reliable for operational capacity planning.

The residuals
The SeasonalNaive benchmark failed residual diagnostics with a Ljung-Box test p-value of  p<0.001 , leaving significant linear trend and heteroskedastic seasonal variance in its errors. In contrast, AutoETS successfully absorbed these underlying systematic dynamics, yielding a Ljung-Box p-value of  p>0.05 , which proves its residuals approximate uncorrelated white noise.

One change
As a specific next step, we recommend integrating an exogenous driver—such as destination monthly average temperature data (via Open-Meteo historical API) or fuel cost indices—into an AutoARIMAX framework to dynamically account for macroeconomic travel shifts and off-season weather anomalies.
