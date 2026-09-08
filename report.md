# Air Passengers Forecasting Report

## Recommendation

**Recommend shipping AutoARIMA.** In an eight-window rolling-origin evaluation, AutoARIMA reduced the mean absolute scaled error (MASE) from **1.31** for the seasonal-naive benchmark to **0.65**. This is a material improvement: on average, AutoARIMA made about half the scaled point-forecast error of the benchmark while being evaluated over the same forecast origins and 12-month horizon.

The data cover monthly international air passenger volumes from 1949 through 1960. Passenger volumes rise substantially over the period and follow a strong annual pattern. The seasonal fluctuations also grow as the level of the series grows, so a forecast needs to account for more than simply repeating last year's value.

| Model | MASE | RMSSE | Scaled CRPS | 80% coverage |
|---|---:|---:|---:|---:|
| SeasonalNaive (benchmark) | 1.313 | 1.246 | 0.0616 | 51.0% |
| AutoARIMA | **0.647** | **0.683** | **0.0339** | **69.8%** |

AutoARIMA wins on both point-forecast measures (MASE and RMSSE) and on scaled CRPS, which evaluates the quality of the full predictive distribution rather than only the central forecast. Its MASE varied from 0.33 to 1.59 across the eight folds, compared with 0.41 to 1.96 for SeasonalNaive. Thus, although forecasting difficulty changes across years, AutoARIMA is the stronger overall choice.

## Prediction intervals

The 80% prediction intervals are not fully calibrated for either model. If an 80% interval is honest, roughly 80% of future observations should fall inside it. SeasonalNaive covered only **51.0%** of the cross-validation observations, while AutoARIMA covered **69.8%**. Both bands are therefore too narrow or otherwise understate uncertainty, but AutoARIMA is materially closer to the intended coverage.

The practical implication is that AutoARIMA should be used for the central forecast, while planners should treat its default 80% interval as optimistic. Capacity, staffing, and budget decisions should include an additional safety margin until interval calibration is improved and checked again on future data.

## Residuals and remaining structure

The seasonal-naive benchmark uses the value from the same month a year earlier. Its residuals were tested with the Ljung-Box test and were decisively not random: the p-values were **2.32e-43** at lag 12 and **1.71e-44** at lag 24. These values are far below 0.05.

This result means that repeating last year's monthly value does not explain all of the structure in the series. It captures the recurring annual pattern, but leaves systematic variation related to the rising level and changing seasonal fluctuations. The STL analysis supports this conclusion: trend strength was **1.00** and seasonal strength was **0.98**. AutoARIMA improves performance because it models temporal dependence and seasonality rather than relying solely on a year-ago repeat.

## Next improvement

The next step should be **dynamic regression with a measured demand driver**, such as monthly temperature for a selected travel region, an economic activity indicator, or a fuel-price series. The driver should be downloaded from a public source, stored as a local CSV, and made available only up to each forecast origin during cross-validation.

This addition may explain year-to-year changes in the size of the seasonal peaks that a univariate model can only infer from past passenger counts. Success should not be judged only by lower MASE: the revised model should also improve scaled CRPS and move 80% interval coverage closer to 80% under the same rolling-origin evaluation.
