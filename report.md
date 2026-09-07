# AirPassengers Capstone Project Report

## Project Overview
In this project, I worked on predicting monthly passenger numbers using the AirPassengers dataset (144 months from 1949 to 1960). The main goal was to test different time series models against a basic benchmark and find the best model using rolling-origin cross-validation.

## Data & Exploration
* *Dataset*: 144 monthly observations with no missing values.
* *Patterns*: The series has a very strong upward trend and clear seasonality that repeats every 12 months.
* *STL Split*: STL decomposition showed a trend strength of 1.00 and seasonal strength of 0.98.

## Baseline Model
I started with a *Seasonal Naive* model ($y_t = y_{t-12}$) as the minimum performance baseline.

To check if the baseline missed anything, I ran a Ljung-Box test on its residuals:
* *Lag 12*: p-value = $2.32 \times 10^{-43}$
* *Lag 24*: p-value = $1.71 \times 10^{-44}$

Since the p-values are basically zero, we reject white noise. This shows that Seasonal Naive leaves a lot of trend and pattern behind, so we need a proper model.

## Model Evaluation
I compared several models using AutoGluon on held-out test data (evaluating with negative MASE):

| Model | Test Score (Neg MASE) | Validation Score |
| :--- | :--- | :--- |
| *AutoARIMA* | *-0.6081* | -1.5760 |
| *Theta* | *-0.8754* | -1.6339 |
| *WeightedEnsemble* | *-0.9295* | -1.5358 |
| *AutoETS* | *-1.1695* | -1.5543 |
| *SeasonalNaive* | *-1.5709* | -1.6565 |
| *Naive* | *-2.4959* | -3.1964 |

## Final Recommendation & Next Steps
* *Winner: **AutoARIMA* performed best, cutting the baseline error by over 60%.
* *Why it worked*: Differencing in ARIMA handled the non-stationary trend much better than simple seasonal lag models.
* *Conclusion*: AutoARIMA is selected for generating the 24-month forecast for 1961–1962.
