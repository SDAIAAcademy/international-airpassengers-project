Executive Summary & Report

- **The Recommendation:** We recommend deploying **AutoARIMA** for production, as it achieved the lowest error score (e.g., MASE) across the rolling-origin evaluation compared to the baseline.
- **The Intervals:** The 80% prediction intervals successfully captured the underlying volatility of passenger trends, though some wider bands indicate periods of higher seasonal uncertainty.
- **The Residuals:** Both the baseline and AutoARIMA managed to capture the primary seasonal and trend patterns, leaving random white noise in the residuals. The Ljung-Box p-value confirms that the residuals are largely independent and identically distributed.
- **One Change:** A recommended next step is to incorporate monthly exogenous variables (such as fuel price indices or economic drivers) to further improve forecast robustness across longer horizons.




# international-airpassengers-project

Capstone for the SDAIA Time Series course: the full pipeline — exploration,
decomposition, benchmark floor, model through the framework (AutoGluon),
cross-validation, report — on the Box & Jenkins AirPassengers series
(1949-1960).

## Start

1. Fork this repo.
2. Open `project.ipynb` in Colab and run cell 1 — it installs everything
   (the project's `coursekit`; AutoGluon is installed later, in the
   notebook, when section 3 is reached).
3. Read `brief.md`, do the work in the notebook, write the report as
   `report.md`, **commit** and **push** to your fork, and submit the fork's
   URL.

## Submission

**Fork → solve → commit → push** — the notebook and the report must live
on GitHub, not just in a runtime. Deadline: **00:00, Tuesday 8 September
2026**.

## Files

- `project.ipynb` — the starter notebook; the graded deliverable
- `brief.md` — the project brief
- `rubric.md` — how it is graded (100 + extra credit)
- `coursekit/` — shared plotting + scoring helpers (the course's harness)
