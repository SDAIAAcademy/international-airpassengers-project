# international-airpassengers-project

https://up2u2b11.github.io/international-airpassengers-project/

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

## The write-up

- **Live:
  [up2u2b11.github.io/international-airpassengers-project](https://up2u2b11.github.io/international-airpassengers-project/)**
  — published by GitHub Pages from the `gh-pages` branch. To republish after a
  change on `main`: `git push -f origin main:gh-pages`.
- **[index.html](index.html)** — a one-page walkthrough of the finished analysis:
  the shape, the floor, the harness verdict over eight rolling origins, where each
  model fails cell by cell, and the interval honesty. Every number on it is
  recomputed by `site/build_page.py` at build time, so the page cannot drift from
  the notebook. Rebuild with `python3 site/build_page.py`.

## Files

- `project.ipynb` — the starter notebook; the graded deliverable
- `brief.md` — the project brief
- `rubric.md` — how it is graded (100 + extra credit)
- `coursekit/` — shared plotting + scoring helpers (the course's harness)
