# Store Demand Forecast

Forecast normalized demand for the next replenishment cycle.

## Overview

This is a fully synthetic tabular prediction problem exported from
`tabular-bank` in a Kaggle-style layout. The competition framing is
human-readable; the underlying rows and features are procedurally
generated.

## Files

- `train.csv`: 320 labeled rows
- `test.csv`: 160 unlabeled rows for submission
- `sample_submission.csv`: submission template
- `solution/test_labels.csv`: held-out labels for local scoring
- `data_dictionary.csv`: column roles and dtypes
- `starter.py`: minimal sklearn baseline that writes `submission.csv`
- `metadata.json`: generation metadata and benchmark context

## Evaluation

Metric: **RMSE**

Submissions are ranked by root mean squared error.
Submit one numeric forecast per row.

## Dataset Snapshot

- Problem type: `regression`
- Target column: `future_demand`
- Total observed features: `19`
- Informative features: `16`
- Noise features: `3`
- Missingness mechanism: `MCAR`
- Domain tag: `retail`

## Reproducibility

Generated from:

- round id: `examples-kaggle-v1`
- scenario id: `store_demand_forecast`
- dataset id: `examples-kaggle-v1_store_demand_forecast`

In a real competition, `solution/test_labels.csv` would remain hidden.
It is included here only so the example package is self-contained.
