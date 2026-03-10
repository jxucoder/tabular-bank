# Customer Retention Risk

Predict whether an account will churn in the next billing cycle.

## Overview

This is a fully synthetic tabular prediction problem exported from
`tabular-bank` in a Kaggle-style layout. The competition framing is
human-readable; the underlying rows and features are procedurally
generated.

## Files

- `train.csv`: 280 labeled rows
- `test.csv`: 140 unlabeled rows for submission
- `sample_submission.csv`: submission template
- `solution/test_labels.csv`: held-out labels for local scoring
- `data_dictionary.csv`: column roles and dtypes
- `starter.py`: minimal sklearn baseline that writes `submission.csv`
- `metadata.json`: generation metadata and benchmark context

## Evaluation

Metric: **ROC AUC**

Submissions are ranked by ROC AUC on the positive class.
Submit a probability between 0 and 1 for each row.

## Dataset Snapshot

- Problem type: `binary`
- Target column: `will_churn`
- Total observed features: `16`
- Informative features: `14`
- Noise features: `2`
- Missingness mechanism: `MAR`
- Domain tag: `telecom`

## Reproducibility

Generated from:

- round id: `examples-kaggle-v1`
- scenario id: `customer_retention_risk`
- dataset id: `examples-kaggle-v1_customer_retention_risk`

In a real competition, `solution/test_labels.csv` would remain hidden.
It is included here only so the example package is self-contained.
