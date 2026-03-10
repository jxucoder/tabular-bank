# Ticket Priority Routing

Assign each incoming support ticket to the correct priority band.

## Overview

This is a fully synthetic tabular prediction problem exported from
`tabular-bank` in a Kaggle-style layout. The competition framing is
human-readable; the underlying rows and features are procedurally
generated.

## Files

- `train.csv`: 300 labeled rows
- `test.csv`: 150 unlabeled rows for submission
- `sample_submission.csv`: submission template
- `solution/test_labels.csv`: held-out labels for local scoring
- `data_dictionary.csv`: column roles and dtypes
- `starter.py`: minimal sklearn baseline that writes `submission.csv`
- `metadata.json`: generation metadata and benchmark context

## Evaluation

Metric: **Log Loss**

Submissions are ranked by multiclass log loss.
Submit a probability for every class on each row.

## Dataset Snapshot

- Problem type: `multiclass`
- Target column: `priority_band`
- Total observed features: `13`
- Informative features: `12`
- Noise features: `1`
- Missingness mechanism: `MNAR`
- Domain tag: `support`

## Reproducibility

Generated from:

- round id: `examples-kaggle-v1`
- scenario id: `ticket_priority_routing`
- dataset id: `examples-kaggle-v1_ticket_priority_routing`

In a real competition, `solution/test_labels.csv` would remain hidden.
It is included here only so the example package is self-contained.
