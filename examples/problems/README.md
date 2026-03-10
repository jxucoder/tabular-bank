# Kaggle-Style Example Problems

These folders are generated examples that package synthetic `tabular-bank`
datasets in a competition-style layout.

Regenerate them with:

```bash
python examples/scripts/export_kaggle_style_problems.py
```

| Problem | Type | Metric | Train rows | Test rows |
|---------|------|--------|------------|-----------|
| `customer-retention-risk` | `binary` | `roc_auc` | 280 | 140 |
| `store-demand-forecast` | `regression` | `rmse` | 320 | 160 |
| `ticket-priority-routing` | `multiclass` | `log_loss` | 300 | 150 |

Each problem directory includes train/test CSVs, a sample submission,
held-out labels for local evaluation, a simple starter baseline, and
metadata describing the generated task.
