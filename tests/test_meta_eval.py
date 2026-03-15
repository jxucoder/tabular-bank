"""Tests for meta-evaluation metrics."""

import numpy as np
import pandas as pd

from tabular_bank.evaluation.meta_eval import (
    compute_discriminability,
    compute_ranking_concordance,
    compute_task_diversity,
    run_meta_eval,
)


class _ResultStub:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_dataframe(self) -> pd.DataFrame:
        return self._df.copy()


def _task_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_a": [0.9, 0.7, 0.6, 0.4],
            "task_b": [0.88, 0.72, 0.58, 0.42],
            "task_c": [0.1, 0.4, 0.6, 0.9],
        },
        index=["m1", "m2", "m3", "m4"],
    )


def test_compute_discriminability_flags_flat_task():
    scores = pd.DataFrame({"flat": [0.5, 0.5, 0.5], "spread": [0.2, 0.5, 0.9]}, index=["a", "b", "c"])
    out = compute_discriminability(scores, low_threshold=0.2)
    assert out.per_task["flat"] == 0.0
    assert "flat" in out.flagged_tasks
    assert out.per_task["spread"] > 0.0


def test_compute_ranking_concordance_prefers_similar_ordering():
    scores = _task_scores()
    ref = {"m1": 1, "m2": 2, "m3": 3, "m4": 4}
    out = compute_ranking_concordance(scores, ref)
    assert out.n_common_models == 4
    assert out.kendall_tau > 0.5
    assert out.spearman_rho > 0.7


def test_compute_task_diversity_detects_redundant_pair():
    scores = _task_scores()
    out = compute_task_diversity(scores, redundancy_threshold=0.9)
    assert not np.isnan(out.correlation_matrix.loc["task_a", "task_b"])
    assert any({pair[0], pair[1]} == {"task_a", "task_b"} for pair in out.redundant_pairs)


def test_run_meta_eval_uses_leaderboard_task_scores():
    rows = []
    for model, scores in {"m1": [0.9, 0.8], "m2": [0.7, 0.6], "m3": [0.4, 0.5]}.items():
        for task, score in zip(["t1", "t2"], scores):
            rows.append({"model": model, "task": task, "score": score})
    stub = _ResultStub(pd.DataFrame(rows))
    report = run_meta_eval(stub)
    assert report.discriminability.overall > 0
    assert report.diversity.mean_correlation >= 0
