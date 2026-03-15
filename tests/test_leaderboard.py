"""Tests for leaderboard generation."""

import pandas as pd

from tabular_bank.leaderboard import generate_leaderboard, get_task_scores


class _ResultStub:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_dataframe(self) -> pd.DataFrame:
        return self._df.copy()


def _result() -> _ResultStub:
    rows = [
        {"model": "m1", "task": "t1", "score": 0.90},
        {"model": "m1", "task": "t2", "score": 0.88},
        {"model": "m2", "task": "t1", "score": 0.70},
        {"model": "m2", "task": "t2", "score": 0.72},
        {"model": "m3", "task": "t1", "score": 0.55},
        {"model": "m3", "task": "t2", "score": 0.50},
    ]
    return _ResultStub(pd.DataFrame(rows))


def test_get_task_scores_shape():
    scores = get_task_scores(_result())
    assert scores.shape == (3, 2)
    assert list(scores.columns) == ["t1", "t2"]


def test_generate_leaderboard_orders_best_model_first():
    board = generate_leaderboard(_result())
    assert list(board["model"])[0] == "m1"
    assert list(board["model"])[-1] == "m3"
    assert board["n_tasks"].iloc[0] == 2
