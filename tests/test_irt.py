"""Tests for the IRT analysis in meta-eval."""

import numpy as np
import pandas as pd
import pytest

from tabular_bank.evaluation.meta_eval import compute_irt, IRTResult


def _make_task_scores(n_models=6, n_tasks=5, seed=42):
    """Create a score matrix with enough spread to fit IRT."""
    rng = np.random.default_rng(seed)
    models = [f"model_{i}" for i in range(n_models)]
    tasks = [f"task_{i}" for i in range(n_tasks)]
    # Create scores with genuine model strength differences
    abilities = rng.normal(0, 1, n_models)
    difficulties = rng.normal(0, 0.5, n_tasks)
    data = np.zeros((n_models, n_tasks))
    for i in range(n_models):
        for j in range(n_tasks):
            data[i, j] = abilities[i] - difficulties[j] + rng.normal(0, 0.2)
    return pd.DataFrame(data, index=models, columns=tasks)


class TestComputeIRT:

    def test_returns_irt_result(self):
        scores = _make_task_scores()
        result = compute_irt(scores, min_models=4)
        assert isinstance(result, IRTResult)
        assert len(result.items) == 5
        assert len(result.model_abilities) == 6

    def test_returns_none_when_too_few_models(self):
        scores = _make_task_scores(n_models=2)
        result = compute_irt(scores, min_models=4)
        assert result is None

    def test_difficulty_range_property(self):
        scores = _make_task_scores()
        result = compute_irt(scores)
        lo, hi = result.difficulty_range
        assert lo <= hi
        assert np.isfinite(lo)
        assert np.isfinite(hi)

    def test_discrimination_is_positive(self):
        scores = _make_task_scores()
        result = compute_irt(scores)
        for item in result.items:
            assert item.discrimination > 0

    def test_abilities_are_finite(self):
        scores = _make_task_scores()
        result = compute_irt(scores)
        for theta in result.model_abilities.values():
            assert np.isfinite(theta)

    def test_higher_ability_means_higher_scores(self):
        """Models with higher IRT ability should generally have higher scores."""
        scores = _make_task_scores(n_models=8, n_tasks=10, seed=99)
        result = compute_irt(scores)
        # Check correlation between mean score and ability
        mean_scores = scores.mean(axis=1)
        abilities = pd.Series(result.model_abilities)
        common = mean_scores.index.intersection(abilities.index)
        corr = mean_scores[common].corr(abilities[common])
        # Should be positively correlated (not necessarily perfect)
        assert corr > 0.3
