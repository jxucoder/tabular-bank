"""Tests for the scaling analysis module."""

import numpy as np
import pandas as pd
import pytest

from tabular_bank.evaluation.scaling import (
    analyze_ranking_variance,
    analyze_scenario_scaling,
    ScalingReport,
)


def _make_task_scores(n_models=5, n_tasks=10, seed=42):
    """Create a random model-by-task score matrix."""
    rng = np.random.default_rng(seed)
    models = [f"model_{i}" for i in range(n_models)]
    tasks = [f"task_{i}" for i in range(n_tasks)]
    data = rng.uniform(0.5, 1.0, size=(n_models, n_tasks))
    return pd.DataFrame(data, index=models, columns=tasks)


class TestScenarioScaling:

    def test_returns_scaling_report(self):
        scores = _make_task_scores()
        report = analyze_scenario_scaling(scores, scenario_counts=[3, 5, 8])
        assert isinstance(report, ScalingReport)
        assert report.axis == "n_scenarios"
        assert len(report.curve) == 3

    def test_full_data_gives_perfect_tau(self):
        scores = _make_task_scores(n_tasks=10)
        report = analyze_scenario_scaling(scores, scenario_counts=[10])
        # With all tasks included, tau should be very high
        assert report.curve[0].kendall_tau >= 0.9

    def test_more_scenarios_higher_tau(self):
        scores = _make_task_scores(n_tasks=20)
        report = analyze_scenario_scaling(
            scores, scenario_counts=[3, 10, 20], n_bootstrap=100,
        )
        taus = [pt.kendall_tau for pt in report.curve]
        # General trend: more scenarios -> higher tau
        assert taus[-1] >= taus[0]

    def test_summary_is_string(self):
        scores = _make_task_scores()
        report = analyze_scenario_scaling(scores, scenario_counts=[3, 5])
        assert isinstance(report.summary(), str)
        assert "n_scenarios" in report.summary()


class TestRankingVariance:

    def test_returns_dataframe(self):
        scores = _make_task_scores()
        result = analyze_ranking_variance(scores, n_bootstrap=50)
        assert isinstance(result, pd.DataFrame)
        assert "model" in result.columns
        assert "mean_rank" in result.columns
        assert "std_rank" in result.columns
        assert "ci_low" in result.columns
        assert "ci_high" in result.columns
        assert len(result) == 5

    def test_mean_rank_in_valid_range(self):
        scores = _make_task_scores(n_models=4)
        result = analyze_ranking_variance(scores, n_bootstrap=100)
        for _, row in result.iterrows():
            assert 1.0 <= row["mean_rank"] <= 4.0

    def test_ci_contains_mean(self):
        scores = _make_task_scores()
        result = analyze_ranking_variance(scores, n_bootstrap=200)
        for _, row in result.iterrows():
            assert row["ci_low"] <= row["mean_rank"] <= row["ci_high"]
