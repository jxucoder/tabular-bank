"""Tests for dimension-aware diagnostics."""

import numpy as np
import pandas as pd
import pytest

from tabular_bank.evaluation.diagnostics import (
    DiagnosticReport,
    compute_pareto_frontier,
    run_diagnostics,
)


def _make_data(n_models=5, n_tasks=10, seed=42):
    """Create task scores and metadata with difficulty dimensions."""
    rng = np.random.default_rng(seed)
    models = [f"model_{i}" for i in range(n_models)]
    tasks = [f"task_{i}" for i in range(n_tasks)]

    # Scores: models with higher index are generally better
    scores_data = np.zeros((n_models, n_tasks))
    for i in range(n_models):
        for j in range(n_tasks):
            scores_data[i, j] = 0.5 + 0.05 * i + rng.normal(0, 0.05)

    task_scores = pd.DataFrame(scores_data, index=models, columns=tasks)

    # Metadata with difficulty dimensions
    meta_rows = []
    for j, task in enumerate(tasks):
        meta_rows.append({
            "dataset": task,
            "noise_scale": float(rng.uniform(0.1, 1.0)),
            "nonlinear_prob": float(rng.uniform(0.05, 0.6)),
            "n_samples": int(rng.integers(1000, 15000)),
            "n_features": int(rng.integers(5, 30)),
        })
    task_metadata = pd.DataFrame(meta_rows)

    return task_scores, task_metadata


class TestRunDiagnostics:

    def test_returns_diagnostic_report(self):
        scores, meta = _make_data()
        report = run_diagnostics(scores, meta)
        assert isinstance(report, DiagnosticReport)
        assert len(report.profiles) > 0

    def test_detects_available_dimensions(self):
        scores, meta = _make_data()
        report = run_diagnostics(scores, meta)
        dims = set(p.dimension for p in report.profiles)
        assert "noise_scale" in dims
        assert "nonlinear_prob" in dims

    def test_dimension_importance_computed(self):
        scores, meta = _make_data()
        report = run_diagnostics(scores, meta)
        assert len(report.dimension_importance) > 0
        for val in report.dimension_importance.values():
            assert 0 <= val <= 1.0

    def test_summary_is_string(self):
        scores, meta = _make_data()
        report = run_diagnostics(scores, meta)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Dimension-Aware Diagnostics" in summary

    def test_model_strengths_weaknesses(self):
        scores, meta = _make_data()
        report = run_diagnostics(scores, meta)
        strengths = report.get_model_strengths("model_0")
        weaknesses = report.get_model_weaknesses("model_0")
        assert len(strengths) > 0
        assert len(weaknesses) > 0


class TestParetoFrontier:

    def test_returns_dataframe(self):
        scores = pd.DataFrame(
            {"t1": [0.9, 0.8, 0.7], "t2": [0.85, 0.82, 0.75]},
            index=["fast", "medium", "slow"],
        )
        times = pd.DataFrame(
            {"t1": [0.1, 0.5, 1.0], "t2": [0.12, 0.55, 1.1]},
            index=["fast", "medium", "slow"],
        )
        result = compute_pareto_frontier(scores, times)
        assert "is_pareto" in result.columns
        assert "model" in result.columns

    def test_identifies_pareto_optimal(self):
        # "fast" is fastest but least accurate
        # "slow" is most accurate but slowest
        # "medium" is dominated by neither
        scores = pd.DataFrame(
            {"t1": [0.7, 0.8, 0.9]},
            index=["fast", "medium", "slow"],
        )
        times = pd.DataFrame(
            {"t1": [0.1, 0.5, 1.0]},
            index=["fast", "medium", "slow"],
        )
        result = compute_pareto_frontier(scores, times)
        pareto_models = set(result[result["is_pareto"]]["model"])
        # All three are Pareto-optimal (tradeoff between speed and accuracy)
        assert "fast" in pareto_models
        assert "slow" in pareto_models
