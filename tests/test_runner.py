"""Tests for the benchmark runner."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from synthetic_tab.runner import _is_compatible, run_benchmark


def test_classifier_compatible_with_binary():
    assert _is_compatible(DummyClassifier(), "binary") is True


def test_classifier_compatible_with_multiclass():
    assert _is_compatible(DummyClassifier(), "multiclass") is True


def test_classifier_incompatible_with_regression():
    assert _is_compatible(DummyClassifier(), "regression") is False


def test_regressor_compatible_with_regression():
    assert _is_compatible(DummyRegressor(), "regression") is True


def test_regressor_incompatible_with_classification():
    assert _is_compatible(DummyRegressor(), "binary") is False
    assert _is_compatible(DummyRegressor(), "multiclass") is False


def test_untyped_model_compatible_with_all():
    """A model without _estimator_type is allowed on every task type."""

    class BareModel:
        pass

    m = BareModel()
    assert _is_compatible(m, "binary") is True
    assert _is_compatible(m, "regression") is True


def test_run_benchmark_skips_incompatible_tasks():
    """Classifiers should only run on classification tasks, not crash."""
    models = {"dummy_clf": DummyClassifier(strategy="most_frequent")}
    result = run_benchmark(
        models=models,
        round_id="test-round",
        master_secret="test-secret-runner",
    )
    # All results should be from classification tasks only
    for r in result.results:
        assert r.metric_name in ("roc_auc", "log_loss"), (
            f"Classifier ran on non-classification task: metric={r.metric_name}"
        )
    # Should have at least some results (there are classification scenarios)
    assert len(result.results) > 0
