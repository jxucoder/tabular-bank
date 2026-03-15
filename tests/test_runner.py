"""Tests for the benchmark runner."""

import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from tabular_bank.runner import _encode_features, _evaluate_metric, _is_compatible, run_benchmark


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
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_benchmark(
            models=models,
            round_id="test-round",
            master_secret="test-secret-runner",
            cache_dir=tmpdir,
        )
    # All results should be from classification tasks only
    for r in result.results:
        assert r.metric_name in ("roc_auc", "log_loss"), (
            f"Classifier ran on non-classification task: metric={r.metric_name}"
        )
    # Should have at least some results (there are classification scenarios)
    assert len(result.results) > 0


def test_log_loss_fallback_without_predict_proba():
    """Models without predict_proba should not crash on log_loss tasks."""
    from sklearn.tree import DecisionTreeClassifier

    # Build a tiny multiclass dataset
    X_train = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [6, 5, 4, 3, 2, 1]})
    y_train = pd.Series([0, 1, 2, 3, 0, 1])
    X_test = pd.DataFrame({"a": [2, 4], "b": [5, 3]})
    y_test = pd.Series([1, 3])

    # Wrap a fitted model that only exposes predict (no predict_proba)
    class PredictOnlyModel:
        def __init__(self, inner):
            self._inner = inner

        def predict(self, X):
            return self._inner.predict(X)

    inner = DecisionTreeClassifier().fit(X_train, y_train)
    model = PredictOnlyModel(inner)

    # This should not raise; it should fall back to accuracy
    score = _evaluate_metric(model, X_test, y_test, "multiclass", "log_loss")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0  # accuracy is in [0, 1]


def test_encode_features_imputes_numeric_only_missing_values():
    """Numeric-only datasets should still have NaN imputed."""
    X_train = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10.0, 11.0, 12.0]})
    X_test = pd.DataFrame({"a": [np.nan], "b": [13.0]})

    train_enc, test_enc = _encode_features(X_train, X_test)

    assert not train_enc.isna().any().any()
    assert not test_enc.isna().any().any()
    assert train_enc.loc[1, "a"] == pytest.approx(2.0)
    assert test_enc.loc[0, "a"] == pytest.approx(2.0)


def test_encode_features_imputes_test_only_missing_values():
    """Test-set numeric NaN should be filled even when train has no missing values."""
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]})
    X_test = pd.DataFrame({"a": [np.nan], "b": [None]})

    train_enc, test_enc = _encode_features(X_train, X_test)

    assert not train_enc.isna().any().any()
    assert not test_enc.isna().any().any()
    assert test_enc.loc[0, "a"] == pytest.approx(2.0)
    assert test_enc.loc[0, "b"] == -1


def test_roc_auc_falls_back_when_test_split_has_single_class():
    """ROC-AUC should not crash when y_test contains only one class."""

    class ConstantPredictor:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    X_test = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y_test = pd.Series([0, 0, 0])

    score = _evaluate_metric(ConstantPredictor(), X_test, y_test, "binary", "roc_auc")
    assert score == 1.0
