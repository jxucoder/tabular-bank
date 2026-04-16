"""Tests for the forecasting task generation pipeline."""

import numpy as np
import pandas as pd
import pytest

from tabular_bank.generation.engine import (
    GeneratedDataset,
    generate_sampled_datasets,
    _add_lagged_features,
    _create_temporal_splits,
)
from tabular_bank.forecasting import LastValue, LinearTrend, RollingMean
from tabular_bank.templates.scenarios import sample_scenario


class TestForecastingScenarioSampling:

    def test_forecasting_type_sampled(self):
        """Forecasting tasks can be sampled from the scenario space."""
        rng = np.random.default_rng(42)
        # Force forecasting
        tmpl = sample_scenario(
            rng,
            scenario_id="test",
            scenario_space={"problem_type_weights": {"forecasting": 1.0}},
        )
        assert tmpl["problem_type"] == "forecasting"
        assert "n_lags" in tmpl
        assert "forecast_horizon" in tmpl
        assert tmpl["difficulty"]["temporal_prob"] == 1.0

    def test_forecasting_forces_temporal_structure(self):
        rng = np.random.default_rng(0)
        tmpl = sample_scenario(
            rng,
            scenario_id="test",
            scenario_space={"problem_type_weights": {"forecasting": 1.0}},
        )
        assert tmpl["difficulty"]["temporal_prob"] == 1.0
        assert tmpl["difficulty"]["max_autocorr"] >= 0.5


class TestLaggedFeatures:

    def test_adds_lag_columns(self):
        df = pd.DataFrame({
            "f_0": np.arange(20, dtype=float),
            "target": np.arange(20, dtype=float) * 2,
        })
        features = [{"name": "f_0", "type": "continuous"}]
        result = _add_lagged_features(df, "target", features, n_lags=2, forecast_horizon=1)

        assert "target_lag1" in result.columns
        assert "target_lag2" in result.columns
        assert "f_0_lag1" in result.columns
        # Rows should be shorter due to NaN dropping
        assert len(result) < len(df)

    def test_target_shifted_forward(self):
        df = pd.DataFrame({
            "f_0": np.arange(100, dtype=float),
            "target": np.arange(100, dtype=float),
        })
        features = [{"name": "f_0", "type": "continuous"}]
        result = _add_lagged_features(df, "target", features, n_lags=1, forecast_horizon=1)

        # For the first valid row, target should be the *next* value
        # (shifted forward by horizon=1)
        # After shifting: row 1's target becomes original row 2's target
        # After lag: lag1 for row 1 is original row 0's value
        assert len(result) > 0
        # All targets should be finite
        assert result["target"].notna().all()


class TestTemporalSplits:

    def test_creates_temporal_splits(self):
        df = pd.DataFrame({"x": range(100)})
        splits = _create_temporal_splits(df, split_seed=42)

        assert len(splits) == 5  # 5 repeats
        for repeat, folds in splits.items():
            assert len(folds) == 3  # 3 folds (train fractions)
            for fold, (train_idx, test_idx) in folds.items():
                # Train comes before test (temporal)
                assert train_idx.max() < test_idx.min()

    def test_splits_cover_all_data(self):
        df = pd.DataFrame({"x": range(100)})
        splits = _create_temporal_splits(df, split_seed=42)

        for repeat, folds in splits.items():
            for fold, (train_idx, test_idx) in folds.items():
                combined = set(train_idx) | set(test_idx)
                assert len(combined) == 100


class TestForecastingEndToEnd:

    def test_generate_forecasting_dataset(self):
        """Generate a forecasting dataset end-to-end."""
        datasets = generate_sampled_datasets(
            "test-secret",
            round_id="round-fcst",
            n_scenarios=3,
            scenario_space={"problem_type_weights": {"forecasting": 1.0}},
        )
        assert len(datasets) == 3
        for ds in datasets:
            assert ds.problem_type == "forecasting"
            assert ds.metadata["problem_type"] == "forecasting"
            assert "n_lags" in ds.metadata
            # Should have lag columns
            lag_cols = [c for c in ds.data.columns if "_lag" in c]
            assert len(lag_cols) > 0
            # Splits should be temporal (train before test)
            for repeat, folds in ds.splits.items():
                for fold, (train_idx, test_idx) in folds.items():
                    assert train_idx.max() < test_idx.min()


class TestForecastingBaselines:

    def _make_data(self, n=100):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, n).cumsum()
        df = pd.DataFrame({
            "f_0": rng.normal(0, 1, n),
            "target_lag1": np.roll(data, 1),
            "target_lag2": np.roll(data, 2),
            "target_lag3": np.roll(data, 3),
        })
        df.iloc[:3] = 0  # clean up rolled values
        y = pd.Series(data)
        return df, y

    def test_last_value_returns_lag1(self):
        X, y = self._make_data()
        model = LastValue()
        model.fit(X, y)
        pred = model.predict(X)
        # Should return target_lag1
        np.testing.assert_array_equal(pred, X["target_lag1"].values)

    def test_rolling_mean_averages_lags(self):
        X, y = self._make_data()
        model = RollingMean()
        model.fit(X, y)
        pred = model.predict(X)
        # Should be the mean of target_lag columns
        expected = X[["target_lag1", "target_lag2", "target_lag3"]].mean(axis=1).values
        np.testing.assert_allclose(pred, expected)

    def test_linear_trend_extrapolates(self):
        X, y = self._make_data()
        model = LinearTrend()
        model.fit(X, y)
        pred = model.predict(X)
        assert len(pred) == len(X)
        assert np.all(np.isfinite(pred))

    def test_all_baselines_are_regressors(self):
        from sklearn.base import is_regressor
        assert is_regressor(LastValue())
        assert is_regressor(RollingMean())
        assert is_regressor(LinearTrend())


class TestForecastingMetrics:

    def test_mae_metric(self):
        from tabular_bank.runner import _evaluate_metric

        class DummyModel:
            def predict(self, X):
                return np.ones(len(X))

        X = pd.DataFrame({"x": [1, 2, 3]})
        y = pd.Series([1.0, 2.0, 3.0])
        score = _evaluate_metric(DummyModel(), X, y, "forecasting", "mae")
        # MAE of [1,2,3] vs [1,1,1] = (0+1+2)/3 = 1.0, negated = -1.0
        assert score == pytest.approx(-1.0)

    def test_directional_accuracy_metric(self):
        from tabular_bank.runner import _evaluate_metric

        class PerfectModel:
            def predict(self, X):
                return np.array([1.0, 2.0, 3.0])

        X = pd.DataFrame({"x": [1, 2, 3]})
        y = pd.Series([1.0, 2.0, 3.0])
        score = _evaluate_metric(PerfectModel(), X, y, "forecasting", "directional_accuracy")
        assert score == pytest.approx(1.0)
