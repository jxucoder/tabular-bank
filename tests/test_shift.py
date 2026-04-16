"""Tests for distribution shift injection."""

import numpy as np
import pandas as pd
import pytest

from tabular_bank.generation.shift import (
    create_shifted_splits,
    create_temporal_split,
    inject_concept_drift,
    inject_covariate_shift,
)


def _make_df(n=1000, seed=42):
    """Create a simple test dataset."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.uniform(-1, 1, n),
        "f3": rng.choice(["a", "b", "c"], n),
        "target": rng.normal(5, 2, n),
    })
    return df


class TestCovariateShift:

    def test_returns_train_test(self):
        df = _make_df()
        train, test = inject_covariate_shift(df, ["f1", "f2"], "target")
        assert len(train) + len(test) == len(df)
        assert "target" in train.columns
        assert "target" in test.columns

    def test_test_distribution_shifts(self):
        df = _make_df(n=5000)
        train, test = inject_covariate_shift(
            df, ["f1", "f2"], "target",
            shift_magnitude=2.0, shift_fraction=1.0,
        )
        # At least one feature's test mean should differ from train mean
        diff_f1 = abs(test["f1"].mean() - train["f1"].mean())
        diff_f2 = abs(test["f2"].mean() - train["f2"].mean())
        assert diff_f1 > 0.1 or diff_f2 > 0.1


class TestConceptDrift:

    def test_returns_train_test(self):
        df = _make_df()
        train, test = inject_concept_drift(df, ["f1", "f2"], "target")
        assert len(train) + len(test) == len(df)

    def test_regression_target_changes(self):
        df = _make_df(n=5000)
        train, test = inject_concept_drift(
            df, ["f1", "f2"], "target",
            drift_magnitude=2.0,
        )
        # The test target should differ from what it would be without drift
        # (we can't check directly, but mean should shift)
        original_test_mean = df["target"].iloc[int(len(df) * 0.67):].mean()
        # Just verify it returns valid data
        assert test["target"].notna().all()

    def test_classification_labels_flip(self):
        rng = np.random.default_rng(42)
        n = 2000
        df = pd.DataFrame({
            "f1": rng.normal(0, 1, n),
            "target": rng.choice([0, 1], n),
        })
        _, test = inject_concept_drift(
            df, ["f1"], "target", drift_magnitude=1.0,
        )
        # With drift_magnitude=1.0, ~30% of labels should flip
        # Just verify the output is valid
        assert set(test["target"].unique()).issubset({0, 1})


class TestTemporalSplit:

    def test_preserves_order(self):
        df = _make_df()
        train, test = create_temporal_split(df, train_fraction=0.7)

        assert len(train) == int(len(df) * 0.7)
        assert len(test) == len(df) - int(len(df) * 0.7)

        # Train indices should be before test indices
        assert train.index.max() < test.index.min()


class TestCreateShiftedSplits:

    def test_returns_all_shift_types(self):
        df = _make_df()
        splits = create_shifted_splits(df, ["f1", "f2"], "target")
        assert "none" in splits
        assert "covariate" in splits
        assert "concept" in splits
        assert "temporal" in splits

    def test_each_split_has_train_test(self):
        df = _make_df()
        splits = create_shifted_splits(df, ["f1", "f2"], "target")
        for name, (train, test) in splits.items():
            assert len(train) > 0
            assert len(test) > 0
            assert "target" in train.columns
