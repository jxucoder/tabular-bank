"""Tests for the procedural generation engine."""

import numpy as np
import pandas as pd

from synthetic_tab.generation.engine import generate_all_datasets, generate_single_dataset
from synthetic_tab.templates.scenarios import SCENARIOS


SECRET = "test-secret-for-unit-tests"
ROUND = "test-round"


def test_generate_single_dataset():
    """Generate a single dataset and verify basic properties."""
    ds = generate_single_dataset(SECRET, ROUND, 0)

    assert isinstance(ds.data, pd.DataFrame)
    assert ds.n_samples > 0
    assert ds.n_features > 0
    assert ds.target_name in ds.data.columns
    assert ds.problem_type in ("binary", "multiclass", "regression")


def test_generate_all_datasets():
    """Generate all datasets and verify count matches scenarios."""
    datasets = generate_all_datasets(SECRET, ROUND)
    assert len(datasets) == len(SCENARIOS)


def test_dataset_has_correct_problem_type():
    """Each dataset's problem type matches its scenario template."""
    for i, scenario in enumerate(SCENARIOS):
        ds = generate_single_dataset(SECRET, ROUND, i)
        assert ds.problem_type == scenario["problem_type"]


def test_dataset_features_in_range():
    """Feature count falls within the scenario's range."""
    for i, scenario in enumerate(SCENARIOS):
        ds = generate_single_dataset(SECRET, ROUND, i)
        lo, hi = scenario["n_features_range"]
        assert lo <= ds.n_features <= hi, (
            f"Scenario {i}: {ds.n_features} features not in [{lo}, {hi}]"
        )


def test_dataset_samples_in_range():
    """Sample count falls within the scenario's range."""
    for i, scenario in enumerate(SCENARIOS):
        ds = generate_single_dataset(SECRET, ROUND, i)
        lo, hi = scenario["n_samples_range"]
        assert lo <= ds.n_samples <= hi, (
            f"Scenario {i}: {ds.n_samples} samples not in [{lo}, {hi}]"
        )


def test_splits_structure():
    """Verify split structure: 10 repeats x 3 folds."""
    ds = generate_single_dataset(SECRET, ROUND, 0)

    assert len(ds.splits) == 10  # 10 repeats
    for repeat_idx, folds in ds.splits.items():
        assert len(folds) == 3  # 3 folds
        for fold_idx, (train, test) in folds.items():
            assert isinstance(train, np.ndarray)
            assert isinstance(test, np.ndarray)
            # Train + test should cover all indices
            all_indices = set(train.tolist()) | set(test.tolist())
            assert all_indices == set(range(ds.n_samples))
            # No overlap
            assert len(set(train.tolist()) & set(test.tolist())) == 0


def test_binary_target_values():
    """Binary classification targets should be 0 or 1."""
    # Scenario 0 is binary
    ds = generate_single_dataset(SECRET, ROUND, 0)
    assert ds.problem_type == "binary"
    unique_vals = set(ds.data[ds.target_name].unique())
    assert unique_vals.issubset({0, 1})


def test_multiclass_target_values():
    """Multiclass targets should have the right number of classes."""
    # Scenario 1 is multiclass with 4 classes
    ds = generate_single_dataset(SECRET, ROUND, 1)
    assert ds.problem_type == "multiclass"
    unique_vals = ds.data[ds.target_name].unique()
    assert len(unique_vals) <= SCENARIOS[1]["n_classes"]


def test_regression_target_is_continuous():
    """Regression targets should be continuous (float)."""
    # Scenario 2 is regression
    ds = generate_single_dataset(SECRET, ROUND, 2)
    assert ds.problem_type == "regression"
    assert ds.data[ds.target_name].dtype in [np.float64, np.float32]


def test_no_nan_in_features():
    """Generated data should not contain NaN values."""
    ds = generate_single_dataset(SECRET, ROUND, 0)
    assert not ds.data.isna().any().any(), "Found NaN values in generated data"


def test_reproducibility():
    """Same inputs produce identical datasets."""
    ds1 = generate_single_dataset(SECRET, ROUND, 0)
    ds2 = generate_single_dataset(SECRET, ROUND, 0)

    assert ds1.target_name == ds2.target_name
    assert ds1.n_samples == ds2.n_samples
    assert list(ds1.data.columns) == list(ds2.data.columns)
    pd.testing.assert_frame_equal(ds1.data, ds2.data)
