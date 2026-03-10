"""Tests for the procedural generation engine."""

import numpy as np
import pandas as pd

from tabular_bank.generation.engine import generate_sampled_datasets, generate_single_dataset
from tabular_bank.templates.scenarios import sample_scenario

SECRET = "test-secret-for-unit-tests"
ROUND = "test-round"
N_SCENARIOS = 3  # small number for fast tests


def _make_rng():
    return np.random.default_rng(42)


def test_generate_single_dataset():
    """Generate a single dataset and verify basic properties."""
    tmpl = sample_scenario(_make_rng(), scenario_id="test_0")
    ds = generate_single_dataset(SECRET, ROUND, 0, template_override=tmpl)

    assert isinstance(ds.data, pd.DataFrame)
    assert ds.n_samples > 0
    assert ds.n_features > 0
    assert ds.target_name in ds.data.columns
    assert ds.problem_type in ("binary", "multiclass", "regression")
    assert ds.metadata["n_features"] == ds.n_features


def test_generate_sampled_datasets():
    """Generate sampled datasets and verify count."""
    datasets = generate_sampled_datasets(SECRET, ROUND, n_scenarios=N_SCENARIOS)
    assert len(datasets) == N_SCENARIOS


def test_dataset_has_correct_problem_type():
    """Each dataset's problem type matches its sampled template."""
    rng = _make_rng()
    for i in range(N_SCENARIOS):
        tmpl = sample_scenario(rng, scenario_id=f"test_{i}")
        ds = generate_single_dataset(SECRET, ROUND, i, template_override=tmpl)
        assert ds.problem_type == tmpl["problem_type"]


def test_dataset_features_in_range():
    """Informative feature count falls within the scenario's range."""
    rng = _make_rng()
    for i in range(N_SCENARIOS):
        tmpl = sample_scenario(rng, scenario_id=f"test_{i}")
        ds = generate_single_dataset(SECRET, ROUND, i, template_override=tmpl)
        lo, hi = tmpl["n_features_range"]
        n_informative = ds.metadata["n_informative_features"]
        assert lo <= n_informative <= hi, (
            f"Scenario {i}: {n_informative} informative features not in [{lo}, {hi}]"
        )
        assert ds.metadata["n_features"] >= n_informative


def test_dataset_samples_in_range():
    """Sample count falls within the scenario's range."""
    rng = _make_rng()
    for i in range(N_SCENARIOS):
        tmpl = sample_scenario(rng, scenario_id=f"test_{i}")
        ds = generate_single_dataset(SECRET, ROUND, i, template_override=tmpl)
        lo, hi = tmpl["n_samples_range"]
        assert lo <= ds.n_samples <= hi, (
            f"Scenario {i}: {ds.n_samples} samples not in [{lo}, {hi}]"
        )


def test_splits_structure():
    """Verify split structure: 10 repeats x 3 folds."""
    tmpl = sample_scenario(_make_rng(), scenario_id="test_0")
    ds = generate_single_dataset(SECRET, ROUND, 0, template_override=tmpl)

    assert len(ds.splits) == 10
    for repeat_idx, folds in ds.splits.items():
        assert len(folds) == 3
        for fold_idx, (train, test) in folds.items():
            assert isinstance(train, np.ndarray)
            assert isinstance(test, np.ndarray)
            all_indices = set(train.tolist()) | set(test.tolist())
            assert all_indices == set(range(ds.n_samples))
            assert len(set(train.tolist()) & set(test.tolist())) == 0


def test_binary_target_values():
    """Binary classification targets should be 0 or 1."""
    rng = _make_rng()
    for i in range(20):  # sample until we find a binary scenario
        tmpl = sample_scenario(rng, scenario_id=f"test_{i}")
        if tmpl["problem_type"] == "binary":
            ds = generate_single_dataset(SECRET, ROUND, i, template_override=tmpl)
            unique_vals = set(ds.data[ds.target_name].unique())
            assert unique_vals.issubset({0, 1})
            return
    raise AssertionError("No binary scenario found in 20 samples")


def test_multiclass_target_values():
    """Multiclass targets should have the right number of classes."""
    rng = _make_rng()
    for i in range(20):
        tmpl = sample_scenario(rng, scenario_id=f"test_{i}")
        if tmpl["problem_type"] == "multiclass":
            ds = generate_single_dataset(SECRET, ROUND, i, template_override=tmpl)
            unique_vals = ds.data[ds.target_name].unique()
            assert len(unique_vals) <= tmpl["n_classes"]
            return
    raise AssertionError("No multiclass scenario found in 20 samples")


def test_regression_target_is_continuous():
    """Regression targets should be continuous (float)."""
    rng = _make_rng()
    for i in range(20):
        tmpl = sample_scenario(rng, scenario_id=f"test_{i}")
        if tmpl["problem_type"] == "regression":
            ds = generate_single_dataset(SECRET, ROUND, i, template_override=tmpl)
            assert ds.data[ds.target_name].dtype in [np.float64, np.float32]
            return
    raise AssertionError("No regression scenario found in 20 samples")


def test_no_nan_in_target():
    """Target column should never contain NaN."""
    rng = _make_rng()
    for i in range(N_SCENARIOS):
        tmpl = sample_scenario(rng, scenario_id=f"test_{i}")
        ds = generate_single_dataset(SECRET, ROUND, i, template_override=tmpl)
        assert not ds.data[ds.target_name].isna().any(), (
            f"Scenario {i}: Found NaN in target column"
        )


def test_reproducibility():
    """Same inputs produce identical datasets."""
    tmpl = sample_scenario(_make_rng(), scenario_id="repro_test")
    ds1 = generate_single_dataset(SECRET, ROUND, 0, template_override=tmpl)
    ds2 = generate_single_dataset(SECRET, ROUND, 0, template_override=tmpl)

    assert ds1.target_name == ds2.target_name
    assert ds1.n_samples == ds2.n_samples
    assert list(ds1.data.columns) == list(ds2.data.columns)
    pd.testing.assert_frame_equal(ds1.data, ds2.data)
