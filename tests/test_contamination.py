"""Tests verifying anti-contamination properties.

The critical property: different secrets/rounds must produce completely
different datasets — different features, different DAG structures,
different data values.
"""

from synthetic_tab.generation.engine import generate_single_dataset


def test_different_secrets_different_features():
    """Different secrets produce different feature sets."""
    ds1 = generate_single_dataset("secret-alpha", "round-001", 0)
    ds2 = generate_single_dataset("secret-beta", "round-001", 0)

    cols1 = set(ds1.data.columns)
    cols2 = set(ds2.data.columns)
    # Feature names should be mostly different
    overlap = cols1 & cols2
    # At most 1 column can overlap (target might happen to get same name)
    # but feature columns should be different
    feature_overlap = overlap - {ds1.target_name, ds2.target_name}
    assert len(feature_overlap) < len(cols1) * 0.5, (
        f"Too much feature overlap between secrets: {feature_overlap}"
    )


def test_different_rounds_different_features():
    """Different rounds produce different feature sets."""
    ds1 = generate_single_dataset("same-secret", "round-001", 0)
    ds2 = generate_single_dataset("same-secret", "round-002", 0)

    cols1 = set(ds1.data.columns)
    cols2 = set(ds2.data.columns)
    feature_overlap = (cols1 & cols2) - {ds1.target_name, ds2.target_name}
    assert len(feature_overlap) < len(cols1) * 0.5, (
        f"Too much feature overlap between rounds: {feature_overlap}"
    )


def test_different_secrets_different_sample_counts():
    """Different secrets can produce different sample counts."""
    ds1 = generate_single_dataset("secret-one", "round-001", 0)
    ds2 = generate_single_dataset("secret-two", "round-001", 0)
    # Not guaranteed to be different, but likely
    # Just check both are valid
    assert ds1.n_samples > 0
    assert ds2.n_samples > 0


def test_different_secrets_different_data():
    """Different secrets produce different data values."""
    ds1 = generate_single_dataset("secret-aaa", "round-001", 0)
    ds2 = generate_single_dataset("secret-bbb", "round-001", 0)

    # Even if by some chance they have an overlapping column name,
    # the values should be different
    assert ds1.n_samples != ds2.n_samples or not ds1.data.equals(ds2.data), (
        "Datasets with different secrets should not be identical"
    )


def test_same_inputs_identical_output():
    """Same secret + round + scenario = identical dataset."""
    ds1 = generate_single_dataset("fixed-secret", "round-001", 0)
    ds2 = generate_single_dataset("fixed-secret", "round-001", 0)

    assert ds1.target_name == ds2.target_name
    assert ds1.n_samples == ds2.n_samples
    assert list(ds1.data.columns) == list(ds2.data.columns)

    # Check actual data equality
    import pandas.api.types as ptypes
    for col in ds1.data.columns:
        if ptypes.is_string_dtype(ds1.data[col]) or ptypes.is_object_dtype(ds1.data[col]):
            assert (ds1.data[col].astype(str) == ds2.data[col].astype(str)).all(), f"Column {col} differs"
        else:
            assert (ds1.data[col] - ds2.data[col]).abs().max() < 1e-10, f"Column {col} differs"


def test_all_scenarios_differ():
    """Different scenario indices produce different datasets."""
    datasets = [
        generate_single_dataset("test-secret", "round-001", i)
        for i in range(5)
    ]

    # All should have different feature sets
    col_sets = [frozenset(ds.data.columns) for ds in datasets]
    # Allow some overlap but not identical
    for i in range(len(col_sets)):
        for j in range(i + 1, len(col_sets)):
            assert col_sets[i] != col_sets[j], (
                f"Scenarios {i} and {j} have identical columns"
            )
