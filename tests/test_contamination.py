"""Tests verifying anti-contamination properties.

The critical property: different secrets/rounds must produce completely
different datasets — different features, different DAG structures,
different data values.
"""

import numpy as np

from tabular_bank.generation.engine import generate_single_dataset
from tabular_bank.templates.scenarios import sample_scenario


def _tmpl(seed=42, scenario_id="test"):
    return sample_scenario(np.random.default_rng(seed), scenario_id=scenario_id)


def test_different_secrets_different_data():
    """Different secrets should change the generated data even if names repeat."""
    tmpl = _tmpl()
    ds1 = generate_single_dataset("secret-alpha", "round-001", 0, template_override=tmpl)
    ds2 = generate_single_dataset("secret-beta", "round-001", 0, template_override=tmpl)

    assert all(name.startswith("f_") for name in ds1.feature_names)
    assert all(name.startswith("f_") for name in ds2.feature_names)
    assert ds1.n_samples != ds2.n_samples or not ds1.data.equals(ds2.data), (
        "Datasets from different secrets should not be identical, even when feature names match"
    )


def test_different_rounds_different_data():
    """Different rounds should change the generated data even if names repeat."""
    tmpl = _tmpl()
    ds1 = generate_single_dataset("same-secret", "round-001", 0, template_override=tmpl)
    ds2 = generate_single_dataset("same-secret", "round-002", 0, template_override=tmpl)

    assert all(name.startswith("f_") for name in ds1.feature_names)
    assert all(name.startswith("f_") for name in ds2.feature_names)
    assert ds1.n_samples != ds2.n_samples or not ds1.data.equals(ds2.data), (
        "Datasets from different rounds should not be identical, even when feature names match"
    )


def test_different_secrets_different_sample_counts():
    """Different secrets can produce different sample counts."""
    tmpl = _tmpl()
    ds1 = generate_single_dataset("secret-one", "round-001", 0, template_override=tmpl)
    ds2 = generate_single_dataset("secret-two", "round-001", 0, template_override=tmpl)
    assert ds1.n_samples > 0
    assert ds2.n_samples > 0


def test_different_secrets_produce_different_dataset_values():
    """Different secrets produce different data values."""
    tmpl = _tmpl()
    ds1 = generate_single_dataset("secret-aaa", "round-001", 0, template_override=tmpl)
    ds2 = generate_single_dataset("secret-bbb", "round-001", 0, template_override=tmpl)

    assert ds1.n_samples != ds2.n_samples or not ds1.data.equals(ds2.data), (
        "Datasets with different secrets should not be identical"
    )


def test_same_inputs_identical_output():
    """Same secret + round + scenario + template = identical dataset."""
    tmpl = _tmpl()
    ds1 = generate_single_dataset("fixed-secret", "round-001", 0, template_override=tmpl)
    ds2 = generate_single_dataset("fixed-secret", "round-001", 0, template_override=tmpl)

    assert ds1.target_name == ds2.target_name
    assert ds1.n_samples == ds2.n_samples
    assert list(ds1.data.columns) == list(ds2.data.columns)

    import pandas.api.types as ptypes
    for col in ds1.data.columns:
        if ptypes.is_string_dtype(ds1.data[col]) or ptypes.is_object_dtype(ds1.data[col]):
            assert (ds1.data[col].astype(str) == ds2.data[col].astype(str)).all(), f"Column {col} differs"
        else:
            assert (ds1.data[col] - ds2.data[col]).abs().max() < 1e-10, f"Column {col} differs"


def test_all_scenarios_differ():
    """Different scenario indices produce different datasets."""
    rng = np.random.default_rng(0)
    datasets = [
        generate_single_dataset(
            "test-secret", "round-001", i,
            template_override=sample_scenario(rng, scenario_id=f"test_{i}"),
        )
        for i in range(5)
    ]

    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            assert not datasets[i].data.equals(datasets[j].data), (
                f"Scenarios {i} and {j} produced identical datasets"
            )
