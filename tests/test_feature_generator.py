import numpy as np

from tabular_bank.generation.feature_generator import (
    _generate_unique_name,
    generate_features,
)


def test_feature_names_use_simple_positional_labels():
    template = {
        "n_features_range": (12, 12),
        "categorical_ratio": 0.25,
        "problem_type": "binary",
    }
    rng = np.random.default_rng(123)

    features, _ = generate_features(rng, template)

    expected_names = {f"f_{i}" for i in range(12)}
    actual_names = {feature["name"] for feature in features}
    assert actual_names == expected_names


def test_feature_names_append_after_sparse_existing_f_names():
    rng = np.random.default_rng(123)
    used_names = {"feature_alpha", "f_1", "f_7"}

    assert _generate_unique_name(rng, used_names) == "f_8"
