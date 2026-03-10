"""Procedural feature generation from seed.

Generates feature names, types, distributions, and parameters entirely from
the seed. Feature names use a simple positional ``f_<index>`` scheme to keep
generated datasets easy to read.
"""

from __future__ import annotations

import numpy as np


# Distribution families for continuous features
CONTINUOUS_DISTRIBUTIONS = ["normal", "lognormal", "uniform", "exponential", "beta"]


def generate_features(
    rng: np.random.Generator,
    template: dict,
) -> tuple[list[dict], dict]:
    """Procedurally generate feature specifications from seed + template.

    Feature names use a simple positional scheme such as ``f_0`` and ``f_1``.
    Names are only required to be unique within a dataset.

    Returns:
        features: List of feature spec dicts (name, type, distribution info)
        target: Target feature spec dict
    """
    n_features = int(rng.integers(
        template["n_features_range"][0],
        template["n_features_range"][1] + 1,
    ))
    n_categorical = max(1, int(n_features * template["categorical_ratio"]))
    n_continuous = n_features - n_categorical

    features = []
    used_names: set[str] = set()

    # Generate continuous features
    for _ in range(n_continuous):
        name = _generate_unique_name(rng, used_names)
        dist = str(rng.choice(CONTINUOUS_DISTRIBUTIONS))
        params = _sample_distribution_params(rng, dist)
        features.append({
            "name": name,
            "type": "continuous",
            "distribution": dist,
            "params": params,
        })

    # Generate categorical features
    for _ in range(n_categorical):
        name = _generate_unique_name(rng, used_names)
        n_cats = int(rng.integers(2, 7))
        categories = [_generate_category_label(rng, j) for j in range(n_cats)]
        raw_probs = rng.dirichlet(np.ones(n_cats))
        features.append({
            "name": name,
            "type": "categorical",
            "categories": categories,
            "probs": raw_probs.tolist(),
        })

    # Shuffle feature order
    perm = rng.permutation(len(features))
    features = [features[i] for i in perm]

    # Generate target name
    target_name = "target"

    if template["problem_type"] == "regression":
        target = {
            "name": target_name,
            "type": "continuous",
            "problem_type": "regression",
        }
    elif template["problem_type"] == "binary":
        target = {
            "name": target_name,
            "type": "categorical",
            "problem_type": "binary",
            "n_classes": 2,
        }
    else:  # multiclass
        target = {
            "name": target_name,
            "type": "categorical",
            "problem_type": "multiclass",
            "n_classes": template["n_classes"],
        }

    return features, target


def _generate_unique_name(
    rng: np.random.Generator,
    used_names: set[str],
) -> str:
    """Generate the next dataset-local feature name."""
    while True:
        candidate = _format_name_candidate(rng, used_names)
        if candidate != "target" and candidate not in used_names:
            used_names.add(candidate)
            return candidate


def _format_name_candidate(rng: np.random.Generator, used_names: set[str]) -> str:
    """Assemble the next positional feature identifier."""
    del rng  # Naming is intentionally simple and dataset-local.
    return f"f_{len(used_names)}"


def _generate_category_label(rng: np.random.Generator, index: int) -> str:
    """Generate a simple category label like cat_0, cat_1."""
    return f"cat_{index}"


def _sample_distribution_params(rng: np.random.Generator, dist: str) -> dict:
    """Sample random parameters for a distribution family."""
    if dist == "normal":
        return {
            "mean": float(rng.uniform(-50, 150)),
            "std": float(rng.uniform(1, 50)),
        }
    elif dist == "lognormal":
        return {
            "mean": float(rng.uniform(0, 5)),
            "sigma": float(rng.uniform(0.1, 1.5)),
        }
    elif dist == "uniform":
        low = float(rng.uniform(-100, 100))
        high = low + float(rng.uniform(1, 200))
        return {"low": low, "high": high}
    elif dist == "exponential":
        return {"scale": float(rng.uniform(0.5, 50))}
    elif dist == "beta":
        return {
            "a": float(rng.uniform(0.5, 5)),
            "b": float(rng.uniform(0.5, 5)),
        }
    else:
        raise ValueError(f"Unknown distribution: {dist}")
