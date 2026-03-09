"""Data sampling from procedurally constructed DAGs.

Samples tabular data by traversing the DAG in topological order, generating
root nodes from their specified distributions, and computing child nodes
via the functional relationships defined in the DAG edges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, softmax

from synthetic_tab.generation.dag_builder import DAGSpec


def sample_dataset(
    rng: np.random.Generator,
    dag: DAGSpec,
    features: list[dict],
    target: dict,
    n_samples: int,
) -> pd.DataFrame:
    """Sample data following the DAG's causal structure.

    1. Topologically sort the DAG (already sorted in dag.nodes)
    2. For root nodes: sample from their specified distribution
    3. For child nodes: compute value as f(parents) + noise
    4. For categorical features: discretize continuous latent → categories
    5. For classification target: apply sigmoid/softmax → sample class labels
    6. For regression target: keep continuous value + noise

    Returns a pandas DataFrame with all features + target.
    """
    feature_lookup = {f["name"]: f for f in features}
    data: dict[str, np.ndarray] = {}

    for node in dag.nodes:
        if node in dag.root_nodes:
            # Sample root node from its distribution
            if node in feature_lookup:
                data[node] = _sample_root(rng, feature_lookup[node], n_samples)
            else:
                # Target as root (shouldn't happen with our DAG builder, but handle it)
                data[node] = rng.normal(0, 1, size=n_samples)
        else:
            # Compute child node from parents
            parent_edges = dag.get_parents(node)
            latent = np.zeros(n_samples)

            for edge in parent_edges:
                parent_data = data[edge.parent]
                # Normalize parent data for numerical stability
                parent_std = np.std(parent_data)
                if parent_std > 0:
                    parent_normalized = (parent_data - np.mean(parent_data)) / parent_std
                else:
                    parent_normalized = parent_data - np.mean(parent_data)

                contribution = _apply_functional_form(
                    parent_normalized, edge, data
                )
                latent += edge.coefficient * contribution

            # Add noise
            noise_scale = dag.noise_scales.get(node, 0.5)
            latent += rng.normal(0, noise_scale, size=n_samples)

            if node == dag.target:
                # Handle target separately
                data[node] = latent
            elif node in feature_lookup:
                feat = feature_lookup[node]
                if feat["type"] == "continuous":
                    # Transform latent back to distribution's domain
                    data[node] = _transform_to_distribution(rng, latent, feat)
                else:
                    # Store latent for now, will be discretized
                    data[node] = latent
            else:
                data[node] = latent

    # Build the DataFrame
    result: dict[str, np.ndarray | list] = {}

    # Process features
    for feat in features:
        name = feat["name"]
        if feat["type"] == "categorical":
            result[name] = _discretize_categorical(rng, data[name], feat)
        else:
            result[name] = data[name]

    # Process target
    target_name = target["name"]
    if target["problem_type"] == "regression":
        # Scale to a reasonable range
        target_data = data[target_name]
        result[target_name] = target_data
    elif target["problem_type"] == "binary":
        probs = expit(data[target_name])
        result[target_name] = rng.binomial(1, probs).astype(int)
    elif target["problem_type"] == "multiclass":
        n_classes = target["n_classes"]
        # Create n_classes logits by splitting the latent
        # Use random projections from the latent value
        logits = np.zeros((n_samples, n_classes))
        for c in range(n_classes):
            # Each class gets a different linear combination of the latent
            shift = (c - n_classes / 2) * 0.5
            logits[:, c] = data[target_name] * (1 - 0.3 * c) + shift
            logits[:, c] += rng.normal(0, 0.3, size=n_samples)
        probs = softmax(logits, axis=1)
        result[target_name] = np.array([
            rng.choice(n_classes, p=probs[i])
            for i in range(n_samples)
        ]).astype(int)

    return pd.DataFrame(result)


def _sample_root(
    rng: np.random.Generator,
    feature: dict,
    n_samples: int,
) -> np.ndarray:
    """Sample a root node from its specified distribution."""
    if feature["type"] == "categorical":
        # Sample from uniform for latent, will be discretized later
        return rng.normal(0, 1, size=n_samples)

    dist = feature["distribution"]
    params = feature["params"]

    if dist == "normal":
        return rng.normal(params["mean"], params["std"], size=n_samples)
    elif dist == "lognormal":
        return rng.lognormal(params["mean"], params["sigma"], size=n_samples)
    elif dist == "uniform":
        return rng.uniform(params["low"], params["high"], size=n_samples)
    elif dist == "exponential":
        return rng.exponential(params["scale"], size=n_samples)
    elif dist == "beta":
        return rng.beta(params["a"], params["b"], size=n_samples)
    else:
        return rng.normal(0, 1, size=n_samples)


def _apply_functional_form(
    parent_data: np.ndarray,
    edge,
    all_data: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply the edge's functional form to parent data."""
    if edge.form == "linear":
        return parent_data
    elif edge.form == "quadratic":
        return parent_data ** 2
    elif edge.form == "threshold":
        return (parent_data > edge.threshold).astype(float)
    elif edge.form == "interaction":
        if edge.interaction_parent and edge.interaction_parent in all_data:
            other = all_data[edge.interaction_parent]
            other_std = np.std(other)
            if other_std > 0:
                other_normalized = (other - np.mean(other)) / other_std
            else:
                other_normalized = other - np.mean(other)
            return parent_data * other_normalized
        return parent_data
    else:
        return parent_data


def _transform_to_distribution(
    rng: np.random.Generator,
    latent: np.ndarray,
    feature: dict,
) -> np.ndarray:
    """Transform latent values to match the feature's target distribution."""
    dist = feature["distribution"]
    params = feature["params"]

    # Use rank-based transformation to preserve ordering while matching distribution
    # This keeps the causal relationships intact
    ranks = np.argsort(np.argsort(latent))
    n = len(latent)
    quantiles = (ranks + 0.5) / n

    if dist == "normal":
        from scipy.stats import norm
        return norm.ppf(quantiles, loc=params["mean"], scale=params["std"])
    elif dist == "lognormal":
        from scipy.stats import lognorm
        return lognorm.ppf(quantiles, s=params["sigma"], scale=np.exp(params["mean"]))
    elif dist == "uniform":
        return params["low"] + quantiles * (params["high"] - params["low"])
    elif dist == "exponential":
        from scipy.stats import expon
        return expon.ppf(quantiles, scale=params["scale"])
    elif dist == "beta":
        from scipy.stats import beta
        return beta.ppf(quantiles, params["a"], params["b"])
    else:
        return latent


def _discretize_categorical(
    rng: np.random.Generator,
    latent: np.ndarray,
    feature: dict,
) -> list[str]:
    """Discretize continuous latent values into categorical labels."""
    categories = feature["categories"]
    probs = feature["probs"]
    n = len(latent)

    # Use latent values to influence category assignment while
    # maintaining approximate target probabilities
    # Sort by latent value and assign categories by cumulative probability
    sorted_indices = np.argsort(latent)
    result = [""] * n
    cum_probs = np.cumsum(probs)

    for i, idx in enumerate(sorted_indices):
        frac = i / n
        cat_idx = 0
        for j, cp in enumerate(cum_probs):
            if frac < cp:
                cat_idx = j
                break
        else:
            cat_idx = len(categories) - 1
        result[idx] = categories[cat_idx]

    return result
