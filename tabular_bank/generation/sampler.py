"""Data sampling from procedurally constructed DAGs.

Samples tabular data by traversing the DAG in topological order, generating
root nodes from their specified distributions, and computing child nodes
via the functional relationships defined in the DAG edges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, softmax

from tabular_bank.generation.dag_builder import DAGSpec
from tabular_bank.generation.feature_generator import _generate_unique_name


def sample_dataset(
    rng: np.random.Generator,
    dag: DAGSpec,
    features: list[dict],
    target: dict,
    n_samples: int,
    template: dict | None = None,
) -> pd.DataFrame:
    """Sample data following the DAG's causal structure.

    1. Pre-sample latent confounders (unobserved common causes)
    2. For root nodes: sample from their specified distribution,
       optionally with AR(1) temporal autocorrelation
    3. For child nodes: compute value as f(parents) + confounder effects + noise
    4. For categorical features: discretize continuous latent -> categories
    5. For classification target: apply sigmoid/softmax -> sample class labels
    6. For regression target: keep continuous value + noise

    Returns a pandas DataFrame with all features + target.
    """
    feature_lookup = {f["name"]: f for f in features}
    data: dict[str, np.ndarray] = {}

    # Pre-sample latent confounder signals (unobserved, never in output)
    confounder_signals: dict[str, np.ndarray] = {}
    for latent_name in dag.confounders:
        confounder_signals[latent_name] = rng.normal(0, 1, size=n_samples)

    # Pre-sample correlated Gaussian latents for root nodes, then transform
    # each to its target marginal distribution.  The correlation strength is
    # controlled by the template / difficulty preset.
    corr_strength = 0.0
    if template is not None:
        diff = template.get("difficulty", {})
        if isinstance(diff, dict):
            corr_strength = diff.get("root_correlation_strength", 0.0)
        elif isinstance(diff, str):
            from tabular_bank.templates.scenarios import get_difficulty_preset
            corr_strength = get_difficulty_preset(diff).get("root_correlation_strength", 0.0)
    root_latents = _sample_correlated_roots(rng, dag.root_nodes, n_samples, corr_strength)

    for node in dag.nodes:
        if node in dag.root_nodes:
            latent = np.array(root_latents[node], copy=True)

            # Root nodes can still receive latent confounder inputs. Apply them
            # before any autocorrelation or distribution transform so the final
            # feature retains the intended support.
            for edge in dag.get_parents(node):
                if edge.is_confounder:
                    latent += edge.coefficient * confounder_signals[edge.parent]

            rho = dag.autocorr.get(node, 0.0)
            if rho > 0:
                latent = _apply_autocorr(latent, rho)

            if node in feature_lookup:
                feat = feature_lookup[node]
                if feat["type"] == "continuous":
                    raw = _transform_to_distribution(rng, latent, feat)
                else:
                    raw = latent
            else:
                raw = latent
            data[node] = raw
        else:
            # Compute child node from parents
            parent_edges = dag.get_parents(node)
            latent = np.zeros(n_samples)

            for edge in parent_edges:
                if edge.is_confounder:
                    # Pull signal from pre-sampled confounder
                    parent_data = confounder_signals[edge.parent]
                else:
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
        imbalance_ratio = 0.5
        if template is not None:
            imbalance_ratio = template.get("imbalance_ratio", 0.5)
        # Shift the sigmoid to achieve the desired positive-class ratio.
        # expit(x + bias) shifts the mean probability; we solve for the bias
        # that maps the median latent value to the target ratio.
        from scipy.special import logit
        bias = logit(imbalance_ratio)
        probs = expit(data[target_name] + bias)
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

    # Inject noise/redundant features if configured
    noise_ratio = 0.0
    if template is not None:
        noise_ratio = template.get("noise_feature_ratio", 0.0)
    if noise_ratio > 0:
        result = _inject_noise_features(rng, result, features, noise_ratio, n_samples)

    informative_cols = [feat["name"] for feat in features]
    extra_cols = sorted(
        c for c in result if c not in informative_cols and c != target["name"]
    )
    ordered = {c: result[c] for c in informative_cols}
    for col in extra_cols:
        ordered[col] = result[col]
    ordered[target["name"]] = result[target["name"]]
    return pd.DataFrame(ordered)


def _apply_autocorr(raw: np.ndarray, rho: float) -> np.ndarray:
    """Apply AR(1) autocorrelation to an i.i.d. signal.

    Transforms independent samples into a temporally correlated sequence:
        x[t] = rho * x[t-1] + sqrt(1 - rho^2) * raw[t]

    The sqrt(1 - rho^2) scaling preserves unit variance regardless of rho,
    so downstream distribution transforms remain valid.
    """
    out = np.empty_like(raw)
    scale = np.sqrt(1.0 - rho ** 2)
    out[0] = raw[0]
    for t in range(1, len(raw)):
        out[t] = rho * out[t - 1] + scale * raw[t]
    return out


def _sample_correlated_roots(
    rng: np.random.Generator,
    root_nodes: list[str],
    n_samples: int,
    strength: float,
) -> dict[str, np.ndarray]:
    """Sample root node latents jointly from a multivariate Gaussian.

    A random correlation matrix is generated with off-diagonal entries
    scaled by ``strength`` (0 = independent, 1 = maximally correlated).
    The resulting samples are standard-normal marginals with the specified
    correlation structure; downstream callers transform these to the
    target distributions via quantile mapping.
    """
    k = len(root_nodes)
    if k <= 1 or strength <= 0:
        return {n: rng.normal(0, 1, size=n_samples) for n in root_nodes}

    # Build a random correlation matrix via the "onion" method:
    # sample a random unit vector per pair and blend toward identity.
    raw = rng.normal(0, 1, size=(k, k))
    cov = raw @ raw.T
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    # Blend toward identity to control strength
    corr = (1 - strength) * np.eye(k) + strength * corr
    np.fill_diagonal(corr, 1.0)

    # Ensure positive semi-definite (numerical safety)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d_fix = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d_fix, d_fix)

    samples = rng.multivariate_normal(np.zeros(k), corr, size=n_samples)
    return {root_nodes[i]: samples[:, i] for i in range(k)}


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
    elif edge.form == "sigmoid":
        return expit(parent_data)
    elif edge.form == "piecewise_linear":
        below = parent_data <= edge.threshold
        out = np.empty_like(parent_data)
        out[below] = edge.slope_left * (parent_data[below] - edge.threshold)
        out[~below] = edge.slope_right * (parent_data[~below] - edge.threshold)
        return out
    elif edge.form == "sinusoidal":
        return np.sin(edge.frequency * parent_data)
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


def _inject_noise_features(
    rng: np.random.Generator,
    result: dict[str, np.ndarray | list],
    features: list[dict],
    noise_ratio: float,
    n_samples: int,
) -> dict[str, np.ndarray | list]:
    """Inject uninformative features to test model robustness.

    Three types, chosen at random per noise feature:
      - Pure noise: independent random draws (no signal)
      - Near-copy: an existing feature plus Gaussian noise
      - Random combo: random linear combination of 2-3 existing features
    """
    n_informative = len(features)
    n_noise = max(1, int(n_informative * noise_ratio))
    used_names = {f["name"] for f in features}

    continuous_names = [f["name"] for f in features if f["type"] == "continuous"]

    for _ in range(n_noise):
        name = _generate_unique_name(rng, used_names)
        kind = int(rng.integers(0, 3))

        if kind == 0 or not continuous_names:
            # Pure noise
            result[name] = rng.normal(0, 1, size=n_samples)
        elif kind == 1:
            # Near-copy of an existing continuous feature
            src = str(rng.choice(continuous_names))
            src_data = np.asarray(result[src], dtype=float)
            noise_std = np.std(src_data) * float(rng.uniform(0.05, 0.3))
            result[name] = src_data + rng.normal(0, noise_std, size=n_samples)
        else:
            # Random linear combination of 2-3 existing features
            n_combo = min(int(rng.integers(2, 4)), len(continuous_names))
            chosen = rng.choice(continuous_names, size=n_combo, replace=False)
            combo = np.zeros(n_samples)
            for c in chosen:
                w = float(rng.normal(0, 1))
                combo += w * np.asarray(result[c], dtype=float)
            combo += rng.normal(0, np.std(combo) * 0.1, size=n_samples)
            result[name] = combo

    return result
