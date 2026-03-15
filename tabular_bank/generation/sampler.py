"""Data sampling from procedurally constructed DAGs.

Samples tabular data by traversing the DAG in topological order, generating
root nodes from their specified distributions, and computing child nodes
via sampled causal mechanisms attached to DAG edges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.special import expit, softmax
from scipy.stats import rankdata

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

                contribution = _apply_mechanism(
                    parent_normalized, edge, data
                )
                latent += edge.coefficient * contribution

            # Add node-specific residual noise. Newer DAG specs carry a
            # structured noise model; older ones fall back to a scalar scale.
            noise_model = dag.noise_models.get(
                node,
                {"type": "homoscedastic", "scale": dag.noise_scales.get(node, 0.5)},
            )
            latent += _sample_node_noise(rng, noise_model, data, n_samples)

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
        # Centre the latent first so the bias term controls the class ratio
        # regardless of the latent's mean (which can drift after DAG
        # processing with asymmetric mechanisms or noise).
        from scipy.special import logit
        latent_centered = data[target_name] - np.mean(data[target_name])
        bias = logit(imbalance_ratio)
        probs = expit(latent_centered + bias)
        result[target_name] = rng.binomial(1, probs).astype(int)
    elif target["problem_type"] == "multiclass":
        n_classes = target["n_classes"]
        # Create n_classes logits using independent random projections so
        # that every class has a unique, unbiased relationship with the
        # latent signal.  Previous code used a deterministic formula that
        # reversed the signal for higher classes and imposed a fixed
        # frequency ordering.
        weights = rng.normal(0, 1, size=n_classes)
        offsets = rng.normal(0, 0.5, size=n_classes)
        logits = np.zeros((n_samples, n_classes))
        for c in range(n_classes):
            logits[:, c] = weights[c] * data[target_name] + offsets[c]
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

    Note: the temporal structure uses the *row index* as the time axis.
    Because cross-validation splits shuffle rows randomly, models cannot
    exploit the autocorrelation directly.  The primary effect is to reduce
    the effective sample size of the generated data, making estimation
    harder — a legitimate difficulty lever even with random splits.
    """
    rho = float(np.clip(rho, -0.999, 0.999))
    scale = np.sqrt(1.0 - rho ** 2)
    # Vectorized AR(1) via IIR filter: y[t] = rho*y[t-1] + scale*raw[t]
    # lfilter with b=[scale], a=[1, -rho] computes this in C, avoiding a
    # slow Python loop over all samples.
    scaled_raw = raw.copy()
    scaled_raw[0] = raw[0]  # first element passes through unscaled
    scaled_raw[1:] = scale * raw[1:]
    out = lfilter([1.0], [1.0, -rho], scaled_raw)
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
    strength = float(np.clip(strength, 0.0, 1.0))
    if k <= 1 or strength <= 0:
        return {n: rng.normal(0, 1, size=n_samples) for n in root_nodes}

    # Build a random correlation matrix via a Wishart-derived construction:
    # sample a Gaussian matrix, form its Gram matrix, then normalise to a
    # correlation matrix and blend toward identity to control strength.
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


def _apply_mechanism(
    parent_data: np.ndarray,
    edge,
    all_data: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply an edge's mechanism to normalized parent data."""
    mechanism = edge.mechanism
    mechanism_type = mechanism["type"]

    if mechanism_type == "linear":
        return parent_data
    elif mechanism_type == "quadratic":
        center = float(mechanism.get("center", 0.0))
        return (parent_data - center) ** 2
    elif mechanism_type == "threshold":
        threshold = float(mechanism.get("threshold", 0.0))
        low_value = float(mechanism.get("low_value", 0.0))
        high_value = float(mechanism.get("high_value", 1.0))
        return np.where(parent_data > threshold, high_value, low_value)
    elif mechanism_type == "sigmoid":
        slope = float(mechanism.get("slope", 1.0))
        offset = float(mechanism.get("offset", 0.0))
        return expit(slope * (parent_data - offset))
    elif mechanism_type == "tanh":
        slope = float(mechanism.get("slope", 1.0))
        offset = float(mechanism.get("offset", 0.0))
        return np.tanh(slope * (parent_data - offset))
    elif mechanism_type == "piecewise_linear":
        threshold = float(mechanism.get("threshold", 0.0))
        slope_left = float(mechanism.get("slope_left", 0.0))
        slope_right = float(mechanism.get("slope_right", 1.0))
        below = parent_data <= threshold
        out = np.empty_like(parent_data)
        out[below] = slope_left * (parent_data[below] - threshold)
        out[~below] = slope_right * (parent_data[~below] - threshold)
        return out
    elif mechanism_type == "sinusoidal":
        frequency = float(mechanism.get("frequency", 1.0))
        phase = float(mechanism.get("phase", 0.0))
        return np.sin(frequency * parent_data + phase)
    elif mechanism_type == "spline":
        knots = np.asarray(mechanism["knots"], dtype=float)
        values = np.asarray(mechanism["values"], dtype=float)
        order = np.argsort(knots)
        knots = knots[order]
        values = values[order]
        return np.interp(parent_data, knots, values, left=values[0], right=values[-1])
    elif mechanism_type == "interaction":
        interaction_parent = mechanism.get("interaction_parent") or edge.interaction_parent
        if interaction_parent and interaction_parent in all_data:
            other = all_data[interaction_parent]
            other_std = np.std(other)
            if other_std > 0:
                other_normalized = (other - np.mean(other)) / other_std
            else:
                other_normalized = other - np.mean(other)
            return parent_data * other_normalized
        return parent_data
    else:
        return parent_data


def _sample_node_noise(
    rng: np.random.Generator,
    noise_model: dict,
    all_data: dict[str, np.ndarray],
    n_samples: int,
) -> np.ndarray:
    """Sample node residual noise from a structured noise model."""
    noise_type = noise_model.get("type", "homoscedastic")

    if noise_type == "homoscedastic":
        scale = float(noise_model.get("scale", noise_model.get("base_scale", 0.5)))
        return rng.normal(0, scale, size=n_samples)

    if noise_type == "heteroscedastic":
        driver = noise_model.get("driver")
        base_scale = float(noise_model.get("base_scale", noise_model.get("scale", 0.5)))
        if not driver or driver not in all_data:
            return rng.normal(0, base_scale, size=n_samples)

        driver_data = np.asarray(all_data[driver], dtype=float)
        driver_std = np.std(driver_data)
        if driver_std > 0:
            driver_normalized = (driver_data - np.mean(driver_data)) / driver_std
        else:
            driver_normalized = driver_data - np.mean(driver_data)

        low_multiplier = float(noise_model.get("low_multiplier", 0.75))
        high_multiplier = float(noise_model.get("high_multiplier", 1.5))
        blend = expit(driver_normalized)
        scales = base_scale * (low_multiplier + (high_multiplier - low_multiplier) * blend)
        return rng.normal(0, scales, size=n_samples)

    raise ValueError(f"Unknown noise model type: {noise_type}")


def _transform_to_distribution(
    rng: np.random.Generator,
    latent: np.ndarray,
    feature: dict,
) -> np.ndarray:
    """Transform latent values to match the feature's target distribution."""
    dist = feature["distribution"]
    params = feature["params"]

    # Use rank-based transformation to preserve ordering while matching
    # distribution.  ``rankdata`` with method='average' handles ties
    # correctly by assigning the mean rank to tied values, avoiding the
    # artificial noise that argsort-based ranking introduces.
    n = len(latent)
    ranks = rankdata(latent, method="average")  # 1-based
    quantiles = (ranks - 0.5) / n

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
    n_noise = int(round(n_informative * noise_ratio))
    if n_noise < 1:
        return result
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
