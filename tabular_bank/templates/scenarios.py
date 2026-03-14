"""Scenario sampling — parametric scenario space for tabular benchmark generation.

Scenarios are sampled from a continuous parameter space (CausalProfiler-inspired)
rather than fixed hand-crafted templates. Any valid configuration has non-zero
probability of being generated, giving coverage guarantees over the space of
tabular ML problems.

All feature specs, DAG structures, distributions, and data are procedurally
generated from the seed. Column names themselves use simple dataset-local
identifiers like ``f_0`` and ``f_1``.
"""

from __future__ import annotations


# Difficulty presets control noise, mechanism richness, and DAG density
DIFFICULTY_PRESETS: dict[str, dict] = {
    "easy": {
        "noise_scale": 0.3,
        "nonlinear_prob": 0.1,       # 10% of edges use nonlinear mechanisms
        "interaction_prob": 0.05,    # 5% chance of interaction mechanisms
        "edge_density": 0.3,         # Sparse DAG
        "max_parents": 3,
        "heteroscedastic_prob": 0.05,
        # Confounder settings
        "n_confounders": 1,
        "confounder_strength": 0.2,
        # Temporal autocorrelation settings
        "temporal_prob": 0.1,        # 10% of root nodes get AR(1) structure
        "max_autocorr": 0.5,
        # Root feature correlations
        "root_correlation_strength": 0.1,
    },
    "medium": {
        "noise_scale": 0.5,
        "nonlinear_prob": 0.3,
        "interaction_prob": 0.15,
        "edge_density": 0.4,
        "max_parents": 4,
        "heteroscedastic_prob": 0.15,
        # Confounder settings
        "n_confounders": 2,
        "confounder_strength": 0.4,
        # Temporal autocorrelation settings
        "temporal_prob": 0.2,        # 20% of root nodes get AR(1) structure
        "max_autocorr": 0.7,
        # Root feature correlations
        "root_correlation_strength": 0.3,
    },
    "hard": {
        "noise_scale": 0.7,
        "nonlinear_prob": 0.5,
        "interaction_prob": 0.25,
        "edge_density": 0.5,
        "max_parents": 5,
        "heteroscedastic_prob": 0.3,
        # Confounder settings
        "n_confounders": 3,
        "confounder_strength": 0.6,
        # Temporal autocorrelation settings
        "temporal_prob": 0.3,        # 30% of root nodes get AR(1) structure
        "max_autocorr": 0.9,
        # Root feature correlations
        "root_correlation_strength": 0.5,
    },
}


def get_difficulty_preset(difficulty: str) -> dict:
    """Get difficulty parameters.

    Accepts either a named preset ("easy", "medium", "hard") or a raw dict
    of difficulty parameters (as produced by ``sample_scenario``).
    """
    if isinstance(difficulty, dict):
        return difficulty
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"Unknown difficulty: {difficulty}. Use: {list(DIFFICULTY_PRESETS)}")
    return DIFFICULTY_PRESETS[difficulty]


# ---------------------------------------------------------------------------
# Parametric scenario sampling (CausalProfiler-inspired)
# ---------------------------------------------------------------------------

# Continuous ranges defining the full "Space of Interest" for scenario params.
# Any valid configuration has non-zero probability of being sampled.
SCENARIO_SPACE = {
    "problem_type_weights": {"binary": 0.45, "multiclass": 0.25, "regression": 0.3},
    "n_features_range": (5, 30),
    "n_samples_range": (1000, 15000),
    "n_classes_range": (3, 8),
    "categorical_ratio_range": (0.1, 0.6),
    "imbalance_ratio_range": (0.05, 0.5),
    "noise_feature_ratio_range": (0.0, 0.35),
    "missing_rate_range": (0.0, 0.15),
    "missing_mechanisms": ["MCAR", "MAR", "MNAR"],
    # Difficulty parameters are sampled independently, not bundled
    "noise_scale_range": (0.1, 1.0),
    "nonlinear_prob_range": (0.05, 0.6),
    "interaction_prob_range": (0.0, 0.3),
    "heteroscedastic_prob_range": (0.0, 0.45),
    "edge_density_range": (0.45, 0.65),
    "max_parents_range": (2, 6),
    "n_confounders_range": (0, 4),
    "confounder_strength_range": (0.1, 0.7),
    "temporal_prob_range": (0.0, 0.4),
    "max_autocorr_range": (0.3, 0.95),
    "root_correlation_strength_range": (0.0, 0.8),
}


def sample_scenario(
    rng,
    scenario_id: str = "sampled",
    scenario_space: dict | None = None,
) -> dict:
    """Sample a scenario template from the continuous parameter space.

    Instead of picking from the 5 fixed presets, draw all parameters
    from their defined ranges.  This gives coverage guarantees: any
    valid configuration has non-zero probability of appearing.

    Args:
        rng: NumPy random Generator.
        scenario_id: Identifier string for the generated scenario.
        scenario_space: Optional overrides merged on top of
            :data:`SCENARIO_SPACE`.  Only the keys you provide are
            changed; everything else keeps its default.

    Returns:
        A scenario dict compatible with the fixed-template format.
    """
    sp = {**SCENARIO_SPACE, **(scenario_space or {})}

    # Problem type
    types = list(sp["problem_type_weights"].keys())
    weights = list(sp["problem_type_weights"].values())
    problem_type = str(rng.choice(types, p=weights))

    lo, hi = sp["n_features_range"]
    if hi - lo < 3:
        n_feat_lo, n_feat_hi = lo, hi
    else:
        n_feat_lo = int(rng.integers(lo, hi - 2))
        n_feat_hi = int(rng.integers(n_feat_lo + 2, hi + 1))

    lo, hi = sp["n_samples_range"]
    if hi - lo < 500:
        n_samp_lo, n_samp_hi = lo, hi
    else:
        mid = max(lo + 1, hi // 2)
        n_samp_lo = int(rng.integers(lo, mid))
        n_samp_hi = int(rng.integers(n_samp_lo + 500, hi + 1))

    cat_ratio = float(rng.uniform(*sp["categorical_ratio_range"]))

    scenario: dict = {
        "id": scenario_id,
        "domain": "sampled",
        "problem_type": problem_type,
        "n_features_range": (n_feat_lo, n_feat_hi),
        "n_samples_range": (n_samp_lo, n_samp_hi),
        "categorical_ratio": cat_ratio,
        "noise_feature_ratio": float(rng.uniform(*sp["noise_feature_ratio_range"])),
        "missing_rate": float(rng.uniform(*sp["missing_rate_range"])),
        "missing_mechanism": str(rng.choice(sp["missing_mechanisms"])),
    }

    if problem_type in ("binary", "multiclass"):
        scenario["n_classes"] = (
            2 if problem_type == "binary"
            else int(rng.integers(sp["n_classes_range"][0], sp["n_classes_range"][1] + 1))
        )
    if problem_type == "binary":
        scenario["imbalance_ratio"] = float(
            rng.uniform(*sp["imbalance_ratio_range"])
        )

    # Sample difficulty parameters independently
    scenario["difficulty"] = {
        "noise_scale": float(rng.uniform(*sp["noise_scale_range"])),
        "nonlinear_prob": float(rng.uniform(*sp["nonlinear_prob_range"])),
        "interaction_prob": float(rng.uniform(*sp["interaction_prob_range"])),
        "heteroscedastic_prob": float(rng.uniform(*sp["heteroscedastic_prob_range"])),
        "edge_density": float(rng.uniform(*sp["edge_density_range"])),
        "max_parents": int(
            rng.integers(sp["max_parents_range"][0], sp["max_parents_range"][1] + 1)
        ),
        "n_confounders": int(
            rng.integers(sp["n_confounders_range"][0], sp["n_confounders_range"][1] + 1)
        ),
        "confounder_strength": float(rng.uniform(*sp["confounder_strength_range"])),
        "temporal_prob": float(rng.uniform(*sp["temporal_prob_range"])),
        "max_autocorr": float(rng.uniform(*sp["max_autocorr_range"])),
        "root_correlation_strength": float(
            rng.uniform(*sp["root_correlation_strength_range"])
        ),
    }

    return scenario
