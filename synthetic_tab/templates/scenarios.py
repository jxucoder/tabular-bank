"""Scenario templates — minimal problem archetypes.

These templates contain ZERO dataset-specific information. They only define
high-level constraints: problem type, approximate feature count, difficulty,
and categorical ratio. The 'domain' field is purely metadata for human reference
and does NOT affect generation. All features, DAG structures, distributions,
and data — including feature names — are procedurally generated from the seed
using opaque phonetic identifiers.
"""

from __future__ import annotations

SCENARIOS: list[dict] = [
    {
        "id": "scenario_0",
        "domain": "commercial",
        "problem_type": "binary",
        "n_features_range": (8, 15),
        "n_samples_range": (3000, 8000),
        "n_classes": 2,
        "difficulty": "medium",
        "categorical_ratio": 0.3,
    },
    {
        "id": "scenario_1",
        "domain": "healthcare",
        "problem_type": "multiclass",
        "n_features_range": (10, 18),
        "n_samples_range": (2000, 6000),
        "n_classes": 4,
        "difficulty": "hard",
        "categorical_ratio": 0.4,
    },
    {
        "id": "scenario_2",
        "domain": "real_estate",
        "problem_type": "regression",
        "n_features_range": (10, 16),
        "n_samples_range": (4000, 10000),
        "difficulty": "medium",
        "categorical_ratio": 0.25,
    },
    {
        "id": "scenario_3",
        "domain": "financial",
        "problem_type": "binary",
        "n_features_range": (7, 13),
        "n_samples_range": (5000, 12000),
        "n_classes": 2,
        "difficulty": "easy",
        "categorical_ratio": 0.35,
    },
    {
        "id": "scenario_4",
        "domain": "hr",
        "problem_type": "binary",
        "n_features_range": (8, 14),
        "n_samples_range": (3000, 7000),
        "n_classes": 2,
        "difficulty": "hard",
        "categorical_ratio": 0.45,
    },
]


# Difficulty presets control noise, nonlinearity, and DAG density
DIFFICULTY_PRESETS: dict[str, dict] = {
    "easy": {
        "noise_scale": 0.3,
        "nonlinear_prob": 0.1,       # 10% of edges use nonlinear functions
        "interaction_prob": 0.05,    # 5% chance of interaction terms
        "edge_density": 0.3,         # Sparse DAG
        "max_parents": 3,
    },
    "medium": {
        "noise_scale": 0.5,
        "nonlinear_prob": 0.3,
        "interaction_prob": 0.15,
        "edge_density": 0.4,
        "max_parents": 4,
    },
    "hard": {
        "noise_scale": 0.7,
        "nonlinear_prob": 0.5,
        "interaction_prob": 0.25,
        "edge_density": 0.5,
        "max_parents": 5,
    },
}


def get_scenario(index: int) -> dict:
    """Get a scenario template by index."""
    if index < 0 or index >= len(SCENARIOS):
        raise ValueError(f"Scenario index {index} out of range [0, {len(SCENARIOS)})")
    return SCENARIOS[index]


def get_difficulty_preset(difficulty: str) -> dict:
    """Get difficulty parameters."""
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"Unknown difficulty: {difficulty}. Use: {list(DIFFICULTY_PRESETS)}")
    return DIFFICULTY_PRESETS[difficulty]
