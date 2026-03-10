"""Procedural feature generation from seed.

Generates feature names, types, distributions, and parameters entirely from
the seed. Feature names are constructed from phonetic building blocks — NOT
drawn from fixed word banks — making the space of possible names effectively
infinite and unpredictable without the seed.
"""

from __future__ import annotations

import numpy as np


# Phonetic building blocks for name generation.
# These are sub-word fragments, not meaningful words. They combine into
# pronounceable but opaque identifiers like "vel_kortan" or "drimex_ploth".
# With ~40 syllables and names of 2-4 syllables, the space is 40^4 = 2.5M+
# possible roots alone, before considering the full combinatorial explosion.
_ONSET = [
    "b", "br", "cr", "d", "dr", "f", "fl", "fr", "g", "gl", "gr",
    "h", "j", "k", "kl", "kr", "l", "m", "n", "p", "pl", "pr",
    "qu", "r", "s", "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st",
    "str", "sw", "t", "tr", "v", "w", "z",
]
_NUCLEUS = ["a", "e", "i", "o", "u", "ai", "au", "ei", "ou", "ea", "oa", "io"]
_CODA = ["", "b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "x", "z", "nd", "nt", "lk", "rm", "rn", "st", "th"]

# Structural patterns for how feature names are formatted
_NAME_PATTERNS = [
    "{root}",                  # "velkor"
    "{root}_{suffix}",        # "velkor_3"  (suffix is a short token)
    "{prefix}_{root}",        # "x_velkor"
    "{prefix}_{root}_{suffix}",  # "x_velkor_3"
]

# Short prefix/suffix tokens (single chars or tiny abbreviations, not meaningful)
_PREFIX_TOKENS = list("abcdefghijklmnopqrstuvwxyz") + [
    "a0", "b1", "c2", "d3", "q1", "q2", "v0", "v1", "x0", "x1", "z0",
]
_SUFFIX_TOKENS = [str(i) for i in range(100)] + [
    "a", "b", "c", "d", "e", "x", "y", "z", "n", "m",
]

# Distribution families for continuous features
CONTINUOUS_DISTRIBUTIONS = ["normal", "lognormal", "uniform", "exponential", "beta"]


def generate_features(
    rng: np.random.Generator,
    template: dict,
) -> tuple[list[dict], dict]:
    """Procedurally generate feature specifications from seed + template.

    Feature names are constructed from phonetic syllable blocks, producing
    opaque identifiers like "brondek_z0" or "q2_floimark". The space of
    possible names is effectively infinite.

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


def _generate_syllable(rng: np.random.Generator) -> str:
    """Generate a single pronounceable syllable."""
    onset = _ONSET[int(rng.integers(0, len(_ONSET)))]
    nucleus = _NUCLEUS[int(rng.integers(0, len(_NUCLEUS)))]
    coda = _CODA[int(rng.integers(0, len(_CODA)))]
    return onset + nucleus + coda


def _generate_root(rng: np.random.Generator) -> str:
    """Generate a pronounceable root word from 2-3 syllables."""
    n_syllables = int(rng.integers(2, 4))  # 2 or 3 syllables
    return "".join(_generate_syllable(rng) for _ in range(n_syllables))


def _generate_unique_name(
    rng: np.random.Generator,
    used_names: set[str],
) -> str:
    """Generate a unique opaque feature name from phonetic building blocks."""
    while True:
        candidate = _format_name_candidate(rng)
        if candidate != "target" and candidate not in used_names:
            used_names.add(candidate)
            return candidate


def _format_name_candidate(rng: np.random.Generator) -> str:
    """Assemble a pronounceable-but-opaque identifier."""
    pattern = _NAME_PATTERNS[int(rng.integers(0, len(_NAME_PATTERNS)))]
    return pattern.format(
        root=_generate_root(rng),
        prefix=_PREFIX_TOKENS[int(rng.integers(0, len(_PREFIX_TOKENS)))],
        suffix=_SUFFIX_TOKENS[int(rng.integers(0, len(_SUFFIX_TOKENS)))],
    )


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
