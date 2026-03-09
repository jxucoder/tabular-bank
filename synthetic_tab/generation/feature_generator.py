"""Procedural feature generation from seed.

Generates feature names, types, distributions, and parameters entirely from
the seed. The domain vocabulary pools are intentionally large and generic —
they provide plausible-sounding names but the specific combination selected
is unpredictable without the seed.
"""

from __future__ import annotations

import numpy as np


# Domain-specific vocabulary pools for feature name generation.
# These are large enough that the specific combination is unpredictable.
DOMAIN_VOCABULARIES: dict[str, dict] = {
    "commercial": {
        "prefixes": [
            "monthly", "annual", "total", "avg", "max", "min", "net", "gross",
            "daily", "weekly", "cumulative", "recent", "lifetime", "current",
            "previous", "projected", "actual", "adjusted", "weighted", "rolling",
        ],
        "roots": [
            "spend", "revenue", "transaction", "purchase", "order", "visit",
            "session", "click", "conversion", "engagement", "retention",
            "acquisition", "churn", "renewal", "subscription", "usage",
            "activity", "interaction", "feedback", "complaint", "inquiry",
            "referral", "discount", "promotion", "campaign", "channel",
            "segment", "tier", "loyalty", "satisfaction", "rating",
        ],
        "suffixes": [
            "count", "amount", "rate", "ratio", "score", "index",
            "duration", "frequency", "value", "pct", "rank", "level",
            "flag", "category", "type", "class", "group", "band",
        ],
        "target_names": [
            "outcome", "result", "status", "response", "label",
            "target", "indicator", "signal", "event", "flag",
        ],
        "category_pools": [
            ["low", "medium", "high"],
            ["basic", "standard", "premium"],
            ["active", "inactive"],
            ["yes", "no"],
            ["type_a", "type_b", "type_c"],
            ["small", "medium", "large", "enterprise"],
            ["new", "returning", "loyal"],
            ["online", "offline", "hybrid"],
            ["direct", "indirect", "partner"],
            ["tier_1", "tier_2", "tier_3", "tier_4"],
        ],
    },
    "healthcare": {
        "prefixes": [
            "baseline", "peak", "avg", "total", "fasting", "resting",
            "systolic", "diastolic", "pre", "post", "initial", "follow_up",
            "normalized", "adjusted", "measured", "reported", "estimated",
            "clinical", "lab", "vital",
        ],
        "roots": [
            "pressure", "glucose", "cholesterol", "hemoglobin", "platelet",
            "bmi", "heart_rate", "temperature", "oxygen", "creatinine",
            "albumin", "bilirubin", "sodium", "potassium", "calcium",
            "white_cell", "red_cell", "enzyme", "hormone", "protein",
            "antibody", "marker", "dose", "treatment", "therapy",
            "symptom", "diagnosis", "procedure", "visit", "episode",
        ],
        "suffixes": [
            "level", "count", "ratio", "score", "index", "value",
            "measurement", "reading", "result", "grade", "stage",
            "rate", "frequency", "duration", "concentration", "volume",
        ],
        "target_names": [
            "condition", "diagnosis", "outcome", "prognosis", "risk_level",
            "severity", "classification", "category", "status", "assessment",
        ],
        "category_pools": [
            ["normal", "elevated", "high", "critical"],
            ["negative", "positive"],
            ["stage_1", "stage_2", "stage_3", "stage_4"],
            ["mild", "moderate", "severe"],
            ["type_i", "type_ii", "type_iii"],
            ["low_risk", "medium_risk", "high_risk"],
            ["remission", "stable", "progressing"],
            ["acute", "chronic"],
            ["inpatient", "outpatient"],
            ["male", "female", "other"],
        ],
    },
    "real_estate": {
        "prefixes": [
            "total", "avg", "median", "estimated", "assessed", "listed",
            "actual", "adjusted", "gross", "net", "annual", "monthly",
            "nearby", "local", "regional", "current", "historical",
            "projected", "comparative", "weighted",
        ],
        "roots": [
            "area", "room", "floor", "bathroom", "bedroom", "garage",
            "lot", "frontage", "depth", "elevation", "distance",
            "price", "tax", "assessment", "valuation", "rent",
            "income", "expense", "maintenance", "renovation", "age",
            "quality", "condition", "amenity", "view", "noise",
            "school", "transit", "crime", "population", "density",
        ],
        "suffixes": [
            "sqft", "count", "score", "index", "rating", "value",
            "amount", "ratio", "pct", "rank", "grade", "level",
            "distance", "years", "months", "class", "type", "zone",
        ],
        "target_names": [
            "price", "valuation", "assessment", "estimate", "appraisal",
            "value", "amount", "cost", "worth", "figure",
        ],
        "category_pools": [
            ["residential", "commercial", "mixed"],
            ["excellent", "good", "fair", "poor"],
            ["urban", "suburban", "rural"],
            ["zone_a", "zone_b", "zone_c", "zone_d"],
            ["detached", "semi", "townhouse", "condo"],
            ["new", "renovated", "original"],
            ["north", "south", "east", "west"],
            ["low", "medium", "high"],
            ["yes", "no"],
            ["brick", "wood", "concrete", "steel"],
        ],
    },
    "financial": {
        "prefixes": [
            "current", "previous", "avg", "total", "outstanding",
            "available", "minimum", "maximum", "monthly", "annual",
            "cumulative", "rolling", "weighted", "adjusted", "gross",
            "net", "recent", "historical", "projected", "reported",
        ],
        "roots": [
            "balance", "payment", "credit", "debit", "interest",
            "principal", "installment", "fee", "penalty", "income",
            "expense", "savings", "investment", "liability", "asset",
            "equity", "debt", "limit", "utilization", "score",
            "history", "inquiry", "account", "transaction", "transfer",
            "deposit", "withdrawal", "return", "volatility", "exposure",
        ],
        "suffixes": [
            "amount", "count", "ratio", "rate", "score", "pct",
            "value", "duration", "frequency", "index", "rank",
            "level", "grade", "band", "class", "category", "type",
        ],
        "target_names": [
            "default", "risk", "outcome", "status", "decision",
            "result", "classification", "indicator", "event", "flag",
        ],
        "category_pools": [
            ["approved", "denied"],
            ["low", "medium", "high"],
            ["current", "delinquent", "default"],
            ["fixed", "variable"],
            ["secured", "unsecured"],
            ["short_term", "medium_term", "long_term"],
            ["grade_a", "grade_b", "grade_c", "grade_d"],
            ["individual", "joint"],
            ["primary", "secondary"],
            ["employed", "self_employed", "unemployed", "retired"],
        ],
    },
    "hr": {
        "prefixes": [
            "current", "previous", "avg", "total", "annual", "monthly",
            "cumulative", "recent", "initial", "adjusted", "normalized",
            "reported", "measured", "estimated", "weighted", "rolling",
            "baseline", "peak", "minimum", "maximum",
        ],
        "roots": [
            "salary", "bonus", "compensation", "tenure", "experience",
            "performance", "satisfaction", "engagement", "absence",
            "overtime", "training", "certification", "promotion",
            "review", "feedback", "complaint", "project", "team",
            "workload", "commute", "travel", "relocation", "benefit",
            "stock", "option", "leave", "sick_day", "vacation",
            "mentoring", "collaboration",
        ],
        "suffixes": [
            "score", "count", "hours", "days", "months", "years",
            "amount", "ratio", "rate", "pct", "index", "rank",
            "level", "grade", "band", "category", "type", "class",
        ],
        "target_names": [
            "attrition", "departure", "turnover", "retention", "outcome",
            "status", "decision", "result", "event", "indicator",
        ],
        "category_pools": [
            ["junior", "mid", "senior", "lead"],
            ["full_time", "part_time", "contract"],
            ["on_site", "remote", "hybrid"],
            ["engineering", "sales", "marketing", "operations", "hr"],
            ["high", "medium", "low"],
            ["exceeds", "meets", "below"],
            ["yes", "no"],
            ["male", "female", "non_binary"],
            ["single", "married", "divorced"],
            ["bachelors", "masters", "phd", "other"],
        ],
    },
}

# Distribution families for continuous features
CONTINUOUS_DISTRIBUTIONS = ["normal", "lognormal", "uniform", "exponential", "beta"]


def generate_features(
    rng: np.random.Generator,
    template: dict,
) -> tuple[list[dict], dict]:
    """Procedurally generate feature specifications from seed + template.

    Returns:
        features: List of feature spec dicts (name, type, distribution info)
        target: Target feature spec dict
    """
    domain = template["domain"]
    vocab = DOMAIN_VOCABULARIES[domain]

    n_features = rng.integers(
        template["n_features_range"][0],
        template["n_features_range"][1] + 1,
    )
    n_categorical = max(1, int(n_features * template["categorical_ratio"]))
    n_continuous = n_features - n_categorical

    # Shuffle vocabularies
    prefixes = list(vocab["prefixes"])
    roots = list(vocab["roots"])
    suffixes = list(vocab["suffixes"])
    rng.shuffle(prefixes)
    rng.shuffle(roots)
    rng.shuffle(suffixes)

    features = []
    used_names: set[str] = set()

    # Generate continuous features
    for i in range(n_continuous):
        name = _generate_unique_name(rng, prefixes, roots, suffixes, used_names)
        dist = rng.choice(CONTINUOUS_DISTRIBUTIONS)
        params = _sample_distribution_params(rng, dist)
        features.append({
            "name": name,
            "type": "continuous",
            "distribution": dist,
            "params": params,
        })

    # Generate categorical features
    category_pools = list(vocab["category_pools"])
    rng.shuffle(category_pools)
    for i in range(n_categorical):
        name = _generate_unique_name(rng, prefixes, roots, suffixes, used_names)
        pool = category_pools[i % len(category_pools)]
        # Randomly select a subset or all categories
        n_cats = rng.integers(2, min(len(pool) + 1, 7))
        pool_indices = rng.choice(len(pool), size=min(n_cats, len(pool)), replace=False)
        categories = [pool[j] for j in sorted(pool_indices)]
        # Generate random probabilities
        raw_probs = rng.dirichlet(np.ones(len(categories)))
        features.append({
            "name": name,
            "type": "categorical",
            "categories": categories,
            "probs": raw_probs.tolist(),
        })

    # Shuffle feature order
    perm = rng.permutation(len(features))
    features = [features[i] for i in perm]

    # Generate target
    target_names = list(vocab["target_names"])
    rng.shuffle(target_names)
    target_name = target_names[0]

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
    prefixes: list[str],
    roots: list[str],
    suffixes: list[str],
    used_names: set[str],
) -> str:
    """Generate a unique feature name by combining vocabulary parts."""
    for _ in range(100):  # Avoid infinite loop
        parts = []
        # Sometimes skip prefix (40% chance)
        if rng.random() > 0.4:
            parts.append(prefixes[rng.integers(0, len(prefixes))])
        parts.append(roots[rng.integers(0, len(roots))])
        # Sometimes add suffix (60% chance)
        if rng.random() > 0.4:
            parts.append(suffixes[rng.integers(0, len(suffixes))])
        name = "_".join(parts)
        if name not in used_names:
            used_names.add(name)
            return name
    # Fallback: append a number
    base = roots[rng.integers(0, len(roots))]
    i = 0
    while f"{base}_{i}" in used_names:
        i += 1
    name = f"{base}_{i}"
    used_names.add(name)
    return name


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
