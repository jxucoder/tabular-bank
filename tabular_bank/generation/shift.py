"""Distribution shift injection for robustness evaluation.

Real-world ML systems encounter distribution shift continuously. TabReD
(ICLR 2025 Spotlight) showed that academic benchmarks diverge from
production data precisely because they lack temporal/distributional drift.

This module injects controlled shift into generated datasets to test model
robustness.  Three shift types are supported:

1. **Covariate shift** — P(X_test) != P(X_train), but P(y|X) unchanged.
   Simulates deployment on a different population.
2. **Concept drift** — P(y|X) changes between train and test.
   Simulates a changing world where the same features mean different things.
3. **Temporal split** — Time-ordered train/test split that exposes both
   covariate shift and concept drift simultaneously (the realistic case).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_covariate_shift(
    df: pd.DataFrame,
    feature_names: list[str],
    target: str,
    shift_fraction: float = 0.3,
    shift_magnitude: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data and shift test features to create covariate shift.

    Selects a random subset of features, shifts their test-set distribution
    by adding a constant offset (mean shift) and scaling variance.  The
    target P(y|X) is unchanged — only X is shifted.

    Args:
        df: Full dataset with features and target.
        feature_names: List of feature column names.
        target: Target column name.
        shift_fraction: Fraction of features to shift.
        shift_magnitude: Multiplier for shift intensity (1.0 = 1 std dev shift).
        seed: Random seed.

    Returns:
        (train_df, test_df) with covariate shift in test_df.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    split_idx = int(n * 0.67)

    # Shuffle before splitting (no temporal structure for pure covariate shift)
    idx = rng.permutation(n)
    train_df = df.iloc[idx[:split_idx]].copy()
    test_df = df.iloc[idx[split_idx:]].copy()

    # Select features to shift
    numeric_features = [f for f in feature_names if pd.api.types.is_numeric_dtype(df[f])]
    n_shift = max(1, int(len(numeric_features) * shift_fraction))
    shift_features = list(rng.choice(numeric_features, size=n_shift, replace=False))

    for feat in shift_features:
        train_std = train_df[feat].std()
        if train_std > 0:
            # Mean shift
            offset = rng.normal(0, shift_magnitude) * train_std
            test_df[feat] = test_df[feat] + offset
            # Variance shift
            scale = rng.uniform(0.7, 1.5)
            train_mean = train_df[feat].mean()
            test_df[feat] = train_mean + (test_df[feat] - train_mean) * scale

    return train_df, test_df


def inject_concept_drift(
    df: pd.DataFrame,
    feature_names: list[str],
    target: str,
    drift_magnitude: float = 0.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data and alter the target function to create concept drift.

    P(y|X) changes between train and test.  For regression, the target gets
    an additive perturbation that depends on feature values.  For
    classification, class boundaries shift.

    Args:
        df: Full dataset with features and target.
        feature_names: List of feature column names.
        target: Target column name.
        drift_magnitude: Strength of concept drift (0 = none, 1 = strong).
        seed: Random seed.

    Returns:
        (train_df, test_df) with concept drift in test_df.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    split_idx = int(n * 0.67)

    idx = rng.permutation(n)
    train_df = df.iloc[idx[:split_idx]].copy()
    test_df = df.iloc[idx[split_idx:]].copy()

    numeric_features = [f for f in feature_names if pd.api.types.is_numeric_dtype(df[f])]
    if not numeric_features:
        return train_df, test_df

    target_dtype = df[target].dtype

    if pd.api.types.is_float_dtype(target_dtype):
        # Regression: add feature-dependent perturbation to target
        # Pick a random feature and add a nonlinear function of it to the test target
        drift_feature = str(rng.choice(numeric_features))
        x = test_df[drift_feature].values.astype(float)
        x_std = np.std(x)
        if x_std > 0:
            x_norm = (x - np.mean(x)) / x_std
        else:
            x_norm = x - np.mean(x)

        target_std = train_df[target].std()
        if target_std > 0:
            perturbation = drift_magnitude * target_std * np.sin(x_norm)
            test_df[target] = test_df[target].values + perturbation
    else:
        # Classification: flip a fraction of labels proportional to drift_magnitude
        flip_prob = drift_magnitude * 0.3  # max 30% flip at drift=1.0
        n_test = len(test_df)
        flip_mask = rng.random(n_test) < flip_prob
        if flip_mask.any():
            classes = sorted(df[target].unique())
            if len(classes) > 1:
                for i in test_df.index[flip_mask]:
                    current = test_df.at[i, target]
                    others = [c for c in classes if c != current]
                    test_df.at[i, target] = rng.choice(others)

    return train_df, test_df


def create_temporal_split(
    df: pd.DataFrame,
    train_fraction: float = 0.67,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by row order to preserve temporal structure.

    If the dataset was generated with AR(1) autocorrelation on root nodes,
    row order reflects temporal ordering.  This split exposes natural
    distribution shift that random CV splits would destroy.

    Args:
        df: Full dataset (row order = temporal order).
        train_fraction: Fraction of data for training.

    Returns:
        (train_df, test_df) where test_df contains later rows.
    """
    split_idx = int(len(df) * train_fraction)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def create_shifted_splits(
    df: pd.DataFrame,
    feature_names: list[str],
    target: str,
    shift_types: list[str] | None = None,
    shift_magnitude: float = 0.5,
    seed: int = 42,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Create multiple train/test splits with different shift types.

    Convenience function that generates all shift variants for a single
    dataset, enabling direct comparison of model robustness across
    shift types.

    Args:
        df: Full dataset.
        feature_names: Feature column names.
        target: Target column name.
        shift_types: Which shifts to generate.  Defaults to all three:
            ["none", "covariate", "concept", "temporal"].
        shift_magnitude: Shift intensity.
        seed: Random seed.

    Returns:
        Dict of shift_type -> (train_df, test_df).
    """
    if shift_types is None:
        shift_types = ["none", "covariate", "concept", "temporal"]

    rng = np.random.default_rng(seed)
    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for shift_type in shift_types:
        if shift_type == "none":
            # Standard random split (no shift)
            n = len(df)
            idx = rng.permutation(n)
            split_idx = int(n * 0.67)
            splits["none"] = (df.iloc[idx[:split_idx]].copy(), df.iloc[idx[split_idx:]].copy())
        elif shift_type == "covariate":
            splits["covariate"] = inject_covariate_shift(
                df, feature_names, target,
                shift_magnitude=shift_magnitude, seed=seed,
            )
        elif shift_type == "concept":
            splits["concept"] = inject_concept_drift(
                df, feature_names, target,
                drift_magnitude=shift_magnitude, seed=seed,
            )
        elif shift_type == "temporal":
            splits["temporal"] = create_temporal_split(df)

    return splits
