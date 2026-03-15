"""Missingness injection for generated datasets.

Implements the three standard missing-data mechanisms (Rubin, 1976):

  - MCAR: Missing Completely At Random — each entry is independently dropped
    with probability ``rate``.
  - MAR:  Missing At Random — missingness in column A depends on observed
    values in another column B.
  - MNAR: Missing Not At Random — missingness depends on the (unobserved)
    value itself (e.g. high values are more likely to be missing).

All functions operate on a pandas DataFrame and return a copy with NaN
values injected into *feature* columns only — the target is never masked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_missing(
    rng: np.random.Generator,
    df: pd.DataFrame,
    target_col: str,
    rate: float,
    mechanism: str = "MCAR",
) -> pd.DataFrame:
    """Inject missing values into a DataFrame.

    Args:
        rng: NumPy random generator (for reproducibility).
        df: The complete DataFrame.
        target_col: Name of the target column (never masked).
        rate: Overall fraction of feature entries to make missing.
            The actual per-column rate varies by mechanism.
        mechanism: One of "MCAR", "MAR", "MNAR".

    Returns:
        A copy of ``df`` with NaN values injected.
    """
    if rate <= 0:
        return df

    mechanism = mechanism.upper()
    feature_cols = [c for c in df.columns if c != target_col]

    if mechanism == "MCAR":
        return _inject_mcar(rng, df, feature_cols, rate)
    elif mechanism == "MAR":
        return _inject_mar(rng, df, feature_cols, rate)
    elif mechanism == "MNAR":
        return _inject_mnar(rng, df, feature_cols, rate)
    else:
        raise ValueError(f"Unknown missing mechanism: {mechanism}. Use MCAR/MAR/MNAR.")


def _inject_mcar(
    rng: np.random.Generator,
    df: pd.DataFrame,
    feature_cols: list[str],
    rate: float,
) -> pd.DataFrame:
    """Each feature entry is independently dropped with probability ``rate``."""
    df = df.copy()
    for col in feature_cols:
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df


def _inject_mar(
    rng: np.random.Generator,
    df: pd.DataFrame,
    feature_cols: list[str],
    rate: float,
) -> pd.DataFrame:
    """Missingness in each column depends on another column's values.

    For each feature column, pick a random *other* feature as the
    "driver".  Rows where the driver's value is above its median have a
    higher probability of being missing in the target column.
    """
    df = df.copy()
    if len(feature_cols) < 2:
        return _inject_mcar(rng, df, feature_cols, rate)

    for col in feature_cols:
        others = [c for c in feature_cols if c != col]
        driver = str(rng.choice(others))

        driver_vals = pd.to_numeric(df[driver], errors="coerce")
        if driver_vals.isna().all():
            import warnings
            warnings.warn(
                f"MAR driver column '{driver}' is all-NaN — falling back "
                f"to MCAR for column '{col}'. This may produce unexpected "
                f"missingness patterns.",
                stacklevel=2,
            )
            mask = rng.random(len(df)) < rate
        else:
            median = driver_vals.median()
            above = driver_vals >= median
            # Rows above median: 2x the base rate; below: near zero
            probs = np.where(above, min(rate * 2, 1.0), rate * 0.2)
            mask = rng.random(len(df)) < probs

        df.loc[mask, col] = np.nan
    return df


def _inject_mnar(
    rng: np.random.Generator,
    df: pd.DataFrame,
    feature_cols: list[str],
    rate: float,
) -> pd.DataFrame:
    """Missingness depends on the value itself (high values more likely missing)."""
    df = df.copy()
    for col in feature_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().all():
            mask = rng.random(len(df)) < rate
        else:
            # Probability of missing increases with quantile rank
            ranks = vals.rank(pct=True, na_option="keep").fillna(0.5)
            probs = rate * 2 * ranks.values
            probs = np.clip(probs, 0.0, 1.0)
            mask = rng.random(len(df)) < probs

        df.loc[mask, col] = np.nan
    return df
