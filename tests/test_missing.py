"""Tests for missing-value injection mechanisms."""

import numpy as np
import pandas as pd

from tabular_bank.generation.missing import inject_missing


def _base_df(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )


def test_mcar_masks_features_not_target():
    df = _base_df()
    out = inject_missing(np.random.default_rng(1), df, target_col="target", rate=0.2, mechanism="MCAR")
    assert out["target"].isna().sum() == 0
    frac_missing = out[["a", "b"]].isna().mean().mean()
    assert 0.14 <= frac_missing <= 0.26


def test_mar_increases_missing_on_high_driver_values():
    n = 3000
    x = np.linspace(-3.0, 3.0, n)
    df = pd.DataFrame({"a": x, "b": np.random.default_rng(2).normal(size=n), "target": np.zeros(n)})
    out = inject_missing(np.random.default_rng(3), df, target_col="target", rate=0.2, mechanism="MAR")
    high = out.loc[df["a"] >= df["a"].median(), "b"].isna().mean()
    low = out.loc[df["a"] < df["a"].median(), "b"].isna().mean()
    assert high > low
    assert out["target"].isna().sum() == 0


def test_mnar_increases_missing_for_high_values():
    n = 4000
    x = np.linspace(-2.0, 2.0, n)
    df = pd.DataFrame({"a": x, "b": np.random.default_rng(4).normal(size=n), "target": np.zeros(n)})
    out = inject_missing(np.random.default_rng(5), df, target_col="target", rate=0.15, mechanism="MNAR")
    high = out.loc[df["a"] >= np.quantile(df["a"], 0.8), "a"].isna().mean()
    low = out.loc[df["a"] <= np.quantile(df["a"], 0.2), "a"].isna().mean()
    assert high > low
    assert out["target"].isna().sum() == 0
