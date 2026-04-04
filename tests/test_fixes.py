"""Tests for code review fixes: vectorized AR(1), DAG validation, metric warnings, secret permissions, MAR fallback."""

import os
import stat
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from tabular_bank.generation.dag_builder import DAGSpec, DAGValidationError, Edge, build_dag
from tabular_bank.generation.missing import inject_missing
from tabular_bank.generation.sampler import _apply_autocorr, sample_dataset
from tabular_bank.generation.seed import get_master_secret
from tabular_bank.runner import _evaluate_metric


# ---------------------------------------------------------------------------
# Fix 1: Vectorized AR(1) produces identical results to the naive loop
# ---------------------------------------------------------------------------

def _apply_autocorr_naive(raw: np.ndarray, rho: float) -> np.ndarray:
    """Reference (naive loop) implementation for correctness comparison."""
    rho = float(np.clip(rho, -0.999, 0.999))
    out = np.empty_like(raw)
    scale = np.sqrt(1.0 - rho ** 2)
    out[0] = raw[0]
    for t in range(1, len(raw)):
        out[t] = rho * out[t - 1] + scale * raw[t]
    return out


def test_vectorized_autocorr_matches_naive():
    """Vectorized lfilter AR(1) must match the naive Python loop."""
    rng = np.random.default_rng(42)
    raw = rng.normal(size=5000)
    for rho in [0.0, 0.3, 0.7, 0.95, -0.5]:
        result_vec = _apply_autocorr(raw, rho)
        result_naive = _apply_autocorr_naive(raw, rho)
        np.testing.assert_allclose(result_vec, result_naive, atol=1e-10,
                                   err_msg=f"Mismatch at rho={rho}")


def test_vectorized_autocorr_preserves_unit_variance():
    """AR(1) output should have approximately unit variance for large samples."""
    rng = np.random.default_rng(99)
    raw = rng.normal(size=50_000)
    out = _apply_autocorr(raw, 0.8)
    assert 0.85 < np.var(out) < 1.15


def test_autocorr_large_dataset_runs_fast():
    """Vectorized AR(1) should handle 1M samples without issue."""
    rng = np.random.default_rng(7)
    raw = rng.normal(size=1_000_000)
    import time
    t0 = time.time()
    _apply_autocorr(raw, 0.7)
    elapsed = time.time() - t0
    # Should complete in well under 1 second (naive loop takes ~2-5s)
    assert elapsed < 1.0, f"AR(1) on 1M samples took {elapsed:.2f}s — too slow"


# ---------------------------------------------------------------------------
# Fix 2: DAG validation raises on critical structural issues
# ---------------------------------------------------------------------------

def test_dag_validation_raises_on_all_roots():
    """A DAG where every observed node is a root should raise DAGValidationError."""
    dag = DAGSpec(
        nodes=["a", "b", "c", "target"],
        target="target",
        root_nodes=["a", "b", "c"],
        edges=[Edge(parent="a", child="target", coefficient=1.0, mechanism={"type": "linear"})],
        noise_scales={"a": 0.1, "b": 0.1, "c": 0.1, "target": 0.1},
    )
    from tabular_bank.generation.dag_builder import _validate_dag_stats
    with pytest.raises(DAGValidationError, match="no causal structure"):
        _validate_dag_stats(dag)


def test_dag_validation_warns_on_soft_deviations():
    """Unusual but not degenerate graphs should warn, not raise."""
    # Graph where all non-root nodes have only 1 parent (mean_in_degree=1.0,
    # right at the boundary — should at most warn, not raise).
    dag = DAGSpec(
        nodes=["a", "b", "c", "target"],
        target="target",
        root_nodes=["a"],
        edges=[
            Edge(parent="a", child="b", coefficient=1.0, mechanism={"type": "linear"}),
            Edge(parent="b", child="c", coefficient=1.0, mechanism={"type": "linear"}),
            Edge(parent="c", child="target", coefficient=1.0, mechanism={"type": "linear"}),
        ],
        noise_scales={"a": 0.1, "b": 0.1, "c": 0.1, "target": 0.1},
    )
    from tabular_bank.generation.dag_builder import _validate_dag_stats
    # Should not raise (mean_in_degree=1.0 is within critical range)
    _validate_dag_stats(dag)


# ---------------------------------------------------------------------------
# Fix 3: Metric fallback warnings
# ---------------------------------------------------------------------------

def test_roc_auc_single_class_warns():
    """ROC-AUC fallback to accuracy should emit a warning."""

    class ConstantPredictor:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    X_test = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y_test = pd.Series([0, 0, 0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        score = _evaluate_metric(ConstantPredictor(), X_test, y_test, "binary", "roc_auc")
        assert any("single-class test split" in str(warning.message) for warning in w)
    assert isinstance(score, float)


def test_log_loss_fallback_warns():
    """log_loss fallback to accuracy should emit a warning."""

    class PredictOnlyModel:
        def predict(self, X):
            return np.array([0] * len(X))

        def fit(self, X, y):
            pass

    X_test = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y_test = pd.Series([0, 1, 0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        score = _evaluate_metric(PredictOnlyModel(), X_test, y_test, "multiclass", "log_loss")
        assert any("predict_proba" in str(warning.message) for warning in w)
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Fix 4: Secret file permission check
# ---------------------------------------------------------------------------

def test_secret_file_permissive_warns():
    """Reading a world-readable .secret file should emit a permission warning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        secret_file = os.path.join(tmpdir, ".secret")
        with open(secret_file, "w") as f:
            f.write("my-secret-value")
        # Make world-readable
        os.chmod(secret_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            secret = get_master_secret(secret=None, cache_dir=tmpdir)
            assert secret == "my-secret-value"
            assert any("permissive permissions" in str(warning.message) for warning in w)


def test_secret_file_restricted_no_warning():
    """A properly restricted .secret file should not warn."""
    with tempfile.TemporaryDirectory() as tmpdir:
        secret_file = os.path.join(tmpdir, ".secret")
        with open(secret_file, "w") as f:
            f.write("my-secret-value")
        os.chmod(secret_file, stat.S_IRUSR | stat.S_IWUSR)  # 600

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            secret = get_master_secret(secret=None, cache_dir=tmpdir)
            assert secret == "my-secret-value"
            perm_warnings = [x for x in w if "permissive permissions" in str(x.message)]
            assert len(perm_warnings) == 0


# ---------------------------------------------------------------------------
# Fix 5: MAR fallback warning when driver column is all-NaN
# ---------------------------------------------------------------------------

def test_mar_all_nan_driver_warns():
    """MAR should warn when a driver column is all-NaN and fall back to MCAR."""
    df = pd.DataFrame({
        "a": [np.nan] * 100,
        "b": np.random.default_rng(0).normal(size=100),
        "target": np.zeros(100),
    })

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        inject_missing(np.random.default_rng(1), df, target_col="target", rate=0.2, mechanism="MAR")
        # At least one warning about falling back to MCAR should appear
        assert any("falling back to MCAR" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# Integration: full pipeline still works with all fixes applied
# ---------------------------------------------------------------------------

def test_full_pipeline_with_autocorr_still_works():
    """End-to-end: dataset with AR(1) autocorrelation generates correctly."""
    rng = np.random.default_rng(0)
    features = [
        {
            "name": "root_feature",
            "type": "continuous",
            "distribution": "exponential",
            "params": {"scale": 2.0},
        }
    ]
    target = {"name": "target", "type": "continuous", "problem_type": "regression"}
    dag = DAGSpec(
        nodes=["root_feature", "target"],
        target="target",
        root_nodes=["root_feature"],
        edges=[Edge(parent="root_feature", child="target", form="linear", coefficient=1.0)],
        noise_scales={"root_feature": 0.1, "target": 0.1},
        autocorr={"root_feature": 0.85},
    )

    df = sample_dataset(rng, dag, features, target, n_samples=1000)
    assert len(df) == 1000
    assert (df["root_feature"] >= 0).all()
    assert np.isfinite(df["target"]).all()
