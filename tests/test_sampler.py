"""Focused tests for low-level sampling behavior."""

import numpy as np

from tabular_bank.generation.dag_builder import DAGSpec, Edge
from tabular_bank.generation.sampler import sample_dataset


def test_root_autocorr_preserves_positive_distribution_support():
    """Autocorrelation should be applied before transforming to the output support."""
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

    df = sample_dataset(rng, dag, features, target, n_samples=400)

    assert (df["root_feature"] >= 0).all()


def test_root_confounders_are_applied_to_root_nodes():
    """Latent confounders should induce dependence even for root features."""
    rng = np.random.default_rng(1)
    features = [
        {
            "name": "a",
            "type": "continuous",
            "distribution": "normal",
            "params": {"mean": 0.0, "std": 1.0},
        },
        {
            "name": "b",
            "type": "continuous",
            "distribution": "normal",
            "params": {"mean": 0.0, "std": 1.0},
        },
    ]
    target = {"name": "target", "type": "continuous", "problem_type": "regression"}
    dag = DAGSpec(
        nodes=["a", "b", "target"],
        target="target",
        root_nodes=["a", "b"],
        edges=[
            Edge(parent="_latent_0", child="a", form="linear", coefficient=3.0, is_confounder=True),
            Edge(parent="_latent_0", child="b", form="linear", coefficient=3.0, is_confounder=True),
            Edge(parent="a", child="target", form="linear", coefficient=1.0),
        ],
        noise_scales={"a": 0.1, "b": 0.1, "target": 0.1},
        confounders={"_latent_0": ["a", "b"]},
        autocorr={"a": 0.0, "b": 0.0},
    )

    df = sample_dataset(rng, dag, features, target, n_samples=500)

    assert abs(df[["a", "b"]].corr().iloc[0, 1]) > 0.2


def test_edge_legacy_form_fields_are_converted_to_mechanism():
    """Legacy Edge(form=...) construction should populate a structured mechanism."""
    edge = Edge(
        parent="a",
        child="b",
        form="piecewise_linear",
        coefficient=1.0,
        threshold=0.25,
        slope_left=0.1,
        slope_right=1.7,
    )

    assert edge.mechanism["type"] == "piecewise_linear"
    assert edge.mechanism["threshold"] == 0.25
    assert edge.mechanism["slope_left"] == 0.1
    assert edge.mechanism["slope_right"] == 1.7


def test_tanh_mechanism_stays_bounded_without_noise():
    """Tanh mechanisms should produce bounded latent outputs when target noise is zero."""
    rng = np.random.default_rng(2)
    features = [
        {
            "name": "root_feature",
            "type": "continuous",
            "distribution": "normal",
            "params": {"mean": 0.0, "std": 1.0},
        }
    ]
    target = {"name": "target", "type": "continuous", "problem_type": "regression"}
    dag = DAGSpec(
        nodes=["root_feature", "target"],
        target="target",
        root_nodes=["root_feature"],
        edges=[
            Edge(
                parent="root_feature",
                child="target",
                coefficient=1.0,
                mechanism={"type": "tanh", "slope": 1.8, "offset": 0.15},
            )
        ],
        noise_scales={"root_feature": 0.0, "target": 0.0},
        noise_models={
            "root_feature": {"type": "homoscedastic", "scale": 0.0},
            "target": {"type": "homoscedastic", "scale": 0.0},
        },
        autocorr={"root_feature": 0.0},
    )

    df = sample_dataset(rng, dag, features, target, n_samples=400)

    assert (df["target"] <= 1.0 + 1e-8).all()
    assert (df["target"] >= -1.0 - 1e-8).all()


def test_spline_mechanism_produces_finite_nonconstant_values():
    """Spline mechanisms should yield finite, smoothly varying outputs."""
    rng = np.random.default_rng(3)
    features = [
        {
            "name": "root_feature",
            "type": "continuous",
            "distribution": "normal",
            "params": {"mean": 0.0, "std": 1.0},
        }
    ]
    target = {"name": "target", "type": "continuous", "problem_type": "regression"}
    dag = DAGSpec(
        nodes=["root_feature", "target"],
        target="target",
        root_nodes=["root_feature"],
        edges=[
            Edge(
                parent="root_feature",
                child="target",
                coefficient=1.0,
                mechanism={
                    "type": "spline",
                    "knots": [-2.5, -0.5, 0.5, 2.5],
                    "values": [-1.0, -0.2, 0.9, 0.1],
                },
            )
        ],
        noise_scales={"root_feature": 0.0, "target": 0.0},
        noise_models={
            "root_feature": {"type": "homoscedastic", "scale": 0.0},
            "target": {"type": "homoscedastic", "scale": 0.0},
        },
        autocorr={"root_feature": 0.0},
    )

    df = sample_dataset(rng, dag, features, target, n_samples=400)

    assert np.isfinite(df["target"]).all()
    assert df["target"].std() > 0.1
    assert df["target"].min() >= -1.0 - 1e-8
    assert df["target"].max() <= 0.9 + 1e-8


def test_spline_mechanism_handles_unsorted_knots():
    """Custom spline knots can be unsorted and should still interpolate safely."""
    rng = np.random.default_rng(31)
    features = [
        {
            "name": "root_feature",
            "type": "continuous",
            "distribution": "normal",
            "params": {"mean": 0.0, "std": 1.0},
        }
    ]
    target = {"name": "target", "type": "continuous", "problem_type": "regression"}
    dag = DAGSpec(
        nodes=["root_feature", "target"],
        target="target",
        root_nodes=["root_feature"],
        edges=[
            Edge(
                parent="root_feature",
                child="target",
                coefficient=1.0,
                mechanism={
                    "type": "spline",
                    "knots": [2.5, -2.5, 0.5, -0.5],
                    "values": [0.1, -1.0, 0.9, -0.2],
                },
            )
        ],
        noise_scales={"root_feature": 0.0, "target": 0.0},
        noise_models={
            "root_feature": {"type": "homoscedastic", "scale": 0.0},
            "target": {"type": "homoscedastic", "scale": 0.0},
        },
        autocorr={"root_feature": 0.0},
    )

    df = sample_dataset(rng, dag, features, target, n_samples=300)
    assert np.isfinite(df["target"]).all()


def test_heteroscedastic_noise_depends_on_driver_feature():
    """Heteroscedastic noise models should change residual variance with the driver."""
    rng = np.random.default_rng(4)
    features = [
        {
            "name": "root_feature",
            "type": "continuous",
            "distribution": "normal",
            "params": {"mean": 0.0, "std": 1.0},
        }
    ]
    target = {"name": "target", "type": "continuous", "problem_type": "regression"}
    dag = DAGSpec(
        nodes=["root_feature", "target"],
        target="target",
        root_nodes=["root_feature"],
        edges=[Edge(parent="root_feature", child="target", form="linear", coefficient=1.0)],
        noise_scales={"root_feature": 0.0, "target": 0.35},
        noise_models={
            "root_feature": {"type": "homoscedastic", "scale": 0.0},
            "target": {
                "type": "heteroscedastic",
                "driver": "root_feature",
                "base_scale": 0.35,
                "low_multiplier": 0.35,
                "high_multiplier": 2.0,
            },
        },
        autocorr={"root_feature": 0.0},
    )

    df = sample_dataset(rng, dag, features, target, n_samples=4000)
    root = df["root_feature"].to_numpy()
    root_norm = (root - root.mean()) / root.std()
    residual = df["target"].to_numpy() - root_norm

    lo_cut = np.quantile(root_norm, 0.2)
    hi_cut = np.quantile(root_norm, 0.8)
    low_var = float(np.var(residual[root_norm <= lo_cut]))
    high_var = float(np.var(residual[root_norm >= hi_cut]))

    assert high_var > low_var * 1.8
