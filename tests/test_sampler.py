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
