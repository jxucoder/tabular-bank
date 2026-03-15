"""Tests for DAG construction."""

import numpy as np

from tabular_bank.generation.dag_builder import build_dag


def _features(n: int = 8) -> list[dict]:
    return [
        {
            "name": f"f_{i}",
            "type": "continuous",
            "distribution": "normal",
            "params": {"mean": 0.0, "std": 1.0},
        }
        for i in range(n)
    ]


def _target() -> dict:
    return {"name": "target", "type": "continuous", "problem_type": "regression"}


def _template() -> dict:
    return {
        "difficulty": {
            "noise_scale": 0.5,
            "nonlinear_prob": 0.3,
            "interaction_prob": 0.2,
            "edge_density": 0.5,
            "max_parents": 4,
            "heteroscedastic_prob": 0.2,
            "n_confounders": 2,
            "confounder_strength": 0.5,
            "temporal_prob": 0.2,
            "max_autocorr": 0.7,
            "root_correlation_strength": 0.3,
        }
    }


def test_build_dag_is_acyclic_by_topological_order():
    dag = build_dag(np.random.default_rng(0), _features(), _target(), _template())
    index = {node: i for i, node in enumerate(dag.nodes)}
    for edge in dag.edges:
        if edge.is_confounder:
            continue
        assert index[edge.parent] < index[edge.child]


def test_confounder_edges_are_flagged_and_linear():
    dag = build_dag(np.random.default_rng(1), _features(), _target(), _template())
    confounder_edges = [e for e in dag.edges if e.is_confounder]
    assert confounder_edges
    for edge in confounder_edges:
        assert edge.parent.startswith("_latent_")
        assert edge.mechanism["type"] == "linear"
