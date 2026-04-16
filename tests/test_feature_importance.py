"""Tests for ground-truth feature importance extraction."""

import numpy as np
import pytest

from tabular_bank.evaluation.feature_importance import (
    FeatureImportanceProfile,
    compute_ground_truth_importance,
    evaluate_importance_fidelity,
)
from tabular_bank.generation.dag_builder import DAGSpec, Edge


def _simple_dag():
    """A -> B -> target, C is disconnected (noise)."""
    return DAGSpec(
        nodes=["A", "B", "C", "target"],
        target="target",
        root_nodes=["A", "C"],
        edges=[
            Edge(parent="A", child="B", coefficient=0.8, mechanism={"type": "linear"}),
            Edge(parent="B", child="target", coefficient=0.5, mechanism={"type": "linear"}),
        ],
        noise_scales={"A": 0.1, "B": 0.2, "C": 0.1, "target": 0.3},
    )


def _diamond_dag():
    """A -> B -> target, A -> C -> target (two paths from A)."""
    return DAGSpec(
        nodes=["A", "B", "C", "target"],
        target="target",
        root_nodes=["A"],
        edges=[
            Edge(parent="A", child="B", coefficient=0.6, mechanism={"type": "linear"}),
            Edge(parent="A", child="C", coefficient=0.4, mechanism={"type": "sigmoid"}),
            Edge(parent="B", child="target", coefficient=0.5, mechanism={"type": "linear"}),
            Edge(parent="C", child="target", coefficient=0.3, mechanism={"type": "threshold"}),
        ],
        noise_scales={"A": 0.1, "B": 0.2, "C": 0.2, "target": 0.3},
    )


class TestGroundTruthImportance:

    def test_simple_chain(self):
        dag = _simple_dag()
        profile = compute_ground_truth_importance(dag, ["A", "B", "C"])

        # A has a path to target (A->B->target), B has direct path, C has none
        assert profile.importance["A"] > 0
        assert profile.importance["B"] > 0
        assert profile.importance["C"] == 0.0

        # B is more important than A (direct vs indirect)
        assert profile.importance["B"] > profile.importance["A"]

    def test_noise_detection(self):
        dag = _simple_dag()
        profile = compute_ground_truth_importance(dag, ["A", "B", "C"])

        assert profile.is_noise["C"] is True
        assert profile.is_noise["A"] is False
        assert profile.is_noise["B"] is False
        assert profile.noise_features == ["C"]

    def test_diamond_multiple_paths(self):
        dag = _diamond_dag()
        profile = compute_ground_truth_importance(dag, ["A", "B", "C"])

        # A has two paths to target: A->B->target and A->C->target
        assert profile.n_paths_to_target["A"] == 2
        assert profile.n_paths_to_target["B"] == 1
        assert profile.n_paths_to_target["C"] == 1

        # All features have nonzero importance (all connected to target)
        assert profile.importance["A"] > 0
        assert profile.importance["B"] > 0
        assert profile.importance["C"] > 0

        # A's importance is sum of path products: |0.6|*|0.5| + |0.4|*|0.3| = 0.42
        # B's importance is direct: |0.5| = 0.5
        # C's importance is direct: |0.3| = 0.3
        # So B > A > C (direct strong connection beats indirect multi-path)
        assert profile.importance["B"] > profile.importance["C"]

    def test_mechanism_types_tracked(self):
        dag = _diamond_dag()
        profile = compute_ground_truth_importance(dag, ["A", "B", "C"])

        # A's paths use linear and sigmoid/threshold
        assert "linear" in profile.mechanism_types["A"]
        assert len(profile.mechanism_types["A"]) >= 2

    def test_ranked_features(self):
        dag = _simple_dag()
        profile = compute_ground_truth_importance(dag, ["A", "B", "C"])
        ranked = profile.ranked_features

        # B should be first (most important), C last (zero)
        assert ranked[0][0] == "B"
        assert ranked[-1][0] == "C"


class TestImportanceFidelity:

    def test_perfect_match(self):
        dag = _simple_dag()
        profile = compute_ground_truth_importance(dag, ["A", "B", "C"])

        # Estimated importance matches ground truth exactly
        estimated = dict(profile.importance)
        result = evaluate_importance_fidelity(profile, estimated, "perfect_model")

        assert result.kendall_tau == pytest.approx(1.0)
        assert result.top_k_overlap == pytest.approx(1.0)

    def test_inverted_importance(self):
        dag = _simple_dag()
        profile = compute_ground_truth_importance(dag, ["A", "B", "C"])

        # Inverted: noise feature gets highest score
        estimated = {f: -v for f, v in profile.importance.items()}
        estimated["C"] = 10.0  # noise feature ranked highest
        result = evaluate_importance_fidelity(profile, estimated, "bad_model")

        assert result.kendall_tau < 0

    def test_noise_detection_metrics(self):
        dag = _simple_dag()
        profile = compute_ground_truth_importance(dag, ["A", "B", "C"])

        # Good model: noise feature ranked last
        estimated = {"A": 0.5, "B": 0.8, "C": 0.0}
        result = evaluate_importance_fidelity(profile, estimated, "good")

        assert result.noise_detection_recall == pytest.approx(1.0)
